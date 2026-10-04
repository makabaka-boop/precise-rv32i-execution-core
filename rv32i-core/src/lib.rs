#![forbid(unsafe_code)]
//! # RV32I 教学执行核
//!
//! 从零实现的 RV32I 模拟器，不依赖任何现成模拟器或外部 crate。
//!
//! ## 覆盖范围
//! * RV32I 整数运算、分支、跳转，字节 / 半字 / 字访存（小端，**严格对齐**）。
//! * 特权指令 `CSRRW`、`CSRRS`、`MRET`；仅机器模式，陷阱一律走**直接入口**
//!   （`pc ← mtvec & !3`）。不实现中断、分页、`mstatus` 以及任何其它扩展。
//! * `ECALL` / `EBREAK` 是基础 ISA 的一部分，按规范产生异常（mcause = 11 / 3）；
//!   `FENCE` 视为空操作。其余未识别编码一律按非法指令处理。
//!
//! ## 内存映射
//! | 地址                        | 用途                                                |
//! |-----------------------------|-----------------------------------------------------|
//! | `0x0000_0000 .. 0x0001_0000`| 64 KiB RAM（可取指、可读写，小端）                   |
//! | `0x8000_0000`               | 接收队列读端口（只读；对它写 → 存储访问故障）         |
//! | `0x8000_0004`               | 发送写端口（只写；对它读 → 加载访问故障）             |
//! | 其它地址                    | 未映射 → 访问故障                                     |
//!
//! ## 端口语义
//! * 读 `RX_ADDR`：从接收队列弹出**一个字节**（任何访存宽度都只弹一个字节），
//!   零扩展到访存宽度后再按加载指令的规则符号扩展；队列为空时按弹出 `0xFF` 处理。
//!   即使 `rd = x0`（结果被丢弃）也照常弹出——读端口是有副作用的操作。
//! * 写 `TX_ADDR`：把存储值的低 N 字节（N = 访存宽度，小端序）追加到发送日志。
//!
//! ## 陷阱约定
//! * 故障指令**不提交**任何寄存器、RAM 或端口副作用；随后硬件写入
//!   `mepc / mcause / mtval` 并跳转到 `mtvec & !3`。
//! * `mtval`：取指未对齐 / 取指故障 → 出错地址；非法指令 → 指令原始位；
//!   访存未对齐 / 访存故障 → 出错的有效地址；`ECALL/EBREAK` → 0。
//! * 跳转类指令（JAL/JALR/分支）目标是未对齐地址时，异常算在该跳转指令头上
//!   （`mepc` = 跳转指令的 PC，链接寄存器不写入）。JALR 先清除目标最低位，
//!   **再**做 4 字节对齐检查——两者不可互相替代。`MRET` 对 `mepc` 做同样的检查。
//! * 同一条访存指令既未对齐又越界时，先报未对齐（先算地址对齐、再查映射）。
//!
//! ## 步数预算
//! `run` 的预算上限为 [`MAX_STEPS`]（2000 步），预算耗尽即停，
//! 当前机器状态可通过 [`Machine::snapshot`] 取回。

use std::collections::VecDeque;
use std::fmt;

/// RAM 容量：64 KiB。
pub const RAM_SIZE: usize = 64 * 1024;
/// 接收队列读端口地址（只读）。
pub const RX_ADDR: u32 = 0x8000_0000;
/// 发送写端口地址（只写）。
pub const TX_ADDR: u32 = 0x8000_0004;
/// 单次 `run` 允许的最大步数。
pub const MAX_STEPS: u64 = 2000;

pub const CSR_MTVEC: u32 = 0x305;
pub const CSR_MEPC: u32 = 0x341;
pub const CSR_MCAUSE: u32 = 0x342;
pub const CSR_MTVAL: u32 = 0x343;

pub const CAUSE_FETCH_MISALIGNED: u32 = 0;
pub const CAUSE_FETCH_FAULT: u32 = 1;
pub const CAUSE_ILLEGAL: u32 = 2;
pub const CAUSE_EBREAK: u32 = 3;
pub const CAUSE_LOAD_MISALIGNED: u32 = 4;
pub const CAUSE_LOAD_FAULT: u32 = 5;
pub const CAUSE_STORE_MISALIGNED: u32 = 6;
pub const CAUSE_STORE_FAULT: u32 = 7;
pub const CAUSE_ECALL: u32 = 11;

/// mcause 数值对应的可读名称。
pub fn cause_name(cause: u32) -> &'static str {
    match cause {
        CAUSE_FETCH_MISALIGNED => "instruction address misaligned",
        CAUSE_FETCH_FAULT => "instruction access fault",
        CAUSE_ILLEGAL => "illegal instruction",
        CAUSE_EBREAK => "breakpoint",
        CAUSE_LOAD_MISALIGNED => "load address misaligned",
        CAUSE_LOAD_FAULT => "load access fault",
        CAUSE_STORE_MISALIGNED => "store address misaligned",
        CAUSE_STORE_FAULT => "store access fault",
        CAUSE_ECALL => "environment call from M-mode",
        _ => "unknown",
    }
}

/// 一次同步异常（陷阱）。进入陷阱时由硬件写入对应 CSR。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trap {
    pub cause: u32,
    pub mepc: u32,
    pub mtval: u32,
}

impl Trap {
    fn new(cause: u32, mepc: u32, mtval: u32) -> Self {
        Trap { cause, mepc, mtval }
    }
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mcause={} ({}) mepc={:08x} mtval={:08x}",
            self.cause,
            cause_name(self.cause),
            self.mepc,
            self.mtval
        )
    }
}

/// 一条指令提交后的架构可见变化（仅记录成功退休的指令）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Commit {
    /// 写通用寄存器（rd != x0）。
    Reg { rd: u8, val: u32 },
    /// 写 RAM：`addr` 起 `size` 字节，值为 `val` 的低 `size` 字节（小端）。
    Ram { addr: u32, size: u8, val: u32 },
    /// 写发送端口：`val` 的低 `size` 字节已进入发送日志。
    Tx { size: u8, val: u32 },
    /// 读接收端口弹出的字节；`None` 表示队列已空（按 0xFF 处理）。
    RxPop { byte: Option<u8> },
    /// 写 CSR。
    Csr { addr: u32, val: u32 },
}

/// 寄存器 ABI 名，用于轨迹显示。
pub const REG_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

/// CSR 地址对应的名称。
pub fn csr_name(addr: u32) -> &'static str {
    match addr {
        CSR_MTVEC => "mtvec",
        CSR_MEPC => "mepc",
        CSR_MCAUSE => "mcause",
        CSR_MTVAL => "mtval",
        _ => "csr?",
    }
}

impl fmt::Display for Commit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Commit::Reg { rd, val } => write!(f, "{} <= {:08x}", REG_NAMES[rd as usize], val),
            Commit::Ram { addr, size, val } => {
                let end = addr + size as u32;
                match size {
                    1 => write!(f, "ram[{:08x}..{:08x}] <= {:02x}", addr, end, val & 0xff),
                    2 => write!(f, "ram[{:08x}..{:08x}] <= {:04x}", addr, end, val & 0xffff),
                    _ => write!(f, "ram[{:08x}..{:08x}] <= {:08x}", addr, end, val),
                }
            }
            Commit::Tx { size, val } => match size {
                1 => write!(f, "tx <= {:02x} (1B)", val & 0xff),
                2 => write!(f, "tx <= {:04x} (2B)", val & 0xffff),
                _ => write!(f, "tx <= {:08x} (4B)", val),
            },
            Commit::RxPop { byte: Some(b) } => write!(f, "rx => {:02x}", b),
            Commit::RxPop { byte: None } => write!(f, "rx => <empty:ff>"),
            Commit::Csr { addr, val } => write!(f, "{} <= {:08x}", csr_name(addr), val),
        }
    }
}

// ---------------------------------------------------------------------------
// 译码
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AluOp {
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
}

fn alu_name(op: AluOp) -> &'static str {
    match op {
        AluOp::Add => "add",
        AluOp::Sub => "sub",
        AluOp::Sll => "sll",
        AluOp::Slt => "slt",
        AluOp::Sltu => "sltu",
        AluOp::Xor => "xor",
        AluOp::Srl => "srl",
        AluOp::Sra => "sra",
        AluOp::Or => "or",
        AluOp::And => "and",
    }
}

#[derive(Clone, Copy, Debug)]
enum Instr {
    Lui { rd: u8, imm: u32 },
    Auipc { rd: u8, imm: u32 },
    Jal { rd: u8, imm: i32 },
    Jalr { rd: u8, rs1: u8, imm: i32 },
    Br { f3: u32, rs1: u8, rs2: u8, imm: i32 },
    Ld { f3: u32, rd: u8, rs1: u8, imm: i32 },
    St { f3: u32, rs1: u8, rs2: u8, imm: i32 },
    AluI { kind: AluOp, rd: u8, rs1: u8, imm: i32 },
    ShI { kind: AluOp, rd: u8, rs1: u8, shamt: u8 },
    AluR { kind: AluOp, rd: u8, rs1: u8, rs2: u8 },
    Fence,
    CsrRw { rd: u8, rs1: u8, csr: u32 },
    CsrRs { rd: u8, rs1: u8, csr: u32 },
    Mret,
    Ecall,
    Ebreak,
}

fn is_known_csr(n: u32) -> bool {
    matches!(n, CSR_MTVEC | CSR_MEPC | CSR_MCAUSE | CSR_MTVAL)
}

/// 纯函数译码：无法识别（含未实现的 CSR、扩展指令）时返回 `None`。
fn decode(w: u32) -> Option<Instr> {
    let opcode = w & 0x7f;
    let rd = ((w >> 7) & 0x1f) as u8;
    let f3 = (w >> 12) & 0x7;
    let rs1 = ((w >> 15) & 0x1f) as u8;
    let rs2 = ((w >> 20) & 0x1f) as u8;
    let f7 = (w >> 25) & 0x7f;
    // 各类立即数（符号已扩展）。
    let imm_i = (w as i32) >> 20;
    let imm_s = (((w as i32) >> 25) << 5) | (((w >> 7) & 0x1f) as i32);
    let imm_b = (((w as i32) >> 31) << 12)
        | ((((w >> 7) & 0x1) as i32) << 11)
        | ((((w >> 25) & 0x3f) as i32) << 5)
        | ((((w >> 8) & 0xf) as i32) << 1);
    let imm_u = w & 0xffff_f000;
    let imm_j = (((w as i32) >> 31) << 20)
        | ((((w >> 12) & 0xff) as i32) << 12)
        | ((((w >> 20) & 0x1) as i32) << 11)
        | ((((w >> 21) & 0x3ff) as i32) << 1);

    Some(match opcode {
        0x37 => Instr::Lui { rd, imm: imm_u },
        0x17 => Instr::Auipc { rd, imm: imm_u },
        0x6f => Instr::Jal { rd, imm: imm_j },
        0x67 => {
            if f3 == 0 {
                Instr::Jalr { rd, rs1, imm: imm_i }
            } else {
                return None;
            }
        }
        0x63 => match f3 {
            0 | 1 | 4 | 5 | 6 | 7 => Instr::Br { f3, rs1, rs2, imm: imm_b },
            _ => return None,
        },
        0x03 => match f3 {
            0 | 1 | 2 | 4 | 5 => Instr::Ld { f3, rd, rs1, imm: imm_i },
            _ => return None,
        },
        0x23 => match f3 {
            0 | 1 | 2 => Instr::St { f3, rs1, rs2, imm: imm_s },
            _ => return None,
        },
        0x13 => match f3 {
            0 => Instr::AluI { kind: AluOp::Add, rd, rs1, imm: imm_i },
            2 => Instr::AluI { kind: AluOp::Slt, rd, rs1, imm: imm_i },
            3 => Instr::AluI { kind: AluOp::Sltu, rd, rs1, imm: imm_i },
            4 => Instr::AluI { kind: AluOp::Xor, rd, rs1, imm: imm_i },
            6 => Instr::AluI { kind: AluOp::Or, rd, rs1, imm: imm_i },
            7 => Instr::AluI { kind: AluOp::And, rd, rs1, imm: imm_i },
            1 => {
                if f7 != 0x00 {
                    return None;
                }
                Instr::ShI { kind: AluOp::Sll, rd, rs1, shamt: rs2 }
            }
            5 => match f7 {
                0x00 => Instr::ShI { kind: AluOp::Srl, rd, rs1, shamt: rs2 },
                0x20 => Instr::ShI { kind: AluOp::Sra, rd, rs1, shamt: rs2 },
                _ => return None,
            },
            _ => return None,
        },
        0x33 => {
            let kind = match (f7, f3) {
                (0x00, 0) => AluOp::Add,
                (0x20, 0) => AluOp::Sub,
                (0x00, 1) => AluOp::Sll,
                (0x00, 2) => AluOp::Slt,
                (0x00, 3) => AluOp::Sltu,
                (0x00, 4) => AluOp::Xor,
                (0x00, 5) => AluOp::Srl,
                (0x20, 5) => AluOp::Sra,
                (0x00, 6) => AluOp::Or,
                (0x00, 7) => AluOp::And,
                _ => return None,
            };
            Instr::AluR { kind, rd, rs1, rs2 }
        }
        0x0f => {
            if f3 == 0 {
                Instr::Fence
            } else {
                return None;
            }
        }
        0x73 => match f3 {
            0 => match w {
                0x0000_0073 => Instr::Ecall,
                0x0010_0073 => Instr::Ebreak,
                0x3020_0073 => Instr::Mret,
                _ => return None,
            },
            1 | 2 => {
                let csr = (w >> 20) & 0xfff;
                if !is_known_csr(csr) {
                    return None;
                }
                if f3 == 1 {
                    Instr::CsrRw { rd, rs1, csr }
                } else {
                    Instr::CsrRs { rd, rs1, csr }
                }
            }
            _ => return None,
        },
        _ => return None,
    })
}

fn br_name(f3: u32) -> &'static str {
    match f3 {
        0 => "beq",
        1 => "bne",
        4 => "blt",
        5 => "bge",
        6 => "bltu",
        _ => "bgeu",
    }
}

fn ld_name(f3: u32) -> &'static str {
    match f3 {
        0 => "lb",
        1 => "lh",
        2 => "lw",
        4 => "lbu",
        _ => "lhu",
    }
}

fn st_name(f3: u32) -> &'static str {
    match f3 {
        0 => "sb",
        1 => "sh",
        _ => "sw",
    }
}

/// 生成用于轨迹显示的汇编助记串。
fn disasm(ins: &Instr) -> String {
    use Instr::*;
    let r = |i: u8| REG_NAMES[i as usize];
    match *ins {
        Lui { rd, imm } => format!("lui {}, 0x{:x}", r(rd), imm >> 12),
        Auipc { rd, imm } => format!("auipc {}, 0x{:x}", r(rd), imm >> 12),
        Jal { rd, imm } => format!("jal {}, {:+}", r(rd), imm),
        Jalr { rd, rs1, imm } => format!("jalr {}, {}({})", r(rd), imm, r(rs1)),
        Br { f3, rs1, rs2, imm } => format!("{} {}, {}, {:+}", br_name(f3), r(rs1), r(rs2), imm),
        Ld { f3, rd, rs1, imm } => format!("{} {}, {}({})", ld_name(f3), r(rd), imm, r(rs1)),
        St { f3, rs1, rs2, imm } => format!("{} {}, {}({})", st_name(f3), r(rs2), imm, r(rs1)),
        AluI { kind, rd, rs1, imm } => {
            format!("{}i {}, {}, {}", alu_name(kind), r(rd), r(rs1), imm)
        }
        ShI { kind, rd, rs1, shamt } => {
            format!("{}i {}, {}, {}", alu_name(kind), r(rd), r(rs1), shamt)
        }
        AluR { kind, rd, rs1, rs2 } => {
            format!("{} {}, {}, {}", alu_name(kind), r(rd), r(rs1), r(rs2))
        }
        Fence => "fence".to_string(),
        CsrRw { rd, rs1, csr } => format!("csrrw {}, {}, {}", r(rd), csr_name(csr), r(rs1)),
        CsrRs { rd, rs1, csr } => format!("csrrs {}, {}, {}", r(rd), csr_name(csr), r(rs1)),
        Mret => "mret".to_string(),
        Ecall => "ecall".to_string(),
        Ebreak => "ebreak".to_string(),
    }
}

// ---------------------------------------------------------------------------
// 轨迹
// ---------------------------------------------------------------------------

/// 单步执行记录：执行前后 PC、译码结果、已提交变化、陷阱。
#[derive(Clone, Debug)]
pub struct StepRecord {
    /// 步序号（从 1 开始）。
    pub step: u64,
    /// 执行前 PC。
    pub pc_before: u32,
    /// 取到的指令原始位；取指失败时为 `None`。
    pub instr: Option<u32>,
    /// 汇编助记串（取指失败 / 译码失败时有标注）。
    pub disasm: String,
    /// 本步提交的变化（故障指令为空）。
    pub commits: Vec<Commit>,
    /// 本步触发的陷阱。
    pub trap: Option<Trap>,
    /// 执行后 PC（若陷入则为 mtvec 入口）。
    pub pc_after: u32,
}

impl fmt::Display for StepRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let instr = match self.instr {
            Some(w) => format!("{:08x}", w),
            None => "--------".to_string(),
        };
        writeln!(
            f,
            "step {:04}  pc={:08x}  instr={}  {}",
            self.step, self.pc_before, instr, self.disasm
        )?;
        if !self.commits.is_empty() {
            let list: Vec<String> = self.commits.iter().map(|c| c.to_string()).collect();
            writeln!(f, "      commits: {}", list.join("; "))?;
        }
        if let Some(t) = &self.trap {
            writeln!(f, "      trap: {}", t)?;
        }
        write!(f, "      next pc={:08x}", self.pc_after)
    }
}

/// 机器状态快照（预算耗尽或任意时刻可取）。
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub regs: [u32; 32],
    pub pc: u32,
    pub mtvec: u32,
    pub mepc: u32,
    pub mcause: u32,
    pub mtval: u32,
    pub steps: u64,
    pub rx_pending: usize,
    pub tx_len: usize,
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "pc={:08x}  steps={}", self.pc, self.steps)?;
        writeln!(
            f,
            "mtvec={:08x} mepc={:08x} mcause={:08x} mtval={:08x}",
            self.mtvec, self.mepc, self.mcause, self.mtval
        )?;
        for row in 0..8 {
            let mut line = String::new();
            for col in 0..4 {
                let i = row * 4 + col;
                line.push_str(&format!("x{:<2}={:08x}  ", i, self.regs[i]));
            }
            writeln!(f, "{}", line.trim_end())?;
        }
        write!(f, "rx_pending={} tx_len={}", self.rx_pending, self.tx_len)
    }
}

/// `run` 的返回：完整轨迹 + 结束时的机器状态。
#[derive(Clone, Debug)]
pub struct RunReport {
    pub trace: Vec<StepRecord>,
    pub state: Snapshot,
    /// 实际使用的步数预算（已夹到 [`MAX_STEPS`] 以内）。
    pub budget: u64,
}

// ---------------------------------------------------------------------------
// 执行核
// ---------------------------------------------------------------------------

/// RV32I 机器：32 个通用寄存器、PC、4 个机器模式 CSR、64 KiB RAM 与两个端口。
pub struct Machine {
    pub regs: [u32; 32],
    pub pc: u32,
    pub mtvec: u32,
    pub mepc: u32,
    pub mcause: u32,
    pub mtval: u32,
    pub ram: Box<[u8; RAM_SIZE]>,
    /// 接收队列（`RX_ADDR` 的读取来源）。
    pub rx: VecDeque<u8>,
    /// 发送日志（`TX_ADDR` 的写入去向，按字节累积）。
    pub tx: Vec<u8>,
    /// 已执行的指令条数。
    pub steps: u64,
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

impl Machine {
    pub fn new() -> Self {
        Machine {
            regs: [0; 32],
            pc: 0,
            mtvec: 0,
            mepc: 0,
            mcause: 0,
            mtval: 0,
            ram: Box::new([0; RAM_SIZE]),
            rx: VecDeque::new(),
            tx: Vec::new(),
            steps: 0,
        }
    }

    /// 把原始机器码 / 数据装入 RAM。
    pub fn load_bytes(&mut self, addr: u32, data: &[u8]) {
        let start = addr as usize;
        assert!(
            start + data.len() <= RAM_SIZE,
            "装载范围 {:08x}..{:08x} 超出 RAM",
            addr,
            addr as usize + data.len()
        );
        self.ram[start..start + data.len()].copy_from_slice(data);
    }

    /// 向接收队列压入一个字节。
    pub fn push_rx(&mut self, byte: u8) {
        self.rx.push_back(byte);
    }

    /// 取当前机器状态快照。
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            regs: self.regs,
            pc: self.pc,
            mtvec: self.mtvec,
            mepc: self.mepc,
            mcause: self.mcause,
            mtval: self.mtval,
            steps: self.steps,
            rx_pending: self.rx.len(),
            tx_len: self.tx.len(),
        }
    }

    /// 以不超过 [`MAX_STEPS`] 的预算连续执行，返回轨迹与最终状态。
    pub fn run(&mut self, budget: u64) -> RunReport {
        let budget = budget.min(MAX_STEPS);
        let mut trace = Vec::with_capacity(budget as usize);
        for _ in 0..budget {
            trace.push(self.step());
        }
        RunReport { trace, state: self.snapshot(), budget }
    }

    /// 执行一条指令，返回该步记录。
    pub fn step(&mut self) -> StepRecord {
        let pc_before = self.pc;
        let mut commits: Vec<Commit> = Vec::new();
        let mut trap: Option<Trap> = None;
        let mut instr: Option<u32> = None;
        let disasm: String;

        match self.fetch(pc_before) {
            Err(t) => {
                trap = Some(t);
                disasm = "<fetch fault>".to_string();
            }
            Ok(w) => {
                instr = Some(w);
                match decode(w) {
                    None => {
                        trap = Some(Trap::new(CAUSE_ILLEGAL, pc_before, w));
                        disasm = "???".to_string();
                    }
                    Some(ins) => {
                        disasm = disasm_of(&ins);
                        if let Err(t) = self.exec(ins, pc_before, &mut commits) {
                            trap = Some(t);
                        }
                    }
                }
            }
        }

        if let Some(t) = trap {
            // 故障指令不提交任何寄存器 / RAM / 端口副作用。
            // （exec 内所有检查都排在副作用之前，这里清空只是双保险。）
            commits.clear();
            self.mepc = t.mepc;
            self.mcause = t.cause;
            self.mtval = t.mtval;
            self.pc = self.mtvec & !3; // 仅支持直接入口
        }

        self.steps += 1;
        StepRecord {
            step: self.steps,
            pc_before,
            instr,
            disasm,
            commits,
            trap,
            pc_after: self.pc,
        }
    }

    fn fetch(&self, pc: u32) -> Result<u32, Trap> {
        if pc % 4 != 0 {
            return Err(Trap::new(CAUSE_FETCH_MISALIGNED, pc, pc));
        }
        let a = pc as usize;
        if a + 4 > RAM_SIZE {
            return Err(Trap::new(CAUSE_FETCH_FAULT, pc, pc));
        }
        Ok(u32::from_le_bytes(self.ram[a..a + 4].try_into().unwrap()))
    }

    fn set_reg(&mut self, rd: u8, val: u32, commits: &mut Vec<Commit>) {
        if rd != 0 {
            self.regs[rd as usize] = val;
            commits.push(Commit::Reg { rd, val });
        }
    }

    /// 跳转目标必须是 4 字节对齐，否则异常算在发起跳转的指令头上。
    fn check_target(target: u32, pc: u32) -> Result<u32, Trap> {
        if target % 4 != 0 {
            Err(Trap::new(CAUSE_FETCH_MISALIGNED, pc, target))
        } else {
            Ok(target)
        }
    }

    fn csr_read(&self, n: u32) -> u32 {
        match n {
            CSR_MTVEC => self.mtvec,
            CSR_MEPC => self.mepc,
            CSR_MCAUSE => self.mcause,
            CSR_MTVAL => self.mtval,
            _ => unreachable!("decode 已过滤未知 CSR"),
        }
    }

    fn csr_write(&mut self, n: u32, val: u32, commits: &mut Vec<Commit>) {
        match n {
            CSR_MTVEC => self.mtvec = val,
            CSR_MEPC => self.mepc = val,
            CSR_MCAUSE => self.mcause = val,
            CSR_MTVAL => self.mtval = val,
            _ => unreachable!("decode 已过滤未知 CSR"),
        }
        commits.push(Commit::Csr { addr: n, val });
    }

    /// 读 `size` 字节（调用前已完成对齐检查）。返回零扩展到 32 位的原始值；
    /// `None` 表示地址未映射（访问故障）。
    fn mem_read(&mut self, addr: u32, size: u32, commits: &mut Vec<Commit>) -> Option<u32> {
        let a = addr as usize;
        let n = size as usize;
        if a + n <= RAM_SIZE {
            let v = match size {
                1 => self.ram[a] as u32,
                2 => u16::from_le_bytes([self.ram[a], self.ram[a + 1]]) as u32,
                _ => u32::from_le_bytes([
                    self.ram[a],
                    self.ram[a + 1],
                    self.ram[a + 2],
                    self.ram[a + 3],
                ]),
            };
            Some(v)
        } else if addr == RX_ADDR {
            // 端口读取是副作用：即使 rd=x0 也必须弹出。
            let byte = self.rx.pop_front();
            commits.push(Commit::RxPop { byte });
            Some(byte.unwrap_or(0xff) as u32)
        } else {
            // 含 TX_ADDR（只写）在内的未映射 / 方向错误地址。
            None
        }
    }

    /// 写 `size` 字节（调用前已完成对齐检查）。返回是否成功。
    fn mem_write(&mut self, addr: u32, size: u32, val: u32, commits: &mut Vec<Commit>) -> bool {
        let a = addr as usize;
        let n = size as usize;
        if a + n <= RAM_SIZE {
            let bytes = val.to_le_bytes();
            self.ram[a..a + n].copy_from_slice(&bytes[..n]);
            commits.push(Commit::Ram { addr, size: size as u8, val });
            true
        } else if addr == TX_ADDR {
            let bytes = val.to_le_bytes();
            self.tx.extend_from_slice(&bytes[..n]);
            commits.push(Commit::Tx { size: size as u8, val });
            true
        } else {
            // 含 RX_ADDR（只读）在内的未映射 / 方向错误地址。
            false
        }
    }

    fn exec(&mut self, ins: Instr, pc: u32, commits: &mut Vec<Commit>) -> Result<(), Trap> {
        match ins {
            Instr::Lui { rd, imm } => {
                let v = imm;
                self.set_reg(rd, v, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::Auipc { rd, imm } => {
                let v = pc.wrapping_add(imm);
                self.set_reg(rd, v, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::Jal { rd, imm } => {
                let target = pc.wrapping_add(imm as u32);
                let target = Self::check_target(target, pc)?; // 失败则不写链接寄存器
                let link = pc.wrapping_add(4);
                self.set_reg(rd, link, commits);
                self.pc = target;
            }
            Instr::Jalr { rd, rs1, imm } => {
                let base = self.regs[rs1 as usize];
                // 先清除目标最低位，再做 4 字节对齐检查；二者不可互相替代。
                let target = base.wrapping_add(imm as u32) & !1;
                let target = Self::check_target(target, pc)?;
                let link = pc.wrapping_add(4);
                self.set_reg(rd, link, commits);
                self.pc = target;
            }
            Instr::Br { f3, rs1, rs2, imm } => {
                let a = self.regs[rs1 as usize];
                let b = self.regs[rs2 as usize];
                let taken = match f3 {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i32) < (b as i32),
                    5 => (a as i32) >= (b as i32),
                    6 => a < b,
                    _ => a >= b,
                };
                if taken {
                    let target = pc.wrapping_add(imm as u32);
                    self.pc = Self::check_target(target, pc)?;
                } else {
                    self.pc = pc.wrapping_add(4);
                }
            }
            Instr::Ld { f3, rd, rs1, imm } => {
                let (size, signed) = match f3 {
                    0 => (1u32, true),
                    1 => (2, true),
                    2 => (4, true),
                    4 => (1, false),
                    _ => (2, false),
                };
                let addr = self.regs[rs1 as usize].wrapping_add(imm as u32);
                if addr % size != 0 {
                    return Err(Trap::new(CAUSE_LOAD_MISALIGNED, pc, addr));
                }
                let raw = match self.mem_read(addr, size, commits) {
                    Some(v) => v,
                    None => return Err(Trap::new(CAUSE_LOAD_FAULT, pc, addr)),
                };
                let v = if signed { sext(raw, size) } else { raw };
                self.set_reg(rd, v, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::St { f3, rs1, rs2, imm } => {
                let size = match f3 {
                    0 => 1u32,
                    1 => 2,
                    _ => 4,
                };
                let addr = self.regs[rs1 as usize].wrapping_add(imm as u32);
                if addr % size != 0 {
                    return Err(Trap::new(CAUSE_STORE_MISALIGNED, pc, addr));
                }
                let val = self.regs[rs2 as usize];
                if !self.mem_write(addr, size, val, commits) {
                    return Err(Trap::new(CAUSE_STORE_FAULT, pc, addr));
                }
                self.pc = pc.wrapping_add(4);
            }
            Instr::AluI { kind, rd, rs1, imm } => {
                let a = self.regs[rs1 as usize];
                let v = match kind {
                    AluOp::Add => a.wrapping_add(imm as u32),
                    AluOp::Slt => ((a as i32) < imm) as u32,
                    AluOp::Sltu => (a < imm as u32) as u32,
                    AluOp::Xor => a ^ (imm as u32),
                    AluOp::Or => a | (imm as u32),
                    AluOp::And => a & (imm as u32),
                    _ => unreachable!("decode 已过滤"),
                };
                self.set_reg(rd, v, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::ShI { kind, rd, rs1, shamt } => {
                let a = self.regs[rs1 as usize];
                let sh = shamt as u32;
                let v = match kind {
                    AluOp::Sll => a << sh,
                    AluOp::Srl => a >> sh,
                    AluOp::Sra => ((a as i32) >> sh) as u32,
                    _ => unreachable!("decode 已过滤"),
                };
                self.set_reg(rd, v, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::AluR { kind, rd, rs1, rs2 } => {
                let a = self.regs[rs1 as usize];
                let b = self.regs[rs2 as usize];
                let v = match kind {
                    AluOp::Add => a.wrapping_add(b),
                    AluOp::Sub => a.wrapping_sub(b),
                    AluOp::Sll => a << (b & 31),
                    AluOp::Slt => ((a as i32) < (b as i32)) as u32,
                    AluOp::Sltu => (a < b) as u32,
                    AluOp::Xor => a ^ b,
                    AluOp::Srl => a >> (b & 31),
                    AluOp::Sra => ((a as i32) >> (b & 31)) as u32,
                    AluOp::Or => a | b,
                    AluOp::And => a & b,
                };
                self.set_reg(rd, v, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::Fence => {
                self.pc = pc.wrapping_add(4); // 单核教学机，视为空操作
            }
            Instr::CsrRw { rd, rs1, csr } => {
                let old = self.csr_read(csr);
                let new = self.regs[rs1 as usize];
                self.csr_write(csr, new, commits);
                self.set_reg(rd, old, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::CsrRs { rd, rs1, csr } => {
                let old = self.csr_read(csr);
                if rs1 != 0 {
                    // rs1=x0 时只读不写
                    let new = old | self.regs[rs1 as usize];
                    self.csr_write(csr, new, commits);
                }
                self.set_reg(rd, old, commits);
                self.pc = pc.wrapping_add(4);
            }
            Instr::Mret => {
                // mepc 未对齐时异常算在 MRET 自己头上。
                self.pc = Self::check_target(self.mepc, pc)?;
            }
            Instr::Ecall => return Err(Trap::new(CAUSE_ECALL, pc, 0)),
            Instr::Ebreak => return Err(Trap::new(CAUSE_EBREAK, pc, 0)),
        }
        Ok(())
    }
}

/// 供 `step` 使用的反汇编入口（隔离私有 `disasm` 名称冲突）。
fn disasm_of(ins: &Instr) -> String {
    disasm(ins)
}

/// 按 `size` 字节对 `v` 做符号扩展。
fn sext(v: u32, size: u32) -> u32 {
    match size {
        1 => v as u8 as i8 as i32 as u32,
        2 => v as u16 as i16 as i32 as u32,
        _ => v,
    }
}
