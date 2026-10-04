//! 手工可复核机器码测试。
//!
//! 所有程序都以字面十六进制常量给出，不调用任何汇编器 / 编译器，
//! 每条指令旁标注汇编形式与字段分解，可对照《RISC-V 指令集手册》手工核验。
//!
//! 编码格式速查：
//!   R 型: funct7 | rs2 | rs1 | f3 | rd | opcode
//!   I 型: imm[11:0] | rs1 | f3 | rd | opcode
//!   S 型: imm[11:5] | rs2 | rs1 | f3 | imm[4:0] | opcode
//!   B 型: imm[12|10:5] | rs2 | rs1 | f3 | imm[4:1|11] | opcode
//!   U 型: imm[31:12] | rd | opcode
//!   J 型: imm[20|10:1|11|19:12] | rd | opcode

use rv32i_core::*;

/// 陷阱处理程序 1（装载在 0x40）：记录 mcause/mtval/mepc 到 x28/x29/x30，
/// mepc += 4 跳过故障指令，然后 mret 返回。
const H1: [u32; 6] = [
    0x34202E73, // csrrs x28, mcause, x0   imm=0x342 rs1=0  f3=2 rd=28 op=0x73
    0x34302EF3, // csrrs x29, mtval,  x0   imm=0x343 rs1=0  f3=2 rd=29 op=0x73
    0x34102F73, // csrrs x30, mepc,   x0   imm=0x341 rs1=0  f3=2 rd=30 op=0x73
    0x004F0F13, // addi  x30, x30, 4       imm=4     rs1=30 f3=0 rd=30 op=0x13
    0x341F1073, // csrrw x0,  mepc, x30    imm=0x341 rs1=30 f3=1 rd=0  op=0x73
    0x30200073, // mret                     规范定义的固定编码
];

/// 陷阱处理程序 2（装载在 0x40）：记录 mcause/mtval/mepc 后原地打转（停机观察用）。
const H2: [u32; 4] = [
    0x34202E73, // csrrs x28, mcause, x0
    0x34302EF3, // csrrs x29, mtval,  x0
    0x34102F73, // csrrs x30, mepc,   x0
    0x0000006F, // jal x0, 0                imm=0 -> 原地循环
];

fn load_words(m: &mut Machine, addr: u32, words: &[u32]) {
    for (i, w) in words.iter().enumerate() {
        let a = addr as usize + 4 * i;
        m.ram[a..a + 4].copy_from_slice(&w.to_le_bytes());
    }
}

/// 程序从 0x00 开始装载的机器。
fn machine(prog: &[u32]) -> Machine {
    let mut m = Machine::new();
    load_words(&mut m, 0, prog);
    m
}

/// 程序在 0x00、处理程序在 0x40 的机器。
fn with_handler(prog: &[u32], handler: &[u32]) -> Machine {
    let mut m = machine(prog);
    load_words(&mut m, 0x40, handler);
    m
}

// ---------------------------------------------------------------------------
// 有符号比较与移位
// ---------------------------------------------------------------------------
#[test]
fn signed_compare_and_shifts() {
    let prog = [
        0xFFF00293, // addi x5,  x0, -1      imm=0xFFF            -> x5  = 0xFFFFFFFF (-1)
        0x00100313, // addi x6,  x0, 1       imm=0x001            -> x6  = 1
        0x0062A3B3, // slt  x7,  x5, x6      f7=00 rs2=6 rs1=5 f3=2 -> -1 < 1     => 1
        0x0062B433, // sltu x8,  x5, x6      f7=00 rs2=6 rs1=5 f3=3 -> 0xFFFF.. < 1 => 0
        0x4012D493, // srai x9,  x5, 1       imm=0x401 (f7=0x20, sh=1) f3=5 -> 0xFFFFFFFF
        0x0012D513, // srli x10, x5, 1       imm=0x001 (f7=0x00, sh=1) f3=5 -> 0x7FFFFFFF
        0x00129593, // slli x11, x5, 1       imm=0x001            f3=1 -> 0xFFFFFFFE
        0x02100693, // addi x13, x0, 33      移位量 33（硬件只取低 5 位 = 1）
        0x40D2D633, // sra  x12, x5, x13     f7=0x20 rs2=13 rs1=5 f3=5 -> >>1 算术 = 0xFFFFFFFF
        0x00D2D733, // srl  x14, x5, x13     f7=0x00 rs2=13 rs1=5 f3=5 -> >>1 逻辑 = 0x7FFFFFFF
        0x00D317B3, // sll  x15, x6, x13     f7=0x00 rs2=13 rs1=6 f3=1 -> 1<<1 = 2
        0x0062C463, // blt  x5, x6, +8       f3=4 imm=8  -> -1<1 成立，跳到 0x34
        0x00100813, // addi x16, x0, 1       （被跳过）
        0x00200813, // addi x16, x0, 2       0x34: -> x16 = 2
        0x0062D463, // bge  x5, x6, +8       f3=5 -> -1>=1 不成立，顺序执行
        0x00300893, // addi x17, x0, 3       -> x17 = 3
        0x0062F463, // bgeu x5, x6, +8       f3=7 -> 0xFFFF..>=1 成立，跳到 0x48
        0x00400913, // addi x18, x0, 4       （被跳过）
        0x00500913, // addi x18, x0, 5       0x48: -> x18 = 5
        0x0062E463, // bltu x5, x6, +8       f3=6 -> 0xFFFF..<1 不成立，顺序执行
        0x00600993, // addi x19, x0, 6       -> x19 = 6
        0x0012AA93, // slti  x21, x5, 1      f3=2 -> -1<1  => 1
        0x0012BB13, // sltiu x22, x5, 1      f3=3 -> 0xFFFF..<1 => 0
        0x0000006F, // jal x0, 0             原地循环
    ];
    let mut m = machine(&prog);
    let report = m.run(64);
    let r = &m.regs;
    assert_eq!(r[5], 0xFFFF_FFFF);
    assert_eq!(r[6], 1);
    assert_eq!(r[7], 1, "slt 有符号: -1 < 1");
    assert_eq!(r[8], 0, "sltu 无符号: 0xFFFFFFFF > 1");
    assert_eq!(r[9], 0xFFFF_FFFF, "srai 保留符号");
    assert_eq!(r[10], 0x7FFF_FFFF, "srli 补零");
    assert_eq!(r[11], 0xFFFF_FFFE, "slli");
    assert_eq!(r[12], 0xFFFF_FFFF, "sra 移位量取低 5 位 (33&31=1)");
    assert_eq!(r[14], 0x7FFF_FFFF, "srl 移位量取低 5 位");
    assert_eq!(r[15], 2, "sll 移位量取低 5 位");
    assert_eq!(r[16], 2, "blt 跳转成立");
    assert_eq!(r[17], 3, "bge 不跳转");
    assert_eq!(r[18], 5, "bgeu 跳转成立");
    assert_eq!(r[19], 6, "bltu 不跳转");
    assert_eq!(r[21], 1, "slti 有符号");
    assert_eq!(r[22], 0, "sltiu 无符号");

    // 轨迹抽查：slt 那一步的前后 PC 与提交记录。
    let slt = report.trace.iter().find(|s| s.disasm.starts_with("slt ")).unwrap();
    assert_eq!(slt.pc_before, 0x08);
    assert_eq!(slt.pc_after, 0x0C);
    assert_eq!(slt.commits, vec![Commit::Reg { rd: 7, val: 1 }]);
    // 分支成立那一步：pc 从 0x2C 跳到 0x34。
    let blt = report.trace.iter().find(|s| s.disasm.starts_with("blt ")).unwrap();
    assert_eq!(blt.pc_before, 0x2C);
    assert_eq!(blt.pc_after, 0x34);
    assert!(blt.commits.is_empty());
}

#[test]
fn auipc_and_lui() {
    let prog = [
        0x00001297, // auipc x5, 1   imm[31:12]=1 -> x5 = pc(0x00) + 0x1000 = 0x1000
        0x00000317, // auipc x6, 0                -> x6 = pc(0x04) + 0      = 0x0004
        0x00ABC3B7, // lui   x7, 0xABC            -> x7 = 0x00ABC000
        0x0000006F, // jal x0, 0
    ];
    let mut m = machine(&prog);
    m.run(8);
    assert_eq!(m.regs[5], 0x1000);
    assert_eq!(m.regs[6], 0x0004);
    assert_eq!(m.regs[7], 0x00AB_C000);
}

// ---------------------------------------------------------------------------
// 字节 / 半字 / 字访存与符号扩展
// ---------------------------------------------------------------------------
#[test]
fn load_store_sign_extension_and_endianness() {
    let prog = [
        0x10000513, // addi x10, x0, 0x100   数据基址 0x100
        0xFFF00293, // addi x5,  x0, -1       x5 = 0xFFFFFFFF
        0x00552023, // sw   x5, 0(x10)        S型 imm=0 rs2=5 rs1=10 f3=2
        0x00050303, // lb   x6, 0(x10)        f3=0 -> 0xFFFFFFFF（符号扩展）
        0x00054383, // lbu  x7, 0(x10)        f3=4 -> 0x000000FF
        0x00051403, // lh   x8, 0(x10)        f3=1 -> 0xFFFFFFFF
        0x00055483, // lhu  x9, 0(x10)        f3=5 -> 0x0000FFFF
        0x00052583, // lw   x11, 0(x10)       f3=2 -> 0xFFFFFFFF
        0x07F00613, // addi x12, x0, 0x7F
        0x00C50223, // sb   x12, 4(x10)       S型 imm=4  -> mem[0x104] = 0x7F
        0x00C51323, // sh   x12, 6(x10)       S型 imm=6  -> mem[0x106..0x108] = 7F 00
        0x00450683, // lb   x13, 4(x10)       -> 0x0000007F
        0x00651703, // lh   x14, 6(x10)       -> 0x0000007F
        0x00554783, // lbu  x15, 5(x10)       mem[0x105] 从未写过 -> 0
        0x0000006F, // jal x0, 0
    ];
    let mut m = machine(&prog);
    let report = m.run(32);
    let r = &m.regs;
    assert_eq!(r[6], 0xFFFF_FFFF, "lb 符号扩展");
    assert_eq!(r[7], 0x0000_00FF, "lbu 零扩展");
    assert_eq!(r[8], 0xFFFF_FFFF, "lh 符号扩展");
    assert_eq!(r[9], 0x0000_FFFF, "lhu 零扩展");
    assert_eq!(r[11], 0xFFFF_FFFF, "lw");
    assert_eq!(r[13], 0x7F);
    assert_eq!(r[14], 0x7F);
    assert_eq!(r[15], 0);
    // 小端序直接核验 RAM 字节。
    assert_eq!(&m.ram[0x100..0x104], &[0xFF, 0xFF, 0xFF, 0xFF]);
    assert_eq!(m.ram[0x104], 0x7F);
    assert_eq!(&m.ram[0x106..0x108], &[0x7F, 0x00]);
    // 轨迹中的 RAM 提交记录。
    let sw = &report.trace[2];
    assert_eq!(sw.pc_before, 0x08);
    assert_eq!(
        sw.commits,
        vec![Commit::Ram { addr: 0x100, size: 4, val: 0xFFFF_FFFF }]
    );
}

// ---------------------------------------------------------------------------
// 陷阱：未对齐访存
// ---------------------------------------------------------------------------
#[test]
fn misaligned_load_traps_and_commits_nothing() {
    let prog = [
        0x04000293, // addi x5, x0, 0x40      处理程序地址
        0x30529073, // csrrw x0, mtvec, x5    imm=0x305 rs1=5 f3=1 rd=0
        0x10000513, // addi x10, x0, 0x100
        0x00551783, // lh   x15, 5(x10)       EA=0x105 半字未对齐 -> mcause=4
        0x00100813, // addi x16, x0, 1        mret 后从这里继续
        0x0000006F, // jal x0, 0
    ];
    let mut m = with_handler(&prog, &H1);
    let report = m.run(32);
    assert_eq!(m.regs[28], CAUSE_LOAD_MISALIGNED);
    assert_eq!(m.regs[29], 0x105, "mtval = 出错地址");
    assert_eq!(m.regs[30], 0x10, "mepc(0x0C)+4");
    assert_eq!(m.regs[15], 0, "故障加载不写 rd");
    assert_eq!(m.regs[16], 1, "mret 跳过故障指令后继续");
    let rec = report.trace.iter().find(|s| s.trap.is_some()).unwrap();
    assert_eq!(rec.pc_before, 0x0C);
    assert_eq!(rec.pc_after, 0x40, "直接入口 pc = mtvec");
    assert!(rec.commits.is_empty(), "故障指令无提交");
    assert_eq!(rec.trap.unwrap().mepc, 0x0C);

    // 读入 x0 也不能豁免对齐检查。
    let prog_x0 = [
        0x04000293, // addi x5, x0, 0x40
        0x30529073, // csrrw x0, mtvec, x5
        0x10000513, // addi x10, x0, 0x100
        0x00551003, // lh   x0, 5(x10)        rd=x0 仍然未对齐 -> 陷阱
        0x0000006F, // jal x0, 0
    ];
    let mut m2 = with_handler(&prog_x0, &H1);
    m2.run(32);
    assert_eq!(m2.regs[28], CAUSE_LOAD_MISALIGNED);
    assert_eq!(m2.regs[29], 0x105);
}

#[test]
fn misaligned_store_commits_nothing() {
    let prog = [
        0x04000293, // addi x5, x0, 0x40
        0x30529073, // csrrw x0, mtvec, x5
        0x10000513, // addi x10, x0, 0x100
        0x05500413, // addi x8, x0, 0x55
        0x00852323, // sw   x8, 6(x10)        EA=0x106 字未对齐 -> mcause=6
        0x0000006F, // jal x0, 0
    ];
    let mut m = with_handler(&prog, &H1);
    m.ram[0x104..0x108].copy_from_slice(&[0xAA; 4]); // 预置哨兵字节
    m.run(32);
    assert_eq!(m.regs[28], CAUSE_STORE_MISALIGNED);
    assert_eq!(m.regs[29], 0x106);
    assert_eq!(&m.ram[0x104..0x108], &[0xAA; 4], "未对齐存储不得有部分写入");
}

// ---------------------------------------------------------------------------
// 跨界写入：RAM 末尾边界
// ---------------------------------------------------------------------------
#[test]
fn boundary_writes_at_ram_end() {
    let prog = [
        0x04000293, // addi x5, x0, 0x40
        0x30529073, // csrrw x0, mtvec, x5
        0x00010537, // lui  x10, 0x10         x10 = 0x10000（RAM 之外第一地址）
        0x12345437, // lui  x8, 0x12345       x8 = 0x12345000
        0x67840413, // addi x8, x8, 0x678     x8 = 0x12345678
        0xFE852E23, // sw   x8, -4(x10)       EA=0xFFFC：最后一个合法字，完整写入
        0xFE850FA3, // sb   x8, -1(x10)       EA=0xFFFF：最后一个合法字节
        0x00852023, // sw   x8, 0(x10)        EA=0x10000：越界 -> mcause=7，不得写入
        0x0000006F, // jal x0, 0
    ];
    let mut m = with_handler(&prog, &H1);
    m.run(32);
    // 0xFFFC..0x10000：sw 写入 78 56 34 12，随后 sb 把最后一字节改成 78。
    assert_eq!(&m.ram[0xFFFC..0x10000], &[0x78, 0x56, 0x34, 0x78]);
    assert_eq!(m.regs[28], CAUSE_STORE_FAULT);
    assert_eq!(m.regs[29], 0x10000, "mtval = 越界地址");
    assert_eq!(m.regs[30], 0x20, "mepc(0x1C)+4");
}

// ---------------------------------------------------------------------------
// 陷阱：取指越界
// ---------------------------------------------------------------------------
#[test]
fn fetch_fault_out_of_ram() {
    let prog = [
        0x0002006F, // jal x0, +0x20000   imm[19:12]=0x20 -> 目标 0x20000（RAM 之外）
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    let report = m.run(16);
    assert_eq!(m.regs[28], CAUSE_FETCH_FAULT);
    assert_eq!(m.regs[29], 0x20000, "mtval = 失败的取指地址");
    assert_eq!(m.regs[30], 0x20000, "mepc = 失败的取指地址");
    let rec = &report.trace[1];
    assert_eq!(rec.pc_before, 0x20000);
    assert_eq!(rec.instr, None, "取指失败没有指令字");
    assert_eq!(rec.pc_after, 0x40);
}

// ---------------------------------------------------------------------------
// 陷阱返回：非法指令 -> 处理程序 -> mret 继续
// ---------------------------------------------------------------------------
#[test]
fn illegal_instruction_then_mret() {
    let prog = [
        0x04000293, // addi x5, x0, 0x40
        0x30529073, // csrrw x0, mtvec, x5
        0xFFFFFFFF, // 非法指令（opcode=0x7F 未分配）
        0x00700493, // addi x9, x0, 7        mret 后应执行到这里
        0x0000006F, // jal x0, 0
    ];
    let mut m = with_handler(&prog, &H1);
    let report = m.run(32);
    assert_eq!(m.regs[28], CAUSE_ILLEGAL);
    assert_eq!(m.regs[29], 0xFFFF_FFFF, "mtval = 指令原始位");
    assert_eq!(m.regs[30], 0x0C, "mepc(0x08)+4");
    assert_eq!(m.regs[9], 7, "mret 返回到故障指令之后");
    assert_eq!(m.mcause, CAUSE_ILLEGAL);
    assert_eq!(m.mepc, 0x0C);
    assert_eq!(m.mtval, 0xFFFF_FFFF);
    assert_eq!(m.pc, 0x10, "停在原地循环处");
    let rec = report.trace.iter().find(|s| s.trap.is_some()).unwrap();
    assert_eq!(rec.instr, Some(0xFFFF_FFFF));
    assert_eq!(rec.pc_before, 0x08);
    assert_eq!(rec.pc_after, 0x40);
    assert!(rec.commits.is_empty());
}

#[test]
fn ecall_and_ebreak_trap() {
    let mut m = with_handler(&[0x00000073], &H2); // ecall
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_ECALL);
    assert_eq!(m.regs[29], 0);
    assert_eq!(m.regs[30], 0x00);

    let mut m = with_handler(&[0x00100073], &H2); // ebreak
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_EBREAK);
}

// ---------------------------------------------------------------------------
// JALR：最低位清除与对齐检查不可互相替代
// ---------------------------------------------------------------------------
#[test]
fn jalr_clears_lsb_then_checks_alignment() {
    let prog = [
        0x02100293, // addi x5, x0, 0x21      0x00: 目标 0x21（bit0=1）
        0x000280E7, // jalr x1, 0(x5)         0x04: 清 bit0 -> 0x20，对齐检查通过，x1=0x08
        0x00000013, // nop (addi x0,x0,0)     0x08: 填充
        0x00000013, // nop                    0x0C
        0x00000013, // nop                    0x10
        0x00000013, // nop                    0x14
        0x00000013, // nop                    0x18
        0x00000013, // nop                    0x1C
        0x00900313, // addi x6, x0, 9         0x20: 落点，x6=9
        0x02200293, // addi x5, x0, 0x22      0x24: 目标 0x22（bit0=0 但 2 mod 4）
        0x000283E7, // jalr x7, 0(x5)         0x28: 清 bit0 仍是 0x22 -> 未对齐陷阱，x7 不写入
        0x00300413, // addi x8, x0, 3         0x2C: mret 后执行，x8=3
        0x0000006F, // jal x0, 0              0x30
    ];
    let mut m = with_handler(&prog, &H1);
    m.mtvec = 0x40;
    m.run(64);
    assert_eq!(m.regs[1], 0x08, "JALR 链接寄存器 = pc+4");
    assert_eq!(m.regs[6], 9, "0x21 清 bit0 后落在 0x20");
    assert_eq!(m.regs[28], CAUSE_FETCH_MISALIGNED);
    assert_eq!(m.regs[29], 0x22, "mtval = 清 bit0 之后仍未对齐的目标");
    assert_eq!(m.regs[30], 0x2C, "mepc(0x28)+4");
    assert_eq!(m.regs[7], 0, "故障 JALR 不写链接寄存器");
    assert_eq!(m.regs[8], 3, "mret 后继续执行");
}

#[test]
fn misaligned_branch_and_jal_targets() {
    // 分支目标未对齐：异常算在分支指令头上。
    let prog = [
        0x00000163, // beq x0, x0, +2    目标 0x02 -> mcause=0, mtval=2, mepc=0
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_FETCH_MISALIGNED);
    assert_eq!(m.regs[29], 2);
    assert_eq!(m.regs[30], 0);

    // JAL 目标未对齐：链接寄存器不写入。
    let prog = [
        0x002000EF, // jal x1, +2        目标 0x02 -> 陷阱，x1 保持 0
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_FETCH_MISALIGNED);
    assert_eq!(m.regs[29], 2);
    assert_eq!(m.regs[1], 0, "故障 JAL 不写链接寄存器");
}

#[test]
fn mret_to_misaligned_mepc() {
    let prog = [
        0x04200293, // addi x5, x0, 0x42
        0x34129073, // csrrw x0, mepc, x5    mepc = 0x42（未对齐）
        0x30200073, // mret                  -> 异常算在 MRET 头上：mepc=0x08, mtval=0x42
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_FETCH_MISALIGNED);
    assert_eq!(m.regs[29], 0x42);
    assert_eq!(m.regs[30], 0x08, "mepc = MRET 自己的 PC");
}

// ---------------------------------------------------------------------------
// 端口：接收队列读端口与发送写端口
// ---------------------------------------------------------------------------
#[test]
fn rx_tx_ports() {
    let prog = [
        0x80000537, // lui  x10, 0x80000      x10 = 0x80000000（RX 端口）
        0x00050283, // lb   x5, 0(x10)        弹出 0x11 -> x5
        0x00054303, // lbu  x6, 0(x10)        弹出 0x22 -> x6
        0x00050003, // lb   x0, 0(x10)        rd=x0 仍弹出 0x33，结果丢弃
        0x00054383, // lbu  x7, 0(x10)        队列已空 -> 0xFF
        0x05500413, // addi x8, x0, 0x55
        0x00850223, // sb   x8, 4(x10)        TX: 55
        0x00852223, // sw   x8, 4(x10)        TX: 55 00 00 00
        0x00851223, // sh   x8, 4(x10)        TX: 55 00
        0x0000006F, // jal x0, 0
    ];
    let mut m = machine(&prog);
    for b in [0x11u8, 0x22, 0x33] {
        m.push_rx(b);
    }
    let report = m.run(32);
    assert_eq!(m.regs[5], 0x11);
    assert_eq!(m.regs[6], 0x22);
    assert_eq!(m.regs[7], 0xFF, "空队列读出 0xFF");
    assert_eq!(m.regs[0], 0, "x0 恒为 0");
    assert!(m.rx.is_empty(), "三次读取把队列弹空（含 rd=x0 那次）");
    assert_eq!(m.tx, vec![0x55, 0x55, 0, 0, 0, 0x55, 0], "sb/sw/sh 依次落日志");

    // rd=x0 的端口读取：有 RxPop 提交、无寄存器提交。
    let rec = &report.trace[3];
    assert_eq!(rec.pc_before, 0x0C);
    assert!(rec.commits.contains(&Commit::RxPop { byte: Some(0x33) }));
    assert!(!rec.commits.iter().any(|c| matches!(c, Commit::Reg { .. })));
    // 空队列读取的记录。
    let rec = &report.trace[4];
    assert!(rec.commits.contains(&Commit::RxPop { byte: None }));
}

#[test]
fn port_wrong_direction_faults() {
    // 读只写端口（TX）-> 加载访问故障。
    let prog = [
        0x80000537, // lui x10, 0x80000
        0x00452283, // lw  x5, 4(x10)         EA=0x80000004 = TX_ADDR
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_LOAD_FAULT);
    assert_eq!(m.regs[29], TX_ADDR);
    assert_eq!(m.regs[5], 0);

    // 写只读端口（RX）-> 存储访问故障，接收队列不受影响。
    let prog = [
        0x80000537, // lui  x10, 0x80000
        0x00100413, // addi x8, x0, 1
        0x00852023, // sw   x8, 0(x10)        EA=0x80000000 = RX_ADDR
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    m.push_rx(0xAB);
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_STORE_FAULT);
    assert_eq!(m.regs[29], RX_ADDR);
    assert_eq!(m.rx.len(), 1, "故障存储不改变接收队列");
    assert!(m.tx.is_empty());
}

#[test]
fn unmapped_address_fault() {
    let prog = [
        0x40000537, // lui x10, 0x40000       x10 = 0x40000000（未映射）
        0x00052283, // lw  x5, 0(x10)
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_LOAD_FAULT);
    assert_eq!(m.regs[29], 0x4000_0000);
}

// ---------------------------------------------------------------------------
// CSR 语义
// ---------------------------------------------------------------------------
#[test]
fn csr_rw_rs_semantics() {
    let prog = [
        0x04000293, // addi x5, x0, 0x40
        0x30529373, // csrrw x6,  mtvec, x5   x6 <- 旧 mtvec(0)，mtvec <- 0x40
        0x305023F3, // csrrs x7,  mtvec, x0   rs1=x0：只读不写，x7 <- 0x40
        0x00300413, // addi x8, x0, 3
        0x305424F3, // csrrs x9,  mtvec, x8   x9 <- 0x40，mtvec <- 0x40|3 = 0x43
        0x30501573, // csrrw x10, mtvec, x0   x10 <- 0x43，mtvec <- 0（rs1=x0 也照写）
        0x305025F3, // csrrs x11, mtvec, x0   x11 <- 0
        0x0000006F, // jal x0, 0
    ];
    let mut m = machine(&prog);
    m.run(32);
    assert_eq!(m.regs[6], 0);
    assert_eq!(m.regs[7], 0x40);
    assert_eq!(m.regs[9], 0x40);
    assert_eq!(m.regs[10], 0x43);
    assert_eq!(m.regs[11], 0);
    assert_eq!(m.mtvec, 0);
}

#[test]
fn unknown_csr_is_illegal() {
    let prog = [
        0x777012F3, // csrrw x5, 0x777, x0    未实现的 CSR -> 非法指令
    ];
    let mut m = with_handler(&prog, &H2);
    m.mtvec = 0x40;
    m.run(16);
    assert_eq!(m.regs[28], CAUSE_ILLEGAL);
    assert_eq!(m.regs[29], 0x7770_12F3, "mtval = 指令原始位");
    assert_eq!(m.regs[30], 0);
}

// ---------------------------------------------------------------------------
// 步数预算与 x0
// ---------------------------------------------------------------------------
#[test]
fn step_budget_caps_run() {
    let prog = [0x0000006F]; // jal x0, 0：无限循环
    let mut m = machine(&prog);
    let report = m.run(2000);
    assert_eq!(report.trace.len(), 2000);
    assert_eq!(report.state.steps, 2000);
    assert!(report.trace.iter().all(|s| s.pc_before == 0 && s.pc_after == 0));

    let mut m = machine(&prog);
    assert_eq!(m.run(10).trace.len(), 10);

    // 超出上限的预算被夹到 2000。
    let mut m = machine(&prog);
    let report = m.run(5000);
    assert_eq!(report.trace.len(), 2000);
    assert_eq!(report.budget, 2000);
    assert_eq!(m.steps, 2000);
}

#[test]
fn writes_to_x0_are_dropped() {
    let prog = [
        0x00500013, // addi x0, x0, 5     写 x0：被丢弃
        0x0000006F, // jal x0, 0
    ];
    let mut m = machine(&prog);
    let report = m.run(4);
    assert_eq!(m.regs[0], 0);
    assert!(report.trace[0].commits.is_empty(), "写 x0 不产生提交记录");
}

#[test]
fn trace_lines_are_informative() {
    let prog = [
        0x00100093, // addi x1, x0, 1
        0x0000006F, // jal x0, 0
    ];
    let mut m = machine(&prog);
    let report = m.run(2);
    let line = report.trace[0].to_string();
    assert!(line.contains("pc=00000000"), "轨迹含执行前 PC: {line}");
    assert!(line.contains("instr=00100093"), "轨迹含指令位: {line}");
    assert!(line.contains("addi ra, zero, 1"), "轨迹含助记符: {line}");
    assert!(line.contains("ra <= 00000001"), "轨迹含提交变化: {line}");
    assert!(line.contains("next pc=00000004"), "轨迹含执行后 PC: {line}");
}
