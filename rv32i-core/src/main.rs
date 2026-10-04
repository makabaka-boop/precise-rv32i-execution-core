#![forbid(unsafe_code)]
//! RV32I 教学执行核的命令行前端：读取原始机器码、初始寄存器与 RAM 镜像，
//! 以不超过 2000 步的预算执行并打印逐步轨迹与最终机器状态。

use rv32i_core::{Machine, MAX_STEPS};
use std::{env, fs, process};

const USAGE: &str = "\
用法: rv32i-core <程序.bin> [选项]

读取原始 RV32I 机器码并执行（64 KiB RAM、仅机器模式、RX/TX 端口）。

选项:
  --addr 0xN    程序装载地址（默认 0x0），PC 同时初始化为该地址
  --regs 文件   初始寄存器，每行一条：
                  x5 = 0x0000002a
                  pc / mtvec / mepc / mcause / mtval = 0xV
  --ram  文件   初始 RAM 内容，每行一段：
                  0x1000: 78 56 34 12
  --rx   文件   接收队列的初始字节（原始二进制，按序入队）
  --steps N     步数预算（默认 2000，上限 2000）
  --quiet       不打印逐步轨迹，只打印最终状态
";

fn die(msg: &str) -> ! {
    eprintln!("错误: {msg}\n\n{USAGE}");
    process::exit(2);
}

/// 解析 0x 前缀的十六进制或十进制无符号数。
fn parse_u32(s: &str) -> u32 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).unwrap_or_else(|_| die(&format!("无法解析数值: {s}")))
    } else {
        s.parse::<u32>().unwrap_or_else(|_| die(&format!("无法解析数值: {s}")))
    }
}

fn read_file(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| die(&format!("无法读取文件 {path}: {e}")))
}

/// 应用初始寄存器文件：`xN = 0xV` / `pc = 0xV` / `mtvec = 0xV` 等。
fn apply_regs(m: &mut Machine, text: &str) {
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let (name, val) = line
            .split_once('=')
            .unwrap_or_else(|| die(&format!("寄存器文件第 {} 行缺少 '='", ln + 1)));
        let v = parse_u32(val);
        let name = name.trim();
        match name {
            "pc" => m.pc = v,
            "mtvec" => m.mtvec = v,
            "mepc" => m.mepc = v,
            "mcause" => m.mcause = v,
            "mtval" => m.mtval = v,
            _ => match name.strip_prefix('x').and_then(|n| n.parse::<usize>().ok()) {
                Some(0) => eprintln!("警告: x0 恒为 0，忽略第 {} 行", ln + 1),
                Some(n) if n < 32 => m.regs[n] = v,
                _ => die(&format!("寄存器文件第 {} 行名称无法识别: {name}", ln + 1)),
            },
        }
    }
}

/// 应用 RAM 初始化文件：`0x地址: AA BB CC ...`。
fn apply_ram(m: &mut Machine, text: &str) {
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let (addr, bytes) = line
            .split_once(':')
            .unwrap_or_else(|| die(&format!("RAM 文件第 {} 行缺少 ':'", ln + 1)));
        let base = parse_u32(addr) as usize;
        for (i, tok) in bytes.split_whitespace().enumerate() {
            let b = u8::from_str_radix(tok, 16)
                .unwrap_or_else(|_| die(&format!("RAM 文件第 {} 行字节无法解析: {tok}", ln + 1)));
            if base + i >= rv32i_core::RAM_SIZE {
                die(&format!("RAM 文件第 {} 行超出 64 KiB 边界", ln + 1));
            }
            m.ram[base + i] = b;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut prog_path: Option<String> = None;
    let mut addr: u32 = 0;
    let mut regs_file: Option<String> = None;
    let mut ram_file: Option<String> = None;
    let mut rx_file: Option<String> = None;
    let mut steps: u64 = MAX_STEPS;
    let mut quiet = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--addr" => {
                i += 1;
                addr = parse_u32(args.get(i).unwrap_or_else(|| die("--addr 缺少参数")));
            }
            "--regs" => {
                i += 1;
                regs_file = Some(args.get(i).unwrap_or_else(|| die("--regs 缺少参数")).clone());
            }
            "--ram" => {
                i += 1;
                ram_file = Some(args.get(i).unwrap_or_else(|| die("--ram 缺少参数")).clone());
            }
            "--rx" => {
                i += 1;
                rx_file = Some(args.get(i).unwrap_or_else(|| die("--rx 缺少参数")).clone());
            }
            "--steps" => {
                i += 1;
                steps = parse_u32(args.get(i).unwrap_or_else(|| die("--steps 缺少参数"))) as u64;
            }
            "--quiet" => quiet = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return;
            }
            s if !s.starts_with('-') && prog_path.is_none() => prog_path = Some(s.to_string()),
            s => die(&format!("未知参数: {s}")),
        }
        i += 1;
    }

    let prog_path = prog_path.unwrap_or_else(|| die("缺少程序文件"));
    let prog = read_file(&prog_path);

    let mut m = Machine::new();
    m.load_bytes(addr, &prog);
    m.pc = addr;
    if let Some(f) = &ram_file {
        apply_ram(&mut m, &String::from_utf8_lossy(&read_file(f)));
    }
    if let Some(f) = &regs_file {
        apply_regs(&mut m, &String::from_utf8_lossy(&read_file(f)));
    }
    if let Some(f) = &rx_file {
        for &b in &read_file(f) {
            m.push_rx(b);
        }
    }

    let report = m.run(steps);
    if !quiet {
        for rec in &report.trace {
            println!("{rec}");
        }
    }

    println!("--- 最终状态 ---");
    println!("{}", report.state);
    print!("tx 日志 ({} 字节):", m.tx.len());
    for b in &m.tx {
        print!(" {b:02x}");
    }
    println!();
    println!("rx 剩余: {} 字节", m.rx.len());
}
