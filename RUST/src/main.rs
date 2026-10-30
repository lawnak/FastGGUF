// fastgguf (Rust) — GGUF inference engine, a port of fastgguf.c
//
//   fastgguf chat     model.gguf [ctx] [--temp 1.0] [--top-k 64] [--top-p 0.95]
//                     [--repeat-penalty 1.0] [--repeat-last-n 64] [--max-tokens 512] [--seed N]
//   fastgguf serve    model.gguf [model2.gguf ...] [port 8080] [ctx 4096]
//   fastgguf inspect  model.gguf
//   fastgguf dump     model.gguf
//   fastgguf selftest
//
//   fastgguf gpu
//
// CPU (AVX2 / AVX-VNNI / AVX-512 VNNI picked at start-up) and GPU (OpenCL, loaded at run time).

mod batch;
mod cli;
mod gguf;
mod gpu;
mod json;
mod matvec;
mod model;
mod ops;
mod quant;
mod repack;
mod sampler;
mod selftest;
mod server;
mod simd;
mod tokenizer;

use sampler::Sampler;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const CHAT_OPTIONS_HELP: &str = "chat options: --temp 1.0 (0 = greedy)  --top-k 64 (0 = off)  --top-p 0.95
              --repeat-penalty 1.0 (1 = off)  --repeat-last-n 64
              --max-tokens 512 (0 = until stop)  --seed N
";

/// Windows 11 puts a process that waits a long time into efficiency mode (EcoQoS), especially
/// on battery, which slows the compute ~3x: ask not to be throttled
#[cfg(windows)]
fn no_power_throttling() {
    #[repr(C)]
    struct PowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetProcessInformation(process: isize, class: i32, info: *const PowerThrottlingState, size: u32) -> i32;
    }
    const PROCESS_POWER_THROTTLING: i32 = 4; // ProcessPowerThrottling
    let st = PowerThrottlingState { version: 1, control_mask: 1, state_mask: 0 }; // EXECUTION_SPEED, off
    // SAFETY: plain Win32 call with a correctly sized struct
    unsafe {
        SetProcessInformation(GetCurrentProcess(), PROCESS_POWER_THROTTLING, &st, std::mem::size_of::<PowerThrottlingState>() as u32);
    }
}
#[cfg(not(windows))]
fn no_power_throttling() {}

/// options after the model path: positional [ctx] then --temp/--top-k/...
fn parse_chat_args(args: &[String], ctx: &mut usize, max_tokens: &mut i64, smp: &mut Sampler) -> Result<(), String> {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if !a.starts_with('-') {
            *ctx = a.parse().map_err(|_| format!("bad context size {a}"))?; // positional [ctx]
            i += 1;
            continue;
        }
        let v = args.get(i + 1).ok_or(format!("missing value for {a}"))?;
        let bad = || format!("bad value for {a}: {v}");
        match a.as_str() {
            "--temp" => smp.temp = v.parse().map_err(|_| bad())?,
            "--top-k" => smp.top_k = v.parse().map_err(|_| bad())?,
            "--top-p" => smp.top_p = v.parse().map_err(|_| bad())?,
            "--repeat-penalty" => smp.repeat_penalty = v.parse().map_err(|_| bad())?,
            "--repeat-last-n" => smp.repeat_last_n = v.parse().map_err(|_| bad())?,
            "--max-tokens" => *max_tokens = v.parse().map_err(|_| bad())?,
            "--seed" => smp.rng = v.parse().map_err(|_| bad())?,
            _ => return Err(format!("unknown option {a}")),
        }
        i += 2;
    }
    *ctx = (*ctx).max(64);
    Ok(())
}

fn usage() {
    print!(
        "Usage: fastgguf <chat|serve|inspect|dump|gpu|selftest> <model.gguf> [...]
  chat:  fastgguf chat model.gguf [ctx] [options]
  serve: fastgguf serve model1.gguf [model2.gguf ... up to 4] [port 8080] [ctx 4096]
{CHAT_OPTIONS_HELP}  GPU: FASTGGUF_GPU=on|off|auto  FASTGGUF_GPU_DEVICE=<n>  FASTGGUF_GPU_MEM=<GB>
       FASTGGUF_GPU_LAYERS=<n>
  KV:  FASTGGUF_KV=turbo|turbo8|turbo4|turbo2   TurboQuant KV cache
  CPU: FASTGGUF_CPU=generic|avx2|avxvnni|avx512   FASTGGUF_REPACK=0   FASTGGUF_BATCH=0
       RAYON_NUM_THREADS=<n>
  API: FASTGGUF_API_KEY=<key>   require 'Authorization: Bearer <key>'
"
    );
}

fn run(args: Vec<OsString>) -> Result<i32, String> {
    // file paths stay OsString (any Unicode name on Windows); the options are text
    let text: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
    if text.len() < 2 {
        usage();
        return Ok(1);
    }
    match text[1].as_str() {
        "selftest" => return Ok(selftest::cmd_selftest()),
        "gpu" => return Ok(gpu::cmd_info()),
        "serve" => {
            // numbers = port then context; everything else = model paths
            let (mut port, mut ctx, mut nums) = (8080u16, 4096usize, 0);
            let mut paths = Vec::new();
            for (a, raw) in text[2..].iter().zip(&args[2..]) {
                if !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) {
                    if nums == 0 {
                        port = a.parse().map_err(|_| format!("bad port {a}"))?;
                    } else {
                        ctx = a.parse().map_err(|_| format!("bad context size {a}"))?;
                    }
                    nums += 1;
                } else if paths.len() < server::MAX_MODELS {
                    paths.push(PathBuf::from(raw));
                }
            }
            if paths.is_empty() {
                println!("missing model path");
                return Ok(1);
            }
            server::cmd_serve(&paths, port, ctx.max(64))?;
            return Ok(0);
        }
        "help" | "-h" | "--help" => {
            usage();
            return Ok(0);
        }
        _ => {}
    }
    if text.len() < 3 {
        println!("missing model path");
        return Ok(1);
    }
    let path = PathBuf::from(&args[2]);
    match text[1].as_str() {
        "chat" => {
            let (mut ctx, mut max_tokens) = (2048usize, 512i64);
            let mut smp = Sampler::default();
            parse_chat_args(&text[3..], &mut ctx, &mut max_tokens, &mut smp)?;
            cli::cmd_chat(&path, ctx, max_tokens, &mut smp)?;
        }
        "inspect" => cli::cmd_inspect(&path)?,
        "dump" => cli::cmd_dump(&path)?,
        cmd => {
            println!("unknown command {cmd}");
            return Ok(1);
        }
    }
    Ok(0)
}

fn main() -> ExitCode {
    no_power_throttling();
    match run(std::env::args_os().collect()) {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
