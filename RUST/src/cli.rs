// Commands: chat / inspect / dump

use crate::gguf::{Gguf, KvVal};
use crate::model::{moe_summary, Model};
use crate::quant::type_name;
use crate::sampler::Sampler;
use crate::tokenizer::{Tokenizer, Utf8Out};
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// write bytes to stdout; invalid UTF-8 is replaced (the Windows console accepts only UTF-8)
pub fn emit(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(String::from_utf8_lossy(bytes).as_bytes());
    let _ = out.flush();
}

/// one line from stdin without the line end, or None at the end of input
fn read_line() -> Option<String> {
    let mut s = String::new();
    match std::io::stdin().lock().read_line(&mut s) {
        Ok(0) | Err(_) => None,
        Ok(_) => {
            if s.starts_with('\u{1a}') {
                return None; // Ctrl+Z
            }
            while s.ends_with('\n') || s.ends_with('\r') {
                s.pop();
            }
            Some(s)
        }
    }
}

/// max_tokens <= 0: no limit (until a stop token or the context is full)
pub fn cmd_chat(path: &Path, n_ctx: usize, max_tokens: i64, smp: &mut Sampler) -> Result<(), String> {
    let g = Gguf::open(path)?;
    let tk = Tokenizer::load(&g)?;
    let mut m = Model::load(g, n_ctx)?;

    println!(
        "\n=== fastgguf-rs | CPU kernels: {} | {} threads ===",
        crate::simd::cpu_name(),
        rayon::current_num_threads()
    );
    // without --seed: a seed from the clock, printed so the same reply can be reproduced
    if smp.rng == 0 {
        let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        smp.rng = t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (t >> 17);
    }
    if smp.rng == 0 {
        smp.rng = 0x9E37_79B9_7F4A_7C15;
    }
    if smp.temp <= 0.0 {
        print!("[Sampler] greedy");
    } else {
        print!("[Sampler] temp={:.2} top_k={} top_p={:.2}", smp.temp, smp.top_k, smp.top_p);
    }
    println!(
        " repeat_penalty={:.2} (last {}) max_tokens={} seed={}",
        smp.repeat_penalty, smp.repeat_last_n, max_tokens, smp.rng
    );
    println!("Type your message, /reset for a new conversation, or /exit to quit");

    let mut pos = 0usize; // conversation length in the KV cache — kept between messages until /reset
    let mut hist: Vec<i32> = Vec::with_capacity(n_ctx); // hist[p] = token at position p
    loop {
        emit(b"\nyou> ");
        let Some(line) = read_line() else { break };
        if line.is_empty() {
            continue;
        }
        if line == "/exit" {
            break;
        }
        if line == "/reset" {
            pos = 0;
            hist.clear();
            println!("[new conversation]");
            continue;
        }

        let mut ids = tk.encode_turn(&line, pos);
        // the turn + one reply token must fit; otherwise start again with this message alone —
        // unless even that does not fit, then reject it and keep the conversation
        if pos > 0 && pos + ids.len() >= n_ctx {
            let fresh = tk.encode_turn(&line, 0);
            if fresh.len() < n_ctx {
                println!("[context full ({pos}/{n_ctx}) - starting a new conversation]");
                ids = fresh;
                pos = 0;
                hist.clear();
            }
        }
        if pos + ids.len() >= n_ctx {
            println!("[message too long: {} tokens, context is {n_ctx} - run with a larger ctx]", ids.len());
            continue;
        }
        emit(format!("[{} tokens | ctx {}/{}]\n{}> ", ids.len(), pos + ids.len(), n_ctx, m.arch).as_bytes());

        let t0 = Instant::now();
        if let Err(e) = m.forward_batch(&ids, pos) {
            // the cache is in a partial state now: it cannot be built upon
            println!("\n[ERROR in prompt: {e} - conversation reset]");
            pos = 0;
            hist.clear();
            continue;
        }
        hist.extend_from_slice(&ids);
        pos += ids.len();
        let t_prompt = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut n_gen = 0;
        let mut u8out = Utf8Out::new();
        let mut buf = Vec::new();
        let mut i = 0i64;
        while max_tokens <= 0 || i < max_tokens {
            i += 1;
            let next = smp.sample(&tk, &mut m.logits, &hist);
            if tk.is_stop(next) {
                break;
            }
            buf.clear();
            u8out.push(&tk.token_bytes(next), &mut buf);
            emit(&buf);
            n_gen += 1;
            if pos >= n_ctx {
                buf.clear();
                u8out.flush(&mut buf);
                emit(&buf);
                emit(b"\n[context full]");
                break;
            }
            if let Err(e) = m.forward(next, pos) {
                buf.clear();
                u8out.flush(&mut buf);
                emit(&buf);
                println!("\n[ERROR in gen: {e} - conversation reset]");
                pos = 0;
                hist.clear();
                break;
            }
            hist.push(next);
            pos += 1;
        }
        buf.clear();
        u8out.flush(&mut buf);
        emit(&buf);
        let t_gen = t1.elapsed().as_secs_f64();
        println!(
            "\n[prompt: {:.1}s ({:.2} tok/s) | gen: {} tok @ {:.2} tok/s]",
            t_prompt,
            ids.len() as f64 / if t_prompt > 0.0 { t_prompt } else { 1.0 },
            n_gen,
            n_gen as f64 / if t_gen > 0.0 { t_gen } else { 1.0 }
        );
    }
    Ok(())
}

pub fn cmd_inspect(path: &Path) -> Result<(), String> {
    let g = Gguf::open(path)?;
    println!("GGUF v{} | tensors={} | meta={} | align={}", g.version, g.tensors.len(), g.kvs.len(), g.alignment);
    if let Some(arch) = g.str("general.architecture") {
        println!("arch: {arch}");
        if let Some(s) = moe_summary(&g, arch) {
            println!("{s}");
        }
    }
    let mut counts = [0usize; 64];
    for t in &g.tensors {
        if t.typ < 64 {
            counts[t.typ as usize] += 1;
        }
    }
    println!("\ntensor types:");
    for (t, &c) in counts.iter().enumerate() {
        if c > 0 {
            println!("  {:<6} {}", type_name(t as u32), c);
        }
    }
    println!("\nfirst tensors:");
    for t in g.tensors.iter().take(15) {
        let dims: Vec<String> = (0..t.n_dims).map(|j| t.ne[j].to_string()).collect();
        println!("  {:<44} {:<5} [{}]", t.name, type_name(t.typ), dims.join(","));
    }
    Ok(())
}

/// printf("%g")
pub fn fmt_g(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if !v.is_finite() {
        return format!("{v}");
    }
    let trim = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    let exp = v.abs().log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        let s = format!("{v:.5e}");
        let (m, e) = s.split_once('e').unwrap();
        let e: i32 = e.parse().unwrap_or(0);
        format!("{}e{}{:02}", trim(m.to_string()), if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        trim(format!("{:.*}", (5 - exp).max(0) as usize, v))
    }
}

pub fn cmd_dump(path: &Path) -> Result<(), String> {
    let g = Gguf::open(path)?;
    let mut out = std::io::stdout().lock();
    for kv in &g.kvs {
        let line = match &kv.val {
            KvVal::Str(s) => format!("{} = {}\n", kv.key, s),
            KvVal::Num(n) => format!("{} = {}\n", kv.key, fmt_g(*n)),
            KvVal::IntArr(a) => format!("{} = [{} ints]\n", kv.key, a.len()),
            KvVal::StrArr(a) => format!("{} = [{} strings]\n", kv.key, a.len()),
        };
        let _ = out.write_all(line.as_bytes());
    }
    Ok(())
}
