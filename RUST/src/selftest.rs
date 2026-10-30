// `fastgguf selftest`: every kernel against an independent float reference, the AVX2
// kernels against the portable ones, the repacked layout, the sampler, RoPE and TurboQuant.

use crate::gguf::Tensor;
use crate::matvec::{act_kind, dot_row, matvec, Act, QAct};
use crate::ops::{self, kvq};
use crate::quant::*;
use crate::repack;
use crate::sampler::{top_k_select, Sampler};
use crate::simd::{self, Level};
use crate::tokenizer::Tokenizer;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        (ops::rng_next(&mut self.0) >> 33) as u32
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    /// like the C test: (rand() % 2000 - 1000) / 500
    fn x(&mut self) -> f32 {
        (self.next() % 2000) as f32 / 500.0 - 2.0
    }
}

/// a random row of n values with sensible fp16 scales (no NaN/Inf)
fn random_row(rng: &mut Rng, typ: u32, n: usize) -> Vec<u8> {
    let (be, bb) = block_size(typ).unwrap();
    let (be, bb) = (be as usize, bb as usize);
    let mut row: Vec<u8> = (0..n / be * bb).map(|_| rng.byte()).collect();
    let put = |row: &mut Vec<u8>, at: usize, h: u16| row[at..at + 2].copy_from_slice(&h.to_le_bytes());
    for b in 0..n / be {
        let p = b * bb;
        let d = 0x2C00 + (rng.next() & 0x3FF) as u16; // ~0.06..0.12
        let m = 0x2400 + (rng.next() & 0x3FF) as u16; // ~0.016..0.03
        match typ {
            T_Q4_0 | T_Q5_0 | T_Q8_0 | T_IQ4_NL | T_IQ4_XS => put(&mut row, p, d),
            T_Q4_1 | T_Q5_1 | T_Q4_K | T_Q5_K => {
                put(&mut row, p, d);
                put(&mut row, p + 2, m)
            }
            T_Q2_K => {
                put(&mut row, p + 80, d);
                put(&mut row, p + 82, m)
            }
            T_Q3_K => put(&mut row, p + 108, d),
            T_Q6_K => put(&mut row, p + 208, d),
            _ => {}
        }
    }
    row
}

fn report(name: &str, ok: bool, detail: String, fails: &mut i32) {
    println!("{name:<28} {detail} {}", if ok { "OK" } else { "FAIL" });
    if !ok {
        *fails += 1;
    }
}

fn leak(v: Vec<u8>) -> &'static [u8] {
    Box::leak(v.into_boxed_slice())
}

/// every kernel set this CPU can run
fn levels() -> Vec<Level> {
    [Level::Generic, Level::Avx2, Level::AvxVnni, Level::Avx512].into_iter().filter(|l| simd::supported(*l)).collect()
}

fn short(l: Level) -> &'static str {
    match l {
        Level::Generic => "generic",
        Level::Avx2 => "avx2",
        Level::AvxVnni => "avxvnni",
        Level::Avx512 => "avx512",
    }
}

fn test_kernels(rng: &mut Rng, fails: &mut i32) {
    let n = 1024;
    let types = [
        T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1, T_Q8_0, T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_IQ4_NL, T_IQ4_XS,
    ];
    for typ in types {
        let row = random_row(rng, typ, n);
        let x: Vec<f32> = (0..n).map(|_| rng.x()).collect();
        let kind = act_kind(typ, n);
        let a = QAct::new(kind, &x);
        // the activation as the kernel sees it
        let xq: Vec<f32> = match kind {
            Act::Q8 => (0..n).map(|i| a.q8[i / 32].d * a.q8[i / 32].qs[i % 32] as f32).collect(),
            Act::Q8K => (0..n).map(|i| a.q8k[i / 256].d * a.q8k[i / 256].qs[i % 256] as f32).collect(),
            Act::F32 => x.clone(),
        };
        let mut w = vec![0.0f32; n];
        dequantize_row(typ, &row, &mut w);
        let (mut r, mut mag) = (0.0f64, 0.0f64);
        for i in 0..n {
            let t = w[i] as f64 * xq[i] as f64;
            r += t;
            mag += t.abs();
        }
        let (mut worst, mut detail) = (0.0f64, String::new());
        for l in levels() {
            simd::set_level(l);
            let v = dot_row(typ, kind, &row, &a, &x).unwrap();
            worst = worst.max((v as f64 - r).abs() / (mag + 1e-9));
            detail += &format!("{}={v:.4} ", short(l));
        }
        report(&format!("{} {:?}", type_name(typ), kind), worst < 1e-4, format!("{detail}ref={r:.4} err={worst:.1e}"), fails);
    }

    // float rows
    for typ in [T_F32, T_F16, T_BF16] {
        let x: Vec<f32> = (0..n).map(|_| rng.x()).collect();
        let vals: Vec<f32> = (0..n).map(|_| rng.x()).collect();
        let row: Vec<u8> = match typ {
            T_F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            T_F16 => vals.iter().flat_map(|v| f32_to_fp16(*v).to_le_bytes()).collect(),
            _ => vals.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect(),
        };
        let mut w = vec![0.0f32; n];
        dequantize_row(typ, &row, &mut w);
        let r: f64 = w.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let mag: f64 = w.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        let a = QAct::default();
        let mut e = 0.0f64;
        for l in levels() {
            simd::set_level(l);
            let v = dot_row(typ, Act::F32, &row, &a, &x).unwrap();
            e = e.max((v as f64 - r).abs() / (mag + 1e-9));
        }
        report(&type_name(typ), e < 1e-5, format!("all kernel sets vs ref={r:.4} err={e:.1e}"), fails);
    }
}

fn test_repack(rng: &mut Rng, fails: &mut i32) {
    for typ in [T_Q4_K, T_Q4_0] {
        let (cols, rows) = (512, 24);
        let mut data = Vec::new();
        for _ in 0..rows {
            data.extend(random_row(rng, typ, cols));
        }
        let data = leak(data);
        let mut t = Tensor {
            name: "selftest".into(),
            typ,
            n_dims: 2,
            ne: [cols as u64, rows as u64, 1, 1],
            data,
            nbytes: data.len() as u64,
            rp: None,
            gpu: None,
        };
        let x: Vec<f32> = (0..cols).map(|_| rng.x()).collect();
        let mut plain = vec![0.0f32; rows];
        matvec(&t, &x, &mut plain).unwrap();
        t.rp = repack::repack(typ, data, cols, rows);
        let mut worst = 0.0f32;
        for l in levels() {
            simd::set_level(l);
            let mut rp = vec![0.0f32; rows];
            matvec(&t, &x, &mut rp).unwrap();
            for (a, b) in plain.iter().zip(&rp) {
                worst = worst.max((a - b).abs() / (a.abs() + 1.0));
            }
        }
        report(
            &format!("{} repacked 8-row", type_name(typ)),
            t.rp.is_some() && worst < 1e-4,
            format!("vs single-row kernel err={worst:.1e}"),
            fails,
        );
    }
}

/// prompt-batch matmul (GEMM / tiles) against one matvec per token
fn test_matmul(rng: &mut Rng, fails: &mut i32) {
    let (cols, rows) = (512, 11); // odd row count: the 1-row tiles run too
    for typ in [T_Q4_K, T_Q6_K, T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1, T_Q8_0, T_Q2_K] {
        let mut data = Vec::new();
        for _ in 0..rows {
            data.extend(random_row(rng, typ, cols));
        }
        let data = leak(data);
        let t = Tensor {
            name: "selftest".into(),
            typ,
            n_dims: 2,
            ne: [cols as u64, rows as u64, 1, 1],
            data,
            nbytes: data.len() as u64,
            rp: None,
            gpu: None,
        };
        let mut worst = 0.0f32;
        for (l, nb) in levels().into_iter().flat_map(|l| [13usize, 5, 24].map(|nb| (l, nb))) {
            simd::set_level(l);
            let (ldx, ldy) = (cols + 3, rows + 2);
            let x: Vec<f32> = (0..nb * ldx).map(|_| rng.x()).collect();
            let mut y = vec![0.0f32; nb * ldy];
            crate::batch::matmul(&t, &x, ldx, &mut y, ldy, nb).unwrap();
            for b in 0..nb {
                let mut r1 = vec![0.0f32; rows];
                matvec(&t, &x[b * ldx..b * ldx + cols], &mut r1).unwrap();
                let scale = r1.iter().fold(0.0f32, |m, v| m.max(v.abs())) + 1e-6;
                for r in 0..rows {
                    worst = worst.max((y[b * ldy + r] - r1[r]).abs() / scale);
                }
            }
        }
        report(&format!("{} batch matmul", type_name(typ)), worst < 1e-4, format!("vs single kernel err={worst:.1e}"), fails);
    }
}

fn test_sampler(fails: &mut i32) {
    let n = 200_000;
    let tk = Tokenizer::empty();
    let mut s = Sampler::default();
    (s.temp, s.top_p, s.repeat_penalty, s.top_k, s.repeat_last_n, s.rng) = (1.0, 1.0, 1.0, 0, 64, 12345);
    let ninf = f32::NEG_INFINITY;
    let base = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln(), ninf, ninf, ninf, ninf, ninf];
    let mut cnt = [0usize; 8];
    for _ in 0..n {
        let mut lg = base;
        cnt[s.sample(&tk, &mut lg, &[]) as usize] += 1;
    }
    let f: Vec<f32> = cnt.iter().map(|&c| c as f32 / n as f32).collect();
    let ok = (f[0] - 0.5).abs() < 0.01 && (f[1] - 0.3).abs() < 0.01 && (f[2] - 0.2).abs() < 0.01 && cnt[0] + cnt[1] + cnt[2] == n;
    report("sampler dist", ok, format!("{:.3} {:.3} {:.3} (want .5 .3 .2)", f[0], f[1], f[2]), fails);

    // top_p 0.6: 0.5 < 0.6 so the second is added (0.8) and it stops — the third never appears
    s.top_p = 0.6;
    cnt = [0; 8];
    for _ in 0..n {
        let mut lg = base;
        cnt[s.sample(&tk, &mut lg, &[]) as usize] += 1;
    }
    let f0 = cnt[0] as f32 / n as f32;
    let ok = cnt[2] == 0 && (f0 - 0.625).abs() < 0.01;
    report("sampler top-p", ok, format!("{:.3} {:.3} {:.3} (want .625 .375 0)", f0, cnt[1] as f32 / n as f32, cnt[2] as f32 / n as f32), fails);

    // temp 0 and top_k 1 are both argmax
    let tie = [0.1f32, 2.0, 1.9, 0.0, 0.0, 0.0, 0.0, 0.0];
    s.top_p = 1.0;
    s.temp = 0.0;
    let mut lg = tie;
    let g0 = s.sample(&tk, &mut lg, &[]);
    s.temp = 1.0;
    s.top_k = 1;
    let g1 = (0..1000).all(|_| {
        let mut lg = tie;
        s.sample(&tk, &mut lg, &[]) == 1
    });
    report("sampler greedy", g0 == 1 && g1, format!("temp0={g0} top_k1={}", if g1 { "always 1" } else { "varies" }), fails);

    // penalty 1.5 on token 1 (twice in the history, applied once): 2.0/1.5 < 1.9 → token 2 wins
    s.temp = 0.0;
    s.repeat_penalty = 1.5;
    let mut lg = tie;
    let pw = s.sample(&tk, &mut lg, &[1, 5, 1]);
    let ok = pw == 2 && (lg[1] - 2.0 / 1.5).abs() < 1e-6;
    report("sampler penalty", ok, format!("winner={pw} logit={:.4}", lg[1]), fails);

    // insertion top-k = the top 10 of a full sort
    let mut r = Rng(99);
    let big: Vec<f32> = (0..1000).map(|_| (r.next() % 100_000) as f32 / 7.0).collect();
    let (mut ten, mut all) = (Vec::new(), Vec::new());
    top_k_select(&big, 10, &mut ten);
    top_k_select(&big, 0, &mut all);
    let ok = (0..10).all(|i| ten[i].v == all[i].v);
    report("top-k select", ok, "10 of 1000".into(), fails);
}

fn test_rope(rng: &mut Rng, fails: &mut i32) {
    // linear scale 0.5 at position 2p equals no scaling at p, and really differs from no scaling
    let x: Vec<f32> = (0..128).map(|_| rng.x()).collect();
    let (mut a, mut b, mut c) = (x.clone(), x.clone(), x.clone());
    ops::apply_rope(&mut a, 14, 64, 64, 10000.0, 0.5, None, true);
    ops::apply_rope(&mut b, 7, 64, 64, 10000.0, 1.0, None, true);
    ops::apply_rope(&mut c, 14, 64, 64, 10000.0, 1.0, None, true);
    let same = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    let diff = a.iter().zip(&c).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    report("rope linear scale", same < 1e-5 && diff > 0.1, format!("same={same:.1e} changed={diff:.2}"), fails);
}

fn test_kvq(rng: &mut Rng, fails: &mut i32) {
    let hd = 128;
    let x: Vec<f32> = (0..hd).map(|_| rng.x()).collect();
    let mut y = x.clone();
    kvq::rotate(&mut y);
    kvq::unrotate(&mut y);
    let rel = |a: &[f32], b: &[f32]| {
        let e: f32 = a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
        let n: f32 = b.iter().map(|q| q * q).sum();
        (e / (n + 1e-9)).sqrt()
    };
    let r = rel(&y, &x);
    report("kvq rotate roundtrip", r < 1e-5, format!("rel_err={r:.1e}"), fails);

    // rotated slice → quantized → decoded
    let mut xr = x.clone();
    kvq::rotate(&mut xr);
    let q: Vec<f32> = (0..hd).map(|_| rng.x()).collect();
    let mut qr = q.clone();
    kvq::rotate(&mut qr);
    let reference: f32 = q.iter().zip(&x).map(|(a, b)| a * b).sum();
    for (bits, lim) in [(4usize, 0.15f32), (2, 0.45), (8, 0.01)] {
        let mut rec = vec![0u8; kvq::rec(hd, bits)];
        kvq::quant_slice(&xr, &mut rec, bits);
        let mut back = vec![0.0f32; hd];
        kvq::dequant_slice(&rec, &mut back, bits);
        let e = rel(&back, &xr);
        let mut detail = format!("quant rel_err={e:.3}");
        let mut ok = e < lim;
        if bits != 2 {
            let d = kvq::dot_k(&qr, &rec, bits);
            let de = (d - reference).abs() / (q.iter().map(|v| v * v).sum::<f32>().sqrt() * x.iter().map(|v| v * v).sum::<f32>().sqrt());
            detail += &format!(" dot ref={reference:.3} quant={d:.3} err={de:.3}");
            ok &= de < lim;
        }
        report(&format!("kvq {}{bits}", if bits == 2 { "V" } else { "K" }), ok, detail, fails);
    }
}

pub fn cmd_selftest() -> i32 {
    let start = simd::level();
    println!("kernels: {} | testing: {}", simd::cpu_name(), levels().iter().map(|l| short(*l)).collect::<Vec<_>>().join(", "));
    let mut rng = Rng(42);
    let mut fails = 0;

    // fp16: the fast path against the full conversion for all 65536 values
    let bad = (0..=u16::MAX)
        .filter(|&h| {
            let (a, b) = (fp16(h), fp16_slow(h));
            !(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
        })
        .count();
    let rt = (0..=u16::MAX).filter(|&h| fp16(h).is_finite() && f32_to_fp16(fp16(h)) != h).count();
    report("fp16 table", bad == 0 && rt == 0, format!("{bad} mismatches, {rt} round-trip errors"), &mut fails);

    test_kernels(&mut rng, &mut fails);
    test_repack(&mut rng, &mut fails);
    test_matmul(&mut rng, &mut fails);
    test_sampler(&mut fails);
    test_rope(&mut rng, &mut fails);
    test_kvq(&mut rng, &mut fails);
    simd::set_level(start);

    if fails == 0 {
        println!("\nALL TESTS PASSED");
    } else {
        println!("\n{fails} TESTS FAILED");
    }
    (fails != 0) as i32
}
