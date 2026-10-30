// AVX2 / AVX-VNNI / AVX-512 VNNI kernels, the same integer math as the C version. The kernel
// code below is a template built three times with different CPU features; all three are in
// every binary and the best one the CPU supports is picked at start-up (FASTGGUF_CPU=generic |
// avx2 | avxvnni | avx512 forces one), so one binary runs well on any x86-64 CPU. The only
// difference between them is the int8 dot instruction: maddubs+madd on AVX2, one dpbusd with
// VNNI. The portable versions are in quant.rs.
#![allow(clippy::missing_safety_doc, clippy::needless_range_loop)]

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Generic = 0,
    Avx2 = 1,
    AvxVnni = 2,
    Avx512 = 3,
}

static LEVEL: AtomicU8 = AtomicU8::new(255); // 255 = not checked yet

fn from_u8(v: u8) -> Level {
    match v {
        1 => Level::Avx2,
        2 => Level::AvxVnni,
        3 => Level::Avx512,
        _ => Level::Generic,
    }
}

/// the kernel set in use
#[inline]
pub fn level() -> Level {
    match LEVEL.load(Ordering::Relaxed) {
        255 => {
            let l = detect();
            LEVEL.store(l as u8, Ordering::Relaxed);
            l
        }
        v => from_u8(v),
    }
}

/// true when the AVX2 (or better) kernels are in use
#[inline]
pub fn avx2() -> bool {
    level() >= Level::Avx2
}

/// the best kernel set this CPU supports
pub fn best() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        let f = |name: &str| match name {
            "avx2" => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c"),
            "avxvnni" => is_x86_feature_detected!("avxvnni"),
            "avx512" => {
                is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512vl")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vnni")
            }
            _ => false,
        };
        if !f("avx2") {
            Level::Generic
        } else if f("avx512") {
            Level::Avx512
        } else if f("avxvnni") {
            Level::AvxVnni
        } else {
            Level::Avx2
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Level::Generic
    }
}

fn detect() -> Level {
    let best = best();
    let want = match std::env::var("FASTGGUF_CPU").unwrap_or_default().as_str() {
        "generic" | "sse2" => Level::Generic,
        "avx2" => Level::Avx2,
        "avxvnni" => Level::AvxVnni,
        "avx512" => Level::Avx512,
        _ => best,
    };
    // never more than the CPU has (AVX-VNNI and AVX-512 VNNI are separate features)
    if supported(want) {
        want
    } else if supported(Level::Avx2) && want > Level::Avx2 {
        Level::Avx2
    } else {
        best.min(want)
    }
}

/// is this kernel set usable on this CPU?
pub fn supported(l: Level) -> bool {
    let best = best();
    match l {
        Level::Generic => true,
        Level::Avx2 => best >= Level::Avx2,
        #[cfg(target_arch = "x86_64")]
        Level::AvxVnni => best >= Level::Avx2 && is_x86_feature_detected!("avxvnni"),
        #[cfg(not(target_arch = "x86_64"))]
        Level::AvxVnni => false,
        Level::Avx512 => best == Level::Avx512,
    }
}

/// use this kernel set (self-test compares them); ignored when the CPU does not support it
pub fn set_level(l: Level) {
    if supported(l) {
        LEVEL.store(l as u8, Ordering::Relaxed);
    }
}

pub fn level_name(l: Level) -> &'static str {
    match l {
        Level::Generic => "generic (no AVX2)",
        Level::Avx2 => "AVX2",
        Level::AvxVnni => "AVX2 + AVX-VNNI",
        Level::Avx512 => "AVX-512 VNNI",
    }
}

pub fn cpu_name() -> String {
    format!("{} (runtime detected)", level_name(level()))
}

/// call the kernel of the kernel set in use: simd_call!(dot_q4_0(row, xq, nb)).
/// Only valid when avx2() is true; the call is unsafe.
macro_rules! simd_call {
    ($f:ident($($a:expr),* $(,)?)) => {
        match $crate::simd::level() {
            $crate::simd::Level::Avx512 => $crate::simd::avx512::$f($($a),*),
            $crate::simd::Level::AvxVnni => $crate::simd::avxvnni::$f($($a),*),
            _ => $crate::simd::avx2::$f($($a),*),
        }
    };
}
pub(crate) use simd_call;

#[cfg(target_arch = "x86_64")]
mod dp {
    use std::arch::x86_64::*;
    /// u8·s8 → 4 products summed into each int32 lane (no saturation for the ranges used here)
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn madd(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1)))
    }
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    pub unsafe fn avxvnni(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(acc, a, b)
    }
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    pub unsafe fn avx512(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_dpbusd_epi32(acc, a, b)
    }
}

/// the kernels, built once per kernel set. $dp: the int8 dot instruction, $nr: weight rows per
/// GEMM tile (AVX-512's 32 vector registers allow 4)
#[cfg(target_arch = "x86_64")]
macro_rules! x86_kernels {
    ($feat:literal, $dp:path, $nr:literal) => {
    /// weight rows per GEMM tile
    pub const GEMM_NR: usize = $nr;

    use crate::quant::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn ld16(p: *const u8) -> u16 {
        (p as *const u16).read_unaligned()
    }
    #[inline(always)]
    unsafe fn ld32(p: *const u8) -> i32 {
        (p as *const i32).read_unaligned()
    }
    #[inline(always)]
    unsafe fn ldu(p: *const u8) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }
    #[inline(always)]
    unsafe fn ldi8(p: *const i8) -> __m256i {
        _mm256_loadu_si256(p as *const __m256i)
    }

    /// the fp16 at p → f32 with the F16C instruction. vmovd writes the whole register, so the
    /// conversion does not depend on its old contents. (Written as an intrinsic, LLVM folds the
    /// load into a vpinsrw that merges into whatever register it picks — often an accumulator,
    /// which then chains every block to the previous one and halves the speed.)
    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn h2fp(p: *const u8) -> f32 {
        let h = (p as *const u16).read_unaligned() as u32;
        let v: __m128i;
        std::arch::asm!("vmovd {v}, {h:e}", v = lateout(xmm_reg) v, h = in(reg) h, options(pure, nomem, nostack, preserves_flags));
        _mm_cvtss_f32(_mm_cvtph_ps(v))
    }

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        let lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        hsum128_ps(lo)
    }

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn hsum128_ps(v: __m128) -> f32 {
        let sh = _mm_movehdup_ps(v);
        let s = _mm_add_ps(v, sh);
        let sh = _mm_movehl_ps(sh, s);
        _mm_cvtss_f32(_mm_add_ss(s, sh))
    }

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let lo = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let lo = _mm_hadd_epi32(lo, lo);
        let lo = _mm_hadd_epi32(lo, lo);
        _mm_cvtsi128_si32(lo)
    }

    /// 16 nibble bytes → 32 bytes [low nibbles × 16 | high nibbles × 16] (the Q4_0 layout)
    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn nibbles32(q: *const u8) -> __m256i {
        let t = _mm_loadu_si128(q as *const __m128i);
        let m = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(t, m);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(t), m);
        _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }

    /// Σ a·b over int8 pairs, in 8 int32 lanes (sign trick so maddubs can be used)
    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn mul_sum_i8(a: __m256i, b: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(a, a);
        let sy = _mm256_sign_epi8(b, a);
        $dp(_mm256_setzero_si256(), ax, sy)
    }

    /// 32 bits → 32 bytes: 0xFF where the bit is set (byte j = bit j)
    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn bytes_from_bits_32(p: *const u8) -> __m256i {
        let x32 = ld32(p);
        let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf);
        let bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe_u64 as i64));
        _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
    }

    #[inline(always)]
    pub unsafe fn prefetch(p: *const u8) {
        _mm_prefetch::<_MM_HINT_T0>(p as *const i8);
    }

    // ---- 32-value blocks against Q8 ----

    /// the 32 weights of one block as signed int8 (Q4_0 / Q8_0 / Q5_0 / IQ4_NL); typ is a
    /// constant after inlining, so the match disappears
    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn q32_vals(typ: u32, p: *const u8) -> __m256i {
        match typ {
            T_Q8_0 => ldu(p.add(2)),
            T_Q4_0 => _mm256_sub_epi8(nibbles32(p.add(2)), _mm256_set1_epi8(8)),
            // q5 - 16 as int8: high bit set → nibble, clear → nibble | 0xF0 (= nibble - 16)
            T_Q5_0 => _mm256_or_si256(
                nibbles32(p.add(6)),
                _mm256_andnot_si256(bytes_from_bits_32(p.add(2)), _mm256_set1_epi8(0xF0u8 as i8)),
            ),
            _ => {
                let ktbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i));
                _mm256_shuffle_epi8(ktbl, nibbles32(p.add(2)))
            }
        }
    }

    /// dot(32-value block row, Q8 x) with two accumulators, so consecutive blocks do not wait
    /// for each other's FMA
    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn dot_q32(typ: u32, bb: usize, row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
        let r = row.as_ptr();
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b = 0;
        while b + 2 <= nb {
            let (p0, p1) = (r.add(b * bb), r.add((b + 1) * bb));
            let (y0, y1) = (xq.get_unchecked(b), xq.get_unchecked(b + 1));
            let s0 = mul_sum_i8(q32_vals(typ, p0), ldi8(y0.qs.as_ptr()));
            let s1 = mul_sum_i8(q32_vals(typ, p1), ldi8(y1.qs.as_ptr()));
            a0 = _mm256_fmadd_ps(_mm256_set1_ps(h2fp(p0) * y0.d), _mm256_cvtepi32_ps(s0), a0);
            a1 = _mm256_fmadd_ps(_mm256_set1_ps(h2fp(p1) * y1.d), _mm256_cvtepi32_ps(s1), a1);
            b += 2;
        }
        if b < nb {
            let p = r.add(b * bb);
            let y = xq.get_unchecked(b);
            let s = mul_sum_i8(q32_vals(typ, p), ldi8(y.qs.as_ptr()));
            a0 = _mm256_fmadd_ps(_mm256_set1_ps(h2fp(p) * y.d), _mm256_cvtepi32_ps(s), a0);
        }
        hsum_ps(_mm256_add_ps(a0, a1))
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q4_0(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
        dot_q32(T_Q4_0, 18, row, xq, nb)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q8_0(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
        dot_q32(T_Q8_0, 34, row, xq, nb)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q5_0(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
        dot_q32(T_Q5_0, 22, row, xq, nb)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q5_1(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
        let r = row.as_ptr();
        let x10 = _mm256_set1_epi8(0x10);
        let mut acc = _mm256_setzero_ps();
        let mut macc = 0.0f32;
        for b in 0..nb {
            let p = r.add(b * 24);
            let y = xq.get_unchecked(b);
            let hb = _mm256_and_si256(bytes_from_bits_32(p.add(4)), x10);
            let bx = _mm256_or_si256(nibbles32(p.add(8)), hb);
            let s = mul_sum_i8(bx, ldi8(y.qs.as_ptr()));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(h2fp(p) * y.d), _mm256_cvtepi32_ps(s), acc);
            macc += h2fp(p.add(2)) * y.d * (y.s16[0] as i32 + y.s16[1] as i32) as f32;
        }
        hsum_ps(acc) + macc
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q4_1(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
        let r = row.as_ptr();
        let mut acc = _mm256_setzero_ps();
        let mut macc = 0.0f32;
        for b in 0..nb {
            let p = r.add(b * 20);
            let y = xq.get_unchecked(b);
            let s = mul_sum_i8(nibbles32(p.add(4)), ldi8(y.qs.as_ptr()));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(h2fp(p) * y.d), _mm256_cvtepi32_ps(s), acc);
            macc += h2fp(p.add(2)) * y.d * (y.s16[0] as i32 + y.s16[1] as i32) as f32;
        }
        hsum_ps(acc) + macc
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_iq4_nl(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
        dot_q32(T_IQ4_NL, 18, row, xq, nb)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_iq4_xs(row: &[u8], xq: &[Q8Blk], nb256: usize) -> f32 {
        let ktbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i));
        let mut sum = 0.0f32;
        for b in 0..nb256 {
            let ps = &row[b * 136..b * 136 + 136];
            let p = ps.as_ptr();
            let d = h2fp(p);
            let sh = ld16(p.add(2));
            for ib in 0..8 {
                let sl = *p.add(4 + ib / 2);
                let ls = (((sl >> (4 * (ib % 2))) & 0xF) as i32) | ((((sh >> (2 * ib)) & 3) as i32) << 4);
                let y = xq.get_unchecked(b * 8 + ib);
                let bx = _mm256_shuffle_epi8(ktbl, nibbles32(p.add(8 + ib * 16)));
                let s32 = hsum_i32(mul_sum_i8(bx, ldi8(y.qs.as_ptr())));
                sum += d * (ls - 32) as f32 * y.d * s32 as f32;
            }
        }
        sum
    }

    // ---- K-quants ----

    const fn make_kshuf_w() -> [[u8; 32]; 8] {
        let mut t = [[0u8; 32]; 8];
        let mut i = 0;
        while i < 8 {
            let mut k = 0;
            while k < 32 {
                t[i][k] = (2 * i + (k & 1)) as u8;
                k += 1;
            }
            i += 1;
        }
        t
    }
    const fn make_kshuf_q6() -> [[u8; 32]; 4] {
        let mut t = [[0u8; 32]; 4];
        let mut i = 0;
        while i < 4 {
            let mut k = 0;
            while k < 32 {
                t[i][k] = (4 * i + if k < 16 { 0 } else { 2 } + (k & 1)) as u8;
                k += 1;
            }
            i += 1;
        }
        t
    }
    // kshuf_w[i] puts 16-bit element i of each 128-bit lane into every word of that lane
    static KSHUF_W: [[u8; 32]; 8] = make_kshuf_w();
    // low lane ← element 2k, high lane ← element 2k+1 (Q6_K: one scale per 16 values)
    static KSHUF_Q6: [[u8; 32]; 4] = make_kshuf_q6();

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn k4_min_vec(mins16: __m128i, y: &Q8KBlk) -> __m128i {
        let bs = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256::<1>(bs));
        _mm_madd_epi16(mins16, q8s)
    }

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn q6_corr_vec(sc16: __m256i, y: &Q8KBlk) -> __m128i {
        let v = _mm256_madd_epi16(sc16, _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i));
        _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q4_k(row: &[u8], y: &[Q8KBlk], nb: usize) -> f32 {
        let r = row.as_ptr();
        let m4 = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let mut macc = _mm_setzero_ps();
        for b in 0..nb {
            let p = r.add(b * 144);
            let qs = p.add(16);
            let yb = y.get_unchecked(b);
            let q8 = yb.qs.as_ptr();
            let u = unpack_scales_k4(std::slice::from_raw_parts(p.add(4), 12));
            let ms = _mm256_cvtepu8_epi16(_mm_loadu_si128(u.as_ptr() as *const __m128i));
            let sc128 = _mm256_castsi256_si128(ms);
            let scales = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(sc128), sc128);
            let mut sumi = _mm256_setzero_si256();
            for j in 0..4 {
                let sl = _mm256_shuffle_epi8(scales, ldu(KSHUF_W[2 * j].as_ptr()));
                let sh = _mm256_shuffle_epi8(scales, ldu(KSHUF_W[2 * j + 1].as_ptr()));
                let q4 = ldu(qs.add(32 * j));
                let lo = _mm256_and_si256(q4, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(q4), m4);
                let pl = _mm256_madd_epi16(sl, _mm256_maddubs_epi16(lo, ldi8(q8.add(64 * j))));
                let ph = _mm256_madd_epi16(sh, _mm256_maddubs_epi16(hi, ldi8(q8.add(64 * j + 32))));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
            }
            let yd = yb.d;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(h2fp(p) * yd), _mm256_cvtepi32_ps(sumi), acc);
            let mins = _mm256_extracti128_si256::<1>(ms);
            macc = _mm_fmadd_ps(_mm_set1_ps(h2fp(p.add(2)) * yd), _mm_cvtepi32_ps(k4_min_vec(mins, yb)), macc);
        }
        hsum_ps(acc) - hsum128_ps(macc)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q5_k(row: &[u8], y: &[Q8KBlk], nb: usize) -> f32 {
        let r = row.as_ptr();
        let m4 = _mm256_set1_epi8(0x0F);
        let one8 = _mm256_set1_epi8(1);
        let mut acc = _mm256_setzero_ps();
        let mut macc = _mm_setzero_ps();
        for b in 0..nb {
            let p = r.add(b * 176);
            let (qh, ql) = (p.add(16), p.add(48));
            let yb = y.get_unchecked(b);
            let q8 = yb.qs.as_ptr();
            let sm = unpack_scales_k4(std::slice::from_raw_parts(p.add(4), 12));
            // 5th bit: sub-block s takes bit s of every qh byte — shift by 2 per pair
            let mut hb = ldu(qh);
            let mut sumi = _mm256_setzero_si256();
            for j in 0..4 {
                let q = ldu(ql.add(32 * j));
                let lo = _mm256_or_si256(
                    _mm256_and_si256(q, m4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(hb, one8)),
                );
                let hi = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(q), m4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<1>(hb), one8)),
                );
                hb = _mm256_srli_epi16::<2>(hb);
                let pl = _mm256_maddubs_epi16(lo, ldi8(q8.add(64 * j)));
                let ph = _mm256_maddubs_epi16(hi, ldi8(q8.add(64 * j + 32)));
                let pl = _mm256_madd_epi16(pl, _mm256_set1_epi16(sm[2 * j] as i16));
                let ph = _mm256_madd_epi16(ph, _mm256_set1_epi16(sm[2 * j + 1] as i16));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
            }
            let yd = yb.d;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(h2fp(p) * yd), _mm256_cvtepi32_ps(sumi), acc);
            let mins16 = _mm_cvtepu8_epi16(_mm_loadl_epi64(sm.as_ptr().add(8) as *const __m128i));
            macc = _mm_fmadd_ps(_mm_set1_ps(h2fp(p.add(2)) * yd), _mm_cvtepi32_ps(k4_min_vec(mins16, yb)), macc);
        }
        hsum_ps(acc) - hsum128_ps(macc)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q6_k(row: &[u8], y: &[Q8KBlk], nb: usize) -> f32 {
        let r = row.as_ptr();
        let m4 = _mm256_set1_epi8(0x0F);
        let m3 = _mm256_set1_epi8(3);
        let mut acc = _mm256_setzero_ps();
        let mut cacc = _mm_setzero_ps();
        for b in 0..nb {
            let p = r.add(b * 210);
            let yb = y.get_unchecked(b);
            let dd = h2fp(p.add(208)) * yb.d;
            let sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p.add(192) as *const __m128i));
            let mut sumi = _mm256_setzero_si256();
            for half in 0..2 {
                let ql = p.add(half * 64);
                let qh = p.add(128 + half * 32);
                let q8 = yb.qs.as_ptr().add(half * 128);
                let s8 = if half == 1 { _mm256_extracti128_si256::<1>(sc16) } else { _mm256_castsi256_si128(sc16) };
                let sv8 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(s8), s8);
                let a0 = ldu(ql);
                let a1 = ldu(ql.add(32));
                let hh = ldu(qh);
                let q = [
                    _mm256_or_si256(_mm256_and_si256(a0, m4), _mm256_slli_epi16::<4>(_mm256_and_si256(hh, m3))),
                    _mm256_or_si256(
                        _mm256_and_si256(a1, m4),
                        _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(hh), m3)),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(a0), m4),
                        _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(hh), m3)),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(a1), m4),
                        _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(hh), m3)),
                    ),
                ];
                for k in 0..4 {
                    let sv = _mm256_shuffle_epi8(sv8, ldu(KSHUF_Q6[k].as_ptr()));
                    let dt = _mm256_maddubs_epi16(q[k], ldi8(q8.add(32 * k)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dt, sv));
                }
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(sumi), acc);
            cacc = _mm_fmadd_ps(_mm_set1_ps(dd * 32.0), _mm_cvtepi32_ps(q6_corr_vec(sc16, yb)), cacc);
        }
        hsum_ps(acc) - hsum128_ps(cacc)
    }

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn two_scales(a: i32, b: i32) -> __m256i {
        let (a, b) = (a as i16, b as i16);
        _mm256_setr_epi16(a, a, a, a, a, a, a, a, b, b, b, b, b, b, b, b)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q2_k(row: &[u8], xq: &[Q8Blk], nb256: usize) -> f32 {
        let r = row.as_ptr();
        let m3 = _mm256_set1_epi8(3);
        // two vector accumulators (no horizontal sum per 32 values); the 8 min terms of a block
        // go into one more vector
        let mut acc = [_mm256_setzero_ps(); 2];
        let mut macc = _mm256_setzero_ps();
        for b in 0..nb256 {
            let p = r.add(b * 84);
            let qs = p.add(16);
            let d = h2fp(p.add(80));
            let dmin = h2fp(p.add(82));
            let (mut dma, mut ma) = ([0.0f32; 8], [0i32; 8]);
            for half in 0..2 {
                let bits = ldu(qs.add(half * 32));
                for j in 0..4 {
                    let qv = _mm256_and_si256(_mm256_srl_epi16(bits, _mm_cvtsi32_si128((2 * j) as i32)), m3);
                    let sa = *p.add(half * 8 + 2 * j);
                    let sb = *p.add(half * 8 + 2 * j + 1);
                    let xk = xq.get_unchecked(b * 8 + half * 4 + j);
                    let p16 = _mm256_maddubs_epi16(qv, ldi8(xk.qs.as_ptr()));
                    let s = _mm256_madd_epi16(p16, two_scales((sa & 0xF) as i32, (sb & 0xF) as i32));
                    acc[j & 1] = _mm256_fmadd_ps(_mm256_set1_ps(xk.d * d), _mm256_cvtepi32_ps(s), acc[j & 1]);
                    dma[half * 4 + j] = xk.d * dmin;
                    ma[half * 4 + j] = (sa >> 4) as i32 * xk.s16[0] as i32 + (sb >> 4) as i32 * xk.s16[1] as i32;
                }
            }
            macc = _mm256_fmadd_ps(
                _mm256_loadu_ps(dma.as_ptr()),
                _mm256_cvtepi32_ps(_mm256_loadu_si256(ma.as_ptr() as *const __m256i)),
                macc,
            );
        }
        hsum_ps(_mm256_sub_ps(_mm256_add_ps(acc[0], acc[1]), macc))
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q3_k(row: &[u8], xq: &[Q8Blk], nb256: usize) -> f32 {
        let m3 = _mm256_set1_epi8(3);
        let four = _mm256_set1_epi8(4);
        // two vector accumulators (no horizontal sum per 32 values); the 8 offset terms of a
        // block go into one more vector, so there is no chain of scalar additions either
        let mut acc = [_mm256_setzero_ps(); 2];
        let mut offacc = _mm256_setzero_ps();
        for b in 0..nb256 {
            let ps = &row[b * 110..b * 110 + 110];
            let p = ps.as_ptr();
            let qs = p.add(32);
            let d = h2fp(p.add(108));
            let scs = q3k_scales(ps);
            // the 16 scales − 32 as int16, once per block; per pair of 16-value groups one
            // shuffle puts scale 2j in the low lane and 2j+1 in the high lane (as in Q6_K)
            let sc16 = _mm256_sub_epi16(
                _mm256_cvtepi8_epi16(_mm_loadu_si128(scs.as_ptr() as *const __m128i)),
                _mm256_set1_epi16(32),
            );
            let hbits = ldu(p);
            let (mut dsa, mut offa) = ([0.0f32; 8], [0i32; 8]);
            for half in 0..2 {
                let bits = ldu(qs.add(half * 32));
                let s8 = if half == 1 { _mm256_extracti128_si256::<1>(sc16) } else { _mm256_castsi256_si128(sc16) };
                let sv8 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(s8), s8);
                for j in 0..4 {
                    let mv = _mm256_set1_epi8((1u8 << (half * 4 + j)) as i8);
                    let hv = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hbits, mv), mv), four);
                    let qu = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srl_epi16(bits, _mm_cvtsi32_si128((2 * j) as i32)), m3),
                        hv,
                    );
                    let sc_a = scs[half * 8 + 2 * j] as i32 - 32;
                    let sc_b = scs[half * 8 + 2 * j + 1] as i32 - 32;
                    let xk = xq.get_unchecked(b * 8 + half * 4 + j);
                    let p16 = _mm256_maddubs_epi16(qu, ldi8(xk.qs.as_ptr()));
                    let sv = _mm256_shuffle_epi8(sv8, ldu(KSHUF_Q6[j].as_ptr()));
                    let s = _mm256_madd_epi16(p16, sv);
                    let ds = d * xk.d;
                    acc[j & 1] = _mm256_fmadd_ps(_mm256_set1_ps(ds), _mm256_cvtepi32_ps(s), acc[j & 1]);
                    dsa[half * 4 + j] = ds;
                    offa[half * 4 + j] = 4 * (sc_a * xk.s16[0] as i32 + sc_b * xk.s16[1] as i32);
                }
            }
            offacc = _mm256_fmadd_ps(
                _mm256_loadu_ps(dsa.as_ptr()),
                _mm256_cvtepi32_ps(_mm256_loadu_si256(offa.as_ptr() as *const __m256i)),
                offacc,
            );
        }
        hsum_ps(_mm256_sub_ps(_mm256_add_ps(acc[0], acc[1]), offacc))
    }

    // ---- float rows ----

    /// float dot product with two 8-wide accumulators
    #[target_feature(enable = $feat)]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), s1);
            i += 16;
        }
        if i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            i += 8;
        }
        let mut s = hsum_ps(_mm256_add_ps(s0, s1));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    /// dot(f32 weight row stored as bytes, x)
    #[target_feature(enable = $feat)]
    pub unsafe fn dot_f32_row(row: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let pr = row.as_ptr() as *const f32;
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(x.as_ptr().add(i)), acc);
            i += 8;
        }
        let mut s = hsum_ps(acc);
        while i < n {
            s += pr.add(i).read_unaligned() * x[i];
            i += 1;
        }
        s
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_f16(row: &[u8], x: &[f32]) -> f32 {
        let n = x.len();
        let r = row.as_ptr();
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let w = _mm256_cvtph_ps(_mm_loadu_si128(r.add(i * 2) as *const __m128i));
            acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x.as_ptr().add(i)), acc);
            i += 8;
        }
        let mut s = hsum_ps(acc);
        while i < n {
            s += h2fp(r.add(i * 2)) * x[i];
            i += 1;
        }
        s
    }

    /// y += w · x
    #[target_feature(enable = $feat)]
    pub unsafe fn axpy(y: &mut [f32], w: f32, x: &[f32]) {
        let n = y.len().min(x.len());
        let (py, px) = (y.as_mut_ptr(), x.as_ptr());
        let vw = _mm256_set1_ps(w);
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(py.add(i), _mm256_fmadd_ps(vw, _mm256_loadu_ps(px.add(i)), _mm256_loadu_ps(py.add(i))));
            i += 8;
        }
        while i < n {
            *py.add(i) += w * *px.add(i);
            i += 1;
        }
    }

    // ---- prompt batches ----

    /// 4 inputs at once for the 32-value types Q8_0 / Q4_0 / Q5_0: each weight block is loaded
    /// and unpacked once and multiplied with 4 activation vectors (same integer math as single)
    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q32_x4(typ: u32, row: &[u8], y: [&[Q8Blk]; 4], nb: usize) -> [f32; 4] {
        let r = row.as_ptr();
        let bb = match typ {
            T_Q8_0 => 34,
            T_Q4_0 => 18,
            _ => 22,
        };
        let f0 = _mm256_set1_epi8(0xF0u8 as i8);
        let off8 = _mm256_set1_epi8(8);
        let mut acc = [_mm256_setzero_ps(); 4];
        for b in 0..nb {
            let p = r.add(b * bb);
            let bx = match typ {
                T_Q8_0 => ldu(p.add(2)),
                T_Q4_0 => _mm256_sub_epi8(nibbles32(p.add(2)), off8),
                _ => _mm256_or_si256(nibbles32(p.add(6)), _mm256_andnot_si256(bytes_from_bits_32(p.add(2)), f0)),
            };
            let ax = _mm256_sign_epi8(bx, bx);
            let d = h2fp(p);
            for t in 0..4 {
                let yb = y[t].get_unchecked(b);
                let sy = _mm256_sign_epi8(ldi8(yb.qs.as_ptr()), bx);
                let s = $dp(_mm256_setzero_si256(), ax, sy);
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * yb.d), _mm256_cvtepi32_ps(s), acc[t]);
            }
        }
        [hsum_ps(acc[0]), hsum_ps(acc[1]), hsum_ps(acc[2]), hsum_ps(acc[3])]
    }

    /// tokens per vector in the GEMM (8 int32 lanes)
    pub const GV: usize = 8;

    /// u8·s8 → 4 products summed into each int32 lane. a ≤ 63 and |b| ≤ 127, so the pairs stay
    /// below the int16 limit of maddubs (no saturation)
    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn dp_u8s8(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        $dp(acc, a, b)
    }

    /// the packed activations of one GEMM call (see batch.rs)
    pub struct Packed<'a> {
        pub pk: &'a [i8],   // per vector of GV tokens: cols × GV bytes, 4 values of a token side by side
        pub gyd: &'a [f32], // per (vector, scale group, token): activation scale
        pub gbs: &'a [f32], // per (vector, min group, token): activation sum (K-quants) or xd·Σxq (Q8)
    }

    /// one unpacked weight row block: NR rows of q bytes (stride qoff) and scales (stride soff)
    pub struct Unpacked<'a> {
        pub q: &'a [u8],
        pub wsc: &'a [f32],
        pub wmn: &'a [f32],
        pub qoff: usize,
        pub soff: usize,
    }

    /// K-quant GEMM tile: NR weight rows × NV vectors of tokens, from vector c0.
    /// Q4_K: GS = MS = 32, Q6_K: GS = MS = 16. y[tok·ldy + r + k] = result
    #[target_feature(enable = $feat)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemmk_kern<const NR: usize, const NV: usize, const GS: usize, const MS: usize>(
        w: &Unpacked,
        cols: usize,
        c0: usize,
        a: &Packed,
        y: *mut f32,
        ldy: usize,
        r: usize,
        nb: usize,
    ) {
        let (nbl, ng) = (cols / 256, cols / MS);
        let mut pv = [std::ptr::null::<i8>(); NV];
        for v in 0..NV {
            pv[v] = a.pk.as_ptr().add((c0 + v) * cols * GV);
        }
        let (q, wsc, wmn) = (w.q.as_ptr(), w.wsc.as_ptr(), w.wmn.as_ptr());
        let (gyd, gbs) = (a.gyd.as_ptr(), a.gbs.as_ptr());
        let mut f = [[_mm256_setzero_ps(); NV]; NR];
        for b in 0..nbl {
            let mut acc = [[_mm256_setzero_ps(); NV]; NR];
            for g in 0..256 / GS {
                let mut ia = [[_mm256_setzero_si256(); NV]; NR];
                let k0 = 256 * b + GS * g;
                let mut o = k0 * GV;
                for s in 0..GS / 4 {
                    let mut x = [_mm256_setzero_si256(); NV];
                    for v in 0..NV {
                        x[v] = ldi8(pv[v].add(o));
                    }
                    for k in 0..NR {
                        let wv = _mm256_set1_epi32(ld32(q.add(k * w.qoff + k0 + 4 * s)));
                        for v in 0..NV {
                            ia[k][v] = dp_u8s8(ia[k][v], wv, x[v]);
                        }
                    }
                    o += 4 * GV;
                }
                for k in 0..NR {
                    let ws = _mm256_set1_ps(*wsc.add(k * w.soff + k0 / GS));
                    for v in 0..NV {
                        acc[k][v] = _mm256_fmadd_ps(ws, _mm256_cvtepi32_ps(ia[k][v]), acc[k][v]);
                    }
                }
            }
            for g in 0..256 / MS {
                let gi = b * (256 / MS) + g;
                let mut sv = [_mm256_setzero_ps(); NV];
                for v in 0..NV {
                    sv[v] = _mm256_loadu_ps(gbs.add(((c0 + v) * ng + gi) * GV));
                }
                for k in 0..NR {
                    let wm = _mm256_set1_ps(*wmn.add(k * w.soff + gi));
                    for v in 0..NV {
                        acc[k][v] = _mm256_fnmadd_ps(wm, sv[v], acc[k][v]);
                    }
                }
            }
            for v in 0..NV {
                let yd = _mm256_loadu_ps(gyd.add(((c0 + v) * nbl + b) * GV));
                for k in 0..NR {
                    f[k][v] = _mm256_fmadd_ps(acc[k][v], yd, f[k][v]);
                }
            }
        }
        store_tile::<NR, NV>(&f, c0, y, ldy, r, nb);
    }

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn store_tile<const NR: usize, const NV: usize>(
        f: &[[__m256; NV]; NR],
        c0: usize,
        y: *mut f32,
        ldy: usize,
        r: usize,
        nb: usize,
    ) {
        for k in 0..NR {
            for v in 0..NV {
                let mut o = [0.0f32; GV];
                _mm256_storeu_ps(o.as_mut_ptr(), f[k][v]);
                for (i, val) in o.iter().enumerate() {
                    let tok = (c0 + v) * GV + i;
                    if tok < nb {
                        *y.add(tok * ldy + r + k) = *val;
                    }
                }
            }
        }
    }

    /// GEMM tile for the 32-value types: per group and token
    ///     Σ w·x = xd · (wsc · Σ q·xq) − wmn · (xd · Σ xq)
    #[target_feature(enable = $feat)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm32_kern<const NR: usize, const NV: usize>(
        w: &Unpacked,
        cols: usize,
        c0: usize,
        a: &Packed,
        y: *mut f32,
        ldy: usize,
        r: usize,
        nb: usize,
    ) {
        let ng = cols / 32;
        let mut pv = [std::ptr::null::<i8>(); NV];
        for v in 0..NV {
            pv[v] = a.pk.as_ptr().add((c0 + v) * cols * GV);
        }
        let (q, wsc, wmn) = (w.q.as_ptr(), w.wsc.as_ptr(), w.wmn.as_ptr());
        let (gyd, gbs) = (a.gyd.as_ptr(), a.gbs.as_ptr());
        let mut acc = [[_mm256_setzero_ps(); NV]; NR];
        for g in 0..ng {
            let mut ia = [[_mm256_setzero_si256(); NV]; NR];
            let mut o = g * 32 * GV;
            for s in 0..8 {
                let mut x = [_mm256_setzero_si256(); NV];
                for v in 0..NV {
                    x[v] = ldi8(pv[v].add(o));
                }
                for k in 0..NR {
                    let wv = _mm256_set1_epi32(ld32(q.add(k * w.qoff + 32 * g + 4 * s)));
                    for v in 0..NV {
                        ia[k][v] = dp_u8s8(ia[k][v], wv, x[v]);
                    }
                }
                o += 4 * GV;
            }
            for v in 0..NV {
                let gi = ((c0 + v) * ng + g) * GV;
                let yd = _mm256_loadu_ps(gyd.add(gi));
                let ys = _mm256_loadu_ps(gbs.add(gi));
                for k in 0..NR {
                    let ws = _mm256_mul_ps(_mm256_set1_ps(*wsc.add(k * w.soff + g)), yd);
                    acc[k][v] = _mm256_fmadd_ps(ws, _mm256_cvtepi32_ps(ia[k][v]), acc[k][v]);
                    acc[k][v] = _mm256_fnmadd_ps(_mm256_set1_ps(*wmn.add(k * w.soff + g)), ys, acc[k][v]);
                }
            }
        }
        store_tile::<NR, NV>(&acc, c0, y, ldy, r, nb);
    }

    // ---- repacked weights: 8 rows side by side (see repack.rs) ----

    #[inline]
    #[target_feature(enable = $feat)]
    unsafe fn dp_u8s8_x8(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        $dp(acc, a, b)
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q4kx8(blocks: &[crate::repack::Q4Kx8], y: &[Q8KBlk], bsp: &[i32], out: &mut [f32]) {
        let m4 = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let mut macc = _mm256_setzero_ps();
        for (b, bl) in blocks.iter().enumerate() {
            let base = bl as *const _ as *const u8;
            let mut c = 0;
            while c < std::mem::size_of::<crate::repack::Q4Kx8>() {
                prefetch(base.wrapping_add(2048 + c));
                c += 64;
            }
            let yb = y.get_unchecked(b);
            let q8 = yb.qs.as_ptr();
            let mut ia = [_mm256_setzero_si256(); 8];
            for s in 0..8 {
                for j in 0..4 {
                    // 8 independent chains: sub-blocks 2j (low nibbles) and 2j+1 (high)
                    let v = ldu(bl.qs[j][s].as_ptr() as *const u8);
                    let a0 = ld32(q8.add(64 * j + 4 * s) as *const u8);
                    let a1 = ld32(q8.add(64 * j + 32 + 4 * s) as *const u8);
                    ia[2 * j] = dp_u8s8_x8(ia[2 * j], _mm256_and_si256(v, m4), _mm256_set1_epi32(a0));
                    ia[2 * j + 1] = dp_u8s8_x8(
                        ia[2 * j + 1],
                        _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4),
                        _mm256_set1_epi32(a1),
                    );
                }
            }
            let mut sumi = _mm256_setzero_si256();
            let mut mi = _mm256_setzero_si256();
            for k in 0..8 {
                let sc = _mm256_cvtepu8_epi32(_mm_loadl_epi64(bl.sc[k].as_ptr() as *const __m128i));
                sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(ia[k], sc));
            }
            for p in 0..4 {
                let mn = _mm256_cvtepu8_epi16(_mm_loadu_si128(bl.mn[p].as_ptr() as *const __m128i));
                mi = _mm256_add_epi32(mi, _mm256_madd_epi16(mn, _mm256_set1_epi32(*bsp.get_unchecked(4 * b + p))));
            }
            let yd = _mm256_set1_ps(yb.d);
            let d = _mm256_cvtph_ps(_mm_loadu_si128(bl.d.as_ptr() as *const __m128i));
            let dm = _mm256_cvtph_ps(_mm_loadu_si128(bl.dmin.as_ptr() as *const __m128i));
            acc = _mm256_fmadd_ps(_mm256_mul_ps(d, yd), _mm256_cvtepi32_ps(sumi), acc);
            macc = _mm256_fmadd_ps(_mm256_mul_ps(dm, yd), _mm256_cvtepi32_ps(mi), macc);
        }
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_sub_ps(acc, macc));
    }

    #[target_feature(enable = $feat)]
    pub unsafe fn dot_q40x8(blocks: &[crate::repack::Q40x8], y: &[Q8Blk], out: &mut [f32]) {
        let m4 = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        for (b, bl) in blocks.iter().enumerate() {
            let base = bl as *const _ as *const u8;
            prefetch(base.wrapping_add(2048));
            prefetch(base.wrapping_add(2048 + 64));
            prefetch(base.wrapping_add(2048 + 128));
            let yb = y.get_unchecked(b);
            let mut ia = [_mm256_setzero_si256(); 4];
            for s in 0..4 {
                let v = ldu(bl.qs[s].as_ptr() as *const u8);
                let a0 = ld32(yb.qs.as_ptr().add(4 * s) as *const u8);
                let a1 = ld32(yb.qs.as_ptr().add(16 + 4 * s) as *const u8);
                ia[s] = dp_u8s8_x8(
                    dp_u8s8_x8(_mm256_setzero_si256(), _mm256_and_si256(v, m4), _mm256_set1_epi32(a0)),
                    _mm256_and_si256(_mm256_srli_epi16::<4>(v), m4),
                    _mm256_set1_epi32(a1),
                );
            }
            let mut isum = _mm256_add_epi32(_mm256_add_epi32(ia[0], ia[1]), _mm256_add_epi32(ia[2], ia[3]));
            isum = _mm256_sub_epi32(isum, _mm256_set1_epi32(8 * (yb.s16[0] as i32 + yb.s16[1] as i32)));
            let d = _mm256_cvtph_ps(_mm_loadu_si128(bl.d.as_ptr() as *const __m128i));
            acc = _mm256_fmadd_ps(_mm256_mul_ps(d, _mm256_set1_ps(yb.d)), _mm256_cvtepi32_ps(isum), acc);
        }
        _mm256_storeu_ps(out.as_mut_ptr(), acc);
    }

    };
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    x86_kernels!("avx2,fma,f16c", super::dp::madd, 2);
}
#[cfg(target_arch = "x86_64")]
pub mod avxvnni {
    x86_kernels!("avx2,fma,f16c,avxvnni", super::dp::avxvnni, 2);
}
#[cfg(target_arch = "x86_64")]
pub mod avx512 {
    x86_kernels!("avx2,fma,f16c,avx512f,avx512vl,avx512bw,avx512vnni", super::dp::avx512, 4);
}
