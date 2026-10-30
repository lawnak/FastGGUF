// GGML tensor types, fp16, dequantization, activation quantization and the portable
// (scalar) dot-product kernels. The AVX2 versions of the kernels live in simd.rs.

pub const T_F32: u32 = 0;
pub const T_F16: u32 = 1;
pub const T_Q4_0: u32 = 2;
pub const T_Q4_1: u32 = 3;
pub const T_Q5_0: u32 = 6;
pub const T_Q5_1: u32 = 7;
pub const T_Q8_0: u32 = 8;
pub const T_Q2_K: u32 = 10;
pub const T_Q3_K: u32 = 11;
pub const T_Q4_K: u32 = 12;
pub const T_Q5_K: u32 = 13;
pub const T_Q6_K: u32 = 14;
pub const T_IQ4_NL: u32 = 20;
pub const T_IQ4_XS: u32 = 23;
pub const T_BF16: u32 = 30;

pub fn type_name(t: u32) -> String {
    match t {
        T_F32 => "F32",
        T_F16 => "F16",
        T_BF16 => "BF16",
        T_Q4_0 => "Q4_0",
        T_Q4_1 => "Q4_1",
        T_Q5_0 => "Q5_0",
        T_Q5_1 => "Q5_1",
        T_Q8_0 => "Q8_0",
        T_Q2_K => "Q2_K",
        T_Q3_K => "Q3_K",
        T_Q4_K => "Q4_K",
        T_Q5_K => "Q5_K",
        T_Q6_K => "Q6_K",
        T_IQ4_NL => "IQ4_NL",
        T_IQ4_XS => "IQ4_XS",
        _ => return format!("TYPE_{t}"),
    }
    .to_string()
}

/// (elements per block, bytes per block)
pub fn block_size(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        T_F32 => (1, 4),
        T_F16 | T_BF16 => (1, 2),
        T_Q4_0 => (32, 18),
        T_Q4_1 => (32, 20),
        T_Q5_0 => (32, 22),
        T_Q5_1 => (32, 24),
        T_Q8_0 => (32, 34),
        T_Q2_K => (256, 84),
        T_Q3_K => (256, 110),
        T_Q4_K => (256, 144),
        T_Q5_K => (256, 176),
        T_Q6_K => (256, 210),
        T_IQ4_NL => (32, 18),
        T_IQ4_XS => (256, 136),
        _ => return None,
    })
}

pub fn tensor_nbytes(t: u32, elems: u64) -> u64 {
    match block_size(t) {
        Some((be, bb)) => elems.div_ceil(be) * bb,
        None => 0,
    }
}

pub fn row_stride(t: u32, cols: u64) -> u64 {
    tensor_nbytes(t, cols)
}

// ---------------------------------------------------------------------------
// little-endian reads and fp16 / bf16
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn rd_u16(p: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([p[o], p[o + 1]])
}
#[inline(always)]
pub fn rd_u32(p: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]])
}
#[inline(always)]
pub fn rd_f32(p: &[u8], o: usize) -> f32 {
    f32::from_bits(rd_u32(p, o))
}

/// exact fp16 → f32 for every input (subnormals, inf, NaN)
pub fn fp16_slow(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1f) as u32;
    let mut mant = (h & 0x03ff) as u32;
    let f = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            exp = exp.wrapping_add(127 - 15);
            sign | (exp << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f)
}

/// fp16 → f32: normal numbers with a few bit operations, the rest through fp16_slow
#[inline(always)]
pub fn fp16(h: u16) -> f32 {
    let exp = (h >> 10) & 0x1f;
    if exp != 0 && exp != 31 {
        f32::from_bits(
            (((h & 0x8000) as u32) << 16) | ((exp as u32 + 112) << 23) | (((h & 0x3ff) as u32) << 13),
        )
    } else {
        fp16_slow(h)
    }
}

#[inline(always)]
pub fn bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 → fp16, round to nearest even (tests and the TurboQuant self-test only)
pub fn f32_to_fp16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && (r & 1) != 0) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) != 0) {
        r += 1;
    }
    sign | r as u16
}

// non-linear value table for the IQ4 family (llama.cpp kvalues_iq4nl)
pub static KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// the 12 packed scale bytes of Q4_K/Q5_K → [0..8) six-bit scales, [8..16) six-bit mins
#[inline(always)]
pub fn unpack_scales_k4(s12: &[u8]) -> [u8; 16] {
    const K1: u32 = 0x3f3f_3f3f;
    const K2: u32 = 0x0f0f_0f0f;
    const K3: u32 = 0x0303_0303;
    let mut u = [rd_u32(s12, 0), rd_u32(s12, 4), rd_u32(s12, 8), 0];
    u[3] = ((u[2] >> 4) & K2) | (((u[1] >> 6) & K3) << 4);
    let aux = u[1] & K1;
    u[1] = (u[2] & K2) | (((u[0] >> 6) & K3) << 4);
    u[2] = aux;
    u[0] &= K1;
    let mut out = [0u8; 16];
    for i in 0..4 {
        out[4 * i..4 * i + 4].copy_from_slice(&u[i].to_le_bytes());
    }
    out
}

/// the 16 signed six-bit scales of a Q3_K block (before the -32 offset)
#[inline(always)]
pub fn q3k_scales(p: &[u8]) -> [i8; 16] {
    let a0 = rd_u32(p, 96);
    let a1 = rd_u32(p, 100);
    let tmp = rd_u32(p, 104);
    let aux = [
        (a0 & 0x0f0f_0f0f) | ((tmp & 0x0303_0303) << 4),
        (a1 & 0x0f0f_0f0f) | (((tmp >> 2) & 0x0303_0303) << 4),
        ((a0 >> 4) & 0x0f0f_0f0f) | (((tmp >> 4) & 0x0303_0303) << 4),
        ((a1 >> 4) & 0x0f0f_0f0f) | (((tmp >> 6) & 0x0303_0303) << 4),
    ];
    let mut s = [0i8; 16];
    for k in 0..4 {
        let b = aux[k].to_le_bytes();
        for i in 0..4 {
            s[4 * k + i] = b[i] as i8;
        }
    }
    s
}

#[inline(always)]
fn iq4xs_scale(p: &[u8], ib: usize) -> i32 {
    let sh = rd_u16(p, 2);
    let sl = p[4 + ib / 2];
    (((sl >> (4 * (ib % 2))) & 0xF) as i32) | ((((sh >> (2 * ib)) & 3) as i32) << 4)
}

// ---------------------------------------------------------------------------
// Dequantization — one row of n values (n = out.len())
// ---------------------------------------------------------------------------

pub fn dequantize_row(typ: u32, raw: &[u8], out: &mut [f32]) -> bool {
    let n = out.len();
    match typ {
        T_F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = rd_f32(raw, i * 4);
            }
        }
        T_F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = fp16(rd_u16(raw, i * 2));
            }
        }
        T_BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16(rd_u16(raw, i * 2));
            }
        }
        T_Q4_0 | T_Q4_1 | T_IQ4_NL => {
            let bb = if typ == T_Q4_1 { 20 } else { 18 };
            for b in 0..n.div_ceil(32) {
                let p = &raw[b * bb..];
                let d = fp16(rd_u16(p, 0));
                let m = if typ == T_Q4_1 { fp16(rd_u16(p, 2)) } else { 0.0 };
                let qs = &p[bb - 16..];
                let base = b * 32;
                for j in 0..16 {
                    let (lo, hi) = (qs[j] & 0x0f, qs[j] >> 4);
                    let (v0, v1) = match typ {
                        T_Q4_0 => (d * (lo as i32 - 8) as f32, d * (hi as i32 - 8) as f32),
                        T_Q4_1 => (d * lo as f32 + m, d * hi as f32 + m),
                        _ => (d * KVALUES_IQ4NL[lo as usize] as f32, d * KVALUES_IQ4NL[hi as usize] as f32),
                    };
                    if base + j < n {
                        out[base + j] = v0;
                    }
                    if base + j + 16 < n {
                        out[base + j + 16] = v1;
                    }
                }
            }
        }
        T_Q5_0 | T_Q5_1 => {
            let one = typ == T_Q5_1;
            let bb = if one { 24 } else { 22 };
            for b in 0..n.div_ceil(32) {
                let p = &raw[b * bb..];
                let d = fp16(rd_u16(p, 0));
                let m = if one { fp16(rd_u16(p, 2)) } else { 0.0 };
                let qh = rd_u32(p, if one { 4 } else { 2 });
                let qs = &p[if one { 8 } else { 6 }..];
                let base = b * 32;
                for j in 0..16 {
                    let xh0 = (((qh >> j) << 4) & 0x10) as u8;
                    let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
                    let q0 = ((qs[j] & 0x0f) | xh0) as i32;
                    let q1 = ((qs[j] >> 4) | xh1) as i32;
                    let (v0, v1) = if one {
                        (d * q0 as f32 + m, d * q1 as f32 + m)
                    } else {
                        (d * (q0 - 16) as f32, d * (q1 - 16) as f32)
                    };
                    if base + j < n {
                        out[base + j] = v0;
                    }
                    if base + j + 16 < n {
                        out[base + j + 16] = v1;
                    }
                }
            }
        }
        T_Q8_0 => {
            for b in 0..n.div_ceil(32) {
                let p = &raw[b * 34..];
                let d = fp16(rd_u16(p, 0));
                let base = b * 32;
                for j in 0..32 {
                    if base + j >= n {
                        break;
                    }
                    out[base + j] = d * (p[2 + j] as i8) as f32;
                }
            }
        }
        T_Q2_K => {
            for b in 0..n.div_ceil(256) {
                let p = &raw[b * 84..];
                let (sc, qs) = (&p[..16], &p[16..80]);
                let d = fp16(rd_u16(p, 80));
                let dmin = fp16(rd_u16(p, 82));
                let mut is = 0;
                for half in 0..2 {
                    let q = &qs[half * 32..];
                    for j in 0..4 {
                        for part in 0..2 {
                            let s = sc[is];
                            is += 1;
                            let dl = d * (s & 0xF) as f32;
                            let ml = dmin * (s >> 4) as f32;
                            let base = b * 256 + half * 128 + j * 32 + part * 16;
                            for l in 0..16 {
                                if base + l < n {
                                    out[base + l] = dl * ((q[part * 16 + l] >> (2 * j)) & 3) as f32 - ml;
                                }
                            }
                        }
                    }
                }
            }
        }
        T_Q3_K => {
            for b in 0..n.div_ceil(256) {
                let p = &raw[b * 110..];
                let (hm, qs) = (&p[..32], &p[32..96]);
                let d = fp16(rd_u16(p, 108));
                let scs = q3k_scales(p);
                let mut is = 0;
                for half in 0..2 {
                    let q = &qs[half * 32..];
                    for j in 0..4 {
                        let m = 1u8 << (half * 4 + j);
                        for part in 0..2 {
                            let dl = d * (scs[is] as i32 - 32) as f32;
                            is += 1;
                            let base = b * 256 + half * 128 + j * 32 + part * 16;
                            for l in 0..16 {
                                if base + l >= n {
                                    continue;
                                }
                                let lo = ((q[part * 16 + l] >> (2 * j)) & 3) as i32;
                                let hi = if hm[part * 16 + l] & m != 0 { 0 } else { 4 };
                                out[base + l] = dl * (lo - hi) as f32;
                            }
                        }
                    }
                }
            }
        }
        T_Q4_K | T_Q5_K => {
            let five = typ == T_Q5_K;
            let bb = if five { 176 } else { 144 };
            for b in 0..n.div_ceil(256) {
                let p = &raw[b * bb..];
                let d = fp16(rd_u16(p, 0));
                let dmin = fp16(rd_u16(p, 2));
                let sc = &p[4..16];
                let qh = &p[16..48];
                let ql = &p[if five { 48 } else { 16 }..];
                let y = b * 256;
                let (mut is, mut qi) = (0, 0);
                let (mut u1, mut u2) = (1u8, 2u8);
                for j in (0..256).step_by(64) {
                    let (s1, m1) = scale_min_k4(is, sc);
                    let (s2, m2) = scale_min_k4(is + 1, sc);
                    let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
                    let (d2, mm2) = (d * s2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        let (h1, h2) = if five {
                            (
                                if qh[l] & u1 != 0 { 16.0 } else { 0.0 },
                                if qh[l] & u2 != 0 { 16.0 } else { 0.0 },
                            )
                        } else {
                            (0.0, 0.0)
                        };
                        if y + j + l < n {
                            out[y + j + l] = d1 * ((ql[qi + l] & 0xF) as f32 + h1) - mm1;
                        }
                        if y + j + l + 32 < n {
                            out[y + j + l + 32] = d2 * ((ql[qi + l] >> 4) as f32 + h2) - mm2;
                        }
                    }
                    qi += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        T_Q6_K => {
            for b in 0..n.div_ceil(256) {
                let p = &raw[b * 210..];
                let (ql, qh) = (&p[..128], &p[128..192]);
                let d = fp16(rd_u16(p, 208));
                let sc = |i: usize| p[192 + i] as i8 as f32;
                let y = b * 256;
                for half in 0..2 {
                    let (yo, qlo, qho, so) = (y + half * 128, half * 64, half * 32, half * 8);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = (ql[qlo + l] & 0xF) as i32 | (((qh[qho + l] & 3) as i32) << 4);
                        let q2 = (ql[qlo + l + 32] & 0xF) as i32 | ((((qh[qho + l] >> 2) & 3) as i32) << 4);
                        let q3 = (ql[qlo + l] >> 4) as i32 | ((((qh[qho + l] >> 4) & 3) as i32) << 4);
                        let q4 = (ql[qlo + l + 32] >> 4) as i32 | ((((qh[qho + l] >> 6) & 3) as i32) << 4);
                        let vals = [
                            d * sc(so + is) * (q1 - 32) as f32,
                            d * sc(so + is + 2) * (q2 - 32) as f32,
                            d * sc(so + is + 4) * (q3 - 32) as f32,
                            d * sc(so + is + 6) * (q4 - 32) as f32,
                        ];
                        for (k, v) in vals.into_iter().enumerate() {
                            if yo + l + 32 * k < n {
                                out[yo + l + 32 * k] = v;
                            }
                        }
                    }
                }
            }
        }
        T_IQ4_XS => {
            for b in 0..n.div_ceil(256) {
                let p = &raw[b * 136..];
                let d = fp16(rd_u16(p, 0));
                let qs = &p[8..];
                for ib in 0..8 {
                    let dl = d * (iq4xs_scale(p, ib) - 32) as f32;
                    let q = &qs[ib * 16..];
                    let base = b * 256 + ib * 32;
                    for j in 0..16 {
                        if base + j < n {
                            out[base + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
                        }
                        if base + j + 16 < n {
                            out[base + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
                        }
                    }
                }
            }
        }
        _ => return false,
    }
    true
}

// ---------------------------------------------------------------------------
// Activation quantization: x is quantized once per MatVec and every row reuses it
// ---------------------------------------------------------------------------

/// Q8 activation block: scale + 32 int8 values + the sum of each 16
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Q8Blk {
    pub d: f32,
    pub qs: [i8; 32],
    pub s16: [i16; 2],
}

/// Q8_K activation block for K-quants: one scale per 256 values + the sum of each 16
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Q8KBlk {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

impl Default for Q8Blk {
    fn default() -> Self {
        Q8Blk { d: 0.0, qs: [0; 32], s16: [0; 2] }
    }
}
impl Default for Q8KBlk {
    fn default() -> Self {
        Q8KBlk { d: 0.0, qs: [0; 256], bsums: [0; 16] }
    }
}

/// round to the nearest integer, ties to even (lrintf). On x86-64 one cvtss2si instruction —
/// round_ties_even() would be a library call per value on the baseline target
#[inline(always)]
pub fn rnd(v: f32) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of x86-64; MXCSR is in its default round-to-nearest-even mode
        unsafe { std::arch::x86_64::_mm_cvtss_si32(std::arch::x86_64::_mm_set_ss(v)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        v.round_ties_even() as i32
    }
}

pub fn quantize_q8(x: &[f32], out: &mut [Q8Blk]) {
    for (b, o) in out.iter_mut().enumerate().take(x.len() / 32) {
        let xb = &x[b * 32..b * 32 + 32];
        let amax = xb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        o.d = d;
        let (mut s0, mut s1) = (0i32, 0i32);
        for j in 0..32 {
            let q = rnd(xb[j] * id) as i8;
            o.qs[j] = q;
            if j < 16 {
                s0 += q as i32
            } else {
                s1 += q as i32
            }
        }
        o.s16 = [s0 as i16, s1 as i16];
    }
}

pub fn quantize_q8k(x: &[f32], out: &mut [Q8KBlk]) {
    for (b, o) in out.iter_mut().enumerate().take(x.len() / 256) {
        let xb = &x[b * 256..b * 256 + 256];
        let amax = xb.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        o.d = d;
        for g in 0..16 {
            let mut s = 0i32;
            for j in 0..16 {
                let v = rnd(xb[16 * g + j] * id);
                o.qs[16 * g + j] = v as i8;
                s += v;
            }
            o.bsums[g] = s as i16;
        }
    }
}

// ---------------------------------------------------------------------------
// Portable dot kernels: dot(weight row, quantized x). nb = blocks of the activation
// type (32 values for Q8, 256 for Q8_K); nb256 = 256-value weight blocks.
// ---------------------------------------------------------------------------

#[inline(always)]
fn q5_elem(qs: &[u8], qh: u32, j: usize) -> i32 {
    let nib = if j < 16 { qs[j] & 0x0F } else { qs[j - 16] >> 4 } as i32;
    nib | ((((qh >> j) & 1) as i32) << 4)
}

pub fn dot_q4_0(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 18..];
        let y = &xq[b];
        let mut si = 0i32;
        for j in 0..16 {
            si += ((p[2 + j] & 0x0F) as i32 - 8) * y.qs[j] as i32;
            si += ((p[2 + j] >> 4) as i32 - 8) * y.qs[j + 16] as i32;
        }
        sum += fp16(rd_u16(p, 0)) * y.d * si as f32;
    }
    sum
}

pub fn dot_q8_0(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 34..];
        let y = &xq[b];
        let mut si = 0i32;
        for j in 0..32 {
            si += (p[2 + j] as i8 as i32) * y.qs[j] as i32;
        }
        sum += fp16(rd_u16(p, 0)) * y.d * si as f32;
    }
    sum
}

pub fn dot_q4_1(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 20..];
        let y = &xq[b];
        let mut si = 0i32;
        for j in 0..16 {
            si += (p[4 + j] & 0x0F) as i32 * y.qs[j] as i32;
            si += (p[4 + j] >> 4) as i32 * y.qs[j + 16] as i32;
        }
        sum += fp16(rd_u16(p, 0)) * y.d * si as f32
            + fp16(rd_u16(p, 2)) * y.d * (y.s16[0] as i32 + y.s16[1] as i32) as f32;
    }
    sum
}

pub fn dot_q5_0(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 22..];
        let y = &xq[b];
        let qh = rd_u32(p, 2);
        let mut si = 0i32;
        for j in 0..32 {
            si += (q5_elem(&p[6..], qh, j) - 16) * y.qs[j] as i32;
        }
        sum += fp16(rd_u16(p, 0)) * y.d * si as f32;
    }
    sum
}

pub fn dot_q5_1(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 24..];
        let y = &xq[b];
        let qh = rd_u32(p, 4);
        let mut si = 0i32;
        for j in 0..32 {
            si += q5_elem(&p[8..], qh, j) * y.qs[j] as i32;
        }
        sum += fp16(rd_u16(p, 0)) * y.d * si as f32
            + fp16(rd_u16(p, 2)) * y.d * (y.s16[0] as i32 + y.s16[1] as i32) as f32;
    }
    sum
}

fn k4_min_sum(sm: &[u8; 16], y: &Q8KBlk) -> i32 {
    (0..8).map(|j| sm[8 + j] as i32 * (y.bsums[2 * j] as i32 + y.bsums[2 * j + 1] as i32)).sum()
}

pub fn dot_q4_k(row: &[u8], y: &[Q8KBlk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 144..];
        let qs = &p[16..];
        let q8 = &y[b].qs;
        let sm = unpack_scales_k4(&p[4..]);
        let mut sumi = 0i32;
        for j in 0..4 {
            let (mut i1, mut i2) = (0i32, 0i32);
            for l in 0..32 {
                i1 += (qs[32 * j + l] & 0xF) as i32 * q8[64 * j + l] as i32;
                i2 += (qs[32 * j + l] >> 4) as i32 * q8[64 * j + 32 + l] as i32;
            }
            sumi += sm[2 * j] as i32 * i1 + sm[2 * j + 1] as i32 * i2;
        }
        let yd = y[b].d;
        sum += fp16(rd_u16(p, 0)) * yd * sumi as f32;
        sum -= fp16(rd_u16(p, 2)) * yd * k4_min_sum(&sm, &y[b]) as f32;
    }
    sum
}

pub fn dot_q5_k(row: &[u8], y: &[Q8KBlk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 176..];
        let (qh, ql) = (&p[16..48], &p[48..]);
        let q8 = &y[b].qs;
        let sm = unpack_scales_k4(&p[4..]);
        let mut sumi = 0i32;
        for j in 0..4 {
            let (mut i1, mut i2) = (0i32, 0i32);
            for l in 0..32 {
                let h = (qh[l] >> (2 * j)) as i32;
                i1 += ((ql[32 * j + l] & 0xF) as i32 | ((h & 1) << 4)) * q8[64 * j + l] as i32;
                i2 += ((ql[32 * j + l] >> 4) as i32 | ((h & 2) << 3)) * q8[64 * j + 32 + l] as i32;
            }
            sumi += sm[2 * j] as i32 * i1 + sm[2 * j + 1] as i32 * i2;
        }
        let yd = y[b].d;
        sum += fp16(rd_u16(p, 0)) * yd * sumi as f32;
        sum -= fp16(rd_u16(p, 2)) * yd * k4_min_sum(&sm, &y[b]) as f32;
    }
    sum
}

pub fn dot_q6_k(row: &[u8], y: &[Q8KBlk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 210..];
        let sc = |i: usize| p[192 + i] as i8 as i32;
        let dd = fp16(rd_u16(p, 208)) * y[b].d;
        let corr: i32 = (0..16).map(|g| sc(g) * y[b].bsums[g] as i32).sum();
        let mut sumi = 0i32;
        for half in 0..2 {
            let (ql, qh) = (&p[half * 64..], &p[128 + half * 32..]);
            let q8 = &y[b].qs[half * 128..];
            for k in 0..4 {
                let lq = &ql[(k & 1) * 32..];
                let (lsh, hsh) = ((k >> 1) * 4, 2 * k);
                let (mut i0, mut i1) = (0i32, 0i32);
                for l in 0..32 {
                    let q = ((lq[l] >> lsh) & 0xF) as i32 | ((((qh[l] >> hsh) & 3) as i32) << 4);
                    if l < 16 {
                        i0 += q * q8[32 * k + l] as i32
                    } else {
                        i1 += q * q8[32 * k + l] as i32
                    }
                }
                sumi += sc(half * 8 + 2 * k) * i0 + sc(half * 8 + 2 * k + 1) * i1;
            }
        }
        sum += dd * (sumi - 32 * corr) as f32;
    }
    sum
}

pub fn dot_q2_k(row: &[u8], xq: &[Q8Blk], nb256: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb256 {
        let p = &row[b * 84..];
        let (sc, qs) = (&p[..16], &p[16..80]);
        let d = fp16(rd_u16(p, 80));
        let dmin = fp16(rd_u16(p, 82));
        for half in 0..2 {
            let q = &qs[half * 32..];
            for j in 0..4 {
                let (sa, sb) = (sc[half * 8 + 2 * j], sc[half * 8 + 2 * j + 1]);
                let (sc_a, m_a, sc_b, m_b) = ((sa & 0xF) as i32, (sa >> 4) as i32, (sb & 0xF) as i32, (sb >> 4) as i32);
                let xk = &xq[b * 8 + half * 4 + j];
                let mut s = 0i32;
                for l in 0..32 {
                    let qv = ((q[l] >> (2 * j)) & 3) as i32;
                    s += qv * if l < 16 { sc_a } else { sc_b } * xk.qs[l] as i32;
                }
                let m = m_a * xk.s16[0] as i32 + m_b * xk.s16[1] as i32;
                sum += xk.d * (d * s as f32 - dmin * m as f32);
            }
        }
    }
    sum
}

pub fn dot_q3_k(row: &[u8], xq: &[Q8Blk], nb256: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb256 {
        let p = &row[b * 110..];
        let (hm, qs) = (&p[..32], &p[32..96]);
        let d = fp16(rd_u16(p, 108));
        let scs = q3k_scales(p);
        for half in 0..2 {
            let q = &qs[half * 32..];
            for j in 0..4 {
                let m = 1u8 << (half * 4 + j);
                let sc_a = scs[half * 8 + 2 * j] as i32 - 32;
                let sc_b = scs[half * 8 + 2 * j + 1] as i32 - 32;
                let xk = &xq[b * 8 + half * 4 + j];
                let mut s = 0i32;
                for l in 0..32 {
                    let qu = ((q[l] >> (2 * j)) & 3) as i32 | if hm[l] & m != 0 { 4 } else { 0 };
                    s += qu * if l < 16 { sc_a } else { sc_b } * xk.qs[l] as i32;
                }
                let off = 4 * (sc_a * xk.s16[0] as i32 + sc_b * xk.s16[1] as i32);
                sum += d * xk.d * (s - off) as f32;
            }
        }
    }
    sum
}

pub fn dot_iq4_nl(row: &[u8], xq: &[Q8Blk], nb: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb {
        let p = &row[b * 18..];
        let y = &xq[b];
        let mut si = 0i32;
        for j in 0..16 {
            si += KVALUES_IQ4NL[(p[2 + j] & 0xF) as usize] as i32 * y.qs[j] as i32;
            si += KVALUES_IQ4NL[(p[2 + j] >> 4) as usize] as i32 * y.qs[j + 16] as i32;
        }
        sum += fp16(rd_u16(p, 0)) * y.d * si as f32;
    }
    sum
}

pub fn dot_iq4_xs(row: &[u8], xq: &[Q8Blk], nb256: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..nb256 {
        let p = &row[b * 136..];
        let d = fp16(rd_u16(p, 0));
        let qs = &p[8..];
        for ib in 0..8 {
            let y = &xq[b * 8 + ib];
            let q = &qs[ib * 16..];
            let mut si = 0i32;
            for j in 0..16 {
                si += KVALUES_IQ4NL[(q[j] & 0xF) as usize] as i32 * y.qs[j] as i32;
                si += KVALUES_IQ4NL[(q[j] >> 4) as usize] as i32 * y.qs[j + 16] as i32;
            }
            sum += d * (iq4xs_scale(p, ib) - 32) as f32 * y.d * si as f32;
        }
    }
    sum
}

/// float dot product with four independent sums (the compiler vectorizes this)
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0;
    while i + 4 <= n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        s0 += a[i] * b[i];
        i += 1;
    }
    (s0 + s1) + (s2 + s3)
}

pub fn dot_f16(row: &[u8], x: &[f32]) -> f32 {
    x.iter().enumerate().map(|(i, v)| fp16(rd_u16(row, i * 2)) * v).sum()
}

pub fn dot_bf16(row: &[u8], x: &[f32]) -> f32 {
    x.iter().enumerate().map(|(i, v)| bf16(rd_u16(row, i * 2)) * v).sum()
}

pub fn dot_f32_row(row: &[u8], x: &[f32]) -> f32 {
    x.iter().enumerate().map(|(i, v)| rd_f32(row, i * 4) * v).sum()
}
