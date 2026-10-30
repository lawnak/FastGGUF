// Basic ops: RMSNorm / GELU / SiLU / RoPE / softmax, the RNG, and the TurboQuant KV cache.

/// note: Gemma norm weights in GGUF already include the (+1) — multiply directly
pub fn rmsnorm_w(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
    for (v, w) in x.iter_mut().zip(w) {
        *v = *v * inv * w;
    }
}

pub fn rmsnorm_plain(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RoPE on the first n_rot dims of each head of x (length = heads × head_dim)
///   neox: halves (Gemma/Qwen), otherwise adjacent pairs (Llama/Mistral)
///   freq_scale: linear position scaling (1/factor) — large Gemma 3 models use it on global layers
#[allow(clippy::too_many_arguments)]
pub fn apply_rope(
    x: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_rot: usize,
    base: f64,
    freq_scale: f64,
    freq_factors: Option<&[f32]>,
    neox: bool,
) {
    let half = (n_rot / 2).min(512);
    // the angles depend only on the position and the pair index: compute them once
    let mut cs = [0.0f32; 512];
    let mut sn = [0.0f32; 512];
    for j in 0..half {
        let mut theta = pos as f64 * freq_scale / base.powf((2 * j) as f64 / n_rot as f64);
        if let Some(ff) = freq_factors {
            theta /= ff[j] as f64;
        }
        cs[j] = theta.cos() as f32;
        sn[j] = theta.sin() as f32;
    }
    for h in x.chunks_exact_mut(head_dim) {
        for j in 0..half {
            let (i0, i1) = if neox { (j, j + half) } else { (2 * j, 2 * j + 1) };
            let (v0, v1) = (h[i0], h[i1]);
            h[i0] = v0 * cs[j] - v1 * sn[j];
            h[i1] = v0 * sn[j] + v1 * cs[j];
        }
    }
}

pub fn softmax(s: &mut [f32]) {
    let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in s.iter_mut() {
        *v = (*v - mx).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in s.iter_mut() {
            *v *= inv;
        }
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// xorshift64*
pub fn rng_next(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// [0,1) with 24-bit precision
pub fn rng_uniform(s: &mut u64) -> f32 {
    (rng_next(s) >> 40) as f32 / 16_777_216.0
}

// ===========================================================================
// TurboQuant — quantized KV cache (arXiv 2504.19874): a random Hadamard rotation makes the
// coordinates near-Gaussian, then an optimal Lloyd-Max quantizer per coordinate.
// Keys 4-bit (or 8-bit), values 4-bit (or 2-bit), one f32 scale per head slice.
// Enable: FASTGGUF_KV=turbo — needs head_dim to be a power of 2.
// ===========================================================================

pub mod kvq {
    use std::sync::OnceLock;

    pub const MAX_HD: usize = 1024;

    // Lloyd-Max centroids for N(0,1)
    pub static C4: [f32; 16] = [
        -2.7326, -2.0690, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284, 0.1284, 0.3881, 0.6568, 0.9424,
        1.2562, 1.6181, 2.0690, 2.7326,
    ];
    pub static C2: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

    struct Rot {
        sign: [f32; MAX_HD],
        norot: bool,
    }

    fn rot() -> &'static Rot {
        static R: OnceLock<Rot> = OnceLock::new();
        R.get_or_init(|| {
            // fixed seed: always the same rotation
            let mut s = 0x243F_6A88_85A3_08D3u64;
            let mut sign = [0.0f32; MAX_HD];
            for v in sign.iter_mut() {
                *v = if super::rng_next(&mut s) >> 63 != 0 { 1.0 } else { -1.0 };
            }
            let norot = std::env::var("FASTGGUF_KV_NOROT").map(|v| v.starts_with('1')).unwrap_or(false);
            Rot { sign, norot }
        })
    }

    /// bytes of one head slice: f32 scale + packed codes
    pub const fn rec(hd: usize, bits: usize) -> usize {
        4 + hd * bits / 8
    }

    /// in-place Walsh-Hadamard transform divided by sqrt(n) (n power of 2) — orthogonal
    pub fn fwht(x: &mut [f32]) {
        let n = x.len();
        let mut len = 1;
        while len < n {
            let mut i = 0;
            while i < n {
                for j in i..i + len {
                    let (a, b) = (x[j], x[j + len]);
                    x[j] = a + b;
                    x[j + len] = a - b;
                }
                i += len << 1;
            }
            len <<= 1;
        }
        let inv = 1.0 / (n as f32).sqrt();
        for v in x.iter_mut() {
            *v *= inv;
        }
    }

    /// R·x = H(D·x)/sqrt(n)
    pub fn rotate(x: &mut [f32]) {
        let r = rot();
        if r.norot {
            return;
        }
        for (v, s) in x.iter_mut().zip(r.sign.iter()) {
            *v *= s;
        }
        fwht(x);
    }

    /// R⁻¹·y = D·(H(y)/sqrt(n))
    pub fn unrotate(x: &mut [f32]) {
        let r = rot();
        if r.norot {
            return;
        }
        fwht(x);
        for (v, s) in x.iter_mut().zip(r.sign.iter()) {
            *v *= s;
        }
    }

    fn enc4(u: f32) -> u8 {
        let mut c = 0;
        while c < 15 && u > (C4[c] + C4[c + 1]) * 0.5 {
            c += 1;
        }
        c as u8
    }
    fn enc2(u: f32) -> u8 {
        let mut c = 0;
        while c < 3 && u > (C2[c] + C2[c + 1]) * 0.5 {
            c += 1;
        }
        c as u8
    }

    /// quantize a rotated head slice: bits 8 = linear int8 (absmax), 4/2 = RMS scale + centroid
    pub fn quant_slice(x: &[f32], rec: &mut [u8], bits: usize) {
        let hd = x.len();
        if bits == 8 {
            let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let s = amax / 127.0;
            rec[..4].copy_from_slice(&s.to_le_bytes());
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for i in 0..hd {
                rec[4 + i] = (x[i] * inv).round_ties_even() as i8 as u8;
            }
            return;
        }
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let s = (ss / hd as f32).sqrt();
        rec[..4].copy_from_slice(&s.to_le_bytes());
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        let q = &mut rec[4..];
        if bits == 4 {
            for i in (0..hd).step_by(2) {
                q[i >> 1] = enc4(x[i] * inv) | (enc4(x[i + 1] * inv) << 4);
            }
        } else {
            for i in (0..hd).step_by(4) {
                q[i >> 2] = enc2(x[i] * inv)
                    | (enc2(x[i + 1] * inv) << 2)
                    | (enc2(x[i + 2] * inv) << 4)
                    | (enc2(x[i + 3] * inv) << 6);
            }
        }
    }

    fn scale(rec: &[u8]) -> f32 {
        f32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]])
    }

    /// q · k̂ for a quantized key slice (bits 8 or 4)
    pub fn dot_k(qh: &[f32], rec: &[u8], bits: usize) -> f32 {
        let s = scale(rec);
        let q = &rec[4..];
        let mut acc = 0.0f32;
        if bits == 8 {
            for (i, v) in qh.iter().enumerate() {
                acc += v * q[i] as i8 as f32;
            }
        } else {
            for i in (0..qh.len()).step_by(2) {
                let b = q[i >> 1];
                acc += qh[i] * C4[(b & 0xF) as usize] + qh[i + 1] * C4[(b >> 4) as usize];
            }
        }
        acc * s
    }

    /// oh += w · v̂ for a quantized value slice (bits 8, 4 or 2)
    pub fn axpy_v(oh: &mut [f32], w: f32, rec: &[u8], bits: usize) {
        let ws = w * scale(rec);
        let q = &rec[4..];
        let hd = oh.len();
        match bits {
            8 => {
                for i in 0..hd {
                    oh[i] += ws * q[i] as i8 as f32;
                }
            }
            4 => {
                for i in (0..hd).step_by(2) {
                    let b = q[i >> 1];
                    oh[i] += ws * C4[(b & 0xF) as usize];
                    oh[i + 1] += ws * C4[(b >> 4) as usize];
                }
            }
            _ => {
                for i in (0..hd).step_by(4) {
                    let b = q[i >> 2];
                    oh[i] += ws * C2[(b & 3) as usize];
                    oh[i + 1] += ws * C2[((b >> 2) & 3) as usize];
                    oh[i + 2] += ws * C2[((b >> 4) & 3) as usize];
                    oh[i + 3] += ws * C2[(b >> 6) as usize];
                }
            }
        }
    }

    /// decode a whole slice (self-test)
    pub fn dequant_slice(rec: &[u8], out: &mut [f32], bits: usize) {
        out.iter_mut().for_each(|v| *v = 0.0);
        axpy_v(out, 1.0, rec, bits);
    }
}
