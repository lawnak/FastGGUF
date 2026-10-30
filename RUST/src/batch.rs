// Prompt batches: Y = W · X for up to FG_BATCH tokens, so every weight row is read from memory
// once per batch instead of once per token.
//   Q4_K / Q6_K and Q4_0 / Q4_1 / Q5_0 / Q5_1 (+ Q8_0 with VNNI) with ≥ 8 tokens: int8 GEMM (each
//   row unpacked once into plain bytes + float scales, 2 or 4 rows × 2 vectors of 8 tokens per tile)
//   everything else: row blocks × token groups that stay in L1, with the single-token kernels
// Summation order differs from the single-token kernels: results match up to float rounding.

use crate::gguf::Tensor;
use crate::matvec::{act_kind, dot_f32, dot_row, matvec, Act, QAct};
use crate::quant::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

/// a raw output pointer shared by the threads; each (token, row) cell is written by one thread only
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

thread_local! {
    static ROWBUF: RefCell<(Vec<u8>, Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    static DEQ: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// y[b·ldy + r] = dot(W[r], x[b·ldx ..]) for b < nb
pub fn matmul(t: &Tensor, x: &[f32], ldx: usize, y: &mut [f32], ldy: usize, nb: usize) -> Result<(), String> {
    let (cols, rows) = (t.cols(), t.rows());
    let on_gpu = t.gpu.is_some() && crate::gpu::ready();
    if nb == 1 || on_gpu || (t.rp.is_some() && nb < 8) {
        for b in 0..nb {
            matvec(t, &x[b * ldx..b * ldx + cols], &mut y[b * ldy..b * ldy + rows])?;
        }
        return Ok(());
    }
    assert!(y.len() >= (nb - 1) * ldy + rows && x.len() >= (nb - 1) * ldx + cols);
    let kind = act_kind(t.typ, cols);
    let acts: Vec<QAct> = (0..nb).into_par_iter().map(|b| QAct::new(kind, &x[b * ldx..b * ldx + cols])).collect();
    let out = OutPtr(y.as_mut_ptr());

    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() && nb >= crate::simd::avx2::GV {
        use crate::simd::Level;
        let level = crate::simd::level();
        // Q8_0 is stored as q + 128, which needs the full u8 range of dpbusd: VNNI CPUs only
        let vnni = level >= Level::AvxVnni;
        if kind == Act::Q8K && (t.typ == T_Q4_K || t.typ == T_Q6_K) {
            match level {
                Level::Avx512 => gemm_avx512::kquant(t, &acts, out, ldy, nb),
                Level::AvxVnni => gemm_avxvnni::kquant(t, &acts, out, ldy, nb),
                _ => gemm_avx2::kquant(t, &acts, out, ldy, nb),
            }
            return Ok(());
        }
        if kind == Act::Q8 && (matches!(t.typ, T_Q4_0 | T_Q4_1 | T_Q5_0 | T_Q5_1) || (vnni && t.typ == T_Q8_0)) {
            match level {
                Level::Avx512 => gemm_avx512::q32(t, &acts, out, ldy, nb),
                Level::AvxVnni => gemm_avxvnni::q32(t, &acts, out, ldy, nb),
                _ => gemm_avx2::q32(t, &acts, out, ldy, nb),
            }
            return Ok(());
        }
    }

    let stride = row_stride(t.typ, cols as u64) as usize;
    let fail = AtomicBool::new(false);
    let put = |b: usize, r: usize, v: f32| {
        let o = out;
        // SAFETY: b < nb and r < rows, checked by the assert above; each cell has one writer
        unsafe { *o.0.add(b * ldy + r) = v }
    };
    if kind == Act::F32 && !matches!(t.typ, T_F32 | T_F16 | T_BF16) {
        // no integer kernel: dequantize each row once, then float dots with every input
        (0..rows).into_par_iter().for_each(|r| {
            DEQ.with(|s| {
                let mut s = s.borrow_mut();
                s.resize(cols, 0.0);
                if !dequantize_row(t.typ, &t.data[r * stride..(r + 1) * stride], &mut s) {
                    fail.store(true, Ordering::Relaxed);
                    return;
                }
                for b in 0..nb {
                    put(b, r, dot_f32(&s, &x[b * ldx..b * ldx + cols]));
                }
            })
        });
    } else {
        // tiles: a block of rows × a group of inputs small enough to stay in L1
        let abytes = match kind {
            Act::Q8K => cols / 256 * std::mem::size_of::<Q8KBlk>(),
            _ => cols / 32 * std::mem::size_of::<Q8Blk>(),
        };
        let tt = ((24 * 1024 / abytes.max(1)) & !3).clamp(4, nb.max(4)).min(nb);
        const RB: usize = 16;
        #[cfg(target_arch = "x86_64")]
        let x4 = crate::simd::avx2() && kind == Act::Q8 && matches!(t.typ, T_Q8_0 | T_Q4_0 | T_Q5_0);
        #[cfg(not(target_arch = "x86_64"))]
        let x4 = false;
        (0..rows.div_ceil(RB)).into_par_iter().for_each(|blk| {
            let (r0, r1) = (blk * RB, ((blk + 1) * RB).min(rows));
            let mut b0 = 0;
            while b0 < nb {
                let b1 = (b0 + tt).min(nb);
                for r in r0..r1 {
                    let rb = &t.data[r * stride..(r + 1) * stride];
                    let mut b = b0;
                    #[cfg(target_arch = "x86_64")]
                    if x4 {
                        while b + 4 <= b1 {
                            let ys = [&acts[b].q8[..], &acts[b + 1].q8[..], &acts[b + 2].q8[..], &acts[b + 3].q8[..]];
                            // SAFETY: AVX2 checked above, rb is a whole row
                            let o = unsafe { crate::simd::simd_call!(dot_q32_x4(t.typ, rb, ys, cols / 32)) };
                            for (i, v) in o.into_iter().enumerate() {
                                put(b + i, r, v);
                            }
                            b += 4;
                        }
                    }
                    while b < b1 {
                        match dot_row(t.typ, kind, rb, &acts[b], &x[b * ldx..b * ldx + cols]) {
                            Some(v) => put(b, r, v),
                            None => fail.store(true, Ordering::Relaxed),
                        }
                        b += 1;
                    }
                }
                b0 = b1;
            }
        });
    }
    if fail.load(Ordering::Relaxed) {
        return Err(format!("matmul: unsupported type {} in {}", type_name(t.typ), t.name));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// row unpacking (portable): q = plain bytes, wsc / wmn = float scale and offset per group
// ---------------------------------------------------------------------------

/// Q4_K row → q 0..15 in natural order, wsc per 32 (d·sc), wmn per 32 (dmin·m)
fn unpack_row_q4k(row: &[u8], nb: usize, q: &mut [u8], wsc: &mut [f32], wmn: &mut [f32]) {
    for b in 0..nb {
        let p = &row[b * 144..];
        let sm = unpack_scales_k4(&p[4..]);
        let (d, dmin) = (fp16(rd_u16(p, 0)), fp16(rd_u16(p, 2)));
        for j in 0..4 {
            for l in 0..32 {
                let v = p[16 + 32 * j + l];
                q[256 * b + 64 * j + l] = v & 0x0F;
                q[256 * b + 64 * j + 32 + l] = v >> 4;
            }
        }
        for s in 0..8 {
            wsc[8 * b + s] = d * sm[s] as f32;
            wmn[8 * b + s] = dmin * sm[8 + s] as f32;
        }
    }
}

/// Q6_K row → q 0..63 in natural order, wsc per 16 (d·sc), wmn per 16 (32·d·sc, the −32 offset)
fn unpack_row_q6k(row: &[u8], nb: usize, q: &mut [u8], wsc: &mut [f32], wmn: &mut [f32]) {
    for b in 0..nb {
        let p = &row[b * 210..];
        let d = fp16(rd_u16(p, 208));
        for half in 0..2 {
            let (ql, qh) = (&p[half * 64..], &p[128 + half * 32..]);
            let o = &mut q[256 * b + 128 * half..];
            for l in 0..32 {
                o[l] = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
                o[32 + l] = (ql[32 + l] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                o[64 + l] = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                o[96 + l] = (ql[32 + l] >> 4) | (((qh[l] >> 6) & 3) << 4);
            }
        }
        for g in 0..16 {
            let sc = p[192 + g] as i8 as f32;
            wsc[16 * b + g] = d * sc;
            wmn[16 * b + g] = 32.0 * d * sc;
        }
    }
}

/// 32-value block row → unsigned q per value, value = wsc·q − wmn per block (Q8_0: VNNI only)
fn unpack_row_q32(typ: u32, row: &[u8], nb: usize, q: &mut [u8], wsc: &mut [f32], wmn: &mut [f32]) {
    for b in 0..nb {
        let (bb, qs, qh_at) = match typ {
            T_Q4_0 => (18, 2, None),
            T_Q4_1 => (20, 4, None),
            T_Q5_0 => (22, 6, Some(2)),
            T_Q8_0 => (34, 2, None),
            _ => (24, 8, Some(4)), // Q5_1
        };
        let p = &row[b * bb..];
        let d = fp16(rd_u16(p, 0));
        if typ == T_Q8_0 {
            // int8 as u8 = q + 128 (VNNI only), value = d·(u − 128)
            for j in 0..32 {
                q[32 * b + j] = p[2 + j] ^ 0x80;
            }
            wsc[b] = d;
            wmn[b] = 128.0 * d;
            continue;
        }
        let (sc, mn) = match typ {
            T_Q4_0 => (d, 8.0 * d),
            T_Q5_0 => (d, 16.0 * d),
            _ => (d, -fp16(rd_u16(p, 2))), // Q4_1 / Q5_1: value = d·q + m
        };
        let qh = qh_at.map(|o| rd_u32(p, o)).unwrap_or(0);
        for j in 0..16 {
            let hi0 = (((qh >> j) & 1) << 4) as u8;
            let hi1 = (((qh >> (j + 16)) & 1) << 4) as u8;
            q[32 * b + j] = (p[qs + j] & 0x0F) | hi0;
            q[32 * b + 16 + j] = (p[qs + j] >> 4) | hi1;
        }
        wsc[b] = sc;
        wmn[b] = mn;
    }
}

/// the GEMM drivers, built once per kernel set (crate::simd::avx2 / avxvnni / avx512)
#[cfg(target_arch = "x86_64")]
macro_rules! gemm_drivers {
    ($($m:ident)::+) => {
    use super::*;
    use $($m)::+::{gemm32_kern, gemmk_kern, Packed, Unpacked, GEMM_NR, GV};

    const NR: usize = GEMM_NR; // weight rows per tile (4 with AVX-512's 32 registers, else 2)
    const NV: usize = 2; // vectors of GV tokens per tile

    /// run fill(r, nr, q, wsc, wmn, qoff, soff) to unpack nr rows from row r, then the tiles
    fn rows_loop(
        rows: usize,
        cols: usize,
        nvec: usize,
        unpack: &(dyn Fn(usize, &mut [u8], &mut [f32], &mut [f32]) + Sync),
        tile_main: &(dyn Fn(&Unpacked, usize, usize) + Sync), // (w, c0, r): NR rows × NV vectors
        tile_r1vn: &(dyn Fn(&Unpacked, usize, usize) + Sync), // 1 row × NV vectors
        tile_r1v1: &(dyn Fn(&Unpacked, usize, usize) + Sync), // 1 row × 1 vector
    ) {
        let (qoff, soff) = (cols + 64, cols / 16 + 1);
        (0..rows.div_ceil(NR)).into_par_iter().for_each(|st| {
            ROWBUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                let (q, wsc, wmn) = &mut *buf;
                q.resize(NR * qoff, 0);
                wsc.resize(NR * soff, 0.0);
                wmn.resize(NR * soff, 0.0);
                let r = st * NR;
                let nr = (rows - r).min(NR);
                for k in 0..nr {
                    unpack(r + k, &mut q[k * qoff..], &mut wsc[k * soff..], &mut wmn[k * soff..]);
                }
                let mut c = 0;
                if nr == NR {
                    let w = Unpacked { q: &q[..], wsc: &wsc[..], wmn: &wmn[..], qoff, soff };
                    while c + NV <= nvec {
                        tile_main(&w, c, r);
                        c += NV;
                    }
                }
                for k in 0..nr {
                    let w = Unpacked { q: &q[k * qoff..], wsc: &wsc[k * soff..], wmn: &wmn[k * soff..], qoff: 0, soff: 0 };
                    let mut cc = c;
                    while cc + NV <= nvec {
                        tile_r1vn(&w, cc, r + k);
                        cc += NV;
                    }
                    while cc < nvec {
                        tile_r1v1(&w, cc, r + k);
                        cc += 1;
                    }
                }
            })
        });
    }

    /// Q4_K / Q6_K
    pub fn kquant(t: &Tensor, acts: &[QAct], out: OutPtr, ldy: usize, nb: usize) {
        let q6 = t.typ == T_Q6_K;
        let ms = if q6 { 16 } else { 32 };
        let (cols, rows) = (t.cols(), t.rows());
        let (nbl, ng, nvec) = (cols / 256, cols / ms, nb.div_ceil(GV));
        let stride = row_stride(t.typ, cols as u64) as usize;

        // pack: 4 consecutive values of GV tokens side by side, + scales and min-group sums
        let mut pk = vec![0i8; nvec * cols * GV];
        pk.par_chunks_mut(cols * GV).enumerate().for_each(|(c, dst)| {
            for ti in 0..GV {
                let tok = c * GV + ti;
                if tok >= nb {
                    continue;
                }
                for (b, y) in acts[tok].q8k.iter().enumerate() {
                    for i in 0..64 {
                        let at = ((b * 64 + i) * GV + ti) * 4;
                        dst[at..at + 4].copy_from_slice(&y.qs[4 * i..4 * i + 4]);
                    }
                }
            }
        });
        let mut gyd = vec![0.0f32; nvec * nbl * GV];
        let mut gbs = vec![0.0f32; nvec * ng * GV];
        for tok in 0..nb {
            let (c, ti) = (tok / GV, tok % GV);
            for (b, y) in acts[tok].q8k.iter().enumerate() {
                gyd[(c * nbl + b) * GV + ti] = y.d;
                for g in 0..256 / ms {
                    let s: i32 = (0..ms / 16).map(|h| y.bsums[g * (ms / 16) + h] as i32).sum();
                    gbs[(c * ng + b * (256 / ms) + g) * GV + ti] = s as f32;
                }
            }
        }
        let packed = Packed { pk: &pk, gyd: &gyd, gbs: &gbs };
        let a = &packed;
        let unpack = |r: usize, q: &mut [u8], sc: &mut [f32], mn: &mut [f32]| {
            let rb = &t.data[r * stride..(r + 1) * stride];
            if q6 {
                unpack_row_q6k(rb, nbl, q, sc, mn)
            } else {
                unpack_row_q4k(rb, nbl, q, sc, mn)
            }
        };
        let y = out;
        // SAFETY (all three): the caller checked the CPU features; tiles write only cells tok < nb, r < rows
        fn tile<const R: usize, const V: usize>(q6: bool, w: &Unpacked, cols: usize, c0: usize, a: &Packed, y: OutPtr, ldy: usize, r: usize, nb: usize) {
            unsafe {
                if q6 {
                    gemmk_kern::<R, V, 16, 16>(w, cols, c0, a, y.0, ldy, r, nb)
                } else {
                    gemmk_kern::<R, V, 32, 32>(w, cols, c0, a, y.0, ldy, r, nb)
                }
            }
        }
        rows_loop(
            rows,
            cols,
            nvec,
            &unpack,
            &|w, c0, r| tile::<NR, NV>(q6, w, cols, c0, a, y, ldy, r, nb),
            &|w, c0, r| tile::<1, NV>(q6, w, cols, c0, a, y, ldy, r, nb),
            &|w, c0, r| tile::<1, 1>(q6, w, cols, c0, a, y, ldy, r, nb),
        );
    }

    /// Q4_0 / Q4_1 / Q5_0 / Q5_1 (and Q8_0 with VNNI)
    pub fn q32(t: &Tensor, acts: &[QAct], out: OutPtr, ldy: usize, nb: usize) {
        let (cols, rows) = (t.cols(), t.rows());
        let (ng, nvec) = (cols / 32, nb.div_ceil(GV));
        let stride = row_stride(t.typ, cols as u64) as usize;
        let mut pk = vec![0i8; nvec * cols * GV];
        pk.par_chunks_mut(cols * GV).enumerate().for_each(|(c, dst)| {
            for ti in 0..GV {
                let tok = c * GV + ti;
                if tok >= nb {
                    continue;
                }
                for (g, y) in acts[tok].q8.iter().enumerate() {
                    for i in 0..8 {
                        let at = ((g * 8 + i) * GV + ti) * 4;
                        dst[at..at + 4].copy_from_slice(&y.qs[4 * i..4 * i + 4]);
                    }
                }
            }
        });
        let mut gyd = vec![0.0f32; nvec * ng * GV];
        let mut gbs = vec![0.0f32; nvec * ng * GV];
        for tok in 0..nb {
            let (c, ti) = (tok / GV, tok % GV);
            for (g, y) in acts[tok].q8.iter().enumerate() {
                gyd[(c * ng + g) * GV + ti] = y.d;
                gbs[(c * ng + g) * GV + ti] = y.d * (y.s16[0] as i32 + y.s16[1] as i32) as f32;
            }
        }
        let packed = Packed { pk: &pk, gyd: &gyd, gbs: &gbs };
        let a = &packed;
        let typ = t.typ;
        let unpack = |r: usize, q: &mut [u8], sc: &mut [f32], mn: &mut [f32]| {
            unpack_row_q32(typ, &t.data[r * stride..(r + 1) * stride], ng, q, sc, mn)
        };
        let y = out;
        fn tile<const R: usize, const V: usize>(w: &Unpacked, cols: usize, c0: usize, a: &Packed, y: OutPtr, ldy: usize, r: usize, nb: usize) {
            // SAFETY: as in kquant
            unsafe { gemm32_kern::<R, V>(w, cols, c0, a, y.0, ldy, r, nb) }
        }
        rows_loop(
            rows,
            cols,
            nvec,
            &unpack,
            &|w, c0, r| tile::<NR, NV>(w, cols, c0, a, y, ldy, r, nb),
            &|w, c0, r| tile::<1, NV>(w, cols, c0, a, y, ldy, r, nb),
            &|w, c0, r| tile::<1, 1>(w, cols, c0, a, y, ldy, r, nb),
        );
    }
    };
}

#[cfg(target_arch = "x86_64")]
mod gemm_avx2 {
    gemm_drivers!(crate::simd::avx2);
}
#[cfg(target_arch = "x86_64")]
mod gemm_avxvnni {
    gemm_drivers!(crate::simd::avxvnni);
}
#[cfg(target_arch = "x86_64")]
mod gemm_avx512 {
    gemm_drivers!(crate::simd::avx512);
}
