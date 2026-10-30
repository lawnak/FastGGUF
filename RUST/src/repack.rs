// Q4_K / Q4_0 weights re-arranged for single-token generation: 8 rows side by side, so one
// 32-byte load holds 4 weights of each of the 8 rows and each vector lane is one row from
// start to finish. Done at load time (FASTGGUF_REPACK=0 turns it off).

use crate::quant::*;
use rayon::prelude::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Q4Kx8 {
    pub d: [u16; 8],              // fp16, one per row
    pub dmin: [u16; 8],
    pub sc: [[u8; 8]; 8],         // [sub-block][row] six-bit scales
    pub mn: [[[u8; 2]; 8]; 4],    // [sub-block pair][row][2] six-bit mins
    pub qs: [[[[u8; 4]; 8]; 8]; 4], // [pair j][step s][row][i]: the file's byte for value 4s+i
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Q40x8 {
    pub d: [u16; 8],
    pub qs: [[[u8; 4]; 8]; 4], // [step s][row][i]
}

pub enum Repacked {
    Q4K(Vec<Q4Kx8>), // groups × blocks per row
    Q40(Vec<Q40x8>),
}

fn repack_q4k_group(src: &[u8], stride: usize, nb: usize, dst: &mut [Q4Kx8]) {
    for b in 0..nb {
        for r in 0..8 {
            let p = &src[r * stride + b * 144..];
            let sm = unpack_scales_k4(&p[4..]);
            let o = &mut dst[b];
            o.d[r] = rd_u16(p, 0);
            o.dmin[r] = rd_u16(p, 2);
            for j in 0..8 {
                o.sc[j][r] = sm[j];
            }
            for q in 0..4 {
                o.mn[q][r] = [sm[8 + 2 * q], sm[9 + 2 * q]];
            }
            for j in 0..4 {
                for s in 0..8 {
                    o.qs[j][s][r].copy_from_slice(&p[16 + 32 * j + 4 * s..16 + 32 * j + 4 * s + 4]);
                }
            }
        }
    }
}

fn repack_q40_group(src: &[u8], stride: usize, nb: usize, dst: &mut [Q40x8]) {
    for b in 0..nb {
        for r in 0..8 {
            let p = &src[r * stride + b * 18..];
            dst[b].d[r] = rd_u16(p, 0);
            for s in 0..4 {
                dst[b].qs[s][r].copy_from_slice(&p[2 + 4 * s..6 + 4 * s]);
            }
        }
    }
}

/// re-arrange a [cols, rows] matrix if it qualifies; returns None otherwise
pub fn repack(typ: u32, data: &[u8], cols: usize, rows: usize) -> Option<Repacked> {
    if rows % 8 != 0 {
        return None;
    }
    let stride = row_stride(typ, cols as u64) as usize;
    let ng = rows / 8;
    if typ == T_Q4_K && cols % 256 == 0 {
        let nb = cols / 256;
        let zero = Q4Kx8 { d: [0; 8], dmin: [0; 8], sc: [[0; 8]; 8], mn: [[[0; 2]; 8]; 4], qs: [[[[0; 4]; 8]; 8]; 4] };
        let mut v = vec![zero; ng * nb];
        v.par_chunks_mut(nb).enumerate().for_each(|(g, dst)| repack_q4k_group(&data[g * 8 * stride..], stride, nb, dst));
        return Some(Repacked::Q4K(v));
    }
    if typ == T_Q4_0 && cols % 32 == 0 {
        let nb = cols / 32;
        let zero = Q40x8 { d: [0; 8], qs: [[[0; 4]; 8]; 4] };
        let mut v = vec![zero; ng * nb];
        v.par_chunks_mut(nb).enumerate().for_each(|(g, dst)| repack_q40_group(&data[g * 8 * stride..], stride, nb, dst));
        return Some(Repacked::Q40(v));
    }
    None
}

impl Repacked {
    pub fn bytes(&self) -> usize {
        match self {
            Repacked::Q4K(v) => v.len() * std::mem::size_of::<Q4Kx8>(),
            Repacked::Q40(v) => v.len() * std::mem::size_of::<Q40x8>(),
        }
    }
}

/// activation sums per 32 values for the Q4_K min term, two per int32 (low/high 16 bits)
pub fn q4kx8_bsums(y: &[Q8KBlk], nb: usize) -> Vec<i32> {
    let mut bsp = vec![0i32; nb * 4];
    for b in 0..nb {
        for p in 0..4 {
            let s0 = y[b].bsums[4 * p] as i32 + y[b].bsums[4 * p + 1] as i32;
            let s1 = y[b].bsums[4 * p + 2] as i32 + y[b].bsums[4 * p + 3] as i32;
            bsp[4 * b + p] = (((s1 as u16 as u32) << 16) | (s0 as u16 as u32)) as i32;
        }
    }
    bsp
}

/// out[0..8] = dot(row r, y) for the 8 rows of one Q4_K group (portable version)
pub fn dot_q4kx8(blocks: &[Q4Kx8], y: &[Q8KBlk], bsp: &[i32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { crate::simd::simd_call!(dot_q4kx8(blocks, y, bsp, out)) };
    }
    let mut acc = [0.0f32; 8];
    for (b, bl) in blocks.iter().enumerate() {
        for r in 0..8 {
            let (mut sumi, mut mi) = (0i32, 0i32);
            for j in 0..4 {
                let (mut lo, mut hi) = (0i32, 0i32);
                for s in 0..8 {
                    for i in 0..4 {
                        let v = bl.qs[j][s][r][i];
                        lo += (v & 0x0F) as i32 * y[b].qs[64 * j + 4 * s + i] as i32;
                        hi += (v >> 4) as i32 * y[b].qs[64 * j + 32 + 4 * s + i] as i32;
                    }
                }
                sumi += lo * bl.sc[2 * j][r] as i32 + hi * bl.sc[2 * j + 1][r] as i32;
            }
            for p in 0..4 {
                let w = bsp[4 * b + p] as u32;
                mi += bl.mn[p][r][0] as i32 * (w & 0xFFFF) as u16 as i16 as i32
                    + bl.mn[p][r][1] as i32 * (w >> 16) as u16 as i16 as i32;
            }
            acc[r] += fp16(bl.d[r]) * y[b].d * sumi as f32 - fp16(bl.dmin[r]) * y[b].d * mi as f32;
        }
    }
    out[..8].copy_from_slice(&acc);
}

/// out[0..8] for one Q4_0 group (portable version)
pub fn dot_q40x8(blocks: &[Q40x8], y: &[Q8Blk], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { crate::simd::simd_call!(dot_q40x8(blocks, y, out)) };
    }
    let mut acc = [0.0f32; 8];
    for (b, bl) in blocks.iter().enumerate() {
        for r in 0..8 {
            let mut si = 0i32;
            for s in 0..4 {
                for i in 0..4 {
                    let v = bl.qs[s][r][i];
                    si += ((v & 0x0F) as i32 - 8) * y[b].qs[4 * s + i] as i32
                        + ((v >> 4) as i32 - 8) * y[b].qs[16 + 4 * s + i] as i32;
                }
            }
            acc[r] += fp16(bl.d[r]) * y[b].d * si as f32;
        }
    }
    out[..8].copy_from_slice(&acc);
}
