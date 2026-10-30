// GGUF reader: the file is memory-mapped and kept for the life of the program, so tensors
// point straight into it (the OS reads the pages from disk on first use).

use crate::quant::*;
use crate::repack::Repacked;
use std::fs::File;
use std::path::Path;

pub enum KvVal {
    Str(String),
    Num(f64),
    IntArr(Vec<i32>),
    StrArr(Vec<Vec<u8>>), // raw bytes: tokenizer tokens are not always valid UTF-8
}

pub struct Kv {
    pub key: String,
    pub val: KvVal,
}

pub struct Tensor {
    pub name: String,
    pub typ: u32,
    pub n_dims: usize,
    pub ne: [u64; 4],
    pub data: &'static [u8],
    pub nbytes: u64,
    pub rp: Option<Repacked>, // the same weights re-arranged for the CPU kernels
    pub gpu: Option<crate::gpu::GpuMem>, // a copy of the weights in VRAM
}

impl Tensor {
    pub fn cols(&self) -> usize {
        self.ne[0] as usize
    }
    pub fn rows(&self) -> usize {
        (1..self.n_dims).map(|i| self.ne[i] as usize).product()
    }
    pub fn elems(&self) -> usize {
        (0..self.n_dims).map(|i| self.ne[i] as usize).product()
    }
}

pub struct Gguf {
    pub version: u32,
    pub alignment: u64,
    pub kvs: Vec<Kv>,
    pub tensors: Vec<Tensor>,
}

struct Rdr<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Rdr<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        if self.pos.checked_add(n).is_none_or(|e| e > self.b.len()) {
            return Err("unexpected end of file".into());
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let n = self.u64()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    fn string(&mut self) -> Result<String, String> {
        Ok(String::from_utf8_lossy(&self.bytes()?).into_owned())
    }
    fn scalar(&mut self, vt: u32) -> Result<f64, String> {
        Ok(match vt {
            0 => self.u8()? as f64,
            1 => self.u8()? as i8 as f64,
            2 => self.u16()? as f64,
            3 => self.u16()? as i16 as f64,
            4 => self.u32()? as f64,
            5 => self.u32()? as i32 as f64,
            6 => f32::from_bits(self.u32()?) as f64,
            7 => self.u8()? as f64,
            10 => self.u64()? as f64,
            11 => self.u64()? as i64 as f64,
            12 => f64::from_bits(self.u64()?),
            _ => return Err(format!("unknown metadata value type {vt}")),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        // SAFETY: the file is opened read-only; like the C version we assume it is not changed
        // while the program runs
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("cannot map {}: {e}", path.display()))?;
        let map: &'static memmap2::Mmap = Box::leak(Box::new(map));
        let data: &'static [u8] = &map[..];
        if data.len() < 16 || &data[..4] != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let mut r = Rdr { b: data, pos: 4 };
        let version = r.u32()?;
        let n_tensors = r.u64()? as usize;
        let n_kvs = r.u64()? as usize;

        let mut kvs = Vec::with_capacity(n_kvs.min(1 << 16));
        for _ in 0..n_kvs {
            let key = r.string().map_err(|e| format!("metadata parse error: {e}"))?;
            let vt = r.u32()?;
            let val = match vt {
                8 => KvVal::Str(r.string()?),
                9 => {
                    let et = r.u32()?;
                    let n = r.u64()? as usize;
                    match et {
                        8 => {
                            let mut v = Vec::with_capacity(n.min(1 << 22));
                            for _ in 0..n {
                                v.push(r.bytes()?);
                            }
                            KvVal::StrArr(v)
                        }
                        0 | 1 | 4 | 5 | 7 => {
                            let mut v = Vec::with_capacity(n.min(1 << 22));
                            for _ in 0..n {
                                v.push(r.scalar(et)? as i32);
                            }
                            KvVal::IntArr(v)
                        }
                        _ => {
                            // array of another type: skip it
                            for _ in 0..n {
                                r.scalar(et)?;
                            }
                            KvVal::Num(n as f64)
                        }
                    }
                }
                _ => KvVal::Num(r.scalar(vt).map_err(|e| format!("metadata parse error: {e}"))?),
            };
            kvs.push(Kv { key, val });
        }

        let mut g = Gguf { version, alignment: 32, kvs, tensors: Vec::with_capacity(n_tensors.min(1 << 20)) };
        if let Some(KvVal::Num(a)) = g.kv("general.alignment") {
            if *a > 0.0 {
                g.alignment = *a as u64;
            }
        }

        let mut offsets = Vec::with_capacity(n_tensors.min(1 << 20));
        for _ in 0..n_tensors {
            let name = r.string().map_err(|e| format!("tensor header parse error: {e}"))?;
            let n_dims = r.u32()? as usize;
            let mut ne = [1u64; 4];
            let mut elems = 1u64;
            for j in 0..n_dims {
                let v = r.u64()?;
                if j < 4 {
                    ne[j] = v;
                }
                elems = elems.saturating_mul(v);
            }
            let typ = r.u32()?;
            offsets.push(r.u64()?);
            let nbytes = tensor_nbytes(typ, elems);
            if nbytes == 0 {
                return Err(format!("tensor {name}: unsupported type {}", type_name(typ)));
            }
            g.tensors.push(Tensor { name, typ, n_dims: n_dims.min(4), ne, data: &[], nbytes, rp: None, gpu: None });
        }

        let data_start = (r.pos as u64).div_ceil(g.alignment) * g.alignment;
        for (t, off) in g.tensors.iter_mut().zip(offsets) {
            let s = data_start + off;
            if s + t.nbytes > data.len() as u64 {
                return Err(format!("tensor {} extends past EOF", t.name));
            }
            t.data = &data[s as usize..(s + t.nbytes) as usize];
        }
        Ok(g)
    }

    pub fn kv(&self, key: &str) -> Option<&KvVal> {
        self.kvs.iter().find(|k| k.key == key).map(|k| &k.val)
    }

    pub fn num(&self, key: &str, def: f64) -> f64 {
        match self.kv(key) {
            Some(KvVal::Num(v)) => *v,
            Some(KvVal::IntArr(a)) if !a.is_empty() => a[0] as f64,
            _ => def,
        }
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        match self.kv(key) {
            Some(KvVal::Str(s)) => Some(s),
            _ => None,
        }
    }

    pub fn tensor_index(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// a whole tensor as floats
    pub fn load_f32(&self, name: &str) -> Option<Vec<f32>> {
        let t = &self.tensors[self.tensor_index(name)?];
        let mut out = vec![0.0f32; t.elems()];
        dequantize_row(t.typ, t.data, &mut out).then_some(out)
    }
}
