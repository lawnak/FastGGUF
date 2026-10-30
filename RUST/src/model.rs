// The model: config + weights + KV cache + buffers, and the forward pass.
// Architectures: gemma* (Gemma 2/3/3n/4) / qwen2 / qwen3 / qwen3moe / llama (includes Mistral).

use crate::batch;
use crate::gguf::{Gguf, KvVal, Tensor};
use crate::gpu;
use crate::matvec::{act_kind, axpy, dot_f32, dot_row, matvec, QAct};
use crate::ops::{self, kvq};
use crate::quant::*;
use crate::repack;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub const MAX_LAYERS: usize = 128;
/// room kept in a sliding-window cache for one prompt batch (same layout as the C version)
pub const FG_BATCH: usize = 64;

pub struct Layer {
    attn_norm: Vec<f32>,
    wq: usize,
    wk: Option<usize>,
    wv: Option<usize>, // None in Gemma: V = K
    wo: usize,
    bq: Option<Vec<f32>>, // Qwen2 biases
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>, // per-head norms (Gemma/Qwen3)
    k_norm: Option<Vec<f32>>,
    attn_post_norm: Option<Vec<f32>>, // Gemma
    ffn_norm: Vec<f32>,
    ffn_gate: Option<usize>,
    ffn_up: Option<usize>,
    ffn_down: Option<usize>,
    // MoE (qwen3moe): router + experts stacked 3D [cols, rows, n_expert], read from the mmap
    ffn_gate_inp: Option<usize>,
    gate_exps: Option<usize>,
    up_exps: Option<usize>,
    down_exps: Option<usize>,
    ffn_post_norm: Option<Vec<f32>>,
    ple_inp_gate: Option<usize>,
    ple_proj: Option<usize>,
    ple_post_norm: Option<Vec<f32>>,
    out_scale: f32,
    pub head_dim: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub kv_dim: usize,
    pub q_dim: usize,
    pub ff_dim: usize,
}

#[allow(dead_code)] // n_ctx, n_vocab, n_expert and seq_ids are for the server (next phase)
pub struct Model {
    pub g: Gguf,
    pub arch: String,
    v_norm: bool,
    attn_scale: f32, // 0 = 1/sqrt(head_dim)
    rope_scale: f64, // freq_scale for global layers only
    act_gelu: bool,
    rope_neox: bool,
    embd_scale: f32,
    pub n_layer: usize,
    pub n_embd: usize,
    n_ple: usize,
    n_swa: usize,
    n_kv_from_start: usize,
    n_rot: usize,
    n_rot_swa: usize,
    rope_base: f64,
    rope_base_swa: f64,
    norm_eps: f32,
    final_softcap: f32,
    is_swa: Vec<bool>,
    kv_donor: Vec<usize>,
    pub n_ctx: usize,
    pub n_vocab: usize,
    n_expert: usize,
    n_expert_used: usize,
    ff_exp: usize,

    tok_embd: usize,
    output: usize,
    output_norm: Vec<f32>,
    ple_tok_embd: Option<usize>,
    ple_model_proj: Option<usize>,
    ple_proj_norm: Vec<f32>,
    rope_freqs: Option<Vec<f32>>,
    pub layers: Vec<Layer>,

    // KV cache per owning layer — a ring buffer for sliding-window layers
    cache_k: Vec<Vec<f32>>,
    cache_v: Vec<Vec<f32>>,
    cache_cap: Vec<usize>,
    // TurboQuant (FASTGGUF_KV=turbo): quantized cache instead of f32
    kvq: bool,
    kbits: usize,
    vbits: usize,
    cache_kq: Vec<Vec<u8>>,
    cache_vq: Vec<Vec<u8>>,

    // tokens resident in the cache (prefix reuse between requests)
    pub seq_ids: Vec<i32>,

    // buffers allocated once
    x: Vec<f32>,
    cur: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    ao: Vec<f32>,
    f: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn_out: Vec<f32>,
    ple_in: Vec<f32>,
    ple_t: Vec<f32>,
    ple_t2: Vec<f32>,
    pub logits: Vec<f32>,
    moe_probs: Vec<f32>,
    moe_h: Vec<f32>,
    bb: Option<Box<BatchBufs>>,
}

thread_local! {
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn env_is(name: &str, val: &str) -> bool {
    std::env::var(name).map(|v| v == val).unwrap_or(false)
}

impl Model {
    pub fn load(g: Gguf, n_ctx: usize) -> Result<Model, String> {
        let arch = g.str("general.architecture").ok_or("no architecture")?.to_string();
        let key = |s: &str| format!("{arch}.{s}");

        let is_gemma = arch.starts_with("gemma");
        if !is_gemma && !arch.starts_with("qwen") && arch != "llama" && arch != "mistral" {
            eprintln!("[warn] arch '{arch}' is untested - continuing with the standard llama path");
        }
        // Gemma 3: the differences as llama.cpp implements them; the Gemma 4 path is unchanged
        let is_g3 = arch == "gemma3";
        let mut attn_scale = if is_gemma && !is_g3 { 1.0 } else { 0.0 };
        let mut rope_scale = 1.0;
        if is_g3 {
            // linear scaling (4B/12B/27B: factor 8) on global layers only
            let f = g.num(&key("rope.scaling.factor"), 0.0);
            let linear = match g.kv(&key("rope.scaling.type")) {
                None => true,
                Some(KvVal::Str(s)) => s == "linear",
                _ => false,
            };
            if f > 0.0 && linear {
                rope_scale = 1.0 / f;
            }
        }

        let n_layer = g.num(&key("block_count"), 0.0) as i64;
        let n_embd = g.num(&key("embedding_length"), 0.0) as i64;
        if n_layer <= 0 || n_embd <= 0 || n_layer as usize > MAX_LAYERS {
            return Err(format!("bad config: layers={n_layer} embd={n_embd}"));
        }
        let (n_layer, n_embd) = (n_layer as usize, n_embd as usize);
        let n_ple = g.num(&key("embedding_length_per_layer_input"), 0.0).max(0.0) as usize;
        let n_swa = g.num(&key("attention.sliding_window"), 0.0).max(0.0) as usize;
        let shared_kv = g.num(&key("attention.shared_kv_layers"), 0.0).max(0.0) as usize;
        let n_kv_from_start = n_layer.saturating_sub(shared_kv);
        let rope_base = g.num(&key("rope.freq_base"), if is_gemma { 1e6 } else { 10000.0 });
        let rope_base_swa = g.num(&key("rope.freq_base_swa"), 10000.0);
        let n_rot = g.num(&key("rope.dimension_count"), 0.0).max(0.0) as usize;
        let n_rot_swa = g.num(&key("rope.dimension_count_swa"), 0.0).max(0.0) as usize;
        let norm_eps = g.num(&key("attention.layer_norm_rms_epsilon"), 1e-6) as f32;
        let final_softcap = g.num(&key("final_logit_softcapping"), 0.0) as f32;

        let n_expert = g.num(&key("expert_count"), 0.0).max(0.0) as usize;
        let n_expert_used = g.num(&key("expert_used_count"), 0.0).max(0.0) as usize;
        let mut ff_exp = g.num(&key("expert_feed_forward_length"), 0.0).max(0.0) as usize;
        if n_expert > 0 && (n_expert_used == 0 || n_expert_used > n_expert) {
            return Err(format!("bad MoE config: experts={n_expert} used={n_expert_used}"));
        }

        // which layers use the sliding window
        let mut is_swa = vec![false; n_layer];
        match g.kv(&key("attention.sliding_window_pattern")) {
            Some(KvVal::IntArr(a)) if a.len() >= n_layer => {
                for i in 0..n_layer {
                    is_swa[i] = a[i] != 0;
                }
            }
            Some(KvVal::Num(p)) if *p > 0.0 => {
                let p = *p as usize;
                for i in 0..n_layer {
                    is_swa[i] = (i + 1) % p != 0;
                }
            }
            _ if is_g3 && n_swa > 0 => {
                // Gemma 3 files do not store the pattern: 5 sliding-window layers then one global
                for i in 0..n_layer {
                    is_swa[i] = (i + 1) % 6 != 0;
                }
            }
            _ => {}
        }

        // layers past n_kv_from_start reuse the cache of the last owning layer of the same kind
        let mut kv_donor = vec![0usize; n_layer];
        let (mut last_swa, mut last_full) = (None, None);
        for i in 0..n_layer {
            if i < n_kv_from_start {
                if is_swa[i] {
                    last_swa = Some(i)
                } else {
                    last_full = Some(i)
                }
                kv_donor[i] = i;
            } else {
                kv_donor[i] = (if is_swa[i] { last_swa } else { last_full }).ok_or(format!("layer {i}: no KV donor"))?;
            }
        }

        let must_t = |name: &str| g.tensor_index(name).ok_or_else(|| format!("missing tensor: {name}"));
        let must_f = |name: &str| g.load_f32(name).ok_or_else(|| format!("missing tensor: {name}"));
        let rows = |i: usize| g.tensors[i].rows();

        let tok_embd = must_t("token_embd.weight")?;
        let n_vocab = rows(tok_embd);
        let output = g.tensor_index("output.weight").unwrap_or(tok_embd);
        let output_norm = must_f("output_norm.weight")?;
        let (mut ple_tok_embd, mut ple_model_proj, mut ple_proj_norm) = (None, None, Vec::new());
        if n_ple > 0 {
            ple_tok_embd = Some(must_t("per_layer_token_embd.weight")?);
            ple_model_proj = Some(must_t("per_layer_model_proj.weight")?);
            ple_proj_norm = must_f("per_layer_proj_norm.weight")?;
        }
        let rope_freqs = g.load_f32("rope_freqs.weight");

        // head dim from the metadata (used when there is no q_norm)
        let meta_n_head = g.num(&key("attention.head_count"), 0.0).max(0.0) as usize;
        let mut meta_head_dim = g.num(&key("attention.key_length"), 0.0).max(0.0) as usize;
        if meta_head_dim == 0 && meta_n_head > 0 {
            meta_head_dim = n_embd / meta_n_head;
        }

        let (mut max_q, mut max_kv, mut max_ff) = (0, 0, 0);
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let blk = |s: &str| format!("blk.{i}.{s}");
            let gemma_needs = |v: Option<Vec<f32>>, s: &str| -> Result<Option<Vec<f32>>, String> {
                if is_gemma && v.is_none() {
                    return Err(format!("missing tensor: {}", blk(s)));
                }
                Ok(v)
            };
            let attn_norm = must_f(&blk("attn_norm.weight"))?;
            let wq = must_t(&blk("attn_q.weight"))?;
            let wo = must_t(&blk("attn_output.weight"))?;
            let q_norm = gemma_needs(g.load_f32(&blk("attn_q_norm.weight")), "attn_q_norm.weight")?;
            let attn_post_norm = gemma_needs(g.load_f32(&blk("post_attention_norm.weight")), "post_attention_norm.weight")?;
            let bq = g.load_f32(&blk("attn_q.bias"));

            // the true head dim is the length of q_norm when there is one
            let head_dim = q_norm.as_ref().map(|v| v.len()).unwrap_or(meta_head_dim);
            if head_dim == 0 {
                return Err(format!("L{i}: cannot determine head_dim"));
            }
            let q_dim = rows(wq);
            let n_head = q_dim / head_dim;
            if n_head * head_dim != q_dim {
                return Err(format!("L{i}: q_dim {q_dim} % head_dim {head_dim} != 0"));
            }

            let (mut wk, mut wv, mut k_norm, mut bk, mut bv) = (None, None, None, None, None);
            let (mut kv_dim, mut n_head_kv) = (0, 0);
            if i < n_kv_from_start {
                wk = Some(must_t(&blk("attn_k.weight"))?);
                wv = g.tensor_index(&blk("attn_v.weight"));
                if wv.is_none() && !is_gemma {
                    return Err(format!("missing tensor: {}", blk("attn_v.weight")));
                }
                k_norm = gemma_needs(g.load_f32(&blk("attn_k_norm.weight")), "attn_k_norm.weight")?;
                bk = g.load_f32(&blk("attn_k.bias"));
                bv = g.load_f32(&blk("attn_v.bias"));
                if let (Some(kn), Some(qn)) = (&k_norm, &q_norm) {
                    if kn.len() != qn.len() {
                        return Err(format!("L{i}: k_norm {} != q_norm {}", kn.len(), qn.len()));
                    }
                }
                kv_dim = rows(wk.unwrap());
                n_head_kv = kv_dim / head_dim;
                if n_head_kv * head_dim != kv_dim {
                    return Err(format!("L{i}: kv_dim {kv_dim} % head_dim {head_dim} != 0"));
                }
            }

            let ffn_norm = must_f(&blk("ffn_norm.weight"))?;
            let ffn_gate_inp = if n_expert > 0 { g.tensor_index(&blk("ffn_gate_inp.weight")) } else { None };
            let (mut ffn_gate, mut ffn_up, mut ffn_down) = (None, None, None);
            let (mut gate_exps, mut up_exps, mut down_exps) = (None, None, None);
            let ff_dim;
            if let Some(gi) = ffn_gate_inp {
                // MoE: the experts stay in the mmap so the OS reads only the chosen experts' pages
                if g.tensor_index(&blk("ffn_gate_inp_shexp.weight")).is_some() {
                    return Err(format!("L{i}: shared experts (qwen2moe) not supported"));
                }
                let (ge, ue, de) =
                    (must_t(&blk("ffn_gate_exps.weight"))?, must_t(&blk("ffn_up_exps.weight"))?, must_t(&blk("ffn_down_exps.weight"))?);
                if ff_exp == 0 {
                    ff_exp = g.tensors[ge].ne[1] as usize;
                }
                let shape = |t: &Tensor, c: usize, r: usize| {
                    t.n_dims == 3 && t.ne[0] == c as u64 && t.ne[1] == r as u64 && t.ne[2] == n_expert as u64
                };
                let git = &g.tensors[gi];
                if git.n_dims != 2
                    || git.ne[0] != n_embd as u64
                    || git.ne[1] != n_expert as u64
                    || !shape(&g.tensors[ge], n_embd, ff_exp)
                    || !shape(&g.tensors[ue], n_embd, ff_exp)
                    || !shape(&g.tensors[de], ff_exp, n_embd)
                {
                    return Err(format!(
                        "L{i}: MoE tensor shapes do not match embd={n_embd} ff_exp={ff_exp} experts={n_expert}"
                    ));
                }
                (gate_exps, up_exps, down_exps) = (Some(ge), Some(ue), Some(de));
                ff_dim = ff_exp;
            } else {
                let gt = must_t(&blk("ffn_gate.weight"))?;
                ffn_gate = Some(gt);
                ffn_up = Some(must_t(&blk("ffn_up.weight"))?);
                ffn_down = Some(must_t(&blk("ffn_down.weight"))?);
                ff_dim = rows(gt);
            }
            let ffn_post_norm = gemma_needs(g.load_f32(&blk("post_ffw_norm.weight")), "post_ffw_norm.weight")?;

            let (mut ple_inp_gate, mut ple_proj, mut ple_post_norm) = (None, None, None);
            if n_ple > 0 {
                ple_inp_gate = Some(must_t(&blk("inp_gate.weight"))?);
                ple_proj = Some(must_t(&blk("proj.weight"))?);
                ple_post_norm = Some(must_f(&blk("post_norm.weight"))?);
            }
            let out_scale = g.load_f32(&blk("layer_output_scale.weight")).and_then(|v| v.first().copied()).unwrap_or(1.0);

            max_q = max_q.max(q_dim);
            max_kv = max_kv.max(kv_dim);
            max_ff = max_ff.max(ff_dim);
            layers.push(Layer {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                bq,
                bk,
                bv,
                q_norm,
                k_norm,
                attn_post_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_gate_inp,
                gate_exps,
                up_exps,
                down_exps,
                ffn_post_norm,
                ple_inp_gate,
                ple_proj,
                ple_post_norm,
                out_scale,
                head_dim,
                n_head,
                n_head_kv,
                kv_dim,
                q_dim,
                ff_dim,
            });
        }

        if is_g3 {
            // query_pre_attn_scalar: n_embd/n_head for 27B (62 layers), the head dim for the others
            let l0 = &layers[0];
            let qpas = if n_layer == 62 { (n_embd / l0.n_head) as f32 } else { l0.head_dim as f32 };
            attn_scale = 1.0 / qpas.sqrt();
        }

        // TurboQuant — needs head_dim to be a power of 2:
        //   turbo  K4+V4 (Qwen: keys automatically 8-bit, its attention logits are huge)
        //   turbo4 force K4 even on Qwen, turbo8 K8+V4, turbo2 K4+V2
        let kvenv = std::env::var("FASTGGUF_KV").unwrap_or_default();
        let mut kvq_on = kvenv.starts_with("turbo") || kvenv == "on";
        let mut kbits = if kvenv == "turbo8" { 8 } else { 4 };
        let vbits = if kvenv == "turbo2" { 2 } else { 4 };
        if kvq_on && kbits == 4 && arch.starts_with("qwen") && kvenv != "turbo4" {
            kbits = 8;
            println!("[kv] {arch} attention logits are huge - keys auto-upgraded to 8-bit (force 4-bit with FASTGGUF_KV=turbo4)");
        }
        if kvq_on {
            for l in &layers[..n_kv_from_start] {
                let hd = l.head_dim;
                if !(8..=kvq::MAX_HD).contains(&hd) || !hd.is_power_of_two() {
                    kvq_on = false;
                }
            }
            if !kvq_on {
                println!("[kv] TurboQuant needs power-of-2 head_dim - using f32 cache");
            }
        }

        // the KV cache — a sliding-window layer keeps only its window (+ one batch)
        let (mut cache_k, mut cache_v, mut cache_kq, mut cache_vq) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut cache_cap = vec![0usize; n_layer];
        let (mut kv_bytes, mut kv_f32_bytes) = (0usize, 0usize);
        for i in 0..n_layer {
            let l = &layers[i];
            if i >= n_kv_from_start {
                cache_k.push(Vec::new());
                cache_v.push(Vec::new());
                cache_kq.push(Vec::new());
                cache_vq.push(Vec::new());
                continue;
            }
            let cap = if is_swa[i] && n_swa > 0 { (n_swa + FG_BATCH).min(n_ctx) } else { n_ctx };
            cache_cap[i] = cap;
            kv_f32_bytes += 2 * cap * l.kv_dim * 4;
            if kvq_on {
                let hd = l.head_dim;
                let (ks, vs) = (l.n_head_kv * kvq::rec(hd, kbits), l.n_head_kv * kvq::rec(hd, vbits));
                cache_kq.push(vec![0u8; cap * ks]);
                cache_vq.push(vec![0u8; cap * vs]);
                cache_k.push(Vec::new());
                cache_v.push(Vec::new());
                kv_bytes += cap * (ks + vs);
            } else {
                cache_k.push(vec![0.0f32; cap * l.kv_dim]);
                cache_v.push(vec![0.0f32; cap * l.kv_dim]);
                cache_kq.push(Vec::new());
                cache_vq.push(Vec::new());
                kv_bytes += 2 * cap * l.kv_dim * 4;
            }
        }
        if kvq_on {
            println!(
                "[kv] TurboQuant cache: K {kbits}-bit + V {vbits}-bit, {:.0} MB instead of {:.0} MB",
                kv_bytes as f64 / 1048576.0,
                kv_f32_bytes as f64 / 1048576.0
            );
        }

        let mut m = Model {
            arch: arch.clone(),
            v_norm: is_gemma && !is_g3,
            attn_scale,
            rope_scale,
            act_gelu: is_gemma,
            rope_neox: !(arch == "llama" || arch == "mistral"),
            embd_scale: if is_gemma { (n_embd as f32).sqrt() } else { 1.0 },
            n_layer,
            n_embd,
            n_ple,
            n_swa,
            n_kv_from_start,
            n_rot,
            n_rot_swa,
            rope_base,
            rope_base_swa,
            norm_eps,
            final_softcap,
            is_swa,
            kv_donor,
            n_ctx,
            n_vocab,
            n_expert,
            n_expert_used,
            ff_exp,
            tok_embd,
            output,
            output_norm,
            ple_tok_embd,
            ple_model_proj,
            ple_proj_norm,
            rope_freqs,
            cache_k,
            cache_v,
            cache_cap,
            kvq: kvq_on,
            kbits,
            vbits,
            cache_kq,
            cache_vq,
            seq_ids: Vec::with_capacity(n_ctx),
            x: vec![0.0; n_embd],
            cur: vec![0.0; n_embd],
            q: vec![0.0; max_q.max(1)],
            k: vec![0.0; max_kv.max(1)],
            v: vec![0.0; max_kv.max(1)],
            attn_out: vec![0.0; max_q.max(1)],
            ao: vec![0.0; n_embd],
            f: vec![0.0; n_embd],
            gate: vec![0.0; max_ff.max(1)],
            up: vec![0.0; max_ff.max(1)],
            ffn_out: vec![0.0; n_embd],
            ple_in: vec![0.0; n_ple * n_layer],
            ple_t: vec![0.0; n_ple],
            ple_t2: vec![0.0; if n_ple > 0 { n_embd } else { 0 }],
            logits: vec![0.0; n_vocab],
            moe_probs: vec![0.0; n_expert],
            moe_h: vec![0.0; n_expert_used * ff_exp],
            bb: None,
            layers,
            g,
        };

        // GPU: upload the large matrices (everything that goes through matvec) to VRAM. A model
        // larger than VRAM is split: the first layers that fit go to the GPU, the rest stay on the
        // CPU. FASTGGUF_GPU_LAYERS=N sets the layer count (N > layer count = include the output head)
        gpu::init(false);
        if gpu::ready() {
            let (n0, st0, sm0, used0) = gpu::counters();
            let max_layers: Option<usize> = std::env::var("FASTGGUF_GPU_LAYERS").ok().and_then(|v| v.trim().parse().ok());
            let (mut full, mut part) = (0, 0);
            let up = |m: &mut Model, id: Option<usize>| -> (usize, usize) {
                let Some(id) = id else { return (0, 0) };
                let t = &mut m.g.tensors[id];
                if t.gpu.is_none() {
                    t.gpu = gpu::upload(t.typ, t.cols(), t.data);
                }
                (1, t.gpu.is_some() as usize)
            };
            for il in 0..n_layer {
                if max_layers.is_some_and(|n| il >= n) {
                    break;
                }
                let ids = {
                    let l = &m.layers[il];
                    [Some(l.wq), l.wk, l.wv, Some(l.wo), l.ffn_gate, l.ffn_up, l.ffn_down, l.ple_inp_gate, l.ple_proj]
                };
                let (mut present, mut on) = (0, 0);
                for id in ids {
                    let (p, o) = up(&mut m, id);
                    present += p;
                    on += o;
                }
                if on == present {
                    full += 1;
                } else if on > 0 {
                    part += 1;
                }
                if on == 0 {
                    break; // memory is full — no point trying the next layers
                }
            }
            if max_layers.is_none_or(|n| n > n_layer) {
                let (pm, out) = (m.ple_model_proj, m.output);
                up(&mut m, pm);
                up(&mut m, Some(out));
            }
            let (n1, st1, sm1, used1) = gpu::counters();
            let (name, budget) = gpu::device();
            println!(
                "[gpu] offloaded {full}/{n_layer} layers{}{}{}: {} tensors, {:.0} MB on {name} (budget {:.0} MB)",
                if part > 0 { " (+1 partial)" } else { "" },
                if m.g.tensors[m.output].gpu.is_some() { " + output head" } else { "" },
                if n_expert > 0 { " (MoE experts stay on CPU)" } else { "" },
                n1 - n0,
                (used1 - used0) as f64 / 1048576.0,
                budget as f64 / 1048576.0
            );
            let (st, sm) = (st1 - st0, sm1 - sm0);
            if st + sm > 0 {
                println!("[gpu] {} tensors stay on CPU ({st} unsupported type, {sm} vram full)", st + sm);
            }
        }

        // Q4_K and Q4_0 matrices on the CPU re-arranged 8 rows side by side for the generation kernels
        if !env_is("FASTGGUF_REPACK", "0") {
            let t0 = Instant::now();
            let mut ids: Vec<usize> = Vec::new();
            for l in &m.layers {
                ids.extend([Some(l.wq), l.wk, l.wv, Some(l.wo), l.ffn_gate, l.ffn_up, l.ffn_down].into_iter().flatten());
            }
            ids.push(m.output);
            ids.sort_unstable();
            ids.dedup();
            let (mut bytes, mut n) = (0usize, 0usize);
            for id in ids {
                let t = &mut m.g.tensors[id];
                if t.n_dims != 2 || t.rp.is_some() || t.gpu.is_some() {
                    continue;
                }
                if let Some(rp) = repack::repack(t.typ, t.data, t.cols(), t.rows()) {
                    bytes += rp.bytes();
                    n += 1;
                    t.rp = Some(rp);
                }
            }
            if n > 0 {
                println!(
                    "[repack] {n} Q4_K / Q4_0 tensors ({:.0} MB) arranged for the 8-row kernel in {:.1}s",
                    bytes as f64 / 1048576.0,
                    t0.elapsed().as_secs_f64()
                );
            }
        }

        let nswa = m.is_swa.iter().filter(|&&s| s).count();
        println!(
            "[Config] {}: layers={} (swa={}, kv_own={}), embd={}, ple={}, ctx={}, kv_mem={:.0}MB",
            m.arch,
            n_layer,
            nswa,
            n_kv_from_start,
            n_embd,
            n_ple,
            n_ctx,
            kv_bytes as f64 / 1048576.0
        );
        let out_t = &m.g.tensors[m.output];
        println!("[Model] {} tensors, output head: {} ({}), vocab={}", m.g.tensors.len(), out_t.name, type_name(out_t.typ), n_vocab);
        if n_expert > 0 {
            println!(
                "[MoE] experts={n_expert}, used/token={n_expert_used}, ff_exp={} - experts read from mmap on demand",
                m.ff_exp
            );
        }
        if is_g3 {
            println!(
                "[Gemma3] attn_scale={:.4} v_norm=off rope_scale_global={:.4} swa_window={}",
                m.attn_scale, m.rope_scale, n_swa
            );
        }
        Ok(m)
    }

    /// one token at position pos through the whole model; the result is in self.logits
    pub fn forward(&mut self, token: i32, pos: usize) -> Result<(), String> {
        let (n_embd, n_ple, eps) = (self.n_embd, self.n_ple, self.norm_eps);
        let tensors = &self.g.tensors;

        // 1. embedding (× sqrt(n_embd) for Gemma)
        read_embedding(&tensors[self.tok_embd], token, &mut self.x, self.embd_scale)?;

        // 2. per-layer inputs (Gemma 3n / 4)
        if n_ple > 0 {
            let tot = n_ple * self.n_layer;
            read_embedding(&tensors[self.ple_tok_embd.unwrap()], token, &mut self.ple_in[..tot], (n_ple as f32).sqrt())?;
            let mut proj = vec![0.0f32; tot];
            matvec(&tensors[self.ple_model_proj.unwrap()], &self.x, &mut proj)?;
            let ps = 1.0 / (n_embd as f32).sqrt();
            proj.iter_mut().for_each(|v| *v *= ps);
            for chunk in proj.chunks_exact_mut(n_ple) {
                ops::rmsnorm_w(chunk, &self.ple_proj_norm, eps);
            }
            let inv2 = 1.0 / 2.0f32.sqrt();
            for (a, b) in self.ple_in[..tot].iter_mut().zip(&proj) {
                *a = (*a + b) * inv2;
            }
        }

        // 3. the layers
        for il in 0..self.n_layer {
            let l = &self.layers[il];
            let swa = self.is_swa[il];
            let hd = l.head_dim;
            let mut n_rot = if swa { self.n_rot_swa } else { self.n_rot };
            if n_rot == 0 || n_rot > hd {
                n_rot = hd;
            }
            let base = if swa { self.rope_base_swa } else { self.rope_base };
            let fscale = if swa { 1.0 } else { self.rope_scale };
            let ff: Option<&[f32]> = if swa { None } else { self.rope_freqs.as_deref().filter(|f| f.len() >= n_rot / 2) };
            let (q_dim, n_head) = (l.q_dim, l.n_head);

            self.cur.copy_from_slice(&self.x);
            ops::rmsnorm_w(&mut self.cur, &l.attn_norm, eps);

            // Q → bias → per-head norm → RoPE
            let q = &mut self.q[..q_dim];
            matvec(&tensors[l.wq], &self.cur, q)?;
            if let Some(b) = &l.bq {
                q.iter_mut().zip(b).for_each(|(v, b)| *v += b);
            }
            if let Some(qn) = &l.q_norm {
                q.chunks_exact_mut(hd).for_each(|h| ops::rmsnorm_w(h, qn, eps));
            }
            ops::apply_rope(q, pos, hd, n_rot, base, fscale, ff, self.rope_neox);

            // K/V for layers that own their cache entries
            if il < self.n_kv_from_start {
                let kv_dim = l.kv_dim;
                let k = &mut self.k[..kv_dim];
                let v = &mut self.v[..kv_dim];
                matvec(&tensors[l.wk.unwrap()], &self.cur, k)?;
                if let Some(b) = &l.bk {
                    k.iter_mut().zip(b).for_each(|(x, b)| *x += b);
                }
                match l.wv {
                    Some(wv) => {
                        matvec(&tensors[wv], &self.cur, v)?;
                        if let Some(b) = &l.bv {
                            v.iter_mut().zip(b).for_each(|(x, b)| *x += b);
                        }
                    }
                    None => v.copy_from_slice(k), // V = the raw K projection
                }
                for h in 0..l.n_head_kv {
                    if let Some(kn) = &l.k_norm {
                        ops::rmsnorm_w(&mut k[h * hd..(h + 1) * hd], kn, eps);
                    }
                    if self.v_norm {
                        ops::rmsnorm_plain(&mut v[h * hd..(h + 1) * hd], eps);
                    }
                }
                ops::apply_rope(k, pos, hd, n_rot, base, fscale, ff, self.rope_neox);

                let slot = pos % self.cache_cap[il];
                if self.kvq {
                    // TurboQuant: rotate each head slice, then quantize it into the cache record
                    let (rk, rv) = (kvq::rec(hd, self.kbits), kvq::rec(hd, self.vbits));
                    let (ks, vs) = (l.n_head_kv * rk, l.n_head_kv * rv);
                    for h in 0..l.n_head_kv {
                        let kh = &mut k[h * hd..(h + 1) * hd];
                        kvq::rotate(kh);
                        kvq::quant_slice(kh, &mut self.cache_kq[il][slot * ks + h * rk..][..rk], self.kbits);
                        let vh = &mut v[h * hd..(h + 1) * hd];
                        kvq::rotate(vh);
                        kvq::quant_slice(vh, &mut self.cache_vq[il][slot * vs + h * rv..][..rv], self.vbits);
                    }
                } else {
                    self.cache_k[il][slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(k);
                    self.cache_v[il][slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(v);
                }
            }

            // attention — Gemma 4: scale 1 (q/k normalized), Gemma 3: 1/√qpas, others 1/√head_dim
            let qk_scale = if self.attn_scale > 0.0 { self.attn_scale } else { 1.0 / (hd as f32).sqrt() };
            let dl = self.kv_donor[il];
            let (kv_dim, n_head_kv, cap) = (self.layers[dl].kv_dim, self.layers[dl].n_head_kv, self.cache_cap[dl]);
            let start = window_start(swa, self.n_swa, pos, cap);
            let np = pos + 1 - start;

            let kvq_on = self.kvq;
            if kvq_on {
                // rotate q like the keys: the dot product is unchanged (orthogonal rotation)
                q.chunks_exact_mut(hd).for_each(kvq::rotate);
            }
            let q: &[f32] = q;
            let at = Attn {
                kvq: kvq_on,
                kbits: self.kbits,
                vbits: self.vbits,
                hd,
                cap,
                kv_dim,
                n_head_kv,
                n_head,
                qk_scale,
                ck: &self.cache_k[dl],
                cv: &self.cache_v[dl],
                ckq: &self.cache_kq[dl],
                cvq: &self.cache_vq[dl],
            };
            let out = &mut self.attn_out[..q_dim];
            let job = |h: usize, oh: &mut [f32]| at.head(h, &q[h * hd..(h + 1) * hd], oh, start, pos);
            if np * n_head >= 2048 {
                out.par_chunks_mut(hd).enumerate().for_each(|(h, oh)| job(h, oh));
            } else {
                out.chunks_mut(hd).enumerate().for_each(|(h, oh)| job(h, oh));
            }
            if kvq_on {
                // the value sum is in rotated space: rotate it back before the output projection
                out.chunks_exact_mut(hd).for_each(kvq::unrotate);
            }

            // output projection + post-attention norm + residual
            matvec(&tensors[l.wo], &self.attn_out[..q_dim], &mut self.ao)?;
            if let Some(pn) = &l.attn_post_norm {
                ops::rmsnorm_w(&mut self.ao, pn, eps);
            }
            self.ao.iter_mut().zip(&self.x).for_each(|(a, x)| *a += x);

            // FFN — GELU (Gemma), SiLU (the rest) or the MoE experts
            self.f.copy_from_slice(&self.ao);
            ops::rmsnorm_w(&mut self.f, &l.ffn_norm, eps);
            if l.ffn_gate_inp.is_some() {
                moe_ffn(
                    tensors,
                    l,
                    n_embd,
                    self.ff_exp,
                    self.n_expert_used,
                    &self.f,
                    &mut self.ffn_out,
                    &mut self.moe_probs,
                    &mut self.moe_h,
                )?;
            } else {
                let ffd = l.ff_dim;
                let (gate, up) = (&mut self.gate[..ffd], &mut self.up[..ffd]);
                matvec(&tensors[l.ffn_gate.unwrap()], &self.f, gate)?;
                matvec(&tensors[l.ffn_up.unwrap()], &self.f, up)?;
                if self.act_gelu {
                    gate.iter_mut().zip(up.iter()).for_each(|(g, u)| *g = ops::gelu(*g) * u);
                } else {
                    gate.iter_mut().zip(up.iter()).for_each(|(g, u)| *g = ops::silu(*g) * u);
                }
                matvec(&tensors[l.ffn_down.unwrap()], gate, &mut self.ffn_out)?;
            }
            if let Some(pn) = &l.ffn_post_norm {
                ops::rmsnorm_w(&mut self.ffn_out, pn, eps);
            }
            self.ffn_out.iter_mut().zip(&self.ao).for_each(|(o, a)| *o += a);

            // per-layer input
            if n_ple > 0 {
                matvec(&tensors[l.ple_inp_gate.unwrap()], &self.ffn_out, &mut self.ple_t)?;
                let pl = &self.ple_in[il * n_ple..(il + 1) * n_ple];
                self.ple_t.iter_mut().zip(pl).for_each(|(t, p)| *t = ops::gelu(*t) * p);
                matvec(&tensors[l.ple_proj.unwrap()], &self.ple_t, &mut self.ple_t2)?;
                ops::rmsnorm_w(&mut self.ple_t2, l.ple_post_norm.as_ref().unwrap(), eps);
                self.ffn_out.iter_mut().zip(&self.ple_t2).for_each(|(o, p)| *o += p);
            }
            if l.out_scale != 1.0 {
                let s = l.out_scale;
                self.ffn_out.iter_mut().for_each(|v| *v *= s);
            }
            self.x.copy_from_slice(&self.ffn_out);
        }

        // 4. final norm + LM head + soft-capping
        ops::rmsnorm_w(&mut self.x, &self.output_norm, eps);
        matvec(&tensors[self.output], &self.x, &mut self.logits)?;
        if self.final_softcap > 0.0 {
            let cp = self.final_softcap;
            self.logits.iter_mut().for_each(|v| *v = cp * (*v / cp).tanh());
        }
        Ok(())
    }

    /// prompt tokens at positions pos0..; the logits of the last one end up in self.logits.
    /// Batches of FG_BATCH tokens go through each layer together. Gemma 3n/4 per-layer inputs go
    /// token by token, and so does everything with FASTGGUF_BATCH=0 (same results, slower)
    pub fn forward_batch(&mut self, tokens: &[i32], pos0: usize) -> Result<(), String> {
        let off = std::env::var("FASTGGUF_BATCH").map(|v| v == "0" || v == "off").unwrap_or(false);
        if tokens.len() == 1 || off || self.n_ple > 0 {
            for (i, &t) in tokens.iter().enumerate() {
                self.forward(t, pos0 + i)?;
            }
            return Ok(());
        }
        let mut bb = self.bb.take().unwrap_or_else(|| Box::new(BatchBufs::new(self)));
        let mut res = Ok(());
        for (ci, chunk) in tokens.chunks(FG_BATCH).enumerate() {
            let last = (ci + 1) * FG_BATCH >= tokens.len();
            res = self.forward_chunk(&mut bb, chunk, pos0 + ci * FG_BATCH, last);
            if res.is_err() {
                break;
            }
        }
        self.bb = Some(bb);
        res
    }

    /// one batch of nb tokens at positions pos0..pos0+nb-1 (logits of the last one if want_logits)
    fn forward_chunk(&mut self, bb: &mut BatchBufs, toks: &[i32], pos0: usize, want_logits: bool) -> Result<(), String> {
        let nb = toks.len();
        let (e, lq, lkv, lf) = (self.n_embd, bb.lq, bb.lkv, bb.lf);
        let eps = self.norm_eps;
        let tensors = &self.g.tensors;
        for (b, &t) in toks.iter().enumerate() {
            read_embedding(&tensors[self.tok_embd], t, &mut bb.x[b * e..(b + 1) * e], self.embd_scale)?;
        }

        for il in 0..self.n_layer {
            let l = &self.layers[il];
            let swa = self.is_swa[il];
            let hd = l.head_dim;
            let mut n_rot = if swa { self.n_rot_swa } else { self.n_rot };
            if n_rot == 0 || n_rot > hd {
                n_rot = hd;
            }
            let base = if swa { self.rope_base_swa } else { self.rope_base };
            let fscale = if swa { 1.0 } else { self.rope_scale };
            let ff: Option<&[f32]> = if swa { None } else { self.rope_freqs.as_deref().filter(|f| f.len() >= n_rot / 2) };
            let (q_dim, n_head, neox) = (l.q_dim, l.n_head, self.rope_neox);

            bb.cur[..nb * e].copy_from_slice(&bb.x[..nb * e]);
            bb.cur[..nb * e].par_chunks_mut(e).for_each(|c| ops::rmsnorm_w(c, &l.attn_norm, eps));

            // Q → bias → per-head norm → RoPE
            batch::matmul(&tensors[l.wq], &bb.cur, e, &mut bb.q, lq, nb)?;
            bb.q[..nb * lq].par_chunks_mut(lq).enumerate().for_each(|(b, qrow)| {
                let q = &mut qrow[..q_dim];
                if let Some(bias) = &l.bq {
                    q.iter_mut().zip(bias).for_each(|(v, b)| *v += b);
                }
                if let Some(qn) = &l.q_norm {
                    q.chunks_exact_mut(hd).for_each(|h| ops::rmsnorm_w(h, qn, eps));
                }
                ops::apply_rope(q, pos0 + b, hd, n_rot, base, fscale, ff, neox);
            });

            // K/V for owning layers, written into the cache for every token of the batch
            if il < self.n_kv_from_start {
                let kv_dim = l.kv_dim;
                batch::matmul(&tensors[l.wk.unwrap()], &bb.cur, e, &mut bb.k, lkv, nb)?;
                if let Some(wv) = l.wv {
                    batch::matmul(&tensors[wv], &bb.cur, e, &mut bb.v, lkv, nb)?;
                }
                let cap = self.cache_cap[il];
                let (rk, rv) = (kvq::rec(hd, self.kbits), kvq::rec(hd, self.vbits));
                let (ks, vs) = (l.n_head_kv * rk, l.n_head_kv * rv);
                for b in 0..nb {
                    let k = &mut bb.k[b * lkv..b * lkv + kv_dim];
                    let v = &mut bb.v[b * lkv..b * lkv + kv_dim];
                    let pos = pos0 + b;
                    if let Some(bias) = &l.bk {
                        k.iter_mut().zip(bias).for_each(|(x, b)| *x += b);
                    }
                    if l.wv.is_some() {
                        if let Some(bias) = &l.bv {
                            v.iter_mut().zip(bias).for_each(|(x, b)| *x += b);
                        }
                    } else {
                        v.copy_from_slice(k); // V = the raw K projection
                    }
                    for h in 0..l.n_head_kv {
                        if let Some(kn) = &l.k_norm {
                            ops::rmsnorm_w(&mut k[h * hd..(h + 1) * hd], kn, eps);
                        }
                        if self.v_norm {
                            ops::rmsnorm_plain(&mut v[h * hd..(h + 1) * hd], eps);
                        }
                    }
                    ops::apply_rope(k, pos, hd, n_rot, base, fscale, ff, neox);
                    let slot = pos % cap;
                    if self.kvq {
                        for h in 0..l.n_head_kv {
                            let kh = &mut k[h * hd..(h + 1) * hd];
                            kvq::rotate(kh);
                            kvq::quant_slice(kh, &mut self.cache_kq[il][slot * ks + h * rk..][..rk], self.kbits);
                            let vh = &mut v[h * hd..(h + 1) * hd];
                            kvq::rotate(vh);
                            kvq::quant_slice(vh, &mut self.cache_vq[il][slot * vs + h * rv..][..rv], self.vbits);
                        }
                    } else {
                        self.cache_k[il][slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(k);
                        self.cache_v[il][slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(v);
                    }
                }
            }

            // attention: every (token, head) pair is independent; token b sees positions up to pos0+b
            let qk_scale = if self.attn_scale > 0.0 { self.attn_scale } else { 1.0 / (hd as f32).sqrt() };
            let dl = self.kv_donor[il];
            let (kv_dim, n_head_kv, cap) = (self.layers[dl].kv_dim, self.layers[dl].n_head_kv, self.cache_cap[dl]);
            if self.kvq {
                bb.q[..nb * lq].par_chunks_mut(lq).for_each(|r| r[..q_dim].chunks_exact_mut(hd).for_each(kvq::rotate));
            }
            let at = Attn {
                kvq: self.kvq,
                kbits: self.kbits,
                vbits: self.vbits,
                hd,
                cap,
                kv_dim,
                n_head_kv,
                n_head,
                qk_scale,
                ck: &self.cache_k[dl],
                cv: &self.cache_v[dl],
                ckq: &self.cache_kq[dl],
                cvq: &self.cache_vq[dl],
            };
            let (q_all, n_swa) = (&bb.q, self.n_swa);
            bb.att[..nb * lq].par_chunks_mut(lq).enumerate().for_each(|(b, arow)| {
                let pos = pos0 + b;
                let start = window_start(swa, n_swa, pos, cap);
                let qrow = &q_all[b * lq..b * lq + q_dim];
                arow[..q_dim].par_chunks_mut(hd).enumerate().for_each(|(h, oh)| {
                    at.head(h, &qrow[h * hd..(h + 1) * hd], oh, start, pos);
                    if at.kvq {
                        kvq::unrotate(oh);
                    }
                });
            });

            // output projection + post-attention norm + residual, then the FFN norm
            batch::matmul(&tensors[l.wo], &bb.att, lq, &mut bb.ao, e, nb)?;
            let x_all = &bb.x;
            bb.ao[..nb * e].par_chunks_mut(e).zip(bb.f[..nb * e].par_chunks_mut(e)).enumerate().for_each(|(b, (ao, f))| {
                if let Some(pn) = &l.attn_post_norm {
                    ops::rmsnorm_w(ao, pn, eps);
                }
                ao.iter_mut().zip(&x_all[b * e..(b + 1) * e]).for_each(|(a, x)| *a += x);
                f.copy_from_slice(ao);
                ops::rmsnorm_w(f, &l.ffn_norm, eps);
            });

            // FFN — GELU for Gemma, SiLU for the rest; MoE experts token by token
            if l.ffn_gate_inp.is_some() {
                for b in 0..nb {
                    moe_ffn(
                        tensors,
                        l,
                        e,
                        self.ff_exp,
                        self.n_expert_used,
                        &bb.f[b * e..(b + 1) * e],
                        &mut bb.ffn[b * e..(b + 1) * e],
                        &mut self.moe_probs,
                        &mut self.moe_h,
                    )?;
                }
            } else {
                batch::matmul(&tensors[l.ffn_gate.unwrap()], &bb.f, e, &mut bb.gate, lf, nb)?;
                batch::matmul(&tensors[l.ffn_up.unwrap()], &bb.f, e, &mut bb.up, lf, nb)?;
                let (fd, gelu) = (l.ff_dim, self.act_gelu);
                let up = &bb.up;
                bb.gate[..nb * lf].par_chunks_mut(lf).enumerate().for_each(|(b, g)| {
                    let u = &up[b * lf..b * lf + fd];
                    if gelu {
                        g[..fd].iter_mut().zip(u).for_each(|(g, u)| *g = ops::gelu(*g) * u);
                    } else {
                        g[..fd].iter_mut().zip(u).for_each(|(g, u)| *g = ops::silu(*g) * u);
                    }
                });
                batch::matmul(&tensors[l.ffn_down.unwrap()], &bb.gate, lf, &mut bb.ffn, e, nb)?;
            }
            let ao_all = &bb.ao;
            let out_scale = l.out_scale;
            bb.ffn[..nb * e].par_chunks_mut(e).zip(bb.x[..nb * e].par_chunks_mut(e)).enumerate().for_each(|(b, (fo, x))| {
                if let Some(pn) = &l.ffn_post_norm {
                    ops::rmsnorm_w(fo, pn, eps);
                }
                fo.iter_mut().zip(&ao_all[b * e..(b + 1) * e]).for_each(|(o, a)| *o += a);
                if out_scale != 1.0 {
                    fo.iter_mut().for_each(|v| *v *= out_scale);
                }
                x.copy_from_slice(fo);
            });
        }

        if !want_logits {
            return Ok(());
        }
        // final norm + LM head + soft-capping — only the last token's logits are needed
        self.x.copy_from_slice(&bb.x[(nb - 1) * e..nb * e]);
        ops::rmsnorm_w(&mut self.x, &self.output_norm, eps);
        matvec(&tensors[self.output], &self.x, &mut self.logits)?;
        if self.final_softcap > 0.0 {
            let cp = self.final_softcap;
            self.logits.iter_mut().for_each(|v| *v = cp * (*v / cp).tanh());
        }
        Ok(())
    }

    /// how many tokens at the start of ids are already in the KV cache and can be kept
    /// (the longest common prefix with the previous request; at least the last token is re-run)
    pub fn reusable_prefix(&self, ids: &[i32]) -> usize {
        let lim = ids.len().saturating_sub(1).min(self.seq_ids.len());
        let mut keep = 0;
        while keep < lim && ids[keep] == self.seq_ids[keep] {
            keep += 1;
        }
        // sliding-window layers are ring buffers: if the window that is needed has been
        // overwritten, start from scratch
        if keep > 0 && self.n_swa > 0 {
            if let Some(il) = (0..self.n_kv_from_start).find(|&i| self.is_swa[i]) {
                let oldest_needed = (keep + 1).saturating_sub(self.n_swa);
                let oldest_resident = self.seq_ids.len().saturating_sub(self.cache_cap[il]);
                if oldest_needed < oldest_resident {
                    keep = 0;
                }
            }
        }
        keep
    }
}

/// the first position a query at pos attends to
fn window_start(swa: bool, n_swa: usize, pos: usize, cap: usize) -> usize {
    let mut start = 0;
    if swa && n_swa > 0 && pos + 1 > n_swa {
        start = pos + 1 - n_swa;
    }
    if pos + 1 - start > cap {
        start = pos + 1 - cap; // never past the donor cache
    }
    start
}

/// what one layer's attention reads: the donor layer's cache and the shapes
struct Attn<'a> {
    kvq: bool,
    kbits: usize,
    vbits: usize,
    hd: usize,
    cap: usize,
    kv_dim: usize,
    n_head_kv: usize,
    n_head: usize,
    qk_scale: f32,
    ck: &'a [f32],
    cv: &'a [f32],
    ckq: &'a [u8],
    cvq: &'a [u8],
}

impl Attn<'_> {
    /// oh = softmax(q·K / scale) · V over positions start..=pos for query head h
    fn head(&self, h: usize, qh: &[f32], oh: &mut [f32], start: usize, pos: usize) {
        let (hd, cap, kv_dim) = (self.hd, self.cap, self.kv_dim);
        let kvh = h * self.n_head_kv / self.n_head;
        let (rk, rv) = (kvq::rec(hd, self.kbits), kvq::rec(hd, self.vbits));
        let np = pos + 1 - start;
        SCORES.with(|s| {
            let mut s = s.borrow_mut();
            s.resize(np, 0.0);
            for p in start..=pos {
                let slot = p % cap;
                let d = if self.kvq {
                    kvq::dot_k(qh, &self.ckq[slot * self.n_head_kv * rk + kvh * rk..][..rk], self.kbits)
                } else {
                    dot_f32(qh, &self.ck[slot * kv_dim + kvh * hd..][..hd])
                };
                s[p - start] = d * self.qk_scale;
            }
            ops::softmax(&mut s[..np]);
            oh.fill(0.0);
            for p in start..=pos {
                let w = s[p - start];
                if w == 0.0 {
                    continue;
                }
                let slot = p % cap;
                if self.kvq {
                    kvq::axpy_v(oh, w, &self.cvq[slot * self.n_head_kv * rv + kvh * rv..][..rv], self.vbits);
                } else {
                    axpy(oh, w, &self.cv[slot * kv_dim + kvh * hd..][..hd]);
                }
            }
        });
    }
}

/// prompt-batch buffers: FG_BATCH rows each (allocated on first use)
struct BatchBufs {
    lq: usize,
    lkv: usize,
    lf: usize,
    x: Vec<f32>,
    cur: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    ao: Vec<f32>,
    f: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn: Vec<f32>,
}

impl BatchBufs {
    fn new(m: &Model) -> BatchBufs {
        let lq = m.layers.iter().map(|l| l.q_dim).max().unwrap_or(1).max(1);
        let lkv = m.layers.iter().map(|l| l.kv_dim).max().unwrap_or(1).max(1);
        let lf = m.layers.iter().map(|l| l.ff_dim).max().unwrap_or(1).max(1);
        let (b, e) = (FG_BATCH, m.n_embd);
        BatchBufs {
            lq,
            lkv,
            lf,
            x: vec![0.0; b * e],
            cur: vec![0.0; b * e],
            q: vec![0.0; b * lq],
            k: vec![0.0; b * lkv],
            v: vec![0.0; b * lkv],
            att: vec![0.0; b * lq],
            ao: vec![0.0; b * e],
            f: vec![0.0; b * e],
            gate: vec![0.0; b * lf],
            up: vec![0.0; b * lf],
            ffn: vec![0.0; b * e],
        }
    }
}

/// one embedding row × scale
fn read_embedding(t: &Tensor, row: i32, out: &mut [f32], scale: f32) -> Result<(), String> {
    if row < 0 || row as usize >= t.rows() {
        return Err(format!("token {row} is outside the vocabulary"));
    }
    let stride = row_stride(t.typ, t.ne[0]) as usize;
    let r = row as usize;
    if !dequantize_row(t.typ, &t.data[r * stride..(r + 1) * stride], out) {
        return Err(format!("embedding: unsupported type {}", type_name(t.typ)));
    }
    if scale != 1.0 {
        out.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(())
}

/// FFN with experts (llama.cpp build_moe_ffn for qwen3moe): router → softmax → top-k experts →
/// renormalize their weights → Σ w · down(SiLU(gate·x) · up·x). Row r of expert e starts at
/// (e·rows + r)·stride inside the 3D tensor — no copies.
#[allow(clippy::too_many_arguments)]
fn moe_ffn(
    tensors: &[Tensor],
    l: &Layer,
    n_embd: usize,
    ff: usize,
    k: usize,
    x: &[f32],
    out: &mut [f32],
    probs: &mut [f32],
    h: &mut [f32],
) -> Result<(), String> {
    matvec(&tensors[l.ffn_gate_inp.unwrap()], x, probs)?;
    ops::softmax(probs);
    let mut sel = vec![0usize; k];
    let mut w = vec![0.0f32; k];
    let mut wsum = 0.0f32;
    for j in 0..k {
        let e = ops::argmax(probs);
        sel[j] = e;
        w[j] = probs[e];
        wsum += probs[e];
        probs[e] = -1.0; // probabilities are ≥ 0: it will not be picked again
    }
    let wsum = wsum.max(6.103_515_6e-5); // same limit as llama.cpp
    w.iter_mut().for_each(|v| *v /= wsum);

    let (tg, tu, td) = (&tensors[l.gate_exps.unwrap()], &tensors[l.up_exps.unwrap()], &tensors[l.down_exps.unwrap()]);
    let x = &x[..n_embd];
    let (sg, su) = (row_stride(tg.typ, n_embd as u64) as usize, row_stride(tu.typ, n_embd as u64) as usize);
    let (qg, qu) = (act_kind(tg.typ, n_embd), act_kind(tu.typ, n_embd));
    let mut a = QAct::new(qg, x);
    a.add(qu, x);
    let fail = AtomicBool::new(false);
    let ch = 16;
    // gate/up: the same input for every expert — the rows of all selected experts in one loop
    h[..k * ff].par_chunks_mut(ch).enumerate().for_each(|(ci, o)| {
        for (i, v) in o.iter_mut().enumerate() {
            let r = ci * ch + i;
            let row = sel[r / ff] * ff + r % ff;
            let gv = dot_row(tg.typ, qg, &tg.data[row * sg..(row + 1) * sg], &a, x);
            let uv = dot_row(tu.typ, qu, &tu.data[row * su..(row + 1) * su], &a, x);
            match (gv, uv) {
                (Some(g), Some(u)) => *v = ops::silu(g) * u,
                _ => fail.store(true, Ordering::Relaxed),
            }
        }
    });

    // down: a different input per expert — each output row sums k weighted experts
    let sd = row_stride(td.typ, ff as u64) as usize;
    let qd = act_kind(td.typ, ff);
    let h: &[f32] = h;
    let acts: Vec<QAct> = (0..k).map(|j| QAct::new(qd, &h[j * ff..(j + 1) * ff])).collect();
    out[..n_embd].par_chunks_mut(ch).enumerate().for_each(|(ci, o)| {
        for (i, v) in o.iter_mut().enumerate() {
            let oi = ci * ch + i;
            let mut s = 0.0f32;
            for j in 0..k {
                let row = sel[j] * n_embd + oi;
                match dot_row(td.typ, qd, &td.data[row * sd..(row + 1) * sd], &acts[j], &h[j * ff..(j + 1) * ff]) {
                    Some(d) => s += w[j] * d,
                    None => fail.store(true, Ordering::Relaxed),
                }
            }
            *v = s;
        }
    });
    if fail.load(Ordering::Relaxed) {
        return Err(format!(
            "moe: unsupported expert type {}/{}/{}",
            type_name(tg.typ),
            type_name(tu.typ),
            type_name(td.typ)
        ));
    }
    Ok(())
}

/// what `inspect` prints about a MoE file: dense bytes, expert bytes and bytes read per token
pub fn moe_summary(g: &Gguf, arch: &str) -> Option<String> {
    let n_exp = g.num(&format!("{arch}.expert_count"), 0.0) as i64;
    if n_exp <= 0 {
        return None;
    }
    let (mut exp_b, mut all_b) = (0u64, 0u64);
    for t in &g.tensors {
        all_b += t.nbytes;
        if t.name.contains("_exps.") {
            exp_b += t.nbytes;
        }
    }
    let n_used = g.num(&format!("{arch}.expert_used_count"), 0.0) as i64;
    Some(format!(
        "MoE: experts={n_exp}, used/token={n_used}, ff_exp={}\n     dense weights {:.2} GB (always read) + experts {:.2} GB (~{:.2} GB read per token)",
        g.num(&format!("{arch}.expert_feed_forward_length"), 0.0) as i64,
        (all_b - exp_b) as f64 / 1e9,
        exp_b as f64 / 1e9,
        exp_b as f64 / 1e9 * n_used as f64 / n_exp as f64
    ))
}
