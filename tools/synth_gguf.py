#!/usr/bin/env python3
"""Write a GGUF file with the exact shapes of a real model but random weights.

Decode speed depends only on tensor shapes and quant types, not on the values,
so these files give honest tok/s numbers without downloading a real model.
The text they generate is gibberish - use real models for real answers.

    pip install gguf numpy
    python tools/synth_gguf.py qwen2.5-0.5b q4_0 out.gguf
    python tools/synth_gguf.py --list
"""
import argparse
import sys

import numpy as np
from gguf import GGMLQuantizationType as Q, GGUFWriter, TokenType

# Shapes copied from the official config.json of each model.
PRESETS = {
    #  name             arch     embd  layers heads kv  head_dim  ffn    vocab   tied  bias
    "qwen2.5-0.5b":   ("qwen2",   896,  24,   14,   2,   64,     4864, 151936, True,  True),
    "qwen2.5-1.5b":   ("qwen2",  1536,  28,   12,   2,  128,     8960, 151936, True,  True),
    "qwen2.5-3b":     ("qwen2",  2048,  36,   16,   2,  128,    11008, 151936, True,  True),
    "qwen2.5-7b":     ("qwen2",  3584,  28,   28,   4,  128,    18944, 152064, False, True),
    "llama-3.2-1b":   ("llama",  2048,  16,   32,   8,   64,     8192, 128256, True,  False),
    "llama-3.2-3b":   ("llama",  3072,  28,   24,   8,  128,     8192, 128256, True,  False),
    "llama-3.1-8b":   ("llama",  4096,  32,   32,   8,  128,    14336, 128256, False, False),
    "qwen3-4b":       ("qwen3",  2560,  36,   32,   8,  128,     9728, 151936, True,  False),
    "gemma-3-1b":     ("gemma3", 1152,  26,    4,   1,  256,     6912, 262144, True,  False),
    "gemma-3-4b":     ("gemma3", 2560,  34,    8,   4,  256,    10240, 262144, True,  False),
    "tiny-test":      ("llama",   256,   2,    4,   2,   64,      512,   1024, True,  False),
}

# bytes per block, elements per block, byte offsets of fp16 scale fields
BLOCK = {
    Q.Q4_0: (18, 32, [0]),
    Q.Q5_0: (22, 32, [0]),
    Q.Q8_0: (34, 32, [0]),
    Q.Q4_K: (144, 256, [0, 2]),
    Q.Q6_K: (210, 256, [208]),
}

# quant "recipes": type for (most weights, output head / ffn_down / attn_v)
RECIPES = {
    "q4_0": (Q.Q4_0, Q.Q4_0),
    "q8_0": (Q.Q8_0, Q.Q8_0),
    "q4_k_m": (Q.Q4_K, Q.Q6_K),   # close to llama.cpp Q4_K_M
}

rng = np.random.default_rng(1234)


# llama.cpp falls back to these when a row is not a multiple of 256
FALLBACK = {Q.Q4_K: Q.Q5_0, Q.Q6_K: Q.Q8_0}


def qtype_for(cols, qtype):
    return FALLBACK[qtype] if BLOCK[qtype][1] == 256 and cols % 256 else qtype


def add_q(w, name, rows, cols, qtype):
    qtype = qtype_for(cols, qtype)
    w.add_tensor(name, quant_tensor(rows, cols, qtype), raw_dtype=qtype)


def quant_tensor(rows, cols, qtype):
    """Random quantized blocks with small, finite fp16 scales."""
    bb, be, scale_offs = BLOCK[qtype]
    assert cols % be == 0, (cols, be)
    nblk = rows * (cols // be)
    raw = rng.integers(0, 256, size=(nblk, bb), dtype=np.uint8)
    if qtype == Q.Q6_K:
        raw[:, 192:208] = rng.integers(1, 16, size=(nblk, 16), dtype=np.uint8)  # int8 scales
    if qtype == Q.Q4_K:
        raw[:, 4:16] &= 0x3F
    scale = np.full(nblk, 0.0004 if qtype in (Q.Q4_K, Q.Q6_K) else 0.004, dtype=np.float16)
    sb = scale.view(np.uint8).reshape(nblk, 2)
    for off in scale_offs:
        raw[:, off:off + 2] = sb
    return raw.reshape(rows, (cols // be) * bb)


def f32(n, val=None):
    if val is not None:
        return np.full(n, val, dtype=np.float32)
    return (rng.standard_normal(n) * 0.02).astype(np.float32)


def vocab(arch, n):
    """A vocab that the engine can tokenize: specials + byte tokens + filler."""
    if arch.startswith("gemma"):
        specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
        toks = specials + [f"<0x{b:02X}>" for b in range(256)] + ["▁"]
        model = "llama"
    else:
        if arch == "llama":
            specials = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>",
                        "<|end_header_id|>", "<|eot_id|>"]
        else:
            specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        # GPT-2 byte map: printable bytes stay, the rest shift to 256+
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs, k = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + k); k += 1
        toks = specials + [chr(c) for c in cs]
        model = "gpt2"
    toks += [f"tok{i:06d}" for i in range(n - len(toks))]
    types = [TokenType.CONTROL if t in specials else TokenType.NORMAL for t in toks]
    return model, toks, types, specials


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?")
    ap.add_argument("quant", nargs="?", choices=sorted(RECIPES))
    ap.add_argument("out", nargs="?")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.preset and a.quant and a.out):
        for k, v in PRESETS.items():
            print(f"{k:14s} arch={v[0]:7s} embd={v[1]} layers={v[2]} vocab={v[7]}")
        return 0 if a.list else 1

    arch, embd, nl, nh, nkv, hd, ff, nv, tied, bias = PRESETS[a.preset]
    main_t, hi_t = RECIPES[a.quant]
    gemma = arch.startswith("gemma")
    w = GGUFWriter(a.out, arch)
    w.add_name(f"synthetic-{a.preset}-{a.quant}")
    w.add_block_count(nl)
    w.add_context_length(32768)
    w.add_embedding_length(embd)
    w.add_feed_forward_length(ff)
    w.add_head_count(nh)
    w.add_head_count_kv(nkv)
    w.add_key_length(hd)
    w.add_value_length(hd)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1e6 if gemma or arch.startswith("qwen") else 500000.0)
    if gemma:
        w.add_sliding_window(512)
        if embd > 2000:  # Gemma 3 4B and up use linear RoPE scaling on global layers
            w.add_float32(f"{arch}.rope.scaling.factor", 8.0)

    model, toks, types, specials = vocab(arch, nv)
    w.add_tokenizer_model(model)
    w.add_token_list(toks)
    w.add_token_types(types)
    if model == "gpt2":  # llama.cpp refuses BPE vocabs without merges (fastgguf ignores them)
        w.add_tokenizer_pre("llama-bpe" if arch == "llama" else "qwen2")
        w.add_token_merges(["Ġ t"])
    else:
        w.add_token_scores([0.0] * len(toks))
    if gemma:
        w.add_bos_token_id(2); w.add_eos_token_id(1)
    elif arch == "llama":
        w.add_bos_token_id(0); w.add_eos_token_id(4)
    else:
        w.add_eos_token_id(2)

    # Tied models: token_embd doubles as the output head, so it gets the higher-quality type
    add_q(w, "token_embd.weight", nv, embd, hi_t if tied else main_t)
    if not tied:
        add_q(w, "output.weight", nv, embd, hi_t)
    w.add_tensor("output_norm.weight", f32(embd, 1.0))

    q_dim, kv_dim = nh * hd, nkv * hd
    for i in range(nl):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", f32(embd, 1.0))
        add_q(w, p + "attn_q.weight", q_dim, embd, main_t)
        add_q(w, p + "attn_k.weight", kv_dim, embd, main_t)
        vt = hi_t if i % 2 == 0 else main_t
        add_q(w, p + "attn_v.weight", kv_dim, embd, vt)
        add_q(w, p + "attn_output.weight", embd, q_dim, main_t)
        if bias:
            w.add_tensor(p + "attn_q.bias", f32(q_dim))
            w.add_tensor(p + "attn_k.bias", f32(kv_dim))
            w.add_tensor(p + "attn_v.bias", f32(kv_dim))
        if gemma or arch == "qwen3":
            w.add_tensor(p + "attn_q_norm.weight", f32(hd, 1.0))
            w.add_tensor(p + "attn_k_norm.weight", f32(hd, 1.0))
        if gemma:
            w.add_tensor(p + "post_attention_norm.weight", f32(embd, 1.0))
            w.add_tensor(p + "post_ffw_norm.weight", f32(embd, 1.0))
        w.add_tensor(p + "ffn_norm.weight", f32(embd, 1.0))
        add_q(w, p + "ffn_gate.weight", ff, embd, main_t)
        add_q(w, p + "ffn_up.weight", ff, embd, main_t)
        dt = hi_t if i % 2 == 0 else main_t
        add_q(w, p + "ffn_down.weight", embd, ff, dt)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
