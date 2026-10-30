<p align="center">
  <img src="../docs/images/logo-rust.webp" alt="fastgguf Rust logo" width="400">
</p>

# fastgguf — Rust version

A Rust port of [`fastgguf.c`](../fastgguf.c): runs GGUF language models on your own computer, on
the CPU and on NVIDIA, AMD or Intel GPUs through OpenCL. Every feature of the C version is ported:
the engine, terminal chat, the server (web UI + OpenAI-compatible API), the GPU, and the
AVX2 / AVX-VNNI / AVX-512 VNNI kernels.

```
fastgguf chat     model.gguf [ctx] [--temp 1.0] [--top-k 64] [--top-p 0.95]
                  [--repeat-penalty 1.0] [--repeat-last-n 64] [--max-tokens 512] [--seed N]
fastgguf serve    model.gguf [model2.gguf ... up to 4] [port 8080] [ctx 4096]
fastgguf inspect  model.gguf
fastgguf dump     model.gguf
fastgguf gpu      list GPUs / check OpenCL support
fastgguf selftest
```

## What is ported

| Part | Status |
|---|---|
| GGUF reader (mmap), `inspect`, `dump` | done — output identical to the C version |
| All quant types: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS | done |
| CPU kernels: AVX2, AVX-VNNI, AVX-512 VNNI — all in one binary, the best one picked at start-up | done |
| Q4_K / Q4_0 8-row repacked kernels | done |
| Gemma 2/3/3n/4, Qwen2/3, Qwen3 MoE, Llama/Mistral forward pass | done |
| Tokenizer (SPM + BPE), chat templates, sampler, terminal chat | done |
| TurboQuant KV cache (`FASTGGUF_KV=turbo…`) | done |
| Batched prompt reading: int8 GEMM for Q4_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1 (+ Q8_0 with VNNI), tiles for the rest | done |
| `serve`: web UI, `/v1/chat/completions` (stream + non-stream), `/v1/models`, up to 4 models, parallel / chain / loop modes, prompt-cache reuse, `FASTGGUF_API_KEY` | done |
| GPU (OpenCL, loaded at run time): F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K IQ4_NL, self-check before use, models larger than VRAM split between GPU and CPU, `gpu` command | done |
| `selftest`: every kernel on every kernel set the CPU has | done |

### How it was checked

On `gemma-3-1b-it.Q2_K.gguf`, against the C release, on an Intel i5-1135G7 with an NVIDIA MX450:

- with the portable kernels (`FASTGGUF_CPU=sse2` for C, `generic` for Rust) both print the same
  greedy reply token for token — token by token, batched, and with the GPU on;
- `inspect` / `dump` print exactly the same text, and the GPU offload report is the same
  (131 tensors, 554 MB);
- with AVX the results differ only in float rounding.

Speed on that machine, CPU only, 8 threads (prompt / generation, tok/s, three runs each):
C 47–56 / 21–24, Rust 52–67 / 21–25 — prompt reading is faster than C, generation the same.
On one thread Rust generates ~8% slower. With the GPU both versions run at ~15 / 9: the MX450 is
a small GPU and this model's IQ4_NL/Q3_K weights partly stay on the CPU, so every layer goes back
and forth. Use `FASTGGUF_GPU=off` when the CPU is faster.

What made the kernels fast (worth knowing before changing `simd.rs`):

- **fp16 scales are loaded with `vmovd` through inline asm** (`h2fp`). Written with intrinsics,
  LLVM folds the 16-bit load into a `vpinsrw` that merges into a register it picks, often an
  accumulator; every block then waits for the previous one. This alone halved the speed of the
  32-value kernels.
- **Two accumulators** in the 32-value kernels (Q4_0, Q8_0, Q5_0, IQ4_NL) and in Q2_K/Q3_K, so
  consecutive blocks do not wait for each other's FMA.
- **No serial float sums**: Rust keeps float order strict, so a scalar `sum +=` per sub-block
  becomes a chain of dependent additions. Q2_K/Q3_K collect them per block and add them as one vector.
- **`cvtss2si` for rounding** in the activation quantizer (`quant::rnd`): `round_ties_even()`
  is a library call per value on the baseline x86-64 target.

Differences from the C server: requests are parsed with a real JSON parser (`content` may also be
a list of text parts), and a streamed reply never splits a UTF-8 character across two chunks.

## Build

```sh
cargo build --release          # → target/release/fastgguf(.exe)
./target/release/fastgguf selftest
```

Two crates: `rayon` (thread pool) and `memmap2` (memory-mapped file). The binary itself has no
runtime dependencies; the GPU driver's OpenCL library is loaded only if it is there.

For the fastest build on one machine (not portable to older CPUs):

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

**Windows without Visual Studio C++ tools:** use the GNU toolchain, which brings its own linker:

```sh
rustup toolchain install stable-x86_64-pc-windows-gnu --profile minimal
rustup override set stable-x86_64-pc-windows-gnu   # run inside this folder
```

Build from PowerShell or cmd, not Git Bash: Git Bash's `link` command hides the MSVC linker.

## Settings

| Variable | Effect |
|---|---|
| `FASTGGUF_GPU=auto` / `on` / `off` | GPU: discrete only (default) / integrated too / never |
| `FASTGGUF_GPU_DEVICE=<n>` | use device n from the `gpu` list |
| `FASTGGUF_GPU_MEM=<GB>` | cap the GPU weight memory (default ~87% of VRAM) |
| `FASTGGUF_GPU_LAYERS=<n>` | offload exactly n layers (n > layer count: the output head too) |
| `FASTGGUF_KV=turbo` / `turbo8` / `turbo4` / `turbo2` | TurboQuant compressed KV cache |
| `FASTGGUF_CPU=generic` / `avx2` / `avxvnni` / `avx512` | force a CPU kernel set |
| `FASTGGUF_REPACK=0` | do not re-arrange Q4_K / Q4_0 weights |
| `FASTGGUF_BATCH=0` | read prompts token by token (same results, slower) |
| `FASTGGUF_API_KEY=<key>` | the server requires `Authorization: Bearer <key>` |
| `RAYON_NUM_THREADS=<n>` | number of threads |

## Layout

```
src/main.rs       command line
src/cli.rs        chat / inspect / dump
src/gguf.rs       GGUF reader
src/quant.rs      types, fp16, dequantization, portable dot kernels
src/simd.rs       AVX2 / AVX-VNNI / AVX-512 kernels (one template, built 3 times) + CPU detection
src/repack.rs     Q4_K / Q4_0 8-row layout
src/matvec.rs     multi-threaded matrix × vector
src/batch.rs      prompt batches: matrix × matrix (GEMM)
src/gpu.rs        OpenCL: runtime loading, kernels, upload, self-check
src/server.rs     HTTP server, OpenAI API, web UI
src/json.rs       JSON reader / string escaping
src/model.rs      model loading and forward pass (incl. MoE)
src/ops.rs        RMSNorm, RoPE, softmax, TurboQuant
src/tokenizer.rs  tokenizer + chat templates
src/sampler.rs    temperature / top-k / top-p / repeat penalty
src/selftest.rs   kernel and sampler tests
```
