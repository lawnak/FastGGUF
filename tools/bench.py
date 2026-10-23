#!/usr/bin/env python3
"""Measure fastgguf decode speed (tokens/second) on this machine.

Uses synthetic models (real shapes, random weights - see synth_gguf.py), so
nothing has to be downloaded. You can also pass real .gguf files.

    python tools/bench.py                          # default set
    python tools/bench.py qwen2.5-0.5b:q4_0 llama-3.2-1b:q4_k_m
    python tools/bench.py --gguf path/to/model.gguf
    OMP_NUM_THREADS=2 python tools/bench.py ...    # limit threads
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = ["qwen2.5-0.5b:q4_0", "qwen2.5-0.5b:q8_0", "llama-3.2-1b:q4_k_m",
           "qwen2.5-1.5b:q4_k_m", "llama-3.2-3b:q4_k_m"]
PROMPT = "Write a short paragraph about the history of computing and why fast local inference matters."


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_once(binary, model, ctx, n_gen, env):
    p = subprocess.run([binary, "chat", str(model), str(ctx), "--temp", "0", "--max-tokens", str(n_gen)],
                       input=PROMPT + "\n", capture_output=True, text=True, errors="replace", env=env, timeout=3600)
    out = p.stdout + p.stderr
    m = re.search(r"\[prompt: ([\d.]+)s \(([\d.]+) tok/s\) \| gen: (\d+) tok @ ([\d.]+) tok/s\]", out)
    if not m:
        sys.exit(f"could not parse output of {model}:\n{out[-2000:]}")
    gpu = re.search(r"\[gpu\] using (.+?) \(", out)
    return {"prompt_tok_s": float(m.group(2)), "gen_tok_s": float(m.group(4)),
            "gen_tokens": int(m.group(3)), "gpu": gpu.group(1) if gpu else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="preset:quant (see synth_gguf.py --list)")
    ap.add_argument("--gguf", action="append", default=[], help="benchmark a real .gguf file")
    ap.add_argument("--bin", default=str(HERE.parent / ("fastgguf.exe" if os.name == "nt" else "fastgguf")))
    ap.add_argument("--models", default=str(HERE.parent / "models" / "synthetic"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=128, help="tokens to generate per run")
    ap.add_argument("--runs", type=int, default=2, help="runs per model, best is kept")
    ap.add_argument("--json", help="also write results to this file")
    a = ap.parse_args()

    mdir = Path(a.models)
    mdir.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(g).stem, Path(g)) for g in a.gguf]
    for spec in (a.specs or ([] if a.gguf else DEFAULT)):
        preset, quant = spec.split(":")
        path = mdir / f"{preset}-{quant}.gguf"
        if not path.exists():
            print(f"generating {path.name} ...", flush=True)
            subprocess.run([sys.executable, str(HERE / "synth_gguf.py"), preset, quant, str(path)], check=True)
        jobs.append((spec, path))

    env = dict(os.environ)
    threads = env.get("OMP_NUM_THREADS") or str(os.cpu_count())
    print(f"\nCPU: {cpu_name()} | threads: {threads} | ctx {a.ctx} | {a.tokens} tokens x {a.runs} runs\n")
    print(f"| {'model':28s} | {'size':>8s} | {'gen tok/s':>9s} |")
    print(f"|{'-' * 30}|{'-' * 10}|{'-' * 11}|")
    results = []
    for name, path in jobs:
        best = max((run_once(a.bin, path, a.ctx, a.tokens, env) for _ in range(a.runs)),
                   key=lambda r: r["gen_tok_s"])
        size_gb = path.stat().st_size / 1e9
        print(f"| {name:28s} | {size_gb:6.2f}GB | {best['gen_tok_s']:9.2f} |", flush=True)
        results.append({"model": name, "file_gb": round(size_gb, 3), "threads": int(threads), **best})
    if a.json:
        Path(a.json).write_text(json.dumps({"cpu": cpu_name(), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
