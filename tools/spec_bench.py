#!/usr/bin/env python3
"""Measure speculative decoding on a real model: tokens/second with it off and on, for 7 kinds of prompt.

Synthetic models (synth_gguf.py) are no use here: random weights make a model repeat itself, so
prompt lookup would guess almost everything. Pass a real instruction-tuned .gguf file.

    python tools/spec_bench.py model.gguf                 # greedy (the text must be the same both ways)
    python tools/spec_bench.py model.gguf --temp 0.7      # sampling (the text differs run to run)
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = ("The Apollo program was the third United States human spaceflight program carried out by NASA, which "
       "succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived "
       "during Dwight D. Eisenhower's administration as a three-person spacecraft to follow the one-person Project "
       "Mercury, which put the first Americans in space. Apollo was later dedicated to President John F. Kennedy's "
       "national goal for the 1960s of landing a man on the Moon and returning him safely to the Earth. Apollo ran "
       "from 1961 to 1972, with the first crewed flight in 1968. It encountered a major setback in 1967 when an "
       "Apollo 1 cabin fire killed the entire crew during a prelaunch test. After the first Moon landing, "
       "sufficient flight hardware remained for nine follow-on Moon landings with a plan for extended lunar "
       "geological and astrophysical exploration.")
CODE = ("def parse_config(path):\n    with open(path) as f:\n        lines = f.readlines()\n    cfg = {}\n"
        "    for line in lines:\n        line = line.strip()\n        if not line or line.startswith('#'):\n"
        "            continue\n        key, value = line.split('=')\n        cfg[key.strip()] = value.strip()\n"
        "    return cfg\n").replace("\n", " \\n ")   # the chat reads one line per message
PROMPTS = {
    "story":    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain":  "Explain how a hash table works and when to use one.",
    "code":     "Write a Python class for a bank account with deposit, withdraw and balance methods, with docstrings.",
    "rewrite":  "Rewrite the following text in simpler English, keeping all the facts: " + DOC,
    "qa":       "Read this text and list every year it mentions, with what happened in that year: " + DOC,
    "refactor": "Add type hints and comments to this Python code and return the full code: " + CODE,
    "json":     "Convert to a JSON array of objects with fields name, age, city: Alice 31 Paris; Bob 25 London; "
                "Carol 40 Berlin; Dave 35 Madrid; Eve 28 Rome; Frank 52 Vienna.",
}


def run(binary, model, prompt, spec, temp, ntok, ctx):
    env = dict(os.environ, FASTGGUF_SPEC=str(spec))
    p = subprocess.run([binary, "chat", str(model), str(ctx), "--temp", str(temp), "--seed", "7",
                        "--max-tokens", str(ntok)], input=prompt + "\n", capture_output=True, text=True,
                       errors="replace", env=env, timeout=3600)
    out = p.stdout
    m = re.search(r"gen: (\d+) tok @ ([\d.]+) tok/s", out)
    if not m:
        sys.exit(f"could not parse the output:\n{out[-2000:]}")
    a = re.search(r"speculative: (\d+) of (\d+) drafted tokens accepted in (\d+) checks", out)
    text = out.split("> ", 1)[-1].rsplit("[prompt:", 1)[0]
    return {"tok_s": float(m.group(2)), "tokens": int(m.group(1)), "text": text,
            "accepted": int(a.group(1)) if a else 0, "drafted": int(a.group(2)) if a else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--bin", default=str(HERE.parent / ("fastgguf.exe" if os.name == "nt" else "fastgguf")))
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--runs", type=int, default=2, help="runs per prompt and setting, best is kept")
    a = ap.parse_args()

    t_off = t_on = 0.0
    print(f"| {'prompt':9s} | {'off tok/s':>9s} | {'on tok/s':>8s} | speed-up | accepted | text |")
    print("|-----------|-----------|----------|----------|----------|------|")
    for name, prompt in PROMPTS.items():
        off = max((run(a.bin, a.model, prompt, 0, a.temp, a.tokens, a.ctx) for _ in range(a.runs)),
                  key=lambda r: r["tok_s"])
        on = max((run(a.bin, a.model, prompt, 8, a.temp, a.tokens, a.ctx) for _ in range(a.runs)),
                 key=lambda r: r["tok_s"])
        same = "-" if a.temp > 0 else ("same" if on["text"] == off["text"] else "DIFFERENT")
        print(f"| {name:9s} | {off['tok_s']:9.1f} | {on['tok_s']:8.1f} | {on['tok_s'] / off['tok_s']:7.2f}x "
              f"| {on['accepted']:3d}/{on['drafted']:<4d} | {same} |", flush=True)
        t_off += off["tokens"] / off["tok_s"]
        t_on += on["tokens"] / on["tok_s"]
    print(f"\nall prompts together: {t_off / t_on:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
