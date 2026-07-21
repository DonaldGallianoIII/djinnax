#!/usr/bin/env python3
"""Statistical sweep: N independent fresh-process bench runs, then median +
spread of per-run env-steps/s AND per-run ratios (the within-run ratio is
the noise-robust quantity on this host — clocks drift between runs).

Runs are strictly sequential (never parallel JAX on this box). Each run is
a new process so compile caches, allocator state, and clock ramps count as
independent samples.

Usage:
    python sweep_stats.py --n 5 --batches 64 1024 8192 65536 [--unroll 1 --rng threefry2x32]
    python sweep_stats.py --aggregate-only sweep_results.jsonl
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PY = Path(sys.executable)   # current interpreter; set LD_PRELOAD yourself (HOW_TO_RUN.md)

RATIO_PAIRS = [
    ("ttt", "ttt/djinn", "ttt/pgx"),
    ("ttt-bb", "ttt/djinn-bb", "ttt/pgx"),
    ("2048", "2048/djinn", "2048/jumanji"),
    ("2048-lut", "2048/djinn-lut", "2048/jumanji"),
    ("2048-mega", "2048/djinn-mega", "2048/jumanji"),
    ("soko", "soko/djinn", "soko/jumanji"),
]


def run_sweep(n, batches, unroll, rng, out_path):
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    for i in range(n):
        run_file = f"{out_path}.run{i}"
        Path(run_file).unlink(missing_ok=True)
        cmd = [str(VENV_PY), str(HERE / "bench_head_to_head.py"),
               "--batches", *map(str, batches),
               "--unroll", str(unroll), "--rng", rng,
               "--json", run_file]
        print(f"=== sweep run {i + 1}/{n} ===", flush=True)
        r = subprocess.run(cmd, env=env, cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            sys.exit(f"run {i} failed")
        # tag records with run id and append to the main file
        with open(run_file) as f, open(out_path, "a") as out:
            for line in f:
                rec = json.loads(line)
                rec["run"] = i
                out.write(json.dumps(rec) + "\n")
        Path(run_file).unlink(missing_ok=True)


def aggregate(path):
    by_engine = defaultdict(dict)      # (engine, B) -> {run: sps}
    runs = set()
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        by_engine[(rec["engine"], rec["B"])][rec["run"]] = rec["steps_per_sec"]
        runs.add(rec["run"])
    runs = sorted(runs)

    def fmt(v):
        return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"

    print(f"\n=== {len(runs)} runs — env-steps/s: median [min .. max] ===")
    batches = sorted({b for (_, b) in by_engine})
    engines = sorted({e for (e, _) in by_engine})
    for B in batches:
        print(f"\nB={B}")
        for e in engines:
            vals = [by_engine[(e, B)][r] for r in runs if r in by_engine.get((e, B), {})]
            if not vals:
                continue
            print(f"  {e:18s} {fmt(statistics.median(vals)):>9s}"
                  f"  [{fmt(min(vals))} .. {fmt(max(vals))}]")

    print("\n=== within-run ratios (djinn / reference): median [min .. max] ===")
    for B in batches:
        parts = []
        for label, ours, theirs in RATIO_PAIRS:
            ratios = []
            for r in runs:
                a = by_engine.get((ours, B), {}).get(r)
                b = by_engine.get((theirs, B), {}).get(r)
                if a and b:
                    ratios.append(a / b)
            if ratios:
                parts.append(f"{label}: {statistics.median(ratios):.1f}x"
                             f" [{min(ratios):.1f} .. {max(ratios):.1f}]")
        print(f"  B={B:<6d} " + "   ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 1024, 8192, 65536])
    ap.add_argument("--unroll", type=int, default=1)
    ap.add_argument("--rng", default="threefry2x32")
    ap.add_argument("--out", default=str(HERE / "sweep_results.jsonl"))
    ap.add_argument("--aggregate-only", default=None)
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate(args.aggregate_only)
        return
    Path(args.out).unlink(missing_ok=True)
    run_sweep(args.n, args.batches, args.unroll, args.rng, args.out)
    aggregate(args.out)


if __name__ == "__main__":
    main()
