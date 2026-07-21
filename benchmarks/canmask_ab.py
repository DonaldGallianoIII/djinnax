#!/usr/bin/env python3
"""P5 A/B (external review R1): CHANGED_LUT legality probe vs
unpack-and-compare, on the production LUT engine. Interleaved pairwise;
run n fresh processes and take medians across runs:

    for i in 1 2 3 4 5; do python benchmarks/canmask_ab.py --json out.jsonl; done

Probe equivalence is gated by check_2048_can_lut before this script is
trusted (full-range boards, exact equality).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys as _sys
import time
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import jax

from benchmarks.bench_head_to_head import make_2048_djinn, _root_key
from djinnax.game2048 import Djinn2048
from djinnax.game2048_lut import can_move_all_lut, move_left_lut

N_STEPS_DEFAULT = 64


def _pair(Bn, n_steps, rounds):
    prev = Djinn2048(move_left_fn=move_left_lut)                  # unpack-compare
    new = Djinn2048(move_left_fn=move_left_lut, can_move_fn=can_move_all_lut)
    sa, ra = make_2048_djinn(Bn, n_steps, game=prev)
    sb, rb = make_2048_djinn(Bn, n_steps, game=new)
    key = _root_key(1)
    jax.block_until_ready(ra(sa, key))
    jax.block_until_ready(rb(sb, key))
    ratios, tb = [], None
    for _ in range(rounds):
        t0 = time.perf_counter(); jax.block_until_ready(ra(sa, key)); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(rb(sb, key)); tb = time.perf_counter() - t0
        ratios.append(ta / tb)
    return {
        "B": Bn, "pair": "unpackcmp->changedlut",
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "steps_per_s_b": Bn * n_steps / tb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 8192, 65536])
    ap.add_argument("--steps", type=int, default=N_STEPS_DEFAULT)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    assert jax.default_backend() == "gpu", "GPU required (see HOW_TO_RUN.md shim)"
    print("backend: gpu  (LUT engine; CHANGED_LUT probe vs unpack-and-compare)")
    for Bn in args.batches:
        row = _pair(Bn, args.steps, args.rounds)
        print(f"B={Bn:<6d} changed-lut speedup median {row['ratio_median']:.3f}x "
              f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
