#!/usr/bin/env python3
"""P4 A/B (external review R1): oriented move via (4,16) permutation
gather vs materializing both (4,B,4,4) orientation stacks, on the
production LUT engine. Bit-identical forms (gated before this script is
trusted). Interleaved pairwise; run n fresh processes:

    for i in 1 2 3 4 5; do python benchmarks/orient_gather_ab.py --json out.jsonl; done
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

from benchmarks.ab_timing import abba_ratios
from benchmarks.bench_head_to_head import make_2048_djinn, _root_key
from djinnax.game2048 import Djinn2048, _oriented_move_stack
from djinnax.game2048_lut import move_left_lut


def _pair(Bn, n_steps, rounds):
    prev = Djinn2048(move_left_fn=move_left_lut, oriented_move_fn=_oriented_move_stack)
    new = Djinn2048(move_left_fn=move_left_lut)
    sa, ra = make_2048_djinn(Bn, n_steps, game=prev)
    sb, rb = make_2048_djinn(Bn, n_steps, game=new)
    key = _root_key(1)
    jax.block_until_ready(ra(sa, key))
    jax.block_until_ready(rb(sb, key))
    ratios, tb = abba_ratios(lambda: ra(sa, key), lambda: rb(sb, key), rounds)
    return {
        "B": Bn, "pair": "stack->gather",
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "steps_per_s_b": Bn * n_steps / tb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 8192, 65536])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    assert jax.default_backend() == "gpu", "GPU required (see HOW_TO_RUN.md shim)"
    print("backend: gpu  (LUT engine; oriented move: permutation gather vs stacks)")
    for Bn in args.batches:
        row = _pair(Bn, args.steps, args.rounds)
        print(f"B={Bn:<6d} gather speedup median {row['ratio_median']:.3f}x "
              f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
