#!/usr/bin/env python3
"""P2/P3 A/B (external review R1): spawn-machinery variants of the XLA
2048 engine, attributed separately on the production LUT engine.

Configs (all distribution-identical, gated by check_2048_spawn /
check_2048_reset before this script is trusted):
  old : categorical spawn + reset via full spawn machinery (pre-review)
  p2  : categorical spawn + direct randint reset template
  new : rank-pick spawn   + direct randint reset template (production)

Interleaved pairwise per LEARNINGS §3 rule 9, adjacent configs only
(old vs p2 attributes P2; p2 vs new attributes P3). Run n fresh
processes and take medians across runs:

    for i in 1 2 3 4 5; do python benchmarks/spawn_variants_ab.py \
        --json out.jsonl; done
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
import jax.numpy as jnp

from benchmarks.bench_head_to_head import make_2048_djinn, _root_key

N_STEPS_DEFAULT = 64  # matches bench_head_to_head --steps default
from djinnax.game2048 import Djinn2048, _spawn_categorical
from djinnax.game2048_lut import move_left_lut


def _make_cfg(name, B, n_steps):
    if name == "old":
        game = Djinn2048(move_left_fn=move_left_lut, spawn_fn=_spawn_categorical)
        game._reset_spawn = game._reset_spawn_via_spawn
    elif name == "p2":
        game = Djinn2048(move_left_fn=move_left_lut, spawn_fn=_spawn_categorical)
    elif name == "new":
        game = Djinn2048(move_left_fn=move_left_lut)
    else:
        raise ValueError(name)
    return make_2048_djinn(B, n_steps, game=game)


def _pair(Bn, n_steps, name_a, name_b, rounds):
    """Interleaved pairwise: ratio time(a)/time(b) — >1 means b faster."""
    sa, ra = _make_cfg(name_a, Bn, n_steps)
    sb, rb = _make_cfg(name_b, Bn, n_steps)
    key = _root_key(1)
    jax.block_until_ready(ra(sa, key))          # compile+warm both
    jax.block_until_ready(rb(sb, key))
    ratios, tb = [], None
    for _ in range(rounds):
        t0 = time.perf_counter(); jax.block_until_ready(ra(sa, key)); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(rb(sb, key)); tb = time.perf_counter() - t0
        ratios.append(ta / tb)
    return {
        "B": Bn, "pair": f"{name_a}->{name_b}",
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
    print("backend: gpu  (LUT engine; interleaved adjacent pairs: "
          "old->p2 attributes P2, p2->new attributes P3)")
    for Bn in args.batches:
        for a, b in (("old", "p2"), ("p2", "new")):
            row = _pair(Bn, args.steps, a, b, args.rounds)
            print(f"B={Bn:<6d} {row['pair']:<9s} speedup median "
                  f"{row['ratio_median']:.3f}x [{row['ratio_min']:.3f} .. "
                  f"{row['ratio_max']:.3f}]")
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
