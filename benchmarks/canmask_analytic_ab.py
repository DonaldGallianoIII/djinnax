#!/usr/bin/env python3
"""E2 A/B (external review R2 / audit PERF-01): analytic legality
predicate vs derive-mask-from-4-moves, on BOTH XLA engines (branchless
and LUT). ABBA pairwise; run n fresh processes and take medians:

    for i in 1 2 3 4 5; do python benchmarks/canmask_analytic_ab.py --json out.jsonl; done

Predicate equivalence is gated exhaustively by check_2048_can_analytic
(all 65536 rows x both orientations) before this script is trusted;
this script additionally asserts bit-identical end states through a
full warmup rollout per engine pair before timing.

Context: P5 (CHANGED_LUT, killed 0.82-0.88x) replaced the compare with
a GATHER next to an existing gather. This replaces four full move
passes with 48 comparisons — a different mechanism; measured, not
assumed (op counts lie).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from benchmarks.ab_timing import abba_ratios
from benchmarks.bench_head_to_head import make_2048_djinn, _root_key
from djinnax.game2048 import Djinn2048, can_move_all_analytic, move_all_directions
from djinnax.game2048_lut import move_left_lut

N_STEPS_DEFAULT = 64

ENGINES = {
    "branchless": dict(),
    "lut": dict(move_left_fn=move_left_lut),
}


def _pair(engine, Bn, n_steps, rounds):
    kw = ENGINES[engine]
    # explicit fns on BOTH sides so this receipt stays valid regardless
    # of which variant is the Djinn2048 default
    mv = kw.get("move_left_fn")
    prev = Djinn2048(**kw, can_move_fn=lambda b: move_all_directions(b, mv)[2])
    new = Djinn2048(**kw, can_move_fn=can_move_all_analytic)  # analytic predicate
    sa, ra = make_2048_djinn(Bn, n_steps, game=prev)
    sb, rb = make_2048_djinn(Bn, n_steps, game=new)
    key = _root_key(1)
    fa = jax.block_until_ready(ra(sa, key))                   # compile+warm both
    fb = jax.block_until_ready(rb(sb, key))
    # bit-identical rollouts before any timing (same keys, same sampler)
    assert jnp.array_equal(fa.board, fb.board), f"{engine}: board divergence"
    assert jnp.array_equal(fa.score, fb.score), f"{engine}: score divergence"
    assert jnp.array_equal(fa.action_mask, fb.action_mask), f"{engine}: mask divergence"
    ratios, tb = abba_ratios(lambda: ra(sa, key), lambda: rb(sb, key), rounds)
    return {
        "B": Bn, "engine": engine, "pair": "movemask->analytic",
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
    print("backend: gpu  (analytic can-predicate vs mask-from-4-moves)")
    for engine in ENGINES:
        for Bn in args.batches:
            row = _pair(engine, Bn, args.steps, args.rounds)
            print(f"{engine:10s} B={Bn:<6d} analytic speedup median "
                  f"{row['ratio_median']:.3f}x "
                  f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]")
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
