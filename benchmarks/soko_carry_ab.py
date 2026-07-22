#!/usr/bin/env python3
"""P6 A/B (external review R1): sokoban carried n_on_target vs per-step
recount. Interleaved pairwise; run n fresh processes:

    for i in 1 2 3 4 5; do python benchmarks/soko_carry_ab.py --json out.jsonl; done

Reward-stream equality between the variants is gated by the exact
jumanji replay (check_sokoban) before this script is trusted.
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
from jax import lax

from benchmarks.ab_timing import abba_ratios
from benchmarks.bench_head_to_head import UNROLL, _root_key
from djinnax.sokoban import DjinnSokoban


def _make_runner(game, B, n_steps):
    state0 = game.init(_root_key(0), B)

    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        action = jax.random.randint(k, (B,), 0, 4)
        state, _, _, _ = game.step(state, action, jax.random.fold_in(k, 1))
        return (state, key), None

    @jax.jit
    def runner(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(n_steps), unroll=UNROLL)
        return state

    return state0, runner


def _pair(Bn, n_steps, rounds):
    sa, ra = _make_runner(DjinnSokoban(carry_on_target=False), Bn, n_steps)
    sb, rb = _make_runner(DjinnSokoban(carry_on_target=True), Bn, n_steps)
    key = _root_key(1)
    jax.block_until_ready(ra(sa, key))
    jax.block_until_ready(rb(sb, key))
    ratios, tb = abba_ratios(lambda: ra(sa, key), lambda: rb(sb, key), rounds)
    return {
        "B": Bn, "pair": "recount->carry",
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
    print("backend: gpu  (sokoban; carried n_on_target vs per-step recount)")
    for Bn in args.batches:
        row = _pair(Bn, args.steps, args.rounds)
        print(f"B={Bn:<6d} carry speedup median {row['ratio_median']:.3f}x "
              f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
