#!/usr/bin/env python3
"""P-S1 A/B (audit PERF-02): batch-gated Sokoban reset —
lax.cond(any(done)) around level sampling — vs unconditional reset.
Measured in BOTH episode regimes, because the answer differs:

  sync:   all envs init at step 0 (the head-to-head's regime) — episodes
          terminate together, any(done) is true ~1 step in 120, the cond
          skips reset work on the other 119.
  desync: initial step_count uniform over [0, TIME_LIMIT) (training's
          regime) — at large B some env terminates virtually every step,
          any(done) ~always true, the cond is pure overhead.

ABBA pairwise; n fresh processes:

    for i in 1 2 3 4 5; do python benchmarks/soko_gated_ab.py --json out.jsonl; done

Equivalence gate before timing: gated ≡ ungated end states bit-for-bit
in both regimes (same keys → same samples wherever any(done)).
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
from jax import lax

from benchmarks.ab_timing import abba_ratios
from benchmarks.bench_head_to_head import _root_key
from djinnax.sokoban import TIME_LIMIT, DjinnSokoban

N_STEPS = 64


def _make_runner(game, B):
    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        action = jax.random.randint(k, (B,), 0, 4)
        state, _, _, _ = game.step(state, action, jax.random.fold_in(k, 1))
        return (state, key), None

    @jax.jit
    def runner(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(N_STEPS))
        return state
    return runner


def _pair(Bn, regime, rounds):
    plain = DjinnSokoban()
    gated = DjinnSokoban(batch_gated_reset=True)
    s0 = plain.init(_root_key(0), Bn)
    if regime == "desync":
        counts = jax.random.randint(_root_key(7), (Bn,), 0, TIME_LIMIT)
        s0 = s0.replace(step_count=counts.astype(jnp.int16))
    ra, rb = _make_runner(plain, Bn), _make_runner(gated, Bn)
    key = _root_key(1)
    fa = jax.block_until_ready(ra(s0, key))
    fb = jax.block_until_ready(rb(s0, key))
    for f in ("fixed_grid", "variable_grid", "agent_yx", "step_count", "terminated"):
        assert jnp.array_equal(getattr(fa, f), getattr(fb, f)), f"{regime}: {f} divergence"
    ratios, tb = abba_ratios(lambda: ra(s0, key), lambda: rb(s0, key), rounds)
    return {
        "B": Bn, "regime": regime, "pair": "plain->gated",
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "steps_per_s_b": Bn * N_STEPS / tb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 8192, 65536])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    assert jax.default_backend() == "gpu", "GPU required (see HOW_TO_RUN.md shim)"
    print("backend: gpu  (soko batch-gated reset vs unconditional; sync + desync)")
    for regime in ("sync", "desync"):
        for Bn in args.batches:
            row = _pair(Bn, regime, args.rounds)
            print(f"{regime:6s} B={Bn:<6d} gated speedup median "
                  f"{row['ratio_median']:.3f}x "
                  f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]")
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
