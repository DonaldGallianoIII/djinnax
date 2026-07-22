#!/usr/bin/env python3
"""E1 A/B (external review R2 / audit PERF-01): analytic mask predicate
vs mask-from-4-move-networks, inside the self-contained Mode B
megakernel. ABBA pairwise; n fresh processes for the official pattern:

    for i in 1 2 3 4 5; do python benchmarks/megakernel_canmask_ab.py --json out.jsonl; done

Binding gates (run before trusting this): check_analytic_mask_chain_link
(lane predicate ≡ jumanji-chained changed flags, incl. adversarial
boards) and check_analytic_mask_step (full-rollout bit-equivalence vs
the _move_dir mask path) — NOT kernel-vs-XLA parity, which the
same-function trick makes blind to a wrong shared component. This
script also asserts kernel-output equivalence old-vs-new before timing.
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
from djinnax.megakernel import N_STEPS, _fresh_inputs, step_lanes, step_lanes_movemask
from djinnax.megakernel_rng import run_megakernel_rng


def bench(Bn: int, rounds: int = 8):
    board, _ = _fresh_inputs(Bn, 1)
    seed = jnp.asarray([42, 0], dtype=jnp.uint32)
    new = jax.jit(lambda b, s: run_megakernel_rng(b, s, step_fn=step_lanes))
    old = jax.jit(lambda b, s: run_megakernel_rng(b, s, step_fn=step_lanes_movemask))
    rn = jax.block_until_ready(new(board, seed))
    ro = jax.block_until_ready(old(board, seed))
    for a, b, name in zip(rn, ro, ("board", "score", "done")):
        assert jnp.array_equal(a, b), f"variant divergence: {name}"
    ratios, t_new = abba_ratios(lambda: old(board, seed), lambda: new(board, seed), rounds)
    return {
        "B": Bn, "pair": "movemask->analytic-kernel",
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "steps_per_s_new": Bn * N_STEPS / t_new,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 8192, 65536])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    assert jax.default_backend() == "gpu", "GPU required (see HOW_TO_RUN.md shim)"
    print("backend: gpu  (megakernel: analytic mask vs mask-from-4-moves)")
    for Bn in args.batches:
        row = bench(Bn, args.rounds)
        print(f"B={Bn:<6d} analytic-mask kernel speedup median "
              f"{row['ratio_median']:.3f}x "
              f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]  "
              f"({row['steps_per_s_new'] / 1e6:,.0f}M steps/s)")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
