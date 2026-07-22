#!/usr/bin/env python3
"""E5(+E6) A/B: 9-bubble row move (odd-even pre-compaction + single
post-merge pass) vs the 18-bubble triple-pass form, in the Mode B
megakernel. ABBA pairwise; n fresh processes:

    for i in 1 2 3 4 5; do python benchmarks/megakernel_rowmove_ab.py --json out.jsonl; done

Gated before trusting: check_row_move_exhaustive (all 65536 rows
bit-identical incl. reward) and the jumanji-anchored chain link; this
script asserts kernel-output equivalence before timing.

The OLD side is built by re-tracing the kernel with
_row_move_4_triplepass patched in. Both sides carry the current E6
was_legal form (n_legal>0); E6's old masked-or is 8 boolean ops and
rides below noise — this receipt measures E5's bubble reduction.
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

import djinnax.megakernel as mk
from benchmarks.ab_timing import abba_ratios
from djinnax.megakernel import N_STEPS, _fresh_inputs
from djinnax.megakernel_rng import run_megakernel_rng


def bench(Bn: int, rounds: int = 8):
    board, _ = _fresh_inputs(Bn, 1)
    seed = jnp.asarray([42, 0], dtype=jnp.uint32)

    new = jax.jit(lambda b, s: run_megakernel_rng(b, s))
    rn = jax.block_until_ready(new(board, seed))      # trace with 9-bubble form

    orig = mk._row_move_4
    mk._row_move_4 = mk._row_move_4_triplepass
    try:
        # fresh step functions so the patched row move is picked up at trace
        step_old = mk._make_step(mk._apply_move_oriented)
        old = jax.jit(lambda b, s: run_megakernel_rng(b, s, step_fn=step_old))
        ro = jax.block_until_ready(old(board, seed))  # trace with 18-bubble form
    finally:
        mk._row_move_4 = orig

    for a, b, name in zip(rn, ro, ("board", "score", "done")):
        assert jnp.array_equal(a, b), f"variant divergence: {name}"
    ratios, t_new = abba_ratios(lambda: old(board, seed), lambda: new(board, seed), rounds)
    return {
        "B": Bn, "pair": "triplepass18->oddeven9",
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
    print("backend: gpu  (megakernel: 9-bubble vs 18-bubble row move)")
    for Bn in args.batches:
        row = bench(Bn, args.rounds)
        print(f"B={Bn:<6d} 9-bubble kernel speedup median "
              f"{row['ratio_median']:.3f}x "
              f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
