#!/usr/bin/env python3
"""P1 A/B (external review R1): oriented megakernel (permute-by-action ->
ONE move network -> inverse permute) vs the all-four-moves variant, Mode B
(in-kernel counter RNG — the production form).

Interleaved pairwise per LEARNINGS §3 rule 9; run n fresh processes and
take medians across runs (sweep loop lives in the caller):

    for i in 1 2 3 4 5; do python benchmarks/megakernel_orient_ab.py \
        --json out.jsonl; done

Variant parity is asserted in-process before any timing (both kernels
must produce bit-identical boards/scores/done on the bench inputs).
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

from djinnax.megakernel import N_STEPS, _fresh_inputs, step_lanes_allmoves
from djinnax.megakernel_rng import run_megakernel_rng


def bench(Bn: int, rounds: int = 8):
    board, _ = _fresh_inputs(Bn, 1)
    seed = jnp.asarray([42, 0], dtype=jnp.uint32)
    new = jax.jit(lambda b, s: run_megakernel_rng(b, s))
    old = jax.jit(lambda b, s: run_megakernel_rng(b, s, step_fn=step_lanes_allmoves))

    t0 = time.perf_counter()
    rn = jax.block_until_ready(new(board, seed))
    c_new = time.perf_counter() - t0
    t0 = time.perf_counter()
    ro = jax.block_until_ready(old(board, seed))
    c_old = time.perf_counter() - t0
    for a, b, name in zip(rn, ro, ("board", "score", "done")):
        assert jnp.array_equal(a, b), f"variant divergence: {name}"

    ratios, t_new = [], None
    for _ in range(rounds):
        t0 = time.perf_counter(); jax.block_until_ready(old(board, seed)); t_old = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(new(board, seed)); t_new = time.perf_counter() - t0
        ratios.append(t_old / t_new)
    return {
        "B": Bn,
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "steps_per_s_new": Bn * N_STEPS / t_new,
        "compile_new_s": round(c_new, 1),
        "compile_old_s": round(c_old, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 8192, 65536])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    assert jax.default_backend() == "gpu", "GPU required (see HOW_TO_RUN.md shim)"
    print("backend: gpu  (oriented vs all-moves, interleaved pairwise, "
          "variant parity asserted first)")
    for Bn in args.batches:
        row = bench(Bn, args.rounds)
        print(f"B={Bn:<6d} oriented speedup median {row['ratio_median']:.3f}x "
              f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]  "
              f"({row['steps_per_s_new']:,.0f} env-steps/s)")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
