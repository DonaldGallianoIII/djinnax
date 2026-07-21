#!/usr/bin/env python3
"""P8 A/B (external review R1): single vs double fmix32 finalizer in the
megakernel's counter RNG. The quality battery (means, 16-bucket
flatness, cross-salt/step/env/seed correlations) passes for BOTH — that
gate ran before this script existed; speed decides. Different rounds =
different bit streams, so in-process parity here is kernel ≡ its own
XLA reference per variant (the same-function trick), not cross-variant.

    for i in 1 2 3 4 5; do python benchmarks/rng_rounds_ab.py --json out.jsonl; done
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

from djinnax.megakernel import N_STEPS, _fresh_inputs
from djinnax.megakernel_rng import run_megakernel_rng, run_xla_reference_rng


def _pair(Bn, rounds):
    board, _ = _fresh_inputs(Bn, 1)
    seed = jnp.asarray([42, 0], dtype=jnp.uint32)
    two = jax.jit(lambda b, s: run_megakernel_rng(b, s))
    one = jax.jit(lambda b, s: run_megakernel_rng(b, s, rng_rounds=1))
    # per-variant chain-anchor parity before timing
    for rr, mk in ((2, two), (1, one)):
        km = jax.block_until_ready(mk(board, seed))
        kx = jax.block_until_ready(jax.jit(
            lambda b, s: run_xla_reference_rng_rounds(b, s, rr))(board, seed))
        for a, b_, name in zip(km, kx, ("board", "score", "done")):
            assert jnp.array_equal(a, b_), f"rounds={rr}: {name}"
    ratios, t1 = [], None
    for _ in range(rounds):
        t0 = time.perf_counter(); jax.block_until_ready(two(board, seed)); t2 = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(one(board, seed)); t1 = time.perf_counter() - t0
        ratios.append(t2 / t1)
    return {
        "B": Bn, "pair": "fmix2->fmix1",
        "ratio_median": statistics.median(ratios),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "steps_per_s_b": Bn * N_STEPS / t1,
    }


def run_xla_reference_rng_rounds(board, seed, rng_rounds):
    from djinnax.megakernel import _initial_mask
    from djinnax.megakernel_rng import step_rng
    from jax import lax

    Bn = board.shape[0]
    env_id = jnp.arange(Bn, dtype=jnp.uint32)
    lanes = tuple(board[:, i].astype(jnp.int32) for i in range(16))
    mask = _initial_mask(lanes)
    score = jnp.zeros_like(lanes[0], dtype=jnp.float32)
    done = jnp.zeros_like(lanes[0], dtype=jnp.bool_)
    t_offset = seed[1].astype(jnp.int32)

    def body(carry, t):
        lanes, mask, score, done = carry
        return step_rng(lanes, mask, score, env_id, t + t_offset, seed[0],
                        rng_rounds=rng_rounds), None

    (lanes, mask, score, done), _ = lax.scan(
        body, (lanes, mask, score, done), jnp.arange(N_STEPS))
    out_board = jnp.stack([l.astype(jnp.int8) for l in lanes], axis=-1)
    return out_board, score, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 8192, 65536])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    assert jax.default_backend() == "gpu", "GPU required (see HOW_TO_RUN.md shim)"
    print("backend: gpu  (megakernel Mode B; single vs double fmix32 finalizer)")
    for Bn in args.batches:
        row = _pair(Bn, args.rounds)
        print(f"B={Bn:<6d} fmix1 speedup median {row['ratio_median']:.3f}x "
              f"[{row['ratio_min']:.3f} .. {row['ratio_max']:.3f}]")
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
