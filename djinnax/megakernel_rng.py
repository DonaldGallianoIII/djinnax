#!/usr/bin/env python3
"""Megakernel Mode B — in-kernel counter RNG (MEGAKERNEL_PLAN stage 4).

Mode A streams a pre-generated (N_STEPS, 5, B) uniform buffer from HBM.
Mode B generates every uniform in-register from a counter hash of
(env_id, step, salt, seed) — double-fmix32 (murmur3 finalizer x2), pure
uint32 ops, no randomness traffic from memory at all. Not crypto; it is
a statistically-tested game RNG (see checks below).

Because the hash is plain jnp uint32 arithmetic, the same-function parity
trick extends to the RNG itself: `step_rng()` runs unchanged inside the
Triton kernel and under an XLA lax.scan, so Mode B is BIT-verified, not
just distribution-tested. Distribution checks run anyway (mean/std,
bucket flatness, cross-salt correlation, spawn 0.9/0.1 ratio) because a
bit-identical pair of engines can still share a bad RNG.

Benches (interleaved, unlocked-clock caveats as ever):
  1. Mode B vs Mode A self-contained  -> what in-kernel RNG buys
  2. Mode B vs production LUT runner  -> the new production number
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

from djinnax.megakernel import (
    BLOCK, N_STEPS, N_UNI, _TRITON, _fresh_inputs, _initial_mask,
    run_megakernel, step_lanes,
)


# --- counter hash RNG --------------------------------------------------------


def _fmix32(h):
    h = h ^ (h >> jnp.uint32(16))
    h = h * jnp.uint32(0x85EBCA6B)
    h = h ^ (h >> jnp.uint32(13))
    h = h * jnp.uint32(0xC2B2AE35)
    h = h ^ (h >> jnp.uint32(16))
    return h


# numpy scalars (not jnp) — device arrays captured in a pallas kernel
# closure raise "captures constants"; np scalars inline as literals.
_SALTS = tuple(np.uint32(s) for s in
               (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1))


def hash_uniform(env_id, t, j, seed):
    """f32 uniform in [0, 1) from counters. env_id (B,) uint32; t traced
    int32 step; j static salt index; seed uint32 scalar."""
    h = env_id * jnp.uint32(0x9E3779B1)
    h = h ^ (t.astype(jnp.uint32) * jnp.uint32(0x9E3779B9) + _SALTS[j])
    h = h ^ seed
    h = _fmix32(_fmix32(h))
    return (h >> jnp.uint32(8)).astype(jnp.float32) * jnp.float32(1.0 / (1 << 24))


def step_rng(lanes, mask, score, env_id, t, seed):
    """One step with in-register RNG. Same function in-kernel and in XLA."""
    u = tuple(hash_uniform(env_id, t, j, seed) for j in range(N_UNI))
    return step_lanes(lanes, mask, score, u)


# --- Mode B megakernel -------------------------------------------------------


def _make_kernel_rng(n_steps: int, block: int = BLOCK):
    def kernel(board_ref, score_ref, seed_ref, out_board_ref, out_score_ref, out_done_ref):
        pid = pl.program_id(0)
        env_id = (pid * block + jnp.arange(block)).astype(jnp.uint32)
        seed = seed_ref[0]
        t_offset = seed_ref[1].astype(jnp.int32)
        lanes = tuple(board_ref[:, i].astype(jnp.int32) for i in range(16))
        mask = _initial_mask(lanes)
        score = score_ref[...]
        done = jnp.zeros_like(lanes[0], dtype=jnp.bool_)

        def body(t, carry):
            lanes, mask, score, done = carry
            return step_rng(lanes, mask, score, env_id, t + t_offset, seed)

        lanes, mask, score, done = lax.fori_loop(0, n_steps, body, (lanes, mask, score, done))
        for i in range(16):
            out_board_ref[:, i] = lanes[i].astype(jnp.int8)
        out_score_ref[...] = score
        out_done_ref[...] = done
    return kernel


def run_megakernel_rng(board: jax.Array, seed: jax.Array, score: jax.Array | None = None,
                       n_steps: int = N_STEPS, block: int = BLOCK,
                       compiler_params=None):
    """board (B, 16) int8; seed (2,) uint32 = [seed, t_offset]; score (B,)
    f32 carried across chained launches (mask is recomputed from the board
    at entry — exact, because the analytic reset mask provably equals the
    computed mask; see check_megakernel.py). Fully self-contained.

    B must be a multiple of BLOCK — the grid would silently DROP the tail
    otherwise (hardening: loud error instead)."""
    Bn = board.shape[0]
    if Bn % block != 0:
        raise ValueError(f"B={Bn} must be a multiple of BLOCK={block} "
                         f"(a truncated grid would silently skip the tail)")
    if score is None:
        score = jnp.zeros((Bn,), dtype=jnp.float32)
    params = {"compiler_params": compiler_params} if compiler_params else _TRITON
    return pl.pallas_call(
        _make_kernel_rng(n_steps, block),
        out_shape=(
            jax.ShapeDtypeStruct((Bn, 16), jnp.int8),
            jax.ShapeDtypeStruct((Bn,), jnp.float32),
            jax.ShapeDtypeStruct((Bn,), jnp.bool_),
        ),
        grid=(Bn // block,),
        in_specs=[
            pl.BlockSpec((block, 16), lambda i: (i, 0)),
            pl.BlockSpec((block,), lambda i: (i,)),
            pl.BlockSpec((2,), lambda i: (0,)),
        ],
        out_specs=(
            pl.BlockSpec((block, 16), lambda i: (i, 0)),
            pl.BlockSpec((block,), lambda i: (i,)),
            pl.BlockSpec((block,), lambda i: (i,)),
        ),
        **params,
    )(board, score, seed)


def run_xla_reference_rng(board: jax.Array, seed: jax.Array, score: jax.Array | None = None,
                          n_steps: int = N_STEPS):
    Bn = board.shape[0]
    env_id = jnp.arange(Bn, dtype=jnp.uint32)
    lanes = tuple(board[:, i].astype(jnp.int32) for i in range(16))
    mask = _initial_mask(lanes)
    if score is None:
        score = jnp.zeros_like(lanes[0], dtype=jnp.float32)
    done = jnp.zeros_like(lanes[0], dtype=jnp.bool_)
    t_offset = seed[1].astype(jnp.int32)

    def body(carry, t):
        lanes, mask, score, done = carry
        return step_rng(lanes, mask, score, env_id, t + t_offset, seed[0]), None

    (lanes, mask, score, done), _ = lax.scan(
        body, (lanes, mask, score, done), jnp.arange(n_steps))
    out_board = jnp.stack([l.astype(jnp.int8) for l in lanes], axis=-1)
    return out_board, score, done


# --- checks ------------------------------------------------------------------


xla_reference_rng_jit = jax.jit(run_xla_reference_rng, static_argnames=("n_steps",))


def check_parity(Bn=1024, seed_val=7):
    board, _ = _fresh_inputs(Bn, 0)
    seed = jnp.asarray([seed_val, 0], dtype=jnp.uint32)
    mb, ms, md = jax.jit(run_megakernel_rng)(board, seed)
    xb, xs, xd = xla_reference_rng_jit(board, seed)
    assert jnp.array_equal(mb, xb), "Mode B boards diverge"
    assert jnp.array_equal(ms, xs), "Mode B scores diverge"
    assert jnp.array_equal(md, xd), "Mode B done flags diverge"
    assert bool(jnp.all((mb >= 0) & (mb <= 15))), "board exponents out of range"
    print(f"Mode B parity OK — kernel ≡ XLA over {N_STEPS} steps at B={Bn}, "
          f"RNG included (bit-identical)")


def check_rng_quality(n=1 << 20, seed_val=3):
    env_id = jnp.arange(n, dtype=jnp.uint32)
    seed = jnp.uint32(seed_val)
    t = jnp.int32(11)
    us = [np.asarray(hash_uniform(env_id, t, j, seed)) for j in range(N_UNI)]
    for j, u in enumerate(us):
        assert abs(u.mean() - 0.5) < 0.002, f"salt {j}: mean {u.mean()}"
        assert abs(u.std() - (1 / 12) ** 0.5) < 0.002, f"salt {j}: std {u.std()}"
        counts, _ = np.histogram(u, bins=16, range=(0.0, 1.0))
        flat = np.abs(counts - n / 16) / (n / 16)
        assert flat.max() < 0.02, f"salt {j}: bucket dev {flat.max():.3f}"
    for a in range(N_UNI):
        for b in range(a + 1, N_UNI):
            c = float(np.corrcoef(us[a], us[b])[0, 1])
            assert abs(c) < 0.01, f"salts {a},{b} correlated: {c:.4f}"
    frac4 = float((us[2] >= 0.9).mean())
    assert 0.095 < frac4 < 0.105, f"spawn-4 fraction {frac4}"
    print(f"RNG quality OK — {N_UNI} salts x {n} draws: mean/std/16-bucket "
          f"flatness/cross-salt corr/0.9-split all within tolerance")


# --- benches -----------------------------------------------------------------


def _interleave(name, run_a, run_b, args_a, args_b, rounds=8):
    jax.block_until_ready(run_a(*args_a))
    jax.block_until_ready(run_b(*args_b))
    ratios = []
    for _ in range(rounds):
        t0 = time.perf_counter(); jax.block_until_ready(run_a(*args_a)); ta = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(run_b(*args_b)); tb = time.perf_counter() - t0
        ratios.append(ta / tb)
    print(f"{name}: B-speedup median {statistics.median(ratios):.2f}x "
          f"[{min(ratios):.2f} .. {max(ratios):.2f}]")


def bench(Bn):
    import benchmarks.bench_head_to_head as bh  # repo-only; not shipped in the wheel

    board, _ = _fresh_inputs(Bn, 1)
    seed = jnp.asarray([42, 0], dtype=jnp.uint32)
    mk_b = jax.jit(run_megakernel_rng)

    @jax.jit
    def mk_a_selfcontained(board, key):
        u = jax.random.uniform(key, (N_STEPS, N_UNI, Bn), dtype=jnp.float32)
        return run_megakernel(board, u)

    key = jax.random.PRNGKey(0)
    _interleave(f"B={Bn:<6d} modeA(self) vs modeB",
                mk_a_selfcontained, mk_b, (board, key), (board, seed))

    lut_state, lut_run = bh.make_2048_djinn_lut(Bn, N_STEPS)
    lut_state = jax.block_until_ready(lut_run(lut_state, jax.random.PRNGKey(1)))
    _interleave(f"B={Bn:<6d} prod-LUT vs modeB   ",
                lambda s, k: lut_run(s, k), mk_b,
                (lut_state, jax.random.PRNGKey(2)), (board, seed))
    # raw throughput readout
    t0 = time.perf_counter(); jax.block_until_ready(mk_b(board, seed)); tm = time.perf_counter() - t0
    print(f"         modeB last-call: {Bn * N_STEPS / tm:,.0f} env-steps/s")


def main():
    print(f"backend: {jax.default_backend()}  (clocks UNLOCKED — medians "
          f"with spread; a multiple is a result)")
    check_rng_quality()
    check_parity()
    for Bn in (1024, 8192, 65536):
        bench(Bn)


if __name__ == "__main__":
    main()
