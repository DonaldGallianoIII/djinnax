#!/usr/bin/env python3
"""Pallas lab — learning ladder rung 4 hands-on. Educational, not a claim.

Pallas = JAX's kernel DSL: you write the *body* that runs per grid tile,
with explicit control of block shapes and memory movement, instead of
letting XLA fuse whole-array ops. Inside a kernel you still write jnp on
the loaded tiles — the mental model is "jnp on register blocks."

Three exhibits:
  A. hello kernel (saxpy) — pallas_call / Ref / BlockSpec / grid basics.
  B. the 2048 branchless row-move (compact+merge+compact) as ONE fused
     kernel, reusing the SAME jnp code that runs under XLA — parity-gated
     against the XLA version, then interleaved-timed.
  C. the honest lesson: why the LUT (gather) variant is NOT here — kernel
     DSLs make arbitrary gathers the hard part while XLA excels at them;
     rung 3 (LUT+XLA) is hard to beat from below.

BACKEND LESSON (learned iterating on this file): JAX's default Pallas-GPU
lowering is Mosaic GPU, which emits Hopper TMA instructions
(cp.async.bulk) — sm_90+ ONLY. On this RTX 4070 (Ada, sm_89) it fails in
three escalating ways (128B warpgroup copy granularity, 256-elem/dim
async-copy limit, then 'not supported on sm_89'). The legacy Triton
lowering (`compiler_params=plt.CompilerParams()`) runs fine on sm_89 and
has none of those tile constraints. Rung-4 work is hardware-generation-
specific in a way rungs 1-3 never are.

Measurement caveat printed at runtime: clocks are UNLOCKED on the measurement host;
interleaved medians with spread only, per LEARNINGS §3 rule 9.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import statistics
import time

import djinnax.refs  # noqa: F401

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from djinnax.game2048 import _compact_left, _merge_left, _move_left

_TRITON = dict(compiler_params=plt.CompilerParams())

# --- Exhibit A: hello kernel -------------------------------------------------


def saxpy(a: float, x: jax.Array, y: jax.Array) -> jax.Array:
    """`a` baked as a compile-time constant (scalar Refs are where the
    Mosaic backend first fought us; Triton simply doesn't need one)."""
    n = x.shape[0]
    block = 4096

    def kernel(x_ref, y_ref, o_ref):
        # Refs are views of the current tile; index to load, assign to store.
        o_ref[...] = a * x_ref[...] + y_ref[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[
            pl.BlockSpec((block,), lambda i: (i,)),
            pl.BlockSpec((block,), lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
        **_TRITON,
    )(x, y)


# --- Exhibit B: fused row-move kernel ---------------------------------------

_ROWS_BLOCK = 4096     # rows per tile (Triton has no 256/dim limit)


def _rowmove_kernel(x0_ref, x1_ref, x2_ref, x3_ref,
                    o0_ref, o1_ref, o2_ref, o3_ref, reward_ref):
    """Structure-of-arrays row move. Triton's lowering has no `slice` /
    `.at[]` on in-kernel arrays, so the row's 4 lanes arrive as separate
    (block,) refs and the compaction becomes an explicit zero-bubbling
    swap network — pure elementwise `where`s, the shape Triton wants.
    """
    x0 = x0_ref[...].astype(jnp.int32)
    x1 = x1_ref[...].astype(jnp.int32)
    x2 = x2_ref[...].astype(jnp.int32)
    x3 = x3_ref[...].astype(jnp.int32)

    def bubble(a, b):
        z = a == 0
        return jnp.where(z, b, a), jnp.where(z, 0, b)

    def compact(x0, x1, x2, x3):
        for _ in range(3):                             # 3 bubble passes
            x0, x1 = bubble(x0, x1)
            x1, x2 = bubble(x1, x2)
            x2, x3 = bubble(x2, x3)
        return x0, x1, x2, x3

    x0, x1, x2, x3 = compact(x0, x1, x2, x3)
    reward = jnp.zeros_like(x0, dtype=jnp.float32)

    def merge(a, b, reward):
        m = (a != 0) & (a == b)
        # exact integer 2^(a+1) via shift — Triton's float pow is inexact
        # (parity vs XLA failed on 2.0 ** x before this)
        merged_val = (jnp.int32(1) << (a + 1)).astype(jnp.float32)
        reward = reward + jnp.where(m, merged_val, 0.0)
        return jnp.where(m, a + 1, a), jnp.where(m, 0, b), reward

    x0, x1, reward = merge(x0, x1, reward)
    x1, x2, reward = merge(x1, x2, reward)
    x2, x3, reward = merge(x2, x3, reward)
    x0, x1, x2, x3 = compact(x0, x1, x2, x3)

    o0_ref[...] = x0.astype(jnp.int8)
    o1_ref[...] = x1.astype(jnp.int8)
    o2_ref[...] = x2.astype(jnp.int8)
    o3_ref[...] = x3.astype(jnp.int8)
    reward_ref[...] = reward


def rowmove_pallas(rows: jax.Array):
    """rows: (R, 4) int8 -> (moved (R, 4) int8, reward (R,) f32)."""
    r = rows.shape[0]
    cols = [rows[:, i] for i in range(4)]              # SoA layout outside
    spec = pl.BlockSpec((_ROWS_BLOCK,), lambda i: (i,))
    out = pl.pallas_call(
        _rowmove_kernel,
        out_shape=tuple(
            [jax.ShapeDtypeStruct((r,), jnp.int8)] * 4
            + [jax.ShapeDtypeStruct((r,), jnp.float32)]
        ),
        grid=(r // _ROWS_BLOCK,),
        in_specs=[spec] * 4,
        out_specs=tuple([spec] * 5),
        **_TRITON,
    )(*cols)
    moved = jnp.stack(out[:4], axis=-1)
    return moved, out[4]


def rowmove_xla(rows: jax.Array):
    # NB: not _move_left — that's board-shaped ((..., 4, 4)) and its final
    # sum(-1) would collapse per-ROW rewards to a scalar on (R, 4) input.
    r = _compact_left(rows.astype(jnp.int32))
    r, reward = _merge_left(r)
    return _compact_left(r).astype(jnp.int8), reward.astype(jnp.float32)


# --- parity + interleaved timing --------------------------------------------


def check_parity(n_rows: int = 65536, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    rows = jax.random.randint(key, (n_rows, 4), 0, 8, dtype=jnp.int32).astype(jnp.int8)
    pm, pr = jax.jit(rowmove_pallas)(rows)
    xm, xr = jax.jit(rowmove_xla)(rows)
    assert jnp.array_equal(pm, xm), "pallas/XLA moved rows differ"
    assert jnp.array_equal(pr, xr), "pallas/XLA rewards differ"
    print(f"parity OK — pallas row-move ≡ XLA row-move on {n_rows} random rows")


def interleaved(n_rows: int, rounds: int = 8, iters: int = 200):
    """Median per-round time ratio (XLA / pallas; >1 => pallas faster).

    Each timed call loops the kernel `iters` times (fori composition via
    python loop under one jit) so we time compute, not dispatch of a
    single microsecond kernel.
    """
    key = jax.random.PRNGKey(1)
    rows = jax.random.randint(key, (n_rows, 4), 0, 8, dtype=jnp.int32).astype(jnp.int8)

    @jax.jit
    def many_pallas(r):
        for _ in range(iters):
            r, _ = rowmove_pallas(r)
        return r

    @jax.jit
    def many_xla(r):
        for _ in range(iters):
            r, _ = rowmove_xla(r)
        return r

    jax.block_until_ready(many_pallas(rows))
    jax.block_until_ready(many_xla(rows))
    ratios = []
    for _ in range(rounds):
        t0 = time.perf_counter(); jax.block_until_ready(many_xla(rows)); tx = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(many_pallas(rows)); tp = time.perf_counter() - t0
        ratios.append(tx / tp)
    med = statistics.median(ratios)
    row_moves = n_rows * iters
    print(f"rows={n_rows:<7d} pallas-speedup median {med:.2f}x "
          f"[{min(ratios):.2f} .. {max(ratios):.2f}]  "
          f"(~{row_moves / (tp / 1):,.0f} row-moves/s pallas last-round)")


def main():
    print(f"backend: {jax.default_backend()}  (clocks UNLOCKED — medians "
          f"with spread only; educational, not a claim)")
    x = jnp.arange(1 << 20, dtype=jnp.float32)
    y = jnp.ones(1 << 20, dtype=jnp.float32)
    out = saxpy(2.0, x, y)
    assert jnp.allclose(out[:4], 2.0 * x[:4] + 1.0)
    print("exhibit A OK — saxpy kernel runs (pallas_call/BlockSpec/grid)")

    check_parity()
    for n_rows in (4096, 32768, 262144):
        interleaved(n_rows)

    print("\nExhibit C (why no LUT kernel here): the 65,536-entry gather is "
          "the LUT variant's whole trick, and arbitrary vectorized gathers "
          "are exactly what kernel DSLs make hard while XLA excels at them. "
          "Rung 4 is for fusion XLA can't find — not for re-doing gathers.")


if __name__ == "__main__":
    main()
