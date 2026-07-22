#!/usr/bin/env python3
"""Collapse principle measured on 2048: conditional draw vs the naive
rejection loop most reference ports use for tile spawning.

The djinnax spawn is already the collapsed form (one conditional draw
over empty cells). The ORIGINAL 2048 (and typical C++/JS ports) spawns
by "pick a random cell; if occupied, retry" — an unbounded rejection
loop. This script builds that naive spawn faithfully (batch-native
lax.while_loop, iterating until every env in the batch has placed), then:
  1. proves distribution parity (same cell distribution conditioned on
     empties, same 0.9/0.1 value split),
  2. interleave-times full mask-guided games: rejection vs collapsed,
  3. probes the worst case (boards with ONE empty cell — where the
     rejection loop's batch-max tail explodes).

The rejection spawn lives HERE, not in the package: it is an exhibit of
the anti-pattern, quantified. See PORTING_PLAYBOOK step 1.5.
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
import numpy as np
from jax import lax

from benchmarks.ab_timing import abba_ratios
from djinnax.game2048 import Djinn2048, _spawn as spawn_collapsed
from djinnax.runtime import sample_uniform_legal

N_STEPS = 64


def spawn_rejection(board: jax.Array, key: jax.Array, enabled: jax.Array):
    """Naive-port spawn: uniform random cell, retry while occupied.
    Batch-native while_loop — runs until the LAST env in the batch
    lands, which is exactly why reference ports of this pattern are slow.
    Distribution-identical to the collapsed spawn (uniform over empties,
    exp 1 with p=0.9 else 2)."""
    B = board.shape[0]
    flat = board.reshape(B, 16)
    kv, kloop = jax.random.split(key)
    val = jnp.where(jax.random.uniform(kv, (B,)) < 0.9, jnp.int8(1), jnp.int8(2))
    has_empty = jnp.any(flat == 0, axis=-1)
    need = enabled & has_empty

    def cond(carry):
        _, _, placed_cell, i = carry
        return jnp.any((placed_cell < 0) & need) & (i < 10_000)

    def body(carry):
        flat, key, placed_cell, i = carry
        k = jax.random.fold_in(key, i)
        cell = jax.random.randint(k, (B,), 0, 16)
        hit = (placed_cell < 0) & need & (
            jnp.take_along_axis(flat, cell[:, None], axis=-1)[:, 0] == 0)
        cells = jnp.arange(16)[None, :]
        write = (cells == cell[:, None]) & hit[:, None]
        flat = jnp.where(write, val[:, None], flat).astype(jnp.int8)
        placed_cell = jnp.where(hit, cell.astype(jnp.int32), placed_cell)
        return flat, key, placed_cell, i + 1

    placed0 = jnp.full((B,), -1, dtype=jnp.int32)
    flat, _, placed_cell, _ = lax.while_loop(
        cond, body, (flat, kloop, placed0, jnp.int32(0)))
    return flat.reshape(B, 4, 4), jnp.maximum(placed_cell, 0)


def check_distribution_parity(n=40_000):
    """Same partially-filled board, many spawns each way: cell histogram
    over empties and the value split must match."""
    grid = np.zeros((4, 4), np.int8)
    grid[0, :] = 1; grid[1, :2] = 2; grid[2, 0] = 3     # 9 empties
    board = jnp.asarray(np.tile(grid.reshape(1, 16), (n, 1)).reshape(n, 4, 4))
    en = jnp.ones((n,), bool)
    b_c, cell_c = jax.jit(spawn_collapsed)(board, jax.random.PRNGKey(0), en)
    b_r, cell_r = jax.jit(spawn_rejection)(board, jax.random.PRNGKey(1), en)
    empties = np.flatnonzero(np.asarray(grid).reshape(16) == 0)
    for name, cells, b in (("collapsed", cell_c, b_c), ("rejection", cell_r, b_r)):
        h = np.bincount(np.asarray(cells), minlength=16)[empties] / n
        assert np.all(np.abs(h - 1 / len(empties)) < 0.01), f"{name}: cell dist {h}"
        vals = np.asarray(b.reshape(n, 16))[np.arange(n), np.asarray(cells)]
        frac2 = (vals == 2).mean()
        assert 0.09 < frac2 < 0.11, f"{name}: value split {frac2}"
    print(f"distribution parity OK — both spawns uniform over {len(empties)} "
          f"empties, ~0.9/0.1 values ({n} draws each)")


def _make_runner(game, B):
    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        a = sample_uniform_legal(k, state.action_mask)
        state, _ = game.step(state, a.astype(jnp.int32), jax.random.fold_in(k, 1))
        return (state, key), None

    @jax.jit
    def runner(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(N_STEPS))
        return state
    return runner


def bench_full_games(B, rounds=8):
    g_c = Djinn2048()
    g_r = Djinn2048(spawn_fn=spawn_rejection)
    r_c, r_r = _make_runner(g_c, B), _make_runner(g_r, B)
    s_c = g_c.init(jax.random.PRNGKey(0), B)
    s_r = g_r.init(jax.random.PRNGKey(0), B)
    key = jax.random.PRNGKey(42)
    s_c = jax.block_until_ready(r_c(s_c, key))
    s_r = jax.block_until_ready(r_r(s_r, key))
    # states thread through rounds; boxed so the ABBA thunks can update them
    box = {"r": s_r, "c": s_c, "i": 0}

    def run_r():
        k = jax.random.fold_in(key, box["i"])
        box["r"] = r_r(box["r"], k)
        return box["r"]

    def run_c():
        k = jax.random.fold_in(key, box["i"])
        box["c"] = r_c(box["c"], k)
        return box["c"]

    ratios = []
    for i in range(rounds):
        box["i"] = i
        r, _ = abba_ratios(run_r, run_c, 1)
        ratios.extend(r)
    print(f"B={B:<6d} full games — collapsed speedup over rejection: "
          f"median {statistics.median(ratios):.2f}x [{min(ratios):.2f} .. {max(ratios):.2f}]")


def bench_worst_case(B=8192, rounds=8):
    """Boards with ONE empty cell — the rejection loop's nightmare
    (E[tries]=16 per env; batch-max much worse)."""
    grid = np.ones((4, 4), np.int8); grid[3, 3] = 0
    board = jnp.asarray(np.tile(grid.reshape(1, 16), (B, 1)).reshape(B, 4, 4))
    en = jnp.ones((B,), bool)
    f_c = jax.jit(spawn_collapsed)
    f_r = jax.jit(spawn_rejection)
    jax.block_until_ready(f_c(board, jax.random.PRNGKey(0), en))
    jax.block_until_ready(f_r(board, jax.random.PRNGKey(0), en))
    ratios = []
    for i in range(rounds):
        k = jax.random.fold_in(jax.random.PRNGKey(1), i)
        r, _ = abba_ratios(lambda: f_r(board, k, en), lambda: f_c(board, k, en), 1)
        ratios.extend(r)
    print(f"B={B:<6d} spawn-only, 1 empty cell — collapsed speedup: "
          f"median {statistics.median(ratios):.2f}x [{min(ratios):.2f} .. {max(ratios):.2f}]")


def main():
    print(f"backend: {jax.default_backend()}")
    check_distribution_parity()
    for B in (1024, 8192, 65536):
        bench_full_games(B)
    bench_worst_case()


if __name__ == "__main__":
    main()
