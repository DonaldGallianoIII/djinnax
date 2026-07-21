#!/usr/bin/env python3
"""Game-agnostic floor bench: protocol v1 vs v2, same process, per B.

Measures (a) the NULL env — pure runtime floor, no game logic — and
(b) real engines under both protocols, so every delta shown is a
game-agnostic runtime effect, not a game optimization.

v1: in-loop fold_in keys + masked-categorical sampler + no donation.
v2: bulk-hoisted keys + rank-pick sampler + donated carry.

Also validates sampler equivalence first: rank-pick and masked
categorical must both be exactly uniform over legal actions (chi-square
style tolerance on a fixed mask pattern).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import time

import djinnax.refs  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from djinnax.runtime import NullEnv, build_runner, sample_uniform_legal
from djinnax.ttt import DjinnTicTacToe
from djinnax.game2048_lut import make_game2048_lut
from djinnax.sokoban import DjinnSokoban

N_STEPS = 64
REPS = 5


def check_sampler_uniformity(n=200_000):
    """Both samplers uniform over legal; rank-pick never picks illegal."""
    mask = jnp.asarray([[True, False, True, True, False, False, True, False, False]])
    mask_b = jnp.broadcast_to(mask, (n, 9))
    key = jax.random.PRNGKey(0)
    a2 = sample_uniform_legal(key, mask_b)
    legal = np.flatnonzero(np.asarray(mask[0]))
    counts2 = np.bincount(np.asarray(a2), minlength=9)
    assert counts2[[i for i in range(9) if i not in legal]].sum() == 0, "rank-pick picked illegal"
    logits = jnp.where(mask_b, 0.0, -jnp.inf)
    a1 = jax.random.categorical(jax.random.PRNGKey(1), logits, axis=-1)
    counts1 = np.bincount(np.asarray(a1), minlength=9)
    exp = n / len(legal)
    for c in (counts1, counts2):
        dev = np.abs(c[legal] - exp) / exp
        assert np.all(dev < 0.03), f"sampler not uniform: {c[legal]} vs {exp}"
    print(f"sampler uniformity OK — both uniform over legal (max dev "
          f"v1 {np.abs(counts1[legal]-exp).max()/exp:.3f}, "
          f"v2 {np.abs(counts2[legal]-exp).max()/exp:.3f})")


# --- v1-style runner (matches bench_head_to_head's historical protocol) -----


def runner_v1(game, state0, mask_of, step_of):
    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        logits = jnp.where(mask_of(state), 0.0, -jnp.inf)
        action = jax.random.categorical(k, logits, axis=-1).astype(jnp.int32)
        state = step_of(state, action, jax.random.fold_in(k, 1))
        return (state, key), None

    @jax.jit
    def run(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(N_STEPS))
        return state

    return state0, run


def runner_v2(game, state0, mask_of, step_of):
    def one_step(state, keys):
        k_act, k_step = keys
        action = sample_uniform_legal(k_act, mask_of(state))
        return step_of(state, action, k_step)

    return state0, build_runner(one_step, N_STEPS, n_keys_per_step=2, donate=True)


def bench(label, state0, run, B):
    key = jax.random.PRNGKey(42)
    state = jax.block_until_ready(run(state0, key))
    best = float("inf")
    for r in range(REPS):
        k = jax.random.fold_in(key, r + 1)
        t0 = time.perf_counter()
        state = jax.block_until_ready(run(state, k))
        best = min(best, time.perf_counter() - t0)
    sps = B * N_STEPS / best
    print(f"  {label:26s} {sps:>14,.0f} env-steps/s  {1e6*best/(B*N_STEPS):>8.3f} µs/env-step")
    return sps


def engines(B):
    null_env = NullEnv()
    ttt = DjinnTicTacToe()
    ttt_template = ttt.init(jax.random.PRNGKey(0), B)

    def ttt_step(state, action, key):
        s = ttt.step(state, action, key)
        return jax.tree_util.tree_map(
            lambda t, x: jnp.where(
                s.terminated.reshape(B, *([1] * (x.ndim - 1))), t, x),
            ttt_template, s,
        )

    g2048 = make_game2048_lut()
    soko = DjinnSokoban()
    return [
        ("null", null_env.init(jax.random.PRNGKey(0), B),
         lambda s: s.mask, null_env.step),
        ("ttt/djinn", ttt_template, lambda s: s.legal_action_mask, ttt_step),
        ("2048/djinn-lut", g2048.init(jax.random.PRNGKey(0), B),
         lambda s: s.action_mask,
         lambda s, a, k: g2048.step(s, a, k)[0]),
        ("soko/djinn", soko.init(jax.random.PRNGKey(0), B),
         lambda s: jnp.ones((B, 4), dtype=jnp.bool_),
         lambda s, a, k: soko.step(s, a, k)[0]),
    ]


def main():
    print(f"backend: {jax.default_backend()}")
    check_sampler_uniformity()
    for B in (64, 1024, 8192, 65536):
        print(f"\n===== B={B} =====")
        for name, state0, mask_of, step_of in engines(B):
            s0a, run1 = runner_v1(None, state0, mask_of, step_of)
            v1 = bench(f"{name}  [v1]", s0a, run1, B)
            # fresh state for v2 (v1 consumed/evolved its own)
            s0b = jax.tree_util.tree_map(jnp.copy, state0)
            _, run2 = runner_v2(None, s0b, mask_of, step_of)
            v2 = bench(f"{name}  [v2 agnostic]", s0b, run2, B)
            print(f"  {'':26s} v2/v1 = {v2 / v1:.2f}x")


if __name__ == "__main__":
    main()
