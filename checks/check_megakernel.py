#!/usr/bin/env python3
"""Megakernel hardening battery. Run after any change to megakernel_* or
step_lanes. Closes the parity chain, guards boundaries, proves the
properties the chaining feature relies on, and deepens the RNG tests.

Chain closure (the big one): megakernel was bit-verified against ITS OWN
XLA reference — but the headline vs-reference claim needs step_lanes to be the
same GAME as game2048 (which is what's parity-chained to jumanji).
check_move_chain_link proves the deterministic move logic identical;
spawn equivalence is distributional by design (rank-pick uniforms) and
covered by the RNG battery + game2048's own spawn checks.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import djinnax.refs  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from djinnax.game2048 import move_all_directions
from djinnax.megakernel import _fresh_inputs, _move_dir, N_STEPS
from djinnax.megakernel_rng import (
    BLOCK, hash_uniform, run_megakernel_rng, xla_reference_rng_jit,
)


def check_move_chain_link(n_boards: int = 2048, seed: int = 0):
    """Lane-based _move_dir ≡ game2048.move_all_directions (which is
    parity-chained to jumanji): moved boards, rewards, changed flags,
    all 4 directions, random boards."""
    key = jax.random.PRNGKey(seed)
    boards = jax.random.randint(key, (n_boards, 4, 4), 0, 8, dtype=jnp.int32).astype(jnp.int8)
    ref_moved, ref_rewards, ref_can = jax.jit(move_all_directions)(boards)

    flat = boards.reshape(n_boards, 16).astype(jnp.int32)
    lanes = tuple(flat[:, i] for i in range(16))
    for d in range(4):
        new_lanes, reward, changed = jax.jit(lambda l, d=d: _move_dir(l, d))(lanes)
        ours = jnp.stack(new_lanes, axis=-1).reshape(n_boards, 4, 4).astype(jnp.int8)
        assert jnp.array_equal(ours, ref_moved[d]), f"dir {d}: moved boards differ"
        assert jnp.array_equal(reward, ref_rewards[d]), f"dir {d}: rewards differ"
        assert jnp.array_equal(changed, ref_can[d]), f"dir {d}: changed flags differ"
    print(f"chain link OK — lane moves ≡ game2048 (jumanji-chained) on "
          f"{n_boards} boards x 4 dirs (moved/reward/changed)")


def check_analytic_reset_mask():
    """The chaining feature recomputes the mask from the board at entry;
    that is exact only if the analytic single-tile mask == the computed
    mask. Prove it for all 16 cells x both spawn values."""
    for cell in range(16):
        for val in (1, 2):
            flat = jnp.zeros((1, 16), dtype=jnp.int32).at[0, cell].set(val)
            lanes = tuple(flat[:, i] for i in range(16))
            computed = [bool(_move_dir(lanes, d)[2][0]) for d in range(4)]
            y, x = cell // 4, cell % 4
            analytic = [y > 0, x < 3, y < 3, x > 0]
            assert computed == analytic, (
                f"cell {cell} val {val}: computed {computed} vs analytic {analytic}"
            )
    print("analytic reset mask OK — equals computed mask for all 16 cells x 2 values")


def check_b_divisibility_guard():
    """BOTH entry points refuse a non-block-multiple B, and Mode A also
    refuses a mis-shaped uniforms buffer (cross-model audit S/E4: Mode A
    previously launched a truncated grid and returned uninitialized tail
    rows with a full-B output shape)."""
    from djinnax.megakernel import N_UNI, run_megakernel

    bad_board = jnp.zeros((BLOCK + 1, 16), dtype=jnp.int8)
    seed = jnp.asarray([1, 0], dtype=jnp.uint32)
    try:
        run_megakernel_rng(bad_board, seed)
        raise AssertionError("Mode B: non-multiple B did not raise")
    except ValueError as e:
        assert "multiple of BLOCK" in str(e)

    uni = jnp.zeros((N_STEPS, N_UNI, BLOCK + 1), dtype=jnp.float32)
    try:
        run_megakernel(bad_board, uni)
        raise AssertionError("Mode A: non-multiple B did not raise")
    except ValueError as e:
        assert "multiple of BLOCK" in str(e)

    good_board = jnp.zeros((BLOCK, 16), dtype=jnp.int8)
    short_uni = jnp.zeros((N_STEPS - 1, N_UNI, BLOCK), dtype=jnp.float32)
    try:
        run_megakernel(good_board, short_uni)
        raise AssertionError("Mode A: short uniforms buffer did not raise")
    except ValueError as e:
        assert "uniforms shape" in str(e)

    print("B-divisibility guards OK — Mode A + Mode B raise on non-multiple "
          "B; Mode A raises on mis-shaped uniforms")


def check_bit_determinism(Bn=1024):
    board, _ = _fresh_inputs(Bn, 5)
    seed = jnp.asarray([9, 0], dtype=jnp.uint32)
    mk = jax.jit(run_megakernel_rng)
    a = mk(board, seed)
    b = mk(board, seed)
    for x, y, name in zip(a, b, ("board", "score", "done")):
        assert jnp.array_equal(x, y), f"non-deterministic {name}"
    print(f"bit determinism OK — repeated kernel runs identical at B={Bn}")


def check_parity_sweep():
    """Parity across B (incl. exactly-one-block and odd multiples) and seeds."""
    for Bn in (BLOCK, 3 * BLOCK, 1024):
        for seed_val in (1, 2, 3):
            board, _ = _fresh_inputs(Bn, seed_val)
            seed = jnp.asarray([seed_val, 0], dtype=jnp.uint32)
            m = jax.jit(run_megakernel_rng)(board, seed)
            x = xla_reference_rng_jit(board, seed)
            for a, b, name in zip(m, x, ("board", "score", "done")):
                assert jnp.array_equal(a, b), f"B={Bn} seed={seed_val}: {name} diverges"
    print(f"parity sweep OK — B in {{{BLOCK}, {3*BLOCK}, 1024}} x 3 seeds bit-identical")


def check_adversarial_boards(Bn=BLOCK):
    """Deadlocks, empties, and merge storms through both paths."""
    checker = np.zeros((4, 4), np.int8)
    checker[::2, ::2] = 1; checker[1::2, 1::2] = 1
    checker[::2, 1::2] = 2; checker[1::2, ::2] = 2   # full, no equal neighbors
    cases = {
        "checkerboard deadlock": checker,
        "all zeros": np.zeros((4, 4), np.int8),
        "all 15 merge storm": np.full((4, 4), 15, np.int8),
        "near-saturation rows": np.asarray(
            [[14, 14, 15, 15], [13, 13, 14, 14], [0, 0, 0, 0], [1, 1, 2, 2]], np.int8),
    }
    for name, grid in cases.items():
        board = jnp.asarray(np.tile(grid.reshape(1, 16), (Bn, 1)), dtype=jnp.int8)
        seed = jnp.asarray([11, 0], dtype=jnp.uint32)
        m = jax.jit(run_megakernel_rng)(board, seed)
        x = xla_reference_rng_jit(board, seed)
        for a, b, field in zip(m, x, ("board", "score", "done")):
            assert jnp.array_equal(a, b), f"{name}: {field} diverges"
    print(f"adversarial boards OK — {len(cases)} cases x {N_STEPS} steps bit-identical "
          "(deadlock->reset, empty->reset, saturation merges)")


def check_chained_rollout(Bn=1024, seed_val=21):
    """Two 32-step launches (score carried, t_offset=32 on the second)
    ≡ one 64-step launch, bit-for-bit. This is the production chunked-
    rollout mode."""
    board, _ = _fresh_inputs(Bn, seed_val)
    one = jax.jit(lambda b, s: run_megakernel_rng(b, s, n_steps=64))(
        board, jnp.asarray([seed_val, 0], dtype=jnp.uint32))
    half = jax.jit(lambda b, s, sc: run_megakernel_rng(b, s, score=sc, n_steps=32))
    b1, s1, _ = half(board, jnp.asarray([seed_val, 0], dtype=jnp.uint32),
                     jnp.zeros((Bn,), jnp.float32))
    b2, s2, d2 = half(b1, jnp.asarray([seed_val, 32], dtype=jnp.uint32), s1)
    assert jnp.array_equal(one[0], b2), "chained boards diverge"
    assert jnp.array_equal(one[1], s2), "chained scores diverge"
    assert jnp.array_equal(one[2], d2), "chained done diverges"
    print("chained rollout OK — 2 x 32-step launches ≡ 1 x 64-step launch bit-for-bit")


def check_rng_deep(n=1 << 18):
    """Beyond the per-salt battery: correlations across steps, adjacent
    env ids, and seeds — the axes counter RNGs actually fail on."""
    env = jnp.arange(n, dtype=jnp.uint32)
    seed = jnp.uint32(3)
    u_t = [np.asarray(hash_uniform(env, jnp.int32(t), 0, seed)) for t in (5, 6, 7)]
    for i in range(2):
        c = float(np.corrcoef(u_t[i], u_t[i + 1])[0, 1])
        assert abs(c) < 0.01, f"step correlation t={5+i}/{6+i}: {c:.4f}"
    u = np.asarray(hash_uniform(env, jnp.int32(9), 0, seed))
    c = float(np.corrcoef(u[:-1], u[1:])[0, 1])
    assert abs(c) < 0.01, f"adjacent-env correlation: {c:.4f}"
    ua = np.asarray(hash_uniform(env, jnp.int32(9), 0, jnp.uint32(1)))
    ub = np.asarray(hash_uniform(env, jnp.int32(9), 0, jnp.uint32(2)))
    c = float(np.corrcoef(ua, ub)[0, 1])
    assert abs(c) < 0.01, f"cross-seed correlation: {c:.4f}"
    print(f"RNG deep OK — {n} draws: cross-step, adjacent-env, cross-seed "
          "correlations all < 0.01")


if __name__ == "__main__":
    check_move_chain_link()
    check_analytic_reset_mask()
    check_b_divisibility_guard()
    check_bit_determinism()
    check_parity_sweep()
    check_adversarial_boards()
    check_chained_rollout()
    check_rng_deep()
    print("ALL MEGAKERNEL HARDENING CHECKS PASSED")
