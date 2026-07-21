#!/usr/bin/env python3
"""Megakernel 2048 — the whole rollout in ONE kernel launch (Mode A).

See MEGAKERNEL_PLAN.md. Grid over env blocks; each program owns BLOCK
envs and runs all N_STEPS inside a lax.fori_loop, state living entirely
in registers as 16 structure-of-arrays board lanes.

The core trick for parity: `step_lanes()` is a pure jnp function on lane
tuples consuming pre-generated uniforms. The SAME function is executed
(a) inside the Triton kernel body and (b) under an XLA lax.scan — so the
two paths are bit-identical by construction, and any timing difference is
pure execution strategy: one persistent launch vs an op-graph of many
kernels per step. RNG-generation cost is excluded from BOTH sides (the
uniform buffer is an input), isolating the launch/fusion question.

Semantics = game2048 (jumanji parity family): action order
0=up 1=right 2=down 3=left; spawn 2 (p=.9)/4 (p=.1) uniform over empty
cells (rank-pick form); illegal action -> no move, no spawn; done when no
direction moves; in-step reset to a fresh single-tile board with the
analytic mask. Spawn cell selection uses rank-pick on a raw uniform
(same distribution as categorical-over-empties; different mapping than
jax's Gumbel — which is why the XLA reference here consumes the same
uniforms rather than jax.random keys).
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

N_STEPS = 64
BLOCK = 128
N_UNI = 5           # uniforms per step: action, spawn cell, spawn val, reset cell, reset val
_TRITON = dict(compiler_params=plt.CompilerParams())

# Direction lane wiring (cell = y*4+x). Each direction = row groups in
# push order; moving = shifting toward group index 0.
_GROUPS = {
    0: [[x, 4 + x, 8 + x, 12 + x] for x in range(4)],          # up: columns
    1: [[4 * y + 3, 4 * y + 2, 4 * y + 1, 4 * y] for y in range(4)],   # right: reversed rows
    2: [[12 + x, 8 + x, 4 + x, x] for x in range(4)],           # down: reversed columns
    3: [[4 * y, 4 * y + 1, 4 * y + 2, 4 * y + 3] for y in range(4)],   # left: rows
}


def _bubble(a, b):
    z = a == 0
    return jnp.where(z, b, a), jnp.where(z, 0, b)


def _row_move_4(a, b, c, d):
    """Move one 4-lane group toward lane a. Returns 4 lanes + reward."""
    for _ in range(3):
        a, b = _bubble(a, b)
        b, c = _bubble(b, c)
        c, d = _bubble(c, d)
    reward = jnp.zeros_like(a, dtype=jnp.float32)

    def merge(x, y, reward):
        m = (x != 0) & (x == y)
        reward = reward + jnp.where(
            m, (jnp.int32(1) << (x + 1)).astype(jnp.float32), 0.0)
        return jnp.where(m, x + 1, x), jnp.where(m, 0, y), reward

    a, b, reward = merge(a, b, reward)
    b, c, reward = merge(b, c, reward)
    c, d, reward = merge(c, d, reward)
    for _ in range(3):
        a, b = _bubble(a, b)
        b, c = _bubble(b, c)
        c, d = _bubble(c, d)
    return a, b, c, d, reward


def _move_dir(lanes, d):
    """Apply direction d (static). Returns (new_lanes 16-tuple, reward, changed)."""
    new = list(lanes)
    reward = jnp.zeros_like(lanes[0], dtype=jnp.float32)
    for group in _GROUPS[d]:
        g = [lanes[i] for i in group]
        a, b, c, e, r = _row_move_4(*g)
        for idx, val in zip(group, (a, b, c, e)):
            new[idx] = val
        reward = reward + r
    changed = jnp.zeros_like(lanes[0], dtype=jnp.bool_)
    for i in range(16):
        changed = changed | (new[i] != lanes[i])
    return tuple(new), reward, changed


def _select4(options, action):
    """options: list of 4 lane-tuples/arrays; select per-env by action."""
    if isinstance(options[0], tuple):
        return tuple(
            _select4([o[i] for o in options], action) for i in range(16)
        )
    out = options[0]
    for d in (1, 2, 3):
        out = jnp.where(action == d, options[d], out)
    return out


# Orient-select wiring (P1): direction d's move IS the canonical left move
# on lanes permuted group-major by _GROUPS[d]. _PERM[d][slot] = source cell
# for canonical slot; _INV[d][cell] = canonical slot holding that cell.
# _PERM[3] is the identity (left is canonical), and reward accumulation
# order (canonical group g == _GROUPS[d][g]) matches _move_dir exactly,
# so the selected direction's outputs are bit-identical to _move_dir's.
_PERM = {d: [c for group in _GROUPS[d] for c in group] for d in range(4)}
_INV = {d: [0] * 16 for d in range(4)}
for _d in range(4):
    for _slot, _cell in enumerate(_PERM[_d]):
        _INV[_d][_cell] = _slot


def _permute_by_action(lanes, action, table):
    """lanes[table[d][j]] selected per-env by action — 3 wheres per slot."""
    out = []
    for j in range(16):
        v = lanes[table[0][j]]
        for d in (1, 2, 3):
            v = jnp.where(action == d, lanes[table[d][j]], v)
        out.append(v)
    return tuple(out)


def _apply_move_oriented(lanes, action):
    """ONE move network: permute into canonical layout by action, move
    left, inverse-permute. ~35-40% less ALU than computing all four
    directions and selecting (external review R1 finding P1)."""
    sel = _permute_by_action(lanes, action, _PERM)
    new_sel = list(sel)
    reward = jnp.zeros_like(lanes[0], dtype=jnp.float32)
    for g in range(4):
        a, b, c, e, r = _row_move_4(*new_sel[4 * g : 4 * g + 4])
        new_sel[4 * g : 4 * g + 4] = (a, b, c, e)
        reward = reward + r
    return _permute_by_action(tuple(new_sel), action, _INV), reward


def _apply_move_allmoves(lanes, action):
    """Reference variant: all four direction networks, then select —
    kept for the interleaved A/B against _apply_move_oriented."""
    moved, rewards = [], []
    for d in range(4):
        nl, rw, _ = _move_dir(lanes, d)
        moved.append(nl)
        rewards.append(rw)
    return _select4(moved, action), _select4(rewards, action)


def _spawn_lanes(lanes, u_cell, u_val, enabled):
    """Rank-pick a uniform empty cell; write exp 1 (p=.9) / 2 (p=.1)."""
    empties = [(l == 0) for l in lanes]
    n_empty = empties[0].astype(jnp.int32)
    for e in empties[1:]:
        n_empty = n_empty + e.astype(jnp.int32)
    n_safe = jnp.maximum(n_empty, 1)
    r = jnp.minimum((u_cell * n_safe.astype(jnp.float32)).astype(jnp.int32),
                    n_safe - 1)
    val = jnp.where(u_val < 0.9, jnp.int32(1), jnp.int32(2))
    out = []
    csum = jnp.zeros_like(n_empty)
    for i in range(16):
        hit = empties[i] & (csum == r) & enabled & (n_empty > 0)
        out.append(jnp.where(hit, val, lanes[i]))
        csum = csum + empties[i].astype(jnp.int32)
    return tuple(out), r


def _make_step(apply_move):
    """Build a step function around one of the two move-apply variants.
    Everything except step 2 is shared, so the variants cannot drift."""

    def step(lanes, mask, score, u):
        u_act, u_cell, u_val, u_rcell, u_rval = u

        # 1. sample action from carried mask (rank-pick)
        n_legal = mask[0].astype(jnp.int32)
        for m in mask[1:]:
            n_legal = n_legal + m.astype(jnp.int32)
        n_safe = jnp.maximum(n_legal, 1)
        r = jnp.minimum((u_act * n_safe.astype(jnp.float32)).astype(jnp.int32),
                        n_safe - 1)
        csum = jnp.zeros_like(n_legal)
        action = jnp.zeros_like(n_legal)
        for d in range(4):
            hit = mask[d] & (csum == r)
            action = jnp.where(hit, d, action)
            csum = csum + mask[d].astype(jnp.int32)
        was_legal = jnp.zeros_like(mask[0])
        for d in range(4):
            was_legal = was_legal | (mask[d] & (action == d))

        # 2. apply the chosen move (variant point: oriented vs all-moves)
        new_lanes, reward = apply_move(lanes, action)
        reward = jnp.where(was_legal, reward, 0.0)

        # 3. spawn (gated on legality)
        new_lanes, _ = _spawn_lanes(new_lanes, u_cell, u_val, was_legal)

        # 4. next mask from the post-spawn board
        new_mask = []
        for d in range(4):
            _, _, ch = _move_dir(new_lanes, d)
            new_mask.append(ch)
        done = ~(new_mask[0] | new_mask[1] | new_mask[2] | new_mask[3])

        # 5. in-register reset where done: fresh single-tile board + analytic mask
        rcell = jnp.minimum((u_rcell * 16.0).astype(jnp.int32), 15)
        rval = jnp.where(u_rval < 0.9, jnp.int32(1), jnp.int32(2))
        fresh = tuple(
            jnp.where(done & (rcell == i), rval, jnp.where(done, 0, new_lanes[i]))
            for i in range(16)
        )
        ry, rx = rcell // 4, rcell % 4
        analytic = (ry > 0, rx < 3, ry < 3, rx > 0)             # up,right,down,left
        out_mask = tuple(
            jnp.where(done, analytic[d], new_mask[d]) for d in range(4)
        )
        new_score = jnp.where(done, 0.0, score + reward)
        return fresh, out_mask, new_score, done

    return step


# Production step (P1 orient-select) and the pre-P1 reference variant.
step_lanes = _make_step(_apply_move_oriented)
step_lanes_allmoves = _make_step(_apply_move_allmoves)
step_lanes.__doc__ = """One env step on 16 board lanes. u = 5 uniforms
(BLOCK,) each. Runs identically inside the kernel and under XLA scan."""


def _initial_mask(lanes):
    return tuple(_move_dir(lanes, d)[2] for d in range(4))


# --- the megakernel ---------------------------------------------------------


def _make_mega_kernel(step):
    def kernel(board_ref, uni_ref, out_board_ref, out_score_ref, out_done_ref):
        lanes = tuple(board_ref[:, i].astype(jnp.int32) for i in range(16))
        mask = _initial_mask(lanes)
        score = jnp.zeros_like(lanes[0], dtype=jnp.float32)
        done = jnp.zeros_like(lanes[0], dtype=jnp.bool_)

        def body(t, carry):
            lanes, mask, score, done = carry
            u = tuple(uni_ref[t, j, :] for j in range(N_UNI))
            lanes, mask, score, done = step(lanes, mask, score, u)
            return (lanes, mask, score, done)

        lanes, mask, score, done = lax.fori_loop(0, N_STEPS, body, (lanes, mask, score, done))
        for i in range(16):
            out_board_ref[:, i] = lanes[i].astype(jnp.int8)
        out_score_ref[...] = score
        out_done_ref[...] = done

    return kernel


def run_megakernel(board: jax.Array, uniforms: jax.Array, step_fn=None):
    """board (B, 16) int8; uniforms (N_STEPS, N_UNI, B) f32."""
    Bn = board.shape[0]
    return pl.pallas_call(
        _make_mega_kernel(step_fn or step_lanes),
        out_shape=(
            jax.ShapeDtypeStruct((Bn, 16), jnp.int8),
            jax.ShapeDtypeStruct((Bn,), jnp.float32),
            jax.ShapeDtypeStruct((Bn,), jnp.bool_),
        ),
        grid=(Bn // BLOCK,),
        in_specs=[
            pl.BlockSpec((BLOCK, 16), lambda i: (i, 0)),
            pl.BlockSpec((N_STEPS, N_UNI, BLOCK), lambda i: (0, 0, i)),
        ],
        out_specs=(
            pl.BlockSpec((BLOCK, 16), lambda i: (i, 0)),
            pl.BlockSpec((BLOCK,), lambda i: (i,)),
            pl.BlockSpec((BLOCK,), lambda i: (i,)),
        ),
        **_TRITON,
    )(board, uniforms)


# --- the XLA reference: SAME step function under lax.scan --------------------


def _xla_reference(board: jax.Array, uniforms: jax.Array, step):
    lanes = tuple(board[:, i].astype(jnp.int32) for i in range(16))
    mask = _initial_mask(lanes)
    score = jnp.zeros_like(lanes[0], dtype=jnp.float32)
    done = jnp.zeros_like(lanes[0], dtype=jnp.bool_)

    def body(carry, u):
        lanes, mask, score, done = carry
        lanes, mask, score, done = step(
            lanes, mask, score, tuple(u[j] for j in range(N_UNI)))
        return (lanes, mask, score, done), None

    (lanes, mask, score, done), _ = lax.scan(body, (lanes, mask, score, done), uniforms)
    out_board = jnp.stack([l.astype(jnp.int8) for l in lanes], axis=-1)
    return out_board, score, done


@jax.jit
def run_xla_reference(board: jax.Array, uniforms: jax.Array):
    return _xla_reference(board, uniforms, step_lanes)


# --- parity + bench ----------------------------------------------------------


def _fresh_inputs(Bn, seed):
    kb, ku = jax.random.split(jax.random.PRNGKey(seed))
    board = jnp.zeros((Bn, 16), dtype=jnp.int8)
    cell = jax.random.randint(kb, (Bn,), 0, 16)
    board = jnp.where(jax.nn.one_hot(cell, 16, dtype=jnp.bool_), 1, board).astype(jnp.int8)
    uniforms = jax.random.uniform(ku, (N_STEPS, N_UNI, Bn), dtype=jnp.float32)
    return board, uniforms


def check_parity(Bn=1024, seed=0):
    board, uniforms = _fresh_inputs(Bn, seed)
    mb, ms, md = jax.jit(run_megakernel)(board, uniforms)
    xb, xs, xd = run_xla_reference(board, uniforms)
    assert jnp.array_equal(mb, xb), "boards diverge"
    assert jnp.array_equal(ms, xs), "scores diverge"
    assert jnp.array_equal(md, xd), "done flags diverge"
    print(f"parity OK — megakernel ≡ XLA over {N_STEPS} steps at B={Bn} "
          f"(boards, scores, done bit-identical)")


def interleaved(Bn, rounds=8):
    board, uniforms = _fresh_inputs(Bn, 1)
    mk = jax.jit(run_megakernel)
    t0 = time.perf_counter()
    jax.block_until_ready(mk(board, uniforms))
    c_mk = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(run_xla_reference(board, uniforms))
    c_x = time.perf_counter() - t0
    ratios = []
    for i in range(rounds):
        t0 = time.perf_counter(); jax.block_until_ready(run_xla_reference(board, uniforms)); tx = time.perf_counter() - t0
        t0 = time.perf_counter(); jax.block_until_ready(mk(board, uniforms)); tm = time.perf_counter() - t0
        ratios.append(tx / tm)
    med = statistics.median(ratios)
    sps = Bn * N_STEPS / tm
    print(f"B={Bn:<6d} megakernel-speedup median {med:.2f}x "
          f"[{min(ratios):.2f} .. {max(ratios):.2f}]  "
          f"(last-round {sps:,.0f} env-steps/s; compile mk {c_mk:.1f}s xla {c_x:.1f}s)")


def main():
    print(f"backend: {jax.default_backend()}  (clocks UNLOCKED — medians "
          f"with spread; a multiple is a result, a percentage is not)")
    check_parity()
    for Bn in (1024, 8192, 65536):
        interleaved(Bn)


if __name__ == "__main__":
    main()
