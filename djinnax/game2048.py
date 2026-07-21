"""2048 in the djinn house style — head-to-head port of jumanji's Game2048.

House conventions (the djinnax discipline): batch-native (leading B, NO vmap),
flax.struct state, int8 board (exponents), fully branchless step — where
jumanji uses per-row `lax.while_loop`s and `lax.switch` under vmap, this
port does the whole batch with a stable-argsort compaction + an unrolled
3-pair merge pass. In-step auto-reset; key threaded per step, not stored.

Functional parity with jumanji (verified in check_parity.py): identical
move results and rewards (sum of 2^new_exponent per merge), identical
per-direction action masks, tile spawn 2 (p=0.9) / 4 (p=0.1) on a uniform
empty cell only when the action was legal, done when no direction moves.
Action order matches jumanji: 0=up, 1=right, 2=down, 3=left.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class G2048State:
    board: jax.Array        # (B, 4, 4) int8 — exponents, 0 = empty
    action_mask: jax.Array  # (B, 4) bool — up/right/down/left
    score: jax.Array        # (B,) float32
    step_count: jax.Array   # (B,) int16
    terminated: jax.Array   # (B,) bool


def _compact_left(rows: jax.Array) -> jax.Array:
    """Stable-shift nonzeros left along the last axis. rows: (..., 4).

    Perf v2: rank-scatter instead of argsort — each nonzero's target index
    is the count of nonzeros before it (exclusive cumsum), scattered via a
    4x4 one-hot select. ~16 fused vector ops; argsort lowered to a full
    sorting network with key/value plumbing and was the step's hottest op.
    """
    nz = rows != 0
    rank = jnp.cumsum(nz, axis=-1) - nz.astype(jnp.int32)     # (..., 4)
    dst = jnp.arange(4)
    sel = nz[..., None] & (rank[..., None] == dst)            # (..., 4src, 4dst)
    return (rows[..., None] * sel).sum(axis=-2).astype(rows.dtype)


def _merge_left(rows: jax.Array):
    """One left merge pass on compacted rows (..., 4). Returns (rows, reward).

    Unrolled pair scan (0,1)->(1,2)->(2,3): each tile merges at most once,
    leftmost pairs first — equivalent to the sequential 2048 rule.
    """
    reward = jnp.zeros(rows.shape[:-1], dtype=jnp.float32)
    for i in range(3):
        a, b = rows[..., i], rows[..., i + 1]
        m = (a != 0) & (a == b)
        rows = rows.at[..., i].set(jnp.where(m, a + 1, a))
        rows = rows.at[..., i + 1].set(jnp.where(m, 0, b))
        reward = reward + jnp.where(m, 2.0 ** (a.astype(jnp.float32) + 1.0), 0.0)
    return rows, reward


def _move_left(board: jax.Array):
    """board: (..., 4, 4) -> (moved, reward (...,)). Fully vectorized."""
    rows = _compact_left(board)
    rows, reward = _merge_left(rows)
    return _compact_left(rows), reward.sum(axis=-1)


# jumanji transform_board: action 0 = transpose, 1 = flip cols, 2 = flip of
# transpose, 3 = identity; each is its own inverse.
def _orient(board: jax.Array, action_idx: int) -> jax.Array:
    if action_idx == 0:
        return jnp.swapaxes(board, -2, -1)
    if action_idx == 1:
        return jnp.flip(board, axis=-1)
    if action_idx == 2:
        # jumanji uses jnp.flip(transpose) with NO axis arg = flip over BOTH axes
        return jnp.flip(jnp.swapaxes(board, -2, -1), axis=(-2, -1))
    return board


def move_all_directions(board: jax.Array, move_left_fn=None):
    """All 4 moves for the whole batch. board: (B, 4, 4).

    Returns moved (4, B, 4, 4), rewards (4, B), can_move (4, B).
    The 4 orientations are compile-time constants (python loop unrolls) —
    no lax.switch, no vmap, no while_loop. `move_left_fn` is pluggable so
    the bitboard-LUT variant (game2048_lut) reuses this structure.
    """
    fn = move_left_fn or _move_left
    moved, rewards, can = [], [], []
    for a in range(4):
        m, r = fn(_orient(board, a))
        m = _orient(m, a)
        moved.append(m)
        rewards.append(r)
        can.append(jnp.any(m != board, axis=(-2, -1)))
    return jnp.stack(moved), jnp.stack(rewards), jnp.stack(can)


def _spawn(board: jax.Array, key: jax.Array, enabled: jax.Array):
    """Add a 1-exp (p=.9) or 2-exp (p=.1) tile on a uniform empty cell.

    board: (B, 4, 4); enabled: (B,) bool gates the write (illegal actions
    spawn nothing, matching jumanji's cond). Returns (board, cell) — cell
    (B,) int32 flat index of the spawned tile (used by the analytic
    fresh-board mask).
    """
    B = board.shape[0]
    flat = board.reshape(B, 16)
    empty = flat == 0
    k1, k2 = jax.random.split(key)
    logits = jnp.where(empty, 0.0, -jnp.inf)
    # Guard all-full rows (can't happen when enabled, but keep logits finite)
    logits = jnp.where(jnp.any(empty, axis=-1, keepdims=True), logits, 0.0)
    cell = jax.random.categorical(k1, logits, axis=-1)        # (B,)
    val = jnp.where(
        jax.random.uniform(k2, (B,)) < 0.9, jnp.int8(1), jnp.int8(2)
    )
    onehot = jax.nn.one_hot(cell, 16, dtype=jnp.bool_)
    write = onehot & enabled[:, None] & empty
    out = jnp.where(write, val[:, None], flat).reshape(B, 4, 4).astype(jnp.int8)
    return out, cell


def _single_tile_mask(cell: jax.Array) -> jax.Array:
    """Perf v2: legal mask of a fresh board holding ONE tile at flat `cell` —
    analytic, no move passes. Tile at (y, x): up legal iff y>0, right iff
    x<3, down iff y<3, left iff x>0 (jumanji action order 0..3).
    Replaces 4 full move passes per step for the reset template.
    """
    y, x = cell // 4, cell % 4
    return jnp.stack([y > 0, x < 3, y < 3, x > 0], axis=-1)   # (B, 4)


class Djinn2048:
    """Batch-native 2048, jumanji-parity semantics.

    `move_left_fn` swaps the row-move kernel (v2 branchless default; the
    bitboard-LUT v3 lives in game2048_lut.py).
    """

    n_actions: int = 4

    def __init__(self, move_left_fn=None):
        self._move_left = move_left_fn or _move_left

    def init(self, key: jax.Array, n_envs: int) -> G2048State:
        B = n_envs
        board = jnp.zeros((B, 4, 4), dtype=jnp.int8)
        board, _ = _spawn(board, key, jnp.ones((B,), dtype=jnp.bool_))
        _, _, can = move_all_directions(board, self._move_left)
        return G2048State(
            board=board,
            action_mask=can.T,                                 # (B, 4)
            score=jnp.zeros((B,), dtype=jnp.float32),
            step_count=jnp.zeros((B,), dtype=jnp.int16),
            terminated=jnp.zeros((B,), dtype=jnp.bool_),
        )

    def step(self, state: G2048State, action: jax.Array, key: jax.Array):
        """action: (B,) int32. Returns (state, reward (B,)). In-step auto-reset.

        The chosen move is ONE move pass: select-orient the board by action
        (pure data movement), move-left once, inverse-orient. The per-step
        move-pass budget is 1 (move) + 4 (next mask) + 4 (reset template
        mask) — comparable to jumanji's 1 move + 4 can_move + vmap-forced
        reset branch.
        """
        B = action.shape[0]
        sel = action[None, :, None, None] == jnp.arange(4)[:, None, None, None]
        oriented = jnp.sum(
            jnp.where(sel, jnp.stack([_orient(state.board, a) for a in range(4)]), 0),
            axis=0,
        ).astype(jnp.int8)
        moved_o, reward = self._move_left(oriented)
        new_board = jnp.sum(
            jnp.where(sel, jnp.stack([_orient(moved_o, a) for a in range(4)]), 0),
            axis=0,
        ).astype(jnp.int8)

        was_legal = jnp.take_along_axis(
            state.action_mask, action[:, None].astype(jnp.int32), axis=-1
        )[:, 0]
        reward = jnp.where(was_legal, reward, 0.0)

        k_spawn, k_reset = jax.random.split(key)
        new_board, _ = _spawn(new_board, k_spawn, was_legal)

        _, _, can = move_all_directions(new_board, self._move_left)  # next mask
        mask = can.T                                           # (B, 4)
        done = ~jnp.any(mask, axis=-1)

        # In-step auto-reset (house style): fresh single-tile board per env.
        # Its mask is analytic (perf v2) — no move passes.
        fresh, fresh_cell = _spawn(
            jnp.zeros((B, 4, 4), dtype=jnp.int8), k_reset,
            jnp.ones((B,), dtype=jnp.bool_),
        )
        new_board = jnp.where(done[:, None, None], fresh, new_board)
        mask = jnp.where(done[:, None], _single_tile_mask(fresh_cell), mask)

        state = G2048State(
            board=new_board,
            action_mask=mask,
            score=jnp.where(done, 0.0, state.score + reward),
            step_count=jnp.where(done, 0, state.step_count + 1).astype(jnp.int16),
            terminated=done,
        )
        return state, reward
