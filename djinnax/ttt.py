"""Tic-tac-toe in the djinn house style — head-to-head port of pgx's rules.

House conventions (the djinnax discipline): batch-native state (leading B on every
field, NO vmap), flax.struct dataclass, int8 dtype economy, branchless step
via jnp.where masking, key threaded per step (not stored). Terminal states
freeze (mask forced all-True) exactly like pgx; the bench resets both
engines identically via where-select to an init template.

Functional parity with pgx tic_tac_toe (verified in check_parity.py):
same board encoding (-1 empty / 0 X / 1 O), same rewards (+1/-1 indexed by
player id with pgx's should-flip convention), same terminal conditions, and
the same per-step framework work pgx's core.Env.step performs: skip-if-
terminated, illegal-action -> immediate loss, terminal mask all-True, and a
(3, 3, 2) observation for the new current player.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

EMPTY = -1
_LINES = jnp.array(
    [[0, 1, 2], [3, 4, 5], [6, 7, 8],
     [0, 3, 6], [1, 4, 7], [2, 5, 8],
     [0, 4, 8], [2, 4, 6]], dtype=jnp.int32,
)                                                            # (8, 3)

# Perf: bitboard win check — a player's occupancy is 9 bits, so "is there a
# completed line" is a 512-entry boolean LUT gathered by the packed occ.
import numpy as _np

_WON_NP = _np.zeros(512, dtype=bool)
for _occ in range(512):
    for _line in ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                  (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)):
        if all(_occ >> c & 1 for c in _line):
            _WON_NP[_occ] = True
WON_LUT = jnp.asarray(_WON_NP)                               # (512,) bool
_BITS = jnp.asarray([1 << i for i in range(9)], dtype=jnp.int32)


@flax.struct.dataclass
class TttState:
    board: jax.Array           # (B, 9) int8 — -1 empty, 0 X, 1 O
    color: jax.Array           # (B,) int8 — whose mark goes next (0=X first)
    current_player: jax.Array  # (B,) int8 — player id to act (pgx random swap)
    winner: jax.Array          # (B,) int8 — -1 none, else color
    terminated: jax.Array      # (B,) bool
    rewards: jax.Array         # (B, 2) float32 — indexed by player id
    observation: jax.Array     # (B, 3, 3, 2) bool — new current player's view
    legal_action_mask: jax.Array  # (B, 9) bool


class DjinnTicTacToe:
    """Batch-native TTT. All methods take/return full-batch arrays.

    win_lut=True swaps the (B, 8, 3) line-gather win check for a packed
    9-bit occupancy + 512-entry LUT gather (bitboard style).
    """

    n_actions: int = 9

    def __init__(self, win_lut: bool = False):
        self._win_lut = win_lut

    def init(self, key: jax.Array, n_envs: int) -> TttState:
        B = n_envs
        board = jnp.full((B, 9), EMPTY, dtype=jnp.int8)
        current_player = jax.random.bernoulli(key, shape=(B,)).astype(jnp.int8)
        state = TttState(
            board=board,
            color=jnp.zeros((B,), dtype=jnp.int8),
            current_player=current_player,
            winner=jnp.full((B,), -1, dtype=jnp.int8),
            terminated=jnp.zeros((B,), dtype=jnp.bool_),
            rewards=jnp.zeros((B, 2), dtype=jnp.float32),
            observation=jnp.zeros((B, 3, 3, 2), dtype=jnp.bool_),
            legal_action_mask=jnp.ones((B, 9), dtype=jnp.bool_),
        )
        return state.replace(observation=self._observe(state))

    def _observe(self, state: TttState) -> jax.Array:
        """(B, 3, 3, 2) planes for the current player's color, pgx layout."""
        # my color: current_player acts with `color`; pgx flips planes so
        # plane 0 = my marks, plane 1 = opponent marks.
        my_color = state.color[:, None]                       # (B, 1)
        mine = (state.board == my_color).reshape(-1, 3, 3)
        theirs = (state.board == (1 - my_color)).reshape(-1, 3, 3)
        return jnp.stack([mine, theirs], axis=-1)

    def step(self, state: TttState, action: jax.Array, key: jax.Array) -> TttState:
        """action: (B,) int32. Branchless; equivalent work to pgx core.step."""
        del key                                               # deterministic game
        B = action.shape[0]
        was_terminated = state.terminated                     # (B,)
        illegal = ~jnp.take_along_axis(
            state.legal_action_mask, action[:, None].astype(jnp.int32), axis=-1
        )[:, 0] & ~was_terminated

        # --- apply move (masked off for already-terminated envs) ---
        onehot = jax.nn.one_hot(action, 9, dtype=jnp.bool_)   # (B, 9)
        place = onehot & ~was_terminated[:, None] & ~illegal[:, None]
        new_board = jnp.where(place, state.color[:, None], state.board).astype(jnp.int8)

        if self._win_lut:
            occ = ((new_board == state.color[:, None]) * _BITS).sum(axis=-1)
            won = WON_LUT[occ]                                 # (B,)
        else:
            lines = new_board[:, _LINES]                       # (B, 8, 3)
            won = jnp.any(jnp.all(lines == state.color[:, None, None], axis=-1), axis=-1)
        new_winner = jnp.where(won & ~was_terminated, state.color, state.winner)

        new_color = jnp.where(was_terminated, state.color, (state.color + 1) % 2).astype(jnp.int8)
        new_player = jnp.where(
            was_terminated, state.current_player, (state.current_player + 1) % 2
        ).astype(jnp.int8)

        full = jnp.all(new_board != EMPTY, axis=-1)
        game_over = (new_winner >= 0) | full
        terminated = was_terminated | game_over | illegal

        # --- rewards, pgx convention: (winner_color -> +1, loser -> -1), then
        # flipped into player-id indexing; illegal actor loses immediately ---
        win_rewards = jnp.where(
            (new_winner[:, None] >= 0)
            & (jnp.arange(2, dtype=jnp.int8)[None, :] == new_winner[:, None]),
            1.0, -1.0,
        )
        win_rewards = jnp.where(new_winner[:, None] >= 0, win_rewards, 0.0)
        # color-indexed -> player-id-indexed: player who just moved has id
        # state.current_player and color state.color.
        flip = (state.current_player != state.color)[:, None]
        win_rewards = jnp.where(flip, win_rewards[:, ::-1], win_rewards)
        illegal_rewards = jnp.where(
            jnp.arange(2, dtype=jnp.int8)[None, :] == state.current_player[:, None],
            -1.0, 0.0,
        )
        rewards = jnp.where(illegal[:, None], illegal_rewards, win_rewards)
        rewards = jnp.where(was_terminated[:, None], 0.0, rewards).astype(jnp.float32)

        mask = (new_board == EMPTY) & ~terminated[:, None]
        # pgx: terminal states force mask all-True (policy-normalization guard)
        mask = jnp.where(terminated[:, None], True, mask)

        state = TttState(
            board=new_board, color=new_color, current_player=new_player,
            winner=new_winner, terminated=terminated, rewards=rewards,
            observation=state.observation, legal_action_mask=mask,
        )
        return state.replace(observation=self._observe(state))
