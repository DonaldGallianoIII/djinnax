"""Sokoban in the djinn house style — head-to-head port of jumanji's env.

Env #3 of the paradigm test. Unlike 2048, jumanji's Sokoban step is already
select-based (no while_loop) — this measures the paradigm on reference code
that is NOT pathologically control-flow-heavy.

Semantics (parity-gated in check_parity.py): jumanji constants/encoding,
MOVES order 0=up 1=right 2=down 3=left, blocked/push-blocked actions are
no-ops, DenseReward = +1 per box gaining a target -1 per box leaving
+10 on completion -0.1 per step, done = solved | step_count >= 120.

House style: batch-native (B,10,10) uint8 grids, flat-index gathers +
one-hot where-writes (fully branchless), in-step auto-reset sampling from
the shared soko_levels fixture, key threaded per step. Step also builds
the stacked (B,10,10,2) observation and the solved/prop extras jumanji's
step produces, so per-step work is comparable.
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from djinnax.soko_levels import (
    AGENT_YX, FIXED_LEVELS, GRID, N_BOXES, N_LEVELS, VARIABLE_LEVELS,
    AGENT, BOX, EMPTY, TARGET, WALL,
)

MOVES = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)
TIME_LIMIT = 120
LEVEL_COMPLETE_BONUS = 10.0
SINGLE_BOX_BONUS = 1.0
STEP_BONUS = -0.1

_FIXED = jnp.asarray(FIXED_LEVELS)          # (N, 10, 10) uint8
_VARIABLE = jnp.asarray(VARIABLE_LEVELS)    # (N, 10, 10) uint8
_AGENTS = jnp.asarray(AGENT_YX)             # (N, 2) int32

# The carried on-target count (review P6) resets to 0, not to a counted
# value — legal only because every fixture level starts with all boxes
# off-target. Assert that property once at import so a future fixture
# change cannot silently break the carry.
import numpy as _np
assert not _np.any(
    (_np.asarray(VARIABLE_LEVELS) == BOX) & (_np.asarray(FIXED_LEVELS) == TARGET)
), 'a fixture level starts with a box on target - carried reset-to-0 invalid'


@flax.struct.dataclass
class SokoState:
    fixed_grid: jax.Array     # (B, 10, 10) uint8
    variable_grid: jax.Array  # (B, 10, 10) uint8
    agent_yx: jax.Array       # (B, 2) int32
    n_on_target: jax.Array    # (B,) int32 — carried count (review P6)
    step_count: jax.Array     # (B,) int16
    terminated: jax.Array     # (B,) bool — done flag of the step just taken


def _flat(yx):
    return yx[..., 0] * GRID + yx[..., 1]                        # (B,)


def _gather(grid_flat, idx):
    return jnp.take_along_axis(grid_flat, idx[:, None], axis=-1)[:, 0]


def _count_on_target(variable, fixed):
    return jnp.sum((variable == BOX) & (fixed == TARGET), axis=(-2, -1))


class DjinnSokoban:
    n_actions: int = 4

    def __init__(self, carry_on_target: bool = False,
                 batch_gated_reset: bool = False):
        # Carrying n_on_target measured NULL (0.99-1.02x at every B, n=5
        # interleaved — data/p6_soko_carry_ab.jsonl): XLA fuses the count
        # reduction for free. Default stays the simpler recount; the flag
        # and field remain so the receipt is reproducible. Rewards are
        # bit-identical either way (exact jumanji replay gates both).
        self._carry = carry_on_target
        # batch_gated_reset (audit PERF-02): lax.cond(any(done)) skips
        # level sampling+gather on steps where NO env terminated. Under
        # the bench's synchronized episodes (~1 reset step in 120) the
        # skip is nearly free; under desynchronized training at large B,
        # any(done) is ~always true and the cond is pure overhead —
        # measured in both regimes (data/ps1_soko_gated_ab.jsonl).
        # DEFAULT OFF: the head-to-head runs the synchronized regime, and
        # a default that wins only there would inflate the published row.
        self._batch_gated = batch_gated_reset

    def _sample_levels(self, key: jax.Array, B: int):
        idx = jax.random.randint(key, (B,), 0, N_LEVELS)
        return _FIXED[idx], _VARIABLE[idx], _AGENTS[idx]

    def init(self, key: jax.Array, n_envs: int) -> SokoState:
        f, v, a = self._sample_levels(key, n_envs)
        return SokoState(
            fixed_grid=f, variable_grid=v, agent_yx=a,
            n_on_target=_count_on_target(v, f).astype(jnp.int32),
            step_count=jnp.zeros((n_envs,), dtype=jnp.int16),
            terminated=jnp.zeros((n_envs,), dtype=jnp.bool_),
        )

    def step(self, state: SokoState, action: jax.Array, key: jax.Array):
        """action (B,) int32. Returns (state, reward, obs, extras)."""
        B = action.shape[0]
        f_flat = state.fixed_grid.reshape(B, GRID * GRID)
        v_flat = state.variable_grid.reshape(B, GRID * GRID)

        delta = MOVES[action]                                    # (B, 2)
        new_loc = state.agent_yx + delta
        box_loc = new_loc + delta
        in1 = jnp.all((new_loc >= 0) & (new_loc < GRID), axis=-1)
        in2 = jnp.all((box_loc >= 0) & (box_loc < GRID), axis=-1)

        idx_agent = _flat(state.agent_yx)
        idx_new = _flat(jnp.clip(new_loc, 0, GRID - 1))
        idx_box = _flat(jnp.clip(box_loc, 0, GRID - 1))

        dest_wall = (_gather(f_flat, idx_new) == WALL) | ~in1
        dest_box = (_gather(v_flat, idx_new) == BOX) & in1
        push_blocked = (
            (_gather(v_flat, idx_box) == BOX)
            | (_gather(f_flat, idx_box) == WALL)
            | ~in2
        )
        can_push = dest_box & ~push_blocked
        moves = (~dest_wall & ~dest_box) | can_push              # (B,)

        cells = jnp.arange(GRID * GRID)[None, :]
        clear_agent = (cells == idx_agent[:, None]) & moves[:, None]
        set_agent = (cells == idx_new[:, None]) & moves[:, None]
        set_box = (cells == idx_box[:, None]) & can_push[:, None]

        # P6: n_before is the previous step's n_after for non-reset envs
        # (and 0 after reset, by the fixture property asserted above) —
        # carrying it halves the (B,100) count reductions per step.
        if self._carry:
            n_before = state.n_on_target
        else:
            n_before = _count_on_target(state.variable_grid, state.fixed_grid)

        new_v = jnp.where(clear_agent, EMPTY, v_flat)
        new_v = jnp.where(set_agent, AGENT, new_v)
        new_v = jnp.where(set_box, BOX, new_v).astype(jnp.uint8)
        new_v = new_v.reshape(B, GRID, GRID)
        new_agent = jnp.where(moves[:, None], new_loc, state.agent_yx)
        new_count = state.step_count + 1

        n_after = _count_on_target(new_v, state.fixed_grid)
        solved = n_after == N_BOXES
        reward = (
            SINGLE_BOX_BONUS * (n_after - n_before).astype(jnp.float32)
            + LEVEL_COMPLETE_BONUS * solved.astype(jnp.float32)
            + STEP_BONUS
        )
        done = solved | (new_count >= TIME_LIMIT)

        # Extras describe the transition just taken (terminal values on
        # terminal rows) — jumanji AutoResetWrapper keeps these too.
        extras = {
            "prop_correct_boxes": n_after.astype(jnp.float32) / N_BOXES,
            "solved": solved,
        }

        # In-step auto-reset from the shared fixture. Same-key sampling in
        # both forms, so gated ≡ ungated bit-for-bit whenever any(done).
        def _with_reset(op):
            fixed, v, agent = op
            rf, rv, ra = self._sample_levels(key, B)
            return (jnp.where(done[:, None, None], rf, fixed),
                    jnp.where(done[:, None, None], rv, v),
                    jnp.where(done[:, None], ra, agent))

        op = (state.fixed_grid, new_v, new_agent)
        if self._batch_gated:
            new_fixed, new_v, new_agent = jax.lax.cond(
                jnp.any(done), _with_reset, lambda o: o, op)
        else:
            new_fixed, new_v, new_agent = _with_reset(op)
        new_state = SokoState(
            fixed_grid=new_fixed,
            variable_grid=new_v,
            agent_yx=new_agent,
            n_on_target=jnp.where(done, 0, n_after).astype(jnp.int32),
            step_count=jnp.where(done, 0, new_count).astype(jnp.int16),
            terminated=done,
        )
        # Observation describes the RETURNED state — on terminal rows that
        # is the freshly reset level, matching jumanji's AutoResetWrapper
        # (reward/extras stay those of the terminal transition). Audit S1:
        # previously terminal rows paired the pre-reset observation with
        # the post-reset state, so a policy acting on obs would act on a
        # level that no longer existed.
        obs = jnp.stack(
            [new_state.variable_grid, new_state.fixed_grid], axis=-1
        )                                                        # (B,10,10,2)
        return new_state, reward, obs, extras
