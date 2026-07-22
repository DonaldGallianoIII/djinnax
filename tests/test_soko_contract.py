"""Sokoban terminal-step contract (cross-model audit S1): the returned
observation must describe the RETURNED state. On terminal rows that is
the freshly reset level — the jumanji AutoResetWrapper convention —
while reward/extras keep the terminal transition's values.

Pure-djinn coherence property: needs no reference clones, so it runs on
CPU in every fresh clone. Covers BOTH termination paths (solve and
time-limit), the two cases the reference replay's random policy almost
never distinguishes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from djinnax.sokoban import (
    LEVEL_COMPLETE_BONUS, SINGLE_BOX_BONUS, STEP_BONUS, TIME_LIMIT,
    DjinnSokoban, SokoState, _count_on_target,
)
from djinnax.soko_levels import AGENT, BOX, GRID, N_BOXES, TARGET, WALL

assert N_BOXES == 4, "crafted layouts below assume the 4-box fixture spec"


def _state(v, f, agent_yx, step_count=0):
    v, f = jnp.asarray(v)[None], jnp.asarray(f)[None]
    return SokoState(
        fixed_grid=f, variable_grid=v,
        agent_yx=jnp.asarray([agent_yx], jnp.int32),
        n_on_target=_count_on_target(v, f).astype(jnp.int32),
        step_count=jnp.full((1,), step_count, jnp.int16),
        terminated=jnp.zeros((1,), jnp.bool_),
    )


def _one_push_from_solved():
    """3 boxes on target, 4th box one push (action 0 = up) from its
    target, agent directly below it."""
    f = np.zeros((GRID, GRID), np.uint8)
    v = np.zeros((GRID, GRID), np.uint8)
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = WALL
    for y, x in [(2, 2), (2, 4), (2, 6)]:
        f[y, x] = TARGET
        v[y, x] = BOX
    f[3, 5] = TARGET
    v[4, 5] = BOX
    v[5, 5] = AGENT
    return _state(v, f, (5, 5))


def _assert_obs_describes(state, obs):
    assert np.array_equal(np.asarray(obs[..., 0]), np.asarray(state.variable_grid))
    assert np.array_equal(np.asarray(obs[..., 1]), np.asarray(state.fixed_grid))


def test_solve_termination_returns_reset_observation():
    game = DjinnSokoban()
    state, reward, obs, extras = game.step(
        _one_push_from_solved(), jnp.zeros((1,), jnp.int32), jax.random.PRNGKey(0)
    )
    assert bool(state.terminated[0])
    assert bool(extras["solved"][0])
    assert float(extras["prop_correct_boxes"][0]) == 1.0
    # reward is the TERMINAL transition's: +1 box gained, +10 solve, -0.1
    assert float(reward[0]) == pytest.approx(
        SINGLE_BOX_BONUS + LEVEL_COMPLETE_BONUS + STEP_BONUS
    )
    # returned state is a fresh fixture level...
    assert int(state.step_count[0]) == 0
    assert int(state.n_on_target[0]) == 0
    # ...and obs describes THAT level, not the solved board (fixture
    # levels start with every box off-target; the solved board had all 4
    # on-target, so coherence + this count fully separates the two).
    _assert_obs_describes(state, obs)
    assert int(_count_on_target(obs[..., 0], obs[..., 1])[0]) == 0


def test_time_limit_termination_returns_reset_observation():
    game = DjinnSokoban()
    # Agent walled into pushing nothing; step_count at the brink.
    f = np.zeros((GRID, GRID), np.uint8)
    v = np.zeros((GRID, GRID), np.uint8)
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = WALL
    for y, x in [(2, 2), (2, 4), (2, 6), (3, 5)]:
        f[y, x] = TARGET
    for y, x in [(6, 2), (6, 4), (6, 6), (7, 5)]:
        v[y, x] = BOX
    v[5, 5] = AGENT
    s0 = _state(v, f, (5, 5), step_count=TIME_LIMIT - 1)
    state, reward, obs, extras = game.step(
        s0, jnp.full((1,), 1, jnp.int32), jax.random.PRNGKey(1)
    )
    assert bool(state.terminated[0])
    assert not bool(extras["solved"][0])
    assert float(reward[0]) == pytest.approx(STEP_BONUS)
    assert int(state.step_count[0]) == 0
    _assert_obs_describes(state, obs)


def test_nonterminal_observation_coherence():
    game = DjinnSokoban()
    state = game.init(jax.random.PRNGKey(2), 64)
    for t in range(8):
        k = jax.random.fold_in(jax.random.PRNGKey(3), t)
        action = jax.random.randint(k, (64,), 0, 4)
        state, _, obs, _ = game.step(state, action, jax.random.fold_in(k, 1))
        _assert_obs_describes(state, obs)
