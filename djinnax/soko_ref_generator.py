"""Fixed-level Generator injected into jumanji's Sokoban — both engines draw
from the identical soko_levels fixture, so the level distribution cannot
differ between them.

Reference glue: importing this module requires the reference clones and
their deps (see HOW_TO_RUN.md). The rest of the package never imports it.
"""

from __future__ import annotations

import djinnax.refs  # noqa: F401

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.sokoban.generator import Generator
from jumanji.environments.routing.sokoban.types import State

from djinnax.soko_levels import AGENT_YX, FIXED_LEVELS, N_LEVELS, VARIABLE_LEVELS

_FIXED = jnp.asarray(FIXED_LEVELS)
_VARIABLE = jnp.asarray(VARIABLE_LEVELS)
_AGENTS = jnp.asarray(AGENT_YX)


class FixedLevelsGenerator(Generator):
    """Uniform draw from the shared fixture (jit/vmap-safe)."""

    def __init__(self) -> None:
        self._fixed_grids = _FIXED
        self._variable_grids = _VARIABLE

    def __call__(self, rng_key: chex.PRNGKey) -> State:
        k_idx, k_state = jax.random.split(rng_key)
        idx = jax.random.randint(k_idx, (), 0, N_LEVELS)
        return State(
            key=k_state,
            fixed_grid=_FIXED[idx],
            variable_grid=_VARIABLE[idx],
            agent_location=_AGENTS[idx],
            step_count=jnp.array(0, jnp.int32),
        )
