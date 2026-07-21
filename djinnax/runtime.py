"""Game-AGNOSTIC runtime layer — the machinery every env shares.

Everything here applies to any env plugged into the bench/training loop,
ours or the references': the scan driver, per-step key derivation, the
legal-action sampler, and buffer-donation policy. Optimizing this file
optimizes every game at once; measuring it is the null-env's job.

Protocol v1 (what the head-to-head used so far):
  - keys derived INSIDE the scan body (2-3 fold_ins per step)
  - masked-uniform actions via categorical over where(mask, 0, -inf)
    (one Gumbel per ACTION: O(B*A) exponentials per step)
  - no donation (carry buffers copied every runner invocation)

Protocol v2 (this file):
  - bulk key hoisting: ALL per-step keys derived in one batched split
    outside the loop, threaded through scan xs
  - rank-pick sampler: uniform-over-legal == one uniform per ROW + a
    cumsum rank select (distribution identical to v1's masked categorical
    with equal logits; the random STREAM differs, which is allowed —
    both are exactly uniform over legal actions)
  - jit(donate_argnums=(0,)) on the runner carry
"""

from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp
from jax import lax


# --- sampler ----------------------------------------------------------------


def sample_uniform_legal(key: jax.Array, mask: jax.Array) -> jax.Array:
    """One action per row, uniform over legal entries of mask (..., A).

    Rank-pick: r ~ U[0, n_legal); return the r-th legal index. One uniform
    draw per row instead of one Gumbel per action. Rows with no legal
    action fall back to index 0 (callers per H4 guarantee non-empty rows;
    the guard keeps arithmetic finite).
    """
    n_legal = mask.sum(axis=-1)                                  # (...,)
    n_safe = jnp.maximum(n_legal, 1)
    u = jax.random.uniform(key, n_legal.shape)
    r = jnp.minimum((u * n_safe).astype(jnp.int32), n_safe - 1)  # (...,)
    c = jnp.cumsum(mask.astype(jnp.int32), axis=-1)              # (..., A)
    hit = mask & (c == (r[..., None] + 1))
    return jnp.argmax(hit, axis=-1).astype(jnp.int32)


# --- scan driver ------------------------------------------------------------


def build_runner(one_step, n_steps: int, n_keys_per_step: int = 2,
                 donate: bool = True, unroll: int = 1):
    """Build a jitted runner(carry, key) -> carry.

    `one_step(carry, keys)` receives a tuple of `n_keys_per_step` per-step
    keys (already derived — no fold_in needed in the body) and returns the
    next carry. All n_steps * n_keys_per_step keys are derived in ONE
    batched split before the scan and threaded via xs.
    """

    def runner(carry, key):
        keys = jax.random.split(key, n_steps * n_keys_per_step)
        keys = keys.reshape(n_steps, n_keys_per_step, *keys.shape[1:])

        def body(c, ks):
            return one_step(c, tuple(ks[i] for i in range(n_keys_per_step))), None

        carry, _ = lax.scan(body, carry, keys, unroll=unroll)
        return carry

    if donate:
        return jax.jit(runner, donate_argnums=(0,))
    return jax.jit(runner)


# --- null env: measures the floor -------------------------------------------


@flax.struct.dataclass
class NullState:
    """Minimal state exercising every runtime element and no game logic:
    a mask to sample from, a tiny state array to touch (bandwidth), a
    counter, and a periodic done to exercise the reset select."""
    data: jax.Array        # (B, 16) int8
    mask: jax.Array        # (B, 9) bool
    step_count: jax.Array  # (B,) int16


class NullEnv:
    """step = sample action + touch state + where-reset. Pure floor."""

    n_actions: int = 9

    def init(self, key: jax.Array, n_envs: int) -> NullState:
        del key
        return NullState(
            data=jnp.zeros((n_envs, 16), dtype=jnp.int8),
            mask=jnp.ones((n_envs, 9), dtype=jnp.bool_),
            step_count=jnp.zeros((n_envs,), dtype=jnp.int16),
        )

    def step(self, state: NullState, action: jax.Array, key: jax.Array) -> NullState:
        del key
        # Write, don't accumulate: int8 += action wrapped past 127 within
        # one 32-step reset period, so the "touch state" work being
        # floor-probed included overflow wrapping. A set is the same
        # state traffic with defined values.
        data = state.data.at[:, 0].set(action.astype(jnp.int8))
        count = state.step_count + 1
        done = count >= 32                                       # periodic reset
        data = jnp.where(done[:, None], 0, data).astype(jnp.int8)
        count = jnp.where(done, 0, count).astype(jnp.int16)
        return NullState(data=data, mask=state.mask, step_count=count)
