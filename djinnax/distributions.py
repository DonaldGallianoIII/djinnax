"""Closed-form collapse of stochastic processes — sample the OUTCOME,
not the process.

Unbounded stochastic loops (retry-until-success, reroll-until-valid,
draw-until-drop) are the stochastic twin of the while_loop anti-pattern:
data-dependent iteration that runs every batch element to the worst
case. Every sampler here replaces such a loop with an O(1) draw from the
process's exact closed-form distribution — same distribution, zero
iteration, counter-hash friendly (feed uniforms from
`djinnax.megakernel_rng.hash_uniform`).

Legality rule (see PORTING_PLAYBOOK): a loop may be collapsed ONLY if
nothing observable or decidable happens between iterations. If an
opponent acts, state renders, or the agent chooses between tries, the
tries are real game steps — collapsing them changes the MDP.

Parity note: collapsed sites consume randomness differently than the
reference implementation, so they are gated by DISTRIBUTION parity
(statistical tests vs the naive loop), not bit parity — the same
two-tier pattern used for the megakernel's counter RNG.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def geometric_tries(u: jax.Array, p, max_tries=None) -> jax.Array:
    """Number of attempts until first success for per-try probability p.

    Exact inverse-CDF sample of the geometric distribution (support
    1, 2, ...): distributionally identical to looping `while
    uniform() >= p`. One uniform in, int32 out; E[N] = 1/p.

    u: uniforms in [0, 1); p: scalar or broadcastable array in (0, 1].
    max_tries: optional clamp for downstream fixed-size consumers (note:
    clamping truncates the tail — document it where used).
    """
    p = jnp.asarray(p, dtype=jnp.float32)
    # N = 1 + floor(log(1-u)/log(1-p)); log1p keeps small-p precision.
    n = 1 + jnp.floor(jnp.log1p(-u) / jnp.log1p(-p)).astype(jnp.int32)
    n = jnp.maximum(n, 1)                     # u==0 edge
    if max_tries is not None:
        n = jnp.minimum(n, max_tries)
    return n


def conditional_categorical(u: jax.Array, probs: jax.Array,
                            allowed: jax.Array) -> jax.Array:
    """Collapse reroll-until-valid: sampling category k from `probs`
    repeatedly until `allowed[k]` IS sampling from the renormalized
    conditional distribution — one draw, exact.

    probs: (..., K) nonnegative weights; allowed: (..., K) bool;
    u: (...,) uniforms. Rows with nothing allowed return argmax(allowed)
    = 0 deterministically (guard upstream).
    """
    w = jnp.where(allowed, probs, 0.0)
    total = w.sum(axis=-1, keepdims=True)
    c = jnp.cumsum(w, axis=-1)
    target = u[..., None] * total
    return jnp.argmax(c > target, axis=-1).astype(jnp.int32)


def build_alias_table(weights) -> tuple[np.ndarray, np.ndarray]:
    """Precompute Walker alias tables for O(1) categorical sampling —
    the LUT rung applied to distributions. Build in numpy at import;
    sample at runtime with `alias_sample` (two uniforms, no cumsum,
    no rejection loop).
    """
    w = np.asarray(weights, dtype=np.float64)
    k = len(w)
    prob = np.zeros(k, dtype=np.float32)
    alias = np.zeros(k, dtype=np.int32)
    scaled = w * k / w.sum()
    small = [i for i, s in enumerate(scaled) if s < 1.0]
    large = [i for i, s in enumerate(scaled) if s >= 1.0]
    while small and large:
        s, l = small.pop(), large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] - (1.0 - scaled[s])
        (small if scaled[l] < 1.0 else large).append(l)
    for i in small + large:
        prob[i] = 1.0
    return prob, alias


def alias_sample(u1: jax.Array, u2: jax.Array, prob: jax.Array,
                 alias: jax.Array) -> jax.Array:
    """O(1) draw from tables built by `build_alias_table`."""
    k = prob.shape[0]
    i = jnp.minimum((u1 * k).astype(jnp.int32), k - 1)
    return jnp.where(u2 < prob[i], i, alias[i]).astype(jnp.int32)


def expected_tries(p) -> jax.Array:
    """Expectation substitution — the VARIANCE-REMOVING sibling of
    geometric_tries (always returns ~1/p). Changes the game's dynamics;
    only for debugging/curriculum modes, never silently."""
    return jnp.ceil(1.0 / jnp.asarray(p, dtype=jnp.float32)).astype(jnp.int32)
