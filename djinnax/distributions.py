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
    Inputs are clamped to their contract: u to [0, 1-2^-24] (an inclusive
    [0,1] source would send log1p(-1) to -inf, and the int cast of ±inf
    is undefined — it typically lands on INT32_MIN, which the n>=1 guard
    would then silently launder into n=1, the OPPOSITE tail), and the
    try count to int32 range before the cast (tiny p, e.g. 1e-9 with
    unlucky u, can exceed 2^31 tries).
    max_tries: optional clamp for downstream fixed-size consumers (note:
    clamping truncates the tail — document it where used).
    """
    p = jnp.asarray(p, dtype=jnp.float32)
    u = jnp.clip(u, 0.0, 1.0 - 2.0 ** -24)
    # N = 1 + floor(log(1-u)/log(1-p)); log1p keeps small-p precision.
    tries = jnp.floor(jnp.log1p(-u) / jnp.log1p(-p))
    # Clamp below int32 max BEFORE the cast. The bound must itself be
    # exactly representable in float32 (ulp at 2^31 is 256; 2^31-2 would
    # round UP to 2^31 and overflow the cast it exists to prevent).
    n = 1 + jnp.minimum(tries, 2.0 ** 31 - 256).astype(jnp.int32)
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
    u: (...,) uniforms. Degenerate rows — nothing allowed, or every
    allowed category has zero weight — return index 0 deterministically
    (guard upstream).
    """
    w = jnp.where(allowed, probs, 0.0)
    c = jnp.cumsum(w, axis=-1)
    # Normalize by the cumsum's OWN last element, not a separately
    # computed sum: the two reductions aren't bit-identical, and a
    # target above c[..., -1] would select index 0 even if disallowed.
    total = c[..., -1:]
    target = u[..., None] * total
    hit = c > target
    idx = jnp.argmax(hit, axis=-1)
    # Rounding at u -> 1 can still land target == total exactly; fall
    # back to the last positive-weight category, never a bogus index 0.
    k = w.shape[-1]
    last_pos = (k - 1) - jnp.argmax(jnp.flip(w > 0.0, axis=-1), axis=-1)
    idx = jnp.where(jnp.any(hit, axis=-1), idx, last_pos)
    # Degenerate rows (no allowed mass) return 0, as documented.
    idx = jnp.where(total[..., 0] > 0.0, idx, 0)
    return idx.astype(jnp.int32)


def build_alias_table(weights) -> tuple[np.ndarray, np.ndarray]:
    """Precompute Walker alias tables for O(1) categorical sampling —
    the LUT rung applied to distributions. Build in numpy at import;
    sample at runtime with `alias_sample` (two uniforms, no cumsum,
    no rejection loop).
    """
    w = np.asarray(weights, dtype=np.float64)
    # Host-side, build-once — strict validation is free (audit S6: an
    # all-zero or NaN weight vector previously produced NaN tables of
    # plausible shape that only failed at sample time, statistically).
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"weights must be a non-empty 1-D array, got shape {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("weights must have positive sum")
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
    assert np.all((prob >= 0) & (prob <= 1)) and np.all((alias >= 0) & (alias < k))
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
    only for debugging/curriculum modes, never silently.

    Same pre-cast clamp as geometric_tries (audit S5 — the sibling was
    fixed in R1-C1, this one was missed): tiny p sends ceil(1/p) past
    int32 range or to +inf, and the cast of either is undefined. The
    bound 2^31-256 is the largest float32 below 2^31 (ulp there is 256).
    Saturation, not exactness, is the contract for p < ~2^-31.
    """
    p = jnp.asarray(p, dtype=jnp.float32)
    n = jnp.ceil(1.0 / p)
    return jnp.minimum(n, 2.0 ** 31 - 256).astype(jnp.int32)
