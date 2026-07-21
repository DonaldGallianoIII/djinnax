"""Distribution-parity gates: each collapsed sampler vs the naive loop
it replaces (CPU-runnable; CI core)."""

import jax
import jax.numpy as jnp
import numpy as np

from djinnax.distributions import (
    alias_sample, build_alias_table, conditional_categorical,
    geometric_tries,
)

N = 200_000


def test_geometric_matches_naive_loop():
    p = 0.3
    u = jax.random.uniform(jax.random.PRNGKey(0), (N,))
    ours = np.asarray(geometric_tries(u, p))
    rng = np.random.default_rng(1)
    naive = np.array([next(k for k in range(1, 1000)
                           if rng.random() < p) for _ in range(20_000)])
    assert abs(ours.mean() - 1 / p) < 0.05          # E[N]=3.33 — "three tries"
    assert abs(ours.mean() - naive.mean()) < 0.1
    assert abs(ours.var() - (1 - p) / p**2) < 0.3   # exact variance kept
    for k in (1, 2, 3, 5):                           # pointwise pmf check
        expect = (1 - p) ** (k - 1) * p
        assert abs((ours == k).mean() - expect) < 0.01, f"P(N={k})"
    assert ours.min() >= 1


def test_conditional_matches_rejection_loop():
    probs = jnp.asarray([0.5, 0.2, 0.2, 0.1])
    allowed = jnp.asarray([True, False, True, True])
    u = jax.random.uniform(jax.random.PRNGKey(2), (N,))
    ours = np.asarray(conditional_categorical(
        u, jnp.broadcast_to(probs, (N, 4)), jnp.broadcast_to(allowed, (N, 4))))
    assert not np.any(ours == 1), "sampled a disallowed category"
    renorm = np.asarray([0.5, 0.0, 0.2, 0.1]) / 0.8
    for k in (0, 2, 3):
        assert abs((ours == k).mean() - renorm[k]) < 0.01, f"P(k={k})"


def test_alias_matches_weights():
    weights = [5.0, 1.0, 3.0, 1.0, 10.0]
    prob, alias = build_alias_table(weights)
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    s = np.asarray(alias_sample(
        jax.random.uniform(k1, (N,)), jax.random.uniform(k2, (N,)),
        jnp.asarray(prob), jnp.asarray(alias)))
    target = np.asarray(weights) / sum(weights)
    for k in range(5):
        assert abs((s == k).mean() - target[k]) < 0.01, f"P(k={k})"
