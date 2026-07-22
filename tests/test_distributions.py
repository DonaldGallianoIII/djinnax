"""Distribution-parity gates: each collapsed sampler vs the naive loop
it replaces (CPU-runnable; CI core)."""

import jax
import jax.numpy as jnp
import numpy as np

import pytest

from djinnax.distributions import (
    alias_sample, build_alias_table, conditional_categorical,
    expected_tries, geometric_tries,
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


def test_geometric_edge_inputs():
    """External review R1 finding C1: u==1.0 (inclusive-source contract
    break) and tiny p used to hit an undefined float->int32 cast whose
    garbage the n>=1 guard laundered into n=1 — the opposite tail."""
    u = jnp.asarray([0.0, 0.5, 1.0 - 2.0 ** -24, 1.0])
    for p in (1e-9, 0.3, 1.0):
        n = np.asarray(geometric_tries(u, p))
        assert np.all(n >= 1), (p, n)
        assert np.all(n < 2 ** 31 - 1) and np.all(n > -(2 ** 31)), (p, n)
        # monotone in u: a later quantile is never fewer tries
        assert np.all(np.diff(n) >= 0), (p, n)
    # the killer case: u=1.0 with small p must be the DEEP tail, not 1
    n_top = int(geometric_tries(jnp.asarray(1.0), 1e-3))
    assert n_top > 10_000, n_top
    assert int(geometric_tries(jnp.asarray(1.0), 1.0)) == 1


def test_conditional_never_selects_disallowed_at_high_u():
    """External review R1 finding C2: normalizing with a separately
    computed sum (not the cumsum's last element) let u->1 produce a
    target above c[-1], selecting index 0 even when disallowed."""
    probs = jnp.asarray([0.5, 0.2, 0.2, 0.1])
    allowed = jnp.asarray([False, True, True, True])  # index 0 forbidden
    us = jnp.asarray([1.0 - k * 2.0 ** -24 for k in range(64)] + [0.0])
    idx = np.asarray(conditional_categorical(
        us, jnp.broadcast_to(probs, (65, 4)), jnp.broadcast_to(allowed, (65, 4))))
    assert not np.any(idx == 0), idx
    # degenerate rows: nothing allowed, or allowed mass is all zero -> 0
    z = conditional_categorical(
        jnp.asarray([0.7, 0.7]),
        jnp.asarray([[0.5, 0.5, 0.0, 0.0]] * 2),
        jnp.asarray([[False] * 4, [False, False, True, True]]))
    assert np.all(np.asarray(z) == 0), z


def test_expected_tries_edge_inputs():
    """Audit S5: same unclamped int32(ceil(1/p)) overflow R1-C1 fixed in
    geometric_tries — the sibling was missed. Tiny p must saturate, not
    cast garbage."""
    assert int(expected_tries(1.0)) == 1
    assert int(expected_tries(0.3)) == 4          # ceil(3.33)
    assert int(expected_tries(1e-3)) == 1000
    for p in (1e-9, 1e-12, 5e-40):                # 1/p far past int32
        n = int(expected_tries(p))
        assert 0 < n < 2 ** 31, (p, n)
    n = np.asarray(expected_tries(jnp.asarray([1.0, 1e-3, 1e-12])))
    assert np.all(n >= 1) and np.all(n < 2 ** 31)


def test_alias_rejects_invalid_weights():
    """Audit S6: all-zero weights used to divide by zero and hand back
    NaN tables of plausible shape; negatives/NaN built nonsense."""
    for bad in ([], [0.0, 0.0], [1.0, -2.0], [1.0, float("nan")],
                [float("inf"), 1.0], [[1.0, 2.0]]):
        with pytest.raises(ValueError):
            build_alias_table(bad)
    # single-category degenerate is VALID: always index 0
    prob, alias = build_alias_table([7.0])
    s = np.asarray(alias_sample(jnp.asarray([0.1, 0.9]), jnp.asarray([0.2, 0.8]),
                                jnp.asarray(prob), jnp.asarray(alias)))
    assert np.all(s == 0)
    # zero-weight CATEGORY inside a valid vector never gets sampled
    prob, alias = build_alias_table([2.0, 0.0, 1.0])
    k1, k2 = jax.random.split(jax.random.PRNGKey(4))
    s = np.asarray(alias_sample(jax.random.uniform(k1, (N,)),
                                jax.random.uniform(k2, (N,)),
                                jnp.asarray(prob), jnp.asarray(alias)))
    assert not np.any(s == 1)
    assert abs((s == 0).mean() - 2 / 3) < 0.01


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
