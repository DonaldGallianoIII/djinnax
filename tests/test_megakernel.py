"""Megakernel hardening battery as pytest. Kernel-touching tests need the
GPU (Triton lowering); the chain-link and analytic-mask proofs are pure
XLA and run anywhere — they are the CI-on-CPU core for the megakernel."""

import jax
import pytest

import checks.check_megakernel as cm

requires_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="Triton megakernel requires GPU (sm_89+ tested)",
)


def test_oriented_move_equals_allmoves():
    """P1 (external review R1): the orient-select step is bit-identical
    to the all-four-moves step under XLA scan — the kernel rewrite is
    the same game by construction, not by luck."""
    import jax.numpy as jnp
    from djinnax.megakernel import (
        _fresh_inputs, _xla_reference, step_lanes, step_lanes_allmoves,
    )

    board, uniforms = _fresh_inputs(384, 3)
    new_out = jax.jit(lambda b, u: _xla_reference(b, u, step_lanes))(board, uniforms)
    old_out = jax.jit(lambda b, u: _xla_reference(b, u, step_lanes_allmoves))(board, uniforms)
    for a, b, name in zip(new_out, old_out, ("board", "score", "done")):
        assert jnp.array_equal(a, b), name


def test_move_chain_link():
    cm.check_move_chain_link(n_boards=512)


def test_analytic_reset_mask():
    cm.check_analytic_reset_mask()


def test_analytic_mask_chain_link():
    cm.check_analytic_mask_chain_link()


def test_analytic_mask_step_equivalence():
    cm.check_analytic_mask_step()


def test_b_divisibility_guard():
    # Pure host-side argument validation (raises before any launch) —
    # CPU-runnable, so CI covers both entry points' guards.
    cm.check_b_divisibility_guard()


@requires_gpu
def test_bit_determinism():
    cm.check_bit_determinism()


@requires_gpu
def test_parity_sweep():
    cm.check_parity_sweep()


@requires_gpu
def test_adversarial_boards():
    cm.check_adversarial_boards()


@requires_gpu
def test_chained_rollout():
    cm.check_chained_rollout()
