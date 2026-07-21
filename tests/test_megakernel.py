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


def test_move_chain_link():
    cm.check_move_chain_link(n_boards=512)


def test_analytic_reset_mask():
    cm.check_analytic_reset_mask()


@requires_gpu
def test_b_divisibility_guard():
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
