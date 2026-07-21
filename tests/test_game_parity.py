"""Reference parity — the correctness gates (CPU-runnable; CI core).

Tests that compare against pgx/jumanji need the reference clones
(see HOW_TO_RUN.md). Without them they skip — unless
DJINNAX_REQUIRE_REFS is set (CI sets it), in which case a missing
reference is a failure, not a skip.
"""

import os

import pytest

import checks.check_parity as cp
from djinnax.game2048_lut import move_left_lut
from djinnax.refs import refs_available

requires_refs = pytest.mark.skipif(
    not refs_available() and not os.environ.get("DJINNAX_REQUIRE_REFS"),
    reason="reference engines (pgx/jumanji) not available — see HOW_TO_RUN.md",
)


@requires_refs
def test_ttt_parity_line_gather():
    cp.check_ttt(n_games=100, win_lut=False)


@requires_refs
def test_ttt_parity_bitboard():
    cp.check_ttt(n_games=100, win_lut=True)


@requires_refs
def test_2048_move_parity_branchless():
    cp.check_2048_moves(n_boards=300, move_fn=None)


@requires_refs
def test_2048_move_parity_lut():
    cp.check_2048_moves(n_boards=300, move_fn=move_left_lut)


def test_2048_variant_step_equivalence():
    cp.check_2048_variants_step_equivalence(n_steps=30)


def test_2048_spawn_distribution():
    cp.check_2048_spawn()


@requires_refs
def test_sokoban_parity():
    cp.check_sokoban(n_episodes=20)


@requires_refs
def test_ttt_offpath_parity():
    cp.check_ttt_offpath(n_games=60)


@requires_refs
def test_2048_reset_parity():
    cp.check_2048_reset(n_seeds=512)


def test_2048_exp15_divergence_pinned():
    cp.check_2048_exp15_divergence()


def test_2048_can_lut_probe():
    cp.check_2048_can_lut(n_boards=2048)
