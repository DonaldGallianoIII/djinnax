"""Reference parity — the correctness gates (CPU-runnable; CI core)."""

import checks.check_parity as cp
from djinnax.game2048_lut import move_left_lut


def test_ttt_parity_line_gather():
    cp.check_ttt(n_games=100, win_lut=False)


def test_ttt_parity_bitboard():
    cp.check_ttt(n_games=100, win_lut=True)


def test_2048_move_parity_branchless():
    cp.check_2048_moves(n_boards=300, move_fn=None)


def test_2048_move_parity_lut():
    cp.check_2048_moves(n_boards=300, move_fn=move_left_lut)


def test_2048_variant_step_equivalence():
    cp.check_2048_variants_step_equivalence(n_steps=30)


def test_2048_spawn_distribution():
    cp.check_2048_spawn()


def test_sokoban_parity():
    cp.check_sokoban(n_episodes=20)
