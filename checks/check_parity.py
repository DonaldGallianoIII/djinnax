#!/usr/bin/env python3
"""Correctness gate for the head-to-head ports. Speed of a wrong engine is
meaningless — run this before bench_head_to_head.py.

TTT:  replay identical random-legal action sequences through pgx and
      DjinnTicTacToe from identical starts; boards, terminals, winners and
      rewards must match move-for-move (the game is deterministic).
2048: drive jumanji's own move()/can_move() (single board) against our
      batched move pass on random boards, all 4 directions — moved boards,
      rewards, and masks must match exactly. Spawn distribution smoke-checked.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import djinnax.refs  # noqa: F401  (sys.path + stubs)

import jax
import jax.numpy as jnp
import numpy as np

from djinnax.ttt import DjinnTicTacToe
from djinnax.game2048 import Djinn2048, _move_left, move_all_directions
from djinnax.game2048_lut import move_left_lut, make_game2048_lut


def check_ttt(n_games: int = 200, seed: int = 0, win_lut: bool = False) -> None:
    import pgx

    env = pgx.make("tic_tac_toe")
    ours = DjinnTicTacToe(win_lut=win_lut)
    rng = np.random.default_rng(seed)

    p_init = jax.jit(env.init)
    p_step = jax.jit(env.step)
    B = 1
    d_init = jax.jit(lambda k: ours.init(k, B))
    d_step = jax.jit(ours.step)

    for g in range(n_games):
        key = jax.random.PRNGKey(1000 + g)
        ps = p_init(key)
        ds = d_init(key)  # same key -> same bernoulli current_player
        assert int(ds.current_player[0]) == int(ps.current_player), f"game {g}: init player"

        for t in range(9):
            legal = np.flatnonzero(np.asarray(ps.legal_action_mask))
            a = int(rng.choice(legal))
            ps = p_step(ps, jnp.int32(a), key)
            ds = d_step(ds, jnp.full((B,), a, dtype=jnp.int32), key)

            assert np.array_equal(
                np.asarray(ps._x.board), np.asarray(ds.board[0])
            ), f"game {g} move {t}: board mismatch"
            assert bool(ps.terminated) == bool(ds.terminated[0]), f"game {g} move {t}: terminated"
            assert int(ps._x.winner) == int(ds.winner[0]), f"game {g} move {t}: winner"
            assert np.allclose(
                np.asarray(ps.rewards), np.asarray(ds.rewards[0])
            ), f"game {g} move {t}: rewards {ps.rewards} vs {ds.rewards[0]}"
            assert np.array_equal(
                np.asarray(ps.legal_action_mask), np.asarray(ds.legal_action_mask[0])
            ), f"game {g} move {t}: mask"
            assert np.array_equal(
                np.asarray(ps.observation), np.asarray(ds.observation[0])
            ), f"game {g} move {t}: observation"
            if bool(ps.terminated):
                break
    label = "bitboard-LUT" if win_lut else "line-gather"
    print(f"TTT parity OK ({label}) — {n_games} full games, boards/winners/rewards/masks/obs identical")


def check_2048_moves(n_boards: int = 500, seed: int = 1, move_fn=None) -> None:
    from jumanji.environments.logic.game_2048 import utils as ju

    rng = np.random.default_rng(seed)
    # Random boards incl. plenty of zeros and adjacent equal tiles
    boards = rng.choice(
        np.arange(0, 8), size=(n_boards, 4, 4), p=[0.35, 0.2, 0.2, 0.1, 0.06, 0.05, 0.03, 0.01]
    ).astype(np.int32)

    j_move = jax.jit(ju.move, static_argnums=1)
    j_can = jax.jit(ju.can_move, static_argnums=1)
    d_all = jax.jit(lambda b: move_all_directions(b, move_fn))

    ours_moved, ours_rewards, ours_can = d_all(jnp.asarray(boards, dtype=jnp.int8))
    for a in range(4):
        for i in range(n_boards):
            jb, jr = j_move(jnp.asarray(boards[i]), a)
            assert np.array_equal(np.asarray(jb), np.asarray(ours_moved[a, i])), (
                f"board {i} action {a}: move mismatch\njumanji:\n{np.asarray(jb)}\nours:\n{np.asarray(ours_moved[a, i])}"
            )
            assert float(jr) == float(ours_rewards[a, i]), (
                f"board {i} action {a}: reward {float(jr)} vs {float(ours_rewards[a, i])}"
            )
            jc = bool(j_can(jnp.asarray(boards[i]), a))
            assert jc == bool(ours_can[a, i]), f"board {i} action {a}: can_move {jc} vs ours"
    label = "LUT" if move_fn is not None else "branchless"
    print(f"2048 move parity OK ({label}) — {n_boards} boards x 4 directions, moves/rewards/masks identical")


def check_2048_variants_step_equivalence(seed: int = 3, n_steps: int = 50) -> None:
    """v2 (branchless) and v3 (LUT) full steps must be bit-identical: same
    state + action + key -> same state and reward, for many chained steps."""
    a_game = Djinn2048()
    b_game = make_game2048_lut()
    B = 128
    key = jax.random.PRNGKey(seed)
    a_state = a_game.init(key, B)
    b_state = b_game.init(key, B)
    a_step = jax.jit(a_game.step)
    b_step = jax.jit(b_game.step)
    for t in range(n_steps):
        k = jax.random.fold_in(key, t)
        action = jax.random.categorical(
            jax.random.fold_in(k, 9),
            jnp.where(a_state.action_mask, 0.0, -jnp.inf), axis=-1,
        ).astype(jnp.int32)
        a_state, a_r = a_step(a_state, action, k)
        b_state, b_r = b_step(b_state, action, k)
        for name in ("board", "action_mask", "score", "step_count", "terminated"):
            assert np.array_equal(
                np.asarray(getattr(a_state, name)), np.asarray(getattr(b_state, name))
            ), f"step {t}: v2/v3 diverge on {name}"
        assert np.array_equal(np.asarray(a_r), np.asarray(b_r)), f"step {t}: reward diverges"
    print(f"2048 v2/v3 step equivalence OK — {n_steps} chained steps at B={B} bit-identical")


def check_2048_spawn(seed: int = 2) -> None:
    """Distribution smoke: spawned tile is on a previously-empty cell, one per
    step, value 1 or 2 with ~0.9/0.1 split."""
    game = Djinn2048()
    B = 2048
    key = jax.random.PRNGKey(seed)
    state = game.init(key, B)
    board_before = np.asarray(state.board)
    action = jnp.zeros((B,), dtype=jnp.int32)
    legal = np.asarray(state.action_mask[:, 0])
    new_state, _ = jax.jit(game.step)(state, action, jax.random.fold_in(key, 1))
    # Only check envs where action 0 was legal and env didn't reset
    ok = legal & ~np.asarray(new_state.terminated)
    _, _, can = move_all_directions(jnp.asarray(board_before, dtype=jnp.int8))
    diffs = []
    vals = []
    for b in np.flatnonzero(ok):
        nb = np.asarray(new_state.board[b])
        # replay the move deterministically to isolate the spawn
        moved = np.asarray(
            jax.jit(lambda x: move_all_directions(x)[0][0])(
                jnp.asarray(board_before[b : b + 1], dtype=jnp.int8)
            )[0]
        )
        delta = (nb != moved)
        assert delta.sum() == 1, f"env {b}: expected exactly 1 spawned tile, got {delta.sum()}"
        y, x = np.argwhere(delta)[0]
        assert moved[y, x] == 0, f"env {b}: spawn on non-empty cell"
        vals.append(int(nb[y, x]))
        diffs.append((y, x))
    vals = np.asarray(vals)
    frac2 = float((vals == 2).mean())
    assert 0.05 < frac2 < 0.18, f"4-tile spawn fraction {frac2} outside ~0.1"
    assert set(vals.tolist()) <= {1, 2}
    print(f"2048 spawn OK — {len(vals)} spawns, single-tile on empty cell, P(4-tile)={frac2:.3f}")


def check_sokoban(n_episodes: int = 40, seed: int = 4) -> None:
    """Replay identical uniform-random action sequences through jumanji's raw
    Sokoban (no AutoReset) and DjinnSokoban on the SAME fixture level;
    grids, agent, reward, and done timing must match step-for-step."""
    from jumanji.environments.routing.sokoban.env import Sokoban
    from djinnax.soko_ref_generator import FixedLevelsGenerator
    from djinnax.sokoban import DjinnSokoban
    from djinnax.soko_levels import FIXED_LEVELS, VARIABLE_LEVELS, AGENT_YX

    env = Sokoban(generator=FixedLevelsGenerator())
    ours = DjinnSokoban()
    j_step = jax.jit(env.step)
    d_step = jax.jit(ours.step)
    rng = np.random.default_rng(seed)

    from jumanji.environments.routing.sokoban.types import State as JState
    from djinnax.sokoban import SokoState

    for ep in range(n_episodes):
        lvl = int(rng.integers(0, len(FIXED_LEVELS)))
        js = JState(
            key=jax.random.PRNGKey(0),
            fixed_grid=jnp.asarray(FIXED_LEVELS[lvl]),
            variable_grid=jnp.asarray(VARIABLE_LEVELS[lvl]),
            agent_location=jnp.asarray(AGENT_YX[lvl]),
            step_count=jnp.array(0, jnp.int32),
        )
        ds = SokoState(
            fixed_grid=jnp.asarray(FIXED_LEVELS[lvl])[None],
            variable_grid=jnp.asarray(VARIABLE_LEVELS[lvl])[None],
            agent_yx=jnp.asarray(AGENT_YX[lvl])[None],
            step_count=jnp.zeros((1,), jnp.int32),
            terminated=jnp.zeros((1,), jnp.bool_),
        )
        for t in range(130):
            a = int(rng.integers(0, 4))
            js, ts = j_step(js, jnp.array(a, jnp.int32))
            pre_reset_done = bool(ds.terminated[0])
            assert not pre_reset_done
            ds, dr, dobs, dex = d_step(
                ds, jnp.full((1,), a, jnp.int32), jax.random.PRNGKey(t)
            )
            j_done = bool(ts.last())
            d_done = bool(ds.terminated[0])
            assert j_done == d_done, f"ep {ep} t {t}: done {j_done} vs {d_done}"
            assert float(ts.reward) == float(dr[0]), (
                f"ep {ep} t {t}: reward {float(ts.reward)} vs {float(dr[0])}"
            )
            if not j_done:
                assert np.array_equal(
                    np.asarray(js.variable_grid), np.asarray(ds.variable_grid[0])
                ), f"ep {ep} t {t}: variable grid mismatch"
                assert np.array_equal(
                    np.asarray(js.agent_location), np.asarray(ds.agent_yx[0])
                ), f"ep {ep} t {t}: agent location mismatch"
                assert np.array_equal(
                    np.asarray(ts.observation.grid), np.asarray(dobs[0])
                ), f"ep {ep} t {t}: obs mismatch"
            else:
                break
    print(f"Sokoban parity OK — {n_episodes} episodes, grids/agent/rewards/done identical")



def check_ttt_offpath(n_games: int = 60, seed: int = 5) -> None:
    """External review R1 finding C3: gate the paths the ttt docstring
    claims pgx parity for but the main replay never reaches — the
    illegal-action loss and the step-past-terminated freeze."""
    import pgx

    env = pgx.make("tic_tac_toe")
    ours = DjinnTicTacToe()
    rng = np.random.default_rng(seed)
    p_init, p_step = jax.jit(env.init), jax.jit(env.step)
    B = 1
    d_init = jax.jit(lambda k: ours.init(k, B))
    d_step = jax.jit(ours.step)

    def assert_match(ps, ds, tag):
        assert np.array_equal(np.asarray(ps._x.board), np.asarray(ds.board[0])), f"{tag}: board"
        assert bool(ps.terminated) == bool(ds.terminated[0]), f"{tag}: terminated"
        assert np.allclose(np.asarray(ps.rewards), np.asarray(ds.rewards[0])), (
            f"{tag}: rewards {ps.rewards} vs {ds.rewards[0]}"
        )
        assert np.array_equal(
            np.asarray(ps.legal_action_mask), np.asarray(ds.legal_action_mask[0])
        ), f"{tag}: mask"

    n_illegal = n_pastterm = 0
    for g in range(n_games):
        key = jax.random.PRNGKey(3000 + g)
        ps, ds = p_init(key), d_init(key)
        for _ in range(int(rng.integers(1, 4))):          # short legal prefix
            a = int(rng.choice(np.flatnonzero(np.asarray(ps.legal_action_mask))))
            ps = p_step(ps, jnp.int32(a), key)
            ds = d_step(ds, jnp.full((B,), a, dtype=jnp.int32), key)
        if bool(ps.terminated):
            continue
        occupied = np.flatnonzero(~np.asarray(ps.legal_action_mask))
        if len(occupied) == 0:
            continue
        a = int(rng.choice(occupied))                     # deliberate illegal
        ps = p_step(ps, jnp.int32(a), key)
        ds = d_step(ds, jnp.full((B,), a, dtype=jnp.int32), key)
        assert_match(ps, ds, f"game {g}: illegal step")
        assert bool(ds.terminated[0]), f"game {g}: illegal move must terminate"
        n_illegal += 1
        a2 = int(rng.integers(0, 9))                      # step past terminal
        ps = p_step(ps, jnp.int32(a2), key)
        ds = d_step(ds, jnp.full((B,), a2, dtype=jnp.int32), key)
        assert_match(ps, ds, f"game {g}: step past terminal")
        n_pastterm += 1
    assert n_illegal >= 20 and n_pastterm >= 20, (n_illegal, n_pastterm)
    print(f"TTT off-path parity OK — {n_illegal} illegal-step cases, "
          f"{n_pastterm} step-past-terminal cases identical to pgx")


def check_2048_reset(n_seeds: int = 1024) -> None:
    """External review R1 finding C4: episode-START parity. The move and
    in-play spawn gates never asserted that both engines begin (and
    in-step reset) with the same board population; a mismatch would skew
    episode length and therefore the head-to-head ratio."""
    from jumanji.environments.logic.game_2048 import Game2048

    env = Game2048(board_size=4)
    ours = Djinn2048()

    j_reset = jax.jit(env.reset)
    d_init = jax.jit(lambda k: ours.init(k, 1))

    j_counts, j_vals, d_counts, d_vals = [], [], [], []
    for i in range(n_seeds):
        js, _ = j_reset(jax.random.PRNGKey(i))
        jb = np.asarray(js.board)
        j_counts.append(int((jb != 0).sum()))
        j_vals.extend(jb[jb != 0].tolist())
        db = np.asarray(d_init(jax.random.PRNGKey(i)).board[0])
        d_counts.append(int((db != 0).sum()))
        d_vals.extend(db[db != 0].tolist())

    assert set(j_counts) == set(d_counts), (
        f"reset tile count differs: reference {sorted(set(j_counts))} "
        f"vs djinn {sorted(set(d_counts))}"
    )
    j_vals, d_vals = np.asarray(j_vals), np.asarray(d_vals)
    assert set(np.unique(j_vals)) == set(np.unique(d_vals)), (j_vals, d_vals)
    for v in np.unique(j_vals):
        jp, dp = (j_vals == v).mean(), (d_vals == v).mean()
        assert abs(jp - dp) < 0.05, f"reset value {v}: P_ref={jp:.3f} P_djinn={dp:.3f}"
    print(f"2048 reset parity OK — {n_seeds} seeds: tile count {sorted(set(j_counts))}, "
          f"value distribution matched within 0.05")


def check_2048_exp15_divergence() -> None:
    """External review R1 finding C5: pin the ONE known divergence between
    the LUT and branchless engines. The LUT's 4-bit row code cannot
    represent exponent 16, so a 15+15 merge saturates at 15 while the
    branchless engine produces 16. Unreachable from 2/4 spawns in real
    play; asserted here so the divergence is documented behavior, not a
    latent surprise, and so the variant-equivalence claim is scoped."""
    board = jnp.zeros((1, 4, 4), dtype=jnp.int8).at[0, 0, 0].set(15).at[0, 0, 1].set(15)
    moved_b, reward_b = _move_left(board)
    moved_l, reward_l = move_left_lut(board)
    assert int(moved_b[0, 0, 0]) == 16, moved_b
    assert int(moved_l[0, 0, 0]) == 15, moved_l  # saturated: documented limit
    assert float(reward_b[0]) == 2.0 ** 16
    # below exponent 15 the engines are bit-identical (existing gates);
    # this is the only divergent input class.
    print("2048 exp-15 divergence pinned — LUT saturates at 15 (4-bit code), "
          "branchless produces 16; unreachable in play, documented in game2048_lut")


if __name__ == "__main__":
    check_ttt(win_lut=False)
    check_ttt(win_lut=True)
    check_2048_moves(move_fn=None)
    check_2048_moves(move_fn=move_left_lut)
    check_2048_variants_step_equivalence()
    check_2048_spawn()
    check_sokoban()
    check_ttt_offpath()
    check_2048_reset()
    check_2048_exp15_divergence()
    print("ALL PARITY CHECKS PASSED")
