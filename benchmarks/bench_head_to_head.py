#!/usr/bin/env python3
"""Head-to-head: the SAME game on the djinn engine style vs the reference.

Run check_parity.py first — these ports are move-for-move identical to the
references, so this measures ENGINE engineering only: batch-native
branchless code (ours) vs single-env code under vmap with while_loop/switch
(theirs), identical rules, identical protocol, same GPU.

Protocol per engine (identical): jitted lax.scan of 64 steps, actions
sampled in-graph from the legal mask, terminated envs reset every step
(TTT: where-select to init template on BOTH; 2048: jumanji's
AutoResetWrapper vs our in-step reset). 1 compile + best of 5 reps.

Usage (GPU shim + 50% cap):
    LD_PRELOAD=$VENV/.../nvidia/nvjitlink/lib/libnvJitLink.so.12 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    $VENV/bin/python benchmarks/bench_head_to_head.py

Protocol note (deliberate): every XLA runner here uses protocol v1 —
per-step fold_in keys and masked-categorical action sampling — applied
IDENTICALLY to both sides of every ratio, because symmetric overhead is
what makes a ratio fair. runtime.py's v2 (bulk key hoist, rank-pick,
donation) is faster in absolute terms; floor_bench.py measures that
delta. Consequence: the ratios are the headline numbers, while absolute
env-steps/s from THIS file are conservative for the djinn side.

EXCEPTION (audit S7): the 2048/djinn-mega row is an END-TO-END SYSTEM
measurement, not a matched-protocol one — the persistent kernel samples
actions in-kernel (rank-pick over its counter-hash RNG), so --rng and
the v1 sampler do not apply to it; its JSON rows are labeled
rng="counter-hash" accordingly. Its starting boards (_fresh_inputs) are
single-exponent-1-tile states, a one-step transient vs the engine's
0.9/0.1 two-tile reset distribution — boards converge to the in-kernel
reset distribution within the first episodes of the 64-step rollout.

Fail-closed (audit S3): strict mode is the DEFAULT — a non-GPU backend
or any engine failure exits nonzero so an official sweep can never
silently lose an engine or fall back to CPU. --best-effort restores the
exploratory keep-going behavior.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import djinnax.refs  # noqa: F401

import jax
import jax.numpy as jnp
from jax import lax

from djinnax.ttt import DjinnTicTacToe
from djinnax.game2048 import Djinn2048

# Set by main() from --unroll / --rng; runners read them at build time.
UNROLL = 1
RNG_IMPL = "threefry2x32"


def _root_key(seed: int) -> jax.Array:
    return jax.random.key(seed, impl=RNG_IMPL)


def _masked_categorical(key, mask):
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1)


def _where_reset(template, state, done):
    def sel(t, s):
        return jnp.where(done.reshape(done.shape[0], *([1] * (s.ndim - 1))), t, s)

    return jax.tree_util.tree_map(sel, template, state)


# --- runners: (carry0, jitted runner(carry, key) -> carry) ------------------


def make_ttt_pgx(B, n_steps):
    import pgx

    env = pgx.make("tic_tac_toe")
    init = jax.vmap(env.init)
    step = jax.vmap(env.step)
    template = init(jax.random.split(_root_key(0), B))

    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        action = _masked_categorical(k, state.legal_action_mask)
        state = step(state, action, jax.random.split(jax.random.fold_in(k, 1), B))
        state = _where_reset(template, state, state.terminated)
        return (state, key), None

    @jax.jit
    def runner(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(n_steps), unroll=UNROLL)
        return state

    return template, runner


def make_ttt_djinn(B, n_steps):
    game = DjinnTicTacToe()
    template = game.init(_root_key(0), B)

    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        action = _masked_categorical(k, state.legal_action_mask)
        state = game.step(state, action, jax.random.fold_in(k, 1))
        state = _where_reset(template, state, state.terminated)
        return (state, key), None

    @jax.jit
    def runner(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(n_steps), unroll=UNROLL)
        return state

    return template, runner


def make_2048_jumanji(B, n_steps):
    from jumanji.wrappers import AutoResetWrapper
    from jumanji.environments.logic.game_2048.env import Game2048

    env = AutoResetWrapper(Game2048())
    reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)
    state0, ts0 = reset(jax.random.split(_root_key(0), B))

    def one_step(carry, i):
        state, ts, key = carry
        k = jax.random.fold_in(key, i)
        action = _masked_categorical(k, ts.observation.action_mask)
        state, ts = step(state, action.astype(jnp.int32))
        return (state, ts, key), None

    @jax.jit
    def runner(carry, key):
        state, ts = carry
        (state, ts, _), _ = lax.scan(one_step, (state, ts, key), jnp.arange(n_steps), unroll=UNROLL)
        return state, ts

    return (state0, ts0), runner


def make_2048_djinn(B, n_steps, game=None):
    game = game or Djinn2048()
    state0 = game.init(_root_key(0), B)

    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        action = _masked_categorical(k, state.action_mask)
        state, _ = game.step(state, action.astype(jnp.int32), jax.random.fold_in(k, 1))
        return (state, key), None

    @jax.jit
    def runner(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(n_steps), unroll=UNROLL)
        return state

    return state0, runner


def make_2048_djinn_lut(B, n_steps):
    from djinnax.game2048_lut import make_game2048_lut

    return make_2048_djinn(B, n_steps, game=make_game2048_lut())


def make_ttt_djinn_bb(B, n_steps):
    game = DjinnTicTacToe(win_lut=True)
    template = game.init(_root_key(0), B)

    def one_step(carry, i):
        state, key = carry
        k = jax.random.fold_in(key, i)
        action = _masked_categorical(k, state.legal_action_mask)
        state = game.step(state, action, jax.random.fold_in(k, 1))
        state = _where_reset(template, state, state.terminated)
        return (state, key), None

    @jax.jit
    def runner(state, key):
        (state, _), _ = lax.scan(one_step, (state, key), jnp.arange(n_steps), unroll=UNROLL)
        return state

    return template, runner


def make_soko_jumanji(B, n_steps):
    from jumanji.wrappers import AutoResetWrapper
    from jumanji.environments.routing.sokoban.env import Sokoban
    from djinnax.soko_ref_generator import FixedLevelsGenerator

    env = AutoResetWrapper(Sokoban(generator=FixedLevelsGenerator()))
    reset = jax.vmap(env.reset)
    step = jax.vmap(env.step)
    state0, ts0 = reset(jax.random.split(_root_key(0), B))

    def one_step(carry, i):
        state, ts, key = carry
        k = jax.random.fold_in(key, i)
        action = jax.random.randint(k, (B,), 0, 4)   # no action mask in sokoban
        state, ts = step(state, action)
        return (state, ts, key), None

    @jax.jit
    def runner(carry, key):
        state, ts = carry
        (state, ts, _), _ = lax.scan(one_step, (state, ts, key), jnp.arange(n_steps), unroll=UNROLL)
        return state, ts

    return (state0, ts0), runner


def make_soko_djinn(B, n_steps):
    from djinnax.sokoban import DjinnSokoban

    game = DjinnSokoban()
    state0 = game.init(_root_key(0), B)
    # Live-output symmetry (review E3 / audit SOL-05): jumanji's runner
    # carries its TimeStep through the scan, keeping observation/reward/
    # extras live in the compiled graph. Discarding ours let XLA DCE the
    # whole output surface (HLO before this fix: 0 ops with the
    # (B,10,10,2) obs shape on our side vs 20 on jumanji's). Carry the
    # per-step outputs so both engines do the work their API promises.
    obs_shape = (B,) + state0.fixed_grid.shape[1:] + (2,)
    out0 = (
        jnp.zeros((B,), jnp.float32),                       # reward
        jnp.zeros(obs_shape, jnp.uint8),                    # obs
        {"prop_correct_boxes": jnp.zeros((B,), jnp.float32),
         "solved": jnp.zeros((B,), jnp.bool_)},             # extras
    )

    def one_step(carry, i):
        state, _, key = carry
        k = jax.random.fold_in(key, i)
        action = jax.random.randint(k, (B,), 0, 4)
        state, reward, obs, extras = game.step(state, action, jax.random.fold_in(k, 1))
        return (state, (reward, obs, extras), key), None

    @jax.jit
    def runner(carry, key):
        state, out = carry
        (state, out, _), _ = lax.scan(one_step, (state, out, key), jnp.arange(n_steps), unroll=UNROLL)
        return state, out

    return (state0, out0), runner


def make_2048_megakernel(B, n_steps):
    """Self-contained megakernel as a standing-harness engine. Carry =
    (board, score); each call reseeds from the provided key (t_offset 0 —
    fresh stream per call, fine for throughput). Block falls back to 64
    when B is not a multiple of the default BLOCK."""
    from djinnax.megakernel import _fresh_inputs, BLOCK
    from djinnax.megakernel_rng import run_megakernel_rng

    block = BLOCK if B % BLOCK == 0 else 64
    board0, _ = _fresh_inputs(B, 0)
    score0 = jnp.zeros((B,), jnp.float32)

    @jax.jit
    def runner(carry, key):
        board, score = carry
        bits = jax.random.key_data(key).ravel()
        seed = jnp.stack([bits[0], jnp.uint32(0)])
        b, s, _ = run_megakernel_rng(board, seed, score=score,
                                     n_steps=n_steps, block=block)
        return (b, s)

    return (board0, score0), runner


def bench(name, maker, B, n_steps, reps):
    carry, runner = maker(B, n_steps)
    key = _root_key(42)
    t0 = time.perf_counter()
    carry = jax.block_until_ready(runner(carry, key))
    compile_s = time.perf_counter() - t0
    best = float("inf")
    for r in range(reps):
        k = jax.random.fold_in(key, r + 1)
        t0 = time.perf_counter()
        carry = jax.block_until_ready(runner(carry, k))
        best = min(best, time.perf_counter() - t0)
    sps = B * n_steps / best
    print(f"{name:22s} B={B:<6d} {sps:>14,.0f} env-steps/s"
          f"  {1e6 * best / (B * n_steps):>8.3f} µs/env-step  (compile {compile_s:.1f}s)")
    return sps


def main():
    global UNROLL, RNG_IMPL
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 1024, 8192])
    ap.add_argument("--unroll", type=int, default=1, help="lax.scan unroll factor (all engines)")
    ap.add_argument("--rng", default="threefry2x32",
                    choices=["threefry2x32", "rbg", "unsafe_rbg"],
                    help="PRNG impl for all keys (all engines)")
    ap.add_argument("--json", default=None,
                    help="append one JSON line per (engine, B) to this file")
    ap.add_argument("--best-effort", action="store_true",
                    help="exploratory mode: tolerate CPU backend and per-engine "
                         "failures (default is strict/fail-closed)")
    args = ap.parse_args()
    UNROLL = args.unroll
    RNG_IMPL = args.rng

    backend = jax.default_backend()
    print(f"backend: {backend}  device: {jax.devices()[0]}"
          f"  unroll={UNROLL} rng={RNG_IMPL}"
          f"  mode={'best-effort' if args.best_effort else 'strict'}")
    if not args.best_effort and backend != "gpu":
        _sys.exit(f"strict mode: backend is '{backend}', not 'gpu' — a CPU "
                  f"fallback would silently produce non-comparable numbers "
                  f"(pass --best-effort to run anyway)")

    # Provenance stamped into every JSON row (audit S3): a row must be
    # attributable to a machine state and tree without archaeology.
    try:
        import subprocess
        _commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_Path(__file__).resolve().parents[1]),
            capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        _commit = "unknown"
    prov = {"backend": backend, "device": str(jax.devices()[0]),
            "jax": jax.__version__, "commit": _commit,
            "steps": args.steps, "reps": args.reps}
    pairs = [
        ("ttt/pgx", make_ttt_pgx), ("ttt/djinn", make_ttt_djinn),
        ("ttt/djinn-bb", make_ttt_djinn_bb),
        ("2048/jumanji", make_2048_jumanji), ("2048/djinn", make_2048_djinn),
        ("2048/djinn-lut", make_2048_djinn_lut),
        ("2048/djinn-mega", make_2048_megakernel),
        ("soko/jumanji", make_soko_jumanji), ("soko/djinn", make_soko_djinn),
    ]
    records = []
    failures = []
    for B in args.batches:
        print(f"\n--- B={B}, {args.steps} steps/scan, best of {args.reps} ---")
        speeds = {}
        for name, maker in pairs:
            # The mega row samples in-kernel (counter-hash RNG); labeling
            # it with --rng would claim a protocol it does not run (S7).
            rng_label = "counter-hash" if name == "2048/djinn-mega" else RNG_IMPL
            try:
                speeds[name] = bench(name, maker, B, args.steps, args.reps)
                records.append({"engine": name, "B": B, "steps_per_sec": speeds[name],
                                "unroll": UNROLL, "rng": rng_label, **prov})
            except Exception as e:
                print(f"{name:22s} B={B:<6d} FAILED: {type(e).__name__}: {e}")
                failures.append(f"{name} @ B={B}: {type(e).__name__}: {e}")
        for game_name, ours, theirs in [
            ("ttt", "ttt/djinn", "ttt/pgx"),
            ("ttt-bb", "ttt/djinn-bb", "ttt/pgx"),
            ("2048", "2048/djinn", "2048/jumanji"),
            ("2048-lut", "2048/djinn-lut", "2048/jumanji"),
            ("2048-mega", "2048/djinn-mega", "2048/jumanji"),
            ("soko", "soko/djinn", "soko/jumanji"),
        ]:
            if ours in speeds and theirs in speeds:
                r = speeds[ours] / speeds[theirs]
                print(f"  -> {game_name}: djinn is {r:.2f}x the reference")

    if args.json:
        with open(args.json, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    if failures and not args.best_effort:
        _sys.exit("strict mode: engine failures (rows above are INCOMPLETE):\n  "
                  + "\n  ".join(failures))


if __name__ == "__main__":
    main()
