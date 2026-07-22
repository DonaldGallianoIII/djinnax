# djinnax — batch-native JAX game engines, with receipts

Game environments written in a strict batch-native, branchless JAX
discipline, raced head-to-head against the well-known JAX env suites
(pgx, jumanji) on the **same games**, parity-gated move-for-move before
any timing. Built for [Crawdata.ai](https://crawdata.ai).

**The result is a spectrum, not a number** — the win scales with how much
control flow the reference keeps in its hot path, ending at a persistent
"environment-on-chip" Triton megakernel:

| B | tic-tac-toe | sokoban | 2048 branchless | 2048 LUT | 2048 megakernel |
|---:|---|---|---|---|---|
| 64 | 1.1× | 1.5× | 34.6× | 56.3× | **98×** |
| 1,024 | 1.1× | 2.2× | 22.6× | 48.1× | **213×** |
| 8,192 | 1.2× | 3.0× | 14.8× | 49.4× | **237×** |
| 65,536 | 1.7× | 2.4× | 6.2× | 14.9× | **91×** |

Within-run ratios vs the reference implementation, medians of an n=5
frozen-code sweep on a quiet RTX 4070 (full intervals + history in
`docs/RESULTS_HISTORY.md`, raw rows in `data/sweep_official_v2.jsonl`).
Peak absolute: **1.9B env-steps/s** for full 2048 games, RNG included,
in one kernel launch.

*Provenance (honesty policy):* measured 2026-07-21. Two known biases
relative to the current tree, both to be replaced by the next
quiet-window sweep: the megakernel column predates the orient-select
kernel (review P1, ~1.5× kernel-side — the current kernel is FASTER
than this table), and the sokoban column predates the live-outputs
runner fix (review E3 — our side was ~1.6× inflated by dead-code
elimination, so the current honest ratio is LOWER; receipt in
`data/e3_soko_dce_ab.jsonl`).

## Thirty seconds of usage

```python
import jax, djinnax

env = djinnax.make_game2048_lut()            # or djinnax.Djinn2048()
key = jax.random.PRNGKey(0)
state = env.init(key, n_envs=8192)           # leading-B everywhere
actions = jax.numpy.zeros((8192,), dtype=jax.numpy.int32)
state, reward = env.step(state, actions, jax.random.fold_in(key, 1))
```

Step signatures are per-env (each returns exactly what its game
defines — docstrings state them); every env is batch-native, so the
line above is the whole API. Episode boundaries follow each env's
parity reference: **2048 and sokoban auto-reset in-step** (jumanji
`AutoResetWrapper` convention — `terminated` flags the transition that
ended, the returned state/observation are the already-reset episode),
while **tic-tac-toe freezes terminal states** (pgx convention — reset
externally). The returned observation always describes the returned
state. The megakernel entry point is `djinnax.run_megakernel_rng`
(GPU required; see HOW_TO_RUN.md).

## Read this before anything else

| you want to… | read |
|---|---|
| run anything (setup, GPU shim, scripts) | **HOW_TO_RUN.md** |
| write or port a game the fast way | **PORTING_PLAYBOOK.md**, then **WRITING_FAST_ENVS.md** |
| point an AI coding agent at this style | **CLAUDE.md** (auto-read by Claude Code) / AGENTS.md |
| understand why the rules exist (evidence) | **LEARNINGS.md** |
| the megakernel design + results | **MEGAKERNEL_PLAN.md** |

## What's in here

- `djinnax/` — the engines (ttt, game2048 + LUT, sokoban), each
  parity-gated against pgx/jumanji; `djinnax/runtime.py` shared loop machinery.
- `djinnax/megakernel*.py` — the whole rollout in
  ONE Triton kernel launch (in-register state, in-kernel counter RNG,
  bit-verified against an XLA reference running the same step function).
- `checks/` + `tests/` — the correctness
  gates. **No number in this repo was measured before its engine passed
  parity.** `pytest tests/` runs the full suite (exact counts live in
  HOW_TO_RUN.md only, so they can't drift here); GPU-kernel tests skip
  on CPU, reference-parity tests skip if pgx/jumanji aren't set up.
- `benchmarks/` — the
  measurement harness (interleaved pairwise, frozen-code sweeps,
  medians-with-spread; see LEARNINGS §3 for the methodology rules).
- `benchmarks/pallas_lab.py` — the guided walk up to the megakernel.

## Honesty policy

Medians with [min..max] from multiple fresh-process runs; interleaved
pairwise for anything sub-2×; null results kept and labeled; known
caveats (RNG-stream differences, hardware specificity, single-GPU-model
evidence) stated next to the numbers they qualify. The head-to-head
harness deliberately runs the same driving protocol on both sides of
every ratio (see its docstring), so its absolute env-steps/s are
conservative for the djinn engines.
