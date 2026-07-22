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
| 64 | 1.1× | 1.4× | 33.9× | 33.2× | **239×** |
| 1,024 | 1.2× | 1.5× | 25.2× | 31.7× | **213×** |
| 8,192 | 1.1× | 2.8× | 22.7× | 21.9× | **240×** |
| 65,536 | 1.2× | 1.2× | 8.0× | 13.8× | **57×**† |

Within-run ratios vs the reference implementation, medians of an n=5
frozen-code fresh-process sweep on an RTX 4070 (full intervals +
history in `docs/RESULTS_HISTORY.md`, raw rows in
`data/sweep_official_v3.jsonl`). Peak absolute observed: **2.5B
env-steps/s** for full 2048 games, RNG included, in one kernel launch
(contended-run median 1.7B).

*Provenance (honesty policy):* measured 2026-07-21 on the current tree
(analytic-mask kernel + live-output sokoban runner); the host held
~2.4GB VRAM throughout — permanent on this machine. †The megakernel
row at B=65,536 is a **lower bound**: the per-run receipts are bimodal
(three runs host-time-sliced at 50-57×, two at 87-88×) — the
persistent kernel loses ~3× to time-slicing that barely moves the
small-op engines. Direct head-to-head receipts put the current kernel
at **2.9-3.3× the kernel that measured 91×/1.93B on a verified-quiet
GPU** (`data/p1_orient_ab.jsonl`, `data/e1_megakernel_canmask_ab.jsonl`,
`data/e5_rowmove_ab.jsonl`). The sokoban column is honestly LOWER than
earlier tables: the old runner let the compiler delete its output
work, inflating it ~1.63× (`data/e3_soko_dce_ab.jsonl`).

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
  measurement harness (counterbalanced ABBA pairwise, frozen-code
  sweeps, medians-with-spread; see LEARNINGS §3 for the methodology
  rules) plus the focused `*_ab.py` receipt scripts (HOW_TO_RUN lists
  them).
- `benchmarks/pallas_lab.py` — the guided walk up to the megakernel.
- `data/` — the receipts: one `*.jsonl` per adopt/kill/null decision,
  raw rows from the n=5 fresh-process A/Bs the docs cite.
- `djinnax/distributions.py` — closed-form collapse of stochastic
  loops (geometric tries, conditional categorical, alias tables).

## Honesty policy

Medians with [min..max] from multiple fresh-process runs; interleaved
pairwise for anything sub-2×; null results kept and labeled; known
caveats (RNG-stream differences, hardware specificity, single-GPU-model
evidence) stated next to the numbers they qualify. The head-to-head
harness deliberately runs the same driving protocol on both sides of
every ratio (see its docstring), so its absolute env-steps/s are
conservative for the djinn engines.
