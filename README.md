# djinn engine-bench — batch-native JAX game engines, with receipts

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
frozen-code sweep on a quiet RTX 4070 (full intervals in
`docs`/commit history). Peak absolute: **1.9B env-steps/s** for full
2048 games, RNG included, in one kernel launch.

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
- 

- `checks/` + `tests/` — the correctness
  gates. **No number in this repo was measured before its engine passed
  parity.** `pytest tests/` runs 16 tests on GPU, 11 (+5 skips) on CPU.
- `benchmarks/` — the
  measurement harness (interleaved pairwise, frozen-code sweeps,
  medians-with-spread; see LEARNINGS §3 for the methodology rules).
- `benchmarks/pallas_lab.py` — the guided walk up to the megakernel.

## Honesty policy

Medians with [min..max] from multiple fresh-process runs; interleaved
pairwise for anything sub-2×; null results kept and labeled; known
caveats (RNG-stream differences, hardware specificity, single-GPU-model
evidence) stated next to the numbers they qualify.
