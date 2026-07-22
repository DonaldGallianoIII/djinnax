# HOW TO RUN — environment, the shim, and every script

## Requirements

- Python 3.12, JAX 0.10.x with CUDA 12 pip wheels (`jax[cuda12]`).
- NVIDIA GPU. **Tested on RTX 4070 (Ada, sm_89).** The megakernel/pallas
  code uses the Pallas **Triton** lowering (works on sm_80+ class GPUs);
  the default Mosaic-GPU lowering needs Hopper (sm_90) and is NOT used.
- CPU-only machines: parity/RNG/chain-link tests run fine (`pytest
  tests/` passes with the GPU-kernel tests skipped); benches are
  meaningless on CPU.
- Extra pip deps for the reference comparisons: `svgwrite dm-env
  gymnasium requests matplotlib` (jumanji's heavy optional deps are
  stubbed by `refs.py`, not installed).

## THE SHIM (read this or you will silently benchmark a CPU)

JAX 0.10's cuda12 pip wheels load cuSPARSE before nvJitLink and **fall
back to CPU silently** when that fails. Two defenses, use both:

1. Preload nvJitLink in every GPU invocation:
   ```bash
   export LD_PRELOAD=<venv>/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12
   ```
2. Trust nothing: every bench here prints `backend: gpu|cpu` — check it.
   If it says cpu, your numbers are garbage.

GPU-sharing etiquette baked into all examples (adjust to taste):
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

## Reference engines (needed for parity + head-to-head only)

```bash
mkdir -p reference-engines && cd reference-engines
git clone --depth 1 https://github.com/sotetsuk/pgx.git
git clone --depth 1 https://github.com/instadeepai/jumanji.git
```
`refs.py` puts them on sys.path; nothing is pip-installed from the
clones.

Without the clones, the reference-parity tests **skip** with a
pointer here. CI sets
`DJINNAX_REQUIRE_REFS=1` so a missing reference fails loudly there —
set it locally too if you want the strict behavior.

## The scripts, in the order you should run them

| command | what it does | needs GPU |
|---|---|---|
| `python checks/check_parity.py` | all engines ≡ pgx/jumanji move-for-move | no (slow-ok on CPU) |
| `python checks/check_megakernel.py` | megakernel battery: chain link, determinism, chaining, adversarial, RNG | yes |
| `pytest tests/ -q` | the same gates, CI-shaped — 41 tests (canonical count; GPU-kernel tests skip on CPU, reference-parity tests skip without the clones) | optional |
| `python benchmarks/bench_head_to_head.py` | one bench run, all engines, within-run ratios | yes |
| `python benchmarks/sweep_stats.py --n 5` | the OFFICIAL numbers: n fresh-process runs, median [min..max] | yes |
| `python benchmarks/floor_bench.py` | runtime-floor probe (NullEnv) + protocol A/B | yes |
| `python benchmarks/pallas_lab.py` | guided Pallas intro (kernel basics → fused row-move) | yes |
| `python -m djinnax.megakernel_rng` | megakernel parity + RNG battery + headline benches | yes |

Useful flags on `benchmarks/bench_head_to_head.py`: `--batches 64 1024 8192 65536`,
`--unroll N`, `--rng threefry2x32|rbg|unsafe_rbg`, `--json out.jsonl`,
`--best-effort` (default is strict: non-GPU backend or any engine
failure exits nonzero — official sweeps must fail closed).

## The A/B receipt scripts (all GPU)

Every adopt/kill/null decision in the docs traces to one of these, run
as `for i in 1 2 3 4 5; do python benchmarks/<script> --json data/<receipt>.jsonl; done`
(fresh process per run, counterbalanced ABBA rounds via
`benchmarks/ab_timing.py`). Raw rows live in `data/*.jsonl` — one file
per experiment, kept forever.

| script | decision it settled | receipt |
|---|---|---|
| `canmask_analytic_ab.py` | analytic legality predicate (ADOPTED, both XLA engines) | data/e2_canmask_analytic_ab.jsonl |
| `megakernel_canmask_ab.py` | analytic mask in-kernel (ADOPTED) | data/e1_megakernel_canmask_ab.jsonl |
| `megakernel_rowmove_ab.py` | 9-bubble row move (ADOPTED) | data/e5_rowmove_ab.jsonl |
| `megakernel_orient_ab.py` | orient-select vs all-moves kernel (ADOPTED, P1) | data/p1_orient_ab.jsonl |
| `orient_gather_ab.py` | permutation-gather orientation (ADOPTED, P4) | data/p4_orient_gather_ab.jsonl |
| `spawn_variants_ab.py` | rank-pick spawn + direct reset (ADOPTED, P2/P3) | data/p23_spawn_ab.jsonl |
| `spawn_collapse_ab.py` | collapsed spawn vs rejection loop (exhibit, 47-75×) | printed |
| `canmask_ab.py` | CHANGED_LUT probe (KILLED, P5) | data/p5_canmask_ab.jsonl |
| `soko_carry_ab.py` | carried on-target count (NULL, P6) | data/p6_soko_carry_ab.jsonl |
| `soko_gated_ab.py` | batch-gated reset, sync+desync (KILLED, P-S1) | data/ps1_soko_gated_ab.jsonl |
| `rng_rounds_ab.py` | 1 vs 2 fmix rounds (NULL — 2 stays, P8) | data/p8_rng_rounds_ab.jsonl |

## Measurement rules (the short version — LEARNINGS §3 is the law)

1. Check the backend line. 2. Never compare numbers across runs — only
within-run ratios. 3. Headlines come from `sweep_stats.py` (frozen code,
n≥5, median with spread). 4. Anything sub-2× needs interleaved pairwise
A/B (sequential A-then-B measures your GPU's clock ramp, not your code).
5. Close your games/browsers — and if you didn't, disclose it; ratios
mostly survive contention, absolutes and bandwidth ablations don't.

## Known machine-specific numbers

Compile times: XLA engines 1-16s; megakernel ~30s (flat in n_steps).
All published ratios were measured on one RTX 4070 under WSL2 — treat
them as one machine's evidence, reproduce on yours with `sweep_stats.py`.
