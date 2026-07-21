# Megakernel plan — "environment-on-chip" for 2048

**Status:** STAGES 1-3 COMPLETE (2026-07-20, `megakernel.py`) — it
works and it moved the floor. Probes all passed (static ref indexing,
fori tuple carries, 16-lane pressure at BLOCK=128). Parity:
**bit-identical over 64 steps** via the same-function trick
(`step_lanes()` runs unchanged inside the kernel AND under XLA scan).
Results (interleaved medians, unlocked clocks):
- vs same-formulation XLA (uniforms pre-generated both sides — pure
  execution-strategy comparison): **3.71× [2.44..4.54]** at B=1024,
  2.20× at 8192, **3.04× [2.15..4.41]** at 65536; peak observed
  ~2.8B env-steps/s.
- vs the PRODUCTION LUT runner (self-contained both, RNG gen cost
  included on the megakernel side): **1.67× [1.01..2.23] / 2.25×
  [2.01..2.65] / 2.33× [2.07..2.65]** at B=1024/8192/65536.
The megakernel is the new production champion. Kernel compile ~33-36s at
large B (Triton, 64-step fori) vs 11-16s XLA — one-time cost, noted.
**Stage 4 COMPLETE (`megakernel_rng.py`)** — in-kernel double-fmix32
counter RNG of (env_id, step, salt, seed); pure uint32 ops, so the
same-function trick extends to the RNG: Mode B is **bit-verified vs its
XLA reference** AND distribution-tested (mean/std/bucket flatness/
cross-salt corr/0.9-split). Results (interleaved):
- Mode B vs Mode A self-contained: **1.9-2.3×** (in-kernel RNG kills the
  uniform-buffer HBM traffic — that's what it buys).
- Mode B vs production LUT runner: **4.7× / 7.6× [7.2..10.0] /
  6.2× [5.8..9.4]** at B=1024/8192/65536; 2.17B env-steps/s
  self-contained.
- **Direct vs jumanji (same game): 268× [239..360] at B=8192,
  116× [88..133] at B=65536.** RNG-stream caveat: megakernel uses the
  counter hash (same spawn distribution, rank-pick), jumanji uses
  threefry — game semantics parity-chained through game2048; streams
  differ by design.
**Goal:** one kernel launch runs the ENTIRE rollout — every env steps N
times inside the kernel, state never leaves registers/SMEM between steps.
This attacks the launch/intermediate-materialization floor itself, which
every XLA variant converges to (NullEnv evidence, LEARNINGS §6). It is
the only idea on the table that could move the floor rather than approach
it. Whatever we learn transfers directly to the WebGPU browser demo
(persistent WGSL compute is the same shape).

## Why 2048

Most logic of any ported env, and we've proven the whole step fits in
registers (pallas_lab row-move = swap network, no gathers needed). Envs
are independent across B (embarrassingly parallel), steps are sequential
per env — exactly the persistent-kernel shape: grid over env blocks,
in-kernel loop over steps.

## Architecture

- **Backend:** Pallas **Triton** lowering (`plt.CompilerParams()`) — the
  Mosaic default is Hopper-only (sm_90 TMA; we are sm_89). Known Triton
  constraints from pallas_lab: no `slice`/`.at[]` on in-kernel arrays →
  **structure-of-arrays**: the board's 16 cells travel as separate
  `(BLOCK,)` lane arrays (refs built programmatically); float `pow` is
  inexact → integer shifts for rewards.
- **Grid:** `B // BLOCK` programs, each owning BLOCK envs (start 128;
  tune 64/256 for register pressure).
- **Step loop:** `lax.fori_loop` inside the kernel body (unrolling 64
  steps would explode compile time); carry = 16 board lanes + action
  mask (4 lanes) + step_count + any counters.
- **Per step, fully in registers:**
  1. sample action from the CARRIED mask (rank-pick: one uniform,
     cumsum-over-4 select — integer ops only);
  2. compute all 4 direction moves of the current board via the swap
     network on static lane rewirings (no orientation select needed in
     SoA — each direction is just different lane wiring, free in
     registers); select moved board + merge reward by sampled action;
  3. spawn: one uniform → rank-pick over the 16 empty lanes; value
     uniform < 0.9 → exp 1 else 2; gated on action-was-legal;
  4. new mask = any-changed of the 4 moves of the NEW board (reuses the
     machinery from 2); done = no legal direction;
  5. in-register reset where done: zero lanes + spawn + analytic
     single-tile mask (WRITING_FAST_ENVS rung 2).
- **Outputs:** final board lanes + step/return accumulators (whatever the
  bench needs; keep minimal).

## RNG — two modes, two purposes

- **Mode A (parity): pre-generated uniforms.** A `(n_steps, 5, B)` f32
  buffer of uniforms generated OUTSIDE by jax.random, consumed by both
  the megakernel AND a refactored XLA reference pipeline (same step
  semantics, same uniforms). States must then be **bit-identical over
  all N steps** — this is the parity gate for the loop structure + all
  game logic, sidestepping "match JAX's threefry bit-for-bit in Triton"
  (fragile, not the point).
- **Mode B (performance): in-kernel counter RNG.** Philox-lite /
  squares-style counter hash of `(env_id, step, salt)` — pure integer
  ops, no randomness traffic from HBM. Not bit-comparable to Mode A by
  design; gets DISTRIBUTION tests instead (uniformity of picks, spawn
  value ratio ~0.9/0.1, spawn-on-empty invariant) plus the standard
  conformance invariants on the resulting states.
- Mode A vs Mode B interleaved A/B also isolates "what does in-kernel
  RNG actually buy" — a clean ablation.

## Build order (each stage committed)

1. **Probes:** static ref indexing (`ref[:, i]`) vs SoA refs; fori_loop
   with tuple-of-arrays carry under the Triton lowering; register
   pressure at BLOCK=128 (compile + run a 2-step loop).
2. **Mode A megakernel** + refactored uniform-consuming XLA reference +
   bit-parity gate over 64 steps at B=1024.
3. **Bench:** interleaved vs the current best XLA runner (protocol v1 +
   LUT step) at B=1024/8192/65536. Claim bar: interval clearly > 1
   (unlocked clocks — only a multiple is a result).
4. **Mode B** in-kernel RNG + distribution tests + Mode A/B ablation.
5. LEARNINGS + WRITING_FAST_ENVS updates (rung-4 verdict with data).

## Risks / abort criteria

- Triton lowering rejects fori carries or blows registers → fall back to
  smaller BLOCK, then to a chunked loop (K steps per launch), before
  declaring the approach dead. A chunked megakernel (e.g. 8 steps per
  launch) still divides launch count by 8 and is a valid partial win.
- Compile time explodes → cap unroll, document.
- If the win at B=1024-8192 isn't visibly > 1 through clock noise, the
  verdict is "the floor is bandwidth/scheduling, not launches" — that is
  a publishable null result and closes rung 4 honestly.

## Success = any of

- ≥2× median (interval > 1) over best XLA pipeline at any training-
  relevant B; or
- a clean null result that pins the floor's composition; or
- (either way) the in-kernel RNG + persistent-loop machinery documented
  for the WebGPU port.
