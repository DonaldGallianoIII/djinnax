# djinnax — agent instructions

JAX game engines in the djinn batch-native discipline, parity-gated
against pgx/jumanji, measured to ~100-270× the references (exact table:
README, with provenance). When writing or porting env code here (or in
a repo that points at this one): **follow this file; it overrides
generic JAX habits.**

## Read order for any env work
1. PORTING_PLAYBOOK.md — decide representation + method FIRST
2. WRITING_FAST_ENVS.md — the idioms while writing (§3c for kernels)
3. HOW_TO_RUN.md — environment, the CUDA shim, scripts
4. LEARNINGS.md — evidence behind every rule (consult, don't skip rules)
5. docs/MEGAKERNEL_PLAN.md — historical design log ONLY; build kernels
   from WRITING_FAST_ENVS §3c, not from the plan (its banner explains).

## Hard rules (violations are bugs, not style)
- Batch-native: leading (B, …) on every state field; flax.struct; never
  vmap-a-single-env as the architecture.
- NO `lax.while_loop`, NO data-dependent `lax.switch`/`cond`, NO python
  branching on traced values in the step path. Compute-all + `where`
  mask-select; one-hot writes; python-for only over small static ranges.
- Smallest dtype that fits (in HBM; int32 in kernel registers);
  widen→add→clamp→narrow near limits; sentinel -1 for empties; fixed
  shapes forever.
- Keys threaded per step, never stored; `fold_in` with a documented salt
  per site; per-entity keys via `vmap(fold_in)` (NEVER broadcast a key).
  Doctrine boundary: threaded keys for XLA envs; counter-hash uniforms
  (`megakernel_rng.hash_uniform`) in-kernel or where state-free replay
  matters.
- Mask and apply share ONE predicate/choke point; a mask row is never
  all-False (keep an always-legal inert action).
- Obs returns through `lax.stop_gradient`, normalized, layout documented.

## Before proposing ANY perf change
Check LEARNINGS §6 (the confirmed/killed ledger) and the `data/`
receipts first. Already measured and KILLED or NULL — do not re-propose
without new evidence: CHANGED_LUT-style compare-gathers (0.82-0.88×),
single-round fmix, carried on-target counts, unsafe_rbg, batch-gated
reset conds around fused-cheap blocks (0.14-0.54×), compute-all-4-moves
in registers (0.71×). Op counts lie; only interleaved A/B receipts count.

## Kernel work (megakernel_*/step_lanes/pallas)
- ANY change reruns the full battery: `checks/check_megakernel.py` /
  `tests/test_megakernel.py`. Kernel≡scan parity CANNOT catch a wrong
  shared component — analytic shortcuts need their own reference-chained
  gate (exhaustive where the domain allows).
- Guard every grid: `B % BLOCK != 0` raises loudly (never a silent tail
  drop). Validate input buffer shapes.
- Triton-on-sm_89 rules: no `slice`/`.at[]` on in-kernel arrays (SoA
  lanes instead); integer shifts, never float pow; np/python scalars for
  kernel-closure constants (jnp scalars break lowering).
- Chunk coarsely — chunked launches cost 2.2-3.9×; the chaining contract
  (carry score, recompute mask at entry) is exact only because the
  analytic mask is gated ≡ the computed one.

## Receipts (house convention)
Superseded variants are KEPT and pluggable, never deleted; every
adopt/kill decision has a `data/*.jsonl` receipt from an n=5
fresh-process ABBA A/B and is cross-referenced where the idiom is
taught. If you bound coverage, say what was dropped.

## The gate — nothing merges without
Parity vs the reference implementation (build the replay harness BEFORE
optimizing) and variant≡variant bit-equivalence when optimizing —
templates in checks/ and tests/. For envs feeding a training loop, also:
mask/apply property test, conformance driver,
`chex.assert_max_traces(step, 1)`, serialization round-trip (those gates
live in the training integration; see WRITING_FAST_ENVS §7).

## Measurement rules
Print + check `jax.default_backend()` (CPU fallback is silent — see
HOW_TO_RUN for the LD_PRELOAD shim; the bench pipeline is strict by
default and refuses CPU). Within-run ratios only; headlines via
`sweep_stats.py` (frozen code, n≥5, median [min..max]); sub-2× claims
require counterbalanced ABBA pairwise A/B (`benchmarks/ab_timing.py`);
never edit benched code mid-sweep; disclose GPU contention.

## Escalation ladder
branchless → analytic deletion → LUT (any ≤16-bit sub-state) →
megakernel (persistent Triton kernel, SoA lanes, same-function parity,
in-kernel counter RNG). Below rung 4, stop at the NullEnv floor; do NOT
stop before rung 4 because of the floor — the floor is an XLA artifact
the persistent kernel deletes (WRITING_FAST_ENVS §3/§3c). Pallas here =
Triton lowering (`plt.CompilerParams()`); the Mosaic default is
Hopper-only.
