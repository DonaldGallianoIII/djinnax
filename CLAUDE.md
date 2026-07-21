# engine-bench — agent instructions

JAX game engines in the djinn batch-native discipline, parity-gated
against pgx/jumanji, measured to 98-237× the references. When writing or
porting env code here (or in a repo that points at this one): **follow
this file; it overrides generic JAX habits.**

## Read order for any env work
1. PORTING_PLAYBOOK.md — decide representation + method FIRST
2. WRITING_FAST_ENVS.md — the idioms while writing
3. HOW_TO_RUN.md — environment, the CUDA shim, scripts
4. LEARNINGS.md — evidence behind every rule (consult, don't skip rules)

## Hard rules (violations are bugs, not style)
- Batch-native: leading (B, …) on every state field; flax.struct; never
  vmap-a-single-env as the architecture.
- NO `lax.while_loop`, NO data-dependent `lax.switch`/`cond`, NO python
  branching on traced values in the step path. Compute-all + `where`
  mask-select; one-hot writes; python-for only over small static ranges.
- Smallest dtype that fits; widen→add→clamp→narrow near limits;
  sentinel -1 for empties; fixed shapes forever.
- Keys threaded per step, never stored; `fold_in` with a documented salt
  per site; per-entity keys via `vmap(fold_in)` (NEVER broadcast a key);
  prefer counter-hash RNG (`megakernel_rng.hash_uniform` pattern).
- Mask and apply share ONE predicate/choke point; a mask row is never
  all-False (keep an always-legal inert action).
- Obs returns through `lax.stop_gradient`, normalized, layout documented.

## The gate — nothing merges without
Parity vs the reference implementation (build the replay harness BEFORE
optimizing), variant≡variant bit-equivalence when optimizing, mask/apply
property test, conformance driver, `chex.assert_max_traces(step, 1)`,
serialization round-trip. Templates: checks/, 
tests/.

## Measurement rules
Print + check `jax.default_backend()` (CPU fallback is silent — see
HOW_TO_RUN for the LD_PRELOAD shim). Within-run ratios only; headlines
via `sweep_stats.py` (frozen code, n≥5, median [min..max]); sub-2×
claims require interleaved pairwise A/B; never edit benched code
mid-sweep; disclose GPU contention.

## Escalation ladder (stop when at the NullEnv floor)
branchless → analytic deletion → LUT (any ≤16-bit sub-state) →
megakernel (persistent Triton kernel, SoA lanes, same-function parity,
in-kernel counter RNG). Pallas here = Triton lowering
(`plt.CompilerParams()`); the Mosaic default is Hopper-only.
