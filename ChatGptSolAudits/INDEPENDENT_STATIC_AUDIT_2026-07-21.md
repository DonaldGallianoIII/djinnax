# Djinnax independent static code audit

**Reviewer:** OpenAI Codex (GPT/Sol session)  
**Date:** 2026-07-21  
**Method:** Static source review only

## Independence and scope

I did **not** open or read `docs/EXTERNAL_REVIEW_R2.md`, nor did I read any other review/audit report. I did not execute Djinnax, import its Python package, run tests, compile JAX, or run benchmarks. I did not modify source code. The only repository write made for this review is this Markdown report and its containing directory.

The current source and tests themselves contain comments naming earlier external-review findings. Those comments are inseparable from a review of the current code. I did not treat tagged, already-resolved items as discoveries in this report. Every issue below is supported by current control flow, current public contracts, or a current test/benchmark gap.

Reviewed deeply:

- `djinnax/` engine, runtime, sampler, LUT, reference-glue, and megakernel modules
- `tests/` and `checks/`
- The official benchmark path: `bench_head_to_head.py`, `floor_bench.py`, and `sweep_stats.py`
- The focused A/B and Pallas benchmark call sites
- Public package metadata, README, and repository engineering rules

No dynamic performance claim is made here. Performance ideas are labeled as hypotheses until they pass parity and interleaved A/B measurement.

## Executive summary

The most urgent issues are not in the core 2048 move arithmetic. They are at episode boundaries and system boundaries:

1. Sokoban can return a reset state paired with an observation of the terminated state.
2. `terminated` and auto-reset mean different things across the three exported environments, despite public documentation claiming a shared convention.
3. Mode A `run_megakernel` silently leaves a tail uncovered when `B` is not divisible by 128.
4. The official benchmark pipeline can accept CPU fallback or partial engine failures and still emit an apparently successful aggregate.
5. Some head-to-head workloads are asymmetric after JIT dead-code elimination, so the repository cannot attribute every published ratio purely to engine implementation from this harness alone.

| ID | Severity | Finding | Static confidence |
|---|---|---|---|
| SOL-01 | High | Sokoban reset state and returned observation diverge on terminal steps | Confirmed |
| SOL-02 | High | Auto-reset and `terminated` contracts conflict across environments | Confirmed |
| SOL-03 | High | Mode A megakernel silently truncates non-block-aligned batches | Confirmed |
| SOL-04 | High | Official benchmark orchestration fails open | Confirmed |
| SOL-05 | High | Head-to-head work is asymmetric for Sokoban and the megakernel | Confirmed/strong inference |
| SOL-06 | Medium | Sub-2× comparisons are produced by sequential, fixed-order timing | Confirmed |
| SOL-07 | Medium | The 2048 LUT is not correct for all reachable game states | Confirmed limitation |
| SOL-08 | Medium | `expected_tries` cannot represent valid small-probability results safely | Confirmed |
| SOL-09 | Medium | Alias-table construction accepts invalid distributions silently | Confirmed |
| SOL-10 | Medium | Counter-RNG identities alias across shards/devices without a global offset | Confirmed design risk |
| SOL-11 | Medium | Exported Sokoban defaults to unvalidated, potentially unsolvable benchmark fixtures | Confirmed product risk |
| SOL-12 | Low/Medium | Top-level import eagerly builds tables and levels and initializes JAX arrays | Confirmed |
| SOL-13 | Low | Public docs and test-count claims have drifted from the code | Confirmed |

## Detailed findings

### SOL-01 — Sokoban returns an observation that does not describe its returned state

**Severity:** High  
**Location:** `djinnax/sokoban.py:151-166`; coverage gap in `checks/check_parity.py:217-228`

`DjinnSokoban.step` constructs `obs` from `new_v` and the old `state.fixed_grid` before auto-reset. It then replaces the returned state's grids and agent with `rf`, `rv`, and `ra` wherever `done` is true.

Consequently, for every terminal row:

- `new_state.variable_grid` describes a newly sampled level;
- `new_state.fixed_grid` describes that newly sampled level;
- returned `obs` describes the final position of the previous level.

A policy that chooses its next action from returned `obs` will apply that action to a different hidden state. The parity test compares observations only inside `if not j_done`, so it deliberately leaves this boundary unchecked.

**Recommendation:** Choose and document one terminal-step contract:

- Return the reset observation with the reset state, and place the terminal observation in `extras["final_observation"]`; or
- Return the terminal state and terminal observation, and perform reset outside `step`.

Add a test that forces both solved and time-limit termination and asserts observation/state coherence on the returned value.

### SOL-02 — The exported environments do not share an episode-boundary contract

**Severity:** High  
**Locations:** `README.md:36-40`; `djinnax/__init__.py:6-10`; `djinnax/ttt.py:93-157`; `djinnax/game2048.py:291-303`; `djinnax/sokoban.py:158-167`

The package says all environments use in-step auto-reset, but their behavior differs:

- Tic-tac-toe does not auto-reset. Once terminated, later calls freeze the terminal state.
- 2048 auto-resets its board and mask but sets `terminated=True` on that already-reset state.
- Sokoban similarly returns an already-reset state with `terminated=True`, while returning the previous episode's observation.

The same field therefore means “this state is terminal” in TTT and “the transition that produced this already-reset state terminated” in 2048/Sokoban. A generic wrapper can freeze TTT forever, double-reset the other games, or bootstrap from the wrong observation.

**Recommendation:** Introduce one explicit transition type, for example:

```text
StepOutput(state, observation, reward, terminated, final_observation, extras)
```

Then implement either external reset everywhere or consistent autoreset everywhere. Keep `terminated` as transition metadata rather than overloading a state field with different meanings.

### SOL-03 — Mode A megakernel silently truncates non-aligned batches

**Severity:** High  
**Locations:** `djinnax/megakernel.py:269-290`; contrast `djinnax/megakernel_rng.py:120-125`

`run_megakernel` launches `grid=(Bn // BLOCK,)` with `BLOCK=128` and has no divisibility check. If `Bn` is not divisible by 128, no program owns the tail. The declared output still has shape `(Bn, ...)`, so those rows are not valid computed results.

Mode B correctly rejects this condition, but its test covers only `run_megakernel_rng`. The same unsafe grid arithmetic also appears in educational Pallas helpers.

**Recommendation:** Apply the Mode B guard to Mode A and every Pallas helper. Better still, support a masked final tile or explicitly pad/unpad. Validate `B > 0`, block positivity, input rank, and uniform-buffer shape. Add a CPU-level argument test and a GPU tail-coverage test for Mode A.

### SOL-04 — The official benchmark path fails open

**Severity:** High  
**Locations:** `benchmarks/bench_head_to_head.py:291-311`; `benchmarks/sweep_stats.py:43-66`

The core head-to-head script prints the backend but does not require GPU. It catches every per-engine exception, prints `FAILED`, and exits successfully. `sweep_stats.py` treats process exit code zero as a valid run and aggregates whichever engine records happen to exist.

This allows an “official” sweep to complete when:

- JAX silently fell back to CPU;
- reference engines failed to import;
- one or more engines OOMed or failed to compile;
- a batch size was unsupported by one variant.

The resulting output can look statistically complete while ratios or engines are silently missing.

**Recommendation:** Make sweep-invoked runs strict by default:

- Assert `jax.default_backend() == "gpu"` before any timing.
- Fail the process if any required engine fails.
- Require a complete `(run, B, engine)` matrix before aggregation.
- Record backend, device, JAX version, commit/tree identity, protocol, sampler, RNG, steps, reps, and failure status in every JSON row.
- Keep a separate `--best-effort` mode for exploratory local runs.

Also resolve `--out` to an absolute path before passing it to the child process; relative output paths are interpreted under different working directories by the parent and child.

### SOL-05 — Some “same protocol” head-to-head comparisons do different work

**Severity:** High for attribution of benchmark claims  
**Locations:** `benchmarks/bench_head_to_head.py:186-230`, `233-254`; `djinnax/sokoban.py:151-156`; `djinnax/megakernel.py:177-228`

There are two separate asymmetries.

**Sokoban:** The Djinn runner selects only the returned state from `game.step`; reward, observation, and extras are unused inside the jitted scan. JAX can eliminate those outputs. The Jumanji runner carries and returns its `TimeStep`, keeping its output surface live. The source comment that Djinn “also builds” comparable outputs is insufficient when the benchmark's observable graph discards them.

**Megakernel:** The benchmark description says protocol v1 is applied identically, but the megakernel samples actions internally with rank-pick plus its counter RNG. Jumanji uses masked categorical sampling with the selected JAX RNG. `--rng` is nevertheless recorded against the megakernel even though it does not change the kernel RNG. In addition, `_fresh_inputs` initializes every starting tile as exponent 1, rather than the engine's 0.9/0.1 exponent-1/exponent-2 distribution.

These may be valid end-to-end product comparisons, but they do not isolate engine execution under one identical protocol.

**Recommendation:** Publish two clearly separated measurements:

1. **Execution-strategy A/B:** identical pre-generated actions/uniforms, identical live outputs, and the same starting states.
2. **End-to-end system throughput:** each engine's preferred RNG/sampler/reset path, explicitly labeled as such.

For Sokoban, consume a checksum or carry a matched transition output on both sides so compiler dead-code elimination cannot make the workloads asymmetric.

### SOL-06 — Timing order does not support sub-2× conclusions

**Severity:** Medium  
**Locations:** `benchmarks/bench_head_to_head.py:301-322`; `benchmarks/sweep_stats.py:93-106`; focused A/B loops such as `benchmarks/canmask_ab.py:41-44`

The official head-to-head measures all engines sequentially and later divides their best times. This is a within-process ratio, but not an interleaved pairwise ratio. The focused A/B scripts interleave calls but always use the same A-then-B order, leaving a fixed position effect.

The repository's own methodology says sub-2× results require interleaved pairwise timing. Therefore the advertised official pipeline cannot, on its own, substantiate 1.x TTT/Sokoban rows.

**Recommendation:** Use paired rounds with counterbalanced order (`ABBA` or randomized balanced order), retain each round's paired ratio, and aggregate those ratios across fresh processes. Do not derive small ratios from independently selected best-of-repetition minima.

### SOL-07 — The 2048 LUT is not correct for the full reachable state space

**Severity:** Medium; higher if `make_game2048_lut()` is advertised as a fully conforming environment  
**Locations:** `djinnax/game2048_lut.py:10-17`, `33-47`; `checks/check_parity.py:322-338`

The LUT saturates a `15 + 15` merge at exponent 15 and awards `2^15`; the branchless/reference behavior produces exponent 16 and reward `2^16`. This is documented and pinned, but the assertion that it is “unreachable in play” is not enforced by any invariant. A long-running or deliberately optimized policy can enter this state class; the engine has no rule that prevents it.

Once full reference parity is claimed, rarity is not a correctness boundary.

**Recommendation:** Either:

- Add a correct branchless escape path for rows containing exponent 15 or greater;
- Expand the representation/LUT and still retain an escape for its next boundary; or
- Rename the variant as explicitly bounded and reject unsupported boards before a divergent transition.

The parity suite should treat this as unsupported behavior or correct behavior, not as a passing divergence.

### SOL-08 — `expected_tries` overflows its output contract for valid probabilities

**Severity:** Medium  
**Location:** `djinnax/distributions.py:123-127`

`expected_tries` converts `ceil(1 / p)` directly to `int32`. For valid positive probabilities below roughly `1 / INT32_MAX`, the mathematical answer cannot fit. Smaller values can also produce float infinity before the cast. Unlike `geometric_tries`, this sibling has no pre-cast clamp or documented saturation.

**Recommendation:** Define the intended contract explicitly: return float expectation, return int64 where enabled, or saturate safely before the cast. Validate `0 < p <= 1` for host inputs and add tiny-`p`, zero, negative, NaN, and `p > 1` tests.

### SOL-09 — Alias-table construction silently accepts invalid distributions

**Severity:** Medium  
**Locations:** `djinnax/distributions.py:91-120`; `tests/test_distributions.py:79-88`

`build_alias_table` does not reject empty, multidimensional, negative, non-finite, or all-zero weights. All-zero input divides by zero; negative/NaN input can create nonsensical tables that still have plausible shapes. The test suite covers only one valid distribution.

This validation is host-side NumPy work performed once, so strict checks do not affect the hot path.

**Recommendation:** Require a non-empty 1-D finite array with nonnegative weights and positive finite sum. Validate that generated probabilities are within `[0, 1]` and aliases within range. Add invalid-input and zero-weight-category tests.

### SOL-10 — Counter-RNG identities are local to one launch, not globally unique

**Severity:** Medium  
**Locations:** `djinnax/megakernel_rng.py:89-103`; adapter at `benchmarks/bench_head_to_head.py:245-251`

The kernel derives `env_id` from local program/block indices. Two devices, hosts, or independently sharded calls using the same seed and time offset therefore generate identical streams for corresponding local rows. The API exposes no global environment offset.

The benchmark adapter also reduces a JAX key to only `bits[0]`, discarding the other key word and unnecessarily shrinking stream identity.

**Recommendation:** Accept a caller-provided global `env_id_base` or per-row IDs, and mix all key words plus process/device/shard identity into a wider counter construction. Add a test proving that concatenated shards match a monolithic run without cross-shard duplicate streams.

### SOL-11 — The exported Sokoban environment uses benchmark fixtures that are not validated as games

**Severity:** Medium product risk  
**Locations:** `djinnax/soko_levels.py:1-8`, `21-45`; export in `djinnax/__init__.py:20`

The generator intentionally does not require solvability. It chooses interior walls, targets, boxes, and the agent from shuffled free cells without connectivity, dead-square, box-to-target reachability, or full puzzle-solvability checks. The comment that boxes are off corner pockets is not implemented beyond keeping them off occupied cells.

That is acceptable for a throughput fixture, but `DjinnSokoban` is exported as the package's default Sokoban environment. A training user can receive unsolvable levels whose only terminal path is the time limit.

**Recommendation:** Separate `BenchmarkSokoban` fixtures from a production default backed by known-solvable levels, or validate generated levels offline and ship only accepted fixtures. Document the level distribution as part of the environment specification.

### SOL-12 — Lightweight imports eagerly do substantial unrelated work

**Severity:** Low/Medium performance and usability  
**Locations:** `djinnax/__init__.py:13-21`; `djinnax/game2048_lut.py:50-67`; `djinnax/soko_levels.py:21-50`

Top-level `import djinnax` imports the LUT module, runs a 65,536-entry Python LUT build, converts three tables to JAX arrays, imports Sokoban, and generates 256 levels. Importing a submodule such as `djinnax.distributions` also executes package initialization first.

This adds avoidable startup and backend-initialization cost to CLIs, tests, worker processes, and users who need only a sampler or one simple environment.

**Recommendation:** Make game exports lazy, ship versioned precomputed NumPy assets with checksums, and convert tables to device arrays only when the LUT engine is instantiated. Keep the small public namespace through `__getattr__` without eager imports.

### SOL-13 — Documentation and test inventory have drifted

**Severity:** Low  
**Locations:** `README.md:59-62`; `HOW_TO_RUN.md:41-62`; current `tests/test_*.py`

There are 28 statically declared test functions, while public documentation reports 26 and gives CPU pass/skip counts that no longer describe all reference and GPU skips. The README also says every environment auto-resets, contradicted by TTT.

**Recommendation:** Generate test counts in CI/release notes rather than hard-coding them, and add documentation contract tests for reset behavior rather than only executing the README's 2048 snippet.

## Performance opportunities requiring measurement

These are hypotheses, not measured claims.

### PERF-01 — Replace four full 2048 move probes with a direct legality predicate

Both XLA 2048 and the persistent kernel compute full moved boards in all four directions to obtain four `changed` bits (`game2048.py:239-244`; `megakernel.py:208-213`). In the megakernel this means compaction and merge networks for every direction after already applying the chosen move.

For a fixed four-cell row, legality can be expressed directly as:

- a nonzero tile having an empty slot before it in movement order; or
- equal consecutive tiles in the nonzero sequence.

An unrolled boolean predicate avoids reward arithmetic, output-lane construction, and LUT gathers. The existing changed-LUT regression does not rule this out; it tested an extra gather, not a direct predicate. Gate against `_move_dir(...)[2]` over exhaustive 4-cell rows and then run an interleaved full-step A/B.

### PERF-02 — Batch-gate Sokoban reset sampling

`DjinnSokoban.step` samples and gathers a full `(B, 10, 10)` fixed and variable level for every row on every step, even when no row terminated (`sokoban.py:158-164`). The usual timeout is synchronized at 120 steps, and solved events appear rare under the benchmark's random policy.

This is a good candidate for the repository's sanctioned batch-level `lax.cond(jnp.any(done), reset_block, no_reset, ...)`. Measure coverage and `any(done)` frequency; abandon it if training desynchronization makes the condition nearly always true.

### PERF-03 — Build a minimal entity-list Sokoban core

The hot state carries two 100-cell grids per environment to move one agent and four boxes. A core state of `level_id`, agent coordinate, four box coordinates, step count, and target occupancy can use static level tables for wall/target lookups. Materialize the dense observation only at the API boundary when requested.

This changes the workload and therefore needs a new end-to-end parity gate. It should be offered as a production engine, while the current full-grid version can remain as the strict reference-work comparison.

### PERF-04 — Separate minimal internal state from compatibility outputs

TTT stores rewards, observation, and legal masks alongside core board state. Sokoban always exposes a dense observation and extras. At very large `B`, derived fields increase carry traffic and make benchmark fairness dependent on whether outputs remain live.

Use a minimal internal state plus explicit `observe`, `action_mask`, and compatibility-adapter functions. Bench both “core transition” and “full public transition” so eliminated output work is visible rather than accidental.

### PERF-05 — Carry a four-bit legality mask across megakernel chunks

Mode B recomputes `_initial_mask` from the board at every launch. For small rollout chunks, carrying the mask can remove four direction probes per launch. The extra four booleans of HBM traffic may cost more than the saved ALU, so test it only in the chunked configurations where launch tax is already material.

### PERF-06 — Lazy/prebuilt LUT assets

Precompute the 2048 tables during packaging, store compact host arrays, validate them with a checksum, and lazily place them on the active device. This targets import/worker startup rather than step throughput and should be measured separately from kernel execution.

## Test and verification gaps to close first

1. Terminal Sokoban state/observation coherence for both solve and time-limit paths.
2. A cross-environment reset-contract test defining what `terminated`, returned state, and returned observation mean.
3. Mode A `B % BLOCK != 0`, `B < BLOCK`, and zero-batch coverage.
4. Official-benchmark smoke tests that reject CPU, incomplete engine matrices, and mislabeled RNG protocols.
5. A Sokoban benchmark graph that keeps equivalent output fields live on both sides.
6. `expected_tries` boundary tests and alias-table invalid-input tests.
7. Multi-shard counter-RNG uniqueness/equivalence tests using global environment IDs.
8. A correct or explicitly rejected 2048 exponent-15 merge transition.
9. Solvability/quality validation for any Sokoban levels exposed as production defaults.
10. A package-import budget test so lightweight imports do not build unrelated games or initialize device tables.

## Suggested remediation order

1. Fix or explicitly redefine Sokoban terminal observation semantics.
2. Standardize the reset/termination contract across all public environments and update documentation.
3. Guard every Pallas grid boundary, starting with Mode A.
4. Make the official sweep fail closed and separate matched-protocol from end-to-end benchmarks.
5. Correct or bound the 2048 LUT saturation behavior.
6. Harden distribution helpers and global counter-RNG identity.
7. Separate benchmark Sokoban fixtures from a solvable production environment.
8. Evaluate PERF-01 and PERF-02 first; they offer the strongest source-level mechanism for step-time gains without immediately redesigning the whole engine.
9. Pursue entity-list state and lazy imports as separate architecture/startup projects with their own measurements.

## Strengths worth preserving

- The code makes parity a first-class engineering constraint rather than a post-hoc benchmark check.
- State shapes and dtypes are explicit, and the hot paths largely follow a disciplined batch-native model.
- The megakernel/XLA shared-step structure is a strong way to isolate execution strategy.
- Known limitations are often written near the code and pinned by tests, which makes remaining contract decisions visible.
- Focused A/B scripts generally warm both variants and retain parity gates; adding fail-closed orchestration and counterbalanced order would make that evidence substantially stronger.

---

**Audit boundary:** This report is a static review. None of the proposed fixes or speedups has been implemented or executed, and no source file was changed.
