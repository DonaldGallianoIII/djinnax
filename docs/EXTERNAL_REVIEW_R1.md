# External Review R1 — findings and remediation plan

Two independent external reviews of this repository (Development branch,
working tree at `79c04b3`), run 2026-07-21. Reviewer 1 was scoped to
**correctness**: anything that could make a published number or parity
claim wrong. Reviewer 2 was scoped to **performance opportunities** and
**public-release readiness**. Both were static-analysis only (no code
executed), confined to this repository, and forbidden from reading the
reference clones. Findings were reported without fixes; every item below
is worked through here with its own verification plan, and no fix lands
without its gate passing.

Status legend: `OPEN` → `FIXED (commit)` / `WON'T FIX (reason)` /
`NO CHANGE NEEDED (reason)`.

---

## What the review cleared (verbatim scope, so the findings have context)

Reviewer 1 explicitly cleared, after reading:

- megakernel lane wiring (`_GROUPS` verified against `cell = y*4+x` for
  all four directions), `_row_move_4`/`_bubble` merge order vs
  `_merge_left` semantics, in-kernel reset + analytic mask, block/grid
  index maps, B-divisibility guard, `t_offset` chaining, Mode B env-id
  construction.
- The uint32→float conversion (`h >> 8` × 2⁻²⁴): bias-free, no modulo
  bias, cannot produce 1.0. Rank-pick samplers exactly uniform.
- **Benchmark timing discipline in every harness file**: all timed calls
  wrapped in `block_until_ready`, compile/warmup excluded symmetrically
  on both sides, interleaved-ratio methodology applied identically,
  `sweep_stats.py` aggregation correct (fresh process per run,
  within-run ratios).
- sokoban step logic (bounds/wall/push gating, rewards, time limit) and
  its exact-equality replay gate; Walker alias-table construction;
  `test_distributions.py` assertions are real (can fail).

None of the findings below invalidates a published number. They are:
edge-case math in the newest module, parity gates that claim more than
they test, performance left on the table, and release friction.

---

## C-series — correctness (Reviewer 1)

### C1. `geometric_tries` returns the *opposite tail* for u = 1.0 or tiny p — MEDIUM — FIXED (4a46326)
`djinnax/distributions.py:43-44`

`log1p(-1.0) = -inf` (or a ratio exceeding 2³¹ for p ≲ 1e-8) hits
`astype(int32)` — an undefined float→int conversion that typically
produces INT32_MIN — and the `jnp.maximum(n, 1)` guard then launders
that garbage into **n = 1**. A process that should take ~10⁹ tries
reports one try: no NaN, no error, silently the wrong tail.

Not live today (our counter-hash uniforms cannot produce 1.0, per
Reviewer 1's clearance above), but the function's contract is "a
uniform", and `jax.random.uniform` is `[0, 1)` while other sources may
be inclusive. A caller-contract trap in a module whose whole point is
being safe to drop into any env.

**Fix:** clamp `u` to `[0, 1 - 2^-24]` inside the function, and clamp
the float ratio before the int cast.
**Gate:** unit test feeding `u ∈ {0.0, 1.0, 1-2^-24}` and
`p ∈ {1e-9, 0.3, 1.0}`; assert n finite, ≥ 1, and monotone in u.

### C2. `conditional_categorical` can select a disallowed category at u → 1 — MEDIUM — FIXED (4a46326)
`djinnax/distributions.py:61-64`

The target is `u * w.sum()` but membership is tested against a
separately computed `cumsum`. The two reductions are not guaranteed
bit-identical (pairwise vs sequential summation), so for u close to 1
the target can exceed `c[..., -1]`, making `c > target` all-False and
`argmax` return index 0 — **even when category 0 is disallowed**.

**Fix:** normalize with the cumsum's own last element:
`total = c[..., -1:]`. Then `target < total` for all u < 1 by
construction, in the same float stream.
**Gate:** unit test with `allowed[0] = False`, u swept through
`1 - k·2^-24`; assert the returned index is always allowed. Existing
distribution-parity tests re-run.

### C3. TTT parity gate never tests the paths its docstring claims parity for — MEDIUM — FIXED (d0d826b) — **gate found a real divergence**: pgx applies the illegal placement and rewards the opponent +1; engine corrected, 60+60 off-path cases now identical
`checks/check_parity.py:49-70` vs `djinnax/ttt.py:98-141`

`check_ttt` always plays a legal action and stops at the first terminal
state, so the illegal-action-loss path and the step-past-terminated
freeze path are never compared against pgx. If pgx awards the opponent
+1 on an illegal move where we award 0, every gate still passes.

**Fix:** add replay cases that deliberately step an occupied cell and
step once past termination, asserting rewards/state against pgx.
**Gate:** the new cases themselves (must be shown to fail when our
reward table is perturbed, then pass unperturbed).

### C4. No gate on episode-*start* parity with the 2048 reference — MEDIUM (PLAUSIBLE) — CLOSED CLEAN (d0d826b): 512-seed gate — reference resets with exactly 1 tile, value distribution matches within 0.05; no bench impact
`djinnax/game2048.py:150-161`, `checks/check_parity.py`

Our `init` and in-step reset spawn exactly one tile. The reference's
reset semantics (tile count / value distribution) are asserted nowhere —
parity covers moves and in-play spawns only. If the reference resets
with, say, two tiles, board density (and therefore episode length)
differs between the compared engines, quietly biasing the head-to-head
ratio in an unknown direction.

**Fix:** one replay assert comparing a reference `reset` board
population (tile count and value histogram over N seeds) against
`Djinn2048.init`.
**Gate:** the assert itself; if it *fails*, the head-to-head 2048 rows
must be re-run after aligning reset semantics, and the change disclosed
in RESULTS_HISTORY.

### C5. LUT saturates merges at exponent 15; the variant-equivalence gate never reaches the divergent region — LOW — FIXED (d0d826b): divergence pinned by test, bit-identical claim scoped in docstring
`djinnax/game2048_lut.py:34-35` vs `djinnax/game2048.py:57-59`

The v3 LUT caps merged exponents at 15 while the branchless engine and
megakernel produce 16, so "bit-identical variants" is only true on
boards that never contain a mergeable 15-pair — and `check_2048_moves`
draws exponents 0-7, so the divergent region is untested everywhere.
(The megakernel's "all-15 merge storm" adversarial board exercises the
non-LUT pair only, which is why everything passes.) Exponent 15 = tile
32768, unreachable in normal play from 2/4 spawns on a 4×4 board before
the board locks, so this is a latent trap, not a live bug.

**Fix:** exponent 16 cannot be represented in the LUT's 4-bit-per-cell
row code, so the saturation is inherent to the representation. Either
document it as a stated LUT-engine limit and scope the "bit-identical"
claim to boards without a mergeable 15-pair, or saturate the branchless
engine identically under a flag so the claim is unconditional. Decision
recorded when worked.
**Gate:** a test constructing a 15-pair board and asserting the
*documented* behavior for each engine, so the divergence is pinned, not
latent.

### C6. Batch-wide RNG correlation at one unreachable Δt — LOW — FIXED (996dbf4, comment)
`djinnax/megakernel_rng.py:60-67`

The pre-hash state is 32 bits with the env term XORed linearly, so
(a) distinct (env, t) pairs birthday-collide (~2¹¹ duplicate uniforms
per salt per 2²² draws at B = 65536), and (b) for each salt pair there
is exactly one Δt at which the entire batch's uniforms for salt j at
step t equal salt j′ at t+Δt — batch-wide, not per-env. Both far outside
reachable step ranges and statistically invisible at tested scales;
acceptable for a game RNG, but the failure mode is correlated across the
whole batch, which deserves a comment where the hash is defined.

**Fix:** documentation comment stating the structure and its bounds.
**Gate:** none (no behavior change); RNG batteries re-run as usual.

### C7. `conditional_categorical` all-zero-weight rows undocumented — LOW — FIXED (4a46326)
`djinnax/distributions.py:59-64`

Rows where every allowed category has zero weight fall through to
index 0; the docstring's "nothing allowed" caveat doesn't cover this
case. **Fix:** extend the docstring (or guard on `total > 0`) alongside
C2. **Gate:** covered by C2's test.

### C8. pgx TTT bench side may be charged O(B) RNG work the djinn side doesn't pay — LOW (PLAUSIBLE) — NO CHANGE NEEDED: compiled HLO of the bench's pgx step contains zero RNG ops (unused split is dead-code-eliminated)
`benchmarks/bench_head_to_head.py:76`

The reference side derives B split keys per step that a deterministic
TTT step almost certainly never consumes. If unused, XLA dead-code
eliminates the split (no cost); if the reference folds them in
unconditionally, it is charged extra RNG work per step. Note the
direction of this error is **in the reference's favor being overstated
against us is impossible — the risk is the opposite**: we could be
*flattering ourselves*. Must be settled, not assumed.

**Fix/Gate:** profile diff (or HLO dump) of the reference step with and
without the split; if the split is not DCE'd, restructure key handling
symmetrically and re-run the ttt rows.

### C9. NullEnv floor-probe accumulates into int8 with wraparound — LOW — FIXED (996dbf4; floor continuity rides the next GPU sweep)
`djinnax/runtime.py:108`

Harmless (the floor bench makes no correctness claim), but the "touch
state" work being measured includes overflow wrapping rather than
representative arithmetic. **Fix:** widen to int32 or mask; note in
floor_bench that absolute floor numbers before/after are not comparable.
**Gate:** floor_bench re-run for continuity.

---

## P-series — performance (Reviewer 2; every item gated by parity + interleaved pairwise A/B, medians [min..max], per LEARNINGS §3)

### P1. Megakernel computes all four direction-move networks every step — HIGH (PLAUSIBLE) — CONFIRMED + ADOPTED
n=5 fresh-process interleaved sweep (quiet GPU, data/p1_orient_ab.jsonl):
oriented vs all-moves **1.45× [1.36..1.47] @B=1024, 1.41× [1.34..1.46]
@8192, 1.57× [1.51..1.59] @65536** — every run-median above 1.33.
Bit-identical outputs (variant parity asserted in-process before every
timing; full 8-test megakernel battery green; permanent CPU equivalence
test added). Old variant kept as `step_lanes_allmoves` for the receipt.
Official README headline refresh deferred to ONE end-of-Batch-C sweep.
`djinnax/megakernel.py:153-158`

Each step runs the full move network (6 bubble passes + 3 merges over
16 lanes) **four times** — once per direction — plus four mask networks,
then selects one result by action. But direction is a *static lane
permutation*: select-lanes-by-action (≈48 wheres) → **one** move network
→ inverse permutation computes the same thing with 5 networks' worth of
ALU instead of 8 (~35-40% of kernel compute). This is exactly the
orient-select idiom the XLA engine (`game2048.py:172-182`) and
WRITING_FAST_ENVS §2 already prescribe; MEGAKERNEL_PLAN.md asserted the
4-way compute was "free in registers", but only the *wiring* is free —
the four executions are not.

**Risk:** the permutation must also apply to the can-move mask select;
register pressure may change occupancy at BLOCK=128.
**Gate:** full `check_megakernel.py` battery (bit parity is the chain
anchor), then interleaved A/B old-kernel vs new-kernel at
B ∈ {1024, 8192, 65536}, n=5 fresh processes. Headline numbers only
update if the sweep clears the interval.

### P2. In-step reset spawn runs a 16-way categorical where the answer is `randint(0,16)` — MEDIUM — OPEN
`djinnax/game2048.py:196-203`

The reset template spawns onto an all-empty board, where the masked
categorical (16 Gumbel exponentials per env) is exactly uniform over 16
cells. `jax.random.randint(k, (B,), 0, 16)` + the existing one-hot write
is distribution-identical and removes a categorical + logits
construction from every step for every env. Spawn sites are
distribution-gated, not bit-gated, so the swap is legal.

**Gate:** `check_2048_spawn` + variant-equivalence note (bit streams
change; distribution parity is the contract) + interleaved A/B.

### P3. Mid-game `_spawn` uses masked categorical; house doctrine is rank-pick — MEDIUM — OPEN
`djinnax/game2048.py:114-121`

O(B·16) Gumbels vs one uniform + cumsum. The megakernel's
`_spawn_lanes` already does rank-pick; runtime.py's rank-pick sampler is
the adopted recommendation — the XLA engine predates it. Bonus finding:
WRITING_FAST_ENVS §4 codifies the categorical form for spawns two lines
below its own rank-pick bullet; the doctrine contradiction gets fixed
whichever way the measurement goes.

**Gate:** distribution tests (spawn histogram) + interleaved A/B at
B ∈ {1024, 8192, 65536}.

### P4. Oriented move materializes two (4,B,4,4) stacks per step — MEDIUM (PLAUSIBLE) — OPEN
`djinnax/game2048.py:173-182`

`jnp.stack([_orient(board, a) for a in range(4)])` twice per step, then
where+sum-reduce (with an int8→int32→int8 round-trip via `jnp.sum`). A
precomputed (4,16) cell-permutation table gathered by action
(`take_along_axis` on the flat board) does the same select as one gather
with no 4× intermediate. XLA may already fuse the stack/where/sum —
hence PLAUSIBLE; the measurement decides.

**Gate:** move-parity check + interleaved A/B at B ∈ {8192, 65536}
where the intermediate is 4-8 MB.

### P5. Missing `CHANGED_LUT` makes the every-step can-move check unpack full boards — MEDIUM — OPEN
`djinnax/game2048.py:98` + `djinnax/game2048_lut.py`

The per-step legality mask runs `move_all_directions` on the post-move
board — 4 of the 5 move passes per step — and determines "changed" by
unpacking LUT codes to (B,4,4) and comparing all 16 cells. A third
65,536-entry boolean LUT (`moved[code] != code`) reduces each
direction's check to one (B,4) gather + row-OR. The game2048_lut
docstring *already claims* this LUT exists ("changed 8 KB (bool)") —
building it makes the docstring true (see R8).

**Gate:** `check_2048_moves` covers `can` exactly; then interleaved A/B
on the LUT engine rows.

### P6. Sokoban recounts boxes-on-target twice per step; the count is carryable state — MEDIUM — OPEN
`djinnax/sokoban.py:108,117`

`n_before` is exactly the previous step's `n_after` for non-reset envs,
and provably 0 at reset (levels place boxes off-target by construction).
Carrying `n_on_target` (B,) in SokoState halves the (B,100) reductions
to one per step.

**Gate:** `check_sokoban` exact replay (reward stream must be
bit-identical) + interleaved A/B on soko rows.

### P7. Float `2.0**(a+1)` for merge reward where the integer shift form exists 100 lines away — LOW — OPEN
`djinnax/game2048.py:59`

The megakernel already uses `(1 << (x+1)).astype(f32)` — cheaper, exact,
and immune to the pow-precision issue pallas_lab.py documents.
**Gate:** move-parity (rewards asserted there) + it rides along with
whichever P2-P4 sweep runs first.

### P8. Five double-fmix32 hashes per env-step may be over-provisioned — LOW (PLAUSIBLE) — OPEN
`djinnax/megakernel_rng.py:60-73`

A single-fmix variant, or one base hash per (env, t) with a cheap
per-salt finalizer, would roughly halve RNG ALU in an ALU-bound kernel.
The existing RNG batteries (`check_rng_quality`, deep-correlation tests)
are precisely the gate: if single-fmix fails them, double-fmix stays and
this closes as WON'T FIX with the battery as the receipt.

**Gate:** RNG batteries first (kill criterion), then bit-parity is
*expected to change* (different stream) → distribution-level megakernel
parity + interleaved A/B.

### P9. Headline harness runs protocol v1 while the repo recommends v2 — LOW — OPEN
`benchmarks/bench_head_to_head.py:64-222`

Per-step `fold_in`, masked categorical, no donation — the protocol the
repo's own runtime.py supersedes. Symmetric-protocol fairness justifies
it for the *ratios*, but the absolute "env-steps/s" headline is produced
under the slower protocol and the docs don't say so.

**Fix:** state the v1-for-symmetry choice in the bench docstring and
README, and/or add a v2 djinn row so both absolutes are visible.
**Gate:** doc change; if a v2 row is added, standard sweep.

### P10. Sokoban `step_count` is int32 against house dtype doctrine; grid-form vs entity-list doc tension — LOW — OPEN
`djinnax/sokoban.py:46,73`

TIME_LIMIT=120 fits int8 (int16 per doctrine). Separately,
PORTING_PLAYBOOK teaches the entity-list lesson while the shipped
sokoban is the grid form — either note the grid form was kept for
reference work-parity, or land the entity-list rewrite (already in the
open queue) and let the playbook point at it.

**Gate:** dtype: replay parity + rides along a soko sweep. Doc note:
none.

### P11. Cleared areas (Reviewer 2)
ttt.py, runtime.py, distributions.py, and megakernel I/O design (int8 at
the boundary, int32 lanes in-register, Mode B killing the uniforms
buffer): no further redundant per-step computation, host syncs, or
dtype waste found. No stray `block_until_ready` outside benchmark
timing loops.

---

## R-series — release readiness (Reviewer 2)

### R1. Tracked `reference-engines` symlink leaks a local absolute path into every clone — HIGH — FIXED (c81370b)
Repo root. Verified: `git ls-files` includes `reference-engines`.

The symlink target is an absolute path containing a local username.
Every public clone gets a broken symlink, and HOW_TO_RUN's first setup
command (`mkdir -p reference-engines && cd reference-engines`) fails
against it with "File exists". The `.gitignore` pattern
`reference-engines/` ignores a *directory*, not a symlink — which is how
it got committed.

**Fix:** `git rm --cached reference-engines`; change the gitignore
pattern to `reference-engines` (no slash) so the local symlink stays
usable but untracked.
**Gate:** `git ls-files | grep reference` empty; fresh-clone simulation
runs HOW_TO_RUN's setup block cleanly.

### R2. `pip install djinnax` is broken: package module imports `benchmarks.*` at module scope — HIGH — FIXED (2f5293f; clean-venv wheel gate passed)
`djinnax/megakernel_rng.py:39`; `pyproject.toml` packages only
`["djinnax"]`. Verified.

Any install outside the repo makes `import djinnax.megakernel_rng`
raise ModuleNotFoundError, and library import drags in an argparse
benchmark script. CI can't catch it: it installs `-e` from the repo
root, where `benchmarks/` happens to be importable.

**Fix:** move the bench-only use into a lazy import inside `bench()`.
**Gate:** build a wheel, install into a clean venv, `import
djinnax.megakernel_rng` — wired into CI as a smoke test (see R5).

### R3. `refs.py` stubs `tqdm`/`huggingface_hub`/`esquilax` process-wide even when genuinely installed, as a library-import side effect — HIGH — FIXED (2f5293f; stub-only-if-missing + side-effect imports dropped)
`djinnax/refs.py:38-46`; imported at module scope by `megakernel.py:31`,
`megakernel_rng.py:27`, `soko_ref_generator.py:7`.

`_StubFinder` sits at `sys.meta_path[0]` and unconditionally shadows
those packages — a downstream program that imports the megakernel gets a
fake tqdm for the whole process. refs.py also mutates `sys.path` on
import.

**Fix:** (a) stub only if `importlib.util.find_spec` fails; (b) remove
the `import djinnax.refs` side effect from megakernel*.py (they never
use the refs — verify and drop); keep it in checks/ and
soko_ref_generator (which genuinely need reference paths).
**Gate:** clean-venv import test (with real tqdm installed, `import
djinnax.megakernel; import tqdm; tqdm.tqdm` is the real one); full CPU
suite.

### R4. Package exports nothing; no usage snippet; step signatures undocumented — MEDIUM — FIXED (c5092cd; README snippet executed verbatim in tests)
`djinnax/__init__.py`

`import djinnax; djinnax.Djinn2048` fails; step signatures diverge
across envs (by design — documented nowhere). First-10-minutes friction.

**Fix:** re-export engine classes, `run_megakernel_rng`, runtime
helpers; add a 5-line init/step example to the README; one sentence on
the per-env signature convention.
**Gate:** the example in the README is executed verbatim in a test.

### R5. CI never runs on the development branch, pins no jax, and can't catch packaging breaks — MEDIUM — FIXED (wheel-import smoke job added)
`.github/workflows/ci.yml:3`

**Fix:** add `Development` to push branches; floor-pin jax to the tested
line; add a build-wheel → clean-install → import smoke job (catches R2
class permanently).
**Gate:** CI green on Development with the new jobs.

### R6. Docs mandate a test-gate table whose named tests don't exist in this repo — MEDIUM — FIXED (e248552; training-tier rows scoped explicitly)
`WRITING_FAST_ENVS.md:179-183` (echoed in AGENTS.md/CLAUDE.md,
PORTING_PLAYBOOK.md)

Three of the eight "no env ships without these" gates
(mask/apply-consistency, conformance driver, serialization round-trip)
name files that live in a downstream training setup, not here. A public
reader following the law finds the law's own repo out of compliance.

**Fix:** annotate those rows as applying to training-integration repos,
with the in-repo equivalents named for the rest.
**Gate:** doc-only; re-read pass.

### R7. Informal/internal references survive in live public docs — MEDIUM — FIXED (grep gate: only disclaimed historical hits remain)
`WRITING_FAST_ENVS.md:79,144` (unexplained internal project name),
`CLAUDE.md:1`/`AGENTS.md:1` (pre-rename "engine-bench" titles),
`tests/conftest.py:1`, `benchmarks/bench_head_to_head.py:18` (stale
`engine-bench/` usage path — copy-paste wrong, not just tone),
`LEARNINGS.md:5` (consulting reference), "this box/this host" phrasing.

**Fix:** rename titles to djinnax; replace internal project references
with neutral descriptions ("a production training env we measured");
fix the stale usage path; keep RESULTS_HISTORY's "imported log"
disclaimer pattern for anything historical.
**Gate:** grep sweep for the pre-rename name and internal identifiers
returns only intentional historical-context hits.

### R8. game2048_lut docstring claims a third LUT that doesn't exist — LOW — OPEN (resolves with P5)
`djinnax/game2048_lut.py:15-16` — resolves with P5 (build it), which
makes the docstring true. If P5 measures null, fix the docstring
instead.

### R9. Stale superseded numbers in two files — LOW — FIXED
`checks/check_megakernel.py:7` ("275×" vs official 237×),
`LEARNINGS.md:280` + `docs/HARDENING_ROUND2_PLAN.md:10` ("16/16 GPU,
11+5 CPU" vs the current 19-test suite). **Fix:** update or tag as
historical. **Gate:** grep sweep.

### R10. pyproject metadata: marketing in description, version floors contradict HOW_TO_RUN, no repo URL/classifiers — LOW — FIXED (c5092cd; wheel metadata inspected)
**Fix:** neutral description; align `requires-python`/jax floor with
what is actually tested (HOW_TO_RUN says Python 3.12 / JAX 0.10.x);
add Repository URL + classifiers + license-files.
**Gate:** `pip install` of the built wheel shows clean metadata.

### R11. `.gitignore` missing `.pytest_cache/`, `.benchmarks/`; empty `.benchmarks/` in tree — LOW — FIXED (c81370b)
**Fix:** add both, delete the empty dir. **Gate:** `git status` clean
after a test run.

### R12. Cleared areas (Reviewer 2)
LICENSE/NOTICE/pyproject license field mutually consistent; NOTICE
correctly scopes the reference projects; README test counts match the
suite; HOW_TO_RUN script table matches existing files.

---

## Work plan

**Batch A — mechanical, land first (no measurement questions):**
R1, R2, R3, C1, C2, C7, R7, R4, R5, R9, R10, R11, C6, R6.
Each with its stated gate; one commit per logical fix; full CPU suite +
GPU suite once at batch end.

**Batch B — gate strengthening (tests that could *fail* and force
follow-up):** C3, C4, C5, C8, C9. If C4 or C8 fail, affected bench rows
are re-run and the change disclosed before anything else lands.

**Batch C — performance experiments, one at a time, each behind parity
gate + interleaved pairwise A/B (n=5 fresh processes, medians
[min..max], code frozen during sweeps, quiet GPU disclosed):**
order P1 (headline-relevant), P2, P3, P5 (+R8), P4, P6, P7 (rides
along), P10, P8, P9. Null results get recorded in LEARNINGS's ledger,
not silently dropped.

Headline table only updates from Batch C results that clear their
intervals; anything within noise stays at current official numbers.
