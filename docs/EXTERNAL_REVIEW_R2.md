# External Review R2 — findings and remediation plan

Round 2 of external review, run 2026-07-21 on the Development branch at
`1d8fe73` (immediately after the R1 remediation merged to main). Two
independent reviewers, both required to read `EXTERNAL_REVIEW_R1.md`
first and forbidden from resurfacing its 31 closed findings — the
mission was what round 1 missed. Reviewer 1 was scoped to remaining
**engine** opportunities and residual correctness risk; Reviewer 2 did
the deep **documentation** pass round 1 never performed, checked against
current source. Same rules as R1: static analysis only, this repository
only, findings reported without fixes; every item below carries its own
gate and no fix lands without its gate passing.

Status legend: `OPEN` → `FIXED (commit)` / `CONFIRMED + ADOPTED` /
`NULL` / `KILLED` / `WON'T FIX (reason)` / `NO CHANGE NEEDED (reason)`.

---

## E-series — engine (Reviewer 1)

### E1. Megakernel mask pass runs 4 full move networks; an analytic pair-condition probe is bit-equivalent at ~10% of the cost — HIGH — OPEN
`djinnax/megakernel.py` step 4 (`new_mask` via `_move_dir` ×4 on the
post-spawn board) and `_initial_mask`.

Post-P1, the mask pass is 4 of the 5 remaining networks per step
(~80% of network ALU). But "direction d changes the board" is
analytic: **∃ adjacent pair (prev, cur) in push order with
`cur != 0 & (prev == 0 | prev == cur)`** — a zero-behind-a-tile forces
a compaction shift; an equal adjacent pair forces (or is preceded by) a
merge; conversely a row with no such pair is compacted with no merges,
hence fixed. Hand-verified on the standard cases (gap-shift, merge,
separated-equals, compacted-no-merge) before this ledger was written.
~150 ops (with zero-flags and adjacency-equality terms shared across
opposite directions) vs ~1,400 for four bubble+merge networks. Note
this is NOT the killed P5: P5 *added* an L2-competing table gather to
the LUT engine; this is pure in-register ALU deletion, no tables.

**Critical gate note (Reviewer 1):** megakernel-vs-XLA parity CANNOT
catch a wrong probe — both sides share `step_lanes`. Binding gates:
(a) probe ≡ `move_all_directions(...)[2]` on random + adversarial
boards (the jumanji-anchored chain); (b) the permanent CPU equivalence
test vs `step_lanes_allmoves`, whose mask path still uses `_move_dir`;
(c) full `check_megakernel.py` battery; then n=5 interleaved A/B at
B ∈ {1024, 8192, 65536}. Calibrated expectation if ALU-bound:
**1.5-2.5× on the megakernel rows** on top of P1.

Side option, same experiment: with the mask this cheap, the 4 carried
mask lanes in the fori_loop carry can be dropped and recomputed at step
entry — 4 registers vs ~150 ops, occupancy-dependent, PLAUSIBLE;
measure both variants.

### E2. Same disease in the XLA engines: the default `can_move` runs 4 full move passes for a mask the analytic probe derives in ~10 elementwise ops/direction — HIGH — OPEN
`djinnax/game2048.py` (`can_move_fn` default via `move_all_directions`,
consumed in `step`; the step docstring itself prices the step at
"1 move + 4 next-mask" passes).

Each mask pass builds the (B,4,4,4) one-hot compaction selects twice
plus the merge chain, purely to compare `moved != board`. The E1 pair
condition applied to the raw (B,4,4) board with shifted compares needs
no compaction, no scatter, and — unlike killed P5 — **no gather**, so
the mechanism P5 died on does not apply. Drops into the existing
`can_move_fn` seam, so the LUT engine (the production row) gets it too,
where it deletes the pack+gather+unpack passes currently paid for the
mask.

**Gates:** `check_2048_moves` asserts `can` per board per direction
against the reference — run with the probe plugged; plus
`check_2048_variants_step_equivalence`; then interleaved A/B on the
2048/djinn and 2048/djinn-lut rows. Magnitude PLAUSIBLE (XLA may
already share work between the four passes; P5's lesson is measure
before believing op counts), but strictly-less-work with no new memory
traffic.

### E3. Sokoban head-to-head may charge obs/reward construction to the reference only — MEDIUM (PLAUSIBLE) — OPEN
`benchmarks/bench_head_to_head.py`: the djinn sokoban runner discards
obs/reward/extras inside the traced scan body (`state, _, _, _ =`), so
XLA dead-codes the (B,10,10,2) stack and reward math; the jumanji
runner carries `ts` (observation + reward) in the scan carry, which the
while-loop simplifier cannot strip. sokoban.py explicitly claims
"per-step work is comparable" — the code builds it, the harness
un-builds it on one side. Error direction flatters the djinn soko rows.

**Gate (same method as R1's C8):** dump compiled HLO of both runners;
grep for the (B,10,10,2) concatenate. If the asymmetry is real, either
carry a live-out of obs+reward on the djinn side, or disclose next to
the soko rows — and re-run those rows. (XLA could in principle strip
jumanji's side too since only the final `ts` returns; the dump settles
it.)

### E4. Mode A `run_megakernel` still silently drops the batch tail on B % BLOCK ≠ 0 — MEDIUM — OPEN
`djinnax/megakernel.py` (`grid=(Bn // BLOCK,)`, no guard) vs Mode B's
loud ValueError landed in R1-era hardening. `check_b_divisibility_guard`
tests only Mode B. A B=1000 Mode-A parity run today compares 896 envs
and prints "parity OK". Also: `run_megakernel` silently assumes
`uniforms.shape[0] == N_STEPS` — assert it.

**Gate:** extend the guard test to both entry points + a
shape-mismatched uniforms buffer.

### E5. `_row_move_4` bubble network: 18 comparators per group where 9 are provably sufficient, bit-identically — MEDIUM (PLAUSIBLE) — OPEN
`djinnax/megakernel.py` (`_bubble`/`_row_move_4`).

(a) Pre-merge compaction: 3 sequential passes of (a,b)(b,c)(c,d) = 9
bubbles; the odd-even transposition order — (a,b)(c,d), (b,c),
(a,b)(c,d), (b,c) — compacts any 4-lane group in 6 (compaction output
unique ⇒ bit parity automatic). (b) Post-merge: starting from a
compacted prefix, the sequential merge scan can only zero lanes in
{}, {b}, {c}, {d}, {b,d} (merge(b,c) can't fire after merge(a,b);
merge(c,d) can't fire after merge(b,c)), so survivors displace ≤1 and
ONE (a,b)(b,c)(c,d) pass finishes — 3, not 9. Net 18 → 9 bubbles per
group ×4 groups per network; compounds with E1 (shrinking denominator).

**Gates:** `check_move_chain_link` (bit-anchored to the reference) +
full battery + interleaved A/B. Expect P4-sized standalone; larger
post-E1.

### E6. Two bit-identical micro-simplifications in `step_lanes` — LOW — OPEN
(a) The 4-iteration `was_legal` loop is identically `n_legal > 0`
(rank-pick selects a legal hit whenever one exists): 8 ops → 1.
(b) `_spawn_lanes` re-ANDs `enabled & (n_empty > 0)` into all 16
lane-hit terms; a sentinel rank (`r_gated = where(enabled, r, -1)`)
removes ~32 ops — and `n_empty > 0` is implied by `enabled` (a legal
move leaves ≥1 empty: moves add no tiles, and a full-board legal move
must merge). Rides along whichever kernel sweep runs first;
individually unmeasurable.

### E7. Reset uniforms can come from spare bits of the spawn hashes (Mode A buffer −40%; Mode B expected null) — LOW — OPEN
`u_rcell` needs 4 bits, `u_rval` one threshold; `hash_uniform` discards
the low 8 bits of each 32-bit hash, so both reset draws extract from
the low bytes of the `u_cell`/`u_val` hashes: N_UNI 5 → 3. Mode B
throughput expected ~null per P8's calibration; the real effect is
Mode A's streamed buffer −40% (80 → 48 MB/launch at B=65536; 1.31 →
0.79 GB at B=1M). **Warning (Reviewer 1):** do NOT reuse `u_cell`
itself for `rcell` — `done` is a function of `u_cell` via the
post-spawn board, so conditioning on reset biases the reused draw, and
the batteries plausibly wouldn't catch it. **Gates:** RNG batteries +
distribution-level parity (stream changes allowed for spawn/reset
sites).

### E8. B=1M readiness notes (roadmap target; none blocking Mode B) — LOW — OPEN
(a) Mode A's (64, 5, B) f32 buffer is 1.31 GB at B=2^20 — tight under
the 50%-VRAM cap before the kernel is; E7 buys 40% back. (b) The C6
structural-caveat numbers in `hash_uniform`'s docstring are stated for
B=65536; duplicate-pair density grows ~×256 at B=2^20 — restate for
the 1M target or promote the 64-bit pre-hash. (c) `check_parity_sweep`
tops out at B=1024; add a large-B parity point when 1M work starts.
(d) Generality: the kernel zeroes score on in-step reset and outputs
only final score/done — mid-launch episode returns are destroyed;
training integration needs per-step or per-episode-flush HBM outputs,
which changes the memory-traffic profile the headline is measured
under. Plan-level note.

### E9. Cleared areas (Reviewer 1's negative space)
ttt.py / runtime.py / distributions.py: nothing beyond R1. Sokoban step
body beyond E3: batched-scatter idea predicted null by P6's lesson —
not worth a sweep. Timing/aggregation harness beyond E3: clean.
fori_loop carry dtypes: all necessary.

---

## D-series — documentation (Reviewer 2)

### D1. Rung 4 has no authoring recipe; the flagship technique lives in scattered comments — HIGH — OPEN
WRITING_FAST_ENVS gives rungs 1-3 inline code recipes; rung 4 is four
lines pointing at a lab script and a log, and the canonical-examples
list omits megakernel.py. Missing from the guide (each exists only in
code comments / MEGAKERNEL_PLAN / LEARNINGS): SoA lane design + static
lane-rewiring moves; the same-function bit-parity workflow ("write
step_lanes as a pure lane function FIRST, gate under scan, then wrap in
pallas_call"); the counter-RNG pattern (state-free hash, one salt per
site, t_offset chaining, sizing the per-step uniform registry); the
chunking contract (score carried, mask recomputed at entry — valid
because analytic ≡ computed mask); B % BLOCK; Triton-on-sm_89 rules
(no slice/.at in-kernel, integer shifts over float pow); the
"captures constants" jnp-closure trap (np-scalar workaround); BlockSpec/
grid index-map authoring; int32-lanes-in-register vs int8-at-boundary
(which contradicts the "smallest dtype" doctrine unless explained).

**Fix:** a full rung-4 chapter in WRITING_FAST_ENVS at rungs-1-3 depth:
lane recipe, same-function workflow, counter-RNG registry, kernel
gotchas table; add megakernel.py to the canonical examples.
**Gate:** doc builds + the D9 agent-file rules reference it.

### D2. All three prescriptive docs still teach the pre-megakernel stop-rule the flagship result disproved — HIGH — OPEN
WRITING_FAST_ENVS rung-4 intro and "when they're equal, stop
optimizing"; LEARNINGS §2 ("2048-LUT hit TTT speed → Pallas not
warranted"); PORTING_PLAYBOOK ("Megakernel — ONLY when your env at huge
B is still slower than the null floor"). History: 2048-LUT WAS at the
trivial-game floor and the megakernel still won 5-7.6× — LEARNINGS §6
records the correction, §2 and both how-to docs never got it. A reader
applying the taught test stops at rung 3 and never builds the thing the
repo is known for.

**Fix:** rewrite the eligibility rule in all three places: at the
XLA/NullEnv floor + sequential steps per env + state fits in registers
⇒ rung 4 is exactly what applies (the floor rungs 1-3 converge to is
the launch/materialization floor rung 4 attacks). Add the worked
eligibility checklist (state ≤ register budget at BLOCK=128, branchless
step exists, RNG expressible as (env, t, salt, seed) draws, B multiple
of BLOCK, envs independent).

### D3. MEGAKERNEL_PLAN.md describes the superseded kernel and repeats the disproven "free in registers" claim — HIGH — OPEN
The README routes readers here as the design reference; the plan still
prescribes compute-all-4-directions ("each direction is just different
lane wiring, free in registers") — the exact assertion R1-P1 disproved
at 1.41-1.57×. No _PERM/_INV, no _make_step, no rng_rounds; status
numbers are pre-P1. A reader or agent implementing from the plan builds
the killed variant. **Fix:** dated P1 addendum + "historical plan —
current design is megakernel.py + the rung-4 chapter" banner (same
pattern as RESULTS_HISTORY).

### D4. Official megakernel numbers' provenance is wrong-by-omission — MEDIUM — OPEN
README and RESULTS_HISTORY present 98/213/237/91× as "the megakernel";
those were measured on what is now `step_lanes_allmoves`. The shipped
default is 1.41-1.57× faster; anyone reproducing per HOW_TO_RUN gets
systematically higher ratios with no explanation. **Fix:** one
provenance line under both tables ("measured on the pre-P1 all-moves
kernel; current default measured 1.41-1.57× over it —
data/p1_orient_ab.jsonl; official refresh pending a verified quiet
window").

### D5. No quantitative what-fits-a-megakernel guidance — MEDIUM — OPEN
The only proven register budget (2048: 16 int32 lanes + 4 bool mask +
f32 score + done at BLOCK=128) is stated nowhere as a budget; the
plan's probe stage (compile a 2-step loop at BLOCK=128 before building
anything) reads as history, not method. **Fix:** promote the probe into
PORTING_PLAYBOOK as a numbered pre-flight step, with 2048's inventory
as the known data point and sokoban named as a known doesn't-fit.

### D6. Killed/null results not surfaced where the idioms are taught — MEDIUM — OPEN
The rung-3 section and the anti-pattern row "gathering what a LUT could
store" actively point toward the CHANGED_LUT idea that measured
0.82-0.88×; the §2 orientation snippet still shows the superseded
stack/select form with no pointer to the adopted permutation-gather or
its null-at-small-B result; §4 omits the single-fmix null. **Fix:**
one-line "measured: … (data/*.jsonl)" cross-refs at each idiom site,
the pattern §4 already uses for rank-pick.

### D7. Test counts drifted again after the R1 fix — MEDIUM — OPEN
Actual suite: 28. Stale: README ("26 GPU / 21+5"), HOW_TO_RUN (fresh
clone counts + "7 reference-parity tests" + table row), LEARNINGS.
**Fix:** recount once, keep exact numbers in ONE canonical location and
point the others at pytest output so this stops recurring.

### D8. HOW_TO_RUN omits the entire A/B-receipt layer — MEDIUM — OPEN
Seven A/B benchmarks and the `data/` receipts directory are invisible in
the "every script, in order" doc and the README repo map —
spawn_collapse_ab is even cited as evidence by two docs the reader
can't route to. **Fix:** an "A/B receipts" table section + one line
documenting data/*.jsonl.

### D9. CLAUDE.md/AGENTS.md lack the rules an agent needs (~8 lines) — MEDIUM — OPEN
Missing: check LEARNINGS' confirmed/killed ledger before proposing perf
changes (nothing stops re-proposing CHANGED_LUT/single-fmix/carried
counts/unsafe_rbg); any change to megakernel_*/step_lanes runs the
check_megakernel battery; kernel hard rules (B % BLOCK, no slice/.at
in-kernel, np-scalars for kernel-closure constants, chunk coarsely —
2.2-3.9× tax); the house convention that superseded variants are KEPT
with a data/ receipt, not deleted; MEGAKERNEL_PLAN in the read order.
**Fix:** a "kernel work" bullet block + a "receipts" bullet.

### D10. The pedagogy thread breaks exactly at rung 4 — MEDIUM — OPEN (resolves with D1)
Path is clean through rung 3, then forks four ways (pallas_lab teaches
a superseded kernel; LEARNINGS §6 is a log; MEGAKERNEL_PLAN is stale;
source comments). None tells a NEW game's author how to derive their
_GROUPS/_PERM analogue, what replaces the 4-direction structure for
non-grid games, or how to size the uniform registry. **Fix:** D1's
chapter becomes the canonical entry; pallas_lab + MEGAKERNEL_PLAN get
"guided history, not the current design" banners.

### D11. §7 gate table has no rung-4 gate class — LOW — OPEN
Add a "kernel parity battery" row (same-function bit parity, chained
rollouts, determinism, adversarial boards) pointing at
check_megakernel.py / tests/test_megakernel.py.

### D12. Counter-hash vs threaded-keys doctrine boundary unstated — LOW — OPEN
AGENTS.md says "prefer counter-hash"; §4 mandates threaded keys; the
shipped XLA envs use keys. **Fix:** one sentence in §4: keys for XLA
envs; counter hash in-kernel or when state-free replay matters.

### D13. README repo map omits `data/` and `distributions.py` — LOW — OPEN
Two lines.

### D14. README's spread pointer is malformed — LOW — OPEN
"full intervals in `docs`/commit history" → point at
docs/RESULTS_HISTORY.md + data/sweep_official_v2.jsonl explicitly.

### D15. Sufficient areas (Reviewer 2)
Rungs 1-3 authoring depth is recipe-grade and matches the code; the
collapse/step-1.5 thread is the best-integrated in the repo;
measurement discipline is consistently stated and the A/B scripts
implement it; R1's doc-drift closures held — no regressions.

---

## Work plan

**Batch A — correctness/honesty first (actively misleading today):**
E4 (Mode A tail-drop guard + uniforms-shape assert), E3 (HLO dump →
fix-or-disclose, re-run soko rows if real), D2 (stop-rule rewrite ×3),
D4 (provenance lines), D3 (plan banner + addendum), D7 (counts), D14.

**Batch B — the analytic-probe experiments, one at a time, full gates +
n=5 interleaved sweeps (P5's lesson: op counts lie, measure):**
E2 first (cheaper, gates exist, touches production LUT row), then E1
(kernel; the binding gate is the probe-vs-move_all_directions chain
link, NOT kernel-vs-XLA parity), then E5 riding the same battery, E6
riding along, E7 (Mode A memory item). If E1/E2 confirm, the
quiet-window official sweep afterward captures everything at once.

**Batch C — the docs build:** D1 rung-4 chapter (canonical entry),
D2's eligibility checklist, D5 probe pre-flight, D6 receipts cross-refs,
D8 A/B table, D9 agent rules, D10 banners, D11, D12, D13.

**Deferred with owner sign-off:** E8 items land when B=1M work starts.

Headline table updates only from a quiet-window official sweep after
Batch B settles.

---

## Addendum — cross-model audit (2026-07-21)

Before remediation began, an independent audit ran with a **different
model family** (OpenAI Codex / GPT-Sol), blind to this ledger by
explicit instruction — see
`ChatGptSolAudits/INDEPENDENT_STATIC_AUDIT_2026-07-21.md`. Rationale:
all three Fable passes (author + two R2 reviewers) share priors; a
different family is adversarial in a way another same-family pass is
not. Every claim below was re-verified against source before triage.

**Convergent (validates this ledger):** SOL-03 ≡ E4, SOL-05(soko) ≡ E3,
SOL-13 ≡ D7, PERF-01 ≡ E1/E2. Two blind reviews from different model
families independently landed on the same headline items.

**New, confirmed, adopted into the plan:**

- **S1 (SOL-01, High)** — `sokoban.py` terminal rows return the
  pre-reset observation with the post-reset state; the jumanji
  `AutoResetWrapper` we benchmark against returns the RESET observation,
  and the parity gate skips terminal rows (`if not j_done`), hiding the
  divergence. **Decision (owner):** match the jumanji wrapper — return
  the reset observation on terminal rows; extend the parity gate across
  terminal rows (the gate exclusion is the bug's camouflage; removing it
  is the binding gate). Status: `OPEN`.
- **S2 (SOL-02, docs half)** — README claims all envs auto-reset; TTT
  freezes (correct: pgx parity), 2048/soko auto-reset (correct: jumanji
  parity). Behavior stands; the docs get an honest per-env contract
  table (TTT = pgx-style external reset; 2048/soko = jumanji-style
  autoreset with `terminated` as transition metadata). Status: `OPEN`.
- **S3 (SOL-04, High)** — official bench path fails open: per-engine
  exceptions print FAILED and exit 0; `sweep_stats` aggregates partial
  matrices; backend never asserted; `--out` relative paths resolve
  against different cwds in parent vs child. Fix: `--strict` default
  for sweep-invoked runs (assert GPU backend, fail on any engine
  failure, require complete (run, B, engine) matrix), record
  backend/device/jax-version/commit in every JSON row, resolve `--out`
  absolute; keep `--best-effort` for exploratory runs. Status: `OPEN`.
- **S4 (SOL-06)** — focused A/B scripts interleave but in fixed A-then-B
  order; counterbalance to ABBA before any Batch B sweep so all new
  sub-2× numbers use the stronger protocol. Status: `OPEN`.
- **S5 (SOL-08)** — `expected_tries` has the same unclamped
  `int32(ceil(1/p))` overflow R1-C1 fixed in `geometric_tries`; the
  sibling was missed. Clamp identically + edge tests. Status: `OPEN`.
- **S6 (SOL-09)** — `build_alias_table` accepts empty/negative/NaN/
  all-zero weights silently (all-zero → NaN tables of plausible shape).
  Host-side, validation is free. Status: `OPEN`.
- **S7 (SOL-05, mega labeling)** — `--rng` is recorded in the
  megakernel's JSON rows though it does not affect the kernel RNG;
  also verify the `_fresh_inputs` spawn-distribution claim (all tiles
  exponent 1 vs engine's 0.9/0.1). Fix the receipt labeling; disclose
  or align the init. Status: `OPEN`.
- **S8 (SOL-10, Low)** — counter-RNG env-ids are launch-local; no
  global offset for multi-device. Single-GPU by project policy —
  docstring caveat + optional `env_id_base` if free. Status: `OPEN`.
- **S9 (SOL-12, Low)** — `import djinnax` eagerly builds the 65k LUT +
  256 levels + device arrays. Measure import cost first; lazify only if
  it matters. Status: `OPEN`.
- **S10 (SOL-07 framing)** — exp-15 LUT saturation is pinned by
  `check_2048_exp15_divergence`, but "rare ≠ unreachable": upgrade the
  disclosure to state the variant is bounded and what happens past the
  bound. Docs item. Status: `OPEN`.
- **S11 (SOL-11 framing)** — soko fixtures are unvalidated throughput
  levels by design; document the level distribution as part of the env
  spec so a training user isn't ambushed. Docs item. Status: `OPEN`.
- **P-S1 (PERF-02)** — batch-gated soko reset:
  `lax.cond(jnp.any(done), reset, no_reset)` — sanctioned batch-level
  cond; cheap A/B alongside Batch B. Hypothesis only. Status: `OPEN`.
- **P-S2 (PERF-03)** — entity-list Sokoban core (agent + 4 box coords
  vs two 100-cell grids). **Decision (owner): parked on the deferred
  list** — new workload needs its own end-to-end gate; measure the
  cheap items first. Status: `DEFERRED`.

**Batch A closeout (2026-07-21):**

| item | status | commit |
|---|---|---|
| E4 / SOL-03 | `FIXED` — Mode A + both pallas_lab helpers guarded; uniforms shape validated; guard test covers both entry points, now CPU-runnable | `e3f5806` |
| S1 (SOL-01) | `FIXED` — terminal rows return the reset observation (AutoResetWrapper convention); new pure-djinn contract test forces both termination paths; replay gate asserts coherence at terminal rows | `f21bca4` |
| S5 (SOL-08) | `FIXED` — expected_tries clamped like geometric_tries; edge tests to subnormal p | `9fbc16e` |
| S6 (SOL-09) | `FIXED` — alias-table input validation + post-build invariants; invalid-input tests | `9fbc16e` |
| E3 / SOL-05(soko) | `CONFIRMED + FIXED` — HLO showed 0 obs-shape ops (ours) vs 20 (jumanji); runner now carries outputs, HLO symmetric 20/20; honest cost 1.63× measured ABBA (`data/e3_soko_dce_ab.jsonl`); README carries the inflation caveat until the next official sweep | `51576e2` |
| S3 (SOL-04) | `FIXED` — strict/fail-closed default (GPU assert, engine failures exit nonzero), complete-matrix requirement in sweep_stats, absolute --out, provenance in every JSON row; pinned by tests/test_bench_strict.py | `0ba3351` |
| S7 (SOL-05 labeling) | `FIXED` — mega row labeled rng="counter-hash"; end-to-end vs matched-protocol disclosed in bench docstring incl. _fresh_inputs transient (claim verified: single exp-1 tile) | `0ba3351` |
| D2 + S2 | `FIXED` — stop-before-megakernel rule rewritten as disproven in LEARNINGS §2, WRITING_FAST_ENVS rung 4, PORTING_PLAYBOOK (rung-4 structural checklist added); README episode-boundary contract now per-reference (TTT=pgx freeze, 2048/soko=jumanji autoreset) | docs commit below |
| D3 | `FIXED` — MEGAKERNEL_PLAN historical banner incl. P1's disproof of "free in registers" | docs commit below |
| D4 + D14 | `FIXED` — README provenance block (date, pre-P1 kernel bias, E3 soko inflation) + pointer to RESULTS_HISTORY/data | docs commit below |
| D7 (SOL-13) | `FIXED` — canonical count (36) lives in HOW_TO_RUN only; README/other mentions non-numeric | docs commit below |

**Amended work plan:**

- **Batch A adds:** S1 (soko obs + gate extension), S5, S6, S3
  (fail-closed sweep + paths + S7 labeling), S2 folded into D2/D4's
  docs pass.
- **Batch B adds:** S4 FIRST (ABBA counterbalance lands before any new
  sweep), P-S1 after E1/E2 settle.
- **Batch C adds:** S8–S11 disclosure/docs items; P-S2 onto the
  deferred list.

**Batch B closeout (2026-07-21):**

| item | status | commit |
|---|---|---|
| S4 (SOL-06) | `FIXED` — shared ABBA core (benchmarks/ab_timing.py), all 7 A/B scripts counterbalanced; landed before any Batch B measurement | `981527d` |
| E2 | `CONFIRMED + ADOPTED` — analytic predicate default for both XLA engines; n=5 medians 1.97/1.90/2.22× branchless, 1.11/1.16/1.13× LUT (B=1k/8k/65k); exhaustive 65536-row gate + bit-identical rollouts (`data/e2_canmask_analytic_ab.jsonl`) | `043f157` |
| E1 | `CONFIRMED + ADOPTED` — analytic mask default in the megakernel (incl. _initial_mask); n=5 medians 1.49/1.52/1.85×, 65k medians span 1.835-1.877; binding gates = jumanji-chained chain link + full-rollout equivalence vs the _move_dir mask path, per the ledger's warning (`data/e1_megakernel_canmask_ab.jsonl`) | `043f157` |
| E5 | `CONFIRMED + ADOPTED` — 9-bubble row move (odd-even + single post-merge pass); exhaustive 65536-row bit-identity incl. reward; n=5 medians 1.06/1.08/1.13×, 14/15 run-medians >1 (`data/e5_rowmove_ab.jsonl`) | `88b6d86` |
| E6 | `FIXED` — was_legal ≡ n_legal>0, gated over all 16 mask patterns × 4096 uniforms | `88b6d86` |
| P-S1 (PERF-02) | `KILLED` — 0.14-0.54× in BOTH sync and desync regimes: the cond boundary breaks step fusion and materializes (B,10,10) operands every step, dwarfing the skipped sampling. Flag + two-regime script kept as the receipt (`data/ps1_soko_gated_ab.jsonl`); sanctioned-cond doctrine updated with this second failure mode | `59bfc46` |
| E7 | `DEFERRED (rides E8)` — Mode A uniform-buffer packing only pays at B=1M (E8, deferred); Mode A is the verification mode, and the contract churn (uniforms shape touches every parity check) buys nothing until then. Implement alongside E8 with distribution gates for derived low-bit uniforms; the rcell-bias warning in this ledger stands. | — |
| E8 | `DEFERRED` (unchanged — lands when B=1M work starts) | — |

**Batch C closeout (2026-07-21):**

| item | status |
|---|---|
| D1 + D10 | `FIXED` — full rung-4 authoring chapter (WRITING_FAST_ENVS §3c): register pre-flight, SoA lanes + int32-in-register rationale, same-function workflow with its blindness caveat, static lane rewiring, counter-RNG registry, pallas_call rules (each a shipped bug or measured cost), gate battery, costs; megakernel.py added to canonical examples; pallas_lab + MEGAKERNEL_PLAN carry guided-history banners |
| D5 | `FIXED` — numbered pre-flight in PORTING_PLAYBOOK (register inventory with 2048's ≈22 data point + sokoban as doesn't-fit, 2-step lowering probe, scan-gated step before pallas_call) |
| D6 | `FIXED` — receipts cross-referenced at idiom sites: rung-2 (E1/E2 receipts), rung-3 boundary (P5 kill), §2 orientation (P4), §2 sanctioned-cond (P-S1 kill), §4 (P8 null) |
| D8 | `FIXED` — HOW_TO_RUN gains the A/B-receipt script table (11 scripts → decisions → receipts) + strict/best-effort flag docs; README maps data/ and distributions.py (D13) |
| D9 | `FIXED` — CLAUDE.md/AGENTS.md: killed-ledger check before perf proposals (with the kill list inline), kernel battery requirement, Triton hard rules, chunk tax, receipts convention, MEGAKERNEL_PLAN read-order caveat; also fixed the agent files' own copy of the disproven stop rule (missed by D2's sweep) |
| D11 | `FIXED` — §7 gate table gains the kernel-parity-battery row |
| D12 | `FIXED` — §4 doctrine boundary sentence (keys for XLA envs; counter-hash in-kernel/state-free replay) |
| S8 (SOL-10) | `FIXED` — hash_uniform docstring: launch-local env_id scope caveat + multi-device offset recipe |
| S9 (SOL-12) | `NO CHANGE NEEDED` — measured: `import djinnax` adds 0.47s on top of jax's 0.33s (CPU), dominated by the 65k LUT build; lazifying the flat public API isn't justified at half a second per process. Revisit if worker-fleet startup becomes real cost. |
| S10 (SOL-07) | `FIXED` — game2048_lut docstring now states the BOUNDED-variant contract explicitly (rarity is not a correctness boundary; bound pinned by check_2048_exp15_divergence) |
| S11 (SOL-11) | `FIXED` — DjinnSokoban docstring states the level-distribution contract (encoding-validated fixtures, solvability not guaranteed, replace fixture for training curricula) |
| P-S2 (PERF-03) | `DEFERRED` — entity-list sokoban core parked on the roadmap (owner decision); needs its own end-to-end gate as a new workload |
