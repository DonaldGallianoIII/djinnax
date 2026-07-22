# LEARNINGS — the djinn JAX engine discipline

Living document. Everything empirically learned building and racing the
djinn-style envs against pgx/jumanji. This is the methodology behind every
number the repo publishes, and the ground truth for the eventual djinn DSL.
Update it every time a claim is confirmed, killed, or bounded.

---

## 1. The discipline (what "djinn style" means)

1. **Batch-native, never vmap-a-single-env.** State carries a leading `B`
   on every field; step logic is written for the whole batch. vmap is a
   *porting* tool, not an architecture.
2. **No control flow in the hot path.** No `lax.while_loop`, no
   `lax.switch`, no data-dependent `cond` inside the per-step body.
   Everything is `where`-masked dense vector ops. Control flow under vmap
   is poison: a vmapped `while_loop` runs every element to the batch
   worst case; a vmapped `switch`/`cond` executes all branches and
   selects.
3. **dtype economy.** Smallest dtype that fits (int8 boards, bool masks),
   widen-before-add where overflow is possible, clamp on write.
4. **Keys are threaded, never stored.** Per-step key in, `fold_in` with a
   documented salt registry per consumption site. No key field in state.
5. **Terminal states are never special.** Auto-reset via `where`-select
   (in-step or template), and the legal mask is NEVER all-False — one
   designated inert action (PASS) stays legal so a masked softmax can't
   NaN.
6. **Mask and apply share one predicate** (or a single choke point), and a
   property test quantifies both directions over the whole action space:
   illegal ⇒ leaf-identical no-op, legal ⇒ documented effect.

## 2. The escalation ladder (apply in order — rung 4 moves the floor itself)

1. **Branchless rewrite** — replace loops/switches with compact
   (rank-scatter), unrolled fixed-length passes, orientation canonicalization
   (transform so every direction becomes "left", operate, transform back).
2. **Delete work analytically** — if a state is structurally simple
   (fresh reset board = one tile), derive its properties (legal mask) in
   closed form instead of simulating.
3. **LUT-ify** — when a sub-state fits in ≤ ~2^16 configurations
   (a 2048 row = 4 nibbles = 16 bits), precompute EVERYTHING about it in
   numpy at import and turn runtime into pack → gather → unpack. 384KB of
   tables beats any amount of clever arithmetic.
4. **Persistent kernel (Pallas)** — the rung that removes the floor the
   others stop at. We originally taught "if your complex game is as fast
   as your trivial game at huge B, a kernel can't help — it chases the
   same floor." **That rule is DISPROVEN by our own measurement** (§6):
   2048-LUT had hit TTT speed at B=65536, and the persistent megakernel
   still delivered ~5-7.6× over it — because the "platform floor" is an
   XLA artifact (per-step kernel launches + HBM state materialization
   between ops), not a hardware limit. A persistent kernel holds state
   in registers and launches once per rollout; it doesn't chase that
   floor, it deletes it. Eligibility is structural, not floor-based —
   see PORTING_PLAYBOOK's rung-4 checklist. Expect ~2-8× over your best
   XLA when the step fits in registers; costs: 30s+ compiles,
   hardware-generation-specific lowering, 2.2-3.9× chunk tax if the
   training loop splits rollouts.

Measured on 2048 vs jumanji (within-run): branchless ≈ 7-13×,
+LUT ≈ 15-61× depending on B, megakernel ≈ 100-270×. TTT (trivially
branchless already) ≈ 1× — the discipline is free when logic is simple
and decisive when it isn't.

## 3. Measurement methodology (non-negotiable order)

1. **Parity BEFORE performance.** Speed of a wrong engine is meaningless.
   Gate every variant on move-for-move identity with the reference
   (replayed games / direct kernel comparison), plus variant≡variant
   bit-identical chained-step checks when optimizing.
2. **Identical protocol for every engine** — same jitted scan, same
   in-graph mask sampling, same reset handling, same reps. The sampler is
   part of the measured system; keep it constant.
3. **Within-run ratios are the stable quantity.** On a shared/WSL2 GPU,
   absolute numbers swing 2-3× run-to-run with clocks and contention
   (pgx TTT itself halved between two of our runs, same code). Never
   compare across runs; print ratios per run.
4. **N independent fresh-process runs, median + spread** for any headline
   number. Best-of-reps within one run is a *lower bound on noise*, not
   statistics.
5. **Know your noise floor and respect it.** A same-run A/B of two
   compaction kernels flipped direction across batch sizes — sub-2×
   deltas on an unlocked-clock host are weather, not signal. Only claim gaps that
   dwarf the spread.
6. **Coverage floors on any random-play driver** (did it actually reach
   combat/carousel/termination?) so invariant tests can't pass vacuously.
7. **Report null results** (TTT win-LUT ≈ noise) and keep them in the
   repo behind flags. Honesty is the moat: a benchmark nobody can
   falsify is a benchmark nobody trusts.
8. **Check the backend every time** — JAX silently falls back to CPU when
   CUDA libs don't load; assert `jax.default_backend()` in every bench.
9. **Sequential A/B is INVALID on boosting GPUs.** Running config A then
   config B in sequence measures clock ramp, not configs: in one
   five-config comparison the ranking *inverted* when the order was
   reversed — position explained more variance than the code did. Any
   sub-2× comparison must be **interleaved pairwise**: alternate
   A-call/B-call, take the per-round time ratio, report the median ratio
   with min..max. (This retroactively demoted two of our own headline
   deltas — a "3× regression" and a "16× floor win" — both artifacts.)

## 4. The proto-language (djinn DSL seed)

The same small vocabulary keeps re-emerging in every port. These are the
candidate primitives of the future pseudo-lang — a game described in these
terms compiles mechanically to both JAX and WebGPU:

| primitive | meaning | JAX realization |
|---|---|---|
| `orient(board, d)` | canonicalize a direction to one case | compile-time transpose/flip, self-inverse |
| `compact(row)` | stable-shift nonzeros | exclusive-cumsum rank + one-hot scatter |
| `pairscan(row, f)` | fixed-length unrolled adjacent-pair pass | python-unrolled where chain |
| `lut(sub_state)` | total function of a ≤2^16 sub-state | numpy precompute + gather |
| `spawn(board, dist)` | random write to a masked cell | categorical over masked logits + one-hot write |
| `mask(state)` | legal actions, branchless | boolean algebra per action range |
| `pick(mask, key)` | policy-shaped sampling | categorical over `where(mask, 0, -inf)` |
| `reset_where(done)` | auto-reset | tree-map where-select vs template |
| `analytic(prop)` | closed-form property of structured state | hand-derived expression (e.g. single-tile mask) |
| `salt(site)` | RNG stream per consumption site | `fold_in` + documented registry |
| `collapse(process)` | outcome-draw for a stochastic loop with no observable intermediates | closed-form inverse CDF / conditional renorm / alias tables (`distributions.py`) |

Design rule for the DSL: **if a game can't be expressed in these
primitives, that's the signal it will be slow** — the language should make
the fast path the only path.

## 5. Consulting checklist (compressed)

- [ ] Reference implementation identified; parity harness built FIRST
- [ ] Ladder applied in order; stopped at demonstrated floor, not fatigue
- [ ] Trivial-game control benched (isolates framework overhead from logic)
- [ ] N-run median + spread; within-run ratios; noise floor stated
- [ ] Null results and caveats in the writeup
- [ ] Every claim reproducible by a stranger with one command

## 6. Confirmed / killed (2026-07-19 evening)

**The 61× is DEAD as a headline; the medians are the claim.** Official
frozen-code n=5 fresh-process sweep, within-run ratio median [min..max],
2048-LUT vs jumanji: **15.1× [13.9..86.8]** at B=64, **25.2× [22.8..34.8]**
at 1024, **24.8× [17.6..61.4]** at 8192, **17.7× [15.9..19.2]** at 65536.
The tightest, most defensible number is the floor-limited one. Quote
medians with spread, never a single run's max. (Meta-lesson: the
reviewer's instinct to "kill or confirm the 61" was correct — always
stress the headline before building on it.)

**Absolutes swing ~4× run-to-run; ratios mostly hold.** Same code, same
box: ttt/djinn spanned 92M..380M env-steps/s across 5 runs at B=8192.
Ratio spreads are far tighter except where one hot run inflates a single
engine. §3's rules are not paranoia; they are the difference between a
result and an anecdote.

**Code freeze during sweeps** (new rule, learned by violating it): editing
the bench mid-sweep let unfinished Sokoban rows leak into runs 4-5 of the
first sweep. A sweep's code must be frozen for all N runs; anything else
taints the sample. The official sweep was re-run frozen.

**Floor tuning is unmeasurable on contended hardware.** Same-process A/B
of scan unroll {1,4} × PRNG {threefry, unsafe_rbg} at B=8192/65536 gave
contradictory winners per engine (unroll=4: ttt +3.5×, soko −2.3×, in the
SAME block; unsafe_rbg flipped sign between settings). Both knobs live
inside the measurement host's clock-noise envelope; unroll=4 also 3-4×'s compile
time. Defaults stay (unroll=1, threefry); flags remain for a
controlled-clock machine. Corollary: some engineering questions cannot be
answered on the hardware you have — say so instead of picking the run you
like.

**Sokoban (env #3) parity-gated and in.** 40 replayed episodes
grid/agent/reward/done identical to jumanji. Official ratios: **1.7× /
1.8× / 2.3× / 2.3×** at B=64/1024/8192/65536. Reading: jumanji's sokoban
step is already select-based (no while_loop), so the paradigm delta
shrinks to ~2× — consistent with the theory: **the win scales with how
much control flow the reference left in its hot path** (ttt ≈ 1.1-1.4×,
sokoban ≈ 2×, 2048 ≈ 15-25×). That spectrum IS the paradigm claim, not
any single number.

**Game-agnostic runtime (protocol v2) — weakly positive, adopt with
stated evidence level.** `runtime.py`: bulk-hoisted per-step keys
(one batched split threaded through scan xs instead of in-loop fold_ins),
rank-pick sampler (uniform-over-legal via ONE uniform per row + cumsum
rank select — distribution-identical to masked categorical, O(B·A)
Gumbels → O(B) integers; uniformity-tested), optional carry donation, and
a NullEnv that measures the pure runtime floor. Interleaved-pairwise
verdict across ttt/2048/soko/null at B=1024/8192/65536: one solid win
(sokoban 1.73× [1.51..1.79] at 1024), the rest within noise
(medians 0.86-1.63× straddling 1), never clearly worse. Adopted as the
recommended runtime on "cheaper by construction + never worse" grounds;
magnitude claims deferred to locked-clock hardware. The sequential
version of this experiment produced dramatic false deltas in BOTH
directions before interleaving corrected it (see §3 rule 9).

**Pallas (rung 4) — explored hands-on (`pallas_lab.py`), 2026-07-20.**
- **Backend gotcha first:** JAX's default Pallas-GPU lowering (Mosaic GPU)
  emits Hopper TMA instructions — **sm_90+ only**. On the RTX 4070
  (sm_89) it fails in three escalating ways (128B warpgroup copy
  granularity → 256 elems/dim async-copy limit → flat "not supported on
  sm_89"). The legacy **Triton lowering**
  (`compiler_params=plt.CompilerParams()`) runs fine on sm_89 with none
  of those constraints. Rung-4 code is hardware-generation-specific in a
  way rungs 1-3 never are.
- **Kernel style:** in-kernel arrays support jnp elementwise but NOT
  `slice`/`.at[]` (Triton) — the natural shape is **structure-of-arrays**
  (lanes as separate refs) with explicit swap networks (zero-bubbling
  compaction). Also: Triton's float `pow` is inexact — parity failed on
  `2.0 ** x` until replaced with an integer shift. Parity gates catch
  kernel DSL precision differences; keep them.
- **Result (unlocked clocks, interleaved medians):** the fused row-move
  kernel beat the **rung-1 branchless XLA** version ~1.8-2.4× (interval
  fully >1 at 262K rows) — real fusion win: XLA materializes
  intermediates between compact/merge; the kernel keeps the whole
  pipeline in registers. BUT it was benched against rung 1, not rung 3:
  the LUT had already made this computation nearly free, and arbitrary
  gathers (the LUT's trick) are exactly what kernel DSLs make hard while
  XLA excels at them. **Rung 4 is for fusion XLA can't find, not for
  re-doing gathers** — and it stays below rung 3 in the ladder for
  anything LUT-able.
- Debugging lesson: when parity fails, suspect the test GLUE as much as
  the kernel (the "kernel bug" was a board-shaped helper collapsing
  per-row rewards on row-shaped input).

**The megakernel WORKS — rung 4 has a real seat at the table
(2026-07-20, `megakernel.py`).** One persistent Triton kernel runs
the entire 64-step rollout per env block, state living in registers as 16
SoA lanes; per-direction moves are static lane rewirings (no orientation
select at all), sampler/spawn/reset are the §4 primitives in-register.
Verdict: **2.1-2.7× over the production LUT pipeline** (self-contained,
interleaved, intervals entirely >2 at B≥8192) and 2.2-3.7× over the
same-formulation XLA scan — through unlocked-clock noise, i.e. a real
multiple. Revised ladder claim: rung 4 is for **launch-count and
inter-step materialization** — exactly what a persistent loop eliminates
and what no amount of XLA-side cleverness could touch (every XLA variant
had converged to the same floor). Two techniques to keep:
- **Same-function parity**: write the step as a pure jnp function on
  lane tuples and execute it both in-kernel and under lax.scan —
  bit-identical by construction, so the A/B measures execution strategy
  ONLY. Made kernel parity trivial where it's usually the hard part.
- **Uniform-buffer RNG split**: pre-generate uniforms as an input to
  both sides to isolate loop/launch effects from RNG cost; then include
  generation in the timed region for the production comparison.
Cost: kernel compile 33-36s at large B (vs 11-16s XLA).

**Mode B (in-kernel counter RNG) — the capstone.** Double-fmix32 hash of
(env_id, step, salt, seed) in pure uint32 — which means the same-function
parity trick covers the RNG too (bit-verified vs XLA reference), plus
distribution tests. In-kernel RNG bought **1.9-2.3×** over the
uniform-buffer mode (HBM randomness traffic was that expensive). The
fully self-contained megakernel is **~5-7.6× the production LUT
pipeline** and — directly measured, interleaved — **268× [239..360]
jumanji at B=8192, 116× [88..133] at B=65536**, at 2.17B env-steps/s.
Final ladder on one game: reference → branchless (≈8×) → LUT (≈18-25×) →
megakernel (≈100-270×). New primitive for §4: `ctrhash(counters) → u01`
(counter RNG); new rule: RNG is state-free — derive from (who, when,
what, seed), never carry generator state. RNG-stream caveat vs jumanji
(counter hash vs threefry; same distributions) stays attached to the
number.

**The contention test (accidental robustness validation, 2026-07-20).**
A GPU-intensive application was running on the host for the entire
Mode B benchmark session — disclosed after the fact. Quiet-GPU re-run vs the
contended originals: jumanji-vs-megakernel 268×→275× (@8192) and
116×→110× (@65536); LUT-vs-megakernel 6.2×→6.3×; every quiet median
landed inside the contended run's interval. **Interleaved pairwise
ratios survived live gameplay contention** — the contention hit both
sides of each round and cancelled. What DID move: absolute throughput
(−10%) and the bandwidth-sensitive Mode A/B ablation (1.9×→2.9× — the
game was eating exactly the memory bandwidth that ablation measures).
Lessons: (a) the ratio methodology is robust enough that a disclosed
contention event downgrades, not destroys, a session; (b) ablations of
bandwidth effects are the fragile ones — re-run those when contention is
disclosed; (c) always ask what else the machine was doing.

**Hardening round 2 (2026-07-20/21, `858aa0a`+`bbd80c2`).** Highlights:
- **The parity chain is closed end-to-end**: jumanji ≡ game2048 ≡
  megakernel lanes, tested at every joint. A hardening pass also found a
  REAL latent bug: B not divisible by BLOCK silently dropped the tail
  envs (now a loud ValueError). Grid-coverage arithmetic is a standing
  audit item for any pallas_call.
- **Chunked rollouts are proven** (score input + RNG t_offset; 2×32 ≡
  1×64 bit-for-bit) — but the **chunk tax is 2.2×/3.9× for k=2/8**: the
  kernel is so fast that per-launch overhead dominates when split.
  Training loops should chunk as coarsely as the algorithm allows.
- **Tuning null**: the guessed block=128/default-warps was already
  optimal (everything else regressed) — Triton's defaults + a sane tile
  guess go far; sweep to CONFIRM, not to explore forever.
- **Quiet-GPU resolution of old nulls**: unsafe_rbg is a genuine 0.63×
  REGRESSION on these workloads (rejected — "faster RNG" isn't); scan
  unroll=4 is +56% on the launch-bound trivial env, neutral on heavy
  ones — engine-dependent, defaults kept. Verdict quality scales with
  measurement quality: yesterday "unresolvable", today resolved.
- **CI contract established**: pytest suite runs fully on GPU, kernel
  tests self-skip on CPU (current counts: 26 GPU / 21+5skip CPU) — the chain-link, analytic-mask, and RNG batteries gate on any
  machine; kernel tests self-skip without a GPU.

**Single-fmix RNG NULL (2026-07-21).** Halving the counter-hash
finalizer passes the full quality battery but measures 1.00-1.03x
(n=5 interleaved, data/p8_rng_rounds_ab.jsonl) — after the orient-select
kernel cut, RNG is not a meaningful ALU fraction. Double-fmix kept: when
a cheaper variant is statistically indistinguishable AND not faster,
keep the stronger one.

**Sokoban carried-count NULL (2026-07-21).** Carrying n_on_target in
state instead of recounting (B,100) per step: 0.99-1.02x at every B
(n=5 interleaved, data/p6_soko_carry_ab.jsonl) — the reduction fuses
into the step's existing grid passes. "Don't recompute what you know"
is a perf doctrine, and here the recompute is free; simpler recount
kept as default, flag + field kept for the receipt.

**CHANGED_LUT probe KILLED (2026-07-21).** Replacing the legality
mask's unpack-and-compare with a third 64 KB bool-LUT gather measured a
consistent 0.82-0.88x REGRESSION at every B (n=5 interleaved,
data/p5_canmask_ab.jsonl): on RTX 4070 the extra gather costs more than
the unpack ALU it deletes. Rule reinforced: LUT-ify *logic*, not
*comparisons already fused next to an existing gather*. Code + exactness
gate kept unwired as the receipt.

**collapse(process) measured (2026-07-21).** 2048's spawn ported as the
reference's rejection loop (random cell, retry if occupied) vs the
collapsed conditional draw — distribution-parity-verified identical, then
interleaved on full games: **collapsed is 47-75× faster** (75× at
B=1024, 47× at 65536; worst-case single-empty-cell boards 49× on the
spawn alone). A while_loop inside the step doesn't just iterate — it
breaks step fusion and pays batch-max tail latency every step. One
collapsed RNG site ≈ the entire branchless-vs-while_loop gap measured
earlier. The stochastic-loop rung is not a refinement; it is co-equal
with rung 1.

## 7. Open questions
- [ ] Shared-memory-tier megakernel: the register tier caps at a few
      hundred bytes/env (README scope note); SMEM offers a few KB/env
      while still deleting the launch/materialization floor. Does the
      win survive the bandwidth re-entry? Needs its own A/B on a
      KB-state env (entity-list sokoban is the natural candidate);
      precedent at the tier above: Madrona (Stanford).
- [ ] WebGPU port: do the same primitives hit the same ratios in-browser?
- [ ] Controlled-hardware floor study (headless box): unroll/RNG/donation.
- [ ] DSL: can sokoban's step be expressed purely in §4 primitives?
      (gather/one-hot-write/analytic — yes on paper; prove by codegen.)
