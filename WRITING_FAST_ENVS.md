# Writing fast envs — the djinn authoring guide

How to write a game in JAX so it is the fastest version of that code that
can exist on this engine. LEARNINGS.md is *why* (evidence, methodology);
this is *how* — the rules to follow while writing. Canonical worked
examples: `ttt.py` (trivial), `game2048.py` + `game2048_lut.py`
(the full ladder), `sokoban.py` (grid + entities), `runtime.py`
(the shared loop).

**The one-sentence thesis:** you are not writing a program about one game;
you are writing a dense array program about B games, where B is huge —
every line must do its work for the whole batch at once, with no
data-dependent control flow anywhere in the hot path.

---

## 1. State design

```python
@flax.struct.dataclass
class MyState:
    board: jax.Array        # (B, H, W) int8   — leading B on EVERY field
    flags: jax.Array        # (B, N) bool
    count: jax.Array        # (B,) int16
```

- **Leading `B` on every field.** Never a scalar-per-env design that needs
  vmap. Batch-native is the architecture, not an afterthought.
- **Smallest dtype that fits**: `int8` for boards/counters ≤127, `int16`
  for ids/steps, `bool_` for flags, `float32` only where fractional math
  happens. Rule for arithmetic near a dtype's edge: **widen → add → clamp
  → narrow** (`x.astype(int16) + y` then `clip` then `.astype(int8)`).
- **Sentinel `-1` (`EMPTY`)** for absent entities; guard lookups with
  `jnp.maximum(idx, 0)` or `clip` and mask the result.
- **Fixed shapes forever.** Over-allocate to the maximum (32 unit slots, 6
  pool slots) and mask, never grow. If you're tempted by a dynamic shape,
  you're designing the wrong state.
- **No PRNG key in state.** Keys are threaded per step (see §5).
- **Representation follows entity density**: full grid when most cells
  matter; **entity list** (coords per entity) when entities ≪ cells
  (sokoban: 5 entities vs 100 cells → 10 bytes beats 100). Bandwidth at
  the floor is proportional to state bytes.
- **Precompute presence caches** at phase boundaries for anything the hot
  loop would repeatedly derive (`unit_has_X: (B, N) bool` computed at
  deploy, read per tick).

## 2. Step function

Signature: `step(state, action, key) -> (state, ...)` — pure, jittable,
batch-in/batch-out.

- **The universal idiom is compute-all, mask-select:**
  ```python
  fires = (action == MY_ACTION) & precondition          # (B,) or (B, P)
  new_x = jnp.where(fires, computed_x, state.x)
  ```
  Every actor's effect is computed unconditionally and masked off. This is
  not waste — it is the shape the GPU wants.
- **Writes are one-hot selects**, not `.at[]` with traced per-env indices
  in a loop:
  ```python
  cells = jnp.arange(N_CELLS)[None, :]
  write = (cells == target_idx[:, None]) & fires[:, None]
  grid = jnp.where(write, value[:, None], grid)
  ```
- **Reads are `take_along_axis`** on a flat index (`y * W + x`), with
  clipped indices + validity masks for out-of-bounds.
- **Python `for` over small compile-time ranges is GOOD** (4 directions, 6
  slots, 3 merge pairs): it unrolls into straight-line fused code. What's
  forbidden is *traced* control flow.
- **Forbidden in the hot path:** `lax.while_loop` (runs the whole batch to
  the worst case — measured 15-25× on 2048), data-dependent `lax.switch`
  or `cond` (executes all branches + select under batching), python `if`
  on a traced value (either a trace error, or worse, a silent recompile).
- **The one sanctioned `cond`:** batch-gating an *expensive, batch-rare*
  block — `lax.cond(jnp.any(need), expensive, noop, state)`. Know its
  failure mode: if envs desynchronize, `any(need)` ≈ always true and the
  amortization collapses (measured 20× on a production training env's combat phase). Design phases to
  stay batch-synchronized if you rely on this.
- **Orientation canonicalization**: never write per-direction logic four
  times. Transform so every case becomes one case, operate, transform
  back. The transforms are compile-time python (unrolled), self-inverse:
  ```python
  oriented = select_by_action([_orient(board, a) for a in range(4)], action)
  moved    = move_left(oriented)
  board    = select_by_action([_orient(moved, a) for a in range(4)], action)
  ```
- **Terminal states are never special-cased** — they flow through the same
  step, masked. Auto-reset in-step via tree-map `where`-select against a
  template (or freeze + external template reset; pick one, test it).

## 3. The escalation ladder (write rung 1, then climb while it pays)

**Rung 1 — branchless rewrite.** Recipes:
- *Compaction* (shift nonzeros left, order-preserving):
  ```python
  nz = rows != 0
  rank = jnp.cumsum(nz, -1) - nz          # target slot of each survivor
  sel = nz[..., None] & (rank[..., None] == jnp.arange(K))
  out = (rows[..., None] * sel).sum(-2)
  ```
- *Pair scan* (adjacent merge/compare, fixed width): unrolled python loop
  of `where` updates, left to right — each element touched once.
- *First-empty / first-match*: `argmax` of the boolean + `any` guard.

**Rung 1b — collapse stochastic processes (co-equal with rung 1).**
Any retry-until-success / reroll-until-valid / draw-until-drop loop with
no observable intermediates collapses to ONE draw from its closed-form
distribution (`djinnax/distributions.py`): geometric inverse-CDF for
retries (`N = 1 + floor(log(1-U)/log(1-p))`), renormalized conditional
for rerolls, alias tables for weighted picks. Exact — full variance
preserved, only the iteration deleted. Measured on 2048's spawn: the
faithful rejection-loop port is **47-75× slower on the whole game** than
the collapsed draw (distribution-parity-verified identical). Legality
test and the process→distribution table: PORTING_PLAYBOOK step 1.5.

**Rung 2 — delete work analytically.** If a state is structurally simple,
derive its properties in closed form instead of simulating. (A fresh
board with one tile: its legal mask is four coordinate comparisons, not
four move simulations.) Ask of every recompute: "do I already know this?"

**Rung 3 — LUT-ify.** When a sub-state fits in ≤ ~2^16 configurations,
precompute the *total function* of it in numpy at import:
```python
# a 2048 row = 4 nibbles = 16 bits -> tables of ALL 65,536 outcomes
codes = (board.astype(int32) * PACK_WEIGHTS).sum(-1)   # pack
out   = MOVED_LUT[codes]                               # ONE gather
moved = ((out[..., None] >> SHIFTS) & 15).astype(int8) # unpack
```
Runtime logic cost → zero; 384KB of tables sit in L2. Candidates: any
row/line/neighborhood of ≤16 bits (2048 rows, TTT occupancy 9 bits,
tetris rows, connect-4 columns, rule-based cellular updates).

**Rung 4 — custom kernel (Pallas).** Only after rung 3 leaves a measured
gap to the platform floor. Test first: *is your complex env as fast as a
trivial env at huge B?* If yes, logic is free — a kernel chases the same
launch/bandwidth floor. See `pallas_lab.py` + LEARNINGS for evidence.

## 4. RNG

- **Thread keys, never store them.** `step(state, action, key)`.
- **One salt per consumption site**, documented in a registry comment;
  derive with `fold_in(key, SALT)`. Never reuse a salt.
- **Per-entity keys via `vmap(fold_in)`** over entity index — NEVER
  `broadcast_to(key, ...)` (correlated streams — a bug we shipped once).
- **Hoist per-step keys out of the loop** (runtime v2): one batched
  `split` before `lax.scan`, threaded through `xs` — not fold_ins inside.
- **Sampling uniform-over-legal**: one uniform per row + cumsum rank pick
  (`djinn_runtime.sample_uniform_legal`), not a Gumbel per action.
- Random cell writes: categorical over `where(empty, 0, -inf)` logits +
  one-hot write (see `_spawn` in game2048).

## 5. Action masking

- Mask lives in ONE place and shares its predicate with apply — a single
  choke point (`chosen = where(can_act, chosen, -1)`) beats re-stating the
  predicate in nine functions.
- **A mask row must never be all-False** (dead/terminal actors): -inf
  logits → NaN softmax. Keep one designated legal-but-inert action (PASS).
- Mask semantics are a *contract*: illegal ⇒ apply is a leaf-identical
  no-op; legal ⇒ the documented effect happens. Both directions get a
  property test over the full action space (see §7).

## 6. Observations

- Wrap the return in `lax.stop_gradient`.
- Normalize to a documented envelope ([0,1] / [-1,1]) with per-field
  divisors; the conformance driver asserts the envelope.
- Derive from LUTs (`ROSTER_*_NORM[ids]`), never recompute per step.
- Layout documented byte-by-byte in the docstring; a shape/offset change
  bumps obs_dim and its tests in the same commit.

## 7. The test gate (no env ships without all of these)

| test | what it proves | canonical example |
|---|---|---|
| Parity vs reference | the game is CORRECT (else speed is meaningless) | check_parity.py |
| Variant equivalence | optimized rung ≡ naive rung, bit-identical chained steps | v2≡v3 in check_parity |
| Mask/apply property | contract of §5, all actions × both directions | training-repo tier* |
| Conformance driver | invariants under mask-guided random play, every step | training-repo tier* |
| Trace guard | `chex.assert_max_traces(step, n=1)` — no silent recompiles | training-repo tier* |
| Serialization | state survives to_bytes/from_bytes with dtypes + behavior | training-repo tier* |
| Sampler uniformity | any custom sampler is distribution-correct | floor_bench.py |
| Distribution parity | collapsed stochastic sites ≡ the naive loop statistically | tests/test_distributions.py, benchmarks/spawn_collapse_ab.py |

\* The starred rows apply when the env feeds a training loop (masked
policies, checkpointing, long-lived jit caches). The benchmark envs in
this repo have no action masks or serialization surface, so those gates
live in the downstream training integration, not here — the in-repo
gates are the unstarred rows, all present in `checks/` and `tests/`.

Write the parity harness BEFORE optimizing. Every ladder rung re-runs it.

## 8. How to verify speed (short form; LEARNINGS §3 is the law)

1. `assert jax.default_backend() == "gpu"` (JAX falls back to CPU
   silently; use the LD_PRELOAD shim, HOW_TO_RUN.md).
2. Identical protocol across compared engines; within-run ratios only.
3. Headlines: n≥5 fresh-process sweep, median [min..max]
   (`sweep_stats.py`). Code frozen for all N runs.
4. Sub-2× deltas: interleaved pairwise A/B or don't claim them.
5. Compare against the NullEnv floor: your env's µs/step minus the null's
   is what your game logic actually costs. When they're equal, stop
   optimizing.

## 9. Anti-pattern quick table

| anti-pattern | symptom | fix | measured cost |
|---|---|---|---|
| `while_loop` in step | batch runs to worst case | unroll / rank-scatter / LUT | 15-25× (2048) |
| retry/reroll RNG loop in step | breaks fusion + batch-max tail every step | collapse to closed-form draw (`distributions.py`) | **47-75×** (2048 spawn) |
| data-dep `switch`/`cond` | all branches execute | select-by-index over unrolled variants | part of above |
| per-direction copies of logic | 4× code, 4× bugs | orientation canonicalization | correctness |
| python `if` on traced val | recompile every call | `where`; trace guard catches it | seconds/step |
| `broadcast_to` a PRNG key | correlated randomness | `vmap(fold_in)` | silent wrongness |
| key stored in state | replay/planning leaks, extra state bytes | thread keys | design debt |
| all-False mask rows | NaN softmax downstream | PASS always legal | training crash |
| int8 add near 127 | silent wraparound | widen→add→clamp→narrow | silent wrongness |
| host sync per step (`float(x)`, `if bool(x)`) | GPU starves | keep everything traced | 10-100× |
| dynamic shapes / growing arrays | retrace per shape | fixed max + mask | compile storm |
| predicate duplicated in mask & apply | silent drift | one choke point + property test | latent bugs |
| gathering what a LUT could store | per-step recompute | numpy precompute at import | rung-3 wins |

## 10. Authoring workflow for a new game

1. Read the reference implementation; write down state, actions, rewards,
   termination, and every RNG site (assign salts now).
2. Design state per §1 (dtype table + entity-density call). Write
   `init`/`step`/`mask`/`obs` at rung 1 (branchless, no cleverness).
3. Build the parity harness against the reference. Green before anything.
4. Climb the ladder (§3) while the NullEnv gap says there's headroom;
   re-run parity + variant-equivalence at every rung.
5. Add the §7 test gate; wire into `bench_head_to_head.py` +
   `sweep_stats.py`; record numbers per §8.
6. Write the LEARNINGS entry: what generalized, what was game-specific,
   any new primitive for the DSL vocabulary.
