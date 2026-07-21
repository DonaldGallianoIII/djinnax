# PORTING PLAYBOOK — reverse-engineered game → fastest-method JAX

The decision procedure. WRITING_FAST_ENVS.md teaches the idioms; this
doc answers the question that costs 10-100× when guessed wrong: **which
method does THIS game get?** Work the census, read the table, then build.

## Step 0 — extract the spec (before any JAX)

From the source (C++, TS, whatever), write down:
1. **State census**: every field, its range (→ dtype), and whether it's
   per-cell (grid) or per-entity. Count: cells, entities, players.
2. **Action census**: how many discrete actions; what makes each legal.
3. **RNG census**: every random draw site and its distribution.
4. **Termination + reward rules.**
5. **The reference's step order** — you will mirror it exactly, because
   your parity harness replays the reference move-for-move (build that
   harness FIRST; speed of a wrong engine is meaningless).

## Step 1 — representation (the biggest single decision)

| census says | representation | why |
|---|---|---|
| most cells occupied / cell-local rules | grid, int8, (B, H, W) | dense ops match dense state |
| entities ≪ cells (agent + few objects) | **entity list**: coords per entity, (B, N_ent, 2) | 10 bytes beats 100; bandwidth is the floor at scale |
| per-player sub-boards, shops, benches | fixed-size padded arrays + sentinel -1, leading (B, P, …) | never dynamic shapes |
| a sub-state fits in ≤ ~16 bits (a row, a column, a small occupancy) | **note it now — it's LUT fuel for step 3** | 2^16 tables fit in L2 |

Sokoban's lesson: it's a 100-cell grid with 5 entities — the entity-list
form is ~10× less state traffic. 2048's lesson: a row is 16 bits →
the whole move rule became a 65,536-entry table.

## Step 2 — write rung 1 (branchless baseline)

Straight from WRITING_FAST_ENVS §2: compute-all/mask-select, one-hot
writes, orientation canonicalization (one move rule + static rewiring,
never four copies), python-for over small fixed ranges, **no while_loop /
switch / traced-if anywhere**. Any loop in the C++ whose iteration count
depends on data must become: a fixed-bound unrolled pass, a prefix-sum
(rank-scatter), or a precomputed table. If you can't see how — that's
the design problem to solve BEFORE writing more code, not after.

Pass parity. Bench against the NullEnv floor (`floor_bench.py`): the gap
between your env and the null env is what your logic costs.

## Step 3 — climb while the floor gap pays

1. **Analytic deletion** — anything structurally known (fresh boards,
   single-entity masks) gets a closed form, not a simulation.
2. **LUT-ify** every ≤16-bit sub-state from your census: precompute the
   total function (result + reward + changed) in numpy at import;
   runtime = pack → gather → unpack.
3. **Megakernel** (rung 4) — ONLY when your env at huge B is still
   slower than the null floor and steps are sequential-per-env: port the
   step to structure-of-arrays lanes (the SAME jnp function runs
   in-kernel and under lax.scan → bit parity for free), counter-hash RNG
   (`ctrhash(env_id, t, salt, seed)` — state-free), grid over env
   blocks, fori over steps. Expect ~2-7× over your best XLA and know the
   costs: ~30s compiles, hardware-generation-specific lowering, and a
   2-4× chunk tax if the training loop splits rollouts (chunk coarsely).

Stop when your game runs as fast as tic-tac-toe at B=65536 — the logic
is free, and further effort belongs to the training loop, not the env.

## Step 4 — the gate (no exceptions)

Parity vs reference · variant≡variant bit-equivalence at every rung ·
mask/apply property both directions · conformance driver with coverage
floors · trace guard · serialization round-trip · RNG distribution tests
if you added a sampler or hash. Then, and only then, `sweep_stats.py`.

## Worked example — how 2048's decisions were made

census: 16 cells, all can fill → grid int8. 4 actions. rows are 4
nibbles = 16 bits → LUT fuel. RNG: spawn cell + value → 2 sites + salts.
rung 1: argsort compact → later rank-scatter; merge = 3 unrolled pairs;
directions = orient-to-left. rung 2: reset board is single-tile →
analytic mask (deleted 4 of 9 move passes). rung 3: row-move LUT →
logic ~free at 65k. rung 4: steps sequential, env-parallel → megakernel;
final: 98-237× the reference. Every rung kept the same parity gate.

## Porting smells (each cost us or a reference real performance)

- "I'll vmap the single-env version for now" → that IS the slow method.
- A while-loop that "usually exits early" → batch-worst-case forever.
- Mask logic re-stated in the apply path → drift; one choke point.
- Storing the RNG key in state → thread keys; hash counters.
- `float(x)`/`if bool(x)` mid-step → host sync, 10-100×.
- Benchmarks before parity → you optimized a different game.
