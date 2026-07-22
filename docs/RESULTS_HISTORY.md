# Results history (imported development log)

> Internal results log imported from the development workspace; file
> names referenced below may predate this repository's layout.

# Head-to-head: djinn engine style vs pgx/jumanji — SAME games (2026-07-19)

Follow-up to an earlier internal cross-engine log, after a scope correction: not
different games under one protocol — the **same games**, implemented in the
djinn house style, raced against the reference implementations. Same rules,
same GPU (RTX 4070, 50%-cap env), same jitted-scan protocol, same in-graph
mask sampling, same reset handling. The only variable is engine engineering.

**Ports are correctness-gated first** (`engine-bench/check_parity.py`):
- TTT: 200 full games replayed through both engines — boards, winners,
  rewards, masks, observations identical move-for-move.
- 2048: 500 random boards × 4 directions against jumanji's own
  `move`/`can_move` — moved boards, rewards, masks identical; spawn checked
  (single tile on an empty cell, P(4-tile)=0.094 ≈ 0.1).

The style contrast being measured:
- **djinn style** (the djinnax discipline): batch-native state (leading B, no
  vmap), flax.struct, int8 boards, fully branchless (`where` masking,
  stable-argsort compaction + unrolled merge pairs for 2048).
- **reference style**: single-env logic under `jax.vmap`; jumanji's 2048
  builds each row move from `lax.while_loop` + `lax.switch`.

## Results (env-steps/s, best of 5 × 64-step scans)

| game | B | reference | djinn | djinn ÷ ref |
|---|---:|---:|---:|---:|
| ttt (vs pgx) | 64 | 3.11M | 3.40M | 1.09× |
| ttt | 1024 | 19.4M | 18.5M | 0.95× |
| ttt | 8192 | 190M | 222M | 1.17× |
| **2048 (vs jumanji)** | 64 | 33.2K | 643K | **19.4×** |
| 2048 | 1024 | 468K | 6.24M | **13.3×** |
| 2048 | 8192 | 3.59M | 11.9M | **3.3×** |

Compile: ttt ~0.5s both; 2048 jumanji ~2s, djinn ~3.4s.

## Interpretation

1. **TTT: a wash (±10-17%).** For trivially branchless logic, single-env
   code under vmap and batch-native code compile to essentially the same
   XLA. The house style costs nothing here, but wins nothing either —
   vmap's abstraction is free for simple games. (Honest null result.)
2. **2048: the house style wins 3-19×.** The gap is exactly where the
   styles diverge: a vmapped `lax.while_loop` runs every batch element to
   the batch-worst-case iteration count with divergent-control overhead,
   and a vmapped `lax.switch` lowers to predicated select-all-branches.
   The branchless argsort-compact + unrolled-merge formulation replaces
   all of it with dense vector ops the GPU actually likes. And the djinn
   step does MORE work per step than jumanji's (it pays an honest
   in-step reset-template + mask, ~9 move passes vs their ~5+reset) and
   still wins 19× at training-typical batch sizes.
3. **Gap narrows as B grows** (19× → 3.3× at 8192): the reference's launch
   overhead amortizes and raw FLOPs converge; but at PPO-typical B (64-1024)
   the difference is an order of magnitude.

**Takeaway for the portfolio pitch:** the djinnax engineering conventions
(batch-native, branchless, no while_loop in the hot path) are empirically
worth ~an order of magnitude on control-flow-heavy game logic vs the
mainstream vmap-a-single-env style used by the well-known JAX env suites —
and cost nothing when the logic is simple. This also generalizes: any env
that still carries a long sequential `fori_loop` in its hot path is
running reference-style control flow, and that loop will be its
throughput ceiling.

## Perf ladder v2/v3 (2026-07-19, same-day follow-up)

Next question: what else could be squeezed? Two upgrades, both parity-gated
(all four checks + a new v2≡v3 bit-identical 50-step chain test):

- **v2 (in-place):** argsort compaction → prefix-sum rank-scatter;
  reset-template mask computed **analytically** (single-tile board: legal
  directions from the tile's coordinates — deletes 4 of 9 move passes).
- **v3 (`game2048_lut.py`):** classic bitboard-LUT — a row is 4 nibbles
  = 16 bits, so ALL 65,536 row-moves (+ rewards) are precomputed in numpy
  at import; a move is pack → gather → unpack. 384 KB of tables, sits in
  L2. Exponents saturate at 15 (divergence only beyond the 32,768 tile —
  unreachable). TTT got a 512-entry win-LUT variant too.

One run, RTX 4070 (ratios within-run; absolutes vary with host clocks):

| B | jumanji | djinn v2 | djinn v3 (LUT) | v3 ÷ jumanji |
|---:|---:|---:|---:|---:|
| 64 | 29.7K | 275K | 436K | **14.7×** |
| 1024 | 419K | 2.88M | 8.13M | **19.4×** |
| 8192 | 3.28M | 43.9M | 200.7M | **61.3×** |
| 65536 | 29.6M | 229M | 519.5M | **17.6×** |

Findings:

1. **At B=65536 the LUT engine runs 2048 as fast as tic-tac-toe**
   (~520M env-steps/s, 0.002 µs/env-step) — the game logic is now free;
   both sit on the same platform floor (launch overhead + state bandwidth
   + counter-based RNG). That's the end of this road: a Pallas/custom
   kernel could only chase the same floor, which is why it wasn't built.
2. **The 61× at B=8192** is the sweet spot: jumanji still pays vmapped
   while_loop/switch overhead there while the LUT engine is already
   near-floor.
3. **TTT win-LUT: honest null result** (±5-10%, noise) — the game was
   already launch-bound; kept behind a flag.
4. **Micro-choice caveat:** a same-run A/B of argsort- vs rank-scatter
   compaction (v1 vs v2's kernel) flipped direction across batch sizes —
   sub-2× differences are inside this host's clock/contention noise
   (WSL2, shared GPU). Only the order-of-magnitude LUT gap is a stable
   claim; the bench prints within-run ratios for exactly this reason.

Next rungs if ever needed: packed uint16-row state as the canonical
representation (halves state bytes at the bandwidth floor), cheaper
counter-based RNG for spawn sampling, buffer donation on the scan carry.

## OFFICIAL statistics (2026-07-19 late — supersedes single-run tables above)

Review-driven validation pass: (1) n=5 fresh-process frozen-code sweep
(`sweep_stats.py`), (2) unroll/counter-RNG floor A/B, (3) Sokoban as env #3
(jumanji port, parity-gated over 40 replayed episodes, shared 256-level
fixture injected into both engines via jumanji's own Generator seam).

**Within-run ratios, median [min..max], n=5** (the defensible numbers):

| B | ttt | sokoban | 2048 branchless | 2048 LUT |
|---:|---|---|---|---|
| 64 | 1.1× | 1.7× | 9.4× | **15.1×** [13.9..86.8] |
| 1024 | 1.3× | 1.8× | 8.2× | **25.2×** [22.8..34.8] |
| 8192 | 1.2× | 2.3× | 5.0× | **24.8×** [17.6..61.4] |
| 65536 | 1.4× | 2.3× | 7.5× | **17.7×** [15.9..19.2] |

- **The 61× is retired** — it was a single hot run (it appears inside the
  spread at 8192). The headline is: median 15-25×, tightest at the floor
  (17.7× [15.9..19.2]).
- **The paradigm claim is the spectrum, not one number:** the win scales
  with how much control flow the reference keeps in its hot path —
  ttt (none) ≈ 1.1-1.4×, sokoban (select-based) ≈ 2×, 2048
  (while_loop+switch) ≈ 15-25×.
- Absolutes swung up to ~4× across runs (same code); ratios held far
  tighter — within-run ratios are the only quotable quantity from this
  host.
- **Floor A/B (unroll {1,4} × threefry/unsafe_rbg): unresolvable here** —
  contradictory winners per engine within the same block; both knobs sit
  inside clock noise, unroll=4 also 3-4×'s compile. Defaults kept; flags
  (`--unroll`, `--rng`) remain for controlled hardware.

Methodology, lessons, and the proto-DSL vocabulary live in
`engine-bench/LEARNINGS.md`.

## FINAL official table (2026-07-21 — quiet GPU, n=5 frozen-code sweep,
megakernel included; supersedes all tables above)

Within-run ratios djinn/reference, median [min..max]
(`sweep_official_v2.jsonl`):

| B | ttt | sokoban | 2048 branchless | 2048 LUT | **2048 megakernel** |
|---:|---|---|---|---|---|
| 64 | 1.1× | 1.5× | 34.6× | 56.3× | **98.0×** [48..222] |
| 1024 | 1.1× | 2.2× | 22.6× | 48.1× [45..50] | **213.1×** [101..264] |
| 8192 | 1.2× | 3.0× | 14.8× | 49.4× | **236.6×** [86..381] |
| 65536 | 1.7× | 2.4× | 6.2× | 14.9× | **90.5×** [10..139] |

Absolute (medians): megakernel 849M env-steps/s at B=8192, 1.93B at
65536; ttt 1.1B and sokoban 730M at 65536.

Notes: (1) the quiet-GPU sweep reads HIGHER than the contended-era one
everywhere — contention had been suppressing the djinn engines more than
the references; earlier officials were conservative. (2) The megakernel's
65536 spread includes one 10.4× low draw — its wide intervals are the
honest cost of measuring a sub-millisecond kernel against multi-second
references. (3) The paradigm spectrum stands, now with a fourth point:
ttt ~1×, sokoban ~2-3×, 2048 XLA-style 6-56×, environment-on-chip
~90-240×.

## OFFICIAL v3 (2026-07-21 night — post-R2 tree; supersedes the FINAL
table above for the CURRENT code)

n=5 frozen-code fresh-process sweep at commit `ce8f70f`-era
Development: analytic-mask kernel (E1) + 9-bubble row move (E5) +
analytic XLA masks (E2) + live-output sokoban runner (E3 fix) + strict
fail-closed pipeline (S3). Host held ~2.4GB VRAM throughout (permanent
on this machine). Raw rows: `sweep_official_v3.jsonl` (now with
backend/device/jax/commit provenance per row).

Within-run ratios djinn/reference, median [min..max]:

| B | ttt | sokoban | 2048 branchless | 2048 LUT | **2048 megakernel** |
|---:|---|---|---|---|---|
| 64 | 1.1× | 1.4× | 33.9× [19.7..92.4] | 33.2× | **239.1×** [186..341] |
| 1024 | 1.2× | 1.5× | 25.2× | 31.7× [14.8..105] | **213.3×** [208..447] |
| 8192 | 1.1× | 2.8× [1.3..5.5] | 22.7× | 21.9× | **240.1×** [78..309] |
| 65536 | 1.2× | 1.2× [1.0..1.3] | 8.0× [5.9..9.2] | 13.8× | **56.6×** [50..88]† |

Absolute (medians): megakernel 827M env-steps/s at B=8192, 1.71B at
65536 — **peak observed 2.53B** (both r3/r4 at 65536); branchless 2048
227M, LUT 398M, ttt 846M, sokoban 577M at 65536.

Notes:
1. **†The 65536 megakernel row is a LOWER BOUND — 3/5 runs were
   host-time-sliced.** Per-run ratios are bimodal: 50/57/52× (r0-r2,
   mega at 1.4-1.7B) vs 88/87× (r3-r4, mega at 2.53B). The persistent
   kernel loses ~3× to time-slicing that barely moves the small-op
   engines (same asymmetry that discarded an earlier re-sweep). The
   kernel's real progression is pinned by direct same-process A/Bs:
   current kernel = ×1.51-1.57 (P1) × 1.85 (E1) × 1.13 (E5) ≈ **2.9-3.3×
   the kernel that measured 91×/1.93B on a verified-quiet GPU** —
   receipts p1_orient_ab / e1_megakernel_canmask_ab / e5_rowmove_ab.
   A verified-quiet re-sweep would be expected to read ~3× the v2 mega
   row at 65536; not yet certified, so not claimed.
2. **Sokoban is honestly DOWN from v2** (2.4×→1.2× at 65536): the v2
   runner discarded obs/reward/extras and XLA deleted that work
   (HLO receipt in the E3 ledger entry; measured inflation 1.63×,
   data/e3_soko_dce_ab.jsonl). v3 is the first sweep where both soko
   engines carry equivalent live outputs. The remaining 1.0-1.3× at
   65536 is the honest paradigm delta on reference code that was
   already select-based.
3. **2048 XLA engines are UP where E2 bites**: branchless 14.8×→22.7×
   at 8192 and 6.2×→8.0× at 65536 (the analytic mask deleted 4 of its
   5 per-step move passes). LUT moved less (its mask was already a
   gather) and its small-B medians moved with contention noise — the
   LUT-vs-branchless ordering at B≤8192 is within the run spread, don't
   read a regression into it.
4. 2048/jumanji itself read ~35% faster at 65536 than in v2 (21.3M→
   28.8M) — environmental, both sweeps' reference code identical; the
   ratio moves accordingly. Within-run ratios remain the only numbers
   we compare across time, and even they carry this caveat at the
   largest B.

## Repro

```
cd engine-bench
$VENV/bin/python check_parity.py          # must pass first
LD_PRELOAD=$VENV/.../nvidia/nvjitlink/lib/libnvJitLink.so.12 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
$VENV/bin/python bench_head_to_head.py
```

Requires the gitignored `reference-engines/` clones. Files:
`engine-bench/{djinn_ttt,game2048,check_parity,bench_head_to_head,_ref_paths}.py`.
