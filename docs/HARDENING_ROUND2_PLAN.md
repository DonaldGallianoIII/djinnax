# Hardening round 2 — tuning, curves, null-result revisits, CI-ification

> Historical plan document, imported from the pre-release tree; file
> paths and test counts reflect that point in time.

**Status:** items 1-4 COMPLETE (`bbd80c2`); item 5 sweep running.
Results: (1) tuning null — block=128/default warps already optimal, all
alternatives regress or straddle 1.0; (2) amortization 189M→1.2B→2.0B
env-steps/s at 8/64/1024 steps/launch, compile flat ~30s, **chunk tax
2.2× (k=2) / 3.9× (k=8)** — chunk coarsely; (3) unsafe_rbg RESOLVED as
a 0.63-0.67× regression (rejected), unroll=4 = +56% on launch-bound ttt
but neutral on 2048 (defaults kept, engine-dependent); (4) pytest suite:
GPU 16/16, CPU 11 passed/5 skipped — CI contract green both ways.

## 1. Megakernel tuning sweep (BLOCK × num_warps)

The megakernel has never been *tuned*, only written (BLOCK=128 was a
guess; Triton's num_warps/num_stages left at defaults). Work:
- Parametrize `block` in `run_megakernel_rng` (kernel factory closes over
  it; the B%block guard follows it).
- Round-robin interleaved sweep: block ∈ {64, 128, 256} × num_warps ∈
  {2, 4, 8} at B=8192 and 65536; every config timed once per round so
  clock drift hits all configs equally; report median ratio vs the
  current default config with [min..max].
- Adopt a new default only if its interval clears 1.0; else record null.
- Parity re-run at the adopted config (block changes grid coverage).

## 2. N_STEPS amortization curve

The training loop needs to know what chunking costs. Work:
- Time `n_steps ∈ {8, 32, 64, 256, 1024}` at B=8192 (env-steps/s,
  µs/env-step, compile time per variant — fori shouldn't scale compile
  with n_steps; verify).
- Chunk-overhead measurement: k×(64/k)-step chained launches vs 1×64
  monolithic for k ∈ {1, 2, 8}, interleaved. The delta is the per-launch
  tax a training loop pays to insert policy updates between chunks.

## 3. Quiet-GPU revisit of the "unresolvable" null results

The unroll/RNG-impl verdicts were rendered under clock noise (and some
under disclosed GPU contention). With a quiet GPU + round-robin interleaving:
- lax.scan unroll {1, 4} on the XLA runners (ttt/djinn, 2048/djinn-lut)
  at B=8192.
- PRNG impl {threefry2x32, unsafe_rbg} same engines/batch.
- Outcome either resolves to a direction (adopt if interval clears 1) or
  re-confirms the null with a cleaner interval — both are results;
  LEARNINGS gets the update either way.

## 4. Pytest-ification (CI readiness for the standalone repo)

The check_* scripts are assert-based but not collectable. Work:
- `engine-bench/tests/` with a conftest.py (sys.path shim) and thin
  pytest wrappers: test_game_parity.py (check_parity.py suite),
  test_megakernel.py (check_megakernel battery; kernel-touching tests
  `skipif backend != "gpu"` — chain-link, analytic-mask, and RNG tests
  run on CPU and become the GitHub-Actions core), test_rng.py.
- One full `pytest engine-bench/tests` run on GPU must pass; one
  CPU-only run must pass-with-skips (that's the CI contract).

## 5. Megakernel into the standing harness

- `make_2048_megakernel` runner in bench_head_to_head (carry = board +
  score + t_offset so chained calls continue the RNG stream), registered
  in the engine list and RATIO_PAIRS ("2048-mega" vs jumanji).
- Fresh official n=5 frozen-code sweep including it → the final table
  (and first quiet-GPU sweep, so ALL engines get cleaner intervals).
- Docs: head-to-head doc + LEARNINGS get the new official numbers.

## Success criteria

- 1-3: every claim/adoption carries an interleaved interval; nulls
  recorded as nulls.
- 4: green GPU run; green-with-skips CPU run.
- 5: sweep table includes megakernel; docs updated; code frozen during
  the sweep (rule 9 discipline — no edits while it runs).
