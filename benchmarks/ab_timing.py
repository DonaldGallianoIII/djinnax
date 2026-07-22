"""Shared A/B timing core (audit S4): counterbalanced ABBA rounds.

The previous per-round order was always A-then-B — interleaved, but
position-biased: whichever variant runs second in a round sees warmer
clocks/caches every single round, so a fixed-order protocol folds a
constant position effect into every sub-2× ratio. Each round here times
A,B,B,A and forms ONE paired ratio from the summed times, cancelling
the position effect within the round.

Aggregation stays the house pattern (LEARNINGS §3): median [min..max]
of per-round ratios within a process; n fresh processes for anything
official.
"""

from __future__ import annotations

import time

import jax


def _t(thunk):
    t0 = time.perf_counter()
    jax.block_until_ready(thunk())
    return time.perf_counter() - t0


def abba_ratios(run_a, run_b, rounds):
    """Per-round paired ratios t_A / t_B, ABBA-counterbalanced.

    run_a / run_b: zero-arg callables performing one timed invocation
    (close over their own state; return the value to block on).
    Returns (ratios, last_b_seconds) — last_b_seconds is the final B
    timing, for scripts that report absolute B throughput.
    """
    ratios, tb_last = [], None
    for _ in range(rounds):
        ta1 = _t(run_a)
        tb1 = _t(run_b)
        tb2 = _t(run_b)
        ta2 = _t(run_a)
        ratios.append((ta1 + ta2) / (tb1 + tb2))
        tb_last = tb2
    return ratios, tb_last
