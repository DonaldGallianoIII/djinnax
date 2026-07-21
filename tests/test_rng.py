"""Counter-RNG statistical batteries (CPU-runnable; CI core)."""

import checks.check_megakernel as cm
from djinnax.megakernel_rng import check_rng_quality


def test_rng_quality_battery():
    check_rng_quality(n=1 << 18)


def test_rng_deep_correlations():
    cm.check_rng_deep(n=1 << 16)
