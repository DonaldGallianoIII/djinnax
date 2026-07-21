"""Pytest shim: make engine-bench modules importable from tests/.

CI contract: on GPU every test runs; on CPU the kernel-touching tests
skip and the rest (game parity, chain link, RNG batteries) still gate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
