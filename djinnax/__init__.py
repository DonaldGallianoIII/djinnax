"""djinnax — batch-native, branchless JAX game environments.

Parity-gated against the reference suites before any timing; see
README.md for the numbers and PORTING_PLAYBOOK.md for the method.

Step signatures are per-env by design (each env exposes exactly the
outputs its game defines — see each class docstring) rather than forced
through a common wrapper: ttt returns state, 2048 returns
(state, reward), sokoban returns its 4-tuple. All envs share the
leading-B batch convention and in-step auto-reset.
"""

from djinnax.distributions import (
    alias_sample, build_alias_table, conditional_categorical,
    geometric_tries,
)
from djinnax.game2048 import Djinn2048, G2048State
from djinnax.game2048_lut import make_game2048_lut, move_left_lut
from djinnax.runtime import NullEnv, build_runner, sample_uniform_legal
from djinnax.sokoban import DjinnSokoban, SokoState
from djinnax.ttt import DjinnTicTacToe, TttState

__version__ = "0.1.0"

__all__ = [
    "Djinn2048", "G2048State", "DjinnSokoban", "SokoState",
    "DjinnTicTacToe", "TttState", "make_game2048_lut", "move_left_lut",
    "build_runner", "sample_uniform_legal", "NullEnv",
    "geometric_tries", "conditional_categorical", "build_alias_table",
    "alias_sample", "run_megakernel_rng",
]


def __getattr__(name):
    # Megakernel entry point is lazy: importing it pulls in
    # jax.experimental.pallas, which top-level `import djinnax` shouldn't.
    if name == "run_megakernel_rng":
        from djinnax.megakernel_rng import run_megakernel_rng
        return run_megakernel_rng
    raise AttributeError(f"module 'djinnax' has no attribute {name!r}")
