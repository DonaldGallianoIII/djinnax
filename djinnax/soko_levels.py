"""Shared fixed Sokoban level set — identical for both engines by construction.

256 seeded 10x10 levels in jumanji's encoding: fixed_grid {0 empty, 1 wall,
2 target}, variable_grid {0 empty, 3 agent, 4 box}. Border walls + sparse
interior walls + 4 targets + 4 boxes + 1 agent on distinct free cells.
Random-play throughput/parity doesn't require solvability (episodes end at
the 120-step time limit under a random policy, same as unfiltered Boxoban).
"""

from __future__ import annotations

import numpy as np

GRID = 10
N_LEVELS = 256
N_BOXES = 4

EMPTY, WALL, TARGET, AGENT, BOX = 0, 1, 2, 3, 4


def _build_levels(seed: int = 7):
    rng = np.random.default_rng(seed)
    fixed = np.zeros((N_LEVELS, GRID, GRID), dtype=np.uint8)
    variable = np.zeros((N_LEVELS, GRID, GRID), dtype=np.uint8)
    for i in range(N_LEVELS):
        f = np.zeros((GRID, GRID), dtype=np.uint8)
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = WALL
        interior = [(y, x) for y in range(1, GRID - 1) for x in range(1, GRID - 1)]
        rng.shuffle(interior)
        n_walls = int(rng.integers(8, 14))
        walls, rest = interior[:n_walls], interior[n_walls:]
        for y, x in walls:
            f[y, x] = WALL
        targets, rest = rest[:N_BOXES], rest[N_BOXES:]
        for y, x in targets:
            f[y, x] = TARGET
        v = np.zeros((GRID, GRID), dtype=np.uint8)
        # boxes off-target and off-border-corner pockets; next free cells
        boxes, rest = rest[:N_BOXES], rest[N_BOXES:]
        for y, x in boxes:
            v[y, x] = BOX
        ay, ax = rest[0]
        v[ay, ax] = AGENT
        fixed[i], variable[i] = f, v
    return fixed, variable


FIXED_LEVELS, VARIABLE_LEVELS = _build_levels()
AGENT_YX = np.stack(
    [np.argwhere(VARIABLE_LEVELS[i] == AGENT)[0] for i in range(N_LEVELS)]
).astype(np.int32)                                              # (N, 2)
