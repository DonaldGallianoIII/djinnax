"""2048 perf v3 — bitboard row-LUT kernel (classic fast-2048 technique).

A row is 4 exponent nibbles = 16 bits = 65,536 possible rows, so the entire
compact+merge+compact computation (and its reward) is precomputed in numpy
at import into three 65,536-entry tables. A row move at runtime is then:
pack nibbles (int dot) -> gather -> unpack (shifts) — no sort, no scatter,
no merge arithmetic. Board orientation reuses game2048's verified
_orient; everything else (spawn, analytic reset mask, state, semantics)
comes from game2048, so parity gates cover this variant identically.

Saturation note: exponents cap at 15 (nibble limit) — merging two 15s
yields 15, diverging from jumanji's unbounded int32 ONLY beyond the 32,768
tile, which is unreachable in any realistic (or random) play.

LUT footprint: moved 128 KB (uint16) + reward 256 KB (f32) + changed 8 KB
(bool) — sits in L2 on any modern GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from djinnax.game2048 import Djinn2048


def _move_row_np(row):
    """Reference python move-left on one 4-nibble row (saturating at 15)."""
    tiles = [t for t in row if t != 0]
    out, reward, i = [], 0.0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = min(tiles[i] + 1, 15)
            out.append(merged)
            reward += float(2 ** merged)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (4 - len(out))
    return out, reward


def _build_luts():
    moved = np.zeros(65536, dtype=np.uint16)
    reward = np.zeros(65536, dtype=np.float32)
    for code in range(65536):
        row = [(code >> (4 * k)) & 15 for k in range(4)]
        out, r = _move_row_np(row)
        moved[code] = sum(out[k] << (4 * k) for k in range(4))
        reward[code] = r
    return moved, reward


_MOVED_NP, _REWARD_NP = _build_luts()
MOVED_LUT = jnp.asarray(_MOVED_NP)      # (65536,) uint16
REWARD_LUT = jnp.asarray(_REWARD_NP)    # (65536,) float32

_PACK = jnp.asarray([1, 16, 256, 4096], dtype=jnp.int32)      # nibble weights
_SHIFTS = jnp.asarray([0, 4, 8, 12], dtype=jnp.uint16)


def move_left_lut(board: jax.Array):
    """board (..., 4, 4) int8 -> (moved, reward (...,)). One gather per row."""
    codes = (board.astype(jnp.int32) * _PACK).sum(axis=-1)     # (..., 4)
    new_codes = MOVED_LUT[codes]                               # (..., 4) uint16
    reward = REWARD_LUT[codes].sum(axis=-1)                    # (...,)
    moved = ((new_codes[..., None] >> _SHIFTS) & jnp.uint16(15)).astype(jnp.int8)
    return moved, reward


def make_game2048_lut() -> Djinn2048:
    return Djinn2048(move_left_fn=move_left_lut)
