# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""GF(2^255 - 19) on uint32 lanes — the field under the X25519 ladder.

**The layout is 16 limbs of radix 2^16**, little-endian, each riding a uint32
lane, and neither number is a tuning choice. This stack has no 64-bit integer
lane (the constraint `chacha/poly1305.py` states), so a limb product must fit
uint32, capping the radix at 16 bits. Poly1305's cleaner move — a radix whose
products *and* column sums fit, so the reduction is one convolution — has no
uint32 analogue at this modulus: 255 = 3·5·17 offers radix 17 (products 2^34,
overflow) or radix 15 (the wrapped columns scale by 19 into 2^38, overflow).
So the limbs tile 2^256 instead, reduction wraps through `2^256 ≡ 38 (mod p)`,
and every 32-bit product is split into 16-bit halves before accumulation —
the split is what buys back the headroom the modulus refuses to give.

**Accumulator bound** (asserted by `field_test.test_accumulator_bound_holds`):
a raw column of the product sums at most 16 lo-halves and 16 hi-halves, each
below 2^16, so a column is below `32·(2^16 - 1) < 2^21`; the wrap adds
`38·(a column)`, keeping every accumulator below `2^21 + 38·2^21 < 2^26.4`,
inside uint32 with over five bits to spare. The carry sweep (`_carry`) runs
three chain-and-fold passes because the value can sit within 38 of a 2^256
multiple after two — the third pass is what makes "every limb below 2^16" an
invariant rather than a probability.

Elements are uint32 `[..., 16]` with every limb below 2^16 ("carried"), values
in `[0, 2^256)` — reduced only enough to keep the bound, canonicalized once at
`to_bytes`. All arithmetic is data-independent: fixed loops, no comparisons on
values, `select`-free except the canonical subtract at the boundary.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array

LIMBS = 16
RADIX_BITS = 16
_MASK = np.uint32((1 << RADIX_BITS) - 1)
# 2^256 = 2·(2^255 - 19) + 38, so the limb one past the top folds back as 38.
_WRAP = np.uint32(38)

# p = 2^255 - 19 in the limb layout, for the canonical subtract at to_bytes.
_P_LIMBS = np.array([0xFFED] + [0xFFFF] * 14 + [0x7FFF], dtype=np.uint32)
# 4p = [2^17 - 76, (2^17 - 2)·x^1..15]: every limb at least 2^16, so
# `a + 4p - b` never underflows a lane when a and b are carried. 2p's limbs
# (0xFFDA, 0xFFFF...) sit below some b limbs, which is why the doubling.
_4P_LIMBS = np.array([0x1FFB4] + [0x1FFFE] * 15, dtype=np.uint32)


def zero(shape: tuple[int, ...]) -> Array:
    return fnp.zeros((*shape, LIMBS), dtype=fnp.uint32)


def one(shape: tuple[int, ...]) -> Array:
    return fnp.concatenate(
        [
            fnp.ones((*shape, 1), dtype=fnp.uint32),
            fnp.zeros((*shape, LIMBS - 1), dtype=fnp.uint32),
        ],
        axis=-1,
    )


def from_bytes(data: Array) -> Array:
    """uint8 `[..., 32]` little-endian -> carried limbs uint32 `[..., 16]`."""
    wide = data.astype(fnp.uint32)
    return wide[..., 0::2] | (wide[..., 1::2] << np.uint32(8))


def _carry(lanes: list[Array]) -> Array:
    """Bring every limb below 2^16, folding overflow back through 38.

    Three chain-and-fold passes. After one pass the fold can leave limb 0 above
    the radix; after two the value can still sit in `[2^256, 2^256 + 38)`,
    whose chain carries out one more wrap. The third pass starts from a value
    below `2^256`, so its chain cannot carry out and its fold adds zero —
    which is what makes the carried invariant unconditional.
    """
    for _ in range(3):
        carry = fnp.zeros_like(lanes[0])
        for index in range(LIMBS):
            total = lanes[index] + carry
            carry = total >> np.uint32(RADIX_BITS)
            lanes[index] = total & _MASK
        lanes[0] = lanes[0] + carry * _WRAP
    return fnp.stack(lanes, axis=-1)


def add(left: Array, right: Array) -> Array:
    return _carry([left[..., i] + right[..., i] for i in range(LIMBS)])


def sub(left: Array, right: Array) -> Array:
    """`left - right` via `left + 4p - right`, which never underflows a lane
    (see `_4P_LIMBS`) and never exceeds 2^18 in one, then a carry sweep."""
    return _carry([left[..., i] + _4P_LIMBS[i] - right[..., i] for i in range(LIMBS)])


def mul(left: Array, right: Array) -> Array:
    """Schoolbook product with 16-bit half accumulation, then the 38-wrap.

    The outer product's entries are full 32-bit values (16-bit limbs, so they
    just fit); each is split into halves and the halves land one column apart.
    Columns then hold sums of sub-2^16 terms — the accumulator bound in the
    module docstring — and columns 16..31 fold into 0..15 through 38.
    """
    product = left[..., :, None] * right[..., None, :]
    lo = product & _MASK
    hi = product >> np.uint32(RADIX_BITS)

    columns: list[Array] = []
    for k in range(2 * LIMBS):
        terms = [
            lo[..., i, k - i]
            for i in range(max(0, k - LIMBS + 1), min(LIMBS - 1, k) + 1)
        ] + [
            hi[..., i, k - 1 - i]
            for i in range(max(0, k - LIMBS), min(LIMBS - 1, k - 1) + 1)
        ]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        columns.append(total)

    folded = [columns[k] + columns[k + LIMBS] * _WRAP for k in range(LIMBS)]
    return _carry(folded)


def square(element: Array) -> Array:
    return mul(element, element)


def _square_step(_: Array, acc: Array) -> Array:
    """The repeated-squaring loop body, at module level so `fori_loop`'s
    lowering cache can hit — frx keys that cache on the body function's
    identity (the gotcha `aes/ghash._absorb` measured), and a lambda minted
    per call can never match."""
    return square(acc)


def invert(element: Array) -> Array:
    """`element^(p - 2)` by the standard 254-squaring addition chain for
    `2^255 - 21` — a fixed sequence, nothing data-dependent."""

    def pow2k(value: Array, squarings: int) -> Array:
        return frx.lax.fori_loop(0, squarings, _square_step, value)

    z2 = square(element)
    z9 = mul(pow2k(z2, 2), element)
    z11 = mul(z9, z2)
    z_5_0 = mul(square(z11), z9)
    z_10_0 = mul(pow2k(z_5_0, 5), z_5_0)
    z_20_0 = mul(pow2k(z_10_0, 10), z_10_0)
    z_40_0 = mul(pow2k(z_20_0, 20), z_20_0)
    z_50_0 = mul(pow2k(z_40_0, 10), z_10_0)
    z_100_0 = mul(pow2k(z_50_0, 50), z_50_0)
    z_200_0 = mul(pow2k(z_100_0, 100), z_100_0)
    z_250_0 = mul(pow2k(z_200_0, 50), z_50_0)
    return mul(pow2k(z_250_0, 5), z11)


def _sub_p_if_ge(limbs: Array) -> Array:
    """One conditional canonical step: `limbs - p` when `limbs >= p`, else
    unchanged — a borrow chain and an arithmetic select, no branch."""
    lanes = []
    borrow = fnp.zeros_like(limbs[..., 0])
    for index in range(LIMBS):
        diff = limbs[..., index] - _P_LIMBS[index] - borrow
        borrow = diff >> np.uint32(31)
        lanes.append(diff & _MASK)
    keep = borrow  # 1 -> the value was below p; keep it
    mask = fnp.uint32(0) - keep
    reduced = fnp.stack(lanes, axis=-1)
    return (limbs & mask[..., None]) | (reduced & ~mask[..., None])


def to_bytes(limbs: Array) -> Array:
    """Carried limbs -> canonical little-endian uint8 `[..., 32]`.

    A carried value is below 2^256 < 3p, so two conditional subtracts reach
    the canonical residue.
    """
    canonical = _sub_p_if_ge(_sub_p_if_ge(limbs))
    lo = (canonical & np.uint32(0xFF)).astype(fnp.uint8)
    hi = (canonical >> np.uint32(8)).astype(fnp.uint8)
    return fnp.stack([lo, hi], axis=-1).reshape(*limbs.shape[:-1], 2 * LIMBS)
