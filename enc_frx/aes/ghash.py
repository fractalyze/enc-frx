# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""GHASH, per NIST SP 800-38D §6.3-6.4.

A polynomial evaluated over GF(2^128), and the part of GCM that is hardest in C —
carry-less multiplication and reduction by `x^128 + x^7 + x^2 + x + 1`. Here it is
one multiply: **`zk_dtypes.binary_field_ghash` is that field**, reduction
polynomial included.

**The bit order is the whole subtlety.** SP 800-38D §6.1 numbers a block's bits
`b_0 … b_127` and reads them as `b_0 + b_1·x + … + b_127·x^127`, so the *first*
bit of the *first* byte is the constant term. The dtype uses the natural basis,
where bit `i` of the integer is the coefficient of `x^i`. Measured rather than
assumed: `x^127 · x` reduces to `0x87`, which fixes both the polynomial and the
basis (`enc_frx/testing/binary_fields_test.py`).

The two differ by reversing the whole 128-bit string, which factors into
something cheap: **reverse the bits within each byte, then read the sixteen bytes
little-endian**. Wire bytes cannot be reinterpreted as this dtype directly, and
skipping the reversal yields a wrong tag for every input while looking like a
field bug rather than a convention bug.

The reversal is bit arithmetic on `uint8`, not on the field: shifts and masks do
not lower on the field dtype, so `view` is the only bridge and everything else
happens on the byte side.

**Parallelism is the batch, not the message.** Block `i + 1`'s accumulator
depends on block `i`'s, so `B` messages are `B` independent Horner chains scanned
together — the same shape as Poly1305, and for the same reason. Precomputed
powers of `H` would parallelize within one message; that waits for a benchmark.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from frx.typing import ArrayLike

BLOCK_SIZE = 16

GF128 = zk_dtypes.binary_field_ghash

# The standard byte-reversal network: swap nibbles, then pairs, then adjacent
# bits. Three masked shift-pairs, no table.
_BIT_REVERSE_STEPS = ((4, 0xF0, 0x0F), (2, 0xCC, 0x33), (1, 0xAA, 0x55))


def _reverse_bits(data: Array) -> Array:
    """Reverse the bit order within every byte."""
    value = fnp.asarray(data, dtype=fnp.uint8)
    for shift, high, low in _BIT_REVERSE_STEPS:
        value = ((value & np.uint8(high)) >> np.uint8(shift)) | (
            (value & np.uint8(low)) << np.uint8(shift)
        )
    return value


def to_field(block: ArrayLike) -> Array:
    """`uint8 [..., 16]` in GCM's bit order -> one field element per block."""
    return _reverse_bits(block).view(GF128)[..., 0]


def from_field(value: Array) -> Array:
    """The inverse of `to_field`: one field element per block -> `uint8 [..., 16]`."""
    return _reverse_bits(value[..., None].view(fnp.uint8))


def ghash(subkey: ArrayLike, blocks: ArrayLike) -> Array:
    """`uint8 [B, 16]`, `uint8 [B, L, 16]` -> `uint8 [B, 16]`.

    SP 800-38D §6.4: `Y_i = (Y_{i-1} + X_i) · H`, with `Y_0 = 0`. Addition in
    this field is XOR, so `+` is the standard's `⊕` unchanged.
    """
    subkey = fnp.asarray(subkey, dtype=fnp.uint8)
    blocks = fnp.asarray(blocks, dtype=fnp.uint8)
    factor = to_field(subkey)
    initial = fnp.zeros(blocks.shape[:-2], dtype=GF128)

    def absorb(accumulator: Array, block: Array) -> tuple[Array, None]:
        return (accumulator + to_field(block)) * factor, None

    accumulated, _ = frx.lax.scan(absorb, initial, fnp.moveaxis(blocks, -2, 0))
    return from_field(accumulated)


def length_block(first: int, second: int) -> Array:
    """SP 800-38D's trailing block: two 64-bit big-endian bit counts.

    Both lengths are static, so this is a trace-time constant rather than
    anything derived from the data.
    """
    return fnp.asarray(
        np.frombuffer(
            (first * 8).to_bytes(8, "big") + (second * 8).to_bytes(8, "big"),
            dtype=np.uint8,
        )
    )


def pad_to_blocks(data: Array) -> Array:
    """`uint8 [..., N]` -> `uint8 [..., ceil(N/16), 16]`, zero-padded.

    `N` is static, so the padding is a constant-shaped concatenation.
    """
    length = data.shape[-1]
    blocks = -(-length // BLOCK_SIZE)
    padded = fnp.pad(
        data, [(0, 0)] * (data.ndim - 1) + [(0, blocks * BLOCK_SIZE - length)]
    )
    return padded.reshape(*data.shape[:-1], blocks, BLOCK_SIZE)
