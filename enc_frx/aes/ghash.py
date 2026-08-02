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


def _absorb(
    carry: tuple[Array, Array], block: Array
) -> tuple[tuple[Array, Array], None]:
    """One Horner step, with `H` carried rather than captured.

    `H` rides the carry — where it is loop-invariant and costs nothing — instead
    of being closed over, because `frx.lax.scan` caches its lowering on the body
    function's *identity*. A body defined inside `ghash` is a fresh object per
    call, so every call missed that cache: measured at 137 ms and 2.1 MB of
    retained lowering per call, against 0.4 ms and nothing once hoisted. It never
    looked like a bug — the answers were right and one call is not slow.
    """
    accumulator, factor = carry
    return ((accumulator + to_field(block)) * factor, factor), None


def ghash(subkey: ArrayLike, blocks: ArrayLike) -> Array:
    """`uint8 [B, 16]`, `uint8 [B, L, 16]` -> `uint8 [B, 16]`.

    SP 800-38D §6.4: `Y_i = (Y_{i-1} + X_i) · H`, with `Y_0 = 0`. Addition in
    this field is XOR, so `+` is the standard's `⊕` unchanged.
    """
    subkey = fnp.asarray(subkey, dtype=fnp.uint8)
    blocks = fnp.asarray(blocks, dtype=fnp.uint8)
    factor = to_field(subkey)
    initial = fnp.zeros(blocks.shape[:-2], dtype=GF128)

    (accumulated, _), _ = frx.lax.scan(
        _absorb, (initial, factor), fnp.moveaxis(blocks, -2, 0)
    )
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


def hash_input(first: ArrayLike | None, second: ArrayLike) -> Array:
    """Two byte strings and their lengths, as whole blocks ready to hash.

    `uint8 [B, M]` or None, `uint8 [B, N]` -> `uint8 [B, L, 16]`, laid out as
    `first ‖ 0^v ‖ second ‖ 0^u ‖ [len(first)]_64 ‖ [len(second)]_64`.

    Both places SP 800-38D hashes have this shape. §7.1 step 5 hashes the
    associated data against the ciphertext, which is the two-part case. §7.1's
    `J_0` for an IV that is not 96 bits is written `IV ‖ pad ‖ 0^64 ‖
    [len(IV)]_64` — the one-part case, since `0^64 ‖ [len(IV)]_64` is exactly a
    length block whose first count is zero.

    Both lengths are static, so the padding and the trailing block are
    constant-shaped rather than derived from the data.
    """
    second = fnp.asarray(second, dtype=fnp.uint8)
    parts = []
    first_length = 0
    if first is not None:
        first = fnp.asarray(first, dtype=fnp.uint8)
        first_length = first.shape[-1]
        parts.append(pad_to_blocks(first))
    parts.append(pad_to_blocks(second))
    parts.append(
        fnp.broadcast_to(
            length_block(first_length, second.shape[-1]),
            (*second.shape[:-1], 1, BLOCK_SIZE),
        )
    )
    return fnp.concatenate(parts, axis=-2)
