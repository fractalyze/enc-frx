# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Poly1305, per RFC 8439 §2.5.

A polynomial evaluated over `GF(2^130 - 5)`. Not cryptographically hard, but the
one part of ChaCha20-Poly1305 that is numerically awkward here, for a reason
worth stating: **this stack has no 64-bit integer lane.** FRX runs with x64
disabled, so a `uint64` request is silently truncated to `uint32`, and
`zk_dtypes.uint128` is a host dtype that no traced array can hold.

That rules out the layout every reference implementation uses — five limbs of 26
bits, whose products are 52 bits and need a 64-bit accumulator. With `uint32`
lanes a product must fit in 32 bits, so the limbs must be at most 16.

**The radix is 2^13 with ten limbs**, and 13 is not a tuning choice: the
reduction `2^130 = 5` is a clean convolution only when the limbs tile 130 bits
exactly, and 130 factors as 13 x 10 (26 x 5 being the layout that does not fit).
Each product is then under 2^26, and the widest accumulator is

    d_0 = h_0 r_0 + 5 (h_1 r_9 + ... + h_9 r_1)

which is at most 46 units of 2^26 — under 2^32 with 28% to spare. Every limb is
carried below 2^13 before it is multiplied, which is what keeps that bound true.

**Parallelism comes from the batch, not from within a message.** Block `i + 1`'s
accumulator depends on block `i`'s, so `B` messages are `B` independent Horner
chains scanned together over the block axis. Unlike ChaCha20, the `scan` is
correct here — the dependency is real. Splitting one long message across
precomputed powers of `r` is the known way to parallelize further; it multiplies
the code by the number of lanes and only pays for long single messages, so it
waits for a benchmark that asks.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

KEY_SIZE = 32
TAG_SIZE = 16
BLOCK_SIZE = 16

_LIMBS = 10
_RADIX_BITS = 13
_LIMB_MASK = np.uint32((1 << _RADIX_BITS) - 1)
# The 17th byte carries bit 128 — the `1` appended to every block — so the byte
# form a limb decomposition reads is one byte wider than the block.
_PADDED_BLOCK = BLOCK_SIZE + 1

# RFC 8439 §2.5.1: the top four bits of every fourth byte, and the bottom two of
# the three that follow the first, are cleared before `r` is used.
_CLAMP = np.array(
    [0xFF, 0xFF, 0xFF, 0x0F, 0xFC, 0xFF, 0xFF, 0x0F]
    + [0xFC, 0xFF, 0xFF, 0x0F, 0xFC, 0xFF, 0xFF, 0x0F],
    dtype=np.uint8,
)


def _limbs(data: Array) -> Array:
    """`uint8 [..., 17]` -> `uint32 [..., 10]`, radix 2^13, little-endian."""
    wide = data.astype(fnp.uint32)
    lanes = []
    for limb in range(_LIMBS):
        start = limb * _RADIX_BITS
        byte, offset = divmod(start, 8)
        value = wide[..., byte] >> np.uint32(offset)
        value = value | (wide[..., byte + 1] << np.uint32(8 - offset))
        if byte + 2 < data.shape[-1]:
            value = value | (wide[..., byte + 2] << np.uint32(16 - offset))
        lanes.append(value & _LIMB_MASK)
    return fnp.stack(lanes, axis=-1)


def _carry(lanes: list[Array]) -> Array:
    """Normalize every limb below 2^13, folding the overflow back through 5.

    Two passes, because the fold puts up to `5 * carry` back into limb 0 and that
    can itself exceed the radix. After the second pass every limb is in range and
    the value is below 2^130, which is what the multiply's bound assumes.
    """
    for _ in range(2):
        carry = fnp.zeros_like(lanes[0])
        for index in range(_LIMBS):
            total = lanes[index] + carry
            carry = total >> np.uint32(_RADIX_BITS)
            lanes[index] = total & _LIMB_MASK
        lanes[0] = lanes[0] + carry * np.uint32(5)
    return fnp.stack(lanes, axis=-1)


def _add(left: Array, right: Array) -> Array:
    return _carry([left[..., i] + right[..., i] for i in range(_LIMBS)])


def _mul(accumulator: Array, r: Array, r5: Array) -> Array:
    """`accumulator * r` reduced mod 2^130 - 5.

    `r5` is `5 * r`, precomputed because every term that wraps past limb 9 picks
    up that factor — the whole reason the radix tiles 130 bits exactly.
    """
    products = []
    for index in range(_LIMBS):
        total = accumulator[..., 0] * r[..., index]
        for j in range(1, index + 1):
            total = total + accumulator[..., j] * r[..., index - j]
        for j in range(index + 1, _LIMBS):
            total = total + accumulator[..., j] * r5[..., index + _LIMBS - j]
        products.append(total)
    return _carry(products)


def _reduce_fully(accumulator: Array) -> Array:
    """Subtract the modulus once if the accumulator is at or above it.

    The value is already below 2^130 and the modulus is 2^130 - 5, so one
    conditional subtraction suffices. It is expressed as "add 5 and keep the
    result when it carries past bit 130", which is the same test without a
    borrow.
    """
    lanes = [accumulator[..., i] for i in range(_LIMBS)]
    candidate = []
    carry = np.uint32(5)
    for index in range(_LIMBS):
        total = lanes[index] + carry
        carry = total >> np.uint32(_RADIX_BITS)
        candidate.append(total & _LIMB_MASK)
    # A carry out of the top limb means the addition passed 2^130, so the
    # candidate is the reduced value. Selected arithmetically, per entry.
    use_candidate = (carry != 0)[..., None]
    return fnp.where(use_candidate, fnp.stack(candidate, axis=-1), accumulator)


def _to_bytes(accumulator: Array) -> Array:
    """`uint32 [..., 10]` radix-2^13 limbs -> the low 16 bytes, little-endian."""
    lanes = []
    for byte in range(TAG_SIZE):
        start = byte * 8
        limb, offset = divmod(start, _RADIX_BITS)
        value = accumulator[..., limb] >> np.uint32(offset)
        if limb + 1 < _LIMBS:
            value = value | (
                accumulator[..., limb + 1] << np.uint32(_RADIX_BITS - offset)
            )
        lanes.append(value & np.uint32(0xFF))
    return fnp.stack(lanes, axis=-1)


def _add_mod_2_128(left: Array, right: Array) -> Array:
    """Little-endian 16-byte addition, discarding the carry out of the top."""
    carry = fnp.zeros_like(left[..., 0])
    lanes = []
    for byte in range(TAG_SIZE):
        total = left[..., byte] + right[..., byte] + carry
        lanes.append(total & np.uint32(0xFF))
        carry = total >> np.uint32(8)
    return fnp.stack(lanes, axis=-1).astype(fnp.uint8)


def _blocks(message: Array) -> Array:
    """`uint8 [B, N]` -> `uint8 [blocks, B, 17]`, each block's high bit set.

    The appended `1` is RFC 8439 §2.5's, and where it lands depends on the block:
    past the sixteenth byte for a full block, and immediately after the message
    for a short final one. `N` is static, so the positions are a trace-time
    constant added to zero padding rather than an indexed write.
    """
    length = message.shape[-1]
    blocks = -(-length // BLOCK_SIZE)
    padded = fnp.pad(
        message,
        [(0, 0)] * (message.ndim - 1) + [(0, blocks * BLOCK_SIZE - length)],
    )
    grouped = padded.reshape(*message.shape[:-1], blocks, BLOCK_SIZE)
    grouped = fnp.concatenate(
        [grouped, fnp.zeros((*grouped.shape[:-1], 1), dtype=fnp.uint8)], axis=-1
    )
    high = np.zeros((blocks, _PADDED_BLOCK), dtype=np.uint8)
    for index in range(blocks):
        used = min(BLOCK_SIZE, length - index * BLOCK_SIZE)
        high[index, used] = 1
    return fnp.moveaxis(grouped + fnp.asarray(high), -2, 0)


def mac(key: ArrayLike, message: ArrayLike) -> Array:
    """`uint8 [B, 32]`, `uint8 [B, N]` -> `uint8 [B, 16]`.

    The key is `r || s`: `r` clamped per RFC 8439 §2.5.1 and used as the
    polynomial's point, `s` added to the result mod 2^128. One-time — a key
    reused across two messages loses the authenticator.
    """
    key = fnp.asarray(key, dtype=fnp.uint8)
    message = fnp.asarray(message, dtype=fnp.uint8)

    r_bytes = key[..., :BLOCK_SIZE] & fnp.asarray(_CLAMP)
    r = _limbs(fnp.pad(r_bytes, [(0, 0)] * (r_bytes.ndim - 1) + [(0, 1)]))
    r5 = r * np.uint32(5)

    accumulator = fnp.zeros((*message.shape[:-1], _LIMBS), dtype=fnp.uint32)
    if message.shape[-1]:
        blocked = _blocks(message)

        def absorb(state: Array, block: Array) -> tuple[Array, None]:
            return _mul(_add(state, _limbs(block)), r, r5), None

        accumulator, _ = frx.lax.scan(absorb, accumulator, blocked)

    tag = _to_bytes(_reduce_fully(accumulator))
    return _add_mod_2_128(tag, key[..., BLOCK_SIZE:].astype(fnp.uint32))
