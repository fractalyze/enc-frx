# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""GHASH's field arithmetic straight from SP 800-38D §6.3, in plain Python.

Deliberately the definitional algorithm — shift and conditionally XOR, one bit at
a time, over the standard's own bit order — rather than anything clever. Its
whole purpose is to be independent of the traced implementation, which reaches
the same result by reversing the bit order and handing the work to a native
GF(2^128) dtype. If both agree, the reversal is right.

TEST ONLY. Never re-exported from the package.
"""

from __future__ import annotations

# SP 800-38D §6.3: R = 11100001 || 0^120, the reduction constant in the
# standard's bit order.
_R = 0xE1 << 120
_MASK = (1 << 128) - 1


def _bits(value: int) -> list[int]:
    """The block's bits in SP 800-38D order: `b_0` first, most significant."""
    return [(value >> (127 - index)) & 1 for index in range(128)]


def multiply(left: bytes, right: bytes) -> bytes:
    """SP 800-38D §6.3's Algorithm 1, on 16-byte blocks."""
    x = int.from_bytes(left, "big")
    y = int.from_bytes(right, "big")
    product = 0
    value = y
    for bit in _bits(x):
        if bit:
            product ^= value
        if value & 1:
            value = (value >> 1) ^ _R
        else:
            value >>= 1
    return (product & _MASK).to_bytes(16, "big")


def ghash(subkey: bytes, blocks: bytes) -> bytes:
    """SP 800-38D §6.4, over a byte string that is a whole number of blocks."""
    assert len(blocks) % 16 == 0, "GHASH takes whole blocks"
    accumulator = bytes(16)
    for start in range(0, len(blocks), 16):
        mixed = bytes(
            a ^ b for a, b in zip(accumulator, blocks[start : start + 16], strict=True)
        )
        accumulator = multiply(mixed, subkey)
    return accumulator
