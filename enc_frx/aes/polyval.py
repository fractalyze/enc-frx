# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""POLYVAL, per RFC 8452 §3 — GHASH's little-endian sibling, over the same
native field dtype.

POLYVAL and GHASH are the same GF(2^128) dot product under conjugate
conventions, and RFC 8452 Appendix A states the bridge exactly:

    POLYVAL(H, X_1..X_n) =
        ByteReverse(GHASH(mulX_GHASH(ByteReverse(H)), ByteReverse(X_i)...))

This module *is* that identity — two byte reversals per call plus one
multiply-by-x at key setup — rather than a second field implementation,
because the identity is the standard's own and the GHASH path it reuses
(`zk_dtypes.binary_field_ghash`, one multiply per block) is already vetted.
The cost of the detour is a reversed copy of the blocks; a native POLYVAL
ordering would be a measured optimization, not a different answer.

Like GHASH, the chain is a Horner scan: parallel across the batch, sequential
within a message.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aes import ghash

BLOCK_SIZE = 16

# The GHASH-convention encoding of the field element x: coefficient of x^1 is
# bit 1, and GCM numbers bits from the most significant bit of byte 0.
_X_GHASH = np.zeros(BLOCK_SIZE, dtype=np.uint8)
_X_GHASH[0] = 0x40


def _byte_reverse(data: Array) -> Array:
    return data[..., ::-1]


def mulx_ghash(block: ArrayLike) -> Array:
    """Multiply a GHASH-convention element by x: uint8 `[..., 16]` in, same
    out (RFC 8452 Appendix A's `mulX_GHASH`)."""
    block = fnp.asarray(block, dtype=fnp.uint8)
    return ghash.from_field(ghash.to_field(block) * ghash.to_field(_X_GHASH))


def polyval(subkey: ArrayLike, blocks: ArrayLike) -> Array:
    """`uint8 [B, 16]`, `uint8 [B, L, 16]` -> `uint8 [B, 16]`, per RFC 8452
    §3 — computed through the Appendix A identity above."""
    subkey = fnp.asarray(subkey, dtype=fnp.uint8)
    blocks = fnp.asarray(blocks, dtype=fnp.uint8)
    bridged = mulx_ghash(_byte_reverse(subkey))
    return _byte_reverse(ghash.ghash(bridged, _byte_reverse(blocks)))


def length_block(aad_bytes: int, plaintext_bytes: int) -> Array:
    """RFC 8452 §5's trailing block: two 64-bit *little*-endian bit counts —
    the endianness GHASH's `length_block` flips. Both lengths are static, so
    this is a trace-time constant."""
    return fnp.asarray(
        np.frombuffer(
            (aad_bytes * 8).to_bytes(8, "little")
            + (plaintext_bytes * 8).to_bytes(8, "little"),
            dtype=np.uint8,
        )
    )
