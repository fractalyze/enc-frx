# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""X25519, per RFC 7748 §5 — the Montgomery ladder over GF(2^255 - 19).

Batch-first like every hot path in this repo: `x25519(k, u)` takes uint8
`[B, 32]` scalars and u-coordinates and runs `B` independent ladders as one
traced computation — the ladder is 255 identical, data-independent iterations
(`lax.fori_loop`), so the batch is pure width. The scalar enters through the
§5 clamp and the u-coordinate through the top-bit mask, both spelled here
exactly as the RFC spells them; the conditional swap is an XOR mask, not a
branch, though per `docs/reference/security.md` no constant-time claim follows
from that.

The functions every consumer needs are `x25519` and `public_key` (the ladder
at the basepoint 9). DHKEM(X25519, HKDF-SHA256) — the `Kem` seam wrapper RFC
9180 §4.1 defines over these — lives in `dhkem.py` once its HKDF dependency
lands in the pinned hash-frx.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.x25519 import field

KEY_SIZE = 32

# a24 = (486662 - 2) / 4 for curve25519, encoded as bytes so it enters the
# field through `from_bytes` like every other value — the limb layout stays
# field.py's own business, and a radix change there cannot strand a hand-laid
# constant here.
_A24_BYTES = np.frombuffer((121665).to_bytes(KEY_SIZE, "little"), dtype=np.uint8)


def _clamp(scalar: Array) -> Array:
    """RFC 7748 §5 `decodeScalar25519`: clear the low 3 bits, clear the top
    bit, set bit 254."""
    return fnp.concatenate(
        [
            scalar[..., :1] & np.uint8(248),
            scalar[..., 1:31],
            (scalar[..., 31:] & np.uint8(127)) | np.uint8(64),
        ],
        axis=-1,
    )


def _scalar_bits(scalar: Array) -> Array:
    """Clamped scalar bytes `[B, 32]` -> bits `[B, 255]`, bit `t` at index
    `t` — laid out ahead of the ladder so each iteration is one dynamic
    slice rather than byte arithmetic."""
    positions = np.arange(255)
    selected = scalar[..., positions // 8].astype(fnp.uint32)
    shifts = (positions % 8).astype(np.uint32)
    return (selected >> shifts) & np.uint32(1)


def _cswap(swap: Array, left: Array, right: Array) -> tuple[Array, Array]:
    """Swap the two field elements where `swap` (uint32 `[B, 1]`) is 1 —
    an XOR mask, the RFC's own formulation."""
    mask = fnp.uint32(0) - swap
    delta = (left ^ right) & mask
    return left ^ delta, right ^ delta


def _ladder_step(index: Array, carry: tuple[Array, ...]) -> tuple[Array, ...]:
    """One RFC 7748 §5 ladder iteration. The loop-invariants (`bits`, `x1`,
    `a24`) ride the carry — where they cost nothing — instead of being closed
    over, because frx keys the loop-body lowering cache on the body function's
    identity (the gotcha `aes/ghash._absorb` measured): a closure minted per
    `x25519` call would re-trace this ~10-multiply body every call."""
    x2, z2, x3, z3, swap, bits, x1, a24 = carry
    bit = frx.lax.dynamic_slice_in_dim(bits, 254 - index, 1, axis=-1)
    swap = swap ^ bit
    x2, x3 = _cswap(swap, x2, x3)
    z2, z3 = _cswap(swap, z2, z3)
    swap = bit

    a = field.add(x2, z2)
    aa = field.square(a)
    b = field.sub(x2, z2)
    bb = field.square(b)
    e = field.sub(aa, bb)
    c = field.add(x3, z3)
    d = field.sub(x3, z3)
    da = field.mul(d, a)
    cb = field.mul(c, b)
    x3 = field.square(field.add(da, cb))
    z3 = field.mul(x1, field.square(field.sub(da, cb)))
    x2 = field.mul(aa, bb)
    z2 = field.mul(e, field.add(aa, field.mul(a24, e)))
    return (x2, z2, x3, z3, swap, bits, x1, a24)


def x25519(scalar: ArrayLike, u: ArrayLike) -> Array:
    """RFC 7748 X25519: uint8 `[B, 32]` scalars x u-coordinates -> uint8
    `[B, 32]` outputs, little-endian, per batch entry."""
    scalar = fnp.asarray(scalar, dtype=fnp.uint8)
    u = fnp.asarray(u, dtype=fnp.uint8)
    batch = scalar.shape[:-1]

    bits = _scalar_bits(_clamp(scalar))
    # §5 decodeUCoordinate: mask the top bit; the value is used unreduced.
    x1 = field.from_bytes(
        fnp.concatenate([u[..., :31], u[..., 31:] & np.uint8(127)], axis=-1)
    )

    a24 = field.from_bytes(
        fnp.broadcast_to(fnp.asarray(_A24_BYTES), (*batch, KEY_SIZE))
    )
    x2, z2 = field.one(batch), field.zero(batch)
    x3, z3 = x1, field.one(batch)
    swap = fnp.zeros((*batch, 1), dtype=fnp.uint32)

    x2, z2, x3, z3, swap, _, _, _ = frx.lax.fori_loop(
        0, 255, _ladder_step, (x2, z2, x3, z3, swap, bits, x1, a24)
    )
    x2, _ = _cswap(swap, x2, x3)
    z2, _ = _cswap(swap, z2, z3)
    return field.to_bytes(field.mul(x2, field.invert(z2)))


def basepoint(batch: tuple[int, ...] = (1,)) -> Array:
    """The curve25519 base u-coordinate 9, encoded, broadcast to `[B, 32]`."""
    encoded = np.zeros(KEY_SIZE, dtype=np.uint8)
    encoded[0] = 9
    return fnp.broadcast_to(fnp.asarray(encoded), (*batch, KEY_SIZE))


def public_key(scalar: ArrayLike) -> Array:
    """`x25519(scalar, 9)` — the §6.1 public-key derivation, batch-first."""
    scalar = fnp.asarray(scalar, dtype=fnp.uint8)
    return x25519(scalar, basepoint(scalar.shape[:-1]))
