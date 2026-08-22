# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""X25519, per RFC 7748 §5 — the Montgomery ladder over GF(2^255 - 19).

Batch-first like every hot path in this repo: `x25519(k, u)` takes uint8
`[B, 32]` scalars and u-coordinates and runs `B` independent ladders as one
traced computation — the ladder is 255 identical, data-independent iterations
(`lax.fori_loop`), so the batch is pure width. The scalar enters through the
§5 clamp and the u-coordinate through the top-bit mask, both spelled here
exactly as the RFC spells them; the conditional swap is a `where` over a bit
that is a value rather than a branch, though per `docs/reference/security.md`
no constant-time claim follows from that.

**The field is `zk_dtypes.curve25519_bf`**, whose arithmetic lowers to
frxlib's native prime-field kernels, in the Montgomery storage variant
`curve25519_bf_mont` where the fast multiply lives. A field element is one
value carrying a trailing unit axis, not a limb vector, so the ladder is
spelled in operators.

The RFC's byte encoding *is* the canonical storage of `curve25519_bf`, so the
uint8 boundary is a free `view` on each side and the move into Montgomery
storage is an `astype` inside the trace. That boundary costs nothing
measurable: `testing/ladder_bench.py` prices it against host-entered
Montgomery material and the two track each other. Run that bench after a frx
pin bump — it is what would catch the convert falling out of the ladder's
`fori_loop` fusion, and nothing runs it automatically.

The functions every consumer needs are `x25519` and `public_key` (the ladder
at the basepoint 9). DHKEM(X25519, HKDF-SHA256) — the `Kem` seam wrapper RFC
9180 §4.1 defines over these — is `dhkem.py`.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from frx.typing import ArrayLike

KEY_SIZE = 32

# Canonical storage is the RFC's little-endian encoding, so bytes cross into
# WIRE as a view; WORK is where the multiply is fast, an `astype` away.
WIRE = np.dtype(zk_dtypes.curve25519_bf)
WORK = np.dtype(zk_dtypes.curve25519_bf_mont)

# a24 = (486662 - 2) / 4 for curve25519. A plain integer: the limb layout that
# once forced this through a 32-byte round-trip is gone.
_A24 = 121665


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
    slice rather than byte arithmetic.

    Bool rather than uint32: the only consumer is the ladder's `where`, so
    converting here does it once on `[B, 255]` instead of once per iteration
    on the slice.
    """
    positions = np.arange(255)
    selected = scalar[..., positions // 8].astype(fnp.uint32)
    shifts = (positions % 8).astype(np.uint32)
    return ((selected >> shifts) & np.uint32(1)).astype(bool)


def _cswap(swap: Array, left: Array, right: Array) -> tuple[Array, Array]:
    """Swap the two field elements where `swap` (bool `[..., 1]`) is set.

    `where`, not the RFC's XOR mask: a bitwise op on a field dtype is a type
    error, correctly — an element is a value, not a bit pattern.
    """
    return fnp.where(swap, right, left), fnp.where(swap, left, right)


def _constant(value: int, dtype: np.dtype, batch: tuple[int, ...]) -> Array:
    """A field constant broadcast over the batch. Field elements carry a
    trailing unit axis — the shape `[..., 32]` bytes view as.

    The source is rank 1, not rank 2: `keygen` is unbatched per the seam rule,
    so `batch` is `()` there and the target shape is `(1,)`, which a `(1, 1)`
    source cannot broadcast to.
    """
    return fnp.broadcast_to(fnp.asarray(np.array([value], dtype=dtype)), (*batch, 1))


def _ladder_step(index: Array, carry: tuple[Array, ...]) -> tuple[Array, ...]:
    """One RFC 7748 §5 ladder iteration. The loop-invariants (`bits`, `x1`,
    `a24`) ride the carry — where they cost nothing — instead of being closed
    over, because frx keys the loop-body lowering cache on the body function's
    identity (the gotcha `aes/ghash._absorb` measured): a closure minted per
    `x25519` call would re-trace this ~10-multiply body every call.
    """
    x2, z2, x3, z3, swap, bits, x1, a24 = carry
    bit = frx.lax.dynamic_slice_in_dim(bits, 254 - index, 1, axis=-1)
    swap = swap ^ bit
    x2, x3 = _cswap(swap, x2, x3)
    z2, z3 = _cswap(swap, z2, z3)
    swap = bit

    a = x2 + z2
    aa = a * a
    b = x2 - z2
    bb = b * b
    e = aa - bb
    c = x3 + z3
    d = x3 - z3
    da = d * a
    cb = c * b
    t = da + cb
    x3 = t * t
    s = da - cb
    z3 = x1 * (s * s)
    x2 = aa * bb
    z2 = e * (aa + a24 * e)
    return (x2, z2, x3, z3, swap, bits, x1, a24)


def _square_step(_: Array, acc: Array) -> Array:
    return acc * acc


def _invert(element: Array) -> Array:
    """`element^(p - 2)` by the standard 254-squaring addition chain for
    `2^255 - 21` — a fixed sequence, nothing data-dependent."""

    def pow2k(value: Array, squarings: int) -> Array:
        return frx.lax.fori_loop(0, squarings, _square_step, value)

    z2 = element * element
    z9 = pow2k(z2, 2) * element
    z11 = z9 * z2
    z_5_0 = (z11 * z11) * z9
    z_10_0 = pow2k(z_5_0, 5) * z_5_0
    z_20_0 = pow2k(z_10_0, 10) * z_10_0
    z_40_0 = pow2k(z_20_0, 20) * z_20_0
    z_50_0 = pow2k(z_40_0, 10) * z_10_0
    z_100_0 = pow2k(z_50_0, 50) * z_50_0
    z_200_0 = pow2k(z_100_0, 100) * z_100_0
    z_250_0 = pow2k(z_200_0, 50) * z_50_0
    return pow2k(z_250_0, 5) * z11


def _ladder(bits: Array, x1: Array) -> Array:
    """The RFC 7748 §5 ladder proper: the scalar's bits and the field-typed
    u-coordinate in, the recovered `x2/z2` field element out.

    Split out of `x25519` so a caller that already holds field material can
    drive the ladder without the byte boundary around it — which is what lets
    `testing/ladder_bench.py` measure that boundary rather than
    re-implementing the body to compare against. Everything else the ladder
    needs is a curve constant, taken from `x1`'s dtype and batch rather than
    passed, so there is one correct way to build them and it lives here.
    """
    batch = x1.shape[:-1]
    zero = _constant(0, x1.dtype, batch)
    one = _constant(1, x1.dtype, batch)
    a24 = _constant(_A24, x1.dtype, batch)

    x2, z2 = one, zero
    x3, z3 = x1, one
    swap = fnp.zeros((*batch, 1), dtype=bool)
    x2, z2, x3, z3, swap, _, _, _ = frx.lax.fori_loop(
        0, 255, _ladder_step, (x2, z2, x3, z3, swap, bits, x1, a24)
    )
    x2, _ = _cswap(swap, x2, x3)
    z2, _ = _cswap(swap, z2, z3)
    return x2 * _invert(z2)


def x25519(scalar: ArrayLike, u: ArrayLike) -> Array:
    """RFC 7748 X25519: uint8 `[B, 32]` scalars x u-coordinates -> uint8
    `[B, 32]` outputs, little-endian, per batch entry.

    A bare `[32]` is the empty batch and works the same — `keygen` is
    unbatched per the seam rule, so `dhkem` arrives that way.
    """
    scalar = fnp.asarray(scalar, dtype=fnp.uint8)
    u = fnp.asarray(u, dtype=fnp.uint8)

    bits = _scalar_bits(_clamp(scalar))
    # §5 decodeUCoordinate: mask the top bit; the value is used unreduced.
    x1 = fnp.concatenate([u[..., :31], u[..., 31:] & np.uint8(127)], axis=-1)
    x1 = x1.view(WIRE).astype(WORK)

    return _ladder(bits, x1).astype(WIRE).view(fnp.uint8)


def basepoint(batch: tuple[int, ...] = (1,)) -> Array:
    """The curve25519 base u-coordinate 9, encoded, broadcast to `[B, 32]`."""
    encoded = np.zeros(KEY_SIZE, dtype=np.uint8)
    encoded[0] = 9
    return fnp.broadcast_to(fnp.asarray(encoded), (*batch, KEY_SIZE))


def public_key(scalar: ArrayLike) -> Array:
    """`x25519(scalar, 9)` — the §6.1 public-key derivation, batch-first."""
    scalar = fnp.asarray(scalar, dtype=fnp.uint8)
    return x25519(scalar, basepoint(scalar.shape[:-1]))
