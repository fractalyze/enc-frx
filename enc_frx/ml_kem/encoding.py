# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM's byte encoding, compression, and wire formats, per FIPS 203 §4.2.

Everything here is batch-first over leading axes and traced, so a batch of `B`
keys is one computation rather than a Python loop over `B`.

**Compression is the scheme's only lossy step, and the only one that fails
quietly.** `Compress_d` throws bits away on purpose — it is why ML-KEM has a
decryption failure probability at all — so a rounding mistake does not produce a
wrong answer that a round trip catches. It shifts that probability, and every
round trip still passes. The standard defines the rounding `⌈·⌋` as round-half-
**up** over integers; the float form `round(x * 2**d / q)` is wrong at the ties
and wrong again from double rounding. Both directions here are exact integer
arithmetic, and the tests enumerate all 3329 inputs per `d` rather than sampling.

`ByteDecode_12` reduces mod `q` where the smaller widths reduce mod `2^d`, which
is not an inconsistency in the standard but the mechanism the modulus check
relies on: re-encoding a decoded key only reproduces the input when every
coefficient was already below `q`, so `ByteEncode_12(ByteDecode_12(x)) == x` is
exactly the normative check of FIPS 203 §7.2.

Validation results are **values, not exceptions**. A batch cannot raise on entry
7, and `decaps` must not branch on validity anyway — implicit rejection requires
a malformed ciphertext to yield a wrong-looking shared secret rather than an
error (`enc_frx/kem.py`).
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

Q = 3329
N = 256

# The widths FIPS 203 encodes at: 1 for the message, 4/5/10/11 for ciphertext
# compression, 12 for uncompressed coefficients.
WIDTHS = (1, 4, 5, 10, 11, 12)

SEED_SIZE = 32


def _as_int(x: ArrayLike) -> Array:
    """int32 throughout: the widest intermediate is `x * 2^(d+1)` at d = 12,
    about 2.7e7, so nothing here approaches the lane."""
    return fnp.asarray(x).astype(np.int32)


def compress(x: ArrayLike, d: int) -> Array:
    """`⌈(2^d / q) · x⌋ mod 2^d`, FIPS 203 §4.2.1.

    Round half up, done exactly: `⌊y + 1/2⌋` with `y = x·2^d/q` is
    `(2·x·2^d + q) // 2q`, which avoids representing the half at all.
    """
    _check_width(d)
    xi = _as_int(x)
    num = xi * np.int32(1 << (d + 1)) + np.int32(Q)
    return (num // np.int32(2 * Q)) & np.int32((1 << d) - 1)


def decompress(y: ArrayLike, d: int) -> Array:
    """`⌈(q / 2^d) · y⌋`, FIPS 203 §4.2.1. Same rounding, same reasoning."""
    _check_width(d)
    yi = _as_int(y)
    num = yi * np.int32(2 * Q) + np.int32(1 << d)
    return num // np.int32(1 << (d + 1))


def byte_encode(f: ArrayLike, d: int) -> Array:
    """`[..., 256]` coefficients to `[..., 32d]` bytes, FIPS 203 Algorithm 5.

    A shift-and-mask over a reshaped view rather than a bit-at-a-time loop: the
    bit stream is little-endian within each coefficient and each byte, so the
    whole thing is one reshape once the bits exist.
    """
    _check_width(d)
    fi = _as_int(f)
    bits = (fi[..., None] >> fnp.arange(d, dtype=np.int32)) & np.int32(1)
    return _bits_to_bytes(bits.reshape(*fi.shape[:-1], N * d))


def byte_decode(b: ArrayLike, d: int) -> Array:
    """`[..., 32d]` bytes to `[..., 256]` coefficients, FIPS 203 Algorithm 6.

    Reduces mod `q` at `d = 12` and mod `2^d` below it, as the standard does.
    That asymmetry is what makes the modulus check work — see the module
    docstring.
    """
    _check_width(d)
    bits = _bytes_to_bits(b).reshape(*fnp.asarray(b).shape[:-1], N, d)
    weights = (np.int32(1) << fnp.arange(d, dtype=np.int32)).astype(np.int32)
    coeffs = (bits * weights).sum(axis=-1).astype(np.int32)
    return coeffs % np.int32(Q) if d == 12 else coeffs


def _bits_to_bytes(bits: Array) -> Array:
    grouped = bits.reshape(*bits.shape[:-1], bits.shape[-1] // 8, 8)
    weights = (np.int32(1) << fnp.arange(8, dtype=np.int32)).astype(np.int32)
    return (grouped * weights).sum(axis=-1).astype(np.uint8)


def _bytes_to_bits(b: ArrayLike) -> Array:
    bi = _as_int(b)
    return (bi[..., None] >> fnp.arange(8, dtype=np.int32)) & np.int32(1)


def _check_width(d: int) -> None:
    if d not in WIDTHS:
        raise ValueError(f"FIPS 203 encodes at d in {WIDTHS}, got {d}")


def encode_vector(f: ArrayLike, d: int) -> Array:
    """`[..., k, 256]` to `[..., 32dk]` — the k polynomials, concatenated."""
    encoded = byte_encode(f, d)
    return encoded.reshape(*encoded.shape[:-2], encoded.shape[-2] * encoded.shape[-1])


def decode_vector(b: ArrayLike, d: int, k: int) -> Array:
    """`[..., 32dk]` back to `[..., k, 256]`."""
    bi = fnp.asarray(b)
    return byte_decode(bi.reshape(*bi.shape[:-1], k, 32 * d), d)


def encode_ek(t_hat: ArrayLike, rho: ArrayLike) -> Array:
    """`ByteEncode_12(t̂) ‖ ρ` — the encapsulation key, FIPS 203 §7.1."""
    return fnp.concatenate(
        [encode_vector(t_hat, 12), fnp.asarray(rho).astype(np.uint8)], axis=-1
    )


def decode_ek(ek: ArrayLike, k: int) -> tuple[Array, Array]:
    """Inverse of `encode_ek`, returning `(t̂, ρ)`."""
    e = fnp.asarray(ek)
    split = 384 * k
    return decode_vector(e[..., :split], 12, k), e[..., split:].astype(np.uint8)


def encode_ciphertext(u: ArrayLike, v: ArrayLike, du: int, dv: int) -> Array:
    """`ByteEncode_du(Compress_du(u)) ‖ ByteEncode_dv(Compress_dv(v))`."""
    return fnp.concatenate(
        [
            encode_vector(compress(u, du), du),
            byte_encode(compress(v, dv), dv),
        ],
        axis=-1,
    )


def decode_ciphertext(c: ArrayLike, k: int, du: int, dv: int) -> tuple[Array, Array]:
    """Inverse of `encode_ciphertext`, decompressed back to `Z_q`."""
    ci = fnp.asarray(c)
    split = 32 * du * k
    return (
        decompress(decode_vector(ci[..., :split], du, k), du),
        decompress(byte_decode(ci[..., split:], dv), dv),
    )


def encode_dk(dk_pke: ArrayLike, ek: ArrayLike, h_ek: ArrayLike, z: ArrayLike) -> Array:
    """`dk_PKE ‖ ek ‖ H(ek) ‖ z`, FIPS 203 §7.3.

    Takes `H(ek)` rather than computing it: the hash is the sampling layer's
    (#19), and this module depends on nothing outside the repo.
    """
    return fnp.concatenate(
        [fnp.asarray(x).astype(np.uint8) for x in (dk_pke, ek, h_ek, z)], axis=-1
    )


def decode_dk(dk: ArrayLike, k: int) -> tuple[Array, Array, Array, Array]:
    """Inverse of `encode_dk`, splitting at the four fixed offsets."""
    d = fnp.asarray(dk)
    a = 384 * k
    b = a + (384 * k + SEED_SIZE)
    c = b + SEED_SIZE
    return d[..., :a], d[..., a:b], d[..., b:c], d[..., c:]


def coefficients_are_reduced(b: ArrayLike, d: int = 12) -> Array:
    """The modulus check of FIPS 203 §7.2, as a per-entry boolean.

    `ByteEncode_12(ByteDecode_12(x)) == x` holds exactly when every coefficient
    the bytes carry was already below `q` — the decode's mod-`q` is what makes a
    12-bit value of 3329 or more fail to round-trip. Normative, and a malformed
    key that skips it is a documented attack surface.

    A value rather than an exception because the batch axis has no way to raise
    for entry 7 alone.
    """
    bi = fnp.asarray(b).astype(np.uint8)
    coeffs = byte_decode(bi, d)
    return (byte_encode(coeffs, d) == bi).all(axis=-1)
