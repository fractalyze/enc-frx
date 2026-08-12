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

from enc_frx.ml_kem.params import (
    POLY_BYTES,
    SEED_SIZE,
    N,
    Q,
    ciphertext_size,
    decapsulation_key_size,
    decryption_key_size,
    encapsulation_key_size,
)

# The widths FIPS 203 encodes at: 1 for the message, 4/5/10/11 for ciphertext
# compression, 12 for uncompressed coefficients. `Compress_d` is defined only
# for d < 12 — `compress(x, 12)` wraps rather than being the identity.
WIDTHS = (1, 4, 5, 10, 11, 12)


def checked_length(value: ArrayLike, size: int, name: str) -> Array:
    """Pin a byte string's length at trace time, mirroring `AesGcm._checked`.

    Public because `sampling.py` checks its seeds and its PRF output the same
    way, and already depends on this module for the bit order. FIPS 203 fixes
    every one of those lengths, so a copy per call site would be four chances to
    word one rule differently.

    It does not follow that every module should reach for it. `hashes.py` keeps
    its own three-line check rather than take a dependency on the wire formats to
    validate a seed, which is what `AesGcm._checked` does too.

    The type check of FIPS 203 §7.2/§7.3 is normative and sits at a different
    altitude from the modulus check below it: a length is static in a traced
    program, so it is an exception, while a coefficient's range is data and so
    is a value.

    Skipping it is not a missing niceness. A `dk` short by 40 bytes used to
    decode to a **zero-length** `z` — the implicit-rejection seed — so the
    rejection secret would have been derived from nothing, silently.
    """
    array = fnp.asarray(value).astype(np.uint8)
    if array.shape[-1] != size:
        raise ValueError(f"{name} is {size} bytes, got {array.shape[-1]}")
    return array


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
    # `>>` is exactly floor division by a power of two, including for negatives;
    # `//` on a signed lane emits a sign-correction chain XLA cannot drop.
    return num >> np.int32(d + 1)


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


def _decode_raw(b: ArrayLike, d: int) -> Array:
    """The d-bit values the bytes carry, before any reduction."""
    bits = bytes_to_bits(b)
    grouped = bits.reshape(*bits.shape[:-2], N, d)
    weights = np.int32(1) << np.arange(d, dtype=np.int32)
    return (grouped * weights).sum(axis=-1).astype(np.int32)


def byte_decode(b: ArrayLike, d: int) -> Array:
    """`[..., 32d]` bytes to `[..., 256]` coefficients, FIPS 203 Algorithm 6.

    Reduces mod `q` at `d = 12` and mod `2^d` below it, as the standard does.
    That asymmetry is what makes the modulus check work — see the module
    docstring.
    """
    _check_width(d)
    coeffs = _decode_raw(b, d)
    return coeffs % np.int32(Q) if d == 12 else coeffs


def _bits_to_bytes(bits: Array) -> Array:
    grouped = bits.reshape(*bits.shape[:-1], bits.shape[-1] // 8, 8)
    weights = (np.int32(1) << fnp.arange(8, dtype=np.int32)).astype(np.int32)
    return (grouped * weights).sum(axis=-1).astype(np.uint8)


def bytes_to_bits(b: ArrayLike) -> Array:
    """`[..., L]` bytes to `[..., L, 8]` bits, little-endian within a byte.

    Public because `sampling.py` reads the same bit stream: FIPS 203 fixes one
    bit order for the whole standard, so `SamplePolyCBD` and `ByteDecode` must
    not each have their own reading of it.
    """
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
    """Inverse of `encode_ek`, returning `(t̂, ρ)`. Length-checked, §7.2."""
    e = checked_length(ek, encapsulation_key_size(k), "an encapsulation key")
    split = POLY_BYTES * k
    return decode_vector(e[..., :split], 12, k), e[..., split:]


def encode_dk_pke(s_hat: ArrayLike) -> Array:
    """`ByteEncode_12(ŝ)` — K-PKE's decryption key, FIPS 203 §5.1.

    A name rather than a bare `encode_vector(·, 12)` at each end because the two
    ends are in different modules: key generation writes it and decryption reads
    it back, and the width is the only thing that makes them agree. Named, the
    pair is one definition; open-coded, it is two readings of the same line.
    """
    return encode_vector(s_hat, 12)


def decode_dk_pke(dk_pke: ArrayLike, k: int) -> Array:
    """Inverse of `encode_dk_pke`, returning `ŝ` as `[..., k, 256]`.

    Distinct from `decode_dk` below, which splits ML-KEM's `dk` into its four
    fields — the first of which is this.
    """
    return decode_vector(
        checked_length(dk_pke, decryption_key_size(k), "a K-PKE decryption key"), 12, k
    )


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
    ci = checked_length(c, ciphertext_size(k, du, dv), "a ciphertext")
    split = 32 * du * k
    return (
        decompress(decode_vector(ci[..., :split], du, k), du),
        decompress(byte_decode(ci[..., split:], dv), dv),
    )


def encode_dk(dk_pke: ArrayLike, ek: ArrayLike, h_ek: ArrayLike, z: ArrayLike) -> Array:
    """`dk_PKE ‖ ek ‖ H(ek) ‖ z`, FIPS 203 §7.3.

    `h_ek` is `H(ek)` — SHA3-256, 32 bytes, §4.1 — taken as an argument rather
    than computed, so this module keeps depending on nothing outside `frx` and
    `numpy`; the hash belongs to the sampling layer. Named explicitly because
    unchecked bytes of the right length could equally be `G`, and a positive
    round trip would not notice.
    """
    return fnp.concatenate(
        [fnp.asarray(x).astype(np.uint8) for x in (dk_pke, ek, h_ek, z)], axis=-1
    )


def decode_dk(dk: ArrayLike, k: int) -> tuple[Array, Array, Array, Array]:
    """Inverse of `encode_dk`, splitting at the four fixed offsets."""
    parts = checked_length(dk, decapsulation_key_size(k), "a decapsulation key")
    pke_end = decryption_key_size(k)
    ek_end = pke_end + encapsulation_key_size(k)
    hash_end = ek_end + SEED_SIZE
    return (
        parts[..., :pke_end],
        parts[..., pke_end:ek_end],
        parts[..., ek_end:hash_end],
        parts[..., hash_end:],
    )


def coefficients_are_reduced(b: ArrayLike) -> Array:
    """The modulus check of FIPS 203 §7.2, as a per-entry boolean.

    The standard states it as `ByteEncode_12(ByteDecode_12(x)) == x`, which holds
    exactly when every 12-bit value the bytes carry is already below `q`: the
    decode maps `v` to `v % q`, and that re-encodes to `v` only when `v < q`. So
    the predicate is the same one asked directly, without materializing a decode
    and an encode to compare. Normative, and a malformed key that skips it is a
    documented attack surface.

    Only meaningful at d = 12, so the width is not a parameter: below it
    `byte_decode` reduces mod `2^d` and re-encoding always reproduces the input,
    making the check a constant `True`.

    Takes one polynomial or a whole encoded vector — §7.2 asks the question of
    `ek`'s `t̂`, which is `k` of them — and reduces over both, so the result is
    one boolean per batch entry either way. How many there are is the byte
    length divided by `POLY_BYTES`, which is static at trace time.

    A value rather than an exception because the batch axis has no way to raise
    for entry 7 alone.
    """
    bi = fnp.asarray(b)
    if bi.shape[-1] % POLY_BYTES:
        raise ValueError(
            f"a ByteEncode_12 value is a multiple of {POLY_BYTES} bytes, "
            f"got {bi.shape[-1]}"
        )
    polynomials = bi.reshape(*bi.shape[:-1], bi.shape[-1] // POLY_BYTES, POLY_BYTES)
    reduced = _decode_raw(polynomials, 12) < np.int32(Q)
    return reduced.all(axis=-1).all(axis=-1)
