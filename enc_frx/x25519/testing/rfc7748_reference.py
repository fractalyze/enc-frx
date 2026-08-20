# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RFC 7748's X25519 in plain Python integers.

Section 5's ladder transcribed one line at a time over `int` — no arrays, no
limbs, no vectorization — the `fips203_reference.py` pattern: it exists to be
*obviously* the specification, so a disagreement with the traced implementation
is a bug in the traced implementation rather than a question about which
convention either one meant. The published §5.2/§6.1 vectors gate both; this
module is what extends that gate to arbitrary batch shapes and to the extreme
field inputs a vector set never approaches.
"""

from __future__ import annotations

P = 2**255 - 19
A24 = 121665


def decode_scalar(k: bytes) -> int:
    """RFC 7748 §5: clamp, then read little-endian."""
    buf = bytearray(k)
    buf[0] &= 248
    buf[31] &= 127
    buf[31] |= 64
    return int.from_bytes(buf, "little")


def decode_u(u: bytes) -> int:
    """RFC 7748 §5: mask the top bit, read little-endian, no reduction."""
    return int.from_bytes(u, "little") & ((1 << 255) - 1)


def encode_u(x: int) -> bytes:
    return (x % P).to_bytes(32, "little")


def x25519(k: bytes, u: bytes) -> bytes:
    """The §5 Montgomery ladder over integers, conditional-swap form."""
    scalar = decode_scalar(k)
    x1 = decode_u(u)
    x2, z2 = 1, 0
    x3, z3 = x1, 1
    swap = 0
    for t in range(254, -1, -1):
        kt = (scalar >> t) & 1
        swap ^= kt
        if swap:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = kt
        a = (x2 + z2) % P
        aa = (a * a) % P
        b = (x2 - z2) % P
        bb = (b * b) % P
        e = (aa - bb) % P
        c = (x3 + z3) % P
        d = (x3 - z3) % P
        da = (d * a) % P
        cb = (c * b) % P
        x3 = pow(da + cb, 2, P)
        z3 = (x1 * pow(da - cb, 2, P)) % P
        x2 = (aa * bb) % P
        z2 = (e * (aa + A24 * e)) % P
    if swap:
        x2, x3 = x3, x2
        z2, z3 = z3, z2
    return encode_u((x2 * pow(z2, P - 2, P)) % P)
