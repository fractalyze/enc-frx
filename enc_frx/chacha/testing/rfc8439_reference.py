# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""RFC 8439 in plain Python integers — the oracle the traced code answers to.

Deliberately the least clever implementation available: exact arithmetic, one
block at a time, no limbs. It exists to be *obviously* the specification, so that
a disagreement with the traced implementation is a bug in the traced
implementation rather than a question.

That matters most for Poly1305, whose limb layout is squeezed against a 32-bit
lane (see `enc_frx/chacha/poly1305.py`) with a margin no published vector
approaches, and for the AEAD's MAC-input assembly, where a padding or
length-encoding mistake produces a wrong tag for every input at once.

TEST ONLY. Never re-exported from the package.
"""

from __future__ import annotations

_MASK32 = 0xFFFFFFFF
_P1305 = (1 << 130) - 5
_CLAMP = 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
_CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)


def _rotl(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (32 - bits))) & _MASK32


def _quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotl(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotl(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotl(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotl(state[b] ^ state[c], 7)


def chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """RFC 8439 §2.3: one 64-byte keystream block."""
    words = [int.from_bytes(key[i : i + 4], "little") for i in range(0, 32, 4)]
    nonce_words = [int.from_bytes(nonce[i : i + 4], "little") for i in range(0, 12, 4)]
    initial = [*_CONSTANTS, *words, counter & _MASK32, *nonce_words]

    state = list(initial)
    for _ in range(10):
        _quarter_round(state, 0, 4, 8, 12)
        _quarter_round(state, 1, 5, 9, 13)
        _quarter_round(state, 2, 6, 10, 14)
        _quarter_round(state, 3, 7, 11, 15)
        _quarter_round(state, 0, 5, 10, 15)
        _quarter_round(state, 1, 6, 11, 12)
        _quarter_round(state, 2, 7, 8, 13)
        _quarter_round(state, 3, 4, 9, 14)
    return b"".join(
        ((state[i] + initial[i]) & _MASK32).to_bytes(4, "little") for i in range(16)
    )


def chacha20_encrypt(key: bytes, counter: int, nonce: bytes, data: bytes) -> bytes:
    """RFC 8439 §2.4."""
    stream = b"".join(
        chacha20_block(key, counter + index, nonce)
        for index in range(-(-len(data) // 64) or 1)
    )
    return bytes(a ^ b for a, b in zip(data, stream, strict=False))


def poly1305_mac(key: bytes, message: bytes) -> bytes:
    """RFC 8439 §2.5, in exact integers."""
    r = int.from_bytes(key[:16], "little") & _CLAMP
    s = int.from_bytes(key[16:], "little")
    accumulator = 0
    for start in range(0, len(message), 16):
        block = message[start : start + 16]
        accumulator = (
            (accumulator + int.from_bytes(block + b"\x01", "little")) * r
        ) % _P1305
    return ((accumulator + s) % (1 << 128)).to_bytes(16, "little")


def _pad16(data: bytes) -> bytes:
    return b"\x00" * (-len(data) % 16)


def _mac_input(associated_data: bytes, ciphertext: bytes) -> bytes:
    """RFC 8439 §2.8's layout, spelled out."""
    return (
        associated_data
        + _pad16(associated_data)
        + ciphertext
        + _pad16(ciphertext)
        + len(associated_data).to_bytes(8, "little")
        + len(ciphertext).to_bytes(8, "little")
    )


def aead_seal(
    key: bytes, nonce: bytes, associated_data: bytes, plaintext: bytes
) -> bytes:
    """RFC 8439 §2.8: ciphertext with the tag appended."""
    one_time_key = chacha20_block(key, 0, nonce)[:32]
    ciphertext = chacha20_encrypt(key, 1, nonce, plaintext)
    return ciphertext + poly1305_mac(
        one_time_key, _mac_input(associated_data, ciphertext)
    )


def aead_open(
    key: bytes, nonce: bytes, associated_data: bytes, ciphertext: bytes
) -> tuple[bytes, bool]:
    """The inverse, returning the plaintext and whether the tag matched."""
    body, tag = ciphertext[:-16], ciphertext[-16:]
    one_time_key = chacha20_block(key, 0, nonce)[:32]
    expected = poly1305_mac(one_time_key, _mac_input(associated_data, body))
    return chacha20_encrypt(key, 1, nonce, body), expected == tag
