# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The AES block cipher, per FIPS 197. Encryption and the key schedule.

The usual difficulty with AES in software is the S-box: a 256-entry table indexed
by a secret byte is a cache-timing channel, and avoiding it normally means
bitslicing — a rewrite of the whole cipher into transposed bit planes.

That is not the shape of the problem here. **`zk_dtypes.binary_field_gf8_aes` is
a registered dtype**: GF(2^8) with AES's own reduction polynomial, the field
SubBytes and MixColumns are defined over. So the S-box is written the way FIPS
197 defines it — inversion followed by an affine map — and inversion is `x^254`,
a fixed addition chain of eleven field multiplications. No table, no gather, no
secret-dependent index anywhere in the cipher.

That makes the specification-shaped implementation and the side-channel-conscious
one the same implementation, which is not true in C. It is still not a
constant-time claim; see `docs/reference/security.md`.

**Encryption only.** GCM never invokes the inverse cipher — it is CTR mode
underneath — so `InvSubBytes`, `InvMixColumns` and the equivalent inverse key
schedule would be code with no caller.

The state is `[..., 4, 4]` in FIPS 197's `(row, column)` order, which is the
transpose of how the bytes arrive: §3.4 fills the state down columns.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from frx.typing import ArrayLike

BLOCK_SIZE = 16
KEY_SIZES = (16, 24, 32)

GF8 = zk_dtypes.binary_field_gf8_aes

_ROWS = 4
_COLUMNS = 4
_WORD = 4

# FIPS 197 §5.2: the round constant is x^(i-1) in GF(2^8). Ten entries covers
# AES-128's ten uses; the wider key sizes need fewer.
_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)

# FIPS 197 §5.1.1: the affine map is b ^ rotl(b,1) ^ rotl(b,2) ^ rotl(b,3)
# ^ rotl(b,4) ^ 0x63, over the bits of the inverse.
_AFFINE_ROTATIONS = (1, 2, 3, 4)
_AFFINE_CONSTANT = 0x63


def _field(value: int) -> Array:
    return fnp.asarray(np.array(value, dtype=GF8))


def _square(value: Array) -> Array:
    return value * value


def _inverse(value: Array) -> Array:
    """`x^254`, which is `x^-1` in GF(2^8) and maps 0 to 0 as FIPS 197 requires.

    The addition chain is eleven multiplications: `x^254 = x^240 * x^14`, with
    `x^240 = (x^15)^16` and `x^15 = x^12 * x^3`. Chosen over a table because a
    table would be indexed by a secret byte.
    """
    square = _square(value)
    cube = square * value
    twelfth = _square(_square(cube))
    fourteenth = twelfth * square
    fifteenth = twelfth * cube
    two_forty = _square(_square(_square(_square(fifteenth))))
    return two_forty * fourteenth


def _rotl8(value: Array, bits: int) -> Array:
    return (value << np.uint8(bits)) | (value >> np.uint8(8 - bits))


def sub_bytes(state: Array) -> Array:
    """FIPS 197 §5.1.1: inversion in GF(2^8), then the affine map over GF(2)."""
    inverted = _inverse(state).astype(fnp.uint8)
    affine = inverted
    for bits in _AFFINE_ROTATIONS:
        affine = affine ^ _rotl8(inverted, bits)
    return (affine ^ np.uint8(_AFFINE_CONSTANT)).astype(GF8)


def _rotate(values: Array, by: int) -> Array:
    """Rotate the trailing axis left by a compile-time constant.

    Two static slices and a concatenate rather than `fnp.take`, which lowers to a
    gather. The gather's indices would be constant and so leak nothing — but a
    cipher whose HLO contains no gather at all is a claim a test can check in one
    line, and this one is free.
    """
    if by == 0:
        return values
    return fnp.concatenate([values[..., by:], values[..., :by]], axis=-1)


def shift_rows(state: Array) -> Array:
    """FIPS 197 §5.1.2: row `r` rotates left by `r`. A static permutation."""
    return fnp.stack(
        [_rotate(state[..., row, :], row) for row in range(_ROWS)], axis=-2
    )


def mix_columns(state: Array) -> Array:
    """FIPS 197 §5.1.3: multiply each column by the fixed `{02,03,01,01}` circulant."""
    two, three = _field(0x02), _field(0x03)
    rows = [
        two * state[..., row, :]
        + three * state[..., (row + 1) % _ROWS, :]
        + state[..., (row + 2) % _ROWS, :]
        + state[..., (row + 3) % _ROWS, :]
        for row in range(_ROWS)
    ]
    return fnp.stack(rows, axis=-2)


def _sub_word(word: Array) -> Array:
    return sub_bytes(word)


def _rot_word(word: Array) -> Array:
    return _rotate(word, 1)


def key_schedule(key: ArrayLike) -> list[Array]:
    """FIPS 197 §5.2: `uint8 [B, 16 | 24 | 32]` -> `Nr + 1` round keys `[B, 4, 4]`.

    The word count and round count follow from the key length, so both are
    trace-time constants and the expansion unrolls.
    """
    key = fnp.asarray(key, dtype=fnp.uint8)
    key_size = key.shape[-1]
    if key_size not in KEY_SIZES:
        raise ValueError(f"AES takes a 16, 24, or 32 byte key, got {key_size}")

    key_words = key_size // _WORD
    rounds = key_words + 6
    words = [
        key[..., index * _WORD : (index + 1) * _WORD].astype(GF8)
        for index in range(key_words)
    ]

    for index in range(key_words, _COLUMNS * (rounds + 1)):
        temp = words[index - 1]
        if index % key_words == 0:
            temp = _sub_word(_rot_word(temp))
            constant = fnp.asarray(
                np.array(
                    [_RCON[index // key_words - 1], 0, 0, 0],
                    dtype=GF8,
                )
            )
            temp = temp + constant
        elif key_words > 6 and index % key_words == _WORD:
            temp = _sub_word(temp)
        words.append(words[index - key_words] + temp)

    # A round key's four words are its four columns, and a word's four bytes are
    # the rows of that column — the same transpose the state undergoes.
    return [
        fnp.stack(words[_COLUMNS * round : _COLUMNS * (round + 1)], axis=-1)
        for round in range(rounds + 1)
    ]


def _to_state(block: Array) -> Array:
    """FIPS 197 §3.4: the input fills the state down columns, so bytes transpose."""
    return fnp.swapaxes(
        block.astype(GF8).reshape(*block.shape[:-1], _COLUMNS, _ROWS), -1, -2
    )


def _from_state(state: Array) -> Array:
    """The inverse of `_to_state`, back to the `uint8` a caller holds."""
    return (
        fnp.swapaxes(state, -1, -2)
        .reshape(*state.shape[:-2], BLOCK_SIZE)
        .astype(fnp.uint8)
    )


def encrypt_with_schedule(round_keys: list[Array], block: ArrayLike) -> Array:
    """FIPS 197 §5.1's Cipher over an already-expanded key.

    The entry point a mode wants. One key covers many blocks in CTR — and so in
    GCM — while expanding it costs about as much as encrypting one: measured at
    8192 blocks on CPU, the schedule is 10.7 ms of AES-128's 15.2 ms. Expanding
    once per key rather than once per block is therefore the difference between
    paying that once and paying it per block.
    """
    state = _to_state(fnp.asarray(block, dtype=fnp.uint8)) + round_keys[0]
    for round_key in round_keys[1:-1]:
        state = mix_columns(shift_rows(sub_bytes(state))) + round_key
    return _from_state(shift_rows(sub_bytes(state)) + round_keys[-1])


def encrypt_block(key: ArrayLike, block: ArrayLike) -> Array:
    """`uint8 [B, 16 | 24 | 32]`, `uint8 [B, 16]` -> `uint8 [B, 16]`.

    Expands the key and encrypts one block per batch entry. A caller encrypting
    many blocks under one key should expand once with `key_schedule` and call
    `encrypt_with_schedule`; see its docstring for what that saves.
    """
    return encrypt_with_schedule(key_schedule(key), block)


def encrypt_blocks(round_keys: list[Array], blocks: ArrayLike) -> Array:
    """`Nr + 1` round keys `[..., 4, 4]`, `uint8 [..., N, 16]` -> the same
    shape encrypted: one schedule shared across a stack of blocks.

    The share works by giving every round key a length-one block axis so it
    broadcasts against the stack — the axis sits at `-3` because the state is
    `[..., 4, 4]`. That placement is a fact about this module's state layout,
    which is why the helper lives here rather than being respelled at every
    mode's call site (CTR streams, GCM-SIV's key derivation, CCM's counters
    all encrypt such stacks).
    """
    shared = [key[..., None, :, :] for key in round_keys]
    return encrypt_with_schedule(shared, blocks)


def pad_to_blocks(data: ArrayLike) -> Array:
    """`uint8 [..., N]` -> `uint8 [..., ceil(N/16), 16]`, zero-padded.

    `N` is static, so the padding is a constant-shaped concatenation. Byte
    chunking, not field arithmetic — it lives with the block size it encodes,
    and GHASH, GCM-SIV and CCM all format their inputs through it.
    """
    data = fnp.asarray(data, dtype=fnp.uint8)
    length = data.shape[-1]
    blocks = -(-length // BLOCK_SIZE)
    padded = fnp.pad(
        data, [(0, 0)] * (data.ndim - 1) + [(0, blocks * BLOCK_SIZE - length)]
    )
    return padded.reshape(*data.shape[:-1], blocks, BLOCK_SIZE)
