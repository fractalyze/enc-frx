# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ChaCha20, per RFC 8439 §2.1-2.4.

The cipher half of ChaCha20-Poly1305. Pure lane arithmetic — add, xor, rotate —
over a 4x4 state of `uint32`, with no table lookup and no data-dependent
indexing anywhere, which is why it vectorizes cleanly and why it has no
secret-dependent memory access to begin with.

**Blocks are the parallel axis; rounds are the sequential one.** Every block is
independent given its counter, so the keystream for `B` messages of `L` blocks is
one `[B, L]` computation with the 20 rounds applied across all of it. A `scan`
over blocks would serialize the only source of parallelism the cipher has. The
rounds cannot be parallelized and are not.

The state is carried as sixteen separate `[B, L]` arrays rather than one
`[B, L, 16]` array, so a quarter-round is Python-level index bookkeeping over
elementwise operations. The alternative — indexed writes into a packed state —
would put sixteen scatters in the trace per round for no gain.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

BLOCK_SIZE = 64
KEY_SIZE = 32
NONCE_SIZE = 12

_WORDS = 16
_ROUNDS = 10  # doubled: each iteration is a column round and a diagonal round

# "expand 32-byte k", RFC 8439 §2.3.
_CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)

# RFC 8439 §2.3.1: four column rounds, then four diagonal rounds.
_COLUMN_ROUNDS = ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15))
_DIAGONAL_ROUNDS = ((0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))


def _rotl(value: Array, bits: int) -> Array:
    return (value << np.uint32(bits)) | (value >> np.uint32(32 - bits))


def _quarter_round(state: list[Array], a: int, b: int, c: int, d: int) -> None:
    """RFC 8439 §2.1, in place on the sixteen-word state."""
    state[a] = state[a] + state[b]
    state[d] = _rotl(state[d] ^ state[a], 16)
    state[c] = state[c] + state[d]
    state[b] = _rotl(state[b] ^ state[c], 12)
    state[a] = state[a] + state[b]
    state[d] = _rotl(state[d] ^ state[a], 8)
    state[c] = state[c] + state[d]
    state[b] = _rotl(state[b] ^ state[c], 7)


def _le_words(data: Array, words: int) -> list[Array]:
    """Little-endian `uint8 [..., 4 * words]` -> `words` arrays of `uint32`."""
    grouped = data.reshape(*data.shape[:-1], words, 4).astype(fnp.uint32)
    return [
        grouped[..., index, 0]
        | (grouped[..., index, 1] << np.uint32(8))
        | (grouped[..., index, 2] << np.uint32(16))
        | (grouped[..., index, 3] << np.uint32(24))
        for index in range(words)
    ]


def _le_bytes(words: list[Array]) -> Array:
    """`uint32` words -> little-endian `uint8`, four bytes each."""
    lanes = [
        ((word >> np.uint32(8 * shift)) & np.uint32(0xFF)).astype(fnp.uint8)
        for word in words
        for shift in range(4)
    ]
    return fnp.stack(lanes, axis=-1)


def block_function(state: list[Array]) -> list[Array]:
    """The 20-round core, RFC 8439 §2.3: twenty rounds plus the feedforward add."""
    working = list(state)
    for _ in range(_ROUNDS):
        for indices in _COLUMN_ROUNDS:
            _quarter_round(working, *indices)
        for indices in _DIAGONAL_ROUNDS:
            _quarter_round(working, *indices)
    return [working[index] + state[index] for index in range(_WORDS)]


def initial_state(key: Array, nonce: Array, counters: Array) -> list[Array]:
    """The state before the rounds, broadcast over whatever axes `counters` has.

    `key` is `uint8 [..., 32]` and `nonce` `uint8 [..., 12]`; `counters` carries
    one extra trailing axis, which is what makes every block of a message an
    independent lane rather than a loop iteration.
    """
    key_words = _le_words(key, 8)
    nonce_words = _le_words(nonce, 3)
    shape = counters.shape
    constants = [fnp.full(shape, value, dtype=fnp.uint32) for value in _CONSTANTS]
    return (
        constants
        + [fnp.broadcast_to(word[..., None], shape) for word in key_words]
        + [counters]
        + [fnp.broadcast_to(word[..., None], shape) for word in nonce_words]
    )


def keystream(
    key: ArrayLike, nonce: ArrayLike, counter: ArrayLike, blocks: int
) -> Array:
    """`uint8 [B, 32]`, `uint8 [B, 12]`, `uint32 [B]` -> `uint8 [B, blocks * 64]`.

    Block `i` uses `counter + i`, and every block is computed in the same traced
    operation. RFC 8439 §2.3 fixes the counter at 32 bits, so it wraps rather
    than carrying into the nonce; a caller that would wrap has exceeded the
    256 GiB a single (key, nonce) pair may encrypt.
    """
    key = fnp.asarray(key, dtype=fnp.uint8)
    nonce = fnp.asarray(nonce, dtype=fnp.uint8)
    counters = fnp.asarray(counter, dtype=fnp.uint32)[..., None] + fnp.arange(
        blocks, dtype=fnp.uint32
    )
    words = block_function(initial_state(key, nonce, counters))
    serialized = _le_bytes(words)
    return serialized.reshape(*serialized.shape[:-2], blocks * BLOCK_SIZE)


def encrypt(
    key: ArrayLike, nonce: ArrayLike, counter: ArrayLike, data: ArrayLike
) -> Array:
    """`uint8 [B, N]` -> `uint8 [B, N]`, xored with the keystream (RFC 8439 §2.4).

    Encryption and decryption are the same operation. `N` is static, so the block
    count is a trace-time constant and the trailing keystream bytes are dropped
    rather than masked.
    """
    data = fnp.asarray(data, dtype=fnp.uint8)
    length = data.shape[-1]
    blocks = -(-length // BLOCK_SIZE)
    return data ^ keystream(key, nonce, counter, blocks)[..., :length]
