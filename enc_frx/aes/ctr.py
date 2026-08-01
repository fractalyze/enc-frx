# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES in counter mode, per NIST SP 800-38A §6.5 and SP 800-38D §6.5.

Counter blocks are independent given the counter, so the keystream for `B`
messages of `L` blocks is one `[B, L]` computation — the same shape rule as
ChaCha20, and a `scan` over blocks would serialize the only parallelism the mode
has.

**The key is expanded once, not per block.** Measured on the block cipher, the
schedule is roughly 71% of a single-block encryption's cost, and CTR covers many
blocks with one key — so every entry point here takes an already-expanded
schedule and `key_schedule` is called once by the caller.

`inc32` increments only the rightmost 32 bits, wrapping there rather than
carrying into the nonce, which is what bounds a single (key, IV) pair.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aes import block as aes_block
from enc_frx.aes import ghash

BLOCK_SIZE = 16
_COUNTER_BYTES = 4


def _inc32(counter: Array, blocks: int) -> Array:
    """`uint8 [B, 16]` -> `uint8 [B, blocks, 16]`, the counter advanced per block.

    SP 800-38D §6.2: only the trailing 32 bits move, and they wrap. The addition
    is done on a big-endian `uint32` assembled from the last four bytes, so the
    wrap is the type's rather than a masked subtraction.
    """
    head = counter[..., :-_COUNTER_BYTES]
    tail = counter[..., -_COUNTER_BYTES:].astype(fnp.uint32)
    value = (
        (tail[..., 0] << np.uint32(24))
        | (tail[..., 1] << np.uint32(16))
        | (tail[..., 2] << np.uint32(8))
        | tail[..., 3]
    )
    stepped = value[..., None] + fnp.arange(blocks, dtype=fnp.uint32)
    bytes_out = fnp.stack(
        [
            ((stepped >> np.uint32(24 - 8 * index)) & np.uint32(0xFF)).astype(fnp.uint8)
            for index in range(_COUNTER_BYTES)
        ],
        axis=-1,
    )
    heads = fnp.broadcast_to(
        head[..., None, :], (*head.shape[:-1], blocks, head.shape[-1])
    )
    return fnp.concatenate([heads, bytes_out], axis=-1)


def keystream(round_keys: list[Array], counter: ArrayLike, blocks: int) -> Array:
    """`Nr + 1` round keys `[B, 4, 4]`, `uint8 [B, 16]` -> `uint8 [B, blocks * 16]`.

    Every block is encrypted in the same traced operation; the round keys carry a
    length-one block axis so one schedule broadcasts across all of them.
    """
    counter = fnp.asarray(counter, dtype=fnp.uint8)
    counter_blocks = _inc32(counter, blocks)
    shared = [key[..., None, :, :] for key in round_keys]
    encrypted = aes_block.encrypt_with_schedule(shared, counter_blocks)
    return encrypted.reshape(*counter.shape[:-1], blocks * BLOCK_SIZE)


def encrypt(round_keys: list[Array], counter: ArrayLike, data: ArrayLike) -> Array:
    """`uint8 [B, N]` -> `uint8 [B, N]`, xored with the keystream.

    Encryption and decryption are the same operation. `N` is static, so the block
    count is a trace-time constant and the trailing keystream bytes are dropped
    rather than masked.
    """
    data = fnp.asarray(data, dtype=fnp.uint8)
    length = data.shape[-1]
    blocks = -(-length // BLOCK_SIZE) or 1
    return data ^ keystream(round_keys, counter, blocks)[..., :length]


def initial_counter(subkey: ArrayLike, iv: ArrayLike) -> Array:
    """SP 800-38D §7.1's `J_0`: `uint8 [B, 16]`, `uint8 [B, N]` -> `uint8 [B, 16]`.

    A 96-bit IV is used directly with a counter of one. Any other length is
    hashed — `GHASH(H, IV ‖ pad ‖ 0^64 ‖ len(IV))` — which is why this lives
    beside CTR rather than inside GCM: it needs the hash subkey, and it is the
    only place the counter's construction depends on more than the IV.

    96 bits is the only length a caller should choose. The others exist because
    the standard defines them and the published vectors exercise them.
    """
    iv = fnp.asarray(iv, dtype=fnp.uint8)
    if iv.shape[-1] == 12:
        one = fnp.zeros((*iv.shape[:-1], _COUNTER_BYTES), dtype=fnp.uint8)
        return fnp.concatenate([iv, one.at[..., -1].set(1)], axis=-1)

    padded = ghash.pad_to_blocks(iv)
    trailing = fnp.broadcast_to(
        ghash.length_block(0, iv.shape[-1])[None, :],
        (*iv.shape[:-1], 1, BLOCK_SIZE),
    )
    return ghash.ghash(subkey, fnp.concatenate([padded, trailing], axis=-2))
