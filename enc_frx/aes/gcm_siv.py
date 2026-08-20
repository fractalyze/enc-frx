# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES-GCM-SIV, per RFC 8452. The nonce-misuse-resistant `Aead`.

What SIV buys over `AesGcm` is the failure mode: repeat a nonce under GCM and
the GHASH subkey leaks — every later message is forgeable; repeat one here and
an observer learns only whether two plaintexts were equal. The price is one
extra pass (the tag is a PRF over the *plaintext*, and the counter comes from
the tag, so nothing can stream) plus a per-message key derivation.

The construction, and where each part already lives:

- **§4 key derivation** — per (key, nonce): six (or four) AES blocks of
  `LE32(i) ‖ nonce` under the key-generating key, keeping the *first eight
  bytes* of each; two pairs form the message-authentication key, the rest the
  message-encryption key. Everything downstream uses only derived keys.
- **§5 tag** — `POLYVAL(auth_key, AAD ‖ pad ‖ P ‖ pad ‖ LE lengths)`, nonce
  XORed into the first twelve bytes, top bit of the last byte cleared, then
  one AES block under the encryption key (`polyval.py`, `block.py`).
- **§5 CTR** — the counter block is the tag with its top bit set, and the
  increment is over the *first* four bytes as a little-endian word — both
  ends opposite to GCM's, which is why the keystream lives here rather than
  in `ctr.py`: the two conventions sharing one module would put an endianness
  parameter on a function whose whole job is to not be configurable.
  Appendix C.3's counter-wrap vectors pin the uint32 wrap.

`open` decrypts, recomputes the tag over the recovered plaintext, and masks
per entry — the comparison is an arithmetic reduction, the mask is what keeps
an unread `ok` from costing plaintext, exactly the seam's contract. Nonce and
tag sizes are RFC constants (12 and 16), not constructor parameters: the RFC
defines no other, so offering knobs would manufacture a downgrade surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aead import Aead
from enc_frx.aes import block, polyval

BLOCK_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16
# RFC 8452 defines AEAD_AES_128_GCM_SIV and AEAD_AES_256_GCM_SIV; there is no
# 192-bit member to offer.
KEY_SIZES = (16, 32)


class AesGcmSiv:
    """RFC 8452's AEAD over AES, parameterized by the one size it leaves open."""

    nonce_size = NONCE_SIZE
    tag_size = TAG_SIZE

    def __init__(self, key_size: int = 16) -> None:
        if key_size not in KEY_SIZES:
            raise ValueError(f"RFC 8452 defines 16 and 32 byte keys, got {key_size}")
        self.key_size = key_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AesGcmSiv):
            return NotImplemented
        return self.key_size == other.key_size

    def __hash__(self) -> int:
        return hash((type(self), self.key_size))

    def __repr__(self) -> str:
        return f"AesGcmSiv(key_size={self.key_size})"

    def seal(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        plaintext: ArrayLike,
    ) -> Array:
        key = self._checked(key, self.key_size, "key")
        nonce = self._checked(nonce, NONCE_SIZE, "nonce")
        plaintext = fnp.asarray(plaintext, dtype=fnp.uint8)

        auth_key, schedule = self._derive_keys(key, nonce)
        tag = self._tag(auth_key, schedule, nonce, associated_data, plaintext)
        keystream = self._keystream(schedule, tag, plaintext.shape[-1])
        return fnp.concatenate([plaintext ^ keystream, tag], axis=-1)

    def open(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        ciphertext: ArrayLike,
    ) -> tuple[Array, Array]:
        key = self._checked(key, self.key_size, "key")
        nonce = self._checked(nonce, NONCE_SIZE, "nonce")
        ciphertext = fnp.asarray(ciphertext, dtype=fnp.uint8)
        if ciphertext.shape[-1] < TAG_SIZE:
            raise ValueError(
                f"a {TAG_SIZE}-byte tag does not fit in a "
                f"{ciphertext.shape[-1]}-byte ciphertext"
            )
        body = ciphertext[..., :-TAG_SIZE]
        tag = ciphertext[..., -TAG_SIZE:]

        auth_key, schedule = self._derive_keys(key, nonce)
        plaintext = body ^ self._keystream(schedule, tag, body.shape[-1])

        # The tag is recomputed over the *decrypted* plaintext — SIV verifies
        # what it recovered, not what it was handed. An arithmetic reduction
        # per entry, never an early exit and never a `lax.cond`.
        expected = self._tag(auth_key, schedule, nonce, associated_data, plaintext)
        accepted = fnp.all(expected == tag, axis=-1)
        return fnp.where(accepted[..., None], plaintext, 0), accepted

    def _checked(self, value: ArrayLike, size: int, name: str) -> Array:
        array = fnp.asarray(value, dtype=fnp.uint8)
        if array.shape[-1] != size:
            raise ValueError(
                f"this scheme's {name} is {size} bytes, got {array.shape[-1]}"
            )
        return array

    def _derive_keys(self, key: Array, nonce: Array) -> tuple[Array, list[Array]]:
        """RFC 8452 §4: uint8 `[B, K]`, `[B, 12]` -> the 16-byte
        message-authentication key and the expanded message-encryption
        schedule. Each derived half comes from the first eight bytes of one
        AES block over `LE32(i) ‖ nonce`."""
        count = 2 + self.key_size // 8
        counters = np.zeros((count, 4), dtype=np.uint8)
        counters[:, 0] = np.arange(count, dtype=np.uint8)
        blocks_in = fnp.concatenate(
            [
                fnp.broadcast_to(fnp.asarray(counters), (*key.shape[:-1], count, 4)),
                fnp.broadcast_to(
                    nonce[..., None, :], (*nonce.shape[:-1], count, NONCE_SIZE)
                ),
            ],
            axis=-1,
        )
        schedule = block.key_schedule(key)
        halves = block.encrypt_blocks(schedule, blocks_in)[..., :8]
        derived = halves.reshape(*key.shape[:-1], count * 8)
        auth_key = derived[..., :BLOCK_SIZE]
        enc_key = derived[..., BLOCK_SIZE : BLOCK_SIZE + self.key_size]
        return auth_key, block.key_schedule(enc_key)

    def _tag(
        self,
        auth_key: Array,
        schedule: list[Array],
        nonce: Array,
        associated_data: ArrayLike | None,
        plaintext: Array,
    ) -> Array:
        """§5's tag: POLYVAL over `AAD ‖ P` (each zero-padded to blocks) and
        the little-endian length block, nonce folded in, top bit cleared,
        one AES block under the encryption key."""
        if associated_data is None:
            associated_data = fnp.zeros((*plaintext.shape[:-1], 0), dtype=fnp.uint8)
        associated_data = fnp.asarray(associated_data, dtype=fnp.uint8)
        lengths = polyval.length_block(associated_data.shape[-1], plaintext.shape[-1])
        blocks_in = fnp.concatenate(
            [
                block.pad_to_blocks(associated_data),
                block.pad_to_blocks(plaintext),
                fnp.broadcast_to(lengths, (*plaintext.shape[:-1], 1, BLOCK_SIZE)),
            ],
            axis=-2,
        )
        digest = polyval.polyval(auth_key, blocks_in)
        mixed = fnp.concatenate(
            [
                digest[..., :NONCE_SIZE] ^ nonce,
                digest[..., NONCE_SIZE:-1],
                digest[..., -1:] & np.uint8(0x7F),
            ],
            axis=-1,
        )
        return block.encrypt_with_schedule(schedule, mixed)

    def _keystream(self, schedule: list[Array], tag: Array, length: int) -> Array:
        """§5's CTR: the counter block is the tag with its top bit set; the
        first four bytes advance as one little-endian uint32, wrapping (the
        Appendix C.3 rows exist to catch exactly that wrap)."""
        if length == 0:
            return fnp.zeros((*tag.shape[:-1], 0), dtype=fnp.uint8)
        blocks = -(-length // BLOCK_SIZE)
        initial = fnp.concatenate(
            [tag[..., :-1], tag[..., -1:] | np.uint8(0x80)], axis=-1
        )
        word = initial[..., :4].astype(fnp.uint32)
        value = (
            word[..., 0]
            | (word[..., 1] << np.uint32(8))
            | (word[..., 2] << np.uint32(16))
            | (word[..., 3] << np.uint32(24))
        )
        stepped = value[..., None] + fnp.arange(blocks, dtype=fnp.uint32)
        counter_bytes = fnp.stack(
            [
                ((stepped >> np.uint32(8 * index)) & np.uint32(0xFF)).astype(fnp.uint8)
                for index in range(4)
            ],
            axis=-1,
        )
        tails = fnp.broadcast_to(
            initial[..., None, 4:], (*initial.shape[:-1], blocks, BLOCK_SIZE - 4)
        )
        counter_blocks = fnp.concatenate([counter_bytes, tails], axis=-1)
        stream = block.encrypt_blocks(schedule, counter_blocks)
        return stream.reshape(*tag.shape[:-1], blocks * BLOCK_SIZE)[..., :length]


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Aead] = AesGcmSiv
