# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ChaCha20-Poly1305, per RFC 8439 §2.6-2.8. The first `Aead` implementation.

The construction is thin: a one-time Poly1305 key taken from ChaCha20's block 0,
the message encrypted from block 1, and a tag over the associated data, the
ciphertext, and their lengths. Everything interesting is in the two seam rules it
is the first scheme to have to honor.

**The tag comparison is an arithmetic reduction.** `fnp.all` over the sixteen tag
bytes, per batch entry — never an early exit, never a `lax.cond`. A `cond` on
secret-derived data would be a design bug rather than a slow path
(`docs/reference/security.md`), and in a batch it forces both sides anyway.

**A failing entry's plaintext comes back zeroed.** The decryption exists — it is
computed before the verdict is known, because a traced batch computes both — so
the mask is what keeps it from reaching a caller that did not read `ok`.

The MAC input is assembled at trace time: the associated data and the ciphertext
each padded up to a multiple of sixteen, then their lengths as little-endian
64-bit counts. Both lengths are static, so the padding is a constant-shaped
concatenation rather than anything data-dependent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aead import Aead
from enc_frx.chacha import chacha20, poly1305

KEY_SIZE = 32
NONCE_SIZE = 12
TAG_SIZE = 16

# RFC 8439 §2.8: the one-time key comes from block 0 and the message from 1.
_ONE_TIME_KEY_COUNTER = 0
_MESSAGE_COUNTER = 1

_MAC_ALIGNMENT = 16
_LENGTH_FIELD = 8


def _padded_length(length: int) -> int:
    return -(-length // _MAC_ALIGNMENT) * _MAC_ALIGNMENT


def _length_bytes(length: int, shape: tuple[int, ...]) -> Array:
    """A little-endian 64-bit count, broadcast over the batch."""
    encoded = np.frombuffer(length.to_bytes(_LENGTH_FIELD, "little"), dtype=np.uint8)
    return fnp.broadcast_to(fnp.asarray(encoded), (*shape, _LENGTH_FIELD))


def _zero_pad(data: Array, width: int) -> Array:
    return fnp.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, width - data.shape[-1])])


class ChaCha20Poly1305:
    """RFC 8439's AEAD: a 256-bit key, a 96-bit nonce, and a 128-bit tag."""

    def __init__(self) -> None:
        self.key_size = KEY_SIZE
        self.nonce_size = NONCE_SIZE
        self.tag_size = TAG_SIZE

    def __eq__(self, other: object) -> bool:
        # The scheme has no parameters, so every instance is the same value. The
        # seam still requires this: identity equality would re-trace an enclosing
        # jit zone for every freshly built instance.
        return NotImplemented if not isinstance(other, ChaCha20Poly1305) else True

    def __hash__(self) -> int:
        return hash(type(self))

    def seal(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        plaintext: ArrayLike,
    ) -> Array:
        key = fnp.asarray(key, dtype=fnp.uint8)
        nonce = fnp.asarray(nonce, dtype=fnp.uint8)
        plaintext = fnp.asarray(plaintext, dtype=fnp.uint8)

        ciphertext = chacha20.encrypt(
            key, nonce, self._counter(key, _MESSAGE_COUNTER), plaintext
        )
        tag = self._tag(key, nonce, associated_data, ciphertext)
        return fnp.concatenate([ciphertext, tag], axis=-1)

    def open(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        ciphertext: ArrayLike,
    ) -> tuple[Array, Array]:
        key = fnp.asarray(key, dtype=fnp.uint8)
        nonce = fnp.asarray(nonce, dtype=fnp.uint8)
        ciphertext = fnp.asarray(ciphertext, dtype=fnp.uint8)
        body, tag = ciphertext[..., :-TAG_SIZE], ciphertext[..., -TAG_SIZE:]

        accepted = fnp.all(self._tag(key, nonce, associated_data, body) == tag, axis=-1)
        plaintext = chacha20.encrypt(
            key, nonce, self._counter(key, _MESSAGE_COUNTER), body
        )
        return fnp.where(accepted[..., None], plaintext, 0), accepted

    def _counter(self, key: Array, value: int) -> Array:
        return fnp.full(key.shape[:-1], value, dtype=fnp.uint32)

    def _one_time_key(self, key: Array, nonce: Array) -> Array:
        """RFC 8439 §2.6: the first 32 bytes of ChaCha20's block 0."""
        block = chacha20.keystream(
            key, nonce, self._counter(key, _ONE_TIME_KEY_COUNTER), 1
        )
        return block[..., : poly1305.KEY_SIZE]

    def _tag(
        self,
        key: Array,
        nonce: Array,
        associated_data: ArrayLike | None,
        ciphertext: Array,
    ) -> Array:
        return poly1305.mac(
            self._one_time_key(key, nonce),
            self._mac_input(associated_data, ciphertext),
        )

    def _mac_input(self, associated_data: ArrayLike | None, ciphertext: Array) -> Array:
        """RFC 8439 §2.8: aad ‖ pad16 ‖ ciphertext ‖ pad16 ‖ len(aad) ‖ len(ct)."""
        batch = ciphertext.shape[:-1]
        parts = []
        aad_length = 0
        if associated_data is not None:
            aad = fnp.asarray(associated_data, dtype=fnp.uint8)
            aad_length = aad.shape[-1]
            parts.append(_zero_pad(aad, _padded_length(aad_length)))
        parts.append(_zero_pad(ciphertext, _padded_length(ciphertext.shape[-1])))
        parts.append(_length_bytes(aad_length, batch))
        parts.append(_length_bytes(ciphertext.shape[-1], batch))
        return fnp.concatenate(parts, axis=-1)


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Aead] = ChaCha20Poly1305
