# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A stand-in with the Aead seam's shape and none of its security. TEST ONLY.

A keystream derived from the key, the nonce, and the byte position, plus a tag
that folds the key, the nonce, the associated data, and the ciphertext into four
bytes. Anyone who wants to forge it can. It is never re-exported from the
package.

It exists so the seam and the known-answer harness can be exercised before any
real scheme lands, and it reproduces the two structural properties that matter:
the tag comparison is an arithmetic reduction rather than an early exit, and
**a failing entry's plaintext comes back masked** rather than raw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aead import Aead

KEY_SIZE = 8
NONCE_SIZE = 4
TAG_SIZE = 4


class ToyAead:
    """Not an AEAD. See the module docstring."""

    def __init__(self, domain: int) -> None:
        self._domain = domain
        self.key_size = KEY_SIZE
        self.nonce_size = NONCE_SIZE
        self.tag_size = TAG_SIZE

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToyAead):
            return NotImplemented
        return self._domain == other._domain

    def __hash__(self) -> int:
        return hash(self._domain)

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

        ciphertext = plaintext ^ self._keystream(key, nonce, plaintext.shape[-1])
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

        # An arithmetic reduction over the whole tag — no early exit, and one
        # verdict per batch entry rather than one for the batch.
        accepted = fnp.all(self._tag(key, nonce, associated_data, body) == tag, axis=-1)
        plaintext = body ^ self._keystream(key, nonce, body.shape[-1])
        # Masked, not raw: a caller that forgets to read `accepted` gets zeros
        # instead of the unauthenticated decryption.
        return fnp.where(accepted[..., None], plaintext, 0), accepted

    def _keystream(self, key: Array, nonce: Array, length: int) -> Array:
        position = fnp.arange(length, dtype=fnp.uint8)
        repeats = -(-length // KEY_SIZE)
        stretched = fnp.tile(key, repeats)[..., :length]
        return stretched + position + self._nonce_fold(nonce)[..., None] + self._domain

    def _tag(
        self,
        key: Array,
        nonce: Array,
        associated_data: ArrayLike | None,
        body: Array,
    ) -> Array:
        acc = fnp.sum(body, axis=-1).astype(fnp.uint8) + self._nonce_fold(nonce)
        if associated_data is not None:
            acc = acc + fnp.sum(
                fnp.asarray(associated_data, dtype=fnp.uint8), axis=-1
            ).astype(fnp.uint8)
        weights = fnp.arange(1, TAG_SIZE + 1, dtype=fnp.uint8)
        return key[..., :TAG_SIZE] + acc[..., None] * weights

    def _nonce_fold(self, nonce: Array) -> Array:
        return fnp.sum(nonce, axis=-1).astype(fnp.uint8)


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Aead] = ToyAead
