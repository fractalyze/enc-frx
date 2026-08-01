# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""XChaCha20-Poly1305, per draft-irtf-cfrg-xchacha.

RFC 8439's nonce is 96 bits, which makes random nonces unsafe past a birthday
bound a real deployment reaches — around 2^32 messages under one key. This
variant takes 192 bits, which is safe to pick at random, and that is its whole
reason to exist.

The construction is a key derivation and nothing else: HChaCha20 turns
`(key, nonce[:16])` into a subkey, and the remaining eight nonce bytes — prefixed
with four zero bytes, since RFC 8439 wants twelve — drive the unmodified AEAD.
Every property of ChaCha20-Poly1305 carries over, including how `open` reports
and masks failure, because the same implementation does the work.

A draft rather than an RFC. The construction has been stable for years and is in
wide production use (libsodium, age), and the draft carries test vectors; the
scheme page records that status.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aead import Aead
from enc_frx.chacha import chacha20
from enc_frx.chacha.chacha20_poly1305 import ChaCha20Poly1305

KEY_SIZE = 32
NONCE_SIZE = 24
TAG_SIZE = 16

_DERIVATION_NONCE = 16  # the half HChaCha20 consumes
_INNER_PREFIX = 4  # zero bytes, since RFC 8439's nonce is twelve


class XChaCha20Poly1305:
    """A 256-bit key, a 192-bit nonce, and a 128-bit tag."""

    def __init__(self) -> None:
        self.key_size = KEY_SIZE
        self.nonce_size = NONCE_SIZE
        self.tag_size = TAG_SIZE
        self._inner = ChaCha20Poly1305()

    def __eq__(self, other: object) -> bool:
        return NotImplemented if not isinstance(other, XChaCha20Poly1305) else True

    def __hash__(self) -> int:
        return hash(type(self))

    def seal(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        plaintext: ArrayLike,
    ) -> Array:
        subkey, inner_nonce = self._derive(key, nonce)
        return self._inner.seal(subkey, inner_nonce, associated_data, plaintext)

    def open(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        ciphertext: ArrayLike,
    ) -> tuple[Array, Array]:
        subkey, inner_nonce = self._derive(key, nonce)
        return self._inner.open(subkey, inner_nonce, associated_data, ciphertext)

    def _derive(self, key: ArrayLike, nonce: ArrayLike) -> tuple[Array, Array]:
        nonce = fnp.asarray(nonce, dtype=fnp.uint8)
        subkey = chacha20.hchacha20(key, nonce[..., :_DERIVATION_NONCE])
        zeros = fnp.zeros((*nonce.shape[:-1], _INNER_PREFIX), dtype=fnp.uint8)
        return subkey, fnp.concatenate([zeros, nonce[..., _DERIVATION_NONCE:]], axis=-1)


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Aead] = XChaCha20Poly1305
