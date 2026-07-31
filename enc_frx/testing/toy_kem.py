# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""A stand-in with the Kem seam's shape and none of its security. TEST ONLY.

Encapsulation masks the sender's randomness with the secret key and appends a
checksum of it; decapsulation unmasks, recomputes the ciphertext, and compares.
Anyone who wants to break it can. It is never re-exported from the package.

It exists so the seam and the known-answer harness can be exercised before any
real scheme lands, and it reproduces the one structural property that matters:
**the re-encryption check and implicit rejection**. A ciphertext that fails the
check yields a rejection secret derived from the decapsulation key's `z` half —
deterministic, and different from what any sender computed. That is the shape
FIPS 203's FO transform has, minus every reason it is secure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from enc_frx.kem import Kem

SECRET_SIZE = 8
CHECKSUM_SIZE = 4


class ToyKem:
    """Not a KEM. See the module docstring."""

    def __init__(self, domain: int) -> None:
        self._domain = domain
        self.seed_size = 2 * SECRET_SIZE
        self.randomness_size = SECRET_SIZE
        self.encapsulation_key_size = SECRET_SIZE
        self.decapsulation_key_size = 2 * SECRET_SIZE
        self.ciphertext_size = SECRET_SIZE + CHECKSUM_SIZE
        self.shared_secret_size = SECRET_SIZE

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToyKem):
            return NotImplemented
        return self._domain == other._domain

    def __hash__(self) -> int:
        return hash(self._domain)

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        # The decapsulation key carries the rejection seed `z` alongside the
        # secret, which is what makes rejection deterministic without being
        # predictable to a caller holding only the ciphertext.
        decapsulation_key = fnp.asarray(seed, dtype=fnp.uint8)[: 2 * SECRET_SIZE]
        return 255 - decapsulation_key[:SECRET_SIZE], decapsulation_key

    def encaps(
        self, encapsulation_key: ArrayLike, *, randomness: ArrayLike
    ) -> tuple[Array, Array]:
        secret = 255 - fnp.asarray(encapsulation_key, dtype=fnp.uint8)
        message = fnp.asarray(randomness, dtype=fnp.uint8)
        return self._encrypt(secret, message), self._derive(message)

    def decaps(self, decapsulation_key: ArrayLike, ciphertext: ArrayLike) -> Array:
        key = fnp.asarray(decapsulation_key, dtype=fnp.uint8)
        secret, rejection_seed = key[..., :SECRET_SIZE], key[..., SECRET_SIZE:]
        ciphertext = fnp.asarray(ciphertext, dtype=fnp.uint8)

        message = ciphertext[..., :SECRET_SIZE] - secret
        # Re-encrypt and compare the whole ciphertext: the FO transform's check,
        # and the only thing standing between this and a malleable scheme.
        accepted = fnp.all(self._encrypt(secret, message) == ciphertext, axis=-1)
        return fnp.where(
            accepted[..., None],
            self._derive(message),
            self._reject(rejection_seed, ciphertext),
        )

    def _encrypt(self, secret: Array, message: Array) -> Array:
        checksum = fnp.sum(
            message.reshape(*message.shape[:-1], CHECKSUM_SIZE, -1), axis=-1
        ).astype(fnp.uint8)
        return fnp.concatenate([message + secret, checksum], axis=-1)

    def _derive(self, message: Array) -> Array:
        return message * 3 + self._domain

    def _reject(self, rejection_seed: Array, ciphertext: Array) -> Array:
        # Deterministic in (z, ciphertext), so a rejected ciphertext decapsulates
        # to the same secret every time — a caller cannot tell it apart by
        # repeating the call.
        return (
            rejection_seed + fnp.sum(ciphertext, axis=-1).astype(fnp.uint8)[..., None]
        )


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Kem] = ToyKem
