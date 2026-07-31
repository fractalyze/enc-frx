# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The harness drives either seam, and refuses what it cannot gate.

Two things are under test and the second is the point. The drivers must run a
conforming scheme end to end — and they must *fail* a scheme that breaks one of
the rules the seams exist to enforce. A harness that only ever passes is
indistinguishable from no harness, so the broken variants below are the real
subject: each one changes exactly what its name says and must be caught.
"""

from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from frx.typing import ArrayLike

from enc_frx.testing.kat import (
    AeadVector,
    KatError,
    KemVector,
    check_aead,
    check_kem,
    to_bytes,
)
from enc_frx.testing.toy_aead import ToyAead
from enc_frx.testing.toy_kem import ToyKem

_PARAMS = "toy"
_CASES = 3
_MESSAGE_LEN = 16
_AAD_LEN = 5


def _kem_vectors(scheme: ToyKem) -> list[KemVector]:
    """A published-looking set: keygen, encapsulation and decapsulation cases."""
    rng = np.random.default_rng(0)
    vectors: list[KemVector] = []
    for case in range(_CASES):
        seed = bytes(rng.integers(0, 256, scheme.seed_size, dtype=np.uint8))
        randomness = bytes(rng.integers(0, 256, scheme.randomness_size, dtype=np.uint8))
        encaps_key, decaps_key = scheme.keygen(np.frombuffer(seed, dtype=np.uint8))
        ciphertext, secret = scheme.encaps(
            np.frombuffer(to_bytes(encaps_key), dtype=np.uint8)[None],
            randomness=np.frombuffer(randomness, dtype=np.uint8)[None],
        )
        vectors += [
            KemVector(
                case_id=f"keygen/{case}",
                function="keygen",
                seed=seed,
                encapsulation_key=to_bytes(encaps_key),
                decapsulation_key=to_bytes(decaps_key),
                parameter_set=_PARAMS,
            ),
            KemVector(
                case_id=f"encaps/{case}",
                function="encapsulation",
                encapsulation_key=to_bytes(encaps_key),
                randomness=randomness,
                ciphertext=to_bytes(ciphertext),
                shared_secret=to_bytes(secret),
                parameter_set=_PARAMS,
            ),
            KemVector(
                case_id=f"decaps/{case}",
                function="decapsulation",
                decapsulation_key=to_bytes(decaps_key),
                ciphertext=to_bytes(ciphertext),
                shared_secret=to_bytes(secret),
                parameter_set=_PARAMS,
            ),
        ]
    return vectors


def _aead_vectors(scheme: ToyAead) -> list[AeadVector]:
    rng = np.random.default_rng(1)

    def draw(size: int) -> bytes:
        return bytes(rng.integers(1, 256, size, dtype=np.uint8))

    vectors: list[AeadVector] = []
    for case in range(_CASES):
        key, nonce = draw(scheme.key_size), draw(scheme.nonce_size)
        aad, plaintext = draw(_AAD_LEN), draw(_MESSAGE_LEN)
        ciphertext = scheme.seal(
            np.frombuffer(key, dtype=np.uint8)[None],
            np.frombuffer(nonce, dtype=np.uint8)[None],
            np.frombuffer(aad, dtype=np.uint8)[None],
            np.frombuffer(plaintext, dtype=np.uint8)[None],
        )
        vectors.append(
            AeadVector(
                case_id=f"seal/{case}",
                parameter_set=_PARAMS,
                key=key,
                nonce=nonce,
                associated_data=aad,
                plaintext=plaintext,
                ciphertext=to_bytes(ciphertext),
            )
        )
    return vectors


class LeakyAead(ToyAead):
    """Returns the unauthenticated decryption instead of masking it."""

    def open(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        ciphertext: ArrayLike,
    ) -> tuple[Array, Array]:
        _, ok = super().open(key, nonce, associated_data, ciphertext)
        body = fnp.asarray(ciphertext, dtype=fnp.uint8)[..., : -self.tag_size]
        raw = body ^ self._keystream(
            fnp.asarray(key, dtype=fnp.uint8),
            fnp.asarray(nonce, dtype=fnp.uint8),
            body.shape[-1],
        )
        return raw, ok


class BatchWideAead(ToyAead):
    """Decides one verdict for the whole batch instead of one per entry."""

    def open(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        ciphertext: ArrayLike,
    ) -> tuple[Array, Array]:
        plaintext, ok = super().open(key, nonce, associated_data, ciphertext)
        collapsed = fnp.full(ok.shape, fnp.all(ok))
        return fnp.where(collapsed[..., None], plaintext, 0), collapsed


class NonRejectingKem(ToyKem):
    """Skips the re-encryption check, so a tampered ciphertext still "works"."""

    def decaps(self, decapsulation_key: ArrayLike, ciphertext: ArrayLike) -> Array:
        secret = fnp.asarray(decapsulation_key, dtype=fnp.uint8)[
            ..., : self.shared_secret_size
        ]
        message = (
            fnp.asarray(ciphertext, dtype=fnp.uint8)[..., : self.shared_secret_size]
            - secret
        )
        return self._derive(message)


class KemHarnessTest(absltest.TestCase):
    def test_a_conforming_scheme_passes_end_to_end(self) -> None:
        scheme = ToyKem(domain=7)
        check_kem(scheme, _kem_vectors(scheme))

    def test_an_empty_set_is_refused(self) -> None:
        with self.assertRaisesRegex(KatError, "empty set"):
            check_kem(ToyKem(domain=7), [])

    def test_encapsulation_alone_is_refused(self) -> None:
        # The enforceable form of "negatives are mandatory": ACVP marks no
        # decapsulation case as a rejection, so what the harness can require is
        # that the run reaches decapsulation at all.
        scheme = ToyKem(domain=7)
        positives = [v for v in _kem_vectors(scheme) if v.function != "decapsulation"]
        with self.assertRaisesRegex(KatError, "no decapsulation vectors"):
            check_kem(scheme, positives)

    def test_a_key_check_function_is_refused_not_skipped(self) -> None:
        scheme = ToyKem(domain=7)
        vectors = _kem_vectors(scheme)
        vectors.append(
            KemVector(
                case_id="keycheck/0",
                parameter_set=_PARAMS,
                function="encapsulationKeyCheck",
                encapsulation_key=b"\x00" * scheme.encapsulation_key_size,
                valid=False,
            )
        )
        with self.assertRaisesRegex(KatError, "does not name"):
            check_kem(scheme, vectors)

    def test_mixed_parameter_sets_are_refused(self) -> None:
        scheme = ToyKem(domain=7)
        vectors = _kem_vectors(scheme)
        vectors[0] = replace(vectors[0], parameter_set="other")
        with self.assertRaisesRegex(KatError, "one parameter set"):
            check_kem(scheme, vectors)

    def test_an_unexpressible_field_is_refused(self) -> None:
        scheme = ToyKem(domain=7)
        vectors = _kem_vectors(scheme)
        vectors[0] = replace(vectors[0], unsupported=("someNewMode",))
        with self.assertRaisesRegex(KatError, "cannot express"):
            check_kem(scheme, vectors)

    def test_a_scheme_that_never_rejects_is_caught(self) -> None:
        # Caught by the *trailing* flip specifically: NonRejectingKem reads only
        # the message-bearing head of the ciphertext, so a leading flip changes
        # its answer and proves nothing. Asserting the position keeps this test
        # honest about which check does the work.
        scheme = NonRejectingKem(domain=7)
        with self.assertRaisesRegex(
            KatError, "trailing ciphertext bytes.*not consumed"
        ):
            check_kem(scheme, _kem_vectors(ToyKem(domain=7)))


class AeadHarnessTest(absltest.TestCase):
    def test_a_conforming_scheme_passes_end_to_end(self) -> None:
        scheme = ToyAead(domain=7)
        check_aead(scheme, _aead_vectors(scheme))

    def test_an_empty_set_is_refused(self) -> None:
        with self.assertRaisesRegex(KatError, "empty set"):
            check_aead(ToyAead(domain=7), [])

    def test_a_scheme_that_releases_unverified_plaintext_is_caught(self) -> None:
        base = ToyAead(domain=7)
        with self.assertRaisesRegex(KatError, "non-zero"):
            check_aead(LeakyAead(domain=7), _aead_vectors(base))

    def test_a_batch_wide_verdict_is_caught(self) -> None:
        base = ToyAead(domain=7)
        with self.assertRaisesRegex(KatError, "not deciding per entry"):
            check_aead(BatchWideAead(domain=7), _aead_vectors(base))

    def test_a_published_failure_is_honored(self) -> None:
        # A set carrying a deliberate failure must be reported as rejected rather
        # than quietly opened; CAVP's GCM decrypt sets are mostly these.
        scheme = ToyAead(domain=7)
        vectors = _aead_vectors(scheme)
        corrupted = bytes([vectors[0].ciphertext[0] ^ 1]) + vectors[0].ciphertext[1:]
        vectors[0] = replace(vectors[0], ciphertext=corrupted, valid=False)
        check_aead(scheme, vectors)


if __name__ == "__main__":
    absltest.main()
