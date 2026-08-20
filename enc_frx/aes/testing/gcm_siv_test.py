# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES-GCM-SIV against RFC 8452 Appendix C, through the full `check_aead`
gate — every published case for both key sizes (the C.3 counter-wrap rows
included, which is what pins the little-endian uint32 wrap in the keystream),
plus the tampering and per-entry masking checks no published set carries.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest

from enc_frx.aes import gcm_siv
from enc_frx.aes.testing.rfc8452_vectors import CASES
from enc_frx.testing.kat import AeadVector, check_aead


def _vectors(key_size: int) -> list[AeadVector]:
    rows = []
    for index, (key, nonce, aad, plaintext, result) in enumerate(CASES):
        if len(key) != 2 * key_size:
            continue
        rows.append(
            AeadVector(
                case_id=f"rfc8452-C-{key_size * 8}-{index}",
                parameter_set=f"AEAD_AES_{key_size * 8}_GCM_SIV",
                key=bytes.fromhex(key),
                nonce=bytes.fromhex(nonce),
                plaintext=bytes.fromhex(plaintext),
                ciphertext=bytes.fromhex(result),
                # A zero-length AAD is the absent AAD; None also keeps the
                # tamper driver from flipping a bit that does not exist.
                associated_data=bytes.fromhex(aad) or None,
            )
        )
    return rows


class GcmSivTest(absltest.TestCase):
    def test_rfc8452_vectors_aes_128(self) -> None:
        check_aead(gcm_siv.AesGcmSiv(16), _vectors(16))

    def test_rfc8452_vectors_aes_256(self) -> None:
        check_aead(gcm_siv.AesGcmSiv(32), _vectors(32))

    def test_key_sizes_the_rfc_does_not_define_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gcm_siv.AesGcmSiv(24)

    def test_wrong_width_arguments_are_rejected(self) -> None:
        scheme = gcm_siv.AesGcmSiv(16)
        key = np.zeros((1, 16), np.uint8)
        with self.assertRaises(ValueError):
            scheme.seal(key, np.zeros((1, 11), np.uint8), None, key)
        with self.assertRaises(ValueError):
            scheme.open(
                key, np.zeros((1, 12), np.uint8), None, np.zeros((1, 15), np.uint8)
            )

    def test_value_equality_over_the_parameter(self) -> None:
        self.assertEqual(gcm_siv.AesGcmSiv(16), gcm_siv.AesGcmSiv(16))
        self.assertEqual(hash(gcm_siv.AesGcmSiv(32)), hash(gcm_siv.AesGcmSiv(32)))
        self.assertNotEqual(gcm_siv.AesGcmSiv(16), gcm_siv.AesGcmSiv(32))

    def test_traced_matches_eager(self) -> None:
        # The gpu pipeline aborts compiling this whole-call trace:
        # MoveCopyToUsers rebuilds a convert to an algebraic dtype without its
        # element_algebra and HloInstruction check-fails (the byte-reversal
        # copies POLYVAL introduces sit next to the AES field converts, which
        # is the pattern that reaches it). Tracked as a compiler bug on the
        # fractalyze xla work board; the eager GPU path and every vector gate
        # above still run there.
        if frx.default_backend() == "gpu":
            self.skipTest("gpu pipeline check-fails on algebraic converts")
        scheme = gcm_siv.AesGcmSiv(16)
        rng = np.random.default_rng(0)
        key = rng.integers(0, 256, size=(3, 16), dtype=np.uint8)
        nonce = rng.integers(0, 256, size=(3, 12), dtype=np.uint8)
        aad = rng.integers(0, 256, size=(3, 7), dtype=np.uint8)
        plaintext = rng.integers(0, 256, size=(3, 40), dtype=np.uint8)

        eager = np.asarray(scheme.seal(key, nonce, aad, plaintext))
        traced = np.asarray(frx.jit(scheme.seal)(key, nonce, aad, plaintext))
        np.testing.assert_array_equal(traced, eager)

        opened, ok = frx.jit(scheme.open)(key, nonce, aad, traced)
        np.testing.assert_array_equal(np.asarray(opened), plaintext)
        self.assertTrue(bool(np.asarray(ok).all()))


if __name__ == "__main__":
    absltest.main()
