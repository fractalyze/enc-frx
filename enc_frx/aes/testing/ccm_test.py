# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES-CCM against ACVP's vectors, and against the rules a vector cannot see.

The gate that runs per PR: the corners of the published parameter space — every
key length crossed with the extreme nonce and tag lengths, since `q = 15 - n`
makes the nonce length change the whole `B_0`/`A_i` layout rather than only a
width — each driven through `check_aead` on a batch that carries both a
tampering-ready shape group and a published-rejection group. The exhaustive
pass over all 8310 cases is `ccm_sweep_test`, tagged `slow_kat`.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.aes import ccm
from enc_frx.aes.testing import ccm_vectors
from enc_frx.testing.kat import check_aead

# Every key length at the corners of the (nonce, tag) space, plus the GCM-like
# center (13, 16) a caller is most likely to pick.
_CORNERS = tuple(
    (key_size, nonce_size, tag_size)
    for key_size in (16, 24, 32)
    for (nonce_size, tag_size) in ((7, 4), (7, 16), (13, 4), (13, 16), (11, 8))
)


class CcmTest(parameterized.TestCase):
    @parameterized.parameters(*_CORNERS)
    def test_acvp_corner(self, key_size: int, nonce_size: int, tag_size: int) -> None:
        vector_set = ccm_vectors.instance(key_size, nonce_size, tag_size)
        check_aead(
            ccm.AesCcm(key_size, nonce_size, tag_size),
            ccm_vectors.gate_batch(vector_set),
        )

    def test_sizes_outside_the_standard_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ccm.AesCcm(nonce_size=6)
        with self.assertRaises(ValueError):
            ccm.AesCcm(nonce_size=14)
        with self.assertRaises(ValueError):
            ccm.AesCcm(tag_size=5)
        with self.assertRaises(ValueError):
            ccm.AesCcm(key_size=20)

    def test_value_equality_over_full_parameter_surface(self) -> None:
        self.assertEqual(ccm.AesCcm(16, 13, 16), ccm.AesCcm(16, 13, 16))
        self.assertEqual(hash(ccm.AesCcm(16, 7, 4)), hash(ccm.AesCcm(16, 7, 4)))
        self.assertNotEqual(ccm.AesCcm(16, 13, 16), ccm.AesCcm(16, 13, 14))
        self.assertNotEqual(ccm.AesCcm(16, 13, 16), ccm.AesCcm(16, 12, 16))
        self.assertNotEqual(ccm.AesCcm(16, 13, 16), ccm.AesCcm(32, 13, 16))

    def test_traced_matches_eager(self) -> None:
        scheme = ccm.AesCcm(16, 13, 16)
        rng = np.random.default_rng(0)
        key = rng.integers(0, 256, size=(3, 16), dtype=np.uint8)
        nonce = rng.integers(0, 256, size=(3, 13), dtype=np.uint8)
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
