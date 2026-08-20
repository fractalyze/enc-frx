# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every case ACVP publishes for AES-CCM. Tagged `slow_kat`.

8310 cases across 147 (key, nonce, tag) instances; the decrypt groups carry
published `testPassed: false` rows, so grouping by shape mixes verdicts inside
a batch here the way the GCM sweep's does. The per-PR gate is `ccm_test`,
which runs the corners with the tampering pass; this runs the whole space
without it — the split `gcm_sweep_test` explains. Nothing is refused: unlike
GCM, `AesCcm` admits the full published tag list, so the accounting here is
simply that every case ran.
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from enc_frx.aes.ccm import AesCcm
from enc_frx.aes.testing import ccm_vectors
from enc_frx.testing.kat import check_aead_published


class SweepTest(parameterized.TestCase):
    @parameterized.parameters(16, 24, 32)
    def test_every_published_case(self, key_size: int) -> None:
        run = 0
        for vector_set in ccm_vectors.sets():
            if vector_set.key_size != key_size:
                continue
            check_aead_published(
                AesCcm(
                    vector_set.key_size,
                    vector_set.nonce_size,
                    vector_set.tag_size,
                ),
                vector_set.vectors,
            )
            run += len(vector_set.vectors)
        # Counted independently of the loop above, so a filter bug that
        # silently dropped a set would leave the two unequal rather than
        # both shrinking in step.
        published = sum(
            len(s.vectors) for s in ccm_vectors.sets() if s.key_size == key_size
        )
        self.assertGreater(run, 0, key_size)
        self.assertEqual(run, published, key_size)


if __name__ == "__main__":
    absltest.main()
