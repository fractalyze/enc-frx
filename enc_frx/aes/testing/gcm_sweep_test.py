# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Every case CAVP publishes for AES-GCM. Tagged `slow_kat`.

47250 cases: three key lengths, three IV lengths, seven tag lengths and five
payload lengths, of which 11908 are decrypt cases whose published expected result
is `FAIL`. The per-PR gate is `gcm_test`, which runs a batch out of each corner of
that space with the tampering pass; this runs the whole space without it.

The division is a cost decision, and the halves are not interchangeable.
`check_aead` derives a tampered batch per entry of a batch, so it is quadratic in
a group and a full set at that depth would take hours rather than minutes. What it
sees that this cannot: a verdict mixed inside one batch, a moved nonce, a moved
associated-data byte. What this sees that the gate cannot: the other 47000 cases.
So `check_aead_published` is used here, and its docstring says plainly that a
scheme gated on it alone would prove nothing about rejection.

The 32- and 64-bit tag sections are refused rather than run. They are SP 800-38D
Appendix C's, admitted only under invocation and payload limits that live with the
caller, and `AesGcm` does not offer them — see `TAG_SIZES`. Counting the refusals
here is what keeps "refused" from decaying into "skipped".
"""

from __future__ import annotations

from absl.testing import absltest, parameterized

from enc_frx.aes.gcm import TAG_SIZES, AesGcm
from enc_frx.aes.testing import gcm_vectors
from enc_frx.testing.kat import check_aead_published


class SweepTest(parameterized.TestCase):
    @parameterized.parameters(*gcm_vectors.FILES)
    def test_every_published_case(self, name: str) -> None:
        run = refused = 0
        for vector_set in gcm_vectors.sets(name):
            if vector_set.tag_size not in TAG_SIZES:
                refused += len(vector_set.vectors)
                continue
            check_aead_published(
                AesGcm(vector_set.key_size, vector_set.nonce_size, vector_set.tag_size),
                vector_set.vectors,
            )
            run += len(vector_set.vectors)

        # Every case was either run or refused by name, and neither count is
        # zero. This is the assertion that keeps "refused" from decaying into
        # "skipped": a filter that quietly dropped a section would leave the two
        # summing to less than the file holds, and a suite reporting green over
        # a subset is the failure mode the harness exists to prevent.
        self.assertGreater(run, 0, name)
        self.assertGreater(refused, 0, name)
        self.assertEqual(
            run + refused,
            sum(len(s.vectors) for s in gcm_vectors.sets(name)),
            name,
        )


if __name__ == "__main__":
    absltest.main()
