# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The binary fields AES-GCM is defined over are reachable as native dtypes.

AES's S-box and MixColumns live in GF(2^8) with the reduction polynomial
x^8 + x^4 + x^3 + x + 1; GHASH lives in GF(2^128) with x^128 + x^7 + x^2 + x + 1.
Both are registered dtypes here — `binary_field_gf8_aes` and
`binary_field_ghash` — which is why AES-GCM is written in this stack rather than
called out to a library: the S-box becomes inversion plus an affine map instead
of a 256-entry table indexed by a secret byte.

That makes the dtype floor load-bearing rather than incidental, and this guards
it. `zk_dtypes` arrives through frx transitively, so a frx bump could otherwise
remove these dtypes with nothing failing until AES-GCM was written.

The values are the specifications' own published examples, so this checks the
reduction polynomial rather than merely checking that multiplication runs.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
import zk_dtypes
from absl.testing import absltest


class AesFieldTest(absltest.TestCase):
    def test_multiplication_matches_the_published_example(self) -> None:
        # FIPS 197 §4.2: 0x57 · 0x83 = 0xC1.
        a = fnp.asarray(np.array([0x57], dtype=zk_dtypes.binary_field_gf8_aes))
        b = fnp.asarray(np.array([0x83], dtype=zk_dtypes.binary_field_gf8_aes))
        self.assertEqual(int(np.asarray(a * b).astype(np.uint8)[0]), 0xC1)

    def test_xtime_reduces_by_the_aes_polynomial(self) -> None:
        # FIPS 197 §4.2.1: xtime(0x87) = 0x15, which only holds under
        # x^8 + x^4 + x^3 + x + 1.
        x = fnp.asarray(np.array([0x87], dtype=zk_dtypes.binary_field_gf8_aes))
        two = fnp.asarray(np.array([0x02], dtype=zk_dtypes.binary_field_gf8_aes))
        self.assertEqual(int(np.asarray(x * two).astype(np.uint8)[0]), 0x15)

    def test_addition_is_xor(self) -> None:
        a = fnp.asarray(np.array([0x53], dtype=zk_dtypes.binary_field_gf8_aes))
        b = fnp.asarray(np.array([0xCA], dtype=zk_dtypes.binary_field_gf8_aes))
        self.assertEqual(int(np.asarray(a + b).astype(np.uint8)[0]), 0x53 ^ 0xCA)


class GhashFieldTest(absltest.TestCase):
    def test_the_element_is_128_bits_wide(self) -> None:
        x = np.array([1], dtype=zk_dtypes.binary_field_ghash)
        self.assertEqual(x.itemsize, 16)

    def test_reduction_is_gcms_polynomial_in_the_natural_basis(self) -> None:
        # x^127 · x reduces to x^7 + x^2 + x + 1 = 0x87, which fixes both the
        # reduction polynomial and the basis: bit i of the integer is the
        # coefficient of x^i.
        #
        # GCM specifies the *reflected* order — its bit 0 is the leading
        # coefficient — so a GHASH implementation reverses at the boundary
        # rather than feeding the wire bytes in directly. This test is where
        # that convention is pinned down.
        hi = np.array([1 << 127], dtype=zk_dtypes.binary_field_ghash)
        one_x = np.array([2], dtype=zk_dtypes.binary_field_ghash)
        self.assertEqual(int((hi * one_x).astype(object)[0]), 0x87)


if __name__ == "__main__":
    absltest.main()
