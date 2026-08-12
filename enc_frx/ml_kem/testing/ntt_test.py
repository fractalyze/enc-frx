# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM's NTT against FIPS 203's own algorithms, and against its shape.

The reference pins the arithmetic. The structural tests gate what a
round-trip cannot see, and here that is unusually important: **any** primitive
256th root yields a transform that round-trips and whose pointwise product is a
negacyclic convolution, so a wrong root passes every self-consistent check while
disagreeing with every other ML-KEM implementation on the wire. Only
`test_matches_fips203_ntt` can fail on that, which is why it is the load-bearing test
rather than the round trip.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest

from enc_frx.ml_kem import ntt as mlkem_ntt
from enc_frx.ml_kem.testing import fips203_reference as ref


def _random_poly(rng: np.random.Generator) -> list[int]:
    return rng.integers(0, ref.Q, size=ref.N, dtype=np.int64).tolist()


class NttTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rng = np.random.default_rng(20260812)

    def test_matches_fips203_ntt(self) -> None:
        """The forward transform, coefficient for coefficient."""
        for _ in range(4):
            f = _random_poly(self.rng)
            got = mlkem_ntt.as_ints(mlkem_ntt.ntt(mlkem_ntt.as_field(f)))
            self.assertEqual(np.asarray(got).tolist(), ref.ntt(f))

    def test_matches_fips203_intt(self) -> None:
        """The inverse, including its `128^-1` scale."""
        for _ in range(4):
            f_hat = _random_poly(self.rng)
            got = mlkem_ntt.as_ints(mlkem_ntt.intt(mlkem_ntt.as_field(f_hat)))
            self.assertEqual(np.asarray(got).tolist(), ref.intt(f_hat))

    def test_matches_fips203_base_mul(self) -> None:
        """The degree-1 products, which a pointwise multiply would fail."""
        f_hat, g_hat = _random_poly(self.rng), _random_poly(self.rng)
        got = mlkem_ntt.as_ints(
            mlkem_ntt.base_mul(mlkem_ntt.as_field(f_hat), mlkem_ntt.as_field(g_hat))
        )
        self.assertEqual(np.asarray(got).tolist(), ref.multiply_ntts(f_hat, g_hat))

    def test_base_mul_is_not_pointwise(self) -> None:
        """Guards the ML-DSA-shaped mistake explicitly.

        Without it, a pointwise implementation would still pass a round trip and
        would fail `test_matches_fips203_base_mul` for reasons a reader has to
        derive. This says what is wrong.
        """
        f_hat, g_hat = _random_poly(self.rng), _random_poly(self.rng)
        got = np.asarray(
            mlkem_ntt.as_ints(
                mlkem_ntt.base_mul(mlkem_ntt.as_field(f_hat), mlkem_ntt.as_field(g_hat))
            )
        ).tolist()
        pointwise = [(a * b) % ref.Q for a, b in zip(f_hat, g_hat)]
        self.assertNotEqual(got, pointwise)

    def test_multiplies_in_the_negacyclic_ring(self) -> None:
        """The property the whole domain exists for, independent of the transform.

        `intt(base_mul(ntt(f), ntt(g)))` is `f * g` in `Z_q[X]/(X^256 + 1)`,
        where wrap-around comes back negated. Pins the three as a composition
        rather than each against itself.
        """
        f, g = _random_poly(self.rng), _random_poly(self.rng)
        got = mlkem_ntt.as_ints(
            mlkem_ntt.intt(
                mlkem_ntt.base_mul(
                    mlkem_ntt.ntt(mlkem_ntt.as_field(f)),
                    mlkem_ntt.ntt(mlkem_ntt.as_field(g)),
                )
            )
        )
        self.assertEqual(np.asarray(got).tolist(), ref.negacyclic_convolution(f, g))

    def test_round_trips(self) -> None:
        f = _random_poly(self.rng)
        got = mlkem_ntt.as_ints(mlkem_ntt.intt(mlkem_ntt.ntt(mlkem_ntt.as_field(f))))
        self.assertEqual(np.asarray(got).tolist(), f)

    def test_batches_over_leading_axes(self) -> None:
        """A `[2, 3, 256]` batch is the per-polynomial answer, entry for entry.

        Batch-first is the point of routing through the opcode: this shape is two
        calls rather than twelve.
        """
        polys = [[_random_poly(self.rng) for _ in range(3)] for _ in range(2)]
        batched = np.asarray(
            mlkem_ntt.as_ints(mlkem_ntt.ntt(mlkem_ntt.as_field(polys)))
        )
        self.assertEqual(batched.shape, (2, 3, ref.N))
        for i in range(2):
            for j in range(3):
                self.assertEqual(batched[i][j].tolist(), ref.ntt(polys[i][j]))

    def test_generator_is_the_standards_zeta(self) -> None:
        """`_GENERATOR^13 == 17`, the reason the values match FIPS 203 at all.

        Cheap to state and expensive to rediscover: unpinned, the opcode finds a
        different primitive 256th root and every other test in this file except
        the FIPS-203 ones would still pass.
        """
        self.assertEqual(
            pow(mlkem_ntt._GENERATOR, (ref.Q - 1) // ref.N, ref.Q), ref.ZETA
        )

    def test_jits(self) -> None:
        """The transform traces, which is how every caller will use it."""
        f = mlkem_ntt.as_field(_random_poly(self.rng))
        jitted = frx.jit(mlkem_ntt.ntt)(f)
        self.assertEqual(
            np.asarray(mlkem_ntt.as_ints(jitted)).tolist(),
            np.asarray(mlkem_ntt.as_ints(mlkem_ntt.ntt(f))).tolist(),
        )


if __name__ == "__main__":
    absltest.main()
