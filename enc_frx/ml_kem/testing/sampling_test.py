# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The §4.2.2 samplers, against the standard's own intermediate values.

Three gates, and they catch different things.

**CCTV's `intermediate/` files** are the primary one. They publish `ρ`, `σ` and
the resulting `A`, `s`, `e`, `r`, `e1` and `e2` as separate labelled values, so
each sampler is pinned on its own rather than jointly through a key generation
that has not been written yet. All three parameter sets are loaded because `η1`
is 3 for ML-KEM-512 and 2 for the other two — one file gates one width.

**CCTV's `unluckysample/` seeds** gate the fixed XOF budget, and nothing else
does. They are the worst rejection runs known, at 384 candidates against a mean
of 315, and a three-block implementation passes every other vector in this file
and fails only here.

**The reference oracle** sweeps random seeds, which the published vectors cannot:
there are a handful of those and a bug can sit between them. It is also the only
gate that is *obviously* the standard, so a disagreement localizes.

What none of them catch is the thing the module's shape exists for, so it is
asserted directly: that the traced path has no data-dependent control flow.
"""

from __future__ import annotations

import hashlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from python.runfiles import runfiles

from enc_frx.ml_kem import hashes, sampling
from enc_frx.ml_kem.encoding import byte_decode
from enc_frx.ml_kem.ntt import as_ints
from enc_frx.ml_kem.params import N
from enc_frx.ml_kem.testing import cctv_vectors
from enc_frx.ml_kem.testing import fips203_reference as ref

# (parameter set, k, eta1, eta2) — FIPS 203 Table 2.
_PARAMETER_SETS = (
    ("ML-KEM-512", 2, 3, 2),
    ("ML-KEM-768", 3, 2, 2),
    ("ML-KEM-1024", 4, 2, 2),
)


def _path(repo: str, name: str) -> str:
    location = runfiles.Create().Rlocation(f"{repo}/file/{name}")
    assert location is not None, f"{repo} not in runfiles"
    return location


def _intermediate(parameter_set: str) -> cctv_vectors.Intermediate:
    bits = parameter_set.split("-")[-1]
    return cctv_vectors.load_intermediate(
        _path(f"cctv_ml_kem_intermediate_{bits}", f"ML-KEM-{bits}.txt")
    )


def _decode_vector(packed: bytes) -> np.ndarray:
    """ByteEncode_12 of `k` polynomials back to `[k, 256]` integers."""
    data = np.frombuffer(packed, dtype=np.uint8).reshape(-1, 384)
    return np.asarray(byte_decode(data, 12))


class SampleNttTest(parameterized.TestCase):
    @parameterized.named_parameters(*[(p[0], *p) for p in _PARAMETER_SETS])
    def test_matrix_matches_the_published_intermediate_value(
        self, parameter_set: str, k: int, _eta1: int, _eta2: int
    ) -> None:
        vectors = _intermediate(parameter_set)
        rho = vectors.hex_at("ρ")
        want = _decode_vector(vectors.hex_at("A")).reshape(k, k, N)

        got = as_ints(sampling.expand_matrix(np.frombuffer(rho, dtype=np.uint8), k))
        np.testing.assert_array_equal(np.asarray(got), want)

    @parameterized.named_parameters(*[(p[0], *p) for p in _PARAMETER_SETS])
    def test_the_first_entry_matches_the_published_coefficients(
        self, parameter_set: str, k: int, _eta1: int, _eta2: int
    ) -> None:
        # `A[0, 0]` is published as decimals as well as inside the packed `A`, so
        # this is the one check that does not route through `byte_decode` — a
        # decoder bug cannot make both pass.
        vectors = _intermediate(parameter_set)
        rho = np.frombuffer(vectors.hex_at("ρ"), dtype=np.uint8)
        got = as_ints(sampling.expand_matrix(rho, k))
        np.testing.assert_array_equal(
            np.asarray(got)[0, 0], np.array(vectors.ints_at("A[0, 0]"))
        )

    def test_the_column_index_is_absorbed_first(self) -> None:
        # `Â[i, j] = SampleNTT(rho ‖ j ‖ i)`. The transposed reading is a
        # self-consistent scheme, so only a published vector separates them —
        # this asserts the matrix is not symmetric, which is what makes the
        # vector check above able to tell.
        vectors = _intermediate("ML-KEM-512")
        rho = np.frombuffer(vectors.hex_at("ρ"), dtype=np.uint8)
        got = np.asarray(as_ints(sampling.expand_matrix(rho, 2)))
        self.assertFalse(np.array_equal(got[0, 1], got[1, 0]))

    def test_matches_the_oracle_on_random_seeds(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(4):
            rho = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
            seeds = np.array(
                [list(rho) + [j, i] for i in range(2) for j in range(2)], dtype=np.uint8
            )
            got = np.asarray(as_ints(sampling.sample_ntt(seeds)))
            for row, (i, j) in enumerate((i, j) for i in range(2) for j in range(2)):
                want = ref.sample_ntt(ref.shake128_stream(rho + bytes([j, i])))
                np.testing.assert_array_equal(
                    got[row], np.array(want), err_msg=f"{i},{j}"
                )

    def test_every_coefficient_is_reduced(self) -> None:
        rng = np.random.default_rng(1)
        seeds = rng.integers(0, 256, size=(16, 34), dtype=np.uint8)
        got = np.asarray(as_ints(sampling.sample_ntt(seeds)))
        # Bound from the reference, not from the module under test: a wrong
        # constant there would otherwise sweep its own wrong domain.
        self.assertTrue((got >= 0).all() and (got < ref.Q).all())

    def test_rejects_a_seed_of_the_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            sampling.sample_ntt(np.zeros((1, 32), dtype=np.uint8))


class UnluckySeedTest(parameterized.TestCase):
    """The vectors that see an undersized XOF budget, and nothing else does."""

    @parameterized.named_parameters(*[(p[0], p[0]) for p in _PARAMETER_SETS])
    def test_the_worst_known_rejection_runs_still_fit(self, parameter_set: str) -> None:
        bits = parameter_set.split("-")[-1]
        seeds = cctv_vectors.load_unlucky_seeds(
            _path(f"cctv_ml_kem_unlucky_{bits}", f"ML-KEM-{bits}.txt")
        )
        self.assertNotEmpty(seeds)
        for seed in seeds:
            # `rho` is the first half of `G(d)`; taken from `hashlib` rather than
            # from `hashes.g` so this gates the sampler alone.
            rho = hashlib.sha3_512(seed).digest()[:32]
            xof_seed = np.frombuffer(rho + bytes([0, 0]), dtype=np.uint8)[None, :]
            got = np.asarray(as_ints(sampling.sample_ntt(xof_seed)))[0]
            want = ref.sample_ntt(ref.shake128_stream(rho + bytes([0, 0])))
            np.testing.assert_array_equal(got, np.array(want), err_msg=seed.hex())

    def test_the_budget_covers_the_worst_known_run_with_headroom(self) -> None:
        # States the margin as a number rather than leaving it to the pass above,
        # so shrinking the budget fails here with the reason attached.
        seeds = cctv_vectors.load_unlucky_seeds(
            _path("cctv_ml_kem_unlucky_512", "ML-KEM-512.txt")
        )
        worst = 0
        for seed in seeds:
            rho = hashlib.sha3_512(seed).digest()[:32]
            stream = np.asarray(
                hashes.xof(
                    sampling.XOF_BYTES,
                    np.frombuffer(rho + bytes([0, 0]), dtype=np.uint8)[None, :],
                )
            )[0].astype(np.int64)
            groups = stream[: sampling.GROUPS * 3].reshape(sampling.GROUPS, 3)
            d1 = groups[:, 0] + 256 * (groups[:, 1] % 16)
            d2 = (groups[:, 1] // 16) + 16 * groups[:, 2]
            candidates = np.stack([d1, d2], axis=-1).reshape(-1)
            accepted = np.cumsum(candidates < ref.Q)
            worst = max(worst, int(np.argmax(accepted >= N)) + 1)
        # C2SP's generator reports 384 candidates as the worst it found.
        self.assertEqual(worst, 384)
        self.assertLess(worst, sampling.CANDIDATES)


class BudgetMissTest(absltest.TestCase):
    """What happens below 256 acceptances — a `2^-261` event, pinned anyway.

    Not reachable from any seed, so it is driven through `_compact` directly.
    The point is not that the answer is right, because it cannot be; it is that
    the answer is the *same* everywhere. An unclamped gather fills out-of-bounds
    slots with `INT32_MIN`, which is the backend's choice rather than this
    module's, so `_compact` pins the index and these assert that it did.
    """

    def _starved(self) -> fnp.ndarray:
        # Three acceptances in a stream of 560; every other value is above q.
        candidates = fnp.full((1, sampling.CANDIDATES), 4000, dtype=fnp.int32)
        return candidates.at[0, 0].set(11).at[0, 5].set(22).at[0, 9].set(33)

    def test_the_accepted_prefix_is_still_correct(self) -> None:
        got = np.asarray(sampling._compact(self._starved()))[0]
        np.testing.assert_array_equal(got[:3], np.array([11, 22, 33]))

    def test_the_unfilled_tail_reads_the_final_candidate(self) -> None:
        starved = self._starved()
        got = np.asarray(sampling._compact(starved))[0]
        final = int(np.asarray(starved)[0, -1])
        self.assertTrue((got[3:] == final).all())
        # The failure this guards: an unclamped gather puts INT32_MIN here.
        self.assertGreaterEqual(int(got.min()), 0)

    def test_the_miss_is_the_same_eager_and_traced(self) -> None:
        starved = self._starved()
        np.testing.assert_array_equal(
            np.asarray(sampling._compact(starved)),
            np.asarray(frx.jit(sampling._compact)(starved)),
        )


class SamplePolyCbdTest(parameterized.TestCase):
    @parameterized.named_parameters(*[(p[0], *p) for p in _PARAMETER_SETS])
    def test_key_generation_vectors_match(
        self, parameter_set: str, k: int, eta1: int, _eta2: int
    ) -> None:
        # Algorithm 13: `s` takes nonces 0..k-1 and `e` takes k..2k-1, both at
        # eta1, both from sigma.
        vectors = _intermediate(parameter_set)
        sigma = np.frombuffer(vectors.hex_at("σ"), dtype=np.uint8)
        nonces = np.arange(2 * k, dtype=np.uint8)
        got = np.asarray(
            as_ints(sampling.sample_poly_cbd(hashes.prf(eta1, sigma, nonces), eta1))
        )
        np.testing.assert_array_equal(got[:k], _decode_vector(vectors.hex_at("s")))
        np.testing.assert_array_equal(got[k:], _decode_vector(vectors.hex_at("e")))

    @parameterized.named_parameters(*[(p[0], *p) for p in _PARAMETER_SETS])
    def test_encryption_vectors_match_at_both_etas(
        self, parameter_set: str, k: int, eta1: int, eta2: int
    ) -> None:
        # Algorithm 14: `r` takes nonces 0..k-1 at eta1, then `e1` takes k..2k-1
        # and `e2` takes 2k, both at eta2. The two widths and the nonce that
        # continues across them are the part a per-vector transcription gets
        # wrong.
        vectors = _intermediate(parameter_set)
        seed = np.frombuffer(vectors.hex_at("r", 0), dtype=np.uint8)

        r = sampling.sample_poly_cbd(
            hashes.prf(eta1, seed, np.arange(k, dtype=np.uint8)), eta1
        )
        np.testing.assert_array_equal(
            np.asarray(as_ints(r)), _decode_vector(vectors.hex_at("r", 1))
        )

        tail = sampling.sample_poly_cbd(
            hashes.prf(eta2, seed, np.arange(k, 2 * k + 1, dtype=np.uint8)), eta2
        )
        packed = np.asarray(as_ints(tail))
        np.testing.assert_array_equal(packed[:k], _decode_vector(vectors.hex_at("e1")))
        np.testing.assert_array_equal(packed[k:], _decode_vector(vectors.hex_at("e2")))

    @parameterized.parameters(2, 3)
    def test_matches_the_oracle_on_random_input(self, eta: int) -> None:
        rng = np.random.default_rng(eta)
        data = rng.integers(0, 256, size=(8, 64 * eta), dtype=np.uint8)
        got = np.asarray(as_ints(sampling.sample_poly_cbd(data, eta)))
        for row in range(data.shape[0]):
            # Plain `int`s: the oracle reduces mod q, which overflows a uint8.
            want = ref.sample_poly_cbd([int(b) for b in data[row]], eta)
            np.testing.assert_array_equal(got[row], np.array(want))

    @parameterized.parameters(2, 3)
    def test_coefficients_land_in_the_centered_range(self, eta: int) -> None:
        rng = np.random.default_rng(100 + eta)
        data = rng.integers(0, 256, size=(64, 64 * eta), dtype=np.uint8)
        got = np.asarray(as_ints(sampling.sample_poly_cbd(data, eta)))
        # `x - y` lies in [-eta, eta], so after reduction every value is either
        # small or within eta of q. Nothing in between may appear.
        centered = np.where(got > ref.Q // 2, got - ref.Q, got)
        self.assertTrue((np.abs(centered) <= eta).all())

    @parameterized.parameters(2, 3)
    def test_rejects_the_wrong_input_length(self, eta: int) -> None:
        with self.assertRaises(ValueError):
            sampling.sample_poly_cbd(np.zeros((1, 64 * eta + 1), dtype=np.uint8), eta)

    def test_rejects_an_eta_the_standard_does_not_use(self) -> None:
        with self.assertRaises(ValueError):
            sampling.sample_poly_cbd(np.zeros((1, 64), dtype=np.uint8), 1)


class TracedShapeTest(absltest.TestCase):
    """No data-dependent control flow, asserted on the lowering rather than hoped.

    The whole reason `sample_ntt` is a fixed squeeze instead of the standard's
    `while` is that a traced program cannot have a data-dependent trip count. A
    reviewer cannot see from the source that `searchsorted` was asked for the
    unrolled binary search rather than the scanning default, so the property is
    read back off the compiled program.
    """

    def _lowered(self) -> str:
        seeds = fnp.zeros((4, sampling.MATRIX_SEED_SIZE), dtype=fnp.uint8)
        return frx.jit(sampling.sample_ntt).lower(seeds).as_text()

    def test_the_sampling_path_has_no_data_dependent_control_flow(self) -> None:
        text = self._lowered()
        for op in ("stablehlo.while", "stablehlo.case", "stablehlo.if"):
            self.assertNotIn(op, text)

    def test_the_sampling_path_has_no_host_callback(self) -> None:
        # `custom_call` is what a host callback lowers to; asserting on the op
        # rather than on the word "host" keeps this from passing or failing on
        # an unrelated attribute that happens to contain it.
        self.assertNotIn("custom_call", self._lowered())

    def test_the_compaction_is_a_gather_and_not_a_scatter(self) -> None:
        # The design claim of `_compact`, and the one a reader cannot check from
        # the source: `searchsorted` + `take_along_axis` must lower to a gather.
        # A scatter here would still be correct and would serialize on GPU.
        text = self._lowered()
        self.assertIn("stablehlo.gather", text)
        self.assertNotIn("stablehlo.scatter", text)
        # `sort` is the other compaction that would work and cost more.
        self.assertNotIn("stablehlo.sort", text)

    def test_sample_poly_cbd_traces_without_a_branch(self) -> None:
        data = fnp.zeros((4, 64 * 2), dtype=fnp.uint8)
        text = frx.jit(lambda d: sampling.sample_poly_cbd(d, 2)).lower(data).as_text()
        for op in ("stablehlo.while", "stablehlo.case", "stablehlo.if"):
            self.assertNotIn(op, text)


if __name__ == "__main__":
    absltest.main()
