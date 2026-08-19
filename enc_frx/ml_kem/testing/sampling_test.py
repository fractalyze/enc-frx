# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The §4.2.2 samplers, against the standard's own intermediate values.

Three gates, and what makes them worth having together is that they fail on
different things:

- **CCTV's `intermediate/` files** pin each sampler against a published value.
- **CCTV's `unluckysample/` seeds** pin the XOF budget, which nothing else can.
- **The reference oracle** sweeps random seeds, where the published vectors are
  a handful of points a bug can sit between. It is also the only gate that is
  *obviously* the standard, so a disagreement localizes.

Why those vector sets and not ACVP's is stated at their declaration in
[`//MODULE.bazel`](../../../MODULE.bazel).

What none of them catch is the thing the module's shape exists for, so it is
asserted directly: that the traced path has no data-dependent control flow.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
from collections.abc import Iterator
from unittest import mock

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.ml_kem import hashes, sampling
from enc_frx.ml_kem.ntt import as_ints
from enc_frx.ml_kem.params import ML_KEM_512, PARAMETER_SETS, MlKemParams, N
from enc_frx.ml_kem.testing import cctv_vectors
from enc_frx.ml_kem.testing import fips203_reference as ref

_NAMED = tuple((params.name, params) for params in PARAMETER_SETS)


def _xof_seeds(seeds: tuple[bytes, ...]) -> np.ndarray:
    """`d` seeds to the `[n, 34]` batch `sample_ntt` takes.

    `rho` is the first half of `G(d)`, taken from `hashlib` rather than from
    `hashes.g` so these gate the sampler alone. Stacked into one batch because
    a Python loop over the batch axis is what this repo treats as the bug.
    """
    return np.array(
        [list(hashlib.sha3_512(seed).digest()[:32]) + [0, 0] for seed in seeds],
        dtype=np.uint8,
    )


@contextlib.contextmanager
def _compaction(scatter: bool) -> Iterator[None]:
    """Force one compaction form, whichever backend the suite is running on.

    `_compact` picks its form from `frx.default_backend()`, so a leg only ever
    traces one of the two. Pinning the tuple it reads is what lets both be
    asserted from either leg, and it is the only way the CPU leg covers the form
    the GPU leg ships.
    """
    backends = (frx.default_backend(),) if scatter else ()
    with mock.patch.object(sampling, "_SCATTER_BACKENDS", backends):
        yield


@functools.lru_cache(maxsize=None)
def _sample_ntt_lowering(scatter: bool) -> str:
    """`sample_ntt` lowered to StableHLO, once per compaction form.

    Cached because several tests read the same text, and a fresh `frx.jit`
    wrapper carries its own trace cache — building one per test would re-trace
    the whole program rather than hit a shared one.
    """
    seeds = fnp.zeros((4, sampling.MATRIX_SEED_SIZE), dtype=fnp.uint8)
    with _compaction(scatter):
        # Wrapped in a fresh lambda per call on purpose: the form is read from a
        # module global at trace time, so lowering `sample_ntt` itself twice
        # would answer the second form out of the first one's trace cache.
        traced = frx.jit(lambda s: sampling.sample_ntt(s))
        return str(traced.lower(seeds).as_text())


class SampleNttTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_matrix_matches_the_published_intermediate_value(
        self, params: MlKemParams
    ) -> None:
        k = params.k
        vectors = cctv_vectors.intermediate(params.name)
        want = cctv_vectors.decode_polys(vectors.hex_at("A"), k * k).reshape(k, k, N)

        got = as_ints(sampling.expand_matrix(vectors.array_at("ρ"), k))
        np.testing.assert_array_equal(np.asarray(got), want)

    @parameterized.named_parameters(*_NAMED)
    def test_the_first_entry_matches_the_published_coefficients(
        self, params: MlKemParams
    ) -> None:
        # `A[0, 0]` is published as decimals as well as inside the packed `A`, so
        # this is the one check that does not route through `byte_decode` — a
        # decoder bug cannot make both pass.
        vectors = cctv_vectors.intermediate(params.name)
        rho = vectors.array_at("ρ")
        got = as_ints(sampling.expand_matrix(rho, params.k))
        np.testing.assert_array_equal(
            np.asarray(got)[0, 0], np.array(vectors.ints_at("A[0, 0]"))
        )

    def test_the_column_index_is_absorbed_first(self) -> None:
        # `Â[i, j] = SampleNTT(rho ‖ j ‖ i)`. The transposed reading is a
        # self-consistent scheme, so only a published vector separates them —
        # this asserts the matrix is not symmetric, which is what makes the
        # vector check above able to tell.
        vectors = cctv_vectors.intermediate(ML_KEM_512.name)
        rho = vectors.array_at("ρ")
        got = np.asarray(as_ints(sampling.expand_matrix(rho, ML_KEM_512.k)))
        self.assertFalse(np.array_equal(got[0, 1], got[1, 0]))

    def test_matches_the_oracle_on_random_seeds(self) -> None:
        # Through `expand_matrix` against the oracle's own matrix expansion, so
        # the random sweep covers the `rho ‖ j ‖ i` index order too — the
        # published vectors are the only other thing that pins it.
        rng = np.random.default_rng(0)
        for _ in range(4):
            rho = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
            got = np.asarray(
                as_ints(sampling.expand_matrix(np.frombuffer(rho, dtype=np.uint8), 2))
            )
            np.testing.assert_array_equal(got, np.array(ref.sample_matrix(rho, 2)))

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

    @parameterized.named_parameters(*_NAMED)
    def test_the_worst_known_rejection_runs_still_fit(
        self, params: MlKemParams
    ) -> None:
        seeds = cctv_vectors.unlucky_seeds(params.name)
        self.assertNotEmpty(seeds)
        got = np.asarray(as_ints(sampling.sample_ntt(_xof_seeds(seeds))))
        for row, seed in enumerate(seeds):
            rho = hashlib.sha3_512(seed).digest()[:32]
            want = ref.sample_ntt(ref.shake128_stream(rho + bytes([0, 0])))
            np.testing.assert_array_equal(got[row], np.array(want), err_msg=seed.hex())

    def test_the_budget_covers_the_worst_known_run_with_headroom(self) -> None:
        # States the margin as a number rather than leaving it to the pass above,
        # so shrinking the budget fails here with the reason attached.
        #
        # Driven through `sampling._candidates` rather than a hand-copied bit
        # split: this test measures the margin, and independence from the
        # production reading is already `fips203_reference`'s job above.
        seeds = cctv_vectors.unlucky_seeds(ML_KEM_512.name)
        stream = hashes.xof(sampling.XOF_BYTES, _xof_seeds(seeds))
        candidates = np.asarray(sampling._candidates(stream))
        accepted = np.cumsum(candidates < ref.Q, axis=-1)
        consumed = np.argmax(accepted >= N, axis=-1) + 1
        # C2SP's generator reports 384 candidates as the worst it found.
        self.assertEqual(int(consumed.max()), 384)
        self.assertLess(int(consumed.max()), sampling.CANDIDATES)


class BudgetMissTest(parameterized.TestCase):
    """What happens below 256 acceptances — a `2^-261` event, pinned anyway.

    Not reachable from any seed, so it is driven through `_compact` directly.
    The point is not that the answer is right, because it cannot be; it is that
    the answer is the *same* everywhere — both forms, both backends, eager and
    traced. The two reach that edge by opposite means, one selecting on the index
    it read and the other dropping the write it did not want, so each is asserted
    rather than the pair being assumed to agree.
    """

    def _starved(self) -> fnp.ndarray:
        # Three acceptances in a stream of 560; every other value is above q.
        candidates = fnp.full((1, sampling.CANDIDATES), 4000, dtype=fnp.int32)
        return candidates.at[0, [0, 5, 9]].set(fnp.asarray([11, 22, 33]))

    @parameterized.named_parameters(("scatter", True), ("search", False))
    def test_the_accepted_prefix_is_still_correct(self, scatter: bool) -> None:
        with _compaction(scatter):
            got = np.asarray(sampling._compact(self._starved()))[0]
        np.testing.assert_array_equal(got[:3], np.array([11, 22, 33]))

    @parameterized.named_parameters(("scatter", True), ("search", False))
    def test_the_unfilled_tail_is_zero(self, scatter: bool) -> None:
        with _compaction(scatter):
            got = np.asarray(sampling._compact(self._starved()))[0]
        np.testing.assert_array_equal(got[3:], np.zeros(N - 3, dtype=got.dtype))
        # The failures this guards: an unclamped gather puts INT32_MIN here, and
        # a search that clamped without selecting puts the final candidate.
        self.assertGreaterEqual(int(got.min()), 0)

    @parameterized.named_parameters(("scatter", True), ("search", False))
    def test_the_miss_is_the_same_eager_and_traced(self, scatter: bool) -> None:
        starved = self._starved()
        with _compaction(scatter):
            np.testing.assert_array_equal(
                np.asarray(sampling._compact(starved)),
                np.asarray(frx.jit(lambda c: sampling._compact(c))(starved)),
            )


class CompactionFormTest(absltest.TestCase):
    """The two forms are one function, and nothing else in the suite sees both.

    `_compact` reads its form off the backend, so a CI leg only ever exercises
    the one its backend picked and every other gate here is blind to the other.
    Which form is faster is a measurement and belongs where it was taken; that
    they are the same function is a property, and this is where it is held.
    """

    def test_the_two_forms_agree_over_random_seeds(self) -> None:
        rng = np.random.default_rng(0)
        seeds = fnp.asarray(
            rng.integers(0, 256, (64, sampling.MATRIX_SEED_SIZE), dtype=np.uint8)
        )
        # A fresh lambda per form, for the trace-cache reason `_compaction` gives.
        forms = []
        for scatter in (True, False):
            with _compaction(scatter):
                traced = frx.jit(lambda s: sampling.sample_ntt(s))
                forms.append(np.asarray(as_ints(traced(seeds))))
        np.testing.assert_array_equal(forms[0], forms[1])


class SamplePolyCbdTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_key_generation_vectors_match(self, params: MlKemParams) -> None:
        # Algorithm 13: `s` takes nonces 0..k-1 and `e` takes k..2k-1, both at
        # eta1, both from sigma.
        k, eta1 = params.k, params.eta1
        vectors = cctv_vectors.intermediate(params.name)
        sigma = vectors.array_at("σ")
        nonces = np.arange(2 * k, dtype=np.uint8)
        got = np.asarray(
            as_ints(sampling.sample_poly_cbd(hashes.prf(eta1, sigma, nonces), eta1))
        )
        np.testing.assert_array_equal(
            got[:k], cctv_vectors.decode_polys(vectors.hex_at("s"), k)
        )
        np.testing.assert_array_equal(
            got[k:], cctv_vectors.decode_polys(vectors.hex_at("e"), k)
        )

    @parameterized.named_parameters(*_NAMED)
    def test_encryption_vectors_match_at_both_etas(self, params: MlKemParams) -> None:
        # Algorithm 14: `r` takes nonces 0..k-1 at eta1, then `e1` takes k..2k-1
        # and `e2` takes 2k, both at eta2. The two widths and the nonce that
        # continues across them are the part a per-vector transcription gets
        # wrong.
        k, eta1, eta2 = params.k, params.eta1, params.eta2
        vectors = cctv_vectors.intermediate(params.name)
        seed = vectors.array_at("r", 0)

        r = sampling.sample_poly_cbd(
            hashes.prf(eta1, seed, np.arange(k, dtype=np.uint8)), eta1
        )
        np.testing.assert_array_equal(
            np.asarray(as_ints(r)), cctv_vectors.decode_polys(vectors.hex_at("r", 1), k)
        )

        tail = sampling.sample_poly_cbd(
            hashes.prf(eta2, seed, np.arange(k, 2 * k + 1, dtype=np.uint8)), eta2
        )
        packed = np.asarray(as_ints(tail))
        np.testing.assert_array_equal(
            packed[:k], cctv_vectors.decode_polys(vectors.hex_at("e1"), k)
        )
        np.testing.assert_array_equal(
            packed[k:], cctv_vectors.decode_polys(vectors.hex_at("e2"), 1)
        )

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


class TracedShapeTest(parameterized.TestCase):
    """No data-dependent control flow, asserted on the lowering rather than hoped.

    The whole reason `sample_ntt` is a fixed squeeze instead of the standard's
    `while` is that a traced program cannot have a data-dependent trip count. A
    reviewer cannot see from the source which form the compaction lowered to, so
    the property is read back off the compiled program.
    """

    @parameterized.named_parameters(("scatter", True), ("search", False))
    def test_the_sampling_path_has_no_data_dependent_control_flow(
        self, scatter: bool
    ) -> None:
        text = _sample_ntt_lowering(scatter)
        for op in ("stablehlo.while", "stablehlo.case", "stablehlo.if"):
            self.assertNotIn(op, text)

    @parameterized.named_parameters(("scatter", True), ("search", False))
    def test_the_sampling_path_has_no_host_callback(self, scatter: bool) -> None:
        # `custom_call` is what a host callback lowers to; asserting on the op
        # rather than on the word "host" keeps this from passing or failing on
        # an unrelated attribute that happens to contain it.
        self.assertNotIn("custom_call", _sample_ntt_lowering(scatter))

    def test_the_scatter_form_lowers_to_a_scatter(self) -> None:
        # The design claim of `_scattered`, and the one a reader cannot check
        # from the source: each accepted candidate is written to the slot its
        # rank names, so nothing searches for a source index.
        text = _sample_ntt_lowering(scatter=True)
        self.assertIn("stablehlo.scatter", text)
        self.assertNotIn("stablehlo.gather", text)

    def test_the_search_form_lowers_to_a_gather(self) -> None:
        # The matching claim for `_searched`: `searchsorted` + `take_along_axis`
        # must be a gather, and the unrolled binary search is what keeps its
        # trip count static rather than the scanning default.
        text = _sample_ntt_lowering(scatter=False)
        self.assertIn("stablehlo.gather", text)
        self.assertNotIn("stablehlo.scatter", text)

    @parameterized.named_parameters(("scatter", True), ("search", False))
    def test_the_compaction_never_sorts(self, scatter: bool) -> None:
        # `sort` is the third compaction that would work and cost more than
        # either of these.
        self.assertNotIn("stablehlo.sort", _sample_ntt_lowering(scatter))

    def test_sample_poly_cbd_traces_without_a_branch(self) -> None:
        data = fnp.zeros((4, 64 * 2), dtype=fnp.uint8)
        text = frx.jit(lambda d: sampling.sample_poly_cbd(d, 2)).lower(data).as_text()
        for op in ("stablehlo.while", "stablehlo.case", "stablehlo.if"):
            self.assertNotIn(op, text)


if __name__ == "__main__":
    absltest.main()
