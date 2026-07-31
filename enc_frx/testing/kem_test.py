# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Kem seam holds its shape against an implementation.

Three obligations, and the last two are what a scheme gets wrong. The Protocol
must accept a conforming implementation; `decaps` must reject *implicitly*,
returning a specific other secret rather than a verdict; and rejection must be
decided per batch entry. A `decaps` that reduced its check over the batch passes
every all-valid and every all-invalid test ever written, so the cases here
corrupt some entries and pin the others alongside them.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from enc_frx.kem import Kem
from enc_frx.testing.toy_kem import ToyKem

_BATCH = 4
_DOMAIN = 7


def _batch() -> tuple[ToyKem, Array, Array, Array, Array]:
    """A scheme plus `(encaps_keys, decaps_keys, ciphertexts, shared_secrets)`."""
    scheme = ToyKem(domain=_DOMAIN)
    rng = np.random.default_rng(0)
    seeds = fnp.asarray(
        rng.integers(0, 256, (_BATCH, scheme.seed_size)), dtype=fnp.uint8
    )
    randomness = fnp.asarray(
        rng.integers(0, 256, (_BATCH, scheme.randomness_size)), dtype=fnp.uint8
    )
    # keygen is single-key; vmap is how a caller batches it, which is what the
    # seam's docstring says.
    encaps_keys, decaps_keys = frx.vmap(scheme.keygen)(seeds)
    ciphertexts, shared_secrets = scheme.encaps(encaps_keys, randomness=randomness)
    return scheme, encaps_keys, decaps_keys, ciphertexts, shared_secrets


def _corrupt(batch: Array, entries: tuple[int, ...]) -> Array:
    """Flip a bit in the named entries, leaving every other entry intact."""
    for entry in entries:
        batch = batch.at[entry, 0].set(batch[entry, 0] ^ 1)
    return batch


class KemSeamTest(absltest.TestCase):
    def test_protocol_accepts_a_conforming_implementation(self) -> None:
        # `Kem` is runtime_checkable, so this is the structural check a
        # consumer's constructor gets for free when it takes the seam.
        self.assertIsInstance(ToyKem(domain=_DOMAIN), Kem)

    def test_decaps_recovers_the_senders_secret(self) -> None:
        scheme, _, decaps_keys, ciphertexts, shared_secrets = _batch()
        recovered = scheme.decaps(decaps_keys, ciphertexts)
        self.assertEqual(recovered.shape, (_BATCH, scheme.shared_secret_size))
        np.testing.assert_array_equal(np.asarray(recovered), np.asarray(shared_secrets))

    def test_encaps_is_deterministic_in_its_randomness(self) -> None:
        # The seam takes randomness as an argument precisely so this holds; a
        # scheme that drew its own could not be gated on a published vector, and
        # a decapsulator's re-encryption check could never match.
        scheme, encaps_keys, _, ciphertexts, shared_secrets = _batch()
        rng = np.random.default_rng(0)
        rng.integers(0, 256, (_BATCH, scheme.seed_size))  # advance past the seeds
        randomness = fnp.asarray(
            rng.integers(0, 256, (_BATCH, scheme.randomness_size)), dtype=fnp.uint8
        )
        again_ct, again_ss = scheme.encaps(encaps_keys, randomness=randomness)
        np.testing.assert_array_equal(np.asarray(again_ct), np.asarray(ciphertexts))
        np.testing.assert_array_equal(np.asarray(again_ss), np.asarray(shared_secrets))

    def test_a_corrupted_ciphertext_yields_a_different_secret_not_a_failure(
        self,
    ) -> None:
        scheme, _, decaps_keys, ciphertexts, shared_secrets = _batch()
        rejected = scheme.decaps(decaps_keys, _corrupt(ciphertexts, (0, 1, 2, 3)))

        # A shared secret, in full, rather than a verdict or an exception.
        self.assertEqual(rejected.shape, shared_secrets.shape)
        self.assertEqual(rejected.dtype, fnp.uint8)
        self.assertFalse(bool(fnp.any(fnp.all(rejected == shared_secrets, axis=-1))))

    def test_rejection_is_deterministic(self) -> None:
        # Implicit rejection is only implicit if repeating the call gives the
        # same answer; a caller who saw it vary would have the verdict back.
        scheme, _, decaps_keys, ciphertexts, _ = _batch()
        corrupted = _corrupt(ciphertexts, (0, 1, 2, 3))
        np.testing.assert_array_equal(
            np.asarray(scheme.decaps(decaps_keys, corrupted)),
            np.asarray(scheme.decaps(decaps_keys, corrupted)),
        )

    def test_rejection_is_decided_per_batch_entry(self) -> None:
        # The case a batch-wide check passes: corrupt some entries, and require
        # the others to still decapsulate to the sender's secret.
        scheme, _, decaps_keys, ciphertexts, shared_secrets = _batch()
        recovered = scheme.decaps(decaps_keys, _corrupt(ciphertexts, (1, 3)))

        intact = np.asarray(fnp.all(recovered == shared_secrets, axis=-1))
        np.testing.assert_array_equal(intact, [True, False, True, False])

    def test_instances_compare_by_value(self) -> None:
        # A scheme rides pytree aux, where identity equality re-traces the
        # enclosing jit zone for every freshly built instance — slow, never an
        # error, so nothing else would catch it.
        self.assertEqual(ToyKem(domain=_DOMAIN), ToyKem(domain=_DOMAIN))
        self.assertEqual(hash(ToyKem(domain=_DOMAIN)), hash(ToyKem(domain=_DOMAIN)))
        self.assertNotEqual(ToyKem(domain=_DOMAIN), ToyKem(domain=_DOMAIN + 1))

    def test_decaps_traces_as_one_computation_over_the_batch(self) -> None:
        scheme, _, decaps_keys, ciphertexts, shared_secrets = _batch()
        jitted = frx.jit(scheme.decaps)
        np.testing.assert_array_equal(
            np.asarray(jitted(decaps_keys, ciphertexts)), np.asarray(shared_secrets)
        )


if __name__ == "__main__":
    absltest.main()
