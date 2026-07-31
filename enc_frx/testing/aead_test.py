# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Aead seam holds its shape against an implementation.

The obligation that needs the sharpest test is masking. `open` returning
`ok = False` is easy; returning zeros *instead of the decryption it just
computed* is the part a scheme skips, and the only way to test it is to arrange a
failure where the unauthenticated plaintext is known to be non-zero — which is
what flipping a bit in the tag alone does, since the ciphertext body is then
byte-identical to a valid one.

The second is that both the verdict and the mask are per entry. A scheme that
reduced its tag comparison over the batch, or masked from `any(ok)`, passes every
all-valid and every all-invalid test ever written.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from enc_frx.aead import Aead
from enc_frx.testing.toy_aead import ToyAead

_BATCH = 4
_MESSAGE_LEN = 16
_AAD_LEN = 5
_DOMAIN = 7


def _batch() -> tuple[ToyAead, Array, Array, Array, Array, Array]:
    """A scheme plus `(keys, nonces, aad, plaintexts, ciphertexts)`."""
    scheme = ToyAead(domain=_DOMAIN)
    rng = np.random.default_rng(0)

    def draw(*shape: int) -> Array:
        return fnp.asarray(rng.integers(1, 256, shape), dtype=fnp.uint8)

    keys = draw(_BATCH, scheme.key_size)
    nonces = draw(_BATCH, scheme.nonce_size)
    aad = draw(_BATCH, _AAD_LEN)
    # Drawn from 1 so no plaintext byte is zero: the masking test needs the
    # unauthenticated decryption to be distinguishable from a zeroed one.
    plaintexts = draw(_BATCH, _MESSAGE_LEN)
    return (
        scheme,
        keys,
        nonces,
        aad,
        plaintexts,
        scheme.seal(keys, nonces, aad, plaintexts),
    )


def _corrupt(batch: Array, entries: tuple[int, ...], index: int) -> Array:
    """Flip a bit at `index` in the named entries, leaving the others intact."""
    for entry in entries:
        batch = batch.at[entry, index].set(batch[entry, index] ^ 1)
    return batch


class AeadSeamTest(absltest.TestCase):
    def test_protocol_accepts_a_conforming_implementation(self) -> None:
        self.assertIsInstance(ToyAead(domain=_DOMAIN), Aead)

    def test_seal_appends_the_tag_and_open_round_trips(self) -> None:
        scheme, keys, nonces, aad, plaintexts, ciphertexts = _batch()
        self.assertEqual(ciphertexts.shape, (_BATCH, _MESSAGE_LEN + scheme.tag_size))
        opened, ok = scheme.open(keys, nonces, aad, ciphertexts)
        self.assertEqual(opened.shape, (_BATCH, _MESSAGE_LEN))
        self.assertEqual(ok.shape, (_BATCH,))
        self.assertEqual(ok.dtype, fnp.bool_)
        self.assertTrue(bool(fnp.all(ok)))
        np.testing.assert_array_equal(np.asarray(opened), np.asarray(plaintexts))

    def test_a_failing_entry_comes_back_masked_not_raw(self) -> None:
        # Only the tag is corrupted, so the ciphertext body — and therefore the
        # unauthenticated decryption — is byte-identical to the valid case. The
        # zeros can only come from the mask.
        scheme, keys, nonces, aad, plaintexts, ciphertexts = _batch()
        self.assertTrue(bool(fnp.all(plaintexts != 0)))

        tag_corrupted = _corrupt(ciphertexts, tuple(range(_BATCH)), -1)
        opened, ok = scheme.open(keys, nonces, aad, tag_corrupted)

        self.assertFalse(bool(fnp.any(ok)))
        np.testing.assert_array_equal(
            np.asarray(opened), np.zeros_like(np.asarray(plaintexts))
        )

    def test_every_corruption_is_rejected(self) -> None:
        scheme, keys, nonces, aad, _, ciphertexts = _batch()
        entries = tuple(range(_BATCH))
        cases = {
            "tag": (keys, nonces, aad, _corrupt(ciphertexts, entries, -1)),
            "ciphertext": (keys, nonces, aad, _corrupt(ciphertexts, entries, 0)),
            "associated_data": (keys, nonces, _corrupt(aad, entries, 0), ciphertexts),
            "nonce": (keys, _corrupt(nonces, entries, 0), aad, ciphertexts),
        }
        for name, args in cases.items():
            with self.subTest(name):
                _, ok = scheme.open(*args)
                self.assertFalse(bool(fnp.any(ok)))

    def test_the_verdict_and_the_mask_are_per_batch_entry(self) -> None:
        # The case a batch-wide check or a batch-wide mask passes.
        scheme, keys, nonces, aad, plaintexts, ciphertexts = _batch()
        opened, ok = scheme.open(keys, nonces, aad, _corrupt(ciphertexts, (1, 3), -1))

        np.testing.assert_array_equal(np.asarray(ok), [True, False, True, False])
        recovered = np.asarray(fnp.all(opened == plaintexts, axis=-1))
        np.testing.assert_array_equal(recovered, [True, False, True, False])
        masked = np.asarray(fnp.all(opened == 0, axis=-1))
        np.testing.assert_array_equal(masked, [False, True, False, True])

    def test_associated_data_is_optional(self) -> None:
        scheme, keys, nonces, _, plaintexts, _ = _batch()
        ciphertexts = scheme.seal(keys, nonces, None, plaintexts)
        opened, ok = scheme.open(keys, nonces, None, ciphertexts)
        self.assertTrue(bool(fnp.all(ok)))
        np.testing.assert_array_equal(np.asarray(opened), np.asarray(plaintexts))

    def test_instances_compare_by_value(self) -> None:
        self.assertEqual(ToyAead(domain=_DOMAIN), ToyAead(domain=_DOMAIN))
        self.assertEqual(hash(ToyAead(domain=_DOMAIN)), hash(ToyAead(domain=_DOMAIN)))
        self.assertNotEqual(ToyAead(domain=_DOMAIN), ToyAead(domain=_DOMAIN + 1))

    def test_open_traces_as_one_computation_over_the_batch(self) -> None:
        scheme, keys, nonces, aad, plaintexts, ciphertexts = _batch()
        opened, ok = frx.jit(scheme.open)(keys, nonces, aad, ciphertexts)
        self.assertTrue(bool(fnp.all(ok)))
        np.testing.assert_array_equal(np.asarray(opened), np.asarray(plaintexts))


if __name__ == "__main__":
    absltest.main()
