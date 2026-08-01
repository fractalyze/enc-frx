# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES-GCM against CAVP's vectors, and against the rules a vector cannot see.

The gate that runs per PR. It drives `check_aead` — every published check plus
the tampering the standard does not publish — over a batch drawn from each corner
of the published parameter space: all three key lengths, both `J_0` constructions,
and every tag length the scheme admits. The exhaustive pass over all 47250 cases
is `gcm_sweep_test`, which is tagged `slow_kat` and runs on the schedule.

Splitting them is a cost decision with one real consequence, so it is stated here
rather than left implicit. `check_aead` derives a tampered batch per entry, so it
is quadratic in a group and cannot run over the whole set; the sweep is linear and
cannot see a mixed verdict inside one batch. Neither alone is the gate this repo
says it wants, and the batches below are what makes the quadratic half affordable
every time.

**The mixed-validity batch comes from the standard.** CAVP's decrypt sections are
roughly half `FAIL`, so a batch drawn from one already mixes verdicts and the
masking requirement is checked against published expectations rather than against
corruption invented here.

The structural tests cover what no vector can. A vector cannot see that the
payload's keystream starts one counter block after the tag's mask — get that
wrong and every vector fails, but the failure says "wrong ciphertext" rather than
naming the counter. And a vector cannot see that the tag length is fixed at
construction, because the downgrade it prevents is a call the tests would have to
make on purpose.
"""

from __future__ import annotations

import inspect

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from enc_frx.aead import Aead
from enc_frx.aes import block, ctr
from enc_frx.aes.gcm import TAG_SIZES, AesGcm
from enc_frx.aes.testing import gcm_vectors
from enc_frx.testing.kat import check_aead, to_bytes

# Four entries is the whole cost knob: `check_aead` runs four tamperings against
# every entry of a batch, so the gate's runtime is roughly quadratic in this.
_BATCH = 4

_DEFAULT_NONCE_SIZE = 12
_FULL_TAG_SIZE = 16


def _array(data: bytes) -> Array:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))[None]


class CavpGateTest(parameterized.TestCase):
    @parameterized.parameters(*sorted(gcm_vectors.ENCRYPT_FILES))
    def test_each_key_length_reproduces_the_published_ciphertexts(
        self, key_size: int
    ) -> None:
        vector_set = gcm_vectors.instance(
            gcm_vectors.ENCRYPT_FILES[key_size],
            key_size,
            _DEFAULT_NONCE_SIZE,
            _FULL_TAG_SIZE,
        )
        batch = gcm_vectors.accepted_batch(vector_set, _BATCH)
        self.assertLen(batch, _BATCH)
        check_aead(AesGcm(key_size, _DEFAULT_NONCE_SIZE, _FULL_TAG_SIZE), batch)

    @parameterized.parameters(*sorted(gcm_vectors.DECRYPT_FILES))
    def test_each_key_length_agrees_with_the_published_verdicts(
        self, key_size: int
    ) -> None:
        vector_set = gcm_vectors.instance(
            gcm_vectors.DECRYPT_FILES[key_size],
            key_size,
            _DEFAULT_NONCE_SIZE,
            _FULL_TAG_SIZE,
        )
        batch = gcm_vectors.mixed_batch(vector_set, _BATCH)
        # The mixed-validity requirement, met by the published set rather than by
        # corruption invented here. `check_aead` is what asserts that exactly the
        # failing entries come back masked.
        self.assertEqual({v.valid for v in batch}, {True, False})
        check_aead(AesGcm(key_size, _DEFAULT_NONCE_SIZE, _FULL_TAG_SIZE), batch)

    @parameterized.parameters(*gcm_vectors.NONCE_SIZES)
    def test_both_initial_counter_constructions(self, nonce_size: int) -> None:
        # 12 bytes is used directly as `J_0`'s prefix; 1 and 128 are hashed
        # through GHASH. The branch has its own unit test in `ctr_test`; what
        # this adds is that GCM reaches the right one, against vectors.
        vector_set = gcm_vectors.instance(
            gcm_vectors.DECRYPT_FILES[16], 16, nonce_size, _FULL_TAG_SIZE
        )
        batch = gcm_vectors.mixed_batch(vector_set, _BATCH)
        check_aead(AesGcm(16, nonce_size, _FULL_TAG_SIZE), batch)

    @parameterized.parameters(*(size for size in TAG_SIZES if size != _FULL_TAG_SIZE))
    def test_each_truncated_tag_length(self, tag_size: int) -> None:
        # A truncated tag is a different verifier, not a different call, so each
        # length is gated as its own scheme instance. The full-length case is
        # covered by the key-length tests above.
        vector_set = gcm_vectors.instance(
            gcm_vectors.DECRYPT_FILES[16], 16, _DEFAULT_NONCE_SIZE, tag_size
        )
        batch = gcm_vectors.mixed_batch(vector_set, _BATCH)
        check_aead(AesGcm(16, _DEFAULT_NONCE_SIZE, tag_size), batch)


class PublishedSetTest(absltest.TestCase):
    """What the files carry, so a silent change to them is a failure here."""

    def test_the_set_covers_the_parameters_the_scheme_claims(self) -> None:
        for name in gcm_vectors.FILES:
            published = {(s.nonce_size, s.tag_size) for s in gcm_vectors.sets(name)}
            self.assertEqual(
                published,
                {
                    (nonce_size, tag_size)
                    for nonce_size in gcm_vectors.NONCE_SIZES
                    for tag_size in gcm_vectors.PUBLISHED_TAG_SIZES
                },
                name,
            )

    def test_the_decrypt_files_publish_failures(self) -> None:
        # The negative half of the gate. If a regeneration ever dropped them,
        # every positive test would still pass and the suite would be claiming a
        # rejection gate it no longer has.
        for key_size, name in sorted(gcm_vectors.DECRYPT_FILES.items()):
            rejected = sum(
                1 for s in gcm_vectors.sets(name) for v in s.vectors if not v.valid
            )
            self.assertGreater(rejected, 1000, f"AES-{key_size * 8}")

    def test_the_encrypt_files_publish_none(self) -> None:
        for name in gcm_vectors.ENCRYPT_FILES.values():
            self.assertTrue(
                all(v.valid for s in gcm_vectors.sets(name) for v in s.vectors), name
            )

    def test_no_case_carries_a_field_the_record_cannot_express(self) -> None:
        for name in gcm_vectors.FILES:
            self.assertEmpty(
                {
                    field
                    for s in gcm_vectors.sets(name)
                    for v in s.vectors
                    for field in v.unsupported
                },
                name,
            )

    def test_every_published_length_is_a_whole_number_of_bytes(self) -> None:
        # The loader refuses a ragged section, so reaching this point at all is
        # the assertion; it is stated as a test because it is the reason this set
        # is usable and ACVP's is not.
        for name in gcm_vectors.FILES:
            self.assertNotEmpty(gcm_vectors.sets(name), name)


class TagLengthTest(absltest.TestCase):
    def test_the_shorter_published_tags_are_refused(self) -> None:
        # CAVP publishes 32- and 64-bit tags, which are SP 800-38D Appendix C's
        # rather than §5.2.1.2's. Refused rather than skipped: a suite that
        # quietly ran 5 of the 7 published tag lengths and reported green would
        # be claiming coverage it does not have.
        appendix_c = [
            size for size in gcm_vectors.PUBLISHED_TAG_SIZES if size not in TAG_SIZES
        ]
        self.assertEqual(appendix_c, [4, 8])
        for tag_size in appendix_c:
            with self.assertRaisesRegex(ValueError, "5.2.1.2"):
                AesGcm(16, _DEFAULT_NONCE_SIZE, tag_size)

    def test_the_length_is_not_a_call_argument(self) -> None:
        # The downgrade this prevents: a verifier that took its tag length from
        # the caller could be handed a shorter one than it was built for.
        self.assertEqual(
            list(inspect.signature(AesGcm.seal).parameters),
            ["self", "key", "nonce", "associated_data", "plaintext"],
        )
        self.assertEqual(
            list(inspect.signature(AesGcm.open).parameters),
            ["self", "key", "nonce", "associated_data", "ciphertext"],
        )

    def test_a_shorter_tag_is_a_shorter_ciphertext(self) -> None:
        key, nonce, plaintext = _array(bytes(16)), _array(bytes(12)), _array(bytes(32))
        full = AesGcm(16, 12, 16).seal(key, nonce, None, plaintext)
        truncated = AesGcm(16, 12, 12).seal(key, nonce, None, plaintext)
        self.assertEqual(full.shape[-1], 32 + 16)
        self.assertEqual(truncated.shape[-1], 32 + 12)
        # The truncation is a prefix of the full tag, per `MSB_t`.
        self.assertEqual(to_bytes(truncated[0])[32:], to_bytes(full[0])[32:-4])

    def test_a_ciphertext_shorter_than_the_tag_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit"):
            AesGcm(16, 12, 16).open(
                _array(bytes(16)), _array(bytes(12)), None, _array(bytes(8))
            )


class ConstructionTest(absltest.TestCase):
    def test_the_payload_starts_one_counter_block_after_the_tag_mask(self) -> None:
        """SP 800-38D §7.1: the keystream begins at `inc32(J_0)`, the mask at `J_0`.

        Checked against `block.encrypt_block` on hand-built counters rather than
        against this module's own arithmetic. Getting it wrong fails every vector,
        but as "wrong ciphertext" — nothing in that says which counter moved.
        """
        rng = np.random.default_rng(0)
        key = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 12, dtype=np.uint8))

        # A 96-bit IV makes `J_0` the IV with a counter of one, so the expected
        # counters are writable by hand.
        mask = to_bytes(
            block.encrypt_block(_array(key), _array(nonce + (1).to_bytes(4, "big")))[0]
        )
        keystream = to_bytes(
            block.encrypt_block(_array(key), _array(nonce + (2).to_bytes(4, "big")))[0]
        )

        # A zero payload makes the ciphertext the keystream itself.
        sealed = to_bytes(
            AesGcm(16, 12, 16).seal(
                _array(key), _array(nonce), None, _array(bytes(16))
            )[0]
        )
        self.assertEqual(sealed[:16], keystream)
        self.assertNotEqual(sealed[:16], mask)

    def test_an_empty_message_tags_to_the_mask_alone(self) -> None:
        """With nothing to hash, the tag reduces to `E_K(J_0)` exactly.

        No payload and no associated data leave GHASH one all-zero length block,
        and `(0 ⊕ 0) · H` is zero — so §7.1 step 6 is the mask by itself. That
        identity pins the wiring the vectors cannot name individually: that the
        subkey handed to `J_0`'s derivation is the cipher at zero, and that the
        tag is masked at `J_0` rather than at the counter the payload starts from.

        An 8-byte IV rather than 12, so the GHASH-derived branch of
        `ctr.initial_counter` is the one under test. That function is gated by
        `ctr_test`; what this adds is that GCM reaches it with the right subkey.
        """
        rng = np.random.default_rng(1)
        key = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 8, dtype=np.uint8))
        subkey = block.encrypt_block(_array(key), _array(bytes(16)))
        counter = ctr.initial_counter(subkey, _array(nonce))
        mask = to_bytes(block.encrypt_block(_array(key), counter)[0])

        sealed = to_bytes(
            AesGcm(16, 8, 16).seal(_array(key), _array(nonce), None, _array(b""))[0]
        )
        self.assertLen(sealed, 16)
        self.assertEqual(sealed, mask)

    def test_argument_widths_are_pinned_to_the_instance(self) -> None:
        # The hazard the sizes being parameters introduces: a 16-byte key handed
        # to an `AesGcm(32)` would otherwise run as AES-128 under a valid-looking
        # tag.
        scheme = AesGcm(32, 12, 16)
        with self.assertRaisesRegex(ValueError, "key is 32 bytes"):
            scheme.seal(_array(bytes(16)), _array(bytes(12)), None, _array(bytes(8)))
        with self.assertRaisesRegex(ValueError, "nonce is 12 bytes"):
            scheme.seal(_array(bytes(32)), _array(bytes(8)), None, _array(bytes(8)))

    def test_instances_compare_by_value(self) -> None:
        # A scheme instance rides pytree aux, where identity equality does not
        # error — it silently re-traces the enclosing jit zone for every freshly
        # built instance, which surfaces as a slow call and never as a failure.
        self.assertEqual(AesGcm(16, 12, 16), AesGcm(16, 12, 16))
        self.assertEqual(hash(AesGcm(16, 12, 16)), hash(AesGcm(16, 12, 16)))
        for other in (AesGcm(32, 12, 16), AesGcm(16, 8, 16), AesGcm(16, 12, 12)):
            self.assertNotEqual(AesGcm(16, 12, 16), other)

    def test_it_satisfies_the_seam(self) -> None:
        self.assertIsInstance(AesGcm(), Aead)

    def test_it_traces_as_one_computation(self) -> None:
        vector_set = gcm_vectors.instance(gcm_vectors.ENCRYPT_FILES[16], 16, 12, 16)
        vector = gcm_vectors.accepted_batch(vector_set, 1)[0]
        scheme = AesGcm(16, 12, 16)
        sealed = frx.jit(scheme.seal)(
            _array(vector.key),
            _array(vector.nonce),
            _array(vector.associated_data or b""),
            _array(vector.plaintext),
        )
        self.assertEqual(to_bytes(sealed[0]), vector.ciphertext)

    def test_a_batch_of_one_is_not_a_special_case(self) -> None:
        vector_set = gcm_vectors.instance(gcm_vectors.ENCRYPT_FILES[16], 16, 12, 16)
        batch = gcm_vectors.accepted_batch(vector_set, 3)
        scheme = AesGcm(16, 12, 16)
        together = scheme.seal(
            fnp.asarray(np.stack([np.frombuffer(v.key, np.uint8) for v in batch])),
            fnp.asarray(np.stack([np.frombuffer(v.nonce, np.uint8) for v in batch])),
            fnp.asarray(
                np.stack(
                    [np.frombuffer(v.associated_data or b"", np.uint8) for v in batch]
                )
            ),
            fnp.asarray(
                np.stack([np.frombuffer(v.plaintext, np.uint8) for v in batch])
            ),
        )
        for index, vector in enumerate(batch):
            alone = scheme.seal(
                _array(vector.key),
                _array(vector.nonce),
                _array(vector.associated_data or b""),
                _array(vector.plaintext),
            )
            self.assertEqual(to_bytes(together[index]), to_bytes(alone[0]))


if __name__ == "__main__":
    absltest.main()
