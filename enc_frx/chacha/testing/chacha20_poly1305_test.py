# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ChaCha20-Poly1305 against RFC 8439, and against the two seam rules.

Three layers. The published vectors gate the construction; the differential
reference gates the MAC-input assembly across lengths the vectors do not cover
(a padding or length-encoding mistake produces a wrong tag for every input, so
one vector cannot distinguish "right" from "consistently wrong"); and the KAT
harness supplies the negative cases and the mixed-validity batch, which is where
masking and per-entry verdicts are actually decided.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array

from enc_frx.aead import Aead
from enc_frx.chacha.chacha20_poly1305 import ChaCha20Poly1305
from enc_frx.chacha.testing import rfc8439_reference as reference
from enc_frx.testing.kat import AeadVector, check_aead, to_bytes

# RFC 8439 §2.8.2.
_KEY_2_8_2 = bytes(range(0x80, 0xA0))
_NONCE_2_8_2 = bytes.fromhex("070000004041424344454647")
_AAD_2_8_2 = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
_PLAINTEXT_2_8_2 = (
    b"Ladies and Gentlemen of the class of '99: If I could offer you only one "
    b"tip for the future, sunscreen would be it."
)
_TAG_2_8_2 = bytes.fromhex("1ae10b594f09e26a7e902ecbd0600691")
_CIPHERTEXT_2_8_2 = bytes.fromhex(
    "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
    "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
    "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
    "3ff4def08e4b7a9de576d26586cec64b6116"
)

# RFC 8439 §A.5, a decryption case: 265 bytes of ciphertext whose plaintext
# carries non-ASCII punctuation, so both are kept as hex rather than as a
# literal.
_KEY_A_5 = bytes.fromhex(
    "1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0"
)
_NONCE_A_5 = bytes.fromhex("000000000102030405060708")
_AAD_A_5 = bytes.fromhex("f33388860000000000004e91")
_TAG_A_5 = bytes.fromhex("eead9d67890cbb22392336fea1851f38")


def _batched(data: bytes, count: int = 1) -> Array:
    return fnp.asarray(
        np.tile(np.frombuffer(data, dtype=np.uint8), (count, 1)), dtype=fnp.uint8
    )


def _seal(key: bytes, nonce: bytes, aad: bytes | None, plaintext: bytes) -> bytes:
    sealed = ChaCha20Poly1305().seal(
        _batched(key),
        _batched(nonce),
        None if aad is None else _batched(aad),
        _batched(plaintext),
    )
    return to_bytes(sealed[0])


def _open(
    key: bytes, nonce: bytes, aad: bytes | None, ciphertext: bytes
) -> tuple[bytes, bool]:
    plaintext, ok = ChaCha20Poly1305().open(
        _batched(key),
        _batched(nonce),
        None if aad is None else _batched(aad),
        _batched(ciphertext),
    )
    return to_bytes(plaintext[0]), bool(np.asarray(ok)[0])


class VectorTest(absltest.TestCase):
    def test_the_reference_reproduces_the_rfc_tag(self) -> None:
        # The oracle earns its place first: an unverified reference proves
        # nothing about the implementation it is used to check.
        sealed = reference.aead_seal(
            _KEY_2_8_2, _NONCE_2_8_2, _AAD_2_8_2, _PLAINTEXT_2_8_2
        )
        self.assertEqual(sealed[-16:], _TAG_2_8_2)

    def test_seal_matches_rfc_8439_section_2_8_2(self) -> None:
        sealed = _seal(_KEY_2_8_2, _NONCE_2_8_2, _AAD_2_8_2, _PLAINTEXT_2_8_2)
        self.assertEqual(sealed, _CIPHERTEXT_2_8_2 + _TAG_2_8_2)
        self.assertEqual(
            sealed,
            reference.aead_seal(_KEY_2_8_2, _NONCE_2_8_2, _AAD_2_8_2, _PLAINTEXT_2_8_2),
        )

    def test_open_round_trips_section_2_8_2(self) -> None:
        sealed = _seal(_KEY_2_8_2, _NONCE_2_8_2, _AAD_2_8_2, _PLAINTEXT_2_8_2)
        plaintext, ok = _open(_KEY_2_8_2, _NONCE_2_8_2, _AAD_2_8_2, sealed)
        self.assertTrue(ok)
        self.assertEqual(plaintext, _PLAINTEXT_2_8_2)

    def test_open_matches_rfc_8439_section_a_5(self) -> None:
        published = _a5_ciphertext()
        plaintext, ok = _open(_KEY_A_5, _NONCE_A_5, _AAD_A_5, published)
        self.assertTrue(ok)
        self.assertEqual(plaintext, _a5_plaintext())
        self.assertEqual(_seal(_KEY_A_5, _NONCE_A_5, _AAD_A_5, plaintext), published)


class DifferentialTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("empty_plaintext", 0, 12),
        ("one_byte", 1, 12),
        ("under_a_block", 15, 12),
        ("exact_mac_alignment", 16, 16),
        ("aad_needs_padding", 64, 5),
        ("both_ragged", 70, 13),
        ("no_aad", 48, 0),
        ("long", 300, 24),
    )
    def test_lengths_match_the_reference(
        self, plaintext_length: int, aad_length: int
    ) -> None:
        # The MAC input pads the associated data and the ciphertext separately
        # and then appends both lengths. A vector at one length cannot tell a
        # correct assembly from a consistently wrong one; a sweep across the
        # padding boundaries can.
        rng = np.random.default_rng(plaintext_length * 31 + aad_length)
        key = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        aad = bytes(rng.integers(0, 256, aad_length, dtype=np.uint8))
        plaintext = bytes(rng.integers(0, 256, plaintext_length, dtype=np.uint8))
        self.assertEqual(
            _seal(key, nonce, aad, plaintext),
            reference.aead_seal(key, nonce, aad, plaintext),
        )

    def test_absent_aad_equals_empty_aad(self) -> None:
        # `None` is what a caller with nothing to authenticate passes; the
        # standard's length field is zero either way, so the tags must agree.
        rng = np.random.default_rng(0)
        key = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        plaintext = bytes(rng.integers(0, 256, 40, dtype=np.uint8))
        self.assertEqual(
            _seal(key, nonce, None, plaintext),
            reference.aead_seal(key, nonce, b"", plaintext),
        )


class SeamTest(absltest.TestCase):
    def test_it_satisfies_the_aead_protocol(self) -> None:
        self.assertIsInstance(ChaCha20Poly1305(), Aead)

    def test_instances_compare_by_value(self) -> None:
        self.assertEqual(ChaCha20Poly1305(), ChaCha20Poly1305())
        self.assertEqual(hash(ChaCha20Poly1305()), hash(ChaCha20Poly1305()))

    def test_the_harness_negative_cases_pass(self) -> None:
        # The mandatory tampering — tag, ciphertext, nonce, associated data —
        # plus the mixed-validity batch, all derived by the harness rather than
        # written here.
        check_aead(ChaCha20Poly1305(), _kat_vectors())

    def test_a_failing_entry_is_masked_not_released(self) -> None:
        # Only the tag is corrupted, so the ciphertext body and therefore the
        # unauthenticated decryption are byte-identical to the valid case: the
        # zeros can only have come from the mask.
        sealed = _seal(_KEY_2_8_2, _NONCE_2_8_2, _AAD_2_8_2, _PLAINTEXT_2_8_2)
        tampered = sealed[:-1] + bytes([sealed[-1] ^ 1])
        plaintext, ok = _open(_KEY_2_8_2, _NONCE_2_8_2, _AAD_2_8_2, tampered)
        self.assertFalse(ok)
        self.assertEqual(plaintext, bytes(len(_PLAINTEXT_2_8_2)))

    def test_the_verdict_is_per_batch_entry(self) -> None:
        scheme = ChaCha20Poly1305()
        keys, nonces = _batched(_KEY_2_8_2, 4), _batched(_NONCE_2_8_2, 4)
        aad = _batched(_AAD_2_8_2, 4)
        sealed = scheme.seal(keys, nonces, aad, _batched(_PLAINTEXT_2_8_2, 4))
        tampered = sealed
        for entry in (1, 3):
            tampered = tampered.at[entry, -1].set(tampered[entry, -1] ^ 1)
        plaintext, ok = scheme.open(keys, nonces, aad, tampered)
        np.testing.assert_array_equal(np.asarray(ok), [True, False, True, False])
        masked = np.asarray(fnp.all(plaintext == 0, axis=-1))
        np.testing.assert_array_equal(masked, [False, True, False, True])

    def test_it_traces_as_one_computation(self) -> None:
        scheme = ChaCha20Poly1305()
        sealed = frx.jit(scheme.seal)(
            _batched(_KEY_2_8_2),
            _batched(_NONCE_2_8_2),
            _batched(_AAD_2_8_2),
            _batched(_PLAINTEXT_2_8_2),
        )
        self.assertEqual(to_bytes(sealed[0])[-16:], _TAG_2_8_2)


def _kat_vectors() -> list[AeadVector]:
    rng = np.random.default_rng(7)
    vectors = []
    for case in range(3):
        key = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        aad = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        plaintext = bytes(rng.integers(1, 256, 48, dtype=np.uint8))
        vectors.append(
            AeadVector(
                case_id=f"chacha20-poly1305/{case}",
                parameter_set="chacha20-poly1305",
                key=key,
                nonce=nonce,
                associated_data=aad,
                plaintext=plaintext,
                ciphertext=reference.aead_seal(key, nonce, aad, plaintext),
            )
        )
    return vectors


def _a5_ciphertext() -> bytes:
    """The published ciphertext with the published tag appended."""
    return (
        bytes.fromhex(
            "64a0861575861af460f062c79be643bd5e805cfd345cf389f108670ac76c8cb2"
            "4c6cfc18755d43eea09ee94e382d26b0bdb7b73c321b0100d4f03b7f355894cf"
            "332f830e710b97ce98c8a84abd0b948114ad176e008d33bd60f982b1ff37c855"
            "9797a06ef4f0ef61c186324e2b3506383606907b6a7c02b0f9f6157b53c867e4"
            "b9166c767b804d46a59b5216cde7a4e99040c5a40433225ee282a1b0a06c523e"
            "af4534d7f83fa1155b0047718cbc546a0d072b04b3564eea1b422273f548271a"
            "0bb2316053fa76991955ebd63159434ecebb4e466dae5a1073a6727627097a10"
            "49e617d91d361094fa68f0ff77987130305beaba2eda04df997b714d6c6f2c29"
            "a6ad5cb4022b02709b"
        )
        + _TAG_A_5
    )


def _a5_plaintext() -> bytes:
    return bytes.fromhex(
        "496e7465726e65742d4472616674732061726520647261667420646f63756d65"
        "6e74732076616c696420666f722061206d6178696d756d206f6620736978206d"
        "6f6e74687320616e64206d617920626520757064617465642c207265706c6163"
        "65642c206f72206f62736f6c65746564206279206f7468657220646f63756d65"
        "6e747320617420616e792074696d652e20497420697320696e617070726f7072"
        "6961746520746f2075736520496e7465726e65742d4472616674732061732072"
        "65666572656e6365206d6174657269616c206f7220746f206369746520746865"
        "6d206f74686572207468616e206173202fe2809c776f726b20696e2070726f67"
        "726573732e2fe2809d"
    )


if __name__ == "__main__":
    absltest.main()
