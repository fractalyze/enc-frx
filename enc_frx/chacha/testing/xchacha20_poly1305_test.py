# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""XChaCha20-Poly1305 against draft-irtf-cfrg-xchacha's vectors.

Two gates that matter here beyond the AEAD's own, which
`chacha20_poly1305_test.py` already covers because the same implementation does
the work once the subkey is derived.

HChaCha20 is checked on its own vector rather than only through the AEAD: it is
the rounds without the feedforward add, and an accidental add produces a
different subkey and therefore a wrong tag for every input — a failure that looks
like a MAC bug from the outside.

And the derivation is checked to be the only difference: sealing through this
scheme must equal sealing through ChaCha20-Poly1305 with the derived subkey and
the zero-prefixed nonce, which is what "a key derivation and nothing else" means
operationally.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from enc_frx.aead import Aead
from enc_frx.chacha import chacha20
from enc_frx.chacha.chacha20_poly1305 import ChaCha20Poly1305
from enc_frx.chacha.xchacha20_poly1305 import XChaCha20Poly1305
from enc_frx.testing.kat import AeadVector, check_aead, to_bytes

# draft-irtf-cfrg-xchacha §2.2.1.
_HCHACHA_KEY = bytes(range(32))
_HCHACHA_NONCE = bytes.fromhex("000000090000004a0000000031415927")
_HCHACHA_SUBKEY = bytes.fromhex(
    "82413b4227b27bfed30e42508a877d73a0f9e4d58a74a853c12ec41326d3ecdc"
)

# draft-irtf-cfrg-xchacha §A.1.
_KEY = bytes.fromhex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
_NONCE = bytes.fromhex("404142434445464748494a4b4c4d4e4f5051525354555657")
_AAD = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
_PLAINTEXT = bytes.fromhex(
    "4c616469657320616e642047656e746c656d656e206f662074686520636c6173"
    "73206f66202739393a204966204920636f756c64206f6666657220796f75206f"
    "6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73"
    "637265656e20776f756c642062652069742e"
)
_POLY1305_KEY = bytes.fromhex(
    "7b191f80f361f099094f6f4b8fb97df847cc6873a8f2b190dd73807183f907d5"
)
_CIPHERTEXT = bytes.fromhex(
    "bd6d179d3e83d43b9576579493c0e939572a1700252bfaccbed2902c21396cbb"
    "731c7f1b0b4aa6440bf3a82f4eda7e39ae64c6708c54c216cb96b72e1213b452"
    "2f8c9ba40db5d945b11b69b982c1bb9e3f3fac2bc369488f76b2383565d3fff9"
    "21f9664c97637da9768812f615c68b13b52e"
)
_TAG = bytes.fromhex("c0875924c1c7987947deafd8780acf49")


def _batched(data: bytes, count: int = 1) -> Array:
    return fnp.asarray(
        np.tile(np.frombuffer(data, dtype=np.uint8), (count, 1)), dtype=fnp.uint8
    )


class HChaCha20Test(absltest.TestCase):
    def test_the_draft_vector(self) -> None:
        subkey = chacha20.hchacha20(_batched(_HCHACHA_KEY), _batched(_HCHACHA_NONCE))
        self.assertEqual(to_bytes(subkey[0]), _HCHACHA_SUBKEY)

    def test_the_feedforward_add_is_absent(self) -> None:
        # The one structural difference from `block_function`, and adding it by
        # accident would still yield 32 plausible-looking bytes. Built from the
        # same state, the two must disagree.
        key, nonce = _batched(_HCHACHA_KEY), _batched(_HCHACHA_NONCE)
        state = (
            [
                fnp.full((1,), value, dtype=fnp.uint32)
                for value in (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
            ]
            + chacha20._le_words(key, 8)
            + chacha20._le_words(nonce, 4)
        )

        without_add = chacha20._le_bytes(
            chacha20.rounds(state)[:4] + chacha20.rounds(state)[12:]
        )
        with_add = chacha20._le_bytes(
            chacha20.block_function(state)[:4] + chacha20.block_function(state)[12:]
        )
        self.assertEqual(to_bytes(without_add[0]), _HCHACHA_SUBKEY)
        self.assertNotEqual(to_bytes(with_add[0]), _HCHACHA_SUBKEY)


class XChaCha20Poly1305VectorTest(absltest.TestCase):
    def test_seal_matches_the_draft(self) -> None:
        sealed = XChaCha20Poly1305().seal(
            _batched(_KEY), _batched(_NONCE), _batched(_AAD), _batched(_PLAINTEXT)
        )
        self.assertEqual(to_bytes(sealed[0]), _CIPHERTEXT + _TAG)

    def test_open_recovers_the_plaintext(self) -> None:
        plaintext, ok = XChaCha20Poly1305().open(
            _batched(_KEY),
            _batched(_NONCE),
            _batched(_AAD),
            _batched(_CIPHERTEXT + _TAG),
        )
        self.assertTrue(bool(np.asarray(ok)[0]))
        self.assertEqual(to_bytes(plaintext[0]), _PLAINTEXT)

    def test_the_one_time_key_matches_the_draft(self) -> None:
        # The draft publishes the derived Poly1305 key, which pins the subkey and
        # the zero-prefixed nonce together rather than only their joint effect.
        subkey = chacha20.hchacha20(_batched(_KEY), _batched(_NONCE)[..., :16])
        inner_nonce = fnp.concatenate(
            [fnp.zeros((1, 4), dtype=fnp.uint8), _batched(_NONCE)[..., 16:]], axis=-1
        )
        block = chacha20.keystream(
            subkey, inner_nonce, fnp.asarray(np.array([0], dtype=np.uint32)), 1
        )
        self.assertEqual(to_bytes(block[0])[:32], _POLY1305_KEY)


class DerivationOnlyTest(absltest.TestCase):
    def test_it_equals_the_inner_aead_on_the_derived_key(self) -> None:
        rng = np.random.default_rng(0)
        key = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 24, dtype=np.uint8))
        aad = bytes(rng.integers(0, 256, 9, dtype=np.uint8))
        plaintext = bytes(rng.integers(0, 256, 55, dtype=np.uint8))

        outer = XChaCha20Poly1305().seal(
            _batched(key), _batched(nonce), _batched(aad), _batched(plaintext)
        )
        subkey = chacha20.hchacha20(_batched(key), _batched(nonce)[..., :16])
        inner_nonce = fnp.concatenate(
            [fnp.zeros((1, 4), dtype=fnp.uint8), _batched(nonce)[..., 16:]], axis=-1
        )
        inner = ChaCha20Poly1305().seal(
            subkey, inner_nonce, _batched(aad), _batched(plaintext)
        )
        self.assertEqual(to_bytes(outer[0]), to_bytes(inner[0]))

    def test_the_nonce_halves_are_both_load_bearing(self) -> None:
        # A derivation that dropped either half would still produce a tag.
        scheme = XChaCha20Poly1305()
        base = to_bytes(
            scheme.seal(
                _batched(_KEY), _batched(_NONCE), _batched(_AAD), _batched(_PLAINTEXT)
            )[0]
        )
        for index in (0, 20):
            moved = bytearray(_NONCE)
            moved[index] ^= 1
            other = to_bytes(
                scheme.seal(
                    _batched(_KEY),
                    _batched(bytes(moved)),
                    _batched(_AAD),
                    _batched(_PLAINTEXT),
                )[0]
            )
            self.assertNotEqual(base, other, f"nonce byte {index} changed nothing")


class SeamTest(absltest.TestCase):
    def test_it_satisfies_the_aead_protocol(self) -> None:
        self.assertIsInstance(XChaCha20Poly1305(), Aead)

    def test_instances_compare_by_value(self) -> None:
        self.assertEqual(XChaCha20Poly1305(), XChaCha20Poly1305())
        self.assertEqual(hash(XChaCha20Poly1305()), hash(XChaCha20Poly1305()))

    def test_the_harness_negative_cases_pass(self) -> None:
        check_aead(XChaCha20Poly1305(), _kat_vectors())


def _kat_vectors() -> list[AeadVector]:
    scheme = XChaCha20Poly1305()
    rng = np.random.default_rng(11)
    vectors = []
    for case in range(3):
        key = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 24, dtype=np.uint8))
        aad = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        plaintext = bytes(rng.integers(1, 256, 48, dtype=np.uint8))
        sealed = scheme.seal(
            _batched(key), _batched(nonce), _batched(aad), _batched(plaintext)
        )
        vectors.append(
            AeadVector(
                case_id=f"xchacha20-poly1305/{case}",
                parameter_set="xchacha20-poly1305",
                key=key,
                nonce=nonce,
                associated_data=aad,
                plaintext=plaintext,
                ciphertext=to_bytes(sealed[0]),
            )
        )
    return vectors


if __name__ == "__main__":
    absltest.main()
