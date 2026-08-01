# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ChaCha20 against RFC 8439's published vectors, and against its own shape.

The vectors gate the arithmetic. The two structural tests gate what the vectors
cannot see: that a batch entry is independent of its neighbours, and that block
`i` really is computed from `counter + i` rather than from a loop that happened
to produce the same bytes.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array

from enc_frx.chacha import chacha20

# RFC 8439 §2.3.2.
_KEY_2_3_2 = bytes(range(32))
_NONCE_2_3_2 = bytes.fromhex("000000090000004a00000000")
_COUNTER_2_3_2 = 1
_KEYSTREAM_2_3_2 = bytes.fromhex(
    "10f1e7e4d13b5915500fdd1fa32071c4"
    "c7d1f4c733c068030422aa9ac3d46c4e"
    "d2826446079faa0914c2d705d98b02a2"
    "b5129cd1de164eb9cbd083e8a2503c4e"
)

# RFC 8439 §2.4.2.
_KEY_2_4_2 = bytes(range(32))
_NONCE_2_4_2 = bytes.fromhex("000000000000004a00000000")
_COUNTER_2_4_2 = 1
_PLAINTEXT_2_4_2 = (
    b"Ladies and Gentlemen of the class of '99: If I could offer you only one "
    b"tip for the future, sunscreen would be it."
)
_CIPHERTEXT_2_4_2 = bytes.fromhex(
    "6e2e359a2568f98041ba0728dd0d6981"
    "e97e7aec1d4360c20a27afccfd9fae0b"
    "f91b65c5524733ab8f593dabcd62b357"
    "1639d624e65152ab8f530c359f0861d8"
    "07ca0dbf500d6a6156a38e088a22b65e"
    "52bc514d16ccf806818ce91ab7793736"
    "5af90bbf74a35be6b40b8eedf2785e42"
    "874d"
)


def _batched(data: bytes, count: int = 1) -> Array:
    return fnp.asarray(
        np.tile(np.frombuffer(data, dtype=np.uint8), (count, 1)), dtype=fnp.uint8
    )


def _counters(value: int, count: int = 1) -> Array:
    return fnp.asarray(np.full(count, value, dtype=np.uint32), dtype=fnp.uint32)


class ChaCha20VectorTest(absltest.TestCase):
    def test_the_block_function_matches_rfc_8439(self) -> None:
        stream = chacha20.keystream(
            _batched(_KEY_2_3_2), _batched(_NONCE_2_3_2), _counters(_COUNTER_2_3_2), 1
        )
        self.assertEqual(bytes(np.asarray(stream)[0]), _KEYSTREAM_2_3_2)

    def test_encryption_matches_rfc_8439(self) -> None:
        ciphertext = chacha20.encrypt(
            _batched(_KEY_2_4_2),
            _batched(_NONCE_2_4_2),
            _counters(_COUNTER_2_4_2),
            _batched(_PLAINTEXT_2_4_2),
        )
        self.assertEqual(bytes(np.asarray(ciphertext)[0]), _CIPHERTEXT_2_4_2)

    def test_encryption_is_its_own_inverse(self) -> None:
        args = (_batched(_KEY_2_4_2), _batched(_NONCE_2_4_2), _counters(_COUNTER_2_4_2))
        ciphertext = chacha20.encrypt(*args, _batched(_PLAINTEXT_2_4_2))
        recovered = chacha20.encrypt(*args, ciphertext)
        self.assertEqual(bytes(np.asarray(recovered)[0]), _PLAINTEXT_2_4_2)


class ChaCha20ShapeTest(absltest.TestCase):
    def test_block_i_uses_counter_plus_i(self) -> None:
        # The parallel-block layout is only correct if it agrees with computing
        # each block on its own, which is what the RFC describes.
        multi = chacha20.keystream(
            _batched(_KEY_2_3_2), _batched(_NONCE_2_3_2), _counters(7), 4
        )
        singles = [
            chacha20.keystream(
                _batched(_KEY_2_3_2), _batched(_NONCE_2_3_2), _counters(7 + index), 1
            )
            for index in range(4)
        ]
        self.assertEqual(
            bytes(np.asarray(multi)[0]),
            b"".join(bytes(np.asarray(single)[0]) for single in singles),
        )

    def test_batch_entries_are_independent(self) -> None:
        rng = np.random.default_rng(0)
        keys = fnp.asarray(rng.integers(0, 256, (3, 32)), dtype=fnp.uint8)
        nonces = fnp.asarray(rng.integers(0, 256, (3, 12)), dtype=fnp.uint8)
        counters = fnp.asarray(np.array([0, 5, 9], dtype=np.uint32))
        together = np.asarray(chacha20.keystream(keys, nonces, counters, 2))
        for index in range(3):
            alone = chacha20.keystream(
                keys[index][None], nonces[index][None], counters[index][None], 2
            )
            np.testing.assert_array_equal(together[index], np.asarray(alone)[0])

    def test_the_counter_wraps_at_32_bits(self) -> None:
        # RFC 8439 §2.3 fixes the counter at 32 bits, so it must wrap rather than
        # carry into the nonce words.
        wrapped = chacha20.keystream(
            _batched(_KEY_2_3_2), _batched(_NONCE_2_3_2), _counters(0xFFFFFFFF), 2
        )
        from_zero = chacha20.keystream(
            _batched(_KEY_2_3_2), _batched(_NONCE_2_3_2), _counters(0), 1
        )
        self.assertEqual(
            bytes(np.asarray(wrapped)[0])[chacha20.BLOCK_SIZE :],
            bytes(np.asarray(from_zero)[0]),
        )

    def test_a_partial_final_block_is_truncated_not_padded(self) -> None:
        plaintext = _batched(bytes(range(70)))
        ciphertext = chacha20.encrypt(
            _batched(_KEY_2_4_2), _batched(_NONCE_2_4_2), _counters(1), plaintext
        )
        self.assertEqual(ciphertext.shape, plaintext.shape)

    def test_it_traces_as_one_computation(self) -> None:
        jitted = frx.jit(chacha20.encrypt, static_argnums=())
        ciphertext = jitted(
            _batched(_KEY_2_4_2),
            _batched(_NONCE_2_4_2),
            _counters(_COUNTER_2_4_2),
            _batched(_PLAINTEXT_2_4_2),
        )
        self.assertEqual(bytes(np.asarray(ciphertext)[0]), _CIPHERTEXT_2_4_2)


if __name__ == "__main__":
    absltest.main()
