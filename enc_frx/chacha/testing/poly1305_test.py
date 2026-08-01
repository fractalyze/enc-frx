# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Poly1305 against RFC 8439, and against exact integer arithmetic.

The published vectors gate the construction. They do not gate the thing most
likely to be wrong here: the limb layout. This implementation carries ten limbs
of 13 bits in `uint32` lanes because the stack has no 64-bit integer, and its
correctness rests on an accumulator bound that a handful of vectors will never
approach.

So the real gate is `differential`: the same messages and keys through a
`GF(2^130 - 5)` reference written in Python's arbitrary-precision integers, on
random inputs and on the inputs chosen to stress the carry chain — all-`0xff`
keys and messages, values that sit just under the modulus, lengths that
straddle the block boundary. A carry bug survives the RFC vectors; it does not
survive this.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.chacha import poly1305
from enc_frx.chacha.testing.rfc8439_reference import poly1305_mac as _reference

_P = (1 << 130) - 5

# RFC 8439 §2.5.2.
_KEY_2_5_2 = bytes.fromhex(
    "85d6be7857556d337f4452fe42d506a8" "0103808afb0db2fd4abff6af4149f51b"
)
_MESSAGE_2_5_2 = b"Cryptographic Forum Research Group"
_TAG_2_5_2 = bytes.fromhex("a8061dc1305136c6c22b8baf0c0127a9")


def _mac(key: bytes, message: bytes) -> bytes:
    tag = poly1305.mac(
        fnp.asarray(np.frombuffer(key, dtype=np.uint8))[None],
        fnp.asarray(np.frombuffer(message, dtype=np.uint8).copy())[None],
    )
    return bytes(np.asarray(tag)[0])


class Poly1305VectorTest(absltest.TestCase):
    def test_the_rfc_8439_vector(self) -> None:
        self.assertEqual(_mac(_KEY_2_5_2, _MESSAGE_2_5_2), _TAG_2_5_2)

    def test_the_reference_agrees_with_the_rfc(self) -> None:
        # The differential oracle is only worth anything if it is itself right.
        self.assertEqual(_reference(_KEY_2_5_2, _MESSAGE_2_5_2), _TAG_2_5_2)


class Poly1305DifferentialTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("one_byte", 1),
        ("under_a_block", 15),
        ("exactly_one_block", 16),
        ("just_over_a_block", 17),
        ("several_blocks", 64),
        ("ragged_tail", 70),
    )
    def test_random_inputs_match_exact_arithmetic(self, length: int) -> None:
        rng = np.random.default_rng(length)
        for _ in range(8):
            key = bytes(rng.integers(0, 256, poly1305.KEY_SIZE, dtype=np.uint8))
            message = bytes(rng.integers(0, 256, length, dtype=np.uint8))
            self.assertEqual(_mac(key, message), _reference(key, message))

    @parameterized.named_parameters(
        # The carry chain's worst cases: a maximal clamped `r`, maximal message
        # limbs, and the accumulator sitting immediately below the modulus.
        ("all_ones", b"\xff" * 32, b"\xff" * 48),
        ("max_r_zero_s", b"\xff" * 16 + b"\x00" * 16, b"\xff" * 32),
        ("zero_r_max_s", b"\x00" * 16 + b"\xff" * 16, b"\xff" * 16),
        (
            "modulus_minus_one",
            b"\xff" * 32,
            ((_P - 1) % (1 << 128)).to_bytes(16, "little"),
        ),
        ("zero_message_block", b"\xff" * 32, b"\x00" * 32),
        ("single_high_bit", b"\xff" * 32, b"\x00" * 15 + b"\x80"),
    )
    def test_extreme_inputs_match_exact_arithmetic(
        self, key: bytes, message: bytes
    ) -> None:
        self.assertEqual(_mac(key, message), _reference(key, message))

    def test_the_empty_message(self) -> None:
        # No blocks to absorb, so the tag is `s` alone.
        self.assertEqual(_mac(_KEY_2_5_2, b""), _reference(_KEY_2_5_2, b""))


class Poly1305LayoutTest(absltest.TestCase):
    def test_the_accumulator_cannot_overflow_a_uint32_lane(self) -> None:
        # The invariant the whole limb layout rests on, restated where a change
        # to the radix would trip it. Limbs are carried below 2^RADIX before any
        # multiply, so each product is under 2^(2*RADIX); term `d_i` sums `i + 1`
        # of them directly and `LIMBS - 1 - i` more with the factor 5 the
        # reduction contributes.
        unit = 1 << (2 * poly1305._RADIX_BITS)
        worst = max(
            (index + 1) + 5 * (poly1305._LIMBS - 1 - index)
            for index in range(poly1305._LIMBS)
        )
        self.assertLess(worst * unit, 1 << 32)
        # And the radix must tile 130 bits exactly, or the reduction is not a
        # single-factor convolution.
        self.assertEqual(poly1305._LIMBS * poly1305._RADIX_BITS, 130)


class Poly1305BatchTest(absltest.TestCase):
    def test_batch_entries_are_independent(self) -> None:
        rng = np.random.default_rng(0)
        keys = rng.integers(0, 256, (4, poly1305.KEY_SIZE), dtype=np.uint8)
        messages = rng.integers(0, 256, (4, 48), dtype=np.uint8)
        together = np.asarray(poly1305.mac(fnp.asarray(keys), fnp.asarray(messages)))
        for index in range(4):
            alone = _reference(bytes(keys[index]), bytes(messages[index]))
            self.assertEqual(bytes(together[index]), alone)


if __name__ == "__main__":
    absltest.main()
