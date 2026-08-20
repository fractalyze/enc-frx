# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""GHASH against SP 800-38D's own algorithm, and the bit order against itself.

There is no published standalone GHASH vector set — it is only ever exercised
through GCM — so the gate is a differential test against §6.3's definitional
shift-and-XOR, written over the standard's bit order and touching none of the
machinery under test. If the two agree, the reversal that bridges GCM's reflected
order to the dtype's natural basis is right.

The reversal gets its own tests as well, because it is the one thing that fails
identically for every input: a wrong basis produces a wrong tag always, which
reads as a field bug rather than a convention bug.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.aes import block, ghash
from enc_frx.aes.testing import gcm_reference


def _array(data: bytes) -> fnp.ndarray:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))[None]


def _bytes(array: fnp.ndarray) -> bytes:
    return bytes(np.asarray(array)[0])


class BitOrderTest(absltest.TestCase):
    def test_the_multiplicative_identity_is_gcms_first_bit(self) -> None:
        # SP 800-38D §6.1 reads the first bit of the first byte as the constant
        # term, so GCM's `1` is 0x80 followed by zeros — and must land on the
        # dtype's `1`.
        one = bytes([0x80]) + bytes(15)
        self.assertEqual(
            int(np.asarray(ghash.to_field(_array(one))).astype(object)[0]), 1
        )

    def test_x_is_gcms_second_bit(self) -> None:
        value = bytes([0x40]) + bytes(15)
        self.assertEqual(
            int(np.asarray(ghash.to_field(_array(value))).astype(object)[0]), 2
        )

    def test_the_conversion_round_trips(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(8):
            data = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
            self.assertEqual(
                _bytes(ghash.from_field(ghash.to_field(_array(data)))), data
            )

    def test_reversing_bits_is_an_involution(self) -> None:
        every = fnp.asarray(np.arange(256, dtype=np.uint8))
        np.testing.assert_array_equal(
            np.asarray(ghash._reverse_bits(ghash._reverse_bits(every))),
            np.arange(256, dtype=np.uint8),
        )


class MultiplyTest(absltest.TestCase):
    def test_field_multiplication_matches_the_standards_algorithm(self) -> None:
        # The dtype multiplies in the natural basis; §6.3 multiplies in GCM's.
        # Agreement across random inputs is what says the bridge is right.
        rng = np.random.default_rng(1)
        for _ in range(32):
            left = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
            right = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
            product = ghash.from_field(
                ghash.to_field(_array(left)) * ghash.to_field(_array(right))
            )
            self.assertEqual(_bytes(product), gcm_reference.multiply(left, right))

    def test_the_reduction_polynomial(self) -> None:
        # `x^127 · x` must reduce, and it is the case that distinguishes GCM's
        # polynomial from any other.
        high = bytes(15) + bytes([0x01])  # b_127 set, i.e. x^127
        x = bytes([0x40]) + bytes(15)
        self.assertEqual(
            _bytes(
                ghash.from_field(
                    ghash.to_field(_array(high)) * ghash.to_field(_array(x))
                )
            ),
            gcm_reference.multiply(high, x),
        )


class GhashTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("one_block", 1), ("two_blocks", 2), ("many", 9), ("long", 33)
    )
    def test_matches_the_standards_algorithm(self, blocks: int) -> None:
        rng = np.random.default_rng(blocks)
        for _ in range(4):
            subkey = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
            data = bytes(rng.integers(0, 256, 16 * blocks, dtype=np.uint8))
            produced = ghash.ghash(_array(subkey), _array(data).reshape(1, blocks, 16))
            self.assertEqual(_bytes(produced), gcm_reference.ghash(subkey, data))

    def test_a_zero_subkey_hashes_to_zero(self) -> None:
        # Degenerate but worth pinning: H = 0 makes every product zero, and a
        # scheme that special-cased it would diverge from the standard.
        data = bytes(range(32))
        produced = ghash.ghash(_array(bytes(16)), _array(data).reshape(1, 2, 16))
        self.assertEqual(_bytes(produced), gcm_reference.ghash(bytes(16), data))

    def test_batch_entries_are_independent(self) -> None:
        rng = np.random.default_rng(7)
        subkeys = rng.integers(0, 256, (4, 16), dtype=np.uint8)
        data = rng.integers(0, 256, (4, 3, 16), dtype=np.uint8)
        together = np.asarray(ghash.ghash(fnp.asarray(subkeys), fnp.asarray(data)))
        for index in range(4):
            self.assertEqual(
                bytes(together[index]),
                gcm_reference.ghash(
                    bytes(subkeys[index]), bytes(data[index].reshape(-1))
                ),
            )

    def test_it_traces_as_one_computation(self) -> None:
        rng = np.random.default_rng(3)
        subkey = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        data = bytes(rng.integers(0, 256, 48, dtype=np.uint8))
        produced = frx.jit(ghash.ghash)(_array(subkey), _array(data).reshape(1, 3, 16))
        self.assertEqual(_bytes(produced), gcm_reference.ghash(subkey, data))


class HelperTest(absltest.TestCase):
    def test_the_length_block_is_two_big_endian_bit_counts(self) -> None:
        encoded = bytes(np.asarray(ghash.length_block(12, 48)))
        self.assertEqual(encoded[:8], (12 * 8).to_bytes(8, "big"))
        self.assertEqual(encoded[8:], (48 * 8).to_bytes(8, "big"))

    def test_padding_rounds_up_to_whole_blocks(self) -> None:
        for length, blocks in ((0, 0), (1, 1), (16, 1), (17, 2), (48, 3)):
            data = fnp.asarray(np.zeros((1, length), dtype=np.uint8))
            self.assertEqual(block.pad_to_blocks(data).shape, (1, blocks, 16))


if __name__ == "__main__":
    absltest.main()
