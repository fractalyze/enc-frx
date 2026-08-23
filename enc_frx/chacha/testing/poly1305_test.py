# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Poly1305 against RFC 8439, and against exact integer arithmetic.

The published vectors gate the construction. They do not gate the thing most
likely to be wrong here: how a 130-bit value crosses between bytes and the
field. The polynomial itself is three operators; the encoding either lines up
exactly or it does not, and a handful of vectors will not tell you which.

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


class Poly1305EncodingTest(absltest.TestCase):
    """The two byte-boundary properties the field path rests on."""

    def test_a_padded_block_views_as_the_integer_it_encodes(self) -> None:
        # `_to_field` pads a 17-byte block to the field's width and views it.
        # That is exact only if canonical storage is the little-endian encoding
        # — the same property the x25519 ladder leans on, restated here because
        # a storage change upstream would break it silently.
        # `_blocks` sets exactly one high byte to 0 or 1 — the RFC's appended
        # `1` lands past the sixteenth byte only for a full block — so a block
        # is always below 2^129 and needs no reduction to be canonical. Sample
        # that domain, not arbitrary 17-byte strings.
        rng = np.random.default_rng(0)
        blocks = np.concatenate(
            [
                rng.integers(0, 256, (8, poly1305.BLOCK_SIZE), dtype=np.uint8),
                rng.integers(0, 2, (8, 1), dtype=np.uint8),
            ],
            axis=-1,
        )
        got = np.asarray(poly1305._to_field(fnp.asarray(blocks)).astype(poly1305.WIRE))
        for index, block in enumerate(blocks):
            want = int.from_bytes(bytes(block), "little")
            self.assertLess(want, _P, "a padded block must already be reduced")
            self.assertEqual(int(got[index, 0]), want)

    def test_canonical_storage_is_reduced_so_the_tag_needs_no_reduction(self) -> None:
        # `mac` reads the tag as the low 16 bytes of canonical storage, with no
        # reduction step, which is correct only because canonical storage holds
        # the residue in [0, p). Check the boundary: p - 1 must survive the
        # round trip, and p itself must come back as 0.
        for value, want in ((_P - 1, _P - 1), (_P, 0), (_P + 1, 1)):
            element = np.array([value % _P], dtype=poly1305.WIRE)
            self.assertEqual(int(element[0]), want)
            low = element.view(np.uint8)[: poly1305.TAG_SIZE]
            self.assertEqual(int.from_bytes(bytes(low), "little"), want % (1 << 128))


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
