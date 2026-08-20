# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""POLYVAL — the RFC 8452 Appendix A worked example, the appendix's own
mulX rows, and a differential against a from-the-definition plain-integer
POLYVAL (§3's field, little-endian encoding, `S_j = (S_{j-1} ⊕ X_j)·H·x^-128`)
— because the implementation under test is the Appendix A *identity*, so a
test that only used the identity again would prove the plumbing, not the
convention."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from enc_frx.aes import polyval

# GF(2^128) with x^128 + x^127 + x^126 + x^121 + 1, POLYVAL's little-endian
# encoding: bit i of the integer is the coefficient of x^i.
_POLY = (1 << 128) | (1 << 127) | (1 << 126) | (1 << 121) | 1


def _field_mul(left: int, right: int) -> int:
    product = 0
    for bit in range(128):
        if (left >> bit) & 1:
            product ^= right << bit
    for bit in range(255, 127, -1):
        if (product >> bit) & 1:
            product ^= _POLY << (bit - 128)
    return product


def _field_pow(base: int, exponent: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _field_mul(result, base)
        base = _field_mul(base, base)
        exponent >>= 1
    return result


def _int_polyval(subkey: bytes, blocks: list[bytes]) -> bytes:
    h = int.from_bytes(subkey, "little")
    x128 = _field_mul(1 << 127, 1 << 1)  # x^128, reduced into the field
    h_star = _field_mul(h, _field_pow(x128, 2**128 - 2))  # H * x^-128
    state = 0
    for block in blocks:
        state = _field_mul(state ^ int.from_bytes(block, "little"), h_star)
    return state.to_bytes(16, "little")


def _batch(*hex_blocks: str) -> np.ndarray:
    rows = [np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hex_blocks]
    return np.stack(rows)[None, :, :]


class PolyvalTest(absltest.TestCase):
    def test_appendix_a_worked_example(self) -> None:
        subkey = np.frombuffer(
            bytes.fromhex("25629347589242761d31f826ba4b757b"), dtype=np.uint8
        )[None, :]
        blocks = _batch(
            "4f4f95668c83dfb6401762bb2d01a262",
            "d1a24ddd2721d006bbe45f20d3c9f362",
        )
        got = bytes(np.asarray(polyval.polyval(subkey, blocks))[0])
        self.assertEqual(got.hex(), "f7a3b47b846119fae5b7866cf5e5b77e")

    def test_appendix_a_mulx_rows(self) -> None:
        one = np.zeros((1, 16), dtype=np.uint8)
        one[0, 0] = 0x01
        got = bytes(np.asarray(polyval.mulx_ghash(one))[0])
        self.assertEqual(got.hex(), "00800000000000000000000000000000")
        general = np.frombuffer(
            bytes.fromhex("9c98c04df9387ded828175a92ba652d8"), dtype=np.uint8
        )[None, :]
        got = bytes(np.asarray(polyval.mulx_ghash(general))[0])
        self.assertEqual(got.hex(), "4e4c6026fc9c3ef6c140bad495d3296c")

    def test_appendix_a_bridged_key(self) -> None:
        # The exact intermediate the identity pivots on: the GHASH key that
        # computes POLYVAL for the worked example's H.
        subkey = np.frombuffer(
            bytes.fromhex("25629347589242761d31f826ba4b757b"), dtype=np.uint8
        )[None, ::-1]
        got = bytes(np.asarray(polyval.mulx_ghash(subkey))[0])
        self.assertEqual(got.hex(), "dcbaa5dd137c188ebb21492c23c9b112")

    def test_matches_definition_on_random_batches(self) -> None:
        rng = np.random.default_rng(0)
        for blocks_count in (1, 2, 5):
            subkeys = rng.integers(0, 256, size=(4, 16), dtype=np.uint8)
            blocks = rng.integers(0, 256, size=(4, blocks_count, 16), dtype=np.uint8)
            got = np.asarray(polyval.polyval(subkeys, blocks))
            for row in range(4):
                want = _int_polyval(
                    bytes(subkeys[row]),
                    [bytes(blocks[row, j]) for j in range(blocks_count)],
                )
                self.assertEqual(bytes(got[row]), want)

    def test_length_block_is_little_endian(self) -> None:
        got = bytes(np.asarray(polyval.length_block(1, 8)))
        self.assertEqual(got.hex(), "08000000000000004000000000000000")


if __name__ == "__main__":
    absltest.main()
