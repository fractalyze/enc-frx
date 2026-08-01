# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The AES block cipher against NIST's published vectors, and against its shape.

The gate is ACVP's AES-ECB set, fetched and sha256-pinned in //MODULE.bazel —
1069 encryption cases across the three key lengths, run batched by key length
rather than one at a time. Deriving vectors from a reference would only prove
this agrees with something written by the same hand from the same reading.

The set also publishes decryption cases and a Monte Carlo test type. This repo
implements encryption only, and a Monte Carlo case is a chained thousand-iteration
procedure rather than a block, so both are **refused rather than skipped** — a
suite that silently ran 1069 of 2140 cases and reported green would be claiming
coverage it does not have.

The structural tests cover what a vector cannot see: that the S-box really is
arithmetic (no table), and that the state's row/column transpose is the one
FIPS 197 §3.4 specifies rather than one that happens to round-trip.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array
from python.runfiles import runfiles

from enc_frx.aes import block
from enc_frx.testing.kat import AesVector, KatError, load_acvp_aes, to_bytes

_PARAMETER_SETS = ("AES-128", "AES-192", "AES-256")


def _path(repo: str, name: str) -> str:
    location = runfiles.Create().Rlocation(f"{repo}/file/{name}")
    assert location is not None, f"{repo} not in runfiles"
    return location


def _vectors() -> list[AesVector]:
    return load_acvp_aes(
        _path("acvp_aes_ecb_prompt", "prompt.json"),
        _path("acvp_aes_ecb_expected", "expectedResults.json"),
    )


def _runnable(vectors: list[AesVector], parameter_set: str) -> list[AesVector]:
    return [
        v
        for v in vectors
        if v.parameter_set == parameter_set
        and v.direction == "encrypt"
        and v.test_type == "AFT"
    ]


def _stack(items: list[bytes]) -> Array:
    return fnp.asarray(
        np.stack([np.frombuffer(item, dtype=np.uint8) for item in items])
    )


def _encrypt_ecb(cases: list[AesVector]) -> list[bytes]:
    """Every case's payload, batched by length.

    ACVP's payloads run from one block to ten, so a single stack would be ragged.
    Grouping by length keeps each call batched, and ECB is exactly "the block
    cipher, per block" — the chunking lives here rather than in the library
    because ECB is not a mode this repo implements.
    """
    by_length: dict[int, list[int]] = {}
    for index, vector in enumerate(cases):
        by_length.setdefault(len(vector.plaintext), []).append(index)

    produced: dict[int, bytes] = {}
    for length, indices in by_length.items():
        assert length % block.BLOCK_SIZE == 0, f"ECB payload of {length} bytes"
        blocks = length // block.BLOCK_SIZE
        group = [cases[index] for index in indices]
        keys = np.repeat(
            np.stack([np.frombuffer(v.key, dtype=np.uint8) for v in group]),
            blocks,
            axis=0,
        )
        payload = _stack([v.plaintext for v in group]).reshape(-1, block.BLOCK_SIZE)
        out = np.asarray(block.encrypt_block(fnp.asarray(keys), payload)).reshape(
            len(group), length
        )
        for row, index in enumerate(indices):
            produced[index] = bytes(out[row])
    return [produced[index] for index in range(len(cases))]


class AcvpVectorTest(parameterized.TestCase):
    @parameterized.parameters(*_PARAMETER_SETS)
    def test_every_published_encryption_case(self, parameter_set: str) -> None:
        cases = _runnable(_vectors(), parameter_set)
        self.assertNotEmpty(cases)
        for produced, vector in zip(_encrypt_ecb(cases), cases, strict=True):
            self.assertEqual(produced, vector.ciphertext, vector.case_id)

    def test_the_set_carries_cases_this_repo_cannot_run(self) -> None:
        # Recorded rather than filtered at load time, so the refusal below is a
        # decision this test makes rather than one the loader made silently.
        vectors = _vectors()
        directions = {v.direction for v in vectors}
        test_types = {v.test_type for v in vectors}
        self.assertEqual(directions, {"encrypt", "decrypt"})
        self.assertEqual(test_types, {"AFT", "MCT"})

    def test_only_the_monte_carlo_cases_carry_an_unexpressible_field(self) -> None:
        # MCT publishes a `resultsArray` — the thousand-iteration chain's
        # checkpoints — which is a shape this record has no room for and this
        # repo has no operation for. AFT must carry nothing, so a new ACVP field
        # on the cases actually run surfaces as a refusal.
        vectors = _vectors()
        self.assertEmpty(
            {name for v in vectors if v.test_type == "AFT" for name in v.unsupported},
            "ACVP grew a field this record does not express",
        )
        self.assertEqual(
            {name for v in vectors if v.test_type == "MCT" for name in v.unsupported},
            {"resultsArray"},
        )

    def test_a_bad_pairing_is_an_error(self) -> None:
        with self.assertRaisesRegex(KatError, "no expected result"):
            load_acvp_aes(
                _path("acvp_aes_ecb_prompt", "prompt.json"),
                _path("acvp_ml_kem_keygen_expected", "expectedResults.json"),
            )


class SBoxTest(absltest.TestCase):
    def test_inversion_agrees_with_the_field(self) -> None:
        # `x^254` is only `x^-1` if the chain is right, and the check is the
        # field's own: every non-zero byte times its inverse is one.
        values = fnp.asarray(np.arange(1, 256, dtype=np.uint8)).astype(block.GF8)
        product = values * block._inverse(values)
        np.testing.assert_array_equal(
            np.asarray(product).astype(np.uint8), np.ones(255, dtype=np.uint8)
        )

    def test_zero_inverts_to_zero(self) -> None:
        # FIPS 197 §5.1.1 defines the S-box on 0 through this convention, and the
        # addition chain gives it without a special case.
        zero = fnp.asarray(np.array([0], dtype=np.uint8)).astype(block.GF8)
        self.assertEqual(int(np.asarray(block._inverse(zero)).astype(np.uint8)[0]), 0)

    def test_the_s_box_is_a_permutation(self) -> None:
        # A table would be one by construction; an arithmetic S-box has to earn
        # it, and a wrong affine map or a wrong chain would collide somewhere.
        every = fnp.asarray(np.arange(256, dtype=np.uint8)).astype(block.GF8)
        image = np.asarray(block.sub_bytes(every)).astype(np.uint8)
        self.assertLen(set(image.tolist()), 256)

    def test_the_s_box_fixes_the_published_endpoints(self) -> None:
        # FIPS 197 §5.1.1's affine constant makes S(0) = 0x63; S(1) = 0x7c
        # follows from 1 being its own inverse.
        every = fnp.asarray(np.arange(256, dtype=np.uint8)).astype(block.GF8)
        image = np.asarray(block.sub_bytes(every)).astype(np.uint8)
        self.assertEqual(int(image[0]), 0x63)
        self.assertEqual(int(image[1]), 0x7C)

    def test_the_whole_cipher_lowers_without_a_gather(self) -> None:
        # The property the arithmetic S-box exists for, asserted over the cipher
        # rather than over SubBytes alone — a table would be a gather indexed by
        # a secret byte, and `ShiftRows` is expressed as static slices so that
        # *no* gather survives and the check needs no exemption.
        zeros = fnp.asarray(np.zeros((2, 16), dtype=np.uint8))
        text = frx.jit(block.encrypt_block).lower(zeros, zeros).as_text()
        self.assertNotIn("gather", text)


class StateLayoutTest(absltest.TestCase):
    def test_bytes_fill_the_state_down_columns(self) -> None:
        # FIPS 197 §3.4. A row-major reading round-trips just as well, and would
        # move ShiftRows and MixColumns onto the wrong axis.
        block_bytes = fnp.asarray(np.arange(16, dtype=np.uint8))[None]
        state = np.asarray(block._to_state(block_bytes)).astype(np.uint8)[0]
        np.testing.assert_array_equal(state[:, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(state[0, :], [0, 4, 8, 12])

    def test_shift_rows_rotates_row_r_by_r(self) -> None:
        state = fnp.asarray(np.arange(16, dtype=np.uint8).reshape(1, 4, 4)).astype(
            block.GF8
        )
        shifted = np.asarray(block.shift_rows(state)).astype(np.uint8)[0]
        np.testing.assert_array_equal(shifted[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(shifted[1], [5, 6, 7, 4])
        np.testing.assert_array_equal(shifted[3], [15, 12, 13, 14])

    def test_the_round_trip_is_the_identity(self) -> None:
        block_bytes = fnp.asarray(np.arange(16, dtype=np.uint8))[None]
        recovered = block._from_state(block._to_state(block_bytes))
        np.testing.assert_array_equal(
            np.asarray(recovered).astype(np.uint8), np.arange(16)[None]
        )


class KeyScheduleTest(absltest.TestCase):
    def test_the_round_count_follows_the_key_length(self) -> None:
        for key_size, rounds in ((16, 10), (24, 12), (32, 14)):
            keys = fnp.asarray(np.zeros((1, key_size), dtype=np.uint8))
            self.assertLen(block.key_schedule(keys), rounds + 1)

    def test_the_first_round_key_is_the_key(self) -> None:
        # FIPS 197 §5.2: the first Nk words are the key itself, and a round key's
        # words are its columns.
        key = np.arange(16, dtype=np.uint8)
        schedule = block.key_schedule(fnp.asarray(key)[None])
        first = np.asarray(schedule[0]).astype(np.uint8)[0]
        np.testing.assert_array_equal(first.T.reshape(-1), key)

    def test_a_wrong_key_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "16, 24, or 32"):
            block.key_schedule(fnp.asarray(np.zeros((1, 20), dtype=np.uint8)))


class ScheduleReuseTest(absltest.TestCase):
    def test_encrypting_through_a_shared_schedule_agrees(self) -> None:
        # The entry point a mode uses: expand once, encrypt many. It must produce
        # exactly what expanding per block does, or hoisting changes the answer.
        cases = [
            v
            for v in _runnable(_vectors(), "AES-128")
            if len(v.plaintext) == block.BLOCK_SIZE
        ][:16]
        keys = _stack([v.key for v in cases])
        blocks = _stack([v.plaintext for v in cases])
        hoisted = block.encrypt_with_schedule(block.key_schedule(keys), blocks)
        for index, vector in enumerate(cases):
            self.assertEqual(to_bytes(hoisted[index]), vector.ciphertext)

    def test_one_schedule_serves_many_blocks(self) -> None:
        # What CTR mode does: a single key's schedule broadcast over many blocks.
        rng = np.random.default_rng(1)
        key = fnp.asarray(rng.integers(0, 256, (1, 16), dtype=np.uint8))
        blocks = fnp.asarray(rng.integers(0, 256, (8, 16), dtype=np.uint8))
        schedule = block.key_schedule(key)
        shared = block.encrypt_with_schedule(schedule, blocks)
        for index in range(8):
            alone = block.encrypt_block(key, blocks[index][None])
            np.testing.assert_array_equal(
                np.asarray(shared)[index], np.asarray(alone)[0]
            )


class BatchTest(absltest.TestCase):
    def test_entries_are_independent(self) -> None:
        rng = np.random.default_rng(0)
        keys = fnp.asarray(rng.integers(0, 256, (4, 16), dtype=np.uint8))
        blocks = fnp.asarray(rng.integers(0, 256, (4, 16), dtype=np.uint8))
        together = np.asarray(block.encrypt_block(keys, blocks))
        for index in range(4):
            alone = block.encrypt_block(keys[index][None], blocks[index][None])
            np.testing.assert_array_equal(together[index], np.asarray(alone)[0])

    def test_it_traces_as_one_computation(self) -> None:
        cases = [
            v
            for v in _runnable(_vectors(), "AES-128")
            if len(v.plaintext) == block.BLOCK_SIZE
        ][:8]
        produced = frx.jit(block.encrypt_block)(
            _stack([v.key for v in cases]), _stack([v.plaintext for v in cases])
        )
        for index, vector in enumerate(cases):
            self.assertEqual(to_bytes(produced[index]), vector.ciphertext)


if __name__ == "__main__":
    absltest.main()
