# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES-CTR against ACVP's published cases, and against its counter discipline.

**ACVP measures a CTR payload in bits**, and most of its cases are not a whole
number of bytes — the very first is nine bits. The mode here is byte-granular,
because no caller in this repo has a sub-byte message and GCM never produces one.

That is a presentation difference rather than a different operation: the trailing
bits of ACVP's plaintext are zero, so the keystream is unchanged and only the
final partial byte's *encoding* differs, with the unused bits published as zero.
So the cases run with those bits masked, and a test pins that the mask never
reaches beyond the final byte — if it did, this would be adapting the operation
to the vectors rather than the other way around.

The vectors gate the keystream. What they cannot see is the counter: the payloads
are short, so most never cross a byte boundary in the counter, let alone the
32-bit wrap that bounds a single (key, IV) pair. Those get their own tests,
against `encrypt_block` on hand-built counter blocks — the definition, rather
than this mode's own arithmetic.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array
from python.runfiles import runfiles

from enc_frx.aes import block, ctr
from enc_frx.testing.kat import AesVector, load_acvp_aes, to_bytes

_PARAMETER_SETS = ("AES-128", "AES-192", "AES-256")


def _path(repo: str, name: str) -> str:
    location = runfiles.Create().Rlocation(f"{repo}/file/{name}")
    assert location is not None, f"{repo} not in runfiles"
    return location


def _vectors() -> list[AesVector]:
    return load_acvp_aes(
        _path("acvp_aes_ctr_prompt", "prompt.json"),
        _path("acvp_aes_ctr_expected", "expectedResults.json"),
    )


def _runnable(parameter_set: str) -> list[AesVector]:
    return [
        v
        for v in _vectors()
        if v.parameter_set == parameter_set
        and v.direction == "encrypt"
        and v.test_type == "AFT"
    ]


def _array(data: bytes) -> Array:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))[None]


def _mask_tail(data: bytes, bits: int) -> bytes:
    """Zero the bits past `bits`, which is how ACVP publishes a partial byte."""
    if bits % 8 == 0:
        return data
    keep = bits // 8
    head = data[:keep]
    partial = data[keep] & (0xFF << (8 - bits % 8)) & 0xFF
    return head + bytes([partial]) + bytes(len(data) - keep - 1)


class AcvpVectorTest(parameterized.TestCase):
    @parameterized.parameters(*_PARAMETER_SETS)
    def test_every_published_encryption_case(self, parameter_set: str) -> None:
        cases = _runnable(parameter_set)
        self.assertNotEmpty(cases)
        for vector in cases:
            schedule = block.key_schedule(_array(vector.key))
            produced = ctr.encrypt(
                schedule, _array(vector.iv), _array(vector.plaintext)
            )
            self.assertEqual(
                _mask_tail(to_bytes(produced[0]), vector.payload_bits),
                vector.ciphertext,
                vector.case_id,
            )

    def test_the_set_carries_a_direction_this_repo_does_not_run(self) -> None:
        self.assertEqual({v.direction for v in _vectors()}, {"encrypt", "decrypt"})

    def test_no_case_carries_a_field_the_record_cannot_express(self) -> None:
        self.assertEmpty({name for v in _vectors() for name in v.unsupported})

    def test_most_published_cases_are_not_a_whole_number_of_bytes(self) -> None:
        # The reason the mask exists. If ACVP ever became byte-aligned this test
        # fails and the mask should go.
        cases = [v for v in _vectors() if v.direction == "encrypt"]
        ragged = [v for v in cases if v.payload_bits % 8]
        self.assertNotEmpty(ragged)
        self.assertGreater(len(ragged), len(cases) // 4)

    def test_the_mask_never_reaches_past_the_final_byte(self) -> None:
        # What keeps the mask honest: it must change nothing before the last
        # byte, or it would be hiding a keystream error rather than an encoding
        # difference.
        for vector in _runnable("AES-128"):
            produced = to_bytes(
                ctr.encrypt(
                    block.key_schedule(_array(vector.key)),
                    _array(vector.iv),
                    _array(vector.plaintext),
                )[0]
            )
            masked = _mask_tail(produced, vector.payload_bits)
            whole = vector.payload_bits // 8
            self.assertEqual(produced[:whole], masked[:whole], vector.case_id)

    def test_decryption_is_the_same_operation(self) -> None:
        # CTR is its own inverse, so the published decrypt cases are runnable
        # after all — through `encrypt`. Checked rather than assumed.
        cases = [
            v
            for v in _vectors()
            if v.direction == "decrypt" and v.parameter_set == "AES-128"
        ][:8]
        self.assertNotEmpty(cases)
        for vector in cases:
            schedule = block.key_schedule(_array(vector.key))
            produced = ctr.encrypt(
                schedule, _array(vector.iv), _array(vector.ciphertext)
            )
            self.assertEqual(
                _mask_tail(to_bytes(produced[0]), vector.payload_bits),
                vector.plaintext,
                vector.case_id,
            )


class CounterTest(absltest.TestCase):
    def test_block_i_is_the_counter_incremented_i_times(self) -> None:
        # Against the block cipher on hand-built counters, so the mode's own
        # arithmetic is not both the subject and the oracle.
        rng = np.random.default_rng(0)
        key = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        counter = bytes(rng.integers(0, 256, 12, dtype=np.uint8)) + (7).to_bytes(
            4, "big"
        )
        schedule = block.key_schedule(_array(key))
        produced = to_bytes(ctr.keystream(schedule, _array(counter), 4)[0])
        for index in range(4):
            expected_counter = counter[:12] + (7 + index).to_bytes(4, "big")
            expected = block.encrypt_block(_array(key), _array(expected_counter))
            self.assertEqual(
                produced[index * 16 : (index + 1) * 16], to_bytes(expected[0])
            )

    def test_the_counter_wraps_at_32_bits(self) -> None:
        # SP 800-38D §6.2: only the trailing 32 bits move, and they wrap rather
        # than carrying into the nonce.
        rng = np.random.default_rng(1)
        key = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        nonce = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        schedule = block.key_schedule(_array(key))
        wrapped = to_bytes(
            ctr.keystream(schedule, _array(nonce + b"\xff\xff\xff\xff"), 2)[0]
        )
        from_zero = to_bytes(ctr.keystream(schedule, _array(nonce + bytes(4)), 1)[0])
        self.assertEqual(wrapped[16:], from_zero)

    def test_a_partial_final_block_is_truncated(self) -> None:
        rng = np.random.default_rng(2)
        key = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        schedule = block.key_schedule(_array(key))
        data = _array(bytes(rng.integers(0, 256, 37, dtype=np.uint8)))
        self.assertEqual(
            ctr.encrypt(schedule, _array(bytes(16)), data).shape, data.shape
        )

    def test_one_schedule_serves_the_whole_message(self) -> None:
        # The reason every entry point takes a schedule: the key is expanded once
        # per message, not once per block.
        rng = np.random.default_rng(3)
        key = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        data = bytes(rng.integers(0, 256, 160, dtype=np.uint8))
        schedule = block.key_schedule(_array(key))
        produced = to_bytes(ctr.encrypt(schedule, _array(bytes(16)), _array(data))[0])
        for index in range(10):
            counter = bytes(12) + index.to_bytes(4, "big")
            keystream = to_bytes(block.encrypt_block(_array(key), _array(counter))[0])
            chunk = data[index * 16 : (index + 1) * 16]
            self.assertEqual(
                produced[index * 16 : (index + 1) * 16],
                bytes(a ^ b for a, b in zip(chunk, keystream, strict=True)),
            )


class InitialCounterTest(absltest.TestCase):
    def test_a_96_bit_iv_is_used_directly(self) -> None:
        # SP 800-38D §7.1: the counter starts at one, appended to the IV.
        iv = bytes(range(12))
        produced = ctr.initial_counter(_array(bytes(16)), _array(iv))
        self.assertEqual(to_bytes(produced[0]), iv + (1).to_bytes(4, "big"))

    def test_another_length_is_hashed(self) -> None:
        # The GHASH-derived path. It must differ from the trivial one, or the
        # branch is not doing anything.
        subkey = _array(bytes(range(16)))
        derived = ctr.initial_counter(subkey, _array(bytes(range(8))))
        self.assertEqual(derived.shape, (1, 16))
        self.assertNotEqual(to_bytes(derived[0]), bytes(range(8)) + bytes(8))


class BatchTest(absltest.TestCase):
    def test_entries_are_independent(self) -> None:
        rng = np.random.default_rng(4)
        keys = fnp.asarray(rng.integers(0, 256, (4, 16), dtype=np.uint8))
        counters = fnp.asarray(rng.integers(0, 256, (4, 16), dtype=np.uint8))
        data = fnp.asarray(rng.integers(0, 256, (4, 48), dtype=np.uint8))
        together = np.asarray(ctr.encrypt(block.key_schedule(keys), counters, data))
        for index in range(4):
            alone = ctr.encrypt(
                block.key_schedule(keys[index][None]),
                counters[index][None],
                data[index][None],
            )
            np.testing.assert_array_equal(together[index], np.asarray(alone)[0])

    def test_it_traces_as_one_computation(self) -> None:
        vector = next(v for v in _runnable("AES-128") if v.payload_bits % 8 == 0)
        schedule = block.key_schedule(_array(vector.key))
        produced = frx.jit(ctr.encrypt)(
            schedule, _array(vector.iv), _array(vector.plaintext)
        )
        self.assertEqual(to_bytes(produced[0]), vector.ciphertext)


if __name__ == "__main__":
    absltest.main()
