# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The FO transform against CCTV's published run, and around the rejection path.

The published files carry the whole of one encapsulation and its decapsulation —
`ek`, `dk`, `m`, `c`, `K`, and `KBar` — so `encaps` and `decaps` are gated on
values someone else computed, at every parameter set. Key generation is not: the
files predate FIPS 203 Algorithm 13's `G(d ‖ k)` and so cannot reproduce a key
from `d` (see [`docs/schemes/ml-kem.md`](../../../docs/schemes/ml-kem.md)), which
leaves the *layout* of `dk` as what this file gates and the lattice work to
[`_k_pke_test.py`](_k_pke_test.py).

**`KBar` is the published implicit-rejection secret** — `J(z ‖ c)` for the file's
own `c`. Reaching it needs a `dk` that fails a check while `z` and `c` stay
intact, which a corrupted *key* gives and a corrupted ciphertext cannot: change
`c` and the expected secret changes with it. So both negatives are here, one
compared against the published value and one against `hashlib`.

Nothing here is self-consistency. A `decaps` that returned `J(z ‖ c)` for every
ciphertext, valid ones included, round-trips against itself perfectly.
"""

from __future__ import annotations

import hashlib
import re
from unittest import mock

import frx
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.kem import Kem
from enc_frx.ml_kem import _k_pke, encoding
from enc_frx.ml_kem.ml_kem import MlKem
from enc_frx.ml_kem.params import SEED_SIZE, decryption_key_size
from enc_frx.ml_kem.testing import cctv_vectors
from enc_frx.testing.kat import to_bytes

# FIPS 203 Table 2.
_PARAMETERS = {
    "ML-KEM-512": {"k": 2, "eta1": 3, "eta2": 2, "du": 10, "dv": 4},
    "ML-KEM-768": {"k": 3, "eta1": 2, "eta2": 2, "du": 10, "dv": 4},
    "ML-KEM-1024": {"k": 4, "eta1": 2, "eta2": 2, "du": 11, "dv": 5},
}
_SCHEMES = {name: MlKem(**row) for name, row in _PARAMETERS.items()}
_NAMED = tuple((name, name) for name in _SCHEMES)

# ML-KEM-768 for the cases that are about the transform rather than about the
# numbers, so they run once instead of three times.
_768_SET = "ML-KEM-768"
_768 = _SCHEMES[_768_SET]
_768_K = _PARAMETERS[_768_SET]["k"]
_BATCH = 4

# `cond` and `while` as jaxpr primitives — a decapsulation that branched on
# validity would show up as one of these.
_BRANCH = re.compile(r"\b(cond|while)\[")


def _published(parameter_set: str, name: str, index: int = 0) -> np.ndarray:
    return cctv_vectors.intermediate(parameter_set).array_at(name, index)


def _j(z: np.ndarray, c: np.ndarray) -> bytes:
    """`J(z ‖ c) = SHAKE256(z ‖ c, 32)`, §4.1 — the rejection secret, from
    `hashlib` rather than from this repo's own Keccak."""
    return hashlib.shake_256(bytes(z) + bytes(c)).digest(SEED_SIZE)


def _flip(value: np.ndarray, index: int) -> np.ndarray:
    out = value.copy()
    out[..., index] ^= 1
    return out


class PublishedRunTest(parameterized.TestCase):
    """Every value FIPS 203 names in one run of the transform, as published."""

    @parameterized.named_parameters(*_NAMED)
    def test_encaps_matches_the_published_ciphertext_and_secret(
        self, parameter_set: str
    ) -> None:
        c, shared = _SCHEMES[parameter_set].encaps(
            _published(parameter_set, "ek"), randomness=_published(parameter_set, "m")
        )
        vectors = cctv_vectors.intermediate(parameter_set)
        self.assertEqual(to_bytes(c), vectors.hex_at("c"))
        self.assertEqual(to_bytes(shared), vectors.hex_at("K"))

    @parameterized.named_parameters(*_NAMED)
    def test_encaps_internal_is_the_same_call_under_its_own_name(
        self, parameter_set: str
    ) -> None:
        # The seam's randomness argument and the standard's derandomized entry
        # point are the same operation here, and a harness driving one must not
        # be driving something else.
        scheme = _SCHEMES[parameter_set]
        ek, m = _published(parameter_set, "ek"), _published(parameter_set, "m")
        self.assertEqual(
            [to_bytes(part) for part in scheme.encaps(ek, randomness=m)],
            [to_bytes(part) for part in scheme.encaps_internal(ek, m)],
        )

    @parameterized.named_parameters(*_NAMED)
    def test_decaps_recovers_the_published_secret(self, parameter_set: str) -> None:
        got = _SCHEMES[parameter_set].decaps(
            _published(parameter_set, "dk"), _published(parameter_set, "c")
        )
        self.assertEqual(
            to_bytes(got), cctv_vectors.intermediate(parameter_set).hex_at("K")
        )

    @parameterized.named_parameters(*_NAMED)
    def test_a_key_whose_halves_disagree_yields_the_published_rejection_secret(
        self, parameter_set: str
    ) -> None:
        """`KBar` exactly — the one negative the files publish a value for.

        Corrupting `H(ek)` inside `dk` fails FIPS 203 §7.3's hash check while
        leaving `z` and `c` untouched, so the expected secret is still `J(z ‖ c)`
        — which the file publishes as `KBar`. A ciphertext corruption cannot
        reach it: it changes the value being derived.
        """
        scheme = _SCHEMES[parameter_set]
        dk = _published(parameter_set, "dk").copy()
        # `dk = dk_PKE ‖ ek ‖ H(ek) ‖ z`, so the hash starts 64 bytes from the end.
        dk[-2 * SEED_SIZE] ^= 1
        got = scheme.decaps(dk, _published(parameter_set, "c"))
        self.assertEqual(
            to_bytes(got), cctv_vectors.intermediate(parameter_set).hex_at("KBar")
        )

    @parameterized.named_parameters(*_NAMED)
    def test_a_corrupted_ciphertext_yields_j_of_z_and_that_ciphertext(
        self, parameter_set: str
    ) -> None:
        # Not merely "a different secret": the expected value is computed from
        # `hashlib`, so a rejection path deriving with `H` instead of `J`, or
        # over the wrong `z`, fails here rather than passing as "different".
        scheme = _SCHEMES[parameter_set]
        c = _flip(_published(parameter_set, "c"), 0)
        z = _published(parameter_set, "z")
        got = scheme.decaps(_published(parameter_set, "dk"), c)
        self.assertEqual(to_bytes(got), _j(z, c))

    @parameterized.named_parameters(*_NAMED)
    def test_keygen_lays_the_decapsulation_key_out_as_published(
        self, parameter_set: str
    ) -> None:
        """`dk = dk_PKE ‖ ek ‖ H(ek) ‖ z`, §7.1, against the published `dk`.

        The vectors' `d` cannot drive `keygen` — the expansion they predate is
        the module docstring's subject — so what is gated here is the frame the
        transform adds: which field holds what, that the hash is `H` and not `G`,
        and that `z` arrives from the seed rather than from anywhere else.
        """
        scheme = _SCHEMES[parameter_set]
        vectors = cctv_vectors.intermediate(parameter_set)
        framed = encoding.encode_dk(
            _published(parameter_set, "dkPKE", 0),
            _published(parameter_set, "ek"),
            _published(parameter_set, "H(ek)"),
            _published(parameter_set, "z"),
        )
        self.assertEqual(to_bytes(framed), vectors.hex_at("dk"))

        z = _published(parameter_set, "z")
        ek, dk = scheme.keygen(np.concatenate([_published(parameter_set, "d"), z]))
        _, ek_field, hash_field, z_field = encoding.decode_dk(
            dk, _PARAMETERS[parameter_set]["k"]
        )
        self.assertEqual(to_bytes(ek_field), to_bytes(ek))
        self.assertEqual(to_bytes(hash_field), hashlib.sha3_256(to_bytes(ek)).digest())
        self.assertEqual(to_bytes(z_field), bytes(z))

    @parameterized.named_parameters(*_NAMED)
    def test_a_generated_key_round_trips(self, parameter_set: str) -> None:
        # The published run enters at `ek`/`dk`; this is the only case that
        # drives the amended key generation into the transform.
        scheme = _SCHEMES[parameter_set]
        rng = np.random.default_rng(len(parameter_set))
        seed = rng.integers(0, 256, scheme.seed_size, dtype=np.uint8)
        m = rng.integers(0, 256, scheme.randomness_size, dtype=np.uint8)
        ek, dk = scheme.keygen(seed)
        c, shared = scheme.encaps(ek, randomness=m)
        self.assertEqual(to_bytes(scheme.decaps(dk, c)), to_bytes(shared))


class RejectionTest(absltest.TestCase):
    """That the re-encryption runs, consumes the whole ciphertext, and decides
    per batch entry — none of which a published positive vector can see."""

    def test_re_encryption_runs_once_over_the_whole_batch(self) -> None:
        """The cost, asserted rather than inferred from the answer.

        Every entry is invalid here, which is where an implementation tempted to
        skip the work would skip it — and skipping is not an optimization,
        because there is no signal that a ciphertext is fine other than this
        computation.
        """
        keys, ciphertexts, _, _ = _batch()
        corrupted = _flip(ciphertexts, 0)
        with mock.patch.object(
            _k_pke, "encrypt", wraps=_k_pke.encrypt
        ) as re_encryption:
            _768.decaps(keys, corrupted)
        re_encryption.assert_called_once()
        self.assertEqual(
            np.shape(re_encryption.call_args.args[0])[0],
            _BATCH,
            "re-encryption ran on a subset of the batch",
        )

    def test_a_ciphertext_that_still_decrypts_to_the_message_is_rejected(self) -> None:
        """The check consumes the whole ciphertext, not just what decrypts.

        Compression throws bits away, so a small change to the tail of `c`
        recovers the *same* message — and an implementation that stopped at
        "decryption succeeded" would answer with the sender's own secret. Only
        re-encryption sees the difference.
        """
        keys, ciphertexts, secrets, seeds = _batch()
        tampered = _flip(ciphertexts, -1)
        dk_pke, _, _, _ = encoding.decode_dk(keys, _768_K)
        args = {name: _PARAMETERS[_768_SET][name] for name in ("k", "du", "dv")}
        np.testing.assert_array_equal(
            np.asarray(_k_pke.decrypt(dk_pke, tampered, **args)),
            np.asarray(_k_pke.decrypt(dk_pke, ciphertexts, **args)),
            "the flipped bit changed the message, so this case proves nothing",
        )
        got = np.asarray(_768.decaps(keys, tampered))
        for entry, (z, c) in enumerate(zip(seeds, tampered)):
            self.assertEqual(bytes(got[entry]), _j(z, c), f"entry {entry}")
        self.assertNotEqual(bytes(got[0]), bytes(secrets[0]))

    def test_a_mixed_batch_is_decided_per_entry(self) -> None:
        """Entries 1 and 2 corrupted, and only those two rejected.

        A check reduced over the batch rather than over each ciphertext passes
        every all-valid and every all-invalid case ever written.
        """
        keys, ciphertexts, secrets, seeds = _batch()
        corrupted = (1, 2)
        mixed = ciphertexts.copy()
        for entry in corrupted:
            mixed[entry] = _flip(mixed[entry], 0)
        got = np.asarray(_768.decaps(keys, mixed))
        for entry in range(_BATCH):
            want = (
                _j(seeds[entry], mixed[entry])
                if entry in corrupted
                else bytes(secrets[entry])
            )
            self.assertEqual(bytes(got[entry]), want, f"entry {entry}")

    def test_rejection_repeats(self) -> None:
        # Deterministic in `(z, c)`, so a caller cannot separate a rejection from
        # a real secret by asking twice.
        keys, ciphertexts, _, _ = _batch()
        corrupted = _flip(ciphertexts, 0)
        np.testing.assert_array_equal(
            np.asarray(_768.decaps(keys, corrupted)),
            np.asarray(_768.decaps(keys, corrupted)),
        )


class TracedShapeTest(absltest.TestCase):
    def test_the_decapsulation_path_has_no_branch(self) -> None:
        # The select is arithmetic: both secrets are computed, and which one
        # comes back is a mask. A `cond` here would be the timing signal the
        # transform exists to withhold.
        keys, ciphertexts, _, _ = _batch()
        text = str(frx.make_jaxpr(_768.decaps)(keys, ciphertexts))
        # Positively, so the negative below cannot pass on an empty trace.
        self.assertIn("select_n", text, "decaps traced without a select")
        self.assertIsNone(_BRANCH.search(text), "decaps traced to a branch")

    def test_the_traced_secret_matches_the_eager_one(self) -> None:
        keys, ciphertexts, _, _ = _batch()
        np.testing.assert_array_equal(
            np.asarray(frx.jit(_768.decaps)(keys, ciphertexts)),
            np.asarray(_768.decaps(keys, ciphertexts)),
        )


class KeyCheckTest(absltest.TestCase):
    """FIPS 203 §7.2 and §7.3 as standalone operations, and folded into `decaps`."""

    def test_the_published_keys_pass_both_checks(self) -> None:
        self.assertTrue(_768.check_encapsulation_key(_published(_768_SET, "ek")))
        self.assertTrue(_768.check_decapsulation_key(_published(_768_SET, "dk")))

    def test_an_unreduced_coefficient_fails_the_encapsulation_key_check(self) -> None:
        # The first 12-bit value becomes 0xfff = 4095, which is not below `q`.
        ek = _published(_768_SET, "ek").copy()
        ek[0], ek[1] = 0xFF, 0xFF
        self.assertFalse(_768.check_encapsulation_key(ek))

    def test_halves_that_disagree_fail_the_decapsulation_key_check(self) -> None:
        dk = _published(_768_SET, "dk").copy()
        dk[-2 * SEED_SIZE] ^= 1
        self.assertFalse(_768.check_decapsulation_key(dk))

    def test_a_key_check_is_per_entry_over_a_batch(self) -> None:
        good = _published(_768_SET, "ek")
        bad = good.copy()
        bad[0], bad[1] = 0xFF, 0xFF
        got = _768.check_encapsulation_key(np.stack([good, bad, good]))
        self.assertEqual(np.asarray(got).tolist(), [True, False, True])

    def test_decaps_rejects_an_unreduced_key_rather_than_raising(self) -> None:
        """§7.2's check reaches `decaps` through the rejection path, not an
        exception — the same channel a bad ciphertext takes, for the same
        reason."""
        dk = _published(_768_SET, "dk").copy()
        c = _published(_768_SET, "c")
        # The embedded `ek` starts where `dk_PKE` ends.
        start = decryption_key_size(_768_K)
        dk[start], dk[start + 1] = 0xFF, 0xFF
        got = _768.decaps(dk, c)
        self.assertEqual(to_bytes(got), _j(_published(_768_SET, "z"), c))


class SeamTest(absltest.TestCase):
    def test_the_protocol_accepts_the_scheme(self) -> None:
        self.assertIsInstance(_768, Kem)

    def test_the_sizes_are_the_standards(self) -> None:
        # ML-KEM-768, FIPS 203 Table 3.
        self.assertEqual(_768.encapsulation_key_size, 1184)
        self.assertEqual(_768.decapsulation_key_size, 2400)
        self.assertEqual(_768.ciphertext_size, 1088)
        self.assertEqual(_768.shared_secret_size, 32)
        self.assertEqual(_768.seed_size, 64)

    def test_equality_is_by_value(self) -> None:
        # Identity equality would not error; it would silently re-trace the
        # enclosing jit zone for every freshly built instance.
        twin = MlKem(k=3, eta1=2, eta2=2, du=10, dv=4)
        self.assertEqual(_768, twin)
        self.assertEqual(hash(_768), hash(twin))
        self.assertNotEqual(_768, _SCHEMES["ML-KEM-1024"])

    def test_rejects_wrong_lengths(self) -> None:
        zeros = np.zeros(_768.decapsulation_key_size, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "key generation seed"):
            _768.keygen(zeros[: _768.seed_size - 1])
        with self.assertRaisesRegex(ValueError, "encapsulation key"):
            _768.encaps(
                zeros[: _768.encapsulation_key_size - 1],
                randomness=zeros[:SEED_SIZE],
            )
        with self.assertRaisesRegex(ValueError, "encapsulation randomness"):
            _768.encaps(
                zeros[: _768.encapsulation_key_size], randomness=zeros[: SEED_SIZE - 1]
            )
        with self.assertRaisesRegex(ValueError, "decapsulation key"):
            _768.decaps(zeros[:-1], zeros[: _768.ciphertext_size])
        with self.assertRaisesRegex(ValueError, "ciphertext"):
            _768.decaps(zeros, zeros[: _768.ciphertext_size - 1])


def _batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(decaps_keys, ciphertexts, shared_secrets, rejection_seeds)` at 768.

    Generated rather than published: one published run is one entry, and every
    case below is about the batch axis or about a value no file carries.
    """
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 256, (_BATCH, _768.seed_size), dtype=np.uint8)
    m = rng.integers(0, 256, (_BATCH, _768.randomness_size), dtype=np.uint8)
    encaps_keys, decaps_keys = frx.vmap(_768.keygen)(seeds)
    ciphertexts, secrets = _768.encaps(encaps_keys, randomness=m)
    return (
        np.asarray(decaps_keys),
        np.asarray(ciphertexts),
        np.asarray(secrets),
        seeds[:, SEED_SIZE:],
    )


if __name__ == "__main__":
    absltest.main()
