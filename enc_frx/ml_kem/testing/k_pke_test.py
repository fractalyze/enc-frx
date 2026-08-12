# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""K-PKE against FIPS 203's own intermediate values, and around the loop.

Two gates that fail on different things:

- **CCTV's `intermediate/` files** publish `ek_PKE`, `dk_PKE`, the ciphertext and
  the recovered message for one run of each parameter set, so every stage is
  compared against a value someone else computed.
- **Round trips over random input** cover the space between those three points,
  and the batch axis, which a single published run cannot.

**The published vectors predate one line of the final standard.** FIPS 203
Algorithm 13 expands `(ρ, σ) ← G(d ‖ k)`; the draft expanded `G(d)`, and CCTV's
files still do — their `ρ` is `SHA3-512(d)[:32]` for all three parameter sets.
Everything downstream of `ρ` and `σ` is current, which is why `_key_pair` is
entered directly here and `key_gen`'s own test pins the expansion against
`hashlib`. Feeding CCTV's `d` to `key_gen` and expecting CCTV's `ek` is the trap
this arrangement exists to avoid: it fails, and the standard-conforming code is
what would look wrong.
"""

from __future__ import annotations

import hashlib

import frx
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.ml_kem import k_pke
from enc_frx.ml_kem.encoding import decode_vector
from enc_frx.ml_kem.ntt import as_ints
from enc_frx.ml_kem.params import (
    POLY_BYTES,
    SEED_SIZE,
    ciphertext_size,
    encapsulation_key_size,
)
from enc_frx.ml_kem.testing import cctv_vectors

# (parameter set, k, eta1, eta2, du, dv) — FIPS 203 Table 2.
_PARAMETER_SETS = (
    ("ML-KEM-512", 2, 3, 2, 10, 4),
    ("ML-KEM-768", 3, 2, 2, 10, 4),
    ("ML-KEM-1024", 4, 2, 2, 11, 5),
)

# The same rows with the name repeated as the absl case label.
_NAMED = tuple((row[0], *row) for row in _PARAMETER_SETS)

# One parameter set for the tests that are about the assembly rather than about
# the numbers, so they run once instead of three times.
_MID = dict(zip(("k", "eta1", "eta2", "du", "dv"), _PARAMETER_SETS[1][1:]))


def _bytes(value: object) -> bytes:
    return bytes(np.asarray(value).astype(np.uint8))


def _polys(packed: bytes, count: int) -> np.ndarray:
    """A CCTV polynomial value — `ByteEncode_12`'d — back to `[count, 256]`."""
    return np.asarray(decode_vector(np.frombuffer(packed, dtype=np.uint8), 12, count))


def _hex_array(vectors: cctv_vectors.Intermediate, name: str, index: int = 0) -> object:
    return np.frombuffer(vectors.hex_at(name, index), dtype=np.uint8)


def _zeros(size: int) -> np.ndarray:
    return np.zeros(size, dtype=np.uint8)


class KeyGenTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_the_key_pair_matches_the_published_vectors(
        self, parameter_set: str, k: int, eta1: int, _eta2: int, _du: int, _dv: int
    ) -> None:
        vectors = cctv_vectors.intermediate(parameter_set)
        ek, dk = k_pke._key_pair(
            _hex_array(vectors, "ρ"), _hex_array(vectors, "σ"), k=k, eta1=eta1
        )
        # `ek_PKE` is `ByteEncode_12(t̂) ‖ ρ`, so this pins `t̂` exactly as well.
        self.assertEqual(_bytes(ek), vectors.hex_at("ek"))
        # `dkPKE` appears twice in the file: first as `dkPKE = NTT(s) = …`, the
        # 384k bytes meant here, and later as an 800-byte value equal to `ek` —
        # an upstream slip for `ekPKE`. Index 0 is the one to ask for.
        self.assertEqual(_bytes(dk), vectors.hex_at("dkPKE", 0))

    @parameterized.named_parameters(*_NAMED)
    def test_the_seed_expansion_binds_the_parameter_set(
        self, parameter_set: str, k: int, eta1: int, _eta2: int, _du: int, _dv: int
    ) -> None:
        # `G(d ‖ k)` is Algorithm 13 line 1 of the final standard, and no vector
        # loaded here can see the `k` byte — see the module docstring. It is
        # pinned against `hashlib` instead, which is where SHA3-512 is
        # established for this repo anyway, and the lattice work below it is
        # already pinned by the vectors above.
        d = _hex_array(cctv_vectors.intermediate(parameter_set), "d")
        expanded = hashlib.sha3_512(bytes(np.asarray(d)) + bytes([k])).digest()
        want = k_pke._key_pair(
            np.frombuffer(expanded[:SEED_SIZE], dtype=np.uint8),
            np.frombuffer(expanded[SEED_SIZE:], dtype=np.uint8),
            k=k,
            eta1=eta1,
        )
        got = k_pke.key_gen(d, k=k, eta1=eta1)
        self.assertEqual(
            [_bytes(part) for part in got], [_bytes(part) for part in want]
        )

    def test_a_batch_of_seeds_generates_the_same_keys_one_at_a_time(self) -> None:
        # `keygen` is not on the hot path and the `Kem` seam does not batch it,
        # but nothing here is written per-entry either, so the leading axis works
        # — and an implementation that mixed entries would still round-trip.
        seeds = np.random.default_rng(0).integers(0, 256, (4, SEED_SIZE), np.uint8)
        batched = k_pke.key_gen(seeds, k=_MID["k"], eta1=_MID["eta1"])
        for row, seed in enumerate(seeds):
            solo = k_pke.key_gen(seed, k=_MID["k"], eta1=_MID["eta1"])
            for batched_part, solo_part in zip(batched, solo):
                np.testing.assert_array_equal(
                    np.asarray(batched_part)[row], np.asarray(solo_part)
                )


class EncryptTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_the_ciphertext_matches_the_published_vector(
        self, parameter_set: str, k: int, eta1: int, eta2: int, du: int, dv: int
    ) -> None:
        vectors = cctv_vectors.intermediate(parameter_set)
        # `r` names two different things in the file — the 32-byte randomness
        # first, the vector sampled from it later. Occurrence 0 is the seed.
        got = k_pke.encrypt(
            _hex_array(vectors, "ek"),
            _hex_array(vectors, "m"),
            _hex_array(vectors, "r", 0),
            k=k,
            eta1=eta1,
            eta2=eta2,
            du=du,
            dv=dv,
        )
        self.assertEqual(_bytes(got), vectors.hex_at("c"))

    def test_the_same_randomness_gives_the_same_ciphertext(self) -> None:
        # The property decapsulation's re-encryption check rests on. It cannot
        # fail today — nothing here draws randomness — which is the point: an
        # `encrypt` that started to would break `decaps` and nothing else, and
        # would break it into always returning the implicit-rejection secret.
        vectors = cctv_vectors.intermediate("ML-KEM-768")
        args = (
            _hex_array(vectors, "ek"),
            _hex_array(vectors, "m"),
            _hex_array(vectors, "r", 0),
        )
        self.assertEqual(
            _bytes(k_pke.encrypt(*args, **_MID)), _bytes(k_pke.encrypt(*args, **_MID))
        )

    def test_different_randomness_gives_a_different_ciphertext(self) -> None:
        # And `r` is actually read: an `encrypt` that ignored it would be
        # deterministic too, and every round trip would still pass.
        vectors = cctv_vectors.intermediate("ML-KEM-768")
        ek, m = _hex_array(vectors, "ek"), _hex_array(vectors, "m")
        first = k_pke.encrypt(ek, m, _zeros(SEED_SIZE), **_MID)
        second = k_pke.encrypt(ek, m, _zeros(SEED_SIZE) + 1, **_MID)
        self.assertNotEqual(_bytes(first), _bytes(second))

    def test_the_traced_ciphertext_matches_the_eager_one(self) -> None:
        vectors = cctv_vectors.intermediate("ML-KEM-768")
        args = (
            _hex_array(vectors, "ek"),
            _hex_array(vectors, "m"),
            _hex_array(vectors, "r", 0),
        )
        traced = frx.jit(lambda ek, m, r: k_pke.encrypt(ek, m, r, **_MID))
        self.assertEqual(_bytes(traced(*args)), _bytes(k_pke.encrypt(*args, **_MID)))


class DecryptTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_the_message_matches_the_published_vector(
        self, parameter_set: str, k: int, _eta1: int, _eta2: int, du: int, dv: int
    ) -> None:
        vectors = cctv_vectors.intermediate(parameter_set)
        got = k_pke.decrypt(
            _hex_array(vectors, "dkPKE", 0), _hex_array(vectors, "c"), k=k, du=du, dv=dv
        )
        self.assertEqual(_bytes(got), vectors.hex_at("m"))

    @parameterized.named_parameters(*_NAMED)
    def test_the_noisy_message_matches_before_the_rounding(
        self, parameter_set: str, k: int, _eta1: int, _eta2: int, du: int, dv: int
    ) -> None:
        # `Compress_1` hides its own input, so `w` is compared before it runs —
        # see `_noisy_message` for the sign error the rounding would forgive on
        # every ciphertext ever generated.
        vectors = cctv_vectors.intermediate(parameter_set)
        got = k_pke._noisy_message(
            _hex_array(vectors, "dkPKE", 0), _hex_array(vectors, "c"), k=k, du=du, dv=dv
        )
        np.testing.assert_array_equal(
            np.asarray(as_ints(got)), _polys(vectors.hex_at("w"), 1)[0]
        )


class RoundTripTest(parameterized.TestCase):
    """What the three published points cannot cover: the space between them."""

    @parameterized.named_parameters(*_NAMED)
    def test_decrypt_recovers_the_message_across_the_batch(
        self, parameter_set: str, k: int, eta1: int, eta2: int, du: int, dv: int
    ) -> None:
        rng = np.random.default_rng(len(parameter_set))
        batch = 4
        ek, dk = k_pke.key_gen(
            rng.integers(0, 256, (batch, SEED_SIZE), np.uint8), k=k, eta1=eta1
        )
        m = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        r = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        c = k_pke.encrypt(ek, m, r, k=k, eta1=eta1, eta2=eta2, du=du, dv=dv)
        got = k_pke.decrypt(dk, c, k=k, du=du, dv=dv)
        np.testing.assert_array_equal(np.asarray(got), m)

    def test_each_batch_entry_is_independent(self) -> None:
        # A batched call has to agree with the solo calls entry by entry. An
        # implementation that leaked across the batch axis — a reduction over the
        # wrong axis, a broadcast that should have been an index — round-trips
        # perfectly, because both directions leak the same way.
        rng = np.random.default_rng(7)
        batch = 3
        ek, _ = k_pke.key_gen(
            rng.integers(0, 256, (batch, SEED_SIZE), np.uint8),
            k=_MID["k"],
            eta1=_MID["eta1"],
        )
        m = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        r = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        got = np.asarray(k_pke.encrypt(ek, m, r, **_MID))
        for row in range(batch):
            solo = k_pke.encrypt(np.asarray(ek)[row], m[row], r[row], **_MID)
            np.testing.assert_array_equal(got[row], np.asarray(solo))


class LengthCheckTest(absltest.TestCase):
    """FIPS 203 §7.2/§7.3's type check, which is static and so may raise.

    A coefficient's range is data and comes back as a value
    (`encoding.coefficients_are_reduced`); a byte string's length is fixed at
    trace time, so a wrong one is a programming error rather than a malformed
    input to route through a rejection path.
    """

    _EK = encapsulation_key_size(_MID["k"])
    _DK = POLY_BYTES * _MID["k"]
    _C = ciphertext_size(_MID["k"], _MID["du"], _MID["dv"])

    def test_rejects_a_short_encapsulation_key(self) -> None:
        with self.assertRaises(ValueError):
            k_pke.encrypt(
                _zeros(self._EK - 1), _zeros(SEED_SIZE), _zeros(SEED_SIZE), **_MID
            )

    def test_rejects_a_message_that_is_not_32_bytes(self) -> None:
        with self.assertRaises(ValueError):
            k_pke.encrypt(
                _zeros(self._EK), _zeros(SEED_SIZE + 1), _zeros(SEED_SIZE), **_MID
            )

    def test_rejects_randomness_that_is_not_32_bytes(self) -> None:
        with self.assertRaises(ValueError):
            k_pke.encrypt(
                _zeros(self._EK), _zeros(SEED_SIZE), _zeros(SEED_SIZE - 1), **_MID
            )

    def test_rejects_a_short_key_generation_seed(self) -> None:
        with self.assertRaises(ValueError):
            k_pke.key_gen(_zeros(SEED_SIZE - 1), k=_MID["k"], eta1=_MID["eta1"])

    def test_rejects_a_short_decryption_key(self) -> None:
        with self.assertRaises(ValueError):
            k_pke.decrypt(
                _zeros(self._DK - 1),
                _zeros(self._C),
                k=_MID["k"],
                du=_MID["du"],
                dv=_MID["dv"],
            )

    def test_rejects_a_short_ciphertext(self) -> None:
        with self.assertRaises(ValueError):
            k_pke.decrypt(
                _zeros(self._DK),
                _zeros(self._C - 1),
                k=_MID["k"],
                du=_MID["du"],
                dv=_MID["dv"],
            )


if __name__ == "__main__":
    absltest.main()
