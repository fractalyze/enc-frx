# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""K-PKE against FIPS 203's own intermediate values, and around the loop.

Two gates that fail on different things:

- **CCTV's `intermediate/` files** publish `ek_PKE`, `dk_PKE`, the ciphertext and
  the recovered message for one run of each parameter set, so every stage is
  compared against a value someone else computed.
- **Round trips over random input** cover the space between those three points,
  and the batch axis, which a single published run cannot.

**The published vectors predate one line of the final standard** — they expand
`(ρ, σ) ← G(d)` where FIPS 203 Algorithm 13 line 1 is `G(d ‖ k)` — so they can
only enter key generation below that expansion, at `_key_pair`. The expansion
itself is gated by ACVP, through the whole of `MlKem.keygen`
([`acvp_test.py`](acvp_test.py)). The full account, and the wrong repair it
exists to prevent, is in
[`docs/schemes/ml-kem.md`](../../../docs/schemes/ml-kem.md).
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.ml_kem import _k_pke
from enc_frx.ml_kem.ntt import as_ints
from enc_frx.ml_kem.params import (
    SEED_SIZE,
    ciphertext_size,
    decryption_key_size,
    encapsulation_key_size,
)
from enc_frx.ml_kem.testing import cctv_vectors
from enc_frx.testing.kat import to_bytes

# (parameter set, k, eta1, eta2, du, dv) — FIPS 203 Table 2.
_PARAMETER_SETS = (
    ("ML-KEM-512", 2, 3, 2, 10, 4),
    ("ML-KEM-768", 3, 2, 2, 10, 4),
    ("ML-KEM-1024", 4, 2, 2, 11, 5),
)

# The same rows with the name repeated as the absl case label.
_NAMED = tuple((row[0], *row) for row in _PARAMETER_SETS)

# ML-KEM-768, for the tests that are about the assembly rather than about the
# numbers, so they run once instead of three times. `decrypt` and `key_gen` take
# subsets, and spelling those out per call site is what buries the argument that
# matters under four that do not.
_768 = dict(zip(("k", "eta1", "eta2", "du", "dv"), _PARAMETER_SETS[1][1:], strict=True))
_768_KEY_GEN = {name: _768[name] for name in ("k", "eta1")}
_768_DECRYPT = {name: _768[name] for name in ("k", "du", "dv")}


def _zeros(size: int) -> np.ndarray:
    return np.zeros(size, dtype=np.uint8)


def _encrypt_inputs(parameter_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(ek, m, r)` as published.

    `r` names two things in the file — the 32-byte randomness first, the vector
    sampled from it later — so occurrence 0 is the one `encrypt` takes.
    """
    vectors = cctv_vectors.intermediate(parameter_set)
    return (
        vectors.array_at("ek"),
        vectors.array_at("m"),
        vectors.array_at("r", 0),
    )


class KeyGenTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_the_key_pair_matches_the_published_vectors(
        self, parameter_set: str, k: int, eta1: int, _eta2: int, _du: int, _dv: int
    ) -> None:
        vectors = cctv_vectors.intermediate(parameter_set)
        ek, dk = _k_pke._key_pair(
            vectors.array_at("ρ"), vectors.array_at("σ"), k=k, eta1=eta1
        )
        # `ek_PKE` is `ByteEncode_12(t̂) ‖ ρ`, so this pins `t̂` exactly as well.
        self.assertEqual(to_bytes(ek), vectors.hex_at("ek"))
        # `dkPKE` appears twice in the file: first as `dkPKE = NTT(s) = …`, the
        # 384k bytes meant here, and later as an 800-byte value equal to `ek` —
        # an upstream slip for `ekPKE`. Index 0 is the one to ask for.
        self.assertEqual(to_bytes(dk), vectors.hex_at("dkPKE", 0))

    def test_a_batch_of_seeds_generates_the_same_keys_one_at_a_time(self) -> None:
        # `keygen` is not on the hot path and the `Kem` seam does not batch it,
        # but nothing here is written per-entry either, so the leading axis works
        # — and an implementation that mixed entries would still round-trip.
        seeds = np.random.default_rng(0).integers(0, 256, (4, SEED_SIZE), np.uint8)
        batched = _k_pke.key_gen(seeds, **_768_KEY_GEN)
        for row, seed in enumerate(seeds):
            solo = _k_pke.key_gen(seed, **_768_KEY_GEN)
            for batched_part, solo_part in zip(batched, solo):
                np.testing.assert_array_equal(
                    np.asarray(batched_part)[row], np.asarray(solo_part)
                )


class EncryptTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_the_ciphertext_matches_the_published_vector(
        self, parameter_set: str, k: int, eta1: int, eta2: int, du: int, dv: int
    ) -> None:
        got = _k_pke.encrypt(
            *_encrypt_inputs(parameter_set),
            k=k,
            eta1=eta1,
            eta2=eta2,
            du=du,
            dv=dv,
        )
        self.assertEqual(
            to_bytes(got), cctv_vectors.intermediate(parameter_set).hex_at("c")
        )

    def test_the_same_randomness_gives_the_same_ciphertext(self) -> None:
        # The property decapsulation's re-encryption check rests on, and it
        # cannot fail today — which is the point. See `_k_pke`'s module docstring
        # for what an `encrypt` that drew its own randomness would break.
        args = _encrypt_inputs("ML-KEM-768")
        self.assertEqual(
            to_bytes(_k_pke.encrypt(*args, **_768)),
            to_bytes(_k_pke.encrypt(*args, **_768)),
        )

    def test_different_randomness_gives_a_different_ciphertext(self) -> None:
        # And `r` is actually read: an `encrypt` that ignored it would be
        # deterministic too, and every round trip would still pass.
        ek, m, _ = _encrypt_inputs("ML-KEM-768")
        first = _k_pke.encrypt(ek, m, _zeros(SEED_SIZE), **_768)
        second = _k_pke.encrypt(ek, m, _zeros(SEED_SIZE) + 1, **_768)
        self.assertNotEqual(to_bytes(first), to_bytes(second))

    def test_the_traced_ciphertext_matches_the_eager_one(self) -> None:
        args = _encrypt_inputs("ML-KEM-768")
        traced = frx.jit(lambda ek, m, r: _k_pke.encrypt(ek, m, r, **_768))
        self.assertEqual(
            to_bytes(traced(*args)), to_bytes(_k_pke.encrypt(*args, **_768))
        )


class DecryptTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_the_message_matches_the_published_vector(
        self, parameter_set: str, k: int, _eta1: int, _eta2: int, du: int, dv: int
    ) -> None:
        vectors = cctv_vectors.intermediate(parameter_set)
        got = _k_pke.decrypt(
            vectors.array_at("dkPKE", 0), vectors.array_at("c"), k=k, du=du, dv=dv
        )
        self.assertEqual(to_bytes(got), vectors.hex_at("m"))

    @parameterized.named_parameters(*_NAMED)
    def test_the_noisy_message_matches_before_the_rounding(
        self, parameter_set: str, k: int, _eta1: int, _eta2: int, du: int, dv: int
    ) -> None:
        # `Compress_1` hides its own input, so `w` is compared before it runs —
        # see `_noisy_message` for the sign error the rounding would forgive on
        # every ciphertext ever generated.
        vectors = cctv_vectors.intermediate(parameter_set)
        got = _k_pke._noisy_message(
            vectors.array_at("dkPKE", 0), vectors.array_at("c"), k=k, du=du, dv=dv
        )
        np.testing.assert_array_equal(
            np.asarray(as_ints(got)),
            cctv_vectors.decode_polys(vectors.hex_at("w"), 1)[0],
        )


class RoundTripTest(parameterized.TestCase):
    """What the three published points cannot cover: the space between them."""

    @parameterized.named_parameters(*_NAMED)
    def test_decrypt_recovers_the_message_across_the_batch(
        self, parameter_set: str, k: int, eta1: int, eta2: int, du: int, dv: int
    ) -> None:
        rng = np.random.default_rng(len(parameter_set))
        batch = 4
        ek, dk = _k_pke.key_gen(
            rng.integers(0, 256, (batch, SEED_SIZE), np.uint8), k=k, eta1=eta1
        )
        m = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        r = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        c = _k_pke.encrypt(ek, m, r, k=k, eta1=eta1, eta2=eta2, du=du, dv=dv)
        got = _k_pke.decrypt(dk, c, k=k, du=du, dv=dv)
        np.testing.assert_array_equal(np.asarray(got), m)

    def test_each_batch_entry_is_independent(self) -> None:
        # A batched call has to agree with the solo calls entry by entry. An
        # implementation that leaked across the batch axis — a reduction over the
        # wrong axis, a broadcast that should have been an index — round-trips
        # perfectly, because both directions leak the same way.
        rng = np.random.default_rng(7)
        # The same batch the tests above use, so this reuses their compiled
        # shapes rather than forcing a whole second set for one more entry.
        batch = 4
        ek, _ = _k_pke.key_gen(
            rng.integers(0, 256, (batch, SEED_SIZE), np.uint8),
            **_768_KEY_GEN,
        )
        m = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        r = rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8)
        got = np.asarray(_k_pke.encrypt(ek, m, r, **_768))
        for row in range(batch):
            solo = _k_pke.encrypt(np.asarray(ek)[row], m[row], r[row], **_768)
            np.testing.assert_array_equal(got[row], np.asarray(solo))


class LengthCheckTest(absltest.TestCase):
    """FIPS 203 §7.2/§7.3's type check, which is static and so may raise.

    A coefficient's range is data and comes back as a value
    (`encoding.coefficients_are_reduced`); a byte string's length is fixed at
    trace time, so a wrong one is a programming error rather than a malformed
    input to route through a rejection path.
    """

    _EK = encapsulation_key_size(_768["k"])
    _DK = decryption_key_size(_768["k"])
    _C = ciphertext_size(_768["k"], _768["du"], _768["dv"])

    def test_rejects_a_short_encapsulation_key(self) -> None:
        with self.assertRaises(ValueError):
            _k_pke.encrypt(
                _zeros(self._EK - 1), _zeros(SEED_SIZE), _zeros(SEED_SIZE), **_768
            )

    def test_rejects_a_message_that_is_not_32_bytes(self) -> None:
        with self.assertRaises(ValueError):
            _k_pke.encrypt(
                _zeros(self._EK), _zeros(SEED_SIZE + 1), _zeros(SEED_SIZE), **_768
            )

    def test_rejects_randomness_that_is_not_32_bytes(self) -> None:
        with self.assertRaises(ValueError):
            _k_pke.encrypt(
                _zeros(self._EK), _zeros(SEED_SIZE), _zeros(SEED_SIZE - 1), **_768
            )

    def test_rejects_a_short_key_generation_seed(self) -> None:
        with self.assertRaises(ValueError):
            _k_pke.key_gen(_zeros(SEED_SIZE - 1), **_768_KEY_GEN)

    def test_rejects_a_short_decryption_key(self) -> None:
        with self.assertRaises(ValueError):
            _k_pke.decrypt(
                _zeros(self._DK - 1),
                _zeros(self._C),
                **_768_DECRYPT,
            )

    def test_rejects_a_short_ciphertext(self) -> None:
        with self.assertRaises(ValueError):
            _k_pke.decrypt(
                _zeros(self._DK),
                _zeros(self._C - 1),
                **_768_DECRYPT,
            )


if __name__ == "__main__":
    absltest.main()
