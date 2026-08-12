# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM's encoding against FIPS 203's algorithms, and against its lossiness.

The packing is checked against the reference. Compression is checked
**exhaustively** — all 3329 inputs for every width — because it is the one step
whose failure is silent: it discards bits by design, so a rounding mistake shifts
the decryption failure probability rather than breaking a round trip, and every
round-trip test still passes. Sampling would not find a tie-breaking error, which
is the specific bug this arithmetic exists to avoid.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest

from enc_frx.ml_kem import encoding
from enc_frx.ml_kem.testing import fips203_reference as ref

# Seeds are per-test-class only so a failure names one; nothing needs
# independent streams.
_SEED = 20260812


def _malformed_bytes(f: list[int]) -> np.ndarray:
    """Encode coefficients the field does not contain.

    Neither encoder reduces — `byte_encode` is shift-and-mask, and FIPS 203
    Algorithm 5 has no mod-q either — so the reference produces these bytes
    happily. It is used rather than `encoding.byte_encode` only so the malformed
    input is not built by the module under test.
    """
    return np.array(ref.byte_encode(f, 12), dtype=np.uint8)


class CompressTest(absltest.TestCase):
    def test_matches_the_integer_rounding_on_every_input(self) -> None:
        """All q inputs per width — the whole domain, not a sample."""
        xs = np.arange(ref.Q, dtype=np.int64)
        for d in encoding.WIDTHS:
            got = np.asarray(encoding.compress(xs, d)).tolist()
            self.assertEqual(got, [ref.compress(int(x), d) for x in xs], f"d={d}")

    def test_decompress_matches_on_every_input(self) -> None:
        for d in encoding.WIDTHS:
            ys = np.arange(1 << d, dtype=np.int64)
            got = np.asarray(encoding.decompress(ys, d)).tolist()
            self.assertEqual(got, [ref.decompress(int(y), d) for y in ys], f"d={d}")

    def test_compression_is_lossy_but_bounded(self) -> None:
        """Decompress(Compress(x)) is not x, and that is the point.

        Stated as a test so nobody 'fixes' the round trip: the error is bounded
        by the standard's own bound, and being *exactly* recoverable at d < 12
        would mean the compression was not compressing.
        """
        xs = np.arange(ref.Q, dtype=np.int64)
        for d in (4, 5, 10, 11):
            back = np.asarray(encoding.decompress(encoding.compress(xs, d), d))
            err = np.minimum((back - xs) % ref.Q, (xs - back) % ref.Q)
            self.assertLessEqual(int(err.max()), ref.Q // (1 << (d + 1)) + 1)
            self.assertGreater(int(err.max()), 0, f"d={d} was lossless")


class ByteEncodeTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rng = np.random.default_rng(_SEED)

    def _poly(self, d: int) -> list[int]:
        hi = ref.Q if d == 12 else (1 << d)
        return self.rng.integers(0, hi, size=ref.N, dtype=np.int64).tolist()

    def test_matches_fips203_byte_encode(self) -> None:
        for d in encoding.WIDTHS:
            f = self._poly(d)
            got = np.asarray(encoding.byte_encode(f, d)).tolist()
            self.assertEqual(got, ref.byte_encode(f, d), f"d={d}")

    def test_matches_fips203_byte_decode(self) -> None:
        for d in encoding.WIDTHS:
            b = self.rng.integers(0, 256, size=32 * d, dtype=np.int64).tolist()
            got = np.asarray(encoding.byte_decode(b, d)).tolist()
            self.assertEqual(got, ref.byte_decode(b, d), f"d={d}")

    def test_round_trips_at_every_width(self) -> None:
        for d in encoding.WIDTHS:
            f = self._poly(d)
            back = encoding.byte_decode(encoding.byte_encode(f, d), d)
            self.assertEqual(np.asarray(back).tolist(), f, f"d={d}")

    def test_batches_over_leading_axes(self) -> None:
        polys = [[self._poly(12) for _ in range(3)] for _ in range(2)]
        got = np.asarray(encoding.byte_encode(polys, 12))
        self.assertEqual(got.shape, (2, 3, 384))
        for i in range(2):
            for j in range(3):
                self.assertEqual(got[i][j].tolist(), ref.byte_encode(polys[i][j], 12))


class ModulusCheckTest(absltest.TestCase):
    """FIPS 203 §7.2's normative check, which is the reason d=12 reduces mod q."""

    def setUp(self) -> None:
        super().setUp()
        self.rng = np.random.default_rng(_SEED)

    def test_rejects_a_coefficient_at_or_above_q(self) -> None:
        """3329 and 4095 are representable in 12 bits and are not field elements.

        This is the attack surface the check exists for: without it they decode
        to 0 and 766 silently.
        """
        for bad in (ref.Q, 4095):
            f = [bad] + [0] * (ref.N - 1)
            b = _malformed_bytes(f)
            self.assertFalse(bool(encoding.coefficients_are_reduced(b)), f"{bad}")

    def test_is_per_entry_over_a_batch(self) -> None:
        """A batch cannot raise on entry 1, so validity is a value."""
        good = self.rng.integers(0, ref.Q, size=ref.N, dtype=np.int64)
        ok = np.asarray(encoding.byte_encode(good.tolist(), 12))
        bad = np.asarray(_malformed_bytes([ref.Q] + [0] * (ref.N - 1)))
        got = encoding.coefficients_are_reduced(np.stack([ok, bad, ok]))
        self.assertEqual(np.asarray(got).tolist(), [True, False, True])


class WireFormatTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rng = np.random.default_rng(_SEED)

    def test_encapsulation_key_round_trips_and_has_the_standard_length(self) -> None:
        for k in (2, 3, 4):
            t_hat = self.rng.integers(0, ref.Q, size=(k, ref.N), dtype=np.int64)
            rho = self.rng.integers(0, 256, size=32, dtype=np.int64)
            ek = encoding.encode_ek(t_hat, rho)
            self.assertEqual(ek.shape[-1], 384 * k + 32, f"k={k}")
            back_t, back_rho = encoding.decode_ek(ek, k)
            self.assertEqual(np.asarray(back_t).tolist(), t_hat.tolist())
            self.assertEqual(np.asarray(back_rho).tolist(), rho.tolist())

    def test_k_pke_decryption_key_round_trips_at_every_k(self) -> None:
        """Lossless, unlike the ciphertext below: `ŝ` is stored uncompressed."""
        for k in (2, 3, 4):
            s_hat = self.rng.integers(0, ref.Q, size=(k, ref.N), dtype=np.int64)
            dk_pke = encoding.encode_dk_pke(s_hat)
            self.assertEqual(dk_pke.shape[-1], 384 * k, f"k={k}")
            back = encoding.decode_dk_pke(dk_pke, k)
            self.assertEqual(np.asarray(back).tolist(), s_hat.tolist())

    def test_ciphertext_length_matches_the_parameter_set(self) -> None:
        """(k, du, dv) for ML-KEM-512 / -768 / -1024."""
        for k, du, dv in ((2, 10, 4), (3, 10, 4), (4, 11, 5)):
            u = self.rng.integers(0, ref.Q, size=(k, ref.N), dtype=np.int64)
            v = self.rng.integers(0, ref.Q, size=ref.N, dtype=np.int64)
            c = encoding.encode_ciphertext(u, v, du, dv)
            self.assertEqual(c.shape[-1], 32 * (du * k + dv), f"k={k}")

    def test_ciphertext_round_trip_is_lossy_by_design(self) -> None:
        """Decoding recovers the *compressed* value, not the original.

        Asserting equality here would be asserting the ciphertext is not
        compressed, so the check is that decode(encode(x)) equals what
        decompress(compress(x)) gives — the same lossy map, applied once.
        """
        k, du, dv = 3, 10, 4
        u = self.rng.integers(0, ref.Q, size=(k, ref.N), dtype=np.int64)
        v = self.rng.integers(0, ref.Q, size=ref.N, dtype=np.int64)
        got_u, got_v = encoding.decode_ciphertext(
            encoding.encode_ciphertext(u, v, du, dv), k, du, dv
        )
        self.assertEqual(
            np.asarray(got_u).tolist(),
            np.asarray(encoding.decompress(encoding.compress(u, du), du)).tolist(),
        )
        self.assertEqual(
            np.asarray(got_v).tolist(),
            np.asarray(encoding.decompress(encoding.compress(v, dv), dv)).tolist(),
        )

    def test_decapsulation_key_splits_at_the_standard_offsets(self) -> None:
        k = 3
        dk_pke = self.rng.integers(0, 256, size=384 * k, dtype=np.int64)
        ek = self.rng.integers(0, 256, size=384 * k + 32, dtype=np.int64)
        h_ek = self.rng.integers(0, 256, size=32, dtype=np.int64)
        z = self.rng.integers(0, 256, size=32, dtype=np.int64)
        dk = encoding.encode_dk(dk_pke, ek, h_ek, z)
        self.assertEqual(dk.shape[-1], 768 * k + 96)
        parts = encoding.decode_dk(dk, k)
        for got, want in zip(parts, (dk_pke, ek, h_ek, z)):
            self.assertEqual(np.asarray(got).tolist(), want.tolist())

    def test_rejects_a_wrong_length(self) -> None:
        """The type check of §7.2/§7.3 — an exception, not a value.

        A length is static in a traced program, so it cannot be a per-entry
        result the way the modulus check is. Untested, a `dk` short by 40 bytes
        decoded to a zero-length `z`, which is the implicit-rejection seed: the
        rejection secret would have been derived from nothing.
        """
        k, du, dv = 3, 10, 4
        with self.assertRaisesRegex(ValueError, "encapsulation key"):
            encoding.decode_ek(
                np.zeros(encoding.encapsulation_key_size(k) + 32, dtype=np.uint8), k
            )
        with self.assertRaisesRegex(ValueError, "decapsulation key"):
            encoding.decode_dk(
                np.zeros(encoding.decapsulation_key_size(k) - 40, dtype=np.uint8), k
            )
        with self.assertRaisesRegex(ValueError, "K-PKE decryption key"):
            encoding.decode_dk_pke(
                np.zeros(encoding.decryption_key_size(k) - 1, dtype=np.uint8), k
            )
        with self.assertRaisesRegex(ValueError, "ciphertext"):
            encoding.decode_ciphertext(
                np.zeros(encoding.ciphertext_size(k, du, dv) - 1, dtype=np.uint8),
                k,
                du,
                dv,
            )

    def test_jits(self) -> None:
        f = self.rng.integers(0, ref.Q, size=ref.N, dtype=np.int64)
        jitted = frx.jit(lambda x: encoding.byte_encode(x, 12))(f)
        self.assertEqual(
            np.asarray(jitted).tolist(),
            np.asarray(encoding.byte_encode(f, 12)).tolist(),
        )


if __name__ == "__main__":
    absltest.main()
