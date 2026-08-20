# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""X25519 — gated on the published RFC 7748 vectors (§5.2 singles, the
iterated ladder, the §6.1 Diffie-Hellman exchange), with batch shapes checked
against the plain-integer reference.

The iterated vector is the one that catches accumulated drift: a wrong carry
that survives one ladder is amplified through a thousand. The RFC's
million-iteration row is deliberately absent — it adds hours, not coverage,
over the thousand-iteration row.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest

from enc_frx.x25519 import x25519 as x
from enc_frx.x25519.testing import rfc7748_reference as ref

# RFC 7748 §5.2, both single-shot vectors: (scalar, u, output).
_VECTORS = (
    (
        "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
        "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c",
        "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552",
    ),
    (
        "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
        "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493",
        "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957",
    ),
)

# RFC 7748 §6.1: Alice and Bob's secrets, public keys, and shared secret.
_ALICE_SK = "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"
_ALICE_PK = "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a"
_BOB_SK = "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb"
_BOB_PK = "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"
_SHARED = "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"


def _arr(hex_string: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8)[None, :]


class X25519Test(absltest.TestCase):
    def test_rfc7748_section_5_2_vectors(self) -> None:
        # Both vectors as one batch of two — the vector check and a batch
        # check in the same call.
        scalars = np.concatenate([_arr(s) for (s, _, _) in _VECTORS])
        coords = np.concatenate([_arr(u) for (_, u, _) in _VECTORS])
        got = np.asarray(x.x25519(scalars, coords))
        for i, (_, _, out) in enumerate(_VECTORS):
            self.assertEqual(bytes(got[i]).hex(), out)

    def test_rfc7748_iterated_ladder(self) -> None:
        # §5.2's iterated vector: k, u start at the basepoint and feed back.
        k = np.asarray(x.basepoint())
        u = k.copy()
        expected = {
            1: "422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079",
            1000: "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51",
        }
        step = frx.jit(x.x25519)
        for iteration in range(1, 1001):
            k, u = np.asarray(step(k, u)), k
            if iteration in expected:
                self.assertEqual(bytes(k[0]).hex(), expected[iteration])

    def test_rfc7748_section_6_1_diffie_hellman(self) -> None:
        # Alice and Bob ride one batch of two; the shared secret must agree
        # in both directions and with the published value.
        secrets = np.concatenate([_arr(_ALICE_SK), _arr(_BOB_SK)])
        publics = np.asarray(x.public_key(secrets))
        self.assertEqual(bytes(publics[0]).hex(), _ALICE_PK)
        self.assertEqual(bytes(publics[1]).hex(), _BOB_PK)
        shared = np.asarray(x.x25519(secrets, publics[::-1]))
        self.assertEqual(bytes(shared[0]).hex(), _SHARED)
        self.assertEqual(bytes(shared[1]).hex(), _SHARED)

    def test_batch_matches_reference_per_row(self) -> None:
        rng = np.random.default_rng(0)
        scalars = rng.integers(0, 256, size=(8, 32), dtype=np.uint8)
        coords = rng.integers(0, 256, size=(8, 32), dtype=np.uint8)
        got = np.asarray(x.x25519(scalars, coords))
        for i in range(scalars.shape[0]):
            want = ref.x25519(bytes(scalars[i]), bytes(coords[i]))
            self.assertEqual(bytes(got[i]), want)

    def test_traced_matches_eager(self) -> None:
        rng = np.random.default_rng(1)
        scalars = rng.integers(0, 256, size=(3, 32), dtype=np.uint8)
        coords = rng.integers(0, 256, size=(3, 32), dtype=np.uint8)
        eager = np.asarray(x.x25519(scalars, coords))
        traced = np.asarray(frx.jit(x.x25519)(scalars, coords))
        np.testing.assert_array_equal(traced, eager)


if __name__ == "__main__":
    absltest.main()
