# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""DHKEM(X25519, HKDF-SHA256) against RFC 9180's published vectors.

The corpus is the machine-readable form of the RFC's Appendix A, sha256-pinned
in //MODULE.bazel. Every suite gates the KEM step three ways — `DeriveKeyPair`
from both published ikms, `Encap` against `(pkRm, ikmE) -> (enc,
shared_secret)`, `Decap` against `(skRm, enc) -> shared_secret` — and
`check_kem` adds what the file cannot express: that a tampered `enc` yields a
*different* secret rather than a failure, that it does so deterministically,
and that the answer is decided per batch entry.

The auth-mode cases ride along refused rather than dropped: `AuthEncap` mixes
the sender's static key into the DH step, so running base `Encap` over an auth
suite's inputs would reproduce `enc` and silently miss the shared secret. The
loader records the sender fields as unsupported, and the driver's refusal is
asserted here so a future "filter fix" that runs them wrongly has a test to
break.
"""

from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest
from python.runfiles import runfiles

from enc_frx.kem import Kem
from enc_frx.testing.kat import KatError, check_kem, load_hpke_kem
from enc_frx.x25519.dhkem import DhKemX25519

_KEM_ID = 0x0020  # RFC 9180 §7.1, Table 2: DHKEM(X25519, HKDF-SHA256)
_SEAM_FUNCTIONS = frozenset({"keygen", "encapsulation", "decapsulation"})


def _vectors() -> list:
    location = runfiles.Create().Rlocation("hpke_test_vectors/file/test-vectors.json")
    assert location is not None, "hpke_test_vectors not in runfiles"
    return load_hpke_kem(location, kem_id=_KEM_ID)


class DhKemRfc9180Test(absltest.TestCase):
    def test_rfc9180_vectors(self) -> None:
        vectors = [v for v in _vectors() if v.function in _SEAM_FUNCTIONS]
        # Both base modes, both published kdf/aead spreads: the corpus carries
        # enough cases that an empty filter result would be a loader bug.
        self.assertGreaterEqual(
            len([v for v in vectors if v.function == "decapsulation"]), 8
        )
        check_kem(DhKemX25519(), vectors)

    def test_auth_mode_cases_are_refused_not_dropped(self) -> None:
        vectors = _vectors()
        self.assertNotEmpty([v for v in vectors if v.function == "authEncap"])
        self.assertNotEmpty([v for v in vectors if v.function == "authDecap"])
        with self.assertRaisesRegex(KatError, "cannot express"):
            check_kem(DhKemX25519(), vectors)

    def test_conforms_to_the_seam(self) -> None:
        scheme = DhKemX25519()
        self.assertIsInstance(scheme, Kem)
        # Parameterless, so equality is by type — the pytree-aux requirement
        # (`enc_frx/kem.py`): identity equality would silently re-trace an
        # enclosing jit zone for every freshly built instance.
        self.assertEqual(scheme, DhKemX25519())
        self.assertEqual(hash(scheme), hash(DhKemX25519()))

    def test_traced_matches_eager(self) -> None:
        scheme = DhKemX25519()
        rng = np.random.default_rng(0)
        randomness = rng.integers(0, 256, (3, scheme.randomness_size), dtype=np.uint8)
        seeds = rng.integers(0, 256, (3, scheme.seed_size), dtype=np.uint8)
        encaps_keys, decaps_keys = frx.vmap(scheme.keygen)(seeds)

        eager_ct, eager_ss = scheme.encaps(encaps_keys, randomness=randomness)
        jit_ct, jit_ss = frx.jit(scheme.encaps)(encaps_keys, randomness=randomness)
        np.testing.assert_array_equal(np.asarray(jit_ct), np.asarray(eager_ct))
        np.testing.assert_array_equal(np.asarray(jit_ss), np.asarray(eager_ss))

        eager = scheme.decaps(decaps_keys, eager_ct)
        traced = frx.jit(scheme.decaps)(decaps_keys, eager_ct)
        np.testing.assert_array_equal(np.asarray(traced), np.asarray(eager))
        np.testing.assert_array_equal(np.asarray(eager), np.asarray(eager_ss))


if __name__ == "__main__":
    absltest.main()
