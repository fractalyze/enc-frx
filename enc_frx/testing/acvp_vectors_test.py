# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The ACVP loader against NIST's real files, not a fixture.

A loader tested on a fixture tests the fixture. These run it over the published
ML-KEM sets, fetched and sha256-pinned in //MODULE.bazel, and assert the
structure the harness depends on — which is not what a reading of the ACVP
documentation would predict:

- `encapDecap` publishes **four** functions, not two. Beyond encapsulation and
  decapsulation it carries `encapsulationKeyCheck` and `decapsulationKeyCheck`,
  the FIPS 203 §7.2/§7.3 input validation, which the `Kem` seam does not name.
- The decapsulation cases carry **no pass/fail marker**. Only the two key-check
  functions publish `testPassed`. That is correct rather than missing: under
  implicit rejection a modified ciphertext yields a shared secret, so there is no
  verdict to publish — which is why the harness gates rejection by comparing `k`
  exactly and by requiring a run to reach decapsulation at all.

If a future ACVP regeneration changes either, these fail and the harness's design
gets revisited rather than silently under-testing.
"""

from __future__ import annotations

from collections import Counter

from absl.testing import absltest
from python.runfiles import runfiles

from enc_frx.testing.kat import KatError, load_acvp_ml_kem

_KEYGEN = ("acvp_ml_kem_keygen_prompt", "acvp_ml_kem_keygen_expected")
_ENCAPDECAP = ("acvp_ml_kem_encapdecap_prompt", "acvp_ml_kem_encapdecap_expected")

_PARAMETER_SETS = frozenset({"ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"})


def _path(repo: str, name: str) -> str:
    location = runfiles.Create().Rlocation(f"{repo}/file/{name}")
    assert location is not None, f"{repo} not in runfiles"
    return location


def _load(pair: tuple[str, str]) -> list:
    prompt, expected = pair
    return load_acvp_ml_kem(
        _path(prompt, "prompt.json"), _path(expected, "expectedResults.json")
    )


class AcvpKeyGenTest(absltest.TestCase):
    def test_every_case_carries_a_seed_and_both_keys(self) -> None:
        vectors = _load(_KEYGEN)
        self.assertNotEmpty(vectors)
        for vector in vectors:
            self.assertEqual(vector.function, "keygen")
            self.assertIsNotNone(vector.seed, vector.case_id)
            self.assertIsNotNone(vector.encapsulation_key, vector.case_id)
            self.assertIsNotNone(vector.decapsulation_key, vector.case_id)

    def test_the_seed_is_d_and_z_concatenated(self) -> None:
        # FIPS 203 §7.1 takes 32 bytes of `d` and 32 of `z`; the seam takes one
        # `seed`, so the loader joins them in the standard's order.
        for vector in _load(_KEYGEN):
            self.assertLen(vector.seed, 64, vector.case_id)

    def test_all_three_parameter_sets_are_present(self) -> None:
        self.assertEqual({v.parameter_set for v in _load(_KEYGEN)}, _PARAMETER_SETS)


class AcvpEncapDecapTest(absltest.TestCase):
    def test_the_set_carries_four_functions(self) -> None:
        functions = Counter(v.function for v in _load(_ENCAPDECAP))
        self.assertEqual(
            set(functions),
            {
                "encapsulation",
                "decapsulation",
                "encapsulationKeyCheck",
                "decapsulationKeyCheck",
            },
        )

    def test_only_the_key_checks_publish_a_verdict(self) -> None:
        # The fact the harness is built around: decapsulation has no verdict to
        # publish, because implicit rejection produces a secret rather than a
        # failure.
        for vector in _load(_ENCAPDECAP):
            if vector.function.endswith("KeyCheck"):
                self.assertIsNotNone(vector.valid, vector.case_id)
            else:
                self.assertIsNone(vector.valid, vector.case_id)

    def test_encapsulation_cases_carry_the_senders_randomness(self) -> None:
        for vector in _load(_ENCAPDECAP):
            if vector.function != "encapsulation":
                continue
            self.assertIsNotNone(vector.randomness, vector.case_id)
            self.assertIsNotNone(vector.encapsulation_key, vector.case_id)
            self.assertIsNotNone(vector.ciphertext, vector.case_id)
            self.assertIsNotNone(vector.shared_secret, vector.case_id)

    def test_decapsulation_cases_carry_a_key_ciphertext_and_secret(self) -> None:
        for vector in _load(_ENCAPDECAP):
            if vector.function != "decapsulation":
                continue
            self.assertIsNotNone(vector.decapsulation_key, vector.case_id)
            self.assertIsNotNone(vector.ciphertext, vector.case_id)
            self.assertIsNotNone(vector.shared_secret, vector.case_id)

    def test_no_case_carries_a_field_the_record_cannot_express(self) -> None:
        # The guard against a silent ACVP regeneration: a new field surfaces here
        # as a refusal rather than as a mode nobody noticed.
        carrying = [v for v in _load(_ENCAPDECAP) if v.unsupported]
        self.assertEmpty(
            {name for v in carrying for name in v.unsupported},
            "ACVP grew a field this record does not express",
        )


class AcvpLoaderErrorTest(absltest.TestCase):
    def test_mismatched_halves_are_an_error(self) -> None:
        # A prompt joined against another set's expectedResults shares no
        # (tgId, tcId), which must fail loudly rather than yield an empty set.
        with self.assertRaisesRegex(KatError, "no expected result"):
            load_acvp_ml_kem(
                _path(_ENCAPDECAP[0], "prompt.json"),
                _path(_KEYGEN[1], "expectedResults.json"),
            )


if __name__ == "__main__":
    absltest.main()
