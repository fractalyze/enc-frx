# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM against every case ACVP publishes, at all three parameter sets.

`ML-KEM-keyGen-FIPS203` is the first thing here that gates key generation end to
end. CCTV's files cannot: they expand `(ρ, σ) ← G(d)` where FIPS 203 Algorithm 13
line 1 is `G(d ‖ k)`, so they enter below the expansion and the parameter-set
byte is invisible to them
([`docs/schemes/ml-kem.md`](../../../docs/schemes/ml-kem.md)). ACVP is generated
against the final standard and publishes `(d, z) -> (ek, dk)`, which closes that
gap with a vector rather than with a second gate.

**Comparing `k` exactly is what gates rejection**, and there is no subset to
single out. ACVP marks no decapsulation case as modified — under implicit
rejection a wrong ciphertext yields a shared secret, so there is no verdict to
publish — so "run the rejection cases" means "run the decapsulation cases", and
`check_kem` adds the three properties the file cannot express: that the whole
ciphertext is consumed, that rejection repeats, and that it is decided per batch
entry.

**The two key-check groups are a distinct operation, not a filtered-out
remainder.** `encapDecap` publishes four functions, and `encapsulationKeyCheck`
and `decapsulationKeyCheck` are the FIPS 203 §7.2 and §7.3 input validation,
which the `Kem` seam does not name and the harness refuses rather than skips
([`kat.py`](../../testing/kat.py)). Here they drive `MlKem`'s two predicates
directly, one per section — the groups publish separate verdicts, so a merged
"is this key valid" would answer neither. Each group is 5 valid and 5 invalid
cases, which also makes it a published mixed-validity batch.

**The whole corpus runs on every PR rather than as a `slow_kat` sweep.** It is
240 cases, and the expensive half is `check_kem`'s tampering pass, which is
quadratic in the decapsulation group — 10 entries per set. Sharded by parameter
set it is well under a minute, so the exhaustive pass is affordable as a gate,
and a `slow_kat` tag would instead mean this repo's standards-exactness claim did
not block a merge (`.bazelrc.ci`).
"""

from __future__ import annotations

import functools
import hashlib
from typing import NamedTuple

import numpy as np
from absl.testing import absltest, parameterized
from python.runfiles import runfiles

from enc_frx.ml_kem import encoding
from enc_frx.ml_kem.ml_kem import MlKem
from enc_frx.ml_kem.params import SEED_SIZE
from enc_frx.testing.kat import (
    KatError,
    KemVector,
    check_kem,
    load_acvp_ml_kem,
    to_bytes,
)

# The two `(prompt, expectedResults)` pairs, sha256-pinned in //MODULE.bazel.
# Key generation and `encapDecap` are published separately, but one `MlKem`
# performs all of it and `check_kem` takes one instance's cases whatever file
# they came from — so the two are joined and partitioned by parameter set.
_SETS = (
    ("acvp_ml_kem_keygen_prompt", "acvp_ml_kem_keygen_expected"),
    ("acvp_ml_kem_encapdecap_prompt", "acvp_ml_kem_encapdecap_expected"),
)

# FIPS 203 Table 2, keyed by the name ACVP publishes. A case cannot carry these
# and they are not recoverable from one: `ek`'s length gives `k`, and nothing
# gives `eta1`, `du` or `dv`. Which named sets exist is a layer above `MlKem`, so
# the mapping lives with the vectors that need it.
_PARAMETERS = {
    "ML-KEM-512": {"k": 2, "eta1": 3, "eta2": 2, "du": 10, "dv": 4},
    "ML-KEM-768": {"k": 3, "eta1": 2, "eta2": 2, "du": 10, "dv": 4},
    "ML-KEM-1024": {"k": 4, "eta1": 2, "eta2": 2, "du": 11, "dv": 5},
}

# The three functions the `Kem` seam names, in the order a scheme performs them.
_SEAM_FUNCTIONS = ("keygen", "encapsulation", "decapsulation")


class _KeyCheck(NamedTuple):
    """What one key-check group is published for.

    `predicate` is the `MlKem` method that answers that section and only that
    section; `field` and `size` are the `KemVector` field it asks about and the
    length FIPS 203 Table 3 fixes for it.
    """

    predicate: str
    field: str
    size: str


# The two functions the seam does not name — one section each, and so one
# predicate each.
_KEY_CHECKS = {
    "encapsulationKeyCheck": _KeyCheck(
        "check_encapsulation_key", "encapsulation_key", "encapsulation_key_size"
    ),
    "decapsulationKeyCheck": _KeyCheck(
        "check_decapsulation_key", "decapsulation_key", "decapsulation_key_size"
    ),
}

_NAMED = tuple((name, name) for name in _PARAMETERS)

_NAMED_KEY_CHECKS = tuple(
    (f"{parameter_set}_{function}", parameter_set, function)
    for parameter_set in _PARAMETERS
    for function in _KEY_CHECKS
)


def _path(repo: str, name: str) -> str:
    location = runfiles.Create().Rlocation(f"{repo}/file/{name}")
    assert location is not None, f"{repo} not in runfiles"
    return location


@functools.cache
def _vectors() -> tuple[KemVector, ...]:
    """Every case both files publish. Cached — the pair is a few megabytes."""
    return tuple(
        vector
        for prompt, expected in _SETS
        for vector in load_acvp_ml_kem(
            _path(prompt, "prompt.json"), _path(expected, "expectedResults.json")
        )
    )


def _group(parameter_set: str, function: str) -> list[KemVector]:
    """One published group: one parameter set's cases for one function."""
    return [
        vector
        for vector in _vectors()
        if vector.parameter_set == parameter_set and vector.function == function
    ]


def _seam(parameter_set: str) -> list[KemVector]:
    """One instance's keygen, encapsulation and decapsulation cases."""
    return [
        vector
        for function in _SEAM_FUNCTIONS
        for vector in _group(parameter_set, function)
    ]


def _scheme(parameter_set: str) -> MlKem:
    return MlKem(**_PARAMETERS[parameter_set])


def _j(z: bytes, c: bytes) -> bytes:
    """`J(z ‖ c) = SHAKE256(z ‖ c, 32)`, §4.1 — the implicit-rejection secret,
    from `hashlib` rather than from this repo's own Keccak."""
    return hashlib.shake_256(z + c).digest(SEED_SIZE)


def _stack(items: list[bytes]) -> np.ndarray:
    return np.stack([np.frombuffer(item, dtype=np.uint8) for item in items])


class AcvpTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_every_published_case(self, parameter_set: str) -> None:
        """Every check the standard requires, plus the tampering it does not.

        All three sets rather than one: `eta1` is 3 at ML-KEM-512 and 2 at the
        other two, and `du`/`dv` change at ML-KEM-1024, so a run at 768 alone
        would leave one centered-binomial width and the whole compression path
        ungated.
        """
        vectors = _seam(parameter_set)
        self.assertNotEmpty(vectors)
        check_kem(_scheme(parameter_set), vectors)


class KeyCheckTest(parameterized.TestCase):
    """FIPS 203 §7.2 and §7.3 against the only groups here that publish a
    verdict."""

    @parameterized.named_parameters(*_NAMED_KEY_CHECKS)
    def test_the_published_verdicts_are_reproduced_per_entry(
        self, parameter_set: str, function: str
    ) -> None:
        """One group, one batched call, one verdict per entry.

        The group is half valid and half invalid, so this is a published
        mixed-validity batch: a predicate reduced over the batch, or one that
        answered a constant, fails here rather than passing every all-valid and
        every all-invalid set ever written.
        """
        vectors = _group(parameter_set, function)
        self.assertNotEmpty(vectors)
        published = [vector.valid for vector in vectors]
        # Both verdicts appear, or the comparison below proves nothing.
        self.assertIn(True, published)
        self.assertIn(False, published)

        check = _KEY_CHECKS[function]
        keys = _stack([getattr(vector, check.field) for vector in vectors])
        predicate = getattr(_scheme(parameter_set), check.predicate)
        got = np.asarray(predicate(keys)).tolist()
        self.assertEqual(got, published, [vector.case_id for vector in vectors])

    @parameterized.named_parameters(*_NAMED_KEY_CHECKS)
    def test_every_published_key_is_the_length_the_standard_fixes(
        self, parameter_set: str, function: str
    ) -> None:
        """No case is a type-check case, which is why the comparison above can
        run as one batched call.

        FIPS 203 §7.2 opens with a type check, and a length is static in a traced
        program — so `MlKem` raises on it rather than returning a value
        ([`encoding.py`](../encoding.py)). Were ACVP to publish a short key, the
        predicate would raise where its group expects a `False`, and the reason
        belongs at the corpus rather than as a mystery failure inside a batch.
        """
        check = _KEY_CHECKS[function]
        size = getattr(_scheme(parameter_set), check.size)
        for vector in _group(parameter_set, function):
            self.assertLen(getattr(vector, check.field), size, vector.case_id)

    def test_the_harness_refuses_a_key_check_case(self) -> None:
        # Against the published `function` strings rather than a stand-in, and
        # against the mixture a caller who forgot to filter would actually hold:
        # what the harness must not do is run these through decapsulation and
        # report a pass for a case nobody ran. Any one set says it.
        parameter_set = "ML-KEM-512"
        with self.assertRaisesRegex(KatError, "seam does not name"):
            check_kem(
                _scheme(parameter_set),
                _seam(parameter_set) + _group(parameter_set, "encapsulationKeyCheck"),
            )


class MalformedKeyTest(parameterized.TestCase):
    @parameterized.named_parameters(*_NAMED)
    def test_a_published_malformed_key_rejects_a_ciphertext_meant_for_it(
        self, parameter_set: str
    ) -> None:
        """The rejection path, driven by keys this repo did not corrupt.

        Every key in the group is encapsulated *to* — the ciphertext comes from
        the encapsulation key that key itself carries — so the two halves say
        different things about one construction: a valid key returns the sender's
        own secret, and a malformed one returns `J(z ‖ c)` instead of raising,
        through the same channel a wrong ciphertext takes.

        That control is what the case rests on. Handing every key a ciphertext
        from elsewhere would fail the re-encryption comparison on its own, so
        every entry would reject whatever its key said — which a `decaps` that
        rejected unconditionally would also produce.

        What this cannot isolate is *which* check fired on the malformed half.
        ACVP does not publish how a key was made invalid, and the answer is
        `J(z ‖ c)` for §7.2, §7.3 and a failed re-encryption alike;
        [`ml_kem_test.py`](ml_kem_test.py) isolates §7.3 against CCTV's published
        `KBar`.
        """
        scheme = _scheme(parameter_set)
        vectors = _group(parameter_set, "decapsulationKeyCheck")
        keys = _stack([vector.decapsulation_key for vector in vectors])  # type: ignore[misc]

        _, embedded, _, z = encoding.decode_dk(keys, _PARAMETERS[parameter_set]["k"])
        # Any fixed randomness: what is compared is the pair of answers the same
        # ciphertext draws from a valid and a malformed key, not the ciphertext.
        randomness = np.arange(len(vectors) * SEED_SIZE, dtype=np.uint8).reshape(
            len(vectors), SEED_SIZE
        )
        ciphertexts, secrets = scheme.encaps(embedded, randomness=randomness)

        got = np.asarray(scheme.decaps(keys, ciphertexts))
        for index, vector in enumerate(vectors):
            want = (
                to_bytes(secrets[index])
                if vector.valid
                else _j(bytes(np.asarray(z)[index]), to_bytes(ciphertexts[index]))
            )
            self.assertEqual(bytes(got[index]), want, vector.case_id)


class CorpusTest(absltest.TestCase):
    def test_every_published_case_is_driven_by_something(self) -> None:
        """The partition covers the corpus, and neither half is empty.

        The guard against a silent drop: a regenerated set that grew a function
        would leave the two counts summing to less than the files hold, and a
        suite reporting green over a subset is what the harness exists to
        prevent.
        """
        driven = sum(len(_seam(name)) for name in _PARAMETERS)
        key_checked = sum(
            len(_group(name, function))
            for name in _PARAMETERS
            for function in _KEY_CHECKS
        )
        self.assertGreater(driven, 0)
        self.assertGreater(key_checked, 0)
        self.assertEqual(driven + key_checked, len(_vectors()))


if __name__ == "__main__":
    absltest.main()
