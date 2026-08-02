# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The known-answer-test harness: one driver per seam, any implementation.

Every scheme here is gated on published vectors. The formats differ per standard
— ACVP JSON for FIPS 203, inline vectors in RFC 8439, CAVP response files for
AES-GCM — while the shape of the test does not: load a vector, run the seam's
operations, compare bytes. So a loader normalizes its format into a vector record
and `check_kem` / `check_aead` drive any implementation of the corresponding
seam.

**Two records and two drivers, not one of each.** A KEM case carries a seed, an
encapsulation key and a shared secret; an AEAD case carries a nonce, associated
data and a tag. The field sets barely overlap, and the checks do not overlap at
all — which is the same reason the seams are separate. One record would be a
dozen optional fields of which any case uses half, and one driver would have to
guess which seam it was holding.

**The negative cases are not optional, and there is no flag to skip them.** An
`open` that returns `ok = True` unconditionally passes every positive vector ever
published. So the drivers derive a tampered batch from whatever positives they
are handed and require the rejection; a scheme never gets the choice.

`check_aead_published` is not that flag. It answers a narrower question — does
each published case produce its published answer — because deriving a tampered
batch per entry is quadratic in a group and a set like CAVP's GCM vectors runs to
tens of thousands of cases. A sweep runs it and a gate runs `check_aead`; its
docstring says what a scheme gated on it alone would fail to prove.

**A KEM's rejection cases cannot be counted, so the gate is placed elsewhere.**
ACVP's ML-KEM decapsulation cases carry no pass/fail marker — every case has an
expected shared secret and nothing says which ciphertexts were modified. That is
not an omission: under implicit rejection there is nothing to mark, because a
modified ciphertext produces a shared secret rather than a failure. Comparing `k`
exactly therefore already gates rejection, and what the harness enforces instead
is that a run *reaches* decapsulation at all — a scheme gated on encapsulation
vectors alone has never executed its rejection path.

**A vector the harness cannot run faithfully is an error, not a skip.** ACVP's
`encapsulationKeyCheck` and `decapsulationKeyCheck` groups test the FIPS 203 §7.2
and §7.3 input validation, which is not an operation either seam names. A loader
records the function rather than dropping the case, and the driver refuses it —
running decapsulation against a vector published for key validation would report
a pass for a case nobody ran.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import Array

from enc_frx.aead import Aead
from enc_frx.kem import Kem


class KatError(Exception):
    """A vector set the harness will not run, or a scheme that failed it."""


@dataclass(frozen=True)
class KemVector:
    """One KEM case, normalized out of whatever format published it."""

    # Identifies the case in a failure message. A bare index is useless when a
    # sweep of thousands fails on one.
    case_id: str
    parameter_set: str
    # Which operation the case was published for. `keygen`, `encapsulation` and
    # `decapsulation` are the seam's; the two key-check functions are FIPS 203
    # §7.2/§7.3 validation, which the seam does not name and the driver refuses.
    function: str
    seed: bytes | None = None
    encapsulation_key: bytes | None = None
    decapsulation_key: bytes | None = None
    randomness: bytes | None = None
    ciphertext: bytes | None = None
    shared_secret: bytes | None = None
    # Published only by the key-check functions; `None` everywhere else, because
    # decapsulation has no verdict to publish.
    valid: bool | None = None
    # Published fields the record cannot express, by name. A loader records what
    # it could not feed rather than dropping it.
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class AeadVector:
    """One AEAD case, normalized out of whatever format published it."""

    case_id: str
    parameter_set: str
    key: bytes
    nonce: bytes
    plaintext: bytes
    ciphertext: bytes  # body and tag, in the standard's layout
    associated_data: bytes | None = None
    # The published verdict. CAVP's GCM decrypt sets carry deliberate failures,
    # so this is load-bearing rather than a formality.
    valid: bool = True
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class AeadVectorSet:
    """One scheme instance's parameters, and the cases published under them.

    An AEAD's parameters are not all recoverable from a case. The key and nonce
    sizes are the lengths of those fields, but the tag size is not: the tag lives
    *inside* `ciphertext`, and the body's length varies per case, so nothing in a
    record says where the split is. A set carries it once, alongside the two that
    are redundant, because those three together are exactly the arguments a scheme
    instance is built from.

    So a set maps one-to-one onto a `check_aead` call, which is also what that
    driver requires — it refuses a mixed parameter set. Formats that publish a
    single instance's cases (RFC 8439's) need no such grouping and a test builds
    the list directly; CAVP's files interleave 63 instances, and splitting them is
    the loader's job rather than every caller's.
    """

    key_size: int
    nonce_size: int
    tag_size: int
    vectors: tuple[AeadVector, ...]


@dataclass(frozen=True)
class AesVector:
    """One AES block case, normalized out of whatever format published it.

    The block cipher is not a seam implementation — it is an internal primitive
    of AES-GCM — so this record has a loader and no driver. Its own test is the
    gate, which `docs/reference/conventions.md` allows on exactly that condition.
    """

    case_id: str
    parameter_set: str  # AES-128 / AES-192 / AES-256
    direction: str  # encrypt / decrypt
    test_type: str  # AFT (one block) / MCT (a chained Monte Carlo procedure)
    key: bytes
    plaintext: bytes
    ciphertext: bytes
    # CTR publishes the counter block the implementation chose; ECB has none.
    iv: bytes = b""
    # ACVP measures a CTR payload in **bits**, and most of its cases are not a
    # whole number of bytes. Zero where the mode does not publish one.
    payload_bits: int = 0
    unsupported: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# ACVP splits a vector set in two: prompt.json holds the inputs, and
# expectedResults.json the outputs, joined on (tgId, tcId).
#
# Key generation takes its seed as two named pieces, in the order FIPS 203 §7.1
# takes them. Concatenating here is what lets the seam keep one `keygen(seed)`.
_ACVP_SEED_FIELDS = ("d", "z")

_ACVP_BYTE_FIELDS = {
    "ek": "encapsulation_key",
    "dk": "decapsulation_key",
    "m": "randomness",
    "c": "ciphertext",
    "k": "shared_secret",
}

# Fields that identify or annotate a case rather than feed it. Anything outside
# these, the mapped fields and the seed fields is a mode nothing here expresses
# and is recorded as unsupported — a new ACVP field then surfaces as a refusal
# rather than a silent behavior change.
_ACVP_IGNORED_FIELDS = frozenset(
    {"tgId", "tcId", "testType", "parameterSet", "function", "tests", "testPassed"}
)

# `keyGen` publishes no `function`; the mode names the operation instead.
_ACVP_KEYGEN_FUNCTION = "keygen"


def load_acvp_ml_kem(
    prompt_path: Path | str, expected_path: Path | str
) -> list[KemVector]:
    """Normalize one ACVP ML-KEM set — a `(prompt, expectedResults)` pair."""
    prompt = json.loads(Path(prompt_path).read_text())
    expected = json.loads(Path(expected_path).read_text())

    results = {
        (group["tgId"], test["tcId"]): test
        for group in expected["testGroups"]
        for test in group["tests"]
    }

    vectors: list[KemVector] = []
    for group in prompt["testGroups"]:
        parameter_set = group["parameterSet"]
        function = group.get("function", _ACVP_KEYGEN_FUNCTION)
        for test in group["tests"]:
            key = (group["tgId"], test["tcId"])
            if key not in results:
                raise KatError(
                    f"{parameter_set} tg{key[0]}/tc{key[1]} has no expected result; "
                    f"the prompt and expectedResults files are not a matching pair"
                )
            # Group-level fields are merged under the test's own, which win —
            # ACVP puts a field at whichever level it is constant, and that level
            # differs per set.
            merged = {**group, **results[key], **test}
            fields: dict[str, Any] = {
                dest: bytes.fromhex(merged[src])
                for src, dest in _ACVP_BYTE_FIELDS.items()
                if src in merged
            }
            seed = b"".join(
                bytes.fromhex(merged[name])
                for name in _ACVP_SEED_FIELDS
                if name in merged
            )
            unsupported = tuple(
                sorted(
                    name
                    for name in merged
                    if name not in _ACVP_IGNORED_FIELDS
                    and name not in _ACVP_BYTE_FIELDS
                    and name not in _ACVP_SEED_FIELDS
                )
            )
            vectors.append(
                KemVector(
                    case_id=f"{parameter_set}/{function}/tg{key[0]}/tc{key[1]}",
                    parameter_set=parameter_set,
                    function=function,
                    seed=seed or None,
                    valid=merged.get("testPassed"),
                    unsupported=unsupported,
                    **fields,
                )
            )
    return vectors


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def to_bytes(value: Any) -> bytes:
    """The wire form of a key or ciphertext, for comparison against a vector.

    A uint8 array is the only form the harness can compare, because bytes are
    what a standard publishes. A scheme whose output is something else needs a
    wire codec at the seam before its vectors can be run — and that is a seam
    decision, not one a test harness should make silently.
    """
    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise KatError(
            f"expected a uint8 array to compare against published bytes, got "
            f"dtype {array.dtype}; the scheme needs a wire codec at the seam"
        )
    return bytes(array.reshape(-1))


def _as_array(data: bytes) -> Array:
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _stack(items: Sequence[bytes]) -> Array:
    return fnp.stack([_as_array(item) for item in items])


def _flip_first_bit(data: bytes) -> bytes:
    return bytes([data[0] ^ 1]) + data[1:]


def _flip_last_bit(data: bytes) -> bytes:
    return data[:-1] + bytes([data[-1] ^ 1])


def _one_parameter_set(vectors: Sequence[Any]) -> None:
    sets = {v.parameter_set for v in vectors}
    if len(sets) != 1:
        raise KatError(
            f"one scheme instance is one parameter set, got {sorted(sets)}; "
            f"group the vectors and build an instance per set"
        )


def _reject_unsupported(vectors: Sequence[Any]) -> None:
    carrying = [v for v in vectors if v.unsupported]
    if carrying:
        names = sorted({name for v in carrying for name in v.unsupported})
        raise KatError(
            f"{len(carrying)} of {len(vectors)} vectors carry fields {names} that "
            f"this record cannot express (first: {carrying[0].case_id}). Running "
            f"them anyway would report a pass for a case nobody ran."
        )


# ---------------------------------------------------------------------------
# The KEM driver
# ---------------------------------------------------------------------------

_KEM_SEAM_FUNCTIONS = frozenset(
    {_ACVP_KEYGEN_FUNCTION, "encapsulation", "decapsulation"}
)


def check_kem(scheme: Kem, vectors: Sequence[KemVector]) -> None:
    """Run every check the standard requires, plus the tampering it does not.

    `vectors` are one parameter set's, and may mix keygen, encapsulation and
    decapsulation — one ML-KEM instance performs all three. Raises `KatError` on
    the first failure, naming the case.
    """
    if not vectors:
        raise KatError("no vectors: an empty set passes trivially and proves nothing")

    _reject_unsupported(vectors)
    _one_parameter_set(vectors)

    unnamed = sorted({v.function for v in vectors} - _KEM_SEAM_FUNCTIONS)
    if unnamed:
        raise KatError(
            f"vectors published for {unnamed}, which the Kem seam does not name — "
            f"those are the FIPS 203 §7.2/§7.3 input-validation checks, and the "
            f"seam has no validation entry point. Filter them out and gate them "
            f"where that validation lives."
        )

    decapsulation = [v for v in vectors if v.function == "decapsulation"]
    if not decapsulation:
        # ACVP marks no decapsulation case as a rejection — under implicit
        # rejection there is nothing to mark — so the enforceable requirement is
        # that the run reaches decapsulation at all. A scheme gated on
        # encapsulation vectors alone has never executed its rejection path.
        raise KatError(
            "no decapsulation vectors: encapsulation alone never runs the "
            "re-encryption check or the rejection path, which is the half of the "
            "scheme most likely to be silently wrong"
        )

    _check_kem_keygen(scheme, vectors)
    _check_encaps(scheme, [v for v in vectors if v.function == "encapsulation"])
    _check_decaps(scheme, decapsulation)


def _check_kem_keygen(scheme: Kem, vectors: Sequence[KemVector]) -> None:
    """Keygen is deterministic in the seed and reproduces the published keys."""
    for vector in vectors:
        if vector.seed is None or vector.function != _ACVP_KEYGEN_FUNCTION:
            continue
        encapsulation_key, decapsulation_key = scheme.keygen(_as_array(vector.seed))
        if (
            vector.encapsulation_key is not None
            and to_bytes(encapsulation_key) != vector.encapsulation_key
        ):
            raise KatError(
                f"{vector.case_id}: keygen produced the wrong encapsulation key"
            )
        if (
            vector.decapsulation_key is not None
            and to_bytes(decapsulation_key) != vector.decapsulation_key
        ):
            raise KatError(
                f"{vector.case_id}: keygen produced the wrong decapsulation key"
            )


def _check_encaps(scheme: Kem, vectors: Sequence[KemVector]) -> None:
    """Encapsulation reproduces the published ciphertext and shared secret."""
    runnable = [
        v
        for v in vectors
        if v.encapsulation_key is not None and v.randomness is not None
    ]
    if not runnable:
        return
    ciphertexts, secrets = scheme.encaps(
        _stack([v.encapsulation_key for v in runnable]),  # type: ignore[misc]
        randomness=_stack([v.randomness for v in runnable]),  # type: ignore[misc]
    )
    for index, vector in enumerate(runnable):
        if (
            vector.ciphertext is not None
            and to_bytes(ciphertexts[index]) != vector.ciphertext
        ):
            raise KatError(f"{vector.case_id}: encaps produced the wrong ciphertext")
        if (
            vector.shared_secret is not None
            and to_bytes(secrets[index]) != vector.shared_secret
        ):
            raise KatError(f"{vector.case_id}: encaps produced the wrong shared secret")


def _check_decaps(scheme: Kem, vectors: Sequence[KemVector]) -> None:
    """Decapsulation reproduces every published secret, then survives tampering.

    The published secrets are what gate rejection: a modified ciphertext's
    expected `k` *is* the implicit-rejection value, which is why no case is
    marked. The tampering pass cannot add to that — without an oracle for the
    rejection secret, a wrong-but-different answer is indistinguishable from a
    correctly rejected one. What it does add is the three properties the file
    cannot express: that the whole ciphertext is consumed, that rejection
    repeats, and that it is decided per batch entry.
    """
    runnable = [
        v
        for v in vectors
        if v.decapsulation_key is not None
        and v.ciphertext is not None
        and v.shared_secret is not None
    ]
    if not runnable:
        return

    keys = _stack([v.decapsulation_key for v in runnable])  # type: ignore[misc]
    ciphertexts = [v.ciphertext for v in runnable]
    secrets = scheme.decaps(keys, _stack(ciphertexts))  # type: ignore[arg-type]
    for index, vector in enumerate(runnable):
        if to_bytes(secrets[index]) != vector.shared_secret:
            raise KatError(f"{vector.case_id}: decaps produced the wrong shared secret")

    _check_kem_rejection(scheme, runnable, keys, ciphertexts)  # type: ignore[arg-type]


def _check_kem_rejection(
    scheme: Kem,
    vectors: Sequence[KemVector],
    keys: Array,
    ciphertexts: Sequence[bytes],
) -> None:
    """One flipped bit, in one entry, with every other entry's secret pinned.

    Both ends of the ciphertext, because where the bit moves decides what the
    check can see. The leading bytes carry the encrypted message, so *any*
    scheme's output changes when they do — flipping there proves almost nothing.
    A ciphertext's trailing bytes are where the redundancy a re-encryption check
    consumes tends to live, and a scheme that skips that check reads them not at
    all: its answer is then unchanged, which is exactly the failure worth
    catching.
    """
    for position, flip in (("leading", _flip_first_bit), ("trailing", _flip_last_bit)):
        _check_kem_rejection_at(scheme, vectors, keys, ciphertexts, position, flip)


def _check_kem_rejection_at(
    scheme: Kem,
    vectors: Sequence[KemVector],
    keys: Array,
    ciphertexts: Sequence[bytes],
    position: str,
    flip: Callable[[bytes], bytes],
) -> None:
    for index, vector in enumerate(vectors):
        tampered = list(ciphertexts)
        tampered[index] = flip(tampered[index])
        first = scheme.decaps(keys, _stack(tampered))
        if to_bytes(first[index]) == vector.shared_secret:
            raise KatError(
                f"{vector.case_id}: a flipped bit in the {position} ciphertext "
                f"bytes still decapsulated to the published secret — those bytes "
                f"are not consumed, so nothing rejects a ciphertext that changes "
                f"only there"
            )
        # Implicit rejection is only implicit if it repeats; a secret that varied
        # per call would hand the caller the verdict back.
        again = scheme.decaps(keys, _stack(tampered))
        if to_bytes(first[index]) != to_bytes(again[index]):
            raise KatError(
                f"{vector.case_id}: rejection is not deterministic — repeating the "
                f"call gave a different shared secret"
            )
        for other, entry in enumerate(vectors):
            if other != index and to_bytes(first[other]) != entry.shared_secret:
                raise KatError(
                    f"{entry.case_id}: changed because a different entry of the "
                    f"batch was tampered with — decaps is not deciding per entry"
                )


# ---------------------------------------------------------------------------
# The AEAD driver
# ---------------------------------------------------------------------------


def _corrupt_ciphertext(vector: AeadVector) -> AeadVector:
    return replace(vector, ciphertext=_flip_first_bit(vector.ciphertext))


def _corrupt_tag(vector: AeadVector) -> AeadVector:
    body, tag = vector.ciphertext[:-1], vector.ciphertext[-1:]
    return replace(vector, ciphertext=body + bytes([tag[0] ^ 1]))


def _corrupt_nonce(vector: AeadVector) -> AeadVector:
    return replace(vector, nonce=_flip_first_bit(vector.nonce))


def _corrupt_aad(vector: AeadVector) -> AeadVector:
    assert vector.associated_data is not None
    return replace(vector, associated_data=_flip_first_bit(vector.associated_data))


# The four inputs an opener is given, each of which must be able to break it.
_AEAD_TAMPERINGS: tuple[tuple[str, Callable[[AeadVector], AeadVector]], ...] = (
    ("tag", _corrupt_tag),
    ("ciphertext", _corrupt_ciphertext),
    ("nonce", _corrupt_nonce),
    ("associated data", _corrupt_aad),
)


def check_aead(scheme: Aead, vectors: Sequence[AeadVector]) -> None:
    """Run every check the standard requires, plus the tampering it does not."""
    check_aead_published(scheme, vectors)
    for group in group_aead_by_shape(vectors):
        _check_aead_tampering(scheme, [v for v in group if v.valid])


def check_aead_published(scheme: Aead, vectors: Sequence[AeadVector]) -> None:
    """The published cases alone. **This is not the gate; `check_aead` is.**

    The two exist because they answer different questions at different prices.
    `check_aead` asks whether a scheme obeys the seam — that one moved bit is
    rejected, that the rejected entry is masked, that the verdict is per entry —
    and it asks that of every entry of every batch, so its cost is quadratic in a
    group. This asks only whether each published case produces its published
    answer, which is linear, and that is the only affordable shape for a set of
    tens of thousands.

    So a sweep runs this and a gate runs `check_aead`, and a scheme is not
    permitted to be gated by this alone: a set of positives cannot distinguish an
    `open` that verifies from one that returns `ok = True`.

    How much a published set narrows that gap depends on the set, and CAVP's GCM
    vectors narrow it further than they might appear to. Their `FAIL` cases sit in
    the same sections as their passing ones, so grouping by shape puts both in one
    batch — which means this driver does check that exactly the failing entries of
    a mixed batch come back masked. What it still cannot reach is the tampering
    the standard never published: no CAVP case moves a nonce or an
    associated-data byte, and none says *which* input was corrupted. Those are
    `check_aead`'s, and they are why it is the gate.
    """
    if not vectors:
        raise KatError("no vectors: an empty set passes trivially and proves nothing")

    _reject_unsupported(vectors)
    _one_parameter_set(vectors)

    for group in group_aead_by_shape(vectors):
        _check_seal(scheme, group)
        _check_open(scheme, group)


def _check_seal(scheme: Aead, group: Sequence[AeadVector]) -> None:
    """Sealing reproduces the published ciphertext and tag, byte for byte."""
    sealable = [v for v in group if v.valid]
    if not sealable:
        return
    ciphertexts = scheme.seal(
        *_aead_args(sealable), _stack([v.plaintext for v in sealable])
    )
    for index, vector in enumerate(sealable):
        if to_bytes(ciphertexts[index]) != vector.ciphertext:
            raise KatError(f"{vector.case_id}: seal produced the wrong ciphertext")


def _check_open(scheme: Aead, group: Sequence[AeadVector]) -> None:
    """Opening agrees with every published verdict, and masks what it rejects."""
    plaintexts, ok = scheme.open(
        *_aead_args(group), _stack([v.ciphertext for v in group])
    )
    for index, vector in enumerate(group):
        verdict = bool(np.asarray(ok)[index])
        if verdict != vector.valid:
            published = "accept" if vector.valid else "reject"
            raise KatError(
                f"{vector.case_id}: published verdict is {published}, the scheme "
                f"returned {'accept' if verdict else 'reject'}"
            )
        if verdict:
            if to_bytes(plaintexts[index]) != vector.plaintext:
                raise KatError(f"{vector.case_id}: open recovered the wrong plaintext")
        elif any(to_bytes(plaintexts[index])):
            raise KatError(
                f"{vector.case_id}: rejected but the plaintext came back non-zero — "
                f"open is releasing unverified plaintext"
            )


def _check_aead_tampering(scheme: Aead, group: Sequence[AeadVector]) -> None:
    """Every accepted case must be rejected and masked once one input moves.

    One bit, in one entry of the batch, with the other entries pinned: a scheme
    that reduced its tag comparison over the batch, or masked from `any(ok)`,
    fails here rather than passing everything forever.
    """
    if not group:
        return
    for field, corrupt in _AEAD_TAMPERINGS:
        if field == "associated data" and group[0].associated_data is None:
            continue
        for index in range(len(group)):
            tampered = list(group)
            tampered[index] = corrupt(group[index])
            plaintexts, ok = scheme.open(
                *_aead_args(tampered), _stack([v.ciphertext for v in tampered])
            )
            verdicts = np.asarray(ok)
            if verdicts[index]:
                raise KatError(
                    f"{group[index].case_id}: accepted after a bit flip in the "
                    f"{field}"
                )
            if any(to_bytes(plaintexts[index])):
                raise KatError(
                    f"{group[index].case_id}: rejected after a bit flip in the "
                    f"{field} but the plaintext came back non-zero"
                )
            for other, entry in enumerate(group):
                if other != index and not verdicts[other]:
                    raise KatError(
                        f"{entry.case_id}: rejected because a different entry of "
                        f"the batch was tampered with — open is not deciding per "
                        f"entry"
                    )


def _aead_args(group: Sequence[AeadVector]) -> tuple[Array, Array, Array | None]:
    associated_data = (
        None
        if group[0].associated_data is None
        else _stack([v.associated_data for v in group])  # type: ignore[misc]
    )
    return (
        _stack([v.key for v in group]),
        _stack([v.nonce for v in group]),
        associated_data,
    )


def group_aead_by_shape(vectors: Sequence[AeadVector]) -> list[list[AeadVector]]:
    """Split into batches of equal byte lengths.

    A batch axis needs one static shape, and published sets deliberately vary
    message and associated-data length. Grouping keeps the checks batched within
    each combination rather than falling back to a case-by-case loop.
    """
    groups: dict[tuple[int, int, int, int], list[AeadVector]] = {}
    for vector in vectors:
        key = (
            len(vector.key),
            len(vector.nonce),
            len(vector.associated_data) if vector.associated_data is not None else -1,
            len(vector.ciphertext),
        )
        groups.setdefault(key, []).append(vector)
    return list(groups.values())


_ACVP_AES_IGNORED_FIELDS = frozenset(
    {"tgId", "tcId", "testType", "direction", "keyLen", "tests", "payloadLen"}
)

_ACVP_AES_BYTE_FIELDS = {
    "key": "key",
    "pt": "plaintext",
    "ct": "ciphertext",
    "iv": "iv",
}


def load_acvp_aes(
    prompt_path: Path | str, expected_path: Path | str
) -> list[AesVector]:
    """Normalize one ACVP AES set — a `(prompt, expectedResults)` pair.

    Covers the modes whose groups carry `direction` and `keyLen` — ECB and CTR
    share that shape, differing only in whether a counter block is published.

    Both directions and every test type are loaded rather than filtered here: a
    caller that runs only encryption has to *refuse* the rest, and it cannot
    refuse what the loader silently dropped.
    """
    prompt = json.loads(Path(prompt_path).read_text())
    expected = json.loads(Path(expected_path).read_text())

    results = {
        (group["tgId"], test["tcId"]): test
        for group in expected["testGroups"]
        for test in group["tests"]
    }

    vectors: list[AesVector] = []
    for group in prompt["testGroups"]:
        parameter_set = f"AES-{group['keyLen']}"
        for test in group["tests"]:
            key = (group["tgId"], test["tcId"])
            if key not in results:
                raise KatError(
                    f"{parameter_set} tg{key[0]}/tc{key[1]} has no expected result; "
                    f"the prompt and expectedResults files are not a matching pair"
                )
            merged = {**group, **results[key], **test}
            fields = {
                dest: bytes.fromhex(merged[src])
                for src, dest in _ACVP_AES_BYTE_FIELDS.items()
                if src in merged
            }
            unsupported = tuple(
                sorted(
                    name
                    for name in merged
                    if name not in _ACVP_AES_IGNORED_FIELDS
                    and name not in _ACVP_AES_BYTE_FIELDS
                )
            )
            vectors.append(
                AesVector(
                    case_id=(
                        f"{parameter_set}/{group['direction']}/{group['testType']}"
                        f"/tg{key[0]}/tc{key[1]}"
                    ),
                    parameter_set=parameter_set,
                    direction=group["direction"],
                    test_type=group["testType"],
                    key=fields.get("key", b""),
                    plaintext=fields.get("plaintext", b""),
                    ciphertext=fields.get("ciphertext", b""),
                    iv=fields.get("iv", b""),
                    payload_bits=merged.get("payloadLen", 0),
                    unsupported=unsupported,
                )
            )
    return vectors


# CAVS response files are flat text rather than JSON. `[Name = Value]` lines open
# a section whose five lengths hold until the next one; `Name = Value` lines
# accumulate into a case that `Count` opens; and a decrypt case the standard
# expects to be rejected carries the bare marker `FAIL` where its plaintext would
# be. That marker is why this set is the negative gate — the failures are
# published rather than synthesized here.
_CAVP_SECTION = re.compile(r"^\[\s*(\w+)\s*=\s*(\w+)\s*\]$")
_CAVP_FIELD = re.compile(r"^(\w+)\s*=\s*([0-9a-fA-F]*)$")
_CAVP_FAIL = "FAIL"
_CAVP_COUNT = "Count"

# All five hold at the section level, and all five are stated in bits.
_CAVP_LENGTHS = ("Keylen", "IVlen", "PTlen", "AADlen", "Taglen")

_CAVP_BYTE_FIELDS = ("Key", "IV", "PT", "AAD", "CT", "Tag")


def _cavp_case(stem: str, lengths: dict[str, int], case: dict[str, str]) -> AeadVector:
    """One `Count = N` block, against the section lengths that govern it."""
    case_id = (
        f"{stem}/k{lengths['Keylen']}iv{lengths['IVlen']}tag{lengths['Taglen']}"
        f"/pt{lengths['PTlen']}aad{lengths['AADlen']}/count{case[_CAVP_COUNT]}"
    )

    ragged = {n: lengths[n] for n in _CAVP_LENGTHS if lengths[n] % 8}
    if ragged:
        raise KatError(
            f"{case_id}: {ragged} are not a whole number of bytes. GCM cannot run "
            f"a sub-byte case: the tag covers the ciphertext, so masking the "
            f"payload would be a different computation rather than a different "
            f"encoding."
        )

    fields = {field: bytes.fromhex(case.get(field, "")) for field in _CAVP_BYTE_FIELDS}
    failed = _CAVP_FAIL in case
    declared = [
        ("Key", "Keylen"),
        ("IV", "IVlen"),
        ("CT", "PTlen"),
        ("AAD", "AADlen"),
        ("Tag", "Taglen"),
    ]
    # A rejected case publishes no plaintext, which is right rather than missing:
    # there is nothing the standard expects an opener to produce.
    if not failed:
        declared.append(("PT", "PTlen"))
    for field, length in declared:
        if len(fields[field]) * 8 != lengths[length]:
            raise KatError(
                f"{case_id}: {field} is {len(fields[field])} bytes but the section "
                f"declares {length} = {lengths[length]} bits"
            )

    unsupported = tuple(
        sorted(set(case) - set(_CAVP_BYTE_FIELDS) - {_CAVP_COUNT, _CAVP_FAIL})
    )
    return AeadVector(
        case_id=case_id,
        parameter_set=(
            f"AES-{lengths['Keylen']}-GCM"
            f"/iv{lengths['IVlen']}/tag{lengths['Taglen']}"
        ),
        key=fields["Key"],
        nonce=fields["IV"],
        plaintext=fields["PT"],
        # The seam's layout, which CAVP does not use: it publishes the body and
        # the tag as separate fields, and `seal` returns them concatenated.
        ciphertext=fields["CT"] + fields["Tag"],
        # An empty AAD crosses the seam as `None`, not as a zero-width array —
        # see `Aead.seal`. Passing `b""` here would exercise a shape no caller
        # with nothing to authenticate produces.
        associated_data=fields["AAD"] or None,
        valid=not failed,
        unsupported=unsupported,
    )


def load_cavp_gcm(path: Path | str) -> list[AeadVectorSet]:
    """Normalize one CAVP GCM response file, split into scheme instances.

    One file is one key length and one direction, and inside it the sections vary
    the IV, payload, associated-data and tag lengths. Three of those five name a
    scheme instance rather than a case, so the file carries many instances and the
    return is grouped by them.

    The two directions load through this one function because the record is
    symmetric: a decrypt case is a `(key, IV, AAD, ciphertext)` with a published
    verdict, and an encrypt case is the same thing with the verdict `accept`. So
    the decrypt files gate `seal` too, GCM being deterministic.
    """
    stem = Path(path).stem
    lengths: dict[str, int] = {}
    case: dict[str, str] | None = None
    # Each entry pairs a case with the lengths in force when `Count` opened it.
    # Snapshotting there rather than reading `lengths` afterwards is what keeps a
    # later section header from retroactively regoverning a case already parsed.
    parsed: list[tuple[dict[str, int], dict[str, str]]] = []

    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        section = _CAVP_SECTION.match(line)
        if section:
            case = None
            length, value = section.groups()
            if length not in _CAVP_LENGTHS:
                raise KatError(f"{stem}:{number}: unknown section parameter {length}")
            lengths[length] = int(value)
            continue

        if line == _CAVP_FAIL:
            if case is None:
                raise KatError(f"{stem}:{number}: FAIL outside a case")
            case[_CAVP_FAIL] = ""
            continue

        matched = _CAVP_FIELD.match(line)
        if matched is None:
            raise KatError(f"{stem}:{number}: cannot parse {line!r}")
        field, value = matched.groups()
        if field == _CAVP_COUNT:
            missing = [n for n in _CAVP_LENGTHS if n not in lengths]
            if missing:
                raise KatError(f"{stem}:{number}: case opened before {missing}")
            case = {_CAVP_COUNT: value}
            parsed.append((dict(lengths), case))
            continue
        if case is None:
            raise KatError(f"{stem}:{number}: {field} outside a case")
        case[field] = value

    if not parsed:
        raise KatError(f"{stem}: no cases; an empty set passes trivially")

    sections: dict[tuple[int, int, int], list[AeadVector]] = {}
    for case_lengths, case_fields in parsed:
        key = (
            case_lengths["Keylen"] // 8,
            case_lengths["IVlen"] // 8,
            case_lengths["Taglen"] // 8,
        )
        sections.setdefault(key, []).append(_cavp_case(stem, case_lengths, case_fields))
    return [
        AeadVectorSet(
            key_size=key_size,
            nonce_size=nonce_size,
            tag_size=tag_size,
            vectors=tuple(vectors),
        )
        for (key_size, nonce_size, tag_size), vectors in sections.items()
    ]
