# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Reaching CAVP's GCM files, and picking batches out of them.

Shared by the gate (`gcm_test`) and the sweep (`gcm_sweep_test`), which run the
same vectors at different depths on different schedules.

TEST ONLY. Never re-exported from the package.
"""

from __future__ import annotations

import functools

from python.runfiles import runfiles

from enc_frx.testing.kat import AeadVector, AeadVectorSet, load_cavp_gcm

# One file per key length and direction, which is how CAVP splits the set. The
# key length is in the name; the IV, payload, associated-data and tag lengths
# vary in sections inside.
ENCRYPT_FILES = {
    16: "gcmEncryptExtIV128.rsp",
    24: "gcmEncryptExtIV192.rsp",
    32: "gcmEncryptExtIV256.rsp",
}

DECRYPT_FILES = {
    16: "gcmDecrypt128.rsp",
    24: "gcmDecrypt192.rsp",
    32: "gcmDecrypt256.rsp",
}

FILES = tuple(ENCRYPT_FILES.values()) + tuple(DECRYPT_FILES.values())

# The three IV lengths the set publishes, in bytes. 12 is used directly as `J_0`;
# the other two go through GHASH, so both branches of `ctr.initial_counter` are
# exercised by published vectors rather than only by its own unit test.
NONCE_SIZES = (1, 12, 128)

# The tag lengths the set publishes, in bytes. `AesGcm` admits the last five —
# SP 800-38D §5.2.1.2 — and refuses 4 and 8, which are Appendix C's.
PUBLISHED_TAG_SIZES = (4, 8, 12, 13, 14, 15, 16)


def path(name: str) -> str:
    location = runfiles.Create().Rlocation(f"cavp_aes_gcm/{name}")
    assert location is not None, f"{name} not in runfiles"
    return location


@functools.cache
def sets(name: str) -> tuple[AeadVectorSet, ...]:
    """Every scheme instance one CAVP file publishes.

    Cached because a file is a few megabytes and a parameterized suite reads the
    same one a dozen times.
    """
    return tuple(load_cavp_gcm(path(name)))


def instance(name: str, key_size: int, nonce_size: int, tag_size: int) -> AeadVectorSet:
    for vector_set in sets(name):
        if (vector_set.key_size, vector_set.nonce_size, vector_set.tag_size) == (
            key_size,
            nonce_size,
            tag_size,
        ):
            return vector_set
    raise AssertionError(
        f"{name} publishes no AES-{key_size * 8} instance with a {nonce_size}-byte "
        f"IV and a {tag_size}-byte tag"
    )


def _shape_groups(
    vector_set: AeadVectorSet,
) -> dict[tuple[int, int], list[AeadVector]]:
    """Split an instance by byte lengths, as the harness does internally.

    A batch axis needs one static shape, and a CAVP instance deliberately varies
    the payload and associated-data lengths across its sections.
    """
    groups: dict[tuple[int, int], list[AeadVector]] = {}
    for vector in vector_set.vectors:
        key = (len(vector.associated_data or b""), len(vector.ciphertext))
        groups.setdefault(key, []).append(vector)
    return groups


def _gate_group(vector_set: AeadVectorSet) -> list[AeadVector]:
    """The shape group the gate runs: the smallest with both a payload and an AAD.

    Both non-empty so that all four of the harness's tamperings say something. A
    flipped associated-data byte is skipped when there is no associated data, and
    a flipped *first ciphertext byte* lands inside the tag when the payload is
    empty — which is the tag tampering over again rather than a second check.

    Smallest, because the gate runs on every PR and the tampering pass is
    quadratic in the group; the long payloads are the sweep's job.
    """
    groups = _shape_groups(vector_set)
    key = min(k for k in groups if k[0] > 0 and k[1] > vector_set.tag_size)
    return groups[key]


def accepted_batch(vector_set: AeadVectorSet, size: int) -> list[AeadVector]:
    """A batch of cases the standard expects to open."""
    return [v for v in _gate_group(vector_set) if v.valid][:size]


def mixed_batch(vector_set: AeadVectorSet, size: int) -> list[AeadVector]:
    """A batch that alternates accepted and rejected cases.

    The mixed batch is its own requirement: a masking bug applied batch-wide
    passes every all-valid and every all-invalid set ever published, so a suite
    without one has not tested the batch axis at all. CAVP's decrypt sections
    supply the mix — roughly half their cases carry `FAIL` — but taking the first
    `size` of one would leave which verdicts appear to chance, and alternating
    them also puts a rejection at a position other than the end.
    """
    group = _gate_group(vector_set)
    accepted = [v for v in group if v.valid]
    rejected = [v for v in group if not v.valid]
    assert accepted and rejected, "a decrypt section publishes both verdicts"
    return [v for pair in zip(accepted, rejected, strict=False) for v in pair][:size]
