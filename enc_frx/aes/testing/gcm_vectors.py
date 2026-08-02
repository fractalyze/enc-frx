# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Reaching CAVP's GCM files, and picking batches out of them.

Shared by the gate (`gcm_test`) and the sweep (`gcm_sweep_test`), which run the
same vectors at different depths on different schedules.

TEST ONLY. Never re-exported from the package.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import frx.numpy as fnp
import numpy as np
from frx import Array
from python.runfiles import runfiles

from enc_frx.testing.kat import (
    AeadVector,
    AeadVectorSet,
    group_aead_by_shape,
    load_cavp_gcm,
)

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


def instance(
    files: dict[int, str], key_size: int, nonce_size: int, tag_size: int
) -> AeadVectorSet:
    """One scheme instance out of the file that publishes that key length.

    Takes the file *map* rather than a file, because one CAVP file is one key
    length: passing both would let a caller name a key size the file cannot hold.
    """
    name = files[key_size]
    for vector_set in sets(name):
        if (vector_set.nonce_size, vector_set.tag_size) == (nonce_size, tag_size):
            return vector_set
    raise AssertionError(
        f"{name} publishes no instance with a {nonce_size}-byte IV and a "
        f"{tag_size}-byte tag"
    )


def array(data: bytes) -> Array:
    """One byte string as a batch of one."""
    return fnp.asarray(np.frombuffer(data, dtype=np.uint8))[None]


def stack(items: Sequence[bytes]) -> Array:
    """Equal-length byte strings as a batch."""
    return fnp.asarray(
        np.stack([np.frombuffer(item, dtype=np.uint8) for item in items])
    )


def _gate_group(vector_set: AeadVectorSet) -> list[AeadVector]:
    """The shape group the gate runs: the smallest with both a payload and an AAD.

    Partitioned by the harness's own `group_aead_by_shape`, so the batch selected
    here is by construction a batch the driver will form rather than one that
    merely resembles it.

    Both parts non-empty so that all four of the harness's tamperings say
    something. A flipped associated-data byte is skipped when there is no
    associated data, and a flipped *first ciphertext byte* lands inside the tag
    when the payload is empty — which is the tag tampering over again rather than
    a second check.

    Smallest, because the gate runs on every PR and the tampering pass is
    quadratic in the group; the long payloads are the sweep's job.
    """
    groups = [
        group
        for group in group_aead_by_shape(vector_set.vectors)
        if group[0].associated_data and len(group[0].ciphertext) > vector_set.tag_size
    ]
    assert groups, "a CAVP instance publishes sections with both an AAD and a payload"
    return min(
        groups,
        key=lambda group: (
            len(group[0].associated_data or b""),
            len(group[0].ciphertext),
        ),
    )


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
