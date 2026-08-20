# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Reaching ACVP's AES-CCM set, and picking gate batches out of it.

Shared by the gate (`ccm_test`) and the sweep (`ccm_sweep_test`), which run
the same vectors at different depths on different schedules — the split
`gcm_vectors.py` explains.

TEST ONLY. Never re-exported from the package.
"""

from __future__ import annotations

import functools

from python.runfiles import runfiles

from enc_frx.testing.kat import (
    AeadVector,
    AeadVectorSet,
    group_aead_by_shape,
    load_acvp_ccm,
)


def _path(repo: str, name: str) -> str:
    location = runfiles.Create().Rlocation(f"{repo}/file/{name}")
    assert location is not None, f"{repo}/{name} not in runfiles"
    return location


@functools.cache
def sets() -> tuple[AeadVectorSet, ...]:
    """Every scheme instance the ACVP pair publishes — one per
    (key, nonce, tag) size triple, 147 in all."""
    return tuple(
        load_acvp_ccm(
            _path("acvp_aes_ccm_prompt", "prompt.json"),
            _path("acvp_aes_ccm_expected", "expectedResults.json"),
        )
    )


def instance(key_size: int, nonce_size: int, tag_size: int) -> AeadVectorSet:
    for vector_set in sets():
        if (vector_set.key_size, vector_set.nonce_size, vector_set.tag_size) == (
            key_size,
            nonce_size,
            tag_size,
        ):
            return vector_set
    raise AssertionError(
        f"ACVP publishes no CCM instance with a {key_size}-byte key, "
        f"{nonce_size}-byte nonce and {tag_size}-byte tag"
    )


def gate_batch(vector_set: AeadVectorSet) -> list[AeadVector]:
    """What the per-PR gate runs for one instance: the smallest shape group
    carrying both an AAD and a payload (so all four tamperings bite — the
    rationale `gcm_vectors._gate_group` states), plus the smallest group
    carrying a published rejection, so the mixed-validity masking path runs
    against the standard's own failures rather than only synthesized ones."""
    groups = group_aead_by_shape(vector_set.vectors)
    tampering = [
        group
        for group in groups
        if group[0].associated_data and len(group[0].ciphertext) > vector_set.tag_size
    ]
    assert tampering, "ACVP publishes sections with both an AAD and a payload"
    chosen = min(
        tampering,
        key=lambda group: (
            len(group[0].associated_data or b""),
            len(group[0].ciphertext),
        ),
    )
    mixed = [group for group in groups if any(not vector.valid for vector in group)]
    if mixed:
        chosen = chosen + min(mixed, key=lambda group: len(group[0].ciphertext))
    return chosen
