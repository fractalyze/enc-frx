# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM's two samplers, per FIPS 203 §4.2.2.

`SampleNTT` expands the public matrix `Â` from a seed; `SamplePolyCBD_eta` draws
the secret and error vectors. They look alike and are not, and the difference is
**what their seed is**:

- `SampleNTT`'s seed is `rho`, which ships in the encapsulation key. It is
  public. So how many XOF bytes it consumes, and how long it takes, reveal
  nothing an attacker does not already hold — which is what makes the fixed
  budget below a throughput choice rather than a security one.
- `SamplePolyCBD`'s input is `PRF_eta(sigma, N)` output, derived from the secret
  seed. It therefore has no rejection step at all and must stay branch-free: a
  fixed `64*eta` bytes in, 256 coefficients out, pure bit arithmetic.

That asymmetry is the standard's design, not an implementation liberty, and it
is the reason only one of the two needs the treatment below.

## The fixed block, and why 5

`SampleNTT` is rejection sampling: three XOF bytes give two 12-bit candidates,
each kept iff it is below `q = 3329`, until 256 are collected. The bytes consumed
depend on their own values, and **a traced program has no data-dependent trip
count** — so the standard's `while` cannot be transcribed. What replaces it is a
fixed squeeze, masked and compacted.

Acceptance is `3329/4096 ~ 0.8127` per candidate, so 256 coefficients need ~315
candidates on average. Sizing the budget from the mean would fail constantly; it
comes from the tail instead, at `P(Binomial(n, 0.8127) < 256)`:

| SHAKE128 blocks | bytes | candidates | P(budget too small) |
|---|---|---|---|
| 3 | 504 | 336 | 2^-6.9 |
| 4 | 672 | 448 | 2^-105 |
| **5** | **840** | **560** | **2^-261** |

Five, because it puts the miss below 2^-256 rather than merely below 2^-128 —
under the bar for ML-KEM-1024's category-5 claim, not just for the smallest
parameter set — and costs one Keccak permutation per matrix entry over four.

The bound is not only computed. C2SP's `unluckysample` vectors are seeds found
by search whose rejection run is the worst known, and they consume **576 bytes /
384 candidates**, which three blocks does not cover and five clears with 176
candidates to spare. Those vectors are in the suite for exactly that reason: an
undersized budget passes every ordinary vector ever published and fails only
there.

**A miss is deterministic, not undefined.** Below 256 acceptances every unfilled
slot reads the final candidate and the result is a wrong array — the same wrong
array on every backend, because `_compact` clamps the index itself rather than
leaving the gather's out-of-bounds behaviour to XLA. Nothing detects it, and
nothing should: a validity flag threaded through the whole scheme for a 2^-261
event would be dead weight that the seams would have to carry, and `Kem.decaps`
has no failure channel to carry it in anyway.

## Compaction is a gather

Selecting the accepted candidates in order is a stream compaction, and the
obvious form — scatter each accepted value to its rank — is the wrong one here:
XLA serializes a large scatter on GPU, so it costs far more than the sampling it
serves. The inverse is a gather. With `rank = cumsum(accepted)`, the source index
of output slot `t` is `#{i : rank[i] <= t}`, which `searchsorted` answers
directly, and `take_along_axis` then reads the values. No scatter, no sort, and
no `while` — `searchsorted` is asked for the unrolled binary search so its trip
count is static too.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.ml_kem import hashes
from enc_frx.ml_kem.encoding import bytes_to_bits
from enc_frx.ml_kem.ntt import as_field
from enc_frx.ml_kem.params import SEED_SIZE, N, Q

# Five blocks — see the module docstring for where the number comes from. Whole
# blocks because a partial one costs the same permutation as a full one, and the
# rate comes from `hashes` rather than being restated here.
XOF_BLOCKS = 5
XOF_BYTES = hashes.XOF_RATE * XOF_BLOCKS

# Three bytes give two candidates, and 840 divides by three exactly.
BYTES_PER_GROUP = 3
GROUPS = XOF_BYTES // BYTES_PER_GROUP
CANDIDATES = 2 * GROUPS

# `SampleNTT`'s seed is `rho ‖ j ‖ i`: 32 bytes plus two index bytes.
MATRIX_SEED_SIZE = SEED_SIZE + 2


def _candidates(stream: Array) -> Array:
    """`[B, 840]` XOF bytes to the `[B, 560]` 12-bit candidates of Algorithm 7.

    The middle byte is split across both candidates and **the halves are not
    interchangeable**: `d1` takes its *low* nibble as high bits, `d2` its *high*
    nibble as low bits. Swapping them still samples uniformly from `[0, 2^12)`,
    so the output stays plausible and every statistical check still passes.
    """
    groups = stream.astype(np.int32).reshape(*stream.shape[:-1], GROUPS, 3)
    b0, b1, b2 = groups[..., 0], groups[..., 1], groups[..., 2]
    d1 = b0 + np.int32(256) * (b1 & np.int32(0x0F))
    d2 = (b1 >> np.int32(4)) + np.int32(16) * b2
    return fnp.stack([d1, d2], axis=-1).reshape(*stream.shape[:-1], CANDIDATES)


def _compact(candidates: Array) -> Array:
    """The first 256 candidates below `q`, in order: `[B, 560] -> [B, 256]`.

    `rank` is the running count of acceptances, so it is non-decreasing and the
    first index at which it exceeds `t` is where output slot `t` comes from.
    Asking `searchsorted` on the right side answers that for all 256 slots at
    once.

    **The clamp is not redundant.** Under the 2^-261 miss, `searchsorted` returns
    `CANDIDATES` for every unfilled slot — one past the end. Left alone, the
    gather's out-of-bounds behaviour is the backend's to define, and it is not
    the intuitive one: it fills with `INT32_MIN` rather than clamping. Pinning
    the index here makes the miss read the final candidate instead, which is what
    lets the module docstring promise the same wrong array everywhere rather than
    whatever each backend does at the edge.
    """
    rank = fnp.cumsum((candidates < np.int32(Q)).astype(np.int32), axis=-1)
    slots = fnp.arange(N, dtype=np.int32)
    picked = frx.vmap(
        lambda row: fnp.searchsorted(row, slots, side="right", method="scan_unrolled")
    )(rank)
    clamped = fnp.minimum(picked, np.int32(CANDIDATES - 1))
    return fnp.take_along_axis(candidates, clamped, axis=-1)


def sample_ntt(seeds: ArrayLike) -> Array:
    """FIPS 203 Algorithm 7 on `[..., 34]` seeds, to `[..., 256]` field elements.

    The output is already in the NTT domain — the name is what the array *is*,
    not a transform applied here — so it comes back as field elements ready for
    `base_mul`, never as integers needing a later crossing.
    """
    seed_array = fnp.asarray(seeds).astype(np.uint8)
    if seed_array.shape[-1] != MATRIX_SEED_SIZE:
        raise ValueError(
            f"SampleNTT seed is {MATRIX_SEED_SIZE} bytes "
            f"(rho ‖ j ‖ i), got {seed_array.shape[-1]}"
        )
    lead = seed_array.shape[:-1]
    stream = hashes.xof(XOF_BYTES, seed_array.reshape(-1, MATRIX_SEED_SIZE))
    coefficients = _compact(_candidates(stream))
    return as_field(coefficients.reshape(*lead, N))


def expand_matrix(rho: ArrayLike, k: int) -> Array:
    """`Â` from `rho`: `[..., 32] -> [..., k, k, 256]` field elements.

    **The column index is absorbed first** — `Â[i, j] = SampleNTT(rho ‖ j ‖ i)`,
    per Algorithms 13 and 14. Feeding `rho ‖ i ‖ j` builds the transpose, which
    is a self-consistent scheme that encapsulates and decapsulates against itself
    perfectly and fails every published vector.

    All `k^2` entries are one XOF call rather than `k^2` sequential ones: they
    share nothing but the seed prefix, so the only thing that would serialize
    them is writing the loop.
    """
    seed = fnp.asarray(rho).astype(np.uint8)
    if seed.shape[-1] != SEED_SIZE:
        raise ValueError(f"rho is {SEED_SIZE} bytes, got {seed.shape[-1]}")
    lead = seed.shape[:-1]
    # `indices[i, j] = (j, i)`, the standard's order, built on the host because
    # `k` is static.
    indices = np.array(
        [[[j, i] for j in range(k)] for i in range(k)], dtype=np.uint8
    ).reshape(k * k, 2)
    seeds = fnp.concatenate(
        [
            fnp.broadcast_to(seed[..., None, :], (*lead, k * k, SEED_SIZE)),
            fnp.broadcast_to(fnp.asarray(indices), (*lead, k * k, 2)),
        ],
        axis=-1,
    )
    return sample_ntt(seeds).reshape(*lead, k, k, N)


def sample_poly_cbd(data: ArrayLike, eta: int) -> Array:
    """FIPS 203 Algorithm 8 on `[..., 64*eta]` bytes, to `[..., 256]` elements.

    Each coefficient is `x - y`, the difference of the set-bit counts of two
    adjacent `eta`-bit windows, so it lands in `[-eta, eta]` before reduction.
    No rejection and no branch: this one's input is secret-derived.

    Reduction is left to the field dtype rather than done as `% q` on integers —
    `x - y` is negative for half the draws, and a signed remainder is the step an
    integer path gets wrong in exactly the cases a round trip cannot see.
    """
    if eta not in hashes.ETAS:
        raise ValueError(f"FIPS 203 uses eta in {hashes.ETAS}, got {eta}")
    array = fnp.asarray(data).astype(np.uint8)
    expected = hashes.CBD_BYTES_PER_ETA * eta
    if array.shape[-1] != expected:
        raise ValueError(
            f"SamplePolyCBD_{eta} takes {expected} bytes, got {array.shape[-1]}"
        )
    bits = bytes_to_bits(array)
    windows = bits.reshape(*array.shape[:-1], N, 2, eta)
    counts = windows.sum(axis=-1)
    return as_field(counts[..., 0]) - as_field(counts[..., 1])
