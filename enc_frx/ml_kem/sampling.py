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
slot holds zero and the result is a wrong array — the same wrong array on every
backend and in both compaction forms below, which each pin that edge themselves
rather than leaving an out-of-range index to XLA. Nothing detects it, and nothing
should: a validity flag threaded through the whole scheme for a 2^-261 event
would be dead weight that the seams would have to carry, and `Kem.decaps` has no
failure channel to carry it in anyway.

## Compaction is a scatter, or a search, and the backend decides

Selecting the accepted candidates in order is a stream compaction, and
`rank = cumsum(accepted)` is what orders it: the candidate at position `i`, if
accepted, belongs in output slot `rank[i] - 1`. There are two ways to act on
that, and they differ in which side they iterate.

- **Scatter** — write each accepted candidate to the slot its rank names. One
  pass over the 560 candidates, at an index the cumsum already produced, with
  writes that are disjoint by construction.
- **Search** — ask, for each of the 256 output slots, which candidate feeds it.
  The answer is `#{i : rank[i] <= t}`, which `searchsorted` gives and
  `take_along_axis` then reads. It costs a binary search per slot and uses
  nothing of what makes `rank` special, which is that it steps by one.

The textbook reason to prefer the search is that a large scatter serializes.
That is a statement about a backend and not about this compaction, so it is read
off the backend rather than assumed. On GPU the scatter is the whole reason this
stage stopped being the expensive one: at `B = 8192` it takes `SampleNTT` from
2.01 ms to 1.02 ms and a whole ML-KEM-768 decapsulation from 3.06 ms to 1.76 ms,
and it retires a >1 s constant-fold of the search's unrolled index arithmetic at
compile time. On CPU the same substitution costs 1.42× at `B = 256` and only
repays above roughly four times that, so the CPU keeps the search.

Both forms return the same array for every input, including under the miss
above, and `CompactionFormTest` is what holds them to that rather than the claim
being made here. Neither needs a `while`, which is not negotiable in this module:
the scatter's trip count is the candidate count and the search is asked for the
unrolled binary variant, so both are fixed. `TracedShapeTest` reads the form back
off the lowering, because the source cannot show it.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.ml_kem import hashes
from enc_frx.ml_kem.encoding import bytes_to_bits, checked_length
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
    groups = stream.astype(np.int32).reshape(
        *stream.shape[:-1], GROUPS, BYTES_PER_GROUP
    )
    b0, b1, b2 = groups[..., 0], groups[..., 1], groups[..., 2]
    d1 = b0 + np.int32(256) * (b1 & np.int32(0x0F))
    d2 = (b1 >> np.int32(4)) + np.int32(16) * b2
    return fnp.stack([d1, d2], axis=-1).reshape(*stream.shape[:-1], CANDIDATES)


# Which backends compact by scattering. A tuple rather than a test against `cpu`,
# and for the same reason `hash_frx.keccak.permutation` keeps one for its emitter:
# a backend is on the search path until the scatter is *measured* to win on it, so
# a leg that earns the scatter joins this tuple and nothing else here moves.
_SCATTER_BACKENDS = ("gpu",)


def _compacts_by_scatter() -> bool:
    """Whether this backend writes each candidate out or searches for it.

    Read per call rather than at import, so importing this module does not
    initialize a backend and so a test can pin either answer. The lookup behind
    `frx.default_backend()` is memoized, so it is cheap enough to sit on the
    tracing path of every matrix expansion.
    """
    return frx.default_backend() in _SCATTER_BACKENDS


def _scattered(candidates: Array, accepted: Array, rank: Array) -> Array:
    """Each accepted candidate written to the slot its rank names.

    **The drop is not redundant.** A rejected candidate has no slot, and under
    the 2^-261 miss neither has anything from slot 256 up; both are aimed at `N`,
    one past the end, where `mode="drop"` discards them rather than letting XLA
    decide what an out-of-range write does. What is left in an unwritten slot is
    the zero its row started at, which is what `_searched` puts there too.

    Seeding the row with anything the input has to be read for — the final
    candidate, say — would cost more than it looks: it keeps `candidates` live
    across the write and stops the expansion fusing into the program around it,
    which is paid whole even though this function on its own gets slightly
    faster.
    """
    slot = fnp.where(accepted, (rank - np.int16(1)).astype(np.int32), np.int32(N))

    def row(values: Array, target: Array) -> Array:
        return fnp.zeros(N, dtype=values.dtype).at[target].set(values, mode="drop")

    return frx.vmap(row)(candidates, slot)


def _searched(candidates: Array, rank: Array) -> Array:
    """For each slot, the candidate feeding it, found by binary search.

    **Neither the clamp nor the select is redundant.** Under the 2^-261 miss,
    `searchsorted` returns `CANDIDATES` for every unfilled slot — one past the
    end. The clamp is what keeps the read itself in bounds, because an
    out-of-range gather is the backend's to define and it is not the intuitive
    one: it fills with `INT32_MIN` rather than clamping. The select is then what
    decides the value, and it says zero, because that is what `_scattered` leaves
    in the same slots and the module docstring promises the two forms agree.
    """
    slots = fnp.arange(N, dtype=np.int16)
    picked = frx.vmap(
        lambda row: fnp.searchsorted(row, slots, side="right", method="scan_unrolled")
    )(rank)
    clamped = fnp.minimum(picked, np.int32(CANDIDATES - 1))
    taken = fnp.take_along_axis(candidates, clamped, axis=-1)
    return fnp.where(picked < np.int32(CANDIDATES), taken, np.int32(0))


def _compact(candidates: Array) -> Array:
    """The first 256 candidates below `q`, in order: `[B, 560] -> [B, 256]`.

    `rank` is the running count of acceptances, so the candidate at position `i`,
    if accepted, belongs in output slot `rank[i] - 1`. Both forms below answer
    that and return the same array; the module docstring says why which one runs
    is the backend's answer rather than this module's.
    """
    accepted = candidates < np.int32(Q)
    # `int16` because `rank` counts acceptances among `CANDIDATES` of them and so
    # cannot leave `[0, 560]`: the search form compares the whole rank row against
    # every one of the 256 slots, and halving its width halves that traffic.
    rank = fnp.cumsum(accepted.astype(np.int16), axis=-1)
    if _compacts_by_scatter():
        return _scattered(candidates, accepted, rank)
    return _searched(candidates, rank)


def sample_ntt(seeds: ArrayLike) -> Array:
    """FIPS 203 Algorithm 7 on `[..., 34]` seeds, to `[..., 256]` field elements.

    The output is already in the NTT domain — the name is what the array *is*,
    not a transform applied here — so it comes back as field elements ready for
    `base_mul`, never as integers needing a later crossing.
    """
    seed_array = checked_length(
        seeds, MATRIX_SEED_SIZE, "a SampleNTT seed (rho ‖ j ‖ i)"
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
    seed = checked_length(rho, SEED_SIZE, "rho")
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
    array = checked_length(
        data, hashes.CBD_BYTES_PER_ETA * eta, f"SamplePolyCBD_{eta} input"
    )
    bits = bytes_to_bits(array)
    windows = bits.reshape(*array.shape[:-1], N, 2, eta)
    counts = windows.sum(axis=-1)
    return as_field(counts[..., 0]) - as_field(counts[..., 1])
