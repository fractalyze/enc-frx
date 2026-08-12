# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 203's sampling, NTT and encoding algorithms in plain Python integers.

Algorithms 5-12 transcribed, one coefficient at a time, no arrays and no
vectorization. It exists to be *obviously* the specification, so a disagreement
with the traced implementation is a bug in the traced implementation rather than
a question about which convention either one meant.

That matters more here than for a keystream cipher, because a wrong NTT is not
obviously wrong: any primitive 256th root produces a transform that round-trips
and whose pointwise product is still a negacyclic convolution. Only the
standard's own `zeta = 17` and its `BitRev7` output order make it *this*
transform, and those are exactly what a plausible-looking implementation gets
wrong silently. Pinning against these functions is what catches that.

The sampling algorithms are the same kind of trap for a different reason. Both
read a byte stream as a bit stream, and both have a plausible wrong reading —
`SampleNTT` can pack its two candidates out of the middle byte the other way
round, and `SamplePolyCBD` can take its `2*eta` bits per coefficient in the wrong
order — and *neither* wrong reading changes the output distribution. So a
statistical check passes on both, and only equality with the standard separates
them.

`sample_ntt` here is written the way the standard writes it: a `while` that pulls
from an unbounded stream until 256 coefficients are collected. That is exactly
the control flow the traced implementation cannot have, which is the point — the
oracle is the definition, and how the traced code reaches the same answer under a
fixed budget is its own business.

TEST ONLY. Never re-exported from the package.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

Q = 3329
N = 256
ZETA = 17


def bit_rev7(i: int) -> int:
    """The 7-bit reversal FIPS 203 indexes its zeta table by."""
    return int(format(i, "07b")[::-1], 2)


def ntt(f: list[int]) -> list[int]:
    """Algorithm 9. Seven layers, landing on 128 degree-1 polynomials.

    Seven and not eight because `q - 1 = 2^8 * 13` has 2-adicity 8: there is a
    primitive 256th root of unity and no 512th, so the transform runs out of
    layers one short of scalars. The pairs it lands on are what `base_mul`
    exists for.
    """
    h = list(f)
    i = 1
    length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            zeta = pow(ZETA, bit_rev7(i), Q)
            i += 1
            for j in range(start, start + length):
                t = (zeta * h[j + length]) % Q
                h[j + length] = (h[j] - t) % Q
                h[j] = (h[j] + t) % Q
        length //= 2
    return h


def intt(f_hat: list[int]) -> list[int]:
    """Algorithm 10, including the trailing 128^-1 scale."""
    f = list(f_hat)
    i = 127
    length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            zeta = pow(ZETA, bit_rev7(i), Q)
            i -= 1
            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % Q
                f[j + length] = (zeta * (f[j + length] - t)) % Q
        length *= 2
    inv128 = pow(128, -1, Q)
    return [(x * inv128) % Q for x in f]


def base_case_multiply(
    a0: int, a1: int, b0: int, b1: int, gamma: int
) -> tuple[int, int]:
    """Algorithm 12 — the degree-1 product mod `X^2 - gamma`."""
    c0 = (a0 * b0 + a1 * b1 % Q * gamma) % Q
    c1 = (a0 * b1 + a1 * b0) % Q
    return c0, c1


def multiply_ntts(f_hat: list[int], g_hat: list[int]) -> list[int]:
    """Algorithm 11 — 128 independent degree-1 products, not a pointwise mul.

    This is the step that makes ML-KEM's NTT domain unlike ML-DSA's, where the
    complete transform does reduce multiplication to a pointwise product.
    """
    h = [0] * N
    for i in range(128):
        gamma = pow(ZETA, 2 * bit_rev7(i) + 1, Q)
        h[2 * i], h[2 * i + 1] = base_case_multiply(
            f_hat[2 * i], f_hat[2 * i + 1], g_hat[2 * i], g_hat[2 * i + 1], gamma
        )
    return h


def negacyclic_convolution(f: list[int], g: list[int]) -> list[int]:
    """`f * g` in `Z_q[X]/(X^256 + 1)`, by definition.

    Independent of the transform entirely, so it pins `ntt`/`multiply_ntts`/
    `intt` as a composition rather than pinning each against itself.
    """
    out = [0] * N
    for i, fi in enumerate(f):
        for j, gj in enumerate(g):
            k = i + j
            if k < N:
                out[k] = (out[k] + fi * gj) % Q
            else:
                out[k - N] = (out[k - N] - fi * gj) % Q
    return [x % Q for x in out]


# --- §4.2.1 compression and Algorithms 5-6, in plain integers -----------------
#
# The rounding is the reason these exist separately from the traced code rather
# than being asserted against it: `round-half-up over integers` and
# `round(x * 2**d / q)` agree on almost every input and disagree at the ties,
# and the disagreement is invisible to a round trip.


def compress(x: int, d: int) -> int:
    """`⌈(2^d / q) · x⌋ mod 2^d`, round half up, exactly."""
    return ((2 * x * (1 << d) + Q) // (2 * Q)) % (1 << d)


def decompress(y: int, d: int) -> int:
    """`⌈(q / 2^d) · y⌋`, round half up, exactly."""
    return (2 * y * Q + (1 << d)) // (1 << (d + 1))


def byte_encode(f: list[int], d: int) -> list[int]:
    """Algorithm 5 — little-endian bits within a coefficient and within a byte."""
    bits = [(v >> j) & 1 for v in f for j in range(d)]
    return [sum(bits[i * 8 + j] << j for j in range(8)) for i in range(len(bits) // 8)]


def byte_decode(b: list[int], d: int) -> list[int]:
    """Algorithm 6 — reduces mod q at d = 12, mod 2^d below it."""
    bits = [(byte >> j) & 1 for byte in b for j in range(8)]
    m = Q if d == 12 else (1 << d)
    return [
        sum(bits[i * d + j] << j for j in range(d)) % m for i in range(len(bits) // d)
    ]


# --- §4.2.2 sampling, Algorithms 7-8 ------------------------------------------
#
# Both read bytes as bits, and both have a plausible wrong reading that leaves
# the output distribution unchanged — so a chi-squared test passes on the wrong
# one and only equality with the standard separates them. That is what these
# transcriptions are for.


def shake128_stream(seed: bytes, chunk: int = 168) -> Iterator[int]:
    """SHAKE128(`seed`) as an unbounded byte stream, one byte at a time.

    Genuinely unbounded, which is what lets `sample_ntt` below be the standard's
    `while` rather than a budgeted approximation of it. SHAKE's output is
    prefix-consistent — `digest(n)` is the first `n` bytes of one stream — so
    re-squeezing a longer digest and yielding the tail is the same stream. The
    quadratic re-hashing that costs is irrelevant at these lengths and buys the
    property the oracle exists to have.
    """
    taken = 0
    while True:
        taken += chunk
        block = hashlib.shake_128(seed).digest(taken)
        yield from block[taken - chunk :]


def sample_ntt(stream: Iterator[int]) -> list[int]:
    """Algorithm 7 — rejection sampling, straight from the standard.

    Three bytes give two 12-bit candidates and each is kept iff it is below `q`,
    so how many bytes this consumes depends on their values. The unbounded
    `while` is the whole reason `sampling.py` cannot be a transcription of this:
    a traced program has no data-dependent trip count.

    The bit split of the middle byte is the trap. `d1` takes its **low** nibble
    as the high bits and `d2` its **high** nibble as the low bits; swapping them
    samples uniformly from the same range and produces a different, entirely
    plausible array.

    Returns coefficients already in the NTT domain — the name says `NTT` because
    the output *is* `â`, not because a transform is applied here.
    """
    a_hat: list[int] = []
    while len(a_hat) < N:
        b0, b1, b2 = next(stream), next(stream), next(stream)
        d1 = b0 + 256 * (b1 % 16)
        d2 = (b1 // 16) + 16 * b2
        if d1 < Q:
            a_hat.append(d1)
        if d2 < Q and len(a_hat) < N:
            a_hat.append(d2)
    return a_hat


def sample_poly_cbd(b: list[int], eta: int) -> list[int]:
    """Algorithm 8 — the centered binomial distribution, `eta` in {2, 3}.

    No rejection: `64*eta` bytes in, 256 coefficients out. Each coefficient is
    `x - y` where `x` and `y` count the set bits in two adjacent `eta`-bit
    windows, so it lands in `[-eta, eta]` and is reduced mod `q`.

    The bit order is little-endian *within a byte* — bit `8i + j` of the stream
    is bit `j` of byte `i` — which `byte_decode` above reads the same way and
    which is the reading a big-endian transcription gets wrong without changing
    the distribution.
    """
    bits = [(byte >> j) & 1 for byte in b for j in range(8)]
    f = []
    for i in range(N):
        x = sum(bits[2 * i * eta + j] for j in range(eta))
        y = sum(bits[2 * i * eta + eta + j] for j in range(eta))
        f.append((x - y) % Q)
    return f


def sample_matrix(rho: bytes, k: int) -> list[list[list[int]]]:
    """`Â[i][j] = SampleNTT(rho ‖ j ‖ i)`, per Algorithms 13 and 14.

    **The column index is absorbed first.** Both key generation and encryption
    build the same `Â` this way and encryption then uses its transpose; deriving
    `Â[i][j]` from `rho ‖ i ‖ j` instead produces that transpose, which is a
    self-consistent scheme that fails every published vector and nothing else.
    """
    return [
        [sample_ntt(shake128_stream(rho + bytes([j, i]))) for j in range(k)]
        for i in range(k)
    ]
