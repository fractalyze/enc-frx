# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""FIPS 203's NTT algorithms in plain Python integers — the oracle.

Algorithms 9-12 transcribed, one coefficient at a time, no arrays and no
vectorization. It exists to be *obviously* the specification, so a disagreement
with the traced implementation is a bug in the traced implementation rather than
a question about which convention either one meant.

That matters more here than for a keystream cipher, because a wrong NTT is not
obviously wrong: any primitive 256th root produces a transform that round-trips
and whose pointwise product is still a negacyclic convolution. Only the
standard's own `zeta = 17` and its `BitRev7` output order make it *this*
transform, and those are exactly what a plausible-looking implementation gets
wrong silently. Pinning against these functions is what catches that.

TEST ONLY. Never re-exported from the package.
"""

from __future__ import annotations

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
