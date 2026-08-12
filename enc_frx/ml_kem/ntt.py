# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM's NTT over `Z_q[X]/(X^256 + 1)`, `q = 3329`, per FIPS 203 §4.3.

**The transform is `frx.lax.ntt`; this module is the caller.** FIPS 203's
"incomplete" NTT is exactly two length-128 negacyclic transforms over the even
and odd coefficient halves — write `f(X) = f_e(X^2) + X * f_o(X^2)` and reduce
mod `X^2 - zeta_j`, which sends `X^2 -> zeta_j`, so evaluating `f_e` and `f_o` at
the 128 roots of `Y^128 + 1` *is* a length-128 negacyclic NTT of each. So there
is no layer walk here, no zeta table, no Montgomery or Barrett reduction: the
field dtype reduces internally, and no residue ever occupies a raw integer lane.

Seven layers rather than eight is forced, not chosen. `q - 1 = 2^8 * 13` has
2-adicity 8, so `Z_q` has a primitive 256th root of unity and no 512th — and a
length-`n` negacyclic transform needs a primitive `2n`-th root. Length 256 is
therefore impossible over this field at any generator, which the opcode says in
those words if asked. Length 128 is exactly what 2-adicity 8 supports.

**Two things must be pinned or this computes a different, valid transform.**
Both are silent failures: any primitive 256th root gives something that
round-trips and whose pointwise product is still a negacyclic convolution, so
nothing downstream would catch the mismatch with the standard.

1. *The root.* Unpinned, the opcode searches for a primitive root and finds one.
   FIPS 203 fixes `zeta = 17`. A length-128 negacyclic takes the 256th root
   `psi = g^((q-1)/256) = g^13`, so the generator to pass is the one whose 13th
   power is 17 — and because `gcd(13, 256) == 1` the 13th-power map is a
   bijection on the order-256 subgroup, so it inverts exactly (`_GENERATOR`).
2. *The order.* The opcode returns natural order; FIPS 203 indexes its 128
   degree-1 polynomials by `BitRev7`. `_BIT_REV7` applies that, and is its own
   inverse because bit reversal is an involution.

`base_mul` is the part the opcode does not do and the standard does specify: a
2x2 product mod `X^2 - zeta^(2*BitRev7(i)+1)`, not a pointwise multiply. It is
the single most common place to get ML-KEM wrong by pattern-matching an ML-DSA
implementation, whose complete transform *does* reduce multiplication to a
pointwise product.

Everything is batch-first over leading axes, so a `k x k` matrix of polynomials
is two opcode calls rather than `2k^2`.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from frx.typing import ArrayLike

from enc_frx.ml_kem.params import ZETA, N, Q

# Coefficients live in the field rather than in integer lanes, so every product
# is reduced by the dtype and the 16-bit-lane question never arises.
PRIME_FIELD = zk_dtypes.prime_field(Q)

_HALF = N // 2

# psi = g^((q-1)/256) = g^13, and we need psi == ZETA. See the module docstring.
_GENERATOR = pow(ZETA, pow((Q - 1) // N, -1, N), Q)

# Bit reversal is an involution, so this permutation is its own inverse and the
# forward and inverse transforms apply the same array.
_BIT_REV7 = np.array(
    [int(format(i, "07b")[::-1], 2) for i in range(_HALF)], dtype=np.int32
)

# gamma_i = zeta^(2*BitRev7(i)+1), the modulus of the i-th degree-1 factor.
_GAMMA = np.array([pow(ZETA, 2 * int(r) + 1, Q) for r in _BIT_REV7], dtype=np.int64)


def as_field(coeffs: ArrayLike) -> Array:
    """Integer coefficients to field elements, the boundary conversion."""
    return fnp.asarray(coeffs, dtype=PRIME_FIELD)


def as_ints(poly: Array) -> Array:
    """Field elements back to canonical integers, for packing and comparison.

    `int32` because every residue is below 3329 and x64 is off by default, where
    asking for `int64` truncates with a warning rather than widening.
    """
    return fnp.asarray(poly).astype(np.int32) % Q


def _transform_halves(halves: Array, *, inverse: bool) -> Array:
    """Both halves in one opcode call — they are two entries of its batch axis.

    The opcode transforms the minor dimension and treats everything ahead of it
    as a batch, so `[..., 2, 128]` is one call rather than two. Splitting it
    would double the op count on the hottest data in the scheme for no gain:
    the two halves share a root table and never interact.
    """
    ntt_type = (
        frx.lax.NttType.NEGACYCLIC_INTT if inverse else frx.lax.NttType.NEGACYCLIC_NTT
    )
    return frx.lax.ntt(halves, ntt_type=ntt_type, generator=_GENERATOR)


def _split_halves(f: Array) -> Array:
    """`[..., 256] -> [..., 2, 128]`, even coefficients first.

    A reshape and a transpose rather than two strided slices and a stack: the
    pairs `(f[2i], f[2i+1])` are already adjacent, so the de-interleave is a
    view of the same buffer.
    """
    paired = f.reshape(*f.shape[:-1], _HALF, 2)
    return fnp.swapaxes(paired, -1, -2)


def ntt(f: Array) -> Array:
    """FIPS 203 Algorithm 9, on `[..., 256]` field elements.

    Byte-identical to the standard, not merely a valid transform — see the
    module docstring for the two things that makes true.
    """
    transformed = _transform_halves(_split_halves(f), inverse=False)
    return _interleave(
        transformed[..., 0, :][..., _BIT_REV7],
        transformed[..., 1, :][..., _BIT_REV7],
    )


def intt(f_hat: Array) -> Array:
    """FIPS 203 Algorithm 10. Inverse of `ntt`, including the `128^-1` scale.

    The scale rides in the opcode's inverse direction — a length-128 transform
    normalizes by its own length, which is the 128 the standard applies.
    """
    paired = _split_halves(f_hat)
    halves = fnp.stack(
        [paired[..., 0, :][..., _BIT_REV7], paired[..., 1, :][..., _BIT_REV7]],
        axis=-2,
    )
    transformed = _transform_halves(halves, inverse=True)
    return _interleave(transformed[..., 0, :], transformed[..., 1, :])


def base_mul(f_hat: Array, g_hat: Array) -> Array:
    """FIPS 203 Algorithms 11 and 12 — 128 degree-1 products, batch-first.

    `(a0 + a1 X)(b0 + b1 X) mod (X^2 - gamma)`, so the `a1 b1` term folds back
    scaled by gamma instead of vanishing. A pointwise multiply here is the
    ML-DSA-shaped mistake and produces wrong ciphertexts, not an error.
    """
    a0, a1 = f_hat[..., 0::2], f_hat[..., 1::2]
    b0, b1 = g_hat[..., 0::2], g_hat[..., 1::2]
    gamma = as_field(_GAMMA)
    return _interleave(a0 * b0 + a1 * b1 * gamma, a0 * b1 + a1 * b0)


def _interleave(even: Array, odd: Array) -> Array:
    """`[..., 128] x [..., 128] -> [..., 256]` with `even` on the even indices."""
    stacked = fnp.stack([even, odd], axis=-1)
    return stacked.reshape(*stacked.shape[:-2], N)
