# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""K-PKE, the IND-CPA encryption underneath ML-KEM, per FIPS 203 §5.

The NTT, the wire formats and the samplers are pieces with no scheme. This is
where they assemble: `key_gen` expands a public matrix and samples a secret,
`encrypt` hides 32 bytes in the noise, and `decrypt` subtracts the secret back
out.

**It is its own layer because decapsulation calls `encrypt`.** The
Fujisaki-Okamoto transform re-encrypts the message it recovered and compares
ciphertexts, so `encrypt` is reached from `decaps` as well as from `encaps`.
Folded into `encaps`, it would have to be either duplicated or called backwards
out of `decaps`; FIPS 203 separates the two for the same reason.

**`encrypt` is deterministic in `r`, and the transform rests on that.** Every
value it needs — `y`, `e_1`, `e_2` — comes from `PRF(r, ·)`, and nothing here
draws randomness. An `encrypt` that reached for a generator anywhere would make
the re-encryption check never match, so `decaps` would answer every ciphertext,
valid ones included, with the implicit-rejection secret. That failure raises
nothing and passes every round trip: both sides still agree with themselves.

**No Bazel target outside `//enc_frx/ml_kem` may depend on this.** K-PKE is
IND-CPA and nothing more — its ciphertexts are malleable, and the
chosen-ciphertext attack that breaks it is exactly what the FO transform exists
to prevent, so the build target is the one in this package that is not public.

That fences the build graph, and only the build graph. The module ships in the
wheel like every other, so a determined `import enc_frx.ml_kem.k_pke` still
reaches it; what the visibility buys is that no target here acquires the
dependency by accident, and that adding one is a review-visible edit.

## Where the batch axis is

Everything is leading-axis, so a batch of `B` is one traced computation. See
`_matrix_vector` for the shape argument that makes the products `k`-independent.

## Two internals with their own entry points

`_key_pair` and `_noisy_message` are the algorithms' own step boundaries —
Algorithm 13 line 1 against the rest, and Algorithm 15 line 5 against line 6.
The published intermediate values land on those boundaries because that is where
the standard names its quantities, which is also why the tests enter there. Each
docstring says what its boundary is worth; neither exists only for a test.

## The parameters arrive per call

`k`, `eta1`, `eta2`, `du` and `dv` are arguments rather than fields of a
parameter-set object, which is how FIPS 203 states each algorithm's parameter
list too. Which named sets exist, and what they fix these to, is a layer above
this one.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.ml_kem import encoding, hashes, ntt, sampling
from enc_frx.ml_kem.params import SEED_SIZE, decryption_key_size


def _inner_product(a_hat: Array, b_hat: Array) -> Array:
    """`Σ_j â_j ∘ b̂_j`, contracting the `k` axis ahead of the coefficients.

    `∘` is FIPS 203 §2.4.7's NTT-domain product: 128 degree-1 multiplications,
    which is `ntt.base_mul` and emphatically not a pointwise multiply.
    """
    return fnp.sum(ntt.base_mul(a_hat, b_hat), axis=-2)


def _matrix_vector(matrix_hat: Array, vector_hat: Array) -> Array:
    """`[..., k, k, 256]` against `[..., k, 256]`, giving `[..., k, 256]`.

    Inserting an axis lines the vector up against every row at once, so the whole
    product is one `base_mul` and one sum no matter how large `k` is. A loop over
    the rows would issue `k` of each and serialize work that shares nothing.
    """
    return _inner_product(matrix_hat, vector_hat[..., None, :, :])


def key_gen(d: ArrayLike, *, k: int, eta1: int) -> tuple[Array, Array]:
    """FIPS 203 Algorithm 13 on `[..., 32]` seeds, to `(ek_PKE, dk_PKE)`.

    Deterministic in `d`: the standard defines key generation that way, and the
    known-answer tests reproduce published key bytes from a published seed.
    """
    seed = encoding.checked_length(d, SEED_SIZE, "a K-PKE key generation seed")
    # `(ρ, σ) ← G(d ‖ k)`, Algorithm 13 line 1: the parameter-set byte binds a
    # key to the `k` it was generated under.
    k_byte = fnp.full((*seed.shape[:-1], 1), k, dtype=fnp.uint8)
    rho, sigma = hashes.g(fnp.concatenate([seed, k_byte], axis=-1))
    return _key_pair(rho, sigma, k=k, eta1=eta1)


def _key_pair(rho: Array, sigma: Array, *, k: int, eta1: int) -> tuple[Array, Array]:
    """Algorithm 13 from the expanded seeds onward — everything but line 1.

    Its own entry point because the published intermediate values begin at `ρ`
    and `σ`, and the sets in this tree predate the `d ‖ k` amendment above and so
    cannot gate it. Splitting here lets the vectors gate the lattice work and
    leaves the expansion to a check of its own — see `docs/schemes/ml-kem.md`.
    """
    a_hat = sampling.expand_matrix(rho, k)
    # `s` takes nonces 0..k-1 and `e` takes k..2k-1, both at `eta1` and both from
    # `sigma` (Algorithm 13 lines 8-17), so one PRF call covers both vectors —
    # and one transform too, since `ntt` is batch-first over the leading axes.
    noise = ntt.ntt(
        sampling.sample_poly_cbd(
            hashes.prf(eta1, sigma[..., None, :], np.arange(2 * k, dtype=np.uint8)),
            eta1,
        )
    )
    s_hat, e_hat = noise[..., :k, :], noise[..., k:, :]
    t_hat = _matrix_vector(a_hat, s_hat) + e_hat
    return (
        encoding.encode_ek(ntt.as_ints(t_hat), rho),
        encoding.encode_vector(ntt.as_ints(s_hat), 12),
    )


def encrypt(
    ek_pke: ArrayLike,
    m: ArrayLike,
    r: ArrayLike,
    *,
    k: int,
    eta1: int,
    eta2: int,
    du: int,
    dv: int,
) -> Array:
    """FIPS 203 Algorithm 14: `[..., ek] x [..., 32] x [..., 32] -> [..., c]`.

    A function of its arguments and nothing else — see the module docstring for
    what silently breaks if that ever stops being true.
    """
    t_hat_ints, rho = encoding.decode_ek(ek_pke, k)
    t_hat = ntt.as_field(t_hat_ints)
    message = encoding.checked_length(m, SEED_SIZE, "a K-PKE message")
    seed = encoding.checked_length(r, SEED_SIZE, "K-PKE encryption randomness")[
        ..., None, :
    ]
    a_hat = sampling.expand_matrix(rho, k)

    # `y` at eta1 on nonces 0..k-1, then `e_1` and `e_2` at eta2 on k..2k
    # (Algorithm 14 lines 9-20). The counter runs across the two widths rather
    # than restarting, which is the part a per-vector transcription gets wrong.
    y_hat = ntt.ntt(
        sampling.sample_poly_cbd(
            hashes.prf(eta1, seed, np.arange(k, dtype=np.uint8)), eta1
        )
    )
    errors = sampling.sample_poly_cbd(
        hashes.prf(eta2, seed, np.arange(k, 2 * k + 1, dtype=np.uint8)), eta2
    )
    e_1, e_2 = errors[..., :k, :], errors[..., k, :]

    # `u ← NTT^-1(Â^T ∘ ŷ) + e_1`, Algorithm 14 line 21. `Â` is the matrix key
    # generation built, with the indices absorbed in the same order
    # (`sampling.py`); the transpose belongs to the product, not to the sampling.
    u = ntt.intt(_matrix_vector(fnp.swapaxes(a_hat, -3, -2), y_hat)) + e_1
    # `μ ← Decompress_1(ByteDecode_1(m))`, line 22: each message bit becomes 0 or
    # ⌈q/2⌋ = 1665, so the two are half the modulus apart. That gap is the margin
    # decryption's rounding has to survive.
    mu = ntt.as_field(encoding.decompress(encoding.byte_decode(message, 1), 1))
    v = ntt.intt(_inner_product(t_hat, y_hat)) + e_2 + mu
    return encoding.encode_ciphertext(ntt.as_ints(u), ntt.as_ints(v), du, dv)


def decrypt(dk_pke: ArrayLike, c: ArrayLike, *, k: int, du: int, dv: int) -> Array:
    """FIPS 203 Algorithm 15: `[..., dk] x [..., c] -> [..., 32]` message bytes."""
    return encoding.byte_encode(
        encoding.compress(ntt.as_ints(_noisy_message(dk_pke, c, k=k, du=du, dv=dv)), 1),
        1,
    )


def _noisy_message(
    dk_pke: ArrayLike, c: ArrayLike, *, k: int, du: int, dv: int
) -> Array:
    """`w = v' − NTT^-1(ŝ^T ∘ NTT(u'))`, Algorithm 15 line 5.

    What survives is `μ` plus the noise compression added, and `Compress_1` in
    `decrypt` rounds that away as long as the noise stayed under `q/4` — which is
    where ML-KEM's decryption failure probability comes from.

    Split out because that rounding hides its own input. Negating `w` shifts a
    `μ = 1665` coefficient to 1664, which still compresses to 1, so a subtraction
    written the wrong way round recovers the right message from every ciphertext
    anyone will ever hand it. Only the published `w` separates the two.
    """
    u, v = encoding.decode_ciphertext(c, k, du, dv)
    s_hat = ntt.as_field(
        encoding.decode_vector(
            encoding.checked_length(
                dk_pke, decryption_key_size(k), "a K-PKE decryption key"
            ),
            12,
            k,
        )
    )
    return ntt.as_field(v) - ntt.intt(_inner_product(s_hat, ntt.ntt(ntt.as_field(u))))
