# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM's four hash functions and its XOF, per FIPS 203 §4.1.

`G`, `H`, `J` and `PRF_eta` are SHA3-512, SHA3-256 and two SHAKE256 instances;
the matrix expansion's XOF is SHAKE128. **Every one of them reaches Keccak
through `hash-frx`'s `ByteHash` seam, never a local sponge.** Byte-exactness for
SHA-3 is established once, upstream, against `hashlib`; a second copy here would
be a second thing to establish it for and a second thing to get wrong.

The seam takes and returns a `[B, L]` byte batch, so each wrapper flattens the
leading axes and restores them. That is the whole content of this module beyond
naming: a scheme-level name, the output length the standard fixes, and the shape
bookkeeping.

**An XOF's output length is part of which hash it is**, not a request made of
one — so `Shake256(output_size=32)` and `Shake256(output_size=128)` are two
instances, and `PRF_eta` builds a different one per `eta`. That is hash-frx's
rule and the reason these are constructed per call site rather than hoisted to
one module-level SHAKE256.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx.keccak.byte_hashes import (
    SHAKE128_RATE,
    Sha3_256,
    Sha3_512,
    Shake128,
    Shake256,
)

from enc_frx.ml_kem.params import SEED_SIZE

# Re-exported rather than restated: `sampling.py` sizes its XOF budget in whole
# SHAKE128 blocks and must not carry a second copy of the rate. This module is
# the one place that names Keccak, so it is where the number crosses.
XOF_RATE = SHAKE128_RATE

# FIPS 203 §4.1: `PRF_eta(s, b)` squeezes `64*eta` bytes, and `eta` is 2 or 3.
CBD_BYTES_PER_ETA = 64
ETAS = (2, 3)


def _digest(hash_: Sha3_256 | Sha3_512 | Shake128 | Shake256, data: ArrayLike) -> Array:
    """Run one `ByteHash` over `[..., L]` bytes, restoring the leading axes.

    The seam is strictly `[B, L] -> [B, digest_size]`, so the flatten and the
    restore around it are the whole adapter every function in this module needs.
    """
    array = fnp.asarray(data).astype(np.uint8)
    if array.ndim == 0:
        raise ValueError("expected at least one byte axis")
    lead = array.shape[:-1]
    out = fnp.asarray(hash_.digest(array.reshape(-1, array.shape[-1])), dtype=fnp.uint8)
    return out.reshape(*lead, out.shape[-1])


def g(data: ArrayLike) -> tuple[Array, Array]:
    """`G = SHA3-512`, split into the two 32-byte halves the callers want.

    Every use of `G` in the standard immediately splits its 64 bytes in two —
    `(rho, sigma)` in key generation, `(K, r)` in encapsulation — so returning
    the halves is the seam the scheme actually consumes. Returning the 64 bytes
    would put the same slice at every call site.
    """
    out = _digest(Sha3_512(), data)
    return out[..., :SEED_SIZE], out[..., SEED_SIZE:]


def h(data: ArrayLike) -> Array:
    """`H = SHA3-256`, §4.1. Hashes `ek` into `dk`, and the ciphertext in decaps."""
    return _digest(Sha3_256(), data)


def j(data: ArrayLike) -> Array:
    """`J = SHAKE256(·, 32)`, §4.1 — the implicit-rejection secret's derivation.

    Distinct from `H` despite both producing 32 bytes: `J` is the branch a wrong
    ciphertext lands on, and deriving the rejection secret with `H` instead would
    still round-trip every valid ciphertext. Only a negative vector sees it.
    """
    return _digest(Shake256(output_size=SEED_SIZE), data)


def prf(eta: int, seed: ArrayLike, nonce: ArrayLike) -> Array:
    """`PRF_eta(s, b) = SHAKE256(s ‖ b, 64*eta)`, §4.1.

    `nonce` is the single counter byte the standard appends, broadcast against
    `seed`'s leading axes — which is what lets one call produce a whole vector of
    `k` polynomials' worth of PRF output rather than `k` calls.
    """
    if eta not in ETAS:
        raise ValueError(f"FIPS 203 uses eta in {ETAS}, got {eta}")
    # Checked here rather than through `encoding.checked_length`: this module
    # depends on `params` alone, as every sibling at this layer does, and reusing
    # that helper would pull the whole wire-format module into a hash's build
    # closure for three lines. `AesGcm._checked` keeps its own copy for the same
    # reason.
    s = fnp.asarray(seed).astype(np.uint8)
    if s.shape[-1] != SEED_SIZE:
        raise ValueError(f"PRF seed is {SEED_SIZE} bytes, got {s.shape[-1]}")
    b = fnp.asarray(nonce).astype(np.uint8)
    lead = fnp.broadcast_shapes(s.shape[:-1], b.shape)
    message = fnp.concatenate(
        [
            fnp.broadcast_to(s, (*lead, SEED_SIZE)),
            fnp.broadcast_to(b[..., None], (*lead, 1)),
        ],
        axis=-1,
    )
    return _digest(Shake256(output_size=CBD_BYTES_PER_ETA * eta), message)


def xof(output_size: int, seed: ArrayLike) -> Array:
    """`XOF = SHAKE128`, §4.1 — the matrix expansion's stream.

    The length is the caller's because `SampleNTT` fixes it from a rejection
    bound rather than from the standard; see `sampling.py`.
    """
    return _digest(Shake128(output_size=output_size), seed)
