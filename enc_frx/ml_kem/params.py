# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What FIPS 203 fixes: the scheme-wide constants, and the three parameter sets.

`q`, the ring degree and the seed size are the same for ML-KEM-512, -768 and
-1024. What a parameter set varies is one row of Table 2 — `k`, `eta1`, `eta2`,
`du`, `dv` — plus the key and ciphertext sizes those imply, which Table 3 states
and `MlKemParams` derives.

The two live together because the sizes join them: an encapsulation key is
`POLY_BYTES * k + SEED_SIZE`, so a separate home for the rows would import the
constants straight back and leave a reader two places to look for one number.

Deliberately dependency-free. `ntt.py` needs `zk_dtypes` for the field dtype and
`encoding.py` needs neither it nor `frx` for these values, so a shared home for
the numbers cannot be either module without widening one of their build targets.
The parameter sets keep it that way — a frozen dataclass is stdlib, and the
scheme that takes one is a layer above (`ml_kem.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

# The modulus, and the ring degree of Z_q[X]/(X^256 + 1).
Q = 3329
N = 256

# 2-adicity 8 gives a primitive 256th root and no 512th, so a length-256
# negacyclic transform does not exist over this field and FIPS 203's NTT is two
# length-128 ones. See `ntt.py`.
ZETA = 17

# Every seed, hash and rejection value in the scheme is 32 bytes (§4.1).
SEED_SIZE = 32

# One polynomial at the uncompressed width: 256 coefficients x 12 bits.
POLY_BYTES = 32 * 12


def decryption_key_size(k: int) -> int:
    """`ByteEncode_12(ŝ)` — K-PKE's `dk_PKE`, §5.1.

    Distinct from `decapsulation_key_size` below, which is ML-KEM's `dk` and
    carries this as its first field. K-PKE decrypts with the secret vector alone.
    """
    return POLY_BYTES * k


def encapsulation_key_size(k: int) -> int:
    """`ByteEncode_12(t̂) ‖ ρ`, §7.1."""
    return POLY_BYTES * k + SEED_SIZE


def decapsulation_key_size(k: int) -> int:
    """`dk_PKE ‖ ek ‖ H(ek) ‖ z`, §7.3."""
    return decryption_key_size(k) + encapsulation_key_size(k) + 2 * SEED_SIZE


def ciphertext_size(k: int, du: int, dv: int) -> int:
    """`ByteEncode_du(u) ‖ ByteEncode_dv(v)`, §7.2."""
    return 32 * (du * k + dv)


@dataclass(frozen=True)
class MlKemParams:
    """One row of FIPS 203 Table 2, under the name §8 gives it.

    `name` is a field rather than a label kept beside the row, because it is the
    one column nothing here can derive and someone else publishes: ACVP's
    `parameterSet` is this string verbatim, so it is the join between a published
    case and the set that case was generated for.

    Frozen, so `__eq__` and `__hash__` are by value — which is what the `Kem`
    seam requires of the scheme that carries this (`enc_frx/kem.py`). A scheme
    instance rides pytree aux, where identity equality does not error: it
    silently re-traces the enclosing jit zone for every freshly built instance,
    and that surfaces as a slow call rather than as a failure.

    Everything here shapes the trace rather than riding in it. `k` sets the shape
    of every array in the scheme, so it is a Python `int` and never a traced
    value, and the sizes below are static for the same reason — the seam promises
    them to a consumer that allocates before it calls.
    """

    name: str
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int

    # Table 3's sizes, derived from this row rather than transcribed from it.
    # The published literals are what the tests compare these against, so a wrong
    # formula and a wrong transcription cannot agree with each other.
    @property
    def decryption_key_size(self) -> int:
        return decryption_key_size(self.k)

    @property
    def encapsulation_key_size(self) -> int:
        return encapsulation_key_size(self.k)

    @property
    def decapsulation_key_size(self) -> int:
        return decapsulation_key_size(self.k)

    @property
    def ciphertext_size(self) -> int:
        return ciphertext_size(self.k, self.du, self.dv)


# FIPS 203 Table 2. Two of the three differ from ML-KEM-768 in a code path
# rather than only in a width: `eta1` is 3 at ML-KEM-512 alone, which is the
# second centered-binomial width, and `du`/`dv` change at ML-KEM-1024 alone,
# which is the second compression width.
ML_KEM_512 = MlKemParams(name="ML-KEM-512", k=2, eta1=3, eta2=2, du=10, dv=4)
ML_KEM_768 = MlKemParams(name="ML-KEM-768", k=3, eta1=2, eta2=2, du=10, dv=4)
ML_KEM_1024 = MlKemParams(name="ML-KEM-1024", k=4, eta1=2, eta2=2, du=11, dv=5)

# In Table 2's order, which is also increasing security category (1, 3, 5).
PARAMETER_SETS = (ML_KEM_512, ML_KEM_768, ML_KEM_1024)
