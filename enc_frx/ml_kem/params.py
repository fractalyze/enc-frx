# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The constants FIPS 203 fixes for every ML-KEM parameter set.

Scheme-wide, not per-parameter-set: `q`, the ring degree, and the sizes that
follow from them are the same for ML-KEM-512, -768 and -1024. What varies —
`k`, `du`, `dv`, and the derived key and ciphertext sizes — is the parameter
sets' business and stays out of here.

Deliberately dependency-free. `ntt.py` needs `zk_dtypes` for the field dtype and
`encoding.py` needs neither it nor `frx` for these values, so a shared home for
the numbers cannot be either module without widening one of their build targets.
"""

from __future__ import annotations

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
