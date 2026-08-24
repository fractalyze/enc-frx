# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""DHKEM(X25519, HKDF-SHA256), per RFC 9180 §4.1 — X25519 behind the Kem seam.

What §4.1 adds over the raw curve function is small and entirely about
labeling: `Encap` derives an ephemeral key pair, runs one Diffie-Hellman, and
feeds the result through `ExtractAndExpand` — HKDF-SHA256 under the §4
`LabeledExtract`/`LabeledExpand` scheme with `suite_id = "KEM" ‖ I2OSP(0x0020,
2)` — binding the shared secret to `kem_context = enc ‖ pkRm`. Every size in
sight (`Nsk`, `Npk`, `Nenc`, `Nsecret`, and the seed) is 32 bytes.

**`randomness` is `ikmE`, the §7.1.3 `DeriveKeyPair` input.** The seam takes
encapsulation randomness as an argument (`enc_frx/kem.py`), and RFC 9180's own
vectors are derandomized exactly this way — Appendix A publishes `ikmE` and
derives the ephemeral pair from it — so the seam's `randomness` is that ikm,
not a raw `skE`. `keygen` is the same derivation from the same 32-byte shape:
`sk = LabeledExpand(LabeledExtract("", "dkp_prk", ikm), "sk", "", 32)`, with
no clamping stored — RFC 7748's clamp is part of the X25519 function and is
applied where the scalar is used.

**The decapsulation key is the §7.1.2 serialized private key alone**, 32
bytes, because the seam carries keys in the standard's encoding. `Decap`'s
`kem_context` needs `pkRm`, so `decaps` re-derives it with a second ladder —
the same cost `encaps` pays to derive `pkE`, spent on the recipient's side
instead of stored in a non-standard 64-byte encoding.

**A wrong ciphertext yields a different secret, with no rejection machinery.**
This seam has no failure channel by design (`enc_frx/kem.py`), and DHKEM needs
none of ML-KEM's implicit-rejection scaffolding to satisfy it: a tampered
`enc` is some other curve point, its Diffie-Hellman is some other value, and
the derived secret differs — deterministically, per batch entry, with nothing
to compare against.

**RFC 9180 §7.1.4's all-zero check is deliberately not performed here.** The
RFC has `Encap`/`Decap` abort when the X25519 output is all zeros, which
happens exactly when the peer value is one of the curve's small-order points.
This seam cannot abort — that is its load-bearing decision — and there is no
rejection seed to fall back on, so a small-order `enc` decapsulates to a
secret derived from an all-zero DH: deterministic, well-defined, and
computable by anyone from `enc` and `pkRm` alone. What that costs is HPKE's
contributory-behavior guarantee, not confidentiality against a passive
adversary — an attacker who sends a small-order point learns the recipient
will derive a key the attacker already knows, which matters to protocols that
treat "we derived the same key" as proof the peer contributed to it. A caller
whose protocol needs §7.1.4 screens `enc` against the twelve small-order
encodings before decapsulating, outside the seam, where a verdict is allowed
to exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike
from hash_frx import Hmac, Sha256, hkdf_expand, hkdf_extract

from enc_frx.kem import Kem
from enc_frx.x25519 import x25519 as x

KEY_SIZE = x.KEY_SIZE  # Nsk = Npk = Nenc = Nsecret = Nseed, all 32 (§7.1)

_KEM_ID = 0x0020  # RFC 9180 §7.1, Table 2
_SUITE_ID = b"KEM" + _KEM_ID.to_bytes(2, "big")  # §4: "KEM" ‖ I2OSP(kem_id, 2)
_VERSION = b"HPKE-v1"


def _mac() -> Hmac:
    """HMAC-SHA256, built per call — hash-frx's rule: `Sha256()` reads its
    emitter routing at construction, and an instance hoisted to import time
    would pin that answer before anything could vary it."""
    return Hmac(Sha256())


def _prefix(data: bytes, batch: tuple[int, ...]) -> Array:
    """A static byte string, broadcast in front of a `[B, L]` operand."""
    return fnp.broadcast_to(
        fnp.asarray(np.frombuffer(data, dtype=np.uint8)), (*batch, len(data))
    )


def _labeled_extract(mac: Hmac, label: bytes, ikm: Array) -> Array:
    """§4 `LabeledExtract("", label, ikm)` — every DHKEM extract's salt is the
    empty string, so it is fixed here rather than parameterized."""
    labeled = fnp.concatenate(
        [_prefix(_VERSION + _SUITE_ID + label, ikm.shape[:-1]), ikm], axis=-1
    )
    return fnp.asarray(
        hkdf_extract(mac, fnp.zeros(0, dtype=fnp.uint8), labeled), dtype=fnp.uint8
    )


def _labeled_expand(
    mac: Hmac, prk: Array, label: bytes, info: Array | None, length: int
) -> Array:
    """§4 `LabeledExpand(prk, label, info, L)`: the info is
    `I2OSP(L, 2) ‖ "HPKE-v1" ‖ suite_id ‖ label ‖ info`."""
    prefix = _prefix(
        length.to_bytes(2, "big") + _VERSION + _SUITE_ID + label, prk.shape[:-1]
    )
    labeled = prefix if info is None else fnp.concatenate([prefix, info], axis=-1)
    return fnp.asarray(hkdf_expand(mac, prk, labeled, length), dtype=fnp.uint8)


def _derive_secret_key(ikm: Array) -> Array:
    """§7.1.3 `DeriveKeyPair` for X25519, the secret half: uint8 `[B, 32]` ikm
    -> uint8 `[B, 32]` sk. The public half is `x25519(sk, 9)`."""
    mac = _mac()
    prk = _labeled_extract(mac, b"dkp_prk", ikm)
    return _labeled_expand(mac, prk, b"sk", None, KEY_SIZE)


def _extract_and_expand(dh: Array, kem_context: Array) -> Array:
    """§4.1 `ExtractAndExpand(dh, kem_context)` -> the 32-byte shared secret."""
    mac = _mac()
    prk = _labeled_extract(mac, b"eae_prk", dh)
    return _labeled_expand(mac, prk, b"shared_secret", kem_context, KEY_SIZE)


def _checked(name: str, value: ArrayLike) -> Array:
    array = fnp.asarray(value, dtype=fnp.uint8)
    if array.shape[-1] != KEY_SIZE:
        raise ValueError(
            f"{name} is {KEY_SIZE} bytes in DHKEM(X25519, HKDF-SHA256), got "
            f"shape {array.shape}"
        )
    return array


class DhKemX25519:
    """DHKEM(X25519, HKDF-SHA256) over the `Kem` seam. See the module
    docstring for the two decisions that are not the RFC's: `randomness` is
    `ikmE`, and §7.1.4's all-zero abort has no seam to exist at."""

    seed_size = KEY_SIZE
    randomness_size = KEY_SIZE
    encapsulation_key_size = KEY_SIZE
    decapsulation_key_size = KEY_SIZE
    ciphertext_size = KEY_SIZE
    shared_secret_size = KEY_SIZE

    def __eq__(self, other: object) -> bool:
        # Parameterless, so equality is by type; `type(other) is not
        # type(self)` rather than `isinstance`, which is asymmetric under
        # subclassing (the `ByteHash` convention hash-frx spells out).
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """§7.1.3 `DeriveKeyPair(ikm)`: uint8 `[32]` -> (pk, sk), both 32
        bytes in their §7.1.1/§7.1.2 serializations."""
        seed = _checked("seed", seed)
        lead = seed.shape[:-1]
        secret_key = _derive_secret_key(seed.reshape(-1, KEY_SIZE)).reshape(
            *lead, KEY_SIZE
        )
        return x.public_key(secret_key), secret_key

    def encaps(
        self, encapsulation_key: ArrayLike, *, randomness: ArrayLike
    ) -> tuple[Array, Array]:
        """§4.1 `Encap(pkR)`, derandomized by `ikmE`: uint8 `[B, 32]` keys and
        ikms -> (`enc` = pkE, shared secret), 32 bytes each per entry."""
        pk_r = _checked("encapsulation_key", encapsulation_key)
        ikm_e = _checked("randomness", randomness)
        sk_e = _derive_secret_key(ikm_e)
        enc = x.public_key(sk_e)
        dh = x.x25519(sk_e, pk_r)
        kem_context = fnp.concatenate([enc, pk_r], axis=-1)
        return enc, _extract_and_expand(dh, kem_context)

    def decaps(self, decapsulation_key: ArrayLike, ciphertext: ArrayLike) -> Array:
        """§4.1 `Decap(enc, skR)`: uint8 `[B, 32]` keys and encapsulated
        points -> uint8 `[B, 32]` shared secrets. Always a secret — a
        tampered `enc` derives a different one (module docstring)."""
        sk_r = _checked("decapsulation_key", decapsulation_key)
        enc = _checked("ciphertext", ciphertext)
        dh = x.x25519(sk_r, enc)
        pk_r = x.public_key(sk_r)  # §4.1: kem_context binds pkRm, not skR
        kem_context = fnp.concatenate([enc, pk_r], axis=-1)
        return _extract_and_expand(dh, kem_context)


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Kem] = DhKemX25519
