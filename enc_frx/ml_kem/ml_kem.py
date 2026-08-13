# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""ML-KEM: the Fujisaki-Okamoto transform over K-PKE, per FIPS 203 §6-7.

`_k_pke` is IND-CPA — malleable, and broken by a chosen-ciphertext attack. What
this module adds is the transform that makes it IND-CCA2, and the whole of that
addition is in `decaps`: recover the message, re-derive the sender's randomness
from it, encrypt again, and compare. A ciphertext nobody could have produced by
running `encaps` fails that comparison.

**Rejection is implicit, and it is the reason this seam has no failure channel.**
A ciphertext that fails the comparison does not raise and does not report
anything. It yields `J(z ‖ c)` — a shared secret derived from a rejection seed
`z` that key generation put in the decapsulation key, deterministic in the
ciphertext and indistinguishable from a real secret to anyone who does not hold
`z`. Reporting failure instead would hand an attacker exactly the bit the
transform exists to withhold, and that bit is enough to mount the attack it
prevents (`enc_frx/kem.py`).

Three consequences, each of which looks like an inefficiency and is not:

- **The re-encryption is unconditional.** It costs about an encapsulation, so
  `decaps` is roughly twice `encaps`. There is no signal that would let it be
  skipped: re-encryption *is* the signal.
- **The comparison reduces over the whole ciphertext**, to one boolean per batch
  entry, with no early exit.
- **The final step is an arithmetic select over the 32 output bytes**, never a
  `cond` and never a Python `if`. Both secrets are computed either way, so a
  branch would buy nothing while making the timing depend on the answer.

**Where FIPS 203's input checks land.** §7.2 and §7.3 place the encapsulation-
and decapsulation-key checks at key *import*, and this seam has no import step —
keys cross as bytes on every call (`enc_frx/kem.py`). So `decaps` folds both into
the same reduction as the ciphertext comparison, and a malformed key yields the
rejection secret rather than an exception, for the same reason a malformed
ciphertext does. `check_encapsulation_key` and `check_decapsulation_key` are the
same two predicates as standalone operations, because ACVP publishes them as
distinct operations with their own verdicts and a `keygen`/`encaps`/`decaps` seam
has nowhere to run them.

Each mirrors exactly one section, and `decaps` is what combines them: §7.3 is the
hash check alone, so `check_decapsulation_key` answers what ACVP's
`decapsulationKeyCheck` asks and nothing more, while `decaps` also runs §7.2 over
the embedded encapsulation key — which it is about to encrypt with.

**Randomness is an argument.** `encaps` takes `m` rather than drawing it, so the
whole scheme is a function of its inputs; `encaps_internal` is the same call
under the standard's own name for the derandomized entry point, which is what a
known-answer harness drives.

`keygen` is not batched at the seam, but nothing here is written per-entry, so a
leading axis works and `frx.vmap` covers the caller that wants one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from enc_frx.kem import Kem
from enc_frx.ml_kem import _k_pke, encoding, hashes
from enc_frx.ml_kem.params import POLY_BYTES, SEED_SIZE, MlKemParams


class MlKem:
    """ML-KEM at one parameter set, over the `Kem` seam.

    `MlKem(ML_KEM_768)`: the set is named once, at construction, and no call site
    below names it again. `params.k` shapes every array here, so it is a Python
    `int` that shapes the trace and never a traced value.

    The set stays reachable as `params` because an instance that hid which one it
    is would force every holder to carry the row beside it — the sizes the seam
    exposes are derived from it, and `_k_pke` takes its five numbers directly.
    """

    def __init__(self, params: MlKemParams) -> None:
        self.params = params
        # `d ‖ z`: the K-PKE seed and the rejection seed, 32 bytes each (§7.1).
        self.seed_size = 2 * SEED_SIZE
        self.randomness_size = SEED_SIZE
        self.encapsulation_key_size = params.encapsulation_key_size
        self.decapsulation_key_size = params.decapsulation_key_size
        self.ciphertext_size = params.ciphertext_size
        self.shared_secret_size = SEED_SIZE

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MlKem):
            return NotImplemented
        return self.params == other.params

    def __hash__(self) -> int:
        # Value-based, per the seam: an instance rides pytree aux, where identity
        # equality re-traces the enclosing jit zone for every freshly built one.
        # The frozen dataclass is what makes this a one-liner.
        return hash(self.params)

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """FIPS 203 Algorithm 16: `[..., 64]` -> `(ek, dk)`.

        `ek` is K-PKE's `ek_PKE` unchanged; the transform's additions are all in
        `dk`, which carries the encapsulation key it belongs to, that key's hash,
        and the rejection seed `z` (§7.1). Decapsulation needs all three: `ek` to
        re-encrypt with, `H(ek)` to bind the two halves together, and `z` to
        derive the rejection secret from.
        """
        material = encoding.checked_length(
            seed, self.seed_size, "an ML-KEM key generation seed"
        )
        d, z = material[..., :SEED_SIZE], material[..., SEED_SIZE:]
        ek, dk_pke = _k_pke.key_gen(d, k=self.params.k, eta1=self.params.eta1)
        return ek, encoding.encode_dk(dk_pke, ek, hashes.h(ek), z)

    def encaps(
        self, encapsulation_key: ArrayLike, *, randomness: ArrayLike
    ) -> tuple[Array, Array]:
        """The seam's `encaps`, which is `encaps_internal` under another name.

        FIPS 203 Algorithm 20 draws `m` and calls Algorithm 17 with it. Here the
        randomness arrives as an argument (`enc_frx/kem.py`), so the drawing is
        the caller's and only the internal call is left.
        """
        return self.encaps_internal(encapsulation_key, randomness)

    def encaps_internal(
        self, encapsulation_key: ArrayLike, m: ArrayLike
    ) -> tuple[Array, Array]:
        """FIPS 203 Algorithm 17: `[..., ek] x [..., 32] -> (c, K)`.

        `(K, r) ← G(m ‖ H(ek))` is what ties the shared secret to the key it was
        encapsulated under, and derives the encryption randomness from the
        message rather than sampling it — which is what makes the ciphertext a
        function `decaps` can recompute.
        """
        key = encoding.checked_length(
            encapsulation_key, self.encapsulation_key_size, "an encapsulation key"
        )
        message = encoding.checked_length(
            m, SEED_SIZE, "ML-KEM encapsulation randomness"
        )
        shared, r = hashes.g(fnp.concatenate([message, hashes.h(key)], axis=-1))
        return self._encrypt(key, message, r), shared

    def decaps(self, decapsulation_key: ArrayLike, ciphertext: ArrayLike) -> Array:
        """FIPS 203 Algorithm 18: `[..., dk] x [..., c] -> [..., 32]`.

        Always a shared secret, one per batch entry, and never a verdict — see
        the module docstring for why the alternative is the attack.
        """
        params = self.params
        dk_pke, ek, _, z = encoding.decode_dk(decapsulation_key, params.k)
        c = encoding.checked_length(ciphertext, self.ciphertext_size, "a ciphertext")

        # Lines 1-4. `K̄` is derived whether or not it is used, because the
        # select below evaluates both sides regardless.
        m = _k_pke.decrypt(dk_pke, c, k=params.k, du=params.du, dv=params.dv)
        shared, r = hashes.g(fnp.concatenate([m, hashes.h(ek)], axis=-1))
        rejected = hashes.j(fnp.concatenate([z, c], axis=-1))

        # Line 5, unconditional: this is the transform. `_k_pke.encrypt` is a
        # function of `(ek, m, r)` alone, which is what lets the comparison mean
        # anything at all — see `_k_pke`'s module docstring.
        accepted = (
            fnp.all(self._encrypt(ek, m, r) == c, axis=-1)
            & self.check_encapsulation_key(ek)
            & self.check_decapsulation_key(decapsulation_key)
        )
        # Line 6 as a mask over the output bytes rather than a branch: `where`
        # lowers to a select, so both secrets are computed and the one that is
        # returned costs nothing to choose.
        return fnp.where(accepted[..., None], shared, rejected)

    def check_encapsulation_key(self, encapsulation_key: ArrayLike) -> Array:
        """FIPS 203 §7.2's modulus check, as a `bool[...]` per batch entry.

        The type check is the length, which is static in a traced program and so
        raises; the modulus check is data and so comes back as a value
        (`encoding.py`). A key that fails it decodes to coefficients the standard
        does not admit, and `_k_pke.encrypt` would use them anyway.
        """
        key = encoding.checked_length(
            encapsulation_key, self.encapsulation_key_size, "an encapsulation key"
        )
        # `ek = ByteEncode_12(t̂) ‖ ρ`, §7.1: the seed carries no coefficients.
        return encoding.coefficients_are_reduced(key[..., : POLY_BYTES * self.params.k])

    def check_decapsulation_key(self, decapsulation_key: ArrayLike) -> Array:
        """FIPS 203 §7.3's hash check, as a `bool[...]` per batch entry.

        `H(ek)` is stored in `dk` and recomputed here, so a `dk` whose two halves
        came from different key pairs fails. That pairing is what the transform
        rests on: re-encryption compares against a ciphertext produced under
        `ek`, so an `ek` that does not belong to `dk_PKE` makes the comparison
        meaningless rather than false.

        The §7.2 modulus check is not folded in, though `decaps` runs both: this
        answers the one question §7.3 asks, so a harness driving ACVP's
        `decapsulationKeyCheck` compares against it directly.
        """
        _, ek, h, _ = encoding.decode_dk(decapsulation_key, self.params.k)
        return fnp.all(hashes.h(ek) == h, axis=-1)

    def _encrypt(self, ek: Array, m: Array, r: Array) -> Array:
        """K-PKE encryption at this parameter set, from either direction.

        The five are spelled out rather than forwarded as the set: FIPS 203
        states each K-PKE algorithm's parameter list itself, and a `_k_pke` that
        took an ML-KEM parameter set would run the import edge backwards.
        """
        params = self.params
        return _k_pke.encrypt(
            ek,
            m,
            r,
            k=params.k,
            eta1=params.eta1,
            eta2=params.eta2,
            du=params.du,
            dv=params.dv,
        )


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Kem] = MlKem
