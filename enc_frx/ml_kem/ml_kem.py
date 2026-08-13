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

**`precompute_decaps` is the import step the seam does not have**, and it does
not change that answer. A key parsed once is checked once, so both sections run
there rather than per ciphertext — but the result is carried on the parsed value
and AND-ed into every later acceptance, because a `precompute` that raised would
put the withheld bit back at a new door. The pair it forms with
`decaps_precomputed` is a *narrower* operation than `decaps`, not a faster one:
one key for the whole batch against `decaps`'s one key per entry, which is why
it takes a rank-1 key and refuses a batched one rather than broadcasting.

**Randomness is an argument.** `encaps` takes `m` rather than drawing it, so the
whole scheme is a function of its inputs; `encaps_internal` is the same call
under the standard's own name for the derandomized entry point, which is what a
known-answer harness drives.

`keygen` is not batched at the seam, but nothing here is written per-entry, so a
leading axis works and `frx.vmap` covers the caller that wants one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from enc_frx.kem import Kem
from enc_frx.ml_kem import _k_pke, encoding, hashes, ntt, sampling
from enc_frx.ml_kem.params import POLY_BYTES, SEED_SIZE, MlKemParams


class PrecomputedDecapsulationKey(NamedTuple):
    """One decapsulation key, parsed once — everything in `decaps` that is a
    function of the key alone.

    Built by `MlKem.precompute_decaps` and consumed by
    `MlKem.decaps_precomputed`, which together are a *narrower* operation than
    the seam's `decaps` rather than a faster one: this is one key for a whole
    batch, where `decaps` takes one key per entry.

    **`key_ok` is why this is not just cached public data.** FIPS 203 §7.2 and
    §7.3 are functions of the key, so a key parsed once is checked once — and
    this seam has no failure channel to report the answer through, so the answer
    rides here as a value and is AND-ed into the acceptance of every
    decapsulation the key goes on to perform. A `precompute` that validated
    eagerly and raised would reintroduce the bit the transform exists to
    withhold, at a new door (`enc_frx/kem.py`).

    **Every field is public.** `dk_pke` is carried as the bytes `decode_dk`
    produced and is decoded to `ŝ` per call, so no secret vector is parsed or
    held here beyond what the key's own encoding already is. That keeps this
    value's handling requirements the same as the encapsulation key's, and it is
    why `docs/reference/security.md` says nothing new about it.

    Unbatched by construction — see `MlKem.precompute_decaps`.
    """

    a_hat: Array
    """`Â = expand_matrix(ρ, k)`, `[k, k, 256]`: `k^2` independent `SampleNTT`
    runs, and the largest Keccak stage of a decapsulation on CPU."""

    t_hat: Array
    """`ByteDecode_12(ek[:384k])` as field elements, `[k, 256]` — the other half
    of what Algorithm 14 decodes before line 9."""

    h_ek: Array
    """`H(ek)`, `[32]`: one flat SHA3-256, and the only member of the
    `H`/`J`/`G` group that does not need the ciphertext."""

    dk_pke: Array
    """K-PKE's `dk_PKE` as bytes, `[384k]` — decoded to `ŝ` per call."""

    z: Array
    """The implicit-rejection seed, `[32]`."""

    key_ok: Array
    """§7.2 and §7.3 over this key, as one `bool[]`."""


class MlKem:
    """ML-KEM at one parameter set, over the `Kem` seam.

    `MlKem(ML_KEM_768)`: the set is named once, at construction, and no call site
    below names it again. `_params.k` shapes every array here, so it is a Python
    `int` that shapes the trace and never a traced value.

    The set is private because the seam's whole premise is that a consumer names
    a scheme once (`enc_frx/kem.py`); a public row is what lets generic code
    reach for `scheme.params.k` and turn the swap back into a call-site edit.
    """

    def __init__(self, params: MlKemParams) -> None:
        self._params = params
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
        return self._params == other._params

    def __hash__(self) -> int:
        # Value-based, per the seam: an instance rides pytree aux, where identity
        # equality re-traces the enclosing jit zone for every freshly built one.
        return hash(self._params)

    def __repr__(self) -> str:
        return f"MlKem({self._params.name})"

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
        ek, dk_pke = _k_pke.key_gen(d, k=self._params.k, eta1=self._params.eta1)
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
        params = self._params
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

    def precompute_decaps(
        self, decapsulation_key: ArrayLike
    ) -> PrecomputedDecapsulationKey:
        """Parse one key for many ciphertexts: uint8 `[decapsulation_key_size]`.

        Below the seam, on `MlKem` only — the `Kem` protocol does not gain a
        method and no consumer of it is affected. `enc_frx/kem.py` reserves this
        shape: a scheme parses its own encoding on entry, and one that wants a
        parsed key across many calls exposes that here.

        **Rank-1 by construction, and that is the safety property.** The seam's
        `decaps` takes `[B, decapsulation_key_size]` — one key *per entry* —
        while this pair takes one key for the whole batch. Those are different
        operations, so a caller holding per-entry keys gets a `ValueError` here
        rather than a plausible answer computed under the wrong key.

        **Nothing raises on a malformed key.** The §7.2 and §7.3 checks run
        here, once, and their answer is carried on the result rather than
        reported — see `PrecomputedDecapsulationKey` for why an exception would
        be the attack.
        """
        dk = encoding.checked_length(
            decapsulation_key, self.decapsulation_key_size, "a decapsulation key"
        )
        if dk.ndim != 1:
            raise ValueError(
                "precompute_decaps takes one key, shaped "
                f"[{self.decapsulation_key_size}], got {list(dk.shape)}. A batch "
                "of keys is `decaps`, which pairs key `i` with ciphertext `i`."
            )
        dk_pke, ek, _, z = encoding.decode_dk(dk, self._params.k)
        t_hat_ints, rho = encoding.decode_ek(ek, self._params.k)
        return PrecomputedDecapsulationKey(
            a_hat=sampling.expand_matrix(rho, self._params.k),
            t_hat=ntt.as_field(t_hat_ints),
            h_ek=hashes.h(ek),
            dk_pke=dk_pke,
            z=z,
            # The same two predicates `decaps` folds into its reduction, and
            # reached through the same methods so there is one definition of
            # each. Their cost is paid once per key rather than per ciphertext.
            key_ok=self.check_encapsulation_key(ek) & self.check_decapsulation_key(dk),
        )

    def decaps_precomputed(
        self, precomputed: PrecomputedDecapsulationKey, ciphertext: ArrayLike
    ) -> Array:
        """Algorithm 18 for a batch under one key: `[B, c] -> [B, 32]`.

        Identical to `decaps` in what it returns and how it rejects — the only
        difference is that everything depending on the key alone was already
        done by `precompute_decaps`. Always a shared secret, never a verdict,
        and nothing raises.
        """
        params = self._params
        c = encoding.checked_length(ciphertext, self.ciphertext_size, "a ciphertext")

        m = _k_pke.decrypt(
            precomputed.dk_pke, c, k=params.k, du=params.du, dv=params.dv
        )
        # The key's fields carry no batch axis, so the two concatenations state
        # the broadcast the array arithmetic below gets for free.
        shared, r = hashes.g(
            fnp.concatenate([m, self._spread(precomputed.h_ek, m)], axis=-1)
        )
        rejected = hashes.j(
            fnp.concatenate([self._spread(precomputed.z, c), c], axis=-1)
        )

        accepted = (
            fnp.all(
                _k_pke.encrypt_expanded(
                    t_hat=precomputed.t_hat,
                    a_hat=precomputed.a_hat,
                    m=m,
                    r=r,
                    k=params.k,
                    eta1=params.eta1,
                    eta2=params.eta2,
                    du=params.du,
                    dv=params.dv,
                )
                == c,
                axis=-1,
            )
            & precomputed.key_ok
        )
        return fnp.where(accepted[..., None], shared, rejected)

    @staticmethod
    def _spread(field: Array, batched: Array) -> Array:
        """One of the key's `[L]` fields against a `[..., M]` batch."""
        return fnp.broadcast_to(field, (*batched.shape[:-1], field.shape[-1]))

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
        return encoding.coefficients_are_reduced(
            key[..., : POLY_BYTES * self._params.k]
        )

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
        _, ek, h, _ = encoding.decode_dk(decapsulation_key, self._params.k)
        return fnp.all(hashes.h(ek) == h, axis=-1)

    def _encrypt(self, ek: Array, m: Array, r: Array) -> Array:
        """K-PKE encryption at this parameter set, from either direction.

        The five are spelled out rather than forwarded as the set: FIPS 203
        states each K-PKE algorithm's parameter list itself, and a `_k_pke` that
        took an ML-KEM parameter set would run the import edge backwards.
        """
        params = self._params
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
