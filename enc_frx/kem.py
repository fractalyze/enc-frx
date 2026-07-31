# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Kem seam every key-encapsulation scheme in this repo implements.

`keygen` / `encaps` / `decaps` over keys and ciphertexts in their standard's wire
encoding. A consumer reads the size attributes to allocate statically and calls
the three methods; it picks a scheme by construction, so swapping ML-KEM-768 for
a hybrid changes that one construction rather than every call site.

A KEM establishes a shared secret and encrypts nothing. What uses that secret is
an `Aead` (`enc_frx/aead.py`), and the two seams are separate because their
failure semantics are opposites — see below.

**`decaps` has no failure channel, and that is the load-bearing decision.**
ML-KEM's Fujisaki-Okamoto transform requires *implicit rejection*: a malformed
ciphertext yields a shared secret derived deterministically from a rejection seed
held in the decapsulation key — a real-looking secret that simply differs from
what any sender computed. Reporting failure instead would hand an attacker
exactly the bit the transform exists to withhold, and that bit is enough to mount
the chosen-ciphertext attack the transform prevents. So there is no `ok` here to
return, no exception to raise, and the input checks that reject a malformed key
route through the same rejection path rather than out of band.

This is the one place `Kem` and `Aead` must not be made to look alike. `Aead.open`
*must* report failure; `Kem.decaps` *must not*.

**`encaps` and `decaps` are batch-first.** Every argument carries a leading `[B]`
axis. Decapsulation is the hot path — a service does it per message — and
encapsulation to many recipients is a real workload, so both take the whole batch
as one traced computation. A single operation is `B = 1`, not a separate entry
point; a seam that admitted a scalar `decaps` would get one implemented as a
Python loop, and the parallelism would be gone before anyone noticed. `keygen` is
not batched: it is not the hot path, and `frx.vmap` covers the caller that needs
it.

**Randomness is an argument, never drawn inside.** `encaps` is randomized, and a
traced function that reaches for a global generator is not reproducible from its
arguments — which is precisely what the known-answer tests need it to be. The
standards' derandomized entry point (FIPS 203's `ML-KEM.Encaps_internal`) is a
different operation and lives on the scheme under its own name, below this seam:
the harness that drives it already names the scheme it is testing, and putting it
here would make every future KEM implement a NIST-shaped internal interface.

**Keys, ciphertexts, and shared secrets are `uint8` arrays in the standard's
encoding**, not scheme-named pytrees. What a consumer holds is bytes — off a
wire, out of a key store — so a seam taking a structured form would make every
consumer call a scheme-specific decode first, which means naming the scheme,
which is what the seam exists to prevent. A scheme parses its own encoding on
entry; one that wants a parsed key across many calls exposes that below the seam.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface: a scheme instance rides pytree aux, where identity equality
silently re-traces the enclosing jit zone on every freshly built instance — a
cost that does not error, it just makes every call slow. A Protocol cannot
enforce this; each implementation carries it.

Each implementation module ends with a conformance pin, so mypy fails the module
that drifts from the seam rather than the consumer that calls it::

    if TYPE_CHECKING:
        _: type[Kem] = MlKem
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from frx import Array
from frx.typing import ArrayLike


@runtime_checkable
class Kem(Protocol):
    seed_size: int  # keygen input, in bytes
    randomness_size: int  # encaps input, in bytes
    encapsulation_key_size: int  # serialized public key, in bytes
    decapsulation_key_size: int  # serialized secret key, in bytes
    ciphertext_size: int
    shared_secret_size: int

    def keygen(self, seed: ArrayLike) -> tuple[Array, Array]:
        """Derive a key pair from `seed`: uint8 `[seed_size]` -> (encaps, decaps).

        Both keys come back in the standard's encoding, sized
        `encapsulation_key_size` and `decapsulation_key_size`. Deterministic in
        `seed` — the standards define keygen that way, and the known-answer tests
        reproduce published key bytes from a published seed.
        """
        ...

    def encaps(
        self, encapsulation_key: ArrayLike, *, randomness: ArrayLike
    ) -> tuple[Array, Array]:
        """Encapsulate a batch: uint8 `[B, encapsulation_key_size]`, uint8
        `[B, randomness_size]` -> (uint8 `[B, ciphertext_size]`, uint8
        `[B, shared_secret_size]`).

        Entry `i` of both results belongs to `(encapsulation_key[i],
        randomness[i])`. Deterministic in its arguments: calling twice with the
        same randomness produces the same ciphertext, which is what lets a
        decapsulator's re-encryption check work at all.
        """
        ...

    def decaps(self, decapsulation_key: ArrayLike, ciphertext: ArrayLike) -> Array:
        """Decapsulate a batch: uint8 `[B, decapsulation_key_size]`, uint8
        `[B, ciphertext_size]` -> uint8 `[B, shared_secret_size]`.

        Always returns a shared secret. A ciphertext that does not decapsulate
        yields the scheme's implicit-rejection secret — deterministic in the
        decapsulation key and the ciphertext, and indistinguishable to the caller
        from a real one. There is no verdict to inspect and nothing raises; see
        the module docstring for why adding one would be an attack.

        One call is one traced computation over the whole batch, and rejection is
        decided per entry. A scheme that reduced the check over the batch would
        pass every all-valid and every all-invalid test ever written.
        """
        ...
