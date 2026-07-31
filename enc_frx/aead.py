# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""The Aead seam every authenticated-encryption scheme in this repo implements.

`seal` / `open` over a key, a nonce, associated data, and a message. A consumer
reads the size attributes to allocate statically and calls the two methods; it
picks a scheme by construction, so swapping ChaCha20-Poly1305 for AES-GCM changes
that one construction rather than every call site.

**`open` returns a validity flag, and masks the plaintext of a failing entry.**
This is the decision the seam exists to fix, and it is forced by tracing rather
than chosen. A batch of `B` ciphertexts is one traced computation; some entries
authenticate and some do not, and there is no exception that means "entry 7
failed". So the failure has to be a value — and the moment it is a value, the
plaintext of a failing entry exists as an array a caller can read.

Releasing unverified plaintext is a standing AEAD misuse, and the C-API
convention of handing back the raw buffer beside a status is exactly the shape
that invites it: eventually some caller reads the buffer without reading the
status. So the failing entries come back zeroed under the same mask that produced
`ok`. The caller must still check `ok` — masking does not make that optional, it
makes forgetting cost zeros instead of attacker-chosen bytes.

A caller who genuinely wants the unauthenticated decryption is asking for a
different operation and can say so explicitly on a scheme; the seam does not
offer it by accident.

**Everything is batch-first.** Every argument carries a leading `[B]` axis and
`ok` is `bool[B]`. Opening is the hot path and it is trivially parallel, so a
single message is `B = 1`, not a separate entry point. Rejection is decided per
entry: a scheme that reduced the tag comparison over the batch, or masked
batch-wide from `any(ok)`, would pass every all-valid and every all-invalid test
ever written.

**One length per batch.** Plaintext is `[B, L]` and ciphertext `[B, L +
tag_size]` with `L` static, as it is for `hash_frx.ByteHash`. Messages of
differing length are padded by the caller, which also owns the true length —
inside a trace there is nothing else it could be, and pretending otherwise would
put a data-dependent shape in the seam.

**The nonce belongs to the caller.** A scheme takes one; it never generates one.
What a scheme assumes about uniqueness, and what breaks when that is violated, is
stated on its page — and the consequence is not uniform, since reusing a nonce
under one GCM key leaks the authentication key itself rather than only the
confidentiality of the affected messages.

**Keys, nonces, and messages are `uint8` arrays in the standard's encoding**, and
`seal` returns the ciphertext with the tag in the layout the standard defines
rather than as a separate return value: that is what goes on a wire, and a seam
that split them would make every consumer reassemble.

Implementations MUST define value-based `__eq__`/`__hash__` over their full
parameter surface: a scheme instance rides pytree aux, where identity equality
silently re-traces the enclosing jit zone on every freshly built instance — a
cost that does not error, it just makes every call slow. A Protocol cannot
enforce this; each implementation carries it.

Each implementation module ends with a conformance pin, so mypy fails the module
that drifts from the seam rather than the consumer that calls it::

    if TYPE_CHECKING:
        _: type[Aead] = ChaCha20Poly1305
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from frx import Array
from frx.typing import ArrayLike


@runtime_checkable
class Aead(Protocol):
    key_size: int
    nonce_size: int
    tag_size: int  # the authentication tag, included in `seal`'s output

    def seal(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        plaintext: ArrayLike,
    ) -> Array:
        """Seal a batch: uint8 `[B, key_size]`, uint8 `[B, nonce_size]`, uint8
        `[B, A]` or None, uint8 `[B, L]` -> uint8 `[B, L + tag_size]`.

        The tag is in the standard's layout, not returned separately. `A` and `L`
        are static; `associated_data` of `None` means empty, which is what a
        caller with nothing to authenticate passes rather than a zero-width array.

        Entry `i` of the result belongs to entry `i` of every argument. Repeating
        a nonce under one key is the caller's error and the scheme's page says
        what it costs.
        """
        ...

    def open(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        ciphertext: ArrayLike,
    ) -> tuple[Array, Array]:
        """Open a batch: uint8 `[B, key_size]`, uint8 `[B, nonce_size]`, uint8
        `[B, A]` or None, uint8 `[B, L + tag_size]` -> (uint8 `[B, L]`, bool
        `[B]`).

        `ok[i]` is the verdict on entry `i`, and **the plaintext of every entry
        where `ok` is false is zero** — not the unauthenticated decryption. See
        the module docstring: the mask is what keeps a caller that forgets to read
        `ok` from acting on attacker-chosen bytes.

        The tag comparison is an arithmetic reduction over the whole tag, never an
        early exit and never a `lax.cond`.
        """
        ...
