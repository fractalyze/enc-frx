# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES-GCM, per NIST SP 800-38D §7.1-7.2. The second `Aead` implementation.

The assembly is small because the pieces are already here: CTR for the payload,
GHASH for the authentication, and the block cipher for both the hash subkey and
the tag's mask. What GCM adds is the wiring between them, and every part of that
wiring is a place the construction is easy to get subtly wrong:

- `H = E_K(0^128)` — the hash subkey is the *cipher's* output, so it changes with
  the key and nothing else.
- `J_0` — the IV directly when it is 96 bits, GHASH-derived otherwise
  (`ctr.initial_counter`).
- The payload's keystream starts at `inc32(J_0)`, **not** `J_0`, while the tag is
  masked by `E_K(J_0)` itself. Starting both at the same block would XOR the tag
  with a keystream block a caller already holds.

**The seam it exercises.** This is the second scheme behind `Aead`, and the first
whose internals look nothing like the first one's: field arithmetic rather than
lane arithmetic, a different padding rule, a tag derived by masking rather than by
a one-time MAC key. It fits without an escape hatch, which is the whole reason to
want a second implementation of a shape the first does not share.

**The key is expanded once per call**, not per block and not per primitive. The
schedule is roughly 71% of a single block encryption's cost (`block.py`), and one
call uses it for the whole payload, for `H`, and for the tag mask — so it is
computed at the top of `seal` / `open` and threaded through all three.

**Where the batch axis is.** The payload is `B` messages of independent counter
blocks, one array. GHASH is a Horner chain: parallel across the batch, sequential
within a message. Both come from the modules that own them.

**The tag length is a scheme parameter.** Not an argument to `open` — a verifier
that took the length from its caller could be talked into accepting a shorter tag
than it was built for, which is the classic AEAD downgrade. See `TAG_SIZES` for
what the standard permits and `docs/schemes/aes-gcm.md` for what truncation costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx.numpy as fnp
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aead import Aead
from enc_frx.aes import block, ctr, ghash

BLOCK_SIZE = 16

# SP 800-38D §5.2.1.2: a tag is 128, 120, 112, 104, or 96 bits.
#
# Appendix C also defines 64 and 32 for a short list of applications, under usage
# limits on the number of invocations and the payload length that only the caller
# knows and nothing here could enforce. A scheme that offered them would be
# offering a 2^-32 forgery probability to callers who read the tag length as a
# size rather than as a security level, so they are not offered.
TAG_SIZES = (12, 13, 14, 15, 16)

# The IV length §5.2.1.1 recommends, and the only one a caller should choose: it
# is used directly, so it is the one length that cannot collide through GHASH.
DEFAULT_NONCE_SIZE = 12


class AesGcm:
    """SP 800-38D's AEAD over AES, parameterized by its three sizes.

    `key_size` picks AES-128, -192 or -256; `nonce_size` and `tag_size` are the
    IV and tag lengths in bytes. All three are fixed at construction because all
    three are properties of the verifier rather than of a message.
    """

    def __init__(
        self,
        key_size: int = 16,
        nonce_size: int = DEFAULT_NONCE_SIZE,
        tag_size: int = BLOCK_SIZE,
    ) -> None:
        if key_size not in block.KEY_SIZES:
            raise ValueError(f"AES takes a 16, 24, or 32 byte key, got {key_size}")
        if tag_size not in TAG_SIZES:
            raise ValueError(
                f"SP 800-38D §5.2.1.2 permits a tag of {TAG_SIZES} bytes, got "
                f"{tag_size}"
            )
        if nonce_size < 1:
            raise ValueError(f"GCM's IV is at least one byte, got {nonce_size}")
        self.key_size = key_size
        self.nonce_size = nonce_size
        self.tag_size = tag_size

    def __eq__(self, other: object) -> bool:
        # The seam requires value equality over the full parameter surface: an
        # instance rides pytree aux, where identity equality silently re-traces
        # the enclosing jit zone for every freshly built instance.
        if not isinstance(other, AesGcm):
            return NotImplemented
        return self._parameters == other._parameters

    def __hash__(self) -> int:
        return hash((type(self), self._parameters))

    def __repr__(self) -> str:
        return (
            f"AesGcm(key_size={self.key_size}, nonce_size={self.nonce_size}, "
            f"tag_size={self.tag_size})"
        )

    @property
    def _parameters(self) -> tuple[int, int, int]:
        return (self.key_size, self.nonce_size, self.tag_size)

    def seal(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        plaintext: ArrayLike,
    ) -> Array:
        key = self._checked(key, self.key_size, "key")
        nonce = self._checked(nonce, self.nonce_size, "nonce")
        plaintext = fnp.asarray(plaintext, dtype=fnp.uint8)

        schedule = block.key_schedule(key)
        subkey = self._hash_subkey(schedule, key)
        counter = ctr.initial_counter(subkey, nonce)

        mask, keystream = self._gctr(schedule, counter, plaintext.shape[-1])
        ciphertext = plaintext ^ keystream
        tag = self._tag(subkey, mask, associated_data, ciphertext)
        return fnp.concatenate([ciphertext, tag], axis=-1)

    def open(
        self,
        key: ArrayLike,
        nonce: ArrayLike,
        associated_data: ArrayLike | None,
        ciphertext: ArrayLike,
    ) -> tuple[Array, Array]:
        key = self._checked(key, self.key_size, "key")
        nonce = self._checked(nonce, self.nonce_size, "nonce")
        ciphertext = fnp.asarray(ciphertext, dtype=fnp.uint8)
        if ciphertext.shape[-1] < self.tag_size:
            raise ValueError(
                f"a {self.tag_size}-byte tag does not fit in a "
                f"{ciphertext.shape[-1]}-byte ciphertext"
            )
        body = ciphertext[..., : -self.tag_size]
        tag = ciphertext[..., -self.tag_size :]

        schedule = block.key_schedule(key)
        subkey = self._hash_subkey(schedule, key)
        counter = ctr.initial_counter(subkey, nonce)
        mask, keystream = self._gctr(schedule, counter, body.shape[-1])

        # An arithmetic reduction over the whole tag, per batch entry. Never an
        # early exit and never a `lax.cond`: the comparison is on secret-derived
        # data, and in a batch a `cond` forces both sides anyway.
        expected = self._tag(subkey, mask, associated_data, body)
        accepted = fnp.all(expected == tag, axis=-1)

        # Computed before the verdict is known, because a traced batch computes
        # both — so the mask is the only thing keeping it from a caller who did
        # not read `ok`.
        return fnp.where(accepted[..., None], body ^ keystream, 0), accepted

    def _checked(self, value: ArrayLike, size: int, name: str) -> Array:
        """Pin an argument's width to the instance's, at trace time.

        The sizes are parameters here rather than constants as they are for
        ChaCha20-Poly1305, so a batch of 16-byte keys handed to an `AesGcm(32)`
        would otherwise run as AES-128 and produce a perfectly valid tag under a
        scheme nobody asked for.
        """
        array = fnp.asarray(value, dtype=fnp.uint8)
        if array.shape[-1] != size:
            raise ValueError(
                f"this scheme's {name} is {size} bytes, got {array.shape[-1]}"
            )
        return array

    def _hash_subkey(self, schedule: list[Array], key: Array) -> Array:
        """§7.1 step 1: `H = E_K(0^128)`."""
        zeros = fnp.zeros((*key.shape[:-1], BLOCK_SIZE), dtype=fnp.uint8)
        return block.encrypt_with_schedule(schedule, zeros)

    def _gctr(
        self, schedule: list[Array], counter: Array, length: int
    ) -> tuple[Array, Array]:
        """§7.1's two `GCTR` invocations, which are one keystream.

        `uint8 [B, 16]`, a static payload length -> the tag's mask `E_K(J_0)` and
        `length` bytes of keystream from `inc32(J_0)`.

        The standard writes them apart — step 3 encrypts the payload from
        `inc32(J_0)`, step 6 masks the tag at `J_0` — but `GCTR_K(J_0, ·)` is one
        counter sequence whose first block is the mask and whose rest is the
        payload's. Generating them together is one AES invocation rather than
        two, and it is why an empty payload costs exactly the one block it needs
        rather than a discarded second.
        """
        blocks = -(-length // BLOCK_SIZE)
        stream = ctr.keystream(schedule, counter, blocks + 1)
        return stream[..., :BLOCK_SIZE], stream[..., BLOCK_SIZE : BLOCK_SIZE + length]

    def _tag(
        self,
        subkey: Array,
        mask: Array,
        associated_data: ArrayLike | None,
        ciphertext: Array,
    ) -> Array:
        """§7.1 steps 5-6: `MSB_t(GHASH(H, A ‖ C ‖ lengths) ⊕ E_K(J_0))`.

        Step 6 is `GCTR_K(J_0, S)`, which over a single block is the XOR with
        `E_K(J_0)` written here — the counter does not advance within one block.
        """
        digest = ghash.ghash(subkey, ghash.hash_input(associated_data, ciphertext))
        return (digest ^ mask)[..., : self.tag_size]


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Aead] = AesGcm
