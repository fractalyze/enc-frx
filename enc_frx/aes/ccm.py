# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""AES-CCM, per NIST SP 800-38C (and RFC 3610). The constrained-device `Aead`.

CCM is CTR for confidentiality and CBC-MAC for authenticity, glued by the
Appendix A formatting: a first block `B_0` encoding the flags, nonce, and
payload length; the associated data behind its own length encoding; and
counter blocks `A_i` sharing the nonce. Both halves run under one key —
the mode predates the derive-a-key-per-purpose fashion — and the tag is the
CBC-MAC's head masked by `AES(A_0)`.

**The batch shape is the honest headline.** CBC-MAC chains block to block, so
within one message nothing is parallel — the scan is real, the way Poly1305's
is and GHASH's is, and unlike GCM there is no Horner-with-powers trick waiting
because the chain runs through the block cipher itself. Across the batch every
message advances independently, which is where the width comes from.

**The nonce/payload trade is the standard's.** `q = 15 - nonce_size` bytes
hold the payload length, so a 13-byte nonce caps a payload at 2^16 bytes and a
7-byte nonce at 2^64. Both sizes are fixed at construction (`AesGcm`'s rule:
they are properties of the verifier), and the tag length admits the full
SP 800-38C §5.1 list — including the short tags GCM's page refuses — because
for CCM the standard states them without the appendix-and-usage-limits framing
that made GCM's 32/64-bit tags a refusal here. What a short tag costs is on
the scheme page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.typing import ArrayLike

from enc_frx.aead import Aead
from enc_frx.aes import block

BLOCK_SIZE = 16

# SP 800-38C A.1: t is an even 4..16, n is 7..13 (q = 15 - n is 2..8).
TAG_SIZES = (4, 6, 8, 10, 12, 14, 16)
NONCE_SIZES = (7, 8, 9, 10, 11, 12, 13)


def _cbc_step(
    carry: tuple[Array, tuple[Array, ...]], block_row: Array
) -> tuple[tuple[Array, tuple[Array, ...]], None]:
    """One CBC-MAC link: `X_{i+1} = AES(X_i ^ B_i)`.

    The schedule rides the carry — loop-invariant, costing nothing — for the
    reason `ghash._absorb` states: `lax.scan` caches its lowering on the body
    function's identity, so a closure defined per call would re-trace the AES
    body every call.
    """
    state, keys = carry
    return (block.encrypt_with_schedule(list(keys), state ^ block_row), keys), None


class AesCcm:
    """SP 800-38C's AEAD over AES, parameterized by its three sizes."""

    def __init__(
        self, key_size: int = 16, nonce_size: int = 13, tag_size: int = BLOCK_SIZE
    ) -> None:
        if key_size not in block.KEY_SIZES:
            raise ValueError(f"AES takes a 16, 24, or 32 byte key, got {key_size}")
        if nonce_size not in NONCE_SIZES:
            raise ValueError(
                f"SP 800-38C A.1 permits a nonce of {NONCE_SIZES} bytes, got "
                f"{nonce_size}"
            )
        if tag_size not in TAG_SIZES:
            raise ValueError(
                f"SP 800-38C §5.1 permits a tag of {TAG_SIZES} bytes, got "
                f"{tag_size}"
            )
        self.key_size = key_size
        self.nonce_size = nonce_size
        self.tag_size = tag_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AesCcm):
            return NotImplemented
        return self._parameters == other._parameters

    def __hash__(self) -> int:
        return hash((type(self), self._parameters))

    def __repr__(self) -> str:
        return (
            f"AesCcm(key_size={self.key_size}, nonce_size={self.nonce_size}, "
            f"tag_size={self.tag_size})"
        )

    @property
    def _parameters(self) -> tuple[int, int, int]:
        return (self.key_size, self.nonce_size, self.tag_size)

    @property
    def _q(self) -> int:
        return 15 - self.nonce_size

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
        tag = self._tag(schedule, nonce, associated_data, plaintext)
        mask, keystream = self._ctr(schedule, nonce, plaintext.shape[-1])
        return fnp.concatenate(
            [plaintext ^ keystream, tag ^ mask[..., : self.tag_size]], axis=-1
        )

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
        mask, keystream = self._ctr(schedule, nonce, body.shape[-1])
        plaintext = body ^ keystream

        # CCM authenticates the plaintext, so the MAC runs over the decrypted
        # bytes; an arithmetic reduction per entry, never an early exit and
        # never a `lax.cond`.
        expected = self._tag(schedule, nonce, associated_data, plaintext)
        accepted = fnp.all((expected ^ mask[..., : self.tag_size]) == tag, axis=-1)
        return fnp.where(accepted[..., None], plaintext, 0), accepted

    def _checked(self, value: ArrayLike, size: int, name: str) -> Array:
        array = fnp.asarray(value, dtype=fnp.uint8)
        if array.shape[-1] != size:
            raise ValueError(
                f"this scheme's {name} is {size} bytes, got {array.shape[-1]}"
            )
        return array

    def _aad_prefix(self, length: int) -> np.ndarray:
        """SP 800-38C A.2.2's associated-data length encoding — the 2-byte
        form below `2^16 - 2^8`, the 0xFFFE-marked 6-byte form beyond it.
        The length is static, so the prefix is a host constant."""
        if length < 0xFF00:
            return np.frombuffer(length.to_bytes(2, "big"), dtype=np.uint8)
        return np.frombuffer(b"\xff\xfe" + length.to_bytes(4, "big"), dtype=np.uint8)

    def _tag(
        self,
        schedule: list[Array],
        nonce: Array,
        associated_data: ArrayLike | None,
        plaintext: Array,
    ) -> Array:
        """A.2's formatting and A.2/6.1's CBC-MAC, unmasked: `MSB_t(X_last)`
        over `B_0 ‖ [a]len ‖ A ‖ pad ‖ P ‖ pad`."""
        batch = plaintext.shape[:-1]
        if associated_data is None:
            associated_data = fnp.zeros((*batch, 0), dtype=fnp.uint8)
        associated_data = fnp.asarray(associated_data, dtype=fnp.uint8)
        aad_len = associated_data.shape[-1]

        # A.2.1: flags = [adata present] ‖ (t-2)/2 ‖ q-1, then N, then the
        # payload length in q big-endian bytes.
        flags = (64 if aad_len else 0) | ((self.tag_size - 2) // 2) << 3 | (self._q - 1)
        length_field = np.frombuffer(
            plaintext.shape[-1].to_bytes(self._q, "big"), dtype=np.uint8
        )
        b0 = fnp.concatenate(
            [
                fnp.broadcast_to(fnp.asarray(np.uint8(flags))[None], (*batch, 1)),
                nonce,
                fnp.broadcast_to(fnp.asarray(length_field), (*batch, self._q)),
            ],
            axis=-1,
        )[..., None, :]

        sections = [b0]
        if aad_len:
            prefix = self._aad_prefix(aad_len)
            sections.append(
                block.pad_to_blocks(
                    fnp.concatenate(
                        [
                            fnp.broadcast_to(
                                fnp.asarray(prefix), (*batch, prefix.shape[0])
                            ),
                            associated_data,
                        ],
                        axis=-1,
                    )
                )
            )
        if plaintext.shape[-1]:
            sections.append(block.pad_to_blocks(plaintext))
        blocks = fnp.concatenate(sections, axis=-2)

        state = fnp.zeros((*batch, BLOCK_SIZE), dtype=fnp.uint8)
        (state, _), _ = frx.lax.scan(
            _cbc_step, (state, tuple(schedule)), fnp.moveaxis(blocks, -2, 0)
        )
        return state[..., : self.tag_size]

    def _ctr(
        self, schedule: list[Array], nonce: Array, length: int
    ) -> tuple[Array, Array]:
        """A.3's counter blocks in one AES invocation: `AES(A_0)` (the tag's
        mask) and `length` bytes of keystream from `A_1`. The counter field is
        `q` big-endian bytes; every `A_i` index is static, so the counters are
        trace-time constants beside the nonce."""
        batch = nonce.shape[:-1]
        blocks = -(-length // BLOCK_SIZE)
        # The same big-endian encoding `_tag` writes with to_bytes; an index
        # that no longer fits q bytes raises here at trace time, which is the
        # standard's payload-per-nonce bound surfacing rather than a silent
        # wrap.
        indices = np.frombuffer(
            b"".join(i.to_bytes(self._q, "big") for i in range(blocks + 1)),
            dtype=np.uint8,
        ).reshape(blocks + 1, self._q)
        counters = fnp.concatenate(
            [
                fnp.broadcast_to(
                    fnp.asarray(np.uint8(self._q - 1))[None],
                    (*batch, blocks + 1, 1),
                ),
                fnp.broadcast_to(
                    nonce[..., None, :], (*batch, blocks + 1, self.nonce_size)
                ),
                fnp.broadcast_to(fnp.asarray(indices), (*batch, blocks + 1, self._q)),
            ],
            axis=-1,
        )
        stream = block.encrypt_blocks(schedule, counters)
        mask = stream[..., 0, :]
        keystream = stream[..., 1:, :].reshape(*batch, blocks * BLOCK_SIZE)
        return mask, keystream[..., :length]


if TYPE_CHECKING:
    # The seam conformance pin every implementation module ends with.
    _: type[Aead] = AesCcm
