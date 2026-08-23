# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Poly1305, per RFC 8439 §2.5.

A polynomial evaluated over `GF(2^130 - 5)` — not cryptographically hard, but
the one part of ChaCha20-Poly1305 that has to carry a field this stack has no
native integer width for.

**The field is `zk_dtypes.prime_field(2^130 - 5)`**, worked in its Montgomery
storage variant where the multiply is fast. An element is one value, not a limb
vector, so the polynomial is spelled in operators.

RFC 8439's block encoding is little-endian and so is the field's canonical
storage, so a padded block crosses into the trace as a `view` and reaches
Montgomery storage by an `astype`. The final tag needs no separate reduction
step either: canonical storage already holds the fully reduced residue, so its
low 16 bytes are `acc mod p mod 2^128`, which is what §2.5.1 asks for.

This replaced a hand-rolled layout of ten limbs of radix 2^13 on uint32 lanes.
That layout was forced rather than chosen — FRX runs with x64 disabled, so a
`uint64` request is silently truncated and a limb product had to fit `uint32`,
which put the radix at 13 and made the reduction a ten-way convolution with its
own carry sweep and accumulator bound. A registered field dtype removes the
constraint entirely, and it is faster by 8.6-11x on CPU and 1.7-2.2x on CUDA at
16 KiB messages — measured on frx 0.10.2.dev20260822150923, the last build
carrying both layouts. `testing/mac_bench.py` holds that table and the one it
still re-measures every run.

**Parallelism comes from the batch, not from within a message.** Block `i + 1`'s
accumulator depends on block `i`'s, so `B` messages are `B` independent Horner
chains scanned together over the block axis. Unlike ChaCha20, the `scan` is
correct here — the dependency is real. Splitting one long message across
precomputed powers of `r` is the known way to parallelize further; it multiplies
the code by the number of lanes and only pays for long single messages, so it
waits for a benchmark that asks.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array
from frx.typing import ArrayLike

KEY_SIZE = 32
TAG_SIZE = 16
BLOCK_SIZE = 16

_P = 2**130 - 5
# Canonical storage is the RFC's little-endian encoding, so bytes cross into
# WIRE as a view; WORK is where the multiply is fast, an `astype` away.
WIRE = np.dtype(zk_dtypes.prime_field(_P, "std"))
WORK = np.dtype(zk_dtypes.prime_field(_P, "mont"))
_FIELD_BYTES = 32

# The 17th byte carries bit 128 — the `1` appended to every block — so the byte
# form a block is read as is one byte wider than the block itself.
_PADDED_BLOCK = BLOCK_SIZE + 1

# RFC 8439 §2.5.1: the top four bits of every fourth byte, and the bottom two of
# the three that follow the first, are cleared before `r` is used.
_CLAMP = np.array(
    [0xFF, 0xFF, 0xFF, 0x0F, 0xFC, 0xFF, 0xFF, 0x0F]
    + [0xFC, 0xFF, 0xFF, 0x0F, 0xFC, 0xFF, 0xFF, 0x0F],
    dtype=np.uint8,
)


def _to_field(padded: Array) -> Array:
    """uint8 `[..., 17]` -> one WORK element per row.

    The 17 bytes hold a value below 2^129, so zero-padding to the field's 32
    and viewing is exact — canonical storage *is* the little-endian encoding.
    """
    pad = [(0, 0)] * (padded.ndim - 1) + [(0, _FIELD_BYTES - padded.shape[-1])]
    return fnp.pad(padded, pad).view(WIRE).astype(WORK)


def _absorb(
    state: tuple[Array, Array], block: Array
) -> tuple[tuple[Array, Array], None]:
    """One RFC 8439 §2.5 step: `acc = (acc + block) * r`.

    `r` rides the scan carry rather than being closed over, because frx keys
    the loop-body lowering cache on the body function's identity (the gotcha
    `aes/ghash._absorb` measured): a closure minted per `mac` call would
    re-trace the body every call.
    """
    accumulator, r = state
    return ((accumulator + block) * r, r), None


def _add_mod_2_128(left: Array, right: Array) -> Array:
    """Little-endian 16-byte addition, discarding the carry out of the top."""
    carry = fnp.zeros_like(left[..., 0])
    lanes = []
    for byte in range(TAG_SIZE):
        total = left[..., byte] + right[..., byte] + carry
        lanes.append(total & np.uint32(0xFF))
        carry = total >> np.uint32(8)
    return fnp.stack(lanes, axis=-1).astype(fnp.uint8)


def _blocks(message: Array) -> Array:
    """`uint8 [B, N]` -> `uint8 [blocks, B, 17]`, each block's high bit set.

    The appended `1` is RFC 8439 §2.5's, and where it lands depends on the block:
    past the sixteenth byte for a full block, and immediately after the message
    for a short final one. `N` is static, so the positions are a trace-time
    constant added to zero padding rather than an indexed write.
    """
    length = message.shape[-1]
    blocks = -(-length // BLOCK_SIZE)
    padded = fnp.pad(
        message,
        [(0, 0)] * (message.ndim - 1) + [(0, blocks * BLOCK_SIZE - length)],
    )
    grouped = padded.reshape(*message.shape[:-1], blocks, BLOCK_SIZE)
    grouped = fnp.concatenate(
        [grouped, fnp.zeros((*grouped.shape[:-1], 1), dtype=fnp.uint8)], axis=-1
    )
    high = np.zeros((blocks, _PADDED_BLOCK), dtype=np.uint8)
    for index in range(blocks):
        used = min(BLOCK_SIZE, length - index * BLOCK_SIZE)
        high[index, used] = 1
    return fnp.moveaxis(grouped + fnp.asarray(high), -2, 0)


def mac(key: ArrayLike, message: ArrayLike) -> Array:
    """`uint8 [B, 32]`, `uint8 [B, N]` -> `uint8 [B, 16]`.

    The key is `r || s`: `r` clamped per RFC 8439 §2.5.1 and used as the
    polynomial's point, `s` added to the result mod 2^128. One-time — a key
    reused across two messages loses the authenticator.
    """
    key = fnp.asarray(key, dtype=fnp.uint8)
    message = fnp.asarray(message, dtype=fnp.uint8)

    r = _to_field(key[..., :BLOCK_SIZE] & fnp.asarray(_CLAMP))

    accumulator = fnp.broadcast_to(
        fnp.asarray(np.array([0], dtype=WORK)), (*message.shape[:-1], 1)
    )
    if message.shape[-1]:
        blocked = _to_field(_blocks(message))
        (accumulator, _), _ = frx.lax.scan(_absorb, (accumulator, r), blocked)

    # Canonical storage holds the fully reduced residue, so its low 16 bytes
    # are `acc mod p mod 2^128` — no separate reduction step.
    tag = accumulator.astype(WIRE).view(fnp.uint8)[..., :TAG_SIZE]
    return _add_mod_2_128(
        tag.astype(fnp.uint32), key[..., BLOCK_SIZE:].astype(fnp.uint32)
    )
