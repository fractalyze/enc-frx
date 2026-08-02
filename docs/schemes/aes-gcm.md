# AES-GCM

[NIST SP 800-38D](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf),
over the AES block cipher in [`../../enc_frx/aes/`](../../enc_frx/aes). The second
`Aead` implementation, and the one that shows the seam fits a scheme shaped
unlike the first.

## What the standard fixes, and what this implementation chooses

The standard fixes the construction and every encoding in it: `H = E_K(0^128)`,
`J_0` from the IV, the payload under CTR from `inc32(J_0)`, GHASH over
`A ‖ 0^v ‖ C ‖ 0^u ‖ [len(A)]_64 ‖ [len(C)]_64`, and a tag of `GHASH ⊕ E_K(J_0)`
truncated to the tag length. It also fixes the bit order GHASH's field elements
are written in, which is the detail most likely to be got wrong — see
[`ghash.py`](../../enc_frx/aes/ghash.py), where the reversal that bridges it to
`zk_dtypes.binary_field_ghash`'s natural basis lives.

This implementation chooses:

- **The three sizes are constructor parameters**, not call arguments:
  `AesGcm(key_size, nonce_size, tag_size)`. Key size picks AES-128, -192 or -256.
- **One key schedule per call.** The schedule is roughly 71% of a single block
  encryption's cost, and one call needs the key for the payload, for `H`, and for
  the tag mask, so it is expanded once at the top of `seal` / `open` and threaded
  through all three.
- **Argument widths are pinned to the instance's** at trace time. The sizes are
  parameters here rather than constants as they are for ChaCha20-Poly1305, so a
  batch of 16-byte keys handed to an `AesGcm(32)` would otherwise silently run as
  AES-128 and produce a perfectly valid tag under a scheme nobody asked for.

### The tag length is a property of the verifier

`tag_size` is fixed at construction and there is no per-call override. A verifier
that took the length from its caller could be talked into accepting a shorter tag
than it was built for, which is the classic AEAD downgrade.

SP 800-38D §5.2.1.2 permits 128, 120, 112, 104, and 96 bits, and those are what
`TAG_SIZES` admits. Appendix C also defines 64 and 32 for a short list of
applications, under limits on the number of invocations and the payload length
that only the caller knows. Those are not offered: a 32-bit tag is a 2^-32
forgery probability per attempt, and a parameter named for a size is one callers
read as a size rather than as a security level. CAVP publishes vectors for both,
and the test suite refuses those sections rather than skipping them.

Truncation is not free even inside the permitted range. A `t`-bit tag gives a
forgery roughly `2^-t` per attempt, and SP 800-38D Appendix C bounds how many
attempts a deployment may allow before that stops being the right number. 128
bits is the default here because it is the only length that needs no such
argument.

## Where the batch axis is

`seal` and `open` take a batch of `B` messages and trace as one computation.
Inside, the parallelism differs per piece, and each piece's own module owns the
choice:

| Piece | Parallel over | Sequential over |
| ----- | ------------- | --------------- |
| The key schedule | the batch | its own rounds (unrolled at trace time) |
| CTR ([`ctr.py`](../../enc_frx/aes/ctr.py)) | the batch **and** the payload's blocks | nothing |
| GHASH ([`ghash.py`](../../enc_frx/aes/ghash.py)) | the batch | the message's blocks — a Horner chain |

So a long message parallelizes in CTR and not in GHASH. Precomputed powers of `H`
would parallelize GHASH within one message too; that waits for a benchmark, since
the batch is already the axis that matters for the workloads here.

The verdict is decided per entry: `ok` is `bool[B]` from an arithmetic reduction
over the whole tag, and the plaintext of a failing entry comes back zeroed. A
scheme that reduced the comparison over the batch, or masked from `any(ok)`,
would pass every all-valid and every all-invalid vector set ever published, which
is why the mixed-validity batch is its own test.

## What leaks, and what the caller owes

Read [`../reference/security.md`](../reference/security.md) first; this section
only adds what is specific to GCM.

Nothing in the scheme indexes memory with a secret. The S-box is arithmetic —
inversion in GF(2^8) by a fixed addition chain, then the affine map — so there is
no table and no secret-dependent gather, and GHASH is a native field multiply.
The message length is not secret and is static in the trace. This is not a
constant-time claim; it is the absence of the specific data-dependent operations
that make an AES implementation obviously not one.

### Reusing a nonce leaks the authentication key

This is the sharp one, and it is worse than the corresponding
ChaCha20-Poly1305 failure. Two messages sealed under one key with the same IV
give an attacker the XOR of their plaintexts — that much is the usual
keystream-reuse loss. But GCM's tag is a polynomial in `H` evaluated over
GF(2^128), and two tags under one `J_0` yield a polynomial equation whose
unknown is `H` itself. Solving it recovers the authentication key, and `H`
depends on the key alone, not on the IV. From then on the attacker forges
**any** message under that key, including under IVs never used.

So nonce reuse in GCM is not a confidentiality loss for the affected messages;
it is a permanent authenticity loss for the key. That asymmetry is the reason
[`aead.py`](../../enc_frx/aead.py) puts nonce ownership on the caller explicitly
rather than leaving it as a convention.

A 96-bit IV is what §5.2.1.1 recommends and the only length a caller should
choose: it is used directly as `J_0`'s prefix, so distinct IVs give distinct
`J_0`. Any other length is hashed through GHASH, which means two distinct IVs can
collide into one `J_0` — the collision probability is what bounds how many
messages a key may cover. The other lengths are implemented because the standard
defines them and CAVP exercises them, not because a caller wants them.

## The gate

CAVP's `gcmtestvectors` set, fetched and sha256-pinned in
[`../../MODULE.bazel`](../../MODULE.bazel): 47250 cases over three key lengths,
three IV lengths, seven tag lengths, and five payload lengths — of which 11908
are decrypt cases whose published expected result is `FAIL`. The negative half of
the gate is therefore the standard's own, not corruption synthesized here, and it
covers shapes hand-written tampering would not think of. The harness adds the
tampering anyway, because CAVP has no mixed-validity batch and the batch axis is
where a masking bug hides.

CAVP rather than ACVP's `ACVP-AES-GCM-1.0` for two reasons worth recording. That
set is AES-128 alone, so it cannot gate the other two key lengths. And ACVP
measures payloads in *bits*: its CTR set is mostly not a whole number of bytes,
which CTR absorbs with a documented mask and GCM could not — the tag covers the
ciphertext, so a masked payload is a different computation rather than a
different encoding. Every CAVP length is a whole number of bytes, and the loader
refuses a section that is not, so that stays true rather than being assumed.
