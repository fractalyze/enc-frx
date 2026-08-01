# ChaCha20-Poly1305 and XChaCha20-Poly1305

Two `Aead` implementations that differ only in nonce size. RFC 8439 defines the
first; [draft-irtf-cfrg-xchacha](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-xchacha-03)
derives the second from it with one extra key derivation.

## What the standards fix, and what this implementation chooses

The standards fix everything observable: the ChaCha20 constants and round
schedule, the Poly1305 clamping and modulus, the MAC input's layout
(`aad ‖ pad16 ‖ ciphertext ‖ pad16 ‖ len(aad) ‖ len(ciphertext)`, both lengths
little-endian 64-bit), the block counter starting at 1 with block 0 reserved for
the one-time key, and XChaCha's split of a 192-bit nonce into a 128-bit
derivation half and a 64-bit remainder prefixed with four zero bytes.

What this implementation chooses is the arithmetic layout, and one choice is
forced rather than picked. **Poly1305 carries ten limbs of 13 bits in `uint32`
lanes**, not the five limbs of 26 bits every reference implementation uses,
because there is no 64-bit integer lane here
([`../reference/conventions.md`](../reference/conventions.md)) and 26-bit limbs
produce 52-bit products. The reduction `2^130 = 5` is a clean convolution only
where the limbs tile 130 bits exactly, and of the layouts that do, only 13-bit
limbs fit a 32-bit product. Its accumulator bound is stated where the layout is
defined and asserted in a test.

XChaCha's HChaCha20 lives in `chacha20.py` beside the block function rather than
with the scheme that uses it: it is the same state layout and the same twenty
rounds, differing only in that it omits the feedforward add and returns the first
and last rows. A copy would let a fix to one miss the other.

## Where the batch axis is, and where it is not

`seal` and `open` batch over `B` messages of one static length; both are one
traced computation.

Inside, the two primitives have opposite shapes and the implementation says so:

- **ChaCha20 parallelizes over blocks.** Every block is independent given its
  counter, so `B` messages of `L` blocks are one `[B, L]` computation with the
  rounds applied across all of it. A scan over blocks would serialize the
  cipher's only parallelism.
- **Poly1305 does not.** Block `i + 1`'s accumulator depends on block `i`'s, so
  the parallelism is the batch alone: `B` independent Horner chains scanned
  together over the block axis.

Splitting one long message across precomputed powers of `r` is the known way to
parallelize Poly1305 further. It multiplies the code by the number of lanes and
only pays for long single messages, so it waits for a benchmark that asks.

## What leaks, and what the caller owes

Neither scheme has a table lookup or a data-dependent index anywhere — the cipher
is add/xor/rotate and the authenticator is limb arithmetic — so neither
introduces a secret-dependent address. That is not a constant-time claim; see
[`../reference/security.md`](../reference/security.md) for what this repo does
and does not assert.

The tag comparison is an arithmetic reduction over all sixteen bytes, per batch
entry, and a failing entry's plaintext comes back zeroed rather than raw.

**The nonce is the caller's.** A scheme takes one and never generates one.

- **RFC 8439's nonce is 96 bits**, which is too small to choose at random: the
  birthday bound puts a collision within reach around 2^32 messages under one
  key, and a repeat destroys confidentiality for the affected messages and
  forfeits the one-time authenticator that protects them.
- **XChaCha's is 192 bits**, which is safe to choose at random. That is its only
  reason to exist. A caller without a reliable counter should prefer it.

The counter is 32 bits and wraps rather than carrying into the nonce, so a single
(key, nonce) pair encrypts at most 256 GiB.

## Status

XChaCha is a draft, not an RFC. The construction has been stable for years and is
in wide production use (libsodium, age), and the draft carries the test vectors
this implementation is gated on.
