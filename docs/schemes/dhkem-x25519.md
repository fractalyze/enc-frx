# DHKEM(X25519, HKDF-SHA256)

RFC 9180 §4.1's Diffie-Hellman KEM — HPKE's mandatory-to-implement suite, and
the classical half of every deployed hybrid (TLS's X25519MLKEM768, X-Wing) —
implemented in [`enc_frx/x25519/dhkem.py`](../../enc_frx/x25519/dhkem.py) as a
thin labeling layer over two things that already exist: the X25519 ladder
([`x25519.py`](../../enc_frx/x25519/x25519.py)) and hash-frx's HKDF. Every
size — keys, encapsulation, shared secret, seed — is 32 bytes.

## What the standard fixes, and what this implementation chooses

The RFC fixes the whole derivation: `DeriveKeyPair` (§7.1.3),
`Encap`/`Decap`'s single Diffie-Hellman, and `ExtractAndExpand` under the §4
labeling with `suite_id = "KEM" ‖ 0x0020`, binding the secret to
`kem_context = enc ‖ pkRm`. The known-answer gate is the RFC's own corpus —
the machine-readable Appendix A, sha256-pinned in
[`MODULE.bazel`](../../MODULE.bazel) — driven per keygen / encapsulation /
decapsulation case through the shared harness, tampering pass included.

Three choices are this repo's:

- **`randomness` is `ikmE`**, the `DeriveKeyPair` input, not a raw ephemeral
  scalar — the derandomization the RFC's vectors themselves use, and what
  makes `encaps` a function of its arguments as the seam requires.
- **The decapsulation key is the §7.1.2 serialized private key alone** (32
  bytes). `Decap` needs `pkRm` for `kem_context`, so `decaps` re-derives it
  with a second ladder rather than carrying a non-standard 64-byte encoding.
- **§7.1.4's all-zero abort is not performed.** The seam has no failure
  channel by design; what that costs and who owes the check instead is below.

The curve arithmetic is the registered `zk_dtypes.curve25519_bf` field, worked
in its Montgomery storage variant `curve25519_bf_mont` where the fast multiply
lives. RFC bytes *are* that field's canonical storage, so they cross into the
trace as a free `view` and reach Montgomery storage by an `astype` inside the
trace; [`ladder_bench.py`](../../enc_frx/x25519/testing/ladder_bench.py) prices
that boundary against host-entered material and exists to keep it free.

This replaced a hand-rolled uint32 limb field (16 limbs of radix 2^16, the
no-64-bit-lane layout) that predated the dtype's registration — the dtype path
measured 5.6-27x faster on CUDA and 17-58x on CPU on the last wheel where both
existed, a comparison the bench can no longer reproduce now that the limb arm
is gone with `field.py`. Note what the swap required
beyond speed: a rare-rate fault in the Montgomery multiply
(fractalyze/xla#542 — 1 pair in 200,000, always low by exactly 1) passed every
fixed vector while corrupting ~1.4% of calls, so the gate is RFC 7748 §5.2's
*iterated* vector, not its fixed ones. See
[`ladder_bench.py`](../../enc_frx/x25519/testing/ladder_bench.py) for that
history.

## Where the batch axis is

Everything batches, and nothing inside a message is sequential except the
ladder itself: `encaps`/`decaps` run `B` independent 255-iteration ladders as
one traced computation (the batch is pure width), and the HKDF chain is two
HMAC-SHA256 calls whose `T(i)` unrolling is static. `keygen` follows the seam
rule — unbatched, `frx.vmap` when a caller wants a batch. `decaps` costs two
ladders per entry (the DH and the `pkRm` re-derivation), `encaps` the same two
on the sender's side.

## What leaks, and what the caller owes

The posture is [`security.md`](../reference/security.md)'s: dataflow-uniform
(fixed iteration count, arithmetic cswap, no data-dependent branching), no
constant-time claim on any backend. The long-lived `skR` enters `decaps`
against adversary-chosen `enc`, which is exactly the exposure that page rules
out of scope for side channels.

What the caller owes is the one RFC requirement the seam cannot express:
`Decap` aborting when the Diffie-Hellman output is all-zero (§7.1.4), which
happens exactly for the curve's small-order points. Here a small-order `enc`
decapsulates — deterministically — to a secret computable by anyone from
public values. Confidentiality against a passive adversary is unaffected, but
a protocol that reads "we derived the same key" as proof of peer contribution
must screen `enc` against the small-order encodings before decapsulating,
outside the seam, where a verdict is allowed to exist
([`dhkem.py`](../../enc_frx/x25519/dhkem.py)'s module docstring carries the
full reasoning).

There is no nonce: encapsulation randomness is `ikmE`, one fresh 32-byte input
per entry, and reusing one reuses the ephemeral key — two recipients would see
the same `enc`, and either can compute the other's shared secret. Fresh
randomness per encapsulation is the caller's obligation, as everywhere at
these seams.
