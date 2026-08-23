# enc-frx docs

Topic-organized reference, indexed by what you're trying to do. For the project
overview and the build, see [`../README.md`](../README.md).

The tree is small on purpose: **[`reference/`](reference)** is the rules every
scheme is held to, **[`schemes/`](schemes)** is one design-notes page per scheme.

## `reference/` — the rules

| Question | Where |
| -------- | ----- |
| What does this implementation claim against an adversary who can measure it — and what does that rule out as a use? | [`security.md`](reference/security.md) |
| The rules a scheme is held to — the batch axis, how failure is carried, the KAT gate, what a test's size declares | [`conventions.md`](reference/conventions.md) |

## `schemes/` — one page per scheme

| Question | Where |
| -------- | ----- |
| What a scheme page must answer | [`README.md`](schemes/README.md) |
| The ChaCha20 family — nonce sizes, where the batch axis is, the limb layout | [`chacha20-poly1305.md`](schemes/chacha20-poly1305.md) |
| AES-GCM — the tag length as a parameter, and why a repeated nonce costs the key | [`aes-gcm.md`](schemes/aes-gcm.md) |
| ML-KEM — the fixed XOF budget its rejection sampler runs on, and why a public seed is what makes that sound | [`ml-kem.md`](schemes/ml-kem.md) |
| DHKEM(X25519, HKDF-SHA256) — HPKE's KEM over the ladder, and the one RFC check the seam cannot express | [`dhkem-x25519.md`](schemes/dhkem-x25519.md) |

Detailed design, findings, and open decisions live on the issues, not in the
tree — the epic is
[fractalyze/enc-frx#1](https://github.com/fractalyze/enc-frx/issues/1).

## The two seams

A KEM establishes a shared secret; an AEAD encrypts with one. They are separate
Protocols because their failure semantics are opposites, and collapsing them
would mean admitting the wrong one for each: `Aead.open` **must** report
authentication failure, and `Kem.decaps` **must not** report decapsulation
failure — a wrong ciphertext yields a different shared secret rather than an
error, which is what makes the FO transform work.

Neither has a scalar entry point. `decaps` and `open` are the hot paths and they
are trivially parallel, so a single operation is `B = 1`.

Symmetric hashes are not implemented here. They come from
[`hash-frx`](https://github.com/fractalyze/hash-frx), which owns byte-exactness
and the GPU fusion for every primitive this repo needs.

## Before implementing anything

Read [`reference/security.md`](reference/security.md). This repo's hot paths hold
a long-lived secret key and take adversary-chosen input, which is the inverse of
`sig-frx`, and the posture that follows constrains how every scheme is written.
