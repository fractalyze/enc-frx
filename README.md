# enc-frx

FRX-native encryption — authenticated encryption and post-quantum key
encapsulation.

`enc-frx` builds on [`hash-frx`](https://github.com/fractalyze/hash-frx) for
every symmetric primitive it needs, and on **FRX** — Fractalyze's fork of
[JAX](https://github.com/jax-ml/jax) — for tracing and codegen, lowered through
**Fractalyze XLA**.

## Design philosophy

- **Two seams, no concrete scheme in the consumer.** `Kem` is
  `keygen` / `encaps` / `decaps` over opaque key and ciphertext types; `Aead` is
  `seal` / `open` over a key, a nonce, and associated data. A consumer picks a
  scheme by construction, not by branching on a name.
- **Batch-parallel by construction.** Decapsulation and opening are the hot
  paths and a service runs them by the thousand — a batch of `B` runs in one
  call, so the work maps onto a GPU's width rather than a Python loop.
- **Standards-exact.** Every scheme reproduces its specification byte for byte
  (ML-KEM = FIPS 203, ChaCha20-Poly1305 = RFC 8439, AES-GCM = SP 800-38D), gated
  on the published known-answer tests — the negative vectors included, because an
  `open` that accepts everything passes every positive one.
- **Failure is a value, not an exception.** `open` returns a validity flag beside
  a masked plaintext, and a wrong ML-KEM ciphertext decapsulates to a *different*
  shared secret rather than raising. Both are what the standards require, and
  both are what a traced batch can express.

## Security posture

The hot paths here — `decaps` and `open` — touch a long-lived secret key and
take input an adversary chose. That is the inverse of
[`sig-frx`](https://github.com/fractalyze/sig-frx), whose hot path is
verification over entirely public data, so the posture that repo carries does
not transfer.

**No constant-time claim is made, anywhere.** The implementation is traced by
FRX and compiled by XLA, and nothing in that pipeline promises data-independent
instruction selection, gather timing, or `select` lowering — nor would a claim
established once survive a dependency bump. Tracing does remove data-dependent
branching by construction, which is necessary and nowhere near sufficient.

So: **this is not a production decryption oracle.** Do not stand it up as a
service that decapsulates or opens adversary-supplied ciphertexts under a
long-lived key on a machine an adversary can measure. Batch processing over data
you already hold, test-vector work, and research are what it is for.

## Status

Bootstrapping. Work is tracked on the
[issues](https://github.com/fractalyze/enc-frx/issues).

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
