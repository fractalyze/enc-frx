# AES-GCM-SIV

RFC 8452's nonce-misuse-resistant AEAD, implemented in
[`enc_frx/aes/gcm_siv.py`](../../enc_frx/aes/gcm_siv.py) over the same parts as
[AES-GCM](aes-gcm.md): the AES block cipher, and the GF(2^128) dtype — reached
here as POLYVAL ([`polyval.py`](../../enc_frx/aes/polyval.py)) through the
byte-order identity RFC 8452 Appendix A states, rather than through a second
field implementation.

## What the standard fixes, and what this implementation chooses

The RFC fixes almost everything: the nonce is 12 bytes, the tag is 16, the
counter is the tag with its top bit set and advances as a little-endian 32-bit
word — each the opposite convention to GCM's, which is why the SIV keystream
lives in `gcm_siv.py` rather than growing an endianness parameter on
[`ctr.py`](../../enc_frx/aes/ctr.py). The one open parameter is the key size,
and only 16 and 32 exist — the RFC defines no 192-bit member, so none is
offered.

This implementation chooses to compute POLYVAL via the Appendix A bridge
(`ByteReverse ∘ GHASH(mulX_GHASH(ByteReverse(H)), ·)`). The identity is the
standard's own, the GHASH path it reuses is the one the GCM gate already
exercises, and the cost is a byte-reversed copy per call. A native-order
POLYVAL would be a measured optimization, not a different answer.

## Where the batch axis is

As in GCM: the payload's counter blocks are independent, so a batch is one
array through one AES invocation; POLYVAL is a Horner chain — parallel across
the batch, sequential over a message's blocks. What SIV adds is per-message
*key derivation* (six AES blocks of `LE32(i) ‖ nonce` per entry, first eight
bytes kept from each), which also rides the batch: every entry derives its own
message keys in the same traced call, so per-entry keys — the shape HPKE-style
callers need — cost nothing extra by construction.

Nothing streams, by design: the tag is a PRF over the whole plaintext and the
counter comes from the tag, so `seal` holds the message. That is the price of
misuse resistance, and it is the standard's, not this implementation's.

## What leaks, and what the caller owes

Repeating a nonce under one key here does **not** leak the authentication key
— the failure GCM's page warns about. What a repeated nonce leaks is equality:
identical (nonce, AAD, plaintext) triples produce identical ciphertexts, so an
observer learns that two messages were the same. The caller owes the nonce
discipline that avoids even that, but a violation costs a confidentiality
crumb rather than every future tag.

`open` decrypts before it verifies — SIV authenticates the recovered
plaintext — so the seam's masking rule is load-bearing here in a way even GCM
does not match: the unverified plaintext exists inside the call, and the
per-entry mask is what keeps it from a caller who did not read `ok`. The
repo-wide posture applies unchanged: no constant-time claim, not a production
decryption oracle ([security.md](../reference/security.md)).

## The gate

Every RFC 8452 Appendix C case, both key sizes, C.3's counter-wrap rows
included — those exist to catch a keystream that increments big-endian or
fails to wrap, which every other vector passes. The cases run through
`check_aead`, so the published answers are checked alongside the tampering
and per-entry masking the standard does not publish. POLYVAL is additionally
pinned to the Appendix A worked example and differentially against a
from-the-definition integer implementation, because the production path is
the identity — a test that reused it would prove plumbing, not convention.
