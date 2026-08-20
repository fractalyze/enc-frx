# AES-CCM

SP 800-38C's (and RFC 3610's) AEAD over AES, implemented in
[`enc_frx/aes/ccm.py`](../../enc_frx/aes/ccm.py): CTR for confidentiality,
CBC-MAC for authenticity, both under one key, glued by the Appendix A
formatting. The mode the constrained-device standards (802.15.4, the TLS CCM
suites) require.

## What the standard fixes, and what this implementation chooses

The standard leaves three sizes open and ties two of them together: the nonce
(7–13 bytes) and the payload bound share the 15 bytes beside the flags —
`q = 15 - nonce_size` bytes hold the payload length, so the longest nonce caps
a payload at 2^16 bytes. All three sizes are fixed at construction, `AesGcm`'s
rule: they are properties of the verifier, and a tag length read from a caller
is the classic downgrade.

The tag list here is the full §5.1 one, **including the 4- and 6-byte tags the
GCM page refuses**. The asymmetry is deliberate and worth stating: GCM's short
tags live in an appendix under invocation limits, and truncating a GCM tag
also cheapens forgeries structurally; CCM's short tags are in the standard's
main parameter list, and a CBC-MAC forgery is a straight 2^-8t guess. A caller
choosing 4 bytes is choosing a 2^-32 forgery bound — that is the security
level, stated here so the choice is a choice.

## Where the batch axis is

The honest headline: **within one message, nothing is parallel.** CBC-MAC
chains every block through the cipher, and unlike GHASH there is no
Horner-with-powers rescue, because the chain runs through AES itself. The scan
is real, as it is for Poly1305. Across the batch every message advances
independently — that is where the width comes from, and why the seam's
batch-first shape matters more here than for any other AES mode in the tree.

The counter blocks, by contrast, are near-free: every `A_i` index is static,
so the counters are trace-time constants beside the nonce, and the tag's mask
`AES(A_0)` rides the same single AES invocation as the payload keystream.

## What leaks, and what the caller owes

Nonce discipline, as everywhere: a repeated nonce under one key leaks the XOR
of payloads (CTR) and opens tag-splicing games. The failure is not GCM's
subkey catastrophe, but it is not benign. The caller also owes the payload
bound its nonce choice implies — the `q`-byte length field is not a
formality, and `seal` will happily format only what fits it.

CCM authenticates the plaintext, so `open` decrypts first and MACs what it
recovered; the seam's per-entry masking is what stands between that
intermediate and a caller who did not read `ok`. The repo posture applies
unchanged: no constant-time claim, not a production decryption oracle
([security.md](../reference/security.md)).

## The gate

ACVP's AES-CCM set: 8310 cases over 147 (key, nonce, tag) instances, decrypt
groups carrying published `testPassed: false` rows. The per-PR gate
(`ccm_test`) runs every key length at the corners of the (nonce, tag) space —
corners, because `q` changes the whole block layout with the nonce length —
through `check_aead`, batches chosen to carry both a tampering-ready shape and
a published rejection. The exhaustive pass is `ccm_sweep_test`, tagged
`slow_kat` like GCM's.
