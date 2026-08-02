# Coding conventions

> Code, symbols, and file paths are English.

This page carries only what is specific to implementing encryption schemes. The
rules every FRX consumer shares — `@jit` placement, `for` vs `lax.scan` vs
`vmap`, pytree registration mechanics, seam conformance pins, the `testing/`
layout, the comment rules — are not repeated here. They are identical in every
repo built on FRX, and a copy per repo is exactly how they drift apart.

## The batch is the compilation unit

`decaps` and `open` take the whole batch and trace as one computation. The `@jit`
boundary belongs around them, never around a per-message body that a driver loop
calls `B` times.

This is what the seams exist to enforce ([`kem.py`](../../enc_frx/kem.py),
[`aead.py`](../../enc_frx/aead.py)) — there is no scalar entry point to
implement, so a Python loop over the batch axis is a bug rather than a slow path.
`keygen` batches with `frx.vmap` when a caller needs it.

Where the parallelism *is* differs per primitive and the implementation must say
which it is. ChaCha20's blocks are independent given the counter, so a whole
batch is one array and a `scan` over blocks would serialize the only source of
parallelism. Poly1305 and GHASH are Horner chains: parallel across the batch,
sequential within a message, and a `scan` on the block axis is correct.

## Bit manipulation on a field goes through `view`

`&`, `^`, `<<`, `>>` do not work on a binary-field array, and the rejection is a
type error rather than a missing lowering: StableHLO's integer ops do not accept
`!field.bf<...>`. That is the right outcome — those are not field operations. A
shift by `k` already exists as a multiply by the field constant `x^k`, and
masking has no meaning on a residue class at all.

What is wanted in their place is access to the *representation*, and the
primitive for that is a bitcast:

```python
data.view(zk_dtypes.binary_field_ghash)   # uint8 [..., 16] -> field [..., 1]
element[..., None].view(fnp.uint8)        # field [...]     -> uint8 [..., 16]
```

Both lower to `bitcast_convert` and a reshape — no copy, no arithmetic. So do the
bit work on the `uint8` side and cross with `view`.

## What belongs in hash-frx, and what belongs here

`hash-frx` owns **unkeyed** primitives and the fusion marker seam. Its
`Permutation` is unkeyed by construction — `permute(state)` takes a state and
nothing else, and every consumer calls it that way — so a keyed primitive cannot
be admitted without changing what the seam means.

So a keyed primitive lives with the construction that keys it. AES's round
function and key schedule are here, under `aes/`, not behind a widened
`Permutation`.

A fused kernel is available either way and is not an argument for moving code:
`hash_frx.fusion.fused_region` is public and its marker is deliberately generic —
zorch already emits it for sumcheck and jagged regions, not only for hashes. What
`hash-frx` owns is the marker seam, not every marker.

## There is no 64-bit integer lane

FRX runs with x64 disabled, so a `uint64` request is **silently truncated to
`uint32`** — a warning, not an error, which means an implementation written
against 64-bit limbs runs and returns wrong numbers. `zk_dtypes` widths do not
help: `uint128` is a host dtype that no traced array can hold, unlike the binary
fields, which are registered in the frontend.

So multi-precision arithmetic keeps every intermediate under 2^32 by
construction, since overflow is silent corruption rather than an error. A product
must fit in 32 bits, which caps limbs at 16 bits and rules out the layouts most
reference implementations use.

Two rules follow. A limb layout **states its accumulator bound** where the layout
is defined, and a test asserts it so a later change to the radix trips. And a
layout whose margin rests on that bound is gated by a **differential test against
Python's arbitrary-precision integers**, over extreme inputs as well as random
ones — a published vector set never approaches the worst case.

## Failure is a value, and the two seams disagree on purpose

Nothing here raises. A traced batch has no exception that means "entry 7 failed",
so a failure is data — and the two seams carry it in opposite directions.

**`Aead.open` must report failure.** It returns `(plaintext, ok)` with `ok` a
`bool[B]`, and **the plaintext of a failing entry comes back masked**, not raw.
Releasing unverified plaintext is a standing AEAD misuse, and a seam that handed
back the raw decryption beside a flag is one a caller eventually reads without
checking the flag. Masking makes forgetting cost zeros instead of
attacker-chosen bytes.

**`Kem.decaps` must not report failure.** ML-KEM's FO transform requires implicit
rejection: a malformed ciphertext yields a *different* shared secret, derived
deterministically from the decapsulation key's rejection seed. A validity flag
here would hand an attacker the exact bit the transform exists to withhold. The
seam has no failure channel for that reason, and the input checks that reject a
malformed key route through the same path rather than raising.

Both comparisons are arithmetic reductions over the full input, never an early
exit and never a `lax.cond`.

## Nonce discipline belongs to the caller

A scheme takes a nonce; it does not generate one. What a scheme assumes about
uniqueness, and what breaks when that is violated, is stated on its page — and
"what breaks" is not uniform: reusing a nonce under one GCM key leaks the
authentication key `H` and forges everything under it thereafter, which is
sharper than the corresponding ChaCha20-Poly1305 failure.

One nonce per batch entry, and a scheme that can offer a wider nonce says why a
caller would want it.

## Keys and ciphertexts are bytes at the seam

They cross as `uint8` arrays in the standard's encoding, not as scheme-named
pytrees: a consumer holds bytes, and a seam taking a structured form would make
it call a scheme-specific decode first — which means naming the scheme, which is
what the seam exists to prevent. A scheme parses its own encoding on entry.

Inside a scheme, whatever crosses a `jit` / `vmap` boundary is a registered
frozen dataclass, and a scheme instance carried as pytree aux needs value-based
`__eq__`/`__hash__`. Identity equality does not error; it silently re-traces the
enclosing zone for every freshly built instance, so it surfaces as a slow call
and never as a failure.

Randomness is an argument, never sampled inside a traced function. A standard's
derandomized entry point — ML-KEM's `encaps_internal` — lives below the seam on
the scheme, because the harness that needs it already names the scheme.

## Cite the standard, by section

A magic constant, a domain separator, a padding rule, or a bit order carries the
document and section it comes from: `# FIPS 203 §4.2.1`, not `# compression`.
This code is easy to write plausibly and wrongly, and the section number is what
lets a reviewer check it rather than agree with it.

Bit and byte order deserve the citation most. GCM specifies its field elements in
reflected order while `binary_field_ghash` is the natural basis, and a convention
mismatch produces a wrong answer for every input while looking like a field bug.

## Known-answer tests are the gate

A scheme that reproduces every published ciphertext has proven nothing about
rejection. An `open` that returns `ok = True` unconditionally passes every
positive vector ever published, and a `decaps` whose rejection path is dead
passes them too — that path is only reached when the ciphertext is wrong.

So the negative cases are half the gate, and they are per seam:

- **AEAD** — a flipped bit in the tag, the ciphertext, the associated data, and
  the nonce each set `ok = False` *and* mask the plaintext.
- **KEM** — a malformed ciphertext produces the implicit-rejection secret, the
  specific expected value, not merely something different. Those cases are not
  labelled: ACVP publishes an expected shared secret for every decapsulation
  case and marks none of them as a rejection, because under implicit rejection
  there is no verdict to mark. Comparing the secret exactly is what gates them,
  so a run that never reaches decapsulation has never executed the path.

**Mixed-validity batches are their own case.** A batch where entries 3 and 7 fail
must mask 3 and 7 and nothing else. A masking bug applied batch-wide passes every
all-valid and every all-invalid set, so a suite without a mixed batch has not
tested the batch axis at all.

Self-consistency is not evidence either. Seal-then-open round-trips forever
inside a self-consistent wrong implementation. Property-based tests supplement
the KATs; they never replace them.

An exhaustive sweep — every parameter set against every published vector — is
tagged `slow_kat`, which drops it from the per-PR run and keeps it in the
scheduled one.

### A vector is fetched, never transcribed

Hex copied by hand is not a test vector. Fetch the standard's text or the
published JSON and extract the values programmatically, and when a vector already
in the tree turns out to disagree, **diff it against the source rather than
replacing it** — the diff is what distinguishes one wrong byte from a dropped
nibble that shifted everything after it.

RFC hex dumps parse with `^\s*[0-9]{3}\s+((?:[0-9a-f]{2} ?)+?)\s{2,}`. A
plaintext that is awkward to transcribe — non-ASCII, smart quotes — is a signal
to extract it as hex, not to weaken the assertion to a round trip.

The reason is not tidiness. When a hand-copied vector fails, nothing says whether
the vector or the implementation is wrong, and that ambiguity costs more than the
fetch. What resolves it cheaply is having a second gate that already passes: an
implementation agreeing with an independent reference across hundreds of inputs
is not wrong about one more published case.

### Vectors are fetched and pinned, never committed

A vector set is declared in [`MODULE.bazel`](../../MODULE.bazel) with a sha256 and
reaches a test through `data` — `http_file` for a set published as loose files,
`http_archive` for one published as an archive, as CAVP's are. The published sets
run to tens of megabytes, and committing them taxes every clone forever for data
that never changes after publication. The sha256 means a swapped or truncated
fetch fails the build rather than silently changing what a scheme is gated on, and
the repository cache makes every build after the first offline.

Where the source is a git tree, pin the URL to a commit and never a branch: NIST
regenerates ACVP's files in place, and a moving URL turns an upstream
regeneration into a mystery failure here. Where there is no commit to pin — a
CAVP archive is published once and not regenerated — the sha256 carries that
weight alone, so say so at the declaration rather than leaving the reader to
wonder whether the pin was forgotten.

An archive costs two things a loose file does not. It needs a `build_file_content`
filegroup, since the fetched tree has no `BUILD` of its own; and its runfiles path
is `<repo>/<name>` rather than the `<repo>/file/<name>` an `http_file` produces,
which is the shape a test's `Rlocation` call has to ask for.

## Scheme doc skeleton

Every page in [`../schemes/`](../schemes) answers three things, and everything
else is optional — don't pad to fill a template.

- **What the standard fixes and what this implementation chooses.** Parameter
  sets, encodings, and bit orders are the standard's; the batching, the loop
  shapes, and the pytree layout are this repo's.
- **Where the batch axis is, and where it is not.** Which operations batch, and
  which parts are sequential within a message.
- **What leaks, and what the caller owes.** The scheme's data-dependent
  operations by name (see [`security.md`](security.md)), and its nonce
  assumption.
