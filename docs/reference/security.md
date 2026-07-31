# Security posture

What this repo claims, and — more importantly — what it does not. Read this
before implementing a scheme; it decides what each one owes.

## The posture

**Batch-processing grade.** Encrypting and decrypting data you already hold is
the supported path: bulk re-encryption, test-vector work, research, and anything
where the ciphertexts are yours rather than an adversary's.

**This is not a decryption oracle.** Do not stand it up as a service that
decapsulates or opens adversary-supplied ciphertexts under a long-lived key on a
machine an adversary can measure.

That sentence is the whole document. The rest explains why it cannot be softened.

## Why the sibling repo's argument does not carry over

[`sig-frx`](https://github.com/fractalyze/sig-frx) is verification-grade, and it
gets there honestly: verification consumes a public key, a message, and a
signature, all public. There is no secret input, so execution time and the
address stream carry nothing an adversary did not already have.

Here the hot path is `decaps` and `open`. Both hold a long-lived secret key, and
both take input an adversary chose. Every term in that argument flips, so the
conclusion does not transfer — and the failure mode is specific rather than
theoretical: a timing signal from ML-KEM's decapsulation is a key-recovery
attack, which is what KyberSlash and clangover exploited in hardened C
implementations.

## Why there is no constant-time claim

The implementation is traced by FRX and compiled by XLA. Every layer between the
source and the machine is free to rewrite it:

- The compiler folds constants, reassociates arithmetic, and picks lowerings.
  Nothing in that pipeline promises data-independent instruction selection.
- A `where` is a select at the source level, not contractually at the machine
  level, and a gather's timing depends on its address stream.
- None of it is stable across compiler versions or backends, so a claim
  established once would have to be re-established on every dependency bump.

There is no test in this stack that could establish a constant-time property, and
a claim nobody can test is worse than no claim: it gets read as a guarantee.

## What tracing buys, and where it stops

A traced program has **no data-dependent branching by construction** — there is
no secret-dependent `if` to leak through, because there is no `if`. That is real,
and it is why the schemes here are written the way they are: ML-KEM's implicit
rejection is an arithmetic select, and AES's S-box is inversion in
`binary_field_gf8_aes` rather than a table lookup, so neither introduces a
secret-dependent address.

It is also nowhere near sufficient. What remains is the address stream of any
gather, the machine-level shape of a `select`, and everything the compiler is
free to do between the two. Removing branching removes one channel, not the
category.

## What is explicitly not claimed

- **Constant-time execution**, anywhere, including in the schemes that were
  written to avoid secret-dependent memory access.
- **Resistance to fault injection, power and EM analysis, or microarchitectural
  attacks.**
- **Memory hygiene.** Secret material is not zeroized; a device buffer's lifetime
  belongs to the runtime.

## What each scheme owes

- **Do not claim what the repo does not.** The words "constant-time",
  "side-channel resistant", and "hardened" do not belong in a docstring, a
  comment, or a scheme page here.
- **A scheme's page states its own leaky operations by name**, rather than
  staying silent.
- **A failure is a value, never a branch.** An authentication check reduces to a
  mask, and a KEM's rejection path is a select — see
  [`conventions.md`](conventions.md). A `cond` on secret-derived data is a design
  bug, not a performance trade-off.
- **Reject before you accept.** An `open` that returns `True` unconditionally
  passes every positive known-answer test, and a `decaps` whose rejection branch
  is dead passes them too. The negative cases are half the gate.

## Changing the posture

A hardened decryption path is a different implementation, not a flag on this one:
it needs a language where the codegen is answerable to the source, and it cannot
share this one's compiler. Widening the claim therefore starts by deciding where
that implementation lives, not by tightening the code here.
