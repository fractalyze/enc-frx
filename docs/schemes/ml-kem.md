# ML-KEM (FIPS 203)

Design notes for the lattice KEM. The API is the [`Kem`](../../enc_frx/kem.py)
seam plus [`enc_frx/ml_kem/`](../../enc_frx/ml_kem); this page is what the
modules cannot say about themselves.

## What the standard fixes, and what this implementation chooses

The standard fixes everything observable: `q = 3329`, the ring
`Z_q[X]/(X^256 + 1)`, `zeta = 17` and the `BitRev7` order its NTT lands in, the
little-endian bit order of `ByteEncode`, the round-half-up definition of
`Compress`, and the domain separation of every hash. None of those is a degree
of freedom, and each is cited by section where it appears.

What this implementation chooses is the shape: where the batch axis is, which
loops are unrolled, and — the one choice with a security argument attached —
**how much XOF output `SampleNTT` squeezes**.

### The fixed XOF block

`SampleNTT` is rejection sampling, so the bytes it consumes depend on their own
values, and **a traced program has no data-dependent trip count** — the
standard's `while` cannot be transcribed. It is replaced by a fixed squeeze of
five SHAKE128 blocks, masked and compacted, which leaves a `2^-261` chance of
the budget being too small.

See [`sampling.py`](../../enc_frx/ml_kem/sampling.py) for where that number
comes from: the per-block tail table, why five rather than four, and why a miss
is a deterministic wrong answer rather than an error. What follows here is the
part the module cannot state about itself.

### Why a fixed budget is defensible here and would not be everywhere

**`SampleNTT`'s seed is `rho`, and `rho` is public** — it ships in the
encapsulation key. So the number of XOF bytes the sampler consumes, and the time
it takes, are functions of data an attacker already holds. Over-squeezing to a
fixed bound therefore costs nothing in secrecy, and the bound may be chosen for
throughput.

The same construction over a secret seed would be a different conversation. That
is why the two samplers here are shaped differently rather than uniformly:
`SamplePolyCBD`'s input is `PRF_eta` output derived from the secret seed, so it
has no rejection step at all — a fixed `64*eta` bytes in, 256 coefficients out,
pure bit arithmetic with no branch. The asymmetry is FIPS 203's design, and
flattening the two into one "sampler" abstraction would lose the reason for it.

## Where the batch axis is, and where it is not

Everything is batch-first over leading axes; there is no scalar entry point.

- **The NTT** transforms `[..., 2, 128]` in one opcode call, so both halves of
  one polynomial — and a whole `k x k` matrix of them — are one call rather than
  `2k^2`.
- **The matrix expansion** issues all `k^2` `SampleNTT` calls as one XOF batch.
  The entries share nothing but the seed prefix, so the only thing that would
  serialize them is writing the loop.
- **K-PKE's matrix-vector products** are one `base_mul` over a broadcast pair and
  one sum over the column axis, for any `k` — see
  [`_k_pke.py`](../../enc_frx/ml_kem/_k_pke.py). Writing `Â ∘ ŝ` as a loop over the
  rows would issue `k` of each and serialize work that shares nothing.
- **The samplers' compaction is a gather, not a scatter**, because XLA
  serializes a large scatter on GPU — see
  [`sampling.py`](../../enc_frx/ml_kem/sampling.py) for the construction.
- **Sequential within a message:** nothing. ML-KEM has no Horner chain, which is
  what separates it from Poly1305 and GHASH.

## What leaks, and what the caller owes

Read [`../reference/security.md`](../reference/security.md) first; this repo
makes no constant-time claim, and the items below are named rather than fixed.

- **`SampleNTT`'s rejection pattern is data-dependent** in the sense that which
  candidates are kept depends on the XOF stream. It is not secret-dependent: the
  stream comes from the public `rho`. The fixed budget also means the *amount* of
  work does not vary at all, so this operation's timing is constant as a side
  effect rather than as a claim.
- **`SamplePolyCBD` is branch-free by construction**, because its input is
  secret-derived. That is a design constraint on this module, not an
  optimization, and a `where` on a coefficient value here would be a bug.
- **Decapsulation takes adversary-chosen input against a long-lived key.** That
  is the inverse of a signature verifier's posture and the reason the rejection
  path is a select over a full-width comparison rather than a branch.
- **The caller owes randomness.** `encaps` takes it as an argument; nothing here
  samples it. The derandomized entry point the known-answer tests need lives
  below the seam.

## The gate

C2SP's CCTV vectors, fetched and sha256-pinned in
[`../../MODULE.bazel`](../../MODULE.bazel), in two sets that answer different
questions.

**`intermediate/`** publishes `rho`, `sigma`, the matrix `A`, and the secret and
error vectors as separate labelled values, which is what pins each sampler on its
own. ACVP cannot: it publishes `(d, z) -> (ek, dk)` and nothing between, so it
gates `SampleNTT` and `SamplePolyCBD` only as far as a whole key generation gates
every step inside it — jointly, and only once key generation exists.

All three parameter sets are loaded rather than one, because `eta1` is 3 for
ML-KEM-512 and 2 for the other two. A single file gates one of the two
centered-binomial widths and silently skips the other, and the skip is invisible:
the width that does run passes, and nothing reports the one that did not.

**`unluckysample/`** publishes seeds found by search whose rejection run is the
worst known — 384 candidates against a mean of 315, or 576 bytes. They are the
only published vectors that can see a fixed XOF budget being too small, because
every ordinary vector fits in three SHAKE128 blocks. An undersized implementation
therefore passes the entire rest of the corpus and fails only here.

### One line the CCTV vectors cannot gate

Both sets predate the final standard in exactly one place: they expand
`(ρ, σ) ← G(d)`, where FIPS 203 Algorithm 13 line 1 is `G(d ‖ k)`. The
parameter-set byte binds a key to the `k` it was generated under, and it is the
one thing about key generation no vector loaded here can see — every value the
files publish begins at `ρ` and `σ`, and everything from there on matches the
final standard, the `SampleNTT` index order included.

So key generation is gated in two pieces rather than one: the lattice work
against the vectors, entering below the expansion, and the expansion itself
against `hashlib`, which is where SHA-3 is established for this repo anyway. The
arrangement is worth knowing before reading either test, because the wrong repair
is available and looks right — feeding the files' `d` to `key_gen` fails, and the
standard-conforming code is what appears to be at fault.
