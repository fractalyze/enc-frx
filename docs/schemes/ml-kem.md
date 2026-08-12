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

`SampleNTT` is rejection sampling. Three SHAKE128 bytes give two 12-bit
candidates, each kept iff it is below `q`, until 256 coefficients are collected.
How many bytes that consumes depends on the bytes themselves, and **a traced
program has no data-dependent trip count** — so the standard's `while` cannot be
transcribed. It is replaced by a fixed squeeze, masked and compacted.

The budget comes from the tail of the acceptance distribution, not its mean.
Acceptance is `3329/4096 ~ 0.8127`, so 256 coefficients need ~315 candidates on
average and a budget sized from that would fail constantly:

| SHAKE128 blocks | bytes | candidates | `P(budget too small)` |
| --------------- | ----- | ---------- | --------------------- |
| 3               | 504   | 336        | `2^-6.9`              |
| 4               | 672   | 448        | `2^-105`              |
| **5**           | **840** | **560**  | **`2^-261`**          |

Five blocks, because that puts a miss below `2^-256` rather than merely below
`2^-128` — under the bar for ML-KEM-1024's category-5 claim and not only for the
smallest parameter set. The margin over four blocks costs one Keccak-f
permutation per matrix entry.

**A miss is deterministic rather than undefined.** Below 256 acceptances every
unfilled slot reads the final candidate and the result is a wrong array — the
same wrong array on every backend, because the compaction clamps its own index
instead of leaving the gather's out-of-bounds behaviour to XLA. That behaviour is
not the intuitive one and is worth not depending on: an unclamped gather fills
with `INT32_MIN` rather than saturating.

Nothing detects the miss and nothing should: threading a validity flag for a
`2^-261` event through the whole scheme would put a failure channel into
`Kem.decaps`, which must not have one — implicit rejection requires a wrong
ciphertext to yield a different shared secret rather than an error.

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
- **The samplers' compaction is a gather.** Selecting accepted candidates in
  order is a stream compaction, and the natural form — scatter each value to its
  rank — is the wrong one: XLA serializes a large scatter on GPU. The inverse is
  a `searchsorted` over the running acceptance count followed by a
  `take_along_axis`, which is a gather, with the binary search unrolled so its
  trip count is static too.
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

## Gating

The samplers are pinned against C2SP's CCTV vectors rather than against ACVP
alone, because ACVP publishes `(d, z) -> (ek, dk)` and nothing between: it gates
each sampler only as far as a whole key generation gates every step inside it.
CCTV's `intermediate/` files publish `rho`, `sigma`, `A`, `s`, `e`, `r`, `e1` and
`e2` as separate labelled values, so each sampler is pinned on its own. All three
parameter sets are loaded because `eta1` is 3 for ML-KEM-512 and 2 for the other
two, so one file gates one of the two centered-binomial widths.

CCTV's `unluckysample/` seeds gate the XOF budget, and nothing else does. They
are the worst rejection runs known — 384 candidates against a mean of 315, or 576
bytes — so a three-block implementation passes every other published vector and
fails only there.
