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

### The three parameter sets

§8 approves three, and [`params.py`](../../enc_frx/ml_kem/params.py) names them
as `ML_KEM_512`, `ML_KEM_768` and `ML_KEM_1024`. A consumer names one at
construction — `MlKem(ML_KEM_768)` — and no call site names it again.

| | k | η₁ | η₂ | d_u | d_v | category | RBG strength |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ML-KEM-512 | 2 | **3** | 2 | 10 | 4 | 1 | 128 |
| ML-KEM-768 | 3 | 2 | 2 | 10 | 4 | 3 | 192 |
| ML-KEM-1024 | 4 | 2 | 2 | **11** | **5** | 5 | 256 |

The bold entries are why all three are gated rather than one. `η₁ = 3` at
ML-KEM-512 is the second centered-binomial width, and `d_u`/`d_v` at ML-KEM-1024
are the second compression width, so two of the three exercise a code path the
default set never reaches — not merely a different array width. A suite run at
ML-KEM-768 alone passes with either of those paths broken.

| | encapsulation key | decapsulation key | ciphertext | decapsulation failure |
| --- | --- | --- | --- | --- |
| ML-KEM-512 | 800 | 1632 | 768 | 2⁻¹³⁸·⁸ |
| ML-KEM-768 | 1184 | 2400 | 1088 | 2⁻¹⁶⁴·⁸ |
| ML-KEM-1024 | 1568 | 3168 | 1568 | 2⁻¹⁷⁴·⁸ |

Sizes are Table 3's, in bytes, and the shared secret is 32 at every set. The
failure rates are Table 1's (§3.2): the probability that an *honest* exchange
disagrees, which is not a rejection and produces no signal that it happened. They
are far below anything reachable, and they are here because the number a reader
would otherwise assume is zero.

§8 recommends ML-KEM-768 as the default. This repo has no default: the seam
takes a set, because a scheme that picked one for the caller is a security
decision made where nobody reviews it.

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

## One key, many ciphertexts

The seam's `decaps` takes one key *per* batch entry, which is the general shape
and the only one a `Kem` consumer can express. A server decapsulating under one
long-lived key is a narrower case, and it is worth naming because `Â` is a
function of `ρ` alone: at `B` ciphertexts the general path expands `[B, k, k,
256]` of a matrix whose rows are all equal. Nothing dedupes that, and nothing
can — `ρ` is traced data rather than shape, so a batch whose keys happen to be
equal is indistinguishable at trace time from one whose keys are not.

`MlKem.precompute_decaps` / `MlKem.decaps_precomputed` are that narrower
operation, below the seam on `MlKem` alone. The `Kem` protocol does not gain a
method: a protocol returning a scheme-shaped opaque value would make generic
code hold ML-KEM-shaped state, which is the coupling the seam exists to prevent,
and it would not generalize — a hybrid KEM has no `Â`.

- **It is narrower, not faster.** Passing per-entry keys to it would be a
  different operation, so `precompute_decaps` takes a rank-1 key and raises on a
  leading axis rather than broadcasting. The restriction is a shape error, not a
  docstring.
- **The parsed value carries the key checks.** §7.2 and §7.3 are functions of
  the key, so a key parsed once is checked once — but `precompute_decaps` still
  raises nothing on a malformed key. The verdict rides the parsed value and is
  AND-ed into every later acceptance, so a bad key reaches the same rejection
  secret it reaches through `decaps`. A `precompute` that validated eagerly and
  raised would put back exactly the bit the FO transform withholds, at a new
  door.
- **What it *derives* is public; the value as a whole is still secret.** `Â`,
  `t̂` and `H(ek)` all descend from `ek`, which travels in the clear — that is
  what makes hoisting them a performance question rather than a security one.
  But the parsed value also carries `dk_PKE` and `z`, which are `dk`'s secret
  halves, so it is handled as `dk` is. What the design avoids is a *parsed*
  secret: `ŝ` is deliberately not decoded into the value, `dk_PKE` is carried as
  the bytes the key already contained, and `z` is untouched, so nothing in it is
  more exposed than the key the caller already holds.
- **`H(ek)` is in it because it is the only flat hash that can be.** `H`, `J` and
  `G` are each a single sponge over the whole batch and none is the expansion's
  size, but `H(ek)` is the one member of the group that does not need the
  ciphertext — so it is the one a per-key precompute can hoist at all. Hoisting
  it costs a hash the value already holds the input for.

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
- **The caller owes randomness, at a strength the parameter set fixes.** `encaps`
  takes it as an argument; nothing here samples it. Table 2 states a required RBG
  strength per set — 128, 192, 256 bits — and since the randomness crosses the
  seam as bytes, nothing in this repo can see where they came from. A set chosen
  for category 5 and seeded from a 128-bit generator is category 1, and every
  vector still passes. The derandomized entry point the known-answer tests need
  lives below the seam.
- **The caller owes the encapsulation-key check.** FIPS 203 §7.2 places it at key
  import, and a seam whose keys are bytes on every call has no import step, so
  `encaps` checks the length and nothing else — a length is static and can raise,
  a coefficient's range is data and cannot. `MlKem.check_encapsulation_key` is
  that check as a per-entry value, and a caller taking keys off a wire runs it
  when the key arrives rather than per encapsulation. `decaps` owes the caller
  nothing here: it holds the rejection seed, so it folds §7.2 and §7.3 into the
  same reduction as the ciphertext comparison and a malformed key comes back as
  a rejection secret.

## The gate

Two published corpora, fetched and sha256-pinned in
[`../../MODULE.bazel`](../../MODULE.bazel), and they are not redundant. One is
generated against the final standard and gates the scheme end to end; the other
publishes the values in between, which is what pins each piece on its own.

### NIST's ACVP sets

`ML-KEM-keyGen-FIPS203` publishes `(d, z) -> (ek, dk)` and
`ML-KEM-encapDecap-FIPS203` publishes the rest, at all three parameter sets.
Every case runs on every PR through the harness's `check_kem`
([`../../enc_frx/testing/kat.py`](../../enc_frx/testing/kat.py)), which adds the
properties a published file cannot express: that the whole ciphertext is
consumed, that rejection repeats, and that it is decided per batch entry.

Two things about the set shape the tests around it.

**No decapsulation case is marked as a rejection**, and that is correct rather
than an omission — under implicit rejection a modified ciphertext yields a shared
secret, so there is no verdict to publish. Every case carries an expected `k`, so
comparing it exactly *is* the whole gate. There is no subset to single out, and
no way to count or filter the rejections; what the harness enforces instead is
that a run reaches decapsulation at all.

**`encapDecap` publishes four functions, not two.** Beyond encapsulation and
decapsulation it carries `encapsulationKeyCheck` and `decapsulationKeyCheck` —
the §7.2 and §7.3 input validation, and the only groups here with a published
verdict. The `Kem` seam names no validation operation, so the harness refuses
them rather than running them through decapsulation, and they drive
`MlKem`'s two predicates instead, one per section. Each group is half valid and
half invalid, which makes it a published mixed-validity batch as well.

### C2SP's CCTV vectors

Two sets that answer different questions, and both publish what ACVP does not:
the labelled values between `(d, z)` and `(ek, dk)`.

**`intermediate/`** publishes `rho`, `sigma`, the matrix `A`, and the secret and
error vectors as separate labelled values, which is what pins each sampler on its
own. ACVP cannot: it publishes `(d, z) -> (ek, dk)` and nothing between, so it
gates `SampleNTT` and `SamplePolyCBD` only as far as a whole key generation gates
every step inside it — jointly, and never by name.

All three parameter sets are loaded rather than one, because `eta1` is 3 for
ML-KEM-512 and 2 for the other two. A single file gates one of the two
centered-binomial widths and silently skips the other, and the skip is invisible:
the width that does run passes, and nothing reports the one that did not.

**`unluckysample/`** publishes seeds found by search whose rejection run is the
worst known — 384 candidates against a mean of 315, or 576 bytes. They are the
only published vectors that can see a fixed XOF budget being too small, because
every ordinary vector fits in three SHAKE128 blocks. An undersized implementation
therefore passes the entire rest of the corpus and fails only here.

### The one negative the files publish

`intermediate/` also carries the whole of one encapsulation and its
decapsulation — `ek`, `dk`, `m`, `c`, `K` — and one value past them: `KBar`, the
implicit-rejection secret `J(z ‖ c)` for that same `c`. It is the only published
value a correct rejection path produces and a broken one does not, and rejection
is otherwise unobservable: without an oracle for the rejection secret, a
wrong-but-different answer looks exactly like a correctly rejected one.

Reaching it takes a *key* that fails a check rather than a corrupted ciphertext.
`KBar` is derived from the file's own `c`, so changing `c` changes the expected
answer along with it; changing `dk`'s `H(ek)` field fails the §7.3 hash check
while leaving `z` and `c` alone, and lands on the published value exactly.

### One line the CCTV vectors cannot gate

Both CCTV sets predate the final standard in exactly one place: they expand
`(ρ, σ) ← G(d)`, where FIPS 203 Algorithm 13 line 1 is `G(d ‖ k)`. The
parameter-set byte binds a key to the `k` it was generated under, and it is the
one thing about key generation no CCTV value can see — every value those files
publish begins at `ρ` and `σ`, and everything from there on matches the final
standard, the `SampleNTT` index order included.

So the two corpora enter key generation at different points, and that is the
division of labor between them: CCTV enters below the expansion, at
`_key_pair`, and ACVP's `keyGen` set gates the expansion along with everything
under it, from `d` to `ek` and `dk`. The arrangement is worth knowing before
reading either test, because the wrong repair is available and looks right —
feeding CCTV's `d` to `key_gen` fails, and the standard-conforming code is what
appears to be at fault.
