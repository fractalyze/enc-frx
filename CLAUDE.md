# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file is the map plus the rules every change must respect.

- **Project overview, build, and dev setup:** [`README.md`](README.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Security posture — what this repo claims, and what that rules out as a
  use:** [`docs/reference/security.md`](docs/reference/security.md)
- **Coding conventions — what implementing a scheme here requires, and only
  that:** [`docs/reference/conventions.md`](docs/reference/conventions.md)
- **Per-scheme design notes:** [`docs/schemes/README.md`](docs/schemes/README.md)
- **The two seams every scheme implements:** [`enc_frx/kem.py`](enc_frx/kem.py)
  and [`enc_frx/aead.py`](enc_frx/aead.py)
- **Detailed design & open decisions:** tracked on GitHub — epic issue
  [fractalyze/enc-frx#1](https://github.com/fractalyze/enc-frx/issues/1).

## Three non-negotiables

- **Standards-exact, or it is not done.** Every scheme reproduces its
  specification byte for byte, gated on the published known-answer tests
  including the negative ones. An `open` that returns `True` unconditionally
  passes every positive vector, and a `decaps` whose rejection path is dead
  passes them too — that path is only reached when the ciphertext is wrong.
- **Batch-parallel `decaps` and `open`.** They are the hot paths and they are
  trivially parallel, so a batch of `B` runs in one call. Neither seam has a
  scalar entry point on purpose: a single operation is `B = 1`, and a Python
  loop over the batch axis is a bug rather than a slow implementation.
- **Failure is a value, and the seams disagree on purpose.** `Aead.open` returns
  a validity flag and masks the plaintext of a failing entry; `Kem.decaps` has no
  failure channel at all, because implicit rejection requires a wrong ciphertext
  to yield a different shared secret rather than an error. Making them look alike
  would be the most dangerous change in the repo.

## Measuring a change

Both arms of a performance A/B belong in **one process**. Comparing a run of the
tree before a change against a run after it has produced the wrong *sign* here,
not merely the wrong magnitude — compile-cache and process state dominate
differences of a few percent, which is the size most of these changes are. Two
arms that cannot share a process do not produce a quotable number.

Warm-ups get the same treatment: confirm yours moved the number before building
on it. `kem_bench`'s dispatch floor needs sustained work, and the single-call
warm-up that looks obviously sufficient changes nothing.

**A one-time cost is reported as a break-even, not folded into a ratio.** Work
hoisted out of a per-call path onto a per-key or per-session setup is only a win
after enough calls to repay the setup, and that count is the number a caller
needs. It is also where the backends disagree: an unbatched setup call is
latency-bound on GPU, so its cost is flat across parameter sets and can exceed a
whole batch, while the same call on CPU is real work that scales and repays
inside the first batch. A speedup column alone hides a case that is slower.

**Derive how many warm calls are behind each figure before quoting it.** The
harness averages against a wall-clock budget, so the largest batch sizes silently
collapse toward a single call while looking exactly like the well-sampled rows.
A ratio that breaks its own table's trend is a sampling artifact until shown
otherwise.

The hot paths here hold a long-lived secret key and take adversary-chosen input,
which is the inverse of `sig-frx`. That has consequences for every scheme, so
read [`docs/reference/security.md`](docs/reference/security.md) before
implementing one.
