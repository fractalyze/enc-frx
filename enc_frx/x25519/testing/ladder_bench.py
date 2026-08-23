# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What the traced byte boundary costs the X25519 ladder.

`x25519.py` runs the RFC 7748 ladder over `zk_dtypes.curve25519_bf_mont`.
RFC bytes are the canonical storage of `curve25519_bf`, so they cross into the
trace as a free `view` and reach Montgomery storage by an `astype` — a
boundary that sits *inside* the compiled function, and is therefore something
the compiler can either fuse away or not.

Two arms, sharing `x25519._ladder` rather than re-implementing it, so the only
difference between them is where the field material comes from:

- **mont** — field elements built on the host and handed straight to
  `x25519._ladder`. No boundary at all: the kernel ceiling.
- **wire** — `x25519.x25519` exactly as shipped, bytes in and bytes out, so
  its rows are that ceiling plus the two converts *and* the scalar-bit
  expansion, which `_mont_material` does eagerly for the other arm. The
  asymmetry charges wire for work mont skips, so a ratio at 1.0 is a
  conservative reading of the boundary rather than a flattering one.

**`wire / mont` is the number this bench exists for, and it should stay near
1.0.** If a future pin stops fusing the variant convert into the ladder's
`fori_loop`, this is where it shows up as a number rather than as a slowdown
nobody attributes — but note nothing runs this automatically (it is a
`py_binary`, and CI builds it without executing it), so the check happens
when someone runs it.

## What it measures now

frx 0.10.2.dev20260822150923, RTX 5090 + Ryzen 9 9950X:

```
            CUDA                        CPU
   B   mont     wire   ratio     mont     wire   ratio
   1   3.81ms   3.82ms  1.00     0.08ms   0.11ms  1.34
  32   2.73ms   2.76ms  1.01     1.76ms   1.77ms  1.01
 256   2.14ms   2.15ms  1.00    13.27ms  13.58ms  1.02
1024   2.16ms   2.17ms  1.01    53.44ms  53.76ms  1.01
8192   2.18ms   2.22ms  1.02   128.62ms 128.47ms  1.00
```

The ratio is 1.0 on both legs across the sweep — the boundary is free, which
is the whole claim. CUDA is flat from B = 256 (0.3us/op at 8192, dispatch
floor 0.032ms); CPU saturates around 53us/op and stops improving past B = 32.

Read the ratio, not the millisecond: CPU rows move several percent run to run,
and the B = 1 row is the arms' asymmetry rather than the boundary — 0.02ms of
scalar-bit expansion is most of a 0.09ms ladder there, and nothing at that
batch size is dispatch-bound enough to hide it.

## Why the limb field is gone

It spelled GF(2^255 - 19) out in 16 limbs of radix 2^16 because no registered
curve25519 field existed when it was written. Measured against it on
frx 0.10.2.dev20260822060712 — the last wheel where both arms coexisted, this
file's history has the four-arm table — the wire arm won 5.6-27x on CUDA and
17-58x on CPU across B = 1..8192, tracking the mont ceiling to within ~2% on
both legs. Those ratios are not reproducible from here, the limb arm having
left with `field.py`; the ceiling-tracking half of the claim is, and is what
the table above re-measures every run.

Three upstream fixes had to land first, and the third is the one worth
remembering. A convert between a field's storage forms had to stop being
treated as a no-op (fractalyze/xla#568) and had to lower inside a
`static_while` fusion rather than only standalone (fractalyze/xla#573) —
before that, this arm did not compile on CUDA at all. Then
`curve25519_bf_mont` multiply turned out to be **wrong for roughly 1 operand
pair in 200,000, always low by exactly 1** (fractalyze/xla#542), which every
fixed vector in this file passed while corrupting ~1.4% of X25519 calls; it
was a multi-limb Montgomery reduction whose result bound could cross 2^256
and lose the bit silently (fractalyze/prime-ir#434).

That last one is why the retire is gated on RFC 7748 §5.2's *iterated* vector
in `x25519_test`, not on the fixed vectors here. A bench verifies that an arm
computes the right function before it earns a row; it is not the correctness
gate, and for a rare-rate arithmetic fault it structurally cannot be.

Run:
    bazel run //enc_frx/x25519/testing:ladder_bench
    bazel run //enc_frx/x25519/testing:ladder_bench -- --batches=1,256,8192
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl import app, flags
from frx import Array

from enc_frx.x25519 import x25519
from enc_frx.x25519.testing import rfc7748_reference

_BATCHES = flags.DEFINE_list(
    "batches", ["1", "32", "256", "1024", "8192"], "batch sizes to sweep"
)
_REPEATS = flags.DEFINE_integer("repeats", 20, "warm calls to average over, at most")
_BUDGET = flags.DEFINE_float(
    "budget", 2.0, "seconds to spend on warm calls before averaging what ran"
)
_MUL_BATCH = flags.DEFINE_integer(
    "mul_batch", 1 << 16, "elements for the single-multiply rows; 0 skips them"
)

_P = 2**255 - 19

# Seconds of real ladder work before the dispatch floor is timed — an unwarmed
# floor reports ~8x the warm number on GPU (measured by ml_kem's kem_bench,
# whose floor this reuses the shape of).
_FLOOR_WARMUP_S = 0.5


def _mont_material(scalar: np.ndarray, u: np.ndarray) -> tuple[Array, Array]:
    """`x25519._ladder`'s arguments, built on the host — the ceiling arm's
    inputs, with the byte boundary done in numpy instead of in the trace."""
    batch = scalar.shape[:-1]
    bits = x25519._scalar_bits(x25519._clamp(fnp.asarray(scalar)))
    masked = u.copy()
    masked[..., 31] &= 0x7F
    x1 = np.array(
        [[int.from_bytes(bytes(row), "little")] for row in masked.reshape(-1, 32)],
        dtype=x25519.WORK,
    ).reshape(*batch, 1)
    return bits, fnp.asarray(x1)


def _say(text: str) -> None:
    print(text, flush=True)


def _time(fn: Callable[..., Any], *args: Any) -> tuple[float, float] | None:
    """`(compile_s, warm_s)` for one jitted call site, or `None` where the
    device refuses the shape — kem_bench's protocol, in miniature."""
    compiled = frx.jit(fn)
    try:
        start = time.perf_counter()
        frx.block_until_ready(compiled(*args))
        cold = time.perf_counter() - start

        start = time.perf_counter()
        runs = 0
        while runs < _REPEATS.value:
            frx.block_until_ready(compiled(*args))
            runs += 1
            if time.perf_counter() - start >= _BUDGET.value:
                break
        warm = (time.perf_counter() - start) / runs
    except (RuntimeError, MemoryError) as error:
        _say(f"    ! {type(error).__name__}: {str(error).splitlines()[0][:96]}")
        return None
    return max(cold - warm, 0.0), warm


def _verify() -> None:
    """Both arms against RFC 7748 §5.2 and six random vectors, before any
    timed row — a fast arm that computes the wrong function has no row to
    show. See the module docstring for why this is not the correctness gate.
    """
    vectors = [
        (
            "a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4",
            "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c",
        ),
        (
            "4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d",
            "e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493",
        ),
    ]
    rng = np.random.default_rng(0)
    k = np.vstack(
        [np.frombuffer(bytes.fromhex(kh), dtype=np.uint8) for kh, _ in vectors]
        + [rng.integers(0, 256, (6, 32), dtype=np.uint8)]
    )
    u = np.vstack(
        [np.frombuffer(bytes.fromhex(uh), dtype=np.uint8) for _, uh in vectors]
        + [rng.integers(0, 256, (6, 32), dtype=np.uint8)]
    )
    expect = [rfc7748_reference.x25519(bytes(k[i]), bytes(u[i])) for i in range(len(k))]

    wire = np.asarray(frx.jit(x25519.x25519)(k, u))
    mont = np.asarray(frx.jit(x25519._ladder)(*_mont_material(k, u)))
    for i, want in enumerate(expect):
        assert bytes(wire[i]) == want, f"wire wrong at row {i}"
        assert int(mont[i, 0]).to_bytes(32, "little") == want, f"mont wrong at row {i}"
    _say(f"both arms match RFC 7748 §5.2 + {len(k) - 2} random vectors")


def _dispatch_floor() -> tuple[float, float] | None:
    """One trivial jitted call after sustained real work — kem_bench's floor,
    warmed by the ladder itself."""
    warm_up = frx.jit(x25519.x25519)
    rng = np.random.default_rng(1)
    k = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    u = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    start = time.perf_counter()
    while time.perf_counter() - start < _FLOOR_WARMUP_S:
        frx.block_until_ready(warm_up(k, u))
    return _time(lambda x: x + np.int32(1), fnp.asarray(np.zeros(1, dtype=np.int32)))


def _sweep(batches: list[int]) -> None:
    rng = np.random.default_rng(2)
    _say(
        "\nx25519 by batch size  (mont = host-entered field material, the "
        "ceiling; wire = the shipped path, boundary in the trace)"
    )
    _say(
        f"  {'B':>6}  {'mont warm':>10}  {'wire warm':>10}  "
        f"{'wire/mont':>9}  {'wire/op':>9}"
    )
    last: tuple[float, float] | None = None
    for batch in batches:
        k = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
        u = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
        mont = _time(x25519._ladder, *_mont_material(k, u))
        wire = _time(x25519.x25519, k, u)
        if mont is None or wire is None:
            _say(f"  {batch:>6}  (unavailable at this batch size)")
            continue
        last = (mont[0], wire[0])
        _say(
            f"  {batch:>6}  {mont[1] * 1e3:>8.2f}ms  {wire[1] * 1e3:>8.2f}ms  "
            f"{wire[1] / mont[1]:>8.2f}x  {wire[1] / batch * 1e6:>7.1f}us"
        )
    if last:
        _say(
            f"  compile, at the largest measured B: "
            f"mont {last[0]:.2f}s, wire {last[1]:.2f}s"
        )


def _single_multiply(count: int) -> None:
    """One field multiply over `count` elements, per storage form — the
    operation the ladder rows are made of, without the ladder around it."""
    rng = np.random.default_rng(3)
    values = [
        int.from_bytes(bytes(row), "little") % _P
        for row in rng.integers(0, 256, (2 * count, 32), dtype=np.uint8)
    ]
    _say(f"\none multiply over {count} elements")
    for name, dtype in (("bf", x25519.WIRE), ("mont", x25519.WORK)):
        left = fnp.asarray(np.array(values[:count], dtype=dtype))
        right = fnp.asarray(np.array(values[count:], dtype=dtype))
        timing = _time(lambda x, y: x * y, left, right)
        if timing is None:
            continue
        _say(
            f"  {name:>6}  compile {timing[0]:6.2f}s  warm {timing[1] * 1e3:8.3f}ms  "
            f"{count / timing[1] / 1e6:8.1f} Mmul/s"
        )


def main(argv: list[str]) -> None:
    del argv
    _say(f"backend={frx.default_backend()}  devices={frx.devices()}")
    _verify()
    floor = _dispatch_floor()
    if floor is not None:
        _say(f"dispatch floor: {floor[1] * 1e3:.3f}ms warm")
    if _MUL_BATCH.value:
        _single_multiply(_MUL_BATCH.value)
    _sweep([int(b) for b in _BATCHES.value])


if __name__ == "__main__":
    app.run(main)
