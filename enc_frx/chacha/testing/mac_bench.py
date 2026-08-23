# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What the traced byte boundary costs Poly1305.

`poly1305.py` evaluates RFC 8439 §2.5's polynomial over
`zk_dtypes.prime_field(2^130 - 5)` in Montgomery storage. Blocks are
little-endian and so is the field's canonical storage, so each one crosses into
the trace as a `view` and reaches Montgomery storage by an `astype` — a
boundary that sits *inside* the scan body, once per block, and is therefore
something the compiler can either fuse away or not.

Two arms, sharing `poly1305._absorb` rather than re-implementing it, so the
only difference is where the field material comes from:

- **mont** — blocks framed and converted on the host, handed to the scan
  already field-typed. The scan alone: the kernel ceiling.
- **wire** — `poly1305.mac` exactly as shipped, bytes in and bytes out, so its
  rows are that ceiling plus the block framing (`_blocks`: pad, reshape, and
  add the RFC's appended `1`) and the conversion into the field.

**`wire / mont` is the number this bench exists for, and unlike the x25519
ladder's it is not 1.0** — the plumbing is a real fraction of the work here.
The whole default sweep, on frx 0.10.2.dev20260823002126 (the pin this branch
runs on), RTX 5090 + Ryzen 9 9950X. Median of six passes, with the spread each
ratio showed across them, because most of these cells are not stable enough to
quote as a single number:

```
                CUDA                            CPU
   B    bytes   mont     wire   ratio       mont     wire   ratio
   1       64   0.05ms   0.08ms  1.10-1.85   0.01ms   0.02ms  1.19-3.85
   1     1024   0.22ms   0.24ms  1.04-1.14   0.01ms   0.04ms  2.38-5.55
   1    16384   2.89ms   2.91ms  0.97-1.01   0.04ms   0.10ms  1.83-3.73
 256       64   0.05ms   0.08ms  1.52-1.79   0.03ms   0.11ms  2.69-5.74
 256     1024   0.21ms   0.25ms  1.10-1.22   0.21ms   0.42ms  1.78-2.20
 256    16384   2.92ms   2.98ms  1.00-1.06   3.25ms   5.19ms  1.58-1.64
```

**Only three cells are tight enough to compare against a later run — the
16 KiB rows, minus CPU at B = 1 — so a pin bump is a regression only if it
moves one of those.** The rest is arithmetic, not fusion: a B = 1 CPU call is
0.01-0.10ms, and the `mont` column there is at the printed resolution, so a
0.01ms cell is anything from 0.005 to 0.014 and a ratio recomputed from the
printed columns is worth a factor of three on its own. Run it on an idle box,
too — a pass taken while another job held this one moved the CPU 16 KiB cell
to 2.10, further than any idle pass did.

What the sweep does establish is the shape, and where the spreads above are
tight the ratio falls monotonically with message length. A 16 KiB message is
1024 blocks, and framing them means materialising 1024 x 17 bytes and then
1024 x 32 — fixed plumbing that is most of a short call and a shrinking
fraction of a long one, still 37% of the CPU call at 16 KiB against a scan
that is already fast, and nearly free on CUDA there. So the ratio reads as a
standing question — how much of Poly1305 is byte plumbing — rather than as an
invariant to hold at 1.0, and a pin bump that changed the fusion story would
move it either way.

Nothing runs it automatically: it is a `py_binary`, and CI builds it without
executing it, so the table is only as fresh as the last person to run it.

## Why the limb field is gone

`poly1305.py` used to carry GF(2^130 - 5) by hand: ten limbs of radix 2^13 on
uint32 lanes, with a carry sweep and an accumulator bound of its own. The
layout was forced — no 64-bit integer lane, so a limb product had to fit
uint32 — and a registered field dtype removes the constraint.

Measured against it on frx 0.10.2.dev20260822150923 — the last build where
both layouts existed here, so unlike the table above it does not re-measure —
on an RTX 5090 + Ryzen 9 9950X. `dtype / limb` warm, lower is the dtype
winning:

```
   B    bytes    CPU     CUDA
   1       64   1.25     0.83
   1     1024   0.25     0.54
   1    16384   0.09     0.46
 256       64   0.52     0.80
 256     1024   0.14     0.65
 256    16384   0.12     0.59
```

The dtype won every cell but one, and the exception (B = 1, 64 bytes on CPU)
is 0.01ms against 0.02ms — both at the dispatch floor, measuring fixed
overhead rather than work. Throughput at 16 KiB x 256: the limb field
saturates at 0.09 GB/s on CPU, the dtype reaches 0.77.

The gap widens with message length because the limb path pays a ten-way
convolution plus a three-pass carry sweep per block where the dtype pays one
field multiply, and because the final `_reduce_fully` disappears entirely —
canonical storage is already the reduced residue.

That comparison is not reproducible from here, the limb arm having left with
the layout; the table above is what remains, and it re-measures every run.

Run — one invocation is one leg, so the two-leg table above takes two. A pass
is one whole invocation; the table is the per-column median of six of them per
leg, taken by hand, which is why a single run of your own lands inside the
spreads rather than on the medians:

    FRX_PLATFORMS=cuda bazel run //enc_frx/chacha/testing:mac_bench
    FRX_PLATFORMS=cpu  bazel run //enc_frx/chacha/testing:mac_bench
    FRX_PLATFORMS=cpu  bazel run //enc_frx/chacha/testing:mac_bench -- \
        --sizes=64,1024,65536
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

from enc_frx.chacha import poly1305
from enc_frx.chacha.testing import rfc8439_reference

_SIZES = flags.DEFINE_list(
    "sizes", ["64", "1024", "16384"], "message sizes in bytes to sweep"
)
_BATCHES = flags.DEFINE_list("batches", ["1", "256"], "batch sizes to sweep")
_REPEATS = flags.DEFINE_integer("repeats", 20, "warm calls to average over, at most")
_BUDGET = flags.DEFINE_float(
    "budget", 2.0, "seconds to spend on warm calls before averaging what ran"
)


def _mont_material(key: np.ndarray, message: np.ndarray) -> tuple[Array, Array, Array]:
    """`poly1305`'s scan inputs, built on the host — the ceiling arm's material,
    with the block boundary done in numpy instead of in the trace."""
    r_bytes = key[..., : poly1305.BLOCK_SIZE] & poly1305._CLAMP
    to_int = lambda row: int.from_bytes(bytes(row), "little")
    r = np.array(
        [[to_int(row)] for row in r_bytes.reshape(-1, poly1305.BLOCK_SIZE)],
        dtype=poly1305.WORK,
    ).reshape(*key.shape[:-1], 1)
    blocked = np.asarray(poly1305._blocks(fnp.asarray(message)))
    flat = blocked.reshape(-1, poly1305._PADDED_BLOCK)
    blocks = np.array([[to_int(row)] for row in flat], dtype=poly1305.WORK).reshape(
        *blocked.shape[:-1], 1
    )
    zero = np.zeros((*message.shape[:-1], 1), dtype=poly1305.WORK)
    return fnp.asarray(zero), fnp.asarray(r), fnp.asarray(blocks)


def _mont_mac(accumulator: Array, r: Array, blocks: Array) -> Array:
    """The scan alone, over material that never crossed a byte boundary."""
    (accumulator, _), _ = frx.lax.scan(poly1305._absorb, (accumulator, r), blocks)
    return accumulator


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
    """The shipped path against RFC 8439 §2.5.2 and random messages."""
    key = np.frombuffer(
        bytes.fromhex(
            "85d6be7857556d337f4452fe42d506a8" "0103808afb0db2fd4abff6af4149f51b"
        ),
        dtype=np.uint8,
    )
    text = b"Cryptographic Forum Research Group"
    message = np.frombuffer(text, dtype=np.uint8)
    want = rfc8439_reference.poly1305_mac(bytes(key), text)

    wire = np.asarray(frx.jit(poly1305.mac)(key[None], message[None]))
    assert bytes(wire[0]) == want, "wire arm wrong on the RFC vector"

    rng = np.random.default_rng(0)
    for size in (0, 1, 15, 16, 17, 64, 200):
        k = rng.integers(0, 256, (3, 32), dtype=np.uint8)
        m = rng.integers(0, 256, (3, size), dtype=np.uint8)
        a = np.asarray(frx.jit(poly1305.mac)(k, m))
        for i in range(len(k)):
            ref = rfc8439_reference.poly1305_mac(bytes(k[i]), bytes(m[i]))
            assert bytes(a[i]) == ref, f"wire wrong at size {size} row {i}"
    _say("wire matches RFC 8439 §2.5.2 and random messages (0..200 bytes)")


def _sweep(sizes: list[int], batches: list[int]) -> None:
    rng = np.random.default_rng(1)
    _say(
        "\npoly1305 by message size and batch  (mont = host-entered field "
        "material, the ceiling; wire = the shipped path, boundary in the trace)"
    )
    _say(
        f"  {'B':>5}  {'bytes':>7}  {'mont warm':>10}  {'wire warm':>10}  "
        f"{'wire/mont':>9}  {'wire GB/s':>9}"
    )
    for batch in batches:
        for size in sizes:
            k = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
            m = rng.integers(0, 256, (batch, size), dtype=np.uint8)
            if not size:
                continue
            mont = _time(_mont_mac, *_mont_material(k, m))
            wire = _time(poly1305.mac, k, m)
            if mont is None or wire is None:
                _say(f"  {batch:>5}  {size:>7}  (unavailable)")
                continue
            _say(
                f"  {batch:>5}  {size:>7}  {mont[1] * 1e3:>8.2f}ms  "
                f"{wire[1] * 1e3:>8.2f}ms  {wire[1] / mont[1]:>8.2f}x  "
                f"{batch * size / wire[1] / 1e9:>8.2f}"
            )


def main(argv: list[str]) -> None:
    del argv
    _say(f"backend={frx.default_backend()}  devices={frx.devices()}")
    _verify()
    _sweep([int(s) for s in _SIZES.value], [int(b) for b in _BATCHES.value])


if __name__ == "__main__":
    app.run(main)
