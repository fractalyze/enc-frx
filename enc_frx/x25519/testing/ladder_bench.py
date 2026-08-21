# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What the uint32 limb field costs against the registered curve25519 dtype.

`field.py` spells GF(2^255 - 19) out in 16-bit limbs because no registered
field existed when it was written. The pinned stack now registers
`zk_dtypes.curve25519_bf`, whose arithmetic lowers to frxlib's native prime
field kernels — so the layout is a choice again, and this holds the two to a
number before anyone rewrites anything. Both arms run the identical RFC 7748
ladder — same 255 iterations, same operation schedule, same inversion chain —
so the comparison is only about how a field element is carried and multiplied.

Three arms, because the dtype comes in two storage forms:

- **limb** — `x25519.x25519` as shipped, bytes in, bytes out.
- **dtype bf** — the ladder over `curve25519_bf`, whose storage is the
  canonical little-endian encoding, so RFC bytes enter and leave the trace as
  a free `view`. The conditional swap is `where` — bitwise masks are type
  errors on field dtypes, correctly.
- **dtype mont** — the same ladder over `curve25519_bf_mont`. Montgomery
  storage is where the fast multiply lives, but the canonical<->Montgomery
  convert does not lower today (traced `astype` between the two variants
  check-fails in compilation), so elements enter and leave this arm as host
  material. Its rows therefore price the ladder alone, without the byte
  boundary the other two arms include — read it as the kernel ceiling, not as
  a deployable path.

Every arm is verified against the RFC 7748 §5.2 vectors and against `field`'s
own output before its first timed row; a fast arm that computes the wrong
function has no row to show.

## What it measured, and the decision it held

First run (frx 0.10.2.dev20260820235505, zk_dtypes 0.0.16, RTX 5090): the
mont arm is uniformly 12-25x faster than the limb field on GPU (B >= 32) and
17-72x on CPU — a single multiply over 2^16 elements is 292 Mmul/s against
the limb field's 1.7 on CPU, the scalarization gap in one number. The bf arm
wins on GPU at every batch size past 1 (2.5x at B=256-1024) but *loses* on
CPU at mid-batch (2.8x slower at B=1024, flat ~2.4ms/op where the limb field
amortizes), so the deployable arm does not clear "faster on both legs" and
the limb field stays. The swap reopens when the variant convert lowers —
that unlocks the mont arm's margin with the bf arm's free byte boundary.

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
import zk_dtypes
from absl import app, flags
from frx import Array

from enc_frx.x25519 import field, x25519
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

_BF = np.dtype(zk_dtypes.curve25519_bf)
_MONT = np.dtype(zk_dtypes.curve25519_bf_mont)
_P = 2**255 - 19

# Seconds of real ladder work before the dispatch floor is timed — an unwarmed
# floor reports ~8x the warm number on GPU (measured by ml_kem's kem_bench,
# whose floor this reuses the shape of).
_FLOOR_WARMUP_S = 0.5


def _dtype_ladder_step(index: Array, carry: tuple[Array, ...]) -> tuple[Array, ...]:
    """One RFC 7748 §5 iteration over a field dtype — `x25519._ladder_step`
    with the limb calls replaced by operators, and the XOR cswap by `where`.
    Module-level for the loop-body lowering cache, as there."""
    x2, z2, x3, z3, swap, bits, x1, a24 = carry
    bit = frx.lax.dynamic_slice_in_dim(bits, 254 - index, 1, axis=-1)
    swap = swap ^ bit
    cond = swap.astype(bool)
    x2, x3 = fnp.where(cond, x3, x2), fnp.where(cond, x2, x3)
    z2, z3 = fnp.where(cond, z3, z2), fnp.where(cond, z2, z3)
    swap = bit

    a = x2 + z2
    aa = a * a
    b = x2 - z2
    bb = b * b
    e = aa - bb
    c = x3 + z3
    d = x3 - z3
    da = d * a
    cb = c * b
    t = da + cb
    x3 = t * t
    s = da - cb
    z3 = x1 * (s * s)
    x2 = aa * bb
    z2 = e * (aa + a24 * e)
    return (x2, z2, x3, z3, swap, bits, x1, a24)


def _square_step(_: Array, acc: Array) -> Array:
    return acc * acc


def _dtype_invert(element: Array) -> Array:
    """`field.invert`'s 254-squaring chain for 2^255 - 21, over the dtype."""

    def pow2k(value: Array, squarings: int) -> Array:
        return frx.lax.fori_loop(0, squarings, _square_step, value)

    z2 = element * element
    z9 = pow2k(z2, 2) * element
    z11 = z9 * z2
    z_5_0 = (z11 * z11) * z9
    z_10_0 = pow2k(z_5_0, 5) * z_5_0
    z_20_0 = pow2k(z_10_0, 10) * z_10_0
    z_40_0 = pow2k(z_20_0, 20) * z_20_0
    z_50_0 = pow2k(z_40_0, 10) * z_10_0
    z_100_0 = pow2k(z_50_0, 50) * z_50_0
    z_200_0 = pow2k(z_100_0, 100) * z_100_0
    z_250_0 = pow2k(z_200_0, 50) * z_50_0
    return pow2k(z_250_0, 5) * z11


def _dtype_ladder(bits: Array, x1: Array, a24: Array, one: Array, zero: Array) -> Array:
    """The ladder body every dtype arm shares: bit array and field-typed
    inputs in, the unencoded `x2/z2` field element out."""
    x2, z2 = one, zero
    x3, z3 = x1, one
    swap = fnp.zeros((*x1.shape[:-1], 1), dtype=fnp.uint32)
    x2, z2, x3, z3, swap, _, _, _ = frx.lax.fori_loop(
        0, 255, _dtype_ladder_step, (x2, z2, x3, z3, swap, bits, x1, a24)
    )
    cond = swap.astype(bool)
    x2 = fnp.where(cond, x3, x2)
    z2 = fnp.where(cond, z3, z2)
    return x2 * _dtype_invert(z2)


def _constant(value: int, dtype: np.dtype, batch: tuple[int, ...]) -> Array:
    return fnp.broadcast_to(fnp.asarray(np.array([[value]], dtype=dtype)), (*batch, 1))


def _bf_x25519(scalar: Array, u: Array) -> Array:
    """X25519 over `curve25519_bf`: canonical storage, so the RFC bytes enter
    and leave as views and the whole computation is one trace."""
    batch = scalar.shape[:-1]
    bits = x25519._scalar_bits(x25519._clamp(scalar))
    x1 = fnp.concatenate([u[..., :31], u[..., 31:] & np.uint8(127)], axis=-1).view(_BF)
    result = _dtype_ladder(
        bits,
        x1,
        _constant(121665, _BF, batch),
        _constant(1, _BF, batch),
        _constant(0, _BF, batch),
    )
    return result.view(fnp.uint8)


def _mont_material(
    scalar: np.ndarray, u: np.ndarray
) -> tuple[Array, Array, Array, Array, Array]:
    """Host-side entry into Montgomery storage: clamp and mask as the RFC
    says, then mint the field elements from integers. This is the boundary
    the mont arm keeps off the device (module docstring), so everything here
    stays outside the timed call."""
    batch = scalar.shape[:-1]
    bits = x25519._scalar_bits(x25519._clamp(fnp.asarray(scalar)))
    masked = u.copy()
    masked[..., 31] &= 0x7F
    x1 = np.array(
        [[int.from_bytes(bytes(row), "little")] for row in masked.reshape(-1, 32)],
        dtype=_MONT,
    ).reshape(*batch, 1)
    return (
        bits,
        fnp.asarray(x1),
        _constant(121665, _MONT, batch),
        _constant(1, _MONT, batch),
        _constant(0, _MONT, batch),
    )


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
    """Every arm against RFC 7748 §5.2 and the limb field's own output."""
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

    limb = np.asarray(frx.jit(x25519.x25519)(k, u))
    bf = np.asarray(frx.jit(_bf_x25519)(fnp.asarray(k), fnp.asarray(u)))
    mont = np.asarray(frx.jit(_dtype_ladder)(*_mont_material(k, u)))
    for i, want in enumerate(expect):
        assert bytes(limb[i]) == want, f"limb wrong at row {i}"
        assert bytes(bf[i]) == want, f"dtype bf wrong at row {i}"
        got = int(mont[i, 0]).to_bytes(32, "little")
        assert got == want, f"dtype mont wrong at row {i}"
    _say(f"all arms match RFC 7748 §5.2 + {len(k) - 2} random vectors")


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
        "\nx25519, three arms by batch size  "
        "(mont rows price the ladder alone — no byte boundary)"
    )
    _say(
        f"  {'B':>6}  {'limb warm':>10}  {'bf warm':>10}  {'mont warm':>10}  "
        f"{'bf/limb':>8}  {'mont/limb':>9}  {'limb/op':>9}  {'bf/op':>9}"
    )
    last: dict[str, tuple[float, float]] = {}
    for batch in batches:
        k = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
        u = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
        limb = _time(x25519.x25519, k, u)
        bf = _time(_bf_x25519, fnp.asarray(k), fnp.asarray(u))
        mont = _time(_dtype_ladder, *_mont_material(k, u))
        if limb is None or bf is None or mont is None:
            _say(f"  {batch:>6}  (unavailable at this batch size)")
            continue
        last = {"limb": limb, "bf": bf, "mont": mont}
        _say(
            f"  {batch:>6}  {limb[1] * 1e3:>8.2f}ms  {bf[1] * 1e3:>8.2f}ms  "
            f"{mont[1] * 1e3:>8.2f}ms  {bf[1] / limb[1]:>7.2f}x  "
            f"{mont[1] / limb[1]:>8.2f}x  {limb[1] / batch * 1e6:>7.1f}us  "
            f"{bf[1] / batch * 1e6:>7.1f}us"
        )
    if last:
        _say(
            "  compile, at the largest measured B: "
            + ", ".join(f"{name} {t[0]:.2f}s" for name, t in last.items())
        )


def _single_multiply(count: int) -> None:
    """One field multiply over `count` elements, per layout — the ratio the
    ladder rows are made of, without the ladder around it."""
    rng = np.random.default_rng(3)
    values = [
        int.from_bytes(bytes(row), "little") % _P
        for row in rng.integers(0, 256, (2 * count, 32), dtype=np.uint8)
    ]
    limb_all = field.from_bytes(
        fnp.asarray(
            np.frombuffer(
                b"".join(v.to_bytes(32, "little") for v in values), dtype=np.uint8
            ).reshape(2 * count, 32)
        )
    )
    rows = [
        ("limb", field.mul, limb_all[:count], limb_all[count:]),
        (
            "bf",
            lambda x, y: x * y,
            fnp.asarray(np.array(values[:count], dtype=_BF)),
            fnp.asarray(np.array(values[count:], dtype=_BF)),
        ),
        (
            "mont",
            lambda x, y: x * y,
            fnp.asarray(np.array(values[:count], dtype=_MONT)),
            fnp.asarray(np.array(values[count:], dtype=_MONT)),
        ),
    ]
    _say(f"\none multiply over {count} elements")
    for name, fn, left, right in rows:
        timing = _time(fn, left, right)
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
