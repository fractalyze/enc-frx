# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What a batch of decapsulations costs, and where inside one the time goes.

The reason ML-KEM is written here rather than linked from a library is that a
batch of decapsulations is `B` independent runs of one program, which is the
shape a device is wide for. That is a claim about throughput, and this is what
holds it to a number.

`decaps` is the direction worth measuring closely: it is the hot path, and it
re-encrypts what it decrypted. That re-encryption costs less than it sounds —
measured 1.04-1.53× `encaps`, falling toward 1.05× as `B` grows, because both
directions are dominated by the same matrix expansion — so `encaps` is not a
cheaper target to leave alone. It is the same target.

## What it reports, and what it will not

- **Compile and warm runtime, in separate columns.** Every distinct batch shape
  is its own compilation, so a warm figure alone describes a process that has
  already paid for the shape it is about to run. A single conflated number hides
  which of the two a caller would actually pay.
- **A dispatch floor**, measured as one trivial jitted call on the same device in
  the same process, **after a real program has run on it**. A small-`B` figure at
  or near the floor is a measurement of the call and not of ML-KEM, and the ratio
  between two such rows says nothing. The warm-up is load-bearing rather than
  tidy: unwarmed, the same call reports eight times the number on GPU and the
  floor stops being one.
- **A per-stage breakdown**, so the bar chart says where the time is before
  anything is optimized. Each stage is compiled and timed on its own, which means
  **the stages do not sum to the whole**: each pays a dispatch it does not pay
  inside `decaps`, and none of the fusion across their boundaries survives. The
  breakdown ranks stages; it does not decompose the end-to-end number.
- **A single-shot reference** from OpenSSL's ML-KEM — one core, `B = 1`, on the
  box the batched numbers were taken on. Without it a throughput figure has
  nothing to be honest against.

## Reading a row

`per op` is the warm time divided by `B`, which is the number that has to fall
as `B` rises for the batching claim to hold. Where it stops falling is the
saturating point, and it is the answer this exists to produce.

Run:
    bazel run //enc_frx/ml_kem/testing:kem_bench
    bazel run //enc_frx/ml_kem/testing:kem_bench -- --batches=1,32 --sets=768
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl import app, flags
from frx import Array

from enc_frx.ml_kem import _k_pke, encoding, hashes, ntt, sampling
from enc_frx.ml_kem.ml_kem import MlKem
from enc_frx.ml_kem.params import PARAMETER_SETS, SEED_SIZE, MlKemParams, N

_BATCHES = flags.DEFINE_list(
    "batches", ["1", "32", "256", "1024", "8192"], "batch sizes to sweep"
)
_SETS = flags.DEFINE_list(
    "sets", ["512", "768", "1024"], "parameter sets, by the suffix of their name"
)
_REPEATS = flags.DEFINE_integer("repeats", 20, "warm calls to average over, at most")
_BUDGET = flags.DEFINE_float(
    "budget", 2.0, "seconds to spend on warm calls before averaging what ran"
)
_STAGE_BATCH = flags.DEFINE_integer(
    "stage_batch", 256, "batch size for the per-stage breakdown; 0 skips it"
)
_REFERENCE_SECONDS = flags.DEFINE_integer(
    "reference_seconds", 1, "seconds per OpenSSL reference measurement; 0 skips it"
)

# Seconds of real work before the dispatch floor is timed — see `_dispatch_floor`
# for why an unwarmed one is not a floor, and why one call does not do it.
_FLOOR_WARMUP_S = 0.5


@dataclasses.dataclass(frozen=True)
class _Timing:
    """One call site: what compiling it cost, and what running it warm costs.

    `compile_s` is the cold call minus the warm one, so it is the compilation
    alone rather than the compilation plus a run.
    """

    compile_s: float
    warm_s: float

    def per_op(self, batch: int) -> float:
        return self.warm_s / batch


def _say(text: str) -> None:
    """One line of the report, flushed as it is measured.

    A sweep is long and its rows are the point: a run killed part way through
    has still reported everything it finished, which a buffered stream loses.
    """
    print(text, flush=True)


def _time(fn: Callable[..., Any], *args: Any) -> _Timing | None:
    """Compile `fn`, then run it warm until the budget or the repeat cap.

    `None` when the device refuses the shape — a batch large enough to exhaust
    memory is a result about the sweep, not a reason to lose the rows already
    measured.
    """
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
    return _Timing(max(cold - warm, 0.0), warm)


def _dispatch_floor() -> _Timing | None:
    """One trivial jitted call: what any measurement here cannot go below.

    **The warm-up is the measurement.** Timed as the first thing in a process,
    this call reports 0.185 ms on an RTX 5090; the identical call after a real
    program has run reports 0.022 ms. Every stage row sits between those two, so
    an unwarmed floor is not a floor — it is a threshold that rejects rows which
    are genuinely above the real one, and the rows it rejects are exactly the
    small-`B` ones it exists to judge.

    **Sustained** work, and both words are load-bearing. A single call of the
    program below leaves the floor at 0.180 ms — measured, not assumed — and so
    do repeated trivial calls and an elementwise pass over the same shape. What
    brings it to 0.022 ms is running a real program in a loop, which is why this
    warms for a wall-clock budget rather than a call count.

    The cause is not settled and nothing here claims one: the device's clocks
    read 2.5-2.9 GHz while the high number is being taken, so an idle GPU does
    not cover it. That uncertainty is the second reason to warm rather than to
    subtract a constant — a constant would have to be right about the mechanism.

    CPU shows none of this (0.005 ms either way), which is why the original
    placement went unnoticed.
    """
    warm_up = frx.jit(ntt.ntt)
    argument = ntt.as_field(fnp.asarray(np.zeros((256, 3, N), dtype=np.int32)))
    start = time.perf_counter()
    while time.perf_counter() - start < _FLOOR_WARMUP_S:
        frx.block_until_ready(warm_up(argument))
    return _time(lambda x: x + np.int32(1), fnp.asarray(np.zeros(1, dtype=np.int32)))


def _material(
    scheme: MlKem, batch: int, rng: np.random.Generator
) -> tuple[Array, Array, Array, Array]:
    """A batch of keys, messages and the ciphertexts they encapsulate to.

    Built once per cell and outside every measurement: what is timed is
    `encaps` and `decaps` over material that already exists, which is the
    position a caller holding a key is in.
    """
    seeds = fnp.asarray(rng.integers(0, 256, (batch, scheme.seed_size), dtype=np.uint8))
    ek, dk = frx.block_until_ready(frx.jit(scheme.keygen)(seeds))
    m = fnp.asarray(rng.integers(0, 256, (batch, SEED_SIZE), dtype=np.uint8))
    c, _ = frx.block_until_ready(frx.jit(scheme.encaps_internal)(ek, m))
    return ek, dk, m, c


def _sweep(
    params: MlKemParams, batches: Sequence[int], rng: np.random.Generator
) -> None:
    """The end-to-end table for one parameter set: `encaps` and `decaps` by `B`."""
    scheme = MlKem(params)
    _say(f"\n{params.name}  (k={params.k}, ciphertext {params.ciphertext_size} B)")
    _say(
        f"  {'B':>6}  {'encaps compile':>14}  {'encaps warm':>12}  "
        f"{'decaps compile':>14}  {'decaps warm':>12}  {'decaps/op':>11}  "
        f"{'decaps/s':>10}  {'ratio':>6}"
    )
    for batch in batches:
        material = _material(scheme, batch, rng)
        ek, dk, m, c = material
        encaps = _time(lambda key, msg: scheme.encaps(key, randomness=msg), ek, m)
        decaps = _time(scheme.decaps, dk, c)
        if encaps is None or decaps is None:
            _say(f"  {batch:>6}  {'(unavailable at this batch size)':>60}")
            continue
        per_op = decaps.per_op(batch)
        _say(
            f"  {batch:>6}  {encaps.compile_s:>13.2f}s  "
            f"{encaps.warm_s * 1e3:>10.2f}ms  {decaps.compile_s:>13.2f}s  "
            f"{decaps.warm_s * 1e3:>10.2f}ms  {per_op * 1e6:>9.1f}us  "
            f"{1 / per_op:>10.0f}  {decaps.warm_s / encaps.warm_s:>5.2f}x"
        )


def _stages(
    params: MlKemParams, batch: int, rng: np.random.Generator
) -> list[tuple[str, Callable[..., Any], tuple[Any, ...]]]:
    """The stages of one decapsulation, each with inputs it would really see.

    Every input is derived here, outside the timing, from a real key and a real
    ciphertext — a stage fed synthetic bytes of the right shape would measure
    the same arithmetic over data the scheme never produces, and the samplers
    are the place that would mislead.
    """
    scheme = MlKem(params)
    k, eta1, eta2, du, dv = params.k, params.eta1, params.eta2, params.du, params.dv
    ek, dk, m, c = _material(scheme, batch, rng)

    dk_pke, ek_stored, _, z = encoding.decode_dk(dk, k)
    t_hat_ints, rho = encoding.decode_ek(ek, k)
    t_hat = ntt.as_field(t_hat_ints)
    h_ek = hashes.h(ek)
    _, r = hashes.g(fnp.concatenate([m, h_ek], axis=-1))

    a_hat = sampling.expand_matrix(rho, k)
    y = sampling.sample_poly_cbd(
        hashes.prf(eta1, r[..., None, :], np.arange(k, dtype=np.uint8)), eta1
    )
    y_hat = ntt.ntt(y)
    s_hat = ntt.as_field(encoding.decode_dk_pke(dk_pke, k))
    u, v = encoding.decode_ciphertext(c, k, du, dv)

    def draw(seed: Array) -> tuple[Array, Array]:
        """`y` at eta1 and `e_1 ‖ e_2` at eta2, as `encrypt` draws them."""
        return (
            sampling.sample_poly_cbd(
                hashes.prf(eta1, seed[..., None, :], np.arange(k, dtype=np.uint8)), eta1
            ),
            sampling.sample_poly_cbd(
                hashes.prf(
                    eta2, seed[..., None, :], np.arange(k, 2 * k + 1, dtype=np.uint8)
                ),
                eta2,
            ),
        )

    return [
        ("H(ek), SHA3-256", hashes.h, (ek,)),
        ("G(m‖H(ek)), SHA3-512", hashes.g, (fnp.concatenate([m, h_ek], axis=-1),)),
        ("J(z‖c), SHAKE256", hashes.j, (fnp.concatenate([z, c], axis=-1),)),
        (
            "sample Â, k² SampleNTT",
            lambda seed: sampling.expand_matrix(seed, k),
            (rho,),
        ),
        ("sample y/e, PRF+CBD", draw, (r,)),
        ("NTT, [B,k,256]", ntt.ntt, (y,)),
        ("iNTT, [B,k,256]", ntt.intt, (y_hat,)),
        ("base_mul, Â∘ŷ (k×k)", _k_pke._matrix_vector, (a_hat, y_hat)),
        ("base_mul, t̂∘ŷ (k)", _k_pke._inner_product, (t_hat, y_hat)),
        (
            "decode dk‖c",
            lambda key, ct: (
                encoding.decode_dk(key, k),
                encoding.decode_ciphertext(ct, k, du, dv),
            ),
            (dk, c),
        ),
        (
            "encode c, compress+pack",
            lambda uu, vv: encoding.encode_ciphertext(
                ntt.as_ints(uu), ntt.as_ints(vv), du, dv
            ),
            (ntt.as_field(u), ntt.as_field(v)),
        ),
        (
            "K-PKE decrypt",
            lambda key, ct: _k_pke.decrypt(key, ct, k=k, du=du, dv=dv),
            (dk_pke, c),
        ),
        (
            "K-PKE encrypt, the re-encryption",
            lambda key, msg, seed: _k_pke.encrypt(
                key, msg, seed, k=k, eta1=eta1, eta2=eta2, du=du, dv=dv
            ),
            (ek_stored, m, r),
        ),
        ("decaps, the whole", scheme.decaps, (dk, c)),
    ]


def _breakdown(params: MlKemParams, batch: int, rng: np.random.Generator) -> None:
    """Time every stage of one decapsulation, against the whole call.

    Measured first and printed after, because the share column is against the
    whole `decaps` and that is the last stage timed — a stage list that put it
    first would compile the largest program before any of its parts.
    """
    measured = [
        (name, _time(fn, *args)) for name, fn, args in _stages(params, batch, rng)
    ]
    whole = next(
        (t.warm_s for name, t in measured if name.startswith("decaps") and t), None
    )

    _say(
        f"\n{params.name} per stage at B={batch}"
        "  (each stage compiled alone, so these do not sum to the whole)"
    )
    _say(f"  {'stage':<34}{'compile':>10}{'warm':>12}{'per op':>11}{'of decaps':>11}")
    for name, timing in measured:
        if timing is None:
            _say(f"  {name:<34}{'(unavailable)':>44}")
            continue
        share = f"{timing.warm_s / whole * 100:>10.1f}%" if whole else f"{'':>11}"
        _say(
            f"  {name:<34}{timing.compile_s:>9.2f}s{timing.warm_s * 1e3:>10.2f}ms"
            f"{timing.per_op(batch) * 1e6:>9.1f}us{share}"
        )


def _openssl_reference(names: Sequence[str], seconds: int) -> None:
    """OpenSSL's ML-KEM on this box: one core, one operation at a time.

    The point of comparison the batched rows need. It is a different question —
    latency of a single operation against throughput of a batch — and printing
    it beside them is what keeps a large `ops/s` from being read as a claim
    about the single-shot case, where a mature C implementation is far ahead.
    """
    binary = shutil.which("openssl")
    if binary is None:
        _say("\nno openssl on PATH; reference skipped")
        return
    version = subprocess.run(
        [binary, "version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    _say(f"\n{version}, single-shot reference (one core, B=1)")
    _say(f"  {'':<14}{'keygen/s':>12}{'encaps/s':>12}{'decaps/s':>12}")

    for name in names:
        # One algorithm per call, because `-mr`'s summary row names none of them:
        # it is `+F9:<index>:<keygen/s>:<encaps/s>:<decaps/s>`, and the index is
        # openssl's own table position rather than the order asked for. Asking
        # one at a time is what makes the row's name the one requested.
        command = ["speed", "-mr", "-seconds", str(seconds), "-kem-algorithms", name]
        result = subprocess.run(
            [binary, *command], capture_output=True, text=True, check=False
        )
        summary = next(
            (line for line in result.stdout.splitlines() if line.startswith("+F9:")),
            None,
        )
        if result.returncode != 0 or summary is None:
            _say(f"  {name:<14}(`openssl {' '.join(command)}` gave nothing)")
            continue
        keygen, encaps, decaps = (float(field) for field in summary.split(":")[2:5])
        _say(f"  {name:<14}{keygen:>12.0f}{encaps:>12.0f}{decaps:>12.0f}")


def main(argv: list[str]) -> None:
    del argv
    rng = np.random.default_rng(0)
    batches = [int(b) for b in _BATCHES.value]
    chosen = [p for p in PARAMETER_SETS if p.name.rsplit("-", 1)[1] in _SETS.value]

    _say(f"backend={frx.default_backend()}  devices={frx.devices()}")
    floor = _dispatch_floor()
    if floor is not None:
        _say(
            f"dispatch floor: one trivial jitted call is "
            f"{floor.warm_s * 1e3:.3f}ms warm — nothing below this is ML-KEM"
        )

    for params in chosen:
        _sweep(params, batches, rng)
    if _STAGE_BATCH.value:
        for params in chosen:
            _breakdown(params, _STAGE_BATCH.value, rng)
    if _REFERENCE_SECONDS.value:
        _openssl_reference([p.name for p in chosen], _REFERENCE_SECONDS.value)


if __name__ == "__main__":
    app.run(main)
