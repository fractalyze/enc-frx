# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""What the arithmetic S-box costs against the table it replaces.

The arithmetic form is chosen for what it removes — a gather indexed by a secret
byte — not for speed, and it is plainly more work: eleven field multiplications
per byte against one lookup. This measures how much more, so the choice is made
against a number rather than an assumption.

The table here is built *from* `sub_bytes`, so both paths compute the same
function and the comparison is only about how.

Run:
    bazel run //enc_frx/aes/testing:sbox_bench -- --blocks=4096
"""

from __future__ import annotations

import time
from collections.abc import Callable

import frx
import frx.numpy as fnp
import numpy as np
from absl import app, flags
from frx import Array

from enc_frx.aes import block

_BLOCKS = flags.DEFINE_integer("blocks", 4096, "blocks per measured call")
_REPEATS = flags.DEFINE_integer("repeats", 20, "warm calls to average over")


def _table() -> Array:
    """The 256-entry S-box, computed by the arithmetic path it is compared to."""
    every = fnp.asarray(np.arange(256, dtype=np.uint8)).astype(block.GF8)
    return fnp.asarray(np.asarray(block.sub_bytes(every)).astype(np.uint8))


def _table_sub_bytes(table: Array) -> Callable[[Array], Array]:
    def apply(state: Array) -> Array:
        return fnp.take(table, state.astype(fnp.int32), axis=0).astype(block.GF8)

    return apply


def _measure(name: str, fn: Callable[[Array], Array], state: Array) -> None:
    compiled = frx.jit(fn)
    start = time.perf_counter()
    frx.block_until_ready(compiled(state))
    cold = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(_REPEATS.value):
        frx.block_until_ready(compiled(state))
    warm = (time.perf_counter() - start) / _REPEATS.value

    bytes_done = state.size
    print(
        f"{name:>12}  compile {cold - warm:7.3f}s  warm {warm * 1e3:8.3f}ms  "
        f"{bytes_done / warm / 1e6:8.1f} MB/s"
    )


def main(argv: list[str]) -> None:
    del argv
    rng = np.random.default_rng(0)
    state = fnp.asarray(
        rng.integers(0, 256, (_BLOCKS.value, 4, 4), dtype=np.uint8)
    ).astype(block.GF8)

    print(f"backend={frx.default_backend()}  blocks={_BLOCKS.value}")
    _measure("arithmetic", block.sub_bytes, state)
    _measure("table", _table_sub_bytes(_table()), state)

    # The cipher as a whole, so the S-box's share is visible rather than implied,
    # and the key schedule on its own — it is recomputed per call today, and
    # whether that matters is a question for the modes built on this.
    keys = fnp.asarray(rng.integers(0, 256, (_BLOCKS.value, 16), dtype=np.uint8))
    blocks = fnp.asarray(rng.integers(0, 256, (_BLOCKS.value, 16), dtype=np.uint8))
    for name, fn, args in (
        ("AES-128", block.encrypt_block, (keys, blocks)),
        ("schedule", lambda k: fnp.stack(block.key_schedule(k)), (keys,)),
    ):
        compiled = frx.jit(fn)
        frx.block_until_ready(compiled(*args))
        start = time.perf_counter()
        for _ in range(_REPEATS.value):
            frx.block_until_ready(compiled(*args))
        warm = (time.perf_counter() - start) / _REPEATS.value
        print(f"{name:>12}  {'':17}  warm {warm * 1e3:8.3f}ms")


if __name__ == "__main__":
    app.run(main)
