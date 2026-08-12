# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Loader for C2SP's CCTV ML-KEM vectors — `intermediate/` and `unluckysample/`.

Why these sets rather than ACVP's, and why all three parameter sets, is stated
once where they are pinned: [`//MODULE.bazel`](../../../MODULE.bazel). This
module is the parser.

The file format is ` = `-separated, and **a line is not a pair.** The first
segment is the name and every later one is another way of writing the same
quantity, so `A[0, 0] = {2322, 479, …} = 12f91d03…` carries the coefficients
twice — as decimals and as `ByteEncode_12` — and `dkPKE = NTT(s) = 2d17c4ba…`
carries an alias in the middle. Splitting on the *first* ` = ` therefore yields
a value like `{…} = 12f91d…` that is neither hex nor a list.

So a lookup asks for the representation it wants rather than for "the value":
`hex_at` finds the hex segment and `ints_at` the decimal one, and an alias
segment matches neither and is skipped.

Names also repeat across lines — `r` is both the 32-byte encapsulation
randomness and, three lines later, the sampled vector it seeds — so a name maps
to a list of occurrences and a caller indexes the one it means. Collapsing them
would silently keep the last.

TEST ONLY.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

_HEX = re.compile(r"\A[0-9a-f]+\Z")
_LIST = re.compile(r"\A\{(.*)\}\Z")


@dataclass(frozen=True)
class Intermediate:
    """One `intermediate/` file: name -> occurrences -> representations."""

    values: dict[str, list[list[str]]]

    def hex_at(self, name: str, index: int = 0) -> bytes:
        """The `index`-th `name`, from whichever segment is hex."""
        raw = self._segment(name, index, _HEX, "hex")
        return bytes.fromhex(raw)

    def ints_at(self, name: str, index: int = 0) -> list[int]:
        """The `index`-th `name`, from whichever segment is a decimal list."""
        raw = self._segment(name, index, _LIST, "a decimal list")
        return [int(piece) for piece in raw[1:-1].split(",")]

    def _segment(
        self, name: str, index: int, pattern: re.Pattern[str], described: str
    ) -> str:
        if name not in self.values:
            raise KeyError(f"{name} not in vector file; have {sorted(self.values)}")
        occurrences = self.values[name]
        if not 0 <= index < len(occurrences):
            raise IndexError(
                f"{name} appears {len(occurrences)} times, asked for {index}"
            )
        found = [
            segment for segment in occurrences[index] if pattern.fullmatch(segment)
        ]
        if not found:
            raise ValueError(
                f"{name}[{index}] has no {described} segment; "
                f"has {[s[:24] for s in occurrences[index]]}"
            )
        return found[0]


def load_intermediate(path: str) -> Intermediate:
    values: dict[str, list[list[str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            name, separator, rest = stripped.partition(" = ")
            if not separator:
                raise ValueError(f"{path}:{number}: no ` = ` in {stripped[:40]!r}")
            values[name].append(rest.split(" = "))
    if not values:
        raise ValueError(f"{path}: no values")
    return Intermediate(dict(values))


def load_unlucky_seeds(path: str) -> list[bytes]:
    """The `d` seeds from an `unluckysample/` file.

    `d` alone, because that is what the sampler needs: `rho` is the first half of
    `G(d)`, and everything else in the file is downstream of a full key
    generation this cannot yet run.
    """
    seeds = [
        bytes.fromhex(line.split(" = ", 1)[1].strip())
        for line in open(path, encoding="utf-8")
        if line.startswith("d = ")
    ]
    if not seeds:
        raise ValueError(f"{path}: no `d = ` lines")
    return seeds
