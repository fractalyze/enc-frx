# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""GF(2^255 - 19) limb arithmetic — differential against Python's
arbitrary-precision integers, extreme inputs included, per the multi-precision
rule in docs/reference/conventions.md: the layout's margin rests on a stated
accumulator bound, so random vectors alone never approach the worst case.

The extremes carry the cases that matter: the all-0xFFFF element (2^256 - 1,
the largest carried value, which drives every accumulator to its bound), the
values straddling p (canonicalization), and the additive identities.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from enc_frx.x25519 import field

_P = 2**255 - 19

# Values chosen to sit on every boundary the layout has: identities, the
# modulus and its neighbours, the largest canonical value, the top of the
# loose range, and the wrap constant's own neighbourhood (2^256 - 38 == 2p).
_EXTREMES = (
    0,
    1,
    2,
    19,
    38,
    _P - 1,
    _P,
    _P + 1,
    2**255 - 1,
    2**256 - 38,
    2**256 - 1,
)


def _elements(values: list[int]) -> np.ndarray:
    return np.stack(
        [
            np.frombuffer(value.to_bytes(32, "little"), dtype=np.uint8)
            for value in values
        ]
    )


def _values(limbs: object) -> list[int]:
    encoded = np.asarray(field.to_bytes(limbs))
    return [int.from_bytes(bytes(row), "little") for row in encoded]


# Every ordered extreme pair, as one batch — the sweep runs as three device
# calls instead of hundreds of single-row eager dispatches.
_PAIRS = [(a, b) for a in _EXTREMES for b in _EXTREMES]
_LEFT = _elements([a for (a, _) in _PAIRS])
_RIGHT = _elements([b for (_, b) in _PAIRS])


class FieldTest(parameterized.TestCase):
    def test_accumulator_bound_holds(self) -> None:
        # The bound stated in the module docstring, recomputed from the
        # module's own constants so a radix change trips here first: a column
        # sums at most 2*LIMBS half-products below 2^RADIX_BITS, and the wrap
        # scales a column by 38 once.
        max_half = (1 << field.RADIX_BITS) - 1
        max_column = 2 * field.LIMBS * max_half
        self.assertLess(max_column + 38 * max_column, 1 << 32)
        # A limb product must itself fit uint32.
        self.assertLessEqual(2 * field.RADIX_BITS, 32)
        # The limbs must tile 2^256 exactly for the 38-wrap to be the whole
        # reduction.
        self.assertEqual(field.LIMBS * field.RADIX_BITS, 256)

    def test_mul_matches_python_ints_on_extremes(self) -> None:
        got = _values(field.mul(field.from_bytes(_LEFT), field.from_bytes(_RIGHT)))
        for (a, b), row in zip(_PAIRS, got, strict=True):
            self.assertEqual(row, (a * b) % _P, msg=f"a={a:#x} b={b:#x}")

    def test_add_sub_match_python_ints_on_extremes(self) -> None:
        left = field.from_bytes(_LEFT)
        right = field.from_bytes(_RIGHT)
        added = _values(field.add(left, right))
        subtracted = _values(field.sub(left, right))
        for (a, b), plus, minus in zip(_PAIRS, added, subtracted, strict=True):
            self.assertEqual(plus, (a + b) % _P, msg=f"{a:#x}+{b:#x}")
            self.assertEqual(minus, (a - b) % _P, msg=f"{a:#x}-{b:#x}")

    def test_mul_matches_python_ints_on_random_batch(self) -> None:
        rng = np.random.default_rng(0)
        lhs = rng.integers(0, 256, size=(16, 32), dtype=np.uint8)
        rhs = rng.integers(0, 256, size=(16, 32), dtype=np.uint8)
        got = np.asarray(
            field.to_bytes(field.mul(field.from_bytes(lhs), field.from_bytes(rhs)))
        )
        for i in range(lhs.shape[0]):
            a = int.from_bytes(bytes(lhs[i]), "little")
            b = int.from_bytes(bytes(rhs[i]), "little")
            self.assertEqual(int.from_bytes(bytes(got[i]), "little"), (a * b) % _P)

    def test_invert_is_the_inverse(self) -> None:
        nonzero = [v for v in _EXTREMES if v % _P != 0]
        elements = field.from_bytes(_elements(nonzero))
        got = _values(field.mul(elements, field.invert(elements)))
        for value, row in zip(nonzero, got, strict=True):
            self.assertEqual(row, 1, msg=f"{value:#x}")

    def test_to_bytes_is_canonical(self) -> None:
        cases = (
            (_P, 0),
            (_P + 1, 1),
            (2**255 - 1, 18),
            (2**256 - 1, (2**256 - 1) % _P),
        )
        got = _values(field.from_bytes(_elements([v for (v, _) in cases])))
        for (value, residue), row in zip(cases, got, strict=True):
            self.assertEqual(row, residue, msg=f"{value:#x}")


if __name__ == "__main__":
    absltest.main()
