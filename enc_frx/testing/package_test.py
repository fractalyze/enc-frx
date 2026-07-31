# Copyright 2026 The enc-frx Authors. SPDX-License-Identifier: Apache-2.0
"""Package-integrity guards for accidents tooling keeps reintroducing."""

from absl.testing import absltest

import enc_frx


class PackageTest(absltest.TestCase):
    def test_version_attr_survives(self) -> None:
        # dev-release stamps enc_frx.__version__ at release time, so an emptied
        # enc_frx/__init__.py breaks the wheel build and nothing else — weeks
        # later. Fail here instead.
        self.assertTrue(getattr(enc_frx, "__version__", ""))


if __name__ == "__main__":
    absltest.main()
