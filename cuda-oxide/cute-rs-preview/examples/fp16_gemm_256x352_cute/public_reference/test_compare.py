# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free checks for public comparison imports and CLI help."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class ComparisonCliTests(unittest.TestCase):
    """Keep ignored local references outside the public comparison import path."""

    def test_help_needs_no_runtime_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).resolve().parent.parent / "compare.py"), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--rust-bin", result.stdout)

    def test_ignored_reference_cannot_replace_public_oracle(self):
        # The actual comparison script must select the public package even if
        # old module names exist both beside it and on PYTHONPATH.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(Path(__file__).resolve().parent.parent / "compare.py", root / "compare.py")
            public = root / "public_reference"
            public.mkdir()
            (public / "__init__.py").write_text("")
            (public / "oracle.py").write_text("raise RuntimeError('public oracle selected')\n")
            (public / "gemm.py").write_text("")
            old = root / "reference"
            old.mkdir()
            for name in ("reference.py", "fp16_gemm_3_256x352.py"):
                (old / name).write_text("raise RuntimeError('ignored private reference selected')\n")
            for name in ("torch.py", "numpy.py"):
                (root / name).write_text("")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(old)
            result = subprocess.run(
                [sys.executable, "-S", str(root / "compare.py"), "--rust-bin", sys.executable],
                env=environment, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("public oracle selected", result.stderr)
            self.assertNotIn("ignored private reference selected", result.stderr)


if __name__ == "__main__":
    unittest.main()
