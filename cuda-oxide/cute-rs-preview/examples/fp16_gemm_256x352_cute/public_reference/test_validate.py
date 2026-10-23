# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free CLI tests for early rejection of unsupported hardware."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXAMPLE = Path(__file__).resolve().parent.parent
"""Example directory containing the validation package."""


class ValidationCliTests(unittest.TestCase):
    """Check that CLI and hardware errors cannot accidentally start compilation."""

    def invoke(self, *args: str, torch_stub: str | None = None):
        """Run without site packages, optionally supplying the CUDA query boundary."""
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join((directory, str(EXAMPLE)))
            if torch_stub is not None:
                Path(directory, "torch.py").write_text(torch_stub)
            return subprocess.run(
                [sys.executable, "-S", "-m", "public_reference.validate", *args],
                env=environment,
                cwd=directory,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_help_works_without_cuda_packages(self):
        """A top-level torch or CuTe import would break dependency-free help."""
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--case", result.stdout)

    def test_missing_torch_is_reported_without_a_traceback(self):
        """Missing runtime dependencies must produce a readable failed report."""
        result = self.invoke("--case", "full")
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "failed")
        self.assertIn("torch", report["error"])
        self.assertNotIn("Traceback", result.stderr)

    def test_unavailable_cuda_does_not_query_or_compile_a_kernel(self):
        """Querying an absent device or importing CuTe would replace this error."""
        result = self.invoke(
            "--case", "full",
            torch_stub=(
                "class cuda:\n"
                "    @staticmethod\n"
                "    def is_available(): return False\n"
            ),
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertIn("CUDA is unavailable", report["error"])
        self.assertEqual(report["cases"], [])

    def test_sm120_is_rejected_before_loading_kernel_or_oracle(self):
        """SM120 must fail at the architecture gate even without CuTe installed."""
        result = self.invoke(
            "--case", "full",
            torch_stub=(
                "class cuda:\n"
                "    @staticmethod\n"
                "    def is_available(): return True\n"
                "    @staticmethod\n"
                "    def current_device(): return 0\n"
                "    @staticmethod\n"
                "    def get_device_capability(device): return (12, 0)\n"
                "    @staticmethod\n"
                "    def get_device_name(device): return 'test SM120'\n"
            ),
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertIn("SM100", report["error"])
        self.assertIn("12.0", report["error"])
        self.assertEqual(report["cases"], [])


if __name__ == "__main__":
    unittest.main()
