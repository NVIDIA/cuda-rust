# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise setup recovery and offline reuse without downloading GPU packages."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


class SetupTests(unittest.TestCase):
    """Mock only package-manager processes; keep filesystem/cache behavior real."""

    def setUp(self):
        """Give each test an isolated checkout, including spaces in its path."""
        helper = Path(__file__).resolve().parent.parent / "setup_reference.py"
        self.assertTrue(helper.is_file(), "the example needs automatic Python setup")
        spec = importlib.util.spec_from_file_location("setup_reference", helper)
        self.setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.setup)
        self.temporary = tempfile.TemporaryDirectory(prefix="gemm setup ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        public = self.root / "public_reference"
        public.mkdir()
        self.requirements = public / "requirements.txt"
        self.requirements.write_text("nvidia-cutlass-dsl[cu13]==4.7.0\ntorch\nnumpy\n")
        self.python = self.root / ".venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.touch()
        (self.root / ".venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
        self.installs = 0
        self.creations = 0
        self.healthy = False
        self.fail_install = False
        self.pip_missing = False
        self.fail_seed = False
        self.run_patch = mock.patch.object(self.setup.subprocess, "run", side_effect=self.run_process)
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)
        self.uv_patch = mock.patch.object(self.setup.shutil, "which", return_value=None)
        self.uv_patch.start()
        self.addCleanup(self.uv_patch.stop)

    def run_process(self, command, **kwargs):
        """Stand in for pip and dependency imports, which require large downloads."""
        command = [str(part) for part in command]
        if command[1:3] == ["-m", "venv"] or command[1:2] == ["venv"]:
            self.assertEqual(command[-1], str(self.root / ".venv"))
            self.creations += 1
            self.python.parent.mkdir(parents=True, exist_ok=True)
            self.python.touch()
            (self.root / ".venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
            return subprocess.CompletedProcess(command, 0)
        if command[1:3] == ["-m", "ensurepip"]:
            if self.fail_seed:
                raise subprocess.CalledProcessError(1, command)
            self.pip_missing = False
            return subprocess.CompletedProcess(command, 0)
        if command[1:3] == ["pip", "install"]:
            self.assertEqual(command[3:5], ["--python", str(self.python)])
            command = [str(self.python), "-m", "pip", "install", *command[5:]]
        if command[1:3] == ["-m", "pip"]:
            if "install" in command:
                self.installs += 1
                self.assertEqual(command[0], str(self.python))
                self.assertEqual(command[-2:], ["-r", str(self.requirements)])
                if self.fail_install:
                    raise subprocess.CalledProcessError(1, command)
                self.healthy = True
            return subprocess.CompletedProcess(command, 1 if self.pip_missing else 0)
        if command[1:2] == ["-c"]:
            return subprocess.CompletedProcess(command, 0 if self.healthy else 1,
                                               stdout="", stderr="missing dependency")
        self.fail(f"unexpected subprocess: {command}")

    def test_missing_environment_is_created_before_install(self):
        """The default path must provision a fresh checkout, not just repair one."""
        self.python.unlink()
        (self.root / ".venv/pyvenv.cfg").unlink()
        self.assertEqual(self.setup.ensure_environment(self.root), self.python)
        self.assertTrue(self.python.is_file())
        self.assertEqual(self.creations, 1)
        self.assertEqual(self.installs, 1)

    def test_uv_can_create_environment_without_ensurepip(self):
        """The optional uv path must install into the example's isolated interpreter."""
        self.python.unlink()
        (self.root / ".venv/pyvenv.cfg").unlink()
        with mock.patch.object(self.setup.shutil, "which", return_value="/tools/uv"):
            self.assertEqual(self.setup.ensure_environment(self.root), self.python)
        self.assertEqual(self.creations, 1)
        self.assertEqual(self.installs, 1)

    def test_partial_environment_with_missing_pip_can_retry(self):
        """An existing Python left by a failed venv bootstrap must not prevent recovery."""
        self.pip_missing = self.fail_seed = True
        with self.assertRaisesRegex(RuntimeError, "venv/ensurepip"):
            self.setup.ensure_environment(self.root)
        self.fail_seed = False
        self.assertEqual(self.setup.ensure_environment(self.root), self.python)
        self.assertFalse(self.pip_missing)
        self.assertEqual(self.installs, 1)

    def test_first_setup_then_cached_run_needs_no_installer(self):
        """A completed install must be reusable without network or package-manager work."""
        self.assertEqual(self.setup.ensure_environment(self.root), self.python)
        self.assertEqual(self.installs, 1)
        self.fail_install = True
        self.assertEqual(self.setup.ensure_environment(self.root), self.python)
        self.assertEqual(self.installs, 1)

    def test_changed_requirements_are_installed_again(self):
        """An existing environment must not hide a newly required dependency."""
        self.setup.ensure_environment(self.root)
        self.requirements.write_text(self.requirements.read_text() + "new-dependency\n")
        self.setup.ensure_environment(self.root)
        self.assertEqual(self.installs, 2)

    def test_failed_install_is_retried_without_deleting_environment(self):
        """An interrupted install cannot leave a success marker or destroy local files."""
        preserved = self.root / ".venv/preserved"
        preserved.write_text("keep")
        self.fail_install = True
        with self.assertRaises(RuntimeError):
            self.setup.ensure_environment(self.root)
        self.fail_install = False
        self.assertEqual(self.setup.ensure_environment(self.root), self.python)
        self.assertEqual(self.installs, 2)
        self.assertEqual(preserved.read_text(), "keep")

    def test_broken_cached_dependencies_are_rechecked(self):
        """Removing an installed package must invalidate otherwise unchanged setup."""
        self.setup.ensure_environment(self.root)
        self.healthy = False
        self.setup.ensure_environment(self.root)
        self.assertEqual(self.installs, 2)

    def test_failed_cuda_build_check_never_marks_setup_complete(self):
        """A successful pip exit alone is insufficient if imports or CUDA support fail."""
        real_run = self.run_process

        def cpu_torch(command, **kwargs):
            """Simulate an installed CPU-only wheel at the external probe boundary."""
            result = real_run(command, **kwargs)
            if str(command[1]) == "-c":
                return subprocess.CompletedProcess(command, 1, stdout="",
                                                   stderr="CUDA-enabled PyTorch is required")
            return result

        with mock.patch.object(self.setup.subprocess, "run", side_effect=cpu_torch):
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled"):
                self.setup.ensure_environment(self.root)
        self.setup.ensure_environment(self.root)
        self.assertEqual(self.installs, 2)


if __name__ == "__main__":
    unittest.main()
