#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the shared cute-rs-preview/.venv before comparison timing.

Only that ignored virtualenv is managed, from cute-rs-preview/requirements.txt.
Successful setup prints its Python path to stdout; progress and package-manager
output go to stderr. The launcher bypasses this helper when the caller selects
an explicit Python environment or a prepared virtualenv already exists.
"""

import fcntl
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys


_PROBE = """
import torch
import numpy
from public_reference import gemm
gemm._version()
if torch.version.cuda is None:
    raise RuntimeError('CUDA-enabled PyTorch is required; the installed wheel is CPU-only')
"""
"""Check imports and the pinned DSL without initializing or requiring a GPU."""


def probe(python, directory):
    """Return import diagnostics from the managed interpreter, without installing."""
    return subprocess.run(
        [str(python), "-c", _PROBE], cwd=directory,
        capture_output=True, text=True, check=False,
    )


def install(command):
    """Run a package-manager command with its progress outside benchmark stdout."""
    subprocess.run(command, check=True, stdout=sys.stderr)


def ensure_environment(directory):
    """Create or repair the shared .venv, retry failed installs, and reuse healthy cached setup.

An OS file lock serializes concurrent setup attempts and is released on process
exit. The requirements stamp is written only after installation and imports
succeed. Existing environments and their other files are never deleted.
"""
    directory = Path(directory).resolve()
    preview = directory.parent.parent
    requirements = preview / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    environment = preview / ".venv"
    python = environment / "bin/python"
    environment.mkdir(exist_ok=True)
    with (environment / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stamp = environment / ".requirements.sha256"
        if python.is_file() and stamp.is_file() and stamp.read_text().strip() == digest:
            if probe(python, directory).returncode == 0:
                return python
        # Never trust an earlier success after a failed repair or changed pin.
        stamp.unlink(missing_ok=True)
        print(f"Setting up the shared cute-rs comparison environment in {environment}", file=sys.stderr)
        uv = shutil.which("uv")
        try:
            if not python.is_file() or not (environment / "pyvenv.cfg").is_file():
                if uv:
                    install([uv, "venv", "--allow-existing", "--python", sys.executable,
                             str(environment)])
                else:
                    install([sys.executable, "-m", "venv", str(environment)])
            if uv:
                install([uv, "pip", "install", "--python", str(python),
                         "-r", str(requirements)])
            else:
                pip = subprocess.run([str(python), "-m", "pip", "--version"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if pip.returncode:
                    install([str(python), "-m", "ensurepip", "--upgrade"])
                install([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                         "-r", str(requirements)])
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Could not set up the comparison environment. Install Python's venv/ensurepip "
                "support (python3-venv on Debian/Ubuntu), or put uv on PATH; check the "
                "package-manager error above and network access, then rerun. "
                "You can also select a prepared environment with --python or "
                "CUDA_OXIDE_CUTE_PYTHON."
            ) from error
        result = probe(python, directory)
        if result.returncode:
            raise RuntimeError(
                f"Comparison dependencies installed, but validation failed:\n{result.stderr}"
                f"Install a compatible CUDA-enabled PyTorch build in {environment}, "
                "or select a prepared environment with --python."
            )
        pending = stamp.with_suffix(".tmp")
        pending.write_text(digest + "\n")
        pending.replace(stamp)
        return python


def main():
    """Print the configured interpreter, or an actionable setup failure."""
    if sys.argv[1:] in (["--help"], ["-h"]):
        print(__doc__)
        return 0
    if sys.argv[1:]:
        print("Usage: python3 setup_reference.py", file=sys.stderr)
        return 2
    try:
        print(ensure_environment(Path(__file__).resolve().parent))
    except (OSError, RuntimeError) as error:
        print(f"FP16 comparison setup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
