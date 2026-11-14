# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and run the issue #1786 lease-scaling repro on a Modal GPU.

Inspect this runner before invoking it. It transfers the required local source
directories to Modal and launches GPU kernels when run.

    modal run cuda-core/repro/lease_scaling/modal_runner.py --launches 256000
    modal run cuda-core/repro/lease_scaling/modal_runner.py --launches 16000 --modes core
"""

from pathlib import Path
import shutil
import subprocess

import modal

REPO_ROOT = Path(__file__).resolve().parents[3]
REPRO_DIR = "/workspace/cuda-core/repro/lease_scaling"
PACKAGE_DIRS = (
    "cuda-core",
    "cuda-async",
    "cuda-bindings",
    "cuda-core-derive",
    "cuda-repr",
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("build-essential", "curl", "libclang-dev", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal"
    )
    .env(
        {
            "PATH": (
                "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:"
                "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            )
        }
    )
)
for package_dir in PACKAGE_DIRS:
    image = image.add_local_dir(
        str(REPO_ROOT / package_dir),
        remote_path=f"/workspace/{package_dir}",
    )
image = image.add_local_file(
    str(REPO_ROOT / "rust-toolchain.toml"),
    remote_path="/workspace/rust-toolchain.toml",
)

app = modal.App("cuda-core-lease-scaling")


@app.function(image=image, gpu="L4", timeout=3600)
def run_repro(launches: int, modes: str) -> None:
    selected_modes = modes.split(",")
    allowed_modes = {"one", "sync", "fresh", "core"}
    if not selected_modes or any(mode not in allowed_modes for mode in selected_modes):
        raise ValueError(f"modes must be comma-separated values from {sorted(allowed_modes)}")

    temporary_repro = Path("/tmp/lease-scaling-repro")
    shutil.copytree(
        f"{REPRO_DIR}/src",
        temporary_repro / "src",
        dirs_exist_ok=True,
    )
    manifest = temporary_repro / "Cargo.toml"
    manifest.write_text(
        "[package]\n"
        'name = "lease-scaling-repro"\n'
        'version = "0.1.0"\n'
        'edition = "2024"\n'
        "publish = false\n\n"
        "[dependencies]\n"
        'cuda-core = { path = "/workspace/cuda-core" }\n'
        'cuda-async = { path = "/workspace/cuda-async" }\n\n'
        "[workspace]\n"
    )
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(manifest)],
        cwd="/workspace",
        check=True,
    )
    executable = temporary_repro / "target/release/lease-scaling-repro"
    for mode in selected_modes:
        subprocess.run(
            [str(executable), mode, str(launches)],
            cwd="/workspace",
            check=True,
        )


@app.local_entrypoint()
def main(launches: int = 256_000, modes: str = "one,sync,fresh,core") -> None:
    run_repro.remote(launches, modes)
