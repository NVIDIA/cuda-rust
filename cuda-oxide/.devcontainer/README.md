# cuda-oxide Dev Container

This dev container provides the toolchain expected by cuda-oxide:

- Ubuntu 24.04
- CUDA Toolkit 13.0
- LLVM 21 with NVPTX support
- Clang 21 resource headers for `bindgen`
- Rust `nightly-2026-08-28` with `rust-src`, `rustc-dev`, `rust-analyzer`,
  `rustfmt` and `clippy`. `llvm-tools` is not preinstalled in the image;
  rustup adds it on your first cargo command under `cuda-oxide/`, because
  `cuda-oxide/rust-toolchain.toml` names it. The image ships LLVM 21 and
  `CUDA_OXIDE_LLC` already points at `llc-21`, so nothing waits on it.

Open the repository's `cuda-oxide/` folder in a devcontainer-aware editor and
choose "Reopen in Container". For example, run `code cuda-oxide` from the
repository root. The editor discovers `cuda-oxide/.devcontainer/devcontainer.json`.
The container requests GPU access with `--gpus=all` and uses
`updateRemoteUserUID` so generated Cargo artifacts and exported PTX files stay
writable from the host checkout.

The whole repository is mounted at `/workspaces/cuda-rust`, including the shared
host crates next to `cuda-oxide/`. The editor and commands start in
`/workspaces/cuda-rust/cuda-oxide`. The checkout directory on the host can have
any name. The Docker build context is only `.devcontainer/`; repository files
are mounted when the container starts, rather than copied into the image.

For CLI usage, run these commands from the repository root:

```bash
npx -y @devcontainers/cli up \
  --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json
npx -y @devcontainers/cli exec \
  --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide doctor
npx -y @devcontainers/cli exec \
  --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide run vecadd
```

Inside an editor terminal in the container, run:

```bash
cargo oxide doctor
cargo oxide run vecadd
```

The CUDA Toolkit is provided by the container image; it does not need to be
installed on the host. The host must provide an NVIDIA GPU, a driver compatible
with CUDA 13.0, and the NVIDIA Container Toolkit so Docker can expose the GPU
to the container.

If the host driver is too old, GPU commands such as `nvidia-smi`,
`cargo oxide doctor`, or `cargo oxide run vecadd` will fail inside the
container. Update the host NVIDIA driver rather than installing a different
CUDA Toolkit in the container.

The image does not set `LD_LIBRARY_PATH`; the NVIDIA Container Toolkit should
provide the host driver libraries at runtime.
