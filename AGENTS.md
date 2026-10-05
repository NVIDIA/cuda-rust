# CUDA Rust agent instructions

Repository-wide guidance. When a subdirectory has its own `AGENTS.md`, that file
is the primary guide for the conventions and workflows of that component; this
one covers what applies everywhere.

## Choose the workspace before running Cargo

The root `Cargo.toml` is the stable host workspace. Its members are
`cuda-bindings`, `cuda-core`, `cuda-core-derive`, and `cuda-async`.
`cuda-oxide/` and `cutile-rs/` are separate nested workspaces, excluded from the
root workspace. Running `cargo test` at the root tests the host workspace;
it does not test either kernel programming model.

Run commands from the workspace that owns the package. For example, from the
repository root:

```bash
cargo check -p cuda-core
(cd cuda-oxide && cargo test -p cuda-macros)
(cd cutile-rs && cargo test -p cutile-ir)
just -f cuda-oxide/Justfile check
```

These commands have different prerequisites; see each component's README and
contribution guide. There is no root `Justfile`. The SIMT recipes live in
`cuda-oxide/Justfile` and run with `cuda-oxide/` as their working directory.

## Component map

- `cuda-bindings/`, `cuda-core/`, `cuda-core-derive/`, and `cuda-async/` — shared
  host crates in the root workspace. Both kernel stacks depend on these by path.
- `cuda-oxide/` — the SIMT model. Its `Cargo.toml` explicitly lists workspace
  members under `crates/`; its `rust-toolchain.toml` pins the nightly needed for
  `rustc_private`. The book is `cuda-oxide/cuda-oxide-book/`, and
  `cuda-oxide/intrinsics/` holds the intrinsic catalog and ABI ledger.
  `cuda-oxide/crates/rustc-codegen-cuda/` declares its own workspace and inherits
  the SIMT toolchain pin. Examples and test fixtures can also declare separate
  workspaces; a parent `--workspace` command does not include them.
- `cutile-rs/` — the Tile model, with its own workspace and stable toolchain
  pin. Its member crates are directly under `cutile-rs/`, and its book is
  `cutile-rs/cutile-book/`. `cuda-tile-rs` is outside `default-members` because
  its build downloads and compiles LLVM; `--workspace` includes it unless
  explicitly excluded.
- `docs/` — the umbrella landing page and `docs/build_all_docs.sh`, which builds
  the combined documentation site. The component books retain their own sources.

## Running the checks

Use `just -f cuda-oxide/Justfile --list` from the repository root to list the
SIMT recipes. The commands below are run from `cuda-oxide/`:

| Recipe | What it covers |
| --- | --- |
| `just check` | SIMT formatting, clippy, tests, guards, intrinsics, and rustdoc checks |
| `just fmt` / `just fmt-check` | Formatting scopes selected by `cargo oxide fmt` |
| `just clippy` | SIMT and backend lints, with warnings as errors |
| `just test` | Unit-test packages that need no CUDA toolkit, plus backend library tests |
| `just test-cuda` | Toolkit-dependent packages; no GPU or driver required |
| `just check-guards` | Repository guards and dependency/license checks |
| `just check-intrinsics` | Catalog currency, PTX routes, and the ABI ledger |
| `just doc-check` | SIMT rustdoc/doctests and backend rustdoc |
| `just book` | Builds the cuda-oxide book with warnings as errors |

`just check` needs a CUDA toolkit, `cargo-deny`, and Python. It does not cover
all CI: example compilation, additional example lint scopes, book builds, and
CodeQL have separate jobs. Read the recipe and relevant workflow when choosing
validation for a change.

For Tile and shared-host changes, follow `cutile-rs/CONTRIBUTING.md` and
`.github/workflows/cutile-rs.yml`. The CPU and assembler suite in
`cutile-rs/scripts/run_cpu_tests.sh` runs with `cutile-rs/` as its working
directory and includes the CPU-safe shared-host tests. CPU-only does not mean
toolkit-free: some suites require CUDA headers and the Tile IR assembler. GPU tests have a separate
`cutile-rs/scripts/run_gpu_tests.sh` entry point. A bare root `cargo test` can
include tests that require a GPU.

Dependency checks must cover each affected workspace's dependency graph and
lockfile. `just check-guards` runs `cargo deny` for the host workspace, SIMT
workspace, backend, and device-only fixture; use its actual invocations and
`.github/workflows/cargo-deny.yml` rather than assuming one root check covers
nested workspaces.

## Toolchains

The root `rust-toolchain.toml` and `cutile-rs/rust-toolchain.toml` carry matching
stable pins. `cuda-oxide/rust-toolchain.toml` supplies the SIMT nightly, including
for its nested backend workspace. Do not float a pin to work around a failure.
`cuda-oxide/scripts/check-toolchain-parity.sh` checks the SIMT pin against its
scaffold, development container, and documentation.

rustup normally resolves the toolchain from the working directory; an explicit
override can change that. `--manifest-path` does not change the working directory,
so pointing a root Cargo command at the SIMT manifest does not select the SIMT
nightly. Change directory first or select the required toolchain explicitly.

Nix environments live in `cuda-oxide/flake.nix` and `cutile-rs/flake.nix`.
Run `nix develop` from the relevant component.

## Commits and pull requests

- **Sign off every commit** (`git commit -s`). The DCO is required; see
  `CONTRIBUTING.md`.
- **New first-party source files need an SPDX header** — the NVIDIA copyright
  notice plus `Apache-2.0`, in the form shown under "License Headers" in
  `CONTRIBUTING.md`. Preserve existing and third-party notices.
  `cuda-oxide/scripts/check-spdx-headers.sh` enforces the policy.
- **Never push to the canonical upstream repository, `NVIDIA/cuda-rust`.
  Treat it as read-only.** Branch and push to a fork. Before pushing, run
  `git remote -v` and confirm the push remote is a fork of the base, comparing
  full `OWNER/REPOSITORY` names — not remote names like `origin` or `upstream`,
  and not the owner alone.

## CI

Workflows such as `ci.yml`, `examples-compile.yml`, `docs.yml`, and `book.yml`
carry path filters. The SIMT lanes include shared-host paths as well as
`cuda-oxide/`; book filters follow documentation sources and build scripts.
The reusable `clippy.yml`, `fmt.yml`, `unit-tests.yml`, and `cargo-deny.yml`
workflows inherit the caller's trigger. Check current filters and job conditions
before interpreting a missing run as either a failure or successful validation.

`cutile-rs.yml` deliberately has no workflow-level path filter. Copy-pr-bot
mirrors trusted pull requests to `pull-request/<n>` at an existing SHA, which
GitHub can treat as having no path changes. Its `changes` job gates Tile and
shared-host jobs; an undetectable diff runs them conservatively. SIMT-only
changes can skip them when the diff is available.

## Historical references and fixtures

- Issue and pull-request permalinks may retain historical repository names such
  as `NVlabs/cuda-oxide`. Preserve a historical reference unless its destination
  is demonstrably wrong; use `NVIDIA/cuda-rust` for current repository links.
- Trybuild `.stderr` files can contain `$WORKSPACE`-relative paths. Moving a
  crate can change those diagnostics. `TRYBUILD=overwrite cargo test ...`
  regenerates snapshots; read the diff before accepting it, because it can also
  bless a real regression.

## Paths

Tools resolve repository data relative to the component that owns it.
`cuda-intrinsics-gen` discovers the directory holding `intrinsics/upstream.lock`,
and `cargo-oxide` discovers the one holding `crates/rustc-codegen-cuda`.
Follow the owning tool's root-discovery convention instead of assuming every
command starts at the repository root.
