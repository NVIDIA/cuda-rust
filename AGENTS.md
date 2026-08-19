# CUDA Rust agent instructions

Repository-wide guidance. When a subdirectory has its own `AGENTS.md`, that file
is the primary guide for the conventions and workflows of that component; this
one covers what applies everywhere.

## Component map

Each component owns its README, its documentation and its crates.

- `cuda-oxide/` — the SIMT model. `crates/` holds the rustc codegen backend, the
  device and host crates and the compiler pipeline; `docs/` holds its book;
  `intrinsics/` holds the generated intrinsic catalog and ABI ledger.
- `cutile/` — the Tile model. Arrives with #1; not present yet.
- `crates/` — crates shared by more than one component. Empty today, so the
  workspace `members` glob for it is commented out in `Cargo.toml`; Cargo errors
  on a glob whose directory does not exist.
- `docs/` — the umbrella landing page and `build_all_docs.sh`, which builds each
  component book and stitches the output into `build/html/<component>/`. It
  skips a component whose book is absent, so it works before `cutile/` lands.

Component crates are picked up by the `cuda-oxide/crates/*` glob in the root
`Cargo.toml`. `rustc-codegen-cuda` is deliberately excluded: it is its own
workspace root because it needs a different nightly.

## Running the checks

`just check` is the local mirror of CI. Prefer the recipes over raw `cargo`
invocations — several of them set flags CI depends on.

| Recipe | What it covers |
| --- | --- |
| `just check` | Everything below, in one go |
| `just fmt` / `just fmt-check` | rustfmt |
| `just clippy` | Lints, warnings-as-errors |
| `just test` / `just test-cuda` | Unit tests; the second needs a GPU |
| `just check-guards` | The `scripts/check-*.sh` guards plus `cargo deny` |
| `just check-intrinsics` | Catalog currency, PTX routes, the ABI ledger |
| `just doc-check` | rustdoc and doctests |
| `just book` | Builds the cuda-oxide book |

Run `just --list` for the rest.

## Toolchain

The nightly is pinned in `rust-toolchain.toml` and `cuda-oxide` needs
`rustc_private`. Do not float the pin to work around a build failure —
`scripts/check-toolchain-parity.sh` exists to catch exactly that drift.
`nix develop` gives a reproducible environment via `flake.nix`.

## Commits and pull requests

- **Sign off every commit** (`git commit -s`). The DCO is required; see
  `CONTRIBUTING.md`.
- **New first-party source files need an SPDX header** — the NVIDIA copyright
  notice plus `Apache-2.0`, in the block-comment form shown under "License
  Headers" in `CONTRIBUTING.md`. `scripts/check-spdx-headers.sh` enforces it.
- **Never push to the canonical upstream repository. Treat it as read-only.**
  Branch and push to a fork. Before pushing, run `git remote -v` and confirm the
  push remote is a fork of the base, comparing full `OWNER/REPOSITORY` names —
  not remote names like `origin` or `upstream`, and not the owner alone.

## Paths

Tools in this repository resolve their data relative to the component that owns
it, not to the repository root. `cuda-intrinsics-gen` walks up for the directory
holding `intrinsics/upstream.lock`, and `cargo-oxide` walks up for the one
holding `crates/rustc-codegen-cuda`. If you add a tool that reads repository
data, discover its root the same way rather than assuming the current directory.
