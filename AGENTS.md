# CUDA Rust agent instructions

Repository-wide guidance. When a subdirectory has its own `AGENTS.md`, that file
is the primary guide for the conventions and workflows of that component; this
one covers what applies everywhere.

## Read this first: there is no workspace at the repository root

The root `Cargo.toml` is a workspace with **no members**, on **stable**. Each
component owns its own workspace and its own toolchain, so that a component
needing a particular compiler does not impose it on the rest of the repository.

The consequence is a trap. A bare cargo command at the root succeeds while doing
nothing at all:

```console
$ cargo test                     # from the repository root
error: the workspace has no members
$ echo $?
0
```

Exit code zero. `cargo test`, `cargo build` and `cargo clippy` all report
success at the root having compiled nothing, and `cargo oxide` exits 101 because
the alias resolves `--package cargo-oxide` against a workspace that does not
contain it.

**Run cargo from the component, or go through `just`.**

```console
$ cd cuda-oxide && cargo test -p cuda-macros     # right
$ just test                                      # right, from anywhere
$ cargo test                                     # wrong: silently tests nothing
```

`just` recipes already carry the `cd`, so they work from the repository root.

## Component map

Each component owns its README, its documentation, its crates and its workspace.

- `cuda-oxide/` — the SIMT model. `Cargo.toml` is its workspace, with a
  `crates/*` members glob; `rust-toolchain.toml` beside it pins the nightly it
  needs for `rustc_private`. `crates/` holds the rustc codegen backend, the
  device and host crates and the compiler pipeline; `docs/` holds its book;
  `intrinsics/` holds the generated intrinsic catalog and ABI ledger.
  `crates/rustc-codegen-cuda` is excluded from that glob: it is a third
  workspace, on its own nightly.
- `cutile/` — the Tile model. Arrives with #1. Same shape: its own
  `Cargo.toml`, `crates/`, `docs/`.
- `crates/` — crates shared by more than one component. Nothing is shared yet,
  so the root members glob for it is commented out rather than empty; Cargo
  errors on a glob whose directory does not exist.
- `docs/` — the umbrella landing page and `build_all_docs.sh`, which builds each
  component book and stitches the output into `build/html/<component>/`. It
  skips a component whose book is absent, so it works before `cutile/` lands.

## Running the checks

`just check` is the local mirror of CI. Prefer the recipes over raw `cargo`
invocations — they carry the `cd` above, and several set flags CI depends on.

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

`deny.toml` is at the repository root because the policy is the repository's,
but the lockfile it resolves belongs to a component, so `cargo deny` runs as
`cd cuda-oxide && cargo deny --config ../deny.toml --locked check`.

## Toolchain

The nightly is pinned in `cuda-oxide/rust-toolchain.toml`, and
`cuda-oxide/crates/rustc-codegen-cuda/rust-toolchain.toml` carries a second copy
for that nested workspace. Do not float either to work around a build failure —
`scripts/check-toolchain-parity.sh` exists to catch exactly that drift, and it
checks the `cargo oxide new` scaffold against them too.

rustup resolves a toolchain by walking up from the **working directory**.
`--manifest-path` does not move that lookup, so a cargo command run from the
root gets stable no matter which manifest it is pointed at. This is why the
recipes and scripts change directory rather than pass a manifest path.

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

## CI

Four lanes carry `paths:` filters — `ci.yml`, `examples-compile.yml`,
`docs.yml`, `book.yml` — so a change that cannot affect them does not run them.
`clippy.yml`, `fmt.yml`, `unit-tests.yml` and `cargo-deny.yml` are
`workflow_call` and inherit `ci.yml`'s filter through their caller. If a lane
you expected did not run, check the filter before assuming CI is broken.

`cutile-rs.yml` has no filter, deliberately, and the reason is in a comment
above its trigger: copy-pr-bot mirrors a pull request to `pull-request/<n>` at
an existing SHA, and GitHub often reads that push as having no path changes.
Its jobs are gated on a `detect` job instead, which skips them when `cutile/` is
absent.

## Things that look like bugs and are not

- **13 links point at `NVlabs/cuda-oxide`.** They are issue and pull request
  permalinks. A tracker does not move with the code, so those still resolve
  where they were filed. Everything else was repointed at `NVlabs/cuda-rust`;
  do not "finish the job" on these.
- **10 trybuild `.stderr` files pin `$WORKSPACE`-relative paths.** `$WORKSPACE`
  is the component workspace, so moving a crate rewrites every one of them.
  `TRYBUILD=overwrite cargo test ...` regenerates them; read the diff before
  committing it, because it will happily bless a real regression.

## Paths

Tools here resolve their data relative to the component that owns it, not to the
repository root. `cuda-intrinsics-gen` walks up for the directory holding
`intrinsics/upstream.lock`, and `cargo-oxide` walks up for the one holding
`crates/rustc-codegen-cuda`. If you add a tool that reads repository data,
discover its root the same way rather than assuming the current directory.
