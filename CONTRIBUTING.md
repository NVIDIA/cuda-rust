# Contributing to CUDA Rust

Thank you for your interest in contributing to CUDA Rust! This document
explains the contribution process and requirements.

CUDA Rust is licensed under the [Apache License, Version 2.0](LICENSE).

## Community

Join the project Discord for questions, design discussions, and announcements:
**[discord.gg/ZUEr4AhH5C](https://discord.gg/ZUEr4AhH5C)**

If you are unsure whether something is worth a full issue or PR, the Discord
`#contributors` channel is a good place to ask first.

## Table of Contents

- [Developer Certificate of Origin](#developer-certificate-of-origin)
- [Signing Your Commits](#signing-your-commits)
- [Contribution Process](#contribution-process)
- [Code Requirements](#code-requirements)
- [Continuous Integration](#continuous-integration)
- [AI-Assisted Contributions](#ai-assisted-contributions)
- [IP Review Process](#ip-review-process)

## Developer Certificate of Origin

CUDA Rust requires the Developer Certificate of Origin (DCO) process for all
contributions. The DCO is a lightweight mechanism to certify that you wrote or
otherwise have the right to submit the code you are contributing.

By making a contribution to this project, you agree to the following:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Signing Your Commits

Every commit needs two different things. They are easy to mix up, and one does
not replace the other:

| | DCO sign-off | Cryptographic signature |
| --- | --- | --- |
| What it is | A `Signed-off-by:` line in the commit message | An SSH or GPG signature on the commit, shown as **Verified** on GitHub |
| Command | `git commit -s` | `git commit -S` |
| What it does | Certifies the DCO above | Proves who made the commit |
| If it is missing | The pull request is not merged | The bot holds the pull request for validation instead of starting CI |

`-s` does not sign the commit and `-S` does not add the `Signed-off-by` line, so
use both: `git commit -s -S`.

### DCO sign-off

Every commit in your pull request must include a `Signed-off-by` line.
This certifies that you agree to the DCO above.

To sign off on a commit, use the `-s` flag:

```bash
git commit -s -m "Description of change"
```

This adds a line to your commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```

If you have already made commits without sign-off, you can amend or rebase
to add it:

```bash
# Amend the most recent commit
git commit --amend -s --no-edit

# Rebase and sign all commits in a branch
git rebase --signoff main
```

Your `Signed-off-by` name and email must match your Git configuration
(`user.name` and `user.email`).

### Cryptographic signature

Sign every commit with an SSH or GPG key, so that GitHub shows it as
**Verified**. A pull request with unsigned commits is held for validation (see
[Continuous Integration](#continuous-integration)).

Set up signing once, following GitHub's
[guide to commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification),
and add the public key to your GitHub account as a **Signing key**. Then sign
every commit, or have Git do it by default:

```bash
git commit -S -s -m "Description of change"

# Sign every commit
git config --global commit.gpgsign true
```

Rewriting history (`rebase`, `commit --amend`, `filter-branch`) drops the
signature of every commit it rewrites. Re-sign and sign off together when you
rebase, and force-push the result:

```bash
git rebase --signoff -S main
```

`git log --show-signature` shows whether each commit is signed.

## Contribution Process

1. **Open an issue** describing the bug or feature you want to work on.
2. **Fork the repository** and create a feature branch from `main`.
3. **Implement your changes** following the code requirements below.
4. **Sign off and sign all commits** using `git commit -s -S` (see above).
5. **Submit a pull request** against the `main` branch with a clear
   description of the changes and their motivation.
6. **Respond to review feedback.** All submissions require review before
   merging. Maintainers may request changes or ask questions.

Pull requests that do not meet the requirements below or lack proper DCO
sign-off will not be merged.

### Branch and pull request naming

Prefix branches and pull request titles with the kind of change. Branches use
`/` as the separator and titles use `:`, lowercase: `fix/kernel-launch-sync`
and `fix: resolve kernel launch sync race`.

| Prefix | Use for |
| --- | --- |
| `feat` | New features |
| `fix` | Bug fixes |
| `docs` | Documentation changes |
| `refactor` | Code changes with no behavior change |
| `perf` | Performance improvements with no functional change |
| `test` | Test additions or changes |
| `build` | Build system and dependency changes |
| `ci` | CI changes |
| `chore` | Maintenance tasks |

A scope in parentheses is welcome when a change is confined to one component,
for example `docs(cutile-rs): ...`.

## Code Requirements

### Toolchains

The shared host crates at the repository root use the root stable pin.
Each product has its own workspace and toolchain under `cuda-oxide/` or
`cutile-rs/`. The setup commands below cover cuda-oxide; the Tile checks have
their own section further down.

cuda-oxide pins its stable Rust toolchain (`1.99.1`) and required `rustc-dev`
and `llvm-tools` components in `cuda-oxide/rust-toolchain.toml` (with
`RUSTC_BOOTSTRAP=1` configured via `cuda-oxide/.cargo/config.toml` and
`cargo-oxide`). See the [cuda-oxide README](cuda-oxide/README.md) for setup
instructions.

`cuda-oxide/flake.nix` provides the CUDA 13, Clang/libclang, and pinned Rust
(with LLVM 23 `llvm-tools`) development environment. From the repository root,
run `cd cuda-oxide` and then `nix develop` with flakes enabled. Container setup
is documented in [cuda-oxide/.devcontainer/README.md](cuda-oxide/.devcontainer/README.md).

### Running the cuda-oxide checks

Most of CI is one command. The SIMT tree ships a `Justfile` that mirrors the
workflows:

```bash
just -f cuda-oxide/Justfile check
```

It needs a CUDA toolkit (13.0 or newer, with the cuRAND headers), `cargo-deny`,
and `python3` on `PATH`; it does not need a GPU or a driver: the shared
`cuda-bindings` crate loads `libcuda` at run time, so test binaries load without
one. Individual recipes exist for each piece, and
`just -f cuda-oxide/Justfile --list` shows them with a one-line description
each.

A few CI jobs deliberately stay outside `just check` -- ones that need the
codegen backend, a Python virtualenv, or GitHub's own infrastructure. The
`check` recipe's comment in the `Justfile` names them, and is the place kept in
step when a workflow changes; the commands below are the ones worth knowing by
hand even so.

### Running the cutile-rs checks

cutile-rs builds on the stable toolchain pinned in `cutile-rs/rust-toolchain.toml`
and needs a CUDA 13.2 or newer toolkit with the Tile IR assembler, `tileiras`;
the [cutile-rs book](https://nvidia.github.io/cuda-rust/cutile/) covers setup.
Its test scripts locate the Tile workspace from their own path and reach the
shared host crates at the repository root on their own, so they can be run
from anywhere:

```bash
cutile-rs/scripts/run_cpu_tests.sh   # CPU and assembler tests; needs the toolkit, not a GPU
cutile-rs/scripts/run_gpu_tests.sh   # GPU integration tests
cutile-rs/scripts/run_all.sh         # both of the above, then the examples and benchmarks
```

Run `cutile-rs/scripts/run_all.sh` before marking a Tile pull request ready
for review.

The GPU script also accepts `--archive-dir DIR` to run prebuilt tests without
Cargo or a Rust toolchain. The directory must contain `cargo-nextest`,
`tile-tests.tar.zst`, and `host-tests.tar.zst`, as bundled by CI. This mode runs
tests serially and uses the checkout for source fixtures.
See the archive, bundle, and extract steps in
[the Tile CI workflow](.github/workflows/cutile-rs.yml) for how to produce this
directory. Extract `cutile-gpu-tests.tar` first and use a checkout matching its
`revision.txt`; `cargo-nextest` must be executable. The script's opening comments
describe the directory layout and the contents of each test archive.

CI runs the same suites in `.github/workflows/cutile-rs.yml`; see
[Continuous Integration](#continuous-integration) for how it is triggered on
pull requests.

### Formatting and Style

- Run `(cd cuda-oxide && cargo oxide fmt)` before submitting. All code must be
  formatted with `rustfmt`. Use `cargo oxide fmt` rather than a bare
  `cargo fmt`: the stable host workspace, SIMT workspace, codegen backend,
  examples, and the `cuda-macros` device-only fixture are separate formatting
  scopes. The command mirrors the `fmt` CI job, including the parent host
  workspace in this merged repository and nested example workspaces.
- Run clippy and address any warnings where reasonable. There is no single
  command covering its scopes, and it has more of them than `fmt`: the two
  workspaces below, plus one run per example, plus the nested example
  workspaces that their parent does not list as members, plus the device-only
  fixture. `.github/workflows/clippy.yml` is the full list; the two worth
  running by hand are:

  ```bash
  (cd cuda-oxide && cargo clippy --workspace --all-targets -- -D warnings)
  (cd cuda-oxide/crates/rustc-codegen-cuda && cargo clippy --all-targets -- -D warnings)
  ```
- Follow existing code patterns and conventions in the crate you are
  modifying.

### License Headers

All new first-party source files must carry the NVIDIA copyright notice and an
Apache-2.0 SPDX identifier. Use the block-comment form, which is what the bulk
of the codebase uses:

```rust
/*
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
```

The copyright line must read exactly `Copyright (c) <year(s)> NVIDIA
CORPORATION & AFFILIATES. All rights reserved.` — including the `(c)`, the
`& AFFILIATES`, and both trailing periods. Adapt only the comment syntax for
non-Rust files (`#` for shell, Python, Dockerfiles and TOML; `<!-- -->` for
HTML; `;` for LLVM IR), never the wording.

Preserve existing copyright notices. Add a copyright notice only when you are
the copyright holder or are authorized to name the holder. Vendored and other
third-party files must keep their upstream license and copyright notices, and
must retain their attribution. Embedded cuda-oxide dependencies are recorded
in `cuda-oxide/THIRD_PARTY_NOTICES`.

CI enforces this: `cuda-oxide/scripts/check-spdx-headers.sh` fails on any tracked source
file missing the header (the `cargo-deny / every source file carries the SPDX
header` job). Third-party subtrees and OSRB-reviewed exceptions are listed in
that script.

### Testing

- Compiler pipeline changes should be validated against the existing examples
  in `cuda-oxide/crates/rustc-codegen-cuda/examples/`.
- New GPU intrinsics should include a corresponding example demonstrating
  correct behavior.
- Dialect changes should include appropriate tests in the crate's `tests/`
  directory.
- A new example must print a `SUCCESS`/`PASS`/`Complete` marker once it has
  verified its results, or `cuda-oxide/scripts/smoketest.sh` reports it as
  `FAIL (no success marker)`.
  `cuda-oxide/scripts/check-example-smoketest-contract.sh`
  checks that without a GPU, along with the `*_EXAMPLES` arrays in
  `smoketest.sh`; CI runs it as the `status-guard / smoketest example contract`
  job.

#### Running the CUDA-dependent crates without a GPU

Most crates test on a machine with no GPU and no NVIDIA driver. `cuda-host`
and `cuda-macros` build against the shared `cuda-bindings` crate from
cutile-rs, which needs `cuda.h` and `curand.h` from a CUDA 13.0+ toolkit at
build time but loads `libcuda` at run time through `libloading`. The test
binaries therefore carry no `libcuda.so.1` dependency and load without a
driver; tests that need a real driver are `#[ignore]`d. A driver call made
without `libcuda` present fails with `CUDA_ERROR_NOT_INITIALIZED`, and the
error's `Display` names the library candidates the loader tried.
`.github/workflows/unit-tests.yml` runs the same suites on driverless runners.

### Dependencies

- New dependencies must use permissive licenses (MIT, Apache-2.0, BSD, ISC,
  Zlib, or similar).
- No GPL, AGPL, SSPL, or other copyleft-licensed dependencies.
- If adding a new dependency, update `cuda-oxide/dependency-licenses.csv`
  accordingly. `cuda-oxide/scripts/check-dependency-licenses.sh` reports
  anything the workspace
  declares but that file does not record; CI runs it as the
  `cargo-deny / license-manifest` job. It checks presence, not versions, so a
  routine version bump needs no CSV edit.
- The same applies to an example that pulls third-party code. Each example
  under `cuda-oxide/crates/rustc-codegen-cuda/examples/` is its own workspace, so
  `cargo deny check` does not resolve it; the script reads the example lock
  files directly and asks for a row per third-party crate. Examples that
  depend only on first-party crates by path need nothing.

## Continuous Integration

**All pull request CI goes through [copy-pr-bot](.github/copy-pr-bot.yaml).**
CI runs on NVIDIA self-hosted runners, and those runners reject workflows
triggered by `pull_request`. Instead, copy-pr-bot mirrors the head of your pull
request to a `pull-request/<n>` branch in `NVIDIA/cuda-rust`, and the workflows
run on `push` to that branch (and to `main`). The results show on your pull
request as usual; the runs themselves belong to the `pull-request/<n>` branch.
cutile-rs (Tile) CI already worked this way; it now applies to every workflow.
The bot's own [documentation](https://docs.gha-runners.nvidia.com/apps/copy-pr-bot/)
needs an NVIDIA login.

### When `/ok to test <SHA>` is needed

- **The author is not a member of the NVIDIA GitHub organization.** The bot
  trusts only members of the organization, so a pull request from anyone else is
  not mirrored, and no CI runs, until an NVIDIA employee has reviewed the change
  and comments `/ok to test <full commit SHA>` on it. If you are an external
  contributor, ask a maintainer on the pull request.
- **Only the full SHA works**, and **each new commit needs a new approval.** A
  fix-up push waits for another `/ok to test`.
- **Not needed** for members of the NVIDIA organization: every push to the pull
  request is mirrored on its own, except for drafts (see below).
- **Unsigned commits hold the pull request for validation**, whoever the author
  is. Sign the commits (see
  [Cryptographic signature](#cryptographic-signature)) and push; the bot then
  mirrors the new head without an approval.

### Other things you will notice

- **Draft pull requests** are not mirrored automatically. CI starts when the
  pull request is marked ready for review.
- **Forks.** A pull request from a fork runs in `NVIDIA/cuda-rust` on
  `pull-request/<n>`, not in your fork. A push to your fork's own `main` (for
  example a fork sync) still starts the workflows there. Jobs that need an
  NVIDIA runner label stay queued, because a fork has none; ignore them.
- **Skipped jobs.** Several workflows gate their jobs on a `changes` job that
  compares the diff with `main`, so a job can be skipped when the diff does not
  touch its paths. The jobs run when the diff cannot be detected.

### Adding or changing a workflow

Use the runner labels the existing workflows use (`linux-amd64-cpu4` for what
used to run on `ubuntu-latest`, `linux-amd64-cpu16` for the Tile test jobs) and
trigger on `push` to `main` and `pull-request/[0-9]+`, not on `pull_request`.
The one job that stays on `ubuntu-latest` is the GitHub Pages `deploy` in
`book.yml`, which `actions/deploy-pages` requires.

## AI-Assisted Contributions

- **You are responsible:** Review and understand all AI-generated code before submitting it.
- **Verify correctness:** AI-generated code may contain bugs or security issues. Run the applicable tests and review the diff carefully.
- **No AI-generated sensitive content:** Do not use AI to generate security policies, legal, or license text.

See [AGENTS.md](AGENTS.md) for instructions for coding agents working in this repository.

## IP Review Process

All contributions to CUDA Rust are subject to NVIDIA's IP review process.
Maintainers will ensure that contributions are reviewed in accordance with
NVIDIA's open source policies before merging.

For questions about the contribution process, please open an issue or contact
the maintainers.
