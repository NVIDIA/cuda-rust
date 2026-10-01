# Contributing to cuda-oxide

Thank you for your interest in contributing to cuda-oxide! This document
explains the contribution process and requirements.

cuda-oxide is licensed under the [Apache License, Version 2.0](LICENSE).

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
- [IP Review Process](#ip-review-process)

## Developer Certificate of Origin

cuda-oxide requires the Developer Certificate of Origin (DCO) process for all
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

## Contribution Process

1. **Open an issue** describing the bug or feature you want to work on.
2. **Fork the repository** and create a feature branch from `main`.
3. **Implement your changes** following the code requirements below.
4. **Sign all commits** using `git commit -s` (see above).
5. **Submit a pull request** against the `main` branch with a clear
   description of the changes and their motivation.
6. **Respond to review feedback.** All submissions require review before
   merging. Maintainers may request changes or ask questions.

Pull requests that do not meet the requirements below or lack proper DCO
sign-off will not be merged.

## Code Requirements

### Toolchain

cuda-oxide requires the Rust nightly toolchain with `rustc_private` support.
See the [README](README.md) for setup instructions.

The repository includes a `flake.nix` that provides a fully reproducible development
environment (CUDA 13, LLVM 22, Clang, pinned Rust nightly). If you have Nix with
flakes enabled, `nix develop` is the quickest way to get everything in place.

### Running the checks

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

### Formatting and Style

- Run `cargo oxide fmt` before submitting. All code must be formatted with
  `rustfmt`. Use `cargo oxide fmt` rather than a bare `cargo fmt`: the codegen
  backend, every example and the `cuda-macros` device-only test fixture are
  each their own workspace, so `cargo fmt` at the repository root reaches none
  of them, while the `fmt` CI job checks all four scopes and will fail on code
  you never had a chance to format. `cargo oxide fmt` mirrors that job, nested
  example workspaces included.
- Run clippy and address any warnings where reasonable. There is no single
  command covering its scopes, and it has more of them than `fmt`: the two
  workspaces below, plus one run per example, plus the nested example
  workspaces that their parent does not list as members, plus the device-only
  fixture. `.github/workflows/clippy.yml` is the full list; the two worth
  running by hand are:

  ```bash
  cargo clippy --workspace --all-targets -- -D warnings
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
must be attributed in `THIRD_PARTY_NOTICES` at the repository root.

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

## IP Review Process

All contributions to cuda-oxide are subject to NVIDIA's IP review process.
Maintainers will ensure that contributions are reviewed in accordance with
NVIDIA's open source policies before merging.

For questions about the contribution process, please open an issue or contact
the maintainers.
