# Pliron 0.19.0 compatibility check

**Initial result at commit `1f87f3cf8`: cuda-oxide did not build unchanged against
Pliron 0.19.0.**
Dependency resolution works, and 150 tests in five independent libraries pass.
The MIR adapters need API changes before the compiler, CuTe kernels, or Qwen
workload can be validated on this release.

This is the original compatibility report. The follow-up migration landed on
`main` in [PR #1526](https://github.com/NVIDIA/cuda-rust/pull/1526). The
`exp/cuteir-translation` snapshot documented here retains its earlier Git pin.

## Scope and isolation

Checked on 2026-10-16 against cuda-oxide commit `1f87f3cf8`, using the existing
`nightly-2026-10-11` toolchain. The experiment lives in the detached worktree
`/tmp/cuda-rust-pliron019-validation`; the active branch's dependencies and
pending documentation were preserved.

Only the isolated workspace and nested rustc backend manifests/lockfiles were
changed. All five direct Pliron declarations switched from Git revision
`06d99a3df938394583715aacf5eadb65b4791c09` to exact registry version `=0.19.0`.
The `pliron-llvm` configuration remains
`default-features = false, features = ["std"]`.

The published `pliron`, `pliron-derive`, and `pliron-llvm` archives match the
crates.io checksums. Their recorded release commit is
`6ff5b3c8a6d5553b8cbee19d58f3c039f10e5a97`, 25 commits after the tested Git pin.

## Results

| Check | Result |
| --- | --- |
| Dependency resolution in both compiler workspaces | One registry copy each of `pliron`, `pliron-derive`, and `pliron-llvm`, all 0.19.0; no Git copies and no `llvm-sys` dependency. |
| Current-pin exporter baseline | 164 tests passed across `llvm-export` and `pliron-mlir-export`. |
| Current-pin MIR baseline | All 15 `dialect-mir` library tests passed. |
| Initial full workspace test build | Pliron failed because the retained `combine` 4.6.7 lacks `SourcePosition: Hash`. |
| Retry with only `combine` updated to 4.6.8 | Pliron builds; the broad test invocation exposes an existing CTM host-feature test issue. |
| All-target workspace check with continued error collection | New MIR interface/printer errors and an LLVM exporter test type-inference error; the CTM issue also remains. |
| Separate rustc backend test build | Fails on the same four MIR API errors. |
| Independent library tests on 0.19.0 | **150 passed:** `dialect-iket` 8, `dialect-ptx` 43, `llvm-export` 56, `nvvm-transforms` 36, `pliron-mlir-export` 7. |
| GPU numerics, sanitizers, and Qwen output preservation | Not run: the new compiler cannot build. RTX 5090 access was confirmed; hardware availability is not the blocker. |

The independent library count covers `--lib`, not the whole LLVM exporter
integration suite. Passing that subset is not a workspace compatibility claim.

## Required dependency update

Pliron 0.19.0 derives `Hash` for a location containing `combine::SourcePosition`.
`combine` added that implementation in **4.6.8**. However, Pliron declares
`combine = "4"`, which allows our existing lockfiles to retain 4.6.7.

Updating only that dependency in both isolated workspaces cleared the upstream
build error:

```bash
cargo update -p combine --precise 4.6.8
```

Upstream should declare `combine = "4.6.8"` as its minimum. This is a generic
dependency constraint correction; no CUDA-specific code is involved. Evidence:
[Pliron's manifest](https://github.com/pliron-org/pliron/blob/6ff5b3c8a6d5553b8cbee19d58f3c039f10e5a97/Cargo.toml)
and [location implementation](https://github.com/pliron-org/pliron/blob/6ff5b3c8a6d5553b8cbee19d58f3c039f10e5a97/src/location.rs).

## Confirmed local API changes

| Location | Failure | Adaptation to evaluate |
| --- | --- | --- |
| `dialect-mir/src/ops/control_flow.rs`: `MirGotoOp`, `MirCondBranchOp` | `BranchOpInterface` now requires `verify_successor_operand_layout` and `successor_operand_range`. | Implement the new layout/range contract with rejection of malformed branches. |
| Same file: conditional-branch operand segments | `OperandSegmentInterface` now requires `expected_num_segments`. | Supply the three segments explicitly: condition, true arguments, false arguments. |
| `dialect-mir/src/ops/function.rs` | `indented_block!` was removed. | Use scoped indentation through `State::indent()` and verify printer/parser round trips. |
| `llvm-export/tests/export_test/grid_constant.rs:499` | The attribute-map `remove` bound changed from `Borrow` to `Equivalent`, exposing an ambiguous `try_into()` key type. | Give the key an explicit type or use the supported borrowed-key lookup. |

These are the failures reached by this check, not a guarantee that no further
changes are needed. The compiler components behind `dialect-mir` cannot be
fully checked until that crate builds.

## Separate baseline issue

The broad workspace invocation combines host-enabled kernel macros with a
`ctm-rs` test that cannot resolve `cuda_host`. The same error reproduces on the
unchanged current pin with:

```bash
cargo test --offline --locked -p ctm-rs -p cuda-host --all-targets --no-run
```

It is not a Pliron 0.19 regression. Enabling the CuTe/CTM host features in the
all-target check did not remove it. Follow the repository's separated test
recipes and address this combination explicitly when validating a migration.

## What the source comparison preserves

The release retains the LLVM modeling our pin was selected for. Production
LLVM type/layout code, and the operation section covering GEP/no-wrap,
volatile memory access, atomics, and inline assembly, are byte-identical in
the compared regions. LLVM attribute modeling differs only in comments.

Other changes still need downstream testing: `Printable` for `String` now quotes
and escapes, branch verification is stricter, operation wrappers and interface
registration changed, and LLVM constant-folding support expanded. A source
comparison cannot establish unchanged generated GPU code.

## Reproduce and continue

Run from the isolated checkout's `cuda-oxide/`, after resolving the dependencies:

```bash
cargo check --workspace --all-targets --keep-going --locked \
  --features ctm-rs/host,cute-rs/host

cargo test --locked --lib \
  -p pliron-mlir-export -p llvm-export -p dialect-ptx \
  -p dialect-iket -p nvvm-transforms

(cd crates/rustc-codegen-cuda && cargo test --lib --locked)
```

The raw command logs and metadata are under
`/tmp/pliron019-validation-evidence/`; published-source diffs and checksums are
under `/tmp/pliron019-source-audit/`. These temporary paths are local evidence,
not required project inputs.

Before adoption, port the interfaces and rerun host tests, doctests, feature
variants, both compiler workspaces, and CUTLASS integration tests. Then run
native and CUTLASS kernels on SM120a, including Qwen numerical/output checks
and sanitizers. SM100 examples require separate compile coverage; SM120
execution cannot validate SM100 runtime behavior.

Also update the export profile's `PLIRON_REVISION` and its assertion, dependency
license metadata, old Git exceptions, and documentation. Leaving the old
revision string in place would misidentify generated artifacts even if the
code compiled.
