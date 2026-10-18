<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

This is the tracked source of [pliron-mlir-interop](https://github.com/pliron-org/pliron-mlir-interop)
at commit [`295773c4ab9cd7c5c3a14c2082be564b9cf5af90`](https://github.com/pliron-org/pliron-mlir-interop/commit/295773c4ab9cd7c5c3a14c2082be564b9cf5af90),
checked on 2026-10-18. Its package version is 0.1.0 and its dependencies
use published Pliron 0.20.0. Original source, license, readme, tests, and CI
configuration are retained. The cuda-oxide workspace lockfile pins the production
dependency graph; upstream does not track a library Cargo.lock.

The local extension revision is `cuda-oxide.1`. The exporter includes both the
upstream commit and this revision in its manifest fingerprint. The functional
changes are confined to `src/printers.rs`:

- `GenericOp::attr_attr` is public, letting adapters supply typed discardable
  attributes independently of operation properties.
- Block argument printing preserves the argument's Pliron location.
- `set_block_argument_locations` supplies optional per-value locations in the
  print state. This preserves distinct target argument locations, including an
  explicit unknown location, when Pliron's parent-block location is insufficient.

The additional `tests/block_argument_locations.rs` covers the default block
location and distinct argument overrides, and verifies both outputs with
`mlir-opt` when configured. It carries the cuda-oxide contributor notice.
No CUDA operation mappings or LLVM translation changes are included in this patch.
`src/llvm/types.rs` also has one nonfunctional `dict_key!` argument line-break
adjustment required by cuda-oxide's pinned `nightly-2026-10-11` rustfmt.

`UPSTREAM.json` records the Git revision, tree, and patch checksum.
`UPSTREAM_SOURCE_SHA256` hashes the pristine tracked files.
`VENDORED_SOURCE_SHA256` hashes those files after the patch plus its new test.
`cuda-oxide.patch` reproduces the source changes against the pinned commit.
The provenance files themselves are local additions outside that patch.

Validation used `nightly-2026-10-11` and MLIR 22.1.0-rc3. The complete upstream
suite compiles but fails its existing `llvm_function_attributes` canonical-output
snapshot: MLIR 22 drops `allocsize`, `builtin`, and `default_func_attrs` properties
from a bundled LLVM call translation. Upstream CI now targets MLIR 23. The same
failure occurs with the pristine upstream commit; cuda-oxide uses its own explicit
CUDA mapping plans and target adapter rather than these bundled LLVM translators.

From this directory, with `MLIR_OPT` naming an MLIR 22 binary:

```sh
cargo +nightly-2026-10-11 test --workspace
# Known upstream LLVM 23 snapshot mismatch described above.
cargo +nightly-2026-10-11 test --workspace -- --skip llvm_function_attributes
# 28 integration tests and 4 doctests pass; exactly one test is filtered.
cargo +nightly-2026-10-11 clippy --workspace --all-targets -- -D warnings
# Passes.
sha256sum -c VENDORED_SOURCE_SHA256
```

The cuda-oxide adapter has separate tests for real SSA graph construction,
forward branch destinations, properties versus attributes, exact float bits,
custom target types, string escaping, argument and operation locations, and
fallible invalid-plan diagnostics. Consumer and kernel validation belongs to
cuda-oxide's tests, not this upstream printer suite.

To update or remove the vendor, first select an upstream commit or published
release containing equivalent typed-attribute and argument-location support.
Apply the recorded patch to a pristine checkout to reproduce this baseline,
then compare the upstream implementations and run the adapter and consumer
regressions. Replace the dependency, regenerate all affected lockfiles and the
source hashes, and update the exporter fingerprint when the generic changes are
available upstream. Do not remove location overrides merely because current CUDA
mappings commonly use one location for every argument in a block.
