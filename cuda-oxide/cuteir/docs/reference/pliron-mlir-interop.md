# Using upstream Pliron–MLIR interop

The CuTeIR branch now uses upstream interop with locally owned target adapters.
See [the current translation flow](../pliron-mlir-flow.md) and
[dependency provenance](../../../third_party/pliron-mlir-interop/UPSTREAM.md).
The assessment below describes the earlier 2026-10-16 snapshot and probe; its
API limitations and dependency pins are historical.

## Migration validation (2026-10-18)

The local generic `pliron-mlir-export` crate has been removed. Its typed mapping
plans, registry, and validation now live in `cuda-oxide-mlir-export`. The MIR,
CuTe, and NVVM recipes build those plans; a target adapter materializes a fresh
Pliron graph and calls upstream `GenericOp` and `MlirPrinter`. The source module
remains intact. Printing failures propagate as compiler export diagnostics.

The integration merges main at `82f195bea7da11730cc2d04c3d0b4a7802ff0443`
into the CuTeIR branch, whose preserved baseline is
`3bdf00d9930e18f33f2984a4b45b8240388d76cb`. It does not merge CuTeIR into main.

| Component | Tested version |
| --- | --- |
| Pliron, pliron-derive, pliron-llvm | Published `=0.20.0`; source revision `435536c8611af34c48c8f73d78e77894a05d103d` |
| pliron-mlir-interop | Package `0.1.0`, upstream `295773c4ab9cd7c5c3a14c2082be564b9cf5af90`, local extension `cuda-oxide.1` |
| Rust | `nightly-2026-10-11` |
| CUTLASS compiler | `4.7.0`, `x86_64-cu13`; library SHA-256 `57df017e3585a10443c74c8b4cd99bda854242fb2f4c9534cf56d58c2c741628` |
| Standalone MLIR parser | `22.1.0-rc3` |
| GPU and tools | RTX 5090, `sm_120a`; driver `615.71.09`; CUDA `13.4.92`; Compute Sanitizer `2026.3` |

Both the SIMT workspace and separate backend resolve one registry copy of each
Pliron 0.20.0 crate, no old generic exporter, and no `llvm-sys`. The upstream
snapshot and its small patch are temporarily vendored, with reconstruction
instructions and source checksums in the dependency provenance linked above.

### Results

| Validation | Result |
| --- | --- |
| MIR/NVVM dialects, importer, lowerings, transforms, LLVM export | 2,086 passed; 10 ignored |
| CuTe dialect | 99 passed |
| Mapping and adapter suite, including doctests | 161 passed; 1 optional parser test separately passed |
| cuda-oxide-codegen, including doctest | 1,317 passed; 13 ignored, of which 2 direct-ABI compiler tests separately passed |
| CTM/CuTe facade tests | 136 passed |
| Separate rustc backend library | 40 passed |
| Native backend examples | `vecadd` and `array_for_loop` passed |
| Upstream printer suite | 28 integration tests and 4 doctests passed with one known MLIR-version snapshot mismatch filtered |
| Compiler linting | SIMT and backend `clippy --all-targets -- -D warnings` passed |

`elementwise_cute` checked all 1,003 f32 and 2,005 f16 outputs, including ragged
tails. `blockscale_gemm_cute` checked all 4,194,304 FP16 output bit patterns.
Both passed Compute Sanitizer memcheck with zero errors and zero leaked bytes.
Both produced cubins, SASS, and resource reports byte-for-byte identical to the
preserved baseline:

| Kernel | Cubin bytes | SHA-256, baseline and migration |
| --- | ---: | --- |
| Elementwise | 11,776 | `90edfe6e20107c8de161f286a620542126134d67c3c54c2383b4eb076b225c35` |
| Blockscale GEMM | 29,656 | `3d8fcee8e5c72e6aa3b8518b4a41b305bc3eb8b85cfd91042f9b5a84f3050932` |

GEMM CUDA-event medians were 13.568 microseconds before and 13.376 after,
using 100 warmups and 31 samples each. These are single runs of identical device
code, not evidence of a speedup. Complete emitted modules also verified with
CUTLASS's native parser and matched after generic printing without debug
locations and collision-checked Rust symbol-disambiguator normalization.
This validates two representative SM120 kernels, not every workload or SM100
hardware execution.

Independent adapter probes verified forward branches, nested-region captures,
expansion values, unusual identifiers, locations, escaping, and invalid-plan
errors. They found a hyphenated attribute-key quoting bug, now fixed with a
regression test and verified through MLIR 22. Current kernel keys contained no
hyphens, so that fix leaves the tested kernel output unchanged.

### Upstream versus cuda-oxide changes

The reusable upstream patch exposes typed discardable attributes through
`GenericOp::attr_attr` and preserves block-argument locations, with optional
per-argument location overrides. It contains no CUDA mapping rules. A separate
whitespace adjustment makes the vendored source pass the pinned Rust formatter.

The target adapter, mapping plans, value connections, profile fingerprints,
unsupported-input checks, and CUTLASS integration belong to cuda-oxide. No
bundled upstream Pliron LLVM translator is used by this CuTeIR path.

The full upstream suite has one existing failure against MLIR 22:
`llvm_function_attributes` expects LLVM 23 canonical attributes. It also fails
on the pristine upstream commit; upstream CI targets MLIR 23. The previous
LLVM-translator findings should not be treated as unverified defects in this
new revision or blockers for the tested CuTeIR path.

Workspace validation also exposed an existing CTM test dependency problem:
unified macro features emitted host metadata without a `cuda-host` test
dependency. The same failure reproduced on the baseline. The fixture now has
that development dependency, and both facade test lanes request CUDA headers.
The license inventory now records the existing Qwen `safetensors` dependency.

### Reproduce

Run from `cuda-oxide/`, with the repository toolchain and CUDA toolkit installed:

```sh
cargo metadata --locked --format-version 1 > /tmp/interop-simt-metadata.json
(cd crates/rustc-codegen-cuda && cargo metadata --locked --format-version 1) > /tmp/interop-backend-metadata.json
cargo test --locked -p dialect-mir -p dialect-nvvm -p mir-importer -p mir-lower -p mir-transforms -p nvvm-transforms -p llvm-export
cargo test --locked -p dialect-cute -p cuda-oxide-mlir-export -p cuda-oxide-codegen
cargo test --locked -p ctm-rs -p cute-rs --all-targets
(cd crates/rustc-codegen-cuda && cargo test --locked --lib)
cargo clippy --locked --workspace --all-targets -- -D warnings
(cd crates/rustc-codegen-cuda && cargo clippy --locked --all-targets -- -D warnings)

export CUDA_OXIDE_CUTLASS_COMPILER="${CARGO_HOME:-$HOME/.cargo}/cuda-oxide/toolchains/cutlass/4.7.0/x86_64-cu13/lib/libCutlassCompiler.so"
sha256sum "$CUDA_OXIDE_CUTLASS_COMPILER"
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_MLIR_OUTPUT=/tmp/elementwise.mlir cargo oxide run elementwise_cute --arch sm_120a
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_MLIR_OUTPUT=/tmp/blockscale.mlir cargo oxide run blockscale_gemm_cute --arch sm_120a
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 cuteir/examples/elementwise_cute/target/release/elementwise_cute
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 cuteir/examples/blockscale_gemm_cute/target/release/blockscale_gemm_cute
CUDA_OXIDE_TEST_CUTLASS_COMPILER="$CUDA_OXIDE_CUTLASS_COMPILER" cargo test --locked -p cuda-oxide-codegen direct_launch_abi_compiles -- --ignored
CUDA_OXIDE_DEVICE_BACKEND=native cargo oxide run vecadd --arch sm_120
CUDA_OXIDE_DEVICE_BACKEND=native cargo oxide run array_for_loop --arch sm_120
```

Use an existing CUTLASS DSL 4.7.0 Python environment for the optional parser test:

```sh
CUDA_OXIDE_CUTLASS_PYTHON=/path/to/cutlass47/bin/python cargo test --locked -p cuda-oxide-mlir-export --test sync_restrict selected_cutlass_native_parser_verifies_both_proxy_fences -- --ignored --exact
```

The native rustc backend tests may need the pinned toolchain's `lib/` directory
in `LIBRARY_PATH` and `LD_LIBRARY_PATH`. Upstream tests, the explicit snapshot
filter, patch reconstruction, and checksum commands are recorded in
[UPSTREAM.md](../../../third_party/pliron-mlir-interop/UPSTREAM.md).

## Historical assessment (2026-10-16)

Upstream can provide an extensible printing foundation without CUDA-specific
changes. Replacing our current exporter still requires downstream integration;
this is not a dependency-only swap. Source review has not established any
mandatory upstream PR.

Reviewed on 2026-10-16 at upstream revision
[`665cc56d44fb0cfd689fc013cca5b35c4d1e0b12`](https://github.com/pliron-org/pliron-mlir-interop/tree/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12).
This assessment includes a small extension probe, not a completed CUTLASS
compatibility test.
Start with [the current translation flow](../pliron-mlir-flow.md).

## What already works

Upstream provides `ToMlirOp`, `ToMlirType`, and `ToMlirAttr` interfaces.
Dialect authors implement them for their own operations, types, and attributes.
`MlirPrinter` calls those implementations and produces textual MLIR.
Builtin and LLVM translations are included.
[Source: translation interfaces](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/lib.rs).

An operation implementation has full control over its output. It can emit more
than one target operation. The shared Pliron printing state can hold local
translation data, including profile choices and bookkeeping for new values.
Upstream already uses that state for recursive type printing.
[Source: recursive types](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/llvm/types.rs).

The `GenericOp` helper covers common cases: changing the operation name,
selecting results, adding properties and attributes, and printing nested
regions. More involved lowering can use a local target IR or a custom printer.
[Source: generic operation printer](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/printers.rs).

## What stays in cuda-oxide

- Rust value layouts and signed/unsigned arithmetic decisions.
- MIR, CuTe, and NVVM mapping rules, including expansions into several ops.
- CuTe safety checks, source verification, and rejection of unsupported input.
- CUTLASS version profiles, GPU capabilities, launch ABI, and cubin validation.
- Tests that preserve arithmetic, memory behavior, and generated code.

These are compiler policies supplied through extensions. They do not belong
in upstream's generic printer.

## Integration details we must resolve

| Detail | What the source shows | Downstream approach |
| --- | --- | --- |
| Where implementations live | Rust cannot implement an upstream trait on a type owned by another crate from our separate exporter crate. | Put optional implementations in the owning dialect crates, or translate into locally owned target/adapter types. |
| Integer types | Upstream preserves signedness as `si32`, `ui32`, or `i32`. Our standard MLIR arithmetic uses signless types. | Choose signed/unsigned operations first, then lower the target types. Do not globally change upstream's integer policy. |
| Expanded operations | `GenericOp` uses source operands, types, and names; it is not our target syntax tree. | Preserve typed value/block bookkeeping in a downstream lowering layer. Avoid scattering hand-built strings through mapping recipes. |
| Source locations | The generic upstream printer does not currently emit operation or block-argument locations. | Preserve them through an adapter or contribute a generic location-printing facility. |
| Extra attributes | Dedicated builtin/LLVM printers select known attributes; for example, the module printer only selects `sym_name`. | Define which attributes survive and test preservation explicitly. |
| Dependency identity | Upstream tracks Pliron Git HEAD; its current source does not compile against our `06d99a3` pin. | Resolve one compatible Pliron source/revision across the dependency graph and validate the local compiler against it. |

The integer and module behavior is visible in upstream's
[builtin types](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/builtin/types.rs)
and [module printer](https://github.com/pliron-org/pliron-mlir-interop/blob/665cc56d44fb0cfd689fc013cca5b35c4d1e0b12/src/builtin/ops.rs).
Compare our [builtin mappings](../../../crates/cuda-oxide-mlir-export/src/packs.rs)
and [translation session](../../../crates/cuda-oxide-mlir-export/src/translation/translate.rs).

A scratch build against our Pliron pin failed on changed upstream APIs,
including `BytesAttr` and attribute accessors. This is a dependency compatibility
issue to resolve locally; it does not establish a missing interop extension.

The separate [Pliron 0.19.0 workspace check](pliron-0.19-validation.md) records
the API adaptations needed by the tested experimental snapshot. The migration
subsequently landed on `main` in [PR #1526](https://github.com/NVIDIA/cuda-rust/pull/1526).
That migration does not establish interop compatibility with 0.19.0.

## What the extension probe established

With compatible Pliron revision `d6886bd2ff997dff11b13330c257cfb99308f35f`,
unchanged upstream interop source compiled and ran a scratch example containing
a custom type, custom operations, properties, and an operation expanding into
two target operations. A later operation consumed the expansion's final value.
Only the scratch dependency paths changed; our workspace and pins did not.

LLVM 22 `mlir-opt --allow-unregistered-dialect` accepted the output. This checks
generic MLIR structure and value connections, not the semantics of an
unregistered dialect or compatibility with CUTLASS. It demonstrates that custom
extensions and operation expansion do not inherently need an upstream PR.

## Which upstream PRs would make sense?

Only reusable facilities demonstrated by a small test case:

1. **Location printing:** preserve source locations for any dialect.
2. **Attribute preservation:** define how unconsumed discardable attributes
   survive dedicated builtin/LLVM printers.
3. **Optional external translation hooks:** let a separate client supply a
   translation policy for foreign types or operations, if downstream adapters
   otherwise duplicate the generic machinery.

The third item needs a design discussion and a concrete example. Existing
interfaces may be sufficient after local lowering. Additional builder methods
are conveniences to justify individually, not assumed migration blockers.

Keep CUDA names, CuTe semantics, architecture checks, and CUTLASS requirements
out of these PRs. Tests for generic facilities can use a tiny example dialect.

## Recommended next step

First align dependencies on a compatible Pliron revision and validate that
revision against our compiler. Then adapt one small mixed MIR/NVVM/CuTe kernel
using the unchanged upstream crate.
Check signed arithmetic, one operation expansion, locations, attributes, and
unknown-input rejection, then compile its output with the pinned CUTLASS
consumer. Compare it with the existing exporter before expanding the migration.

That experiment should decide which generic improvements deserve upstream PRs.
Do not delete our current exporter or relax its checks merely because the new
text parses.
