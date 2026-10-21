# Initial SM100 attention validation

This report preserves the initial implementation evidence. Decode has since
reached performance parity; see the [updated profiling and validation report](../../sm100_mla_decode_fp16_cute/reference/measurement-notes.md)
for the optimized executable, nine-case correctness matrix, sanitizer checks
and repeated measurements in both timing orders.

The two examples target B200 (`sm_100a`) and implement the matched profiles
listed in their READMEs. Their Rust kernel symbols end in `_cute_rs`; runnable
Python adapters are labeled `_cutedsl`. Each example includes unchanged public
CUTLASS v4.7.0 sources and helpers, pinned to commit
`dcf215af68a2d08d305076c152a06f201728cd53` with checked SHA-256 digests.

## Environment and reproducibility

Validation uses NVIDIA B200, compute capability 10.0, 148 SMs, driver
615.71.09, the repository nightly toolchain, the official CUTLASS 4.7.0
compiler, `nvidia-cutlass-dsl==4.7.0`, and Torch 2.13.0+cu130. Both attention
implementations compile at optimization level 2. The default compiler level
for existing examples remains 3.

Build and run commands are in the
[prefill README](../README.md) and
[decode README](../../sm100_mla_decode_fp16_cute/README.md).
Their result files retain exact configurations, source/executable hashes,
errors, GPU state and individual timing windows. GPU validation and timings
are serialized across the examples.

## Correctness and synchronization

Prefill covers one-key attention, dense and top-left causal masks, unequal
query/key lengths, GQA, residual query/key tiles, KV-stage ring reuse and
persistent CTA reuse. It checks identical FP16 inputs against an independent
FP32 oracle and the CuTeDSL result. MLA checks output and base-2 LSE against an
independent masked FP64 oracle and CuTeDSL, including permuted pages, key tails,
empty trailing splits, unsplit/split execution and persistent cluster reuse.

The initial B200 matrix passed all eight prefill cases and six decode cases.
Prefill's largest absolute error against the FP32 oracle is `5.8413e-4`;
its largest difference from CuTeDSL is `6.1036e-5`. The oracle explicitly
uses highest-precision FP32 matmul with TF32 disabled. Decode output and LSE
are bitwise identical to CuTeDSL in every matrix case, including output scale
1.73. Its largest FP64-oracle output error is `5.5055e-4` and LSE error is
`2.9167e-6` across the six-case matrix.

See the [prefill case reports](results/correctness/)
and [decode matrix](../../sm100_mla_decode_fp16_cute/reference/results/validation-b200.json).
Both examples pass memcheck and synccheck with zero reported errors, and their
stream traces confirm complete timed invocations. Retained reports were checked
against the SHA-256 of each final executable.

Each comparison refuses timing after nonfinite results, failed tolerances,
invalid reference hashes or incompatible compiler optimization settings.
Compute Sanitizer memcheck and synccheck exercise representative tail/ring
cases. Nsight Systems traces check that the complete invocation is inside the
event interval on the actual launch stream; decode includes its reduction.

## Compiler and API checks

The shared checks cover typed operand/layout errors, canonical swizzles,
TMA destination mutability, semantic import/export, launch bounds, TMEM
transfer widths and scalar intrinsics. Unsupported TMEM-source MMA K sizes
are rejected at construction. Dynamic phases, lane participation, cross-helper
CTA grouping and aliased TMEM lifetimes remain explicit unsafe caller
obligations.

Verified suites include `cute-rs`, `dialect-cute`, `mir-importer`,
`cuda-oxide-mlir-export`, `cuda-oxide-codegen` and the separate backend library, plus all 105 `cuda-device` library tests.
From `cuda-oxide/`, reproduce shared checks with:

```sh
cargo test -p cute-rs -p dialect-cute -p mir-importer \
  -p cuda-oxide-mlir-export -p cuda-oxide-codegen
CUDA_OXIDE_TEST_CUTLASS_COMPILER=/absolute/path/to/libCutlassCompiler.so \
  cargo test -p cuda-oxide-codegen cutlass_attention_ -- --ignored
```

Three explicit compiler probes exercise generic SS/TS MMA, TMEM transfers,
attention layouts/TMA, and numeric/synchronization operations through the
pinned official compiler. New examples also have host/contract unit tests. Deferred TMEM transfers
retain explicit wait/fence obligations; native packed FP32 add/multiply/FMA
are checked through emitted PTX, including round-to-nearest and preserved
subnormals.

Formatting, CuTe API/dialect/example lints, public API rustdoc with warnings
denied, SPDX attribution and affected-workspace dependency checks pass. Dependency audits cover both standalone example lockfiles and
the affected SIMT/backend graphs using `cuda-oxide/deny.toml` explicitly.
A blanket exporter all-target clippy run has existing failures in
`mir_checked.rs` (`manual_is_multiple_of`), `tests/address_contract.rs`
(`unnecessary_unwrap`), and `tests/checked_storage.rs`
(`unnecessary_mut_passed`); these are outside this implementation.
The broader crate-inventory guard also reports pre-existing missing rows for
CuTe/compiler crates, and the all-example license guard stops on six unrelated
example lockfiles that require updates under `--locked`. The new attention
examples and affected workspace dependency audits pass. Toolchain parity,
source-reference, example smoke-contract, intrinsic currency and intrinsic ABI
history checks pass; the ABI ledger retains all 1,030 entries.

Both existing FP16 `256x352` GEMM examples were rebuilt at the unchanged
default optimization level 3. CuTe and CUTLASS-primitives kernels each passed
the CPU oracle for `M=512,N=712,K=392`, bias enabled, one persistent cluster,
and 364,544 checked output values. This exercises residual N/K tiles and
pipeline reuse rather than only the smallest divisible shape.

## Interpreting performance

CUDA events enclose repeated complete invocations with equal configured
warmups and iteration counts, hot buffers and no CUDA graphs. Allocation,
compilation, transfers and oracle checks happen outside timing. These windows
include host submission gaps, which differ between Rust and Python; they are
invocation throughput measurements, not profiler kernel-active latency.
Profiler durations, where recorded, are separate diagnostic measurements.
Small differences within the observed timing spread do not establish an
advantage. Consult the individual reports for the tested shapes and resources;
these examples do not claim coverage of every upstream attention option.

Final resource reports show active source register redistribution in both
kernels. Prefill has zero stack/local-memory traffic and retains one conservative
load-completion wait on its masked softmax path. Decode retains 16 bytes of local
stack per thread. These are explicit implementation limits; matching the selected
layouts, warp roles and pipeline schedule does not claim identical machine code.

## Recorded complete-invocation timings

Each Rust/CuTeDSL pair uses equal settings on the same B200. Medians are in
microseconds; the reports retain every sample and hardware snapshot.

| Profile | Warmups / invocations / windows | CuTe-RS | CuTeDSL |
| --- | --- | ---: | ---: |
| Prefill B1/S1024/H8 | 10 / 100 / 5 | 18.448 | 75.404 |
| Prefill B4/S4096/H8, sustained | 2000 / 301 / 5 | 289.984 | 292.127 |
| Causal GQA prefill B4/S4096/Hq8/Hkv2 | 10 / 100 / 5 | 150.318 | 168.248 |
| Decode B1/K1024/split1 | 10 / 100 / 5 | 29.613 | 18.059 |
| Decode B16/K8192/splits8 | 10 / 100 / 5 | 82.737 | 49.473 |

The small prefill difference includes Python submission gaps and is not evidence
of a fourfold GPU-kernel speedup. The initial dense persistent run drifted across
wide overlapping ranges; a separately retained sustained run stabilizes at
289.334–290.866 µs for Rust and 291.056–292.258 µs for CuTeDSL. The 0.73% median
difference supports comparable invocation throughput on this workload, not a
strong speedup claim. Decode was 1.64–1.67 times slower in these initial measurements. The linked
updated decode report records the subsequent parity results.

Raw evidence is in [prefill timing reports](results/timing/)
and the decode [small](../../sm100_mla_decode_fp16_cute/reference/results/benchmark-b200-small.json)
and [persistent](../../sm100_mla_decode_fp16_cute/reference/results/benchmark-b200-persistent.json)
reports. The sustained prefill run preserves the initial measurements instead
of replacing them.
