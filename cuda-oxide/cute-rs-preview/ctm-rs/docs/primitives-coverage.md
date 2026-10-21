# Rust CUTLASS primitives

`ctm_rs` is a Rust API over the existing NVVM instruction dialect, with no CuTe
layout algebra or Python dependency. It covers two-CTA SM100 FP16 GEMM, SM120
MXFP4 GEMM and explicit instruction paths used by attention kernels; it does
not provide full PTX parity.

- **Initial consumer:** `cutlass-primitives-mlir22`, pinned to CUTLASS compiler 4.7.0 /
  LLVM 22 / `sm_100a` or `sm_120a`; Pliron `06d99a3df938394583715aacf5eadb65b4791c09`.
- **Reference snapshots:** CUTLASS `0b55a2f691d69981583568fd9eb69687b1f0de8a`,
  DKG `b3f067d4db6ecac574dd5f1ba5ab0e9d2f05fdbf`. Their interfaces inform the
  facade; the installed 4.7 compiler determines accepted MLIR.

## Coverage

The primitives-only profile described below excludes CuTe operations. CTM is
also usable from the full `cutlass-mlir` profile alongside CuTe: the
[SM100 prefill](../../examples/sm100_fmha_prefill_cute/README.md) and
[MLA decode](../../examples/sm100_mla_decode_fp16_cute/README.md) examples use
CTM for explicit pipeline, TMEM, numeric and synchronization instructions,
while structured layouts, MMA and TMA retain their CuTe dialect paths. This
integration does not establish that every CuTe operation lowers through Rust
CTM or that an attention kernel is accepted by the primitives-only profile.

| Example | Covered path |
| --- | --- |
| [SM100 FP16](../../examples/fp16_gemm_256x352_ctm/README.md) | Two-CTA TMEM GEMM |
| [SM120 block-scaled](../../examples/blockscale_gemm_ctm/README.md) | Full three-stage persistent GEMM ported to register primitives |

- **No CuTe ops needed:** dynamic shared allocations use one external AS3 global
  with the maximum requested alignment. Full-CuTe translation is a separate model.
- **Discovery metadata:** `cute.kernel` accompanies `gpu.kernel`; the pinned
  compiler does not produce named kernels with `gpu.kernel` alone. The primitives
  profile registers no CuTe operations or types and rejects nested CuTe
  types/attributes before preparation can erase them.
- **Architecture limits:** the profile rejects TCGen05 on SM120a and register
  MXFP4 on SM100a. Native LLVM support for the new intrinsic is tested separately;
  libNVVM support is not advertised.

<details>
<summary>Detailed API mappings and compile probes</summary>

| Public Rust API | Importer boundary | Pliron operation | MLIR mapping | Real consumer probe |
| --- | --- | --- | --- | --- |
| `SharedRegion<F>`, typed local/cluster addresses | Existing shared allocation, pointer projections and explicit CVTA | `mir.shared_alloc`, `nvvm.cvta_generic_to_shared_offset` / `nvvm.cvta_generic_to_cluster_shared_offset` | Aligned AS3 globals; shared/cluster offsets retain address-space bits | TMA and MMA probes; checked constructors execute in the compiled path |
| `TensorMap<f16,F>`, host `ctm_rs::host::tma::encode_2d` | Transparent wrapper around canonical descriptor; grid-constant metadata | MIR array/struct and reference parameter | 128-byte by-value descriptor, alignment 64, `nvvm.grid_constant` | One-map and four-map kernels; descriptor-only cubin frame size zero |
| `TmemGuard<_,N,CtaGroup2>`, borrowed `view`/`read_32x32` | Existing literal group-2 intrinsics | `nvvm.tcgen05_alloc_cg2`, `dealloc_cg2`, `relinquish_alloc_permit_cg2` | `nvvm.tcgen05.alloc`, `.dealloc`, `.relinquish_alloc_permit` | One explicit instruction per lifecycle step; no added barriers/election |
| `SmemOperand<Role,F>`, `InstructionDescriptor<S>`, `mma<S,U>` | Existing shared-A MMA intrinsic, literal kind/group/collector | `nvvm.tcgen05_mma` | `nvvm.tcgen05.mma`, f16, CTA group 2 | Eight issues: M256/N192 and M256/N160, each fill/use/lastuse/discard |
| `load_32x32b_x32`, explicit `load_wait`, unsafe `decode_f32` | Existing raw 32-result load | `nvvm.tcgen05_ld_32x32b_x32_raw`, `tcgen05_load_wait` | One vector load, 32 ordered scalar extracts, one explicit wait | Raw results stay by value through decoding; wait precedes decode |
| `tma::load_2d_cg2`, `store_2d`, `prefetch` | Existing generated TMA and prefetch intrinsics | Multicast CG2 2D load, 2D store, prefetch | `nvvm.cp.async.bulk.tensor.*`, descriptor prefetch | Three typed loads, one store; distinct signed coordinates and mask checked |
| `bulk_commit_group`, `bulk_wait_group_read<N>` | Existing literal wait-group intrinsic | `nvvm.cp_async_bulk_commit_group`, `wait_group_read` | Bulk commit and read wait | Explicit commit followed by wait groups 0 and 7 |
| Barrier init/invalidate, arrive, expect-tx, parity try-wait | Canonical `ManagedBarrier` methods and intrinsics | Shared mbarrier operations | `nvvm.mbarrier.*` | Tokens/predicate remain observable; no software wait loop inserted |
| `cluster::map_shared`, `arrive_remote`, `CtaMask<G>` | Existing mapa and remote-arrival intrinsics | `nvvm.mapa_shared_cluster`, explicit CVTA, cluster arrival | AS7 mapa; explicit generic-to-AS7 offset; cluster-scoped arrival | Actual Rust rank-1 remote barrier path and native/MLIR address regressions |
| Counted CTA barriers, cluster barriers, fences | Existing generated intrinsics | Counted sync/arrive; cluster arrive/wait; proxy and TCGen05 fences | Corresponding NVVM operations; CTA `aligned` is a boolean property | All exposed modes in sync probe |
| `warp::elect_sync`, `warp_id`, thread/block/cluster coordinates | Existing special-register and election importers | NVVM sreg operations and `elect_sync` | NVVM election and sreg reads | Both election results and representative coordinates stored |

| SM120 addition | Instruction contract |
| --- | --- |
| `mma::{Mxf4A, Mxf4B, Accumulator16x8, ScalePairA, ScalePairB}` | Distinct operand roles, A4/B2/C4 register counts, paired UE8M0 scale bytes |
| `mma::mma_mxf4` | Canonical `v1:i1030` → `nvvm.mma.block_scale`, m16n8k64, E2M1, scale_vec 2X; explicit full-warp participation |
| `load_mxf4_a`, `load_mxf4_b_pair`, `store_f16` | Explicit ldmatrix.x4 and packed FP16/stmatrix.x2; kernel supplies physical addresses |
| `Mxf4Tile128`, `Mxf4ScaleTile`, `tma::load_2d` | B64 packed-byte tiles; unswizzled scales with 128-byte TMA alignment; CTA-local completion |

| CuTe example API coverage | Contract |
| --- | --- |
| `convert::{f32_to_bf16_rne,bf16_to_f32}`, `float::{rsqrt_approx_f32,lg2_approx_f32}` | Preserve the underlying conversion rounding and non-FTZ math instructions |
| `f32x2::PackedF32x2`, pack/unpack, add/mul/FMA | Distinct two-lane register carrier; bit-preserving packing and SM100 packed arithmetic |
| `warp::{shuffle_sync,shuffle_xor,reduce_sum_f32,all,sync_mask}` | Unsafe collectives with explicit participating-lane obligations |
| `thread::setmaxnreg_inc/dec<N>` | Unsafe warpgroup operations; immediate counts checked at compilation |
| `async_copy::cp_async_ca_4`, `cp_async_mbarrier_arrive_noinc` | Typed shared destination and read-only source; ready general barrier; explicit asynchronous lifetime obligations |
| `async_copy::{cp_async_ca_4_raw,cp_async_mbarrier_arrive_noinc_raw}` | Unsafe raw-pointer instruction entry points; caller establishes address spaces, initialization and completion lifetimes |
| `tcgen05::{commit,load_wait,store_wait,relinquish_alloc_permit}` | Group-one commit requires a ready barrier; completion, issuer and allocation participation remain explicit |
| `tcgen05::commit_address`, `tcgen05::tmem::{load_32x32b_x2,load_32x32b_x4,load_32x32b_x8,load_32x32b_x16,load_32x32b_x32}` and matching stores | Explicit shared barrier addresses and encoded TMEM addresses; caller establishes readiness, bounds, collective participation and completion |
| `thread::{index_1d,__unroll_config}`, `cluster_launch`, `memory::{U32x4,CuSimd}` | Canonical compiler markers, macro and storage types retain their identity |
| `mma::acc_coords`, `debug::trap` | Const accumulator coordinates and a diverging device trap |

</details>

## Types and explicit responsibilities

- **Physical formats:** A `[128,64]`, B `[96,64]` / `[80,64]` use 128-byte swizzle
  and 1024-byte stage alignment; C `[128,32]` uses 64-byte swizzle and 512-byte
  alignment. Collective M256 is not a local shared-memory extent.
- **Checked contracts:** descriptor offsets and field ranges are checked before
  canonical packing. Types reject wrong element/format/operand roles, unsupported
  MMA shapes and mixed groups. TMEM views borrow their guard and carry checked
  packed row/column addresses. Destructors neither synchronize nor free hardware
  resources.
- **Unsafe duties:** adopted storage must stay live in its claimed address space.
  Authors coordinate election, collective participation, publication, barrier
  phases, collector sequencing, asynchronous completion and remote lifetimes.
  Types establish local contracts, not cross-thread schedules.
- **Raw storage constructors:** `SharedRegion::from_raw_parts` returns validation
  errors; `from_raw_parts_or_trap` applies the same null/alignment/capacity checks,
  trapping on failure.
  Both require the same unsafe storage/lifetime guarantees. Trapping avoids a
  pointer-bearing `Result` carrier when recovery is unnecessary, as in SM120.
- **Host encoder:** checks shape/pitch/base constraints, then copies opaque bytes
  from a separate 128-byte-aligned CUDA-driver output temporary into the canonical
  64-byte-aligned kernel descriptor.

Typical issue site, after checked resource construction inside an unsafe kernel
section:

```rust,ignore
use ctm_rs::{groups::CtaGroup2, tcgen05::{descriptors::*, mma::{self, collector_a}}};
type Tile = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
let descriptor = InstructionDescriptor::<Tile>::new();
// dst, a and b carry matching checked group, format and role types.
mma::mma::<Tile, collector_a::Fill>(
    &dst, &a.k16::<0>(), &b.k16::<0>(), &descriptor, false,
);
// Explicitly commit/wait and coordinate consumers before reuse or readback.
```

## Reproduce validation

From `cuda-oxide/` with its pinned nightly and installed compiler:

```sh
python3 scripts/check-primitives-api-compile.py --output-dir /tmp/primitives-api-evidence
CUDA_OXIDE_TEST_CUTLASS_COMPILER="$HOME/.cargo/cuda-oxide/toolchains/cutlass/4.7.0/x86_64-cu13/lib/libCutlassCompiler.so" \
  cargo test -p cuda-oxide-codegen cutlass_primitives -- --ignored
```

- **Runner:** builds the in-tree CLI/backend with a fresh fixture target, checks
  imported markers and CFG instruction order, and records source status,
  compiler/backend hashes, MLIR, MIR logs and a validated cubin. It never launches
  kernels. See [Rust probes](../../../crates/rustc-codegen-cuda/tests/fixtures/primitives_api_compile/src/kernel.rs),
  [type-rejection tests](../tests/primitives_safety.rs) and
  [consumer tests](../../../crates/cuda-oxide-codegen/src/tests/cutlass/primitives.rs).
- **Evidence limits:** compilation establishes neither numerical correctness,
  progress nor SOL performance. SM120 includes a bit-exact GPU/CPU oracle and
  persistent-grid/sanitizer runner. SM100 execution needs compatible hardware;
  the local RTX 5090 cannot run its cubin.
- **Stack/register pressure:** the MMA probe retains a 128-byte final stack frame
  for its decoded-output carrier; the raw load carrier is removed from the decode
  input path. Measure register pressure and final output materialization during
  the port.
- **Deferred:** LIR, further CuTe equivalences, other architectures/groups/MMA
  kinds, arbitrary physical layouts, token wait convenience APIs and full
  intrinsic parity. Native APIs outside the detailed table may lack primitives
  MLIR mappings; unsupported selectors/operations fail at export.
- **Checked API lowering:** supports Direct and Single Rust enum layouts.
  Niche/unknown layouts and enum payloads requiring bool-byte or named-space-pointer
  storage adapters are explicitly rejected.
