# ctm-rs: typed instruction primitives (preview)

`ctm-rs` is the instruction layer beneath `cute-rs`: typed MMA operands, physical
shared-memory formats, tensor maps, explicit barriers and collective TMEM
allocation over `cuda-device` intrinsics. It knows nothing about CuTe layouts.

```text
cute-rs       tensors, tiles, pipelines, tiled MMA     dialect-cute
ctm-rs        instructions with typed operands        dialect-nvvm
cuda-device   raw intrinsics
```

![ctm-rs at a glance](../../cuda-oxide-book/cute/images/ctm-rs-at-a-glance.svg)

## What it covers

| Family | Operations | Targets |
| --- | --- | --- |
| Shared memory | `SharedRegion<F>` with checked alignment and capacity, typed CTA and cluster addresses | both |
| TMA | 2D load, two-CTA multicast load, 2D store, descriptor prefetch, bulk commit and wait; host `create_tensor_map_tiled` behind the `host` feature | both; multicast on `sm_100a` |
| Barriers, fences and clusters | mbarrier init, arrive, expect-tx, parity wait; named CTA barrier sync and arrive (`bar` and `barrier.cta`, plain and aligned); `fence.mbarrier_init`, `fence.proxy.async` for `shared::cta` and its two `sync_restrict` forms, `fence.acquire.sync_restrict`; cluster arrive and wait, `mapa`, remote arrive, `CtaMask` | both |
| tcgen05 | `TmemGuard` collective alloc and dealloc, F16 two-CTA MMA through `InstructionDescriptor`, 32x32b TMEM load and store, commit, fences | `sm_100a` |
| Register MMA | m16n8k16 F16 and BF16 with F32 accumulate; m16n8k64 MXF4 with UE8M0 scales; ldmatrix x4, stmatrix x2 | both; MXF4 on `sm_120a` |
| Async copy | 4-byte `cp.async`, noinc barrier arrive, commit and wait groups | both |
| Warp | shuffle (full-warp and masked), elect, vote (all, any, ballot), lane id, active mask, sum reduce, `setmaxnreg` | both |
| Numerics | BF16 round-to-nearest convert, packed f32x2 add, mul and fma, approximate ex2, lg2, rcp, rsqrt and sin, trap | both |

Not covered: other tcgen05 kinds and TMEM shapes, atomics and reductions, wgmma,
libdevice, arbitrary physical layouts. An unsupported selector fails at export,
not at run time.

## What the types check, and what you own

| Checked at compile time | Yours at every `unsafe` boundary |
| --- | --- |
| Operand roles, element formats, MMA shapes | Which lanes take part in a collective |
| Descriptor fields and packed address ranges | Barrier phases and arrival counts |
| Borrowed TMEM views and shared regions | Asynchronous completion before reuse |
| Alignment and capacity of adopted storage | Collector fill and release order, pointer lifetimes |

Destructors neither synchronize nor free hardware resources.

## Use

```rust,ignore
use ctm_rs::{groups::CtaGroup2, tcgen05::{descriptors::*, mma::{self, collector_a}}};

type Tile = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
let descriptor = InstructionDescriptor::<Tile>::new();
// dst, a and b carry matching group, format and role types.
mma::mma::<Tile, collector_a::Fill>(&dst, &a.k16::<0>(), &b.k16::<0>(), &descriptor, false);
// Commit, wait and hand off to the consumers before reuse or readback.
```

[`fp16_gemm_256x352_ctm`](../examples/fp16_gemm_256x352_ctm) (`sm_100a`) and
[`blockscale_gemm_ctm`](../examples/blockscale_gemm_ctm) (`sm_120a`) use ctm-rs alone
through the primitives profile, `CUDA_OXIDE_MLIR_PROFILE=cutlass-primitives-mlir22`.
The SM100 attention examples use it beside `cute-rs` in one module under the default
profile.

The crate is `no_std`. Kernel macros and canonical device types are reexported;
crates that use them still depend on `cuda-device` directly because the macro
expansions name it.

## Modules

| Modules | Responsibility |
| --- | --- |
| `memory`, `mma` | storage formats and typed MMA operands |
| `tcgen05` | descriptors, MMA issue, TMEM ownership and completion |
| `async_copy`, `tma` | shared-memory copies and tensor-map operations |
| `mbarrier`, `barrier`, `fence`, `cluster`, `groups` | mbarrier readiness, named CTA barriers, memory fences, cluster operations, CTA-group types |
| `warp`, `thread` | collectives, coordinates, register limits |
| `convert`, `float`, `f32x2`, `debug` | conversion, arithmetic and debugging |
| `host::tma` | host tensor-map encoding behind the `host` feature |

## Check

From `cuda-oxide/` with its pinned toolchain:

```sh
cargo test -p ctm-rs --all-targets
cargo test -p ctm-rs --all-targets --features host
python3 scripts/check-primitives-api-compile.py --example fp16 --output-dir /tmp/ctm-fp16
```

The tests cover formats, carriers, storage checks and compile-time rejection of
invalid contracts. The compile check builds the fixtures through the CUTLASS
consumer and records the MLIR and a validated cubin; it launches nothing. GPU
correctness and sanitizer commands live in each example's README.
