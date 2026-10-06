# Supported Features

This appendix presents the cuda-oxide feature matrix: every compiler capability,
runtime API, and hardware feature along with its current support status. The
data is drawn from the compiler/runtime sources and the test suite.

**Legend:** **Full** = tested and working, **Partial** = ships and works but
has a known gap (called out in the row description), **Planned** = on the
roadmap, **N/A** = not applicable or no identified need.

---

## Compiler: Memory Model

| Feature | Status | Description |
|:--------|:-------|:------------|
| HMM / Unified Memory Management | **Full** | GPU directly reads/writes host memory without `cudaMemcpy`. Reference captures in closures leverage HMM for host pointer access. Requires Turing+ GPU, Linux 6.1.24+, CUDA 12.2+. |
| Unified Struct ABI (no `#[repr(C)]`) | **Full** | Device struct layout matches host exactly. The compiler queries rustc's actual layout and reproduces it with explicit padding in LLVM IR. Works with `#[repr(Rust)]` default. |
| Dynamic Layout Matching | **Full** | Compiler queries rustc's `fields_by_offset_order()` and byte offsets, builds LLVM structs with correct field order and explicit padding bytes. Independent of LLVM's datalayout. |
| Packed Layouts (`#[repr(packed)]`) | **Partial** | Field addresses formed through a pointer (`addr_of!((*p).field)`) use rustc's exact byte offsets, so `read_unaligned`/`write_unaligned` round-trips work, including `packed(N)`. Byte-faithful packed structs can also use packed LLVM storage for by-value construction/load/store and `[Packed; N]` element addressing. Recursively promotable packed constant arrays use the immutable-device-global path when the selected natural or packed LLVM representation reproduces rustc's field offsets and stored size exactly. Overlapping or otherwise non-representable layouts remain rejected. |
| Pointer Distance (`offset_from`) | **Full** | `ptr_offset_from` / `ptr_offset_from_unsigned` intrinsics (and the `offset_from`, `offset_from_unsigned`, `byte_offset_from`, `byte_offset_from_unsigned` methods) lower to an address difference divided by the rustc-reported pointee size, returning `isize` (signed) or `usize` (unsigned). Errors on a zero-sized pointee. |
| Volatile Load/Store | **Full** | `core::ptr::read_volatile` / `write_volatile` carry an explicit volatile bit through MIR import, mem2reg (volatile accesses are never promoted), MIR-to-LLVM lowering, and textual export (`load volatile` / `store volatile`). Emits `ld.volatile` / `st.volatile` in PTX. |
| Bulk Copy (`copy_nonoverlapping`) | **Full** | `core::ptr::copy_nonoverlapping` lowers to a `mir.memcpy` op and then `llvm.memcpy`, with the element count scaled to bytes for the pointee. The intrinsic overload suffix is derived from the operand address spaces and length width. |

## Compiler: Type System

| Feature | Status | Description |
|:--------|:-------|:------------|
| Generics and Monomorphization | **Full** | Generic kernels and device functions with trait bounds. Monomorphized instances collected from rustc MIR. Const generics supported. |
| Enums (`Option<T>`, `Result<T,E>`, custom) | **Full** | Full enum support including discriminant extraction and payload access. Pattern matching on enums works. |
| Struct Construction and Field Access | **Full** | Struct literals, field access, pass-by-value and return values. User-defined structs supported without annotations. |
| Array Types (`[T; N]`) | **Full** | Static construction, constant- and runtime-index access. Array value constants (bare and nested) materialized. Mutable arrays auto-promoted to memory-backed. |
| `CuSimd<T, N>` SIMD Type | **Full** | Generic SIMD register type with named accessors (`x`/`y`/`z`/`w`), runtime and compile-time indexing, `to_array` conversion. |
| ABI Scalarization | **Full** | Slices are scalarized at kernel boundaries (`&[T]` -> `(ptr, len)`, reconstructed inside the function). Structs and closures pass by value as one byval `.param`; field flattening still applies on internal device-to-device calls. |
| Grid-constant parameters | **Full** | `#[grid_constant] value: &T` passes `T` by value and retains a grid-wide read-only address. Host launches are unsafe. Nonzero sized pointees without interior mutability, launch-scoped references, and Volta+ targets are required. |

Array value constants support primitive leaves (integers, `f16`, `f32`,
`f64`), nested arrays, and tuples recursively composed of supported scalar,
enum, tuple, and zero-sized fields. Tuple element strides and field offsets
come from rustc layout, including internal and trailing padding; direct tuple
value constants use the same layout-aware decoder. Struct constants (direct
and promoted-by-reference) also read every field at its rustc layout offset,
so padded, reordered, `#[repr(C)]`, and nested shapes decode correctly, and a
struct's stored size is its padded size, which fixes the element stride for
arrays of padded structs inside constants. Pointer-free initialized union
constants are materialized from rustc's evaluated storage image without
guessing an active field: initialized bytes are preserved, uninitialized
inactive bytes remain `undef`, and the byte image is transmuted into the
layout-exact union type. This includes direct unions, unions nested in tuple or
struct constants, runtime-indexed `[U; N]`, and `MaybeUninit<T>` constants.
Bare arrays whose elements are recursively promotable structs use the same
immutable-device-global path as scalar and tuple tables, avoiding a per-thread
local table copy for read-only uses. This promotion also admits supported thin
pointer/reference leaves. Their evaluated byte image remains byte-exact while
pointer slots are preserved as symbolic device-global relocations, including
non-zero byte addends into referenced device statics. Relocation targets are
materialized explicitly, and promoted-global deduplication includes relocation
identity as well as type and bytes so byte-identical tables that point at
different statics cannot alias.

Pointer-to-array constants such as `const R: &[Struct; N] = &TABLE` use the same
promoted immutable global when every element is recursively promotable and the
converted storage size matches rustc's layout. When the outer pointer selects a
subrange of a backing allocation, relocation source offsets inside that range
are rebased into the promoted initializer while preserving their target and
addend. Unsupported bare array constants retain the existing element-wise
fallback where available; pointer-to-array constants continue to fail closed
when no correct fallback exists. Zero-byte over-aligned struct leaves (for
example, `repr(align(N))` ZSTs) remain on the existing alignment-sensitive value
path instead of this promotion path. Promotion includes `repr(packed)` and
`repr(packed(N))` structs when lowering can reproduce their recorded field
offsets with an exact packed LLVM struct; the value and reference forms
deduplicate to the same immutable initializer. Overlapping or otherwise
non-representable struct layouts remain outside immutable promotion and keep
the existing fail-closed behavior.

Thin pointer fields in array, tuple, and struct **const** values that do not take
the immutable-table promotion path are materialized via `MirGlobalAllocOp` per
field, including non-zero byte addends into a static (see
`struct_constant_provenance`, `tuple_constant_provenance`,
`tuple_array_provenance`). The `array_constants` regression also covers a
promoted pointer-bearing tuple table with both a zero-addend static reference
and a non-zero static-subobject addend, and verifies that optimized code does
not retain a per-thread table depot.

Slice fat-pointer fields in aggregate constants are also supported when their
data pointer relocates to a device static and the pointee is a same-element
array-to-slice view. Their literal `usize` length metadata is decoded
independently, including non-zero static byte addends and nested aggregate field
offsets. Thin-pointer-only union constants preserve the same relocation
provenance, including non-zero addends, by reconstructing one typed pointer
carrier instead of transmuting placeholder bytes. Compatible `SharedRef`
alternatives must have the same translated pointee type; `UniqueRef` union
constants remain rejected. Raw-pointer
alternatives may differ only in pointee view while retaining the same raw kind,
mutability, and address space. Relocation-free pointer/integer unions whose
storage is exactly one fully initialized, naturally aligned pointer word, whose
pointer alternatives are generic raw pointers of one kind, and whose integer
alternatives are full-width may instead use rustc's evaluated byte image. The
importer transmutes that image only to the integer field and inserts the field
into the union, so no inactive pointer alternative is materialized.
Relocation-bearing pointer/integer unions, fat or nested pointer storage in
unions, over-aligned/padded pointer unions, unsupported fat-pointer metadata,
and pointer-to-array union constants (`&[U; N]`) remain rejected. Top-level
thin-pointer-only device-global union initializers may preserve one full-width
relocation at byte zero; mixed pointer/integer device-global initializers remain
rejected.

Enum constants preserve payload relocations to device statics, including
non-zero byte addends. This includes niche-encoded `Option<&T>` and
direct-tagged enum layouts, both for direct thin-reference payloads and for
pointers nested inside tuple, struct, or array payload fields. Relocation-carrying
enum constants can also be nested inside tuple, struct, and array constants.
Anonymous promoted allocations remain unsupported.

## Compiler: Closures

| Feature | Status | Description |
|:--------|:-------|:------------|
| Move Closures (`FnOnce`) | **Full** | Closures that capture by value. The whole closure struct is pushed as one byval kernel argument. `move \|x\| x * factor` pattern. |
| Reference Closures (`Fn`/`FnMut`) | **Full** | Non-move closures that capture by reference. The closure struct (containing host pointers) still travels as one byval argument; the GPU reads through those pointers via HMM. |
| Host-to-Device Closures | **Full** | Closures defined on host passed to generic kernels. Polynomial evaluation with captured coefficients tested. |
| Device-Internal Closures | **Full** | Closures created and used entirely on device, including closures passed to device functions. |

## Compiler: Control Flow

| Feature | Status | Description |
|:--------|:-------|:------------|
| Match Expressions (integer switch) | **Full** | Multi-way match on integers. Generates chain of conditional branches. |
| Match on Enums | **Full** | Pattern matching on `Option<T>` and custom enums. Discriminant extraction + payload access. |
| For Loops (range, iterator, enumerate) | **Full** | Full iterator desugaring: range-based, `slice.iter()`, `enumerate()`, nested loops, `break`, `continue`. |
| While Loops / If-Else | **Full** | Baseline control flow fully supported. |
| Break and Continue | **Full** | `break` and `continue` in for/while loops, including early exit. |
| Loop Unroll Annotations | **Partial** | `#[unroll]` and `#[unroll(N)]` request unrolling of explicit counted `while` loops. Nested loops and multiple `continue` paths work; full unrolling preserves `break` paths and multiple exit targets, while partial unrolling requires a positive-step `<`/`<=` loop with an invariant limit and only the normal header exit. Requests are capped at 1,024 copies, 8,192 cloned blocks, and 65,536 cloned operations. |
| Monomorphization-Dead Branches | **Partial** | Branches that become dead after generic specialization (e.g. the const-false arm of `if M::ENABLED`) are ignored by symbol collection, panic checks, and pointer address-space inference, so panic-only hooks in dead arms compile. Only switches rustc itself folds are pruned: a constant discriminant operand or a direct single-assignment constant. Multi-step constant copy chains keep both arms, matching rustc's host monomorphization; this is deliberate, not a general constant-propagation pass. |

## Compiler: Arithmetic and Casting

| Feature | Status | Description |
|:--------|:-------|:------------|
| 64-bit Arithmetic | **Full** | Full 64-bit integer arithmetic including shifts, bitwise ops, and descriptor field packing. |
| Type Casting (all kinds) | **Full** | IntToInt, IntToFloat, FloatToInt, FloatToFloat, Transmute (bitcast), PtrToPtr, PtrToInt, IntToPtr, pointer coercions. |
| Packed bf16x2 FMA | **Full** | `bf16x2::fma_bf16x2(a, b, c)` lowers to PTX `fma.rn.bf16x2`, two bf16 lanes per `u32`. sm_80+. |

## Compiler: Interop

| Feature | Status | Description |
|:--------|:-------|:------------|
| Bi-directional LTOIR Support | **Full** | Rust kernels call CUDA C++ device functions **and** C++ calls Rust device functions. Via NVVM IR → libNVVM → LTOIR → nvJitLink. |
| Device FFI (`extern "C"`) | **Full** | `#[device] extern "C" { fn ... }` declarations for external LTOIR functions. CUB/CCCL integration demonstrated. |
| MathDx FFI (cuFFTDx / cuBLASDx) | **Full** | cuFFTDx (8/16/32-point thread-level FFT), cuBLASDx (32x32x32 block-level GEMM) via LTOIR. |
| Tile interop | **Partial** | Inter-kernel interop works today: a [cutile-rs Tile kernel](https://github.com/NVlabs/cutile-rs) and a cuda-oxide SIMT PTX kernel can run in one host process on the same CUDA stream over shared device tensors. Intra-kernel Tile interop is work in progress and tracked in [#96](https://github.com/NVlabs/cuda-oxide/issues/96). |
| Cross-Crate Kernels | **Full** | Kernels and device functions defined in library crates with monomorphization at the binary crate use site. |

## Compiler: Functions

| Feature | Status | Description |
|:--------|:-------|:------------|
| `#[kernel]` Attribute | **Full** | Marks functions as GPU kernel entry points (`ptx_kernel` calling convention). Multiple kernels per file. |
| `#[device]` Helper Functions | **Full** | Device-side helper functions callable from kernels. `#[inline(always)]` is preserved as the LLVM `alwaysinline` attribute (emitted alongside the convergent group and any `!dbg` scope), so `opt` honors the inline intent. |
| Standalone `#[device]` Functions | **Full** | Device functions compiled without any kernel present. Clean export names for C++ consumption. |
| Multi-Kernel Modules | **Full** | Multiple `#[kernel]` functions in a single source file compile to a single PTX module. |

## Compiler: Compilation Pipeline

| Feature | Status | Description |
|:--------|:-------|:------------|
| Unified Single-Source Compilation | **Full** | Host and device code in the same file. Custom rustc codegen backend intercepts codegen. No `#[cfg]` needed. |
| PTX Output | **Full** | Default output: Rust MIR → `dialect-mir` → `mem2reg` → annotated loop unroll → LLVM dialect → LLVM IR → `llc` → PTX. Targets sm_80 through sm_100a. |
| NVVM IR Output | **Full** | Selects LLVM 7 typed-pointer syntax for pre-Blackwell GPUs and opaque-pointer syntax for Blackwell and newer GPUs. The generated module is verified by libNVVM, and unsupported legacy operations produce a compile error. |
| LTOIR Linking | **Full** | Device-side LTO via libNVVM and nvJitLink. |
| Float Math Intrinsics (libdevice) | **Full** | Rust `f32`/`f64` math methods (`sin`, `cos`, `exp`, `pow`, `sqrt`, ...) lower to CUDA libdevice (`__nv_*`) on pre-Blackwell and Blackwell GPUs. cuda-oxide selects the matching NVVM IR syntax automatically. On Blackwell, the runtime can also JIT PTX produced from a standard pre-Blackwell target such as `sm_86`. |
| Pipeline Inspection | **Full** | `cargo oxide pipeline <example>` shows imported and post-`mem2reg` MIR, LLVM dialect, exported LLVM IR, and PTX. |
| PTX Inspect | **Full** | `cargo oxide inspect <example>` builds and prints generated PTX without the full pipeline dump. |
| Local Clean | **Full** | `cargo oxide clean` removes project-local `target/` directories and generated device artifacts (`.ptx`, `.ll`, `.opt.ll`, `.ltoir`, `.cubin`, `.target`, `.options`, `.cubin.target`), never the shared `~/.cargo/cuda-oxide/` cache. |
| Compute Sanitizer Wrapper | **Full** | `cargo oxide sanitize <example>` builds the example and runs the host binary under NVIDIA Compute Sanitizer (`memcheck`, `racecheck`, `initcheck`, or `synccheck`). |
| cuda-gdb Source Debugging | **Full** | `cargo oxide debug` builds device debug information on the PTX path and launches `cuda-gdb`. Legacy NVVM IR does not yet support debug metadata. |
| cuda-gdb Local / Argument Inspection | **Partial** | Full mode keeps supported locals in memory. It disables `ScalarReplacementOfAggregates` and `SingleUseConsts`, keeps `ReferencePropagation` and general MIR inlining, and outlines only `DisjointSlice::get_mut`. Common scalars, pointers, aggregates, closures, enums, and selected projections work; see [Debugging and Error Handling](../gpu-programming/error-handling-and-debugging.md) for current limits. |

## Compiler: Inline PTX

| Feature | Status | Description |
|:--------|:-------|:------------|
| `ptx_asm!` Macro | **Partial** | CUDA inline PTX with `%0` operands, `in`, `out`, and `inout`; up to 16 output operands across `out` with `=`-prefixed constraints and `inout` with `+`-prefixed constraints; up to 16 explicit inputs; CUDA register constraints `h`, `r`, `l`, `q`, `f`, and `d`; immediate integer constraint `n`; compile-time string constraint `C`; `clobber("memory")`; and `options(register_only)` for pure register snippets. With multiple output operands, the macro writes tuple results back in declaration order. By default, snippets are treated as side-effecting and stay inside their current control flow. Use `options(register_only, may_diverge)` only for pure snippets that are safe to move across divergent control flow; **never** use it for `.sync` instructions or collectives. More than 16 output operands are not implemented yet. |

---

## Runtime Library: Safety

| Feature | Status | Description |
|:--------|:-------|:------------|
| `DisjointSlice<T, IndexSpace>` | **Full** | Bounds-checked parallel write output slice. `IndexSpace` rejects mismatched layouts; uniqueness also requires matching prepared launch geometry (or a raw unsafe proof). |
| `ThreadIndex<'kernel, IndexSpace>` | **Full** | Opaque, non-transferable witness. `index_1d` uniqueness requires inactive Y/Z dimensions, proven by a `domain = 1` prepared launch or by the caller of a raw unsafe launch. |
| Proof-carrying static views | **Full** | A checked `u32` thread index proves one complete element or tile, then `at_const` accesses compile-time positions without another runtime bounds check. |
| `PreparedLaunch<K>` | **Full** | Checked, reusable launch geometry branded for the exact kernel. Raw `LaunchConfig` generated methods are unsafe. |
| `ManagedBarrier` Typestate | **Full** | Compile-time barrier lifecycle: `Uninit → Ready → Invalidated`. Invalid transitions are compile errors. |

## Runtime Library: Atomics

| Feature | Status | Description |
|:--------|:-------|:------------|
| Device-Scope Atomics | **Full** | `DeviceAtomic{U32,I32,U64,I64,F16,F32,F64}` with `.gpu` scope. All 5 orderings. |
| Block-Scope Atomics | **Full** | `BlockAtomic{U32,I32,U64,I64,F16,F32,F64}` with `.cta` scope. |
| System-Scope Atomics | **Full** | `SystemAtomic{U32,I32,U64,I64,F16,F32,F64}` with `.sys` scope. For CPU-GPU shared data. |
| `core::sync::atomic` Support | **Full** | Standard library atomic types lowered to PTX `atom.sys` instructions. |

## Runtime Library: Shared Memory

| Feature | Status | Description |
|:--------|:-------|:------------|
| Static Shared Memory | **Full** | `SharedArray<T, N, ALIGN>` — compile-time sized, block-scoped. Optional alignment up to 256B. |
| Dynamic Shared Memory | **Full** | `DynamicSharedArray<T, ALIGN>` — runtime-sized, set via `LaunchConfig::shared_mem_bytes`. |
| Distributed Shared Memory (DSMEM) | **Full** | Direct access to other blocks' shared memory within a cluster. `map_shared_rank()` for address mapping. sm_90+. |

## Runtime Library: Thread and Synchronization

| Feature | Status | Description |
|:--------|:-------|:------------|
| Thread/Block/Grid Intrinsics | **Full** | `threadIdx`, `blockIdx`, `blockDim`, `gridDim`. Index witnesses are layout-typed; their uniqueness also depends on matching launch dimensionality. `index_2d_runtime(&slice)` resolves against the slice's own row width, bound once by the host. See [The Safety Model](../gpu-safety/the-safety-model.md). |
| Block Synchronization | **Full** | `sync_threads()` — thread block barrier. |
| Async Barriers (mbarrier) | **Full** | Hardware async barriers for Hopper+: init, arrive, test_wait, try_wait, inval. |
| Cluster Synchronization | **Full** | `cluster_sync()` for all blocks in a cluster. sm_90+. |
| Fence Operations | **Full** | `fence_proxy_async_shared_cta()` for TMA visibility, `nanosleep(ns)`. |

## Runtime Library: Warp

| Feature | Status | Description |
|:--------|:-------|:------------|
| Warp Shuffle Operations | **Full** | `shuffle`, `shuffle_xor`, `shuffle_down`, `shuffle_up`. Unsuffixed forms take `u32`; `_f32`, `_u64`, `_f64` variants and a `_sync` form of each. |
| Warp Vote Operations | **Full** | `all(pred)`, `any(pred)`, `ballot(pred)` → bitmask. |
| Lane/Warp ID | **Full** | `lane_id()` (0–31), `warp_id()`. Direct register reads. |
| Warp Reduction (`redux.sync`) | **Full** | One-instruction full-warp reduction. Integers on sm_80+: `redux_sync_add`, `_min_{u32,i32}`, `_max_{u32,i32}`, `_and`, `_or`, `_xor`. `f32` min/max with optional `.abs` and `.NaN` on `sm_100a`/`sm_100f`/`sm_103a`/`sm_103f`. No `f64` form. |

## Runtime Library: Cooperative Groups

| Feature | Status | Description |
|:--------|:-------|:------------|
| Typed Group Handles | **Full** | `Grid`, `Cluster`, `ThreadBlock`, `WarpTile<N>` (N ∈ {1,2,4,8,16,32}), `CoalescedThreads`. |
| Group Universal API | **Full** | `size()`, `thread_rank()`, `sync()` on every group handle. |
| Warp Tile Partitioning | **Full** | `ThreadBlock::tiled_partition::<N>()` partitions a block into fixed-size `WarpTile<N>` groups; the block thread count must be evenly divisible by `N`. `coalesced_threads()` materialises the active-lane group. |
| Warp Collectives | **Full** | `ballot`, `all`, `any`, `shfl`, `shfl_xor`, `shfl_down`, `shfl_up` (`u32` and `f32`); `match_any` / `match_all` (`i32` and `i64`); `active_mask`. |
| Warp Reductions / Scans | **Full** | `warp_reduce`, `warp_scan` (inclusive). `Sum`/`Min`/`Max` for `u32`/`i32`/`f32`; `BitAnd`/`BitOr`/`BitXor` for `u32`. |
| Block Reductions / Scans | **Full** | `block_reduce`, `block_scan` (inclusive). `NUM_WARPS` is the shared-memory capacity in warp totals and must be at least `ceil(block_threads / 32)`; partial final warps are supported. Same op/type matrix as warp variants. |
| Cooperative Kernel Launch | **Full** | `#[cooperative_launch]` on a `#[cuda_module]` kernel (or `unsafe { cuda_launch! { cooperative: true, ... } }`) enables `Grid::sync()` for grid-wide barriers. |

## Runtime Library: Debug

| Feature | Status | Description |
|:--------|:-------|:------------|
| `gpu_printf!` Macro | **Full** | Formatted GPU output with full format specifier support. Lowers to `vprintf`. |
| `gpu_assert!` Macro | **Full** | The no-message form calls `trap()` on failure. The string-literal message form calls CUDA's device-side `__assertfail`, reports message and call-site metadata, and surfaces `CUDA_ERROR_ASSERT`. |
| Debug Intrinsics | **Full** | `clock()`, `clock64()`, `trap()`, `breakpoint()`, `prof_trigger::<N>()`. |

## Runtime Library: Kernel Launch

| Feature | Status | Description |
|:--------|:-------|:------------|
| `#[cuda_module]` Typed Launch | **Full** | Embedded module loading with typed sync/async arguments. Raw configuration methods are unsafe. |
| `#[launch_contract]` / `PreparedLaunch<K>` | **Full** | Checked dimensionality, exact block shape, resources, capabilities, context, and kernel identity. |
| `cuda_launch!` Macro | **Full** | Unsafe lower-level launch for runtime-loaded modules; requires `unsafe { }`. |
| `cuda_launch_async!` Macro | **Full** | Unsafe lower-level lazy launch; requires `unsafe { }`. |
| `#[launch_bounds]` | **Full** | Occupancy hints: max threads per block, min blocks per SM. |
| `#[cluster_launch]` | **Full** | Compile-time cluster dimensions. Emits `.reqnctapercluster` in PTX. |

## Runtime Library: TMA

| Feature | Status | Description |
|:--------|:-------|:------------|
| TMA Bulk Tensor Copy (1D–5D) | **Full** | `cp_async_bulk_tensor_{1..5}d_g2s`. 128-byte TMA descriptors. sm_90+. |
| TMA Multicast | **Full** | Single TMA load broadcast to all CTAs in cluster. sm_100a for full multicast. |
| TMA Commit/Wait Groups | **Full** | `cp_async_bulk_commit_group`, `cp_async_bulk_wait_group` for async completion tracking. |

---

## Runtime Library: Matrix and Tensor Cores

| Feature | Status | Description |
|:--------|:-------|:------------|
| Warp-Level MMA (`wmma`) | **Full** | Register-only `mma.sync` shapes, `movmatrix`, and warp-cooperative `ldmatrix` loads. |
| Sparse MMA | **Full** | Structured-sparsity `mma.sp` shapes alongside the dense ones, in the same `wmma` module. |
| Warpgroup MMA (`wgmma`) | **Partial** | Hopper `sm_90a`: fence/commit/wait pipeline, shared-memory descriptors, and `m64n64k16` MMA with `bf16`/`f16` inputs and `f32` accumulate. Gap: the lowering covers specific proven loop patterns (`bf16` works in straight-line code, counted K-loops, and partial-wait pipelines; `f16` in straight-line code only), and `tf32` calls are rejected pending [#1076](https://github.com/NVlabs/cuda-oxide/issues/1076). |
| Tensor Core Gen 5 (`tcgen05`) | **Full** | Blackwell sm_100+: TMEM alloc/dealloc, MMA, `stmatrix`, CTA-pair (cg2) variants. |
| Accumulator Fragment Algebra (`mma_frag`) | **Full** | Index algebra for the `m16n8k16` accumulator, so a lane can address its own slots of the `[f32; 4]` fragment. |
| FP8 / FP6 / FP4 Formats | **Partial** | Conversions (`convert::cvt_*` for `e4m3`/`e5m2`) and the matrix path (FP8 `mma.sync` shapes, the `mxf8f6f4` shapes, tcgen05 descriptors) ship; the `mma_mxf8f6f4` example compiles them. Gap: no *arithmetic* on these formats. There's no add/mul/min/max the way `f16` and `bf16` have, so values are carried as packed bit patterns and converted before use. |

---

## Not Yet Implemented

| Feature | Status | Notes |
|:--------|:-------|:------|
| Rust `asm!` macro | **Planned** | Use `ptx_asm!` for CUDA inline PTX. Direct lowering of Rust MIR `InlineAsm` is not implemented. |
| Dynamic Dispatch (`dyn Trait`) | **N/A** | Use generics with static dispatch. Haven't found a real need for this. |
| Heap Allocation (`Box`, `Vec`) | **N/A** | CUDA has a device-side heap (`malloc`/`free` in kernels), and the compiler allows the `alloc` crate through -- but no device-side `#[global_allocator]` is wired up today. Even if it were, device `malloc` is extremely slow (serialized, fragmented, uncoalesced). Use slices and `SharedArray`. |
| `String` / `format_args!` | **N/A** | Use `gpu_printf!` for formatted output. |
| Panic / Unwinding | **N/A** | Panic paths exist in MIR but the compiler strips `core::panicking::*` and all unwind edges. The GPU hardware *can* support unwinding (absolute branches + per-thread call stack tracking post-Volta), but the CUDA toolchain (nvcc/ptxas) doesn't expose it today -- no landing pads survive to PTX. If a panic path is reached at runtime the GPU traps (same as `panic=abort`). NVIDIA has an active project to add C++ exception support to CUDA for automotive safety; the current cuda-oxide design is forward-compatible with that work. Use `gpu_assert!()` + `trap()` for explicit runtime checks today. |
| Standard Library (`std`/`alloc`) | **N/A** | `std` is forbidden. `alloc` is allowed by the collector but has no backing allocator. Only `core` is fully functional. `Option`, `Result`, iterators all work. |
| Texture Memory | **N/A** | Lower priority given TMA availability on Hopper+. |
