# Pliron Dialects

cuda-oxide does not lower Rust to PTX in a single, heroic transformation. It
works across three pliron dialects on the way down, each modeling a different
level of abstraction: two defined locally (`dialect-mir`, `dialect-nvvm`) and
the LLVM dialect provided by the upstream `pliron-llvm` crate. This chapter
walks through all three -- their types, their operations, and how they fit
together to form the compilation pipeline.

Two further pliron dialects live in this tree off that path and are not covered
here: `dialect-iket`, the compiler-facing form of in-kernel event tracing, and
`dialect-ptx`, a structured terminal PTX dialect that can be built directly or
projected from parsed PTX source. Each crate's README is the reference.

If you have not read the [Pliron -- Pliron IR (MLIR-like)](pliron.md) chapter yet, now
is a good time. The concepts there (operations, types, attributes, regions,
`Ptr<T>`, def-use chains) are the building blocks of everything on this page.

---

## The Three Dialects at a Glance

| Dialect          | Purpose                      | Level                                                            |
| :--------------- | :--------------------------- | :--------------------------------------------------------------- |
| **dialect-mir**  | Models Rust MIR semantics    | Highest -- Rust types, tuples, enums, slices, checked arithmetic |
| **LLVM dialect** | Models LLVM IR               | Middle -- flat types, GEP, PHI-ready control flow                |
| **dialect-nvvm** | Models NVIDIA GPU intrinsics | Orthogonal -- thread indexing, warps, TMA, WGMMA, tcgen05        |

The LLVM dialect is not a cuda-oxide crate: its modeling (ops, types,
attributes, op-interfaces) lives upstream in the `pliron-llvm` crate, which
cuda-oxide consumes as a dependency.

`dialect-nvvm` is "orthogonal" rather than a layer in the stack because its
operations appear *alongside* LLVM dialect operations, not below them. A
warp shuffle and an integer add coexist in the same function body.

Data flows through the pipeline like this:

```text
dialect-mir ──(mem2reg)──▶ dialect-mir (SSA) ──(annotated unroll)──▶ dialect-mir
  ──(DialectConversion)──▶ LLVM dialect + dialect-nvvm ops ──(export)──▶ LLVM IR ──(llc)──▶ PTX
```

Each arrow is a well-defined transformation over pliron or LLVM IR. The last
one is LLVM's NVPTX backend doing what it does best.

---

## dialect-mir -- The Rust Layer

`dialect-mir` preserves Rust's type system and control flow semantics as
pliron operations. This is deliberate: we want to reason about Rust concepts
(tuples, enums, checked arithmetic, address spaces) *before* flattening them
to LLVM's type system.

### Types

The dialect defines nine custom types that mirror Rust's own:

| Type                 | Example                                                   | Description                                                   |
| :------------------- | :-------------------------------------------------------- | :------------------------------------------------------------ |
| `mir.tuple`          | `mir.tuple<i32, f32, i64>`                                | Heterogeneous tuples                                          |
| `mir.ptr`            | `mir.ptr<f32, mutable, addrspace: 1>`                     | Pointers with GPU address space                               |
| `mir.array`          | `mir.array<f32, 256>`                                     | Fixed-size arrays                                             |
| `mir.struct`         | `mir.struct<"Point", [f32, f32]>`                         | Named structs with layout info                                |
| `mir.union`          | `mir.union<"Repr", [a, b], [i32, f32], 4, 4>`             | Rust unions -- every field is a view of the same bytes        |
| `mir.slice`          | `mir.slice<f32, addrspace: 1>`                            | Fat pointers (ptr + length)                                   |
| `mir.disjoint_slice` | `mir.disjoint_slice<f32>`                                 | Bounds-checked slice carrying a typed index space              |
| `mir.enum`           | `mir.enum<"Option_i32", [("None", []), ("Some", [i32])]>` | Rust enums with discriminant and variant payloads             |
| `mir.fp16`           | `mir.fp16`                                                | IEEE 754 binary16, Rust's `f16`                               |

The address spaces on `mir.ptr` and `mir.slice` track where data lives in
the GPU memory hierarchy:

| Address Space | Meaning                                  |
| :------------ | :--------------------------------------- |
| 0             | Generic (resolved at runtime)            |
| 1             | Global (device DRAM)                     |
| 3             | Shared (per-block SRAM)                  |
| 4             | Constant (read-only cache)               |
| 5             | Local (per-thread stack, spills to DRAM) |
| 6             | Tensor memory (Blackwell TMEM)           |

### Operations

`dialect-mir` defines 62 operations across 12 categories, one per module under
`crates/dialect-mir/src/ops/`:

| Category     | Examples                                                                                                                                                                                            | Count |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----: |
| Function     | `mir.func`                                                                                                                                                                                          |     1 |
| Control flow | `mir.goto`, `mir.cond_br`, `mir.return`, `mir.assert`, `mir.unreachable`, `mir.unroll_hint`                                                                                                         |     6 |
| Constants    | `mir.constant`, `mir.float_constant`, `mir.undef`                                                                                                                                                   |     3 |
| Memory       | `mir.alloca`, `mir.load`, `mir.store`, `mir.assign`, `mir.ref`, `mir.ptr_offset`, `mir.memcpy`, `mir.memmove`, `mir.shared_alloc`, `mir.global_alloc`, `mir.extern_shared`                          |    11 |
| Arithmetic   | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`, `mir.rem`, `mir.checked_add`, `mir.checked_sub`, `mir.checked_mul`, `mir.neg`, `mir.not`, `mir.shr`, `mir.shl`, `mir.bitand`, `mir.bitor`, `mir.bitxor` |    15 |
| Comparison   | `mir.eq`, `mir.ne`, `mir.lt`, `mir.le`, `mir.gt`, `mir.ge`, `mir.cmp`                                                                                                                               |     7 |
| Aggregate    | `mir.extract_field`, `mir.insert_field`, `mir.construct_struct`, `mir.construct_tuple`, `mir.construct_array`, `mir.construct_slice`, `mir.construct_disjoint_slice`, `mir.extract_array_element`, `mir.field_addr`, `mir.array_element_addr` |    10 |
| Enum         | `mir.get_discriminant`, `mir.set_discriminant`, `mir.construct_enum`, `mir.enum_payload`                                                                                                            |     4 |
| Cast         | `mir.cast`                                                                                                                                                                                          |     1 |
| Storage      | `mir.storage_live`, `mir.storage_dead`                                                                                                                                                              |     2 |
| Call         | `mir.call`                                                                                                                                                                                          |     1 |
| Debug        | `mir.dbg_value`                                                                                                                                                                                     |     1 |

That is a lot of operations, but they fall into natural groups. If you know
Rust MIR (or have read the [rustc_public chapter](rustc-public.md)), each
operation maps directly to a MIR concept.

### What the IR Looks Like

Here are a few examples of `dialect-mir` operations in practice. These are
simplified for readability -- the actual printed form includes more metadata.

**Checked addition** (Rust: `let sum = a + b` where `a, b: i32`):

```text
// mir.checked_add returns a tuple (result, overflow_flag)
%checked = mir.checked_add %a, %b : i32
%sum     = mir.extract_field %checked, 0 : mir.tuple<i32, i1>
%overflowed = mir.extract_field %checked, 1 : mir.tuple<i32, i1>
%ok = mir.not %overflowed : i1
mir.assert %ok
mir.goto bb1
```

`mir.assert` continues within its block when the condition is true and traps
otherwise. It stays separate from the branch so merging blocks cannot erase
the check. LLVM lowering splits the block at the assertion and adds an
explicit branch to either the following operations or a trap block.

**Struct construction and field access** (Rust: `point.x`):

```text
%point = mir.construct_struct %x, %y : mir.struct<"Point", [f32, f32]>
%x_val = mir.extract_field %point, 0 : mir.struct<"Point", [f32, f32]>
```

**Shared memory allocation** (the GPU-specific part):

```text
%shmem = mir.shared_alloc : mir.ptr<f32, mutable, addrspace: 3>
mir.store %value, %shmem : f32
```

### Verification

Every MIR operation verifies type consistency when constructed. This catches
import bugs early -- before they have a chance to propagate through the
lowering pipeline and surface as cryptic LLVM errors three passes later.

Examples of what gets checked:

- `mir.add` verifies that both operands have the same type.
- `mir.cond_br` verifies that the condition is `i1` (a boolean).
- `mir.extract_field` verifies that the field index is in bounds and the
  result type matches the field's type.
- `mir.store` verifies that the value type matches the pointee type of the
  pointer.

`DisjointSlice` accepts only a matching `ThreadIndex`, so the type system checks
the device-side index space. Host launch geometry completes the uniqueness
proof through `PreparedLaunch<K>` or an unsafe raw-launch obligation. There is
no separate disjoint-access compiler pass; the safety comes from these APIs.

---

## The LLVM Dialect -- The LLVM Layer

The LLVM dialect models LLVM IR as pliron operations. It provides a near-1:1
mapping to textual `.ll` files -- every LLVM instruction has a corresponding
pliron operation, and the types map directly to LLVM's type system. The
dialect itself (ops, types, attributes, op-interfaces) is defined upstream in
the `pliron-llvm` crate; cuda-oxide consumes it and re-exports it through the
thin `llvm-export` crate, which also carries the textual `.ll` exporter and a
few GPU-specific extensions (named address spaces, fp16 bit helpers) that
pliron-llvm does not ship.

### Types

| Type          | Example                                 | Description                                          |
| :------------ | :-------------------------------------- | :--------------------------------------------------- |
| Integers      | `i1`, `i8`, `i16`, `i32`, `i64`, `i128` | Pliron built-in, used directly                       |
| Floats        | `half`, `float`, `double`               | Pliron built-in (`FP16Type`, `FP32Type`, `FP64Type`) |
| `llvm.ptr`    | `ptr addrspace(1)`                      | Opaque pointers with optional address space          |
| `llvm.struct` | `{ i32, float }` or `%MyStruct`         | Named or anonymous, may be opaque                    |
| `llvm.array`  | `[256 x float]`                         | Fixed-size arrays                                    |
| `llvm.vector` | `<4 x float>`                           | SIMD vectors                                         |
| `llvm.func`   | `(i32, ptr) -> void`                    | Function signatures                                  |
| `llvm.void`   | `void`                                  | The unit type                                        |

Note the absence of Rust-specific types. By the time code reaches the
LLVM dialect, tuples have become structs, enums have become
discriminant-indexed structs, and slices have become pointer-length pairs.
The lowering pass (covered in [The Lowering Pipeline](lowering-pipeline.md))
handles all of that flattening.

### Operations

At the pinned `pliron` revision the dialect defines 69 operations:

| Category     | Examples                                                                                                                                | Count |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------- | ----: |
| Arithmetic   | `add`, `sub`, `mul`, `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`, ...                                                                |    19 |
| Cast         | `zext`, `sext`, `trunc`, `fpext`, `fptrunc`, `sitofp`, `uitofp`, `fptosi`, `fptoui`, `ptrtoint`, `inttoptr`, `addrspacecast`, `bitcast` |    13 |
| Control flow | `br`, `cond_br`, `switch`, `return`, `unreachable`, `indirectbr`                                                                        |     6 |
| Memory       | `load`, `store`, `alloca`, `gep`                                                                                                        |     4 |
| Atomic       | `atomic_load`, `atomic_store`, `atomicrmw`, `cmpxchg`, `fence`                                                                          |     5 |
| Comparison   | `icmp`, `fcmp`                                                                                                                          |     2 |
| Aggregate    | `extract_value`, `insert_value`, `extract_element`, `insert_element`, `shuffle_vector`                                                  |     5 |
| Call         | `call`, `call_intrinsic`                                                                                                                |     2 |
| Inline asm   | `inline_asm`                                                                                                                            |     1 |
| Constants    | `constant`, `zero`, `undef`, `poison`                                                                                                   |     4 |
| Symbol       | `func`, `global`, `addressof`                                                                                                           |     3 |
| Select       | `select`                                                                                                                                |     1 |
| Other        | `freeze`, `va_arg`, `blockaddress`, `blocktag`                                                                                          |     4 |

`llvm-export` adds one operation of its own on top of those, `llvm.dbg_value`,
alongside the address-space and fp16 helpers mentioned above.

Because the dialect is upstream, this table moves when the `pliron` pin moves
rather than when this repository changes; `pliron-llvm`'s `src/ops.rs` is the
list it is counting.

If you have read LLVM IR before, nothing here will surprise you. The operation
names are intentionally the same as their LLVM counterparts, prefixed with
`llvm.` in the IR.

### The Export Engine

The crown jewel of `llvm-export` is its export module
(`crates/llvm-export/src/export/`) -- the code that converts a pliron IR
module into valid textual LLVM IR. This is the part cuda-oxide keeps local:
pliron-llvm only emits real `.ll` via an `llvm-sys` bridge, which cuda-oxide
avoids. This is not just "print each operation"; several non-trivial
transformations happen during export:

**Block arguments become PHI nodes.** Pliron IR (MLIR-like) models merge points
as block arguments -- a function-style calling convention between basic blocks.
LLVM IR uses PHI nodes instead. The exporter builds a predecessor map from
branch operands and emits `phi` instructions at the top of each non-entry
block.

**Value naming.** A pre-pass assigns sequential SSA names (`%v0`, `%v1`, ...)
to every value. Constants are special-cased: `llvm.constant` results are
mapped to their literal value (not a `%vN` name), so PHIs can reference
constants from blocks that appear later in the output.

**NVVM intrinsic name conversion.** Pliron identifiers use underscores; LLVM
intrinsics use dots. The exporter converts all names starting with `llvm_` by
replacing underscores with dots: `llvm_nvvm_read_ptx_sreg_tid_x` becomes
`llvm.nvvm.read.ptx.sreg.tid.x`. This is a mechanical transformation, not a
lookup table.

**Convergent attribute marking.** Barrier, shuffle, and vote intrinsics must
be marked `convergent` to prevent LLVM from hoisting them out of control flow.
The exporter recognizes these by prefix pattern matching on the (dot-form)
name and appends `#0` to their call sites, emitting `attributes #0 = {
convergent }` at module level.

**Kernel metadata.** Functions marked as kernels get `ptx_kernel` calling
convention and an `!nvvm.annotations` metadata entry.

Here is what the exported LLVM IR looks like for a simple vector-add kernel:

```llvm
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define ptx_kernel void @vecadd(ptr addrspace(1) %v0, i64 %v1,
                                ptr addrspace(1) %v2, i64 %v3,
                                ptr addrspace(1) %v4, i64 %v5) {
entry:
    %v6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
    %v7 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
    %v8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
    %v9 = mul i32 %v8, %v7
    %v10 = add i32 %v9, %v6
    ; ... bounds check, load, add, store ...
    ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @vecadd, !"kernel", i32 1}
attributes #0 = { convergent }
```

Notice the slices have been scalarized: each Rust `&[f32]` becomes a
`ptr addrspace(1)` and an `i64` length. That happened in the lowering pass;
by the time the LLVM dialect sees them, they are flat arguments.

---

## dialect-nvvm -- The GPU Layer

`dialect-nvvm` wraps NVIDIA's GPU intrinsics as typed pliron operations.
These operations do not form a "level" in the lowering chain -- they are
inserted during the `dialect-mir` → LLVM dialect lowering pass and coexist
with LLVM dialect operations in the same function body. At export time,
they become `call` instructions to `@llvm.nvvm.*` intrinsics.

### Architecture Coverage

At catalog SHA-256 `07180a97` (the stamp in every `ops/generated/` file
header), the dialect holds 581 operations across 42 modules, and they come
from two different places. The split is the first thing to know about it,
because it decides where -- and whether -- you would add one. If the header
stamp no longer starts with `07180a97`, the counts on this page predate the
catalog you are reading.

**Hand-written**, directly under `crates/dialect-nvvm/src/ops/`. These are the
ops with bespoke verification or lowering that the intrinsic catalog does not
describe. There are seven modules and 26 operations:

| Module    | Description                                                 | Ops |
| :-------- | :---------------------------------------------------------- | --: |
| `asm`     | `inline_ptx`                                                |   1 |
| `atomic`  | Atomic load/store/RMW/cmpxchg/fence                         |   5 |
| `cluster` | Cluster index and cluster-count registers                   |   2 |
| `debug`   | `assertfail`, `vprintf`                                     |   2 |
| `grid`    | Cooperative `grid_sync`                                     |   1 |
| `memory`  | Generic-to-shared address conversion with a byte offset     |   1 |
| `wgmma`   | Warpgroup MMA descriptors; bf16/f16 at m64n64k16, bf16 at m64n128k16, tf32 at m64n64k8 |  14 |

**Generated**, under `ops/generated/`, from `intrinsics/catalog.json` by
`cuda-intrinsics-gen`. Every file there opens with `// @generated ... DO NOT
EDIT.`, and editing one by hand is undone by the next generator run. This is
the large majority -- 35 modules and 555 operations, resolved from 1034 catalog
entries, since several intrinsics can share one structural op:

| Area                        | Modules                                                                       | Ops |
| :-------------------------- | :---------------------------------------------------------------------------- | --: |
| Tensor Core Gen 5 + TMEM    | `tcgen05`                                                                     | 210 |
| Tensor Memory Accelerator   | `tma`                                                                         | 116 |
| Special registers           | `sreg`                                                                        |  44 |
| Packed (SIMD-in-register)   | `packed_alu`, `packed_conversion`, `packed_atomic`                            |  51 |
| Async copy and barriers     | `cp_async`, `mbarrier_extended`, `mbarrier_basic`, `sync`                     |  35 |
| Warp-level                  | `warp_shuffle`, `redux`, `vote`, `warp_match`, `warp_barrier`, `active_mask`, `elect` |  39 |
| Matrix fragment movement    | `ldmatrix`, `register_mma`, `stmatrix`, `wgmma_control`, `movmatrix`, `sparse_mma` |  22 |
| Execution and debug control | `execution_control`, `debug_control`                                          |  11 |
| Cluster                     | `clc`, `cluster_barrier`, `cluster_memory`                                    |  10 |
| Scalar math                 | `dotprod`, `scalar_arithmetic`, `scalar_conversion`, `scalar_math`, `extended_minmax`, `prmt` |   9 |
| Integer min/max (DPX)       | `integer_minmax`                                                              |   8 |

Architecture requirements live per intrinsic rather than per module -- the
catalog records the PTX version and minimum SM for each, and
`intrinsics/generated-reference.md` renders them alongside the PTX each one is
expected to emit. That file is regenerated with the ops, so it is the list to
consult rather than a count kept by hand here.

Most users will only encounter special registers, warp shuffles and barriers.
The rest are for advanced GPU programming -- TMA, matrix accelerators, and
Blackwell's tensor memory -- covered in the
[Advanced GPU Features](../advanced/tensor-memory-accelerator.md) chapters. If
you are adding an op, read
[Adding New Intrinsics](adding-new-intrinsics.md) first: it walks the catalog
path, which is the one nearly every new intrinsic takes.

### From Rust to PTX: An Intrinsic's Journey

Each NVVM operation maps through three levels of naming:

| Pliron operation               | LLVM intrinsic                                        | PTX instruction                        |
| :----------------------------- | :---------------------------------------------------- | :------------------------------------- |
| `ReadPtxSregTidXOp`            | `llvm.nvvm.read.ptx.sreg.tid.x`                       | `mov.u32 %r1, %tid.x`                  |
| `Barrier0Op`                   | `llvm.nvvm.barrier0`                                  | `bar.sync 0`                           |
| `ShflSyncBflyI32Op`            | `llvm.nvvm.shfl.sync.bfly.i32`                        | `shfl.sync.bfly.b32`                   |
| `CpAsyncBulkTensorG2sTile2dOp` | `llvm.nvvm.cp.async.bulk.tensor.2d.tile.g2s.im2col.*` | `cp.async.bulk.tensor.2d.tile.g2s ...` |

The first column is the Rust struct name in `dialect-nvvm`. The second is what
`llvm-export` emits (after the underscore-to-dot transformation). The third is
what `llc` produces. You never have to write any of these by hand:
`mir-importer` emits the operation when it translates a call to a
`cuda-device` intrinsic such as `thread::threadIdx_x()` or
`warp::shuffle_xor_sync()`, and the later stages derive the other two forms
from it.

### Verification Strategy

NVVM operations use minimal structural verification: each operation checks
its operand count and result count, and a handful verify result types (thread
indexing ops require `i32` results; `tcgen05` loads check exact result counts
for their 32-register and 4-register variants).

This is intentional. NVVM operations are machine-generated by `mir-importer` --
they are never hand-written by users. LLVM's NVPTX backend provides
comprehensive type validation downstream. Adding full type checking to every
NVVM operation would double the dialect's code size for zero practical benefit.

```{note}
The GPU architecture requirements (sm_70, sm_90, sm_100) are documented but
not enforced at the pliron level. Architecture validation happens later, when
`llc` is invoked with a specific `-mcpu=sm_XX` flag. If you use a Hopper
intrinsic and target Volta, `llc` will tell you -- loudly.
```

---

## How the Dialects Interact

Here is the lifecycle of a single Rust operation as it passes through all
three abstraction levels:

```text
Rust source:   let sum = a + b;        // a, b: f32

dialect-mir:   %sum = mir.add %a, %b : f32
                ↓  (DialectConversion)
LLVM dialect:  %v5 = fadd float %v3, %v4
                ↓  (llvm-export)
LLVM IR:       %v5 = fadd float %v3, %v4
                ↓  (llc --mcpu=sm_80)
PTX:           add.f32 %f3, %f1, %f2;
```

The `dialect-mir` → LLVM dialect step is where the interesting work
happens: `mir.add` on `f32` becomes `fadd` (floating-point add), while
`mir.add` on `i32` becomes `add` (integer add). Checked operations like
`mir.checked_add` expand into an `llvm.add`, a constant `i1 false` for the
overflow flag, and an `insertvalue` into a struct -- the GPU path omits
overflow detection (since GPU integer arithmetic wraps). The lowering pass
handles all of these translations.

For GPU-specific operations, `dialect-nvvm` enters the picture:

```text
Rust source:   let tid = thread::threadIdx_x();
                ↓  (mir-importer matches "cuda_device::thread::threadIdx_x")
dialect-nvvm:  %v2 = nvvm.read_ptx_sreg_tid_x : i32
                ↓  (mir-lower, then llvm-export)
LLVM IR:       %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
                ↓  (llc)
PTX:           mov.u32 %r1, %tid.x;
```

The translator in `mir-importer` recognizes calls to `cuda_device` intrinsic
functions by their fully qualified names (FQDNs); the arm that matches
`threadIdx_x` is generated from `intrinsics/catalog.json` into its dispatch.
The call never becomes a `mir.call`: the translator emits the `dialect-nvvm`
operation on the spot, and the intrinsic ends up a direct hardware
instruction.

### The Full Picture

Putting it all together, a compiled kernel body contains a mix of
LLVM dialect and `dialect-nvvm` operations:

```text
llvm.func @vecadd(...) {
  entry:
    %tid    = nvvm.read_ptx_sreg_tid_x     // NVVM: thread index
    %ntid   = nvvm.read_ptx_sreg_ntid_x    // NVVM: block size
    %ctaid  = nvvm.read_ptx_sreg_ctaid_x   // NVVM: block index
    %offset = llvm.mul %ctaid, %ntid        // LLVM: integer math
    %idx    = llvm.add %offset, %tid        // LLVM: integer math
    %cmp    = llvm.icmp slt %idx, %len      // LLVM: bounds check
    llvm.cond_br %cmp, bb1, bb2             // LLVM: branch
  bb1:
    %p_a    = llvm.gep %a, %idx             // LLVM: pointer arithmetic
    %val_a  = llvm.load %p_a                // LLVM: memory access
    %p_b    = llvm.gep %b, %idx
    %val_b  = llvm.load %p_b
    %sum    = llvm.fadd %val_a, %val_b      // LLVM: floating-point add
    %p_c    = llvm.gep %c, %idx
    llvm.store %sum, %p_c
    llvm.br bb2
  bb2:
    llvm.return void
}
```

The `dialect-nvvm` operations at the top compute the global thread index.
Everything else is standard LLVM dialect -- loads, stores, arithmetic,
branches. The export engine serializes all of it into a single `.ll` file,
and `llc` compiles it to PTX.

---

For how these dialects are connected by the lowering pass, see
[The Lowering Pipeline](lowering-pipeline.md).
