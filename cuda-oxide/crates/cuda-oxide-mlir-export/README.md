# CUDA Oxide MLIR export

This crate maps CUDA Oxide operations to MLIR accepted by the pinned CUTLASS
compiler. It uses `pliron-mlir-export` to build and render the MLIR text:

```text
pliron-mlir-export                 this crate
------------------                ----------
syntax tree + renderer      +     builtin / MIR / NVVM / CuTe mappings
registry + diagnostics            pinned consumer profiles
```

The generic crate can translate any Pliron dialect. CUDA-specific mappings
and compiler-version requirements live here.

The profile targets the official CUTLASS 4.7 compiler library. CUDA
Oxide installs the exact NVIDIA release archive, verifies both the archive and
library digests, and loads that versioned library explicitly:

```text
shared high-level Pliron module
                │
                ▼
    shared CuTe safety checks
                │
                ▼
 builtin + MIR + NVVM + CuTe recipes
                │
                ▼
      generic MLIR renderer
                │
                ▼
 official CUTLASS 4.7 compiler
```

The mapping packs translate CuTe operations after shared MIR preparation.
The pinned library compiles the resulting MLIR, and the backend extracts and
validates its cubin. CUDA Oxide embeds and launches that cubin through its
ordinary artifact loader. Set `CUDA_OXIDE_MLIR_OUTPUT=<file>` to keep the MLIR
text for inspection; select `CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir` to compile it.

The SM100 primitive pack additionally maps two-CTA FP16 TCGen05 MMA with A
collector selectors, tensor-memory allocation/load/commit/wait operations,
cluster synchronization, elected-lane results and multicast TMA. The
[`fp16_gemm_256x352_cute`](../../cuteir/examples/fp16_gemm_256x352_cute)
example exercises these together. It also preserves fixed cluster dimensions
in `nvvm.cluster_dim` and the scalar order of compiler-created register arrays.
Unsupported primitive selectors and intrinsic identities fail at export.

## Scalar code and control flow

These mappings handle the ordinary scalar code around a CuTe kernel:

```text
mir.func / mir.return        → func.func / func.return
mir.goto / mir.cond_br       → cf.br / cf.cond_br
mir.assert                  → cf.assert
mir.constant                 → arith.constant
mir.add / sub / mul          → arith.add* / sub* / mul*
mir.div / rem                → signed or unsigned arith operation
mir.lt / le / gt / ge        → arith.cmpi or arith.cmpf
mir.eq / ne                  → arith.cmpi or arith.cmpf
mir.cast                     → the matching arith conversion
```

Rust's `usize` is a 64-bit unsigned integer on the CUDA target. MLIR integer
types do not write signedness in the type, so it becomes `i64`:

```text
Pliron                           MLIR
builtin.integer ui64            i64
mir.div on that ui64            arith.divui
mir.lt  on that ui64            arith.cmpi ult
```

The operation name keeps the part that matters. A signed `i64` division uses
`arith.divsi`; an unsigned `u64` division uses `arith.divui`. The translator
looks at the Pliron type before both types become MLIR `i64`.

Control flow maps to `cf`, not `scf`. Rust MIR is already a graph of basic
blocks and can jump in shapes that are not a neat `if` or `for` loop:

```text
        mir.cond_br
          /      \
    block A     block B
          \      /
            join

                ↓

         cf.cond_br
          /      \
    block A     block B
          \      /
            join
```

Later MLIR passes may rebuild structured loops. The exporter does not guess
that structure while translating the graph.

## Pointers and small Rust values

The elementwise kernel still contains ordinary Rust values around its CuTe
views. They keep their physical shape when they cross into MLIR:

```text
mir.ptr<T>                 → !llvm.ptr
mir.slice<T>               → !llvm.struct<(!llvm.ptr, i64)>
mir.array<T, 4>            → !llvm.array<4 x T>
mir.struct<A, B>           → !llvm.struct<(A, B)>

mir.alloca / load / store  → llvm.alloca / load / store
mir.field_addr             → llvm.getelementptr
mir.extract_field          → llvm.extractvalue
```

Rust can reorder fields and insert padding. The translator reads the MIR
layout and rebuilds that exact physical order. It refuses packed or otherwise
different by-value layouts instead of silently changing an address.

For `#[grid_constant] value: &T`, the host passes `T` by value. Every thread
borrows the same read-only kernel parameter storage:

```text
host T -> one by-value launch argument -> device &T shared by the grid
```

The exporter records the value's size with `llvm.byval`, its Rust alignment
with `llvm.align`, and shared read-only storage with `nvvm.grid_constant`.
For a TMA descriptor this means 128 bytes aligned to 64 bytes. Slices expand
into pointer and length arguments, so these attributes follow the descriptor
to its new argument index.

The backend checks the cubin's parameter sizes and offsets before saving it.
Missing or inconsistent metadata fails compilation. The caller still owns
the [pointer validity and lifetime requirements](../../cuda-oxide-book/gpu-programming/kernels-and-device-functions.md#grid-constant-parameters)
for data accessed through the value.

The CUDA kernel marker also has an exact mapping:

```text
gpu_kernel = "true"        → cute.kernel
anything else              → translation error
```

## Tensor operations

The elementwise path keeps tensor layouts visible until CUTLASS lowers them:

```text
make tensor
    │
    ▼
divide it into tiles
    │
    ▼
pick this thread's tile
    │
    ├── full tile ──► cute.copy
    │
    └── edge tile ──► checked scalar load/store
```

The profile maps `cute.tensor_*` operations to CUTLASS's `cute` and
`cute_nvgpu` operations. First, the shared verifier checks tensor origins,
pipeline state, and operation ordering across the module. The verified CuTe
operations then go directly to MLIR translation.

Thread, block, and grid coordinates are direct NVVM mappings. For example:

```text
nvvm.read_ptx_sreg_tid_x    → nvvm.read.ptx.sreg.tid.x
nvvm.read_ptx_sreg_ntid_y   → nvvm.read.ptx.sreg.ntid.y
nvvm.read_ptx_sreg_nctaid_z → nvvm.read.ptx.sreg.nctaid.z
```

Only the reviewed coordinate reads are registered. Other NVVM operations fail
with a missing-mapping error until they get their own recipe.
