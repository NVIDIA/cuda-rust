# Matrix Multiply Accelerators

Modern NVIDIA GPUs include dedicated hardware for matrix multiply-accumulate
(MMA) operations — commonly called **tensor cores**. These units compute
`D = A × B + C` on small matrix tiles at throughputs far beyond what the
standard floating-point ALUs can achieve. A Hopper H100 delivers over 1000
TFLOPS of FP16 MMA, while the same chip's scalar FP16 throughput is roughly
200 TFLOPS. If your workload involves matrix multiplication — and most
deep learning, HPC, and signal processing workloads do — tensor cores are
where the performance lives.

cuda-oxide provides access to two generations of matrix accelerators:
**WGMMA** on Hopper (SM 90) and **tcgen05** on Blackwell (SM 100). This
chapter covers both, their programming models, and how they connect to the
TMA and barrier machinery from the previous chapters.

:::{seealso}
[CUDA Programming Guide — Warpgroup Level Matrix Operations](https://docs.nvidia.com/cuda/cuda-programming-guide/#warpgroup-level-matrix-operations)
for the hardware specification of WGMMA shapes, element types, and
synchronization requirements.
:::

---

## The big picture

```{figure} images/tensor-core-datapath.svg
:align: center
:width: 100%

Data paths for Hopper WGMMA and Blackwell tcgen05. Both read operands from
shared memory via descriptors. WGMMA accumulates into per-thread registers
(warpgroup-collective). tcgen05 accumulates into dedicated Tensor Memory
(TMEM), issued by a single thread.
```

The evolution across generations follows a clear trend: operands move closer
to the compute units, the issuing scope widens, and the programmer writes
fewer instructions for larger tiles. cuda-oxide tracks this evolution with
generation-specific APIs rather than a one-size-fits-all abstraction.

---

## WGMMA — Hopper (SM 90)

WGMMA (**W**arp**G**roup **M**atrix **M**ultiply-**A**ccumulate) is a
**warpgroup-collective** operation: 4 warps (128 threads) cooperate to
compute a matrix tile. Operands A and B are read from shared memory via
SMEM descriptors, and the result accumulates into per-thread registers.

### The programming model

1. **TMA loads** tiles of A and B into shared memory (see
   [Tensor Memory Accelerator](tensor-memory-accelerator.md)).
2. **SMEM descriptors** encode the base address, stride, and swizzle mode
   for each operand.
3. **WGMMA instructions** consume the descriptors and produce the
   accumulator update. The instruction is async — it commits to a barrier.
4. **Barrier wait** ensures the MMA has completed before reading the
   accumulator.

### Hardware-supported shapes

WGMMA always has M=64 (rows), with N and K depending on the element type:

| Element type | K  | N options      |
| :----------- | :- | :------------- |
| f16, bf16    | 16 | 64, 128, 256   |
| tf32         | 8  | 64, 128, 256   |

Each instruction computes a 64×N×K tile. For larger K dimensions, you issue
multiple WGMMA instructions in a loop, accumulating into the same register
tile.

### cuda-oxide API sketch

cuda-oxide exposes WGMMA through low-level intrinsics that map directly to
the hardware instructions. A typical usage pattern:

```rust
use cuda_device::wgmma::{
    make_smem_desc, wgmma_commit_group, wgmma_fence, wgmma_mma_m64n64k16_f32_bf16, wgmma_wait_group,
};

// After TMA has loaded A and B tiles into shared memory...

// Build SMEM descriptors for the loaded tiles
let a_desc = unsafe { make_smem_desc(tile_a_ptr as *const u8) };
let b_desc = unsafe { make_smem_desc(tile_b_ptr as *const u8) };

// Accumulator (32 floats per thread = 64×64 tile across the warpgroup)
let mut acc = [[0.0f32; 8]; 4];

unsafe {
    wgmma_fence();
    wgmma_mma_m64n64k16_f32_bf16(&mut acc, a_desc, b_desc);
    wgmma_commit_group();
    wgmma_wait_group::<0>();
}

// `acc` is now safe to observe.
```

The BF16 m64n128 full-drain variant uses the same synchronization contract but
a 64-value per-thread accumulator:

```rust
use cuda_device::wgmma::wgmma_mma_m64n128k16_f32_bf16;

let mut wide_acc = [[0.0f32; 8]; 8];
unsafe {
    wgmma_fence();
    wgmma_mma_m64n128k16_f32_bf16(&mut wide_acc, a_desc, b_desc);
    wgmma_commit_group();
    wgmma_wait_group::<0>();
}
```

For TF32, use `wgmma_mma_m64n64k8_f32_tf32`. The K=16 TF32
compatibility entry point remains unsupported because Hopper's TF32 WGMMA
hardware shape uses K=8.

### Current cuda-oxide WGMMA lowering

The compiler supports `m64n64k16.f32.bf16.bf16` across three conservative
lowering shapes, `m64n64k16.f32.f16.f16` for canonical linear full-drain and
counted K-loop regions, `m64n64k8.f32.tf32.tf32` for canonical linear
full-drain regions only, and `m64n128k16.f32.bf16.bf16` for canonical linear
full-drain regions. In every accepted case, the entire asynchronous
accumulator lifetime stays inside one convergent inline-PTX statement so LLVM
cannot insert a spill boundary while a WGMMA group is pending.

**m64n64 linear full drain.** A canonical `[[f32; 8]; 4]` accumulator is loaded
into 32 SSA `f32` values before the WGMMA region and stored once after the final
`wait_group<0>`. BF16, F16, and TF32 use this value-threaded carrier.
Unsupported BF16 full-drain pointer shapes retain the original deferred
pointer-form lowering; F16 and TF32 do not have a pointer-form fallback.

**m64n128 BF16 linear full drain.** A canonical `[[f32; 8]; 8]` accumulator is
carried as 64 tied SSA `f32` values through one or more homogeneous
`m64n128k16.f32.bf16.bf16` instructions, one commit, and a final
`wait_group<0>`. Accumulator element addresses are recomputed after the fused
inline-PTX scope instead of being kept live across it. This shape has no
pointer-form fallback, counted-loop lowering, or partial-wait pipeline lowering.

**Canonical counted K-loop.** For a compile-time counted BF16 or F16 m64n64
loop with one MMA per iteration and affine `u64` descriptor recurrences,
cuda-oxide moves the loop control and descriptor arithmetic into the same
convergent PTX region as the WGMMA instruction. The accumulator is therefore
loaded once before the K-loop and stored once after its final wait rather than
round-tripped every iteration.

**Static partial-wait pipeline.** A BF16 m64n64 `wait_group<N>` with `N > 0`
uses `N + 1` independent accumulator slots. Groups are committed separately and
slots are reused round-robin only after the partial wait has made the oldest
slot safe. For example, two groups can remain overlapped with:

```text
wgmma_fence
mma acc0
commit_group
mma acc1
commit_group
wait_group<1>
wait_group<0>
```

Longer pipelines repeat the slot schedule and partial wait before reusing an
accumulator. A final `wait_group<0>` is mandatory before any accumulator value
escapes the fused region.

Selection is intentionally fail-closed. Dynamic partial waits, unsupported
control flow, malformed accumulator schedules, F16 partial-wait shapes, TF32
counted-loop or partial-wait shapes, m64n128 counted-loop or partial-wait
shapes, F16/TF32/m64n128 non-canonical accumulators, and the legacy K=16 TF32
compatibility entry point are rejected instead of exposing an in-flight
accumulator to LLVM.

:::{tip}
WGMMA is often paired with a **multi-stage pipeline**: while the tensor
cores process tile *k*, TMA loads tile *k+1* into a second shared memory
buffer. The barriers for TMA and MMA are separate, enabling full overlap
of data movement and computation.
:::

---

## tcgen05 — Blackwell (SM 100)

Blackwell introduces a fundamentally different matrix accelerator:
**tcgen05**. The key innovations are:

1. **Single-thread issue.** Instead of a warpgroup-collective instruction,
   one thread issues the MMA. The hardware distributes work internally.
2. **Tensor Memory (TMEM).** A dedicated on-chip memory for accumulators,
   separate from the register file. TMEM is larger and has different
   access characteristics than registers.
3. **Larger tiles.** tcgen05 supports shapes up to 256×256, vs WGMMA's
   64×256.

### TMEM — Tensor Memory

TMEM is a new tier in the memory hierarchy, sitting alongside the register
file but dedicated to matrix accumulation. It must be explicitly allocated
and deallocated:

```rust
use cuda_device::SharedArray;
use cuda_device::tcgen05::{TmemGuard, TmemReady, TmemUninit};

static mut TMEM_ADDR: SharedArray<u32, 1> = SharedArray::UNINIT;

// Allocate TMEM (warp-collective)
let tmem: TmemGuard<TmemReady, 128> = unsafe {
    TmemGuard::<TmemUninit, 128>::from_static(TMEM_ADDR.as_mut_ptr())
        .alloc()
};

// ... use tmem for MMA ...

// Deallocate (warp-collective, required before kernel exit)
unsafe { tmem.dealloc(); }
```

`TmemGuard` uses typestates: `TmemUninit` → `alloc()` → `TmemReady` →
`dealloc()` → `TmemDeallocated`. The type system prevents using TMEM before
allocation or forgetting to deallocate — the latter would leak hardware
resources and fault.

The `N_COLS` const parameter determines the TMEM tile width. Common
configurations:

| `N_COLS` | TMEM per warp       | Use case                      |
| :------- | :------------------ | :---------------------------- |
| 64       | Smallest allocation | Narrow tiles                  |
| 128      | Default             | Standard GEMM tiles           |
| 256      | Maximum             | Wide tiles, higher throughput |

### Instruction and SMEM descriptors

tcgen05 uses two descriptors per MMA instruction:

**Instruction descriptor** — encodes the MMA configuration:

```rust
use cuda_device::tcgen05::{Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape};

let idesc = Tcgen05InstructionDescriptor::builder()
    .shape(Tcgen05MmaShape::M128_N128)
    .a_type(Tcgen05ElementType::F16)
    .b_type(Tcgen05ElementType::F16)
    .build();
```

**SMEM descriptor** — points to the operand tile in shared memory:

```rust
use cuda_device::tcgen05::{Tcgen05SmemDescriptor, Tcgen05SwizzleMode};

let a_desc = Tcgen05SmemDescriptor::for_k_major(
    smem_a_addr,
    m, k,
    2, // bytes per element (f16)
    Tcgen05SwizzleMode::Swizzle128B,
);
```

### Issuing MMA

One thread issues the MMA instruction, then all threads wait on a
barrier:

```rust
use cuda_device::tcgen05::{tcgen05_commit, tcgen05_mma_f16};

if thread::threadIdx_x() == 0 {
    unsafe {
        tcgen05_mma_f16(
            tmem.raw_address(),
            a_desc.raw(),
            b_desc.raw(),
            idesc.raw(),
            true, // enable accumulation into D
        );
        tcgen05_commit(mma_barrier_ptr);
    }
}

// All threads wait for MMA completion
mma_barrier.wait(token);
```

### Epilog — reading results from TMEM

After the MMA loop, the accumulator lives in TMEM. To move it to shared
memory (for a subsequent TMA store to global), you load from TMEM into
registers and then use `stmatrix` to write to shared memory:

```rust
use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::tcgen05::{
    TmemF32x4, stmatrix_m8n8_x2, tcgen05_ld_16x256b_pure, tcgen05_load_wait,
};

unsafe {
    // Load a 16×256-bit slice from TMEM into registers
    let regs: TmemF32x4 = tcgen05_ld_16x256b_pure(tmem.raw_address());
    tcgen05_load_wait();

    // Convert four f32 accumulators into two registers of packed bf16 values.
    let packed0 = cvt_bf16x2_f32(regs[0], regs[1]);
    let packed1 = cvt_bf16x2_f32(regs[2], regs[3]);

    // Store two 8×8 matrices from registers (warp-collective).
    stmatrix_m8n8_x2(smem_ptr, packed0, packed1);
}
```

`stmatrix` is a warp-collective operation — all 32 threads participate,
each contributing its register slice to form the shared memory tile.

### CTA-group-2 (CG2)

Blackwell also supports **CG2 mode**, where two CTAs (from a cluster) issue
MMA instructions that are coordinated as a pair. This doubles the effective
tile width and requires the `_cg2` API variants:

```rust
use cuda_device::tcgen05::{tcgen05_alloc_cg2, tcgen05_mma_f16_cg2};
```

CG2 and CG1 (standard) must not be mixed in the same kernel.

:::{tip}
tcgen05 is only available on **Blackwell datacenter** GPUs (SM 100a).
Consumer Blackwell (SM 120) uses the older `mma.sync` instruction set.
Check the target architecture before reaching for the tcgen05 API.
:::

---

## Choosing the right accelerator

| Feature           | WGMMA (Hopper)          | tcgen05 (Blackwell DC)  | mma.sync (Ampere/consumer) |
| :---------------- | :---------------------- | :---------------------- | :------------------------- |
| Issue model       | Warpgroup (128 threads) | Single thread           | Warp (32 threads)          |
| Accumulator       | Registers               | TMEM (dedicated)        | Registers                  |
| Max tile (M×N)    | 64×256                  | 256×256                 | 16×8                       |
| Async execution   | Yes (commit+barrier)    | Yes (commit+barrier)    | Synchronous                |
| TMA integration   | Native                  | Native + multicast CG2  | Manual loads               |
| Cluster support   | Yes                     | Yes + CG2               | No                         |
| Minimum SM        | 90 (Hopper)             | 100a (Blackwell DC)     | 80 (Ampere)                |

For most users, the choice is determined by the target GPU. If you are
writing a library that targets multiple architectures, you will need
compile-time feature gates or runtime architecture detection to select the
right code path.

---

## The GEMM progression

These accelerators do not exist in isolation. A high-performance GEMM kernel
combines every technique from this chapter and the preceding ones:

1. **TMA** loads tiles from global to shared memory (no thread loads)
2. **Barriers** track TMA completion per stage
3. **MMA instructions** (WGMMA or tcgen05) consume tiles from shared memory
4. **Multi-stage pipeline** overlaps load of tile *k+1* with MMA on tile *k*
5. **Warp specialization** dedicates some warps to loading, others to MMA
6. **Epilog** (tcgen05) loads from TMEM, converts precision, stores via TMA

This is the kernel structure that cuBLAS and CUTLASS use internally. With
cuda-oxide, you can build this same structure in Rust — using `SharedArray`
for tiles, `ManagedBarrier` for synchronization, and the MMA APIs for
compute.

The [`gemm_sol`
example](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_sol)
is the worked-out reference. Its 4-stage `cta_group::2` pipeline reaches
**868 TFLOPS at 4096³ — 57.8 % of `cublasLtMatmul` SoL — on B200 (148 SMs)**.
Absolute throughput scales with SM count on smaller Blackwell datacenter
SKUs (e.g., on an 80-SM variant the same kernel runs at ~204 TFLOPS / ~46 %
SoL — see the per-phase tables in the example's README for both
configurations). The example measures the cublasLt baseline live via
`bench/cublaslt_bench`, so its "% of SoL" column is always relative to the
host GPU's cublasLt peak rather than a fixed B200 number.

:::{seealso}
- [Shared Memory and Synchronization](shared-memory-and-synchronization.md) —
  tile management for MMA operands
- [Tensor Memory Accelerator](tensor-memory-accelerator.md) — feeding tiles
  to the tensor cores
- [Cluster Programming](cluster-programming.md) — CG2 mode and multicast
  for multi-CTA MMA
:::
