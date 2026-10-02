# tcgen05

## tcgen05 - Datacenter Blackwell (sm_100a) 5th Gen Tensor Cores

Tests tcgen05 (Tensor Core Gen 5) infrastructure for Blackwell GPUs. This is the next generation of tensor core instructions, replacing Hopper's WGMMA.

## What This Example Tests

### cta_group::1 (single CTA)

1. **tcgen05_fence_test**: Sync primitives and SMEM descriptor builder
2. **tcgen05_alloc_test**: TMEM (Tensor Memory) allocation/deallocation
3. **tcgen05_commit_test**: Commit with mbarrier integration
4. **tcgen05_mma_minimal**: Full MMA pipeline (alloc → copy → MMA → read → dealloc)

### cta_group::2 (CTA pairs)

5. **tcgen05_alloc_cg2_test**: Cooperative TMEM alloc/dealloc across a 2-CTA cluster
6. **tcgen05_mma_cg2_test**: Cooperative MMA with multicast commit across the pair

CTA pairs place 2 CTAs on adjacent SMs (a TPC). They cooperate on larger MMA
tiles — each SM's tensor core handles its half of the rows. All `tcgen05`
instructions in a kernel must use the same `cta_group` value.

### cta_group::2 key differences

| Aspect                       | cta_group::1        | cta_group::2                                       |
|------------------------------|---------------------|----------------------------------------------------|
| Cluster size                 | Any                 | Must be 2 (one CTA pair)                           |
| Minimum MMA shape            | M64_N64             | M128_N128                                          |
| disable-output-lane vector   | 4 elements          | 8 elements                                         |
| TMEM allocation              | Per-CTA             | Cooperative (both CTAs call alloc)                 |
| MMA issuer                   | Any single thread   | One thread in the pair (rank 0)                    |
| Commit                       | `tcgen05_commit`    | `tcgen05_commit_multicast_cg2` (signals both CTAs) |
| TMEM columns (typical)       | 64                  | 512                                                |

## Key Concepts Demonstrated

### SMEM Descriptor Builder

```rust
let desc = Tcgen05SmemDescriptor::builder()
    .address(smem_addr)
    .leading_dim_bytes(128)
    .stride_bytes(128)
    .swizzle(Tcgen05SwizzleMode::Swizzle32B)
    .build()
    .raw();
```

### TMEM Allocation

```rust
// TMEM is a separate on-chip memory for tensor core accumulators.
// Address 0x0 is a valid TMEM base — don't treat it as failure.
static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

// Warp-synchronous: all 32 threads must execute together
if warp_id == 0 {
    tcgen05_alloc(&raw mut TMEM_ADDR as *mut u32, 64);
    // or for CTA pairs: tcgen05_alloc_cg2(..., 512);
}
thread::sync_threads();

let tmem_addr = *(&raw const TMEM_ADDR as *const u32);

// ... use tmem_addr for MMA operations ...

// Deallocate when done (must match alloc's cta_group and size)
if warp_id == 0 {
    tcgen05_dealloc(tmem_addr, 64);
}
```

### CTA Pair MMA (cta_group::2)

```rust
#[kernel]
#[cluster_launch(2, 1, 1)]  // 2 CTAs = 1 CTA pair
pub unsafe fn cta_pair_mma(mut output: DisjointSlice<u32>) {
    let block_rank = cluster::block_rank();

    // Both CTAs allocate TMEM cooperatively
    if warp_id == 0 {
        tcgen05_alloc_cg2(&raw mut TMEM_ADDR as *mut u32, 512);
    }

    // Only one thread in the pair issues MMA
    if tid == 0 && block_rank == 0 {
        let idesc = Tcgen05InstructionDescriptor::builder()
            .shape(Tcgen05MmaShape::M128_N128)  // must be >= M128 for cta_group::2
            .element_type(Tcgen05ElementType::F16)
            .accumulator_type(Tcgen05AccumulatorType::F32)
            .build().raw();

        tcgen05_mma_f16_cg2(tmem_addr, a_desc, b_desc, idesc, false);
        tcgen05_fence_before_thread_sync();

        // Multicast commit: signal both CTAs' barriers (mask 0b11 = ranks 0 and 1)
        tcgen05_commit_multicast_cg2(&raw mut MBAR as *mut u64, 0b11u16);
    }

    // Both CTAs wait on their local barrier
    mbarrier_try_wait(&raw const MBAR, 0);
}
```

## Build and Run

```bash
cargo oxide run tcgen05
```

## Expected Output

```text
=== Unified tcgen05 Example ===

GPU Compute Capability: sm_100
✓ PTX loaded successfully

--- Test: tcgen05 Fence Primitives ---
SMEM descriptor: 0x8000400800080040
✓ Fence primitives executed successfully

--- Test: tcgen05 TMEM Allocation ---
TMEM address: 0x00000000
✓ TMEM allocation successful! Address: 0x00000000

--- Test: tcgen05 Commit with mbarrier ---
✓ Commit with mbarrier executed successfully

--- Test: tcgen05 MMA Minimal ---
✓ MMA minimal test executed successfully

--- CTA Pair (cta_group::2) Tests ---

--- Test: tcgen05 TMEM Alloc cta_group::2 ---
  CTA rank 0 TMEM addr: 0x00000000
  CTA rank 1 TMEM addr: 0x00000000
✓ CTA pair alloc_cg2 successful (TMEM addr 0 is valid base)

--- Test: tcgen05 MMA cta_group::2 ---
  CTA rank 0 TMEM addr: 0x00000000
  CTA rank 1 TMEM addr: 0x00000000
✓ CTA pair MMA cg2 successful

=== tcgen05 Test Complete ===
```

## Hardware Requirements

- **Required GPU**: Blackwell datacenter B100, B200 or newer (sm_100a)
- **NOT supported**: Hopper (sm_90, uses WGMMA), Ada (sm_89), Consumer Blackwell (sm_120)
- **CUDA Driver**: 13.x (R580+) with Blackwell support

## tcgen05 vs WGMMA (Hopper)

| Feature              | WGMMA (Hopper)       | tcgen05 (Blackwell DC)    |
|----------------------|----------------------|---------------------------|
| Architecture         | sm_90                | sm_100a                   |
| Accumulator storage  | Register file        | TMEM (separate memory)    |
| Accumulator capacity | Limited by registers | Larger (TMEM)             |
| Issue model          | 128-thread warpgroup | Single thread             |
| CTA cooperation      | N/A                  | cta_group::2 (CTA pairs)  |
| Max matrix size      | 64×256×16            | 128×256×K                 |

## TMEM (Tensor Memory)

TMEM is a new on-chip memory in Blackwell specifically for tensor core accumulators:

```text
┌──────────────────────────────────────┐
│            Blackwell SM              │
├──────────────────────────────────────┤
│  Registers  │  SMEM  │    TMEM       │
│  (per-thread)  (shared)  (accumulator)
└──────────────────────────────────────┘

TMEM advantages:
- Larger capacity than registers
- Persistent across MMA operations
- Efficient for multi-stage pipelines
- Shared across CTA pair (cta_group::2) for cooperative MMA
```

## tcgen05 Intrinsics

### cta_group::1

| Function                                     | Description                         |
|----------------------------------------------|-------------------------------------|
| `tcgen05_alloc(ptr, cols)`                   | Allocate TMEM, write address to ptr |
| `tcgen05_dealloc(addr, cols)`                | Free TMEM                           |
| `tcgen05_cp_smem_to_tmem(tmem, smem_desc)`   | Copy SMEM → TMEM                    |
| `tcgen05_ld_16x256b_pure(tmem)`              | Load from TMEM → registers          |
| `tcgen05_load_wait()`                        | Wait for TMEM load                  |
| `tcgen05_mma_ws_f16(...)`                    | Warp-specialized MMA (A from TMEM)  |
| `tcgen05_mma_f16(...)`                       | Standard MMA (A/B from SMEM)        |
| `tcgen05_fence_before/after_thread_sync()`   | Fences                              |
| `tcgen05_commit(mbar)`                       | Commit to barrier                   |
| `tcgen05_commit_shared_cluster(mbar)`        | Commit via shared::cluster          |

### cta_group::2 (CTA pairs)

| Function                                       | Description                             |
|------------------------------------------------|-----------------------------------------|
| `tcgen05_alloc_cg2(ptr, cols)`                 | Cooperative TMEM alloc                  |
| `tcgen05_dealloc_cg2(addr, cols)`              | Cooperative TMEM dealloc                |
| `tcgen05_relinquish_alloc_permit_cg2()`        | Relinquish alloc permit                 |
| `tcgen05_mma_f16_cg2(...)`                     | Cooperative MMA (A/B from SMEM)         |
| `tcgen05_commit_cg2(mbar)`                     | Commit to local barrier                 |
| `tcgen05_commit_shared_cluster_cg2(mbar)`      | Commit via shared::cluster              |
| `tcgen05_commit_multicast_cg2(mbar, mask)`     | Commit + signal multiple CTAs' barriers |
| `tcgen05_cp_smem_to_tmem_cg2(tmem, smem_desc)` | Copy SMEM → TMEM                        |

## Generated PTX

### cta_group::1

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], n_cols;
tcgen05.mma.cta_group::1.kind::f16 [d], a_desc, b_desc, idesc, {0,0,0,0}, pred;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar];
tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, n_cols;
```

### cta_group::2

```ptx
tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], n_cols;
tcgen05.mma.cta_group::2.kind::f16 [d], a_desc, b_desc, idesc, {0,0,0,0,0,0,0,0}, pred;
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [mbar], ctaMask;
tcgen05.dealloc.cta_group::2.sync.aligned.b32 tmem_addr, n_cols;
```

Note the 8-element disable-output-lane vector for `cta_group::2` (vs 4 for `cta_group::1`).

## Potential Errors

| Error                              | Cause                           | Solution                              |
|------------------------------------|---------------------------------|---------------------------------------|
| `CUDA_ERROR_INVALID_PTX`           | Wrong target or bad instruction | Check PTX with `ptxas -arch=sm_100a`  |
| `CUDA_ERROR_ILLEGAL_INSTRUCTION`   | Wrong MMA shape for cta_group   | Use M128_N128+ for cta_group::2       |
| TMEM address = 0xDEADBEEF          | Alloc didn't write              | Check warp_id == 0                    |
| Wrong MMA results                  | Bad descriptor                  | Verify SMEM layout                    |
| Hang after MMA                     | Missing commit/wait             | Follow sync sequence                  |
