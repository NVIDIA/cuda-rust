# wgmma

## WGMMA (Warpgroup MMA) - Hopper (sm_90) Tensor Core Operations

Demonstrates WGMMA infrastructure for Hopper's tensor cores. WGMMA enables warpgroup-wide (128 threads) matrix operations.

## What This Example Does

Tests WGMMA synchronization primitives:
- `wgmma_fence()` - Ensure prior memory operations complete
- `wgmma_commit_group()` - Commit current instruction group
- `wgmma_wait_group::<0>()` - Wait for all groups to complete
- `make_smem_desc()` - Create SMEM descriptor with swizzle

## Key Concepts Demonstrated

### WGMMA Sync Primitives

```rust
#[kernel]
pub unsafe fn wgmma_sync_test(mut output: DisjointSlice<u64>) {
    // Shared memory for matrix tile (128-byte aligned)
    static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x();
    let gid = thread::index_1d();

    // Create SMEM descriptor with swizzle encoding
    let desc = make_smem_desc(&raw const SMEM as *const u8);

    // WGMMA sync sequence (this test validates fence/commit/wait only)
    wgmma_fence();           // Ensure prior ops complete
    wgmma_commit_group();    // Commit instruction group
    wgmma_wait_group::<0>(); // Wait for all groups

    // Write descriptor to output
    if tid == 0 {
        if let Some(output_elem) = output.get_mut(gid) {
            *output_elem = desc;
        }
    }
}
```

### SMEM Descriptor Format

```rust
// make_smem_desc creates a 64-bit descriptor:
// Bits 0-13:   Base address >> 4
// Bits 16-29:  Leading dimension offset >> 4
// Bits 32-45:  Stride offset >> 4
// Bits 62-63:  Swizzle mode (3 = 128B swizzle)

let desc = make_smem_desc(smem_ptr);
// desc encodes the memory layout for WGMMA hardware
```

## Build and Run

```bash
cargo oxide run wgmma
```

## Expected Output

### On Hopper (sm_90):

```text
=== Unified WGMMA Example ===

GPU Compute Capability: sm_90

Loading PTX from: wgmma.ptx
✓ PTX loaded successfully

--- Test: WGMMA Sync Primitives ---

Launching wgmma_sync_test kernel...
SMEM descriptor: 0xC00000080008xxxx
✓ Swizzle mode correct (128B)
  Leading dimension offset: 8 (raw bits)
  Stride offset: 8 (raw bits)

=== WGMMA Test Complete ===
```

`xxxx` is the descriptor's variable 14-bit shared-memory address field. The
fixed `0008` fields decode to raw leading-dimension and stride offsets of `8`.

### On Pre-Hopper or Blackwell:

```text
GPU Compute Capability: sm_86

⚠️  WARNING: WGMMA requires sm_90 (Hopper) or newer!
   Your GPU is sm_86

ℹ️  PTX load failed (expected on non-Hopper): ...
```

```text
GPU Compute Capability: sm_120

⚠️  WGMMA is Hopper-only (sm_90).
   Your GPU is sm_120 (Blackwell).
   WGMMA instructions don't exist on this architecture.

   To test WGMMA, use a Hopper GPU (H100, H200).
```

## Hardware Requirements

- **Required GPU**: Hopper H100, H200 (sm_90 only)
- **NOT supported**: Ada Lovelace (sm_89), Blackwell (sm_100/sm_120)
- **CUDA Driver**: 12.0+

## WGMMA vs tcgen05

| Feature      | WGMMA (Hopper)        | tcgen05 (Blackwell)  |
|--------------|-----------------------|----------------------|
| Architecture | sm_90                 | sm_100/sm_120        |
| Warpgroup    | 4 warps (128 threads) | 1 CTA group          |
| Accumulator  | Register file         | TMEM (separate)      |
| Max shape    | 64×256×16             | 128×256×K            |
| Descriptor   | SMEM-based            | SMEM + TMEM          |

## WGMMA Instruction Sequence

```rust
// 1. Fence before loading data
wgmma_fence();

// 2. Load matrix tiles into SMEM
// ... TMA or manual loads ...

// 3. Fence after loads
wgmma_fence();

// 4. Issue WGMMA instruction
// wgmma_mma_f16_f16_f32(acc, a_desc, b_desc);  // Not in this example

// 5. Commit and wait
wgmma_commit_group();
wgmma_wait_group::<0>();  // Wait for all groups

// 6. Accumulator results now ready in registers
```

## WGMMA Functions

| Function                 | PTX                               | Description       |
|--------------------------|-----------------------------------|-------------------|
| `wgmma_fence()`          | `wgmma.fence.sync.aligned`        | Memory fence      |
| `wgmma_commit_group()`   | `wgmma.commit_group.sync.aligned` | Commit group      |
| `wgmma_wait_group::<N>()`| `wgmma.wait_group.sync.aligned N` | Wait for N groups |
| `make_smem_desc(ptr)`    | Computed                          | Create descriptor |

## Generated PTX

```ptx
// WGMMA sync instructions
wgmma.fence.sync.aligned;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;

// SMEM descriptor encoding (computed on device)
// Uses bit manipulation to create 64-bit descriptor
shr.u64 %rd1, %rd_addr, 4;     // addr >> 4
and.b64 %rd2, %rd1, 0x3FFF;    // Mask to 14 bits
// ... combine with swizzle bits ...
```

## Why This Example is Limited

This example only tests the **synchronization primitives** and **descriptor creation**, not actual WGMMA matrix operations. Full WGMMA usage requires:

1. **Matrix layout understanding**: Row/column major, swizzle patterns
2. **Warpgroup coordination**: 128 threads working together
3. **Accumulator management**: 8-16 register accumulators per thread
4. **Complex descriptor setup**: A and B matrices have different formats

For production GEMM on Hopper, use cuBLAS or CUTLASS which handle these complexities.

## Potential Errors

| Error                      | Cause                   | Solution               |
|----------------------------|-------------------------|------------------------|
| `CUDA_ERROR_INVALID_PTX`   | Running on non-Hopper   | Use sm_90 GPU          |
| Wrong swizzle bits         | Descriptor encoding bug | Check bit positions    |
| WGMMA hang                 | Missing fence/commit    | Follow sync sequence   |
| Wrong results              | Incorrect descriptor    | Verify layout encoding |
