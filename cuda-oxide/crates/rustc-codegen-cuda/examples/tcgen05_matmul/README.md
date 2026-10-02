# tcgen05_matmul

## tcgen05 Matrix Multiplication - Datacenter Blackwell (sm_100a) Tensor Core GEMM

128×128×16 matrix multiplication using Blackwell's 5th generation tensor cores with TMA for data loading and pre-tiled input matrices.

## What This Example Does

- 128×128×16 f16 matmul → bf16 output
- Uses TMA for async global→shared memory transfer
- Pre-tiled input matrices (K-major layout)
- Full pipeline: TMA load → SMEM → MMA → TMEM → epilogue → global

## Key Concepts Demonstrated

### SMEM Descriptor Building

```rust
fn build_smem_descriptor(
    smem_addr: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    swizzle: u8,
) -> u64 {
    let addr_enc = (smem_addr >> 4) & 0x3FFF;
    let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    let fixed_bit = 1u64 << 46;
    let swizzle_bits = (swizzle as u64) << 61;

    addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit | swizzle_bits
}
```

### Full Matmul Kernel

```rust
#[kernel]
pub unsafe fn tcgen05_matmul_128x128_tiled(
    a_tma: *const TmaDescriptor,
    b_tma: *const TmaDescriptor,
    mut out: DisjointSlice<u32>,
    tile_a_x: i32, tile_a_y: i32,
    tile_b_x: i32, tile_b_y: i32,
) {
    // Resources
    static mut SMEM_A: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
    static mut SMEM_B: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
    static mut SMEM_OUT: SharedArray<u32, 8192, 128> = SharedArray::UNINIT;
    static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
    static mut TMA_BAR: Barrier = Barrier::UNINIT;
    static mut MMA_BAR: Barrier = Barrier::UNINIT;

    // PHASE 1: Initialize barriers
    // PHASE 2: Allocate TMEM (512 units for 128×128 accumulators)
    // PHASE 3: TMA load A and B tiles
    // PHASE 4: Build descriptors, execute MMA
    // PHASE 5: Epilogue - convert f32 → bf16, store via stmatrix
    // PHASE 6: Copy to global
    // PHASE 7: Cleanup
}
```

### Pre-Tiled Input (Host Side)

```rust
use cuda_host::tiling::to_k_major_f16;

// Row-major input
let host_a_rowmajor: Vec<f16> = /* ... */;

// Convert to K-major tiled layout
let mut host_a_tiled = vec![f16::ZERO; M * K];
to_k_major_f16(&host_a_rowmajor, &mut host_a_tiled, M, K);
```

### Epilogue (f32 → bf16 + stmatrix)

```rust
// PHASE 5: Epilogue
const N: usize = 128;
let warp_row_base = (warp_id * 32) as usize;

for tmem_row_block in 0..2 {
    for col_block in 0..8 {
        // Load from TMEM (f32 accumulators)
        let regs = tcgen05_ld_16x256b_pure(tmem_addr + offset);
        tcgen05_load_wait();

        // Convert f32 → bf16 (pack two values)
        let packed = cvt_bf16x2_f32(regs[0], regs[1]);

        // Store via stmatrix (8×8 tiles)
        stmatrix_m8n8_x2(smem_addr, packed_lo, packed_hi);
    }
}
```

## Build and Run

```bash
cargo oxide run tcgen05_matmul
```

## Expected Output

### On Blackwell Datacenter (sm_100a):

```text
=== Unified tcgen05 Matmul Example ===

GPU Compute Capability: sm_100
✓ PTX loaded successfully

--- Test: tcgen05_matmul_128x128_tiled ---

Matrix: 128×128×16
Pre-tiling matrices...
Expected: all elements = 1360, sum = 22282240

Launching tcgen05_matmul_128x128_tiled...

OUTPUT (first 4×8):
  r0:  1360  1360  1360  1360  1360  1360  1360  1360
  r1:  1360  1360  1360  1360  1360  1360  1360  1360
  r2:  1360  1360  1360  1360  1360  1360  1360  1360
  r3:  1360  1360  1360  1360  1360  1360  1360  1360

SUM CHECK:
  Expected: 22282240
  Got:      22282240

✅ tcgen05_matmul_128x128_tiled PASSED!

=== tcgen05 Matmul Test Complete ===
```

### On Non-Datacenter GPUs (Consumer Blackwell, Hopper, Ada):

```text
=== Unified tcgen05 Matmul Example ===

GPU Compute Capability: sm_120

⚠️  WARNING: tcgen05 requires sm_100 (datacenter Blackwell)!
   Your GPU is sm_120 (consumer Blackwell has no tcgen05).
   PTX was generated successfully; run on sm_100 to execute kernels.

📝 PTX Verification:
   PTX file generated at: .../tcgen05_matmul/tcgen05_matmul.ptx

📝 To inspect generated PTX:
   cat .../tcgen05_matmul/tcgen05_matmul.ptx

   Look for: tcgen05.mma instructions
```

## Hardware Requirements

- **Required GPU**: Blackwell datacenter B100, B200 or newer (sm_100a)
- **NOT supported**: Consumer Blackwell (sm_120), Hopper (sm_90), Ada (sm_89)
- **CUDA Driver**: 13.x (R580+) with Blackwell support
- **Memory**: ~32KB shared memory per block

## Test Data

The test uses a pattern that produces a known result:
- `A[i,k] = k` for all rows
- `B[n,k] = k+1` for all columns (B stored as N×K)
- `C[i,j] = sum(k * (k+1)) for k in 0..16 = 1360`

Every element of the 128×128 output should be 1360.

## Matrix Layout

### K-Major Tiling

```text
Row-major:           K-Major Tiled:
┌─────────────┐      ┌─────────────┐
│ 0 1 2 3 ... │      │ Tile(0,0)   │ (64×16 subtile)
│ K K+1 ...   │  →   │ Tile(1,0)   │
│ ...         │      │ ...         │
└─────────────┘      └─────────────┘

Each 64×16 subtile is stored contiguously
for efficient tensor core access.
```

### Memory Sizes

- A tile: 128×16 f16 = 4096 bytes
- B tile: 128×16 f16 = 4096 bytes
- Output: 128×128 bf16 = 32768 bytes (8192 packed u32)
- TMEM: 512 units (accumulator space)

## Pipeline Stages

```text
┌───────────────────────────────────────────────────────────────┐
│                     tcgen05 Matmul Pipeline                   │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Global    TMA      Shared      tcgen05     TMEM     Shared   │
│  Memory ──────────→ Memory ──────────────→ (acc) ──────────→  │
│   A, B              SMEM_A,B    MMA         f32      SMEM_OUT │
│                                             │                 │
│                                             ↓                 │
│                                      cvt_f32_bf16             │
│                                      stmatrix                 │
│                                             │                 │
│  Global ←────────────────────────────────────────────────────-│
│  Memory                                     bf16 output       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Generated PTX (Key Instructions)

```ptx
// TMA load
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [SMEM_A], [a_tma, {tile_x, tile_y}], [TMA_BAR];

// TMEM allocation
tcgen05.alloc.cta_group::1.sync.aligned [TMEM_ADDR], 512;

// MMA (128×128 shape, f16 input, f32 accumulator)
tcgen05.mma.cta_group::1.kind::f16 [tmem], a_desc, b_desc, idesc;

// Commit to barrier
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [MMA_BAR];

// Load from TMEM
tcgen05.ld.sync.aligned.16x256b.x1.b32 {regs}, [tmem + offset];

// Convert f32 → bf16 (packed)
cvt.rn.bf16x2.f32 %r_packed, %f0, %f1;

// Store matrix tile
stmatrix.sync.aligned.m8n8.x2.shared.b16 [smem], {%r0, %r1};
```

## Potential Errors

| Error                  | Cause                  | Solution                 |
|------------------------|------------------------|--------------------------|
| Wrong output values    | Pre-tiling incorrect   | Verify `to_k_major_f16`  |
| Sum mismatch           | Epilogue bug           | Check bf16 conversion    |
| TMEM allocation fail   | Insufficient resources | Reduce TMEM size         |
| TMA timeout            | Wrong tile coordinates | Verify x,y ordering      |
