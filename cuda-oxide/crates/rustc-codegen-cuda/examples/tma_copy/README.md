# tma_copy

## TMA (Tensor Memory Accelerator) — Hopper (sm_90+) Async Copy

Demonstrates TMA for efficient tensor data movement. TMA offloads memory copies
to a dedicated hardware unit, enabling overlap with computation.

For TMA multicast (also sm_90+), see the [`tma_multicast`](../tma_multicast/) example.

## What This Example Does

1. **tma_copy_2d_test**: Async 2D tile copy from global to shared memory
2. **tma_pipeline_test**: TMA with mbarrier for completion tracking

Both kernels use `#[grid_constant] tensor_map: &TmaDescriptor`. The generated
host launcher takes the descriptor by value and places its 128 bytes directly
in the kernel's parameter storage. Every thread can address that same read-only
value; no separate device descriptor allocation or upload is needed. The tensor
data itself stays in device global memory and must outlive the launch.
Grid-constant host launches are unsafe, including prepared launches: the caller
must ensure that the encoded tensor addresses remain valid for the GPU until
the work completes. This example retains the input and output buffers until
synchronization finishes.

## Key Concepts Demonstrated

### TMA Descriptor Creation (Host)

```rust
fn create_tma_descriptor(
    global_address: *mut c_void,
    width: u64, height: u64,
    tile_width: u32, tile_height: u32,
) -> CUtensorMap {
    cuTensorMapEncodeTiled(
        &mut tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,  // 2D tensor
        global_address,
        [width, height].as_ptr(),
        [width * sizeof(f32)].as_ptr(),  // Stride in bytes
        [tile_width, tile_height].as_ptr(),
        [1, 1].as_ptr(),  // Element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
}
```

### TMA Copy with Barrier

```rust
#[kernel]
pub fn tma_copy_2d_test(
    #[grid_constant] tensor_map: &TmaDescriptor,
    mut out: DisjointSlice<f32>,
    tile_x: i32, tile_y: i32,
) {
    static mut TILE: SharedArray<f32, 4096, 128> = SharedArray::UNINIT;
    static mut BAR: Barrier = Barrier::UNINIT;

    let tid = thread::threadIdx_x();
    let block_size = thread::blockDim_x();

    if tid == 0 {
        mbarrier_init(&raw mut BAR, block_size);
        fence_proxy_async_shared_cta();
    }
    thread::sync_threads();

    if tid == 0 {
        cp_async_bulk_tensor_2d_g2s(
            &raw mut TILE as *mut u8,
            tensor_map,
            tile_x, tile_y,
            &raw mut BAR,
        );
    }

    let token = if tid == 0 {
        mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)
    } else {
        mbarrier_arrive(&raw const BAR)
    };

    while !mbarrier_test_wait(&raw const BAR, token) {}
    thread::sync_threads();

    let val = unsafe { TILE[thread::index_1d().get()] };
    if let Some(out_elem) = out.get_mut(thread::index_1d()) {
        *out_elem = val;
    }
}
```

### TMA Coordinate System

TMA coordinates are **element offsets** into the global tensor, not tile indices.
Given a 256x256 tensor with 64x64 tiles:

```text
         col 0      col 64     col 128    col 192
        ┌──────────┬──────────┬──────────┬──────────┐
row 0   │ tile     │ tile     │ tile     │ tile     │
        │ (0,0)    │ (1,0)    │ (2,0)    │ (3,0)    │
row 64  ├──────────┼──────────┼──────────┼──────────┤
        │ tile     │ tile     │ tile     │ tile     │
        │ (0,1)    │ (1,1)    │ (2,1)    │ (3,1)    │
        └──────────┴──────────┴──────────┴──────────┘
        ...
```

To load tile `(1, 0)`, pass `{64, 0}` (not `{1, 0}`):

```rust
let tile_x: i32 = 1 * TILE_WIDTH as i32;  // 64
let tile_y: i32 = 0 * TILE_HEIGHT as i32; // 0
```

Passing unaligned coordinates (e.g. `{1, 0}`) causes `CUDA_EXCEPTION_27:
Warp Illegal Instruction Parameter`.

### Critical Details

**128-byte Alignment**: TMA destinations must be 128-byte aligned:

```rust
static mut TILE: SharedArray<f32, 4096, 128> = SharedArray::UNINIT;
//                                    ^^^ alignment parameter
```

**Fence After Init**: Required for TMA to see barrier init:

```rust
mbarrier_init(&raw mut BAR, block_size);
fence_proxy_async_shared_cta();  // Don't skip this!
```

**All Threads Participate**: Unlike sync_threads, mbarrier requires all threads to arrive:

```rust
// Thread 0: arrive_expect_tx (includes expected bytes)
// All others: regular arrive
// Everyone: wait
```

## Build and Run

```bash
cargo oxide run tma_copy
```

## Expected Output

### On Hopper+ (sm_90, sm_100, sm_120):

```text
=== TMA Copy Example ===

GPU Compute Capability: sm_100
Loading PTX from: tma_copy.ptx
✓ PTX loaded successfully

--- Test 1: TMA Copy (tma_copy_2d_test) ---
   ✓ All 4096 values match!
🎉 TMA copy successful!

--- Test 2: TMA Pipeline (tma_pipeline_test) ---
   ✓ All 256 threads completed successfully!
🎉 TMA pipeline test successful!

=== TMA Copy Test Complete ===
```

### On Pre-Hopper:

```text
GPU Compute Capability: sm_86

⚠️  WARNING: TMA requires sm_90+ (Hopper or newer)
   Your GPU is sm_86
   This example will only verify PTX compilation.

ℹ️  PTX load failed (expected on pre-Hopper): ...
```

## Hardware Requirements

- **TMA (Tests 1 & 2)**: Hopper (sm_90) or newer — including consumer Blackwell (sm_120)
- **CUDA Driver**: 12.0+
- **Memory**: Tensor descriptors in read-only kernel parameter storage

## Pipeline Pattern (Double Buffering)

```rust
static mut BUF0: SharedArray<...> = ...;
static mut BUF1: SharedArray<...> = ...;
static mut BAR0: Barrier = ...;
static mut BAR1: Barrier = ...;

// Stage 0: Start loading into BUF0
cp_async_bulk_tensor_2d_g2s(&raw mut BUF0, tensor_map, 0, 0, &raw mut BAR0);

for iter in 0..num_iters {
    // Start loading next tile into alternate buffer
    cp_async_bulk_tensor_2d_g2s(
        if iter % 2 == 0 { &raw mut BUF1 } else { &raw mut BUF0 },
        tensor_map, next_x, next_y,
        if iter % 2 == 0 { &raw mut BAR1 } else { &raw mut BAR0 },
    );

    // Wait for current buffer
    while !mbarrier_try_wait(current_bar, token) {}

    // Compute on current buffer
    compute(current_buf);

    // Swap buffers
    swap(&mut current_buf, &mut next_buf);
    swap(&mut current_bar, &mut next_bar);
}
```

## TMA vs Traditional Memory Copy

| Aspect                 | Traditional            | TMA                    |
|------------------------|------------------------|------------------------|
| Who does the work      | SM threads             | Dedicated hardware     |
| Overlap with compute   | Manual double-buffering| Automatic              |
| 2D/3D support          | Manual indexing        | Hardware native        |
| Completion signal      | None (sync)            | mbarrier integration   |
| L2 residency hints     | None                   | Built-in               |

## TMA Copy Functions

| Function                                 | Description                                                    |
|------------------------------------------|----------------------------------------------------------------|
| `cp_async_bulk_tensor_2d_g2s`            | Global → Shared (2D)                                           |
| `cp_async_bulk_tensor_3d_g2s`            | Global → Shared (3D)                                           |
| `cp_async_bulk_tensor_2d_s2g`            | Shared → Global (2D)                                           |
| `cp_async_bulk_tensor_2d_g2s_multicast`  | Global → Shared (2D) + multicast (see `tma_multicast` example) |

## Generated PTX

```ptx
// TMA descriptor parameter (128-byte opaque blob)
.param .align 64 .b8 tensor_map[128]

// Async 2D tensor copy
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [%rd_smem], [%rd_tensor_map, {%r_x, %r_y}], [%rd_mbar];

// Barrier operations
mbarrier.init.shared.b64 [%rd_bar], %r_count;
fence.proxy.async.shared::cta;
mbarrier.arrive.expect_tx.shared.b64 %rd_token, [%rd_bar], %r_bytes;
mbarrier.try_wait.shared.b64 %p, [%rd_bar], %rd_token;
```

## Potential Errors

| Error                        | Cause                            | Solution                              |
|------------------------------|----------------------------------|---------------------------------------|
| `CUDA_ERROR_NOT_SUPPORTED`   | Pre-Hopper GPU                   | Use sm_90+ hardware                   |
| Misaligned access            | SMEM not 128-byte aligned        | Add alignment to SharedArray          |
| TMA never completes          | Missing fence after mbarrier_init| Add `fence_proxy_async_shared_cta()`  |
| Wrong data                   | Wrong tile coordinates           | Check x,y ordering                    |
| `CUDA_EXCEPTION_27`          | Unaligned TMA coordinates        | Multiply tile index by tile dimension |
