# warp_reduce

## Warp-Level Primitives - Shuffle Operations

Demonstrates warp-level `shuffle_xor`, `shuffle_down`, and `shuffle`
(broadcast) operations without shared-memory storage.

The i32/f32 synchronized shuffle leaves are generated as intrinsic records
`i0050`-`i0057`. The friendly `cuda_device::warp` helpers and this example's
output remain unchanged.

## What This Example Does

1. **warp_reduce_sum**: Butterfly reduction using `shuffle_xor` - all lanes get the sum
2. **warp_reduce_sum_down**: Sequential reduction using `shuffle_down` - only lane 0 gets sum
3. **warp_broadcast**: Broadcast lane 0's value to all lanes using `shuffle`
4. **test_lane_id**: Verify `lane_id()` intrinsic
5. **warp_reduce_sum_util / warp_reduce_max_util / warp_reduce_min_util**:
   The packaged `warp::reduce_{sum,max,min}_f32` utilities - every lane
   writes the reduced value it received, and the host checks all of them
6. **warp_reduce_sum_f64_util**: The f64 flavor, `warp::reduce_sum_f64`

## Key Concepts Demonstrated

### Butterfly Reduction (shuffle_xor)

```rust
// All lanes end up with the complete sum
val = val + warp::shuffle_xor_f32(val, 16);  // Exchange with lane ± 16
val = val + warp::shuffle_xor_f32(val, 8);   // Exchange with lane ± 8
val = val + warp::shuffle_xor_f32(val, 4);   // Exchange with lane ± 4
val = val + warp::shuffle_xor_f32(val, 2);   // Exchange with lane ± 2
val = val + warp::shuffle_xor_f32(val, 1);   // Exchange with lane ± 1
// val now contains sum of all 32 lanes
```

### Sequential Reduction (shuffle_down)

```rust
// Only lane 0 gets the final sum
val = val + warp::shuffle_down_f32(val, 16);
val = val + warp::shuffle_down_f32(val, 8);
val = val + warp::shuffle_down_f32(val, 4);
val = val + warp::shuffle_down_f32(val, 2);
val = val + warp::shuffle_down_f32(val, 1);
// Lane 0 has the sum; other lanes have partial sums
```

### Broadcast

```rust
// Get lane 0's value to all lanes
let broadcast_val = warp::shuffle_f32(my_val, 0);
// All lanes now have lane 0's original value
```

### Lane ID

```rust
let lane = warp::lane_id();  // 0-31 within the warp
let warp_id = warp::warp_id();  // Which warp in the block
```

### Packaged Reduce Utilities

The hand-rolled butterfly above is also available as one-call utilities,
for both precisions:

```rust
let total = warp::reduce_sum_f32(val);      // every lane gets the sum
let hi = warp::reduce_max_f32(val);         // every lane gets the max
let lo = warp::reduce_min_f32(val);         // every lane gets the min
let total_f64 = warp::reduce_sum_f64(val);  // f64 flavors too
```

They use the full-warp mask, so all 32 lanes must be converged and
participate; calling them from divergent control flow or from a block
with fewer than 32 threads is undefined.

## Build and Run

```bash
cargo oxide run warp_reduce
```

## Expected Output

```text
=== Unified Warp Reduction Example ===

Input data: 256 elements, 8 warps
  First warp values: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
  Expected warp sum: 496

--- Test 1: Butterfly Reduction (shuffle_xor) ---
Warp sums: [496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0]
✓ All 8 warp sums correct (each = 496)

--- Test 2: Sequential Reduction (shuffle_down) ---
Warp sums: [496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0, 496.0]
✓ All 8 warp sums correct (each = 496)

--- Test 3: Broadcast (shuffle to lane 0) ---
✓ Broadcast correct: all lanes have lane 0's value

--- Test 4: Lane ID ---
✓ Lane IDs correct: 0-31 pattern for each warp

--- Test 5: Reduce Utilities (f32) ---
✓ reduce_sum_f32 correct: all 256 lanes hold 496
✓ reduce_max_f32 correct: all 256 lanes hold 31
✓ reduce_min_f32 correct: all 256 lanes hold 0

--- Test 6: Reduce Utility (f64) ---
✓ reduce_sum_f64 correct: all 256 lanes hold 496

✓ SUCCESS: All warp tests passed!
```

## Hardware Requirements

- **Instruction floor**: PTX 6.0 and sm_30
- **Project target**: Ampere or newer
- **CUDA Driver**: 11.0+

## Shuffle Functions

| Function                       | Description                      | Result Location |
|--------------------------------|----------------------------------|-----------------|
| `shuffle_f32(val, src_lane)`   | Read from specific lane          | All lanes       |
| `shuffle_xor_f32(val, mask)`   | Exchange with lane XOR'd by mask | All lanes       |
| `shuffle_down_f32(val, delta)` | Read from lane + delta           | Lower lanes     |
| `shuffle_up_f32(val, delta)`   | Read from lane - delta           | Upper lanes     |

## Why Warp Operations?

Shuffles move register values directly between lanes without shared-memory
storage. They still have a participation contract: every named, non-exited
lane must execute the same synchronized shuffle with the same member mask.
The full-warp helpers in this example use `u32::MAX`, so all 32 lanes must
reach each call. A shuffle is not a general memory barrier.

On `sm_6x` and earlier, named lanes must be converged and no lane outside the
mask may be active. Ampere and newer use the synchronized forms shown here.

## Common Patterns

### Warp-Wide Max

```rust
val = max(val, warp::shuffle_xor_f32(val, 16));
val = max(val, warp::shuffle_xor_f32(val, 8));
val = max(val, warp::shuffle_xor_f32(val, 4));
val = max(val, warp::shuffle_xor_f32(val, 2));
val = max(val, warp::shuffle_xor_f32(val, 1));
```

This is exactly what `warp::reduce_max_f32(val)` packages; prefer the
utility unless you need a non-standard offset schedule.

### Prefix Sum (Scan)

```rust
let mut offset = 1;
while offset < 32 {
    let n = warp::shuffle_up_f32(val, offset);
    if lane >= offset { val += n; }
    offset *= 2;
}
```

### Warp-Level Output

```rust
if lane == 0 {
    let warp_idx = gid / 32;
    output[warp_idx] = val;  // One write per warp
}
```

## Generated PTX

```ptx
// Shuffle XOR
shfl.sync.bfly.b32 %f_result, %f_val, %r_mask, 0x1f, 0xffffffff;

// Shuffle down
shfl.sync.down.b32 %f_result, %f_val, %r_delta, 0x1f, 0xffffffff;

// Broadcast (shuffle to lane 0)
shfl.sync.idx.b32 %f_result, %f_val, 0, 0x1f, 0xffffffff;

// Lane ID
mov.u32 %r_lane, %laneid;
```

Generated code owns the raw APIs, importer dispatch, dialect operations,
typed lowering, and target metadata. LLVM receives a fixed clamp of 31 for
`idx`, `bfly`, and `down`, and 0 for `up`. The generated probes cover both
register and immediate lane/member-mask operands.

The generated-intrinsics migration passed this example's compile-only gate
with the same host output contract. No GPU run was available for that batch.
