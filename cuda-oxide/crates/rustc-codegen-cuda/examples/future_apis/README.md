# future_apis

## Future APIs - CuSimd and ManagedBarrier Typestate

Tests new type-safe abstractions: `CuSimd<T, N>` for SIMD values and `ManagedBarrier<State, Kind>` for typestate-based barrier management.

## What This Example Does

### CuSimd Tests

1. **test_cusimd**: CuSimd<f32, 4> operations (indexing, accessors, conversion)
2. **test_cusimd_u32**: CuSimd<u32, 4> runtime indexing

### ManagedBarrier Tests

1. **test_managed_barrier**: Single barrier typestate flow
2. **test_multi_barrier**: Multiple barriers (TMA + General)
3. **test_double_buffered_barriers**: Ping-pong pattern with two barriers

## Key Concepts Demonstrated

### CuSimd<T, N> - SIMD Type

```rust
#[kernel]
pub fn test_cusimd(mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let tid = idx.get();

    // Construct from array
    let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let simd4 = CuSimd::<f32, 4>::new(values);

    // Runtime index access
    let val = simd4[tid % 4];

    // Named accessors (like GLSL/HLSL)
    let x = simd4.x();  // First element
    let y = simd4.y();  // Second element
    let z = simd4.z();  // Third element
    let w = simd4.w();  // Fourth element

    // Compile-time indexed access
    let first = simd4.get::<0>();
    let last = simd4.get::<3>();

    // CuSimd<f32, 2> with lo/hi
    let simd2 = CuSimd::<f32, 2>::new([10.0, 20.0]);
    let (lo, hi) = simd2.xy();

    // Convert to array
    let arr = simd4.to_array();
}
```

### ManagedBarrier Typestate

```rust
#[kernel]
pub fn test_managed_barrier(mut output: DisjointSlice<u32>) {
    // Static barrier declaration
    static mut BAR: Barrier = Barrier::UNINIT;

    let tid = thread::threadIdx_x();
    let is_thread0 = tid == 0;

    // 1. Create Uninit handle (all threads)
    let bar = ManagedBarrier::<Uninit, GeneralBarrier>::from_static(&raw mut BAR);

    // 2. Initialize → State transition: Uninit -> Ready
    // Only thread 0 actually initializes; includes sync_threads
    let bar = unsafe { bar.init(32) };

    // 3. Use barrier (all threads)
    let token = bar.arrive();
    bar.wait(token);

    // 4. Invalidate → State transition: Ready -> Invalidated
    thread::sync_threads();
    if is_thread0 {
        let _dead = unsafe { bar.inval() };  // Consumes bar
    }

    // Write result
    if let Some(output_elem) = output.get_mut(thread::index_1d()) {
        *output_elem = 42;
    }
}
```

### Barrier Kinds

```rust
// TMA barrier - for TMA operations with expected TX bytes
let bar_tma = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR_TMA);

// General barrier - for thread synchronization
let bar_gen = ManagedBarrier::<Uninit, GeneralBarrier>::from_static(&raw mut BAR_GEN);
```

### Double Buffering Pattern

```rust
static mut BUF0_BAR: Barrier = Barrier::UNINIT;
static mut BUF1_BAR: Barrier = Barrier::UNINIT;

let buf0_bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BUF0_BAR);
let buf1_bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BUF1_BAR);

let buf0_bar = unsafe { buf0_bar.init(32) };
let buf1_bar = unsafe { buf1_bar.init(32) };

// Ping-pong between barriers
let t0 = buf0_bar.arrive();
buf0_bar.wait(t0);

let t1 = buf1_bar.arrive();
buf1_bar.wait(t1);

let t2 = buf0_bar.arrive();  // Reuse buf0
buf0_bar.wait(t2);
```

## Build and Run

```bash
cargo oxide run future_apis
```

## Expected Output

```text
=== Future APIs Test (Unified) ===

--- Test 1: CuSimd<f32, 4> operations ---
  Thread 0: 1.0 (val_at_tid)
  Thread 7: 30.0 (lo+hi)
✓ CuSimd<f32, 4> PASSED

--- Test 2: CuSimd<u32, 4> indexing ---
  Results: [10, 20, 30, 40, 10, 20, 30, 40]
✓ CuSimd<u32, 4> PASSED

--- Test 3: ManagedBarrier<Uninit/Ready> typestate ---
  All 32 threads wrote 42 after barrier sync
✓ ManagedBarrier PASSED

--- Test 4: Multiple barriers (TmaBarrier + GeneralBarrier) ---
  All 32 threads wrote 99 after both barrier syncs
✓ Multi-barrier PASSED

--- Test 5: Double-buffered barriers (ping-pong) ---
  All 32 threads completed 3 iterations
✓ Double-buffered PASSED

=== ALL FUTURE APIS TESTS COMPLETED ===
```

## Hardware Requirements

- **Minimum GPU**: Hopper (sm_90) or newer. `ManagedBarrier::init_by` emits
  `fence.proxy.async`, which ptxas rejects below sm_90, so `main` skips
  cleanly on anything lower (`if major < 9`). The mbarrier instructions
  themselves only need sm_80+, but the fence is what sets the floor here.
- **CUDA Driver**: 11.0+

## CuSimd API Reference

| Method               | Description                       |
|----------------------|-----------------------------------|
| `new([T; N])`        | Construct from array              |
| `[idx]`              | Runtime index access              |
| `get::<I>()`         | Compile-time index access         |
| `at(idx)`            | Runtime index access (method)     |
| `x(), y(), z(), w()` | Named accessors (N ≥ 1, 2, 3, 4)  |
| `xy()`               | Returns tuple (N ≥ 2)             |
| `to_array()`         | Convert to `[T; N]`               |

## Typestate Pattern

The `ManagedBarrier<State, Kind>` uses Rust's type system to enforce correct barrier usage:

```text
Uninit ─────init()────→ Ready ─────inval()────→ Invalidated
          (creates)            (destroys)

Ready state supports:
  - arrive() → Token
  - wait(Token)
  - arrive_expect_tx() (TmaBarrier only)
  - try_wait(Token) → bool
  - try_wait_parity(phase) → bool
```

Invalid state transitions are **compile-time errors**:

```rust
let bar = ManagedBarrier::<Uninit, _>::from_static(...);
bar.arrive();  // ERROR: Uninit doesn't have arrive()

let bar = bar.init(32);  // Now Ready
let _dead = bar.inval();
_dead.arrive();  // ERROR: Invalidated doesn't have arrive()
```

## Generated PTX

CuSimd generates efficient register operations:

```ptx
// CuSimd<f32, 4>::new([1.0, 2.0, 3.0, 4.0])
mov.f32 %f1, 1.0;
mov.f32 %f2, 2.0;
mov.f32 %f3, 3.0;
mov.f32 %f4, 4.0;

// Runtime indexing generates a switch
@%p1 mov.f32 %f_result, %f1;
@%p2 mov.f32 %f_result, %f2;
// ...
```

ManagedBarrier generates standard mbarrier instructions:

```ptx
mbarrier.init.shared.b64 [%bar], %count;
mbarrier.arrive.shared.b64 %token, [%bar];
mbarrier.test_wait.shared.b64 %p, [%bar], %token;
mbarrier.inval.shared.b64 [%bar];
```
