# debug

## Debug & Utility Intrinsics - Profiling and Error Handling

Tests GPU debugging and profiling features: clock measurement, trap, assertions, breakpoints, profiler triggers, and launch bounds.

## What This Example Does

1. **clock_test**: Measure GPU clock cycles for operations
2. **trap_test**: Abort kernel on error condition
3. **assert_test**: Runtime assertions with `gpu_assert!`
4. **breakpoint_test**: cuda-gdb breakpoints
5. **profiler_test**: NVIDIA profiler trigger points
6. **launch_bounds_test**: Compiler hints for occupancy

## Key Concepts Demonstrated

### Clock Cycle Measurement

```rust
#[kernel]
pub fn clock_test(mut output: DisjointSlice<u64>) {
    let idx = thread::index_1d();
    if let Some(output_elem) = output.get_mut(idx) {
        let start = debug::clock64();

        // Code to measure
        let mut sum: u64 = 0;
        for i in 0..100u64 {
            sum = sum.wrapping_add(i);
        }

        let end = debug::clock64();
        *output_elem = end - start;
    }
}
```

### Trap (Abort Kernel)

```rust
#[kernel]
pub fn trap_test(input: &[i32], mut output: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    if let Some(output_elem) = output.get_mut(idx) {
        let val = input[idx.get()];

        if val < 0 {
            debug::trap();  // Kernel aborts here
        }

        *output_elem = val * 2;
    }
}
```

### GPU Assertions

```rust
#[kernel]
pub fn assert_test(input: &[i32], mut output: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    if let Some(output_elem) = output.get_mut(idx) {
        let val = input[idx.get()];

        // With message
        gpu_assert!(val >= 0, "Expected non-negative value");

        // Without message
        gpu_assert!(val < 1000);

        *output_elem = val + 1;
    }
}
```

The two forms intentionally use different failure mechanisms:

- `gpu_assert!(condition)` lowers to `trap()`.
- `gpu_assert!(condition, "message")` lowers to CUDA's device-side
  `__assertfail` system call and reports the message and call-site metadata.

### Breakpoints (cuda-gdb)

```rust
#[kernel]
pub fn breakpoint_test(mut output: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    if let Some(output_elem) = output.get_mut(idx) {
        if idx.get() == 0 {
            debug::breakpoint();  // cuda-gdb stops here
        }
        *output_elem = idx.get() as i32;
    }
}
```

### Profiler Triggers

```rust
#[kernel]
pub fn profiler_test(input: &[f32], mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(output_elem) = output.get_mut(idx) {
        debug::prof_trigger::<0>();  // Region start marker

        let val = input[idx.get()];
        let result = val * val;

        debug::prof_trigger::<1>();  // Region end marker

        *output_elem = result;
    }
}
```

### Launch Bounds

```rust
#[kernel]
#[launch_bounds(256, 2)]  // Max 256 threads/block, min 2 blocks/SM
pub fn clock_test(mut output: DisjointSlice<u64>) {
    // ...
}

#[kernel]
#[launch_bounds(128, 4)]  // Max 128 threads/block, min 4 blocks/SM
pub fn launch_bounds_test(...) {
    // ...
}
```

## Build and Run

```bash
# Run the non-failing debug suite.
cargo oxide run debug

# Run this mode only under cuda-gdb; continuing from `brkpt` must finish it.
CUDA_OXIDE_DEBUG=full cargo oxide build debug
cuda-gdb --args ./crates/rustc-codegen-cuda/examples/debug/target/release/debug --breakpoint

# Run the isolated assertion-failure path.
cargo oxide run debug -- --fail-assert
```

The failing mode runs separately because a device-side assertion leaves the
current CUDA context in an error state. The CUDA driver prints the assertion
diagnostic to stderr, and synchronization reports `CUDA_ERROR_ASSERT`:

```text
--- Failing mode: gpu_assert!() with negative input ---
src/main.rs:117: debug::kernels: block: [0,0,0], thread: [2,0,0] Assertion `Expected non-negative value` failed.
✓ assertion failure surfaced as expected: DriverError(710, "device-side assert triggered")
  (assertion message printed to stderr by the CUDA driver)
```

## Expected Output

```text
=== GPU Debug & Utility Intrinsics Test (Unified) ===

--- Test 1: clock64() / globaltimer() measurement ---
  Average cycles for 100 iterations: <varies by GPU>
✓ clock_test completed

--- Test 2: trap() with valid input ---
  Input:  [1, 2, 3, 4, 5, 6, 7, 8]
  Output: [2, 4, 6, 8, 10, 12, 14, 16]
✓ trap_test PASSED (no trap triggered)

--- Test 3: gpu_assert!() with valid input ---
  Input:  [0, 1, 2, 3, 4, 5, 6, 7]
  Output: [1, 2, 3, 4, 5, 6, 7, 8]
✓ assert_test PASSED (no assertion failed)

--- Test 4: breakpoint() ---
  ⚠ Skipping breakpoint_test (requires cuda-gdb)

--- Test 5: prof_trigger() ---
  Input:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  Output: [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
✓ profiler_test PASSED

--- Test 6: #[launch_bounds(128, 4)] ---
  Input:  [1, 2, 3, 4, 5, 6, 7, 8]
  Output: [4, 7, 10, 13, 16, 19, 22, 25]
✓ launch_bounds_test PASSED
  (Check PTX for .maxntid 128 .minnctapersm 4)

=== ALL DEBUG TESTS COMPLETED ===
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU
- **CUDA Driver**: 11.0+
- **For breakpoints**: cuda-gdb
- **For profiling**: Nsight Compute or Nsight Systems

## Debug Functions

| Function                            | PTX Instruction         | Purpose                               |
|-------------------------------------|-------------------------|---------------------------------------|
| `clock64()`                         | `mov.u64 %rd, %clock64` | Cycle counter                         |
| `trap()`                            | `trap`                  | Abort kernel                          |
| `gpu_assert!(condition)`            | `trap`                  | Assertion without diagnostic metadata |
| `gpu_assert!(condition, "message")` | `call.uni __assertfail` | CUDA assertion diagnostic             |
| `breakpoint()`                      | `brkpt`                 | Debugger break                        |
| `prof_trigger::<N>()`               | `pmevent N`             | Profiler marker                       |

## Launch Bounds Explained

```rust
#[launch_bounds(maxthreads, minblocks)]
```

| Parameter    | Effect                                              |
|--------------|-----------------------------------------------------|
| `maxthreads` | Max threads per block; compiler can use more regs   |
| `minblocks`  | Min blocks per SM; compiler limits register usage   |

Higher `minblocks` = more blocks = fewer registers per thread = better occupancy
Lower `minblocks` = fewer blocks = more registers per thread = better per-thread performance

## Using cuda-gdb

```bash
# Compile with full device debug information.
CUDA_OXIDE_DEBUG=full cargo oxide build debug

# Launch the fixture's isolated breakpoint mode from its example directory.
cd crates/rustc-codegen-cuda/examples/debug
cuda-gdb --args ./target/release/debug --breakpoint

# In cuda-gdb:
(cuda-gdb) run
(cuda-gdb) # Stops at debug::breakpoint() in thread 0
(cuda-gdb) cuda thread       # Show current CUDA thread
(cuda-gdb) frame 0
(cuda-gdb) print idx_raw     # Must be 0
(cuda-gdb) backtrace         # Frame 0 must be breakpoint_test<<<...>>>
(cuda-gdb) continue
# Host output must end with:
# PASS breakpoint_test: runtime values = [0, 1, 2, 3, 4, 5, 6, 7]
```

## Using Profiler Triggers

```bash
# With Nsight Compute
ncu --nvtx --nvtx-include "prof_trigger" ./debug

# With Nsight Systems
nsys profile --trace=cuda ./debug
```

The `prof_trigger::<0>()` and `prof_trigger::<1>()` calls emit `pmevent` instructions that appear in profiler timelines.

## Generated PTX

```ptx
// Clock measurement
mov.u64 %rd_start, %clock64;
// ... work ...
mov.u64 %rd_end, %clock64;

// Trap
trap;

// Breakpoint
brkpt;

// Profiler trigger
pmevent 0;  // prof_trigger::<0>()
pmevent 1;  // prof_trigger::<1>()

// Launch bounds in entry
.entry launch_bounds_test
    .maxntid 128, 1, 1
    .minnctapersm 4
```

## Potential Errors

| Error              | Cause                        | Solution             |
|--------------------|------------------------------|----------------------|
| Kernel aborted     | trap() hit                   | Check input validity |
| Assertion failed   | gpu_assert! condition false  | Debug condition      |
| Launch failure     | brkpt without debugger       | Run under cuda-gdb   |
| No profiler events | Wrong nsys/ncu flags         | Use `--trace=cuda`   |
