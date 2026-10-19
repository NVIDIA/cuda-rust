# barrier

## Async Barriers (mbarrier) - Advanced Synchronization

Demonstrates mbarrier (asynchronous barrier) intrinsics for fine-grained synchronization. Mbarriers are more powerful than `__syncthreads()` and are essential for TMA and tensor core operations.

## What This Example Does

1. **barrier_sync_test**: Basic mbarrier arrive/wait pattern
2. **barrier_shared_data_test**: Shared memory data exchange with mbarrier synchronization

## Key Concepts Demonstrated

### Mbarrier Lifecycle

```rust
static mut BAR: Barrier = Barrier::UNINIT;

// 1. Initialize (one thread)
if tid == 0 {
    mbarrier_init(&raw mut BAR, block_size);  // Expected arrivals
}
sync_threads();  // Ensure all see initialized barrier

// 2. Arrive (all threads)
let token = mbarrier_arrive(&raw const BAR);

// 3. Wait (all threads)
while !mbarrier_test_wait(&raw const BAR, token) {
    // Spin-wait until all arrivals complete
}

// 4. Invalidate (one thread)
if tid == 0 {
    mbarrier_inval(&raw mut BAR);
}
```

### Token-Based Synchronization

- `mbarrier_arrive()` returns a **token** representing this phase
- `mbarrier_test_wait(token)` checks if the phase completed
- Tokens enable tracking multiple barrier phases

### Why Mbarriers Over sync_threads()?

| Feature     | sync_threads()    | mbarrier                 |
|-------------|-------------------|--------------------------|
| Blocking    | Yes (all threads) | No (non-blocking test)   |
| Async ops   | No                | Yes (TMA, tensor cores)  |
| Phases      | Single            | Multiple (via tokens)    |
| Flexibility | All-or-nothing    | Partial arrival counts   |

## Build and Run

```bash
cargo oxide run barrier
```

## Expected Output

```text
=== Unified Barrier Test ===

--- Test 1: barrier_sync_test ---
✓ All 256 threads completed barrier sync

--- Test 2: barrier_shared_data_test ---
✓ Shared memory + barrier pattern correct
  First 8 values: [1, 2, 3, 4, 5, 6, 7, 8]
  Last 3 values: [254, 255, 0]

✓ SUCCESS: All barrier tests passed!
```

## Hardware Requirements

- **Minimum GPU**: Ampere (sm_80) or newer. `main` skips cleanly below that
  (`if major < 8`), matching the `mbarrier requires sm_80+` message it
  prints. If you have seen `cuda::barrier` work on sm_70, that is libcu++'s
  software fallback; the PTX `mbarrier` instructions this example emits
  need sm_80+.
- **CUDA Driver**: 11.0+

## Mbarrier Functions

| Function                                | Description                                   |
|-----------------------------------------|-----------------------------------------------|
| `mbarrier_init(bar, count)`             | Initialize barrier with expected arrival count|
| `mbarrier_arrive(bar)`                  | Arrive at barrier, return token               |
| `mbarrier_test_wait(bar, token)`        | Non-blocking check if phase complete          |
| `mbarrier_inval(bar)`                   | Invalidate barrier (required cleanup)         |
| `mbarrier_arrive_expect_tx(bar, tx)`    | Arrive with expected async transaction bytes  |

## Common Patterns

### Producer-Consumer

```rust
// Producer thread
produce_data(&mut SHARED_DATA);
mbarrier_arrive(&raw const BAR);

// Consumer threads
let token = mbarrier_arrive(&raw const BAR);
while !mbarrier_test_wait(&raw const BAR, token) {}
consume_data(&SHARED_DATA);
```

### Double Buffering

```rust
static mut BAR_A: Barrier = Barrier::UNINIT;
static mut BAR_B: Barrier = Barrier::UNINIT;

// Ping-pong between two buffers with separate barriers
```

### TMA Integration

```rust
// Thread 0: issue TMA + arrive with expected bytes
cp_async_bulk_tensor_2d_g2s(..., &raw mut BAR);
mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES);

// All threads wait for TMA completion
while !mbarrier_test_wait(&raw const BAR, token) {}
```

## Potential Errors

| Error                              | Cause                      | Solution                                    |
|------------------------------------|----------------------------|---------------------------------------------|
| Hang / deadlock                    | Mismatched arrival count   | Ensure init count matches actual arrivals   |
| Race condition                     | Missing fence after init   | Add `fence_proxy_async_shared_cta()` for TMA|

## Generated PTX

```ptx
// Initialize
mbarrier.init.shared.b64 [%rd_bar], %r_count;

// Arrive
mbarrier.arrive.shared.b64 %rd_token, [%rd_bar];

// Test wait
mbarrier.test_wait.shared.b64 %p_done, [%rd_bar], %rd_token;

// Invalidate
mbarrier.inval.shared.b64 [%rd_bar];
```
