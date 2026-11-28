# Provably Safe Kernels (preview)

:::{admonition} Preview
:class: warning
Compile-time Lean 4 verification (`#[verify(...)]` and `cuda-oxide/oxide-proofs-preview/`) is in preview. It covers a validated subset of `PostMem2Reg` integer, shared-memory, barrier, 2-CTA cluster, and Blackwell `tcgen05` / `TMEM` MIR through the pinned Lean `v4.33.1` toolchain. Kernels outside that subset fail closed at compile time, and attribute syntax and supported operations may change between releases.
:::

On a CPU, **Safe Rust** means your program has zero undefined behavior:
- No out-of-bounds memory accesses
- No data races between threads
- No dangling pointers

Notice what Safe Rust does *not* mean: it does not prove **functional correctness** (that your math formula is right). Writing `a[i] - b[i]` when you meant `a[i] + b[i]` is a logic bug, but it is still 100% memory-safe Rust.

On a GPU, thousands of threads run the same kernel function at once, and the Rust borrow checker cannot reason about thread coordinates or barriers:
- **What Rust types already cover**: [Safe Kernels Using Rust's Type System](the-safety-model.md) proves one-thread-per-element writes with disjoint slices, and [Bounds Checks](bounds-checks.md) hoists row and column checks with matrix views.
- **Where Rust types stop**: When 256 threads share an on-chip tile across barriers, the borrow checker cannot tell if two threads collide or if a barrier sits inside an `if` branch. Without compile-time verification, those kernels require `unsafe` blocks or unchecked indexing flags.
- **How we make them provably safe**: The `#[verify(...)]` attribute pairs Rust's type system with compile-time **Lean 4** safety proofs, giving you **Provably Safe SIMT** across shared memory, barriers, and async hardware.

```{figure} images/three-gpu-hazards.svg
:alt: Three GPU safety hazards proved at compile time by the verify attribute
:align: center
:width: 100%

The three GPU safety bugs caught at compile time: out-of-bounds indexing, same-phase shared-memory data races, and divergent barriers.
```

---

## 1. The Core Idea: Specify and Verify

You declare the safety properties you want on the kernel, and the compiler proves them before generating PTX.

We do not prove functional correctness or floating-point math formulas. Instead, we prove the exact memory and concurrency rules needed to make the kernel safe and strip out redundant bounds checks:

| Property You Write  | What Lean 4 Proves at Compile Time                        | What Is Not Claimed                       |
|:--------------------|:----------------------------------------------------------|:------------------------------------------|
| `no_oob_idx`        | Every buffer read, write, and atomic index stays in range | Values stored at those indices (`NaN`)    |
| `no_data_race`      | No non-serialized or wrong-scope races on shared/global   | Shared-memory bank conflicts (speed only) |
| `disjoint_writes`   | Two different threads never write to the same output slot | Which formula is written (`a - b` vs `+`) |
| `writes_all_elems`  | Every output slot is written by a launched thread         | Whether the written value is right        |
| `uniform_barriers`  | Convergent barriers, warp votes, and CTA-uniform exits    | Waiting across different thread blocks    |
| `tmem_lifecycle`    | Valid `TMEM` bounds, warpgroup slices, and async order    | Floating-point Tensor Core rounding       |

Because shared-memory race checks group accesses by each thread's barrier phase, `no_data_race(TILE)` on a `SharedArray` always requires `uniform_barriers` in the same `#[verify(...)]` attribute.

---

## 2. How It Works

Verification splits the work between compile time and launch time so GPU threads pay zero runtime cost:
1. **Before launch on the CPU**: The host checks buffer sizes and launch dimensions once per kernel call.
2. **At compile time in Lean 4**: Right after variables are promoted to SSA registers (and before loops are unrolled), the compiler translates the kernel line by line into Lean 4, proves every safety rule, and deletes the proven bounds checks before generating PTX.

```{figure} images/specify-and-verify-pipeline.svg
:alt: Specify and Verify pipeline
:align: center
:width: 100%

The compiler pauses in memory before loop unrolling, proves the kernel safe in Lean 4, erases the proven bounds checks, and resumes lowering to PTX.
```

### Compile Time vs. Runtime

Almost all of the safety checking happens once when you build the kernel. Only the few numbers that do not exist until runtime (like dynamic slice lengths) are checked on the CPU right before launch, leaving the GPU threads completely free of extra checks:

- **Proved once at compile time (Lean 4)**:
  - Every global slice, shared-memory array, barrier array, and Tensor Memory index stays in bounds.
  - Shared-memory and mutable-slice writes never collide across threads in the same phase.
  - Block and cluster barriers are reached uniformly by all participating threads.
  - Async DMA copies, producer-consumer ring buffers, 2-CTA cluster ranks, and Tensor Memory lifecycles happen in the right order.
- **Checked once per launch (Host CPU)**:
  - Block and cluster dimensions match what the kernel declared.
  - Runtime buffer sizes and unsigned scalar extents satisfy every `requires` relation in `#[launch_contract(requires = (...))]` (slice `.len()`, unsigned integer scalars, literals, `+`, `-`, `*`, and `>=`, `>`, `<=`, `<`, `==`, `!=`, evaluated with checked `u64` arithmetic and forwarded 1-to-1 into Lean 4).
  - The launch grid is large enough when the kernel promises to write every output element.
- **Paid per element on the GPU**:
  - **Nothing.** Proven bounds checks are stripped out before PTX codegen, so inner loops compile down to the exact same machine instructions as hand-written unchecked code.

| What Needs Checking        | Compile Time (Lean 4)                      | Launch Time (Host CPU)              | GPU Threads at Runtime |
|:---------------------------|:-------------------------------------------|:------------------------------------|:-----------------------|
| Slice & shared bounds      | Proved from guards, loops, and block shape | Checks slice length rules once      | Bounds checks removed  |
| Race freedom & phases      | Proved across thread pairs and barriers    | Nothing                             | Zero extra sync cost   |
| Async rings & clusters     | Proved by the control-flow state machine   | Checks block & cluster sizes once   | Native hardware ops    |
| Tensor Memory lifecycle    | Proves warp slices, order, and column caps | Nothing                             | Native Tensor Core ops |
| Full output coverage       | Proves active threads write their slot     | Checks grid covers output length    | Nothing                |

### What Rust Proves vs. What Lean 4 Proves

Unlike a C or CUDA C++ verifier, Lean 4 does not need to prove pointer validity, alignment, or buffer non-aliasing from scratch. However, the Rust compiler type-checks a kernel function as if a **single thread** called it, so we split the safety proof across the host boundary and the GPU launch:

- **What transfers directly from Rust (Host to GPU)**:
  - **Valid, aligned, live buffers**: Device buffers on the host are guaranteed non-null, aligned, and live for the launch.
  - **Separate parameters never alias**: At the launch call site, Rust's borrow checker proves that a mutable output buffer does not overlap the read-only input buffers.
  - **Shared read-only slices**: Copying `&[f32]` to thousands of GPU threads is sound because no thread mutates the input slices.
- **What does *not* transfer from Rust alone**:
  - Giving thousands of GPU threads a raw `&mut [f32]` (or shared-memory array) would let two threads mutate the same element at the same time.
  - Per-element exclusivity across GPU threads is proved by `DisjointSlice` (for unverified kernels) or by Lean 4 (`#[verify(no_oob_idx(c), disjoint_writes(c))]`, which unlocks plain `c: &mut [f32]` with zero `DisjointSlice` or `ThreadIndex` wrapper types).

| Property                                  | Who Proves It                          | Why It Holds on the GPU                                        |
|:------------------------------------------|:---------------------------------------|:---------------------------------------------------------------|
| **Buffers are non-null, aligned, live**   | Rust compiler at host launch           | Host passes typed device buffer references                     |
| **Output does not alias inputs**          | Rust borrow checker at host launch     | Host cannot pass the same buffer as both shared and mutable    |
| **Multiple readers on read-only slices**  | Rust type system (`Sync`)              | Immutable slices never race across GPU threads                 |
| **Disjoint per-thread output writes**     | `DisjointSlice` or Lean 4              | Each thread only writes to its own unique output slot          |
| **Shared-memory exclusivity**             | Lean 4 (`no_data_race` + barriers)     | Phase 0 writes one slot per thread or leader; Phase 1 reads    |
| **In-bounds indexing**                    | Lean 4 (`no_oob_idx` + host check)     | Proved across all active threads before erasing bounds checks  |

### How the Compiler Translates Your Kernel

cuda-oxide translates the kernel instruction by instruction before loop unrolling (while loops are still small and compact):
- **Index math and branches** (adds, shifts, masks, loop counters, and `if` checks) are copied one-for-one into Lean 4.
- **Floating-point and per-lane scalar/packed math is ignored for indexing** (including standard float operators, math functions like `.sin()`, and explicit-rounding intrinsics like `mul_rn_f32` / `add_rn_f32`), because float values cannot change which memory slot a thread touches.
- **Barriers and async hardware ops** (block syncs, cluster syncs, hardware barriers, DMA copies, work-stealing, and Tensor Memory) are tracked in a small control-flow state machine.
- **Lean 4 does all the proving**: it turns every memory access and thread pair into plain integer math, checks the hardware ordering rules, and only lets the compiler delete a bounds check once the proof succeeds.

### Trusted Computing Base

Lean 4 checks every proof from scratch and rejects anything that leaves a gap or adds custom assumptions. Everything outside that proof check is part of the **Trusted Computing Base (TCB)**:
- **MIR import and SSA promotion**: Lowering Rust into compiler IR and turning stack variables into SSA registers (`--device-debug` skips SSA promotion and is rejected on `#[verify]` modules; pass `--lineinfo` for source-line info instead).
- **The Lean transcriber**: Copying basic blocks and control-flow ops into Lean 4, and stopping immediately if it sees an unsupported instruction or attribute.
- **Downstream lowering**: Deleting proven bounds checks, unrolling loops, and generating LLVM IR and PTX after verification passes.
- **Host launch checks and CUDA runtime**: Checking block sizes and buffer lengths on the CPU before launching the kernel.

---

## 3. Reading Verification Errors

When a kernel has a safety bug, compilation stops before PTX is generated. You will see one of two kinds of Lean 4 errors depending on what went wrong:

### Bounds and Data-Race Bugs (Arithmetic Counterexample)

If an index can walk out of bounds or two threads can collide on the same shared-memory slot (for example, writing `a[idx + 1]` under `if idx < c.len()`, forgetting `if lane == 0`, or dropping a barrier between a write and a read), Lean's integer solver prints a counterexample:

```text
error: omega could not prove the goal:
a possible counterexample may satisfy the constraints
  0 ≤ c ≤ 255
  a - 256*b - c ≥ 1
  256*b + c - e ≥ -1
  a - e ≤ 0
  ...
where
  a := ↑(hvc_512.sliceLen 2)   -- length of parameter 2 (output c)
  b := ↑hvc_511.bx             -- blockIdx.x
  c := ↑hvc_511.tx             -- threadIdx.x (0..255)
  e := ↑(hvc_512.sliceLen 0)   -- length of parameter 0 (input a)
```

How to read that output in three quick steps:
1. **Start with the `where` legend at the bottom**:
   - `sliceLen 0`, `sliceLen 1`, `sliceLen 2` are your slice parameters in function order (`a`, `b`, `c`).
   - `bx` and `tx` are the block and thread coordinates (`256 * b + c` is the global thread index `idx`).
   - Any `/ 18446744073709551616` terms are `2^64` from 64-bit `usize` wraparound checks; you can ignore them unless your index math itself overflows 64 bits.
2. **Read the constraints with those names**:
   - `a - (256*b + c) >= 1` means the thread passed `idx < c.len()`.
   - `a - e <= 0` means the host launch check `a.len() >= c.len()` holds.
   - `(256*b + c) - e >= -1` means `idx + 1 >= a.len()`, so `a[idx + 1]` walks off the end of `a`!
3. **Plug in a tiny test case**: If both `a` and `c` have length 10, thread 9 passes `9 < 10` and reads `a[9 + 1] = a[10]`, which is out of bounds.

### Barrier, Ring, and Tensor Memory Bugs (State-Machine Failure)

If a barrier sits inside a divergent `if`, a ring producer forgets to wait on its empty barrier, or a Tensor Memory op runs out of order, Lean's control-flow state machine returns `false` instead of `true` and names the check that failed:

```text
error: Application type mismatch: The argument
  rfl
... is expected to have type
  checkTmemLifecycleWith controlFlowCfg cfgStateMap = true
```

- `checkUniformBarriersWith ... = true` failing means a block sync, cluster sync, or cyclic ring barrier is not reached cleanly on all active paths.
- `checkTmemLifecycleWith ... = true` failing means an async proxy fence, barrier wait, full-warp allocation/free, or Tensor Memory column bound was violated.
- Run `cargo oxide lean-export <target>` at any time to inspect the generated Lean file alongside your MIR blocks.

---

## 4. Walkthroughs

### Example 1: 1D Vector Addition

In `safe_vecadd`, each thread checks `c.get_mut(idx)` (`idx < c.len()`) and then reads `a[idx_raw]` and `b[idx_raw]`:

```rust
#[launch_contract(
    domain = 1,
    block = (256, 1, 1),
    requires = (a.len() >= c.len(), b.len() >= c.len()),
)]
#[verify(
    no_oob_idx(a, b, c),
    disjoint_writes(c),
    writes_all_elems(c),
)]
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx_raw] + b[idx_raw];
    }
}
```

What happens when you build:
- The host contract guarantees `a` and `b` are at least as long as `c`.
- Lean 4 proves that any thread inside `if let Some(c_elem)` is also in bounds for `a` and `b`, and deletes those bounds checks before PTX lowering.
- Because `#[verify]` proves `no_oob_idx(c)` and `disjoint_writes(c)` across all GPU threads, you can also write the exact same kernel using plain `c: &mut [f32]` and ordinary slice indexing (with all three bounds checks `a[idx]`, `b[idx]`, and `c[idx]` elided):

```rust
#[launch_contract(
    domain = 1,
    block = (256, 1, 1),
    requires = (a.len() >= c.len(), b.len() >= c.len()),
)]
#[verify(no_oob_idx(a, b, c), disjoint_writes(c), writes_all_elems(c))]
#[kernel]
pub fn vecadd_mut_slice(a: &[f32], b: &[f32], c: &mut [f32]) {
    let idx = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
        + (thread::threadIdx_x() as usize);
    if idx < c.len() {
        c[idx] = a[idx] + b[idx];
    }
}
```

You can inspect the MIR input and generated Lean theorems at any time:

```console
$ cargo oxide lean-export safe_vecadd
```

---

### Example 2: 2D Tiled Matrix Multiply (Shared Memory + Barriers)

In a 2D tiled matrix multiply (`safe_gemm_views`), a 16x16 block of 256 threads cooperates in two phases separated by a barrier:
- **Phase 0 (Stage)**: Each thread `(tx, ty)` writes one float into shared memory at its own slot (`ty * 16 + tx`).
- **Barrier**: All 256 threads wait at `sync_threads()` until both shared tiles are full.
- **Phase 1 (Scan)**: Each thread loops 16 times (`i = 0..16`), reading across row `ty` of tile A and down column `tx` of tile B.

```{figure} images/phased-shared-memory.svg
:alt: Phased shared memory verification across sync_threads
:align: center
:width: 100%

Phase 0 writes one unique slot per thread. After the barrier wall, Phase 1 only reads. Keeping writes and cross-thread reads in separate phases prevents data races.
```

Because Lean 4 proves bounds, race freedom, and uniform barriers at compile time:
- **Zero `unsafe` blocks with matrix views**: Adding `#[verify(no_oob_idx(TILE_A, TILE_B), no_data_race(TILE_A, TILE_B), uniform_barriers)]` proves both phases of shared-memory access and removes all 4 shared-memory bounds checks.
- **Plain slice indexing without wrapper views**: Even without matrix view types, `sgemm_tiled_raw` is 100% safe Rust and removes all 6 bounds checks (2 on global slices `a` and `b`, plus 4 on shared tiles `TILE_A` and `TILE_B`).

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
    requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
#[verify(
    no_oob_idx(a, b, TILE_A, TILE_B),
    no_data_race(TILE_A, TILE_B),
    uniform_barriers,
)]
pub fn sgemm_tiled_raw(
    m: u32,
    n: Uniform<u32>,
    k: u32,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;
    // ...
    while tile < num_tiles {
        // ...
        // Phase 0 (Lean-verified bounds and disjoint per-thread slot):
        {
            TILE_A[smem_idx] = a_value;
            TILE_B[smem_idx] = b_value;
        }

        thread::sync_threads();

        // Phase 1 (Lean-verified bounds and barrier separation):
        {
            let mut i = 0usize;
            while i < TILE as usize {
                sum += TILE_A[ty as usize * 16 + i] * TILE_B[i * 16 + tx as usize];
                i += 1;
            }
        }

        thread::sync_threads();
        tile = tile.wrapping_add(1);
    }
    // ...
}
```

Inspect both verified kernels or run the benchmark with:

```console
$ cargo oxide lean-export safe_gemm_views
$ cargo oxide run safe_gemm_views
```

| Kernel Variant                   | Safety Status | Bounds Checks Removed by Lean 4  | Throughput (1024³) |
|:---------------------------------|:--------------|:---------------------------------|-------------------:|
| Naive with views                 | Safe          | None (uses matrix views)         | ~7,225 GFLOPS      |
| Tiled with views                 | Safe          | 4 shared-memory checks           | ~9,405 GFLOPS      |
| Tiled without views (`tiled_raw`)| Safe          | 2 global + 4 shared-memory checks| ~9,390 GFLOPS      |

---

### Example 3: Warp & 2-Phase Block Reduction

In a 256-thread block reduction, 8 warps of 32 threads sum 256 elements in two phases using an 8-element shared array (`WARP_SUMS`):
- **Phase 0 (Warp Leader Stage)**: Every thread reduces across its 32-lane warp. Guarded by `if lane == 0`, only lane 0 of each warp writes its warp sum to `WARP_SUMS[tid / 32]`.
- **Barrier**: All 256 threads wait at `sync_threads()`.
- **Phase 1 (Block Scan)**: Each thread scans `w = 0..8` across `WARP_SUMS[w]` to accumulate the 256-thread block sum.

```{figure} images/warp-leader-reduction.svg
:alt: Leader-guarded warp and block reduction in shared memory
:align: center
:width: 100%

In Phase 0, only lane 0 of each warp writes `WARP_SUMS[tid / 32]`, so each warp slot has a single writer. After the barrier, Phase 1 only reads.
```

Notice that the kernel uses the **exact same three verification clauses** as 2D tiled GEMM (`no_oob_idx`, `no_data_race`, `uniform_barriers`):

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(
    domain = 1,
    coordinates = u32,
    block = (256, 1, 1),
    requires = (data.len() >= out.len()),
)]
#[verify(
    no_oob_idx(WARP_SUMS),
    no_data_race(WARP_SUMS),
    uniform_barriers,
)]
pub fn block_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;

    let idx = thread::index_1d();
    let gid = idx.get();
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id();

    let val = if gid < data.len() { data[gid] } else { 0.0 };
    let warp_sum = warp::reduce_sum_f32(val);

    // Phase 0 (Lean-verified bounds and single leader writer per warp):
    if lane == 0 {
        WARP_SUMS[tid / 32] = warp_sum;
    }

    thread::sync_threads();

    // Phase 1 (Lean-verified bounds and barrier separation):
    let mut block_sum = 0.0f32;
    let mut w = 0usize;
    while w < 8 {
        block_sum += WARP_SUMS[w];
        w += 1;
    }

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = block_sum;
    }
}
```

How Lean 4 proves this kernel (and what happens if you make a mistake):
- **Phase 0 bounds**: Because `tid < 256`, `tid / 32` is at most 7, which is strictly below 8.
- **Phase 0 race freedom**: The exporter sees that the write happens inside `if lane == 0`. Lean 4 proves that two threads with `lane == 0` and the same warp index `tid / 32` must be the exact same thread. If you forget `if lane == 0`, Lean 4 rejects the build with a counterexample (threads 0 and 1 both write slot 0).
- **Phase 1 bounds and barrier separation**: The loop bound `w < 8` stays inside the 8-element array, and `sync_threads()` separates Phase 0 writes from Phase 1 reads, erasing both shared-memory bounds checks before PTX lowering.

Inspect both exported reduction kernels or run them on the GPU with:

```console
$ cargo oxide lean-export safe_warp_reduce
$ cargo oxide run safe_warp_reduce
```

---

### Example 4: Sub-Warp Bitwise Splits & Multi-Buffer Staging

High-throughput kernels often split a 256-thread block into **16 sub-warps of 16 lanes** using bit shifts and masks (`tid >> 4` and `tid & 15`) and stage data across **three shared-memory buffers** of different sizes over three barrier phases:
- **Phase 0 (Guarded & packed stage)**: The first 64 threads each write one scale into the 64-float buffer and four packed activations into the 256-float buffer.
- **Phase 1 (Sub-warp scan & leader write)**: Every thread scans across its sub-warp's slice, reduces across 16 lanes with warp shuffles, and lets lane 0 of each sub-warp write one output into the 16-float buffer.
- **Phase 2 (Epilogue read)**: After a second barrier, the first 16 threads read the 16 sub-warp results.

```{figure} images/subwarp-multibuffer-staging.svg
:alt: Three-phase sub-warp bit-split and multi-buffer staging in shared memory
:align: center
:width: 100%

Two barriers separate guarded packing in Phase 0, sub-warp scanning and leader writes in Phase 1, and the final epilogue read in Phase 2.
```

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(
    domain = 1,
    coordinates = u32,
    block = (256, 1, 1),
    requires = (data.len() >= out.len()),
)]
#[verify(
    no_oob_idx(S_X_DIRECT, S_XS_DIRECT, S_ACT_DIRECT),
    no_data_race(S_X_DIRECT, S_XS_DIRECT, S_ACT_DIRECT),
    uniform_barriers,
)]
pub fn multibuf_subwarp_stage(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut S_X_DIRECT: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut S_XS_DIRECT: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut S_ACT_DIRECT: SharedArray<f32, 16> = SharedArray::UNINIT;

    let idx = thread::index_1d();
    let gid = idx.get();
    let tid = thread::threadIdx_x() as usize;
    let m_local = tid >> 4;
    let k_lane = tid & 15;
    let base_val = if gid < data.len() { data[gid] } else { 0.0 };

    // Phase 0 (Lean-verified under `if tid < 64`):
    if tid < 64 {
        S_XS_DIRECT[tid] = 0.5f32;
        let mut w = 0usize;
        while w < 4 {
            S_X_DIRECT[tid * 4 + w] = base_val;
            w += 1;
        }
    }

    thread::sync_threads();

    // Phase 1 (Lean-verified sub-warp lane scan + `if k_lane == 0` leader write):
    let mut sum = 0.0f32;
    let mut g = 0usize;
    while g < 4 {
        sum += S_X_DIRECT[k_lane * 16 + g] * S_XS_DIRECT[k_lane * 4 + g];
        g += 1;
    }
    sum += warp::shuffle_xor_f32(sum, 8);
    sum += warp::shuffle_xor_f32(sum, 4);
    sum += warp::shuffle_xor_f32(sum, 2);
    sum += warp::shuffle_xor_f32(sum, 1);

    if k_lane == 0 {
        S_ACT_DIRECT[m_local] = sum;
    }

    thread::sync_threads();

    // Phase 2 (Lean-verified under `if tid < 16` after second barrier):
    let mut final_val = sum;
    if tid < 16 {
        final_val = S_ACT_DIRECT[tid];
    }

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = final_val;
    }
}
```

How Lean 4 proves all three buffers and removes all 6 shared-memory bounds checks:
- **Phase 0 bounds and disjoint chunks**: Inside `if tid < 64`, each active thread stays below 64 on the scale buffer and owns its own 4-float chunk (`tid * 4 + w < 256`) on the activation buffer.
- **Phase 1 sub-warp bounds and single writer**: Masking with `tid & 15` keeps every lane index below 16 across all 256 threads, while `if k_lane == 0` ensures only one leader thread writes each sub-warp slot (`tid >> 4 < 16`).
- **Phase 2 barrier separation**: Two block-wide barriers separate Phase 0 writes, Phase 1 reads/writes, and Phase 2 reads. Widening the guard from 64 to 128 or dropping the second barrier is rejected at compile time.

Inspect both sub-warp kernels or run them on the GPU with:

```console
$ cargo oxide lean-export safe_subwarp_stage
$ cargo oxide run safe_subwarp_stage
```

---

### Example 5: Asynchronous TMA Bulk Copy & Barrier Ordering

On Hopper and Blackwell GPUs, global-to-shared copies run asynchronously on the **Tensor Memory Accelerator (TMA)** engine while threads coordinate through a shared-memory hardware barrier instead of a full block sync:
- **Producer (thread 0)**: Sets the expected byte count on the barrier and kicks off the 2D TMA copy into shared memory.
- **Consumer wait (all 256 threads)**: Spins on the barrier until both the thread arrival and the byte transfer finish.
- **Consumer read (all 256 threads)**: Reads from the shared tile with zero `unsafe` blocks in the kernel body.

```{figure} images/async-tma-mbarrier.svg
:alt: Asynchronous 2D TMA bulk copy and hardware barrier handoff
:align: center
:width: 100%

Thread 0 arms the byte count and launches the 2D TMA transfer; all 256 threads wait on the barrier before reading the staged tile.
```

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(
    domain = 1,
    coordinates = u32,
    block = (256, 1, 1),
)]
#[verify(
    no_oob_idx(TILE),
    no_data_race(TILE),
    uniform_barriers,
)]
pub fn async_copy_stage(
    #[grid_constant] tensor_map: &TmaDescriptor,
    mut out: DisjointSlice<f32>,
    tile_x: i32,
    tile_y: i32,
) {
    static mut TILE: SharedArray<f32, 4096, 128> = SharedArray::UNINIT;
    static mut BAR: Barrier = Barrier::UNINIT;

    let idx = thread::index_1d();
    let tid = thread::threadIdx_x() as usize;

    if tid == 0 {
        mbarrier_init(&raw mut BAR, 1);
        fence_proxy_async_shared_cta();
    }
    thread::sync_threads();

    // Producer (thread 0 kicks off the 16 KiB TMA copy):
    if tid == 0 {
        let _ = mbarrier_arrive_expect_tx(&raw const BAR, 1, 16384);
        cp_async_bulk_tensor_2d_g2s_cta(
            &raw mut TILE as *mut u8,
            tensor_map,
            tile_x,
            tile_y,
            &raw mut BAR,
        );
    }

    // All 256 threads wait until the TMA transfer completes:
    while !mbarrier_try_wait_parity(&raw const BAR, 0) {}

    // Safe read with bounds check removed:
    let val = TILE[tid];
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = val;
    }
}
```

How Lean 4 checks this handoff:
- **Single producer**: Because the TMA launch sits inside `if tid == 0`, only thread 0 starts the shared-memory write.
- **Transfer size fits**: Lean 4 checks that the 16,384-byte transfer (4,096 floats) fits inside the shared array.
- **Wait before read**: Lean 4 checks that the barrier is initialized, fenced, armed with the expected byte count, and awaited on every path before any thread reads the tile. Skipping the wait or forgetting to arm the byte count stops compilation.

Inspect the exported Lean proof or run the TMA copy on the GPU with:

```console
$ cargo oxide lean-export safe_async_copy
$ cargo oxide run safe_async_copy
```

---

### Example 6: Blackwell Tensor Cores & Tensor Memory

On datacenter Blackwell GPUs (such as NVIDIA B200), 5th-generation Tensor Cores accumulate into dynamically allocated **Tensor Memory (TMEM)** instead of regular registers:
- **Allocation (warp 0)**: Warp 0 allocates 128 columns of Tensor Memory and stores the base address in shared memory.
- **TMA + Tensor Core issue (thread 0)**: Thread 0 loads the input tiles via TMA, issues the matrix multiply into Tensor Memory, and commits completion to a barrier.
- **Epilogue & free (all 4 warps)**: After waiting on the barrier, each warp reads its own 32-lane slice out of Tensor Memory, syncs across the block, and lets warp 0 free the 128 columns, all with zero `unsafe` blocks in the kernel.

```{figure} images/blackwell-tcgen05-tmem.svg
:alt: Blackwell Tensor Core and Tensor Memory lifecycle verification
:align: center
:width: 100%

Warp 0 allocates 128 Tensor Memory columns, thread 0 issues and commits the Tensor Core multiply, each warp reads its own 32-lane slice, and warp 0 frees the columns after a block sync.
```

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(128)]
#[launch_contract(
    domain = 1,
    coordinates = u32,
    block = (128, 1, 1),
)]
#[verify(
    no_oob_idx(SMEM_A, SMEM_B, TMEM_ADDR),
    no_data_race(SMEM_A, SMEM_B, TMEM_ADDR),
    uniform_barriers,
    tmem_lifecycle
)]
pub fn tcgen05_matmul_verified(
    #[grid_constant] a_tmap: &TmaDescriptor,
    #[grid_constant] b_tmap: &TmaDescriptor,
    mut out: DisjointSlice<f32, LinearTiles<128>>,
    tile_x: i32,
    tile_y: i32,
) {
    static mut SMEM_A: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
    static mut SMEM_B: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
    static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
    static mut TMA_A_BAR: Barrier = Barrier::UNINIT;
    static mut TMA_B_BAR: Barrier = Barrier::UNINIT;
    static mut MMA_BAR: Barrier = Barrier::UNINIT;

    let idx32 = thread::index_1d_u32(lc);
    let tid = thread::threadIdx_x() as usize;
    let warp_id = warp::warp_id();

    if tid == 0 {
        mbarrier_init(&raw mut TMA_A_BAR, 1);
        mbarrier_init(&raw mut TMA_B_BAR, 1);
        mbarrier_init(&raw mut MMA_BAR, 1);
        fence_proxy_async_shared_cta();
    }
    thread::sync_threads();

    if warp_id == 0 {
        tcgen05_alloc(&raw mut TMEM_ADDR[0], 128);
    }
    thread::sync_threads();
    let tmem_addr = TMEM_ADDR[0];
    // ... TMA loads, Tensor Core MMA + commit, barrier wait, and TMEM load ...
    thread::sync_threads();
    if warp_id == 0 {
        tcgen05_dealloc(tmem_addr, 128);
    }
}
```

Inspect the generated Lean proof or build and run on Blackwell:

```console
$ cargo oxide lean-export safe_tcgen05_matmul
$ cargo oxide run safe_tcgen05_matmul --arch sm_100a
```

---

### Example 7: Warp-Specialized Ring Buffers

Fast persistent kernels rarely stop the whole block at a single barrier. Instead, they split warps by role: one **producer warp** streams tiles into a multi-stage ring buffer, while **consumer warps** compute on earlier stages at the same time, coordinating through arrays of barriers:

```{figure} images/warp-specialized-ring.svg
:alt: Warp-specialized 4-stage producer and consumer ring buffer
:align: center
:width: 100%

Warp 4 fills ring slots ahead of Warps 0..3; full-barrier and empty-barrier arrays keep each slot's writes and reads ordered across laps.
```

What Lean 4 checks in the warp-ring example (erasing all 9 bounds checks with zero `unsafe` blocks):
- **Ring wrap stays in bounds**: Whether you advance the stage counter with an `if` wraparound, `% 4`, or `& 3`, Lean 4 proves the stage index stays below 4 on every lap. An off-by-one wrap like `== 5` is caught at compile time.
- **Matching barrier and data slots**: Waiting on slot `s` of the full-barrier array only unlocks slot `s` of the ring buffer. Waiting on slot `s` and accidentally reading `(s + 1) & 3` is rejected as a data race.
- **No overwriting live data**: If the producer loop forgets to wait on the empty barrier before refilling a slot on the next lap, Lean 4 rejects the loop.

```console
$ cargo oxide run safe_warp_ring
```

---

### Example 8: 2-CTA Clusters & Hardware Work-Stealing

On Blackwell, two neighboring thread blocks can team up as a **2-CTA cluster** to share tiles across Distributed Shared Memory (DSMEM) and pull work dynamically from hardware **Cluster Launch Control (CLC)**:

```{figure} images/two-cta-cluster-clc.svg
:alt: Two-CTA cluster DSMEM exchange and hardware work-stealing
:align: center
:width: 100%

Two partner blocks synchronize across the cluster before reading each other's shared memory, while hardware work-stealing delivers the next tile into a verified 16-byte buffer.
```

- **Safe peer rank**: In a 2-block cluster, your block rank is 0 or 1, so flipping the low bit (`my_rank ^ 1`) always points to your valid partner block. Lean 4 proves that peer rank is in bounds, and rejects bugs like `my_rank + 1` (which walks out of bounds on block 1).
- **Cluster-wide syncs**: A local block sync only waits for your own block. Lean 4 checks that you run a full cluster sync before reading your partner block's shared memory or freeing shared 2-CTA Tensor Memory.
- **Work-stealing mailbox**: Hardware work-stealing writes a 16-byte response asynchronously. Lean 4 checks that the response buffer is at least 16 bytes wide and that threads wait on its completion barrier before reading the stolen tile ID.

```console
$ cargo oxide run safe_cluster_clc
```

---

### Example 9: Speed-of-Light Blackwell GEMM

The final example puts every piece together in a full production FP16/BF16 matrix multiply for NVIDIA B200 (6 warps per block across a 2-block cluster):
- **Warp 4 (Producer + Work-Stealing)**: Streams A and B tiles into a 7-stage, 224 KiB shared-memory ring using 2-CTA multicast TMA, overlaps hardware work-stealing for the next tile, and posts tile coordinates to a 4-slot mailbox.
- **Warp 5 (2-CTA Tensor Cores)**: Issues cooperative 2-CTA Tensor Core instructions into two alternating 256-column halves of a 512-column Tensor Memory buffer.
- **Warps 0..3 (Epilogue)**: Unload Tensor Memory into registers, release the Tensor Memory stage early so the next tile can start, shuffle BF16 pairs across lanes, and write 128-bit stores to global memory.

```{figure} images/verified-blackwell-sol.svg
:alt: Verified 2-CTA Blackwell GEMM pipeline in safe_gemm_sol_final
:align: center
:width: 100%

Lean 4 verifies all seven shared-memory and barrier arrays, partner CTA ranks, work-stealing responses, and 512-column Tensor Memory stages at compile time, erasing all 27 bounds checks with zero register spills.
```

All 224 KiB of shared memory, all 11 hardware barriers, cross-block DSMEM signaling, hardware work-stealing, and all 512 columns of Tensor Memory run without `unsafe` (leaving only one 4-line `unsafe` block at the very end for the 128-bit vectorized global output store on dynamically stolen tile coordinates).

Measured back-to-back on an NVIDIA B200 GPU, the Lean-verified kernel runs just as fast as the hand-tuned unsafe baseline:

| Problem Size | Unsafe Baseline            | Lean-Verified Safe         | Safe vs. Unsafe |
|:-------------|:---------------------------|:---------------------------|:----------------|
| 4096^3       | 1,855.5 TFLOPS (74.1 us)   | 1,858.4 TFLOPS (74.0 us)   | 100.15%         |
| 8192^3       | 2,081.4 TFLOPS (528.3 us)  | 2,082.3 TFLOPS (528.0 us)  | 100.04%         |
| 16384^3      | 2,118.1 TFLOPS (4.15 ms)   | 2,118.4 TFLOPS (4.15 ms)   | 100.01%         |

Build and benchmark on B200 with:

```console
$ cargo oxide build safe_gemm_sol_final --arch sm_100a
$ cargo oxide run safe_gemm_sol_final --arch sm_100a
```
