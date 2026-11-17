# Provably Safe Kernels

On a CPU, **Safe Rust** means your program has zero undefined behavior:
- No out-of-bounds memory accesses
- No data races between threads
- No dangling pointers

Notice what Safe Rust does *not* mean: it does not prove **functional correctness** (that your math formula is right). Writing `a[i] - b[i]` when you meant `a[i] + b[i]` is a logic bug, but it is still 100% memory-safe Rust.

On a GPU, thousands of threads run the same kernel function at once, and the Rust borrow checker cannot reason about thread coordinates or barriers:
- **What Rust types already cover**: [Safe Kernels Using Rust's Type System](the-safety-model.md) proves one-thread-per-element writes with disjoint slices, and [Bounds Checks](bounds-checks.md) hoists row and column checks with matrix views.
- **Where Rust types stop**: When 256 threads share an on-chip tile across barriers, the borrow checker cannot tell if two threads collide or if a barrier sits inside an `if` branch. Until now, those kernels needed `unsafe` blocks or unchecked indexing flags.
- **How we make them provably safe**: The `#[verify(...)]` attribute pairs Rust's type system with compile-time **Lean 4** safety proofs, turning "safe(ish) SIMT" into **Provably Safe SIMT**.

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
| `no_oob_idx`        | Every buffer read and write index stays below its length  | Values stored at those indices (`NaN`)    |
| `no_data_race`      | No two threads conflict on shared memory in the same step | Shared-memory bank conflicts (speed only) |
| `disjoint_writes`   | Two different threads never write to the same output slot | Which formula is written (`a - b` vs `+`) |
| `writes_all_elems`  | Every output slot is written by a launched thread         | Whether the written value is right        |
| `uniform_barriers`  | Every thread in the block reaches every barrier           | Waiting across different thread blocks    |
| `tmem_lifecycle`    | Valid `TMEM` bounds, warpgroup slices, and async order    | Floating-point Tensor Core rounding       |

---

## 2. How It Works

Verification splits the work between two places so GPU threads pay zero runtime cost:
1. **At launch time on the CPU**: The host launcher checks buffer sizes and block dimensions once before launching the kernel.
2. **At compile time in Lean 4**: The compiler pauses at `PostMem2Reg` (after promoting variables to SSA registers in `dialect-mir`), hands the kernel's indexing structure to Lean 4, and deletes the proven bounds checks before emitting PTX.

```{figure} images/specify-and-verify-pipeline.svg
:alt: Specify and Verify pipeline
:align: center
:width: 100%

The compiler pauses in memory at `PostMem2Reg`, proves the kernel's obligations in Lean 4, erases the proven bounds checks, and resumes lowering to PTX.
```

### Compile Time vs. Runtime

| Piece                 | Compile Time (Lean 4)                       | Runtime (Host CPU)                  | Runtime (GPU Threads)     |
|:----------------------|:--------------------------------------------|:------------------------------------|:--------------------------|
| Verification clauses  | Proves bounds, race freedom, and barriers   | Nothing                             | Bounds checks removed     |
| Block dimensions      | Taken as a given fact; stamped into PTX     | Checked once when preparing launch  | Hardware enforces block   |
| Length requirements   | Taken as a given assumption in the proof    | Checked once right before launch    | Nothing (already checked) |
| Grid coverage         | Assumed in `writes_all_elems` proof         | Checks `out.len() <= grid * block`  | Nothing (already checked) |

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

### We Summarize the Kernel Instead of Translating Every Line

The Lean exporter has two modes:
- **Scalar mode**: Translates a pure integer function line-by-line into fixed-width bitvector math.
- **Kernel mode**: Walks the kernel's control-flow graph using two buckets of state (**Bucket 1**: an SSA map that builds compositional index expressions `IdxExpr` and branch conditions `GuardExpr`, and **Bucket 2**: CFG lists tracking each basic block's barrier phase and dominating guards) to extract a **kernel summary** and emit one theorem per verification clause. Floating-point math is ignored because it cannot change array indices.

How one `no_oob_idx` clause maps to the right reusable proof in Lean:
- **1D global slice**: proved from the `if` guard and the host length check.
- **2D global row or column strip**: proved from the strip start check and loop bound.
- **2D shared memory tile**: proved from the 2D thread coordinates (`tx < 16`, `ty < 16`) and loop bound (`i < 16`).
- **Shared warp-leader slot and scan**: proved from the block size (`tid < 256` implies `tid / 32 < 8`) and scan loop bound (`w < 8`).
- **Sub-warp bit-splits and packed multi-buffer staging**: proved from the bit-split bijection (`(tid >> 4) * 16 + (tid & 15) = tid`), active-thread guards (`if tid < 64`), and packed loop bounds (`w < 4` implies `tid * 4 + w < 256`).
- **Asynchronous TMA bulk copies and `mbarrier` token handoffs**: proved from single-leader TMA issue (`if tid == 0`), expected-byte arrival (`mbarrier_arrive_expect_tx`), proxy coherence (`fence_proxy_async_shared_cta`), and dominating `mbarrier_try_wait_parity` completion before consumer reads.
- **Blackwell `tcgen05` Tensor Cores and `TMEM` lifecycle (`tmem_lifecycle`)**: `lean-exporter` transcribes the kernel's basic-block control-flow graph (`ControlFlowCfg`), and Lean 4 proves full-warp `tcgen05_alloc` / `tcgen05_dealloc` symmetry, `mma -> commit -> wait -> ld -> load_wait` ordering, `128`-lane and `allocCols <= 512` bounds, and warpgroup 32-lane `tcgen05_ld` slice bounds.

### Trusted Computing Base

Lean 4's kernel checks every proof obligation and rejects any proof that uses `sorryAx` or custom axioms. Everything outside that proof check is part of the **Trusted Computing Base (TCB)**:
- **MIR import and SSA promotion**: Translating Rust into MIR and promoting stack locals to SSA registers.
- **The Lean translator**: Extracting the kernel summary and proof obligations from MIR (fails closed on unsupported ops, unguarded accesses, or unknown attributes).
- **Downstream lowering**: Erasing proven bounds checks, unrolling loops, and lowering MIR to LLVM and PTX after verification succeeds.
- **Host launch checks and CUDA runtime**: Enforcing launch-contract block dimensions and buffer length relations before the kernel runs.

---

## 3. Reading Verification Errors

When a kernel has a bug, compilation stops before PTX is generated:
- **Missing guards** (such as reading a slice outside an `if` check, or using `<=` instead of `<`) are caught immediately during the MIR walk.
- **Index math or barrier bugs** (such as forgetting a host length check, indexing `a[idx + 1]`, or forgetting a barrier between a shared write and read) fail in Lean 4's arithmetic solver, which prints a counterexample:

```text
error: omega could not prove the goal:
a possible counterexample may satisfy the constraints
  b - c ≥ 1
  a - b ≥ 0
  a - c ≤ 1
where
  a := ↑(lengths ⟨0, ...⟩)                 -- length of input a
  b := ↑(lengths ⟨2, ...⟩)                 -- length of output c
  c := ↑(Mir.globalIndex 256 block lane)   -- thread index
```

How to read that error in three quick steps:
1. **Read the `where` legend at the bottom first**: Lean renames expressions to `a`, `b`, and `c`. Here `a` is the length of `a`, `b` is the length of `c`, and `c` is the thread index.
2. **Plug those names into the inequalities**:
   - `b - c >= 1` means the thread passed the `if` check (`idx < c.len()`).
   - `a - b >= 0` means the host check passed (`a.len() >= c.len()`).
   - `a - c <= 1` means `idx + 1 >= a.len()`, so `a[idx + 1]` walks off the end of `a`!
3. **Test with simple numbers**: If both slices have length 10, thread 9 passes `9 < 10` and then reads `a[9 + 1] = a[10]`, which is out of bounds.

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

Because Lean 4 proves `no_oob_idx`, `no_data_race`, and `uniform_barriers` at compile time:
- **Both `unsafe` blocks in `sgemm_tiled_views` are gone**: In `gemm_views`, Phase 0 staging and Phase 1 scanning each needed an `unsafe { ... }` block around `TILE_A` and `TILE_B`. Adding `#[verify(no_oob_idx(TILE_A, TILE_B), no_data_race(TILE_A, TILE_B), uniform_barriers)]` removes both `unsafe` blocks and elides all 4 shared-memory bounds checks.
- **Plain slice indexing is now safe too**: Even without matrix view types (`MatrixView32`), `sgemm_tiled_raw` is now 100% safe Rust and elides all 6 bounds checks (2 on global slices `a` and `b`, plus 4 on shared tiles `TILE_A` and `TILE_B`).

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
| Tiled without views (`tiled_raw`)| **Now Safe!** | 2 global + 4 shared-memory checks| ~9,390 GFLOPS      |

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

High-throughput kernels often split a 256-thread 1D block into **16 sub-warps of 16 lanes** using bitwise shifts and masks (`m_local = tid >> 4`, `k_lane = tid & 15`) and stage data across **multiple shared-memory buffers** of different sizes (`S_X_DIRECT`, `S_XS_DIRECT`, and `S_ACT_DIRECT`) over three barrier phases:
- **Phase 0 (Guarded & Packed Stage)**: Under `if tid < 64`, each active thread writes one scale into `S_XS_DIRECT[tid]` (`64` floats) and four packed activations into `S_X_DIRECT[tid * 4 + w]` (`256` floats).
- **Phase 1 (Sub-Warp Scan & Leader Write)**: Every thread scans `S_X_DIRECT[k_lane * 16 + g]` and `S_XS_DIRECT[k_lane * 4 + g]` (`k_lane < 16`), reduces across its 16-lane sub-warp, and under `if k_lane == 0` writes `S_ACT_DIRECT[m_local]` (`16` floats).
- **Phase 2 (Epilogue Read)**: After a second `sync_threads()`, threads under `if tid < 16` read `S_ACT_DIRECT[tid]`.

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
- **Guarded and packed bounds (`Phase 0`)**: The exporter tracks `if tid < 64` as the active thread bound for Phase 0, proving `tid < 64` for `S_XS_DIRECT` and `tid * 4 + w < 64 * 4 = 256` for `S_X_DIRECT`. Because `w < 4`, distinct threads own disjoint 4-element segments (`tid₁ * 4 + w₁ = tid₂ * 4 + w₂` implies `tid₁ = tid₂`).
- **Sub-warp bit-split bounds and leader exclusivity (`Phase 1`)**: Because `k_lane = tid & 15 < 16` for every thread, `k_lane * 16 + g < 256` and `k_lane * 4 + g < 64` hold across all 256 threads, while `if k_lane == 0` guarantees a single writer per sub-warp slot `m_local = tid >> 4 < 16`.
- **Three-phase barrier separation (`Phase 0 -> 1 -> 2`)**: Two uniform `sync_threads()` barriers separate Phase 0 writes, Phase 1 reads/writes, and Phase 2 reads. Widening `if tid < 64` to `if tid < 128` or dropping the second barrier is rejected by Lean 4 at compile time.

Inspect both Stage 4 kernels (`subwarp_bit_split_stage` and `multibuf_subwarp_stage`) or run them on the GPU with:

```console
$ cargo oxide lean-export safe_subwarp_stage
$ cargo oxide run safe_subwarp_stage
```

---

### Example 5: Asynchronous TMA Bulk Copy & `mbarrier` Token Ordering

On Hopper and Blackwell GPUs, global-to-shared transfers run asynchronously on the **Tensor Memory Accelerator (TMA)** hardware engine while threads synchronize through a 64-bit shared-memory **`mbarrier`** (`BAR`) instead of `sync_threads()`:
- **Producer (`if tid == 0`)**: Sets the expected arrival and byte counts via `mbarrier_arrive_expect_tx(&raw const BAR, 1, TILE_BYTES)` and launches `cp_async_bulk_tensor_2d_g2s_cta(&raw mut TILE as *mut u8, tensor_map, tile_x, tile_y, &raw mut BAR)`.
- **Consumer Wait (all 256 threads)**: Spin-waits on `while !mbarrier_try_wait_parity(&raw const BAR, 0) {}` until both the arrival count and byte count hit zero.
- **Consumer Read (all 256 threads)**: Reads `TILE[tid]` with **zero `unsafe` blocks** in the kernel body.

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

    // Producer (`if tid == 0` mints token `StageProduced(TILE, BAR)`):
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

    // Token handoff (`StageProduced(TILE, BAR)` -> `StageReady(BAR)`):
    while !mbarrier_try_wait_parity(&raw const BAR, 0) {}

    // Consumer read dominated by `StageReady(BAR)` (0 `unsafe` blocks!):
    let val = TILE[tid];
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = val;
    }
}
```

How `lean-exporter` and Lean 4 verify the asynchronous token chain:
- **Single-leader producer (`no_data_race_single_leader`)**: Because the TMA copy is guarded by `if tid == 0`, Lean 4 proves only thread 0 issues the write to `TILE`.
- **Transfer byte bounds (`no_oob_idx(TILE)`)**: The exporter extracts the expected byte count from `mbarrier_arrive_expect_tx` (`16384` bytes = `4096` floats) and proves in Lean 4 that the transfer fits inside `TILE`'s capacity.
- **Token-ordered phase separation (`StageProduced` -> `StageReady`)**: The exporter checks that `BAR` is armed by `mbarrier_arrive_expect_tx`, bound to `TILE` by `cp_async_bulk_tensor_2d_g2s_cta`, and awaited on all paths to `TILE[tid]` via `mbarrier_try_wait_parity(&raw const BAR, 0)`. Omitting `mbarrier_arrive_expect_tx`, skipping the wait, or waiting on a different barrier leaves `TILE[tid]` in the producer phase and is rejected by Lean 4 at compile time.

Inspect the exported Lean proof or run the TMA copy on the GPU with:

```console
$ cargo oxide lean-export safe_async_copy
$ cargo oxide run safe_async_copy
```

---

### Example 6: Blackwell `tcgen05` Tensor Cores & `TMEM` Lifecycle

On datacenter Blackwell GPUs (`sm_100a`, such as NVIDIA B200), 5th-generation Tensor Cores (`tcgen05.mma`) accumulate into dynamically allocated **Tensor Memory (`TMEM`)**:
- **Warp 0 Allocation (`if warp_id == 0`)**: Collectively allocates `128` `TMEM` columns into `TMEM_ADDR[0]` via `tcgen05_alloc(&raw mut TMEM_ADDR[0], 128)`.
- **2D TMA + Single-Thread MMA (`if tid == 0`)**: Stages `A` and `B` into `SMEM_A` and `SMEM_B`, builds shared-memory descriptors via `cvta_generic_to_shared_offset` and `Tcgen05SmemDescriptor::for_k_major`, issues `tcgen05_mma_f16`, and signals `MMA_BAR` via `tcgen05_commit_shared_cluster`.
- **Warp-Partitioned `TMEM` Epilogue & Deallocation**: After `mbarrier_try_wait_parity(&raw const MMA_BAR, 0)`, each warp `0..4` reads its own 32-lane slice `(warp_id * 32) << 16` via `tcgen05_ld_32x32b_x1_raw` + `tcgen05_load_wait()`, syncs at `thread::sync_threads()`, and frees all `128` columns via `tcgen05_dealloc(tmem_addr, 128)` with **zero `unsafe` blocks** in the kernel.

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
    // ... TMA loads, tcgen05_mma_f16 + tcgen05_commit_shared_cluster,
    //     mbarrier_try_wait_parity, tcgen05_ld_32x32b_x1_raw + tcgen05_load_wait ...
    thread::sync_threads();
    if warp_id == 0 {
        tcgen05_dealloc(tmem_addr, 128);
    }
}
```

Inspect the transcribed `ControlFlowCfg` and Lean 4 theorems, or build and run on Blackwell (`sm_100a`):

```console
$ cargo oxide lean-export safe_tcgen05_matmul
$ cargo oxide run safe_tcgen05_matmul --arch sm_100a
```
