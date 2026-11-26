# Bounds Checks: Pay Once or Opt Out

Every `a[i]` in Rust compiles to a bounds comparison and a conditional trap branch. On a CPU branch predictor you rarely notice; inside a GPU inner loop, two extra checks per multiply-add cut naive `1024^3` SGEMM on RTX 5090 from **7,160 GFLOPS** down to **2,940 GFLOPS**.

`cuda-oxide` gives you **four ways** to remove per-element bounds checks, ordered from simplest safe APIs to the unverified escape hatch:

| Tool                               | How It Removes Hot-Loop Checks                         | Safety                       |
|:-----------------------------------|:-------------------------------------------------------|:-----------------------------|
| **Read views (`MatrixView32`)**    | Checks a whole row or column once before the loop      | Safe (pure Rust types)       |
| **`requires = (...)`**             | Checks buffer sizes once on the CPU before launch      | Safe (host launch check)     |
| **`#[verify(no_oob_idx(...))]`**   | Proves general global/shared indices in Lean 4         | Safe, preview (check erased) |
| **`#[kernel(unchecked_indexing)]`**| Deletes slice/shared bounds checks without proof       | Unsafe contract (UB if OOB)  |

---

## 1. Why Per-Access Bounds Checks Hurt GPU Loops

Consider a naive matrix-multiply inner loop:

```rust
let mut sum = 0.0f32;
let mut i = 0;
while i < k {
    sum += a[row * k + i] * b[i * n + col];
    i += 1;
}
```

Each iteration executes three comparisons instead of one:

```text
i < k               loop exit condition             (needed)
row * k + i < a.len bounds check + branch to trap   (overhead)
i * n + col < b.len bounds check + branch to trap   (overhead)
```

Those extra trap branches also split the loop body across basic blocks and block loop unrolling. Because `a.len()` is an opaque runtime parameter to `rustc`, the compiler cannot erase those checks unless you give it a strip view, a launch precondition, or a `#[verify]` proof.

---

## 2. Read Views: Check a Whole Row or Column Once

A row-major matrix of width `stride` stores `(row, col)` at flat index `row * stride + col`:

```text
3 x 4 matrix (stride = 4):
  row 0: [ 0   1   2   3 ]     row(1, 4) covers [4, 5, 6, 7]
  row 1: [ 4   5   6   7 ]     col(2, 3) covers [2, 6, 10] (step = 4)
  row 2: [ 8   9  10  11 ]
```

`MatrixView32` wraps `&[f32]` with its row stride. Requesting `row(...)` or `col(...)` performs **one** bounds check for the entire strip, and `zip_exact` drives both strips with a single loop counter:

```rust
let a_mat = MatrixView32::new(a, k);
let b_mat = MatrixView32::new(b, n);

if let Some(a_row) = a_mat.row(row, k)
    && let Some(b_col) = b_mat.col(col, k)
    && let Some(pair) = a_row.zip_exact(b_col)
{
    let mut sum = 0.0f32;
    for (x, y) in pair {
        sum += x * y;
    }
}
```

The emitted inner loop (`load a, load b, fma, advance, branch`) has zero trap branches and matches hand-written `unsafe` pointer code on `1024^3` SGEMM (`examples/gemm_views`):

| Kernel Variant          | RTX 5090 GFLOPS | Bounds Checks in Inner Loop |
|:------------------------|----------------:|:----------------------------|
| Plain `a[...]` indexing |           2,942 | 2 per FMA                   |
| `MatrixView32` (safe)   |           7,159 | 0 (checked once per thread) |
| Raw pointers (`unsafe`) |           7,161 | 0                           |

- **Rule with `sync_threads()`**: Never return early before a block barrier if a strip check fails. Use `.unwrap_or(RowView32::empty())` and `.get(i).unwrap_or(0.0)` so out-of-range edge threads zero-fill and still reach `thread::sync_threads()`.

---

## 3. `requires`: Check Buffer Sizes Once on the Host

`#[launch_contract(..., requires = (...))]` states buffer-length relations right on the kernel:

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(
    domain = 2,
    coordinates = u32,
    block = (16, 16, 1),
    requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n),
)]
pub fn sgemm_naive_views(
    m: u32, n: u32, k: u32, alpha: f32,
    a: &[f32], b: &[f32], beta: f32,
    mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
) { ... }
```

- **Evaluated once on the CPU** inside `prepare_*` / typed launch using checked `u64` arithmetic (so `m * k` cannot overflow and falsely pass).
- **Clear host error** instead of a mid-kernel GPU trap when a buffer is undersized:
  ```text
  sgemm_naive_views: size requirement `a.len() >= m * k` violated:
  left-hand side is 8192, right-hand side is 16384
  ```
- **Feeds directly into `#[verify]`**: When paired with `#[verify(no_oob_idx(...))]`, Lean 4 uses `requires` preconditions to prove in-kernel indexing safe and erase `mir.assert` checks automatically.

---

## 4. `#[verify(no_oob_idx(...))]` (preview): Erase General & Shared-Memory Checks Safely

Read views cover contiguous rows, columns, and fixed 2D tiles, while GPU kernels also index `SharedArray` tiles (`TILE[ty * 16 + i]`), sub-warp bit-splits (`S_X[tid * 4 + w]`), and multi-stage rings.

Adding `#[verify(no_oob_idx(...))]` (preview) proves those index expressions in Lean 4 at compile time and **deletes every proven `mir.assert` check before PTX lowering** with zero `unsafe`:

- On 2D tiled SGEMM (`safe_gemm_views`), Lean 4 erases all 6 global and shared-memory bounds checks (`~9,390 GFLOPS` on RTX 5090).
- On 2-CTA Blackwell GEMM (`safe_gemm_sol_final`), Lean 4 erases all 27 shared-memory and barrier-array checks (`2,118 TFLOPS` on B200, `100%+` of `--unchecked-indexing`).

See [Provably Safe Kernels (preview)](verified-kernels.md) for walkthroughs.

---

## 5. The Unverified Escape Hatch: `unchecked_indexing`

When you are prototyping an unverified kernel and accept full `get_unchecked` responsibility, `#[kernel(unchecked_indexing)]` (or `--unchecked-indexing` / `CUDA_OXIDE_UNCHECKED_INDEXING=1`) strips slice, array, and `SharedArray` bounds checks without a proof:

```rust
#[kernel(unchecked_indexing)]
pub fn hot_loop(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out) = c.get_mut(idx) {
        *out = a[i] + b[i]; // no check: OOB is undefined behavior
    }
}
```

- **Narrow scope**: Removes only `x[i]` bounds checks inside that kernel; division-by-zero, overflow, and range slicing (`&a[i..j]`) still trap.
- **Treat like `unsafe`**: Validate with `cargo oxide sanitize <target>` (`compute-sanitizer --tool memcheck`) before enabling it.

---

## Which Tool Should I Pick?

```text
Need zero bounds-check overhead in a hot loop?
├── Row, column, or fixed 1D/2D tile?     -> MatrixView32 / LinearTiles + `requires` (Safe, pure Rust)
├── SharedArray, bit-splits, or custom i? -> `#[verify(no_oob_idx(...))]`            (Safe, proved in Lean 4)
└── Unverified prototype escape hatch?    -> `#[kernel(unchecked_indexing)]`         (UB if out of bounds)
```
