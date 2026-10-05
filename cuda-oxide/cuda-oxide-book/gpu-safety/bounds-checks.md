# Bounds Checks: Pay Once or Opt Out

Every `a[i]` in a kernel hides a safety check. The compiler turns it into
"is `i` inside the slice? If not, stop the kernel" -- a compare and a
branch, on every access, in every thread. On a CPU you rarely notice. In a
GPU hot loop you do: the naive GEMM in the `gemm` example spends most of
its inner loop on two such checks per multiply-add and reaches about
2,940 GFLOPS on an RTX 5090. The same kernel with the checks removed runs
at 7,160 GFLOPS.

This chapter is about closing that gap without giving up safety. It builds
on [the safety model](the-safety-model.md): you already know that
`DisjointSlice` makes parallel writes race-free and that a checked
`PreparedLaunch` proves launch geometry on the host. The same idea -- check
a fact once, in the right place, and carry the proof in a type -- extends
to reads and to buffer sizes.

Three tools, from safest to bluntest:

| Tool                              | What it does                                             | Safety            |
|:----------------------------------|:---------------------------------------------------------|:------------------|
| Read views                        | Check a whole row or column once, then read check-free   | Fully safe        |
| `requires` on the launch contract | Check buffer sizes once per launch, on the CPU           | Fully safe        |
| `#[kernel(unchecked_indexing)]`   | Delete the indexing checks and trust you                 | UB if you're wrong|

---

## What a bounds check costs

A dot product between a row of A and a column of B looks harmless:

```rust
let mut sum = 0.0f32;
let mut i = 0;
while i < k {
    sum += a[row * k + i] * b[i * n + col];
    i += 1;
}
```

Count what each iteration actually executes:

```text
i < k              the loop's own exit test        (necessary)
row*k + i < a.len  bounds check, branch to trap    (overhead)
load a
i*n + col < b.len  bounds check, branch to trap    (overhead)
load b
fma
```

Two of the three compares are overhead, and they also split the loop body
into multiple basic blocks, which blocks unrolling. The compiler cannot
remove them on its own: it sees `a.len()` as an opaque number and has no
way to know that `row * k + i` stays below it. *You* know it -- because you
sized the buffers on the host -- but that fact lives in a different program.
The rest of this chapter is about handing that fact to the compiler.

---

## Read views: check a whole strip once

A matrix lives in GPU memory as one long array, row after row. The **row
width** (called `stride` in the API) is how many elements one row occupies.
Element `(row, col)` lives at flat index `row * stride + col`:

```text
 3 x 4 matrix, stride = 4:

  col:      0   1   2   3
  row 0:  [ 0   1   2   3 ]     row(1) covers flat 4, 5, 6, 7
  row 1:  [ 4   5   6   7 ]     col(2) covers flat 2, 6, 10
  row 2:  [ 8   9  10  11 ]              (a jump of 4 per step)
```

`MatrixView32` wraps a plain `&[f32]` parameter together with its row
width. Asking it for a row or a column performs **one** check -- "does the
whole strip fit inside the slice?" -- and returns a view whose reads need
no further checks:

```rust
let a_mat = MatrixView32::new(a, k);   // A is m rows, each k wide
let b_mat = MatrixView32::new(b, n);   // B is k rows, each n wide

if let Some(a_row) = a_mat.row(row, k)      // one check: A's whole row
    && let Some(b_col) = b_mat.col(col, k)  // one check: B's whole column
{
    // every read through a_row / b_col is now check-free
}
```

If a check fails you get `None` and no pointer is ever created. There is
no `unsafe` anywhere: reading cannot race, so any number of threads may
hold overlapping read views.

### Fusing the dot product: `zip_exact`

Iterating one view is already check-free: an iterator's "is there a next
element?" test *is* the loop's exit condition, which even a raw pointer
loop needs. But a dot product iterates two views, and zipping two
iterators normally checks both sides every step. `zip_exact` checks
**once** that the row and the column have the same length, then drives
both with a single counter:

```rust
if let Some(pair) = a_row.zip_exact(b_col) {
    let mut sum = 0.0f32;
    for (x, y) in pair {
        sum += x * y;
    }
}
```

The loop that comes out the other end is:

```text
load a, load b, fma, advance, compare, branch
```

-- exactly the shape of a hand-written raw-pointer loop. The
[`gemm_views`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_views)
example proves this literally: it compiles each safe kernel next to an
`unsafe` raw-pointer twin and verifies that the PTX has the same branch
counts, the same memory instructions, no traps in the loop, and
bit-identical results on the GPU. Measured at 1024x1024x1024:

| Kernel               | GFLOPS | Notes                          |
|:---------------------|-------:|:-------------------------------|
| `gemm` (plain `a[i]`)|  2,942 | two checks per multiply-add    |
| naive views (safe)   |  7,159 | 2.4x -- checks paid once       |
| naive raw (`unsafe`) |  7,161 | safety cost: about 0.1%        |

### Views and barriers

One rule when a kernel uses `sync_threads()`: every thread must reach the
barrier, so a thread whose view check failed must not return early. Fall
back to an empty view instead, and let reads degrade to a fill value:

```rust
let a_row = a_mat.row(row, k).unwrap_or(RowView32::empty());
// ... later, in the staging loop:
tile[idx] = a_row.get(i).unwrap_or(0.0);   // one compare, zero-fill at edges
```

`get` costs a single compare -- this is the accessor for edges, not for hot
loops. The tiled kernel in `gemm_views` uses exactly this pattern.

---

## Writing the result: a tile with a runtime row width

The safe way to *write* is still `DisjointSlice`, as described in
[the safety model](the-safety-model.md). What's new is
`tile_2d32_rt`: the 2D tile whose row width is a runtime value like `n`,
for output matrices whose size isn't known at compile time:

```rust
// `c` carries its own row width, bound on the host to this same `n`,
// so the call is safe: no per-thread width crosses the call.
if let Some(mut cell) = c.tile_2d32_rt(coord) {
    let previous = cell.at_const::<0, 0>().read();
    cell.at_const::<0, 0>().write(alpha * sum + beta * previous);
}
```

Why does the width come from the slice and not from an argument? The
tile-disjointness argument -- "distinct thread coordinates own
non-overlapping rectangles" -- silently assumes all threads agree on the
row width. Two threads that disagree describe two different grids over the
same memory: with 1x1 tiles, thread `(1, 0)` with width 5 resolves flat
index `1*5 + 0 = 5`, and thread `(0, 5)` with width 100 resolves
`0*100 + 5 = 5`. Same element, two `&mut`, a data race.

An earlier design took the width as an `unsafe` per-call argument, but no
call-site obligation can close that hole: safe code can select between two
"uniform" widths under a thread-varying condition. So the row width lives
in the slice instead. The host binds it once for the launch through
`cuda_host::RowWidth`, device code cannot choose it, and every thread that
addresses `c` reads the same width by construction. When the width *is*
known at compile time, use `tile_2d32` instead and a mismatch becomes a
type error.

---

## `requires`: promise the sizes at launch

The views check strips against `a.len()` at runtime because nothing else
guarantees the buffers are big enough. But the host *knows* the sizes -- it
allocated the buffers. The `requires` key on a launch contract writes that
knowledge down, next to the kernel:

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
    requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
pub fn sgemm_naive_views(m: u32, n: u32, k: u32, alpha: f32,
    a: &[f32], b: &[f32], beta: f32,
    mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
```

The generated launcher evaluates every relation once per launch, on the
CPU, before anything touches the GPU. A launch that breaks a relation
doesn't fault mid-kernel -- it returns an error that names the promise:

```text
sgemm_naive_views: size requirement `a.len() >= m * k` violated:
left-hand side is 8192, right-hand side is 16384
```

Relations are deliberately *not* arbitrary Rust. Each one is a single
comparison built from slice parameters as `name.len()`, unsigned integer
scalar parameters, integer literals, parentheses, and `+ - *`. Two reasons:

1. Every identifier is validated against the kernel's actual parameter
   list, so a typo is a compile error, not a check against the wrong
   value.
2. Every `+`, `-`, `*` compiles to checked `u64` arithmetic, so a huge
   `m * k` cannot wrap around to a small number and falsely pass.

The `_unchecked` launcher variants skip these checks, exactly as they skip
the geometry checks; upholding the relations becomes part of their
`SAFETY` contract.

---

## The escape hatch: `unchecked_indexing`

Sometimes no view fits the access pattern, and you've verified the kernel
is correct. For that case there is a blunt instrument:

```rust
#[kernel(unchecked_indexing)]
pub fn hot_loop(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d().get();
    if let Some(out) = c.get_mut(thread::index_1d()) {
        *out = a[idx] + b[idx];   // no check, no trap
    }
}
```

The flag deletes the check behind every `a[i]` in the kernel. The contract
is `get_unchecked`'s: you promise every index is in bounds, and if you're
wrong the result is undefined behavior -- no trap, possibly silent
corruption of unrelated memory. It doesn't require the `unsafe` keyword,
but treat it as if it did.

There is also a whole-build switch for experiments:

```console
$ CUDA_OXIDE_UNCHECKED_INDEXING=1 cargo oxide build my_example
$ cargo oxide build my_example --unchecked-indexing   # same thing
```

Scope, precisely:

- Only indexing checks (`a[i]`) are removed. Arithmetic overflow, division
  by zero, and every other safety check still traps. Range indexing
  (`&a[i..j]`) also still traps.
- The per-kernel flag covers that kernel's body, including functions
  inlined into it. It never leaks: a kernel that calls a flagged kernel's
  helper function keeps its own checks.
- Turn it on only after the checked build passes, ideally under
  `compute-sanitizer`. The
  [`unchecked_indexing`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/unchecked_indexing)
  example shows the PTX with and without the flag.

---

## Choosing between them

```text
Can a view express the access pattern (rows, columns, tiles, dot products)?
├── yes -> use views + a `requires` contract.  Safe, and measurably as
│          fast as raw pointers.
└── no  -> is the kernel verified correct (sanitizer, tests)?
    ├── yes -> #[kernel(unchecked_indexing)], documented like unsafe code.
    └── no  -> keep the checks. 2,900 GFLOPS with traps beats
               garbage at 7,000.
```

## Summary

- `a[i]` costs a compare and a branch per access, and in hot loops that is
  the dominant cost -- 2.4x on naive GEMM.
- Read views (`MatrixView32::row` / `col`, `zip_exact`) check a whole
  strip once and make every read inside it check-free. Fully safe, and
  the PTX matches hand-written unsafe code.
- `requires = (...)` moves buffer-size checks to the host: one typed error
  per launch instead of a mid-kernel trap.
- `#[kernel(unchecked_indexing)]` deletes indexing checks outright. Same
  contract as `get_unchecked`; out of bounds is undefined behavior.
