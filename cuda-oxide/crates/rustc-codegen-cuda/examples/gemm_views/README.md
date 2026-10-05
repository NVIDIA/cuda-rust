# GEMM through read-side proof-carrying views

Matrix multiply, `C = alpha * A * B + beta * C`, with the sizes `m`, `n`,
`k` as ordinary runtime arguments. This example shows safe kernels reaching
the same machine code as hand-written unsafe ones, and proves it three
ways: results, PTX structure, and a benchmark.

## The idea in one picture

A matrix lives in GPU memory as one long array, row after row. `stride` is
the row width: how many elements one row occupies. Element `(row, col)`
lives at flat index `row * stride + col`:

```text
 3 x 4 matrix, stride = 4:        one thread's slice of the work:
                                    a row of A: consecutive elements
  col:      0   1   2   3           a column of B: elements one row
  row 0:  [ 0   1   2   3 ]                        width apart
  row 1:  [ 4   5   6   7 ]
  row 2:  [ 8   9  10  11 ]       row(1) covers flat 4, 5, 6, 7
                                  col(2) covers flat 2, 6, 10
```

Plain safe Rust checks every single `a[i]` against the array length. The
views check once, up front, that the whole strip a thread will read lies
inside the array:

```text
ordinary indexing:  check, load, check, load, fma, ...     per iteration
views:              whole-row check ONCE, whole-column check ONCE, then
                    load, load, fma, advance, branch       no checks left
```

`MatrixView32::row(row, k)` is the one-time row check,
`MatrixView32::col(col, k)` the column check, and `zip_exact` verifies once
that both have equal length. After that the dot-product loop advances one
counter whose "am I done" compare is the only compare in the loop, which is
exactly the shape of a hand-written raw-pointer loop.

## What runs

Two safe/raw pairs, checked against a CPU reference:

- `sgemm_naive_views` vs `sgemm_naive_raw`: one thread per C element.
- `sgemm_tiled_views` vs `sgemm_tiled_raw`: 16x16 shared-memory staging.
  The row/column checks are hoisted before the tile loop. A thread whose
  check fails gets an *empty* view instead of returning early, because
  every thread must reach both `sync_threads()` barriers (a thread that
  leaves early hangs the block). Staging reads `.get(i).unwrap_or(0.0)`,
  so an out-of-range load becomes zero fill, with no extra control flow.

Both safe kernels write C through `tile_2d32_rt`, which is safe: the row
width is not passed at the call site at all. The host binds it into C's
slice once for the launch (`cuda_host::RowWidth`), so every thread reads
the same width by construction and no call-site obligation remains.

The launch contract makes the buffer sizes part of the kernel's interface:

```rust
requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n)
```

The generated launcher checks these once per launch, on the CPU. The demo
deliberately launches once with a doubled `k` and prints the typed error it
gets back instead of a GPU fault.

## Running it

```bash
cargo oxide run gemm_views     # GPU: correctness vs CPU reference, then the
                               # 1024^3 benchmark below
gemm_views --verify-ptx        # no GPU needed: structural PTX comparison
gemm_views --bench             # GPU: benchmark only
```

(The built binary lands in
`cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_views/target/release/`.)

`--verify-ptx` checks that no compile-time contract markers leak into the
module, that no kernel contains a `trap` instruction, that the naive pair
keeps identical conditional-branch counts, and that both pairs have
identical global-memory load/store instruction sets.

`--bench` measured on an RTX 5090 (1024 x 1024 x 1024, average of 5 runs):

| kernel             | time     | GFLOPS |
| ------------------ | -------- | ------ |
| naive views (safe) | 0.298 ms | 7201   |
| naive raw (unsafe) | 0.298 ms | 7201   |
| tiled views (safe) | 0.229 ms | 9370   |
| tiled raw (unsafe) | 0.230 ms | 9337   |

These numbers depend on the pipeline disabling llc's late branch folding:
LLVM 23 started rewriting loop branches into a single negated conditional,
which ptxas's SASS unroller does not recognize, and the naive kernels lose
about a quarter of their throughput. The rationale and measurements live
on `DISABLE_BRANCH_FOLD` in `cuda-oxide/crates/cuda-oxide-codegen/src/ptx.rs`.

For scale: the `gemm` example (plain `a[i]`, checked on every read) runs
the same problem at roughly 2940 GFLOPS. Removing the per-read checks
safely is worth 2.4x on the naive kernel, and the safety itself costs about
0.1% against the unsafe twins.

The default run uses deliberately non-square sizes (`m = 128`, `n = 96`,
`k = 64`) so a mixed-up row width shows up immediately. Sizes that are not
multiples of the 16x16 block (partial tiles) are a documented non-goal of
this first runtime-size layer.
