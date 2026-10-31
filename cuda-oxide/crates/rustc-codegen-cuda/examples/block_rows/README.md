# block_rows

Block-per-row LayerNorm through `DisjointSlice<f32, BlockRows>`, with no raw
indexing and no `unsafe` on the output.

## What this tests

Block `b` owns row `b`, and thread `t` owns columns `t`, `t + blockDim.x`,
and so on. `block_row32(lc)` hands the block its row, and `thread_columns()`
walks the thread's columns. The host binds the row width with
`cuda_host::RowWidth`, so the kernel takes it from the launch packet rather
than as an argument.

Two kernels:

  - `layernorm` reads `x: &[f32]`, reduces each row with `block_reduce`, and
    writes `out` through the column witnesses.
  - `layernorm_inplace` normalizes one buffer over itself, reading and
    writing each element through the same witness (`y.read()`, `y.write()`).

The row width, 1000, is not a multiple of the 256-thread block, so every
thread's last column pass is partial. The output buffer ends in a 5-element
partial row and the grid launches one block past it. Both of those blocks get
`None` from `block_row32` and return together before the block reduction, and
the host checks that the partial row still holds its sentinel.

## Usage

```bash
cargo oxide run block_rows
```

To run the repository smoketest:

```bash
cuda-oxide/scripts/smoketest.sh -x -v '^block_rows$'
```

## Expected output

```text
layernorm: 37 rows of 1000 match the reference
layernorm_inplace: 37 rows of 1000 match the reference

SUCCESS: block-owned rows normalized with no raw indexing
```
