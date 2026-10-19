# disjoint_from_raw_parts

Building a `DisjointSlice` inside a kernel, for both index-space shapes.

## What this tests

`DisjointSlice::from_raw_parts` writes a struct literal. The literal usually
folds into its use before import, and it survives whenever the slice crosses a
call the optimiser keeps, such as a `#[device]` helper taking
`&mut DisjointSlice`. The importer then met an aggregate whose translated type
is the slice's own rather than a struct, took it for a scalar-lowered ADT, and
refused it (issue #667):

```text
Unsupported construct: Scalar-lowered ADT expected exactly one runtime field, found 0
```

Two kernels cover the shapes that differ in construction:

  - `increment_from_raw_parts` builds the two-word slice, whose index space
    carries no runtime layout.
  - `scale_row_width_slice` builds the three-word form, where the row width
    read at every access site is the third operand.

The row width is 37, which is neither a power of two nor a multiple of the
block width, so a width written into the length slot resolves rows somewhere
visible instead of aliasing back onto the right element.

## Usage

```bash
cargo oxide run disjoint_from_raw_parts
```

To run the repository smoketest:

```bash
cuda-oxide/scripts/smoketest.sh -x -v '^disjoint_from_raw_parts$'
```

## Expected output

```text
increment_from_raw_parts: 4096 elements, exact match
scale_row_width_slice: 888 elements at row width 37, exact match
```

Each element's expected value depends on its own index, so a kernel that wrote
a constant, or that addressed the wrong element, reports a mismatch rather
than passing.
