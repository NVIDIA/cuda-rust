# slice_iterators

Device-side conformance coverage for slice iterators that carry runtime
fat-slice state.

The example covers:

- `slice::windows` with overlapping forward views;
- `slice::chunks_exact` with forward chunk iteration and a persistent remainder;
- `slice::chunks_exact_mut` with mutable forward chunks and a mutable remainder.

The regression intentionally stays on forward iterator paths. It does not cover:

- reverse/from-end iterator methods such as `next_back` or `nth_back`;
- dedicated MIR `Subslice` projection regressions;
- local-array iterator scalarization;
- `as_chunks` / `as_rchunks` and their `exact_div` lowering.

Run:

```text
cargo oxide run slice_iterators
CUDA_OXIDE_NO_OPT=1 cargo oxide run slice_iterators
```
