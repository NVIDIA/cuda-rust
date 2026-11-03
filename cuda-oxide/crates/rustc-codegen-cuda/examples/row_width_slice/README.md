# row_width_slice

Regression tests for the runtime row width carried inside
`DisjointSlice<T, Runtime2DIndex>`:

1. **Nonzero width readback**: the row width bound on the host via
   `cuda_host::RowWidth` must reach every device thread. An entry prologue
   that drops the third kernel parameter compiles and runs while giving
   every thread width 0; checking a nonzero value catches that.
2. **Two-width witness mixing**: witnesses minted from two slices with
   different row widths and selected under a thread-varying condition must
   still resolve against the addressed slice's own width, keeping every
   thread on its own cell.
3. **By-value runtime-width slice across a non-inlined call**: the internal
   call ABI must marshal all three fields (ptr, len, width) to match the
   three-parameter callee signature.

Run:

```bash
cargo oxide run row_width_slice
```
