# slice_index_bounds_check

Regression test for issue #396: 
`src[j]` inside a `#[kernel]` must keep its bounds check.
When `j` is out of range the kernel must trap or skip the access. It must never actually read out of bounds.

`j` is loaded from device memory, so the compiler cannot prove `j < src.len()`.
Under Rust semantics the access must be bounds-checked at runtime. 
The check lowers to `mir.assert`, whose failure path must stay observable (`llvm.trap` → PTX `trap`) all the way through `opt -O2`. 
When the failure path is a bare `llvm.unreachable`, 
SimplifyCFG rewrites the check  into `llvm.assume` and the kernel performs a silent out-of-bounds global read.

## Tests

- `in_bounds_gather`: gathers through a permutation of valid indices and verifies every element.
- `out_of_bounds_gather`: plants one index ~4 MB past the end of `src`. 
  The launch must either trap (bounds check fired) or leave the output sentinel untouched (thread exited at the check). 
  A `CUDA_ERROR_ILLEGAL_ADDRESS` failure means the out-of-bounds load reached memory.

Run:

```
cargo oxide run slice_index_bounds_check
```

Prints `SUCCESS` when both tests pass.
