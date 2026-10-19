# Bounds checks survive unrolling

`#[unroll]` changes loop structure; ordinary Rust indexing must still trap
when the index is outside the slice. This example includes the original
failure: `arr[999] = 7` **before** an annotated loop lost its bounds check.

```bash
cargo oxide run unroll_bounds_check
```

The example checks accesses before, inside, and after fully and partially
unrolled loops, plus an ordinary loop as a control. It also checks division
by zero, which uses the same compiler assertion operation. All kernels run
with exactly one thread, so their mutable slices have a single owner.

Valid inputs must produce the expected values. Invalid inputs must report
`CUDA_ERROR_LAUNCH_FAILED` from the device trap; an illegal-address error or
another driver error fails the test. Each invalid launch runs in its own
process because a trap poisons the CUDA context. Partial unrolling tests
both the four-iteration group and the remainder.

GPU-less CI runs the example's `verify-code-shape.sh` automatically through
`scripts/smoketest.sh --compile-only`. The script requires a comparison,
conditional branch, trap, and store in every kernel's generated PTX. Each
kernel has only the checked operation under test as a possible trap source,
so a surviving check in another kernel cannot hide the regression.

To run that CI check locally:

```bash
scripts/smoketest.sh --compile-only unroll_bounds_check
```
