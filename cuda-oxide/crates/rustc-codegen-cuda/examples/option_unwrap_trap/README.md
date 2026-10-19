# option_unwrap_trap

Regression test for issue #407:
`Option::unwrap` on a runtime `None` inside a `#[kernel]` must trap.
When the panic path is compiled away instead, the kernel keeps running past the failure and produces wrong data.

`j` is a kernel argument, so the compiler cannot prove `j < src.len()`.
`src.get(j)` returns `None` at runtime and `.unwrap()` calls the diverging `unwrap_failed`.
The panic machinery cannot run on the GPU, a trap must take its place (`nvvm.trap` → PTX `trap`).

## Test

Launches one thread with `j = 1_000_000` against a 4-element `src`.
The launch must fail with the kernel trapping (`CUDA_ERROR_LAUNCH_FAILED`).
A successful launch means the panic path was deleted.

Run:

```
cargo oxide run option_unwrap_trap
```

Prints `PASS (kernel trapped: ...)` when the trap fires.
