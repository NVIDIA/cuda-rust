# assume_init_uninhabited_trap

Regression test for issue #407 (assert_inhabited site):
`MaybeUninit::assume_init` on an uninhabited type inside a `#[kernel]` must trap.
When the panic path is compiled away instead, the kernel keeps running past the failure and produces wrong data.

`MaybeUninit::<Infallible>::uninit().assume_init()` reaches rustc's `assert_inhabited::<Infallible>` guard, which panics at runtime ("attempted to instantiate uninhabited type").
The guard sits behind a `flag != 0` check whose value is a kernel argument, so the compiler cannot remove the branch.
The importer lowers the uninhabited case to a trap (`nvvm.trap` → PTX `trap`).

## Test

Launches one thread with `flag = 1`.
The launch must fail with the kernel trapping (`CUDA_ERROR_LAUNCH_FAILED`).
A successful launch means the panic path was deleted.

Run:

```
cargo oxide run assume_init_uninhabited_trap
```

Prints `PASS (kernel trapped: ...)` when the trap fires.
