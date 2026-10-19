# select_unpredictable

Regression test for lowering `core::hint::select_unpredictable`.

`core::hint::select_unpredictable(cond, a, b)` (stable since 1.88) is the
branchless form of `if cond { a } else { b }`. It bottoms out in the
`core::intrinsics::select_unpredictable` intrinsic, which the device backend did
not recognize:

```
error: [rustc_codegen_cuda] Device codegen failed: ...
  Unsupported construct: rustc intrinsic `std::intrinsics::select_unpredictable`
  is not yet supported on the device
```

libcore reaches it pervasively from branchless helpers (slice sorting, `Ord`
combinators), so any non-trivial `no_std` port trips it. The fix lowers the
intrinsic to an `llvm.select`; the "unpredictable" branch-weight hint carries no
device semantics and is dropped.

## Run

```
cargo oxide run select_unpredictable
```

The kernels compute elementwise `max` and `min` two ways — with
`select_unpredictable` and with a plain `if` — and the host asserts they agree.
The module also includes a compile-only raw-pointer kernel. That case makes
libcore instantiate both its raw-pointer selects and its final
`MaybeUninit<*mut T>` select, so the pipeline checks nested pointer-kind
preservation rather than only equal-width machine values.
