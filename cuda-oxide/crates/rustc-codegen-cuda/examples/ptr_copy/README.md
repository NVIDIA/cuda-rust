# ptr_copy

Regression test for lowering `core::ptr::copy` (the overlap-safe move).

Unlike `copy_nonoverlapping` — which reaches MIR as a
`StatementKind::Intrinsic(CopyNonOverlapping)` and already lowers to
`llvm.memcpy` (see the `copy_nonoverlapping` example) — `core::ptr::copy`
bottoms out in the `core::intrinsics::copy` intrinsic *call*, which the device
backend did not lower:

```
error: [rustc_codegen_cuda] Device codegen failed: ...
  Unsupported construct: rustc intrinsic `std::intrinsics::copy`
  is not yet supported on the device
```

libcore reaches it from `ptr::swap`, slice rotates, and similar routines. The
fix lowers it to the overlap-safe `llvm.memmove`.

## Run

```
cargo oxide run ptr_copy
```

Thread 0 loads `input` into `out`, then shifts `out[0..n-1]` up by one with
`ptr::copy` (`dst = src + 1`, a forward-overlapping move a plain memcpy would
corrupt). The host asserts `out[0] == input[0]` and `out[k] == input[k-1]`.
