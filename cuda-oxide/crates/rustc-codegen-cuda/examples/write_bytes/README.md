# write_bytes

Regression test for lowering `core::intrinsics::write_bytes` (memset).

`ptr::write_bytes`, `mem::zeroed`, `MaybeUninit::zeroed` and `<[T]>::fill`
all reach the `write_bytes` intrinsic, which the device backend did not lower:

```
error: [rustc_codegen_cuda] Device codegen failed: ...
  Unsupported construct: rustc intrinsic `std::intrinsics::write_bytes`
  is not yet supported on the device
```

`fill` failed for every element type, not just bytes: its generic body has a
`write_bytes` branch for one-byte elements, and that branch is translated for
each instantiation. The fix lowers the intrinsic to `mir.memset` and then
`llvm.memset`, scaling the element count to bytes.

## Run

```
cargo oxide run write_bytes
```

Each kernel writes a thread-local buffer and stores a checksum of it:

| Kernel                | Route                                                         |
|-----------------------|---------------------------------------------------------------|
| `write_bytes_tail`    | `ptr::write_bytes` with a runtime byte and element count      |
| `zeroed_struct`       | `mem::zeroed` of a struct that then accumulates inputs        |
| `maybe_uninit_zeroed` | `MaybeUninit::<[u32; 8]>::zeroed` with one element overwritten |
| `fill_bytes`          | `<[u8]>::fill` over a runtime suffix                          |
| `fill_words`          | `<[u32]>::fill` over a runtime suffix                         |

The host computes the same checksums and asserts they match.
