# slice_from_raw_parts_cast

Regression test for `slice::from_raw_parts`/`from_raw_parts_mut` over a
reinterpret-cast pointer.

Building `&mut [(u64, u64)]` from a `*mut u64` via `p as *mut (u64, u64)` left
the fat pointer's data operand typed to the pre-cast pointee (`*mut u64`) while
the slice element type is `(u64, u64)`, so `mir.construct_slice` failed to
verify:

```
Lowering failed: MirConstructSliceOp data pointer pointee mismatch.
  Expected: mir.tuple <[u64, u64], ...>, Actual: builtin.integer ui64
```

The fix coerces the data pointer to the slice element type at the
`from_raw_parts` site (a `PtrToPtr` cast preserving address space and
mutability; a no-op when the pointee already matches).

## Run

```
cargo oxide run slice_from_raw_parts_cast
```

Thread 0 reinterprets a `2*n`-element `u64` buffer as `n` `(u64, u64)` pairs via
`buf.as_mut_ptr() as *mut (u64, u64)` + `from_raw_parts_mut`, then bumps both
lanes of every pair; the host checks the reinterpreted layout.
