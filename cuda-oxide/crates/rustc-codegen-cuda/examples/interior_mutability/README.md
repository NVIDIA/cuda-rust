# interior_mutability

End-to-end device-code conformance coverage for `core::cell::Cell` and
`core::cell::UnsafeCell` without allocator support.

## What this tests

The example covers two independent interior-mutability paths.

### `Cell<u32>`

A kernel creates a `#[repr(C)]` struct containing guard fields around a
`Cell<u32>`, then accesses the cell through a shared reference. It verifies:

- `Cell::get` reads the initial value;
- `Cell::set` updates the value through `&Cell<T>`;
- `Cell::replace` returns the previous value and stores the replacement;
- the adjacent guard fields remain unchanged.

This exercises the field-projection and raw-pointer path used internally by
`Cell` through `UnsafeCell`.

### `UnsafeCell<u32>`

A second kernel creates another guarded `#[repr(C)]` struct, obtains a raw
pointer with `UnsafeCell::get`, reads through that pointer, writes a new value,
and reads it back. The guard fields are checked independently.

The example intentionally uses only kernel-local storage, projections, and raw
pointers. It does not require `alloc`, `Box`, `Vec`, `Rc`, `Arc`, or `RefCell`.

## Usage

```bash
cargo oxide run interior_mutability
CUDA_OXIDE_NO_OPT=1 cargo oxide run interior_mutability
```

## Expected output

```text
=== interior_mutability ===
PASS: Cell::get
PASS: Cell::set
PASS: Cell::replace
PASS: Cell guard fields preserved
PASS: UnsafeCell::get raw-pointer read/write
PASS: UnsafeCell guard fields preserved
PASS: interior_mutability
```
