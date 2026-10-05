# unaligned_memory

Regression coverage for device-side unaligned memory access, including
`#[repr(packed)]` field projections.

## What this tests

The example covers two related but distinct paths.

First, it forms a `*const u32` / `*mut u32` manually from a byte address at
`base + 1`. This pins the existing `core::ptr::read_unaligned` and
`core::ptr::write_unaligned` conformance path.

Second, it forms raw field pointers with `addr_of!` / `addr_of_mut!` from
packed aggregates:

```rust
#[repr(C, packed)]
struct PackedPacket {
    tag: u8,
    value: u32,
}
```

For this layout rustc places `value` at byte offset 1. A normal LLVM
`{ i8, i32 }` struct would place the `i32` at byte offset 4, so the compiler
must address the field using rustc's physical byte offset rather than relying
on a typed struct GEP.

The example checks:

1. Manual `read_unaligned` from `base + 1`.
2. Manual `write_unaligned` to `base + 1`, including adjacent guard bytes.
3. `#[repr(C, packed)]` field read through `addr_of!`, at rustc byte offset 1.
4. `#[repr(C, packed)]` field write through `addr_of_mut!`, at rustc byte offset 1.
5. `#[repr(C, packed(2))]` field read at rustc byte offset 2, preserving the
   stronger two-byte alignment instead of collapsing every packed access to
   alignment 1.

With cuda-oxide's pinned Rust toolchain, `read_unaligned` and
`write_unaligned` are implemented by libcore through byte-oriented
`copy_nonoverlapping`. The compiler-specific regression here is therefore the
packed field-address projection that feeds those operations.

## Usage

```bash
cargo oxide run unaligned_memory
CUDA_OXIDE_NO_OPT=1 cargo oxide run unaligned_memory
```

## Expected output

```text
=== unaligned_memory ===
PASS: read_unaligned from base + 1
PASS: write_unaligned to base + 1
PASS: guard bytes preserved
PASS: repr(packed) field read at rustc offset 1
PASS: repr(packed) field write at rustc offset 1
PASS: repr(packed(2)) field read at rustc offset 2
PASS: unaligned_memory
```
