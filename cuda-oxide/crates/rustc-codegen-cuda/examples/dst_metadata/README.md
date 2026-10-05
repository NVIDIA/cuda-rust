# dst_metadata

Regression coverage for device-side DST layout metadata through
`core::mem::size_of_val` and `core::mem::align_of_val`.

## What this tests

CUDA Oxide represents slice-shaped fat pointers as `(data_ptr, len)`. This
example verifies the two libcore layout intrinsics for DST shapes whose metadata
is a runtime length:

- `&[u32]`: `size_of_val` must compute `len * size_of::<u32>()`, while
  `align_of_val` is `align_of::<u32>()`.
- `str`: the metadata is a UTF-8 byte length, so `size_of_val` returns that byte
  length and `align_of_val` is 1.
- `&Header<[u16]>`: the metadata is the length of the trailing slice. The
  runtime size must include the statically laid out prefix, the inline tail, and
  final ABI padding. The test also reads prefix fields and the trailing-slice
  length metadata through the fat pointer.

The struct case uses:

```rust
#[repr(C)]
struct Header<T: ?Sized> {
    head: u64,
    tag: u8,
    tail: T,
}
```

and unsizes `&Header<[u16; 4]>` to `&Header<[u16]>`. The mixed-width prefix is
intentional: it makes padding and final size rounding observable instead of
testing only a packed-looking layout.

The string input is `oxide✓`. It has six Unicode scalar values but eight UTF-8
bytes, so the test distinguishes byte-length metadata from character count.

## Usage

```bash
cargo oxide run dst_metadata
```

Also validate the low-MIR-optimization path:

```bash
CUDA_OXIDE_NO_OPT=1 cargo oxide run dst_metadata
```

## Expected output

```text
=== dst_metadata ===
PASS: size_of_val on &[u32]
PASS: align_of_val on &[u32]
PASS: size_of_val on str
PASS: align_of_val on str
PASS: size_of_val on &Header<[u16]>
PASS: align_of_val on &Header<[u16]>
PASS: unsized-tail metadata and prefix fields
PASS: dst_metadata
```
