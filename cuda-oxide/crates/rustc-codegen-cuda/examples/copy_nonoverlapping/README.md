# copy_nonoverlapping

Regression test for device-side `core::ptr::copy_nonoverlapping` lowering.

## What this tests

`core::ptr::copy_nonoverlapping` lowers to MIR as
`StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(_))`.
The importer translates that statement to `mir.memcpy`, and MIR lowering emits
an LLVM memcpy with the element count scaled to bytes for the pointee type.

This example launches a kernel that copies `u32` values from one device buffer
to another with `copy_nonoverlapping`, then checks the copied host output
exactly.

## Usage

```bash
cargo oxide run copy_nonoverlapping
```

## Expected output

The example should build, run, and print:

```text
PASS: copy_nonoverlapping copied 128 u32 values on the device
```
