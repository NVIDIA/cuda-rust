# Static Slice Addend

Positive runtime regression for an array-to-slice unsize whose data pointer
contains a non-zero byte addend into a device static.

The example defines:

```rust
static TABLE: [[f32; 2]; 4] =
    [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];

const PAIR_SLICE: &[f32] = &TABLE[2];
```

`PAIR_SLICE` is represented as a fat pointer:

 - The data word points 16 bytes into TABLE.
 - The metadata word stores a length of 2.

The kernel writes both the sum of the selected elements and the observed slice
length. This verifies that the importer preserves the interior addend and the
fat-pointer metadata independently.


Run:

```bash
cargo oxide run static_slice_addend
```

Expected output:

```text
static_slice_addend: PASS (sum 12, len 2)
```
