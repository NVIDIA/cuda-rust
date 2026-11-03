# `tuple_constant_provenance`

Runtime regression for preserving pointer provenance in direct tuple value
constants.

Covered cases:

- multiple thin references stored in one direct tuple constant;
- a reference to a complete device static;
- an interior reference with a non-zero byte addend;
- scalar fields stored alongside pointer fields;
- rustc layout offsets for pointer and scalar tuple fields.

The tested constant has the following shape:

```rust
static DATA: [u8; 16] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

const POINTERS: (&[u8; 16], &u8, bool, u8) =
    (&DATA, &DATA[7], true, 3);
```

Run with:

```shell
cargo oxide run tuple_constant_provenance
```
