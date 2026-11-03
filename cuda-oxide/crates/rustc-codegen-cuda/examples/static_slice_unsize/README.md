# static_slice_unsize

Positive test for zero-addend device-static array→slice unsize:

```rust
static TABLE: [f32; 4] = [0.25, 0.5, 1.0, 2.0];
const TABLE_SLICE: &[f32] = &TABLE;
```

`&TABLE` is `&[f32; 4]`. The unsize coercion to `&[f32]` keeps a zero addend
and adds length metadata. cuda-oxide materializes a fat pointer via
`mir.construct_slice` (thin global pointer + slice length).

The slice length comes from the constant's own fat-pointer metadata word,
not from the array type, so a zero-addend prefix subslice carries its true
length:

```rust
const TABLE_HEAD: &[f32] = {
    let s: &[f32] = &TABLE;
    s.split_at(2).0 // len 2, not TABLE's 4
};
```

Interior addends are covered by `static_slice_addend`, which verifies that both
the interior data-pointer offset and the stored slice length are preserved.

```bash
cargo oxide run static_slice_unsize
```
