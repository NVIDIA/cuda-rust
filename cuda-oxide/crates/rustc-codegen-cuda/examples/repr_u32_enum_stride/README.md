# `repr_u32_enum_stride` — enum discriminant layout repro

Reads from `*const E` where `E` is a fieldless `#[repr(u32)] enum` are
miscompiled by `rustc-codegen-cuda`: pointer arithmetic strides by
**1 byte** instead of the expected 4.

The example uses a generic `Tag` enum with four valid discriminants:
`Foo = 0`, `Bar = 1`, `Baz = 2`, and `Qux = 3`. The kernel buffer at
offset 4 reads as `1` via `*const u32`, but a buggy enum pointer path does
not produce the slot-1 discriminant. The failure pattern matches 1-byte
stride instead of the expected 4-byte stride.

## Run

```bash
cargo oxide run repr_u32_enum_stride
```

## Expected output before the fix

```
control_u32   [0, 1, 2, 3]   PASS
enum_ptr      [0, 0, 0, 0]   FAIL
RESULT: FAIL - at least one enum shape miscompiled (see above).
```

## Expected output after the fix

```
control_u32   [0, 1, 2, 3]   PASS
enum_ptr      [0, 1, 2, 3]   PASS
repr_c_enum   [0, 1, 2, 3]   PASS
sparse_enum   [0, 1000000, 1000000, 0]   PASS
neg_enum      [-1, 0, -1, 0]   PASS
usize_enum    [0, 1, 2, 3]   PASS
RESULT: PASS - every enum shape strides and sign-extends correctly.
```

## Root cause and fix

The MIR importer lowered fieldless enum discriminants from the number of
variants alone. For this four-variant `#[repr(u32)]` enum, that selected an
8-bit discriminant type even though Rust's explicit representation requires a
32-bit layout.

Pointer arithmetic over `*const Tag` then used the lowered enum element size,
so `base.add(1)` advanced by 1 byte instead of 4 bytes. The fix in
`cuda-oxide/crates/mir-importer/src/translator/types.rs` sources the discriminant type
from **rustc's layout** (`rust_ty.layout()`): for `TagEncoding::Direct` enums
the tag scalar's width and signedness are used directly. Reading the layout
(rather than the `repr` attribute or the variant count) covers every tag
shape with one mechanism, which is why this example also tests:

| Kernel        | Enum shape                        | What must hold        |
| :------------ | :-------------------------------- | :-------------------- |
| `enum_ptr`    | `#[repr(u32)]`, 4 unit variants   | stride 4              |
| `repr_c_enum` | `#[repr(C)]`, 4 unit variants     | stride 4 (C `int`)    |
| `sparse_enum` | default repr, `B = 1_000_000`     | u32 tag, stride 4     |
| `neg_enum`    | default repr, `N = -1`            | SIGNED i8 tag, `sext` |
| `usize_enum`  | `#[repr(usize)]`, 4 unit variants | stride 8              |

`neg_enum` is the signedness half of the bug: the old model hardcoded an
unsigned tag, so a memory-loaded `e as i32` zero-extended the `0xFF` byte to
`255` instead of sign-extending it to `-1`.

Niched enums (for example `Option<&T>`) use a different rustc layout path:
their otherwise-invalid payload value is recorded as the physical carrier,
and cuda-oxide does not add a separate variant-count tag.
