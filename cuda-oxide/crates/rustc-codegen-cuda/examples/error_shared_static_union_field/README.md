# `error_shared_static_union_field`

Negative test for a `SharedArray` field inside a union that has a
same-typed sibling at the same offset:

```rust
union Inner { tile: SharedArray<u32, 64> }
struct Wrap { direct: SharedArray<u32, 64>, hidden: Inner }
```

`SharedArray` is zero-sized, and every union field starts at the union's
base, so `WRAP.direct` and `WRAP.hidden.tile` both sit at byte offset zero.
Constant pointers to them have the same type and offset, and the compiler
cannot tell which one a pointer names. One global for both would make the two
arrays share storage, so it rejects the static, as it does two same-typed
fields of a tuple (see `error_shared_static_same_fields`):

```bash
cargo oxide build error_shared_static_union_field
```

Expected diagnostic:

```text
static `error_shared_static_union_field::WRAP` has fields `.0` and `.1.0` of the same shared-memory type at byte offset 0; a pointer to one cannot be told apart from a pointer to the other, so they would share storage. Give each `SharedArray` or `Barrier` its own static
```
