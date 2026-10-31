# `error_shared_static_same_fields`

Negative test for a static with two `SharedArray` fields of the same type:

```rust
static mut PAIR: (SharedArray<u32, 64>, SharedArray<u32, 64>) =
    (SharedArray::UNINIT, SharedArray::UNINIT);
```

A pointer to a `SharedArray` or `Barrier` field of a tuple, struct or array
static gets its own shared-memory global, named after the static and the
field. `SharedArray` is zero-sized, though, so both fields of `PAIR` sit at
byte offset zero, and a constant pointer to either has the same
type and offset. The compiler cannot tell which field it names, and one global
for both would make the two arrays share storage, so it rejects the static:

```bash
cargo oxide build error_shared_static_same_fields
```

Expected diagnostic:

```text
static `error_shared_static_same_fields::PAIR` has fields `.0` and `.1` of the same shared-memory type at byte offset 0; a pointer to one cannot be told apart from a pointer to the other, so they would share storage. Give each `SharedArray` or `Barrier` its own static
```

Fields of different types, such as `SharedArray<u32, 64>` and
`SharedArray<u32, 32>`, or the same element count with a different alignment,
are distinct and allowed; see `shared_static_names`.
