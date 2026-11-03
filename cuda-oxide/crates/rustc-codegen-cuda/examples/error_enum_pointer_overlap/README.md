# `error_enum_pointer_overlap`

This negative test protects LLVM pointer provenance in enum payload storage.

```rust
#[repr(C)]
enum PointerOrBits {
    Bits(u64),
    Pointer(*const u8),
}
```

Rust lays both payloads over the same eight bytes. cuda-oxide cannot model
that storage as either a plain integer or a plain pointer without losing the
other variant's meaning. It therefore fails closed until lowering has an
explicit byte-preserving representation that also preserves pointer
provenance.

Run it with:

```bash
cargo oxide build error_enum_pointer_overlap
```

The build must fail with a diagnostic containing:

```text
overlapping pointer and non-identical storage
```
