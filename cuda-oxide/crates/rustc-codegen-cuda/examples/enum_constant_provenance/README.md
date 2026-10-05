# `enum_constant_provenance`

Runtime regression for pointer provenance in enum constants.

The example covers:

- a niche-encoded `Option<&T>` pointing at an interior device-static element;
- the niche variant without a pointer relocation;
- a direct-tagged `#[repr(u8)]` enum without a payload;
- a direct-tagged enum with a pointer payload;
- a direct-tagged relocation-carrying enum nested at a non-zero offset in a
  struct constant;
- a niche-encoded `Option<&T>` nested in a tuple constant;
- a relocation-carrying enum nested in an array constant.

The importer must preserve the enclosing allocation while recursively
reconstructing enum pointer fields from rustc relocation entries. The bytes
stored under a relocation represent the addend into the target allocation, not
an exposed pointer address.

```bash
cargo oxide run enum_constant_provenance
```

Expected result:

```text
enum_constant_provenance: PASS
```

Pointer/integer overlapping enum storage, address-space-3 pointer layouts, and
anonymous promoted allocations remain unsupported.
