# `error_union_shared_pointer_layout`

Negative test for a union whose field is an array of seventeen
`*mut SharedArray<...>` values.

Union fields that hold shared-memory pointers, directly, through structs, or
through bounded arrays, use target-stable generic-pointer storage, so the
union's byte image does not depend on whether the backend sizes a
shared-memory pointer at 8 bytes (PTX and legacy NVVM) or 4 bytes (modern
NVVM, `p3:32`). Array fields are rebuilt leaf by leaf at the union's construct
and extract boundary, with one extract, address-space cast and insert sequence
per pointer. To keep that expansion bounded, union lowering accepts at most 16
array-expanded shared-pointer leaves per field, whether they come from one
array or from several arrays nested through structs. The enum payload path has
the same bound; see `error_enum_shared_pointer_layout`.

This fixture exceeds the bound with 17 leaves. The compiler must reject it
instead of generating unbounded reconstruction or keeping the
target-dependent shared-pointer storage:

```bash
cargo oxide build error_union_shared_pointer_layout
cargo oxide build error_union_shared_pointer_layout --emit-nvvm-ir --arch sm_90
cargo oxide build error_union_shared_pointer_layout --emit-nvvm-ir --arch sm_100
```

Expected diagnostic:

```text
union field storage: arrays containing shared-memory pointers are not supported above the bounded rewrite limit; rewrite requires 17 pointer conversions, supported bound is 16
```
