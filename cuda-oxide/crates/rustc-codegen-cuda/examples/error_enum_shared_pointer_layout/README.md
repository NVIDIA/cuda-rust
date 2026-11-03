# `error_enum_shared_pointer_layout`

Negative test for `Option<SharedPointerArrayWrapper>`, where the wrapper contains
an array of seventeen `&SharedArray<...>` values.

Direct shared-pointer enum fields, shared pointers nested through ordinary
structs/tuples, and bounded arrays of shared-pointer leaves use target-stable
CUDA generic physical storage. Array conversion rebuilds the value recursively,
emitting extraction, address-space cast, and insertion sequences for each shared
pointer leaf. To keep that expansion bounded, enum payload lowering accepts at
most 16 array-expanded shared-pointer leaves in total per payload, whether they
come from one array or from several arrays nested through structs.

This fixture deliberately exceeds that contract with 17 shared-pointer leaves.
The compiler must reject it instead of generating unbounded reconstruction or
retaining target-dependent address-space-3 pointer storage:

```bash
cargo oxide build error_enum_shared_pointer_layout
cargo oxide build error_enum_shared_pointer_layout --emit-nvvm-ir --arch sm_90
cargo oxide build error_enum_shared_pointer_layout --emit-nvvm-ir --arch sm_100
```

Expected diagnostic:

```text
enum payload storage: arrays containing shared-memory pointers are not supported above the bounded rewrite limit; rewrite requires 17 pointer conversions, supported bound is 16
```
