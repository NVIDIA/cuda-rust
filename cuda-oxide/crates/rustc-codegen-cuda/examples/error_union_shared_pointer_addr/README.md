# `error_union_shared_pointer_addr`

Negative test for an in-place write to a union field that holds a
shared-memory pointer (`view.ptr = &raw mut STAGE`).

A shared-memory pointer is 8 bytes wide on the PTX and legacy NVVM paths but
4 bytes wide on the modern NVVM path (`p3:32`). A union that holds one
therefore stores it as a generic pointer, which is 8 bytes on every path, and
the value paths (construct and extract) convert at that boundary. An in-place
write or a borrow of the field goes through the field's address instead. The
loads and stores made through that address carry the field's semantic type,
the raw shared pointer, so they would write or read a form of the pointer that
every other reader of the union treats as a generic address. On the modern
NVVM path that is a 4-byte store into an 8-byte slot. On the other paths the
shared-window offset comes back as a generic pointer, which is a wrong pointer
or an illegal address.

The compiler refuses to form such an address. The same refusal covers
`&view.ptr`, `&mut view.ptr`, and writes into a struct nested in the field
(`view.inner.x = v`). Build the union by value and read its fields by value;
the `shared_ptr_transmute` example runs that path on the GPU.

```bash
cargo oxide build error_union_shared_pointer_addr
```

Expected: the build FAILS, and the smoketest checks the log for this
diagnostic:

```text
cannot hand out the in-place address of union field 0 of `SharedPtrBits`
```

together with the `canonical storage type` explanation from the
`mir.field_addr` lowering gate.
