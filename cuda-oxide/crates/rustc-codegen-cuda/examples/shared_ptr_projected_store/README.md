# shared_ptr_projected_store

A shared-memory pointer written through a deref, field, or array-element place
whose declared type is a generic `*mut u32`.

## What it checks

`DynamicSharedArray::get()` returns an `addrspace(3)` pointer, and a local
that only ever holds it gets a shared-memory slot. When that value is stored
into a place whose declared pointee is a generic `*mut u32`, the stored value
has to be widened to the generic address space first. Under
`--nvptx-short-ptr` a shared pointer is 4 bytes and a generic one is 8, so the
value is converted to the place's type, never the reverse.

Each kernel stores the pointer through a different importer store path:

| kernel | store | path |
|---|---|---|
| `call_destination_store` | `Call((*p) = get())` (custom MIR) | prepared call destination |
| `deref_call_store` | `*p = get()` | `Deref`, through a call temporary |
| `deref_assign_store` | `*p = s; *q = s` | `Deref` |
| `field_store` | `h.ptr = s` | `Field` on a local |
| `deref_field_store` | `(*h).ptr = s` | `(Deref, Field)` |
| `nested_field_store` | `o.inner.ptr = s` | `(Field, Field)` on a local |
| `array_element_store` | `a[i] = s` | runtime-indexed array element |
| `deref_index_store` | `(*a)[i] = s` | place-address walker |
| `deref_nested_field_store` | `(*o).inner.ptr = s` | place-address walker, three projections |
| `volatile_store` | `write_volatile(p, s)` | `volatile_store` intrinsic |

Surface Rust calls into a temporary and then stores it, so only custom MIR
can make `(*p)` a call's own destination; `call_destination_store` uses it for
that path. The structs are `repr(C)` with a leading field, checked when the
pointer is read back, so a store aimed at the wrong offset is caught.

Every kernel then reads the pointer back out of the place, writes a lane-unique
value through it, synchronizes the block, and reads its neighbor's value. The
host checks all 32 lanes of every kernel.

## Running it

```bash
cargo oxide run shared_ptr_projected_store
```

Before the importer widened these stores, every kernel failed device
verification with `MirStoreOp value type must match pointer element type`.
