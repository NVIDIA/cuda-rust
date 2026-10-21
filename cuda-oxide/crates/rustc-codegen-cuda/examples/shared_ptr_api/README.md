# shared_ptr_api

`SharedPtr<T>` is a raw pointer into the issuing block's shared memory whose
address space (`addrspace(3)`) is part of its type. A generic `*mut T` loses
that fact at the first helper boundary; a `SharedPtr` keeps it through
parameters, return values, struct fields, closures and arrays, so the compiler
can use the shared-memory forms of the instructions that take it. This example
moves one through each of those boundaries and checks every value on the host.

| Kernel              | Boundary                                                        | Check                                   |
| :------------------ | :-------------------------------------------------------------- | :-------------------------------------- |
| `static_round_trip` | `SharedArray::shared_ptr`, `from_raw(as_ptr())`, negative `offset` | each thread reads back its own slot |
| `closure_capture`   | a closure that captured a `SharedPtr`, called through a generic `#[inline(never)]` helper | `3 * tid` |
| `addr_u32_deltas`   | `addr_u32` of `shared_ptr_at(64)`, `add(3)`, `byte_add(7)`, and of `get_raw()` through `cvta_generic_to_shared_u32` | `[64, 12, 1, 7]` |
| `behind_references` | `&Region<T>` (a struct holding the pointer) and `&mut SharedPtr<T>` through non-inlined helpers | three reads summed |
| `returned_pointer`  | a non-inlined helper returning `SharedPtr<T>`                   | odd lanes read the pool, even the array |
| `array_by_value`    | `[SharedPtr<T>; 2]` passed by value                             | `7 + tid`                               |
| `struct_pointee`    | `SharedPtr<Pair>` and `cast::<f32>()` to its first field        | `a + b + a`                             |
| `byte_views`        | `byte_add` into a `u8` pool, viewed as `f64` and `u16`          | `0.25 * tid + tid + 7`                  |
| `option_pointer`    | `Option<SharedPtr<u32>>` returned and passed                    | `Some` stores, `None` does not          |
| `stored_in_shared`  | `SharedArray<SharedPtr<u32>, 64>`: pointers kept in shared memory | the mirrored slot's value             |
| `transmuted`        | `transmute` to `usize` and back in non-inlined helpers          | the store lands and the stride is 4     |

The helpers carry `#[inline(never)]`, so each boundary is a real call. In the
LLVM IR the default build leaves next to this file, `smem_fill`, `smem_load`,
`pick`, `maybe` and the closure's `apply` take `ptr addrspace(3)`,
`store_via_array` takes `[2 x ptr addrspace(3)]`, `pick` and `from_bits`
return `ptr addrspace(3)`, and `SLOTS` is a `[64 x ptr addrspace(3)]` global
in address space 3. The PTX converts no generic address to a shared one
except in `from_bits`: the integer view of a shared pointer is its generic
address, as for unions, so the transmute back is one `cvta.to.shared`.

Structs that mix a `SharedPtr` with a narrower field, such as
`{ p: SharedPtr<u8>, x: u32 }`, stay out of this example. On the libNVVM
route a shared pointer is 4 bytes wide, and such a struct is not yet
supported in arrays or in shared memory; the book's shared memory chapter
lists this limit.

## Build and run

```bash
cargo oxide run shared_ptr_api
```

The compile-only lane of `scripts/smoketest.sh` also compiles this example
through libNVVM (`NVVM_VERIFY_EXAMPLES`), where the shared pointer is 4 bytes
wide:

```bash
cargo oxide emit-ltoir shared_ptr_api --arch=sm_120
```

## Expected output

```text
=== SharedPtr API (64 threads) ===
  ok    static_round_trip
  ok    closure_capture
  ok    addr_u32_deltas
  ok    behind_references
  ok    returned_pointer
  ok    array_by_value
  ok    struct_pointee
  ok    byte_views
  ok    option_pointer
  ok    stored_in_shared
  ok    transmuted
SUCCESS: shared_ptr_api, 11 kernels, 0 mismatches
```

## Hardware requirements

Any CUDA GPU; the kernels use plain shared memory and `sync_threads()`. The
launch supplies 1024 bytes of dynamic shared memory.
