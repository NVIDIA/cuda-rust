# `error_kernel_shared_ptr_param`

Negative test for a `#[kernel]` whose parameter is a struct that holds a
`SharedPtr<f32>`, a pointer that lowers to `addrspace(3)` by its type.

A kernel receives its parameters in `.param` space, filled by the host at
launch. Shared memory is allocated per block by the device, so the host holds
no address in it to pass. `error_kernel_shared_param` pins the refusal of a
bare shared pointer parameter. This fixture pins the same refusal for a
pointer carried inside an aggregate: before the exporter looked through struct
fields and array elements, `Args { p: SharedPtr<f32>, n: u32 }` compiled, the
entry declared a 16-byte `.param`, and the kernel applied `cvta.shared.u64` to
the host-supplied bytes as if they were a shared address.

The compiler must reject the signature at compile time instead:

```bash
cargo oxide build error_kernel_shared_ptr_param
```

Expected diagnostic (pinned by `cuda-oxide/scripts/smoketest.sh`):

```text
holds a pointer into shared memory
```

The diagnostic names the field path inside the parameter (`.0` here), which
keeps it distinct from the bare parameter's `is a pointer into shared memory`.
Structs of global (`addrspace(1)`) and constant (`addrspace(4)`) pointers stay
allowed as kernel parameters: the host allocates in those spaces, so it has an
address to supply. A `SharedPtr` belongs on the device side, created inside
the kernel from a `static mut SharedArray` or from `DynamicSharedArray`.
