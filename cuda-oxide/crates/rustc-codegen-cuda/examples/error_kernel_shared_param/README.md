# `error_kernel_shared_param`

Negative test for a `#[kernel]` whose parameter points into shared memory
(`*mut Barrier`, which lowers to an `addrspace(3)` pointer).

A kernel receives its parameters in `.param` space, filled by the host at
launch. Shared memory is allocated per block and local memory per thread,
both by the device, so the host holds no address in either space to pass.
Before the exporter refused this shape, the address space reached the entry
signature as `.ptr .shared`: ptxas assembled the module and the driver then
refused to load it, taking every other kernel in the module down with it
(this is how `generated_intrinsics_blackwell` became unloadable).

The compiler must reject the signature at compile time instead:

```bash
cargo oxide build error_kernel_shared_param
```

Expected diagnostic (pinned by `cuda-oxide/scripts/smoketest.sh`):

```text
is a pointer into shared memory
```

Global (`addrspace(1)`) and constant (`addrspace(4)`) pointer parameters stay
allowed: the host allocates in those spaces, so it has an address to supply.
A barrier belongs in a device-side `static mut BAR: Barrier` instead, as in
`examples/barrier`.
