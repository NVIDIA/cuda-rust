# CUDA host runtime

`cuda-core` wraps `cuda-bindings` with the CUDA types and functions used by
the rest of the workspace. Tile and SIMT use the same handles and allocation
owners; there is no `simt` module in this crate.

`Device` identifies a physical GPU. `Context` owns or borrows a CUDA execution
context on that GPU. `Stream`, `Event`, `Module`, and `Function` retain their
actual context, and compatibility checks compare context handles rather than
device ordinals. The `CudaContext`, `CudaStream`, `CudaEvent`, `CudaModule`, and
`CudaFunction` spellings are aliases of those shared types.

`DeviceBuffer<T>` provides typed storage. `RawDeviceBuffer` shares allocation
ownership, byte offsets, context identity, and access tracking with typed
buffers and Tile tensors. Owned synchronous allocations, stream-ordered
allocations, borrowed pointers, and foreign owners use this same representation.
Foreign `DeviceAllocation` implementations supply their actual context.

Safe submissions retain allocation access leases independently of their future.
Conflicting accesses on different streams return `CUDA_ERROR_NOT_READY` until
completion; accesses in one stream remain ordered by CUDA. Buffer and stream
destructors retire resources without waiting for GPU work on the dropping
thread. If completion cannot be established, retirement retains the resource
instead of freeing or reusing it. Unsafe raw-pointer users must establish their
own lifetime, initialization, and ordering guarantees.

`DeviceCopy` permits byte transfers of valid values. `DType` describes Tile
element types. Neither trait implies the other; types supported by both models
implement both. Safe casts to another representation additionally require
`AnyBitPattern`, while `DeviceRepr` supports ownership-preserving wrappers such
as device atomics. A `DeviceBuffer<bool>` remains supported, but arbitrary bytes
cannot safely be cast to `bool`.

For migration, replace `cuda_core::simt::...` imports with `cuda_core::...`.
Code using the old context-owning `Device` should use `Context`; stream and pool
context accessors are named `context()`. With cuTile, use `Tensor::from_buffer`
and `Tensor::into_buffer` for ownership transfer without copying device memory,
or `Tensor::as_buffer` / `TensorView::as_buffer` for contiguous borrowed kernel
arguments. Ownership transfer requires unique storage; borrowed slices preserve
their byte offset and reject strided layouts.

# Testing

Run the crate tests with:

```bash
cargo test -p cuda-core
```
