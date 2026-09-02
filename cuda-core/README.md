# cuda-core

`cuda-core` is the thin safe wrapper layer over `cuda-bindings`. It exposes the
lower-level CUDA concepts used by the rest of the workspace without requiring
most crates to touch raw FFI directly.

## Two host models

The crate currently carries two host-side models of the same driver:

- **The crate root** (`Device`, `Stream`, `Module`, `MemPool`, `LaunchConfig`,
  and the reviewed `vmm` module) is the model the rest of the workspace
  (`cutile`, `cuda-async`) is written against.
- **`cuda_core::simt`** is the cuda-oxide host surface (`CudaContext`,
  `CudaStream`, `CudaModule`, `DeviceBuffer`, `PinnedHostBuffer`,
  `ConstantHandle`, the typed launch contracts, and its own `vmm` and `peer`
  modules), carried over as-is for the shared host-crate migration. It is a
  temporary bridge: the two models get reconciled after the repository
  migration, and `simt` is deletable as a unit once that happens. Every
  `simt` name that does not collide with the root surface is also re-exported
  at the crate root; the two that collide, `simt::LaunchConfig` and
  `simt::vmm`, are reachable only through `simt::`.

New code in this workspace targets the root types. Code ported from
cuda-oxide keeps using `simt` until the reconciliation.

The multicast half of `simt::vmm` (`MulticastObject`, `Mapping::new_multicast`)
is compiled only when the CUDA toolkit headers declare the `cuMulticast*` API
(CUDA 12.1+); `build.rs` probes for it and sets the `cuda_has_multicast` cfg.

## Features

- `f16` (default off, nightly only): implements `DeviceCopy` for the unstable
  `f16` primitive. `half::f16` and `half::bf16` are supported unconditionally.

## Companion crate

`cuda-core-derive` provides `#[derive(DeviceCopy)]` for plain-data structs and
unions. It is re-exported next to the trait, so `use cuda_core::DeviceCopy;`
brings both the trait and the derive into scope.

## Testing

Unit and doc tests run without a GPU:

```bash
cargo test -p cuda-core --lib
cargo test -p cuda-core --doc
```

The integration tests need a GPU. `tests/vmm.rs` and `tests/vmm_multicast.rs`
exercise the root `vmm` module; `tests/simt_*.rs` exercise the `simt` surface.
Tests that need hardware this machine may lack self-skip: a second GPU for
`simt_vmm_p2p` and `vmm_multicast`, NVLink switch multicast for the multicast
tests. `scripts/run_gpu_tests.sh` lists every GPU test target and is the
reference for what must pass; a new `simt_*` test file must be added there.
