# Cross-Crate Kernel Example

This example demonstrates **defining type- and const-generic kernels in a
library crate** and instantiating them from a binary crate.

## Structure

```text
cross_crate_kernel/
├── Cargo.toml           # Binary crate
├── src/main.rs          # Uses kernels from kernel-lib
├── README.md            # This file
└── kernel-lib/          # Library crate with #[kernel] functions
    ├── Cargo.toml
    └── src/lib.rs       # Generic kernels: scale<T>, add_const<N>, etc.
```

## What This Tests

1. **Generic kernels in library crates** - `kernel_lib::scale<T>` is defined externally
2. **Monomorphization at use site** - `scale::<f32>`, `scale::<i32>` instantiated in binary
3. **Cross-crate PTX generation** - All kernel monomorphizations compiled to PTX
4. **Const specialization identity** - `add_const::<4>` and `<8>` remain distinct
5. **Device helper functions** - `kernel_lib::device_scale_helper` called from kernel

## Run

```bash
cargo oxide run cross_crate_kernel
```

The built executable can compare its own lookup names against generated PTX
without creating a CUDA context:

```bash
cargo oxide build cross_crate_kernel
./cuda-oxide/crates/rustc-codegen-cuda/examples/cross_crate_kernel/target/release/cross_crate_kernel \
  --verify-ptx
```

The executable still links the CUDA driver library, so this explicit local
check requires `libcuda.so.1`. Compile-only CI does not execute it.

This covers `scale::<f32>`, `scale::<i32>`, `add_const::<4>`, and
`add_const::<8>`.

The consuming crate can inspect those names without touching macro internals:

```rust
let f32_entry = kernels::scale_ptx_name::<f32>();
let four_entry = kernels::add_const_ptx_name::<4>();
```

## Expected Output

```text
=== Cross-Crate Kernel Test ===

Testing kernels defined in kernel-lib crate.

Test 1: kernel_lib::scale::<f32>
  ✓ PASSED: scale::<f32> from library works!

Test 2: kernel_lib::scale::<i32>
  ✓ PASSED: scale::<i32> from library works!

Test 3: kernel_lib::add::<f32>
  ✓ PASSED: add::<f32> from library works!

Test 4: kernel_lib::scale_with_helper::<f32> (uses device helper)
  ✓ PASSED: scale_with_helper uses device function from library!

Test 5: kernel_lib::add_const::<4/8>
  ✓ PASSED: const-generic library entries remain distinct!

=== All Cross-Crate Tests Passed! ===
```

## How It Works

### 1. Kernel Definition (kernel-lib/src/lib.rs)

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use core::ops::Mul;

#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[idx.get()] * factor;
    }
}
```

### 2. Kernel Usage (src/main.rs)

```rust
use kernel_lib::kernels;

fn main() {
    // scale::<f32> is monomorphized HERE, not in kernel-lib
    let module = kernels::from_module(raw_module).expect("typed module");
    // SAFETY: this is a 1D launch and both buffers contain N elements.
    unsafe {
        module.scale::<f32>(
            stream.as_ref(),
            LaunchConfig::for_num_elems(N as u32),
            factor,
            &input_dev,
            &mut output_dev,
        )
    }
    .expect("Kernel launch failed");
}
```

### 3. PTX Generation

The codegen backend:

1. Finds `cuda_oxide_kernel_<hash>_scale` marked with `#[kernel]` attribute (the
   `<hash>` is the fixed `246e25db_` suffix owned by `cuda-oxide/crates/reserved-oxide-symbols/`)
2. Discovers all monomorphizations: `scale::<f32>`, `scale::<i32>`, etc.
3. Generates unique PTX entry points: `scale_TID_<hex32>`, one entry per
   monomorphization (see "Generic Kernel Naming" below)
4. Collects device helper functions transitively

## Key Implementation Details

### Generic Kernel Naming

Each specialization gets a unique PTX name derived from rustc's stable
128-bit hash of the concrete kernel function-item type. That `FnDef` contains
the function identity and every ordered type and const argument, while the
on-wire name remains one fixed-length hex chunk:

| Rust code       | PTX entry point                    |
|-----------------|------------------------------------|
| `scale::<f32>`  | `scale_TID_<32 hex characters>`    |
| `scale::<i32>`  | `scale_TID_<different 32 hex>`     |
| `add_const::<4>`| `add_const_TID_<32 hex characters>`|
| `add_const::<8>`| `add_const_TID_<different 32 hex>` |

The naming scheme:
- Base name extracted from `cuda_oxide_kernel_<hash>_<name>` via
  `reserved_oxide_symbols::kernel_base_name`.
- Suffix: `_TID_` + 32 lowercase hex chars.
- Backend hashes the concrete `Instance::ty`.
- Host hashes `&kernel_entry::<T, N>` with
  `cuda_host::type_id_u128_of_val`.
- Both go through `erase_and_anonymize_regions` + stable hash with the pinned
  rustc toolchain and the same crate identity, so the values match
  byte-for-byte within one unified build.

### Cross-Crate Intrinsic Handling

When `kernel-lib` calls `cuda_device::thread::index_1d()`:
- The collector finds the call in the library's MIR
- The MIR importer recognizes it as an intrinsic
- It's expanded inline to NVVM operations (no function call)

### Panic/Unwind Handling

Library code may use standard library types like `core::ops::Mul`. These have unwind paths
in their MIR (compiled without `panic=abort`). The codegen backend handles this by:

1. **Accepting** `UnwindAction::Continue` from external crates
2. **Treating** unwind paths as unreachable (CUDA toolchain doesn't support unwinding today)
3. **No special flags needed** - works with vanilla Rust code

This enables seamless cross-crate kernel development without custom sysroots or special
build configurations.

## See Also

- [generic](../generic/) - Generic kernels in same crate
- [rustc-codegen-cuda README](../../README.md) - Backend documentation
