# cuda_module_contract

Focused regression coverage for the typed `#[cuda_module]` host ABI.

The kernel mixes common argument shapes that the generated launch method must
marshal correctly:

- scalar `f32` arguments
- `&[f32]` input through `&DeviceBuffer<f32>`
- raw device pointer argument
- `DisjointSlice<f32>` output through `&mut DeviceBuffer<f32>`

Regular Rust struct layout is tested separately by `abi_hmm`; cuda-oxide does
not require `#[repr(C)]` for Rust-only shared structs.

The example also pins the prepared 1-D launch contract:

```text
LaunchConfig1D
      │ validate once: exact 256-thread block + live CUDA limits
      ▼
PreparedLaunch<mixed_abi>
      │ reusable; cannot accept Y/Z geometry or another kernel's config
      ▼
typed mixed_abi launch
```

The kernel's `DisjointSlice` uses the 1-D index space, so
`#[launch_contract(domain = 1, ...)]` makes the host geometry match that
assumption. Raw `LaunchConfig` remains available only through the generated
unsafe `mixed_abi_unchecked` escape hatch.

The example also crosses the module-provenance boundary once with
`unsafe { kernels::load(&ctx) }`: this crate has one device-code owner, so the
package-named embedded bundle is the code generated from this module. Prepared
launches are safe after that assertion.

The second kernel proves the dynamic shared-memory alignment path. Its body
requests 16-byte alignment while the launch contract requires 128 bytes; the
compiler takes the stronger requirement and emits:

```ptx
.extern .shared .align 128 .b8 __dynamic_smem_aligned_dynamic_shared[];
```

Contracts also follow ordinary device calls, including more than one level of
helpers. Two 32-thread entries reach the same shared-memory owner; one requires
32-byte alignment and the other requires 256:

```text
helper_contract_32  ─┐
                     ├─> forward helper -> shared owner -> `.align 256`
helper_contract_256 ─┘
```

The maximum is required because the helper has one compiled extern-shared
declaration that may run under either entry.

The generic closure kernel checks the other ownership path. Its exported entry
calls an inline generic helper, and that helper owns the shared-memory access:

```text
generic_aligned entry (contract: 64-byte alignment)
        -> generic helper (body asks for 16 bytes)
        -> PTX extern shared symbol uses 64 bytes
```

This matters because the helper still exists when MIR is lowered; waiting for
LLVM to inline it would be too late to choose the extern-shared declaration.
The GPU-free verifier checks both that `.align 64` declaration and the generic
entry's `.maxntid 64, 1, 1` directive.

That kernel deliberately writes `#[launch_contract]` and `#[launch_bounds]`
above `#[kernel]`. In this order the attributes have already become body
markers when the generic entry is generated, so the macro must copy those exact
compiler markers to the entry rather than depending on attribute order.

A small `#[kernel(u32, launch_context = launch_context)]` compile-only kernel checks the
explicit-instantiation form too: its helper asks for 8-byte alignment, its
contract raises that to 32, and the generated `explicit_aligned_u32` entry
keeps `.maxntid 32, 1, 1`. It uses the opposite pre-`#[kernel]` ordering of
bounds and contract attributes. It also calls
`thread::index_1d_u32(launch_context)`, proving that
the generated entry—not its generic helper—creates and forwards the prepared
1-D launch context.

```bash
cargo oxide run cuda_module_contract

# GPU-free: build, then verify launch bounds and shared-memory alignment.
cargo oxide build cuda_module_contract
cuda-oxide/crates/rustc-codegen-cuda/examples/cuda_module_contract/target/release/cuda_module_contract --verify-ptx
```
