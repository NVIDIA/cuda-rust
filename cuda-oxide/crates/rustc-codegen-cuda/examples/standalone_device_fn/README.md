# standalone_device_fn

Standalone `#[device]` function compilation — no `#[kernel]` required.

## What This Demonstrates

This example verifies that `#[device]` functions can be compiled to PTX without
any `#[kernel]` entry point in the crate. This is the foundation for building
Rust device libraries that are consumed by CUDA C++ via LTOIR linking.

## Test Coverage

| # | Check | What It Verifies |
|---|---|---|
| 1 | `fast_sqrt` | A simple standalone device function appears in PTX |
| 2 | `clamp_f32` | A second independent device function appears in PTX |
| 3 | `safe_sqrt` | Transitive calls collect both helper functions |
| 4 | `fma_f32` | A concrete f32 wrapper monomorphizes `fma<T>` |
| 5 | `fma_i32` | A second concrete type produces another monomorphization |
| 6 | `get_global_thread_id` | Device thread-index intrinsics compile |
| 7 | `mma_m16n8k16_f32_f16_stub` | The public F16 MMA stub reaches the device pipeline |
| 8 | `mma_m16n8k8_f32_tf32_raw_stub` | The raw TF32 MMA stub reaches the device pipeline |
| 9 | `mma_m16n8k8_f32_tf32_from_f32_stub` | The f32-to-TF32 conversion path compiles |
| 10 | `int8_mma_registers` | The signed INT8 MMA stub reaches the device pipeline |
| 11 | Exact F16 MMA mnemonic | PTX contains `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| 12 | Exact TF32 MMA mnemonic | PTX contains `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` |
| 13 | Six TF32 conversions | PTX contains six `cvt.rna.tf32.f32` instructions feeding the MMA |
| 14 | Exact INT8 MMA mnemonic | PTX contains `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` |
| 15 | PTX 7.0 floor | PTX selects version 7.0 or newer for the Ampere MMA forms |
| 16 | `sm_80` floor | PTX targets `sm_80` or newer |
| 17 | `lerp` absent | An uninstantiated generic is not compiled |
| 18 | No `.entry` directives | Every emitted symbol is a device `.func`, not a kernel |

## How to Run

```bash
# From workspace root
cargo oxide run standalone_device_fn --arch sm_80
```

The explicit target keeps these MMA and conversion instructions on their
minimum supported architecture and PTX ISA: `sm_80` with PTX 7.0.

Expected output:

```text
=== Standalone Device Function Example ===

PTX file: standalone_device_fn.ptx (... bytes)
  PASS  fast_sqrt — Test 1: simple standalone fn
  PASS  clamp_f32 — Test 1: simple standalone fn
  ...
  PASS  mma_m16n8k16_f32_f16_stub — Test 5: F16 warp-MMA stub
  PASS  mma_m16n8k8_f32_tf32_raw_stub — Test 5: raw TF32 warp-MMA stub
  PASS  mma_m16n8k8_f32_tf32_from_f32_stub — Test 5: f32-to-TF32 conversion path
  PASS  int8_mma_registers — Test 5: signed INT8 warp-MMA stub
  PASS  exact F16 warp-MMA instruction emitted
  PASS  exact TF32 warp-MMA instruction emitted
  PASS  six f32-to-TF32 register conversions emitted
  PASS  lerp absent — Test 3b: uninstantiated generic not compiled
  PASS  No .entry directives (all are .func)
  PASS  INT8 MMA lowered to the exact PTX instruction
  PASS  INT8 MMA selected PTX 7.0 or newer
  PASS  INT8 MMA selected sm_80 or newer

SUCCESS: 18/18 tests passed — all device functions compiled to PTX!
```

## How It Works

1. The `#[device]` macro renames functions with the reserved
   `cuda_oxide_device_<hash>_` prefix (owned by
   `cuda-oxide/crates/reserved-oxide-symbols/`) and generates an `#[inline(always)]`
   wrapper with the original name.
2. The collector (`rustc-codegen-cuda/src/collector.rs`) detects standalone
   `#[device]` functions as compilation roots when no `#[kernel]` is present
3. The LLVM export layer strips the `cuda_oxide_device_<hash>_` prefix for clean
   names in the final output
4. Generic `#[device]` functions are only compiled if monomorphized by a
   concrete call site (standard Rust monomorphization rules)
5. The F16, TF32, and INT8 wrappers force their register fragments and the
   `cvt.rna.tf32.f32` conversion path through MIR import and lowering; the host
   checks each exact PTX instruction and its minimum target requirements.

## Related

- `cpp_consumes_rust_device/` — Takes this further: compiles to LTOIR and links with C++
