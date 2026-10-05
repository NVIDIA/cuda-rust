/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[cuda_module]` in a library crate (regression test for issue #72).
//!
//! ## Structure
//!
//! ```text
//! cuda_module_in_lib/
//! ├── Cargo.toml          # Binary crate (this file)
//! ├── src/main.rs         # Loads the module defined in kernel-lib
//! └── kernel-lib/         # Library crate "module-kernels"
//!     ├── Cargo.toml
//!     └── src/lib.rs      # Holds the #[cuda_module] itself
//! ```
//!
//! Unlike `cross_crate_kernel` (where the library exports *generic*
//! kernels and the PTX is generated while compiling the binary), the
//! library here holds concrete kernels, so its PTX is embedded while the
//! *library* is compiled and travels inside the library's `.rlib` as an
//! extra object file containing only the `.oxart` data section.
//!
//! Linkers drop archive members that nothing references. Before the fix
//! for issue #72 the artifact member defined no symbols, so it never
//! reached the final binary and `kernels::load(&ctx)` failed at runtime
//! with `ModuleNotFound { name: "module-kernels" }`. The fix gives the
//! artifact object a link-anchor symbol that the generated `load_named()`
//! references, which forces the linker to keep the member.
//!
//! ## What This Tests
//!
//! 1. The library bundle ("module-kernels") is present in the executable
//!    (checked by parsing our own binary, before any CUDA call).
//! 2. The binary's own bundle ("cuda_module_in_lib") still coexists with
//!    the library bundle, and both load by name.
//! 3. Kernels from both modules launch and produce correct results.
//! 4. The library module root has only nested kernels, so its root `load()`
//!    must pin descendant device code without depending on a child view call.
//!
//! Step 1 runs before CUDA initialization on purpose: on a GPU-less
//! machine the linkage regression is still caught by the bundle check.
//!
//! ## Build and Run
//!
//! ```bash
//! cargo oxide run cuda_module_in_lib
//! ```

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use module_kernels::kernels;

/// The binary crate keeps a small module of its own so the example also
/// proves that the same-crate embedding path (artifact object passed
/// straight to the linker, no archive involved) keeps working next to
/// the library-crate path.
#[cuda_module]
mod bin_kernels {
    use super::*;

    #[kernel]
    pub fn iota_f32(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = idx_raw as f32;
        }
    }
}

/// Bundle name of the kernel library (its `CARGO_PKG_NAME`).
const LIB_BUNDLE: &str = "module-kernels";
/// Bundle name of this binary (its `CARGO_PKG_NAME`).
const BIN_BUNDLE: &str = "cuda_module_in_lib";

fn main() {
    println!("=== #[cuda_module] in Library Crate Test ===\n");

    // =========================================================================
    // Test 1: both artifact bundles survived linking (no CUDA needed)
    // =========================================================================
    println!("Test 1: embedded bundles in the executable");
    let bundle_names: Vec<String> = cuda_host::embedded::artifact_bundles_from_current_exe()
        .expect("failed to parse the current executable")
        .into_iter()
        .map(|bundle| bundle.name)
        .collect();
    println!("  found bundles: {bundle_names:?}");
    for expected in [LIB_BUNDLE, BIN_BUNDLE] {
        if !bundle_names.iter().any(|name| name == expected) {
            println!("  ✗ FAILED: bundle '{expected}' is missing from the executable");
            std::process::exit(1);
        }
    }
    println!("  ✓ PASSED: library and binary bundles are both embedded\n");

    if std::env::args().any(|arg| arg == "--verify-bundles") {
        println!("SUCCESS: nested-only library bundle survived archive linking");
        return;
    }

    // =========================================================================
    // CUDA setup
    // =========================================================================
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    const N: usize = 1024;

    // =========================================================================
    // Test 2: load the library module by its package name and launch
    // =========================================================================
    println!("Test 2: module_kernels::kernels::load + scale_f32/add_f32");
    {
        // This is the exact call that used to fail with
        // ModuleNotFound { name: "module-kernels" }.
        let module = kernels::load(&ctx).expect("Failed to load library cuda_module");
        let ops = kernels::ops::LoadedModule::from_parent(&module)
            .expect("Failed to bind library kernel launchers");

        let factor: f32 = 2.5;
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            ops.scale_f32(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("scale_f32 launch failed");
        let scaled: Vec<f32> = output_dev.to_host_vec(&stream).unwrap();

        let mut sum_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            ops.add_f32(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &input_dev,
                &output_dev,
                &mut sum_dev,
            )
        }
        .expect("add_f32 launch failed");
        let sums: Vec<f32> = sum_dev.to_host_vec(&stream).unwrap();

        let errors = (0..N)
            .filter(|&i| {
                (scaled[i] - input[i] * factor).abs() > 1e-5
                    || (sums[i] - (input[i] + scaled[i])).abs() > 1e-5
            })
            .count();
        if errors == 0 {
            println!("  ✓ PASSED: library kernels load and run by bundle name\n");
        } else {
            println!("  ✗ FAILED: {errors} errors\n");
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 3: the binary's own module still loads alongside
    // =========================================================================
    println!("Test 3: bin_kernels::load + iota_f32");
    {
        let module = bin_kernels::load(&ctx).expect("Failed to load binary cuda_module");
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.iota_f32(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev) }
            .expect("iota_f32 launch failed");
        let out: Vec<f32> = out_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N).filter(|&i| out[i] != i as f32).count();
        if errors == 0 {
            println!("  ✓ PASSED: binary kernels unaffected by the fix\n");
        } else {
            println!("  ✗ FAILED: {errors} errors\n");
            std::process::exit(1);
        }
    }

    println!("SUCCESS: #[cuda_module] in a library crate loads and runs");
}
