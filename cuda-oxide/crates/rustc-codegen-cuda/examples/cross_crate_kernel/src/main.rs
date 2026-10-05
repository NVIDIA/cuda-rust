/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-Crate Kernel Example
//!
//! This example tests the ability to use kernels defined in a library crate.
//!
//! ## Structure
//!
//! ```text
//! cross_crate_kernel/
//! ├── Cargo.toml          # Binary crate (this file)
//! ├── src/main.rs         # Uses kernels from kernel-lib
//! └── kernel-lib/         # Library crate
//!     ├── Cargo.toml
//!     └── src/lib.rs      # Defines #[kernel] functions
//! ```
//!
//! ## What This Tests
//!
//! 1. Generic kernels defined in external crate (`kernel_lib::scale<T>`)
//! 2. Monomorphization at use site (`scale::<f32>` instantiated here)
//! 3. PTX generation for cross-crate kernels
//! 4. Const-generic entries instantiated for two values in the consuming crate
//! 5. Device helper functions from external crates
//!
//! ## Build and Run
//!
//! ```bash
//! cargo oxide run cross_crate_kernel
//! ```

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};

// Import the public kernel functions and their generated `<kernel>_ptx_name`
// helpers. Entry symbols and marker types remain implementation details.
use kernel_lib::kernels;

fn specialization_names() -> [&'static str; 4] {
    [
        kernels::scale_ptx_name::<f32>(),
        kernels::scale_ptx_name::<i32>(),
        kernels::add_const_ptx_name::<4>(),
        kernels::add_const_ptx_name::<8>(),
    ]
}

fn verify_generated_ptx() {
    use cuda_host::embedded::{ArtifactPayloadKind, artifact_bundles_from_current_exe};

    // Read the PTX back out of the artifact bundle embedded in this binary
    // rather than a loose `cross_crate_kernel.ptx`. These are the same bytes
    // the module load below hands to the driver, so the assertion cannot
    // diverge from the code that runs -- and it holds under
    // `--materialize-cubin`, where no loose file exists at all.
    let bundles = artifact_bundles_from_current_exe()
        .expect("failed to read embedded artifact bundles from the current executable");
    let bundle = bundles
        .into_iter()
        .find(|bundle| bundle.name == env!("CARGO_PKG_NAME"))
        .expect("embedded device bundle not found in the current executable");

    // A `--materialize-cubin` build embeds a cubin, which carries no PTX text.
    // Say so and move on: the entry names are still exercised by the launches
    // below. The wording deliberately avoids a leading `skipping:`, which the
    // smoketest reads as the whole example opting out.
    let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) else {
        println!(
            "note: embedded bundle holds a cubin, so there is no PTX text to \
             inspect; entry-name check not applicable"
        );
        return;
    };
    let ptx = std::str::from_utf8(ptx)
        .expect("embedded PTX payload is not valid UTF-8")
        .trim_end_matches('\0');
    let document = ptx_parse::Document::parse(ptx).expect("parse embedded PTX");

    for name in specialization_names() {
        assert!(
            document.callables_named(name).any(|callable| {
                callable.kind() == ptx_parse::CallableKind::Entry && callable.body_text().is_some()
            }),
            "missing or incomplete cross-crate PTX entry `{name}`"
        );
    }
}

fn main() {
    println!("=== Cross-Crate Kernel Test ===\n");
    println!("Testing kernels defined in kernel-lib crate.\n");

    if std::env::args().any(|arg| arg == "--print-specializations") {
        for name in specialization_names() {
            println!("{name}");
        }
        return;
    }
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        verify_generated_ptx();
        println!("cross-crate host lookup names match all four PTX entries");
        return;
    }

    verify_generated_ptx();

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // The device code is embedded in this binary, so there is no loose `.ptx`
    // to locate or keep in step.
    //
    // `kernels` is generic, and a generic module's generated loader merges the
    // PTX bundles of every crate in the build instead of selecting one by name
    // (it discards the crate-name hint outright). That merge is what makes the
    // specializations below resolvable at all: they are monomorphized here, in
    // the binary, from kernels declared in kernel-lib.
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // Test data
    const N: usize = 1024;

    // =========================================================================
    // Test 1: Generic scale kernel from library (f32)
    // =========================================================================
    println!("Test 1: kernel_lib::scale::<f32>");
    {
        let factor: f32 = 2.5;
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();

        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.scale::<f32>(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("Kernel launch failed");

        let output: Vec<f32> = output_dev.to_host_vec(&stream).unwrap();

        let errors = (0..N)
            .filter(|&i| (output[i] - input[i] * factor).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ PASSED: scale::<f32> from library works!\n");
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 2: Generic scale kernel from library (i32)
    // =========================================================================
    println!("Test 2: kernel_lib::scale::<i32>");
    {
        let factor: i32 = 3;
        let input: Vec<i32> = (0..N as i32).collect();

        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut output_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.scale::<i32>(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("Kernel launch failed");

        let output: Vec<i32> = output_dev.to_host_vec(&stream).unwrap();

        let errors = (0..N).filter(|&i| output[i] != input[i] * factor).count();

        if errors == 0 {
            println!("  ✓ PASSED: scale::<i32> from library works!\n");
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 3: Generic add kernel from library
    // =========================================================================
    println!("Test 3: kernel_lib::add::<f32>");
    {
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.add::<f32>(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
        }
        .expect("Kernel launch failed");

        let c: Vec<f32> = c_dev.to_host_vec(&stream).unwrap();

        let errors = (0..N)
            .filter(|&i| (c[i] - (a[i] + b[i])).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ PASSED: add::<f32> from library works!\n");
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 4: Kernel that uses device helper function from library
    // =========================================================================
    println!("Test 4: kernel_lib::scale_with_helper::<f32> (uses device helper)");
    {
        let factor: f32 = 4.0;
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();

        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.scale_with_helper::<f32>(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )
        }
        .expect("Kernel launch failed");

        let output: Vec<f32> = output_dev.to_host_vec(&stream).unwrap();

        let errors = (0..N)
            .filter(|&i| (output[i] - input[i] * factor).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ PASSED: scale_with_helper uses device function from library!\n");
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 5: Const-generic entries from the library
    // =========================================================================
    println!("Test 5: kernel_lib::add_const::<4/8>");
    {
        let input: Vec<u32> = (0..N as u32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut output_4 = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        let mut output_8 = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        let config = LaunchConfig::for_num_elems(N as u32);

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.add_const::<4>((stream).as_ref(), config, &input_dev, &mut output_4) }
            .expect("add_const::<4> launch failed");
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.add_const::<8>((stream).as_ref(), config, &input_dev, &mut output_8) }
            .expect("add_const::<8> launch failed");

        let result_4 = output_4.to_host_vec(&stream).unwrap();
        let result_8 = output_8.to_host_vec(&stream).unwrap();
        assert!((0..N).all(|i| result_4[i] == input[i] + 4));
        assert!((0..N).all(|i| result_8[i] == input[i] + 8));
        println!("  ✓ PASSED: const-generic library entries remain distinct!\n");
    }

    println!("=== All Cross-Crate Tests Passed! ===");
    println!("\nThis demonstrates:");
    println!("  - Generic kernels can be defined in library crates");
    println!("  - They are monomorphized when used in the application");
    println!("  - PTX is generated for all used instantiations");
    println!("  - Const values participate in cross-crate kernel identity");
    println!("  - Device helper functions from libraries also work");
}
