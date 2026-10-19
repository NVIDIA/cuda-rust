/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

//! Small Type FFI Test - cuda-oxide kernel calling small-scalar LTOIR functions
//!
//! This example exercises sub-32-bit scalars (i8/u8/i16/u16/bool) and f16
//! across the device FFI boundary, in both parameter and return position,
//! against nvcc-compiled LTOIR definitions.
//!
//! It requires the modern NVVM path (sm_100+): the legacy CUDA 12 LLVM 7
//! dialect rejects scalar f16 and sub-32-bit by-value externs by design.
//!
//! ## Usage
//!
//! ```bash
//! cargo oxide run small_type_ffi_test --emit-nvvm-ir --arch=<your_arch>  # e.g., sm_120
//! ```

use cuda_device::{device, kernel};
use cuda_host::cuda_module;

// =============================================================================
// Small scalar ABI functions from small_type_funcs.cu
//
// Sub-32-bit scalars (i8/u8/i16/u16/bool) cross the boundary as narrow LLVM
// types with signext/zeroext attributes; f16 is passed directly as `half`.
// The emitted declarations must match the nvcc-compiled LTOIR definitions.
// =============================================================================

#[device]
unsafe extern "C" {
    fn small_widen_i8(x: i8) -> i32;
    fn small_widen_u16(x: u16) -> u32;
    fn small_scale_i8(x: i8) -> i8;
    fn small_add_u8(a: u8, b: u8) -> u8;
    fn small_scale_i16(x: i16) -> i16;
    fn small_add_u16(a: u16, b: u16) -> u16;
    fn small_not_bool(b: bool) -> bool;
    fn small_half_add(a: f16, b: f16) -> f16;
}

// =============================================================================
// Test Kernels
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test small scalar types (i8/u8/i16/u16/bool/f16) across the FFI
    /// boundary, in both parameter and return position.
    #[kernel]
    pub fn test_small_type_ffi(output: *mut i32) {
        let tid = cuda_device::thread::threadIdx_x();
        if tid == 0 {
            // Sign extension of a negative i8 parameter.
            let widened_i8 = unsafe { small_widen_i8(-3) };
            // Zero extension of a high-bit u16 parameter: must stay 65280,
            // not sign-extend to -256.
            let widened_u16 = unsafe { small_widen_u16(0xFF00) } as i32;
            // Small returns: signext i8 / zeroext u8/u16 and i16 round-trips.
            let scaled_i8 = unsafe { small_scale_i8(-5) } as i32;
            let added_u8 = unsafe { small_add_u8(200, 55) } as i32;
            let scaled_i16 = unsafe { small_scale_i16(-1000) } as i32;
            let added_u16 = unsafe { small_add_u16(65000, 500) } as i32;
            // bool round-trip in both directions.
            let not_false = unsafe { small_not_bool(false) };
            let not_true = unsafe { small_not_bool(true) };
            let bools = (if not_false { 1 } else { 0 }) | (if not_true { 2 } else { 0 });
            // f16 round-trip: 1.5 + 2.0 = 3.5 (half bits 0x4300).
            let half_sum =
                unsafe { small_half_add(f16::from_bits(0x3e00), f16::from_bits(0x4000)) };
            unsafe {
                *output.add(0) = widened_i8;
                *output.add(1) = widened_u16;
                *output.add(2) = scaled_i8;
                *output.add(3) = added_u8;
                *output.add(4) = scaled_i16;
                *output.add(5) = added_u16;
                *output.add(6) = bools;
                *output.add(7) = half_sum.to_bits() as i32;
            }
        }
    }
}

// =============================================================================
// Build Pipeline
//
// These functions automate the LTOIR build process:
// 1. Build C tools (compile_ltoir, link_ltoir) if not present
// 2. Build external CUDA C++ to LTOIR if sources changed
// 3. Compile cuda-oxide LLVM IR to LTOIR
// 4. Link all LTOIR files into a cubin
// =============================================================================

use std::path::Path;
use std::process::Command;

/// Target GPU architecture for LTOIR compilation and linking.
/// Override with CUDA_OXIDE_TARGET environment variable.
fn get_arch() -> &'static str {
    // Check environment variable first
    if let Ok(target) = std::env::var("CUDA_OXIDE_TARGET") {
        // Leak the string to get 'static lifetime (fine for a CLI tool)
        return Box::leak(target.into_boxed_str());
    }
    // Default to sm_120 (RTX 5090 consumer Blackwell)
    "sm_120"
}

/// Returns the path to this example's directory (where Cargo.toml lives).
fn get_example_dir() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).to_path_buf()
}

/// Runs a shell command in the specified directory.
///
/// # Arguments
/// * `cmd` - The command to run
/// * `args` - Arguments to pass to the command
/// * `cwd` - Working directory for the command
///
/// # Returns
/// `Ok(())` on success, `Err(message)` on failure
fn run_command(cmd: &str, args: &[&str], cwd: &Path) -> Result<(), String> {
    let status = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .status()
        .map_err(|e| format!("Failed to run {}: {}", cmd, e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed with exit code {:?}", cmd, status.code()))
    }
}

/// Checks if a target file needs to be rebuilt based on source file timestamps.
///
/// Returns `true` if:
/// - The target file doesn't exist, OR
/// - Any source file is newer than the target
fn file_needs_rebuild(target: &Path, sources: &[&Path]) -> bool {
    if !target.exists() {
        return true;
    }
    let target_time = target.metadata().and_then(|m| m.modified()).ok();
    for src in sources {
        if let Ok(src_time) = src.metadata().and_then(|m| m.modified())
            && target_time.map(|t| src_time > t).unwrap_or(true)
        {
            return true;
        }
    }
    false
}

/// Builds the LTOIR tools (compile_ltoir, link_ltoir) if they don't exist.
///
/// These C tools wrap libNVVM and nvJitLink respectively.
fn build_tools(example_dir: &Path) -> Result<(), String> {
    let tools_dir = example_dir.join("tools");
    let compile_ltoir = tools_dir.join("compile_ltoir");
    let link_ltoir = tools_dir.join("link_ltoir");
    let compile_source = tools_dir.join("compile_ltoir.c");
    let link_source = tools_dir.join("link_ltoir.c");
    let options_header = tools_dir.join("compile_options.h");
    let build_script = tools_dir.join("build_tools.sh");

    let compile_sources = [
        compile_source.as_path(),
        options_header.as_path(),
        build_script.as_path(),
    ];
    let link_sources = [
        link_source.as_path(),
        options_header.as_path(),
        build_script.as_path(),
    ];
    if !file_needs_rebuild(&compile_ltoir, &compile_sources)
        && !file_needs_rebuild(&link_ltoir, &link_sources)
    {
        return Ok(());
    }

    println!("=== Building LTOIR tools ===");
    run_command("./build_tools.sh", &[], &tools_dir)?;
    println!("  ✓ Tools built\n");
    Ok(())
}

/// Builds external CUDA C++ files to LTOIR if sources have changed or the
/// cached LTOIR was built for a different architecture.
///
/// Compiles `extern-libs/*.cu` to LTOIR using nvcc with `-dc -dlto` flags.
///
/// ## Arch stamping
///
/// Plain mtime-based caching is not enough: nvJitLink rejects linking an
/// LTOIR built for arch `X` into a cubin targeted at arch `Y` (it errors with
/// `ARCH_MISMATCH`). When the user switches `--arch`, the cached `.ltoir`
/// from a prior build is newer than the `.cu` but was built for the wrong
/// arch. We side-step that by writing a `.ltoir_arch` stamp next to the
/// LTOIR files; a mismatch forces a rebuild.
fn build_external_ltoir(example_dir: &Path) -> Result<(), String> {
    let extern_libs_dir = example_dir.join("extern-libs");
    let small_type_ltoir = extern_libs_dir.join("small_type_funcs.ltoir");
    let small_type_cu = extern_libs_dir.join("small_type_funcs.cu");
    let arch_stamp = extern_libs_dir.join(".ltoir_arch");

    let arch = get_arch();
    let stamp_mismatch = std::fs::read_to_string(&arch_stamp)
        .map(|s| s.trim() != arch)
        .unwrap_or(true);
    let needs_rebuild = stamp_mismatch || file_needs_rebuild(&small_type_ltoir, &[&small_type_cu]);

    if !needs_rebuild {
        return Ok(());
    }

    if stamp_mismatch && small_type_ltoir.exists() {
        println!(
            "=== Rebuilding external LTOIR (arch changed → {}) ===",
            arch
        );
    } else {
        println!("=== Building external LTOIR ({}) ===", arch);
    }
    run_command("./build_ltoir.sh", &[arch], &extern_libs_dir)?;
    std::fs::write(&arch_stamp, arch).map_err(|e| format!("Failed to write arch stamp: {}", e))?;
    println!("  ✓ External LTOIR built\n");
    Ok(())
}

/// Compiles cuda-oxide LLVM IR (.ll) to LTOIR using libNVVM.
///
/// Requires the .ll file to be generated first by:
/// `cargo oxide run small_type_ffi_test --emit-nvvm-ir --arch=<your_arch>`  (e.g., sm_120)
fn compile_cuda_oxide_ltoir(example_dir: &Path) -> Result<(), String> {
    let ll_file = example_dir.join("small_type_ffi_test.ll");
    let options_file = example_dir.join("small_type_ffi_test.options");
    let ltoir_file = example_dir.join("small_type_ffi_test.ltoir");
    let tools_dir = example_dir.join("tools");

    if !ll_file.exists() {
        return Err(format!(
            "cuda-oxide LLVM IR not found: {}\n\
             Run: cargo oxide run small_type_ffi_test --emit-nvvm-ir --arch={}",
            ll_file.display(),
            get_arch()
        ));
    }

    if !file_needs_rebuild(&ltoir_file, &[&ll_file, &options_file]) {
        return Ok(());
    }

    println!("=== Compiling cuda-oxide LLVM IR to LTOIR ===");
    run_command(
        "./compile_ltoir",
        &[
            ll_file.to_str().unwrap(),
            get_arch(),
            ltoir_file.to_str().unwrap(),
        ],
        &tools_dir,
    )?;
    println!("  ✓ cuda-oxide LTOIR compiled\n");
    Ok(())
}

/// Links all LTOIR files into a single cubin using nvJitLink.
///
/// Combines:
/// - `small_type_ffi_test.ltoir` (cuda-oxide kernel)
/// - `extern-libs/small_type_funcs.ltoir` (small scalar ABI functions)
///
/// Returns the path to the merged cubin on success.
fn link_ltoir(example_dir: &Path) -> Result<std::path::PathBuf, String> {
    let tools_dir = example_dir.join("tools");
    let extern_libs_dir = example_dir.join("extern-libs");
    let cubin_file = example_dir.join("merged.cubin");
    let cuda_oxide_ltoir = example_dir.join("small_type_ffi_test.ltoir");
    let small_type_ltoir = extern_libs_dir.join("small_type_funcs.ltoir");

    let sources = [&cuda_oxide_ltoir, &small_type_ltoir];
    let source_refs: Vec<&Path> = sources.iter().map(|p| p.as_path()).collect();

    if !file_needs_rebuild(&cubin_file, &source_refs) {
        return Ok(cubin_file);
    }

    println!("=== Linking LTOIR files ===");
    run_command(
        "./link_ltoir",
        &[
            &format!("-arch={}", get_arch()),
            "-o",
            cubin_file.to_str().unwrap(),
            cuda_oxide_ltoir.to_str().unwrap(),
            small_type_ltoir.to_str().unwrap(),
        ],
        &tools_dir,
    )?;
    println!("  ✓ LTOIR linked to cubin\n");
    Ok(cubin_file)
}

/// Runs the complete build pipeline.
///
/// 1. Builds tools (if needed)
/// 2. Builds external LTOIR (if sources changed)
/// 3. Compiles cuda-oxide LLVM IR to LTOIR
/// 4. Links all LTOIR to cubin
///
/// Returns the path to the final cubin on success.
fn build_pipeline() -> Result<std::path::PathBuf, String> {
    let example_dir = get_example_dir();

    build_tools(&example_dir)?;
    build_external_ltoir(&example_dir)?;
    compile_cuda_oxide_ltoir(&example_dir)?;
    link_ltoir(&example_dir)
}

// =============================================================================
// Test Harness
//
// Uses cuda-driver to load the merged cubin, launch the kernel, and verify
// the results.
// =============================================================================

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use std::sync::Arc;

/// Main entry point - builds the pipeline and runs the GPU test.
fn main() {
    println!("=== Small Type FFI Test ===\n");

    // Build pipeline (tools, external LTOIR, link)
    let cubin_path = match build_pipeline() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("Build failed: {}", e);
            std::process::exit(1);
        }
    };

    // Load and run tests
    println!("=== Running GPU Tests ===");
    println!("Cubin: {}", cubin_path.display());

    let cubin_data = std::fs::read(&cubin_path).expect("Failed to read cubin file");
    println!("Loaded {} bytes", cubin_data.len());

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("Device ordinal: {}\n", ctx.ordinal());

    let cubin_path_str = cubin_path.to_str().expect("cubin path must be UTF-8");
    let module = ctx
        .load_module_from_file(cubin_path_str)
        .expect("Failed to load cubin module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    let mut tests_passed = 0;
    let mut tests_failed = 0;

    test_small_type_ffi_runner(&ctx, &module, &mut tests_passed, &mut tests_failed);

    println!("\n=== Summary ===");
    println!("Passed: {}", tests_passed);
    println!("Failed: {}", tests_failed);

    if tests_failed == 0 && tests_passed > 0 {
        println!("\n✓ All tests PASSED!");
    } else if tests_passed == 0 {
        println!("\nNo tests ran (kernels not found in cubin)");
        std::process::exit(1);
    } else {
        println!("\n✗ Some tests FAILED");
        std::process::exit(1);
    }
}

// =============================================================================
// Test Runner
// =============================================================================

/// Test 1: Small scalar types across the FFI boundary
///
/// Verifies i8/u8/i16/u16/bool/f16 parameters AND returns against the
/// nvcc-compiled LTOIR definitions:
/// - sign extension of a negative i8 parameter (`small_widen_i8(-3)` = -3)
/// - zero extension of a high-bit u16 parameter (0xFF00 stays 65280)
/// - small-return round-trips for i8/u8/i16/u16
/// - bool round-trip in both directions
/// - f16 round-trip (1.5 + 2.0 = 3.5)
fn test_small_type_ffi_runner(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    passed: &mut i32,
    failed: &mut i32,
) {
    println!("--- Test 1: test_small_type_ffi ---");
    println!("    Functions: small_widen_i8/u16, small_scale_i8/i16, small_add_u8/u16,");
    println!("               small_not_bool, small_half_add");

    let stream = ctx.default_stream();
    let d_output = DeviceBuffer::<i32>::zeroed(&stream, 8).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_small_type_ffi(
            (stream).as_ref(),
            config,
            d_output.cu_deviceptr() as *mut i32,
        )
    }
    .expect("Kernel launch failed");

    let h_output = d_output.to_host_vec(&stream).unwrap();

    let expected: [(i32, &str); 8] = [
        (-3, "small_widen_i8(-3): negative i8 sign extension"),
        (65280, "small_widen_u16(0xFF00): u16 zero extension"),
        (-10, "small_scale_i8(-5): signext i8 return"),
        (255, "small_add_u8(200, 55): zeroext u8 return"),
        (-3000, "small_scale_i16(-1000): signext i16 return"),
        (65500, "small_add_u16(65000, 500): zeroext u16 return"),
        (0b01, "small_not_bool round-trips"),
        (0x4300, "small_half_add(1.5, 2.0): f16 round-trip"),
    ];

    let mut errors = 0;
    for (slot, (want, what)) in expected.iter().enumerate() {
        if h_output[slot] != *want {
            println!(
                "    MISMATCH [{}] {}: expected {}, got {}",
                slot, what, want, h_output[slot]
            );
            errors += 1;
        }
    }

    if errors == 0 {
        println!("    ✓ PASSED (all small-type round-trips correct)");
        *passed += 1;
    } else {
        println!("    ✗ FAILED ({} errors)", errors);
        *failed += 1;
    }
}
