/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::needless_range_loop)]

//! Device FFI Test - cuda-oxide kernel calling external LTOIR functions
//!
//! This example demonstrates calling device functions defined in external LTOIR
//! (compiled from CUDA C++) from a cuda-oxide kernel.
//!
//! ## Usage
//!
//! ```bash
//! # Step 1: Generate cuda-oxide LLVM IR (requires cuda-oxide compiler)
//! cargo oxide run device_ffi_test --emit-nvvm-ir --arch=<your_arch>  # e.g., sm_120
//!
//! # Step 2: Build and run (handles everything else automatically)
//! cargo run --release
//! ```

use cuda_device::{device, gpu_printf, kernel, shared::DynamicSharedArray};
use cuda_host::cuda_module;

// =============================================================================
// External device functions from external_device_funcs.cu
//
// NOTE: No NVVM attributes (#[convergent], #[pure], #[readonly]) are needed here.
// The external LTOIR already contains proper attributes on the function definitions.
// When nvJitLink performs LTO linking, it uses the definition's attributes.
// See README.md "Why No NVVM Attributes?" for details.
// =============================================================================

#[device]
unsafe extern "C" {
    fn magnitude_squared(x: f32, y: f32) -> f32;
    fn fast_rsqrt(x: f32) -> f32;
    fn dot_product(a: *const f32, b: *const f32, n: i32) -> f32;
    fn warp_reduce_sum(val: f32) -> f32;
    fn warp_ballot(predicate: i32) -> u32;
    fn simple_add(a: f32, b: f32) -> f32;
    // `char` is a plain 32-bit device-extern slot. Rust still warns because C
    // has no native `char` equivalent, so acknowledge that contract here.
    #[allow(improper_ctypes)]
    fn char_to_upper(c: char) -> char;
    #[allow(improper_ctypes)]
    fn char_store(output: *mut char, c: char);
    fn clamp_value(val: f32, min_val: f32, max_val: f32) -> f32;
    fn smem_write_aligned_128(offset: i32, value: f32);
    fn smem_read_aligned_128(offset: i32) -> f32;
    fn smem_get_base_addr() -> u64;
}

// =============================================================================
// CCCL (CUB) wrapper functions from cccl_wrappers.cu
//
// These wrap CUB template functions as extern "C" for FFI.
// The CCCL LTOIR already has proper convergent/nounwind attributes.
// =============================================================================

#[device]
unsafe extern "C" {
    fn cub_block_reduce_sum_f32_256(input: f32, temp: *mut u8) -> f32;
    fn cub_block_reduce_max_f32_256(input: f32, temp: *mut u8) -> f32;
    fn cub_warp_reduce_sum_f32(input: f32) -> f32;
    fn cub_get_block_reduce_temp_size_256() -> i32;
}

// =============================================================================
// Test Kernels
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel using simple device functions
    #[kernel]
    pub fn test_simple_device_funcs(output: *mut f32) {
        let tid = cuda_device::thread::threadIdx_x();

        // Test pure function
        let x = tid as f32;
        let y = (tid + 1) as f32;
        let mag = unsafe { magnitude_squared(x, y) };

        // Test simple function
        let added = unsafe { simple_add(mag, 1.0) };

        // Test convergent function (all threads participate)
        let sum = unsafe { warp_reduce_sum(added) };

        // Lane 0 writes result
        if tid.is_multiple_of(32) {
            unsafe {
                *output.add((tid / 32) as usize) = sum;
            }
        }
    }

    /// Round-trip a `char` through an external CUDA `unsigned int` function.
    #[kernel]
    pub fn test_char_ffi(output: *mut u32) {
        let tid = cuda_device::thread::threadIdx_x();
        if tid == 0 {
            let upper = unsafe { char_to_upper('q') };
            let unchanged = unsafe { char_to_upper('Z') };
            unsafe {
                *output = upper as u32;
                *output.add(1) = unchanged as u32;
                char_store(output.add(2).cast::<char>(), 'λ');
            }
        }
    }

    /// Test kernel using CUB warp reduce
    #[kernel]
    pub fn test_cub_warp_reduce(input: *const f32, output: *mut f32) {
        let tid = cuda_device::thread::threadIdx_x();
        let lane = tid % 32;
        let warp_id = tid / 32;

        // Each thread loads a value
        let value = unsafe { *input.add(tid as usize) };

        // Warp-level reduction using CUB
        let warp_sum = unsafe { cub_warp_reduce_sum_f32(value) };

        // Lane 0 of each warp writes result
        if lane == 0 {
            unsafe {
                *output.add(warp_id as usize) = warp_sum;
            }
        }
    }

    /// Test kernel with multiple attribute combinations
    #[kernel]
    pub fn test_mixed_attrs(a: *const f32, b: *const f32, output: *mut f32, n: i32) {
        let tid = cuda_device::thread::threadIdx_x() as usize;

        // Use readonly function
        let dot = unsafe { dot_product(a, b, n) };

        // Use pure function
        let rsqrt = unsafe { fast_rsqrt(dot) };

        // Use convergent function
        let ballot = unsafe { warp_ballot(if rsqrt > 0.5 { 1 } else { 0 }) };
        unsafe {
            *output.add(tid) = rsqrt * (ballot as f32);
        }
    }

    /// Test dynamic shared memory alignment with extern functions.
    ///
    /// This kernel tests what happens when:
    /// - Rust uses DynamicSharedArray with 16B alignment (default)
    /// - Extern C++ function expects 128B alignment (TmaAligned128)
    ///
    /// The question: Does the linker take the max alignment (128B)?
    /// Or does Rust's 16B declaration "win" and cause misalignment?
    ///
    /// Scenario A (16B only from Rust side):
    /// ```
    /// .extern .shared .align 16 .b8 __dynamic_smem_test_smem_alignment_cross_module[];
    /// ```
    ///
    /// Scenario B (if linker merges correctly):
    /// ```
    /// .extern .shared .align 128 .b8 ...;  // max(16, 128) = 128
    /// ```
    #[kernel]
    pub fn test_smem_alignment_cross_module(output: *mut u64) {
        let tid = cuda_device::thread::threadIdx_x();

        // Rust side: request 16B alignment (deliberately low)
        let rust_smem: *mut f32 = DynamicSharedArray::<f32, 16>::get();

        // Get addresses from both Rust and extern C++ perspectives
        let rust_addr = rust_smem as u64;
        let extern_addr = unsafe { smem_get_base_addr() };

        if tid == 0 {
            gpu_printf!("[DEBUG] Rust smem addr: {:x}\n", rust_addr);
            gpu_printf!("[DEBUG] Extern smem addr: {:x}\n", extern_addr);
            gpu_printf!("[DEBUG] Before write: rust[0] = {:.2f}\n", unsafe {
                *rust_smem
            });
        }

        // Write via extern function (expects 128B alignment internally)
        if tid == 0 {
            unsafe {
                gpu_printf!("[DEBUG] Writing 42.0 via extern function...\n");
                smem_write_aligned_128(0, 42.0);
                gpu_printf!("[DEBUG] Write done\n");
            }
        }

        cuda_device::sync_threads();

        // Read back via Rust pointer
        let value_via_rust = unsafe { *rust_smem };

        // Read back via extern function
        let value_via_extern = unsafe { smem_read_aligned_128(0) };

        if tid == 0 {
            // Use pointer cast to get actual bits (to_bits() might not work on GPU)
            let rust_bits: u32 = unsafe { *((&value_via_rust) as *const f32 as *const u32) };
            let extern_bits: u32 = unsafe { *((&value_via_extern) as *const f32 as *const u32) };

            gpu_printf!("[DEBUG] After write:\n");
            gpu_printf!(
                "[DEBUG]   Rust read:   {:.2f} (bits via cast: {:x})\n",
                value_via_rust,
                rust_bits
            );
            gpu_printf!(
                "[DEBUG]   Extern read: {:.2f} (bits via cast: {:x})\n",
                value_via_extern,
                extern_bits
            );
            gpu_printf!(
                "[DEBUG]   to_bits():   rust={:x}, extern={:x}\n",
                value_via_rust.to_bits(),
                value_via_extern.to_bits()
            );

            // Also try direct write via Rust and read via extern
            gpu_printf!("[DEBUG] Writing 99.0 via Rust pointer...\n");
            unsafe { *rust_smem = 99.0 };
            let extern_read_after_rust_write = unsafe { smem_read_aligned_128(0) };
            gpu_printf!(
                "[DEBUG]   Extern read after Rust write: {:.2f}\n",
                extern_read_after_rust_write
            );

            // Output results using pointer cast for bits
            unsafe {
                *output.add(0) = rust_addr;
                *output.add(1) = extern_addr;
                *output.add(2) = rust_bits as u64;
                *output.add(3) = extern_bits as u64;
                *output.add(4) = rust_addr % 128;
                *output.add(5) = extern_addr % 128;
            }

            gpu_printf!(
                "[DEBUG] Output written: bits[2]={:x}, bits[3]={:x}\n",
                rust_bits,
                extern_bits
            );
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
    let external_ltoir = extern_libs_dir.join("external_device_funcs.ltoir");
    let cccl_ltoir = extern_libs_dir.join("cccl_wrappers.ltoir");
    let external_cu = extern_libs_dir.join("external_device_funcs.cu");
    let cccl_cu = extern_libs_dir.join("cccl_wrappers.cu");
    let arch_stamp = extern_libs_dir.join(".ltoir_arch");

    let arch = get_arch();
    let stamp_mismatch = std::fs::read_to_string(&arch_stamp)
        .map(|s| s.trim() != arch)
        .unwrap_or(true);
    let needs_rebuild = stamp_mismatch
        || file_needs_rebuild(&external_ltoir, &[&external_cu])
        || file_needs_rebuild(&cccl_ltoir, &[&cccl_cu]);

    if !needs_rebuild {
        return Ok(());
    }

    if stamp_mismatch && external_ltoir.exists() {
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
/// `cargo oxide run device_ffi_test --emit-nvvm-ir --arch=<your_arch>`  (e.g., sm_120)
fn compile_cuda_oxide_ltoir(example_dir: &Path) -> Result<(), String> {
    let ll_file = example_dir.join("device_ffi_test.ll");
    let options_file = example_dir.join("device_ffi_test.options");
    let ltoir_file = example_dir.join("device_ffi_test.ltoir");
    let tools_dir = example_dir.join("tools");

    if !ll_file.exists() {
        return Err(format!(
            "cuda-oxide LLVM IR not found: {}\n\
             Run: cargo oxide run device_ffi_test --emit-nvvm-ir --arch={}",
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
/// - `device_ffi_test.ltoir` (cuda-oxide kernels)
/// - `extern-libs/external_device_funcs.ltoir` (simple device functions)
/// - `extern-libs/cccl_wrappers.ltoir` (CUB wrappers)
///
/// Returns the path to the merged cubin on success.
fn link_ltoir(example_dir: &Path) -> Result<std::path::PathBuf, String> {
    let tools_dir = example_dir.join("tools");
    let extern_libs_dir = example_dir.join("extern-libs");
    let cubin_file = example_dir.join("merged.cubin");
    let cuda_oxide_ltoir = example_dir.join("device_ffi_test.ltoir");
    let external_ltoir = extern_libs_dir.join("external_device_funcs.ltoir");
    let cccl_ltoir = extern_libs_dir.join("cccl_wrappers.ltoir");

    let sources = [&cuda_oxide_ltoir, &external_ltoir, &cccl_ltoir];
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
            external_ltoir.to_str().unwrap(),
            cccl_ltoir.to_str().unwrap(),
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
// Uses cuda-driver to load the merged cubin, launch kernels, and verify results.
// =============================================================================

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use std::sync::Arc;

/// Main entry point - builds the pipeline and runs GPU tests.
fn main() {
    println!("=== Device FFI Test ===\n");

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

    test_simple_device_funcs_runner(&ctx, &module, &mut tests_passed, &mut tests_failed);
    test_char_ffi_runner(&ctx, &module, &mut tests_passed, &mut tests_failed);
    test_cub_warp_reduce_runner(&ctx, &module, &mut tests_passed, &mut tests_failed);
    test_mixed_attrs_runner(&ctx, &module, &mut tests_passed, &mut tests_failed);
    test_smem_alignment_cross_module_runner(&ctx, &module, &mut tests_passed, &mut tests_failed);

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
// Test Runners
//
// Each test runner:
// 1. Allocates device memory
// 2. Launches the corresponding kernel via the typed module API
// 3. Copies results back to host
// 4. Verifies the results
// =============================================================================

/// Test 1: Simple device function calls (magnitude_squared, simple_add, warp_reduce_sum)
///
/// Each warp computes: sum of (x² + y² + 1) for threads 0-31
/// where x = threadIdx and y = threadIdx + 1
fn test_simple_device_funcs_runner(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    passed: &mut i32,
    failed: &mut i32,
) {
    println!("--- Test 1: test_simple_device_funcs ---");
    println!("    Functions: magnitude_squared, simple_add, warp_reduce_sum");

    let n = 256usize;
    let warps = n / 32;

    let stream = ctx.default_stream();
    let d_output = DeviceBuffer::<f32>::zeroed(&stream, warps).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_simple_device_funcs(
            (stream).as_ref(),
            config,
            d_output.cu_deviceptr() as *mut f32,
        )
    }
    .expect("Kernel launch failed");

    let h_output = d_output.to_host_vec(&stream).unwrap();

    // Verify: each warp computes sum of (x² + y² + 1) for its threads
    let mut errors = 0;
    for warp in 0..warps {
        let base_tid = warp * 32;
        let mut expected = 0.0f32;
        for lane in 0..32 {
            let tid = base_tid + lane;
            let x = tid as f32;
            let y = (tid + 1) as f32;
            expected += x * x + y * y + 1.0;
        }
        if (h_output[warp] - expected).abs() > 1.0 {
            errors += 1;
        }
    }

    if errors == 0 {
        println!("    ✓ PASSED");
        *passed += 1;
    } else {
        println!("    ✗ FAILED ({} errors)", errors);
        *failed += 1;
    }
}

/// Test: `char` across a `#[device]` extern boundary.
fn test_char_ffi_runner(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    passed: &mut i32,
    failed: &mut i32,
) {
    println!("--- Test: test_char_ffi ---");

    let stream = ctx.default_stream();
    let d_output = DeviceBuffer::<u32>::zeroed(&stream, 3).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: the launch writes exactly three 32-bit values into d_output.
    unsafe {
        module.test_char_ffi(
            (stream).as_ref(),
            config,
            d_output.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Kernel launch failed");

    let actual = d_output.to_host_vec(&stream).unwrap();
    let expected = [u32::from('Q'), u32::from('Z'), u32::from('λ')];
    if actual[..3] == expected {
        println!("    ✓ PASSED ('q' -> 'Q', 'Z' unchanged, pointer -> 'λ')");
        *passed += 1;
    } else {
        println!(
            "    ✗ FAILED (got {:?}, expected {:?})",
            &actual[..3],
            expected
        );
        *failed += 1;
    }
}

/// Test 2: CUB warp-level reduction (cub_warp_reduce_sum_f32)
///
/// Each thread loads its lane ID (0-31), then CUB reduces within the warp.
/// Expected result: each warp outputs 0+1+2+...+31 = 496
fn test_cub_warp_reduce_runner(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    passed: &mut i32,
    failed: &mut i32,
) {
    println!("--- Test 2: test_cub_warp_reduce ---");
    println!("    Functions: cub_warp_reduce_sum_f32 (CCCL)");

    let n = 256usize;
    let warps = n / 32;

    let stream = ctx.default_stream();
    let h_input: Vec<f32> = (0..n).map(|i| (i % 32) as f32).collect();
    let d_input = DeviceBuffer::from_host(&stream, &h_input).unwrap();
    let d_output = DeviceBuffer::<f32>::zeroed(&stream, warps).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_cub_warp_reduce(
            (stream).as_ref(),
            config,
            d_input.cu_deviceptr() as *const f32,
            d_output.cu_deviceptr() as *mut f32,
        )
    }
    .expect("Kernel launch failed");

    let h_output = d_output.to_host_vec(&stream).unwrap();

    // Expected: sum of 0..31 = 496 for each warp
    let expected = 496.0f32;
    let errors = h_output
        .iter()
        .filter(|&&v| (v - expected).abs() > 1.0)
        .count();

    if errors == 0 {
        println!("    ✓ PASSED (all warps sum to 496)");
        *passed += 1;
    } else {
        println!("    ✗ FAILED ({} errors)", errors);
        *failed += 1;
    }
}

/// Test 3: Mixed device functions (dot_product, fast_rsqrt, warp_ballot)
///
/// Tests calling multiple external device functions together:
/// - `dot_product` - reads two vectors, computes dot product
/// - `fast_rsqrt` - computes fast inverse sqrt
/// - `warp_ballot` - warp-level ballot
///
/// Verifies outputs are finite (not NaN/Inf).
fn test_mixed_attrs_runner(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    passed: &mut i32,
    failed: &mut i32,
) {
    println!("--- Test 3: test_mixed_attrs ---");
    println!("    Functions: dot_product, fast_rsqrt, warp_ballot");

    let n = 32usize;
    let vec_size = 4i32;

    let stream = ctx.default_stream();
    let h_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let h_b = vec![1.0f32, 1.0, 1.0, 1.0];

    let d_a = DeviceBuffer::from_host(&stream, &h_a).unwrap();
    let d_b = DeviceBuffer::from_host(&stream, &h_b).unwrap();
    let d_output = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_mixed_attrs(
            (stream).as_ref(),
            config,
            d_a.cu_deviceptr() as *const f32,
            d_b.cu_deviceptr() as *const f32,
            d_output.cu_deviceptr() as *mut f32,
            vec_size,
        )
    }
    .expect("Kernel launch failed");

    let h_output = d_output.to_host_vec(&stream).unwrap();

    // Basic validation: outputs should be finite
    let valid = h_output.iter().all(|v| v.is_finite());

    if valid {
        println!("    ✓ PASSED (outputs are finite)");
        *passed += 1;
    } else {
        println!("    ✗ FAILED (invalid outputs)");
        *failed += 1;
    }
}

/// Test 4: Cross-module dynamic shared memory alignment
///
/// Tests alignment behavior when:
/// - Rust declares DynamicSharedArray with 16B alignment
/// - Extern C++ function internally expects 128B alignment
///
/// This verifies whether the linker takes max(16, 128) = 128.
fn test_smem_alignment_cross_module_runner(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    passed: &mut i32,
    failed: &mut i32,
) {
    println!("--- Test 4: test_smem_alignment_cross_module ---");
    println!("    Testing: Rust (16B align) + Extern C++ (128B align)");

    let stream = ctx.default_stream();
    let d_output = DeviceBuffer::<u64>::zeroed(&stream, 6).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 256, // Enough for the test
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.test_smem_alignment_cross_module(
            (stream).as_ref(),
            config,
            d_output.cu_deviceptr() as *mut u64,
        )
    }
    .expect("Kernel launch failed");

    // Synchronize to flush printf output
    stream.synchronize().expect("Sync failed");

    let h_output = d_output.to_host_vec(&stream).unwrap();

    let rust_addr = h_output[0];
    let extern_addr = h_output[1];
    let value_rust_bits = h_output[2] as u32;
    let value_extern_bits = h_output[3] as u32;
    let rust_align_mod = h_output[4];
    let extern_align_mod = h_output[5];

    let value_rust = f32::from_bits(value_rust_bits);
    let value_extern = f32::from_bits(value_extern_bits);

    // Extract low 32 bits for offset comparison (shared space vs generic space)
    let rust_offset = rust_addr & 0xFFFF_FFFF;
    let extern_offset = extern_addr & 0xFFFF_FFFF;

    println!("    Results:");
    println!(
        "      Rust smem addr:   0x{:x} (offset: 0x{:x}, mod 128 = {})",
        rust_addr, rust_offset, rust_align_mod
    );
    println!(
        "      Extern smem addr: 0x{:x} (offset: 0x{:x}, mod 128 = {})",
        extern_addr, extern_offset, extern_align_mod
    );
    println!("      Value via Rust:   {}", value_rust);
    println!("      Value via Extern: {}", value_extern);

    // Check if offsets match (same shared memory location)
    let offsets_match = rust_offset == extern_offset;

    // Check if values match (correctly shared)
    let values_match = (value_rust - 42.0).abs() < 0.001 && (value_extern - 42.0).abs() < 0.001;

    // Check alignment - if linker took max, both should be 128B aligned
    let is_128_aligned = rust_align_mod == 0 && extern_align_mod == 0;

    println!("    Analysis:");
    println!(
        "      Offsets match: {} (both at 0x{:x})",
        offsets_match, rust_offset
    );
    println!("      128B aligned:  {}", is_128_aligned);
    println!("      Values = 42.0: {}", values_match);

    if offsets_match && values_match {
        println!("    ✓ PASSED (same shared memory, values correct)");
        *passed += 1;
    } else if offsets_match && !values_match {
        println!("    ⚠ PARTIAL: Same offset but values wrong!");
        println!("      → Rust/extern use different symbols that resolve to same offset");
        println!("      → But writes via one symbol don't appear via the other");
        *failed += 1;
    } else {
        println!("    ✗ FAILED");
        if !offsets_match {
            println!("      - Offsets don't match (different allocations)");
        }
        if !values_match {
            println!("      - Values don't match");
        }
        *failed += 1;
    }
}
