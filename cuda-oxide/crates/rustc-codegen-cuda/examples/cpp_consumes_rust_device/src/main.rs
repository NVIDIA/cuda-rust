/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! C++ Consumes Rust Device Functions — via LTOIR
//!
//! This crate defines `#[device]` functions that are compiled to NVVM IR by
//! cuda-oxide. The NVVM IR is then compiled to LTOIR (by libNVVM) and linked
//! with a C++ kernel (by nvJitLink). The C++ side handles building, linking,
//! and GPU execution.
//!
//! ## Pipeline
//!
//! ```text
//! This crate (Rust #[device] fns)
//!     │
//!     ▼
//! cargo oxide run cpp_consumes_rust_device --emit-nvvm-ir --arch=<your_arch>  # e.g., sm_120
//!     │
//!     ▼
//! cpp_consumes_rust_device.ll (NVVM IR)
//!     │
//!     ▼  (handed off to C++ build pipeline — see cuda-caller/run_test.sh)
//! ```
//!
//! ## What This Crate Contains
//!
//! 1. Simple device functions (fast_sqrt, clamp_f32)
//! 2. Device function calling another (safe_sqrt → fast_sqrt + clamp_f32)
//! 3. Generic device function instantiations (fma_f32, fma_i32)
//!
//! The `main()` function is host-only — it just verifies the .ll was generated.
//! GPU testing is done by the C++ test runner in `cuda-caller/`.

use cuda_device::device;

// =============================================================================
// Device Functions — compiled to LTOIR, consumed by C++
// =============================================================================

/// Newton's method square root.
#[device]
pub fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x * 0.5;
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess
}

/// Simple clamp function.
#[device]
pub fn clamp_f32(val: f32, min_val: f32, max_val: f32) -> f32 {
    if val < min_val {
        min_val
    } else if val > max_val {
        max_val
    } else {
        val
    }
}

/// Clamped square root — calls fast_sqrt and clamp_f32.
#[device]
pub fn safe_sqrt(x: f32) -> f32 {
    let clamped = clamp_f32(x, 0.0, 1e10);
    fast_sqrt(clamped)
}

/// Generic fused multiply-add.
#[device]
pub fn fma<T: core::ops::Mul<Output = T> + core::ops::Add<Output = T>>(a: T, b: T, c: T) -> T {
    a * b + c
}

/// Concrete f32 FMA wrapper.
#[device]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    fma(a, b, c)
}

/// Concrete i32 FMA wrapper.
#[device]
pub fn fma_i32(a: i32, b: i32, c: i32) -> i32 {
    fma(a, b, c)
}

// =============================================================================
// Host main — verifies NVVM IR was generated
// =============================================================================

fn main() {
    println!("=== C++ Consumes Rust Device — Rust side ===\n");

    let ll_path = "cpp_consumes_rust_device.ll";

    if std::path::Path::new(ll_path).exists() {
        let content = std::fs::read_to_string(ll_path).unwrap();
        println!("NVVM IR generated: {} ({} bytes)", ll_path, content.len());

        // Verify clean export names (no reserved cuda_oxide_device_<hash>_ prefix)
        let expected = ["fast_sqrt", "clamp_f32", "safe_sqrt", "fma_f32", "fma_i32"];
        let mut all_found = true;
        for name in &expected {
            if content.contains(&format!("@{}", name)) {
                println!("  ✓ {} — found with clean name", name);
            } else {
                println!("  ✗ {} — NOT found", name);
                all_found = false;
            }
        }

        // Verify @llvm.used is present (for libNVVM DCE protection)
        if content.contains("@llvm.used") {
            println!("  ✓ @llvm.used — present");
        } else {
            println!("  ✗ @llvm.used — missing (libNVVM may discard functions)");
            all_found = false;
        }

        // Verify !nvvmir.version metadata
        if content.contains("!nvvmir.version") {
            println!("  ✓ !nvvmir.version — present");
        } else {
            println!("  ✗ !nvvmir.version — missing (libNVVM will reject)");
            all_found = false;
        }

        if all_found {
            println!("\n✓ NVVM IR is ready for libNVVM compilation.");
            println!("  Next:");
            println!(
                "    cd crates/rustc-codegen-cuda/examples/cpp_consumes_rust_device/cuda-caller"
            );
            println!("    ./run_test.sh");
        } else {
            println!("\n✗ NVVM IR has issues — check above.");
            std::process::exit(1);
        }
    } else {
        println!("No .ll file found. Generate NVVM IR first (from workspace root):");
        println!(
            "  cargo oxide run cpp_consumes_rust_device --emit-nvvm-ir --arch=<your_arch>  # e.g., sm_120"
        );
    }
}
