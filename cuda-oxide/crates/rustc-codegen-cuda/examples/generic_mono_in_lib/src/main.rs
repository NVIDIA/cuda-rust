/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! All-generic `#[cuda_module]` monomorphized in a library (issue #1365).
//!
//! ## Structure
//!
//! ```text
//! generic_mono_in_lib/
//! ├── Cargo.toml              # Thin binary
//! ├── src/main.rs             # Calls library host API only
//! └── generic-mono-lib/       # All-generic cuda_module + fill_f64 mono
//! ```
//!
//! Unlike `cross_crate_embedded` (generics mono only in the binary) and
//! `cuda_module_in_lib` (concrete kernels in the library), this library owns
//! an all-generic module *and* monomorphizes it locally. The binary never
//! names a kernel type; it only calls `fill_f64`.
//!
//! Before the fix, the library `.oxart` was dropped from the rlib and
//! `kernels::load` panicked with `NoModules`.
//!
//! ## Build and Run
//!
//! ```bash
//! cargo oxide run generic_mono_in_lib
//! # GPU-less linkage check:
//! cargo oxide run -- generic_mono_in_lib -- --verify-bundles
//! ```

const LIB_BUNDLE: &str = "generic-mono-lib";

fn main() {
    println!("=== all-generic cuda_module mono in library (issue #1365) ===\n");

    println!("Test 1: library bundle survived archive linking");
    let bundle_names: Vec<String> = cuda_host::embedded::artifact_bundles_from_current_exe()
        .expect("failed to parse the current executable")
        .into_iter()
        .map(|bundle| bundle.name)
        .collect();
    println!("  found bundles: {bundle_names:?}");
    if !bundle_names.iter().any(|name| name == LIB_BUNDLE) {
        println!("  FAILED: bundle '{LIB_BUNDLE}' is missing from the executable");
        std::process::exit(1);
    }
    println!("  PASSED: library package bundle is embedded\n");

    if std::env::args().any(|arg| arg == "--verify-bundles") {
        println!("SUCCESS: generic-mono library bundle survived archive linking");
        return;
    }

    println!("Test 2: fill_f64 host API (library mono + load)");
    let out = generic_mono_lib::fill_f64(4, 1.5);
    assert_eq!(out, [1.5; 4]);
    println!("  PASSED: fill_f64 returned [1.5; 4]\n");
    println!("ok");
    println!("SUCCESS: all-generic cuda_module mono in library loads and runs");
}
