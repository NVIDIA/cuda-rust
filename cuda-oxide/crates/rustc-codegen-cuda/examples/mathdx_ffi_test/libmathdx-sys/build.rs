/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Build script for libmathdx-sys
//!
//! This script:
//! 1. Uses bindgen to generate Rust FFI bindings from libmathdx headers
//! 2. Links against libmathdx
//!
//! Configuration:
//! - Set `LIBMATHDX_PATH` to the libmathdx installation directory (containing include/ and lib/)
//! - Recommended: Configure in `.cargo/config.toml` for persistence across sessions

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LIBMATHDX_PATH");
    println!("cargo:rerun-if-changed=wrapper.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find libmathdx installation
    let libmathdx_path = find_libmathdx();

    if let Some(ref install_path) = libmathdx_path {
        let include_path = install_path.join("include");
        let lib_path = install_path.join("lib");

        // Link against libmathdx dynamically
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=dylib=mathdx");

        // Export the lib path as metadata (can be read by dependent crates via DEP_MATHDX_LIB_DIR)
        // Note: cargo oxide reads LIBMATHDX_PATH directly to set LD_LIBRARY_PATH at runtime
        println!("cargo:lib_dir={}", lib_path.display());

        // Generate bindings with bindgen
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg(format!("-I{}", include_path.display()))
            // Parse all libmathdx types and functions
            .allowlist_function("mathdx.*")
            .allowlist_function("commondx.*")
            .allowlist_function("cublasdx.*")
            .allowlist_function("cufftdx.*")
            .allowlist_function("cusolverdx.*")
            .allowlist_function("curanddx.*")
            .allowlist_type("commondx.*")
            .allowlist_type("cublasdx.*")
            .allowlist_type("cufftdx.*")
            .allowlist_type("cusolverdx.*")
            .allowlist_type("curanddx.*")
            .allowlist_var("COMMONDX.*")
            .allowlist_var("CUBLASDX.*")
            .allowlist_var("CUFFTDX.*")
            .allowlist_var("LIBMATHDX.*")
            // Derive useful traits
            .derive_debug(true)
            .derive_default(true)
            .derive_eq(true)
            .derive_hash(true)
            // Use core types
            .use_core()
            // Generate documentation
            .generate_comments(true)
            .generate()
            .expect("Unable to generate bindings");

        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    } else {
        // No libmathdx found - create empty bindings file
        println!(
            "cargo:warning=libmathdx not found. Set LIBMATHDX_PATH to the installation directory."
        );
        std::fs::write(
            out_path.join("bindings.rs"),
            "// libmathdx not found - no bindings generated\n",
        )
        .expect("Couldn't write empty bindings file");
    }
}

fn find_libmathdx() -> Option<PathBuf> {
    // 1. Check LIBMATHDX_PATH environment variable
    if let Ok(path) = env::var("LIBMATHDX_PATH") {
        let path = PathBuf::from(&path);
        // Check if this looks like a libmathdx installation
        if path.join("include").join("libmathdx.h").exists() {
            return Some(path);
        }
        // Maybe they pointed directly at lib/ or include/
        if path.join("libmathdx.h").exists() {
            return path.parent().map(|p| p.to_path_buf());
        }
    }

    // 2. Check common installation paths
    let common_paths = ["/usr/local", "/opt/nvidia/libmathdx"];

    for base in &common_paths {
        let path = PathBuf::from(base);
        if path.join("include").join("libmathdx.h").exists() {
            return Some(path);
        }
    }

    None
}
