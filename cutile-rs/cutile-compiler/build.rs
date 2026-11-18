/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Build-time device debug defaults, the source digest the JIT cache keys
//! on, and an advisory Tile-floor warning.
//!
//! The authoritative check runs at tool discovery in
//! `cuda_tile_runtime_utils`, on the machine that executes the JIT. This
//! warning covers the common case where the build box and the run box are
//! the same, so a too-old toolkit is reported at compile time instead of
//! first launch. The toolkit check never fails the build: emitting bytecode and
//! cross-building are legitimate on machines without a 13.2+ toolkit.

use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TOOLKIT_ENV_VARS: &[&str] = &["CUDA_TOOLKIT_PATH", "CUDA_HOME"];
const DEFAULT_TOOLKIT_DIR: &str = "/usr/local/cuda";
const MIN_TILE_CUDA_VERSION: u32 = 13020;

fn main() {
    println!(
        "cargo:rustc-env=CUTILE_COMPILER_SOURCE_DIGEST={}",
        source_digest()
    );
    println!("cargo:rerun-if-env-changed=CUDA_RUST_DEBUG");
    let debug_info = match env::var("CUDA_RUST_DEBUG") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => {
            // Cargo supplies the profile of this library, not the profile of
            // the build script or proc macro. `debug_assertions` is a separate
            // setting and must not select device debug information.
            // DEBUG is only a boolean: Cargo does not expose the distinction
            // between line tables, limited, and full debug info here.
            match env::var("DEBUG").as_deref() {
                Ok("true") => "full".to_string(),
                Ok("false") => "none".to_string(),
                other => panic!("expected Cargo's DEBUG=true or false, got {other:?}"),
            }
        }
        Err(error) => panic!("CUDA_RUST_DEBUG must be none, line, or full: {error}"),
    };
    assert!(
        matches!(debug_info.as_str(), "none" | "line" | "full"),
        "invalid CUDA_RUST_DEBUG={debug_info:?}; expected none, line, or full"
    );
    println!("cargo:rustc-env=CUTILE_BUILD_DEBUG_INFO={debug_info}");

    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let toolkit = TOOLKIT_ENV_VARS
        .iter()
        .find_map(|var| env::var(var).ok())
        .unwrap_or_else(|| DEFAULT_TOOLKIT_DIR.to_string());
    let cuda_h = Path::new(&toolkit).join("include").join("cuda.h");
    let Ok(header) = fs::read_to_string(&cuda_h) else {
        return;
    };
    let version = header.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next(), parts.next()) {
            (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse::<u32>().ok(),
            _ => None,
        }
    });
    if let Some(version) = version {
        if version < MIN_TILE_CUDA_VERSION {
            println!(
                "cargo:warning=cutile-compiler: the toolkit at {} is CUDA {}.{}; \
                 cuTile requires CUDA 13.2+ at run time (tileiras ships with it). \
                 The shared CUDA host-side crates support 13.0+.",
                toolkit,
                version / 1000,
                (version % 1000) / 10
            );
        }
    }
}

/// SHA-256 over `src/`, sorted relative paths and contents: cuTile's JIT
/// cache keys on it, so any edit to this crate retires cached front-end
/// output, not only a version bump.
fn source_digest() -> String {
    println!("cargo:rerun-if-changed=src");
    let mut files = Vec::new();
    let mut dirs = vec![PathBuf::from("src")];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).expect("read src/") {
            let path = entry.expect("read src/ entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut h = Sha256::new();
    for path in files {
        let name = path.to_string_lossy().into_owned();
        let contents = fs::read(&path).expect("read source file");
        for field in [name.as_bytes(), &contents] {
            h.update((field.len() as u64).to_le_bytes());
            h.update(field);
        }
    }
    format!("{:x}", h.finalize())
}
