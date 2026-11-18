/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The source digest cuTile's JIT cache keys on.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

fn main() {
    println!(
        "cargo:rustc-env=CUTILE_IR_SOURCE_DIGEST={}",
        source_digest()
    );
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
