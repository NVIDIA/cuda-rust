/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Target-aware platform helpers for cargo-oxide.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

/// Returns true when `target` is a Windows target triple.
pub fn is_windows_target(target: &str) -> bool {
    target.contains("windows")
}

/// Returns true when `target` is an Apple/Darwin target triple.
pub fn is_apple_target(target: &str) -> bool {
    target.contains("apple") || target.contains("darwin")
}

/// Shared-library filename for a Rust codegen backend crate on `target`.
pub fn dylib_filename(crate_name: &str, target: &str) -> String {
    if is_windows_target(target) {
        format!("{crate_name}.dll")
    } else if is_apple_target(target) {
        format!("lib{crate_name}.dylib")
    } else {
        format!("lib{crate_name}.so")
    }
}

/// Object-file extension used by `target`.
#[allow(dead_code)]
pub fn object_extension(target: &str) -> &'static str {
    if is_windows_target(target) {
        "obj"
    } else {
        "o"
    }
}

/// Executable filename for `name` on `target`.
pub fn executable_filename(name: &str, target: &str) -> String {
    if is_windows_target(target) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// Dynamic loader search-path environment variable for `target`.
pub fn loader_env_var(target: &str) -> &'static str {
    if is_windows_target(target) {
        "PATH"
    } else if is_apple_target(target) {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Joins env paths with the platform-native separator.
pub fn join_env_paths(paths: Vec<PathBuf>) -> Option<OsString> {
    std::env::join_paths(paths).ok()
}

/// Returns `new_paths` followed by the current value of `env_var`.
pub fn prepend_env_paths(env_var: &str, new_paths: Vec<PathBuf>) -> Option<OsString> {
    let mut paths = new_paths;
    if let Some(existing) = std::env::var_os(env_var) {
        paths.extend(std::env::split_paths(&existing));
    }
    join_env_paths(paths)
}

/// Splits an env path list using the platform-native separator.
pub fn split_env_paths(paths: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(paths).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_dylib_filenames_are_target_aware() {
        assert_eq!(
            dylib_filename("rustc_codegen_cuda", "x86_64-pc-windows-msvc"),
            "rustc_codegen_cuda.dll"
        );
        assert_eq!(
            dylib_filename("rustc_codegen_cuda", "aarch64-apple-darwin"),
            "librustc_codegen_cuda.dylib"
        );
        assert_eq!(
            dylib_filename("rustc_codegen_cuda", "x86_64-unknown-linux-gnu"),
            "librustc_codegen_cuda.so"
        );
    }

    #[test]
    fn platform_object_extensions_are_target_aware() {
        assert_eq!(object_extension("x86_64-pc-windows-msvc"), "obj");
        assert_eq!(object_extension("x86_64-unknown-linux-gnu"), "o");
    }

    #[test]
    fn platform_executable_filenames_are_target_aware() {
        assert_eq!(
            executable_filename("llc", "x86_64-pc-windows-msvc"),
            "llc.exe"
        );
        assert_eq!(
            executable_filename("llc", "x86_64-unknown-linux-gnu"),
            "llc"
        );
    }

    #[test]
    fn platform_loader_env_vars_are_target_aware() {
        assert_eq!(loader_env_var("x86_64-pc-windows-msvc"), "PATH");
        assert_eq!(loader_env_var("aarch64-apple-darwin"), "DYLD_LIBRARY_PATH");
        assert_eq!(
            loader_env_var("x86_64-unknown-linux-gnu"),
            "LD_LIBRARY_PATH"
        );
    }

    #[test]
    fn platform_path_helpers_round_trip_with_native_separators() {
        let first = PathBuf::from("first");
        let second = PathBuf::from("second");
        let joined = join_env_paths(vec![first.clone(), second.clone()]).unwrap();
        assert_eq!(split_env_paths(&joined), vec![first, second]);
    }
}
