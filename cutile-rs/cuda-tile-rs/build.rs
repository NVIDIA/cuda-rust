/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builds the bundled cuda-tile submodule via cmake (which downloads and
//! builds LLVM/MLIR), generates Rust FFI bindings for the MLIR and cuda-tile
//! C-APIs via bindgen, and links Rust consumers against the static LLVM/MLIR
//! archives produced by that build.
//!
//! # First-build cost
//!
//! The first build clones a pinned LLVM commit and builds LLVM + MLIR + the
//! cuda-tile static libs. Tens of minutes to an hour, several GB of disk,
//! network required.
//!
//! # Caching (opt-in)
//!
//! See the crate README. `CUDA_TILE_RS_CACHE=1` to enable, default location
//! `$XDG_CACHE_HOME/cuda-tile-rs` or `$HOME/.cache/cuda-tile-rs`.
//!
//! # Other env vars
//!
//! - `CUDA_TILE_TRANSLATE_INSTALL_DIR`: if set (existing dir), the built
//!   `cuda-tile-translate` is copied there.

use cmake::Config;
use std::path::{Path, PathBuf};
use std::{env, fs};

const SUBMODULE_PATH: &str = "cuda-tile";
const TRANSLATE_BIN: &str = "cuda-tile-translate";
const INSTALL_DIR_ENV: &str = "CUDA_TILE_TRANSLATE_INSTALL_DIR";
const CACHE_ENABLE_ENV: &str = "CUDA_TILE_RS_CACHE";
const CACHE_DIR_ENV: &str = "CUDA_TILE_RS_CACHE_DIR";
const CLEAR_CACHE_ENV: &str = "CUDA_TILE_RS_CLEAR_CACHE";
const SHA_LEN: usize = 12;

const CUDA_TILE_STATIC_LIBS: &[&str] = &[
    "CudaTileBytecodeCommon",
    "CudaTileBytecodeReader",
    "CudaTileBytecodeTranslation",
    "CudaTileBytecodeWriter",
    "CudaTileCAPIDialects",
    "CudaTileCAPIRegistration",
    "CudaTileDialect",
    "CudaTileTransforms",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    for var in [
        INSTALL_DIR_ENV,
        CACHE_ENABLE_ENV,
        CACHE_DIR_ENV,
        CLEAR_CACHE_ENV,
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let manifest_dir = fs::canonicalize(PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    ))
    .expect("canonicalize CARGO_MANIFEST_DIR");

    let submodule = manifest_dir.join(SUBMODULE_PATH);
    assert!(
        submodule.join("CMakeLists.txt").exists(),
        "cuda-tile submodule missing at {}. Run `git submodule update --init --recursive`.",
        submodule.display()
    );
    println!(
        "cargo:rerun-if-changed={}/CMakeLists.txt",
        submodule.display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // 1. Resolve build+install tree location (default: $OUT_DIR; opt-in cache
    //    under ~/.cache/cuda-tile-rs/<key>/).
    let build_root = resolve_build_root(&submodule, &out_dir);
    fs::create_dir_all(&build_root).expect("failed to create build root");

    // 2. Build cuda-tile + LLVM via cmake. We don't ask for the combined
    //    MLIR-C dylib — linking is done by enumerating static libs below.
    let install_prefix = Config::new(&submodule)
        .out_dir(&build_root)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LLVM_ENABLE_ASSERTIONS", "OFF")
        .define("CUDA_TILE_ENABLE_TESTING", "OFF")
        .cxxflag("-Wno-deprecated-declarations")
        .cxxflag("-Wno-unused-result")
        .build();

    // 3. Include-path roots.
    let cuda_tile_include = submodule.join("include");
    let cuda_tile_install_include = install_prefix.join("include");
    let llvm_src = build_root.join("build").join("llvm-project");
    let llvm_build = build_root.join("build").join("llvm-project-build");
    let mlir_src_include = llvm_src.join("mlir").join("include");
    let llvm_src_include = llvm_src.join("llvm").join("include");
    let llvm_build_include = llvm_build.join("include");
    let mlir_build_include = llvm_build.join("tools").join("mlir").join("include");

    for p in [
        &cuda_tile_include,
        &mlir_src_include,
        &llvm_src_include,
        &llvm_build_include,
        &mlir_build_include,
    ] {
        assert!(p.exists(), "expected include path missing: {}", p.display());
    }

    // 4. Generate bindings. Layout tests off (MLIR macro-generated opaque
    //    structs occasionally trip bindgen's size assertions).
    let bindings = bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", cuda_tile_include.display()))
        .clang_arg(format!("-I{}", cuda_tile_install_include.display()))
        .clang_arg(format!("-I{}", mlir_src_include.display()))
        .clang_arg(format!("-I{}", llvm_src_include.display()))
        .clang_arg(format!("-I{}", mlir_build_include.display()))
        .clang_arg(format!("-I{}", llvm_build_include.display()))
        .allowlist_file(".*/mlir-c/.*")
        .allowlist_file(".*/cuda_tile-c/.*")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate MLIR + cuda-tile bindings");
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // 5. Link directives.
    //
    // Strategy: glob the MLIR and LLVM static libs cuda-tile's build
    // produced, wrap them in linker `--start-group`/`--end-group` so the
    // linker resolves cross-references without hand-ordering.
    let lib_dir = install_prefix.join("lib");
    let llvm_lib_dir = llvm_build.join("lib");

    // -L paths go on the linker command line directly.
    println!("cargo:rustc-link-arg=-L{}", lib_dir.to_string_lossy());
    println!("cargo:rustc-link-arg=-L{}", llvm_lib_dir.to_string_lossy());

    // Enumerate LLVM and MLIR static libs by globbing the build's lib dir.
    // Simpler and more robust than `llvm-config --libs all` which asks for
    // components cuda-tile's build didn't produce. Whatever was built gets
    // linked; --start-group below handles dependency ordering.
    let llvm_libs = glob_static_libs(&llvm_lib_dir, "libLLVM");
    let mlir_libs = glob_static_libs(&llvm_lib_dir, "libMLIR");
    assert!(
        !mlir_libs.is_empty(),
        "no libMLIR*.a found in {} — cuda-tile/LLVM build produced no MLIR static libs",
        llvm_lib_dir.display()
    );
    assert!(
        !llvm_libs.is_empty(),
        "no libLLVM*.a found in {} — cuda-tile/LLVM build produced no LLVM static libs",
        llvm_lib_dir.display()
    );
    // Standard system libs LLVM needs. Kept small and explicit — these are
    // widely available (glibc + librt + zlib are on every Linux dev host).
    let system_libs: Vec<String> = ["pthread", "dl", "m", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Group MLIR + LLVM + cuda-tile static libs so the linker rescans them
    // until cross-references are resolved. System libs (-lpthread, -lm, etc.)
    // stay outside the group.
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    for lib in &mlir_libs {
        println!("cargo:rustc-link-arg=-l{lib}");
    }
    for lib in &llvm_libs {
        println!("cargo:rustc-link-arg=-l{lib}");
    }
    for lib in CUDA_TILE_STATIC_LIBS {
        println!("cargo:rustc-link-arg=-l{lib}");
    }
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    for lib in &system_libs {
        println!("cargo:rustc-link-arg=-l{lib}");
    }
    println!("cargo:rustc-link-arg=-lstdc++");

    // 6. Optional install-copy of cuda-tile-translate.
    let translate_bin = locate_translate_bin(&install_prefix, &build_root).unwrap_or_else(|| {
        panic!(
            "failed to locate {TRANSLATE_BIN} under {}",
            install_prefix.display()
        )
    });
    if let Ok(dir) = env::var(INSTALL_DIR_ENV) {
        let dest_dir = PathBuf::from(dir);
        assert!(
            dest_dir.is_dir(),
            "{INSTALL_DIR_ENV} = {} is not an existing directory",
            dest_dir.display()
        );
        let dest = dest_dir.join(TRANSLATE_BIN);
        fs::copy(&translate_bin, &dest).unwrap_or_else(|e| {
            panic!(
                "failed to copy {} -> {}: {e}",
                translate_bin.display(),
                dest.display()
            )
        });
        println!(
            "cargo:warning=Installed {TRANSLATE_BIN} to {}",
            dest.display()
        );
    }
}

/// Collect static-lib basenames (`libFoo.a` → `Foo`) starting with a given
/// prefix from a lib dir.
fn glob_static_libs(lib_dir: &Path, name_prefix: &str) -> Vec<String> {
    let Ok(entries) = fs::read_dir(lib_dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with(name_prefix) && name.ends_with(".a") {
                let stem = name
                    .trim_start_matches("lib")
                    .trim_end_matches(".a")
                    .to_owned();
                Some(stem)
            } else {
                None
            }
        })
        .collect();
    out.sort();
    out
}

/// Where cmake's build+install tree lives. See top-of-file docs.
fn resolve_build_root(submodule: &Path, out_dir: &Path) -> PathBuf {
    let custom = env::var(CACHE_DIR_ENV).ok();
    let enabled = env::var(CACHE_ENABLE_ENV).ok();
    let cache_root = match (custom, enabled) {
        (Some(dir), _) => Some(PathBuf::from(dir)),
        (None, Some(_)) => default_cache_root(),
        (None, None) => None,
    };
    let Some(cache_root) = cache_root else {
        return out_dir.to_path_buf();
    };

    if env::var(CLEAR_CACHE_ENV).is_ok() && cache_root.exists() {
        println!(
            "cargo:warning={CLEAR_CACHE_ENV} set — wiping {}",
            cache_root.display()
        );
        fs::remove_dir_all(&cache_root).expect("failed to wipe cache root");
    }
    let key = format!(
        "{}-{}",
        cuda_tile_commit_sha(submodule),
        llvm_commit_sha(submodule)
    );
    cache_root.join(key)
}

fn default_cache_root() -> Option<PathBuf> {
    if let Ok(xdg) = env::var("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg).join("cuda-tile-rs"));
    }
    if let Ok(home) = env::var("HOME") {
        return Some(PathBuf::from(home).join(".cache").join("cuda-tile-rs"));
    }
    None
}

fn cuda_tile_commit_sha(submodule: &Path) -> String {
    match std::process::Command::new("git")
        .arg("-C")
        .arg(submodule)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
    {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            truncate_sha(&s)
        }
        _ => "unknown-submodule".to_string(),
    }
}

fn llvm_commit_sha(submodule: &Path) -> String {
    let path = submodule.join("cmake").join("IncludeLLVM.cmake");
    let content = fs::read_to_string(&path).unwrap_or_default();
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("set(LLVM_BUILD_COMMIT_HASH ") {
            let sha = rest.trim_end_matches(')').trim();
            return truncate_sha(sha);
        }
    }
    "unknown-llvm".to_string()
}

fn truncate_sha(s: &str) -> String {
    s.chars().take(SHA_LEN).collect()
}

fn locate_translate_bin(install_prefix: &Path, build_root: &Path) -> Option<PathBuf> {
    let candidates = [
        install_prefix.join("bin").join(TRANSLATE_BIN),
        build_root.join("build").join("bin").join(TRANSLATE_BIN),
        build_root
            .join("build")
            .join("tools")
            .join("cuda-tile-translate")
            .join(TRANSLATE_BIN),
    ];
    candidates.into_iter().find(|p| p.is_file())
}
