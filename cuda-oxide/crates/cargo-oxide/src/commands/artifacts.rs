/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};

use super::*;

/// Touch main.rs to force recompilation (faster than cargo clean).
pub(super) fn touch_main_rs(example_dir: &Path) {
    // Force a rebuild so the codegen backend re-runs and emits a fresh
    // .ptx alongside the example. Touch every source file that might
    // host `#[kernel]` items so multi-bin layouts (kernels in `lib.rs`,
    // tests in `main.rs`, perf bench in `bin/<name>.rs`, etc.) all
    // re-codegen on every `cargo oxide run/build` invocation.
    for rel in ["src/main.rs", "src/lib.rs"] {
        touch_source_file(&example_dir.join(rel));
    }
}

pub(super) fn touch_source_file(path: &Path) {
    if path.exists()
        && let Ok(content) = std::fs::read(path)
    {
        let _ = std::fs::write(path, content);
    }
}

/// Artifacts are named after the crate, and cargo normalizes hyphens in
/// package names to underscores (`rustlantis-smoke` emits
/// `rustlantis_smoke.ptx`). Always go through this when deriving an
/// artifact filename from an example name, or hyphenated examples keep
/// stale artifacts forever.
pub(super) fn artifact_stem(example: &str) -> String {
    example.replace('-', "_")
}

/// Return the PTX artifacts generated for a regular or metadata-interop project.
pub(super) fn ptx_artifact_paths(example_dir: &Path, example: &str) -> Vec<PathBuf> {
    if let Some(interop) =
        load_interop_config(example_dir).filter(|config| !config.device_crates.is_empty())
    {
        return interop
            .device_crates
            .iter()
            .filter(|device_crate| device_crate.artifact_kind == InteropArtifactKind::Ptx)
            .map(|device_crate| {
                let manifest_path = example_dir.join(&device_crate.manifest_path);
                let artifact_name = interop_device_artifact_name(&manifest_path, device_crate);

                interop_device_artifact_path(example_dir, device_crate, &artifact_name)
            })
            .collect();
    }

    let stem = artifact_stem(example);
    vec![example_dir.join(format!("{stem}.ptx"))]
}

pub(super) fn read_ptx_artifact(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|error| format!("could not read generated PTX {}: {error}", path.display()))
}

/// Print one generated PTX artifact.
pub(super) fn print_ptx_artifact(path: &Path) -> Result<(), String> {
    let content = read_ptx_artifact(path)?;

    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());

    println!();
    println!("=========================================");
    println!("PTX ({name})");
    println!("=========================================");
    print!("{content}");

    if !content.ends_with('\n') {
        println!();
    }

    Ok(())
}

/// Path to the NVVM IR (`.ll`) the backend emits for `example`. Named after the
/// Cargo-normalized crate stem, so a hyphenated example resolves to the
/// underscore-spelled file the build actually wrote. Route `emit-ltoir` reads
/// through here rather than deriving the name from the raw example.
pub(super) fn emitted_ll_path(example_dir: &Path, example: &str) -> PathBuf {
    example_dir.join(format!("{}.ll", artifact_stem(example)))
}

/// Default LTOIR output path for `example` when no explicit `--output` is given.
/// Uses the same Cargo-normalized crate stem as [`emitted_ll_path`] so reads and
/// writes agree on hyphenated examples.
pub(super) fn default_ltoir_path(example_dir: &Path, example: &str) -> PathBuf {
    example_dir.join(format!("{}.ltoir", artifact_stem(example)))
}

pub(super) const GENERATED_ARTIFACT_SUFFIXES: &[&str] = &[
    "ptx",
    "ll",
    "opt.ll",
    "ltoir",
    "cubin",
    "cubin.tmp",
    "cubin.identity",
    "ptx.identity",
    "target",
    "options",
    "cubin.target",
];

pub(super) fn generated_artifact_paths(project_dir: &Path, package_name: &str) -> Vec<PathBuf> {
    let stem = artifact_stem(package_name);

    GENERATED_ARTIFACT_SUFFIXES
        .iter()
        .map(|suffix| project_dir.join(format!("{stem}.{suffix}")))
        .collect()
}

/// Remove stale generated artifacts (`.ptx`, `.ll`, `.ltoir`, `.cubin`) from a
/// previous run so we can verify the build produces fresh output.
pub(super) fn clean_generated_files(example_dir: &Path, example: &str) {
    for file in generated_artifact_paths(example_dir, example) {
        if file.exists() {
            let _ = std::fs::remove_file(file);
        }
    }
}

/// Human-readable label for the selected output format.
pub(super) fn format_label(emit_nvvm_ir: bool) -> &'static str {
    if emit_nvvm_ir { "NVVM IR" } else { "PTX" }
}

/// Print generated artifacts (LLVM IR or PTX) to stdout after a pipeline build.
pub(super) fn show_generated_artifacts(example_dir: &Path, example: &str) {
    let stem = artifact_stem(example);
    let ll_file = example_dir.join(format!("{}.ll", stem));
    let ptx_file = example_dir.join(format!("{}.ptx", stem));

    if ll_file.exists() {
        println!();
        println!("=========================================");
        println!("LLVM IR ({}.ll)", stem);
        println!("=========================================");
        if let Ok(content) = std::fs::read_to_string(&ll_file) {
            println!("{}", content);
        }
    }

    if ptx_file.exists() {
        println!();
        println!("=========================================");
        println!("PTX ({}.ptx)", stem);
        println!("=========================================");
        if let Ok(content) = std::fs::read_to_string(&ptx_file) {
            println!("{}", content);
        }
    }
}
