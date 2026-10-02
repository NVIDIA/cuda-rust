/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend;

use super::*;

// =============================================================================
// Fmt command
// =============================================================================

/// Format (or check formatting of) every scope the `fmt` CI gate checks.
///
/// `.github/workflows/fmt.yml` checks five: the stable shared-host workspace,
/// the SIMT workspace, the codegen backend crate, the cuda-macros device-only
/// fixture, and every `Cargo.toml` under `examples/`, nested ones included.
/// This mirrors that set on purpose -- the reason CONTRIBUTING tells
/// contributors to prefer this command over a bare `cargo fmt` is so the gate
/// cannot fail on code they had no way to format, which only holds while the
/// two cover the same ground.
///
/// In `check` mode, reports which files need formatting without modifying them.
pub fn format_all(ctx: &Context, check: bool) {
    let mode = if check { "Checking" } else { "Formatting" };
    let mut failed = false;
    let oxide_root = ctx.oxide_workspace_root();

    // In the merged repository, shared host crates live in the stable parent
    // workspace. Standalone cuda-oxide checkouts have no such parent manifest.
    if let Some(shared_root) = oxide_root
        .parent()
        .filter(|root| root.join("Cargo.toml").is_file() && root.join("cuda-core").is_dir())
    {
        println!("📦 {} shared host workspace...", mode);
        if !run_cargo_fmt(shared_root, check) {
            failed = true;
        }
    }

    println!("📦 {} CUDA Oxide workspace...", mode);
    if !run_cargo_fmt(oxide_root, check) {
        failed = true;
    }

    println!("📦 {} rustc-codegen-cuda...", mode);
    if !run_cargo_fmt(&ctx.codegen_crate, check) {
        failed = true;
    }

    // Its own `[workspace]`, so neither run above reaches it and the examples
    // walk below never sees it either. The gate carries a dedicated step for
    // exactly this reason.
    let fixture = backend::simt_crate_path(&ctx.workspace_root, "cuda-macros")
        .join("tests")
        .join("device-only");
    if fixture.join("Cargo.toml").is_file() {
        println!("📦 {} cuda-macros device-only fixture...", mode);
        if !run_cargo_fmt(&fixture, check) {
            failed = true;
        }
    }

    // One `--manifest-path` run per manifest found, rather than `cargo fmt
    // --all` once per top-level example directory. Both reasons match the gate:
    //
    //   * `--all` stops at a nested `[workspace]` boundary, and two examples
    //     declare one (`cutile_inter_kernel/simt`,
    //     `interop_cubin_identity/device`), so neither was ever formatted here.
    //   * `--all` also formats an example's path dependencies, which means
    //     re-formatting the large shared workspaces once per example.
    let mut manifests = Vec::new();
    for root in example_roots(ctx) {
        collect_example_manifests(&root, &mut manifests);
    }
    manifests.sort();

    for manifest in &manifests {
        let label = manifest
            .parent()
            .and_then(|dir| dir.strip_prefix(oxide_root).ok())
            .unwrap_or(Path::new("."))
            .display()
            .to_string();
        println!("📦 {} example: {}...", mode, label);
        if !run_cargo_fmt_manifest(manifest, check) {
            failed = true;
        }
    }

    if failed {
        if check {
            eprintln!();
            eprintln!("❌ Some files need formatting. Run: cargo oxide fmt");
        } else {
            eprintln!();
            eprintln!("⚠️  Some formatting commands failed (see above)");
        }
        std::process::exit(1);
    } else {
        println!();
        if check {
            println!("✅ All files are properly formatted");
        } else {
            println!("✅ All crates formatted");
        }
    }
}

/// Run `cargo fmt --all` in a single directory. Returns `true` on success.
fn run_cargo_fmt(dir: &Path, check: bool) -> bool {
    let mut cmd = Command::new("cargo");
    cmd.arg("fmt").arg("--all").current_dir(dir);

    if check {
        cmd.arg("--check");
    }

    run_fmt_command(cmd)
}

/// Run `cargo fmt` from beside one manifest. Returns `true` on success.
///
/// No `--all`: the caller walks every manifest, so a workspace member is
/// visited through its own manifest rather than through its parent's.
fn run_cargo_fmt_manifest(manifest: &Path, check: bool) -> bool {
    let mut cmd = Command::new("cargo");
    cmd.arg("fmt");
    if let Some(dir) = manifest.parent() {
        // rustup selects from the working directory, not --manifest-path.
        // This is required for examples using nightly-only Cargo features.
        cmd.current_dir(dir);
    }

    if check {
        cmd.arg("--check");
    }

    run_fmt_command(cmd)
}

fn run_fmt_command(mut cmd: Command) -> bool {
    match cmd.status() {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("  Failed to run cargo fmt: {}", e);
            false
        }
    }
}

/// Collect every `Cargo.toml` under `dir`, recursively.
///
/// Mirrors the gate's `examples/**/Cargo.toml` glob, with one difference that
/// only matters off CI: `target` directories are skipped. A fresh checkout has
/// none, but a working tree does, and a packaged or vendored manifest under one
/// is not a crate this repository formats.
pub(super) fn collect_example_manifests(dir: &Path, out: &mut Vec<PathBuf>) {
    let manifest = dir.join("Cargo.toml");
    if manifest.is_file() {
        out.push(manifest);
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "target" || name.starts_with('.') {
            continue;
        }
        collect_example_manifests(&path, out);
    }
}
