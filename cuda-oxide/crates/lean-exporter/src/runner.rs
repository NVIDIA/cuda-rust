/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runs `lake env lean` on the generated `.lean` file and checks that no
//! shortcuts (`sorry` or custom axioms) were used.
//!
//! ```text
//!   Generated `.lean` string
//!     │
//!     ▼  writes temp file & runs `lake env lean` in `cuda-oxide/oxide-proofs-preview/`
//!   Lean 4 Type-Checker + Kernel
//!     ├── Exit code != 0?                 ──► Reject build (proof failed!)
//!     ├── Output mentions `sorryAx`?      ──► Reject build (unproved theorem!)
//!     └── Axiom other than `propext` /    ──► Reject build (unexpected axiom!)
//!         `Quot.sound`?
//! ```

use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use dialect_mir::ops::MirFuncOp;
use pliron::{
    builtin::op_interfaces::SymbolOpInterface,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    printable::Printable,
};
use sha2::{Digest, Sha256};

use crate::{
    ExportError,
    codegen::{proved_obligation_descriptions, render_lean_module},
    reject,
    spec::has_verification_attributes,
    translate::{TranslatedFunction, translate_function},
};

/// Environment variable naming the output file where `cargo oxide lean-export`
/// captures `=== MIR input ===` and `=== Generated Lean ===` at `PostMem2Reg`.
pub const LEAN_EXPORT_FILE_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_FILE";

/// Optional environment variable filtering `cargo oxide lean-export` capture
/// to a single `#[verify(...)]` kernel symbol name.
pub const LEAN_EXPORT_FUNCTION_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_FUNCTION";

/// Sidecar file recording all `#[verify(...)]` kernel names seen during an
/// export run, used to report available symbols when `--function` misses.
pub const LEAN_EXPORT_DISCOVERED_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_DISCOVERED_FILE";

/// Environment flag set by `cargo oxide lean-export` so `PostMem2Reg` captures
/// the MIR and generated Lean module without invoking `lake env lean` or
/// erasing `mir.assert` bounds checks.
pub const LEAN_EXPORT_ONLY_ENV: &str = "CUDA_OXIDE_LEAN_EXPORT_ONLY";

/// Environment flag disabling the content-addressed Lean proof cache (used by
/// `check-exporter.py`, integration tests, and CI proof jobs so every test run
/// invokes the real Lean 4 kernel).
pub const NO_LEAN_CACHE_ENV: &str = "CUDA_OXIDE_NO_LEAN_CACHE";

/// Recursively collect sorted `(relative_path, full_path)` pairs matching `extension` under `root`.
fn collect_files_sorted(root: &Path, extension: &str) -> Vec<(String, PathBuf)> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some(extension) {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned();
                files.push((rel, path));
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

/// Compute a content-addressed SHA-256 cache key over the Lean toolchain pin,
/// every compiled `.olean` (or `.lean` source when `.olean`s are absent) in
/// `lean_package_dir`, and the exact generated `lean_source`.
fn proof_cache_key(lean_source: &str, lean_package_dir: &Path) -> Option<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"cuda-oxide-lean-proof-cache-v1\0");
    let toolchain = fs::read(lean_package_dir.join("lean-toolchain")).ok()?;
    hasher.update(&toolchain);
    hasher.update(b"\0");

    let olean_dir = lean_package_dir.join(".lake/build/lib/lean");
    let library_files = if olean_dir.join("CudaOxide.olean").is_file() {
        collect_files_sorted(&olean_dir, "olean")
    } else {
        collect_files_sorted(&lean_package_dir.join("CudaOxide"), "lean")
    };
    if library_files.is_empty() {
        return None;
    }
    for (rel, path) in library_files {
        hasher.update(rel.as_bytes());
        hasher.update(b"\0");
        let bytes = fs::read(path).ok()?;
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    hasher.update(b"\0source\0");
    hasher.update((lean_source.len() as u64).to_le_bytes());
    hasher.update(lean_source.as_bytes());

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    Some(hex)
}

/// Directory holding verified Lean proof cache markers (`~/.cargo/cuda-oxide/lean-cache`).
fn proof_cache_dir() -> PathBuf {
    if let Some(dir) = env::var_os("CUDA_OXIDE_LEAN_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = env::var_os("HOME") {
        let candidate = PathBuf::from(home).join(".cargo/cuda-oxide/lean-cache");
        if fs::create_dir_all(&candidate).is_ok() {
            return candidate;
        }
    }
    let fallback = env::temp_dir().join("cuda-oxide-lean-cache");
    let _ = fs::create_dir_all(&fallback);
    fallback
}

/// Locate the `cuda-oxide/oxide-proofs-preview` package directory relative to this crate.
pub fn default_lean_package_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("cuda-oxide workspace root")
        .join("oxide-proofs-preview")
}

/// Locate the `lake` build tool on `PATH` or inside `~/.elan/bin/lake`.
/// Returns a fail-closed [`ExportError::Rejected`] error if Lean is not installed.
fn find_lake_executable() -> Result<PathBuf, ExportError> {
    if let Some(paths) = env::var_os("PATH") {
        for dir in env::split_paths(&paths) {
            let candidate = dir.join("lake");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let home = env::var_os("ELAN_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".elan")))
        .ok_or_else(|| {
            reject(
                "lean verifier",
                "could not locate `lake` on PATH or in ~/.elan/bin",
            )
        })?;
    let candidate = home.join("bin").join("lake");
    if candidate.is_file() {
        Ok(candidate)
    } else {
        Err(reject(
            "lean verifier",
            format!(
                "could not locate `lake` on PATH or at {}",
                candidate.display()
            ),
        ))
    }
}

/// Return `true` when `lake` is installed and the pinned toolchain in `lean_package_dir` resolves.
pub fn is_lean_available(lean_package_dir: &Path) -> bool {
    let Ok(lake) = find_lake_executable() else {
        return false;
    };
    Command::new(lake)
        .arg("--version")
        .current_dir(lean_package_dir)
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Run Lean 4 on `lean_source` (using direct `lean` + `LEAN_PATH` when `.olean` artifacts
/// are already built, falling back to `lake env lean`) and audit proof axioms.
pub fn check_generated_lean(lean_source: &str, lean_package_dir: &Path) -> Result<(), ExportError> {
    let cache_dir = if env::var_os(NO_LEAN_CACHE_ENV).is_some_and(|v| v == "1") {
        None
    } else {
        Some(proof_cache_dir())
    };
    check_generated_lean_in_cache(lean_source, lean_package_dir, cache_dir.as_deref())
}

/// Same as [`check_generated_lean`] with an explicit optional `cache_dir` (pass `None` to bypass caching).
pub fn check_generated_lean_in_cache(
    lean_source: &str,
    lean_package_dir: &Path,
    cache_dir: Option<&Path>,
) -> Result<(), ExportError> {
    let lake = find_lake_executable()?;
    let lean_bin = lake.with_file_name("lean");
    let olean_dir = lean_package_dir.join(".lake/build/lib/lean");
    let use_direct_lean = lean_bin.is_file() && olean_dir.join("CudaOxide.olean").is_file();

    if let Ok(dump_path) = env::var("CUDA_OXIDE_DUMP_LEAN") {
        let _ = fs::write(dump_path, lean_source);
    }

    let cache_marker = cache_dir.and_then(|dir| {
        let _ = fs::create_dir_all(dir);
        proof_cache_key(lean_source, lean_package_dir).map(|key| dir.join(format!("{key}.ok")))
    });
    if let Some(ref marker) = cache_marker
        && marker.is_file()
    {
        return Ok(());
    }

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = env::temp_dir().join(format!(
        "cuda_oxide_verified_{}_{unique}.lean",
        std::process::id()
    ));
    fs::write(&temp_path, lean_source).map_err(|e| {
        reject(
            "lean verifier",
            format!("cannot write temporary Lean file: {e}"),
        )
    })?;

    let output = if use_direct_lean {
        Command::new(&lean_bin)
            .current_dir(lean_package_dir)
            .env("LEAN_PATH", &olean_dir)
            .arg(&temp_path)
            .output()
    } else {
        Command::new(&lake)
            .current_dir(lean_package_dir)
            .args(["env", "lean"])
            .arg(&temp_path)
            .output()
    };
    let _ = fs::remove_file(&temp_path);

    let output = output.map_err(|e| {
        reject(
            "lean verifier",
            format!("failed to execute Lean type-checker: {e}"),
        )
    })?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        return Err(reject(
            "lean proof check failed",
            combined.trim().to_string(),
        ));
    }
    if combined.contains("sorryAx") || combined.contains("declaration uses 'sorry'") {
        return Err(reject(
            "lean proof check failed",
            format!("admitted proof (`sorry`) detected:\n{}", combined.trim()),
        ));
    }
    for line in combined.lines() {
        if let Some((_, list)) = line.split_once("depends on axioms: [")
            && let Some((inside, _)) = list.split_once(']')
        {
            for axiom in inside.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if !matches!(axiom, "propext" | "Quot.sound" | "Classical.choice") {
                    return Err(reject(
                        "lean proof check failed",
                        format!("unexpected axiom `{axiom}` in proof:\n{}", combined.trim()),
                    ));
                }
            }
        }
    }
    if let Some(marker) = cache_marker {
        let tmp = marker.with_extension(format!("tmp.{}", std::process::id()));
        if fs::write(&tmp, b"ok\n").is_ok() {
            let _ = fs::rename(&tmp, &marker);
        }
    }
    Ok(())
}

/// Verify every `mir.func` in `module` that carries `#[verify(...)]` attributes,
/// and elide `mir.assert` bounds checks proven redundant by Lean 4.
///
/// Called by `cuda-oxide-codegen` at the `PostMem2Reg` checkpoint before loop
/// unrolling and LLVM lowering.
pub fn verify_module_kernels(
    ctx: &mut Context,
    module: Ptr<Operation>,
    _verbose: bool,
) -> Result<(), ExportError> {
    let functions: Vec<Ptr<Operation>> = {
        let module_ref = module.deref(ctx);
        if module_ref.num_regions() != 1 {
            return Ok(());
        }
        let region = module_ref.get_region(0).deref(ctx);
        let Some(block) = region.get_head() else {
            return Ok(());
        };
        block
            .deref(ctx)
            .iter(ctx)
            .filter(|&op| {
                MirFuncOp::wrap(ctx, op).is_some() && has_verification_attributes(ctx, op)
            })
            .collect()
    };
    if functions.is_empty() {
        return Ok(());
    }

    let export_file = env::var_os(LEAN_EXPORT_FILE_ENV).map(PathBuf::from);
    let export_fn_filter = env::var(LEAN_EXPORT_FUNCTION_ENV)
        .ok()
        .filter(|name| !name.is_empty());
    let discovered_file = env::var_os(LEAN_EXPORT_DISCOVERED_ENV).map(PathBuf::from);
    let export_only = env::var_os(LEAN_EXPORT_ONLY_ENV).is_some_and(|val| val == "1");

    let lean_dir = default_lean_package_dir();
    let mut prepared = Vec::with_capacity(functions.len());
    let mut lean_modules = Vec::with_capacity(functions.len());
    for func in functions {
        let name = MirFuncOp::wrap(ctx, func)
            .map(|f| f.get_symbol_name(ctx).to_string())
            .unwrap_or_else(|| "<kernel>".to_string());
        if let Some(ref disc_path) = discovered_file
            && let Ok(mut file) = OpenOptions::new().create(true).append(true).open(disc_path)
        {
            let _ = writeln!(file, "{name}");
        }
        let mir_snapshot = if export_file.is_some() {
            Some(func.deref(ctx).disp(ctx).to_string())
        } else {
            None
        };
        let translated = translate_function(ctx, func)?;
        let lean_code = render_lean_module(&translated)?;
        if let (Some(out_path), Some(mir_src)) = (&export_file, mir_snapshot)
            && export_fn_filter
                .as_deref()
                .is_none_or(|wanted| wanted == name)
        {
            append_inspection_bundle(out_path, &mir_src, &lean_code)?;
        }
        lean_modules.push(lean_code);
        prepared.push((name, translated));
    }

    if export_only {
        return Ok(());
    }

    if lean_modules.len() == 1 {
        check_generated_lean(&lean_modules[0], &lean_dir)?;
    } else {
        std::thread::scope(|s| -> Result<(), ExportError> {
            let handles: Vec<_> = lean_modules
                .iter()
                .map(|src| {
                    let lean_dir_ref = &lean_dir;
                    s.spawn(move || check_generated_lean(src, lean_dir_ref))
                })
                .collect();
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| reject("lean verifier", "parallel Lean worker panicked"))??;
            }
            Ok(())
        })?;
    }

    for (name, translated) in &prepared {
        match translated {
            TranslatedFunction::MirProgram {
                spec,
                proven_assert_ops,
                ..
            } => {
                let elided_count = proven_assert_ops.len();
                for &assert_op in proven_assert_ops {
                    Operation::erase(assert_op, ctx);
                }
                let requires_list = if spec.requires_len_ge.is_empty() {
                    "none".to_string()
                } else {
                    spec.requires_len_ge
                        .iter()
                        .map(|(lhs, rhs)| format!("param[{lhs}].len() >= param[{rhs}].len()"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                eprintln!(
                    "\n[lean-verifier] Verified kernel `{name}` ({}D block = {}x{}x1):",
                    spec.domain, spec.block_size, spec.block_y
                );
                eprintln!("  requires:      {requires_list}");
                for obligation in proved_obligation_descriptions(spec) {
                    eprintln!("  proved:        {obligation}");
                }
                eprintln!(
                    "  bounds elided: {elided_count} proven `mir.assert` check(s) removed before PTX lowering"
                );
            }
            TranslatedFunction::Scalar { .. } => {}
        }
    }
    Ok(())
}

/// Append one `=== MIR input ===` + `=== Generated Lean ===` inspection block
/// to `out_path`, inserting a blank separator line when earlier kernels were
/// already written to the same capture file.
fn append_inspection_bundle(
    out_path: &Path,
    mir_source: &str,
    lean_source: &str,
) -> Result<(), ExportError> {
    let needs_separator = out_path.metadata().is_ok_and(|meta| meta.len() > 0);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .map_err(|e| {
            reject(
                "lean-export",
                format!("cannot open export capture {}: {e}", out_path.display()),
            )
        })?;
    let mir_trimmed = mir_source.trim_end();
    if needs_separator {
        writeln!(file).map_err(|e| reject("lean-export", format!("cannot write export: {e}")))?;
    }
    write!(
        file,
        "=== MIR input ===\n{mir_trimmed}\n\n=== Generated Lean ===\n{lean_source}"
    )
    .map_err(|e| reject("lean-export", format!("cannot write export: {e}")))
}
