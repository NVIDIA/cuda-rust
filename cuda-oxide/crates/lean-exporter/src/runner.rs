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
//!     ▼  writes temp file & runs `lake env lean` in `cuda-oxide/lean/`
//!   Lean 4 Type-Checker + Kernel
//!     ├── Exit code != 0?                 ──► Reject build (proof failed!)
//!     ├── Output mentions `sorryAx`?      ──► Reject build (unproved theorem!)
//!     └── Axiom other than `propext` /    ──► Reject build (unexpected axiom!)
//!         `Quot.sound`?
//! ```

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use dialect_mir::ops::MirFuncOp;
use pliron::{
    builtin::op_interfaces::SymbolOpInterface,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
};

use crate::{
    ExportError,
    codegen::{proved_obligation_descriptions, render_lean_module},
    reject,
    spec::has_verification_attributes,
    translate::{TranslatedFunction, translate_function},
};

/// Locate the `cuda-oxide/lean` package directory relative to this crate.
pub fn default_lean_package_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("cuda-oxide workspace root")
        .join("lean")
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

/// Run Lean 4 on `lean_source` (using direct `lean` + `LEAN_PATH` when `.olean` artifacts
/// are already built, falling back to `lake env lean`) and audit proof axioms.
pub fn check_generated_lean(lean_source: &str, lean_package_dir: &Path) -> Result<(), ExportError> {
    let lake = find_lake_executable()?;
    let lean_bin = lake.with_file_name("lean");
    let olean_dir = lean_package_dir.join(".lake/build/lib/lean");
    let use_direct_lean = lean_bin.is_file() && olean_dir.join("CudaOxide.olean").is_file();

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
    Ok(())
}

/// Combine multiple generated kernel `.lean` modules into a single `.lean` source file
/// with one shared `import CudaOxide` header so all kernels in a crate check in one run.
fn combine_lean_modules(modules: &[String]) -> String {
    if modules.len() == 1 {
        return modules[0].clone();
    }
    let mut combined = String::from(
        "import CudaOxide\n\nset_option linter.unusedSimpArgs false\nset_option maxRecDepth 4096\n\n",
    );
    for (idx, src) in modules.iter().enumerate() {
        let body = src
            .lines()
            .filter(|l| {
                !l.starts_with("--") && !l.starts_with("import ") && !l.starts_with("set_option ")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let namespaced = body
            .replace(
                "namespace CudaOxide.Generated",
                &format!("namespace CudaOxide.Generated.Kernel{idx}"),
            )
            .replace(
                "end CudaOxide.Generated",
                &format!("end CudaOxide.Generated.Kernel{idx}"),
            );
        combined.push_str(&namespaced);
        combined.push_str("\n\n");
    }
    combined
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

    let lean_dir = default_lean_package_dir();
    let mut prepared = Vec::with_capacity(functions.len());
    let mut lean_modules = Vec::with_capacity(functions.len());
    for func in functions {
        let name = MirFuncOp::wrap(ctx, func)
            .map(|f| f.get_symbol_name(ctx).to_string())
            .unwrap_or_else(|| "<kernel>".to_string());
        let translated = translate_function(ctx, func)?;
        let lean_code = render_lean_module(&translated)?;
        lean_modules.push(lean_code);
        prepared.push((name, translated));
    }

    let batched_source = combine_lean_modules(&lean_modules);
    check_generated_lean(&batched_source, &lean_dir)?;

    for (name, translated) in &prepared {
        let strip_slice: &[crate::translate::GlobalStripAccess2D] = match translated {
            TranslatedFunction::TiledKernel2D {
                global_strip_accesses,
                ..
            } => global_strip_accesses,
            _ => &[],
        };
        match translated {
            TranslatedFunction::Kernel {
                spec,
                proven_assert_ops,
                ..
            }
            | TranslatedFunction::TiledKernel2D {
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
                for obligation in proved_obligation_descriptions(spec, strip_slice) {
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
