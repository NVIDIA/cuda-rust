/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

const DEFAULT_SANITIZER_ERROR_EXITCODE: &str = "86";

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SanitizerInvocationArgs {
    pub(super) args: Vec<String>,
    pub(super) uses_default_error_exitcode: bool,
    pub(super) status_checks_weakened: bool,
}

pub(super) fn sanitizer_invocation_args(sanitizer_args: &[String]) -> SanitizerInvocationArgs {
    let has_explicit_error_exitcode = sanitizer_args
        .iter()
        .any(|arg| arg == "--error-exitcode" || arg.starts_with("--error-exitcode="));
    if has_explicit_error_exitcode {
        return SanitizerInvocationArgs {
            args: sanitizer_args.to_vec(),
            uses_default_error_exitcode: false,
            status_checks_weakened: sanitizer_option_is_no(sanitizer_args, "check-exit-code")
                || sanitizer_option_is_no(sanitizer_args, "require-cuda-init"),
        };
    }

    let mut args = Vec::with_capacity(sanitizer_args.len() + 2);
    args.extend([
        "--error-exitcode".to_string(),
        DEFAULT_SANITIZER_ERROR_EXITCODE.to_string(),
    ]);
    args.extend_from_slice(sanitizer_args);
    SanitizerInvocationArgs {
        args,
        uses_default_error_exitcode: true,
        status_checks_weakened: sanitizer_option_is_no(sanitizer_args, "check-exit-code")
            || sanitizer_option_is_no(sanitizer_args, "require-cuda-init"),
    }
}

fn sanitizer_option_is_no(args: &[String], name: &str) -> bool {
    let option = format!("--{name}");
    let equals_prefix = format!("{option}=");
    args.iter().enumerate().any(|(index, arg)| {
        arg.strip_prefix(&equals_prefix)
            .is_some_and(|value| value.eq_ignore_ascii_case("no"))
            || (arg == &option
                && args
                    .get(index + 1)
                    .is_some_and(|value| value.eq_ignore_ascii_case("no")))
    })
}

/// Fallback locations probed for `compute-sanitizer` when it is neither on
/// PATH nor under the configured CUDA toolkit root. Shared by `sanitize`
/// (`run_compute_sanitizer`) and `doctor` so both use the same discovery
/// order by construction.
pub(super) const COMPUTE_SANITIZER_FALLBACK_PATHS: &[&str] = &[
    "/usr/local/cuda/bin/compute-sanitizer",
    "/opt/cuda/bin/compute-sanitizer",
    "/usr/bin/compute-sanitizer",
];

/// Fallback locations probed for `nvcc`, in the same idiom and for the same
/// reason as the list above. `nvcc` ships in the toolkit's `bin/` beside
/// `compute-sanitizer` and `cuda-gdb`, so it has to be found the same way.
pub(super) const NVCC_FALLBACK_PATHS: &[&str] = &[
    "/usr/local/cuda/bin/nvcc",
    "/opt/cuda/bin/nvcc",
    "/usr/bin/nvcc",
];

/// Fallback locations probed for `cuda-gdb`. Shared by `debug`
/// (`codegen_debug`) and `doctor` so both use the same discovery order by
/// construction -- doctor exists to predict whether `debug` will work, so the
/// two answering differently is the bug this list prevents.
pub(super) const CUDA_GDB_FALLBACK_PATHS: &[&str] = &[
    "/usr/local/cuda/bin/cuda-gdb",
    "/opt/cuda/bin/cuda-gdb",
    "/usr/bin/cuda-gdb",
];

/// Every CUDA toolkit executable `doctor` probes, with its fallbacks.
///
/// One table so a fourth tool cannot be added with a different discovery rule.
/// Before this existed, `compute-sanitizer` resolved through the toolkit root
/// while `nvcc` and `cuda-gdb` used a bare PATH lookup, so doctor reported the
/// toolkit missing on an install where it had already found `cuda.h`, libNVVM,
/// nvJitLink and libdevice under that same root.
pub(super) const DOCTOR_TOOLKIT_TOOLS: [(&str, &[&str]); 3] = [
    ("nvcc", NVCC_FALLBACK_PATHS),
    ("cuda-gdb", CUDA_GDB_FALLBACK_PATHS),
    ("compute-sanitizer", COMPUTE_SANITIZER_FALLBACK_PATHS),
];

/// Resolve one of [`DOCTOR_TOOLKIT_TOOLS`] the way the command that uses it
/// does: PATH first, then the configured toolkit root, then the standard
/// install roots.
/// The standard install roots to try for `name`, or none if the table does not
/// know it. Split out so the pairing can be checked without a filesystem or an
/// ambient environment, neither of which a test can control here: the PATH
/// probe inside `find_cuda_toolkit_executable` shells out to `which` and has no
/// injection point.
pub(super) fn toolkit_tool_fallbacks(name: &str) -> &'static [&'static str] {
    DOCTOR_TOOLKIT_TOOLS
        .iter()
        .find(|(tool, _)| *tool == name)
        .map(|(_, paths)| *paths)
        .unwrap_or(&[])
}

pub(super) fn doctor_toolkit_tool(ctx: &Context, name: &str) -> Option<PathBuf> {
    find_cuda_toolkit_executable(ctx, name, toolkit_tool_fallbacks(name))
}

pub(super) fn run_compute_sanitizer(
    ctx: &Context,
    example_dir: &Path,
    tool: &str,
    sanitizer_args: &[String],
    application_args: &[String],
    binary: &Path,
) {
    let compute_sanitizer = find_cuda_toolkit_executable(
        ctx,
        "compute-sanitizer",
        COMPUTE_SANITIZER_FALLBACK_PATHS,
    )
    .unwrap_or_else(|| {
        eprintln!("Error: compute-sanitizer not found.");
        eprintln!(
            "It is installed with the CUDA Toolkit; run `cargo oxide doctor` to check CUDA setup."
        );
        std::process::exit(1);
    });

    let invocation_args = sanitizer_invocation_args(sanitizer_args);
    let mut cmd = Command::new(compute_sanitizer);
    cmd.args(["--tool", tool])
        .args(&invocation_args.args)
        .arg(binary)
        .args(application_args)
        .current_dir(example_dir);
    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    let forwarded_args = if invocation_args.args.is_empty() {
        String::new()
    } else {
        format!(" {}", invocation_args.args.join(" "))
    };
    let displayed_application_args = if application_args.is_empty() {
        String::new()
    } else {
        format!(" {}", application_args.join(" "))
    };
    println!(
        "Running compute-sanitizer --tool {tool}{forwarded_args} {}{displayed_application_args}...",
        binary.display()
    );
    println!();

    let status = cmd.status().expect("Failed to run compute-sanitizer");
    if !status.success() {
        eprintln!(
            "\nCompute Sanitizer failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    println!();
    println!("Compute Sanitizer completed with exit code 0.");
    if !invocation_args.uses_default_error_exitcode {
        println!(
            "An explicit --error-exitcode was supplied, so it controls whether findings fail the command."
        );
    }
    if invocation_args.status_checks_weakened {
        println!(
            "The supplied sanitizer options can allow target or CUDA-initialization failures to exit 0."
        );
    }
    println!(
        "Inspect the sanitizer report above; exit status alone is not a clean-report assertion."
    );
}

/// Locate an executable by name, first via `which` (PATH lookup), then by
/// checking a list of common fallback absolute paths.
pub(super) fn find_executable(name: &str, fallback_paths: &[&str]) -> Option<PathBuf> {
    if let Ok(output) = Command::new("which").arg(name).output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    for path in fallback_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Locate a CUDA Toolkit executable using the same configured toolkit roots as
/// `doctor`, after the user's PATH and before generic system fallbacks.
pub(super) fn find_cuda_toolkit_executable(
    ctx: &Context,
    name: &str,
    fallback_paths: &[&str],
) -> Option<PathBuf> {
    find_cuda_toolkit_executable_with_env(ctx, name, fallback_paths, |key| std::env::var(key).ok())
}

/// `find_cuda_toolkit_executable` with the ambient environment injected.
///
/// The process environment takes precedence over `cuda-oxide.toml`'s `env`, so
/// resolution has to be injectable for unit tests: a developer with a real
/// `CUDA_TOOLKIT_PATH` (or `CUDA_HOME`) exported would otherwise shadow the
/// configured root a test is trying to assert on. Same rationale as
/// `cuda_toolkit_root` and `cuda_header_candidates`.
pub(super) fn find_cuda_toolkit_executable_with_env(
    ctx: &Context,
    name: &str,
    fallback_paths: &[&str],
    mut get_env: impl FnMut(&str) -> Option<String>,
) -> Option<PathBuf> {
    if let Some(path) = find_executable(name, &[]) {
        return Some(path);
    }

    let toolkit = cuda_toolkit_root(|key| {
        get_env(key)
            .filter(|value| !value.trim().is_empty())
            .or_else(|| project_config_env(ctx, key).map(str::to_owned))
    });
    let configured = toolkit.join("bin").join(name);
    if configured.exists() {
        return Some(configured);
    }

    for path in fallback_paths {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    None
}
