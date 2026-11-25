/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared process launcher for the SM120a prefill and decode comparisons.

use crate::fixture::Shape;
use std::{
    error::Error,
    path::{Path, PathBuf},
    process::Command,
};

/// Host-side errors from comparison setup, paths, and process execution.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Choose explicit, environment, local-venv, shared-venv, then PATH Python.
fn resolve_python(
    directory: &Path,
    explicit: Option<&PathBuf>,
    environment: Option<PathBuf>,
) -> PathBuf {
    let local_python = directory.join(".venv/bin/python");
    explicit.cloned().or(environment).unwrap_or_else(|| {
        if local_python.is_file() {
            local_python
        } else {
            directory
                .parent()
                .and_then(Path::parent)
                .map(|cute_rs_preview| cute_rs_preview.join(".venv/bin/python"))
                .filter(|shared_python| shared_python.is_file())
                .unwrap_or_else(|| PathBuf::from("python3"))
        }
    })
}

/// Reject Python selection in standalone mode and raw output in comparison mode.
pub fn validate_mode(
    rust_only: bool,
    python: Option<&PathBuf>,
    output: Option<&PathBuf>,
) -> std::result::Result<(), String> {
    if rust_only && python.is_some() {
        return Err("--python applies only to the CuTeDSL comparison".into());
    }
    if !rust_only && output.is_some() {
        return Err("--output requires --rust-only".into());
    }
    Ok(())
}

/// Run the Python comparison with the same shape, inputs, and timing options.
#[allow(clippy::too_many_arguments)]
pub fn run(
    workload: &str,
    python: Option<&PathBuf>,
    all_cases: bool,
    shape: Shape,
    input_scale: u32,
    zero_query: bool,
    device: usize,
    warmup: usize,
    iters: usize,
    mode_flag: &str,
    mode: &str,
    json: Option<&PathBuf>,
) -> Result<()> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let prefill = directory
        .parent()
        .ok_or("example directory has no parent")?
        .join("fmha_prefill_sm120a_cute");
    let script = prefill.join("compare.py");
    let reference = directory.join("reference");
    let kernel = if workload == "decode" {
        "attention_decode.py"
    } else {
        "flash_attention_v2.py"
    };
    if !script.is_file()
        || !reference.join("launch.py").is_file()
        || !reference.join(kernel).is_file()
        || !prefill.join("reference/launch.py").is_file()
    {
        return Err(format!("CuTeDSL comparison requires local compare.py and reference/ assets in {}; use --rust-only for standalone execution", directory.display()).into());
    }
    let python = resolve_python(
        directory,
        python,
        std::env::var_os("CUDA_OXIDE_CUTE_PYTHON").map(PathBuf::from),
    );
    // Dispatch before CUDA context creation; Python invokes this binary with --rust-only.
    let mut command = Command::new(&python);
    command
        .arg(script)
        .args(["--workload", workload])
        .arg("--rust-binary")
        .arg(std::env::current_exe()?)
        .args([
            "--seq-q",
            &shape.seq_q.to_string(),
            "--seq-k",
            &shape.seq_k.to_string(),
            "--heads",
            &shape.heads.to_string(),
            "--input-scale",
            &input_scale.to_string(),
            "--device",
            &device.to_string(),
            "--warmup",
            &warmup.to_string(),
            "--iters",
            &iters.to_string(),
            mode_flag,
            mode,
        ])
        .arg(if shape.causal {
            "--causal"
        } else {
            "--noncausal"
        });
    if zero_query {
        command.arg("--zero-query");
    }
    if all_cases {
        command.arg("--all-cases");
    }
    if let Some(path) = json {
        command.arg("--json").arg(path);
    }
    let status = command.status().map_err(|error| format!(
        "could not start CuTeDSL comparison with {}: {error}; select its Python environment with --python or CUDA_OXIDE_CUTE_PYTHON", python.display()))?;
    if !status.success() {
        return Err(format!(
            "CuTeDSL comparison failed with {} ({status})",
            python.display()
        )
        .into());
    }
    Ok(())
}

/// Tests for interpreter overrides and virtualenv fallback order.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Temporary test directory removed when the fixture is dropped.
    struct TempDirectory(#[doc = "Root directory containing fake interpreter paths."] PathBuf);

    impl TempDirectory {
        /// Create a unique directory for one interpreter-selection test.
        fn new() -> Self {
            /// Process-local counter that keeps temporary test directory names distinct.
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "fmha-python-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        /// Create a placeholder interpreter file at the requested relative path.
        fn interpreter(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "").unwrap();
            path
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Check that explicit and environment Python choices override discovered virtualenvs.
    #[test]
    fn python_overrides_take_precedence_over_virtualenvs() {
        let temp = TempDirectory::new();
        let directory = temp.0.join("examples/fmha");
        temp.interpreter("examples/fmha/.venv/bin/python");
        temp.interpreter(".venv/bin/python");
        let explicit = PathBuf::from("explicit-python");
        let environment = PathBuf::from("environment-python");
        assert_eq!(
            resolve_python(&directory, Some(&explicit), Some(environment.clone())),
            explicit
        );
        assert_eq!(
            resolve_python(&directory, None, Some(environment.clone())),
            environment
        );
    }

    /// Check the local-venv, shared-venv, and PATH fallback order.
    #[test]
    fn python_falls_back_from_local_to_shared_virtualenv_then_path() {
        let temp = TempDirectory::new();
        let directory = temp.0.join("examples/fmha");
        assert_eq!(
            resolve_python(&directory, None, None),
            PathBuf::from("python3")
        );
        let shared = temp.interpreter(".venv/bin/python");
        assert_eq!(resolve_python(&directory, None, None), shared);
        let local = temp.interpreter("examples/fmha/.venv/bin/python");
        assert_eq!(resolve_python(&directory, None, None), local);
    }
}
