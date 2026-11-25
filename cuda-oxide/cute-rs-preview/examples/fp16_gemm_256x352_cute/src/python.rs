/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Select caller-managed Python or a prepared virtualenv, or set up the shared one.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// Honor explicit interpreters and prepared virtualenvs; otherwise finish setup before any GPU timing.
pub(super) fn select(
    directory: &Path,
    explicit: Option<&Path>,
    environment: Option<OsString>,
) -> super::Result<PathBuf> {
    if let Some(python) = explicit {
        return Ok(python.to_owned());
    }
    if let Some(python) = environment {
        return Ok(python.into());
    }
    if let Some(python) = prepared(directory) {
        return Ok(python);
    }
    let setup = directory.join("setup_reference.py");
    if !setup.is_file() {
        return Err(format!("missing comparison setup helper: {}", setup.display()).into());
    }
    let output = Command::new("python3")
        .arg(setup)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("could not start comparison setup with python3: {error}"))?;
    if !output.status.success() {
        return Err(format!("comparison environment setup failed ({})", output.status).into());
    }
    let python = PathBuf::from(String::from_utf8(output.stdout)?.trim());
    if !python.is_file() {
        return Err(format!("setup returned an unavailable Python: {}", python.display()).into());
    }
    Ok(python)
}

/// The example's own `.venv`, then the shared `cute-rs-preview/.venv`.
fn prepared(directory: &Path) -> Option<PathBuf> {
    [
        directory.join(".venv/bin/python"),
        directory.join("../../.venv/bin/python"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_virtualenvs_are_used_before_setup() {
        let temp = std::env::temp_dir().join(format!("fp16-gemm-python-{}", std::process::id()));
        let example = temp.join("examples/gemm");
        let shared = temp.join(".venv/bin/python");
        let local = example.join(".venv/bin/python");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::create_dir_all(local.parent().unwrap()).unwrap();
        std::fs::write(&shared, "").unwrap();
        // The example directory has no setup helper, so only a prepared virtualenv can succeed.
        assert_eq!(
            select(&example, None, None).unwrap(),
            example.join("../../.venv/bin/python")
        );
        std::fs::write(&local, "").unwrap();
        assert_eq!(select(&example, None, None).unwrap(), local);
        std::fs::remove_dir_all(&temp).unwrap();
    }

    #[test]
    fn explicit_python_overrides_environment_without_running_setup() {
        let selected = select(
            Path::new("/no-setup-assets"),
            Some(Path::new("/custom python/bin/python")),
            Some("/other/python".into()),
        )
        .unwrap();
        assert_eq!(selected, Path::new("/custom python/bin/python"));
    }

    #[test]
    fn environment_python_bypasses_local_setup() {
        let selected = select(
            Path::new("/no-setup-assets"),
            None,
            Some("/configured/python".into()),
        )
        .unwrap();
        assert_eq!(selected, Path::new("/configured/python"));
    }
}
