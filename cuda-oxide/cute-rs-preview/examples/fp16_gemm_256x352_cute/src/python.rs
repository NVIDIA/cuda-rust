/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Select caller-managed Python or prepare the example's local environment.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// Honor explicit interpreters; otherwise finish setup before any GPU timing.
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

#[cfg(test)]
mod tests {
    use super::*;

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
