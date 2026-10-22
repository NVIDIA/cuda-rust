/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Content-bound external inputs shared by Cargo fingerprinting and finalization.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::FinalizerError;

/// Identity of an external LTO-IR file captured before Cargo decides freshness.
///
/// The absolute path is also the linker's diagnostic input name. Preserve the
/// order of these records when fingerprinting and linking a list of inputs.
/// [`Self::read_verified`] returns owned bytes so a later file change cannot
/// alter the data submitted to the linker.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LtoirFile {
    path: PathBuf,
    sha256: [u8; 32],
}

impl LtoirFile {
    /// Capture the canonical path and SHA-256 of a nonempty input file.
    ///
    /// Input paths must be Unicode so the same name can reach nvJitLink and
    /// the compiler process without lossy conversion. CUDA validates the
    /// LTO-IR format and target compatibility during finalization.
    pub fn capture(path: &Path) -> Result<Self, FinalizerError> {
        let path = path.canonicalize().map_err(|source| FinalizerError::Io {
            path: path.to_owned(),
            source,
        })?;
        if path.to_str().is_none() {
            return Err(FinalizerError::InvalidLtoirPath { path });
        }
        let bytes = read(&path)?;
        Ok(Self {
            path,
            sha256: Sha256::digest(&bytes).into(),
        })
    }

    /// Diagnostic name included in the ordered compilation identity.
    pub fn name(&self) -> Result<&str, FinalizerError> {
        self.path
            .to_str()
            .ok_or_else(|| FinalizerError::InvalidLtoirPath {
                path: self.path.clone(),
            })
    }

    /// Read the exact captured content or fail if it changed before codegen.
    pub fn read_verified(&self) -> Result<Vec<u8>, FinalizerError> {
        let bytes = read(&self.path)?;
        let actual: [u8; 32] = Sha256::digest(&bytes).into();
        if actual != self.sha256 {
            return Err(FinalizerError::LtoirInputChanged {
                path: self.path.clone(),
            });
        }
        Ok(bytes)
    }
}

/// Reject a generated output that aliases an existing external link input.
///
/// Call before deleting or writing outputs. Canonical paths cover symlinks;
/// Unix file identity also covers hard links with different canonical paths.
pub fn check_ltoir_input_output(input: &Path, output: &Path) -> Result<(), FinalizerError> {
    let input_path = input.canonicalize().map_err(|source| FinalizerError::Io {
        path: input.to_owned(),
        source,
    })?;
    let output_path = match output.canonicalize() {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(FinalizerError::Io {
                path: output.to_owned(),
                source,
            });
        }
    };
    if input_path == output_path || same_file(&input_path, &output_path) {
        return Err(FinalizerError::LtoirInputOutputOverlap {
            input: input.to_owned(),
            output: output.to_owned(),
        });
    }
    Ok(())
}

#[cfg(unix)]
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_file(_a: &Path, _b: &Path) -> bool {
    false
}

fn read(path: &Path) -> Result<Vec<u8>, FinalizerError> {
    let bytes = std::fs::read(path).map_err(|source| FinalizerError::Io {
        path: path.to_owned(),
        source,
    })?;
    if bytes.is_empty() {
        return Err(FinalizerError::EmptyInput {
            name: path.display().to_string(),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_input_checks_content_and_owns_verified_bytes() {
        let path = std::env::temp_dir().join(format!(
            "oxide-ltoir-{}-{}.test",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"first").unwrap();
        let identity = LtoirFile::capture(&path).unwrap();
        let json = serde_json::to_string(&identity).unwrap();
        let restored: LtoirFile = serde_json::from_str(&json).unwrap();
        let owned = restored.read_verified().unwrap();
        std::fs::write(&path, b"other").unwrap();
        assert_eq!(owned, b"first");
        assert!(matches!(
            restored.read_verified(),
            Err(FinalizerError::LtoirInputChanged { .. })
        ));
        assert_ne!(identity, LtoirFile::capture(&path).unwrap());
        std::fs::write(&path, []).unwrap();
        assert!(matches!(
            LtoirFile::capture(&path),
            Err(FinalizerError::EmptyInput { .. })
        ));
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            restored.read_verified(),
            Err(FinalizerError::Io { .. })
        ));
    }
}
