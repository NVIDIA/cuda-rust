/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

const FORMAT_VERSION: &str = "cuda-oxide-artifact-identity-v1";

pub(crate) fn write(
    artifact_path: &Path,
    depfile_path: &Path,
    manifest_path: &Path,
    base_dir: &Path,
    cargo_target_dir: &Path,
    target: &str,
    device_features: Option<&str>,
) -> io::Result<PathBuf> {
    let base_dir = base_dir.canonicalize()?;
    // Build outputs (OUT_DIR-generated sources) are excluded by the actual
    // cargo target directory, so a source tree that happens to contain a
    // directory named `target` (e.g. `src/target/mod.rs`) is still digested.
    let cargo_target_dir = cargo_target_dir
        .canonicalize()
        .unwrap_or_else(|_| cargo_target_dir.to_path_buf());
    let dependency_base = manifest_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "device manifest has no parent directory",
        )
    })?;
    let mut sources = parse_depfile(&std::fs::read_to_string(depfile_path)?)?;
    sources.push(manifest_path.to_path_buf());
    if let Some(device_dir) = manifest_path.parent() {
        let lockfile = device_dir.join("Cargo.lock");
        if lockfile.is_file() {
            sources.push(lockfile);
        }
    }

    let mut source_hashes = BTreeMap::new();
    for source in sources {
        let source = if source.is_absolute() {
            source
        } else {
            dependency_base.join(source)
        };
        let source = source.canonicalize()?;
        if !is_identity_source(&source, &cargo_target_dir) {
            continue;
        }
        let relative = relative_path(&base_dir, &source);
        if relative.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "artifact identity source cannot be represented relative to the artifact directory",
            ));
        }
        let relative = path_text(&relative)?;
        if relative.contains(['\n', '\r', '\t']) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "artifact identity source path contains a control separator",
            ));
        }
        source_hashes.insert(relative, sha256(&std::fs::read(source)?));
    }
    if source_hashes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "artifact dependency file contained no Rust sources",
        ));
    }

    let mut source_identity = Sha256::new();
    for (path, digest) in &source_hashes {
        source_identity.update(path.as_bytes());
        source_identity.update([0]);
        source_identity.update(digest.as_bytes());
        source_identity.update([0]);
    }

    let artifact_hash = sha256(&std::fs::read(artifact_path)?);
    let source_identity = hex(source_identity.finalize().as_slice());
    let device_features = normalize_features(device_features)?;
    let device_features = if device_features.is_empty() {
        "<none>".to_owned()
    } else {
        device_features.join(",")
    };
    let mut document = String::new();
    document.push_str(FORMAT_VERSION);
    document.push('\n');
    document.push_str("target\t");
    document.push_str(target);
    document.push('\n');
    document.push_str("device_features\t");
    document.push_str(&device_features);
    document.push('\n');
    document.push_str("artifact_sha256\t");
    document.push_str(&artifact_hash);
    document.push('\n');
    document.push_str("sources_sha256\t");
    document.push_str(&source_identity);
    document.push('\n');
    for (path, digest) in source_hashes {
        document.push_str("source\t");
        document.push_str(&path);
        document.push('\t');
        document.push_str(&digest);
        document.push('\n');
    }

    let identity_path = appended_path(artifact_path, ".identity");
    let temporary_path = appended_path(&identity_path, ".tmp");
    std::fs::write(&temporary_path, document)?;
    std::fs::rename(&temporary_path, &identity_path)?;
    Ok(identity_path)
}

fn normalize_features(features: Option<&str>) -> io::Result<Vec<&str>> {
    let Some(features) = features else {
        return Ok(Vec::new());
    };
    let mut normalized = features.split(',').map(str::trim).collect::<Vec<_>>();
    if normalized.iter().any(|feature| feature.is_empty()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "device feature list contains an empty feature",
        ));
    }
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

fn parse_depfile(contents: &str) -> io::Result<Vec<PathBuf>> {
    let Some((_, dependencies)) = contents.split_once(':') else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "artifact dependency file has no target separator",
        ));
    };
    let dependencies = dependencies.replace("\\\n", "");
    let mut paths = Vec::new();
    let mut current = String::new();
    let mut escaped = false;
    for character in dependencies.chars() {
        if escaped {
            current.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character.is_whitespace() {
            if !current.is_empty() {
                paths.push(PathBuf::from(std::mem::take(&mut current)));
            }
        } else {
            current.push(character);
        }
    }
    if escaped {
        current.push('\\');
    }
    if !current.is_empty() {
        paths.push(PathBuf::from(current));
    }
    Ok(paths)
}

fn is_identity_source(path: &Path, cargo_target_dir: &Path) -> bool {
    // Exclude build outputs by the real cargo target directory, not by any
    // path component named "target": a crate may legitimately keep sources
    // under e.g. `src/target/mod.rs`, and dropping those silently
    // under-reported identity completeness.
    if path.starts_with(cargo_target_dir) {
        return false;
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::Normal(name)
                if name == ".pixi" || name == ".git"
        )
    }) {
        return false;
    }
    path.extension().is_some_and(|extension| extension == "rs")
        || path
            .file_name()
            .is_some_and(|name| name == "Cargo.toml" || name == "Cargo.lock")
}

fn relative_path(base: &Path, target: &Path) -> PathBuf {
    let base = base.components().collect::<Vec<_>>();
    let target = target.components().collect::<Vec<_>>();
    let common = base
        .iter()
        .zip(&target)
        .take_while(|(left, right)| left == right)
        .count();
    if common == 0 {
        return target.iter().collect();
    }
    let mut relative = PathBuf::new();
    for _ in common..base.len() {
        relative.push("..");
    }
    for component in &target[common..] {
        relative.push(component.as_os_str());
    }
    relative
}

fn path_text(path: &Path) -> io::Result<String> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "artifact identity source path is not valid UTF-8",
        )
    })
}

fn sha256(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes).as_slice())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn appended_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = OsString::from(path.as_os_str());
    value.push(suffix);
    PathBuf::from(value)
}

#[cfg(test)]
mod tests {
    use super::{
        FORMAT_VERSION, is_identity_source, normalize_features, parse_depfile, relative_path,
    };
    use std::path::{Path, PathBuf};

    #[test]
    fn artifact_identity_format_is_versioned() {
        assert_eq!(FORMAT_VERSION, "cuda-oxide-artifact-identity-v1");
    }

    #[test]
    fn parses_escaped_dependency_paths() -> Result<(), Box<dyn std::error::Error>> {
        let paths =
            parse_depfile("target: src/lib.rs ../catalog/a.rs path\\ with\\ spaces/b.rs\n")?;
        assert_eq!(
            paths,
            vec![
                PathBuf::from("src/lib.rs"),
                PathBuf::from("../catalog/a.rs"),
                PathBuf::from("path with spaces/b.rs"),
            ]
        );
        Ok(())
    }

    #[test]
    fn excludes_only_the_real_cargo_target_directory() {
        let target_dir = Path::new("/workspace/engine/device/target");
        // Build outputs (e.g. OUT_DIR-generated sources) are not identity
        // sources.
        assert!(!is_identity_source(
            Path::new("/workspace/engine/device/target/release/build/gen/out/tables.rs"),
            target_dir,
        ));
        // A source directory that happens to be named `target` is still a
        // source: only the actual cargo target directory is excluded.
        assert!(is_identity_source(
            Path::new("/workspace/engine/device/src/target/mod.rs"),
            target_dir,
        ));
        assert!(is_identity_source(
            Path::new("/workspace/engine/device/Cargo.toml"),
            target_dir,
        ));
        assert!(!is_identity_source(
            Path::new("/workspace/engine/device/src/notes.md"),
            target_dir,
        ));
    }

    #[test]
    fn produces_artifact_relative_paths() {
        assert_eq!(
            relative_path(
                Path::new("/workspace/engine/device"),
                Path::new("/workspace/engine/kernel_catalog/fc1.rs"),
            ),
            PathBuf::from("../kernel_catalog/fc1.rs")
        );
    }

    #[cfg(windows)]
    #[test]
    fn preserves_an_absolute_path_when_no_relative_form_exists() {
        assert_eq!(
            relative_path(
                Path::new(r"C:\workspace\engine\device"),
                Path::new(r"D:\catalog\fc1.rs"),
            ),
            PathBuf::from(r"D:\catalog\fc1.rs")
        );
    }

    #[test]
    fn normalizes_device_features_for_artifact_identity() -> Result<(), Box<dyn std::error::Error>>
    {
        assert_eq!(
            normalize_features(Some("tensor-cores, diagnostics,tensor-cores"))?,
            vec!["diagnostics", "tensor-cores"]
        );
        assert!(normalize_features(Some("tensor-cores,")).is_err());
        Ok(())
    }
}
