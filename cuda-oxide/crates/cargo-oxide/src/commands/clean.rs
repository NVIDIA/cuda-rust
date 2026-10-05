/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::Path;

use super::*;

// =============================================================================
// Clean command
// =============================================================================

pub fn clean(ctx: &Context) {
    match clean_context(ctx) {
        Ok(summary) if summary.removed_directories == 0 && summary.removed_files == 0 => {
            println!("Nothing to clean.");
        }
        Ok(summary) => {
            println!(
                "Removed {} directories and {} generated artifacts.",
                summary.removed_directories, summary.removed_files
            );
        }
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(super) struct CleanSummary {
    pub(super) removed_directories: usize,
    pub(super) removed_files: usize,
}

pub(super) fn clean_context(ctx: &Context) -> Result<CleanSummary, String> {
    let mut summary = CleanSummary::default();

    if ctx.is_workspace {
        clean_workspace(ctx, &mut summary)?;
    } else {
        clean_standalone_project(&ctx.workspace_root, &mut summary)?;
    }

    Ok(summary)
}

fn clean_standalone_project(project_dir: &Path, summary: &mut CleanSummary) -> Result<(), String> {
    let manifest_path = project_dir.join("Cargo.toml");
    let package_name = package_name_for_clean(&manifest_path)?;

    if remove_local_target(project_dir)? {
        summary.removed_directories += 1;
    }

    summary.removed_files += remove_generated_artifacts(project_dir, &package_name)?;

    Ok(())
}

fn clean_workspace(ctx: &Context, summary: &mut CleanSummary) -> Result<(), String> {
    if remove_local_target(&ctx.workspace_root)? {
        summary.removed_directories += 1;
    }

    if remove_local_target(&ctx.codegen_crate)? {
        summary.removed_directories += 1;
    }

    let entries = std::fs::read_dir(&ctx.examples_dir).map_err(|error| {
        format!(
            "could not read examples directory {}: {error}",
            ctx.examples_dir.display()
        )
    })?;

    let mut example_dirs = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "could not read an entry in {}: {error}",
                ctx.examples_dir.display()
            )
        })?;

        let file_type = entry.file_type().map_err(|error| {
            format!(
                "could not inspect example entry {}: {error}",
                entry.path().display()
            )
        })?;

        if !file_type.is_dir() {
            continue;
        }

        let example_dir = entry.path();
        if example_dir.join("Cargo.toml").is_file() {
            example_dirs.push(example_dir);
        }
    }

    example_dirs.sort();

    for example_dir in example_dirs {
        clean_example(&example_dir, summary)?;
    }

    Ok(())
}

fn clean_example(example_dir: &Path, summary: &mut CleanSummary) -> Result<(), String> {
    let manifest_path = example_dir.join("Cargo.toml");
    let package_name = package_name_for_clean(&manifest_path)?;

    if remove_local_target(example_dir)? {
        summary.removed_directories += 1;
    }

    summary.removed_files += remove_generated_artifacts(example_dir, &package_name)?;

    Ok(())
}

fn package_name_for_clean(manifest_path: &Path) -> Result<String, String> {
    let source = std::fs::read_to_string(manifest_path).map_err(|error| {
        format!(
            "could not read manifest {}: {error}",
            manifest_path.display()
        )
    })?;

    let document: toml::Value = toml::from_str(&source).map_err(|error| {
        format!(
            "could not parse manifest {}: {error}",
            manifest_path.display()
        )
    })?;

    document
        .get("package")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "manifest {} is missing package.name",
                manifest_path.display()
            )
        })
}

fn remove_local_target(project_dir: &Path) -> Result<bool, String> {
    let target_dir = project_dir.join("target");

    let metadata = match std::fs::symlink_metadata(&target_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(false);
        }
        Err(error) => {
            return Err(format!(
                "could not inspect {}: {error}",
                target_dir.display()
            ));
        }
    };

    if metadata.file_type().is_symlink() {
        return Err(format!(
            "refusing to remove symlinked target directory {}",
            target_dir.display()
        ));
    }

    if !metadata.is_dir() {
        return Err(format!(
            "expected {} to be a directory",
            target_dir.display()
        ));
    }

    std::fs::remove_dir_all(&target_dir).map_err(|error| {
        format!(
            "could not remove target directory {}: {error}",
            target_dir.display()
        )
    })?;

    println!("Removed {}", target_dir.display());

    Ok(true)
}

fn remove_generated_artifacts(project_dir: &Path, package_name: &str) -> Result<usize, String> {
    let mut removed = 0;

    for path in generated_artifact_paths(project_dir, package_name) {
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                continue;
            }
            Err(error) => {
                return Err(format!("could not inspect {}: {error}", path.display()));
            }
        };

        if metadata.file_type().is_symlink() {
            return Err(format!(
                "refusing to remove symlinked generated artifact {}",
                path.display()
            ));
        }

        if !metadata.is_file() {
            return Err(format!(
                "expected generated artifact {} to be a file",
                path.display()
            ));
        }

        std::fs::remove_file(&path)
            .map_err(|error| format!("could not remove {}: {error}", path.display()))?;

        println!("Removed {}", path.display());
        removed += 1;
    }

    Ok(removed)
}
