/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_host_cargo(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    cargo_subcommand: &str,
    features: Option<&str>,
    bin: Option<&str>,
    verbose: bool,
    app_args: &[String],
) {
    let mut cmd = Command::new("cargo");
    cmd.arg(cargo_subcommand)
        .arg("--release")
        .current_dir(example_dir);

    if cargo_subcommand == "run"
        && let Some(bin) = bin
    {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }
    if cargo_subcommand == "run" && !app_args.is_empty() {
        cmd.arg("--").args(app_args);
    }

    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    if cargo_subcommand == "run" {
        if let Some(bin) = bin {
            println!("Building and running {} (bin: {})...", example, bin);
        } else {
            println!("Building and running {}...", example);
        }
    } else {
        println!("Building host crate {}...", example);
    }
    println!();

    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }

    let status = cmd.status().expect("Failed to run host cargo command");
    if !status.success() {
        eprintln!(
            "\nHost cargo command failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn codegen_build_host_binary(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialization: &MaterializationMode,
) -> PathBuf {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_common_codegen_env(
        &mut cmd,
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );
    apply_default_sanitizer_line_tables(&mut cmd, ctx, device_debug);
    let fingerprint = sanitize_codegen_fingerprint(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        arch,
        detected_device_arch,
        None,
        materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    );
    apply_output_mode(&mut cmd, false, arch, materialization);
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch);

    if let Some(bin) = bin {
        println!("Building {} (bin: {})...", example, bin);
    } else {
        println!("Building {}...", example);
    }
    println!();

    run_cargo_build_for_executable(&mut cmd, example_dir, bin).unwrap_or_else(|message| {
        eprintln!("\nBuild failed: {message}");
        std::process::exit(1);
    })
}

pub(super) fn build_host_cargo(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    features: Option<&str>,
    bin: Option<&str>,
    verbose: bool,
) -> PathBuf {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    if let Some(bin) = bin {
        println!("Building host crate {} (bin: {})...", example, bin);
    } else {
        println!("Building host crate {}...", example);
    }
    println!();

    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }

    run_cargo_build_for_executable(&mut cmd, example_dir, bin).unwrap_or_else(|message| {
        eprintln!("\nHost cargo build failed: {message}");
        std::process::exit(1);
    })
}

pub(super) fn run_cargo_build_for_executable(
    cmd: &mut Command,
    manifest_dir: &Path,
    explicit_bin: Option<&str>,
) -> Result<PathBuf, String> {
    let selection = cargo_executable_selection(manifest_dir, explicit_bin)?;

    cmd.arg("--message-format=json-render-diagnostics");
    let output = cmd
        .output()
        .map_err(|error| format!("could not start Cargo: {error}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }

    let mut executables = Vec::<CargoExecutableArtifact>::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let message: serde_json::Value = match serde_json::from_str(line) {
            Ok(message) => message,
            Err(_) => {
                if !line.is_empty() {
                    println!("{line}");
                }
                continue;
            }
        };

        if let Some(rendered) = message
            .get("message")
            .and_then(|message| message.get("rendered"))
            .and_then(|rendered| rendered.as_str())
        {
            eprint!("{rendered}");
        }

        if message.get("reason").and_then(|reason| reason.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let is_binary = message
            .get("target")
            .and_then(|target| target.get("kind"))
            .and_then(|kind| kind.as_array())
            .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
        if !is_binary {
            continue;
        }
        let Some(path) = message.get("executable").and_then(|path| path.as_str()) else {
            continue;
        };
        let Some(package_id) = message
            .get("package_id")
            .and_then(|package_id| package_id.as_str())
        else {
            continue;
        };
        let Some(name) = message
            .get("target")
            .and_then(|target| target.get("name"))
            .and_then(|name| name.as_str())
        else {
            continue;
        };
        executables.push(CargoExecutableArtifact {
            package_id: package_id.to_string(),
            target_name: name.to_string(),
            path: PathBuf::from(path),
        });
    }

    if !output.status.success() {
        return Err(format!("Cargo exited with status {}", output.status));
    }

    select_cargo_executable_artifact(&selection, &executables)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CargoExecutableSelection {
    pub(super) packages: Vec<CargoSelectedPackage>,
    pub(super) explicit_bin: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CargoSelectedPackage {
    pub(super) package_id: String,
    pub(super) package_name: String,
    pub(super) default_run: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CargoExecutableArtifact {
    pub(super) package_id: String,
    pub(super) target_name: String,
    pub(super) path: PathBuf,
}

pub(super) fn cargo_executable_selection(
    manifest_dir: &Path,
    explicit_bin: Option<&str>,
) -> Result<CargoExecutableSelection, String> {
    let metadata = cargo_metadata(manifest_dir)?;
    let manifest_path = manifest_dir.join("Cargo.toml");
    let manifest_path = manifest_path
        .canonicalize()
        .map_err(|error| format!("could not resolve {}: {error}", manifest_path.display()))?;

    let packages = metadata
        .get("packages")
        .and_then(|packages| packages.as_array())
        .ok_or_else(|| "Cargo metadata did not include packages".to_string())?;

    let selected_packages = cargo_selected_packages(&metadata, packages, &manifest_path)?;
    let packages = selected_packages
        .into_iter()
        .map(cargo_selected_package)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CargoExecutableSelection {
        packages,
        explicit_bin: explicit_bin.map(str::to_owned),
    })
}

/// Return the packages Cargo selects for a command launched from
/// `manifest_path`.
///
/// At a workspace root, Cargo uses `workspace.default-members` even when the
/// root manifest also contains a `[package]`. Inside a member directory, Cargo
/// instead selects that member. `cargo metadata` has already resolved the
/// workspace defaults for us, so mirror that distinction here.
fn cargo_selected_packages<'a>(
    metadata: &serde_json::Value,
    packages: &'a [serde_json::Value],
    manifest_path: &Path,
) -> Result<Vec<&'a serde_json::Value>, String> {
    let workspace_root = metadata
        .get("workspace_root")
        .and_then(|path| path.as_str())
        .ok_or_else(|| "Cargo metadata did not include workspace_root".to_string())?;
    let workspace_manifest = PathBuf::from(workspace_root).join("Cargo.toml");
    let workspace_manifest = workspace_manifest.canonicalize().map_err(|error| {
        format!(
            "could not resolve workspace manifest {}: {error}",
            workspace_manifest.display()
        )
    })?;

    if manifest_path != workspace_manifest {
        let package = packages
            .iter()
            .find(|package| cargo_package_manifest_matches(package, manifest_path))
            .ok_or_else(|| {
                format!(
                    "could not determine the Cargo package for {}",
                    manifest_path.display()
                )
            })?;
        return Ok(vec![package]);
    }

    let default_members = metadata
        .get("workspace_default_members")
        .and_then(|members| members.as_array())
        .ok_or_else(|| "Cargo metadata did not include workspace_default_members".to_string())?;
    if default_members.is_empty() {
        return Err("Cargo selected no workspace default members".to_string());
    }

    default_members
        .iter()
        .map(|member| {
            let package_id = member.as_str().ok_or_else(|| {
                "Cargo metadata contained a non-string workspace default member".to_string()
            })?;
            packages
                .iter()
                .find(|package| cargo_package_id(package).ok() == Some(package_id))
                .ok_or_else(|| {
                    format!(
                        "Cargo workspace default member `{package_id}` was missing from metadata packages"
                    )
                })
        })
        .collect()
}

fn cargo_package_manifest_matches(package: &serde_json::Value, manifest_path: &Path) -> bool {
    package
        .get("manifest_path")
        .and_then(|path| path.as_str())
        .and_then(|path| PathBuf::from(path).canonicalize().ok())
        .is_some_and(|path| path == manifest_path)
}

fn cargo_metadata(manifest_dir: &Path) -> Result<serde_json::Value, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .current_dir(manifest_dir)
        .output()
        .map_err(|error| format!("could not start cargo metadata: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cargo metadata failed with status {}{}{}",
            output.status,
            if stderr.is_empty() { "" } else { ": " },
            stderr.trim()
        ));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("could not parse cargo metadata JSON: {error}"))
}

fn cargo_package_id(package: &serde_json::Value) -> Result<&str, String> {
    package
        .get("id")
        .and_then(|id| id.as_str())
        .ok_or_else(|| "Cargo metadata package is missing id".to_string())
}

fn cargo_package_name(package: &serde_json::Value) -> Result<&str, String> {
    package
        .get("name")
        .and_then(|name| name.as_str())
        .ok_or_else(|| "Cargo metadata package is missing name".to_string())
}

fn cargo_selected_package(package: &serde_json::Value) -> Result<CargoSelectedPackage, String> {
    Ok(CargoSelectedPackage {
        package_id: cargo_package_id(package)?.to_string(),
        package_name: cargo_package_name(package)?.to_string(),
        default_run: package
            .get("default_run")
            .and_then(|name| name.as_str())
            .map(str::to_owned),
    })
}

pub(super) fn select_cargo_executable_artifact(
    selection: &CargoExecutableSelection,
    executables: &[CargoExecutableArtifact],
) -> Result<PathBuf, String> {
    if let Some(explicit_bin) = selection.explicit_bin.as_deref() {
        let matches = selection
            .packages
            .iter()
            .flat_map(|package| {
                executables
                    .iter()
                    .filter(move |artifact| {
                        artifact.package_id == package.package_id
                            && artifact.target_name == explicit_bin
                    })
                    .map(move |artifact| (package, artifact))
            })
            .collect::<Vec<_>>();
        return match matches.as_slice() {
            [(_, artifact)] => Ok(artifact.path.clone()),
            [] => Err(format!(
                "Cargo produced no executable artifact for target `{explicit_bin}` in selected packages {}",
                selected_package_names(selection)
            )),
            matches => Err(format!(
                "Cargo produced executable target `{explicit_bin}` for multiple selected packages: {}; run from a package directory",
                matches
                    .iter()
                    .map(|(package, _)| package.package_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        };
    }

    let mut candidates = Vec::new();
    for package in &selection.packages {
        let artifacts = executables
            .iter()
            .filter(|artifact| artifact.package_id == package.package_id)
            .collect::<Vec<_>>();

        if let Some(default_run) = package.default_run.as_deref() {
            let matches = artifacts
                .iter()
                .copied()
                .filter(|artifact| artifact.target_name == default_run)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [artifact] => candidates.push((package, *artifact)),
                [] => {
                    return Err(format!(
                        "Cargo produced no executable artifact for package `{}` default-run target `{default_run}`",
                        package.package_name
                    ));
                }
                _ => {
                    return Err(format!(
                        "Cargo produced multiple executable artifacts for package `{}` default-run `{default_run}`",
                        package.package_name
                    ));
                }
            }
            continue;
        }

        // A selected package without an emitted binary may simply be a
        // library-only workspace member. A package with `default-run` is
        // handled above: silently skipping its missing target could launch a
        // different default member's program instead.
        if artifacts.is_empty() {
            continue;
        }

        match artifacts.as_slice() {
            [artifact] => candidates.push((package, *artifact)),
            artifacts => {
                let choices = artifacts
                    .iter()
                    .map(|artifact| artifact.target_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "Cargo produced multiple executable targets for package `{}`: {choices}; pass --bin <name>",
                    package.package_name
                ));
            }
        }
    }

    match candidates.as_slice() {
        [(_, artifact)] => Ok(artifact.path.clone()),
        [] => Err(format!(
            "Cargo produced no executable artifact for selected packages {}",
            selected_package_names(selection)
        )),
        candidates => Err(format!(
            "Cargo produced executables for multiple selected packages: {}; pass --bin <name> that is unique among them",
            candidates
                .iter()
                .map(|(package, _)| package.package_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn selected_package_names(selection: &CargoExecutableSelection) -> String {
    selection
        .packages
        .iter()
        .map(|package| package.package_name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}
