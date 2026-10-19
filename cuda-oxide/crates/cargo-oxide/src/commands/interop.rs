/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

// =============================================================================
// Interop host/device workflow
// =============================================================================

#[derive(Debug, Clone)]
pub(super) struct InteropConfig {
    pub(super) kind: Option<String>,
    pub(super) device_crates: Vec<DeviceCrateConfig>,
}

#[derive(Debug, Clone)]
pub(super) struct DeviceCrateConfig {
    pub(super) manifest_path: PathBuf,
    pub(super) artifact_dir: PathBuf,
    pub(super) artifact_name: Option<String>,
    pub(super) artifact_kind: InteropArtifactKind,
    pub(super) source_identity: bool,
    pub(super) bin: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum InteropArtifactKind {
    #[default]
    Ptx,
    Cubin,
}

impl InteropArtifactKind {
    fn extension(self) -> &'static str {
        match self {
            Self::Ptx => "ptx",
            Self::Cubin => "cubin",
        }
    }

    fn emits_nvvm_ir(self) -> bool {
        self == Self::Cubin
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) struct InteropBinaryTarget {
    pub(super) source_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InteropDeviceBuildRoute {
    Build,
    Run,
    Sanitize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InteropDeviceBuildOptions {
    pub(super) no_fmad: bool,
    pub(super) unchecked_indexing: bool,
    pub(super) device_debug: DeviceDebug,
    pub(super) sanitizer_line_tables: bool,
    pub(super) debug_assertions: bool,
}

impl InteropDeviceBuildOptions {
    pub(super) fn for_route(
        route: InteropDeviceBuildRoute,
        no_fmad: bool,
        unchecked_indexing: bool,
        device_debug: DeviceDebug,
        debug_assertions: bool,
    ) -> Self {
        Self {
            no_fmad,
            unchecked_indexing,
            device_debug,
            sanitizer_line_tables: route == InteropDeviceBuildRoute::Sanitize,
            debug_assertions,
        }
    }

    pub(super) fn codegen_profile(self) -> CodegenProfilePolicy {
        CodegenProfilePolicy::release_like(self.debug_assertions)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn codegen_run_interop(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    features: Option<&str>,
    device_features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialization: &MaterializationMode,
    app_args: &[String],
) {
    reject_interop_output_mode(emit_nvvm_ir, materialization);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA INTEROP: {}", example);
    println!("=========================================");
    if let Some(kind) = &interop.kind {
        println!("Interop kind: {}", kind);
    }
    if let Some(dev) = detected_device_arch {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }
    println!();

    build_interop_device_crates(
        ctx,
        example_dir,
        interop,
        verbose,
        arch,
        detected_device_arch,
        device_features,
        InteropDeviceBuildOptions::for_route(
            InteropDeviceBuildRoute::Run,
            no_fmad,
            unchecked_indexing,
            device_debug,
            false,
        ),
        materialization,
    );
    run_host_cargo(
        ctx,
        example,
        example_dir,
        "run",
        features,
        bin,
        verbose,
        app_args,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn codegen_build_interop(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    device_features: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    debug_assertions: bool,
    materialization: &MaterializationMode,
) {
    reject_interop_output_mode(emit_nvvm_ir, materialization);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA INTEROP BUILD: {}", example);
    println!("=========================================");
    if let Some(kind) = &interop.kind {
        println!("Interop kind: {}", kind);
    }
    println!();

    // `build` may cross-compile for another machine, so no device-arch hint:
    // only an explicit `--arch` pins the target here.
    build_interop_device_crates(
        ctx,
        example_dir,
        interop,
        verbose,
        arch,
        None,
        device_features,
        InteropDeviceBuildOptions::for_route(
            InteropDeviceBuildRoute::Build,
            no_fmad,
            unchecked_indexing,
            device_debug,
            debug_assertions,
        ),
        materialization,
    );
    run_host_cargo(
        ctx,
        example,
        example_dir,
        "build",
        features,
        None,
        verbose,
        &[],
    );
}

pub(super) fn reject_interop_output_mode(
    emit_nvvm_ir: bool,
    materialization: &MaterializationMode,
) {
    if materialization.enabled() {
        eprintln!("Error: --materialize-cubin is not supported for metadata interop examples yet.");
        eprintln!(
            "Declare `artifact-kind = \"cubin\"` on each device crate that requires native output."
        );
        std::process::exit(2);
    }
    if emit_nvvm_ir {
        eprintln!("Error: --emit-nvvm-ir is not supported for metadata interop examples yet.");
        eprintln!("Interop device output is selected by each metadata `artifact-kind`.");
        std::process::exit(2);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_interop_device_crates(
    ctx: &Context,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    device_features: Option<&str>,
    options: InteropDeviceBuildOptions,
    materialization: &MaterializationMode,
) {
    for device_crate in &interop.device_crates {
        build_interop_device_crate(
            ctx,
            example_dir,
            device_crate,
            verbose,
            arch,
            detected_device_arch,
            device_features,
            options,
            materialization,
        );
    }
}

pub(super) fn interop_device_artifact_name(
    manifest_path: &Path,
    device_crate: &DeviceCrateConfig,
) -> String {
    device_crate.artifact_name.clone().unwrap_or_else(|| {
        normalize_crate_name(&interop_device_cargo_target_name(
            manifest_path,
            device_crate,
        ))
    })
}

pub(super) fn interop_device_cargo_target_name(
    manifest_path: &Path,
    device_crate: &DeviceCrateConfig,
) -> String {
    device_crate
        .bin
        .clone()
        .unwrap_or_else(|| package_name_from_manifest(manifest_path))
}

pub(super) fn interop_device_artifact_path(
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    artifact_name: &str,
) -> PathBuf {
    example_dir.join(&device_crate.artifact_dir).join(format!(
        "{}.{}",
        artifact_stem(artifact_name),
        device_crate.artifact_kind.extension()
    ))
}

/// Pre-build requirement check for `artifact-kind = "cubin"` device crates.
///
/// A native cubin needs a deliberate target, so require one from `--arch`,
/// `CUDA_OXIDE_TARGET`, project configuration, or the device detected by
/// `run` before spending a device build. This is a requirement check only:
/// the arch the finalizer actually compiles for comes from the
/// backend-recorded `.target` sidecar (see
/// [`read_interop_recorded_target`]), because the backend may resolve a
/// different arch than the hint (e.g. escalate a detected `sm_120a` to the
/// `sm_90a` WGMMA floor).
pub(super) fn interop_cubin_target(
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
) -> Result<cuda_artifact_finalizer::CudaArch, String> {
    let target = arch
        .map(str::to_owned)
        .or_else(|| std::env::var("CUDA_OXIDE_TARGET").ok())
        .or_else(|| detected_device_arch.map(str::to_owned))
        .ok_or_else(|| {
            "cubin interop artifacts require --arch, CUDA_OXIDE_TARGET, a configured target, or a detected run device"
                .to_string()
        })?;
    parse_nvvm_arch(&target)
        .map_err(|error| format!("invalid cubin interop target {target:?}: {error}"))
}

/// Path of the NVVM IR a cubin-kind device crate emits into its artifact dir.
fn interop_device_ir_path(
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    artifact_name: &str,
) -> PathBuf {
    example_dir
        .join(&device_crate.artifact_dir)
        .join(format!("{}.ll", artifact_stem(artifact_name)))
}

/// Read the CUDA target the backend recorded next to an emitted NVVM IR.
///
/// The backend publishes `<name>.target` last, after the `.ll` and
/// `.options`: it is both the authoritative arch record (NVVM IR does not
/// encode its target) and the completion marker saying the sibling
/// `.options` file is present and required (see
/// `write_nvvm_target_sidecar` in mir-importer). A missing or malformed
/// sidecar therefore means the device build did not complete its artifact
/// contract, never that some other arch should be guessed.
pub(super) fn read_interop_recorded_target(ir_path: &Path) -> Result<String, String> {
    let path = ir_path.with_extension("target");
    let text = std::fs::read_to_string(&path).map_err(|error| {
        format!(
            "device build did not record its CUDA target at {} ({error}); \
             the .target sidecar is the backend's completion marker, so the \
             emitted NVVM IR cannot be trusted without it",
            path.display()
        )
    })?;
    let mut lines = text.lines();
    let target = lines.next().unwrap_or_default().trim();
    if target.is_empty() {
        return Err(format!("recorded CUDA target {} is empty", path.display()));
    }
    match (lines.next(), lines.next()) {
        (None, None) => Ok(target.to_string()),
        (Some(marker), None) if marker == oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER => {
            let options_path = ir_path.with_extension("options");
            if !options_path.is_file() {
                return Err(format!(
                    "recorded CUDA target {} requires the sibling compile options {}, which is missing",
                    path.display(),
                    options_path.display()
                ));
            }
            Ok(target.to_string())
        }
        _ => Err(format!(
            "recorded CUDA target {} has an unrecognized format: {:?}",
            path.display(),
            text.trim()
        )),
    }
}

/// Read the `.target sm_XX` directive from an emitted PTX artifact.
///
/// PTX carries its own target record, so the identity sidecar can state the
/// arch the artifact was actually compiled for instead of echoing a request
/// hint (which the backend is allowed to override).
pub(super) fn ptx_recorded_target(ptx_path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(ptx_path).map_err(|error| {
        format!(
            "could not read emitted PTX at {}: {error}",
            ptx_path.display()
        )
    })?;
    let document = ptx_parse::Document::parse(&text).map_err(|error| {
        format!(
            "could not parse emitted PTX {} to read its target: {error}",
            ptx_path.display()
        )
    })?;
    document
        .directives()
        .iter()
        .find(|directive| directive.name() == ".target")
        .and_then(|directive| ptx_parse::split_top_level(directive.arguments()))
        .and_then(|arguments| arguments.first().copied())
        .map(str::to_string)
        .filter(|target| !target.is_empty())
        .ok_or_else(|| {
            format!(
                "emitted PTX {} does not declare a .target directive",
                ptx_path.display()
            )
        })
}

/// The CUDA target an interop device artifact was actually built for, read
/// from the emitted artifact itself: the backend `.target` sidecar for
/// cubin (the same record the finalizer compiled with) and the `.target`
/// directive for PTX.
pub(super) fn interop_artifact_recorded_target(
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    artifact_name: &str,
) -> Result<String, String> {
    match device_crate.artifact_kind {
        InteropArtifactKind::Ptx => ptx_recorded_target(&interop_device_artifact_path(
            example_dir,
            device_crate,
            artifact_name,
        )),
        InteropArtifactKind::Cubin => read_interop_recorded_target(&interop_device_ir_path(
            example_dir,
            device_crate,
            artifact_name,
        )),
    }
}

fn finalize_interop_device_artifact(
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    artifact_name: &str,
) -> PathBuf {
    let artifact_path = interop_device_artifact_path(example_dir, device_crate, artifact_name);
    match device_crate.artifact_kind {
        InteropArtifactKind::Ptx => artifact_path,
        InteropArtifactKind::Cubin => {
            let ir_path = interop_device_ir_path(example_dir, device_crate, artifact_name);
            // The backend-recorded target, not the CLI/env/detected hint:
            // the backend may have resolved a different arch, and this
            // sidecar doubles as the completion marker for the .ll/.options
            // pair consumed below.
            let recorded_target = read_interop_recorded_target(&ir_path).unwrap_or_else(|error| {
                eprintln!("Error: {error}");
                std::process::exit(1);
            });
            let target = parse_nvvm_arch(&recorded_target).unwrap_or_else(|error| {
                eprintln!(
                    "Error: invalid recorded CUDA target {recorded_target:?} next to {}: {error}",
                    ir_path.display()
                );
                std::process::exit(1);
            });
            let ir = std::fs::read(&ir_path).unwrap_or_else(|error| {
                eprintln!(
                    "Error: could not read emitted NVVM IR at {}: {error}",
                    ir_path.display()
                );
                std::process::exit(1);
            });
            let options_path = ir_path.with_extension("options");
            let options_text = std::fs::read_to_string(&options_path).unwrap_or_else(|error| {
                eprintln!(
                    "Error: could not read emitted compile options at {}: {error}",
                    options_path.display()
                );
                std::process::exit(1);
            });
            let compile_options =
                oxide_artifacts::ArtifactCompileOptions::from_sidecar_text(&options_text)
                    .unwrap_or_else(|error| {
                        eprintln!(
                            "Error: invalid emitted compile options at {}: {error}",
                            options_path.display()
                        );
                        std::process::exit(1);
                    });
            let finalizer = cuda_artifact_finalizer::Finalizer::discover().unwrap_or_else(|error| {
                eprintln!("Error: could not initialize the CUDA artifact finalizer: {error}");
                eprintln!(
                    "libNVVM and nvJitLink ship with the CUDA Toolkit; run `cargo oxide doctor` to check discovery."
                );
                std::process::exit(1);
            });
            let options = finalization_options_from_artifact(&target, compile_options);
            let cubin = finalizer
                .materialize_nvvm_ir(artifact_name, &ir, &options)
                .unwrap_or_else(|error| {
                    eprintln!(
                        "Error: could not finalize {} for {}: {error}",
                        ir_path.display(),
                        target.sm()
                    );
                    std::process::exit(1);
                });
            let temporary_path = artifact_path.with_extension("cubin.tmp");
            std::fs::write(&temporary_path, cubin).unwrap_or_else(|error| {
                eprintln!(
                    "Error: could not write temporary cubin {}: {error}",
                    temporary_path.display()
                );
                std::process::exit(1);
            });
            std::fs::rename(&temporary_path, &artifact_path).unwrap_or_else(|error| {
                eprintln!(
                    "Error: could not install cubin {}: {error}",
                    artifact_path.display()
                );
                std::process::exit(1);
            });
            artifact_path
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_interop_device_crate(
    ctx: &Context,
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    device_features: Option<&str>,
    options: InteropDeviceBuildOptions,
    materialization: &MaterializationMode,
) {
    let manifest_path = example_dir.join(&device_crate.manifest_path);
    let manifest_path = manifest_path.canonicalize().unwrap_or_else(|e| {
        eprintln!(
            "Error: could not resolve device crate manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let device_dir = manifest_path.parent().unwrap_or(example_dir);
    // One `cargo metadata` invocation answers both consumers: bin-target
    // resolution before the build and the uplifted depfile location after
    // it. Skip it entirely when neither feature is requested.
    let cargo_metadata = (device_crate.bin.is_some() || device_crate.source_identity).then(|| {
        interop_cargo_metadata(ctx, device_dir, &manifest_path).unwrap_or_else(|error| {
            eprintln!("Error: could not query cargo metadata for the device crate: {error}");
            std::process::exit(1);
        })
    });
    let binary_target = device_crate.bin.as_deref().map(|bin| {
        let metadata = cargo_metadata
            .as_ref()
            .expect("cargo metadata is fetched whenever a bin target is configured");
        interop_binary_target_from_metadata(metadata, &manifest_path, bin).unwrap_or_else(|error| {
            eprintln!("Error: could not resolve device binary target: {error}");
            std::process::exit(1);
        })
    });
    let artifact_dir = example_dir.join(&device_crate.artifact_dir);
    std::fs::create_dir_all(&artifact_dir).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not create device artifact directory {}: {}",
            artifact_dir.display(),
            e
        );
        std::process::exit(1);
    });

    if device_crate.artifact_kind == InteropArtifactKind::Cubin
        && let Err(error) = interop_cubin_target(arch, detected_device_arch)
    {
        eprintln!("Error: {error}");
        std::process::exit(2);
    }

    let artifact_name = interop_device_artifact_name(&manifest_path, device_crate);
    clean_generated_files(&artifact_dir, &artifact_name);
    if let Some(target) = &binary_target {
        touch_source_file(&target.source_path);
    } else {
        touch_main_rs(device_dir);
    }

    println!("Building device crate {}...", manifest_path.display());

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--manifest-path"])
        .arg(&manifest_path)
        .current_dir(device_dir);
    if let Some(device_features) = device_features {
        cmd.args(["--features", device_features]);
    }
    if let Some(bin) = &device_crate.bin {
        cmd.args(["--bin", bin]);
    }

    apply_interop_device_codegen_options(&mut cmd, ctx, verbose, options);
    let fingerprint = interop_codegen_fingerprint(
        ctx,
        verbose,
        options,
        arch,
        detected_device_arch,
        &artifact_dir,
        device_crate.artifact_kind.emits_nvvm_ir(),
        device_features,
        materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        options.codegen_profile(),
        &[],
        &fingerprint,
    );
    // This is an internal artifact contract, so it must override a project
    // `[env]` default for the same variable.
    cmd.env("CUDA_OXIDE_PTX_DIR", &artifact_dir);
    apply_output_mode(
        &mut cmd,
        device_crate.artifact_kind.emits_nvvm_ir(),
        arch,
        materialization,
    );
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch);

    let status = cmd.status().expect("Failed to build interop device crate");
    if !status.success() {
        eprintln!(
            "\nDevice crate build failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    let artifact_path = finalize_interop_device_artifact(example_dir, device_crate, &artifact_name);
    if !artifact_path.exists() {
        eprintln!(
            "Error: device crate build succeeded but did not produce {}",
            artifact_path.display()
        );
        std::process::exit(1);
    }
    println!(
        "{} written: {}",
        device_crate.artifact_kind.extension().to_ascii_uppercase(),
        artifact_path.display()
    );
    if device_crate.source_identity {
        // The identity records the arch the artifact was actually built
        // for, read back from the emitted artifact, never the request hint.
        let artifact_target =
            interop_artifact_recorded_target(example_dir, device_crate, &artifact_name)
                .unwrap_or_else(|error| {
                    eprintln!(
                        "Error: could not determine the built CUDA target for {}: {error}",
                        artifact_path.display()
                    );
                    std::process::exit(1);
                });
        let cargo_target_name = interop_device_cargo_target_name(&manifest_path, device_crate);
        let metadata = cargo_metadata
            .as_ref()
            .expect("cargo metadata is fetched whenever source-identity is configured");
        let cargo_target_dir = cargo_target_directory(metadata).unwrap_or_else(|error| {
            eprintln!("Error: could not locate the device cargo target directory: {error}");
            std::process::exit(1);
        });
        let depfile_path = release_depfile_path(&cargo_target_dir, &cargo_target_name);
        if !depfile_path.is_file() {
            eprintln!(
                "Error: device crate build succeeded but did not produce dependency file {}",
                depfile_path.display()
            );
            std::process::exit(1);
        }
        let identity_base = artifact_path.parent().unwrap_or(example_dir);
        let identity_path = crate::artifact_identity::write(
            &artifact_path,
            &depfile_path,
            &manifest_path,
            identity_base,
            &cargo_target_dir,
            &artifact_target,
            device_features,
        )
        .unwrap_or_else(|error| {
            eprintln!(
                "Error: could not write source identity for {}: {error}",
                artifact_path.display()
            );
            std::process::exit(1);
        });
        println!("Artifact identity written: {}", identity_path.display());
    }
}

pub(super) fn load_interop_config(example_dir: &Path) -> Option<InteropConfig> {
    let manifest_path = example_dir.join("Cargo.toml");
    let source = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let document: toml::Value = toml::from_str(&source).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not parse manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });

    let oxide = document
        .get("package")
        .and_then(|value| value.get("metadata"))
        .and_then(|value| value.get("cuda-oxide"))?;

    let kind = oxide.get("interop").and_then(|value| {
        value.as_str().map(str::to_string).or_else(|| {
            value
                .get("kind")
                .and_then(|kind| kind.as_str())
                .map(str::to_string)
        })
    });

    let device_crates = oxide
        .get("device-crates")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .map(|item| parse_device_crate_config(item, &manifest_path))
                .collect()
        })
        .unwrap_or_default();

    Some(InteropConfig {
        kind,
        device_crates,
    })
}

fn parse_device_crate_config(value: &toml::Value, manifest_path: &Path) -> DeviceCrateConfig {
    let table = value.as_table().unwrap_or_else(|| {
        eprintln!(
            "Error: each package.metadata.cuda-oxide.device-crates entry in {} must be a table",
            manifest_path.display()
        );
        std::process::exit(1);
    });

    let device_manifest = required_metadata_string(table, "manifest-path", manifest_path);
    let artifact_dir = optional_metadata_string(table, "artifact-dir")
        .or_else(|| optional_metadata_string(table, "ptx-dir"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(&device_manifest)
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    let artifact_name = optional_metadata_string(table, "artifact-name");
    let artifact_kind = match optional_metadata_string(table, "artifact-kind").as_deref() {
        None | Some("ptx") => InteropArtifactKind::Ptx,
        Some("cubin") => InteropArtifactKind::Cubin,
        Some(value) => {
            eprintln!(
                "Error: package.metadata.cuda-oxide.device-crates `artifact-kind` in {} must be `ptx` or `cubin`, got {value:?}",
                manifest_path.display()
            );
            std::process::exit(2);
        }
    };
    let source_identity = match table.get("source-identity") {
        None => false,
        Some(value) => value.as_bool().unwrap_or_else(|| {
            eprintln!(
                "Error: package.metadata.cuda-oxide.device-crates `source-identity` in {} must be a boolean",
                manifest_path.display()
            );
            std::process::exit(2);
        }),
    };
    let bin = optional_metadata_string(table, "bin");

    DeviceCrateConfig {
        manifest_path: PathBuf::from(device_manifest),
        artifact_dir,
        artifact_name,
        artifact_kind,
        source_identity,
        bin,
    }
}

fn required_metadata_string(table: &toml::Table, key: &str, manifest_path: &Path) -> String {
    optional_metadata_string(table, key).unwrap_or_else(|| {
        eprintln!(
            "Error: package.metadata.cuda-oxide.device-crates entry in {} is missing string field `{}`",
            manifest_path.display(),
            key
        );
        std::process::exit(1);
    })
}

fn optional_metadata_string(table: &toml::Table, key: &str) -> Option<String> {
    table
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn package_name_from_manifest(manifest_path: &Path) -> String {
    let source = std::fs::read_to_string(manifest_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read device manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let document: toml::Value = toml::from_str(&source).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not parse device manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });

    document
        .get("package")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            eprintln!(
                "Error: device manifest {} is missing package.name",
                manifest_path.display()
            );
            std::process::exit(1);
        })
}

fn normalize_crate_name(package_name: &str) -> String {
    package_name.replace('-', "_")
}

/// The target directory cargo reports for a device crate, which locates both
/// its uplifted build products and the build outputs the identity sidecar
/// must exclude.
pub(super) fn cargo_target_directory(metadata: &serde_json::Value) -> Result<PathBuf, String> {
    metadata
        .get("target_directory")
        .and_then(|value| value.as_str())
        .map(PathBuf::from)
        .ok_or_else(|| "cargo metadata omitted target_directory".to_string())
}

/// Where cargo uplifts the dep-info file for a release binary target.
///
/// Cargo names the uplifted copy after the bin target verbatim, hyphens
/// preserved: building the `simt-device` target writes
/// `target/release/simt-device` and `target/release/simt-device.d` (this
/// repo's own build writes `target/debug/cargo-oxide.d`). Only the internal
/// per-unit copies under `target/release/deps/` use the underscore-normalized
/// crate name, so the stem here must NOT be `normalize_crate_name`d.
pub(super) fn release_depfile_path(cargo_target_dir: &Path, cargo_target_name: &str) -> PathBuf {
    cargo_target_dir
        .join("release")
        .join(format!("{cargo_target_name}.d"))
}

fn interop_cargo_metadata(
    ctx: &Context,
    device_dir: &Path,
    manifest_path: &Path,
) -> Result<serde_json::Value, String> {
    let mut command = Command::new("cargo");
    command
        .args([
            "metadata",
            "--format-version=1",
            "--no-deps",
            "--manifest-path",
        ])
        .arg(manifest_path)
        .current_dir(device_dir);
    apply_config_env(&mut command, ctx);
    let output = command
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

pub(super) fn interop_binary_target_from_metadata(
    metadata: &serde_json::Value,
    manifest_path: &Path,
    bin: &str,
) -> Result<InteropBinaryTarget, String> {
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata omitted packages".to_string())?;
    let package = packages
        .iter()
        .find(|package| {
            package
                .get("manifest_path")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|path| Path::new(path) == manifest_path)
        })
        .ok_or_else(|| {
            format!(
                "cargo metadata omitted package for manifest {}",
                manifest_path.display()
            )
        })?;
    let targets = package
        .get("targets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata omitted package targets".to_string())?;
    let is_binary = |target: &&serde_json::Value| {
        target
            .get("kind")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")))
    };
    let target = targets
        .iter()
        .filter(is_binary)
        .find(|target| target.get("name").and_then(serde_json::Value::as_str) == Some(bin))
        .ok_or_else(|| {
            let mut available = targets
                .iter()
                .filter(is_binary)
                .filter_map(|target| target.get("name").and_then(serde_json::Value::as_str))
                .collect::<Vec<_>>();
            available.sort_unstable();
            format!(
                "manifest {} has no binary target {bin:?}; available binary targets: {}",
                manifest_path.display(),
                if available.is_empty() {
                    "<none>".to_string()
                } else {
                    available.join(", ")
                }
            )
        })?;
    let source_path = target
        .get("src_path")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| format!("cargo metadata omitted source path for binary target {bin:?}"))?;
    Ok(InteropBinaryTarget { source_path })
}
