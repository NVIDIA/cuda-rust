/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cutlass_toolchain;

use super::*;

/// Device debug-information policy requested on the command line.
///
/// Mirrors nvcc: `--lineinfo` is `-lineinfo` (line tables, optimization intact)
/// and `--device-debug` is `-G` (full debug, libNVVM optimization disabled).
/// The two are ordered, not exclusive: asking for both yields [`Self::Full`],
/// because full debug already carries line tables.
///
/// This is the CLI surface for a policy that already exists end to end --
/// `CUDA_OXIDE_DEBUG`, `ArtifactCompileOptions`'s debug bits, and
/// `FinalizationOptions::with_debug_policy` all predate it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceDebug {
    /// Request no device debug information (the default).
    #[default]
    Off,
    /// Preserve source line mappings without disabling optimization.
    LineTables,
    /// Emit full debug information; the LLVM side runs unoptimized (nvcc `-G`),
    /// debugger-hostile MIR passes are disabled, and `DisjointSlice::get_mut`
    /// is temporarily non-inline for accurate NVPTX frame ranges.
    Full,
}

impl DeviceDebug {
    /// Resolve the two independent CLI booleans into one ordered policy.
    #[must_use]
    pub fn from_flags(lineinfo: bool, device_debug: bool) -> Self {
        match (device_debug, lineinfo) {
            (true, _) => Self::Full,
            (false, true) => Self::LineTables,
            (false, false) => Self::Off,
        }
    }

    /// Value for `CUDA_OXIDE_DEBUG`, or `None` when nothing must be exported.
    ///
    /// `Off` deliberately returns `None` rather than `"off"`: exporting `off`
    /// would override a debug level the surrounding environment had already
    /// asked for, turning an absent flag into an active opt-out.
    #[must_use]
    pub fn env_value(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::LineTables => Some("line"),
            Self::Full => Some("full"),
        }
    }
}

/// Options for `cargo oxide build -- ...` / `cargo oxide test -- ...`.
#[derive(Clone, Copy)]
pub struct CargoPassthroughOptions<'a> {
    pub verbose: bool,
    pub emit_nvvm_ir: bool,
    pub arch: Option<&'a str>,
    pub features: Option<&'a str>,
    pub cargo_target_dir: Option<&'a Path>,
    pub device_codegen_crate: Option<&'a str>,
    pub device_cfgs: &'a [String],
    pub no_fmad: bool,
    pub unchecked_indexing: bool,
    pub materialize_cubin: bool,
    pub device_debug: DeviceDebug,
    pub debug_assertions: bool,
}

/// Cargo operations supported by the passthrough path.
///
/// The subcommand determines who owns profile-related rustc flags: regular
/// builds retain cuda-oxide's release-like defaults, while tests leave the
/// selected Cargo profile intact (including `--release` and `--profile`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoPassthroughSubcommand {
    Build,
    Test,
}

impl CargoPassthroughSubcommand {
    fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Test => "test",
        }
    }

    pub(super) fn codegen_profile(self, debug_assertions: bool) -> CodegenProfilePolicy {
        match self {
            Self::Build => CodegenProfilePolicy::release_like(debug_assertions),
            Self::Test => CodegenProfilePolicy::CargoSelected,
        }
    }
}

pub(super) fn normalize_device_codegen_crates(raw: &str) -> Result<String, String> {
    let mut normalized = Vec::new();
    for item in raw.split(',') {
        let name = item.trim().replace('-', "_");
        if name.is_empty() {
            return Err(
                "--device-codegen-crate requires a comma-separated list without empty entries"
                    .to_string(),
            );
        }
        if !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            return Err(format!(
                "invalid device-codegen crate name `{item}`; use Cargo crate names separated by commas"
            ));
        }
        if !normalized.contains(&name) {
            normalized.push(name);
        }
    }
    Ok(normalized.join(","))
}

pub(super) fn project_config_env<'a>(ctx: &'a Context, key: &str) -> Option<&'a str> {
    ctx.config
        .env
        .iter()
        .find(|(configured_key, _)| configured_key == key)
        .map(|(_, value)| value.as_str())
}

fn configured_device_codegen_crates(
    ctx: &Context,
    explicit: Option<&str>,
) -> Result<Option<String>, String> {
    let inherited = std::env::var(DEVICE_CODEGEN_CRATE_ENV).ok();
    resolve_device_codegen_crates(
        explicit,
        inherited.as_deref(),
        project_config_env(ctx, DEVICE_CODEGEN_CRATE_ENV),
    )
}

pub(super) fn resolve_device_codegen_crates(
    explicit: Option<&str>,
    inherited: Option<&str>,
    configured: Option<&str>,
) -> Result<Option<String>, String> {
    if let Some(explicit) = explicit {
        return normalize_device_codegen_crates(explicit).map(Some);
    }

    inherited
        .or(configured)
        .filter(|value| !value.trim().is_empty())
        .map(normalize_device_codegen_crates)
        .transpose()
}

/// The ambient environment, in the shape the fingerprint helpers consume.
///
/// Split out so both fingerprint wrappers share one collection, and so their
/// `_with_env` counterparts stay the only entry points a unit test needs.
pub(super) fn inherited_process_env() -> BTreeMap<String, Vec<u8>> {
    std::env::vars_os()
        .filter_map(|(key, value)| {
            key.into_string()
                .ok()
                .map(|key| (key, value.as_encoded_bytes().to_vec()))
        })
        .collect()
}

pub(super) const DEVICE_BACKEND_ENV: &str = "CUDA_OXIDE_DEVICE_BACKEND";
pub(super) const CUTLASS_DIGEST_FINGERPRINT_ENV: &str =
    "CUDA_OXIDE_INTERNAL_CUTLASS_COMPILER_SHA256";

/// Include the selected CUTLASS compiler and its content identity in the
/// semantic environment seen by Cargo.
pub(super) fn inherited_process_env_with_cutlass_compiler_identity(
    ctx: &Context,
) -> BTreeMap<String, Vec<u8>> {
    let mut inherited = inherited_process_env();
    let inherited_compiler_path =
        std::env::var_os(cutlass_toolchain::CUTLASS_COMPILER_ENV).map(PathBuf::from);
    augment_env_with_cutlass_compiler_identity(
        ctx,
        &mut inherited,
        inherited_compiler_path.as_deref(),
    );
    inherited
}

pub(super) fn augment_env_with_cutlass_compiler_identity(
    ctx: &Context,
    inherited: &mut BTreeMap<String, Vec<u8>>,
    inherited_compiler_path: Option<&Path>,
) {
    let backend = inherited
        .get(DEVICE_BACKEND_ENV)
        .map(Vec::as_slice)
        .or_else(|| project_config_env(ctx, DEVICE_BACKEND_ENV).map(str::as_bytes));
    if !backend.is_some_and(selector_is_cutlass_mlir) {
        return;
    }

    let explicit_compiler = if inherited.contains_key(cutlass_toolchain::CUTLASS_COMPILER_ENV) {
        inherited_compiler_path.map(Path::to_path_buf)
    } else {
        project_config_env(ctx, cutlass_toolchain::CUTLASS_COMPILER_ENV).map(PathBuf::from)
    };
    if inherited.contains_key(cutlass_toolchain::CUTLASS_COMPILER_ENV)
        || explicit_compiler.is_some()
    {
        let digest = match explicit_compiler.as_deref() {
            Some(path) => cutlass_toolchain::sha256_file(path)
                .unwrap_or_else(|error| format!("unavailable:{error}")),
            None => "unavailable:non-unicode inherited compiler path".to_string(),
        };
        inherited.insert(
            CUTLASS_DIGEST_FINGERPRINT_ENV.to_string(),
            digest.into_bytes(),
        );
        return;
    }

    if let Ok(Some(paths)) = cutlass_toolchain::resolve_official() {
        inherited.insert(
            cutlass_toolchain::CUTLASS_COMPILER_ENV.to_string(),
            paths
                .compiler_library
                .as_os_str()
                .as_encoded_bytes()
                .to_vec(),
        );
        inherited.insert(
            CUTLASS_DIGEST_FINGERPRINT_ENV.to_string(),
            paths.compiler_library_sha256.as_bytes().to_vec(),
        );
    }
}

pub(super) fn selector_is_cutlass_mlir(value: &[u8]) -> bool {
    std::str::from_utf8(value).is_ok_and(|value| value.trim() == "cutlass-mlir")
}

fn cargo_passthrough_command(
    ctx: &Context,
    cargo_subcommand: CargoPassthroughSubcommand,
    opts: &CargoPassthroughOptions<'_>,
    cargo_args: &[String],
) -> Result<Command, String> {
    cargo_passthrough_command_with_env(
        ctx,
        cargo_subcommand,
        opts,
        cargo_args,
        std::env::var_os(MATERIALIZE_ENV),
    )
}

/// `cargo_passthrough_command` with the ambient
/// `CUDA_OXIDE_MATERIALIZE_CUBIN` injected.
///
/// Unit tests must call this with `None`: the ambient value outranks
/// `opts.materialize_cubin`, so an exported one turns materialization on and
/// sends the test into `discover_materializer_provenance`, which re-executes
/// `current_exe` -- the libtest binary under `cargo test` -- and then exits the
/// process over the unusable digest, taking the whole suite with it.
pub(super) fn cargo_passthrough_command_with_env(
    ctx: &Context,
    cargo_subcommand: CargoPassthroughSubcommand,
    opts: &CargoPassthroughOptions<'_>,
    cargo_args: &[String],
    materialize_env: Option<std::ffi::OsString>,
) -> Result<Command, String> {
    let target_arch = configured_arch(ctx, opts.arch);
    let materialization = prepare_materialization_with_env(
        ctx,
        opts.materialize_cubin,
        opts.arch,
        opts.emit_nvvm_ir,
        materialize_env,
    );
    let owner_filter = configured_device_codegen_crates(ctx, opts.device_codegen_crate)?;
    // Device-owning macros track this identity in their crate dep-info. Keep it
    // out of global rustflags so host-only dependencies retain one cache key.
    let fingerprint = passthrough_codegen_fingerprint(
        ctx,
        opts,
        owner_filter.as_deref(),
        target_arch,
        &materialization,
    );
    let mut cmd = Command::new("cargo");
    cmd.arg(cargo_subcommand.as_str());
    if let Some(features) = opts.features {
        cmd.args(["--features", features]);
    }
    cmd.args(cargo_args).current_dir(&ctx.workspace_root);

    // Project configuration provides defaults. Explicit wrapper flags and
    // internal compiler requirements are applied afterward and therefore win.
    apply_common_codegen_env(
        &mut cmd,
        ctx,
        opts.verbose,
        opts.no_fmad,
        opts.unchecked_indexing,
        opts.device_debug,
    );
    apply_codegen_configuration(
        &mut cmd,
        ctx,
        cargo_subcommand.codegen_profile(opts.debug_assertions),
        opts.device_cfgs,
        &fingerprint,
    )?;

    if let Some(cargo_target_dir) = opts.cargo_target_dir {
        cmd.env("CARGO_TARGET_DIR", cargo_target_dir);
    }
    if let Some(owner_filter) = owner_filter {
        cmd.env(DEVICE_CODEGEN_CRATE_ENV, owner_filter);
    }
    apply_output_mode(&mut cmd, opts.emit_nvvm_ir, target_arch, &materialization);
    Ok(cmd)
}

/// Run an arbitrary Cargo build-like subcommand through the cuda-oxide backend.
///
/// Unlike example mode, this does not touch source files or clean generated
/// artifacts. It is intended for final-target workspace builds where Cargo's
/// incremental behavior should remain intact.
pub fn codegen_cargo_passthrough(
    ctx: &Context,
    cargo_subcommand: CargoPassthroughSubcommand,
    opts: CargoPassthroughOptions<'_>,
    cargo_args: &[String],
) {
    let cargo_subcommand_name = cargo_subcommand.as_str();
    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA CARGO {}", cargo_subcommand_name);
    println!("=========================================");
    println!();

    let mut cmd = cargo_passthrough_command(ctx, cargo_subcommand, &opts, cargo_args)
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(2);
        });

    let displayed_args: Vec<_> = cmd
        .get_args()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    if displayed_args.is_empty() {
        println!("Running cargo {}...", cargo_subcommand_name);
    } else {
        println!(
            "Running cargo {} {}...",
            cargo_subcommand_name,
            displayed_args.join(" ")
        );
    }
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!(
            "\nCargo {} failed with exit code: {:?}",
            cargo_subcommand_name,
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    println!();
    println!("✓ Cargo {} succeeded", cargo_subcommand_name);
}
