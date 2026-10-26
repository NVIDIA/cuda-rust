/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use sha2::Digest as _;
use std::collections::BTreeMap;
use std::path::Path;

use super::*;

pub(super) fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}

pub(super) fn passthrough_codegen_fingerprint(
    ctx: &Context,
    opts: &CargoPassthroughOptions<'_>,
    owner_filter: Option<&str>,
    target_arch: Option<&str>,
    materialization: &MaterializationMode,
) -> String {
    passthrough_codegen_fingerprint_with_env(
        ctx,
        opts,
        owner_filter,
        target_arch,
        materialization,
        &inherited_process_env_with_cutlass_compiler_identity(ctx),
    )
}

pub(super) fn passthrough_codegen_fingerprint_with_env(
    ctx: &Context,
    opts: &CargoPassthroughOptions<'_>,
    owner_filter: Option<&str>,
    target_arch: Option<&str>,
    materialization: &MaterializationMode,
    inherited_env: &BTreeMap<String, Vec<u8>>,
) -> String {
    let mut effective_env = BTreeMap::new();

    // Project-configured CUDA_OXIDE_* variables are defaults. Mirror the same
    // parent override rule as `apply_config_env` so changes that can affect
    // codegen also change Cargo's rustflags fingerprint.
    for (key, configured_value) in &ctx.config.env {
        if !key.starts_with("CUDA_OXIDE_") {
            continue;
        }
        if let Some(value) = inherited_env.get(key) {
            // Keep the platform encoding. Presence-only backend switches such
            // as CUDA_OXIDE_NO_FMA remain effective even when their value is
            // not Unicode, so dropping those bytes could reuse stale code.
            effective_env.insert(key.clone(), value.clone());
        } else {
            effective_env.insert(key.clone(), configured_value.as_bytes().to_vec());
        }
    }
    // Capture backend settings inherited outside project config, including
    // current and future CUDA_OXIDE_* switches.
    for (key, value) in inherited_env.iter().filter(|(key, _)| {
        key.starts_with("CUDA_OXIDE_") && key.as_str() != CODEGEN_FINGERPRINT_ENV
    }) {
        effective_env.insert(key.clone(), value.clone());
    }

    // These are wrapper-owned semantic values. Normalize away inherited
    // false/stale handshakes before inserting the effective materialization
    // state below, so no-op values do not create distinct Cargo identities.
    effective_env.remove(CODEGEN_FINGERPRINT_ENV);
    effective_env.remove(MATERIALIZE_ENV);
    effective_env.remove(EXPECTED_PROVENANCE_ENV);
    // Descriptor identity only accelerates verification; artifact identity is
    // already represented by the content-derived provenance above.
    effective_env.remove(MATERIALIZER_HANDSHAKE_ENV);

    if opts.verbose {
        effective_env.insert("CUDA_OXIDE_VERBOSE".to_string(), b"1".to_vec());
    }
    if opts.no_fmad {
        effective_env.insert("CUDA_OXIDE_NO_FMA".to_string(), b"1".to_vec());
    }
    if opts.unchecked_indexing {
        effective_env.insert("CUDA_OXIDE_UNCHECKED_INDEXING".to_string(), b"1".to_vec());
    }
    if let Some(level) = opts.device_debug.env_value() {
        effective_env.insert("CUDA_OXIDE_DEBUG".to_string(), level.as_bytes().to_vec());
    }
    if opts.emit_nvvm_ir || materialization.enabled() {
        effective_env.insert("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), b"1".to_vec());
    }
    if let Some(prepared) = &materialization.prepared {
        effective_env.insert(MATERIALIZE_ENV.to_string(), b"1".to_vec());
        effective_env.insert(
            EXPECTED_PROVENANCE_ENV.to_string(),
            prepared.provenance_sha256_hex.as_bytes().to_vec(),
        );
    }
    if let Some(target_arch) = target_arch {
        effective_env.insert(
            "CUDA_OXIDE_TARGET".to_string(),
            target_arch.as_bytes().to_vec(),
        );
    }
    if let Some(owner_filter) = owner_filter {
        effective_env.insert(
            DEVICE_CODEGEN_CRATE_ENV.to_string(),
            owner_filter.as_bytes().to_vec(),
        );
    }

    // SHA-256 over length-delimited key/value pairs. The complete digest is
    // tracked by device-owning procedural macros, so settings are neither
    // exposed verbatim in diagnostics nor reduced to a small collision space.
    let mut hash = sha2::Sha256::new();
    for (key, value) in effective_env {
        update_codegen_fingerprint_hash(&mut hash, key.as_bytes());
        update_codegen_fingerprint_hash(&mut hash, &value);
    }
    finish_codegen_fingerprint(hash)
}

fn update_codegen_fingerprint_hash(hash: &mut sha2::Sha256, bytes: &[u8]) {
    use sha2::Digest as _;

    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn finish_codegen_fingerprint(hash: sha2::Sha256) -> String {
    use sha2::Digest as _;

    let digest: [u8; 32] = hash.finalize().into();
    digest_hex(&digest)
}

/// Track sanitizer-only device output settings in crates that declare device
/// code, without invalidating their host-only dependency graph.
#[allow(clippy::too_many_arguments)]
pub(super) fn sanitize_codegen_fingerprint(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    ptx_dir: Option<&Path>,
    materialization: &MaterializationMode,
) -> String {
    sanitize_codegen_fingerprint_with_env(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        target_arch,
        detected_device_arch,
        ptx_dir,
        materialization,
        &inherited_process_env_with_cutlass_compiler_identity(ctx),
    )
}

/// `sanitize_codegen_fingerprint` with the inherited environment injected, the
/// counterpart to `passthrough_codegen_fingerprint_with_env`.
#[allow(clippy::too_many_arguments)]
pub(super) fn sanitize_codegen_fingerprint_with_env(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    ptx_dir: Option<&Path>,
    materialization: &MaterializationMode,
    inherited_env: &BTreeMap<String, Vec<u8>>,
) -> String {
    let opts = CargoPassthroughOptions {
        verbose,
        emit_nvvm_ir: false,
        arch: target_arch,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad,
        unchecked_indexing,
        materialize_cubin: materialization.enabled(),
        device_debug,
        debug_assertions: false,
    };
    let base = passthrough_codegen_fingerprint_with_env(
        ctx,
        &opts,
        None,
        target_arch,
        materialization,
        inherited_env,
    );
    let mut hash = sha2::Sha256::new();
    for bytes in [
        "sanitize-line-tables-v1".as_bytes(),
        base.as_bytes(),
        detected_device_arch.unwrap_or("").as_bytes(),
    ] {
        update_codegen_fingerprint_hash(&mut hash, bytes);
    }
    if let Some(ptx_dir) = ptx_dir {
        update_codegen_fingerprint_hash(&mut hash, ptx_dir.as_os_str().as_encoded_bytes());
    }
    finish_codegen_fingerprint(hash)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn standard_codegen_fingerprint(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    emit_nvvm_ir: bool,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    materialization: &MaterializationMode,
) -> String {
    standard_codegen_fingerprint_with_env(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        detected_device_arch,
        materialization,
        &inherited_process_env(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn standard_codegen_fingerprint_with_env(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    emit_nvvm_ir: bool,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    materialization: &MaterializationMode,
    inherited_env: &BTreeMap<String, Vec<u8>>,
) -> String {
    let opts = CargoPassthroughOptions {
        verbose,
        emit_nvvm_ir,
        arch: target_arch,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad,
        unchecked_indexing,
        materialize_cubin: materialization.enabled(),
        device_debug,
        debug_assertions: false,
    };
    let base = passthrough_codegen_fingerprint_with_env(
        ctx,
        &opts,
        None,
        target_arch,
        materialization,
        inherited_env,
    );
    let mut hash = sha2::Sha256::new();
    for bytes in [
        "standard-codegen-v1".as_bytes(),
        base.as_bytes(),
        detected_device_arch.unwrap_or("").as_bytes(),
    ] {
        update_codegen_fingerprint_hash(&mut hash, bytes);
    }
    finish_codegen_fingerprint(hash)
}

pub(super) fn pipeline_codegen_fingerprint(
    ctx: &Context,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    emit_nvvm_ir: bool,
    target_arch: Option<&str>,
    materialization: &MaterializationMode,
) -> String {
    let base = standard_codegen_fingerprint(
        ctx,
        true,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        None,
        materialization,
    );
    let mut hash = sha2::Sha256::new();
    for value in [
        base.as_str(),
        "CUDA_OXIDE_SHOW_RUSTC_MIR=1",
        "CUDA_OXIDE_DUMP_MIR=1",
        "CUDA_OXIDE_DUMP_LLVM=1",
    ] {
        update_codegen_fingerprint_hash(&mut hash, value.as_bytes());
    }
    finish_codegen_fingerprint(hash)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn interop_codegen_fingerprint(
    ctx: &Context,
    verbose: bool,
    options: InteropDeviceBuildOptions,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    artifact_dir: &Path,
    emit_nvvm_ir: bool,
    device_features: Option<&str>,
    materialization: &MaterializationMode,
) -> String {
    interop_codegen_fingerprint_with_env(
        ctx,
        verbose,
        options,
        target_arch,
        detected_device_arch,
        artifact_dir,
        emit_nvvm_ir,
        device_features,
        materialization,
        &inherited_process_env(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn interop_codegen_fingerprint_with_env(
    ctx: &Context,
    verbose: bool,
    options: InteropDeviceBuildOptions,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    artifact_dir: &Path,
    emit_nvvm_ir: bool,
    device_features: Option<&str>,
    materialization: &MaterializationMode,
    inherited_env: &BTreeMap<String, Vec<u8>>,
) -> String {
    // Canonicalize the sanitizer's implicit line-table default into the same
    // policy that an explicit --lineinfo request uses. Higher-priority CLI,
    // ambient, and project values already participate in the standard key and
    // suppress this default in the child command. Route identity itself is not
    // codegen identity: sanitize+full and build+full must share artifacts.
    let inherited_debug = inherited_env.contains_key("CUDA_OXIDE_DEBUG");
    let project_debug = ctx
        .config
        .env
        .iter()
        .any(|(key, _)| key == "CUDA_OXIDE_DEBUG");
    let effective_device_debug = if options.device_debug == DeviceDebug::Off
        && options.sanitizer_line_tables
        && !inherited_debug
        && !project_debug
    {
        DeviceDebug::LineTables
    } else {
        options.device_debug
    };
    let base = standard_codegen_fingerprint_with_env(
        ctx,
        verbose,
        options.no_fmad,
        options.unchecked_indexing,
        effective_device_debug,
        emit_nvvm_ir,
        target_arch,
        detected_device_arch,
        materialization,
        inherited_env,
    );
    let mut hash = sha2::Sha256::new();
    for bytes in [
        "interop-codegen-v1".as_bytes(),
        base.as_bytes(),
        artifact_dir.as_os_str().as_encoded_bytes(),
        device_features.unwrap_or("").as_bytes(),
    ] {
        update_codegen_fingerprint_hash(&mut hash, bytes);
    }
    finish_codegen_fingerprint(hash)
}

pub(super) fn backend_artifact_digest(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut hasher = Sha256::new();
    if path == Path::new("llvm") {
        hasher.update(b"rustc built-in LLVM backend");
        let digest: [u8; 32] = hasher.finalize().into();
        return Ok(digest_hex(&digest));
    }

    let canonical = path
        .canonicalize()
        .map_err(|error| format!("could not resolve backend {}: {error}", path.display()))?;
    let mut file = std::fs::File::open(&canonical).map_err(|error| {
        format!(
            "could not open backend {} for fingerprinting: {error}",
            canonical.display()
        )
    })?;
    let mut hasher = Sha256::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut chunk).map_err(|error| {
            format!(
                "could not read backend {} for fingerprinting: {error}",
                canonical.display()
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    let digest: [u8; 32] = hasher.finalize().into();
    Ok(digest_hex(&digest))
}
