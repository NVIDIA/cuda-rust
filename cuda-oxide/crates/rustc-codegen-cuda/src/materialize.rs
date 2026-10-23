/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Strict, opt-in build-time finalization of embedded device artifacts.
//!
//! `cargo oxide --materialize-cubin` discovers and fingerprints the exact
//! libNVVM, nvJitLink, and libdevice inputs before invoking Cargo. Device
//! macros record the complete codegen identity and exact provenance as Cargo
//! environment dependencies. We reopen the tools, bind the parent digest to
//! matching retained-file identities, and revalidate those identities around
//! compilation. Setting the internal opt-in around raw Cargo is unsupported:
//! Cargo can reuse an existing artifact without invoking this backend, and
//! when the backend does run it rejects a missing handshake.

use cuda_artifact_finalizer::{
    CudaArch, CudaArchParseError, DebugPolicy, FinalizationOptions, Finalizer, FinalizerError,
    FinalizerOutput, KernelResourceUsage, NamedInput,
};
use thiserror::Error;

pub(crate) const MATERIALIZE_ENV: &str = reserved_oxide_symbols::MATERIALIZE_CUBIN_ENV;
pub(crate) const EXPECTED_PROVENANCE_ENV: &str =
    reserved_oxide_symbols::MATERIALIZER_PROVENANCE_ENV;
pub(crate) const MATERIALIZER_HANDSHAKE_ENV: &str =
    reserved_oxide_symbols::MATERIALIZER_HANDSHAKE_ENV;
pub(crate) const CODEGEN_FINGERPRINT_ENV: &str = reserved_oxide_symbols::CODEGEN_FINGERPRINT_ENV;

#[derive(Debug)]
pub(crate) struct MaterializationRequest {
    expected_provenance: [u8; 32],
    tool_identity_handshake: cuda_artifact_finalizer::MaterializerHandshakeV1,
    external_inputs: Vec<(String, Vec<u8>)>,
    library_inputs: Vec<(String, Vec<u8>)>,
}

impl MaterializationRequest {
    pub(crate) fn add_library_inputs(&mut self, inputs: Vec<(String, Vec<u8>)>) {
        self.library_inputs.extend(inputs);
    }
}

/// Cubin bytes plus ptxas resource diagnostics from the final link.
pub(crate) struct MaterializedCubin {
    pub(crate) bytes: Vec<u8>,
    pub(crate) resource_usage: Vec<KernelResourceUsage>,
}

/// Failures in the wrapper/backend materialization contract.
#[derive(Debug, Error)]
pub(crate) enum MaterializeError {
    #[error(
        "{MATERIALIZE_ENV} must be a boolean (accepted true values: 1, true, yes, on; false values: 0, false, no, off), got {value:?}"
    )]
    InvalidBoolean { value: String },

    #[error("{MATERIALIZE_ENV} is not valid Unicode")]
    NonUnicodeBoolean,

    #[error(
        "{MATERIALIZE_ENV}=true requires cargo-oxide's provenance handshake; use `cargo oxide build --materialize-cubin` instead of invoking raw Cargo"
    )]
    MissingExpectedProvenance,

    #[error(
        "{MATERIALIZE_ENV}=true is missing cargo-oxide's tracked codegen fingerprint; use `cargo oxide build --materialize-cubin` instead of invoking raw Cargo"
    )]
    MissingCargoFingerprint,

    #[error(
        "{MATERIALIZE_ENV}=true requires cargo-oxide's named v1 tool-identity handshake in {MATERIALIZER_HANDSHAKE_ENV}"
    )]
    MissingToolIdentityHandshake,

    #[error("{MATERIALIZER_HANDSHAKE_ENV} is not valid Unicode")]
    NonUnicodeToolIdentityHandshake,

    #[error("{MATERIALIZER_HANDSHAKE_ENV} is not a valid named v1 handshake: {reason}")]
    InvalidToolIdentityHandshake { reason: String },

    #[error(
        "cargo-oxide's tracked codegen fingerprint must be exactly 64 lowercase hexadecimal characters, got {value:?}"
    )]
    InvalidCargoFingerprint { value: String },

    #[error(
        "{EXPECTED_PROVENANCE_ENV} must be exactly 64 lowercase hexadecimal characters, got {value:?}"
    )]
    InvalidExpectedProvenance { value: String },

    #[error(
        "the loaded CUDA tools cannot be tied to exact files, so their provenance cannot be verified; refusing build-time cubin materialization"
    )]
    UnverifiableProvenance,

    #[error(
        "CUDA materializer provenance changed after Cargo fingerprinting (expected {expected}, loaded {actual}); rerun `cargo oxide build --materialize-cubin`"
    )]
    ProvenanceMismatch { expected: String, actual: String },

    #[error(
        "build-time cubin materialization of #[device] extern declarations requires their definitions; supply ordered --link-ltoir inputs"
    )]
    HasDeviceExterns,

    #[error("invalid cargo-oxide external LTO-IR input list: {reason}")]
    InvalidExternalInputs { reason: String },

    #[error(
        "build-time cubin materialization requires an NVVM IR or LTOIR artifact, but codegen produced PTX; use cargo-oxide so materialization can force NVVM IR emission"
    )]
    PtxInput,

    #[error(
        "build-time cubin materialization expected compiler IR, but codegen already produced a cubin; refusing to bypass the provenance-checked finalization recipe"
    )]
    CubinInput,

    #[error(transparent)]
    InvalidTarget(#[from] CudaArchParseError),

    #[error(transparent)]
    Finalizer(#[from] FinalizerError),
}

/// Parse the strict opt-in and its wrapper-generated provenance handshake.
/// No CUDA library is loaded here.
pub(crate) fn request_from_env() -> Result<Option<MaterializationRequest>, MaterializeError> {
    let enabled = match std::env::var(MATERIALIZE_ENV) {
        Ok(value) => parse_bool(&value)?,
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(MaterializeError::NonUnicodeBoolean);
        }
    };
    if !enabled {
        return Ok(None);
    }
    let value = std::env::var(EXPECTED_PROVENANCE_ENV)
        .map_err(|_| MaterializeError::MissingExpectedProvenance)?;
    let expected_provenance = parse_digest(&value)?;
    let handshake_json =
        std::env::var(MATERIALIZER_HANDSHAKE_ENV).map_err(|error| match error {
            std::env::VarError::NotPresent => MaterializeError::MissingToolIdentityHandshake,
            std::env::VarError::NotUnicode(_) => MaterializeError::NonUnicodeToolIdentityHandshake,
        })?;
    let handshake: cuda_artifact_finalizer::MaterializerHandshakeV1 =
        serde_json::from_str(&handshake_json).map_err(|error| {
            MaterializeError::InvalidToolIdentityHandshake {
                reason: error.to_string(),
            }
        })?;
    validate_tool_identity_handshake(expected_provenance, &handshake)?;
    validate_codegen_fingerprint()?;
    let external_inputs = external_inputs_from_env()?;
    Ok(Some(MaterializationRequest {
        expected_provenance,
        tool_identity_handshake: handshake,
        external_inputs,
        library_inputs: Vec::new(),
    }))
}

fn external_inputs_from_env() -> Result<Vec<(String, Vec<u8>)>, MaterializeError> {
    let value = match std::env::var(reserved_oxide_symbols::EXTERNAL_LTOIR_ENV) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(error) => {
            return Err(MaterializeError::InvalidExternalInputs {
                reason: error.to_string(),
            });
        }
    };
    let files: Vec<cuda_artifact_finalizer::LtoirFile> =
        serde_json::from_str(&value).map_err(|error| MaterializeError::InvalidExternalInputs {
            reason: error.to_string(),
        })?;
    files
        .into_iter()
        .map(|file| {
            let bytes = file.read_verified()?;
            Ok((file.name()?.to_owned(), bytes))
        })
        .collect()
}

fn validate_tool_identity_handshake(
    expected_provenance: [u8; 32],
    handshake: &cuda_artifact_finalizer::MaterializerHandshakeV1,
) -> Result<(), MaterializeError> {
    if !handshake.has_consistent_provenance() || handshake.provenance_sha256 != expected_provenance
    {
        return Err(MaterializeError::InvalidToolIdentityHandshake {
            reason: format!(
                "version {} and provenance {} do not match expected v1 provenance {}",
                handshake.version,
                digest_hex(&handshake.provenance_sha256),
                digest_hex(&expected_provenance),
            ),
        });
    }
    Ok(())
}

/// Reject artifact-loading models the finalizer cannot reproduce, before any
/// CUDA compiler library is discovered or loaded.
pub(crate) fn validate_collection(
    request: Option<&MaterializationRequest>,
    has_device_externs: bool,
) -> Result<(), MaterializeError> {
    let Some(request) = request else {
        return Ok(());
    };
    if has_device_externs && request.external_inputs.is_empty() && request.library_inputs.is_empty()
    {
        return Err(MaterializeError::HasDeviceExterns);
    }
    Ok(())
}

/// Use the actual rustc crate name and configured output directory, including
/// for Cargo passthrough builds whose working directory differs from the CLI's.
pub(crate) fn validate_output_paths(
    request: Option<&MaterializationRequest>,
    output_dir: &std::path::Path,
    output_name: &str,
) -> Result<(), MaterializeError> {
    let Some(request) = request else {
        return Ok(());
    };
    let mut outputs =
        mir_importer::pipeline::stale_compilation_artifact_paths(output_dir, output_name);
    outputs.push(output_dir.join(format!(
        "{output_name}.{}",
        oxide_tile_export::DESCRIPTIONS_SUFFIX
    )));
    for (input, _) in &request.external_inputs {
        for output in &outputs {
            cuda_artifact_finalizer::check_ltoir_input_output(std::path::Path::new(input), output)?;
        }
    }
    Ok(())
}

pub(crate) fn nvvm_ir_to_cubin(
    request: MaterializationRequest,
    nvvm_ir: &[u8],
    module_name: &str,
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<MaterializedCubin, MaterializeError> {
    let options = options(target, allow_fma_contraction, debug_policy)?;
    let finalizer = checked_finalizer(&request)?;
    let ltoir = finalizer
        .compiler()
        .compile_nvvm_ir_to_ltoir(module_name, nvvm_ir, &options)?;
    let name = format!("{module_name}.ltoir");
    let inputs = link_inputs(&request, &name, &ltoir);
    let report = finalizer.link_ltoir_with_report(&inputs, &options, FinalizerOutput::Cubin)?;
    Ok(MaterializedCubin {
        bytes: report.image,
        resource_usage: report.resource_usage,
    })
}

pub(crate) fn ltoir_to_cubin(
    request: MaterializationRequest,
    ltoir: &[u8],
    module_name: &str,
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<MaterializedCubin, MaterializeError> {
    let options = options(target, allow_fma_contraction, debug_policy)?;
    let finalizer = checked_finalizer(&request)?;
    let inputs = link_inputs(&request, module_name, ltoir);
    let report = finalizer.link_ltoir_with_report(&inputs, &options, FinalizerOutput::Cubin)?;
    Ok(MaterializedCubin {
        bytes: report.image,
        resource_usage: report.resource_usage,
    })
}

fn link_inputs<'a>(
    request: &'a MaterializationRequest,
    name: &'a str,
    bytes: &'a [u8],
) -> Vec<NamedInput<'a>> {
    std::iter::once(NamedInput::new(name, bytes))
        .chain(
            request
                .external_inputs
                .iter()
                .chain(&request.library_inputs)
                .map(|(name, bytes)| NamedInput::new(name, bytes)),
        )
        .collect()
}

fn options(
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<FinalizationOptions, MaterializeError> {
    let target: CudaArch = target.parse()?;
    Ok(FinalizationOptions::new(target)
        .with_fma_contraction(allow_fma_contraction)
        .with_debug_policy(debug_policy))
}

fn checked_finalizer(request: &MaterializationRequest) -> Result<Finalizer, MaterializeError> {
    let finalizer = Finalizer::discover_with_handshake(&request.tool_identity_handshake)?;
    let actual = finalizer
        .provenance_digest()
        .ok_or(MaterializeError::UnverifiableProvenance)?;
    if actual != request.expected_provenance {
        return Err(MaterializeError::ProvenanceMismatch {
            expected: digest_hex(&request.expected_provenance),
            actual: digest_hex(&actual),
        });
    }
    Ok(finalizer)
}

fn parse_bool(value: &str) -> Result<bool, MaterializeError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(MaterializeError::InvalidBoolean {
            value: value.to_string(),
        }),
    }
}

fn validate_codegen_fingerprint() -> Result<(), MaterializeError> {
    let value = std::env::var(CODEGEN_FINGERPRINT_ENV).ok();
    validate_codegen_fingerprint_value(value.as_deref())
}

fn validate_codegen_fingerprint_value(value: Option<&str>) -> Result<(), MaterializeError> {
    let value = value.ok_or(MaterializeError::MissingCargoFingerprint)?;
    parse_digest(value)
        .map(|_| ())
        .map_err(|_| MaterializeError::InvalidCargoFingerprint {
            value: value.to_string(),
        })
}

fn parse_digest(value: &str) -> Result<[u8; 32], MaterializeError> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(MaterializeError::InvalidExpectedProvenance {
            value: value.to_string(),
        });
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16).map_err(|_| {
            MaterializeError::InvalidExpectedProvenance {
                value: value.to_string(),
            }
        })?;
    }
    Ok(digest)
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_tool_file_identity() -> cuda_artifact_finalizer::ToolFileIdentity {
        cuda_artifact_finalizer::ToolFileIdentity {
            length: 0,
            modified_seconds: 0,
            modified_nanoseconds: 0,
            device: None,
            inode: None,
            change_time_seconds: None,
            change_time_nanoseconds: None,
        }
    }

    fn test_handshake() -> cuda_artifact_finalizer::MaterializerHandshakeV1 {
        let file = empty_tool_file_identity();
        cuda_artifact_finalizer::MaterializerHandshakeV1::new(
            cuda_artifact_finalizer::PinnedToolProvenance {
                sha256: [1; 32],
                file,
            },
            cuda_artifact_finalizer::PinnedToolProvenance {
                sha256: [2; 32],
                file,
            },
            [3; 32],
        )
    }

    #[test]
    fn strict_boolean_parser_accepts_only_documented_values() {
        for value in ["1", " true ", "YES", "on"] {
            assert!(parse_bool(value).unwrap());
        }
        for value in ["0", " false ", "NO", "off"] {
            assert!(!parse_bool(value).unwrap());
        }
        for value in ["", "enabled", "2", "truthy"] {
            assert!(matches!(
                parse_bool(value),
                Err(MaterializeError::InvalidBoolean { .. })
            ));
        }
    }

    #[test]
    fn provenance_digest_requires_canonical_lower_hex() {
        let value = "0123456789abcdef".repeat(4);
        let digest = parse_digest(&value).unwrap();
        assert_eq!(digest_hex(&digest), value);
        assert!(parse_digest(&"A".repeat(64)).is_err());
        assert!(parse_digest(&"0".repeat(63)).is_err());
        assert!(parse_digest(&format!("{}g", "0".repeat(63))).is_err());
    }

    #[test]
    fn tool_identity_handshake_fails_closed_on_version_or_provenance_mismatch() {
        let mut handshake = test_handshake();
        let expected = handshake.provenance_sha256;
        assert!(validate_tool_identity_handshake(expected, &handshake).is_ok());

        handshake.version += 1;
        assert!(matches!(
            validate_tool_identity_handshake(expected, &handshake),
            Err(MaterializeError::InvalidToolIdentityHandshake { .. })
        ));
        handshake.version = cuda_artifact_finalizer::MaterializerHandshakeV1::VERSION;
        assert!(matches!(
            validate_tool_identity_handshake([8; 32], &handshake),
            Err(MaterializeError::InvalidToolIdentityHandshake { .. })
        ));
    }

    #[test]
    fn output_overlap_uses_the_actual_crate_and_output_directory() {
        let root = std::env::temp_dir().join(format!(
            "oxide-materialize-overlap-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("custom_target.ll");
        std::fs::write(&input, b"external input").unwrap();
        let handshake = test_handshake();
        let request = MaterializationRequest {
            expected_provenance: handshake.provenance_sha256,
            tool_identity_handshake: handshake,
            library_inputs: Vec::new(),
            external_inputs: vec![(
                input.to_str().unwrap().to_owned(),
                b"external input".to_vec(),
            )],
        };
        assert!(validate_output_paths(Some(&request), &root, "package_name").is_ok());
        assert!(matches!(
            validate_output_paths(Some(&request), &root, "custom_target"),
            Err(MaterializeError::Finalizer(
                FinalizerError::LtoirInputOutputOverlap { .. }
            ))
        ));
        assert_eq!(std::fs::read(&input).unwrap(), b"external input");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsupported_collection_models_fail_without_tools() {
        let handshake = test_handshake();
        let mut request = MaterializationRequest {
            expected_provenance: handshake.provenance_sha256,
            tool_identity_handshake: handshake,
            library_inputs: Vec::new(),
            external_inputs: Vec::new(),
        };
        assert!(validate_collection(Some(&request), false).is_ok());
        assert!(matches!(
            validate_collection(Some(&request), true),
            Err(MaterializeError::HasDeviceExterns)
        ));
        assert!(validate_collection(None, true).is_ok());
        request
            .external_inputs
            .push(("definition.ltoir".into(), vec![1]));
        assert!(validate_collection(Some(&request), true).is_ok());
    }

    #[test]
    fn backend_invocation_rejects_raw_cargo_materialization_without_fingerprint() {
        assert!(matches!(
            validate_codegen_fingerprint_value(None),
            Err(MaterializeError::MissingCargoFingerprint)
        ));
        assert!(validate_codegen_fingerprint_value(Some(&"00".repeat(32))).is_ok());
        assert!(matches!(
            validate_codegen_fingerprint_value(Some("not-a-digest")),
            Err(MaterializeError::InvalidCargoFingerprint { .. })
        ));
    }
}
