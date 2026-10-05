/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end check for native-cubin metadata interop artifacts.
//!
//! The device crate under `device/` is declared with
//! `[[package.metadata.cuda-oxide.device-crates]]` using
//! `artifact-kind = "cubin"`, `source-identity = true`, and the hyphenated
//! `bin = "scale-offset-device"` target. `cargo oxide run` builds it through
//! the cuda-oxide backend in NVVM IR mode, finalizes the IR into
//! `device/scale_offset_device.cubin` via libNVVM + nvJitLink, and writes
//! the versioned `.identity` sidecar next to it.
//!
//! This host then verifies the whole contract against reality:
//! 1. the identity sidecar is versioned and its artifact digest matches the
//!    cubin bytes on disk;
//! 2. its recorded target equals the backend-recorded `.target` sidecar,
//!    the authoritative arch record for NVVM IR artifacts;
//! 3. every recorded source resolves relative to the artifact directory,
//!    every recorded source digest matches the file's current bytes, and
//!    the hyphenated bin's own source file is among them (cargo names the
//!    uplifted dep-info after the bin target verbatim);
//! 4. the cubin actually loads and runs: `out[i] = in[i] * scale + offset`.

use cuda_core::{CudaContext, DeviceBuffer, launch_kernel_on_stream};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::Path;

const IDENTITY_VERSION: &str = "cuda-oxide-artifact-identity-v1";
const ELEMENTS: usize = 4096;
// Exact in f32 whether or not the device contracts mul+add into an FMA:
// small-integer inputs times a power of two plus 0.5 never round.
const SCALE: f32 = 2.0;
const OFFSET: f32 = 0.5;

struct Identity {
    target: String,
    device_features: String,
    artifact_sha256: String,
    /// Relative source path -> recorded SHA-256, as written in the sidecar.
    sources: BTreeMap<String, String>,
}

fn parse_identity(text: &str) -> Result<Identity, String> {
    let mut lines = text.lines();
    let version = lines.next().unwrap_or_default();
    if version != IDENTITY_VERSION {
        return Err(format!(
            "identity version line is {version:?}, expected {IDENTITY_VERSION:?}"
        ));
    }
    let mut fields = BTreeMap::new();
    let mut sources = BTreeMap::new();
    for line in lines {
        let mut parts = line.split('\t');
        match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some("source"), Some(path), Some(digest), None) => {
                sources.insert(path.to_string(), digest.to_string());
            }
            (Some(key), Some(value), None, None) => {
                fields.insert(key.to_string(), value.to_string());
            }
            _ => return Err(format!("unrecognized identity line {line:?}")),
        }
    }
    let mut field = |key: &str| {
        fields
            .remove(key)
            .ok_or_else(|| format!("identity is missing the {key} field"))
    };
    Ok(Identity {
        target: field("target")?,
        device_features: field("device_features")?,
        artifact_sha256: field("artifact_sha256")?,
        sources,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn verify_identity(device_dir: &Path, cubin: &[u8]) -> Result<(), String> {
    let identity_path = device_dir.join("scale_offset_device.cubin.identity");
    let identity = std::fs::read_to_string(&identity_path)
        .map_err(|error| format!("could not read {}: {error}", identity_path.display()))?;
    let identity = parse_identity(&identity)?;

    // 1. The recorded artifact digest must match the cubin we are about to
    //    load, byte for byte.
    let cubin_digest = sha256_hex(cubin);
    if identity.artifact_sha256 != cubin_digest {
        return Err(format!(
            "identity artifact_sha256 {} does not match the cubin on disk {}",
            identity.artifact_sha256, cubin_digest
        ));
    }

    // 2. The recorded target must be the backend's own record (the .target
    //    sidecar written next to the NVVM IR), not a request hint.
    let target_sidecar = device_dir.join("scale_offset_device.target");
    let recorded = std::fs::read_to_string(&target_sidecar)
        .map_err(|error| format!("could not read {}: {error}", target_sidecar.display()))?;
    let recorded = recorded.lines().next().unwrap_or_default().trim();
    if identity.target != recorded {
        return Err(format!(
            "identity target {:?} does not match the backend record {recorded:?}",
            identity.target
        ));
    }
    if !identity.target.starts_with("sm_") {
        return Err(format!(
            "identity target {:?} is not a concrete sm_XX architecture",
            identity.target
        ));
    }

    // 3. No --device-features were passed, and the sidecar must say so
    //    canonically.
    if identity.device_features != "<none>" {
        return Err(format!(
            "identity device_features {:?}, expected \"<none>\"",
            identity.device_features
        ));
    }

    // 4. Every recorded source must resolve relative to the artifact dir and
    //    still hash to its recorded digest; the hyphenated bin's source and
    //    the device manifest must be among them.
    if identity.sources.is_empty() {
        return Err("identity records no sources".to_string());
    }
    for (path, digest) in &identity.sources {
        let resolved = device_dir.join(path);
        let bytes = std::fs::read(&resolved).map_err(|error| {
            format!(
                "identity source {path:?} does not resolve against the artifact dir ({}): {error}",
                resolved.display()
            )
        })?;
        let actual = sha256_hex(&bytes);
        if &actual != digest {
            return Err(format!(
                "identity source {path:?} hashes to {actual}, but {digest} was recorded"
            ));
        }
    }
    for required in ["src/bin/scale_offset_device.rs", "Cargo.toml"] {
        if !identity.sources.contains_key(required) {
            return Err(format!("identity does not record the {required} source"));
        }
    }

    println!(
        "identity verified: target {}, {} sources, artifact sha256 {}",
        identity.target,
        identity.sources.len(),
        &identity.artifact_sha256[..12]
    );
    Ok(())
}

fn run_kernel(cubin_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = context
        .load_module_from_file(cubin_path.to_str().ok_or("cubin path is not valid UTF-8")?)?;
    let function = module.load_function("scale_offset_f32")?;

    let input: Vec<f32> = (0..ELEMENTS).map(|i| i as f32).collect();
    let input_buffer = DeviceBuffer::from_host(&stream, &input)?;
    let output_buffer = DeviceBuffer::<f32>::zeroed(&stream, ELEMENTS)?;

    let mut n = u32::try_from(ELEMENTS)?;
    let mut scale = SCALE;
    let mut offset = OFFSET;
    let mut input_ptr = input_buffer.cu_deviceptr();
    let mut output_ptr = output_buffer.cu_deviceptr();
    let grid = n.div_ceil(256);
    // SAFETY: the argument list mirrors scale_offset_f32's signature
    // (u32, f32, f32, *const f32, *mut f32); both buffers stay alive across
    // the launch and the stream belongs to the module's context.
    unsafe {
        launch_kernel_on_stream(
            &function,
            (grid, 1, 1),
            (256, 1, 1),
            0,
            &stream,
            &mut [
                (&raw mut n).cast::<c_void>(),
                (&raw mut scale).cast::<c_void>(),
                (&raw mut offset).cast::<c_void>(),
                (&raw mut input_ptr).cast::<c_void>(),
                (&raw mut output_ptr).cast::<c_void>(),
            ],
        )?;
    }

    let output = output_buffer.to_host_vec(&stream)?;
    for (i, (&got, &fed)) in output.iter().zip(&input).enumerate() {
        let expected = fed * SCALE + OFFSET;
        if got != expected {
            return Err(format!("output[{i}] = {got}, expected {expected}").into());
        }
    }
    println!("kernel verified: {ELEMENTS} elements of in * {SCALE} + {OFFSET}");
    Ok(())
}

fn main() {
    let device_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/device"));
    let cubin_path = device_dir.join("scale_offset_device.cubin");
    let cubin = std::fs::read(&cubin_path).unwrap_or_else(|error| {
        panic!(
            "could not read {} (build with `cargo oxide run interop_cubin_identity`): {error}",
            cubin_path.display()
        )
    });

    verify_identity(device_dir, &cubin).unwrap_or_else(|error| {
        panic!("artifact identity mismatch: {error}");
    });
    run_kernel(&cubin_path).unwrap_or_else(|error| {
        panic!("cubin execution failed: {error}");
    });

    println!("SUCCESS: cubin interop artifact verified and executed");
}
