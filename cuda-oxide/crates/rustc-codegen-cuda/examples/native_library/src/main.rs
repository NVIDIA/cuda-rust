// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::embedded::{
    ArtifactDebugPolicy, ArtifactPayloadKind, artifact_bundles_from_current_exe,
};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig3D};
use cuda_device::{kernel, launch_bounds, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 3, dynamic_shared = inferred)]
    pub unsafe fn one(input: *const i32, output: *mut i32) {
        unsafe {
            native_library::load_add::<1>(input, output);
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 3, dynamic_shared = inferred)]
    pub unsafe fn four(input: *const i32, output: *mut i32) {
        unsafe {
            native_library::load_add::<4>(input, output);
        }
    }
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 3, block = (128, 1, 1), dynamic_shared = inferred)]
    pub unsafe fn payloads(output: *mut i32) {
        unsafe {
            native_library::payload_roundtrip(output);
        }
    }
}

fn verify() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the embedded image was produced from this module's declarations.
    let module = unsafe { kernels::load(&context)? };
    for items in [1, 4] {
        let input: Vec<i32> = (0..3 * 128 * items).map(|i| i * 3 - 11).collect();
        let device_input = DeviceBuffer::from_host(&stream, &input)?;
        let device_output = DeviceBuffer::<i32>::zeroed(&stream, input.len())?;
        let config = LaunchConfig3D::auto_shared((3, 1, 1), (128, 1, 1));
        macro_rules! launch {
            ($prepare:ident, $kernel:ident) => {{
                let prepared = module.$prepare(config)?;
                let facts = prepared.function().launch_facts().unwrap();
                assert_eq!(facts.exact_block, Some((128, 1, 1)));
                assert_eq!(facts.dynamic_shared_bytes, 0);
                assert!(
                    module
                        .$prepare(LaunchConfig3D::auto_shared((3, 1, 1), (64, 2, 1)))
                        .is_err()
                );
                // SAFETY: complete, immutable input and disjoint output tiles
                // outlive execution. Preparation enforces exact geometry.
                unsafe {
                    module.$kernel(
                        &stream,
                        &prepared,
                        device_input.cu_deviceptr() as *const i32,
                        device_output.cu_deviceptr() as *mut i32,
                    )?;
                }
            }};
        }
        if items == 1 {
            launch!(prepare_one, one);
        } else {
            launch!(prepare_four, four);
        }
        assert_eq!(
            device_output.to_host_vec(&stream)?,
            input.iter().map(|value| value + 7).collect::<Vec<_>>()
        );
        println!("PASS native library: items={items}, {} values", input.len());
    }
    let output = DeviceBuffer::<i32>::zeroed(&stream, 3 * 128)?;
    let prepared = module.prepare_payloads(LaunchConfig3D::auto_shared((3, 1, 1), (128, 1, 1)))?;
    // SAFETY: the output covers three complete, disjoint blocks and outlives
    // the launch; preparation checks the exact root geometry and scratch facts.
    unsafe {
        module.payloads(&stream, &prepared, output.cu_deviceptr() as *mut i32)?;
    }
    assert_eq!(
        output.to_host_vec(&stream)?,
        (0..3 * 128)
            .map(|i| i + 1 + (i % 128) / 8)
            .collect::<Vec<i32>>()
    );
    println!("PASS checked payloads: empty Block controls, W8 seed, 384 values");
    Ok(())
}

// Artifact-only smoke lanes call this before creating any CUDA context. Read
// the executable's actual bundle, including its launch facts and debug policy.
fn check_artifact(
    target: &str,
    output: &std::path::Path,
    full_debug: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let bundles = artifact_bundles_from_current_exe()?;
    let [bundle] = bundles.as_slice() else {
        return Err("expected exactly one embedded device bundle".into());
    };
    assert_eq!(bundle.target, target);
    for symbol in ["one", "four", "payloads"] {
        let facts = bundle.entry(symbol).unwrap().launch_facts.unwrap();
        assert_eq!(facts.exact_block, Some((128, 1, 1)));
        assert_eq!(facts.dynamic_shared_bytes, 0);
    }
    if full_debug {
        assert_eq!(
            bundle.compile_options.debug_policy(),
            ArtifactDebugPolicy::Full
        );
    }
    let cubin = bundle
        .payload(ArtifactPayloadKind::Cubin)
        .ok_or("missing embedded CUBIN")?;
    assert!(cubin.starts_with(b"\x7fELF"));
    std::fs::write(output, cubin)?;
    println!("PASS embedded CUBIN: target={target}, kernels=one,four,payloads");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "--check-artifact") {
        if !(args.len() == 3 || (args.len() == 4 && args[3] == "--full-debug")) {
            return Err("expected --check-artifact TARGET OUTPUT [--full-debug]".into());
        }
        return check_artifact(
            args[1].to_str().ok_or("invalid target")?,
            std::path::Path::new(&args[2]),
            args.len() == 4,
        );
    }
    if !args.is_empty() {
        return Err("unexpected fixture argument".into());
    }
    verify()
}

#[test]
fn native_provider_executes() {
    verify().unwrap();
}
