// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Materialize embedded NVVM IR, record its provenance, and execute the cubin.
//!
//! The target must use the legacy typed-pointer NVVM IR dialect and be
//! supported by CUDA device 0. For example:
//!
//! ```text
//! CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 cargo run -p cuda-host \
//!   --example finalize_nvvm_ir -- /tmp/example-sm86.cubin sm_86
//! ```

use cuda_artifact_finalizer::{
    CudaArch, FinalizationOptions, Finalizer, FinalizerOutput, recipe_digest,
};
use cuda_core::{CudaContext, DeviceBuffer, launch_kernel_on_stream};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::path::PathBuf;

const USAGE: &str = "usage: finalize_nvvm_ir OUTPUT.cubin [sm_XX]";

const EXAMPLE_NVVM_IR: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (i32*)* @kernel to i8*)], section "llvm.metadata"

define void @kernel(i32* %out) {
entry:
  store i32 7, i32* %out, align 4
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void (i32*)* @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let output_path = PathBuf::from(args.next().ok_or(USAGE)?);
    let target: CudaArch = args
        .next()
        .unwrap_or_else(|| "sm_86".into())
        .into_string()
        .map_err(|_| "architecture must be UTF-8")?
        .parse()?;
    if args.next().is_some() {
        return Err(USAGE.into());
    }
    if !target.uses_legacy_llvm() {
        return Err(
            format!("{target} requires opaque-pointer NVVM IR; use a target below sm_100").into(),
        );
    }

    let options = FinalizationOptions::new(target.clone());
    let finalizer = Finalizer::discover()?;
    let provenance = finalizer.provenance();
    let cubin = finalizer.materialize_nvvm_ir("example.ll", EXAMPLE_NVVM_IR, &options)?;
    std::fs::write(&output_path, &cubin)?;

    let e_ident_version = cubin[6];
    let abi_version = cubin[8];
    let e_version = u32::from_le_bytes(cubin[20..24].try_into()?);
    let artifact_sha256: [u8; 32] = Sha256::digest(&cubin).into();
    let plan_sha256 = finalizer
        .nvvm_ir_artifact_digest(
            "example.ll",
            "example.ll.ltoir",
            EXAMPLE_NVVM_IR,
            &options,
            FinalizerOutput::Cubin,
        )
        .ok_or("exact finalizer provenance is unavailable")?;

    println!("output={}", output_path.display());
    println!("target={target}");
    println!("bytes={}", cubin.len());
    println!("elf_e_ident_version=0x{e_ident_version:x}");
    println!("elf_abi_version={abi_version}");
    println!("elf_e_version=0x{e_version:x}");
    println!("cubin_sha256={}", hex(artifact_sha256));
    println!("recipe_sha256={}", hex(recipe_digest()));
    println!("plan_sha256={}", hex(plan_sha256));
    println!(
        "libnvvm_sha256={}",
        hex(provenance
            .libnvvm_sha256
            .ok_or("exact libNVVM provenance is unavailable")?)
    );
    println!(
        "nvjitlink_sha256={}",
        hex(provenance
            .nvjitlink_sha256
            .ok_or("exact nvJitLink provenance is unavailable")?)
    );
    println!("libdevice_sha256={}", hex(provenance.libdevice_sha256));

    let context = CudaContext::new(0)?;
    let module = context.load_module_from_image(&cubin)?;
    let kernel = module.load_function("kernel")?;
    let stream = context.default_stream();
    let device_output = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let mut output_pointer = device_output.cu_deviceptr();
    let mut parameters = [(&mut output_pointer as *mut _) as *mut std::ffi::c_void];
    // SAFETY: `kernel` has one pointer parameter, both launch dimensions are
    // one, and `device_output` remains alive through the synchronized copy.
    unsafe { launch_kernel_on_stream(&kernel, (1, 1, 1), (1, 1, 1), 0, &stream, &mut parameters)? };
    let observed = device_output.to_host_vec(&stream)?;
    if observed != [7] {
        return Err(format!("kernel output mismatch: {observed:?}").into());
    }
    println!("cuda_driver_execution=ok (device=0, function=kernel, output=7)");
    Ok(())
}
