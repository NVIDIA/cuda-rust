/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end LLVM-IR coverage for rustc-proven kernel reference validity.
//!
//! Build with `cargo oxide build kernel_reference_validity`, then run the host
//! binary (or `cargo oxide run kernel_reference_validity`) to inspect the
//! generated `.ll` parameter lines and execute valid slice-launch controls.

use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread};

#[repr(align(16))]
#[derive(Clone, Copy)]
pub struct AlignedZst;

// SAFETY: this zero-sized type has no resources or pointers to host memory.
unsafe impl cuda_core::DeviceCopy for AlignedZst {}

#[derive(Clone, Copy)]
pub struct ByValue {
    pub pointer: *const f32,
}

#[kernel]
pub fn shared_ref(_value: &f32) {}

#[kernel]
pub fn unique_ref(_value: &mut f32) {}

#[kernel]
pub fn shared_slice(_value: &[f32]) {}

#[kernel]
pub fn unique_slice(_value: &mut [f32]) {}

#[kernel]
pub fn align_one(_value: &u8) {}

#[kernel]
pub fn aligned_zst(_value: &AlignedZst) {}

#[kernel]
pub fn raw_pointer(_value: *const f32) {}

#[kernel]
pub fn disjoint_slice(_value: DisjointSlice<f32>) {}

/// Keeps a libdevice call and a bare `DisjointSlice` control in the module.
/// Explicit NVVM verification also checks this module: an ordinary build may
/// resolve libdevice through LLVM linking instead.
#[kernel]
pub fn nvvm_slice(input: &[f32], mut output: DisjointSlice<f32>) {
    let index = thread::index_1d();
    let i = index.get();
    if let Some(out) = output.get_mut(index) {
        *out = input[i].exp();
    }
}

#[kernel]
pub fn by_value(_value: ByValue) {}

#[allow(improper_ctypes_definitions)]
#[kernel]
pub extern "C" fn c_reference(_value: &f32) {}

fn kernel_header<'a>(llvm_ir: &'a str, name: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let needle = format!("@{name}(");
    llvm_ir
        .lines()
        .find(|line| line.trim_start().starts_with("define ") && line.contains(&needle))
        .ok_or_else(|| format!("missing LLVM kernel definition `{name}`").into())
}

fn require_reference_attrs(
    llvm_ir: &str,
    name: &str,
    alignment: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    let expected = format!("nonnull align {alignment}");
    if !header.contains(&expected) {
        return Err(format!(
            "kernel `{name}` is missing `{expected}` on its reference parameter:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn require_nonnull_without_alignment(
    llvm_ir: &str,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    if !header.contains("nonnull") || header.contains(" align ") {
        return Err(format!(
            "kernel `{name}` must carry nonnull without a redundant align-1 attribute:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn require_single_slice_pointer_fact(
    llvm_ir: &str,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    if header.matches("nonnull").count() != 1 || header.matches("align 4").count() != 1 {
        return Err(format!(
            "kernel `{name}` must annotate only the slice data pointer:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn require_bare(llvm_ir: &str, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    if header.contains("nonnull") || header.contains(" align ") {
        return Err(format!(
            "kernel `{name}` unexpectedly acquired Rust-reference validity:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn verify_generated_llvm_ir() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernel_reference_validity.ll");
    let llvm_ir = std::fs::read_to_string(&path)?;

    require_reference_attrs(&llvm_ir, "shared_ref", 4)?;
    require_reference_attrs(&llvm_ir, "unique_ref", 4)?;
    require_single_slice_pointer_fact(&llvm_ir, "shared_slice")?;
    require_single_slice_pointer_fact(&llvm_ir, "unique_slice")?;
    require_single_slice_pointer_fact(&llvm_ir, "nvvm_slice")?;
    require_nonnull_without_alignment(&llvm_ir, "align_one")?;
    require_reference_attrs(&llvm_ir, "aligned_zst", 16)?;

    require_bare(&llvm_ir, "raw_pointer")?;
    require_bare(&llvm_ir, "disjoint_slice")?;
    require_bare(&llvm_ir, "by_value")?;
    require_bare(&llvm_ir, "c_reference")?;

    Ok(())
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct AlignedValue {
    values: [u32; 8],
}

// SAFETY: this C-layout value contains only integers, with no padding or
// resources requiring destruction. Host and device use the same layout.
unsafe impl cuda_core::DeviceCopy for AlignedValue {}

#[cuda_module]
mod execution {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn inspect_values(input: &[AlignedValue], mut output: DisjointSlice<[u64; 3]>) {
        if let Some(out) = output.get_mut(thread::index_1d()) {
            let first = input.first().map_or(0, |value| value.values[0]);
            *out = [input.as_ptr() as u64, input.len() as u64, u64::from(first)];
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn inspect_zsts(input: &[AlignedZst], mut output: DisjointSlice<[u64; 2]>) {
        if let Some(out) = output.get_mut(thread::index_1d()) {
            *out = [input.as_ptr() as u64, input.len() as u64];
        }
    }
}

fn verify_launch_packets() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    use cuda_host::cuda_async::simt::{
        device_context::{init_device_contexts, with_cuda_context},
        device_operation::DeviceOperation,
    };

    // All allocations and both launch styles share one CUDA context.
    init_device_contexts(0, 1)?;
    // SAFETY: the embedded module and generated launch ABI come from this crate.
    let module = unsafe { execution::load_async(0)? };
    let stream = with_cuda_context(0, |ctx| ctx.default_stream())?;
    let prepared = module.prepare_inspect_values(LaunchConfig1D::new(1, 1, 0))?;
    let prepared_zst = module.prepare_inspect_zsts(LaunchConfig1D::new(1, 1, 0))?;

    for values in [vec![], vec![AlignedValue { values: [37; 8] }; 2]] {
        let input = DeviceBuffer::from_host(&stream, &values)?;
        let mut output = DeviceBuffer::<[u64; 3]>::zeroed(&stream, 1)?;
        let expected_ptr = if values.is_empty() {
            std::mem::align_of::<AlignedValue>() as u64
        } else {
            input.cu_deviceptr()
        };
        let expected = [[
            expected_ptr,
            values.len() as u64,
            if values.is_empty() { 0 } else { 37 },
        ]];

        module.inspect_values(&stream, &prepared, &input, &mut output)?;
        assert_eq!(output.to_host_vec(&stream)?, expected, "sync slice packet");

        module
            .inspect_values_async(&prepared, &input, &mut output)
            .sync()?;
        assert_eq!(
            output.to_host_vec(&stream)?,
            expected,
            "borrowed async slice packet"
        );

        let (_input, output) = module
            .inspect_values_async_owned(&prepared, input, output)
            .sync()?;
        assert_eq!(
            output.to_host_vec(&stream)?,
            expected,
            "owned async slice packet"
        );
    }

    // A nonempty ZST slice has zero byte extent. Check the actual pointer bits
    // received by the kernel, not an `is_null` predicate LLVM could fold from
    // the new nonnull attribute. No dangling pointer is dereferenced.
    let input = DeviceBuffer::from_host(&stream, &[AlignedZst; 7])?;
    let mut output = DeviceBuffer::<[u64; 2]>::zeroed(&stream, 1)?;
    let expected = [[std::mem::align_of::<AlignedZst>() as u64, 7]];
    module.inspect_zsts(&stream, &prepared_zst, &input, &mut output)?;
    assert_eq!(output.to_host_vec(&stream)?, expected, "sync ZST packet");
    module
        .inspect_zsts_async(&prepared_zst, &input, &mut output)
        .sync()?;
    assert_eq!(
        output.to_host_vec(&stream)?,
        expected,
        "borrowed async ZST packet"
    );
    let (_input, output) = module
        .inspect_zsts_async_owned(&prepared_zst, input, output)
        .sync()?;
    assert_eq!(
        output.to_host_vec(&stream)?,
        expected,
        "owned async ZST packet"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    verify_generated_llvm_ir()?;
    verify_launch_packets()?;
    println!(
        "SUCCESS: kernel reference validity checked in IR and sync/borrowed/owned async GPU launches"
    );
    Ok(())
}
