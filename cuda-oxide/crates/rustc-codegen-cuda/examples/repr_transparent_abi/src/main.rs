/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel-boundary and device-return ABI regression coverage for
//! `#[repr(transparent)]` scalar ADTs.
//!
//! Kernel entries must expose the underlying scalar/pointer parameter rather
//! than an aggregate `.param .b8[...]`. Device helper returns must likewise use
//! the underlying scalar ABI while callers reconstruct the Rust wrapper value.
//! An ordinary one-field struct is kept as the negative control on both paths.

use core::marker::PhantomData;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{Uniform, cuda_module, device, kernel};

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Scalar(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Pointer(pub *const u32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Marked(pub u32, pub PhantomData<()>);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Inner(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Outer(pub Inner);

#[derive(Clone, Copy)]
pub struct Ordinary(pub u32);

#[inline(never)]
#[device]
pub fn return_scalar(value: u32) -> Scalar {
    Scalar(value)
}

#[inline(never)]
#[device]
pub fn return_pointer(value: *const u32) -> Pointer {
    Pointer(value)
}

#[inline(never)]
#[device]
pub fn return_marked(value: u32) -> Marked {
    Marked(value, PhantomData)
}

#[inline(never)]
#[device]
pub fn return_nested(value: u32) -> Outer {
    Outer(Inner(value))
}

#[inline(never)]
#[device]
pub fn return_ordinary(value: u32) -> Ordinary {
    Ordinary(value)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn scalar(value: Scalar, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0) };
    }

    #[kernel]
    pub unsafe fn pointer(value: Pointer, out: *mut u32) {
        // SAFETY: the host launch passes one readable u32 through `value` and
        // one writable u32 through `out`, both live through synchronization.
        unsafe { out.write(value.0.read()) };
    }

    #[kernel]
    pub unsafe fn marked(value: Marked, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0) };
    }

    #[kernel]
    pub unsafe fn nested(value: Outer, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0.0) };
    }

    #[kernel]
    pub unsafe fn uniform(value: Uniform<u32>, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.get()) };
    }

    #[kernel]
    pub unsafe fn ordinary(value: Ordinary, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0) };
    }

    #[kernel]
    pub unsafe fn scalar_return(value: u32, out: *mut u32) {
        let wrapped = return_scalar(value);
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(wrapped.0) };
    }

    #[kernel]
    pub unsafe fn pointer_return(value: *const u32, out: *mut u32) {
        let wrapped = return_pointer(value);
        // SAFETY: the host launch keeps `value` readable and `out` writable
        // through synchronization.
        unsafe { out.write(wrapped.0.read()) };
    }

    #[kernel]
    pub unsafe fn marked_return(value: u32, out: *mut u32) {
        let wrapped = return_marked(value);
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(wrapped.0) };
    }

    #[kernel]
    pub unsafe fn nested_return(value: u32, out: *mut u32) {
        let wrapped = return_nested(value);
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(wrapped.0.0) };
    }

    #[kernel]
    pub unsafe fn ordinary_return(value: u32, out: *mut u32) {
        let wrapped = return_ordinary(value);
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(wrapped.0) };
    }
}

fn entry_header<'source>(
    document: &ptx_parse::Document<'source>,
    name: &str,
) -> Result<&'source str, Box<dyn std::error::Error>> {
    document
        .callables_named(name)
        .find(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
        .and_then(ptx_parse::Callable::definition_header_text)
        .ok_or_else(|| format!("missing or incomplete PTX entry `{name}`").into())
}

fn require_scalar_header(
    document: &ptx_parse::Document<'_>,
    name: &str,
    scalar_token: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = entry_header(document, name)?;
    if header.contains(".b8") {
        return Err(
            format!("kernel `{name}` still exposes an aggregate PTX parameter:\n{header}").into(),
        );
    }
    if !header.contains(scalar_token) {
        return Err(format!(
            "kernel `{name}` does not expose expected `{scalar_token}` parameter:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn require_llvm_return_shape(
    llvm_ir: &str,
    name_fragment: &str,
    return_tokens: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let header = llvm_ir
        .lines()
        .find(|line| line.trim_start().starts_with("define ") && line.contains(name_fragment))
        .ok_or_else(|| format!("missing LLVM definition containing `{name_fragment}`"))?;
    if !return_tokens.iter().any(|token| header.contains(token)) {
        return Err(format!(
            "device helper `{name_fragment}` does not expose any expected return token {:?}:\n{header}",
            return_tokens
        )
        .into());
    }
    Ok(())
}

fn verify_generated_llvm_ir() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("repr_transparent_abi.ll");
    let llvm_ir = std::fs::read_to_string(&path)?;

    require_llvm_return_shape(&llvm_ir, "return_scalar", &["i32 @"])?;
    require_llvm_return_shape(&llvm_ir, "return_pointer", &["ptr @", "* @"])?;
    require_llvm_return_shape(&llvm_ir, "return_marked", &["i32 @"])?;
    require_llvm_return_shape(&llvm_ir, "return_nested", &["i32 @"])?;
    require_llvm_return_shape(&llvm_ir, "return_ordinary", &["{ i32 } @"])?;

    Ok(())
}

fn verify_generated_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("repr_transparent_abi.ptx");
    let ptx = std::fs::read_to_string(&path)?;
    let document = ptx_parse::Document::parse(&ptx)?;

    require_scalar_header(&document, "scalar", ".param .u32")?;
    require_scalar_header(&document, "pointer", ".param .u64")?;
    require_scalar_header(&document, "marked", ".param .u32")?;
    require_scalar_header(&document, "nested", ".param .u32")?;
    require_scalar_header(&document, "uniform", ".param .u32")?;

    let ordinary = entry_header(&document, "ordinary")?;
    if !ordinary.contains(".b8") {
        return Err(format!(
            "ordinary one-field struct was incorrectly scalarized at the kernel boundary:\n{ordinary}"
        )
        .into());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        verify_generated_llvm_ir()?;
        verify_generated_ptx()?;
        println!("repr_transparent_abi: PASS (LLVM return ABI and PTX parameter shapes)");
        return Ok(());
    }

    verify_generated_llvm_ir()?;
    verify_generated_ptx()?;

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let input = DeviceBuffer::from_host(&stream, &[17u32])?;
    let scalar_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let pointer_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let marked_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let nested_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let uniform_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let ordinary_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let scalar_return_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let pointer_return_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let marked_return_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let nested_return_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let ordinary_return_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;

    // SAFETY: every kernel launches one thread. Each output buffer owns one
    // writable u32, and `input` owns the readable u32 used by pointer tests.
    unsafe {
        module.scalar(
            &stream,
            config,
            Scalar(11),
            scalar_out.cu_deviceptr() as *mut u32,
        )?;
        module.pointer(
            &stream,
            config,
            Pointer(input.cu_deviceptr() as *const u32),
            pointer_out.cu_deviceptr() as *mut u32,
        )?;
        module.marked(
            &stream,
            config,
            Marked(23, PhantomData),
            marked_out.cu_deviceptr() as *mut u32,
        )?;
        module.nested(
            &stream,
            config,
            Outer(Inner(29)),
            nested_out.cu_deviceptr() as *mut u32,
        )?;
        // `#[cuda_module]` intentionally maps `Uniform<u32>` to a bare host
        // `u32`; the device type remains the proof-carrying transparent ADT.
        module.uniform(
            &stream,
            config,
            31u32,
            uniform_out.cu_deviceptr() as *mut u32,
        )?;
        module.ordinary(
            &stream,
            config,
            Ordinary(37),
            ordinary_out.cu_deviceptr() as *mut u32,
        )?;

        module.scalar_return(
            &stream,
            config,
            41u32,
            scalar_return_out.cu_deviceptr() as *mut u32,
        )?;
        module.pointer_return(
            &stream,
            config,
            input.cu_deviceptr() as *const u32,
            pointer_return_out.cu_deviceptr() as *mut u32,
        )?;
        module.marked_return(
            &stream,
            config,
            43u32,
            marked_return_out.cu_deviceptr() as *mut u32,
        )?;
        module.nested_return(
            &stream,
            config,
            47u32,
            nested_return_out.cu_deviceptr() as *mut u32,
        )?;
        module.ordinary_return(
            &stream,
            config,
            53u32,
            ordinary_return_out.cu_deviceptr() as *mut u32,
        )?;
    }

    assert_eq!(scalar_out.to_host_vec(&stream)?, [11]);
    assert_eq!(pointer_out.to_host_vec(&stream)?, [17]);
    assert_eq!(marked_out.to_host_vec(&stream)?, [23]);
    assert_eq!(nested_out.to_host_vec(&stream)?, [29]);
    assert_eq!(uniform_out.to_host_vec(&stream)?, [31]);
    assert_eq!(ordinary_out.to_host_vec(&stream)?, [37]);
    assert_eq!(scalar_return_out.to_host_vec(&stream)?, [41]);
    assert_eq!(pointer_return_out.to_host_vec(&stream)?, [17]);
    assert_eq!(marked_return_out.to_host_vec(&stream)?, [43]);
    assert_eq!(nested_return_out.to_host_vec(&stream)?, [47]);
    assert_eq!(ordinary_return_out.to_host_vec(&stream)?, [53]);

    println!("repr_transparent_abi: PASS (kernel parameters, device returns, and runtime values)");
    Ok(())
}
