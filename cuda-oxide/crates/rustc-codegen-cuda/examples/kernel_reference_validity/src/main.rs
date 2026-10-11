/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end LLVM-IR coverage for rustc-proven kernel reference validity.
//!
//! Build with `cargo oxide build kernel_reference_validity`, then run the host
//! binary (or `cargo oxide run kernel_reference_validity`) for valid launches.
//! `verify-code-shape.sh` checks frontend attributes and optimized PTX loads.

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

#[repr(C, align(16))]
pub struct GridConstants {
    pub values: [u32; 4],
}

/// # Safety
/// `output` must point to writable, aligned `u32` storage outside `constants`.
/// Concurrent invocations must not race on that storage.
#[kernel]
pub unsafe fn grid_constant(#[grid_constant] constants: &GridConstants, output: *mut u32) {
    unsafe { *output = constants.values[0] };
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

// Interior mutability and pinning must follow rustc's flags, not `&` spelling.
#[kernel]
pub fn atomic_ref(_value: &core::sync::atomic::AtomicU32) {}

pub struct PinnedValue {
    pub value: u32,
    pub pin: core::marker::PhantomPinned,
}

#[kernel]
pub fn pinned_ref(_value: &mut PinnedValue) {}

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

/// With a valid shared input, the raw write cannot change the value read.
/// LLVM can reuse the first load once the reference guarantees reach it.
///
/// # Safety
/// `scratch` and `output` must be writable, aligned `u32` pointers outside
/// `input`. Concurrent invocations must not race on their writes.
#[kernel]
pub unsafe fn read_across_write(input: &u32, scratch: *mut u32, output: *mut u32) {
    unsafe {
        let before = read_input(input);
        *scratch = 13;
        *output = before.wrapping_add(read_input(input));
    }
}

// Preserve two reads through MIR optimization; LLVM sees the reference facts
// on the entry parameter when it later optimizes these calls.
#[inline(never)]
unsafe fn read_input(input: *const u32) -> u32 {
    unsafe { *input }
}

/// Even an immutable input must be read twice when both reads are volatile.
/// Explicit `readonly` plus `noalias` made NVPTX lose `volatile` and allowed
/// ptxas to merge the reads. Withholding `readonly` preserves both accesses.
///
/// # Safety
/// `output` must be writable, aligned, outside `input`, and owned by this thread.
#[kernel]
pub unsafe fn volatile_direct(input: &u32, output: *mut u32) {
    let before = unsafe { core::ptr::read_volatile(input) };
    let after = unsafe { core::ptr::read_volatile(input) };
    unsafe { *output = before.wrapping_add(after) };
}

/// The same guarantee must survive calls that LLVM later inlines.
///
/// # Safety
/// `output` must be writable, aligned, outside `input`, and owned by this thread.
#[kernel]
pub unsafe fn volatile_helper(input: &u32, output: *mut u32) {
    unsafe { *output = read_volatile_input(input).wrapping_add(read_volatile_input(input)) };
}

#[inline(never)]
fn read_volatile_input(input: &u32) -> u32 {
    // SAFETY: the reference is aligned and initialized; reading does not mutate it.
    unsafe { core::ptr::read_volatile(input) }
}

#[kernel]
pub fn by_value(_value: ByValue) {}

#[allow(improper_ctypes_definitions)]
#[kernel]
pub extern "C" fn c_reference(_value: &f32) {}

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
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn add_inputs(a: &[f32], b: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(out) = output.get_mut(index) {
            *out = a[i] + b[i];
        }
    }

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

    // Two read-only parameters may share the same allocation across threads.
    // Only the DisjointSlice output requires disjoint writable elements.
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let input = DeviceBuffer::from_host(&stream, &values)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, values.len())?;
    let add = module.prepare_add_inputs(LaunchConfig1D::new(2, 32, 0))?;
    module.add_inputs(&stream, &add, &input, &input, &mut output)?;
    assert_eq!(
        output.to_host_vec(&stream)?,
        values.iter().map(|value| value * 2.0).collect::<Vec<_>>(),
        "aliased read-only inputs with disjoint parallel output"
    );

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
    verify_launch_packets()?;
    println!(
        "SUCCESS: kernel reference validity checked in sync/borrowed/owned async GPU launches"
    );
    Ok(())
}
