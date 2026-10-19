/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime regression for pointer provenance in enum constants.
//!
//! Covers niche-encoded enums pointing to a device static or an interior
//! static subobject, direct-tagged enums, and relocation-carrying enums nested
//! inside struct, tuple, and array constants.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::kernel;
use cuda_host::cuda_module;

const FIRST: u64 = 0x1122_3344_5566_7788;
const SECOND: u64 = 0x8877_6655_4433_2211;

static TARGETS: [u64; 2] = [FIRST, SECOND];

// This is a zero-addend pointer to an aggregate subobject: the static's
// physical address has pointee `[u64; 2]`, while the Rust constant is `&u64`.
// It pins shape normalization before the StaticAddress kind boundary.
const FIRST_TARGET: &u64 = &TARGETS[0];
const NICHE_STATIC: Option<&'static u64> = Some(&TARGETS[1]);
const NICHE_NONE: Option<&'static u64> = None;

#[repr(u8)]
#[derive(Clone, Copy)]
enum TaggedReference {
    Empty,
    Present(&'static u64),
}

#[derive(Clone, Copy)]
struct EnumHolder {
    marker: u32,
    item: TaggedReference,
}

const DIRECT_EMPTY: TaggedReference = TaggedReference::Empty;
const DIRECT_TAGGED: TaggedReference = TaggedReference::Present(&TARGETS[0]);
const NESTED_STRUCT: EnumHolder = EnumHolder {
    marker: 17,
    item: TaggedReference::Present(&TARGETS[1]),
};
const NESTED_TUPLE: (u32, Option<&'static u64>) = (23, Some(&TARGETS[0]));
const NESTED_ARRAY: [TaggedReference; 2] = [
    TaggedReference::Empty,
    TaggedReference::Present(&TARGETS[1]),
];

#[inline(never)]
fn niche_static() -> Option<&'static u64> {
    NICHE_STATIC
}

#[inline(never)]
fn first_target() -> &'static u64 {
    FIRST_TARGET
}

#[inline(never)]
fn niche_none() -> Option<&'static u64> {
    NICHE_NONE
}

#[inline(never)]
fn direct_empty() -> TaggedReference {
    DIRECT_EMPTY
}

#[inline(never)]
fn direct_tagged() -> TaggedReference {
    DIRECT_TAGGED
}

#[inline(never)]
fn nested_struct() -> EnumHolder {
    NESTED_STRUCT
}

#[inline(never)]
fn nested_tuple() -> (u32, Option<&'static u64>) {
    NESTED_TUPLE
}

#[inline(never)]
fn nested_array() -> [TaggedReference; 2] {
    NESTED_ARRAY
}

#[inline(never)]
fn tagged_value(value: TaggedReference) -> u64 {
    match value {
        TaggedReference::Empty => 0,
        TaggedReference::Present(pointer) => *pointer,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `output` must point to writable device memory for nine `u64` values.
    #[kernel]
    pub unsafe fn enum_pointer_constants(output: *mut u64) {
        let static_value = niche_static().map_or(0, |pointer| *pointer);
        let none_value = niche_none().map_or(0, |pointer| *pointer);
        let direct_empty_value = tagged_value(direct_empty());
        let direct_tagged_value = tagged_value(direct_tagged());

        let holder = nested_struct();
        let nested_struct_marker = holder.marker as u64;
        let nested_struct_value = tagged_value(holder.item);

        let nested_tuple_value = nested_tuple().1.map_or(0, |pointer| *pointer);

        let array_value = nested_array();
        let nested_array_value = tagged_value(array_value[1]);
        let first_target_value = *first_target();

        unsafe {
            output.add(0).write(static_value);
            output.add(1).write(none_value);
            output.add(2).write(direct_empty_value);
            output.add(3).write(direct_tagged_value);
            output.add(4).write(nested_struct_marker);
            output.add(5).write(nested_struct_value);
            output.add(6).write(nested_tuple_value);
            output.add(7).write(nested_array_value);
            output.add(8).write(first_target_value);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const OUTPUT_COUNT: usize = 9;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let output = DeviceBuffer::<u64>::zeroed(&stream, OUTPUT_COUNT)?;

    // SAFETY: the output allocation contains nine u64 values and exactly one
    // thread is launched.
    unsafe {
        module.enum_pointer_constants(
            &stream,
            LaunchConfig::for_num_elems(1),
            output.cu_deviceptr() as *mut u64,
        )
    }?;

    let actual = output.to_host_vec(&stream)?;
    let expected = [SECOND, 0, 0, FIRST, 17, SECOND, FIRST, SECOND, FIRST];

    assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "enum constant pointer provenance produced incorrect GPU output"
    );

    println!("enum_constant_provenance: PASS");
    Ok(())
}
