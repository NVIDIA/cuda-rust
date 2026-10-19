// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime regression for pointer provenance in direct and nested struct constants.
//!
//! Thin pointer fields and slice fat-pointer fields both store pointer addends
//! in the allocation bytes while rustc's provenance table identifies the Rust
//! static being referenced. Slice fields additionally carry a literal `usize`
//! metadata word. The importer must preserve both pieces when the constant is
//! materialized for GPU code.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel};

static FIRST: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
static TABLE: [[u32; 4]; 3] = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]];

pub struct Holder {
    pub pointer: &'static [u8; 16],
    pub flag: bool,
}

const DIRECT: Holder = Holder {
    pointer: &FIRST,
    flag: true,
};

#[repr(C)]
pub struct View {
    pub tag: u32,
    pub values: &'static [u32],
}

const DIRECT_SLICE: View = View {
    tag: 7,
    values: &TABLE[1],
};

#[repr(C)]
pub struct NestedView {
    pub prefix: u16,
    pub view: View,
}

const NESTED_SLICE: NestedView = NestedView {
    prefix: 3,
    view: View {
        tag: 11,
        values: &TABLE[2],
    },
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn direct_struct_pointer(mut output: DisjointSlice<u8>) {
        if let Some((slot, index)) = output.get_mut_indexed() {
            let holder = DIRECT;
            *slot = holder.pointer[index.get() & 15] + holder.flag as u8;
        }
    }

    #[kernel]
    pub fn aggregate_slice_provenance(mut output: DisjointSlice<u32>) {
        if let Some((slot, index)) = output.get_mut_indexed() {
            let direct = DIRECT_SLICE;
            let nested = NESTED_SLICE;

            *slot = match index.get() {
                0 => direct.tag,
                1 => direct.values.len() as u32,
                2 => direct.values[0],
                3 => direct.values[3],
                4 => nested.prefix as u32,
                5 => nested.view.tag,
                6 => nested.view.values.len() as u32,
                7 => nested.view.values[0],
                8 => nested.view.values[3],
                _ => 0,
            };
        }
    }
}

fn main() {
    println!("=== struct_constant_provenance ===");

    const N: usize = 64;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let mut output =
        DeviceBuffer::<u8>::zeroed(&stream, N).expect("Failed to allocate device output buffer");

    // SAFETY: the launch is one-dimensional and the output buffer contains one
    // element for every launched thread.
    unsafe {
        module.direct_struct_pointer(&stream, LaunchConfig::for_num_elems(N as u32), &mut output)
    }
    .expect("direct_struct_pointer launch failed");

    let actual = output
        .to_host_vec(&stream)
        .expect("Failed to copy device output to host");
    let expected: Vec<u8> = (0..N)
        .map(|index| FIRST[index & 15] + u8::from(DIRECT.flag))
        .collect();

    assert_eq!(
        actual, expected,
        "struct constant pointer provenance produced incorrect GPU output"
    );

    let mut slice_output = DeviceBuffer::<u32>::zeroed(&stream, 9)
        .expect("Failed to allocate aggregate-slice output buffer");

    // SAFETY: the launch is one-dimensional and the output buffer contains one
    // element for every launched thread.
    unsafe {
        module.aggregate_slice_provenance(
            &stream,
            LaunchConfig::for_num_elems(9),
            &mut slice_output,
        )
    }
    .expect("aggregate_slice_provenance launch failed");

    let slice_actual = slice_output
        .to_host_vec(&stream)
        .expect("Failed to copy aggregate-slice output to host");
    let slice_expected = [7, 4, 20, 23, 3, 11, 4, 30, 33];

    assert_eq!(
        slice_actual.as_slice(),
        slice_expected.as_slice(),
        "aggregate slice constant provenance produced incorrect GPU output"
    );

    println!("PASS: struct constant pointer and slice provenance preserved at runtime");
}
