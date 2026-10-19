/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for issue #400.
//!
//! A Copy aggregate passed to a kernel by value is borrowed only by inlined
//! read-only methods. Runtime indexing into its small array fields must not
//! force the aggregate into NVPTX local memory.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct GridShape {
    pub counts: [u32; 3],
    pub periodic: [bool; 3],
}

impl GridShape {
    #[inline(always)]
    fn count(&self, axis: usize) -> u32 {
        self.counts[axis]
    }

    #[inline(always)]
    fn periodic_bit(&self, axis: usize) -> u32 {
        self.periodic[axis] as u32
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn borrowed_copy_aggregate(mut output: DisjointSlice<u32>, shape: GridShape) {
        let index = thread::index_1d();
        let linear_index = index.get() as usize;
        if let Some(output) = output.get_mut(index) {
            let axis = linear_index % shape.counts.len();
            *output = shape.count(axis) + 1_000 * shape.periodic_bit(axis);
        }
    }
}

fn main() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let module = kernels::load(&context).expect("Failed to load embedded CUDA module");
    let stream = context.default_stream();

    let shape = GridShape {
        counts: [7, 11, 19],
        periodic: [false, true, true],
    };
    let element_count = 12usize;
    let mut output =
        DeviceBuffer::<u32>::zeroed(&stream, element_count).expect("allocate output buffer");

    unsafe {
        module.borrowed_copy_aggregate(
            stream.as_ref(),
            LaunchConfig::for_num_elems(
                u32::try_from(element_count).expect("element count fits in u32"),
            ),
            &mut output,
            shape,
        )
    }
    .expect("launch borrowed_copy_aggregate");

    let actual = output.to_host_vec(&stream).expect("copy output");
    let expected: Vec<u32> = (0..element_count)
        .map(|index| {
            let axis = index % shape.counts.len();
            shape.counts[axis] + 1_000 * shape.periodic[axis] as u32
        })
        .collect();
    assert_eq!(actual, expected);

    println!("copy_aggregate_borrow: PASS");
}
