/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end IKET annotation example.
//!
//! Build and run with:
//!
//! ```text
//! cargo oxide build iket_trace --arch sm_120
//! ```

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, iket, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn traced_vecadd(a: &[f32], b: &[f32], mut output: DisjointSlice<f32>) {
        iket::mark!("kernel.enter");

        let compute = iket::range_start!("vecadd.compute");
        let index = thread::index_1d();
        let raw_index = index.get();
        if let Some(element) = output.get_mut(index) {
            *element = a[raw_index] + b[raw_index];
        }
        iket::range_end!(compute);

        // Names longer than the NativeDump inline limit remain supported. The
        // compiler emits a hashed metadata reference and preserves the full
        // string in a device global for the IKET runtime to recover.
        iket::mark!(
            "vecadd.completed.with.a.name.longer.than.thirty.one.bytes",
            raw_index as u32
        );
    }
}

fn main() {
    const ELEMENTS: usize = 1024;

    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let a = (0..ELEMENTS).map(|value| value as f32).collect::<Vec<_>>();
    let b = (0..ELEMENTS)
        .map(|value| (value * 2) as f32)
        .collect::<Vec<_>>();
    let a_device = DeviceBuffer::from_host(&stream, &a).expect("copy a");
    let b_device = DeviceBuffer::from_host(&stream, &b).expect("copy b");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, ELEMENTS).expect("allocate output");
    let module = kernels::load(&context).expect("load traced module");

    // SAFETY: the launch covers exactly `ELEMENTS` values and all buffers have
    // that length.
    unsafe {
        module.traced_vecadd(
            &stream,
            LaunchConfig::for_num_elems(ELEMENTS as u32),
            &a_device,
            &b_device,
            &mut output,
        )
    }
    .expect("launch traced_vecadd");

    let result = output.to_host_vec(&stream).expect("copy output");
    assert!(
        result
            .iter()
            .enumerate()
            .all(|(index, value)| *value == (index * 3) as f32)
    );
    println!("SUCCESS: IKET trace example passed for {ELEMENTS} elements");
}
