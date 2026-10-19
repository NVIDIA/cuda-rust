/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const WIDTH: usize = 16;
const HEIGHT: usize = 8;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn copy_2d_const_width(
        input: &[f32],
        mut output: DisjointSlice<f32, thread::Index2D<WIDTH>>,
        height: u32,
    ) {
        let row = thread::index_2d_row();

        if let Some(idx) = thread::index_2d::<WIDTH>()
            && row < height as usize
        {
            let i = idx.get();
            if let Some(out_elem) = output.get_mut(idx) {
                *out_elem = input[i];
            }
        }
    }
}

fn main() {
    println!("=== Const-Stride 2D Indexing Example ===");
    println!("Copying a {HEIGHT}x{WIDTH} matrix with index_2d::<{WIDTH}>()");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let input_host: Vec<f32> = (0..(WIDTH * HEIGHT)).map(|i| i as f32).collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, WIDTH * HEIGHT).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.copy_2d_const_width(
            &stream,
            LaunchConfig {
                grid_dim: (1, HEIGHT as u32, 1),
                block_dim: (WIDTH as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &input_dev,
            &mut output_dev,
            HEIGHT as u32,
        )
    }
    .expect("Kernel launch failed");

    let output_host = output_dev.to_host_vec(&stream).unwrap();
    for i in 0..(WIDTH * HEIGHT) {
        let expected = input_host[i];
        let actual = output_host[i];
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "mismatch at {i}: expected {expected}, got {actual}"
        );
    }

    println!("SUCCESS: const-stride 2D copy verified.");
}
