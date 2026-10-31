/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host/tile side of the cuda-oxide <> cutile-rs inter-kernel example.
//!
//! Pipeline:
//!   1. cutile-rs Tile kernel computes row-wise softmax.
//!   2. cuda-oxide SIMT kernel thresholds and scales the softmax result.
//!   3. Both kernels run on the same CUDA stream and share device tensors.

use cuda_async::device_context::load_module_from_ptx;
use cuda_async::device_operation::DeviceOp;
use cuda_async::launch::{AsyncKernelLaunchBuilder, OwnedAsyncKernelLaunch};
use cuda_core::LaunchConfig;
use cutile::api::{copy_host_vec_to_device, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
use std::sync::Arc;

const CUDA_OXIDE_SIMT_PTX_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/simt/cutile_inter_kernel_simt.ptx"
);

#[cutile::module]
mod tile_softmax {
    use cutile::core::*;

    #[cutile::entry()]
    fn row_softmax<const BM: i32, const BN: i32>(
        output: &mut Tensor<f32, { [BM, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile: Tile<f32, { [BM, BN] }> = load_tile_like(input, output);
        let max_per_row: Tile<f32, { [BM] }> = reduce_max(tile, 1i32);
        let max_per_row = max_per_row
            .reshape(const_shape![BM, 1])
            .broadcast(output.shape());

        let numerator: Tile<f32, { [BM, BN] }> = exp(tile - max_per_row);
        let denom: Tile<f32, { [BM] }> = reduce_sum(numerator, 1i32);
        let denom = denom.reshape(const_shape![BM, 1]).broadcast(output.shape());

        output.store(numerator / denom);
    }
}

fn softmax_rows(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_values = &input[start..end];
        let max = row_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = row_values.iter().map(|x| (x - max).exp()).sum();
        for col in 0..cols {
            out[start + col] = (input[start + col] - max).exp() / denom;
        }
    }
    out
}

fn main() -> Result<(), Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(run())
}

async fn run() -> Result<(), Error> {
    const ROWS: usize = 4;
    const COLS: usize = 8;
    const BM: usize = 2;
    const BN: usize = COLS;
    const N: usize = ROWS * COLS;

    let threshold = 0.08f32;
    let scale = 4.0f32;

    let simt_ptx = std::fs::read_to_string(CUDA_OXIDE_SIMT_PTX_PATH).unwrap_or_else(|e| {
        panic!(
            "failed to read generated SIMT PTX at {}: {}\nrun `cargo oxide run cutile_inter_kernel` to build it",
            CUDA_OXIDE_SIMT_PTX_PATH, e
        )
    });
    let module = load_module_from_ptx(&simt_ptx, 0)?;
    let threshold_scale = Arc::new(module.load_function("threshold_scale_f32")?);

    let input_host: Arc<Vec<f32>> = Arc::new(
        (0..N)
            .map(|i| {
                let row = i / COLS;
                let col = i % COLS;
                (row as f32 * 0.25) + (col as f32 * 0.5) - 1.5
            })
            .collect(),
    );

    let input: Arc<Tensor<f32>> = copy_host_vec_to_device(&input_host)
        .await?
        .reshape(&[ROWS, COLS])
        .expect("input reshape")
        .into();
    let softmax_out = zeros::<f32>(&[ROWS, COLS]).await?;
    let allocation = softmax_out.as_buffer()?.storage().allocation_id();
    let address = softmax_out.device_pointer().cu_deviceptr();

    // `then` is the key inter-kernel interop piece here: cutile-rs executes
    // both DeviceOps with the same ExecutionContext, so the Tile launch and
    // the cuda-oxide PTX launch are submitted to the same CUDA stream.
    let pipeline = tile_softmax::row_softmax(softmax_out.partition([BM, BN]), input.clone()).then(
        move |(softmax_part, _input)| {
            // Transfer the Tile result to the shared typed buffer. No allocation
            // or device copy is needed for the SIMT stage.
            let buffer = softmax_part.unpartition().into_buffer();
            assert_eq!(buffer.storage().allocation_id(), allocation);
            assert_eq!(buffer.cu_deviceptr(), address);
            let mut builder = AsyncKernelLaunchBuilder::new(threshold_scale);
            builder.retain_buffer(buffer.storage().clone(), true);
            builder.push_args((N as u32, threshold, scale, address, address));
            // SAFETY: the SIMT function's five parameters match this packet;
            // each thread reads then overwrites its own f32 within 0..N. Raw
            // pointers permit this in-place operation. Storage and exclusion
            // are retained independently of the returned typed buffer.
            let launch =
                unsafe { builder.finalize_unchecked(LaunchConfig::for_num_elems(N as u32)) };
            OwnedAsyncKernelLaunch::new(launch, buffer)
        },
    );

    let buffer = pipeline.await?;
    assert_eq!(buffer.storage().allocation_id(), allocation);
    assert_eq!(buffer.cu_deviceptr(), address);
    let gated_out = Tensor::from_buffer(buffer, vec![ROWS as i32, COLS as i32]);
    assert_eq!(gated_out.device_pointer().cu_deviceptr(), address);

    let got = gated_out.to_host_vec().await?;
    let softmax_expected = softmax_rows(&input_host, ROWS, COLS);
    let expected: Vec<f32> = softmax_expected
        .iter()
        .map(|x| if *x >= threshold { *x * scale } else { 0.0 })
        .collect();

    for (i, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-5,
            "mismatch at {i}: got {actual}, expected {want}"
        );
    }

    println!("PASS: cutile-rs Tile softmax -> cuda-oxide SIMT threshold/scale passed");
    println!("first row input:    {:?}", &input_host[..COLS]);
    println!("first row softmax:  {:?}", &softmax_expected[..COLS]);
    println!("first row output:   {:?}", &got[..COLS]);

    Ok(())
}
