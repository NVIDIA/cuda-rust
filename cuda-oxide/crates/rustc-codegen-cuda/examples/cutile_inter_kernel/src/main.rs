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

use cuda_async::device_context::{load_module_from_ptx, with_default_device_policy};
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_async::launch::AsyncKernelLaunch;
use cutile::api::{copy_host_vec_to_device, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
use cutile_cuda_core::{Function, LaunchConfig};
use std::future::IntoFuture;
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

struct ThresholdScaleKernel {
    function: Arc<Function>,
    n: u32,
    threshold: f32,
    scale: f32,
    input: Arc<Tensor<f32>>,
    output: Tensor<f32>,
}

impl DeviceOp for ThresholdScaleKernel {
    type Output = (Arc<Tensor<f32>>, Tensor<f32>);

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let mut launcher = AsyncKernelLaunch::new(self.function);
        launcher.push_arg(self.n);
        launcher.push_arg(self.threshold);
        launcher.push_arg(self.scale);
        unsafe {
            launcher
                .push_device_ptr(self.input.device_pointer().cu_deviceptr())
                .push_device_ptr(self.output.device_pointer().cu_deviceptr());
        }
        launcher.set_launch_config(LaunchConfig {
            grid_dim: (self.n.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        });
        unsafe { launcher.execute(ctx)? };
        Ok((self.input, self.output))
    }
}

impl IntoFuture for ThresholdScaleKernel {
    type Output = Result<(Arc<Tensor<f32>>, Tensor<f32>), DeviceError>;
    type IntoFuture = DeviceFuture<(Arc<Tensor<f32>>, Tensor<f32>), ThresholdScaleKernel>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
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
    let gated_out: Tensor<f32> = zeros::<f32>(&[ROWS, COLS]).await?;

    // `then` is the key inter-kernel interop piece here: cutile-rs executes
    // both DeviceOps with the same ExecutionContext, so the Tile launch and
    // the cuda-oxide PTX launch are submitted to the same CUDA stream.
    let pipeline = tile_softmax::row_softmax(softmax_out.partition([BM, BN]), input.clone()).then(
        move |(softmax_part, _input)| {
            let softmax_out: Arc<Tensor<f32>> = softmax_part.unpartition().into();
            ThresholdScaleKernel {
                function: threshold_scale,
                n: N as u32,
                threshold,
                scale,
                input: softmax_out,
                output: gated_out,
            }
        },
    );

    let (_softmax_out, gated_out) = pipeline.await?;

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
