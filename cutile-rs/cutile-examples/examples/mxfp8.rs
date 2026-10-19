/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: run a CUDA 13.3 MXFP8-style linear-tile kernel and check its output.
 *
 * This uses FP8 E4M3 data with E8M0 block-scale tensors. The data and scale
 * tensors are separate at the API boundary; the kernel loads both and calls
 * `mmaf_scaled`.
 *
 * Run with: cargo run -p cutile-examples --example mxfp8
 */

// Still on the deprecated `Dim::new` spelling; migrate with the examples pass.
#![allow(deprecated)]

use cutile::cuda_core::{f8e4m3fn, f8e8m0fnu};
use cutile::prelude::*;
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

#[cutile::module]
mod mxfp8_linear {
    use cutile::core::*;

    /// One tile block computes a BM x BN output tile of:
    ///
    ///   z[M, N] = alpha * x[M, K] @ y[N, K].T
    ///
    /// Inputs use an MXFP8-style block-scaled layout:
    /// - `x`:        f8e4m3fn[M, K]
    /// - `y`:        f8e4m3fn[N, K]
    /// - `x_scales`: f8e8m0fnu[M, K / 32]
    /// - `y_scales`: f8e8m0fnu[N, K / 32]
    #[cutile::entry()]
    fn linear_tile<const BM: i32, const BN: i32, const BK: i32, const BK_SCALES: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f8e4m3fn, { [-1, -1] }>,
        y: &Tensor<f8e4m3fn, { [-1, -1] }>,
        x_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        y_scales: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let k_tiles = Dim::new(x.shape()[1] / BK);

        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BN, BK]);
        let part_x_scales = x_scales.partition(shape![BM, BK_SCALES]);
        let part_y_scales = y_scales.partition(shape![BN, BK_SCALES]);

        let mut tile_z = constant(0.0f32, shape![BM, BN]);

        for k_tile in k_tiles {
            let tile_x = part_x.load([pid.0, k_tile]);
            let tile_y = part_y.load([pid.1, k_tile]).transpose();
            let tile_x_scales = part_x_scales.load([pid.0, k_tile]);
            let tile_y_scales = part_y_scales.load([pid.1, k_tile]).transpose();

            tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
        }

        let alpha_tile = broadcast_scalar(alpha, z.shape());
        z.store(tile_z * alpha_tile);
    }
}

use mxfp8_linear::linear_tile;

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BLOCK_SCALED_MMA.check("mxfp8", 0)? {
        return Ok(());
    }

    run()
}

fn run() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let gpu_name = get_gpu_name(0);
    println!("Target GPU: {gpu_name}");

    const BM: usize = 16;
    const BN: usize = 16;
    const BK: usize = 64;
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 128;

    let fp8_one = f8e4m3fn(0x38);
    let scale_one = f8e8m0fnu(0x7F);

    let x: Arc<Tensor<f8e4m3fn>> = api::copy_host_vec_to_device(&Arc::new(vec![fp8_one; M * K]))
        .reshape(&[M, K])
        .sync_on(&stream)?
        .into();
    let y: Arc<Tensor<f8e4m3fn>> = api::copy_host_vec_to_device(&Arc::new(vec![fp8_one; N * K]))
        .reshape(&[N, K])
        .sync_on(&stream)?
        .into();
    let x_scales: Arc<Tensor<f8e8m0fnu>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; M * K / 32]))
            .reshape(&[M, K / 32])
            .sync_on(&stream)?
            .into();
    let y_scales: Arc<Tensor<f8e8m0fnu>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; N * K / 32]))
            .reshape(&[N, K / 32])
            .sync_on(&stream)?
            .into();
    let z = api::zeros::<f32>(&[M, N])
        .sync_on(&stream)?
        .partition([BM, BN]);

    let generics = vec![
        BM.to_string(),
        BN.to_string(),
        BK.to_string(),
        (BK / 32).to_string(),
    ];

    let (z, _x, _y, _x_scales, _y_scales, _alpha) =
        linear_tile(z, x, y, x_scales, y_scales, 1.0f32)
            .generics(generics)
            .sync_on(&stream)?;

    let z_host = z.unpartition().to_host_vec().sync_on(&stream)?;
    let expected = K as f32;
    let mut max_abs_err = 0.0f32;
    for (idx, value) in z_host.iter().enumerate() {
        let abs_err = (*value - expected).abs();
        max_abs_err = max_abs_err.max(abs_err);
        assert!(abs_err <= 1e-3, "z[{idx}] = {value}, expected {expected}");
    }

    println!(
        "MXFP8 linear_tile passed: {} outputs matched {expected}, max_abs_err={max_abs_err}",
        z_host.len()
    );
    Ok(())
}
