/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: run a CUDA 13.3 NVFP4 linear-tile kernel and check its output.
 *
 * This models the inference boundary where activations and weights are already
 * represented as packed FP4 pairs, with separate FP8 block-scale tensors.
 *
 * Run with: cargo run -p cutile-examples --example nvfp4
 */

use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

#[cutile::module]
mod nvfp4_linear {
    use cutile::core::*;

    /// One tile block computes a BM x BN output tile of:
    ///
    ///   z[M, N] = alpha * x[M, K] @ y[N, K].T
    ///
    /// Inputs use the usual NVFP4 inference layout:
    /// - `x`:        f4e2m1fnx2[M, K / 2], packed activations
    /// - `y`:        f4e2m1fnx2[N, K / 2], packed weights
    /// - `x_scales`: f8e4m3fn[M, K / 16]
    /// - `y_scales`: f8e4m3fn[N, K / 16]
    ///
    /// The example keeps the scale tensors in logical row-major layout. Kernels
    /// with different scale tensor layouts should adapt their load indices and
    /// host-side tensor shapes accordingly.
    #[cutile::entry()]
    fn linear_tile<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BK_PACKED: i32,
        const BK_SCALES: i32,
    >(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        y: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        y_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();

        let part_x = x.partition(shape![BM, BK_PACKED]);
        let part_y = y.partition(shape![BN, BK_PACKED]);
        let part_x_scales = x_scales.partition(shape![BM, BK_SCALES]);
        let part_y_scales = y_scales.partition(shape![BN, BK_SCALES]);

        let mut tile_z = constant(0.0f32, shape![BM, BN]);

        for k_tile in 0i32..num_tiles(&part_x, 1) {
            let tile_x_packed = part_x.load([pid.0, k_tile]);
            let tile_y_packed = part_y.load([pid.1, k_tile]);

            let tile_x = tile_x_packed.unpack(shape![BM, BK]);
            let tile_y = tile_y_packed.unpack(shape![BN, BK]).transpose();

            let tile_x_scales = part_x_scales.load([pid.0, k_tile]);
            let tile_y_scales = part_y_scales.load([pid.1, k_tile]).transpose();

            tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
        }

        let alpha_tile = broadcast_scalar(alpha, z.shape());
        z.store(tile_z * alpha_tile);
    }
}

use nvfp4_linear::linear_tile;

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BLOCK_SCALED_MMA.check("nvfp4", 0)? {
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

    let fp4_one = f4e2m1fnx2::from_nibbles(0x2, 0x2);
    let scale_one = f8e4m3fn(0x38);

    let x: Arc<Tensor<f4e2m1fnx2>> =
        api::copy_host_vec_to_device(&Arc::new(vec![fp4_one; M * K / 2]))
            .reshape(&[M, K / 2])
            .sync_on(&stream)?
            .into();
    let y: Arc<Tensor<f4e2m1fnx2>> =
        api::copy_host_vec_to_device(&Arc::new(vec![fp4_one; N * K / 2]))
            .reshape(&[N, K / 2])
            .sync_on(&stream)?
            .into();
    let x_scales: Arc<Tensor<f8e4m3fn>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; M * K / 16]))
            .reshape(&[M, K / 16])
            .sync_on(&stream)?
            .into();
    let y_scales: Arc<Tensor<f8e4m3fn>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; N * K / 16]))
            .reshape(&[N, K / 16])
            .sync_on(&stream)?
            .into();
    let z = api::zeros::<f32>(&[M, N])
        .sync_on(&stream)?
        .partition([BM, BN]);

    let generics = vec![
        BM.to_string(),
        BN.to_string(),
        BK.to_string(),
        (BK / 2).to_string(),
        (BK / 16).to_string(),
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
        "NVFP4 linear_tile passed: {} outputs matched {expected}, max_abs_err={max_abs_err}",
        z_host.len()
    );
    Ok(())
}
