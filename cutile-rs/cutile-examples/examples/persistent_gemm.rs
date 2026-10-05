/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: run a persistent GEMM kernel that iterates private PartitionIndex
 * values from a mapped output partition.
 *
 * Run with:
 *   cargo run -p cutile-examples --example persistent_gemm
 */

use cutile::prelude::*;
use persistent_gemm_kernels::gemm_persistent as gemm_kernel;
use std::sync::Arc;

#[cutile::module]
mod persistent_gemm_kernels {
    use cutile::core::*;

    /// No annotations and no declared contract: the compiler derives every
    /// safety fact from how the kernel walks its inputs — the mapped
    /// components range over `z`'s axes, `k_tile` over `x`'s k axis — and
    /// verifies the cross-tensor ties as launch checks (for example
    /// `ceil(dim(x, 1)/BK) <= ceil(dim(y, 0)/BK)`) before any GPU work.
    /// `deny_in_kernel_checks` makes any check that would remain in the
    /// kernel a compile error, so the body is assert-free by construction.
    #[cutile::entry(
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
            sm_100 = (num_cta_in_cga = 2,),
        ),
        deny_in_kernel_checks = true,
    )]
    fn gemm_persistent<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);

        for out_idx in z.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();

            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, shape![BM, BN]);
            for k_tile in 0i32..num_tiles(&part_x, 1) {
                let tile_x = part_x.load([bid_m, k_tile]);
                let tile_y = part_y.load([k_tile, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, out_idx);
        }
    }
}

fn main() {
    if !cutile_examples::requirements::BASELINE
        .check("persistent_gemm", 0)
        .expect("query example capabilities")
    {
        return;
    }

    run().unwrap();
}

fn run() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    const BM: usize = 16;
    const BN: usize = 16;
    const BK: usize = 8;
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 32;
    const MAP_M: i32 = 4;
    const MAP_N: i32 = 1;
    let num_tile_blocks = 4;

    let generics = vec![
        "f32".into(),
        BM.to_string(),
        BN.to_string(),
        BK.to_string(),
        MAP_M.to_string(),
        MAP_N.to_string(),
    ];

    let x_host = patterned_matrix(M, K, |row, col| {
        ((row % 7) as f32 - 3.0) * 0.25 + ((col % 5) as f32) * 0.125
    });
    let y_host = patterned_matrix(K, N, |row, col| {
        ((row % 3) as f32 + 1.0) * 0.2 - ((col % 11) as f32) * 0.05
    });

    let z = api::zeros::<f32>(&[M, N])
        .sync_on(&stream)?
        .partition([BM, BN])
        .map([MAP_M as usize, MAP_N as usize], num_tile_blocks);
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
        .reshape(&[M, K])
        .sync_on(&stream)?
        .into();
    let y: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(y_host.clone()))
        .reshape(&[K, N])
        .sync_on(&stream)?
        .into();

    let (z, _x, _y) = gemm_kernel(z, x, y).generics(generics).sync_on(&stream)?;

    let z_host = z.unpartition().to_host_vec().sync_on(&stream)?;
    let expected = cpu_gemm(&x_host, &y_host, M, N, K);
    let mut max_abs_err = 0.0f32;
    for (i, (value, expected)) in z_host.iter().zip(expected.iter()).enumerate() {
        let abs_err = (*value - *expected).abs();
        max_abs_err = max_abs_err.max(abs_err);
        assert!(abs_err < 1e-3, "z[{i}] = {value}, expected {expected}");
    }

    println!(
        "Persistent GEMM passed: {} non-uniform outputs matched CPU reference, max_abs_err={max_abs_err}",
        z_host.len(),
    );
    Ok(())
}

fn patterned_matrix<F>(rows: usize, cols: usize, mut f: F) -> Vec<f32>
where
    F: FnMut(usize, usize) -> f32,
{
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            values.push(f(row, col));
        }
    }
    values
}

fn cpu_gemm(x: &[f32], y: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut z = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += x[row * k + inner] * y[inner * n + col];
            }
            z[row * n + col] = acc;
        }
    }
    z
}
