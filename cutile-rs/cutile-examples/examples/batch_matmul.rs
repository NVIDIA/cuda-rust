/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{ones, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn batch_matmul<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        c: &mut Tensor<E, { [1, BM, BN] }>,
        a: &Tensor<E, { [-1, -1, K] }>,
        b: &Tensor<E, { [-1, K, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id(); // (batch_idx, m_idx, n_idx)
        let batch_idx = pid.0;
        let m_idx = pid.1;
        let n_idx = pid.2;

        let a_part: Partition<E, { [1, BM, BK] }> = a.partition(shape![1, BM, BK]);
        let b_part: Partition<E, { [1, BK, BN] }> = b.partition(shape![1, BK, BN]);

        let acc_val: E = convert_scalar(0i32);
        let mut acc: Tile<E, { [BM, BN] }> = broadcast_scalar(acc_val, shape![BM, BN]);
        for k_idx in 0i32..(K / BK) {
            let a_tile: Tile<E, { [BM, BK] }> = a_part
                .load([batch_idx, m_idx, k_idx])
                .reshape(shape![BM, BK]);
            let b_tile: Tile<E, { [BK, BN] }> = b_part
                .load([batch_idx, k_idx, n_idx])
                .reshape(shape![BK, BN]);
            acc = mma(a_tile, b_tile, acc);
        }
        c.store(acc.reshape(shape![1, BM, BN]));
    }
}

use my_module::batch_matmul;

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("batch_matmul", 0)? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let batch = 4usize;
    let (m, n, k) = (128usize, 256usize, 64usize);
    let (bm, bn, bk) = (64, 64, 32);

    let a: Arc<Tensor<f32>> = ones(&[batch, m, k]).sync_on(&stream)?.into();
    let b: Arc<Tensor<f32>> = ones(&[batch, k, n]).sync_on(&stream)?.into();
    let c: Partition<Tensor<f32>> = zeros(&[batch, m, n])
        .sync_on(&stream)?
        .partition([1, bm, bn]);

    let generics = vec![
        "f32".to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];
    let (c, _a, _b) = batch_matmul(c, a, b).generics(generics).sync_on(&stream)?;
    let c_host: Vec<f32> = c.unpartition().to_host_vec().sync_on(&stream)?;

    let expected = k as f32;
    for (idx, value) in c_host.iter().enumerate().take(10) {
        println!("c_host[{idx}] = {value}, expected = {expected}");
        assert!((value - expected).abs() <= 1e-3);
    }

    Ok(())
}
