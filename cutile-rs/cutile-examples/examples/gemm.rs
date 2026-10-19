/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::*;
use cuda_core::Device;
use cutile::api;
use cutile::error::Error;
use cutile::tensor::*;
use cutile::tile_kernel::*;
use cutile::DType;
use my_module::gemm as gemm_kernel;
use std::sync::Arc;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry(print_ir = true)]
    fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

fn gemm<T: DType + std::fmt::Display>() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let scale = 2usize.pow(10); // On the order of megabytes.
    let (bm, bn, bk) = (16, 16, 8);
    let (m, n, k) = (scale * bm, scale * bn, scale * bk as usize);
    let generics = vec![
        T::DTYPE.as_str().to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];
    let z = api::zeros(&[m, n]).partition([bm, bn]).sync_on(&stream)?;
    let x: Arc<Tensor<T>> = api::ones(&[m, k]).sync_on(&stream)?.into();
    let y: Arc<Tensor<T>> = api::ones(&[k, n]).sync_on(&stream)?.into();
    let launcher = gemm_kernel(z, x.clone(), y.clone());
    let (z, _x, _y) = launcher.generics(generics.clone()).sync_on(&stream)?;
    let z_host: Vec<T> = z.unpartition().to_host_vec().sync_on(&stream)?;
    for (i, z) in z_host.iter().enumerate().take(10) {
        println!("z_host[{i}] = {} answer = {}", z, k);
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("gemm", 0)? {
        return Ok(());
    }

    gemm::<f32>()
}
