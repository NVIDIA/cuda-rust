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
    fn gemm<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const M: i32,
        const N: i32,
        const K: i32,
    >(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [M, K] }>,
        y: &Tensor<E, { [K, N] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let mut tile_z = load_tile_mut(z);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let nk = num_tiles(&part_y, 0);
        for i in 0i32..nk {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
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
        m.to_string(),
        n.to_string(),
        k.to_string(),
    ];
    let z = api::zeros(&[m, n]).partition([bm, bn]).sync_on(&stream)?;
    let x: Arc<Tensor<T>> = api::ones(&[m, k]).sync_on(&stream)?.into();
    let y: Arc<Tensor<T>> = api::ones(&[k, n]).sync_on(&stream)?.into();
    println!(
        "Everything created. x size(mb)={}, y size(mb)={}, z size(mb)={}",
        x.num_bytes() / 1_000_000,
        y.num_bytes() / 1_000_000,
        z.num_bytes() / 1_000_000
    );
    let grid = z.grid()?;
    let (z, _x, _y) = gemm_kernel(z, x, y)
        .const_grid(grid)
        .generics(generics)
        .sync_on(&stream)?;
    println!("CUDA_GEMM Done.");
    let z_host: Vec<T> = z.unpartition().to_host_vec().sync_on(&stream)?;
    for (i, z) in z_host.iter().enumerate().take(10) {
        println!("z_host[{i}] = {} answer = {}", z, k);
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("gemm_static", 0)? {
        return Ok(());
    }

    gemm::<f32>()
}
