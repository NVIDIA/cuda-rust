/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::prelude::*;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2]>(y: &mut Tensor<f32, S>, a: f32, x: &Tensor<f32, { [-1, -1] }>) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = x.load_like(y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

use my_module::saxpy;

// TODO (hme): Answer question about whether main should return Result<(), ...>
fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("saxpy", 0)? {
        return Ok(());
    }

    // Create a context. Device 0 is associated with the context.
    let device = Device::new(0)?;
    // Create a new stream on which we run CUDA operations.
    let stream = device.new_stream()?;
    let a = 2.0;
    let input: Arc<Tensor<f32>> = api::arange(2usize.pow(5)).sync_on(&stream)?.into();
    let x: Arc<Tensor<f32>> = input
        .dup()
        .sync_on(&stream)?
        .reshape(&[4, 8])
        .unwrap()
        .into();
    let y = input
        .dup()
        .sync_on(&stream)?
        .reshape(&[4, 8])
        .unwrap()
        .partition([2, 2]);
    let y_host: Vec<f32> = saxpy(y, a, x)
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;
    let input_host: Vec<f32> = input.to_host_vec().sync_on(&stream)?;
    for i in 0..input_host.len() {
        let x_i: f32 = input_host[i];
        let y_i: f32 = input_host[i];
        let answer = a * x_i + y_i;
        println!("{} * {} + {} = {}", a, x_i, y_i, y_host[i]);
        assert_eq!(answer, y_host[i], "{} != {} ?", answer, y_host[i]);
    }
    Ok(())
}
