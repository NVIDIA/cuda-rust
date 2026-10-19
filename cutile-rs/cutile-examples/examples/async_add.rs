/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::prelude::*;
use my_module::add;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry(print_ir = true)]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), cuda_async::error::DeviceError> {
    if !cutile_examples::requirements::BASELINE.check("async_add", 0)? {
        return Ok(());
    }

    let len = 2usize.pow(5);

    // Run add kernel twice, chaining lazily with .then().
    let z_host: Vec<f32> = add(
        api::zeros(&[len]).partition([2]),
        api::arange(len),
        api::ones(&[len]),
    )
    .then(|(z, x, y)| add(z, x, y))
    .first()
    .unpartition()
    .to_host_vec()
    .await?;

    // Both calls compute z = x + y with the same x, y.
    // The second overwrites the first, so the result is just x + y.
    for (i, &v) in z_host.iter().enumerate() {
        let expected = i as f32 + 1.0;
        assert_eq!(expected, v, "index {}: {} != {}", i, expected, v);
    }
    println!("async_add: all values correct");
    Ok(())
}
