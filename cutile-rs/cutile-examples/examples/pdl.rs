/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Programmatic dependent launch (PDL): the consumer kernel may start before
 * the producer kernel has finished.
 *
 * Producer: store results, then `gdc_launch_dependents_tko` lets the
 * consumer start. Consumer: `gdc_wait_tko` waits for the producer to finish
 * and its stores to become visible; `input.set_token` makes every later
 * load of `input` depend on that wait. Host: `.programmatic_dependent_launch()`
 * on the consumer's launch enables PDL. Without it both device calls are no-ops.
 *
 * Requires Tile IR 13.4 and sm_90 or newer.
 *
 * Usage:
 *   cargo run -p cutile-examples --example pdl
 */

use cuda_core::Device;
use cutile::error::Error;
use cutile::prelude::*;

const N: usize = 1024;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// staging[i] = i, then let dependents start.
    #[cutile::entry()]
    pub fn producer(staging: &mut Tensor<i32, { [128] }>) {
        let offsets: Tile<i32, { [128] }> =
            broadcast_scalar(program_id(0) * 128, shape![128]) + iota(shape![128]);
        // Chain the signal after the store's completion token.
        let stored: Token = staging.store(offsets);
        let _signal: Token = unsafe { gdc_launch_dependents_tko(Some(stored)) };
    }

    /// out[i] = 2 * input[i].
    #[cutile::entry()]
    pub fn consumer(out: &mut Tensor<i32, { [128] }>, input: &Tensor<i32, { [-1] }>) {
        // Wait for the producer; every load of `input` from here on depends on it.
        unsafe { input.set_token(gdc_wait_tko(None)) };

        let x: Tile<i32, { [128] }> = input.load_like(out);
        let two: Tile<i32, { [128] }> = broadcast_scalar(2, shape![128]);
        out.store(x * two);
    }
}

use kernels::{consumer, producer};

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::PDL.check("pdl", 0)? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let staging = api::zeros::<i32>(&[N]).sync_on(&stream)?;
    let output = api::zeros::<i32>(&[N]).sync_on(&stream)?;
    let grid = ((N / 128) as u32, 1, 1);

    // The producer's output is the consumer's input: composing the ops
    // enqueues them back to back on the stream.
    let staging = producer(staging.partition([128]))
        .grid(grid)
        .map(|(staging,)| staging.unpartition());
    let (output, _staging) =
        unsafe { consumer(output.partition([128]), staging).programmatic_dependent_launch() }
            .grid(grid)
            .sync_on(&stream)?;

    let host = output.unpartition().to_host_vec().sync_on(&stream)?;
    let expected: Vec<i32> = (0..N as i32).map(|i| 2 * i).collect();
    assert_eq!(host, expected);
    println!("PDL producer/consumer: output[i] == 2 * i for {N} elements.");
    Ok(())
}
