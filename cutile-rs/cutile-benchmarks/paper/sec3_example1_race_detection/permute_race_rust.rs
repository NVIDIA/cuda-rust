/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Data race prevention: kernel-internal index bug on head permutation.
 *
 * The cuTile Python version (permute_race.py) has a kernel-internal bug:
 * the store index has `m` and `h2` swapped. H blocks with different h1
 * race on the same destination address, writing different source tiles.
 * The race is real per Tile IR's memory model (the writes are not morally
 * strong: no device-scope ordering, and tokens only order within a single
 * tile thread). Outputs are non-deterministic across runs.
 *
 * In cuTile Rust, the equivalent bug is inexpressible. The kernel
 * receives `dst` as a &mut Tensor partition view -- the destination tile
 * assigned to this block. The store writes to the partition view
 * directly; there is no destination index for the programmer to get
 * wrong.
 *
 * Build and run:
 *   cd /path/to/cutile-rs
 *   cargo run -p cutile-examples --example permute_race_rust
 */

use cuda_core::Device;
use cutile::error::Error;
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// Permute head dimensions of a rank-4 attention tensor.
    ///
    /// `dst` is a partition view (one destination tile per tile thread).
    /// `src` is shared and immutable; the kernel reads its block id to
    /// choose which source head to load.
    ///
    /// There is no destination index in the store: the partition IS the
    /// destination. A "swap `m` and `h2`" bug in the store is not
    /// expressible, and neither is any other wrong-destination bug.
    #[cutile::entry()]
    pub fn permute_heads<const BM: i32, const BD: i32>(
        dst: &mut Tensor<f32, { [1, 1, BM, BD] }>,
        src: &Tensor<f32, { [-1, -1, -1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let b = pid.0;
        let h1 = pid.1;
        let part = src.partition(const_shape![1, 1, BM, BD]);
        let tile = part.load([b, h1, 0, 0]);
        dst.store(tile);
    }
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let b = 2usize;
    let h = 8usize;
    let m = 128usize;
    let d = 64usize;

    // `src` is shared; `dst` is partitioned. The launcher passes the
    // partition structure so that each tile thread owns one `&mut` to
    // a disjoint destination tile.
    let src: Arc<Tensor<f32>> = cutile::api::randn([b, h, m, d], None)
        .sync_on(&stream)?
        .into();
    let dst = cutile::api::zeros(&[b, h, m, d])
        .sync_on(&stream)?
        .partition([1, 1, 32, 64]); // BM=32, BD=64

    let _ = kernels::permute_heads(dst, src.clone()).sync_on(&stream)?;
    println!("Kernel ran. Each dst partition element is written by");
    println!("exactly one tile thread -- no concurrent writes, no race.");
    println!("The cuTile Python bug class (wrong destination index) is");
    println!("not expressible here: `dst.store(tile)` has no index to");
    println!("get wrong.");

    Ok(())
}
