/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Token ordering comparison: cuTile Rust side.
 *
 * Compile and run this to dump the IR for comparison with
 * token_ordering_comparison.py.
 *
 * Usage:
 *   cd /path/to/cutile-rs
 *   cargo run -p cutile-examples --example token_ordering_rust
 *
 * Look for print_ir output on stderr.
 */

use cuda_core::Device;
use cutile::error::Error;
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// Kernel 1: add_accum
    /// c is &mut (mutable: token-ordered)
    /// a, b are & (immutable: should be unconstrained)
    #[cutile::entry(print_ir = true)]
    pub fn add_accum<const S: [i32; 1]>(
        c: &mut Tensor<f32, S>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let tile_a = load_tile_like(a, c);
        let tile_b = load_tile_like(b, c);
        let tile_c: Tile<f32, S> = c.load();
        c.store(tile_a + tile_b + tile_c);
    }

    /// Kernel 2: multi_read_single_write
    /// dst is &mut (token-ordered)
    /// a, b, c are & (unconstrained)
    #[cutile::entry(print_ir = true)]
    pub fn multi_read<const S: [i32; 1]>(
        dst: &mut Tensor<f32, S>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        c: &Tensor<f32, { [-1] }>,
    ) {
        let ta = load_tile_like(a, dst);
        let tb = load_tile_like(b, dst);
        let tc = load_tile_like(c, dst);
        dst.store(ta + tb + tc);
    }

    /// Kernel 3: interleaved_access
    /// out is &mut (token-ordered)
    /// data is & (unconstrained)
    /// Interleaved: read data, write out, read data again, write out again.
    #[cutile::entry(print_ir = true)]
    pub fn interleaved<const S: [i32; 1]>(out: &mut Tensor<f32, S>, data: &Tensor<f32, { [-1] }>) {
        let t1 = load_tile_like(data, out);
        let old: Tile<f32, S> = out.load();
        out.store(old + t1);
        let t2 = load_tile_like(data, out);
        let cur: Tile<f32, S> = out.load();
        out.store(cur + t2);
    }
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let n = 1024usize;
    let tile = 128usize;

    // Kernel 1: add_accum
    let a: Arc<Tensor<f32>> = cutile::api::ones(&[n]).sync_on(&stream)?.into();
    let b: Arc<Tensor<f32>> = cutile::api::ones(&[n]).sync_on(&stream)?.into();
    let c_part = cutile::api::zeros(&[n]).sync_on(&stream)?.partition([tile]);
    let _ = kernels::add_accum(c_part, a.clone(), b.clone()).sync_on(&stream)?;
    eprintln!("add_accum: compiled (check IR above)");

    // Kernel 2: multi_read
    let c: Arc<Tensor<f32>> = cutile::api::ones(&[n]).sync_on(&stream)?.into();
    let dst_part = cutile::api::zeros(&[n]).sync_on(&stream)?.partition([tile]);
    let _ = kernels::multi_read(dst_part, a.clone(), b.clone(), c.clone()).sync_on(&stream)?;
    eprintln!("multi_read: compiled (check IR above)");

    // Kernel 3: interleaved
    let data: Arc<Tensor<f32>> = cutile::api::ones(&[n]).sync_on(&stream)?.into();
    let out_part = cutile::api::zeros(&[n]).sync_on(&stream)?.partition([tile]);
    let _ = kernels::interleaved(out_part, data.clone()).sync_on(&stream)?;
    eprintln!("interleaved: compiled (check IR above)");

    Ok(())
}
