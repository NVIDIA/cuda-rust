/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device tests for the aggregation kernels, each against the host reference
//! from `tests/aggregation.rs`'s edge-case inputs: repeated indices, an empty
//! segment, duplicate bins, negatives for signed max, and values above the
//! signed midpoint for unsigned max.
//!
//! Skips print a `SKIP Tile IR execution:` line, so an evidence harness can
//! refuse a silently green run.

use cutile::prelude::*;
use std::sync::Arc;

/// Runtime prerequisites only; mirrors the cutile gpu suite's gate.
fn supports_tile_ir() -> bool {
    use cutile_compiler::cuda_tile_runtime_utils::ToolkitCapabilities;
    use cutile_ir::ir::{Location, Module};
    let caps = ToolkitCapabilities::for_device(0).expect("selected toolkit capabilities");
    let result = caps
        .target
        .validate_module(&Module::new("test_prerequisites"))
        .and_then(|()| {
            cutile_ir::requirements::Feature::AtomicRedView.check(&caps.target, &Location::Unknown)
        });
    if let Err(error) = result {
        eprintln!("SKIP Tile IR execution: {error}");
        return false;
    }
    true
}

fn device_u32(host: Vec<u32>) -> Arc<Tensor<u32>> {
    let n = host.len();
    api::copy_host_vec_to_device(&Arc::new(host))
        .reshape(&[n])
        .sync()
        .expect("to device")
        .into()
}

fn device_f32(host: Vec<f32>) -> Arc<Tensor<f32>> {
    let n = host.len();
    api::copy_host_vec_to_device(&Arc::new(host))
        .reshape(&[n])
        .sync()
        .expect("to device")
        .into()
}

#[test]
fn scatter_add_matches_the_reference() {
    if !supports_tile_ir() {
        return;
    }
    let indices = vec![1u32, 1, 3, 1, 0, 3];
    let values = vec![1.0f32, 2.0, 5.0, 0.5, -1.0, 1.5];
    let mut expected = vec![0.0f32; 4];
    for (&i, &v) in indices.iter().zip(&values) {
        expected[i as usize] += v;
    }
    let n = indices.len() as u32;
    let mut out = api::zeros::<f32>(&[4]).sync().expect("zeros");
    let _ = cutile_kernels::scatter_add(device_u32(indices), device_f32(values), out.atomic())
        .grid((n, 1, 1))
        .sync()
        .expect("launch");
    let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, expected);
}

#[test]
fn segment_sum_keeps_an_empty_segments_seed() {
    if !supports_tile_ir() {
        return;
    }
    // Segment 1 is empty and must keep the identity seed.
    let segment_ids = vec![0u32, 0, 2, 2, 2];
    let values = vec![2.0f32, -2.0, 1.0, 1.0, 1.0];
    let n = segment_ids.len() as u32;
    let mut out = api::zeros::<f32>(&[3]).sync().expect("zeros");
    let _ = cutile_kernels::segment_sum(device_u32(segment_ids), device_f32(values), out.atomic())
        .grid((n, 1, 1))
        .sync()
        .expect("launch");
    let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, vec![0.0, 0.0, 3.0]);
}

#[test]
fn histogram_counts_duplicate_bins_exactly() {
    if !supports_tile_ir() {
        return;
    }
    let bin_ids = vec![2u32, 2, 2, 0, 1, 2];
    let n = bin_ids.len() as u32;
    let mut bins = api::zeros::<u32>(&[3]).sync().expect("zeros");
    let _ = cutile_kernels::histogram(device_u32(bin_ids), bins.atomic())
        .grid((n, 1, 1))
        .sync()
        .expect("launch");
    let host: Vec<u32> = bins.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, vec![1, 1, 4]);
}

#[test]
fn weighted_histogram_accumulates_weights() {
    if !supports_tile_ir() {
        return;
    }
    let bin_ids = vec![1u32, 1, 0, 1];
    let weights = vec![0.5f32, 0.25, 0.0, 0.25];
    let n = bin_ids.len() as u32;
    let mut bins = api::zeros::<f32>(&[2]).sync().expect("zeros");
    let _ =
        cutile_kernels::weighted_histogram(device_u32(bin_ids), device_f32(weights), bins.atomic())
            .grid((n, 1, 1))
            .sync()
            .expect("launch");
    let host: Vec<f32> = bins.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, vec![0.0, 1.0]);
}

#[test]
fn signed_max_reduce_handles_negatives() {
    if !supports_tile_ir() {
        return;
    }
    // All negative, seeded with the signed identity: the answer is the
    // largest (least negative) value, which an unsigned comparison would not
    // pick.
    let values: Vec<i32> = (0..64).map(|i| -1 - (i % 7)).collect();
    let dev: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(values))
        .reshape(&[1, 64])
        .sync()
        .expect("to device")
        .into();
    let mut out = api::full::<i32>(i32::MIN, &[1]).sync().expect("seed");
    let _ = cutile_kernels::max_reduce(dev, out.atomic())
        .grid((1, 1, 1))
        .sync()
        .expect("launch");
    let host: Vec<i32> = out.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, vec![-1]);
}

#[test]
fn unsigned_max_reduce_crosses_the_signed_midpoint() {
    if !supports_tile_ir() {
        return;
    }
    let mut values = vec![3u32; 64];
    values[17] = 0x7FFF_FFF0;
    values[41] = 0x8000_0010; // negative under a signed comparison
    let mut out = api::zeros::<u32>(&[1]).sync().expect("seed");
    let values_dev: Arc<Tensor<u32>> = api::copy_host_vec_to_device(&Arc::new(values))
        .reshape(&[1, 64])
        .sync()
        .expect("to device")
        .into();
    let _ = cutile_kernels::max_reduce(values_dev, out.atomic())
        .grid((1, 1, 1))
        .sync()
        .expect("launch");
    let host: Vec<u32> = out.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, vec![0x8000_0010]);
}

#[test]
fn bitmask_or_merges_duplicate_destinations() {
    if !supports_tile_ir() {
        return;
    }
    let dest_ids = vec![0u32, 0, 1, 0];
    let masks = vec![0b01u32, 0b10, 0b100, 0b01];
    let n = masks.len() as u32;
    let mut out = api::zeros::<u32>(&[2]).sync().expect("zeros");
    let _ = cutile_kernels::bitmask_or(device_u32(dest_ids), device_u32(masks), out.atomic())
        .grid((n, 1, 1))
        .sync()
        .expect("launch");
    let host: Vec<u32> = out.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, vec![0b11, 0b100]);
}

#[test]
fn embedding_backward_accumulates_repeated_token_ids() {
    if !supports_tile_ir() {
        return;
    }
    const D: usize = 8;
    let token_ids = vec![1u32, 1, 0, 1];
    let rows = token_ids.len();
    let grads: Vec<f32> = (0..rows * D).map(|i| (i % 5) as f32 - 2.0).collect();
    let mut expected = vec![0.0f32; 3 * D];
    for (r, &t) in token_ids.iter().enumerate() {
        for d in 0..D {
            expected[t as usize * D + d] += grads[r * D + d];
        }
    }
    let grads_dev: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(grads))
        .reshape(&[rows, D])
        .sync()
        .expect("grads")
        .into();
    let mut table = api::zeros::<f32>(&[3, D]).sync().expect("zeros");
    let _ = cutile_kernels::embedding_backward_scatter_add(
        device_u32(token_ids),
        grads_dev,
        table.atomic(),
    )
    .grid((rows as u32, 1, 1))
    .sync()
    .expect("launch");
    let host: Vec<f32> = table.dup().to_host_vec().sync().expect("to_host");
    assert_eq!(host, expected);
}
