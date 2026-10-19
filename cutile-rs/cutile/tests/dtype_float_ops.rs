/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests that all float ElementTypes can:
//! 1. Create a 1D tensor of ones
//! 2. Add two such tensors in a kernel
//! 3. Convert the result to f32
//! 4. Verify correctness on the host

use cuda_core::{f8e4m3fn, f8e5m2};
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::{DeviceOp, ToHostVecOp};
use half::{bf16, f16};
use std::sync::Arc;

mod common;

#[cutile::module]
mod float_add_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add_f16<const B: i32>(
        out: &mut Tensor<f16, { [B] }>,
        a: &Tensor<f16, { [-1] }>,
        b: &Tensor<f16, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn add_bf16<const B: i32>(
        out: &mut Tensor<bf16, { [B] }>,
        a: &Tensor<bf16, { [-1] }>,
        b: &Tensor<bf16, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn add_f32<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    #[cutile::entry()]
    fn add_f64<const B: i32>(
        out: &mut Tensor<f64, { [B] }>,
        a: &Tensor<f64, { [-1] }>,
        b: &Tensor<f64, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        let tile_b = b.load_tile(shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    // Note: FP8 types (f8e4m3fn, f8e5m2) don't support direct arithmetic
    // on GPU — they're storage/quantization formats. FP8 add kernels are
    // not included here. Use convert to f16/f32 for compute.
}

use float_add_module::{add_bf16, add_f16, add_f32, add_f64};

#[test]
fn add_ones_f16_and_convert_to_f32() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<f16>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<f16>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<f16>(&[128]).sync().expect("alloc out");

        add_f16((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_f16 failed");

        let result_f32: Vec<f32> = cutile::api::convert::<f16, f32>(Arc::new(out))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-3,
                "f16: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_ones_bf16_and_convert_to_f32() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<bf16>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<bf16>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<bf16>(&[128])
            .sync()
            .expect("alloc out");

        add_bf16((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_bf16 failed");

        let result_f32: Vec<f32> = cutile::api::convert::<bf16, f32>(Arc::new(out))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-2,
                "bf16: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_ones_f32_direct() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<f32>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<f32>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc out");

        add_f32((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_f32 failed");

        let host: Vec<f32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 128);
        for (i, &v) in host.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "f32: element {i} = {v}, expected 2.0"
            );
        }
    });
}

#[test]
fn add_ones_f64_and_convert_to_f32() {
    common::with_test_stack(|| {
        let a = cutile::api::ones::<f64>(&[128]).sync().expect("alloc a");
        let b = cutile::api::ones::<f64>(&[128]).sync().expect("alloc b");
        let mut out = cutile::api::zeros::<f64>(&[128]).sync().expect("alloc out");

        add_f64((&mut out).partition([128]), &a, &b)
            .sync()
            .expect("add_f64 failed");

        let result_f32: Vec<f32> = cutile::api::convert::<f64, f32>(Arc::new(out))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "f64: element {i} = {v}, expected 2.0"
            );
        }
    });
}

// FP8 types don't support direct arithmetic (addf) on GPU — they're
// storage formats. Test the create → convert → check pipeline instead.

#[test]
fn create_ones_f8e4m3fn_and_convert_to_f32() {
    common::with_test_stack(|| {
        let ones = cutile::api::ones::<f8e4m3fn>(&[128]).sync().expect("alloc");

        let result_f32: Vec<f32> = cutile::api::convert::<f8e4m3fn, f32>(Arc::new(ones))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.5,
                "f8e4m3fn: element {i} = {v}, expected 1.0"
            );
        }
    });
}

#[test]
fn create_ones_f8e5m2_and_convert_to_f32() {
    common::with_test_stack(|| {
        let ones = cutile::api::ones::<f8e5m2>(&[128]).sync().expect("alloc");

        let result_f32: Vec<f32> = cutile::api::convert::<f8e5m2, f32>(Arc::new(ones))
            .sync()
            .expect("convert failed")
            .dup()
            .to_host_vec()
            .sync()
            .expect("to_host");

        assert_eq!(result_f32.len(), 128);
        for (i, &v) in result_f32.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.5,
                "f8e5m2: element {i} = {v}, expected 1.0"
            );
        }
    });
}
