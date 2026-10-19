/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for arange correctness and multi-block immutable tensor reads.
//!
//! These tests verify that `api::arange` produces correct sequential values
//! across multiple thread blocks, and that multi-block reads from immutable
//! tensors return data from the correct addresses.

use cutile::api;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::{DeviceOp, ToHostVecOp};

mod common;

#[cutile::module]
mod test_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn copy_immutable<const B: i32>(out: &mut Tensor<i32, { [B] }>, a: &Tensor<i32, { [-1] }>) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape![B], [pid]);
        out.store(tile_a);
    }
}

use test_kernels::copy_immutable;

// ── arange correctness ────────────────────────────────────────────────────

#[test]
fn arange_i32_256() {
    common::with_test_stack(|| {
        let a = api::arange::<i32>(256).sync().expect("arange");
        let host: Vec<i32> = a.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 256);
        for (i, &v) in host.iter().enumerate() {
            assert_eq!(v, i as i32, "arange i32: element {i}");
        }
    });
}

#[test]
fn arange_f32_512() {
    common::with_test_stack(|| {
        let a = api::arange::<f32>(512).sync().expect("arange");
        let host: Vec<f32> = a.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 512);
        for (i, &v) in host.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-5, "arange f32: element {i} = {v}");
        }
    });
}

#[test]
fn arange_i32_1024() {
    common::with_test_stack(|| {
        let a = api::arange::<i32>(1024).sync().expect("arange");
        let host: Vec<i32> = a.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 1024);
        for (i, &v) in host.iter().enumerate() {
            assert_eq!(v, i as i32, "arange i32 1024: element {i}");
        }
    });
}

// ── multi-block immutable copy ────────────────────────────────────────────

fn run_copy_test(size: usize, block: usize) {
    common::with_test_stack(move || {
        let a = api::arange::<i32>(size).sync().expect("arange");
        let mut out = api::zeros::<i32>(&[size]).sync().expect("zeros");

        copy_immutable((&mut out).partition([block]), &a)
            .sync()
            .expect("copy failed");

        let host: Vec<i32> = out.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), size);
        for (i, &v) in host.iter().enumerate() {
            assert_eq!(v, i as i32, "copy size={size} block={block}: element {i}");
        }
    });
}

#[test]
fn copy_256_b128() {
    run_copy_test(256, 128);
}

#[test]
fn copy_512_b128() {
    run_copy_test(512, 128);
}

#[test]
fn copy_1024_b128() {
    run_copy_test(1024, 128);
}

// ── eye ───────────────────────────────────────────────────────────────────

#[test]
fn eye_4x4() {
    common::with_test_stack(|| {
        let m = api::eye(4).sync().expect("eye");
        let host: Vec<f32> = m.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 16);
        for r in 0..4 {
            for c in 0..4 {
                let expected = if r == c { 1.0 } else { 0.0 };
                let v = host[r * 4 + c];
                assert!(
                    (v - expected).abs() < 1e-5,
                    "eye(4)[{r},{c}] = {v}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn eye_32x32() {
    common::with_test_stack(|| {
        let m = api::eye(32).sync().expect("eye");
        let host: Vec<f32> = m.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 1024);
        for r in 0..32 {
            for c in 0..32 {
                let expected = if r == c { 1.0 } else { 0.0 };
                let v = host[r * 32 + c];
                assert!(
                    (v - expected).abs() < 1e-5,
                    "eye(32)[{r},{c}] = {v}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn eye_rect_16x32() {
    // numpy: eye(16, 32) — tall → wide, diagonal on first 16 cols
    common::with_test_stack(|| {
        let (rows, cols) = (16, 32);
        let m = api::eye_rect(rows, cols).sync().expect("eye_rect");
        let host: Vec<f32> = m.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let expected = if r == c { 1.0 } else { 0.0 };
                let v = host[r * cols + c];
                assert!(
                    (v - expected).abs() < 1e-5,
                    "eye_rect({rows},{cols})[{r},{c}] = {v}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn eye_rect_32x16() {
    // numpy: eye(32, 16) — wide → tall, diagonal on first 16 rows
    common::with_test_stack(|| {
        let (rows, cols) = (32, 16);
        let m = api::eye_rect(rows, cols).sync().expect("eye_rect");
        let host: Vec<f32> = m.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let expected = if r == c { 1.0 } else { 0.0 };
                let v = host[r * cols + c];
                assert!(
                    (v - expected).abs() < 1e-5,
                    "eye_rect({rows},{cols})[{r},{c}] = {v}, expected {expected}"
                );
            }
        }
    });
}

// ── linspace ──────────────────────────────────────────────────────────────

#[test]
fn linspace_basic() {
    // linspace(0, 10, 8): 8 evenly spaced values from 0 to 10
    common::with_test_stack(|| {
        let n = 8;
        let t = api::linspace(0.0, 10.0, n).sync().expect("linspace");
        let host: Vec<f32> = t.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        let step = 10.0 / (n - 1) as f32;
        for (i, &v) in host.iter().enumerate() {
            let expected = i as f32 * step;
            assert!(
                (v - expected).abs() < 1e-4,
                "linspace[{i}] = {v}, expected {expected}"
            );
        }
    });
}

#[test]
fn linspace_single() {
    // numpy: linspace(5, 5, 1) = [5.0]
    common::with_test_stack(|| {
        let t = api::linspace(5.0, 5.0, 1).sync().expect("linspace");
        let host: Vec<f32> = t.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), 1);
        assert!((host[0] - 5.0).abs() < 1e-5);
    });
}

#[test]
fn linspace_256() {
    // numpy: linspace(0, 1, 256) — multi-block, endpoints correct
    common::with_test_stack(|| {
        let n = 256;
        let t = api::linspace(0.0, 1.0, n).sync().expect("linspace");
        let host: Vec<f32> = t.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        assert!((host[0] - 0.0).abs() < 1e-5, "start: {}", host[0]);
        assert!((host[n - 1] - 1.0).abs() < 1e-4, "end: {}", host[n - 1]);
        // Monotonically increasing
        for i in 1..n {
            assert!(host[i] >= host[i - 1], "not monotonic at {i}");
        }
    });
}

#[test]
fn linspace_negative_range() {
    // linspace(10, -10, 8): descending range
    common::with_test_stack(|| {
        let n = 8;
        let t = api::linspace(10.0, -10.0, n).sync().expect("linspace");
        let host: Vec<f32> = t.dup().to_host_vec().sync().expect("to_host");
        assert_eq!(host.len(), n);
        let step = -20.0 / (n - 1) as f32;
        for (i, &v) in host.iter().enumerate() {
            let expected = 10.0 + i as f32 * step;
            assert!(
                (v - expected).abs() < 1e-4,
                "linspace[{i}] = {v}, expected {expected}"
            );
        }
    });
}
