/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use cutile::api;
use cutile::tensor::{IntoPartition, Reshape, ToHostVec};
use cutile::tile_kernel::DeviceOp;

mod common;

#[cutile::module]
mod tensor_views_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn passthrough_1d(output: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let tile: Tile<f32, { [4] }> = load_tile_like(input, output);
        output.store(tile);
    }

    #[cutile::entry()]
    fn passthrough_2d(output: &mut Tensor<f32, { [2, 2] }>, input: &Tensor<f32, { [-1, -1] }>) {
        let tile: Tile<f32, { [2, 2] }> = load_tile_like(input, output);
        output.store(tile);
    }

    #[cutile::entry()]
    fn passthrough_1d_2(output: &mut Tensor<f32, { [2] }>, input: &Tensor<f32, { [-1] }>) {
        let tile: Tile<f32, { [2] }> = load_tile_like(input, output);
        output.store(tile);
    }

    /// Copies a 4-element 1D slice to output. Used to verify sliced views
    /// with offset produce correct data.
    #[cutile::entry()]
    fn copy_4(output: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let tile: Tile<f32, { [4] }> = load_tile_like(input, output);
        output.store(tile);
    }

    /// Copies a 2x2 tile. Used to verify 2D sliced views with non-contiguous
    /// strides produce correct data.
    #[cutile::entry()]
    fn copy_2x2(output: &mut Tensor<f32, { [2, 2] }>, input: &Tensor<f32, { [-1, -1] }>) {
        let tile: Tile<f32, { [2, 2] }> = load_tile_like(input, output);
        output.store(tile);
    }
}

#[test]
fn arc_views_are_zero_copy() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
        let view = base.reshape(&[2, 4]).unwrap();
        let dyn_view = base.reshape(&[4, 2]).unwrap();
        let flat = base.reshape(&[8]).unwrap();

        assert_eq!(
            base.device_pointer().cu_deviceptr(),
            view.device_pointer().cu_deviceptr()
        );
        assert_eq!(
            base.device_pointer().cu_deviceptr(),
            dyn_view.device_pointer().cu_deviceptr()
        );
        assert_eq!(
            base.device_pointer().cu_deviceptr(),
            flat.device_pointer().cu_deviceptr()
        );
        assert_eq!(view.shape(), vec![2, 4]);
        assert_eq!(view.strides(), vec![4, 1]);
        assert_eq!(dyn_view.shape(), vec![4, 2]);
        assert_eq!(dyn_view.strides(), vec![2, 1]);
        assert_eq!(flat.shape(), vec![8]);
        assert_eq!(flat.strides(), vec![1]);

        let base_host: Vec<f32> = (&base).to_host_vec().sync().expect("Failed.");
        let flat_host: Vec<f32> = flat.to_host_vec().sync().expect("Failed.");
        assert_eq!(base_host, flat_host);

        let copied = base.dup().sync().expect("Failed.");
        assert_ne!(
            base.device_pointer().cu_deviceptr(),
            copied.device_pointer().cu_deviceptr()
        );
    });
}

#[test]
fn invalid_view_shape_is_rejected() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
        assert!(base.reshape(&[5]).is_err());
        assert!(base.reshape(&[2, 2]).is_err());
    });
}

#[test]
fn shared_storage_blocks_mutable_partition() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
        let _view = base.reshape(&[2, 4]).unwrap();
        let owned = Arc::try_unwrap(base).expect("Expected unique outer Arc.");
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = owned.partition([8]);
        }));
        assert!(result.is_err(), "Expected mutable partition to be rejected");
    });
}

#[test]
fn arc_views_work_with_different_rank_kernels() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(4).sync().expect("Failed."));
        let input_1d = base.reshape(&[4]).unwrap();
        let input_2d = base.reshape(&[2, 2]).unwrap();

        let output_1d = api::zeros::<f32>(&[4]).sync().expect("Failed.");
        let (result_1d, _input_1d) =
            tensor_views_module::passthrough_1d(output_1d.partition([4]), input_1d)
                .sync()
                .expect("Failed.");
        let host_1d: Vec<f32> = result_1d
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("Failed.");

        let output_2d = api::zeros::<f32>(&[2, 2]).sync().expect("Failed.");
        let (result_2d, _input_2d) =
            tensor_views_module::passthrough_2d(output_2d.partition([2, 2]), input_2d)
                .sync()
                .expect("Failed.");
        let host_2d: Vec<f32> = result_2d
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("Failed.");

        assert_eq!(host_1d, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(host_2d, vec![0.0, 1.0, 2.0, 3.0]);
    });
}

// ── Slice tests ─────────────────────────────────────────────────────────────

#[test]
fn slice_1d_offset() {
    // arange(8) = [0, 1, 2, 3, 4, 5, 6, 7]
    // slice [4..8] = [4, 5, 6, 7]
    // Pass through a kernel and verify.
    common::with_test_stack(|| {
        let tensor = api::arange::<f32>(8).sync().expect("Failed.");
        let sliced = tensor.slice(&[4..8]).expect("Failed.");

        assert_eq!(sliced.shape(), &[4]);
        assert_eq!(sliced.strides(), &[1]);

        let output = api::zeros::<f32>(&[4]).sync().expect("Failed.");
        let (result, _) = tensor_views_module::copy_4(output.partition([4]), &sliced)
            .sync()
            .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![4.0, 5.0, 6.0, 7.0]);
    });
}

#[test]
fn slice_1d_middle() {
    // arange(8) = [0, 1, 2, 3, 4, 5, 6, 7]
    // slice [2..6] = [2, 3, 4, 5]
    common::with_test_stack(|| {
        let tensor = api::arange::<f32>(8).sync().expect("Failed.");
        let sliced = tensor.slice(&[2..6]).expect("Failed.");

        assert_eq!(sliced.shape(), &[4]);

        let output = api::zeros::<f32>(&[4]).sync().expect("Failed.");
        let (result, _) = tensor_views_module::copy_4(output.partition([4]), &sliced)
            .sync()
            .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![2.0, 3.0, 4.0, 5.0]);
    });
}

#[test]
fn slice_2d_rows_only() {
    // arange(16).view([4,4]):
    //   row 0: [0,  1,  2,  3]
    //   row 1: [4,  5,  6,  7]
    //   row 2: [8,  9, 10, 11]
    //   row 3: [12, 13, 14, 15]
    // slice [1..3, 0..2] -> [[4,5], [8,9]]
    // shape [2,2], strides [4,1] — non-contiguous (stride[0]=4, not 2).
    common::with_test_stack(|| {
        let tensor = api::arange::<f32>(16).sync().expect("Failed.");
        let tensor_2d = tensor.view(&[4, 4]).expect("Failed.");
        let sliced = tensor_2d.slice(&[1..3, 0..2]).expect("Failed.");

        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.strides(), &[4, 1]); // Non-contiguous.

        let output = api::zeros::<f32>(&[2, 2]).sync().expect("Failed.");
        let (result, _) = tensor_views_module::copy_2x2(output.partition([2, 2]), &sliced)
            .sync()
            .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![4.0, 5.0, 8.0, 9.0]);
    });
}

#[test]
fn slice_2d_columns_non_contiguous() {
    // arange(16).view([4,4]):
    //   row 0: [0,  1,  2,  3]
    //   row 1: [4,  5,  6,  7]
    //   row 2: [8,  9, 10, 11]
    //   row 3: [12, 13, 14, 15]
    // slice [0..2, 2..4] (first 2 rows, columns 2-3):
    //   [[2, 3], [6, 7]]
    // shape [2,2], strides [4,1] — non-contiguous.
    common::with_test_stack(|| {
        let tensor = api::arange::<f32>(16).sync().expect("Failed.");
        let tensor_2d = tensor.view(&[4, 4]).expect("Failed.");
        let sliced = tensor_2d.slice(&[0..2, 2..4]).expect("Failed.");

        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.strides(), &[4, 1]); // Non-contiguous.

        let output = api::zeros::<f32>(&[2, 2]).sync().expect("Failed.");
        let (result, _) = tensor_views_module::copy_2x2(output.partition([2, 2]), &sliced)
            .sync()
            .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![2.0, 3.0, 6.0, 7.0]);
    });
}

#[test]
fn slice_chained() {
    // arange(16).view([4,4]):
    //   [0,  1,  2,  3]
    //   [4,  5,  6,  7]
    //   [8,  9, 10, 11]
    //   [12, 13, 14, 15]
    // slice [1..3] -> rows 1-2: [[4,5,6,7], [8,9,10,11]]
    // then slice [:, 1..3] -> cols 1-2: [[5,6], [9,10]]
    common::with_test_stack(|| {
        let tensor = api::arange::<f32>(16).sync().expect("Failed.");
        let tensor_2d = tensor.view(&[4, 4]).expect("Failed.");
        let row_slice = tensor_2d.slice(&[1..3]).expect("Failed.");
        let final_slice = row_slice.slice(&[0..2, 1..3]).expect("Failed.");

        assert_eq!(final_slice.shape(), &[2, 2]);
        assert_eq!(final_slice.strides(), &[4, 1]);

        let output = api::zeros::<f32>(&[2, 2]).sync().expect("Failed.");
        let (result, _) = tensor_views_module::copy_2x2(output.partition([2, 2]), &final_slice)
            .sync()
            .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![5.0, 6.0, 9.0, 10.0]);
    });
}

#[test]
#[allow(clippy::reversed_empty_ranges)] // `5..3` is the rejected input under test
fn slice_out_of_bounds_rejected() {
    common::with_test_stack(|| {
        let tensor = api::arange::<f32>(8).sync().expect("Failed.");
        assert!(tensor.slice(&[0..9]).is_err());
        assert!(tensor.slice(&[5..3]).is_err());

        let tensor_2d = tensor.view(&[2, 4]).expect("Failed.");
        assert!(tensor_2d.slice(&[0..3]).is_err()); // axis 0 has dim 2
        assert!(tensor_2d.slice(&[0..2, 0..5]).is_err()); // axis 1 has dim 4
    });
}

#[test]
fn slice_preserves_view_reshape_rejection() {
    // A non-contiguous slice should reject reshape via view().
    common::with_test_stack(|| {
        let tensor = api::arange::<f32>(16).sync().expect("Failed.");
        let tensor_2d = tensor.view(&[4, 4]).expect("Failed.");
        let col_slice = tensor_2d.slice(&[0..4, 1..3]).expect("Failed.");

        // col_slice has strides [4, 1] but shape [4, 2] — non-contiguous.
        assert!(
            col_slice.view(&[8]).is_err(),
            "reshape on non-contiguous view should fail"
        );
    });
}

// ── Slice metadata (CPU-only, no kernel launch) ───────────────────────────
//
// Inspired by numpy: verify shape, strides, and offset for various
// slicing patterns without launching GPU kernels.

#[test]
fn slice_full_range_is_identity() {
    // numpy: a[0:n] is a view with same shape
    common::with_test_stack(|| {
        let a = api::arange::<f32>(16).sync().expect("Failed.");
        let v = a.slice(&[0..16]).expect("Failed.");
        assert_eq!(v.shape(), &[16]);
        assert_eq!(v.strides(), &[1]);
    });
}

#[test]
fn slice_single_element() {
    // numpy: a[5:6] → shape [1]
    common::with_test_stack(|| {
        let a = api::arange::<f32>(16).sync().expect("Failed.");
        let v = a.slice(&[5..6]).expect("Failed.");
        assert_eq!(v.shape(), &[1]);
    });
}

#[test]
fn slice_2d_partial_axes() {
    // numpy: a[1:3] on a 4x4 → rows 1-2, all columns preserved
    common::with_test_stack(|| {
        let a = api::arange::<f32>(16).sync().expect("Failed.");
        let a2d = a.view(&[4, 4]).expect("Failed.");
        let v = a2d.slice(&[1..3]).expect("Failed.");
        assert_eq!(v.shape(), &[2, 4]);
        assert_eq!(v.strides(), &[4, 1]);
    });
}

#[test]
fn slice_chained_accumulates_offset() {
    // numpy: a[2:10][3:5] == a[5:7]
    // Verify that chained slices compose offsets correctly.
    common::with_test_stack(|| {
        let a = api::arange::<f32>(16).sync().expect("Failed.");
        let s1 = a.slice(&[2..10]).expect("Failed.");
        assert_eq!(s1.shape(), &[8]);
        let s2 = s1.slice(&[3..5]).expect("Failed.");
        assert_eq!(s2.shape(), &[2]);

        let output = api::zeros::<f32>(&[2]).sync().expect("Failed.");
        let (result, _) = tensor_views_module::passthrough_1d_2(output.partition([2]), &s2)
            .sync()
            .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        // a[5:7] = [5.0, 6.0]
        assert_eq!(host, vec![5.0, 6.0]);
    });
}
