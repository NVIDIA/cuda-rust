/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Ordered streaming reductions over independently laid-out tensor views.

use cute_rs::partition::StridedTensor;
use cute_rs::{Composed, RowMajor, SmemTile, Swizzle};

#[test]
fn zipped_row_reduction_preserves_coordinate_order_and_both_layouts() {
    type Layout = Composed<Swizzle<2, 0, 2>, 0, RowMajor<4, 4>>;
    let mut shared = [0.0_f32; 16];
    for row in 0..4 {
        for column in 0..4 {
            shared[cute_rs::static_layout::offset_2d::<Layout>(row, column)] =
                (row * 10 + column) as f32;
        }
    }
    let tile = SmemTile::<f32, Layout> {
        base: shared.as_mut_ptr(),
        capacity: shared.len(),
        layout: core::marker::PhantomData,
    };
    let global = [1.0_f32, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
    let global = StridedTensor::from_slice(&global).partition::<2>(0);
    let mut visited = Vec::new();
    // SAFETY: the entire logical row is initialized, in bounds and has no writer.
    let row = unsafe { tile.row_view::<4>(2, 0) };
    let result = row.zip_fold(global, 0.0, |accumulator, shared, global| {
        visited.push((shared, global));
        accumulator + shared * global
    });
    assert_eq!(
        visited,
        [(20.0, 1.0), (21.0, 2.0), (22.0, 3.0), (23.0, 4.0)]
    );
    assert_eq!(result, 220.0);
}

#[test]
fn zipped_row_reduction_does_not_reassociate() {
    let mut shared = [1.0e20_f32, 1.0, -1.0e20, 3.0];
    let tile = SmemTile::<f32, RowMajor<1, 4>> {
        base: shared.as_mut_ptr(),
        capacity: 4,
        layout: core::marker::PhantomData,
    };
    let ones = [1.0_f32; 4];
    let row = unsafe { tile.row_view::<4>(0, 0) };
    assert_eq!(
        row.zip_fold(StridedTensor::from_slice(&ones), 0.0, |a, b, c| a + b * c),
        3.0
    );
}

#[test]
fn empty_shared_row_never_calls_its_reduction() {
    let tile = SmemTile::<f32, RowMajor<1, 4>> {
        base: core::ptr::null_mut(),
        capacity: 0,
        layout: core::marker::PhantomData,
    };
    // SAFETY: an empty row has no coordinates, so it never accesses the pointer.
    let row = unsafe { tile.row_view::<0>(0, 0) };
    assert_eq!(
        row.zip_fold(
            StridedTensor::from_slice(&[]),
            9.0,
            |_: f32, _: f32, _: f32| panic!("empty")
        ),
        9.0
    );
}
