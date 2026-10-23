/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked BF16 matrix extents and FFN dimensions require no device allocation.
#![cfg(feature = "host")]

use cute_rs::ffn::{FfnError, FfnShape, RowMajorShape};

#[test]
fn dense_and_padded_rows_preserve_logical_shape_and_minimum_span() {
    let dense = RowMajorShape::dense(3, 5).unwrap();
    assert_eq!((dense.rows(), dense.cols(), dense.row_stride()), (3, 5, 5));
    assert_eq!(dense.required_len(), 15);
    let padded = RowMajorShape::new(3, 5, 8).unwrap();
    assert_eq!(padded.required_len(), 21);
    assert_eq!(padded.validate_len(20), Err(FfnError::BufferTooShort));
    assert_eq!(padded.validate_len(21), Ok(()));
    assert_eq!(padded.validate_len(24), Ok(()));
}

#[test]
fn zero_rows_need_no_storage_but_width_and_stride_remain_valid() {
    let empty = RowMajorShape::new(0, 7, 9).unwrap();
    assert_eq!(empty.required_len(), 0);
    assert_eq!(empty.validate_len(0), Ok(()));
    assert_eq!(empty.validate_len(9), Ok(()));
    assert_eq!(RowMajorShape::new(0, 0, 0), Err(FfnError::ZeroWidth));
    assert_eq!(RowMajorShape::new(0, 7, 6), Err(FfnError::StrideTooSmall));
}

#[test]
fn strided_shapes_reject_short_pitch_and_overflow_before_allocation() {
    assert_eq!(RowMajorShape::new(2, 9, 8), Err(FfnError::StrideTooSmall));
    assert_eq!(
        RowMajorShape::new(3, 1, usize::MAX),
        Err(FfnError::ExtentOverflow)
    );
    assert_eq!(
        RowMajorShape::new(2, 2, usize::MAX),
        Err(FfnError::ExtentOverflow)
    );
    assert_eq!(
        RowMajorShape::dense(usize::MAX, 2),
        Err(FfnError::ExtentOverflow)
    );
}

#[test]
fn bf16_slice_byte_extent_must_fit_isize_even_when_element_count_fits() {
    let max_elements = isize::MAX as usize / 2;
    let largest = RowMajorShape::dense(1, max_elements).unwrap();
    assert_eq!(largest.required_len(), max_elements);
    assert_eq!(largest.validate_len(max_elements), Ok(()));
    assert_eq!(
        largest.validate_len(max_elements + 1),
        Err(FfnError::ExtentOverflow)
    );
    assert_eq!(
        RowMajorShape::dense(1, max_elements + 1),
        Err(FfnError::ExtentOverflow)
    );
    assert_eq!(
        RowMajorShape::dense(1, 1).unwrap().validate_len(usize::MAX),
        Err(FfnError::ExtentOverflow)
    );
}

#[test]
fn model_shapes_keep_input_pitch_separate_from_dense_workspace() {
    let talker = FfnShape::<2048, 6144>::new(3, 2051).unwrap();
    assert_eq!(talker.rows(), 3);
    assert_eq!(talker.input(), RowMajorShape::new(3, 2048, 2051).unwrap());
    assert_eq!(talker.hidden(), RowMajorShape::dense(3, 2048).unwrap());
    assert_eq!(
        talker.intermediate(),
        RowMajorShape::dense(3, 6144).unwrap()
    );
    let predictor = FfnShape::<1024, 3072>::new(1, 1024).unwrap();
    assert_eq!(predictor.hidden().required_len(), 1024);
    assert_eq!(predictor.intermediate().required_len(), 3072);
}

#[test]
fn ffn_rejects_zero_dimensions_short_inputs_and_workspace_overflow() {
    assert_eq!(FfnShape::<0, 7>::new(1, 1), Err(FfnError::ZeroWidth));
    assert_eq!(FfnShape::<7, 0>::new(1, 7), Err(FfnError::ZeroWidth));
    assert_eq!(FfnShape::<7, 13>::new(1, 6), Err(FfnError::StrideTooSmall));
    assert_eq!(
        FfnShape::<1, 2>::new(isize::MAX as usize / 2, 1),
        Err(FfnError::ExtentOverflow)
    );
    let empty = FfnShape::<7, 13>::new(0, 7).unwrap();
    assert_eq!(empty.hidden().required_len(), 0);
    assert_eq!(empty.intermediate().required_len(), 0);
}
