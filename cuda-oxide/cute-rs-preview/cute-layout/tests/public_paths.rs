/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Public-path compatibility across the responsibility-based layout modules.

use cute_layout::{IntTuple, Layout, ParseLayoutError, algebra, crd2idx};
use cute_layout::{layout, parse, tuple};

/// Grouped paths must expose the same types and mappings as the existing imports.
#[test]
fn grouped_and_compatibility_paths_share_layout_contracts() {
    let layout: layout::Layout = "(2,3):(3,1)".parse().unwrap();
    let coordinate: tuple::IntTuple = IntTuple::Tuple(vec![IntTuple::Leaf(1), IntTuple::Leaf(2)]);
    let root_layout: &Layout = &layout;
    assert_eq!(root_layout.checked_call(&coordinate), Some(5));
    assert_eq!(crd2idx(&coordinate, &layout.shape, &layout.stride), 5);
    assert_eq!(
        layout::crd2idx(&coordinate, &layout.shape, &layout.stride),
        5
    );

    let error: parse::ParseLayoutError = "(2,3):1".parse::<Layout>().unwrap_err();
    let root_error: ParseLayoutError = error;
    assert_eq!(root_error.to_string(), "invalid CuTe layout/tuple syntax");
}

/// Algebra families must remain interoperable with the original algebra imports.
#[test]
fn grouped_algebra_paths_preserve_coordinate_maps() {
    let whole = Layout::contiguous(8);
    let tile = Layout::contiguous(4);
    let divided = algebra::tiling::logical_divide(&whole, &tile).unwrap();
    let divided_compat = algebra::logical_divide(&whole, &tile).unwrap();
    assert_eq!(divided, divided_compat);
    assert_eq!(divided.to_string(), "(4,2):(1,4)");
    assert_eq!(algebra::mapping::coalesce(&divided), whole);
    assert_eq!(algebra::coalesce(&divided), whole);
    let zipped = algebra::tiling::zipped_divide(&whole, &[4]).unwrap();
    assert_eq!(zipped, algebra::zipped_divide(&whole, &[4]).unwrap());
    assert_eq!(zipped.to_string(), "(4,2):(1,4)");
    let nested: Layout = "((2,4),3):((1,2),8)".parse().unwrap();
    assert_eq!(
        algebra::mapping::flatten(&nested),
        algebra::flatten(&nested)
    );
    assert_eq!(algebra::flatten(&nested).to_string(), "(2,4,3):(1,2,8)");

    let thread: Layout = "(2,3):(3,1)".parse().unwrap();
    let value: Layout = "(2,2):(2,1)".parse().unwrap();
    let (shape, tv) = algebra::thread_value::make_layout_tv(&thread, &value).unwrap();
    let tv_compat = algebra::make_layout_tv(&thread, &value).unwrap();
    assert_eq!(shape, vec![4, 6]);
    assert_eq!(tv.to_string(), "((3,2),(2,2)):((8,2),(4,1))");
    assert_eq!((shape, tv), tv_compat);
}
