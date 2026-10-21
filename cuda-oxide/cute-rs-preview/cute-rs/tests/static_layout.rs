/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Static offset-span validation, including layouts with a tiny logical extent.
use cute_rs::markers::{C, Composed, L, RowMajor, Swizzle, T2};
use cute_rs::static_layout::StaticLayout;

#[test]
fn valid_boundary_offsets_and_swizzle_bounds_are_representable() {
    type Boundary = L<C<2>, C<{ i64::MAX }>>;
    assert_eq!(Boundary::offset(1), i64::MAX as usize);
    assert_eq!(Boundary::MAX_OFFSET, i64::MAX as usize);
    type Pitched = L<T2<C<2>, C<3>>, T2<C<17>, C<4>>>;
    assert_eq!(Pitched::MAX_OFFSET, 25);
    type ComposedMap = Composed<Swizzle<2, 1, -3>, 7, RowMajor<3, 5>>;
    for i in 0..ComposedMap::SIZE {
        assert!(ComposedMap::offset(i) <= ComposedMap::MAX_OFFSET);
    }
}

#[test]
fn overflowing_physical_layouts_fail_before_device_execution() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/static_layout_leaf_overflow.rs");
    tests.compile_fail("tests/compile_fail/static_layout_sum_overflow.rs");
    tests.compile_fail("tests/compile_fail/static_layout_composed_overflow.rs");
    tests.compile_fail("tests/compile_fail/static_layout_2d_overflow.rs");
}
