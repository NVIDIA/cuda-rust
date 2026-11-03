/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests for CuTe operation, type, and semantic verification contracts.

use cute_layout::{ComposedLayout, IntTuple, Layout, OffsetUnit, Swizzle};
use pliron::{common_traits::Verify, context::Context};

use super::{layout::parse_composed_layout, *};

/// Printing and parsing preserve the swizzle, offset, unit, shape, and stride together.
#[test]
fn composed_layout_text_round_trips_every_addressing_piece() {
    let outer = Layout::new(
        IntTuple::Tuple(vec![IntTuple::Leaf(8), IntTuple::Leaf(32)]),
        IntTuple::Tuple(vec![IntTuple::Leaf(32), IntTuple::Leaf(1)]),
    );
    let composed =
        ComposedLayout::new(Swizzle::new(3, 4, 3), 8, outer, OffsetUnit::Elements).unwrap();
    let attr = CuteComposedLayoutAttr(composed);
    let printed = attr.to_string();
    let reparsed = parse_composed_layout(printed.trim_matches('"').to_string()).unwrap();
    assert_eq!(reparsed, attr);
}

/// Parsed swizzle bit fields cannot overlap in ways that violate the layout contract.
#[test]
fn composed_layout_text_rejects_overlapping_swizzle_fields() {
    // B=4 cannot fit into a shift distance of 3:
    //
    // [ source ]
    //      [ target ]   <- overlap, so XOR is not reversible
    assert!(parse_composed_layout("4,3,3;0;bytes;16:4".to_string()).is_err());
}

/// The textual attribute syntax preserves valid negative swizzle shifts without normalization loss.
#[test]
fn composed_layout_text_preserves_negative_cute_shifts() {
    let attr = parse_composed_layout("2,0,-3;0;bytes;16:4".to_string()).unwrap();
    assert_eq!(attr.0.inner(), Swizzle::new(2, 0, -3));
    assert_eq!(attr.to_string(), "\"2,0,-3;0;bytes;16:4\"");
}

/// Zero tile/scale widths and invalid alignment promises fail attribute verification.
#[test]
fn tensor_structure_rejects_impossible_sizes() {
    let ctx = Context::new();
    assert!(CuteTensorLayoutAttr::Contiguous1D.verify(&ctx).is_ok());
    assert!(CuteTensorLayoutAttr::Zipped1D(4).verify(&ctx).is_ok());
    assert!(CuteTensorLayoutAttr::Tile1D(0).verify(&ctx).is_err());
    assert!(
        CuteTensorLayoutAttr::BlockScaleKMajor(16)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        CuteTensorLayoutAttr::BlockScaleKMajor(0)
            .verify(&ctx)
            .is_err()
    );
    assert!(CuteScaledLayoutAttr::KTile(64).verify(&ctx).is_ok());
    assert!(CuteScaledLayoutAttr::KTile(0).verify(&ctx).is_err());
    assert!(CuteAlignmentAttr(16).verify(&ctx).is_ok());
    assert!(CuteAlignmentAttr(12).verify(&ctx).is_err());
    assert!(CuteAlignmentAttr(0).verify(&ctx).is_err());
}
