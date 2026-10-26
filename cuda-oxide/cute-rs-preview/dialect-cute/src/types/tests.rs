/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests for CuTe operation, type, and semantic verification contracts.

use dialect_mir::types::{MirArrayType, MirFP16Type, MirStructType, MirTupleType, StructAbiKind};
use pliron::{
    builtin::types::{FP32Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    r#type::TypeHandle,
};

use super::*;
use crate::{
    attributes::{
        CuteScaledLayoutAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr,
        CuteTensorFormatAttr, CuteTensorLayoutAttr, CuteTensorRoleAttr,
    },
    layout::{ComposedLayout, Layout, OffsetUnit, Swizzle},
};

/// The tensor type retains element, storage, access, alignment, and selection information together.
#[test]
fn tensor_view_records_the_whole_elementwise_view() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let view = CuteTensorViewType::get(
        &mut ctx,
        f32_ty,
        f32_ty,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadOnly,
        4,
        CuteTensorLayoutAttr::Tile1D(4),
    );

    assert_eq!(view.deref(&ctx).selected_tile_size(), Some(4));
    assert!(view.deref(&ctx).verify(&ctx).is_ok());
}

/// Unsupported formats, storage types, or selection sizes fail type verification before lowering.
#[test]
fn tensor_view_rejects_invalid_v0_facts() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();

    let mismatched = CuteTensorViewType::get(
        &mut ctx,
        f32_ty,
        f16_ty,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadOnly,
        4,
        CuteTensorLayoutAttr::Contiguous1D,
    );
    assert!(
        mismatched
            .deref(&ctx)
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("matching logical and storage")
    );

    let zero_tile = CuteTensorViewType::get(
        &mut ctx,
        f32_ty,
        f32_ty,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadOnly,
        4,
        CuteTensorLayoutAttr::Tile1D(0),
    );
    assert!(
        zero_tile
            .deref(&ctx)
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("tile size")
    );
}

/// Transparent scalar wrappers preserve their logical identity while copying scalar bits.
#[test]
fn tensor_view_accepts_transparent_scalar_storage() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let u16_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
    let mut element = u16_ty;
    // Nested transparent wrappers are equally valid; no defining type name
    // should be required to copy their unchanged physical representation.
    for name in ["BFloat16", "TypedBFloat16"] {
        element = MirStructType::get_with_full_layout_and_abi(
            &mut ctx,
            name.into(),
            vec!["0".into()],
            vec![element],
            vec![0],
            vec![0],
            2,
            2,
            StructAbiKind::TransparentScalar,
        )
        .into();
        let view = CuteTensorViewType::get(
            &mut ctx,
            element,
            element,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadWrite,
            2,
            CuteTensorLayoutAttr::Tile1D(8),
        );
        assert_eq!(view.deref(&ctx).storage_bytes(&ctx), Some(2));
        assert!(view.deref(&ctx).verify(&ctx).is_ok());
        assert_eq!(view.deref(&ctx).logical, element);
        assert_eq!(view.deref(&ctx).storage, element);
    }
}

/// A transparent ABI flag alone cannot authorize a scalar copy with a different layout.
#[test]
fn tensor_view_rejects_unproven_scalar_wrappers() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let u16_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
    for (name, count, offsets, size, alignment, abi) in [
        ("ordinary", 1, vec![0], 2, 2, StructAbiKind::Aggregate),
        (
            "padding",
            1,
            vec![0],
            4,
            2,
            StructAbiKind::TransparentScalar,
        ),
        ("offset", 1, vec![1], 2, 2, StructAbiKind::TransparentScalar),
        (
            "alignment",
            1,
            vec![0],
            2,
            4,
            StructAbiKind::TransparentScalar,
        ),
        (
            "extra_field",
            2,
            vec![0, 0],
            2,
            2,
            StructAbiKind::TransparentScalar,
        ),
        (
            "unknown_offset",
            1,
            vec![],
            2,
            2,
            StructAbiKind::TransparentScalar,
        ),
        (
            "unknown_size",
            1,
            vec![0],
            0,
            2,
            StructAbiKind::TransparentScalar,
        ),
        (
            "unknown_alignment",
            1,
            vec![0],
            2,
            0,
            StructAbiKind::TransparentScalar,
        ),
    ] {
        let element: TypeHandle = MirStructType::get_with_full_layout_and_abi(
            &mut ctx,
            name.into(),
            (0..count).map(|i| i.to_string()).collect(),
            vec![u16_ty; count],
            (0..count).collect(),
            offsets,
            size,
            alignment,
            abi,
        )
        .into();
        let view = CuteTensorViewType::get(
            &mut ctx,
            element,
            element,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            4,
            CuteTensorLayoutAttr::Contiguous1D,
        );
        assert!(view.deref(&ctx).verify(&ctx).is_err(), "accepted {name}");
    }
}

/// Packed value/scale formats and coordinate roles survive row selection and fragment loading.
#[test]
fn packed_views_keep_format_role_scale_and_fragment_facts() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();

    let values: TypeHandle = CuteTensorViewType::get_with_facts(
        &mut ctx,
        f32_ty,
        u8_ty,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadOnly,
        16,
        CuteTensorFormatAttr::E2M1,
        CuteTensorRoleAttr::Mkl,
        CuteTensorLayoutAttr::KMajor,
    )
    .into();
    let scales: TypeHandle = CuteTensorViewType::get_with_facts(
        &mut ctx,
        f32_ty,
        u8_ty,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadOnly,
        4,
        CuteTensorFormatAttr::UE8M0,
        CuteTensorRoleAttr::Mkl,
        CuteTensorLayoutAttr::BlockScaleKMajor(16),
    )
    .into();
    let tile: TypeHandle = CuteScaledViewType::get(
        &mut ctx,
        values,
        scales,
        CuteTensorRoleAttr::Mkl,
        CuteScaledLayoutAttr::KTile(64),
    )
    .into();
    let fragment = CuteFragmentType::get(&mut ctx, tile);

    assert!(values.deref(&ctx).verify(&ctx).is_ok());
    assert!(scales.deref(&ctx).verify(&ctx).is_ok());
    assert!(tile.deref(&ctx).verify(&ctx).is_ok());
    assert!(fragment.deref(&ctx).verify(&ctx).is_ok());
    let source = fragment.deref(&ctx).source_view(&ctx).unwrap();
    assert_eq!(source.role, CuteTensorRoleAttr::Mkl);
    assert_eq!(source.values_per_scale(&ctx), Some(16));
    assert_eq!(source.k_width(), Some(64));
}

/// Shared packed data retains its physical f16/u32 carrier types rather than pretending to store decoded FP32.
#[test]
fn shared_packed_views_keep_their_real_transport_carriers() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

    let values: TypeHandle = CuteTensorViewType::get_with_facts(
        &mut ctx,
        f32_ty,
        f16_ty,
        CuteTensorAddressSpaceAttr::Smem,
        CuteTensorAccessAttr::ReadOnly,
        16,
        CuteTensorFormatAttr::E2M1,
        CuteTensorRoleAttr::Mkl,
        CuteTensorLayoutAttr::KMajor,
    )
    .into();
    let scales: TypeHandle = CuteTensorViewType::get_with_facts(
        &mut ctx,
        f32_ty,
        u32_ty,
        CuteTensorAddressSpaceAttr::Smem,
        CuteTensorAccessAttr::ReadOnly,
        4,
        CuteTensorFormatAttr::UE8M0,
        CuteTensorRoleAttr::Mkl,
        CuteTensorLayoutAttr::BlockScaleKMajor(32),
    )
    .into();
    assert!(values.deref(&ctx).verify(&ctx).is_ok());
    assert!(scales.deref(&ctx).verify(&ctx).is_ok());

    let data_inner: Layout = "(128,32):(32,1)".parse().unwrap();
    let data_placement =
        ComposedLayout::new(Swizzle::new(2, 3, 3), 0, data_inner, OffsetUnit::Elements).unwrap();
    let shared = CuteSmemTensorType::get(&mut ctx, values, data_placement);
    assert!(shared.deref(&ctx).verify(&ctx).is_ok());
    assert_eq!(shared.deref(&ctx).storage_elements(), Some(4096));
    assert_eq!(shared.deref(&ctx).storage_bytes(&ctx), Some(8192));

    let wrong_shared_carrier = CuteTensorViewType::get_with_facts(
        &mut ctx,
        f32_ty,
        u8_ty,
        CuteTensorAddressSpaceAttr::Smem,
        CuteTensorAccessAttr::ReadOnly,
        1,
        CuteTensorFormatAttr::E2M1,
        CuteTensorRoleAttr::Mkl,
        CuteTensorLayoutAttr::KMajor,
    );
    assert!(wrong_shared_carrier.deref(&ctx).verify(&ctx).is_err());
}

/// Register counting traverses nested arrays/structs while zero-sized marker fields contribute no registers.
#[test]
fn mma_carrier_shape_walks_nested_rust_aggregates_and_ignores_markers() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
    let words: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 4).into();
    let marker: TypeHandle =
        MirStructType::get(&mut ctx, "PhantomData<Role>".into(), vec![], vec![]).into();
    let fragment: TypeHandle = MirTupleType::get(&mut ctx, vec![words, marker]).into();
    let two_fragments: TypeHandle = MirArrayType::get(&mut ctx, fragment, 2).into();
    assert_eq!(
        mma_carrier_shape(&ctx, two_fragments),
        Some(MmaCarrierShape::u32(8))
    );

    let mixed: TypeHandle = MirTupleType::get(&mut ctx, vec![two_fragments, f32_ty]).into();
    assert_eq!(
        mma_carrier_shape(&ctx, mixed),
        Some(MmaCarrierShape {
            u32_registers: 8,
            f32_registers: 1,
        })
    );
    assert_eq!(mma_carrier_shape(&ctx, mixed).unwrap().bytes(), Some(36));
}
