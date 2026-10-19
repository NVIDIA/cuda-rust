/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#![allow(clippy::disallowed_methods)]

use dialect_mir::ops as mir;
use dialect_mir::types::{
    EnumCarrierKind, EnumEncoding, EnumLayoutKind, EnumVariant, MirEnumType, MirStructType,
};
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;

pub(super) fn insert_indices(ctx: &Context, inserts: &[llvm::InsertValueOp]) -> Vec<Vec<u32>> {
    inserts.iter().map(|op| op.indices(ctx)).collect()
}

pub(super) fn empty_struct_ty(ctx: &mut Context, name: &str) -> TypeHandle {
    MirStructType::get(ctx, name.to_string(), vec![], vec![]).into()
}

pub(super) fn padded_struct_with_zst_ty(ctx: &mut Context) -> (TypeHandle, TypeHandle) {
    let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let i64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signless).into();
    let zst_ty = empty_struct_ty(ctx, "Marker");

    let struct_ty = MirStructType::get_with_full_layout(
        ctx,
        "Padded".to_string(),
        vec!["a".to_string(), "marker".to_string(), "b".to_string()],
        vec![i8_ty, zst_ty, i64_ty],
        vec![0, 1, 2],
        vec![0, 1, 8],
        16,
        8,
    );

    (struct_ty.into(), zst_ty)
}

pub(super) fn append_empty_struct_value(
    ctx: &mut Context,
    block: Ptr<pliron::basic_block::BasicBlock>,
    zst_ty: TypeHandle,
) -> Value {
    let op = Operation::new(
        ctx,
        mir::MirConstructStructOp::get_concrete_op_info(),
        vec![zst_ty],
        vec![],
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

pub(super) fn unit_niche_enum(
    ctx: &mut Context,
    carrier: (EnumCarrierKind, u32, u32),
    niche_start: u128,
    niche_range: std::ops::RangeInclusive<u32>,
    untagged_variant: u32,
    inhabited: Vec<u8>,
) -> TypeHandle {
    let (carrier_kind, carrier_width, carrier_address_space) = carrier;
    let logical_width = if inhabited.len() > u8::MAX as usize {
        16
    } else {
        8
    };
    let logical_ty: TypeHandle = IntegerType::get(ctx, logical_width, Signedness::Unsigned).into();
    let variants = (0..inhabited.len())
        .map(|index| EnumVariant::unit(format!("V{index}")))
        .collect::<Vec<_>>();
    let discriminants = (0..inhabited.len() as u64).collect();
    let carrier_size = u64::from(carrier_width).div_ceil(8);
    let carrier_align = carrier_size.next_power_of_two().min(16);
    MirEnumType::get_with_encoding(
        ctx,
        "UnitNiche".into(),
        logical_ty,
        discriminants,
        variants,
        EnumEncoding {
            tag_offset: 0,
            total_size: carrier_size,
            abi_align: carrier_align,
            layout_kind: EnumLayoutKind::Niche,
            carrier_kind,
            carrier_width,
            carrier_address_space,
            niche_start,
            niche_variant_start: *niche_range.start(),
            niche_variant_end: *niche_range.end(),
            untagged_variant,
            variant_inhabited: inhabited,
            ..EnumEncoding::default()
        },
    )
    .into()
}
