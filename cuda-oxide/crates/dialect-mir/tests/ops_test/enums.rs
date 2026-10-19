/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::{FieldIndexAttr, VariantIndexAttr},
    ops::{MirConstructEnumOp, MirEnumPayloadOp, MirGetDiscriminantOp, MirSetDiscriminantOp},
    types::{EnumVariant, MirEnumType, MirPtrType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn test_mir_set_discriminant_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signed);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let unit = |name: &str| EnumVariant::unit(name.to_string());

    let enum_ty = MirEnumType::get(
        &mut ctx,
        "DeviceState".to_string(),
        i8_ty.into(),
        vec![0, 1],
        vec![
            unit("Empty"),
            EnumVariant::new("Full".to_string(), vec![i32_ty.into()]),
        ],
    );

    let enum_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty.into(), true);
    let blk = BasicBlock::new(&mut ctx, None, vec![enum_ptr_ty.into(), i8_ty.into()]);
    let enum_ptr = blk.deref(&ctx).get_argument(0);
    let discr_val = blk.deref(&ctx).get_argument(1);

    // Malformed IR must be diagnosed without panicking, regardless of whether
    // the generated operand-count interface happens to run first.
    let op_no_operands = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let no_operands = MirSetDiscriminantOp::new(op_no_operands);
    no_operands.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(no_operands.verify(&ctx).is_err(), "zero operands rejected");

    let op_extra_operand = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr, discr_val],
        vec![],
        0,
    );
    let extra_operand = MirSetDiscriminantOp::new(op_extra_operand);
    extra_operand.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(
        extra_operand.verify(&ctx).is_err(),
        "extra operand rejected"
    );

    // Valid: pointer to enum plus the semantic target variant attribute.
    let op_valid = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr],
        vec![],
        0,
    );
    let valid = MirSetDiscriminantOp::new(op_valid);
    valid.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(1));
    valid.set_attr_set_discriminant_enum_ty(&ctx, TypeAttr::new(enum_ty.into()));
    assert!(valid.verify(&ctx).is_ok(), "Valid set_discriminant");

    // Invalid: first operand is not a pointer.
    let op_bad_ptr = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![discr_val],
        vec![],
        0,
    );
    assert!(
        MirSetDiscriminantOp::new(op_bad_ptr).verify(&ctx).is_err(),
        "Non-pointer enum operand rejected"
    );

    // Invalid: SetDiscriminant writes memory and therefore cannot accept an
    // immutable pointer even when the pointee is the right enum.
    let immutable_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty.into(), false);
    let immutable_block = BasicBlock::new(&mut ctx, None, vec![immutable_ptr_ty.into()]);
    let immutable_ptr = immutable_block.deref(&ctx).get_argument(0);
    let op_immutable = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![immutable_ptr],
        vec![],
        0,
    );
    let immutable = MirSetDiscriminantOp::new(op_immutable);
    immutable.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(
        immutable.verify(&ctx).is_err(),
        "immutable pointer rejected"
    );

    // Invalid: pointer does not point to an enum.
    let i32_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let blk_i32 = BasicBlock::new(&mut ctx, None, vec![i32_ptr_ty.into(), i8_ty.into()]);
    let i32_ptr = blk_i32.deref(&ctx).get_argument(0);
    let op_bad_pointee = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![i32_ptr],
        vec![],
        0,
    );
    let bad_pointee = MirSetDiscriminantOp::new(op_bad_pointee);
    bad_pointee.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(
        bad_pointee.verify(&ctx).is_err(),
        "Non-enum pointee rejected"
    );

    // Invalid: target attribute is required.
    let op_missing_target = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr],
        vec![],
        0,
    );
    assert!(
        MirSetDiscriminantOp::new(op_missing_target)
            .verify(&ctx)
            .is_err(),
        "Missing target rejected"
    );

    // Invalid: target index is out of bounds.
    let op_oob = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr],
        vec![],
        0,
    );
    let oob = MirSetDiscriminantOp::new(op_oob);
    oob.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(2));
    assert!(oob.verify(&ctx).is_err(), "Out-of-bounds target rejected");
}

#[test]
fn test_mir_enum_ops_malformed_arity_is_diagnostic() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let unit_enum = MirEnumType::get(
        &mut ctx,
        "Unit".into(),
        i8_ty.into(),
        vec![0],
        vec![EnumVariant::unit("Only".into())],
    );
    let payload_enum = MirEnumType::get(
        &mut ctx,
        "Payload".into(),
        i8_ty.into(),
        vec![0],
        vec![EnumVariant::new("Only".into(), vec![i32_ty.into()])],
    );
    let block = BasicBlock::new(&mut ctx, None, vec![unit_enum.into(), payload_enum.into()]);
    let unit_value = block.deref(&ctx).get_argument(0);
    let payload_value = block.deref(&ctx).get_argument(1);

    let construct_no_result = Operation::new(
        &mut ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let construct_no_result = MirConstructEnumOp::new(construct_no_result);
    construct_no_result.set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
    assert!(construct_no_result.verify(&ctx).is_err());

    let construct_extra_result = Operation::new(
        &mut ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![unit_enum.into(), unit_enum.into()],
        vec![],
        vec![],
        0,
    );
    let construct_extra_result = MirConstructEnumOp::new(construct_extra_result);
    construct_extra_result.set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
    assert!(construct_extra_result.verify(&ctx).is_err());

    let get_empty = Operation::new(
        &mut ctx,
        MirGetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(MirGetDiscriminantOp::new(get_empty).verify(&ctx).is_err());
    let get_extra = Operation::new(
        &mut ctx,
        MirGetDiscriminantOp::get_concrete_op_info(),
        vec![i8_ty.into(), i8_ty.into()],
        vec![unit_value, unit_value],
        vec![],
        0,
    );
    assert!(MirGetDiscriminantOp::new(get_extra).verify(&ctx).is_err());

    let payload_empty = Operation::new(
        &mut ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let payload_empty = MirEnumPayloadOp::new(payload_empty);
    payload_empty.set_attr_payload_variant_index(&ctx, VariantIndexAttr(0));
    payload_empty.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    assert!(payload_empty.verify(&ctx).is_err());

    let payload_extra = Operation::new(
        &mut ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![i32_ty.into(), i32_ty.into()],
        vec![payload_value, payload_value],
        vec![],
        0,
    );
    let payload_extra = MirEnumPayloadOp::new(payload_extra);
    payload_extra.set_attr_payload_variant_index(&ctx, VariantIndexAttr(0));
    payload_extra.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    assert!(payload_extra.verify(&ctx).is_err());
}

#[test]
fn test_mir_enum_payload_rejects_uninhabited_variant() {
    use dialect_mir::types::{EnumCarrierKind, EnumEncoding, EnumLayoutKind};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let enum_ty = MirEnumType::get_with_encoding(
        &mut ctx,
        "HasImpossibleField".into(),
        i8_ty.into(),
        vec![0, 1],
        vec![
            EnumVariant::unit("Live".into()),
            // An uninhabited variant's unused physical offsets need not fit
            // the object. The verifier must stop them reaching lowering.
            EnumVariant::new_with_layout(
                "Impossible".into(),
                vec![i32_ty.into()],
                vec![64],
                vec![4],
            ),
        ],
        EnumEncoding {
            tag_offset: 0,
            total_size: 1,
            abi_align: 1,
            layout_kind: EnumLayoutKind::Direct,
            carrier_kind: EnumCarrierKind::Integer,
            carrier_width: 8,
            variant_inhabited: vec![1, 0],
            ..EnumEncoding::default()
        },
    );
    assert!(enum_ty.verify(&ctx).is_ok());

    let block = BasicBlock::new(&mut ctx, None, vec![enum_ty.into()]);
    let value = block.deref(&ctx).get_argument(0);
    let op = Operation::new(
        &mut ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![value],
        vec![],
        0,
    );
    let payload = MirEnumPayloadOp::new(op);
    payload.set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
    payload.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    assert!(payload.verify(&ctx).is_err());
}

/// Reachable-but-dead MIR (e.g. the residual arms of `array::try_from_fn`)
/// can name uninhabited variants. The importer must lower such reads and
/// constructions to typed undefs; if it ever emits the real ops again, these
/// verifiers are the loud stop.
#[test]
fn test_uninhabited_enum_construct_and_discriminant_fail_verification() {
    use dialect_mir::types::{EnumCarrierKind, EnumEncoding, EnumLayoutKind};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    // Constructing an uninhabited variant must fail verification.
    let partial = MirEnumType::get_with_encoding(
        &mut ctx,
        "HasImpossibleVariant".into(),
        i8_ty.into(),
        vec![0, 1],
        vec![
            EnumVariant::unit("Live".into()),
            EnumVariant::new_with_layout(
                "Impossible".into(),
                vec![i32_ty.into()],
                vec![0],
                vec![4],
            ),
        ],
        EnumEncoding {
            tag_offset: 0,
            total_size: 8,
            abi_align: 4,
            layout_kind: EnumLayoutKind::Direct,
            carrier_kind: EnumCarrierKind::Integer,
            carrier_width: 8,
            variant_inhabited: vec![1, 0],
            ..EnumEncoding::default()
        },
    );
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let field = block.deref(&ctx).get_argument(0);
    let construct = Operation::new(
        &mut ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![partial.into()],
        vec![field],
        vec![],
        0,
    );
    let construct = MirConstructEnumOp::new(construct);
    construct.set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
    assert!(construct.verify(&ctx).is_err());

    // Reading the discriminant of a fully uninhabited enum must fail
    // verification.
    let never = MirEnumType::get_with_encoding(
        &mut ctx,
        "Never".into(),
        i8_ty.into(),
        vec![],
        vec![],
        EnumEncoding {
            tag_offset: 0,
            total_size: 0,
            abi_align: 1,
            layout_kind: EnumLayoutKind::Empty,
            carrier_kind: EnumCarrierKind::None,
            carrier_width: 0,
            variant_inhabited: vec![],
            ..EnumEncoding::default()
        },
    );
    assert!(never.verify(&ctx).is_ok());
    let never_block = BasicBlock::new(&mut ctx, None, vec![never.into()]);
    let never_value = never_block.deref(&ctx).get_argument(0);
    let get_discriminant = Operation::new(
        &mut ctx,
        MirGetDiscriminantOp::get_concrete_op_info(),
        vec![i8_ty.into()],
        vec![never_value],
        vec![],
        0,
    );
    let get_discriminant = MirGetDiscriminantOp::new(get_discriminant);
    assert!(get_discriminant.verify(&ctx).is_err());
}
