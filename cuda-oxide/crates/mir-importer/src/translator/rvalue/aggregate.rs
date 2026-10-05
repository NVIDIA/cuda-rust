/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Aggregate construction and field-value helpers.

use super::coerce::cast_to_expected_pointer_type_if_needed;
use super::coerce::{
    cast_enum_fields_to_expected_types, cast_struct_fields_to_expected_types,
    coerce_slice_data_pointee,
};
use super::const_bytes::translate_constant_value_from_bytes;
use super::operand::translate_operand;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{MirCastOp, MirConstructArrayOp, MirConstructEnumOp, MirConstructStructOp};
use dialect_mir::ops::{MirInsertFieldOp, MirUndefOp};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_error;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::{input_err, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::CrateDefType;
use rustc_public::mir;
use rustc_public::ty::AdtKind;
use rustc_public_bridge::IndexedVal;

/// Build a `DisjointSlice` value from the fields of its MIR aggregate.
///
/// The literal lists `ptr`, `len`, the index space's runtime layout, and the
/// marker fields; the markers are zero-sized and carry nothing, so dropping
/// them leaves the operands `mir.construct_disjoint_slice` takes, in the same
/// order. An index space with no runtime layout (`Index1D`, `Index2D<S>`)
/// stores `()` there, which drops with the markers and leaves the two-word
/// slice.
///
/// Field selection is positional, which the op's verifier then checks against
/// the result type: the data pointer must point to the element type, the
/// length must be an integer, and each remaining operand must match the index
/// space's layout types in order. A reordered or retyped field therefore
/// fails at verification rather than silently writing a row width into the
/// length slot.
pub(super) fn construct_disjoint_slice_aggregate(
    ctx: &mut Context,
    adt_ty: TypeHandle,
    field_values: &[Value],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    let (element_type, space_tys) = {
        let ty_obj = adt_ty.deref(ctx);
        let slice_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
            .expect("caller checked the disjoint slice type");
        (slice_ty.element_type(), slice_ty.space_types().to_vec())
    };

    let runtime_fields: Vec<Value> = field_values
        .iter()
        .copied()
        .filter(|value| !types::is_zst_type(ctx, value.get_type(ctx)))
        .collect();

    let expected = 2 + space_tys.len();
    if runtime_fields.len() != expected {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "DisjointSlice aggregate expected {} runtime fields for {}, found {}",
                expected,
                adt_ty.disp(ctx),
                runtime_fields.len()
            ))
        );
    }

    // The data pointer reaches the slice through the generic address space,
    // as the fat-pointer arm does for `*mut [T]`: a value coming from shared
    // memory carries addrspace(3) and would not match the element pointer the
    // verifier expects.
    let expected_ptr_ty: TypeHandle =
        facts::mint_generic_ptr_type(ctx, element_type, facts::abi_disjoint_slice_data_ptr())
            .into();
    let (data_val, current_prev_op) = cast_to_expected_pointer_type_if_needed(
        ctx,
        runtime_fields[0],
        expected_ptr_ty,
        block_ptr,
        prev_op,
        loc.clone(),
    );

    let mut operands = vec![data_val];
    operands.extend_from_slice(&runtime_fields[1..]);

    let op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![adt_ty],
        operands,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    let result = op.deref(ctx).get_result(0);
    Ok((Some(op), result, current_prev_op))
}

/// Translate ADT aggregate operands, synthesizing omitted runtime-ZST fields when
/// MIR carries only the non-ZST runtime operands.
pub(super) fn translate_adt_aggregate_field_values(
    ctx: &mut Context,
    body: &mir::Body,
    adt_def: rustc_public::ty::AdtDef,
    variant_idx: rustc_public::ty::VariantIdx,
    substs: &rustc_public::ty::GenericArgs,
    operands: &[mir::Operand],
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Vec<Value>, Option<Ptr<Operation>>)> {
    let variant_index = variant_idx.to_index();
    let variant = &adt_def.variants()[variant_index];

    let mut field_infos = Vec::with_capacity(variant.fields().len());
    for field in variant.fields() {
        let field_rust_ty = field.ty_with_args(substs);
        let translated_ty = types::translate_type(ctx, &field_rust_ty)?;
        let is_runtime_zst = field_rust_ty
            .layout()
            .map(|layout| layout.shape().is_1zst())
            .unwrap_or(false);
        field_infos.push((field_rust_ty, translated_ty, is_runtime_zst));
    }

    let total_field_count = field_infos.len();
    let non_zst_count = field_infos
        .iter()
        .filter(|(_, _, is_runtime_zst)| !*is_runtime_zst)
        .count();

    let synthesize_runtime_zsts = if operands.len() == total_field_count {
        false
    } else if operands.len() == non_zst_count {
        true
    } else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ADT aggregate '{}' variant '{}' has {} translated fields, {} non-ZST runtime fields, but MIR provided {} operands",
                adt_def.trimmed_name(),
                variant.name(),
                total_field_count,
                non_zst_count,
                operands.len()
            ))
        );
    };

    let mut field_values = Vec::with_capacity(total_field_count);
    let mut current_prev_op = prev_op;
    let mut operand_iter = operands.iter();

    for (field_rust_ty, translated_ty, is_runtime_zst) in field_infos {
        if synthesize_runtime_zsts && is_runtime_zst {
            let (value, new_prev_op) = translate_constant_value_from_bytes(
                ctx,
                &field_rust_ty,
                translated_ty,
                &[],
                block_ptr,
                current_prev_op,
                loc.clone(),
            )?;
            field_values.push(value);
            current_prev_op = new_prev_op;
            continue;
        }

        let operand = operand_iter.next().ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "ADT aggregate '{}' variant '{}' ran out of MIR operands while translating fields",
                adt_def.trimmed_name(),
                variant.name()
            )))
        })?;
        let (value, new_prev_op) = translate_operand(
            ctx,
            body,
            operand,
            value_map,
            block_ptr,
            current_prev_op,
            loc.clone(),
        )?;
        field_values.push(value);
        current_prev_op = new_prev_op;
    }

    if operand_iter.next().is_some() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ADT aggregate '{}' variant '{}' left unused MIR operands after field translation",
                adt_def.trimmed_name(),
                variant.name()
            ))
        );
    }

    Ok((field_values, current_prev_op))
}

/// Construct a union by writing the one active field into shared storage.
///
/// MIR supplies exactly one operand plus the declaration index of its active
/// field. Start with undefined union storage and use `mir.insert_field` to
/// write that typed view at byte zero. The union-specific lowering preserves
/// every other byte as undefined; it never invents one independent slot per
/// field.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_union_aggregate(
    ctx: &mut Context,
    body: &mir::Body,
    adt_def: rustc_public::ty::AdtDef,
    union_ty: TypeHandle,
    active_field_idx: Option<usize>,
    operands: &[mir::Operand],
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    let active_field_idx = active_field_idx.ok_or_else(|| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Union aggregate '{}' did not identify an active field",
            adt_def.trimmed_name()
        )))
    })?;

    if operands.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Union aggregate '{}' expected exactly one operand for active field {}, found {}",
                adt_def.trimmed_name(),
                active_field_idx,
                operands.len()
            ))
        );
    }

    let (field_count, expected_field_ty) = {
        let ty_ref = union_ty.deref(ctx);
        let union = ty_ref
            .downcast_ref::<dialect_mir::types::MirUnionType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Union aggregate '{}' did not translate to MirUnionType",
                    adt_def.trimmed_name()
                )))
            })?;
        (union.field_count(), union.get_field_type(active_field_idx))
    };
    if active_field_idx >= field_count {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Union aggregate '{}' active field {} is out of bounds for {} fields",
                adt_def.trimmed_name(),
                active_field_idx,
                field_count
            ))
        );
    }
    let expected_field_ty = expected_field_ty.expect("active union field was bounds-checked");

    let (active_value, current_prev_op) = translate_operand(
        ctx,
        body,
        &operands[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (active_value, current_prev_op) = cast_to_expected_pointer_type_if_needed(
        ctx,
        active_value,
        expected_field_ty,
        block_ptr,
        current_prev_op,
        loc.clone(),
    );

    let undef_op = MirUndefOp::new(ctx, union_ty).get_operation();
    undef_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = current_prev_op {
        undef_op.insert_after(ctx, prev);
    } else {
        undef_op.insert_at_front(block_ptr, ctx);
    }
    let undef_value = undef_op.deref(ctx).get_result(0);

    let insert_op = Operation::new(
        ctx,
        MirInsertFieldOp::get_concrete_op_info(),
        vec![union_ty],
        vec![undef_value, active_value],
        vec![],
        0,
    );
    insert_op.deref_mut(ctx).set_loc(loc);
    MirInsertFieldOp::new(insert_op).set_attr_insert_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(active_field_idx as u32),
    );
    let result = insert_op.deref(ctx).get_result(0);

    Ok((Some(insert_op), result, Some(undef_op)))
}

/// Translate a `Rvalue::Aggregate` into `dialect-mir` construction ops.
///
/// Aggregate constructs a compound type from individual values: tuples,
/// structs/enums/unions, arrays, closures, and raw fat/thin pointers.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_aggregate_rvalue(
    ctx: &mut Context,
    body: &mir::Body,
    rvalue: &mir::Rvalue,
    aggregate_kind: &mir::AggregateKind,
    operands: &[mir::Operand],
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    // Aggregate constructs a compound type from individual values.
    // This is used for:
    // - Tuple construction: (a, b, c)
    // - Struct construction: MyStruct { field1: a, field2: b }
    // - Array construction: [a, b, c]

    match aggregate_kind {
        mir::AggregateKind::Adt(adt_def, variant_idx, substs, _, active_field_idx) => {
            let adt_kind = adt_def.kind();

            // Get the type using adt_def.ty_with_args()
            let adt_ty_rust = adt_def.ty_with_args(substs);
            let adt_ty = types::translate_type(ctx, &adt_ty_rust)?;
            let translated_field_values = if matches!(adt_kind, AdtKind::Union) {
                None
            } else {
                Some(translate_adt_aggregate_field_values(
                    ctx,
                    body,
                    *adt_def,
                    *variant_idx,
                    substs,
                    operands,
                    value_map,
                    block_ptr,
                    prev_op,
                    loc.clone(),
                )?)
            };

            match adt_kind {
                AdtKind::Struct => {
                    let (field_values, current_prev_op) = translated_field_values
                        .expect("non-union ADT fields should have been translated");
                    // Check if the translated type is a struct type.
                    // Scalar-lowered newtypes like ThreadIndex are translated to
                    // their single runtime field type. They may still have ZST
                    // marker fields in MIR, so select the one field whose
                    // translated value matches the scalar result type.
                    let is_struct_type = {
                        let ty_obj = adt_ty.deref(ctx);
                        ty_obj.is::<dialect_mir::types::MirStructType>()
                            || ty_obj.is::<dialect_mir::types::MirTupleType>()
                    };

                    // A `DisjointSlice` literal, which
                    // `DisjointSlice::from_raw_parts` builds. The type
                    // translator gives the ADT its own slice type rather
                    // than a struct, so without this arm the shape falls
                    // into the scalar-lowered path below, where no field
                    // carries the slice type and the search reports zero
                    // runtime fields (issue #667).
                    if adt_ty
                        .deref(ctx)
                        .is::<dialect_mir::types::MirDisjointSliceType>()
                    {
                        return construct_disjoint_slice_aggregate(
                            ctx,
                            adt_ty,
                            &field_values,
                            block_ptr,
                            current_prev_op,
                            loc,
                        );
                    }

                    if !is_struct_type {
                        // Scalar-lowered ADT: layout collapsed to a single runtime
                        // value. The MIR Aggregate may still list ZST fields
                        // (PhantomData, etc.) -- those translate to types other
                        // than `adt_ty`, so filtering by "type matches the
                        // collapsed scalar" reliably picks the one runtime field.
                        //
                        // This works for shapes like
                        //     ThreadIndex { raw: usize, _kernel: PhantomData<...>, ... }
                        // where exactly one field shares the scalar type. If a
                        // future scalar-lowered ADT has two runtime fields with
                        // the same type, the filter returns >1 match and we bail
                        // -- the assumption is wrong and the translator needs an
                        // explicit story for that shape.
                        let runtime_fields: Vec<Value> = field_values
                            .iter()
                            .copied()
                            .filter(|value| value.get_type(ctx) == adt_ty)
                            .collect();

                        if runtime_fields.len() == 1 {
                            Ok((None, runtime_fields[0], current_prev_op))
                        } else {
                            input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Scalar-lowered ADT expected exactly one runtime field, found {}",
                                    runtime_fields.len()
                                ))
                            )
                        }
                    } else {
                        // Cast field values to expected types (address space normalization)
                        // This handles cases where field values have specific address spaces
                        // (e.g., addrspace:3 for shared memory) but the struct type expects
                        // generic address space (addrspace:0)
                        let (casted_field_values, prev_after_casts) =
                            cast_struct_fields_to_expected_types(
                                ctx,
                                field_values,
                                adt_ty,
                                block_ptr,
                                current_prev_op,
                                loc.clone(),
                            );

                        // Create the construct_struct operation
                        let op = Operation::new(
                            ctx,
                            MirConstructStructOp::get_concrete_op_info(),
                            vec![adt_ty],
                            casted_field_values,
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc);

                        let result = op.deref(ctx).get_result(0);

                        Ok((Some(op), result, prev_after_casts))
                    }
                }
                AdtKind::Enum => {
                    let (field_values, current_prev_op) = translated_field_values
                        .expect("non-union ADT fields should have been translated");
                    // Get the variant index for the enum
                    // NOTE: variant_idx IS the index (0, 1, 2, ...), NOT the discriminant!
                    // discriminant_for_variant returns the discriminant VALUE which may differ
                    // (e.g., enum Foo { A = 0, B = 2, C = 6 } has indices 0,1,2 but discriminants 0,2,6)
                    let variant_index_val: usize = variant_idx.to_index();

                    // A value inhabiting this variant cannot exist,
                    // so this construction sits on a dynamically dead
                    // path rustc keeps in MIR (e.g. building
                    // `ControlFlow::Break(NeverShortCircuitResidual)`
                    // inside `array::try_from_fn`).
                    // `mir.construct_enum` refuses uninhabited
                    // variants by verification, so keep the dead path
                    // representable with a typed undef instead.
                    let variant_is_uninhabited = adt_ty
                        .deref(ctx)
                        .downcast_ref::<dialect_mir::types::MirEnumType>()
                        .and_then(|enum_ty| enum_ty.variant_is_inhabited(variant_index_val))
                        .is_some_and(|inhabited| !inhabited);
                    if variant_is_uninhabited {
                        let undef = MirUndefOp::new(ctx, adt_ty).get_operation();
                        undef.deref_mut(ctx).set_loc(loc);
                        let result = undef.deref(ctx).get_result(0);
                        return Ok((Some(undef), result, current_prev_op));
                    }

                    // Cast field values to expected types (address space normalization)
                    // This handles cases where field values have specific address spaces
                    // (e.g., addrspace:3 for shared memory) but the enum type expects
                    // generic address space (addrspace:0)
                    let (casted_field_values, prev_after_casts) =
                        cast_enum_fields_to_expected_types(
                            ctx,
                            field_values,
                            adt_ty,
                            variant_index_val,
                            block_ptr,
                            current_prev_op,
                            loc.clone(),
                        );

                    // Create the construct_enum operation with variant_index attribute
                    let op = Operation::new(
                        ctx,
                        MirConstructEnumOp::get_concrete_op_info(),
                        vec![adt_ty],
                        casted_field_values,
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc.clone());

                    let enum_op = MirConstructEnumOp::new(op);
                    enum_op.set_attr_construct_enum_variant_index(
                        ctx,
                        dialect_mir::attributes::VariantIndexAttr(variant_index_val as u32),
                    );

                    let result = op.deref(ctx).get_result(0);

                    Ok((Some(op), result, prev_after_casts))
                }
                AdtKind::Union => translate_union_aggregate(
                    ctx,
                    body,
                    *adt_def,
                    adt_ty,
                    *active_field_idx,
                    operands,
                    value_map,
                    block_ptr,
                    prev_op,
                    loc,
                ),
            }
        }
        mir::AggregateKind::Tuple => {
            // Tuple construction: (a, b, c)
            // Similar to struct construction but with positional fields

            // Translate all element operands
            let mut element_values = Vec::with_capacity(operands.len());
            let mut current_prev_op = prev_op;

            for operand in operands {
                let (val, new_prev_op) = translate_operand(
                    ctx,
                    body,
                    operand,
                    value_map,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                element_values.push(val);
                current_prev_op = new_prev_op;
            }

            // Translate the tuple type from the rvalue's rustc type
            // so it carries rustc's layout and uniques with the
            // tuple type of the destination place.
            let rust_tuple_ty = rvalue.ty(body.locals()).map_err(|e| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Failed to query tuple aggregate type: {:?}",
                    e
                )))
            })?;
            let tuple_ty = types::translate_type(ctx, &rust_tuple_ty)?;

            // Create mir.construct_tuple operation
            use dialect_mir::ops::MirConstructTupleOp;

            let op = Operation::new(
                ctx,
                MirConstructTupleOp::get_concrete_op_info(),
                vec![tuple_ty],
                element_values,
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, current_prev_op))
        }
        mir::AggregateKind::Array(elem_ty) => {
            // Array construction: [e0, e1, e2, ...] -> mir.construct_array
            // Translate the element type
            let element_type = types::translate_type(ctx, elem_ty)?;
            let array_size = operands.len() as u64;

            // Translate all element operands
            let mut element_values = Vec::with_capacity(operands.len());
            let mut current_prev_op = prev_op;

            for operand in operands {
                let (val, new_prev_op) = translate_operand(
                    ctx,
                    body,
                    operand,
                    value_map,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                let (val, new_prev_op) = cast_to_expected_pointer_type_if_needed(
                    ctx,
                    val,
                    element_type,
                    block_ptr,
                    new_prev_op,
                    loc.clone(),
                );
                element_values.push(val);
                current_prev_op = new_prev_op;
            }

            // Create the array type
            let array_ty = dialect_mir::types::MirArrayType::get(ctx, element_type, array_size);

            // Create mir.construct_array operation
            let op = Operation::new(
                ctx,
                MirConstructArrayOp::get_concrete_op_info(),
                vec![array_ty.into()],
                element_values,
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, current_prev_op))
        }
        mir::AggregateKind::Closure(closure_def, substs) => {
            // Closure construction with captures
            // The operands are the captured values that form the closure environment
            //
            // MIR: _N = Aggregate(Closure(...), [captured_val1, captured_val2, ...])
            // We construct a struct with the captured values as fields

            // Translate all captured operands
            let mut capture_values = Vec::with_capacity(operands.len());
            let mut current_prev_op = prev_op;

            for operand in operands {
                let (val, new_prev_op) = translate_operand(
                    ctx,
                    body,
                    operand,
                    value_map,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                capture_values.push(val);
                current_prev_op = new_prev_op;
            }

            // Get the closure type
            let closure_ty_rust = rustc_public::ty::Ty::new_closure(*closure_def, substs.clone());
            let closure_ty = types::translate_type(ctx, &closure_ty_rust)?;

            if capture_values.is_empty() {
                // ZST closure (no captures) - create empty struct
                let op = Operation::new(
                    ctx,
                    MirConstructStructOp::get_concrete_op_info(),
                    vec![closure_ty],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc);
                let result = op.deref(ctx).get_result(0);
                Ok((Some(op), result, current_prev_op))
            } else {
                // Closure with captures - create struct with captured values
                // Cast captured values to expected types (address space normalization)
                let (casted_capture_values, prev_after_casts) =
                    cast_struct_fields_to_expected_types(
                        ctx,
                        capture_values,
                        closure_ty,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    );

                let op = Operation::new(
                    ctx,
                    MirConstructStructOp::get_concrete_op_info(),
                    vec![closure_ty],
                    casted_capture_values,
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc);
                let result = op.deref(ctx).get_result(0);
                Ok((Some(op), result, prev_after_casts))
            }
        }
        mir::AggregateKind::RawPtr(pointee_ty, mutability) => {
            // Raw pointer construction from parts: rustc lowers the
            // `aggregate_raw_ptr` intrinsic to this aggregate kind.
            // It is reached by re-slicing (`&bytes[2..]` goes through
            // `slice::index::get_offset_len_noubcheck`) and by
            // `ptr::slice_from_raw_parts` / `ptr::from_raw_parts`.
            // The two operands are (data_pointer, metadata).
            use rustc_public::ty::{RigidTy, TyKind};

            if operands.len() != 2 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "RawPtr aggregate expected 2 operands (data, metadata), found {}",
                        operands.len()
                    ))
                );
            }

            let origin = facts::pointer_origin_of_raw_mutability(*mutability);
            let is_mutable = origin.is_mutable();

            match pointee_ty.kind() {
                TyKind::RigidTy(RigidTy::Slice(elem_ty)) => {
                    // `*const [T]` / `*mut [T]`: the metadata operand is
                    // the element count. `*const [T]` translates to
                    // `MirSliceType` (same runtime layout as `&[T]`), so
                    // build the fat pointer with `mir.construct_slice`.
                    let element_type = types::translate_type(ctx, &elem_ty)?;

                    let (data_val, prev_after_data) = translate_operand(
                        ctx,
                        body,
                        &operands[0],
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;
                    let (len_val, prev_after_len) = translate_operand(
                        ctx,
                        body,
                        &operands[1],
                        value_map,
                        block_ptr,
                        prev_after_data,
                        loc.clone(),
                    )?;

                    // The fat pointer's data slot is a generic-addrspace
                    // pointer. Values coming from shared memory carry
                    // addrspace(3); normalize them like the struct/array
                    // arms do.
                    let expected_ptr_ty: TypeHandle =
                        facts::mint_generic_ptr_type(ctx, element_type, origin).into();
                    let (data_val, current_prev_op) = cast_to_expected_pointer_type_if_needed(
                        ctx,
                        data_val,
                        expected_ptr_ty,
                        block_ptr,
                        prev_after_len,
                        loc.clone(),
                    );

                    // Coerce the data pointer to the slice element type: a
                    // reinterpret cast feeding `from_raw_parts` can leave it
                    // typed to the pre-cast pointee (e.g. `*mut u64` for a
                    // `[(u64, u64)]` slice), which the fat pointer rejects.
                    let (data_val, current_prev_op) = coerce_slice_data_pointee(
                        ctx,
                        data_val,
                        element_type,
                        is_mutable,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    );

                    let slice_ty = facts::mint_slice_type(ctx, element_type, origin);

                    use dialect_mir::ops::MirConstructSliceOp;
                    let op = Operation::new(
                        ctx,
                        MirConstructSliceOp::get_concrete_op_info(),
                        vec![slice_ty.into()],
                        vec![data_val, len_val],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let result = op.deref(ctx).get_result(0);

                    Ok((Some(op), result, current_prev_op))
                }
                TyKind::RigidTy(RigidTy::Str) => {
                    // Blocked on `str` having a device-side type
                    // translation (issue #76).
                    input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "RawPtr aggregate with `str` pointee not yet supported \
                                     (no `str` type translation on device)"
                                .to_string()
                        )
                    )
                }
                TyKind::RigidTy(RigidTy::Dynamic(..)) => {
                    // Trait objects need a vtable, which has no
                    // device-side story.
                    input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "RawPtr aggregate with `dyn Trait` pointee not supported \
                                     (no vtable support on device)"
                                .to_string()
                        )
                    )
                }
                _ => {
                    // `Sized` pointee: the metadata operand is `()`, so
                    // the aggregate is just the data pointer re-typed as
                    // `*const P` / `*mut P`. Confirm the metadata really
                    // is unit before dropping it; an unsized-tail struct
                    // pointee would carry real metadata here.
                    let metadata_ty = operands[1].ty(body.locals()).map_err(|e| {
                        input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "Cannot get RawPtr aggregate metadata type: {e}"
                            ))
                        )
                    })?;
                    let metadata_is_unit = matches!(
                        metadata_ty.kind(),
                        TyKind::RigidTy(RigidTy::Tuple(tys)) if tys.is_empty()
                    );
                    if !metadata_is_unit {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "RawPtr aggregate with non-unit metadata of type {:?} \
                                         not yet supported",
                                metadata_ty
                            ))
                        );
                    }

                    // Translate the target pointer type through the same
                    // path as the destination local, so the two agree
                    // (including SharedArray/Barrier special cases).
                    let raw_ptr_ty_rust = rustc_public::ty::Ty::new_ptr(*pointee_ty, *mutability);
                    let target_ty = types::translate_type(ctx, &raw_ptr_ty_rust)?;

                    let (data_val, current_prev_op) = translate_operand(
                        ctx,
                        body,
                        &operands[0],
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    if data_val.get_type(ctx) == target_ty {
                        // Already the right pointer type: pass through.
                        Ok((None, data_val, current_prev_op))
                    } else {
                        // Pointer re-typing, e.g. `*const ()` data
                        // pointer becoming `*const P`.
                        let cast_op = Operation::new(
                            ctx,
                            MirCastOp::get_concrete_op_info(),
                            vec![target_ty],
                            vec![data_val],
                            vec![],
                            0,
                        );
                        cast_op.deref_mut(ctx).set_loc(loc);
                        MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);

                        let result = cast_op.deref(ctx).get_result(0);

                        Ok((Some(cast_op), result, current_prev_op))
                    }
                }
            }
        }
        _ => {
            input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Aggregate kind {:?} not yet supported",
                    aggregate_kind
                ))
            )
        }
    }
}
