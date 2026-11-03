/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Byte-preserving storage for pointer-free unions and aggregate reinterpretation.
//!
//! Independent byte lanes preserve partially initialized union contents. Typed
//! scratch stores and loads reinterpret those bytes without numeric conversion.
//! Pointer provenance and bool's canonical storage require separate adapters and
//! remain unsupported here.

use crate::{
    MlirOperation, MlirType, OperationInput, TranslationError, TranslationRegistry,
    TranslationSession, TypeTranslation,
};
use dialect_mir::{
    ops::{MirCastOp, MirExtractFieldOp, MirInsertFieldOp},
    types::{
        MirArrayType, MirDisjointSliceType, MirEnumType, MirFP16Type, MirPtrType, MirSliceType,
        MirStructType, MirTupleType, MirUnionType,
    },
};
use pliron::{
    builtin::types::{FP16Type, FP32Type, FP64Type, IntegerType, UnitType},
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

use super::{
    checked::{load, op, slot, store},
    memory::{source_abi_align, source_stored_size},
};

/// Register union types without changing the scalar or pointer representation.
pub(super) fn register(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_type::<MirUnionType>(UnionTypeTranslation)
}

/// Reject values whose LLVM storage needs a canonicalization or provenance adapter.
fn check_byte_storage(ctx: &Context, source: TypeHandle) -> Result<(), String> {
    let ty = source.deref(ctx);
    if let Some(integer) = ty.downcast_ref::<IntegerType>() {
        return if matches!(integer.width(), 8 | 16 | 32 | 64 | 128) {
            Ok(())
        } else {
            Err("byte storage requires whole-byte scalar integers; bool needs an adapter".into())
        };
    }
    if ty.is::<MirPtrType>() || ty.is::<MirSliceType>() || ty.is::<MirDisjointSliceType>() {
        return Err("pointer-bearing byte storage requires a provenance-preserving adapter".into());
    }
    if ty.is::<UnitType>()
        || ty.is::<FP16Type>()
        || ty.is::<MirFP16Type>()
        || ty.is::<FP32Type>()
        || ty.is::<FP64Type>()
    {
        return Ok(());
    }
    if let Some(union) = ty.downcast_ref::<MirUnionType>() {
        union_storage_type(ctx, union)?;
        return Ok(());
    }
    if let Some(array) = ty.downcast_ref::<MirArrayType>() {
        return check_byte_storage(ctx, array.element_type());
    }
    let fields = if let Some(structure) = ty.downcast_ref::<MirStructType>() {
        structure.field_types()
    } else if let Some(tuple) = ty.downcast_ref::<MirTupleType>() {
        tuple.get_types()
    } else if let Some(enumeration) = ty.downcast_ref::<MirEnumType>() {
        super::checked::enum_storage_type(enumeration)?;
        &enumeration.all_field_types
    } else {
        return Err("unreviewed type in byte storage".into());
    };
    for &field in fields {
        check_byte_storage(ctx, field)?;
    }
    Ok(())
}

/// Validate every union view before choosing independent bytes plus an alignment anchor.
pub(super) fn union_storage_type(ctx: &Context, union: &MirUnionType) -> Result<MlirType, String> {
    union.verify(ctx).map_err(|error| error.to_string())?;
    let size = union.total_size();
    let align = union.abi_align();
    if !matches!(align, 1 | 2 | 4 | 8 | 16) || !size.is_multiple_of(align) {
        return Err("union storage needs a size divisible by alignment 1, 2, 4, 8 or 16".into());
    }
    for &field in union.field_types() {
        check_byte_storage(ctx, field)?;
        let field_size = source_stored_size(ctx, field).ok_or("unknown union field size")?;
        let field_align = source_abi_align(ctx, field).ok_or("unknown union field alignment")?;
        if field_size > size || field_align > align {
            return Err("union field exceeds its recorded storage size or alignment".into());
        }
    }
    MlirType::dialect(format!(
        "!llvm.struct<(array<{size} x i8>, array<0 x i{}>)>",
        align * 8
    ))
}

/// Translate only union layouts whose bytes can be moved without adapters.
struct UnionTypeTranslation;

impl TypeTranslation for UnionTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        union_storage_type(
            ctx,
            source
                .deref(ctx)
                .downcast_ref::<MirUnionType>()
                .ok_or("expected a MIR union")?,
        )
    }
}

/// Read one declared field from byte zero while retaining its typed representation.
pub(crate) fn extract_union_field(
    ctx: &Context,
    source: Ptr<Operation>,
    input: OperationInput,
    session: &mut TranslationSession<'_>,
    index: usize,
) -> Result<Vec<MlirOperation>, String> {
    MirExtractFieldOp::new(source)
        .verify(ctx)
        .map_err(|error| error.to_string())?;
    check_shape(&input, 1)?;
    let source_type = source.deref(ctx).get_operand(0).get_type(ctx);
    let union_ref = source_type.deref(ctx);
    let union = union_ref
        .downcast_ref::<MirUnionType>()
        .ok_or("expected a union")?;
    let storage = union_storage_type(ctx, union)?;
    let field = union
        .get_field_type(index)
        .ok_or("union field index is out of bounds")?;
    let result = input.results[0].clone();
    if source_stored_size(ctx, field) == Some(0) {
        return Ok(vec![op(
            "ub.poison",
            vec![],
            vec![result],
            &input.location,
        )?]);
    }
    let mut operations = vec![];
    let pointer = slot(
        storage,
        union.abi_align(),
        session,
        &input.location,
        &mut operations,
    )?;
    store(
        input.operands[0].clone(),
        pointer.clone(),
        &input.location,
        &mut operations,
    )?;
    load(pointer, result, &input.location, &mut operations)?;
    Ok(operations)
}

/// Overwrite one union view at byte zero, preserving all other bytes.
pub(crate) fn insert_union_field(
    ctx: &Context,
    source: Ptr<Operation>,
    input: OperationInput,
    session: &mut TranslationSession<'_>,
    index: usize,
) -> Result<Vec<MlirOperation>, String> {
    MirInsertFieldOp::new(source)
        .verify(ctx)
        .map_err(|error| error.to_string())?;
    check_shape(&input, 2)?;
    let source_type = source.deref(ctx).get_operand(0).get_type(ctx);
    let union_ref = source_type.deref(ctx);
    let union = union_ref
        .downcast_ref::<MirUnionType>()
        .ok_or("expected a union")?;
    let storage = union_storage_type(ctx, union)?;
    let field = union
        .get_field_type(index)
        .ok_or("union field index is out of bounds")?;
    let result = input.results[0].clone();
    if source_stored_size(ctx, field) == Some(0) {
        return Ok(vec![op(
            "builtin.unrealized_conversion_cast",
            vec![input.operands[0].clone()],
            vec![result],
            &input.location,
        )?]);
    }
    let mut operations = vec![];
    let pointer = slot(
        storage,
        union.abi_align(),
        session,
        &input.location,
        &mut operations,
    )?;
    store(
        input.operands[0].clone(),
        pointer.clone(),
        &input.location,
        &mut operations,
    )?;
    store(
        input.operands[1].clone(),
        pointer.clone(),
        &input.location,
        &mut operations,
    )?;
    load(pointer, result, &input.location, &mut operations)?;
    Ok(operations)
}

/// Identify aggregate casts while leaving scalar and pointer rules with their existing mapping.
pub(super) fn is_aggregate(ctx: &Context, source: TypeHandle) -> bool {
    let ty = source.deref(ctx);
    ty.is::<MirArrayType>()
        || ty.is::<MirStructType>()
        || ty.is::<MirTupleType>()
        || ty.is::<MirUnionType>()
        || ty.is::<MirEnumType>()
}

/// Reinterpret equal-sized, byte-faithful aggregates using one private aligned scratch slot.
pub(super) fn transmute(
    ctx: &Context,
    source: Ptr<Operation>,
    input: OperationInput,
    session: &mut TranslationSession<'_>,
) -> Result<Vec<MlirOperation>, String> {
    MirCastOp::new(source)
        .verify(ctx)
        .map_err(|error| error.to_string())?;
    check_shape(&input, 1)?;
    let from = source.deref(ctx).get_operand(0).get_type(ctx);
    let to = source.deref(ctx).get_result(0).get_type(ctx);
    check_byte_storage(ctx, from)?;
    check_byte_storage(ctx, to)?;
    let size = source_stored_size(ctx, from).ok_or("unknown transmute source size")?;
    if source_stored_size(ctx, to) != Some(size) {
        return Err("aggregate transmute requires equal stored sizes".into());
    }
    let result = input.results[0].clone();
    if size == 0 {
        return Ok(vec![op(
            "ub.poison",
            vec![],
            vec![result],
            &input.location,
        )?]);
    }
    let alignment = source_abi_align(ctx, from)
        .ok_or("unknown transmute source alignment")?
        .max(source_abi_align(ctx, to).ok_or("unknown transmute target alignment")?);
    let mut operations = vec![];
    let pointer = slot(
        input.operands[0].ty.clone(),
        alignment,
        session,
        &input.location,
        &mut operations,
    )?;
    store(
        input.operands[0].clone(),
        pointer.clone(),
        &input.location,
        &mut operations,
    )?;
    load(pointer, result, &input.location, &mut operations)?;
    Ok(operations)
}

/// Keep scratch-based translations limited to one result and ordinary SSA operands.
fn check_shape(input: &OperationInput, operands: usize) -> Result<(), String> {
    if input.results.len() != 1
        || input.operands.len() != operands
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err("invalid byte-storage operation shape".into());
    }
    Ok(())
}
