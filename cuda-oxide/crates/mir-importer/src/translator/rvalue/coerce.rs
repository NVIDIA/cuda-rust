/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pointer and type normalization helpers shared by expr/operand/place
//! translation.

use crate::translator::facts;
use crate::translator::values::generic_pointer_kind_retype_allowed;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::MirCastOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// Normalize a pointer-like value without creating new Rust pointer semantics.
///
/// This helper is for representation adjustments such as address-space
/// normalization and aggregate/local type reconciliation. Thin pointers must
/// have the same pointee and fat slices the same element type. A concrete kind
/// may be preserved or deliberately erased, but `Erased` cannot regain a
/// concrete Rust kind and two distinct concrete kinds cannot be interconverted.
pub(super) fn cast_to_expected_pointer_type_if_needed(
    ctx: &mut Context,
    value: Value,
    expected_type: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Option<Ptr<Operation>>) {
    let value_type = value.get_type(ctx);
    if value_type == expected_type {
        return (value, prev_op);
    }

    let compatible = {
        let value_ref = value_type.deref(ctx);
        let expected_ref = expected_type.deref(ctx);

        match (
            value_ref.downcast_ref::<dialect_mir::types::MirPtrType>(),
            expected_ref.downcast_ref::<dialect_mir::types::MirPtrType>(),
        ) {
            (Some(value_ptr), Some(expected_ptr)) => {
                value_ptr.pointee == expected_ptr.pointee
                    && value_ptr.is_mutable == expected_ptr.is_mutable
                    && generic_pointer_kind_retype_allowed(value_ptr.kind, expected_ptr.kind)
            }
            _ => match (
                value_ref.downcast_ref::<dialect_mir::types::MirSliceType>(),
                expected_ref.downcast_ref::<dialect_mir::types::MirSliceType>(),
            ) {
                (Some(value_slice), Some(expected_slice)) => {
                    value_slice.element_ty == expected_slice.element_ty
                        && value_slice.is_mutable == expected_slice.is_mutable
                        && generic_pointer_kind_retype_allowed(
                            value_slice.kind,
                            expected_slice.kind,
                        )
                }
                _ => false,
            },
        }
    };

    if !compatible {
        return (value, prev_op);
    }

    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![expected_type],
        vec![value],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);

    match prev_op {
        Some(prev) => cast_op.insert_after(ctx, prev),
        None => cast_op.insert_at_front(block_ptr, ctx),
    }

    (cast_op.deref(ctx).get_result(0), Some(cast_op))
}

/// Establish the exact pointer/reference type declared by rustc at a semantic boundary.
///
/// `Rvalue::Ref` and `Rvalue::AddressOf` are not representation-only
/// normalizations: they create a new Rust pointer/reference value. The result
/// type supplied by rustc is therefore authoritative, including legitimate
/// transitions such as `RawMut -> UniqueRef` for `unsafe { &mut *raw }`. This
/// helper still requires the same thin-pointee or fat-slice element shape so it
/// cannot hide an unrelated representation mismatch.
pub(super) fn cast_to_declared_rust_pointer_type_if_needed(
    ctx: &mut Context,
    value: Value,
    expected_type: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
    authority: MirPointerKindAuthorityAttr,
) -> (Value, Option<Ptr<Operation>>) {
    let value_type = value.get_type(ctx);
    if value_type == expected_type {
        return (value, prev_op);
    }

    let compatible = {
        let value_ref = value_type.deref(ctx);
        let expected_ref = expected_type.deref(ctx);

        match (
            value_ref.downcast_ref::<dialect_mir::types::MirPtrType>(),
            expected_ref.downcast_ref::<dialect_mir::types::MirPtrType>(),
        ) {
            (Some(value_ptr), Some(expected_ptr)) => value_ptr.pointee == expected_ptr.pointee,
            _ => match (
                value_ref.downcast_ref::<dialect_mir::types::MirSliceType>(),
                expected_ref.downcast_ref::<dialect_mir::types::MirSliceType>(),
            ) {
                (Some(value_slice), Some(expected_slice)) => {
                    value_slice.element_ty == expected_slice.element_ty
                }
                _ => false,
            },
        }
    };

    if !compatible {
        return (value, prev_op);
    }

    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![expected_type],
        vec![value],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    let cast = MirCastOp::new(cast_op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    cast.set_pointer_kind_authority(ctx, authority);

    match prev_op {
        Some(prev) => cast_op.insert_after(ctx, prev),
        None => cast_op.insert_at_front(block_ptr, ctx),
    }

    (cast_op.deref(ctx).get_result(0), Some(cast_op))
}

/// Return the physical-storage form of a thin pointer type.
///
/// Allocation operations create addresses, not Rust borrows/raw-pointer
/// values, so their results stay `Erased`. The translated Rust constant type
/// is established by a separately marked boundary cast.
pub(super) fn erase_thin_pointer_kind(
    ctx: &mut Context,
    pointer_ty: TypeHandle,
) -> Option<TypeHandle> {
    let (pointee, is_mutable, address_space) = {
        let pointer_ty = pointer_ty.deref(ctx);
        let pointer_ty = pointer_ty.downcast_ref::<dialect_mir::types::MirPtrType>()?;
        (
            pointer_ty.pointee,
            pointer_ty.is_mutable,
            pointer_ty.address_space,
        )
    };
    Some(dialect_mir::types::MirPtrType::get(ctx, pointee, is_mutable, address_space).into())
}

/// Coerce a raw data pointer so it points to `element_type` in the generic
/// address space.
///
/// `from_raw_parts`/`from_raw_parts_mut` build a `[T]` slice from a thin data
/// pointer, but a reinterpret cast at the call site (e.g. `p as *mut (u64, u64)`
/// feeding a `[(u64, u64)]` slice) can leave the data operand typed to the
/// pre-cast pointee. The fat pointer's data slot must be a generic-address-space
/// pointer to the slice element type, so when the pointee differs, emit a
/// `PtrToPtr` cast to that type. The data carrier retains its existing pointer
/// kind; otherwise constructing the fat pointer could launder an Erased carrier
/// back into a concrete slice kind.
pub(super) fn coerce_slice_data_pointee(
    ctx: &mut Context,
    value: Value,
    element_type: TypeHandle,
    _is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Option<Ptr<Operation>>) {
    let (pointee_matches, origin) = {
        let value_type = value.get_type(ctx);
        let ty_ref = value_type.deref(ctx);
        match ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>() {
            Some(pt) => (
                pt.pointee == element_type,
                facts::pointer_origin_of_ptr_carrier(pt),
            ),
            None => return (value, prev_op),
        }
    };
    if pointee_matches {
        return (value, prev_op);
    }
    let target_ptr_ty: TypeHandle = facts::mint_generic_ptr_type(ctx, element_type, origin).into();
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![target_ptr_ty],
        vec![value],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    match prev_op {
        Some(p) => cast_op.insert_after(ctx, p),
        None => cast_op.insert_at_front(block_ptr, ctx),
    }
    (cast_op.deref(ctx).get_result(0), Some(cast_op))
}

/// Cast struct field values to match expected field types (address space normalization).
///
/// When constructing a struct, field values may have specific address spaces (e.g., addrspace:3)
/// but the struct type's field definitions use generic address space (addrspace:0).
/// This function casts each field value to match its expected type.
pub(super) fn cast_struct_fields_to_expected_types(
    ctx: &mut Context,
    field_values: Vec<Value>,
    struct_type: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Vec<Value>, Option<Ptr<Operation>>) {
    // Get field types from the struct type
    let field_types: Vec<TypeHandle> = {
        let ty_ref = struct_type.deref(ctx);
        if let Some(st) = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>() {
            st.field_types.clone()
        } else {
            // Not a struct type, return as-is
            return (field_values, prev_op);
        }
    };

    let mut result_values = Vec::with_capacity(field_values.len());
    let mut current_prev_op = prev_op;

    for (i, value) in field_values.into_iter().enumerate() {
        if let Some(expected_type) = field_types.get(i) {
            let (casted_value, new_prev_op) = cast_to_expected_pointer_type_if_needed(
                ctx,
                value,
                *expected_type,
                block_ptr,
                current_prev_op,
                loc.clone(),
            );
            result_values.push(casted_value);
            current_prev_op = new_prev_op;
        } else {
            result_values.push(value);
        }
    }

    (result_values, current_prev_op)
}

/// Cast enum variant field values to match expected field types (address space normalization).
///
/// Similar to cast_struct_fields_to_expected_types, but for enum variants.
/// Gets the field types for the specific variant and casts each field value.
pub(super) fn cast_enum_fields_to_expected_types(
    ctx: &mut Context,
    field_values: Vec<Value>,
    enum_type: TypeHandle,
    variant_idx: usize,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Vec<Value>, Option<Ptr<Operation>>) {
    // Get the field types for this variant from the enum type
    let variant_field_types: Vec<TypeHandle> = {
        let ty_ref = enum_type.deref(ctx);
        if let Some(et) = ty_ref.downcast_ref::<dialect_mir::types::MirEnumType>() {
            // Calculate the field offset for this variant
            let field_offset: usize = et.variant_field_counts[..variant_idx]
                .iter()
                .map(|&x| x as usize)
                .sum();
            let field_count = et.variant_field_counts[variant_idx] as usize;

            // Get the field types for this variant
            et.all_field_types[field_offset..field_offset + field_count].to_vec()
        } else {
            // Not an enum type, return as-is
            return (field_values, prev_op);
        }
    };

    let mut result_values = Vec::with_capacity(field_values.len());
    let mut current_prev_op = prev_op;

    for (i, value) in field_values.into_iter().enumerate() {
        if let Some(expected_type) = variant_field_types.get(i) {
            let (casted_value, new_prev_op) = cast_to_expected_pointer_type_if_needed(
                ctx,
                value,
                *expected_type,
                block_ptr,
                current_prev_op,
                loc.clone(),
            );
            result_values.push(casted_value);
            current_prev_op = new_prev_op;
        } else {
            result_values.push(value);
        }
    }

    (result_values, current_prev_op)
}

#[cfg(test)]
// Tests build kinded fixture types directly; production code mints via facts::PointerOrigin.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use dialect_mir::types::{MirPointerKind, MirPtrType};
    use pliron::builtin::types::{IntegerType, Signedness};

    #[test]
    fn expected_pointer_normalization_rejects_concrete_kind_change() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let pointee_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let source_ty: TypeHandle = MirPtrType::get_generic_with_kind(
            &mut ctx,
            pointee_ty,
            false,
            MirPointerKind::SharedRef,
        )
        .into();
        let target_ty: TypeHandle = MirPtrType::get_generic_with_kind(
            &mut ctx,
            pointee_ty,
            true,
            MirPointerKind::UniqueRef,
        )
        .into();
        let block = BasicBlock::new(&mut ctx, None, vec![source_ty]);
        let source = block.deref(&ctx).get_argument(0);

        let (normalized, last_op) = cast_to_expected_pointer_type_if_needed(
            &mut ctx,
            source,
            target_ty,
            block,
            None,
            Location::Unknown,
        );

        assert_eq!(normalized.get_type(&ctx), source_ty);
        assert!(
            last_op.is_none(),
            "generic normalization must not strengthen SharedRef into UniqueRef"
        );

        let erased_read: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee_ty, false).into();
        let erased_write: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee_ty, true).into();
        let erased_block = BasicBlock::new(&mut ctx, None, vec![erased_read]);
        let erased_value = erased_block.deref(&ctx).get_argument(0);
        let (normalized, last_op) = cast_to_expected_pointer_type_if_needed(
            &mut ctx,
            erased_value,
            erased_write,
            erased_block,
            None,
            Location::Unknown,
        );
        assert_eq!(normalized.get_type(&ctx), erased_read);
        assert!(
            last_op.is_none(),
            "generic normalization must not manufacture writable Erased evidence"
        );
    }

    #[test]
    fn rust_pointer_boundary_allows_raw_mut_reborrow_to_unique_ref() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let pointee_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let raw_mut_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee_ty, true, MirPointerKind::RawMut)
                .into();
        let unique_ref_ty: TypeHandle = MirPtrType::get_generic_with_kind(
            &mut ctx,
            pointee_ty,
            true,
            MirPointerKind::UniqueRef,
        )
        .into();
        let block = BasicBlock::new(&mut ctx, None, vec![raw_mut_ty]);
        let raw_mut = block.deref(&ctx).get_argument(0);

        let (reborrow, last_op) = cast_to_declared_rust_pointer_type_if_needed(
            &mut ctx,
            raw_mut,
            unique_ref_ty,
            block,
            None,
            Location::Unknown,
            MirPointerKindAuthorityAttr::Reborrow,
        );

        assert_eq!(reborrow.get_type(&ctx), unique_ref_ty);
        assert!(
            last_op.is_some(),
            "Rvalue::Ref must trust rustc's declared reborrow result kind"
        );
    }

    #[test]
    fn projected_address_normalization_matches_expected_pointer_type() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let pointee_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        let physical_ptr_ty: TypeHandle = MirPtrType::get(&mut ctx, pointee_ty, false, 1).into();

        let expected_rust_ptr_ty: TypeHandle = MirPtrType::get_generic_with_kind(
            &mut ctx,
            pointee_ty,
            false,
            MirPointerKind::SharedRef,
        )
        .into();

        let block = BasicBlock::new(&mut ctx, None, vec![physical_ptr_ty]);
        let physical_pointer = block.deref(&ctx).get_argument(0);

        // Rvalue::Ref and Rvalue::AddressOf both use this normalization after
        // computing a projected address in its physical address space.
        let (normalized_pointer, last_op) = cast_to_declared_rust_pointer_type_if_needed(
            &mut ctx,
            physical_pointer,
            expected_rust_ptr_ty,
            block,
            None,
            Location::Unknown,
            MirPointerKindAuthorityAttr::Reborrow,
        );

        assert_eq!(
            normalized_pointer.get_type(&ctx),
            expected_rust_ptr_ty,
            "projected addresses must be normalized to the exact Rust pointer type and kind"
        );

        let cast_op = last_op.expect(
            "normalizing addrspace(1) to the Rust generic address space must insert a cast",
        );

        assert!(
            Operation::get_op::<MirCastOp>(cast_op, &ctx).is_some(),
            "normalization must insert mir.cast"
        );
    }
}
