/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Emit validated native plans from original typed library operands.

use super::*;
use dialect_mir::{
    attributes::{FieldIndexAttr, MirPointerKindAuthorityAttr},
    library::{LibraryCallPlan, LibraryNativeMetadata},
    ops::{MirCallOp, MirExternSharedOp, MirExtractFieldOp},
    types::{MirPointerKind, MirPtrType, MirTupleType},
};
use pliron::{
    builtin::{type_interfaces::FunctionTypeInterface, types::FunctionType},
    r#type::Typed,
    verify_err,
};

const FIXED_SHARED_BYTES: &str = "cuda_oxide_fixed_shared_bytes";

pub(super) fn validate(ctx: &Context, op: Ptr<Operation>, plan: &LibraryCallPlan) -> Result<()> {
    let library = MirLibraryOp::new(op);
    let LibraryRoleAttr::Native(native) = *library.get_attr_role(ctx).unwrap();
    if String::from((*library.get_attr_library_callee(ctx).unwrap()).clone()) != plan.marker {
        return verify_err!(
            op.deref(ctx).loc(),
            "retained library callee does not match its finalized plan"
        );
    }
    if matches!(native, LibraryNativeMetadata::Raw { .. }) {
        let signature = library
            .get_attr_library_signature(ctx)
            .unwrap()
            .get_type(ctx);
        let args = signature
            .deref(ctx)
            .downcast_ref::<FunctionType>()
            .unwrap()
            .arg_types();
        if args[0]
            .deref(ctx)
            .downcast_ref::<MirTupleType>()
            .unwrap()
            .get_types()
            .len()
            < 2
        {
            return verify_err!(
                op.deref(ctx).loc(),
                "native scratch ABI requires at least two arguments"
            );
        }
    }
    let (checked, resource) = match native {
        LibraryNativeMetadata::Raw { .. } => (false, false),
        LibraryNativeMetadata::CheckedBlock { .. } => (true, false),
        LibraryNativeMetadata::CheckedBlockResource { .. } => (true, true),
    };
    if (plan.checked_block, plan.resource_witness) != (checked, resource)
        || plan.symbol.is_empty()
        || !plan.scratch_alignment.is_power_of_two()
    {
        return verify_err!(
            op.deref(ctx).loc(),
            "native plan does not match the retained source contract"
        );
    }
    if resource && plan.scratch_bytes != 0 {
        return verify_err!(
            op.deref(ctx).loc(),
            "checked resource library calls require scratch-free artifacts"
        );
    }
    Ok(())
}

pub(super) fn lower(ctx: &mut Context, anchor: Ptr<Operation>, plan: &LibraryCallPlan) {
    let LibraryRoleAttr::Native(native) = *MirLibraryOp::new(anchor).get_attr_role(ctx).unwrap();
    native_call(ctx, anchor, native, plan);
}

fn pointer_cast(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    value: Value,
    target: TypeHandle,
    authority: MirPointerKindAuthorityAttr,
) -> Value {
    if value.get_type(ctx) == target {
        return value;
    }
    let op = emit::<MirCastOp>(ctx, anchor, vec![target], vec![value]);
    let cast = MirCastOp::new(op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    cast.set_pointer_kind_authority(ctx, authority);
    op.deref(ctx).get_result(0)
}

/// The unsafe native ABI grants only its raw scratch pointer, never a reference.
#[allow(clippy::disallowed_methods)]
fn scratch_pointer(ctx: &mut Context, pointee: TypeHandle, shared: bool) -> TypeHandle {
    if shared {
        MirPtrType::get_shared_with_kind(ctx, pointee, true, MirPointerKind::RawMut).into()
    } else {
        MirPtrType::get_generic_with_kind(ctx, pointee, true, MirPointerKind::RawMut).into()
    }
}

fn shared_view(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    pointee: TypeHandle,
    plan: &LibraryCallPlan,
) -> Value {
    let ty = MirPtrType::get_shared(ctx, pointee, true).into();
    let op = emit::<MirExternSharedOp>(ctx, anchor, vec![ty], vec![]);
    let shared = MirExternSharedOp::new(op);
    shared.set_byte_offset_value(ctx, 0);
    shared.set_alignment_value(ctx, u64::from(plan.scratch_alignment));
    let layout_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    op.deref_mut(ctx).attributes.set(
        FIXED_SHARED_BYTES.try_into().unwrap(),
        IntegerAttr::new(
            layout_ty,
            APInt::from_u64(
                u64::from(plan.scratch_bytes),
                NonZeroUsize::new(64).unwrap(),
            ),
        ),
    );
    op.deref(ctx).get_result(0)
}

fn flatten(ctx: &mut Context, anchor: Ptr<Operation>, value: Value) -> Vec<Value> {
    let fields = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirTupleType>()
        .unwrap()
        .get_types()
        .to_vec();
    fields
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            let op = emit::<MirExtractFieldOp>(ctx, anchor, vec![ty], vec![value]);
            MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(index as u32));
            op.deref(ctx).get_result(0)
        })
        .collect()
}

fn native_call(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    native: LibraryNativeMetadata,
    plan: &LibraryCallPlan,
) {
    use LibraryNativeMetadata::*;
    let library = MirLibraryOp::new(anchor);
    let signature = library
        .get_attr_library_signature(ctx)
        .unwrap()
        .get_type(ctx);
    let types = signature
        .deref(ctx)
        .downcast_ref::<FunctionType>()
        .unwrap()
        .arg_types();
    let operands = anchor.deref(ctx).operands().collect::<Vec<_>>();
    let indices: &[usize] = match native {
        Raw { .. } => &[0],
        CheckedBlock { .. } => &[0, 1],
        CheckedBlockResource { .. } => &[1, 2],
    };
    let mut arguments = Vec::new();
    let mut declared = Vec::new();
    let mut uniform_len = 0;
    for &index in indices {
        let fields = types[index]
            .deref(ctx)
            .downcast_ref::<MirTupleType>()
            .unwrap()
            .get_types()
            .to_vec();
        if uniform_len == 0 {
            uniform_len = fields.len();
        }
        declared.extend(fields);
        arguments.extend(flatten(ctx, anchor, operands[index]));
    }
    if matches!(native, CheckedBlock { .. } | CheckedBlockResource { .. }) {
        arguments[1..].rotate_left(uniform_len - 1);
        declared[1..].rotate_left(uniform_len - 1);
    }
    let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let mut scratch = shared_view(ctx, anchor, byte, plan);
    let raw_shared = scratch_pointer(ctx, byte, true);
    scratch = pointer_cast(
        ctx,
        anchor,
        scratch,
        raw_shared,
        MirPointerKindAuthorityAttr::RawAddress,
    );
    arguments.insert(2, scratch);
    declared.insert(2, scratch_pointer(ctx, byte, false));
    for (argument, &expected) in arguments.iter_mut().zip(&declared) {
        let actual = argument.get_type(ctx);
        let address_only = {
            let actual = actual.deref(ctx);
            let expected = expected.deref(ctx);
            match (
                actual.downcast_ref::<MirPtrType>(),
                expected.downcast_ref::<MirPtrType>(),
            ) {
                (Some(actual), Some(expected)) => {
                    actual.pointee == expected.pointee
                        && actual.is_mutable == expected.is_mutable
                        && actual.kind == expected.kind
                        && actual.address_space != expected.address_space
                }
                _ => false,
            }
        };
        if address_only {
            *argument = pointer_cast(
                ctx,
                anchor,
                *argument,
                expected,
                MirPointerKindAuthorityAttr::AbiBoundary,
            );
        }
    }
    let signature = FunctionType::get(ctx, declared, vec![]).into();
    let unit = MirTupleType::get(ctx, vec![]).into();
    let call = emit::<MirCallOp>(ctx, anchor, vec![unit], arguments);
    let call = MirCallOp::new(call);
    call.set_attr_callee(ctx, StringAttr::new(plan.symbol.clone()));
    call.set_external_callee_signature(ctx, signature);
}
