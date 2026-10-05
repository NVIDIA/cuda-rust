/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Function-pointer reification.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::MirCastOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use std::num::NonZeroUsize;

/// Lower a `fn item -> fn pointer` coercion (`ReifyFnPointer`).
///
/// Emits a stable per-function token (hash of the function's mangled
/// name, never 0 so it cannot look like a null pointer) and casts it
/// int -> ptr. See the comment at the `Rvalue::Cast` arm for why a token
/// stands in for a code address on the device.
pub(super) fn translate_reify_fn_pointer(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    dest_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    use rustc_public::mir::mono::Instance;
    use std::hash::{Hash, Hasher};

    // The operand's type names the function being reified.
    let operand_ty = operand.ty(body.locals()).map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "ReifyFnPointer: cannot read operand type: {e:?}"
        )))
    })?;
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, substs)) =
        operand_ty.kind()
    else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ReifyFnPointer on a non-fn-item operand of type {operand_ty:?}"
            ))
        );
    };
    let raw_intrinsic =
        crate::translator::terminator::intrinsics::generated::require_supported_raw_intrinsic(
            fn_def, &loc,
        )?;
    let compatibility_path = fn_def.name().as_str().to_string();
    if let Some(path) = raw_intrinsic.or_else(|| {
        crate::translator::terminator::intrinsics::generated::is_generated_intrinsic_path(
            &compatibility_path,
        )
        .then_some(compatibility_path)
    }) {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "generated CUDA intrinsic `{path}` must be called directly and cannot be converted to a function pointer"
            ))
        );
    }
    let mangled = Instance::resolve(fn_def, &substs)
        .map_err(|e| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "ReifyFnPointer: cannot resolve fn item: {e:?}"
            )))
        })?
        .mangled_name();
    let token = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        mangled.hash(&mut h);
        h.finish() | 1
    };

    materialize_function_pointer_token(ctx, dest_ty, token, block_ptr, prev_op, loc)
}

/// Lower a non-capturing `closure -> fn pointer` coercion.
///
/// Like named function items, a closure value is zero-sized and contains no
/// code address to extract. Resolve rustc's `FnOnce` closure shim and use its
/// mangled identity to create the same non-null comparison token used by
/// `translate_reify_fn_pointer`.
pub(super) fn translate_closure_fn_pointer(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    dest_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    use rustc_public::{
        mir::mono::Instance,
        ty::{ClosureKind, RigidTy, TyKind},
    };
    use std::hash::{Hash, Hasher};

    let operand_ty = operand.ty(body.locals()).map_err(|error| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "ClosureFnPointer: cannot read operand type: {error:?}"
        )))
    })?;
    let TyKind::RigidTy(RigidTy::Closure(closure_def, substs)) = operand_ty.kind() else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ClosureFnPointer on a non-closure operand of type {operand_ty:?}"
            ))
        );
    };
    let mangled = Instance::resolve_closure(closure_def, &substs, ClosureKind::FnOnce)
        .map_err(|error| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "ClosureFnPointer: cannot resolve closure shim: {error:?}"
            )))
        })?
        .mangled_name();
    let token = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mangled.hash(&mut hasher);
        hasher.finish() | 1
    };

    materialize_function_pointer_token(ctx, dest_ty, token, block_ptr, prev_op, loc)
}

fn materialize_function_pointer_token(
    ctx: &mut Context,
    dest_ty: &rustc_public::ty::Ty,
    token: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirConstantOp;

    // Materialize the token and cast it to the fn-pointer type, the same
    // two-op shape used for provenance-carrying pointer constants.
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let apint = APInt::from_u64(token, NonZeroUsize::new(64).unwrap());
    let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let int_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    int_op.deref_mut(ctx).set_loc(loc.clone());
    MirConstantOp::new(int_op).set_attr_value(ctx, int_attr);
    match prev_op {
        Some(prev) => int_op.insert_after(ctx, prev),
        None => int_op.insert_at_front(block_ptr, ctx),
    }
    let int_val = int_op.deref(ctx).get_result(0);

    let result_type = types::translate_type(ctx, dest_ty)?;
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![result_type],
        vec![int_val],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    let cast = MirCastOp::new(cast_op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);

    let result = cast_op.deref(ctx).get_result(0);
    Ok((Some(cast_op), result, Some(int_op)))
}

// Byte-offset lookups over rustc enum layout live in the shared
// `translator::layout` module so type import and constant decoding cannot
// drift on how an offset is derived.
