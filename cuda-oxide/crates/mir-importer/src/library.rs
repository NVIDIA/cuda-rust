// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lower compiler-validated library calls at an explicit scalar/pointer ABI.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::terminator::helpers;
use crate::translator::values::{ValueMap, establish_declared_pointer_type};
use crate::translator::{facts, rvalue, types};
use dialect_mir::attributes::{FieldIndexAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{MirExternSharedOp, MirExtractFieldOp};
use dialect_mir::types::{MirPtrType, MirTupleType};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use rustc_public::mir;

/// One monomorphized call whose Rust tuple and provider ABI already agree.
#[derive(Clone, Debug)]
pub struct LibraryCallPlan {
    /// Exact external symbol emitted by the native provider.
    pub symbol: String,
    /// Size of the assigned shared scratch view in bytes.
    pub scratch_bytes: u32,
    /// Required alignment of the shared scratch base.
    pub scratch_alignment: u32,
}

thread_local! {
    static CALLS: RefCell<BTreeMap<String, LibraryCallPlan>> = const { RefCell::new(BTreeMap::new()) };
}

pub(crate) fn set_calls(calls: BTreeMap<String, LibraryCallPlan>) {
    CALLS.with(|current| *current.borrow_mut() = calls);
}

pub(crate) fn get_call(instance: &str) -> Option<LibraryCallPlan> {
    CALLS.with(|calls| calls.borrow().get(instance).cloned())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn translate_call(
    ctx: &mut Context,
    body: &mir::Body,
    plan: &LibraryCallPlan,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    previous: Option<Ptr<Operation>>,
    values: &mut ValueMap,
    blocks: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let [_, tuple] = args else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "library marker requires a registration and argument tuple"
            )
        );
    };
    let tuple_ty = match tuple.ty(body.locals()) {
        Ok(ty) => ty,
        Err(e) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!("invalid library argument tuple: {e:?}"))
            );
        }
    };
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(fields)) =
        tuple_ty.kind()
    else {
        return input_err!(
            loc,
            TranslationErr::unsupported("library arguments must be a tuple")
        );
    };
    let mut declared = fields
        .iter()
        .map(|ty| types::translate_type(ctx, ty))
        .collect::<TranslationResult<Vec<_>>>()?;
    let (tuple_value, mut last) =
        rvalue::translate_operand(ctx, body, tuple, values, block, previous, loc.clone())?;
    let tuple_handle = tuple_value.get_type(ctx);
    let field_types = tuple_handle
        .deref(ctx)
        .downcast_ref::<MirTupleType>()
        .expect("validated library tuple")
        .get_types()
        .to_vec();
    let mut arguments = Vec::new();
    for (index, ty) in field_types.into_iter().enumerate() {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![ty],
            vec![tuple_value],
            vec![],
            0,
        );
        op.deref_mut(ctx).set_loc(loc.clone());
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(index as u32));
        helpers::insert_op(ctx, op, block, last);
        last = Some(op);
        arguments.push(op.deref(ctx).get_result(0));
    }
    // The unsafe marker grants a raw scratch address. Keep the internal shared
    // pointer through construction, then adapt only its address space at the
    // declared generic-pointer ABI boundary, using the ordinary foreign-call path.
    let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let shared = MirPtrType::get_shared(ctx, byte, true).into();
    let op = Operation::new(
        ctx,
        MirExternSharedOp::get_concrete_op_info(),
        vec![shared],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    let scratch = MirExternSharedOp::new(op);
    scratch.set_byte_offset_value(ctx, 0);
    scratch.set_alignment_value(ctx, u64::from(plan.scratch_alignment));
    let layout_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    op.deref_mut(ctx).attributes.set(
        crate::kernel_resources::FIXED_SHARED_BYTES
            .try_into()
            .unwrap(),
        IntegerAttr::new(
            layout_ty,
            APInt::from_u64(
                u64::from(plan.scratch_bytes),
                std::num::NonZeroUsize::new(64).unwrap(),
            ),
        ),
    );
    helpers::insert_op(ctx, op, block, last);
    let internal_scratch = op.deref(ctx).get_result(0);
    let raw_shared = facts::mint_shared_ptr_type(ctx, byte, facts::abi_library_scratch()).into();
    let (scratch_value, last) = establish_declared_pointer_type(
        ctx,
        internal_scratch,
        raw_shared,
        block,
        Some(op),
        MirPointerKindAuthorityAttr::RawAddress,
    );
    arguments.insert(2, scratch_value);
    declared.insert(
        2,
        facts::mint_generic_ptr_type(ctx, byte, facts::abi_library_scratch()).into(),
    );
    let signature = FunctionType::get(ctx, declared, vec![]).into();
    let unit = MirTupleType::get(ctx, vec![]).into();
    helpers::emit_function_call_values(
        ctx,
        body,
        &plan.symbol,
        arguments,
        destination,
        unit,
        Some(signature),
        target,
        block,
        last,
        values,
        blocks,
        loc,
    )
}
