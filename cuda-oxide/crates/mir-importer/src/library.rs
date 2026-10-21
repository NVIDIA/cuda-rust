// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lower compiler-validated library calls at an explicit scalar/pointer ABI.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::terminator::helpers;
use crate::translator::values::{ValueMap, establish_declared_pointer_type};
use crate::translator::{facts, rvalue, types};
use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{
    MirCastOp, MirConstantOp, MirExternSharedOp, MirExtractFieldOp, MirInsertFieldOp, MirUndefOp,
};
use dialect_mir::types::{MirPtrType, MirStructType, MirTupleType};
use dialect_nvvm::ops::Barrier0Op;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{IntegerAttr, StringAttr};
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use rustc_public::mir;

/// Internal call targets, keyed by emitted caller and original collected callee
/// export name. Values name the same Rust body in that caller's root context.
pub type CallTargetRedirects = BTreeMap<String, BTreeMap<String, String>>;

/// A terminator in one concrete emitted Rust body (not the original Instance).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibraryCallSite {
    pub caller: String,
    pub block: usize,
}

/// Compiler-owned operation at one exact MIR terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibraryOperation {
    Native,
    StorageNew,
    StorageReconfigure,
    StorageReserve,
    StorageRelease,
    Redirect,
}

/// One monomorphized call whose Rust tuple and provider ABI already agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryCallPlan {
    pub operation: LibraryOperation,
    pub storage_witness: bool,
    pub scratch_offset: u32,
    pub planned_root: Option<String>,
    pub barrier_before: bool,
    pub barrier_after: bool,
    /// Exact external symbol emitted by the native provider.
    pub symbol: String,
    /// Expected marker identity at this exact MIR terminator.
    pub marker: String,
    /// Split uniform/payload operands of the checked full-block marker.
    pub checked_block: bool,
    /// An additional typed memory witness, consumed only by compiler checks.
    pub resource_witness: bool,
    /// A synchronous callback appends its borrowed opaque environment.
    pub callback: bool,
    /// Size of the assigned shared scratch view in bytes.
    pub scratch_bytes: u32,
    /// Required alignment of the shared scratch base.
    pub scratch_alignment: u32,
}

impl LibraryCallPlan {
    pub fn internal(operation: LibraryOperation, marker: String) -> Self {
        Self {
            operation,
            marker,
            symbol: String::new(),
            storage_witness: false,
            scratch_offset: 0,
            planned_root: None,
            barrier_before: false,
            barrier_after: false,
            checked_block: false,
            resource_witness: false,
            callback: false,
            scratch_bytes: 0,
            scratch_alignment: 1,
        }
    }
}

thread_local! {
    static CALLS: RefCell<BTreeMap<LibraryCallSite, LibraryCallPlan>> = const { RefCell::new(BTreeMap::new()) };
}

pub(crate) fn set_calls(calls: BTreeMap<LibraryCallSite, LibraryCallPlan>) {
    CALLS.with(|current| *current.borrow_mut() = calls);
}

pub(crate) fn get_call(caller: &str, block: usize) -> Option<LibraryCallPlan> {
    CALLS.with(|calls| {
        calls
            .borrow()
            .get(&LibraryCallSite {
                caller: caller.to_string(),
                block,
            })
            .cloned()
    })
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
    if plan.operation != LibraryOperation::Native {
        return translate_storage(
            ctx,
            body,
            plan,
            destination,
            target,
            block,
            previous,
            values,
            blocks,
            loc,
        );
    }
    let mut previous = previous;
    if plan.barrier_before {
        previous = Some(barrier(ctx, block, previous, loc.clone()));
    }
    let tuples: &[mir::Operand] = if plan.storage_witness && args.len() == 5 {
        &args[3..]
    } else {
        match (
            plan.checked_block,
            plan.resource_witness,
            plan.callback,
            args,
        ) {
            (false, false, false, [_, _]) | (true, false, false, [_, _, _]) => &args[1..],
            (true, true, false, [_, _, _, _]) => &args[2..],
            (false, false, true, [_, _, _, _]) => &args[1..2],
            _ => {
                return input_err!(
                    loc,
                    TranslationErr::invalid_op(
                        "library marker operand count changed after validation"
                    )
                );
            }
        }
    };
    let mut declared = Vec::new();
    let mut arguments = Vec::new();
    let mut last = previous;
    let mut uniform_len = 0;
    for tuple in tuples {
        let tuple_ty = tuple.ty(body.locals()).map_err(|e| {
            pliron::input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!("invalid library tuple: {e:?}"))
            )
        })?;
        let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(fields)) =
            tuple_ty.kind()
        else {
            return input_err!(
                loc,
                TranslationErr::unsupported("library arguments must be tuples")
            );
        };
        declared.extend(
            fields
                .iter()
                .map(|ty| types::translate_type(ctx, ty))
                .collect::<TranslationResult<Vec<_>>>()?,
        );
        let (tuple_value, previous) =
            rvalue::translate_operand(ctx, body, tuple, values, block, last, loc.clone())?;
        last = previous;
        let handle = tuple_value.get_type(ctx);
        let field_types = handle
            .deref(ctx)
            .downcast_ref::<MirTupleType>()
            .expect("validated library tuple")
            .get_types()
            .to_vec();
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
        if uniform_len == 0 {
            uniform_len = fields.len();
        }
    }
    if plan.checked_block {
        // [memory, controls..., payload...] -> [memory, payload..., controls...]
        declared[1..].rotate_left(uniform_len - 1);
        arguments[1..].rotate_left(uniform_len - 1);
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
    scratch.set_byte_offset_value(ctx, u64::from(plan.scratch_offset));
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
    tag_planned(ctx, op, plan);
    helpers::insert_op(ctx, op, block, last);
    let internal_scratch = op.deref(ctx).get_result(0);
    let raw_shared = facts::mint_shared_ptr_type(ctx, byte, facts::abi_library_scratch()).into();
    let (scratch_value, mut last) = establish_declared_pointer_type(
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
    if plan.callback {
        let (environment, previous) =
            rvalue::translate_operand(ctx, body, &args[3], values, block, last, loc.clone())?;
        // The environment's Rust layout stays private to the thunk. This one
        // validated ABI boundary converts &Environment to an opaque *const u8;
        // it neither grants mutation nor fabricates a Rust reference.
        let origin = facts::pointer_origin_of_raw_ptr(mir::RawPtrKind::Const);
        let pointee = environment
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .expect("validated environment reference")
            .pointee;
        let raw_environment = facts::mint_generic_ptr_type(ctx, pointee, origin).into();
        let (environment, previous) = establish_declared_pointer_type(
            ctx,
            environment,
            raw_environment,
            block,
            previous,
            MirPointerKindAuthorityAttr::RawAddress,
        );
        let opaque = facts::mint_generic_ptr_type(ctx, byte, origin).into();
        let cast_op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![opaque],
            vec![environment],
            vec![],
            0,
        );
        cast_op.deref_mut(ctx).set_loc(loc.clone());
        let cast = MirCastOp::new(cast_op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        helpers::insert_op(ctx, cast_op, block, previous);
        last = Some(cast_op);
        arguments.push(cast_op.deref(ctx).get_result(0));
        declared.push(opaque);
    }
    let signature = FunctionType::get(ctx, declared, vec![]).into();
    let unit = MirTupleType::get(ctx, vec![]).into();
    let goto = helpers::emit_function_call_values(
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
        loc.clone(),
    )?;
    if plan.barrier_after {
        let sync = barrier_op(ctx, loc);
        sync.insert_before(ctx, goto);
    }
    Ok(goto)
}

fn barrier_op(ctx: &mut Context, loc: Location) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    helpers::set_generated_intrinsic_marker(ctx, op, "v1:i0034");
    op
}
fn barrier(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    previous: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    let op = barrier_op(ctx, loc);
    helpers::insert_op(ctx, op, block, previous);
    op
}
fn tag_planned(ctx: &mut Context, op: Ptr<Operation>, plan: &LibraryCallPlan) {
    if let Some(root) = &plan.planned_root {
        op.deref_mut(ctx).attributes.set(
            crate::kernel_resources::PLANNED_ROOT.try_into().unwrap(),
            StringAttr::new(root.clone()),
        );
    }
}
#[allow(clippy::too_many_arguments)]
fn translate_storage(
    ctx: &mut Context,
    body: &mir::Body,
    plan: &LibraryCallPlan,
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    previous: Option<Ptr<Operation>>,
    values: &mut ValueMap,
    blocks: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let result_ty = types::translate_destination_type(ctx, body, destination, &loc)?;
    let (prepared, mut previous) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        values,
        block,
        previous,
        loc.clone(),
    )?;
    let result = if plan.operation == LibraryOperation::StorageRelease {
        let op = MirUndefOp::new(ctx, result_ty).get_operation();
        helpers::insert_op(ctx, op, block, previous);
        previous = Some(op);
        op.deref(ctx).get_result(0)
    } else {
        let field = result_ty
            .deref(ctx)
            .downcast_ref::<MirStructType>()
            .and_then(|ty| ty.get_field_type(0))
            .ok_or_else(|| {
                pliron::input_error!(
                    loc.clone(),
                    TranslationErr::invalid_op("storage result must have one private field")
                )
            })?;
        let value = if plan.operation == LibraryOperation::StorageReserve {
            let pointee = field
                .deref(ctx)
                .downcast_ref::<MirPtrType>()
                .ok_or_else(|| {
                    pliron::input_error!(
                        loc.clone(),
                        TranslationErr::invalid_op("reservation field must be a raw pointer")
                    )
                })?
                .pointee;
            let shared = MirPtrType::get_shared(ctx, pointee, true).into();
            let op = Operation::new(
                ctx,
                MirExternSharedOp::get_concrete_op_info(),
                vec![shared],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let shared = MirExternSharedOp::new(op);
            shared.set_byte_offset_value(ctx, u64::from(plan.scratch_offset));
            shared.set_alignment_value(ctx, u64::from(plan.scratch_alignment));
            let ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
            op.deref_mut(ctx).attributes.set(
                crate::kernel_resources::FIXED_SHARED_BYTES
                    .try_into()
                    .unwrap(),
                IntegerAttr::new(
                    ty,
                    APInt::from_u64(
                        u64::from(plan.scratch_bytes),
                        std::num::NonZeroUsize::new(64).unwrap(),
                    ),
                ),
            );
            tag_planned(ctx, op, plan);
            helpers::insert_op(ctx, op, block, previous);
            let internal = op.deref(ctx).get_result(0);
            let (value, last) = establish_declared_pointer_type(
                ctx,
                internal,
                field,
                block,
                Some(op),
                MirPointerKindAuthorityAttr::RawAddress,
            );
            previous = last;
            value
        } else {
            let ty = IntegerType::get(ctx, 8, Signedness::Unsigned);
            let op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![field],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(op).set_attr_value(
                ctx,
                IntegerAttr::new(
                    ty,
                    APInt::from_u64(0, std::num::NonZeroUsize::new(8).unwrap()),
                ),
            );
            helpers::insert_op(ctx, op, block, previous);
            previous = Some(op);
            op.deref(ctx).get_result(0)
        };
        let undef = MirUndefOp::new(ctx, result_ty).get_operation();
        helpers::insert_op(ctx, undef, block, previous);
        let undef_value = undef.deref(ctx).get_result(0);
        let op = Operation::new(
            ctx,
            MirInsertFieldOp::get_concrete_op_info(),
            vec![result_ty],
            vec![undef_value, value],
            vec![],
            0,
        );
        MirInsertFieldOp::new(op).set_attr_insert_index(ctx, FieldIndexAttr(0));
        helpers::insert_op(ctx, op, block, Some(undef));
        previous = Some(op);
        op.deref(ctx).get_result(0)
    };
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared,
        result,
        target,
        block,
        previous.unwrap(),
        values,
        blocks,
        loc,
        "storage marker requires a continuation",
    )
}

/// Pure canonical rank query, deliberately independent of validated native plans.
#[allow(clippy::too_many_arguments)]
pub(crate) fn translate_warp_query(
    ctx: &mut Context,
    body: &mir::Body,
    width: u32,
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    previous: Option<Ptr<Operation>>,
    values: &mut ValueMap,
    blocks: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirAddOp, MirDivOp, MirMulOp};
    use dialect_nvvm::ops::{
        ReadPtxSregNtidXOp, ReadPtxSregNtidYOp, ReadPtxSregTidXOp, ReadPtxSregTidYOp,
        ReadPtxSregTidZOp,
    };
    let (prepared, mut previous) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        values,
        block,
        previous,
        loc.clone(),
    )?;
    let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    macro_rules! operation {
        ($op:ty, $args:expr) => {{
            let op = Operation::new(
                ctx,
                <$op>::get_concrete_op_info(),
                vec![ty.into()],
                $args,
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            helpers::insert_op(ctx, op, block, previous);
            previous = Some(op);
            op
        }};
    }
    macro_rules! register {
        ($op:ty, $marker:literal) => {{
            let op = operation!($op, vec![]);
            helpers::set_generated_intrinsic_marker(ctx, op, $marker);
            op.deref(ctx).get_result(0)
        }};
    }
    let x = register!(ReadPtxSregTidXOp, "v1:i0001");
    let y = register!(ReadPtxSregTidYOp, "v1:i0005");
    let z = register!(ReadPtxSregTidZOp, "v1:i0009");
    let nx = register!(ReadPtxSregNtidXOp, "v1:i0003");
    let ny = register!(ReadPtxSregNtidYOp, "v1:i0007");
    let yz = operation!(MirMulOp, vec![ny, z]).deref(ctx).get_result(0);
    let yz = operation!(MirAddOp, vec![y, yz]).deref(ctx).get_result(0);
    let rank = operation!(MirMulOp, vec![nx, yz]).deref(ctx).get_result(0);
    let rank = operation!(MirAddOp, vec![x, rank]).deref(ctx).get_result(0);
    let constant = operation!(MirConstantOp, vec![]);
    MirConstantOp::new(constant).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(u64::from(width), std::num::NonZeroUsize::new(32).unwrap()),
        ),
    );
    let divisor = constant.deref(ctx).get_result(0);
    let division = operation!(MirDivOp, vec![rank, divisor]);
    let result = division.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared,
        result,
        target,
        block,
        previous.unwrap(),
        values,
        blocks,
        loc,
        "Warp query has no successor",
    )
}
