/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Import independent SM100 attention MMA and typed tensor-copy contracts.
//!
//! Layouts come from structural Rust type arguments. The ordinary scalar
//! carriers keep source control flow intact; only these recognized unsafe
//! boundaries recover the caller's shared-address-space promise.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::CuteComposedLayoutAttr;
use dialect_cute::sm100_attention_ops::*;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::MirCastOp;
use dialect_mir::types::{MirPtrType, address_space};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;
use rustc_public::ty::{GenericArgKind, GenericArgs, RigidTy, TyKind};

use super::smem_mma::emit::{finish, insert};

/// Independent SM100 attention hooks recognized by the importer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionFn {
    /// Issue independent shared/shared FP16 MMA into TMEM.
    MmaSs,
    /// Issue independent TMEM/shared FP16 MMA into TMEM.
    MmaTs,
    /// Load a rank-three global tile into the attention shared layout.
    TmaLoad,
    /// Issue a shared-to-global tile store.
    TmaStore,
}
/// Report an unsupported source contract at this CuTe boundary.
fn invalid(message: impl core::fmt::Display) -> pliron::result::Error {
    pliron::input_error_noloc!(TranslationErr::unsupported(format!(
        "invalid SM100 CuTe boundary: {message}"
    )))
}

/// Extract monomorphized generic arguments from a direct constant function callee.
fn generics(func: &mir::Operand) -> TranslationResult<GenericArgs> {
    let mir::Operand::Constant(constant) = func else {
        return Err(invalid("marker requires a direct constant callee"));
    };
    let TyKind::RigidTy(RigidTy::FnDef(_, args)) = constant.const_.ty().kind() else {
        return Err(invalid("marker callee must be a function definition"));
    };
    Ok(args)
}

/// Evaluate one generic constant as a checked `u32` plan parameter.
fn constant(args: &GenericArgs, index: usize, loc: &Location) -> TranslationResult<u32> {
    let Some(GenericArgKind::Const(value)) = args.0.get(index) else {
        return Err(invalid(format!(
            "generic argument {index} must be a constant"
        )));
    };
    let value = super::layout::const_u64(value, "SM100 plan", loc)?;
    u32::try_from(value).map_err(|_| invalid("SM100 plan constant exceeds u32"))
}

/// Decode an independent attention layout and its MN-major selector from a type argument.
fn layout(args: &GenericArgs, index: usize) -> TranslationResult<(CuteComposedLayoutAttr, bool)> {
    let Some(GenericArgKind::Type(ty)) = args.0.get(index) else {
        return Err(invalid("attention layout must be a type argument"));
    };
    super::static_config::decode_attention_layout(ty)
        .map(|(l, m)| (CuteComposedLayoutAttr(l), m))
        .map_err(invalid)
}
/// List the operands whose unsafe source contract permits shared-address-space recovery.
fn shared_operands(kind: AttentionFn) -> &'static [usize] {
    match kind {
        AttentionFn::MmaSs => &[1, 2],
        AttentionFn::MmaTs => &[2],
        AttentionFn::TmaLoad => &[1, 5],
        AttentionFn::TmaStore => &[1],
    }
}
/// Return the exact scalar-carrier count required by this compiler hook.
fn operand_count(kind: AttentionFn) -> usize {
    match kind {
        AttentionFn::MmaSs => 6,
        AttentionFn::MmaTs => 5,
        AttentionFn::TmaLoad => 7,
        AttentionFn::TmaStore => 5,
    }
}
/// Import independent attention MMA or TMA while preserving source control flow and pointer authority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    kind: AttentionFn,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != operand_count(kind) {
        return Err(invalid(format!(
            "{kind:?} expects {} operands, got {}",
            operand_count(kind),
            args.len()
        )));
    }
    let static_args = generics(func)?;
    let mut operands: Vec<Value> = Vec::with_capacity(args.len());
    let mut after = prev;
    for arg in args {
        let (value, last) =
            translate_operand(ctx, body, arg, value_map, block, after, loc.clone())?;
        after = last.or(after);
        operands.push(value);
    }
    for &index in shared_operands(kind) {
        let pointer = operands[index]
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .cloned()
            .ok_or_else(|| invalid(format!("{kind:?} operand {index} must be a shared pointer")))?;
        if !matches!(
            pointer.address_space,
            address_space::GENERIC | address_space::SHARED
        ) {
            return Err(invalid(format!(
                "{kind:?} operand {index} has non-shared address space {}",
                pointer.address_space
            )));
        }
        let shared: TypeHandle = MirPtrType::get(
            ctx,
            pointer.pointee,
            pointer.is_mutable,
            address_space::SHARED,
        )
        .into();
        if operands[index].get_type(ctx) != shared {
            let cast = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![shared],
                vec![operands[index]],
                vec![],
                0,
            );
            cast.deref_mut(ctx).set_loc(loc.clone());
            MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
            if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, shared) {
                MirCastOp::new(cast)
                    .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RawAddress);
            }
            insert(ctx, cast, block, after);
            operands[index] = cast.deref(ctx).get_result(0);
            after = Some(cast);
        }
    }
    let operation = match kind {
        AttentionFn::MmaSs | AttentionFn::MmaTs => {
            let ts = matches!(kind, AttentionFn::MmaTs);
            let offset = if ts { 1 } else { 2 };
            let m = constant(&static_args, offset, &loc)?;
            let n = constant(&static_args, offset + 1, &loc)?;
            let k = constant(&static_args, offset + 2, &loc)?;
            let cta_group = constant(&static_args, offset + 3, &loc)?;
            if !matches!(cta_group, 1 | 2) {
                return Err(invalid("CTA group must be 1 or 2"));
            }
            let (a_layout, a_mn) = if ts {
                (shared_layout(m, k, false).map_err(invalid)?, false)
            } else {
                layout(&static_args, 0)?
            };
            let (b_layout, b_mn) = layout(&static_args, if ts { 0 } else { 1 })?;
            let plan = CuteSm100MmaAttr {
                m,
                n,
                k,
                cta_group,
                a_layout,
                b_layout,
                a_mn,
                b_mn,
            };
            if ts {
                CuteSm100MmaTsOp::new(ctx, operands, plan).get_operation()
            } else {
                CuteSm100MmaSsOp::new(ctx, operands, plan).get_operation()
            }
        }
        AttentionFn::TmaLoad | AttentionFn::TmaStore => {
            let plan = CuteSm100TmaAttr {
                layout: layout(&static_args, 0)?.0,
                cta_group: constant(&static_args, 1, &loc)?,
            };
            if matches!(kind, AttentionFn::TmaLoad) {
                CuteSm100TmaLoadOp::new(ctx, operands, plan).get_operation()
            } else {
                CuteSm100TmaStoreOp::new(ctx, operands, plan).get_operation()
            }
        }
    };
    operation.deref_mut(ctx).set_loc(loc.clone());
    insert(ctx, operation, block, after);
    finish(
        ctx,
        operation,
        target,
        block_map,
        loc,
        "SM100 attention CuTe operation",
    )
}
