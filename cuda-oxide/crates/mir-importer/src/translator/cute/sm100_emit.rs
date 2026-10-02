/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Import SM100 CuTe contracts before any target instruction is selected.
//!
//! Layouts come from structural Rust type arguments. The ordinary scalar
//! carriers keep source control flow intact; only these recognized unsafe
//! boundaries recover the caller's shared-address-space promise.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue::translate_operand;
use crate::translator::values::ValueMap;
use dialect_cute::attributes::{CuteComposedLayoutAttr, CuteTmaStorePipelineAttr};
use dialect_cute::sm100_ops::*;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::MirCastOp;
use dialect_mir::types::{MirPtrType, address_space};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;
use rustc_public::ty::{GenericArgKind, GenericArgs, RigidTy, TyKind};

use super::smem_mma_emit::{finish, insert};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sm100Fn {
    TmemAlloc,
    TmemDealloc,
    TiledMma,
    TmemEpilogue,
    ClusterTmaLoad,
    TmaStore,
    StoreCommit,
    StoreAcquire,
    StoreTail,
    PipelineInit,
    PipelineAcquire,
    PipelineExpect,
    PipelineWait,
    PipelineRelease,
    AccumulatorInit,
    AccumulatorAcquire,
    AccumulatorCommit,
    AccumulatorWait,
    AccumulatorRelease,
}

fn invalid(message: impl core::fmt::Display) -> pliron::result::Error {
    pliron::input_error_noloc!(TranslationErr::unsupported(format!(
        "invalid SM100 CuTe boundary: {message}"
    )))
}

fn generics(func: &mir::Operand) -> TranslationResult<GenericArgs> {
    let mir::Operand::Constant(constant) = func else {
        return Err(invalid("marker requires a direct constant callee"));
    };
    let TyKind::RigidTy(RigidTy::FnDef(_, args)) = constant.const_.ty().kind() else {
        return Err(invalid("marker callee must be a function definition"));
    };
    Ok(args)
}

fn constant(args: &GenericArgs, index: usize, loc: &Location) -> TranslationResult<u32> {
    let Some(GenericArgKind::Const(value)) = args.0.get(index) else {
        return Err(invalid(format!(
            "generic argument {index} must be a constant"
        )));
    };
    let value = super::layout::const_u64(value, "SM100 plan", loc)?;
    u32::try_from(value).map_err(|_| invalid("SM100 plan constant exceeds u32"))
}

fn layout(args: &GenericArgs, index: usize) -> TranslationResult<CuteComposedLayoutAttr> {
    let Some(GenericArgKind::Type(ty)) = args.0.get(index) else {
        return Err(invalid(format!(
            "generic argument {index} must be a layout type"
        )));
    };
    super::static_config::decode_smem_layout(ty)
        .map(CuteComposedLayoutAttr)
        .map_err(invalid)
}

fn pipeline_plan(
    args: &GenericArgs,
    loc: &Location,
) -> TranslationResult<CuteSm100PipelinePlanAttr> {
    Ok(CuteSm100PipelinePlanAttr {
        stages: constant(args, 0, loc)?,
        transaction_bytes: constant(args, 1, loc)?,
        consumer_arrivals: 1,
        cta_group: 2,
        leader_rank: 0,
        multicast_mask: 3,
    })
}

fn accumulator_plan(
    args: &GenericArgs,
    loc: &Location,
) -> TranslationResult<CuteSm100PipelinePlanAttr> {
    let consumers = constant(args, 0, loc)?
        .checked_mul(2)
        .ok_or_else(|| invalid("accumulator consumer arrival count overflows"))?;
    Ok(CuteSm100PipelinePlanAttr {
        stages: 1,
        transaction_bytes: 0,
        consumer_arrivals: consumers,
        cta_group: 2,
        leader_rank: 0,
        multicast_mask: 3,
    })
}

fn shared_operands(kind: Sm100Fn) -> &'static [usize] {
    use Sm100Fn::*;
    match kind {
        TmemAlloc => &[0],
        TmemDealloc | StoreCommit | StoreAcquire | StoreTail => &[],
        TmaStore => &[1],
        TiledMma => &[1, 2, 3],
        TmemEpilogue => &[1],
        ClusterTmaLoad => &[1, 4],
        PipelineInit | AccumulatorInit => &[0, 1],
        PipelineAcquire | PipelineExpect | PipelineWait | PipelineRelease | AccumulatorAcquire
        | AccumulatorCommit | AccumulatorWait | AccumulatorRelease => &[0],
    }
}

fn operand_count(kind: Sm100Fn) -> usize {
    use Sm100Fn::*;
    match kind {
        TmemAlloc | TmemDealloc | AccumulatorCommit | AccumulatorRelease => 1,
        PipelineInit | PipelineExpect | PipelineRelease | AccumulatorInit | AccumulatorAcquire
        | AccumulatorWait => 2,
        PipelineAcquire | PipelineWait => 3,
        TmemEpilogue | TmaStore => 4,
        StoreCommit | StoreAcquire | StoreTail => 0,
        TiledMma => 5,
        ClusterTmaLoad => 6,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    kind: Sm100Fn,
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
    use Sm100Fn::*;
    let operation = match kind {
        TmemAlloc | TmemDealloc => {
            let plan = CuteSm100TmemPlanAttr {
                columns: constant(&static_args, 0, &loc)?,
                cta_group: 2,
            };
            if kind == TmemAlloc {
                CuteSm100TmemAllocOp::new(ctx, operands, plan).get_operation()
            } else {
                CuteSm100TmemDeallocOp::new(ctx, operands, plan).get_operation()
            }
        }
        TiledMma => {
            let plan = CuteSm100MmaPlanAttr {
                m: 256,
                n0: constant(&static_args, 3, &loc)?,
                n1: constant(&static_args, 4, &loc)?,
                k: constant(&static_args, 5, &loc)?,
                tmem_columns: 512,
                cta_group: 2,
                collector: CuteSm100CollectorAttr::AFillLastUse,
                a_layout: layout(&static_args, 0)?,
                b0_layout: layout(&static_args, 1)?,
                b1_layout: layout(&static_args, 2)?,
            };
            CuteSm100TiledMmaOp::new(ctx, operands, plan).get_operation()
        }
        TmemEpilogue => CuteSm100TmemEpilogueOp::new(
            ctx,
            operands,
            CuteSm100EpiloguePlanAttr {
                layout: layout(&static_args, 0)?,
                rows: 128,
                columns: 32,
            },
        )
        .get_operation(),
        ClusterTmaLoad => CuteSm100ClusterTmaLoadOp::new(
            ctx,
            operands,
            CuteSm100ClusterCopyPlanAttr {
                layout: layout(&static_args, 0)?,
                cta_group: 2,
                leader_rank: 0,
            },
        )
        .get_operation(),
        TmaStore => CuteSm100TmaStoreOp::new(
            ctx,
            operands,
            CuteSm100EpiloguePlanAttr {
                layout: layout(&static_args, 0)?,
                rows: 128,
                columns: 32,
            },
        )
        .get_operation(),
        StoreCommit => CuteSm100StoreCommitOp::new(
            ctx,
            operands,
            CuteTmaStorePipelineAttr {
                stages: constant(&static_args, 0, &loc)?,
            },
        )
        .get_operation(),
        StoreAcquire => CuteSm100StoreAcquireOp::new(
            ctx,
            operands,
            CuteTmaStorePipelineAttr {
                stages: constant(&static_args, 0, &loc)?,
            },
        )
        .get_operation(),
        StoreTail => CuteSm100StoreTailOp::new(
            ctx,
            operands,
            CuteTmaStorePipelineAttr {
                stages: constant(&static_args, 0, &loc)?,
            },
        )
        .get_operation(),
        PipelineInit => {
            CuteSm100PipelineInitOp::new(ctx, operands, pipeline_plan(&static_args, &loc)?)
                .get_operation()
        }
        PipelineAcquire => {
            CuteSm100PipelineAcquireOp::new(ctx, operands, pipeline_plan(&static_args, &loc)?)
                .get_operation()
        }
        PipelineExpect => {
            CuteSm100PipelineExpectOp::new(ctx, operands, pipeline_plan(&static_args, &loc)?)
                .get_operation()
        }
        PipelineWait => {
            CuteSm100PipelineWaitOp::new(ctx, operands, pipeline_plan(&static_args, &loc)?)
                .get_operation()
        }
        PipelineRelease => {
            CuteSm100PipelineReleaseOp::new(ctx, operands, pipeline_plan(&static_args, &loc)?)
                .get_operation()
        }
        AccumulatorInit => {
            CuteSm100AccumulatorInitOp::new(ctx, operands, accumulator_plan(&static_args, &loc)?)
                .get_operation()
        }
        AccumulatorAcquire => {
            CuteSm100AccumulatorAcquireOp::new(ctx, operands, accumulator_plan(&static_args, &loc)?)
                .get_operation()
        }
        AccumulatorCommit => {
            CuteSm100AccumulatorCommitOp::new(ctx, operands, accumulator_plan(&static_args, &loc)?)
                .get_operation()
        }
        AccumulatorWait => {
            CuteSm100AccumulatorWaitOp::new(ctx, operands, accumulator_plan(&static_args, &loc)?)
                .get_operation()
        }
        AccumulatorRelease => {
            CuteSm100AccumulatorReleaseOp::new(ctx, operands, accumulator_plan(&static_args, &loc)?)
                .get_operation()
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
        "SM100 CuTe operation",
    )
}
