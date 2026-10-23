/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Descriptor-backed raw transport views and their direct copy contracts.

use pliron::{
    context::{Context, Ptr},
    op::{Op, OpId},
    operation::Operation,
    value::Value,
};

use super::{Result, invalid};
use crate::{
    ops::epilogue::CuteTmaStore2dSemanticOp,
    ops::gemm_tma::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp},
    ops::pipeline::CutePipelineProducerExpectTxOp,
};

/// Require a view operand to come directly from the expected constructor in this function scope.
pub(super) fn direct_tma_view_producer(
    ctx: &Context,
    value: Value,
    expected: &OpId,
    what: &str,
    scope: &[Ptr<Operation>],
) -> Result<Ptr<Operation>> {
    let Some(producer) = value.defining_op() else {
        return Err(invalid(format!(
            "semantic TMA {what} reached a block argument; a direct view producer is required"
        )));
    };
    let actual = Operation::get_opid(producer, ctx);
    if actual != *expected {
        return Err(invalid(format!(
            "semantic TMA {what} is produced by `{actual}` instead of `{expected}`"
        )));
    }
    if !scope.contains(&producer) {
        return Err(invalid(format!(
            "semantic TMA {what} is produced outside its function/story scope"
        )));
    }
    Ok(producer)
}

/// Restrict global/shared view ghosts to the matching TMA copy operand positions.
fn verify_tma_view_uses(ctx: &Context, producer: Ptr<Operation>, gmem: bool) -> Result {
    let value = producer.deref(ctx).get_result(0);
    let uses = value.uses(ctx);
    if uses.is_empty() {
        return Err(invalid(format!(
            "semantic TMA view from `{}` has no copy or store consumer",
            Operation::get_opid(producer, ctx)
        )));
    }
    for r#use in uses {
        let user = r#use.user_op();
        let user_id = Operation::get_opid(user, ctx);
        let index = r#use.find_index(ctx);
        let allowed = if gmem {
            (user_id == CuteTmaCopy2dOp::get_opid_static() && index == 0)
                || (user_id == CuteTmaStore2dSemanticOp::get_opid_static() && index == 1)
        } else {
            (user_id == CuteTmaCopy2dOp::get_opid_static() && index == 1)
                || (user_id == CuteTmaStore2dSemanticOp::get_opid_static() && index == 0)
        };
        if !allowed {
            return Err(invalid(format!(
                "semantic TMA view from `{}` has unsupported use by `{user_id}` operand {index}",
                Operation::get_opid(producer, ctx)
            )));
        }
    }
    Ok(())
}

/// Check direct global/shared view pairing and reject unused or escaping transport views.
pub(super) fn verify_tma_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if opid == CuteTmaGmemViewOp::get_opid_static() {
            verify_tma_view_uses(ctx, *operation, true)?;
        } else if opid == CuteTmaSmemViewOp::get_opid_static() {
            verify_tma_view_uses(ctx, *operation, false)?;
        } else if opid == CuteTmaCopy2dOp::get_opid_static() {
            let copy = CuteTmaCopy2dOp::wrap(*operation);
            direct_tma_view_producer(
                ctx,
                copy.source(ctx),
                &CuteTmaGmemViewOp::get_opid_static(),
                "load source",
                all_ops,
            )?;
            direct_tma_view_producer(
                ctx,
                copy.destination(ctx),
                &CuteTmaSmemViewOp::get_opid_static(),
                "load destination",
                all_ops,
            )?;
            let barrier = copy.completion_barrier(ctx);
            let Some(expect) = barrier.defining_op() else {
                return Err(invalid(
                    "cute.tma_copy_2d completion barrier must come directly from pipeline_producer_expect_tx",
                ));
            };
            if Operation::get_opid(expect, ctx) != CutePipelineProducerExpectTxOp::get_opid_static()
                || !all_ops.contains(&expect)
                || CutePipelineProducerExpectTxOp::wrap(expect).completion_barrier(ctx) != barrier
            {
                return Err(invalid(
                    "cute.tma_copy_2d completion barrier is not the result of an expect_tx in this semantic scope",
                ));
            }
        } else if opid == CuteTmaStore2dSemanticOp::get_opid_static() {
            let store = CuteTmaStore2dSemanticOp::wrap(*operation);
            direct_tma_view_producer(
                ctx,
                store.source(ctx),
                &CuteTmaSmemViewOp::get_opid_static(),
                "store source",
                all_ops,
            )?;
            direct_tma_view_producer(
                ctx,
                store.destination(ctx),
                &CuteTmaGmemViewOp::get_opid_static(),
                "store destination",
                all_ops,
            )?;
        }
    }
    Ok(())
}
