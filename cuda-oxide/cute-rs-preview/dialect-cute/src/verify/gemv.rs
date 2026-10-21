/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed-value and scale-view contracts for the block-scaled GEMV flow.

use pliron::{
    context::{Context, Ptr},
    op::{Op, OpId},
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use super::{Result, ghosts::gemv_semantic_ids, invalid};
use crate::{
    attributes::CuteTensorRoleAttr,
    ops::gemv::{
        CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp, CuteScaledViewRowOp,
        CuteTensorMake2DOp,
    },
    types::CuteTensorViewType,
};

/// Runtime extents and coordinate role recovered from one tensor constructor.
#[derive(Clone, Copy)]
struct GemvTensorState {
    /// Runtime row-count SSA value of the original tensor.
    rows: Value,
    /// Runtime K-extent SSA value of the original tensor.
    k: Value,
    /// Whether rows select matrix M or vector N coordinates.
    role: CuteTensorRoleAttr,
}

/// The original value/scale tensors and the selections applied to their joint view.
#[derive(Clone, Copy)]
struct GemvSelectionState {
    /// Origin and extents of the packed value tensor.
    values: GemvTensorState,
    /// Origin and extents of the scale tensor.
    scales: GemvTensorState,
    /// Selected batch coordinate, absent on the full view.
    batch: Option<Value>,
    /// Selected row coordinate, absent before row selection.
    row: Option<Value>,
    /// Selected K-tile coordinate, absent before tiling.
    tile_index: Option<Value>,
}

/// Require a direct two-dimensional tensor constructor and recover its extents and role.
fn resolve_gemv_tensor(ctx: &Context, value: Value) -> Result<GemvTensorState> {
    let Some(producer) = value.defining_op() else {
        return Err(invalid(
            "GEMV tensor reached a block argument; direct semantic SSA is required",
        ));
    };
    let opid = Operation::get_opid(producer, ctx);
    if opid != CuteTensorMake2DOp::get_opid_static() {
        return Err(invalid(format!(
            "GEMV tensor is produced by `{opid}` instead of `cute.tensor_make_2d`"
        )));
    }
    let op = producer.deref(ctx);
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let view = ty
        .downcast_ref::<CuteTensorViewType>()
        .ok_or_else(|| invalid("cute.tensor_make_2d result is not a tensor view"))?;
    Ok(GemvTensorState {
        rows: op.get_operand(2),
        k: op.get_operand(3),
        role: view.role,
    })
}

/// Walk the make/row/K-tile chain, bounding recursion and preserving selection identity.
fn resolve_gemv_selection(ctx: &Context, value: Value, depth: usize) -> Result<GemvSelectionState> {
    if depth > 16 {
        return Err(invalid(
            "GEMV scaled-view producer chain is unexpectedly deep",
        ));
    }
    let Some(producer) = value.defining_op() else {
        return Err(invalid(
            "GEMV scaled view reached a block argument; direct semantic SSA is required",
        ));
    };
    let opid = Operation::get_opid(producer, ctx);
    let op = producer.deref(ctx);
    if opid == CuteScaledViewMakeOp::get_opid_static() {
        let values = resolve_gemv_tensor(ctx, op.get_operand(0))?;
        let scales = resolve_gemv_tensor(ctx, op.get_operand(1))?;
        if values.rows != scales.rows || values.k != scales.k {
            return Err(invalid(
                "cute.scaled_view_make values and scales must use the same runtime rows and K operands",
            ));
        }
        return Ok(GemvSelectionState {
            values,
            scales,
            batch: None,
            row: None,
            tile_index: None,
        });
    }
    if opid == CuteScaledViewRowOp::get_opid_static() {
        let mut state = resolve_gemv_selection(ctx, op.get_operand(0), depth + 1)?;
        if state.batch.is_some() || state.row.is_some() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.scaled_view_row must select one row before selecting a K tile",
            ));
        }
        state.batch = Some(op.get_operand(1));
        state.row = Some(op.get_operand(2));
        return Ok(state);
    }
    if opid == CuteScaledViewKTileOp::get_opid_static() {
        let mut state = resolve_gemv_selection(ctx, op.get_operand(0), depth + 1)?;
        if state.batch.is_none() || state.row.is_none() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.scaled_view_k_tile must follow exactly one row selection",
            ));
        }
        state.tile_index = Some(op.get_operand(1));
        return Ok(state);
    }
    Err(invalid(format!(
        "scaled GEMV view is produced by unsupported operation `{opid}`"
    )))
}

/// Require a load from one complete, consistently selected block-scaled K tile.
fn resolve_gemv_fragment(ctx: &Context, value: Value) -> Result {
    let Some(load) = value.defining_op() else {
        return Err(invalid(
            "GEMV fragment reached a block argument; a direct load -> dot edge is required",
        ));
    };
    let opid = Operation::get_opid(load, ctx);
    if opid != CuteScaledViewLoadOp::get_opid_static() {
        return Err(invalid(format!(
            "GEMV fragment is produced by `{opid}` instead of `cute.scaled_view_load`"
        )));
    }
    let typed = CuteScaledViewLoadOp::wrap(load);
    let value_alignment = typed
        .promised_value_alignment(ctx)
        .ok_or_else(|| invalid("cute.scaled_view_load is missing its value alignment promise"))?;
    let scale_alignment = typed
        .promised_scale_alignment(ctx)
        .ok_or_else(|| invalid("cute.scaled_view_load is missing its scale alignment promise"))?;
    if value_alignment < 16 || scale_alignment < 4 {
        return Err(invalid(
            "GEMV K=64 load needs value alignment >= 16 and scale alignment >= 4",
        ));
    }
    let state = resolve_gemv_selection(ctx, load.deref(ctx).get_operand(0), 0)?;
    if state.batch.is_none() || state.row.is_none() || state.tile_index.is_none() {
        return Err(invalid("GEMV load needs a selected row and K tile"));
    }
    if state.values.role != state.scales.role {
        return Err(invalid("GEMV values and scales must keep the same role"));
    }
    Ok(())
}

/// Require the producer's ghost result to have exactly one user of the expected operation kind.
fn require_one_user(ctx: &Context, producer: Ptr<Operation>, expected: &OpId) -> Result {
    let result = producer.deref(ctx).get_result(0);
    let uses = result.uses(ctx);
    if uses.len() != 1
        || Operation::get_opid(uses[0].user_op(), ctx) != *expected
        || uses[0].find_index(ctx) >= uses[0].user_op().deref(ctx).get_num_operands()
    {
        return Err(invalid(format!(
            "GEMV semantic value from `{}` must have exactly one `{expected}` consumer",
            Operation::get_opid(producer, ctx)
        )));
    }
    Ok(())
}

/// Check direct ghost chains and compatible matrix/vector selections for each dot product.
pub(super) fn verify_gemv_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = gemv_semantic_ids();
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        let expected = if opid == ids[0] {
            Some(&ids[1])
        } else if opid == ids[1] {
            Some(&ids[2])
        } else if opid == ids[2] {
            Some(&ids[3])
        } else if opid == ids[3] {
            Some(&ids[4])
        } else if opid == ids[4] {
            Some(&ids[5])
        } else {
            None
        };
        if let Some(expected) = expected {
            require_one_user(ctx, *operation, expected)?;
        } else if opid == ids[5] {
            let op = operation.deref(ctx);
            resolve_gemv_fragment(ctx, op.get_operand(0))?;
            resolve_gemv_fragment(ctx, op.get_operand(1))?;
        }
    }
    Ok(())
}
