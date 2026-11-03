/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor storage, access, format, and selection metadata.

use pliron::{
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use super::{Result, ghosts::tensor_semantic_ids, invalid};
use crate::{
    ops::tensor::{
        CuteTensorMakeOp, CuteTensorSliceOp, CuteTensorStoreFromOp, CuteTensorZippedDivideOp,
    },
    types::CuteTensorViewType,
};

/// Tile width and selection recovered from a direct tensor-view chain.
#[derive(Clone, Copy)]
struct TensorViewState {
    /// Static tile width in elements, absent before zipped division.
    tile_size: Option<u64>,
    /// Runtime selected tile number, absent before slicing.
    tile_index: Option<Value>,
}

/// Walk direct make/divide/slice producers and reject unsupported or excessively deep chains.
fn resolve_tensor_view(ctx: &Context, value: Value, depth: usize) -> Result<TensorViewState> {
    if depth > 16 {
        return Err(invalid(
            "CuTe tensor-view producer chain is unexpectedly deep",
        ));
    }
    let Some(defining_op) = value.defining_op() else {
        return Err(invalid(
            "CuTe tensor view reached a block argument; a direct make/divide/slice chain is required",
        ));
    };
    let opid = Operation::get_opid(defining_op, ctx);
    if opid == CuteTensorMakeOp::get_opid_static() {
        return Ok(TensorViewState {
            tile_size: None,
            tile_index: None,
        });
    }
    if opid == CuteTensorZippedDivideOp::get_opid_static() {
        let mut state = resolve_tensor_view(ctx, defining_op.deref(ctx).get_operand(0), depth + 1)?;
        if state.tile_size.is_some() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.tensor_zipped_divide must occur exactly once before tensor_slice",
            ));
        }
        let result_ty = value.get_type(ctx);
        let result_ty = result_ty.deref(ctx);
        let view = result_ty
            .downcast_ref::<CuteTensorViewType>()
            .ok_or_else(|| invalid("cute.tensor_zipped_divide result is not a tensor view"))?;
        state.tile_size = view.layout.tile_size();
        return Ok(state);
    }
    if opid == CuteTensorSliceOp::get_opid_static() {
        let mut state = resolve_tensor_view(ctx, defining_op.deref(ctx).get_operand(0), depth + 1)?;
        if state.tile_size.is_none() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.tensor_slice must follow exactly one zipped divide and may select only once",
            ));
        }
        state.tile_index = Some(defining_op.deref(ctx).get_operand(1));
        return Ok(state);
    }
    Err(invalid(format!(
        "tensor view is produced by unsupported operation `{opid}`"
    )))
}

/// Require direct view provenance and valid full-tile or scalar-tail consumers.
pub(super) fn verify_tensor_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = tensor_semantic_ids();
    let consumer_ids = &ids[3..];
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        let op = operation.deref(ctx);
        if opid == ids[0] || opid == ids[1] || opid == ids[2] {
            let result = op.get_result(0);
            let uses = result.uses(ctx);
            if uses.is_empty() {
                return Err(invalid(format!(
                    "tensor view from `{opid}` has no semantic consumer"
                )));
            }
            for r#use in uses {
                let user = r#use.user_op();
                let user_id = Operation::get_opid(user, ctx);
                let allowed = if opid == ids[0] {
                    user_id == ids[1] && r#use.find_index(ctx) == 0
                } else if opid == ids[1] {
                    user_id == ids[2] && r#use.find_index(ctx) == 0
                } else {
                    consumer_ids.contains(&user_id)
                };
                if !allowed {
                    return Err(invalid(format!(
                        "tensor view from `{opid}` has unsupported consumer `{user_id}`"
                    )));
                }
            }
        } else {
            let tensor = if opid == CuteTensorStoreFromOp::get_opid_static() {
                op.get_operand(1)
            } else {
                op.get_operand(0)
            };
            let state = resolve_tensor_view(ctx, tensor, 0)?;
            if state.tile_size.is_none() || state.tile_index.is_none() {
                return Err(invalid(format!(
                    "`{opid}` needs a complete make -> zipped_divide -> slice tensor story"
                )));
            }
        }
    }
    Ok(())
}
