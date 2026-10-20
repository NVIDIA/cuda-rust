/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-block-cluster queries for known targets below sm90.
//!
//! Intercept compatibility paths before generated dispatch erases their
//! identity. Raw `cuda_intrinsics` calls keep their native target requirements.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::{terminator::helpers, values::ValueMap};
use dialect_mir::ops::MirConstantOp;
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregCtaidZOp, ReadPtxSregNctaidXOp,
    ReadPtxSregNctaidYOp, ReadPtxSregNctaidZOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::utils::apint::APInt;
use rustc_public::mir;
use std::num::NonZeroUsize;

#[derive(Debug, PartialEq, Eq)]
enum Fallback {
    Constant(u32),
    BlockX,
    BlockY,
    BlockZ,
    GridX,
    GridY,
    GridZ,
}

fn fallback(name: &str) -> Option<Fallback> {
    Some(match name {
        "cuda_device::cluster::cluster_ctaidX"
        | "cuda_device::cluster::cluster_ctaidY"
        | "cuda_device::cluster::cluster_ctaidZ" => Fallback::Constant(0),
        "cuda_device::cluster::cluster_nctaidX"
        | "cuda_device::cluster::cluster_nctaidY"
        | "cuda_device::cluster::cluster_nctaidZ" => Fallback::Constant(1),
        "cuda_device::cluster::__cluster_idxX" => Fallback::BlockX,
        "cuda_device::cluster::__cluster_idxY" => Fallback::BlockY,
        "cuda_device::cluster::__cluster_idxZ" => Fallback::BlockZ,
        "cuda_device::cluster::__cluster_grid_dimX" => Fallback::GridX,
        "cuda_device::cluster::__cluster_grid_dimY" => Fallback::GridY,
        "cuda_device::cluster::__cluster_grid_dimZ" => Fallback::GridZ,
        _ => return None,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_dispatch(
    ctx: &mut Context,
    body: &mir::Body,
    name: &str,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    values: &mut ValueMap,
    blocks: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    if !values.pre90_cluster_queries {
        return Ok(None);
    }
    let Some(fallback) = fallback(name) else {
        return Ok(None);
    };
    if !args.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "cluster query `{name}` expects zero arguments, got {}",
                args.len()
            ))
        );
    }

    macro_rules! register {
        ($op:ty, $marker:literal) => {
            helpers::emit_generated_nvvm_intrinsic(
                ctx,
                body,
                <$op>::get_concrete_op_info(),
                $marker,
                destination,
                target,
                block,
                prev_op,
                values,
                blocks,
                loc,
            )?
        };
    }
    let operation = match fallback {
        Fallback::Constant(value) => {
            let (destination, prev_op) = helpers::prepare_destination_write(
                ctx,
                body,
                destination,
                values,
                block,
                prev_op,
                loc.clone(),
            )?;
            let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let operation = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![ty.into()],
                vec![],
                vec![],
                0,
            );
            let value = APInt::from_u64(u64::from(value), NonZeroUsize::new(32).unwrap());
            MirConstantOp::new(operation).set_attr_value(ctx, IntegerAttr::new(ty, value));
            operation.deref_mut(ctx).set_loc(loc.clone());
            helpers::insert_op(ctx, operation, block, prev_op);
            let result = operation.deref(ctx).get_result(0);
            helpers::emit_prepared_result_and_goto(
                ctx,
                destination,
                result,
                target,
                block,
                operation,
                values,
                blocks,
                loc,
                "cluster query has no successor",
            )?
        }
        Fallback::BlockX => register!(ReadPtxSregCtaidXOp, "v1:i0002"),
        Fallback::BlockY => register!(ReadPtxSregCtaidYOp, "v1:i0006"),
        Fallback::BlockZ => register!(ReadPtxSregCtaidZOp, "v1:i0010"),
        Fallback::GridX => register!(ReadPtxSregNctaidXOp, "v1:i0004"),
        Fallback::GridY => register!(ReadPtxSregNctaidYOp, "v1:i0008"),
        Fallback::GridZ => register!(ReadPtxSregNctaidZOp, "v1:i0012"),
    };
    values.used_cluster_query_fallback = true;
    Ok(Some(operation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_compatibility_queries_receive_fallbacks() {
        assert_eq!(
            fallback("cuda_device::cluster::cluster_ctaidX"),
            Some(Fallback::Constant(0))
        );
        assert_eq!(
            fallback("cuda_device::cluster::cluster_nctaidZ"),
            Some(Fallback::Constant(1))
        );
        assert_eq!(
            fallback("cuda_device::cluster::__cluster_idxY"),
            Some(Fallback::BlockY)
        );
        assert_eq!(
            fallback("cuda_device::cluster::__cluster_grid_dimZ"),
            Some(Fallback::GridZ)
        );
        for name in [
            "cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i0266",
            "cuda_intrinsics::sreg::cluster_dim_x",
            "cuda_device::cluster::cluster_sync",
            "cuda_device::cluster::map_shared_rank",
            "other::cluster::cluster_ctaidX",
        ] {
            assert_eq!(fallback(name), None, "{name}");
        }
    }
}
