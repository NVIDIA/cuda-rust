/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 tensor-memory, collector-aware tiled MMA, and cluster pipelines.
//!
//! These operations keep ordinary MIR scalar carriers and retain the static
//! layout and collective contracts as typed attributes. A paired MMA is one
//! semantic operation: its A collector is filled and last-used within each
//! K=16 step, so a backend cannot accidentally carry it across K iterations.
//!
//! Cluster TMA loads are collective across all 32 producer-warp lanes. Native
//! CuTe lowering owns lane election and warp synchronization; callers must
//! not place a load inside an elected-lane branch. TMA stores retain a single
//! issuer, and accumulator release fences all lanes before electing an arrival.
//!
//! All operations have zero SSA results. Their docs give the operand order for
//! `new`; effects update shared memory, tensor memory, or hardware barriers.
//! TMEM addresses are encoded `u32` addresses, not byte pointers. Ring stages
//! are zero-based indices, and phases are parity bits maintained by the caller.

use dialect_mir::ops::MirConstantOp;
use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::attribute::Attribute;
use pliron::builtin::types::{FP32Type, IntegerType};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::op::{Op, OpId};
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::Typed;
use pliron::value::Value;
use pliron::{verify_err, verify_err_noloc};
use pliron_derive::{pliron_attr, pliron_op};

use crate::attributes::{CuteComposedLayoutAttr, CuteTmaStorePipelineAttr};
use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};

/// Build a zero-offset FP16 row-major layout with the requested hardware swizzle.
fn row_major(rows: u32, columns: u32, swizzle_bits: u32) -> ComposedLayout {
    let inner: Layout = format!("({rows},{columns}):({columns},1)")
        .parse()
        .expect("fixed rank-two layout");
    ComposedLayout::new(
        Swizzle::new(swizzle_bits, 3, 3),
        0,
        inner,
        OffsetUnit::Elements,
    )
    .expect("fixed SM100 FP16 swizzle")
}

mod operations;
mod plans;
mod verify;

pub use operations::{
    CuteSm100AccumulatorAcquireOp, CuteSm100AccumulatorCommitOp, CuteSm100AccumulatorInitOp,
    CuteSm100AccumulatorReleaseOp, CuteSm100AccumulatorWaitOp, CuteSm100ClusterTmaLoadOp,
    CuteSm100PipelineAcquireOp, CuteSm100PipelineExpectOp, CuteSm100PipelineInitOp,
    CuteSm100PipelineReleaseOp, CuteSm100PipelineWaitOp, CuteSm100StoreAcquireOp,
    CuteSm100StoreCommitOp, CuteSm100StoreTailOp, CuteSm100TiledMmaOp, CuteSm100TmaStoreOp,
    CuteSm100TmemAllocOp, CuteSm100TmemDeallocOp, CuteSm100TmemEpilogueOp, register, semantic_ids,
    verify_local,
};
pub use plans::{
    CuteSm100ClusterCopyPlanAttr, CuteSm100CollectorAttr, CuteSm100EpiloguePlanAttr,
    CuteSm100MmaPlanAttr, CuteSm100PipelinePlanAttr, CuteSm100TmemPlanAttr,
};
pub use verify::verify_story;

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
