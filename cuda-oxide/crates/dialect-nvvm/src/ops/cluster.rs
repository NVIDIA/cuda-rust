/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compatibility operations for derived cluster-grid values.
//!
//! New intrinsic imports use generated per-axis special-register operations.
//! These types keep the former public dialect API and serialized operation
//! names available.

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

/// The cluster's linear index within the grid.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_cluster_idx",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClusterIdxOp;

impl ReadPtxSregClusterIdxOp {
    /// Wrap an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregClusterIdxOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_i32_result(ctx, self.get_operation(), "nvvm.read_ptx_sreg_cluster_idx")
    }
}

/// The total number of clusters in the grid.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_nclusterid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNclusterIdOp;

impl ReadPtxSregNclusterIdOp {
    /// Wrap an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for ReadPtxSregNclusterIdOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_i32_result(ctx, self.get_operation(), "nvvm.read_ptx_sreg_nclusterid")
    }
}

fn verify_i32_result(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let operation = op.deref(ctx);
    let result_type = operation.get_result(0).get_type(ctx);
    let result_type = result_type.deref(ctx);
    let Some(integer_type) = result_type.downcast_ref::<IntegerType>() else {
        return verify_err!(operation.loc(), "{name} result must be integer");
    };
    if integer_type.width() != 32 {
        return verify_err!(operation.loc(), "{name} result must be 32-bit integer");
    }
    Ok(())
}

pub(super) fn register(ctx: &mut Context) {
    ReadPtxSregClusterIdxOp::register(ctx);
    ReadPtxSregNclusterIdOp::register(ctx);
}
