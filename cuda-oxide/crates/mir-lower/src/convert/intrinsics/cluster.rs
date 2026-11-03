/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compatibility lowering for derived cluster-grid values.

use super::common::inline_asm_convergent;
use pliron::{
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo},
    irbuild::rewriter::Rewriter,
    operation::Operation,
    result::Result,
};

const CLUSTER_IDX_ASM: &str = concat!(
    "{ .reg .u32 %cx, %cy, %cz, %nx, %ny, %nxy, %xy; ",
    "mov.u32 %cx, %clusterid.x; mov.u32 %cy, %clusterid.y; ",
    "mov.u32 %cz, %clusterid.z; mov.u32 %nx, %nclusterid.x; ",
    "mov.u32 %ny, %nclusterid.y; mul.lo.u32 %nxy, %nx, %ny; ",
    "mad.lo.u32 %xy, %cy, %nx, %cx; ",
    "mad.lo.u32 $0, %cz, %nxy, %xy; }"
);

const NUM_CLUSTERS_ASM: &str = concat!(
    "{ .reg .u32 %nx, %ny, %nz, %nxy; ",
    "mov.u32 %nx, %nclusterid.x; mov.u32 %ny, %nclusterid.y; ",
    "mov.u32 %nz, %nclusterid.z; mul.lo.u32 %nxy, %nx, %ny; ",
    "mul.lo.u32 $0, %nxy, %nz; }"
);

pub(crate) fn convert_cluster_idx(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_type = IntegerType::get(ctx, 32, Signedness::Signless);
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        i32_type.into(),
        vec![],
        CLUSTER_IDX_ASM,
        "=r",
    );
    rewriter.replace_operation(ctx, op, inline_asm);
    Ok(())
}

pub(crate) fn convert_num_clusters(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_type = IntegerType::get(ctx, 32, Signedness::Signless);
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        i32_type.into(),
        vec![],
        NUM_CLUSTERS_ASM,
        "=r",
    );
    rewriter.replace_operation(ctx, op, inline_asm);
    Ok(())
}
