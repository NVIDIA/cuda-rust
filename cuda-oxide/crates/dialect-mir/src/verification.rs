/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Whole-tree MIR verification helpers shared by every lowering entry point.

use crate::types::pointer_carriers_in_type;
use pliron::{
    builtin::ops::ConstantOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::Located,
    operation::Operation,
    result::Result,
    r#type::Typed,
    verify_err,
};

/// Reject generic builtin constants that claim to produce MIR pointer values.
///
/// `builtin.constant` deliberately accepts any result type and therefore
/// cannot establish a Rust pointer category. Every pointer-bearing constant
/// must instead use a MIR producer whose verifier records the boundary.
#[doc(hidden)]
pub fn verify_pointer_kind_producers(ctx: &Context, root: Ptr<Operation>) -> Result<()> {
    let mut to_visit = vec![root];
    while let Some(operation) = to_visit.pop() {
        let op = operation.deref(ctx);
        if Operation::get_op::<ConstantOp>(operation, ctx).is_some()
            && op.get_num_results() == 1
            && !pointer_carriers_in_type(ctx, op.get_result(0).get_type(ctx)).is_empty()
        {
            return verify_err!(
                op.loc(),
                "builtin.constant cannot produce a MIR pointer carrier; use a typed MIR address producer"
            );
        }
        for region in op.regions() {
            for block in region.deref(ctx).iter(ctx) {
                to_visit.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    Ok(())
}
