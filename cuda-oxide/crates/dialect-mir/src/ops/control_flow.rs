/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR control flow operations.
//!
//! This module defines assertions, terminators, and branches for the MIR dialect.

use pliron::{
    builtin::{
        op_interfaces::{
            BranchOpInterface, IsTerminatorInterface, NResultsInterface, OperandSegmentInterface,
        },
        type_interfaces::FunctionTypeInterface,
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::op_interface_impl,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{Typed, type_cast},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

use super::function::MirFuncOp;
use crate::attributes::UnrollAttr;

// ============================================================================
// MirReturnOp
// ============================================================================

/// MIR return operation.
///
/// Terminator for returning from a function.
///
/// # Operands
///
/// Variadic operands matching the function return type.
///
/// # Verification
///
/// - Must have 0 successors.
/// - Must be inside a `MirFuncOp`.
/// - Operand types must match the function result types.
#[pliron_op(
    name = "mir.return",
    format,
    interfaces = [NResultsInterface<0>, IsTerminatorInterface]
)]
pub struct MirReturnOp;

impl MirReturnOp {
    /// Create a new MirReturnOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirReturnOp { op }
    }
}

impl Verify for MirReturnOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_successors() != 0 {
            return verify_err!(op.loc(), "MirReturnOp must have 0 successors");
        }

        let parent_op = match op.get_parent_op(ctx) {
            Some(p) => p,
            None => return verify_err!(op.loc(), "MirReturnOp must be within a function"),
        };

        let mir_func = match MirFuncOp::wrap(ctx, parent_op) {
            Some(f) => f,
            None => return verify_err!(op.loc(), "MirReturnOp must be within a MirFuncOp"),
        };

        let func_ty = mir_func.get_type(ctx);
        let func_ty_ref = func_ty.deref(ctx);
        let interface = type_cast::<dyn FunctionTypeInterface>(&*func_ty_ref);

        let result_types = if let Some(interface) = interface {
            interface.res_types()
        } else {
            return verify_err!(
                op.loc(),
                "FunctionType does not implement FunctionTypeInterface"
            );
        };

        if op.get_num_operands() != result_types.len() {
            return verify_err!(
                op.loc(),
                "MirReturnOp operand count must match function result count"
            );
        }

        for (i, res_ty) in result_types.iter().enumerate() {
            let opd = op.get_operand(i);
            if opd.get_type(ctx) != *res_ty {
                return verify_err!(
                    op.loc(),
                    "MirReturnOp operand type mismatch with function result"
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// MirGotoOp
// ============================================================================

/// MIR goto operation.
///
/// Unconditional branch to a successor block.
///
/// # Successors
///
/// Exactly 1 successor block.
///
/// # Verification
///
/// - Must have exactly 1 successor.
/// - Operand types must match successor block argument types.
#[pliron_op(
    name = "mir.goto",
    format,
    interfaces = [NResultsInterface<0>, IsTerminatorInterface]
)]
pub struct MirGotoOp;

impl MirGotoOp {
    /// Create a new MirGotoOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirGotoOp { op }
    }
}

impl Verify for MirGotoOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_successors() != 1 {
            return verify_err!(op.loc(), "MirGotoOp must have exactly 1 successor");
        }
        let succ = op.get_successor(0);
        let succ_block = succ.deref(ctx);

        if op.get_num_operands() != succ_block.get_num_arguments() {
            return verify_err!(
                op.loc(),
                "MirGotoOp operand count must match successor argument count"
            );
        }

        for i in 0..succ_block.get_num_arguments() {
            let arg = succ_block.get_argument(i);
            let opd = op.get_operand(i);
            if opd.get_type(ctx) != arg.get_type(ctx) {
                return verify_err!(
                    op.loc(),
                    "MirGotoOp operand type mismatch with successor argument"
                );
            }
        }

        Ok(())
    }
}

#[op_interface_impl]
impl BranchOpInterface for MirGotoOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(succ_idx == 0, "MirGotoOp has exactly one successor");
        self.get_operation().deref(ctx).operands().collect()
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        assert!(succ_idx == 0, "MirGotoOp has exactly one successor");
        Operation::push_operand(self.get_operation(), ctx, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        assert!(succ_idx == 0, "MirGotoOp has exactly one successor");
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }
}

// ============================================================================
// MirCondBranchOp
// ============================================================================

/// MIR conditional branch operation.
///
/// Branch based on boolean condition.
///
/// # Operands
///
/// - `cond`: Boolean (i1) condition.
/// - Variadic operands for true/false block arguments.
///
/// # Successors
///
/// Exactly 2 successors: true block, then false block.
///
/// # Verification
///
/// - Condition must be `i1`.
/// - Must have exactly 2 successors.
/// - Operand types must match successor block argument types.
#[pliron_op(
    name = "mir.cond_br",
    format,
    interfaces = [NResultsInterface<0>, IsTerminatorInterface, OperandSegmentInterface]
)]
pub struct MirCondBranchOp;

impl MirCondBranchOp {
    /// Create a new MirCondBranchOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCondBranchOp { op }
    }
}

impl Verify for MirCondBranchOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_successors() != 2 {
            return verify_err!(op.loc(), "MirCondBranchOp must have exactly 2 successors");
        }
        let true_block = op.get_successor(0).deref(ctx);
        let false_block = op.get_successor(1).deref(ctx);

        if op.get_num_operands() < 1 {
            return verify_err!(
                op.loc(),
                "MirCondBranchOp must have at least 1 operand (condition)"
            );
        }
        let cond = op.get_operand(0);
        let cond_ty = cond.get_type(ctx);
        let cond_ty_obj = cond_ty.deref(ctx);

        if let Some(int_ty) = cond_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirCondBranchOp condition must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirCondBranchOp condition must be integer type");
        }

        let expected_args = 1 + true_block.get_num_arguments() + false_block.get_num_arguments();
        if op.get_num_operands() != expected_args {
            return verify_err!(
                op.loc(),
                "MirCondBranchOp operand count must match condition + successors arguments"
            );
        }

        let mut op_idx = 1;
        for i in 0..true_block.get_num_arguments() {
            let arg = true_block.get_argument(i);
            let opd = op.get_operand(op_idx);
            if opd.get_type(ctx) != arg.get_type(ctx) {
                return verify_err!(
                    op.loc(),
                    "MirCondBranchOp true successor argument type mismatch"
                );
            }
            op_idx += 1;
        }

        for i in 0..false_block.get_num_arguments() {
            let arg = false_block.get_argument(i);
            let opd = op.get_operand(op_idx);
            if opd.get_type(ctx) != arg.get_type(ctx) {
                return verify_err!(
                    op.loc(),
                    "MirCondBranchOp false successor argument type mismatch"
                );
            }
            op_idx += 1;
        }

        Ok(())
    }
}

#[op_interface_impl]
impl BranchOpInterface for MirCondBranchOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(
            succ_idx == 0 || succ_idx == 1,
            "MirCondBranchOp has exactly two successors"
        );
        // Segment 0 = condition, segment 1 = true args, segment 2 = false args
        self.get_segment(ctx, succ_idx + 1)
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }
}

// ============================================================================
// MirAssertOp
// ============================================================================

/// MIR assert operation.
///
/// Continue when the condition is true; otherwise trap (`llvm.trap` -> PTX `trap`).
/// This is an effectful operation within a block, not a branch or terminator.
/// A Rust MIR assert terminator imports as this operation followed by `mir.goto`.
/// Keeping the check separate from the branch lets generic CFG simplification
/// merge blocks without erasing the check's observable failure behavior.
/// LLVM lowering splits the block here to make both outcomes explicit.
///
/// # Operands
///
/// - `cond`: Boolean (i1) condition.
///
/// # Verification
///
/// - Exactly one operand, whose type must be `i1`.
/// - No successors or results.
#[pliron_op(
    name = "mir.assert",
    format,
    interfaces = [NResultsInterface<0>]
)]
pub struct MirAssertOp;

impl MirAssertOp {
    /// Check `condition` and trap if it is false.
    pub fn new(ctx: &mut Context, condition: Value) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![condition],
                vec![],
                0,
            ),
        }
    }
}

impl Verify for MirAssertOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_successors() != 0 {
            return verify_err!(op.loc(), "MirAssertOp must have no successors");
        }

        if op.get_num_operands() != 1 {
            return verify_err!(
                op.loc(),
                "MirAssertOp must have exactly 1 operand (condition)"
            );
        }
        let cond = op.get_operand(0);
        let cond_ty = cond.get_type(ctx);
        let cond_ty_obj = cond_ty.deref(ctx);

        if let Some(int_ty) = cond_ty_obj.downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(op.loc(), "MirAssertOp condition must be i1");
            }
        } else {
            return verify_err!(op.loc(), "MirAssertOp condition must be integer type");
        }

        Ok(())
    }
}

// ============================================================================
// MirUnreachableOp
// ============================================================================

/// MIR unreachable operation.
///
/// Represents unreachable code - this code path should never be executed.
/// In CUDA, this will be lowered to LLVM's unreachable instruction which
/// triggers undefined behavior if reached.
///
/// # Verification
///
/// - Must have 0 operands.
/// - Must have 0 successors.
#[pliron_op(
    name = "mir.unreachable",
    format = "",
    interfaces = [
        pliron::builtin::op_interfaces::NOpdsInterface<0>,
        pliron::builtin::op_interfaces::NResultsInterface<0>,
        IsTerminatorInterface
    ]
)]
pub struct MirUnreachableOp;

impl MirUnreachableOp {
    /// Create a new MirUnreachableOp.
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        MirUnreachableOp { op }
    }
}

impl Verify for MirUnreachableOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_successors() != 0 {
            return verify_err!(op.loc(), "MirUnreachableOp must have 0 successors");
        }
        Ok(())
    }
}

/// A per-loop unroll request, planted inside a loop body by the `#[unroll]`
/// frontend and read by the loop-unroll pass.
///
/// The author writes `#[unroll]` / `#[unroll(N)]` on a loop; the `#[kernel]` or
/// `#[device]` macro turns that into one of these ops at the loop body's start. Because
/// the op physically sits inside the loop, the unroll pass can map it back to the
/// exact loop it belongs to (no fragile source-location matching). The pass reads
/// the factor, unrolls that loop, and deletes the hint, so it never reaches
/// lowering. It has no operands or results: it carries only the `factor`
/// attribute (`0` = full unroll, `n >= 2` = unroll by `n`).
#[pliron_op(
    name = "mir.unroll_hint",
    format,
    interfaces = [
        pliron::builtin::op_interfaces::NOpdsInterface<0>,
        pliron::builtin::op_interfaces::NResultsInterface<0>,
    ],
    attributes = (factor: UnrollAttr)
)]
pub struct MirUnrollHintOp;

impl MirUnrollHintOp {
    /// Create a hint requesting unroll `factor` (`0` = full).
    pub fn new(ctx: &mut Context, factor: u32) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let hint = MirUnrollHintOp { op };
        hint.set_attr_factor(ctx, UnrollAttr(factor));
        hint
    }

    /// The requested unroll factor (`0` = full unroll).
    pub fn factor(&self, ctx: &Context) -> u32 {
        self.get_attr_factor(ctx).map(|a| a.0).unwrap_or(0)
    }
}

impl Verify for MirUnrollHintOp {
    fn verify(&self, _ctx: &Context) -> pliron::result::Result<()> {
        Ok(())
    }
}

/// Register control flow operations into the given context.
pub fn register(ctx: &mut Context) {
    MirReturnOp::register(ctx);
    MirGotoOp::register(ctx);
    MirCondBranchOp::register(ctx);
    MirAssertOp::register(ctx);
    MirUnreachableOp::register(ctx);
    MirUnrollHintOp::register(ctx);
}
