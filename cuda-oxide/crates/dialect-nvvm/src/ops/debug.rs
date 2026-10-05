/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Debug and Profiling Operations
//!
//! This module provides operations for debugging and profiling GPU kernels:
//!
//! ```text
//! ┌──────────────────────────┬──────────────────────────────┬─────────────────────────┐
//! │ Operation                │ PTX / LLVM Intrinsic         │ Description             │
//! ├──────────────────────────┼──────────────────────────────┼─────────────────────────┤
//! │ ReadPtxSregClockOp       │ %clock / read.ptx.sreg.clock │ 32-bit clock counter    │
//! │ ReadPtxSregClock64Op     │ %clock64 / ...clock64        │ 64-bit clock counter    │
//! │ ReadPtxSregGlobaltimerOp │ %globaltimer / ...globaltimer│ Global timer counter    │
//! │ AssertFailOp             │ call @__assertfail           │ Assertion diagnostics   │
//! │ VprintfOp                │ vprintf / call @vprintf      │ Formatted output        │
//! └──────────────────────────┴──────────────────────────────┴─────────────────────────┘
//! ```

use dialect_mir::types::MirPtrType;
use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

// =============================================================================
// Assertion Operations
// =============================================================================

/// CUDA device-side assertion failure operation.
///
/// Corresponds to CUDA's
/// `__assertfail(message, file, line, function, char_size)` system call.
///
/// # Operands
///
/// * `message` - Pointer to a null-terminated assertion message
/// * `file` - Pointer to a null-terminated source file name
/// * `line` - Source line number (`u32`)
/// * `function` - Pointer to null-terminated function or module context
/// * `char_size` - Size of one message character (`usize`, 64-bit on NVPTX64)
///
/// # Results
///
/// This operation has no results.
///
/// # Verification
///
/// - Must have five operands and no results
/// - Message, file, and function operands must be MIR pointers
/// - Line must be a 32-bit integer
/// - Character size must be a 64-bit integer
#[pliron_op(
    name = "nvvm.assertfail",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct AssertFailOp;

impl AssertFailOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        AssertFailOp { op }
    }

    /// Create a CUDA assertion failure operation.
    pub fn build(
        ctx: &mut Context,
        message: pliron::value::Value,
        file: pliron::value::Value,
        line: pliron::value::Value,
        function: pliron::value::Value,
        char_size: pliron::value::Value,
    ) -> Ptr<Operation> {
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![message, file, line, function, char_size],
            vec![],
            0,
        )
    }
}

impl Verify for AssertFailOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_operands() != 5 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.assertfail requires five operands and no results"
            );
        }

        for operand in [0, 1, 3] {
            let ty = op.get_operand(operand).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<MirPtrType>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.assertfail message, file, and function operands must be MIR pointers"
                );
            }
        }

        let line_ty = op.get_operand(2).get_type(ctx);
        let line_ty_obj = line_ty.deref(ctx);
        let line_ty = match line_ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.assertfail line operand must be a 32-bit integer"
                );
            }
        };
        if line_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.assertfail line operand must be a 32-bit integer"
            );
        }

        let char_size_ty = op.get_operand(4).get_type(ctx);
        let char_size_ty_obj = char_size_ty.deref(ctx);
        let char_size_ty = match char_size_ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.assertfail character-size operand must be a 64-bit integer"
                );
            }
        };
        if char_size_ty.width() != 64 {
            return verify_err!(
                op.loc(),
                "nvvm.assertfail character-size operand must be a 64-bit integer"
            );
        }

        Ok(())
    }
}

// =============================================================================
// Printf Operations
// =============================================================================

/// GPU vprintf operation for formatted output.
///
/// Corresponds to CUDA's device-side `vprintf(format, args)` function.
/// The GPU stores format pointer and arguments to a FIFO buffer,
/// which the host reads and formats during synchronization.
///
/// # Operands
///
/// * `format_ptr` - Pointer to null-terminated format string (i8*)
/// * `args_ptr` - Pointer to packed argument buffer (i8*)
///
/// # Results
///
/// * `i32` - Number of arguments on success, negative on error
///
/// # Verification
///
/// - Must have 2 operands (format_ptr, args_ptr)
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.vprintf",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct VprintfOp;

impl VprintfOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VprintfOp { op }
    }

    /// Create a new vprintf operation.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The context
    /// * `format_ptr` - Pointer to format string (i8*)
    /// * `args_ptr` - Pointer to packed arguments (i8*)
    ///
    /// # Returns
    ///
    /// Operation pointer with single i32 result (arg count on success)
    pub fn build(
        ctx: &mut Context,
        format_ptr: pliron::value::Value,
        args_ptr: pliron::value::Value,
    ) -> Ptr<Operation> {
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![i32_ty.to_handle()],   // Result: i32
            vec![format_ptr, args_ptr], // Operands: format_ptr, args_ptr
            vec![],
            0,
        )
    }
}

impl Verify for VprintfOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.vprintf requires two operands and one result"
            );
        }
        for operand in 0..2 {
            let ty = op.get_operand(operand).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<MirPtrType>().is_none() {
                return verify_err!(op.loc(), "nvvm.vprintf operands must be MIR pointers");
            }
        }

        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);

        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => return verify_err!(op.loc(), "nvvm.vprintf result must be integer"),
        };

        if int_ty.width() != 32 {
            return verify_err!(op.loc(), "nvvm.vprintf result must be 32-bit integer");
        }

        Ok(())
    }
}

/// Register debug operations with the context.
pub(super) fn register(ctx: &mut Context) {
    AssertFailOp::register(ctx);
    VprintfOp::register(ctx);
}
