/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helper functions for `dialect-mir` → LLVM dialect lowering.
//!
//! This module provides utility functions that are shared across multiple
//! operation converters. These helpers handle common tasks like:
//!
//! - Creating LLVM constant values (integers of various widths)
//! - Declaring LLVM intrinsic functions in the module
//! - Navigating the IR hierarchy (finding parent functions and modules)
//!
//! # Organization
//!
//! The helpers are organized by functionality:
//!
//! ## Constants
//!
//! | Function                | Type  | Description                    |
//! |-------------------------|-------|--------------------------------|
//! | [`create_i1_constant`]  | `i1`  | Boolean constants (true/false) |
//! | [`create_i32_constant`] | `i32` | 32-bit integer constants       |
//! | [`create_i64_constant`] | `i64` | 64-bit integer constants       |
//!
//! ## Intrinsic Declaration
//!
//! | Function                          | Description                              |
//! |-----------------------------------|------------------------------------------|
//! | [`ensure_intrinsic_declared`]     | Declare intrinsic in module              |
//!
//! ## IR Navigation
//!
//! | Function                  | Description                          |
//! |---------------------------|--------------------------------------|
//! | [`get_parent_func`]       | Get the function containing a block  |
//! | [`get_module_from_block`] | Get the module containing a block    |
//!
//! # Usage Pattern
//!
//! These helpers are typically called from operation converters:
//!
//! ```ignore
//! // In an intrinsic converter:
//! fn convert_thread_id(ctx: &mut Context, block: Ptr<BasicBlock>) -> Result<()> {
//!     let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
//!     let func_ty = FuncType::get(ctx, i32_ty.into(), vec![], false);
//!     ensure_intrinsic_declared(ctx, block, "llvm_nvvm_read_ptx_sreg_tid_x", func_ty)?;
//!     // ...
//! }
//! ```

use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::value::Value;

// ============================================================================
// Constant Creation
// ============================================================================

/// Create an i1 (boolean) constant value.
///
/// Creates an LLVM `ConstantOp` producing an `i1` value, which is the LLVM
/// representation of a boolean. The operation is inserted at the back of
/// the specified basic block.
///
/// # Arguments
///
/// * `ctx` - The pliron context for IR manipulation
/// * `llvm_block` - The basic block to insert the constant into
/// * `value` - The boolean value (`true` or `false`)
///
/// # Returns
///
/// `Ok(Value)` containing the SSA value produced by the constant operation,
/// or an error if constant creation fails.
///
/// # Generated IR
///
/// ```llvm
/// %0 = llvm.mlir.constant(1 : i1) : i1   ; for value = true
/// %0 = llvm.mlir.constant(0 : i1) : i1   ; for value = false
/// ```
///
/// # Example
///
/// ```ignore
/// // Create a boolean constant for a predicate
/// let predicate = create_i1_constant(ctx, llvm_block, true)?;
/// // predicate can now be used as an operand in branch or select operations
/// ```
pub fn create_i1_constant(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    value: bool,
) -> Result<Value, anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    // Create the i1 type
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    // Convert boolean to integer (1 for true, 0 for false)
    let const_value = if value { 1i64 } else { 0i64 };

    // Create the arbitrary-precision integer with 1-bit width
    let width = NonZeroUsize::new(1).expect("1 is non-zero");
    let apint = APInt::from_i64(const_value, width);

    // Create the integer attribute
    let int_attr = IntegerAttr::new(i1_ty, apint);

    // Create and insert the LLVM constant operation
    let const_op = llvm::ConstantOp::new(ctx, Box::new(int_attr));
    const_op.get_operation().insert_at_back(llvm_block, ctx);

    // Return the result value
    Ok(const_op.get_operation().deref(ctx).get_result(0))
}

/// Create an i32 (32-bit integer) constant value.
///
/// Creates an LLVM `ConstantOp` producing an `i32` value. The operation is
/// inserted at the back of the specified basic block.
///
/// # Arguments
///
/// * `ctx` - The pliron context for IR manipulation
/// * `llvm_block` - The basic block to insert the constant into
/// * `value` - The 32-bit integer value
///
/// # Returns
///
/// `Ok(Value)` containing the SSA value produced by the constant operation,
/// or an error if constant creation fails.
///
/// # Generated IR
///
/// ```llvm
/// %0 = llvm.mlir.constant(42 : i32) : i32
/// ```
///
/// # Example
///
/// ```ignore
/// // Create constant for array index
/// let index = create_i32_constant(ctx, llvm_block, 0)?;
///
/// // Create constant for warp mask (all lanes)
/// let full_mask = create_i32_constant(ctx, llvm_block, -1)?; // 0xFFFFFFFF
/// ```
#[allow(dead_code)]
pub fn create_i32_constant(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    value: i32,
) -> Result<Value, anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    // Create the i32 type
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    // Create the arbitrary-precision integer with 32-bit width
    let width = NonZeroUsize::new(32).expect("32 is non-zero");
    let apint = APInt::from_i64(i64::from(value), width);

    // Create the integer attribute
    let int_attr = IntegerAttr::new(i32_ty, apint);

    // Create and insert the LLVM constant operation
    let const_op = llvm::ConstantOp::new(ctx, Box::new(int_attr));
    const_op.get_operation().insert_at_back(llvm_block, ctx);

    // Return the result value
    Ok(const_op.get_operation().deref(ctx).get_result(0))
}

/// Create an i64 (64-bit integer) constant value.
///
/// Creates an LLVM `ConstantOp` producing an `i64` value. The operation is
/// inserted at the back of the specified basic block.
///
/// # Arguments
///
/// * `ctx` - The pliron context for IR manipulation
/// * `llvm_block` - The basic block to insert the constant into
/// * `value` - The 64-bit integer value
///
/// # Returns
///
/// `Ok(Value)` containing the SSA value produced by the constant operation,
/// or an error if constant creation fails.
///
/// # Generated IR
///
/// ```llvm
/// %0 = llvm.mlir.constant(1234567890123 : i64) : i64
/// ```
///
/// # Example
///
/// ```ignore
/// // Create constant for pointer arithmetic
/// let offset = create_i64_constant(ctx, llvm_block, 1024)?;
///
/// // Create constant for size calculation
/// let element_size = create_i64_constant(ctx, llvm_block, 4)?;
/// ```
pub fn create_i64_constant(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    value: i64,
) -> Result<Value, anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    // Create the i64 type
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    // Create the arbitrary-precision integer with 64-bit width
    let width = NonZeroUsize::new(64).expect("64 is non-zero");
    let apint = APInt::from_i64(value, width);

    // Create the integer attribute
    let int_attr = IntegerAttr::new(i64_ty, apint);

    // Create and insert the LLVM constant operation
    let const_op = llvm::ConstantOp::new(ctx, Box::new(int_attr));
    const_op.get_operation().insert_at_back(llvm_block, ctx);

    // Return the result value
    Ok(const_op.get_operation().deref(ctx).get_result(0))
}

// ============================================================================
// Intrinsic Declaration
// ============================================================================

/// Ensure an intrinsic function is declared in the module.
///
/// LLVM intrinsics must be declared as function symbols in the module before
/// they can be called. This function checks if the intrinsic already exists
/// and creates a declaration if it doesn't.
///
/// # Arguments
///
/// * `ctx` - The pliron context for IR manipulation
/// * `llvm_block` - A block in the function where we need the intrinsic
///   (used to navigate to the parent module)
/// * `intrinsic_name` - The name of the intrinsic (e.g., `"llvm_nvvm_read_ptx_sreg_tid_x"`)
/// * `func_ty` - The function type of the intrinsic
///
/// # Returns
///
/// Returns the existing or newly created LLVM function declaration,
/// or an error if the IR structure is invalid.
///
/// # IR Navigation
///
/// The function navigates the IR hierarchy as follows:
///
/// ```text
/// llvm_block → parent_func → module → module_block
///                                          ↓
///                                   insert func declaration here
/// ```
///
/// # Example
///
/// ```ignore
/// // Declare the thread ID intrinsic
/// let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
/// let func_ty = FuncType::get(ctx, i32_ty.into(), vec![], false);
/// ensure_intrinsic_declared(ctx, llvm_block, "llvm_nvvm_read_ptx_sreg_tid_x", func_ty)?;
///
/// // Now we can call the intrinsic
/// let call_op = llvm::CallOp::new(ctx, "llvm_nvvm_read_ptx_sreg_tid_x", vec![], ...);
/// ```
///
/// # Note on Intrinsic Naming
///
/// LLVM NVVM intrinsics use underscores in their names when represented in the
/// Pliron IR symbol system (e.g., `llvm_nvvm_read_ptx_sreg_tid_x`). These are
/// translated to the standard LLVM intrinsic names with dots when exported to
/// LLVM IR (e.g., `llvm.nvvm.read.ptx.sreg.tid.x`).
/// Escape-mode identifiers preserve literal underscores with `_u`.
pub fn ensure_intrinsic_declared(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    intrinsic_name: &str,
    func_ty: pliron::r#type::TypedHandle<llvm_export::types::FuncType>,
) -> Result<Ptr<pliron::operation::Operation>, anyhow::Error> {
    // Navigate from block to parent function
    let func_op = llvm_block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Block has no parent operation (expected function)"))?;

    // Navigate from function to parent module
    let module_op = func_op
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Function has no parent operation (expected module)"))?;

    // Get the module's single region and its entry block
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow::anyhow!("Module region is empty (no entry block)"))?;

    // Convert intrinsic name to identifier
    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .try_into()
        .map_err(|e| anyhow::anyhow!("Invalid intrinsic name '{}': {:?}", intrinsic_name, e))?;

    // Check if the intrinsic is already declared
    let mut existing_declaration = None;
    for existing_op in module_block.deref(ctx).iter(ctx) {
        if let Some(existing_func) =
            pliron::operation::Operation::get_op::<llvm::FuncOp>(existing_op, ctx)
            && existing_func.get_symbol_name(ctx) == sym_name
        {
            existing_declaration = Some(existing_op);
            break;
        }
    }

    // If not declared, create a function declaration (no body)
    if let Some(existing_declaration) = existing_declaration {
        return Ok(existing_declaration);
    }

    let func_decl = llvm::FuncOp::new(ctx, sym_name, func_ty);
    let func_decl_op = func_decl.get_operation();
    func_decl_op.insert_before(ctx, func_op);

    Ok(func_decl_op)
}

// ============================================================================
// IR Navigation
// ============================================================================

/// Get the parent function operation from a basic block.
///
/// In the pliron IR hierarchy, a basic block is contained in a region,
/// which is contained in an operation (typically a function).
///
/// # Arguments
///
/// * `ctx` - The pliron context
/// * `llvm_block` - The basic block to find the parent of
///
/// # Returns
///
/// `Ok(Ptr<Operation>)` pointing to the parent function operation,
/// or an error if the block has no parent.
///
/// # IR Hierarchy
///
/// ```text
/// ModuleOp
/// └── Region
///     └── Block
///         └── FuncOp          ← returned by this function
///             └── Region
///                 ├── Block   ← llvm_block parameter
///                 ├── Block
///                 └── ...
/// ```
pub fn get_parent_func(
    ctx: &Context,
    llvm_block: Ptr<BasicBlock>,
) -> Result<Ptr<pliron::operation::Operation>, anyhow::Error> {
    llvm_block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Block has no parent operation"))
}

/// Get the module operation from a basic block.
///
/// Navigates two levels up in the IR hierarchy: from block to function,
/// then from function to module.
///
/// # Arguments
///
/// * `ctx` - The pliron context
/// * `llvm_block` - A basic block within a function within the module
///
/// # Returns
///
/// `Ok(Ptr<Operation>)` pointing to the module operation,
/// or an error if the hierarchy is incomplete.
///
/// # IR Hierarchy
///
/// ```text
/// ModuleOp                    ← returned by this function
/// └── Region
///     └── Block
///         ├── FuncOp
///         │   └── Region
///         │       └── Block   ← llvm_block parameter
///         └── FuncOp
///             └── ...
/// ```
///
/// # Example
///
/// ```ignore
/// // Find the module to add a global declaration
/// let module_op = get_module_from_block(ctx, conv_ctx.llvm_block)?;
/// let module_region = module_op.deref(ctx).get_region(0);
/// let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
/// // Now we can insert global declarations in module_block
/// ```
pub fn get_module_from_block(
    ctx: &Context,
    llvm_block: Ptr<BasicBlock>,
) -> Result<Ptr<pliron::operation::Operation>, anyhow::Error> {
    let func_op = get_parent_func(ctx, llvm_block)?;
    func_op
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| anyhow::anyhow!("Function has no parent operation (expected module)"))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Note: Testing these helpers requires a full pliron context with
    // dialects registered, which is complex to set up. Integration tests
    // in the `tests/` directory cover these functions indirectly.
}
