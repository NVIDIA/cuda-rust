/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conversion of `mir.extern_shared` to dynamic shared-memory extern globals.

use super::common::anyhow_to_pliron;
use crate::context::{
    DynamicSmemAlignmentMap, SharedGlobalDeclaration, SharedGlobalKind, SharedGlobalRecord,
    SharedGlobalsMap,
};
use crate::helpers;
use llvm_export::ops as llvm;
use llvm_export::ops::GlobalOpExt;
use llvm_export::types::ArrayType;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `mir.extern_shared` to LLVM extern global variable in shared address space.
///
/// Dynamic (extern) shared memory is represented as an external global variable
/// with address space 3 and zero-length array type `[0 x i8]`. The actual size
/// is determined at kernel launch via `LaunchConfig::shared_mem_bytes`.
///
/// # Per-Owner Symbols
///
/// Each function that owns an access gets a dynamic shared-memory symbol
/// (`__dynamic_smem_{function_name}`).
///
/// # Alignment
///
/// The alignment is pre-computed during the lowering pre-pass. It is the
/// maximum of the owner's body requirements and every launch-contract marker
/// that can reach it.
///
/// # Byte Offset
///
/// - `DynamicSharedArray::get()` / `get_raw()`: offset = 0, returns base pointer
/// - `DynamicSharedArray::offset(N)`: offset = N bytes, returns base + N
///
/// Called directly from `MirToLlvmConversionDriver::rewrite` (not through
/// op_cast dispatch) because it needs cross-function state maps.
pub fn convert_extern_shared_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    shared_globals: &mut SharedGlobalsMap,
    dynamic_smem_alignments: &mut DynamicSmemAlignmentMap,
) -> Result<()> {
    let (byte_offset, alignment) = {
        let extern_shared_op = dialect_mir::ops::MirExternSharedOp::new(op);
        let byte_offset = extern_shared_op.get_byte_offset_value(ctx);
        let alignment = extern_shared_op.get_alignment_value(ctx);
        (byte_offset, alignment)
    };

    let func_name: String = {
        let parent_block = op
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
        let func_op_ptr = helpers::get_parent_func(ctx, parent_block).map_err(anyhow_to_pliron)?;
        let llvm_func = Operation::get_op::<llvm::FuncOp>(func_op_ptr, ctx)
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Parent op is not an llvm.func")))?;
        llvm_func.get_symbol_name(ctx).to_string()
    };

    let global_name = get_or_create_extern_shared_global(
        ctx,
        op,
        &func_name,
        shared_globals,
        dynamic_smem_alignments,
        alignment,
    )?;

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, 3);
    rewriter.insert_operation(ctx, address_of_op.get_operation());

    let base_ptr = address_of_op.get_operation().deref(ctx).get_result(0);

    if byte_offset > 0 {
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let offset_attr = pliron::builtin::attributes::IntegerAttr::new(
            i64_ty,
            pliron::utils::apint::APInt::from_u64(
                byte_offset,
                std::num::NonZeroUsize::new(64).unwrap(),
            ),
        );
        let offset_const = llvm::ConstantOp::new(ctx, Box::new(offset_attr));
        rewriter.insert_operation(ctx, offset_const.get_operation());
        let offset_value = offset_const.get_operation().deref(ctx).get_result(0);

        let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
        // Keep this GEP unflagged. The LLVM symbol is nominally `[0 x i8]`,
        // while the actual dynamic shared-memory allocation is supplied at
        // kernel launch, so this lowering cannot prove LLVM's `inbounds`
        // allocated-object contract for a nonzero byte offset.
        let gep_op = llvm::GetElementPtrOp::new(
            ctx,
            base_ptr,
            vec![llvm_export::ops::GepIndex::Value(offset_value)],
            i8_ty.into(),
        );
        rewriter.insert_operation(ctx, gep_op.get_operation());
        rewriter.replace_operation(ctx, op, gep_op.get_operation());
    } else {
        rewriter.replace_operation(ctx, op, address_of_op.get_operation());
    }

    Ok(())
}

/// Get or create the extern shared memory global for an owning function.
///
/// Creates an LLVM global variable with:
/// - Zero-length array type: `[0 x i8]`
/// - External linkage (no initializer)
/// - Address space 3 (shared memory)
/// - Pre-computed body and calling-kernel contract alignment
///
/// Each owning function gets its own dynamic shared memory symbol. Uses
/// `shared_globals` for deduplication (only one global per function).
fn get_or_create_extern_shared_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    func_name: &str,
    shared_globals: &mut SharedGlobalsMap,
    dynamic_smem_alignments: &mut DynamicSmemAlignmentMap,
    _requested_alignment: u64,
) -> Result<pliron::identifier::Identifier> {
    let (symbol_name, max_alignment) = dynamic_smem_alignments.get(func_name).cloned().ok_or_else(
        || {
            anyhow_to_pliron(anyhow::anyhow!(
                "Internal error: dynamic shared memory alignment not pre-computed for function '{}'. \
                 This should have been done in compute_max_dynamic_smem_alignment.",
                func_name
            ))
        },
    )?;

    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    let declaration = SharedGlobalDeclaration {
        kind: SharedGlobalKind::DynamicExtern,
        mir_elem_type: i8_ty.into(),
        size: 0,
        alignment: max_alignment,
        // The dynamic pool is per-launch storage with no originating static;
        // it never carries a source-level debug identity.
        debug_info: None,
        debug_owner_function: None,
    };
    let global_created_key = format!("__dynamic_smem_global_created_{}", func_name);
    if let Some(existing) = shared_globals.get(&global_created_key) {
        if existing.declaration != declaration || existing.symbol != symbol_name {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "dynamic shared-memory key {:?} has an incompatible declaration",
                global_created_key
            )));
        }
        return Ok(existing.symbol.clone());
    }

    let array_type = ArrayType::get(ctx, i8_ty.into(), 0);

    let global_op = llvm::GlobalOp::new_with_alignment(
        ctx,
        symbol_name.clone(),
        array_type.into(),
        max_alignment,
    );
    global_op.set_address_space(ctx, llvm_export::types::address_space::SHARED);

    {
        use llvm_export::attributes::LinkageAttr;
        global_op.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    }

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);

    shared_globals.insert(
        global_created_key,
        SharedGlobalRecord {
            symbol: symbol_name.clone(),
            declaration,
        },
    );

    Ok(symbol_name)
}
