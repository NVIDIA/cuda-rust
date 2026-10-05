/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conversions for `mir.store`, `mir.load`, `mir.alloca`, `mir.ref`, and `mir.ptr_offset`.

use super::common::{
    anyhow_to_pliron, copy_local_memory_provenance, fail_on_target_dependent_packed_aggregate,
    pointer_proved_alignment, value_abi_align, value_mir_type,
};
use super::debug::copy_debug_local_variable;
use crate::convert::target_stable_storage::coerce_target_stable_value;
use crate::convert::types::{convert_type, mir_type_abi_align};
use crate::packed_shared_local_storage::carrier_storage_type;
use dialect_mir::types::MirPtrType;
use llvm_export::attributes::GepNoWrapFlags;
use llvm_export::op_interfaces::VolatilityOpInterface;
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

/// Convert `mir.store` to `llvm.store`.
///
/// Operand order: `[ptr, value]` - stores `value` to address `ptr`.
/// No result is produced (store is a side effect).
pub(crate) fn convert_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, val) = match operands.as_slice() {
        [ptr, val] => (*ptr, *val),
        _ => {
            return pliron::input_err_noloc!("Store operation requires exactly 2 operands");
        }
    };

    let stored_val = if let Some(storage_ty) = carrier_storage_type(ctx, op) {
        // Carrier identity was proven on MIR before conversion. Convert exactly
        // at the memory boundary; never rediscover storage provenance from the
        // converted pointer or its defining operation.
        coerce_target_stable_value(
            ctx,
            rewriter,
            val,
            storage_ty,
            "packed shared carrier-local store",
        )?
    } else {
        // Packed whole-value stores are byte-faithful now that divergent rustc
        // layouts lower to LLVM packed structs. Keep the target-dependent AS3
        // case fail-closed for arbitrary memory; only pre-proven carrier-local
        // accesses are exempt.
        fail_on_target_dependent_packed_aggregate(
            ctx,
            value_mir_type(ctx, operands_info, val),
            "storing",
        )?;
        val
    };

    let llvm_store = llvm::StoreOp::new(ctx, stored_val, ptr);
    if dialect_mir::ops::MirStoreOp::new(op).is_volatile(ctx) {
        llvm_store.set_volatile(ctx, true);
    }
    // The stored value's own type answers first, as it did before. A scalar
    // records none, though, so fall back to whatever the address itself proved
    // when it was computed -- for a field projection that is the aggregate's
    // `abi_align` narrowed to the field's offset, which is otherwise lost here
    // and costs the pair its vectorization. This mirrors `convert_load`, which
    // consults the same record for the same reason. When both answer, the
    // weaker wins: a field of a packed aggregate can place an abi-aligned type
    // at a byte-aligned address, and the address's proved alignment is the
    // ceiling of what the store may claim.
    let abi = value_abi_align(ctx, operands_info, val);
    let proved = pointer_proved_alignment(ctx, ptr);
    let align = match (abi, proved) {
        (Some(abi), Some(proved)) => Some(abi.min(proved)),
        (abi, proved) => abi.or(proved),
    };
    if let Some(align) = align {
        llvm_export::ops::set_op_alignment(ctx, llvm_store.get_operation(), align as u32);
    }
    crate::convert::preserve_location(ctx, op, llvm_store.get_operation());
    rewriter.insert_operation(ctx, llvm_store.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.load` to `llvm.load`.
///
/// Takes a single pointer operand and returns the loaded value.
/// The result type is derived from the MIR operation's result type.
pub(crate) fn convert_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr = op.deref(ctx).get_operand(0);
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let semantic_llvm_ty = convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?;

    let storage_ty = if let Some(storage_ty) = carrier_storage_type(ctx, op) {
        storage_ty
    } else {
        // Packed whole-value loads are byte-faithful now that divergent rustc
        // layouts lower to LLVM packed structs. Keep only the target-dependent
        // AS3 physical-image case fail-closed for arbitrary memory.
        fail_on_target_dependent_packed_aggregate(ctx, result_ty, "loading")?;
        semantic_llvm_ty
    };

    let llvm_load = llvm::LoadOp::new(ctx, ptr, storage_ty);
    if dialect_mir::ops::MirLoadOp::new(op).is_volatile(ctx) {
        llvm_load.set_volatile(ctx, true);
    }
    // The loaded value's ABI alignment comes from this op's own result type,
    // which is still the MIR type: result types are only converted by the
    // op's own rewrite. A scalar records none, so fall back to whatever the
    // address itself proved when it was computed -- for a field projection
    // that is the aggregate's `abi_align` narrowed to the field's offset,
    // which is otherwise lost here and costs the pair its vectorization.
    // When both answer, the weaker wins: a field of a packed aggregate can
    // place an abi-aligned type at a byte-aligned address, and the address's
    // proved alignment is the ceiling of what the load may claim.
    let abi = mir_type_abi_align(ctx, result_ty);
    let proved = pointer_proved_alignment(ctx, ptr);
    let align = match (abi, proved) {
        (Some(abi), Some(proved)) => Some(abi.min(proved)),
        (abi, proved) => abi.or(proved),
    };
    if let Some(align) = align {
        llvm_export::ops::set_op_alignment(ctx, llvm_load.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, llvm_load.get_operation());

    if storage_ty == semantic_llvm_ty {
        rewriter.replace_operation(ctx, op, llvm_load.get_operation());
    } else {
        let physical_value = llvm_load.get_operation().deref(ctx).get_result(0);
        let semantic_value = coerce_target_stable_value(
            ctx,
            rewriter,
            physical_value,
            semantic_llvm_ty,
            "packed shared carrier-local load",
        )?;
        rewriter.replace_operation_with_values(ctx, op, vec![semantic_value]);
    }

    Ok(())
}

/// Convert `mir.alloca` to `llvm.alloca`.
///
/// `mir.alloca` carries its element type on the result pointer's pointee, and
/// emits a single-element stack slot of that type. We therefore convert the
/// pointee to an LLVM type and emit `llvm.alloca <pointee_ty>, i32 1`.
///
/// No value is stored into the slot; that is the caller's job via subsequent
/// `mir.store` / `llvm.store` ops. This matches the mem2reg-ready translator
/// model where every local is backed by one alloca in the entry block and
/// defs/uses go through `store`/`load` rather than SSA values.
pub(crate) fn convert_alloca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let mir_pointee = {
        let ty_ref = result_ty.deref(ctx);
        let mir_ptr = ty_ref.downcast_ref::<MirPtrType>().ok_or_else(|| {
            anyhow_to_pliron(anyhow::anyhow!(
                "MirAllocaOp result must be MirPtrType (enforced by verifier)"
            ))
        })?;
        mir_ptr.pointee
    };
    let llvm_pointee = match carrier_storage_type(ctx, op) {
        Some(storage_ty) => storage_ty,
        None => convert_type(ctx, mir_pointee).map_err(anyhow_to_pliron)?,
    };

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, Box::new(one_attr));
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, llvm_pointee, one_val, 0);
    // The allocated type's ABI alignment comes from this op's own result
    // pointee, which is still the MIR type at rewrite time.
    if let Some(align) = mir_type_abi_align(ctx, mir_pointee) {
        llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align as u32);
    }
    copy_debug_local_variable(ctx, op, alloca.get_operation());
    copy_local_memory_provenance(ctx, op, alloca.get_operation());
    rewriter.insert_operation(ctx, alloca.get_operation());
    rewriter.replace_operation(ctx, op, alloca.get_operation());

    Ok(())
}

/// Convert `mir.ref` — materialize the operand in stack memory via alloca+store.
///
/// `mir.ref` creates a pointer to an SSA value. In SSA form, values don't have
/// addresses, so we must place the value in memory to obtain a pointer.
/// This applies to all types: scalars (e.g. `&factor` where factor is `u32`),
/// aggregates (e.g. `&closure_env`), and pointers (e.g. `&&T`).
pub(crate) fn convert_ref(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operand = op.deref(ctx).get_operand(0);
    let operand_ty = operand.get_type(ctx);
    let abi_align = value_abi_align(ctx, operands_info, operand);

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, Box::new(one_attr));
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, operand_ty, one_val, 0);
    // Honour the referent's repr(align(N)) ABI alignment. Without this, the
    // synthesised alloca would be under-aligned relative to any loads/stores
    // that claim the struct's true alignment.
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, alloca.get_operation());
    let alloca_ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, operand, alloca_ptr);
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, store.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, store.get_operation());

    rewriter.replace_operation_with_values(ctx, op, vec![alloca_ptr]);

    Ok(())
}

/// Convert `mir.ptr_offset` to `llvm.getelementptr`.
///
/// Operands: `[ptr, offset]` where offset is an integer index.
/// Element sizing comes from the op's own result type, which is still the
/// MIR pointer type when this converter runs. The operand's recorded type
/// history is not usable here: a kind-only `mir.cast` lowers to a plain
/// value forwarding, and the history does not follow that replacement
/// edge, so a history miss would silently misscale the offset.
pub(crate) fn convert_ptr_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, offset) = match operands.as_slice() {
        [ptr, offset] => (*ptr, *offset),
        _ => return pliron::input_err!(loc, "PtrOffset requires exactly 2 operands"),
    };

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let pointee = result_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|mir_ptr| mir_ptr.pointee)
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                "mir.ptr_offset result must be a MIR pointer type; \
                 element sizing has no fact to derive from"
            )
        })?;
    let elem_ty = convert_type(ctx, pointee).map_err(anyhow_to_pliron)?;

    let no_wrap_flags = if dialect_mir::ops::MirPtrOffsetOp::new(op).is_inbounds(ctx) {
        GepNoWrapFlags::INBOUNDS
    } else {
        GepNoWrapFlags::empty()
    };
    let llvm_gep = llvm::GetElementPtrOp::new_with_no_wrap_flags(
        ctx,
        ptr,
        vec![llvm_export::ops::GepIndex::Value(offset)],
        elem_ty,
        no_wrap_flags,
    );
    rewriter.insert_operation(ctx, llvm_gep.get_operation());
    rewriter.replace_operation(ctx, op, llvm_gep.get_operation());

    Ok(())
}
