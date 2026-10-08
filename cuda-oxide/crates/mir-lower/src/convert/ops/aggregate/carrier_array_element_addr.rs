/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Array-element address lowering for verified packed-AS3 carrier locals.

use super::addressing;
use crate::convert::types::{llvm_type_size_align, mir_element_stride, mir_type_abi_align};
use crate::packed_shared_local_storage::carrier_gep_source_type;
use dialect_mir::ops::MirConstantOp;
use dialect_mir::types::MirPtrType;
use llvm_export::ops as llvm;
use pliron::builtin::attributes::IntegerAttr;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// Lower an array element address using the exact physical element type stamped
/// by the pre-lowering carrier proof.
///
/// Ordinary array addressing remains on the established path. Carrier array
/// projections never infer storage from the converted pointer or its defining
/// operation; the typed GEP source is a private fact on this exact MIR op.
pub(crate) fn convert_array_element_addr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let Some(carrier_element_ty) = carrier_gep_source_type(ctx, op) else {
        return addressing::convert_array_element_addr(ctx, rewriter, op, operands_info);
    };

    let loc = op.deref(ctx).loc();
    let arr_ptr = op.deref(ctx).get_operand(0);
    let index = op.deref(ctx).get_operand(1);
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let semantic_element_ty = result_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|mir_ptr| mir_ptr.pointee)
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                "mir.array_element_addr result must be a MIR pointer type; carrier element sizing has no fact to derive from"
            )
        })?;

    // The root carrier proof guarantees byte-faithfulness. Recheck the exact
    // projected stride here so a malformed private fact still fails closed at
    // its consumer instead of silently changing pointer arithmetic.
    let rustc_stride = mir_element_stride(ctx, semantic_element_ty);
    let llvm_size = llvm_type_size_align(ctx, carrier_element_ty).map(|(size, _)| size);
    if let (Some(stride), Some(llvm_size)) = (rustc_stride, llvm_size)
        && stride != llvm_size
    {
        return pliron::input_err_noloc!(
            "packed-AS3 carrier array element stride mismatch: rustc strides by {} bytes but the physical carrier element occupies {}",
            stride,
            llvm_size
        );
    }

    use llvm_export::ops::GepIndex;
    let element_align =
        element_address_provable_alignment(ctx, arr_ptr, semantic_element_ty, index);
    let gep = llvm::GetElementPtrOp::new_with_no_wrap_flags(
        ctx,
        arr_ptr,
        vec![GepIndex::Value(index)],
        carrier_element_ty,
        llvm_export::attributes::GepNoWrapFlags::INBOUNDS,
    );
    rewriter.insert_operation(ctx, gep.get_operation());
    if let Some(align) = element_align {
        llvm_export::ops::set_address_alignment(ctx, gep.get_operation(), align);
    }
    rewriter.replace_operation(ctx, op, gep.get_operation());
    Ok(())
}

fn element_address_provable_alignment(
    ctx: &Context,
    arr_ptr: Value,
    element_ty: TypeHandle,
    index: Value,
) -> Option<u32> {
    const fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    let base_align = arr_ptr
        .defining_op()
        .and_then(|def| llvm_export::ops::address_alignment(ctx, def))
        .map(u64::from)
        .or_else(|| mir_type_abi_align(ctx, element_ty))?;
    if base_align == 0 {
        return None;
    }
    let stride = mir_element_stride(ctx, element_ty)?;
    if stride == 0 {
        return None;
    }

    let provable = match constant_index_value(ctx, index) {
        Some(0) => base_align,
        Some(i) => gcd(base_align, i.checked_mul(stride)?),
        None => gcd(base_align, stride),
    };
    if !provable.is_power_of_two() {
        return None;
    }
    u32::try_from(provable).ok()
}

fn constant_index_value(ctx: &Context, index: Value) -> Option<u64> {
    let defining_op = index.defining_op()?;
    if let Some(constant) = Operation::get_op::<MirConstantOp>(defining_op, ctx) {
        let value = constant.get_attr_value(ctx)?.value();
        return (value.bw() <= 64).then(|| value.to_u64());
    }
    let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, ctx)?;
    let attribute = constant.get_value(ctx);
    let integer =
        (&*attribute as &dyn pliron::attribute::Attribute).downcast_ref::<IntegerAttr>()?;
    let value = integer.value();
    (value.bw() <= 64).then(|| value.to_u64())
}
