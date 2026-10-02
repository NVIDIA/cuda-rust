/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct field-address lowering for verified packed-AS3 carrier locals.

use super::addressing;
use super::common::anyhow_to_pliron;
use crate::convert::types::{StructLayoutInfo, build_struct_slot_map};
use crate::packed_shared_local_storage::carrier_gep_source_type;
use dialect_mir::ops::MirFieldAddrOp;
use dialect_mir::types::MirStructType;
use llvm_export::ops as llvm;
use llvm_export::types::{StructLayout, StructType};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};

/// Lower a field address, using the physical carrier struct only when the
/// pre-lowering carrier proof stamped this exact projection.
///
/// Ordinary field projections stay on the established #859 path. Carrier
/// projections never reconstruct provenance from operand history: the physical
/// GEP source type is an LLVM `TypeAttr` placed directly on this MIR operation
/// by the closed-world preparation pass.
pub(crate) fn convert_field_addr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let Some(carrier_source_ty) = carrier_gep_source_type(ctx, op) else {
        return addressing::convert_field_addr(ctx, rewriter, op, operands_info);
    };

    let field_addr = MirFieldAddrOp::new(op);
    let field_index = field_addr
        .get_attr_field_index(ctx)
        .ok_or_else(|| pliron::input_error_noloc!("MirFieldAddrOp missing field_index attribute"))?
        .0 as usize;
    let semantic_aggregate = field_addr
        .get_attr_aggregate_ty(ctx)
        .ok_or_else(|| {
            pliron::input_error_noloc!("MirFieldAddrOp missing verified aggregate_ty attribute")
        })?
        .get_type(ctx);

    let (layout, aggregate_abi_align) = {
        let aggregate_ref = semantic_aggregate.deref(ctx);
        let Some(struct_ty) = aggregate_ref.downcast_ref::<MirStructType>() else {
            return pliron::input_err_noloc!(
                "packed-AS3 carrier field projection requires a struct root"
            );
        };
        (StructLayoutInfo::of_struct(struct_ty), struct_ty.abi_align)
    };
    let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;

    let carrier_is_packed = {
        let carrier_ref = carrier_source_ty.deref(ctx);
        carrier_ref
            .downcast_ref::<StructType>()
            .is_some_and(|ty| ty.layout() == StructLayout::Packed)
    };
    if !carrier_is_packed {
        return pliron::input_err_noloc!(
            "packed-AS3 carrier field projection was stamped with a non-packed physical source type"
        );
    }

    let slot = match map.decl_to_llvm.get(field_index) {
        Some(Some(slot)) => *slot,
        Some(None) => {
            // Match the ordinary field-address contract for stripped ZSTs: a
            // distinct zero-offset byte GEP keeps value identity unambiguous.
            use llvm_export::ops::GepIndex;
            let ptr = op.deref(ctx).get_operand(0);
            let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
            let gep = llvm::GetElementPtrOp::new(ctx, ptr, vec![GepIndex::Constant(0)], i8_ty);
            rewriter.insert_operation(ctx, gep.get_operation());
            rewriter.replace_operation(ctx, op, gep.get_operation());
            return Ok(());
        }
        None => {
            return pliron::input_err_noloc!(
                "packed-AS3 carrier field index {} out of bounds for struct with {} fields",
                field_index,
                map.decl_to_llvm.len()
            );
        }
    };

    // The carrier itself is the byte-faithful physical representation. Unlike
    // the ordinary semantic path, do not enter #859's natural-layout byte-GEP
    // fallback: `[0, slot]` over this packed carrier is already the exact
    // storage address and also selects the physical p0 field type.
    use llvm_export::ops::GepIndex;
    let ptr = op.deref(ctx).get_operand(0);
    let gep = llvm::GetElementPtrOp::new(
        ctx,
        ptr,
        vec![GepIndex::Constant(0), GepIndex::Constant(slot)],
        carrier_source_ty,
    );
    rewriter.insert_operation(ctx, gep.get_operation());
    stamp_field_address_alignment(
        ctx,
        gep.get_operation(),
        aggregate_abi_align,
        layout.field_offsets.get(field_index).copied(),
    );
    rewriter.replace_operation(ctx, op, gep.get_operation());
    Ok(())
}

fn stamp_field_address_alignment(
    ctx: &mut Context,
    gep: Ptr<Operation>,
    abi_align: u64,
    field_offset: Option<u64>,
) {
    const fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    if abi_align == 0 {
        return;
    }
    let Some(offset) = field_offset else {
        return;
    };
    let provable = if offset == 0 {
        abi_align
    } else {
        gcd(abi_align, offset)
    };
    if !provable.is_power_of_two() {
        return;
    }
    if let Ok(align) = u32::try_from(provable) {
        llvm_export::ops::set_address_alignment(ctx, gep, align);
    }
}
