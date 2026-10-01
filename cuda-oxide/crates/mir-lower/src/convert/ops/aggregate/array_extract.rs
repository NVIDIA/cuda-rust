/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::common::anyhow_to_pliron;
use crate::convert::types::{convert_type, mir_type_abi_align};
use dialect_mir::types::MirArrayType;
use llvm_export::attributes::{GepNoWrapFlags, ICmpPredicateAttr};
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Convert `mir.extract_array_element` to LLVM operations.
///
/// A runtime index normally requires materializing the array in memory because
/// LLVM `extractvalue` accepts only constant indices. When the index is proven
/// to be `urem value, C`, however, it is in `0..C`. For small `C` within the
/// array bounds, emit one constant `extractvalue` per candidate and select the
/// runtime result in SSA. This avoids the temporary alloca that otherwise
/// becomes NVPTX local memory.
///
/// Unbounded, oversized, or otherwise unsupported indices retain the existing
/// alloca+store+GEP+load fallback.
pub(crate) fn convert_extract_array_element(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    // One shared cap for the candidate chain, so the mir-transforms
    // canonicalization and this lowering fast path cannot drift apart.
    use dialect_mir::ops::MAX_SCALARIZED_CANDIDATES;

    fn integer_constant_u64(ctx: &Context, value: Value) -> Option<u64> {
        let defining_op = value.defining_op()?;
        let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, ctx)?;
        let attribute = constant.get_value(ctx);
        let integer = (&*attribute as &dyn pliron::attribute::Attribute)
            .downcast_ref::<pliron::builtin::attributes::IntegerAttr>()?;
        let integer_value = integer.value();
        // `APInt::to_u64` truncates wider values, so a >64-bit constant could
        // be misread as a small in-range divisor. Fail closed on such widths.
        (integer_value.bw() <= 64).then(|| integer_value.to_u64())
    }

    fn bounded_urem_candidate_count(ctx: &Context, index: Value, array_size: u64) -> Option<u64> {
        let defining_op = index.defining_op()?;
        Operation::get_op::<llvm::URemOp>(defining_op, ctx)?;

        let divisor = defining_op.deref(ctx).get_operand(1);
        let candidate_count = integer_constant_u64(ctx, divisor)?;
        (candidate_count > 0
            && candidate_count <= array_size
            && candidate_count <= MAX_SCALARIZED_CANDIDATES)
            .then_some(candidate_count)
    }

    fn integer_constant_like(
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        reference: Value,
        value: u64,
    ) -> Result<Value> {
        let reference_ty = reference.get_type(ctx);
        let width = reference_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!(
                    "mir.extract_array_element index must lower to an integer"
                )
            })?
            .width();

        let integer_ty = IntegerType::get(ctx, width, Signedness::Signless);
        let attribute = pliron::builtin::attributes::IntegerAttr::new(
            integer_ty,
            APInt::from_u64(
                value,
                NonZeroUsize::new(width as usize).expect("integer width is nonzero"),
            ),
        );
        let constant = llvm::ConstantOp::new(ctx, Box::new(attribute));
        rewriter.insert_operation(ctx, constant.get_operation());
        Ok(constant.get_operation().deref(ctx).get_result(0))
    }

    let array_val = op.deref(ctx).get_operand(0);
    let index_val = op.deref(ctx).get_operand(1);

    let (element_ty, array_size) = {
        match operands_info.lookup_most_recent_of_type::<MirArrayType>(ctx, array_val) {
            Some(r) => (r.element_type(), r.size()),
            None => return pliron::input_err_noloc!("Expected MirArrayType"),
        }
    };

    if let Some(candidate_count) = bounded_urem_candidate_count(ctx, index_val, array_size) {
        let mut candidates = Vec::with_capacity(candidate_count as usize);
        for candidate_index in 0..candidate_count {
            let extract = llvm::ExtractValueOp::new(ctx, array_val, vec![candidate_index as u32])?;
            rewriter.insert_operation(ctx, extract.get_operation());
            candidates.push(extract.get_operation().deref(ctx).get_result(0));
        }

        let mut selected = *candidates
            .last()
            .expect("candidate count is proven nonzero");
        for candidate_index in (0..candidates.len().saturating_sub(1)).rev() {
            let candidate_constant =
                integer_constant_like(ctx, rewriter, index_val, candidate_index as u64)?;
            let compare =
                llvm::ICmpOp::new(ctx, ICmpPredicateAttr::EQ, index_val, candidate_constant);
            rewriter.insert_operation(ctx, compare.get_operation());
            let condition = compare.get_operation().deref(ctx).get_result(0);

            let select = llvm::SelectOp::new(ctx, condition, candidates[candidate_index], selected);
            rewriter.insert_operation(ctx, select.get_operation());
            selected = select.get_operation().deref(ctx).get_result(0);
        }

        rewriter.replace_operation_with_values(ctx, op, vec![selected]);
        return Ok(());
    }

    let llvm_element_ty = convert_type(ctx, element_ty).map_err(anyhow_to_pliron)?;
    let llvm_array_ty = llvm_export::types::ArrayType::get(ctx, llvm_element_ty, array_size);
    let abi_align = mir_type_abi_align(ctx, element_ty);

    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let one_val = {
        let one_apint = APInt::from_i64(1, NonZeroUsize::new(64).unwrap());
        let one_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, one_apint);
        let const_op = llvm::ConstantOp::new(ctx, Box::new(one_attr));
        rewriter.insert_operation(ctx, const_op.get_operation());
        const_op.get_operation().deref(ctx).get_result(0)
    };

    let alloca_op = llvm::AllocaOp::new(ctx, llvm_array_ty.into(), one_val, 0);
    rewriter.insert_operation(ctx, alloca_op.get_operation());
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, alloca_op.get_operation(), align as u32);
    }
    let array_ptr = alloca_op.get_operation().deref(ctx).get_result(0);

    let store_op = llvm::StoreOp::new(ctx, array_val, array_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, store_op.get_operation(), align as u32);
    }

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Value(index_val)];
    let gep_op = llvm::GetElementPtrOp::new_with_no_wrap_flags(
        ctx,
        array_ptr,
        gep_indices,
        llvm_array_ty.into(),
        GepNoWrapFlags::INBOUNDS,
    );
    rewriter.insert_operation(ctx, gep_op.get_operation());
    let element_ptr = gep_op.get_operation().deref(ctx).get_result(0);

    let load_op = llvm::LoadOp::new(ctx, element_ptr, llvm_element_ty);
    rewriter.insert_operation(ctx, load_op.get_operation());
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, load_op.get_operation(), align as u32);
    }
    rewriter.replace_operation(ctx, op, load_op.get_operation());

    Ok(())
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;

    use dialect_mir::ops as mir;
    use dialect_mir::types::{MirArrayType, MirStructType, MirTupleType};

    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    fn over_aligned_tuple_ty(ctx: &mut Context) -> TypeHandle {
        let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let marker: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "Align32".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            32,
        )
        .into();
        MirTupleType::get_with_layout(ctx, vec![marker, byte], vec![0, 1], vec![0, 0], 32, 32)
            .into()
    }

    fn lower_array_extract_case(
        ctx: &mut Context,
        array_size: u64,
        divisor: Option<u64>,
    ) -> Ptr<Operation> {
        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let index_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let index_handle: TypeHandle = index_type.into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, array_size).into();

        let (module, block) = build_kernel(ctx, vec![index_handle], vec![element_type]);
        let raw_index = block.deref(ctx).get_argument(0);

        let undef = mir::MirUndefOp::new(ctx, array_type);
        undef.get_operation().insert_at_back(block, ctx);
        let array = undef.get_operation().deref(ctx).get_result(0);

        let index = if let Some(divisor) = divisor {
            let constant = Operation::new(
                ctx,
                mir::MirConstantOp::get_concrete_op_info(),
                vec![index_handle],
                vec![],
                vec![],
                0,
            );
            mir::MirConstantOp::new(constant).set_attr_value(
                ctx,
                IntegerAttr::new(
                    index_type,
                    APInt::from_u64(divisor, NonZeroUsize::new(64).unwrap()),
                ),
            );
            constant.insert_at_back(block, ctx);
            let divisor_value = constant.deref(ctx).get_result(0);

            let rem = Operation::new(
                ctx,
                mir::MirRemOp::get_concrete_op_info(),
                vec![index_handle],
                vec![raw_index, divisor_value],
                vec![],
                0,
            );
            rem.insert_at_back(block, ctx);
            rem.deref(ctx).get_result(0)
        } else {
            raw_index
        };

        let extract = Operation::new(
            ctx,
            mir::MirExtractArrayElementOp::get_concrete_op_info(),
            vec![element_type],
            vec![array, index],
            vec![],
            0,
        );
        extract.insert_at_back(block, ctx);
        let result = extract.deref(ctx).get_result(0);
        append_mir_return(ctx, block, vec![result]);

        crate::lower_mir_to_llvm(ctx, module).expect("lowering failed");
        module
    }

    fn assert_array_extract_memory_fallback(ctx: &Context, module: Ptr<Operation>) {
        let body = kernel_blocks(ctx, module);
        assert_eq!(count_ops::<llvm::AllocaOp>(ctx, &body), 1);
        assert_eq!(count_ops::<llvm::StoreOp>(ctx, &body), 1);
        assert_eq!(count_ops::<llvm::GetElementPtrOp>(ctx, &body), 1);
        assert_eq!(count_ops::<llvm::LoadOp>(ctx, &body), 1);
        assert_eq!(count_ops::<llvm::SelectOp>(ctx, &body), 0);
    }

    #[test]
    fn bounded_urem_array_extract_stays_in_ssa() {
        let mut ctx = make_ctx();
        let module = lower_array_extract_case(&mut ctx, 3, Some(3));
        let body = kernel_blocks(&ctx, module);

        assert_eq!(count_ops::<llvm::ExtractValueOp>(&ctx, &body), 3);
        assert_eq!(count_ops::<llvm::ICmpOp>(&ctx, &body), 2);
        assert_eq!(count_ops::<llvm::SelectOp>(&ctx, &body), 2);
        assert_eq!(count_ops::<llvm::AllocaOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::GetElementPtrOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 0);
    }

    #[test]
    fn unbounded_array_extract_keeps_memory_fallback() {
        let mut ctx = make_ctx();
        let module = lower_array_extract_case(&mut ctx, 3, None);
        assert_array_extract_memory_fallback(&ctx, module);
    }

    #[test]
    fn oversized_urem_array_extract_keeps_memory_fallback() {
        let mut ctx = make_ctx();
        let module = lower_array_extract_case(&mut ctx, 17, Some(17));
        assert_array_extract_memory_fallback(&ctx, module);
    }

    #[test]
    fn dynamic_array_extract_preserves_recursive_element_alignment() {
        let mut ctx = make_ctx();
        let tuple_ty = over_aligned_tuple_ty(&mut ctx);
        let inner: TypeHandle = MirArrayType::get(&mut ctx, tuple_ty, 2).into();
        let outer: TypeHandle = MirArrayType::get(&mut ctx, inner, 3).into();
        let index_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![index_ty], vec![]);
        let index = block.deref(&ctx).get_argument(0);

        let undef = mir::MirUndefOp::new(&mut ctx, outer);
        undef.get_operation().insert_at_back(block, &ctx);
        let array = undef.get_operation().deref(&ctx).get_result(0);
        let extract = Operation::new(
            &mut ctx,
            mir::MirExtractArrayElementOp::get_concrete_op_info(),
            vec![inner],
            vec![array, index],
            vec![],
            0,
        );
        extract.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
        let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected llvm.store");
        let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected llvm.load");
        for memory_op in [
            alloca.get_operation(),
            store.get_operation(),
            load.get_operation(),
        ] {
            assert_eq!(llvm_export::ops::op_alignment(&ctx, memory_op), Some(32));
        }
    }
}
