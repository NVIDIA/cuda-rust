/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory address-space conversion intrinsics.
//!
//! `nvvm.cvta_generic_to_shared_offset` lowers without inline PTX: an
//! `addrspacecast` into `addrspace(3)` followed by `ptrtoint`, which `llc`
//! selects as `cvta.to.shared`. The `ptrtoint` here deliberately reads the
//! space-local shared offset; it is a hardware descriptor/instruction-address
//! boundary, not a Rust pointer-address observation (those genericize first,
//! see `convert/ops/cast.rs`).
//!
//! `nvvm.tcgen05_smem_descriptor` lowers to the plain integer arithmetic of the
//! descriptor encoding, so LLVM folds constant fields as it did when the
//! encoding was written out in Rust.

use crate::convert::intrinsics::common::{
    cast_to_cluster_shared_addrspace, cast_to_shared_addrspace, create_i64_const,
};
use llvm_export::attributes::IntegerOverflowFlagsAttr;
use llvm_export::op_interfaces::{BinArithOp, CastOpInterface, CastOpWithNNegInterface};
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

/// Convert `nvvm.cvta_generic_to_shared_offset` to `addrspacecast` + `ptrtoint`.
pub(crate) fn convert_cvta_generic_to_shared_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("cvta_generic_to_shared_offset requires an operand");
    }
    let space = operands[0]
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|ty| ty.address_space());
    let shared_ptr = match space {
        Some(3 | 7) => operands[0], // AS7 retains the remote CTA rank.
        Some(0) => cast_to_shared_addrspace(ctx, rewriter, operands[0]),
        _ => {
            return pliron::input_err_noloc!(
                "shared offset requires generic, shared or cluster shared pointer"
            );
        }
    };

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let Some(result_width) = result_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(IntegerType::width)
    else {
        return pliron::input_err_noloc!(
            "cvta_generic_to_shared_offset requires an integer result"
        );
    };
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    let ptr_to_int = llvm::PtrToIntOp::new(ctx, shared_ptr, result_ty.into());
    rewriter.insert_operation(ctx, ptr_to_int.get_operation());
    rewriter.replace_operation(ctx, op, ptr_to_int.get_operation());
    Ok(())
}

pub(crate) fn convert_cvta_generic_to_cluster_shared_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("cvta_generic_to_shared_offset requires an operand");
    }
    let space = operands[0]
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|ty| ty.address_space());
    if !matches!(space, Some(0 | 3 | 7)) {
        return pliron::input_err_noloc!(
            "cluster shared offset requires generic or shared pointer"
        );
    }
    let shared_ptr = cast_to_cluster_shared_addrspace(ctx, rewriter, operands[0]);

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let Some(result_width) = result_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(IntegerType::width)
    else {
        return pliron::input_err_noloc!(
            "cvta_generic_to_shared_offset requires an integer result"
        );
    };
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    let ptr_to_int = llvm::PtrToIntOp::new(ctx, shared_ptr, result_ty.into());
    rewriter.insert_operation(ctx, ptr_to_int.get_operation());
    rewriter.replace_operation(ctx, op, ptr_to_int.get_operation());
    Ok(())
}

/// Field layout of the tcgen05 shared-memory descriptor: `(drop, mask, shift)`
/// for the operands `[address, leading_dim_bytes, stride_bytes, base_offset,
/// leading_dim_absolute, swizzle]`. A field loses its low `drop` bits, is
/// masked and then shifted into place.
const TCGEN05_SMEM_DESCRIPTOR_FIELDS: [(i64, i64, i64); 6] = [
    (4, 0x3FFF, 0),
    (4, 0x3FFF, 16),
    (4, 0x3FFF, 32),
    (0, 0x7, 49),
    (0, 0x1, 52),
    (0, 0x7, 61),
];

/// Fixed descriptor bits 46-48, value `0b001`.
const TCGEN05_SMEM_DESCRIPTOR_FIXED: i64 = 1 << 46;

/// Convert `nvvm.tcgen05_smem_descriptor` to `zext`, `lshr`, `and`, `shl` and `or`.
///
/// The result is the value `cuda_device::tcgen05::tcgen05_smem_descriptor`
/// computes on the host. Plain arithmetic keeps constant fields foldable.
pub(crate) fn convert_tcgen05_smem_descriptor(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != TCGEN05_SMEM_DESCRIPTOR_FIELDS.len() {
        return pliron::input_err_noloc!(
            "tcgen05_smem_descriptor requires [address, leading_dim_bytes, stride_bytes, base_offset, leading_dim_absolute, swizzle]"
        );
    }
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let mut descriptor = create_i64_const(ctx, rewriter, TCGEN05_SMEM_DESCRIPTOR_FIXED);
    for (operand, (drop, mask, shift)) in operands.into_iter().zip(TCGEN05_SMEM_DESCRIPTOR_FIELDS) {
        let Some(width) = operand
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width)
        else {
            return pliron::input_err_noloc!("tcgen05_smem_descriptor operands must be integers");
        };
        let mut field = operand;
        if width < 64 {
            let wide = llvm::ZExtOp::new_with_nneg(ctx, field, i64_ty.into(), false);
            rewriter.insert_operation(ctx, wide.get_operation());
            field = wide.get_operation().deref(ctx).get_result(0);
        }
        if drop != 0 {
            let amount = create_i64_const(ctx, rewriter, drop);
            let dropped = llvm::LShrOp::new(ctx, field, amount).get_operation();
            rewriter.insert_operation(ctx, dropped);
            field = dropped.deref(ctx).get_result(0);
        }
        let mask = create_i64_const(ctx, rewriter, mask);
        let masked = llvm::AndOp::new(ctx, field, mask).get_operation();
        rewriter.insert_operation(ctx, masked);
        field = masked.deref(ctx).get_result(0);
        if shift != 0 {
            let amount = create_i64_const(ctx, rewriter, shift);
            let shifted = llvm::ShlOp::new(ctx, field, amount).get_operation();
            shifted.deref_mut(ctx).attributes.set(
                llvm_export::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
                IntegerOverflowFlagsAttr::default(),
            );
            rewriter.insert_operation(ctx, shifted);
            field = shifted.deref(ctx).get_result(0);
        }
        let packed = llvm::OrOp::new(ctx, descriptor, field).get_operation();
        rewriter.insert_operation(ctx, packed);
        descriptor = packed.deref(ctx).get_result(0);
    }
    rewriter.replace_operation_with_values(ctx, op, vec![descriptor]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::{
        append_mir_return, build_kernel, find_all, kernel_blocks, make_ctx,
    };
    use dialect_mir::ops::MirConstantOp;
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::{
        CvtaGenericToClusterSharedOffsetOp, CvtaGenericToSharedOffsetOp, Tcgen05SmemDescriptorOp,
    };
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::r#type::TypeHandle;
    use pliron::utils::apint::APInt;
    use pliron::value::Value;
    use std::num::NonZeroUsize;

    /// The descriptor encoding as `Tcgen05SmemDescriptorBuilder::build` wrote
    /// it out before the descriptor operation existed. Inputs are the builder's
    /// fields: byte address, byte offsets, base offset, mode and swizzle.
    fn reference_smem_descriptor(
        address: u64,
        leading_dim_bytes: u32,
        stride_bytes: u32,
        base_offset: u8,
        leading_dim_absolute: bool,
        swizzle: u8,
    ) -> u64 {
        let mut raw: u64 = 0;
        raw |= (address >> 4) & 0x3FFF;
        raw |= (((leading_dim_bytes >> 4) & 0x3FFF) as u64) << 16;
        raw |= (((stride_bytes >> 4) & 0x3FFF) as u64) << 32;
        raw |= 1u64 << 46;
        raw |= ((base_offset & 0x7) as u64) << 49;
        if leading_dim_absolute {
            raw |= 1u64 << 52;
        }
        raw |= (swizzle as u64) << 61;
        raw
    }

    /// Evaluate lowered integer arithmetic whose leaves are constants.
    fn evaluate(ctx: &Context, value: Value) -> u64 {
        let op = value
            .defining_op()
            .expect("lowered descriptor arithmetic is defined by operations");
        if let Some(constant) = Operation::get_op::<llvm::ConstantOp>(op, ctx) {
            let attribute = constant.get_value(ctx);
            return (&*attribute as &dyn pliron::attribute::Attribute)
                .downcast_ref::<IntegerAttr>()
                .expect("integer constant")
                .value()
                .to_u64();
        }
        let operand = |index: usize| evaluate(ctx, op.deref(ctx).get_operand(index));
        if Operation::get_op::<llvm::ZExtOp>(op, ctx).is_some() {
            operand(0)
        } else if Operation::get_op::<llvm::LShrOp>(op, ctx).is_some() {
            operand(0) >> operand(1)
        } else if Operation::get_op::<llvm::AndOp>(op, ctx).is_some() {
            operand(0) & operand(1)
        } else if Operation::get_op::<llvm::ShlOp>(op, ctx).is_some() {
            operand(0) << operand(1)
        } else if Operation::get_op::<llvm::OrOp>(op, ctx).is_some() {
            operand(0) | operand(1)
        } else {
            panic!("unexpected operation in the lowered descriptor")
        }
    }

    /// Lower one descriptor with constant fields and evaluate the result.
    fn lowered_smem_descriptor(fields: [u64; 6]) -> u64 {
        let mut ctx = make_ctx();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let (module, block) = build_kernel(&mut ctx, vec![], vec![u64_ty]);
        let mut operands = vec![];
        for (value, width) in fields.into_iter().zip([64, 32, 32, 8, 1, 8]) {
            // The importer types `bool` as a signless one-bit integer.
            let signedness = if width == 1 {
                Signedness::Signless
            } else {
                Signedness::Unsigned
            };
            let ty = IntegerType::get(&ctx, width, signedness);
            let constant = Operation::new(
                &mut ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![ty.into()],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(constant).set_attr_value(
                &ctx,
                IntegerAttr::new(
                    ty,
                    APInt::from_u64(value, NonZeroUsize::new(width as usize).unwrap()),
                ),
            );
            constant.insert_at_back(block, &ctx);
            operands.push(constant.deref(&ctx).get_result(0));
        }
        let descriptor = Operation::new(
            &mut ctx,
            Tcgen05SmemDescriptorOp::get_concrete_op_info(),
            vec![u64_ty],
            operands,
            vec![],
            0,
        );
        descriptor.insert_at_back(block, &ctx);
        let value = descriptor.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![value]);
        crate::lower_mir_to_llvm(&mut ctx, module).unwrap();
        let blocks = kernel_blocks(&ctx, module);
        assert!(find_all::<Tcgen05SmemDescriptorOp>(&ctx, &blocks).is_empty());
        let returned = find_all::<llvm::ReturnOp>(&ctx, &blocks);
        assert_eq!(returned.len(), 1);
        evaluate(&ctx, returned[0].get_operation().deref(&ctx).get_operand(0))
    }

    #[test]
    fn tcgen05_smem_descriptor_lowers_to_the_written_out_encoding() {
        // The two descriptors of the canonical tests in cutlass-primitives.
        assert_eq!(
            lowered_smem_descriptor([0x1000, 16, 1024, 0, 0, 2]),
            0x4000_4040_0001_0100
        );
        assert_eq!(
            lowered_smem_descriptor([0x1000, 2048, 128, 0, 0, 0]),
            0x0000_4008_0080_0100
        );
        // In-range values, field boundaries, and values wider than a field.
        let addresses = [0u64, 0x1000, 0x3_FFF0, 0x4_0000, 0xFFFF_FFFF_FFFF_FFFF];
        let offsets = [0u32, 16, 2048, 0x3_FFF0, 0xFFFF_FFFF];
        for address in addresses {
            for leading in offsets {
                for stride in offsets {
                    for (base_offset, absolute, swizzle) in [
                        (0u8, false, 0u8),
                        (5, true, 2),
                        (0xFF, false, 6),
                        (7, true, 4),
                    ] {
                        let lowered = lowered_smem_descriptor([
                            address,
                            leading as u64,
                            stride as u64,
                            base_offset as u64,
                            absolute as u64,
                            swizzle as u64,
                        ]);
                        assert_eq!(
                            lowered,
                            reference_smem_descriptor(
                                address,
                                leading,
                                stride,
                                base_offset,
                                absolute,
                                swizzle
                            ),
                            "address={address:#x} leading={leading:#x} stride={stride:#x}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn cvta_preserves_local_and_cluster_offsets() {
        for space in [0, 3, 7] {
            for width in [32, 64] {
                let mut ctx = make_ctx();
                let word: TypeHandle = IntegerType::get(&ctx, width, Signedness::Unsigned).into();
                let ptr: TypeHandle = MirPtrType::get(&mut ctx, word, true, space).into();
                let (module, block) = build_kernel(&mut ctx, vec![ptr], vec![word]);
                let arg = block.deref(&ctx).get_argument(0);
                let cvta = Operation::new(
                    &mut ctx,
                    CvtaGenericToSharedOffsetOp::get_concrete_op_info(),
                    vec![word],
                    vec![arg],
                    vec![],
                    0,
                );
                cvta.insert_at_back(block, &ctx);
                let value = cvta.deref(&ctx).get_result(0);
                append_mir_return(&mut ctx, block, vec![value]);
                crate::lower_mir_to_llvm(&mut ctx, module).unwrap();
                let blocks = kernel_blocks(&ctx, module);
                let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &blocks);
                let ints = find_all::<llvm::PtrToIntOp>(&ctx, &blocks);
                assert_eq!(casts.len(), usize::from(space == 0));
                assert_eq!(ints.len(), 1);
                let operand = ints[0].get_operation().deref(&ctx).get_operand(0);
                let ty = operand.get_type(&ctx);
                assert_eq!(
                    ty.deref(&ctx)
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .unwrap()
                        .address_space(),
                    if space == 0 { 3 } else { space }
                );
                if space == 0 {
                    assert_eq!(operand, casts[0].get_operation().deref(&ctx).get_result(0));
                }
            }
        }
    }
    #[test]
    fn cluster_cvta_targets_cluster_window_from_generic_and_named_pointers() {
        for space in [0, 3, 7] {
            for width in [32, 64] {
                let mut ctx = make_ctx();
                let word: TypeHandle = IntegerType::get(&ctx, width, Signedness::Unsigned).into();
                let ptr: TypeHandle = MirPtrType::get(&mut ctx, word, true, space).into();
                let (module, block) = build_kernel(&mut ctx, vec![ptr], vec![word]);
                let arg = block.deref(&ctx).get_argument(0);
                let cvta = Operation::new(
                    &mut ctx,
                    CvtaGenericToClusterSharedOffsetOp::get_concrete_op_info(),
                    vec![word],
                    vec![arg],
                    vec![],
                    0,
                );
                cvta.insert_at_back(block, &ctx);
                let value = cvta.deref(&ctx).get_result(0);
                append_mir_return(&mut ctx, block, vec![value]);
                crate::lower_mir_to_llvm(&mut ctx, module).unwrap();
                let blocks = kernel_blocks(&ctx, module);
                let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &blocks);
                let ints = find_all::<llvm::PtrToIntOp>(&ctx, &blocks);
                assert_eq!(casts.len(), usize::from(space != 7));
                assert_eq!(ints.len(), 1);
                let operand = ints[0].get_operation().deref(&ctx).get_operand(0);
                let ty = operand.get_type(&ctx);
                assert_eq!(
                    ty.deref(&ctx)
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .unwrap()
                        .address_space(),
                    7
                );
                if space != 7 {
                    assert_eq!(operand, casts[0].get_operation().deref(&ctx).get_result(0));
                }
            }
        }
    }
}
