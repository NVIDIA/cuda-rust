/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts constant definitions from `dialect-mir` to the LLVM dialect.
//!
//! # Supported Operations
//!
//! | `dialect-mir` Op     | LLVM dialect Op   | Description       |
//! |----------------------|-------------------|-------------------|
//! | `mir.constant`       | `llvm.constant`   | Integer constants |
//! | `mir.float_constant` | `llvm.constant`   | Float constants   |
//!
//! # Type Handling
//!
//! `dialect-mir` uses signed/unsigned integer types (`ui64`, `si64`), while
//! the LLVM dialect uses signless integers (`i64`). The conversion preserves
//! bit-width but changes to the signless representation.
//!
//! Float constants (f32, f64) pass through unchanged.

use dialect_mir::attributes::MirFP16Attr;
use dialect_mir::ops::{MirConstantOp, MirFloatConstantOp, MirUndefOp};
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

use crate::convert::types::convert_type;

/// Convert `mir.constant` (integer) to `llvm.constant`.
///
/// MIR integer types are signed/unsigned (`ui64`, `si64`), but LLVM uses
/// signless integers. This conversion preserves the bit pattern and width
/// while changing to signless representation.
pub(crate) fn convert_integer(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use pliron::builtin::attributes::IntegerAttr;

    let (ap_int_value, width) = {
        let mir_const = MirConstantOp::new(op);
        let int_attr = mir_const.get_attr_value(ctx).ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "Missing value attribute on mir.constant"
            )
        })?;

        let ap_int = int_attr.value().clone();
        let mir_int_ty = int_attr.get_type();
        let w = mir_int_ty.deref(ctx).width();
        (ap_int, w)
    };

    // Create signless LLVM integer type (MIR uses signed/unsigned, LLVM uses signless)
    let llvm_int_ty = IntegerType::get(ctx, width, Signedness::Signless);
    let llvm_int_attr = IntegerAttr::new(llvm_int_ty, ap_int_value);

    let llvm_const = llvm::ConstantOp::new(ctx, Box::new(llvm_int_attr));
    rewriter.insert_operation(ctx, llvm_const.get_operation());
    rewriter.replace_operation(ctx, op, llvm_const.get_operation());

    Ok(())
}

/// Normalise a signed/unsigned `builtin.constant` (materialised by `sccp`) to a
/// signless integer constant, exactly like `convert_integer` does for
/// `mir.constant`. Only non-signless builtin.constants are routed here (see
/// `can_convert_op`), so the signless result is final (the worklist converges).
pub(crate) fn convert_builtin_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    value: Box<dyn pliron::builtin::attr_interfaces::TypedAttrInterface>,
) -> Result<()> {
    use pliron::builtin::attributes::IntegerAttr;

    let (apint_value, width) = {
        let int_attr = (&*value as &dyn pliron::attribute::Attribute)
            .downcast_ref::<IntegerAttr>()
            .ok_or_else(|| {
                pliron::input_error!(
                    op.deref(ctx).loc(),
                    "builtin.constant routed to lowering must hold an IntegerAttr"
                )
            })?;
        (
            int_attr.value().clone(),
            int_attr.get_type().deref(ctx).width(),
        )
    };

    let llvm_int_ty = IntegerType::get(ctx, width, Signedness::Signless);
    let llvm_int_attr = IntegerAttr::new(llvm_int_ty, apint_value);

    let llvm_const = llvm::ConstantOp::new(ctx, Box::new(llvm_int_attr));
    rewriter.insert_operation(ctx, llvm_const.get_operation());
    rewriter.replace_operation(ctx, op, llvm_const.get_operation());

    Ok(())
}

/// Convert `mir.float_constant` to `llvm.constant`.
///
/// Float constants pass through with their type preserved (f32, f64).
pub(crate) fn convert_float(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    enum FloatAttr {
        F16(MirFP16Attr),
        F32(pliron::builtin::attributes::FPSingleAttr),
        F64(pliron::builtin::attributes::FPDoubleAttr),
    }

    let float_attr = {
        let mir_const = MirFloatConstantOp::new(op);
        if let Some(attr) = mir_const.get_attr_float_value_f16(ctx) {
            FloatAttr::F16(attr.clone())
        } else if let Some(attr) = mir_const.get_attr_float_value(ctx) {
            FloatAttr::F32(attr.clone())
        } else if let Some(attr) = mir_const.get_attr_float_value_f64(ctx) {
            FloatAttr::F64(attr.clone())
        } else {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "Missing float_value or float_value_f64 attribute on mir.float_constant"
            );
        }
    };

    let llvm_const = match float_attr {
        FloatAttr::F16(attr) => llvm::ConstantOp::new(
            ctx,
            Box::new(llvm_export::fp16_attr_from_bits(attr.to_bits())),
        ),
        FloatAttr::F32(attr) => llvm::ConstantOp::new(ctx, Box::new(attr)),
        FloatAttr::F64(attr) => llvm::ConstantOp::new(ctx, Box::new(attr)),
    };

    rewriter.insert_operation(ctx, llvm_const.get_operation());
    rewriter.replace_operation(ctx, op, llvm_const.get_operation());

    Ok(())
}

/// Convert `mir.undef` to `llvm.undef`.
///
/// Passes the converted result type through to `llvm::UndefOp::new`.
pub(crate) fn convert_undef(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let _mir_undef = Operation::get_op::<MirUndefOp>(op, ctx).expect("expected MirUndefOp");
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let llvm_result_ty = convert_type(ctx, result_ty).map_err(|e| {
        pliron::create_error!(
            op.deref(ctx).loc(),
            pliron::result::ErrorKind::VerificationFailed,
            "{e}"
        )
    })?;

    let llvm_undef = llvm::UndefOp::new(ctx, llvm_result_ty);
    rewriter.insert_operation(ctx, llvm_undef.get_operation());
    rewriter.replace_operation(ctx, op, llvm_undef.get_operation());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;
    use llvm_export::attributes::FPHalfAttr;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr};
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::TypeHandle;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    fn append_op_and_return(ctx: &mut Context, block: Ptr<BasicBlock>, op: Ptr<Operation>) {
        op.insert_at_back(block, ctx);
        let result = op.deref(ctx).get_result(0);
        append_mir_return(ctx, block, vec![result]);
    }

    fn lowered_constants(ctx: &mut Context, module_ptr: Ptr<Operation>) -> Vec<llvm::ConstantOp> {
        crate::lower_mir_to_llvm(ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(ctx, module_ptr);
        find_all::<llvm::ConstantOp>(ctx, &body)
    }

    fn assert_i32_constant_lowers_to_signless(signedness: Signedness, value: APInt) {
        let mut ctx = make_ctx();
        let mir_i32_ty = IntegerType::get(&ctx, 32, signedness);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![mir_i32_ty.into()]);
        let const_op = Operation::new(
            &mut ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![mir_i32_ty.into()],
            vec![],
            vec![],
            0,
        );
        MirConstantOp::new(const_op)
            .set_attr_value(&ctx, IntegerAttr::new(mir_i32_ty, value.clone()));
        append_op_and_return(&mut ctx, block, const_op);

        let constants = lowered_constants(&mut ctx, module_ptr);
        assert_eq!(constants.len(), 1, "expected exactly one lowered constant");

        let attr = constants[0].get_value(&ctx);
        let int_attr = (&*attr as &dyn pliron::attribute::Attribute)
            .downcast_ref::<IntegerAttr>()
            .expect("expected lowered integer attribute");
        let int_ty = int_attr.get_type();
        let int_ty = int_ty.deref(&ctx);

        assert_eq!(int_ty.width(), 32);
        assert_eq!(int_ty.signedness(), Signedness::Signless);
        assert_eq!(int_attr.value(), value);
    }

    #[test]
    fn convert_integer_preserves_bits_and_makes_type_signless() {
        assert_i32_constant_lowers_to_signless(
            Signedness::Signed,
            APInt::from_i64(-7, NonZeroUsize::new(32).unwrap()),
        );
        assert_i32_constant_lowers_to_signless(
            Signedness::Unsigned,
            APInt::from_u64(42, NonZeroUsize::new(32).unwrap()),
        );
    }

    #[test]
    fn convert_float_preserves_f32_attr() {
        let mut ctx = make_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f32_value = 1.25_f32;

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![f32_ty]);
        let f32_op = Operation::new(
            &mut ctx,
            MirFloatConstantOp::get_concrete_op_info(),
            vec![f32_ty],
            vec![],
            vec![],
            0,
        );
        MirFloatConstantOp::new(f32_op).set_attr_float_value(&ctx, FPSingleAttr::from(f32_value));
        append_op_and_return(&mut ctx, block, f32_op);

        let constants = lowered_constants(&mut ctx, module_ptr);
        assert_eq!(constants.len(), 1, "expected exactly one lowered constant");

        let attr = constants[0].get_value(&ctx);
        let attr = (&*attr as &dyn pliron::attribute::Attribute)
            .downcast_ref::<FPSingleAttr>()
            .expect("expected lowered f32 attribute");
        assert_eq!(f32::from(attr.clone()).to_bits(), f32_value.to_bits());
    }

    #[test]
    fn convert_float_preserves_f64_attr() {
        let mut ctx = make_ctx();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let f64_value = -2.5_f64;

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![f64_ty]);
        let f64_op = Operation::new(
            &mut ctx,
            MirFloatConstantOp::get_concrete_op_info(),
            vec![f64_ty],
            vec![],
            vec![],
            0,
        );
        MirFloatConstantOp::new(f64_op)
            .set_attr_float_value_f64(&ctx, FPDoubleAttr::from(f64_value));
        append_op_and_return(&mut ctx, block, f64_op);

        let constants = lowered_constants(&mut ctx, module_ptr);
        assert_eq!(constants.len(), 1, "expected exactly one lowered constant");

        let attr = constants[0].get_value(&ctx);
        let attr = (&*attr as &dyn pliron::attribute::Attribute)
            .downcast_ref::<FPDoubleAttr>()
            .expect("expected lowered f64 attribute");
        assert_eq!(f64::from(attr.clone()).to_bits(), f64_value.to_bits());
    }

    #[test]
    fn convert_f16_constant_rewrites_mir_attr_to_builtin_half_attr() {
        let mut ctx = make_ctx();
        let f16_ty: TypeHandle = dialect_mir::types::MirFP16Type::get(&ctx).into();
        let bits = 0x3c00;

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![f16_ty]);
        let const_op = Operation::new(
            &mut ctx,
            MirFloatConstantOp::get_concrete_op_info(),
            vec![f16_ty],
            vec![],
            vec![],
            0,
        );
        MirFloatConstantOp::new(const_op)
            .set_attr_float_value_f16(&ctx, MirFP16Attr::from_bits(bits));
        append_op_and_return(&mut ctx, block, const_op);

        let constants = lowered_constants(&mut ctx, module_ptr);
        assert_eq!(constants.len(), 1, "expected exactly one lowered constant");

        let attr = constants[0].get_value(&ctx);
        let attr = &*attr as &dyn pliron::attribute::Attribute;
        assert!(
            attr.downcast_ref::<MirFP16Attr>().is_none(),
            "lowering must not keep the MIR-specific f16 attribute"
        );

        let half_attr = attr
            .downcast_ref::<FPHalfAttr>()
            .expect("expected builtin half attribute");
        assert_eq!(llvm_export::fp16_attr_to_bits(half_attr), bits);
    }
}
