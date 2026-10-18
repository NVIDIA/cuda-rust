// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! Lowering of LLVM dialect IR to a form that MLIR's LLVM dialect can hold.
//!
//! Call [lower_for_mlir] before printing the IR with [MlirPrinter](crate::MlirPrinter).

use pliron::{
    attribute::{AttrObj, Attribute, boxed_attr_cast},
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::IntegerAttr,
        given_names::{get_operation_result_name, set_operation_result_name},
        op_interfaces::OneResultInterface,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    dyn_clone, input_err_noloc,
    irbuild::{
        dialect_conversion::{
            DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
        },
        inserter::Inserter,
        rewriter::Rewriter,
    },
    location::Location,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};
use pliron_llvm::{
    attributes::{AggregateAttr, PoisonAttr, SplatAttr, SymbolAddrAttr},
    ops::{
        ConstantOp, GlobalOp, InsertElementOp, InsertValueOp, ReturnOp, UndefOp,
        global_op_attr_names::ATTR_KEY_LLVM_GLOBAL_INITIALIZER,
    },
    types::VectorType,
};

use crate::{
    Error,
    llvm::constants::{constant_type, is_constant_attr},
};

/// Lower the LLVM dialect IR in `op` to a form that MLIR's LLVM dialect can hold.
pub fn lower_for_mlir(ctx: &mut Context, op: Ptr<Operation>) -> Result<()> {
    apply_dialect_conversion(ctx, &mut MlirLowering, op)?;
    Ok(())
}

/// Should [lower_for_mlir] lower the constant `value`?
pub fn constant_needs_lowering(ctx: &Context, value: &dyn Attribute) -> bool {
    (value.is::<AggregateAttr>() || value.is::<SplatAttr>()) && !is_constant_attr(ctx, value)
}

/// Should [lower_for_mlir] lower the global initializer `value`?
pub fn global_initializer_needs_lowering(ctx: &Context, value: &dyn Attribute) -> bool {
    value.is::<PoisonAttr>() || value.is::<SymbolAddrAttr>() || constant_needs_lowering(ctx, value)
}

/// Implements [DialectConversion] to MLIR compatible LLVM-IR.
struct MlirLowering;

impl DialectConversion for MlirLowering {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        let op_obj = Operation::get_op_dyn(op, ctx);
        if let Some(constant) = op_obj.downcast_ref::<ConstantOp>() {
            return constant_needs_lowering(ctx, &*constant.get_value(ctx) as &dyn Attribute);
        }
        if let Some(global) = op_obj.downcast_ref::<GlobalOp>() {
            return global
                .get_initializer_value(ctx)
                .is_some_and(|value| global_initializer_needs_lowering(ctx, &*value));
        }
        false
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let loc = op.deref(ctx).loc().clone();
        let op_obj = Operation::get_op_dyn(op, ctx);

        if let Some(constant) = op_obj.downcast_ref::<ConstantOp>().copied() {
            let value = dyn_clone::clone_box(&*constant.get_value(ctx)) as AttrObj;
            let built = build(ctx, rewriter, value, &loc)?;
            // Replace the constant, and keep its result name.
            if let Some(defining_op) = built.defining_op() {
                let name = get_operation_result_name(ctx, op, 0);
                set_operation_result_name(ctx, defining_op, 0, name);
            }
            rewriter.replace_operation_with_values(ctx, op, vec![built]);
            return Ok(());
        }

        let global = op_obj
            .downcast_ref::<GlobalOp>()
            .copied()
            .expect("MlirLowering converts only constants and globals");
        let value = global
            .get_initializer_value(ctx)
            .expect("MlirLowering converts only globals with an initializer value");
        op.deref_mut(ctx)
            .attributes
            .0
            .remove(&ATTR_KEY_LLVM_GLOBAL_INITIALIZER);
        global.add_initializer_region(ctx);
        let entry = global
            .get_initializer_block(ctx)
            .expect("the initializer region was just added");
        rewriter.set_insertion_point_to_block_end(entry);
        let built = build(ctx, rewriter, value, &loc)?;
        let ret = ReturnOp::new(ctx, Some(built));
        insert(ctx, rewriter, &ret, &loc);
        rewriter.mark_modified();
        Ok(())
    }
}

/// Insert `op` at the insertion point of `rewriter`, at `loc`.
fn insert(ctx: &Context, rewriter: &mut DialectConversionRewriter, op: &dyn Op, loc: &Location) {
    op.get_operation().deref_mut(ctx).set_loc(loc.clone());
    rewriter.insert_op(ctx, op);
}

/// Insert the operations that build the constant `value`, at `loc`.
/// Returns the built value.
fn build(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: AttrObj,
    loc: &Location,
) -> Result<Value> {
    let ty = constant_type(ctx, &*value)?;

    if !constant_needs_lowering(ctx, &*value) {
        let typed = boxed_attr_cast::<dyn TypedAttrInterface>(value)
            .expect("constant_type checked that the constant has a type");
        let leaf = ConstantOp::new(ctx, typed);
        insert(ctx, rewriter, &leaf, loc);
        return Ok(leaf.get_result(ctx));
    }

    let elems: Vec<Value> = if let Some(aggr) = value.downcast_ref::<AggregateAttr>() {
        aggr.elements()
            .iter()
            .map(|elem| build(ctx, rewriter, elem.clone() as AttrObj, loc))
            .collect::<Result<_>>()?
    } else {
        let splat = value
            .downcast_ref::<SplatAttr>()
            .expect("only aggregates and splats need lowering");
        let vector = splat.ty();
        let (is_scalable, num_elements) = {
            let vector = vector.deref(ctx);
            (vector.is_scalable(), vector.num_elements())
        };
        if is_scalable {
            return input_err_noloc!(Error::Untranslatable(
                "MLIR cannot build a scalable vector splat element by element".to_string()
            ));
        }
        // Build the element once, and insert it into every position.
        let elem = dyn_clone::clone_box(splat.element()) as AttrObj;
        let elem = build(ctx, rewriter, elem, loc)?;
        vec![elem; num_elements as usize]
    };
    insert_elements(ctx, rewriter, ty, &elems, loc)
}

/// Build a value of type `ty` from `elems`. Builds
/// - `llvm.mlir.undef`
/// - `llvm.insertelement` (vector) or `llvm.insertvalue` (array, struct) for each element.
///
/// Returns the built value.
fn insert_elements(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ty: TypeHandle,
    elems: &[Value],
    loc: &Location,
) -> Result<Value> {
    let undef = UndefOp::new(ctx, ty);
    insert(ctx, rewriter, &undef, loc);
    let mut acc = undef.get_result(ctx);

    let is_vector = ty.deref(ctx).is::<VectorType>();
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    for (idx, elem) in elems.iter().enumerate() {
        let op: Box<dyn Op> = if is_vector {
            let idx_attr =
                IntegerAttr::new(i64_ty, APInt::from_u64(idx as u64, 64.try_into().unwrap()));
            let idx_op = ConstantOp::new(ctx, Box::new(idx_attr));
            insert(ctx, rewriter, &idx_op, loc);
            let idx_value = idx_op.get_result(ctx);
            Box::new(InsertElementOp::new(ctx, acc, *elem, idx_value))
        } else {
            Box::new(InsertValueOp::new(ctx, acc, *elem, vec![idx as u32]))
        };
        insert(ctx, rewriter, &*op, loc);
        acc = op.get_operation().deref(ctx).get_result(0);
    }
    Ok(acc)
}
