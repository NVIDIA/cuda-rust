/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Primitive integer `Ord::min` and `Ord::max` intrinsics.

use super::super::helpers;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::{rvalue, types, values::ValueMap};
use dialect_mir::{
    ops::{MirCallOp, MirLtOp},
    rust_intrinsics,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::{
    attributes::StringAttr,
    types::{IntegerType, Signedness},
};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

/// Integer ordering intrinsics emitted by libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustIntegerMinMaxIntrinsic {
    /// Select the smaller operand.
    Min,
    /// Select the larger operand.
    Max,
}

impl RustIntegerMinMaxIntrinsic {
    /// Recognize exact libcore intrinsic paths, including the std reexports.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::integer_min" | "std::intrinsics::integer_min" => Some(Self::Min),
            "core::intrinsics::integer_max" | "std::intrinsics::integer_max" => Some(Self::Max),
            _ => None,
        }
    }
}

/// Match core's specified fallback: `a < b` selects `(a, b)` for min and
/// `(b, a)` for max. Keeping the original integer types lets existing compare
/// lowering recover signedness, without narrowing any width (including 128).
fn emit_integer_minmax(
    ctx: &mut Context,
    intrinsic: RustIntegerMinMaxIntrinsic,
    args: &[Value],
    return_type: TypeHandle,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let [lhs, rhs] = args else {
        return input_err!(
            loc,
            TranslationErr::type_error("integer min/max expects two operands")
        );
    };
    let valid_integer = return_type
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|ty| {
            matches!(ty.width(), 8 | 16 | 32 | 64 | 128)
                && matches!(ty.signedness(), Signedness::Signed | Signedness::Unsigned)
        });
    if !valid_integer || lhs.get_type(ctx) != return_type || rhs.get_type(ctx) != return_type {
        return input_err!(
            loc,
            TranslationErr::type_error(
                "integer min/max requires matching primitive integer operands and result"
            )
        );
    }

    let bool_type = types::get_bool_type(ctx).into();
    let cmp = Operation::new(
        ctx,
        MirLtOp::get_concrete_op_info(),
        vec![bool_type],
        vec![*lhs, *rhs],
        vec![],
        0,
    );
    cmp.deref_mut(ctx).set_loc(loc.clone());
    helpers::insert_op(ctx, cmp, block, prev_op);
    let condition = cmp.deref(ctx).get_result(0);
    let (if_true, if_false) = match intrinsic {
        RustIntegerMinMaxIntrinsic::Min => (*lhs, *rhs),
        RustIntegerMinMaxIntrinsic::Max => (*rhs, *lhs),
    };
    // This existing placeholder lowers to an ordinary LLVM select; it carries
    // no unpredictable metadata or other optimization promises.
    let select = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![return_type],
        vec![condition, if_true, if_false],
        vec![],
        0,
    );
    select.deref_mut(ctx).set_loc(loc);
    select.deref_mut(ctx).attributes.set(
        "callee".try_into().unwrap(),
        StringAttr::new(rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE.into()),
    );
    select.insert_after(ctx, cmp);
    Ok(select)
}

/// Translate integer min/max using the ordinary compare and select lowering.
pub fn emit_rust_integer_minmax_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustIntegerMinMaxIntrinsic,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;
    let mut values = Vec::with_capacity(args.len());
    let mut last_op = prev_op;
    for arg in args {
        let (value, last) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        values.push(value);
        last_op = last;
    }
    let (prepared, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    let select = emit_integer_minmax(
        ctx,
        intrinsic,
        &values,
        return_type,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    let result = select.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared,
        result,
        target,
        block_ptr,
        select,
        value_map,
        block_map,
        loc,
        "integer min/max call without target block",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm_export::{attributes::ICmpPredicateAttr, ops as llvm};
    use pliron::builtin::{
        attributes::TypeAttr, op_interfaces::SymbolOpInterface, ops::ModuleOp, types::FunctionType,
    };
    use pliron::common_traits::Verify;
    use pliron::linked_list::ContainsLinkedList;

    #[test]
    fn recognizes_only_integer_intrinsic_paths() {
        for prefix in ["core", "std"] {
            assert_eq!(
                RustIntegerMinMaxIntrinsic::from_core_path(&format!(
                    "{prefix}::intrinsics::integer_min"
                )),
                Some(RustIntegerMinMaxIntrinsic::Min)
            );
            assert_eq!(
                RustIntegerMinMaxIntrinsic::from_core_path(&format!(
                    "{prefix}::intrinsics::integer_max"
                )),
                Some(RustIntegerMinMaxIntrinsic::Max)
            );
        }
        for path in [
            "user::intrinsics::integer_min",
            "core::intrinsics::integer_minimum",
            "core::intrinsics::integer_max_extra",
            "core::cmp::min",
            "core::intrinsics::minimum_number_nsz",
        ] {
            assert_eq!(RustIntegerMinMaxIntrinsic::from_core_path(path), None);
        }
    }

    #[test]
    fn lowers_minmax_with_original_width_signedness_and_selected_operands() {
        for width in [8, 16, 32, 64, 128] {
            for (signedness, predicate) in [
                (Signedness::Signed, ICmpPredicateAttr::SLT),
                (Signedness::Unsigned, ICmpPredicateAttr::ULT),
            ] {
                for (intrinsic, true_index, false_index) in [
                    (RustIntegerMinMaxIntrinsic::Min, 0, 1),
                    (RustIntegerMinMaxIntrinsic::Max, 1, 0),
                ] {
                    let mut ctx = Context::new();
                    dialect_mir::register(&mut ctx);
                    mir_lower::register(&mut ctx);
                    let ty = IntegerType::get(&ctx, width, signedness).into();
                    let module =
                        ModuleOp::new(&mut ctx, "minmax".try_into().unwrap()).get_operation();
                    let func_type = FunctionType::get(&ctx, vec![ty, ty], vec![ty]);
                    let func = Operation::new(
                        &mut ctx,
                        dialect_mir::ops::MirFuncOp::get_concrete_op_info(),
                        vec![],
                        vec![],
                        vec![],
                        1,
                    );
                    let mir_func = dialect_mir::ops::MirFuncOp::new(
                        &mut ctx,
                        func,
                        TypeAttr::new(func_type.into()),
                    );
                    mir_func.set_symbol_name(&mut ctx, "minmax".try_into().unwrap());
                    let module_region = module.deref(&ctx).get_region(0);
                    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
                    func.insert_at_back(module_block, &ctx);
                    let region = func.deref(&ctx).get_region(0);
                    let block = BasicBlock::new(&mut ctx, None, vec![ty, ty]);
                    block.insert_at_back(region, &ctx);
                    let args = [
                        block.deref(&ctx).get_argument(0),
                        block.deref(&ctx).get_argument(1),
                    ];
                    let select = emit_integer_minmax(
                        &mut ctx,
                        intrinsic,
                        &args,
                        ty,
                        block,
                        None,
                        Location::Unknown,
                    )
                    .unwrap();
                    let result = select.deref(&ctx).get_result(0);
                    let ret = Operation::new(
                        &mut ctx,
                        dialect_mir::ops::MirReturnOp::get_concrete_op_info(),
                        vec![],
                        vec![result],
                        vec![],
                        0,
                    );
                    ret.insert_after(&ctx, select);
                    mir_lower::lower_mir_to_llvm(&mut ctx, module).unwrap();
                    module.deref(&ctx).verify(&ctx).unwrap();
                    let func = module_block.deref(&ctx).iter(&ctx).next().unwrap();
                    let region = func.deref(&ctx).get_region(0);
                    let operations: Vec<_> = region
                        .deref(&ctx)
                        .iter(&ctx)
                        .flat_map(|block| block.deref(&ctx).iter(&ctx))
                        .collect();
                    let compare = operations
                        .iter()
                        .find_map(|op| Operation::get_op::<llvm::ICmpOp>(*op, &ctx))
                        .unwrap();
                    assert_eq!(compare.predicate(&ctx), predicate);
                    let compare = compare.get_operation();
                    let select = operations
                        .iter()
                        .find_map(|op| Operation::get_op::<llvm::SelectOp>(*op, &ctx))
                        .unwrap()
                        .get_operation();
                    assert_eq!(
                        select.deref(&ctx).get_operand(0),
                        compare.deref(&ctx).get_result(0)
                    );
                    assert_eq!(
                        select.deref(&ctx).get_operand(1),
                        block.deref(&ctx).get_argument(true_index)
                    );
                    assert_eq!(
                        select.deref(&ctx).get_operand(2),
                        block.deref(&ctx).get_argument(false_index)
                    );
                    assert_eq!(
                        compare.deref(&ctx).get_operand(0),
                        block.deref(&ctx).get_argument(0)
                    );
                    assert_eq!(
                        compare.deref(&ctx).get_operand(1),
                        block.deref(&ctx).get_argument(1)
                    );
                    assert_eq!(
                        select
                            .deref(&ctx)
                            .get_result(0)
                            .get_type(&ctx)
                            .deref(&ctx)
                            .downcast_ref::<IntegerType>()
                            .unwrap()
                            .width(),
                        width
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_integer_minmax_signatures_before_emission() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let integer: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let signed: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
        let wide: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let signless: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let odd_width: TypeHandle = IntegerType::get(&ctx, 24, Signedness::Unsigned).into();
        let float: TypeHandle = pliron::builtin::types::FP32Type::get(&ctx).into();
        for (args, result) in [
            (vec![integer], integer),
            (vec![integer, integer, integer], integer),
            (vec![integer, wide], integer),
            (vec![integer, signed], integer),
            (vec![integer, integer], wide),
            (vec![boolean, boolean], boolean),
            (vec![signless, signless], signless),
            (vec![odd_width, odd_width], odd_width),
            (vec![float, float], float),
        ] {
            let block = BasicBlock::new(&mut ctx, None, args.clone());
            let values: Vec<_> = (0..args.len())
                .map(|index| block.deref(&ctx).get_argument(index))
                .collect();
            assert!(
                emit_integer_minmax(
                    &mut ctx,
                    RustIntegerMinMaxIntrinsic::Min,
                    &values,
                    result,
                    block,
                    None,
                    Location::Unknown
                )
                .is_err()
            );
            assert!(block.deref(&ctx).iter(&ctx).next().is_none());
        }
    }
}
