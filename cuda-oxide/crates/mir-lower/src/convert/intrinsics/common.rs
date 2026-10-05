/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Common helpers for GPU intrinsic conversion.
//!
//! This module provides shared utility functions used across all GPU intrinsic
//! converters. These helpers handle common patterns like:
//!
//! - Creating LLVM constants (i1, i32, i64)
//! - Address space pointer casting (generic → shared)
//! - Declaring and calling LLVM intrinsics
//! - Creating inline PTX assembly with convergent attribute
//! - Type conversions for intrinsic results

use crate::helpers;
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::ops as llvm;
use llvm_export::ops::{AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Create an i1 (boolean) constant with the given value.
pub fn create_i1_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: bool,
) -> Value {
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let const_value = if value { 1i64 } else { 0i64 };
    let apint = APInt::from_i64(const_value, NonZeroUsize::new(1).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i1_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, Box::new(attr));
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Create an i32 constant with the given value.
pub fn create_i32_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: i32,
) -> Value {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let apint = APInt::from_i64(value as i64, NonZeroUsize::new(32).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, Box::new(attr));
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Create an i64 constant with the given value.
pub fn create_i64_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: i64,
) -> Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let apint = APInt::from_i64(value, NonZeroUsize::new(64).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, Box::new(attr));
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Cast a pointer value to address space 3 (shared memory) if needed.
pub fn cast_to_shared_addrspace(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ptr: Value,
) -> Value {
    let ptr_ty = ptr.get_type(ctx);
    let current_addrspace = ptr_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);

    if current_addrspace != 3 {
        let cast_ty = llvm_types::PointerType::get(ctx, 3).into();
        let cast_op = llvm::AddrSpaceCastOp::new(ctx, ptr, cast_ty);
        rewriter.insert_operation(ctx, cast_op.get_operation());
        cast_op.get_operation().deref(ctx).get_result(0)
    } else {
        ptr
    }
}

/// Cast a pointer to the cluster shared address space (`addrspace(7)`).
pub fn cast_to_cluster_shared_addrspace(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ptr: Value,
) -> Value {
    let ptr_ty = ptr.get_type(ctx);
    let current_addrspace = ptr_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);

    if current_addrspace != 7 {
        let cast_ty = llvm_types::PointerType::get(ctx, 7).into();
        let cast_op = llvm::AddrSpaceCastOp::new(ctx, ptr, cast_ty);
        rewriter.insert_operation(ctx, cast_op.get_operation());
        cast_op.get_operation().deref(ctx).get_result(0)
    } else {
        ptr
    }
}

/// Create an LLVM function call to an intrinsic.
///
/// Ensures the intrinsic is declared in the module, then creates a call.
/// `current_op` is the MIR op being converted (used to find the parent module).
pub fn call_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    intrinsic_name: &str,
    func_ty: pliron::r#type::TypedHandle<llvm_types::FuncType>,
    args: Vec<Value>,
) -> Result<Ptr<Operation>> {
    let parent_block = current_op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name.try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, args);
    crate::convert::preserve_location(ctx, current_op, llvm_call.get_operation());
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    Ok(llvm_call.get_operation())
}

/// Create an inline assembly operation with the convergent attribute.
///
/// `current_op` is the MIR op being converted; its source location is copied
/// onto the inline asm op so line info survives lowering.
pub fn inline_asm_convergent(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    result_ty: pliron::r#type::TypeHandle,
    inputs: Vec<Value>,
    asm_template: &str,
    constraints: &str,
) -> Ptr<Operation> {
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty,
        inputs,
        asm_template,
        constraints,
        AsmKind::Convergent,
    );
    crate::convert::preserve_location(ctx, current_op, inline_asm.get_operation());
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    inline_asm.get_operation()
}

/// Create an inline assembly operation with the sideeffect attribute (non-convergent).
///
/// Use this for operations that write to memory but are NOT warp-synchronous
/// (e.g., `cp.async` copies). Unlike `inline_asm_convergent`, the emitted asm
/// is marked `sideeffect` only, allowing LLVM to move or duplicate it across
/// divergent control flow when legal.
///
/// `current_op` is the MIR op being converted; its source location is copied
/// onto the inline asm op so line info survives lowering.
pub fn inline_asm_sideeffect(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    result_ty: pliron::r#type::TypeHandle,
    inputs: Vec<Value>,
    asm_template: &str,
    constraints: &str,
) -> Ptr<Operation> {
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty,
        inputs,
        asm_template,
        constraints,
        AsmKind::SideEffect,
    );
    crate::convert::preserve_location(ctx, current_op, inline_asm.get_operation());
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    inline_asm.get_operation()
}

/// Truncate an i32 result to i1 (for predicate results).
pub fn trunc_to_i1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    i32_val: Value,
) -> Value {
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let trunc_op = llvm::TruncOp::new(ctx, i32_val, i1_ty.into());
    rewriter.insert_operation(ctx, trunc_op.get_operation());
    trunc_op.get_operation().deref(ctx).get_result(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops as mir;
    use llvm_export::ops as llvm;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
    use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::FunctionType;
    use pliron::irbuild::dialect_conversion::{
        DialectConversion, OperandsInfo, apply_dialect_conversion,
    };
    use pliron::irbuild::rewriter::Rewriter;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::r#type::TypeHandle;

    const TEST_FUNC_NAME: &str = "helper_func";
    const TEST_INTRINSIC_NAME: &str = "llvm_test_intrinsic";

    #[derive(Clone, Copy)]
    enum HelperAction {
        Constants,
        SharedCast,
        ClusterSharedCast,
        CallIntrinsic,
        InlineAsmConvergent,
        InlineAsmSideEffect,
        TruncToI1,
    }

    struct HelperConversion {
        action: HelperAction,
        returned_value: Option<Value>,
    }

    impl DialectConversion for HelperConversion {
        fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
            Operation::get_opid(op, ctx) == mir::MirStorageLiveOp::get_opid_static()
        }

        fn can_convert_type(&self, _ctx: &Context, _ty: TypeHandle) -> bool {
            false
        }

        fn convert_type(&mut self, _ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
            Ok(ty)
        }

        fn rewrite(
            &mut self,
            ctx: &mut Context,
            rewriter: &mut DialectConversionRewriter,
            op: Ptr<Operation>,
            _operands_info: &OperandsInfo,
        ) -> Result<()> {
            let block = op
                .deref(ctx)
                .get_parent_block()
                .expect("test trigger must be inside a block");

            self.returned_value = match self.action {
                HelperAction::Constants => {
                    create_i1_const(ctx, rewriter, false);
                    create_i1_const(ctx, rewriter, true);
                    create_i32_const(ctx, rewriter, -7);
                    create_i64_const(ctx, rewriter, 42);
                    None
                }
                HelperAction::SharedCast => {
                    let ptr = block.deref(ctx).get_argument(0);
                    Some(cast_to_shared_addrspace(ctx, rewriter, ptr))
                }
                HelperAction::ClusterSharedCast => {
                    let ptr = block.deref(ctx).get_argument(0);
                    Some(cast_to_cluster_shared_addrspace(ctx, rewriter, ptr))
                }
                HelperAction::CallIntrinsic => {
                    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
                    let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);
                    call_intrinsic(ctx, rewriter, op, TEST_INTRINSIC_NAME, func_ty, vec![])?;
                    None
                }
                HelperAction::InlineAsmConvergent => {
                    let void_ty = llvm_types::VoidType::get(ctx);
                    inline_asm_convergent(
                        ctx,
                        rewriter,
                        op,
                        void_ty.into(),
                        vec![],
                        "bar.sync 0;",
                        "~{memory}",
                    );
                    None
                }
                HelperAction::InlineAsmSideEffect => {
                    let void_ty = llvm_types::VoidType::get(ctx);
                    let dst = block.deref(ctx).get_argument(0);
                    let value = block.deref(ctx).get_argument(1);
                    inline_asm_sideeffect(
                        ctx,
                        rewriter,
                        op,
                        void_ty.into(),
                        vec![dst, value],
                        "st.global.u32 [$0], $1;",
                        "l,r,~{memory}",
                    );
                    None
                }
                HelperAction::TruncToI1 => {
                    let i32_val = block.deref(ctx).get_argument(0);
                    Some(trunc_to_i1(ctx, rewriter, i32_val))
                }
            };

            rewriter.erase_operation(ctx, op);
            Ok(())
        }
    }

    fn make_ctx() -> Context {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        crate::register(&mut ctx);
        ctx
    }

    fn build_test_func(
        ctx: &mut Context,
        arg_tys: Vec<TypeHandle>,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>) {
        let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
        let module_ptr = module.get_operation();

        let func_ty = FunctionType::get(ctx, arg_tys.clone(), vec![]);
        let func_op_ptr = Operation::new(
            ctx,
            mir::MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = mir::MirFuncOp::new(ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
        func.set_symbol_name(ctx, TEST_FUNC_NAME.try_into().unwrap());

        let region = func.get_operation().deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, arg_tys);
        entry.insert_at_back(region, ctx);

        let module_region = module_ptr.deref(ctx).get_region(0);
        let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
        func.get_operation().insert_at_back(module_block, ctx);

        (module_ptr, entry)
    }

    fn append_trigger(ctx: &mut Context, block: Ptr<BasicBlock>) -> Ptr<Operation> {
        let trigger = Operation::new(
            ctx,
            mir::MirStorageLiveOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        trigger.insert_at_back(block, ctx);
        trigger
    }

    fn run_helper_action(
        ctx: &mut Context,
        module_ptr: Ptr<Operation>,
        action: HelperAction,
    ) -> Option<Value> {
        let mut conversion = HelperConversion {
            action,
            returned_value: None,
        };
        apply_dialect_conversion(ctx, &mut conversion, module_ptr)
            .expect("helper conversion failed");
        conversion.returned_value
    }

    fn module_ops(ctx: &Context, module_ptr: Ptr<Operation>) -> Vec<Ptr<Operation>> {
        let region = module_ptr.deref(ctx).get_region(0);
        let block = region.deref(ctx).iter(ctx).next().unwrap();
        block.deref(ctx).iter(ctx).collect()
    }

    fn body_ops(ctx: &Context, module_ptr: Ptr<Operation>) -> Vec<Ptr<Operation>> {
        for op in module_ops(ctx, module_ptr) {
            let Some(func) = Operation::get_op::<mir::MirFuncOp>(op, ctx) else {
                continue;
            };

            if func.get_symbol_name(ctx).to_string() != TEST_FUNC_NAME {
                continue;
            }

            let func_region = func.get_operation().deref(ctx).get_region(0);
            return func_region
                .deref(ctx)
                .iter(ctx)
                .flat_map(|block| block.deref(ctx).iter(ctx))
                .collect();
        }

        panic!("{TEST_FUNC_NAME} not found");
    }

    fn find_body_ops<T: Op>(ctx: &Context, module_ptr: Ptr<Operation>) -> Vec<T> {
        body_ops(ctx, module_ptr)
            .into_iter()
            .filter_map(|op| Operation::get_op::<T>(op, ctx))
            .collect()
    }

    fn integer_constant_signature(
        ctx: &Context,
        constant: &llvm::ConstantOp,
    ) -> (Signedness, u32, u64) {
        let result_ty = constant
            .get_operation()
            .deref(ctx)
            .get_result(0)
            .get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let result_integer_ty = result_ty_ref
            .downcast_ref::<IntegerType>()
            .expect("constant result must have an integer type");
        let value_attr = constant.get_value(ctx);
        let integer_attr = (&*value_attr as &dyn pliron::attribute::Attribute)
            .downcast_ref::<IntegerAttr>()
            .expect("constant value must be an integer attribute");
        let attr_ty: TypeHandle = integer_attr.get_type().into();
        let value = integer_attr.value();

        assert_eq!(result_ty, attr_ty, "result and attribute types must match");
        assert_eq!(value.bw(), result_integer_ty.width() as usize);

        (
            result_integer_ty.signedness(),
            result_integer_ty.width(),
            value.to_u64(),
        )
    }

    fn integer_width(ctx: &Context, ty: TypeHandle) -> u32 {
        ty.deref(ctx)
            .downcast_ref::<IntegerType>()
            .expect("expected integer type")
            .width()
    }

    fn pointer_addr_space(ctx: &Context, ty: TypeHandle) -> u32 {
        ty.deref(ctx)
            .downcast_ref::<llvm_types::PointerType>()
            .expect("expected LLVM pointer type")
            .address_space()
    }

    #[test]
    fn create_integer_constants_emit_expected_types_and_bit_patterns() {
        let mut ctx = make_ctx();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![]);
        append_trigger(&mut ctx, entry);

        assert!(run_helper_action(&mut ctx, module_ptr, HelperAction::Constants).is_none());

        let constants: Vec<(Signedness, u32, u64)> =
            find_body_ops::<llvm::ConstantOp>(&ctx, module_ptr)
                .into_iter()
                .map(|op| integer_constant_signature(&ctx, &op))
                .collect();

        assert_eq!(
            constants,
            vec![
                (Signedness::Signless, 1, 0),
                (Signedness::Signless, 1, 1),
                (Signedness::Signless, 32, (-7_i32) as u32 as u64),
                (Signedness::Signless, 64, 42),
            ]
        );
    }

    #[test]
    fn cast_to_shared_addr_space_inserts_cast_to_addr_space_3() {
        let mut ctx = make_ctx();
        let generic_ptr_ty: TypeHandle = llvm_types::PointerType::get(&ctx, 0).into();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![generic_ptr_ty]);
        append_trigger(&mut ctx, entry);

        let returned = run_helper_action(&mut ctx, module_ptr, HelperAction::SharedCast)
            .expect("shared cast helper must return a value");

        let casts = find_body_ops::<llvm::AddrSpaceCastOp>(&ctx, module_ptr);
        assert_eq!(casts.len(), 1);

        let result_ty = casts[0]
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        assert_eq!(returned, casts[0].get_operation().deref(&ctx).get_result(0));
        assert_eq!(pointer_addr_space(&ctx, result_ty), 3);
    }

    #[test]
    fn cast_to_shared_addr_space_skips_cast_when_already_shared() {
        let mut ctx = make_ctx();
        let shared_ptr_ty: TypeHandle = llvm_types::PointerType::get(&ctx, 3).into();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![shared_ptr_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        append_trigger(&mut ctx, entry);

        let returned = run_helper_action(&mut ctx, module_ptr, HelperAction::SharedCast)
            .expect("shared cast helper must return a value");

        let casts = find_body_ops::<llvm::AddrSpaceCastOp>(&ctx, module_ptr);
        assert!(casts.is_empty());
        assert_eq!(returned, input);
    }

    #[test]
    fn cast_to_cluster_shared_addr_space_inserts_cast_to_addr_space_7() {
        let mut ctx = make_ctx();
        let generic_ptr_ty: TypeHandle = llvm_types::PointerType::get(&ctx, 0).into();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![generic_ptr_ty]);
        append_trigger(&mut ctx, entry);

        let returned = run_helper_action(&mut ctx, module_ptr, HelperAction::ClusterSharedCast)
            .expect("cluster-shared cast helper must return a value");

        let casts = find_body_ops::<llvm::AddrSpaceCastOp>(&ctx, module_ptr);
        assert_eq!(casts.len(), 1);

        let result_ty = casts[0]
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        assert_eq!(returned, casts[0].get_operation().deref(&ctx).get_result(0));
        assert_eq!(pointer_addr_space(&ctx, result_ty), 7);
    }

    #[test]
    fn cast_to_cluster_shared_addr_space_skips_cast_when_already_cluster_shared() {
        let mut ctx = make_ctx();
        let cluster_ptr_ty: TypeHandle = llvm_types::PointerType::get(&ctx, 7).into();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![cluster_ptr_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        append_trigger(&mut ctx, entry);

        let returned = run_helper_action(&mut ctx, module_ptr, HelperAction::ClusterSharedCast)
            .expect("cluster-shared cast helper must return a value");

        let casts = find_body_ops::<llvm::AddrSpaceCastOp>(&ctx, module_ptr);
        assert!(casts.is_empty());
        assert_eq!(returned, input);
    }

    #[test]
    fn call_intrinsic_declares_and_calls_direct_intrinsic() {
        let mut ctx = make_ctx();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![]);
        append_trigger(&mut ctx, entry);

        assert!(run_helper_action(&mut ctx, module_ptr, HelperAction::CallIntrinsic).is_none());

        let found_decl = module_ops(&ctx, module_ptr)
            .into_iter()
            .filter_map(|op| Operation::get_op::<llvm::FuncOp>(op, &ctx))
            .any(|func| func.get_symbol_name(&ctx).to_string() == TEST_INTRINSIC_NAME);

        let found_call = find_body_ops::<llvm::CallOp>(&ctx, module_ptr)
            .into_iter()
            .any(|call| {
                if let CallOpCallable::Direct(sym) = call.callee(&ctx) {
                    sym.to_string() == TEST_INTRINSIC_NAME
                } else {
                    false
                }
            });

        assert!(found_decl, "expected intrinsic declaration");
        assert!(found_call, "expected direct intrinsic call");
    }

    #[test]
    fn inline_asm_convergent_sets_template_constraints_and_convergent_attr() {
        let mut ctx = make_ctx();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![]);
        append_trigger(&mut ctx, entry);

        assert!(
            run_helper_action(&mut ctx, module_ptr, HelperAction::InlineAsmConvergent).is_none()
        );

        let asms = find_body_ops::<llvm::InlineAsmOp>(&ctx, module_ptr);
        assert_eq!(asms.len(), 1);

        let asm = &asms[0];
        assert_eq!(
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|s| String::from((*s).clone()))
                .as_deref(),
            Some("bar.sync 0;")
        );
        assert_eq!(
            asm.get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|s| String::from((*s).clone()))
                .as_deref(),
            Some("~{memory}")
        );
        assert_eq!(llvm::asm_kind_opt(&ctx, asm), Some(AsmKind::Convergent));
        assert!(
            asm.get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent"))
        );
    }

    #[test]
    fn inline_asm_convergent_copies_the_source_op_location() {
        use pliron::location::{Located, Location};

        let mut ctx = make_ctx();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![]);
        let trigger = append_trigger(&mut ctx, entry);
        let expected = Location::Named {
            name: "source-intrinsic".to_string(),
            child_loc: Box::new(Location::Unknown),
        };
        trigger.deref_mut(&ctx).set_loc(expected.clone());

        assert!(
            run_helper_action(&mut ctx, module_ptr, HelperAction::InlineAsmConvergent).is_none()
        );

        let asms = find_body_ops::<llvm::InlineAsmOp>(&ctx, module_ptr);
        assert_eq!(asms.len(), 1);
        assert_eq!(asms[0].get_operation().deref(&ctx).loc(), expected);
    }

    #[test]
    fn inline_asm_sideeffect_sets_template_constraints_and_sideeffect_kind() {
        let mut ctx = make_ctx();
        let global_ptr_ty: TypeHandle = llvm_types::PointerType::get(&ctx, 1).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![global_ptr_ty, i32_ty]);
        let dst = entry.deref(&ctx).get_argument(0);
        let value = entry.deref(&ctx).get_argument(1);
        append_trigger(&mut ctx, entry);

        assert!(
            run_helper_action(&mut ctx, module_ptr, HelperAction::InlineAsmSideEffect).is_none()
        );

        let asms = find_body_ops::<llvm::InlineAsmOp>(&ctx, module_ptr);
        assert_eq!(asms.len(), 1);

        let asm = &asms[0];
        assert_eq!(
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|s| String::from((*s).clone()))
                .as_deref(),
            Some("st.global.u32 [$0], $1;")
        );
        assert_eq!(
            asm.get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|s| String::from((*s).clone()))
                .as_deref(),
            Some("l,r,~{memory}")
        );
        assert_eq!(
            asm.get_operation()
                .deref(&ctx)
                .operands()
                .collect::<Vec<_>>(),
            vec![dst, value]
        );
        assert_eq!(llvm::asm_kind_opt(&ctx, asm), Some(AsmKind::SideEffect));
        assert!(
            !asm.get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent"))
        );
    }

    #[test]
    fn trunc_to_i1_emits_i1_trunc_result() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_test_func(&mut ctx, vec![i32_ty]);
        append_trigger(&mut ctx, entry);

        let returned = run_helper_action(&mut ctx, module_ptr, HelperAction::TruncToI1)
            .expect("truncation helper must return a value");

        let trunks = find_body_ops::<llvm::TruncOp>(&ctx, module_ptr);
        assert_eq!(trunks.len(), 1);

        let result_ty = trunks[0]
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        assert_eq!(
            returned,
            trunks[0].get_operation().deref(&ctx).get_result(0)
        );
        assert_eq!(integer_width(&ctx, result_ty), 1);
    }
}
