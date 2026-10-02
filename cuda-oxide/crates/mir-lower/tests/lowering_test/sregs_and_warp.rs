/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{
    CallOpCallable, CallOpInterface, OneRegionInterface, SymbolOpInterface,
};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

#[test]
fn test_standalone_lowering_rejects_builtin_pointer_constant() {
    use dialect_mir::types::{MirPointerKind, MirPtrType};
    use pliron::builtin::{
        attributes::IntegerAttr,
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    };
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "pointer_constant".try_into().unwrap());
    let block = module
        .get_region(&ctx)
        .deref(&ctx)
        .iter(&ctx)
        .next()
        .unwrap();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, u32_ty.into(), true, MirPointerKind::UniqueRef);
    let value = APInt::from_u64(0, NonZeroUsize::new(32).unwrap());
    let constant = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(u32_ty, value)));
    let result = constant.get_operation().deref(&ctx).get_result(0);
    result.set_type(&ctx, pointer_ty.into());
    constant.get_operation().insert_at_back(block, &ctx);

    let error = mir_lower::lower_mir_to_llvm(&mut ctx, module.get_operation()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("builtin.constant cannot produce a MIR pointer carrier")
    );
}

#[test]
fn test_intrinsic_insertion() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    // Create Module
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    // Create MirFunc
    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, vec![], vec![]);

    // Manual construction of MirFuncOp
    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1, // 1 region
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    // Add body - MirFuncOp has 1 region
    let region = func.get_operation().deref(&ctx).get_region(0);

    // Create block if empty (it is empty by default from Operation::new)
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    // Add ReadPtxSregTidXOp
    let int32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );

    let tid_op_ptr = Operation::new(
        &mut ctx,
        nvvm::ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![int32_ty.into()],
        vec![],
        vec![],
        0,
    );
    let tid_op = nvvm::ReadPtxSregTidXOp::new(tid_op_ptr);
    let expected_location = Location::Named {
        name: "source-tid-x".to_string(),
        child_loc: Box::new(Location::Unknown),
    };
    tid_op
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(expected_location.clone());
    tid_op.get_operation().insert_at_back(block, &ctx);

    // Add Return
    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    // Add Func to Module
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    // Run DialectConversion-based lowering
    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Verify result
    let mut found_intrinsic = false;
    let mut found_intrinsic_call = false;
    let mut found_kernel = false;

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        if let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) {
            let name = func_op.get_symbol_name(&ctx).to_string();
            if name == "llvm_nvvm_read_ptx_sreg_tid_x" {
                found_intrinsic = true;
                // Intrinsic (declaration) should have 0 regions or empty region
                let num_regions = func_op.get_operation().deref(&ctx).regions().count();
                if num_regions > 0 {
                    assert!(
                        func_op
                            .get_operation()
                            .deref(&ctx)
                            .get_region(0)
                            .deref(&ctx)
                            .iter(&ctx)
                            .next()
                            .is_none()
                    );
                }
            } else if name == "kernel_func" {
                found_kernel = true;
                // Kernel should have body (1 region, not empty)
                assert!(func_op.get_operation().deref(&ctx).regions().count() > 0);
                assert!(
                    func_op
                        .get_operation()
                        .deref(&ctx)
                        .get_region(0)
                        .deref(&ctx)
                        .iter(&ctx)
                        .next()
                        .is_some()
                );
                let kernel_region = func_op.get_operation().deref(&ctx).get_region(0);
                for kernel_block in kernel_region.deref(&ctx).iter(&ctx) {
                    for body_op in kernel_block.deref(&ctx).iter(&ctx) {
                        let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx) else {
                            continue;
                        };
                        if matches!(
                            call.callee(&ctx),
                            CallOpCallable::Direct(symbol)
                                if symbol.to_string() == "llvm_nvvm_read_ptx_sreg_tid_x"
                        ) {
                            found_intrinsic_call = true;
                            assert_eq!(call.get_operation().deref(&ctx).loc(), expected_location);
                        }
                    }
                }
            }
        }
    }

    assert!(found_intrinsic, "Intrinsic function declaration not found");
    assert!(found_intrinsic_call, "Intrinsic call not found in kernel");
    assert!(found_kernel, "Kernel function not found");

    Ok(())
}

#[test]
fn test_assertfail_orphaned_successor_block_is_removed() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let u8_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        8,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let ptr_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let u32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let u64_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        64,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let arg_tys: Vec<pliron::r#type::TypeHandle> = vec![
        ptr_ty.into(),
        ptr_ty.into(),
        u32_ty.into(),
        ptr_ty.into(),
        u64_ty.into(),
    ];
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, arg_tys.clone(), vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let entry = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, arg_tys);
        b.insert_at_back(region, &ctx);
        b
    };
    // A block reachable only through the assert's success edge, carrying a
    // block argument: the CFG a constant-false `gpu_assert!` leaves behind
    // after MIR optimization folds the direct edge away.
    let tail = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![u32_ty.into()]);
        b.insert_at_back(region, &ctx);
        b
    };

    let message = entry.deref(&ctx).get_argument(0);
    let file = entry.deref(&ctx).get_argument(1);
    let line = entry.deref(&ctx).get_argument(2);
    let function = entry.deref(&ctx).get_argument(3);
    let char_size = entry.deref(&ctx).get_argument(4);

    let assertfail_op =
        nvvm::AssertFailOp::build(&mut ctx, message, file, line, function, char_size);
    assertfail_op.insert_at_back(entry, &ctx);

    // The success edge forwards `line` into the tail block's argument.
    // Lowering erases this branch (the call never returns), orphaning the
    // tail block, which then cannot be exported: a block argument needs one
    // PHI incoming value per predecessor and there are none left.
    let goto_op = Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![line],
        vec![tail],
        0,
    );
    goto_op.insert_at_back(entry, &ctx);

    let ret_op = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op.insert_at_back(tail, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut found_kernel = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != func_name {
            continue;
        }
        found_kernel = true;
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        let blocks: Vec<_> = func_region.deref(&ctx).iter(&ctx).collect();
        for block in blocks.iter().skip(1) {
            assert!(
                !block.preds(&ctx).is_empty(),
                "an unreachable block survived lowering; with block arguments \
                 it cannot be exported (a PHI needs an incoming value per \
                 predecessor and there are none)"
            );
        }
    }
    assert!(found_kernel, "Kernel function not found");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir =
        llvm_export::export::export_module_to_string(&ctx, &module).map_err(anyhow::Error::msg)?;
    let lines: Vec<_> = ir.lines().collect();
    let call_line = lines
        .iter()
        .position(|line| line.contains("call void @__assertfail("))
        .expect("expected exported __assertfail call");
    assert_eq!(lines[call_line + 1].trim(), "unreachable", "{ir}");

    Ok(())
}

#[test]
fn test_globaltimer_lowers_to_intrinsic_call() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, vec![], vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    let i64_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        64,
        pliron::builtin::types::Signedness::Signless,
    );
    let timer_op = Operation::new(
        &mut ctx,
        nvvm::ReadPtxSregGlobaltimerOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    timer_op.insert_at_back(block, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    const INTRINSIC: &str = "llvm_nvvm_read_ptx_sreg_globaltimer";

    let mut found_decl = false;
    let mut found_call = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) else {
            continue;
        };
        let name = func_op.get_symbol_name(&ctx).to_string();

        if name == INTRINSIC {
            // Intrinsic declaration: present with empty body.
            found_decl = true;
            let num_regions = func_op.get_operation().deref(&ctx).regions().count();
            if num_regions > 0 {
                assert!(
                    func_op
                        .get_operation()
                        .deref(&ctx)
                        .get_region(0)
                        .deref(&ctx)
                        .iter(&ctx)
                        .next()
                        .is_none(),
                    "intrinsic declaration must have empty body"
                );
            }
        } else if name == func_name {
            let func_region = func_op.get_operation().deref(&ctx).get_region(0);
            for func_block in func_region.deref(&ctx).iter(&ctx) {
                for body_op in func_block.deref(&ctx).iter(&ctx) {
                    if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                        && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                        && sym.to_string() == INTRINSIC
                    {
                        found_call = true;
                    }
                    assert!(
                        Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx).is_none(),
                        "globaltimer must not lower to inline asm"
                    );
                }
            }
        }
    }

    assert!(
        found_decl,
        "Expected `{INTRINSIC}` declaration in lowered module"
    );
    assert!(
        found_call,
        "Expected call to `{INTRINSIC}` in lowered kernel body"
    );
    Ok(())
}

#[test]
fn test_assertfail_lowers_to_noreturn_call_and_unreachable() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let u8_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        8,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let ptr_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let u32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let u64_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        64,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let arg_tys: Vec<pliron::r#type::TypeHandle> = vec![
        ptr_ty.into(),
        ptr_ty.into(),
        u32_ty.into(),
        ptr_ty.into(),
        u64_ty.into(),
    ];
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, arg_tys.clone(), vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, arg_tys);
        b.insert_at_back(region, &ctx);
        b
    };

    let message = block.deref(&ctx).get_argument(0);
    let file = block.deref(&ctx).get_argument(1);
    let line = block.deref(&ctx).get_argument(2);
    let function = block.deref(&ctx).get_argument(3);
    let char_size = block.deref(&ctx).get_argument(4);

    let assertfail_op =
        nvvm::AssertFailOp::build(&mut ctx, message, file, line, function, char_size);
    assertfail_op.insert_at_back(block, &ctx);

    // This return must be removed by lowering because __assertfail never returns.
    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op_ptr.insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    const EXTERN: &str = "__assertfail";

    let mut found_decl = false;
    let mut found_call = false;
    let mut found_unreachable = false;

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        let name = func_op.get_symbol_name(&ctx).to_string();

        if name == EXTERN {
            found_decl = true;
            assert!(
                func_op.get_attr_llvm_func_attrs(&ctx).is_some_and(|attrs| {
                    matches!(attrs.get("noreturn"), Some(llvm::LlvmAttrValue::Unit))
                }),
                "__assertfail declaration must be marked noreturn"
            );
            continue;
        }

        if name != func_name {
            continue;
        }

        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            let body_ops: Vec<_> = func_block.deref(&ctx).iter(&ctx).collect();

            for (index, body_op) in body_ops.iter().copied().enumerate() {
                if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                    && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                    && sym.to_string() == EXTERN
                {
                    found_call = true;

                    assert!(
                        call.get_attr_llvm_call_attrs(&ctx).is_some_and(|attrs| {
                            matches!(attrs.get("noreturn"), Some(llvm::LlvmAttrValue::Unit))
                        }),
                        "__assertfail call must be marked noreturn"
                    );
                    assert_eq!(
                        body_op.deref(&ctx).get_num_operands(),
                        5,
                        "__assertfail call must forward all five operands"
                    );
                    assert!(
                        body_ops.get(index + 1).is_some_and(|next| {
                            Operation::get_op::<llvm::UnreachableOp>(*next, &ctx).is_some()
                        }),
                        "llvm.unreachable must immediately follow __assertfail"
                    );
                    assert_eq!(
                        index + 2,
                        body_ops.len(),
                        "no operations may remain after llvm.unreachable"
                    );
                }

                if Operation::get_op::<llvm::UnreachableOp>(body_op, &ctx).is_some() {
                    found_unreachable = true;
                }

                assert!(
                    Operation::get_op::<nvvm::AssertFailOp>(body_op, &ctx).is_none(),
                    "nvvm.assertfail must be fully consumed by the lowering"
                );
            }
        }
    }

    assert!(
        found_decl,
        "Expected `{EXTERN}` declaration in lowered module"
    );
    assert!(
        found_call,
        "Expected call to `{EXTERN}` in lowered kernel body"
    );
    assert!(
        found_unreachable,
        "__assertfail must terminate the block with llvm.unreachable"
    );

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir =
        llvm_export::export::export_module_to_string(&ctx, &module).map_err(anyhow::Error::msg)?;

    let declaration = ir
        .lines()
        .find(|line| line.contains("declare void @__assertfail("))
        .expect("expected exported __assertfail declaration");
    assert!(declaration.contains("noreturn"), "{ir}");

    let lines: Vec<_> = ir.lines().collect();
    let call_line = lines
        .iter()
        .position(|line| line.contains("call void @__assertfail("))
        .expect("expected exported __assertfail call");
    assert!(lines[call_line].contains("noreturn"), "{ir}");
    assert_eq!(lines[call_line + 1].trim(), "unreachable", "{ir}");

    Ok(())
}

/// Lower a single zero-operand, i32-result special-register op and assert it
/// emits a declaration of and direct call to `intrinsic` (and no inline asm).
fn assert_sreg_i32_lowers_to_intrinsic(
    op_info: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    intrinsic: &str,
) -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, vec![], vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let sreg_op = Operation::new(&mut ctx, op_info, vec![i32_ty.into()], vec![], vec![], 0);
    sreg_op.insert_at_back(block, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_decl = false;
    let mut found_call = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) else {
            continue;
        };
        let name = func_op.get_symbol_name(&ctx).to_string();

        if name == intrinsic {
            found_decl = true;
        } else if name == func_name {
            let func_region = func_op.get_operation().deref(&ctx).get_region(0);
            for func_block in func_region.deref(&ctx).iter(&ctx) {
                for body_op in func_block.deref(&ctx).iter(&ctx) {
                    if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                        && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                        && sym.to_string() == intrinsic
                    {
                        found_call = true;
                    }
                    assert!(
                        Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx).is_none(),
                        "{intrinsic} must not lower to inline asm"
                    );
                }
            }
        }
    }

    assert!(
        found_decl,
        "Expected `{intrinsic}` declaration in lowered module"
    );
    assert!(
        found_call,
        "Expected call to `{intrinsic}` in lowered kernel body"
    );
    Ok(())
}

fn assert_sreg_lowers_to_inline_asm(
    op_info: (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    result_width: u32,
    expected_template: &str,
    expected_constraints: &str,
    expected_kind: llvm::AsmKind,
) -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let result_ty = IntegerType::get(&ctx, result_width, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    let sreg_op = Operation::new(&mut ctx, op_info, vec![result_ty.into()], vec![], vec![], 0);
    sreg_op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut matches = 0usize;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                if template.as_deref() != Some(expected_template) {
                    continue;
                }

                matches += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some(expected_constraints)
                );
                assert_eq!(llvm::asm_kind(&ctx, &inline_asm), expected_kind);
            }
        }
    }

    assert_eq!(matches, 1, "expected one exact `{expected_template}` read");
    Ok(())
}

#[test]
fn test_lanemask_ops_lower_to_sreg_intrinsic_calls() -> Result<(), anyhow::Error> {
    // Each lane-position mask op lowers to its matching read-only sreg intrinsic
    // (underscores become dots on export: `..._lanemask_lt` -> `...lanemask.lt`).
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskLtOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_lt",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskLeOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_le",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskEqOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_eq",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskGeOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_ge",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskGtOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_gt",
    )?;
    Ok(())
}

#[test]
fn test_generated_vote_sync_family_lowers_to_exact_typed_intrinsics() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::Typed;

    let mut ctx = make_test_ctx();
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into(), i1_ty.into()]);
    let mask = entry.deref(&ctx).get_argument(0);
    let predicate = entry.deref(&ctx).get_argument(1);

    for vote in [
        nvvm::VoteSyncAllOp::build(&mut ctx, mask, predicate),
        nvvm::VoteSyncAnyOp::build(&mut ctx, mask, predicate),
        nvvm::VoteSyncBallotOp::build(&mut ctx, mask, predicate),
        nvvm::VoteSyncUniOp::build(&mut ctx, mask, predicate),
    ] {
        vote.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        ("llvm_nvvm_vote_all_sync", 1),
        ("llvm_nvvm_vote_any_sync", 1),
        ("llvm_nvvm_vote_ballot_sync", 32),
        ("llvm_nvvm_vote_uni_sync", 1),
    ];
    let mut found = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "generated vote.sync operations must use typed LLVM intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        let Some((_, result_width)) = expected.iter().find(|(name, _)| *name == callee) else {
            continue;
        };

        let call = call.get_operation().deref(&ctx);
        assert_eq!(call.get_num_operands(), 2);
        assert_eq!(call.get_num_results(), 1);

        let integer_shape = |value: pliron::value::Value| {
            let ty = value.get_type(&ctx);
            let ty = ty.deref(&ctx);
            let integer = ty
                .downcast_ref::<IntegerType>()
                .expect("vote.sync operands and results are integers");
            (integer.width(), integer.signedness())
        };
        assert_eq!(
            [
                integer_shape(call.get_operand(0)),
                integer_shape(call.get_operand(1)),
            ],
            [(32, Signedness::Signless), (1, Signedness::Signless),],
            "{callee} must preserve [mask, predicate] operand order"
        );
        assert_eq!(
            integer_shape(call.get_result(0)),
            (*result_width, Signedness::Signless),
            "{callee} returned the wrong LLVM integer type"
        );
        found.push((callee, *result_width));
    }

    found.sort();
    let mut expected = expected
        .map(|(name, width)| (name.to_owned(), width))
        .to_vec();
    expected.sort();
    assert_eq!(found, expected);
    Ok(())
}

fn lower_generated_active_mask(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    nvvm::ActiveMaskOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_generated_active_mask_llvm_nvptx_uses_typed_intrinsic() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_generated_active_mask(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut found = 0;
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX active_mask must use the typed intrinsic"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        if callee.to_string() != "llvm_nvvm_activemask" {
            continue;
        }

        found += 1;
        let call = op.deref(&ctx);
        assert_eq!(call.get_num_operands(), 0);
        assert_eq!(call.get_num_results(), 1);
        let result_ty = call.get_result(0).get_type(&ctx);
        let result_ty = result_ty.deref(&ctx);
        let result_ty = result_ty
            .downcast_ref::<IntegerType>()
            .expect("active_mask returns an integer");
        assert_eq!(result_ty.width(), 32);
    }
    assert_eq!(found, 1, "expected one typed active_mask call");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(ir.contains("call i32 @llvm.nvvm.activemask()"), "{ir}");
    Ok(())
}

#[test]
fn test_generated_active_mask_libnvvm_uses_convergent_sideeffect_asm() -> Result<(), anyhow::Error>
{
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_generated_active_mask(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut found = 0;
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert_ne!(callee.to_string(), "llvm_nvvm_activemask");
        }
        let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };

        found += 1;
        assert_eq!(
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("activemask.b32 $0;")
        );
        assert_eq!(
            asm.get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r,~{memory}")
        );
        assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::Convergent);
        assert!(
            asm.get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent"))
        );
        let asm = op.deref(&ctx);
        assert_eq!(asm.get_num_operands(), 0);
        assert_eq!(asm.get_num_results(), 1);
        let result_ty = asm.get_result(0).get_type(&ctx);
        let result_ty = result_ty.deref(&ctx);
        let result_ty = result_ty
            .downcast_ref::<IntegerType>()
            .expect("active_mask inline asm returns an integer");
        assert_eq!(result_ty.width(), 32);
    }
    assert_eq!(found, 1, "expected one exact active_mask asm block");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(
        ir.contains("call i32 asm sideeffect \"activemask.b32 $0;\", \"=r,~{memory}\"()"),
        "{ir}"
    );
    assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    Ok(())
}

#[test]
fn test_generated_warp_match_family_uses_exact_typed_calls_and_mask_projection()
-> Result<(), anyhow::Error> {
    use llvm_export::types::StructType;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::Typed;

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i64_ty.into()]);
    let mask = entry.deref(&ctx).get_argument(0);
    let value32 = entry.deref(&ctx).get_argument(1);
    let value64 = entry.deref(&ctx).get_argument(2);
    for warp_match in [
        nvvm::MatchAnySyncI32Op::build(&mut ctx, mask, value32),
        nvvm::MatchAnySyncI64Op::build(&mut ctx, mask, value64),
        nvvm::MatchAllSyncI32Op::build(&mut ctx, mask, value32),
        nvvm::MatchAllSyncI64Op::build(&mut ctx, mask, value64),
    ] {
        warp_match.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LlvmNvptx,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        ("llvm_nvvm_match_any_sync_i32", 32, false),
        ("llvm_nvvm_match_any_sync_i64", 64, false),
        ("llvm_nvvm_match_all_sync_i32p", 32, true),
        ("llvm_nvvm_match_all_sync_i64p", 64, true),
    ];
    let body = lowered_kernel_body(&ctx, module_ptr);
    let integer_width = |value: pliron::value::Value| {
        let ty = value.get_type(&ctx);
        let ty = ty.deref(&ctx);
        ty.downcast_ref::<IntegerType>()
            .expect("warp-match value is an integer")
            .width()
    };
    let mut found = Vec::new();
    let mut aggregate_results = Vec::new();
    for &op in &body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "warp match must use typed LLVM intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        let Some((_, value_width, aggregate)) =
            expected.iter().find(|(name, _, _)| *name == callee)
        else {
            continue;
        };

        let call = op.deref(&ctx);
        assert_eq!(call.get_num_operands(), 2);
        assert_eq!(
            [
                integer_width(call.get_operand(0)),
                integer_width(call.get_operand(1)),
            ],
            [32, *value_width],
            "{callee} has the wrong typed signature"
        );
        assert_eq!(call.get_num_results(), 1);
        let result = call.get_result(0);
        if *aggregate {
            let result_ty = result.get_type(&ctx);
            let result_ty = result_ty.deref(&ctx);
            let result_ty = result_ty
                .downcast_ref::<StructType>()
                .expect("match.all returns an LLVM aggregate");
            assert_eq!(result_ty.num_fields(), 2);
            let field_widths = (0..2)
                .map(|index| {
                    let field = result_ty.field_type(index);
                    let field = field.deref(&ctx);
                    field
                        .downcast_ref::<IntegerType>()
                        .expect("match.all aggregate fields are integers")
                        .width()
                })
                .collect::<Vec<_>>();
            assert_eq!(field_widths, [32, 1]);
            aggregate_results.push((callee.clone(), result));
        } else {
            assert_eq!(integer_width(result), 32);
        }
        found.push(callee);
    }

    found.sort();
    let mut expected_calls = expected.map(|(name, _, _)| name.to_owned());
    expected_calls.sort();
    assert_eq!(found, expected_calls);

    let mut projected = Vec::new();
    for &op in &body {
        let Some(extract) = Operation::get_op::<llvm::ExtractValueOp>(op, &ctx) else {
            continue;
        };
        assert_eq!(extract.indices(&ctx), vec![0]);
        let extract = op.deref(&ctx);
        assert_eq!(extract.get_num_operands(), 1);
        assert_eq!(extract.get_num_results(), 1);
        assert_eq!(integer_width(extract.get_result(0)), 32);
        let aggregate = extract.get_operand(0);
        let callee = aggregate_results
            .iter()
            .find_map(|(callee, result)| (*result == aggregate).then(|| callee.clone()))
            .expect("match.all must extract from its aggregate call result");
        projected.push(callee);
    }
    projected.sort();
    assert_eq!(
        projected,
        [
            "llvm_nvvm_match_all_sync_i32p".to_owned(),
            "llvm_nvvm_match_all_sync_i64p".to_owned(),
        ]
    );
    Ok(())
}

#[test]
fn test_warpid_ops_preserve_snapshot_semantics() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregWarpIdOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %warpid;",
        "=r",
        llvm::AsmKind::SideEffect,
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregNwarpIdOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_nwarpid",
    )?;
    Ok(())
}

#[test]
fn test_smid_ops_preserve_snapshot_semantics() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregSmIdOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %smid;",
        "=r",
        llvm::AsmKind::SideEffect,
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregNsmIdOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_nsmid",
    )?;
    Ok(())
}

#[test]
fn test_repeated_location_samples_remain_side_effecting_reads() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);

    for op_info in [
        nvvm::ReadPtxSregWarpIdOp::get_concrete_op_info(),
        nvvm::ReadPtxSregSmIdOp::get_concrete_op_info(),
        nvvm::ReadPtxSregWarpIdOp::get_concrete_op_info(),
        nvvm::ReadPtxSregSmIdOp::get_concrete_op_info(),
    ] {
        let op = Operation::new(&mut ctx, op_info, vec![i32_ty.into()], vec![], vec![], 0);
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut warpid_reads = 0usize;
    let mut smid_reads = 0usize;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                match template.as_deref() {
                    Some("mov.u32 $0, %warpid;") => warpid_reads += 1,
                    Some("mov.u32 $0, %smid;") => smid_reads += 1,
                    _ => continue,
                }
                assert_eq!(
                    llvm::asm_kind(&ctx, &inline_asm),
                    llvm::AsmKind::SideEffect,
                    "location snapshots must survive LLVM CSE"
                );
            }
        }
    }

    assert_eq!(warpid_reads, 2);
    assert_eq!(smid_reads, 2);
    Ok(())
}

#[test]
fn test_gridid_op_lowers_to_full_width_inline_ptx() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregGridIdOp::get_concrete_op_info(),
        64,
        "mov.u64 $0, %gridid;",
        "=l",
        llvm::AsmKind::Pure,
    )
}

#[test]
fn test_smem_size_ops_lower_to_portable_inline_ptx() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregDynamicSmemSizeOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %dynamic_smem_size;",
        "=r",
        llvm::AsmKind::Pure,
    )?;
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregTotalSmemSizeOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %total_smem_size;",
        "=r",
        llvm::AsmKind::Pure,
    )?;
    Ok(())
}

#[test]
fn generated_threadfences_use_typed_intrinsics_on_both_backends() -> Result<(), anyhow::Error> {
    const EXPECTED: [&str; 3] = [
        "llvm_nvvm_membar_cta",
        "llvm_nvvm_membar_gl",
        "llvm_nvvm_membar_sys",
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
        for op_info in [
            nvvm::ThreadfenceBlockOp::get_concrete_op_info(),
            nvvm::ThreadfenceOp::get_concrete_op_info(),
            nvvm::ThreadfenceSystemOp::get_concrete_op_info(),
        ] {
            Operation::new(&mut ctx, op_info, vec![], vec![], vec![], 0)
                .insert_at_back(entry, &ctx);
        }
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut found = Vec::new();
        for op in lowered_kernel_body(&ctx, module_ptr) {
            assert!(
                Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
                "threadfences must use their reviewed typed route"
            );
            let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                continue;
            };
            let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                continue;
            };
            let callee = callee.to_string();
            if EXPECTED.contains(&callee.as_str()) {
                assert_eq!(op.deref(&ctx).get_num_operands(), 0);
                found.push(callee);
            }
        }
        found.sort();
        assert_eq!(found, EXPECTED);

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        for symbol in [
            "llvm.nvvm.membar.cta",
            "llvm.nvvm.membar.gl",
            "llvm.nvvm.membar.sys",
        ] {
            assert!(ir.contains(&format!("call void @{symbol}()")), "{ir}");
        }
    }
    Ok(())
}

/// LLVM uses its typed intrinsic. libNVVM uses the reviewed inline-PTX fallback.
#[test]
fn generated_elect_sync_uses_the_selected_backend_route() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
        let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
        let mask = entry.deref(&ctx).get_argument(0);
        Operation::new(
            &mut ctx,
            nvvm::ElectSyncOp::get_concrete_op_info(),
            vec![i32_ty.into(), i1_ty.into()],
            vec![mask],
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut inline_asm_count = 0;
        let mut typed_call_count = 0;
        let mut extract_count = 0;
        let mut trunc_count = 0;
        for body_op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) {
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_template(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some("{ .reg .pred p; elect.sync $0|p, $2; selp.b32 $1, 1, 0, p; }")
                );
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some("=r,=r,r")
                );
                assert!(
                    inline_asm
                        .get_attr_llvm_inline_asm_attrs(&ctx)
                        .is_some_and(|attrs| attrs.has("convergent"))
                );
                inline_asm_count += 1;
            }
            if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
                && callee.to_string() == "llvm_nvvm_elect_sync"
            {
                typed_call_count += 1;
            }
            if Operation::get_op::<llvm::ExtractValueOp>(body_op, &ctx).is_some() {
                extract_count += 1;
            }
            if Operation::get_op::<llvm::TruncOp>(body_op, &ctx).is_some() {
                trunc_count += 1;
            }
        }

        assert_eq!(extract_count, 2);
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert_eq!(typed_call_count, 1);
                assert_eq!(inline_asm_count, 0);
                assert_eq!(trunc_count, 0);
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                assert_eq!(typed_call_count, 0);
                assert_eq!(inline_asm_count, 1);
                assert_eq!(trunc_count, 1);
            }
        }
    }
    Ok(())
}

/// The exact inline-PTX template `convert_shuffle_i64` must emit for `mode`/`clamp`.
/// Mirrors the production `format!` so a drift in either side fails the test.
fn expected_shfl_i64_template(mode: &str, clamp: i32) -> String {
    format!(
        "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; \
         shfl.sync.{mode}.b32 lo, lo, $2, {clamp}, $3; \
         shfl.sync.{mode}.b32 hi, hi, $2, {clamp}, $3; \
         mov.b64 $0, {{lo, hi}}; }}"
    )
}

/// 64-bit warp shuffle has no LLVM intrinsic (`shfl.sync` is 32-bit only), so it
/// lowers to convergent inline PTX that splits the value into two halves and runs
/// two `shfl.sync.*.b32`. Inline asm is opaque to LLVM, so a wrong mnemonic,
/// swapped operand order, wrong clamp, or missing `convergent` would only surface
/// as bad PTX downstream. This pins, for every mode, the exact template (incl. the
/// per-mode clamp: 31 for idx/bfly/down, 0 for up), the `=l,l,r,r` constraints,
/// and the convergent flag.
#[test]
fn test_shuffle_i64_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    // Kernel args: [mask (i32), value (i64), lane/delta (i32)].
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i64_ty.into(), i32_ty.into()]);
    let mask = entry.deref(&ctx).get_argument(0);
    let value = entry.deref(&ctx).get_argument(1);
    let lane = entry.deref(&ctx).get_argument(2);

    // One op per mode, all sharing the same [mask, value, lane] operands.
    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let modes: [(OpInfo, &str, i32); 4] = [
        (nvvm::ShflSyncIdxI64Op::get_concrete_op_info(), "idx", 31),
        (nvvm::ShflSyncBflyI64Op::get_concrete_op_info(), "bfly", 31),
        (nvvm::ShflSyncDownI64Op::get_concrete_op_info(), "down", 31),
        (nvvm::ShflSyncUpI64Op::get_concrete_op_info(), "up", 0),
    ];
    for (opid, _, _) in modes {
        let op = Operation::new(
            &mut ctx,
            opid,
            vec![i64_ty.into()],
            vec![mask, value, lane],
            vec![],
            0,
        );
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Collect every inline-asm template emitted into the kernel body.
    let mut templates: Vec<String> = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .as_deref(),
                    Some("=l,l,r,r"),
                    "shfl.b64 constraints must be [out i64, value i64, lane i32, mask i32]"
                );
                assert!(
                    inline_asm
                        .get_attr_llvm_inline_asm_attrs(&ctx)
                        .is_some_and(|attrs| attrs.has("convergent")),
                    "shfl.b64 inline asm must be convergent"
                );
                templates.push(
                    inline_asm
                        .get_attr_llvm_inline_asm_template(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .unwrap_or_default(),
                );
            }
        }
    }

    assert_eq!(
        templates.len(),
        4,
        "each of the 4 shfl.b64 modes must lower to one inline-asm op"
    );
    for (_, mode, clamp) in modes {
        let want = expected_shfl_i64_template(mode, clamp);
        assert!(
            templates.contains(&want),
            "missing inline PTX for shfl.sync.{mode}.b32 (clamp {clamp}); got {templates:?}"
        );
    }

    Ok(())
}

type ReduxBuild =
    fn(&mut Context, pliron::value::Value, pliron::value::Value) -> pliron::context::Ptr<Operation>;

/// A device function returning a `redux.sync` result directly:
/// `fn reduce(mask: u32, value: T) -> T { redux(mask, value) }`, where `T` is
/// `u32`/`i32` for `Some(signedness)` and `f32` for `None`. Returns the
/// lowered module after LLVM-dialect verification.
fn lower_returned_redux(
    build: ReduxBuild,
    signedness: Option<pliron::builtin::types::Signedness>,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::types::{FP32Type, FunctionType, IntegerType, Signedness};
    use pliron::common_traits::Verify;

    let mut ctx = make_test_ctx();
    let mask_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let value_ty = match signedness {
        Some(signedness) => IntegerType::get(&ctx, 32, signedness).into(),
        None => FP32Type::get(&ctx).into(),
    };
    let module = ModuleOp::new(&mut ctx, "returned_redux".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_block = module_ptr
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .iter(&ctx)
        .next()
        .unwrap();

    let func_ty = FunctionType::get(&ctx, vec![mask_ty, value_ty], vec![value_ty]);
    let func_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(&mut ctx, func_ptr, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(&mut ctx, "reduce".try_into().unwrap());
    let entry = BasicBlock::new(&mut ctx, None, vec![mask_ty, value_ty]);
    entry.insert_at_back(func.get_operation().deref(&ctx).get_region(0), &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let mask = entry.deref(&ctx).get_argument(0);
    let value = entry.deref(&ctx).get_argument(1);
    let redux = build(&mut ctx, mask, value);
    redux.insert_at_back(entry, &ctx);
    let reduced = redux.deref(&ctx).get_result(0);
    Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![reduced],
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{e}"))?;
    module_ptr
        .deref(&ctx)
        .verify(&ctx)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((ctx, module_ptr))
}

/// #811: a redux result returned straight out of a device function must
/// lower with a signless integer type. The dialect op's `ui32`/`si32` result
/// used to leak into the intrinsic declaration, so the lowered `llvm.return`
/// failed verification against the function's `i32` result. The `f32`
/// variants lower to the same type and are controls.
#[test]
fn test_redux_result_returned_from_device_function_verifies() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::builtin::types::Signedness::{self, Signed, Unsigned};
    use pliron::r#type::Typed;

    let cases: [(&str, ReduxBuild, Option<Signedness>); 16] = [
        ("add", nvvm::ReduxSyncAddOp::build, Some(Unsigned)),
        ("and", nvvm::ReduxSyncAndOp::build, Some(Unsigned)),
        ("or", nvvm::ReduxSyncOrOp::build, Some(Unsigned)),
        ("xor", nvvm::ReduxSyncXorOp::build, Some(Unsigned)),
        ("umin", nvvm::ReduxSyncUminOp::build, Some(Unsigned)),
        ("umax", nvvm::ReduxSyncUmaxOp::build, Some(Unsigned)),
        ("min", nvvm::ReduxSyncMinOp::build, Some(Signed)),
        ("max", nvvm::ReduxSyncMaxOp::build, Some(Signed)),
        ("fmin", nvvm::ReduxSyncFminOp::build, None),
        ("fmax", nvvm::ReduxSyncFmaxOp::build, None),
        ("fmin_abs", nvvm::ReduxSyncFminAbsOp::build, None),
        ("fmax_abs", nvvm::ReduxSyncFmaxAbsOp::build, None),
        ("fmin_NaN", nvvm::ReduxSyncFminNanOp::build, None),
        ("fmax_NaN", nvvm::ReduxSyncFmaxNanOp::build, None),
        ("fmin_abs_NaN", nvvm::ReduxSyncFminAbsNanOp::build, None),
        ("fmax_abs_NaN", nvvm::ReduxSyncFmaxAbsNanOp::build, None),
    ];

    for (name, build, signedness) in cases {
        let (ctx, module_ptr) = lower_returned_redux(build, signedness)
            .map_err(|e| anyhow::anyhow!("redux_sync_{name}: {e}"))?;

        let module_block = module_ptr
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let func = module_block
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::FuncOp>(op, &ctx))
            .find(|func| func.get_symbol_name(&ctx).to_string() == "reduce")
            .expect("lowered reduce function");
        let calls: Vec<_> = func
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .iter(&ctx)
            .flat_map(|block| block.deref(&ctx).iter(&ctx).collect::<Vec<_>>())
            .filter_map(|op| Operation::get_op::<llvm::CallOp>(op, &ctx))
            .collect();
        assert_eq!(calls.len(), 1, "redux_sync_{name}");
        let CallOpCallable::Direct(callee) = calls[0].callee(&ctx) else {
            panic!("redux_sync_{name} must remain a direct intrinsic call");
        };
        assert_eq!(callee.to_string(), format!("llvm_nvvm_redux_sync_{name}"));
        let call = calls[0].get_operation().deref(&ctx);
        let entry = call.get_parent_block().unwrap();
        assert_eq!(
            call.operands().collect::<Vec<_>>(),
            vec![
                entry.deref(&ctx).get_argument(1),
                entry.deref(&ctx).get_argument(0)
            ],
            "redux_sync_{name} must pass value before member mask"
        );
        assert_eq!(
            call.get_result(0).get_type(&ctx),
            entry.deref(&ctx).get_argument(1).get_type(&ctx),
            "redux_sync_{name} result must retain the converted value type"
        );
        if signedness.is_some() {
            let result_ty = calls[0]
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx);
            let integer = result_ty
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .map(|integer| (integer.width(), integer.signedness()));
            assert_eq!(
                integer,
                Some((32, Signedness::Signless)),
                "redux_sync_{name} call result"
            );
        }
    }
    Ok(())
}
