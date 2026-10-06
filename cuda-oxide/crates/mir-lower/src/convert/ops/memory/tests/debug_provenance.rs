/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::disallowed_methods)]

use super::*;

#[test]
fn convert_dbg_value_lowers_to_llvm_dbg_value() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();

    let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
    let value = block.deref(&ctx).get_argument(0);

    let dbg_op = mir::MirDbgValueOp::new(&mut ctx, value);
    let dbg_loc = pliron::location::Location::Named {
        name: "current value location".to_string(),
        child_loc: Box::new(pliron::location::Location::Unknown),
    };
    dbg_op
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(dbg_loc.clone());
    llvm::set_debug_local_variable(
        &mut ctx,
        dbg_op.get_operation(),
        llvm::DebugLocalVariableInfo {
            name: "x".to_string(),
            argument_index: None,
            ty: llvm::DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm::set_debug_local_source_scope(&mut ctx, dbg_op.get_operation(), 42);
    llvm::set_debug_fragment_variables(
        &mut ctx,
        dbg_op.get_operation(),
        &[llvm::DebugFragmentVariableInfo {
            variable: llvm::DebugLocalVariableInfo {
                name: "pair".to_string(),
                argument_index: None,
                ty: llvm::DebugLocalTypeKind::Array {
                    name: "[u32; 2]".to_string(),
                    size_bits: 64,
                    element: Box::new(llvm::DebugLocalTypeKind::Basic {
                        name: "u32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_unsigned",
                    }),
                    count: 2,
                },
            },
            fragment: llvm::DebugFragment {
                offset_bits: 32,
                size_bits: 32,
            },
            source_scope: Some(42),
            declaration: Some(llvm::DebugSourcePosition {
                file: PathBuf::from("decl.rs"),
                line: 7,
                column: 3,
            }),
        }],
    );
    llvm::set_debug_local_declaration_location(
        &mut ctx,
        dbg_op.get_operation(),
        PathBuf::from("decl.rs"),
        7,
        3,
    );
    dbg_op.get_operation().insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(count_ops::<mir::MirDbgValueOp>(&ctx, &body), 0);
    let dbg_value = find_first::<llvm::DebugValueOp>(&ctx, &body)
        .expect("expected lowered llvm.dbg_value marker");
    assert_eq!(
        dbg_value.get_operation().deref(&ctx).loc(),
        dbg_loc,
        "dbg.value lowering should keep the current-value source location"
    );
    let info = llvm::debug_local_variable(&ctx, dbg_value.get_operation())
        .expect("debug local metadata should survive dbg_value lowering");

    assert_eq!(info.name, "x");
    assert_eq!(
        llvm::debug_local_source_scope(&ctx, dbg_value.get_operation()),
        Some(42),
        "dbg.value lowering should keep the MIR source-scope owner"
    );
    let (decl_file, decl_pos) =
        llvm::debug_local_declaration_location(&ctx, dbg_value.get_operation())
            .expect("declaration location should survive dbg_value lowering");
    assert_eq!(decl_file, PathBuf::from("decl.rs"));
    assert_eq!(decl_pos.line, 7);
    assert_eq!(decl_pos.column, 3);
    assert_eq!(
        info.ty,
        llvm::DebugLocalTypeKind::Basic {
            name: "i32".to_string(),
            size_bits: 32,
            encoding: "DW_ATE_signed",
        }
    );
    let fragments = llvm::debug_fragment_variables(&ctx, dbg_value.get_operation());
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].fragment.offset_bits, 32);
    assert_eq!(fragments[0].fragment.size_bits, 32);
    assert_eq!(fragments[0].variable.name, "pair");
}

#[test]
fn convert_dbg_value_list_preserves_operands_and_expression() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();

    let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into(), i64_ty], vec![]);
    let base = block.deref(&ctx).get_argument(0);
    let index = block.deref(&ctx).get_argument(1);

    let dbg_op = mir::MirDbgValueListOp::new(&mut ctx, vec![base, index]);
    llvm::set_debug_local_variable(
        &mut ctx,
        dbg_op.get_operation(),
        llvm::DebugLocalVariableInfo {
            name: "item".to_string(),
            argument_index: None,
            ty: llvm::DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
        },
    );
    let expression = llvm::DebugValueExpression::new(vec![
        llvm::DebugValueExpressionOp::Arg(0),
        llvm::DebugValueExpressionOp::Arg(1),
        llvm::DebugValueExpressionOp::ConstU(4),
        llvm::DebugValueExpressionOp::Mul,
        llvm::DebugValueExpressionOp::Plus,
        llvm::DebugValueExpressionOp::Deref,
    ]);
    llvm::set_debug_value_expression(&mut ctx, dbg_op.get_operation(), &expression);
    dbg_op.get_operation().insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(count_ops::<mir::MirDbgValueListOp>(&ctx, &body), 0);
    let dbg_value = find_first::<llvm::DebugValueListOp>(&ctx, &body)
        .expect("expected lowered llvm.dbg_value_list marker");
    assert_eq!(dbg_value.values(&ctx), vec![base, index]);
    assert_eq!(
        llvm::debug_value_expression(&ctx, dbg_value.get_operation()),
        Some(expression)
    );
}

#[test]
fn convert_alloca_preserves_local_memory_provenance() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

    let alloca_op = Operation::new(
        &mut ctx,
        mir::MirAllocaOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    let provenance = llvm_export::ops::LocalMemoryProvenanceAttr {
        local_index: 3,
        size_bytes: 16,
        binding_name: "scratch".into(),
        type_name: "[u32; 4]".into(),
    };
    llvm_export::ops::set_local_memory_provenance(&mut ctx, alloca_op, provenance.clone());
    alloca_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
    let copied = llvm_export::ops::local_memory_provenance(&ctx, alloca.get_operation()).unwrap();
    assert_eq!(copied, provenance);
}

#[test]
fn mem2reg_salvages_tagged_alloca_into_mir_dbg_value() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![i32_ty]);
    let arg = block.deref(&ctx).get_argument(0);

    let alloca_op = Operation::new(
        &mut ctx,
        mir::MirAllocaOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    let decl_loc = src_location(&mut ctx, "kernel.rs", 12, 9);
    alloca_op.deref_mut(&ctx).set_loc(decl_loc.clone());
    llvm::set_debug_local_variable(
        &mut ctx,
        alloca_op,
        llvm::DebugLocalVariableInfo {
            name: "x".to_string(),
            argument_index: Some(1),
            ty: llvm::DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm::set_debug_local_source_scope(&mut ctx, alloca_op, 9);
    alloca_op.insert_at_back(block, &ctx);
    let slot = alloca_op.deref(&ctx).get_result(0);

    let store_op = Operation::new(
        &mut ctx,
        mir::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![slot, arg],
        vec![],
        0,
    );
    store_op.insert_at_back(block, &ctx);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty],
        vec![slot],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    let loaded = load_op.deref(&ctx).get_result(0);
    append_mir_return(&mut ctx, block, vec![loaded]);

    let mut analyses = pliron::pass::AnalysisManager::default();
    pliron::opts::mem2reg::mem2reg(module_ptr, &mut ctx, &mut analyses)
        .expect("mem2reg should promote the local slot");

    let blocks = vec![block];
    assert_eq!(count_ops::<mir::MirAllocaOp>(&ctx, &blocks), 0);
    assert_eq!(count_ops::<mir::MirStoreOp>(&ctx, &blocks), 0);
    assert_eq!(count_ops::<mir::MirLoadOp>(&ctx, &blocks), 0);

    let dbg_values = find_all::<mir::MirDbgValueOp>(&ctx, &blocks);
    assert!(
        !dbg_values.is_empty(),
        "mem2reg should leave value-based debug records for promoted locals"
    );
    let info = llvm::debug_local_variable(&ctx, dbg_values[0].get_operation())
        .expect("mir.dbg_value should carry the promoted local metadata");
    assert_eq!(info.name, "x");
    assert_eq!(info.argument_index, Some(1));
    assert_eq!(
        llvm::debug_local_source_scope(&ctx, dbg_values[0].get_operation()),
        Some(9),
        "mem2reg salvage should keep the local's MIR source-scope owner"
    );
    assert_eq!(
        dbg_values[0].get_operation().deref(&ctx).loc(),
        decl_loc,
        "debug records for source-less promoted ops should fall back to the local declaration"
    );
}

#[test]
fn mem2reg_salvages_fragment_only_alloca_into_mir_dbg_value() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![i32_ty]);
    let arg = block.deref(&ctx).get_argument(0);

    let alloca_op = Operation::new(
        &mut ctx,
        mir::MirAllocaOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    let alloca_loc = src_location(&mut ctx, "kernel.rs", 20, 9);
    alloca_op.deref_mut(&ctx).set_loc(alloca_loc);
    llvm::set_debug_fragment_variables(
        &mut ctx,
        alloca_op,
        &[llvm::DebugFragmentVariableInfo {
            variable: llvm::DebugLocalVariableInfo {
                name: "pair".to_string(),
                argument_index: None,
                ty: llvm::DebugLocalTypeKind::Array {
                    name: "[u32; 2]".to_string(),
                    size_bits: 64,
                    element: Box::new(llvm::DebugLocalTypeKind::Basic {
                        name: "u32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_unsigned",
                    }),
                    count: 2,
                },
            },
            fragment: llvm::DebugFragment {
                offset_bits: 32,
                size_bits: 32,
            },
            source_scope: Some(9),
            declaration: Some(llvm::DebugSourcePosition {
                file: PathBuf::from("kernel.rs"),
                line: 20,
                column: 9,
            }),
        }],
    );
    alloca_op.insert_at_back(block, &ctx);
    let slot = alloca_op.deref(&ctx).get_result(0);

    let store_op = Operation::new(
        &mut ctx,
        mir::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![slot, arg],
        vec![],
        0,
    );
    store_op.insert_at_back(block, &ctx);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty],
        vec![slot],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    let loaded = load_op.deref(&ctx).get_result(0);
    append_mir_return(&mut ctx, block, vec![loaded]);

    let mut analyses = pliron::pass::AnalysisManager::default();
    pliron::opts::mem2reg::mem2reg(module_ptr, &mut ctx, &mut analyses)
        .expect("mem2reg should promote fragment storage");

    let dbg_values = find_all::<mir::MirDbgValueOp>(&ctx, &[block]);
    assert!(
        !dbg_values.is_empty(),
        "fragment-only storage should still produce mir.dbg_value salvage"
    );
    let fragments = llvm::debug_fragment_variables(&ctx, dbg_values[0].get_operation());
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].variable.name, "pair");
    assert_eq!(fragments[0].fragment.offset_bits, 32);
    assert_eq!(fragments[0].fragment.size_bits, 32);
}
