/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        DebugKind, FunctionLocalStaticPlacement, NvvmExportConfig, NvvmIrDialect, PtxExportConfig,
        export_module_to_string_with_config,
    },
    op_interfaces::VolatilityOpInterface,
    ops::{
        AllocaOp, CallOp, ConstantOp, DebugEnumDiscriminant, DebugEnumVariant, DebugFragment,
        DebugFragmentVariableInfo, DebugGlobalVariableInfo, DebugLocalTypeKind,
        DebugLocalVariableInfo, DebugProjectedVariableInfo, DebugSourcePosition, DebugSourceScope,
        DebugSourceScopeLocation, DebugSourceScopeMap, DebugValueExpression,
        DebugValueExpressionOp, DebugValueListOp, DebugValueOp, DebugWholeVariableInfo, FuncOp,
        GlobalInitializerRelocation, GlobalOp, GlobalOpExt, ReturnOp, StoreOp,
        encode_global_initializer_relocations,
    },
    types::{ArrayType, FuncType, PointerType, StructLayout, StructType, VoidType},
};
use pliron::{
    builtin::{
        attributes::{IntegerAttr, StringAttr},
        op_interfaces::CallOpCallable,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    location::{Located, Location},
    op::Op,
    utils::apint::APInt,
};
use std::{num::NonZero, path::PathBuf};

use crate::common::{DebugConfig, PlacementConfig, metadata_id, module_top_block, src_location};

#[test]
fn legacy_export_rejects_debug_metadata() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_debug".try_into().unwrap());
    let config = DebugConfig {
        inner: NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        debug_kind: DebugKind::LineTables,
    };

    let error = export_module_to_string_with_config(&ctx, &module, &config)
        .expect_err("legacy debug output must be rejected");
    assert!(error.contains("legacy LLVM 7"), "{error}");
    assert!(error.contains("debug"), "{error}");
}

#[test]
fn line_table_debug_metadata_emits_function_scope_and_instruction_locations() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    llvm_export::ops::set_debug_function_name(
        &mut ctx,
        func.get_operation(),
        "source_crate::debug_kernel",
    );
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 7, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 8, 5);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let define_line = ir
        .lines()
        .find(|line| line.starts_with("define "))
        .expect("function definition");
    assert!(
        define_line.contains("!dbg !"),
        "function definition should reference its DISubprogram:\n{ir}"
    );

    let ret_line = ir
        .lines()
        .find(|line| line.trim_start().starts_with("ret void"))
        .expect("return instruction");
    assert!(
        ret_line.contains(", !dbg !"),
        "real instructions should carry DILocation attachments:\n{ir}"
    );

    assert!(
        ir.contains("!llvm.dbg.cu = !{!"),
        "module should reference a compile unit:\n{ir}"
    );
    assert!(
        ir.contains("!llvm.module.flags = !{!"),
        "module should declare debug-info flags:\n{ir}"
    );
    assert!(
        ir.contains("!DIFile(filename: \"kernel.rs\", directory: \"/tmp/cuda-oxide/tests\")"),
        "source path should be split into DIFile filename and directory:\n{ir}"
    );
    assert!(
        ir.contains("distinct !DICompileUnit(language: DW_LANG_Rust"),
        "debug export should describe the Rust compile unit:\n{ir}"
    );
    assert!(
        ir.contains(
            "distinct !DISubprogram(name: \"source_crate::debug_kernel\", linkageName: \"debug_kernel\""
        ),
        "function definition should separate its source name from its physical linkage name:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 8, column: 5, scope: !"),
        "instruction location should preserve the source line and column:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_keeps_generic_source_name_separate_from_mangled_symbol() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let linkage_name = "_RNvMCexampleINtCsource7WrapperjE7get_mut";
    let func = FuncOp::new(&mut ctx, linkage_name.try_into().unwrap(), func_ty);
    llvm_export::ops::set_debug_function_name(
        &mut ctx,
        func.get_operation(),
        "source_crate::Wrapper::<u16>::get_mut",
    );
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/generic.rs", 19, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains(&format!("define void @{linkage_name}()")),
        "debug naming must not change the physical function symbol:\n{ir}"
    );
    assert!(
        ir.contains(&format!(
            "distinct !DISubprogram(name: \"source_crate::Wrapper::<u16>::get_mut\", linkageName: \"{linkage_name}\""
        )),
        "generic debug metadata must carry both source and linkage spellings:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_uses_file_scope_for_cross_file_locations() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 38, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(
        &mut ctx,
        "/tmp/cuda-oxide/crates/cuda-device/src/thread.rs",
        292,
        19,
    );
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let ret_line = ir
        .lines()
        .find(|line| line.trim_start().starts_with("ret void"))
        .expect("return instruction");
    assert!(
        ret_line.contains(", !dbg !"),
        "cross-file instructions should keep source locations:\n{ir}"
    );
    assert!(
        ir.contains(
            "!DIFile(filename: \"thread.rs\", directory: \"/tmp/cuda-oxide/crates/cuda-device/src\")"
        ),
        "cross-file locations should get their own DIFile:\n{ir}"
    );
    assert!(
        ir.contains("!DILexicalBlockFile(scope: !"),
        "cross-file locations should use a file-specific debug scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 292, column: 19, scope: !"),
        "cross-file locations should preserve their real source line:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_emits_inlined_at_for_callsite_locations() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 38, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let callee = src_location(
        &mut ctx,
        "/tmp/cuda-oxide/crates/cuda-device/src/thread.rs",
        292,
        19,
    );
    let caller = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 39, 13);
    ret.get_operation()
        .deref_mut(&ctx)
        .set_loc(Location::CallSite {
            callee: Box::new(callee),
            caller: Box::new(caller),
        });
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("!DILocation(line: 39, column: 13, scope: !"),
        "callsite metadata should preserve the caller location:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 292, column: 19, scope: !") && ir.contains(", inlinedAt: !"),
        "callsite metadata should describe the callee location as inlined at the caller:\n{ir}"
    );
}

#[test]
fn debug_locations_use_rustc_source_scope_positions() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 10, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    llvm_export::ops::set_debug_source_scope_map(
        &mut ctx,
        func.get_operation(),
        &DebugSourceScopeMap {
            scopes: vec![
                DebugSourceScope {
                    id: 0,
                    parent: None,
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                        line: 10,
                        column: 1,
                    }),
                    inlined: None,
                },
                DebugSourceScope {
                    id: 1,
                    parent: Some(0),
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                        line: 12,
                        column: 9,
                    }),
                    inlined: None,
                },
            ],
            locations: vec![DebugSourceScopeLocation {
                pos: DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                    line: 12,
                    column: 9,
                },
                scope: 1,
            }],
        },
    );

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 12, 9);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let block_id = ir
        .lines()
        .find_map(|line| {
            if line.contains("!DILexicalBlock(scope: !") && line.contains("line: 12, column: 9") {
                line.split_once(" = ")
                    .map(|(id, _)| id.trim_start_matches('!').to_string())
            } else {
                None
            }
        })
        .expect("nested lexical block should be emitted");

    assert!(
        ir.contains(&format!(
            "!DILocation(line: 12, column: 9, scope: !{block_id})"
        )),
        "instruction location should use the exact rustc source scope, not the function scope:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_emits_dbg_declare_for_tagged_allocas() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    one.get_operation().insert_at_back(entry, &ctx);
    let one_val = one.get_operation().deref(&ctx).get_result(0);

    let tid = AllocaOp::new(&mut ctx, i32_ty.into(), one_val, 0);
    let tid_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 31, 9);
    tid.get_operation().deref_mut(&ctx).set_loc(tid_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        tid.get_operation(),
        DebugLocalVariableInfo {
            name: "tid".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
        },
    );
    tid.get_operation().insert_at_back(entry, &ctx);

    let ptr_ty = PointerType::get(&ctx, 0);
    let ptr = AllocaOp::new(&mut ctx, ptr_ty.into(), one_val, 0);
    let ptr_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 32, 9);
    ptr.get_operation().deref_mut(&ctx).set_loc(ptr_loc);
    let ptr_debug_ty = DebugLocalTypeKind::TypedPointer {
        name: "*mut f32".to_string(),
        size_bits: 64,
        pointee: Box::new(DebugLocalTypeKind::Basic {
            name: "f32".to_string(),
            size_bits: 32,
            encoding: "DW_ATE_float",
        }),
    };
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        ptr.get_operation(),
        DebugLocalVariableInfo {
            name: "ptr".to_string(),
            argument_index: None,
            ty: ptr_debug_ty.clone(),
        },
    );
    llvm_export::ops::set_debug_whole_variable_aliases(
        &mut ctx,
        ptr.get_operation(),
        &[
            DebugWholeVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "first".to_string(),
                    argument_index: None,
                    ty: ptr_debug_ty.clone(),
                },
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                    line: 32,
                    column: 13,
                }),
            },
            DebugWholeVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "second".to_string(),
                    argument_index: None,
                    ty: ptr_debug_ty,
                },
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                    line: 33,
                    column: 13,
                }),
            },
        ],
    );
    ptr.get_operation().insert_at_back(entry, &ctx);

    let nested_ptr = AllocaOp::new(&mut ctx, ptr_ty.into(), one_val, 0);
    let nested_ptr_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 33, 9);
    nested_ptr
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(nested_ptr_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        nested_ptr.get_operation(),
        DebugLocalVariableInfo {
            name: "nested_ptr".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::TypedPointer {
                name: "*const *mut i32".to_string(),
                size_bits: 64,
                pointee: Box::new(DebugLocalTypeKind::TypedPointer {
                    name: "*mut i32".to_string(),
                    size_bits: 64,
                    pointee: Box::new(DebugLocalTypeKind::Basic {
                        name: "i32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_signed",
                    }),
                }),
            },
        },
    );
    nested_ptr.get_operation().insert_at_back(entry, &ctx);

    let array_ptr = AllocaOp::new(&mut ctx, ptr_ty.into(), one_val, 0);
    let array_ptr_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 34, 9);
    array_ptr
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(array_ptr_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        array_ptr.get_operation(),
        DebugLocalVariableInfo {
            name: "array_ptr".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::TypedPointer {
                name: "*const [u16; 4]".to_string(),
                size_bits: 64,
                pointee: Box::new(DebugLocalTypeKind::Array {
                    name: "[u16; 4]".to_string(),
                    size_bits: 64,
                    element: Box::new(DebugLocalTypeKind::Basic {
                        name: "u16".to_string(),
                        size_bits: 16,
                        encoding: "DW_ATE_unsigned",
                    }),
                    count: 4,
                }),
            },
        },
    );
    array_ptr.get_operation().insert_at_back(entry, &ctx);

    let opaque_ptr = AllocaOp::new(&mut ctx, ptr_ty.into(), one_val, 0);
    let opaque_ptr_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 35, 9);
    opaque_ptr
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(opaque_ptr_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        opaque_ptr.get_operation(),
        DebugLocalVariableInfo {
            name: "opaque_ptr".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Pointer {
                name: "*const _".to_string(),
                size_bits: 64,
            },
        },
    );
    opaque_ptr.get_operation().insert_at_back(entry, &ctx);

    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 36, 1);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("emissionKind: FullDebug"),
        "full debug should request full DWARF metadata:\n{ir}"
    );
    assert!(
        ir.contains("isOptimized: false"),
        "full debug export should describe the unoptimized debug path:\n{ir}"
    );
    assert!(
        ir.contains("declare void @llvm.dbg.declare(metadata, metadata, metadata)"),
        "full debug should declare the debug intrinsic it calls:\n{ir}"
    );
    assert!(
        ir.contains("call void @llvm.dbg.declare(metadata ptr %"),
        "tagged allocas should be bound to variables with dbg.declare:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"tid\", arg: 1, scope: !"),
        "argument debug metadata should preserve the argument number:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"ptr\", scope: !"),
        "local debug metadata should omit the arg field:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"first\", scope: !")
            && ir.contains("!DILocalVariable(name: \"second\", scope: !"),
        "each whole identity sharing the ptr alloca must get its own DILocalVariable:\n{ir}"
    );
    assert_eq!(
        ir.matches("call void @llvm.dbg.declare").count(),
        7,
        "five allocas plus two aliases must produce seven independent dbg.declare calls:\n{ir}"
    );
    assert!(
        ir.contains("!DIBasicType(name: \"u32\", size: 32, encoding: DW_ATE_unsigned)"),
        "basic integer variables should get DIBasicType metadata:\n{ir}"
    );
    let f32_id = metadata_id(
        &ir,
        "!DIBasicType(name: \"f32\", size: 32, encoding: DW_ATE_float)",
    );
    assert!(
        ir.contains(&format!(
            "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"*mut f32\", baseType: {f32_id}, size: 64)"
        )),
        "pointer variables should reference their pointee DIType:\n{ir}"
    );
    let i32_id = metadata_id(
        &ir,
        "!DIBasicType(name: \"i32\", size: 32, encoding: DW_ATE_signed)",
    );
    let inner_pointer = format!(
        "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"*mut i32\", baseType: {i32_id}, size: 64)"
    );
    let inner_pointer_id = metadata_id(&ir, &inner_pointer);
    assert!(
        ir.contains(&format!(
            "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"*const *mut i32\", baseType: {inner_pointer_id}, size: 64)"
        )),
        "nested pointer metadata must preserve the complete base-type chain:\n{ir}"
    );

    let u16_id = metadata_id(
        &ir,
        "!DIBasicType(name: \"u16\", size: 16, encoding: DW_ATE_unsigned)",
    );
    let subrange_id = metadata_id(&ir, "!DISubrange(count: 4)");
    let subrange_tuple_id = metadata_id(&ir, &format!("!{{{subrange_id}}}"));
    let array_type = format!(
        "!DICompositeType(tag: DW_TAG_array_type, baseType: {u16_id}, size: 64, elements: {subrange_tuple_id})"
    );
    let array_type_id = metadata_id(&ir, &array_type);
    assert!(
        ir.contains(&format!(
            "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"*const [u16; 4]\", baseType: {array_type_id}, size: 64)"
        )),
        "array pointer metadata must reference the exact element/subrange graph:\n{ir}"
    );

    assert!(
        ir.contains("!DILocalVariable(name: \"opaque_ptr\", scope: !"),
        "unsupported pointers must remain visible as source variables:\n{ir}"
    );
    assert!(
        ir.contains(
            "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"*const _\", baseType: null, size: 64)"
        ),
        "the compatibility pointer must retain its legacy null-base metadata:\n{ir}"
    );
    let null_base_pointer_count = ir
        .lines()
        .filter(|line| line.contains("DW_TAG_pointer_type") && line.contains("baseType: null"))
        .count();
    assert!(
        null_base_pointer_count == 1,
        "only the explicit opaque compatibility pointer may have a null baseType:\n{ir}"
    );

    let Some(llvm_as) = ["llvm-as-22", "llvm-as-21", "llvm-as"]
        .into_iter()
        .find(|tool| {
            std::process::Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|out| out.status.success())
        })
    else {
        eprintln!("skipping pointer-debug llvm-as parse gate: no supported llvm-as on PATH");
        return;
    };
    let ll_path = std::env::temp_dir().join(format!(
        "cuda_oxide_pointer_debug_parse_gate_{}.ll",
        std::process::id()
    ));
    std::fs::write(&ll_path, &ir).expect("write pointer-debug temp .ll");
    let output = std::process::Command::new(llvm_as)
        .arg("-o")
        .arg("/dev/null")
        .arg(&ll_path)
        .output()
        .expect("run llvm-as for pointer debug metadata");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_file(&ll_path);
    assert!(
        output.status.success() && !stderr.contains("invalid debug info"),
        "{llvm_as} rejected pointer debug metadata:\n{stderr}\n--- module ---\n{ir}"
    );
}

#[test]
fn full_debug_metadata_describes_as1_global_with_semantic_rust_type() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let storage_ty = ArrayType::get(&ctx, i8_ty.into(), 8);
    let global = GlobalOp::new_with_alignment(
        &mut ctx,
        "__device_global_7".try_into().unwrap(),
        storage_ty.into(),
        8,
    );
    global.set_address_space(&mut ctx, llvm_export::types::address_space::GLOBAL);
    global.set_initializer_hex(&mut ctx, "0000000000000000");
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        global.get_operation(),
        &DebugGlobalVariableInfo {
            name: "GLOBAL_COUNTER".to_string(),
            namespace: vec!["debuginfo".to_string(), "state".to_string()],
            ty: DebugLocalTypeKind::Basic {
                name: "u64".to_string(),
                size_bits: 64,
                encoding: "DW_ATE_unsigned",
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                line: 90,
                column: 1,
            },
            is_local_to_unit: true,
            is_function_local: false,
        },
    );
    global.get_operation().insert_at_back(module_block, &ctx);

    let full = export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::Full,
        },
    )
    .expect("full debug export succeeds");

    let definition = full
        .lines()
        .find(|line| line.starts_with("@__device_global_7 = "))
        .expect("device-global definition");
    assert!(
        definition.contains("addrspace(1) global [8 x i8] c\"\\00\\00\\00\\00\\00\\00\\00\\00\"")
            && definition.contains(", align 8, !dbg !"),
        "the physical byte storage should retain a global debug attachment:\n{full}"
    );
    assert!(
        full.contains(
            "distinct !DIGlobalVariable(name: \"GLOBAL_COUNTER\", linkageName: \"__device_global_7\""
        ),
        "DWARF should separate the source name from the generated linkage name:\n{full}"
    );
    assert!(
        full.contains("!DINamespace(name: \"debuginfo\", scope: null)")
            && full.contains("!DINamespace(name: \"state\", scope: !")
            && full.contains("scope: !"),
        "the source crate/module hierarchy should scope the leaf name:\n{full}"
    );
    assert!(
        full.contains("isLocal: true, isDefinition: true"),
        "private Rust statics should remain local to the compile unit:\n{full}"
    );
    assert!(
        full.contains("file: !")
            && full.contains("line: 90, type: !")
            && full.contains("align: 64)"),
        "the declaration location and source alignment should be retained:\n{full}"
    );
    assert!(
        full.contains("!DIBasicType(name: \"u64\", size: 64, encoding: DW_ATE_unsigned)"),
        "the debug type must be semantic u64, not the physical [8 x i8] storage:\n{full}"
    );
    assert!(
        full.contains("!DIGlobalVariableExpression(var: !")
            && full.contains("expr: !DIExpression())")
            && full.contains("emissionKind: FullDebug, globals: !"),
        "the global expression should be retained by the compile unit:\n{full}"
    );

    let off = export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::Off,
        },
    )
    .expect("non-debug export succeeds");
    let line_tables = export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::LineTables,
        },
    )
    .expect("line-table export succeeds");
    assert_eq!(
        line_tables, off,
        "a global-only module must not gain variable metadata outside full debug"
    );
}

#[test]
fn full_debug_metadata_describes_as4_constant_with_semantic_rust_type() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "constant_debug".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let storage_ty = ArrayType::get(&ctx, i8_ty.into(), 4);
    let linkage_name = reserved_oxide_symbols::constant_symbol("kernels_L11C5_SCALE");
    let constant = GlobalOp::new_with_alignment(
        &mut ctx,
        linkage_name.as_str().try_into().unwrap(),
        storage_ty.into(),
        4,
    );
    constant.set_address_space(&mut ctx, llvm_export::types::address_space::CONSTANT);
    constant.set_initializer_hex(&mut ctx, "00000000");
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        constant.get_operation(),
        &DebugGlobalVariableInfo {
            name: "SCALE".to_string(),
            namespace: vec!["constant_debug".to_string(), "kernels".to_string()],
            ty: DebugLocalTypeKind::Struct {
                name: "ConstantMemory".to_string(),
                size_bits: 32,
                members: vec![llvm_export::ops::DebugTypeMember {
                    name: "0".to_string(),
                    offset_bits: 0,
                    ty: DebugLocalTypeKind::Struct {
                        name: "UnsafeCell".to_string(),
                        size_bits: 32,
                        members: vec![llvm_export::ops::DebugTypeMember {
                            name: "value".to_string(),
                            offset_bits: 0,
                            ty: DebugLocalTypeKind::Basic {
                                name: "f32".to_string(),
                                size_bits: 32,
                                encoding: "DW_ATE_float",
                            },
                        }],
                    },
                }],
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/constant.rs"),
                line: 11,
                column: 5,
            },
            is_local_to_unit: false,
            is_function_local: false,
        },
    );
    constant.get_operation().insert_at_back(module_block, &ctx);

    // A synthetically tagged AS5 global makes the exporter allow-list
    // adversarial: adding AS4 must not accidentally enable arbitrary spaces.
    let unsupported = GlobalOp::new_with_alignment(
        &mut ctx,
        "local_space_sentinel".try_into().unwrap(),
        storage_ty.into(),
        4,
    );
    unsupported.set_address_space(&mut ctx, llvm_export::types::address_space::LOCAL);
    unsupported.set_initializer_hex(&mut ctx, "00000000");
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        unsupported.get_operation(),
        &DebugGlobalVariableInfo {
            name: "MUST_NOT_APPEAR".to_string(),
            namespace: vec!["constant_debug".to_string()],
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/constant.rs"),
                line: 15,
                column: 5,
            },
            is_local_to_unit: true,
            is_function_local: false,
        },
    );
    unsupported
        .get_operation()
        .insert_at_back(module_block, &ctx);

    let full = export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::Full,
        },
    )
    .expect("full constant-memory debug export succeeds");

    let definition = full
        .lines()
        .find(|line| line.starts_with(&format!("@{linkage_name} = ")))
        .expect("constant-memory definition");
    assert!(
        definition.contains("addrspace(4) global [4 x i8] c\"\\00\\00\\00\\00\"")
            && definition.contains(", align 4, !dbg !"),
        "physical AS4 byte storage must retain a global debug attachment:\n{full}"
    );
    assert!(
        full.contains(&format!(
            "distinct !DIGlobalVariable(name: \"SCALE\", linkageName: \"{linkage_name}\""
        )),
        "DWARF must separate the Rust leaf name from constant-memory linkage:\n{full}"
    );
    assert!(
        full.lines()
            .find(|line| line.contains("distinct !DIGlobalVariable(name: \"SCALE\""))
            .is_some_and(|line| line.contains("isLocal: false, isDefinition: true")),
        "a host-settable constant symbol must retain external visibility:\n{full}"
    );
    assert!(
        full.contains("!DINamespace(name: \"constant_debug\", scope: null)")
            && full.contains("!DINamespace(name: \"kernels\", scope: !"),
        "the crate/module namespace must scope the AS4 source name:\n{full}"
    );

    let f32_id = metadata_id(
        &full,
        "!DIBasicType(name: \"f32\", size: 32, encoding: DW_ATE_float)",
    );
    let value_member_id = metadata_id(
        &full,
        &format!(
            "!DIDerivedType(tag: DW_TAG_member, name: \"value\", baseType: {f32_id}, size: 32, offset: 0)"
        ),
    );
    let unsafe_cell_elements_id = metadata_id(&full, &format!("!{{{value_member_id}}}"));
    let unsafe_cell_id = metadata_id(
        &full,
        &format!(
            "!DICompositeType(tag: DW_TAG_structure_type, name: \"UnsafeCell\", size: 32, elements: {unsafe_cell_elements_id})"
        ),
    );
    let wrapper_member_id = metadata_id(
        &full,
        &format!(
            "!DIDerivedType(tag: DW_TAG_member, name: \"0\", baseType: {unsafe_cell_id}, size: 32, offset: 0)"
        ),
    );
    let wrapper_elements_id = metadata_id(&full, &format!("!{{{wrapper_member_id}}}"));
    let wrapper_id = metadata_id(
        &full,
        &format!(
            "!DICompositeType(tag: DW_TAG_structure_type, name: \"ConstantMemory\", size: 32, elements: {wrapper_elements_id})"
        ),
    );
    assert!(
        full.lines()
            .find(|line| line.contains("distinct !DIGlobalVariable(name: \"SCALE\""))
            .is_some_and(|line| line.contains(&format!("type: {wrapper_id},"))),
        "the AS4 DIE must use the ConstantMemory/UnsafeCell/f32 source graph, not physical [4 x i8] storage:\n{full}"
    );

    let variable_id = metadata_id(
        &full,
        "distinct !DIGlobalVariable(name: \"SCALE\", linkageName:",
    );
    let expression_line = full
        .lines()
        .find(|line| line.contains(&format!("!DIGlobalVariableExpression(var: {variable_id},")))
        .expect("constant-memory global expression");
    assert!(
        expression_line.ends_with("expr: !DIExpression())"),
        "AS4 must keep the empty LLVM expression that NVPTX lowers to address class 4:\n{full}"
    );
    assert!(
        full.contains("emissionKind: FullDebug, globals: !"),
        "the compile unit must retain the AS4 expression:\n{full}"
    );
    let unsupported_definition = full
        .lines()
        .find(|line| line.starts_with("@local_space_sentinel = "))
        .expect("unsupported-space sentinel definition");
    assert!(
        !unsupported_definition.contains("!dbg") && !full.contains("MUST_NOT_APPEAR"),
        "the exporter must continue to reject debug globals outside AS1/AS3/AS4:\n{full}"
    );

    let off = export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::Off,
        },
    )
    .expect("non-debug export succeeds");
    let line_tables = export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::LineTables,
        },
    )
    .expect("line-table export succeeds");
    assert_eq!(
        line_tables, off,
        "AS4 variable metadata must remain gated to full debug"
    );
}

#[test]
fn full_debug_metadata_describes_function_local_as3_array_with_shared_address_class() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "shared_debug".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let raw_owner = reserved_oxide_symbols::device_symbol("shared_kernel");
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let storage_ty = ArrayType::get(&ctx, i32_ty.into(), 32);
    let global = GlobalOp::new_with_alignment(
        &mut ctx,
        "__shared_mem_0".try_into().unwrap(),
        storage_ty.into(),
        4,
    );
    global.set_address_space(&mut ctx, llvm_export::types::address_space::SHARED);
    let info = DebugGlobalVariableInfo {
        name: "TILE".to_string(),
        namespace: vec![
            "debuginfo".to_string(),
            "kernels".to_string(),
            "shared_kernel".to_string(),
        ],
        ty: DebugLocalTypeKind::Array {
            name: "[i32; 32]".to_string(),
            size_bits: 1024,
            element: Box::new(DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            }),
            count: 32,
        },
        declaration: DebugSourcePosition {
            file: PathBuf::from("/tmp/cuda-oxide/tests/shared.rs"),
            line: 40,
            column: 9,
        },
        is_local_to_unit: true,
        is_function_local: true,
    };
    llvm_export::ops::set_debug_global_variable(&mut ctx, global.get_operation(), &info);
    llvm_export::ops::set_debug_global_owner_function(&mut ctx, global.get_operation(), &raw_owner);
    global.get_operation().insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let function = FuncOp::new(&mut ctx, raw_owner.as_str().try_into().unwrap(), func_ty);
    let function_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/shared.rs", 35, 1);
    function
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(function_loc);
    let entry = function.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    function.get_operation().insert_at_back(module_block, &ctx);

    let full = export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::Full,
        },
    )
    .expect("full shared debug export succeeds");

    let definition = full
        .lines()
        .find(|line| line.starts_with("@__shared_mem_0 = "))
        .expect("shared definition");
    assert!(
        definition.contains("addrspace(3) global [32 x i32]")
            && definition.contains(", align 4, !dbg !"),
        "physical AS3 array must retain the debug attachment:\n{full}"
    );
    assert!(
        full.contains("!DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef)"),
        "AS3 must carry CUDA DWARF shared address class 8:\n{full}"
    );
    assert!(
        full.contains("!DINamespace(name: \"debuginfo\", scope: null)")
            && full.contains("!DINamespace(name: \"kernels\", scope: !")
            && full.contains(
                "distinct !DISubprogram(name: \"shared_kernel\", \
                 linkageName: \"shared_kernel\", scope: !"
            ),
        "the owning subprogram must be nested under the structured namespace:\n{full}"
    );
    assert_eq!(
        full.matches("distinct !DISubprogram(name: \"shared_kernel\"")
            .count(),
        1,
        "pre-reservation and function export must reuse one DISubprogram:\n{full}"
    );
    assert!(
        full.contains("distinct !DIGlobalVariable(name: \"TILE\", linkageName: \"__shared_mem_0\"")
            && full.contains("!DISubrange(count: 32)")
            && full.contains("!DICompositeType(tag: DW_TAG_array_type, baseType: !")
            && full.contains("size: 1024, elements: !"),
        "the DIE must keep leaf/linkage and the logical array type:\n{full}"
    );
    assert!(
        full.contains("emissionKind: FullDebug, globals: !"),
        "the CU must retain the one shared expression:\n{full}"
    );
}

fn export_mixed_shared_owner_order(malformed_first: bool) -> String {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "shared_owner_validation".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let raw_owner = reserved_oxide_symbols::device_symbol("shared_kernel");
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let storage_ty = ArrayType::get(&ctx, i32_ty.into(), 4);

    let valid = GlobalOp::new_with_alignment(
        &mut ctx,
        "__shared_mem_valid".try_into().unwrap(),
        storage_ty.into(),
        4,
    );
    valid.set_address_space(&mut ctx, llvm_export::types::address_space::SHARED);
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        valid.get_operation(),
        &DebugGlobalVariableInfo {
            name: "VALID".to_string(),
            namespace: vec!["owner_fixture".to_string(), "shared_kernel".to_string()],
            ty: DebugLocalTypeKind::Array {
                name: "[i32; 4]".to_string(),
                size_bits: 128,
                element: Box::new(DebugLocalTypeKind::Basic {
                    name: "i32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_signed",
                }),
                count: 4,
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/shared-owner.rs"),
                line: 20,
                column: 5,
            },
            is_local_to_unit: true,
            is_function_local: true,
        },
    );
    llvm_export::ops::set_debug_global_owner_function(&mut ctx, valid.get_operation(), &raw_owner);

    let malformed = GlobalOp::new_with_alignment(
        &mut ctx,
        "__shared_mem_malformed".try_into().unwrap(),
        storage_ty.into(),
        4,
    );
    malformed.set_address_space(&mut ctx, llvm_export::types::address_space::SHARED);
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        malformed.get_operation(),
        &DebugGlobalVariableInfo {
            name: "MALFORMED".to_string(),
            namespace: vec!["owner_fixture".to_string(), "shared_kernel".to_string()],
            ty: DebugLocalTypeKind::Array {
                name: "[i32; 4]".to_string(),
                size_bits: 128,
                element: Box::new(DebugLocalTypeKind::Basic {
                    name: "i32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_signed",
                }),
                count: 4,
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/shared-owner.rs"),
                line: 24,
                column: 5,
            },
            is_local_to_unit: true,
            is_function_local: true,
        },
    );
    // This raw spelling normalizes to `shared_kernel`, but it is not the raw
    // symbol indexed for that function definition. It must not borrow the
    // valid sibling's pre-reserved DISubprogram.
    llvm_export::ops::set_debug_global_owner_function(
        &mut ctx,
        malformed.get_operation(),
        "shared_kernel",
    );

    if malformed_first {
        malformed.get_operation().insert_at_back(module_block, &ctx);
        valid.get_operation().insert_at_back(module_block, &ctx);
    } else {
        valid.get_operation().insert_at_back(module_block, &ctx);
        malformed.get_operation().insert_at_back(module_block, &ctx);
    }

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let function = FuncOp::new(&mut ctx, raw_owner.as_str().try_into().unwrap(), func_ty);
    let function_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/shared-owner.rs", 16, 1);
    function
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(function_loc);
    let entry = function.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    function.get_operation().insert_at_back(module_block, &ctx);

    export_module_to_string_with_config(
        &ctx,
        &module,
        &DebugConfig {
            inner: PtxExportConfig,
            debug_kind: DebugKind::Full,
        },
    )
    .expect("a malformed shared owner must not break its valid sibling")
}

fn assert_malformed_shared_owner_fails_closed(full: &str) {
    let valid_definition = full
        .lines()
        .find(|line| line.starts_with("@__shared_mem_valid = "))
        .expect("valid shared definition");
    let malformed_definition = full
        .lines()
        .find(|line| line.starts_with("@__shared_mem_malformed = "))
        .expect("malformed-owner shared definition");
    assert!(
        valid_definition.contains(", !dbg !"),
        "the valid sibling must retain its debug attachment:\n{full}"
    );
    assert!(
        !malformed_definition.contains(", !dbg !"),
        "the malformed owner must not borrow the valid sibling's scope:\n{full}"
    );
    assert!(
        full.contains("distinct !DIGlobalVariable(name: \"VALID\"")
            && !full.contains("distinct !DIGlobalVariable(name: \"MALFORMED\""),
        "only the valid shared source identity may become a DIE:\n{full}"
    );
    assert_eq!(
        full.matches("!DIGlobalVariableExpression(var:").count(),
        1,
        "only the valid sibling may be retained by the compile unit:\n{full}"
    );
    assert_eq!(
        full.matches("distinct !DISubprogram(name: \"shared_kernel\"")
            .count(),
        1,
        "the valid owner must still reuse exactly one DISubprogram:\n{full}"
    );
}

#[test]
fn malformed_shared_owner_before_valid_sibling_cannot_borrow_its_scope() {
    assert_malformed_shared_owner_fails_closed(&export_mixed_shared_owner_order(true));
}

#[test]
fn malformed_shared_owner_after_valid_sibling_cannot_borrow_its_scope() {
    assert_malformed_shared_owner_fails_closed(&export_mixed_shared_owner_order(false));
}

#[test]
fn full_debug_globals_preserve_qualified_identity_visibility_and_relocations() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "global_debug_adversarial".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    let left_ty = ArrayType::get(&ctx, i8_ty.into(), 4);
    let left = GlobalOp::new_with_alignment(
        &mut ctx,
        "__device_global_20".try_into().unwrap(),
        left_ty.into(),
        4,
    );
    left.set_address_space(&mut ctx, llvm_export::types::address_space::GLOBAL);
    left.set_initializer_hex(&mut ctx, "01000000");
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        left.get_operation(),
        &DebugGlobalVariableInfo {
            name: "SAME_LEAF".to_string(),
            namespace: vec!["qualified_fixture".to_string(), "left".to_string()],
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/qualified.rs"),
                line: 20,
                column: 5,
            },
            is_local_to_unit: true,
            is_function_local: false,
        },
    );
    left.get_operation().insert_at_back(module_block, &ctx);

    let right_ty = ArrayType::get(&ctx, i8_ty.into(), 8);
    let right = GlobalOp::new_with_alignment(
        &mut ctx,
        "__device_global_21".try_into().unwrap(),
        right_ty.into(),
        8,
    );
    right.set_address_space(&mut ctx, llvm_export::types::address_space::GLOBAL);
    right.set_initializer_hex(&mut ctx, "0200000000000000");
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        right.get_operation(),
        &DebugGlobalVariableInfo {
            name: "SAME_LEAF".to_string(),
            namespace: vec!["qualified_fixture".to_string(), "right".to_string()],
            ty: DebugLocalTypeKind::Basic {
                name: "u64".to_string(),
                size_bits: 64,
                encoding: "DW_ATE_unsigned",
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/qualified.rs"),
                line: 24,
                column: 5,
            },
            is_local_to_unit: false,
            is_function_local: false,
        },
    );
    right.get_operation().insert_at_back(module_block, &ctx);

    let target_ty = ArrayType::get(&ctx, i8_ty.into(), 4);
    let target = GlobalOp::new_with_alignment(
        &mut ctx,
        "__device_global_22".try_into().unwrap(),
        target_ty.into(),
        4,
    );
    target.set_address_space(&mut ctx, llvm_export::types::address_space::GLOBAL);
    target.set_source_global_key(&mut ctx, "qualified_fixture::RELOCATION_TARGET");
    target.set_initializer_hex(&mut ctx, "78563412");
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        target.get_operation(),
        &DebugGlobalVariableInfo {
            name: "RELOCATION_TARGET".to_string(),
            namespace: vec!["qualified_fixture".to_string()],
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/qualified.rs"),
                line: 30,
                column: 1,
            },
            is_local_to_unit: true,
            is_function_local: false,
        },
    );
    target.get_operation().insert_at_back(module_block, &ctx);

    let references_ty = StructType::get_unnamed(
        &ctx,
        (vec![i64_ty.into(), i64_ty.into()], StructLayout::Unpacked),
    );
    let references = GlobalOp::new_with_alignment(
        &mut ctx,
        "__device_global_23".try_into().unwrap(),
        references_ty.into(),
        8,
    );
    references.set_address_space(&mut ctx, llvm_export::types::address_space::GLOBAL);
    references.set_source_global_key(&mut ctx, "qualified_fixture::REFERENCES");
    references.set_initializer_hex(&mut ctx, "00000000000000000000000000000000");
    references.set_initializer_relocations(
        &mut ctx,
        &encode_global_initializer_relocations(&[
            GlobalInitializerRelocation {
                source_offset: 0,
                width_bytes: 8,
                target_address_space: llvm_export::types::address_space::GLOBAL,
                target_addend: 0,
                target_key: "qualified_fixture::RELOCATION_TARGET".to_string(),
            },
            GlobalInitializerRelocation {
                source_offset: 8,
                width_bytes: 8,
                target_address_space: llvm_export::types::address_space::GLOBAL,
                target_addend: 0,
                target_key: "qualified_fixture::RELOCATION_TARGET".to_string(),
            },
        ]),
    );
    llvm_export::ops::set_debug_global_variable(
        &mut ctx,
        references.get_operation(),
        &DebugGlobalVariableInfo {
            name: "REFERENCES".to_string(),
            namespace: vec!["qualified_fixture".to_string()],
            ty: DebugLocalTypeKind::Array {
                name: "[&u32; 2]".to_string(),
                size_bits: 128,
                element: Box::new(DebugLocalTypeKind::Pointer {
                    name: "&u32".to_string(),
                    size_bits: 64,
                }),
                count: 2,
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/qualified.rs"),
                line: 31,
                column: 1,
            },
            is_local_to_unit: true,
            is_function_local: false,
        },
    );
    references
        .get_operation()
        .insert_at_back(module_block, &ctx);

    let export = |debug_kind| {
        export_module_to_string_with_config(
            &ctx,
            &module,
            &DebugConfig {
                inner: PtxExportConfig,
                debug_kind,
            },
        )
        .expect("global debug export succeeds")
    };
    let full = export(DebugKind::Full);

    let metadata_id = |needle: &str| {
        full.lines()
            .find(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("missing `{needle}` in:\n{full}"))
            .split_once(" = ")
            .expect("metadata definition")
            .0
            .trim_start_matches('!')
            .to_string()
    };
    let left_scope = metadata_id("!DINamespace(name: \"left\"");
    let right_scope = metadata_id("!DINamespace(name: \"right\"");
    let left_variable = full
        .lines()
        .find(|line| {
            line.contains("!DIGlobalVariable(name: \"SAME_LEAF\"")
                && line.contains("linkageName: \"__device_global_20\"")
        })
        .expect("left source global");
    let right_variable = full
        .lines()
        .find(|line| {
            line.contains("!DIGlobalVariable(name: \"SAME_LEAF\"")
                && line.contains("linkageName: \"__device_global_21\"")
        })
        .expect("right source global");
    assert!(left_variable.contains(&format!("scope: !{left_scope}")));
    assert!(left_variable.contains("line: 20") && left_variable.contains("isLocal: true"));
    assert!(right_variable.contains(&format!("scope: !{right_scope}")));
    assert!(right_variable.contains("line: 24") && right_variable.contains("isLocal: false"));

    let target_definition = full
        .lines()
        .find(|line| line.starts_with("@__device_global_22 = "))
        .expect("relocation-only target definition");
    assert!(target_definition.contains("[4 x i8]") && target_definition.contains("!dbg !"));
    let target_variable = full
        .lines()
        .find(|line| line.contains("!DIGlobalVariable(name: \"RELOCATION_TARGET\""))
        .expect("relocation-only target source identity");
    assert!(
        target_variable.contains("linkageName: \"__device_global_22\"")
            && target_variable.contains("line: 30")
    );

    let reference_definition = full
        .lines()
        .find(|line| line.starts_with("@__device_global_23 = "))
        .expect("relocation-backed definition");
    assert_eq!(
        reference_definition.matches("@__device_global_22").count(),
        2,
        "both initializer references must target the same one physical global"
    );
    assert!(
        full.contains("!DICompositeType(tag: DW_TAG_array_type, baseType: !")
            && full.contains("size: 128, elements: !")
            && full.contains("!DIDerivedType(tag: DW_TAG_pointer_type, name: \"&u32\"")
    );

    assert_eq!(
        full.matches("!DIGlobalVariableExpression(var:").count(),
        4,
        "one expression per physical source global, despite repeated relocations"
    );
    let compile_unit = full
        .lines()
        .find(|line| line.contains("!DICompileUnit("))
        .expect("debug compile unit");
    let globals_id = compile_unit
        .split("globals: !")
        .nth(1)
        .and_then(|tail| tail.strip_suffix(')'))
        .expect("compile-unit globals tuple");
    let globals_tuple = full
        .lines()
        .find(|line| line.starts_with(&format!("!{globals_id} = !{{")))
        .expect("compile-unit globals tuple definition");
    assert_eq!(globals_tuple.matches('!').count() - 2, 4, "{globals_tuple}");

    assert_eq!(
        export(DebugKind::LineTables),
        export(DebugKind::Off),
        "initialized and relocation-backed globals must not gain metadata outside Full debug"
    );

    let Some(llvm_as) = ["llvm-as-22", "llvm-as-21", "llvm-as"]
        .into_iter()
        .find(|tool| {
            std::process::Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|out| out.status.success())
        })
    else {
        eprintln!("skipping global-debug parse gate: no llvm-as on PATH");
        return;
    };
    let ll_path = std::env::temp_dir().join(format!(
        "cuda_oxide_global_debug_parse_gate_{}.ll",
        std::process::id()
    ));
    std::fs::write(&ll_path, &full).expect("write temp .ll");
    let output = std::process::Command::new(llvm_as)
        .arg("-o")
        .arg("/dev/null")
        .arg(&ll_path)
        .output()
        .expect("run llvm-as");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_file(&ll_path);
    assert!(
        output.status.success() && !stderr.contains("invalid debug info"),
        "{llvm_as} rejected global debug metadata:\n{stderr}\n--- module ---\n{full}"
    );
}

#[test]
fn full_debug_metadata_emits_rust_enum_variant_parts() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "enum_debug".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "enum_debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/enum.rs", 10, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    func.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::try_from("gpu_kernel").unwrap(),
        StringAttr::new("true".into()),
    );
    let entry = func.get_or_create_entry_block(&mut ctx);

    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    one.get_operation().insert_at_back(entry, &ctx);
    let one_val = one.get_operation().deref(&ctx).get_result(0);

    let direct = AllocaOp::new(&mut ctx, i64_ty.into(), one_val, 0);
    let direct_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/enum.rs", 11, 9);
    direct.get_operation().deref_mut(&ctx).set_loc(direct_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        direct.get_operation(),
        DebugLocalVariableInfo {
            name: "direct".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Enum {
                name: "Direct".to_string(),
                size_bits: 64,
                discriminant: Some(DebugEnumDiscriminant {
                    offset_bits: 0,
                    ty: Box::new(DebugLocalTypeKind::Basic {
                        name: "u8".to_string(),
                        size_bits: 8,
                        encoding: "DW_ATE_unsigned",
                    }),
                }),
                variants: vec![
                    DebugEnumVariant {
                        name: "Small".to_string(),
                        discriminant: Some(3),
                        members: vec![llvm_export::ops::DebugTypeMember {
                            name: "0".to_string(),
                            offset_bits: 32,
                            ty: DebugLocalTypeKind::Basic {
                                name: "u32".to_string(),
                                size_bits: 32,
                                encoding: "DW_ATE_unsigned",
                            },
                        }],
                    },
                    DebugEnumVariant {
                        name: "Empty".to_string(),
                        discriminant: Some(9),
                        members: vec![],
                    },
                ],
            },
        },
    );
    direct.get_operation().insert_at_back(entry, &ctx);

    let signed_direct = AllocaOp::new(&mut ctx, i8_ty.into(), one_val, 0);
    let signed_direct_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/enum.rs", 12, 9);
    signed_direct
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(signed_direct_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        signed_direct.get_operation(),
        DebugLocalVariableInfo {
            name: "signed_direct".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Enum {
                name: "SignedDirect".to_string(),
                size_bits: 8,
                discriminant: Some(DebugEnumDiscriminant {
                    offset_bits: 0,
                    ty: Box::new(DebugLocalTypeKind::Basic {
                        name: "i8".to_string(),
                        size_bits: 8,
                        encoding: "DW_ATE_signed",
                    }),
                }),
                variants: vec![
                    DebugEnumVariant {
                        name: "MinusOne".to_string(),
                        discriminant: Some(255),
                        members: vec![],
                    },
                    DebugEnumVariant {
                        name: "MinusFive".to_string(),
                        discriminant: Some(251),
                        members: vec![],
                    },
                ],
            },
        },
    );
    let signed_direct_ptr = signed_direct.get_operation().deref(&ctx).get_result(0);
    signed_direct.get_operation().insert_at_back(entry, &ctx);
    let minus_one_attr = IntegerAttr::new(i8_ty, APInt::from_u32(255, NonZero::new(8).unwrap()));
    let minus_one = ConstantOp::new(&mut ctx, Box::new(minus_one_attr));
    let minus_one_val = minus_one.get_operation().deref(&ctx).get_result(0);
    minus_one.get_operation().insert_at_back(entry, &ctx);
    let keep_signed_direct = StoreOp::new(&mut ctx, minus_one_val, signed_direct_ptr);
    keep_signed_direct.set_volatile(&ctx, true);
    let keep_signed_direct_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/enum.rs", 12, 9);
    keep_signed_direct
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(keep_signed_direct_loc);
    keep_signed_direct
        .get_operation()
        .insert_at_back(entry, &ctx);

    let signed_scalar = AllocaOp::new(&mut ctx, i8_ty.into(), one_val, 0);
    let signed_scalar_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/enum.rs", 13, 9);
    signed_scalar
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(signed_scalar_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        signed_scalar.get_operation(),
        DebugLocalVariableInfo {
            name: "signed_scalar".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Basic {
                name: "i8".to_string(),
                size_bits: 8,
                encoding: "DW_ATE_signed",
            },
        },
    );
    signed_scalar.get_operation().insert_at_back(entry, &ctx);

    let niche = AllocaOp::new(&mut ctx, i64_ty.into(), one_val, 0);
    let niche_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/enum.rs", 14, 9);
    niche.get_operation().deref_mut(&ctx).set_loc(niche_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        niche.get_operation(),
        DebugLocalVariableInfo {
            name: "niche".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Enum {
                name: "OptionRef".to_string(),
                size_bits: 64,
                discriminant: Some(DebugEnumDiscriminant {
                    offset_bits: 0,
                    ty: Box::new(DebugLocalTypeKind::Basic {
                        name: "usize".to_string(),
                        size_bits: 64,
                        encoding: "DW_ATE_unsigned",
                    }),
                }),
                variants: vec![
                    DebugEnumVariant {
                        name: "None".to_string(),
                        discriminant: Some(0),
                        members: vec![],
                    },
                    DebugEnumVariant {
                        name: "Some".to_string(),
                        discriminant: None,
                        members: vec![llvm_export::ops::DebugTypeMember {
                            name: "0".to_string(),
                            offset_bits: 0,
                            ty: DebugLocalTypeKind::TypedPointer {
                                name: "&u32".to_string(),
                                size_bits: 64,
                                pointee: Box::new(DebugLocalTypeKind::Basic {
                                    name: "u32".to_string(),
                                    size_bits: 32,
                                    encoding: "DW_ATE_unsigned",
                                }),
                            },
                        }],
                    },
                ],
            },
        },
    );
    niche.get_operation().insert_at_back(entry, &ctx);

    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/enum.rs", 15, 5);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("enum debug export");

    assert!(
        ir.contains("!DICompositeType(tag: DW_TAG_variant_part"),
        "Rust enums must contain a DW_TAG_variant_part:\n{ir}"
    );
    assert!(
        ir.contains("flags: DIFlagArtificial"),
        "the physical enum carrier must be marked artificial:\n{ir}"
    );
    let discriminator_member = ir
        .lines()
        .find(|line| {
            line.contains("!DIDerivedType(tag: DW_TAG_member")
                && line.contains("flags: DIFlagArtificial")
        })
        .expect("enum discriminator member");
    assert!(
        discriminator_member.contains("scope: !"),
        "the physical enum carrier must be scoped to its enum object:\n{discriminator_member}\n{ir}"
    );
    let variant_part = ir
        .lines()
        .find(|line| line.contains("!DICompositeType(tag: DW_TAG_variant_part"))
        .expect("enum variant part");
    assert!(
        variant_part.contains("scope: !"),
        "the variant part must be scoped to its enum object:\n{variant_part}\n{ir}"
    );
    let direct_variant_member = ir
        .lines()
        .find(|line| {
            line.contains("!DIDerivedType(tag: DW_TAG_member, name: \"Small\"")
                && line.contains("extraData: i8 3")
        })
        .expect("direct enum variant member");
    assert!(
        direct_variant_member.contains("scope: !"),
        "variant members must be scoped to their DW_TAG_variant_part:\n{direct_variant_member}\n{ir}"
    );
    assert!(
        ir.contains("extraData: i8 3") && ir.contains("extraData: i8 9"),
        "direct-tag variants must carry their physical discriminant values:\n{ir}"
    );
    assert!(
        ir.contains("extraData: i8 255") && ir.contains("extraData: i8 251"),
        "signed direct-tag variants must retain their physical bit patterns:\n{ir}"
    );

    let signed_enum_id = metadata_id(
        &ir,
        "!DICompositeType(tag: DW_TAG_structure_type, name: \"SignedDirect\"",
    );
    let signed_discriminator_member = ir
        .lines()
        .find(|line| {
            line.contains("!DIDerivedType(tag: DW_TAG_member")
                && line.contains(&format!("scope: {signed_enum_id},"))
                && line.contains("flags: DIFlagArtificial")
        })
        .expect("signed enum discriminator member");
    let signed_discriminator_type_id = signed_discriminator_member
        .split("baseType: ")
        .nth(1)
        .and_then(|tail| tail.split(',').next())
        .expect("signed enum discriminator base type");
    let signed_discriminator_type = ir
        .lines()
        .find(|line| line.starts_with(&format!("{signed_discriminator_type_id} = ")))
        .expect("signed enum discriminator base type definition");
    assert!(
        signed_discriminator_type
            .contains("!DIBasicType(name: \"u8\", size: 8, encoding: DW_ATE_unsigned)"),
        "enum discriminant metadata must describe physical tag bits as unsigned:\n{signed_discriminator_type}\n{ir}"
    );

    let signed_scalar_id = metadata_id(&ir, "!DILocalVariable(name: \"signed_scalar\"");
    let signed_scalar_variable = ir
        .lines()
        .find(|line| line.starts_with(&format!("{signed_scalar_id} = ")))
        .expect("signed scalar variable definition");
    let signed_scalar_type_id = signed_scalar_variable
        .split("type: ")
        .nth(1)
        .and_then(|tail| tail.strip_suffix(')'))
        .expect("signed scalar type");
    let signed_scalar_type = ir
        .lines()
        .find(|line| line.starts_with(&format!("{signed_scalar_type_id} = ")))
        .expect("signed scalar type definition");
    assert!(
        signed_scalar_type.contains("!DIBasicType(name: \"i8\", size: 8, encoding: DW_ATE_signed)"),
        "ordinary signed locals must retain signed metadata:\n{signed_scalar_type}\n{ir}"
    );
    assert!(
        ir.contains("extraData: i64 0"),
        "the tagged niche variant must carry the niche value:\n{ir}"
    );

    let some_variant_member = ir
        .lines()
        .find(|line| {
            line.contains("!DIDerivedType(tag: DW_TAG_member, name: \"Some\"")
                && line.contains("baseType:")
        })
        .expect("Some variant member");
    assert!(
        !some_variant_member.contains("extraData:"),
        "the untagged niche variant must be the default branch:\n{some_variant_member}\n{ir}"
    );

    let tools = discover_llc_tools();
    if tools.is_empty() {
        eprintln!("skipping enum discriminator PTX gate: llc unavailable");
        return;
    }
    for (tool, major) in tools {
        let output = run_llc(&tool, &ir, &format!("enum_{major}"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "{tool} (LLVM {major}) rejected enum debug metadata:\n{stderr}\n--- module ---\n{ir}"
        );
        let ptx = String::from_utf8(output.stdout)
            .unwrap_or_else(|error| panic!("{tool} (LLVM {major}) emitted invalid UTF-8: {error}"));
        let discriminant_values = ptx
            .lines()
            .filter(|line| line.contains("DW_AT_discr_value"))
            .map(|record| {
                let value = parse_ptx_byte_record(record).unwrap_or_else(|| {
                    panic!(
                        "{tool} (LLVM {major}) emitted an unreadable DW_AT_discr_value record: {record}"
                    )
                });
                assert!(
                    value <= u128::from(u8::MAX),
                    "{tool} (LLVM {major}) emitted an out-of-range PTX byte record: {record}"
                );
                value
            })
            .collect::<Vec<_>>();
        assert!(
            discriminant_values.contains(&255) && discriminant_values.contains(&251),
            "{tool} (LLVM {major}) must preserve signed enum physical tag bits:\n{ptx}"
        );
    }
}

#[test]
fn full_debug_metadata_uses_file_scope_for_cross_file_local_variables() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    one.get_operation().insert_at_back(entry, &ctx);
    let one_val = one.get_operation().deref(&ctx).get_result(0);

    let tid = AllocaOp::new(&mut ctx, i32_ty.into(), one_val, 0);
    let tid_loc = src_location(
        &mut ctx,
        "/tmp/cuda-oxide/crates/cuda-device/src/thread.rs",
        292,
        19,
    );
    tid.get_operation().deref_mut(&ctx).set_loc(tid_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        tid.get_operation(),
        DebugLocalVariableInfo {
            name: "tid".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
        },
    );
    tid.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("!DILexicalBlockFile(scope: !"),
        "cross-file local variables should get a file-specific debug scope:\n{ir}"
    );
    assert!(
        ir.contains(
            "!DIFile(filename: \"thread.rs\", directory: \"/tmp/cuda-oxide/crates/cuda-device/src\")"
        ),
        "cross-file local variables should reference their source file:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"tid\", scope: !") && ir.contains("line: 292"),
        "cross-file local variables should preserve the variable file scope and line:\n{ir}"
    );
    assert!(
        ir.contains("call void @llvm.dbg.declare"),
        "cross-file local variables should still get dbg.declare bindings:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_emits_dbg_value_for_promoted_locals() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let arg = entry.deref(&ctx).get_argument(0);
    let dbg_value = DebugValueOp::new(&mut ctx, arg);
    let dbg_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 31, 13);
    dbg_value.get_operation().deref_mut(&ctx).set_loc(dbg_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        dbg_value.get_operation(),
        DebugLocalVariableInfo {
            name: "x".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm_export::ops::set_debug_local_declaration_location(
        &mut ctx,
        dbg_value.get_operation(),
        PathBuf::from("/tmp/cuda-oxide/tests/declarations.rs"),
        12,
        5,
    );
    llvm_export::ops::set_debug_fragment_variables(
        &mut ctx,
        dbg_value.get_operation(),
        &[DebugFragmentVariableInfo {
            variable: DebugLocalVariableInfo {
                name: "pair".to_string(),
                argument_index: None,
                ty: DebugLocalTypeKind::Array {
                    name: "[u32; 2]".to_string(),
                    size_bits: 64,
                    element: Box::new(DebugLocalTypeKind::Basic {
                        name: "u32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_unsigned",
                    }),
                    count: 2,
                },
            },
            fragment: DebugFragment {
                offset_bits: 32,
                size_bits: 32,
            },
            source_scope: None,
            declaration: Some(DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/declarations.rs"),
                line: 13,
                column: 5,
            }),
        }],
    );
    dbg_value.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("declare void @llvm.dbg.value(metadata, metadata, metadata)"),
        "full debug should declare dbg.value when it emits one:\n{ir}"
    );
    assert!(
        ir.contains("call void @llvm.dbg.value(metadata i32 %v0, metadata !"),
        "dbg.value should describe the local as the current SSA value:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"x\", arg: 1, scope: !"),
        "dbg.value should preserve formal-argument metadata when the source local is an argument:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"x\", arg: 1, scope: !")
            && ir.contains("file: !")
            && ir.contains("line: 12"),
        "DILocalVariable should use the source declaration line, not the dbg.value line:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 31, column: 13, scope: !"),
        "dbg.value should still be located at the value's current source point:\n{ir}"
    );
    assert!(
        ir.contains("DW_OP_LLVM_fragment, 32, 32"),
        "dbg.value should preserve scalarized source-variable fragments:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"pair\", scope: !") && ir.contains("line: 13"),
        "fragment dbg.value should describe the complete source variable:\n{ir}"
    );
    assert!(
        !ir.contains("llvm.dbg.declare"),
        "a value-only debug record should not force dbg.declare:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_emits_diarglist_for_multi_value_locations() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let ptr_ty = PointerType::get(&ctx, 0);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(
        &ctx,
        void_ty.to_handle(),
        vec![ptr_ty.into(), i64_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 40, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let base = entry.deref(&ctx).get_argument(0);
    let index = entry.deref(&ctx).get_argument(1);
    let dbg_value = DebugValueListOp::new(&mut ctx, vec![base, index]);
    let dbg_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 41, 17);
    dbg_value.get_operation().deref_mut(&ctx).set_loc(dbg_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        dbg_value.get_operation(),
        DebugLocalVariableInfo {
            name: "item".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
        },
    );
    llvm_export::ops::set_debug_value_expression(
        &mut ctx,
        dbg_value.get_operation(),
        &DebugValueExpression::new(vec![
            DebugValueExpressionOp::Arg(0),
            DebugValueExpressionOp::Arg(1),
            DebugValueExpressionOp::ConstU(4),
            DebugValueExpressionOp::Mul,
            DebugValueExpressionOp::Plus,
            DebugValueExpressionOp::Deref,
        ]),
    );
    dbg_value.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("call void @llvm.dbg.value(metadata !DIArgList(ptr %v0, i64 %v1), metadata !"),
        "multi-value debug records should emit an inline DIArgList:\n{ir}"
    );
    assert!(
        ir.contains(
            "!DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 4, DW_OP_mul, DW_OP_plus, DW_OP_deref)"
        ),
        "multi-value debug records should emit the typed location recipe:\n{ir}"
    );
    assert!(
        ir.contains("declare void @llvm.dbg.value(metadata, metadata, metadata)"),
        "multi-value records should reuse the ordinary dbg.value intrinsic declaration:\n{ir}"
    );

    let Some(llvm_as) = ["llvm-as-22", "llvm-as-21", "llvm-as"]
        .into_iter()
        .find(|tool| {
            std::process::Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|out| out.status.success())
        })
    else {
        eprintln!("skipping llvm-as parse gate: no llvm-as-22/llvm-as-21/llvm-as on PATH");
        return;
    };

    let ll_path = std::env::temp_dir().join(format!(
        "cuda_oxide_diarglist_parse_gate_{}.ll",
        std::process::id()
    ));
    std::fs::write(&ll_path, &ir).expect("write temp .ll");
    let output = std::process::Command::new(llvm_as)
        .arg("-o")
        .arg("/dev/null")
        .arg(&ll_path)
        .output()
        .expect("run llvm-as");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_file(&ll_path);
    assert!(
        output.status.success() && !stderr.contains("invalid debug info"),
        "{llvm_as} rejected the emitted multi-value debug module:\n{stderr}\n--- module ---\n{ir}"
    );
}

#[test]
fn full_debug_metadata_uses_inlined_callee_scope_for_inlined_arguments() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "caller_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    llvm_export::ops::set_debug_source_scope_map(
        &mut ctx,
        func.get_operation(),
        &DebugSourceScopeMap {
            scopes: vec![
                DebugSourceScope {
                    id: 0,
                    parent: None,
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                        line: 30,
                        column: 1,
                    }),
                    inlined: None,
                },
                DebugSourceScope {
                    id: 1,
                    parent: Some(0),
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/helper.rs"),
                        line: 7,
                        column: 1,
                    }),
                    inlined: Some(llvm_export::ops::DebugInlinedScope {
                        callee_name: "helper::next".to_string(),
                        callsite: Some(DebugSourcePosition {
                            file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                            line: 41,
                            column: 13,
                        }),
                    }),
                },
            ],
            locations: vec![],
        },
    );

    let entry = func.get_or_create_entry_block(&mut ctx);
    let arg = entry.deref(&ctx).get_argument(0);

    let caller_value = DebugValueOp::new(&mut ctx, arg);
    let caller_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 31, 9);
    caller_value
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(caller_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        caller_value.get_operation(),
        DebugLocalVariableInfo {
            name: "data".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm_export::ops::set_debug_local_source_scope(&mut ctx, caller_value.get_operation(), 0);
    caller_value.get_operation().insert_at_back(entry, &ctx);

    let inlined_value = DebugValueOp::new(&mut ctx, arg);
    let inlined_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/helper.rs", 8, 17);
    inlined_value
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(inlined_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        inlined_value.get_operation(),
        DebugLocalVariableInfo {
            name: "self".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm_export::ops::set_debug_local_source_scope(&mut ctx, inlined_value.get_operation(), 1);
    inlined_value.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("distinct !DISubprogram(name: \"caller_kernel\""),
        "caller should keep its own function debug scope:\n{ir}"
    );
    assert!(
        ir.contains("distinct !DISubprogram(name: \"helper::next\""),
        "inlined callee should get its own DISubprogram scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"data\", arg: 1, scope: !"),
        "caller argument should remain arg #1 in the caller scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"self\", arg: 1, scope: !"),
        "inlined callee argument should remain arg #1 in the callee scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 8, column: 17, scope: !") && ir.contains("inlinedAt: !"),
        "inlined dbg.value location should point at the callee line and caller callsite:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_ignores_tagged_alloca_variables() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 40, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    one.get_operation().insert_at_back(entry, &ctx);
    let one_val = one.get_operation().deref(&ctx).get_result(0);

    let local = AllocaOp::new(&mut ctx, i32_ty.into(), one_val, 0);
    let local_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 41, 9);
    local.get_operation().deref_mut(&ctx).set_loc(local_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        local.get_operation(),
        DebugLocalVariableInfo {
            name: "x".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    local.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("emissionKind: LineTablesOnly"),
        "line-table mode should stay line-table-only:\n{ir}"
    );
    assert!(
        !ir.contains("llvm.dbg.declare"),
        "line-table mode should not emit variable bindings:\n{ir}"
    );
    assert!(
        !ir.contains("DILocalVariable"),
        "line-table mode should not emit local-variable metadata:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_adds_fallback_locations_to_calls() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let helper_ty = FuncType::get(&ctx, i32_ty.to_handle(), vec![], false);
    let helper = FuncOp::new(&mut ctx, "helper".try_into().unwrap(), helper_ty);
    helper.get_operation().insert_at_back(module_block, &ctx);

    let caller_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let caller = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), caller_ty);
    let caller_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 20, 3);
    caller.get_operation().deref_mut(&ctx).set_loc(caller_loc);

    let entry = caller.get_or_create_entry_block(&mut ctx);
    let call = CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("helper".try_into().unwrap()),
        helper_ty,
        vec![],
    );
    call.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    caller.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let call_line = ir
        .lines()
        .find(|line| line.contains("call i32 @helper()"))
        .expect("call instruction");
    assert!(
        call_line.contains(", !dbg !"),
        "calls without their own source span should use the function fallback location:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 20, column: 3, scope: !"),
        "fallback call location should point at the caller's function line:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_emits_explicit_artificial_calls_at_line_zero() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let helper_ty = FuncType::get(&ctx, i32_ty.to_handle(), vec![], false);
    let helper = FuncOp::new(&mut ctx, "helper".try_into().unwrap(), helper_ty);
    helper.get_operation().insert_at_back(module_block, &ctx);

    let caller_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let caller = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), caller_ty);
    let caller_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 20, 3);
    caller.get_operation().deref_mut(&ctx).set_loc(caller_loc);

    let entry = caller.get_or_create_entry_block(&mut ctx);
    let call = CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("helper".try_into().unwrap()),
        helper_ty,
        vec![],
    );
    call.get_operation()
        .deref_mut(&ctx)
        .set_loc(llvm_export::artificial_debug_location());
    call.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    caller.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let call_line = ir
        .lines()
        .find(|line| line.contains("call i32 @helper()"))
        .expect("call instruction");
    assert!(
        call_line.contains(", !dbg !"),
        "LLVM still requires an artificial call location:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 0, column: 0, scope: !"),
        "artificial setup must not reuse the caller's user line:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_emits_scalarized_fragment_dbg_declares() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "fragment_debug".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let func = FuncOp::new(
        &mut ctx,
        "fragment_debug_kernel".try_into().unwrap(),
        func_ty,
    );
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/fragments.rs", 10, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    let entry = func.get_or_create_entry_block(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);

    let whole_ty = DebugLocalTypeKind::Array {
        name: "[u32; 2]".to_string(),
        size_bits: 64,
        element: Box::new(DebugLocalTypeKind::Basic {
            name: "u32".to_string(),
            size_bits: 32,
            encoding: "DW_ATE_unsigned",
        }),
        count: 2,
    };
    for (index, offset_bits) in [0u64, 32].into_iter().enumerate() {
        let alloca = AllocaOp::new(&mut ctx, i32_ty.into(), one_value, 0);
        llvm_export::ops::set_debug_fragment_variables(
            &mut ctx,
            alloca.get_operation(),
            &[DebugFragmentVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "pair".to_string(),
                    argument_index: None,
                    ty: whole_ty.clone(),
                },
                fragment: DebugFragment {
                    offset_bits,
                    size_bits: 32,
                },
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/fragments.rs"),
                    line: 11,
                    column: 9,
                }),
            }],
        );
        let loc = src_location(
            &mut ctx,
            "/tmp/cuda-oxide/tests/fragments.rs",
            12 + index as i32,
            9,
        );
        alloca.get_operation().deref_mut(&ctx).set_loc(loc);
        alloca.get_operation().insert_at_back(entry, &ctx);
    }

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("DW_OP_LLVM_fragment, 0, 32"),
        "first scalarized piece should describe the low fragment:\n{ir}"
    );
    assert!(
        ir.contains("DW_OP_LLVM_fragment, 32, 32"),
        "second scalarized piece should describe the high fragment:\n{ir}"
    );
    assert_eq!(
        ir.matches("!DILocalVariable(name: \"pair\"").count(),
        1,
        "all pieces must share one DILocalVariable identity:\n{ir}"
    );
    assert_eq!(
        ir.matches("call void @llvm.dbg.declare").count(),
        2,
        "each scalarized storage piece should emit one dbg.declare:\n{ir}"
    );

    let Some(llvm_as) = ["llvm-as-22", "llvm-as-21", "llvm-as"]
        .into_iter()
        .find(|tool| {
            std::process::Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|out| out.status.success())
        })
    else {
        eprintln!("skipping fragment llvm-as parse gate: no llvm-as on PATH");
        return;
    };
    let ll_path = std::env::temp_dir().join(format!(
        "cuda_oxide_fragment_parse_gate_{}.ll",
        std::process::id()
    ));
    std::fs::write(&ll_path, &ir).expect("write fragment temp .ll");
    let output = std::process::Command::new(llvm_as)
        .arg("-o")
        .arg("/dev/null")
        .arg(&ll_path)
        .output()
        .expect("run llvm-as for fragments");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let _ = std::fs::remove_file(&ll_path);
    assert!(
        output.status.success() && !stderr.contains("invalid debug info"),
        "{llvm_as} rejected scalarized fragment debug metadata:\n{stderr}\n--- module ---\n{ir}"
    );
}

#[test]
fn full_debug_metadata_emits_projected_dbg_declares() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "projected_debug".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let func = FuncOp::new(
        &mut ctx,
        "projected_debug_kernel".try_into().unwrap(),
        func_ty,
    );
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/projected.rs", 10, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    let entry = func.get_or_create_entry_block(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);

    let storage_ty = ArrayType::get(&ctx, i32_ty.into(), 8);
    let alloca = AllocaOp::new(&mut ctx, storage_ty.into(), one_value, 0);
    llvm_export::ops::set_debug_projected_variables(
        &mut ctx,
        alloca.get_operation(),
        &[
            DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "field_value".to_string(),
                    argument_index: None,
                    ty: DebugLocalTypeKind::Basic {
                        name: "u32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_unsigned",
                    },
                },
                dereference_base: false,
                offset_bytes: 8,
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/projected.rs"),
                    line: 11,
                    column: 9,
                }),
            },
            DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "tuple_value".to_string(),
                    argument_index: None,
                    ty: DebugLocalTypeKind::Basic {
                        name: "u64".to_string(),
                        size_bits: 64,
                        encoding: "DW_ATE_unsigned",
                    },
                },
                dereference_base: false,
                offset_bytes: 16,
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/projected.rs"),
                    line: 12,
                    column: 9,
                }),
            },
            DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "array_value".to_string(),
                    argument_index: None,
                    ty: DebugLocalTypeKind::Basic {
                        name: "u32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_unsigned",
                    },
                },
                dereference_base: false,
                offset_bytes: 24,
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/projected.rs"),
                    line: 13,
                    column: 9,
                }),
            },
            DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "deref_value".to_string(),
                    argument_index: None,
                    ty: DebugLocalTypeKind::Basic {
                        name: "u32".to_string(),
                        size_bits: 32,
                        encoding: "DW_ATE_unsigned",
                    },
                },
                dereference_base: true,
                offset_bytes: 0,
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/projected.rs"),
                    line: 14,
                    column: 9,
                }),
            },
            DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "deref_field_value".to_string(),
                    argument_index: None,
                    ty: DebugLocalTypeKind::Basic {
                        name: "u64".to_string(),
                        size_bits: 64,
                        encoding: "DW_ATE_unsigned",
                    },
                },
                dereference_base: true,
                offset_bytes: 32,
                source_scope: None,
                declaration: Some(DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/projected.rs"),
                    line: 15,
                    column: 9,
                }),
            },
        ],
    );
    alloca.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("projected debug export succeeds");

    assert_eq!(
        ir.matches("call void @llvm.dbg.declare").count(),
        5,
        "each projected source variable should get its own dbg.declare:\n{ir}"
    );
    for name in [
        "field_value",
        "tuple_value",
        "array_value",
        "deref_value",
        "deref_field_value",
    ] {
        assert!(
            ir.contains(&format!("!DILocalVariable(name: \"{name}\"")),
            "missing projected variable {name}:\n{ir}"
        );
    }
    for offset in [8u64, 16, 24] {
        assert!(
            ir.contains(&format!("!DIExpression(DW_OP_plus_uconst, {offset})")),
            "missing static projection offset {offset}:\n{ir}"
        );
    }
    assert!(
        ir.contains("!DIExpression(DW_OP_deref)"),
        "missing dereference-only debug expression:\n{ir}"
    );
    assert!(
        ir.contains("!DIExpression(DW_OP_deref, DW_OP_plus_uconst, 32)"),
        "missing dereference-plus-field debug expression:\n{ir}"
    );
}

/// A kernel `shared_kernel` owning the function-local AS3 static `TILE`,
/// optionally next to the module-level AS1 static `GLOBAL_COUNTER`.
fn function_local_shared_static_module(ctx: &mut Context, with_module_global: bool) -> ModuleOp {
    let module = ModuleOp::new(ctx, "shared_placement".try_into().unwrap());
    let module_block = module_top_block(ctx, &module);
    let raw_owner = reserved_oxide_symbols::device_symbol("shared_kernel");

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let storage_ty = ArrayType::get(ctx, i32_ty.into(), 32);
    let shared = GlobalOp::new_with_alignment(
        ctx,
        "__shared_mem_0".try_into().unwrap(),
        storage_ty.into(),
        4,
    );
    shared.set_address_space(ctx, llvm_export::types::address_space::SHARED);
    llvm_export::ops::set_debug_global_variable(
        ctx,
        shared.get_operation(),
        &DebugGlobalVariableInfo {
            name: "TILE".to_string(),
            namespace: vec![
                "debuginfo".to_string(),
                "kernels".to_string(),
                "shared_kernel".to_string(),
            ],
            ty: DebugLocalTypeKind::Array {
                name: "[i32; 32]".to_string(),
                size_bits: 1024,
                element: Box::new(DebugLocalTypeKind::Basic {
                    name: "i32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_signed",
                }),
                count: 32,
            },
            declaration: DebugSourcePosition {
                file: PathBuf::from("/tmp/cuda-oxide/tests/shared.rs"),
                line: 40,
                column: 9,
            },
            is_local_to_unit: true,
            is_function_local: true,
        },
    );
    llvm_export::ops::set_debug_global_owner_function(ctx, shared.get_operation(), &raw_owner);
    shared.get_operation().insert_at_back(module_block, ctx);

    if with_module_global {
        let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
        let bytes_ty = ArrayType::get(ctx, i8_ty.into(), 8);
        let global = GlobalOp::new_with_alignment(
            ctx,
            "__device_global_0".try_into().unwrap(),
            bytes_ty.into(),
            8,
        );
        global.set_address_space(ctx, llvm_export::types::address_space::GLOBAL);
        global.set_initializer_hex(ctx, "0000000000000000");
        llvm_export::ops::set_debug_global_variable(
            ctx,
            global.get_operation(),
            &DebugGlobalVariableInfo {
                name: "GLOBAL_COUNTER".to_string(),
                namespace: vec!["debuginfo".to_string(), "state".to_string()],
                ty: DebugLocalTypeKind::Basic {
                    name: "u64".to_string(),
                    size_bits: 64,
                    encoding: "DW_ATE_unsigned",
                },
                declaration: DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/shared.rs"),
                    line: 9,
                    column: 1,
                },
                is_local_to_unit: true,
                is_function_local: false,
            },
        );
        global.get_operation().insert_at_back(module_block, ctx);
    }

    let void_ty = VoidType::get(ctx);
    let func_ty = FuncType::get(ctx, void_ty.to_handle(), vec![], false);
    let function = FuncOp::new(ctx, raw_owner.as_str().try_into().unwrap(), func_ty);
    let function_loc = src_location(ctx, "/tmp/cuda-oxide/tests/shared.rs", 35, 1);
    function
        .get_operation()
        .deref_mut(ctx)
        .set_loc(function_loc);
    let entry = function.get_or_create_entry_block(ctx);
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(entry, ctx);
    function.get_operation().insert_at_back(module_block, ctx);
    module
}

fn export_full_debug_with_placement(
    ctx: &Context,
    module: &ModuleOp,
    placement: FunctionLocalStaticPlacement,
) -> String {
    export_module_to_string_with_config(
        ctx,
        module,
        &PlacementConfig {
            inner: DebugConfig {
                inner: PtxExportConfig,
                debug_kind: DebugKind::Full,
            },
            placement,
        },
    )
    .expect("full debug export succeeds")
}

fn line_containing<'a>(ir: &'a str, needle: &str) -> &'a str {
    ir.lines()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("missing line containing {needle:?}:\n{ir}"))
}

/// LLVM 23's verifier rejects a function-scoped variable in the compile
/// unit's `globals:` list and then drops the whole debug graph, so the
/// retained-nodes placement must move the expression onto its owner and
/// leave the compile unit without a `globals:` tuple when nothing else
/// needs one.
#[test]
fn retained_node_placement_moves_function_local_static_onto_its_owner() {
    let mut ctx = Context::new();
    let module = function_local_shared_static_module(&mut ctx, false);

    let retained = export_full_debug_with_placement(
        &ctx,
        &module,
        FunctionLocalStaticPlacement::SubprogramRetainedNodes,
    );
    let expression = metadata_id(&retained, "!DIGlobalVariableExpression(var: !");
    let owner = metadata_id(&retained, "distinct !DISubprogram(name: \"shared_kernel\"");
    let subprogram = line_containing(&retained, "distinct !DISubprogram(name: \"shared_kernel\"");
    assert!(
        subprogram.ends_with(&format!("retainedNodes: !{{{expression}}})")),
        "the owning subprogram must retain the static's expression:\n{retained}"
    );
    assert!(
        line_containing(&retained, "!DIGlobalVariable(name: \"TILE\"")
            .contains(&format!("scope: {owner},")),
        "the variable stays scoped to its owning subprogram:\n{retained}"
    );
    assert!(
        !line_containing(&retained, "!DICompileUnit(").contains("globals:"),
        "a function-local static must not appear in the compile unit's globals:\n{retained}"
    );
    assert!(
        retained.contains("!DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef)"),
        "the shared address class is unchanged by the placement:\n{retained}"
    );

    // The LLVM 21/22 form is the pre-existing output: the compile unit
    // retains the expression and the owner's tuple stays empty.
    let compile_unit = export_full_debug_with_placement(
        &ctx,
        &module,
        FunctionLocalStaticPlacement::CompileUnitGlobals,
    );
    assert!(
        line_containing(
            &compile_unit,
            "distinct !DISubprogram(name: \"shared_kernel\""
        )
        .ends_with("retainedNodes: !{})"),
        "{compile_unit}"
    );
    assert!(
        line_containing(&compile_unit, "!DICompileUnit(").contains("globals: !"),
        "{compile_unit}"
    );
}

/// Module-level statics always belong to the compile unit; only the
/// function-local one moves under the retained-nodes placement.
#[test]
fn retained_node_placement_leaves_module_globals_in_the_compile_unit() {
    let mut ctx = Context::new();
    let module = function_local_shared_static_module(&mut ctx, true);
    let ir = export_full_debug_with_placement(
        &ctx,
        &module,
        FunctionLocalStaticPlacement::SubprogramRetainedNodes,
    );

    let shared_expression = metadata_id(
        &ir,
        "expr: !DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef))",
    );
    let global_expression = metadata_id(&ir, "expr: !DIExpression())");
    assert_ne!(shared_expression, global_expression);

    let subprogram = line_containing(&ir, "distinct !DISubprogram(name: \"shared_kernel\"");
    assert!(
        subprogram.ends_with(&format!("retainedNodes: !{{{shared_expression}}})")),
        "{ir}"
    );

    let compile_unit = line_containing(&ir, "!DICompileUnit(");
    let globals_id = compile_unit
        .split("globals: !")
        .nth(1)
        .and_then(|rest| rest.strip_suffix(')'))
        .expect("compile-unit globals tuple");
    let globals_tuple = line_containing(&ir, &format!("!{globals_id} = !{{"));
    assert_eq!(
        globals_tuple,
        format!("!{globals_id} = !{{{global_expression}}}"),
        "only the module-level static stays in the compile unit:\n{ir}"
    );
}

/// Every `llvm-as` reachable from the test, with its LLVM major: the
/// `llvm-as-NN` / `llvm-as` names on `PATH` plus the Rust sysroot's
/// llvm-tools copy, deduplicated by major.
fn discover_llvm_as_tools() -> Vec<(String, u32)> {
    discover_llvm_tools("llvm-as")
}

fn discover_llc_tools() -> Vec<(String, u32)> {
    discover_llvm_tools("llc")
}

fn discover_llvm_tools(name: &str) -> Vec<(String, u32)> {
    let mut candidates: Vec<String> = [23, 22, 21]
        .into_iter()
        .map(|major| format!("{name}-{major}"))
        .chain(std::iter::once(name.to_owned()))
        .collect();
    if let Some(sysroot) = std::process::Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        && let Some(host) = std::process::Command::new("rustc")
            .arg("-vV")
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| {
                String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .find_map(|line| line.strip_prefix("host: ").map(str::to_owned))
            })
    {
        candidates.push(format!("{sysroot}/lib/rustlib/{host}/bin/{name}"));
    }
    let mut tools: Vec<(String, u32)> = Vec::new();
    for tool in candidates {
        let Some(output) = std::process::Command::new(&tool)
            .arg("--version")
            .output()
            .ok()
            .filter(|out| out.status.success())
        else {
            continue;
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let Some(major) = text
            .split("LLVM version ")
            .nth(1)
            .and_then(|rest| rest.split('.').next())
            .and_then(|major| major.parse::<u32>().ok())
        else {
            continue;
        };
        if tools.iter().all(|(_, seen)| *seen != major) {
            tools.push((tool, major));
        }
    }
    tools
}

fn run_llc(tool: &str, ir: &str, tag: &str) -> std::process::Output {
    let ll_path = std::env::temp_dir().join(format!(
        "cuda_oxide_enum_debug_gate_{}_{tag}.ll",
        std::process::id()
    ));
    std::fs::write(&ll_path, ir).expect("write temp .ll");
    let output = std::process::Command::new(tool)
        .args(["-O0", "-mtriple=nvptx64-nvidia-cuda", "-mcpu=sm_80"])
        .arg("-o")
        .arg("-")
        .arg(&ll_path)
        .output()
        .expect("run llc");
    let _ = std::fs::remove_file(&ll_path);
    output
}

fn parse_ptx_byte_record(record: &str) -> Option<u128> {
    let literal = record
        .split("//")
        .next()?
        .trim()
        .strip_prefix(".b8")?
        .trim();
    if let Some(hex) = literal.strip_prefix("0x") {
        u128::from_str_radix(hex, 16).ok()
    } else {
        literal.parse().ok()
    }
}

fn run_llvm_as(tool: &str, ir: &str, tag: &str) -> std::process::Output {
    let ll_path = std::env::temp_dir().join(format!(
        "cuda_oxide_placement_gate_{}_{tag}.ll",
        std::process::id()
    ));
    std::fs::write(&ll_path, ir).expect("write temp .ll");
    let output = std::process::Command::new(tool)
        .arg("-o")
        .arg("/dev/null")
        .arg(&ll_path)
        .output()
        .expect("run llvm-as");
    let _ = std::fs::remove_file(&ll_path);
    output
}

/// Checks the LLVM 23 boundary in `FunctionLocalStaticPlacement::for_llvm_major`
/// against every reachable `llvm-as`: the form selected for its major must
/// verify, and the other form must be rejected with the verifier's known
/// message. Like `opt` and `llc`, `llvm-as` strips a broken debug graph and
/// still exits 0, so the verdict is read from its stderr: an accepted module
/// produces no "invalid debug info" warning at all.
#[test]
fn llvm_as_verifies_the_placement_selected_for_its_major() {
    let mut ctx = Context::new();
    let module = function_local_shared_static_module(&mut ctx, true);
    let tools = discover_llvm_as_tools();
    if tools.is_empty() {
        eprintln!("skipping placement verifier gate: no llvm-as on PATH or in the sysroot");
        return;
    }
    for (tool, major) in tools {
        let selected = FunctionLocalStaticPlacement::for_llvm_major(major);
        let (rejected, rejection) = match selected {
            FunctionLocalStaticPlacement::CompileUnitGlobals => (
                FunctionLocalStaticPlacement::SubprogramRetainedNodes,
                "invalid retained nodes",
            ),
            FunctionLocalStaticPlacement::SubprogramRetainedNodes => (
                FunctionLocalStaticPlacement::CompileUnitGlobals,
                "function-local variables are not allowed in a DICompileUnit's global variables list",
            ),
        };

        let accepted_ir = export_full_debug_with_placement(&ctx, &module, selected);
        let output = run_llvm_as(&tool, &accepted_ir, &format!("{major}_selected"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && !stderr.contains("invalid debug info"),
            "{tool} (LLVM {major}) rejected the {selected:?} form:\n{stderr}\n--- module ---\n{accepted_ir}"
        );

        let rejected_ir = export_full_debug_with_placement(&ctx, &module, rejected);
        let output = run_llvm_as(&tool, &rejected_ir, &format!("{major}_rejected"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("ignoring invalid debug info"),
            "{tool} (LLVM {major}) accepted the {rejected:?} form, so the boundary in \
             FunctionLocalStaticPlacement::for_llvm_major is stale:\n{stderr}\n{rejected_ir}"
        );
        assert!(
            stderr.contains(rejection),
            "{tool} (LLVM {major}) rejected the {rejected:?} form for an unexpected reason:\n{stderr}"
        );
    }
}
