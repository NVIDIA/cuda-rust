/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        NvvmExportConfig, NvvmIrDialect, export_module_to_string,
        export_module_to_string_with_config,
    },
    ops::{AddressOfOp, BrOp, CallOp, FuncOp, GepIndex, GetElementPtrOp, GlobalOp, ReturnOp},
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::CallOpCallable,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
};

use crate::common::{assert_no_undefined_temporaries, module_top_block};

#[test]
fn indirect_call_rejects_non_program_address_space() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_indirect_callee".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let shared_ptr_ty = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let callee_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let caller_ty = FuncType::get(&ctx, void_ty.into(), vec![shared_ptr_ty.into()], false);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), caller_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let callee = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(callee),
        callee_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .expect_err("NVPTX must reject a shared-memory function pointer");
        assert!(error.contains("address space 3"), "{error}");
        assert!(error.contains("function pointers"), "{error}");
    }
}

#[test]
fn intrinsic_export_preserves_legacy_dots_and_literal_underscores() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "intrinsic_names".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let function_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let legacy = "llvm_nvvm_wgmma_fence_sync_aligned";
    let escaped = "llvm__nvvm_dwgmma_dcommit_ugroup_dsync_daligned";

    for name in [legacy, escaped] {
        FuncOp::new(&mut ctx, name.try_into().unwrap(), function_ty)
            .get_operation()
            .insert_at_back(module_block, &ctx);
    }

    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), function_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    for name in [legacy, escaped] {
        CallOp::new(
            &mut ctx,
            CallOpCallable::Direct(name.try_into().unwrap()),
            function_ty,
            vec![],
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
    }
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("intrinsic export succeeds");
    assert!(ir.contains("@llvm.nvvm.wgmma.fence.sync.aligned"), "{ir}");
    assert!(
        ir.contains("@llvm.nvvm.wgmma.commit_group.sync.aligned"),
        "{ir}"
    );
    assert!(!ir.contains("@llvm.nvvm.wgmma.commit.group"), "{ir}");
}

#[test]
fn legacy_function_address_defined_later_round_trips_through_indirect_call() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "function_address".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let callee_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);

    // Print the caller first to prove symbol typing is a module pre-pass, not
    // an accidental dependency on textual definition order.
    let caller = FuncOp::new(
        &mut ctx,
        "call_function_pointer".try_into().unwrap(),
        callee_ty,
    );
    let caller_entry = caller.get_or_create_entry_block(&mut ctx);
    let address = AddressOfOp::new(&mut ctx, "target".try_into().unwrap(), 0);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(caller_entry, &ctx);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(address_value),
        callee_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(caller_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(caller_entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let target = FuncOp::new(&mut ctx, "target".try_into().unwrap(), callee_ty);
    let target_entry = target.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(target_entry, &ctx);
    target.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy function-address export succeeds");
    assert!(
        ir.contains("bitcast void ()* @target to i8*"),
        "function address must normalize to the canonical byte pointer:\n{ir}"
    );
    assert!(
        ir.contains("bitcast i8*") && ir.contains("to void ()*"),
        "indirect call must restore the exact function pointer type:\n{ir}"
    );
    assert!(ir.contains("call void %"), "{ir}");
}

#[test]
fn modern_function_address_uses_the_normalized_definition_name() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "modern_function_address".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let callee_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let prefixed_name = reserved_oxide_symbols::device_symbol("target");

    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), callee_ty);
    let caller_entry = caller.get_or_create_entry_block(&mut ctx);
    let address = AddressOfOp::new(&mut ctx, prefixed_name.as_str().try_into().unwrap(), 0);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(caller_entry, &ctx);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(address_value),
        callee_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(caller_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(caller_entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let target = FuncOp::new(
        &mut ctx,
        prefixed_name.as_str().try_into().unwrap(),
        callee_ty,
    );
    let target_entry = target.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(target_entry, &ctx);
    target.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern function-address export succeeds");
    assert!(ir.contains("define void @target()"), "{ir}");
    assert!(ir.contains("call void @target()"), "{ir}");
    assert!(!ir.contains(&prefixed_name), "{ir}");
}

#[test]
fn modern_addressof_rejects_global_and_function_address_space_mismatches() {
    let mut ctx = Context::new();
    let void_ty = VoidType::get(&ctx);
    let no_args = FuncType::get(&ctx, void_ty.into(), vec![], false);

    let global_module = ModuleOp::new(&mut ctx, "bad_global_address".try_into().unwrap());
    let global_module_block = module_top_block(&mut ctx, &global_module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(&mut ctx, "shared_value".try_into().unwrap(), i32_ty.into());
    global.set_address_space(&mut ctx, 3);
    global
        .get_operation()
        .insert_at_back(global_module_block, &ctx);
    let global_user = FuncOp::new(&mut ctx, "global_user".try_into().unwrap(), no_args);
    let global_entry = global_user.get_or_create_entry_block(&mut ctx);
    AddressOfOp::new(&mut ctx, "shared_value".try_into().unwrap(), 0)
        .get_operation()
        .insert_at_back(global_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(global_entry, &ctx);
    global_user
        .get_operation()
        .insert_at_back(global_module_block, &ctx);
    let error = export_module_to_string_with_config(
        &ctx,
        &global_module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect_err("modern global addressof must preserve address spaces");
    assert!(
        error.contains("result is 0, global is 3"),
        "unexpected global error: {error}"
    );

    let function_module = ModuleOp::new(&mut ctx, "bad_function_address".try_into().unwrap());
    let function_module_block = module_top_block(&mut ctx, &function_module);
    FuncOp::new(&mut ctx, "target".try_into().unwrap(), no_args)
        .get_operation()
        .insert_at_back(function_module_block, &ctx);
    let function_user = FuncOp::new(&mut ctx, "function_user".try_into().unwrap(), no_args);
    let function_entry = function_user.get_or_create_entry_block(&mut ctx);
    AddressOfOp::new(&mut ctx, "target".try_into().unwrap(), 3)
        .get_operation()
        .insert_at_back(function_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(function_entry, &ctx);
    function_user
        .get_operation()
        .insert_at_back(function_module_block, &ctx);
    let error = export_module_to_string_with_config(
        &ctx,
        &function_module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect_err("modern function addressof must use program address space");
    assert!(error.contains("program-address-space (0)"), "{error}");
}

#[test]
fn export_addressof_uses_symbol_when_definition_block_prints_later() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let existing = {
            let region = module_region.deref(&ctx);
            region.iter(&ctx).next()
        };
        if let Some(block) = existing {
            block
        } else {
            let block = BasicBlock::new(&mut ctx, None, vec![]);
            block.insert_at_back(module_region, &ctx);
            block
        }
    };

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(
        &mut ctx,
        "__shared_mem_20".try_into().unwrap(),
        i32_ty.to_handle(),
    );
    global.set_address_space(&mut ctx, 3);
    global.get_operation().insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "uses_late_addressof".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let func_region = func.get_operation().deref(&ctx).get_region(0);
    let use_block = BasicBlock::new(&mut ctx, None, vec![]);
    use_block.insert_at_back(func_region, &ctx);
    let address_block = BasicBlock::new(&mut ctx, None, vec![]);
    address_block.insert_at_back(func_region, &ctx);

    BrOp::new(&mut ctx, address_block, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let address = AddressOfOp::new(&mut ctx, "__shared_mem_20".try_into().unwrap(), 3);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(address_block, &ctx);
    BrOp::new(&mut ctx, use_block, vec![])
        .get_operation()
        .insert_at_back(address_block, &ctx);

    let gep = GetElementPtrOp::new(
        &mut ctx,
        address_value,
        vec![GepIndex::Constant(0)],
        i32_ty.to_handle(),
    );
    gep.get_operation().insert_at_back(use_block, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(use_block, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    // The shared global must be declared at module scope.
    assert!(
        ir.contains("@__shared_mem_20 = addrspace(3) global"),
        "module must declare the shared global:\n{ir}"
    );

    // The GEP base operand must be the global symbol, not a stale `%vN`.
    let gep_line = ir
        .lines()
        .find(|line| line.contains("getelementptr"))
        .expect("exported GEP line");
    assert!(
        gep_line.contains("@__shared_mem_20"),
        "GEP must use the global symbol, not a stale temporary:\n{ir}"
    );

    // Bug class from issue #54: every `%vN` reference in the IR must have a
    // matching `%vN = ...` definition. With the bug present the addressof
    // result was named `%v1` but never defined; this catches that and any
    // future regression that re-introduces a dangling SSA reference.
    assert_no_undefined_temporaries(&ir);

    let legacy = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy addressof export succeeds");
    assert!(
        legacy.contains("@__shared_mem_20 = addrspace(3) global i32 undef"),
        "NVVM shared globals must be uninitialized:\n{legacy}"
    );
    assert!(
        legacy.contains("bitcast i32 addrspace(3)* @__shared_mem_20 to i8 addrspace(3)*"),
        "legacy addressof must normalize the global pointer:\n{legacy}"
    );
    assert!(
        legacy.contains("bitcast i8 addrspace(3)*") && legacy.contains("to i32 addrspace(3)*"),
        "legacy GEP must repair its canonical base pointer:\n{legacy}"
    );
    assert_no_undefined_temporaries(&legacy);
}
