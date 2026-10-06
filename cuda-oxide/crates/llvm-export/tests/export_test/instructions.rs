/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    attributes::GepNoWrapFlags,
    export::{
        NvvmExportConfig, NvvmIrDialect, export_module_to_string,
        export_module_to_string_with_config,
    },
    op_interfaces::{CastOpInterface, VolatilityOpInterface},
    ops::{
        AddrSpaceCastOp, AllocaOp, AsmKind, BitcastOp, ConstantOp, FuncOp, GepIndex,
        GetElementPtrOp, InlineAsmOp, InlineAsmOpExt, LoadOp, ReturnOp, SelectOp, StoreOp,
    },
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        ops::ModuleOp,
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
    utils::apint::APInt,
};
use std::num::NonZero;

use crate::common::module_top_block;

#[test]
fn export_volatile_load_prints_keyword() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![ptr_ty.to_handle()], false);
    let func = FuncOp::new(&mut ctx, "volatile_load_test".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ptr = entry.deref(&ctx).get_argument(0);

    let load = LoadOp::new(&mut ctx, ptr, i32_ty.to_handle());
    load.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    // An absent attribute and an explicitly cleared attribute are both ordinary
    // accesses. Checking true -> false catches presence-only interpretations.
    for volatile in [None, Some(true), Some(false)] {
        if let Some(volatile) = volatile {
            load.set_volatile(&ctx, volatile);
        }
        module
            .get_operation()
            .verify(&ctx)
            .expect("valid LLVM dialect");
        for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
            let config = NvvmExportConfig::new(dialect);
            let ir = export_module_to_string_with_config(&ctx, &module, &config)
                .expect("export succeeds");
            let lines: Vec<_> = ir
                .lines()
                .filter(|line| line.contains(" = load "))
                .collect();
            assert_eq!(lines.len(), 1, "one load must remain:\n{ir}");
            let keyword = if volatile == Some(true) {
                "volatile "
            } else {
                ""
            };
            assert!(
                lines[0].contains(&format!(" = load {keyword}i32")),
                "load volatility must follow the boolean value:\n{ir}"
            );
        }
    }
}

#[test]
fn export_volatile_store_prints_keyword() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(
        &ctx,
        void_ty.to_handle(),
        vec![ptr_ty.to_handle(), i32_ty.to_handle()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "volatile_store_test".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ptr = entry.deref(&ctx).get_argument(0);
    let val = entry.deref(&ctx).get_argument(1);

    let store = StoreOp::new(&mut ctx, val, ptr);
    store.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    // An absent attribute and an explicitly cleared attribute are both ordinary
    // accesses. Checking true -> false catches presence-only interpretations.
    for volatile in [None, Some(true), Some(false)] {
        if let Some(volatile) = volatile {
            store.set_volatile(&ctx, volatile);
        }
        module
            .get_operation()
            .verify(&ctx)
            .expect("valid LLVM dialect");
        for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
            let config = NvvmExportConfig::new(dialect);
            let ir = export_module_to_string_with_config(&ctx, &module, &config)
                .expect("export succeeds");
            let lines: Vec<_> = ir.lines().filter(|line| line.contains("store ")).collect();
            assert_eq!(lines.len(), 1, "one store must remain:\n{ir}");
            let keyword = if volatile == Some(true) {
                "volatile "
            } else {
                ""
            };
            assert!(
                lines[0].contains(&format!("store {keyword}i32")),
                "store volatility must follow the boolean value:\n{ir}"
            );
        }
    }
}

#[test]
fn legacy_export_uses_one_canonical_pointer_with_multiple_typed_views() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_views".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let ptr_ty = PointerType::get(&ctx, 0);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "multiple_views".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);

    LoadOp::new(&mut ctx, pointer, i32_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    LoadOp::new(&mut ctx, pointer, f32_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    LoadOp::new(&mut ctx, pointer, i8_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("legacy export succeeds");

    assert!(ir.contains("define void @multiple_views(i8* %v0)"), "{ir}");
    assert!(ir.contains("bitcast i8* %v0 to i32*"), "{ir}");
    assert!(ir.contains("load i32, i32*"), "{ir}");
    assert!(ir.contains("bitcast i8* %v0 to float*"), "{ir}");
    assert!(ir.contains("load float, float*"), "{ir}");
    assert!(ir.contains("load i8, i8* %v0"), "{ir}");
    assert!(!ir.contains("bitcast i8* %v0 to i8*"), "{ir}");
    assert!(
        !ir.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|t| t == "ptr")
    );
}

#[test]
fn legacy_alloca_rejects_a_non_default_result_address_space() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_alloca_as".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let func = FuncOp::new(&mut ctx, "invalid_alloca".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);

    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);
    let alloca = AllocaOp::new(&mut ctx, i32_ty.into(), one_value, 3);
    alloca.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream alloca accepts an explicit non-default result address space");
    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("legacy export must reject an alloca address-space mismatch");
    assert!(
        error.contains("alloca result uses address space 3"),
        "{error}"
    );
}

#[test]
fn legacy_gep_rejects_a_result_address_space_different_from_its_base() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_gep_as".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let global_pointer = PointerType::get(&ctx, 1);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![global_pointer.into()], false);
    let func = FuncOp::new(&mut ctx, "invalid_gep".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let base = entry.deref(&ctx).get_argument(0);
    let gep = GetElementPtrOp::new(&mut ctx, base, vec![GepIndex::Constant(0)], i32_ty.into());
    let gep_result = gep.get_operation().deref(&ctx).get_result(0);
    let shared_pointer = PointerType::get(&ctx, 3);
    gep_result.set_type(&ctx, shared_pointer.into());
    gep.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream verification currently does not enforce GEP result/base AS equality");
    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("legacy export must reject a GEP address-space mismatch");
    assert!(
        error.contains("GEP result address-space mismatch: base is 1, result is 3"),
        "{error}"
    );
}

#[test]
fn gep_no_wrap_flags_control_exported_pointer_semantics() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "gep_semantics".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let pointer = PointerType::get(&ctx, 0);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![pointer.into()], false);
    let func = FuncOp::new(&mut ctx, "offsets".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let base = entry.deref(&ctx).get_argument(0);

    let cases = [
        (1, GepNoWrapFlags::empty()),
        (2, GepNoWrapFlags::NUSW),
        (3, GepNoWrapFlags::NUW),
        (4, GepNoWrapFlags::INBOUNDS),
        (5, GepNoWrapFlags::INBOUNDS | GepNoWrapFlags::NUW),
    ];
    for (index, flags) in cases {
        let gep = GetElementPtrOp::new_with_no_wrap_flags(
            &mut ctx,
            base,
            vec![GepIndex::Constant(index)],
            i32_ty.into(),
            flags,
        );
        gep.get_operation().insert_at_back(entry, &ctx);
    }

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("GEP export succeeds");
    let gep_lines: Vec<_> = ir
        .lines()
        .filter(|line| line.contains("getelementptr"))
        .collect();
    assert_eq!(gep_lines.len(), 5, "{ir}");

    assert!(gep_lines[0].contains("getelementptr i32"), "{ir}");
    assert!(
        !gep_lines[0].contains("inbounds")
            && !gep_lines[0].contains("nusw")
            && !gep_lines[0].contains("nuw"),
        "{ir}"
    );
    assert!(gep_lines[1].contains("getelementptr nusw i32"), "{ir}");
    assert!(gep_lines[2].contains("getelementptr nuw i32"), "{ir}");
    assert!(gep_lines[3].contains("getelementptr inbounds i32"), "{ir}");
    assert!(
        !gep_lines[3].contains("nusw"),
        "inbounds must not redundantly print implied nusw:\n{ir}"
    );
    assert!(
        gep_lines[4].contains("getelementptr inbounds nuw i32"),
        "{ir}"
    );
    assert!(
        !gep_lines[4].contains("nusw"),
        "inbounds nuw must not redundantly print implied nusw:\n{ir}"
    );
}

#[test]
fn legacy_pointer_select_keeps_one_canonical_type() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_pointer_select".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let ptr_ty = PointerType::get(&ctx, 0);
    let func_ty = FuncType::get(
        &ctx,
        ptr_ty.into(),
        vec![i1_ty.into(), ptr_ty.into(), ptr_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "choose_pointer".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let condition = entry.deref(&ctx).get_argument(0);
    let if_true = entry.deref(&ctx).get_argument(1);
    let if_false = entry.deref(&ctx).get_argument(2);
    let select = SelectOp::new(&mut ctx, condition, if_true, if_false);
    let selected = select.get_operation().deref(&ctx).get_result(0);
    select.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, Some(selected))
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy pointer select export succeeds");
    assert!(
        ir.contains("select i1 %v0, i8* %v1, i8* %v2"),
        "pointer select must use the canonical byte-pointer type:\n{ir}"
    );
    assert!(ir.contains("ret i8*"), "{ir}");
}

#[test]
fn pointer_bitcast_cannot_cross_address_spaces_in_either_nvvm_dialect() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_pointer_bitcast".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let global_pointer = PointerType::get(&ctx, 1);
    let shared_pointer = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![global_pointer.into()], false);
    let func = FuncOp::new(&mut ctx, "invalid_cast".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let input = entry.deref(&ctx).get_argument(0);
    BitcastOp::new(&mut ctx, input, shared_pointer.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream bitcast verification currently does not enforce pointer AS equality");
    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .expect_err("a cross-address-space pointer bitcast must be rejected");
        assert!(
            error.contains("pointer bitcast cannot cross address spaces 1 -> 3"),
            "{dialect:?}: {error}"
        );
    }
}

#[test]
fn addrspacecast_must_change_address_spaces_in_either_nvvm_dialect() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_addrspacecast".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let shared_pointer = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![shared_pointer.into()], false);
    let func = FuncOp::new(&mut ctx, "invalid_cast".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let input = entry.deref(&ctx).get_argument(0);
    AddrSpaceCastOp::new(&mut ctx, input, shared_pointer.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream addrspacecast verification currently permits equal address spaces");
    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .expect_err("addrspacecast must not encode a no-op address-space conversion");
        assert!(
            error.contains(
                "addrspacecast must change address spaces; source and result are both address space 3"
            ),
            "{dialect:?}: {error}"
        );
    }
}

#[test]
fn legacy_pointer_slot_is_recursively_canonical() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_pointer_slot".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "pointer_slot".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let incoming = entry.deref(&ctx).get_argument(0);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);

    let slot = AllocaOp::new(&mut ctx, ptr_ty.into(), one_value, 0);
    let slot_value = slot.get_operation().deref(&ctx).get_result(0);
    slot.get_operation().insert_at_back(entry, &ctx);
    StoreOp::new(&mut ctx, incoming, slot_value)
        .get_operation()
        .insert_at_back(entry, &ctx);
    LoadOp::new(&mut ctx, slot_value, ptr_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy pointer-slot export succeeds");
    assert!(ir.contains("alloca i8*"), "{ir}");
    assert!(ir.contains("bitcast i8**"), "{ir}");
    assert!(ir.matches("to i8**").count() >= 2, "{ir}");
    assert!(ir.contains("store i8* %v0, i8**"), "{ir}");
    assert!(ir.contains("load i8*, i8**"), "{ir}");
}

#[test]
fn export_inline_asm_respects_upstream_semantics() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "has_inline_asm".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);

    let default_asm = InlineAsmOp::build(
        &mut ctx,
        void_ty.into(),
        vec![],
        "bar.sync 0;",
        "",
        AsmKind::Convergent,
    );
    default_asm.get_operation().insert_at_back(entry, &ctx);

    let register_only_asm = InlineAsmOp::build(
        &mut ctx,
        void_ty.into(),
        vec![],
        "nop;",
        "",
        AsmKind::ConvergentPure,
    );
    register_only_asm
        .get_operation()
        .insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    assert!(
        ir.contains("call void asm sideeffect \"bar.sync 0;\", \"\"() #0"),
        "side-effecting inline asm should emit the LLVM sideeffect marker:\n{ir}"
    );
    assert!(
        ir.contains("call void asm \"nop;\", \"\"() #0"),
        "inline asm marked sideeffect=false should omit the keyword while preserving convergent:\n{ir}"
    );
    assert!(
        ir.contains("attributes #0 = { convergent }"),
        "convergent inline asm must emit the convergent attr group:\n{ir}"
    );
}

#[test]
fn export_inline_asm_checks_the_supported_attribute_subset() {
    use llvm_export::ops::{LlvmAttrValue, LlvmAttributesAttr};
    let export = |side_effects: bool, attribute: Option<(&str, LlvmAttrValue)>| {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
        let module_block = module_top_block(&mut ctx, &module);
        let void_ty = VoidType::get(&ctx);
        let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
        let func = FuncOp::new(&mut ctx, "attributes".try_into().unwrap(), func_ty);
        let entry = func.get_or_create_entry_block(&mut ctx);
        let asm = InlineAsmOp::new(&mut ctx, void_ty.into(), vec![], "nop;", "", side_effects);
        if let Some((name, value)) = attribute {
            let mut attrs = LlvmAttributesAttr::new();
            attrs.set(name, value);
            asm.set_attr_llvm_inline_asm_attrs(&ctx, attrs);
        }
        // Upstream accepts arbitrary LLVM attributes; export must preserve
        // their semantics or explain that the attribute is unsupported.
        asm.verify(&ctx).unwrap();
        asm.get_operation().insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(entry, &ctx);
        func.get_operation().insert_at_back(module_block, &ctx);
        export_module_to_string(&ctx, &module)
    };
    for side_effects in [false, true] {
        for convergent in [false, true] {
            let ir = export(
                side_effects,
                convergent.then_some(("convergent", LlvmAttrValue::Unit)),
            )
            .unwrap();
            let sideeffect = if side_effects { " sideeffect" } else { "" };
            let attribute = if convergent { " #0" } else { "" };
            assert!(
                ir.contains(&format!(
                    "call void asm{sideeffect} \"nop;\", \"\"(){attribute}\n"
                )),
                "{ir}"
            );
            if convergent {
                assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
            }
        }
        assert_eq!(
            export(side_effects, Some(("noduplicate", LlvmAttrValue::Unit))).unwrap_err(),
            "unsupported LLVM inline asm call-site attribute `noduplicate`; only unit `convergent` is supported"
        );
        for value in [LlvmAttrValue::Int(0), LlvmAttrValue::Str("false".into())] {
            assert_eq!(
                export(side_effects, Some(("convergent", value))).unwrap_err(),
                "LLVM inline asm `convergent` requires a unit attribute"
            );
        }
    }
}

#[test]
fn export_inline_asm_escapes_llvm_string_literals() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(
        &mut ctx,
        "has_escaped_inline_asm".try_into().unwrap(),
        func_ty,
    );
    let entry = func.get_or_create_entry_block(&mut ctx);

    let asm = InlineAsmOp::new(
        &mut ctx,
        void_ty.into(),
        vec![],
        "mov.u32 $0, %laneid;\n// \"quoted\" \\22",
        "~{memory}\\raw",
        true,
    );
    asm.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    assert!(
        ir.contains(
            "call void asm sideeffect \"mov.u32 $0, %laneid;\\0A// \\22quoted\\22 \\5C22\", \"~{memory}\\5Craw\"()"
        ),
        "inline asm template and constraints must be escaped as LLVM string literals:\n{ir}"
    );
}

/// A float binop carrying `FastmathFlags` must export with the matching LLVM
/// fast-math keyword (`fast` for the all-bits set), while a float binop with no
/// flags must export with none. Regression guard: the textual exporter
/// previously dropped fast-math flags entirely, making the `f*_fast` intrinsic
/// lowering inert end to end.
#[test]
fn export_emits_fast_math_flags_only_on_flagged_float_ops() {
    use llvm_export::attributes::FastmathFlags;
    use llvm_export::op_interfaces::{BinArithOp, FloatBinArithOpWithFastMathFlags};
    use llvm_export::ops::{FAddOp, FMulOp};
    use pliron::builtin::types::FP32Type;

    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let f32_ty = FP32Type::get(&ctx);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(
        &ctx,
        void_ty.to_handle(),
        vec![f32_ty.into(), f32_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "fast_math".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let a = entry.deref(&ctx).get_argument(0);
    let b = entry.deref(&ctx).get_argument(1);

    // fadd with the full fast-math set.
    let fadd = FAddOp::new_with_fast_math_flags(&mut ctx, a, b, FastmathFlags::FAST.into());
    fadd.get_operation().insert_at_back(entry, &ctx);
    // fmul with no flags: must stay flag-free.
    let fmul = FMulOp::new(&mut ctx, a, b);
    fmul.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    assert!(
        ir.contains("fadd fast float"),
        "fast-math fadd must export the `fast` keyword:\n{ir}"
    );
    let fmul_line = ir
        .lines()
        .find(|line| line.contains("fmul"))
        .expect("exported fmul line");
    assert!(
        !fmul_line.contains("fast"),
        "a float binop with no fast-math flags must not gain them:\n{ir}"
    );
}
