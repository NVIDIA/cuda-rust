/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    attributes::ICmpPredicateAttr,
    export::{NvvmExportConfig, NvvmIrDialect, export_module_to_string_with_config},
    op_interfaces::CastOpInterface,
    ops::{
        AddrSpaceCastOp, AllocaOp, BitcastOp, BrOp, CallOp, CondBrOp, ConstantOp, FuncOp, GepIndex,
        GetElementPtrOp, ICmpOp, LoadOp, PtrToIntOp, ReturnOp, SelectOp, StoreOp,
    },
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::CallOpCallable,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    printable::Printable,
    r#type::Typed,
    utils::apint::APInt,
    value::Value,
};
use std::num::NonZero;

/// Find the module-level block that owns LLVM function definitions.
/// Fixtures insert their declarations and definitions in this shared scope.
fn module_top_block(ctx: &Context, module: &ModuleOp) -> Ptr<BasicBlock> {
    module
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .unwrap()
}

/// Uses that can reveal a scratch allocation's identity across iterations.
/// The same uses are safe to hoist when the allocation executes at most once.
#[derive(Clone, Copy, Debug)]
enum AddressUse {
    /// Direct memory accesses through a chain of pointer views are local.
    /// This is the positive control beside every rejection case.
    Local,
    /// Passing the address to a callee may retain it or compare it.
    /// Reject calls even when the declaration has no visible implementation.
    Call,
    /// A backedge can retain the previous iteration's pointer as a block argument.
    /// This is the LLVM-dialect representation of a pointer PHI.
    Phi,
    /// A pointer equality test can observe repeated allocation identity.
    /// It must not become a comparison against one reused slot.
    Compare,
    /// Converting an address to an integer exposes its numeric identity.
    /// The numeric result may outlive the pointer-typed use chain.
    Convert,
    /// Returning an address exposes it outside this function's use graph.
    /// Conservative classification does not assume anything about the caller.
    Return,
    /// Storing the pointer as data can keep it alive across loop iterations.
    /// A store through the pointer is accepted, but storing the pointer is not.
    Store,
    /// A select can forward an address to later operations or iterations.
    /// Unsupported pointer-producing operations must fail closed.
    Select,
}

/// Build a scratch slot followed by each accepted pointer-view operation.
/// Its last view is used both for ordinary loads/stores and the chosen escape.
fn address_use_fixture(
    ctx: &mut Context,
    usage: AddressUse,
    cyclic: bool,
    multi_block: bool,
) -> (ModuleOp, Ptr<Operation>, Ptr<BasicBlock>, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, "address_use".try_into().unwrap());
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let pointer_ty = PointerType::get(ctx, 0);
    let global_ty = PointerType::get(ctx, 1);
    let ty = FuncType::get(
        ctx,
        pointer_ty.into(),
        vec![i1_ty.into(), pointer_ty.into()],
        false,
    );
    let func = FuncOp::new(ctx, "scratch_views".try_into().unwrap(), ty);
    let entry = func.get_or_create_entry_block(ctx);
    func.get_operation()
        .insert_at_back(module_top_block(ctx, &module), ctx);
    let repeat = entry.deref(ctx).get_argument(0);
    let destination = entry.deref(ctx).get_argument(1);
    let region = func.get_operation().deref(ctx).get_region(0);
    let body = BasicBlock::new(ctx, None, vec![pointer_ty.into()]);
    let exit = BasicBlock::new(ctx, None, vec![]);
    body.insert_at_back(region, ctx);
    exit.insert_at_back(region, ctx);
    BrOp::new(ctx, body, vec![destination])
        .get_operation()
        .insert_at_back(entry, ctx);
    let count = one(ctx, body);
    let slot = AllocaOp::new(ctx, i32_ty.into(), count, 0);
    slot.get_operation().insert_at_back(body, ctx);
    let pointer = slot.get_operation().deref(ctx).get_result(0);
    let gep = GetElementPtrOp::new(ctx, pointer, vec![GepIndex::Constant(0)], i32_ty.into());
    gep.get_operation().insert_at_back(body, ctx);
    let view = gep.get_operation().deref(ctx).get_result(0);
    let cast = BitcastOp::new(ctx, view, pointer_ty.into());
    cast.get_operation().insert_at_back(body, ctx);
    let view = cast.get_operation().deref(ctx).get_result(0);
    let global = AddrSpaceCastOp::new(ctx, view, global_ty.into());
    global.get_operation().insert_at_back(body, ctx);
    let view = global.get_operation().deref(ctx).get_result(0);
    let generic = AddrSpaceCastOp::new(ctx, view, pointer_ty.into());
    generic.get_operation().insert_at_back(body, ctx);
    let view = generic.get_operation().deref(ctx).get_result(0);
    StoreOp::new(ctx, count, view)
        .get_operation()
        .insert_at_back(body, ctx);
    LoadOp::new(ctx, view, i32_ty.into())
        .get_operation()
        .insert_at_back(body, ctx);
    match usage {
        AddressUse::Call => {
            let callee_ty = FuncType::get(
                ctx,
                VoidType::get(ctx).into(),
                vec![pointer_ty.into()],
                false,
            );
            let name: pliron::identifier::Identifier = "observe".try_into().unwrap();
            FuncOp::new(ctx, name.clone(), callee_ty)
                .get_operation()
                .insert_at_back(module_top_block(ctx, &module), ctx);
            CallOp::new(ctx, CallOpCallable::Direct(name), callee_ty, vec![view])
                .get_operation()
                .insert_at_back(body, ctx);
        }
        AddressUse::Compare => {
            ICmpOp::new(ctx, ICmpPredicateAttr::EQ, view, destination)
                .get_operation()
                .insert_at_back(body, ctx);
        }
        AddressUse::Convert => {
            PtrToIntOp::new(ctx, view, i64_ty.into())
                .get_operation()
                .insert_at_back(body, ctx);
        }
        AddressUse::Store => {
            StoreOp::new(ctx, view, destination)
                .get_operation()
                .insert_at_back(body, ctx);
        }
        AddressUse::Select => {
            SelectOp::new(ctx, repeat, view, destination)
                .get_operation()
                .insert_at_back(body, ctx);
        }
        AddressUse::Local | AddressUse::Phi | AddressUse::Return => {}
    }
    let forwarded = if matches!(usage, AddressUse::Phi) {
        view
    } else {
        destination
    };
    if cyclic {
        let latch = if multi_block {
            let latch = BasicBlock::new(ctx, None, vec![]);
            latch.insert_at_back(region, ctx);
            BrOp::new(ctx, latch, vec![])
                .get_operation()
                .insert_at_back(body, ctx);
            latch
        } else {
            body
        };
        CondBrOp::new(ctx, repeat, body, vec![forwarded], exit, vec![])
            .get_operation()
            .insert_at_back(latch, ctx);
    } else {
        BrOp::new(ctx, exit, vec![])
            .get_operation()
            .insert_at_back(body, ctx);
    }
    let returned = if matches!(usage, AddressUse::Return) {
        view
    } else {
        destination
    };
    ReturnOp::new(ctx, Some(returned))
        .get_operation()
        .insert_at_back(exit, ctx);
    (module, slot.get_operation(), entry, body)
}

#[test]
fn loop_address_uses_fail_closed_through_pointer_views_in_both_dialects() {
    for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
        for multi_block in [false, true] {
            for usage in [
                AddressUse::Local,
                AddressUse::Call,
                AddressUse::Phi,
                AddressUse::Compare,
                AddressUse::Convert,
                AddressUse::Return,
                AddressUse::Store,
                AddressUse::Select,
            ] {
                let mut ctx = Context::new();
                let (module, slot, entry, body) =
                    address_use_fixture(&mut ctx, usage, true, multi_block);
                normalize(&mut ctx, &module, dialect);
                let expected = if matches!(usage, AddressUse::Local) {
                    entry
                } else {
                    body
                };
                assert_eq!(
                    slot.deref(&ctx).get_parent_block(),
                    Some(expected),
                    "{dialect:?}: {usage:?}, multi_block={multi_block}"
                );
            }
        }
    }
}

#[test]
fn once_only_slots_can_hoist_even_when_their_addresses_are_observable() {
    for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
        for usage in [
            AddressUse::Call,
            AddressUse::Compare,
            AddressUse::Convert,
            AddressUse::Return,
            AddressUse::Store,
            AddressUse::Select,
        ] {
            let mut ctx = Context::new();
            let (module, slot, entry, _) = address_use_fixture(&mut ctx, usage, false, false);
            normalize(&mut ctx, &module, dialect);
            assert_eq!(
                slot.deref(&ctx).get_parent_block(),
                Some(entry),
                "{dialect:?}: {usage:?}"
            );
        }
    }
}

/// Exercise the public normalization entry point and verify the mutated IR.
/// Running twice also checks that already-normalized slots and counts stay put.
fn normalize(ctx: &mut Context, module: &ModuleOp, dialect: NvvmIrDialect) {
    module.get_operation().deref(ctx).verify(ctx).unwrap();
    nvvm_transforms::legalize_for_nvvm(ctx, module.get_operation(), dialect, 90).unwrap();
    module.get_operation().deref(ctx).verify(ctx).unwrap();
    let once = module.get_operation().deref(ctx).disp(ctx).to_string();
    nvvm_transforms::legalize_for_nvvm(ctx, module.get_operation(), dialect, 90).unwrap();
    module.get_operation().deref(ctx).verify(ctx).unwrap();
    assert_eq!(
        once,
        module.get_operation().deref(ctx).disp(ctx).to_string(),
        "normalization must be idempotent"
    );
}

/// Build a fixed count at the allocation site, outside the entry block.
/// Normalization must provide a dominating count without moving this definition.
fn one(ctx: &mut Context, block: Ptr<BasicBlock>) -> Value {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let op = ConstantOp::new(ctx, Box::new(attr));
    op.get_operation().insert_at_back(block, ctx);
    op.get_operation().deref(ctx).get_result(0)
}

#[test]
fn fixed_slot_precedes_body_phis_but_initialization_stays_in_body() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "entry_alloca".try_into().unwrap());
    let top = module_top_block(&ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ty = FuncType::get(&ctx, i32_ty.into(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "scratch".try_into().unwrap(), ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let input = entry.deref(&ctx).get_argument(0);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let body = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    body.insert_at_back(region, &ctx);
    BrOp::new(&mut ctx, body, vec![input])
        .get_operation()
        .insert_at_back(entry, &ctx);
    let value = body.deref(&ctx).get_argument(0);
    let count = one(&mut ctx, body);
    let slot = AllocaOp::new(&mut ctx, i32_ty.into(), count, 0);
    llvm_export::ops::set_op_alignment(&mut ctx, slot.get_operation(), 4);
    slot.get_operation().insert_at_back(body, &ctx);
    let pointer = slot.get_operation().deref(&ctx).get_result(0);
    StoreOp::new(&mut ctx, value, pointer)
        .get_operation()
        .insert_at_back(body, &ctx);
    let load = LoadOp::new(&mut ctx, pointer, i32_ty.into());
    let result = load.get_operation().deref(&ctx).get_result(0);
    load.get_operation().insert_at_back(body, &ctx);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(body, &ctx);
    func.get_operation().insert_at_back(top, &ctx);
    module.get_operation().deref(&ctx).verify(&ctx).unwrap();

    // Raw export is a printer; even NVVM export must not silently move storage.
    let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
    for config in [
        NvvmExportConfig::new(NvvmIrDialect::Modern),
        NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    ] {
        let ir = export_module_to_string_with_config(&ctx, &module, &config).unwrap();
        let (entry_text, body_text) = ir.split_once("bb0:\n").unwrap();
        assert!(!entry_text.contains(" = alloca "), "{ir}");
        assert!(body_text.contains(" = alloca "), "{ir}");
    }
    let ptx_ir = llvm_export::export::export_module_to_string(&ctx, &module).unwrap();
    assert!(
        !ptx_ir
            .split_once("bb0:\n")
            .unwrap()
            .0
            .contains(" = alloca ")
    );
    assert_eq!(
        before,
        module.get_operation().deref(&ctx).disp(&ctx).to_string()
    );
    for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
        normalize(&mut ctx, &module, dialect);
        assert_eq!(
            slot.get_operation().deref(&ctx).get_parent_block(),
            Some(entry)
        );
        let entry_ops: Vec<_> = entry.deref(&ctx).iter(&ctx).collect();
        let count = slot
            .get_operation()
            .deref(&ctx)
            .get_operand(0)
            .defining_op()
            .unwrap();
        assert!(
            entry_ops.iter().position(|op| *op == count).unwrap()
                < entry_ops
                    .iter()
                    .position(|op| *op == slot.get_operation())
                    .unwrap()
        );
        assert_eq!(count.deref(&ctx).get_parent_block(), Some(entry));
        assert_eq!(value.get_type(&ctx), i32_ty.into());
        let ir =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .unwrap();
        let (entry, body) = ir.split_once("bb0:\n").unwrap();
        assert_eq!(ir.matches(" = alloca ").count(), 1, "{ir}");
        assert!(entry.contains(" = alloca i32, align 4"), "{ir}");
        assert!(
            body.trim_start().starts_with('%') && body.contains(" = phi i32"),
            "{ir}"
        );
        assert!(
            body.contains("store i32") && body.contains("load i32"),
            "{ir}"
        );
        assert!(!entry.contains("store i32"), "{ir}");
    }
}

#[test]
fn dynamic_alloca_count_stays_in_its_original_block() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "dynamic_alloca".try_into().unwrap());
    let top = module_top_block(&ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let pointer_ty = PointerType::get(&ctx, 0);
    let ty = FuncType::get(&ctx, pointer_ty.into(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "dynamic".try_into().unwrap(), ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let count = entry.deref(&ctx).get_argument(0);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let body = BasicBlock::new(&mut ctx, None, vec![]);
    body.insert_at_back(region, &ctx);
    BrOp::new(&mut ctx, body, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);
    let slot = AllocaOp::new(&mut ctx, i32_ty.into(), count, 0);
    let pointer = slot.get_operation().deref(&ctx).get_result(0);
    slot.get_operation().insert_at_back(body, &ctx);
    ReturnOp::new(&mut ctx, Some(pointer))
        .get_operation()
        .insert_at_back(body, &ctx);
    func.get_operation().insert_at_back(top, &ctx);

    normalize(&mut ctx, &module, NvvmIrDialect::Modern);
    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .unwrap();
    let (entry, body) = ir.split_once("bb0:\n").unwrap();
    assert!(!entry.contains(" = alloca "), "{ir}");
    assert!(body.contains(" = alloca i32, i32 %v0"), "{ir}");
}

#[test]
fn hoisting_preserves_nonunit_count_width_alignment_and_body_uses() {
    for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "count_dominance".try_into().unwrap());
        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
        let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
        let ty = FuncType::get(&ctx, i64_ty.into(), vec![], false);
        let function = FuncOp::new(&mut ctx, "counted".try_into().unwrap(), ty);
        let entry = function.get_or_create_entry_block(&mut ctx);
        function
            .get_operation()
            .insert_at_back(module_top_block(&ctx, &module), &ctx);
        let region = function.get_operation().deref(&ctx).get_region(0);
        let body = BasicBlock::new(&mut ctx, None, vec![]);
        body.insert_at_back(region, &ctx);
        BrOp::new(&mut ctx, body, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let attribute = IntegerAttr::new(i64_ty, APInt::from_u64(3, NonZero::new(64).unwrap()));
        let original = ConstantOp::new(&mut ctx, Box::new(attribute));
        original.get_operation().insert_at_back(body, &ctx);
        let count = original.get_operation().deref(&ctx).get_result(0);
        let slot = AllocaOp::new(&mut ctx, i32_ty.into(), count, 0);
        llvm_export::ops::set_op_alignment(&mut ctx, slot.get_operation(), 32);
        slot.get_operation().insert_at_back(body, &ctx);
        let returned = ReturnOp::new(&mut ctx, Some(count));
        returned.get_operation().insert_at_back(body, &ctx);

        normalize(&mut ctx, &module, dialect);
        assert_eq!(
            slot.get_operation().deref(&ctx).get_parent_block(),
            Some(entry)
        );
        let new_count = slot.get_operation().deref(&ctx).get_operand(0);
        assert_eq!(new_count.get_type(&ctx), i64_ty.into());
        assert_eq!(
            new_count
                .defining_op()
                .unwrap()
                .deref(&ctx)
                .get_parent_block(),
            Some(entry)
        );
        assert_eq!(
            original.get_operation().deref(&ctx).get_parent_block(),
            Some(body)
        );
        assert_eq!(returned.get_operation().deref(&ctx).get_operand(0), count);
        let ir =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .unwrap();
        assert!(ir.contains("alloca i32, i64 3, align 32"), "{ir}");
    }
}

#[test]
fn loop_scratch_hoists_only_when_its_address_does_not_escape() {
    for escapes in [false, true] {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "loop_alloca".try_into().unwrap());
        let top = module_top_block(&ctx, &module);
        let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
        let pointer_ty = PointerType::get(&ctx, 0);
        let ty = FuncType::get(
            &ctx,
            VoidType::get(&ctx).into(),
            vec![i1_ty.into(), pointer_ty.into()],
            false,
        );
        let func = FuncOp::new(&mut ctx, "loop_scratch".try_into().unwrap(), ty);
        let entry = func.get_or_create_entry_block(&mut ctx);
        let repeat = entry.deref(&ctx).get_argument(0);
        let escape_destination = entry.deref(&ctx).get_argument(1);
        let region = func.get_operation().deref(&ctx).get_region(0);
        let body = BasicBlock::new(&mut ctx, None, vec![]);
        let exit = BasicBlock::new(&mut ctx, None, vec![]);
        body.insert_at_back(region, &ctx);
        exit.insert_at_back(region, &ctx);
        BrOp::new(&mut ctx, body, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let count = one(&mut ctx, body);
        let slot = AllocaOp::new(&mut ctx, i32_ty.into(), count, 0);
        let pointer = slot.get_operation().deref(&ctx).get_result(0);
        slot.get_operation().insert_at_back(body, &ctx);
        let gep = GetElementPtrOp::new(
            &mut ctx,
            pointer,
            vec![GepIndex::Constant(0)],
            i32_ty.into(),
        );
        let view = gep.get_operation().deref(&ctx).get_result(0);
        gep.get_operation().insert_at_back(body, &ctx);
        StoreOp::new(&mut ctx, count, view)
            .get_operation()
            .insert_at_back(body, &ctx);
        LoadOp::new(&mut ctx, view, i32_ty.into())
            .get_operation()
            .insert_at_back(body, &ctx);
        if escapes {
            StoreOp::new(&mut ctx, view, escape_destination)
                .get_operation()
                .insert_at_back(body, &ctx);
        }
        CondBrOp::new(&mut ctx, repeat, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(body, &ctx);
        ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(exit, &ctx);
        func.get_operation().insert_at_back(top, &ctx);
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        normalize(&mut ctx, &module, NvvmIrDialect::Modern);
        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::Modern),
        )
        .unwrap();
        let (entry, body) = ir.split_once("bb0:\n").unwrap();
        assert_eq!(entry.contains(" = alloca "), !escapes, "{ir}");
        assert_eq!(body.contains(" = alloca "), escapes, "{ir}");
        assert!(
            body.contains("getelementptr") && body.contains("store i32"),
            "{ir}"
        );
    }
}
