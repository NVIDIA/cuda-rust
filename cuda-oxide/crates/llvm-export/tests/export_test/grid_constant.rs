/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        DeviceExternAttrs, DeviceExternDecl, DeviceExternType, NvvmExportConfig, NvvmIrDialect,
        export_module_to_string_with_config, export_module_with_externs,
    },
    ops::{
        AddressOfOp, BrOp, CallOp, FuncOp, GepIndex, GetElementPtrOp, LoadOp, ReturnOp, StoreOp,
    },
    types::{ArrayType, FuncType, PointerType, StructLayout, StructType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{CallOpCallable, SymbolOpInterface},
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
};
use reserved_oxide_symbols::{
    LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX, LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
};
use std::num::NonZero;

use crate::common::module_top_block;

fn mark_grid_constant(
    ctx: &Context,
    function: &FuncOp,
    index: usize,
    pointee: TypeHandle,
    align: u64,
) {
    let integer = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let mut operation = function.get_operation().deref_mut(ctx);
    operation.attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    operation.attributes.set(
        format!("{LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX}{index}")
            .as_str()
            .try_into()
            .unwrap(),
        TypeAttr::new(pointee),
    );
    operation.attributes.set(
        format!("{LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX}{index}")
            .as_str()
            .try_into()
            .unwrap(),
        IntegerAttr::new(integer, APInt::from_u64(align, NonZero::new(64).unwrap())),
    );
}

/// Three distinct by-value shapes mixed with ordinary pointer/scalar arguments.
/// A PHI and a device helper must both see the canonical body pointer, while
/// declarations, roots and annotations must retain the exact by-value pointee.
fn mixed_module(ctx: &mut Context, include_forward_address: bool) -> ModuleOp {
    let module = ModuleOp::new(ctx, "grid_constants".try_into().unwrap());
    let top = module_top_block(ctx, &module);
    let void = VoidType::get(ctx);
    let pointer = PointerType::get(ctx, 0);
    let byte = IntegerType::get(ctx, 8, Signedness::Signless);
    let word = IntegerType::get(ctx, 32, Signedness::Signless);
    let index = IntegerType::get(ctx, 64, Signedness::Signless);
    let descriptor = ArrayType::get(ctx, byte.into(), 128);
    let words = ArrayType::get(ctx, word.into(), 2);
    let packed =
        StructType::get_unnamed(ctx, (vec![byte.into(), words.into()], StructLayout::Packed));
    if include_forward_address {
        let address_type = FuncType::get(ctx, pointer.into(), vec![], false);
        let address_function =
            FuncOp::new(ctx, "get_kernel_address".try_into().unwrap(), address_type);
        let entry = address_function.get_or_create_entry_block(ctx);
        let address = AddressOfOp::new(ctx, "mixed_maps".try_into().unwrap(), 0);
        let value = address.get_operation().deref(ctx).get_result(0);
        address.get_operation().insert_at_back(entry, ctx);
        ReturnOp::new(ctx, Some(value))
            .get_operation()
            .insert_at_back(entry, ctx);
        address_function.get_operation().insert_at_back(top, ctx);
    }
    let function_type = FuncType::get(
        ctx,
        void.into(),
        vec![
            pointer.into(),
            index.into(),
            pointer.into(),
            pointer.into(),
            pointer.into(),
            pointer.into(),
        ],
        false,
    );
    let kernel = FuncOp::new(ctx, "mixed_maps".try_into().unwrap(), function_type);
    mark_grid_constant(ctx, &kernel, 2, descriptor.into(), 64);
    mark_grid_constant(ctx, &kernel, 4, packed.into(), 1);
    mark_grid_constant(ctx, &kernel, 5, byte.into(), 1);
    let entry = kernel.get_or_create_entry_block(ctx);
    let output = entry.deref(ctx).get_argument(0);
    let map = entry.deref(ctx).get_argument(2);
    let raw = entry.deref(ctx).get_argument(3);
    let packed_map = entry.deref(ctx).get_argument(4);
    let scalar = entry.deref(ctx).get_argument(5);
    let continuation = BasicBlock::new(ctx, None, vec![pointer.into()]);
    continuation.insert_at_back(kernel.get_operation().deref(ctx).get_region(0), ctx);
    BrOp::new(ctx, continuation, vec![map])
        .get_operation()
        .insert_at_back(entry, ctx);
    let merged = continuation.deref(ctx).get_argument(0);
    let helper_type = FuncType::get(ctx, byte.into(), vec![pointer.into()], false);
    let call = CallOp::new(
        ctx,
        CallOpCallable::Direct("read_last_byte".try_into().unwrap()),
        helper_type,
        vec![merged],
    );
    let last = call.get_operation().deref(ctx).get_result(0);
    call.get_operation().insert_at_back(continuation, ctx);
    StoreOp::new(ctx, last, output)
        .get_operation()
        .insert_at_back(continuation, ctx);
    for (slot, input, offset) in [(1, packed_map, 8), (2, scalar, 0), (3, raw, 0)] {
        let source =
            GetElementPtrOp::new(ctx, input, vec![GepIndex::Constant(offset)], byte.into());
        let source_value = source.get_operation().deref(ctx).get_result(0);
        source.get_operation().insert_at_back(continuation, ctx);
        let load = LoadOp::new(ctx, source_value, byte.into());
        let value = load.get_operation().deref(ctx).get_result(0);
        load.get_operation().insert_at_back(continuation, ctx);
        let destination =
            GetElementPtrOp::new(ctx, output, vec![GepIndex::Constant(slot)], byte.into());
        let destination_value = destination.get_operation().deref(ctx).get_result(0);
        destination
            .get_operation()
            .insert_at_back(continuation, ctx);
        StoreOp::new(ctx, value, destination_value)
            .get_operation()
            .insert_at_back(continuation, ctx);
    }
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(continuation, ctx);
    kernel.get_operation().insert_at_back(top, ctx);
    let helper = FuncOp::new(ctx, "read_last_byte".try_into().unwrap(), helper_type);
    let entry = helper.get_or_create_entry_block(ctx);
    let map = entry.deref(ctx).get_argument(0);
    let source = GetElementPtrOp::new(ctx, map, vec![GepIndex::Constant(127)], byte.into());
    let source_value = source.get_operation().deref(ctx).get_result(0);
    source.get_operation().insert_at_back(entry, ctx);
    let load = LoadOp::new(ctx, source_value, byte.into());
    let value = load.get_operation().deref(ctx).get_result(0);
    load.get_operation().insert_at_back(entry, ctx);
    ReturnOp::new(ctx, Some(value))
        .get_operation()
        .insert_at_back(entry, ctx);
    helper.get_operation().insert_at_back(top, ctx);
    module
        .get_operation()
        .deref(ctx)
        .verify(ctx)
        .expect("well-formed LLVM dialect");
    module
}

#[test]
fn grid_constant_legacy_preserves_pointees_in_all_symbol_references_and_erases_body_pointers() {
    let mut ctx = Context::new();
    let module = mixed_module(&mut ctx, false);
    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .unwrap();
    let signature = "void (i8*, i64, [128 x i8]*, i8*, <{ i8, [2 x i32] }>*, i8*)* @mixed_maps";
    assert_eq!(
        ir.matches(signature).count(),
        3,
        "llvm.used, kernel annotation and grid annotation:\n{ir}"
    );
    assert!(ir.contains("[128 x i8]* byval align 64 %v2.byval"), "{ir}");
    assert!(
        ir.contains("<{ i8, [2 x i32] }>* byval align 1 %v4.byval"),
        "{ir}"
    );
    assert!(ir.contains("i8* byval align 1 %v5"), "{ir}");
    assert!(
        ir.contains("%v2 = bitcast [128 x i8]* %v2.byval to i8*"),
        "{ir}"
    );
    assert!(
        ir.contains("%v4 = bitcast <{ i8, [2 x i32] }>* %v4.byval to i8*"),
        "{ir}"
    );
    assert!(
        !ir.contains("%v5.byval"),
        "an i8 pointee needs no cast:\n{ir}"
    );
    assert!(ir.contains("phi i8* [ %v2, %entry ]"), "{ir}");
    assert!(ir.contains("call i8 @read_last_byte(i8*"), "{ir}");
    assert!(ir.contains("!{i32 3, i32 5, i32 6}"), "{ir}");
    assert_eq!(
        ir.matches(" byval").count(),
        3,
        "ordinary pointers must retain pointer ABI:\n{ir}"
    );
}

#[test]
fn grid_constant_modern_preserves_opaque_pointer_body_and_explicit_byval_types() {
    let mut ctx = Context::new();
    let module = mixed_module(&mut ctx, false);
    let ir =
        export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default()).unwrap();
    assert!(ir.contains("ptr byval([128 x i8]) align 64 %v2"), "{ir}");
    assert!(
        ir.contains("ptr byval(<{ i8, [2 x i32] }>) align 1 %v4"),
        "{ir}"
    );
    assert!(ir.contains("ptr byval(i8) align 1 %v5"), "{ir}");
    assert!(
        !ir.contains(".byval"),
        "opaque pointer bodies need no adapter:\n{ir}"
    );
    assert!(ir.contains("phi ptr [ %v2, %entry ]"), "{ir}");
    assert!(ir.contains("!{i32 3, i32 5, i32 6}"), "{ir}");
}

#[test]
fn grid_constant_legacy_compiles_to_exact_ptx_parameter_storage() {
    let nvvm = match libnvvm_sys::LibNvvm::load() {
        Ok(nvvm) => nvvm,
        Err(libnvvm_sys::NvvmError::LibraryNotFound { .. }) => {
            eprintln!("skipping libNVVM PTX gate: CUDA Toolkit not installed");
            return;
        }
        Err(error) => panic!("load libNVVM: {error}"),
    };
    let mut ctx = Context::new();
    let module = mixed_module(&mut ctx, false);
    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .unwrap();
    // LLVM-level noinline guarantees a surviving device call; a source Rust
    // #[inline(never)] alone does not constrain later LLVM/NVVM inlining.
    for noinline in [false, true] {
        let ir = if noinline {
            let signature = "define i8 @read_last_byte(i8* %v0) #0";
            assert!(ir.contains(signature), "{ir}");
            ir.replace(signature, "define i8 @read_last_byte(i8* %v0) noinline #0")
        } else {
            ir.clone()
        };
        let mut program = libnvvm_sys::Program::new(&nvvm).unwrap();
        program
            .add_module(ir.as_bytes(), "grid_constants.ll")
            .unwrap();
        let ptx = String::from_utf8(program.compile(&["-arch=compute_80"]).unwrap()).unwrap();
        assert!(
            ptx.contains(".param .align 64 .b8 mixed_maps_param_2[128]"),
            "{ptx}"
        );
        assert!(
            ptx.contains(".param .align 1 .b8 mixed_maps_param_4[9]"),
            "{ptx}"
        );
        assert!(
            ptx.contains(".param .align 1 .b8 mixed_maps_param_5[1]"),
            "{ptx}"
        );
        assert!(
            ptx.contains(".param .u64 mixed_maps_param_3"),
            "raw pointer control:\n{ptx}"
        );
        if noinline {
            assert!(
                ptx.contains("call.uni"),
                "helper must remain a PTX call:\n{ptx}"
            );
            assert!(
                ptx.contains("read_last_byte,"),
                "the surviving call must target the descriptor helper:\n{ptx}"
            );
        } else {
            assert!(
                ptx.contains("mixed_maps_param_2+127"),
                "read the last descriptor byte:\n{ptx}"
            );
        }
        assert!(
            ptx.contains("mixed_maps_param_4+8"),
            "read the last packed byte:\n{ptx}"
        );
        assert!(
            !ptx.contains(".local"),
            "grid parameters must not become thread-local copies:\n{ptx}"
        );
    }
}

#[test]
fn grid_constant_declaration_retains_byval_pointee() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "declaration".try_into().unwrap());
    let top = module_top_block(&mut ctx, &module);
    let void = VoidType::get(&ctx);
    let pointer = PointerType::get(&ctx, 0);
    let byte = IntegerType::get(&ctx, 8, Signedness::Signless);
    let array = ArrayType::get(&ctx, byte.into(), 128);
    let function_type = FuncType::get(&ctx, void.into(), vec![pointer.into()], false);
    let function = FuncOp::new(&mut ctx, "external_map".try_into().unwrap(), function_type);
    mark_grid_constant(&ctx, &function, 0, array.into(), 64);
    function.get_operation().insert_at_back(top, &ctx);
    let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
    let ir = export_module_to_string_with_config(&ctx, &module, &config).unwrap();
    assert!(
        ir.contains("declare void @external_map([128 x i8]* byval align 64)"),
        "{ir}"
    );
    assert!(ir.contains("void ([128 x i8]*)* @external_map"), "{ir}");
}

#[test]
fn grid_constant_declaration_cannot_be_replaced_by_an_erased_shape_device_extern() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "extern_launch_abi_conflict".try_into().unwrap());
    let top = module_top_block(&mut ctx, &module);
    let void = VoidType::get(&ctx);
    let pointer = PointerType::get(&ctx, 0);
    let byte = IntegerType::get(&ctx, 8, Signedness::Signless);
    let descriptor = ArrayType::get(&ctx, byte.into(), 128);
    let function_type = FuncType::get(&ctx, void.into(), vec![pointer.into()], false);
    let declaration = FuncOp::new(&mut ctx, "external_map".try_into().unwrap(), function_type);
    mark_grid_constant(&ctx, &declaration, 0, descriptor.into(), 64);
    declaration.get_operation().insert_at_back(top, &ctx);
    // The external declaration has the same erased pointer shape, but its
    // ordinary pointer argument cannot replace 128 bytes of launch storage.
    let externs = [DeviceExternDecl {
        export_name: "external_map".into(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 0)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_with_externs(&ctx, &module, &externs, &NvvmExportConfig::new(dialect))
                .expect_err("a device extern must not suppress a kernel launch declaration");
        assert!(
            error.contains(
                "device extern `@external_map` conflicts with a grid-constant kernel declaration"
            ),
            "{error}"
        );
        assert!(
            error.contains("launch ABI is not an ordinary device function ABI"),
            "{error}"
        );
    }
}

#[test]
fn grid_constant_rejects_out_of_range_parameter_metadata() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_metadata".try_into().unwrap());
    let top = module_top_block(&mut ctx, &module);
    let void = VoidType::get(&ctx);
    let byte = IntegerType::get(&ctx, 8, Signedness::Signless);
    let function_type = FuncType::get(&ctx, void.into(), vec![], false);
    let function = FuncOp::new(&mut ctx, "invalid_map".try_into().unwrap(), function_type);
    mark_grid_constant(&ctx, &function, 1, byte.into(), 1);
    function.get_operation().insert_at_back(top, &ctx);
    let error = export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default())
        .unwrap_err();
    assert!(
        error.contains("parameter index 1 is out of range for 0 parameters"),
        "{error}"
    );
}

#[test]
fn grid_constant_rejects_function_address_even_before_kernel_definition() {
    let mut ctx = Context::new();
    let module = mixed_module(&mut ctx, true);
    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .unwrap_err();
        assert!(error.contains("cannot take the device function address of grid-constant kernel entry `@mixed_maps`"), "{error}");
    }
}

#[test]
fn grid_constant_rejects_device_calls_to_kernel_entry() {
    let mut ctx = Context::new();
    let module = mixed_module(&mut ctx, false);
    let top = module_top_block(&mut ctx, &module);
    let function_type = top
        .deref(&ctx)
        .iter(&ctx)
        .find_map(|operation| {
            let function = Operation::get_op::<FuncOp>(operation, &ctx)?;
            (function.get_symbol_name(&ctx).as_ref() == "mixed_maps")
                .then(|| function.get_type(&ctx))
        })
        .unwrap();
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), function_type);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let arguments = entry.deref(&ctx).arguments().collect();
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("mixed_maps".try_into().unwrap()),
        function_type,
        arguments,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(top, &ctx);
    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .unwrap_err();
        assert!(
            error.contains(
                "grid-constant kernel entry `@mixed_maps` cannot be called as a device function"
            ),
            "{error}"
        );
    }
}

#[test]
fn grid_constant_rejects_invalid_parameter_contracts() {
    for (case, expected) in [
        ("false_kernel", "non-kernel function"),
        ("non_pointer", "not a pointer"),
        ("specific_space", "generic address space 0"),
        ("zero_size", "sized, non-zero pointee"),
        ("void", "sized, non-zero pointee"),
        ("incomplete", "incomplete grid-constant metadata"),
        ("bad_alignment", "non-zero power of two"),
        ("wrong_attribute_kind", "malformed grid-constant attribute"),
        ("noncanonical_index", "malformed grid-constant attribute"),
    ] {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "invalid_contract".try_into().unwrap());
        let top = module_top_block(&mut ctx, &module);
        let void = VoidType::get(&ctx);
        let byte = IntegerType::get(&ctx, 8, Signedness::Signless);
        let argument: TypeHandle = match case {
            "non_pointer" => byte.into(),
            "specific_space" => PointerType::get(&ctx, 1).into(),
            _ => PointerType::get(&ctx, 0).into(),
        };
        let pointee = match case {
            "zero_size" => ArrayType::get(&ctx, byte.into(), 0).into(),
            "void" => void.into(),
            _ => byte.into(),
        };
        let function_type = FuncType::get(&ctx, void.into(), vec![argument], false);
        let function = FuncOp::new(&mut ctx, "invalid_map".try_into().unwrap(), function_type);
        mark_grid_constant(
            &ctx,
            &function,
            0,
            pointee,
            if case == "bad_alignment" { 3 } else { 1 },
        );
        {
            let attrs = &mut function.get_operation().deref_mut(&ctx).attributes;
            match case {
                "false_kernel" => {
                    attrs.set(
                        "gpu_kernel".try_into().unwrap(),
                        StringAttr::new("false".into()),
                    );
                }
                "incomplete" => {
                    attrs.0.remove(
                        &format!("{LLVM_GRID_CONSTANT_ALIGN_ATTR_PREFIX}0")
                            .as_str()
                            .try_into()
                            .unwrap(),
                    );
                }
                "wrong_attribute_kind" => {
                    attrs.set(
                        format!("{LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX}0")
                            .as_str()
                            .try_into()
                            .unwrap(),
                        StringAttr::new("i8".into()),
                    );
                }
                "noncanonical_index" => {
                    attrs.set(
                        format!("{LLVM_GRID_CONSTANT_POINTEE_ATTR_PREFIX}00")
                            .as_str()
                            .try_into()
                            .unwrap(),
                        TypeAttr::new(pointee),
                    );
                }
                _ => {}
            }
        }
        function.get_operation().insert_at_back(top, &ctx);
        for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
            let error =
                export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                    .unwrap_err();
            assert!(error.contains(expected), "{case}: {error}");
        }
    }
}
