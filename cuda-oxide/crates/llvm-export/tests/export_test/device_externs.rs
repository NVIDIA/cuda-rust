/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        DeviceExternAttrs, DeviceExternDecl, DeviceExternType, NvvmExportConfig, NvvmIrDialect,
        export_module_with_externs,
    },
    ops::{AddressOfOp, CallOp, FuncOp, ReturnOp},
    types::{FuncType, HalfType, PointerType, VoidType},
};
use pliron::{
    builtin::{
        op_interfaces::CallOpCallable,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    op::Op,
};

use crate::common::module_top_block;

#[test]
fn legacy_device_extern_adapts_exact_pointer_arguments_and_results() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_extern".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let ptr_ty = PointerType::get(&ctx, 0);
    let external_ty = FuncType::get(&ctx, ptr_ty.into(), vec![ptr_ty.into()], false);
    FuncOp::new(&mut ctx, "float_roundtrip".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&ctx);
    let caller_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), caller_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("float_roundtrip".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "float_roundtrip".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 0)],
        return_type: DeviceExternType::pointer_to(DeviceExternType::Float32, 0),
        attrs: DeviceExternAttrs::default(),
    }];
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy extern export succeeds");

    assert!(
        ir.contains("declare float* @float_roundtrip(float*)"),
        "{ir}"
    );
    assert!(ir.contains("bitcast i8* %v0 to float*"), "{ir}");
    assert!(ir.contains("call float* @float_roundtrip(float*"), "{ir}");
    assert!(
        ir.contains(" = bitcast float* ") && ir.contains(" to i8*"),
        "{ir}"
    );
    assert!(
        !ir.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|token| token == "ptr"),
        "{ir}"
    );
}

#[test]
fn legacy_device_extern_preserves_pointer_address_spaces() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_extern_as".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let external_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    FuncOp::new(&mut ctx, "shared_float".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), external_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("shared_float".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "shared_float".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 3)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy extern export succeeds");
    assert!(
        ir.contains("declare void @shared_float(float addrspace(3)*)"),
        "{ir}"
    );
    assert!(
        ir.contains("bitcast i8 addrspace(3)* %v0 to float addrspace(3)*"),
        "{ir}"
    );
}

#[test]
fn modern_device_extern_erases_pointee_without_boundary_casts() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "modern_extern".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let external_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    FuncOp::new(&mut ctx, "takes_float".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), external_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("takes_float".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "takes_float".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 0)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern extern export succeeds");
    assert!(ir.contains("declare void @takes_float(ptr)"), "{ir}");
    assert!(ir.contains("call void @takes_float(ptr %v0)"), "{ir}");
    assert!(!ir.contains("bitcast"), "{ir}");
}

#[test]
fn device_extern_rejects_invalid_symbol_and_address_space_mismatch() {
    let mut ctx = Context::new();
    let empty = ModuleOp::new(&mut ctx, "empty".try_into().unwrap());
    let invalid = [DeviceExternDecl {
        export_name: "bad.name".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &invalid,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("invalid NVVM symbol must fail");
    assert!(err.contains("global-identifier subset"), "{err}");

    let reserved_intrinsic_prefix = [DeviceExternDecl {
        export_name: "llvm_external".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &reserved_intrinsic_prefix,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("the reserved intrinsic namespace must not be ambiguous");
    assert!(err.contains("reserves for LLVM intrinsics"), "{err}");

    let by_value_array = [DeviceExternDecl {
        export_name: "array_by_value".to_string(),
        param_types: vec![DeviceExternType::Array {
            element: Box::new(DeviceExternType::Float32),
            len: 4,
        }],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &by_value_array,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("by-value aggregate externs must be rejected");
    assert!(err.contains("passes an array by value"), "{err}");

    let nested_half = [DeviceExternDecl {
        export_name: "half_buffer".to_string(),
        param_types: vec![DeviceExternType::pointer_to(
            DeviceExternType::Array {
                element: Box::new(DeviceExternType::Float16),
                len: 4,
            },
            0,
        )],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &nested_half,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("legacy half nested in a pointer must fail");
    assert!(
        err.contains("CUDA 12 legacy") && err.contains("half"),
        "{err}"
    );
    let modern = export_module_with_externs(
        &ctx,
        &empty,
        &nested_half,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern opaque-pointer extern may use half pointees");
    assert!(
        modern.contains("declare void @half_buffer(ptr)"),
        "{modern}"
    );

    let module = ModuleOp::new(&mut ctx, "mismatch".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr0 = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let external_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr0.into()], false);
    FuncOp::new(&mut ctx, "shared_only".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), external_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("shared_only".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);
    let mismatch = [DeviceExternDecl {
        export_name: "shared_only".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 3)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &module,
        &mismatch,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("address-space mismatch must fail");
    assert!(
        err.contains("parameter, result, or pointer address-space types"),
        "{err}"
    );
}

#[test]
fn device_extern_rejects_same_name_declaration_shape_without_a_call() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "extern_decl_conflict".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let lowered_type = FuncType::get(&ctx, void_ty.into(), vec![i32_ty.into()], false);
    FuncOp::new(
        &mut ctx,
        "conflicting_decl".try_into().unwrap(),
        lowered_type,
    )
    .get_operation()
    .insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "conflicting_decl".to_string(),
        param_types: vec![DeviceExternType::Float32],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let error = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("the exporter must independently reject a conflicting declaration");
    assert!(
        error.contains("parameter, result, or pointer address-space types"),
        "{error}"
    );
}

#[test]
fn device_extern_rejects_definition_and_address_taken_shape_conflicts() {
    let mut ctx = Context::new();
    let void_ty = VoidType::get(&ctx);

    let definition_module = ModuleOp::new(&mut ctx, "extern_definition".try_into().unwrap());
    let definition_block = module_top_block(&mut ctx, &definition_module);
    let no_args = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let definition = FuncOp::new(&mut ctx, "defined_external".try_into().unwrap(), no_args);
    let definition_entry = definition.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(definition_entry, &ctx);
    definition
        .get_operation()
        .insert_at_back(definition_block, &ctx);
    let definition_extern = [DeviceExternDecl {
        export_name: "defined_external".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let error = export_module_with_externs(
        &ctx,
        &definition_module,
        &definition_extern,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("a side-table extern must not collide with a definition");
    assert!(error.contains("function definition"), "{error}");

    let address_module = ModuleOp::new(&mut ctx, "extern_address".try_into().unwrap());
    let address_block = module_top_block(&mut ctx, &address_module);
    let generic_pointer = PointerType::get(&ctx, 0);
    let lowered_type = FuncType::get(&ctx, void_ty.into(), vec![generic_pointer.into()], false);
    FuncOp::new(
        &mut ctx,
        "addressed_external".try_into().unwrap(),
        lowered_type,
    )
    .get_operation()
    .insert_at_back(address_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), lowered_type);
    let caller_entry = caller.get_or_create_entry_block(&mut ctx);
    let argument = caller_entry.deref(&ctx).get_argument(0);
    let address = AddressOfOp::new(&mut ctx, "addressed_external".try_into().unwrap(), 0);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(caller_entry, &ctx);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(address_value),
        lowered_type,
        vec![argument],
    )
    .get_operation()
    .insert_at_back(caller_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(caller_entry, &ctx);
    caller.get_operation().insert_at_back(address_block, &ctx);
    let address_extern = [DeviceExternDecl {
        export_name: "addressed_external".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 3)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let error = export_module_with_externs(
        &ctx,
        &address_module,
        &address_extern,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("address-taking must not bypass exact extern shape validation");
    assert!(
        error.contains("parameter, result, or pointer address-space types"),
        "{error}"
    );
}

#[test]
fn modern_device_extern_emits_signext_zeroext_for_small_integer_params() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "small_types_extern".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    // The extern takes (i8 signext, i16 zeroext, i1 zeroext, half) and
    // returns void. The declared IR types stay NARROW (matching cuda-oxide's
    // own i8/i16/i1 SSA values and the clang-compiled LTOIR definition); the
    // ABI extension attributes live only in the DeviceExternType metadata.
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i16_ty = IntegerType::get(&ctx, 16, Signedness::Signless);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let half_ty = HalfType::get(&ctx);
    let void_ty = VoidType::get(&ctx);

    let external_ty = FuncType::get(
        &ctx,
        void_ty.into(),
        vec![i8_ty.into(), i16_ty.into(), i1_ty.into(), half_ty.into()],
        false,
    );
    FuncOp::new(&mut ctx, "small_types_fn".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);

    // Caller that forwards its own narrow parameters to the extern with no
    // value conversions.
    let caller_ty = FuncType::get(
        &ctx,
        void_ty.into(),
        vec![i8_ty.into(), i16_ty.into(), i1_ty.into(), half_ty.into()],
        false,
    );
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), caller_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let arg0 = entry.deref(&ctx).get_argument(0);
    let arg1 = entry.deref(&ctx).get_argument(1);
    let arg2 = entry.deref(&ctx).get_argument(2);
    let arg3 = entry.deref(&ctx).get_argument(3);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("small_types_fn".try_into().unwrap()),
        external_ty,
        vec![arg0, arg1, arg2, arg3],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "small_types_fn".to_string(),
        param_types: vec![
            DeviceExternType::SignExtInteger(8), // i8, sign-extended by NVPTX
            DeviceExternType::ZeroExtInteger(16), // u16, zero-extended by NVPTX
            DeviceExternType::ZeroExtInteger(1), // bool
            DeviceExternType::Float16,           // f16 as native half
        ],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];

    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern extern with small types export succeeds");

    // Declaration: narrow types with parameter-position attributes.
    assert!(
        ir.contains("declare void @small_types_fn(i8 signext, i16 zeroext, i1 zeroext, half)"),
        "declaration should keep narrow types with signext/zeroext attributes:\n{ir}"
    );

    // Call site keeps the narrow types and attributes too.
    assert!(
        ir.contains("i8 signext %v0"),
        "call should use signext on first arg:\n{ir}"
    );
    assert!(
        ir.contains("i16 zeroext %v1"),
        "call should use zeroext on second arg:\n{ir}"
    );
    assert!(
        ir.contains("i1 zeroext %v2"),
        "call should use zeroext on the bool arg:\n{ir}"
    );
    assert!(
        ir.contains("half %v3"),
        "call should use half for the f16 arg:\n{ir}"
    );
}

#[test]
fn modern_device_extern_signext_return_type() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "signext_return".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);

    // Extern returns i8 with signext. The declared type stays i8; only the
    // attribute marks the NVPTX widening, and in return position LLVM's
    // grammar requires the attribute BEFORE the type.
    let external_ty = FuncType::get(&ctx, i8_ty.into(), vec![], false);
    FuncOp::new(&mut ctx, "get_small_val".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);

    // Caller that calls the extern and discards the result.
    let caller_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), caller_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("get_small_val".try_into().unwrap()),
        external_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "get_small_val".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::SignExtInteger(8),
        attrs: DeviceExternAttrs::default(),
    }];

    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern extern with signext return export succeeds");

    // Declaration: attribute precedes the narrow return type.
    assert!(
        ir.contains("declare signext i8 @get_small_val()"),
        "declaration should have signext BEFORE the return type:\n{ir}"
    );

    // Call site uses the same return-position placement.
    assert!(
        ir.contains("call signext i8 @get_small_val()"),
        "call should have signext BEFORE the return type:\n{ir}"
    );
}

#[test]
fn plain_integer_device_extern_has_no_extension_attributes() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "plain_int".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);

    let external_ty = FuncType::get(&ctx, void_ty.into(), vec![i32_ty.into()], false);
    FuncOp::new(&mut ctx, "plain_int_fn".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);

    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), external_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let arg0 = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("plain_int_fn".try_into().unwrap()),
        external_ty,
        vec![arg0],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "plain_int_fn".to_string(),
        param_types: vec![DeviceExternType::Integer(32)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];

    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("plain int extern export succeeds");

    // Plain i32 should have no signext/zeroext.
    assert!(
        ir.contains("declare void @plain_int_fn(i32)"),
        "declaration should have plain i32 without attributes:\n{ir}"
    );
    assert!(
        !ir.contains("signext") && !ir.contains("zeroext"),
        "plain i32 should have no extension attributes:\n{ir}"
    );
}

#[test]
fn legacy_device_extern_rejects_small_integers_by_value() {
    let mut ctx = Context::new();
    let empty = ModuleOp::new(&mut ctx, "empty".try_into().unwrap());
    let externs = [DeviceExternDecl {
        export_name: "takes_small".to_string(),
        param_types: vec![DeviceExternType::SignExtInteger(8)],
        return_type: DeviceExternType::ZeroExtInteger(16),
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("legacy sub-32-bit by-value externs must fail cleanly");
    assert!(
        err.contains("CUDA 12 legacy") && err.contains("sub-32-bit"),
        "{err}"
    );
}

/// Build a module whose externs carry small params AND a small return, in
/// both declare and call positions, so the emitted attribute placement can
/// be checked against a real LLVM parser.
fn small_type_extern_module(ctx: &mut Context) -> (ModuleOp, Vec<DeviceExternDecl>) {
    let module = ModuleOp::new(ctx, "small_types_parse_gate".try_into().unwrap());
    let module_block = module_top_block(ctx, &module);

    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    let i16_ty = IntegerType::get(ctx, 16, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let half_ty = HalfType::get(ctx);
    let void_ty = VoidType::get(ctx);

    let take_small_ty = FuncType::get(
        ctx,
        void_ty.into(),
        vec![i8_ty.into(), i16_ty.into(), i1_ty.into(), half_ty.into()],
        false,
    );
    FuncOp::new(ctx, "take_small".try_into().unwrap(), take_small_ty)
        .get_operation()
        .insert_at_back(module_block, &*ctx);

    let give_small_ty = FuncType::get(ctx, i8_ty.into(), vec![], false);
    FuncOp::new(ctx, "give_small".try_into().unwrap(), give_small_ty)
        .get_operation()
        .insert_at_back(module_block, &*ctx);

    let caller_ty = FuncType::get(
        ctx,
        void_ty.into(),
        vec![i8_ty.into(), i16_ty.into(), i1_ty.into(), half_ty.into()],
        false,
    );
    let caller = FuncOp::new(ctx, "caller".try_into().unwrap(), caller_ty);
    let entry = caller.get_or_create_entry_block(ctx);
    let arg0 = entry.deref(ctx).get_argument(0);
    let arg1 = entry.deref(ctx).get_argument(1);
    let arg2 = entry.deref(ctx).get_argument(2);
    let arg3 = entry.deref(ctx).get_argument(3);
    CallOp::new(
        ctx,
        CallOpCallable::Direct("take_small".try_into().unwrap()),
        take_small_ty,
        vec![arg0, arg1, arg2, arg3],
    )
    .get_operation()
    .insert_at_back(entry, &*ctx);
    CallOp::new(
        ctx,
        CallOpCallable::Direct("give_small".try_into().unwrap()),
        give_small_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(entry, &*ctx);
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(entry, &*ctx);
    caller.get_operation().insert_at_back(module_block, &*ctx);

    let externs = vec![
        DeviceExternDecl {
            export_name: "take_small".to_string(),
            param_types: vec![
                DeviceExternType::SignExtInteger(8),
                DeviceExternType::ZeroExtInteger(16),
                DeviceExternType::ZeroExtInteger(1),
                DeviceExternType::Float16,
            ],
            return_type: DeviceExternType::Void,
            attrs: DeviceExternAttrs::default(),
        },
        DeviceExternDecl {
            export_name: "give_small".to_string(),
            param_types: vec![],
            return_type: DeviceExternType::SignExtInteger(8),
            attrs: DeviceExternAttrs::default(),
        },
    ];
    (module, externs)
}

/// Parse gate: the emitted attribute placement (`i8 signext` in parameter
/// position, `signext i8` in return position, for both `declare` and `call`)
/// must be accepted by a real LLVM parser, not just string-matched.
#[test]
fn small_type_extern_module_parses_with_llvm_as() {
    let mut ctx = Context::new();
    let (module, externs) = small_type_extern_module(&mut ctx);
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern small-type extern export succeeds");

    // Belt-and-braces string checks before the external parse.
    assert!(
        ir.contains("declare void @take_small(i8 signext, i16 zeroext, i1 zeroext, half)"),
        "{ir}"
    );
    assert!(ir.contains("declare signext i8 @give_small()"), "{ir}");
    assert!(ir.contains("call signext i8 @give_small()"), "{ir}");

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
        "cuda_oxide_small_type_parse_gate_{}.ll",
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
        output.status.success(),
        "{llvm_as} rejected the emitted module:\n{stderr}\n--- module ---\n{ir}"
    );
}
