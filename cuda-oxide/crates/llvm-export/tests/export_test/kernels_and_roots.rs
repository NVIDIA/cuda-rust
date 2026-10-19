/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        DeviceExternDecl, NvvmExportConfig, NvvmIrDialect, PtxExportConfig,
        export_module_to_string, export_module_to_string_with_config,
        export_module_with_externs_and_roots,
    },
    ops::{FuncOp, GlobalOp, GlobalOpExt, ReturnOp},
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::StringAttr,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    op::Op,
};

use crate::common::module_top_block;

#[test]
fn legacy_kernel_metadata_uses_typed_function_references() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_metadata".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "metadata_kernel".try_into().unwrap(), func_ty);
    func.get_operation().deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("legacy metadata export succeeds");
    assert!(
        ir.contains(
            "@llvm.used = appending global [1 x i8*] [i8* bitcast (void (i8*)* @metadata_kernel to i8*)]"
        ),
        "{ir}"
    );
    assert!(
        ir.contains("!{void (i8*)* @metadata_kernel, !\"kernel\", i32 1}"),
        "{ir}"
    );
}

#[test]
fn llvm_used_roots_only_explicitly_retained_globals() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "retained_globals".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    let retained = GlobalOp::new(&mut ctx, "IKET_META".try_into().unwrap(), i32_ty.into());
    retained.set_address_space(&mut ctx, 1);
    retained.mark_retained(&mut ctx);
    retained.get_operation().insert_at_back(module_block, &ctx);

    let ordinary = GlobalOp::new(&mut ctx, "ORDINARY".try_into().unwrap(), i32_ty.into());
    ordinary.set_address_space(&mut ctx, 1);
    ordinary.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(&ctx, &module, &PtxExportConfig)
        .expect("retained global export succeeds");
    assert!(
        ir.contains(
            "@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @IKET_META to ptr)], section \"llvm.metadata\""
        ),
        "{ir}"
    );
    assert!(!ir.contains("@ORDINARY to ptr"), "{ir}");
}

#[test]
fn legacy_llvm_used_roots_retained_address_space_global() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_retained_global".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let retained = GlobalOp::new(&mut ctx, "IKET_META".try_into().unwrap(), i32_ty.into());
    retained.set_address_space(&mut ctx, 1);
    retained.mark_retained(&mut ctx);
    retained.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("legacy retained global export succeeds");
    assert!(
        ir.contains(
            "@llvm.used = appending global [1 x i8*] [i8* addrspacecast (i32 addrspace(1)* @IKET_META to i8*)], section \"llvm.metadata\""
        ),
        "{ir}"
    );
}

#[test]
fn ptx_export_records_kernel_roots_for_internalization() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "ptx_roots".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(&mut ctx, "COEFFS".try_into().unwrap(), i32_ty.into());
    global.set_address_space(&mut ctx, 4);
    global.get_operation().insert_at_back(module_block, &ctx);
    let func_ty = FuncType::get(&ctx, VoidType::get(&ctx).into(), vec![], false);
    let func = FuncOp::new(&mut ctx, "entry_kernel".try_into().unwrap(), func_ty);
    func.get_operation().deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let exported = export_module_with_externs_and_roots::<DeviceExternDecl>(
        &ctx,
        &module,
        &[],
        &PtxExportConfig,
    )
    .expect("PTX export succeeds");
    let ir = exported.llvm_ir;
    assert!(
        ir.contains(
            "@llvm.used = appending global [1 x ptr] [ptr @entry_kernel], section \"llvm.metadata\""
        ),
        "{ir}"
    );
    assert_eq!(exported.public_symbols, ["COEFFS", "entry_kernel"]);
}

#[test]
fn ptx_export_records_standalone_device_function_roots_for_internalization() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "ptx_device_root".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let func_ty = FuncType::get(&ctx, VoidType::get(&ctx).into(), vec![], false);
    let prefixed_name = format!(
        "{}standalone_export",
        reserved_oxide_symbols::LEGACY_DEVICE_PREFIX
    );
    let func = FuncOp::new(
        &mut ctx,
        prefixed_name.as_str().try_into().unwrap(),
        func_ty,
    );
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let exported = export_module_with_externs_and_roots::<DeviceExternDecl>(
        &ctx,
        &module,
        &[],
        &PtxExportConfig,
    )
    .expect("standalone PTX export succeeds");
    assert_eq!(exported.public_symbols, ["standalone_export"]);
    assert!(
        exported.llvm_ir.contains(
            "@llvm.used = appending global [1 x ptr] [ptr @standalone_export], section \"llvm.metadata\""
        ),
        "{}",
        exported.llvm_ir
    );
}

/// Builds a `void` function taking pointer parameters in the given address
/// spaces, with an empty body, optionally carrying the `gpu_kernel` marker
/// the exporter uses to distinguish an entry from a device function.
fn pointer_param_function(
    ctx: &mut Context,
    module_block: Ptr<BasicBlock>,
    name: &str,
    address_spaces: &[u32],
    is_kernel: bool,
) {
    let params: Vec<_> = address_spaces
        .iter()
        .map(|space| PointerType::get(ctx, *space).into())
        .collect();
    let void_ty = VoidType::get(ctx);
    let func_ty = FuncType::get(ctx, void_ty.into(), params, false);
    let func = FuncOp::new(ctx, name.try_into().unwrap(), func_ty);
    if is_kernel {
        let kernel_key: Identifier = "gpu_kernel".try_into().unwrap();
        func.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(kernel_key, StringAttr::new("true".to_string()));
    }
    let entry = func.get_or_create_entry_block(ctx);
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(entry, ctx);
    func.get_operation().insert_at_back(module_block, ctx);
}

/// A kernel receives its parameters in `.param` space from the host, which
/// holds no shared-memory address to pass. The exporter must refuse the
/// signature instead of emitting `.ptr .shared`, which ptxas accepts but the
/// driver rejects at module load, taking every kernel in the module down.
#[test]
fn kernel_rejects_a_shared_memory_pointer_parameter() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "kernel_shared_param".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    pointer_param_function(&mut ctx, module_block, "shared_param", &[0, 3], true);

    let error = export_module_to_string(&ctx, &module)
        .expect_err("export must refuse a kernel parameter in shared memory");
    assert!(
        error.contains(
            "kernel `@shared_param` parameter 1 is a pointer into shared memory (address space 3)"
        ),
        "{error}"
    );
}

/// Local memory is per-thread, so the host has no address in it to pass
/// either; the exporter refuses it on the same grounds as shared.
#[test]
fn kernel_rejects_a_local_memory_pointer_parameter() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "kernel_local_param".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    pointer_param_function(&mut ctx, module_block, "local_param", &[5], true);

    let error = export_module_to_string(&ctx, &module)
        .expect_err("export must refuse a kernel parameter in local memory");
    assert!(
        error.contains(
            "kernel `@local_param` parameter 0 is a pointer into local memory (address space 5)"
        ),
        "{error}"
    );
}

/// The refusal is scoped to what the host cannot supply. Generic, global,
/// and constant pointer parameters are host-addressable and stay accepted on
/// kernels, and a device (non-kernel) function may carry any state space on
/// its parameters, shared included.
#[test]
fn kernel_keeps_host_addressable_pointer_parameters_and_device_functions_keep_shared() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "kernel_allowed_params".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    pointer_param_function(&mut ctx, module_block, "allowed_param", &[0, 1, 4], true);
    pointer_param_function(&mut ctx, module_block, "device_shared", &[3], false);

    let ir = export_module_to_string(&ctx, &module)
        .expect("generic/global/constant kernel parameters and device shared must export");
    assert!(
        ir.contains(
            "define ptx_kernel void @allowed_param(ptr %v0, ptr addrspace(1) %v1, ptr addrspace(4) %v2)"
        ),
        "{ir}"
    );
    assert!(
        ir.contains("define void @device_shared(ptr addrspace(3) %v0)"),
        "{ir}"
    );
}
