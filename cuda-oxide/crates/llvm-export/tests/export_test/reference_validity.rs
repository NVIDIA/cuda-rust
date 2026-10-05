/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{
        NvvmExportConfig, NvvmIrDialect, PtxExportConfig, export_module_to_string_with_config,
    },
    ops::{
        FuncOp, KernelReferenceParamValidityAttr, ReturnOp, set_kernel_reference_param_validity,
    },
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    builtin::{
        attributes::StringAttr,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    op::Op,
};

use crate::common::module_top_block;

fn kernel_with_params(
    ctx: &mut Context,
    name: &str,
    params: Vec<pliron::r#type::TypeHandle>,
    is_kernel: bool,
) -> FuncOp {
    let func_ty = FuncType::get(ctx, VoidType::get(ctx).into(), params, false);
    let func = FuncOp::new(ctx, name.try_into().unwrap(), func_ty);
    if is_kernel {
        func.get_operation().deref_mut(ctx).attributes.set(
            "gpu_kernel".try_into().unwrap(),
            StringAttr::new("true".to_string()),
        );
    }
    let entry = func.get_or_create_entry_block(ctx);
    ReturnOp::new(ctx, None)
        .get_operation()
        .insert_at_back(entry, ctx);
    func
}

#[test]
fn kernel_reference_validity_exports_nonnull_and_alignment() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "reference_validity".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr = PointerType::get(&ctx, 0);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let func = kernel_with_params(
        &mut ctx,
        "reference_attrs",
        vec![ptr.into(), ptr.into(), i32_ty.into()],
        true,
    );
    set_kernel_reference_param_validity(
        &mut ctx,
        func.get_operation(),
        0,
        KernelReferenceParamValidityAttr(4),
    );
    set_kernel_reference_param_validity(
        &mut ctx,
        func.get_operation(),
        1,
        KernelReferenceParamValidityAttr(1),
    );
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(&ctx, &module, &PtxExportConfig)
        .expect("reference validity export succeeds");
    assert!(
        ir.contains(
            "define ptx_kernel void @reference_attrs(ptr nonnull align 4 %v0, ptr nonnull %v1, i32 %v2)"
        ),
        "{ir}"
    );
}

#[test]
fn legacy_kernel_reference_validity_follows_typed_pointer_spelling() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_reference_validity".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr = PointerType::get(&ctx, 0);
    let func = kernel_with_params(&mut ctx, "legacy_reference_attrs", vec![ptr.into()], true);
    set_kernel_reference_param_validity(
        &mut ctx,
        func.get_operation(),
        0,
        KernelReferenceParamValidityAttr(4),
    );
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("legacy reference validity export succeeds");
    assert!(ir.contains("i8* nonnull align 4 %v0"), "{ir}");
}

#[test]
fn reference_validity_rejects_non_kernel_non_pointer_and_out_of_range() {
    let mut ctx = Context::new();
    let ptr = PointerType::get(&ctx, 0);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    let non_kernel_module = ModuleOp::new(&mut ctx, "non_kernel".try_into().unwrap());
    let non_kernel_block = module_top_block(&mut ctx, &non_kernel_module);
    let non_kernel = kernel_with_params(&mut ctx, "device_fn", vec![ptr.into()], false);
    set_kernel_reference_param_validity(
        &mut ctx,
        non_kernel.get_operation(),
        0,
        KernelReferenceParamValidityAttr(4),
    );
    non_kernel
        .get_operation()
        .insert_at_back(non_kernel_block, &ctx);
    let error = export_module_to_string_with_config(&ctx, &non_kernel_module, &PtxExportConfig)
        .expect_err("non-kernel validity must fail");
    assert!(error.contains("is not a kernel entry"), "{error}");

    let scalar_module = ModuleOp::new(&mut ctx, "scalar_param".try_into().unwrap());
    let scalar_block = module_top_block(&mut ctx, &scalar_module);
    let scalar = kernel_with_params(&mut ctx, "scalar_kernel", vec![i32_ty.into()], true);
    set_kernel_reference_param_validity(
        &mut ctx,
        scalar.get_operation(),
        0,
        KernelReferenceParamValidityAttr(4),
    );
    scalar.get_operation().insert_at_back(scalar_block, &ctx);
    let error = export_module_to_string_with_config(&ctx, &scalar_module, &PtxExportConfig)
        .expect_err("non-pointer validity must fail");
    assert!(error.contains("is not an LLVM pointer"), "{error}");

    let range_module = ModuleOp::new(&mut ctx, "range_param".try_into().unwrap());
    let range_block = module_top_block(&mut ctx, &range_module);
    let range = kernel_with_params(&mut ctx, "range_kernel", vec![ptr.into()], true);
    set_kernel_reference_param_validity(
        &mut ctx,
        range.get_operation(),
        3,
        KernelReferenceParamValidityAttr(4),
    );
    range.get_operation().insert_at_back(range_block, &ctx);
    let error = export_module_to_string_with_config(&ctx, &range_module, &PtxExportConfig)
        .expect_err("out-of-range validity must fail");
    assert!(error.contains("index 3 is out of range"), "{error}");
}

#[test]
fn reference_validity_rejects_malformed_marker_indices() {
    use reserved_oxide_symbols::KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX;

    for suffix in ["", "invalid", "0_extra", "18446744073709551616"] {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "malformed_marker".try_into().unwrap());
        let module_block = module_top_block(&mut ctx, &module);
        let ptr = PointerType::get(&ctx, 0);
        let func = kernel_with_params(&mut ctx, "malformed_marker", vec![ptr.into()], true);
        let key = format!("{KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX}{suffix}");
        func.get_operation()
            .deref_mut(&ctx)
            .attributes
            .set(key.try_into().unwrap(), KernelReferenceParamValidityAttr(4));
        func.get_operation().insert_at_back(module_block, &ctx);
        let error = export_module_to_string_with_config(&ctx, &module, &PtxExportConfig)
            .expect_err("malformed reference-validity marker must fail");
        assert!(error.contains("invalid parameter index"), "{error}");
    }
}
