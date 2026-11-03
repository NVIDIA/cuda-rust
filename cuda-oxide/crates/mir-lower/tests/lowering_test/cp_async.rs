/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

fn lower_all_classic_cp_async(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = make_test_ctx();
    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let src32_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let src8_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![
            dst_ty.into(),
            src32_ty.into(),
            src8_ty.into(),
            u32_ty.into(),
        ],
    );
    let dst = entry.deref(&ctx).get_argument(0);
    let src32 = entry.deref(&ctx).get_argument(1);
    let src8 = entry.deref(&ctx).get_argument(2);
    let source_size = entry.deref(&ctx).get_argument(3);

    let zero_op = Operation::new(
        &mut ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(zero_op).set_attr_value(
        &ctx,
        IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap())),
    );
    zero_op.insert_at_back(entry, &ctx);
    let zero = zero_op.deref(&ctx).get_result(0);

    for copy in [
        nvvm::CpAsyncCa4Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCa8Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCa16Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCaZfill4Op::build(&mut ctx, dst, src8, source_size),
        nvvm::CpAsyncCaZfill8Op::build(&mut ctx, dst, src8, source_size),
        nvvm::CpAsyncCaZfill16Op::build(&mut ctx, dst, src8, source_size),
        nvvm::CpAsyncCg16Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCgZfill16Op::build(&mut ctx, dst, src8, source_size),
    ] {
        copy.insert_at_back(entry, &ctx);
    }
    for control in [
        nvvm::CpAsyncCommitGroupOp::build(&mut ctx),
        nvvm::CpAsyncWaitGroupOp::build(&mut ctx, zero),
        nvvm::CpAsyncWaitAllOp::build(&mut ctx),
    ] {
        control.insert_at_back(entry, &ctx);
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
    Ok((ctx, module_ptr))
}

fn lower_all_cp_async_mbarrier(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_ty = MirPtrType::get_generic(&mut ctx, u64_ty.into(), true);
    let shared_ty = MirPtrType::get_shared(&mut ctx, u64_ty.into(), true);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![generic_ty.into(), shared_ty.into()]);
    let generic = entry.deref(&ctx).get_argument(0);
    let shared = entry.deref(&ctx).get_argument(1);

    for bridge in [
        nvvm::CpAsyncMbarrierArriveOp::build(&mut ctx, shared),
        nvvm::CpAsyncMbarrierArriveSharedOp::build(&mut ctx, generic),
        nvvm::CpAsyncMbarrierArriveNoIncOp::build(&mut ctx, shared),
        nvvm::CpAsyncMbarrierArriveNoIncSharedOp::build(&mut ctx, generic),
    ] {
        bridge.insert_at_back(entry, &ctx);
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
    Ok((ctx, module_ptr))
}

#[test]
fn test_generated_cp_async_mbarrier_preserves_backend_and_address_routes()
-> Result<(), anyhow::Error> {
    use llvm_export::types::PointerType;
    use pliron::r#type::Typed;

    let typed = [
        ("llvm_nvvm_cp_async_mbarrier_arrive", 0),
        ("llvm_nvvm_cp_async_mbarrier_arrive_shared", 3),
        ("llvm_nvvm_cp_async_mbarrier_arrive_noinc", 0),
        ("llvm_nvvm_cp_async_mbarrier_arrive_noinc_shared", 3),
    ];
    let templates = [
        ("cp.async.mbarrier.arrive.b64 [$0];", 0),
        ("cp.async.mbarrier.arrive.shared.b64 [$0];", 3),
        ("cp.async.mbarrier.arrive.noinc.b64 [$0];", 0),
        ("cp.async.mbarrier.arrive.noinc.shared.b64 [$0];", 3),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_all_cp_async_mbarrier(backend)?;
        let mut call_counts = [0usize; 4];
        let mut asm_counts = [0usize; 4];

        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) {
                let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                    continue;
                };
                let callee = callee.to_string();
                let Some(index) = typed.iter().position(|(name, _)| callee == *name) else {
                    continue;
                };
                call_counts[index] += 1;
                let pointer = op.deref(&ctx).get_operand(0).get_type(&ctx);
                assert_eq!(
                    pointer
                        .deref(&ctx)
                        .downcast_ref::<PointerType>()
                        .unwrap()
                        .address_space(),
                    typed[index].1
                );
            }
            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap();
                let Some(index) = templates
                    .iter()
                    .position(|(expected, _)| template == *expected)
                else {
                    continue;
                };
                asm_counts[index] += 1;
                assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                assert!(
                    inline_asm
                        .get_attr_llvm_inline_asm_attrs(&ctx)
                        .is_some_and(|attrs| attrs.has("convergent"))
                );
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some("l,~{memory}")
                );
                let pointer = op.deref(&ctx).get_operand(0).get_type(&ctx);
                assert_eq!(
                    pointer
                        .deref(&ctx)
                        .downcast_ref::<PointerType>()
                        .unwrap()
                        .address_space(),
                    templates[index].1
                );
            }
        }

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert_eq!(call_counts, [1; 4]);
                assert_eq!(asm_counts, [0; 4]);
                assert!(
                    ir.contains("@llvm.nvvm.cp.async.mbarrier.arrive(ptr"),
                    "{ir}"
                );
                assert!(
                    ir.contains("@llvm.nvvm.cp.async.mbarrier.arrive.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(
                    ir.contains("@llvm.nvvm.cp.async.mbarrier.arrive.noinc(ptr"),
                    "{ir}"
                );
                assert!(
                    ir.contains(
                        "@llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(ptr addrspace(3)"
                    ),
                    "{ir}"
                );
                assert!(!ir.contains("cp.async.mbarrier.arrive.b64 [$0]"), "{ir}");
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                assert_eq!(call_counts, [0; 4]);
                assert_eq!(asm_counts, [1; 4]);
                assert!(!ir.contains("@llvm.nvvm.cp.async.mbarrier"), "{ir}");
                for (template, _) in templates {
                    assert!(ir.contains(template), "{ir}");
                }
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("convergent"), "{ir}");
            }
        }
    }
    Ok(())
}

#[test]
fn test_generated_cp_async_llvm_nvptx_uses_all_typed_intrinsics() -> Result<(), anyhow::Error> {
    use llvm_export::types::PointerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_classic_cp_async(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut found = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX cp.async must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !callee.starts_with("llvm_nvvm_cp_async") {
            continue;
        }
        if callee.contains("shared_global") {
            let call = op.deref(&ctx);
            let destination_ty = call.get_operand(0).get_type(&ctx);
            let source_ty = call.get_operand(1).get_type(&ctx);
            assert_eq!(
                destination_ty
                    .deref(&ctx)
                    .downcast_ref::<PointerType>()
                    .unwrap()
                    .address_space(),
                3
            );
            assert_eq!(
                source_ty
                    .deref(&ctx)
                    .downcast_ref::<PointerType>()
                    .unwrap()
                    .address_space(),
                1
            );
        }
        found.push(callee);
    }
    found.sort();
    let mut expected = [
        "llvm_nvvm_cp_async_ca_shared_global_4",
        "llvm_nvvm_cp_async_ca_shared_global_4_s",
        "llvm_nvvm_cp_async_ca_shared_global_8",
        "llvm_nvvm_cp_async_ca_shared_global_8_s",
        "llvm_nvvm_cp_async_ca_shared_global_16",
        "llvm_nvvm_cp_async_ca_shared_global_16_s",
        "llvm_nvvm_cp_async_cg_shared_global_16",
        "llvm_nvvm_cp_async_cg_shared_global_16_s",
        "llvm_nvvm_cp_async_commit_group",
        "llvm_nvvm_cp_async_wait_all",
        "llvm_nvvm_cp_async_wait_group",
    ]
    .map(str::to_owned);
    expected.sort();
    assert_eq!(found, expected);
    Ok(())
}

#[test]
fn test_generated_cp_async_libnvvm_uses_all_exact_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_classic_cp_async(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut lowered = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::SideEffect);
        assert!(
            !asm.get_attr_llvm_inline_asm_attrs(&ctx)
                .is_some_and(|attrs| attrs.has("convergent"))
        );
        lowered.push((
            asm.get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap(),
            asm.get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap(),
        ));
    }

    let expected = [
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 4;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 8;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 16;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 4, $2;",
            "l,l,r,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 8, $2;",
            "l,l,r,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 16, $2;",
            "l,l,r,~{memory}",
        ),
        (
            "cp.async.cg.shared.global [%smem32], [%gmem64], 16;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.cg.shared.global [%smem32], [%gmem64], 16, $2;",
            "l,l,r,~{memory}",
        ),
        ("cp.async.commit_group;", "~{memory}"),
        ("cp.async.wait_group $0;", "n,~{memory}"),
        ("cp.async.wait_all;", "~{memory}"),
    ];
    for (instruction, constraints) in expected {
        assert_eq!(
            lowered
                .iter()
                .filter(|(template, actual_constraints)| {
                    template.contains(instruction) && actual_constraints == constraints
                })
                .count(),
            1,
            "missing exact `{instruction}`"
        );
    }
    assert_eq!(lowered.len(), expected.len());
    Ok(())
}

#[test]
fn test_cp_async_ca_4_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCa4Op::get_concrete_op_info(),
        vec![],
        vec![dst, src],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_inline_asm_lowering(&mut ctx, module_ptr, 4)
}

#[test]
fn test_cp_async_ca_8_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCa8Op::get_concrete_op_info(),
        vec![],
        vec![dst, src],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_inline_asm_lowering(&mut ctx, module_ptr, 8)
}

fn assert_cp_async_inline_asm_lowering(
    ctx: &mut Context,
    module_ptr: pliron::context::Ptr<Operation>,
    copy_size: u32,
) -> Result<(), anyhow::Error> {
    use pliron::r#type::Typed;

    mir_lower::lower_mir_to_llvm_with_options(
        ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LibNvvm,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let expected_template = format!(
        "{{ .reg .u64 %smem64; .reg .u32 %smem32; .reg .u64 %gmem64; \
         cvta.to.shared.u64 %smem64, $0; cvt.u32.u64 %smem32, %smem64; \
         cvta.to.global.u64 %gmem64, $1; \
         cp.async.ca.shared.global [%smem32], [%gmem64], {copy_size}; }}"
    );
    let mut matches = 0;
    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();

    for op in module_block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        for func_block in func_region.deref(ctx).iter(ctx) {
            for body_op in func_block.deref(ctx).iter(ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(ctx)
                    .map(|s| String::from((*s).clone()));
                if template.as_deref() != Some(expected_template.as_str()) {
                    continue;
                }

                matches += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(ctx)
                        .map(|s| String::from((*s).clone()))
                        .as_deref(),
                    Some("l,l,~{memory}")
                );
                assert_eq!(llvm::asm_kind(ctx, &inline_asm), llvm::AsmKind::SideEffect);
                assert!(
                    !inline_asm
                        .get_attr_llvm_inline_asm_attrs(ctx)
                        .is_some_and(|attrs| attrs.has("convergent"))
                );

                let operands: Vec<_> = inline_asm.get_operation().deref(ctx).operands().collect();
                assert_eq!(operands.len(), 2);
                for operand in operands {
                    let ty = operand.get_type(ctx);
                    let ty = ty.deref(ctx);
                    let ptr_ty = ty
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .expect("cp.async operands must lower to LLVM pointers");
                    assert_eq!(ptr_ty.address_space(), 0);
                }
            }
        }
    }

    assert_eq!(matches, 1, "missing exact {copy_size}-byte cp.async asm");
    Ok(())
}

// =============================================================================
// cp.async zero-fill lowering tests
// =============================================================================

#[test]
fn test_cp_async_ca_zfill_4_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), false);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into(), i32_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);
    let src_size = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCaZfill4Op::get_concrete_op_info(),
        vec![],
        vec![dst, src, src_size],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_zfill_inline_asm_lowering(&mut ctx, module_ptr, 4)
}

#[test]
fn test_cp_async_ca_zfill_8_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), false);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into(), i32_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);
    let src_size = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCaZfill8Op::get_concrete_op_info(),
        vec![],
        vec![dst, src, src_size],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_zfill_inline_asm_lowering(&mut ctx, module_ptr, 8)
}

#[test]
fn test_cp_async_ca_zfill_16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), false);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into(), i32_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);
    let src_size = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCaZfill16Op::get_concrete_op_info(),
        vec![],
        vec![dst, src, src_size],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_zfill_inline_asm_lowering(&mut ctx, module_ptr, 16)
}

fn assert_cp_async_zfill_inline_asm_lowering(
    ctx: &mut Context,
    module_ptr: pliron::context::Ptr<Operation>,
    copy_size: u32,
) -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    mir_lower::lower_mir_to_llvm_with_options(
        ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LibNvvm,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let expected_template = format!(
        "{{ .reg .u64 %smem64; .reg .u32 %smem32; .reg .u64 %gmem64; \
         cvta.to.shared.u64 %smem64, $0; cvt.u32.u64 %smem32, %smem64; \
         cvta.to.global.u64 %gmem64, $1; \
         cp.async.ca.shared.global [%smem32], [%gmem64], {copy_size}, $2; }}"
    );
    let mut matches = 0;
    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();

    for op in module_block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        for func_block in func_region.deref(ctx).iter(ctx) {
            for body_op in func_block.deref(ctx).iter(ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_llvm_inline_asm_template(ctx)
                    .map(|s| String::from((*s).clone()));
                if template.as_deref() != Some(expected_template.as_str()) {
                    continue;
                }

                matches += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_llvm_inline_asm_constraints(ctx)
                        .map(|s| String::from((*s).clone()))
                        .as_deref(),
                    Some("l,l,r,~{memory}")
                );
                assert_eq!(llvm::asm_kind(ctx, &inline_asm), llvm::AsmKind::SideEffect);
                assert!(
                    !inline_asm
                        .get_attr_llvm_inline_asm_attrs(ctx)
                        .is_some_and(|attrs| attrs.has("convergent"))
                );

                let operands: Vec<_> = inline_asm.get_operation().deref(ctx).operands().collect();
                assert_eq!(operands.len(), 3);
                for operand in &operands[..2] {
                    let ty = operand.get_type(ctx);
                    let ty = ty.deref(ctx);
                    let ptr_ty = ty
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .expect("cp.async pointer operands must lower to LLVM pointers");
                    assert_eq!(ptr_ty.address_space(), 0);
                }

                let src_size_ty = operands[2].get_type(ctx);
                let src_size_ty = src_size_ty.deref(ctx);
                let src_size_ty = src_size_ty
                    .downcast_ref::<IntegerType>()
                    .expect("cp.async src_size must lower to an integer");
                assert_eq!(src_size_ty.width(), 32);
            }
        }
    }

    assert_eq!(
        matches, 1,
        "missing exact {copy_size}-byte zero-fill cp.async asm"
    );
    Ok(())
}

// =============================================================================
// Generated packed arithmetic lowering tests
// =============================================================================
