/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::MirPtrType;
use dialect_nvvm::ops as nvvm;
use llvm_export::export::{NvvmExportConfig, NvvmIrDialect, export_module_to_string_with_config};
use llvm_export::ops as llvm;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

fn lower_g2s_forms(
    destination_space: u32,
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, Ptr<Operation>), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let byte = IntegerType::get(&ctx, 8, Signedness::Signless).into();
    let i16_ty = IntegerType::get(&ctx, 16, Signedness::Signless).into();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless).into();
    let destination = MirPtrType::get(&mut ctx, byte, true, destination_space).into();
    let pointer = MirPtrType::get_generic(&mut ctx, byte, true).into();
    let (module, entry) = build_test_kernel(
        &mut ctx,
        vec![
            destination,
            pointer,
            pointer,
            i32_ty,
            i32_ty,
            i32_ty,
            i32_ty,
            i32_ty,
            i16_ty,
            i64_ty,
        ],
    );
    let args: Vec<_> = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();
    for (info, dimensions) in [
        (
            nvvm::CpAsyncBulkTensorG2sTile1dOp::get_concrete_op_info(),
            1,
        ),
        (
            nvvm::CpAsyncBulkTensorG2sTile2dOp::get_concrete_op_info(),
            2,
        ),
        (
            nvvm::CpAsyncBulkTensorG2sTile3dOp::get_concrete_op_info(),
            3,
        ),
        (
            nvvm::CpAsyncBulkTensorG2sTile4dOp::get_concrete_op_info(),
            4,
        ),
        (
            nvvm::CpAsyncBulkTensorG2sTile5dOp::get_concrete_op_info(),
            5,
        ),
        (
            nvvm::CpAsyncBulkTensorG2sTile2dMulticastOp::get_concrete_op_info(),
            2,
        ),
        (
            nvvm::CpAsyncBulkTensorG2sTile2dMulticastCg2Op::get_concrete_op_info(),
            2,
        ),
    ] {
        let mut operands = args[..3 + dimensions].to_vec();
        operands.extend_from_slice(&args[8..10]);
        Operation::new(&mut ctx, info, vec![], operands, vec![], 0).insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);
    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module))
}

#[test]
fn tma_libnvvm_preserves_cluster_address_conversion_without_creating_as7()
-> Result<(), anyhow::Error> {
    for destination_space in [0, 3] {
        let (ctx, module) =
            lower_g2s_forms(destination_space, mir_lower::IntrinsicBackend::LibNvvm)?;
        let mut count = 0;
        for op in lowered_kernel_body(&ctx, module) {
            let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                continue;
            };
            assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::Convergent);
            let ty = op.deref(&ctx).get_operand(0).get_type(&ctx);
            assert_eq!(
                ty.deref(&ctx)
                    .downcast_ref::<llvm_export::types::PointerType>()
                    .unwrap()
                    .address_space(),
                0
            );
            count += 1;
        }
        assert_eq!(count, 7);
        let module = Operation::get_op::<ModuleOp>(module, &ctx).unwrap();
        for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
            let ir =
                export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                    .unwrap();
            assert!(!ir.contains("addrspace(7)"), "{ir}");
            assert_eq!(
                ir.matches("cvta.to.shared::cluster.u64 %cluster_dst, $0;")
                    .count(),
                7,
                "{ir}"
            );
            assert_eq!(ir.matches("[%cluster_dst], [$2,").count(), 7, "{ir}");
            assert_eq!(ir.matches("~{memory}").count(), 7, "{ir}");
            assert!(ir.contains(".multicast::cluster.cta_group::2"), "{ir}");
        }
    }
    Ok(())
}

#[test]
fn tma_llvm_nvptx_keeps_shared_cluster_intrinsic_types() -> Result<(), anyhow::Error> {
    let (ctx, module) = lower_g2s_forms(0, mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let module = Operation::get_op::<ModuleOp>(module, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module).unwrap();
    assert!(ir.contains("addrspace(7)"), "{ir}");
    assert!(
        ir.contains("@llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d"),
        "{ir}"
    );
    assert!(!ir.contains("asm sideeffect"), "{ir}");
    Ok(())
}

#[test]
fn tma_modern_libnvvm_preserves_existing_cluster_pointer_semantics() -> Result<(), anyhow::Error> {
    // An existing AS7 producer still needs a backend that supports it. The
    // TMA repair must not globally relabel that pointer as CTA-local AS3.
    let (ctx, module) = lower_g2s_forms(7, mir_lower::IntrinsicBackend::LibNvvm)?;
    let module = Operation::get_op::<ModuleOp>(module, &ctx).unwrap();
    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .unwrap();
    assert!(ir.contains("addrspacecast ptr addrspace(7)"), "{ir}");
    assert_eq!(
        ir.matches("cvta.to.shared::cluster.u64 %cluster_dst, $0;")
            .count(),
        7,
        "{ir}"
    );
    assert!(!ir.contains("cvta.to.shared::cta"), "{ir}");
    Ok(())
}
