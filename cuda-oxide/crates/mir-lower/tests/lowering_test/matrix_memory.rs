/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

const STMATRIX_TYPED_INTRINSICS: [&str; 4] = [
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x2_b16_p3",
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16_p3",
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x4_b16_p3",
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16_p3",
];

const STMATRIX_PTX: [(&str, &str); 4] = [
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }",
        "l,r,r,~{memory}",
    ),
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%ptr32], {$1, $2}; }",
        "l,r,r,~{memory}",
    ),
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x4.shared.b16 [%ptr32], {$1, $2, $3, $4}; }",
        "l,r,r,r,r,~{memory}",
    ),
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%ptr32], {$1, $2, $3, $4}; }",
        "l,r,r,r,r,~{memory}",
    ),
];

fn lower_all_stmatrix_forms(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), true);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![
            ptr_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
        ],
    );
    let args: Vec<_> = (0..5)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    for (op_info, operands) in [
        (
            nvvm::StmatrixM8n8X2Op::get_concrete_op_info(),
            args[..3].to_vec(),
        ),
        (
            nvvm::StmatrixM8n8X2TransOp::get_concrete_op_info(),
            args[..3].to_vec(),
        ),
        (nvvm::StmatrixM8n8X4Op::get_concrete_op_info(), args.clone()),
        (nvvm::StmatrixM8n8X4TransOp::get_concrete_op_info(), args),
    ] {
        Operation::new(&mut ctx, op_info, vec![], operands, vec![], 0).insert_at_back(entry, &ctx);
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
fn test_stmatrix_llvm_nvptx_uses_exact_typed_p3_intrinsics() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_stmatrix_forms(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut callees = Vec::new();

    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX stmatrix lowering must not emit inline PTX"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !STMATRIX_TYPED_INTRINSICS.contains(&callee.as_str()) {
            continue;
        }

        let call_op = call.get_operation().deref(&ctx);
        assert!(matches!(call_op.get_num_operands(), 3 | 5));
        assert_eq!(call_op.get_num_results(), 1);
        let pointer_ty = call_op.get_operand(0).get_type(&ctx);
        let pointer_ty = pointer_ty.deref(&ctx);
        let pointer_ty = pointer_ty
            .downcast_ref::<llvm_types::PointerType>()
            .expect("stmatrix first argument is a pointer");
        assert_eq!(pointer_ty.address_space(), 3);
        callees.push(callee);
    }

    callees.sort();
    let mut expected = STMATRIX_TYPED_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(callees, expected);

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .expect("typed stmatrix module exports to LLVM IR");
    for intrinsic in STMATRIX_TYPED_INTRINSICS {
        let dotted = intrinsic.replace('_', ".");
        assert!(
            ir.contains(&format!("@{dotted}(ptr addrspace(3)")),
            "missing exact typed stmatrix declaration:\n{ir}"
        );
    }
    assert!(!ir.contains("asm sideeffect"), "{ir}");
    Ok(())
}

#[test]
fn test_stmatrix_libnvvm_uses_exact_convergent_memory_asm() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_stmatrix_forms(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut lowered = Vec::new();

    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !STMATRIX_TYPED_INTRINSICS.contains(&callee.as_ref()),
                "libNVVM stmatrix lowering must not emit typed intrinsic calls"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        lowered.push((
            inline_asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default(),
            inline_asm
                .get_attr_llvm_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default(),
            llvm::asm_kind(&ctx, &inline_asm),
            op.deref(&ctx).get_num_operands(),
            op.deref(&ctx).get_num_results(),
        ));
    }

    assert_eq!(lowered.len(), STMATRIX_PTX.len());
    for (template, constraints) in STMATRIX_PTX {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(actual, _, _, _, _)| actual == template)
            .collect();
        assert_eq!(matches.len(), 1, "missing exact PTX {template}");
        let (_, actual_constraints, kind, operands, results) = matches[0];
        assert_eq!(actual_constraints, constraints);
        assert_eq!(*kind, llvm::AsmKind::Convergent);
        assert!(matches!(*operands, 3 | 5));
        assert_eq!(*results, 1);
    }

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .expect("inline stmatrix module exports to LLVM IR");
    assert_eq!(ir.matches("asm sideeffect").count(), 4, "{ir}");
    assert_eq!(ir.matches("~{memory}").count(), 4, "{ir}");
    assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    assert!(!ir.contains("@llvm.nvvm.stmatrix"), "{ir}");
    Ok(())
}

// =============================================================================
// Warp-level matrix (`movmatrix`) lowering test
// =============================================================================

#[test]
fn test_movmatrix_trans_b16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);

    let a_val = entry.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        nvvm::MovmatrixTransB16Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut found = 0;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for module_op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(module_op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                found += 1;
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                assert_eq!(
                    template.as_deref(),
                    Some("movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;")
                );
                assert_eq!(constraints.as_deref(), Some("=r,r"));
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert!(
                    asm.get_attr_llvm_inline_asm_attrs(&ctx)
                        .is_some_and(|attrs| attrs.has("convergent"))
                );
                assert!(
                    !constraints.as_deref().unwrap().contains("memory"),
                    "register-only movmatrix must not claim a memory clobber"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one movmatrix inline-asm operation");
    Ok(())
}

// =============================================================================
// ldmatrix lowering tests
// =============================================================================

const LDMATRIX_TYPED_INTRINSICS: [&str; 6] = [
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x1_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x4_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16_p3",
];

const LDMATRIX_PTX_TEMPLATES: [&str; 6] = [
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [$1];",
    "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {$0}, [$1];",
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
];

const LDMATRIX_PTX_CONSTRAINTS: [&str; 6] = [
    "=r,r,~{memory}",
    "=r,r,~{memory}",
    "=r,=r,r,~{memory}",
    "=r,=r,r,~{memory}",
    "=r,=r,=r,=r,r,~{memory}",
    "=r,=r,=r,=r,r,~{memory}",
];

const BLACKWELL_LDMATRIX_CASES: [(&str, &str, &str, usize); 12] = [
    (
        "llvm_nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8_p3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8.p3",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm_nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8_p3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8.p3",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx2_dtrans_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx2_dtrans_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx1_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x1.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 {$0}, [$1];",
        1,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx1_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x1.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {$0}, [$1];",
        1,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx2_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x2.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx2_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x2.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx4_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x4.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx4_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x4.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {$0, $1, $2, $3}, [$4];",
        4,
    ),
];

fn ldmatrix_constraints(register_count: usize) -> &'static str {
    match register_count {
        1 => "=r,r,~{memory}",
        2 => "=r,=r,r,~{memory}",
        4 => "=r,=r,=r,=r,r,~{memory}",
        _ => unreachable!("closed ldmatrix register count"),
    }
}

fn lower_all_ldmatrix_forms(
    address_space: u32,
    backend: mir_lower::IntrinsicBackend,
    compatibility: bool,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get(&mut ctx, u32_ty.into(), true, address_space);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
    let pointer = entry.deref(&ctx).get_argument(0);
    if compatibility {
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        for (op_info, result_count) in [
            (nvvm::LdmatrixX1Op::get_concrete_op_info(), 1),
            (nvvm::LdmatrixX1TransOp::get_concrete_op_info(), 1),
            (nvvm::LdmatrixX2Op::get_concrete_op_info(), 2),
            (nvvm::LdmatrixX2TransOp::get_concrete_op_info(), 2),
            (nvvm::LdmatrixX4Op::get_concrete_op_info(), 4),
            (nvvm::LdmatrixX4TransOp::get_concrete_op_info(), 4),
        ] {
            Operation::new(
                &mut ctx,
                op_info,
                vec![u32_ty.into(); result_count],
                vec![pointer],
                vec![],
                0,
            )
            .insert_at_back(entry, &ctx);
        }
    } else {
        for (multiplicity, layout) in [
            (
                nvvm::LdmatrixMultiplicityAttr::X1,
                nvvm::LdmatrixLayoutAttr::Normal,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X1,
                nvvm::LdmatrixLayoutAttr::Transposed,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X2,
                nvvm::LdmatrixLayoutAttr::Normal,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X2,
                nvvm::LdmatrixLayoutAttr::Transposed,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X4,
                nvvm::LdmatrixLayoutAttr::Normal,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X4,
                nvvm::LdmatrixLayoutAttr::Transposed,
            ),
        ] {
            nvvm::LdmatrixOp::build(
                &mut ctx,
                pointer,
                nvvm::LdmatrixShapeAttr::M8n8,
                multiplicity,
                layout,
                nvvm::LdmatrixElementAttr::B16,
                nvvm::LdmatrixStateSpaceAttr::Shared,
            )
            .insert_at_back(entry, &ctx);
        }
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

fn lower_all_blackwell_ldmatrix_forms(
    address_space: u32,
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get(&mut ctx, u8_ty.into(), true, address_space);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
    let pointer = entry.deref(&ctx).get_argument(0);

    for (shape, multiplicity, layout, element) in [
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X4,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X4,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
    ] {
        nvvm::LdmatrixOp::build(
            &mut ctx,
            pointer,
            shape,
            multiplicity,
            layout,
            element,
            nvvm::LdmatrixStateSpaceAttr::Shared,
        )
        .insert_at_back(entry, &ctx);
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

fn assert_ldmatrix_producer_result_shape(
    ctx: &Context,
    op: pliron::context::Ptr<Operation>,
    register_count: usize,
) {
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let ty = op.deref(ctx).get_result(0).get_type(ctx);
    let ty = ty.deref(ctx);
    if register_count == 1 {
        assert_eq!(
            ty.downcast_ref::<IntegerType>()
                .expect("single-register ldmatrix returns i32")
                .width(),
            32
        );
    } else {
        let ty = ty
            .downcast_ref::<llvm_types::StructType>()
            .expect("multi-register ldmatrix returns an LLVM struct");
        assert_eq!(ty.num_fields(), register_count);
        for index in 0..register_count {
            assert_eq!(
                ty.field_type(index)
                    .deref(ctx)
                    .downcast_ref::<IntegerType>()
                    .expect("ldmatrix fragment field is i32")
                    .width(),
                32
            );
        }
    }
}

#[test]
fn test_ldmatrix_llvm_nvptx_uses_exact_typed_p3_intrinsics() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    for address_space in [0, 3] {
        let (ctx, module_ptr) =
            lower_all_ldmatrix_forms(address_space, mir_lower::IntrinsicBackend::LlvmNvptx, false)?;
        let body = lowered_kernel_body(&ctx, module_ptr);
        let mut callees = Vec::new();
        let mut cast_count = 0;
        let mut extract_count = 0;

        for op in body {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                let callee = callee.to_string();
                let register_count = if callee.contains("_x1_") {
                    1
                } else if callee.contains("_x2_") {
                    2
                } else {
                    4
                };
                let function_ty = call.callee_type(&ctx);
                let function_ty = function_ty.deref(&ctx);
                let function_ty = function_ty
                    .downcast_ref::<llvm_types::FuncType>()
                    .expect("ldmatrix callee has an LLVM function type");
                assert_eq!(function_ty.arg_types().len(), 1);
                let argument_ty = function_ty.arg_types()[0].deref(&ctx);
                let argument_ty = argument_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("ldmatrix argument is a pointer");
                assert_eq!(argument_ty.address_space(), 3);

                let result_ty = function_ty.result_type();
                let result_ty = result_ty.deref(&ctx);
                if register_count == 1 {
                    let result_ty = result_ty
                        .downcast_ref::<IntegerType>()
                        .expect("x1 returns i32");
                    assert_eq!(result_ty.width(), 32);
                } else {
                    let result_ty = result_ty
                        .downcast_ref::<llvm_types::StructType>()
                        .expect("x2/x4 return an LLVM struct");
                    assert_eq!(result_ty.num_fields(), register_count);
                    for index in 0..result_ty.num_fields() {
                        let field = result_ty.field_type(index);
                        let field = field.deref(&ctx);
                        let field = field
                            .downcast_ref::<IntegerType>()
                            .expect("fragment register is i32");
                        assert_eq!(field.width(), 32);
                    }
                }

                callees.push(callee);
                assert_eq!(op.deref(&ctx).get_num_operands(), 1);
                assert_eq!(op.deref(&ctx).get_num_results(), 1);
            }
            if Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some() {
                cast_count += 1;
                let cast = op.deref(&ctx);
                let source_ty = cast.get_operand(0).get_type(&ctx);
                let source_ty = source_ty.deref(&ctx);
                let source_ty = source_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("addrspacecast source is a pointer");
                let result_ty = cast.get_result(0).get_type(&ctx);
                let result_ty = result_ty.deref(&ctx);
                let result_ty = result_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("addrspacecast result is a pointer");
                assert_eq!(
                    (source_ty.address_space(), result_ty.address_space()),
                    (0, 3)
                );
            }
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());
            assert!(
                Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
                "LLVM-NVPTX ldmatrix lowering must not emit inline PTX"
            );
            assert!(
                Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_none(),
                "the typed intrinsic consumes the shared pointer directly"
            );
        }

        callees.sort();
        let mut expected = LDMATRIX_TYPED_INTRINSICS.map(str::to_owned);
        expected.sort();
        assert_eq!(callees, expected);
        assert_eq!(cast_count, if address_space == 0 { 6 } else { 0 });
        assert_eq!(
            extract_count, 12,
            "x2/x4 structs must preserve result order"
        );

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("typed ldmatrix module exports to LLVM IR");
        for intrinsic in LDMATRIX_TYPED_INTRINSICS {
            let dotted = intrinsic.replace('_', ".");
            assert!(
                ir.contains(&format!("@{dotted}(ptr addrspace(3)")),
                "underscore symbol must export as exact dotted p3 intrinsic:\n{ir}"
            );
        }
        assert!(!ir.contains("@llvm_nvvm_ldmatrix"));
    }
    Ok(())
}

#[test]
fn test_ldmatrix_libnvvm_uses_exact_convergent_shared_ptx() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::r#type::Typed;

    for address_space in [0, 3] {
        let (ctx, module_ptr) =
            lower_all_ldmatrix_forms(address_space, mir_lower::IntrinsicBackend::LibNvvm, false)?;
        let body = lowered_kernel_body(&ctx, module_ptr);
        let mut lowered = Vec::new();
        let mut cast_count = 0;
        let mut ptrtoint_count = 0;

        for op in body {
            assert!(
                Operation::get_op::<llvm::CallOp>(op, &ctx).is_none(),
                "libNVVM ldmatrix lowering must not emit typed intrinsic calls"
            );
            cast_count +=
                usize::from(Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some());
            if Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_some() {
                ptrtoint_count += 1;
                let cast = op.deref(&ctx);
                let source_ty = cast.get_operand(0).get_type(&ctx);
                let source_ty = source_ty.deref(&ctx);
                let source_ty = source_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("ptrtoint source is a pointer");
                assert_eq!(source_ty.address_space(), 3);
            }

            let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                continue;
            };
            lowered.push((
                inline_asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default(),
                inline_asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default(),
                llvm::asm_kind(&ctx, &inline_asm),
                op.deref(&ctx).get_num_operands(),
                op.deref(&ctx).get_num_results(),
            ));
        }

        assert_eq!(lowered.len(), 6);
        for (index, expected_template) in LDMATRIX_PTX_TEMPLATES.iter().enumerate() {
            let matching: Vec<_> = lowered
                .iter()
                .filter(|(template, _, _, _, _)| template == expected_template)
                .collect();
            assert_eq!(matching.len(), 1, "missing exact PTX `{expected_template}`");
            let (_, constraints, kind, operands, results) = matching[0];
            assert_eq!(constraints, LDMATRIX_PTX_CONSTRAINTS[index]);
            assert_eq!(*kind, llvm::AsmKind::Convergent);
            assert_eq!(*operands, 1, "inline PTX consumes one i32 shared address");
            assert_eq!(*results, 1, "inline PTX returns one scalar or struct");
            assert!(!expected_template.contains("cvta.to.shared"));
        }
        assert_eq!(cast_count, if address_space == 0 { 6 } else { 0 });
        assert_eq!(ptrtoint_count, 6);

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("inline ldmatrix module exports to LLVM IR");
        assert_eq!(ir.matches("asm sideeffect").count(), 6, "{ir}");
        assert_eq!(ir.matches("~{memory}").count(), 6, "{ir}");
        assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
        assert!(!ir.contains("@llvm.nvvm.ldmatrix"), "{ir}");
    }
    Ok(())
}

#[test]
fn native_u32_ldmatrix_stays_narrow_until_backend_boundary() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![u32_ty.into()]);
        let address = entry.deref(&ctx).get_argument(0);
        nvvm::LdmatrixOp::build(
            &mut ctx,
            address,
            nvvm::LdmatrixShapeAttr::M8n8,
            nvvm::LdmatrixMultiplicityAttr::X4,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B16,
            nvvm::LdmatrixStateSpaceAttr::Shared,
        )
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )?;

        let body = lowered_kernel_body(&ctx, module_ptr);
        let int_to_ptr = body
            .iter()
            .filter(|op| Operation::get_op::<llvm::IntToPtrOp>(**op, &ctx).is_some())
            .count();
        let ptr_to_int = body
            .iter()
            .filter(|op| Operation::get_op::<llvm::PtrToIntOp>(**op, &ctx).is_some())
            .count();
        let address_space_cast = body
            .iter()
            .filter(|op| Operation::get_op::<llvm::AddrSpaceCastOp>(**op, &ctx).is_some())
            .count();
        let inline_asm = body
            .iter()
            .filter(|op| Operation::get_op::<llvm::InlineAsmOp>(**op, &ctx).is_some())
            .count();
        let calls = body
            .iter()
            .filter(|op| Operation::get_op::<llvm::CallOp>(**op, &ctx).is_some())
            .count();

        assert_eq!(
            ptr_to_int, 0,
            "native address must not round-trip through a pointer"
        );
        assert_eq!(address_space_cast, 0);
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert_eq!(
                    int_to_ptr, 1,
                    "typed LLVM intrinsic requires one p3 boundary cast"
                );
                assert_eq!(calls, 1);
                assert_eq!(inline_asm, 0);
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                assert_eq!(
                    int_to_ptr, 0,
                    "inline PTX consumes the u32 address directly"
                );
                assert_eq!(calls, 0);
                assert_eq!(inline_asm, 1);
            }
        }
    }
    Ok(())
}

#[test]
fn test_blackwell_ldmatrix_llvm_uses_all_exact_lossless_p3_intrinsics() -> Result<(), anyhow::Error>
{
    use llvm_export::types as llvm_types;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;

    for address_space in [0, 3] {
        let (ctx, module_ptr) = lower_all_blackwell_ldmatrix_forms(
            address_space,
            mir_lower::IntrinsicBackend::LlvmNvptx,
        )?;
        let mut seen = [0; 12];
        let mut cast_count = 0;
        let mut extract_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            assert!(Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none());
            assert!(Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_none());
            cast_count +=
                usize::from(Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some());
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());

            let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                continue;
            };
            let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                panic!("Blackwell ldmatrix intrinsic call must be direct");
            };
            let callee = callee.to_string();
            let index = BLACKWELL_LDMATRIX_CASES
                .iter()
                .position(|(identifier, _, _, _)| *identifier == callee)
                .expect("exact lossless Blackwell ldmatrix identifier");
            seen[index] += 1;
            let register_count = BLACKWELL_LDMATRIX_CASES[index].3;
            assert_ldmatrix_producer_result_shape(&ctx, op, register_count);

            let function_ty = call.callee_type(&ctx);
            let function_ty = function_ty.deref(&ctx);
            let function_ty = function_ty
                .downcast_ref::<llvm_types::FuncType>()
                .expect("Blackwell ldmatrix has an LLVM function type");
            assert_eq!(function_ty.arg_types().len(), 1);
            assert_eq!(
                function_ty.arg_types()[0]
                    .deref(&ctx)
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("Blackwell ldmatrix argument is a pointer")
                    .address_space(),
                3
            );
        }

        assert_eq!(seen, [1; 12]);
        assert_eq!(cast_count, if address_space == 0 { 12 } else { 0 });
        assert_eq!(extract_count, 30, "all multi-register results are unpacked");

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("typed Blackwell ldmatrix module exports to LLVM IR");
        for (identifier, symbol, _, _) in BLACKWELL_LDMATRIX_CASES {
            assert!(
                ir.contains(&format!("@{symbol}(ptr addrspace(3)")),
                "missing exact intrinsic symbol {symbol}:\n{ir}"
            );
            assert!(
                !ir.contains(&format!("@{identifier}(")),
                "encoded Rust identifier leaked into LLVM IR: {identifier}"
            );
        }
        assert!(ir.contains("b4x16_p64.p3"), "literal _p64 was lost: {ir}");
        assert!(ir.contains("b6x16_p32.p3"), "literal _p32 was lost: {ir}");
    }
    Ok(())
}

#[test]
fn test_blackwell_ldmatrix_libnvvm_uses_all_exact_convergent_templates_without_externs()
-> Result<(), anyhow::Error> {
    for address_space in [0, 3] {
        let (ctx, module_ptr) = lower_all_blackwell_ldmatrix_forms(
            address_space,
            mir_lower::IntrinsicBackend::LibNvvm,
        )?;
        let mut seen = [0; 12];
        let mut cast_count = 0;
        let mut ptrtoint_count = 0;
        let mut extract_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            assert!(
                Operation::get_op::<llvm::CallOp>(op, &ctx).is_none(),
                "libNVVM Blackwell ldmatrix must not emit an extern call"
            );
            cast_count +=
                usize::from(Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some());
            ptrtoint_count +=
                usize::from(Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_some());
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());

            let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                continue;
            };
            let template = asm
                .get_attr_llvm_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default();
            let index = BLACKWELL_LDMATRIX_CASES
                .iter()
                .position(|(_, _, expected, _)| *expected == template)
                .expect("exact Blackwell ldmatrix inline-PTX template");
            seen[index] += 1;
            let register_count = BLACKWELL_LDMATRIX_CASES[index].3;
            assert_eq!(
                asm.get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .as_deref(),
                Some(ldmatrix_constraints(register_count))
            );
            assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::Convergent);
            assert_ldmatrix_producer_result_shape(&ctx, op, register_count);
        }

        assert_eq!(seen, [1; 12]);
        assert_eq!(cast_count, if address_space == 0 { 12 } else { 0 });
        assert_eq!(ptrtoint_count, 12);
        assert_eq!(extract_count, 30, "all multi-register results are unpacked");

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("inline Blackwell ldmatrix module exports to LLVM IR");
        assert_eq!(ir.matches("asm sideeffect").count(), 12, "{ir}");
        assert_eq!(ir.matches("~{memory}").count(), 12, "{ir}");
        assert!(!ir.contains("@llvm.nvvm.ldmatrix"), "{ir}");
        assert!(!ir.contains("llvm__nvvm_dldmatrix"), "{ir}");
    }
    Ok(())
}

#[test]
fn test_blackwell_ldmatrix_rejects_unadmitted_m16n16_x4() {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
        let ptr_ty = MirPtrType::get_shared(&mut ctx, u8_ty.into(), false);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
        let pointer = entry.deref(&ctx).get_argument(0);
        nvvm::LdmatrixOp::build(
            &mut ctx,
            pointer,
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X4,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8,
            nvvm::LdmatrixStateSpaceAttr::Shared,
        )
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        let error = mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .expect_err("m16n16.x4 must fail closed")
        .to_string();
        assert!(error.contains("missing or unsupported variant"), "{error}");
    }
}

#[test]
fn test_classic_ldmatrix_compatibility_ops_keep_exact_lowering() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    fn assert_result_shape(
        ctx: &Context,
        op: pliron::context::Ptr<Operation>,
        register_count: usize,
    ) {
        let operation = op.deref(ctx);
        assert_eq!(operation.get_num_operands(), 1);
        assert_eq!(operation.get_num_results(), 1);
        let result_ty = operation.get_result(0).get_type(ctx);
        let result_ty = result_ty.deref(ctx);
        if register_count == 1 {
            let result_ty = result_ty
                .downcast_ref::<IntegerType>()
                .expect("x1 returns i32");
            assert_eq!(result_ty.width(), 32);
        } else {
            let result_ty = result_ty
                .downcast_ref::<llvm_types::StructType>()
                .expect("x2/x4 return an LLVM struct");
            assert_eq!(result_ty.num_fields(), register_count);
            for index in 0..result_ty.num_fields() {
                let field = result_ty.field_type(index);
                let field = field.deref(ctx);
                let field = field
                    .downcast_ref::<IntegerType>()
                    .expect("fragment register is i32");
                assert_eq!(field.width(), 32);
            }
        }
    }

    let register_counts = [1, 1, 2, 2, 4, 4];
    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_all_ldmatrix_forms(3, backend, true)?;
        let mut seen = [0; 6];
        let mut extract_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());
            match backend {
                mir_lower::IntrinsicBackend::LlvmNvptx => {
                    assert!(Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none());
                    let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                        continue;
                    };
                    let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                        panic!("ldmatrix intrinsic call must be direct");
                    };
                    let callee = callee.to_string();
                    let index = LDMATRIX_TYPED_INTRINSICS
                        .iter()
                        .position(|expected| *expected == callee)
                        .expect("exact typed ldmatrix intrinsic");
                    seen[index] += 1;
                    assert_result_shape(&ctx, op, register_counts[index]);
                }
                mir_lower::IntrinsicBackend::LibNvvm => {
                    assert!(Operation::get_op::<llvm::CallOp>(op, &ctx).is_none());
                    let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                        continue;
                    };
                    let template = inline_asm
                        .get_attr_llvm_inline_asm_template(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .unwrap_or_default();
                    let index = LDMATRIX_PTX_TEMPLATES
                        .iter()
                        .position(|expected| *expected == template)
                        .expect("exact ldmatrix PTX template");
                    seen[index] += 1;
                    assert_eq!(
                        inline_asm
                            .get_attr_llvm_inline_asm_constraints(&ctx)
                            .map(|value| String::from((*value).clone()))
                            .as_deref(),
                        Some(LDMATRIX_PTX_CONSTRAINTS[index])
                    );
                    assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                    assert_result_shape(&ctx, op, register_counts[index]);
                }
            }
        }

        assert_eq!(seen, [1; 6]);
        assert_eq!(extract_count, 12, "x2/x4 results keep their order");
    }
    Ok(())
}

#[test]
fn test_ldmatrix_rejects_non_shared_pointer_spaces() {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for address_space in [1, 4, 5] {
            let mut ctx = make_test_ctx();
            let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
            let ptr_ty = MirPtrType::get(&mut ctx, u32_ty.into(), false, address_space);
            let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
            let pointer = entry.deref(&ctx).get_argument(0);
            nvvm::LdmatrixOp::build(
                &mut ctx,
                pointer,
                nvvm::LdmatrixShapeAttr::M8n8,
                nvvm::LdmatrixMultiplicityAttr::X1,
                nvvm::LdmatrixLayoutAttr::Normal,
                nvvm::LdmatrixElementAttr::B16,
                nvvm::LdmatrixStateSpaceAttr::Shared,
            )
            .insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);

            let error = mir_lower::lower_mir_to_llvm_with_options(
                &mut ctx,
                module_ptr,
                mir_lower::LoweringOptions {
                    intrinsic_backend: backend,
                    ..Default::default()
                },
            )
            .expect_err("global/constant pointers must fail closed")
            .to_string();
            assert!(
                error.contains(&format!("not address space {address_space}")),
                "{error}"
            );
        }
    }
}

#[test]
fn test_ldmatrix_rejects_non_pointer_non_u32_operand() {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![u64_ty.into()]);
    let not_an_address = entry.deref(&ctx).get_argument(0);
    nvvm::LdmatrixOp::build(
        &mut ctx,
        not_an_address,
        nvvm::LdmatrixShapeAttr::M8n8,
        nvvm::LdmatrixMultiplicityAttr::X1,
        nvvm::LdmatrixLayoutAttr::Normal,
        nvvm::LdmatrixElementAttr::B16,
        nvvm::LdmatrixStateSpaceAttr::Shared,
    )
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    let error = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("non-pointer non-u32 ldmatrix input must fail closed")
        .to_string();
    assert!(error.contains("pointer or u32 shared address"), "{error}");
}

#[test]
fn test_ldmatrix_rejects_wrong_result_arity() {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
    let pointer = entry.deref(&ctx).get_argument(0);
    let op = Operation::new(
        &mut ctx,
        nvvm::LdmatrixOp::get_concrete_op_info(),
        vec![i32_ty.into(), i32_ty.into()],
        vec![pointer],
        vec![],
        0,
    );
    let ldmatrix = nvvm::LdmatrixOp::new(op);
    ldmatrix.set_attr_nvvm_ldmatrix_shape(&ctx, nvvm::LdmatrixShapeAttr::M8n8);
    ldmatrix.set_attr_nvvm_ldmatrix_multiplicity(&ctx, nvvm::LdmatrixMultiplicityAttr::X1);
    ldmatrix.set_attr_nvvm_ldmatrix_layout(&ctx, nvvm::LdmatrixLayoutAttr::Normal);
    ldmatrix.set_attr_nvvm_ldmatrix_element(&ctx, nvvm::LdmatrixElementAttr::B16);
    ldmatrix.set_attr_nvvm_ldmatrix_state_space(&ctx, nvvm::LdmatrixStateSpaceAttr::Shared);
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    let error = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("x1 must return exactly one register")
        .to_string();
    assert!(error.contains("requires 1 u32 results"), "{error}");
}
