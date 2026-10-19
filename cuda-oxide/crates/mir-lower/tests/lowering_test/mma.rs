/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};

#[test]
fn test_generated_register_mma_variants_lower_to_exact_convergent_inline_ptx()
-> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    #[derive(Clone, Copy)]
    enum Carrier {
        F32,
        F64,
        I32,
        U32,
    }

    struct Case {
        shape: nvvm::RegisterMmaShapeAttr,
        operation: nvvm::RegisterMmaOperationAttr,
        accumulator: nvvm::RegisterMmaAccumulatorAttr,
        a_element: nvvm::RegisterMmaElementAttr,
        b_element: nvvm::RegisterMmaElementAttr,
        overflow: nvvm::RegisterMmaOverflowAttr,
        operands: &'static [Carrier],
        results: &'static [Carrier],
        template: String,
        constraints: &'static str,
    }

    let mut cases = vec![
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M16n8k16,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F32,
            a_element: nvvm::RegisterMmaElementAttr::Bf16,
            b_element: nvvm::RegisterMmaElementAttr::Bf16,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
            ],
            results: &[Carrier::F32; 4],
            template: concat!(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
                "{$0, $1, $2, $3}, {$8, $9, $10, $11}, ",
                "{$12, $13}, {$4, $5, $6, $7};"
            )
            .into(),
            constraints: "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        },
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M16n8k16,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F32,
            a_element: nvvm::RegisterMmaElementAttr::F16,
            b_element: nvvm::RegisterMmaElementAttr::F16,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
            ],
            results: &[Carrier::F32; 4],
            template: concat!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
                "{$0, $1, $2, $3}, {$8, $9, $10, $11}, ",
                "{$12, $13}, {$4, $5, $6, $7};"
            )
            .into(),
            constraints: "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        },
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M16n8k8,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F32,
            a_element: nvvm::RegisterMmaElementAttr::Tf32,
            b_element: nvvm::RegisterMmaElementAttr::Tf32,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
            ],
            results: &[Carrier::F32; 4],
            template: concat!(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
                "{$0, $1, $2, $3}, {$8, $9, $10, $11}, ",
                "{$12, $13}, {$4, $5, $6, $7};"
            )
            .into(),
            constraints: "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        },
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M8n8k4,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F64,
            a_element: nvvm::RegisterMmaElementAttr::F64,
            b_element: nvvm::RegisterMmaElementAttr::F64,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[Carrier::F64; 4],
            results: &[Carrier::F64; 2],
            template: concat!(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 ",
                "{$0, $1}, {$4}, {$5}, {$2, $3};"
            )
            .into(),
            constraints: "=d,=d,d,d,d,d",
        },
    ];
    let c2_a1_b1: &'static [Carrier] = &[Carrier::I32, Carrier::I32, Carrier::U32, Carrier::U32];
    let c4_a2_b1: &'static [Carrier] = &[
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
    ];
    let c4_a4_b2: &'static [Carrier] = &[
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
    ];
    let d2_i32: &'static [Carrier] = &[Carrier::I32; 2];
    let d4_i32: &'static [Carrier] = &[Carrier::I32; 4];
    let register_list = |first, count| {
        format!(
            "{{{}}}",
            (first..first + count)
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    for (shape, shape_attr, operands, results, accumulator_count, a_count, b_count, constraints) in [
        (
            "m8n8k16",
            nvvm::RegisterMmaShapeAttr::M8n8k16,
            c2_a1_b1,
            d2_i32,
            2,
            1,
            1,
            "=r,=r,r,r,r,r",
        ),
        (
            "m16n8k16",
            nvvm::RegisterMmaShapeAttr::M16n8k16,
            c4_a2_b1,
            d4_i32,
            4,
            2,
            1,
            "=r,=r,=r,=r,r,r,r,r,r,r,r",
        ),
        (
            "m16n8k32",
            nvvm::RegisterMmaShapeAttr::M16n8k32,
            c4_a4_b2,
            d4_i32,
            4,
            4,
            2,
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        ),
    ] {
        for (a_name, a_element) in [
            ("s8", nvvm::RegisterMmaElementAttr::S8),
            ("u8", nvvm::RegisterMmaElementAttr::U8),
        ] {
            for (b_name, b_element) in [
                ("s8", nvvm::RegisterMmaElementAttr::S8),
                ("u8", nvvm::RegisterMmaElementAttr::U8),
            ] {
                for (overflow_name, overflow) in [
                    ("", nvvm::RegisterMmaOverflowAttr::Wrapping),
                    (".satfinite", nvvm::RegisterMmaOverflowAttr::Satfinite),
                ] {
                    let result_count = results.len();
                    let d = register_list(0, result_count);
                    let c = register_list(result_count, accumulator_count);
                    let a = register_list(result_count + accumulator_count, a_count);
                    let b = register_list(result_count + accumulator_count + a_count, b_count);
                    cases.push(Case {
                        shape: shape_attr.clone(),
                        operation: nvvm::RegisterMmaOperationAttr::Multiply,
                        accumulator: nvvm::RegisterMmaAccumulatorAttr::S32,
                        a_element: a_element.clone(),
                        b_element: b_element.clone(),
                        overflow,
                        operands,
                        results,
                        template: format!(
                            "mma.sync.aligned.{shape}.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {d}, {a}, {b}, {c};"
                        ),
                        constraints,
                    });
                }
            }
        }
    }

    for (shape, shape_attr, operands, results, accumulator_count, a_count, b_count, constraints) in [
        (
            "m8n8k32",
            nvvm::RegisterMmaShapeAttr::M8n8k32,
            c2_a1_b1,
            d2_i32,
            2,
            1,
            1,
            "=r,=r,r,r,r,r",
        ),
        (
            "m16n8k32",
            nvvm::RegisterMmaShapeAttr::M16n8k32,
            c4_a2_b1,
            d4_i32,
            4,
            2,
            1,
            "=r,=r,=r,=r,r,r,r,r,r,r,r",
        ),
        (
            "m16n8k64",
            nvvm::RegisterMmaShapeAttr::M16n8k64,
            c4_a4_b2,
            d4_i32,
            4,
            4,
            2,
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        ),
    ] {
        for (a_name, a_element) in [
            ("s4", nvvm::RegisterMmaElementAttr::S4),
            ("u4", nvvm::RegisterMmaElementAttr::U4),
        ] {
            for (b_name, b_element) in [
                ("s4", nvvm::RegisterMmaElementAttr::S4),
                ("u4", nvvm::RegisterMmaElementAttr::U4),
            ] {
                for (overflow_name, overflow) in [
                    ("", nvvm::RegisterMmaOverflowAttr::Wrapping),
                    (".satfinite", nvvm::RegisterMmaOverflowAttr::Satfinite),
                ] {
                    let result_count = results.len();
                    let d = register_list(0, result_count);
                    let c = register_list(result_count, accumulator_count);
                    let a = register_list(result_count + accumulator_count, a_count);
                    let b = register_list(result_count + accumulator_count + a_count, b_count);
                    cases.push(Case {
                        shape: shape_attr.clone(),
                        operation: nvvm::RegisterMmaOperationAttr::Multiply,
                        accumulator: nvvm::RegisterMmaAccumulatorAttr::S32,
                        a_element: a_element.clone(),
                        b_element: b_element.clone(),
                        overflow,
                        operands,
                        results,
                        template: format!(
                            "mma.sync.aligned.{shape}.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {d}, {a}, {b}, {c};"
                        ),
                        constraints,
                    });
                }
            }
        }
    }

    for (shape, shape_attr, operands, results, accumulator_count, a_count, b_count, constraints) in [
        (
            "m8n8k128",
            nvvm::RegisterMmaShapeAttr::M8n8k128,
            c2_a1_b1,
            d2_i32,
            2,
            1,
            1,
            "=r,=r,r,r,r,r",
        ),
        (
            "m16n8k128",
            nvvm::RegisterMmaShapeAttr::M16n8k128,
            c4_a2_b1,
            d4_i32,
            4,
            2,
            1,
            "=r,=r,=r,=r,r,r,r,r,r,r,r",
        ),
        (
            "m16n8k256",
            nvvm::RegisterMmaShapeAttr::M16n8k256,
            c4_a4_b2,
            d4_i32,
            4,
            4,
            2,
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        ),
    ] {
        let result_count = results.len();
        let d = register_list(0, result_count);
        let c = register_list(result_count, accumulator_count);
        let a = register_list(result_count + accumulator_count, a_count);
        let b = register_list(result_count + accumulator_count + a_count, b_count);
        for (operation_name, operation) in [
            ("xor", nvvm::RegisterMmaOperationAttr::XorPopc),
            ("and", nvvm::RegisterMmaOperationAttr::AndPopc),
        ] {
            cases.push(Case {
                shape: shape_attr.clone(),
                operation,
                accumulator: nvvm::RegisterMmaAccumulatorAttr::S32,
                a_element: nvvm::RegisterMmaElementAttr::B1,
                b_element: nvvm::RegisterMmaElementAttr::B1,
                overflow: nvvm::RegisterMmaOverflowAttr::Wrapping,
                operands,
                results,
                template: format!(
                    "mma.sync.aligned.{shape}.row.col.s32.b1.b1.s32.{operation_name}.popc {d}, {a}, {b}, {c};"
                ),
                constraints,
            });
        }
    }
    assert_eq!(cases.len(), 58);

    let carrier_type = |ctx: &Context, carrier: Carrier| -> TypeHandle {
        match carrier {
            Carrier::F32 => FP32Type::get(ctx).into(),
            Carrier::F64 => FP64Type::get(ctx).into(),
            Carrier::I32 => IntegerType::get(ctx, 32, Signedness::Signed).into(),
            Carrier::U32 => IntegerType::get(ctx, 32, Signedness::Unsigned).into(),
        }
    };

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for case in &cases {
            let mut ctx = make_test_ctx();
            let argument_types = case
                .operands
                .iter()
                .map(|carrier| carrier_type(&ctx, *carrier))
                .collect();
            let result_types = case
                .results
                .iter()
                .map(|carrier| carrier_type(&ctx, *carrier))
                .collect();
            let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
            let operands = (0..case.operands.len())
                .map(|index| entry.deref(&ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                &mut ctx,
                nvvm::RegisterMmaOp::get_concrete_op_info(),
                result_types,
                operands,
                vec![],
                0,
            );
            let mma = nvvm::RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(&ctx, case.shape.clone());
            let uses_legacy_default = matches!(
                (&case.operation, &case.a_element),
                (
                    nvvm::RegisterMmaOperationAttr::Multiply,
                    nvvm::RegisterMmaElementAttr::Bf16
                )
            );
            if !uses_legacy_default {
                mma.set_attr_nvvm_register_mma_operation(&ctx, case.operation.clone());
            }
            mma.set_attr_nvvm_register_mma_accumulator(&ctx, case.accumulator.clone());
            mma.set_attr_nvvm_register_mma_a_element(&ctx, case.a_element.clone());
            mma.set_attr_nvvm_register_mma_b_element(&ctx, case.b_element.clone());
            mma.set_attr_nvvm_register_mma_a_layout(&ctx, nvvm::RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(&ctx, nvvm::RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(&ctx, case.overflow.clone());
            operation.insert_at_back(entry, &ctx);
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

            let body = lowered_kernel_body(&ctx, module_ptr);
            let lowered = body
                .iter()
                .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                .collect::<Vec<_>>();
            assert_eq!(lowered.len(), 1, "{:?}", backend);
            let asm = &lowered[0];
            assert_eq!(
                asm.get_attr_llvm_inline_asm_template(&ctx)
                    .as_deref()
                    .map(|value| String::from(value.clone())),
                Some(case.template.clone())
            );
            assert_eq!(
                asm.get_attr_llvm_inline_asm_constraints(&ctx)
                    .as_deref()
                    .map(|value| String::from(value.clone())),
                Some(case.constraints.to_string())
            );
            assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
            assert_eq!(
                asm.get_operation().deref(&ctx).get_num_operands(),
                case.operands.len()
            );
            assert_eq!(asm.get_operation().deref(&ctx).get_num_results(), 1);
            assert_eq!(
                body.iter()
                    .filter(|op| Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some())
                    .count(),
                case.results.len()
            );

            let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
            let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                .expect("generated register MMA exports to LLVM IR");
            assert!(ir.contains("asm sideeffect"), "{ir}");
            assert!(ir.contains("{ convergent }"), "{ir}");
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_variants_lower_to_exact_convergent_inline_ptx()
-> Result<(), anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s8",
            "s8",
            "",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s8",
            "u8",
            "",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u8",
            "u8",
            "",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u8",
            "s8",
            "",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s8",
            "s8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "s8",
            "u8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u8",
            "u8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u8",
            "s8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let metadata_modes = [
        ("sp", nvvm::SparseMmaMetadataAttr::Standard),
        ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in &metadata_modes {
            for (index, (a_name, b_name, overflow_name, a_element, b_element, overflow)) in
                cases.iter().enumerate()
            {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..5).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_value = (index % 2) as u32;
                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(
                        u32_ty,
                        APInt::from_u32(selector_value, NonZeroUsize::new(32).unwrap()),
                    ),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..9)
                    .map(|operand| entry.deref(&ctx).get_argument(operand))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k32);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZeroOrOne,
                );
                operation.insert_at_back(entry, &ctx);
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

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k32.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9}}, {{$10, $11}}, {{$4, $5, $6, $7}}, $12, $13;"
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 10);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(9);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = (&*attribute as &dyn pliron::attribute::Attribute)
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().bw(), 32);
                assert_eq!(integer.value().to_u64(), selector_value as u64);
                assert_eq!(
                    body.iter()
                        .filter(|op| {
                            Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some()
                        })
                        .count(),
                    4
                );

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_m16n8k64_lowers_to_exact_convergent_inline_ptx()
-> Result<(), anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s8",
            "u8",
            "",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u8",
            "s8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let metadata_cases = [
        ("sp", nvvm::SparseMmaMetadataAttr::Standard),
        ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in &metadata_cases {
            for (a_name, b_name, overflow_name, a_element, b_element, overflow) in &cases {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..9).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap())),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..13)
                    .map(|index| entry.deref(&ctx).get_argument(index))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k64);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZero,
                );
                operation.insert_at_back(entry, &ctx);
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

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k64.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9, $10, $11}}, {{$12, $13, $14, $15}}, {{$4, $5, $6, $7}}, $16, $17;"
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 14);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(13);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = (&*attribute as &dyn pliron::attribute::Attribute)
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().bw(), 32);
                assert_eq!(integer.value().to_u64(), 0);
                assert_eq!(
                    body.iter()
                        .filter(|op| {
                            Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some()
                        })
                        .count(),
                    4
                );

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_m16n8k64_int4_lowers_both_metadata_modes() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "s4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in [
            ("sp", nvvm::SparseMmaMetadataAttr::Standard),
            ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
        ] {
            for (index, (a_name, b_name, overflow_name, a_element, b_element, overflow)) in
                cases.iter().enumerate()
            {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..5).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_value = (index % 2) as u32;
                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(
                        u32_ty,
                        APInt::from_u32(selector_value, NonZeroUsize::new(32).unwrap()),
                    ),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..9)
                    .map(|operand| entry.deref(&ctx).get_argument(operand))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k64);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZeroOrOne,
                );
                operation.insert_at_back(entry, &ctx);
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

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k64.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9}}, {{$10, $11}}, {{$4, $5, $6, $7}}, $12, $13;"
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 10);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(9);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = (&*attribute as &dyn pliron::attribute::Attribute)
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().to_u64(), selector_value as u64);

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_m16n8k128_int4_lowers_both_metadata_modes() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "s4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in [
            ("sp", nvvm::SparseMmaMetadataAttr::Standard),
            ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
        ] {
            for (a_name, b_name, overflow_name, a_element, b_element, overflow) in &cases {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..9).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap())),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..13)
                    .map(|operand| entry.deref(&ctx).get_argument(operand))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k128);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZero,
                );
                operation.insert_at_back(entry, &ctx);
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

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k128.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9, $10, $11}}, {{$12, $13, $14, $15}}, {{$4, $5, $6, $7}}, $16, $17;"
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_llvm_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 14);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(13);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = (&*attribute as &dyn pliron::attribute::Attribute)
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().to_u64(), 0);

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 bf16 intrinsic lowering test
// ---------------------------------------------------------------------------

#[test]
fn test_mma_m16n8k16_f32_bf16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..4)
        .map(|_| f32_ty.into())
        .chain((0..6).map(|_| i32_ty.into()))
        .collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands,
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
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template.as_deref().is_some_and(|t| {
                    t.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
                }) {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
                        "{$0, $1, $2, $3}, ",
                        "{$8, $9, $10, $11}, ",
                        "{$12, $13}, ",
                        "{$4, $5, $6, $7};"
                    )
                );
                for forbidden in [".reg", "ld.", "st.", "["] {
                    assert!(
                        !template.contains(forbidden),
                        "register-only MMA must not contain {forbidden:?}: {template}"
                    );
                }
                assert_eq!(
                    constraints.as_deref(),
                    Some("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    10,
                    "expected C, A, and B scalar register operands"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns the four D registers as one struct"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

#[test]
fn test_mma_m16n8k16_f32_f16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..4)
        .map(|_| f32_ty.into())
        .chain((0..6).map(|_| i32_ty.into()))
        .collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands,
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
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template.as_deref().is_some_and(|t| {
                    t.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
                }) {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
                        "{$0, $1, $2, $3}, ",
                        "{$8, $9, $10, $11}, ",
                        "{$12, $13}, ",
                        "{$4, $5, $6, $7};"
                    )
                );
                for forbidden in [".reg", "ld.", "st.", "["] {
                    assert!(
                        !template.contains(forbidden),
                        "register-only MMA must not contain {forbidden:?}: {template}"
                    );
                }
                assert_eq!(
                    constraints.as_deref(),
                    Some("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    10,
                    "expected C, A, and B scalar register operands"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns the four D registers as one struct"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

#[test]
fn test_mma_m16n8k8_f32_tf32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..4)
        .map(|_| f32_ty.into())
        .chain((0..6).map(|_| i32_ty.into()))
        .collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands,
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
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template.as_deref().is_some_and(|t| {
                    t.contains("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
                }) {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
                        "{$0, $1, $2, $3}, ",
                        "{$8, $9, $10, $11}, ",
                        "{$12, $13}, ",
                        "{$4, $5, $6, $7};"
                    )
                );
                for forbidden in [".reg", "ld.", "st.", "["] {
                    assert!(
                        !template.contains(forbidden),
                        "register-only MMA must not contain {forbidden:?}: {template}"
                    );
                }
                assert_eq!(
                    constraints.as_deref(),
                    Some("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    10,
                    "expected C, A, and B scalar register operands"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns the four D registers as one struct"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

#[test]
fn test_mma_m8n8k4_f64_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::FP64Type;

    let mut ctx = make_test_ctx();
    let f64_ty = FP64Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![f64_ty.into(), f64_ty.into(), f64_ty.into(), f64_ty.into()],
    );
    let operands = (0..4)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f64_ty.into()],
        operands,
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
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template
                    .as_deref()
                    .is_some_and(|t| t.contains("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"))
                {
                    continue;
                }
                found += 1;
                let template = template.unwrap();
                assert_eq!(
                    template,
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$4}, {$5}, {$2, $3};"
                );
                assert!(!template.contains(".reg"));
                assert!(!template.contains("ld."));
                assert!(!template.contains("st."));
                assert_eq!(constraints.as_deref(), Some("=d,=d,d,d,d,d"));
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    4,
                    "expected four register inputs (c0, c1, a, b)"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns one aggregate containing d0 and d1"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

#[test]
fn test_mma_m16n8k32_s32_s8_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..10).map(|_| i32_ty.into()).collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K32S32S8Op::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        operands,
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
                let template = asm
                    .get_attr_llvm_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_llvm_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template
                    .as_deref()
                    .is_some_and(|t| t.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"))
                {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
                        "{$0, $1, $2, $3}, ",
                        "{$8, $9, $10, $11}, ",
                        "{$12, $13}, ",
                        "{$4, $5, $6, $7};"
                    )
                );
                for forbidden in [".reg", "ld.", "st.", "["] {
                    assert!(
                        !template.contains(forbidden),
                        "register-only MMA must not contain {forbidden:?}: {template}"
                    );
                }
                assert_eq!(
                    constraints.as_deref(),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r")
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    10,
                    "expected C, A, and B scalar register operands"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns the four D registers as one struct"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

#[test]
fn test_generated_plain_sparse_fp8_m16n8k64_lowers_to_exact_convergent_inline_ptx()
-> Result<(), anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    // The plain (standard-metadata) SM89 sparse FP8 forms. Every combination of
    // the reviewed e4m3/e5m2 multiplicand pair must lower to its own exact PTX
    // spelling; mixing them up would silently compute the wrong product.
    let cases = [
        (
            "e4m3",
            "e4m3",
            nvvm::SparseMmaElementAttr::E4m3,
            nvvm::SparseMmaElementAttr::E4m3,
        ),
        (
            "e4m3",
            "e5m2",
            nvvm::SparseMmaElementAttr::E4m3,
            nvvm::SparseMmaElementAttr::E5m2,
        ),
        (
            "e5m2",
            "e4m3",
            nvvm::SparseMmaElementAttr::E5m2,
            nvvm::SparseMmaElementAttr::E4m3,
        ),
        (
            "e5m2",
            "e5m2",
            nvvm::SparseMmaElementAttr::E5m2,
            nvvm::SparseMmaElementAttr::E5m2,
        ),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (a_name, b_name, a_element, b_element) in &cases {
            let mut ctx = make_test_ctx();
            let float_ty = FP32Type::get(&ctx);
            let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| float_ty.into())
                .chain((0..9).map(|_| u32_ty.into()))
                .collect();
            let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

            let selector_op = Operation::new(
                &mut ctx,
                mir::MirConstantOp::get_concrete_op_info(),
                vec![u32_ty.into()],
                vec![],
                vec![],
                0,
            );
            mir::MirConstantOp::new(selector_op).set_attr_value(
                &ctx,
                IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap())),
            );
            selector_op.insert_at_back(entry, &ctx);
            let selector = selector_op.deref(&ctx).get_result(0);

            let operands = (0..13)
                .map(|index| entry.deref(&ctx).get_argument(index))
                .chain(std::iter::once(selector))
                .collect();
            let operation = Operation::new(
                &mut ctx,
                nvvm::SparseMmaOp::get_concrete_op_info(),
                vec![float_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = nvvm::SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::F32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, nvvm::SparseMmaOverflowAttr::NotApplicable);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, nvvm::SparseMmaMetadataAttr::Standard);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, nvvm::SparseMmaSelectorAttr::ImmediateZero);
            operation.insert_at_back(entry, &ctx);
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

            let body = lowered_kernel_body(&ctx, module_ptr);
            let lowered = body
                .iter()
                .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                .collect::<Vec<_>>();
            assert_eq!(lowered.len(), 1, "{backend:?} {a_name}x{b_name}");
            let asm = &lowered[0];
            let expected_template = format!(
                "mma.sp.sync.aligned.m16n8k64.row.col.f32.{a_name}.{b_name}.f32 {{$0, $1, $2, $3}}, {{$8, $9, $10, $11}}, {{$12, $13, $14, $15}}, {{$4, $5, $6, $7}}, $16, $17;"
            );
            assert_eq!(
                asm.get_attr_llvm_inline_asm_template(&ctx)
                    .as_deref()
                    .map(|value| String::from(value.clone())),
                Some(expected_template)
            );
            // Four f32 outputs, the f32 C fragment, the packed A/B registers,
            // the metadata register and the compile-time selector immediate.
            assert_eq!(
                asm.get_attr_llvm_inline_asm_constraints(&ctx)
                    .as_deref()
                    .map(|value| String::from(value.clone())),
                Some("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r,r,r,r,n".to_string())
            );
            assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
            let asm_operation = asm.get_operation().deref(&ctx);
            assert_eq!(asm_operation.get_num_operands(), 14);
            assert_eq!(asm_operation.get_num_results(), 1);
            let lowered_selector = asm_operation.get_operand(13);
            let defining_op = lowered_selector
                .defining_op()
                .expect("sparse MMA selector remains an LLVM constant");
            let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                .expect("sparse MMA selector remains an LLVM integer constant");
            let attribute = constant.get_value(&ctx);
            let integer = (&*attribute as &dyn pliron::attribute::Attribute)
                .downcast_ref::<IntegerAttr>()
                .expect("sparse MMA selector is an integer");
            assert_eq!(integer.value().bw(), 32);
            assert_eq!(integer.value().to_u64(), 0);
            assert_eq!(
                body.iter()
                    .filter(|op| Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some())
                    .count(),
                4
            );

            let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
            let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                .expect("generated sparse MMA exports to LLVM IR");
            assert!(ir.contains("asm sideeffect"), "{ir}");
            assert!(ir.contains("{ convergent }"), "{ir}");
            assert!(ir.contains(a_name), "{ir}");
            assert!(ir.contains(b_name), "{ir}");
        }
    }

    Ok(())
}
