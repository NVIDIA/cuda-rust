/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayIntrinsic, RegisterMma,
    RegisterMmaAccumulator, RegisterMmaAdapter, RegisterMmaCompatibilitySource, RegisterMmaKind,
    RegisterMmaLayout, RegisterMmaOperation, RegisterMmaOverflow, RegisterMmaParticipation,
    RegisterMmaShape, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeSet;

use super::*;
use crate::resolve::guards::*;

pub(in crate::resolve) fn validate_register_mma_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let mma = policy
        .register_mma
        .as_ref()
        .with_context(|| format!("{} has no closed register-MMA contract", policy.id))?;
    match mma.kind {
        Some(RegisterMmaKind::Standard) => {
            return validate_register_mma_fp8_policy(policy, declaration, mma);
        }
        Some(RegisterMmaKind::F8f6f4) => {
            return validate_register_mma_f8f6f4_policy(policy, declaration, mma);
        }
        Some(RegisterMmaKind::Mxf8f6f4) => {
            return validate_register_mma_mxf8f6f4_policy(policy, declaration, mma);
        }
        None if register_mma_f8f6f4_element_name(mma.a_element).is_some()
            || register_mma_f8f6f4_element_name(mma.b_element).is_some() =>
        {
            return validate_register_mma_f8f6f4_policy(policy, declaration, mma);
        }
        None => {}
    }
    let recipe = register_mma_recipe(mma)
        .with_context(|| format!("{} requests an unsupported register-MMA variant", policy.id))?;
    let abi_matches = matches!(
        recipe.id,
        "mma_m16n8k4_f32_tf32"
            | "mma_m16n8k8_f16_f16"
            | "mma_m16n8k8_f32_bf16"
            | "mma_m16n8k8_f32_f16"
            | "mma_m16n8k16_f16_f16"
    ) || policy.abi_id == recipe.abi_id;
    ensure!(
        policy.id == recipe.id
            && abi_matches
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} register-MMA identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.compatibility_rust_paths == [format!("cuda_device::wmma::{}", recipe.id)],
        "{} must preserve its unsafe must-use Rust MMA API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.ptx_result == recipe.rust_result
            && mma.adapter == recipe.adapter
            && mma.compatibility_source == recipe.compatibility_source
            && policy.lowering == "generated_register_mma",
        "{} register-MMA carrier or lowering adapter disagrees with its recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.targets == "all",
        "{} register-MMA effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma",
        "{} register-MMA PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.classes.iter().any(|class| class == "NVVM_MMA")
            && declaration.properties == ["IntrNoCallback", "IntrNoMem"]
            && declaration.selections.len() == 1
            && selection_matches_policy(policy, &declaration.selections[0])?,
        "{} imported MMA declaration changed its class, properties, or exact selection contract",
        policy.id
    );
    // These are imported facts. The BF16 selection is underconstrained.
    let ampere_float_predicates = match recipe.id {
        "mma_m16n8k4_f32_tf32" | "mma_m16n8k16_f16_f16" => Some([
            "Subtarget->getSmVersion() >= 80",
            "Subtarget->getPTXVersion() >= 70",
        ]),
        "mma_m16n8k8_f16_f16" | "mma_m16n8k8_f32_bf16" | "mma_m16n8k8_f32_f16" => Some([
            "Subtarget->getPTXVersion() >= 65",
            "Subtarget->getSmVersion() >= 75",
        ]),
        _ => None,
    };
    if let Some(predicates) = ampere_float_predicates {
        let selection = &declaration.selections[0];
        ensure!(
            declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_MMA"]
                && selection.predicates == predicates
                && selection.constraints.is_empty()
                && mma.kind.is_none()
                && mma.runtime_validation == RuntimeValidation::Unexecuted,
            "{} imported Ampere floating-point MMA contract changed",
            policy.id
        );
    }
    ensure!(
        policy.expected_ptx.mnemonic == "mma"
            && policy.expected_ptx.modifiers == recipe.ptx_modifiers
            && policy.expected_ptx.operands
                == recipe
                    .ptx_register_counts
                    .map(|length| OperandPattern::RegisterList { length }),
        "{} expected PTX does not match its exact register-MMA spelling",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (
                IntrinsicBackend::LlvmNvptx,
                recipe.minimum_ptx,
                Some(recipe.minimum_sm),
            ),
            (
                IntrinsicBackend::LibNvvm,
                recipe.minimum_ptx,
                Some(recipe.minimum_sm),
            ),
        ],
        "register MMA",
    )?;
    ensure_no_other_family_contract(policy, "register MMA")?;
    Ok(())
}

pub(in crate::resolve) fn validate_register_mma_fp8_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    mma: &RegisterMma,
) -> Result<()> {
    ensure!(
        REGISTER_MMA_FP8_SHAPES.contains(&mma.shape),
        "{} has an unsupported standard FP8 shape",
        policy.id
    );
    ensure!(
        REGISTER_MMA_FP8_ACCUMULATORS.contains(&mma.accumulator),
        "{} has an unsupported standard FP8 accumulator",
        policy.id
    );
    ensure!(
        REGISTER_MMA_FP8_ELEMENTS.contains(&mma.a_element),
        "{} has an unsupported standard FP8 A element",
        policy.id
    );
    ensure!(
        REGISTER_MMA_FP8_ELEMENTS.contains(&mma.b_element),
        "{} has an unsupported standard FP8 B element",
        policy.id
    );
    let (shape_name, a_count, b_count) = register_mma_fp8_shape_contract(mma.shape)?;
    let minimum_ptx = register_mma_fp8_minimum_ptx(mma.shape, mma.accumulator);
    let a = register_mma_fp8_element_name(mma.a_element).expect("validated A element");
    let b = register_mma_fp8_element_name(mma.b_element).expect("validated B element");
    let (
        scalar,
        rust_arguments,
        rust_result,
        dialect_operands,
        dialect_results,
        llvm_arguments,
        llvm_results,
        adapter,
    ) = match mma.accumulator {
        RegisterMmaAccumulator::F16 => (
            "f16",
            vec![
                "[u32; 2]".into(),
                format!("[u32; {a_count}]"),
                if b_count == 1 {
                    "u32".into()
                } else {
                    "[u32; 2]".into()
                },
            ],
            "[u32; 2]".to_owned(),
            [
                vec!["i32".to_owned(); 2],
                vec!["i32".to_owned(); a_count],
                vec!["i32".to_owned(); b_count],
            ]
            .concat(),
            vec!["i32".to_owned(); 2],
            [
                vec!["i32".to_owned(); a_count + b_count],
                vec!["v2f16".to_owned(); 2],
            ]
            .concat(),
            vec!["v2f16".to_owned(); 2],
            if mma.shape == RegisterMmaShape::M16n8k16 {
                RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32
            } else {
                RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32
            },
        ),
        RegisterMmaAccumulator::F32 => (
            "f32",
            vec![
                "[f32; 4]".into(),
                format!("[u32; {a_count}]"),
                if b_count == 1 {
                    "u32".into()
                } else {
                    "[u32; 2]".into()
                },
            ],
            "[f32; 4]".to_owned(),
            [
                vec!["f32".to_owned(); 4],
                vec!["i32".to_owned(); a_count],
                vec!["i32".to_owned(); b_count],
            ]
            .concat(),
            vec!["f32".to_owned(); 4],
            [
                vec!["i32".to_owned(); a_count + b_count],
                vec!["f32".to_owned(); 4],
            ]
            .concat(),
            vec!["f32".to_owned(); 4],
            if mma.shape == RegisterMmaShape::M16n8k16 {
                RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32
            } else {
                RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32
            },
        ),
        _ => unreachable!("validated accumulator"),
    };
    let expected_id = format!("mma_{shape_name}_fp8_{scalar}_{a}_{b}");
    let expected_source = format!("int_nvvm_mma_{shape_name}_row_col_{scalar}_{a}_{b}_{scalar}");
    let expected_symbol = format!("llvm.nvvm.mma.{shape_name}.row.col.{scalar}.{a}.{b}.{scalar}");

    ensure!(
        policy.id == expected_id
            && policy.operation_key
                == format!(
                    "matrix.mma.{shape_name}.row.col.standard_fp8.{scalar}.{a}.{b}.{scalar}"
                )
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(expected_source.as_str())
            && policy.llvm_symbol.as_deref() == Some(expected_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} standard FP8 MMA identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == expected_id
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::matrix::{expected_id}")
            && policy.compatibility_rust_paths == [format!("cuda_device::wmma::{expected_id}")],
        "{} must preserve its unsafe must-use standard FP8 API",
        policy.id
    );
    ensure!(
        mma.kind == Some(RegisterMmaKind::Standard)
            && mma.operation == RegisterMmaOperation::Multiply
            && mma.a_layout == RegisterMmaLayout::Row
            && mma.b_layout == RegisterMmaLayout::Col
            && mma.overflow == RegisterMmaOverflow::NotApplicable
            && mma.participation
                == RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes
            && mma.adapter == adapter
            && mma.compatibility_source == RegisterMmaCompatibilitySource::GeneratedStub
            && mma.runtime_validation == RuntimeValidation::Unexecuted
            && policy.dialect_op_type == "RegisterMmaOp"
            && policy.dialect_op_name == "nvvm.register_mma"
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results == dialect_results
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results == llvm_results
            && policy.ptx_result == rust_result
            && policy.lowering == "generated_register_mma",
        "{} standard FP8 MMA carrier or lowering changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == minimum_ptx
            && policy.minimum_sm.as_deref() == Some("sm_89")
            && policy.targets == "all",
        "{} standard FP8 effects or target floor changed",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma",
        "{} standard FP8 PTX provenance changed",
        policy.id
    );
    let [selection] = declaration.selections.as_slice() else {
        bail!(
            "{} must retain exactly one standard FP8 selection",
            policy.id
        );
    };
    let expected_predicates = if mma.shape == RegisterMmaShape::M16n8k16 {
        [
            "Subtarget->getSmVersion() >= 89",
            "Subtarget->getPTXVersion() >= 87",
        ]
    } else {
        [
            "Subtarget->getSmVersion() >= 89",
            "Subtarget->getPTXVersion() >= 84",
        ]
    };
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_MMA"]
            && declaration.properties == ["IntrNoCallback", "IntrNoMem"]
            && selection_matches_policy(policy, selection)?
            && selection.predicates == expected_predicates
            && selection.constraints.is_empty(),
        "{} imported standard FP8 declaration or selection changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: [
                    "sync", "aligned", shape_name, "row", "col", scalar, a, b, scalar
                ]
                .into_iter()
                .map(Into::into)
                .collect(),
                operands: [
                    dialect_results.len(),
                    a_count,
                    b_count,
                    dialect_results.len()
                ]
                .map(|length| OperandPattern::RegisterList { length })
                .into(),
            },
        "{} expected standard FP8 PTX changed",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, minimum_ptx, Some("sm_89")),
            (IntrinsicBackend::LibNvvm, minimum_ptx, Some("sm_89")),
        ],
        "standard FP8 register MMA",
    )?;
    ensure_no_other_family_contract(policy, "standard FP8 register MMA")?;
    Ok(())
}

pub(in crate::resolve) fn validate_register_mma_f8f6f4_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    mma: &RegisterMma,
) -> Result<()> {
    let contract = register_mma_f8f6f4_contract(mma.accumulator)?;
    let a = register_mma_f8f6f4_element_name(mma.a_element)
        .with_context(|| format!("{} has a non-f8f6f4 A format", policy.id))?;
    let b = register_mma_f8f6f4_element_name(mma.b_element)
        .with_context(|| format!("{} has a non-f8f6f4 B format", policy.id))?;
    let scalar = contract.scalar_name;
    let expected_id = format!("mma_m16n8k32_{scalar}_{a}_{b}");
    let expected_source =
        format!("int_nvvm_mma_m16n8k32_row_col_kind_f8f6f4_{scalar}_{a}_{b}_{scalar}");
    let expected_symbol =
        format!("llvm.nvvm.mma.m16n8k32.row.col.kind.f8f6f4.{scalar}.{a}.{b}.{scalar}");
    ensure!(
        policy.id == expected_id
            && policy.operation_key
                == format!("matrix.mma.m16n8k32.row.col.kind_f8f6f4.{scalar}.{a}.{b}.{scalar}")
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(expected_source.as_str())
            && policy.llvm_symbol.as_deref() == Some(expected_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} dense f8f6f4 MMA identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == expected_id
            && policy.rust_arguments == contract.rust_arguments
            && policy.rust_result == contract.rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::matrix::{expected_id}")
            && policy.compatibility_rust_paths == [format!("cuda_device::wmma::{expected_id}")],
        "{} must preserve the unsafe must-use dense f8f6f4 API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "RegisterMmaOp"
            && policy.dialect_op_name == "nvvm.register_mma"
            && policy.dialect_operands == contract.dialect_operands
            && policy.dialect_results == contract.dialect_results
            && policy.llvm_arguments == contract.llvm_arguments
            && policy.llvm_results == contract.llvm_results
            && policy.ptx_result == contract.rust_result
            && mma.shape == RegisterMmaShape::M16n8k32
            && matches!(mma.kind, None | Some(RegisterMmaKind::F8f6f4))
            && mma.operation == RegisterMmaOperation::Multiply
            && mma.accumulator == contract.accumulator
            && mma.a_layout == RegisterMmaLayout::Row
            && mma.b_layout == RegisterMmaLayout::Col
            && mma.overflow == RegisterMmaOverflow::NotApplicable
            && mma.participation
                == RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes
            && mma.adapter == contract.adapter
            && mma.compatibility_source == RegisterMmaCompatibilitySource::GeneratedStub
            && mma.runtime_validation == RuntimeValidation::Unexecuted
            && policy.lowering == "generated_register_mma",
        "{} dense f8f6f4 carrier or lowering changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "8.7"
            && policy.minimum_sm.is_none()
            && policy.targets == REGISTER_MMA_F8F6F4_TARGETS,
        "{} dense f8f6f4 effects or exact target set changed",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma",
        "{} dense f8f6f4 PTX provenance changed",
        policy.id
    );
    let [selection] = declaration.selections.as_slice() else {
        bail!(
            "{} must retain exactly one imported dense f8f6f4 instruction selection",
            policy.id
        );
    };
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_MMA"]
            && declaration.properties == ["IntrNoCallback", "IntrNoMem"]
            && selection_matches_policy(policy, selection)?
            && selection.predicates == ["Subtarget->hasMMABlockScale()"]
            && selection.constraints.is_empty(),
        "{} imported dense f8f6f4 declaration or selection changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: [
                    "sync",
                    "aligned",
                    "m16n8k32",
                    "row",
                    "col",
                    "kind::f8f6f4",
                    scalar,
                    a,
                    b,
                    scalar,
                ]
                .into_iter()
                .map(Into::into)
                .collect(),
                operands: contract
                    .ptx_register_counts
                    .map(|length| OperandPattern::RegisterList { length })
                    .into(),
            },
        "{} expected dense f8f6f4 PTX changed",
        policy.id
    );
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.targets.is_none()
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must inherit the exact reviewed target set on both inline-PTX routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "dense f8f6f4 register MMA")?;
    Ok(())
}

pub(in crate::resolve) fn validate_register_mma_mxf8f6f4_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    mma: &RegisterMma,
) -> Result<()> {
    let a = register_mma_f8f6f4_element_name(mma.a_element)
        .with_context(|| format!("{} has a non-mxf8f6f4 A format", policy.id))?;
    let b = register_mma_f8f6f4_element_name(mma.b_element)
        .with_context(|| format!("{} has a non-mxf8f6f4 B format", policy.id))?;
    let expected_id = format!("mma_m16n8k32_mxf8f6f4_f32_{a}_{b}");
    let expected_source =
        format!("int_nvvm_mma_block_scale_m16n8k32_row_col_mxf8f6f4_f32_{a}_{b}_f32_ue8m0");
    let expected_symbol =
        format!("llvm.nvvm.mma.block.scale.m16n8k32.row.col.mxf8f6f4.f32.{a}.{b}.f32.ue8m0");
    let rust_arguments = [
        "[f32; 4]", "[u32; 4]", "[u32; 2]", "u32", "u16", "u16", "u32", "u16", "u16",
    ];
    let dialect_operands = [
        "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i16", "i16",
        "i32", "i16", "i16",
    ];
    let llvm_arguments = [
        "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32", "i32", "i16", "i16",
        "i32", "i16", "i16",
    ];
    let results = ["f32", "f32", "f32", "f32"];
    ensure!(
        policy.id == expected_id
            && policy.operation_key
                == format!(
                    "matrix.mma.m16n8k32.row.col.kind_mxf8f6f4.scale_vec_1x.f32.{a}.{b}.f32.ue8m0"
                )
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(expected_source.as_str())
            && policy.llvm_symbol.as_deref() == Some(expected_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} dense mxf8f6f4 MMA identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == expected_id
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "[f32; 4]"
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::matrix::{expected_id}")
            && policy.compatibility_rust_paths == [format!("cuda_device::wmma::{expected_id}")],
        "{} must preserve the unsafe must-use dense mxf8f6f4 API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "RegisterMmaOp"
            && policy.dialect_op_name == "nvvm.register_mma"
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results == results
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results == results
            && policy.ptx_result == "[f32; 4]"
            && mma.shape == RegisterMmaShape::M16n8k32
            && mma.kind == Some(RegisterMmaKind::Mxf8f6f4)
            && mma.operation == RegisterMmaOperation::Multiply
            && mma.accumulator == RegisterMmaAccumulator::F32
            && mma.a_layout == RegisterMmaLayout::Row
            && mma.b_layout == RegisterMmaLayout::Col
            && mma.overflow == RegisterMmaOverflow::NotApplicable
            && mma.participation
                == RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes
            && mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32
            && mma.compatibility_source == RegisterMmaCompatibilitySource::GeneratedStub
            && mma.runtime_validation == RuntimeValidation::Unexecuted
            && policy.lowering == "generated_register_mma",
        "{} dense mxf8f6f4 carrier or lowering changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "8.7"
            && policy.minimum_sm.is_none()
            && policy.targets == REGISTER_MMA_F8F6F4_TARGETS,
        "{} dense mxf8f6f4 effects or exact target set changed",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma",
        "{} dense mxf8f6f4 PTX provenance changed",
        policy.id
    );
    let [selection] = declaration.selections.as_slice() else {
        bail!(
            "{} must retain exactly one imported dense mxf8f6f4 instruction selection",
            policy.id
        );
    };
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_MMA_BLOCK_SCALE"]
            && declaration.properties == ["IntrNoCallback", "IntrNoMem"]
            && selection_matches_policy(policy, selection)?
            && selection.predicates == ["Subtarget->hasMMABlockScale()"]
            && selection.constraints.is_empty(),
        "{} imported dense mxf8f6f4 declaration or selection changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: [
                    "sync",
                    "aligned",
                    "m16n8k32",
                    "row",
                    "col",
                    "kind::mxf8f6f4",
                    "block_scale",
                    "f32",
                    a,
                    b,
                    "f32",
                    "ue8m0",
                ]
                .into_iter()
                .map(Into::into)
                .collect(),
                operands: vec![
                    OperandPattern::RegisterList { length: 4 },
                    OperandPattern::RegisterList { length: 4 },
                    OperandPattern::RegisterList { length: 2 },
                    OperandPattern::RegisterList { length: 4 },
                    OperandPattern::Register,
                    OperandPattern::RegisterList { length: 2 },
                    OperandPattern::Register,
                    OperandPattern::RegisterList { length: 2 },
                ],
            },
        "{} expected dense mxf8f6f4 PTX changed",
        policy.id
    );
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.targets.is_none()
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must inherit the exact reviewed target set on both inline-PTX routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "dense mxf8f6f4 register MMA")?;
    Ok(())
}
