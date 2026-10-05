/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayBackendLowering,
    OverlayIntrinsic, Prmt, PrmtAdapter, PrmtAdmission, PrmtMode, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;

#[derive(Clone, Copy)]
pub(in crate::resolve) struct PrmtRecipe {
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) modifier: Option<&'static str>,
    pub(in crate::resolve) adapter: PrmtAdapter,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn prmt_recipe(mode: PrmtMode) -> PrmtRecipe {
    use PrmtAdapter::{DirectThreeOperands, InsertZeroSecondSource};
    match mode {
        PrmtMode::Generic => PrmtRecipe {
            abi_id: "i0252",
            id: "prmt",
            operation_key: "integer.prmt.b32",
            source_record: "int_nvvm_prmt",
            llvm_symbol: "llvm.nvvm.prmt",
            modifier: None,
            adapter: DirectThreeOperands,
            summary: "Permutes bytes selected from two 32-bit inputs.",
        },
        PrmtMode::F4e => PrmtRecipe {
            abi_id: "i0253",
            id: "prmt_f4e",
            operation_key: "integer.prmt.b32.f4e",
            source_record: "int_nvvm_prmt_f4e",
            llvm_symbol: "llvm.nvvm.prmt.f4e",
            modifier: Some("f4e"),
            adapter: DirectThreeOperands,
            summary: "Permutes bytes with the forward four-byte extract mode.",
        },
        PrmtMode::B4e => PrmtRecipe {
            abi_id: "i0254",
            id: "prmt_b4e",
            operation_key: "integer.prmt.b32.b4e",
            source_record: "int_nvvm_prmt_b4e",
            llvm_symbol: "llvm.nvvm.prmt.b4e",
            modifier: Some("b4e"),
            adapter: DirectThreeOperands,
            summary: "Permutes bytes with the backward four-byte extract mode.",
        },
        PrmtMode::Rc8 => PrmtRecipe {
            abi_id: "i0255",
            id: "prmt_rc8",
            operation_key: "integer.prmt.b32.rc8",
            source_record: "int_nvvm_prmt_rc8",
            llvm_symbol: "llvm.nvvm.prmt.rc8",
            modifier: Some("rc8"),
            adapter: InsertZeroSecondSource,
            summary: "Replicates a selected byte across the 32-bit result.",
        },
        PrmtMode::Ecl => PrmtRecipe {
            abi_id: "i0256",
            id: "prmt_ecl",
            operation_key: "integer.prmt.b32.ecl",
            source_record: "int_nvvm_prmt_ecl",
            llvm_symbol: "llvm.nvvm.prmt.ecl",
            modifier: Some("ecl"),
            adapter: InsertZeroSecondSource,
            summary: "Clamps a byte extract toward the least-significant byte.",
        },
        PrmtMode::Ecr => PrmtRecipe {
            abi_id: "i0257",
            id: "prmt_ecr",
            operation_key: "integer.prmt.b32.ecr",
            source_record: "int_nvvm_prmt_ecr",
            llvm_symbol: "llvm.nvvm.prmt.ecr",
            modifier: Some("ecr"),
            adapter: InsertZeroSecondSource,
            summary: "Clamps a byte extract toward the most-significant byte.",
        },
        PrmtMode::Rc16 => PrmtRecipe {
            abi_id: "i0258",
            id: "prmt_rc16",
            operation_key: "integer.prmt.b32.rc16",
            source_record: "int_nvvm_prmt_rc16",
            llvm_symbol: "llvm.nvvm.prmt.rc16",
            modifier: Some("rc16"),
            adapter: InsertZeroSecondSource,
            summary: "Replicates a selected 16-bit half across the result.",
        },
    }
}

pub(in crate::resolve) fn expand_prmt_admission(
    admission: &PrmtAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "prmt runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact prmt admission requires both backend evidence profiles"
    );
    let expected_modes = BTreeSet::from([
        PrmtMode::Generic,
        PrmtMode::F4e,
        PrmtMode::B4e,
        PrmtMode::Rc8,
        PrmtMode::Ecl,
        PrmtMode::Ecr,
        PrmtMode::Rc16,
    ]);
    let actual_modes: BTreeSet<_> = admission
        .variants
        .iter()
        .map(|variant| variant.mode)
        .collect();
    ensure!(
        admission.variants.len() == expected_modes.len() && actual_modes == expected_modes,
        "compact prmt admission must contain each of the seven reviewed modes exactly once"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = prmt_recipe(variant.mode);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let three_operands = recipe.adapter == PrmtAdapter::DirectThreeOperands;
            let rust_arguments = vec!["u32".into(); if three_operands { 3 } else { 2 }];
            let llvm_arguments = vec!["i32".into(); if three_operands { 3 } else { 2 }];
            let mut modifiers = vec!["b32".into()];
            if let Some(modifier) = recipe.modifier {
                modifiers.push(modifier.into());
            }
            let operands = if three_operands {
                vec![
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ]
            } else {
                vec![
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Exact { value: "0".into() },
                    OperandPattern::Register,
                ]
            };
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "prmt".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "prmt".into(),
                rust_name: recipe.id.into(),
                rust_arguments,
                rust_result: "u32".into(),
                safe: true,
                must_use: true,
                safe_allowlist_reason: Some(
                    "it only permutes register bytes and has no caller preconditions.".into(),
                ),
                public_rust_path: format!("cuda_intrinsics::prmt::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::prmt::{}", recipe.id)],
                dialect_op_type: "PrmtOp".into(),
                dialect_op_name: "nvvm.prmt".into(),
                dialect_operands: llvm_arguments.clone(),
                dialect_results: vec!["i32".into()],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments,
                llvm_results: vec!["i32".into()],
                pure: true,
                memory: "none".into(),
                convergent: false,
                execution_scope: "thread".into(),
                minimum_ptx: "2.0".into(),
                minimum_sm: Some("sm_20".into()),
                ptx_result: "u32".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.9.7 Data Movement and Conversion Instructions: prmt".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt".into(),
                lowering: "generated_prmt".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("3.2".into()),
                        minimum_sm: Some("sm_20".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: None,
                        minimum_sm: Some("sm_75".into()),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                integer_minmax: None,
                packed_conversion: None,
                scalar_conversion: None,
                scalar_arithmetic: None,
                scalar_math: None,
                extended_minmax: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: Some(Prmt {
                    mode: variant.mode,
                    adapter: recipe.adapter,
                }),
                cluster_barrier: None,
                wgmma_control: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                tma: None,
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "prmt".into(),
                    modifiers,
                    operands,
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_prmt_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let prmt = policy
        .prmt
        .as_ref()
        .with_context(|| format!("{} has no closed prmt contract", policy.id))?;
    let recipe = prmt_recipe(prmt.mode);
    let three_operands = recipe.adapter == PrmtAdapter::DirectThreeOperands;
    ensure!(
        prmt.adapter == recipe.adapter
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} prmt identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "prmt"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == vec!["u32"; if three_operands { 3 } else { 2 }]
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::prmt::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::prmt::{}", recipe.id)],
        "{} prmt Rust API does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "PrmtOp"
            && policy.dialect_op_name == "nvvm.prmt"
            && policy.dialect_operands == vec!["i32"; if three_operands { 3 } else { 2 }]
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments == policy.dialect_operands
            && policy.llvm_results == ["i32"]
            && policy.lowering == "generated_prmt",
        "{} prmt carrier or lowering does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration
            .classes
            .iter()
            // LLVM 23 migrated prmt to the target-generic PureIntrinsic
            // class and added IntrNoCreateUndefOrPoison.
            .any(|class| class == "NVVMPureIntrinsic" || class == "PureIntrinsic")
            && declaration.properties
                == ["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"]
            && policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread",
        "{} prmt effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "2.0"
            && policy.minimum_sm.as_deref() == Some("sm_20")
            && policy.targets == "all"
            && policy.ptx_result == "u32"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.9.7 Data Movement and Conversion Instructions: prmt"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt",
        "{} prmt target floor or PTX provenance changed",
        policy.id
    );
    let mut modifiers = vec!["b32"];
    if let Some(modifier) = recipe.modifier {
        modifiers.push(modifier);
    }
    let expected_operands = if three_operands {
        vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::Register,
        ]
    } else {
        vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::Exact { value: "0".into() },
            OperandPattern::Register,
        ]
    };
    ensure!(
        policy.expected_ptx.mnemonic == "prmt"
            && policy.expected_ptx.modifiers == modifiers
            && policy.expected_ptx.operands == expected_operands,
        "{} expected PTX does not match its closed prmt mode",
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
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must define exactly the reviewed prmt backend routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.minimum_ptx.as_deref() == Some("3.2")
                    && lowering.minimum_sm.as_deref() == Some("sm_20")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.minimum_ptx.is_none() && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed prmt floor",
            policy.id,
            lowering.backend
        );
    }
    ensure_no_other_family_contract(policy, "prmt")?;
    Ok(())
}
