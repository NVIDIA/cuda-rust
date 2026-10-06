/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogTargetRequirement, ImportedAddressSpace, ImportedIntrinsic,
    IntrinsicBackend, IntrinsicSource, LdmatrixAdapter, LdmatrixAddressContract, LdmatrixElement,
    LdmatrixLayout, LdmatrixMemoryOrder, LdmatrixMultiplicity, LdmatrixParticipation,
    LdmatrixShape, LdmatrixStateSpace, MovmatrixAdapter, MovmatrixParticipation,
    OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation, StmatrixAdmission, StmatrixLayout,
    StmatrixMultiplicity,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;
use crate::resolve::targets::*;

pub(in crate::resolve) const BLACKWELL_LDMATRIX_LLVM_TARGETS: &str =
    "sm_100a|sm_100f|sm_103a|sm_103f|sm_110a|sm_110f|sm_120a|sm_120f|sm_121a|sm_121f";
pub(in crate::resolve) const BLACKWELL_LDMATRIX_LIBNVVM_TARGETS: &str =
    BLACKWELL_LDMATRIX_LLVM_TARGETS;
#[derive(Clone, Copy)]
pub(in crate::resolve) struct StmatrixRecipe {
    multiplicity: StmatrixMultiplicity,
    layout: StmatrixLayout,
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    compatibility_name: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    summary: &'static str,
}

pub(in crate::resolve) fn stmatrix_recipe(
    multiplicity: StmatrixMultiplicity,
    layout: StmatrixLayout,
) -> StmatrixRecipe {
    match (multiplicity, layout) {
        (StmatrixMultiplicity::X2, StmatrixLayout::Normal) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0301",
            id: "stmatrix_m8n8_x2_b16",
            operation_key: "matrix.stmatrix.m8n8.x2.normal.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x2_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16",
            compatibility_name: "stmatrix_m8n8_x2",
            dialect_op_type: "StmatrixM8n8X2Op",
            dialect_op_name: "nvvm.stmatrix_m8n8_x2",
            summary: "Stores two 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
        (StmatrixMultiplicity::X2, StmatrixLayout::Transposed) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0302",
            id: "stmatrix_m8n8_x2_trans_b16",
            operation_key: "matrix.stmatrix.m8n8.x2.transposed.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.trans.b16",
            compatibility_name: "stmatrix_m8n8_x2_trans",
            dialect_op_type: "StmatrixM8n8X2TransOp",
            dialect_op_name: "nvvm.stmatrix_m8n8_x2_trans",
            summary: "Stores two transposed 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
        (StmatrixMultiplicity::X4, StmatrixLayout::Normal) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0303",
            id: "stmatrix_m8n8_x4_b16",
            operation_key: "matrix.stmatrix.m8n8.x4.normal.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x4_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16",
            compatibility_name: "stmatrix_m8n8_x4",
            dialect_op_type: "StmatrixM8n8X4Op",
            dialect_op_name: "nvvm.stmatrix_m8n8_x4",
            summary: "Stores four 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
        (StmatrixMultiplicity::X4, StmatrixLayout::Transposed) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0304",
            id: "stmatrix_m8n8_x4_trans_b16",
            operation_key: "matrix.stmatrix.m8n8.x4.transposed.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.trans.b16",
            compatibility_name: "stmatrix_m8n8_x4_trans",
            dialect_op_type: "StmatrixM8n8X4TransOp",
            dialect_op_name: "nvvm.stmatrix_m8n8_x4_trans",
            summary: "Stores four transposed 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
    }
}

pub(in crate::resolve) fn stmatrix_variant_for_id(
    id: &str,
) -> Option<(StmatrixMultiplicity, StmatrixLayout)> {
    match id {
        "stmatrix_m8n8_x2_b16" => Some((StmatrixMultiplicity::X2, StmatrixLayout::Normal)),
        "stmatrix_m8n8_x2_trans_b16" => {
            Some((StmatrixMultiplicity::X2, StmatrixLayout::Transposed))
        }
        "stmatrix_m8n8_x4_b16" => Some((StmatrixMultiplicity::X4, StmatrixLayout::Normal)),
        "stmatrix_m8n8_x4_trans_b16" => {
            Some((StmatrixMultiplicity::X4, StmatrixLayout::Transposed))
        }
        _ => None,
    }
}

pub(in crate::resolve) fn expand_stmatrix_admission(
    admission: &StmatrixAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "stmatrix runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact stmatrix admission requires both backend evidence profiles"
    );
    let expected_variants = [
        (StmatrixMultiplicity::X2, StmatrixLayout::Normal),
        (StmatrixMultiplicity::X2, StmatrixLayout::Transposed),
        (StmatrixMultiplicity::X4, StmatrixLayout::Normal),
        (StmatrixMultiplicity::X4, StmatrixLayout::Transposed),
    ];
    let actual_variants = admission
        .variants
        .iter()
        .map(|variant| (variant.multiplicity, variant.layout))
        .collect::<Vec<_>>();
    ensure!(
        actual_variants == expected_variants,
        "compact stmatrix admission must contain the four reviewed variants in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = stmatrix_recipe(variant.multiplicity, variant.layout);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let count = recipe.multiplicity.register_count();
            let mut rust_arguments = vec!["*mut u8".to_owned()];
            rust_arguments.extend(std::iter::repeat_n("u32".to_owned(), count));
            let mut dialect_operands = vec!["ptr".to_owned()];
            dialect_operands.extend(std::iter::repeat_n("i32".to_owned(), count));
            let mut llvm_arguments = vec!["anyptr".to_owned()];
            llvm_arguments.extend(std::iter::repeat_n("i32".to_owned(), count));
            let multiplicity = match recipe.multiplicity {
                StmatrixMultiplicity::X2 => "x2",
                StmatrixMultiplicity::X4 => "x4",
            };
            let mut modifiers = vec![
                "sync".into(),
                "aligned".into(),
                "m8n8".into(),
                multiplicity.into(),
            ];
            if recipe.layout == StmatrixLayout::Transposed {
                modifiers.push("trans".into());
            }
            modifiers.extend(["shared".into(), "b16".into()]);
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "stmatrix".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "matrix".into(),
                rust_name: recipe.id.into(),
                rust_arguments,
                rust_result: "()".into(),
                safe: false,
                must_use: false,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::matrix::{}", recipe.id),
                compatibility_rust_paths: vec![format!(
                    "cuda_device::tcgen05::{}",
                    recipe.compatibility_name
                )],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands,
                dialect_results: vec![],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: Some(format!("{}.p3", recipe.llvm_symbol)),
                llvm_arguments,
                llvm_results: vec![],
                pure: false,
                memory: "write".into(),
                convergent: true,
                execution_scope: "warp".into(),
                minimum_ptx: "7.8".into(),
                minimum_sm: Some("sm_90".into()),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.14.5.16 Warp-level matrix store instruction: stmatrix"
                    .into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix".into(),
                lowering: "generated_stmatrix".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("7.8".into()),
                        minimum_sm: Some("sm_90".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some("7.8".into()),
                        minimum_sm: Some("sm_90".into()),
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
                prmt: None,
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
                selected_address_space: Some(ImportedAddressSpace::Shared),
                expected_ptx: InstructionPattern {
                    mnemonic: "stmatrix".into(),
                    modifiers,
                    operands: vec![
                        OperandPattern::Address,
                        OperandPattern::RegisterList { length: count },
                    ],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_stmatrix_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let (multiplicity, layout) = stmatrix_variant_for_id(&policy.id)
        .with_context(|| format!("{} has no closed stmatrix recipe", policy.id))?;
    let recipe = stmatrix_recipe(multiplicity, layout);
    let count = multiplicity.register_count();
    let mut rust_arguments = vec!["*mut u8".to_owned()];
    rust_arguments.extend(std::iter::repeat_n("u32".to_owned(), count));
    let mut dialect_operands = vec!["ptr".to_owned()];
    dialect_operands.extend(std::iter::repeat_n("i32".to_owned(), count));
    let mut llvm_arguments = vec!["anyptr".to_owned()];
    llvm_arguments.extend(std::iter::repeat_n("i32".to_owned(), count));

    ensure!(
        policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.as_deref()
                == Some(format!("{}.p3", recipe.llvm_symbol).as_str()),
        "{} stmatrix identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::matrix::{}", recipe.id)
            && policy.compatibility_rust_paths
                == [format!(
                    "cuda_device::tcgen05::{}",
                    recipe.compatibility_name
                )],
        "{} stmatrix Rust API does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_stmatrix"
            && policy.selected_address_space == Some(ImportedAddressSpace::Shared),
        "{} stmatrix carriers or lowering do not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrArgMemOnly",
                "IntrConvergent",
                "IntrNoCallback",
                "IntrWriteMem",
                "NoCapture<arg0>",
                "WriteOnly<arg0>",
            ]
            && !policy.pure
            && policy.memory == "write"
            && policy.convergent
            && policy.execution_scope == "warp",
        "{} stmatrix effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.ptx_result == "()"
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.5.16 Warp-level matrix store instruction: stmatrix"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix",
        "{} stmatrix target floor or PTX provenance changed",
        policy.id
    );
    let multiplicity_name = match multiplicity {
        StmatrixMultiplicity::X2 => "x2",
        StmatrixMultiplicity::X4 => "x4",
    };
    let mut modifiers = vec![
        "sync".into(),
        "aligned".into(),
        "m8n8".into(),
        multiplicity_name.into(),
    ];
    if layout == StmatrixLayout::Transposed {
        modifiers.push("trans".into());
    }
    modifiers.extend(["shared".into(), "b16".into()]);
    ensure!(
        policy.expected_ptx
            == (InstructionPattern {
                mnemonic: "stmatrix".into(),
                modifiers,
                operands: vec![
                    OperandPattern::Address,
                    OperandPattern::RegisterList { length: count },
                ],
            }),
        "{} expected PTX does not match its closed stmatrix shape",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings.iter().all(|route| {
                !route.evidence_profile.trim().is_empty()
                    && route.minimum_ptx.as_deref() == Some("7.8")
                    && route.minimum_sm.as_deref() == Some("sm_90")
            })
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LlvmNvptx
                    && route.mechanism == BackendLoweringMechanism::TypedNvvm
            })
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LibNvvm
                    && route.mechanism == BackendLoweringMechanism::InlinePtx
            }),
        "{} must keep its reviewed typed-NVVM and inline-PTX routes",
        policy.id
    );
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.cp_async_mbarrier.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.register_mma.is_none()
            && policy.sparse_mma.is_none()
            && policy.prmt.is_none()
            && policy.cluster_barrier.is_none()
            && policy.special_register.is_none()
            && policy.debug_control.is_none(),
        "{} mixes another generated-family contract with stmatrix",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) fn validate_ldmatrix_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let variant = policy
        .ldmatrix_variant
        .as_ref()
        .with_context(|| format!("{} has no closed ldmatrix variant", policy.id))?;
    let safety = policy
        .ldmatrix_safety
        .as_ref()
        .with_context(|| format!("{} has no ldmatrix safety contract", policy.id))?;
    let classic = variant.shape == LdmatrixShape::M8n8 && variant.element == LdmatrixElement::B16;
    let blackwell = matches!(
        (variant.shape, variant.layout, variant.element),
        (
            LdmatrixShape::M8n16,
            LdmatrixLayout::Normal,
            LdmatrixElement::B8x16B4x16P64 | LdmatrixElement::B8x16B6x16P32
        ) | (
            LdmatrixShape::M16n16,
            LdmatrixLayout::Transposed,
            LdmatrixElement::B8 | LdmatrixElement::B8x16B4x16P64 | LdmatrixElement::B8x16B6x16P32
        )
    );
    ensure!(
        variant.shape != LdmatrixShape::M16n16 || variant.multiplicity != LdmatrixMultiplicity::X4,
        "{} requests unsupported m16n16.x4 ldmatrix",
        policy.id
    );
    ensure!(
        (classic || blackwell) && variant.state_space == LdmatrixStateSpace::Shared,
        "{} requests an unsupported ldmatrix shape, element, or state space",
        policy.id
    );
    let expected_participation = if classic {
        LdmatrixParticipation::AllWarpLanesSameInstruction
    } else {
        LdmatrixParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes
    };
    let expected_address_contract = match variant.shape {
        LdmatrixShape::M8n8 => {
            LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadableWithSm75Replication
        }
        LdmatrixShape::M8n16 => {
            LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadable
        }
        LdmatrixShape::M16n16 => {
            LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedThirtyTwoBytesReadable
        }
    };
    ensure!(
        safety.participation == expected_participation
            && safety.address_contract == expected_address_contract
            && safety.memory_order == LdmatrixMemoryOrder::Weak
            && safety.runtime_validation == RuntimeValidation::Unexecuted,
        "{} has an unsupported ldmatrix safety contract",
        policy.id
    );
    let count = variant.register_count();
    let count_name = match variant.multiplicity {
        LdmatrixMultiplicity::X1 => "x1",
        LdmatrixMultiplicity::X2 => "x2",
        LdmatrixMultiplicity::X4 => "x4",
    };
    let trans_record = match variant.layout {
        LdmatrixLayout::Normal => "",
        LdmatrixLayout::Transposed => "_trans",
    };
    let trans_symbol = match variant.layout {
        LdmatrixLayout::Normal => "",
        LdmatrixLayout::Transposed => ".trans",
    };
    let layout_name = match variant.layout {
        LdmatrixLayout::Normal => "normal",
        LdmatrixLayout::Transposed => "transposed",
    };
    let shape_name = match variant.shape {
        LdmatrixShape::M8n8 => "m8n8",
        LdmatrixShape::M8n16 => "m8n16",
        LdmatrixShape::M16n16 => "m16n16",
    };
    let (element_record, element_symbol) = match variant.element {
        LdmatrixElement::B16 => ("b16", "b16"),
        LdmatrixElement::B8 => ("b8", "b8"),
        LdmatrixElement::B8x16B4x16P64 => ("b8x16_b4x16_p64", "b8x16.b4x16_p64"),
        LdmatrixElement::B8x16B6x16P32 => ("b8x16_b6x16_p32", "b8x16.b6x16_p32"),
    };
    let expected_source = format!(
        "int_nvvm_ldmatrix_sync_aligned_{shape_name}_{count_name}{trans_record}_{element_record}"
    );
    let expected_symbol = format!(
        "llvm.nvvm.ldmatrix.sync.aligned.{shape_name}.{count_name}{trans_symbol}.{element_symbol}"
    );
    let expected_name =
        format!("ldmatrix_{shape_name}_{count_name}{trans_record}_{element_record}");
    let expected_result = if count == 1 {
        "u32".to_owned()
    } else {
        format!("[u32; {count}]")
    };
    let expected_adapter = if count == 1 {
        LdmatrixAdapter::SingleResultDirect
    } else {
        LdmatrixAdapter::MultipleResultsToArray
    };
    ensure!(
        policy.source_record.as_deref() == Some(expected_source.as_str())
            && policy.llvm_symbol.as_deref() == Some(expected_symbol.as_str()),
        "{} ldmatrix variant does not match its imported source record or base LLVM symbol",
        policy.id
    );
    ensure!(
        policy.resolved_llvm_symbol.as_deref() == Some(format!("{expected_symbol}.p3").as_str()),
        "{} must keep the imported base symbol distinct from the resolved `.p3` overload",
        policy.id
    );
    ensure!(
        policy.rust_arguments == [if classic { "*const u32" } else { "*const u8" }]
            && policy.rust_result == expected_result
            && policy.llvm_arguments == ["anyptr"]
            && policy.llvm_results == vec!["i32"; count]
            && policy.ptx_result == policy.rust_result,
        "{} ldmatrix Rust, imported LLVM, and PTX carrier signatures disagree",
        policy.id
    );
    ensure!(
        policy.id == expected_name
            && policy.operation_key
                == format!(
                    "matrix.ldmatrix.{shape_name}.{count_name}.{layout_name}.{element_record}.shared"
                )
            && policy.rust_module == "matrix"
            && policy.rust_name == expected_name
            && !policy.safe
            && policy.must_use != classic
            && policy.safe_allowlist_reason.is_none()
            && {
                let expected_compatibility_paths = if classic {
                    vec![
                        format!("cuda_device::wmma::ldmatrix_{count_name}{trans_record}"),
                        format!(
                            "cuda_device::wmma::ldmatrix_{count_name}{trans_record}_shared_u32"
                        ),
                    ]
                } else if matches!(
                    policy.id.as_str(),
                    "ldmatrix_m8n16_x2_b8x16_b4x16_p64" | "ldmatrix_m8n16_x4_b8x16_b4x16_p64"
                ) {
                    vec![format!("cuda_device::wmma::{expected_name}")]
                } else {
                    vec![]
                };
                policy.compatibility_rust_paths == expected_compatibility_paths
            }
            && policy.lowering == "generated_ldmatrix"
            && policy.ldmatrix_adapter == Some(expected_adapter)
            && policy.selected_address_space == Some(ImportedAddressSpace::Shared),
        "{} must preserve the closed raw/compatibility ldmatrix API, result adapter, and shared selection",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read"
            && policy.convergent
            && policy.execution_scope == "warp"
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrArgMemOnly")
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrReadMem"),
        "{} ldmatrix effects disagree with the imported declaration",
        policy.id
    );
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        backend_pairs
            == BTreeSet::from([
                (
                    IntrinsicBackend::LlvmNvptx,
                    BackendLoweringMechanism::TypedNvvm
                ),
                (
                    IntrinsicBackend::LibNvvm,
                    BackendLoweringMechanism::InlinePtx
                ),
            ]),
        "{} must define exactly the reviewed LLVM typed and libNVVM inline-PTX lowerings",
        policy.id
    );
    ensure!(
        policy
            .backend_lowerings
            .iter()
            .all(|lowering| !lowering.evidence_profile.trim().is_empty()),
        "{} backend lowering omits its evidence profile",
        policy.id
    );
    if blackwell {
        ensure!(
            policy.minimum_ptx == "8.6"
                && policy.minimum_sm.is_none()
                && policy.targets == BLACKWELL_LDMATRIX_LLVM_TARGETS,
            "{} must retain the reviewed PTX 8.6 floor and exact LLVM Blackwell target set",
            policy.id
        );
        let llvm = policy
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
            .expect("validated LLVM route");
        let libnvvm = policy
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .expect("validated libNVVM route");
        ensure!(
            llvm.targets.is_none()
                && llvm.minimum_sm.is_none()
                && backend_target_requirement(policy, llvm)?
                    == CatalogTargetRequirement {
                        minimum_ptx: parse_ptx_version("8.6", &policy.id)?,
                        hardware: parse_hardware_target_fields(
                            &policy.id,
                            BLACKWELL_LDMATRIX_LLVM_TARGETS,
                            None,
                        )?,
                    }
                && libnvvm.targets.as_deref() == Some(BLACKWELL_LDMATRIX_LIBNVVM_TARGETS)
                && libnvvm.minimum_sm.is_none()
                && backend_target_requirement(policy, libnvvm)?
                    == CatalogTargetRequirement {
                        minimum_ptx: parse_ptx_version("8.6", &policy.id)?,
                        hardware: parse_hardware_target_fields(
                            &policy.id,
                            BLACKWELL_LDMATRIX_LIBNVVM_TARGETS,
                            None,
                        )?,
                    },
            "{} backend routes do not preserve the reviewed LLVM/libNVVM Blackwell target split",
            policy.id
        );
    }
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none(),
        "{} mixes another generated-family contract with ldmatrix",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) fn validate_movmatrix_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
) -> Result<()> {
    let contract = policy
        .movmatrix
        .as_ref()
        .context("movmatrix requires its closed contract")?;
    ensure!(
        policy.id == "movmatrix_trans_b16"
            && policy.operation_key == "movmatrix.m8n8.trans.b16"
            && matches!(
                source,
                IntrinsicSource::PtxNative { instruction }
                    if instruction == "movmatrix.sync.aligned.m8n8.trans.b16"
            )
            && policy.rust_module == "matrix"
            && policy.rust_name == "movmatrix_trans_b16"
            && policy.rust_arguments == ["u32"]
            && policy.rust_result == "u32"
            && !policy.safe
            && policy.must_use
            && policy.public_rust_path == "cuda_intrinsics::matrix::movmatrix_trans_b16"
            && policy.compatibility_rust_paths == ["cuda_device::wmma::movmatrix_trans_b16"]
            && policy.dialect_op_type == "MovmatrixTransB16Op"
            && policy.dialect_op_name == "nvvm.movmatrix_trans_b16"
            && policy.dialect_operands == ["i32"]
            && policy.dialect_results == ["i32"]
            && !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_75")
            && policy.ptx_result == "u32"
            && policy.targets == "all"
            && policy.lowering == "generated_movmatrix_inline_ptx"
            && policy.expected_ptx
                == InstructionPattern {
                    mnemonic: "movmatrix".into(),
                    modifiers: ["sync", "aligned", "m8n8", "trans", "b16"]
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                    operands: vec![OperandPattern::Register, OperandPattern::Register],
                },
        "{} is outside the closed movmatrix recipe",
        policy.id
    );
    ensure!(
        contract.participation == MovmatrixParticipation::AllWarpLanesSameInstructionNoExitedLanes
            && contract.adapter == MovmatrixAdapter::PackedB16x2U32ToPackedB16x2U32,
        "{} has an unreviewed movmatrix safety or adapter contract",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, "7.8", Some("sm_75")),
            (IntrinsicBackend::LibNvvm, "7.8", Some("sm_75")),
        ],
        "movmatrix",
    )?;
    ensure_no_other_family_contract(policy, "movmatrix")?;
    Ok(())
}
