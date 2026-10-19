/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, IntrinsicBackend, OverlayBackendLowering, OverlayIntrinsic,
    RegisterMma, RegisterMmaAccumulator, RegisterMmaAdapter, RegisterMmaAmpereFloatAdmission,
    RegisterMmaAmpereFloatVariant, RegisterMmaBinaryAdmission, RegisterMmaCompatibilitySource,
    RegisterMmaElement, RegisterMmaF8F6F4Admission, RegisterMmaFp8Admission,
    RegisterMmaIntegerAdmission, RegisterMmaKind, RegisterMmaLayout, RegisterMmaOperation,
    RegisterMmaOverflow, RegisterMmaParticipation, RegisterMmaShape, RuntimeValidation,
    SparseMmaAccumulator, SparseMmaLayout, SparseMmaMetadata, SparseMmaOverflow, SparseMmaSelector,
    SparseMmaShape,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeSet;

use super::*;

pub(in crate::resolve) const REGISTER_MMA_F8F6F4_TARGETS: &str = "sm_120a|sm_120f|sm_121a|sm_121f";
#[derive(Debug, Clone, Copy)]
pub(in crate::resolve) enum RegisterMmaIntegerKind {
    Int4,
    Int8,
}

impl RegisterMmaIntegerKind {
    fn label(self) -> &'static str {
        match self {
            Self::Int4 => "INT4",
            Self::Int8 => "INT8",
        }
    }

    fn supports(self, shape: RegisterMmaShape, element: RegisterMmaElement) -> bool {
        match self {
            Self::Int4 => {
                matches!(
                    shape,
                    RegisterMmaShape::M8n8k32
                        | RegisterMmaShape::M16n8k32
                        | RegisterMmaShape::M16n8k64
                ) && matches!(element, RegisterMmaElement::S4 | RegisterMmaElement::U4)
            }
            Self::Int8 => {
                matches!(
                    shape,
                    RegisterMmaShape::M8n8k16
                        | RegisterMmaShape::M16n8k16
                        | RegisterMmaShape::M16n8k32
                ) && matches!(element, RegisterMmaElement::S8 | RegisterMmaElement::U8)
            }
        }
    }
}

pub(in crate::resolve) fn expand_register_mma_integer_admission(
    kind: RegisterMmaIntegerKind,
    admission: &RegisterMmaIntegerAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaCompatibilitySource::GeneratedStub;
    use RegisterMmaLayout::{Col, Row};
    use RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes;
    use RegisterMmaShape::{M8n8k16, M8n8k32, M16n8k16, M16n8k32, M16n8k64};

    ensure!(
        !admission.variants.is_empty(),
        "compact {} MMA admission has no variants",
        kind.label()
    );
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "{} MMA runtime validation may be marked executed only with GPU evidence",
        kind.label()
    );

    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(admission.variants.len());
    for variant in &admission.variants {
        ensure!(
            seen.insert((
                variant.shape,
                variant.a_element,
                variant.b_element,
                variant.overflow,
            )),
            "compact {} MMA admission contains a duplicate variant",
            kind.label()
        );
        ensure!(
            kind.supports(variant.shape, variant.a_element)
                && kind.supports(variant.shape, variant.b_element),
            "compact {} MMA admission contains an unsupported shape or element",
            kind.label()
        );
        let adapter =
            match (kind, variant.shape) {
                (RegisterMmaIntegerKind::Int8, M8n8k16)
                | (RegisterMmaIntegerKind::Int4, M8n8k32) => C2I32A1U32B1U32ToD2I32,
                (RegisterMmaIntegerKind::Int8, M16n8k16)
                | (RegisterMmaIntegerKind::Int4, M16n8k32) => C4I32A2U32B1U32ToD4I32,
                (RegisterMmaIntegerKind::Int8, M16n8k32)
                | (RegisterMmaIntegerKind::Int4, M16n8k64) => C4I32A4U32B2U32ToD4I32,
                _ => bail!(
                    "compact {} MMA admission contains an unsupported shape",
                    kind.label()
                ),
            };
        let mma = RegisterMma {
            shape: variant.shape,
            operation: RegisterMmaOperation::Multiply,
            kind: None,
            accumulator: RegisterMmaAccumulator::S32,
            a_element: variant.a_element,
            b_element: variant.b_element,
            a_layout: Row,
            b_layout: Col,
            overflow: variant.overflow,
            participation: AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
            adapter,
            compatibility_source: GeneratedStub,
            runtime_validation: admission.runtime_validation,
        };
        let recipe = register_mma_recipe(&mma).with_context(|| {
            format!(
                "compact {} MMA admission requests a variant outside the closed recipe set",
                kind.label()
            )
        })?;
        ensure!(
            recipe.compatibility_source == GeneratedStub,
            "compact {} MMA admission may only add generated compatibility stubs",
            kind.label()
        );

        let element = |element| match (kind, element) {
            (RegisterMmaIntegerKind::Int4, RegisterMmaElement::S4)
            | (RegisterMmaIntegerKind::Int8, RegisterMmaElement::S8) => Ok("signed"),
            (RegisterMmaIntegerKind::Int4, RegisterMmaElement::U4)
            | (RegisterMmaIntegerKind::Int8, RegisterMmaElement::U8) => Ok("unsigned"),
            _ => bail!(
                "compact {} MMA admission contains an unsupported element",
                kind.label()
            ),
        };
        let overflow = match variant.overflow {
            RegisterMmaOverflow::Wrapping => "wrapping",
            RegisterMmaOverflow::Satfinite => "saturating",
            RegisterMmaOverflow::NotApplicable => {
                bail!(
                    "compact {} MMA admission requires an integer overflow mode",
                    kind.label()
                )
            }
        };
        let summary = format!(
            "Multiplies warp-distributed {} A and {} B {} fragments and adds a {overflow} s32 accumulator.",
            element(variant.a_element)?,
            element(variant.b_element)?,
            kind.label(),
        );

        records.push(OverlayIntrinsic {
            id: recipe.id.into(),
            abi_id: recipe.abi_id.into(),
            operation_key: recipe.operation_key.into(),
            family: "register_mma".into(),
            source: None,
            source_record: Some(recipe.source_record.into()),
            rust_module: "matrix".into(),
            rust_name: recipe.id.into(),
            rust_arguments: recipe
                .rust_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            rust_result: recipe.rust_result.into(),
            safe: false,
            must_use: true,
            safe_allowlist_reason: None,
            public_rust_path: format!("cuda_intrinsics::matrix::{}", recipe.id),
            compatibility_rust_paths: vec![format!("cuda_device::wmma::{}", recipe.id)],
            dialect_op_type: recipe.dialect_op_type.into(),
            dialect_op_name: recipe.dialect_op_name.into(),
            dialect_operands: recipe
                .dialect_operands
                .iter()
                .map(|value| (*value).into())
                .collect(),
            dialect_results: recipe
                .dialect_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_symbol: Some(recipe.llvm_symbol.into()),
            resolved_llvm_symbol: None,
            llvm_arguments: recipe
                .llvm_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_results: recipe
                .llvm_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            pure: false,
            memory: "none".into(),
            convergent: true,
            execution_scope: "warp".into(),
            minimum_ptx: recipe.minimum_ptx.into(),
            minimum_sm: Some(recipe.minimum_sm.into()),
            ptx_result: recipe.rust_result.into(),
            targets: "all".into(),
            ptx_isa_version: "9.3".into(),
            ptx_isa_section: "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
            lowering: "generated_register_mma".into(),
            backend_lowerings: vec![
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LlvmNvptx,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.llvm_evidence_profile.clone(),
                    targets: None,
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
                },
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LibNvvm,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.libnvvm_evidence_profile.clone(),
                    targets: None,
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
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
            register_mma: Some(mma),
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
            selected_address_space: None,
            expected_ptx: InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: recipe
                    .ptx_modifiers
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                operands: recipe
                    .ptx_register_counts
                    .map(|length| OperandPattern::RegisterList { length })
                    .into(),
            },
            summary,
        });
    }
    Ok(records)
}

pub(in crate::resolve) const REGISTER_MMA_F8F6F4_ELEMENTS: [RegisterMmaElement; 5] = [
    RegisterMmaElement::E2m1,
    RegisterMmaElement::E2m3,
    RegisterMmaElement::E3m2,
    RegisterMmaElement::E4m3,
    RegisterMmaElement::E5m2,
];

pub(in crate::resolve) fn register_mma_f8f6f4_element_name(
    element: RegisterMmaElement,
) -> Option<&'static str> {
    match element {
        RegisterMmaElement::E2m1 => Some("e2m1"),
        RegisterMmaElement::E2m3 => Some("e2m3"),
        RegisterMmaElement::E3m2 => Some("e3m2"),
        RegisterMmaElement::E4m3 => Some("e4m3"),
        RegisterMmaElement::E5m2 => Some("e5m2"),
        _ => None,
    }
}

pub(in crate::resolve) fn is_dense_f8f6f4_register_mma_policy(policy: &OverlayIntrinsic) -> bool {
    policy.family == "register_mma"
        && policy.targets == REGISTER_MMA_F8F6F4_TARGETS
        && policy.register_mma.as_ref().is_some_and(|mma| {
            mma.kind != Some(RegisterMmaKind::Standard)
                && mma.shape == RegisterMmaShape::M16n8k32
                && mma.operation == RegisterMmaOperation::Multiply
                && matches!(
                    mma.accumulator,
                    RegisterMmaAccumulator::F16 | RegisterMmaAccumulator::F32
                )
                && register_mma_f8f6f4_element_name(mma.a_element).is_some()
                && register_mma_f8f6f4_element_name(mma.b_element).is_some()
                && mma.a_layout == RegisterMmaLayout::Row
                && mma.b_layout == RegisterMmaLayout::Col
                && mma.overflow == RegisterMmaOverflow::NotApplicable
        })
}

pub(in crate::resolve) fn is_sparse_f8f6f4_f16_policy(policy: &OverlayIntrinsic) -> bool {
    policy.family == "sparse_mma"
        && policy.targets == SPARSE_MMA_F8F6F4_TARGETS
        && policy.sparse_mma.as_ref().is_some_and(|mma| {
            mma.shape == SparseMmaShape::M16n8k64
                && mma.accumulator == SparseMmaAccumulator::F16
                && SPARSE_MMA_F8F6F4_ELEMENTS.contains(&mma.a_element)
                && SPARSE_MMA_F8F6F4_ELEMENTS.contains(&mma.b_element)
                && mma.a_layout == SparseMmaLayout::Row
                && mma.b_layout == SparseMmaLayout::Col
                && mma.overflow == SparseMmaOverflow::NotApplicable
                && mma.metadata == SparseMmaMetadata::Ordered
                && mma.selector == SparseMmaSelector::ImmediateZero
        })
}

pub(in crate::resolve) fn is_f8f6f4_mma_target_matrix_policy(policy: &OverlayIntrinsic) -> bool {
    is_dense_f8f6f4_register_mma_policy(policy) || is_sparse_f8f6f4_f16_policy(policy)
}

#[derive(Clone, Copy)]
pub(in crate::resolve) struct RegisterMmaF8F6F4Contract {
    pub(in crate::resolve) accumulator: RegisterMmaAccumulator,
    pub(in crate::resolve) scalar_name: &'static str,
    pub(in crate::resolve) rust_arguments: [&'static str; 3],
    pub(in crate::resolve) rust_result: &'static str,
    pub(in crate::resolve) dialect_operands: &'static [&'static str],
    pub(in crate::resolve) dialect_results: &'static [&'static str],
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_results: &'static [&'static str],
    pub(in crate::resolve) adapter: RegisterMmaAdapter,
    pub(in crate::resolve) ptx_register_counts: [usize; 4],
    pub(in crate::resolve) summary_accumulator: &'static str,
}

pub(in crate::resolve) fn register_mma_f8f6f4_contract(
    accumulator: RegisterMmaAccumulator,
) -> Result<RegisterMmaF8F6F4Contract> {
    match accumulator {
        RegisterMmaAccumulator::F16 => Ok(RegisterMmaF8F6F4Contract {
            accumulator,
            scalar_name: "f16",
            rust_arguments: ["[u32; 2]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[u32; 2]",
            dialect_operands: &["i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32"],
            dialect_results: &["i32", "i32"],
            llvm_arguments: &["i32", "i32", "i32", "i32", "i32", "i32", "v2f16", "v2f16"],
            llvm_results: &["v2f16", "v2f16"],
            adapter: RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32,
            ptx_register_counts: [2, 4, 2, 2],
            summary_accumulator: "F16",
        }),
        RegisterMmaAccumulator::F32 => Ok(RegisterMmaF8F6F4Contract {
            accumulator,
            scalar_name: "f32",
            rust_arguments: ["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[f32; 4]",
            dialect_operands: &[
                "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32",
            ],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &[
                "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32",
            ],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32,
            ptx_register_counts: [4, 4, 2, 4],
            summary_accumulator: "F32",
        }),
        _ => bail!("unsupported dense f8f6f4 MMA accumulator {accumulator:?}"),
    }
}

pub(in crate::resolve) fn expand_register_mma_f8f6f4_admission(
    admission: &RegisterMmaF8F6F4Admission,
    accumulator: RegisterMmaAccumulator,
) -> Result<Vec<OverlayIntrinsic>> {
    let contract = register_mma_f8f6f4_contract(accumulator)?;
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "dense f8f6f4 MMA runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "dense f8f6f4 MMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission.a_elements == REGISTER_MMA_F8F6F4_ELEMENTS
            && admission.b_elements == REGISTER_MMA_F8F6F4_ELEMENTS
            && admission.product_count == 25,
        "dense f8f6f4 MMA admission must contain the canonical 5 by 5 element matrix"
    );
    let expected_targets = ["sm_120a", "sm_120f", "sm_121a", "sm_121f"];
    ensure!(
        admission
            .targets
            .iter()
            .map(String::as_str)
            .eq(expected_targets),
        "dense f8f6f4 MMA admission must retain the reviewed Blackwell target set"
    );

    let mut records = Vec::with_capacity(admission.product_count);
    for &a_element in &admission.a_elements {
        for &b_element in &admission.b_elements {
            let a = register_mma_f8f6f4_element_name(a_element)
                .expect("validated dense f8f6f4 A element");
            let b = register_mma_f8f6f4_element_name(b_element)
                .expect("validated dense f8f6f4 B element");
            let scalar = contract.scalar_name;
            let id = format!("mma_m16n8k32_{scalar}_{a}_{b}");
            let source_record =
                format!("int_nvvm_mma_m16n8k32_row_col_kind_f8f6f4_{scalar}_{a}_{b}_{scalar}");
            let llvm_symbol =
                format!("llvm.nvvm.mma.m16n8k32.row.col.kind.f8f6f4.{scalar}.{a}.{b}.{scalar}");
            let mma = RegisterMma {
                shape: RegisterMmaShape::M16n8k32,
                operation: RegisterMmaOperation::Multiply,
                kind: None,
                accumulator: contract.accumulator,
                a_element,
                b_element,
                a_layout: RegisterMmaLayout::Row,
                b_layout: RegisterMmaLayout::Col,
                overflow: RegisterMmaOverflow::NotApplicable,
                participation:
                    RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter: contract.adapter,
                compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };

            records.push(OverlayIntrinsic {
                id: id.clone(),
                abi_id: String::new(),
                operation_key: format!(
                    "matrix.mma.m16n8k32.row.col.kind_f8f6f4.{scalar}.{a}.{b}.{scalar}"
                ),
                family: "register_mma".into(),
                source: None,
                source_record: Some(source_record),
                rust_module: "matrix".into(),
                rust_name: id.clone(),
                rust_arguments: contract.rust_arguments.map(Into::into).into(),
                rust_result: contract.rust_result.into(),
                safe: false,
                must_use: true,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::matrix::{id}"),
                compatibility_rust_paths: vec![format!("cuda_device::wmma::{id}")],
                dialect_op_type: "RegisterMmaOp".into(),
                dialect_op_name: "nvvm.register_mma".into(),
                dialect_operands: contract
                    .dialect_operands
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                dialect_results: contract
                    .dialect_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_symbol: Some(llvm_symbol),
                resolved_llvm_symbol: None,
                llvm_arguments: contract
                    .llvm_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_results: contract
                    .llvm_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                pure: false,
                memory: "none".into(),
                convergent: true,
                execution_scope: "warp".into(),
                minimum_ptx: "8.7".into(),
                minimum_sm: None,
                ptx_result: contract.rust_result.into(),
                targets: REGISTER_MMA_F8F6F4_TARGETS.into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
                lowering: "generated_register_mma".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: None,
                        minimum_sm: None,
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: None,
                        minimum_sm: None,
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
                register_mma: Some(mma),
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
                selected_address_space: None,
                expected_ptx: InstructionPattern {
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
                summary: format!(
                    "Multiplies warp-distributed {a} A and {b} B fragments and adds an {} accumulator.",
                    contract.summary_accumulator
                ),
            });
        }
    }
    Ok(records)
}

pub(in crate::resolve) fn expand_register_mma_mxf8f6f4_admission(
    admission: &RegisterMmaF8F6F4Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "dense mxf8f6f4 MMA runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "dense mxf8f6f4 MMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission.a_elements == REGISTER_MMA_F8F6F4_ELEMENTS
            && admission.b_elements == REGISTER_MMA_F8F6F4_ELEMENTS
            && admission.product_count == 25,
        "dense mxf8f6f4 MMA admission must contain the canonical 5 by 5 element matrix"
    );
    let expected_targets = ["sm_120a", "sm_120f", "sm_121a", "sm_121f"];
    ensure!(
        admission
            .targets
            .iter()
            .map(String::as_str)
            .eq(expected_targets),
        "dense mxf8f6f4 MMA admission must retain the reviewed Blackwell target set"
    );

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
    let dialect_results = ["f32", "f32", "f32", "f32"];

    let mut records = Vec::with_capacity(admission.product_count);
    for &a_element in &admission.a_elements {
        for &b_element in &admission.b_elements {
            let a = register_mma_f8f6f4_element_name(a_element)
                .expect("validated dense mxf8f6f4 A element");
            let b = register_mma_f8f6f4_element_name(b_element)
                .expect("validated dense mxf8f6f4 B element");
            let id = format!("mma_m16n8k32_mxf8f6f4_f32_{a}_{b}");
            let source_record =
                format!("int_nvvm_mma_block_scale_m16n8k32_row_col_mxf8f6f4_f32_{a}_{b}_f32_ue8m0");
            let llvm_symbol = format!(
                "llvm.nvvm.mma.block.scale.m16n8k32.row.col.mxf8f6f4.f32.{a}.{b}.f32.ue8m0"
            );
            let mma = RegisterMma {
                shape: RegisterMmaShape::M16n8k32,
                operation: RegisterMmaOperation::Multiply,
                kind: Some(RegisterMmaKind::Mxf8f6f4),
                accumulator: RegisterMmaAccumulator::F32,
                a_element,
                b_element,
                a_layout: RegisterMmaLayout::Row,
                b_layout: RegisterMmaLayout::Col,
                overflow: RegisterMmaOverflow::NotApplicable,
                participation:
                    RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter: RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32,
                compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };

            records.push(OverlayIntrinsic {
                id: id.clone(),
                abi_id: String::new(),
                operation_key: format!(
                    "matrix.mma.m16n8k32.row.col.kind_mxf8f6f4.scale_vec_1x.f32.{a}.{b}.f32.ue8m0"
                ),
                family: "register_mma".into(),
                source: None,
                source_record: Some(source_record),
                rust_module: "matrix".into(),
                rust_name: id.clone(),
                rust_arguments: rust_arguments.map(Into::into).into(),
                rust_result: "[f32; 4]".into(),
                safe: false,
                must_use: true,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::matrix::{id}"),
                compatibility_rust_paths: vec![format!("cuda_device::wmma::{id}")],
                dialect_op_type: "RegisterMmaOp".into(),
                dialect_op_name: "nvvm.register_mma".into(),
                dialect_operands: dialect_operands.map(Into::into).into(),
                dialect_results: dialect_results.map(Into::into).into(),
                llvm_symbol: Some(llvm_symbol),
                resolved_llvm_symbol: None,
                llvm_arguments: llvm_arguments.map(Into::into).into(),
                llvm_results: dialect_results.map(Into::into).into(),
                pure: false,
                memory: "none".into(),
                convergent: true,
                execution_scope: "warp".into(),
                minimum_ptx: "8.7".into(),
                minimum_sm: None,
                ptx_result: "[f32; 4]".into(),
                targets: REGISTER_MMA_F8F6F4_TARGETS.into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section:
                    "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
                lowering: "generated_register_mma".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: None,
                        minimum_sm: None,
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: None,
                        minimum_sm: None,
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
                register_mma: Some(mma),
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
                selected_address_space: None,
                expected_ptx: InstructionPattern {
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
                summary: format!(
                    "Multiplies block-scaled warp-distributed {a} A and {b} B fragments and adds an F32 accumulator."
                ),
            });
        }
    }
    Ok(records)
}

pub(in crate::resolve) const REGISTER_MMA_FP8_SHAPES: [RegisterMmaShape; 2] =
    [RegisterMmaShape::M16n8k16, RegisterMmaShape::M16n8k32];
pub(in crate::resolve) const REGISTER_MMA_FP8_ACCUMULATORS: [RegisterMmaAccumulator; 2] =
    [RegisterMmaAccumulator::F16, RegisterMmaAccumulator::F32];
pub(in crate::resolve) const REGISTER_MMA_FP8_ELEMENTS: [RegisterMmaElement; 2] =
    [RegisterMmaElement::E4m3, RegisterMmaElement::E5m2];

pub(in crate::resolve) fn register_mma_fp8_element_name(
    element: RegisterMmaElement,
) -> Option<&'static str> {
    match element {
        RegisterMmaElement::E4m3 => Some("e4m3"),
        RegisterMmaElement::E5m2 => Some("e5m2"),
        _ => None,
    }
}

pub(in crate::resolve) fn register_mma_fp8_shape_contract(
    shape: RegisterMmaShape,
) -> Result<(&'static str, usize, usize)> {
    match shape {
        RegisterMmaShape::M16n8k16 => Ok(("m16n8k16", 2, 1)),
        RegisterMmaShape::M16n8k32 => Ok(("m16n8k32", 4, 2)),
        _ => bail!("unsupported standard FP8 register-MMA shape {shape:?}"),
    }
}

pub(in crate::resolve) fn register_mma_fp8_minimum_ptx(
    shape: RegisterMmaShape,
    accumulator: RegisterMmaAccumulator,
) -> &'static str {
    // ptxas requires PTX 8.7 for the k32 F16 form despite LLVM's 8.4 predicate.
    match (shape, accumulator) {
        (RegisterMmaShape::M16n8k32, RegisterMmaAccumulator::F32) => "8.4",
        _ => "8.7",
    }
}

pub(in crate::resolve) fn expand_register_mma_fp8_admission(
    admission: &RegisterMmaFp8Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "standard FP8 MMA runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "standard FP8 MMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission.shapes == REGISTER_MMA_FP8_SHAPES
            && admission.accumulators == REGISTER_MMA_FP8_ACCUMULATORS
            && admission.a_elements == REGISTER_MMA_FP8_ELEMENTS
            && admission.b_elements == REGISTER_MMA_FP8_ELEMENTS
            && admission.product_count == 16,
        "standard FP8 MMA admission must retain the canonical 2 by 2 by 2 by 2 matrix"
    );

    let mut records = Vec::with_capacity(admission.product_count);
    for &shape in &admission.shapes {
        let (shape_name, a_count, b_count) = register_mma_fp8_shape_contract(shape)?;
        for &accumulator in &admission.accumulators {
            let minimum_ptx = register_mma_fp8_minimum_ptx(shape, accumulator);
            let (scalar, rust_arguments, rust_result, dialect_operands, dialect_results, adapter) =
                match accumulator {
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
                            vec!["i32".into(); 2],
                            vec!["i32".into(); a_count],
                            vec!["i32".into(); b_count],
                        ]
                        .concat(),
                        vec!["i32".into(); 2],
                        if shape == RegisterMmaShape::M16n8k16 {
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
                            vec!["f32".into(); 4],
                            vec!["i32".into(); a_count],
                            vec!["i32".into(); b_count],
                        ]
                        .concat(),
                        vec!["f32".into(); 4],
                        if shape == RegisterMmaShape::M16n8k16 {
                            RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32
                        } else {
                            RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32
                        },
                    ),
                    _ => unreachable!("validated standard FP8 accumulator"),
                };
            let llvm_arguments = [
                vec!["i32".into(); a_count + b_count],
                if accumulator == RegisterMmaAccumulator::F16 {
                    vec!["v2f16".into(); 2]
                } else {
                    vec!["f32".into(); 4]
                },
            ]
            .concat();
            let llvm_results = if accumulator == RegisterMmaAccumulator::F16 {
                vec!["v2f16".into(); 2]
            } else {
                vec!["f32".into(); 4]
            };
            let result_count = dialect_results.len();

            for &a_element in &admission.a_elements {
                for &b_element in &admission.b_elements {
                    let a = register_mma_fp8_element_name(a_element)
                        .expect("validated standard FP8 A element");
                    let b = register_mma_fp8_element_name(b_element)
                        .expect("validated standard FP8 B element");
                    let id = format!("mma_{shape_name}_fp8_{scalar}_{a}_{b}");
                    let source_record =
                        format!("int_nvvm_mma_{shape_name}_row_col_{scalar}_{a}_{b}_{scalar}");
                    let llvm_symbol =
                        format!("llvm.nvvm.mma.{shape_name}.row.col.{scalar}.{a}.{b}.{scalar}");
                    let mma = RegisterMma {
                        shape,
                        operation: RegisterMmaOperation::Multiply,
                        kind: Some(RegisterMmaKind::Standard),
                        accumulator,
                        a_element,
                        b_element,
                        a_layout: RegisterMmaLayout::Row,
                        b_layout: RegisterMmaLayout::Col,
                        overflow: RegisterMmaOverflow::NotApplicable,
                        participation: RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                        adapter,
                        compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
                        runtime_validation: admission.runtime_validation,
                    };
                    records.push(OverlayIntrinsic {
                        id: id.clone(),
                        abi_id: String::new(),
                        operation_key: format!(
                            "matrix.mma.{shape_name}.row.col.standard_fp8.{scalar}.{a}.{b}.{scalar}"
                        ),
                        family: "register_mma".into(),
                        source: None,
                        source_record: Some(source_record),
                        rust_module: "matrix".into(),
                        rust_name: id.clone(),
                        rust_arguments: rust_arguments.clone(),
                        rust_result: rust_result.clone(),
                        safe: false,
                        must_use: true,
                        safe_allowlist_reason: None,
                        public_rust_path: format!("cuda_intrinsics::matrix::{id}"),
                        compatibility_rust_paths: vec![format!("cuda_device::wmma::{id}")],
                        dialect_op_type: "RegisterMmaOp".into(),
                        dialect_op_name: "nvvm.register_mma".into(),
                        dialect_operands: dialect_operands.clone(),
                        dialect_results: dialect_results.clone(),
                        llvm_symbol: Some(llvm_symbol),
                        resolved_llvm_symbol: None,
                        llvm_arguments: llvm_arguments.clone(),
                        llvm_results: llvm_results.clone(),
                        pure: false,
                        memory: "none".into(),
                        convergent: true,
                        execution_scope: "warp".into(),
                        minimum_ptx: minimum_ptx.into(),
                        minimum_sm: Some("sm_89".into()),
                        ptx_result: rust_result.clone(),
                        targets: "all".into(),
                        ptx_isa_version: "9.3".into(),
                        ptx_isa_section: "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
                        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
                        lowering: "generated_register_mma".into(),
                        backend_lowerings: [
                            (IntrinsicBackend::LlvmNvptx, &admission.llvm_evidence_profile),
                            (IntrinsicBackend::LibNvvm, &admission.libnvvm_evidence_profile),
                        ]
                        .into_iter()
                        .map(|(backend, evidence_profile)| OverlayBackendLowering {
                            backend,
                            mechanism: BackendLoweringMechanism::InlinePtx,
                            evidence_profile: evidence_profile.clone(),
                            targets: None,
                            minimum_ptx: Some(minimum_ptx.into()),
                            minimum_sm: Some("sm_89".into()),
                        })
                        .collect(),
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
                        register_mma: Some(mma),
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
                        selected_address_space: None,
                        expected_ptx: InstructionPattern {
                            mnemonic: "mma".into(),
                            modifiers: ["sync", "aligned", shape_name, "row", "col", scalar, a, b, scalar]
                                .into_iter()
                                .map(Into::into)
                                .collect(),
                            operands: [result_count, a_count, b_count, result_count]
                                .map(|length| OperandPattern::RegisterList { length })
                                .into(),
                        },
                        summary: format!(
                            "Multiplies warp-distributed {a} A and {b} B fragments and adds an {scalar} accumulator."
                        ),
                    });
                }
            }
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

pub(in crate::resolve) const REGISTER_MMA_AMPERE_FLOAT_VARIANTS: [RegisterMmaAmpereFloatVariant;
    5] = [
    RegisterMmaAmpereFloatVariant {
        shape: RegisterMmaShape::M16n8k4,
        accumulator: RegisterMmaAccumulator::F32,
        element: RegisterMmaElement::Tf32,
    },
    RegisterMmaAmpereFloatVariant {
        shape: RegisterMmaShape::M16n8k8,
        accumulator: RegisterMmaAccumulator::F16,
        element: RegisterMmaElement::F16,
    },
    RegisterMmaAmpereFloatVariant {
        shape: RegisterMmaShape::M16n8k8,
        accumulator: RegisterMmaAccumulator::F32,
        element: RegisterMmaElement::Bf16,
    },
    RegisterMmaAmpereFloatVariant {
        shape: RegisterMmaShape::M16n8k8,
        accumulator: RegisterMmaAccumulator::F32,
        element: RegisterMmaElement::F16,
    },
    RegisterMmaAmpereFloatVariant {
        shape: RegisterMmaShape::M16n8k16,
        accumulator: RegisterMmaAccumulator::F16,
        element: RegisterMmaElement::F16,
    },
];

pub(in crate::resolve) fn expand_register_mma_ampere_float_admission(
    admission: &RegisterMmaAmpereFloatAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "Ampere floating-point MMA runtime validation requires GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "Ampere floating-point MMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission.product_count == REGISTER_MMA_AMPERE_FLOAT_VARIANTS.len()
            && admission.variants == REGISTER_MMA_AMPERE_FLOAT_VARIANTS,
        "Ampere floating-point MMA admission must retain the five reviewed variants in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let adapter = match (variant.shape, variant.accumulator) {
                (RegisterMmaShape::M16n8k4, RegisterMmaAccumulator::F32)
                | (RegisterMmaShape::M16n8k8, RegisterMmaAccumulator::F32) => {
                    RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32
                }
                (RegisterMmaShape::M16n8k8, RegisterMmaAccumulator::F16) => {
                    RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32
                }
                (RegisterMmaShape::M16n8k16, RegisterMmaAccumulator::F16) => {
                    RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32
                }
                _ => bail!("Ampere floating-point MMA admission contains an unsupported carrier"),
            };
            let mma = RegisterMma {
                shape: variant.shape,
                operation: RegisterMmaOperation::Multiply,
                kind: None,
                accumulator: variant.accumulator,
                a_element: variant.element,
                b_element: variant.element,
                a_layout: RegisterMmaLayout::Row,
                b_layout: RegisterMmaLayout::Col,
                overflow: RegisterMmaOverflow::NotApplicable,
                participation: RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter,
                compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };
            let recipe = register_mma_recipe(&mma).with_context(|| {
                "Ampere floating-point MMA admission requests a variant outside the closed recipe set"
            })?;
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: String::new(),
                operation_key: recipe.operation_key.into(),
                family: "register_mma".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "matrix".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe.rust_arguments.iter().map(|value| (*value).into()).collect(),
                rust_result: recipe.rust_result.into(),
                safe: false,
                must_use: true,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::matrix::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::wmma::{}", recipe.id)],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe.dialect_operands.iter().map(|value| (*value).into()).collect(),
                dialect_results: recipe.dialect_results.iter().map(|value| (*value).into()).collect(),
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe.llvm_arguments.iter().map(|value| (*value).into()).collect(),
                llvm_results: recipe.llvm_results.iter().map(|value| (*value).into()).collect(),
                pure: false,
                memory: "none".into(),
                convergent: true,
                execution_scope: "warp".into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: Some(recipe.minimum_sm.into()),
                ptx_result: recipe.rust_result.into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
                lowering: "generated_register_mma".into(),
                backend_lowerings: [
                    (IntrinsicBackend::LlvmNvptx, &admission.llvm_evidence_profile),
                    (IntrinsicBackend::LibNvvm, &admission.libnvvm_evidence_profile),
                ]
                .into_iter()
                .map(|(backend, evidence_profile)| OverlayBackendLowering {
                    backend,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: evidence_profile.clone(),
                    targets: None,
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
                })
                .collect(),
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
                register_mma: Some(mma),
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
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "mma".into(),
                    modifiers: recipe.ptx_modifiers.iter().map(|value| (*value).into()).collect(),
                    operands: recipe
                        .ptx_register_counts
                        .map(|length| OperandPattern::RegisterList { length })
                        .into(),
                },
                summary: format!("Executes the {} Ampere floating-point MMA form.", recipe.id),
            })
        })
        .collect()
}

pub(in crate::resolve) fn expand_register_mma_binary_admission(
    admission: &RegisterMmaBinaryAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaLayout::{Col, Row};
    use RegisterMmaOperation::{AndPopc, XorPopc};
    use RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes;
    use RegisterMmaShape::{M8n8k128, M16n8k128, M16n8k256};

    ensure!(
        !admission.variants.is_empty(),
        "compact binary MMA admission has no variants"
    );
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "binary MMA runtime validation may be marked executed only with GPU evidence"
    );

    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(admission.variants.len());
    for variant in &admission.variants {
        ensure!(
            seen.insert((variant.shape, variant.operation)),
            "compact binary MMA admission contains a duplicate variant"
        );
        ensure!(
            matches!(variant.operation, AndPopc | XorPopc),
            "compact binary MMA admission contains a non-binary operation"
        );
        let adapter = match variant.shape {
            M8n8k128 => C2I32A1U32B1U32ToD2I32,
            M16n8k128 => C4I32A2U32B1U32ToD4I32,
            M16n8k256 => C4I32A4U32B2U32ToD4I32,
            _ => bail!("compact binary MMA admission contains an unsupported shape"),
        };
        let mma = RegisterMma {
            shape: variant.shape,
            operation: variant.operation,
            kind: None,
            accumulator: RegisterMmaAccumulator::S32,
            a_element: RegisterMmaElement::B1,
            b_element: RegisterMmaElement::B1,
            a_layout: Row,
            b_layout: Col,
            overflow: RegisterMmaOverflow::Wrapping,
            participation: AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
            adapter,
            compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
            runtime_validation: admission.runtime_validation,
        };
        let recipe = register_mma_recipe(&mma).with_context(
            || "compact binary MMA admission requests a variant outside the closed recipe set",
        )?;
        let operation = match variant.operation {
            AndPopc => "AND and population count",
            XorPopc => "XOR and population count",
            RegisterMmaOperation::Multiply => unreachable!(),
        };
        let summary = format!(
            "Computes warp-distributed binary matrix products with {operation}, then adds a wrapping s32 accumulator."
        );

        records.push(OverlayIntrinsic {
            id: recipe.id.into(),
            abi_id: recipe.abi_id.into(),
            operation_key: recipe.operation_key.into(),
            family: "register_mma".into(),
            source: None,
            source_record: Some(recipe.source_record.into()),
            rust_module: "matrix".into(),
            rust_name: recipe.id.into(),
            rust_arguments: recipe
                .rust_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            rust_result: recipe.rust_result.into(),
            safe: false,
            must_use: true,
            safe_allowlist_reason: None,
            public_rust_path: format!("cuda_intrinsics::matrix::{}", recipe.id),
            compatibility_rust_paths: vec![format!("cuda_device::wmma::{}", recipe.id)],
            dialect_op_type: recipe.dialect_op_type.into(),
            dialect_op_name: recipe.dialect_op_name.into(),
            dialect_operands: recipe
                .dialect_operands
                .iter()
                .map(|value| (*value).into())
                .collect(),
            dialect_results: recipe
                .dialect_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_symbol: Some(recipe.llvm_symbol.into()),
            resolved_llvm_symbol: None,
            llvm_arguments: recipe
                .llvm_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_results: recipe
                .llvm_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            pure: false,
            memory: "none".into(),
            convergent: true,
            execution_scope: "warp".into(),
            minimum_ptx: recipe.minimum_ptx.into(),
            minimum_sm: Some(recipe.minimum_sm.into()),
            ptx_result: recipe.rust_result.into(),
            targets: "all".into(),
            ptx_isa_version: "9.3".into(),
            ptx_isa_section: "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
            lowering: "generated_register_mma".into(),
            backend_lowerings: vec![
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LlvmNvptx,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.llvm_evidence_profile.clone(),
                    targets: None,
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
                },
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LibNvvm,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.libnvvm_evidence_profile.clone(),
                    targets: None,
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
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
            register_mma: Some(mma),
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
            selected_address_space: None,
            expected_ptx: InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: recipe
                    .ptx_modifiers
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                operands: recipe
                    .ptx_register_counts
                    .map(|length| OperandPattern::RegisterList { length })
                    .into(),
            },
            summary,
        });
    }
    Ok(records)
}
