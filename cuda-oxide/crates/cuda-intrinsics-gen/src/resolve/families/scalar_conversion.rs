/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayBackendLowering,
    OverlayIntrinsic, RuntimeValidation, ScalarConversion, ScalarConversionAdapter,
    ScalarConversionAdmission, ScalarConversionDestinationFormat,
    ScalarConversionResultRepresentation, ScalarConversionRounding, ScalarConversionSaturation,
    ScalarConversionSourceFormat,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};

use crate::resolve::guards::*;
use crate::resolve::targets::*;

#[derive(Clone, Copy)]
pub(in crate::resolve) struct ScalarConversionRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    rust_name: &'static str,
    selection_record: &'static str,
    selection_asm: &'static str,
    minimum_ptx: &'static str,
    minimum_sm: &'static str,
    ptx_modifiers: &'static [&'static str],
}

pub(in crate::resolve) fn scalar_conversion_recipe(
    rounding: ScalarConversionRounding,
    saturation: ScalarConversionSaturation,
) -> Option<ScalarConversionRecipe> {
    use ScalarConversionRounding::{NearestAway, NearestEven, TowardZero};
    use ScalarConversionSaturation::{None, Relu, ReluSatfinite, Satfinite};

    Some(match (rounding, saturation) {
        (NearestAway, None) => ScalarConversionRecipe {
            id: "cvt_rna_tf32_f32",
            abi_id: "i0368",
            operation_key: "convert.f32.tf32.rna",
            source_record: "int_nvvm_f2tf32_rna",
            llvm_symbol: "llvm.nvvm.f2tf32.rna",
            rust_name: "cvt_rna_tf32_f32",
            selection_record: "CVT_to_tf32_rna",
            selection_asm: "cvt.rna.tf32.f32 \t$dst, $src;",
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: &["rna", "tf32", "f32"],
        },
        (NearestAway, Satfinite) => ScalarConversionRecipe {
            id: "cvt_rna_satfinite_tf32_f32",
            abi_id: "i0369",
            operation_key: "convert.f32.tf32.rna.satfinite",
            source_record: "int_nvvm_f2tf32_rna_satfinite",
            llvm_symbol: "llvm.nvvm.f2tf32.rna.satfinite",
            rust_name: "cvt_rna_satfinite_tf32_f32",
            selection_record: "CVT_to_tf32_rna_satf",
            selection_asm: "cvt.rna.satfinite.tf32.f32 \t$dst, $src;",
            minimum_ptx: "8.1",
            minimum_sm: "sm_80",
            ptx_modifiers: &["rna", "satfinite", "tf32", "f32"],
        },
        (NearestEven, None) => ScalarConversionRecipe {
            id: "cvt_rn_tf32_f32",
            abi_id: "i0370",
            operation_key: "convert.f32.tf32.rn",
            source_record: "int_nvvm_f2tf32_rn",
            llvm_symbol: "llvm.nvvm.f2tf32.rn",
            rust_name: "cvt_rn_tf32_f32",
            selection_record: "CVT_to_tf32_rn",
            selection_asm: "cvt.rn.tf32.f32 \t$dst, $src;",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_modifiers: &["rn", "tf32", "f32"],
        },
        (NearestEven, Relu) => ScalarConversionRecipe {
            id: "cvt_rn_relu_tf32_f32",
            abi_id: "i0371",
            operation_key: "convert.f32.tf32.rn.relu",
            source_record: "int_nvvm_f2tf32_rn_relu",
            llvm_symbol: "llvm.nvvm.f2tf32.rn.relu",
            rust_name: "cvt_rn_relu_tf32_f32",
            selection_record: "CVT_to_tf32_rn_relu",
            selection_asm: "cvt.rn.relu.tf32.f32 \t$dst, $src;",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_modifiers: &["rn", "relu", "tf32", "f32"],
        },
        (NearestEven, Satfinite) => ScalarConversionRecipe {
            id: "cvt_rn_satfinite_tf32_f32",
            abi_id: "i0372",
            operation_key: "convert.f32.tf32.rn.satfinite",
            source_record: "int_nvvm_f2tf32_rn_satfinite",
            llvm_symbol: "llvm.nvvm.f2tf32.rn.satfinite",
            rust_name: "cvt_rn_satfinite_tf32_f32",
            selection_record: "CVT_to_tf32_rn_satf",
            selection_asm: "cvt.rn.satfinite.tf32.f32 \t$dst, $src;",
            minimum_ptx: "8.6",
            minimum_sm: "sm_100",
            ptx_modifiers: &["rn", "satfinite", "tf32", "f32"],
        },
        (NearestEven, ReluSatfinite) => ScalarConversionRecipe {
            id: "cvt_rn_relu_satfinite_tf32_f32",
            abi_id: "i0373",
            operation_key: "convert.f32.tf32.rn.relu.satfinite",
            source_record: "int_nvvm_f2tf32_rn_relu_satfinite",
            llvm_symbol: "llvm.nvvm.f2tf32.rn.relu.satfinite",
            rust_name: "cvt_rn_relu_satfinite_tf32_f32",
            selection_record: "CVT_to_tf32_rn_relu_satf",
            selection_asm: "cvt.rn.relu.satfinite.tf32.f32 \t$dst, $src;",
            minimum_ptx: "8.6",
            minimum_sm: "sm_100",
            ptx_modifiers: &["rn", "relu", "satfinite", "tf32", "f32"],
        },
        (TowardZero, None) => ScalarConversionRecipe {
            id: "cvt_rz_tf32_f32",
            abi_id: "i0374",
            operation_key: "convert.f32.tf32.rz",
            source_record: "int_nvvm_f2tf32_rz",
            llvm_symbol: "llvm.nvvm.f2tf32.rz",
            rust_name: "cvt_rz_tf32_f32",
            selection_record: "CVT_to_tf32_rz",
            selection_asm: "cvt.rz.tf32.f32 \t$dst, $src;",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_modifiers: &["rz", "tf32", "f32"],
        },
        (TowardZero, Relu) => ScalarConversionRecipe {
            id: "cvt_rz_relu_tf32_f32",
            abi_id: "i0375",
            operation_key: "convert.f32.tf32.rz.relu",
            source_record: "int_nvvm_f2tf32_rz_relu",
            llvm_symbol: "llvm.nvvm.f2tf32.rz.relu",
            rust_name: "cvt_rz_relu_tf32_f32",
            selection_record: "CVT_to_tf32_rz_relu",
            selection_asm: "cvt.rz.relu.tf32.f32 \t$dst, $src;",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_modifiers: &["rz", "relu", "tf32", "f32"],
        },
        (TowardZero, Satfinite) => ScalarConversionRecipe {
            id: "cvt_rz_satfinite_tf32_f32",
            abi_id: "i0376",
            operation_key: "convert.f32.tf32.rz.satfinite",
            source_record: "int_nvvm_f2tf32_rz_satfinite",
            llvm_symbol: "llvm.nvvm.f2tf32.rz.satfinite",
            rust_name: "cvt_rz_satfinite_tf32_f32",
            selection_record: "CVT_to_tf32_rz_satf",
            selection_asm: "cvt.rz.satfinite.tf32.f32 \t$dst, $src;",
            minimum_ptx: "8.6",
            minimum_sm: "sm_100",
            ptx_modifiers: &["rz", "satfinite", "tf32", "f32"],
        },
        (TowardZero, ReluSatfinite) => ScalarConversionRecipe {
            id: "cvt_rz_relu_satfinite_tf32_f32",
            abi_id: "i0377",
            operation_key: "convert.f32.tf32.rz.relu.satfinite",
            source_record: "int_nvvm_f2tf32_rz_relu_satfinite",
            llvm_symbol: "llvm.nvvm.f2tf32.rz.relu.satfinite",
            rust_name: "cvt_rz_relu_satfinite_tf32_f32",
            selection_record: "CVT_to_tf32_rz_relu_satf",
            selection_asm: "cvt.rz.relu.satfinite.tf32.f32 \t$dst, $src;",
            minimum_ptx: "8.6",
            minimum_sm: "sm_100",
            ptx_modifiers: &["rz", "relu", "satfinite", "tf32", "f32"],
        },
        _ => return Option::None,
    })
}

pub(in crate::resolve) fn expand_scalar_conversion_admission(
    admission: &ScalarConversionAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "scalar-conversion runtime may be marked executed only with GPU evidence"
    );
    let expected = [
        (
            ScalarConversionRounding::NearestAway,
            ScalarConversionSaturation::None,
        ),
        (
            ScalarConversionRounding::NearestAway,
            ScalarConversionSaturation::Satfinite,
        ),
        (
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::None,
        ),
        (
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::Relu,
        ),
        (
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::Satfinite,
        ),
        (
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::ReluSatfinite,
        ),
        (
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::None,
        ),
        (
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::Relu,
        ),
        (
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::Satfinite,
        ),
        (
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::ReluSatfinite,
        ),
    ];
    let actual = admission
        .variants
        .iter()
        .map(|variant| (variant.rounding, variant.saturation))
        .collect::<Vec<_>>();
    ensure!(
        actual == expected,
        "compact scalar-conversion admission must list the canonical ten variants"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = scalar_conversion_recipe(variant.rounding, variant.saturation)
                .context("scalar conversion is outside the closed recipe set")?;
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must reserve ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            scalar_conversion_overlay_record(
                recipe,
                admission,
                variant.rounding,
                variant.saturation,
            )
        })
        .collect()
}

pub(in crate::resolve) fn scalar_conversion_overlay_record(
    recipe: ScalarConversionRecipe,
    admission: &ScalarConversionAdmission,
    rounding: ScalarConversionRounding,
    saturation: ScalarConversionSaturation,
) -> Result<OverlayIntrinsic> {
    Ok(OverlayIntrinsic {
        id: recipe.id.into(),
        abi_id: recipe.abi_id.into(),
        operation_key: recipe.operation_key.into(),
        family: "scalar_conversion".into(),
        source: None,
        source_record: Some(recipe.source_record.into()),
        rust_module: "convert".into(),
        rust_name: recipe.rust_name.into(),
        rust_arguments: vec!["f32".into()],
        rust_result: "u32".into(),
        safe: true,
        must_use: true,
        safe_allowlist_reason: Some("This conversion has no caller obligations.".into()),
        public_rust_path: format!("cuda_intrinsics::convert::{}", recipe.rust_name),
        compatibility_rust_paths: vec![format!(
            "cuda_device::convert::{}",
            recipe.rust_name
        )],
        dialect_op_type: "ScalarConversionOp".into(),
        dialect_op_name: "nvvm.scalar_conversion".into(),
        dialect_operands: vec!["f32".into()],
        dialect_results: vec!["i32".into()],
        llvm_symbol: Some(recipe.llvm_symbol.into()),
        resolved_llvm_symbol: None,
        llvm_arguments: vec!["f32".into()],
        llvm_results: vec!["i32".into()],
        pure: true,
        memory: "none".into(),
        convergent: false,
        execution_scope: "thread".into(),
        minimum_ptx: recipe.minimum_ptx.into(),
        minimum_sm: Some(recipe.minimum_sm.into()),
        ptx_result: "u32".into(),
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: "9.7.9.22 Data Movement and Conversion Instructions: cvt".into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt".into(),
        lowering: "generated_scalar_conversion".into(),
        backend_lowerings: [
            (IntrinsicBackend::LlvmNvptx, &admission.llvm_evidence_profile),
            (IntrinsicBackend::LibNvvm, &admission.libnvvm_evidence_profile),
        ]
        .into_iter()
        .map(|(backend, evidence_profile)| OverlayBackendLowering {
            backend,
            mechanism: match backend {
                IntrinsicBackend::LlvmNvptx => BackendLoweringMechanism::TypedNvvm,
                IntrinsicBackend::LibNvvm => BackendLoweringMechanism::InlinePtx,
            },
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
        scalar_conversion: Some(ScalarConversion {
            source_format: ScalarConversionSourceFormat::F32,
            destination_format: ScalarConversionDestinationFormat::Tf32,
            rounding,
            saturation,
            result_representation: ScalarConversionResultRepresentation::RawU32Bits,
            adapter: ScalarConversionAdapter::DirectF32ToRawU32Bits,
            runtime_validation: admission.runtime_validation,
        }),
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
        selected_address_space: None,
        expected_ptx: InstructionPattern {
            mnemonic: "cvt".into(),
            modifiers: recipe.ptx_modifiers.iter().map(|value| (*value).into()).collect(),
            operands: vec![OperandPattern::Register, OperandPattern::Register],
        },
        summary: format!(
            "Converts one f32 value with {} and returns raw TF32 bits.",
            recipe.ptx_modifiers[0]
        ),
    })
}

pub(in crate::resolve) fn validate_scalar_conversion_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let conversion = policy
        .scalar_conversion
        .as_ref()
        .with_context(|| format!("{} has no scalar-conversion contract", policy.id))?;
    let recipe = scalar_conversion_recipe(conversion.rounding, conversion.saturation)
        .with_context(|| {
            format!(
                "{} is outside the closed scalar-conversion recipe",
                policy.id
            )
        })?;
    ensure!(
        conversion.source_format == ScalarConversionSourceFormat::F32
            && conversion.destination_format == ScalarConversionDestinationFormat::Tf32
            && conversion.result_representation == ScalarConversionResultRepresentation::RawU32Bits
            && conversion.adapter == ScalarConversionAdapter::DirectF32ToRawU32Bits
            && conversion.runtime_validation == RuntimeValidation::Unexecuted,
        "{} changed its scalar-conversion representation or adapter",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments == ["f32"]
            && policy.llvm_results == ["i32"],
        "{} scalar-conversion identity or LLVM source changed",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"]
            && declaration.selections.len() == 1
            && declaration.selections[0].source_record == recipe.selection_record
            && declaration.selections[0].asm == recipe.selection_asm
            && declaration.selections[0].constraints.is_empty(),
        "{} imported scalar-conversion declaration or selection changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "convert"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["f32"]
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use
            && policy.compatibility_rust_paths
                == [format!("cuda_device::convert::{}", recipe.rust_name)]
            && policy.dialect_op_type == "ScalarConversionOp"
            && policy.dialect_op_name == "nvvm.scalar_conversion"
            && policy.dialect_operands == ["f32"]
            && policy.dialect_results == ["i32"]
            && policy.lowering == "generated_scalar_conversion",
        "{} changed its scalar-conversion API, carrier, or lowering",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} scalar-conversion effects or target floor changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: "cvt".into(),
                modifiers: recipe
                    .ptx_modifiers
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                operands: vec![OperandPattern::Register, OperandPattern::Register],
            },
        "{} expected PTX changed",
        policy.id
    );
    let expected_backends = [
        (
            IntrinsicBackend::LlvmNvptx,
            BackendLoweringMechanism::TypedNvvm,
        ),
        (
            IntrinsicBackend::LibNvvm,
            BackendLoweringMechanism::InlinePtx,
        ),
    ];
    ensure!(
        policy.backend_lowerings.len() == 2
            && expected_backends.into_iter().all(|(backend, mechanism)| {
                policy.backend_lowerings.iter().any(|lowering| {
                    lowering.backend == backend
                        && lowering.mechanism == mechanism
                        && lowering.minimum_ptx.as_deref() == Some(recipe.minimum_ptx)
                        && lowering.minimum_sm.as_deref() == Some(recipe.minimum_sm)
                        && !lowering.evidence_profile.trim().is_empty()
                })
            }),
        "{} must define the typed LLVM and inline-PTX libNVVM routes",
        policy.id
    );
    validate_selected_target_predicates(policy, &declaration.selections[0])?;
    ensure_no_other_family_contract(policy, "scalar conversion")?;
    Ok(())
}
