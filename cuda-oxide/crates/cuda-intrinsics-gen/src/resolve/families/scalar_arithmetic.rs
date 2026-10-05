/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayBackendLowering,
    OverlayIntrinsic, RuntimeValidation, ScalarArithmetic, ScalarArithmeticAdmission,
    ScalarArithmeticFormat, ScalarArithmeticOperation, ScalarArithmeticRounding,
    ScalarArithmeticSaturation, ScalarArithmeticSubnormal,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};

use crate::resolve::abi_ledger::*;
use crate::resolve::guards::*;
use crate::resolve::targets::*;

#[derive(Clone)]
pub(in crate::resolve) struct ScalarArithmeticRecipe {
    id: String,
    operation_key: String,
    source_record: String,
    llvm_symbol: String,
    selection_record: String,
    selection_asm: String,
    rust_type: &'static str,
    argument_count: usize,
    properties: Vec<&'static str>,
    ptx_modifiers: Vec<String>,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
}

pub(in crate::resolve) type ScalarArithmeticVariant = (
    ScalarArithmeticFormat,
    ScalarArithmeticOperation,
    ScalarArithmeticRounding,
    ScalarArithmeticSubnormal,
    ScalarArithmeticSaturation,
);

pub(in crate::resolve) fn canonical_scalar_arithmetic_variants() -> Vec<ScalarArithmeticVariant> {
    use ScalarArithmeticFormat::{F32, F64};
    use ScalarArithmeticOperation::{Add, Div, Fma, Mul};
    use ScalarArithmeticRounding::{Rm, Rn, Rp, Rz};
    use ScalarArithmeticSaturation::{None, Sat};
    use ScalarArithmeticSubnormal::{Ftz, Preserve};

    let roundings = [Rn, Rz, Rm, Rp];
    let mut variants = Vec::with_capacity(64);
    for operation in [Mul, Div, Fma] {
        for rounding in roundings {
            variants.push((F64, operation, rounding, Preserve, None));
        }
    }
    for operation in [Mul, Div] {
        for rounding in roundings {
            variants.push((F32, operation, rounding, Preserve, None));
            variants.push((F32, operation, rounding, Ftz, None));
        }
    }
    for rounding in roundings {
        variants.push((F32, Fma, rounding, Preserve, None));
        variants.push((F32, Fma, rounding, Ftz, None));
        variants.push((F32, Fma, rounding, Preserve, Sat));
        variants.push((F32, Fma, rounding, Ftz, Sat));
    }
    for rounding in roundings {
        variants.push((F64, Add, rounding, Preserve, None));
    }
    for rounding in roundings {
        variants.push((F32, Add, rounding, Preserve, None));
        variants.push((F32, Add, rounding, Ftz, None));
        variants.push((F32, Add, rounding, Preserve, Sat));
        variants.push((F32, Add, rounding, Ftz, Sat));
    }
    variants
}

pub(in crate::resolve) fn scalar_arithmetic_format_name(
    format: ScalarArithmeticFormat,
) -> &'static str {
    match format {
        ScalarArithmeticFormat::F32 => "f32",
        ScalarArithmeticFormat::F64 => "f64",
    }
}

pub(in crate::resolve) fn scalar_arithmetic_operation_name(
    operation: ScalarArithmeticOperation,
) -> &'static str {
    match operation {
        ScalarArithmeticOperation::Mul => "mul",
        ScalarArithmeticOperation::Div => "div",
        ScalarArithmeticOperation::Fma => "fma",
        ScalarArithmeticOperation::Add => "add",
    }
}

pub(in crate::resolve) fn scalar_arithmetic_rounding_name(
    rounding: ScalarArithmeticRounding,
) -> &'static str {
    match rounding {
        ScalarArithmeticRounding::Rn => "rn",
        ScalarArithmeticRounding::Rz => "rz",
        ScalarArithmeticRounding::Rm => "rm",
        ScalarArithmeticRounding::Rp => "rp",
    }
}

pub(in crate::resolve) fn scalar_arithmetic_recipe(
    variant: ScalarArithmeticVariant,
) -> Option<ScalarArithmeticRecipe> {
    if !canonical_scalar_arithmetic_variants().contains(&variant) {
        return None;
    }
    let (format, operation, rounding, subnormal, saturation) = variant;
    let operation_name = scalar_arithmetic_operation_name(operation);
    let rounding_name = scalar_arithmetic_rounding_name(rounding);
    let format_name = scalar_arithmetic_format_name(format);
    let source_format = match format {
        ScalarArithmeticFormat::F32 => "f",
        ScalarArithmeticFormat::F64 => "d",
    };

    let mut modifier_names = vec![rounding_name];
    if subnormal == ScalarArithmeticSubnormal::Ftz {
        modifier_names.push("ftz");
    }
    if saturation == ScalarArithmeticSaturation::Sat {
        modifier_names.push("sat");
    }

    let modifier_id = modifier_names.join("_");
    let modifier_symbol = modifier_names.join(".");
    let id = format!("{operation_name}_{modifier_id}_{format_name}");
    let source_record = format!("int_nvvm_{operation_name}_{modifier_id}_{source_format}");
    let llvm_symbol = format!("llvm.nvvm.{operation_name}.{modifier_symbol}.{source_format}");
    let mut ptx_modifier_names = modifier_names.clone();
    if operation == ScalarArithmeticOperation::Add
        && subnormal == ScalarArithmeticSubnormal::Ftz
        && saturation == ScalarArithmeticSaturation::Sat
    {
        ptx_modifier_names.swap(1, 2);
    }
    let ptx_modifiers = ptx_modifier_names
        .iter()
        .copied()
        .chain(std::iter::once(format_name))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let selection_record = match operation {
        ScalarArithmeticOperation::Fma => {
            format!("INT_NVVM_FMA_{modifier_id}_{format_name}")
        }
        ScalarArithmeticOperation::Mul | ScalarArithmeticOperation::Div => format!(
            "INT_NVVM_{}_{}_{}",
            operation_name.to_ascii_uppercase(),
            modifier_id.to_ascii_uppercase(),
            source_format.to_ascii_uppercase()
        ),
        ScalarArithmeticOperation::Add => {
            let selection_modifier = ptx_modifier_names.join("_").to_ascii_uppercase();
            format!(
                "INT_NVVM_ADD_{selection_modifier}_{}",
                source_format.to_ascii_uppercase()
            )
        }
    };
    let argument_count = if operation == ScalarArithmeticOperation::Fma {
        3
    } else {
        2
    };
    let operands = (0..argument_count)
        .map(|argument| format!("$src{argument}"))
        .collect::<Vec<_>>()
        .join(", ");
    let ptx_modifier_symbol = ptx_modifier_names.join(".");
    let selection_asm =
        format!("{operation_name}.{ptx_modifier_symbol}.{format_name} \t$dst, {operands};");
    // LLVM 23 marks every scalar-arithmetic NVVM intrinsic
    // IntrNoCreateUndefOrPoison uniformly (LLVM 22 only had it on fma.rn.d),
    // so the property set no longer varies by format.
    let properties = match (operation, format) {
        (ScalarArithmeticOperation::Mul | ScalarArithmeticOperation::Add, _) => {
            vec![
                "Commutative",
                "IntrNoCreateUndefOrPoison",
                "IntrNoMem",
                "IntrSpeculatable",
            ]
        }
        (ScalarArithmeticOperation::Div, _) => vec!["IntrNoCreateUndefOrPoison", "IntrNoMem"],
        (ScalarArithmeticOperation::Fma, _) => {
            vec!["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"]
        }
    };
    let (ptx_isa_section, ptx_isa_url) = match operation {
        ScalarArithmeticOperation::Mul => (
            "9.7.3.5 Floating Point Instructions: mul",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul",
        ),
        ScalarArithmeticOperation::Fma => (
            "9.7.3.6 Floating Point Instructions: fma",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma",
        ),
        ScalarArithmeticOperation::Div => (
            "9.7.3.8 Floating Point Instructions: div",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div",
        ),
        ScalarArithmeticOperation::Add => (
            "9.7.3.3 Floating Point Instructions: add",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add",
        ),
    };

    Some(ScalarArithmeticRecipe {
        id,
        operation_key: format!(
            "scalar.arithmetic.{operation_name}.{modifier_symbol}.{format_name}"
        ),
        source_record,
        llvm_symbol,
        selection_record,
        selection_asm,
        rust_type: format_name,
        argument_count,
        properties,
        ptx_modifiers,
        ptx_isa_section,
        ptx_isa_url,
    })
}

pub(in crate::resolve) fn expand_scalar_arithmetic_admission(
    admission: &ScalarArithmeticAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "scalar-arithmetic runtime may be marked executed only with GPU evidence"
    );
    let expected = canonical_scalar_arithmetic_variants();
    let actual = admission
        .variants
        .iter()
        .map(|variant| {
            (
                variant.format,
                variant.operation,
                variant.rounding,
                variant.subnormal,
                variant.saturation,
            )
        })
        .collect::<Vec<_>>();
    ensure!(
        actual == expected,
        "compact scalar-arithmetic admission must list the canonical 64 variants"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            validate_abi_id(&variant.abi_id)?;
            let identity = (
                variant.format,
                variant.operation,
                variant.rounding,
                variant.subnormal,
                variant.saturation,
            );
            let recipe = scalar_arithmetic_recipe(identity)
                .context("scalar arithmetic is outside the closed recipe set")?;
            scalar_arithmetic_overlay_record(recipe, admission, identity, &variant.abi_id)
        })
        .collect()
}

pub(in crate::resolve) fn scalar_arithmetic_overlay_record(
    recipe: ScalarArithmeticRecipe,
    admission: &ScalarArithmeticAdmission,
    variant: ScalarArithmeticVariant,
    abi_id: &str,
) -> Result<OverlayIntrinsic> {
    let (format, operation, rounding, subnormal, saturation) = variant;
    let rust_arguments = vec![recipe.rust_type.into(); recipe.argument_count];
    let ptx_operands = vec![OperandPattern::Register; recipe.argument_count + 1];
    let summary = format!(
        "Computes scalar {} with explicit {} rounding.",
        scalar_arithmetic_operation_name(operation),
        scalar_arithmetic_rounding_name(rounding)
    );
    Ok(OverlayIntrinsic {
        id: recipe.id.clone(),
        abi_id: abi_id.into(),
        operation_key: recipe.operation_key,
        family: "scalar_arithmetic".into(),
        source: None,
        source_record: Some(recipe.source_record),
        rust_module: "float".into(),
        rust_name: recipe.id.clone(),
        rust_arguments: rust_arguments.clone(),
        rust_result: recipe.rust_type.into(),
        safe: true,
        must_use: true,
        safe_allowlist_reason: Some("Scalar arithmetic has no caller obligations.".into()),
        public_rust_path: format!("cuda_intrinsics::float::{}", recipe.id),
        compatibility_rust_paths: vec![format!("cuda_device::float::{}", recipe.id)],
        dialect_op_type: "ScalarArithmeticOp".into(),
        dialect_op_name: "nvvm.scalar_arithmetic".into(),
        dialect_operands: rust_arguments.clone(),
        dialect_results: vec![recipe.rust_type.into()],
        llvm_symbol: Some(recipe.llvm_symbol),
        resolved_llvm_symbol: None,
        llvm_arguments: rust_arguments,
        llvm_results: vec![recipe.rust_type.into()],
        pure: operation != ScalarArithmeticOperation::Div,
        memory: "none".into(),
        convergent: false,
        execution_scope: "thread".into(),
        minimum_ptx: "7.0".into(),
        minimum_sm: Some("sm_80".into()),
        ptx_result: recipe.rust_type.into(),
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: recipe.ptx_isa_section.into(),
        ptx_isa_url: recipe.ptx_isa_url.into(),
        lowering: "generated_scalar_arithmetic".into(),
        backend_lowerings: [
            (
                IntrinsicBackend::LlvmNvptx,
                &admission.llvm_evidence_profile,
            ),
            (
                IntrinsicBackend::LibNvvm,
                &admission.libnvvm_evidence_profile,
            ),
        ]
        .into_iter()
        .map(|(backend, evidence_profile)| OverlayBackendLowering {
            backend,
            mechanism: match backend {
                IntrinsicBackend::LlvmNvptx if saturation == ScalarArithmeticSaturation::Sat => {
                    // LLVM 21 has no typed intrinsic for these LLVM 22 forms.
                    BackendLoweringMechanism::InlinePtx
                }
                IntrinsicBackend::LlvmNvptx => BackendLoweringMechanism::TypedNvvm,
                IntrinsicBackend::LibNvvm => BackendLoweringMechanism::InlinePtx,
            },
            evidence_profile: evidence_profile.clone(),
            targets: None,
            minimum_ptx: Some("7.0".into()),
            minimum_sm: Some("sm_80".into()),
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
        scalar_arithmetic: Some(ScalarArithmetic {
            format,
            operation,
            rounding,
            subnormal,
            saturation,
            runtime_validation: admission.runtime_validation,
        }),
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
            mnemonic: scalar_arithmetic_operation_name(operation).into(),
            modifiers: recipe.ptx_modifiers,
            operands: ptx_operands,
        },
        summary,
    })
}

pub(in crate::resolve) fn validate_scalar_arithmetic_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let arithmetic = policy
        .scalar_arithmetic
        .as_ref()
        .with_context(|| format!("{} has no scalar-arithmetic contract", policy.id))?;
    let variant = (
        arithmetic.format,
        arithmetic.operation,
        arithmetic.rounding,
        arithmetic.subnormal,
        arithmetic.saturation,
    );
    let recipe = scalar_arithmetic_recipe(variant).with_context(|| {
        format!(
            "{} is outside the closed scalar-arithmetic recipe",
            policy.id
        )
    })?;
    ensure!(
        arithmetic.runtime_validation == RuntimeValidation::Unexecuted,
        "{} scalar-arithmetic runtime may be executed only with GPU evidence",
        policy.id
    );
    let signature = vec![recipe.rust_type.to_owned(); recipe.argument_count];
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && policy.source_record.as_deref() == Some(recipe.source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments == signature
            && policy.llvm_results == [recipe.rust_type],
        "{} scalar-arithmetic identity or LLVM source changed",
        policy.id
    );
    let expected_properties = recipe
        .properties
        .iter()
        .map(|property| (*property).to_owned())
        .collect::<Vec<_>>();
    ensure!(
        declaration.properties == expected_properties,
        "{} imported scalar-arithmetic properties changed",
        policy.id
    );
    let direct = declaration
        .selections
        .first()
        .with_context(|| format!("{} has no imported scalar selection", policy.id))?;
    ensure!(
        direct.source_record == recipe.selection_record
            && direct.asm == recipe.selection_asm
            && direct.predicates.is_empty()
            && direct.constraints.is_empty(),
        "{} direct scalar-arithmetic selection changed",
        policy.id
    );
    if arithmetic.operation == ScalarArithmeticOperation::Add {
        let rounding = scalar_arithmetic_rounding_name(arithmetic.rounding);
        let source_modifiers = [
            Some(rounding),
            (arithmetic.subnormal == ScalarArithmeticSubnormal::Ftz).then_some("ftz"),
            (arithmetic.saturation == ScalarArithmeticSaturation::Sat).then_some("sat"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("_");
        let mut expected = Vec::new();
        if arithmetic.format == ScalarArithmeticFormat::F32
            && arithmetic.subnormal == ScalarArithmeticSubnormal::Preserve
        {
            let saturation = if arithmetic.saturation == ScalarArithmeticSaturation::Sat {
                "_sat"
            } else {
                ""
            };
            for instruction in ["ADD", "SUB"] {
                for input_format in ["bf16", "f16"] {
                    expected.push((
                        format!(
                            "INT_NVVM_MIXED_{instruction}_{rounding}{saturation}_f32_{input_format}"
                        ),
                        format!(
                            "{}.{rounding}{}.f32.{input_format} \t$dst, $a, $b;",
                            instruction.to_ascii_lowercase(),
                            if arithmetic.saturation == ScalarArithmeticSaturation::Sat {
                                ".sat"
                            } else {
                                ""
                            }
                        ),
                        vec![
                            "Subtarget->getSmVersion() >= 100".to_owned(),
                            "Subtarget->getPTXVersion() >= 86".to_owned(),
                        ],
                    ));
                }
            }
        }
        let source_format = match arithmetic.format {
            ScalarArithmeticFormat::F32 => "F",
            ScalarArithmeticFormat::F64 => "D",
        };
        expected.push((
            format!("INT_NVVM_SUB_{source_modifiers}_{source_format}"),
            format!("sub.{} \t$dst, $a, $b;", recipe.ptx_modifiers.join(".")),
            Vec::new(),
        ));
        ensure!(
            declaration.selections.len() == expected.len() + 1,
            "{} must retain one direct add and the reviewed non-add alternatives",
            policy.id
        );
        for (selection, (source_record, asm, predicates)) in
            declaration.selections[1..].iter().zip(expected)
        {
            ensure!(
                selection.source_record == source_record
                    && selection.asm == asm
                    && selection.predicates == predicates
                    && selection.constraints.is_empty(),
                "{} non-add scalar selection changed",
                policy.id
            );
        }
    } else if arithmetic.format == ScalarArithmeticFormat::F32
        && arithmetic.operation == ScalarArithmeticOperation::Fma
        && arithmetic.subnormal == ScalarArithmeticSubnormal::Preserve
    {
        ensure!(
            declaration.selections.len() == 3,
            "{} must retain one direct and two mixed-input selections",
            policy.id
        );
        let rounding = scalar_arithmetic_rounding_name(arithmetic.rounding);
        let saturation = if arithmetic.saturation == ScalarArithmeticSaturation::Sat {
            "_sat"
        } else {
            ""
        };
        for (selection, input_format) in declaration.selections[1..].iter().zip(["bf16", "f16"]) {
            ensure!(
                selection.source_record
                    == format!("INT_NVVM_MIXED_FMA_{rounding}{saturation}_f32_{input_format}")
                    && selection.asm
                        == format!(
                            "fma.{rounding}{}.f32.{input_format} \t$dst, $a, $b, $c;",
                            if arithmetic.saturation == ScalarArithmeticSaturation::Sat {
                                ".sat"
                            } else {
                                ""
                            }
                        )
                    && selection.predicates
                        == [
                            "Subtarget->getSmVersion() >= 100",
                            "Subtarget->getPTXVersion() >= 86",
                        ]
                    && selection.constraints.is_empty(),
                "{} mixed-input scalar-arithmetic selection changed",
                policy.id
            );
        }
    } else {
        ensure!(
            declaration.selections.len() == 1,
            "{} gained an unreviewed scalar-arithmetic selection",
            policy.id
        );
    }
    let mut selected = Vec::new();
    for selection in &declaration.selections {
        if selection_matches_policy(policy, selection)? {
            selected.push(selection);
        }
    }
    ensure!(
        selected.len() == 1 && selected[0].source_record == recipe.selection_record,
        "{} must select only its direct scalar arithmetic instruction",
        policy.id
    );
    ensure!(
        policy.rust_module == "float"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == signature
            && policy.rust_result == recipe.rust_type
            && policy.safe
            && policy.must_use
            && policy.compatibility_rust_paths == [format!("cuda_device::float::{}", recipe.id)]
            && policy.dialect_op_type == "ScalarArithmeticOp"
            && policy.dialect_op_name == "nvvm.scalar_arithmetic"
            && policy.dialect_operands == signature
            && policy.dialect_results == [recipe.rust_type]
            && policy.lowering == "generated_scalar_arithmetic",
        "{} changed its scalar-arithmetic API, carrier, or lowering",
        policy.id
    );
    ensure!(
        policy.pure == (arithmetic.operation != ScalarArithmeticOperation::Div)
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == recipe.rust_type
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} scalar-arithmetic effects, provenance, or target floor changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: scalar_arithmetic_operation_name(arithmetic.operation).into(),
                modifiers: recipe.ptx_modifiers.clone(),
                operands: vec![OperandPattern::Register; recipe.argument_count + 1],
            },
        "{} expected scalar-arithmetic PTX changed",
        policy.id
    );
    let llvm_mechanism = if arithmetic.saturation == ScalarArithmeticSaturation::Sat {
        BackendLoweringMechanism::InlinePtx
    } else {
        BackendLoweringMechanism::TypedNvvm
    };
    let expected_backends = [
        (IntrinsicBackend::LlvmNvptx, llvm_mechanism),
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
                        && lowering.minimum_ptx.as_deref() == Some("7.0")
                        && lowering.minimum_sm.as_deref() == Some("sm_80")
                        && !lowering.evidence_profile.trim().is_empty()
                })
            }),
        "{} has the wrong reviewed scalar-arithmetic backend routes",
        policy.id
    );
    validate_selected_target_predicates(policy, direct)?;
    ensure_no_other_family_contract(policy, "scalar arithmetic")?;
    Ok(())
}
