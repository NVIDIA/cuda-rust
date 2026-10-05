/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, IntrinsicSource,
    OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation, ScalarMath, ScalarMathAdmission,
    ScalarMathFormat, ScalarMathOperation, ScalarMathPrecision, ScalarMathSubnormal,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};

use crate::resolve::abi_ledger::*;
use crate::resolve::guards::*;
use crate::resolve::targets::*;

pub(in crate::resolve) type ScalarMathVariant = (
    ScalarMathFormat,
    ScalarMathOperation,
    ScalarMathPrecision,
    ScalarMathSubnormal,
);

#[derive(Clone, PartialEq, Eq)]
pub(in crate::resolve) enum ScalarMathRecipeSource {
    /// Bound to a monomorphic tblgen record in the pinned import.
    Imported {
        source_record: String,
        llvm_symbol: String,
    },
    /// Bound to an overloaded (polymorphic) tblgen record: the import
    /// carries `anonymous_8`/`anyfloat` signature tokens and the concrete
    /// f32 instantiation is recorded as the resolved symbol. Only ex2 uses
    /// this (LLVM 23, like 22, models it as `int_nvvm_ex2_approx{,_ftz}` without a
    /// per-format record).
    ImportedOverloaded {
        source_record: String,
        llvm_symbol: String,
        resolved_llvm_symbol: String,
    },
    /// No record exists in the pinned tblgen import at all. Only tanh uses
    /// this: llc selects `llvm.nvvm.tanh.approx.f32` via NVVMIntrinsic-class
    /// matching, but the import exports no record for it, so the op is
    /// admitted directly against the PTX instruction.
    PtxNative { instruction: String },
}

pub(in crate::resolve) struct ScalarMathRecipe {
    id: String,
    operation_key: String,
    source: ScalarMathRecipeSource,
    rust_type: &'static str,
    dialect_type: &'static str,
    minimum_ptx: &'static str,
    minimum_sm: &'static str,
    properties: Vec<&'static str>,
    ptx_modifiers: Vec<String>,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
    /// The generator's tblgen import found no DAG selection pattern for the
    /// intrinsic, so the lowering routes it through inline PTX. Note this is
    /// an import limitation, not an llc one: llc still selects these
    /// intrinsics through NVVMIntrinsic-class pattern matching, which the
    /// evidence import cannot see. Promoting them to typed calls once the
    /// import understands those patterns would reopen them to LLVM
    /// optimization.
    force_inline_ptx: bool,
}

pub(in crate::resolve) fn scalar_math_format_name(format: ScalarMathFormat) -> &'static str {
    match format {
        ScalarMathFormat::F16 => "f16",
        ScalarMathFormat::F32 => "f32",
        ScalarMathFormat::F64 => "f64",
    }
}

pub(in crate::resolve) fn scalar_math_operation_name(
    operation: ScalarMathOperation,
) -> &'static str {
    match operation {
        ScalarMathOperation::Sin => "sin",
        ScalarMathOperation::Cos => "cos",
        ScalarMathOperation::Ex2 => "ex2",
        ScalarMathOperation::Lg2 => "lg2",
        ScalarMathOperation::Rcp => "rcp",
        ScalarMathOperation::Rsqrt => "rsqrt",
        ScalarMathOperation::Sqrt => "sqrt",
        ScalarMathOperation::Tanh => "tanh",
    }
}

pub(in crate::resolve) fn scalar_math_precision_name(
    precision: ScalarMathPrecision,
) -> &'static str {
    match precision {
        ScalarMathPrecision::Approx => "approx",
        ScalarMathPrecision::Rn => "rn",
        ScalarMathPrecision::Rz => "rz",
        ScalarMathPrecision::Rm => "rm",
        ScalarMathPrecision::Rp => "rp",
    }
}

pub(in crate::resolve) fn canonical_scalar_math_variants() -> Vec<ScalarMathVariant> {
    use ScalarMathFormat::{F16, F32, F64};
    use ScalarMathOperation::{Cos, Ex2, Lg2, Rcp, Rsqrt, Sin, Sqrt, Tanh};
    use ScalarMathPrecision::{Approx, Rm, Rn, Rp, Rz};
    use ScalarMathSubnormal::{Ftz, Preserve};

    vec![
        // sin: approx f32 only
        (F32, Sin, Approx, Preserve),
        (F32, Sin, Approx, Ftz),
        // cos: approx f32 only
        (F32, Cos, Approx, Preserve),
        (F32, Cos, Approx, Ftz),
        // lg2: approx f32 only (f64 approx produces invalid PTX)
        (F32, Lg2, Approx, Preserve),
        (F32, Lg2, Approx, Ftz),
        // rcp: approx only with ftz, rounded for both formats
        (F32, Rcp, Approx, Ftz),
        (F32, Rcp, Rn, Preserve),
        (F32, Rcp, Rn, Ftz),
        (F32, Rcp, Rz, Preserve),
        (F32, Rcp, Rz, Ftz),
        (F32, Rcp, Rm, Preserve),
        (F32, Rcp, Rm, Ftz),
        (F32, Rcp, Rp, Preserve),
        (F32, Rcp, Rp, Ftz),
        (F64, Rcp, Approx, Ftz),
        (F64, Rcp, Rn, Preserve),
        (F64, Rcp, Rz, Preserve),
        (F64, Rcp, Rm, Preserve),
        (F64, Rcp, Rp, Preserve),
        // rsqrt: approx only (PTX has no rounded rsqrt). The f64+ftz variant
        // is valid PTX (`rsqrt.approx.ftz.f64` assembles for sm_80+, and LLVM
        // selects `llvm.nvvm.rsqrt.approx.ftz.d` directly); it is deferred
        // only because it has not been probed under the pinned evidence
        // profile yet, not because the instruction is invalid.
        (F32, Rsqrt, Approx, Preserve),
        (F32, Rsqrt, Approx, Ftz),
        (F64, Rsqrt, Approx, Preserve),
        // sqrt: approx f32 only, rounded for both formats
        (F32, Sqrt, Approx, Preserve),
        (F32, Sqrt, Approx, Ftz),
        (F32, Sqrt, Rn, Preserve),
        (F32, Sqrt, Rn, Ftz),
        (F32, Sqrt, Rz, Preserve),
        (F32, Sqrt, Rz, Ftz),
        (F32, Sqrt, Rm, Preserve),
        (F32, Sqrt, Rm, Ftz),
        (F32, Sqrt, Rp, Preserve),
        (F32, Sqrt, Rp, Ftz),
        (F64, Sqrt, Rn, Preserve),
        (F64, Sqrt, Rz, Preserve),
        (F64, Sqrt, Rm, Preserve),
        (F64, Sqrt, Rp, Preserve),
        // ex2: approx f32 only (PTX has no ex2.approx.f64). Appended after
        // the original 37 so existing ABI ids stay stable.
        (F32, Ex2, Approx, Preserve),
        (F32, Ex2, Approx, Ftz),
        // tanh: approx f32 only; the instruction has no ftz form (PTX ISA
        // Table 29) and no rounded variants. Hardware floor is sm_75; the
        // family contract gates it at the attested sm_80 evidence floor.
        (F32, Tanh, Approx, Preserve),
        // ex2.approx.f16 starts at PTX 7.0 / sm_75. LLVM 22 rejects the
        // .ftz.f16 spelling; .ftz.bf16 is a separate PTX 7.8 / sm_90 variant.
        (F16, Ex2, Approx, Preserve),
    ]
}

pub(in crate::resolve) fn scalar_math_recipe(
    variant: ScalarMathVariant,
) -> Option<ScalarMathRecipe> {
    if !canonical_scalar_math_variants().contains(&variant) {
        return None;
    }
    let (format, operation, precision, subnormal) = variant;
    let operation_name = scalar_math_operation_name(operation);
    let precision_name = scalar_math_precision_name(precision);
    let format_name = scalar_math_format_name(format);
    let source_format = match format {
        ScalarMathFormat::F16 => "f16",
        ScalarMathFormat::F32 => "f",
        ScalarMathFormat::F64 => "d",
    };

    let mut modifier_names = vec![precision_name];
    if subnormal == ScalarMathSubnormal::Ftz {
        modifier_names.push("ftz");
    }

    let modifier_id = modifier_names.join("_");
    let modifier_symbol = modifier_names.join(".");
    let id = format!("{operation_name}_{modifier_id}_{format_name}");
    let source = match operation {
        // LLVM 22 models ex2 as overloaded records without a per-format
        // suffix (`int_nvvm_ex2_approx{,_ftz}` over anyfloat); bind those
        // for declaration-level checks and record the concrete f32
        // instantiation as the resolved symbol.
        ScalarMathOperation::Ex2 => ScalarMathRecipeSource::ImportedOverloaded {
            source_record: format!("int_nvvm_{operation_name}_{modifier_id}"),
            llvm_symbol: format!("llvm.nvvm.{operation_name}.{modifier_symbol}"),
            resolved_llvm_symbol: format!(
                "llvm.nvvm.{operation_name}.{modifier_symbol}.{format_name}"
            ),
        },
        // LLVM 22.1.2's tblgen export has no record for tanh at all, so the
        // op is admitted directly against the PTX instruction.
        ScalarMathOperation::Tanh => ScalarMathRecipeSource::PtxNative {
            instruction: format!("{operation_name}.{modifier_symbol}.{format_name}"),
        },
        _ => ScalarMathRecipeSource::Imported {
            source_record: format!("int_nvvm_{operation_name}_{modifier_id}_{source_format}"),
            llvm_symbol: format!("llvm.nvvm.{operation_name}.{modifier_symbol}.{source_format}"),
        },
    };
    let ptx_modifiers = modifier_names
        .iter()
        .copied()
        .chain(std::iter::once(format_name))
        .map(str::to_owned)
        .collect::<Vec<_>>();

    // Operations whose llvm.nvvm.* records carry no *imported* DAG selection
    // pattern in the pinned evidence (sel=0). llc itself still selects them
    // (via NVVMIntrinsic-class patterns the tblgen import cannot see), but
    // the evidence-driven contract only admits typed calls backed by an
    // imported selection, so these route through inline PTX for now.
    let force_inline_ptx = matches!(
        operation,
        ScalarMathOperation::Sin
            | ScalarMathOperation::Cos
            | ScalarMathOperation::Ex2
            | ScalarMathOperation::Lg2
            | ScalarMathOperation::Rsqrt
            | ScalarMathOperation::Tanh
    );

    let (ptx_isa_section, ptx_isa_url) = match (operation, format) {
        (ScalarMathOperation::Ex2, ScalarMathFormat::F16) => (
            "9.7.4.10 Half Precision Floating Point Instructions: ex2",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2",
        ),
        (ScalarMathOperation::Sin, _) => (
            "9.7.3.10 Floating Point Instructions: sin",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sin",
        ),
        (ScalarMathOperation::Cos, _) => (
            "9.7.3.11 Floating Point Instructions: cos",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-cos",
        ),
        (ScalarMathOperation::Lg2, _) => (
            "9.7.3.12 Floating Point Instructions: lg2",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-lg2",
        ),
        (ScalarMathOperation::Rcp, _) => (
            "9.7.3.7 Floating Point Instructions: rcp",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp",
        ),
        (ScalarMathOperation::Rsqrt, _) => (
            "9.7.3.14 Floating Point Instructions: rsqrt",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt",
        ),
        (ScalarMathOperation::Sqrt, _) => (
            "9.7.3.9 Floating Point Instructions: sqrt",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt",
        ),
        (ScalarMathOperation::Ex2, _) => (
            "9.7.3.13 Floating Point Instructions: ex2",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2",
        ),
        (ScalarMathOperation::Tanh, _) => (
            "9.7.3.15 Floating Point Instructions: tanh",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh",
        ),
    };

    let (rust_type, dialect_type, minimum_ptx, minimum_sm) = match format {
        ScalarMathFormat::F16 => ("u16", "i16", "7.0", "sm_75"),
        ScalarMathFormat::F32 => ("f32", "f32", "7.0", "sm_80"),
        ScalarMathFormat::F64 => ("f64", "f64", "7.0", "sm_80"),
    };
    Some(ScalarMathRecipe {
        id,
        operation_key: format!("scalar.math.{operation_name}.{modifier_symbol}.{format_name}"),
        source,
        rust_type,
        dialect_type,
        minimum_ptx,
        minimum_sm,
        // LLVM 23 marks the scalar-math NVVM intrinsics
        // IntrNoCreateUndefOrPoison, EXCEPT the ex2/lg2 families which keep
        // the bare IntrNoMem set.
        properties: match operation {
            ScalarMathOperation::Ex2 | ScalarMathOperation::Lg2 => vec!["IntrNoMem"],
            _ => vec!["IntrNoCreateUndefOrPoison", "IntrNoMem"],
        },
        ptx_modifiers,
        ptx_isa_section,
        ptx_isa_url,
        force_inline_ptx,
    })
}

pub(in crate::resolve) fn expand_scalar_math_admission(
    admission: &ScalarMathAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "scalar-math runtime may be marked executed only with GPU evidence"
    );
    let expected = canonical_scalar_math_variants();
    let actual = admission
        .variants
        .iter()
        .map(|variant| {
            (
                variant.format,
                variant.operation,
                variant.precision,
                variant.subnormal,
            )
        })
        .collect::<Vec<_>>();
    ensure!(
        actual == expected,
        "compact scalar-math admission must list the canonical 41 variants"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            validate_abi_id(&variant.abi_id)?;
            let identity = (
                variant.format,
                variant.operation,
                variant.precision,
                variant.subnormal,
            );
            let recipe = scalar_math_recipe(identity)
                .context("scalar math is outside the closed recipe set")?;
            let libnvvm_evidence_profile = variant
                .libnvvm_evidence_profile
                .as_ref()
                .unwrap_or(&admission.libnvvm_evidence_profile);
            ensure!(
                !libnvvm_evidence_profile.trim().is_empty(),
                "scalar-math libNVVM evidence profile must not be empty"
            );
            scalar_math_overlay_record(
                recipe,
                admission,
                identity,
                &variant.abi_id,
                libnvvm_evidence_profile,
            )
        })
        .collect()
}

pub(in crate::resolve) fn scalar_math_overlay_record(
    recipe: ScalarMathRecipe,
    admission: &ScalarMathAdmission,
    variant: ScalarMathVariant,
    abi_id: &str,
    libnvvm_evidence_profile: &str,
) -> Result<OverlayIntrinsic> {
    let (format, operation, precision, subnormal) = variant;
    let ptx_operands = vec![OperandPattern::Register; 2]; // 1 result + 1 operand
    let summary = format!(
        "Computes unary {} with {} precision.",
        scalar_math_operation_name(operation),
        scalar_math_precision_name(precision),
    );
    let (source, source_record, llvm_symbol, resolved_llvm_symbol, llvm_arguments, llvm_results) =
        match &recipe.source {
            ScalarMathRecipeSource::Imported {
                source_record,
                llvm_symbol,
            } => (
                None,
                Some(source_record.clone()),
                Some(llvm_symbol.clone()),
                None,
                vec![recipe.rust_type.to_owned()],
                vec![recipe.rust_type.to_owned()],
            ),
            // The polymorphic signature tokens mirror the imported
            // overloaded record verbatim (anyfloat over anonymous_8), the
            // same shape packed_alu uses for llvm.nvvm.fabs.
            ScalarMathRecipeSource::ImportedOverloaded {
                source_record,
                llvm_symbol,
                resolved_llvm_symbol,
            } => (
                None,
                Some(source_record.clone()),
                Some(llvm_symbol.clone()),
                Some(resolved_llvm_symbol.clone()),
                vec!["anonymous_8".to_owned()],
                vec!["anyfloat".to_owned()],
            ),
            ScalarMathRecipeSource::PtxNative { instruction } => (
                Some(IntrinsicSource::PtxNative {
                    instruction: instruction.clone(),
                }),
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
            ),
        };
    Ok(OverlayIntrinsic {
        id: recipe.id.clone(),
        abi_id: abi_id.into(),
        operation_key: recipe.operation_key,
        family: "scalar_math".into(),
        source,
        source_record,
        rust_module: "float".into(),
        rust_name: recipe.id.clone(),
        rust_arguments: vec![recipe.rust_type.into()],
        rust_result: recipe.rust_type.into(),
        safe: true,
        must_use: true,
        safe_allowlist_reason: Some("Scalar math has no caller obligations.".into()),
        public_rust_path: format!("cuda_intrinsics::float::{}", recipe.id),
        compatibility_rust_paths: vec![format!("cuda_device::float::{}", recipe.id)],
        dialect_op_type: "ScalarMathOp".into(),
        dialect_op_name: "nvvm.scalar_math".into(),
        dialect_operands: vec![recipe.dialect_type.into()],
        dialect_results: vec![recipe.dialect_type.into()],
        llvm_symbol,
        resolved_llvm_symbol,
        llvm_arguments,
        llvm_results,
        pure: true,
        memory: "none".into(),
        convergent: false,
        execution_scope: "thread".into(),
        minimum_ptx: recipe.minimum_ptx.into(),
        minimum_sm: Some(recipe.minimum_sm.into()),
        ptx_result: recipe.rust_type.into(),
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: recipe.ptx_isa_section.into(),
        ptx_isa_url: recipe.ptx_isa_url.into(),
        lowering: "generated_scalar_math".into(),
        backend_lowerings: [
            (
                IntrinsicBackend::LlvmNvptx,
                admission.llvm_evidence_profile.as_str(),
            ),
            (IntrinsicBackend::LibNvvm, libnvvm_evidence_profile),
        ]
        .into_iter()
        .map(
            |(backend, evidence_profile): (IntrinsicBackend, &str)| OverlayBackendLowering {
                backend,
                mechanism: match backend {
                    IntrinsicBackend::LlvmNvptx if recipe.force_inline_ptx => {
                        BackendLoweringMechanism::InlinePtx
                    }
                    IntrinsicBackend::LlvmNvptx => BackendLoweringMechanism::TypedNvvm,
                    IntrinsicBackend::LibNvvm => BackendLoweringMechanism::InlinePtx,
                },
                evidence_profile: evidence_profile.to_owned(),
                targets: None,
                minimum_ptx: Some(recipe.minimum_ptx.into()),
                minimum_sm: Some(recipe.minimum_sm.into()),
            },
        )
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
        scalar_math: Some(ScalarMath {
            format,
            operation,
            precision,
            subnormal,
            runtime_validation: admission.runtime_validation,
        }),
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
            mnemonic: scalar_math_operation_name(operation).into(),
            modifiers: recipe.ptx_modifiers,
            operands: ptx_operands,
        },
        summary,
    })
}

pub(in crate::resolve) fn validate_scalar_math_policy(
    policy: &OverlayIntrinsic,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let math = policy
        .scalar_math
        .as_ref()
        .with_context(|| format!("{} has no scalar-math contract", policy.id))?;
    let variant = (math.format, math.operation, math.precision, math.subnormal);
    let recipe = scalar_math_recipe(variant)
        .with_context(|| format!("{} is outside the closed scalar-math recipe", policy.id))?;
    ensure!(
        math.runtime_validation == RuntimeValidation::Unexecuted,
        "{} scalar-math runtime may be executed only with GPU evidence",
        policy.id
    );
    let signature = vec![recipe.rust_type.to_owned()];
    let source_matches = match &recipe.source {
        ScalarMathRecipeSource::Imported {
            source_record,
            llvm_symbol,
        } => {
            policy.source.is_none()
                && policy.source_record.as_deref() == Some(source_record.as_str())
                && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
                && policy.resolved_llvm_symbol.is_none()
                && policy.llvm_arguments == signature
                && policy.llvm_results == [recipe.rust_type]
        }
        ScalarMathRecipeSource::ImportedOverloaded {
            source_record,
            llvm_symbol,
            resolved_llvm_symbol,
        } => {
            policy.source.is_none()
                && policy.source_record.as_deref() == Some(source_record.as_str())
                && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
                && policy.resolved_llvm_symbol.as_deref() == Some(resolved_llvm_symbol.as_str())
                && policy.llvm_arguments == ["anonymous_8"]
                && policy.llvm_results == ["anyfloat"]
        }
        ScalarMathRecipeSource::PtxNative { instruction } => {
            policy.source
                == Some(IntrinsicSource::PtxNative {
                    instruction: instruction.clone(),
                })
                && policy.source_record.is_none()
                && policy.llvm_symbol.is_none()
                && policy.resolved_llvm_symbol.is_none()
                && policy.llvm_arguments.is_empty()
                && policy.llvm_results.is_empty()
        }
    };
    ensure!(
        policy.id == recipe.id && policy.operation_key == recipe.operation_key && source_matches,
        "{} scalar-math identity or LLVM source changed",
        policy.id
    );
    ensure!(
        declaration.is_none() == matches!(recipe.source, ScalarMathRecipeSource::PtxNative { .. }),
        "{} scalar-math source kind and imported declaration disagree",
        policy.id
    );
    if let Some(declaration) = declaration {
        let expected_properties = recipe
            .properties
            .iter()
            .map(|property| (*property).to_owned())
            .collect::<Vec<_>>();
        ensure!(
            declaration.properties == expected_properties,
            "{} imported scalar-math properties changed",
            policy.id
        );
        ensure!(
            declaration.selections.len() <= 1,
            "{} gained an unreviewed scalar-math selection",
            policy.id
        );
        if let Some(direct) = declaration.selections.first() {
            ensure!(
                direct.predicates.is_empty() && direct.constraints.is_empty(),
                "{} direct scalar-math selection changed",
                policy.id
            );
        }
    }
    ensure!(
        policy.rust_module == "float"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == signature
            && policy.rust_result == recipe.rust_type
            && policy.safe
            && policy.must_use
            && policy.compatibility_rust_paths == [format!("cuda_device::float::{}", recipe.id)]
            && policy.dialect_op_type == "ScalarMathOp"
            && policy.dialect_op_name == "nvvm.scalar_math"
            && policy.dialect_operands == [recipe.dialect_type]
            && policy.dialect_results == [recipe.dialect_type]
            && policy.lowering == "generated_scalar_math",
        "{} changed its scalar-math API, carrier, or lowering",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.ptx_result == recipe.rust_type
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} scalar-math effects, provenance, or target floor changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: scalar_math_operation_name(math.operation).into(),
                modifiers: recipe.ptx_modifiers.clone(),
                operands: vec![OperandPattern::Register; 2],
            },
        "{} expected scalar-math PTX changed",
        policy.id
    );
    let llvm_mechanism = if recipe.force_inline_ptx {
        BackendLoweringMechanism::InlinePtx
    } else {
        BackendLoweringMechanism::TypedNvvm
    };
    let expected_backends = [
        (
            IntrinsicBackend::LlvmNvptx,
            llvm_mechanism,
            recipe.minimum_sm,
        ),
        (
            IntrinsicBackend::LibNvvm,
            BackendLoweringMechanism::InlinePtx,
            recipe.minimum_sm,
        ),
    ];
    ensure!(
        policy.backend_lowerings.len() == 2
            && expected_backends
                .into_iter()
                .all(|(backend, mechanism, minimum_sm)| {
                    policy.backend_lowerings.iter().any(|lowering| {
                        lowering.backend == backend
                            && lowering.mechanism == mechanism
                            && lowering.minimum_ptx.as_deref() == Some(recipe.minimum_ptx)
                            && lowering.minimum_sm.as_deref() == Some(minimum_sm)
                            && !lowering.evidence_profile.trim().is_empty()
                    })
                }),
        "{} has the wrong reviewed scalar-math backend routes (expected {} / {})",
        policy.id,
        recipe.minimum_ptx,
        recipe.minimum_sm
    );
    if let Some(direct) = declaration.and_then(|declaration| declaration.selections.first()) {
        validate_selected_target_predicates(policy, direct)?;
    }
    ensure_no_other_family_contract(policy, "scalar math")?;
    Ok(())
}
