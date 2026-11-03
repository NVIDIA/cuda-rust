/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ExtendedMinMax, ExtendedMinMaxAdapter, ExtendedMinMaxAdmission,
    ExtendedMinMaxFormat, ExtendedMinMaxNan, ExtendedMinMaxOperation, ExtendedMinMaxSubnormal,
    ImportedIntrinsic, IntegerMinMaxFormat, IntegerMinMaxOperation, IntrinsicBackend,
    IntrinsicSource, OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};

use crate::resolve::abi_ledger::*;
use crate::resolve::guards::*;
use crate::resolve::targets::*;

pub(in crate::resolve) type ExtendedMinMaxVariant = (
    ExtendedMinMaxFormat,
    ExtendedMinMaxOperation,
    ExtendedMinMaxSubnormal,
    ExtendedMinMaxNan,
    bool,
);

pub(in crate::resolve) struct ExtendedMinMaxRecipe {
    id: String,
    operation_key: String,
    source_record: String,
    llvm_symbol: String,
    llvm_type: &'static str,
    rust_module: &'static str,
    rust_type: &'static str,
    dialect_type: &'static str,
    adapter: ExtendedMinMaxAdapter,
    selection_record: String,
    selection_asm: String,
    predicates: Vec<String>,
    classes: Vec<&'static str>,
    modifiers: Vec<String>,
    minimum_ptx: &'static str,
    minimum_sm: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
}

pub(in crate::resolve) fn canonical_extended_minmax_variants() -> Vec<ExtendedMinMaxVariant> {
    use ExtendedMinMaxFormat::{Bf16, Bf16x2, F16, F16x2, F32};
    use ExtendedMinMaxNan::{Nan, Number};
    use ExtendedMinMaxOperation::{Max, Min};
    use ExtendedMinMaxSubnormal::{Ftz, Preserve};

    let one_operation = |operation| {
        [
            (F16x2, operation, Ftz, Number, false),
            (F16x2, operation, Ftz, Nan, false),
            (F32, operation, Ftz, Nan, true),
            (F16x2, operation, Ftz, Nan, true),
            (F32, operation, Ftz, Number, true),
            (F16x2, operation, Ftz, Number, true),
            (Bf16x2, operation, Preserve, Nan, false),
            (F16x2, operation, Preserve, Nan, false),
            (Bf16x2, operation, Preserve, Nan, true),
            (F32, operation, Preserve, Nan, true),
            (F16x2, operation, Preserve, Nan, true),
            (Bf16x2, operation, Preserve, Number, true),
            (F32, operation, Preserve, Number, true),
            (F16x2, operation, Preserve, Number, true),
        ]
    };
    // The scalar 16-bit forms. LLVM declares every `f16` modifier combination
    // and the four `bf16` combinations that do not request `ftz`; the four
    // `bf16` `ftz` declarations exist but have no NVPTX selection pattern and
    // fail instruction selection, so they are deliberately absent here.
    //
    // ABI identity is intentionally absent from this canonical list. Existing
    // and future IDs are carried by admission entries and enforced by the
    // global append-only ABI ledger.
    let scalar_halves = |operation| {
        [
            (F16, operation, Preserve, Number, false),
            (F16, operation, Ftz, Number, false),
            (F16, operation, Preserve, Nan, false),
            (F16, operation, Ftz, Nan, false),
            (F16, operation, Preserve, Number, true),
            (F16, operation, Ftz, Number, true),
            (F16, operation, Preserve, Nan, true),
            (F16, operation, Ftz, Nan, true),
            (Bf16, operation, Preserve, Number, false),
            (Bf16, operation, Preserve, Nan, false),
            (Bf16, operation, Preserve, Number, true),
            (Bf16, operation, Preserve, Nan, true),
        ]
    };
    one_operation(Min)
        .into_iter()
        .chain(one_operation(Max))
        .chain(scalar_halves(Min))
        .chain(scalar_halves(Max))
        .collect()
}

/// Joins an operation name, its modifiers, and its format into one identifier.
///
/// `min.f16` and `min.bf16` carry no modifier at all, so the separators have to
/// come from joining the parts that are present rather than from a format
/// string with fixed separators.
pub(in crate::resolve) fn extended_minmax_joined_name(
    leading: &str,
    trailing: &str,
    modifiers: &[&str],
    separator: &str,
) -> String {
    let mut parts = vec![leading];
    parts.extend(modifiers.iter().copied());
    parts.push(trailing);
    parts.join(separator)
}

pub(in crate::resolve) fn extended_minmax_format_name(
    format: ExtendedMinMaxFormat,
) -> &'static str {
    match format {
        ExtendedMinMaxFormat::F32 => "f32",
        ExtendedMinMaxFormat::F16 => "f16",
        ExtendedMinMaxFormat::Bf16 => "bf16",
        ExtendedMinMaxFormat::F16x2 => "f16x2",
        ExtendedMinMaxFormat::Bf16x2 => "bf16x2",
    }
}

pub(in crate::resolve) fn extended_minmax_operation_name(
    operation: ExtendedMinMaxOperation,
) -> &'static str {
    match operation {
        ExtendedMinMaxOperation::Min => "min",
        ExtendedMinMaxOperation::Max => "max",
    }
}

pub(in crate::resolve) fn extended_minmax_recipe(
    variant: ExtendedMinMaxVariant,
) -> Option<ExtendedMinMaxRecipe> {
    if !canonical_extended_minmax_variants().contains(&variant) {
        return None;
    }
    let (format, operation, subnormal, nan, xorsign_abs) = variant;
    let operation_name = extended_minmax_operation_name(operation);
    let format_name = extended_minmax_format_name(format);
    let source_type = if format == ExtendedMinMaxFormat::F32 {
        "f"
    } else {
        format_name
    };
    let mut source_modifiers = Vec::new();
    let mut ptx_modifiers = Vec::new();
    if subnormal == ExtendedMinMaxSubnormal::Ftz {
        source_modifiers.push("ftz");
        ptx_modifiers.push("ftz".to_owned());
    }
    if nan == ExtendedMinMaxNan::Nan {
        source_modifiers.push("nan");
        ptx_modifiers.push("NaN".to_owned());
    }
    if xorsign_abs {
        source_modifiers.extend(["xorsign", "abs"]);
        ptx_modifiers.extend(["xorsign".to_owned(), "abs".to_owned()]);
    }
    let id = extended_minmax_joined_name(operation_name, format_name, &source_modifiers, "_");
    let intrinsic_name = format!("f{operation_name}");
    let source_record = format!(
        "int_nvvm_{}",
        extended_minmax_joined_name(&intrinsic_name, source_type, &source_modifiers, "_")
    );
    let llvm_symbol = format!(
        "llvm.nvvm.{}",
        extended_minmax_joined_name(&intrinsic_name, source_type, &source_modifiers, ".")
    );
    let selection_record = if format == ExtendedMinMaxFormat::F32 {
        format!(
            "INT_NVVM_{}",
            source_record
                .trim_start_matches("int_nvvm_")
                .to_ascii_uppercase()
        )
    } else {
        let prefix = match operation {
            ExtendedMinMaxOperation::Min => "FMIN",
            ExtendedMinMaxOperation::Max => "FMAN",
        };
        let selection_modifiers = source_modifiers
            .iter()
            .map(|modifier| if *modifier == "nan" { "NaN" } else { modifier })
            .collect::<Vec<_>>();
        format!(
            "INT_NVVM_{}",
            extended_minmax_joined_name(prefix, format_name, &selection_modifiers, "_")
        )
    };
    let ptx_format = if format == ExtendedMinMaxFormat::F32 {
        "f32"
    } else {
        format_name
    };
    ptx_modifiers.push(ptx_format.to_owned());
    let selection_asm = format!(
        "{operation_name}.{} \t$dst, $src0, $src1;",
        ptx_modifiers.join(".")
    );
    let (minimum_ptx, minimum_sm, predicates) = if xorsign_abs {
        (
            "7.2",
            "sm_86",
            vec![
                "Subtarget->getPTXVersion() >= 72".to_owned(),
                "Subtarget->getSmVersion() >= 86".to_owned(),
            ],
        )
    } else {
        (
            "7.0",
            "sm_80",
            vec![
                "Subtarget->getSmVersion() >= 80".to_owned(),
                "Subtarget->getPTXVersion() >= 70".to_owned(),
            ],
        )
    };
    let (llvm_type, rust_module, rust_type, dialect_type, adapter, classes) = match format {
        ExtendedMinMaxFormat::F32 => (
            "f32",
            "float",
            "f32",
            "f32",
            ExtendedMinMaxAdapter::DirectF32,
            vec![
                "ClangBuiltin",
                "NVVMBuiltin",
                "SDPatternOperator",
                "Intrinsic",
                "DefaultAttrsIntrinsic",
            ],
        ),
        ExtendedMinMaxFormat::F16 => (
            "f16",
            "f16",
            "u16",
            "i16",
            ExtendedMinMaxAdapter::DirectHalfU16,
            vec!["SDPatternOperator", "Intrinsic", "DefaultAttrsIntrinsic"],
        ),
        ExtendedMinMaxFormat::Bf16 => (
            "bf16",
            "bf16",
            "u16",
            "i16",
            ExtendedMinMaxAdapter::DirectHalfU16,
            vec![
                "ClangBuiltin",
                "NVVMBuiltin",
                "SDPatternOperator",
                "Intrinsic",
                "DefaultAttrsIntrinsic",
            ],
        ),
        ExtendedMinMaxFormat::F16x2 => (
            "v2f16",
            "f16x2",
            "u32",
            "i32",
            ExtendedMinMaxAdapter::DirectPackedU32,
            vec!["SDPatternOperator", "Intrinsic", "DefaultAttrsIntrinsic"],
        ),
        ExtendedMinMaxFormat::Bf16x2 => (
            "v2bf16",
            "bf16x2",
            "u32",
            "i32",
            ExtendedMinMaxAdapter::DirectPackedU32,
            vec![
                "ClangBuiltin",
                "NVVMBuiltin",
                "SDPatternOperator",
                "Intrinsic",
                "DefaultAttrsIntrinsic",
            ],
        ),
    };
    let (ptx_isa_section, ptx_isa_url) = match (format, operation) {
        (ExtendedMinMaxFormat::F32, ExtendedMinMaxOperation::Min) => (
            "9.7.3.9 Floating Point Instructions: min",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min",
        ),
        (ExtendedMinMaxFormat::F32, ExtendedMinMaxOperation::Max) => (
            "9.7.3.10 Floating Point Instructions: max",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max",
        ),
        (_, ExtendedMinMaxOperation::Min) => (
            "9.7.4.7 Half Precision Floating Point Instructions: min",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min",
        ),
        (_, ExtendedMinMaxOperation::Max) => (
            "9.7.4.8 Half Precision Floating Point Instructions: max",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max",
        ),
    };
    Some(ExtendedMinMaxRecipe {
        id,
        operation_key: format!(
            "floating.minmax.{format_name}.{operation_name}.{}.{}.{}",
            match subnormal {
                ExtendedMinMaxSubnormal::Preserve => "preserve",
                ExtendedMinMaxSubnormal::Ftz => "ftz",
            },
            match nan {
                ExtendedMinMaxNan::Number => "number",
                ExtendedMinMaxNan::Nan => "nan",
            },
            if xorsign_abs { "xorsign_abs" } else { "direct" }
        ),
        source_record,
        llvm_symbol,
        llvm_type,
        rust_module,
        rust_type,
        dialect_type,
        adapter,
        selection_record,
        selection_asm,
        predicates,
        classes,
        modifiers: ptx_modifiers,
        minimum_ptx,
        minimum_sm,
        ptx_isa_section,
        ptx_isa_url,
    })
}

pub(in crate::resolve) fn expand_extended_minmax_admission(
    admission: &ExtendedMinMaxAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "extended-minmax runtime may be marked executed only with GPU evidence"
    );
    let expected = canonical_extended_minmax_variants();
    let actual = admission
        .variants
        .iter()
        .map(|variant| {
            (
                variant.format,
                variant.operation,
                variant.subnormal,
                variant.nan,
                variant.xorsign_abs,
            )
        })
        .collect::<Vec<_>>();
    ensure!(
        actual == expected,
        "compact extended-minmax admission must list the exact canonical 28 variants"
    );
    admission
        .variants
        .iter()
        .map(|variant| {
            validate_abi_id(&variant.abi_id)?;
            let identity = (
                variant.format,
                variant.operation,
                variant.subnormal,
                variant.nan,
                variant.xorsign_abs,
            );
            let recipe = extended_minmax_recipe(identity)
                .context("extended min/max is outside the closed recipe set")?;
            extended_minmax_overlay_record(recipe, admission, identity, &variant.abi_id)
        })
        .collect()
}

pub(in crate::resolve) fn extended_minmax_overlay_record(
    recipe: ExtendedMinMaxRecipe,
    admission: &ExtendedMinMaxAdmission,
    variant: ExtendedMinMaxVariant,
    abi_id: &str,
) -> Result<OverlayIntrinsic> {
    let (format, operation, subnormal, nan, xorsign_abs) = variant;
    let rust_arguments = vec![recipe.rust_type.to_owned(); 2];
    let dialect_operands = vec![recipe.dialect_type.to_owned(); 2];
    Ok(OverlayIntrinsic {
        id: recipe.id.clone(),
        abi_id: abi_id.into(),
        operation_key: recipe.operation_key,
        family: "extended_minmax".into(),
        source: None,
        source_record: Some(recipe.source_record),
        rust_module: recipe.rust_module.into(),
        rust_name: recipe.id.clone(),
        rust_arguments: rust_arguments.clone(),
        rust_result: recipe.rust_type.into(),
        safe: true,
        must_use: true,
        safe_allowlist_reason: Some("Floating-point min/max has no caller obligations.".into()),
        public_rust_path: format!("cuda_intrinsics::{}::{}", recipe.rust_module, recipe.id),
        compatibility_rust_paths: vec![format!(
            "cuda_device::{}::{}",
            recipe.rust_module, recipe.id
        )],
        dialect_op_type: "ExtendedMinMaxOp".into(),
        dialect_op_name: "nvvm.extended_minmax".into(),
        dialect_operands: dialect_operands.clone(),
        dialect_results: vec![recipe.dialect_type.into()],
        llvm_symbol: Some(recipe.llvm_symbol),
        resolved_llvm_symbol: None,
        llvm_arguments: vec![recipe.llvm_type.into(); 2],
        llvm_results: vec![recipe.llvm_type.into()],
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
        lowering: "generated_extended_minmax".into(),
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
        extended_minmax: Some(ExtendedMinMax {
            format,
            operation,
            subnormal,
            nan,
            xorsign_abs,
            adapter: recipe.adapter,
            runtime_validation: admission.runtime_validation,
        }),
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
            mnemonic: extended_minmax_operation_name(operation).into(),
            modifiers: recipe.modifiers,
            operands: vec![OperandPattern::Register; 3],
        },
        summary: format!(
            "Computes extended {} for {} values.",
            extended_minmax_operation_name(operation),
            extended_minmax_format_name(format)
        ),
    })
}

pub(in crate::resolve) fn validate_extended_minmax_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let minmax = policy
        .extended_minmax
        .as_ref()
        .with_context(|| format!("{} has no extended-minmax contract", policy.id))?;
    let identity = (
        minmax.format,
        minmax.operation,
        minmax.subnormal,
        minmax.nan,
        minmax.xorsign_abs,
    );
    let recipe = extended_minmax_recipe(identity)
        .with_context(|| format!("{} is outside the closed extended-minmax recipe", policy.id))?;
    ensure!(
        minmax.runtime_validation == RuntimeValidation::Unexecuted,
        "{} extended-minmax runtime may be executed only with GPU evidence",
        policy.id
    );
    let expected_adapter = match minmax.format {
        ExtendedMinMaxFormat::F32 => ExtendedMinMaxAdapter::DirectF32,
        ExtendedMinMaxFormat::F16 | ExtendedMinMaxFormat::Bf16 => {
            ExtendedMinMaxAdapter::DirectHalfU16
        }
        ExtendedMinMaxFormat::F16x2 | ExtendedMinMaxFormat::Bf16x2 => {
            ExtendedMinMaxAdapter::DirectPackedU32
        }
    };
    ensure!(
        minmax.adapter == expected_adapter,
        "{} extended-minmax adapter does not match its format",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && policy.source_record.as_deref() == Some(recipe.source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments == vec![recipe.llvm_type; 2]
            && policy.llvm_results == [recipe.llvm_type],
        "{} extended-minmax identity or LLVM source changed",
        policy.id
    );
    ensure!(
        declaration.classes == recipe.classes
            && declaration.properties
                == [
                    "Commutative",
                    "IntrNoCreateUndefOrPoison",
                    "IntrNoMem",
                    "IntrSpeculatable",
                ]
            && declaration.selections.len() == 1,
        "{} imported extended-minmax classes, properties, or selection count changed",
        policy.id
    );
    let selection = &declaration.selections[0];
    ensure!(
        selection.source_record == recipe.selection_record
            && selection.asm == recipe.selection_asm
            && selection.predicates == recipe.predicates
            && selection.constraints.is_empty(),
        "{} imported extended-minmax instruction selection changed",
        policy.id
    );
    let rust_arguments = vec![recipe.rust_type; 2];
    let dialect_operands = vec![recipe.dialect_type; 2];
    ensure!(
        policy.rust_module == recipe.rust_module
            && policy.rust_name == recipe.id
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == recipe.rust_type
            && policy.safe
            && policy.must_use
            && policy.public_rust_path
                == format!("cuda_intrinsics::{}::{}", recipe.rust_module, recipe.id)
            && policy.compatibility_rust_paths
                == [format!(
                    "cuda_device::{}::{}",
                    recipe.rust_module, recipe.id
                )]
            && policy.dialect_op_type == "ExtendedMinMaxOp"
            && policy.dialect_op_name == "nvvm.extended_minmax"
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results == [recipe.dialect_type]
            && policy.lowering == "generated_extended_minmax",
        "{} changed its extended-minmax API, carrier, or lowering",
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
        "{} extended-minmax effects, provenance, or target floor changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: extended_minmax_operation_name(minmax.operation).into(),
                modifiers: recipe.modifiers,
                operands: vec![OperandPattern::Register; 3],
            },
        "{} expected extended-minmax PTX changed",
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
        "extended-minmax",
    )?;
    validate_selected_target_predicates(policy, selection)?;
    ensure_no_other_family_contract(policy, "extended min/max")?;
    Ok(())
}

pub(in crate::resolve) struct IntegerMinMaxRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    rust_module: &'static str,
    rust_name: &'static str,
    /// Rust and PTX carrier type: `i32` for the scalar forms, `u32` for the
    /// packed pairs.
    scalar: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    ptx_mnemonic: &'static str,
    modifiers: &'static [&'static str],
}

/// Returns the closed recipe for an extended integer min/max variant.
///
/// `None` means the combination is outside the family: `.relu` exists only
/// for the signed formats, and the plain scalar forms are ordinary codegen
/// (`min.s32`/`max.s32` from generic Rust min/max), not intrinsics.
pub(in crate::resolve) fn integer_minmax_recipe(
    format: IntegerMinMaxFormat,
    operation: IntegerMinMaxOperation,
    relu: bool,
) -> Option<IntegerMinMaxRecipe> {
    use IntegerMinMaxFormat as Format;
    use IntegerMinMaxOperation as Operation;
    Some(match (format, operation, relu) {
        (Format::S32, Operation::Min, true) => IntegerMinMaxRecipe {
            id: "min_relu_s32",
            abi_id: "i0987",
            operation_key: "integer.minmax.s32.min.relu",
            rust_module: "int",
            rust_name: "min_relu_s32",
            scalar: "i32",
            dialect_op_type: "MinReluS32Op",
            dialect_op_name: "nvvm.min_relu_s32",
            ptx_mnemonic: "min.relu.s32",
            modifiers: &["relu", "s32"],
        },
        (Format::S32, Operation::Max, true) => IntegerMinMaxRecipe {
            id: "max_relu_s32",
            abi_id: "i0988",
            operation_key: "integer.minmax.s32.max.relu",
            rust_module: "int",
            rust_name: "max_relu_s32",
            scalar: "i32",
            dialect_op_type: "MaxReluS32Op",
            dialect_op_name: "nvvm.max_relu_s32",
            ptx_mnemonic: "max.relu.s32",
            modifiers: &["relu", "s32"],
        },
        (Format::S16x2, Operation::Min, false) => IntegerMinMaxRecipe {
            id: "min_s16x2",
            abi_id: "i0989",
            operation_key: "integer.minmax.s16x2.min",
            rust_module: "i16x2",
            rust_name: "min_s16x2",
            scalar: "u32",
            dialect_op_type: "MinS16x2Op",
            dialect_op_name: "nvvm.min_s16x2",
            ptx_mnemonic: "min.s16x2",
            modifiers: &["s16x2"],
        },
        (Format::S16x2, Operation::Max, false) => IntegerMinMaxRecipe {
            id: "max_s16x2",
            abi_id: "i0990",
            operation_key: "integer.minmax.s16x2.max",
            rust_module: "i16x2",
            rust_name: "max_s16x2",
            scalar: "u32",
            dialect_op_type: "MaxS16x2Op",
            dialect_op_name: "nvvm.max_s16x2",
            ptx_mnemonic: "max.s16x2",
            modifiers: &["s16x2"],
        },
        (Format::U16x2, Operation::Min, false) => IntegerMinMaxRecipe {
            id: "min_u16x2",
            abi_id: "i0991",
            operation_key: "integer.minmax.u16x2.min",
            rust_module: "i16x2",
            rust_name: "min_u16x2",
            scalar: "u32",
            dialect_op_type: "MinU16x2Op",
            dialect_op_name: "nvvm.min_u16x2",
            ptx_mnemonic: "min.u16x2",
            modifiers: &["u16x2"],
        },
        (Format::U16x2, Operation::Max, false) => IntegerMinMaxRecipe {
            id: "max_u16x2",
            abi_id: "i0992",
            operation_key: "integer.minmax.u16x2.max",
            rust_module: "i16x2",
            rust_name: "max_u16x2",
            scalar: "u32",
            dialect_op_type: "MaxU16x2Op",
            dialect_op_name: "nvvm.max_u16x2",
            ptx_mnemonic: "max.u16x2",
            modifiers: &["u16x2"],
        },
        (Format::S16x2, Operation::Min, true) => IntegerMinMaxRecipe {
            id: "min_relu_s16x2",
            abi_id: "i0993",
            operation_key: "integer.minmax.s16x2.min.relu",
            rust_module: "i16x2",
            rust_name: "min_relu_s16x2",
            scalar: "u32",
            dialect_op_type: "MinReluS16x2Op",
            dialect_op_name: "nvvm.min_relu_s16x2",
            ptx_mnemonic: "min.relu.s16x2",
            modifiers: &["relu", "s16x2"],
        },
        (Format::S16x2, Operation::Max, true) => IntegerMinMaxRecipe {
            id: "max_relu_s16x2",
            abi_id: "i0994",
            operation_key: "integer.minmax.s16x2.max.relu",
            rust_module: "i16x2",
            rust_name: "max_relu_s16x2",
            scalar: "u32",
            dialect_op_type: "MaxReluS16x2Op",
            dialect_op_name: "nvvm.max_relu_s16x2",
            ptx_mnemonic: "max.relu.s16x2",
            modifiers: &["relu", "s16x2"],
        },
        (Format::S32, _, false) | (Format::U16x2, _, true) => return None,
    })
}

pub(in crate::resolve) fn integer_minmax_isa_reference(
    operation: IntegerMinMaxOperation,
) -> (&'static str, &'static str) {
    match operation {
        IntegerMinMaxOperation::Min => (
            "9.7.1.13 Integer Arithmetic Instructions: min",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-min",
        ),
        IntegerMinMaxOperation::Max => (
            "9.7.1.14 Integer Arithmetic Instructions: max",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-max",
        ),
    }
}

pub(in crate::resolve) fn validate_integer_minmax_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let minmax = policy
        .integer_minmax
        .as_ref()
        .with_context(|| format!("{} has no closed integer-min/max contract", policy.id))?;
    let recipe = integer_minmax_recipe(minmax.format, minmax.operation, minmax.relu)
        .with_context(|| format!("{} is outside the closed integer-min/max recipe", policy.id))?;
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key,
        "{} integer-min/max identity does not match its closed operation recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == recipe.rust_module
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == [recipe.scalar, recipe.scalar]
            && policy.rust_result == recipe.scalar
            && policy.safe
            && policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            && policy.public_rust_path
                == format!(
                    "cuda_intrinsics::{}::{}",
                    recipe.rust_module, recipe.rust_name
                )
            && policy.compatibility_rust_paths
                == [format!(
                    "cuda_device::{}::{}",
                    recipe.rust_module, recipe.rust_name
                )],
        "{} must preserve its reviewed safe integer-min/max API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", "i32"]
            && policy.dialect_results == ["i32"]
            && policy.lowering == "generated_integer_minmax_inline_ptx",
        "{} is outside the closed integer-min/max dialect and lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "8.0"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.ptx_result == recipe.scalar
            && policy.targets == "all"
            && minmax.native_minimum_sm == 90,
        "{} integer-min/max effects, carrier, or target floor disagree",
        policy.id
    );
    let (isa_section, isa_url) = integer_minmax_isa_reference(minmax.operation);
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == isa_section
            && policy.ptx_isa_url == isa_url,
        "{} integer-min/max PTX provenance does not match its reviewed instruction section",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic
            == recipe.ptx_mnemonic.split('.').next().expect("PTX mnemonic")
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == vec![OperandPattern::Register; 3],
        "{} expected PTX does not match its exact integer-min/max instruction",
        policy.id
    );
    ensure!(
        source
            == &IntrinsicSource::PtxNative {
                instruction: recipe.ptx_mnemonic.to_owned(),
            }
            && declaration.is_none(),
        "{} integer-min/max source does not match its PTX-native recipe",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 1
            && policy.backend_lowerings[0].backend == IntrinsicBackend::LlvmNvptx
            && policy.backend_lowerings[0].mechanism == BackendLoweringMechanism::InlinePtx
            && policy.backend_lowerings[0].targets.is_none()
            && policy.backend_lowerings[0].minimum_ptx.is_none()
            && policy.backend_lowerings[0].minimum_sm.is_none()
            && !policy.backend_lowerings[0]
                .evidence_profile
                .trim()
                .is_empty(),
        "{} integer-min/max backend route changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "integer_minmax")?;
    Ok(())
}
