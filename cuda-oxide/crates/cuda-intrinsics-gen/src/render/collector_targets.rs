/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogFile, CatalogIntrinsic, CatalogSelection, ClusterBarrierMode, ExtendedMinMaxFormat,
    ExtendedMinMaxNan, ExtendedMinMaxOperation, ExtendedMinMaxSubnormal, ImportedAddressSpace,
    IntrinsicBackend, LdmatrixElement, LdmatrixLayout, LdmatrixMultiplicity, LdmatrixShape,
    PackedAtomicFormat, PrmtMode, RegisterMmaAccumulator, RegisterMmaElement, RegisterMmaKind,
    RegisterMmaLayout, RegisterMmaOperation, RegisterMmaOverflow, RegisterMmaShape,
    ScalarArithmeticFormat, ScalarArithmeticOperation, ScalarArithmeticRounding,
    ScalarArithmeticSaturation, ScalarArithmeticSubnormal, ScalarConversionRounding,
    ScalarConversionSaturation, SparseMmaAccumulator, SparseMmaElement, SparseMmaLayout,
    SparseMmaMetadata, SparseMmaOverflow, SparseMmaSelector, SparseMmaShape, WgmmaControlMode,
};
use crate::render::common::{
    generated_hardware_target, intrinsic_marker, rust_header, uses_identifier,
};
use crate::render::families::{
    extended_minmax, ldmatrix, ldmatrix_compat_op, register_mma_effective_kind, scalar_arithmetics,
    scalar_conversions, tcgen05_mma_form_name, tcgen05_mma_intrinsics, wgmma_controls,
};
use crate::render::reference::render_string_patterns;
use std::fmt::Write as _;
use std::path::PathBuf;

pub(super) fn render_collector(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    let abi_namespace = format!("__cuda_oxide_intrinsic_abi_v{}", catalog.intrinsic_abi);
    output.push_str("//! Generated collector predicates for intrinsic placeholder functions.\n\n");
    writeln!(
        output,
        "pub(crate) const GENERATED_INTRINSIC_ABI: u32 = {};",
        catalog.intrinsic_abi
    )
    .unwrap();
    writeln!(
        output,
        "pub(crate) const GENERATED_INTRINSIC_ABI_NAMESPACE: &str = {abi_namespace:?};"
    )
    .unwrap();
    output.push_str(
        "pub(crate) const GENERATED_INTRINSIC_CRATES: &[&str] = &[\n    \"cuda_intrinsics\",\n    \"cuda-intrinsics\",\n];\n\npub(crate) const GENERATED_INTRINSIC_CANONICAL_PATHS: &[&str] = &[\n",
    );
    for record in &catalog.intrinsics {
        writeln!(output, "    {:?},", record.rust.canonical_path).unwrap();
    }
    output.push_str("];\n\npub(crate) const GENERATED_INTRINSIC_PUBLIC_PATHS: &[&str] = &[\n");
    for record in &catalog.intrinsics {
        writeln!(output, "    {:?},", record.rust.public_path).unwrap();
    }
    output.push_str("];\n\npub(crate) fn is_generated_intrinsic_crate(crate_name: &str) -> bool {\n    GENERATED_INTRINSIC_CRATES.contains(&crate_name)\n}\n\npub(crate) fn is_generated_intrinsic_canonical_path(path: &str) -> bool {\n    matches!(path,\n");
    let canonical_paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .map(|record| record.rust.canonical_path.as_str())
        .collect();
    render_string_patterns(&mut output, &canonical_paths, "        ");
    output.push_str("    )\n}\n\npub(crate) fn is_generated_intrinsic_compatibility_path(path: &str) -> bool {\n");
    let compatibility_paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .flat_map(|record| record.rust.compatibility_paths.iter().map(String::as_str))
        .collect();
    if compatibility_paths.is_empty() {
        output.push_str("    let _ = path;\n    false\n");
    } else {
        output.push_str("    matches!(path,\n");
        render_string_patterns(&mut output, &compatibility_paths, "        ");
        output.push_str("    )\n");
    }
    output.push_str(
        "}\n\npub(crate) fn is_generated_intrinsic_placeholder(crate_name: &str, path: &str) -> bool {\n    if is_generated_intrinsic_crate(crate_name) {\n        is_generated_intrinsic_canonical_path(path)\n    } else if matches!(crate_name, \"cuda_device\" | \"cuda-device\") {\n        is_generated_intrinsic_compatibility_path(path)\n    } else {\n        false\n    }\n}\n",
    );
    output
}

fn generated_selection_constraints(selection: &CatalogSelection) -> String {
    let address_space = match selection.constraints.address_space {
        None => "None",
        Some(ImportedAddressSpace::Generic) => "Some(GeneratedSelectionAddressSpace::Generic)",
        Some(ImportedAddressSpace::Shared) => "Some(GeneratedSelectionAddressSpace::Shared)",
    };
    let immediate_bindings = selection
        .constraints
        .immediate_bindings
        .iter()
        .map(|binding| {
            format!(
                "GeneratedImmediateBinding {{ argument_index: {}, value: {} }}",
                binding.argument_index, binding.value
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "GeneratedSelectionConstraints {{ address_space: {address_space}, immediate_bindings: &[{immediate_bindings}] }}"
    )
}

pub(super) fn generated_selection_alternatives(selections: &[CatalogSelection]) -> String {
    let mut output = String::from("&[");
    for selection in selections {
        write!(
            output,
            "GeneratedSelectionAlternative {{ source_record: {:?}, asm: {:?}, predicates: &{:?}, constraints: {} }},",
            selection.source_record,
            selection.asm,
            selection.predicates,
            generated_selection_constraints(selection),
        )
        .unwrap();
    }
    output.push(']');
    output
}

pub(super) fn generated_intrinsic_variant(record: &CatalogIntrinsic) -> String {
    if let Some(mma) = record
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.mma.as_ref())
    {
        return format!(
            "GeneratedIntrinsicVariant::Tcgen05Mma {{ form: GeneratedTcgen05MmaForm::{}, target_selector: GeneratedTcgen05MmaTargetSelector::Kind, compatibility_alias: {} }}",
            tcgen05_mma_form_name(mma.form),
            mma.alias.is_some(),
        );
    }
    if let Some(control) = &record.wgmma_control {
        let mode = match control.mode {
            WgmmaControlMode::Fence => "GeneratedWgmmaControlMode::Fence",
            WgmmaControlMode::CommitGroup => "GeneratedWgmmaControlMode::CommitGroup",
            WgmmaControlMode::WaitGroup => "GeneratedWgmmaControlMode::WaitGroup",
        };
        return format!("GeneratedIntrinsicVariant::WgmmaControl {{ mode: {mode} }}");
    }
    if let Some(barrier) = &record.cluster_barrier {
        let mode = match barrier.mode {
            ClusterBarrierMode::Arrive => "GeneratedClusterBarrierMode::Arrive",
            ClusterBarrierMode::ArriveAligned => "GeneratedClusterBarrierMode::ArriveAligned",
            ClusterBarrierMode::ArriveRelaxed => "GeneratedClusterBarrierMode::ArriveRelaxed",
            ClusterBarrierMode::ArriveRelaxedAligned => {
                "GeneratedClusterBarrierMode::ArriveRelaxedAligned"
            }
            ClusterBarrierMode::Wait => "GeneratedClusterBarrierMode::Wait",
            ClusterBarrierMode::WaitAligned => "GeneratedClusterBarrierMode::WaitAligned",
        };
        return format!("GeneratedIntrinsicVariant::ClusterBarrier {{ mode: {mode} }}");
    }
    if let Some(prmt) = &record.prmt {
        let mode = match prmt.mode {
            PrmtMode::Generic => "GeneratedPrmtMode::Generic",
            PrmtMode::F4e => "GeneratedPrmtMode::F4e",
            PrmtMode::B4e => "GeneratedPrmtMode::B4e",
            PrmtMode::Rc8 => "GeneratedPrmtMode::Rc8",
            PrmtMode::Ecl => "GeneratedPrmtMode::Ecl",
            PrmtMode::Ecr => "GeneratedPrmtMode::Ecr",
            PrmtMode::Rc16 => "GeneratedPrmtMode::Rc16",
        };
        return format!("GeneratedIntrinsicVariant::Prmt {{ mode: {mode} }}");
    }
    if let Some(packed) = &record.packed_atomic {
        let format = match packed.format {
            PackedAtomicFormat::F16x2 => "GeneratedPackedAtomicFormat::F16x2",
            PackedAtomicFormat::Bf16x2 => "GeneratedPackedAtomicFormat::Bf16x2",
        };
        return format!("GeneratedIntrinsicVariant::PackedAtomic {{ format: {format} }}");
    }
    if let Some(mma) = &record.register_mma {
        let shape = match mma.shape {
            RegisterMmaShape::M8n8k4 => "GeneratedRegisterMmaShape::M8n8k4",
            RegisterMmaShape::M8n8k16 => "GeneratedRegisterMmaShape::M8n8k16",
            RegisterMmaShape::M8n8k32 => "GeneratedRegisterMmaShape::M8n8k32",
            RegisterMmaShape::M8n8k128 => "GeneratedRegisterMmaShape::M8n8k128",
            RegisterMmaShape::M16n8k4 => "GeneratedRegisterMmaShape::M16n8k4",
            RegisterMmaShape::M16n8k8 => "GeneratedRegisterMmaShape::M16n8k8",
            RegisterMmaShape::M16n8k16 => "GeneratedRegisterMmaShape::M16n8k16",
            RegisterMmaShape::M16n8k32 => "GeneratedRegisterMmaShape::M16n8k32",
            RegisterMmaShape::M16n8k64 => "GeneratedRegisterMmaShape::M16n8k64",
            RegisterMmaShape::M16n8k128 => "GeneratedRegisterMmaShape::M16n8k128",
            RegisterMmaShape::M16n8k256 => "GeneratedRegisterMmaShape::M16n8k256",
        };
        let operation = match mma.operation {
            RegisterMmaOperation::Multiply => "GeneratedRegisterMmaOperation::Multiply",
            RegisterMmaOperation::AndPopc => "GeneratedRegisterMmaOperation::AndPopc",
            RegisterMmaOperation::XorPopc => "GeneratedRegisterMmaOperation::XorPopc",
        };
        let kind = match register_mma_effective_kind(record) {
            RegisterMmaKind::Standard => "GeneratedRegisterMmaKind::Standard",
            RegisterMmaKind::F8f6f4 => "GeneratedRegisterMmaKind::F8f6f4",
            RegisterMmaKind::Mxf8f6f4 => "GeneratedRegisterMmaKind::Mxf8f6f4",
        };
        let accumulator = match mma.accumulator {
            RegisterMmaAccumulator::F16 => "GeneratedRegisterMmaAccumulator::F16",
            RegisterMmaAccumulator::F32 => "GeneratedRegisterMmaAccumulator::F32",
            RegisterMmaAccumulator::F64 => "GeneratedRegisterMmaAccumulator::F64",
            RegisterMmaAccumulator::S32 => "GeneratedRegisterMmaAccumulator::S32",
        };
        let element = |element| match element {
            RegisterMmaElement::Bf16 => "GeneratedRegisterMmaElement::Bf16",
            RegisterMmaElement::E2m1 => "GeneratedRegisterMmaElement::E2m1",
            RegisterMmaElement::E2m3 => "GeneratedRegisterMmaElement::E2m3",
            RegisterMmaElement::E3m2 => "GeneratedRegisterMmaElement::E3m2",
            RegisterMmaElement::E4m3 => "GeneratedRegisterMmaElement::E4m3",
            RegisterMmaElement::E5m2 => "GeneratedRegisterMmaElement::E5m2",
            RegisterMmaElement::F16 => "GeneratedRegisterMmaElement::F16",
            RegisterMmaElement::Tf32 => "GeneratedRegisterMmaElement::Tf32",
            RegisterMmaElement::F64 => "GeneratedRegisterMmaElement::F64",
            RegisterMmaElement::B1 => "GeneratedRegisterMmaElement::B1",
            RegisterMmaElement::S4 => "GeneratedRegisterMmaElement::S4",
            RegisterMmaElement::U4 => "GeneratedRegisterMmaElement::U4",
            RegisterMmaElement::S8 => "GeneratedRegisterMmaElement::S8",
            RegisterMmaElement::U8 => "GeneratedRegisterMmaElement::U8",
        };
        let layout = |layout| match layout {
            RegisterMmaLayout::Row => "GeneratedRegisterMmaLayout::Row",
            RegisterMmaLayout::Col => "GeneratedRegisterMmaLayout::Col",
        };
        let overflow = match mma.overflow {
            RegisterMmaOverflow::NotApplicable => "GeneratedRegisterMmaOverflow::NotApplicable",
            RegisterMmaOverflow::Wrapping => "GeneratedRegisterMmaOverflow::Wrapping",
            RegisterMmaOverflow::Satfinite => "GeneratedRegisterMmaOverflow::Satfinite",
        };
        return format!(
            "GeneratedIntrinsicVariant::RegisterMma {{ shape: {shape}, operation: {operation}, kind: {kind}, accumulator: {accumulator}, a_element: {}, b_element: {}, a_layout: {}, b_layout: {}, overflow: {overflow} }}",
            element(mma.a_element),
            element(mma.b_element),
            layout(mma.a_layout),
            layout(mma.b_layout),
        );
    }
    if let Some(mma) = &record.sparse_mma {
        let shape = match mma.shape {
            SparseMmaShape::M16n8k8 => "GeneratedSparseMmaShape::M16n8k8",
            SparseMmaShape::M16n8k16 => "GeneratedSparseMmaShape::M16n8k16",
            SparseMmaShape::M16n8k32 => "GeneratedSparseMmaShape::M16n8k32",
            SparseMmaShape::M16n8k64 => "GeneratedSparseMmaShape::M16n8k64",
            SparseMmaShape::M16n8k128 => "GeneratedSparseMmaShape::M16n8k128",
        };
        let accumulator = match mma.accumulator {
            SparseMmaAccumulator::F16 => "GeneratedSparseMmaAccumulator::F16",
            SparseMmaAccumulator::F32 => "GeneratedSparseMmaAccumulator::F32",
            SparseMmaAccumulator::S32 => "GeneratedSparseMmaAccumulator::S32",
        };
        let element = |element| match element {
            SparseMmaElement::F16 => "GeneratedSparseMmaElement::F16",
            SparseMmaElement::Bf16 => "GeneratedSparseMmaElement::Bf16",
            SparseMmaElement::Tf32 => "GeneratedSparseMmaElement::Tf32",
            SparseMmaElement::E2m1 => "GeneratedSparseMmaElement::E2m1",
            SparseMmaElement::E2m3 => "GeneratedSparseMmaElement::E2m3",
            SparseMmaElement::E3m2 => "GeneratedSparseMmaElement::E3m2",
            SparseMmaElement::E4m3 => "GeneratedSparseMmaElement::E4m3",
            SparseMmaElement::E5m2 => "GeneratedSparseMmaElement::E5m2",
            SparseMmaElement::S4 => "GeneratedSparseMmaElement::S4",
            SparseMmaElement::U4 => "GeneratedSparseMmaElement::U4",
            SparseMmaElement::S8 => "GeneratedSparseMmaElement::S8",
            SparseMmaElement::U8 => "GeneratedSparseMmaElement::U8",
        };
        let layout = |layout| match layout {
            SparseMmaLayout::Row => "GeneratedSparseMmaLayout::Row",
            SparseMmaLayout::Col => "GeneratedSparseMmaLayout::Col",
        };
        let overflow = match mma.overflow {
            SparseMmaOverflow::NotApplicable => "GeneratedSparseMmaOverflow::NotApplicable",
            SparseMmaOverflow::Wrapping => "GeneratedSparseMmaOverflow::Wrapping",
            SparseMmaOverflow::Satfinite => "GeneratedSparseMmaOverflow::Satfinite",
        };
        let metadata = match mma.metadata {
            SparseMmaMetadata::Standard => "GeneratedSparseMmaMetadata::Standard",
            SparseMmaMetadata::Ordered => "GeneratedSparseMmaMetadata::Ordered",
        };
        let selector = match mma.selector {
            SparseMmaSelector::ImmediateZeroThroughThree => {
                "GeneratedSparseMmaSelector::ImmediateZeroThroughThree"
            }
            SparseMmaSelector::ImmediateZeroOrOne => {
                "GeneratedSparseMmaSelector::ImmediateZeroOrOne"
            }
            SparseMmaSelector::ImmediateZero => "GeneratedSparseMmaSelector::ImmediateZero",
        };
        return format!(
            "GeneratedIntrinsicVariant::SparseMma {{ shape: {shape}, accumulator: {accumulator}, a_element: {}, b_element: {}, a_layout: {}, b_layout: {}, overflow: {overflow}, metadata: {metadata}, selector: {selector} }}",
            element(mma.a_element),
            element(mma.b_element),
            layout(mma.a_layout),
            layout(mma.b_layout),
        );
    }
    if let Some(arithmetic) = &record.scalar_arithmetic {
        let format = match arithmetic.format {
            ScalarArithmeticFormat::F32 => "GeneratedScalarArithmeticFormat::F32",
            ScalarArithmeticFormat::F64 => "GeneratedScalarArithmeticFormat::F64",
        };
        let operation = match arithmetic.operation {
            ScalarArithmeticOperation::Mul => "GeneratedScalarArithmeticOperation::Mul",
            ScalarArithmeticOperation::Div => "GeneratedScalarArithmeticOperation::Div",
            ScalarArithmeticOperation::Fma => "GeneratedScalarArithmeticOperation::Fma",
            ScalarArithmeticOperation::Add => "GeneratedScalarArithmeticOperation::Add",
        };
        let rounding = match arithmetic.rounding {
            ScalarArithmeticRounding::Rn => "GeneratedScalarArithmeticRounding::Rn",
            ScalarArithmeticRounding::Rz => "GeneratedScalarArithmeticRounding::Rz",
            ScalarArithmeticRounding::Rm => "GeneratedScalarArithmeticRounding::Rm",
            ScalarArithmeticRounding::Rp => "GeneratedScalarArithmeticRounding::Rp",
        };
        let subnormal = match arithmetic.subnormal {
            ScalarArithmeticSubnormal::Preserve => "GeneratedScalarArithmeticSubnormal::Preserve",
            ScalarArithmeticSubnormal::Ftz => "GeneratedScalarArithmeticSubnormal::Ftz",
        };
        let saturation = match arithmetic.saturation {
            ScalarArithmeticSaturation::None => "GeneratedScalarArithmeticSaturation::None",
            ScalarArithmeticSaturation::Sat => "GeneratedScalarArithmeticSaturation::Sat",
        };
        return format!(
            "GeneratedIntrinsicVariant::ScalarArithmetic {{ format: {format}, operation: {operation}, rounding: {rounding}, subnormal: {subnormal}, saturation: {saturation} }}"
        );
    }
    if let Some(minmax) = &record.extended_minmax {
        let format = match minmax.format {
            ExtendedMinMaxFormat::F32 => "GeneratedExtendedMinMaxFormat::F32",
            ExtendedMinMaxFormat::F16 => "GeneratedExtendedMinMaxFormat::F16",
            ExtendedMinMaxFormat::Bf16 => "GeneratedExtendedMinMaxFormat::Bf16",
            ExtendedMinMaxFormat::F16x2 => "GeneratedExtendedMinMaxFormat::F16x2",
            ExtendedMinMaxFormat::Bf16x2 => "GeneratedExtendedMinMaxFormat::Bf16x2",
        };
        let operation = match minmax.operation {
            ExtendedMinMaxOperation::Min => "GeneratedExtendedMinMaxOperation::Min",
            ExtendedMinMaxOperation::Max => "GeneratedExtendedMinMaxOperation::Max",
        };
        let subnormal = match minmax.subnormal {
            ExtendedMinMaxSubnormal::Preserve => "GeneratedExtendedMinMaxSubnormal::Preserve",
            ExtendedMinMaxSubnormal::Ftz => "GeneratedExtendedMinMaxSubnormal::Ftz",
        };
        let nan = match minmax.nan {
            ExtendedMinMaxNan::Number => "GeneratedExtendedMinMaxNan::Number",
            ExtendedMinMaxNan::Nan => "GeneratedExtendedMinMaxNan::Nan",
        };
        return format!(
            "GeneratedIntrinsicVariant::ExtendedMinMax {{ format: {format}, operation: {operation}, subnormal: {subnormal}, nan: {nan}, xorsign_abs: {} }}",
            minmax.xorsign_abs
        );
    }
    if let Some(conversion) = &record.scalar_conversion {
        let rounding = match conversion.rounding {
            ScalarConversionRounding::NearestAway => {
                "GeneratedScalarConversionRounding::NearestAway"
            }
            ScalarConversionRounding::NearestEven => {
                "GeneratedScalarConversionRounding::NearestEven"
            }
            ScalarConversionRounding::TowardZero => "GeneratedScalarConversionRounding::TowardZero",
        };
        let saturation = match conversion.saturation {
            ScalarConversionSaturation::None => "GeneratedScalarConversionSaturation::None",
            ScalarConversionSaturation::Relu => "GeneratedScalarConversionSaturation::Relu",
            ScalarConversionSaturation::Satfinite => {
                "GeneratedScalarConversionSaturation::Satfinite"
            }
            ScalarConversionSaturation::ReluSatfinite => {
                "GeneratedScalarConversionSaturation::ReluSatfinite"
            }
        };
        return format!(
            "GeneratedIntrinsicVariant::ScalarConversion {{ rounding: {rounding}, saturation: {saturation} }}"
        );
    }
    let Some(ldmatrix) = &record.ldmatrix else {
        return "GeneratedIntrinsicVariant::Scalar".to_owned();
    };
    let variant = &ldmatrix.variant;
    let shape = match variant.shape {
        LdmatrixShape::M8n8 => "GeneratedLdmatrixShape::M8n8",
        LdmatrixShape::M8n16 => "GeneratedLdmatrixShape::M8n16",
        LdmatrixShape::M16n16 => "GeneratedLdmatrixShape::M16n16",
    };
    let multiplicity = match variant.multiplicity {
        LdmatrixMultiplicity::X1 => "GeneratedLdmatrixMultiplicity::X1",
        LdmatrixMultiplicity::X2 => "GeneratedLdmatrixMultiplicity::X2",
        LdmatrixMultiplicity::X4 => "GeneratedLdmatrixMultiplicity::X4",
    };
    let layout = match variant.layout {
        LdmatrixLayout::Normal => "GeneratedLdmatrixLayout::Normal",
        LdmatrixLayout::Transposed => "GeneratedLdmatrixLayout::Transposed",
    };
    let element = match variant.element {
        LdmatrixElement::B16 => "GeneratedLdmatrixElement::B16",
        LdmatrixElement::B8 => "GeneratedLdmatrixElement::B8",
        LdmatrixElement::B8x16B4x16P64 => "GeneratedLdmatrixElement::B8x16B4x16P64",
        LdmatrixElement::B8x16B6x16P32 => "GeneratedLdmatrixElement::B8x16B6x16P32",
    };
    format!(
        "GeneratedIntrinsicVariant::Ldmatrix {{ shape: {shape}, multiplicity: {multiplicity}, layout: {layout}, element: {element} }}"
    )
}

fn generated_backend_requirements(record: &CatalogIntrinsic) -> String {
    let requirements = record
        .backend_lowerings
        .iter()
        .map(|lowering| {
            let backend = match lowering.backend {
                IntrinsicBackend::LlvmNvptx => "GeneratedIntrinsicBackend::LlvmNvptx",
                IntrinsicBackend::LibNvvm => "GeneratedIntrinsicBackend::LibNvvm",
            };
            format!(
                "GeneratedBackendRequirement {{ backend: {backend}, requirement: GeneratedTargetRequirement {{ minimum_ptx: GeneratedPtxVersion::from_encoded({}), hardware: {} }} }}",
                lowering.target.minimum_ptx.encoded(),
                generated_hardware_target(&lowering.target.hardware),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{requirements}]")
}

fn replace_exact_render_fragment(output: &mut String, fragment: &str, replacement: &str) {
    assert_eq!(
        output.matches(fragment).count(),
        1,
        "render fragment must occur exactly once"
    );
    let start = output.find(fragment).expect("checked render fragment");
    output.replace_range(start..start + fragment.len(), replacement);
}

fn render_target_record(output: &mut String, catalog: &CatalogFile, record: &CatalogIntrinsic) {
    let llvm_facts = match &record.llvm {
        Some(llvm) => {
            let result_range = match &llvm.result_facts.range {
                Some(range) => format!(
                    "Some(GeneratedIntrinsicRange {{ lower: {:?}, upper_exclusive: {:?} }})",
                    range.lower, range.upper_exclusive
                ),
                None => "None".to_owned(),
            };
            format!(
                "Some(GeneratedLlvmFacts {{ properties: &{:?}, result_no_undef: {}, result_range: {} }})",
                llvm.properties, llvm.result_facts.no_undef, result_range
            )
        }
        None => "None".to_owned(),
    };
    writeln!(
            output,
            "    GeneratedIntrinsicTarget {{ marker: {:?}, id: {:?}, abi_id: {:?}, dialect_op: {:?}, variant: {}, requirement: GeneratedTargetRequirement {{ minimum_ptx: GeneratedPtxVersion::from_encoded({}), hardware: {} }}, backend_requirements: {}, selections: {}, llvm: {} }},",
            intrinsic_marker(catalog, record),
            record.id,
            record.rust.abi_id,
            record.dialect.op_name,
            generated_intrinsic_variant(record),
            record.target.minimum_ptx.encoded(),
            generated_hardware_target(&record.target.hardware),
            generated_backend_requirements(record),
            generated_selection_alternatives(&record.selections),
            llvm_facts,
        )
        .unwrap();
}

fn render_target_record_assertions(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    writeln!(
        output,
        "        let target = generated_intrinsic_target_by_marker({:?}).unwrap();",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    writeln!(
            output,
            "        assert_eq!(target.id, {:?});\n        assert_eq!(target.abi_id, {:?});\n        assert_eq!(target.dialect_op, {:?});\n        assert_eq!(target.variant, {});\n        assert_eq!(target.requirement.minimum_ptx.encoded(), {});\n        assert_eq!(target.requirement.hardware, const {{ {} }});\n        assert_eq!(target.backend_requirements, const {{ {} }});\n        assert_eq!(target.selections, {});",
            record.id,
            record.rust.abi_id,
            record.dialect.op_name,
            generated_intrinsic_variant(record),
            record.target.minimum_ptx.encoded(),
            generated_hardware_target(&record.target.hardware),
            generated_backend_requirements(record),
            generated_selection_alternatives(&record.selections),
        )
        .unwrap();
    match &record.llvm {
        Some(llvm) => {
            writeln!(
                output,
                "        assert_eq!(target.llvm.unwrap().properties, &{:?} as &[&str]);",
                llvm.properties
            )
            .unwrap();
        }
        None => output.push_str("        assert!(target.llvm.is_none());\n"),
    }
}

fn targets_mod_file(
    catalog: &CatalogFile,
    hash: &str,
    groups: &[(&'static str, Vec<&CatalogIntrinsic>)],
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated target requirements and separately imported LLVM/selection facts.\n\npub const GENERATED_INTRINSIC_MARKER_ATTR: &str = \"cuda_oxide_intrinsic_marker\";\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]\npub struct GeneratedPtxVersion(u16);\nimpl GeneratedPtxVersion {\n    pub const fn from_encoded(encoded: u16) -> Self { Self(encoded) }\n    pub const fn encoded(self) -> u16 { self.0 }\n    pub const fn major(self) -> u16 { self.0 / 10 }\n    pub const fn minor(self) -> u16 { self.0 % 10 }\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareAlternative { MinimumSm(u16), ExactArchitecture(u16), FamilyTarget(u16) }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetSelectorBinding { pub name: &'static str, pub value: &'static str }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetAlternative { pub minimum_ptx: GeneratedPtxVersion, pub hardware: GeneratedHardwareAlternative }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareTarget { All, AnyOf(&'static [GeneratedHardwareAlternative]), TargetMatrix { selectors: &'static [GeneratedTargetSelectorBinding], alternatives: &'static [GeneratedTargetAlternative] } }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetRequirement { pub minimum_ptx: GeneratedPtxVersion, pub hardware: GeneratedHardwareTarget }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicBackend { LlvmNvptx, LibNvvm }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedBackendRequirement { pub backend: GeneratedIntrinsicBackend, pub requirement: GeneratedTargetRequirement }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSelectionAddressSpace { Generic, Shared }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedImmediateBinding { pub argument_index: usize, pub value: i64 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedSelectionConstraints { pub address_space: Option<GeneratedSelectionAddressSpace>, pub immediate_bindings: &'static [GeneratedImmediateBinding] }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedSelectionAlternative { pub source_record: &'static str, pub asm: &'static str, pub predicates: &'static [&'static str], pub constraints: GeneratedSelectionConstraints }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicRange { pub lower: &'static str, pub upper_exclusive: &'static str }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedLlvmFacts { pub properties: &'static [&'static str], pub result_no_undef: bool, pub result_range: Option<GeneratedIntrinsicRange> }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixShape { M8n8 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixMultiplicity { X1, X2, X4 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixLayout { Normal, Transposed }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedPackedAtomicFormat { F16x2, Bf16x2 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedRegisterMmaShape { M8n8k4, M16n8k8, M16n8k16, M16n8k32 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedRegisterMmaAccumulator { F32, F64, S32 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedRegisterMmaElement { Bf16, F16, Tf32, F64, S8, U8 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedRegisterMmaLayout { Row, Col }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedRegisterMmaOverflow { NotApplicable, Wrapping, Satfinite }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {\n    Scalar,\n    Ldmatrix { shape: GeneratedLdmatrixShape, multiplicity: GeneratedLdmatrixMultiplicity, layout: GeneratedLdmatrixLayout },\n    PackedAtomic { format: GeneratedPackedAtomicFormat },\n    RegisterMma { shape: GeneratedRegisterMmaShape, accumulator: GeneratedRegisterMmaAccumulator, a_element: GeneratedRegisterMmaElement, b_element: GeneratedRegisterMmaElement, a_layout: GeneratedRegisterMmaLayout, b_layout: GeneratedRegisterMmaLayout, overflow: GeneratedRegisterMmaOverflow },\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicTarget {\n    pub marker: &'static str,\n    pub id: &'static str,\n    pub abi_id: &'static str,\n    pub dialect_op: &'static str,\n    pub variant: GeneratedIntrinsicVariant,\n    pub requirement: GeneratedTargetRequirement,\n    pub backend_requirements: &'static [GeneratedBackendRequirement],\n    pub selections: &'static [GeneratedSelectionAlternative],\n    pub llvm: Option<GeneratedLlvmFacts>,\n}\n\nimpl GeneratedIntrinsicTarget {\n    pub fn requirement_for_backend(&self, backend: GeneratedIntrinsicBackend) -> GeneratedTargetRequirement {\n        self.backend_requirements.iter().find(|entry| entry.backend == backend).map(|entry| entry.requirement).unwrap_or(self.requirement)\n    }\n}\n",
    );
    replace_exact_render_fragment(
        &mut output,
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetAlternative { pub minimum_ptx: GeneratedPtxVersion, pub hardware: GeneratedHardwareAlternative }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareTarget { All, AnyOf(&'static [GeneratedHardwareAlternative]), TargetMatrix { selectors: &'static [GeneratedTargetSelectorBinding], alternatives: &'static [GeneratedTargetAlternative] } }",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetAlternative { pub minimum_ptx: GeneratedPtxVersion, pub hardware: GeneratedHardwareAlternative }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetContract { pub selectors: &'static [GeneratedTargetSelectorBinding], pub alternatives: &'static [GeneratedTargetAlternative] }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareTarget { All, AnyOf(&'static [GeneratedHardwareAlternative]), TargetMatrix { contracts: &'static [GeneratedTargetContract] } }",
    );
    replace_exact_render_fragment(
        &mut output,
        "impl GeneratedIntrinsicTarget {\n    pub fn requirement_for_backend(&self, backend: GeneratedIntrinsicBackend) -> GeneratedTargetRequirement {\n        self.backend_requirements.iter().find(|entry| entry.backend == backend).map(|entry| entry.requirement).unwrap_or(self.requirement)\n    }\n}",
        r#"impl GeneratedHardwareTarget {
    pub fn contract_for_selector(
        self,
        name: &str,
        value: &str,
    ) -> Option<&'static GeneratedTargetContract> {
        let Self::TargetMatrix { contracts } = self else { return None; };
        let mut matching = contracts.iter().filter(|contract| {
            matches!(contract.selectors, [binding] if binding.name == name && binding.value == value)
        });
        let contract = matching.next()?;
        if matching.next().is_some() { None } else { Some(contract) }
    }
}

impl GeneratedIntrinsicTarget {
    pub fn requirement_for_backend(&self, backend: GeneratedIntrinsicBackend) -> GeneratedTargetRequirement {
        self.backend_requirements.iter().find(|entry| entry.backend == backend).map(|entry| entry.requirement).unwrap_or(self.requirement)
    }

    pub fn target_contract_for_backend_selector(
        &self,
        backend: GeneratedIntrinsicBackend,
        name: &str,
        value: &str,
    ) -> Option<&'static GeneratedTargetContract> {
        self.requirement_for_backend(backend).hardware.contract_for_selector(name, value)
    }
}"#,
    );
    replace_exact_render_fragment(
        &mut output,
        "GeneratedLdmatrixShape { M8n8 }",
        "GeneratedLdmatrixShape { M8n8, M8n16, M16n16 }",
    );
    replace_exact_render_fragment(
        &mut output,
        "pub enum GeneratedPackedAtomicFormat { F16x2, Bf16x2 }",
        "pub enum GeneratedLdmatrixElement { B16, B8, B8x16B4x16P64, B8x16B6x16P32 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedPackedAtomicFormat { F16x2, Bf16x2 }",
    );
    replace_exact_render_fragment(
        &mut output,
        "Ldmatrix { shape: GeneratedLdmatrixShape, multiplicity: GeneratedLdmatrixMultiplicity, layout: GeneratedLdmatrixLayout }",
        "Ldmatrix { shape: GeneratedLdmatrixShape, multiplicity: GeneratedLdmatrixMultiplicity, layout: GeneratedLdmatrixLayout, element: GeneratedLdmatrixElement }",
    );
    replace_exact_render_fragment(
        &mut output,
        "GeneratedRegisterMmaShape { M8n8k4, M16n8k8, M16n8k16, M16n8k32 }",
        "GeneratedRegisterMmaShape { M8n8k4, M8n8k16, M8n8k32, M8n8k128, M16n8k4, M16n8k8, M16n8k16, M16n8k32, M16n8k64, M16n8k128, M16n8k256 }",
    );
    replace_exact_render_fragment(
        &mut output,
        "GeneratedRegisterMmaElement { Bf16, F16, Tf32, F64, S8, U8 }",
        "GeneratedRegisterMmaElement { Bf16, E2m1, E2m3, E3m2, E4m3, E5m2, F16, Tf32, F64, B1, S4, U4, S8, U8 }",
    );
    replace_exact_render_fragment(
        &mut output,
        "pub enum GeneratedRegisterMmaAccumulator { F32, F64, S32 }",
        "pub enum GeneratedRegisterMmaOperation { Multiply, AndPopc, XorPopc }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedRegisterMmaKind { Standard, F8f6f4, Mxf8f6f4 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedRegisterMmaAccumulator { F16, F32, F64, S32 }",
    );
    replace_exact_render_fragment(
        &mut output,
        "RegisterMma { shape: GeneratedRegisterMmaShape, accumulator: GeneratedRegisterMmaAccumulator,",
        "RegisterMma { shape: GeneratedRegisterMmaShape, operation: GeneratedRegisterMmaOperation, kind: GeneratedRegisterMmaKind, accumulator: GeneratedRegisterMmaAccumulator,",
    );
    replace_exact_render_fragment(
        &mut output,
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSparseMmaShape { M16n8k8, M16n8k16, M16n8k32, M16n8k64, M16n8k128 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSparseMmaAccumulator { F16, F32, S32 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSparseMmaElement { F16, Bf16, Tf32, E2m1, E2m3, E3m2, E4m3, E5m2, S4, U4, S8, U8 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSparseMmaLayout { Row, Col }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSparseMmaOverflow { NotApplicable, Wrapping, Satfinite }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSparseMmaMetadata { Standard, Ordered }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n#[allow(clippy::enum_variant_names)]\npub enum GeneratedSparseMmaSelector { ImmediateZeroThroughThree, ImmediateZeroOrOne, ImmediateZero }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedPrmtMode { Generic, F4e, B4e, Rc8, Ecl, Ecr, Rc16 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedClusterBarrierMode { Arrive, ArriveAligned, ArriveRelaxed, ArriveRelaxedAligned, Wait, WaitAligned }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedScalarConversionRounding { NearestAway, NearestEven, TowardZero }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedScalarConversionSaturation { None, Relu, Satfinite, ReluSatfinite }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
    );
    replace_exact_render_fragment(
        &mut output,
        "    RegisterMma { shape: GeneratedRegisterMmaShape, operation: GeneratedRegisterMmaOperation, kind: GeneratedRegisterMmaKind, accumulator: GeneratedRegisterMmaAccumulator, a_element: GeneratedRegisterMmaElement, b_element: GeneratedRegisterMmaElement, a_layout: GeneratedRegisterMmaLayout, b_layout: GeneratedRegisterMmaLayout, overflow: GeneratedRegisterMmaOverflow },\n}",
        "    RegisterMma { shape: GeneratedRegisterMmaShape, operation: GeneratedRegisterMmaOperation, kind: GeneratedRegisterMmaKind, accumulator: GeneratedRegisterMmaAccumulator, a_element: GeneratedRegisterMmaElement, b_element: GeneratedRegisterMmaElement, a_layout: GeneratedRegisterMmaLayout, b_layout: GeneratedRegisterMmaLayout, overflow: GeneratedRegisterMmaOverflow },\n    SparseMma { shape: GeneratedSparseMmaShape, accumulator: GeneratedSparseMmaAccumulator, a_element: GeneratedSparseMmaElement, b_element: GeneratedSparseMmaElement, a_layout: GeneratedSparseMmaLayout, b_layout: GeneratedSparseMmaLayout, overflow: GeneratedSparseMmaOverflow, metadata: GeneratedSparseMmaMetadata, selector: GeneratedSparseMmaSelector },\n    Prmt { mode: GeneratedPrmtMode },\n    ClusterBarrier { mode: GeneratedClusterBarrierMode },\n    ScalarConversion { rounding: GeneratedScalarConversionRounding, saturation: GeneratedScalarConversionSaturation },\n}",
    );
    if wgmma_controls(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedWgmmaControlMode { Fence, CommitGroup, WaitGroup }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
        );
        replace_exact_render_fragment(
            &mut output,
            "    ClusterBarrier { mode: GeneratedClusterBarrierMode },\n    ScalarConversion { rounding: GeneratedScalarConversionRounding, saturation: GeneratedScalarConversionSaturation },\n}",
            "    ClusterBarrier { mode: GeneratedClusterBarrierMode },\n    ScalarConversion { rounding: GeneratedScalarConversionRounding, saturation: GeneratedScalarConversionSaturation },\n    WgmmaControl { mode: GeneratedWgmmaControlMode },\n}",
        );
    }
    if scalar_arithmetics(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedScalarArithmeticFormat { F32, F64 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedScalarArithmeticOperation { Mul, Div, Fma, Add }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedScalarArithmeticRounding { Rn, Rz, Rm, Rp }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedScalarArithmeticSubnormal { Preserve, Ftz }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedScalarArithmeticSaturation { None, Sat }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
        );
        if wgmma_controls(catalog).next().is_some() {
            replace_exact_render_fragment(
                &mut output,
                "    WgmmaControl { mode: GeneratedWgmmaControlMode },\n}",
                "    WgmmaControl { mode: GeneratedWgmmaControlMode },\n    ScalarArithmetic { format: GeneratedScalarArithmeticFormat, operation: GeneratedScalarArithmeticOperation, rounding: GeneratedScalarArithmeticRounding, subnormal: GeneratedScalarArithmeticSubnormal, saturation: GeneratedScalarArithmeticSaturation },\n}",
            );
        } else {
            replace_exact_render_fragment(
                &mut output,
                "    ScalarConversion { rounding: GeneratedScalarConversionRounding, saturation: GeneratedScalarConversionSaturation },\n}",
                "    ScalarConversion { rounding: GeneratedScalarConversionRounding, saturation: GeneratedScalarConversionSaturation },\n    ScalarArithmetic { format: GeneratedScalarArithmeticFormat, operation: GeneratedScalarArithmeticOperation, rounding: GeneratedScalarArithmeticRounding, subnormal: GeneratedScalarArithmeticSubnormal, saturation: GeneratedScalarArithmeticSaturation },\n}",
            );
        }
    }
    if extended_minmax(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedExtendedMinMaxFormat { F32, F16, Bf16, F16x2, Bf16x2 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedExtendedMinMaxOperation { Min, Max }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedExtendedMinMaxSubnormal { Preserve, Ftz }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedExtendedMinMaxNan { Number, Nan }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
        );
        replace_exact_render_fragment(
            &mut output,
            "}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicTarget",
            "    ExtendedMinMax { format: GeneratedExtendedMinMaxFormat, operation: GeneratedExtendedMinMaxOperation, subnormal: GeneratedExtendedMinMaxSubnormal, nan: GeneratedExtendedMinMaxNan, xorsign_abs: bool },\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicTarget",
        );
    }
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedTcgen05MmaForm { Shared, Tensor, TensorAshift, SpShared, SpTensor, SpTensorAshift, WsShared, WsSharedZeroColMask, WsSpShared, WsSpSharedZeroColMask, WsSpTensor, WsSpTensorZeroColMask, WsTensor, WsTensorZeroColMask }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedTcgen05MmaTargetSelector { Kind }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {",
        );
        replace_exact_render_fragment(
            &mut output,
            "}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicTarget",
            "    Tcgen05Mma { form: GeneratedTcgen05MmaForm, target_selector: GeneratedTcgen05MmaTargetSelector, compatibility_alias: bool },\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicTarget",
        );
    }
    output.push_str(
        "pub const GENERATED_INTRINSIC_TARGET_GROUPS: &[&[GeneratedIntrinsicTarget]] = &[\n",
    );
    for (shard, _) in groups {
        writeln!(output, "    {shard}::TARGETS,").unwrap();
    }
    output.push_str(
        "];\n\npub fn generated_intrinsic_targets() -> impl Iterator<Item = &'static GeneratedIntrinsicTarget> {\n    GENERATED_INTRINSIC_TARGET_GROUPS.iter().flat_map(|group| group.iter())\n}\n\n",
    );
    output.push_str(
        "pub fn generated_intrinsic_target_by_marker(marker: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    generated_intrinsic_targets().find(|target| target.marker == marker)\n}\n\npub fn generated_intrinsic_targets_by_op_name(op_name: &str) -> impl Iterator<Item = &'static GeneratedIntrinsicTarget> + '_ {\n    generated_intrinsic_targets().filter(move |target| target.dialect_op == op_name)\n}\n\npub fn generated_intrinsic_target_by_op_name(op_name: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    generated_intrinsic_targets_by_op_name(op_name).next()\n}\n\npub fn generated_intrinsic_target(op_name: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    generated_intrinsic_target_by_op_name(op_name)\n}\n\npub fn generated_intrinsic_operation_matches(ctx: &Context, target: &GeneratedIntrinsicTarget, operation: Ptr<Operation>) -> bool {\n    match target.variant {\n        GeneratedIntrinsicVariant::Scalar => true,\n        GeneratedIntrinsicVariant::Ldmatrix { shape, multiplicity, layout } => {\n            let Some(op) = Operation::get_op::<LdmatrixOp>(operation, ctx) else { return false; };\n            let shape_matches = matches!(shape, GeneratedLdmatrixShape::M8n8) && op.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() == Some(&LdmatrixShapeAttr::M8n8);\n            let multiplicity_matches = match multiplicity {\n                GeneratedLdmatrixMultiplicity::X1 => op.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref() == Some(&LdmatrixMultiplicityAttr::X1),\n                GeneratedLdmatrixMultiplicity::X2 => op.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref() == Some(&LdmatrixMultiplicityAttr::X2),\n                GeneratedLdmatrixMultiplicity::X4 => op.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref() == Some(&LdmatrixMultiplicityAttr::X4),\n            };\n            let layout_matches = match layout {\n                GeneratedLdmatrixLayout::Normal => op.get_attr_nvvm_ldmatrix_layout(ctx).as_deref() == Some(&LdmatrixLayoutAttr::Normal),\n                GeneratedLdmatrixLayout::Transposed => op.get_attr_nvvm_ldmatrix_layout(ctx).as_deref() == Some(&LdmatrixLayoutAttr::Transposed),\n            };\n            shape_matches && multiplicity_matches && layout_matches\n                && op.get_attr_nvvm_ldmatrix_element(ctx).as_deref() == Some(&LdmatrixElementAttr::B16)\n                && op.get_attr_nvvm_ldmatrix_state_space(ctx).as_deref() == Some(&LdmatrixStateSpaceAttr::Shared)\n        }\n        GeneratedIntrinsicVariant::PackedAtomic { format } => {\n            let Some(op) = Operation::get_op::<PackedAtomicAddOp>(operation, ctx) else { return false; };\n            let format_matches = match format {\n                GeneratedPackedAtomicFormat::F16x2 => op.get_attr_nvvm_packed_atomic_format(ctx).as_deref() == Some(&PackedAtomicFormatAttr::F16x2),\n                GeneratedPackedAtomicFormat::Bf16x2 => op.get_attr_nvvm_packed_atomic_format(ctx).as_deref() == Some(&PackedAtomicFormatAttr::Bf16x2),\n            };\n            format_matches\n                && op.get_attr_nvvm_packed_atomic_state_space(ctx).as_deref() == Some(&PackedAtomicStateSpaceAttr::Global)\n                && op.get_attr_nvvm_packed_atomic_ordering(ctx).as_deref() == Some(&PackedAtomicOrderingAttr::Relaxed)\n                && op.get_attr_nvvm_packed_atomic_scope(ctx).as_deref() == Some(&PackedAtomicScopeAttr::Gpu)\n                && op.get_attr_nvvm_packed_atomic_rounding(ctx).as_deref() == Some(&PackedAtomicRoundingAttr::Rn)\n                && op.get_attr_nvvm_packed_atomic_subnormal(ctx).as_deref() == Some(&PackedAtomicSubnormalAttr::NoFtz)\n                && op.get_attr_nvvm_packed_atomic_atomicity(ctx).as_deref() == Some(&PackedAtomicAtomicityAttr::PerElement)\n        }\n        GeneratedIntrinsicVariant::RegisterMma { shape, accumulator, a_element, b_element, a_layout, b_layout, overflow } => {\n            let Some(op) = Operation::get_op::<RegisterMmaOp>(operation, ctx) else { return false; };\n            let shape_matches = match shape {\n                GeneratedRegisterMmaShape::M8n8k4 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k4),\n                GeneratedRegisterMmaShape::M16n8k8 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k8),\n                GeneratedRegisterMmaShape::M16n8k16 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k16),\n                GeneratedRegisterMmaShape::M16n8k32 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k32),\n            };\n            let accumulator_matches = match accumulator {\n                GeneratedRegisterMmaAccumulator::F32 => op.get_attr_nvvm_register_mma_accumulator(ctx).as_deref() == Some(&RegisterMmaAccumulatorAttr::F32),\n                GeneratedRegisterMmaAccumulator::F64 => op.get_attr_nvvm_register_mma_accumulator(ctx).as_deref() == Some(&RegisterMmaAccumulatorAttr::F64),\n                GeneratedRegisterMmaAccumulator::S32 => op.get_attr_nvvm_register_mma_accumulator(ctx).as_deref() == Some(&RegisterMmaAccumulatorAttr::S32),\n            };\n            let element_matches = |expected, actual: Option<&RegisterMmaElementAttr>| match expected {\n                GeneratedRegisterMmaElement::Bf16 => actual == Some(&RegisterMmaElementAttr::Bf16),\n                GeneratedRegisterMmaElement::F16 => actual == Some(&RegisterMmaElementAttr::F16),\n                GeneratedRegisterMmaElement::Tf32 => actual == Some(&RegisterMmaElementAttr::Tf32),\n                GeneratedRegisterMmaElement::F64 => actual == Some(&RegisterMmaElementAttr::F64),\n                GeneratedRegisterMmaElement::S8 => actual == Some(&RegisterMmaElementAttr::S8),\n                GeneratedRegisterMmaElement::U8 => actual == Some(&RegisterMmaElementAttr::U8),\n            };\n            let layout_matches = |expected, actual: Option<&RegisterMmaLayoutAttr>| match expected {\n                GeneratedRegisterMmaLayout::Row => actual == Some(&RegisterMmaLayoutAttr::Row),\n                GeneratedRegisterMmaLayout::Col => actual == Some(&RegisterMmaLayoutAttr::Col),\n            };\n            let overflow_matches = match overflow {\n                GeneratedRegisterMmaOverflow::NotApplicable => op.get_attr_nvvm_register_mma_overflow(ctx).as_deref() == Some(&RegisterMmaOverflowAttr::NotApplicable),\n                GeneratedRegisterMmaOverflow::Wrapping => op.get_attr_nvvm_register_mma_overflow(ctx).as_deref() == Some(&RegisterMmaOverflowAttr::Wrapping),\n                GeneratedRegisterMmaOverflow::Satfinite => op.get_attr_nvvm_register_mma_overflow(ctx).as_deref() == Some(&RegisterMmaOverflowAttr::Satfinite),\n            };\n            shape_matches && accumulator_matches\n                && element_matches(a_element, op.get_attr_nvvm_register_mma_a_element(ctx).as_deref())\n                && element_matches(b_element, op.get_attr_nvvm_register_mma_b_element(ctx).as_deref())\n                && layout_matches(a_layout, op.get_attr_nvvm_register_mma_a_layout(ctx).as_deref())\n                && layout_matches(b_layout, op.get_attr_nvvm_register_mma_b_layout(ctx).as_deref())\n                && overflow_matches\n        }\n    }\n}\n",
    );
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "pub fn generated_intrinsic_target(op_name: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    generated_intrinsic_target_by_op_name(op_name)\n}\n\npub fn generated_intrinsic_operation_matches",
            "pub fn generated_intrinsic_target(op_name: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    generated_intrinsic_target_by_op_name(op_name)\n}\n\npub fn generated_intrinsic_target_is_direct_dialect_candidate(target: &GeneratedIntrinsicTarget) -> bool {\n    !matches!(target.variant, GeneratedIntrinsicVariant::Tcgen05Mma { compatibility_alias: true, .. })\n}\n\npub fn generated_intrinsic_operation_matches",
        );
        replace_exact_render_fragment(
            &mut output,
            "        GeneratedIntrinsicVariant::Scalar => true,",
            r#"        GeneratedIntrinsicVariant::Scalar => true,
        GeneratedIntrinsicVariant::Tcgen05Mma { form, target_selector, compatibility_alias } => {
            let Some(op) = Operation::get_op::<Tcgen05MmaOp>(operation, ctx) else { return false; };
            let form_matches = match form {
                GeneratedTcgen05MmaForm::Shared => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::Shared),
                GeneratedTcgen05MmaForm::Tensor => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::Tensor),
                GeneratedTcgen05MmaForm::TensorAshift => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::TensorAshift),
                GeneratedTcgen05MmaForm::SpShared => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::SpShared),
                GeneratedTcgen05MmaForm::SpTensor => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::SpTensor),
                GeneratedTcgen05MmaForm::SpTensorAshift => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::SpTensorAshift),
                GeneratedTcgen05MmaForm::WsShared => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsShared),
                GeneratedTcgen05MmaForm::WsSharedZeroColMask => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsSharedZeroColMask),
                GeneratedTcgen05MmaForm::WsSpShared => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsSpShared),
                GeneratedTcgen05MmaForm::WsSpSharedZeroColMask => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsSpSharedZeroColMask),
                GeneratedTcgen05MmaForm::WsSpTensor => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsSpTensor),
                GeneratedTcgen05MmaForm::WsSpTensorZeroColMask => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsSpTensorZeroColMask),
                GeneratedTcgen05MmaForm::WsTensor => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsTensor),
                GeneratedTcgen05MmaForm::WsTensorZeroColMask => op.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() == Some(&Tcgen05MmaFormAttr::WsTensorZeroColMask),
            };
            let selector_matches = matches!(target_selector, GeneratedTcgen05MmaTargetSelector::Kind)
                && op.get_attr_nvvm_tcgen05_mma_kind(ctx).is_some();
            let alias_matches = !compatibility_alias || (
                op.get_attr_nvvm_tcgen05_mma_kind(ctx).as_deref() == Some(&Tcgen05MmaKindAttr::F8f6f4)
                    && match form {
                        GeneratedTcgen05MmaForm::Shared => {
                            op.get_attr_nvvm_tcgen05_mma_cta_group(ctx).as_deref() == Some(&Tcgen05MmaCtaGroupAttr::Cg1)
                                && op.get_attr_nvvm_tcgen05_mma_collector_a(ctx).as_deref() == Some(&Tcgen05MmaCollectorAAttr::Discard)
                        }
                        GeneratedTcgen05MmaForm::WsTensor => {
                            op.get_attr_nvvm_tcgen05_mma_b_buffer(ctx).as_deref() == Some(&Tcgen05MmaBBufferAttr::B0)
                                && op.get_attr_nvvm_tcgen05_mma_b_usage(ctx).as_deref() == Some(&Tcgen05MmaBUsageAttr::Discard)
                        }
                        _ => false,
                    }
            );
            form_matches && selector_matches && alias_matches
        }"#,
        );
    }
    replace_exact_render_fragment(
        &mut output,
        "GeneratedIntrinsicVariant::Ldmatrix { shape, multiplicity, layout } => {",
        "GeneratedIntrinsicVariant::Ldmatrix { shape, multiplicity, layout, element } => {",
    );
    replace_exact_render_fragment(
        &mut output,
        "            let shape_matches = matches!(shape, GeneratedLdmatrixShape::M8n8) && op.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() == Some(&LdmatrixShapeAttr::M8n8);",
        "            let shape_matches = match shape {\n                GeneratedLdmatrixShape::M8n8 => op.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() == Some(&LdmatrixShapeAttr::M8n8),\n                GeneratedLdmatrixShape::M8n16 => op.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() == Some(&LdmatrixShapeAttr::M8n16),\n                GeneratedLdmatrixShape::M16n16 => op.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() == Some(&LdmatrixShapeAttr::M16n16),\n            };",
    );
    replace_exact_render_fragment(
        &mut output,
        "            shape_matches && multiplicity_matches && layout_matches\n                && op.get_attr_nvvm_ldmatrix_element(ctx).as_deref() == Some(&LdmatrixElementAttr::B16)",
        "            let element_matches = match element {\n                GeneratedLdmatrixElement::B16 => op.get_attr_nvvm_ldmatrix_element(ctx).as_deref() == Some(&LdmatrixElementAttr::B16),\n                GeneratedLdmatrixElement::B8 => op.get_attr_nvvm_ldmatrix_element(ctx).as_deref() == Some(&LdmatrixElementAttr::B8),\n                GeneratedLdmatrixElement::B8x16B4x16P64 => op.get_attr_nvvm_ldmatrix_element(ctx).as_deref() == Some(&LdmatrixElementAttr::B8x16B4x16P64),\n                GeneratedLdmatrixElement::B8x16B6x16P32 => op.get_attr_nvvm_ldmatrix_element(ctx).as_deref() == Some(&LdmatrixElementAttr::B8x16B6x16P32),\n            };\n            shape_matches && multiplicity_matches && layout_matches && element_matches",
    );
    let mut compatibility_aliases = String::from(
        "fn generated_intrinsic_compatibility_marker_by_op_name(op_name: &str) -> Option<&'static str> {\n    match op_name {\n",
    );
    let mut compatibility_count = 0;
    for record in ldmatrix(catalog) {
        let Some((_, op_name)) = ldmatrix_compat_op(record) else {
            continue;
        };
        compatibility_count += 1;
        writeln!(
            compatibility_aliases,
            "        {op_name:?} => Some({:?}),",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
    }
    assert_eq!(compatibility_count, 6);
    compatibility_aliases.push_str("        _ => None,\n    }\n}\n\n");
    replace_exact_render_fragment(
        &mut output,
        "pub fn generated_intrinsic_target_by_marker(marker: &str)",
        &format!(
            "{compatibility_aliases}pub fn generated_intrinsic_target_by_marker(marker: &str)"
        ),
    );
    replace_exact_render_fragment(
        &mut output,
        "    generated_intrinsic_targets().filter(move |target| target.dialect_op == op_name)\n",
        "    let compatibility_marker = generated_intrinsic_compatibility_marker_by_op_name(op_name);\n    generated_intrinsic_targets().filter(move |target| {\n        target.dialect_op == op_name || compatibility_marker == Some(target.marker)\n    })\n",
    );
    replace_exact_render_fragment(
        &mut output,
        "        GeneratedIntrinsicVariant::Ldmatrix { shape, multiplicity, layout, element } => {\n            let Some(op) = Operation::get_op::<LdmatrixOp>(operation, ctx) else { return false; };",
        "        GeneratedIntrinsicVariant::Ldmatrix { shape, multiplicity, layout, element } => {\n            let operation_name = Operation::get_opid(operation, ctx).to_string();\n            if let Some(marker) = generated_intrinsic_compatibility_marker_by_op_name(&operation_name) {\n                return marker == target.marker;\n            }\n            let Some(op) = Operation::get_op::<LdmatrixOp>(operation, ctx) else { return false; };",
    );
    replace_exact_render_fragment(
        &mut output,
        "GeneratedIntrinsicVariant::RegisterMma { shape, accumulator,",
        "GeneratedIntrinsicVariant::RegisterMma { shape, operation: mma_operation, kind, accumulator,",
    );
    replace_exact_render_fragment(
        &mut output,
        "            };\n            let accumulator_matches = match accumulator {",
        "            };\n            let operation_matches = match mma_operation {\n                GeneratedRegisterMmaOperation::Multiply => op.operation_or_multiply(ctx) == RegisterMmaOperationAttr::Multiply,\n                GeneratedRegisterMmaOperation::AndPopc => op.operation_or_multiply(ctx) == RegisterMmaOperationAttr::AndPopc,\n                GeneratedRegisterMmaOperation::XorPopc => op.operation_or_multiply(ctx) == RegisterMmaOperationAttr::XorPopc,\n            };\n            let kind_matches = match kind {\n                GeneratedRegisterMmaKind::Standard => op.kind_or_inferred(ctx) == RegisterMmaKindAttr::Standard,\n                GeneratedRegisterMmaKind::F8f6f4 => op.kind_or_inferred(ctx) == RegisterMmaKindAttr::F8f6f4,\n                GeneratedRegisterMmaKind::Mxf8f6f4 => op.kind_or_inferred(ctx) == RegisterMmaKindAttr::Mxf8f6f4,\n            };\n            let accumulator_matches = match accumulator {",
    );
    replace_exact_render_fragment(
        &mut output,
        "            let accumulator_matches = match accumulator {\n                GeneratedRegisterMmaAccumulator::F32",
        "            let accumulator_matches = match accumulator {\n                GeneratedRegisterMmaAccumulator::F16 => op.get_attr_nvvm_register_mma_accumulator(ctx).as_deref() == Some(&RegisterMmaAccumulatorAttr::F16),\n                GeneratedRegisterMmaAccumulator::F32",
    );
    replace_exact_render_fragment(
        &mut output,
        "            shape_matches && accumulator_matches",
        "            shape_matches && operation_matches && kind_matches && accumulator_matches",
    );
    replace_exact_render_fragment(
        &mut output,
        "GeneratedRegisterMmaShape::M8n8k4 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k4),\n                GeneratedRegisterMmaShape::M16n8k8",
        "GeneratedRegisterMmaShape::M8n8k4 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k4),\n                GeneratedRegisterMmaShape::M8n8k16 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k16),\n                GeneratedRegisterMmaShape::M8n8k32 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k32),\n                GeneratedRegisterMmaShape::M8n8k128 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M8n8k128),\n                GeneratedRegisterMmaShape::M16n8k4 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k4),\n                GeneratedRegisterMmaShape::M16n8k8",
    );
    replace_exact_render_fragment(
        &mut output,
        "GeneratedRegisterMmaShape::M16n8k32 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k32),\n            };",
        "GeneratedRegisterMmaShape::M16n8k32 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k32),\n                GeneratedRegisterMmaShape::M16n8k64 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k64),\n                GeneratedRegisterMmaShape::M16n8k128 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k128),\n                GeneratedRegisterMmaShape::M16n8k256 => op.get_attr_nvvm_register_mma_shape(ctx).as_deref() == Some(&RegisterMmaShapeAttr::M16n8k256),\n            };",
    );
    replace_exact_render_fragment(
        &mut output,
        "GeneratedRegisterMmaElement::F64 => actual == Some(&RegisterMmaElementAttr::F64),\n                GeneratedRegisterMmaElement::S8",
        "GeneratedRegisterMmaElement::F64 => actual == Some(&RegisterMmaElementAttr::F64),\n                GeneratedRegisterMmaElement::E2m1 => actual == Some(&RegisterMmaElementAttr::E2m1),\n                GeneratedRegisterMmaElement::E2m3 => actual == Some(&RegisterMmaElementAttr::E2m3),\n                GeneratedRegisterMmaElement::E3m2 => actual == Some(&RegisterMmaElementAttr::E3m2),\n                GeneratedRegisterMmaElement::E4m3 => actual == Some(&RegisterMmaElementAttr::E4m3),\n                GeneratedRegisterMmaElement::E5m2 => actual == Some(&RegisterMmaElementAttr::E5m2),\n                GeneratedRegisterMmaElement::B1 => actual == Some(&RegisterMmaElementAttr::B1),\n                GeneratedRegisterMmaElement::S4 => actual == Some(&RegisterMmaElementAttr::S4),\n                GeneratedRegisterMmaElement::U4 => actual == Some(&RegisterMmaElementAttr::U4),\n                GeneratedRegisterMmaElement::S8",
    );
    replace_exact_render_fragment(
        &mut output,
        "                && overflow_matches\n        }\n    }\n}\n",
        "                && overflow_matches\n        }\n        GeneratedIntrinsicVariant::SparseMma { shape, accumulator, a_element, b_element, a_layout, b_layout, overflow, metadata, selector } => {\n            let Some(op) = Operation::get_op::<SparseMmaOp>(operation, ctx) else { return false; };\n            let element_matches = |expected, actual: Option<&SparseMmaElementAttr>| match expected {\n                GeneratedSparseMmaElement::F16 => actual == Some(&SparseMmaElementAttr::F16),\n                GeneratedSparseMmaElement::Bf16 => actual == Some(&SparseMmaElementAttr::Bf16),\n                GeneratedSparseMmaElement::Tf32 => actual == Some(&SparseMmaElementAttr::Tf32),\n                GeneratedSparseMmaElement::E2m1 => actual == Some(&SparseMmaElementAttr::E2m1),\n                GeneratedSparseMmaElement::E2m3 => actual == Some(&SparseMmaElementAttr::E2m3),\n                GeneratedSparseMmaElement::E3m2 => actual == Some(&SparseMmaElementAttr::E3m2),\n                GeneratedSparseMmaElement::E4m3 => actual == Some(&SparseMmaElementAttr::E4m3),\n                GeneratedSparseMmaElement::E5m2 => actual == Some(&SparseMmaElementAttr::E5m2),\n                GeneratedSparseMmaElement::S4 => actual == Some(&SparseMmaElementAttr::S4),\n                GeneratedSparseMmaElement::U4 => actual == Some(&SparseMmaElementAttr::U4),\n                GeneratedSparseMmaElement::S8 => actual == Some(&SparseMmaElementAttr::S8),\n                GeneratedSparseMmaElement::U8 => actual == Some(&SparseMmaElementAttr::U8),\n            };\n            let layout_matches = |expected, actual: Option<&SparseMmaLayoutAttr>| match expected {\n                GeneratedSparseMmaLayout::Row => actual == Some(&SparseMmaLayoutAttr::Row),\n                GeneratedSparseMmaLayout::Col => actual == Some(&SparseMmaLayoutAttr::Col),\n            };\n            let overflow_matches = match overflow {\n                GeneratedSparseMmaOverflow::NotApplicable => op.get_attr_nvvm_sparse_mma_overflow(ctx).as_deref() == Some(&SparseMmaOverflowAttr::NotApplicable),\n                GeneratedSparseMmaOverflow::Wrapping => op.get_attr_nvvm_sparse_mma_overflow(ctx).as_deref() == Some(&SparseMmaOverflowAttr::Wrapping),\n                GeneratedSparseMmaOverflow::Satfinite => op.get_attr_nvvm_sparse_mma_overflow(ctx).as_deref() == Some(&SparseMmaOverflowAttr::Satfinite),\n            };\n            let metadata_matches = match metadata {\n                GeneratedSparseMmaMetadata::Standard => op.get_attr_nvvm_sparse_mma_metadata(ctx).as_deref() == Some(&SparseMmaMetadataAttr::Standard),\n                GeneratedSparseMmaMetadata::Ordered => op.get_attr_nvvm_sparse_mma_metadata(ctx).as_deref() == Some(&SparseMmaMetadataAttr::Ordered),\n            };\n            matches!(shape, GeneratedSparseMmaShape::M16n8k32)\n                && op.get_attr_nvvm_sparse_mma_shape(ctx).as_deref() == Some(&SparseMmaShapeAttr::M16n8k32)\n                && matches!(accumulator, GeneratedSparseMmaAccumulator::S32)\n                && op.get_attr_nvvm_sparse_mma_accumulator(ctx).as_deref() == Some(&SparseMmaAccumulatorAttr::S32)\n                && element_matches(a_element, op.get_attr_nvvm_sparse_mma_a_element(ctx).as_deref())\n                && element_matches(b_element, op.get_attr_nvvm_sparse_mma_b_element(ctx).as_deref())\n                && layout_matches(a_layout, op.get_attr_nvvm_sparse_mma_a_layout(ctx).as_deref())\n                && layout_matches(b_layout, op.get_attr_nvvm_sparse_mma_b_layout(ctx).as_deref())\n                && overflow_matches\n                && metadata_matches\n                && matches!(selector, GeneratedSparseMmaSelector::ImmediateZeroOrOne)\n                && op.get_attr_nvvm_sparse_mma_selector(ctx).as_deref() == Some(&SparseMmaSelectorAttr::ImmediateZeroOrOne)\n        }\n    }\n}\n",
    );
    replace_exact_render_fragment(
        &mut output,
        "            matches!(shape, GeneratedSparseMmaShape::M16n8k32)\n                && op.get_attr_nvvm_sparse_mma_shape(ctx).as_deref() == Some(&SparseMmaShapeAttr::M16n8k32)\n                && matches!(accumulator, GeneratedSparseMmaAccumulator::S32)\n                && op.get_attr_nvvm_sparse_mma_accumulator(ctx).as_deref() == Some(&SparseMmaAccumulatorAttr::S32)",
        "            let shape_matches = match shape {\n                GeneratedSparseMmaShape::M16n8k8 => op.get_attr_nvvm_sparse_mma_shape(ctx).as_deref() == Some(&SparseMmaShapeAttr::M16n8k8),\n                GeneratedSparseMmaShape::M16n8k16 => op.get_attr_nvvm_sparse_mma_shape(ctx).as_deref() == Some(&SparseMmaShapeAttr::M16n8k16),\n                GeneratedSparseMmaShape::M16n8k32 => op.get_attr_nvvm_sparse_mma_shape(ctx).as_deref() == Some(&SparseMmaShapeAttr::M16n8k32),\n                GeneratedSparseMmaShape::M16n8k64 => op.get_attr_nvvm_sparse_mma_shape(ctx).as_deref() == Some(&SparseMmaShapeAttr::M16n8k64),\n                GeneratedSparseMmaShape::M16n8k128 => op.get_attr_nvvm_sparse_mma_shape(ctx).as_deref() == Some(&SparseMmaShapeAttr::M16n8k128),\n            };\n            let accumulator_matches = match accumulator {\n                GeneratedSparseMmaAccumulator::F16 => op.get_attr_nvvm_sparse_mma_accumulator(ctx).as_deref() == Some(&SparseMmaAccumulatorAttr::F16),\n                GeneratedSparseMmaAccumulator::F32 => op.get_attr_nvvm_sparse_mma_accumulator(ctx).as_deref() == Some(&SparseMmaAccumulatorAttr::F32),\n                GeneratedSparseMmaAccumulator::S32 => op.get_attr_nvvm_sparse_mma_accumulator(ctx).as_deref() == Some(&SparseMmaAccumulatorAttr::S32),\n            };\n            let selector_matches = match selector {\n                GeneratedSparseMmaSelector::ImmediateZeroThroughThree => op.get_attr_nvvm_sparse_mma_selector(ctx).as_deref() == Some(&SparseMmaSelectorAttr::ImmediateZeroThroughThree),\n                GeneratedSparseMmaSelector::ImmediateZeroOrOne => op.get_attr_nvvm_sparse_mma_selector(ctx).as_deref() == Some(&SparseMmaSelectorAttr::ImmediateZeroOrOne),\n                GeneratedSparseMmaSelector::ImmediateZero => op.get_attr_nvvm_sparse_mma_selector(ctx).as_deref() == Some(&SparseMmaSelectorAttr::ImmediateZero),\n            };\n            shape_matches\n                && accumulator_matches",
    );
    replace_exact_render_fragment(
        &mut output,
        "                && metadata_matches\n                && matches!(selector, GeneratedSparseMmaSelector::ImmediateZeroOrOne)\n                && op.get_attr_nvvm_sparse_mma_selector(ctx).as_deref() == Some(&SparseMmaSelectorAttr::ImmediateZeroOrOne)",
        "                && metadata_matches\n                && selector_matches",
    );
    replace_exact_render_fragment(
        &mut output,
        "        }\n    }\n}\n",
        "        }\n        GeneratedIntrinsicVariant::Prmt { mode } => {\n            let Some(op) = Operation::get_op::<PrmtOp>(operation, ctx) else { return false; };\n            match mode {\n                GeneratedPrmtMode::Generic => op.get_attr_nvvm_prmt_mode(ctx).as_deref() == Some(&PrmtModeAttr::Generic),\n                GeneratedPrmtMode::F4e => op.get_attr_nvvm_prmt_mode(ctx).as_deref() == Some(&PrmtModeAttr::F4e),\n                GeneratedPrmtMode::B4e => op.get_attr_nvvm_prmt_mode(ctx).as_deref() == Some(&PrmtModeAttr::B4e),\n                GeneratedPrmtMode::Rc8 => op.get_attr_nvvm_prmt_mode(ctx).as_deref() == Some(&PrmtModeAttr::Rc8),\n                GeneratedPrmtMode::Ecl => op.get_attr_nvvm_prmt_mode(ctx).as_deref() == Some(&PrmtModeAttr::Ecl),\n                GeneratedPrmtMode::Ecr => op.get_attr_nvvm_prmt_mode(ctx).as_deref() == Some(&PrmtModeAttr::Ecr),\n                GeneratedPrmtMode::Rc16 => op.get_attr_nvvm_prmt_mode(ctx).as_deref() == Some(&PrmtModeAttr::Rc16),\n            }\n        }\n    }\n}\n",
    );
    replace_exact_render_fragment(
        &mut output,
        "        }\n    }\n}\n",
        "        }\n        GeneratedIntrinsicVariant::ClusterBarrier { mode } => {\n            let Some(op) = Operation::get_op::<ClusterBarrierOp>(operation, ctx) else { return false; };\n            match mode {\n                GeneratedClusterBarrierMode::Arrive => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Arrive),\n                GeneratedClusterBarrierMode::ArriveAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveAligned),\n                GeneratedClusterBarrierMode::ArriveRelaxed => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxed),\n                GeneratedClusterBarrierMode::ArriveRelaxedAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxedAligned),\n                GeneratedClusterBarrierMode::Wait => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Wait),\n                GeneratedClusterBarrierMode::WaitAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::WaitAligned),\n            }\n        }\n    }\n}\n",
    );
    if wgmma_controls(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "        GeneratedIntrinsicVariant::ClusterBarrier { mode } => {\n            let Some(op) = Operation::get_op::<ClusterBarrierOp>(operation, ctx) else { return false; };\n            match mode {\n                GeneratedClusterBarrierMode::Arrive => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Arrive),\n                GeneratedClusterBarrierMode::ArriveAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveAligned),\n                GeneratedClusterBarrierMode::ArriveRelaxed => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxed),\n                GeneratedClusterBarrierMode::ArriveRelaxedAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxedAligned),\n                GeneratedClusterBarrierMode::Wait => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Wait),\n                GeneratedClusterBarrierMode::WaitAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::WaitAligned),\n            }\n        }\n    }\n}\n",
            "        GeneratedIntrinsicVariant::ClusterBarrier { mode } => {\n            let Some(op) = Operation::get_op::<ClusterBarrierOp>(operation, ctx) else { return false; };\n            match mode {\n                GeneratedClusterBarrierMode::Arrive => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Arrive),\n                GeneratedClusterBarrierMode::ArriveAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveAligned),\n                GeneratedClusterBarrierMode::ArriveRelaxed => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxed),\n                GeneratedClusterBarrierMode::ArriveRelaxedAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxedAligned),\n                GeneratedClusterBarrierMode::Wait => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Wait),\n                GeneratedClusterBarrierMode::WaitAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::WaitAligned),\n            }\n        }\n        GeneratedIntrinsicVariant::WgmmaControl { mode } => match mode {\n            GeneratedWgmmaControlMode::Fence => Operation::get_op::<WgmmaFenceSyncAlignedOp>(operation, ctx).is_some(),\n            GeneratedWgmmaControlMode::CommitGroup => Operation::get_op::<WgmmaCommitGroupSyncAlignedOp>(operation, ctx).is_some(),\n            GeneratedWgmmaControlMode::WaitGroup => Operation::get_op::<WgmmaWaitGroupSyncAlignedOp>(operation, ctx).is_some(),\n        },\n    }\n}\n",
        );
    }
    if scalar_conversions(catalog).next().is_some() {
        let scalar_match = "        GeneratedIntrinsicVariant::ScalarConversion { rounding, saturation } => {\n            let Some(op) = Operation::get_op::<ScalarConversionOp>(operation, ctx) else { return false; };\n            let rounding_matches = match rounding {\n                GeneratedScalarConversionRounding::NearestAway => op.get_attr_nvvm_scalar_conversion_rounding(ctx).as_deref() == Some(&ScalarConversionRoundingAttr::NearestAway),\n                GeneratedScalarConversionRounding::NearestEven => op.get_attr_nvvm_scalar_conversion_rounding(ctx).as_deref() == Some(&ScalarConversionRoundingAttr::NearestEven),\n                GeneratedScalarConversionRounding::TowardZero => op.get_attr_nvvm_scalar_conversion_rounding(ctx).as_deref() == Some(&ScalarConversionRoundingAttr::TowardZero),\n            };\n            let saturation_matches = match saturation {\n                GeneratedScalarConversionSaturation::None => op.get_attr_nvvm_scalar_conversion_saturation(ctx).as_deref() == Some(&ScalarConversionSaturationAttr::None),\n                GeneratedScalarConversionSaturation::Relu => op.get_attr_nvvm_scalar_conversion_saturation(ctx).as_deref() == Some(&ScalarConversionSaturationAttr::Relu),\n                GeneratedScalarConversionSaturation::Satfinite => op.get_attr_nvvm_scalar_conversion_saturation(ctx).as_deref() == Some(&ScalarConversionSaturationAttr::Satfinite),\n                GeneratedScalarConversionSaturation::ReluSatfinite => op.get_attr_nvvm_scalar_conversion_saturation(ctx).as_deref() == Some(&ScalarConversionSaturationAttr::ReluSatfinite),\n            };\n            rounding_matches && saturation_matches\n        }\n";
        if wgmma_controls(catalog).next().is_some() {
            replace_exact_render_fragment(
                &mut output,
                "        GeneratedIntrinsicVariant::WgmmaControl { mode } => match mode {\n            GeneratedWgmmaControlMode::Fence => Operation::get_op::<WgmmaFenceSyncAlignedOp>(operation, ctx).is_some(),\n            GeneratedWgmmaControlMode::CommitGroup => Operation::get_op::<WgmmaCommitGroupSyncAlignedOp>(operation, ctx).is_some(),\n            GeneratedWgmmaControlMode::WaitGroup => Operation::get_op::<WgmmaWaitGroupSyncAlignedOp>(operation, ctx).is_some(),\n        },\n    }\n}\n",
                &format!(
                    "        GeneratedIntrinsicVariant::WgmmaControl {{ mode }} => match mode {{\n            GeneratedWgmmaControlMode::Fence => Operation::get_op::<WgmmaFenceSyncAlignedOp>(operation, ctx).is_some(),\n            GeneratedWgmmaControlMode::CommitGroup => Operation::get_op::<WgmmaCommitGroupSyncAlignedOp>(operation, ctx).is_some(),\n            GeneratedWgmmaControlMode::WaitGroup => Operation::get_op::<WgmmaWaitGroupSyncAlignedOp>(operation, ctx).is_some(),\n        }},\n{scalar_match}    }}\n}}\n"
                ),
            );
        } else {
            replace_exact_render_fragment(
                &mut output,
                "        GeneratedIntrinsicVariant::ClusterBarrier { mode } => {\n            let Some(op) = Operation::get_op::<ClusterBarrierOp>(operation, ctx) else { return false; };\n            match mode {\n                GeneratedClusterBarrierMode::Arrive => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Arrive),\n                GeneratedClusterBarrierMode::ArriveAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveAligned),\n                GeneratedClusterBarrierMode::ArriveRelaxed => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxed),\n                GeneratedClusterBarrierMode::ArriveRelaxedAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxedAligned),\n                GeneratedClusterBarrierMode::Wait => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Wait),\n                GeneratedClusterBarrierMode::WaitAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::WaitAligned),\n            }\n        }\n    }\n}\n",
                &format!(
                    "        GeneratedIntrinsicVariant::ClusterBarrier {{ mode }} => {{\n            let Some(op) = Operation::get_op::<ClusterBarrierOp>(operation, ctx) else {{ return false; }};\n            match mode {{\n                GeneratedClusterBarrierMode::Arrive => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Arrive),\n                GeneratedClusterBarrierMode::ArriveAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveAligned),\n                GeneratedClusterBarrierMode::ArriveRelaxed => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxed),\n                GeneratedClusterBarrierMode::ArriveRelaxedAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::ArriveRelaxedAligned),\n                GeneratedClusterBarrierMode::Wait => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::Wait),\n                GeneratedClusterBarrierMode::WaitAligned => op.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() == Some(&ClusterBarrierModeAttr::WaitAligned),\n            }}\n        }}\n{scalar_match}    }}\n}}\n"
                ),
            );
        }
    }
    if scalar_arithmetics(catalog).next().is_some() {
        let arithmetic_match = r#"        GeneratedIntrinsicVariant::ScalarArithmetic { format, operation: arithmetic_operation, rounding, subnormal, saturation } => {
            let Some(op) = Operation::get_op::<ScalarArithmeticOp>(operation, ctx) else { return false; };
            let format_matches = match format {
                GeneratedScalarArithmeticFormat::F32 => op.get_attr_nvvm_scalar_arithmetic_format(ctx).as_deref() == Some(&ScalarArithmeticFormatAttr::F32),
                GeneratedScalarArithmeticFormat::F64 => op.get_attr_nvvm_scalar_arithmetic_format(ctx).as_deref() == Some(&ScalarArithmeticFormatAttr::F64),
            };
            let operation_matches = match arithmetic_operation {
                GeneratedScalarArithmeticOperation::Mul => op.get_attr_nvvm_scalar_arithmetic_operation(ctx).as_deref() == Some(&ScalarArithmeticOperationAttr::Mul),
                GeneratedScalarArithmeticOperation::Div => op.get_attr_nvvm_scalar_arithmetic_operation(ctx).as_deref() == Some(&ScalarArithmeticOperationAttr::Div),
                GeneratedScalarArithmeticOperation::Fma => op.get_attr_nvvm_scalar_arithmetic_operation(ctx).as_deref() == Some(&ScalarArithmeticOperationAttr::Fma),
                GeneratedScalarArithmeticOperation::Add => op.get_attr_nvvm_scalar_arithmetic_operation(ctx).as_deref() == Some(&ScalarArithmeticOperationAttr::Add),
            };
            let rounding_matches = match rounding {
                GeneratedScalarArithmeticRounding::Rn => op.get_attr_nvvm_scalar_arithmetic_rounding(ctx).as_deref() == Some(&ScalarArithmeticRoundingAttr::Rn),
                GeneratedScalarArithmeticRounding::Rz => op.get_attr_nvvm_scalar_arithmetic_rounding(ctx).as_deref() == Some(&ScalarArithmeticRoundingAttr::Rz),
                GeneratedScalarArithmeticRounding::Rm => op.get_attr_nvvm_scalar_arithmetic_rounding(ctx).as_deref() == Some(&ScalarArithmeticRoundingAttr::Rm),
                GeneratedScalarArithmeticRounding::Rp => op.get_attr_nvvm_scalar_arithmetic_rounding(ctx).as_deref() == Some(&ScalarArithmeticRoundingAttr::Rp),
            };
            let subnormal_matches = match subnormal {
                GeneratedScalarArithmeticSubnormal::Preserve => op.get_attr_nvvm_scalar_arithmetic_subnormal(ctx).as_deref() == Some(&ScalarArithmeticSubnormalAttr::Preserve),
                GeneratedScalarArithmeticSubnormal::Ftz => op.get_attr_nvvm_scalar_arithmetic_subnormal(ctx).as_deref() == Some(&ScalarArithmeticSubnormalAttr::Ftz),
            };
            let saturation_matches = match saturation {
                GeneratedScalarArithmeticSaturation::None => op.get_attr_nvvm_scalar_arithmetic_saturation(ctx).as_deref() == Some(&ScalarArithmeticSaturationAttr::None),
                GeneratedScalarArithmeticSaturation::Sat => op.get_attr_nvvm_scalar_arithmetic_saturation(ctx).as_deref() == Some(&ScalarArithmeticSaturationAttr::Sat),
            };
            format_matches && operation_matches && rounding_matches && subnormal_matches && saturation_matches
        }
"#;
        replace_exact_render_fragment(
            &mut output,
            "            rounding_matches && saturation_matches\n        }\n    }\n}\n",
            &format!(
                "            rounding_matches && saturation_matches\n        }}\n{arithmetic_match}    }}\n}}\n"
            ),
        );
    }
    if extended_minmax(catalog).next().is_some() {
        let minmax_match = r#"        GeneratedIntrinsicVariant::ExtendedMinMax { format, operation: minmax_operation, subnormal, nan, xorsign_abs } => {
            let Some(op) = Operation::get_op::<ExtendedMinMaxOp>(operation, ctx) else { return false; };
            let format_matches = match format {
                GeneratedExtendedMinMaxFormat::F32 => op.get_attr_nvvm_extended_minmax_format(ctx).as_deref() == Some(&ExtendedMinMaxFormatAttr::F32),
                GeneratedExtendedMinMaxFormat::F16 => op.get_attr_nvvm_extended_minmax_format(ctx).as_deref() == Some(&ExtendedMinMaxFormatAttr::F16),
                GeneratedExtendedMinMaxFormat::Bf16 => op.get_attr_nvvm_extended_minmax_format(ctx).as_deref() == Some(&ExtendedMinMaxFormatAttr::Bf16),
                GeneratedExtendedMinMaxFormat::F16x2 => op.get_attr_nvvm_extended_minmax_format(ctx).as_deref() == Some(&ExtendedMinMaxFormatAttr::F16x2),
                GeneratedExtendedMinMaxFormat::Bf16x2 => op.get_attr_nvvm_extended_minmax_format(ctx).as_deref() == Some(&ExtendedMinMaxFormatAttr::Bf16x2),
            };
            let operation_matches = match minmax_operation {
                GeneratedExtendedMinMaxOperation::Min => op.get_attr_nvvm_extended_minmax_operation(ctx).as_deref() == Some(&ExtendedMinMaxOperationAttr::Min),
                GeneratedExtendedMinMaxOperation::Max => op.get_attr_nvvm_extended_minmax_operation(ctx).as_deref() == Some(&ExtendedMinMaxOperationAttr::Max),
            };
            let subnormal_matches = match subnormal {
                GeneratedExtendedMinMaxSubnormal::Preserve => op.get_attr_nvvm_extended_minmax_subnormal(ctx).as_deref() == Some(&ExtendedMinMaxSubnormalAttr::Preserve),
                GeneratedExtendedMinMaxSubnormal::Ftz => op.get_attr_nvvm_extended_minmax_subnormal(ctx).as_deref() == Some(&ExtendedMinMaxSubnormalAttr::Ftz),
            };
            let nan_matches = match nan {
                GeneratedExtendedMinMaxNan::Number => op.get_attr_nvvm_extended_minmax_nan(ctx).as_deref() == Some(&ExtendedMinMaxNanAttr::Number),
                GeneratedExtendedMinMaxNan::Nan => op.get_attr_nvvm_extended_minmax_nan(ctx).as_deref() == Some(&ExtendedMinMaxNanAttr::Nan),
            };
            let xorsign_abs_matches = match xorsign_abs {
                false => op.get_attr_nvvm_extended_minmax_xorsign_abs(ctx).as_deref() == Some(&ExtendedMinMaxXorSignAbsAttr::Disabled),
                true => op.get_attr_nvvm_extended_minmax_xorsign_abs(ctx).as_deref() == Some(&ExtendedMinMaxXorSignAbsAttr::Enabled),
            };
            format_matches && operation_matches && subnormal_matches && nan_matches && xorsign_abs_matches
        }
"#;
        let match_end = output
            .rfind("    }\n}\n")
            .expect("generated target matcher terminator");
        output.insert_str(match_end, minmax_match);
    }
    output.push_str(
        "\nuse dialect_nvvm::ops::{ClusterBarrierModeAttr, ClusterBarrierOp, LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp, LdmatrixShapeAttr, LdmatrixStateSpaceAttr, PackedAtomicAddOp, PackedAtomicAtomicityAttr, PackedAtomicFormatAttr, PackedAtomicOrderingAttr, PackedAtomicRoundingAttr, PackedAtomicScopeAttr, PackedAtomicStateSpaceAttr, PackedAtomicSubnormalAttr, PrmtModeAttr, PrmtOp, RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaKindAttr, RegisterMmaLayoutAttr, RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr, SparseMmaAccumulatorAttr, SparseMmaElementAttr, SparseMmaLayoutAttr, SparseMmaMetadataAttr, SparseMmaOp, SparseMmaOverflowAttr, SparseMmaSelectorAttr, SparseMmaShapeAttr};\nuse pliron::{context::{Context, Ptr}, operation::Operation};\n",
    );
    if wgmma_controls(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "SparseMmaSelectorAttr, SparseMmaShapeAttr};",
            "SparseMmaSelectorAttr, SparseMmaShapeAttr, WgmmaCommitGroupSyncAlignedOp, WgmmaFenceSyncAlignedOp, WgmmaWaitGroupSyncAlignedOp};",
        );
    }
    if scalar_conversions(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "SparseMmaSelectorAttr, SparseMmaShapeAttr",
            "ScalarConversionOp, ScalarConversionRoundingAttr, ScalarConversionSaturationAttr, SparseMmaSelectorAttr, SparseMmaShapeAttr",
        );
    }
    if scalar_arithmetics(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "SparseMmaSelectorAttr, SparseMmaShapeAttr",
            "ScalarArithmeticFormatAttr, ScalarArithmeticOp, ScalarArithmeticOperationAttr, ScalarArithmeticRoundingAttr, ScalarArithmeticSaturationAttr, ScalarArithmeticSubnormalAttr, SparseMmaSelectorAttr, SparseMmaShapeAttr",
        );
    }
    if extended_minmax(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "SparseMmaSelectorAttr, SparseMmaShapeAttr",
            "ExtendedMinMaxFormatAttr, ExtendedMinMaxNanAttr, ExtendedMinMaxOp, ExtendedMinMaxOperationAttr, ExtendedMinMaxSubnormalAttr, ExtendedMinMaxXorSignAbsAttr, SparseMmaSelectorAttr, SparseMmaShapeAttr",
        );
    }
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        replace_exact_render_fragment(
            &mut output,
            "SparseMmaSelectorAttr, SparseMmaShapeAttr",
            "SparseMmaSelectorAttr, SparseMmaShapeAttr, Tcgen05MmaBBufferAttr, Tcgen05MmaBUsageAttr, Tcgen05MmaCollectorAAttr, Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr, Tcgen05MmaKindAttr, Tcgen05MmaOp",
        );
    }
    output.push('\n');
    for (shard, _) in groups {
        writeln!(output, "mod {shard};").unwrap();
    }
    output.push_str("\n#[cfg(test)]\nmod tests;\n");
    output
}

const TARGETS_GENERATED_DIR: &str = "crates/cuda-oxide-codegen/src/generated_intrinsic_targets";

/// Family shard for one target record. Mirrors the dialect-nvvm family
/// coalescings; tcgen05 sub-splits on the same contract members the
/// importer shards use, with the MMA half further cut on the
/// warp-specialized form bit so every table file stays reviewable.
fn targets_record_shard(record: &CatalogIntrinsic) -> &'static str {
    if record.family == "tcgen05" {
        let tcgen05 = record.tcgen05.as_ref().expect("tcgen05 record");
        if let Some(mma) = &tcgen05.mma {
            return if tcgen05_mma_form_name(mma.form).starts_with("Ws") {
                "tcgen05_mma_ws"
            } else {
                "tcgen05_mma"
            };
        }
        if tcgen05.ld.is_some() {
            return "tcgen05_ld";
        }
        if tcgen05.st.is_some() {
            return "tcgen05_st";
        }
        if tcgen05.cp.is_some() {
            return "tcgen05_cp";
        }
        return "tcgen05_other";
    }
    match record.family.as_str() {
        "cp_async_copy" | "cp_async_control" | "cp_async_mbarrier" => "cp_async",
        "counted_barrier" | "grid_dependency" | "register_control" => "execution_control",
        family => TARGETS_FAMILY_SHARDS
            .iter()
            .copied()
            .find(|shard| *shard == family)
            .unwrap_or_else(|| panic!("unmapped generated intrinsic family `{family}`")),
    }
}

const TARGETS_FAMILY_SHARDS: &[&str] = &[
    "sreg",
    "active_mask",
    "ldmatrix",
    "stmatrix",
    "register_mma",
    "sparse_mma",
    "packed_atomic",
    "redux",
    "vote",
    "warp_match",
    "elect",
    "warp_barrier",
    "warp_shuffle",
    "dotprod",
    "packed_alu",
    "integer_minmax",
    "packed_conversion",
    "scalar_conversion",
    "scalar_arithmetic",
    "scalar_math",
    "extended_minmax",
    "movmatrix",
    "prmt",
    "cluster_barrier",
    "cluster_memory",
    "debug_control",
    "clc",
    "wgmma_control",
    "mbarrier_basic",
    "mbarrier_extended",
    "sync",
    "tma",
];

/// Group the abi-sorted records per shard; group order is each shard's first
/// appearance in abi order, so iteration stays as close to the old global
/// table order as a per-family split permits.
fn targets_groups(catalog: &CatalogFile) -> Vec<(&'static str, Vec<&CatalogIntrinsic>)> {
    let mut records = catalog.intrinsics.iter().collect::<Vec<_>>();
    records.sort_by(|left, right| left.rust.abi_id.cmp(&right.rust.abi_id));
    let mut groups: Vec<(&'static str, Vec<&CatalogIntrinsic>)> = Vec::new();
    for record in records {
        let shard = targets_record_shard(record);
        match groups.iter_mut().find(|(name, _)| *name == shard) {
            Some((_, members)) => members.push(record),
            None => groups.push((shard, vec![record])),
        }
    }
    groups
}

/// Every type name a target shard may need from the module root.
const TARGETS_TYPE_CANDIDATES: &[&str] = &[
    "GeneratedBackendRequirement",
    "GeneratedClusterBarrierMode",
    "GeneratedExtendedMinMaxFormat",
    "GeneratedExtendedMinMaxNan",
    "GeneratedExtendedMinMaxOperation",
    "GeneratedExtendedMinMaxSubnormal",
    "GeneratedHardwareAlternative",
    "GeneratedHardwareTarget",
    "GeneratedImmediateBinding",
    "GeneratedIntrinsicBackend",
    "GeneratedIntrinsicRange",
    "GeneratedIntrinsicTarget",
    "GeneratedIntrinsicVariant",
    "GeneratedLdmatrixElement",
    "GeneratedLdmatrixLayout",
    "GeneratedLdmatrixMultiplicity",
    "GeneratedLdmatrixShape",
    "GeneratedLlvmFacts",
    "GeneratedPackedAtomicFormat",
    "GeneratedPrmtMode",
    "GeneratedPtxVersion",
    "GeneratedRegisterMmaAccumulator",
    "GeneratedRegisterMmaElement",
    "GeneratedRegisterMmaKind",
    "GeneratedRegisterMmaLayout",
    "GeneratedRegisterMmaOperation",
    "GeneratedRegisterMmaOverflow",
    "GeneratedRegisterMmaShape",
    "GeneratedScalarArithmeticFormat",
    "GeneratedScalarArithmeticOperation",
    "GeneratedScalarArithmeticRounding",
    "GeneratedScalarArithmeticSaturation",
    "GeneratedScalarArithmeticSubnormal",
    "GeneratedScalarConversionRounding",
    "GeneratedScalarConversionSaturation",
    "GeneratedSelectionAddressSpace",
    "GeneratedSelectionAlternative",
    "GeneratedSelectionConstraints",
    "GeneratedSparseMmaAccumulator",
    "GeneratedSparseMmaElement",
    "GeneratedSparseMmaLayout",
    "GeneratedSparseMmaMetadata",
    "GeneratedSparseMmaOverflow",
    "GeneratedSparseMmaSelector",
    "GeneratedSparseMmaShape",
    "GeneratedTargetAlternative",
    "GeneratedTargetContract",
    "GeneratedTargetRequirement",
    "GeneratedTargetSelectorBinding",
    "GeneratedTcgen05MmaForm",
    "GeneratedTcgen05MmaTargetSelector",
    "GeneratedWgmmaControlMode",
];

fn targets_shard_imports(entries: &str) -> String {
    let items: Vec<&str> = TARGETS_TYPE_CANDIDATES
        .iter()
        .copied()
        .filter(|item| uses_identifier(entries, item))
        .collect();
    format!("use super::{{{}}};\n", items.join(", "))
}

fn targets_shard_file(
    catalog: &CatalogFile,
    hash: &str,
    shard: &str,
    records: &[&CatalogIntrinsic],
) -> String {
    let mut entries = String::new();
    for record in records {
        render_target_record(&mut entries, catalog, record);
    }
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "//! Generated intrinsic target records: `{shard}` intrinsics.\n"
    )
    .unwrap();
    output.push_str(&targets_shard_imports(&entries));
    output.push_str("\npub(super) const TARGETS: &[GeneratedIntrinsicTarget] = &[\n");
    output.push_str(&entries);
    output.push_str("];\n");
    output
}

/// Test files per table shard. `register_mma` alone splits its assertion
/// blocks on the accumulator contract so no generated test file passes the
/// 10k-line review bound; every other shard keeps a single test file.
fn targets_test_shards<'catalog>(
    groups: &[(&'static str, Vec<&'catalog CatalogIntrinsic>)],
) -> Vec<(String, Vec<&'catalog CatalogIntrinsic>)> {
    let mut test_shards = Vec::new();
    for (shard, records) in groups {
        if *shard == "register_mma" {
            let (integer, float): (Vec<_>, Vec<_>) = records.iter().copied().partition(|record| {
                matches!(
                    record
                        .register_mma
                        .as_ref()
                        .expect("register_mma record")
                        .accumulator,
                    RegisterMmaAccumulator::S32
                )
            });
            test_shards.push(("register_mma_float".to_owned(), float));
            test_shards.push(("register_mma_s32".to_owned(), integer));
        } else {
            test_shards.push(((*shard).to_owned(), records.clone()));
        }
    }
    test_shards
}

fn targets_tests_mod_file(
    catalog: &CatalogFile,
    hash: &str,
    test_shards: &[(String, Vec<&CatalogIntrinsic>)],
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated intrinsic target tests, grouped per family shard.\n\nuse super::*;\nuse std::collections::BTreeSet;\n\n",
    );
    for (shard, _) in test_shards {
        writeln!(output, "mod {shard};").unwrap();
    }
    output.push_str(
        "\n#[test]\nfn generated_target_table_is_unique_and_lookup_is_complete() {\n    let mut ids = BTreeSet::new();\n    let mut markers = BTreeSet::new();\n    let mut abi_ids = BTreeSet::new();\n    for group in GENERATED_INTRINSIC_TARGET_GROUPS {\n        let mut previous_abi_id = None;\n        for target in *group {\n            if let Some(previous) = previous_abi_id {\n                assert!(previous < target.abi_id, \"generated ABI IDs are not strictly increasing: {previous} then {}\", target.abi_id);\n            }\n            previous_abi_id = Some(target.abi_id);\n            assert!(abi_ids.insert(target.abi_id), \"duplicate generated ABI ID {}\", target.abi_id);\n            assert!(ids.insert(target.id), \"duplicate generated intrinsic ID {}\", target.id);\n            assert!(markers.insert(target.marker), \"duplicate generated marker {}\", target.marker);\n            assert_eq!(generated_intrinsic_target_by_marker(target.marker), Some(target));\n            assert!(generated_intrinsic_targets_by_op_name(target.dialect_op).any(|candidate| candidate == target));\n        }\n    }\n    assert!(generated_intrinsic_target_by_marker(\"v1:i9999\").is_none());\n}\n",
    );
    output
}

fn targets_tests_shard_file(
    catalog: &CatalogFile,
    hash: &str,
    shard: &str,
    records: &[&CatalogIntrinsic],
) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "//! Generated intrinsic target tests: `{shard}` intrinsics.\n\nuse super::super::*;\n"
    )
    .unwrap();
    for record in records {
        writeln!(
            output,
            "#[test]\nfn {}_target_matches_the_catalog() {{",
            record.id
        )
        .unwrap();
        render_target_record_assertions(&mut output, catalog, record);
        output.push_str("}\n\n");
    }
    output
}

pub(super) fn render_targets_files(catalog: &CatalogFile, hash: &str) -> Vec<(PathBuf, String)> {
    let groups = targets_groups(catalog);
    let test_shards = targets_test_shards(&groups);
    let mut files = vec![
        (
            PathBuf::from(format!("{TARGETS_GENERATED_DIR}/mod.rs")),
            targets_mod_file(catalog, hash, &groups),
        ),
        (
            PathBuf::from(format!("{TARGETS_GENERATED_DIR}/tests/mod.rs")),
            targets_tests_mod_file(catalog, hash, &test_shards),
        ),
    ];
    for (shard, records) in &groups {
        files.push((
            PathBuf::from(format!("{TARGETS_GENERATED_DIR}/{shard}.rs")),
            targets_shard_file(catalog, hash, shard, records),
        ));
    }
    for (shard, records) in &test_shards {
        files.push((
            PathBuf::from(format!("{TARGETS_GENERATED_DIR}/tests/{shard}.rs")),
            targets_tests_shard_file(catalog, hash, shard, records),
        ));
    }
    files
}

#[cfg(test)]
pub(super) fn render_targets(catalog: &CatalogFile, hash: &str) -> String {
    render_targets_files(catalog, hash)
        .into_iter()
        .map(|(_, contents)| contents)
        .collect::<Vec<_>>()
        .join("\n")
}
