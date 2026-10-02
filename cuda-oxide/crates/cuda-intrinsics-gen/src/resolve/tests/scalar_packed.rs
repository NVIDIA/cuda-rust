/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogFile, ExtendedMinMaxAdmission, ExtendedMinMaxFormat,
    ExtendedMinMaxNan, ExtendedMinMaxOperation, ExtendedMinMaxSubnormal, ImportedFile,
    ImportedIntrinsic, IntrinsicBackend, IntrinsicSource, OverlayIntrinsic, OverlayShardFile,
    PackedAluFormat, PackedAluOperation, PackedConversionDestinationFormat,
    PackedConversionRounding, PackedConversionSaturation, PrmtMode, RuntimeValidation,
    ScalarArithmeticAdmission, ScalarArithmeticFormat, ScalarArithmeticOperation,
    ScalarArithmeticRounding, ScalarArithmeticSaturation, ScalarArithmeticSubnormal,
    ScalarConversionAdmission, ScalarConversionRounding, ScalarConversionSaturation,
    ScalarMathAdmission,
};
use crate::util::read_json;
use std::collections::BTreeMap;
use std::path::Path;

use super::fixtures::*;
use crate::model::{ImportedSelection, ScalarConversionAdmissionVariant};
use crate::resolve::driver::*;
use crate::resolve::families::*;
use crate::resolve::guards::*;
use crate::resolve::overlay::*;
use crate::resolve::policy::*;
use crate::resolve::targets::validate_arch_introduction_floor;

#[test]
fn compact_prmt_admission_requires_every_mode_and_reserved_abi_id() {
    assert_eq!(
        expand_prmt_admission(&test_prmt_admission()).unwrap().len(),
        7
    );

    let mut missing = test_prmt_admission();
    missing.variants.pop();
    assert!(expand_prmt_admission(&missing).is_err());

    let mut duplicate = test_prmt_admission();
    duplicate.variants[6].mode = PrmtMode::Rc8;
    assert!(expand_prmt_admission(&duplicate).is_err());

    let mut wrong_abi = test_prmt_admission();
    wrong_abi.variants[0].abi_id = "i9999".into();
    assert!(expand_prmt_admission(&wrong_abi).is_err());
}

#[test]
fn compact_fp8_conversion_axes_require_the_exact_closed_product() {
    let records = expand_packed_conversion_fp8_admission(&test_fp8_conversion_admission()).unwrap();
    assert_eq!(records.len(), 4);
    assert_eq!(records[0].id, "cvt_rn_satfinite_e4m3x2_f32");
    assert_eq!(records[1].id, "cvt_rn_satfinite_relu_e4m3x2_f32");
    assert_eq!(records[2].id, "cvt_rn_satfinite_e5m2x2_f32");
    assert_eq!(records[3].id, "cvt_rn_satfinite_relu_e5m2x2_f32");
    assert!(records.iter().all(|record| {
        record.rust_result == "u16"
            && record.dialect_results == ["i16"]
            && record.llvm_results == ["i16"]
            && record.minimum_ptx == "8.1"
            && record.minimum_sm.as_deref() == Some("sm_89")
            && record.pure
            && !record.convergent
    }));

    let mut missing_format = test_fp8_conversion_admission();
    missing_format.destination_formats.pop();
    assert!(expand_packed_conversion_fp8_admission(&missing_format).is_err());

    let mut reversed_formats = test_fp8_conversion_admission();
    reversed_formats.destination_formats.reverse();
    assert!(expand_packed_conversion_fp8_admission(&reversed_formats).is_err());

    let mut missing_saturation = test_fp8_conversion_admission();
    missing_saturation.saturations.pop();
    assert!(expand_packed_conversion_fp8_admission(&missing_saturation).is_err());

    let mut reversed_saturations = test_fp8_conversion_admission();
    reversed_saturations.saturations.reverse();
    assert!(expand_packed_conversion_fp8_admission(&reversed_saturations).is_err());

    let mut wrong_count = test_fp8_conversion_admission();
    wrong_count.product_count = 3;
    assert!(expand_packed_conversion_fp8_admission(&wrong_count).is_err());

    let mut executed = test_fp8_conversion_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_packed_conversion_fp8_admission(&executed).is_err());
}

#[test]
fn compact_fp8_f16x2_conversion_axes_require_the_exact_closed_product() {
    let records =
        expand_packed_conversion_fp8_f16x2_admission(&test_fp8_f16x2_conversion_admission())
            .unwrap();
    assert_eq!(records.len(), 8);
    assert_eq!(records[0].id, "cvt_rn_satfinite_e4m3x2_f16x2");
    assert_eq!(records[1].id, "cvt_rn_satfinite_relu_e4m3x2_f16x2");
    assert_eq!(records[2].id, "cvt_rn_f16x2_e4m3x2");
    assert_eq!(records[3].id, "cvt_rn_relu_f16x2_e4m3x2");
    assert_eq!(records[4].id, "cvt_rn_satfinite_e5m2x2_f16x2");
    assert_eq!(records[5].id, "cvt_rn_satfinite_relu_e5m2x2_f16x2");
    assert_eq!(records[6].id, "cvt_rn_f16x2_e5m2x2");
    assert_eq!(records[7].id, "cvt_rn_relu_f16x2_e5m2x2");

    // Every form takes one packed operand and carries the Ada floor,
    // including the unpacks whose destination is the older f16x2.
    assert!(records.iter().all(|record| {
        record.rust_arguments.len() == 1
            && record.dialect_operands.len() == 1
            && record.llvm_arguments.len() == 1
            && record.expected_ptx.operands.len() == 2
            && record.minimum_ptx == "8.1"
            && record.minimum_sm.as_deref() == Some("sm_89")
            && record.pure
            && !record.convergent
    }));

    // Packing narrows f16x2 to a 16-bit FP8 pair; unpacking widens back.
    assert!(
        records
            .iter()
            .filter(|record| record.id.ends_with("_f16x2"))
            .all(|record| {
                record.rust_arguments == ["u32"]
                    && record.rust_result == "u16"
                    && record.llvm_arguments == ["v2f16"]
                    && record.llvm_results == ["i16"]
            })
    );
    assert!(
        records
            .iter()
            .filter(|record| !record.id.ends_with("_f16x2"))
            .all(|record| {
                record.rust_arguments == ["u16"]
                    && record.rust_result == "u32"
                    && record.llvm_arguments == ["i16"]
                    && record.llvm_results == ["v2f16"]
            })
    );

    let mut missing_format = test_fp8_f16x2_conversion_admission();
    missing_format.fp8_formats.pop();
    assert!(expand_packed_conversion_fp8_f16x2_admission(&missing_format).is_err());

    let mut reversed_formats = test_fp8_f16x2_conversion_admission();
    reversed_formats.fp8_formats.reverse();
    assert!(expand_packed_conversion_fp8_f16x2_admission(&reversed_formats).is_err());

    let mut missing_direction = test_fp8_f16x2_conversion_admission();
    missing_direction.directions.pop();
    assert!(expand_packed_conversion_fp8_f16x2_admission(&missing_direction).is_err());

    let mut reversed_directions = test_fp8_f16x2_conversion_admission();
    reversed_directions.directions.reverse();
    assert!(expand_packed_conversion_fp8_f16x2_admission(&reversed_directions).is_err());

    let mut without_relu = test_fp8_f16x2_conversion_admission();
    without_relu.relu_variants = false;
    assert!(expand_packed_conversion_fp8_f16x2_admission(&without_relu).is_err());

    let mut wrong_count = test_fp8_f16x2_conversion_admission();
    wrong_count.product_count = 4;
    assert!(expand_packed_conversion_fp8_f16x2_admission(&wrong_count).is_err());

    let mut executed = test_fp8_f16x2_conversion_admission();
    executed.runtime_validation = RuntimeValidation::Executed;
    assert!(expand_packed_conversion_fp8_f16x2_admission(&executed).is_err());
}

#[test]
fn packed_atomic_closed_semantics_reject_every_unreviewed_mutation() {
    let valid = packed_policy("packed_atomic_add_f16x2");
    validate_ptx_native_policy(&valid).unwrap();

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let overlay =
        std::fs::read_to_string(repo_root.join("intrinsics/overlay/packed_atomic.toml")).unwrap();
    for (field, accepted, rejected) in [
        (
            "state space",
            "state_space = \"global\"",
            "state_space = \"shared\"",
        ),
        (
            "ordering",
            "ordering = \"relaxed\"",
            "ordering = \"acquire\"",
        ),
        ("scope", "scope = \"gpu\"", "scope = \"system\""),
        (
            "rounding",
            "rounding = \"nearest_even\"",
            "rounding = \"toward_zero\"",
        ),
        (
            "subnormal",
            "subnormal = \"preserve\"",
            "subnormal = \"flush\"",
        ),
        (
            "atomicity",
            "atomicity = \"per_element\"",
            "atomicity = \"coherent_pair\"",
        ),
        (
            "pointer safety",
            "pointer_contract = \"mutable_global_u32_aligned4\"",
            "pointer_contract = \"unaligned\"",
        ),
        (
            "mixed access safety",
            "access_contract = \"no_mixed_whole_word_or_non_atomic_access\"",
            "access_contract = \"mixed_access_allowed\"",
        ),
        (
            "scope safety",
            "scope_contract = \"racing_atomics_mutually_inclusive\"",
            "scope_contract = \"scope_unchecked\"",
        ),
        (
            "codegen",
            "codegen_contract = \"exact_native_instruction\"",
            "codegen_contract = \"semantic_equivalence\"",
        ),
    ] {
        let mutated = overlay.replacen(accepted, rejected, 1);
        let error = toml::from_str::<OverlayShardFile>(&mutated).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(rejected.split(" = ").next().unwrap()),
            "{field} mutation did not fail closed: {error}"
        );
    }

    let mut safe = valid;
    safe.safe = true;
    safe.safe_allowlist_reason = Some("incorrectly claims no caller obligations".into());
    assert!(
        validate_ptx_native_policy(&safe)
            .unwrap_err()
            .to_string()
            .contains("unsafe must-use packed atomic")
    );
}

#[test]
fn packed_alu_recipes_accept_only_the_reviewed_source_shape_and_floor() {
    let operations = [
        PackedAluOperation::Add,
        PackedAluOperation::Sub,
        PackedAluOperation::Mul,
        PackedAluOperation::Fma,
        PackedAluOperation::FmaRelu,
        PackedAluOperation::Min,
        PackedAluOperation::Max,
        PackedAluOperation::Neg,
        PackedAluOperation::Abs,
    ];
    for format in [PackedAluFormat::Bf16x2, PackedAluFormat::F16x2] {
        for operation in operations {
            let policy = packed_alu_policy(format, operation);
            match packed_alu_declaration(format, operation) {
                Some(declaration) => validate_imported_policy(&policy, &declaration).unwrap(),
                None => validate_ptx_native_policy(&policy).unwrap(),
            }
        }
    }

    let declaration =
        packed_alu_declaration(PackedAluFormat::Bf16x2, PackedAluOperation::Fma).unwrap();
    let reject_imported = |policy: &OverlayIntrinsic, message: &str| {
        let error = validate_imported_policy(policy, &declaration).unwrap_err();
        assert!(error.to_string().contains(message), "{error:#}");
    };

    let valid = packed_alu_policy(PackedAluFormat::Bf16x2, PackedAluOperation::Fma);
    let mut wrong_source = valid.clone();
    wrong_source.source_record = Some("int_nvvm_fma_rn_bf16".into());
    reject_imported(&wrong_source, "source");

    let mut wrong_format = valid.clone();
    wrong_format.packed_alu.as_mut().unwrap().format = PackedAluFormat::F16x2;
    reject_imported(&wrong_format, "identity");

    let mut wrong_operation = valid.clone();
    wrong_operation.packed_alu.as_mut().unwrap().operation = PackedAluOperation::Max;
    reject_imported(&wrong_operation, "identity");

    let mut wrong_floor = valid.clone();
    wrong_floor.minimum_sm = Some("sm_90".into());
    reject_imported(&wrong_floor, "target floor");

    let mut wrong_effects = valid.clone();
    wrong_effects.memory = "read".into();
    reject_imported(&wrong_effects, "effects");

    let mut wrong_section = valid.clone();
    wrong_section.ptx_isa_section = "9.7.4 Floating Point Instructions".into();
    reject_imported(&wrong_section, "PTX provenance");

    let mut wrong_url = valid.clone();
    wrong_url.ptx_isa_url =
        "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions"
            .into();
    reject_imported(&wrong_url, "PTX provenance");

    let mut wrong_adapter = valid.clone();
    wrong_adapter.lowering = "direct_nvvm".into();
    reject_imported(&wrong_adapter, "lowering recipe");

    let mut wrong_backend = valid;
    wrong_backend.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
    reject_imported(&wrong_backend, "inline-PTX routes");

    let mut wrong_native = packed_alu_policy(PackedAluFormat::Bf16x2, PackedAluOperation::Add);
    wrong_native.source = Some(IntrinsicSource::PtxNative {
        instruction: "add.bf16x2".into(),
    });
    let error = validate_ptx_native_policy(&wrong_native).unwrap_err();
    assert!(error.to_string().contains("PTX-native recipe"));

    let mut invented_llvm = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Add);
    invented_llvm.llvm_symbol = Some("llvm.nvvm.add.rn.f16x2".into());
    let error = validate_ptx_native_policy(&invented_llvm).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("must not invent LLVM source facts")
    );

    let mut unreviewed_modifier =
        packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Add);
    unreviewed_modifier.expected_ptx.modifiers = vec!["rn".into(), "ftz".into(), "f16x2".into()];
    let error = validate_ptx_native_policy(&unreviewed_modifier).unwrap_err();
    assert!(error.to_string().contains("exact packed-ALU instruction"));

    let mut wrong_arity = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Add);
    wrong_arity.expected_ptx.operands.pop();
    let error = validate_ptx_native_policy(&wrong_arity).unwrap_err();
    assert!(error.to_string().contains("exact packed-ALU instruction"));

    let f16_declaration =
        packed_alu_declaration(PackedAluFormat::F16x2, PackedAluOperation::Fma).unwrap();
    let reject_f16 = |policy: &OverlayIntrinsic, message: &str| {
        let error = validate_imported_policy(policy, &f16_declaration).unwrap_err();
        assert!(error.to_string().contains(message), "{error:#}");
    };
    let f16 = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Fma);

    let mut wrong_signature = f16.clone();
    wrong_signature.llvm_arguments = vec!["v2bf16".into(); 3];
    reject_f16(&wrong_signature, "LLVM argument signature mismatch");

    let mut missing_must_use = f16.clone();
    missing_must_use.must_use = false;
    reject_f16(&missing_must_use, "reviewed safe packed-ALU API");

    let mut wrong_native_floor = f16.clone();
    wrong_native_floor
        .packed_alu
        .as_mut()
        .unwrap()
        .native_minimum_sm = 30;
    // PTX 4.2 can name sm_30; the recipe must still enforce native sm_53.
    validate_arch_introduction_floor(&wrong_native_floor).unwrap();
    reject_f16(&wrong_native_floor, "target floor");

    let mut wrong_backend_floor = f16;
    wrong_backend_floor.backend_lowerings[0].minimum_ptx = Some("4.2".into());
    reject_f16(&wrong_backend_floor, "exact packed-ALU floor");

    let abs_declaration =
        packed_alu_declaration(PackedAluFormat::F16x2, PackedAluOperation::Abs).unwrap();
    let mut wrong_abs = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Abs);
    wrong_abs.resolved_llvm_symbol = Some("llvm.nvvm.fabs.v2bf16".into());
    let error = validate_imported_policy(&wrong_abs, &abs_declaration).unwrap_err();
    assert!(error.to_string().contains("LLVM source or signature"));
}

#[test]
fn pinned_packed_alu_records_match_the_closed_recipes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let packed: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_alu")
        .collect();
    assert_eq!(packed.len(), 30);
    for policy in packed {
        let source = resolve_policy_source(policy).unwrap();
        let declaration = policy
            .source_record
            .as_deref()
            .map(|record| declarations[record]);
        validate_policy(policy, &source, declaration, 1).unwrap();
    }
}

#[test]
fn pinned_packed_conversion_records_match_the_closed_recipes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let overlay = load_resolution_base(&repo_root).unwrap().overlay;
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let packed: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_conversion")
        .collect();
    assert_eq!(packed.len(), 18);
    for policy in packed {
        let source = resolve_policy_source(policy).unwrap();
        let declaration = policy
            .source_record
            .as_deref()
            .map(|record| declarations[record]);
        validate_policy(policy, &source, declaration, 1).unwrap();
    }
}

#[test]
fn selectionless_packed_conversion_is_admitted_only_by_its_closed_recipe() {
    let cases = [
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::TowardZero,
            PackedConversionSaturation::None,
        ),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ),
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ),
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::TowardZero,
            PackedConversionSaturation::None,
        ),
        (
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ),
        (
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ),
        (
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ),
        (
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ),
    ];
    for (destination, rounding, saturation) in cases {
        let policy = packed_conversion_policy(destination, rounding, saturation);
        let declaration = packed_conversion_declaration(&policy);
        validate_imported_policy(&policy, &declaration).unwrap();
    }

    let valid = packed_conversion_policy(
        PackedConversionDestinationFormat::Bf16x2,
        PackedConversionRounding::NearestEven,
        PackedConversionSaturation::None,
    );
    let declaration = packed_conversion_declaration(&valid);

    let reject = |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, message: &str| {
        let error = validate_imported_policy(policy, declaration).unwrap_err();
        assert!(error.to_string().contains(message), "{error:#}");
    };

    let mut wrong_source = valid.clone();
    wrong_source.source_record = Some("int_nvvm_ff2bf16x2_rz".into());
    reject(&wrong_source, &declaration, "identity or LLVM source");

    let mut wrong_floor = valid.clone();
    wrong_floor.minimum_ptx = "7.8".into();
    reject(&wrong_floor, &declaration, "target floor");

    let mut wrong_effect = valid.clone();
    wrong_effect.pure = false;
    reject(&wrong_effect, &declaration, "effects");

    let mut wrong_section = valid.clone();
    wrong_section.ptx_isa_section = "9.7.9 Data Movement and Conversion Instructions".into();
    reject(&wrong_section, &declaration, "PTX provenance");

    let mut wrong_url = valid.clone();
    wrong_url.ptx_isa_url =
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions"
                .into();
    reject(&wrong_url, &declaration, "PTX provenance");

    let mut wrong_shape = valid.clone();
    wrong_shape.expected_ptx.modifiers.swap(1, 2);
    reject(&wrong_shape, &declaration, "expected PTX does not match");

    let mut wrong_identity = valid.clone();
    wrong_identity.id = "cvt_f16x2_f32".into();
    reject(&wrong_identity, &declaration, "identity or LLVM source");

    let mut unsupported = valid.clone();
    let conversion = unsupported.packed_conversion.as_mut().unwrap();
    conversion.rounding = PackedConversionRounding::TowardZero;
    conversion.saturation = PackedConversionSaturation::Relu;
    reject(
        &unsupported,
        &declaration,
        "unsupported packed-conversion source, destination",
    );

    let mut wrong_compatibility = valid.clone();
    wrong_compatibility.compatibility_rust_paths =
        vec!["cuda_device::tcgen05::cvt_f32x2_bf16x2".into()];
    reject(&wrong_compatibility, &declaration, "conversion API");

    let mut wrong_result = valid.clone();
    wrong_result.llvm_results = vec!["v2f16".into()];
    reject(&wrong_result, &declaration, "result signature mismatch");

    let mut selected = declaration.clone();
    selected.selections.push(ImportedSelection {
        source_record: "UNREVIEWED".into(),
        asm: "cvt.rn.bf16x2.f32 $d, $a, $b;".into(),
        predicates: vec![],
        constraints: Default::default(),
    });
    reject(&valid, &selected, "selectionless");
}

#[test]
fn scalar_conversion_admission_is_closed_and_backend_specific() {
    let variants = [
        (
            "i0368",
            ScalarConversionRounding::NearestAway,
            ScalarConversionSaturation::None,
        ),
        (
            "i0369",
            ScalarConversionRounding::NearestAway,
            ScalarConversionSaturation::Satfinite,
        ),
        (
            "i0370",
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::None,
        ),
        (
            "i0371",
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::Relu,
        ),
        (
            "i0372",
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::Satfinite,
        ),
        (
            "i0373",
            ScalarConversionRounding::NearestEven,
            ScalarConversionSaturation::ReluSatfinite,
        ),
        (
            "i0374",
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::None,
        ),
        (
            "i0375",
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::Relu,
        ),
        (
            "i0376",
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::Satfinite,
        ),
        (
            "i0377",
            ScalarConversionRounding::TowardZero,
            ScalarConversionSaturation::ReluSatfinite,
        ),
    ];
    let admission = ScalarConversionAdmission {
        llvm_evidence_profile: "llvm-scalar".into(),
        libnvvm_evidence_profile: "libnvvm-scalar".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: variants
            .iter()
            .map(
                |(abi_id, rounding, saturation)| ScalarConversionAdmissionVariant {
                    abi_id: (*abi_id).into(),
                    rounding: *rounding,
                    saturation: *saturation,
                },
            )
            .collect(),
    };

    let records = expand_scalar_conversion_admission(&admission).unwrap();
    assert_eq!(records.len(), 10);
    for (record, (abi_id, rounding, saturation)) in records.iter().zip(variants) {
        assert_eq!(record.abi_id, abi_id);
        assert_eq!(record.rust_arguments, ["f32"]);
        assert_eq!(record.rust_result, "u32");
        assert_eq!(
            record.compatibility_rust_paths,
            [format!("cuda_device::convert::{}", record.rust_name)]
        );
        let conversion = record.scalar_conversion.as_ref().unwrap();
        assert_eq!(conversion.rounding, rounding);
        assert_eq!(conversion.saturation, saturation);
        assert!(record.backend_lowerings.iter().any(|lowering| {
            lowering.backend == IntrinsicBackend::LlvmNvptx
                && lowering.mechanism == BackendLoweringMechanism::TypedNvvm
        }));
        assert!(record.backend_lowerings.iter().any(|lowering| {
            lowering.backend == IntrinsicBackend::LibNvvm
                && lowering.mechanism == BackendLoweringMechanism::InlinePtx
        }));
    }

    let mut reordered = admission.clone();
    reordered.variants.swap(0, 1);
    assert!(
        expand_scalar_conversion_admission(&reordered)
            .unwrap_err()
            .to_string()
            .contains("canonical ten variants")
    );
}

#[test]
fn scalar_arithmetic_admission_is_closed_and_selects_only_direct_ptx() {
    let variants = canonical_scalar_arithmetic_variants();
    let admission = ScalarArithmeticAdmission {
        llvm_evidence_profile: "llvm-scalar-arithmetic".into(),
        libnvvm_evidence_profile: "libnvvm-scalar-arithmetic".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: variants
            .iter()
            .copied()
            .enumerate()
            .map(
                |(index, variant)| crate::model::ScalarArithmeticAdmissionVariant {
                    abi_id: format!("i{:04}", 390 + index),
                    format: variant.0,
                    operation: variant.1,
                    rounding: variant.2,
                    subnormal: variant.3,
                    saturation: variant.4,
                },
            )
            .collect(),
    };
    let records = expand_scalar_arithmetic_admission(&admission).unwrap();
    assert_eq!(records.len(), 64);
    assert_eq!(records.first().unwrap().id, "mul_rn_f64");
    assert_eq!(records.first().unwrap().abi_id, "i0390");
    assert_eq!(records.last().unwrap().id, "add_rp_ftz_sat_f32");
    assert_eq!(records.last().unwrap().abi_id, "i0453");
    validate_unique_overlay(&records, 1).unwrap();
    let llvm_inline_count = records
        .iter()
        .filter(|record| {
            record.backend_lowerings.iter().any(|lowering| {
                lowering.backend == IntrinsicBackend::LlvmNvptx
                    && lowering.mechanism == BackendLoweringMechanism::InlinePtx
            })
        })
        .count();
    assert_eq!(llvm_inline_count, 16);
    assert!(records.iter().all(|record| {
        let expected = if record
            .scalar_arithmetic
            .as_ref()
            .is_some_and(|arithmetic| arithmetic.saturation == ScalarArithmeticSaturation::Sat)
        {
            BackendLoweringMechanism::InlinePtx
        } else {
            BackendLoweringMechanism::TypedNvvm
        };
        record.backend_lowerings.iter().any(|lowering| {
            lowering.backend == IntrinsicBackend::LlvmNvptx && lowering.mechanism == expected
        })
    }));

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .into_iter()
        .map(|record| (record.source_record.clone(), record))
        .collect::<BTreeMap<_, _>>();
    let mut mixed_count = 0;
    let mut add_two_selection_count = 0;
    let mut add_six_selection_count = 0;
    for record in &records {
        let declaration = &declarations[record.source_record.as_ref().unwrap()];
        validate_imported_policy(record, declaration).unwrap();
        let selected = declaration
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(record, selection).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), 1, "{}", record.id);
        if declaration.selections.len() == 3 {
            mixed_count += 1;
        }
        if record
            .scalar_arithmetic
            .as_ref()
            .is_some_and(|contract| contract.operation == ScalarArithmeticOperation::Add)
        {
            match declaration.selections.len() {
                2 => add_two_selection_count += 1,
                6 => add_six_selection_count += 1,
                count => panic!("unexpected add selection count {count}"),
            }
        }
    }
    assert_eq!(mixed_count, 8);
    assert_eq!(add_two_selection_count, 12);
    assert_eq!(add_six_selection_count, 8);
    let add_ftz_sat = records
        .iter()
        .find(|record| record.id == "add_rn_ftz_sat_f32")
        .unwrap();
    assert_eq!(
        add_ftz_sat.expected_ptx.modifiers,
        ["rn", "sat", "ftz", "f32"]
    );

    let mut non_contiguous_abi = admission.clone();
    non_contiguous_abi.variants[0].abi_id = "i9999".into();
    let non_contiguous_records = expand_scalar_arithmetic_admission(&non_contiguous_abi).unwrap();
    assert_eq!(non_contiguous_records[0].abi_id, "i9999");

    let mut reordered = admission.clone();
    reordered.variants.swap(0, 1);
    assert!(
        expand_scalar_arithmetic_admission(&reordered)
            .unwrap_err()
            .to_string()
            .contains("canonical 64 variants")
    );
    assert!(
        scalar_arithmetic_recipe((
            ScalarArithmeticFormat::F64,
            ScalarArithmeticOperation::Fma,
            ScalarArithmeticRounding::Rn,
            ScalarArithmeticSubnormal::Ftz,
            ScalarArithmeticSaturation::None,
        ))
        .is_none()
    );
}

#[test]
fn scalar_math_admission_keeps_semantic_order_but_accepts_non_contiguous_abi_ids() {
    let variants = canonical_scalar_math_variants();
    let admission = ScalarMathAdmission {
        llvm_evidence_profile: "llvm-scalar-math".into(),
        libnvvm_evidence_profile: "libnvvm-scalar-math".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: variants
            .iter()
            .copied()
            .enumerate()
            .map(
                |(index, variant)| crate::model::ScalarMathAdmissionVariant {
                    abi_id: format!("i{:04}", 782 + index),
                    libnvvm_evidence_profile: (index == 40).then(|| "libnvvm-ex2-f16".into()),
                    format: variant.0,
                    operation: variant.1,
                    precision: variant.2,
                    subnormal: variant.3,
                },
            )
            .collect(),
    };

    let records = expand_scalar_math_admission(&admission).unwrap();
    assert_eq!(records.len(), 41);
    assert_eq!(records.first().unwrap().abi_id, "i0782");
    assert_eq!(records.last().unwrap().abi_id, "i0822");
    assert!(
        records
            .iter()
            .all(|record| record.backend_lowerings.len() == 2)
    );
    assert_eq!(
        records.last().unwrap().backend_lowerings[1].evidence_profile,
        "libnvvm-ex2-f16"
    );
    assert!(
        records[..40].iter().all(|record| {
            record.backend_lowerings[1].evidence_profile == "libnvvm-scalar-math"
        })
    );

    let mut non_contiguous_abi = admission.clone();
    non_contiguous_abi.variants[40].abi_id = "i9999".into();
    let records = expand_scalar_math_admission(&non_contiguous_abi).unwrap();
    assert_eq!(records.last().unwrap().abi_id, "i9999");

    let mut reordered = admission;
    reordered.variants.swap(0, 1);
    assert!(
        expand_scalar_math_admission(&reordered)
            .unwrap_err()
            .to_string()
            .contains("canonical 41 variants")
    );
}

#[test]
fn checked_in_scalar_math_overlay_pins_the_current_libnvvm_override_set() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let scalar_math = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "scalar_math")
        .collect::<Vec<_>>();
    assert_eq!(scalar_math.len(), 41);
    assert!(
        scalar_math
            .iter()
            .all(|record| record.backend_lowerings.len() == 2)
    );

    let family_profile = "cuda-13.3-libnvvm-13.3.33-scalar-math";
    let overrides = scalar_math
        .iter()
        .filter_map(|record| {
            let libnvvm = record
                .backend_lowerings
                .iter()
                .find(|route| route.backend == IntrinsicBackend::LibNvvm)
                .unwrap();
            (libnvvm.evidence_profile != family_profile).then_some(record.id.as_str())
        })
        .collect::<Vec<_>>();
    assert_eq!(overrides, ["ex2_approx_f16"]);
}

#[test]
fn extended_minmax_admission_is_exact_and_fail_closed() {
    let variants = canonical_extended_minmax_variants();
    let admission = ExtendedMinMaxAdmission {
        llvm_evidence_profile: "llvm-extended-minmax".into(),
        libnvvm_evidence_profile: "libnvvm-extended-minmax".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: variants
            .iter()
            .copied()
            .enumerate()
            .map(
                |(index, variant)| crate::model::ExtendedMinMaxAdmissionVariant {
                    abi_id: if index < 28 {
                        format!("i{:04}", 550 + index)
                    } else {
                        format!("i{:04}", 830 + index - 28)
                    },
                    format: variant.0,
                    operation: variant.1,
                    subnormal: variant.2,
                    nan: variant.3,
                    xorsign_abs: variant.4,
                },
            )
            .collect(),
    };
    let records = expand_extended_minmax_admission(&admission).unwrap();
    assert_eq!(records.len(), 52);
    assert_eq!(records.first().unwrap().id, "min_ftz_f16x2");
    assert_eq!(records.first().unwrap().abi_id, "i0550");
    // The first reserved block ends here; the scalar 16-bit forms admitted
    // afterwards continue at a second base rather than at i0578.
    assert_eq!(records[27].id, "max_xorsign_abs_f16x2");
    assert_eq!(records[27].abi_id, "i0577");
    assert_eq!(records[28].id, "min_f16");
    assert_eq!(records[28].abi_id, "i0830");
    assert_eq!(records.last().unwrap().id, "max_nan_xorsign_abs_bf16");
    assert_eq!(records.last().unwrap().abi_id, "i0853");
    validate_unique_overlay(&records, 1).unwrap();
    assert_eq!(
        records
            .iter()
            .filter(|record| record.minimum_sm.as_deref() == Some("sm_80"))
            .count(),
        20
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| record.minimum_sm.as_deref() == Some("sm_86"))
            .count(),
        32
    );
    assert!(records.iter().all(|record| {
        record.backend_lowerings.len() == 2
            && record
                .backend_lowerings
                .iter()
                .all(|lowering| lowering.mechanism == BackendLoweringMechanism::InlinePtx)
    }));

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations = imported
        .intrinsics
        .into_iter()
        .map(|record| (record.source_record.clone(), record))
        .collect::<BTreeMap<_, _>>();
    for record in &records {
        let declaration = &declarations[record.source_record.as_ref().unwrap()];
        assert_eq!(declaration.selections.len(), 1, "{}", record.id);
        validate_imported_policy(record, declaration).unwrap();
        assert_eq!(
            declaration
                .selections
                .iter()
                .filter(|selection| selection_matches_policy(record, selection).unwrap())
                .count(),
            1,
            "{}",
            record.id
        );
    }
    let packed_max = records
        .iter()
        .find(|record| record.id == "max_nan_bf16x2")
        .unwrap();
    assert_eq!(
        declarations[packed_max.source_record.as_ref().unwrap()].selections[0].source_record,
        "INT_NVVM_FMAN_NaN_bf16x2"
    );
    assert_eq!(
        declarations["int_nvvm_fmax_ftz_nan_xorsign_abs_f"].selections[0].source_record,
        "INT_NVVM_FMAX_FTZ_NAN_XORSIGN_ABS_F"
    );
    assert!(
        declarations["int_nvvm_fmax_ftz_nan_xorsign_abs_f"].selections[0]
            .asm
            .contains(".NaN.")
    );
    for excluded in [
        "int_nvvm_fmin_ftz_bf16x2",
        "int_nvvm_fmax_ftz_nan_bf16x2",
        "int_nvvm_fmin_f",
        "int_nvvm_fmax_nan_f",
    ] {
        assert!(declarations[excluded].selections.is_empty(), "{excluded}");
        assert!(
            records
                .iter()
                .all(|record| record.source_record.as_deref() != Some(excluded)),
            "{excluded} was admitted without an instruction selection"
        );
    }

    let catalog: CatalogFile = read_json(&repo_root.join("intrinsics/catalog.json")).unwrap();
    for id in ["min_f16x2", "max_f16x2", "min_bf16x2", "max_bf16x2"] {
        assert_eq!(
            catalog
                .intrinsics
                .iter()
                .find(|record| record.id == id)
                .unwrap()
                .family,
            "packed_alu"
        );
        assert!(records.iter().all(|record| record.id != id));
    }

    let mut non_contiguous_abi = admission.clone();
    non_contiguous_abi.variants[0].abi_id = "i9999".into();
    let non_contiguous_records = expand_extended_minmax_admission(&non_contiguous_abi).unwrap();
    assert_eq!(non_contiguous_records[0].abi_id, "i9999");

    let mut reordered = admission.clone();
    reordered.variants.swap(0, 1);
    assert!(
        expand_extended_minmax_admission(&reordered)
            .unwrap_err()
            .to_string()
            .contains("exact canonical 28 variants")
    );
    assert!(
        extended_minmax_recipe((
            ExtendedMinMaxFormat::Bf16x2,
            ExtendedMinMaxOperation::Min,
            ExtendedMinMaxSubnormal::Ftz,
            ExtendedMinMaxNan::Nan,
            false,
        ))
        .is_none()
    );
    let mut changed = declarations["int_nvvm_fmin_ftz_f16x2"].clone();
    changed.selections[0].predicates.swap(0, 1);
    assert!(validate_imported_policy(&records[0], &changed).is_err());
}
