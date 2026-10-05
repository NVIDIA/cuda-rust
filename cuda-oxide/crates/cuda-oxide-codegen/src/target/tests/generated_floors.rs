/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::fixtures::{PTX87_EXACT_SM120A, PTX88, PTX90, PTX91_FUTURE, TCGEN_F16, TCGEN_I8};
use crate::generated::GeneratedModuleRequirements;
use crate::generated_intrinsic_targets::{
    GeneratedHardwareAlternative, GeneratedHardwareTarget, GeneratedIntrinsicBackend,
    GeneratedPtxVersion, GeneratedTargetAlternative, GeneratedTargetContract,
};
use crate::target::arch::*;
use crate::target::detect::*;
use crate::target::features::*;
use crate::target::generated_requirements::*;
use crate::target::select::*;
use libnvvm_sys::CudaArch;

#[test]
fn paired_target_floors_compose_with_target_cpu_minima() {
    let generated = GeneratedModuleRequirements::from_targets(vec![&TCGEN_F16]);

    assert_eq!(
        generated_ptx_isa_requirement(&generated).unwrap(),
        PtxIsaRequirement::new(86)
    );
    for target in ["sm_100a", "sm_101a", "sm_103a", "sm_110a"] {
        assert!(
            generated_target_satisfied(&target.parse().unwrap(), &generated),
            "{target}"
        );
    }
    for (target, requirement) in [
        ("sm_100a", PtxIsaRequirement::new(86)),
        ("sm_101a", PtxIsaRequirement::new(86)),
        ("sm_103a", PtxIsaRequirement::new(88)),
        ("sm_110a", PtxIsaRequirement::new(90)),
    ] {
        assert_eq!(
            generated_ptx_isa_requirement_for_target(&generated, &target.parse().unwrap()).unwrap(),
            requirement
        );
        assert_eq!(
            required_ptx_feature(&target.parse().unwrap(), requirement).unwrap(),
            None
        );
    }
    assert_eq!(
        generated_requirement_ptx_floor(&"sm_103a".parse().unwrap(), TCGEN_F16.requirement),
        Some(88)
    );
    assert_eq!(
        generated_requirement_ptx_floor(&"sm_110a".parse().unwrap(), TCGEN_F16.requirement),
        Some(90)
    );
    for target in ["sm_120a", "sm_121a"] {
        assert!(
            !generated_target_satisfied(&target.parse().unwrap(), &generated),
            "{target}"
        );
    }
    for target in ["sm_100f", "sm_101f", "sm_103f", "sm_110f"] {
        assert!(
            generated_target_satisfied(&target.parse().unwrap(), &generated),
            "{target}"
        );
    }
    assert_eq!(
        select_target_with_generated(DetectedFeatures::Basic, &generated)
            .unwrap()
            .sm(),
        "sm_100a"
    );

    let generated = GeneratedModuleRequirements::from_targets(vec![&TCGEN_I8]);
    assert_eq!(
        generated_ptx_isa_requirement(&generated).unwrap(),
        PtxIsaRequirement::new(86)
    );
    assert_eq!(
        generated_ptx_isa_requirement_for_target(&generated, &"sm_100a".parse().unwrap()).unwrap(),
        PtxIsaRequirement::new(86)
    );
    assert_eq!(
        required_ptx_feature(&"sm_100a".parse().unwrap(), PtxIsaRequirement::new(86)).unwrap(),
        None
    );
    for (target, requirement) in [
        ("sm_101a", PtxIsaRequirement::new(86)),
        ("sm_110a", PtxIsaRequirement::new(90)),
    ] {
        assert_eq!(
            generated_ptx_isa_requirement_for_target(&generated, &target.parse().unwrap()).unwrap(),
            requirement
        );
        assert_eq!(
            required_ptx_feature(&target.parse().unwrap(), requirement).unwrap(),
            None
        );
    }
    assert!(!generated_target_satisfied(
        &"sm_103a".parse().unwrap(),
        &generated
    ));
    assert!(!generated_target_satisfied(
        &"sm_100f".parse().unwrap(),
        &generated
    ));
}

#[test]
fn sm101_aliases_reject_an_aggregate_ptx90_requirement() {
    let generated = GeneratedModuleRequirements::from_targets(vec![&TCGEN_F16, &PTX90]);

    for target in ["sm_101a", "sm_101f"] {
        let error = generated_ptx_isa_requirement_for_target(&generated, &target.parse().unwrap())
            .unwrap_err();
        assert!(
            error.contains("renamed the sm_101 target to sm_110"),
            "{error}"
        );

        let f16 = GeneratedModuleRequirements::from_targets(vec![&TCGEN_F16]);
        let text = ModuleRequirements {
            features: DetectedFeatures::Basic,
            ptx_isa: PtxIsaRequirement::new(90),
        };
        let error =
            merge_generated_module_requirements_for_target(text, &f16, &target.parse().unwrap())
                .unwrap_err();
        assert!(
            error.contains("renamed the sm_101 target to sm_110"),
            "{error}"
        );
    }
}

#[test]
fn dynamic_stack_calls_require_ptx73_and_sm52() {
    for llvm in [
        "%saved = call ptr @llvm.stacksave.p0()\n",
        "%saved = tail call ptr addrspace(12) @llvm.stacksave.p12()\n",
        "call void @llvm.stackrestore.p3(ptr addrspace(3) %saved)\n",
        "%saved = call i8* @llvm.stacksave()\n",
        "call void @llvm.stackrestore(i8* %saved)\n",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(llvm);
        assert!(
            requirements
                .features
                .contains(DetectedFeatures::DynamicStack),
            "{llvm}"
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(73), "{llvm}");
    }

    for near_match in [
        "%saved = call ptr @llvm.stacksavex.p0()\n",
        "%saved = call ptr @llvm.stacksave_extra.p0()\n",
        "%saved = call ptr @llvm.stacksave.p()\n",
        "%saved = call ptr @llvm.stacksave.p0x()\n",
        "%saved = call ptr @llvm.stacksave.p0.extra()\n",
        "call void @llvm.stackrestorex.p0(ptr %saved)\n",
        "call void @llvm.stackrestore.p0_extra(ptr %saved)\n",
        "%saved = call ptr @llvm.stacksave.p0\n",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(near_match);
        assert_eq!(
            requirements.features,
            DetectedFeatures::Basic,
            "{near_match}"
        );
        assert_eq!(
            requirements.ptx_isa,
            PtxIsaRequirement::Default,
            "{near_match}"
        );
    }

    assert!(!arch_satisfies_feature(
        50,
        None,
        DetectedFeatures::DynamicStack
    ));
    assert!(arch_satisfies_feature(
        52,
        None,
        DetectedFeatures::DynamicStack
    ));
    for target in ["sm_70", "sm_86"] {
        let parsed = target.parse::<CudaArch>().unwrap();
        assert!(
            validate_target_features(&parsed, DetectedFeatures::DynamicStack).is_ok(),
            "{target}"
        );
    }
    let sm_50 = "sm_50".parse::<CudaArch>().unwrap();
    let error = validate_target_features(&sm_50, DetectedFeatures::DynamicStack).unwrap_err();
    assert!(error.contains("DynamicStack"), "{error}");
}

#[test]
fn dynamic_stack_mentions_without_calls_do_not_raise_requirements() {
    let non_calls = [
        "declare ptr @llvm.stacksave.p0()\n",
        "declare void @llvm.stackrestore.p0(ptr)\n",
        "; %saved = call ptr @llvm.stacksave.p0()\n",
        "!0 = !{!\"call ptr @llvm.stacksave.p0()\"}\n",
        "@message = private constant [39 x i8] c\"call ptr @llvm.stacksave.p0()\\00\"\n",
        "!1 = !{ptr @llvm.stacksave.p0}\n",
        "declare ptr @llvm.stacksave.p0 ; call ptr @llvm.stacksave.p0()\n",
    ];

    for llvm in non_calls {
        let requirements = detect_module_requirements_in_llvm_text(llvm);
        assert_eq!(requirements.features, DetectedFeatures::Basic, "{llvm}");
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Default, "{llvm}");
    }

    let call_after_quoted_semicolon = concat!(
        "@message = private constant [2 x i8] c\";\\00\"\n",
        "%saved = call ptr @llvm.stacksave.p0() ; ignored comment\n",
    );
    assert!(
        detect_module_requirements_in_llvm_text(call_after_quoted_semicolon)
            .features
            .contains(DetectedFeatures::DynamicStack)
    );
}

#[test]
fn ptx73_feature_is_requested_only_when_the_target_default_is_older() {
    assert_eq!(
        ptx_isa_requirement_for_floor(72, "test", "test").unwrap(),
        PtxIsaRequirement::new(73)
    );
    assert_eq!(
        ptx_isa_requirement_for_floor(73, "test", "test").unwrap(),
        PtxIsaRequirement::new(73)
    );
    assert_eq!(
        ptx_isa_requirement_for_floor(74, "test", "test").unwrap(),
        PtxIsaRequirement::new(78)
    );
    for target in ["sm_70", "sm_80", "sm_86"] {
        assert_eq!(
            required_ptx_feature(&target.parse().unwrap(), PtxIsaRequirement::new(73)).unwrap(),
            Some("+ptx73"),
            "{target}"
        );
    }
    assert_eq!(
        required_ptx_feature(&"sm_87".parse().unwrap(), PtxIsaRequirement::new(73)).unwrap(),
        None
    );
}

#[test]
fn unrecorded_target_fails_closed_for_explicit_ptx() {
    let error =
        required_ptx_feature(&"sm_89a".parse().unwrap(), PtxIsaRequirement::new(80)).unwrap_err();
    assert!(error.contains("no recorded PTX ISA floor"));
    let error =
        required_ptx_feature(&"sm_999a".parse().unwrap(), PtxIsaRequirement::Default).unwrap_err();
    assert!(error.contains("no recorded PTX ISA floor"));
}

#[test]
fn paired_minimum_target_diagnostic_preserves_range_meaning() {
    static TARGETS: &[GeneratedTargetAlternative] = &[GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(88),
        hardware: GeneratedHardwareAlternative::MinimumSm(100),
    }];
    static CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
        selectors: &[],
        alternatives: TARGETS,
    }];

    assert_eq!(
        describe_generated_hardware(GeneratedHardwareTarget::TargetMatrix {
            contracts: CONTRACTS,
        }),
        "sm_100 or newer at PTX 8.8"
    );
}

#[test]
fn paired_target_matrix_flows_through_backend_target_resolution() {
    let llvm = GeneratedModuleRequirements::from_targets(vec![&TCGEN_I8])
        .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
    let libnvvm = GeneratedModuleRequirements::from_targets(vec![&TCGEN_I8])
        .for_backend(GeneratedIntrinsicBackend::LibNvvm);

    assert!(generated_target_satisfied(
        &"sm_101a".parse().unwrap(),
        &llvm
    ));
    assert!(!generated_target_satisfied(
        &"sm_101a".parse().unwrap(),
        &libnvvm
    ));
    assert!(!generated_target_satisfied(
        &"sm_103a".parse().unwrap(),
        &llvm
    ));
    assert!(generated_target_satisfied(
        &"sm_110a".parse().unwrap(),
        &libnvvm
    ));

    assert_eq!(
        resolve_ptx_target_with_generated(
            Some("sm_101a"),
            "CUDA_OXIDE_TARGET",
            None,
            DetectedFeatures::Basic,
            &llvm,
        )
        .unwrap(),
        ("sm_101a".parse().unwrap(), "CUDA_OXIDE_TARGET")
    );
    assert!(
        resolve_ptx_target_with_generated(
            Some("sm_103a"),
            "CUDA_OXIDE_TARGET",
            None,
            DetectedFeatures::Basic,
            &llvm,
        )
        .is_err()
    );
    assert_eq!(
        resolve_ptx_target_with_generated(
            None,
            "CUDA_OXIDE_TARGET",
            Some(&"sm_101a".parse::<cuda_target_spec::DeviceArch>().unwrap()),
            DetectedFeatures::Basic,
            &llvm,
        )
        .unwrap(),
        ("sm_101a".parse().unwrap(), "detected GPU")
    );
    assert_eq!(
        resolve_ptx_target_with_generated(
            None,
            "CUDA_OXIDE_TARGET",
            Some(&"sm_101a".parse::<cuda_target_spec::DeviceArch>().unwrap()),
            DetectedFeatures::Basic,
            &libnvvm,
        )
        .unwrap(),
        ("sm_100a".parse().unwrap(), "feature requirement")
    );
    assert_eq!(
        resolve_ptx_target_with_generated(
            None,
            "CUDA_OXIDE_TARGET",
            None,
            DetectedFeatures::Basic,
            &llvm
        )
        .unwrap(),
        ("sm_100a".parse().unwrap(), "feature requirement")
    );

    let base = ModuleRequirements {
        features: DetectedFeatures::Basic,
        ptx_isa: PtxIsaRequirement::Default,
    };
    assert_eq!(
        merge_generated_module_requirements_for_target(base, &llvm, &"sm_101a".parse().unwrap())
            .unwrap()
            .ptx_isa,
        PtxIsaRequirement::new(86)
    );
    assert_eq!(
        merge_generated_module_requirements_for_target(base, &llvm, &"sm_110a".parse().unwrap())
            .unwrap()
            .ptx_isa,
        PtxIsaRequirement::new(90)
    );

    let error =
        crate::export::resolve_nvvm_target_with_generated(Some("sm_103a"), None, None, &libnvvm)
            .unwrap_err()
            .to_string();
    assert!(error.contains("tcgen_i8"), "{error}");
    assert_eq!(
        crate::export::resolve_nvvm_target_with_generated(Some("sm_110a"), None, None, &libnvvm,)
            .unwrap()
            .sm(),
        "sm_110a"
    );
    assert_eq!(
        crate::export::resolve_nvvm_target_with_generated(
            None,
            Some(&"sm_103a".parse::<cuda_target_spec::DeviceArch>().unwrap()),
            None,
            &libnvvm,
        )
        .unwrap()
        .sm(),
        "sm_100a"
    );
}

#[test]
fn generated_ptx87_exact_sm120a_requirement_is_preserved() {
    let generated = GeneratedModuleRequirements::from_targets(vec![&PTX87_EXACT_SM120A]);

    assert_eq!(
        generated_ptx_isa_requirement(&generated).unwrap(),
        PtxIsaRequirement::new(87)
    );
    assert!(PtxIsaRequirement::new(86) < PtxIsaRequirement::new(87));
    assert_eq!(
        required_ptx_feature(&"sm_100a".parse().unwrap(), PtxIsaRequirement::new(87)).unwrap(),
        Some("+ptx87")
    );
    assert_eq!(
        required_ptx_feature(&"sm_120a".parse().unwrap(), PtxIsaRequirement::new(87)).unwrap(),
        None
    );
    assert_eq!(
        required_ptx_feature(&"sm_100f".parse().unwrap(), PtxIsaRequirement::new(87)).unwrap(),
        None
    );
    assert_eq!(
        required_ptx_feature(&"sm_120f".parse().unwrap(), PtxIsaRequirement::new(87)).unwrap(),
        None
    );
    assert_eq!(
        select_target_with_generated(DetectedFeatures::Basic, &generated)
            .unwrap()
            .sm(),
        "sm_120a"
    );
    assert!(generated_target_satisfied(
        &"sm_120a".parse().unwrap(),
        &generated
    ));
    for incompatible in ["sm_120", "sm_120f", "sm_121a"] {
        assert!(
            !generated_target_satisfied(&incompatible.parse().unwrap(), &generated),
            "{incompatible}"
        );
    }
}

#[test]
fn generated_ptx88_and_ptx90_floors_are_preserved() {
    let generated = GeneratedModuleRequirements::from_targets(vec![&PTX88]);
    assert_eq!(
        generated_ptx_isa_requirement(&generated).unwrap(),
        PtxIsaRequirement::new(88)
    );
    assert_eq!(
        required_ptx_feature(&"sm_100a".parse().unwrap(), PtxIsaRequirement::new(88)).unwrap(),
        Some("+ptx88")
    );
    assert_eq!(
        required_ptx_feature(&"sm_103a".parse().unwrap(), PtxIsaRequirement::new(88)).unwrap(),
        None
    );
    assert_eq!(
        ptx_isa_requirement_for_floor(89, "test", "test").unwrap(),
        PtxIsaRequirement::new(90)
    );
    assert_eq!(
        required_ptx_feature(&"sm_100a".parse().unwrap(), PtxIsaRequirement::new(90)).unwrap(),
        Some("+ptx90")
    );
    assert_eq!(
        required_ptx_feature(&"sm_110a".parse().unwrap(), PtxIsaRequirement::new(90)).unwrap(),
        None
    );
    validate_ptx_isa_for_llvm_major(PtxIsaRequirement::new(87), Some(21)).unwrap();
    validate_ptx_isa_for_llvm_major(PtxIsaRequirement::new(88), None).unwrap();
    validate_ptx_isa_for_llvm_major(PtxIsaRequirement::new(88), Some(21)).unwrap();
    validate_ptx_isa_for_llvm_major(PtxIsaRequirement::new(88), Some(22)).unwrap();
    assert!(validate_ptx_isa_for_llvm_major(PtxIsaRequirement::new(90), None).is_err());
    assert!(validate_ptx_isa_for_llvm_major(PtxIsaRequirement::new(90), Some(21)).is_err());
    validate_ptx_isa_for_llvm_major(PtxIsaRequirement::new(90), Some(22)).unwrap();

    let generated = GeneratedModuleRequirements::from_targets(vec![&PTX91_FUTURE]);
    let error = generated_ptx_isa_requirement(&generated).unwrap_err();

    assert!(error.contains("requires PTX 9.1"), "{error}");
    assert!(
        error.contains("newer than cuda-oxide can request"),
        "{error}"
    );
}

#[test]
fn generated_redux_floor_matches_the_lowered_ptx_detector() {
    use crate::generated_intrinsic_targets::generated_intrinsic_target_by_marker;

    let target = generated_intrinsic_target_by_marker("v1:i0017").unwrap();
    let generated = GeneratedModuleRequirements::from_targets(vec![target]);
    let detected = detect_module_requirements_in_llvm_text("redux.sync.add.s32 $0, $1, $2;");

    assert_eq!(
        generated_ptx_isa_requirement(&generated).unwrap(),
        detected.ptx_isa
    );
    assert_eq!(detected.ptx_isa, PtxIsaRequirement::new(70));
    assert!(detected.features.contains(DetectedFeatures::Sm80));
    for arch in ["sm_75", "sm_80", "sm_90"] {
        assert_eq!(
            generated_target_satisfied(&arch.parse().unwrap(), &generated),
            arch_satisfies(&arch.parse().unwrap(), detected.features),
            "{arch}"
        );
    }
}

#[test]
fn generated_packed_atomic_floors_are_backend_specific() {
    use crate::generated_intrinsic_targets::{
        GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
    };

    let f16 = generated_intrinsic_target_by_marker("v1:i0014").unwrap();
    let llvm = GeneratedModuleRequirements::from_targets(vec![f16])
        .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
    assert!(generated_target_satisfied(&"sm_70".parse().unwrap(), &llvm));
    assert_eq!(
        generated_ptx_isa_requirement(&llvm).unwrap(),
        PtxIsaRequirement::new(62)
    );

    let libnvvm = GeneratedModuleRequirements::from_targets(vec![f16])
        .for_backend(GeneratedIntrinsicBackend::LibNvvm);
    assert!(!generated_target_satisfied(
        &"sm_70".parse().unwrap(),
        &libnvvm
    ));
    assert!(generated_target_satisfied(
        &"sm_75".parse().unwrap(),
        &libnvvm
    ));

    let bf16 = generated_intrinsic_target_by_marker("v1:i0015").unwrap();
    let bf16 = GeneratedModuleRequirements::from_targets(vec![bf16]);
    assert!(!generated_target_satisfied(
        &"sm_89".parse().unwrap(),
        &bf16
    ));
    assert!(generated_target_satisfied(&"sm_90".parse().unwrap(), &bf16));
    let error = resolve_ptx_target_with_generated(
        Some("sm_89"),
        "CUDA_OXIDE_TARGET",
        None,
        DetectedFeatures::Basic,
        &bf16,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("packed_atomic_add_bf16x2"), "{error}");
    assert!(error.contains("sm_90 or newer"), "{error}");
}

#[test]
fn generated_non_mma_tcgen05_targets_preserve_the_backend_split() {
    use crate::generated_intrinsic_targets::{
        GeneratedIntrinsicBackend, GeneratedIntrinsicVariant, generated_intrinsic_targets,
    };

    let targets = generated_intrinsic_targets()
        .filter(|target| {
            target.id.starts_with("tcgen05_")
                && !matches!(target.variant, GeneratedIntrinsicVariant::Tcgen05Mma { .. })
        })
        .collect::<Vec<_>>();
    assert_eq!(targets.len(), 209);

    for target in targets {
        let llvm = GeneratedModuleRequirements::from_targets(vec![target])
            .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
        let libnvvm = GeneratedModuleRequirements::from_targets(vec![target])
            .for_backend(GeneratedIntrinsicBackend::LibNvvm);

        assert_eq!(
            generated_ptx_isa_requirement(&llvm).unwrap(),
            PtxIsaRequirement::new(86),
            "{}",
            target.id
        );
        assert_eq!(
            generated_ptx_isa_requirement(&libnvvm).unwrap(),
            PtxIsaRequirement::new(86),
            "{}",
            target.id
        );
        for arch in ["sm_100a", "sm_103a", "sm_110a"] {
            assert!(
                generated_target_satisfied(&arch.parse().unwrap(), &llvm),
                "{} {arch}",
                target.id
            );
            assert!(
                generated_target_satisfied(&arch.parse().unwrap(), &libnvvm),
                "{} {arch}",
                target.id
            );
        }
        assert!(
            generated_target_satisfied(&"sm_101a".parse().unwrap(), &llvm),
            "{}",
            target.id
        );
        assert!(
            !generated_target_satisfied(&"sm_101a".parse().unwrap(), &libnvvm),
            "{}",
            target.id
        );
        assert!(
            !generated_target_satisfied(&"sm_120a".parse().unwrap(), &llvm),
            "{}",
            target.id
        );
        assert!(
            !generated_target_satisfied(&"sm_120a".parse().unwrap(), &libnvvm),
            "{}",
            target.id
        );
    }
}

#[test]
fn generated_packed_conversion_floors_require_ampere() {
    use crate::generated_intrinsic_targets::{
        GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
    };

    for marker in [
        "v1:i0071", "v1:i0081", "v1:i0082", "v1:i0083", "v1:i0084", "v1:i0085",
    ] {
        let target = generated_intrinsic_target_by_marker(marker).unwrap();
        for backend in [
            GeneratedIntrinsicBackend::LlvmNvptx,
            GeneratedIntrinsicBackend::LibNvvm,
        ] {
            let generated =
                GeneratedModuleRequirements::from_targets(vec![target]).for_backend(backend);
            assert_eq!(
                generated_ptx_isa_requirement(&generated).unwrap(),
                PtxIsaRequirement::new(70),
                "{marker} {backend:?}"
            );
            assert!(
                !generated_target_satisfied(&"sm_75".parse().unwrap(), &generated),
                "{marker}"
            );
            assert!(
                generated_target_satisfied(&"sm_80".parse().unwrap(), &generated),
                "{marker}"
            );
            let error =
                validate_generated_target(&"sm_75".parse().unwrap(), &generated).unwrap_err();
            assert!(error.contains(target.id), "{error}");
            assert!(error.contains("sm_80 or newer"), "{error}");
        }
    }
}

#[test]
fn generated_cp_async_floors_require_ampere() {
    use crate::generated_intrinsic_targets::{
        GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
    };

    for marker in [
        "v1:i0086", "v1:i0087", "v1:i0088", "v1:i0089", "v1:i0090", "v1:i0091", "v1:i0092",
        "v1:i0093", "v1:i0094", "v1:i0095", "v1:i0096", "v1:i0101", "v1:i0102", "v1:i0103",
        "v1:i0104",
    ] {
        let target = generated_intrinsic_target_by_marker(marker).unwrap();
        for backend in [
            GeneratedIntrinsicBackend::LlvmNvptx,
            GeneratedIntrinsicBackend::LibNvvm,
        ] {
            let generated =
                GeneratedModuleRequirements::from_targets(vec![target]).for_backend(backend);
            assert_eq!(
                generated_ptx_isa_requirement(&generated).unwrap(),
                PtxIsaRequirement::new(70),
                "{marker} {backend:?}"
            );
            assert!(
                !generated_target_satisfied(&"sm_75".parse().unwrap(), &generated),
                "{marker}"
            );
            assert!(
                generated_target_satisfied(&"sm_80".parse().unwrap(), &generated),
                "{marker}"
            );
            let error =
                validate_generated_target(&"sm_75".parse().unwrap(), &generated).unwrap_err();
            assert!(error.contains(target.id), "{error}");
            assert!(error.contains("sm_80 or newer"), "{error}");
        }
    }
}

#[test]
fn generated_dot_product_floors_record_sm61_and_split_backend_support() {
    use crate::generated_intrinsic_targets::{
        GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
    };

    for marker in ["v1:i0030", "v1:i0031", "v1:i0032", "v1:i0033"] {
        let target = generated_intrinsic_target_by_marker(marker).unwrap();
        let llvm = GeneratedModuleRequirements::from_targets(vec![target])
            .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
        assert!(matches!(
            llvm.requirement(target).hardware,
            GeneratedHardwareTarget::AnyOf(alternatives)
                if alternatives == [GeneratedHardwareAlternative::MinimumSm(61)]
        ));
        assert!(
            !generated_target_satisfied(&"sm_60".parse().unwrap(), &llvm),
            "{marker}"
        );
        assert!(
            generated_target_satisfied(&"sm_70".parse().unwrap(), &llvm),
            "{marker}"
        );
        assert_eq!(
            generated_ptx_isa_requirement(&llvm).unwrap(),
            PtxIsaRequirement::Default
        );

        let error = validate_generated_target(&"sm_60".parse().unwrap(), &llvm)
            .unwrap_err()
            .to_string();
        assert!(error.contains(target.id), "{error}");
        assert!(error.contains("sm_61 or newer"), "{error}");

        let libnvvm = GeneratedModuleRequirements::from_targets(vec![target])
            .for_backend(GeneratedIntrinsicBackend::LibNvvm);
        assert!(!generated_target_satisfied(
            &"sm_74".parse().unwrap(),
            &libnvvm
        ));
        assert!(generated_target_satisfied(
            &"sm_75".parse().unwrap(),
            &libnvvm
        ));
    }
}
