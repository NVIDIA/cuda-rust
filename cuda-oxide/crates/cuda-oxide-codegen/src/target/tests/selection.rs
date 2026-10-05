/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::target::arch::*;
use crate::target::detect::*;
use crate::target::features::*;
use crate::target::select::*;
use cuda_target_spec::recorded_ptx_floor;

#[test]
fn test_select_target_prefers_required_architecture() {
    for (features, expected) in [
        (DetectedFeatures::Blackwell, "sm_100a"),
        (DetectedFeatures::TmaCtaGroup, "sm_100a"),
        (DetectedFeatures::BlackwellAccelerated, "sm_100a"),
        (DetectedFeatures::BlackwellFamily, "sm_100a"),
        (DetectedFeatures::ReduxF32, "sm_100a"),
        (DetectedFeatures::MultimemFp8, "sm_100a"),
        (DetectedFeatures::TmaMulticast, "sm_100a"),
        (DetectedFeatures::MatrixBlackwell, "sm_100a"),
        (DetectedFeatures::Wgmma, "sm_90a"),
        (DetectedFeatures::Sm100, "sm_100"),
        (DetectedFeatures::Tma, "sm_100"),
        (DetectedFeatures::Cluster, "sm_90"),
        (DetectedFeatures::Sm90, "sm_90"),
        (DetectedFeatures::Sm80, "sm_80"),
        (DetectedFeatures::Movmatrix, "sm_75"),
        (DetectedFeatures::Ldmatrix, "sm_75"),
        (DetectedFeatures::Basic, "sm_80"),
    ] {
        assert_eq!(
            select_target(features).unwrap().sm(),
            expected,
            "{features:?}"
        );
    }
}

#[test]
fn target_selection_enforces_feature_intersections() {
    let multicast = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster";
    let hopper_pair = format!("{multicast};\nwgmma.fence.sync.aligned;");
    let hopper_requirements = detect_features_in_llvm_text(&hopper_pair);
    assert!(hopper_requirements.contains(DetectedFeatures::TmaMulticast));
    assert!(hopper_requirements.contains(DetectedFeatures::Wgmma));
    assert_eq!(select_target(hopper_requirements).unwrap().sm(), "sm_90a");
    assert!(arch_satisfies(
        &"sm_90a".parse().unwrap(),
        hopper_requirements
    ));
    assert!(!arch_satisfies(
        &"sm_100a".parse().unwrap(),
        hopper_requirements
    ));

    let blackwell_pair = format!(
        "{multicast};\n{}",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];"
    );
    let blackwell_requirements = detect_features_in_llvm_text(&blackwell_pair);
    assert!(blackwell_requirements.contains(DetectedFeatures::TmaMulticast));
    assert!(blackwell_requirements.contains(DetectedFeatures::MatrixBlackwell));
    assert_eq!(
        select_target(blackwell_requirements).unwrap().sm(),
        "sm_100a"
    );
    assert!(arch_satisfies(
        &"sm_100a".parse().unwrap(),
        blackwell_requirements
    ));
    assert!(!arch_satisfies(
        &"sm_90a".parse().unwrap(),
        blackwell_requirements
    ));

    let impossible = DetectedFeatures::Wgmma | DetectedFeatures::MatrixBlackwell;
    let error = select_target(impossible).expect_err("families have no common target");
    assert!(error.contains("do not share a compatible GPU architecture"));
    assert!(resolve_ptx_target(Some("sm_90a"), "CUDA_OXIDE_TARGET", None, impossible).is_err());
    assert!(resolve_ptx_target(Some("sm_100a"), "CUDA_OXIDE_TARGET", None, impossible).is_err());
}

#[test]
fn rejected_explicit_targets_name_the_source_that_chose_them() {
    let parse_failure = resolve_ptx_target(
        Some("not-a-target"),
        "PipelineConfig::target_arch",
        None,
        DetectedFeatures::Sm80,
    )
    .unwrap_err()
    .to_string();
    assert!(
        parse_failure.contains("invalid CUDA target `not-a-target`"),
        "{parse_failure}"
    );
    assert!(
        parse_failure.contains("(target from PipelineConfig::target_arch)"),
        "{parse_failure}"
    );
    assert_eq!(
        parse_failure.matches("invalid CUDA target").count(),
        1,
        "parse errors must not double-wrap the parser's own prefix: {parse_failure}"
    );

    let floor_rejection = resolve_ptx_target(
        Some("sm_75"),
        "CUDA_OXIDE_TARGET",
        None,
        DetectedFeatures::Sm80,
    )
    .unwrap_err()
    .to_string();
    assert!(
        floor_rejection.contains("cannot lower detected feature Sm80"),
        "{floor_rejection}"
    );
    assert!(
        floor_rejection.contains("(target from CUDA_OXIDE_TARGET)"),
        "{floor_rejection}"
    );
}

#[test]
fn test_arch_major_parses_cuda_spelling() {
    assert_eq!(arch_compute_capability("sm_75"), Some(75));
    assert_eq!(arch_compute_capability("sm_100a"), Some(100));
    assert_eq!(arch_major("sm_75"), Some(7));
    assert_eq!(arch_major("sm_80"), Some(8));
    assert_eq!(arch_major("sm_90a"), Some(9));
    assert_eq!(arch_major("sm_100a"), Some(10));
    assert_eq!(arch_major("sm_103a"), Some(10));
    assert_eq!(arch_major("sm_120a"), Some(12));
    assert_eq!(arch_major("nvvm-ir"), None);
    assert_eq!(arch_major("sm_"), None);
}

#[test]
fn ptx9_targets_require_an_llvm22_backend() {
    for target in ["sm_88", "sm_110", "sm_110a", "sm_110f"] {
        assert!(
            validate_target_for_llvm_major(&target.parse().unwrap(), Some(21)).is_err(),
            "{target}"
        );
        assert!(
            validate_target_for_llvm_major(&target.parse().unwrap(), None).is_err(),
            "{target}"
        );
        assert!(
            validate_target_for_llvm_major(&target.parse().unwrap(), Some(22)).is_ok(),
            "{target}"
        );
        assert!(
            validate_target_for_llvm_major(&target.parse().unwrap(), Some(23)).is_ok(),
            "{target}"
        );
    }
    for target in ["sm_87", "sm_103a", "sm_120a", "sm_121f"] {
        assert!(
            validate_target_for_llvm_major(&target.parse().unwrap(), Some(21)).is_ok(),
            "{target}"
        );
    }
    let target = "sm_999a";
    assert!(
        validate_target_for_llvm_major(&target.parse().unwrap(), Some(21)).is_ok(),
        "unknown target {target} must remain owned by other validators"
    );
    for (target, floor) in [
        ("sm_90a", 80),
        ("sm_100a", 86),
        ("sm_100f", 88),
        ("sm_120a", 87),
        ("sm_120f", 88),
        ("sm_121a", 88),
    ] {
        assert_eq!(recorded_ptx_floor(&target.parse().unwrap()), Ok(floor));
    }
}

#[test]
fn test_arch_satisfies_sm100_only_features() {
    // tcgen05 and explicit cta_group TMA are datacenter-Blackwell only:
    // consumer Blackwell (sm_120) and Hopper (sm_90) cannot run them, even
    // though 120 > 100. This is the gemm_sol regression guard.
    for f in [DetectedFeatures::Blackwell, DetectedFeatures::TmaCtaGroup] {
        assert!(
            arch_satisfies(&"sm_100a".parse().unwrap(), f),
            "sm_100a must satisfy {f:?}"
        );
        assert!(
            arch_satisfies(&"sm_103a".parse().unwrap(), f),
            "sm_103a must satisfy {f:?}"
        );
        assert!(
            arch_satisfies(&"sm_103f".parse().unwrap(), f),
            "sm_103f must satisfy {f:?}"
        );
        assert!(
            !arch_satisfies(&"sm_100".parse().unwrap(), f),
            "generic sm_100 must NOT satisfy {f:?}"
        );
        assert!(
            !arch_satisfies(&"sm_120a".parse().unwrap(), f),
            "sm_120a must NOT satisfy {f:?}"
        );
        assert!(
            !arch_satisfies(&"sm_90a".parse().unwrap(), f),
            "sm_90a must NOT satisfy {f:?}"
        );
        assert!(
            !arch_satisfies(&"sm_102a".parse().unwrap(), f),
            "unknown architecture-specific targets must not be accepted"
        );
        assert!(
            !arch_satisfies(&"sm_102f".parse().unwrap(), f),
            "unknown family-specific targets must not be accepted"
        );
    }
}

#[test]
fn test_arch_satisfies_base_tma_multicast_targets() {
    for arch in [
        "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103f", "sm_110a", "sm_120", "sm_120a",
    ] {
        assert!(
            arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::TmaMulticast),
            "{arch}"
        );
    }
    for arch in ["sm_80", "sm_89", "sm_102a", "sm_102f"] {
        assert!(
            !arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::TmaMulticast),
            "{arch}"
        );
    }
}

#[test]
fn test_arch_satisfies_wgmma_is_hopper_only() {
    assert!(arch_satisfies(
        &"sm_90a".parse().unwrap(),
        DetectedFeatures::Wgmma
    ));
    assert!(!arch_satisfies(
        &"sm_90".parse().unwrap(),
        DetectedFeatures::Wgmma
    ));
    assert!(!arch_satisfies(
        &"sm_100a".parse().unwrap(),
        DetectedFeatures::Wgmma
    ));
    assert!(!arch_satisfies(
        &"sm_120a".parse().unwrap(),
        DetectedFeatures::Wgmma
    ));
}

#[test]
fn test_arch_satisfies_blackwell_matrix_family_targets() {
    for arch in [
        "sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_121a", "sm_100f", "sm_103f", "sm_110f",
        "sm_120f", "sm_121f",
    ] {
        assert!(
            arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::MatrixBlackwell),
            "{arch}"
        );
    }
    for arch in [
        "sm_100a", "sm_101a", "sm_110a", "sm_120a", "sm_100f", "sm_101f", "sm_103f", "sm_110f",
        "sm_120f", "sm_121f",
    ] {
        assert!(
            arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::BlackwellFamily),
            "{arch}"
        );
    }
    for arch in ["sm_101a", "sm_101f"] {
        assert!(!arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::MatrixBlackwell
        ));
    }
    for arch in ["sm_103a", "sm_121a"] {
        assert!(!arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::BlackwellFamily
        ));
    }
    for arch in [
        "sm_100a", "sm_101a", "sm_103a", "sm_110a", "sm_100f", "sm_103f", "sm_110f",
    ] {
        assert!(
            arch_satisfies(
                &arch.parse().unwrap(),
                DetectedFeatures::BlackwellAccelerated
            ),
            "{arch}"
        );
    }
    for arch in ["sm_100", "sm_120a", "sm_120f", "sm_102f"] {
        assert!(
            !arch_satisfies(
                &arch.parse().unwrap(),
                DetectedFeatures::BlackwellAccelerated
            ),
            "{arch}"
        );
    }
    for arch in ["sm_100", "sm_103", "sm_110", "sm_120", "sm_121a"] {
        assert!(
            arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::Sm100),
            "{arch}"
        );
    }
    for arch in ["sm_90a", "sm_102", "sm_102a"] {
        assert!(
            !arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::Sm100),
            "{arch}"
        );
    }
    for arch in ["sm_90a", "sm_100", "sm_102f", "sm_120"] {
        assert!(
            !arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::MatrixBlackwell),
            "{arch}"
        );
        assert!(
            !arch_satisfies(&arch.parse().unwrap(), DetectedFeatures::BlackwellFamily),
            "{arch}"
        );
    }
}

#[test]
fn test_arch_satisfies_forward_compatible_features() {
    // Plain TMA / cluster / sm_90-floor instructions lower on any sm_90+
    // device, sm_80-floor instructions on any sm_80+ device, movmatrix and
    // base ldmatrix on sm_75+, and basic kernels on Volta+.
    // So a consumer sm_120 GPU is a valid target for these (it runs locally
    // instead of being downgraded to the feature floor).
    for arch in ["sm_90a", "sm_100a", "sm_120a"] {
        assert!(arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::Tma
        ));
        assert!(arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::Cluster
        ));
        assert!(arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::Sm90
        ));
        assert!(arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::Sm80
        ));
        assert!(arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::Movmatrix
        ));
        assert!(arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::Ldmatrix
        ));
        assert!(arch_satisfies(
            &arch.parse().unwrap(),
            DetectedFeatures::Basic
        ));
    }
    assert!(arch_satisfies(
        &"sm_80".parse().unwrap(),
        DetectedFeatures::Sm80
    ));
    assert!(!arch_satisfies(
        &"sm_75".parse().unwrap(),
        DetectedFeatures::Sm80
    ));
    assert!(arch_satisfies(
        &"sm_75".parse().unwrap(),
        DetectedFeatures::Movmatrix
    ));
    assert!(arch_satisfies(
        &"sm_80".parse().unwrap(),
        DetectedFeatures::Movmatrix
    ));
    assert!(!arch_satisfies(
        &"sm_70".parse().unwrap(),
        DetectedFeatures::Movmatrix
    ));
    assert!(arch_satisfies(
        &"sm_75".parse().unwrap(),
        DetectedFeatures::Ldmatrix
    ));
    assert!(!arch_satisfies(
        &"sm_70".parse().unwrap(),
        DetectedFeatures::Ldmatrix
    ));
    assert!(arch_satisfies(
        &"sm_80".parse().unwrap(),
        DetectedFeatures::Basic
    ));
    assert!(arch_satisfies(
        &"sm_75".parse().unwrap(),
        DetectedFeatures::Basic
    ));
    assert!(arch_satisfies(
        &"sm_70".parse().unwrap(),
        DetectedFeatures::Basic
    ));
    assert!(!arch_satisfies(
        &"sm_80".parse().unwrap(),
        DetectedFeatures::Tma
    ));
    assert!(!arch_satisfies(
        &"sm_80".parse().unwrap(),
        DetectedFeatures::Sm90
    ));
    assert!(!arch_satisfies(
        &"sm_80a".parse().unwrap(),
        DetectedFeatures::Basic
    ));
    assert!(!arch_satisfies(
        &"sm_90f".parse().unwrap(),
        DetectedFeatures::Tma
    ));
}

#[test]
fn resolve_ptx_target_threads_a_caller_supplied_source_label() {
    let (target, source) = resolve_ptx_target(
        Some("sm_80"),
        "the requested Target",
        None,
        DetectedFeatures::Basic,
    )
    .unwrap();
    assert_eq!(target.sm(), "sm_80");
    assert_eq!(source, "the requested Target");
}

#[test]
fn valid_device_hints_and_explicit_compute_targets_preserve_selection() {
    let hint = "sm_120".parse::<cuda_target_spec::DeviceArch>().unwrap();
    for (explicit, hint, expected, source) in [
        (None, Some(&hint), "sm_120", "detected GPU"),
        (None, None, "sm_80", "feature requirement"),
        (Some("compute_80"), Some(&hint), "sm_80", "test override"),
        (Some("compute_120f"), None, "sm_120f", "test override"),
        (Some("sm_120f"), None, "sm_120f", "test override"),
    ] {
        let (ptx, provenance) =
            resolve_ptx_target(explicit, "test override", hint, DetectedFeatures::Basic).unwrap();
        assert_eq!(ptx.sm(), expected);
        assert_eq!(provenance, source);
        let nvvm =
            crate::export::resolve_nvvm_target(explicit, hint, Some(DetectedFeatures::Basic))
                .unwrap();
        assert_eq!(nvvm.sm(), expected);
    }
}

#[test]
fn resolve_ptx_target_failure_does_not_assume_an_env_var_source() {
    let error = resolve_ptx_target(
        Some("sm_75"),
        "the requested Target",
        None,
        DetectedFeatures::Wgmma,
    )
    .unwrap_err();
    assert!(matches!(error, PipelineError::TargetSelection { .. }));
    assert!(
        !error.to_string().contains("CUDA_OXIDE_TARGET"),
        "a standalone caller that never set CUDA_OXIDE_TARGET should not see it blamed: {error}"
    );

    let parse_error = resolve_ptx_target(
        Some("not-a-target"),
        "the requested Target",
        None,
        DetectedFeatures::Basic,
    )
    .unwrap_err();
    assert!(matches!(parse_error, PipelineError::TargetSelection { .. }));
    assert!(
        !parse_error.to_string().contains("CUDA_OXIDE_TARGET"),
        "{parse_error}"
    );
}

#[test]
fn malformed_low_width_targets_keep_override_errors() {
    for (spelling, expected_reason) in [
        (
            "sm_05",
            "invalid CUDA target `sm_5`: compute capability must contain at least two digits (target from test override)",
        ),
        (
            "sm_",
            "invalid CUDA target `sm_`: compute capability is not a valid integer (target from test override)",
        ),
    ] {
        let explicit = resolve_ptx_target(
            Some(spelling),
            "test override",
            None,
            DetectedFeatures::Basic,
        )
        .unwrap_err();
        assert!(matches!(explicit, PipelineError::TargetSelection { .. }));
        assert_eq!(explicit.to_string(), expected_reason);
    }
}
