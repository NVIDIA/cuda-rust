/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::generated::GeneratedModuleRequirements;
use crate::target::arch::*;
use crate::target::detect::*;
use crate::target::features::*;
use crate::target::generated_requirements::*;
use crate::target::select::*;
use libnvvm_sys::CudaArch;

#[test]
fn test_feature_detection_reads_llvm_ir_snippets() {
    let llvm = r#"
            call void asm sideeffect "wgmma.fence.sync.aligned", ""()
            call void @llvm.nvvm.tcgen05.alloc()
            call void asm sideeffect "cluster.sync.aligned", ""()
            call void asm sideeffect "cp.async.bulk.tensor.2d.shared::cluster.global", ""()
            call void asm sideeffect "cp.async.ca.shared.global", ""()
        "#;

    assert!(contains_wgmma_features(llvm));
    assert!(contains_blackwell_features(llvm));
    assert!(contains_cluster_features(llvm));
    assert!(contains_tma_features(llvm));
    assert!(contains_sm80_features(llvm));
    let detected = detect_features_in_llvm_text(llvm);
    for feature in [
        DetectedFeatures::Blackwell,
        DetectedFeatures::Wgmma,
        DetectedFeatures::Cluster,
        DetectedFeatures::Tma,
        DetectedFeatures::Sm80,
    ] {
        assert!(detected.contains(feature), "missing {feature:?}");
    }
    assert!(
        select_target(detected).is_err(),
        "Hopper-only WGMMA and Blackwell-only tcgen05 are incompatible"
    );
}

#[test]
fn test_sm80_detection_accepts_inline_ptx_and_nvvm_intrinsics() {
    for llvm in [
        r#"call void asm sideeffect "cp.async.ca.shared.global [%0], [%1], 4;", "l,l"()"#,
        "call void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %dst, ptr addrspace(1) %src)",
        r#"call void asm sideeffect "cp.async.commit_group;", ""()"#,
        "call void @llvm.nvvm.cp.async.wait.all()",
    ] {
        assert!(contains_sm80_features(llvm), "missed cp.async in {llvm}");
        assert_eq!(detect_features_in_llvm_text(llvm), DetectedFeatures::Sm80);
    }
}

#[test]
fn test_bf16x2_detection_matches_exact_architecture_floors() {
    for mnemonic in [
        "add.rn.bf16x2 $0, $1, $2;",
        "sub.rn.bf16x2 $0, $1, $2;",
        "mul.rn.bf16x2 $0, $1, $2;",
    ] {
        assert!(contains_sm90_features(mnemonic));
        assert!(!contains_sm80_features(mnemonic));
        assert_eq!(
            detect_features_in_llvm_text(mnemonic),
            DetectedFeatures::Sm90
        );
    }

    for mnemonic in ["add.rn.bf16x2\t$0, $1, $2;", "sub.rn.bf16x2\\09$0, $1, $2;"] {
        assert_eq!(
            detect_features_in_llvm_text(mnemonic),
            DetectedFeatures::Sm90,
            "{mnemonic:?}"
        );
    }

    for mnemonic in [
        "fma.rn.bf16x2 $0, $1, $2, $3;",
        "fma.rn.relu.bf16x2 $0, $1, $2, $3;",
        "min.bf16x2 $0, $1, $2;",
        "max.bf16x2 $0, $1, $2;",
        "neg.bf16x2 $0, $1;",
        "abs.bf16x2 $0, $1;",
    ] {
        assert!(!contains_sm90_features(mnemonic));
        assert!(contains_sm80_features(mnemonic));
        assert_eq!(
            detect_features_in_llvm_text(mnemonic),
            DetectedFeatures::Sm80
        );
    }

    for near_miss in [
        "add.rn.bf16x2x $0, $1, $2;",
        "fma.rn.bf16x2x $0, $1, $2, $3;",
        "add.rn.bf16x2\\5C09$0, $1, $2;",
    ] {
        assert!(!contains_sm90_features(near_miss));
        assert!(!contains_sm80_features(near_miss));
        assert_eq!(
            detect_features_in_llvm_text(near_miss),
            DetectedFeatures::Basic
        );
    }
}

#[test]
fn f32x2_family_detection_requires_sm100_and_ptx86() {
    for mnemonic in [
        "add.f32x2 $0, $1, $2;",
        "add.rz.f32x2 $0, $1, $2;",
        "sub.rm.f32x2 $0, $1, $2;",
        "mul.rp.ftz.f32x2 $0, $1, $2;",
        "fma.rn.f32x2 $0, $1, $2, $3;",
        "fma.rz.ftz.f32x2\\09$0, $1, $2, $3;",
    ] {
        assert!(contains_f32x2_features(mnemonic));
        let requirements = detect_module_requirements_in_llvm_text(mnemonic);
        assert_eq!(requirements.features, DetectedFeatures::Sm100);
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86));
        assert!(!arch_satisfies(
            &"sm_90".parse().unwrap(),
            requirements.features
        ));
        assert!(arch_satisfies(
            &"sm_100".parse().unwrap(),
            requirements.features
        ));
        assert!(arch_satisfies(
            &"sm_120".parse().unwrap(),
            requirements.features
        ));
    }

    for near_miss in [
        "add.rn.f32x2x $0, $1, $2;",
        "fmax.rn.f32x2 $0, $1, $2;",
        "myadd.rn.f32x2 $0, $1, $2;",
        "add.rn.f32 $0, $1, $2;",
    ] {
        assert!(!contains_f32x2_features(near_miss));
    }
}

#[test]
fn f32x2_family_detection_respects_the_first_token_delimiter() {
    for operation in ["add", "sub", "mul", "fma"] {
        for delimiter in [
            " ", "\t", "\n", "\u{2003}", "\"", ";", "\\09", "\\0A", "\\0B", "\\0C", "\\0D",
        ] {
            let present = format!("{operation}.rn.f32x2{delimiter}$0, $1; tail\\09.f32");
            assert!(contains_f32x2_features(&present), "{present:?}");

            let absent = format!("{operation}.rn.f32{delimiter}.f32x2 tail\\09.f32x2");
            assert!(!contains_f32x2_features(&absent), "{absent:?}");
        }
    }
}

#[test]
fn f32x2_family_detection_handles_large_modules_and_late_matches() {
    // Most tokens have ordinary delimiters and no escaped whitespace in the
    // remaining module, which used to trigger repeated full-suffix scans.
    let mut module = "call void asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"\"()\n".repeat(8192);
    assert!(!contains_f32x2_features(&module));
    module.push_str("call void asm sideeffect \"fma.rn.f32x2\\09$0, $1, $2, $3;\", \"\"()\n");
    let requirements = detect_module_requirements_in_llvm_text(&module);
    assert_eq!(requirements.features, DetectedFeatures::Sm100);
    assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86));
}

#[test]
fn dense_bf16_mma_detection_applies_exact_sm80_and_ptx70_floors() {
    let mnemonic = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};";
    for spelling in [
        mnemonic,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\t{$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\09{$0}, {$1}, {$2}, {$3};",
        ";mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "prefix\\0Amma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "{mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "$L:mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "/* comment */mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "@p mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "@!%p\\09mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            contains_mma_m16n8k16_f32_bf16_features(spelling),
            "missed {spelling:?}"
        );
    }

    let requirements = detect_module_requirements_in_llvm_text(mnemonic);
    assert_eq!(
        requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    assert_eq!(select_target(requirements.features).unwrap().sm(), "sm_80");

    let lower_target = resolve_ptx_target(
        Some("sm_75"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .unwrap_err();
    assert!(
        lower_target
            .to_string()
            .contains("cannot lower detected feature Sm80"),
        "{lower_target}"
    );
    let (target, _) = resolve_ptx_target(
        Some("sm_80"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .unwrap();
    assert_eq!(target.sm(), "sm_80");

    for near_miss in [
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32x {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "@mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "!mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "@!mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "not$mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "/mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        ")mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_mma_m16n8k16_f32_bf16_features(near_miss),
            "matched {near_miss:?}"
        );
    }

    let combined = format!(
        "{mnemonic}\n{}",
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
    );
    assert_eq!(
        detect_module_requirements_in_llvm_text(&combined),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
}

#[test]
fn packed_atomic_detection_enforces_native_architecture_and_ptx_floors() {
    for f16 in [
        "atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "atom.global.add.noftz.f16x2\t$0, [$1], $2;",
        "atom.global.add.noftz.f16x2\\09$0, [$1], $2;",
        ";atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "prefix\\0Aatom.global.add.noftz.f16x2 $0, [$1], $2;",
        "\"atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "{atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "$L:atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "/* comment */atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "@p atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "@!%p\\09atom.global.add.noftz.f16x2 $0, [$1], $2;",
    ] {
        assert!(contains_packed_f16_atomic_features(f16), "{f16:?}");
        assert!(!contains_packed_bf16_atomic_features(f16), "{f16:?}");
        assert_eq!(detect_features_in_llvm_text(f16), DetectedFeatures::Basic);
        assert_eq!(
            detect_module_requirements_in_llvm_text(f16).ptx_isa,
            PtxIsaRequirement::new(62)
        );
    }
    assert_eq!(
        required_ptx_feature(&"sm_70".parse().unwrap(), PtxIsaRequirement::new(62)).unwrap(),
        Some("+ptx62")
    );
    assert_eq!(
        resolve_ptx_target(
            Some("sm_70"),
            "CUDA_OXIDE_TARGET",
            None,
            DetectedFeatures::Basic
        )
        .unwrap()
        .0
        .sm(),
        "sm_70"
    );

    for bf16 in [
        "atom.global.add.noftz.bf16x2 $0, [$1], $2;",
        "atom.global.add.noftz.bf16x2\t$0, [$1], $2;",
        "atom.global.add.noftz.bf16x2\\0A$0, [$1], $2;",
    ] {
        assert!(contains_packed_bf16_atomic_features(bf16), "{bf16:?}");
        assert!(!contains_packed_f16_atomic_features(bf16), "{bf16:?}");
        assert_eq!(detect_features_in_llvm_text(bf16), DetectedFeatures::Sm90);
        assert_eq!(
            detect_module_requirements_in_llvm_text(bf16).ptx_isa,
            PtxIsaRequirement::new(78)
        );
    }
    assert_eq!(select_target(DetectedFeatures::Sm90).unwrap().sm(), "sm_90");
    let rejected = resolve_ptx_target(
        Some("sm_80"),
        "CUDA_OXIDE_TARGET",
        None,
        DetectedFeatures::Sm90,
    )
    .expect_err("native bf16x2 atomic add must reject sm_80")
    .to_string();
    assert!(rejected.contains("cannot lower detected feature Sm90"));
    let near_miss = resolve_ptx_target(
        Some("sm_89"),
        "CUDA_OXIDE_TARGET",
        None,
        DetectedFeatures::Sm90,
    )
    .expect_err("the architecture immediately below sm_90 must be rejected")
    .to_string();
    assert!(near_miss.contains("cannot lower detected feature Sm90"));

    let both = "atom.global.add.noftz.f16x2 $0, [$1], $2; \
                    atom.global.add.noftz.bf16x2 $0, [$1], $2;";
    let requirements = detect_module_requirements_in_llvm_text(both);
    assert_eq!(requirements.features, DetectedFeatures::Sm90);
    assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(78));

    let dense_bf16_mma =
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};";
    let mma_f16_requirements = detect_module_requirements_in_llvm_text(&format!(
        "{dense_bf16_mma}\natom.global.add.noftz.f16x2 $0, [$1], $2;"
    ));
    assert_eq!(
        mma_f16_requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    assert_eq!(
        select_target(mma_f16_requirements.features).unwrap().sm(),
        "sm_80"
    );

    let mma_bf16_requirements = detect_module_requirements_in_llvm_text(&format!(
        "{dense_bf16_mma}\natom.global.add.noftz.bf16x2 $0, [$1], $2;"
    ));
    assert_eq!(
        mma_bf16_requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm90 | DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
    assert_eq!(
        select_target(mma_bf16_requirements.features).unwrap().sm(),
        "sm_90"
    );

    for near_miss in [
        "atom.global.add.noftz.f16x2x $0, [$1], $2;",
        "atom.global.add.noftz.bf16x2x $0, [$1], $2;",
        "not_atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "not_atom.global.add.noftz.bf16x2 $0, [$1], $2;",
        "not.atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "not.atom.global.add.noftz.bf16x2 $0, [$1], $2;",
        "$atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "%atom.global.add.noftz.bf16x2 $0, [$1], $2;",
        "@atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "!atom.global.add.noftz.bf16x2 $0, [$1], $2;",
        "@!atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "not$atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "/atom.global.add.noftz.bf16x2 $0, [$1], $2;",
        ")atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "atom.shared.add.noftz.f16x2 $0, [$1], $2;",
        "atom.global.add.bf16x2 $0, [$1], $2;",
        "red.global.add.noftz.bf16x2 [$0], $1;",
        "atom.global.add.noftz.f16x2\\5C09$0, [$1], $2;",
    ] {
        assert!(!contains_packed_f16_atomic_features(near_miss));
        assert!(!contains_packed_bf16_atomic_features(near_miss));
        assert_eq!(
            detect_module_requirements_in_llvm_text(near_miss),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            },
            "{near_miss:?}"
        );
    }
}

#[test]
fn fp64_mma_and_packed_atomics_take_the_strongest_target_floor() {
    let dense_bf16_mma =
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};";
    let dense_fp64_mma =
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};";

    let fp64_f16_requirements = detect_module_requirements_in_llvm_text(&format!(
        "{dense_fp64_mma}\natom.global.add.noftz.f16x2 $0, [$1], $2;"
    ));
    assert_eq!(
        fp64_f16_requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    assert_eq!(
        select_target(fp64_f16_requirements.features).unwrap().sm(),
        "sm_80"
    );

    let fp64_bf16_requirements = detect_module_requirements_in_llvm_text(&format!(
        "{dense_fp64_mma}\natom.global.add.noftz.bf16x2 $0, [$1], $2;"
    ));
    assert_eq!(
        fp64_bf16_requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm90 | DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
    assert_eq!(
        select_target(fp64_bf16_requirements.features).unwrap().sm(),
        "sm_90"
    );

    let all_four = format!(
        "{dense_bf16_mma}\n{dense_fp64_mma}\n\
             atom.global.add.noftz.f16x2 $0, [$1], $2;\n\
             atom.global.add.noftz.bf16x2 $0, [$1], $2;"
    );
    let all_four_requirements = detect_module_requirements_in_llvm_text(&all_four);
    assert_eq!(
        all_four_requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm90 | DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
    assert_eq!(
        select_target(all_four_requirements.features).unwrap().sm(),
        "sm_90"
    );
}

#[test]
fn dense_f16_mma_detection_applies_exact_sm80_and_ptx70_floors() {
    let mnemonic = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};";
    for spelling in [
        mnemonic,
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\t{$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\09{$0}, {$1}, {$2}, {$3};",
        ";mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "prefix\\0Amma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "{mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "$L:mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "/* comment */mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "@p mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "@!%p\\09mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            contains_mma_m16n8k16_f32_f16_features(spelling),
            "missed {spelling:?}"
        );
    }

    let requirements = detect_module_requirements_in_llvm_text(mnemonic);
    assert_eq!(
        requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    assert_eq!(select_target(requirements.features).unwrap().sm(), "sm_80");

    let lower_target = resolve_ptx_target(
        Some("sm_75"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .unwrap_err();
    assert!(
        lower_target
            .to_string()
            .contains("cannot lower detected feature Sm80"),
        "{lower_target}"
    );
    let (target, _) = resolve_ptx_target(
        Some("sm_80"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .unwrap();
    assert_eq!(target.sm(), "sm_80");

    for near_miss in [
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32x {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "@mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "!mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "@!mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "not$mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "/mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        ")mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_mma_m16n8k16_f32_f16_features(near_miss),
            "matched {near_miss:?}"
        );
    }

    let combined = format!(
        "{mnemonic}\n{}",
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
    );
    assert_eq!(
        detect_module_requirements_in_llvm_text(&combined),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
}

#[test]
fn tf32_mma_detection_applies_exact_sm80_and_ptx70_floors() {
    let mnemonic = concat!(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    );
    for spelling in [
        mnemonic,
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\t{$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\\09{$0}, {$1}, {$2}, {$3};",
        ";mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "prefix\\0Amma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "{mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "$L:mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "/* comment */mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "@p mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "@!%p\\09mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            contains_mma_m16n8k8_f32_tf32_features(spelling),
            "missed {spelling:?}"
        );
    }

    let requirements = detect_module_requirements_in_llvm_text(mnemonic);
    assert_eq!(requirements.features, DetectedFeatures::Sm80);
    assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(70));
    let (target, _) = resolve_ptx_target(None, "CUDA_OXIDE_TARGET", None, requirements.features)
        .expect("auto-resolve");
    assert_eq!(target.sm(), "sm_80");

    for near_miss in [
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32x {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "@mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "!mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "@!mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "not$mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        "/mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        ")mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_mma_m16n8k8_f32_tf32_features(near_miss),
            "matched {near_miss:?}"
        );
    }

    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_75, requirements.features).is_err());
    assert!(validate_target_features(&sm_80, requirements.features).is_ok());
    let error = resolve_ptx_target(
        Some("sm_75"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .expect_err("sm_75 must not accept TF32 tensor-core MMA")
    .to_string();
    assert!(
        error.contains("cannot lower detected feature Sm80"),
        "{error}"
    );

    let combined = format!("{mnemonic}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&combined),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
}

#[test]
fn int8_mma_detection_applies_exact_sm80_and_ptx70_floors() {
    let mut forms = 0;
    for shape in ["m16n8k16", "m16n8k32"] {
        for a_type in ["s8", "u8"] {
            for b_type in ["s8", "u8"] {
                for satfinite in [false, true] {
                    let overflow = if satfinite { ".satfinite" } else { "" };
                    let spelling = format!(
                        "mma.sync.aligned.{shape}.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0}}, {{$1}}, {{$2}}, {{$3}};"
                    );
                    assert!(
                        contains_dense_int8_mma_features(&spelling),
                        "missed {spelling:?}"
                    );
                    assert_eq!(
                        detect_module_requirements_in_llvm_text(&spelling),
                        ModuleRequirements {
                            features: DetectedFeatures::Sm80,
                            ptx_isa: PtxIsaRequirement::new(70),
                        },
                        "{spelling}"
                    );
                    forms += 1;
                }
            }
        }
    }
    assert_eq!(forms, 16);

    for spelling in [
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32\t{$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32\\09{$0}, {$1}, {$2}, {$3};",
        ";mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {$0}, {$1}, {$2}, {$3};",
        "prefix\\0Amma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 {$0}, {$1}, {$2}, {$3};",
        "@p mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {$0}, {$1}, {$2}, {$3};",
        "@!%p\\09mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            contains_dense_int8_mma_features(spelling),
            "missed {spelling:?}"
        );
    }

    let representative = concat!(
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32 ",
        "{$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};"
    );
    let requirements = detect_module_requirements_in_llvm_text(representative);
    let (target, _) = resolve_ptx_target(None, "CUDA_OXIDE_TARGET", None, requirements.features)
        .expect("auto-resolve");
    assert_eq!(target.sm(), "sm_80");

    for near_miss in [
        "mma.sync.aligned.m16n8k8.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.col.row.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.satfiniteX.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.u32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.f16.s32 {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "@mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "!mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "@!mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "not$mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "/mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        ")mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_dense_int8_mma_features(near_miss),
            "matched {near_miss:?}"
        );
    }

    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_75, requirements.features).is_err());
    assert!(validate_target_features(&sm_80, requirements.features).is_ok());
    let error = resolve_ptx_target(
        Some("sm_75"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .expect_err("sm_75 must not accept INT8 tensor-core MMA")
    .to_string();
    assert!(
        error.contains("cannot lower detected feature Sm80"),
        "{error}"
    );

    let combined = format!("{representative}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&combined),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
}

#[test]
fn dense_int4_mma_detection_applies_exact_sm80_and_ptx70_floors() {
    let mut forms = 0;
    for shape in ["m16n8k32", "m16n8k64"] {
        for a_type in ["s4", "u4"] {
            for b_type in ["s4", "u4"] {
                for satfinite in [false, true] {
                    let overflow = if satfinite { ".satfinite" } else { "" };
                    let spelling = format!(
                        "mma.sync.aligned.{shape}.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0}}, {{$1}}, {{$2}}, {{$3}};"
                    );
                    assert!(
                        contains_dense_int4_mma_features(&spelling),
                        "missed {spelling:?}"
                    );
                    assert!(
                        !contains_mma_m8n8k32_int4_features(&spelling),
                        "m16 form entered the m8 INT4 detector: {spelling:?}"
                    );
                    assert!(
                        !contains_dense_int8_mma_features(&spelling),
                        "INT4 form entered the dense INT8 detector: {spelling:?}"
                    );
                    assert_eq!(
                        detect_module_requirements_in_llvm_text(&spelling),
                        ModuleRequirements {
                            features: DetectedFeatures::Sm80,
                            ptx_isa: PtxIsaRequirement::new(70),
                        },
                        "{spelling}"
                    );
                    forms += 1;
                }
            }
        }
    }
    assert_eq!(forms, 16);

    for spelling in [
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.u4.s32\t{$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32\\09{$0}, {$1}, {$2}, {$3};",
        ";mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 {$0}, {$1}, {$2}, {$3};",
        "prefix\\0Amma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32 {$0}, {$1}, {$2}, {$3};",
        "@p mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 {$0}, {$1}, {$2}, {$3};",
        "@!%p\\09mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.s4.s32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            contains_dense_int4_mma_features(spelling),
            "missed {spelling:?}"
        );
    }

    let representative = concat!(
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.u4.s32 ",
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    );
    let requirements = detect_module_requirements_in_llvm_text(representative);
    assert_eq!(
        requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    assert_eq!(select_target(requirements.features).unwrap().sm(), "sm_80");
    assert_eq!(
        required_ptx_feature(&"sm_75".parse().unwrap(), requirements.ptx_isa).unwrap(),
        Some("+ptx70")
    );
    assert_eq!(
        required_ptx_feature(&"sm_80".parse().unwrap(), requirements.ptx_isa).unwrap(),
        None
    );

    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_75, requirements.features).is_err());
    assert!(validate_target_features(&sm_80, requirements.features).is_ok());
    let error = resolve_ptx_target(
        Some("sm_75"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .expect_err("sm_75 must not accept dense m16 INT4 MMA")
    .to_string();
    assert!(
        error.contains("cannot lower detected feature Sm80"),
        "{error}"
    );

    for target in [
        "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_120", "sm_120a",
    ] {
        assert!(
            arch_satisfies(&target.parse().unwrap(), requirements.features),
            "rejected {target}"
        );
    }

    let m8_int4 = concat!(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 ",
        "{$0, $1}, {$4}, {$5}, {$2, $3};"
    );
    let mixed = format!("{representative}\n{m8_int4}");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&mixed),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );

    let newer_ptx = format!("{representative}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&newer_ptx),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
}

#[test]
fn dense_int4_mma_detection_rejects_other_mma_families_and_near_misses() {
    for near_miss in [
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "wmma.mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.col.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32.satfinite {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.col.s32.satfinite.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.satfiniteX.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.u32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32x {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "@mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "!mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        ")mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_dense_int4_mma_features(near_miss),
            "matched {near_miss:?}"
        );
    }
}

#[test]
fn dense_b1_mma_detection_applies_exact_operation_floors() {
    let cases = [
        (
            "m8n8k128",
            "xor",
            DetectedFeatures::Sm75,
            PtxIsaRequirement::new(70),
        ),
        (
            "m16n8k128",
            "xor",
            DetectedFeatures::Sm80,
            PtxIsaRequirement::new(70),
        ),
        (
            "m16n8k256",
            "xor",
            DetectedFeatures::Sm80,
            PtxIsaRequirement::new(70),
        ),
        (
            "m8n8k128",
            "and",
            DetectedFeatures::Sm80,
            PtxIsaRequirement::new(71),
        ),
        (
            "m16n8k128",
            "and",
            DetectedFeatures::Sm80,
            PtxIsaRequirement::new(71),
        ),
        (
            "m16n8k256",
            "and",
            DetectedFeatures::Sm80,
            PtxIsaRequirement::new(71),
        ),
    ];

    for (shape, operation, features, ptx_isa) in cases {
        let spelling = format!(
            "mma.sync.aligned.{shape}.row.col.s32.b1.b1.s32.{operation}.popc {{$0}}, {{$1}}, {{$2}}, {{$3}};"
        );
        assert_eq!(contains_b1_xor_mma_features(&spelling), operation == "xor");
        assert_eq!(contains_b1_and_mma_features(&spelling), operation == "and");
        assert_eq!(
            contains_mma_m8n8k128_b1_xor_features(&spelling),
            shape == "m8n8k128" && operation == "xor"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(&spelling),
            ModuleRequirements { features, ptx_isa },
            "{spelling}"
        );
    }

    let m8_xor = concat!(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc ",
        "{$0, $1}, {$4}, {$5}, {$2, $3};"
    );
    let m8_requirements = detect_module_requirements_in_llvm_text(m8_xor);
    assert_eq!(
        select_target(m8_requirements.features).unwrap().sm(),
        "sm_75"
    );
    assert_eq!(
        required_ptx_feature(&"sm_75".parse().unwrap(), m8_requirements.ptx_isa).unwrap(),
        Some("+ptx70")
    );

    let m16_and = concat!(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc ",
        "{$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};"
    );
    let and_requirements = detect_module_requirements_in_llvm_text(m16_and);
    assert_eq!(
        select_target(and_requirements.features).unwrap().sm(),
        "sm_80"
    );
    assert_eq!(
        required_ptx_feature(&"sm_80".parse().unwrap(), and_requirements.ptx_isa).unwrap(),
        Some("+ptx71")
    );
    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_75, and_requirements.features).is_err());
    assert!(validate_target_features(&sm_80, and_requirements.features).is_ok());

    let combined = format!("{m8_xor}\n{m16_and}");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&combined),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
            ptx_isa: PtxIsaRequirement::new(71),
        }
    );
}

#[test]
fn dense_b1_mma_detection_rejects_other_families_and_near_misses() {
    for near_miss in [
        "wmma.mma.xor.popc.sync.aligned.row.col.m8n8k128.s32.b1.b1.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k64.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.col.row.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.or.popc {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.popc.xor {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popcx {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc.satfinite {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.u32.xor.popc {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_b1_xor_mma_features(near_miss),
            "matched {near_miss:?}"
        );
        assert!(
            !contains_b1_and_mma_features(near_miss),
            "matched {near_miss:?}"
        );
    }
}

#[test]
fn generated_b1_floors_match_text_detection_on_both_backends() {
    use crate::generated_intrinsic_targets::{
        GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
    };

    for (marker, mnemonic) in [
        ("v1:i0157", B1_XOR_MMA_MNEMONICS[0]),
        ("v1:i0158", B1_XOR_MMA_MNEMONICS[1]),
        ("v1:i0159", B1_XOR_MMA_MNEMONICS[2]),
        ("v1:i0160", B1_AND_MMA_MNEMONICS[0]),
        ("v1:i0161", B1_AND_MMA_MNEMONICS[1]),
        ("v1:i0162", B1_AND_MMA_MNEMONICS[2]),
    ] {
        let instruction = format!("{mnemonic} {{$0}}, {{$1}}, {{$2}}, {{$3}};");
        let detected = detect_module_requirements_in_llvm_text(&instruction);
        let target = generated_intrinsic_target_by_marker(marker).unwrap();
        for backend in [
            GeneratedIntrinsicBackend::LlvmNvptx,
            GeneratedIntrinsicBackend::LibNvvm,
        ] {
            let generated =
                GeneratedModuleRequirements::from_targets(vec![target]).for_backend(backend);
            assert_eq!(
                generated_ptx_isa_requirement(&generated).unwrap(),
                detected.ptx_isa,
                "{marker} {backend:?}"
            );
            for arch in ["sm_70", "sm_75", "sm_80", "sm_90"] {
                assert_eq!(
                    generated_target_satisfied(&arch.parse().unwrap(), &generated),
                    arch_satisfies(&arch.parse().unwrap(), detected.features),
                    "{marker} {backend:?} {arch}"
                );
            }
        }
    }
}

#[test]
fn m8n8k16_int8_mma_detection_applies_exact_sm75_and_ptx65_floors() {
    let mut forms = 0;
    for a_type in ["s8", "u8"] {
        for b_type in ["s8", "u8"] {
            for satfinite in [false, true] {
                let overflow = if satfinite { ".satfinite" } else { "" };
                let spelling = format!(
                    "mma.sync.aligned.m8n8k16.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0, $1}}, {{$2}}, {{$3}}, {{$4, $5}};"
                );
                assert!(
                    contains_mma_m8n8k16_int8_features(&spelling),
                    "missed {spelling:?}"
                );
                assert!(
                    !contains_dense_int8_mma_features(&spelling),
                    "m8 form entered the m16 detector: {spelling:?}"
                );
                assert_eq!(
                    detect_module_requirements_in_llvm_text(&spelling),
                    ModuleRequirements {
                        features: DetectedFeatures::Sm75,
                        ptx_isa: PtxIsaRequirement::new(65),
                    },
                    "{spelling}"
                );
                forms += 1;
            }
        }
    }
    assert_eq!(forms, 8);

    for spelling in [
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32\t{$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32\\09{$0, $1}, {$2}, {$3}, {$4, $5};",
        ";mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "prefix\\0Amma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "@p mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "@!%p\\09mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
    ] {
        assert!(
            contains_mma_m8n8k16_int8_features(spelling),
            "missed {spelling:?}"
        );
    }

    let representative = concat!(
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32 ",
        "{$0, $1}, {$4}, {$5}, {$2, $3};"
    );
    let requirements = detect_module_requirements_in_llvm_text(representative);
    let (target, _) = resolve_ptx_target(None, "CUDA_OXIDE_TARGET", None, requirements.features)
        .expect("auto-resolve");
    assert_eq!(target.sm(), "sm_75");
    assert_eq!(
        required_ptx_feature(&"sm_75".parse().unwrap(), requirements.ptx_isa).unwrap(),
        Some("+ptx65")
    );
    assert_eq!(
        required_ptx_feature(&"sm_80".parse().unwrap(), requirements.ptx_isa).unwrap(),
        None
    );

    let sm_70: CudaArch = "sm_70".parse().unwrap();
    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_70, requirements.features).is_err());
    assert!(validate_target_features(&sm_75, requirements.features).is_ok());
    assert!(validate_target_features(&sm_80, requirements.features).is_ok());
    let error = resolve_ptx_target(
        Some("sm_70"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .expect_err("sm_70 must not accept m8n8k16 INT8 MMA")
    .to_string();
    assert!(
        error.contains("cannot lower detected feature Sm75"),
        "{error}"
    );

    for near_miss in [
        "mma.sync.aligned.m8n8k8.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.col.row.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.row.col.satfiniteX.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.u32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k16.row.col.s32.u8.f16.s32 {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "@mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "!mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        ")mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_mma_m8n8k16_int8_features(near_miss),
            "matched {near_miss:?}"
        );
    }

    let m16 = concat!(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 ",
        "{$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};"
    );
    assert!(!contains_mma_m8n8k16_int8_features(m16));
    assert!(contains_dense_int8_mma_features(m16));
    assert_eq!(
        detect_module_requirements_in_llvm_text(m16),
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );

    let combined = format!("{representative}\n{m16}");
    let combined_requirements = detect_module_requirements_in_llvm_text(&combined);
    assert_eq!(
        combined_requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    let (target, _) = resolve_ptx_target(
        None,
        "CUDA_OXIDE_TARGET",
        None,
        combined_requirements.features,
    )
    .expect("combined m8 and m16 MMA should auto-resolve");
    assert_eq!(target.sm(), "sm_80");
}

#[test]
fn m8n8k32_int4_mma_detection_applies_exact_sm75_and_ptx65_floors() {
    let mut forms = 0;
    for a_type in ["s4", "u4"] {
        for b_type in ["s4", "u4"] {
            for satfinite in [false, true] {
                let overflow = if satfinite { ".satfinite" } else { "" };
                let spelling = format!(
                    "mma.sync.aligned.m8n8k32.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0, $1}}, {{$4}}, {{$5}}, {{$2, $3}};"
                );
                assert!(
                    contains_mma_m8n8k32_int4_features(&spelling),
                    "missed {spelling:?}"
                );
                assert!(
                    !contains_mma_m8n8k16_int8_features(&spelling),
                    "INT4 form entered the m8n8k16 detector: {spelling:?}"
                );
                assert!(
                    !contains_dense_int8_mma_features(&spelling),
                    "INT4 form entered the dense INT8 detector: {spelling:?}"
                );
                assert_eq!(
                    detect_module_requirements_in_llvm_text(&spelling),
                    ModuleRequirements {
                        features: DetectedFeatures::Sm75,
                        ptx_isa: PtxIsaRequirement::new(65),
                    },
                    "{spelling}"
                );
                forms += 1;
            }
        }
    }
    assert_eq!(forms, 8);

    for spelling in [
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32\t{$0, $1}, {$4}, {$5}, {$2, $3};",
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32\\09{$0, $1}, {$4}, {$5}, {$2, $3};",
        ";mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
        "prefix\\0Amma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
        "@p mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
        "@!%p\\09mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
    ] {
        assert!(
            contains_mma_m8n8k32_int4_features(spelling),
            "missed {spelling:?}"
        );
    }

    let representative = concat!(
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32 ",
        "{$0, $1}, {$4}, {$5}, {$2, $3};"
    );
    let requirements = detect_module_requirements_in_llvm_text(representative);
    assert_eq!(
        requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm75,
            ptx_isa: PtxIsaRequirement::new(65),
        }
    );
    assert_eq!(select_target(requirements.features).unwrap().sm(), "sm_75");
    assert_eq!(
        required_ptx_feature(&"sm_75".parse().unwrap(), requirements.ptx_isa).unwrap(),
        Some("+ptx65")
    );
    assert_eq!(
        required_ptx_feature(&"sm_80".parse().unwrap(), requirements.ptx_isa).unwrap(),
        None
    );

    let sm_72: CudaArch = "sm_72".parse().unwrap();
    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_72, requirements.features).is_err());
    assert!(validate_target_features(&sm_75, requirements.features).is_ok());
    assert!(validate_target_features(&sm_80, requirements.features).is_ok());
    let error = resolve_ptx_target(
        Some("sm_72"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .expect_err("sm_72 must not accept m8n8k32 INT4 MMA")
    .to_string();
    assert!(
        error.contains("cannot lower detected feature Sm75"),
        "{error}"
    );
}

#[test]
fn m8n8k32_int4_mma_detection_rejects_near_misses() {
    for near_miss in [
        "mma.sync.aligned.m8n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.s32.b1.b1.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sp.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.col.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32.satfinite {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.s32.satfinite.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.satfiniteX.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.u32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32x {$0}, {$1}, {$2}, {$3};",
        "not_mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "$mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "%mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "@mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        "!mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        ")mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
    ] {
        assert!(
            !contains_mma_m8n8k32_int4_features(near_miss),
            "matched {near_miss:?}"
        );
    }
}

#[test]
fn m8n8k32_int4_mma_requirements_compose_and_are_forward_compatible() {
    let int4 = concat!(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 ",
        "{$0, $1}, {$4}, {$5}, {$2, $3};"
    );
    let features = detect_features_in_llvm_text(int4);
    for target in [
        "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_120",
        "sm_120a",
    ] {
        assert!(
            arch_satisfies(&target.parse().unwrap(), features),
            "rejected {target}"
        );
    }
    assert!(!arch_satisfies(&"sm_72".parse().unwrap(), features));
    assert_eq!(
        resolve_ptx_target(
            None,
            "CUDA_OXIDE_TARGET",
            Some(&"sm_120".parse::<cuda_target_spec::DeviceArch>().unwrap()),
            features
        )
        .unwrap(),
        ("sm_120".parse().unwrap(), "detected GPU")
    );

    let m16_int8 = concat!(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    );
    let mixed = format!("{int4}\n{m16_int8}");
    let mixed_requirements = detect_module_requirements_in_llvm_text(&mixed);
    assert_eq!(
        mixed_requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    assert_eq!(
        select_target(mixed_requirements.features).unwrap().sm(),
        "sm_80"
    );

    let newer_ptx = format!("{int4}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&newer_ptx),
        ModuleRequirements {
            features: DetectedFeatures::Sm75 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
}

#[test]
fn mma_m8n8k4_f64_detection_enforces_sm80_and_ptx70() {
    let mnemonic = concat!(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 ",
        "{$0, $1}, {$2}, {$3}, {$4, $5};"
    );
    for spelling in [
        mnemonic,
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\t{$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\n{$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\\09{$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\\0A{$0, $1}, {$2}, {$3}, {$4, $5};",
        ";mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "prefix\\0Amma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "\"mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "{mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "$L:mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "/* comment */mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "@p mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "@!%p\\09mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
    ] {
        assert!(contains_mma_m8n8k4_f64_features(spelling), "{spelling:?}");
    }

    let requirements = detect_module_requirements_in_llvm_text(mnemonic);
    assert_eq!(
        requirements,
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        }
    );
    assert_eq!(select_target(requirements.features).unwrap().sm(), "sm_80");

    for near_miss in [
        "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k4.col.row.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k4.row.col.f32.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64x2 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\\5C09{$0, $1}, {$2}, {$3}, {$4, $5};",
        "not_mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "$mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "%mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "@mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "!mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "@!mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "not$mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        "/mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        ")mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
    ] {
        assert!(!contains_mma_m8n8k4_f64_features(near_miss));
        assert_eq!(
            detect_module_requirements_in_llvm_text(near_miss),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            },
            "matched near-miss {near_miss}"
        );
    }

    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_75, requirements.features).is_err());
    assert!(validate_target_features(&sm_80, requirements.features).is_ok());
    let error = resolve_ptx_target(
        Some("sm_75"),
        "CUDA_OXIDE_TARGET",
        None,
        requirements.features,
    )
    .expect_err("sm_75 must not accept FP64 tensor-core MMA")
    .to_string();
    assert!(
        error.contains("cannot lower detected feature Sm80"),
        "{error}"
    );

    let combined = format!("{mnemonic}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&combined),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
}

#[test]
fn test_movmatrix_detection_separates_sm75_from_the_ptx78_floor() {
    let mnemonic = "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;";
    for spelling in [
        mnemonic,
        "movmatrix.sync.aligned.m8n8.trans.b16\t$0, $1;",
        "movmatrix.sync.aligned.m8n8.trans.b16\n$0, $1;",
        "movmatrix.sync.aligned.m8n8.trans.b16\\09$0, $1;",
        "movmatrix.sync.aligned.m8n8.trans.b16\\0A$0, $1;",
        "movmatrix.sync.aligned.m8n8.trans.b16\\0D\\0A$0, $1;",
    ] {
        assert!(contains_movmatrix_features(spelling), "{spelling:?}");
    }
    assert_eq!(
        detect_features_in_llvm_text(mnemonic),
        DetectedFeatures::Movmatrix
    );
    assert_eq!(
        select_target(DetectedFeatures::Movmatrix).unwrap().sm(),
        "sm_75"
    );
    assert_eq!(
        detect_module_requirements_in_llvm_text(mnemonic).ptx_isa,
        PtxIsaRequirement::new(78)
    );

    for near_miss in [
        "movmatrix.sync.aligned.m8n8.b16 $0, $1;",
        "movmatrix.sync.aligned.m16n8.trans.b16 $0, $1;",
        "movmatrix.sync.aligned.m8n8.trans.b32 $0, $1;",
        "movmatrix.sync.aligned.m8n8.trans.b16x2 $0, $1;",
        "movmatrix.sync.aligned.m8n8.trans.b16\\5C09$0, $1;",
    ] {
        assert!(
            !contains_movmatrix_features(near_miss),
            "matched {near_miss}"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(near_miss),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );
    }

    let combined = format!("{mnemonic}\ncp.async.ca.shared.global [$0], [$1], 4;");
    assert_eq!(
        detect_module_requirements_in_llvm_text(&combined),
        ModuleRequirements {
            features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        },
        "the architecture and PTX ISA floors must compose independently"
    );

    let sm_70: CudaArch = "sm_70".parse().unwrap();
    let sm_75: CudaArch = "sm_75".parse().unwrap();
    let sm_80: CudaArch = "sm_80".parse().unwrap();
    assert!(validate_target_features(&sm_70, DetectedFeatures::Movmatrix).is_err());
    assert!(validate_target_features(&sm_75, DetectedFeatures::Movmatrix).is_ok());
    assert!(validate_target_features(&sm_80, DetectedFeatures::Movmatrix).is_ok());

    for target in ["sm_75", "sm_80", "sm_86", "sm_87"] {
        assert_eq!(
            required_ptx_feature(&target.parse().unwrap(), PtxIsaRequirement::new(78)).unwrap(),
            Some("+ptx78"),
            "{target} needs an explicit PTX 7.8 floor"
        );
    }
    assert_eq!(
        required_ptx_feature(&"sm_90".parse().unwrap(), PtxIsaRequirement::new(78)).unwrap(),
        None
    );
    for target in ["sm_88", "sm_89"] {
        assert_eq!(
            required_ptx_feature(&target.parse().unwrap(), PtxIsaRequirement::new(78)).unwrap(),
            None,
            "{target} already requires PTX 7.8 or newer"
        );
    }
    assert_eq!(
        required_ptx_feature(&"sm_75".parse().unwrap(), PtxIsaRequirement::Default).unwrap(),
        None
    );
}

#[test]
fn matrix_memory_detection_composes_architecture_and_ptx_isa_floors() {
    let base_ldmatrix = "ldmatrix.sync.aligned.m8n8.x4.b16 {$0, $1, $2, $3}, [$4];";
    assert_eq!(
        detect_module_requirements_in_llvm_text(base_ldmatrix),
        ModuleRequirements {
            features: DetectedFeatures::Ldmatrix,
            ptx_isa: PtxIsaRequirement::new(65),
        }
    );

    let cta_ldmatrix = "ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {$0}, [$1];";
    assert_eq!(
        detect_module_requirements_in_llvm_text(cta_ldmatrix),
        ModuleRequirements {
            features: DetectedFeatures::Ldmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );

    for stmatrix in [
        "stmatrix.sync.aligned.m8n8.x1.b16 [$0], {$1};",
        "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [$0], {$1, $2, $3, $4};",
    ] {
        assert_eq!(
            detect_module_requirements_in_llvm_text(stmatrix),
            ModuleRequirements {
                features: DetectedFeatures::Sm90,
                ptx_isa: PtxIsaRequirement::new(78),
            }
        );
    }

    for newer in [
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];",
        "ldmatrix.sync.aligned.m8n16.x2.shared::cta.b8x16.b6x16_p32 {$0, $1}, [$2];",
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};",
    ] {
        assert_eq!(
            detect_module_requirements_in_llvm_text(newer),
            ModuleRequirements {
                features: DetectedFeatures::MatrixBlackwell
                    | if newer.starts_with("ldmatrix") {
                        DetectedFeatures::Ldmatrix
                    } else {
                        DetectedFeatures::Sm90
                    },
                ptx_isa: PtxIsaRequirement::new(86),
            },
            "{newer}"
        );
    }

    let mixed = format!(
        "{base_ldmatrix}\n{}",
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
    );
    assert_eq!(
        detect_module_requirements_in_llvm_text(&mixed),
        ModuleRequirements {
            features: DetectedFeatures::Movmatrix | DetectedFeatures::Ldmatrix,
            ptx_isa: PtxIsaRequirement::new(78),
        },
        "the strongest PTX ISA floor must survive equal sm_75 feature families"
    );

    assert_eq!(
        required_ptx_feature(&"sm_75".parse().unwrap(), PtxIsaRequirement::new(65)).unwrap(),
        Some("+ptx65")
    );
    assert_eq!(
        required_ptx_feature(&"sm_80".parse().unwrap(), PtxIsaRequirement::new(65)).unwrap(),
        None
    );
    assert_eq!(
        required_ptx_feature(&"sm_100a".parse().unwrap(), PtxIsaRequirement::new(86)).unwrap(),
        None
    );

    let adjacent_unrelated_b8 = concat!(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [$1]; ",
        "mov.b8 $2, $3;"
    );
    assert_eq!(
        detect_module_requirements_in_llvm_text(adjacent_unrelated_b8),
        ModuleRequirements {
            features: DetectedFeatures::Ldmatrix,
            ptx_isa: PtxIsaRequirement::new(65),
        },
        "an unrelated b8 instruction must not raise the ldmatrix family"
    );
}

#[test]
fn grid_constant_uses_existing_volta_target_floor_even_with_dynamic_stack() {
    // NVVM requires Volta+ for grid-constant storage. The shared target table
    // already enforces that floor, including modules whose only detected
    // instruction feature (dynamic stack) would have a lower hardware floor.
    for prefix in ["", "call ptr @llvm.stacksave.p0()"] {
        let llvm = format!("{prefix}\n!0 = !{{ptr @read, !\"grid_constant\", !1}}");
        let features = detect_features_in_llvm_text(&llvm);
        assert!(validate_target_features(&"sm_52".parse().unwrap(), features).is_err());
        assert!(validate_target_features(&"sm_70".parse().unwrap(), features).is_ok());
        assert!(validate_target_features(&"sm_120".parse().unwrap(), features).is_ok());
    }
}
