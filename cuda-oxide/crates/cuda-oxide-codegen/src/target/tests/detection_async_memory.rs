/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::target::arch::*;
use crate::target::detect::*;
use crate::target::features::*;
use crate::target::select::*;

#[test]
fn tma_and_wgmma_raise_their_independent_ptx_floors() {
    for tma in [
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes;",
        "cp.async.bulk.commit_group;",
        "cp.async.bulk.wait_group 0;",
        "cp.async.bulk.wait_group.read 0;",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(tma);
        assert!(
            requirements.features.contains(DetectedFeatures::Tma),
            "{tma}"
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(80), "{tma}");
    }

    let non_bulk = "cp.async.commit_group;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(non_bulk),
        ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::Default,
        }
    );

    let tma_and_movmatrix = concat!(
        "cp.async.bulk.commit_group; ",
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
    );
    assert_eq!(
        detect_module_requirements_in_llvm_text(tma_and_movmatrix).ptx_isa,
        PtxIsaRequirement::new(80)
    );

    let wgmma = "wgmma.fence.sync.aligned;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(wgmma),
        ModuleRequirements {
            features: DetectedFeatures::Wgmma,
            ptx_isa: PtxIsaRequirement::new(80),
        }
    );

    let shared_cta =
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes;";
    assert!(contains_tma_shared_cta_destination(shared_cta));
    let shared_cta_requirements = detect_module_requirements_in_llvm_text(shared_cta);
    assert_eq!(shared_cta_requirements.features, DetectedFeatures::Tma);
    assert_eq!(shared_cta_requirements.ptx_isa, PtxIsaRequirement::new(86));

    let shared_source = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group;";
    assert!(!contains_tma_shared_cta_destination(shared_source));
    assert_eq!(
        detect_module_requirements_in_llvm_text(shared_source).ptx_isa,
        PtxIsaRequirement::new(80)
    );

    let cta_group = "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(cta_group).ptx_isa,
        PtxIsaRequirement::new(86)
    );

    assert_eq!(
        required_ptx_feature(&"sm_90".parse().unwrap(), PtxIsaRequirement::new(80)).unwrap(),
        Some("+ptx80")
    );
    assert_eq!(
        required_ptx_feature(&"sm_90a".parse().unwrap(), PtxIsaRequirement::new(86)).unwrap(),
        Some("+ptx86")
    );
    assert_eq!(
        required_ptx_feature(&"sm_100a".parse().unwrap(), PtxIsaRequirement::new(80)).unwrap(),
        None
    );
}

#[test]
fn related_cluster_mbarrier_and_clc_requirements_are_detected() {
    for ptx in [
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
        "fence.mbarrier_init.release.cluster;",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(
            requirements.features.contains(DetectedFeatures::Tma),
            "{ptx}"
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(80), "{ptx}");
        assert!(arch_satisfies(
            &"sm_90".parse().unwrap(),
            requirements.features
        ));
    }

    for (ptx, expected_isa) in [
        (
            "mbarrier.init.shared.b64 [$0], 1;",
            PtxIsaRequirement::new(70),
        ),
        (
            "mbarrier.test_wait.parity.shared.b64 $0, [$1], $2;",
            PtxIsaRequirement::new(71),
        ),
        (
            "mbarrier.try_wait.parity.shared::cta.b64 $0, [$1], $2;",
            PtxIsaRequirement::new(78),
        ),
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(
            requirements.features.contains(DetectedFeatures::Sm80),
            "{ptx}"
        );
        assert_eq!(requirements.ptx_isa, expected_isa, "{ptx}");
        if ptx.contains("try_wait") {
            assert!(requirements.features.contains(DetectedFeatures::Tma));
            assert!(!arch_satisfies(
                &"sm_80".parse().unwrap(),
                requirements.features
            ));
        } else {
            assert!(arch_satisfies(
                &"sm_80".parse().unwrap(),
                requirements.features
            ));
            assert!(!arch_satisfies(
                &"sm_75".parse().unwrap(),
                requirements.features
            ));
        }
    }

    for ptx in [
        "redux.sync.add.u32 $0, $1, $2;",
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "cvt.rn.relu.bf16x2.f32 $0, $1, $2;",
        "cvt.rz.bf16x2.f32 $0, $1, $2;",
    ] {
        assert!(
            detect_features_in_llvm_text(ptx).contains(DetectedFeatures::Sm80),
            "{ptx}"
        );
    }
    assert_eq!(
        required_ptx_feature(&"sm_80".parse().unwrap(), PtxIsaRequirement::new(70)).unwrap(),
        None
    );
    assert_eq!(
        required_ptx_feature(&"sm_80".parse().unwrap(), PtxIsaRequirement::new(71)).unwrap(),
        Some("+ptx71")
    );
    for target in ["sm_86", "sm_87", "sm_88", "sm_89"] {
        assert_eq!(
            required_ptx_feature(&target.parse().unwrap(), PtxIsaRequirement::new(71)).unwrap(),
            None,
            "{target} cannot be downgraded below its minimum PTX ISA"
        );
    }

    for ptx in [
        "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 $0, [$1], $2;",
        "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
        "fence.acquire.sync_restrict::shared::cluster.cluster;",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(
            requirements.features.contains(DetectedFeatures::Tma),
            "{ptx}"
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86), "{ptx}");
        assert!(!arch_satisfies(
            &"sm_80".parse().unwrap(),
            requirements.features
        ));
    }

    for ptx in [
        "mbarrier.test_wait.acquire.cta.shared::cta.b64 $0, [$1], $2;",
        "mbarrier.arrive.release.cta.shared::cta.b64 $0, [$1];",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(requirements.features.contains(DetectedFeatures::Tma));
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(80));
        assert!(!arch_satisfies(
            &"sm_80".parse().unwrap(),
            requirements.features
        ));
    }

    let cluster_sync = "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(cluster_sync),
        ModuleRequirements {
            features: DetectedFeatures::Cluster,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
    assert_eq!(
        select_target(DetectedFeatures::Cluster).unwrap().sm(),
        "sm_90"
    );

    let cluster_release = "barrier.cluster.arrive.release;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(cluster_release).ptx_isa,
        PtxIsaRequirement::new(80)
    );

    for ptx in [
        "fence.sc.cluster;",
        "fence.acq_rel.cluster;",
        "ld.shared::cluster.u32 $0, [$1];",
        "ld.acquire.cluster.global.u32 $0, [$1];",
        "getctarank.shared::cluster.u32 $0, $1;",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(requirements.features.contains(DetectedFeatures::Cluster));
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(78));
        assert!(!arch_satisfies(
            &"sm_80".parse().unwrap(),
            requirements.features
        ));
    }

    for ptx in [
        "fence.acquire.cta;",
        "fence.release.gpu;",
        "fence.acquire.cluster;",
        "fence.release.sys;",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(
            requirements.features.contains(DetectedFeatures::Sm90),
            "{ptx}"
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86), "{ptx}");
        assert_eq!(
            requirements.features.contains(DetectedFeatures::Cluster),
            ptx.contains(".cluster"),
            "{ptx}"
        );
        assert!(!arch_satisfies(
            &"sm_80".parse().unwrap(),
            requirements.features
        ));
    }

    let multimem = "multimem.red.relaxed.cluster.global.add.u32 [$0], $1;";
    let requirements = detect_module_requirements_in_llvm_text(multimem);
    assert_eq!(requirements.features, DetectedFeatures::Sm90);
    assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86));
    assert_eq!(select_target(requirements.features).unwrap().sm(), "sm_90");
    let multimem_debug_filename = r#"!9 = !DIFile(filename: "multimem.rs", directory: "/tmp")"#;
    assert_eq!(
        detect_module_requirements_in_llvm_text(multimem_debug_filename),
        ModuleRequirements {
            features: DetectedFeatures::Basic,
            ptx_isa: PtxIsaRequirement::Default,
        }
    );

    for multimem in [
        "multimem.ld_reduce.relaxed.cta.add.v4.e4m3 {$0, $1, $2, $3}, [$4];",
        "multimem.st.relaxed.gpu.e5m2 [$0], $1;",
        "multimem.ld_reduce.add.acc::f16.v4.e5m2 {$0, $1, $2, $3}, [$4];",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(multimem);
        assert_eq!(
            requirements.features,
            DetectedFeatures::MultimemFp8 | DetectedFeatures::Sm90,
            "{multimem}"
        );
        assert_eq!(
            requirements.ptx_isa,
            PtxIsaRequirement::new(86),
            "{multimem}"
        );
        assert_eq!(
            select_target(requirements.features).unwrap().sm(),
            "sm_100a"
        );
        for target in [
            "sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_121a", "sm_100f", "sm_103f", "sm_110f",
        ] {
            assert!(
                arch_satisfies(&target.parse().unwrap(), requirements.features),
                "{target}"
            );
        }
        for target in ["sm_100", "sm_90a", "sm_120f", "sm_121f"] {
            assert!(
                !arch_satisfies(&target.parse().unwrap(), requirements.features),
                "{target}"
            );
        }
    }

    let redux_f32 = "redux.sync.min.abs.NaN.f32 $0, $1, $2;";
    let requirements = detect_module_requirements_in_llvm_text(redux_f32);
    assert_eq!(
        requirements.features,
        DetectedFeatures::ReduxF32 | DetectedFeatures::Sm80
    );
    assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86));
    assert_eq!(
        select_target(requirements.features).unwrap().sm(),
        "sm_100a"
    );
    for target in ["sm_100a", "sm_103a", "sm_100f", "sm_103f"] {
        assert!(
            arch_satisfies(&target.parse().unwrap(), requirements.features),
            "{target}"
        );
    }
    for target in ["sm_100", "sm_110a", "sm_120a", "sm_121f"] {
        assert!(
            !arch_satisfies(&target.parse().unwrap(), requirements.features),
            "{target}"
        );
    }

    for sreg in [
        "mov.u32 $0, %clusterid.x;",
        "mov.u32 $0, %nclusterid.z;",
        "mov.u32 $0, %cluster_ctarank;",
        "mov.u32 $0, %cluster_nctarank;",
        "mov.pred $0, %is_explicit_cluster;",
    ] {
        assert_eq!(
            detect_module_requirements_in_llvm_text(sreg),
            ModuleRequirements {
                features: DetectedFeatures::Cluster,
                ptx_isa: PtxIsaRequirement::new(78),
            },
            "{sreg}"
        );
    }

    let cluster_metadata = r#"!0 = !{!"cluster_dim_x", i32 2}
            !1 = !{!"cluster_dim_y", i32 1}
            !2 = !{!"cluster_dim_z", i32 1}"#;
    assert_eq!(
        detect_module_requirements_in_llvm_text(cluster_metadata),
        ModuleRequirements {
            features: DetectedFeatures::Cluster,
            ptx_isa: PtxIsaRequirement::new(78),
        }
    );
    let cluster_debug_local =
        r#"!8 = !DILocalVariable(name: "cluster_dim_x", scope: !1, file: !2, line: 3)"#;
    assert_eq!(
        detect_module_requirements_in_llvm_text(cluster_debug_local),
        ModuleRequirements {
            features: DetectedFeatures::Basic,
            ptx_isa: PtxIsaRequirement::Default,
        }
    );

    let elect = "elect.sync $0|p, $1;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(elect),
        ModuleRequirements {
            features: DetectedFeatures::Sm90,
            ptx_isa: PtxIsaRequirement::new(80),
        }
    );

    let tcgen_wait = "tcgen05.wait::ld.sync.aligned;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(tcgen_wait),
        ModuleRequirements {
            features: DetectedFeatures::Blackwell,
            ptx_isa: PtxIsaRequirement::new(86),
        }
    );

    let tcgen_debug_filename = r#"!7 = !DIFile(filename: "tcgen05.rs", directory: "/tmp")"#;
    assert_eq!(
        detect_module_requirements_in_llvm_text(tcgen_debug_filename),
        ModuleRequirements {
            features: DetectedFeatures::Basic,
            ptx_isa: PtxIsaRequirement::Default,
        }
    );

    let clc = "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 $0, $1;";
    assert_eq!(
        detect_module_requirements_in_llvm_text(clc),
        ModuleRequirements {
            features: DetectedFeatures::Sm100,
            ptx_isa: PtxIsaRequirement::new(86),
        }
    );
    assert_eq!(
        select_target(DetectedFeatures::Sm100).unwrap().sm(),
        "sm_100"
    );
    assert!(!arch_satisfies(
        &"sm_90".parse().unwrap(),
        DetectedFeatures::Sm100
    ));
    assert!(arch_satisfies(
        &"sm_120".parse().unwrap(),
        DetectedFeatures::Sm100
    ));

    let clc_multicast = "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [$0], [$1];";
    let requirements = detect_module_requirements_in_llvm_text(clc_multicast);
    assert_eq!(
        requirements.features,
        DetectedFeatures::Sm100 | DetectedFeatures::BlackwellFamily
    );
    assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86));
    assert_eq!(
        select_target(requirements.features).unwrap().sm(),
        "sm_100a"
    );
    assert!(!arch_satisfies(
        &"sm_100".parse().unwrap(),
        requirements.features
    ));
    assert!(arch_satisfies(
        &"sm_120a".parse().unwrap(),
        requirements.features
    ));
    for arch in ["sm_100f", "sm_101f", "sm_110f", "sm_121f"] {
        assert!(
            arch_satisfies(&arch.parse().unwrap(), requirements.features),
            "{arch}"
        );
    }
    for arch in ["sm_103a", "sm_121a"] {
        assert!(
            !arch_satisfies(&arch.parse().unwrap(), requirements.features),
            "{arch}"
        );
    }
}

#[test]
fn ptx86_tma_modes_enforce_their_architecture_families() {
    for ptx in [
        "cp.async.bulk.global.shared::cta.bulk_group.cp_mask [$0], [$1], 16, $2;",
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes;",
        "cp.async.bulk.tensor.3d.shared::cta.global.im2col::w.mbarrier::complete_tx::bytes;",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(
            requirements.features.contains(DetectedFeatures::Tma),
            "{ptx}"
        );
        assert!(
            requirements.features.contains(DetectedFeatures::Sm100),
            "{ptx}"
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86), "{ptx}");
        assert!(!arch_satisfies(
            &"sm_90".parse().unwrap(),
            requirements.features
        ));
        assert!(arch_satisfies(
            &"sm_100".parse().unwrap(),
            requirements.features
        ));
    }

    for ptx in [
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes;",
        "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group;",
        "cp.async.bulk.tensor.3d.shared::cta.global.im2col::w::128.mbarrier::complete_tx::bytes;",
        "cp.async.bulk.prefetch.tensor.3d.L2.global.im2col::w::128;",
    ] {
        let requirements = detect_module_requirements_in_llvm_text(ptx);
        assert!(
            requirements.features.contains(DetectedFeatures::Tma),
            "{ptx}"
        );
        assert!(
            requirements
                .features
                .contains(DetectedFeatures::BlackwellAccelerated),
            "{ptx}"
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::new(86), "{ptx}");
        assert_eq!(
            select_target(requirements.features).unwrap().sm(),
            "sm_100a"
        );
        assert!(!arch_satisfies(
            &"sm_100".parse().unwrap(),
            requirements.features
        ));
        assert!(!arch_satisfies(
            &"sm_120a".parse().unwrap(),
            requirements.features
        ));
        assert!(arch_satisfies(
            &"sm_103f".parse().unwrap(),
            requirements.features
        ));
    }

    assert!(!contains_tma_sm100_features("custom.op.cp_mask $0;"));
    assert!(!contains_tma_blackwell_accelerated_features(
        "custom.tile::scatter4 $0;"
    ));
}

#[test]
fn test_sm90_floor_wins_when_sm80_features_are_also_present() {
    let llvm = r#"
            call i32 asm pure "add.rn.bf16x2 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
            call void asm sideeffect "cp.async.ca.shared.global [%0], [%1], 4;", "l,l"()
        "#;

    assert!(contains_sm90_features(llvm));
    assert!(contains_sm80_features(llvm));
    assert_eq!(
        detect_features_in_llvm_text(llvm),
        DetectedFeatures::Sm90 | DetectedFeatures::Sm80
    );
}

#[test]
fn test_tma_multicast_detection_requires_cta_mask() {
    let multicast = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 1, i1 false)";
    let unicast = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 0, i1 false)";
    let literal_multicast = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster";
    let cg1 =
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1";
    let cg2 = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2";
    let cg1_intrinsic = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %dst, i1 0, i1 false, i32 1)";
    let cg2_intrinsic = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %dst, i1 1, i1 false, i32 2)";
    let unrelated_i32 = "call void @unrelated(i32 2)";

    assert!(contains_tma_multicast(multicast));
    assert!(contains_tma_multicast(literal_multicast));
    assert!(!contains_tma_multicast(unicast));
    assert_eq!(
        detect_features_in_llvm_text(multicast),
        DetectedFeatures::TmaMulticast | DetectedFeatures::Tma
    );
    assert_eq!(
        detect_features_in_llvm_text(literal_multicast),
        DetectedFeatures::TmaMulticast | DetectedFeatures::Tma | DetectedFeatures::Cluster
    );
    assert_eq!(detect_features_in_llvm_text(unicast), DetectedFeatures::Tma);
    assert_eq!(
        detect_features_in_llvm_text(cg1),
        DetectedFeatures::TmaCtaGroup | DetectedFeatures::Tma
    );
    assert_eq!(
        detect_features_in_llvm_text(cg1_intrinsic),
        DetectedFeatures::TmaCtaGroup | DetectedFeatures::Tma
    );
    assert_eq!(
        detect_features_in_llvm_text(cg2),
        DetectedFeatures::TmaCtaGroup
            | DetectedFeatures::TmaMulticast
            | DetectedFeatures::Tma
            | DetectedFeatures::Cluster
    );
    assert_eq!(
        detect_features_in_llvm_text(cg2_intrinsic),
        DetectedFeatures::TmaCtaGroup | DetectedFeatures::TmaMulticast | DetectedFeatures::Tma
    );
    assert!(!contains_tma_cta_group_features(unrelated_i32));
}
