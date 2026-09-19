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

/// `redux.sync` is f32 by its own token, never by an `.f32` sitting nearby
/// (#1303).
#[test]
fn test_redux_f32_detection_is_scoped_to_the_instruction() {
    // The shape that found this: integer reductions in one kernel and
    // ordinary float work in another. In the module that was rejected, the
    // two matches were 547 lines apart.
    let integer_redux_and_distant_f32 = r#"
        declare i32 @llvm.nvvm.redux.sync.add(i32, i32) #0
        declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #0
    "#;
    // LLVM does not require a newline between declarations, so scoping to a
    // line is not enough either.
    let same_line = "declare i32 @llvm.nvvm.redux.sync.add(i32, i32) declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32)";
    // An SSA name may end in `.f32` and says nothing about the callee.
    let unrelated_ssa_name = "%acc.f32 = call i32 @llvm.nvvm.redux.sync.umax(i32 -1, i32 %v)";

    for text in [integer_redux_and_distant_f32, same_line, unrelated_ssa_name] {
        assert!(
            !detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "an integer reduction must not select the f32 extension: {text}"
        );
    }

    // The intrinsic spelling has no type suffix -- the `f`-prefixed operation
    // is what makes it float. These were detected before only when some
    // unrelated `.f32` happened to sit nearby.
    let float_intrinsic = "%r = call float @llvm.nvvm.redux.sync.fmin(float %v, i32 -1)";
    let float_intrinsic_qualified =
        "%r = call float @llvm.nvvm.redux.sync.fmax.abs.NaN(float %v, i32 -1)";
    // Inline PTX carries the type as a modifier of the opcode token.
    let float_asm = r#"call float asm sideeffect "redux.sync.min.abs.NaN.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;

    for text in [float_intrinsic, float_intrinsic_qualified, float_asm] {
        assert!(
            detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "a float reduction must select the f32 extension: {text}"
        );
    }
}

/// Mentioning an intrinsic is not using one, and a real call may be spelled
/// with a quoted identifier (#1303).
#[test]
fn test_redux_f32_detection_reads_code_and_not_prose() {
    // Text that names the callee without calling it.
    let in_a_comment =
        "%r = call i32 @llvm.nvvm.redux.sync.add(i32 -1, i32 %v) ; @llvm.nvvm.redux.sync.fmin";
    let in_a_data_string = r#"@.msg = private constant [27 x i8] c"@llvm.nvvm.redux.sync.fmin\00""#;
    let in_metadata = r#"!0 = !{!"@llvm.nvvm.redux.sync.fmin"}"#;
    // The same for the inline-PTX spelling: a comment and a data constant are
    // not instructions.
    let ptx_in_a_comment = "  ; redux.sync.min.abs.NaN.f32 $0, $1, $2;";
    let ptx_in_a_data_string = r#"@.msg = private constant [25 x i8] c"redux.sync.min.f32 a,b,c;""#;

    for text in [
        in_a_comment,
        in_a_data_string,
        in_metadata,
        ptx_in_a_comment,
        ptx_in_a_data_string,
    ] {
        assert!(
            !detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "text that only mentions the f32 reduction must not select it: {text}"
        );
    }

    // A real call, spelled with a quoted identifier and with one of its
    // characters escaped -- `\66` is `f`, so both name `fmin`.
    let quoted = r#"%r = call float @"llvm.nvvm.redux.sync.fmin"(float %v, i32 -1)"#;
    let quoted_escaped = r#"%r = call float @"llvm.nvvm.redux.sync.\66min"(float %v, i32 -1)"#;
    // And the constraint string beside an asm template is data, while the
    // template itself is code.
    let asm_template = r#"call float asm sideeffect "redux.sync.min.abs.NaN.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;

    for text in [quoted, quoted_escaped, asm_template] {
        assert!(
            detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "a float reduction must select the f32 extension: {text}"
        );
    }
}

/// Token identity at the edges the flattened search lost (#1303).
///
/// Each input is a line from a module `llvm-as` from the pinned toolchain
/// accepts, so none of these is a shape valid IR cannot produce. The escape
/// cases were also round-tripped through `llvm-dis` to confirm what LLVM
/// itself decodes them to.
#[test]
fn test_redux_f32_detection_keeps_token_identity() {
    // `unwind` is the last of the four modifiers `asm` may carry, in order:
    // sideeffect, alignstack, inteldialect, unwind.
    let unwind = r#"%r = call float asm sideeffect unwind "redux.sync.min.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    // `\72` is `r`: LLVM decodes the template before the backend sees it.
    let escaped_opcode = r#"%r = call float asm sideeffect "\72edux.sync.min.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    // A PTX comment ended by an *escaped* newline, then a real instruction.
    // Only decoding before stripping comments finds it: the raw template is
    // one line, so a `//` there would appear to run to its end.
    let after_escaped_newline = r#"%r = call float asm sideeffect "// scale first\0Aredux.sync.min.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    // Module-level inline asm has the same grammar and reaches the same PTX.
    let module_asm = r#"module asm "redux.sync.min.f32 %f1, %f2, %r3;""#;

    for text in [unwind, escaped_opcode, after_escaped_newline, module_asm] {
        assert!(
            detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "a float reduction must select the f32 extension: {text}"
        );
    }

    // PTX comments inside a template are prose, in either comment form.
    let ptx_line_comment = r#"%r = call float asm sideeffect "// redux.sync.min.f32 $0, $1, $2;\0Amov.f32 $0, $1;", "=f,f,r"(float %v, i32 %m)"#;
    let ptx_block_comment = r#"%r = call float asm sideeffect "/* redux.sync.min.f32 $0, $1, $2; */ mov.f32 $0, $1;", "=f,f,r"(float %v, i32 %m)"#;
    // `\\` is one backslash and the hex digits after it are plain text: `llc`
    // emits this template as `\20redux.sync.min.f32 ...`, which is not an
    // instruction. A decoder that skipped the `\\` rule would read `\20` as a
    // space instead and find a `redux.sync.min.f32` that is not there.
    let escaped_backslash = r#"%r = call float asm sideeffect "\\20redux.sync.min.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    // One symbol whose quoted name happens to contain another's spelling.
    let embedded_in_a_name = r#"@"unused @llvm.nvvm.redux.sync.fmin" = global i32 0"#;
    // A basic block may be named like an instruction without being one.
    let label = "redux.sync.min.f32:\n  ret void";

    for text in [
        ptx_line_comment,
        ptx_block_comment,
        escaped_backslash,
        embedded_in_a_name,
        label,
    ] {
        assert!(
            !detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "no float reduction is executed here: {text}"
        );
    }
}

/// PTX lexical rules inside an asm template (#1303).
///
/// Every input is a complete module accepted by `llvm-as` and `llc` from the
/// pinned toolchain and by ptxas at sm_100a. The float cases are the ones
/// ptxas then refuses at sm_80 as `redux.f32`; the others assemble there.
#[test]
fn test_redux_f32_detection_reads_ptx_as_ptx() {
    // Whitespace, a newline or a comment may separate an opcode from its
    // modifiers; it is still one instruction.
    let spaced_type = r#"%r = call float asm sideeffect "redux.sync.min .f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    let comment_between = r#"%r = call float asm sideeffect "redux.sync.min/* qualifier */.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    let fully_spaced = r#"%r = call float asm sideeffect "redux .sync .min .f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    let newline_between = r#"%r = call float asm sideeffect "redux.sync.min\0A.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;
    // A backslash is literal inside a PTX string, so `"nounroll\"` closes at
    // its last quote and the instruction after it is real.
    let after_trailing_backslash = r#"%r = call float asm sideeffect ".pragma \22nounroll\5C\22;\0Aredux.sync.min.f32 $0, $1, $2;", "=f,f,r"(float %v, i32 %m)"#;

    for text in [
        spaced_type,
        comment_between,
        fully_spaced,
        newline_between,
        after_trailing_backslash,
    ] {
        assert!(
            detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "a float reduction must select the f32 extension: {text}"
        );
    }

    // A directive's quoted data is not an instruction.
    let file_path = r#"module asm ".file 1 \22name redux.sync.min.f32 label.cu\22""#;
    // Spacing does not make an integer reduction float.
    let spaced_integer = r#"%r = call i32 asm sideeffect "redux .sync .min .u32 $0, $1, $2;", "=r,r,r"(i32 %v, i32 %m)"#;
    // Nor does float work beside it in the same template: `.f32` counts only
    // as a modifier of the `redux` opcode itself.
    let integer_beside_f32 = r#"%r = call i32 asm sideeffect "{ .reg .f32 t; redux.sync.min.u32 $0, $1, $2; mov.f32 t, 0f3F800000; }", "=r,r,r"(i32 %v, i32 %m)"#;

    for text in [file_path, spaced_integer, integer_beside_f32] {
        assert!(
            !detect_features_in_llvm_text(text).contains(DetectedFeatures::ReduxF32),
            "no float reduction is executed here: {text}"
        );
    }
}
