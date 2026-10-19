/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

use crate::model::{
    BackendLoweringMechanism, IntrinsicSource, MbarrierBasicOperation, ReduxAdapter,
    WarpBarrierAdapter, WarpShuffleMode, WarpShuffleValueKind,
};
use crate::render::common::{intrinsic_marker, llvm};
use crate::render::families::{
    active_masks, cp_async_controls, cp_async_copies, cp_async_mbarriers, mbarrier_basics, redux,
    vote_intrinsics, warp_barriers, warp_matches, warp_shuffles,
};
use std::collections::BTreeSet;
use std::path::Path;

#[test]
fn cp_async_rendering_preserves_compatibility_dispatch_and_backend_routes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(cp_async_copies(&catalog).count(), 8);
    assert_eq!(cp_async_controls(&catalog).count(), 3);
    assert_eq!(cp_async_mbarriers(&catalog).count(), 4);

    let compatibility = render_compat_cp_async_copy(&catalog, "test-hash");
    for signature in [
        "pub unsafe fn cp_async_ca_4(_shared_dst: *mut u32, _global_src: *const u32)",
        "pub unsafe fn cp_async_ca_8(_shared_dst: *mut u32, _global_src: *const u32)",
        "pub unsafe fn cp_async_ca_16(_shared_dst: *mut u32, _global_src: *const u32)",
        "pub unsafe fn cp_async_ca_zfill_4(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
        "pub unsafe fn cp_async_ca_zfill_8(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
        "pub unsafe fn cp_async_ca_zfill_16(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
        "pub unsafe fn cp_async_cg_16(_shared_dst: *mut u32, _global_src: *const u32)",
        "pub unsafe fn cp_async_cg_zfill_16(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
        "pub unsafe fn cp_async_commit_group()",
        "pub unsafe fn cp_async_wait_all()",
        "pub unsafe fn cp_async_wait_group(_max_pending: u32)",
        "pub unsafe fn cp_async_mbarrier_arrive(_barrier: *mut crate::barrier::Barrier)",
        "pub unsafe fn cp_async_mbarrier_arrive_shared(_barrier: *mut crate::barrier::Barrier)",
        "pub unsafe fn cp_async_mbarrier_arrive_noinc(_barrier: *mut crate::barrier::Barrier)",
        "pub unsafe fn cp_async_mbarrier_arrive_noinc_shared(_barrier: *mut crate::barrier::Barrier)",
    ] {
        assert!(compatibility.contains(signature));
    }

    let dialect = render_dialect_cp_async_copy(&catalog, "test-hash");
    let importer = render_importer(&catalog, "test-hash");
    let lowering = render_lowering(&catalog, "test-hash");
    let targets = render_targets(&catalog, "test-hash");
    for record in cp_async_copies(&catalog)
        .chain(cp_async_controls(&catalog))
        .chain(cp_async_mbarriers(&catalog))
    {
        assert!(dialect.contains(&format!("pub struct {}", record.dialect.op_type)));
        assert!(dialect.contains(&format!("{}::register(ctx)", record.dialect.op_type)));
        assert!(importer.contains(&record.rust.canonical_path));
        for path in &record.rust.compatibility_paths {
            assert!(importer.contains(path));
        }
        assert!(importer.contains(&intrinsic_marker(&catalog, record)));
        assert!(lowering.contains(&format!(
            "impl MirToLlvmConversion for {}",
            record.dialect.op_type
        )));
        assert!(lowering.contains(&record.llvm_identifier()));
        assert!(
            record
                .backend_lowerings
                .iter()
                .any(|entry| entry.backend == IntrinsicBackend::LlvmNvptx)
        );
        assert!(
            record
                .backend_lowerings
                .iter()
                .any(|entry| entry.backend == IntrinsicBackend::LibNvvm)
        );
        assert!(targets.contains(&format!("id: {:?}", record.id)));
    }
    assert_eq!(dialect.matches("::register(ctx);").count(), 15);
    assert!(lowering.contains("convert_generated_cp_async_copy"));
    assert!(lowering.contains("convert_generated_cp_async_control"));
    assert!(lowering.contains("convert_generated_cp_async_mbarrier"));
}

#[test]
fn cp_async_mbarrier_rendering_preserves_counting_and_state_space_routes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    let records: Vec<_> = cp_async_mbarriers(&catalog).collect();
    assert_eq!(records.len(), 4);

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub unsafe fn i0101(_arg0: *mut u64)"));
    assert!(raw.contains("live, initialized, eight-byte-aligned mbarrier"));
    assert!(raw.contains("initial pending count must already include"));

    let compatibility = render_compat_cp_async_copy(&catalog, "test-hash");
    assert_eq!(
        compatibility
            .matches("That increment must not exceed the barrier's pending-count limit.")
            .count(),
        2
    );

    let dialect = render_dialect_cp_async_copy(&catalog, "test-hash");
    assert!(dialect.contains("mutable generic/shared pointer to u64"));
    let importer = render_importer(&catalog, "test-hash");
    let lowering = render_lowering(&catalog, "test-hash");
    for record in &records {
        assert!(importer.contains(&format!(
            "let bridge = {}::build(ctx, barrier)",
            record.dialect.op_type
        )));
        assert!(lowering.contains(&record.llvm_identifier()));
        let routes: BTreeSet<_> = record
            .backend_lowerings
            .iter()
            .map(|route| (route.backend, route.mechanism))
            .collect();
        assert_eq!(
            routes,
            BTreeSet::from([
                (
                    IntrinsicBackend::LlvmNvptx,
                    BackendLoweringMechanism::TypedNvvm,
                ),
                (
                    IntrinsicBackend::LibNvvm,
                    BackendLoweringMechanism::InlinePtx,
                ),
            ])
        );
    }
    assert!(lowering.contains("\"arrive\", \"generic\""));
    assert!(lowering.contains("\"arrive\", \"shared\""));
    assert!(lowering.contains("\"arrive_no_inc\", \"generic\""));
    assert!(lowering.contains("\"arrive_no_inc\", \"shared\""));

    let generic = records
        .iter()
        .find(|record| record.id == "cp_async_mbarrier_arrive")
        .unwrap();
    let generic_probe = render_probe(&catalog, generic, "test-hash");
    assert!(generic_probe.contains("declare void @llvm.nvvm.cp.async.mbarrier.arrive(ptr)"));
    assert!(!generic_probe.contains("addrspacecast"));
    let shared = records
        .iter()
        .find(|record| record.id == "cp_async_mbarrier_arrive_shared")
        .unwrap();
    let shared_probe = render_probe(&catalog, shared, "test-hash");
    assert!(shared_probe.contains("ptr addrspace(3)"));
    assert!(shared_probe.contains("addrspacecast ptr %barrier_generic"));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## cp.async mbarrier contracts"));
    assert!(reference.contains("cp_async_mbarrier_arrive_noinc`: runtime `unexecuted`"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/async_copy.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/cp_async.rs"
    )));
    assert!(!outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/cp_async_mbarrier.rs"
    )));
}

#[test]
fn redux_rendering_preserves_mask_first_api_and_source_first_llvm_order() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    assert_eq!(catalog.schema, crate::resolve::CATALOG_SCHEMA);
    validate_renderable(&catalog).unwrap();

    assert_eq!(redux(&catalog).count(), 16);
    let record = redux(&catalog).next().unwrap();
    assert_eq!(
        record.redux.as_ref().unwrap().adapter,
        ReduxAdapter::MaskValueToSourceMemberMask
    );

    let dialect = render_dialect_redux(&catalog, "test-hash");
    assert!(dialect.contains("name = \"nvvm.redux_sync_add\""));
    assert!(dialect.contains("name = \"nvvm.redux_sync_min\""));
    assert!(dialect.contains("vec![member_mask, value]"));
    assert!(dialect.contains("ReduxSyncAddOp::register(ctx)"));
    assert!(dialect.contains("Signedness::Signed"));

    assert!(dialect.contains("name = \"nvvm.redux_sync_fmin\""));
    assert!(dialect.contains("types::{FP32Type, IntegerType, Signedness}"));
    assert!(dialect.contains("let result_ty = FP32Type::get(ctx);"));
    assert!(dialect.contains("is_f32(ctx, op.get_operand(1).get_type(ctx))"));
    assert!(dialect.contains("is_f32(ctx, op.get_result(0).get_type(ctx))"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::warp::redux_sync_add"));
    assert!(importer.contains("let (member_mask, last_op)"));
    assert!(importer.contains("let reduction = ReduxSyncAddOp::build(ctx, member_mask, value)"));
    assert!(importer.contains("set_generated_intrinsic_marker(ctx, reduction, \"v1:i0017\")"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("impl MirToLlvmConversion for ReduxSyncAddOp"));
    assert!(lowering.contains("convert_redux(ctx, rewriter, self.get_operation(), operands_info"));
    assert!(lowering.contains("\"llvm_nvvm_redux_sync_add\""));

    let probe = render_probe(&catalog, record, "test-hash");
    assert!(probe.contains("define i32 @probe_redux_sync_add(i32 %member_mask, i32 %value)"));
    assert!(probe.contains("call i32 @llvm.nvvm.redux.sync.add(i32 %value, i32 %member_mask)"));

    let f32_record = redux(&catalog)
        .find(|record| record.id == "redux_sync_min_f32")
        .unwrap();
    let f32_probe = render_probe(&catalog, f32_record, "test-hash");
    assert!(
        f32_probe
            .contains("define float @probe_redux_sync_min_f32(i32 %member_mask, float %value)")
    );
    assert!(
        f32_probe.contains("call float @llvm.nvvm.redux.sync.fmin(float %value, i32 %member_mask)")
    );

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    let signature = "pub unsafe fn i0017(_arg0: u32, _arg1: u32) -> u32";
    let index = raw.find(signature).unwrap();
    assert!(raw[..index].ends_with("#[must_use]\n#[inline(never)]\n"));
    assert!(raw.contains("pub unsafe fn i0019(_arg0: u32, _arg1: i32) -> i32"));
    assert!(raw.contains("pub unsafe fn i0024(_arg0: u32, _arg1: u32) -> u32"));
    assert!(raw.contains("pub unsafe fn i1003(_arg0: u32, _arg1: f32) -> f32"));
    assert!(raw.contains("The executing lane must be named in `mask`"));
}

#[test]
fn vote_rendering_keeps_types_selection_pairs_and_raw_only_uni() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(vote_intrinsics(&catalog).count(), 4);

    let dialect = render_dialect_vote(&catalog, "test-hash");
    for op in [
        "VoteSyncAllOp",
        "VoteSyncAnyOp",
        "VoteSyncBallotOp",
        "VoteSyncUniOp",
    ] {
        assert!(dialect.contains(&format!("pub struct {op}")));
        assert!(dialect.contains(&format!("{op}::register(ctx)")));
    }
    assert!(dialect.contains("requires i32 member mask, i1 predicate, and i1 result"));
    assert!(dialect.contains("requires i32 member mask, i1 predicate, and i32 result"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::warp::all_sync"));
    assert!(!importer.contains("cuda_device::warp::uni_sync"));
    assert!(importer.contains("let vote = VoteSyncUniOp::build(ctx, member_mask, predicate)"));
    assert!(importer.contains("set_generated_intrinsic_marker(ctx, vote, \"v1:i0043\")"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("impl MirToLlvmConversion for VoteSyncUniOp"));
    assert!(lowering.contains("\"llvm_nvvm_vote_uni_sync\""));
    assert!(lowering.contains("convert_vote(ctx, rewriter, self.get_operation(), operands_info"));

    let record = vote_intrinsics(&catalog)
        .find(|record| record.id == "ballot_sync")
        .unwrap();
    assert_eq!(record.selections.len(), 2);
    let probe = render_probe(&catalog, record, "test-hash");
    assert!(probe.contains("define i32 @probe_ballot_sync(i32 %member_mask, i1 %predicate)"));
    assert!(probe.contains("define i32 @probe_ballot_sync_immediate(i1 %predicate)"));
    assert!(probe.contains("i32 -1, i1 %predicate"));

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    for abi_id in ["i0040", "i0041", "i0042", "i0043"] {
        assert!(raw.contains(&format!("pub unsafe fn {abi_id}")));
    }
    assert!(raw.contains("Every non-exited lane named in `mask`"));
}

#[test]
fn active_mask_and_warp_match_rendering_preserves_backend_and_abi_contracts() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(active_masks(&catalog).count(), 1);
    assert_eq!(warp_matches(&catalog).count(), 4);

    let dialect = render_dialect_active_mask(&catalog, "test-hash");
    assert!(dialect.contains("pub struct ActiveMaskOp"));
    assert!(dialect.contains("NOpdsInterface<0>, NResultsInterface<1>"));

    let match_dialect = render_dialect_warp_match(&catalog, "test-hash");
    for op in [
        "MatchAnySyncI32Op",
        "MatchAnySyncI64Op",
        "MatchAllSyncI32Op",
        "MatchAllSyncI64Op",
    ] {
        assert!(match_dialect.contains(&format!("pub struct {op}")));
        assert!(match_dialect.contains(&format!("{op}::register(ctx)")));
    }

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("IntrinsicBackend::LlvmNvptx => {"));
    assert!(lowering.contains("convert_active_mask(ctx, rewriter, op, operands_info)"));
    assert!(lowering.contains("IntrinsicBackend::LibNvvm =>"));
    assert!(lowering.contains("\"activemask.b32 $0;\""));
    assert!(lowering.contains("\"=r,~{memory}\""));
    assert!(lowering.contains("convert_match_any("));
    assert!(lowering.contains("convert_match_all("));
    assert!(lowering.contains("\"llvm_nvvm_match_all_sync_i64p\""));

    let match_all = warp_matches(&catalog)
        .find(|record| record.id == "match_all_sync")
        .unwrap();
    let probe = render_probe(&catalog, match_all, "test-hash");
    for suffix in ["rr", "ri", "ir", "ii"] {
        assert!(probe.contains(&format!("@probe_match_all_sync_{suffix}")));
    }
    assert!(probe.contains("declare { i32, i1 } @llvm.nvvm.match.all.sync.i32p"));

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub fn i0044() -> u32"));
    assert!(!raw.contains("pub unsafe fn i0044() -> u32"));
    assert!(raw.contains("pub unsafe fn i0045(_arg0: u32, _arg1: u32) -> u32"));
    assert!(raw.contains("pub unsafe fn i0046(_arg0: u32, _arg1: u64) -> u32"));
    assert!(raw.contains("pub unsafe fn i0047(_arg0: u32, _arg1: u32) -> u32"));
    assert!(raw.contains("pub unsafe fn i0048(_arg0: u32, _arg1: u64) -> u32"));
}

#[test]
fn warp_barrier_rendering_preserves_mask_and_void_contracts() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(warp_barriers(&catalog).count(), 1);

    let record = warp_barriers(&catalog).next().unwrap();
    assert_eq!(record.id, "sync_mask");
    assert_eq!(
        record.warp_barrier.as_ref().unwrap().adapter,
        WarpBarrierAdapter::DirectMemberMask
    );

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod warp_barrier;"));
    assert!(dialect_mod.contains("warp_barrier::register(ctx)"));

    let dialect = render_dialect_warp_barrier(&catalog, "test-hash");
    assert!(dialect.contains("pub struct BarWarpSyncOp"));
    assert!(dialect.contains("NOpdsInterface<1>, NResultsInterface<0>"));
    assert!(dialect.contains("vec![member_mask]"));
    assert!(dialect.contains("op.get_num_operands() != 1 || op.get_num_results() != 0"));
    assert!(dialect.contains("if !is_i32(ctx, op.get_operand(0).get_type(ctx))"));
    assert!(dialect.contains("requires exactly one member-mask operand and no results"));
    assert!(dialect.contains("member mask must be i32"));
    assert!(dialect.contains("BarWarpSyncOp::register(ctx)"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::warp::sync_mask"));
    assert!(importer.contains("let barrier = BarWarpSyncOp::build(ctx, member_mask)"));
    assert!(importer.contains("set_generated_intrinsic_marker(ctx, barrier, \"v1:i0049\")"));
    assert!(importer.contains("helpers::emit_goto(ctx, *target_idx, barrier"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("impl MirToLlvmConversion for BarWarpSyncOp"));
    assert!(
        lowering
            .contains("convert_bar_warp_sync(ctx, rewriter, self.get_operation(), operands_info)")
    );

    let probe = render_probe(&catalog, record, "test-hash");
    assert!(probe.contains("declare void @llvm.nvvm.bar.warp.sync(i32)"));
    assert!(probe.contains("define void @probe_sync_mask(i32 %member_mask)"));
    assert!(probe.contains("call void @llvm.nvvm.bar.warp.sync(i32 %member_mask)"));
    assert!(probe.contains("define void @probe_sync_mask_immediate()"));
    assert!(probe.contains("call void @llvm.nvvm.bar.warp.sync(i32 -1)"));

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub unsafe fn i0049(_arg0: u32) -> ()"));
    assert!(raw.contains("On `sm_6x` and earlier"));
    assert!(raw.contains("no lane outside `mask` may be active"));
    assert!(raw.contains("The barrier orders memory accesses among participating lanes"));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Warp-barrier contract"));
    assert!(reference.contains("no unnamed lane may be active"));
    assert!(reference.contains("Both immediate and register masks are admitted"));
}

#[test]
fn warp_shuffle_rendering_owns_all_i32_f32_and_i64_modes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(warp_shuffles(&catalog).count(), 12);

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod warp_shuffle;"));
    assert!(dialect_mod.contains("warp_shuffle::register(ctx)"));

    let dialect = render_dialect_warp_shuffle(&catalog, "test-hash");
    for op in [
        "ShflSyncIdxI32Op",
        "ShflSyncBflyI32Op",
        "ShflSyncDownI32Op",
        "ShflSyncUpI32Op",
        "ShflSyncIdxF32Op",
        "ShflSyncBflyF32Op",
        "ShflSyncDownF32Op",
        "ShflSyncUpF32Op",
        "ShflSyncIdxI64Op",
        "ShflSyncBflyI64Op",
        "ShflSyncDownI64Op",
        "ShflSyncUpI64Op",
    ] {
        assert!(dialect.contains(&format!("pub struct {op}")));
        assert!(dialect.contains(&format!("{op}::register(ctx)")));
    }
    assert!(dialect.contains("vec![member_mask, value, lane_or_delta]"));
    assert!(dialect.contains("requires i32 mask/lane and i32 value/result"));
    assert!(dialect.contains("requires i32 mask/lane and f32 value/result"));
    assert!(dialect.contains("requires i32 mask/lane and i64 value/result"));
    assert!(dialect.contains("fn is_i64"));
    assert!(dialect.contains("integer.width() == 64"));
    assert!(dialect.contains("IntegerType::get(ctx, 64, Signedness::Unsigned)"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::warp::shuffle_sync"));
    assert!(importer.contains("cuda_device::warp::shuffle_up_f32_sync"));
    assert!(
        importer.contains(
            "let shuffle = ShflSyncIdxI32Op::build(ctx, member_mask, value, lane_or_delta)"
        )
    );
    assert!(importer.contains("set_generated_intrinsic_marker(ctx, shuffle, \"v1:i0050\")"));
    for (name, op, marker) in [
        ("shuffle_u64_sync", "ShflSyncIdxI64Op", "v1:i0058"),
        ("shuffle_xor_u64_sync", "ShflSyncBflyI64Op", "v1:i0059"),
        ("shuffle_down_u64_sync", "ShflSyncDownI64Op", "v1:i0060"),
        ("shuffle_up_u64_sync", "ShflSyncUpI64Op", "v1:i0061"),
    ] {
        assert!(importer.contains(&format!(
            "cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::{}",
            marker.strip_prefix("v1:").unwrap()
        )));
        assert!(importer.contains(&format!("cuda_device::warp::{name}")));
        assert!(importer.contains(&format!(
            "let shuffle = {op}::build(ctx, member_mask, value, lane_or_delta)"
        )));
        assert!(importer.contains(&format!(
            "set_generated_intrinsic_marker(ctx, shuffle, {marker:?})"
        )));
    }

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("impl MirToLlvmConversion for ShflSyncIdxI32Op"));
    assert!(lowering.contains("convert_shuffle_i32(ctx, rewriter"));
    assert!(lowering.contains("\"llvm_nvvm_shfl_sync_idx_i32\", 31)"));
    assert!(lowering.contains("impl MirToLlvmConversion for ShflSyncUpF32Op"));
    assert!(lowering.contains("convert_shuffle_f32(ctx, rewriter"));
    assert!(lowering.contains("\"llvm_nvvm_shfl_sync_up_f32\", 0)"));

    for (op, mode, clamp) in [
        ("ShflSyncIdxI64Op", "idx", 31),
        ("ShflSyncBflyI64Op", "bfly", 31),
        ("ShflSyncDownI64Op", "down", 31),
        ("ShflSyncUpI64Op", "up", 0),
    ] {
        assert!(lowering.contains(&format!("impl MirToLlvmConversion for {op}")));
        assert!(lowering.contains(&format!(
                "convert_shuffle_i64(ctx, rewriter, self.get_operation(), operands_info, {mode:?}, {clamp})"
            )));
    }
    assert!(!lowering.contains("llvm_nvvm_shfl_sync_idx_i64"));
    assert!(!lowering.contains("llvm_nvvm_shfl_sync_bfly_i64"));
    assert!(!lowering.contains("llvm_nvvm_shfl_sync_down_i64"));
    assert!(!lowering.contains("llvm_nvvm_shfl_sync_up_i64"));

    for record in warp_shuffles(&catalog) {
        let probe = render_probe(&catalog, record, "test-hash");
        let shuffle = record.warp_shuffle.as_ref().unwrap();
        if shuffle.value_kind == WarpShuffleValueKind::I64 {
            let mode = match shuffle.mode {
                WarpShuffleMode::Idx => "idx",
                WarpShuffleMode::Bfly => "bfly",
                WarpShuffleMode::Down => "down",
                WarpShuffleMode::Up => "up",
            };
            let asm = format!(
                "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; shfl.sync.{mode}.b32 lo, lo, $2, {}, $3; shfl.sync.{mode}.b32 hi, hi, $2, {}, $3; mov.b64 $0, {{lo, hi}}; }}",
                shuffle.clamp, shuffle.clamp
            );
            assert!(probe.contains(&format!(
                "define i64 @probe_{}(i32 %member_mask, i64 %value, i32 %lane) #0",
                record.id
            )));
            assert!(probe.contains(&format!(
                    "call i64 asm sideeffect {asm:?}, \"=l,l,r,r\"(i64 %value, i32 %lane, i32 %member_mask) #0"
                )));
            assert_eq!(probe.matches("asm sideeffect").count(), 1);
            assert_eq!(probe.matches("attributes #0 = { convergent }").count(), 1);
            assert!(!probe.contains("declare i64 @llvm.nvvm.shfl"));
            for suffix in ["rr", "ri", "ir", "ii"] {
                assert!(!probe.contains(&format!("@probe_{}_{suffix}", record.id)));
            }
        } else {
            for suffix in ["rr", "ri", "ir", "ii"] {
                assert!(probe.contains(&format!("@probe_{}_{suffix}", record.id)));
            }
            assert!(probe.contains(&format!(", i32 {})", shuffle.clamp)));
        }
    }

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub unsafe fn i0050(_arg0: u32, _arg1: u32, _arg2: u32) -> u32"));
    assert!(raw.contains("pub unsafe fn i0057(_arg0: u32, _arg1: f32, _arg2: u32) -> f32"));
    for abi_id in ["i0058", "i0059", "i0060", "i0061"] {
        assert!(raw.contains(&format!(
            "pub unsafe fn {abi_id}(_arg0: u32, _arg1: u64, _arg2: u32) -> u64"
        )));
    }
    assert!(raw.contains("If the computed source lane is in range"));
    assert!(raw.contains(
        "If PTX marks the computed source out of range, the calling lane's input is copied"
    ));
    assert!(raw.contains("two `b32` shuffles in one convergent block"));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Warp-shuffle contracts"));
    assert!(reference.contains("inserts clamp `31` during lowering"));
    assert!(reference.contains("inserts clamp `0` during lowering"));
    assert!(reference.contains("One convergent, side-effecting inline-PTX block splits `i64`"));
    assert!(reference.contains("PTX-native source; no LLVM record"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/warp_shuffle.rs"
    )));
}

#[test]
fn sync_rendering_keeps_barrier_and_threadfence_contracts_explicit() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    let record = sync_intrinsics(&catalog).next().unwrap();
    assert_eq!(sync_intrinsics(&catalog).count(), 4);

    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    assert!(raw.contains("pub unsafe fn i0034() -> ()"));
    assert!(raw.contains("Every active thread in the CTA must reach the same barrier"));

    let dialect = render_dialect_sync(&catalog, "test-hash");
    assert!(dialect.contains("pub struct Barrier0Op"));
    assert!(dialect.contains("NOpdsInterface<0>, NResultsInterface<0>"));
    assert!(dialect.contains("Barrier0Op::register(ctx)"));

    let importer = render_importer(&catalog, "test-hash");
    assert!(importer.contains("cuda_device::thread::sync_threads"));
    assert!(importer.contains("cuda_device::sync_threads"));
    assert!(importer.contains("Barrier0Op::get_concrete_op_info()"));
    assert!(importer.contains("set_generated_intrinsic_marker(ctx, barrier, \"v1:i0034\")"));
    assert!(importer.contains("helpers::emit_goto(ctx, *target_idx, barrier"));

    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("impl MirToLlvmConversion for Barrier0Op"));
    assert!(lowering.contains("create_i32_const(ctx, rewriter, 0)"));
    assert!(lowering.contains("\"llvm_nvvm_barrier_cta_sync_aligned_all\""));
    assert!(lowering.contains("IntrinsicBackend::LlvmNvptx"));
    assert!(lowering.contains("IntrinsicBackend::LibNvvm"));
    assert!(lowering.contains("\"bar.sync 0;\", \"~{memory}\""));

    let probe = render_probe(&catalog, record, "test-hash");
    assert!(probe.contains("declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32)"));
    assert!(probe.contains("call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"));

    let targets = render_targets(&catalog, "test-hash");
    assert!(targets.contains("id: \"sync_threads\", abi_id: \"i0034\""));
    assert!(targets.contains("source_record: \"BARRIER_CTA_SYNC_ALIGNED_ALL_i\""));
    assert!(targets.contains(
            "backend: GeneratedIntrinsicBackend::LlvmNvptx, requirement: GeneratedTargetRequirement { minimum_ptx: GeneratedPtxVersion::from_encoded(32), hardware: GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(20)]) }"
        ));
    assert!(targets.contains(
            "backend: GeneratedIntrinsicBackend::LibNvvm, requirement: GeneratedTargetRequirement { minimum_ptx: GeneratedPtxVersion::from_encoded(10), hardware: GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(75)]) }"
        ));
    assert!(
        !record
            .selections
            .iter()
            .any(|selection| selection.source_record.ends_with("_r"))
    );

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/sync.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/fence.rs")));

    let compatibility = render_compat_fence(&catalog, "test-hash");
    for (id, abi_id, op_type, op_name, llvm_identifier, ptx) in [
        (
            "threadfence_block",
            "i0298",
            "ThreadfenceBlockOp",
            "nvvm.threadfence_block",
            "llvm_nvvm_membar_cta",
            "membar.cta;",
        ),
        (
            "threadfence",
            "i0299",
            "ThreadfenceOp",
            "nvvm.threadfence",
            "llvm_nvvm_membar_gl",
            "membar.gl;",
        ),
        (
            "threadfence_system",
            "i0300",
            "ThreadfenceSystemOp",
            "nvvm.threadfence_system",
            "llvm_nvvm_membar_sys",
            "membar.sys;",
        ),
    ] {
        let record = sync_intrinsics(&catalog)
            .find(|record| record.id == id)
            .unwrap();
        assert_eq!(record.rust.abi_id, abi_id);
        assert!(compatibility.contains(&format!("pub fn {id}()")));
        assert!(dialect.contains(&format!("pub struct {op_type}")));
        assert!(dialect.contains(&format!("name = \"{op_name}\"")));
        assert!(dialect.contains(&format!("{op_type}::register(ctx)")));
        assert!(importer.contains(&format!("cuda_device::fence::{id}")));
        assert!(importer.contains(&format!("cuda_device::{id}")));
        assert!(importer.contains(&format!(
            "set_generated_intrinsic_marker(ctx, barrier, \"v1:{abi_id}\")"
        )));
        assert!(lowering.contains(&format!("impl MirToLlvmConversion for {op_type}")));
        assert!(lowering.contains(&format!("\"{llvm_identifier}\"")));

        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains(&format!("declare void @{}()", llvm(record).symbol)));
        assert!(probe.contains(&format!("call void @{}()", llvm(record).symbol)));
        assert!(targets.contains(&format!("id: \"{id}\", abi_id: \"{abi_id}\"")));
        assert!(targets.contains(&format!("asm: \"{ptx}\"")));
    }
    for ptx in ["membar.cta;", "membar.gl;", "membar.sys;"] {
        assert!(!lowering.contains(&format!("\"{ptx}\"")));
    }
}

#[test]
fn basic_mbarrier_rendering_preserves_existing_paths_shapes_and_routes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    assert_eq!(mbarrier_basics(&catalog).count(), 5);

    let compatibility = render_compat_mbarrier_basic(&catalog, "test-hash");
    for signature in [
        "pub unsafe fn mbarrier_init(bar: *mut Barrier, expected_count: u32)",
        "pub unsafe fn mbarrier_arrive(bar: *const Barrier) -> u64",
        "pub unsafe fn mbarrier_arrive_no_complete(bar: *const Barrier, count: u32) -> u64",
        "pub unsafe fn mbarrier_test_wait(bar: *const Barrier, token: u64) -> bool",
        "pub unsafe fn mbarrier_inval(bar: *mut Barrier)",
    ] {
        assert!(compatibility.contains(signature));
    }
    assert!(compatibility.contains("eight-byte-aligned `Barrier` in shared memory"));
    assert!(compatibility.contains("expected_count` must be in `1..=0xFFFFF`"));
    let arrive = compatibility.find("pub unsafe fn mbarrier_arrive").unwrap();
    assert!(compatibility[..arrive].ends_with("#[inline(never)]\n"));
    assert!(!compatibility.contains("#[must_use]"));

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod mbarrier_basic;"));
    assert!(dialect_mod.contains("mbarrier_basic::register(ctx)"));
    let dialect = render_dialect_mbarrier_basic(&catalog, "test-hash");
    assert!(dialect.contains("address_space::GENERIC | address_space::SHARED"));
    assert!(dialect.contains("mbarrier expected count must be u32"));
    assert!(dialect.contains("mbarrier arrival token must be u64"));
    assert!(dialect.contains("mbarrier test-wait requires a u64 token and i1 result"));

    let importer = render_importer(&catalog, "test-hash");
    let lowering = render_lowering(&catalog, "test-hash");
    let targets = render_targets(&catalog, "test-hash");
    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    for record in mbarrier_basics(&catalog) {
        assert!(dialect.contains(&format!("pub struct {}", record.dialect.op_type)));
        assert!(dialect.contains(&format!("{}::register(ctx)", record.dialect.op_type)));
        assert!(importer.contains(&record.rust.canonical_path));
        assert!(importer.contains(&format!(
            "set_generated_intrinsic_marker(ctx, mbarrier, {:?})",
            intrinsic_marker(&catalog, record)
        )));
        assert!(lowering.contains(&format!(
            "impl MirToLlvmConversion for {}",
            record.dialect.op_type
        )));
        assert!(targets.contains(&format!("id: {:?}", record.id)));
        assert!(raw.contains(&format!("pub unsafe fn {}", record.rust.abi_id)));

        let probe = render_probe(&catalog, record, "test-hash");
        assert!(
            probe.contains("%barrier = addrspacecast ptr %barrier_generic to ptr addrspace(3)")
        );
        let mbarrier = record.mbarrier_basic.as_ref().unwrap();
        let llvm_route = record
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
            .unwrap();
        let libnvvm_route = record
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap();
        let expected_llvm_mechanism = match mbarrier.operation {
            MbarrierBasicOperation::TestWait => BackendLoweringMechanism::InlinePtx,
            _ => BackendLoweringMechanism::TypedNvvm,
        };
        assert_eq!(llvm_route.mechanism, expected_llvm_mechanism);
        assert_eq!(libnvvm_route.mechanism, BackendLoweringMechanism::InlinePtx);
        match mbarrier.operation {
            MbarrierBasicOperation::Init => {
                assert!(
                    dialect.contains("MbarrierInitSharedOp::build(ctx, barrier, expected_count)")
                        || importer
                            .contains("MbarrierInitSharedOp::build(ctx, barrier, expected_count)")
                );
                assert!(
                    lowering.contains(
                        "convert_init(ctx, rewriter, self.get_operation(), operands_info)"
                    )
                );
                assert!(probe.contains(&format!(
                    "declare void @{}(ptr addrspace(3), i32)",
                    llvm(record).symbol
                )));
            }
            MbarrierBasicOperation::Arrive => {
                assert!(importer.contains("MbarrierArriveSharedOp::build(ctx, barrier)"));
                assert!(lowering.contains(
                    "convert_arrive(ctx, rewriter, self.get_operation(), operands_info)"
                ));
                assert!(probe.contains(&format!(
                    "declare i64 @{}(ptr addrspace(3))",
                    llvm(record).symbol
                )));
            }
            MbarrierBasicOperation::ArriveNoComplete => {
                assert!(
                    importer
                        .contains("MbarrierArriveNoCompleteSharedOp::build(ctx, barrier, count)")
                );
                assert!(lowering.contains(
                    "convert_arrive_no_complete(ctx, rewriter, self.get_operation(), operands_info)"
                ));
                assert!(probe.contains(&format!(
                    "declare i64 @{}(ptr addrspace(3), i32)",
                    llvm(record).symbol
                )));
                assert!(probe.contains("i32 %count"));
                assert!(probe.contains("ret i64 %state"));
            }
            MbarrierBasicOperation::TestWait => {
                assert!(importer.contains("MbarrierTestWaitSharedOp::build(ctx, barrier, token)"));
                assert!(lowering.contains(
                    "convert_test_wait(ctx, rewriter, self.get_operation(), operands_info)"
                ));
                assert!(probe.contains("mbarrier.test_wait.shared.b64"));
                assert!(probe.contains("asm sideeffect"));
                assert!(probe.contains("attributes #0 = { convergent }"));
                assert!(!probe.contains(&format!("declare i1 @{}", llvm(record).symbol)));
            }
            MbarrierBasicOperation::Inval => {
                assert!(importer.contains("MbarrierInvalSharedOp::build(ctx, barrier)"));
                assert!(
                    lowering.contains(
                        "convert_inval(ctx, rewriter, self.get_operation(), operands_info)"
                    )
                );
                assert!(probe.contains(&format!(
                    "declare void @{}(ptr addrspace(3))",
                    llvm(record).symbol
                )));
            }
        }
    }

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Basic mbarrier contracts"));
    assert!(reference.contains("inline PTX on both backends"));
    assert!(reference.contains("typed NVVM intrinsic with LLVM-NVPTX"));
    assert!(reference.contains("inline PTX with libNVVM"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/mbarrier_basic.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/mbarrier_basic.rs"
    )));
}

#[test]
fn extended_mbarrier_rendering_preserves_all_manual_contracts() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    validate_renderable(&catalog).unwrap();
    let records = mbarrier_extended(&catalog).collect::<Vec<_>>();
    assert_eq!(records.len(), 11);

    let compatibility = render_compat_mbarrier_extended(&catalog, "test-hash");
    for signature in [
        "pub unsafe fn mbarrier_arrive_expect_tx(bar: *const Barrier, _tx_count: u32, bytes: u32) -> u64",
        "pub unsafe fn mbarrier_arrive_expect_tx_cluster(bar: *const Barrier, _tx_count: u32, bytes: u32) -> u64",
        "pub unsafe fn mbarrier_arrive_cluster(remote_bar_addr: u64)",
        "pub unsafe fn mbarrier_try_wait(bar: *const Barrier, token: u64) -> bool",
        "pub unsafe fn mbarrier_try_wait_parity(bar: *const Barrier, parity: u32) -> bool",
        "pub unsafe fn mbarrier_try_wait_parity_cluster(bar: *const Barrier, parity: u32) -> bool",
        "pub unsafe fn fence_proxy_async_shared_cta()",
        "pub unsafe fn fence_mbarrier_init_release_cluster()",
        "pub unsafe fn fence_proxy_async_generic_release_shared_cta_cluster()",
        "pub unsafe fn fence_proxy_async_generic_acquire_shared_cluster_cluster()",
        "pub unsafe fn nanosleep(ns: u32)",
    ] {
        assert!(compatibility.contains(signature), "missing {signature}");
    }
    assert!(!compatibility.contains("#[must_use]"));

    let dialect_mod = render_dialect_mod(&catalog, "test-hash");
    assert!(dialect_mod.contains("mod mbarrier_extended;"));
    assert!(dialect_mod.contains("pub use mbarrier_extended::*;"));
    assert!(dialect_mod.contains("mbarrier_extended::register(ctx);"));

    let dialect = render_dialect_mbarrier_extended(&catalog, "test-hash");
    let importer = render_importer(&catalog, "test-hash");
    let lowering = render_lowering(&catalog, "test-hash");
    let raw = render_raw_abi(&catalog, "test-hash").unwrap();
    for record in &records {
        assert!(dialect.contains(&format!("pub struct {}", record.dialect.op_type)));
        assert!(dialect.contains(&format!("{}::register(ctx);", record.dialect.op_type)));
        assert!(importer.contains(&record.rust.canonical_path));
        assert!(importer.contains(&record.rust.compatibility_paths[0]));
        assert!(lowering.contains(&format!(
            "impl MirToLlvmConversion for {}",
            record.dialect.op_type
        )));
        assert!(raw.contains(&format!("pub unsafe fn {}", record.rust.abi_id)));
        for route in &record.backend_lowerings {
            assert_eq!(route.mechanism, BackendLoweringMechanism::InlinePtx);
        }
        let (template, constraints) = crate::resolve::mbarrier_extended_inline_recipe(
            record.mbarrier_extended.as_ref().unwrap().operation,
        );
        assert!(lowering.contains(&format!("{template:?}, {constraints:?}")));
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains("asm sideeffect"));
        assert!(probe.contains(template));
        assert!(probe.contains("~{memory}"));
        assert!(probe.contains("attributes #0 = { convergent }"));
    }

    assert!(dialect.contains("address_space::GENERIC | address_space::SHARED"));
    assert!(dialect.contains("mbarrier arrival requires u32 bytes and a u64 token"));
    assert!(dialect.contains("mbarrier wait requires a u64 token and i1 result"));
    assert!(dialect.contains("mbarrier wait requires u32 parity and i1 result"));
    assert!(dialect.contains("remote mbarrier address must be u64"));
    assert!(dialect.contains("nanosleep duration must be u32"));

    assert!(importer.contains("MbarrierArriveExpectTxSharedOp::build(ctx, barrier, bytes)"));
    assert!(importer.contains("MbarrierArriveExpectTxClusterOp::build(ctx, barrier, bytes)"));
    assert!(
        !importer.contains("MbarrierArriveExpectTxSharedOp::build(ctx, barrier, tx_count, bytes)")
    );
    assert!(importer.contains("MbarrierArriveClusterOp::build(ctx, address)"));
    assert!(importer.contains("MbarrierTryWaitSharedOp::build(ctx, barrier, token)"));
    assert!(importer.contains("MbarrierTryWaitParitySharedOp::build(ctx, barrier, parity)"));
    assert!(importer.contains("MbarrierTryWaitParityClusterOp::build(ctx, barrier, parity)"));
    assert!(importer.contains("FenceProxyAsyncSharedCtaOp::build(ctx)"));
    assert!(importer.contains("FenceMbarrierInitReleaseClusterOp::build(ctx)"));
    assert!(importer.contains("NanosleepOp::build(ctx, ns)"));

    assert!(lowering.contains("cast_to_shared_addrspace"));
    assert!(lowering.contains("trunc_to_i1"));
    assert!(lowering.contains("DefiningEntity::Op(result_op)"));
    assert!(lowering.contains("rewriter.erase_operation(ctx, op)"));
    assert!(lowering.contains("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];"));
    let remote = records
        .iter()
        .find(|record| record.id == "mbarrier_arrive_cluster")
        .unwrap();
    assert!(remote.llvm.is_none());
    assert!(matches!(remote.source, IntrinsicSource::PtxNative { .. }));
    assert!(!render_probe(&catalog, remote, "test-hash").contains("declare"));

    let reference = render_reference(&catalog, "test-hash");
    assert!(reference.contains("## Extended mbarrier contracts"));
    assert!(reference.contains("LLVM 22 TableGen"));
    assert!(reference.contains("PTX-native raw-address contract"));

    let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/cuda-device/src/generated/mbarrier_extended.rs"
    )));
    assert!(outputs.contains_key(&PathBuf::from(
        "crates/dialect-nvvm/src/ops/generated/mbarrier_extended.rs"
    )));
}

#[test]
fn lowering_imports_cast_to_shared_addrspace_exactly_once_across_families() {
    // The stmatrix, cluster_memory, and mbarrier_extended families all pull
    // the `cast_to_shared_addrspace` helper from `convert::intrinsics`. Each
    // sharded lowering file must import the name at most once in the raw
    // generator output (a duplicate inside one use group is a hard rustc
    // error, E0252, that the rustfmt pass would silently dedupe), and every
    // shard whose body calls the helper must import it exactly once.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    assert!(stmatrices(&catalog).next().is_some());
    assert!(cluster_memory(&catalog).next().is_some());
    assert!(mbarrier_extended(&catalog).next().is_some());

    let import_count = |contents: &str| {
        // Everything before the first rendered item is the module header
        // plus the use block; helper call sites in the item bodies below
        // must not count.
        let bodies_start = contents
            .find("#[op_interface_impl]")
            .or_else(|| contents.find("mod "))
            .expect("lowering output must contain rendered items");
        contents[..bodies_start]
            .matches("cast_to_shared_addrspace")
            .count()
    };
    let shard_import_count = |catalog: &CatalogFile, shard: &str| {
        let files = render_lowering_files(catalog, "test-hash");
        let (_, contents) = files
            .iter()
            .find(|(path, _)| path.ends_with(format!("{shard}.rs")))
            .unwrap_or_else(|| panic!("missing lowering shard `{shard}`"));
        import_count(contents)
    };

    for (path, contents) in render_lowering_files(&catalog, "test-hash") {
        assert!(
            import_count(&contents) <= 1,
            "cast_to_shared_addrspace must be imported at most once in the \
                 raw output of {}",
            path.display()
        );
    }
    // The shared stmatrix converter lives in mod.rs; the cluster_memory and
    // mbarrier_extended impls call the helper directly in their own shards.
    assert_eq!(shard_import_count(&catalog, "mod"), 1);
    assert_eq!(shard_import_count(&catalog, "cluster_memory"), 1);
    assert_eq!(shard_import_count(&catalog, "mbarrier_extended"), 1);

    // Dropping stmatrix must not suppress the import where it is still
    // needed: cluster_memory keeps using the helper.
    let mut without_stmatrix = catalog;
    without_stmatrix
        .intrinsics
        .retain(|record| record.family != "stmatrix");
    assert!(stmatrices(&without_stmatrix).next().is_none());
    assert_eq!(
        shard_import_count(&without_stmatrix, "cluster_memory"),
        1,
        "cast_to_shared_addrspace must still be imported when only \
             cluster_memory and mbarrier_extended need it"
    );
}

#[test]
fn elect_dialect_verifies_its_scalar_shape() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = crate::resolve::resolve(&repo_root).unwrap();
    let dialect = render_dialect_elect(&catalog, "test-hash");

    assert!(dialect.contains("impl Verify for ElectSyncOp"));
    assert!(dialect.contains("is_integer_width(ctx, op.get_operand(0).get_type(ctx), 32)"));
    assert!(dialect.contains("is_integer_width(ctx, op.get_result(0).get_type(ctx), 32)"));
    assert!(dialect.contains("is_integer_width(ctx, op.get_result(1).get_type(ctx), 1)"));
    assert!(!dialect.contains("verifier = \"succ\""));
}
