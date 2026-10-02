/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedFile, ImportedIntrinsic, IntrinsicBackend, IntrinsicSource,
    OverlayIntrinsic, PreSm70MemberMaskRule, VoteMode, WarpMatchAdapter, WarpShuffleAdapter,
    WarpShuffleMode, WarpShuffleOperandEncoding, WarpShuffleValueKind,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use crate::util::read_json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::fixtures::*;
use crate::resolve::families::*;
use crate::resolve::guards::*;
use crate::resolve::materialize::*;
use crate::resolve::overlay::*;
use crate::resolve::policy::*;
use crate::resolve::targets::validate_arch_introduction_floor;

#[test]
fn pinned_active_mask_and_warp_match_recipes_resolve() {
    let records = pinned_active_mask_and_warp_match_records();
    assert_eq!(records.len(), 5);

    for (policy, declaration) in records.values() {
        validate_imported_policy(policy, declaration).unwrap();
    }
}

#[test]
fn active_mask_and_warp_match_recipes_fail_closed() {
    let records = pinned_active_mask_and_warp_match_records();
    let reject = |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, expected: &str| {
        let error = validate_imported_policy(policy, declaration).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
    };

    let (active_mask, active_mask_declaration) = &records["active_mask"];
    let mut wrong_identity = active_mask.clone();
    wrong_identity.operation_key = "warp.active_mask.changed".into();
    reject(
        &wrong_identity,
        active_mask_declaration,
        "active-mask identity",
    );

    let mut wrong_effects = active_mask.clone();
    wrong_effects.memory = "none".into();
    reject(
        &wrong_effects,
        active_mask_declaration,
        "active-mask effects or target floor",
    );

    let (match_any, match_any_declaration) = &records["match_any_sync"];
    let mut wrong_adapter = match_any.clone();
    wrong_adapter.warp_match.as_mut().unwrap().adapter =
        WarpMatchAdapter::ProjectMaskDiscardPredicate;
    reject(
        &wrong_adapter,
        match_any_declaration,
        "warp-match participation, adapter, or encoding",
    );

    let (match_all, match_all_declaration) = &records["match_all_sync"];
    let mut wrong_projection = match_all.clone();
    wrong_projection.dialect_results.push("i1".into());
    reject(
        &wrong_projection,
        match_all_declaration,
        "two-operand warp-match lowering recipe",
    );

    let (match_any_i64, match_any_i64_declaration) = &records["match_any_i64_sync"];
    let mut incomplete_selections = match_any_i64_declaration.clone();
    incomplete_selections.selections.pop();
    reject(
        match_any_i64,
        &incomplete_selections,
        "exactly ii/ir/ri/rr selections",
    );

    let (match_all_i64, match_all_i64_declaration) = &records["match_all_i64_sync"];
    let mut wrong_predicates = match_all_i64_declaration.clone();
    wrong_predicates.selections[0].predicates[0] = "Subtarget->getPTXVersion() >= 61".into();
    reject(
        match_all_i64,
        &wrong_predicates,
        "PTX shape, predicates, or constraints",
    );
}

#[test]
fn ptx_native_source_provenance_fails_closed() {
    let mut mixed = packed_policy("packed_atomic_add_f16x2");
    mixed.source_record = Some("invented_llvm_record".into());
    assert!(
        resolve_policy_source(&mixed)
            .unwrap_err()
            .to_string()
            .contains("mixes tagged source provenance")
    );

    let mut fake_llvm = packed_policy("packed_atomic_add_f16x2");
    fake_llvm.llvm_symbol = Some("llvm.fake.packed.atomic".into());
    fake_llvm.llvm_arguments = vec!["ptr".into(), "i32".into()];
    fake_llvm.llvm_results = vec!["i32".into()];
    assert!(
        validate_ptx_native_policy(&fake_llvm)
            .unwrap_err()
            .to_string()
            .contains("must not invent LLVM source facts")
    );

    let mut wrong_instruction = packed_policy("packed_atomic_add_f16x2");
    wrong_instruction.source = Some(IntrinsicSource::PtxNative {
        instruction: "atom.global.add.noftz.bf16x2".into(),
    });
    assert!(
        validate_ptx_native_policy(&wrong_instruction)
            .unwrap_err()
            .to_string()
            .contains("does not match its packed format")
    );

    let mut wrong_kind = packed_policy("packed_atomic_add_f16x2");
    wrong_kind.source = Some(IntrinsicSource::LlvmImported {
        source_record: "invented_llvm_record".into(),
    });
    assert!(
        validate_ptx_native_policy(&wrong_kind)
            .unwrap_err()
            .to_string()
            .contains("source kind and imported declaration disagree")
    );
}

#[test]
fn vote_modes_keep_exact_abi_identity_and_both_selection_encodings() {
    for mode in [
        VoteMode::All,
        VoteMode::Any,
        VoteMode::Ballot,
        VoteMode::Uni,
    ] {
        let policy = vote_policy(mode);
        let declaration = vote_declaration(mode);
        validate_imported_policy(&policy, &declaration).unwrap();
        assert_eq!(
            policy.vote.as_ref().unwrap().legacy_pre_sm70,
            PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
        );

        let selected: Vec<_> = declaration
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(&policy, selection).unwrap())
            .collect();
        assert_eq!(selected.len(), 2);
        assert!(
            selected.iter().any(|selection| {
                selection.source_record == vote_recipe(mode).immediate_selection
            })
        );
        assert!(
            selected.iter().any(|selection| {
                selection.source_record == vote_recipe(mode).register_selection
            })
        );

        let mut record = evidence();
        record.id = policy.id.clone();
        record.source_record = policy.source_record.clone();
        record.llvm_symbol = policy.llvm_symbol.clone();
        record.llvm_arguments = policy.llvm_arguments.clone();
        record.llvm_results = policy.llvm_results.clone();
        record.expected_ptx = policy.expected_ptx.clone();
        let resolved = resolve_record(
            &policy,
            resolve_policy_source(&policy).unwrap(),
            Some(&declaration),
            &record,
            "test",
            "LLVM version test",
            "0123456789abcdef",
            vec![],
            1,
        )
        .unwrap();
        assert_eq!(resolved.selections.len(), 2);
        assert_eq!(resolved.vote, policy.vote);
    }
}

#[test]
fn vote_contract_rejects_unreviewed_identity_effect_and_selection_changes() {
    let valid = vote_policy(VoteMode::All);
    let declaration = vote_declaration(VoteMode::All);

    let mut wrong_abi = valid.clone();
    wrong_abi.abi_id = "i0041".into();
    assert!(
        validate_imported_policy(&wrong_abi, &declaration)
            .unwrap_err()
            .to_string()
            .contains("vote identity")
    );

    let mut safe = valid.clone();
    safe.safe = true;
    safe.safe_allowlist_reason = Some("incorrectly hides participation obligations".into());
    assert!(
        validate_imported_policy(&safe, &declaration)
            .unwrap_err()
            .to_string()
            .contains("unsafe must-use vote")
    );

    let mut wrong_memory = valid.clone();
    wrong_memory.memory = "none".into();
    assert!(
        validate_imported_policy(&wrong_memory, &declaration)
            .unwrap_err()
            .to_string()
            .contains("vote effects")
    );

    let mut register_only_mask = valid.clone();
    register_only_mask.expected_ptx.operands[2] = OperandPattern::Register;
    assert!(
        validate_imported_policy(&register_only_mask, &declaration)
            .unwrap_err()
            .to_string()
            .contains("expected PTX")
    );

    let mut one_selection = declaration.clone();
    one_selection.selections.pop();
    assert!(
        validate_imported_policy(&valid, &one_selection)
            .unwrap_err()
            .to_string()
            .contains("immediate/register selection pair")
    );

    let mut different_predicates = declaration;
    different_predicates.selections[1].predicates[0] = "Subtarget->getPTXVersion() >= 61".into();
    assert!(
        validate_imported_policy(&valid, &different_predicates)
            .unwrap_err()
            .to_string()
            .contains("disagree on PTX shape")
    );
}

#[test]
fn uni_vote_is_raw_only_while_existing_votes_keep_compatibility_paths() {
    for mode in [VoteMode::All, VoteMode::Any, VoteMode::Ballot] {
        assert_eq!(vote_policy(mode).compatibility_rust_paths.len(), 1);
    }
    let uni = vote_policy(VoteMode::Uni);
    assert!(uni.compatibility_rust_paths.is_empty());

    let mut invented_compatibility_path = uni.clone();
    invented_compatibility_path.compatibility_rust_paths =
        vec!["cuda_device::warp::uni_sync".into()];
    assert!(
        validate_imported_policy(
            &invented_compatibility_path,
            &vote_declaration(VoteMode::Uni),
        )
        .unwrap_err()
        .to_string()
        .contains("reviewed compatibility path")
    );
}

#[test]
fn warp_shuffle_variants_keep_exact_identity_clamp_and_eight_selections() {
    for (mode, value_kind, clamp) in [
        (WarpShuffleMode::Idx, WarpShuffleValueKind::I32, 31),
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::I32, 31),
        (WarpShuffleMode::Down, WarpShuffleValueKind::I32, 31),
        (WarpShuffleMode::Up, WarpShuffleValueKind::I32, 0),
        (WarpShuffleMode::Idx, WarpShuffleValueKind::F32, 31),
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::F32, 31),
        (WarpShuffleMode::Down, WarpShuffleValueKind::F32, 31),
        (WarpShuffleMode::Up, WarpShuffleValueKind::F32, 0),
    ] {
        let policy = warp_shuffle_policy(mode, value_kind);
        let declaration = warp_shuffle_declaration(mode, value_kind);
        validate_imported_policy(&policy, &declaration).unwrap();

        assert_eq!(policy.warp_shuffle.as_ref().unwrap().clamp, clamp);
        assert_eq!(
            declaration
                .selections
                .iter()
                .filter(|selection| selection_matches_policy(&policy, selection).unwrap())
                .count(),
            8
        );

        let mut record = evidence();
        record.id = policy.id.clone();
        record.source_record = policy.source_record.clone();
        record.llvm_symbol = policy.llvm_symbol.clone();
        record.llvm_arguments = policy.llvm_arguments.clone();
        record.llvm_results = policy.llvm_results.clone();
        record.expected_ptx = policy.expected_ptx.clone();
        let resolved = resolve_record(
            &policy,
            resolve_policy_source(&policy).unwrap(),
            Some(&declaration),
            &record,
            "test",
            "LLVM version test",
            "0123456789abcdef",
            vec![],
            1,
        )
        .unwrap();
        assert_eq!(resolved.selections.len(), 8);
        assert_eq!(resolved.warp_shuffle, policy.warp_shuffle);
    }
}

#[test]
fn pinned_warp_shuffle_records_match_the_closed_recipes() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (overlay, _) =
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let declarations: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|record| (record.source_record.as_str(), record))
        .collect();
    let all_policies: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_shuffle")
        .collect();
    assert_eq!(all_policies.len(), 12);
    let native_policies: Vec<_> = all_policies
        .iter()
        .copied()
        .filter(|record| record.source.is_some())
        .collect();
    assert_eq!(native_policies.len(), 4);
    for policy in native_policies {
        validate_ptx_native_policy(policy).unwrap();
    }
    let policies: Vec<_> = overlay
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_shuffle" && record.source_record.is_some())
        .collect();
    assert_eq!(policies.len(), 8);
    for policy in policies {
        let declaration = declarations[policy.source_record.as_deref().unwrap()];
        validate_imported_policy(policy, declaration).unwrap();
    }
}

#[test]
fn pinned_llvm_has_no_direct_i64_or_f64_shuffle_record() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
    let direct_64: Vec<_> = imported
        .intrinsics
        .iter()
        .filter(|record| {
            record.llvm_name.starts_with("llvm.nvvm.shfl")
                && (record.llvm_name.contains(".i64") || record.llvm_name.contains(".f64"))
        })
        .map(|record| record.llvm_name.as_str())
        .collect();
    assert!(
        direct_64.is_empty(),
        "unexpected LLVM records: {direct_64:?}"
    );
}

#[test]
fn i64_warp_shuffle_recipes_are_exact_ptx_native_pairs() {
    let cases = [
        (
            WarpShuffleMode::Idx,
            "shuffle_u64_sync",
            "i0058",
            "warp.shuffle.sync.idx.i64",
            "shfl.sync.idx.b32",
            "idx",
            31,
            "ShflSyncIdxI64Op",
            "nvvm.shfl_sync_idx_i64",
        ),
        (
            WarpShuffleMode::Bfly,
            "shuffle_xor_u64_sync",
            "i0059",
            "warp.shuffle.sync.bfly.i64",
            "shfl.sync.bfly.b32",
            "bfly",
            31,
            "ShflSyncBflyI64Op",
            "nvvm.shfl_sync_bfly_i64",
        ),
        (
            WarpShuffleMode::Down,
            "shuffle_down_u64_sync",
            "i0060",
            "warp.shuffle.sync.down.i64",
            "shfl.sync.down.b32",
            "down",
            31,
            "ShflSyncDownI64Op",
            "nvvm.shfl_sync_down_i64",
        ),
        (
            WarpShuffleMode::Up,
            "shuffle_up_u64_sync",
            "i0061",
            "warp.shuffle.sync.up.i64",
            "shfl.sync.up.b32",
            "up",
            0,
            "ShflSyncUpI64Op",
            "nvvm.shfl_sync_up_i64",
        ),
    ];

    for (mode, id, abi_id, operation_key, instruction, ptx_mode, clamp, op_type, op_name) in cases {
        let policy = warp_shuffle_policy(mode, WarpShuffleValueKind::I64);
        validate_ptx_native_policy(&policy).unwrap();

        assert_eq!(policy.id, id);
        assert_eq!(policy.abi_id, abi_id);
        assert_eq!(policy.operation_key, operation_key);
        assert_eq!(
            policy.source,
            Some(IntrinsicSource::PtxNative {
                instruction: instruction.into(),
            })
        );
        assert!(policy.source_record.is_none());
        assert!(policy.llvm_symbol.is_none());
        assert!(policy.resolved_llvm_symbol.is_none());
        assert!(policy.llvm_arguments.is_empty());
        assert!(policy.llvm_results.is_empty());
        assert_eq!(policy.rust_arguments, ["u32", "u64", "u32"]);
        assert_eq!(policy.rust_result, "u64");
        assert!(!policy.safe);
        assert!(policy.must_use);
        assert_eq!(policy.dialect_op_type, op_type);
        assert_eq!(policy.dialect_op_name, op_name);
        assert_eq!(policy.dialect_operands, ["i32", "i64", "i32"]);
        assert_eq!(policy.dialect_results, ["i64"]);
        assert_eq!(policy.lowering, "generated_warp_shuffle_i64_inline_ptx");

        let shuffle = policy.warp_shuffle.as_ref().unwrap();
        assert_eq!(shuffle.clamp, clamp);
        assert_eq!(
            shuffle.adapter,
            WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
        );
        assert_eq!(
            shuffle.lane_encoding,
            WarpShuffleOperandEncoding::RegisterOnly
        );
        assert_eq!(
            shuffle.mask_encoding,
            WarpShuffleOperandEncoding::RegisterOnly
        );
        assert_eq!(
            policy.expected_ptx,
            InstructionPattern::new(
                "shfl",
                &["sync", ptx_mode, "b32"],
                vec![
                    OperandPattern::Exact { value: "lo".into() },
                    OperandPattern::Exact { value: "lo".into() },
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: clamp.to_string(),
                    },
                    OperandPattern::Register,
                ],
            )
        );

        let routes: BTreeMap<_, _> = policy
            .backend_lowerings
            .iter()
            .map(|route| (route.backend, route))
            .collect();
        assert_eq!(routes.len(), 2);
        for backend in [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm] {
            assert_eq!(
                routes[&backend].mechanism,
                BackendLoweringMechanism::InlinePtx
            );
            assert_eq!(routes[&backend].minimum_ptx.as_deref(), Some("6.0"));
        }
        assert_eq!(
            routes[&IntrinsicBackend::LlvmNvptx].minimum_sm.as_deref(),
            Some("sm_30")
        );
        assert_eq!(
            routes[&IntrinsicBackend::LibNvvm].minimum_sm.as_deref(),
            Some("sm_75")
        );

        let mut record = evidence();
        record.id = policy.id.clone();
        record.source = policy.source.clone();
        record.source_record = None;
        record.llvm_symbol = None;
        record.llvm_arguments.clear();
        record.llvm_results.clear();
        record.expected_ptx = policy.expected_ptx.clone();
        let resolved = resolve_record(
            &policy,
            resolve_policy_source(&policy).unwrap(),
            None,
            &record,
            "test",
            "LLVM version test",
            "0123456789abcdef",
            vec![],
            1,
        )
        .unwrap();
        assert!(resolved.llvm.is_none());
        assert!(resolved.selections.is_empty());
        assert_eq!(resolved.warp_shuffle, policy.warp_shuffle);
    }
}

#[test]
fn i64_warp_shuffle_contract_rejects_unreviewed_changes() {
    let valid = warp_shuffle_policy(WarpShuffleMode::Idx, WarpShuffleValueKind::I64);
    validate_ptx_native_policy(&valid).unwrap();

    let reject = |policy: &OverlayIntrinsic, expected: &str| {
        let error = validate_ptx_native_policy(policy).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
    };

    let mut fabricated_llvm = valid.clone();
    fabricated_llvm.source = None;
    fabricated_llvm.source_record = Some("int_nvvm_shfl_sync_idx_i64".into());
    fabricated_llvm.llvm_symbol = Some("llvm.nvvm.shfl.sync.idx.i64".into());
    fabricated_llvm.llvm_arguments = vec!["i32".into(), "i64".into(), "i32".into(), "i32".into()];
    fabricated_llvm.llvm_results = vec!["i64".into()];
    reject(
        &fabricated_llvm,
        "source kind and imported declaration disagree",
    );

    let mut wrong_source = valid.clone();
    wrong_source.source = Some(IntrinsicSource::PtxNative {
        instruction: "shfl.sync.down.b32".into(),
    });
    reject(&wrong_source, "warp-shuffle identity");

    let mut wrong_adapter = valid.clone();
    wrong_adapter.warp_shuffle.as_mut().unwrap().adapter =
        WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp;
    reject(&wrong_adapter, "semantic or operand contract");

    let mut wrong_mode = valid.clone();
    wrong_mode.warp_shuffle.as_mut().unwrap().mode = WarpShuffleMode::Up;
    wrong_mode.warp_shuffle.as_mut().unwrap().clamp = 0;
    wrong_mode.expected_ptx.modifiers[1] = "up".into();
    wrong_mode.expected_ptx.operands[3] = OperandPattern::Exact { value: "0".into() };
    reject(&wrong_mode, "warp-shuffle identity");

    let mut wrong_clamp = valid.clone();
    wrong_clamp.warp_shuffle.as_mut().unwrap().clamp = 0;
    reject(&wrong_clamp, "semantic or operand contract");

    let mut broad_encoding = valid.clone();
    broad_encoding.warp_shuffle.as_mut().unwrap().lane_encoding =
        WarpShuffleOperandEncoding::RegisterOrImmediate;
    reject(&broad_encoding, "semantic or operand contract");

    let mut typed_backend = valid.clone();
    typed_backend.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
    reject(&typed_backend, "reviewed LLVM and libNVVM routes");

    let mut wrong_native_floor = valid.clone();
    wrong_native_floor.minimum_sm = Some("sm_70".into());
    reject(&wrong_native_floor, "target floor");

    let mut wrong_profile_floor = valid.clone();
    wrong_profile_floor
        .backend_lowerings
        .iter_mut()
        .find(|route| route.backend == IntrinsicBackend::LibNvvm)
        .unwrap()
        .minimum_sm = Some("sm_80".into());
    reject(&wrong_profile_floor, "profile floor");

    let mut safe = valid.clone();
    safe.safe = true;
    safe.safe_allowlist_reason = Some("incorrectly hides participation obligations".into());
    reject(&safe, "unsafe must-use warp-shuffle");

    let mut wrong_ptx = valid;
    wrong_ptx.expected_ptx.operands[0] = OperandPattern::Register;
    reject(&wrong_ptx, "closed shfl.sync recipe");
}

#[test]
fn warp_shuffle_contract_rejects_unreviewed_policy_changes() {
    let valid = warp_shuffle_policy(WarpShuffleMode::Idx, WarpShuffleValueKind::I32);
    let declaration = warp_shuffle_declaration(WarpShuffleMode::Idx, WarpShuffleValueKind::I32);

    let reject_policy = |policy: &OverlayIntrinsic, expected: &str| {
        let error = match validate_imported_policy(policy, &declaration) {
            Ok(()) => panic!("{expected} mutation was accepted"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
    };

    let mut wrong_identity = valid.clone();
    wrong_identity.operation_key = "warp.shuffle.sync.idx.changed".into();
    reject_policy(&wrong_identity, "warp-shuffle identity");

    let mut safe = valid.clone();
    safe.safe = true;
    safe.safe_allowlist_reason = Some("incorrectly hides participation obligations".into());
    reject_policy(&safe, "unsafe must-use warp-shuffle");

    let mut wrong_signature = valid.clone();
    wrong_signature.dialect_operands.pop();
    reject_policy(&wrong_signature, "closed warp-shuffle lowering recipe");

    let mut wrong_clamp = valid.clone();
    wrong_clamp.warp_shuffle.as_mut().unwrap().clamp = 0;
    reject_policy(&wrong_clamp, "semantic or operand contract");

    let mut missing_contract = valid.clone();
    missing_contract.warp_shuffle = None;
    reject_policy(&missing_contract, "closed warp-shuffle contract");

    let mut mixed_contract = valid.clone();
    mixed_contract.vote = vote_policy(VoteMode::All).vote;
    reject_policy(&mixed_contract, "mixes another generated-family contract");

    let mut wrong_backend_floor = valid.clone();
    wrong_backend_floor
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
        .unwrap()
        .minimum_sm = Some("sm_80".into());
    reject_policy(&wrong_backend_floor, "profile floor");
}

#[test]
fn warp_shuffle_contract_rejects_selection_drift() {
    let valid = warp_shuffle_policy(WarpShuffleMode::Down, WarpShuffleValueKind::F32);
    let declaration = warp_shuffle_declaration(WarpShuffleMode::Down, WarpShuffleValueKind::F32);
    let reject = |declaration: &ImportedIntrinsic, expected: &str| {
        let error = validate_imported_policy(&valid, declaration).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(expected), "unexpected error: {message}");
    };

    let mut missing_selection = declaration.clone();
    missing_selection.selections.pop();
    reject(
        &missing_selection,
        "eight distinct operand-encoding selections",
    );

    let mut duplicate_selection = declaration.clone();
    duplicate_selection.selections[7].source_record =
        duplicate_selection.selections[0].source_record.clone();
    reject(
        &duplicate_selection,
        "eight distinct operand-encoding selections",
    );

    let mut empty_selection_name = declaration.clone();
    empty_selection_name.selections[7].source_record.clear();
    reject(
        &empty_selection_name,
        "eight distinct operand-encoding selections",
    );

    let mut wrong_asm = declaration.clone();
    wrong_asm.selections[0].asm =
        "shfl.sync.up.b32 \t$dst, $src, $offset, $mask, $threadmask;".into();
    reject(&wrong_asm, "selections disagree on PTX shape");

    let mut wrong_predicate = declaration.clone();
    wrong_predicate.selections[0].predicates[0] = "Subtarget->getPTXVersion() >= 61".into();
    reject(&wrong_predicate, "selections disagree on PTX shape");

    let mut constrained = declaration;
    constrained.selections[0]
        .constraints
        .immediate_bindings
        .push(crate::model::ImportedImmediateBinding {
            argument_index: 2,
            value: 1,
        });
    reject(&constrained, "selections disagree on PTX shape");

    let mut wrong_classes =
        warp_shuffle_declaration(WarpShuffleMode::Down, WarpShuffleValueKind::F32);
    wrong_classes.classes.pop();
    reject(&wrong_classes, "class or effects");
}

#[test]
fn sync_threads_selects_only_the_fixed_immediate_barrier_recipe() {
    let policy = sync_policy();
    let declaration = sync_declaration();
    validate_imported_policy(&policy, &declaration).unwrap();

    let selected: Vec<_> = declaration
        .selections
        .iter()
        .filter(|selection| selection_matches_policy(&policy, selection).unwrap())
        .collect();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].source_record, "BARRIER_CTA_SYNC_ALIGNED_ALL_i");
    assert_eq!(selected[0].asm, "bar.sync \t$i;");
    assert!(policy.expected_ptx.matches("bar.sync 0;").unwrap());
    assert!(!policy.expected_ptx.matches(&selected[0].asm).unwrap());
    assert_eq!(policy.minimum_ptx, "1.0");
    assert!(policy.minimum_sm.is_none());
    let llvm_route = policy
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .unwrap();
    assert_eq!(llvm_route.minimum_ptx.as_deref(), Some("3.2"));
    assert_eq!(llvm_route.minimum_sm.as_deref(), Some("sm_20"));

    let resolved = resolve_record(
        &policy,
        resolve_policy_source(&policy).unwrap(),
        Some(&declaration),
        &sync_evidence(&policy),
        "test",
        "LLVM version test",
        "0123456789abcdef",
        vec![],
        1,
    )
    .unwrap();
    assert!(resolved.dialect.operands.is_empty());
    assert!(resolved.dialect.results.is_empty());
    assert_eq!(resolved.selections.len(), 1);
    assert_eq!(
        resolved.selections[0].source_record,
        "BARRIER_CTA_SYNC_ALIGNED_ALL_i"
    );
}

#[test]
fn sync_threads_recipe_rejects_unreviewed_selection_effect_and_floor_changes() {
    let valid = sync_policy();
    let declaration = sync_declaration();

    let mut register_only = declaration.clone();
    register_only
        .selections
        .retain(|selection| selection.source_record.ends_with("_r"));
    assert!(
        validate_imported_policy(&valid, &register_only)
            .unwrap_err()
            .to_string()
            .contains("does not agree")
    );

    let mut wrong_properties = declaration.clone();
    wrong_properties.properties.pop();
    assert!(
        validate_imported_policy(&valid, &wrong_properties)
            .unwrap_err()
            .to_string()
            .contains("sync properties")
    );

    let mut wrong_source = valid.clone();
    wrong_source.source_record = Some("int_nvvm_barrier0".into());
    assert!(
        validate_imported_policy(&wrong_source, &declaration)
            .unwrap_err()
            .to_string()
            .contains("sync identity")
    );

    let mut wrong_signature = valid.clone();
    wrong_signature.llvm_arguments.clear();
    assert!(
        validate_imported_policy(&wrong_signature, &declaration)
            .unwrap_err()
            .to_string()
            .contains("LLVM argument signature mismatch")
    );

    let mut wrong_path = valid.clone();
    wrong_path.compatibility_rust_paths.swap(0, 1);
    assert!(
        validate_imported_policy(&wrong_path, &declaration)
            .unwrap_err()
            .to_string()
            .contains("both cuda-device compatibility paths")
    );

    let mut safe = valid.clone();
    safe.safe = true;
    safe.safe_allowlist_reason = Some("incorrectly hides the participation contract".into());
    assert!(
        validate_imported_policy(&safe, &declaration)
            .unwrap_err()
            .to_string()
            .contains("unsafe sync_threads raw API")
    );

    let mut wrong_effect = valid.clone();
    wrong_effect.memory = "none".into();
    assert!(
        validate_imported_policy(&wrong_effect, &declaration)
            .unwrap_err()
            .to_string()
            .contains("sync effects")
    );

    let mut native_floor = valid.clone();
    // PTX 1.0 can name sm_11, but sync_threads must retain its all-target recipe.
    native_floor.minimum_sm = Some("sm_11".into());
    validate_arch_introduction_floor(&native_floor).unwrap();
    assert!(
        validate_imported_policy(&native_floor, &declaration)
            .unwrap_err()
            .to_string()
            .contains("native target floor")
    );

    let mut missing_profile_floor = valid;
    missing_profile_floor
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
        .unwrap()
        .minimum_sm = None;
    assert!(
        validate_imported_policy(&missing_profile_floor, &declaration)
            .unwrap_err()
            .to_string()
            .contains("profile floor")
    );

    let mut wrong_llvm_floor = sync_policy();
    wrong_llvm_floor
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .unwrap()
        .minimum_ptx = None;
    assert!(
        validate_imported_policy(&wrong_llvm_floor, &declaration)
            .unwrap_err()
            .to_string()
            .contains("profile floor")
    );
}

#[test]
fn sync_mask_matches_the_closed_warp_barrier_recipe() {
    let policy = warp_barrier_policy();
    let declaration = warp_barrier_declaration();
    validate_imported_policy(&policy, &declaration).unwrap();

    let selected: Vec<_> = declaration
        .selections
        .iter()
        .filter(|selection| selection_matches_policy(&policy, selection).unwrap())
        .collect();
    assert_eq!(selected.len(), 2);
    assert_eq!(
        selected
            .iter()
            .map(|selection| selection.source_record.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["INT_BAR_WARP_SYNC_I", "INT_BAR_WARP_SYNC_R"])
    );

    let mut record = evidence();
    record.id = policy.id.clone();
    record.source_record = policy.source_record.clone();
    record.llvm_symbol = policy.llvm_symbol.clone();
    record.llvm_arguments = policy.llvm_arguments.clone();
    record.llvm_results = policy.llvm_results.clone();
    record.expected_ptx = policy.expected_ptx.clone();
    let resolved = resolve_record(
        &policy,
        resolve_policy_source(&policy).unwrap(),
        Some(&declaration),
        &record,
        "test",
        "LLVM version test",
        "0123456789abcdef",
        vec![],
        1,
    )
    .unwrap();
    assert_eq!(resolved.selections.len(), 2);
    assert_eq!(resolved.warp_barrier, policy.warp_barrier);
}

#[test]
fn sync_mask_recipe_rejects_unreviewed_contract_and_selection_changes() {
    let valid = warp_barrier_policy();
    let declaration = warp_barrier_declaration();

    let mut wrong_identity = valid.clone();
    wrong_identity.id = "bar_warp_sync".into();
    assert!(
        validate_imported_policy(&wrong_identity, &declaration)
            .unwrap_err()
            .to_string()
            .contains("warp-barrier identity")
    );

    let mut missing_contract = valid.clone();
    missing_contract.warp_barrier = None;
    assert!(
        validate_imported_policy(&missing_contract, &declaration)
            .unwrap_err()
            .to_string()
            .contains("closed warp-barrier contract")
    );

    let mut safe_raw_api = valid.clone();
    safe_raw_api.safe = true;
    safe_raw_api.safe_allowlist_reason = Some("incorrectly hides participation rules".into());
    assert!(
        validate_imported_policy(&safe_raw_api, &declaration)
            .unwrap_err()
            .to_string()
            .contains("unsafe raw API")
    );

    let mut wrong_memory = valid.clone();
    wrong_memory.memory = "none".into();
    assert!(
        validate_imported_policy(&wrong_memory, &declaration)
            .unwrap_err()
            .to_string()
            .contains("effects or target floor")
    );

    let mut register_only = valid.clone();
    register_only.expected_ptx.operands[0] = OperandPattern::Register;
    assert!(
        validate_imported_policy(&register_only, &declaration)
            .unwrap_err()
            .to_string()
            .contains("expected PTX")
    );

    let mut one_selection = declaration.clone();
    one_selection.selections.pop();
    assert!(
        validate_imported_policy(&valid, &one_selection)
            .unwrap_err()
            .to_string()
            .contains("immediate/register selection pair")
    );

    let mut wrong_predicate = declaration.clone();
    wrong_predicate.selections[1].predicates[0] = "Subtarget->getPTXVersion() >= 61".into();
    assert!(
        validate_imported_policy(&valid, &wrong_predicate)
            .unwrap_err()
            .to_string()
            .contains("selections disagree")
    );

    let mut missing_libnvvm_floor = valid;
    missing_libnvvm_floor
        .backend_lowerings
        .iter_mut()
        .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
        .unwrap()
        .minimum_sm = None;
    assert!(
        validate_imported_policy(&missing_libnvvm_floor, &declaration)
            .unwrap_err()
            .to_string()
            .contains("profile floor")
    );
}
