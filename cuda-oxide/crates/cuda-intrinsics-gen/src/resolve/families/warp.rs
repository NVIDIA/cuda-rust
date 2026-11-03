/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    ActiveMaskAdapter, ActiveMaskObservation, BackendLoweringMechanism, ImportedIntrinsic,
    IntrinsicBackend, IntrinsicSource, MaskEncoding, MatchOperandEncoding, OverlayIntrinsic,
    PreSm70MemberMaskRule, VoteAdapter, VoteMode, VoteParticipation, WarpBarrierAdapter,
    WarpBarrierMaskEncoding, WarpBarrierMemoryOrdering, WarpBarrierParticipation, WarpMatchAdapter,
    WarpMatchMode, WarpMatchParticipation, WarpMatchValueWidth, WarpShuffleAdapter,
    WarpShuffleMode, WarpShuffleOperandEncoding, WarpShuffleParticipation, WarpShuffleSourceLane,
    WarpShuffleValueKind,
};
use crate::ptx::OperandPattern;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use super::*;

pub(in crate::resolve) fn validate_sync_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    if let Some(scope) = threadfence_scope_for_id(&policy.id) {
        return validate_threadfence_policy(policy, declaration, scope);
    }
    ensure!(
        policy.id == "sync_threads"
            && policy.abi_id == "i0034"
            && policy.operation_key == "synchronization.cta.barrier.aligned.all"
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some("int_nvvm_barrier_cta_sync_aligned_all")
            && policy.llvm_symbol.as_deref() == Some("llvm.nvvm.barrier.cta.sync.aligned.all")
            && policy.resolved_llvm_symbol.is_none(),
        "{} sync identity does not match the closed sync_threads recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "thread"
            && policy.rust_name == "sync_threads"
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == "cuda_intrinsics::thread::sync_threads"
            && policy.compatibility_rust_paths
                == [
                    "cuda_device::thread::sync_threads",
                    "cuda_device::sync_threads",
                ],
        "{} must preserve the unsafe sync_threads raw API and both cuda-device compatibility paths",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "Barrier0Op"
            && policy.dialect_op_name == "nvvm.barrier0"
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == ["i32"]
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_sync_threads",
        "{} is outside the fixed-zero sync_threads lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "cta"
            && policy.minimum_ptx == "1.0"
            && policy.minimum_sm.is_none()
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} sync effects or native target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.1 Parallel Synchronization and Communication Instructions: bar, barrier"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier",
        "{} sync PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrConvergent", "IntrNoCallback"],
        "{} sync properties disagree with the imported LLVM declaration",
        policy.id
    );
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with sync",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "bar"
            && policy.expected_ptx.modifiers == ["sync"]
            && policy.expected_ptx.operands == [OperandPattern::Exact { value: "0".into() }],
        "{} expected PTX does not match literal bar.sync 0",
        policy.id
    );

    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must define exactly the reviewed LLVM typed and libNVVM inline-PTX routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("3.2")
                    && lowering.minimum_sm.as_deref() == Some("sm_20")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed sync profile floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

pub(in crate::resolve) fn validate_vote_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let vote = policy
        .vote
        .as_ref()
        .with_context(|| format!("{} has no closed vote contract", policy.id))?;
    let recipe = vote_recipe(vote.mode);
    ensure!(
        vote.participation
            == VoteParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && vote.legacy_pre_sm70
                == PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
            && vote.adapter == VoteAdapter::DirectMaskPredicate
            && vote.mask_encoding == MaskEncoding::RegisterOrImmediate,
        "{} requests an unsupported vote participation, pre-sm70 rule, adapter, or mask encoding",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} vote identity does not match its closed mode recipe",
        policy.id
    );
    let expected_compatibility_paths: Vec<String> = if recipe.has_compatibility_path {
        vec![format!("cuda_device::warp::{}", recipe.rust_name)]
    } else {
        vec![]
    };
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", "bool"]
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths == expected_compatibility_paths,
        "{} must preserve its unsafe must-use vote raw API and reviewed compatibility path",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", "i1"]
            && policy.dialect_results == [recipe.llvm_result]
            && policy.llvm_arguments == ["i32", "i1"]
            && policy.llvm_results == [recipe.llvm_result]
            && policy.lowering == "generated_vote",
        "{} is outside the closed two-operand vote lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == recipe.rust_result
            && policy.targets == "all",
        "{} vote effects, carrier, or target floor disagree with its mode recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.14.10 Warp Vote Instructions: vote.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync",
        "{} vote PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrConvergent",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
        "{} vote memory and convergence effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.is_empty()
            && policy.packed_atomic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.redux.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with vote",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "vote"
            && policy.expected_ptx.modifiers == ["sync", recipe.ptx_mode, recipe.ptx_type]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::RegisterOrImmediate,
                ],
        "{} expected PTX does not match its closed vote mode recipe",
        policy.id
    );

    let expected_selection_records =
        BTreeSet::from([recipe.immediate_selection, recipe.register_selection]);
    let actual_selection_records: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        declaration.selections.len() == 2 && actual_selection_records == expected_selection_records,
        "{} vote declaration must contain exactly its immediate/register selection pair",
        policy.id
    );
    let expected_asm = format!(
        "vote.sync.{}.{} \t$dest, $pred, $mask;",
        recipe.ptx_mode, recipe.ptx_type
    );
    for selection in &declaration.selections {
        ensure!(
            selection.asm == expected_asm
                && selection.predicates
                    == [
                        "Subtarget->getPTXVersion() >= 60",
                        "Subtarget->getSmVersion() >= 30",
                    ]
                && selection.constraints.is_empty(),
            "{} vote immediate/register selections disagree on PTX shape, target predicates, or constraints",
            policy.id
        );
    }
    Ok(())
}

pub(in crate::resolve) struct VoteRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) rust_result: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) llvm_result: &'static str,
    pub(in crate::resolve) ptx_mode: &'static str,
    pub(in crate::resolve) ptx_type: &'static str,
    pub(in crate::resolve) immediate_selection: &'static str,
    pub(in crate::resolve) register_selection: &'static str,
    pub(in crate::resolve) has_compatibility_path: bool,
}

pub(in crate::resolve) fn vote_recipe(mode: VoteMode) -> VoteRecipe {
    match mode {
        VoteMode::All => VoteRecipe {
            id: "all_sync",
            abi_id: "i0040",
            operation_key: "warp.vote.sync.all.pred",
            source_record: "int_nvvm_vote_all_sync",
            llvm_symbol: "llvm.nvvm.vote.all.sync",
            rust_name: "all_sync",
            rust_result: "bool",
            dialect_op_type: "VoteSyncAllOp",
            dialect_op_name: "nvvm.vote_sync_all",
            llvm_result: "i1",
            ptx_mode: "all",
            ptx_type: "pred",
            immediate_selection: "VOTE_SYNC_ALLi",
            register_selection: "VOTE_SYNC_ALLr",
            has_compatibility_path: true,
        },
        VoteMode::Any => VoteRecipe {
            id: "any_sync",
            abi_id: "i0041",
            operation_key: "warp.vote.sync.any.pred",
            source_record: "int_nvvm_vote_any_sync",
            llvm_symbol: "llvm.nvvm.vote.any.sync",
            rust_name: "any_sync",
            rust_result: "bool",
            dialect_op_type: "VoteSyncAnyOp",
            dialect_op_name: "nvvm.vote_sync_any",
            llvm_result: "i1",
            ptx_mode: "any",
            ptx_type: "pred",
            immediate_selection: "VOTE_SYNC_ANYi",
            register_selection: "VOTE_SYNC_ANYr",
            has_compatibility_path: true,
        },
        VoteMode::Ballot => VoteRecipe {
            id: "ballot_sync",
            abi_id: "i0042",
            operation_key: "warp.vote.sync.ballot.b32",
            source_record: "int_nvvm_vote_ballot_sync",
            llvm_symbol: "llvm.nvvm.vote.ballot.sync",
            rust_name: "ballot_sync",
            rust_result: "u32",
            dialect_op_type: "VoteSyncBallotOp",
            dialect_op_name: "nvvm.vote_sync_ballot",
            llvm_result: "i32",
            ptx_mode: "ballot",
            ptx_type: "b32",
            immediate_selection: "VOTE_SYNC_BALLOTi",
            register_selection: "VOTE_SYNC_BALLOTr",
            has_compatibility_path: true,
        },
        VoteMode::Uni => VoteRecipe {
            id: "uni_sync",
            abi_id: "i0043",
            operation_key: "warp.vote.sync.uni.pred",
            source_record: "int_nvvm_vote_uni_sync",
            llvm_symbol: "llvm.nvvm.vote.uni.sync",
            rust_name: "uni_sync",
            rust_result: "bool",
            dialect_op_type: "VoteSyncUniOp",
            dialect_op_name: "nvvm.vote_sync_uni",
            llvm_result: "i1",
            ptx_mode: "uni",
            ptx_type: "pred",
            immediate_selection: "VOTE_SYNC_UNIi",
            register_selection: "VOTE_SYNC_UNIr",
            has_compatibility_path: false,
        },
    }
}

pub(in crate::resolve) fn validate_active_mask_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let active_mask = policy
        .active_mask
        .as_ref()
        .with_context(|| format!("{} has no closed active-mask contract", policy.id))?;
    ensure!(
        active_mask.observation == ActiveMaskObservation::ExecutingLanesAtInstruction
            && active_mask.adapter == ActiveMaskAdapter::DirectZeroOperandMask,
        "{} requests an unsupported active-mask observation or adapter",
        policy.id
    );
    ensure!(
        policy.id == "active_mask"
            && policy.abi_id == "i0044"
            && policy.operation_key == "warp.active_mask"
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some("int_nvvm_activemask")
            && policy.llvm_symbol.as_deref() == Some("llvm.nvvm.activemask")
            && policy.resolved_llvm_symbol.is_none(),
        "{} active-mask identity does not match the closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == "active_mask"
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.is_empty())
            && policy.public_rust_path == "cuda_intrinsics::warp::active_mask"
            && policy.compatibility_rust_paths == ["cuda_device::warp::active_mask"],
        "{} must preserve its safe must-use raw and compatibility APIs",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "ActiveMaskOp"
            && policy.dialect_op_name == "nvvm.activemask"
            && policy.dialect_operands.is_empty()
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results == ["i32"]
            && policy.lowering == "generated_active_mask",
        "{} is outside the closed zero-operand active-mask lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.2"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} active-mask effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.12 Parallel Synchronization and Communication Instructions: activemask"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-activemask",
        "{} active-mask PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrConvergent",
                "IntrHasSideEffects",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ]
            && declaration.selections.len() == 1
            && declaration.selections[0].source_record == "ACTIVEMASK"
            && declaration.selections[0].asm == "activemask.b32 \t$dest;"
            && declaration.selections[0].predicates
                == [
                    "Subtarget->getSmVersion() >= 30",
                    "Subtarget->getPTXVersion() >= 62",
                ]
            && declaration.selections[0].constraints.is_empty(),
        "{} active-mask declaration or selection facts changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "activemask"
            && policy.expected_ptx.modifiers == ["b32"]
            && policy.expected_ptx.operands == [OperandPattern::Register],
        "{} expected PTX does not match activemask.b32",
        policy.id
    );
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with active_mask",
        policy.id
    );

    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must keep the LLVM typed and libNVVM inline-PTX routes explicit",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("6.2")
                    && lowering.minimum_sm.as_deref() == Some("sm_30")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed active-mask floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

pub(in crate::resolve) struct WarpMatchRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) rust_value: &'static str,
    pub(in crate::resolve) llvm_value: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) ptx_mode: &'static str,
    pub(in crate::resolve) ptx_type: &'static str,
    pub(in crate::resolve) selections: [&'static str; 4],
    pub(in crate::resolve) adapter: WarpMatchAdapter,
}

pub(in crate::resolve) fn warp_match_recipe(
    mode: WarpMatchMode,
    width: WarpMatchValueWidth,
) -> WarpMatchRecipe {
    match (mode, width) {
        (WarpMatchMode::Any, WarpMatchValueWidth::B32) => WarpMatchRecipe {
            id: "match_any_sync",
            abi_id: "i0045",
            operation_key: "warp.match.sync.any.b32",
            source_record: "int_nvvm_match_any_sync_i32",
            llvm_symbol: "llvm.nvvm.match.any.sync.i32",
            rust_name: "match_any_sync",
            rust_value: "u32",
            llvm_value: "i32",
            dialect_op_type: "MatchAnySyncI32Op",
            dialect_op_name: "nvvm.match_any_sync_i32",
            ptx_mode: "any",
            ptx_type: "b32",
            selections: [
                "MATCH_ANY_SYNC_32ii",
                "MATCH_ANY_SYNC_32ir",
                "MATCH_ANY_SYNC_32ri",
                "MATCH_ANY_SYNC_32rr",
            ],
            adapter: WarpMatchAdapter::DirectMask,
        },
        (WarpMatchMode::Any, WarpMatchValueWidth::B64) => WarpMatchRecipe {
            id: "match_any_i64_sync",
            abi_id: "i0046",
            operation_key: "warp.match.sync.any.b64",
            source_record: "int_nvvm_match_any_sync_i64",
            llvm_symbol: "llvm.nvvm.match.any.sync.i64",
            rust_name: "match_any_i64_sync",
            rust_value: "u64",
            llvm_value: "i64",
            dialect_op_type: "MatchAnySyncI64Op",
            dialect_op_name: "nvvm.match_any_sync_i64",
            ptx_mode: "any",
            ptx_type: "b64",
            selections: [
                "MATCH_ANY_SYNC_64ii",
                "MATCH_ANY_SYNC_64ir",
                "MATCH_ANY_SYNC_64ri",
                "MATCH_ANY_SYNC_64rr",
            ],
            adapter: WarpMatchAdapter::DirectMask,
        },
        (WarpMatchMode::All, WarpMatchValueWidth::B32) => WarpMatchRecipe {
            id: "match_all_sync",
            abi_id: "i0047",
            operation_key: "warp.match.sync.all.b32",
            source_record: "int_nvvm_match_all_sync_i32p",
            llvm_symbol: "llvm.nvvm.match.all.sync.i32p",
            rust_name: "match_all_sync",
            rust_value: "u32",
            llvm_value: "i32",
            dialect_op_type: "MatchAllSyncI32Op",
            dialect_op_name: "nvvm.match_all_sync_i32",
            ptx_mode: "all",
            ptx_type: "b32",
            selections: [
                "MATCH_ALLP_SYNC_32ii",
                "MATCH_ALLP_SYNC_32ir",
                "MATCH_ALLP_SYNC_32ri",
                "MATCH_ALLP_SYNC_32rr",
            ],
            adapter: WarpMatchAdapter::ProjectMaskDiscardPredicate,
        },
        (WarpMatchMode::All, WarpMatchValueWidth::B64) => WarpMatchRecipe {
            id: "match_all_i64_sync",
            abi_id: "i0048",
            operation_key: "warp.match.sync.all.b64",
            source_record: "int_nvvm_match_all_sync_i64p",
            llvm_symbol: "llvm.nvvm.match.all.sync.i64p",
            rust_name: "match_all_i64_sync",
            rust_value: "u64",
            llvm_value: "i64",
            dialect_op_type: "MatchAllSyncI64Op",
            dialect_op_name: "nvvm.match_all_sync_i64",
            ptx_mode: "all",
            ptx_type: "b64",
            selections: [
                "MATCH_ALLP_SYNC_64ii",
                "MATCH_ALLP_SYNC_64ir",
                "MATCH_ALLP_SYNC_64ri",
                "MATCH_ALLP_SYNC_64rr",
            ],
            adapter: WarpMatchAdapter::ProjectMaskDiscardPredicate,
        },
    }
}

pub(in crate::resolve) fn validate_warp_match_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let warp_match = policy
        .warp_match
        .as_ref()
        .with_context(|| format!("{} has no closed warp-match contract", policy.id))?;
    let recipe = warp_match_recipe(warp_match.mode, warp_match.value_width);
    ensure!(
        warp_match.participation
            == WarpMatchParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && warp_match.adapter == recipe.adapter
            && warp_match.value_encoding == MatchOperandEncoding::RegisterOrImmediate
            && warp_match.mask_encoding == MatchOperandEncoding::RegisterOrImmediate,
        "{} requests an unsupported warp-match participation, adapter, or encoding",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} warp-match identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", recipe.rust_value]
            && policy.rust_result == "u32"
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::warp::{}", recipe.rust_name)],
        "{} must preserve its unsafe raw and stable compatibility paths",
        policy.id
    );
    let expected_llvm_results: &[&str] = match warp_match.mode {
        WarpMatchMode::Any => &["i32"],
        WarpMatchMode::All => &["i32", "i1"],
    };
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", recipe.llvm_value]
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments == ["i32", recipe.llvm_value]
            && policy.llvm_results == expected_llvm_results
            && policy.lowering == "generated_warp_match",
        "{} is outside the closed two-operand warp-match lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_70")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} warp-match effects, carrier, or target floor disagree with its recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.11 Parallel Synchronization and Communication Instructions: match.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync",
        "{} warp-match PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrConvergent",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
        "{} warp-match effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.is_empty()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with warp_match",
        policy.id
    );
    let destination = match warp_match.mode {
        WarpMatchMode::Any => OperandPattern::Register,
        WarpMatchMode::All => OperandPattern::RegisterPredicatePair,
    };
    ensure!(
        policy.expected_ptx.mnemonic == "match"
            && policy.expected_ptx.modifiers == [recipe.ptx_mode, "sync", recipe.ptx_type]
            && policy.expected_ptx.operands
                == [
                    destination,
                    OperandPattern::RegisterOrImmediate,
                    OperandPattern::RegisterOrImmediate,
                ],
        "{} expected PTX does not match its closed match.sync recipe",
        policy.id
    );
    let actual_selection_records: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        declaration.selections.len() == 4
            && actual_selection_records == BTreeSet::from(recipe.selections),
        "{} warp-match declaration must contain exactly ii/ir/ri/rr selections",
        policy.id
    );
    let destination = if warp_match.mode == WarpMatchMode::All {
        "$dest|$pred"
    } else {
        "$dest"
    };
    let expected_asm = format!(
        "match.{}.sync.{} \t{}, $value, $mask;",
        recipe.ptx_mode, recipe.ptx_type, destination
    );
    for selection in &declaration.selections {
        ensure!(
            selection.asm == expected_asm
                && selection.predicates
                    == [
                        "Subtarget->getPTXVersion() >= 60",
                        "Subtarget->getSmVersion() >= 70",
                    ]
                && selection.constraints.is_empty(),
            "{} warp-match selections disagree on PTX shape, predicates, or constraints",
            policy.id
        );
    }
    Ok(())
}

pub(in crate::resolve) fn validate_elect_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    ensure!(
        policy.id == "elect_sync"
            && policy.abi_id == "i0367"
            && policy.operation_key == "warp.elect.sync"
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some("int_nvvm_elect_sync")
            && policy.llvm_symbol.as_deref() == Some("llvm.nvvm.elect.sync")
            && policy.resolved_llvm_symbol.is_none(),
        "{} elect identity does not match the closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == "elect_sync"
            && policy.rust_arguments == ["u32"]
            && policy.rust_result == "(u32, bool)"
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == "cuda_intrinsics::warp::elect_sync"
            && policy.compatibility_rust_paths == ["cuda_device::warp::elect_sync"],
        "{} must keep its unsafe raw API and stable compatibility path",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "ElectSyncOp"
            && policy.dialect_op_name == "nvvm.elect_sync"
            && policy.dialect_operands == ["i32"]
            && policy.dialect_results == ["i32", "i1"]
            && policy.llvm_arguments == ["i32"]
            && policy.llvm_results == ["i32", "i1"]
            && policy.lowering == "generated_elect",
        "{} is outside the closed elect lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "8.0"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.ptx_result == "(u32, bool)"
            && policy.targets == "all",
        "{} elect effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.15 Parallel Synchronization and Communication Instructions: elect.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync",
        "{} elect PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic", "DefaultAttrsIntrinsic"]
            && declaration.properties == ["IntrConvergent", "IntrInaccessibleMemOnly"]
            && declaration.arguments == ["i32"]
            && declaration.results == ["i32", "i1"],
        "{} elect declaration facts changed",
        policy.id
    );
    ensure!(
        declaration.selections.len() == 2
            && declaration.selections.iter().all(|selection| {
                ["INT_ELECT_SYNC_I", "INT_ELECT_SYNC_R"].contains(&selection.source_record.as_str())
                    && selection.asm == "elect.sync \t$dest|$pred, $mask;"
                    && selection.predicates
                        == [
                            "Subtarget->getPTXVersion() >= 80",
                            "Subtarget->getSmVersion() >= 90",
                        ]
                    && selection.constraints.is_empty()
            }),
        "{} elect selections changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "elect"
            && policy.expected_ptx.modifiers == ["sync"]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::RegisterPredicatePair,
                    OperandPattern::RegisterOrImmediate,
                ],
        "{} expected PTX does not match elect.sync",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LlvmNvptx
                    && route.mechanism == BackendLoweringMechanism::TypedNvvm
                    && route.minimum_ptx.as_deref() == Some("8.0")
                    && route.minimum_sm.as_deref() == Some("sm_90")
                    && !route.evidence_profile.trim().is_empty()
            })
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LibNvvm
                    && route.mechanism == BackendLoweringMechanism::InlinePtx
                    && route.minimum_ptx.as_deref() == Some("8.0")
                    && route.minimum_sm.as_deref() == Some("sm_90")
                    && !route.evidence_profile.trim().is_empty()
            }),
        "{} must keep the LLVM typed and libNVVM inline-PTX routes explicit",
        policy.id
    );
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.cp_async_mbarrier.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.movmatrix.is_none()
            && policy.mbarrier_extended.is_none()
            && policy.register_mma.is_none()
            && policy.sparse_mma.is_none()
            && policy.prmt.is_none()
            && policy.cluster_barrier.is_none()
            && policy.wgmma_control.is_none()
            && policy.special_register.is_none()
            && policy.debug_control.is_none()
            && policy.cluster_memory.is_none()
            && policy.clc.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with elect",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) fn validate_warp_barrier_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let barrier = policy
        .warp_barrier
        .as_ref()
        .with_context(|| format!("{} has no closed warp-barrier contract", policy.id))?;
    ensure!(
        barrier.participation
            == WarpBarrierParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && barrier.legacy_pre_sm70
                == PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
            && barrier.adapter == WarpBarrierAdapter::DirectMemberMask
            && barrier.mask_encoding == WarpBarrierMaskEncoding::RegisterOrImmediate
            && barrier.memory_ordering == WarpBarrierMemoryOrdering::ParticipatingLanes,
        "{} requests an unsupported warp-barrier participation, legacy rule, adapter, mask encoding, or memory ordering",
        policy.id
    );
    ensure!(
        policy.id == "sync_mask"
            && policy.abi_id == "i0049"
            && policy.operation_key == "warp.barrier.sync.masked"
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some("int_nvvm_bar_warp_sync")
            && policy.llvm_symbol.as_deref() == Some("llvm.nvvm.bar.warp.sync")
            && policy.resolved_llvm_symbol.is_none(),
        "{} warp-barrier identity does not match the closed sync_mask recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == "sync_mask"
            && policy.rust_arguments == ["u32"]
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == "cuda_intrinsics::warp::sync_mask"
            && policy.compatibility_rust_paths == ["cuda_device::warp::sync_mask"],
        "{} must keep its unsafe raw API and safe cuda-device compatibility path distinct",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "BarWarpSyncOp"
            && policy.dialect_op_name == "nvvm.bar_warp_sync"
            && policy.dialect_operands == ["i32"]
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == ["i32"]
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_warp_barrier",
        "{} is outside the closed one-mask warp-barrier lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} warp-barrier effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.2 Parallel Synchronization and Communication Instructions: bar.warp.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-warp-sync",
        "{} warp-barrier PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrConvergent", "IntrNoCallback"],
        "{} warp-barrier effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with warp_barrier",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "bar"
            && policy.expected_ptx.modifiers == ["warp", "sync"]
            && policy.expected_ptx.operands == [OperandPattern::RegisterOrImmediate],
        "{} expected PTX does not match bar.warp.sync mask",
        policy.id
    );

    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                ]),
        "{} must define exactly the reviewed typed LLVM and libNVVM routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_30")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed warp-barrier profile floor",
            policy.id,
            lowering.backend
        );
    }

    let expected_selection_records = BTreeSet::from(["INT_BAR_WARP_SYNC_I", "INT_BAR_WARP_SYNC_R"]);
    let actual_selection_records: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        declaration.selections.len() == 2 && actual_selection_records == expected_selection_records,
        "{} warp-barrier declaration must contain exactly its immediate/register selection pair",
        policy.id
    );
    for selection in &declaration.selections {
        ensure!(
            selection.asm == "bar.warp.sync \t$i;"
                && selection.predicates
                    == [
                        "Subtarget->getPTXVersion() >= 60",
                        "Subtarget->getSmVersion() >= 30",
                    ]
                && selection.constraints.is_empty(),
            "{} warp-barrier selections disagree on PTX shape, target predicates, or constraints",
            policy.id
        );
    }
    Ok(())
}

pub(in crate::resolve) fn validate_warp_shuffle_policy(
    policy: &OverlayIntrinsic,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let shuffle = policy
        .warp_shuffle
        .as_ref()
        .with_context(|| format!("{} has no closed warp-shuffle contract", policy.id))?;
    let recipe = warp_shuffle_recipe(shuffle.mode, shuffle.value_kind);
    ensure!(
        shuffle.participation
            == WarpShuffleParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && shuffle.legacy_pre_sm70
                == PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
            && shuffle.source_lane
                == WarpShuffleSourceLane::InRangeSourceActiveAndNamedOutOfRangeCopiesSelf
            && shuffle.adapter == recipe.adapter
            && shuffle.clamp == recipe.clamp
            && shuffle.lane_encoding == recipe.operand_encoding
            && shuffle.mask_encoding == recipe.operand_encoding,
        "{} requests an unsupported warp-shuffle semantic or operand contract",
        policy.id
    );

    let source_matches = match recipe.source {
        WarpShuffleRecipeSource::LlvmImported {
            source_record,
            llvm_symbol,
        } => {
            policy.source.is_none()
                && policy.source_record.as_deref() == Some(source_record)
                && policy.llvm_symbol.as_deref() == Some(llvm_symbol)
                && policy.resolved_llvm_symbol.is_none()
        }
        WarpShuffleRecipeSource::PtxNative { instruction } => {
            policy.source
                == Some(IntrinsicSource::PtxNative {
                    instruction: instruction.into(),
                })
                && policy.source_record.is_none()
                && policy.llvm_symbol.is_none()
                && policy.resolved_llvm_symbol.is_none()
                && policy.llvm_arguments.is_empty()
                && policy.llvm_results.is_empty()
        }
    };
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && source_matches,
        "{} warp-shuffle identity does not match its closed mode and value recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", recipe.rust_value, "u32"]
            && policy.rust_result == recipe.rust_value
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::warp::{}", recipe.rust_name)],
        "{} must preserve its unsafe must-use warp-shuffle raw API and compatibility path",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", recipe.dialect_value, "i32"]
            && policy.dialect_results == [recipe.dialect_value]
            && policy.lowering == recipe.lowering
            && match recipe.source {
                WarpShuffleRecipeSource::LlvmImported { .. } => {
                    policy.llvm_arguments == ["i32", recipe.dialect_value, "i32", "i32"]
                        && policy.llvm_results == [recipe.dialect_value]
                }
                WarpShuffleRecipeSource::PtxNative { .. } => {
                    policy.llvm_arguments.is_empty() && policy.llvm_results.is_empty()
                }
            },
        "{} is outside the closed warp-shuffle lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == recipe.rust_value
            && policy.targets == "all",
        "{} warp-shuffle effects, carrier, or target floor disagree with its recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.9.6 Data Movement and Conversion Instructions: shfl.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync",
        "{} warp-shuffle PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    if let Some(declaration) = declaration {
        ensure!(
            matches!(recipe.source, WarpShuffleRecipeSource::LlvmImported { .. })
                && declaration.classes
                    == [
                        "ClangBuiltin",
                        "NVVMBuiltin",
                        "SDPatternOperator",
                        "Intrinsic"
                    ]
                && declaration.properties
                    == [
                        "IntrConvergent",
                        "IntrInaccessibleMemOnly",
                        "IntrNoCallback",
                    ],
            "{} warp-shuffle class or effects disagree with the imported declaration",
            policy.id
        );
    } else {
        ensure!(
            matches!(recipe.source, WarpShuffleRecipeSource::PtxNative { .. }),
            "{} imported warp shuffle is missing its LLVM declaration",
            policy.id
        );
    }
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with warp_shuffle",
        policy.id
    );
    let expected_operands = match recipe.adapter {
        WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp => vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::RegisterOrImmediate,
            OperandPattern::Exact {
                value: recipe.clamp.to_string(),
            },
            OperandPattern::RegisterOrImmediate,
        ],
        WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble => vec![
            OperandPattern::Exact { value: "lo".into() },
            OperandPattern::Exact { value: "lo".into() },
            OperandPattern::Register,
            OperandPattern::Exact {
                value: recipe.clamp.to_string(),
            },
            OperandPattern::Register,
        ],
    };
    ensure!(
        policy.expected_ptx.mnemonic == "shfl"
            && policy.expected_ptx.modifiers == ["sync", recipe.ptx_mode, "b32"]
            && policy.expected_ptx.operands == expected_operands,
        "{} expected PTX does not match its closed shfl.sync recipe",
        policy.id
    );

    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (IntrinsicBackend::LlvmNvptx, recipe.backend_mechanism),
                    (IntrinsicBackend::LibNvvm, recipe.backend_mechanism),
                ]),
        "{} must define exactly the reviewed LLVM and libNVVM routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == recipe.backend_mechanism
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_30")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == recipe.backend_mechanism
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed warp-shuffle profile floor",
            policy.id,
            lowering.backend
        );
    }

    if let Some(declaration) = declaration {
        let selection_records: BTreeSet<_> = declaration
            .selections
            .iter()
            .map(|selection| selection.source_record.as_str())
            .collect();
        ensure!(
            declaration.selections.len() == 8
                && selection_records.len() == 8
                && selection_records
                    .iter()
                    .all(|source_record| !source_record.trim().is_empty()),
            "{} warp-shuffle declaration must contain exactly eight distinct operand-encoding selections",
            policy.id
        );
        let expected_asm = format!(
            "shfl.sync.{}.b32 \t$dst, $src, $offset, $mask, $threadmask;",
            recipe.ptx_mode
        );
        for selection in &declaration.selections {
            ensure!(
                selection.asm == expected_asm
                    && selection.predicates
                        == [
                            "Subtarget->getPTXVersion() >= 60",
                            "Subtarget->getSmVersion() >= 30",
                        ]
                    && selection.constraints.is_empty(),
                "{} warp-shuffle selections disagree on PTX shape, target predicates, or constraints",
                policy.id
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(in crate::resolve) enum WarpShuffleRecipeSource {
    LlvmImported {
        source_record: &'static str,
        llvm_symbol: &'static str,
    },
    PtxNative {
        instruction: &'static str,
    },
}

pub(in crate::resolve) struct WarpShuffleRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source: WarpShuffleRecipeSource,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) rust_value: &'static str,
    pub(in crate::resolve) dialect_value: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) ptx_mode: &'static str,
    pub(in crate::resolve) clamp: u32,
    pub(in crate::resolve) adapter: WarpShuffleAdapter,
    pub(in crate::resolve) operand_encoding: WarpShuffleOperandEncoding,
    pub(in crate::resolve) lowering: &'static str,
    pub(in crate::resolve) backend_mechanism: BackendLoweringMechanism,
}

pub(in crate::resolve) fn warp_shuffle_recipe(
    mode: WarpShuffleMode,
    value_kind: WarpShuffleValueKind,
) -> WarpShuffleRecipe {
    match (mode, value_kind) {
        (WarpShuffleMode::Idx, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_sync",
            abi_id: "i0050",
            operation_key: "warp.shuffle.sync.idx.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_idx_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.idx.i32",
            },
            rust_name: "shuffle_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncIdxI32Op",
            dialect_op_name: "nvvm.shfl_sync_idx_i32",
            ptx_mode: "idx",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_xor_sync",
            abi_id: "i0051",
            operation_key: "warp.shuffle.sync.bfly.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_bfly_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.bfly.i32",
            },
            rust_name: "shuffle_xor_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncBflyI32Op",
            dialect_op_name: "nvvm.shfl_sync_bfly_i32",
            ptx_mode: "bfly",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Down, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_down_sync",
            abi_id: "i0052",
            operation_key: "warp.shuffle.sync.down.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_down_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.down.i32",
            },
            rust_name: "shuffle_down_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncDownI32Op",
            dialect_op_name: "nvvm.shfl_sync_down_i32",
            ptx_mode: "down",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Up, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_up_sync",
            abi_id: "i0053",
            operation_key: "warp.shuffle.sync.up.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_up_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.up.i32",
            },
            rust_name: "shuffle_up_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncUpI32Op",
            dialect_op_name: "nvvm.shfl_sync_up_i32",
            ptx_mode: "up",
            clamp: 0,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Idx, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_f32_sync",
            abi_id: "i0054",
            operation_key: "warp.shuffle.sync.idx.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_idx_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.idx.f32",
            },
            rust_name: "shuffle_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncIdxF32Op",
            dialect_op_name: "nvvm.shfl_sync_idx_f32",
            ptx_mode: "idx",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_xor_f32_sync",
            abi_id: "i0055",
            operation_key: "warp.shuffle.sync.bfly.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_bfly_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.bfly.f32",
            },
            rust_name: "shuffle_xor_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncBflyF32Op",
            dialect_op_name: "nvvm.shfl_sync_bfly_f32",
            ptx_mode: "bfly",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Down, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_down_f32_sync",
            abi_id: "i0056",
            operation_key: "warp.shuffle.sync.down.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_down_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.down.f32",
            },
            rust_name: "shuffle_down_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncDownF32Op",
            dialect_op_name: "nvvm.shfl_sync_down_f32",
            ptx_mode: "down",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Up, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_up_f32_sync",
            abi_id: "i0057",
            operation_key: "warp.shuffle.sync.up.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_up_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.up.f32",
            },
            rust_name: "shuffle_up_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncUpF32Op",
            dialect_op_name: "nvvm.shfl_sync_up_f32",
            ptx_mode: "up",
            clamp: 0,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Idx, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_u64_sync",
            abi_id: "i0058",
            operation_key: "warp.shuffle.sync.idx.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.idx.b32",
            },
            rust_name: "shuffle_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncIdxI64Op",
            dialect_op_name: "nvvm.shfl_sync_idx_i64",
            ptx_mode: "idx",
            clamp: 31,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_xor_u64_sync",
            abi_id: "i0059",
            operation_key: "warp.shuffle.sync.bfly.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.bfly.b32",
            },
            rust_name: "shuffle_xor_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncBflyI64Op",
            dialect_op_name: "nvvm.shfl_sync_bfly_i64",
            ptx_mode: "bfly",
            clamp: 31,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
        (WarpShuffleMode::Down, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_down_u64_sync",
            abi_id: "i0060",
            operation_key: "warp.shuffle.sync.down.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.down.b32",
            },
            rust_name: "shuffle_down_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncDownI64Op",
            dialect_op_name: "nvvm.shfl_sync_down_i64",
            ptx_mode: "down",
            clamp: 31,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
        (WarpShuffleMode::Up, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_up_u64_sync",
            abi_id: "i0061",
            operation_key: "warp.shuffle.sync.up.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.up.b32",
            },
            rust_name: "shuffle_up_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncUpI64Op",
            dialect_op_name: "nvvm.shfl_sync_up_i64",
            ptx_mode: "up",
            clamp: 0,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
    }
}
