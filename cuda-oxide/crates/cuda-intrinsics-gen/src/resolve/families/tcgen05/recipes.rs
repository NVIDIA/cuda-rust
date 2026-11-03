/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{Tcgen05Adapter, Tcgen05Operation, Tcgen05SourceContract};
use crate::ptx::OperandPattern;

pub(in crate::resolve) const TCGEN05_LLVM_TARGETS: &str = "sm_100a|sm_101a|sm_103a|sm_110a";
pub(in crate::resolve) const TCGEN05_LIBNVVM_TARGETS: &str = "sm_100a|sm_103a|sm_110a";
pub(in crate::resolve) struct Tcgen05Recipe {
    pub(in crate::resolve) operation: Tcgen05Operation,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_arguments: &'static [&'static str],
    pub(in crate::resolve) rust_result: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) dialect_operands: &'static [&'static str],
    pub(in crate::resolve) dialect_results: &'static [&'static str],
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_results: &'static [&'static str],
    pub(in crate::resolve) imported_classes: &'static [&'static str],
    pub(in crate::resolve) imported_properties: &'static [&'static str],
    pub(in crate::resolve) adapter: Tcgen05Adapter,
    pub(in crate::resolve) source_contract: Tcgen05SourceContract,
    pub(in crate::resolve) safe: bool,
    pub(in crate::resolve) safe_reason: Option<&'static str>,
    pub(in crate::resolve) memory: &'static str,
    pub(in crate::resolve) modifiers: Vec<String>,
    pub(in crate::resolve) operands: Vec<OperandPattern>,
    pub(in crate::resolve) selection_record: Option<&'static str>,
    pub(in crate::resolve) selection_asm: Option<&'static str>,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn tcgen05_recipe(operation: Tcgen05Operation) -> Tcgen05Recipe {
    const EMPTY: &[&str] = &[];
    const F32_X4: &[&str] = &["f32", "f32", "f32", "f32"];
    const F32_X32: &[&str] = &["f32"; 32];
    const BASE_CLASSES: &[&str] = &["SDPatternOperator", "Intrinsic"];
    const MMA_CLASSES: &[&str] = &[
        "SDPatternOperator",
        "Intrinsic",
        "DefaultAttrsIntrinsic",
        "DefaultAttrsIntrinsicFlags",
    ];
    const LOAD_CLASSES: &[&str] = &["SDPatternOperator", "Intrinsic", "NVVM_TCGEN05_LD"];
    const CONVERGENT_ARG_MEMORY: &[&str] = &["IntrArgMemOnly", "IntrConvergent", "NoCapture<arg0>"];
    const CONVERGENT_INACCESSIBLE_ARG_MEMORY: &[&str] = &[
        "IntrConvergent",
        "IntrInaccessibleMemOrArgMemOnly",
        "NoCapture<arg0>",
    ];
    const CONVERGENT_INACCESSIBLE_MEMORY: &[&str] = &["IntrConvergent", "IntrInaccessibleMemOnly"];
    const ALLOC_PROPERTIES: &[&str] = &[
        "IntrConvergent",
        "IntrInaccessibleMemOrArgMemOnly",
        "NoCapture<arg0>",
        "WriteOnly<arg0>",
    ];
    const FENCE_PROPERTIES: &[&str] = &["IntrHasSideEffects", "IntrNoMem"];
    const MMA_WS_PROPERTIES: &[&str] = &[
        "ImmArg<arg5>",
        "ImmArg<arg6>",
        "ImmArg<arg7>",
        "IntrArgMemOnly",
        "Range<arg5,0,4>",
        "Range<arg6,0,4>",
        "Range<arg7,0,4>",
        "ReadOnly<arg1>",
        "WriteOnly<arg0>",
    ];
    const MMA_CG1_PROPERTIES: &[&str] = &[
        "ImmArg<arg6>",
        "ImmArg<arg7>",
        "IntrArgMemOnly",
        "Range<arg6,0,4>",
        "Range<arg7,0,4>",
        "WriteOnly<arg0>",
    ];
    const LOAD_PROPERTIES: &[&str] = &[
        "ImmArg<arg1>",
        "IntrArgMemOnly",
        "IntrConvergent",
        "NoCapture<arg0>",
    ];

    let (
        abi_id,
        id,
        operation_key,
        source_record,
        llvm_symbol,
        dialect_op_type,
        dialect_op_name,
        adapter,
        source_contract,
    ) = match operation {
        Tcgen05Operation::Alloc => (
            "i0343",
            "tcgen05_alloc",
            "tcgen05.alloc.cg1",
            "int_nvvm_tcgen05_alloc_shared_cg1",
            "llvm.nvvm.tcgen05.alloc.shared.cg1",
            "Tcgen05AllocOp",
            "nvvm.tcgen05_alloc",
            Tcgen05Adapter::SharedPointerColumnsToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::Dealloc => (
            "i0344",
            "tcgen05_dealloc",
            "tcgen05.dealloc.cg1",
            "int_nvvm_tcgen05_dealloc_cg1",
            "llvm.nvvm.tcgen05.dealloc.cg1",
            "Tcgen05DeallocOp",
            "nvvm.tcgen05_dealloc",
            Tcgen05Adapter::TmemAddressColumnsToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::RelinquishAllocPermit => (
            "i0345",
            "tcgen05_relinquish_alloc_permit",
            "tcgen05.relinquish_alloc_permit.cg1",
            "int_nvvm_tcgen05_relinq_alloc_permit_cg1",
            "llvm.nvvm.tcgen05.relinq.alloc.permit.cg1",
            "Tcgen05RelinquishAllocPermitOp",
            "nvvm.tcgen05_relinquish_alloc_permit",
            Tcgen05Adapter::NoOperands,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::FenceBeforeThreadSync => (
            "i0346",
            "tcgen05_fence_before_thread_sync",
            "tcgen05.fence.before_thread_sync",
            "int_nvvm_tcgen05_fence_before_thread_sync",
            "llvm.nvvm.tcgen05.fence.before.thread.sync",
            "Tcgen05FenceBeforeThreadSyncOp",
            "nvvm.tcgen05_fence_before_thread_sync",
            Tcgen05Adapter::NoOperands,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::FenceAfterThreadSync => (
            "i0347",
            "tcgen05_fence_after_thread_sync",
            "tcgen05.fence.after.thread.sync",
            "int_nvvm_tcgen05_fence_after_thread_sync",
            "llvm.nvvm.tcgen05.fence.after.thread.sync",
            "Tcgen05FenceAfterThreadSyncOp",
            "nvvm.tcgen05_fence_after_thread_sync",
            Tcgen05Adapter::NoOperands,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::Commit => (
            "i0348",
            "tcgen05_commit",
            "tcgen05.commit.cg1",
            "int_nvvm_tcgen05_commit_cg1",
            "llvm.nvvm.tcgen05.commit.cg1",
            "Tcgen05CommitOp",
            "nvvm.tcgen05_commit",
            Tcgen05Adapter::BarrierPointerToVoid,
            Tcgen05SourceContract::TablegenSelectionChangesPtx,
        ),
        Tcgen05Operation::CommitSharedCluster => (
            "i0349",
            "tcgen05_commit_shared_cluster",
            "tcgen05.commit.shared_cluster.cg1",
            "int_nvvm_tcgen05_commit_shared_cg1",
            "llvm.nvvm.tcgen05.commit.shared.cg1",
            "Tcgen05CommitSharedClusterOp",
            "nvvm.tcgen05_commit_shared_cluster",
            Tcgen05Adapter::BarrierPointerToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::MmaWsF16 => (
            "i0350",
            "tcgen05_mma_ws_f16",
            "tcgen05.mma.ws.f16.cg1",
            "int_nvvm_tcgen05_mma_ws_tensor",
            "llvm.nvvm.tcgen05.mma.ws.tensor",
            "Tcgen05MmaWsF16Op",
            "nvvm.tcgen05_mma_ws_f16",
            Tcgen05Adapter::MmaWsDropLegacyADescriptor,
            Tcgen05SourceContract::TablegenSelectionChangesPtx,
        ),
        Tcgen05Operation::MmaF16 => (
            "i0351",
            "tcgen05_mma_f16",
            "tcgen05.mma.f16.cg1",
            "int_nvvm_tcgen05_mma_shared_disable_output_lane_cg1",
            "llvm.nvvm.tcgen05.mma.shared.disable_output_lane.cg1",
            "Tcgen05MmaF16Op",
            "nvvm.tcgen05_mma_f16",
            Tcgen05Adapter::MmaInjectZeroDisableLanes,
            Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection,
        ),
        Tcgen05Operation::MmaWsBf16 => (
            "i0352",
            "tcgen05_mma_ws_bf16",
            "tcgen05.mma.ws.bf16.cg1",
            "int_nvvm_tcgen05_mma_ws_tensor",
            "llvm.nvvm.tcgen05.mma.ws.tensor",
            "Tcgen05MmaWsBf16Op",
            "nvvm.tcgen05_mma_ws_bf16",
            Tcgen05Adapter::MmaWsDropLegacyADescriptor,
            Tcgen05SourceContract::TablegenSelectionChangesPtx,
        ),
        Tcgen05Operation::MmaWsTf32 => (
            "i0353",
            "tcgen05_mma_ws_tf32",
            "tcgen05.mma.ws.tf32.cg1",
            "int_nvvm_tcgen05_mma_ws_tensor",
            "llvm.nvvm.tcgen05.mma.ws.tensor",
            "Tcgen05MmaWsTf32Op",
            "nvvm.tcgen05_mma_ws_tf32",
            Tcgen05Adapter::MmaWsDropLegacyADescriptor,
            Tcgen05SourceContract::TablegenSelectionChangesPtx,
        ),
        Tcgen05Operation::CpSmemToTmem => (
            "i0354",
            "tcgen05_cp_smem_to_tmem",
            "tcgen05.cp.128x256b.cg1",
            "int_nvvm_tcgen05_cp_128x256b_cg1",
            "llvm.nvvm.tcgen05.cp.128x256b.cg1",
            "Tcgen05CpSmemToTmemOp",
            "nvvm.tcgen05_cp_smem_to_tmem",
            Tcgen05Adapter::TmemDescriptorToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::Ld16x256bX8Pure => (
            "i0355",
            "tcgen05_ld_16x256b_x8_pure",
            "tcgen05.ld.16x256b.x8",
            "int_nvvm_tcgen05_ld_16x256b_x8",
            "llvm.nvvm.tcgen05.ld.16x256b.x8",
            "Tcgen05Ld16x256bX8PureOp",
            "nvvm.tcgen05_ld_16x256b_x8_pure",
            Tcgen05Adapter::TmemToF32x32,
            Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection,
        ),
        Tcgen05Operation::Ld16x256bPure => (
            "i0356",
            "tcgen05_ld_16x256b_pure",
            "tcgen05.ld.16x256b.x1",
            "int_nvvm_tcgen05_ld_16x256b_x1",
            "llvm.nvvm.tcgen05.ld.16x256b.x1",
            "Tcgen05Ld16x256bPureOp",
            "nvvm.tcgen05_ld_16x256b_pure",
            Tcgen05Adapter::TmemToF32x4,
            Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection,
        ),
        Tcgen05Operation::LoadWait => (
            "i0357",
            "tcgen05_load_wait",
            "tcgen05.wait.ld",
            "int_nvvm_tcgen05_wait_ld",
            "llvm.nvvm.tcgen05.wait.ld",
            "Tcgen05LoadWaitOp",
            "nvvm.tcgen05_load_wait",
            Tcgen05Adapter::NoOperands,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::StoreWait => (
            "i0358",
            "tcgen05_store_wait",
            "tcgen05.wait.st",
            "int_nvvm_tcgen05_wait_st",
            "llvm.nvvm.tcgen05.wait.st",
            "Tcgen05StoreWaitOp",
            "nvvm.tcgen05_store_wait",
            Tcgen05Adapter::NoOperands,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::AllocCg2 => (
            "i0359",
            "tcgen05_alloc_cg2",
            "tcgen05.alloc.cg2",
            "int_nvvm_tcgen05_alloc_shared_cg2",
            "llvm.nvvm.tcgen05.alloc.shared.cg2",
            "Tcgen05AllocCg2Op",
            "nvvm.tcgen05_alloc_cg2",
            Tcgen05Adapter::SharedPointerColumnsToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::DeallocCg2 => (
            "i0360",
            "tcgen05_dealloc_cg2",
            "tcgen05.dealloc.cg2",
            "int_nvvm_tcgen05_dealloc_cg2",
            "llvm.nvvm.tcgen05.dealloc.cg2",
            "Tcgen05DeallocCg2Op",
            "nvvm.tcgen05_dealloc_cg2",
            Tcgen05Adapter::TmemAddressColumnsToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::RelinquishAllocPermitCg2 => (
            "i0361",
            "tcgen05_relinquish_alloc_permit_cg2",
            "tcgen05.relinquish_alloc_permit.cg2",
            "int_nvvm_tcgen05_relinq_alloc_permit_cg2",
            "llvm.nvvm.tcgen05.relinq.alloc.permit.cg2",
            "Tcgen05RelinquishAllocPermitCg2Op",
            "nvvm.tcgen05_relinquish_alloc_permit_cg2",
            Tcgen05Adapter::NoOperands,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::MmaF16Cg2 => (
            "i0362",
            "tcgen05_mma_f16_cg2",
            "tcgen05.mma.f16.cg2",
            "int_nvvm_tcgen05_mma_shared_disable_output_lane_cg2",
            "llvm.nvvm.tcgen05.mma.shared.disable_output_lane.cg2",
            "Tcgen05MmaF16Cg2Op",
            "nvvm.tcgen05_mma_f16_cg2",
            Tcgen05Adapter::MmaInjectZeroDisableLanes,
            Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection,
        ),
        Tcgen05Operation::CommitCg2 => (
            "i0363",
            "tcgen05_commit_cg2",
            "tcgen05.commit.cg2",
            "int_nvvm_tcgen05_commit_cg2",
            "llvm.nvvm.tcgen05.commit.cg2",
            "Tcgen05CommitCg2Op",
            "nvvm.tcgen05_commit_cg2",
            Tcgen05Adapter::BarrierPointerToVoid,
            Tcgen05SourceContract::TablegenSelectionChangesPtx,
        ),
        Tcgen05Operation::CommitSharedClusterCg2 => (
            "i0364",
            "tcgen05_commit_shared_cluster_cg2",
            "tcgen05.commit.shared_cluster.cg2",
            "int_nvvm_tcgen05_commit_shared_cg2",
            "llvm.nvvm.tcgen05.commit.shared.cg2",
            "Tcgen05CommitSharedClusterCg2Op",
            "nvvm.tcgen05_commit_shared_cluster_cg2",
            Tcgen05Adapter::BarrierPointerToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::CommitMulticastCg2 => (
            "i0365",
            "tcgen05_commit_multicast_cg2",
            "tcgen05.commit.multicast.cg2",
            "int_nvvm_tcgen05_commit_mc_shared_cg2",
            "llvm.nvvm.tcgen05.commit.mc.shared.cg2",
            "Tcgen05CommitMulticastCg2Op",
            "nvvm.tcgen05_commit_multicast_cg2",
            Tcgen05Adapter::BarrierPointerMaskToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::CpSmemToTmemCg2 => (
            "i0366",
            "tcgen05_cp_smem_to_tmem_cg2",
            "tcgen05.cp.128x256b.cg2",
            "int_nvvm_tcgen05_cp_128x256b_cg2",
            "llvm.nvvm.tcgen05.cp.128x256b.cg2",
            "Tcgen05CpSmemToTmemCg2Op",
            "nvvm.tcgen05_cp_smem_to_tmem_cg2",
            Tcgen05Adapter::TmemDescriptorToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::CommitMulticast => (
            "i0760",
            "tcgen05_commit_multicast",
            "tcgen05.commit.multicast.cg1",
            "int_nvvm_tcgen05_commit_mc_shared_cg1",
            "llvm.nvvm.tcgen05.commit.mc.shared.cg1",
            "Tcgen05CommitMulticastOp",
            "nvvm.tcgen05_commit_multicast",
            Tcgen05Adapter::BarrierPointerMaskToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::ShiftDown => (
            "i0761",
            "tcgen05_shift_down",
            "tcgen05.shift.down.cg1",
            "int_nvvm_tcgen05_shift_down_cg1",
            "llvm.nvvm.tcgen05.shift.down.cg1",
            "Tcgen05ShiftDownOp",
            "nvvm.tcgen05_shift_down",
            Tcgen05Adapter::TmemAddressToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::ShiftDownCg2 => (
            "i0762",
            "tcgen05_shift_down_cg2",
            "tcgen05.shift.down.cg2",
            "int_nvvm_tcgen05_shift_down_cg2",
            "llvm.nvvm.tcgen05.shift.down.cg2",
            "Tcgen05ShiftDownCg2Op",
            "nvvm.tcgen05_shift_down_cg2",
            Tcgen05Adapter::TmemAddressToVoid,
            Tcgen05SourceContract::ExactTablegenSelection,
        ),
        Tcgen05Operation::Ld | Tcgen05Operation::St | Tcgen05Operation::Mma => {
            unreachable!("tcgen05 load/store variants use their compact recipes")
        }
    };

    let cg2 = matches!(
        operation,
        Tcgen05Operation::AllocCg2
            | Tcgen05Operation::DeallocCg2
            | Tcgen05Operation::RelinquishAllocPermitCg2
            | Tcgen05Operation::MmaF16Cg2
            | Tcgen05Operation::CommitCg2
            | Tcgen05Operation::CommitSharedClusterCg2
            | Tcgen05Operation::CommitMulticastCg2
            | Tcgen05Operation::CpSmemToTmemCg2
            | Tcgen05Operation::ShiftDownCg2
    );
    let group = if cg2 { "cta_group::2" } else { "cta_group::1" };
    let (
        rust_arguments,
        rust_result,
        dialect_operands,
        dialect_results,
        llvm_arguments,
        llvm_results,
        imported_classes,
        imported_properties,
        safe,
        safe_reason,
        memory,
        modifiers,
        operands,
        selection_record,
        selection_asm,
        summary,
    ) = match operation {
        Tcgen05Operation::Alloc | Tcgen05Operation::AllocCg2 => (
            &["*mut u32", "u32"] as &[_],
            "()",
            &["ptr", "i32"] as &[_],
            EMPTY,
            &["shared_ptr", "i32"] as &[_],
            EMPTY,
            BASE_CLASSES,
            ALLOC_PROPERTIES,
            false,
            None,
            "write",
            vec![
                "alloc".into(),
                group.into(),
                "sync".into(),
                "aligned".into(),
                "shared::cta".into(),
                "b32".into(),
            ],
            vec![OperandPattern::Address, OperandPattern::Register],
            Some(if cg2 {
                "TCGEN05_ALLOC_S64_CG2"
            } else {
                "TCGEN05_ALLOC_S64_CG1"
            }),
            Some(if cg2 {
                "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 \t[$dst], $ncols;"
            } else {
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 \t[$dst], $ncols;"
            }),
            "Allocates tensor-memory columns and writes their address to shared memory.",
        ),
        Tcgen05Operation::Dealloc | Tcgen05Operation::DeallocCg2 => (
            &["u32", "u32"] as &[_],
            "()",
            &["i32", "i32"] as &[_],
            EMPTY,
            &["tmem_ptr", "i32"] as &[_],
            EMPTY,
            BASE_CLASSES,
            CONVERGENT_ARG_MEMORY,
            false,
            None,
            "read_write",
            vec![
                "dealloc".into(),
                group.into(),
                "sync".into(),
                "aligned".into(),
                "b32".into(),
            ],
            vec![OperandPattern::Register, OperandPattern::Register],
            Some(if cg2 {
                "TCGEN05_DEALLOC_CG2"
            } else {
                "TCGEN05_DEALLOC_CG1"
            }),
            Some(if cg2 {
                "tcgen05.dealloc.cta_group::2.sync.aligned.b32 \t$tmem_addr, $ncols;"
            } else {
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 \t$tmem_addr, $ncols;"
            }),
            "Releases tensor-memory columns allocated to the CTA group.",
        ),
        Tcgen05Operation::RelinquishAllocPermit | Tcgen05Operation::RelinquishAllocPermitCg2 => (
            EMPTY,
            "()",
            EMPTY,
            EMPTY,
            EMPTY,
            EMPTY,
            BASE_CLASSES,
            CONVERGENT_INACCESSIBLE_MEMORY,
            true,
            Some(
                "The operation has no Rust memory operand and only releases this CTA group's allocation permit.",
            ),
            "read_write",
            vec![
                "relinquish_alloc_permit".into(),
                group.into(),
                "sync".into(),
                "aligned".into(),
            ],
            vec![],
            Some(if cg2 {
                "TCGEN05_RELINQ_CG2"
            } else {
                "TCGEN05_RELINQ_CG1"
            }),
            Some(if cg2 {
                "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"
            } else {
                "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
            }),
            "Releases the CTA group's tensor-memory allocation permit.",
        ),
        Tcgen05Operation::FenceBeforeThreadSync | Tcgen05Operation::FenceAfterThreadSync => {
            let before = operation == Tcgen05Operation::FenceBeforeThreadSync;
            (
                EMPTY,
                "()",
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                BASE_CLASSES,
                FENCE_PROPERTIES,
                true,
                Some("The operation only orders the calling thread's tcgen05 accesses."),
                "none",
                vec![if before {
                    "fence::before_thread_sync".into()
                } else {
                    "fence::after_thread_sync".into()
                }],
                vec![],
                Some(if before {
                    "tcgen05_fence_before_thread_sync"
                } else {
                    "tcgen05_fence_after_thread_sync"
                }),
                Some(if before {
                    "tcgen05.fence::before_thread_sync;"
                } else {
                    "tcgen05.fence::after_thread_sync;"
                }),
                if before {
                    "Orders prior tcgen05 accesses before thread synchronization."
                } else {
                    "Orders later tcgen05 accesses after thread synchronization."
                },
            )
        }
        Tcgen05Operation::Commit
        | Tcgen05Operation::CommitCg2
        | Tcgen05Operation::CommitSharedCluster
        | Tcgen05Operation::CommitSharedClusterCg2 => {
            let shared = matches!(
                operation,
                Tcgen05Operation::CommitSharedCluster | Tcgen05Operation::CommitSharedClusterCg2
            );
            let llvm_args = if shared {
                &["shared_ptr"] as &[_]
            } else {
                &["ptr"] as &[_]
            };
            let record = match (cg2, shared) {
                (false, false) => "TCGEN05_COMMIT_CG1",
                (true, false) => "TCGEN05_COMMIT_CG2",
                (false, true) => "TCGEN05_COMMIT_S64_CG1",
                (true, true) => "TCGEN05_COMMIT_S64_CG2",
            };
            let asm = match cg2 {
                false => {
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 \t[$mbar];"
                }
                true => {
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 \t[$mbar];"
                }
            };
            let mut modifiers = vec![
                "commit".into(),
                group.into(),
                "mbarrier::arrive::one".into(),
            ];
            if shared {
                modifiers.push("shared::cluster".into());
            }
            modifiers.push("b64".into());
            (
                &["*mut u64"] as &[_],
                "()",
                &["ptr"] as &[_],
                EMPTY,
                llvm_args,
                EMPTY,
                BASE_CLASSES,
                CONVERGENT_INACCESSIBLE_ARG_MEMORY,
                false,
                None,
                "read_write",
                modifiers,
                vec![OperandPattern::Address],
                Some(record),
                Some(asm),
                if shared {
                    "Commits tcgen05 completion to a cluster-shared mbarrier."
                } else {
                    "Commits tcgen05 completion to an mbarrier."
                },
            )
        }
        Tcgen05Operation::MmaWsF16 | Tcgen05Operation::MmaWsBf16 | Tcgen05Operation::MmaWsTf32 => {
            let kind = if operation == Tcgen05Operation::MmaWsTf32 {
                "kind::tf32"
            } else {
                "kind::f16"
            };
            (
                &["u32", "u32", "u64", "u64", "u32", "bool"] as &[_],
                "()",
                &["i32", "i32", "i64", "i64", "i32", "i1"] as &[_],
                EMPTY,
                &[
                    "tmem_ptr", "tmem_ptr", "i64", "i32", "i1", "i32", "i32", "i32",
                ] as &[_],
                EMPTY,
                MMA_CLASSES,
                MMA_WS_PROPERTIES,
                false,
                None,
                "read_write",
                vec![
                    "mma".into(),
                    "ws".into(),
                    "cta_group::1".into(),
                    kind.into(),
                ],
                vec![
                    OperandPattern::Address,
                    OperandPattern::Address,
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: "%enable_pred".into(),
                    },
                ],
                None,
                None,
                match operation {
                    Tcgen05Operation::MmaWsF16 => "Issues warp-specialized f16 tensor-memory MMA.",
                    Tcgen05Operation::MmaWsBf16 => {
                        "Issues warp-specialized bf16 tensor-memory MMA using the f16 instruction class."
                    }
                    _ => "Issues warp-specialized tf32 tensor-memory MMA.",
                },
            )
        }
        Tcgen05Operation::MmaF16 | Tcgen05Operation::MmaF16Cg2 => (
            &["u32", "u64", "u64", "u32", "bool"] as &[_],
            "()",
            &["i32", "i64", "i64", "i32", "i1"] as &[_],
            EMPTY,
            if cg2 {
                &["tmem_ptr", "i64", "i64", "i32", "i1", "v8i32", "i32", "i32"] as &[_]
            } else {
                &["tmem_ptr", "i64", "i64", "i32", "i1", "v4i32", "i32", "i32"] as &[_]
            },
            EMPTY,
            MMA_CLASSES,
            MMA_CG1_PROPERTIES,
            false,
            None,
            "write",
            vec!["mma".into(), group.into(), "kind::f16".into()],
            vec![
                OperandPattern::Address,
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: if cg2 {
                        "{%z, %z, %z, %z, %z, %z, %z, %z}"
                    } else {
                        "{%z, %z, %z, %z}"
                    }
                    .into(),
                },
                OperandPattern::Exact {
                    value: "%enable_pred".into(),
                },
            ],
            None,
            None,
            "Issues f16 tensor-memory MMA with zeroed disable-output-lane controls.",
        ),
        Tcgen05Operation::CpSmemToTmem | Tcgen05Operation::CpSmemToTmemCg2 => (
            &["u32", "u64"] as &[_],
            "()",
            &["i32", "i64"] as &[_],
            EMPTY,
            &["tmem_ptr", "i64"] as &[_],
            EMPTY,
            BASE_CLASSES,
            CONVERGENT_INACCESSIBLE_ARG_MEMORY,
            false,
            None,
            "read_write",
            vec!["cp".into(), group.into(), "128x256b".into()],
            vec![OperandPattern::Address, OperandPattern::Register],
            Some(if cg2 {
                "TCGEN05_CP_128x256b_cg2"
            } else {
                "TCGEN05_CP_128x256b_cg1"
            }),
            Some(if cg2 {
                "tcgen05.cp.cta_group::2.128x256b \t[$tmem_addr], $sdesc;"
            } else {
                "tcgen05.cp.cta_group::1.128x256b \t[$tmem_addr], $sdesc;"
            }),
            "Copies one 128x256-bit tile from shared memory to tensor memory.",
        ),
        Tcgen05Operation::Ld16x256bX8Pure | Tcgen05Operation::Ld16x256bPure => {
            let x8 = operation == Tcgen05Operation::Ld16x256bX8Pure;
            (
                &["u32"] as &[_],
                if x8 { "[f32; 32]" } else { "[f32; 4]" },
                &["i32"] as &[_],
                if x8 { F32_X32 } else { F32_X4 },
                &["tmem_ptr", "i1"] as &[_],
                // Overloaded data type-variables in the pinned LLVM 23 dump
                // (see tcgen05_overloaded_data_token): 32 and 4 registers.
                if x8 {
                    &["anonymous_9953"] as &[_]
                } else {
                    &["anonymous_9941"] as &[_]
                },
                LOAD_CLASSES,
                LOAD_PROPERTIES,
                false,
                None,
                "read",
                vec![
                    "ld".into(),
                    "sync".into(),
                    "aligned".into(),
                    "16x256b".into(),
                    if x8 { "x8".into() } else { "x1".into() },
                    "b32".into(),
                ],
                vec![
                    OperandPattern::RegisterList {
                        length: if x8 { 32 } else { 4 },
                    },
                    OperandPattern::Address,
                ],
                None,
                None,
                if x8 {
                    "Loads 32 f32 register values from tensor memory."
                } else {
                    "Loads four f32 register values from tensor memory."
                },
            )
        }
        Tcgen05Operation::LoadWait | Tcgen05Operation::StoreWait => {
            let load = operation == Tcgen05Operation::LoadWait;
            (
                EMPTY,
                "()",
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                BASE_CLASSES,
                CONVERGENT_INACCESSIBLE_MEMORY,
                true,
                Some("The operation only waits for the calling thread's prior tcgen05 access."),
                "read_write",
                vec![
                    if load {
                        "wait::ld".into()
                    } else {
                        "wait::st".into()
                    },
                    "sync".into(),
                    "aligned".into(),
                ],
                vec![],
                Some(if load {
                    "tcgen05_wait_ld"
                } else {
                    "tcgen05_wait_st"
                }),
                Some(if load {
                    "tcgen05.wait::ld.sync.aligned;"
                } else {
                    "tcgen05.wait::st.sync.aligned;"
                }),
                if load {
                    "Waits for prior tensor-memory loads to complete."
                } else {
                    "Waits for prior tensor-memory stores to complete."
                },
            )
        }
        Tcgen05Operation::CommitMulticast | Tcgen05Operation::CommitMulticastCg2 => (
            &["*mut u64", "u16"] as &[_],
            "()",
            &["ptr", "i16"] as &[_],
            EMPTY,
            &["shared_ptr", "i16"] as &[_],
            EMPTY,
            BASE_CLASSES,
            CONVERGENT_INACCESSIBLE_ARG_MEMORY,
            false,
            None,
            "read_write",
            vec![
                "commit".into(),
                group.into(),
                "mbarrier::arrive::one".into(),
                "shared::cluster".into(),
                "multicast::cluster".into(),
                "b64".into(),
            ],
            vec![OperandPattern::Address, OperandPattern::Register],
            Some(if cg2 {
                "TCGEN05_COMMIT_S64_CG2_MC"
            } else {
                "TCGEN05_COMMIT_S64_CG1_MC"
            }),
            Some(if cg2 {
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 \t[$mbar], $mc;"
            } else {
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 \t[$mbar], $mc;"
            }),
            "Commits tcgen05 completion to the selected cluster mbarriers.",
        ),
        Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2 => (
            &["u32"] as &[_],
            "()",
            &["i32"] as &[_],
            EMPTY,
            &["tmem_ptr"] as &[_],
            EMPTY,
            BASE_CLASSES,
            CONVERGENT_ARG_MEMORY,
            false,
            None,
            "read_write",
            vec!["shift".into(), group.into(), "down".into()],
            vec![OperandPattern::Address],
            Some(if cg2 {
                "TCGEN05_SHIFT_CG2"
            } else {
                "TCGEN05_SHIFT_CG1"
            }),
            Some(if cg2 {
                "tcgen05.shift.cta_group::2.down \t[$tmem_addr];"
            } else {
                "tcgen05.shift.cta_group::1.down \t[$tmem_addr];"
            }),
            "Shifts tensor-memory rows down by one row.",
        ),
        Tcgen05Operation::Ld | Tcgen05Operation::St | Tcgen05Operation::Mma => {
            unreachable!("tcgen05 load/store variants use their compact recipes")
        }
    };

    Tcgen05Recipe {
        operation,
        abi_id,
        id,
        operation_key,
        source_record,
        llvm_symbol,
        rust_arguments,
        rust_result,
        dialect_op_type,
        dialect_op_name,
        dialect_operands,
        dialect_results,
        llvm_arguments,
        llvm_results,
        imported_classes,
        imported_properties,
        adapter,
        source_contract,
        safe,
        safe_reason,
        memory,
        modifiers,
        operands,
        selection_record,
        selection_asm,
        summary,
    }
}
