/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, IntrinsicSource,
    MbarrierExtended, MbarrierExtendedAdapter, MbarrierExtendedAdmission,
    MbarrierExtendedOperation, MbarrierExtendedSourceContract, OverlayBackendLowering,
    OverlayIntrinsic, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;

#[derive(Clone)]
pub(in crate::resolve) struct MbarrierExtendedRecipe {
    pub(in crate::resolve) operation: MbarrierExtendedOperation,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: Option<&'static str>,
    pub(in crate::resolve) llvm_symbol: Option<&'static str>,
    pub(in crate::resolve) ptx_native_instruction: Option<&'static str>,
    pub(in crate::resolve) rust_arguments: &'static [&'static str],
    pub(in crate::resolve) rust_result: &'static str,
    pub(in crate::resolve) must_use: bool,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) dialect_operands: &'static [&'static str],
    pub(in crate::resolve) dialect_results: &'static [&'static str],
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_results: &'static [&'static str],
    pub(in crate::resolve) llvm_properties: &'static [&'static str],
    pub(in crate::resolve) adapter: MbarrierExtendedAdapter,
    pub(in crate::resolve) source_contract: MbarrierExtendedSourceContract,
    pub(in crate::resolve) execution_scope: &'static str,
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) minimum_sm: &'static str,
    pub(in crate::resolve) ptx_result: &'static str,
    pub(in crate::resolve) expected_ptx: InstructionPattern,
    pub(in crate::resolve) inline_ptx: &'static str,
    pub(in crate::resolve) inline_constraints: &'static str,
    pub(in crate::resolve) ptx_isa_section: &'static str,
    pub(in crate::resolve) ptx_isa_url: &'static str,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn mbarrier_extended_recipe(
    operation: MbarrierExtendedOperation,
) -> MbarrierExtendedRecipe {
    let instruction = |modifiers: &[&str], operands| InstructionPattern {
        mnemonic: if modifiers.first() == Some(&"nanosleep") {
            "nanosleep".into()
        } else if modifiers.first() == Some(&"fence") {
            "fence".into()
        } else {
            "mbarrier".into()
        },
        modifiers: modifiers[1..].iter().map(|value| (*value).into()).collect(),
        operands,
    };
    match operation {
        MbarrierExtendedOperation::ArriveExpectTxCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0306",
            id: "mbarrier_arrive_expect_tx",
            operation_key: "barrier.mbarrier.arrive.expect_tx.shared.cta.release.cta",
            source_record: Some("int_nvvm_mbarrier_arrive_expect_tx_scope_cta_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32", "u32"],
            rust_result: "u64",
            must_use: true,
            dialect_op_type: "MbarrierArriveExpectTxSharedOp",
            dialect_op_name: "nvvm.mbarrier_arrive_expect_tx_shared",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i64"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i64"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "u64",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "arrive",
                    "expect_tx",
                    "release",
                    "cta",
                    "shared::cta",
                    "b64",
                ],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;",
            inline_constraints: "=l,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            summary: "Arrives at a CTA-shared barrier and adds expected transaction bytes.",
        },
        MbarrierExtendedOperation::ArriveExpectTxCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0307",
            id: "mbarrier_arrive_expect_tx_cluster",
            operation_key: "barrier.mbarrier.arrive.expect_tx.shared.cta.relaxed.cluster",
            source_record: Some(
                "int_nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cluster_space_cta",
            ),
            llvm_symbol: Some(
                "llvm.nvvm.mbarrier.arrive.expect.tx.relaxed.scope.cluster.space.cta",
            ),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32", "u32"],
            rust_result: "u64",
            must_use: true,
            dialect_op_type: "MbarrierArriveExpectTxClusterOp",
            dialect_op_name: "nvvm.mbarrier_arrive_expect_tx_cluster",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i64"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i64"],
            llvm_properties: &["IntrArgMemOnly", "IntrConvergent", "IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cluster",
            minimum_ptx: "8.6",
            minimum_sm: "sm_90",
            ptx_result: "u64",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "arrive",
                    "expect_tx",
                    "relaxed",
                    "cluster",
                    "shared::cta",
                    "b64",
                ],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 $0, [$1], $2;",
            inline_constraints: "=l,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            summary: "Arrives at a CTA-shared barrier with cluster-scope transaction tracking.",
        },
        MbarrierExtendedOperation::ArriveRemoteCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0308",
            id: "mbarrier_arrive_cluster",
            operation_key: "barrier.mbarrier.arrive.shared.cluster.release.cluster.raw_address",
            source_record: None,
            llvm_symbol: None,
            ptx_native_instruction: Some("mbarrier.arrive.release.cluster.shared::cluster.b64"),
            rust_arguments: &["u64"],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "MbarrierArriveClusterOp",
            dialect_op_name: "nvvm.mbarrier_arrive_cluster",
            dialect_operands: &["i64"],
            dialect_results: &[],
            llvm_arguments: &[],
            llvm_results: &[],
            llvm_properties: &[],
            adapter: MbarrierExtendedAdapter::RawClusterAddressToVoid,
            source_contract: MbarrierExtendedSourceContract::PtxNativeRawClusterAddress,
            execution_scope: "cluster",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "()",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "arrive",
                    "release",
                    "cluster",
                    "shared::cluster",
                    "b64",
                ],
                vec![
                    OperandPattern::Exact { value: "_".into() },
                    OperandPattern::Address,
                ],
            ),
            inline_ptx: "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
            inline_constraints: "l,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            summary: "Arrives at a remote cluster-shared barrier through its raw address.",
        },
        MbarrierExtendedOperation::TryWaitTokenCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0309",
            id: "mbarrier_try_wait",
            operation_key: "barrier.mbarrier.try_wait.shared.cta.token",
            source_record: Some("int_nvvm_mbarrier_try_wait_scope_cta_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u64"],
            rust_result: "bool",
            must_use: true,
            dialect_op_type: "MbarrierTryWaitSharedOp",
            dialect_op_name: "nvvm.mbarrier_try_wait_shared",
            dialect_operands: &["ptr", "i64"],
            dialect_results: &["i1"],
            llvm_arguments: &["shared_ptr", "i64"],
            llvm_results: &["i1"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback", "NoCapture<arg0>"],
            adapter: MbarrierExtendedAdapter::PointerTokenToPredicate,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_result: "bool",
            expected_ptx: instruction(
                &["mbarrier", "try_wait", "shared", "b64"],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "{ .reg .pred %p0; mbarrier.try_wait.shared.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
            inline_constraints: "=r,l,l,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            summary: "Tests a CTA-shared barrier token with a scheduling hint.",
        },
        MbarrierExtendedOperation::TryWaitParityCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0310",
            id: "mbarrier_try_wait_parity",
            operation_key: "barrier.mbarrier.try_wait.parity.shared.cta",
            source_record: Some("int_nvvm_mbarrier_try_wait_parity_scope_cta_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.try.wait.parity.scope.cta.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32"],
            rust_result: "bool",
            must_use: true,
            dialect_op_type: "MbarrierTryWaitParitySharedOp",
            dialect_op_name: "nvvm.mbarrier_try_wait_parity_shared",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i1"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i1"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback", "NoCapture<arg0>"],
            adapter: MbarrierExtendedAdapter::PointerParityToPredicate,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_result: "bool",
            expected_ptx: instruction(
                &["mbarrier", "try_wait", "parity", "shared::cta", "b64"],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "{ .reg .pred %p0; mbarrier.try_wait.parity.shared::cta.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
            inline_constraints: "=r,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            summary: "Tests a CTA-shared barrier phase by parity.",
        },
        MbarrierExtendedOperation::TryWaitParityCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0311",
            id: "mbarrier_try_wait_parity_cluster",
            operation_key: "barrier.mbarrier.try_wait.parity.shared.cta.acquire.cluster",
            source_record: Some("int_nvvm_mbarrier_try_wait_parity_scope_cluster_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.try.wait.parity.scope.cluster.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32"],
            rust_result: "bool",
            must_use: true,
            dialect_op_type: "MbarrierTryWaitParityClusterOp",
            dialect_op_name: "nvvm.mbarrier_try_wait_parity_cluster",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i1"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i1"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback", "NoCapture<arg0>"],
            adapter: MbarrierExtendedAdapter::PointerParityToPredicate,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cluster",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "bool",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "try_wait",
                    "parity",
                    "acquire",
                    "cluster",
                    "shared::cta",
                    "b64",
                ],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "{ .reg .pred %p0; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
            inline_constraints: "=r,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            summary: "Tests barrier parity with cluster-scope acquire ordering.",
        },
        MbarrierExtendedOperation::FenceProxyAsyncSharedCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0312",
            id: "fence_proxy_async_shared_cta",
            operation_key: "fence.proxy.async.shared.cta",
            source_record: Some("int_nvvm_fence_proxy_async_shared_cta"),
            llvm_symbol: Some("llvm.nvvm.fence.proxy.async.shared_cta"),
            ptx_native_instruction: None,
            rust_arguments: &[],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "FenceProxyAsyncSharedCtaOp",
            dialect_op_name: "nvvm.fence_proxy_async_shared_cta",
            dialect_operands: &[],
            dialect_results: &[],
            llvm_arguments: &[],
            llvm_results: &[],
            llvm_properties: &["IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "()",
            expected_ptx: instruction(&["fence", "proxy", "async", "shared::cta"], vec![]),
            inline_ptx: "fence.proxy.async.shared::cta;",
            inline_constraints: "~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.proxy",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy",
            summary: "Makes CTA-shared generic-proxy writes visible to the async proxy.",
        },
        MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0313",
            id: "fence_mbarrier_init_release_cluster",
            operation_key: "fence.mbarrier_init.release.cluster",
            source_record: Some("int_nvvm_fence_mbarrier_init_release_cluster"),
            llvm_symbol: Some("llvm.nvvm.fence.mbarrier_init.release.cluster"),
            ptx_native_instruction: None,
            rust_arguments: &[],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "FenceMbarrierInitReleaseClusterOp",
            dialect_op_name: "nvvm.fence_mbarrier_init_release_cluster",
            dialect_operands: &[],
            dialect_results: &[],
            llvm_arguments: &[],
            llvm_results: &[],
            llvm_properties: &["IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cluster",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "()",
            expected_ptx: instruction(&["fence", "mbarrier_init", "release", "cluster"], vec![]),
            inline_ptx: "fence.mbarrier_init.release.cluster;",
            inline_constraints: "~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.mbarrier_init",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-mbarrier-init",
            summary: "Releases mbarrier initialization at cluster scope.",
        },
        MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster => {
            MbarrierExtendedRecipe {
                operation,
                abi_id: "i0314",
                id: "fence_proxy_async_generic_release_shared_cta_cluster",
                operation_key: "fence.proxy.async_generic.release.sync_restrict.shared_cta.cluster",
                source_record: Some(
                    "int_nvvm_fence_proxy_async_generic_release_sync_restrict_space_cta_scope_cluster",
                ),
                llvm_symbol: Some(
                    "llvm.nvvm.fence.proxy.async_generic.release.sync_restrict.space.cta.scope.cluster",
                ),
                ptx_native_instruction: None,
                rust_arguments: &[],
                rust_result: "()",
                must_use: false,
                dialect_op_type: "FenceProxyAsyncGenericReleaseSharedCtaClusterOp",
                dialect_op_name: "nvvm.fence_proxy_async_generic_release_shared_cta_cluster",
                dialect_operands: &[],
                dialect_results: &[],
                llvm_arguments: &[],
                llvm_results: &[],
                llvm_properties: &["IntrNoCallback"],
                adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
                source_contract: MbarrierExtendedSourceContract::LlvmImported,
                execution_scope: "cluster",
                minimum_ptx: "8.6",
                minimum_sm: "sm_90",
                ptx_result: "()",
                expected_ptx: instruction(
                    &[
                        "fence",
                        "proxy",
                        "async::generic",
                        "release",
                        "sync_restrict::shared::cta",
                        "cluster",
                    ],
                    vec![],
                ),
                inline_ptx: "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
                inline_constraints: "~{memory}",
                ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.proxy",
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy",
                summary: "Releases CTA-shared generic-proxy writes to the async proxy at cluster scope.",
            }
        }
        MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster => {
            MbarrierExtendedRecipe {
                operation,
                abi_id: "i0315",
                id: "fence_proxy_async_generic_acquire_shared_cluster_cluster",
                operation_key: "fence.proxy.async_generic.acquire.sync_restrict.shared_cluster.cluster",
                source_record: Some(
                    "int_nvvm_fence_proxy_async_generic_acquire_sync_restrict_space_cluster_scope_cluster",
                ),
                llvm_symbol: Some(
                    "llvm.nvvm.fence.proxy.async_generic.acquire.sync_restrict.space.cluster.scope.cluster",
                ),
                ptx_native_instruction: None,
                rust_arguments: &[],
                rust_result: "()",
                must_use: false,
                dialect_op_type: "FenceProxyAsyncGenericAcquireSharedClusterClusterOp",
                dialect_op_name: "nvvm.fence_proxy_async_generic_acquire_shared_cluster_cluster",
                dialect_operands: &[],
                dialect_results: &[],
                llvm_arguments: &[],
                llvm_results: &[],
                llvm_properties: &["IntrNoCallback"],
                adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
                source_contract: MbarrierExtendedSourceContract::LlvmImported,
                execution_scope: "cluster",
                minimum_ptx: "8.6",
                minimum_sm: "sm_90",
                ptx_result: "()",
                expected_ptx: instruction(
                    &[
                        "fence",
                        "proxy",
                        "async::generic",
                        "acquire",
                        "sync_restrict::shared::cluster",
                        "cluster",
                    ],
                    vec![],
                ),
                inline_ptx: "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
                inline_constraints: "~{memory}",
                ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.proxy",
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy",
                summary: "Acquires cluster-shared async-proxy writes through the generic proxy.",
            }
        }
        MbarrierExtendedOperation::Nanosleep => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0316",
            id: "nanosleep",
            operation_key: "thread.nanosleep.u32",
            source_record: Some("int_nvvm_nanosleep"),
            llvm_symbol: Some("llvm.nvvm.nanosleep"),
            ptx_native_instruction: None,
            rust_arguments: &["u32"],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "NanosleepOp",
            dialect_op_name: "nvvm.nanosleep",
            dialect_operands: &["i32"],
            dialect_results: &[],
            llvm_arguments: &["i32"],
            llvm_results: &[],
            llvm_properties: &["IntrConvergent", "IntrHasSideEffects", "IntrNoMem"],
            adapter: MbarrierExtendedAdapter::NanosecondsToVoid,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "thread",
            minimum_ptx: "6.3",
            minimum_sm: "sm_70",
            ptx_result: "()",
            expected_ptx: instruction(
                &["nanosleep", "u32"],
                vec![OperandPattern::RegisterOrImmediate],
            ),
            inline_ptx: "nanosleep.u32 $0;",
            inline_constraints: "r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: nanosleep",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-nanosleep",
            summary: "Suspends the executing thread for approximately the requested nanoseconds.",
        },
    }
}

pub(crate) fn mbarrier_extended_inline_recipe(
    operation: MbarrierExtendedOperation,
) -> (&'static str, &'static str) {
    let recipe = mbarrier_extended_recipe(operation);
    (recipe.inline_ptx, recipe.inline_constraints)
}

pub(in crate::resolve) fn mbarrier_extended_backend_floor(
    operation: MbarrierExtendedOperation,
    backend: IntrinsicBackend,
) -> (&'static str, &'static str) {
    let recipe = mbarrier_extended_recipe(operation);
    match (operation, backend) {
        (MbarrierExtendedOperation::Nanosleep, IntrinsicBackend::LibNvvm) => ("6.3", "sm_75"),
        _ => (recipe.minimum_ptx, recipe.minimum_sm),
    }
}

pub(in crate::resolve) fn expand_mbarrier_extended_admission(
    admission: &MbarrierExtendedAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "extended-mbarrier runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact extended-mbarrier admission requires both backend evidence profiles"
    );
    let expected_operations = BTreeSet::from([
        MbarrierExtendedOperation::ArriveExpectTxCta,
        MbarrierExtendedOperation::ArriveExpectTxCluster,
        MbarrierExtendedOperation::ArriveRemoteCluster,
        MbarrierExtendedOperation::TryWaitTokenCta,
        MbarrierExtendedOperation::TryWaitParityCta,
        MbarrierExtendedOperation::TryWaitParityCluster,
        MbarrierExtendedOperation::FenceProxyAsyncSharedCta,
        MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster,
        MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster,
        MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster,
        MbarrierExtendedOperation::Nanosleep,
    ]);
    let actual_operations = admission
        .variants
        .iter()
        .map(|variant| variant.operation)
        .collect::<BTreeSet<_>>();
    ensure!(
        admission.variants.len() == expected_operations.len()
            && actual_operations == expected_operations,
        "compact extended-mbarrier admission must contain each reviewed operation exactly once"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = mbarrier_extended_recipe(variant.operation);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let source =
                recipe
                    .ptx_native_instruction
                    .map(|instruction| IntrinsicSource::PtxNative {
                        instruction: instruction.into(),
                    });
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "mbarrier_extended".into(),
                source,
                source_record: recipe.source_record.map(str::to_owned),
                rust_module: "barrier".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: recipe.rust_result.into(),
                safe: false,
                must_use: recipe.must_use,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::barrier::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::barrier::{}", recipe.id)],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe
                    .dialect_operands
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                dialect_results: recipe
                    .dialect_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_symbol: recipe.llvm_symbol.map(str::to_owned),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe
                    .llvm_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_results: recipe
                    .llvm_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                pure: false,
                memory: "read_write".into(),
                convergent: true,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: Some(recipe.minimum_sm.into()),
                ptx_result: recipe.ptx_result.into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.ptx_isa_section.into(),
                ptx_isa_url: recipe.ptx_isa_url.into(),
                lowering: "generated_mbarrier_extended_inline_ptx".into(),
                backend_lowerings: [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
                    .into_iter()
                    .map(|backend| {
                        let (minimum_ptx, minimum_sm) =
                            mbarrier_extended_backend_floor(recipe.operation, backend);
                        OverlayBackendLowering {
                            backend,
                            mechanism: BackendLoweringMechanism::InlinePtx,
                            evidence_profile: match backend {
                                IntrinsicBackend::LlvmNvptx => {
                                    admission.llvm_evidence_profile.clone()
                                }
                                IntrinsicBackend::LibNvvm => {
                                    admission.libnvvm_evidence_profile.clone()
                                }
                            },
                            targets: None,
                            minimum_ptx: Some(minimum_ptx.into()),
                            minimum_sm: Some(minimum_sm.into()),
                        }
                    })
                    .collect(),
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                integer_minmax: None,
                packed_conversion: None,
                scalar_conversion: None,
                scalar_arithmetic: None,
                scalar_math: None,
                extended_minmax: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: Some(MbarrierExtended {
                    operation: recipe.operation,
                    adapter: recipe.adapter,
                    source_contract: recipe.source_contract,
                    runtime_validation: admission.runtime_validation,
                }),
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                wgmma_control: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                tma: None,
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: recipe.expected_ptx,
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

pub(in crate::resolve) fn validate_mbarrier_extended_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let contract = policy
        .mbarrier_extended
        .as_ref()
        .with_context(|| format!("{} has no closed extended-mbarrier contract", policy.id))?;
    let recipe = mbarrier_extended_recipe(contract.operation);
    ensure!(
        contract.adapter == recipe.adapter
            && contract.source_contract == recipe.source_contract
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key,
        "{} identity or adapter does not match its closed extended-mbarrier recipe",
        policy.id
    );
    match recipe.source_contract {
        MbarrierExtendedSourceContract::LlvmImported => ensure!(
            matches!(
                source,
                IntrinsicSource::LlvmImported { source_record }
                    if Some(source_record.as_str()) == recipe.source_record
            ) && policy.source.is_none()
                && policy.source_record.as_deref() == recipe.source_record
                && policy.llvm_symbol.as_deref() == recipe.llvm_symbol
                && declaration.is_some_and(|record| {
                    record.source_record == recipe.source_record.unwrap()
                        && record.properties == recipe.llvm_properties
                }),
            "{} LLVM source contract changed",
            policy.id
        ),
        MbarrierExtendedSourceContract::PtxNativeRawClusterAddress => ensure!(
            matches!(
                source,
                IntrinsicSource::PtxNative { instruction }
                    if Some(instruction.as_str()) == recipe.ptx_native_instruction
            ) && policy.source_record.is_none()
                && policy.llvm_symbol.is_none()
                && declaration.is_none(),
            "{} must remain the PTX-native raw-cluster-address carrier",
            policy.id
        ),
    }
    ensure!(
        policy.rust_module == "barrier"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use == recipe.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::barrier::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::barrier::{}", recipe.id)],
        "{} Rust API does not match its closed extended-mbarrier recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.lowering == "generated_mbarrier_extended_inline_ptx",
        "{} carrier or lowering does not match its closed extended-mbarrier recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == recipe.execution_scope,
        "{} convergence, memory clobber, or execution scope changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.ptx_result == recipe.ptx_result
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx == recipe.expected_ptx,
        "{} expected PTX does not match its closed extended-mbarrier recipe",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (
                IntrinsicBackend::LlvmNvptx,
                mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LlvmNvptx).0,
                Some(
                    mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LlvmNvptx)
                        .1,
                ),
            ),
            (
                IntrinsicBackend::LibNvvm,
                mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LibNvvm).0,
                Some(
                    mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LibNvvm).1,
                ),
            ),
        ],
        "extended-mbarrier",
    )?;
    ensure_no_other_family_contract(policy, "extended mbarrier")?;
    Ok(())
}
