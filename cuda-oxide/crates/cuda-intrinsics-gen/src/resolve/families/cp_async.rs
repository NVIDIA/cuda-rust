/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CpAsyncAdapter, CpAsyncCachePolicy, CpAsyncControlAdapter,
    CpAsyncControlOperation, CpAsyncCopySize, CpAsyncMbarrierAdapter, CpAsyncMbarrierOperation,
    CpAsyncMbarrierStateSpace, CpAsyncSourceSize, ImportedIntrinsic, IntrinsicBackend,
    MbarrierBasicAdapter, MbarrierBasicOperation, MbarrierStateSpace, OverlayIntrinsic,
    RuntimeValidation,
};
use crate::ptx::OperandPattern;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;

pub(in crate::resolve) struct CpAsyncCopyRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) selections: &'static [&'static str],
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn cp_async_copy_recipe(
    copy: &crate::model::CpAsyncCopy,
) -> Option<CpAsyncCopyRecipe> {
    match (copy.cache_policy, copy.copy_size, copy.source_size) {
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B4, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_4",
                abi_id: "i0086",
                operation_key: "memory.copy.async.global_to_shared.ca.4.full",
                rust_name: "cp_async_ca_4",
                dialect_op_type: "CpAsyncCa4Op",
                dialect_op_name: "nvvm.cp_async_ca_4",
                source_record: "int_nvvm_cp_async_ca_shared_global_4",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.4",
                selections: &["CP_ASYNC_CA_SHARED_GLOBAL_4"],
                summary: "Starts a four-byte cache-all asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B4, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_zfill_4",
                abi_id: "i0087",
                operation_key: "memory.copy.async.global_to_shared.ca.4.runtime_source_size",
                rust_name: "cp_async_ca_zfill_4",
                dialect_op_type: "CpAsyncCaZfill4Op",
                dialect_op_name: "nvvm.cp_async_ca_zfill_4",
                source_record: "int_nvvm_cp_async_ca_shared_global_4_s",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.4.s",
                selections: &[
                    "CP_ASYNC_CA_SHARED_GLOBAL_4_s",
                    "CP_ASYNC_CA_SHARED_GLOBAL_4_si",
                ],
                summary: "Starts a four-byte cache-all asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B8, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_8",
                abi_id: "i0088",
                operation_key: "memory.copy.async.global_to_shared.ca.8.full",
                rust_name: "cp_async_ca_8",
                dialect_op_type: "CpAsyncCa8Op",
                dialect_op_name: "nvvm.cp_async_ca_8",
                source_record: "int_nvvm_cp_async_ca_shared_global_8",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.8",
                selections: &["CP_ASYNC_CA_SHARED_GLOBAL_8"],
                summary: "Starts an eight-byte cache-all asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B8, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_zfill_8",
                abi_id: "i0089",
                operation_key: "memory.copy.async.global_to_shared.ca.8.runtime_source_size",
                rust_name: "cp_async_ca_zfill_8",
                dialect_op_type: "CpAsyncCaZfill8Op",
                dialect_op_name: "nvvm.cp_async_ca_zfill_8",
                source_record: "int_nvvm_cp_async_ca_shared_global_8_s",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.8.s",
                selections: &[
                    "CP_ASYNC_CA_SHARED_GLOBAL_8_s",
                    "CP_ASYNC_CA_SHARED_GLOBAL_8_si",
                ],
                summary: "Starts an eight-byte cache-all asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B16, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_16",
                abi_id: "i0090",
                operation_key: "memory.copy.async.global_to_shared.ca.16.full",
                rust_name: "cp_async_ca_16",
                dialect_op_type: "CpAsyncCa16Op",
                dialect_op_name: "nvvm.cp_async_ca_16",
                source_record: "int_nvvm_cp_async_ca_shared_global_16",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.16",
                selections: &["CP_ASYNC_CA_SHARED_GLOBAL_16"],
                summary: "Starts a sixteen-byte cache-all asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B16, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_zfill_16",
                abi_id: "i0091",
                operation_key: "memory.copy.async.global_to_shared.ca.16.runtime_source_size",
                rust_name: "cp_async_ca_zfill_16",
                dialect_op_type: "CpAsyncCaZfill16Op",
                dialect_op_name: "nvvm.cp_async_ca_zfill_16",
                source_record: "int_nvvm_cp_async_ca_shared_global_16_s",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.16.s",
                selections: &[
                    "CP_ASYNC_CA_SHARED_GLOBAL_16_s",
                    "CP_ASYNC_CA_SHARED_GLOBAL_16_si",
                ],
                summary: "Starts a sixteen-byte cache-all asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        (CpAsyncCachePolicy::Cg, CpAsyncCopySize::B16, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_cg_16",
                abi_id: "i0092",
                operation_key: "memory.copy.async.global_to_shared.cg.16.full",
                rust_name: "cp_async_cg_16",
                dialect_op_type: "CpAsyncCg16Op",
                dialect_op_name: "nvvm.cp_async_cg_16",
                source_record: "int_nvvm_cp_async_cg_shared_global_16",
                llvm_symbol: "llvm.nvvm.cp.async.cg.shared.global.16",
                selections: &["CP_ASYNC_CG_SHARED_GLOBAL_16"],
                summary: "Starts a sixteen-byte cache-global asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Cg, CpAsyncCopySize::B16, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_cg_zfill_16",
                abi_id: "i0093",
                operation_key: "memory.copy.async.global_to_shared.cg.16.runtime_source_size",
                rust_name: "cp_async_cg_zfill_16",
                dialect_op_type: "CpAsyncCgZfill16Op",
                dialect_op_name: "nvvm.cp_async_cg_zfill_16",
                source_record: "int_nvvm_cp_async_cg_shared_global_16_s",
                llvm_symbol: "llvm.nvvm.cp.async.cg.shared.global.16.s",
                selections: &[
                    "CP_ASYNC_CG_SHARED_GLOBAL_16_s",
                    "CP_ASYNC_CG_SHARED_GLOBAL_16_si",
                ],
                summary: "Starts a sixteen-byte cache-global asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        _ => None,
    }
}

pub(in crate::resolve) fn validate_cp_async_copy_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let copy = policy
        .cp_async_copy
        .as_ref()
        .with_context(|| format!("{} has no closed cp.async copy contract", policy.id))?;
    let recipe = cp_async_copy_recipe(copy).with_context(|| {
        format!(
            "{} requests an unsupported classic cp.async copy form",
            policy.id
        )
    })?;
    let dynamic_source_size = copy.source_size == CpAsyncSourceSize::Runtime;
    let expected_adapter = if dynamic_source_size {
        CpAsyncAdapter::DirectPointersAndSourceSize
    } else {
        CpAsyncAdapter::DirectPointers
    };
    ensure!(
        copy.adapter == expected_adapter,
        "{} cp.async source-size form and adapter disagree",
        policy.id
    );
    ensure!(
        copy.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded cp.async runtime validation",
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
        "{} cp.async identity does not match its closed recipe",
        policy.id
    );
    let rust_arguments = if dynamic_source_size {
        vec!["*mut u32", "*const u8", "u32"]
    } else {
        vec!["*mut u32", "*const u32"]
    };
    ensure!(
        policy.rust_module == "async_copy"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path
                == format!("cuda_intrinsics::async_copy::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::async_copy::{}", recipe.rust_name)],
        "{} must preserve its unsafe cp.async raw and compatibility API",
        policy.id
    );
    let llvm_arguments = if dynamic_source_size {
        vec!["shared_ptr", "global_ptr", "i32"]
    } else {
        vec!["shared_ptr", "global_ptr"]
    };
    let dialect_operands = if dynamic_source_size {
        vec!["ptr", "ptr", "i32"]
    } else {
        vec!["ptr", "ptr"]
    };
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cp_async_copy",
        "{} is outside the closed cp.async signature and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} cp.async effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.9.26.3.1 Data Movement and Conversion Instructions: cp.async"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async",
        "{} cp.async PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} cp.async summary does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrArgMemOnly",
                "IntrNoCallback",
                "NoAlias<arg0>",
                "NoAlias<arg1>",
                "ReadOnly<arg1>",
                "WriteOnly<arg0>",
            ],
        "{} cp.async effects disagree with the imported LLVM declaration",
        policy.id
    );
    let cache = match copy.cache_policy {
        CpAsyncCachePolicy::Ca => "ca",
        CpAsyncCachePolicy::Cg => "cg",
    };
    let mut operands = vec![
        OperandPattern::Address,
        OperandPattern::Address,
        OperandPattern::Exact {
            value: copy.copy_size.bytes().to_string(),
        },
    ];
    if dynamic_source_size {
        operands.push(OperandPattern::RegisterOrImmediate);
    }
    ensure!(
        policy.expected_ptx.mnemonic == "cp"
            && policy.expected_ptx.modifiers == ["async", cache, "shared", "global"]
            && policy.expected_ptx.operands == operands,
        "{} expected PTX does not match its cp.async recipe",
        policy.id
    );
    let actual_selections: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        actual_selections == recipe.selections.iter().copied().collect(),
        "{} imported cp.async selection set changed",
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
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define the reviewed typed-LLVM and inline-PTX cp.async routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cp_async_copy")?;
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes ldmatrix state with cp_async_copy",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) struct CpAsyncControlRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) selection: &'static str,
    pub(in crate::resolve) ptx_modifier: &'static str,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn cp_async_control_recipe(
    operation: CpAsyncControlOperation,
) -> CpAsyncControlRecipe {
    match operation {
        CpAsyncControlOperation::CommitGroup => CpAsyncControlRecipe {
            id: "cp_async_commit_group",
            abi_id: "i0094",
            operation_key: "memory.copy.async.group.commit",
            rust_name: "cp_async_commit_group",
            dialect_op_type: "CpAsyncCommitGroupOp",
            dialect_op_name: "nvvm.cp_async_commit_group",
            source_record: "int_nvvm_cp_async_commit_group",
            llvm_symbol: "llvm.nvvm.cp.async.commit.group",
            selection: "CP_ASYNC_COMMIT_GROUP",
            ptx_modifier: "commit_group",
            summary: "Commits this thread's uncommitted asynchronous copies as one group.",
        },
        CpAsyncControlOperation::WaitAll => CpAsyncControlRecipe {
            id: "cp_async_wait_all",
            abi_id: "i0095",
            operation_key: "memory.copy.async.group.wait_all",
            rust_name: "cp_async_wait_all",
            dialect_op_type: "CpAsyncWaitAllOp",
            dialect_op_name: "nvvm.cp_async_wait_all",
            source_record: "int_nvvm_cp_async_wait_all",
            llvm_symbol: "llvm.nvvm.cp.async.wait.all",
            selection: "CP_ASYNC_WAIT_ALL",
            ptx_modifier: "wait_all",
            summary: "Waits for all asynchronous copy groups issued by this thread.",
        },
        CpAsyncControlOperation::WaitGroup => CpAsyncControlRecipe {
            id: "cp_async_wait_group",
            abi_id: "i0096",
            operation_key: "memory.copy.async.group.wait_max_pending",
            rust_name: "cp_async_wait_group",
            dialect_op_type: "CpAsyncWaitGroupOp",
            dialect_op_name: "nvvm.cp_async_wait_group",
            source_record: "int_nvvm_cp_async_wait_group",
            llvm_symbol: "llvm.nvvm.cp.async.wait.group",
            selection: "CP_ASYNC_WAIT_GROUP",
            ptx_modifier: "wait_group",
            summary: "Waits until at most the compile-time constant number of recent asynchronous copy groups remain pending.",
        },
    }
}

pub(in crate::resolve) fn validate_cp_async_control_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let control = policy
        .cp_async_control
        .as_ref()
        .with_context(|| format!("{} has no closed cp.async control contract", policy.id))?;
    let recipe = cp_async_control_recipe(control.operation);
    let has_operand = control.operation == CpAsyncControlOperation::WaitGroup;
    let expected_adapter = if has_operand {
        CpAsyncControlAdapter::CompileTimeConstantMaxPending
    } else {
        CpAsyncControlAdapter::NoOperands
    };
    ensure!(
        control.adapter == expected_adapter,
        "{} cp.async control and adapter disagree",
        policy.id
    );
    ensure!(
        control.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded cp.async control runtime validation",
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
        "{} cp.async control identity does not match its closed recipe",
        policy.id
    );
    let rust_arguments = if has_operand { vec!["u32"] } else { vec![] };
    let dialect_operands = if has_operand { vec!["i32"] } else { vec![] };
    ensure!(
        policy.rust_module == "async_copy"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path
                == format!("cuda_intrinsics::async_copy::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::async_copy::{}", recipe.rust_name)]
            && policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == dialect_operands
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cp_async_control",
        "{} is outside the closed cp.async control API and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} cp.async control effects or target floor disagree with the closed recipe",
        policy.id
    );
    let (ptx_isa_section, ptx_isa_url) = match control.operation {
        CpAsyncControlOperation::CommitGroup => (
            "9.7.9.26.3.2 Data Movement and Conversion Instructions: cp.async.commit_group",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group",
        ),
        CpAsyncControlOperation::WaitAll | CpAsyncControlOperation::WaitGroup => (
            "9.7.9.26.3.3 Data Movement and Conversion Instructions: cp.async.wait_group / cp.async.wait_all",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all",
        ),
    };
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == ptx_isa_section
            && policy.ptx_isa_url == ptx_isa_url,
        "{} cp.async control PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} cp.async control summary does not match its closed recipe",
        policy.id
    );
    let expected_properties: Vec<String> = if has_operand {
        vec!["ImmArg<arg0>".into()]
    } else {
        vec![]
    };
    ensure!(
        declaration.properties == expected_properties,
        "{} cp.async control properties disagree with the imported declaration",
        policy.id
    );
    let operands = if has_operand {
        vec![OperandPattern::Immediate]
    } else {
        vec![]
    };
    ensure!(
        policy.expected_ptx.mnemonic == "cp"
            && policy.expected_ptx.modifiers == ["async", recipe.ptx_modifier]
            && policy.expected_ptx.operands == operands
            && declaration.selections.len() == 1
            && declaration.selections[0].source_record == recipe.selection,
        "{} expected PTX or imported selection disagrees with its cp.async control recipe",
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
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define the reviewed typed-LLVM and inline-PTX cp.async control routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cp_async_control")?;
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes ldmatrix state with cp_async_control",
        policy.id
    );
    Ok(())
}

pub(in crate::resolve) struct CpAsyncMbarrierRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) llvm_argument: &'static str,
    pub(in crate::resolve) selection: &'static str,
    pub(in crate::resolve) selection_asm: &'static str,
    pub(in crate::resolve) modifiers: &'static [&'static str],
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn cp_async_mbarrier_recipe(
    operation: CpAsyncMbarrierOperation,
    state_space: CpAsyncMbarrierStateSpace,
) -> CpAsyncMbarrierRecipe {
    match (operation, state_space) {
        (CpAsyncMbarrierOperation::Arrive, CpAsyncMbarrierStateSpace::Generic) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive",
                abi_id: "i0101",
                operation_key: "memory.copy.async.mbarrier.arrive.generic.increment",
                dialect_op_type: "CpAsyncMbarrierArriveOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive",
                source_record: "int_nvvm_cp_async_mbarrier_arrive",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive",
                llvm_argument: "ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE",
                selection_asm: "cp.async.mbarrier.arrive.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-memory barrier using balanced pending-count accounting.",
            }
        }
        (CpAsyncMbarrierOperation::ArriveNoInc, CpAsyncMbarrierStateSpace::Generic) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive_noinc",
                abi_id: "i0103",
                operation_key: "memory.copy.async.mbarrier.arrive.generic.no_increment",
                dialect_op_type: "CpAsyncMbarrierArriveNoIncOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive_noinc",
                source_record: "int_nvvm_cp_async_mbarrier_arrive_noinc",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive.noinc",
                llvm_argument: "ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE_NOINC",
                selection_asm: "cp.async.mbarrier.arrive.noinc.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "noinc", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-memory barrier without incrementing its pending count.",
            }
        }
        (CpAsyncMbarrierOperation::ArriveNoInc, CpAsyncMbarrierStateSpace::Shared) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive_noinc_shared",
                abi_id: "i0104",
                operation_key: "memory.copy.async.mbarrier.arrive.shared.no_increment",
                dialect_op_type: "CpAsyncMbarrierArriveNoIncSharedOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive_noinc_shared",
                source_record: "int_nvvm_cp_async_mbarrier_arrive_noinc_shared",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared",
                llvm_argument: "shared_ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE_NOINC_SHARED",
                selection_asm: "cp.async.mbarrier.arrive.noinc.shared.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "noinc", "shared", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-address barrier without incrementing its pending count.",
            }
        }
        (CpAsyncMbarrierOperation::Arrive, CpAsyncMbarrierStateSpace::Shared) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive_shared",
                abi_id: "i0102",
                operation_key: "memory.copy.async.mbarrier.arrive.shared.increment",
                dialect_op_type: "CpAsyncMbarrierArriveSharedOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive_shared",
                source_record: "int_nvvm_cp_async_mbarrier_arrive_shared",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive.shared",
                llvm_argument: "shared_ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE_SHARED",
                selection_asm: "cp.async.mbarrier.arrive.shared.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "shared", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-address barrier using balanced pending-count accounting.",
            }
        }
    }
}

pub(in crate::resolve) fn validate_cp_async_mbarrier_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let bridge = policy
        .cp_async_mbarrier
        .as_ref()
        .with_context(|| format!("{} has no closed cp.async mbarrier contract", policy.id))?;
    let recipe = cp_async_mbarrier_recipe(bridge.operation, bridge.state_space);
    ensure!(
        bridge.adapter == CpAsyncMbarrierAdapter::PointerToVoid,
        "{} cp.async mbarrier adapter changed",
        policy.id
    );
    ensure!(
        bridge.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded cp.async mbarrier runtime validation",
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
        "{} cp.async mbarrier identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "async_copy"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == ["*mut u64"]
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::async_copy::{}", recipe.id)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::async_copy::{}", recipe.id)],
        "{} is outside the closed cp.async mbarrier Rust API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["ptr"]
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == [recipe.llvm_argument]
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cp_async_mbarrier",
        "{} is outside the closed cp.async mbarrier signature and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} cp.async mbarrier effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.16.18 Parallel Synchronization and Communication Instructions: cp.async.mbarrier.arrive"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive",
        "{} cp.async mbarrier PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} cp.async mbarrier summary does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrConvergent", "IntrNoCallback"],
        "{} cp.async mbarrier properties disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "cp"
            && policy
                .expected_ptx
                .modifiers
                .iter()
                .map(String::as_str)
                .eq(recipe.modifiers.iter().copied())
            && policy.expected_ptx.operands == [OperandPattern::Address],
        "{} expected PTX does not match its cp.async mbarrier recipe",
        policy.id
    );
    ensure!(
        declaration.selections.len() == 1
            && declaration.selections[0].source_record == recipe.selection
            && declaration.selections[0].asm == recipe.selection_asm
            && declaration.selections[0].predicates
                == [
                    "Subtarget->getSmVersion() >= 80",
                    "Subtarget->getPTXVersion() >= 70",
                ]
            && declaration.selections[0].constraints.is_empty(),
        "{} imported cp.async mbarrier selection changed",
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
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define the reviewed typed-LLVM and inline-PTX cp.async mbarrier routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cp_async_mbarrier")?;
    Ok(())
}

pub(in crate::resolve) struct MbarrierBasicRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) rust_arguments: &'static [&'static str],
    pub(in crate::resolve) rust_result: &'static str,
    pub(in crate::resolve) must_use: bool,
    pub(in crate::resolve) adapter: MbarrierBasicAdapter,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) dialect_operands: &'static [&'static str],
    pub(in crate::resolve) dialect_results: &'static [&'static str],
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_results: &'static [&'static str],
    pub(in crate::resolve) memory: &'static str,
    pub(in crate::resolve) ptx_result: &'static str,
    pub(in crate::resolve) selection: &'static str,
    pub(in crate::resolve) selection_asm: &'static str,
    pub(in crate::resolve) ptx_modifiers: &'static [&'static str],
    pub(in crate::resolve) ptx_isa_section: &'static str,
    pub(in crate::resolve) ptx_isa_url: &'static str,
    pub(in crate::resolve) llvm_nvptx_mechanism: BackendLoweringMechanism,
    pub(in crate::resolve) lib_nvvm_mechanism: BackendLoweringMechanism,
    pub(in crate::resolve) properties: &'static [&'static str],
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn mbarrier_basic_recipe(
    operation: MbarrierBasicOperation,
) -> MbarrierBasicRecipe {
    match operation {
        MbarrierBasicOperation::Init => MbarrierBasicRecipe {
            id: "mbarrier_init",
            abi_id: "i0097",
            operation_key: "barrier.mbarrier.init.shared.cta",
            rust_arguments: &["*mut u64", "u32"],
            rust_result: "()",
            must_use: false,
            adapter: MbarrierBasicAdapter::InitPointerCountToVoid,
            dialect_op_type: "MbarrierInitSharedOp",
            dialect_op_name: "nvvm.mbarrier_init_shared",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &[],
            source_record: "int_nvvm_mbarrier_init_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.init.shared",
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &[],
            memory: "read_write",
            ptx_result: "()",
            selection: "MBARRIER_INIT_SHARED",
            selection_asm: "mbarrier.init.shared.b64 \t[$addr], $count;",
            ptx_modifiers: &["init", "shared", "b64"],
            ptx_isa_section: "9.7.14.16.12 Parallel Synchronization and Communication Instructions: mbarrier.init",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init",
            llvm_nvptx_mechanism: BackendLoweringMechanism::TypedNvvm,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &["IntrConvergent", "IntrNoCallback"],
            summary: "Initializes a CTA shared-memory barrier with the expected arrival count.",
        },
        MbarrierBasicOperation::Arrive => MbarrierBasicRecipe {
            id: "mbarrier_arrive",
            abi_id: "i0098",
            operation_key: "barrier.mbarrier.arrive.shared.cta",
            rust_arguments: &["*const u64"],
            rust_result: "u64",
            must_use: true,
            adapter: MbarrierBasicAdapter::ArrivePointerToToken,
            dialect_op_type: "MbarrierArriveSharedOp",
            dialect_op_name: "nvvm.mbarrier_arrive_shared",
            dialect_operands: &["ptr"],
            dialect_results: &["i64"],
            source_record: "int_nvvm_mbarrier_arrive_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.arrive.shared",
            llvm_arguments: &["shared_ptr"],
            llvm_results: &["i64"],
            memory: "read_write",
            ptx_result: "u64",
            selection: "MBARRIER_ARRIVE_SHARED",
            selection_asm: "mbarrier.arrive.shared.b64 \t$state, [$addr];",
            ptx_modifiers: &["arrive", "shared", "b64"],
            ptx_isa_section: "9.7.14.16.16 Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            llvm_nvptx_mechanism: BackendLoweringMechanism::TypedNvvm,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &["IntrConvergent", "IntrNoCallback"],
            summary: "Arrives at a CTA shared-memory barrier and returns its phase token.",
        },
        MbarrierBasicOperation::ArriveNoComplete => MbarrierBasicRecipe {
            id: "mbarrier_arrive_no_complete",
            abi_id: "i1017",
            operation_key: "barrier.mbarrier.arrive.no_complete.shared.cta",
            rust_arguments: &["*const u64", "u32"],
            rust_result: "u64",
            must_use: true,
            adapter: MbarrierBasicAdapter::ArriveNoCompletePointerCountToToken,
            dialect_op_type: "MbarrierArriveNoCompleteSharedOp",
            dialect_op_name: "nvvm.mbarrier_arrive_no_complete_shared",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i64"],
            source_record: "int_nvvm_mbarrier_arrive_noComplete_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.arrive.noComplete.shared",
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i64"],
            memory: "read_write",
            ptx_result: "u64",
            selection: "MBARRIER_ARRIVE_NOCOMPLETE_SHARED",
            selection_asm: "mbarrier.arrive.noComplete.shared.b64 \t$state, [$addr], $count;",
            ptx_modifiers: &["arrive", "noComplete", "shared", "b64"],
            ptx_isa_section: "9.7.14.16.16 Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            llvm_nvptx_mechanism: BackendLoweringMechanism::TypedNvvm,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &["IntrConvergent", "IntrNoCallback"],
            summary: "Arrives at a CTA shared-memory barrier by a dynamic count without completing the current phase and returns its prior opaque state.",
        },
        MbarrierBasicOperation::TestWait => MbarrierBasicRecipe {
            id: "mbarrier_test_wait",
            abi_id: "i0099",
            operation_key: "barrier.mbarrier.test_wait.shared.cta",
            rust_arguments: &["*const u64", "u64"],
            rust_result: "bool",
            must_use: true,
            adapter: MbarrierBasicAdapter::TestWaitPointerTokenToPredicate,
            dialect_op_type: "MbarrierTestWaitSharedOp",
            dialect_op_name: "nvvm.mbarrier_test_wait_shared",
            dialect_operands: &["ptr", "i64"],
            dialect_results: &["i1"],
            source_record: "int_nvvm_mbarrier_test_wait_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.test.wait.shared",
            llvm_arguments: &["shared_ptr", "i64"],
            llvm_results: &["i1"],
            memory: "read_write",
            ptx_result: "bool",
            selection: "MBARRIER_TEST_WAIT_SHARED",
            selection_asm: "mbarrier.test_wait.shared.b64 \t$res, [$addr], $state;",
            ptx_modifiers: &["test_wait", "shared", "b64"],
            ptx_isa_section: "9.7.14.16.19 Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            llvm_nvptx_mechanism: BackendLoweringMechanism::InlinePtx,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &["IntrConvergent", "IntrNoCallback"],
            summary: "Tests whether the CTA shared-memory barrier phase for a token is complete.",
        },
        MbarrierBasicOperation::Inval => MbarrierBasicRecipe {
            id: "mbarrier_inval",
            abi_id: "i0100",
            operation_key: "barrier.mbarrier.inval.shared.cta",
            rust_arguments: &["*mut u64"],
            rust_result: "()",
            must_use: false,
            adapter: MbarrierBasicAdapter::InvalPointerToVoid,
            dialect_op_type: "MbarrierInvalSharedOp",
            dialect_op_name: "nvvm.mbarrier_inval_shared",
            dialect_operands: &["ptr"],
            dialect_results: &[],
            source_record: "int_nvvm_mbarrier_inval_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.inval.shared",
            llvm_arguments: &["shared_ptr"],
            llvm_results: &[],
            memory: "write",
            ptx_result: "()",
            selection: "MBARRIER_INVAL_SHARED",
            selection_asm: "mbarrier.inval.shared.b64 \t[$addr];",
            ptx_modifiers: &["inval", "shared", "b64"],
            ptx_isa_section: "9.7.14.16.13 Parallel Synchronization and Communication Instructions: mbarrier.inval",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-inval",
            llvm_nvptx_mechanism: BackendLoweringMechanism::TypedNvvm,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &[
                "IntrArgMemOnly",
                "IntrConvergent",
                "IntrNoCallback",
                "IntrWriteMem",
                "NoCapture<arg0>",
                "WriteOnly<arg0>",
            ],
            summary: "Invalidates a CTA shared-memory barrier after its users have finished.",
        },
    }
}

pub(in crate::resolve) fn mbarrier_expected_operands(
    operation: MbarrierBasicOperation,
) -> Vec<OperandPattern> {
    match operation {
        MbarrierBasicOperation::Init => {
            vec![OperandPattern::Address, OperandPattern::RegisterOrImmediate]
        }
        MbarrierBasicOperation::Arrive => {
            vec![OperandPattern::Register, OperandPattern::Address]
        }
        MbarrierBasicOperation::ArriveNoComplete => vec![
            OperandPattern::Register,
            OperandPattern::Address,
            OperandPattern::Register,
        ],
        MbarrierBasicOperation::TestWait => vec![
            OperandPattern::Register,
            OperandPattern::Address,
            OperandPattern::Register,
        ],
        MbarrierBasicOperation::Inval => vec![OperandPattern::Address],
    }
}

pub(in crate::resolve) fn validate_mbarrier_basic_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let mbarrier = policy
        .mbarrier_basic
        .as_ref()
        .with_context(|| format!("{} has no closed basic mbarrier contract", policy.id))?;
    let recipe = mbarrier_basic_recipe(mbarrier.operation);
    ensure!(
        mbarrier.state_space == MbarrierStateSpace::Shared && mbarrier.adapter == recipe.adapter,
        "{} mbarrier operation, state space, and adapter disagree",
        policy.id
    );
    ensure!(
        mbarrier.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded mbarrier runtime validation",
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
        "{} mbarrier identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "barrier"
            && policy.rust_name == recipe.id
            && policy
                .rust_arguments
                .iter()
                .map(String::as_str)
                .eq(recipe.rust_arguments.iter().copied())
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use == recipe.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::barrier::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::barrier::{}", recipe.id)],
        "{} must preserve its unsafe mbarrier raw and compatibility API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy
                .dialect_operands
                .iter()
                .map(String::as_str)
                .eq(recipe.dialect_operands.iter().copied())
            && policy
                .dialect_results
                .iter()
                .map(String::as_str)
                .eq(recipe.dialect_results.iter().copied())
            && policy
                .llvm_arguments
                .iter()
                .map(String::as_str)
                .eq(recipe.llvm_arguments.iter().copied())
            && policy
                .llvm_results
                .iter()
                .map(String::as_str)
                .eq(recipe.llvm_results.iter().copied())
            && policy.lowering == "generated_mbarrier_basic",
        "{} is outside the closed mbarrier signature and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == recipe.memory
            && policy.convergent
            && policy.execution_scope == "cta"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == recipe.ptx_result
            && policy.targets == "all",
        "{} mbarrier effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} mbarrier PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} mbarrier summary does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration
            .properties
            .iter()
            .map(String::as_str)
            .eq(recipe.properties.iter().copied()),
        "{} mbarrier properties disagree with the imported LLVM declaration",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mbarrier"
            && policy
                .expected_ptx
                .modifiers
                .iter()
                .map(String::as_str)
                .eq(recipe.ptx_modifiers.iter().copied())
            && policy.expected_ptx.operands == mbarrier_expected_operands(mbarrier.operation),
        "{} expected PTX does not match its mbarrier recipe",
        policy.id
    );
    ensure!(
        declaration.selections.len() == 1
            && declaration.selections[0].source_record == recipe.selection
            && declaration.selections[0].asm == recipe.selection_asm
            && declaration.selections[0].predicates
                == [
                    "Subtarget->getSmVersion() >= 80",
                    "Subtarget->getPTXVersion() >= 70",
                ]
            && declaration.selections[0].constraints.is_empty(),
        "{} imported mbarrier selection changed",
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
                    (IntrinsicBackend::LlvmNvptx, recipe.llvm_nvptx_mechanism),
                    (IntrinsicBackend::LibNvvm, recipe.lib_nvvm_mechanism),
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define exactly the reviewed mbarrier backend routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "mbarrier_basic")?;
    Ok(())
}
