/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ExecutionControlOperation, ImportedIntrinsic, IntrinsicBackend,
    OverlayIntrinsic,
};
use crate::ptx::OperandPattern;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use super::*;
use crate::resolve::guards::*;

pub(in crate::resolve) const EXECUTION_CONTROL_OPERATIONS: [ExecutionControlOperation; 8] = [
    ExecutionControlOperation::BarrierCtaSync,
    ExecutionControlOperation::BarrierCtaSyncAligned,
    ExecutionControlOperation::BarrierCtaArrive,
    ExecutionControlOperation::BarrierCtaArriveAligned,
    ExecutionControlOperation::GridDependencyLaunchDependents,
    ExecutionControlOperation::GridDependencyWait,
    ExecutionControlOperation::SetMaxNRegInc,
    ExecutionControlOperation::SetMaxNRegDec,
];

pub(in crate::resolve) fn validate_execution_control_family_completeness(
    intrinsics: &[OverlayIntrinsic],
) -> Result<()> {
    let has_execution_control_family = intrinsics.iter().any(|policy| {
        matches!(
            policy.family.as_str(),
            "counted_barrier" | "grid_dependency" | "register_control"
        )
    });
    if !has_execution_control_family {
        return Ok(());
    }
    let actual = intrinsics
        .iter()
        .filter_map(|policy| ExecutionControlOperation::from_catalog_id(&policy.id))
        .collect::<BTreeSet<_>>();
    let expected = EXECUTION_CONTROL_OPERATIONS
        .into_iter()
        .collect::<BTreeSet<_>>();
    ensure!(
        actual == expected,
        "execution-control overlay must admit all four counted barriers, both grid-dependency controls, and both setmaxnreg operations"
    );
    for family in ["counted_barrier", "grid_dependency", "register_control"] {
        ensure!(
            intrinsics
                .iter()
                .filter(|policy| policy.family == family)
                .all(|policy| {
                    ExecutionControlOperation::from_catalog_id(&policy.id)
                        .is_some_and(|operation| operation.family() == family)
                }),
            "{family} overlay contains an operation outside its closed instruction family"
        );
    }
    Ok(())
}

#[derive(Clone)]
pub(in crate::resolve) struct ExecutionControlRecipe {
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_module: &'static str,
    pub(in crate::resolve) rust_arguments: &'static [&'static str],
    pub(in crate::resolve) compatibility_path: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) dialect_operands: &'static [&'static str],
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) imported_classes: &'static [&'static str],
    pub(in crate::resolve) imported_properties: &'static [&'static str],
    pub(in crate::resolve) memory: &'static str,
    pub(in crate::resolve) convergent: bool,
    pub(in crate::resolve) execution_scope: &'static str,
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) minimum_sm: Option<&'static str>,
    pub(in crate::resolve) targets: &'static str,
    pub(in crate::resolve) ptx_isa_section: &'static str,
    pub(in crate::resolve) ptx_isa_url: &'static str,
    pub(in crate::resolve) mnemonic: &'static str,
    pub(in crate::resolve) modifiers: &'static [&'static str],
    pub(in crate::resolve) operands: &'static [OperandPattern],
    pub(in crate::resolve) selection_records: &'static [&'static str],
    pub(in crate::resolve) selection_asm: &'static str,
    pub(in crate::resolve) selection_predicates: &'static [&'static str],
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn execution_control_recipe(
    operation: ExecutionControlOperation,
) -> ExecutionControlRecipe {
    use ExecutionControlOperation::*;

    const EMPTY: &[&str] = &[];
    const I32: &[&str] = &["i32"];
    const I32_I32: &[&str] = &["i32", "i32"];
    const U32: &[&str] = &["u32"];
    const U32_U32: &[&str] = &["u32", "u32"];
    const BASE_CLASSES: &[&str] = &["SDPatternOperator", "Intrinsic"];
    const DEFAULT_CLASSES: &[&str] = &["SDPatternOperator", "Intrinsic", "DefaultAttrsIntrinsic"];
    const BARRIER_PROPERTIES: &[&str] = &["IntrConvergent", "IntrNoCallback"];
    const GRID_PROPERTIES: &[&str] = &["IntrHasSideEffects", "IntrNoMem"];
    const SETMAX_PROPERTIES: &[&str] = &[
        "ImmArg<arg0>",
        "IntrConvergent",
        "IntrHasSideEffects",
        "IntrNoMem",
        "Range<arg0,24,257>",
    ];
    const TWO_REGISTER_OR_IMMEDIATE: &[OperandPattern] = &[
        OperandPattern::RegisterOrImmediate,
        OperandPattern::RegisterOrImmediate,
    ];
    const ONE_IMMEDIATE: &[OperandPattern] = &[OperandPattern::Immediate];
    const NO_OPERANDS: &[OperandPattern] = &[];
    const BARRIER_SYNC_SELECTIONS: &[&str] = &[
        "BARRIER_CTA_SYNC_ii",
        "BARRIER_CTA_SYNC_ir",
        "BARRIER_CTA_SYNC_ri",
        "BARRIER_CTA_SYNC_rr",
    ];
    const BAR_SYNC_SELECTIONS: &[&str] = &[
        "BARRIER_CTA_SYNC_ALIGNED_ii",
        "BARRIER_CTA_SYNC_ALIGNED_ir",
        "BARRIER_CTA_SYNC_ALIGNED_ri",
        "BARRIER_CTA_SYNC_ALIGNED_rr",
    ];
    const BARRIER_ARRIVE_SELECTIONS: &[&str] = &[
        "BARRIER_CTA_ARRIVE_ii",
        "BARRIER_CTA_ARRIVE_ir",
        "BARRIER_CTA_ARRIVE_ri",
        "BARRIER_CTA_ARRIVE_rr",
    ];
    const BAR_ARRIVE_SELECTIONS: &[&str] = &[
        "BARRIER_CTA_ARRIVE_ALIGNED_ii",
        "BARRIER_CTA_ARRIVE_ALIGNED_ir",
        "BARRIER_CTA_ARRIVE_ALIGNED_ri",
        "BARRIER_CTA_ARRIVE_ALIGNED_rr",
    ];
    const PTX_60: &[&str] = &["Subtarget->getPTXVersion() >= 60"];
    const GRID_PREDICATES: &[&str] = &[
        "Subtarget->getSmVersion() >= 90",
        "Subtarget->getPTXVersion() >= 78",
    ];
    const SETMAX_PREDICATES: &[&str] = &["Subtarget->hasSetMaxNRegSupport()"];
    const SYNC_MODIFIERS: &[&str] = &["sync"];
    const ARRIVE_MODIFIERS: &[&str] = &["arrive"];
    const LAUNCH_DEPENDENTS_MODIFIERS: &[&str] = &["launch_dependents"];
    const WAIT_MODIFIERS: &[&str] = &["wait"];
    const SETMAX_INC_MODIFIERS: &[&str] = &["inc", "sync", "aligned", "u32"];
    const SETMAX_DEC_MODIFIERS: &[&str] = &["dec", "sync", "aligned", "u32"];
    const GRID_LAUNCH_SELECTIONS: &[&str] = &["GRIDDEPCONTROL_LAUNCH_DEPENDENTS"];
    const GRID_WAIT_SELECTIONS: &[&str] = &["GRIDDEPCONTROL_WAIT"];
    const SETMAX_INC_SELECTIONS: &[&str] = &["anonymous_21745"];
    const SETMAX_DEC_SELECTIONS: &[&str] = &["anonymous_21747"];
    const BAR_SECTION: &str =
        "9.7.14.1 Parallel Synchronization and Communication Instructions: bar, barrier";
    const BAR_URL: &str = "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier";
    const GRID_SECTION: &str =
        "9.7.15.1 Parallel Synchronization and Communication Instructions: griddepcontrol";
    const GRID_URL: &str = "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol";
    const SETMAX_SECTION: &str = "9.7.16.5 Miscellaneous Instructions: setmaxnreg";
    const SETMAX_URL: &str = "https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg";

    let (
        abi_id,
        id,
        operation_key,
        source_record,
        llvm_symbol,
        rust_module,
        rust_arguments,
        compatibility_path,
        dialect_op_type,
        dialect_op_name,
        dialect_operands,
        llvm_arguments,
        imported_classes,
        imported_properties,
        memory,
        convergent,
        execution_scope,
        minimum_ptx,
        minimum_sm,
        targets,
        ptx_isa_section,
        ptx_isa_url,
        mnemonic,
        modifiers,
        operands,
        selection_records,
        selection_asm,
        selection_predicates,
        summary,
    ) = match operation {
        BarrierCtaSync => (
            "i0883",
            "barrier_cta_sync",
            "synchronization.cta.barrier.count",
            "int_nvvm_barrier_cta_sync_count",
            "llvm.nvvm.barrier.cta.sync.count",
            "barrier",
            U32_U32,
            "cuda_device::barrier::barrier_cta_sync",
            "BarrierCtaSyncCountOp",
            "nvvm.barrier_cta_sync_count",
            I32_I32,
            I32_I32,
            BASE_CLASSES,
            BARRIER_PROPERTIES,
            "read_write",
            true,
            "cta",
            "6.0",
            Some("sm_30"),
            "all",
            BAR_SECTION,
            BAR_URL,
            "barrier",
            SYNC_MODIFIERS,
            TWO_REGISTER_OR_IMMEDIATE,
            BARRIER_SYNC_SELECTIONS,
            "barrier.sync \t$i, $j;",
            PTX_60,
            "Synchronizes the requested number of CTA threads at a numbered barrier.",
        ),
        BarrierCtaSyncAligned => (
            "i0884",
            "barrier_cta_sync_aligned",
            "synchronization.cta.bar.aligned.count",
            "int_nvvm_barrier_cta_sync_aligned_count",
            "llvm.nvvm.barrier.cta.sync.aligned.count",
            "barrier",
            U32_U32,
            "cuda_device::barrier::barrier_cta_sync_aligned",
            "BarrierCtaSyncAlignedCountOp",
            "nvvm.barrier_cta_sync_aligned_count",
            I32_I32,
            I32_I32,
            BASE_CLASSES,
            BARRIER_PROPERTIES,
            "read_write",
            true,
            "cta",
            "2.0",
            Some("sm_20"),
            "all",
            BAR_SECTION,
            BAR_URL,
            "bar",
            SYNC_MODIFIERS,
            TWO_REGISTER_OR_IMMEDIATE,
            BAR_SYNC_SELECTIONS,
            "bar.sync \t$i, $j;",
            EMPTY,
            "Synchronizes the requested number of CTA threads at an aligned numbered barrier.",
        ),
        BarrierCtaArrive => (
            "i0885",
            "barrier_cta_arrive",
            "synchronization.cta.barrier.arrive.count",
            "int_nvvm_barrier_cta_arrive_count",
            "llvm.nvvm.barrier.cta.arrive.count",
            "barrier",
            U32_U32,
            "cuda_device::barrier::barrier_cta_arrive",
            "BarrierCtaArriveCountOp",
            "nvvm.barrier_cta_arrive_count",
            I32_I32,
            I32_I32,
            BASE_CLASSES,
            BARRIER_PROPERTIES,
            "read_write",
            true,
            "cta",
            "6.0",
            Some("sm_30"),
            "all",
            BAR_SECTION,
            BAR_URL,
            "barrier",
            ARRIVE_MODIFIERS,
            TWO_REGISTER_OR_IMMEDIATE,
            BARRIER_ARRIVE_SELECTIONS,
            "barrier.arrive \t$i, $j;",
            PTX_60,
            "Signals arrival of the requested number of CTA threads at a numbered barrier without waiting.",
        ),
        BarrierCtaArriveAligned => (
            "i0886",
            "barrier_cta_arrive_aligned",
            "synchronization.cta.bar.arrive.aligned.count",
            "int_nvvm_barrier_cta_arrive_aligned_count",
            "llvm.nvvm.barrier.cta.arrive.aligned.count",
            "barrier",
            U32_U32,
            "cuda_device::barrier::barrier_cta_arrive_aligned",
            "BarrierCtaArriveAlignedCountOp",
            "nvvm.barrier_cta_arrive_aligned_count",
            I32_I32,
            I32_I32,
            BASE_CLASSES,
            BARRIER_PROPERTIES,
            "read_write",
            true,
            "cta",
            "2.0",
            Some("sm_20"),
            "all",
            BAR_SECTION,
            BAR_URL,
            "bar",
            ARRIVE_MODIFIERS,
            TWO_REGISTER_OR_IMMEDIATE,
            BAR_ARRIVE_SELECTIONS,
            "bar.arrive \t$i, $j;",
            EMPTY,
            "Signals arrival of the requested number of CTA threads at an aligned numbered barrier without waiting.",
        ),
        GridDependencyLaunchDependents => (
            "i0913",
            "grid_dependency_launch_dependents",
            "synchronization.grid.dependency.launch_dependents",
            "int_nvvm_griddepcontrol_launch_dependents",
            "llvm.nvvm.griddepcontrol.launch.dependents",
            "grid",
            EMPTY,
            "cuda_device::grid::dependency::trigger_dependents",
            "GridDependencyLaunchDependentsOp",
            "nvvm.grid_dependency_launch_dependents",
            EMPTY,
            EMPTY,
            BASE_CLASSES,
            GRID_PROPERTIES,
            "none",
            false,
            "grid",
            "7.8",
            Some("sm_90"),
            "all",
            GRID_SECTION,
            GRID_URL,
            "griddepcontrol",
            LAUNCH_DEPENDENTS_MODIFIERS,
            NO_OPERANDS,
            GRID_LAUNCH_SELECTIONS,
            "griddepcontrol.launch_dependents;",
            GRID_PREDICATES,
            "Makes dependent grids eligible to launch before the current grid completes.",
        ),
        GridDependencyWait => (
            "i0914",
            "grid_dependency_wait",
            "synchronization.grid.dependency.wait",
            "int_nvvm_griddepcontrol_wait",
            "llvm.nvvm.griddepcontrol.wait",
            "grid",
            EMPTY,
            "cuda_device::grid::dependency::wait",
            "GridDependencyWaitOp",
            "nvvm.grid_dependency_wait",
            EMPTY,
            EMPTY,
            BASE_CLASSES,
            GRID_PROPERTIES,
            "none",
            false,
            "grid",
            "7.8",
            Some("sm_90"),
            "all",
            GRID_SECTION,
            GRID_URL,
            "griddepcontrol",
            WAIT_MODIFIERS,
            NO_OPERANDS,
            GRID_WAIT_SELECTIONS,
            "griddepcontrol.wait;",
            GRID_PREDICATES,
            "Waits until all prerequisite grids have completed.",
        ),
        SetMaxNRegInc => (
            "i0915",
            "setmaxnreg_inc",
            "register_control.warpgroup.setmaxnreg.increment",
            "int_nvvm_setmaxnreg_inc_sync_aligned_u32",
            "llvm.nvvm.setmaxnreg.inc.sync.aligned.u32",
            "thread",
            U32,
            "cuda_device::thread::__setmaxnreg_inc",
            "SetMaxNRegIncOp",
            "nvvm.setmaxnreg_inc",
            EMPTY,
            I32,
            DEFAULT_CLASSES,
            SETMAX_PROPERTIES,
            "none",
            true,
            "warpgroup",
            "8.0",
            None,
            TENSOR_MAP_REPLACE_TARGETS,
            SETMAX_SECTION,
            SETMAX_URL,
            "setmaxnreg",
            SETMAX_INC_MODIFIERS,
            ONE_IMMEDIATE,
            SETMAX_INC_SELECTIONS,
            "setmaxnreg.inc.sync.aligned.u32 \t$reg_count;",
            SETMAX_PREDICATES,
            "Increases the executing warpgroup's maximum register allocation to the immediate count.",
        ),
        SetMaxNRegDec => (
            "i0916",
            "setmaxnreg_dec",
            "register_control.warpgroup.setmaxnreg.decrement",
            "int_nvvm_setmaxnreg_dec_sync_aligned_u32",
            "llvm.nvvm.setmaxnreg.dec.sync.aligned.u32",
            "thread",
            U32,
            "cuda_device::thread::__setmaxnreg_dec",
            "SetMaxNRegDecOp",
            "nvvm.setmaxnreg_dec",
            EMPTY,
            I32,
            DEFAULT_CLASSES,
            SETMAX_PROPERTIES,
            "none",
            true,
            "warpgroup",
            "8.0",
            None,
            TENSOR_MAP_REPLACE_TARGETS,
            SETMAX_SECTION,
            SETMAX_URL,
            "setmaxnreg",
            SETMAX_DEC_MODIFIERS,
            ONE_IMMEDIATE,
            SETMAX_DEC_SELECTIONS,
            "setmaxnreg.dec.sync.aligned.u32 \t$reg_count;",
            SETMAX_PREDICATES,
            "Decreases the executing warpgroup's maximum register allocation to the immediate count.",
        ),
    };

    ExecutionControlRecipe {
        abi_id,
        id,
        operation_key,
        source_record,
        llvm_symbol,
        rust_module,
        rust_arguments,
        compatibility_path,
        dialect_op_type,
        dialect_op_name,
        dialect_operands,
        llvm_arguments,
        imported_classes,
        imported_properties,
        memory,
        convergent,
        execution_scope,
        minimum_ptx,
        minimum_sm,
        targets,
        ptx_isa_section,
        ptx_isa_url,
        mnemonic,
        modifiers,
        operands,
        selection_records,
        selection_asm,
        selection_predicates,
        summary,
    }
}

pub(in crate::resolve) fn execution_control_backend_floor(
    operation: ExecutionControlOperation,
    backend: IntrinsicBackend,
) -> (&'static str, Option<&'static str>) {
    use ExecutionControlOperation::*;

    match (operation, backend) {
        (BarrierCtaSync | BarrierCtaArrive, IntrinsicBackend::LlvmNvptx) => ("6.0", Some("sm_30")),
        (BarrierCtaSync | BarrierCtaArrive, IntrinsicBackend::LibNvvm) => ("6.0", Some("sm_75")),
        (BarrierCtaSyncAligned | BarrierCtaArriveAligned, IntrinsicBackend::LlvmNvptx) => {
            ("3.2", Some("sm_20"))
        }
        (BarrierCtaSyncAligned | BarrierCtaArriveAligned, IntrinsicBackend::LibNvvm) => {
            ("1.0", Some("sm_75"))
        }
        _ => {
            let recipe = execution_control_recipe(operation);
            (recipe.minimum_ptx, recipe.minimum_sm)
        }
    }
}

pub(in crate::resolve) fn validate_execution_control_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let operation = ExecutionControlOperation::from_catalog_id(&policy.id)
        .with_context(|| format!("{} is not a closed execution-control operation", policy.id))?;
    let recipe = execution_control_recipe(operation);
    ensure!(
        policy.family == operation.family()
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.summary == recipe.summary
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == recipe.source_record
            && declaration.llvm_name == recipe.llvm_symbol,
        "{} execution-control identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == recipe.rust_module
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path
                == format!("cuda_intrinsics::{}::{}", recipe.rust_module, recipe.id)
            && policy.compatibility_rust_paths == [recipe.compatibility_path],
        "{} execution-control Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results.is_empty()
            && declaration.arguments == recipe.llvm_arguments
            && declaration.results.is_empty()
            && policy.lowering == "generated_execution_control",
        "{} execution-control carrier or LLVM adapter changed",
        policy.id
    );
    ensure!(
        declaration.classes == recipe.imported_classes
            && declaration.properties == recipe.imported_properties,
        "{} imported execution-control declaration changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == recipe.memory
            && policy.convergent == recipe.convergent
            && policy.execution_scope == recipe.execution_scope,
        "{} execution-control semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.targets == recipe.targets
            && policy.ptx_result == "()"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url
            && policy.expected_ptx.mnemonic == recipe.mnemonic
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == recipe.operands,
        "{} execution-control target or PTX contract changed",
        policy.id
    );
    let valid_route = |backend, mechanism| {
        let (minimum_ptx, minimum_sm) = execution_control_backend_floor(operation, backend);
        policy.backend_lowerings.iter().any(|route| {
            route.backend == backend
                && route.mechanism == mechanism
                && route.minimum_ptx.as_deref() == Some(minimum_ptx)
                && route.minimum_sm.as_deref() == minimum_sm
                && !route.evidence_profile.trim().is_empty()
        })
    };
    ensure!(
        policy.backend_lowerings.len() == 2
            && valid_route(
                IntrinsicBackend::LlvmNvptx,
                BackendLoweringMechanism::TypedNvvm,
            )
            && valid_route(
                IntrinsicBackend::LibNvvm,
                BackendLoweringMechanism::InlinePtx,
            ),
        "{} execution-control backend route changed",
        policy.id
    );
    let expected_selections = recipe
        .selection_records
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let actual_selections = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        declaration.selections.len() == recipe.selection_records.len()
            && actual_selections == expected_selections
            && declaration.selections.iter().all(|selection| {
                selection.asm == recipe.selection_asm
                    && selection.predicates == recipe.selection_predicates
                    && selection.constraints.is_empty()
            }),
        "{} imported execution-control selection family changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "execution-control")?;
    Ok(())
}
