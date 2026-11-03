/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, DotProductAdapter, DotProductOperation, DotProductSignedness,
    ImportedIntrinsic, IntrinsicBackend, OverlayIntrinsic, ReduxAdapter, ReduxOperation,
    ReduxParticipation,
};
use crate::ptx::OperandPattern;
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

pub(in crate::resolve) fn validate_redux_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let redux = policy
        .redux
        .as_ref()
        .with_context(|| format!("{} has no closed redux contract", policy.id))?;
    let recipe = redux_recipe(redux.operation);
    ensure!(
        redux.participation
            == ReduxParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && redux.adapter == ReduxAdapter::MaskValueToSourceMemberMask,
        "{} requests an unsupported redux participation contract or operand adapter",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} redux identity does not match its closed operation recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", recipe.rust_value]
            && policy.rust_result == recipe.rust_value
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::warp::{}", recipe.rust_name)],
        "{} must preserve the unsafe must-use redux raw API and legacy cuda-device compatibility DefPath",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", recipe.value_type]
            && policy.dialect_results == [recipe.value_type]
            && policy.llvm_arguments == [recipe.value_type, "i32"]
            && policy.llvm_results == [recipe.value_type]
            && policy.lowering == "generated_redux",
        "{} is outside the generated two-operand redux recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.ptx_result == recipe.rust_value
            && policy.targets == recipe.targets,
        "{} redux effects, carrier, or target floor disagree with its operation recipe",
        policy.id
    );
    ensure!(
        declaration
            .properties
            .iter()
            .any(|property| property == "IntrConvergent")
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrInaccessibleMemOnly")
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrNoCallback")
            && !declaration.properties.iter().any(|property| matches!(
                property.as_str(),
                "IntrNoMem" | "IntrReadMem" | "IntrWriteMem"
            )),
        "{} redux memory and convergence effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.is_empty()
            && policy.packed_atomic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with redux",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "redux"
            && policy.expected_ptx.modifiers == recipe.ptx_modifiers
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ],
        "{} expected PTX does not match its closed redux operation recipe",
        policy.id
    );
    Ok(())
}

// Pinned LLVM's hasReduxSyncF32(): accel {100} at PTX 8.6, family {100} at
// PTX 8.8. The catalog carries the accel floor with the a/f target union.
pub(in crate::resolve) const REDUX_F32_TARGETS: &str = "sm_100a|sm_100f|sm_103a|sm_103f";

pub(in crate::resolve) struct ReduxRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) rust_value: &'static str,
    pub(in crate::resolve) value_type: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) ptx_modifiers: &'static [&'static str],
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) minimum_sm: Option<&'static str>,
    pub(in crate::resolve) targets: &'static str,
}

pub(in crate::resolve) fn redux_recipe(operation: ReduxOperation) -> ReduxRecipe {
    match operation {
        ReduxOperation::Add => ReduxRecipe {
            id: "redux_sync_add",
            operation_key: "warp.redux.sync.add.wrap32",
            source_record: "int_nvvm_redux_sync_add",
            llvm_symbol: "llvm.nvvm.redux.sync.add",
            rust_name: "redux_sync_add",
            rust_value: "u32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncAddOp",
            dialect_op_name: "nvvm.redux_sync_add",
            ptx_modifiers: &["sync", "add", "s32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::Umin => ReduxRecipe {
            id: "redux_sync_min_u32",
            operation_key: "warp.redux.sync.min.u32",
            source_record: "int_nvvm_redux_sync_umin",
            llvm_symbol: "llvm.nvvm.redux.sync.umin",
            rust_name: "redux_sync_min_u32",
            rust_value: "u32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncUminOp",
            dialect_op_name: "nvvm.redux_sync_umin",
            ptx_modifiers: &["sync", "min", "u32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::Min => ReduxRecipe {
            id: "redux_sync_min_i32",
            operation_key: "warp.redux.sync.min.s32",
            source_record: "int_nvvm_redux_sync_min",
            llvm_symbol: "llvm.nvvm.redux.sync.min",
            rust_name: "redux_sync_min_i32",
            rust_value: "i32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncMinOp",
            dialect_op_name: "nvvm.redux_sync_min",
            ptx_modifiers: &["sync", "min", "s32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::Umax => ReduxRecipe {
            id: "redux_sync_max_u32",
            operation_key: "warp.redux.sync.max.u32",
            source_record: "int_nvvm_redux_sync_umax",
            llvm_symbol: "llvm.nvvm.redux.sync.umax",
            rust_name: "redux_sync_max_u32",
            rust_value: "u32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncUmaxOp",
            dialect_op_name: "nvvm.redux_sync_umax",
            ptx_modifiers: &["sync", "max", "u32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::Max => ReduxRecipe {
            id: "redux_sync_max_i32",
            operation_key: "warp.redux.sync.max.s32",
            source_record: "int_nvvm_redux_sync_max",
            llvm_symbol: "llvm.nvvm.redux.sync.max",
            rust_name: "redux_sync_max_i32",
            rust_value: "i32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncMaxOp",
            dialect_op_name: "nvvm.redux_sync_max",
            ptx_modifiers: &["sync", "max", "s32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::And => ReduxRecipe {
            id: "redux_sync_and",
            operation_key: "warp.redux.sync.and.b32",
            source_record: "int_nvvm_redux_sync_and",
            llvm_symbol: "llvm.nvvm.redux.sync.and",
            rust_name: "redux_sync_and",
            rust_value: "u32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncAndOp",
            dialect_op_name: "nvvm.redux_sync_and",
            ptx_modifiers: &["sync", "and", "b32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::Or => ReduxRecipe {
            id: "redux_sync_or",
            operation_key: "warp.redux.sync.or.b32",
            source_record: "int_nvvm_redux_sync_or",
            llvm_symbol: "llvm.nvvm.redux.sync.or",
            rust_name: "redux_sync_or",
            rust_value: "u32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncOrOp",
            dialect_op_name: "nvvm.redux_sync_or",
            ptx_modifiers: &["sync", "or", "b32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::Xor => ReduxRecipe {
            id: "redux_sync_xor",
            operation_key: "warp.redux.sync.xor.b32",
            source_record: "int_nvvm_redux_sync_xor",
            llvm_symbol: "llvm.nvvm.redux.sync.xor",
            rust_name: "redux_sync_xor",
            rust_value: "u32",
            value_type: "i32",
            dialect_op_type: "ReduxSyncXorOp",
            dialect_op_name: "nvvm.redux_sync_xor",
            ptx_modifiers: &["sync", "xor", "b32"],
            minimum_ptx: "7.0",
            minimum_sm: Some("sm_80"),
            targets: "all",
        },
        ReduxOperation::Fmin => ReduxRecipe {
            id: "redux_sync_min_f32",
            operation_key: "warp.redux.sync.min.f32",
            source_record: "int_nvvm_redux_sync_fmin",
            llvm_symbol: "llvm.nvvm.redux.sync.fmin",
            rust_name: "redux_sync_min_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFminOp",
            dialect_op_name: "nvvm.redux_sync_fmin",
            ptx_modifiers: &["sync", "min", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
        ReduxOperation::FminNan => ReduxRecipe {
            id: "redux_sync_min_nan_f32",
            operation_key: "warp.redux.sync.min.nan.f32",
            source_record: "int_nvvm_redux_sync_fmin_NaN",
            llvm_symbol: "llvm.nvvm.redux.sync.fmin.NaN",
            rust_name: "redux_sync_min_nan_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFminNanOp",
            dialect_op_name: "nvvm.redux_sync_fmin_nan",
            ptx_modifiers: &["sync", "min", "NaN", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
        ReduxOperation::FminAbs => ReduxRecipe {
            id: "redux_sync_min_abs_f32",
            operation_key: "warp.redux.sync.min.abs.f32",
            source_record: "int_nvvm_redux_sync_fmin_abs",
            llvm_symbol: "llvm.nvvm.redux.sync.fmin.abs",
            rust_name: "redux_sync_min_abs_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFminAbsOp",
            dialect_op_name: "nvvm.redux_sync_fmin_abs",
            ptx_modifiers: &["sync", "min", "abs", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
        ReduxOperation::FminAbsNan => ReduxRecipe {
            id: "redux_sync_min_abs_nan_f32",
            operation_key: "warp.redux.sync.min.abs.nan.f32",
            source_record: "int_nvvm_redux_sync_fmin_abs_NaN",
            llvm_symbol: "llvm.nvvm.redux.sync.fmin.abs.NaN",
            rust_name: "redux_sync_min_abs_nan_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFminAbsNanOp",
            dialect_op_name: "nvvm.redux_sync_fmin_abs_nan",
            ptx_modifiers: &["sync", "min", "abs", "NaN", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
        ReduxOperation::Fmax => ReduxRecipe {
            id: "redux_sync_max_f32",
            operation_key: "warp.redux.sync.max.f32",
            source_record: "int_nvvm_redux_sync_fmax",
            llvm_symbol: "llvm.nvvm.redux.sync.fmax",
            rust_name: "redux_sync_max_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFmaxOp",
            dialect_op_name: "nvvm.redux_sync_fmax",
            ptx_modifiers: &["sync", "max", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
        ReduxOperation::FmaxNan => ReduxRecipe {
            id: "redux_sync_max_nan_f32",
            operation_key: "warp.redux.sync.max.nan.f32",
            source_record: "int_nvvm_redux_sync_fmax_NaN",
            llvm_symbol: "llvm.nvvm.redux.sync.fmax.NaN",
            rust_name: "redux_sync_max_nan_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFmaxNanOp",
            dialect_op_name: "nvvm.redux_sync_fmax_nan",
            ptx_modifiers: &["sync", "max", "NaN", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
        ReduxOperation::FmaxAbs => ReduxRecipe {
            id: "redux_sync_max_abs_f32",
            operation_key: "warp.redux.sync.max.abs.f32",
            source_record: "int_nvvm_redux_sync_fmax_abs",
            llvm_symbol: "llvm.nvvm.redux.sync.fmax.abs",
            rust_name: "redux_sync_max_abs_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFmaxAbsOp",
            dialect_op_name: "nvvm.redux_sync_fmax_abs",
            ptx_modifiers: &["sync", "max", "abs", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
        ReduxOperation::FmaxAbsNan => ReduxRecipe {
            id: "redux_sync_max_abs_nan_f32",
            operation_key: "warp.redux.sync.max.abs.nan.f32",
            source_record: "int_nvvm_redux_sync_fmax_abs_NaN",
            llvm_symbol: "llvm.nvvm.redux.sync.fmax.abs.NaN",
            rust_name: "redux_sync_max_abs_nan_f32",
            rust_value: "f32",
            value_type: "f32",
            dialect_op_type: "ReduxSyncFmaxAbsNanOp",
            dialect_op_name: "nvvm.redux_sync_fmax_abs_nan",
            ptx_modifiers: &["sync", "max", "abs", "NaN", "f32"],
            minimum_ptx: "8.6",
            minimum_sm: None,
            targets: REDUX_F32_TARGETS,
        },
    }
}

pub(in crate::resolve) fn validate_dot_product_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let dot_product = policy
        .dot_product
        .as_ref()
        .with_context(|| format!("{} has no closed dot-product contract", policy.id))?;
    let recipe = dot_product_recipe(dot_product.operation, dot_product.signedness);
    ensure!(
        dot_product.adapter == recipe.adapter,
        "{} dot-product source adapter does not match its operation",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} dot-product identity does not match its closed operation recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "dotprod"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", "u32", recipe.rust_value]
            && policy.rust_result == recipe.rust_value
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::dotprod::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::dotprod::{}", recipe.rust_name)],
        "{} must preserve the safe, non-must-use dotprod raw API and legacy cuda-device DefPath",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", "i32", "i32"]
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == ["i32"]
            && policy.lowering == "generated_dotprod",
        "{} is outside the closed three-operand dot-product lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "5.0"
            && policy.minimum_sm.as_deref() == Some("sm_61")
            && policy.ptx_result == recipe.rust_value
            && policy.targets == "all",
        "{} dot-product effects, carrier, or target floor disagree with its operation recipe",
        policy.id
    );
    ensure!(
        declaration
            .classes
            .iter()
            // LLVM 23 migrated the dot-product declarations from
            // NVVMPureIntrinsic to the target-generic PureIntrinsic class.
            .any(|class| class == "NVVMPureIntrinsic" || class == "PureIntrinsic")
            && declaration.properties == recipe.llvm_properties,
        "{} dot-product effects or immediate contract disagree with the imported declaration",
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
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with dotprod",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == recipe.ptx_mnemonic
            && policy.expected_ptx.modifiers == recipe.ptx_modifiers
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ],
        "{} expected PTX does not match its closed dot-product recipe",
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
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed dot-product profile floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

pub(in crate::resolve) struct DotProductRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) rust_value: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) llvm_arguments: &'static [&'static str],
    pub(in crate::resolve) llvm_properties: &'static [&'static str],
    pub(in crate::resolve) adapter: DotProductAdapter,
    pub(in crate::resolve) ptx_mnemonic: &'static str,
    pub(in crate::resolve) ptx_modifiers: &'static [&'static str],
}

pub(in crate::resolve) fn dot_product_recipe(
    operation: DotProductOperation,
    signedness: DotProductSignedness,
) -> DotProductRecipe {
    match (operation, signedness) {
        (DotProductOperation::Dp4a, DotProductSignedness::Signed) => DotProductRecipe {
            id: "dp4a_s32",
            operation_key: "integer.dot_product.dp4a.s32",
            source_record: "int_nvvm_idp4a_s_s",
            llvm_symbol: "llvm.nvvm.idp4a.s.s",
            rust_name: "dp4a_s32",
            rust_value: "i32",
            dialect_op_type: "Dp4aS32Op",
            dialect_op_name: "nvvm.dp4a_s32",
            llvm_arguments: &["i32", "i32", "i32"],
            llvm_properties: &["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"],
            adapter: DotProductAdapter::DirectThreeOperands,
            ptx_mnemonic: "dp4a",
            ptx_modifiers: &["s32", "s32"],
        },
        (DotProductOperation::Dp4a, DotProductSignedness::Unsigned) => DotProductRecipe {
            id: "dp4a_u32",
            operation_key: "integer.dot_product.dp4a.u32",
            source_record: "int_nvvm_idp4a_u_u",
            llvm_symbol: "llvm.nvvm.idp4a.u.u",
            rust_name: "dp4a_u32",
            rust_value: "u32",
            dialect_op_type: "Dp4aU32Op",
            dialect_op_name: "nvvm.dp4a_u32",
            llvm_arguments: &["i32", "i32", "i32"],
            llvm_properties: &["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"],
            adapter: DotProductAdapter::DirectThreeOperands,
            ptx_mnemonic: "dp4a",
            ptx_modifiers: &["u32", "u32"],
        },
        (DotProductOperation::Dp2a, DotProductSignedness::Signed) => DotProductRecipe {
            id: "dp2a_s32",
            operation_key: "integer.dot_product.dp2a.lo.s32",
            source_record: "int_nvvm_idp2a_s_s",
            llvm_symbol: "llvm.nvvm.idp2a.s.s",
            rust_name: "dp2a_s32",
            rust_value: "i32",
            dialect_op_type: "Dp2aS32Op",
            dialect_op_name: "nvvm.dp2a_s32",
            llvm_arguments: &["i32", "i32", "i1", "i32"],
            llvm_properties: &[
                "ImmArg<arg2>",
                "IntrNoCreateUndefOrPoison",
                "IntrNoMem",
                "IntrSpeculatable",
            ],
            adapter: DotProductAdapter::InsertLowHalfFalse,
            ptx_mnemonic: "dp2a",
            ptx_modifiers: &["lo", "s32", "s32"],
        },
        (DotProductOperation::Dp2a, DotProductSignedness::Unsigned) => DotProductRecipe {
            id: "dp2a_u32",
            operation_key: "integer.dot_product.dp2a.lo.u32",
            source_record: "int_nvvm_idp2a_u_u",
            llvm_symbol: "llvm.nvvm.idp2a.u.u",
            rust_name: "dp2a_u32",
            rust_value: "u32",
            dialect_op_type: "Dp2aU32Op",
            dialect_op_name: "nvvm.dp2a_u32",
            llvm_arguments: &["i32", "i32", "i1", "i32"],
            llvm_properties: &[
                "ImmArg<arg2>",
                "IntrNoCreateUndefOrPoison",
                "IntrNoMem",
                "IntrSpeculatable",
            ],
            adapter: DotProductAdapter::InsertLowHalfFalse,
            ptx_mnemonic: "dp2a",
            ptx_modifiers: &["lo", "u32", "u32"],
        },
    }
}
