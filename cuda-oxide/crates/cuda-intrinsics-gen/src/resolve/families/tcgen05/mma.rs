/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogTargetRequirement, ImportedIntrinsic, IntrinsicBackend,
    OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation, TargetContract,
    TargetSelectorBinding, Tcgen05, Tcgen05Adapter, Tcgen05Admission, Tcgen05Mma,
    Tcgen05MmaAdmissionVariant, Tcgen05MmaAlias, Tcgen05MmaBUsage, Tcgen05MmaFixedSelectors,
    Tcgen05MmaForm, Tcgen05MmaKind, Tcgen05MmaSelectorLayout, Tcgen05Operation,
    Tcgen05SourceContract,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Result, ensure};
use std::collections::BTreeSet;

use super::*;
use crate::resolve::guards::*;
use crate::resolve::targets::*;

pub(in crate::resolve) const TCGEN05_MMA_FORMS: [Tcgen05MmaForm; 14] = [
    Tcgen05MmaForm::Shared,
    Tcgen05MmaForm::Tensor,
    Tcgen05MmaForm::TensorAshift,
    Tcgen05MmaForm::SpShared,
    Tcgen05MmaForm::SpTensor,
    Tcgen05MmaForm::SpTensorAshift,
    Tcgen05MmaForm::WsShared,
    Tcgen05MmaForm::WsSharedZeroColMask,
    Tcgen05MmaForm::WsSpShared,
    Tcgen05MmaForm::WsSpSharedZeroColMask,
    Tcgen05MmaForm::WsSpTensor,
    Tcgen05MmaForm::WsSpTensorZeroColMask,
    Tcgen05MmaForm::WsTensor,
    Tcgen05MmaForm::WsTensorZeroColMask,
];
pub(in crate::resolve) const TCGEN05_MMA_ALIASES: [Tcgen05MmaAlias; 5] = [
    Tcgen05MmaAlias::E4m3,
    Tcgen05MmaAlias::E5m2,
    Tcgen05MmaAlias::E2m3,
    Tcgen05MmaAlias::E3m2,
    Tcgen05MmaAlias::E2m1,
];
pub(in crate::resolve) const TCGEN05_MMA_KINDS: [Tcgen05MmaKind; 4] = [
    Tcgen05MmaKind::F16,
    Tcgen05MmaKind::F8f6f4,
    Tcgen05MmaKind::I8,
    Tcgen05MmaKind::Tf32,
];
pub(in crate::resolve) const TCGEN05_MMA_DIALECT_OP_TYPE: &str = "Tcgen05MmaOp";
pub(in crate::resolve) const TCGEN05_MMA_DIALECT_OP_NAME: &str = "nvvm.tcgen05_mma";
pub(in crate::resolve) fn tcgen05_mma_form_name(form: Tcgen05MmaForm) -> &'static str {
    match form {
        Tcgen05MmaForm::Shared => "shared",
        Tcgen05MmaForm::Tensor => "tensor",
        Tcgen05MmaForm::TensorAshift => "tensor_ashift",
        Tcgen05MmaForm::SpShared => "sp_shared",
        Tcgen05MmaForm::SpTensor => "sp_tensor",
        Tcgen05MmaForm::SpTensorAshift => "sp_tensor_ashift",
        Tcgen05MmaForm::WsShared => "ws_shared",
        Tcgen05MmaForm::WsSharedZeroColMask => "ws_shared_zero_col_mask",
        Tcgen05MmaForm::WsSpShared => "ws_sp_shared",
        Tcgen05MmaForm::WsSpSharedZeroColMask => "ws_sp_shared_zero_col_mask",
        Tcgen05MmaForm::WsSpTensor => "ws_sp_tensor",
        Tcgen05MmaForm::WsSpTensorZeroColMask => "ws_sp_tensor_zero_col_mask",
        Tcgen05MmaForm::WsTensor => "ws_tensor",
        Tcgen05MmaForm::WsTensorZeroColMask => "ws_tensor_zero_col_mask",
    }
}

pub(in crate::resolve) fn tcgen05_mma_alias_name(alias: Tcgen05MmaAlias) -> &'static str {
    match alias {
        Tcgen05MmaAlias::E4m3 => "e4m3",
        Tcgen05MmaAlias::E5m2 => "e5m2",
        Tcgen05MmaAlias::E2m3 => "e2m3",
        Tcgen05MmaAlias::E3m2 => "e3m2",
        Tcgen05MmaAlias::E2m1 => "e2m1",
    }
}

pub(in crate::resolve) fn tcgen05_mma_kind_name(kind: Tcgen05MmaKind) -> &'static str {
    match kind {
        Tcgen05MmaKind::F16 => "f16",
        Tcgen05MmaKind::Tf32 => "tf32",
        Tcgen05MmaKind::F8f6f4 => "f8f6f4",
        Tcgen05MmaKind::I8 => "i8",
    }
}

pub(in crate::resolve) fn tcgen05_mma_b_usage_name(usage: Tcgen05MmaBUsage) -> &'static str {
    match usage {
        Tcgen05MmaBUsage::Discard => "discard",
        Tcgen05MmaBUsage::Fill => "fill",
        Tcgen05MmaBUsage::Use => "use",
        Tcgen05MmaBUsage::LastUse => "lastuse",
    }
}

pub(in crate::resolve) fn tcgen05_mma_is_ws(form: Tcgen05MmaForm) -> bool {
    matches!(
        form,
        Tcgen05MmaForm::WsShared
            | Tcgen05MmaForm::WsSharedZeroColMask
            | Tcgen05MmaForm::WsSpShared
            | Tcgen05MmaForm::WsSpSharedZeroColMask
            | Tcgen05MmaForm::WsSpTensor
            | Tcgen05MmaForm::WsSpTensorZeroColMask
            | Tcgen05MmaForm::WsTensor
            | Tcgen05MmaForm::WsTensorZeroColMask
    )
}

pub(in crate::resolve) fn tcgen05_mma_is_sparse(form: Tcgen05MmaForm) -> bool {
    matches!(
        form,
        Tcgen05MmaForm::SpShared
            | Tcgen05MmaForm::SpTensor
            | Tcgen05MmaForm::SpTensorAshift
            | Tcgen05MmaForm::WsSpShared
            | Tcgen05MmaForm::WsSpSharedZeroColMask
            | Tcgen05MmaForm::WsSpTensor
            | Tcgen05MmaForm::WsSpTensorZeroColMask
    )
}

pub(in crate::resolve) fn tcgen05_mma_is_tensor_a(form: Tcgen05MmaForm) -> bool {
    matches!(
        form,
        Tcgen05MmaForm::Tensor
            | Tcgen05MmaForm::TensorAshift
            | Tcgen05MmaForm::SpTensor
            | Tcgen05MmaForm::SpTensorAshift
            | Tcgen05MmaForm::WsSpTensor
            | Tcgen05MmaForm::WsSpTensorZeroColMask
            | Tcgen05MmaForm::WsTensor
            | Tcgen05MmaForm::WsTensorZeroColMask
    )
}

pub(in crate::resolve) fn tcgen05_mma_is_ashift(form: Tcgen05MmaForm) -> bool {
    matches!(
        form,
        Tcgen05MmaForm::TensorAshift | Tcgen05MmaForm::SpTensorAshift
    )
}

pub(in crate::resolve) fn tcgen05_mma_has_zero_col_mask(form: Tcgen05MmaForm) -> bool {
    matches!(
        form,
        Tcgen05MmaForm::WsSharedZeroColMask
            | Tcgen05MmaForm::WsSpSharedZeroColMask
            | Tcgen05MmaForm::WsSpTensorZeroColMask
            | Tcgen05MmaForm::WsTensorZeroColMask
    )
}

pub(in crate::resolve) fn tcgen05_mma_source_record(form: Tcgen05MmaForm) -> String {
    format!("int_nvvm_tcgen05_mma_{}", tcgen05_mma_form_name(form))
}

pub(in crate::resolve) fn tcgen05_mma_llvm_symbol(form: Tcgen05MmaForm) -> String {
    let suffix = match form {
        Tcgen05MmaForm::Shared => "shared",
        Tcgen05MmaForm::Tensor => "tensor",
        Tcgen05MmaForm::TensorAshift => "tensor.ashift",
        Tcgen05MmaForm::SpShared => "sp.shared",
        Tcgen05MmaForm::SpTensor => "sp.tensor",
        Tcgen05MmaForm::SpTensorAshift => "sp.tensor.ashift",
        Tcgen05MmaForm::WsShared => "ws.shared",
        Tcgen05MmaForm::WsSharedZeroColMask => "ws.shared.zero_col_mask",
        Tcgen05MmaForm::WsSpShared => "ws.sp.shared",
        Tcgen05MmaForm::WsSpSharedZeroColMask => "ws.sp.shared.zero_col_mask",
        Tcgen05MmaForm::WsSpTensor => "ws.sp.tensor",
        Tcgen05MmaForm::WsSpTensorZeroColMask => "ws.sp.tensor.zero_col_mask",
        Tcgen05MmaForm::WsTensor => "ws.tensor",
        Tcgen05MmaForm::WsTensorZeroColMask => "ws.tensor.zero_col_mask",
    };
    format!("llvm.nvvm.tcgen05.mma.{suffix}")
}

pub(in crate::resolve) fn tcgen05_mma_llvm_arguments(form: Tcgen05MmaForm) -> Vec<String> {
    let mut arguments = vec![
        "tmem_ptr".into(),
        if tcgen05_mma_is_tensor_a(form) {
            "tmem_ptr".into()
        } else {
            "i64".into()
        },
        "i64".into(),
        "i32".into(),
        "i1".into(),
    ];
    if tcgen05_mma_is_sparse(form) {
        arguments.push("tmem_ptr".into());
    }
    if tcgen05_mma_has_zero_col_mask(form) {
        arguments.push("i64".into());
    }
    arguments.extend(["i32".into(), "i32".into(), "i32".into()]);
    arguments
}

pub(in crate::resolve) fn tcgen05_mma_selector_layout(
    form: Tcgen05MmaForm,
) -> Tcgen05MmaSelectorLayout {
    let first =
        5 + u8::from(tcgen05_mma_is_sparse(form)) + u8::from(tcgen05_mma_has_zero_col_mask(form));
    if tcgen05_mma_is_ws(form) {
        Tcgen05MmaSelectorLayout::WarpSpecialized {
            kind_argument: first,
            b_buffer_argument: first + 1,
            b_usage_argument: first + 2,
        }
    } else {
        Tcgen05MmaSelectorLayout::Base {
            kind_argument: first,
            cta_group_argument: first + 1,
            collector_a_argument: first + 2,
            collector_a_upper_exclusive: if tcgen05_mma_is_ashift(form) { 2 } else { 4 },
        }
    }
}

pub(in crate::resolve) fn tcgen05_mma_imported_properties(form: Tcgen05MmaForm) -> Vec<String> {
    let layout = tcgen05_mma_selector_layout(form);
    let (kind, second, third, second_lower, second_upper, third_upper) = match layout {
        Tcgen05MmaSelectorLayout::Base {
            kind_argument,
            cta_group_argument,
            collector_a_argument,
            collector_a_upper_exclusive,
        } => (
            kind_argument,
            cta_group_argument,
            collector_a_argument,
            1,
            3,
            collector_a_upper_exclusive,
        ),
        Tcgen05MmaSelectorLayout::WarpSpecialized {
            kind_argument,
            b_buffer_argument,
            b_usage_argument,
        } => (kind_argument, b_buffer_argument, b_usage_argument, 0, 4, 4),
    };
    let mut properties = vec![
        format!("ImmArg<arg{kind}>"),
        format!("ImmArg<arg{second}>"),
        format!("ImmArg<arg{third}>"),
        "IntrArgMemOnly".into(),
        format!("Range<arg{kind},0,4>"),
        format!("Range<arg{second},{second_lower},{second_upper}>"),
        format!("Range<arg{third},0,{third_upper}>"),
        "WriteOnly<arg0>".into(),
    ];
    if tcgen05_mma_is_tensor_a(form) {
        properties.push("ReadOnly<arg1>".into());
    }
    if form == Tcgen05MmaForm::Tensor {
        properties.extend([
            "ArgInfo<arg5>".into(),
            "ArgInfo<arg6>".into(),
            "ArgInfo<arg7>".into(),
        ]);
    }
    properties.sort();
    properties
}

pub(in crate::resolve) fn tcgen05_mma_selection_asm(
    form: Tcgen05MmaForm,
    kind: Tcgen05MmaKind,
    cta_group: u8,
    collector_a: Option<&str>,
    b_buffer: Option<u8>,
    b_usage: Option<Tcgen05MmaBUsage>,
) -> String {
    let mut head = "tcgen05.mma".to_owned();
    if tcgen05_mma_is_ws(form) {
        head.push_str(".ws");
    }
    if tcgen05_mma_is_sparse(form) {
        head.push_str(".sp");
    }
    head.push_str(&format!(
        ".cta_group::{cta_group}.kind::{}",
        tcgen05_mma_kind_name(kind)
    ));
    if tcgen05_mma_is_ws(form) {
        head.push_str(&format!(
            ".collector::b{}::{}",
            b_buffer.expect("warp-specialized B buffer"),
            tcgen05_mma_b_usage_name(b_usage.expect("warp-specialized B usage"))
        ));
    } else {
        head.push_str(&format!(
            ".collector::a::{}",
            collector_a.expect("base collector A usage")
        ));
        if tcgen05_mma_is_ashift(form) {
            head.push_str(".ashift");
        }
    }

    let a = if tcgen05_mma_is_tensor_a(form) {
        "[$a]"
    } else {
        "$a"
    };
    let mut operands = format!("[$dtmem], {a}, $b");
    if tcgen05_mma_is_sparse(form) {
        operands.push_str(", [$spmetadata]");
    }
    operands.push_str(", $idesc, $enable_inp_d");
    if tcgen05_mma_has_zero_col_mask(form) {
        operands.push_str(", $zero_col_mask");
    }
    format!("{head} {operands};")
}

/// The DECLARATION-side spelling of one tcgen05 MMA selection string.
///
/// `tcgen05_mma_selection_asm` produces the PTX ISA instruction cuda-oxide
/// emits and ptxas has validated (`.collector::a::<usage>.ashift`). LLVM 23's
/// TableGen selection strings instead spell `.ashift` BEFORE the collector
/// qualifier (`.ashift.collector::a::<usage>`; LLVM 22 matched the ISA
/// order). Use this wrapper wherever an IMPORTED declaration's selection asm
/// is compared, and keep the base function for emission and evidence.
pub(in crate::resolve) fn tcgen05_mma_declaration_asm(
    form: Tcgen05MmaForm,
    kind: Tcgen05MmaKind,
    cta_group: u8,
    collector_a: Option<&str>,
    b_buffer: Option<u8>,
    b_usage: Option<Tcgen05MmaBUsage>,
) -> String {
    let asm = tcgen05_mma_selection_asm(form, kind, cta_group, collector_a, b_buffer, b_usage);
    if tcgen05_mma_is_ws(form) || !tcgen05_mma_is_ashift(form) {
        return asm;
    }
    asm.replacen(".ashift", "", 1)
        .replacen(".collector::a::", ".ashift.collector::a::", 1)
}

pub(in crate::resolve) fn tcgen05_mma_all_selection_asms(form: Tcgen05MmaForm) -> BTreeSet<String> {
    if tcgen05_mma_is_ws(form) {
        return [
            Tcgen05MmaBUsage::Discard,
            Tcgen05MmaBUsage::LastUse,
            Tcgen05MmaBUsage::Fill,
            Tcgen05MmaBUsage::Use,
        ]
        .into_iter()
        .flat_map(|usage| {
            TCGEN05_MMA_KINDS.into_iter().flat_map(move |kind| {
                (0..4).map(move |buffer| {
                    tcgen05_mma_declaration_asm(form, kind, 1, None, Some(buffer), Some(usage))
                })
            })
        })
        .collect();
    }
    TCGEN05_MMA_KINDS
        .into_iter()
        .flat_map(|kind| {
            (1..=2).flat_map(move |group| {
                ["discard", "lastuse", "fill", "use"].map(move |usage| {
                    tcgen05_mma_declaration_asm(form, kind, group, Some(usage), None, None)
                })
            })
        })
        .collect()
}

pub(in crate::resolve) fn tcgen05_mma_valid_selection_asms(
    form: Tcgen05MmaForm,
) -> BTreeSet<String> {
    if !tcgen05_mma_is_ashift(form) {
        return tcgen05_mma_all_selection_asms(form);
    }
    TCGEN05_MMA_KINDS
        .into_iter()
        .flat_map(|kind| {
            (1..=2).flat_map(move |group| {
                ["discard", "lastuse"].map(move |usage| {
                    tcgen05_mma_declaration_asm(form, kind, group, Some(usage), None, None)
                })
            })
        })
        .collect()
}

pub(in crate::resolve) fn tcgen05_mma_target_contract(
    kind: Tcgen05MmaKind,
    alternatives: &[(&str, &str)],
) -> TargetContract {
    TargetContract {
        selectors: vec![TargetSelectorBinding {
            name: "kind".into(),
            value: tcgen05_mma_kind_name(kind).into(),
        }],
        alternatives: alternatives
            .iter()
            .map(
                |(target, minimum_ptx)| crate::model::TargetContractAlternative {
                    target: (*target).into(),
                    minimum_ptx: (*minimum_ptx).into(),
                },
            )
            .collect(),
    }
}

pub(in crate::resolve) fn expected_tcgen05_mma_target_contracts(
    backend: IntrinsicBackend,
) -> Vec<TargetContract> {
    const LLVM_COMMON: &[(&str, &str)] = &[
        ("sm_100a", "8.6"),
        ("sm_100f", "8.8"),
        ("sm_101a", "8.6"),
        ("sm_101f", "8.8"),
        ("sm_103a", "8.8"),
        ("sm_103f", "8.8"),
        ("sm_110a", "9.0"),
        ("sm_110f", "9.0"),
    ];
    const LLVM_I8: &[(&str, &str)] = &[("sm_100a", "8.6"), ("sm_101a", "8.6"), ("sm_110a", "9.0")];
    const LIBNVVM_COMMON: &[(&str, &str)] = &[
        ("sm_100a", "8.6"),
        ("sm_100f", "8.8"),
        ("sm_103a", "8.8"),
        ("sm_103f", "8.8"),
        ("sm_110a", "9.0"),
        ("sm_110f", "9.0"),
    ];
    const LIBNVVM_I8: &[(&str, &str)] = &[("sm_100a", "8.6"), ("sm_110a", "9.0")];
    TCGEN05_MMA_KINDS
        .into_iter()
        .map(|kind| {
            let alternatives = match (backend, kind) {
                (IntrinsicBackend::LlvmNvptx, Tcgen05MmaKind::I8) => LLVM_I8,
                (IntrinsicBackend::LlvmNvptx, _) => LLVM_COMMON,
                (IntrinsicBackend::LibNvvm, Tcgen05MmaKind::I8) => LIBNVVM_I8,
                (IntrinsicBackend::LibNvvm, _) => LIBNVVM_COMMON,
            };
            tcgen05_mma_target_contract(kind, alternatives)
        })
        .collect()
}

pub(in crate::resolve) fn tcgen05_mma_expected_ptx(
    form: Tcgen05MmaForm,
    alias: Option<Tcgen05MmaAlias>,
) -> InstructionPattern {
    let kind = if alias.is_some() {
        Tcgen05MmaKind::F8f6f4
    } else {
        Tcgen05MmaKind::F16
    };
    let asm = if tcgen05_mma_is_ws(form) {
        tcgen05_mma_selection_asm(
            form,
            kind,
            1,
            None,
            Some(0),
            Some(Tcgen05MmaBUsage::Discard),
        )
    } else {
        tcgen05_mma_selection_asm(form, kind, 1, Some("discard"), None, None)
    };
    let head = asm.split_whitespace().next().unwrap();
    let mut components = head.split('.');
    let mnemonic = components.next().unwrap().into();
    let modifiers = components.map(str::to_owned).collect();
    let mut operands = vec![OperandPattern::Address];
    operands.push(if tcgen05_mma_is_tensor_a(form) {
        OperandPattern::Address
    } else {
        OperandPattern::Register
    });
    operands.push(OperandPattern::Register);
    if tcgen05_mma_is_sparse(form) {
        operands.push(OperandPattern::Address);
    }
    operands.extend([
        OperandPattern::Register,
        OperandPattern::Exact {
            value: "%enable_pred".into(),
        },
    ]);
    if tcgen05_mma_has_zero_col_mask(form) {
        operands.push(OperandPattern::Register);
    }
    InstructionPattern {
        mnemonic,
        modifiers,
        operands,
    }
}

pub(in crate::resolve) fn tcgen05_mma_rust_arguments(
    form: Tcgen05MmaForm,
    alias: Option<Tcgen05MmaAlias>,
) -> Vec<String> {
    if alias.is_some() && form == Tcgen05MmaForm::WsTensor {
        return ["u32", "u32", "u64", "u64", "u32", "bool"]
            .into_iter()
            .map(str::to_owned)
            .collect();
    }
    let mut arguments = tcgen05_mma_llvm_arguments(form);
    if alias.is_some() {
        arguments.truncate(arguments.len() - 3);
    }
    arguments
        .iter()
        .map(|argument| match argument.as_str() {
            "tmem_ptr" | "i32" => "u32",
            "i64" => "u64",
            "i1" => "bool",
            _ => unreachable!("closed tcgen05 MMA LLVM type"),
        })
        .map(str::to_owned)
        .collect()
}

pub(in crate::resolve) fn tcgen05_mma_dialect_operands(
    form: Tcgen05MmaForm,
    _alias: Option<Tcgen05MmaAlias>,
) -> Vec<String> {
    let mut arguments = tcgen05_mma_llvm_arguments(form);
    arguments.truncate(arguments.len() - 3);
    arguments
        .iter()
        .map(|argument| match argument.as_str() {
            "tmem_ptr" | "i32" => "i32",
            "i64" => "i64",
            "i1" => "i1",
            _ => unreachable!("closed tcgen05 MMA LLVM type"),
        })
        .map(str::to_owned)
        .collect()
}

pub(in crate::resolve) fn tcgen05_mma_public_id(
    form: Tcgen05MmaForm,
    alias: Option<Tcgen05MmaAlias>,
) -> String {
    alias.map_or_else(
        || format!("tcgen05_mma_{}", tcgen05_mma_form_name(form)),
        |alias| {
            let alias = tcgen05_mma_alias_name(alias);
            if form == Tcgen05MmaForm::WsTensor {
                format!("tcgen05_mma_ws_{alias}")
            } else {
                format!("tcgen05_mma_{alias}")
            }
        },
    )
}

pub(in crate::resolve) fn tcgen05_mma_operation_key(
    form: Tcgen05MmaForm,
    alias: Option<Tcgen05MmaAlias>,
) -> String {
    let base = tcgen05_mma_form_name(form).replace('_', ".");
    alias.map_or_else(
        || format!("tcgen05.mma.{base}"),
        |alias| format!("tcgen05.mma.{base}.{}", tcgen05_mma_alias_name(alias)),
    )
}

pub(in crate::resolve) fn tcgen05_mma_adapter(
    form: Tcgen05MmaForm,
    alias: Option<Tcgen05MmaAlias>,
) -> Tcgen05Adapter {
    match alias {
        Some(_) if form == Tcgen05MmaForm::WsTensor => {
            Tcgen05Adapter::MmaWsFixedSelectorsDropLegacyADescriptor
        }
        _ => Tcgen05Adapter::MmaDirectSelectors,
    }
}

pub(in crate::resolve) fn materialize_tcgen05_mma_variant(
    admission: &Tcgen05Admission,
    variant: &Tcgen05MmaAdmissionVariant,
    llvm_target: &CatalogTargetRequirement,
    libnvvm_target: &CatalogTargetRequirement,
) -> OverlayIntrinsic {
    let form = variant.form;
    let alias = variant.alias;
    let id = tcgen05_mma_public_id(form, alias);
    let operation_key = tcgen05_mma_operation_key(form, alias);
    let llvm_arguments = tcgen05_mma_llvm_arguments(form);
    let rust_arguments = tcgen05_mma_rust_arguments(form, alias);
    let dialect_operands = tcgen05_mma_dialect_operands(form, alias);
    let fixed_selectors = alias.map(|_| Tcgen05MmaFixedSelectors {
        kind: Tcgen05MmaKind::F8f6f4,
        b_buffer: 0,
        b_usage: Tcgen05MmaBUsage::Discard,
    });

    OverlayIntrinsic {
        id: id.clone(),
        abi_id: variant.abi_id.clone(),
        operation_key,
        family: "tcgen05".into(),
        source: None,
        source_record: Some(tcgen05_mma_source_record(form)),
        rust_module: "tcgen05".into(),
        rust_name: id.clone(),
        rust_arguments,
        rust_result: "()".into(),
        safe: false,
        must_use: false,
        safe_allowlist_reason: None,
        public_rust_path: format!("cuda_intrinsics::tcgen05::{id}"),
        compatibility_rust_paths: vec![if alias.is_some() {
            format!("cuda_device::tcgen05::{id}")
        } else {
            format!("cuda_device::tcgen05::__{id}")
        }],
        dialect_op_type: TCGEN05_MMA_DIALECT_OP_TYPE.into(),
        dialect_op_name: TCGEN05_MMA_DIALECT_OP_NAME.into(),
        dialect_operands,
        dialect_results: vec![],
        llvm_symbol: Some(tcgen05_mma_llvm_symbol(form)),
        resolved_llvm_symbol: None,
        llvm_arguments,
        llvm_results: vec![],
        pure: false,
        memory: "read_write".into(),
        // TableGen omits IntrConvergent, but collective MMA must not move across control flow.
        convergent: true,
        execution_scope: "thread".into(),
        minimum_ptx: "8.6".into(),
        minimum_sm: None,
        ptx_result: "()".into(),
        targets: TCGEN05_LLVM_TARGETS.into(),
        ptx_isa_version: "8.6".into(),
        ptx_isa_section: "Tensor Memory tcgen05 instructions".into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory".into(),
        lowering: "generated_tcgen05_mma".into(),
        backend_lowerings: vec![
            OverlayBackendLowering {
                backend: IntrinsicBackend::LlvmNvptx,
                mechanism: BackendLoweringMechanism::InlinePtx,
                evidence_profile: admission
                    .mma_llvm_evidence_profile
                    .as_ref()
                    .expect("validated tcgen05 MMA LLVM evidence profile")
                    .clone(),
                targets: None,
                minimum_ptx: None,
                minimum_sm: None,
            },
            OverlayBackendLowering {
                backend: IntrinsicBackend::LibNvvm,
                mechanism: BackendLoweringMechanism::InlinePtx,
                evidence_profile: admission
                    .mma_libnvvm_evidence_profile
                    .as_ref()
                    .expect("validated tcgen05 MMA libNVVM evidence profile")
                    .clone(),
                targets: None,
                minimum_ptx: None,
                minimum_sm: None,
            },
        ],
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
        mbarrier_extended: None,
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
        tcgen05: Some(Tcgen05 {
            operation: Tcgen05Operation::Mma,
            cp: None,
            ld: None,
            st: None,
            mma: Some(Tcgen05Mma {
                form,
                selector_layout: tcgen05_mma_selector_layout(form),
                fixed_selectors,
                alias,
                llvm_target: llvm_target.clone(),
                libnvvm_target: libnvvm_target.clone(),
            }),
            adapter: tcgen05_mma_adapter(form, alias),
            source_contract: Tcgen05SourceContract::TablegenSelectionChangesPtx,
            runtime_validation: admission.runtime_validation,
        }),
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: tcgen05_mma_expected_ptx(form, alias),
        summary: match (form, alias) {
            (Tcgen05MmaForm::WsTensor, Some(_)) => {
                "Issues one f8f6f4 warp-specialized tensor-memory MMA.".into()
            }
            (Tcgen05MmaForm::Shared, Some(_)) => {
                "Issues one f8f6f4 standard tensor-memory MMA.".into()
            }
            _ => "Issues one selector-controlled tensor-memory MMA.".into(),
        },
    }
}

pub(in crate::resolve) fn validate_tcgen05_mma_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    tcgen05: &Tcgen05,
    mma: &Tcgen05Mma,
) -> Result<()> {
    let form = mma.form;
    let alias = mma.alias;
    ensure!(
        match alias {
            Some(alias) => {
                matches!(form, Tcgen05MmaForm::Shared | Tcgen05MmaForm::WsTensor)
                    && TCGEN05_MMA_ALIASES.contains(&alias)
            }
            None => TCGEN05_MMA_FORMS.contains(&form),
        },
        "{} has an unsupported tcgen05 MMA identity",
        policy.id
    );
    let id = tcgen05_mma_public_id(form, alias);
    let operation_key = tcgen05_mma_operation_key(form, alias);
    let llvm_arguments = tcgen05_mma_llvm_arguments(form);
    let expected_fixed = alias.map(|_| Tcgen05MmaFixedSelectors {
        kind: Tcgen05MmaKind::F8f6f4,
        b_buffer: 0,
        b_usage: Tcgen05MmaBUsage::Discard,
    });
    ensure!(
        policy.id == id
            && policy.operation_key == operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(tcgen05_mma_source_record(form).as_str())
            && policy.llvm_symbol.as_deref() == Some(tcgen05_mma_llvm_symbol(form).as_str())
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == tcgen05_mma_source_record(form)
            && declaration.llvm_name == tcgen05_mma_llvm_symbol(form),
        "{} tcgen05 MMA identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tcgen05"
            && policy.rust_name == id
            && policy.rust_arguments == tcgen05_mma_rust_arguments(form, alias)
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::tcgen05::{id}")
            && policy.compatibility_rust_paths
                == [if alias.is_some() {
                    format!("cuda_device::tcgen05::{id}")
                } else {
                    format!("cuda_device::tcgen05::__{id}")
                }],
        "{} tcgen05 MMA Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == TCGEN05_MMA_DIALECT_OP_TYPE
            && policy.dialect_op_name == TCGEN05_MMA_DIALECT_OP_NAME
            && policy.dialect_operands == tcgen05_mma_dialect_operands(form, alias)
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results.is_empty()
            && declaration.arguments == llvm_arguments
            && declaration.results.is_empty()
            && declaration.classes
                == [
                    "SDPatternOperator",
                    "Intrinsic",
                    "DefaultAttrsIntrinsic",
                    "DefaultAttrsIntrinsicFlags",
                ]
            && declaration.properties == tcgen05_mma_imported_properties(form)
            && policy.lowering == "generated_tcgen05_mma",
        "{} tcgen05 MMA carrier or imported declaration changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "thread"
            && tcgen05.operation == Tcgen05Operation::Mma
            && tcgen05.cp.is_none()
            && tcgen05.ld.is_none()
            && tcgen05.st.is_none()
            && mma.selector_layout == tcgen05_mma_selector_layout(form)
            && mma.fixed_selectors == expected_fixed
            && tcgen05.adapter == tcgen05_mma_adapter(form, alias)
            && tcgen05.source_contract == Tcgen05SourceContract::TablegenSelectionChangesPtx
            && tcgen05.runtime_validation == RuntimeValidation::Unexecuted,
        "{} tcgen05 MMA semantics or selector contract changed",
        policy.id
    );

    let expected_llvm_contracts =
        expected_tcgen05_mma_target_contracts(IntrinsicBackend::LlvmNvptx);
    let expected_libnvvm_contracts =
        expected_tcgen05_mma_target_contracts(IntrinsicBackend::LibNvvm);
    let selected = [TargetSelectorBinding {
        name: "kind".into(),
        value: "f8f6f4".into(),
    }];
    let expected_llvm = if alias.is_some() {
        resolve_target_contract(
            "tcgen05 MMA LLVM alias",
            &selected,
            &expected_llvm_contracts,
        )?
    } else {
        resolve_target_contracts("tcgen05 MMA LLVM", &expected_llvm_contracts)?
    };
    let expected_libnvvm = if alias.is_some() {
        resolve_target_contract(
            "tcgen05 MMA libNVVM alias",
            &selected,
            &expected_libnvvm_contracts,
        )?
    } else {
        resolve_target_contracts("tcgen05 MMA libNVVM", &expected_libnvvm_contracts)?
    };
    ensure!(
        mma.llvm_target == expected_llvm
            && mma.libnvvm_target == expected_libnvvm
            && policy.minimum_ptx == "8.6"
            && policy.minimum_sm.is_none()
            && policy.targets == TCGEN05_LLVM_TARGETS
            && policy.ptx_result == "()"
            && policy.ptx_isa_version == "8.6"
            && policy.expected_ptx == tcgen05_mma_expected_ptx(form, alias),
        "{} tcgen05 MMA target or PTX contract changed",
        policy.id
    );
    let backend_pairs = policy
        .backend_lowerings
        .iter()
        .map(|route| (route.backend, route.mechanism))
        .collect::<BTreeSet<_>>();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ])
            && policy.backend_lowerings.iter().all(|route| {
                !route.evidence_profile.trim().is_empty()
                    && route.targets.is_none()
                    && route.minimum_ptx.is_none()
                    && route.minimum_sm.is_none()
            }),
        "{} tcgen05 MMA backend routes changed",
        policy.id
    );

    let expected_all = tcgen05_mma_all_selection_asms(form);
    let expected_valid = if alias.is_some() {
        BTreeSet::from([if tcgen05_mma_is_ws(form) {
            tcgen05_mma_declaration_asm(
                form,
                Tcgen05MmaKind::F8f6f4,
                1,
                None,
                Some(0),
                Some(Tcgen05MmaBUsage::Discard),
            )
        } else {
            tcgen05_mma_declaration_asm(
                form,
                Tcgen05MmaKind::F8f6f4,
                1,
                Some("discard"),
                None,
                None,
            )
        }])
    } else {
        tcgen05_mma_valid_selection_asms(form)
    };
    let mut source_records = BTreeSet::new();
    let actual_all = declaration
        .selections
        .iter()
        .map(|selection| {
            ensure!(
                !selection.source_record.is_empty()
                    && source_records.insert(selection.source_record.as_str())
                    && selection.constraints.is_empty(),
                "{} tcgen05 MMA selection provenance changed",
                policy.id
            );
            let predicate = if selection.asm.contains(".kind::i8.") {
                "Subtarget->hasTcgen05MMAI8Kind()"
            } else {
                "Subtarget->hasTcgen05InstSupport()"
            };
            ensure!(
                selection.predicates == [predicate],
                "{} tcgen05 MMA selection predicate changed",
                policy.id
            );
            Ok(selection.asm.clone())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(
        actual_all == expected_all
            && declaration.selections.len() == expected_all.len()
            && expected_valid.is_subset(&expected_all)
            && expected_valid.len()
                == if alias.is_some() {
                    1
                } else if tcgen05_mma_is_ws(form) {
                    64
                } else if tcgen05_mma_is_ashift(form) {
                    16
                } else {
                    32
                },
        "{} tcgen05 MMA selection matrix changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "tcgen05 MMA")?;
    Ok(())
}
