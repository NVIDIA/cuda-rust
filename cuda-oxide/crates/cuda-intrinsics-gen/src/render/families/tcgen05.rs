/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogHardwareAlternative, CatalogHardwareTarget, CatalogIntrinsic,
    IntrinsicBackend, RuntimeValidation, Tcgen05Adapter, Tcgen05CpGroup, Tcgen05CpMember,
    Tcgen05LdMultiplicity, Tcgen05LdShape, Tcgen05Mma, Tcgen05MmaAlias, Tcgen05MmaBUsage,
    Tcgen05MmaForm, Tcgen05MmaKind, Tcgen05MmaSelectorLayout, Tcgen05Operation,
    Tcgen05SourceContract,
};
use crate::render::common::llvm;
use std::fmt::Write as _;

pub(in crate::render) fn tcgen05_mma_form_name(form: Tcgen05MmaForm) -> &'static str {
    match form {
        Tcgen05MmaForm::Shared => "Shared",
        Tcgen05MmaForm::Tensor => "Tensor",
        Tcgen05MmaForm::TensorAshift => "TensorAshift",
        Tcgen05MmaForm::SpShared => "SpShared",
        Tcgen05MmaForm::SpTensor => "SpTensor",
        Tcgen05MmaForm::SpTensorAshift => "SpTensorAshift",
        Tcgen05MmaForm::WsShared => "WsShared",
        Tcgen05MmaForm::WsSharedZeroColMask => "WsSharedZeroColMask",
        Tcgen05MmaForm::WsSpShared => "WsSpShared",
        Tcgen05MmaForm::WsSpSharedZeroColMask => "WsSpSharedZeroColMask",
        Tcgen05MmaForm::WsSpTensor => "WsSpTensor",
        Tcgen05MmaForm::WsSpTensorZeroColMask => "WsSpTensorZeroColMask",
        Tcgen05MmaForm::WsTensor => "WsTensor",
        Tcgen05MmaForm::WsTensorZeroColMask => "WsTensorZeroColMask",
    }
}

pub(in crate::render) fn tcgen05_mma_form_attr(form: Tcgen05MmaForm) -> String {
    format!("Tcgen05MmaFormAttr::{}", tcgen05_mma_form_name(form))
}

pub(in crate::render) fn tcgen05_mma_kind_attr(kind: Tcgen05MmaKind) -> &'static str {
    match kind {
        Tcgen05MmaKind::F16 => "Tcgen05MmaKindAttr::F16",
        Tcgen05MmaKind::Tf32 => "Tcgen05MmaKindAttr::Tf32",
        Tcgen05MmaKind::F8f6f4 => "Tcgen05MmaKindAttr::F8f6f4",
        Tcgen05MmaKind::I8 => "Tcgen05MmaKindAttr::I8",
    }
}

pub(in crate::render) fn tcgen05_mma_b_usage_attr(usage: Tcgen05MmaBUsage) -> &'static str {
    match usage {
        Tcgen05MmaBUsage::Discard => "Tcgen05MmaBUsageAttr::Discard",
        Tcgen05MmaBUsage::Fill => "Tcgen05MmaBUsageAttr::Fill",
        Tcgen05MmaBUsage::Use => "Tcgen05MmaBUsageAttr::Use",
        Tcgen05MmaBUsage::LastUse => "Tcgen05MmaBUsageAttr::LastUse",
    }
}

pub(in crate::render) fn tcgen05_mma_runtime_parameters(
    mma: &Tcgen05Mma,
) -> Vec<(&'static str, &'static str)> {
    if mma.alias.is_some() && mma.form == Tcgen05MmaForm::WsTensor {
        return vec![
            ("d_tmem", "u32"),
            ("a_tmem", "u32"),
            ("legacy_a_desc", "u64"),
            ("b_desc", "u64"),
            ("idesc", "u32"),
            ("enable_d", "bool"),
        ];
    }
    let mut parameters = match mma.form {
        Tcgen05MmaForm::Shared
        | Tcgen05MmaForm::SpShared
        | Tcgen05MmaForm::WsShared
        | Tcgen05MmaForm::WsSharedZeroColMask
        | Tcgen05MmaForm::WsSpShared
        | Tcgen05MmaForm::WsSpSharedZeroColMask => vec![
            ("d_tmem", "u32"),
            ("a_desc", "u64"),
            ("b_desc", "u64"),
            ("idesc", "u32"),
            ("enable_d", "bool"),
        ],
        Tcgen05MmaForm::Tensor
        | Tcgen05MmaForm::TensorAshift
        | Tcgen05MmaForm::SpTensor
        | Tcgen05MmaForm::SpTensorAshift
        | Tcgen05MmaForm::WsSpTensor
        | Tcgen05MmaForm::WsSpTensorZeroColMask
        | Tcgen05MmaForm::WsTensor
        | Tcgen05MmaForm::WsTensorZeroColMask => vec![
            ("d_tmem", "u32"),
            ("a_tmem", "u32"),
            ("b_desc", "u64"),
            ("idesc", "u32"),
            ("enable_d", "bool"),
        ],
    };
    if matches!(
        mma.form,
        Tcgen05MmaForm::SpShared
            | Tcgen05MmaForm::SpTensor
            | Tcgen05MmaForm::SpTensorAshift
            | Tcgen05MmaForm::WsSpShared
            | Tcgen05MmaForm::WsSpSharedZeroColMask
            | Tcgen05MmaForm::WsSpTensor
            | Tcgen05MmaForm::WsSpTensorZeroColMask
    ) {
        parameters.push(("metadata_tmem", "u32"));
    }
    if matches!(
        mma.form,
        Tcgen05MmaForm::WsSharedZeroColMask
            | Tcgen05MmaForm::WsSpSharedZeroColMask
            | Tcgen05MmaForm::WsSpTensorZeroColMask
            | Tcgen05MmaForm::WsTensorZeroColMask
    ) {
        parameters.push(("zero_column_mask", "u64"));
    }
    parameters
}

pub(in crate::render) fn tcgen05_mma_selector_parameters(
    layout: Tcgen05MmaSelectorLayout,
) -> [(&'static str, &'static str); 3] {
    match layout {
        Tcgen05MmaSelectorLayout::Base { .. } => [
            ("KIND", "kind"),
            ("CTA_GROUP", "cta_group"),
            ("COLLECTOR_A", "collector_a"),
        ],
        Tcgen05MmaSelectorLayout::WarpSpecialized { .. } => [
            ("KIND", "kind"),
            ("B_BUFFER", "b_buffer"),
            ("B_USAGE", "b_usage"),
        ],
    }
}

pub(in crate::render) fn tcgen05_mma_is_ws(form: Tcgen05MmaForm) -> bool {
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

fn tcgen05_mma_is_sparse(form: Tcgen05MmaForm) -> bool {
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

fn tcgen05_mma_is_tensor_a(form: Tcgen05MmaForm) -> bool {
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

fn tcgen05_mma_is_ashift(form: Tcgen05MmaForm) -> bool {
    matches!(
        form,
        Tcgen05MmaForm::TensorAshift | Tcgen05MmaForm::SpTensorAshift
    )
}

fn tcgen05_mma_has_zero_col_mask(form: Tcgen05MmaForm) -> bool {
    matches!(
        form,
        Tcgen05MmaForm::WsSharedZeroColMask
            | Tcgen05MmaForm::WsSpSharedZeroColMask
            | Tcgen05MmaForm::WsSpTensorZeroColMask
            | Tcgen05MmaForm::WsTensorZeroColMask
    )
}

fn tcgen05_mma_kind_name(kind: Tcgen05MmaKind) -> &'static str {
    match kind {
        Tcgen05MmaKind::F16 => "f16",
        Tcgen05MmaKind::Tf32 => "tf32",
        Tcgen05MmaKind::F8f6f4 => "f8f6f4",
        Tcgen05MmaKind::I8 => "i8",
    }
}

fn tcgen05_mma_b_usage_name(usage: Tcgen05MmaBUsage) -> &'static str {
    match usage {
        Tcgen05MmaBUsage::Discard => "discard",
        Tcgen05MmaBUsage::LastUse => "lastuse",
        Tcgen05MmaBUsage::Fill => "fill",
        Tcgen05MmaBUsage::Use => "use",
    }
}

pub(in crate::render) fn tcgen05_mma_inline_asm(
    form: Tcgen05MmaForm,
    kind: Tcgen05MmaKind,
    cta_group: u8,
    collector_a: Option<&str>,
    b_buffer: Option<u8>,
    b_usage: Option<Tcgen05MmaBUsage>,
) -> (String, String) {
    let mut instruction = "tcgen05.mma".to_owned();
    if tcgen05_mma_is_ws(form) {
        instruction.push_str(".ws");
    }
    if tcgen05_mma_is_sparse(form) {
        instruction.push_str(".sp");
    }
    write!(
        instruction,
        ".cta_group::{cta_group}.kind::{}",
        tcgen05_mma_kind_name(kind)
    )
    .unwrap();
    if tcgen05_mma_is_ws(form) {
        write!(
            instruction,
            ".collector::b{}::{}",
            b_buffer.expect("warp-specialized B buffer"),
            tcgen05_mma_b_usage_name(b_usage.expect("warp-specialized B usage"))
        )
        .unwrap();
    } else {
        write!(
            instruction,
            ".collector::a::{}",
            collector_a.expect("base collector A usage")
        )
        .unwrap();
        if tcgen05_mma_is_ashift(form) {
            instruction.push_str(".ashift");
        }
    }

    let a = if tcgen05_mma_is_tensor_a(form) {
        "[$1]"
    } else {
        "$1"
    };
    write!(instruction, " [$0], {a}, $2").unwrap();
    if tcgen05_mma_is_sparse(form) {
        instruction.push_str(", [$5]");
    }
    instruction.push_str(", $3, %enable_pred");
    if tcgen05_mma_has_zero_col_mask(form) {
        instruction.push_str(if tcgen05_mma_is_sparse(form) {
            ", $6"
        } else {
            ", $5"
        });
    }
    instruction.push(';');

    let template =
        format!("{{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; {instruction} }}");
    let mut constraints = vec![
        "r",
        if tcgen05_mma_is_tensor_a(form) {
            "r"
        } else {
            "l"
        },
        "l",
        "r",
        "r",
    ];
    if tcgen05_mma_is_sparse(form) {
        constraints.push("r");
    }
    if tcgen05_mma_has_zero_col_mask(form) {
        constraints.push("l");
    }
    constraints.push("~{memory}");
    (template, constraints.join(","))
}

pub(in crate::render) fn tcgen05_participation_doc(
    operation: Tcgen05Operation,
) -> Option<&'static str> {
    match operation {
        Tcgen05Operation::AllocCg2
        | Tcgen05Operation::DeallocCg2
        | Tcgen05Operation::RelinquishAllocPermitCg2 => Some(
            "One full warp in each peer CTA must execute this instruction; lanes within each warp must execute uniformly with the same operands.",
        ),
        Tcgen05Operation::Alloc
        | Tcgen05Operation::Dealloc
        | Tcgen05Operation::RelinquishAllocPermit
        | Tcgen05Operation::Ld16x256bX8Pure
        | Tcgen05Operation::Ld16x256bPure
        | Tcgen05Operation::LoadWait
        | Tcgen05Operation::StoreWait => {
            Some("One full warp must execute this instruction uniformly with the same operands.")
        }
        Tcgen05Operation::Commit
        | Tcgen05Operation::CommitSharedCluster
        | Tcgen05Operation::CommitMulticast
        | Tcgen05Operation::ShiftDown => Some("One thread in the CTA issues this instruction."),
        Tcgen05Operation::CommitCg2
        | Tcgen05Operation::CommitSharedClusterCg2
        | Tcgen05Operation::CommitMulticastCg2
        | Tcgen05Operation::ShiftDownCg2 => Some(
            "One thread in the CTA pair issues this instruction; the peer CTA must be active and must not have exited.",
        ),
        _ => None,
    }
}

pub(in crate::render) fn tcgen05_is_commit(operation: Tcgen05Operation) -> bool {
    matches!(
        operation,
        Tcgen05Operation::Commit
            | Tcgen05Operation::CommitSharedCluster
            | Tcgen05Operation::CommitMulticast
            | Tcgen05Operation::CommitCg2
            | Tcgen05Operation::CommitSharedClusterCg2
            | Tcgen05Operation::CommitMulticastCg2
    )
}

pub(in crate::render) fn tcgen05_is_multicast_commit(operation: Tcgen05Operation) -> bool {
    matches!(
        operation,
        Tcgen05Operation::CommitMulticast | Tcgen05Operation::CommitMulticastCg2
    )
}

pub(in crate::render) fn tcgen05_is_shift(operation: Tcgen05Operation) -> bool {
    matches!(
        operation,
        Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2
    )
}

fn tcgen05_cp_ptx_suffix(member: Tcgen05CpMember) -> &'static str {
    use Tcgen05CpMember::*;
    match member {
        M128x128bB4x16P64 => "128x128b.b8x16.b4x16_p64",
        M128x128bB6x16P32 => "128x128b.b8x16.b6x16_p32",
        M128x128b => "128x128b",
        M128x256bB4x16P64 => "128x256b.b8x16.b4x16_p64",
        M128x256bB6x16P32 => "128x256b.b8x16.b6x16_p32",
        M32x128bWarpx4B4x16P64 => "32x128b.warpx4.b8x16.b4x16_p64",
        M32x128bWarpx4B6x16P32 => "32x128b.warpx4.b8x16.b6x16_p32",
        M32x128bWarpx4 => "32x128b.warpx4",
        M4x256bB4x16P64 => "4x256b.b8x16.b4x16_p64",
        M4x256bB6x16P32 => "4x256b.b8x16.b6x16_p32",
        M4x256b => "4x256b",
        M64x128bWarpx2Pair0123B4x16P64 => "64x128b.warpx2::01_23.b8x16.b4x16_p64",
        M64x128bWarpx2Pair0123B6x16P32 => "64x128b.warpx2::01_23.b8x16.b6x16_p32",
        M64x128bWarpx2Pair0123 => "64x128b.warpx2::01_23",
        M64x128bWarpx2Pair0213B4x16P64 => "64x128b.warpx2::02_13.b8x16.b4x16_p64",
        M64x128bWarpx2Pair0213B6x16P32 => "64x128b.warpx2::02_13.b8x16.b6x16_p32",
        M64x128bWarpx2Pair0213 => "64x128b.warpx2::02_13",
    }
}

fn tcgen05_ld_shape_label(shape: Tcgen05LdShape) -> &'static str {
    match shape {
        Tcgen05LdShape::M16x32bx2 => "16x32bx2",
        Tcgen05LdShape::M16x64b => "16x64b",
        Tcgen05LdShape::M16x128b => "16x128b",
        Tcgen05LdShape::M16x256b => "16x256b",
        Tcgen05LdShape::M32x32b => "32x32b",
    }
}

fn tcgen05_ld_multiplicity_label(multiplicity: Tcgen05LdMultiplicity) -> &'static str {
    match multiplicity {
        Tcgen05LdMultiplicity::X1 => "x1",
        Tcgen05LdMultiplicity::X2 => "x2",
        Tcgen05LdMultiplicity::X4 => "x4",
        Tcgen05LdMultiplicity::X8 => "x8",
        Tcgen05LdMultiplicity::X16 => "x16",
        Tcgen05LdMultiplicity::X32 => "x32",
        Tcgen05LdMultiplicity::X64 => "x64",
        Tcgen05LdMultiplicity::X128 => "x128",
    }
}

/// The pinned LLVM 23 TableGen dump models tcgen05 ld/st data as one
/// OVERLOADED vector type-variable per register count (LLVM 22 declared
/// concrete `i32`/`vNi32` types). Re-derived here independently of the
/// resolve half (see `tcgen05_overloaded_data_token` in
/// resolve/families/tcgen05/ldst_cp.rs): count 1 -> anonymous_9933, then +4
/// per doubling.
fn tcgen05_overloaded_data_token(register_count: usize) -> String {
    let anonymous = match register_count {
        1 => 9933,
        2 => 9937,
        4 => 9941,
        8 => 9945,
        16 => 9949,
        32 => 9953,
        64 => 9957,
        128 => 9961,
        other => unreachable!("tcgen05 ld/st register count {other} has no imported record"),
    };
    format!("anonymous_{anonymous}")
}

pub(in crate::render) fn tcgen05_ld_register_count(record: &CatalogIntrinsic) -> usize {
    let ld = record
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.ld)
        .expect("generated tcgen05 load identity");
    ld.shape.register_multiplier() * ld.multiplicity.count()
}

pub(in crate::render) fn tcgen05_st_register_count(record: &CatalogIntrinsic) -> usize {
    let st = record
        .tcgen05
        .as_ref()
        .and_then(|tcgen05| tcgen05.st)
        .expect("generated tcgen05 store identity");
    st.shape.register_multiplier() * st.multiplicity.count()
}

fn tcgen05_mma_target_matrix_is_closed(
    target: &crate::model::CatalogTargetRequirement,
    backend: IntrinsicBackend,
    fixed_kind: Option<Tcgen05MmaKind>,
) -> bool {
    let CatalogHardwareTarget::TargetMatrix { contracts } = &target.hardware else {
        return false;
    };
    let expected_kinds: Vec<&str> = fixed_kind.map_or_else(
        || vec!["f16", "tf32", "f8f6f4", "i8"],
        |kind| vec![tcgen05_mma_kind_name(kind)],
    );
    if target.minimum_ptx.to_string() != "8.6" || contracts.len() != expected_kinds.len() {
        return false;
    }
    let common = match backend {
        IntrinsicBackend::LlvmNvptx => vec![
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                "8.6",
            ),
            (CatalogHardwareAlternative::FamilyTarget { sm: 100 }, "8.8"),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                "8.6",
            ),
            (CatalogHardwareAlternative::FamilyTarget { sm: 101 }, "8.8"),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                "8.8",
            ),
            (CatalogHardwareAlternative::FamilyTarget { sm: 103 }, "8.8"),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                "9.0",
            ),
            (CatalogHardwareAlternative::FamilyTarget { sm: 110 }, "9.0"),
        ],
        IntrinsicBackend::LibNvvm => vec![
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                "8.6",
            ),
            (CatalogHardwareAlternative::FamilyTarget { sm: 100 }, "8.8"),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
                "8.8",
            ),
            (CatalogHardwareAlternative::FamilyTarget { sm: 103 }, "8.8"),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                "9.0",
            ),
            (CatalogHardwareAlternative::FamilyTarget { sm: 110 }, "9.0"),
        ],
    };
    let i8 = match backend {
        IntrinsicBackend::LlvmNvptx => vec![
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                "8.6",
            ),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                "8.6",
            ),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                "9.0",
            ),
        ],
        IntrinsicBackend::LibNvvm => vec![
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                "8.6",
            ),
            (
                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                "9.0",
            ),
        ],
    };
    expected_kinds.into_iter().all(|kind| {
        let Some(contract) = contracts.iter().find(|contract| {
            contract.selectors.len() == 1
                && contract.selectors[0].name == "kind"
                && contract.selectors[0].value == kind
        }) else {
            return false;
        };
        let expected = if kind == "i8" { &i8 } else { &common };
        contract.alternatives.len() == expected.len()
            && contract.alternatives.iter().zip(expected).all(
                |(actual, (hardware, minimum_ptx))| {
                    actual.hardware == *hardware && actual.minimum_ptx.to_string() == *minimum_ptx
                },
            )
    })
}

fn tcgen05_mma_render_contract(
    record: &CatalogIntrinsic,
    tcgen05: &crate::model::Tcgen05,
    mma: &Tcgen05Mma,
    llvm_route: &crate::model::CatalogBackendLowering,
    libnvvm_route: &crate::model::CatalogBackendLowering,
) -> bool {
    if tcgen05.operation != Tcgen05Operation::Mma
        || tcgen05.cp.is_some()
        || tcgen05.ld.is_some()
        || tcgen05.st.is_some()
        || tcgen05.source_contract != Tcgen05SourceContract::TablegenSelectionChangesPtx
        || tcgen05.runtime_validation != RuntimeValidation::Unexecuted
        || record.rust.module != "tcgen05"
        || record.rust.safe
        || record.rust.must_use
        || record.rust.result != "()"
        || record.dialect.op_type != "Tcgen05MmaOp"
        || record.dialect.op_name != "nvvm.tcgen05_mma"
        || !record.dialect.results.is_empty()
        || record.lowering != "generated_tcgen05_mma"
        || record.semantics.pure
        || record.semantics.memory != "read_write"
        || !record.semantics.convergent
        || record.semantics.execution_scope != "thread"
        || record.target.minimum_ptx.to_string() != "8.6"
        || record.target.hardware != mma.llvm_target.hardware
        || record.target.targets != "sm_100a|sm_101a|sm_103a|sm_110a"
        || llvm_route.backend != IntrinsicBackend::LlvmNvptx
        || llvm_route.mechanism != BackendLoweringMechanism::InlinePtx
        || llvm_route.target != mma.llvm_target
        || !tcgen05_mma_target_matrix_is_closed(
            &llvm_route.target,
            IntrinsicBackend::LlvmNvptx,
            mma.fixed_selectors.map(|fixed| fixed.kind),
        )
        || libnvvm_route.backend != IntrinsicBackend::LibNvvm
        || libnvvm_route.mechanism != BackendLoweringMechanism::InlinePtx
        || libnvvm_route.target != mma.libnvvm_target
        || !tcgen05_mma_target_matrix_is_closed(
            &libnvvm_route.target,
            IntrinsicBackend::LibNvvm,
            mma.fixed_selectors.map(|fixed| fixed.kind),
        )
    {
        return false;
    }

    let mut expected_operands = vec![
        "i32".to_owned(),
        if tcgen05_mma_is_tensor_a(mma.form) {
            "i32".to_owned()
        } else {
            "i64".to_owned()
        },
        "i64".to_owned(),
        "i32".to_owned(),
        "i1".to_owned(),
    ];
    if tcgen05_mma_is_sparse(mma.form) {
        expected_operands.push("i32".into());
    }
    if tcgen05_mma_has_zero_col_mask(mma.form) {
        expected_operands.push("i64".into());
    }

    let llvm = llvm(record);
    let representative_kind = mma
        .fixed_selectors
        .map_or(Tcgen05MmaKind::F16, |fixed| fixed.kind);
    let representative_buffer = mma.fixed_selectors.map_or(0, |fixed| fixed.b_buffer);
    let representative_usage = mma
        .fixed_selectors
        .map_or(Tcgen05MmaBUsage::Discard, |fixed| fixed.b_usage);
    let representative = if tcgen05_mma_is_ws(mma.form) {
        tcgen05_mma_inline_asm(
            mma.form,
            representative_kind,
            1,
            None,
            Some(representative_buffer),
            Some(representative_usage),
        )
    } else {
        tcgen05_mma_inline_asm(
            mma.form,
            representative_kind,
            1,
            Some("discard"),
            None,
            None,
        )
    };
    let instruction = representative
        .0
        .find("tcgen05.mma")
        .map_or("", |start| &representative.0[start..]);
    let head = instruction.split_whitespace().next().unwrap_or_default();
    let mut components = head.split('.');
    let expected_mnemonic = components.next().unwrap_or_default();
    let expected_modifiers = components.map(str::to_owned).collect::<Vec<_>>();
    let mut expected_ptx_operands = vec![crate::ptx::OperandPattern::Address];
    expected_ptx_operands.push(if tcgen05_mma_is_tensor_a(mma.form) {
        crate::ptx::OperandPattern::Address
    } else {
        crate::ptx::OperandPattern::Register
    });
    expected_ptx_operands.push(crate::ptx::OperandPattern::Register);
    if tcgen05_mma_is_sparse(mma.form) {
        expected_ptx_operands.push(crate::ptx::OperandPattern::Address);
    }
    expected_ptx_operands.extend([
        crate::ptx::OperandPattern::Register,
        crate::ptx::OperandPattern::Exact {
            value: "%enable_pred".into(),
        },
    ]);
    if tcgen05_mma_has_zero_col_mask(mma.form) {
        expected_ptx_operands.push(crate::ptx::OperandPattern::Register);
    }
    if record.dialect.operands != expected_operands
        || !llvm.results.is_empty()
        || record.expected_ptx.mnemonic != expected_mnemonic
        || record.expected_ptx.modifiers != expected_modifiers
        || record.expected_ptx.operands != expected_ptx_operands
    {
        return false;
    }

    match (tcgen05.adapter, mma.alias, mma.fixed_selectors) {
        (Tcgen05Adapter::MmaDirectSelectors, None, None) => {
            let selector_indices = match mma.selector_layout {
                Tcgen05MmaSelectorLayout::Base {
                    kind_argument,
                    cta_group_argument,
                    collector_a_argument,
                    collector_a_upper_exclusive,
                } => {
                    if collector_a_upper_exclusive
                        != if tcgen05_mma_is_ashift(mma.form) {
                            2
                        } else {
                            4
                        }
                    {
                        return false;
                    }
                    [kind_argument, cta_group_argument, collector_a_argument]
                }
                Tcgen05MmaSelectorLayout::WarpSpecialized {
                    kind_argument,
                    b_buffer_argument,
                    b_usage_argument,
                } => [kind_argument, b_buffer_argument, b_usage_argument],
            };
            let first_selector = expected_operands.len() as u8;
            record.rust.arguments.len() == expected_operands.len() + 3
                && record
                    .rust
                    .arguments
                    .ends_with(&["u32".into(), "u32".into(), "u32".into()])
                && llvm.arguments.len() == expected_operands.len() + 3
                && selector_indices == [first_selector, first_selector + 1, first_selector + 2]
                && tcgen05_mma_is_ws(mma.form)
                    == matches!(
                        mma.selector_layout,
                        Tcgen05MmaSelectorLayout::WarpSpecialized { .. }
                    )
        }
        (
            Tcgen05Adapter::MmaDirectSelectors,
            Some(
                Tcgen05MmaAlias::E4m3
                | Tcgen05MmaAlias::E5m2
                | Tcgen05MmaAlias::E2m3
                | Tcgen05MmaAlias::E3m2
                | Tcgen05MmaAlias::E2m1,
            ),
            Some(fixed),
        ) => {
            mma.form == Tcgen05MmaForm::Shared
                && fixed.kind == Tcgen05MmaKind::F8f6f4
                && fixed.b_buffer == 0
                && fixed.b_usage == Tcgen05MmaBUsage::Discard
                && matches!(
                    mma.selector_layout,
                    Tcgen05MmaSelectorLayout::Base {
                        kind_argument: 5,
                        cta_group_argument: 6,
                        collector_a_argument: 7,
                        collector_a_upper_exclusive: 4,
                    }
                )
                && record.rust.arguments == ["u32", "u64", "u64", "u32", "bool"]
                && record.dialect.operands == ["i32", "i64", "i64", "i32", "i1"]
                && llvm.arguments.len() == 8
        }
        (
            Tcgen05Adapter::MmaWsFixedSelectorsDropLegacyADescriptor,
            Some(
                Tcgen05MmaAlias::E4m3
                | Tcgen05MmaAlias::E5m2
                | Tcgen05MmaAlias::E2m3
                | Tcgen05MmaAlias::E3m2
                | Tcgen05MmaAlias::E2m1,
            ),
            Some(fixed),
        ) => {
            mma.form == Tcgen05MmaForm::WsTensor
                && fixed.kind == Tcgen05MmaKind::F8f6f4
                && fixed.b_buffer == 0
                && fixed.b_usage == Tcgen05MmaBUsage::Discard
                && record.rust.arguments == ["u32", "u32", "u64", "u64", "u32", "bool"]
                && record.dialect.operands == ["i32", "i32", "i64", "i32", "i1"]
                && llvm.arguments.len() == 8
        }
        _ => false,
    }
}

pub(in crate::render) fn tcgen05_render_contract(record: &CatalogIntrinsic) -> bool {
    let Some(tcgen05) = &record.tcgen05 else {
        return false;
    };
    let [llvm_route, libnvvm_route] = record.backend_lowerings.as_slice() else {
        return false;
    };
    if let Some(mma) = &tcgen05.mma {
        return tcgen05_mma_render_contract(record, tcgen05, mma, llvm_route, libnvvm_route);
    }
    let llvm_hardware = CatalogHardwareTarget::AnyOf {
        alternatives: vec![
            CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
        ],
    };
    let libnvvm_hardware = CatalogHardwareTarget::AnyOf {
        alternatives: vec![
            CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 103 },
            CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
        ],
    };
    if record.rust.module != "tcgen05"
        || record.rust.must_use != (tcgen05.operation == Tcgen05Operation::Ld)
        || record.lowering != "generated_tcgen05"
        || !record.semantics.convergent
        || record.semantics.execution_scope != tcgen05.operation.execution_scope()
        || record.target.minimum_ptx.to_string() != "8.6"
        || record.target.targets != "sm_100a|sm_101a|sm_103a|sm_110a"
        || llvm_route.backend != IntrinsicBackend::LlvmNvptx
        || llvm_route.mechanism != BackendLoweringMechanism::InlinePtx
        || llvm_route.target.minimum_ptx.to_string() != "8.6"
        || llvm_route.target.hardware != llvm_hardware
        || libnvvm_route.backend != IntrinsicBackend::LibNvvm
        || libnvvm_route.mechanism != BackendLoweringMechanism::InlinePtx
        || libnvvm_route.target.minimum_ptx.to_string() != "8.6"
        || libnvvm_route.target.hardware != libnvvm_hardware
        || tcgen05.runtime_validation != RuntimeValidation::Unexecuted
    {
        return false;
    }
    match tcgen05.cp {
        Some(cp) => {
            let expected_operation = match cp.group {
                Tcgen05CpGroup::Cg1 => Tcgen05Operation::CpSmemToTmem,
                Tcgen05CpGroup::Cg2 => Tcgen05Operation::CpSmemToTmemCg2,
            };
            let group = match cp.group {
                Tcgen05CpGroup::Cg1 => "cta_group::1",
                Tcgen05CpGroup::Cg2 => "cta_group::2",
            };
            let modifiers = std::iter::once("cp")
                .chain(std::iter::once(group))
                .chain(tcgen05_cp_ptx_suffix(cp.member).split('.'))
                .collect::<Vec<_>>();
            if tcgen05.operation != expected_operation
                || tcgen05.ld.is_some()
                || tcgen05.st.is_some()
                || record.semantics.pure
                || record.semantics.memory != "read_write"
                || record.expected_ptx.mnemonic != "tcgen05"
                || record.expected_ptx.modifiers != modifiers
                || record.expected_ptx.operands
                    != [
                        crate::ptx::OperandPattern::Address,
                        crate::ptx::OperandPattern::Register,
                    ]
            {
                return false;
            }
        }
        None if !matches!(
            tcgen05.operation,
            Tcgen05Operation::CpSmemToTmem | Tcgen05Operation::CpSmemToTmemCg2
        ) => {}
        None if matches!(
            record.id.as_str(),
            "tcgen05_cp_smem_to_tmem" | "tcgen05_cp_smem_to_tmem_cg2"
        ) => {}
        None => return false,
    }
    match tcgen05.ld {
        Some(ld) => {
            let count = tcgen05_ld_register_count(record);
            let mut modifiers: Vec<String> = vec![
                "ld".into(),
                "sync".into(),
                "aligned".into(),
                tcgen05_ld_shape_label(ld.shape).into(),
                tcgen05_ld_multiplicity_label(ld.multiplicity).into(),
            ];
            if ld.pack16 {
                modifiers.push("pack::16b".into());
            }
            modifiers.push("b32".into());
            let destination = crate::ptx::OperandPattern::RegisterList { length: count };
            let mut operands = vec![destination, crate::ptx::OperandPattern::Address];
            if ld.shape == Tcgen05LdShape::M16x32bx2 {
                operands.push(crate::ptx::OperandPattern::Immediate);
            }
            if tcgen05.operation != Tcgen05Operation::Ld
                || tcgen05.cp.is_some()
                || tcgen05.st.is_some()
                || record.expected_ptx.mnemonic != "tcgen05"
                || record.expected_ptx.modifiers != modifiers
                || record.expected_ptx.operands != operands
            {
                return false;
            }
        }
        None if tcgen05.operation != Tcgen05Operation::Ld => {}
        None => return false,
    }
    match tcgen05.st {
        Some(st) => {
            let count = tcgen05_st_register_count(record);
            let mut modifiers: Vec<String> = vec![
                "st".into(),
                "sync".into(),
                "aligned".into(),
                tcgen05_ld_shape_label(st.shape).into(),
                tcgen05_ld_multiplicity_label(st.multiplicity).into(),
            ];
            if st.unpack16 {
                modifiers.push("unpack::16b".into());
            }
            modifiers.push("b32".into());
            let data = crate::ptx::OperandPattern::RegisterList { length: count };
            let mut operands = vec![crate::ptx::OperandPattern::Address];
            if st.shape == Tcgen05LdShape::M16x32bx2 {
                operands.push(crate::ptx::OperandPattern::Immediate);
            }
            operands.push(data);
            if tcgen05.operation != Tcgen05Operation::St
                || tcgen05.cp.is_some()
                || tcgen05.ld.is_some()
                || record.expected_ptx.mnemonic != "tcgen05"
                || record.expected_ptx.modifiers != modifiers
                || record.expected_ptx.operands != operands
            {
                return false;
            }
        }
        None if tcgen05.operation != Tcgen05Operation::St => {}
        None => return false,
    }
    let llvm = llvm(record);
    match (tcgen05.operation, tcgen05.adapter) {
        (
            Tcgen05Operation::Alloc | Tcgen05Operation::AllocCg2,
            Tcgen05Adapter::SharedPointerColumnsToVoid,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["*mut u32", "u32"]
                && record.rust.result == "()"
                && record.dialect.operands == ["ptr", "i32"]
                && record.dialect.results.is_empty()
                && llvm.arguments == ["shared_ptr", "i32"]
                && llvm.results.is_empty()
                && tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
        }
        (
            Tcgen05Operation::Dealloc | Tcgen05Operation::DeallocCg2,
            Tcgen05Adapter::TmemAddressColumnsToVoid,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["u32", "u32"]
                && record.rust.result == "()"
                && record.dialect.operands == ["i32", "i32"]
                && llvm.arguments == ["tmem_ptr", "i32"]
                && tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
        }
        (
            Tcgen05Operation::RelinquishAllocPermit
            | Tcgen05Operation::RelinquishAllocPermitCg2
            | Tcgen05Operation::FenceBeforeThreadSync
            | Tcgen05Operation::FenceAfterThreadSync
            | Tcgen05Operation::LoadWait
            | Tcgen05Operation::StoreWait,
            Tcgen05Adapter::NoOperands,
        ) => {
            record.rust.safe
                && record.rust.arguments.is_empty()
                && record.rust.result == "()"
                && record.dialect.operands.is_empty()
                && llvm.arguments.is_empty()
                && tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
        }
        (
            Tcgen05Operation::Commit
            | Tcgen05Operation::CommitCg2
            | Tcgen05Operation::CommitSharedCluster
            | Tcgen05Operation::CommitSharedClusterCg2,
            Tcgen05Adapter::BarrierPointerToVoid,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["*mut u64"]
                && record.rust.result == "()"
                && record.dialect.operands == ["ptr"]
                && llvm.arguments.len() == 1
                && if matches!(
                    tcgen05.operation,
                    Tcgen05Operation::Commit | Tcgen05Operation::CommitCg2
                ) {
                    tcgen05.source_contract == Tcgen05SourceContract::TablegenSelectionChangesPtx
                        && llvm.arguments == ["ptr"]
                } else {
                    tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
                        && llvm.arguments == ["shared_ptr"]
                }
        }
        (
            Tcgen05Operation::MmaWsF16 | Tcgen05Operation::MmaWsBf16 | Tcgen05Operation::MmaWsTf32,
            Tcgen05Adapter::MmaWsDropLegacyADescriptor,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["u32", "u32", "u64", "u64", "u32", "bool"]
                && record.rust.result == "()"
                && record.dialect.operands == ["i32", "i32", "i64", "i64", "i32", "i1"]
                && llvm.arguments
                    == [
                        "tmem_ptr", "tmem_ptr", "i64", "i32", "i1", "i32", "i32", "i32",
                    ]
                && tcgen05.source_contract == Tcgen05SourceContract::TablegenSelectionChangesPtx
                && (tcgen05.operation != Tcgen05Operation::MmaWsBf16
                    || (record.expected_ptx.modifiers.contains(&"kind::f16".into())
                        && !record.expected_ptx.modifiers.contains(&"kind::bf16".into())))
        }
        (
            Tcgen05Operation::MmaF16 | Tcgen05Operation::MmaF16Cg2,
            Tcgen05Adapter::MmaInjectZeroDisableLanes,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["u32", "u64", "u64", "u32", "bool"]
                && record.rust.result == "()"
                && record.dialect.operands == ["i32", "i64", "i64", "i32", "i1"]
                && llvm.arguments.len() == 8
                && tcgen05.source_contract
                    == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
        }
        (
            Tcgen05Operation::CpSmemToTmem | Tcgen05Operation::CpSmemToTmemCg2,
            Tcgen05Adapter::TmemDescriptorToVoid,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["u32", "u64"]
                && record.rust.result == "()"
                && record.dialect.operands == ["i32", "i64"]
                && llvm.arguments == ["tmem_ptr", "i64"]
                && tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
        }
        (Tcgen05Operation::Ld16x256bX8Pure, Tcgen05Adapter::TmemToF32x32) => {
            !record.rust.safe
                && record.rust.arguments == ["u32"]
                && record.rust.result == "[f32; 32]"
                && record.dialect.operands == ["i32"]
                && record.dialect.results == vec!["f32"; 32]
                && llvm.arguments == ["tmem_ptr", "i1"]
                && llvm.results == [tcgen05_overloaded_data_token(32)]
                && tcgen05.source_contract
                    == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
        }
        (Tcgen05Operation::Ld16x256bPure, Tcgen05Adapter::TmemToF32x4) => {
            !record.rust.safe
                && record.rust.arguments == ["u32"]
                && record.rust.result == "[f32; 4]"
                && record.dialect.operands == ["i32"]
                && record.dialect.results == ["f32", "f32", "f32", "f32"]
                && llvm.arguments == ["tmem_ptr", "i1"]
                && llvm.results == [tcgen05_overloaded_data_token(4)]
                && tcgen05.source_contract
                    == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
        }
        (
            Tcgen05Operation::CommitMulticast | Tcgen05Operation::CommitMulticastCg2,
            Tcgen05Adapter::BarrierPointerMaskToVoid,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["*mut u64", "u16"]
                && record.rust.result == "()"
                && record.dialect.operands == ["ptr", "i16"]
                && llvm.arguments == ["shared_ptr", "i16"]
                && tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
        }
        (
            Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2,
            Tcgen05Adapter::TmemAddressToVoid,
        ) => {
            !record.rust.safe
                && record.rust.arguments == ["u32"]
                && record.rust.result == "()"
                && record.dialect.operands == ["i32"]
                && record.dialect.results.is_empty()
                && llvm.arguments == ["tmem_ptr"]
                && llvm.results.is_empty()
                && tcgen05.source_contract == Tcgen05SourceContract::ExactTablegenSelection
        }
        (
            Tcgen05Operation::Ld,
            Tcgen05Adapter::TmemInjectPack16ToU32Registers
            | Tcgen05Adapter::TmemHalfSplitOffsetInjectPack16ToU32Registers,
        ) => {
            let count = tcgen05_ld_register_count(record);
            let has_half_split_offset =
                tcgen05.adapter == Tcgen05Adapter::TmemHalfSplitOffsetInjectPack16ToU32Registers;
            let rust_result = if count == 1 {
                "u32".into()
            } else {
                format!("[u32; {count}]")
            };
            let llvm_result = tcgen05_overloaded_data_token(count);
            !record.rust.safe
                && record.rust.arguments
                    == if has_half_split_offset {
                        vec!["u32", "i64"]
                    } else {
                        vec!["u32"]
                    }
                && record.rust.result == rust_result
                && record.dialect.operands
                    == if has_half_split_offset {
                        vec!["i32", "i64"]
                    } else {
                        vec!["i32"]
                    }
                && record.dialect.results == vec!["i32"; count]
                && llvm.arguments
                    == if has_half_split_offset {
                        vec!["tmem_ptr", "i64", "i1"]
                    } else {
                        vec!["tmem_ptr", "i1"]
                    }
                && llvm.results == [llvm_result]
                && !record.semantics.pure
                && record.semantics.memory == "read"
                && tcgen05.source_contract
                    == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
        }
        (
            Tcgen05Operation::St,
            Tcgen05Adapter::TmemU32RegistersInjectUnpack16ToVoid
            | Tcgen05Adapter::TmemHalfSplitOffsetU32RegistersInjectUnpack16ToVoid,
        ) => {
            let count = tcgen05_st_register_count(record);
            let has_half_split_offset = tcgen05.adapter
                == Tcgen05Adapter::TmemHalfSplitOffsetU32RegistersInjectUnpack16ToVoid;
            let rust_data = if count == 1 {
                "u32".into()
            } else {
                format!("[u32; {count}]")
            };
            let llvm_data = tcgen05_overloaded_data_token(count);
            !record.rust.safe
                && record.rust.arguments
                    == if has_half_split_offset {
                        vec!["u32", "i64", rust_data.as_str()]
                    } else {
                        vec!["u32", rust_data.as_str()]
                    }
                && record.rust.result == "()"
                && record.dialect.operands
                    == std::iter::once("i32")
                        .chain(has_half_split_offset.then_some("i64"))
                        .chain(std::iter::repeat_n("i32", count))
                        .collect::<Vec<_>>()
                && record.dialect.results.is_empty()
                && llvm.arguments
                    == if has_half_split_offset {
                        vec!["tmem_ptr", "i64", llvm_data.as_str(), "i1"]
                    } else {
                        vec!["tmem_ptr", llvm_data.as_str(), "i1"]
                    }
                && llvm.results.is_empty()
                && !record.semantics.pure
                && record.semantics.memory == "write"
                && tcgen05.source_contract
                    == Tcgen05SourceContract::LlvmCustomLoweringWithoutSelection
        }
        _ => false,
    }
}

fn tcgen05_carrier_name(ty: &str) -> &'static str {
    match ty {
        "ptr" => "Ptr",
        "i1" => "I1",
        "i16" => "I16",
        "i32" => "I32",
        "i64" => "I64",
        "f32" => "F32",
        _ => panic!("unsupported tcgen05 dialect carrier {ty}"),
    }
}

pub(in crate::render) fn render_tcgen05_carrier_runs(types: &[String]) -> String {
    let mut runs = Vec::<(&str, usize)>::new();
    for ty in types {
        let carrier = tcgen05_carrier_name(ty);
        if let Some((previous, count)) = runs.last_mut()
            && *previous == carrier
        {
            *count += 1;
        } else {
            runs.push((carrier, 1));
        }
    }

    let mut output = String::from("&[");
    for (index, (carrier, count)) in runs.into_iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "(Tcgen05Carrier::{carrier}, {count})").unwrap();
    }
    output.push(']');
    output
}

pub(in crate::render) fn tcgen05_inline_asm(
    record: &CatalogIntrinsic,
) -> (String, String, Option<usize>) {
    let operation = record.tcgen05.as_ref().expect("tcgen05 contract").operation;
    match operation {
        Tcgen05Operation::Alloc | Tcgen05Operation::AllocCg2 => {
            let group = if operation == Tcgen05Operation::AllocCg2 { 2 } else { 1 };
            (
                format!("{{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::{group}.sync.aligned.shared::cta.b32 [%shared32], $1; }}"),
                "l,r,~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::Dealloc | Tcgen05Operation::DeallocCg2 => {
            let group = if operation == Tcgen05Operation::DeallocCg2 { 2 } else { 1 };
            (
                format!("tcgen05.dealloc.cta_group::{group}.sync.aligned.b32 $0, $1;"),
                "r,r,~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::RelinquishAllocPermit
        | Tcgen05Operation::RelinquishAllocPermitCg2 => {
            let group = if operation == Tcgen05Operation::RelinquishAllocPermitCg2 {
                2
            } else {
                1
            };
            (
                format!("tcgen05.relinquish_alloc_permit.cta_group::{group}.sync.aligned;"),
                "~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::FenceBeforeThreadSync => (
            "tcgen05.fence::before_thread_sync;".into(),
            "~{memory}".into(),
            None,
        ),
        Tcgen05Operation::FenceAfterThreadSync => (
            "tcgen05.fence::after_thread_sync;".into(),
            "~{memory}".into(),
            None,
        ),
        Tcgen05Operation::Commit
        | Tcgen05Operation::CommitSharedCluster
        | Tcgen05Operation::CommitCg2
        | Tcgen05Operation::CommitSharedClusterCg2 => {
            let group = if matches!(
                operation,
                Tcgen05Operation::CommitCg2 | Tcgen05Operation::CommitSharedClusterCg2
            ) {
                2
            } else {
                1
            };
            let shared = if matches!(
                operation,
                Tcgen05Operation::CommitSharedCluster
                    | Tcgen05Operation::CommitSharedClusterCg2
            ) {
                ".shared::cluster"
            } else {
                ""
            };
            (
                format!("tcgen05.commit.cta_group::{group}.mbarrier::arrive::one{shared}.b64 [$0];"),
                "r,~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::MmaWsF16 | Tcgen05Operation::MmaWsBf16 => (
            "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $5, 0; tcgen05.mma.ws.cta_group::1.kind::f16 [$0], [$1], $3, $4, %enable_pred; }".into(),
            "r,r,l,l,r,r,~{memory}".into(),
            None,
        ),
        Tcgen05Operation::MmaWsTf32 => (
            "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $5, 0; tcgen05.mma.ws.cta_group::1.kind::tf32 [$0], [$1], $3, $4, %enable_pred; }".into(),
            "r,r,l,l,r,r,~{memory}".into(),
            None,
        ),
        Tcgen05Operation::MmaF16 | Tcgen05Operation::MmaF16Cg2 => {
            let group = if operation == Tcgen05Operation::MmaF16Cg2 { 2 } else { 1 };
            let zeros = std::iter::repeat_n("%z", if group == 2 { 8 } else { 4 })
                .collect::<Vec<_>>()
                .join(", ");
            (
                format!("{{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::{group}.kind::f16 [$0], $1, $2, $3, {{{zeros}}}, %enable_pred; }}"),
                "r,l,l,r,r,~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::CpSmemToTmem | Tcgen05Operation::CpSmemToTmemCg2 => {
            (
                format!(
                    "{}.{} [$0], $1;",
                    record.expected_ptx.mnemonic,
                    record.expected_ptx.modifiers.join(".")
                ),
                "r,l,~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::Ld16x256bX8Pure | Tcgen05Operation::Ld16x256bPure => {
            let count = if operation == Tcgen05Operation::Ld16x256bX8Pure {
                32
            } else {
                4
            };
            let registers = (0..count)
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(",");
            let constraints = std::iter::repeat_n("=f", count)
                .chain(std::iter::once("r"))
                .collect::<Vec<_>>()
                .join(",");
            (
                format!(
                    "tcgen05.ld.sync.aligned.16x256b.{}.b32 {{{registers}}}, [${count}];",
                    if count == 32 { "x8" } else { "x1" }
                ),
                constraints,
                Some(count),
            )
        }
        Tcgen05Operation::Ld => {
            let count = tcgen05_ld_register_count(record);
            let has_half_split_offset = record
                .tcgen05
                .as_ref()
                .and_then(|tcgen05| tcgen05.ld)
                .is_some_and(|ld| ld.shape == Tcgen05LdShape::M16x32bx2);
            let registers = (0..count)
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(",");
            let constraints = std::iter::repeat_n("=r", count)
                .chain(if has_half_split_offset {
                    vec!["r", "n", "~{memory}"]
                } else {
                    vec!["r", "~{memory}"]
                })
                .collect::<Vec<_>>()
                .join(",");
            (
                format!(
                    "{}.{} {{{registers}}}, [${count}]{};",
                    record.expected_ptx.mnemonic,
                    record.expected_ptx.modifiers.join("."),
                    if has_half_split_offset {
                        format!(", ${}", count + 1)
                    } else {
                        String::new()
                    }
                ),
                constraints,
                Some(count),
            )
        }
        Tcgen05Operation::St => {
            let count = tcgen05_st_register_count(record);
            let has_half_split_offset = record
                .tcgen05
                .as_ref()
                .and_then(|tcgen05| tcgen05.st)
                .is_some_and(|st| st.shape == Tcgen05LdShape::M16x32bx2);
            let first_data = if has_half_split_offset { 2 } else { 1 };
            let registers = (first_data..first_data + count)
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(",");
            let constraints = std::iter::once("r")
                .chain(has_half_split_offset.then_some("n"))
                .chain(std::iter::repeat_n("r", count))
                .chain(std::iter::once("~{memory}"))
                .collect::<Vec<_>>()
                .join(",");
            (
                format!(
                    "{}.{} [$0], {}{{{registers}}};",
                    record.expected_ptx.mnemonic,
                    record.expected_ptx.modifiers.join("."),
                    if has_half_split_offset {
                        "$1, "
                    } else {
                        ""
                    }
                ),
                constraints,
                None,
            )
        }
        Tcgen05Operation::LoadWait => (
            "tcgen05.wait::ld.sync.aligned;".into(),
            "~{memory}".into(),
            None,
        ),
        Tcgen05Operation::StoreWait => (
            "tcgen05.wait::st.sync.aligned;".into(),
            "~{memory}".into(),
            None,
        ),
        Tcgen05Operation::CommitMulticast | Tcgen05Operation::CommitMulticastCg2 => {
            let group = if operation == Tcgen05Operation::CommitMulticastCg2 {
                2
            } else {
                1
            };
            (
                format!("tcgen05.commit.cta_group::{group}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;"),
                "r,h,~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::ShiftDown | Tcgen05Operation::ShiftDownCg2 => {
            let group = if operation == Tcgen05Operation::ShiftDownCg2 {
                2
            } else {
                1
            };
            (
                format!("tcgen05.shift.cta_group::{group}.down [$0];"),
                "r,~{memory}".into(),
                None,
            )
        }
        Tcgen05Operation::Mma => unreachable!("generic MMA uses attribute-driven lowering"),
    }
}
