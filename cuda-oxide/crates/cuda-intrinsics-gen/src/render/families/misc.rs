/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogFile, CatalogIntrinsic, ClcOperation, ClusterBarrierMode,
    IntrinsicBackend, SpecialRegisterObservation, SpecialRegisterOutputConstraint,
    SpecialRegisterPtxType, WgmmaControlMode,
};
use crate::render::families::{execution_controls, wgmma_controls};
use std::fmt::Write as _;

pub(in crate::render) fn cluster_barrier_attr(record: &CatalogIntrinsic) -> &'static str {
    match record
        .cluster_barrier
        .as_ref()
        .expect("cluster-barrier contract")
        .mode
    {
        ClusterBarrierMode::Arrive => "ClusterBarrierModeAttr::Arrive",
        ClusterBarrierMode::ArriveAligned => "ClusterBarrierModeAttr::ArriveAligned",
        ClusterBarrierMode::ArriveRelaxed => "ClusterBarrierModeAttr::ArriveRelaxed",
        ClusterBarrierMode::ArriveRelaxedAligned => "ClusterBarrierModeAttr::ArriveRelaxedAligned",
        ClusterBarrierMode::Wait => "ClusterBarrierModeAttr::Wait",
        ClusterBarrierMode::WaitAligned => "ClusterBarrierModeAttr::WaitAligned",
    }
}

pub(in crate::render) fn cluster_barrier_template(record: &CatalogIntrinsic) -> String {
    format!(
        "{}.{};",
        record.expected_ptx.mnemonic,
        record.expected_ptx.modifiers.join(".")
    )
}

#[derive(Clone, Copy)]
pub(in crate::render) enum ClcSafetyArgNames {
    RawAbi,
    Compatibility,
}

fn clc_safety_lines(operation: ClcOperation, names: ClcSafetyArgNames) -> Vec<String> {
    let (response, mbar, resp_lo, resp_hi) = match names {
        ClcSafetyArgNames::RawAbi => ("`_arg0`", "`_arg1`", "`_arg0`", "`_arg1`"),
        ClcSafetyArgNames::Compatibility => ("`response`", "`mbar`", "`resp_lo`", "`resp_hi`"),
    };
    match operation {
        ClcOperation::TryCancel => vec![
            format!(
                "{response} must designate a writable, 16-byte-aligned 16-byte region in `shared::cta` memory, and {mbar} must designate a valid, aligned shared-memory mbarrier."
            ),
            format!(
                "The calling CTA must issue `arrive_expect_tx` for {mbar} with 16 bytes before this request."
            ),
            "This CTA must not have observed a prior CLC try-cancel request fail.".into(),
        ],
        ClcOperation::TryCancelMulticast => vec![
            format!(
                "{response} must designate a writable, 16-byte-aligned 16-byte region in `shared::cta` memory, and {mbar} must designate a valid, aligned shared-memory mbarrier."
            ),
            "Each CTA must issue `arrive_expect_tx` for its own corresponding mbarrier with 16 bytes before this request.".to_string(),
            "This CTA must not have observed a prior CLC try-cancel request fail.".into(),
            "The response is written to every CTA's shared-memory window and signals every CTA's mbarrier.".into(),
            "Every CTA in the cluster must still be active.".into(),
        ],
        ClcOperation::QueryIsCanceled => vec![format!(
            "{resp_lo} and {resp_hi} must be the two halves of an opaque response observed complete from a prior CLC try-cancel request."
        )],
        ClcOperation::QueryGetFirstCtaidX
        | ClcOperation::QueryGetFirstCtaidY
        | ClcOperation::QueryGetFirstCtaidZ => vec![
            format!(
                "{resp_lo} and {resp_hi} must be the two halves of an opaque response observed complete from a prior CLC try-cancel request."
            ),
            "The result is meaningful only when `clc_query_is_canceled` returned one for this response.".into(),
        ],
    }
}

pub(in crate::render) fn render_clc_safety_lines(
    output: &mut String,
    operation: ClcOperation,
    names: ClcSafetyArgNames,
) {
    for line in clc_safety_lines(operation, names) {
        writeln!(output, "/// {line}").unwrap();
    }
}

pub(in crate::render) fn execution_control_family<'a>(
    catalog: &'a CatalogFile,
    family: &'a str,
) -> impl Iterator<Item = &'a CatalogIntrinsic> {
    execution_controls(catalog).filter(move |record| record.family == family)
}

pub(in crate::render) fn wgmma_control(
    catalog: &CatalogFile,
    mode: WgmmaControlMode,
) -> &CatalogIntrinsic {
    wgmma_controls(catalog)
        .find(|record| {
            record
                .wgmma_control
                .as_ref()
                .is_some_and(|control| control.mode == mode)
        })
        .expect("complete WGMMA-control family")
}

pub(in crate::render) fn wgmma_control_template(record: &CatalogIntrinsic) -> String {
    let operand = if record
        .wgmma_control
        .as_ref()
        .is_some_and(|control| control.mode == WgmmaControlMode::WaitGroup)
    {
        " $0"
    } else {
        ""
    };
    format!(
        "{}.{}{operand};",
        record.expected_ptx.mnemonic,
        record.expected_ptx.modifiers.join(".")
    )
}

pub(in crate::render) fn threadfence_ptx_level(record: &CatalogIntrinsic) -> Option<&'static str> {
    match record.id.as_str() {
        "threadfence_block" => Some("cta"),
        "threadfence" => Some("gl"),
        "threadfence_system" => Some("sys"),
        _ => None,
    }
}

fn special_register_ptx_type(record: &CatalogIntrinsic) -> &'static str {
    match record
        .special_register
        .as_ref()
        .expect("special-register record")
        .ptx_type
    {
        SpecialRegisterPtxType::B32 => "b32",
        SpecialRegisterPtxType::U32 => "u32",
        SpecialRegisterPtxType::U64 => "u64",
    }
}

pub(in crate::render) fn special_register_output_constraint(
    record: &CatalogIntrinsic,
) -> &'static str {
    match record
        .special_register
        .as_ref()
        .expect("special-register record")
        .output_constraint
    {
        SpecialRegisterOutputConstraint::Register32 => "=r",
        SpecialRegisterOutputConstraint::Register64 => "=l",
    }
}

pub(in crate::render) fn special_register_inline_template(record: &CatalogIntrinsic) -> String {
    let register = match record.expected_ptx.operands.as_slice() {
        [_, crate::ptx::OperandPattern::Exact { value }] => value,
        _ => unreachable!("closed special-register PTX operands"),
    };
    format!("mov.{} $0, {register};", special_register_ptx_type(record))
}

pub(in crate::render) fn special_register_asm_kind(record: &CatalogIntrinsic) -> &'static str {
    match record
        .special_register
        .as_ref()
        .expect("special-register record")
        .observation
    {
        SpecialRegisterObservation::StablePure => "AsmKind::Pure",
        SpecialRegisterObservation::VolatileObservation => "AsmKind::SideEffect",
    }
}

pub(in crate::render) fn special_register_backend_mechanism(
    record: &CatalogIntrinsic,
    backend: IntrinsicBackend,
) -> BackendLoweringMechanism {
    record
        .backend_lowerings
        .iter()
        .find(|route| route.backend == backend)
        .expect("special-register backend route")
        .mechanism
}
