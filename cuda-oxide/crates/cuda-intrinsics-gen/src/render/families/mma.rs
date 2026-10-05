/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    CatalogIntrinsic, LdmatrixElement, LdmatrixLayout, LdmatrixMultiplicity, LdmatrixShape,
    LdmatrixStateSpace, RegisterMmaAccumulator, RegisterMmaAdapter, RegisterMmaElement,
    RegisterMmaKind, RegisterMmaLayout, RegisterMmaOperation, RegisterMmaOverflow,
    RegisterMmaShape, SparseMma, SparseMmaAccumulator, SparseMmaAdapter, SparseMmaElement,
    SparseMmaLayout, SparseMmaMetadata, SparseMmaOverflow, SparseMmaSelector, SparseMmaShape,
    StmatrixLayout, StmatrixMultiplicity,
};
use std::fmt::Write as _;

pub(in crate::render) fn is_blackwell_ldmatrix(record: &CatalogIntrinsic) -> bool {
    record
        .ldmatrix
        .as_ref()
        .is_some_and(|ldmatrix| ldmatrix.variant.shape != LdmatrixShape::M8n8)
}

pub(in crate::render) const BLACKWELL_LDMATRIX_EFFECTIVE_FLOORS: &str = "sm_100a PTX 8.6, sm_100f PTX 8.8, sm_103a PTX 8.8, sm_103f PTX 8.8, sm_110a PTX 9.0, sm_110f PTX 9.0, sm_120a PTX 8.7, sm_120f PTX 8.8, sm_121a PTX 8.8, sm_121f PTX 8.8";

pub(in crate::render) fn ldmatrix_compat_op(
    record: &CatalogIntrinsic,
) -> Option<(&'static str, &'static str)> {
    match record.id.as_str() {
        "ldmatrix_m8n8_x1_b16" => Some(("LdmatrixX1Op", "nvvm.ldmatrix_x1")),
        "ldmatrix_m8n8_x1_trans_b16" => Some(("LdmatrixX1TransOp", "nvvm.ldmatrix_x1_trans")),
        "ldmatrix_m8n8_x2_b16" => Some(("LdmatrixX2Op", "nvvm.ldmatrix_x2")),
        "ldmatrix_m8n8_x2_trans_b16" => Some(("LdmatrixX2TransOp", "nvvm.ldmatrix_x2_trans")),
        "ldmatrix_m8n8_x4_b16" => Some(("LdmatrixX4Op", "nvvm.ldmatrix_x4")),
        "ldmatrix_m8n8_x4_trans_b16" => Some(("LdmatrixX4TransOp", "nvvm.ldmatrix_x4_trans")),
        _ => None,
    }
}

pub(in crate::render) fn stmatrix_variant(
    record: &CatalogIntrinsic,
) -> Option<(StmatrixMultiplicity, StmatrixLayout)> {
    match record.id.as_str() {
        "stmatrix_m8n8_x2_b16" => Some((StmatrixMultiplicity::X2, StmatrixLayout::Normal)),
        "stmatrix_m8n8_x2_trans_b16" => {
            Some((StmatrixMultiplicity::X2, StmatrixLayout::Transposed))
        }
        "stmatrix_m8n8_x4_b16" => Some((StmatrixMultiplicity::X4, StmatrixLayout::Normal)),
        "stmatrix_m8n8_x4_trans_b16" => {
            Some((StmatrixMultiplicity::X4, StmatrixLayout::Transposed))
        }
        _ => None,
    }
}

pub(in crate::render) fn stmatrix_compatibility_name(record: &CatalogIntrinsic) -> &'static str {
    match record.id.as_str() {
        "stmatrix_m8n8_x2_b16" => "stmatrix_m8n8_x2",
        "stmatrix_m8n8_x2_trans_b16" => "stmatrix_m8n8_x2_trans",
        "stmatrix_m8n8_x4_b16" => "stmatrix_m8n8_x4",
        "stmatrix_m8n8_x4_trans_b16" => "stmatrix_m8n8_x4_trans",
        _ => panic!("unknown stmatrix record {}", record.id),
    }
}

pub(in crate::render) fn movmatrix_template(record: &CatalogIntrinsic) -> String {
    format!(
        "{}.{} $0, $1;",
        record.expected_ptx.mnemonic,
        record.expected_ptx.modifiers.join(".")
    )
}

pub(in crate::render) fn ldmatrix_attr_variants(
    record: &CatalogIntrinsic,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    let variant = &record.ldmatrix.as_ref().expect("ldmatrix record").variant;
    let shape = match variant.shape {
        LdmatrixShape::M8n8 => "LdmatrixShapeAttr::M8n8",
        LdmatrixShape::M8n16 => "LdmatrixShapeAttr::M8n16",
        LdmatrixShape::M16n16 => "LdmatrixShapeAttr::M16n16",
    };
    let multiplicity = match variant.multiplicity {
        LdmatrixMultiplicity::X1 => "LdmatrixMultiplicityAttr::X1",
        LdmatrixMultiplicity::X2 => "LdmatrixMultiplicityAttr::X2",
        LdmatrixMultiplicity::X4 => "LdmatrixMultiplicityAttr::X4",
    };
    let layout = match variant.layout {
        LdmatrixLayout::Normal => "LdmatrixLayoutAttr::Normal",
        LdmatrixLayout::Transposed => "LdmatrixLayoutAttr::Transposed",
    };
    let element = match variant.element {
        LdmatrixElement::B16 => "LdmatrixElementAttr::B16",
        LdmatrixElement::B8 => "LdmatrixElementAttr::B8",
        LdmatrixElement::B8x16B4x16P64 => "LdmatrixElementAttr::B8x16B4x16P64",
        LdmatrixElement::B8x16B6x16P32 => "LdmatrixElementAttr::B8x16B6x16P32",
    };
    let state_space = match variant.state_space {
        LdmatrixStateSpace::Shared => "LdmatrixStateSpaceAttr::Shared",
    };
    (shape, multiplicity, layout, element, state_space)
}

pub(in crate::render) fn register_mma_effective_kind(record: &CatalogIntrinsic) -> RegisterMmaKind {
    let mma = record.register_mma.as_ref().expect("register-MMA record");
    mma.kind.unwrap_or_else(|| {
        if record
            .llvm
            .as_ref()
            .is_some_and(|llvm| llvm.symbol.contains(".kind.f8f6f4."))
        {
            RegisterMmaKind::F8f6f4
        } else {
            RegisterMmaKind::Standard
        }
    })
}

pub(in crate::render) fn register_mma_attr_variants(
    record: &CatalogIntrinsic,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    let mma = record.register_mma.as_ref().expect("register-MMA record");
    let shape = match mma.shape {
        RegisterMmaShape::M8n8k4 => "RegisterMmaShapeAttr::M8n8k4",
        RegisterMmaShape::M8n8k16 => "RegisterMmaShapeAttr::M8n8k16",
        RegisterMmaShape::M8n8k32 => "RegisterMmaShapeAttr::M8n8k32",
        RegisterMmaShape::M8n8k128 => "RegisterMmaShapeAttr::M8n8k128",
        RegisterMmaShape::M16n8k4 => "RegisterMmaShapeAttr::M16n8k4",
        RegisterMmaShape::M16n8k8 => "RegisterMmaShapeAttr::M16n8k8",
        RegisterMmaShape::M16n8k16 => "RegisterMmaShapeAttr::M16n8k16",
        RegisterMmaShape::M16n8k32 => "RegisterMmaShapeAttr::M16n8k32",
        RegisterMmaShape::M16n8k64 => "RegisterMmaShapeAttr::M16n8k64",
        RegisterMmaShape::M16n8k128 => "RegisterMmaShapeAttr::M16n8k128",
        RegisterMmaShape::M16n8k256 => "RegisterMmaShapeAttr::M16n8k256",
    };
    let operation = match mma.operation {
        RegisterMmaOperation::Multiply => "RegisterMmaOperationAttr::Multiply",
        RegisterMmaOperation::AndPopc => "RegisterMmaOperationAttr::AndPopc",
        RegisterMmaOperation::XorPopc => "RegisterMmaOperationAttr::XorPopc",
    };
    let kind = match register_mma_effective_kind(record) {
        RegisterMmaKind::Standard => "RegisterMmaKindAttr::Standard",
        RegisterMmaKind::F8f6f4 => "RegisterMmaKindAttr::F8f6f4",
        RegisterMmaKind::Mxf8f6f4 => "RegisterMmaKindAttr::Mxf8f6f4",
    };
    let accumulator = match mma.accumulator {
        RegisterMmaAccumulator::F16 => "RegisterMmaAccumulatorAttr::F16",
        RegisterMmaAccumulator::F32 => "RegisterMmaAccumulatorAttr::F32",
        RegisterMmaAccumulator::F64 => "RegisterMmaAccumulatorAttr::F64",
        RegisterMmaAccumulator::S32 => "RegisterMmaAccumulatorAttr::S32",
    };
    let element = |element| match element {
        RegisterMmaElement::Bf16 => "RegisterMmaElementAttr::Bf16",
        RegisterMmaElement::E2m1 => "RegisterMmaElementAttr::E2m1",
        RegisterMmaElement::E2m3 => "RegisterMmaElementAttr::E2m3",
        RegisterMmaElement::E3m2 => "RegisterMmaElementAttr::E3m2",
        RegisterMmaElement::E4m3 => "RegisterMmaElementAttr::E4m3",
        RegisterMmaElement::E5m2 => "RegisterMmaElementAttr::E5m2",
        RegisterMmaElement::F16 => "RegisterMmaElementAttr::F16",
        RegisterMmaElement::Tf32 => "RegisterMmaElementAttr::Tf32",
        RegisterMmaElement::F64 => "RegisterMmaElementAttr::F64",
        RegisterMmaElement::B1 => "RegisterMmaElementAttr::B1",
        RegisterMmaElement::S4 => "RegisterMmaElementAttr::S4",
        RegisterMmaElement::U4 => "RegisterMmaElementAttr::U4",
        RegisterMmaElement::S8 => "RegisterMmaElementAttr::S8",
        RegisterMmaElement::U8 => "RegisterMmaElementAttr::U8",
    };
    let layout = |layout| match layout {
        RegisterMmaLayout::Row => "RegisterMmaLayoutAttr::Row",
        RegisterMmaLayout::Col => "RegisterMmaLayoutAttr::Col",
    };
    let overflow = match mma.overflow {
        RegisterMmaOverflow::NotApplicable => "RegisterMmaOverflowAttr::NotApplicable",
        RegisterMmaOverflow::Wrapping => "RegisterMmaOverflowAttr::Wrapping",
        RegisterMmaOverflow::Satfinite => "RegisterMmaOverflowAttr::Satfinite",
    };
    (
        shape,
        operation,
        kind,
        accumulator,
        element(mma.a_element),
        element(mma.b_element),
        layout(mma.a_layout),
        layout(mma.b_layout),
        overflow,
    )
}

pub(in crate::render) fn register_mma_fragment_counts(
    record: &CatalogIntrinsic,
) -> (usize, usize, usize, usize) {
    match record.register_mma.as_ref().unwrap().adapter {
        RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32 => (2, 2, 1, 2),
        RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32 => (2, 4, 2, 2),
        RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32 => (4, 2, 1, 4),
        RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32 | RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32 => {
            (4, 4, 2, 4)
        }
        RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 => (4, 4, 2, 4),
        RegisterMmaAdapter::C2F64A1F64B1F64ToD2F64 | RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32 => {
            (2, 1, 1, 2)
        }
        RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32 => (4, 2, 1, 4),
    }
}

pub(in crate::render) fn register_mma_extra_operand_count(record: &CatalogIntrinsic) -> usize {
    usize::from(
        record.register_mma.as_ref().unwrap().adapter
            == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32,
    ) * 6
}

pub(in crate::render) fn expected_ptx_head(record: &CatalogIntrinsic) -> String {
    let mut head = record.expected_ptx.mnemonic.clone();
    for modifier in &record.expected_ptx.modifiers {
        write!(head, ".{modifier}").unwrap();
    }
    head
}

fn register_list(first: usize, count: usize) -> String {
    format!(
        "{{{}}}",
        (first..first + count)
            .map(|index| format!("${index}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(in crate::render) fn register_mma_template(record: &CatalogIntrinsic) -> String {
    let (c_count, a_count, b_count, d_count) = register_mma_fragment_counts(record);
    let d = register_list(0, d_count);
    let c = register_list(d_count, c_count);
    let a = register_list(d_count + c_count, a_count);
    let b = register_list(d_count + c_count + a_count, b_count);
    if record.register_mma.as_ref().unwrap().adapter
        == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32
    {
        let scale_a = d_count + c_count + a_count + b_count;
        let byte_id_a = scale_a + 1;
        let thread_id_a = scale_a + 2;
        let scale_b = scale_a + 3;
        let byte_id_b = scale_a + 4;
        let thread_id_b = scale_a + 5;
        format!(
            "{} {d}, {a}, {b}, {c}, ${scale_a}, {{${byte_id_a}, ${thread_id_a}}}, ${scale_b}, {{${byte_id_b}, ${thread_id_b}}};",
            expected_ptx_head(record)
        )
    } else {
        format!("{} {d}, {a}, {b}, {c};", expected_ptx_head(record))
    }
}

pub(in crate::render) fn register_mma_constraints(record: &CatalogIntrinsic) -> String {
    let mma = record.register_mma.as_ref().unwrap();
    let (c_count, a_count, b_count, d_count) = register_mma_fragment_counts(record);
    let (output, c, packed) = match mma.adapter {
        RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32 | RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32 => {
            ("=r", "r", "r")
        }
        RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32
        | RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32
        | RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 => ("=f", "f", "r"),
        RegisterMmaAdapter::C2F64A1F64B1F64ToD2F64 => ("=d", "d", "d"),
        RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32
        | RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32
        | RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32 => ("=r", "r", "r"),
    };
    let mut constraints = std::iter::repeat_n(output, d_count)
        .chain(std::iter::repeat_n(c, c_count))
        .chain(std::iter::repeat_n(packed, a_count + b_count))
        .collect::<Vec<_>>();
    if mma.adapter == RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 {
        constraints.extend(["r", "h", "h", "r", "h", "h"]);
    }
    constraints.join(",")
}

pub(in crate::render) fn register_mma_result_variant(record: &CatalogIntrinsic) -> &'static str {
    match record.register_mma.as_ref().unwrap().adapter {
        RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32 | RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32 => {
            "GeneratedMmaResultType::I32"
        }
        RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32 | RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32 => {
            "GeneratedMmaResultType::F32"
        }
        RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 => {
            "GeneratedMmaResultType::F32"
        }
        RegisterMmaAdapter::C2F64A1F64B1F64ToD2F64 => "GeneratedMmaResultType::F64",
        RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32
        | RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32
        | RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32 => "GeneratedMmaResultType::I32",
    }
}

pub(in crate::render) fn sparse_mma_fragment_counts(
    record: &CatalogIntrinsic,
) -> (usize, usize, usize, usize) {
    match record.sparse_mma.as_ref().unwrap().adapter {
        SparseMmaAdapter::C2U32A2U32B2U32MetadataU32SelectorU32ToD2U32 => (2, 2, 2, 2),
        SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32 => (2, 4, 4, 2),
        SparseMmaAdapter::C4F32A2U32B2U32MetadataU32SelectorU32ToD4F32 => (4, 2, 2, 4),
        SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32
        | SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32 => (4, 4, 4, 4),
        SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32 => (4, 2, 2, 4),
    }
}

pub(in crate::render) fn sparse_mma_selector_values(record: &CatalogIntrinsic) -> &'static [u32] {
    match record.sparse_mma.as_ref().unwrap().selector {
        SparseMmaSelector::ImmediateZeroThroughThree => &[0, 1, 2, 3],
        SparseMmaSelector::ImmediateZeroOrOne => &[0, 1],
        SparseMmaSelector::ImmediateZero => &[0],
    }
}

pub(in crate::render) fn sparse_mma_selector_description(
    record: &CatalogIntrinsic,
) -> &'static str {
    match record.sparse_mma.as_ref().unwrap().selector {
        SparseMmaSelector::ImmediateZeroThroughThree => {
            "the compile-time constant `0`, `1`, `2`, or `3`"
        }
        SparseMmaSelector::ImmediateZeroOrOne => "the compile-time constant `0` or `1`",
        SparseMmaSelector::ImmediateZero => "the compile-time constant `0`",
    }
}

pub(in crate::render) fn sparse_mma_selector_error(record: &CatalogIntrinsic) -> &'static str {
    match record.sparse_mma.as_ref().unwrap().selector {
        SparseMmaSelector::ImmediateZeroThroughThree => {
            "sparse MMA selector must be the compile-time constant 0, 1, 2, or 3"
        }
        SparseMmaSelector::ImmediateZeroOrOne => {
            "sparse MMA selector must be the compile-time constant 0 or 1"
        }
        SparseMmaSelector::ImmediateZero => {
            "sparse MMA selector must be the compile-time constant 0"
        }
    }
}

pub(in crate::render) const SPARSE_MMA_STANDARD_METADATA_RULE: &str = "Every 4-bit metadata group must encode two distinct 2-bit indices; `0x0`, `0x5`, `0xa`, and `0xf` are undefined behavior.";
pub(in crate::render) const SPARSE_MMA_ORDERED_METADATA_RULE: &str = "Every 4-bit metadata group must be `0x4`, `0x8`, `0x9`, `0xc`, `0xd`, or `0xe`; any other value is undefined behavior.";
pub(in crate::render) const SPARSE_MMA_ORDERED_TF32_METADATA_RULE: &str =
    "Every 4-bit metadata group must be `0x4` or `0xe`; any other value is undefined behavior.";

pub(in crate::render) fn sparse_mma_metadata_rule(mma: &SparseMma) -> &'static str {
    match (mma.metadata, mma.a_element) {
        (SparseMmaMetadata::Standard, _) => SPARSE_MMA_STANDARD_METADATA_RULE,
        (SparseMmaMetadata::Ordered, SparseMmaElement::Tf32) => {
            SPARSE_MMA_ORDERED_TF32_METADATA_RULE
        }
        (SparseMmaMetadata::Ordered, _) => SPARSE_MMA_ORDERED_METADATA_RULE,
    }
}

pub(in crate::render) fn sparse_mma_import_adapter(record: &CatalogIntrinsic) -> &'static str {
    match record.sparse_mma.as_ref().unwrap().adapter {
        SparseMmaAdapter::C2U32A2U32B2U32MetadataU32SelectorU32ToD2U32 => {
            "GeneratedMmaImportAdapter::C2U32A2U32B2U32ToD2U32"
        }
        SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32 => {
            "GeneratedMmaImportAdapter::C2U32A4U32B4U32ToD2U32"
        }
        SparseMmaAdapter::C4F32A2U32B2U32MetadataU32SelectorU32ToD4F32 => {
            "GeneratedMmaImportAdapter::C4F32A2U32B2U32ToD4F32"
        }
        SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32 => {
            "GeneratedMmaImportAdapter::C4F32A4U32B4U32ToD4F32"
        }
        SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32 => {
            "GeneratedMmaImportAdapter::C4I32A2U32B2U32ToD4I32"
        }
        SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32 => {
            "GeneratedMmaImportAdapter::C4I32A4U32B4U32ToD4I32"
        }
    }
}

pub(in crate::render) fn sparse_mma_attr_variants(
    record: &CatalogIntrinsic,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    let mma = record.sparse_mma.as_ref().expect("sparse-MMA record");
    let shape = match mma.shape {
        SparseMmaShape::M16n8k8 => "SparseMmaShapeAttr::M16n8k8",
        SparseMmaShape::M16n8k16 => "SparseMmaShapeAttr::M16n8k16",
        SparseMmaShape::M16n8k32 => "SparseMmaShapeAttr::M16n8k32",
        SparseMmaShape::M16n8k64 => "SparseMmaShapeAttr::M16n8k64",
        SparseMmaShape::M16n8k128 => "SparseMmaShapeAttr::M16n8k128",
    };
    let accumulator = match mma.accumulator {
        SparseMmaAccumulator::F16 => "SparseMmaAccumulatorAttr::F16",
        SparseMmaAccumulator::F32 => "SparseMmaAccumulatorAttr::F32",
        SparseMmaAccumulator::S32 => "SparseMmaAccumulatorAttr::S32",
    };
    let element = |element| match element {
        SparseMmaElement::F16 => "SparseMmaElementAttr::F16",
        SparseMmaElement::Bf16 => "SparseMmaElementAttr::Bf16",
        SparseMmaElement::Tf32 => "SparseMmaElementAttr::Tf32",
        SparseMmaElement::E2m1 => "SparseMmaElementAttr::E2m1",
        SparseMmaElement::E2m3 => "SparseMmaElementAttr::E2m3",
        SparseMmaElement::E3m2 => "SparseMmaElementAttr::E3m2",
        SparseMmaElement::E4m3 => "SparseMmaElementAttr::E4m3",
        SparseMmaElement::E5m2 => "SparseMmaElementAttr::E5m2",
        SparseMmaElement::S4 => "SparseMmaElementAttr::S4",
        SparseMmaElement::U4 => "SparseMmaElementAttr::U4",
        SparseMmaElement::S8 => "SparseMmaElementAttr::S8",
        SparseMmaElement::U8 => "SparseMmaElementAttr::U8",
    };
    let layout = |layout| match layout {
        SparseMmaLayout::Row => "SparseMmaLayoutAttr::Row",
        SparseMmaLayout::Col => "SparseMmaLayoutAttr::Col",
    };
    let overflow = match mma.overflow {
        SparseMmaOverflow::NotApplicable => "SparseMmaOverflowAttr::NotApplicable",
        SparseMmaOverflow::Wrapping => "SparseMmaOverflowAttr::Wrapping",
        SparseMmaOverflow::Satfinite => "SparseMmaOverflowAttr::Satfinite",
    };
    let metadata = match mma.metadata {
        SparseMmaMetadata::Standard => "SparseMmaMetadataAttr::Standard",
        SparseMmaMetadata::Ordered => "SparseMmaMetadataAttr::Ordered",
    };
    let selector = match mma.selector {
        SparseMmaSelector::ImmediateZeroThroughThree => {
            "SparseMmaSelectorAttr::ImmediateZeroThroughThree"
        }
        SparseMmaSelector::ImmediateZeroOrOne => "SparseMmaSelectorAttr::ImmediateZeroOrOne",
        SparseMmaSelector::ImmediateZero => "SparseMmaSelectorAttr::ImmediateZero",
    };
    (
        shape,
        accumulator,
        element(mma.a_element),
        element(mma.b_element),
        layout(mma.a_layout),
        layout(mma.b_layout),
        overflow,
        metadata,
        selector,
    )
}

pub(in crate::render) fn sparse_mma_ptx_head(record: &CatalogIntrinsic) -> String {
    let mut head = record.expected_ptx.mnemonic.clone();
    for modifier in &record.expected_ptx.modifiers {
        write!(head, ".{modifier}").unwrap();
    }
    head
}

pub(in crate::render) fn sparse_mma_template(record: &CatalogIntrinsic) -> String {
    let (c_count, a_count, b_count, d_count) = sparse_mma_fragment_counts(record);
    let d = register_list(0, d_count);
    let c = register_list(d_count, c_count);
    let a = register_list(d_count + c_count, a_count);
    let b = register_list(d_count + c_count + a_count, b_count);
    let metadata = d_count + c_count + a_count + b_count;
    let selector = metadata + 1;
    format!(
        "{} {d}, {a}, {b}, {c}, ${metadata}, ${selector};",
        sparse_mma_ptx_head(record)
    )
}

pub(in crate::render) fn sparse_mma_constraints(record: &CatalogIntrinsic) -> String {
    let mma = record.sparse_mma.as_ref().unwrap();
    let (c_count, a_count, b_count, d_count) = sparse_mma_fragment_counts(record);
    let (output, accumulator) = match mma.accumulator {
        SparseMmaAccumulator::F16 => ("=r", "r"),
        SparseMmaAccumulator::F32 => ("=f", "f"),
        SparseMmaAccumulator::S32 => ("=r", "r"),
    };
    std::iter::repeat_n(output, d_count)
        .chain(std::iter::repeat_n(accumulator, c_count))
        .chain(std::iter::repeat_n("r", a_count + b_count + 1))
        .chain(std::iter::once("n"))
        .collect::<Vec<_>>()
        .join(",")
}

pub(in crate::render) fn sparse_mma_result_variant(record: &CatalogIntrinsic) -> &'static str {
    match record.sparse_mma.as_ref().unwrap().accumulator {
        SparseMmaAccumulator::F16 => "GeneratedMmaResultType::I32",
        SparseMmaAccumulator::F32 => "GeneratedMmaResultType::F32",
        SparseMmaAccumulator::S32 => "GeneratedMmaResultType::I32",
    }
}

pub(in crate::render) fn sparse_mma_carriers(record: &CatalogIntrinsic) -> (String, String) {
    let mma = record.sparse_mma.as_ref().unwrap();
    let (c_count, a_count, b_count, d_count) = sparse_mma_fragment_counts(record);
    let accumulator = match mma.accumulator {
        SparseMmaAccumulator::F16 => "MmaCarrier::U32",
        SparseMmaAccumulator::F32 => "MmaCarrier::F32",
        SparseMmaAccumulator::S32 => "MmaCarrier::I32",
    };
    let carrier_slice = |carriers: Vec<&str>| format!("&[{}]", carriers.join(", "));
    let operands = carrier_slice(
        std::iter::repeat_n(accumulator, c_count)
            .chain(std::iter::repeat_n(
                "MmaCarrier::U32",
                a_count + b_count + 2,
            ))
            .collect(),
    );
    let results = carrier_slice(std::iter::repeat_n(accumulator, d_count).collect());
    (operands, results)
}

pub(in crate::render) fn register_mma_carriers(
    record: &CatalogIntrinsic,
) -> (&'static str, &'static str) {
    match record.register_mma.as_ref().unwrap().adapter {
        RegisterMmaAdapter::C2U32A2U32B1U32ToD2U32 => (
            "&[MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]",
            "&[MmaCarrier::U32, MmaCarrier::U32]",
        ),
        RegisterMmaAdapter::C2U32A4U32B2U32ToD2U32 => (
            "&[MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]",
            "&[MmaCarrier::U32, MmaCarrier::U32]",
        ),
        RegisterMmaAdapter::C4F32A2U32B1U32ToD4F32 => (
            "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]",
            "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32]",
        ),
        RegisterMmaAdapter::C4F32A4U32B2U32ToD4F32 => (
            "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]",
            "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32]",
        ),
        RegisterMmaAdapter::C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32 => (
            "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U16, MmaCarrier::U16, MmaCarrier::U32, MmaCarrier::U16, MmaCarrier::U16]",
            "&[MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32, MmaCarrier::F32]",
        ),
        RegisterMmaAdapter::C2F64A1F64B1F64ToD2F64 => (
            "&[MmaCarrier::F64, MmaCarrier::F64, MmaCarrier::F64, MmaCarrier::F64]",
            "&[MmaCarrier::F64, MmaCarrier::F64]",
        ),
        RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32 => (
            "&[MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::U32, MmaCarrier::U32]",
            "&[MmaCarrier::I32, MmaCarrier::I32]",
        ),
        RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32 => (
            "&[MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]",
            "&[MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32]",
        ),
        RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32 => (
            "&[MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::U32, MmaCarrier::U32, MmaCarrier::U32]",
            "&[MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32, MmaCarrier::I32]",
        ),
    }
}

pub(in crate::render) fn register_mma_compat_op_type(
    record: &CatalogIntrinsic,
) -> Option<&'static str> {
    match record.id.as_str() {
        "mma_m16n8k16_f32_bf16" => Some("MmaM16N8K16F32Bf16Op"),
        "mma_m16n8k16_f32_f16" => Some("MmaM16N8K16F32F16Op"),
        "mma_m16n8k8_f32_tf32" => Some("MmaM16N8K8F32Tf32Op"),
        "mma_m16n8k32_s32_s8" => Some("MmaM16N8K32S32S8Op"),
        "mma_m8n8k4_f64" => Some("MmaM8N8K4F64Op"),
        _ => None,
    }
}
