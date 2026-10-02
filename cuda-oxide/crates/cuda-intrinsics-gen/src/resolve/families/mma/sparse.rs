/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, OverlayBackendLowering,
    OverlayIntrinsic, RuntimeValidation, SparseMma, SparseMmaAccumulator, SparseMmaAdapter,
    SparseMmaCompatibilitySource, SparseMmaElement, SparseMmaF8F6F4Admission,
    SparseMmaF8F6F4F16Admission, SparseMmaFp8F32Admission, SparseMmaIntegerAdmission,
    SparseMmaLayout, SparseMmaLlvmAdapter, SparseMmaMetadata, SparseMmaOrderedAmpereFloatAdmission,
    SparseMmaOrderedAmpereFloatVariant, SparseMmaOverflow, SparseMmaParticipation,
    SparseMmaSelector, SparseMmaShape,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;

pub(in crate::resolve) const SPARSE_MMA_F8F6F4_TARGETS: &str = "sm_120a|sm_120f|sm_121a|sm_121f";
#[derive(Clone, Copy)]
pub(in crate::resolve) struct SparseMmaCarrierRecipe {
    pub(in crate::resolve) adapter: SparseMmaAdapter,
    pub(in crate::resolve) llvm_adapter: SparseMmaLlvmAdapter,
    pub(in crate::resolve) accumulator: SparseMmaAccumulator,
    pub(in crate::resolve) selector: SparseMmaSelector,
    pub(in crate::resolve) a_registers: usize,
    pub(in crate::resolve) b_registers: usize,
}

impl SparseMmaCarrierRecipe {
    fn accumulator_registers(self) -> usize {
        match self.accumulator {
            SparseMmaAccumulator::F16 => 2,
            SparseMmaAccumulator::F32 | SparseMmaAccumulator::S32 => 4,
        }
    }

    fn operand_count(self) -> usize {
        self.accumulator_registers() + self.a_registers + self.b_registers + 2
    }

    fn selector_index(self) -> usize {
        self.operand_count() - 1
    }

    fn selector_upper_exclusive(self) -> u8 {
        match self.selector {
            SparseMmaSelector::ImmediateZeroThroughThree => 4,
            SparseMmaSelector::ImmediateZeroOrOne => 2,
            SparseMmaSelector::ImmediateZero => 1,
        }
    }

    pub(in crate::resolve) fn rust_arguments(self) -> Vec<String> {
        let accumulator = match self.accumulator {
            SparseMmaAccumulator::F16 => "[u32; 2]",
            SparseMmaAccumulator::F32 => "[f32; 4]",
            SparseMmaAccumulator::S32 => "[i32; 4]",
        };
        vec![
            accumulator.into(),
            format!("[u32; {}]", self.a_registers),
            format!("[u32; {}]", self.b_registers),
            "u32".into(),
            "u32".into(),
        ]
    }

    pub(in crate::resolve) fn dialect_operands(self) -> Vec<String> {
        let accumulator = match self.accumulator {
            SparseMmaAccumulator::F16 => "i32",
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => "i32",
        };
        let accumulator_registers = self.accumulator_registers();
        std::iter::repeat_n(accumulator.to_owned(), accumulator_registers)
            .chain(std::iter::repeat_n(
                "u32".to_owned(),
                self.operand_count() - accumulator_registers,
            ))
            .collect()
    }

    pub(in crate::resolve) fn llvm_arguments(self) -> Vec<String> {
        let accumulator = match self.accumulator {
            SparseMmaAccumulator::F16 => "v2f16",
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => "i32",
        };
        // Ampere f16/bf16 forms carry their multiplicands as <2 x half>
        // vectors in LLVM; every other family carries packed i32.
        let multiplicand = if matches!(
            self.llvm_adapter,
            SparseMmaLlvmAdapter::A2V2F16B2V2F16C2V2F16MetadataI32SelectorI32ToD2V2F16
                | SparseMmaLlvmAdapter::A4V2F16B4V2F16C2V2F16MetadataI32SelectorI32ToD2V2F16
                | SparseMmaLlvmAdapter::A2V2F16B2V2F16C4F32MetadataI32SelectorI32ToD4F32
                | SparseMmaLlvmAdapter::A4V2F16B4V2F16C4F32MetadataI32SelectorI32ToD4F32
        ) {
            "v2f16"
        } else {
            "i32"
        };
        std::iter::repeat_n(multiplicand.to_owned(), self.a_registers + self.b_registers)
            .chain(std::iter::repeat_n(
                accumulator.to_owned(),
                self.accumulator_registers(),
            ))
            .chain(std::iter::repeat_n("i32".to_owned(), 2))
            .collect()
    }

    fn rust_result(self) -> String {
        match self.accumulator {
            SparseMmaAccumulator::F16 => "[u32; 2]",
            SparseMmaAccumulator::F32 => "[f32; 4]",
            SparseMmaAccumulator::S32 => "[i32; 4]",
        }
        .into()
    }

    fn dialect_results(self) -> Vec<String> {
        match self.accumulator {
            SparseMmaAccumulator::F16 => vec!["i32".into(); 2],
            SparseMmaAccumulator::F32 => vec!["f32".into(); 4],
            SparseMmaAccumulator::S32 => vec!["i32".into(); 4],
        }
    }

    fn llvm_results(self) -> Vec<String> {
        match self.accumulator {
            SparseMmaAccumulator::F16 => vec!["v2f16".into(); 2],
            SparseMmaAccumulator::F32 => vec!["f32".into(); 4],
            SparseMmaAccumulator::S32 => vec!["i32".into(); 4],
        }
    }

    pub(in crate::resolve) fn expected_ptx_operands(self) -> Vec<OperandPattern> {
        vec![
            OperandPattern::RegisterList {
                length: self.accumulator_registers(),
            },
            OperandPattern::RegisterList {
                length: self.a_registers,
            },
            OperandPattern::RegisterList {
                length: self.b_registers,
            },
            OperandPattern::RegisterList {
                length: self.accumulator_registers(),
            },
            OperandPattern::Register,
            OperandPattern::Immediate,
        ]
    }

    fn imported_properties(self) -> Vec<String> {
        let selector = self.selector_index();
        vec![
            format!("ImmArg<arg{selector}>"),
            "IntrNoCallback".into(),
            "IntrNoMem".into(),
            format!("Range<arg{selector},0,{}>", self.selector_upper_exclusive()),
        ]
    }
}

pub(in crate::resolve) fn sparse_mma_carrier_recipe(
    shape: SparseMmaShape,
    a_element: SparseMmaElement,
    b_element: SparseMmaElement,
) -> Option<SparseMmaCarrierRecipe> {
    use SparseMmaElement::{E2m1, E2m3, E3m2, E4m3, E5m2, S4, S8, U4, U8};
    use SparseMmaShape::{M16n8k32, M16n8k64, M16n8k128};

    match (shape, a_element, b_element) {
        (M16n8k32, S8 | U8, S8 | U8) | (M16n8k64, S4 | U4, S4 | U4) => {
            Some(SparseMmaCarrierRecipe {
                adapter: SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32,
                llvm_adapter: SparseMmaLlvmAdapter::A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32,
                accumulator: SparseMmaAccumulator::S32,
                selector: SparseMmaSelector::ImmediateZeroOrOne,
                a_registers: 2,
                b_registers: 2,
            })
        }
        (M16n8k64, S8 | U8, S8 | U8) => Some(SparseMmaCarrierRecipe {
            adapter: SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32,
            llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32,
            accumulator: SparseMmaAccumulator::S32,
            selector: SparseMmaSelector::ImmediateZero,
            a_registers: 4,
            b_registers: 4,
        }),
        (M16n8k128, S4 | U4, S4 | U4) => Some(SparseMmaCarrierRecipe {
            adapter: SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32,
            llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32,
            accumulator: SparseMmaAccumulator::S32,
            selector: SparseMmaSelector::ImmediateZero,
            a_registers: 4,
            b_registers: 4,
        }),
        (M16n8k64, E2m1 | E2m3 | E3m2 | E4m3 | E5m2, E2m1 | E2m3 | E3m2 | E4m3 | E5m2) => {
            Some(SparseMmaCarrierRecipe {
                adapter: SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
                llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32,
                accumulator: SparseMmaAccumulator::F32,
                selector: SparseMmaSelector::ImmediateZero,
                a_registers: 4,
                b_registers: 4,
            })
        }
        _ => None,
    }
}

pub(in crate::resolve) fn sparse_mma_f8f6f4_f16_carrier_recipe(
    shape: SparseMmaShape,
    a_element: SparseMmaElement,
    b_element: SparseMmaElement,
) -> Option<SparseMmaCarrierRecipe> {
    use SparseMmaElement::{E2m1, E2m3, E3m2, E4m3, E5m2};

    if shape != SparseMmaShape::M16n8k64
        || !matches!(a_element, E2m1 | E2m3 | E3m2 | E4m3 | E5m2)
        || !matches!(b_element, E2m1 | E2m3 | E3m2 | E4m3 | E5m2)
    {
        return None;
    }
    Some(SparseMmaCarrierRecipe {
        adapter: SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32,
        llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C2V2F16MetadataI32SelectorI32ToD2V2F16,
        accumulator: SparseMmaAccumulator::F16,
        selector: SparseMmaSelector::ImmediateZero,
        a_registers: 4,
        b_registers: 4,
    })
}

fn sparse_mma_ampere_float_carrier_recipe(mma: &SparseMma) -> Option<SparseMmaCarrierRecipe> {
    use SparseMmaAccumulator::{F16 as AccF16, F32 as AccF32};
    use SparseMmaElement::{Bf16, F16, Tf32};
    use SparseMmaShape::{M16n8k8, M16n8k16, M16n8k32};

    // Register width decides the selector domain (PTX 9.7.15.6.1):
    //   2-reg forms (k8 tf32, k16 f16/bf16)  -> one thread per quad, selector 0-3
    //   4-reg forms (k16 tf32, k32 f16/bf16) -> thread pair, selector 0-1
    let (a_registers, b_registers, selector) = match (mma.shape, (mma.a_element, mma.b_element)) {
        (M16n8k8, (Tf32, Tf32)) => (2, 2, SparseMmaSelector::ImmediateZeroThroughThree),
        (M16n8k16, (Tf32, Tf32)) => (4, 4, SparseMmaSelector::ImmediateZeroOrOne),
        (M16n8k16, (Bf16, Bf16) | (F16, F16)) => {
            (2, 2, SparseMmaSelector::ImmediateZeroThroughThree)
        }
        (M16n8k32, (Bf16, Bf16) | (F16, F16)) => (4, 4, SparseMmaSelector::ImmediateZeroOrOne),
        _ => return None,
    };
    let (adapter, llvm_adapter) = match (mma.accumulator, mma.a_element, a_registers) {
        (AccF16, F16, 2) => (
            SparseMmaAdapter::C2U32A2U32B2U32MetadataU32SelectorU32ToD2U32,
            SparseMmaLlvmAdapter::A2V2F16B2V2F16C2V2F16MetadataI32SelectorI32ToD2V2F16,
        ),
        (AccF16, F16, 4) => (
            SparseMmaAdapter::C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32,
            SparseMmaLlvmAdapter::A4V2F16B4V2F16C2V2F16MetadataI32SelectorI32ToD2V2F16,
        ),
        (AccF32, F16, 2) => (
            SparseMmaAdapter::C4F32A2U32B2U32MetadataU32SelectorU32ToD4F32,
            SparseMmaLlvmAdapter::A2V2F16B2V2F16C4F32MetadataI32SelectorI32ToD4F32,
        ),
        (AccF32, F16, 4) => (
            SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
            SparseMmaLlvmAdapter::A4V2F16B4V2F16C4F32MetadataI32SelectorI32ToD4F32,
        ),
        (AccF32, _, 2) => (
            SparseMmaAdapter::C4F32A2U32B2U32MetadataU32SelectorU32ToD4F32,
            SparseMmaLlvmAdapter::A2I32B2I32C4F32MetadataI32SelectorI32ToD4F32,
        ),
        (AccF32, _, 4) => (
            SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
            SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32,
        ),
        _ => return None,
    };
    if mma.accumulator == AccF16 && mma.a_element != F16 {
        return None;
    }
    Some(SparseMmaCarrierRecipe {
        adapter,
        llvm_adapter,
        accumulator: mma.accumulator,
        selector,
        a_registers,
        b_registers,
    })
}

pub(in crate::resolve) struct SparseMmaIdentity {
    pub(in crate::resolve) id: String,
    pub(in crate::resolve) operation_key: String,
    pub(in crate::resolve) source_record: String,
    pub(in crate::resolve) llvm_symbol: String,
    pub(in crate::resolve) ptx_modifiers: Vec<&'static str>,
}

pub(in crate::resolve) struct SparseMmaRecipe {
    pub(in crate::resolve) identity: SparseMmaIdentity,
    pub(in crate::resolve) carrier: SparseMmaCarrierRecipe,
}

pub(in crate::resolve) fn sparse_mma_shape_name(shape: SparseMmaShape) -> &'static str {
    match shape {
        SparseMmaShape::M16n8k8 => "m16n8k8",
        SparseMmaShape::M16n8k16 => "m16n8k16",
        SparseMmaShape::M16n8k32 => "m16n8k32",
        SparseMmaShape::M16n8k64 => "m16n8k64",
        SparseMmaShape::M16n8k128 => "m16n8k128",
    }
}

pub(in crate::resolve) fn sparse_mma_element_name(element: SparseMmaElement) -> &'static str {
    match element {
        SparseMmaElement::F16 => "f16",
        SparseMmaElement::Bf16 => "bf16",
        SparseMmaElement::Tf32 => "tf32",
        SparseMmaElement::E2m1 => "e2m1",
        SparseMmaElement::E2m3 => "e2m3",
        SparseMmaElement::E3m2 => "e3m2",
        SparseMmaElement::E4m3 => "e4m3",
        SparseMmaElement::E5m2 => "e5m2",
        SparseMmaElement::S4 => "s4",
        SparseMmaElement::U4 => "u4",
        SparseMmaElement::S8 => "s8",
        SparseMmaElement::U8 => "u8",
    }
}

/// The reviewed plain sparse FP8 element set. These are the four SM89 forms
/// carrying an F32 accumulator with *standard* (non-block-scale) sparsity
/// metadata; the wider `f8f6f4` matrix on Blackwell uses `kind::f8f6f4` and
/// `ordered_metadata` instead.
pub(in crate::resolve) const SPARSE_MMA_FP8_F32_ELEMENTS: [SparseMmaElement; 2] =
    [SparseMmaElement::E4m3, SparseMmaElement::E5m2];

pub(in crate::resolve) fn is_sparse_mma_fp8_f32_element(element: SparseMmaElement) -> bool {
    SPARSE_MMA_FP8_F32_ELEMENTS.contains(&element)
}

/// True for the plain SM89 sparse FP8 F32 forms: standard sparsity metadata,
/// an F32 accumulator, and an e4m3/e5m2 multiplicand pair. The `kind::f8f6f4`,
/// Ampere-float and `f8f6f4`-F16 families use a different metadata mode or
/// accumulator and must keep their existing identity, target floor and PTX
/// spelling.
pub(in crate::resolve) fn is_sparse_mma_fp8_f32(mma: &SparseMma) -> bool {
    mma.shape == SparseMmaShape::M16n8k64
        && mma.accumulator == SparseMmaAccumulator::F32
        && mma.metadata == SparseMmaMetadata::Standard
        && is_sparse_mma_fp8_f32_element(mma.a_element)
        && is_sparse_mma_fp8_f32_element(mma.b_element)
}

pub(in crate::resolve) fn sparse_mma_identity(mma: &SparseMma) -> SparseMmaIdentity {
    let shape = sparse_mma_shape_name(mma.shape);
    let a_element = sparse_mma_element_name(mma.a_element);
    let b_element = sparse_mma_element_name(mma.b_element);
    let f8f6f4 = matches!(
        mma.a_element,
        SparseMmaElement::E2m1
            | SparseMmaElement::E2m3
            | SparseMmaElement::E3m2
            | SparseMmaElement::E4m3
            | SparseMmaElement::E5m2
    );
    if f8f6f4
        && mma.metadata == SparseMmaMetadata::Ordered
        && matches!(
            mma.accumulator,
            SparseMmaAccumulator::F16 | SparseMmaAccumulator::F32
        )
    {
        let scalar = match mma.accumulator {
            SparseMmaAccumulator::F16 => "f16",
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => unreachable!(),
        };
        return SparseMmaIdentity {
            id: format!(
                "mma_sp_ordered_metadata_{shape}_kind_f8f6f4_{scalar}_{a_element}_{b_element}_{scalar}"
            ),
            operation_key: format!(
                "matrix.mma.sp.{shape}.row.col.kind_f8f6f4.{scalar}.{a_element}.{b_element}.{scalar}.not_applicable.ordered_metadata"
            ),
            source_record: format!(
                "int_nvvm_mma_sp_ordered_metadata_{shape}_row_col_kind_f8f6f4_{scalar}_{a_element}_{b_element}_{scalar}"
            ),
            llvm_symbol: format!(
                "llvm.nvvm.mma.sp.ordered.metadata.{shape}.row.col.kind.f8f6f4.{scalar}.{a_element}.{b_element}.{scalar}"
            ),
            ptx_modifiers: vec![
                "sp::ordered_metadata",
                "sync",
                "aligned",
                shape,
                "row",
                "col",
                "kind::f8f6f4",
                scalar,
                a_element,
                b_element,
                scalar,
            ],
        };
    }
    // Plain (standard-metadata) sparse FP8 with an F32 accumulator. The PTX
    // spelling carries no `kind::f8f6f4` qualifier, and the operation key keeps
    // the sparse family's accumulator-in-slot-7 convention.
    if is_sparse_mma_fp8_f32(mma) {
        return SparseMmaIdentity {
            id: format!("mma_sp_{shape}_f32_{a_element}_{b_element}_f32"),
            operation_key: format!(
                "matrix.mma.sp.{shape}.row.col.f32.{a_element}.{b_element}.f32.not_applicable.standard_metadata"
            ),
            source_record: format!(
                "int_nvvm_mma_sp_{shape}_row_col_f32_{a_element}_{b_element}_f32"
            ),
            llvm_symbol: format!(
                "llvm.nvvm.mma.sp.{shape}.row.col.f32.{a_element}.{b_element}.f32"
            ),
            ptx_modifiers: vec![
                "sp", "sync", "aligned", shape, "row", "col", "f32", a_element, b_element, "f32",
            ],
        };
    }
    if matches!(
        mma.a_element,
        SparseMmaElement::F16 | SparseMmaElement::Bf16 | SparseMmaElement::Tf32
    ) {
        let accumulator = match mma.accumulator {
            SparseMmaAccumulator::F16 => "f16",
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => unreachable!("Ampere float sparse MMA is not integer"),
        };
        // LLVM's symbol tail differs per format family:
        //   f16 -> <acc>.<acc> pair (f16.f16 / f32.f32); bf16/tf32 -> bare element
        let llvm_suffix = match (mma.accumulator, mma.a_element) {
            (SparseMmaAccumulator::F16, SparseMmaElement::F16) => "f16.f16",
            (SparseMmaAccumulator::F32, SparseMmaElement::F16) => "f32.f32",
            (SparseMmaAccumulator::F32, SparseMmaElement::Bf16) => "bf16",
            (SparseMmaAccumulator::F32, SparseMmaElement::Tf32) => "tf32",
            _ => unreachable!("closed Ampere float recipe rejects this combination"),
        };
        return SparseMmaIdentity {
            id: format!("mma_sp_ordered_metadata_{shape}_{accumulator}_{a_element}"),
            operation_key: format!(
                "matrix.mma.sp.{shape}.row.col.{accumulator}.{a_element}.{b_element}.{accumulator}.not_applicable.ordered_metadata"
            ),
            source_record: format!(
                "int_nvvm_mma_sp_ordered_metadata_{shape}_row_col_{}",
                llvm_suffix.replace('.', "_")
            ),
            llvm_symbol: format!("llvm.nvvm.mma.sp.ordered.metadata.{shape}.row.col.{llvm_suffix}"),
            ptx_modifiers: vec![
                "sp::ordered_metadata",
                "sync",
                "aligned",
                shape,
                "row",
                "col",
                accumulator,
                a_element,
                b_element,
                accumulator,
            ],
        };
    }
    let compact_elements = if mma.a_element == mma.b_element {
        a_element.to_owned()
    } else {
        format!("{a_element}_{b_element}")
    };
    let dotted_elements = if mma.a_element == mma.b_element {
        a_element.to_owned()
    } else {
        format!("{a_element}.{b_element}")
    };
    let metadata_id_prefix = match mma.metadata {
        SparseMmaMetadata::Standard => "",
        SparseMmaMetadata::Ordered => "ordered_metadata_",
    };
    let metadata_source_prefix = metadata_id_prefix;
    let metadata_symbol_prefix = match mma.metadata {
        SparseMmaMetadata::Standard => "",
        SparseMmaMetadata::Ordered => "ordered.metadata.",
    };
    let metadata_key = match mma.metadata {
        SparseMmaMetadata::Standard => "standard_metadata",
        SparseMmaMetadata::Ordered => "ordered_metadata",
    };
    let overflow_id_suffix = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "",
        SparseMmaOverflow::Satfinite => "_satfinite",
    };
    let overflow_source_prefix = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "",
        SparseMmaOverflow::Satfinite => "satfinite_",
    };
    let overflow_symbol_prefix = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "",
        SparseMmaOverflow::Satfinite => "satfinite.",
    };
    let overflow_key = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "wrapping",
        SparseMmaOverflow::Satfinite => "satfinite",
    };

    let mut ptx_modifiers = vec![
        match mma.metadata {
            SparseMmaMetadata::Standard => "sp",
            SparseMmaMetadata::Ordered => "sp::ordered_metadata",
        },
        "sync",
        "aligned",
        shape,
        "row",
        "col",
    ];
    if mma.overflow == SparseMmaOverflow::Satfinite {
        ptx_modifiers.push("satfinite");
    }
    ptx_modifiers.extend(["s32", a_element, b_element, "s32"]);

    SparseMmaIdentity {
        id: format!(
            "mma_sp_{metadata_id_prefix}{shape}_s32_{compact_elements}{overflow_id_suffix}"
        ),
        operation_key: format!(
            "matrix.mma.sp.{shape}.row.col.s32.{a_element}.{b_element}.s32.{overflow_key}.{metadata_key}"
        ),
        source_record: format!(
            "int_nvvm_mma_sp_{metadata_source_prefix}{shape}_row_col_{overflow_source_prefix}{compact_elements}"
        ),
        llvm_symbol: format!(
            "llvm.nvvm.mma.sp.{metadata_symbol_prefix}{shape}.row.col.{overflow_symbol_prefix}{dotted_elements}"
        ),
        ptx_modifiers,
    }
}

pub(in crate::resolve) fn sparse_mma_recipe(mma: &SparseMma) -> Option<SparseMmaRecipe> {
    let ampere_float = matches!(
        mma.a_element,
        SparseMmaElement::F16 | SparseMmaElement::Bf16 | SparseMmaElement::Tf32
    );
    let carrier = if ampere_float {
        sparse_mma_ampere_float_carrier_recipe(mma)?
    } else if mma.accumulator == SparseMmaAccumulator::F16 {
        sparse_mma_f8f6f4_f16_carrier_recipe(mma.shape, mma.a_element, mma.b_element)?
    } else {
        sparse_mma_carrier_recipe(mma.shape, mma.a_element, mma.b_element)?
    };

    // The plain SM89 FP8 forms are the only F32-accumulator sparse family that
    // admits standard (non-block-scale) sparsity metadata. Every other floating
    // sparse family stays pinned to ordered metadata.
    if is_sparse_mma_fp8_f32(mma) {
        let scalar_contract = mma.accumulator == SparseMmaAccumulator::F32
            && mma.overflow == SparseMmaOverflow::NotApplicable
            && mma.metadata == SparseMmaMetadata::Standard;
        if !scalar_contract
            || mma.a_layout != SparseMmaLayout::Row
            || mma.b_layout != SparseMmaLayout::Col
            || mma.selector != carrier.selector
            || mma.participation
                != SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes
            || mma.adapter != carrier.adapter
            || mma.llvm_adapter != carrier.llvm_adapter
            || mma.compatibility_source != SparseMmaCompatibilitySource::GeneratedStub
        {
            return None;
        }
        return Some(SparseMmaRecipe {
            identity: sparse_mma_identity(mma),
            carrier,
        });
    }

    let scalar_contract = match carrier.accumulator {
        SparseMmaAccumulator::F16 | SparseMmaAccumulator::F32 => {
            mma.accumulator == carrier.accumulator
                && mma.overflow == SparseMmaOverflow::NotApplicable
                && mma.metadata == SparseMmaMetadata::Ordered
        }
        SparseMmaAccumulator::S32 => {
            mma.accumulator == SparseMmaAccumulator::S32
                && matches!(
                    mma.overflow,
                    SparseMmaOverflow::Wrapping | SparseMmaOverflow::Satfinite
                )
        }
    };
    if !scalar_contract
        || mma.a_layout != SparseMmaLayout::Row
        || mma.b_layout != SparseMmaLayout::Col
        || mma.selector != carrier.selector
        || mma.participation
            != SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes
        || mma.adapter != carrier.adapter
        || mma.llvm_adapter != carrier.llvm_adapter
        || mma.compatibility_source != SparseMmaCompatibilitySource::GeneratedStub
    {
        return None;
    }

    Some(SparseMmaRecipe {
        identity: sparse_mma_identity(mma),
        carrier,
    })
}

pub(in crate::resolve) fn sparse_mma_minimum_ptx(mma: &SparseMma) -> &'static str {
    // PTX ISA 8.4 introduced the plain sparse FP8 forms for sm_89; they are the
    // only sparse FP8 variants outside the 8.7 `kind::f8f6f4` block-scale family.
    if is_sparse_mma_fp8_f32(mma) {
        return "8.4";
    }
    if matches!(
        mma.a_element,
        SparseMmaElement::F16 | SparseMmaElement::Bf16 | SparseMmaElement::Tf32
    ) {
        return "8.5";
    }
    if matches!(
        mma.accumulator,
        SparseMmaAccumulator::F16 | SparseMmaAccumulator::F32
    ) {
        return "8.7";
    }
    match mma.metadata {
        SparseMmaMetadata::Standard => "7.1",
        SparseMmaMetadata::Ordered => "8.5",
    }
}

pub(in crate::resolve) fn sparse_mma_hardware(
    mma: &SparseMma,
) -> (&'static str, Option<&'static str>) {
    // The plain FP8 forms are available from Ada (sm_89) onward on every
    // architecture, not only the Blackwell `kind::f8f6f4` matrix.
    if is_sparse_mma_fp8_f32(mma) {
        return ("all", Some("sm_89"));
    }
    if matches!(
        mma.a_element,
        SparseMmaElement::F16 | SparseMmaElement::Bf16 | SparseMmaElement::Tf32
    ) {
        return ("all", Some("sm_80"));
    }
    match mma.accumulator {
        SparseMmaAccumulator::F16 | SparseMmaAccumulator::F32 => (SPARSE_MMA_F8F6F4_TARGETS, None),
        SparseMmaAccumulator::S32 => ("all", Some("sm_80")),
    }
}

pub(in crate::resolve) fn sparse_mma_ptx_section(_: SparseMmaMetadata) -> &'static str {
    "9.7.15.6.3 Multiply-and-Accumulate Instruction: mma.sp"
}

// The eight reviewed forms in ledger order (i1017..i1024). The admission
// must list exactly these, in this order, so ABI identity stays stable.
pub(in crate::resolve) const SPARSE_MMA_ORDERED_AMPERE_FLOAT_VARIANTS:
    [SparseMmaOrderedAmpereFloatVariant; 8] = [
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k8,
        accumulator: SparseMmaAccumulator::F32,
        element: SparseMmaElement::Tf32,
    },
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k16,
        accumulator: SparseMmaAccumulator::F32,
        element: SparseMmaElement::Tf32,
    },
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k16,
        accumulator: SparseMmaAccumulator::F32,
        element: SparseMmaElement::Bf16,
    },
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k16,
        accumulator: SparseMmaAccumulator::F16,
        element: SparseMmaElement::F16,
    },
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k16,
        accumulator: SparseMmaAccumulator::F32,
        element: SparseMmaElement::F16,
    },
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k32,
        accumulator: SparseMmaAccumulator::F32,
        element: SparseMmaElement::Bf16,
    },
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k32,
        accumulator: SparseMmaAccumulator::F16,
        element: SparseMmaElement::F16,
    },
    SparseMmaOrderedAmpereFloatVariant {
        shape: SparseMmaShape::M16n8k32,
        accumulator: SparseMmaAccumulator::F32,
        element: SparseMmaElement::F16,
    },
];

pub(in crate::resolve) fn expand_sparse_mma_ordered_ampere_float_admission(
    admission: &SparseMmaOrderedAmpereFloatAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "ordered Ampere floating sparse MMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission.variants == SPARSE_MMA_ORDERED_AMPERE_FLOAT_VARIANTS,
        "ordered Ampere floating sparse MMA admission must retain the eight reviewed variants in ABI order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let mut mma = SparseMma {
                shape: variant.shape,
                accumulator: variant.accumulator,
                a_element: variant.element,
                b_element: variant.element,
                a_layout: SparseMmaLayout::Row,
                b_layout: SparseMmaLayout::Col,
                overflow: SparseMmaOverflow::NotApplicable,
                metadata: SparseMmaMetadata::Ordered,
                selector: SparseMmaSelector::ImmediateZero,
                participation:
                    SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter: SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
                llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32,
                compatibility_source: SparseMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };
            let carrier = sparse_mma_ampere_float_carrier_recipe(&mma).context(
                "ordered Ampere floating sparse MMA admission uses an unsupported carrier",
            )?;
            mma.selector = carrier.selector;
            mma.adapter = carrier.adapter;
            mma.llvm_adapter = carrier.llvm_adapter;
            let recipe = sparse_mma_recipe(&mma).context(
                "ordered Ampere floating sparse MMA admission requests a variant outside the closed recipe set",
            )?;
            Ok(sparse_mma_overlay_record(
                String::new(),
                mma,
                recipe,
                &admission.llvm_evidence_profile,
                &admission.libnvvm_evidence_profile,
                format!(
                    "Multiplies warp-distributed ordered sparse {} A and B fragments and adds a {} accumulator.",
                    sparse_mma_element_name(variant.element),
                    match variant.accumulator {
                        SparseMmaAccumulator::F16 => "packed f16",
                        SparseMmaAccumulator::F32 => "f32",
                        SparseMmaAccumulator::S32 => unreachable!(),
                    },
                ),
            ))
        })
        .collect()
}

pub(in crate::resolve) fn expand_sparse_mma_integer_admission(
    admission: &SparseMmaIntegerAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        !admission.variants.is_empty(),
        "compact sparse integer MMA admission has no variants"
    );
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "sparse integer MMA runtime validation may be marked executed only with GPU evidence"
    );

    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(admission.variants.len());
    for variant in &admission.variants {
        ensure!(
            seen.insert((
                variant.shape,
                variant.a_element,
                variant.b_element,
                variant.overflow,
            )),
            "compact sparse integer MMA admission contains a duplicate variant"
        );
        let carrier = sparse_mma_carrier_recipe(
            variant.shape,
            variant.a_element,
            variant.b_element,
        )
        .with_context(
            || "compact sparse integer MMA admission uses unsupported or mixed-width elements",
        )?;
        let mma = SparseMma {
            shape: variant.shape,
            accumulator: SparseMmaAccumulator::S32,
            a_element: variant.a_element,
            b_element: variant.b_element,
            a_layout: SparseMmaLayout::Row,
            b_layout: SparseMmaLayout::Col,
            overflow: variant.overflow,
            metadata: admission.metadata,
            selector: carrier.selector,
            participation:
                SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
            adapter: carrier.adapter,
            llvm_adapter: carrier.llvm_adapter,
            compatibility_source: SparseMmaCompatibilitySource::GeneratedStub,
            runtime_validation: admission.runtime_validation,
        };
        let recipe = sparse_mma_recipe(&mma).with_context(
            || "compact sparse integer MMA admission requests a variant outside the closed recipe set",
        )?;
        let signedness = |element| match element {
            SparseMmaElement::S4 => "signed",
            SparseMmaElement::U4 => "unsigned",
            SparseMmaElement::S8 => "signed",
            SparseMmaElement::U8 => "unsigned",
            _ => unreachable!("integer admission rejects floating formats"),
        };
        let width = match (variant.a_element, variant.b_element) {
            (
                SparseMmaElement::S4 | SparseMmaElement::U4,
                SparseMmaElement::S4 | SparseMmaElement::U4,
            ) => "INT4",
            (
                SparseMmaElement::S8 | SparseMmaElement::U8,
                SparseMmaElement::S8 | SparseMmaElement::U8,
            ) => "INT8",
            _ => unreachable!("carrier selection rejects mixed element widths"),
        };
        let overflow = match variant.overflow {
            SparseMmaOverflow::NotApplicable => {
                unreachable!("integer admission rejects inapplicable overflow")
            }
            SparseMmaOverflow::Wrapping => "wrapping",
            SparseMmaOverflow::Satfinite => "saturating",
        };
        let metadata = match admission.metadata {
            SparseMmaMetadata::Standard => "",
            SparseMmaMetadata::Ordered => " with ordered sparsity metadata",
        };
        let summary = format!(
            "Multiplies warp-distributed sparse {} A and {} B {width} fragments{metadata} and adds a {overflow} s32 accumulator.",
            signedness(variant.a_element),
            signedness(variant.b_element),
        );
        records.push(sparse_mma_overlay_record(
            String::new(),
            mma,
            recipe,
            &admission.llvm_evidence_profile,
            &admission.libnvvm_evidence_profile,
            summary,
        ));
    }
    Ok(records)
}

pub(in crate::resolve) fn expand_sparse_mma_f8f6f4_admission(
    admission: &SparseMmaF8F6F4Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "sparse f8f6f4 MMA runtime validation may be marked executed only with GPU evidence"
    );

    let formats = vec![
        SparseMmaElement::E2m1,
        SparseMmaElement::E2m3,
        SparseMmaElement::E3m2,
        SparseMmaElement::E4m3,
        SparseMmaElement::E5m2,
    ];
    ensure!(
        admission.a_elements == formats,
        "compact sparse f8f6f4 MMA admission must list the canonical five A formats"
    );
    ensure!(
        admission.b_elements == formats,
        "compact sparse f8f6f4 MMA admission must list the canonical five B formats"
    );
    ensure!(
        admission.product_count
            == admission
                .a_elements
                .len()
                .checked_mul(admission.b_elements.len())
                .context("compact sparse f8f6f4 MMA admission product count overflow")?
            && admission.product_count == 25,
        "compact sparse f8f6f4 MMA admission product_count must be exactly 25"
    );
    let mut records = Vec::with_capacity(admission.product_count);
    for &a_element in &admission.a_elements {
        for &b_element in &admission.b_elements {
            let carrier = sparse_mma_carrier_recipe(SparseMmaShape::M16n8k64, a_element, b_element)
                .with_context(
                    || "compact sparse f8f6f4 MMA admission uses an unsupported format",
                )?;
            ensure!(
                carrier.accumulator == SparseMmaAccumulator::F32,
                "compact sparse f8f6f4 MMA admission contains an integer format"
            );
            let mma = SparseMma {
                shape: SparseMmaShape::M16n8k64,
                accumulator: SparseMmaAccumulator::F32,
                a_element,
                b_element,
                a_layout: SparseMmaLayout::Row,
                b_layout: SparseMmaLayout::Col,
                overflow: SparseMmaOverflow::NotApplicable,
                metadata: SparseMmaMetadata::Ordered,
                selector: carrier.selector,
                participation:
                    SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter: carrier.adapter,
                llvm_adapter: carrier.llvm_adapter,
                compatibility_source: SparseMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };
            let recipe = sparse_mma_recipe(&mma).with_context(|| {
            "compact sparse f8f6f4 MMA admission requests a variant outside the closed recipe set"
        })?;
            let summary = format!(
                "Multiplies warp-distributed sparse {} A and {} B fragments and adds an f32 accumulator.",
                sparse_mma_element_name(a_element),
                sparse_mma_element_name(b_element),
            );
            records.push(sparse_mma_overlay_record(
                String::new(),
                mma,
                recipe,
                &admission.llvm_evidence_profile,
                &admission.libnvvm_evidence_profile,
                summary,
            ));
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

pub(in crate::resolve) const SPARSE_MMA_F8F6F4_ELEMENTS: [SparseMmaElement; 5] = [
    SparseMmaElement::E2m1,
    SparseMmaElement::E2m3,
    SparseMmaElement::E3m2,
    SparseMmaElement::E4m3,
    SparseMmaElement::E5m2,
];

pub(in crate::resolve) fn expand_sparse_mma_f8f6f4_f16_admission(
    admission: &SparseMmaF8F6F4F16Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "sparse f8f6f4 F16 MMA runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "sparse f8f6f4 F16 MMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission.a_elements == SPARSE_MMA_F8F6F4_ELEMENTS
            && admission.b_elements == SPARSE_MMA_F8F6F4_ELEMENTS
            && admission.product_count == 25,
        "sparse f8f6f4 F16 MMA admission must contain the canonical 5 by 5 element matrix"
    );

    let mut records = Vec::with_capacity(admission.product_count);
    for &a_element in &admission.a_elements {
        for &b_element in &admission.b_elements {
            let carrier = sparse_mma_f8f6f4_f16_carrier_recipe(
                SparseMmaShape::M16n8k64,
                a_element,
                b_element,
            )
            .context("compact sparse f8f6f4 F16 MMA admission uses an unsupported format")?;
            let mma = SparseMma {
                shape: SparseMmaShape::M16n8k64,
                accumulator: SparseMmaAccumulator::F16,
                a_element,
                b_element,
                a_layout: SparseMmaLayout::Row,
                b_layout: SparseMmaLayout::Col,
                overflow: SparseMmaOverflow::NotApplicable,
                metadata: SparseMmaMetadata::Ordered,
                selector: SparseMmaSelector::ImmediateZero,
                participation:
                    SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter: carrier.adapter,
                llvm_adapter: carrier.llvm_adapter,
                compatibility_source: SparseMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };
            let recipe = sparse_mma_recipe(&mma).context(
                "compact sparse f8f6f4 F16 MMA admission requests a variant outside the closed recipe set",
            )?;
            let summary = format!(
                "Multiplies warp-distributed sparse {} A and {} B fragments and adds a packed F16 accumulator.",
                sparse_mma_element_name(a_element),
                sparse_mma_element_name(b_element),
            );
            records.push(sparse_mma_overlay_record(
                String::new(),
                mma,
                recipe,
                &admission.llvm_evidence_profile,
                &admission.libnvvm_evidence_profile,
                summary,
            ));
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

/// Expands the compact admission for the reviewed plain SM89 sparse FP8 F32
/// forms into the exact four catalog records, in `a_elements` × `b_elements`
/// order.
pub(in crate::resolve) fn expand_sparse_mma_fp8_f32_admission(
    admission: &SparseMmaFp8F32Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "sparse FP8 MMA runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "sparse FP8 MMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission.a_elements == SPARSE_MMA_FP8_F32_ELEMENTS
            && admission.b_elements == SPARSE_MMA_FP8_F32_ELEMENTS,
        "sparse FP8 MMA admission must list the canonical e4m3/e5m2 multiplicand matrix"
    );
    ensure!(
        admission.product_count
            == admission
                .a_elements
                .len()
                .checked_mul(admission.b_elements.len())
                .context("sparse FP8 MMA admission product count overflow")?
            && admission.product_count == 4,
        "sparse FP8 MMA admission product_count must be exactly 4"
    );

    let mut records = Vec::with_capacity(admission.product_count);
    for &a_element in &admission.a_elements {
        for &b_element in &admission.b_elements {
            let carrier = sparse_mma_carrier_recipe(SparseMmaShape::M16n8k64, a_element, b_element)
                .context("compact sparse FP8 MMA admission uses an unsupported format")?;
            ensure!(
                carrier.accumulator == SparseMmaAccumulator::F32,
                "compact sparse FP8 MMA admission contains an integer format"
            );
            let mma = SparseMma {
                shape: SparseMmaShape::M16n8k64,
                accumulator: SparseMmaAccumulator::F32,
                a_element,
                b_element,
                a_layout: SparseMmaLayout::Row,
                b_layout: SparseMmaLayout::Col,
                overflow: SparseMmaOverflow::NotApplicable,
                metadata: SparseMmaMetadata::Standard,
                selector: carrier.selector,
                participation:
                    SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter: carrier.adapter,
                llvm_adapter: carrier.llvm_adapter,
                compatibility_source: SparseMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };
            let recipe = sparse_mma_recipe(&mma).with_context(|| {
                "compact sparse FP8 MMA admission requests a variant outside the closed recipe set"
            })?;
            let summary = format!(
                "Multiplies warp-distributed sparse {} A and {} B fragments and adds an f32 accumulator.",
                sparse_mma_element_name(a_element),
                sparse_mma_element_name(b_element),
            );
            records.push(sparse_mma_overlay_record(
                String::new(),
                mma,
                recipe,
                &admission.llvm_evidence_profile,
                &admission.libnvvm_evidence_profile,
                summary,
            ));
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

pub(in crate::resolve) fn sparse_mma_overlay_record(
    abi_id: String,
    mma: SparseMma,
    recipe: SparseMmaRecipe,
    llvm_evidence_profile: &str,
    libnvvm_evidence_profile: &str,
    summary: String,
) -> OverlayIntrinsic {
    let identity = &recipe.identity;
    let minimum_ptx = sparse_mma_minimum_ptx(&mma);
    let (targets, minimum_sm) = sparse_mma_hardware(&mma);
    OverlayIntrinsic {
        id: identity.id.clone(),
        abi_id,
        operation_key: identity.operation_key.clone(),
        family: "sparse_mma".into(),
        source: None,
        source_record: Some(identity.source_record.clone()),
        rust_module: "matrix".into(),
        rust_name: identity.id.clone(),
        rust_arguments: recipe.carrier.rust_arguments(),
        rust_result: recipe.carrier.rust_result(),
        safe: false,
        must_use: true,
        safe_allowlist_reason: None,
        public_rust_path: format!("cuda_intrinsics::matrix::{}", identity.id),
        compatibility_rust_paths: vec![format!("cuda_device::wmma::{}", identity.id)],
        dialect_op_type: "SparseMmaOp".into(),
        dialect_op_name: "nvvm.sparse_mma".into(),
        dialect_operands: recipe.carrier.dialect_operands(),
        dialect_results: recipe.carrier.dialect_results(),
        llvm_symbol: Some(identity.llvm_symbol.clone()),
        resolved_llvm_symbol: None,
        llvm_arguments: recipe.carrier.llvm_arguments(),
        llvm_results: recipe.carrier.llvm_results(),
        pure: false,
        memory: "none".into(),
        convergent: true,
        execution_scope: "warp".into(),
        minimum_ptx: minimum_ptx.into(),
        minimum_sm: minimum_sm.map(str::to_owned),
        ptx_result: recipe.carrier.rust_result(),
        targets: targets.into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: sparse_mma_ptx_section(mma.metadata).into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma-sp".into(),
        lowering: "generated_sparse_mma".into(),
        backend_lowerings: [
            (IntrinsicBackend::LlvmNvptx, llvm_evidence_profile),
            (IntrinsicBackend::LibNvvm, libnvvm_evidence_profile),
        ]
        .into_iter()
        .map(|(backend, evidence_profile)| OverlayBackendLowering {
            backend,
            mechanism: BackendLoweringMechanism::InlinePtx,
            evidence_profile: evidence_profile.into(),
            targets: None,
            minimum_ptx: Some(minimum_ptx.into()),
            minimum_sm: minimum_sm.map(str::to_owned),
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
        mbarrier_extended: None,
        register_mma: None,
        sparse_mma: Some(mma),
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
        expected_ptx: InstructionPattern {
            mnemonic: "mma".into(),
            modifiers: identity
                .ptx_modifiers
                .iter()
                .map(|value| (*value).into())
                .collect(),
            operands: recipe.carrier.expected_ptx_operands(),
        },
        summary,
    }
}

pub(in crate::resolve) fn validate_sparse_mma_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let mma = policy
        .sparse_mma
        .as_ref()
        .with_context(|| format!("{} has no closed sparse-MMA contract", policy.id))?;
    let recipe = sparse_mma_recipe(mma)
        .with_context(|| format!("{} requests an unsupported sparse-MMA variant", policy.id))?;
    let identity = &recipe.identity;
    let minimum_ptx = sparse_mma_minimum_ptx(mma);
    let (targets, minimum_sm) = sparse_mma_hardware(mma);
    ensure!(
        policy.id == identity.id
            && policy.operation_key == identity.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(identity.source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(identity.llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} sparse-MMA identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == identity.id
            && policy.public_rust_path == format!("cuda_intrinsics::matrix::{}", identity.id)
            && policy.rust_arguments == recipe.carrier.rust_arguments()
            && policy.rust_result == recipe.carrier.rust_result()
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.compatibility_rust_paths == [format!("cuda_device::wmma::{}", identity.id)],
        "{} must preserve its unsafe must-use Rust sparse-MMA API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "SparseMmaOp"
            && policy.dialect_op_name == "nvvm.sparse_mma"
            && policy.dialect_operands == recipe.carrier.dialect_operands()
            && policy.dialect_results == recipe.carrier.dialect_results()
            && policy.llvm_arguments == recipe.carrier.llvm_arguments()
            && policy.llvm_results == recipe.carrier.llvm_results()
            && policy.ptx_result == recipe.carrier.rust_result()
            && policy.lowering == "generated_sparse_mma",
        "{} sparse-MMA carrier or lowering adapter disagrees with its recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == minimum_ptx
            && policy.minimum_sm.as_deref() == minimum_sm
            && policy.targets == targets,
        "{} sparse-MMA effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == sparse_mma_ptx_section(mma.metadata)
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma-sp",
        "{} sparse-MMA PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_MMA_SP"]
            && declaration.properties == recipe.carrier.imported_properties()
            && declaration.selections.len() == 1
            && (if matches!(
                mma.a_element,
                SparseMmaElement::F16 | SparseMmaElement::Bf16 | SparseMmaElement::Tf32
            ) {
                declaration.selections[0].predicates
                    == [
                        "Subtarget->getSmVersion() >= 80",
                        "Subtarget->getPTXVersion() >= 85",
                    ]
            } else if is_sparse_mma_fp8_f32(mma) {
                // The upstream NVPTX selection for the plain SM89 FP8 forms
                // carries the Ampere floor and the Ada/PTX 8.4 floor as four
                // independent predicates.
                declaration.selections[0].predicates
                    == [
                        "Subtarget->getSmVersion() >= 80",
                        "Subtarget->getPTXVersion() >= 71",
                        "Subtarget->getSmVersion() >= 89",
                        "Subtarget->getPTXVersion() >= 84",
                    ]
            } else if matches!(
                mma.accumulator,
                SparseMmaAccumulator::F16 | SparseMmaAccumulator::F32
            ) {
                declaration.selections[0].predicates == ["Subtarget->hasMMABlockScale()"]
            } else {
                true
            })
            && selection_matches_policy(policy, &declaration.selections[0])?,
        "{} imported sparse MMA declaration changed its class, immediate range, properties, or exact selection contract",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mma"
            && policy.expected_ptx.modifiers == identity.ptx_modifiers
            && policy.expected_ptx.operands == recipe.carrier.expected_ptx_operands(),
        "{} expected PTX does not match its exact sparse-MMA spelling",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, minimum_ptx, minimum_sm),
            (IntrinsicBackend::LibNvvm, minimum_ptx, minimum_sm),
        ],
        "sparse MMA",
    )?;
    ensure_no_other_family_contract(policy, "sparse MMA")?;
    Ok(())
}
