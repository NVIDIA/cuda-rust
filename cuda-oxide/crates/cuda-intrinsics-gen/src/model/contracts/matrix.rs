/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::super::core::RuntimeValidation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StmatrixMultiplicity {
    X2,
    X4,
}

impl StmatrixMultiplicity {
    pub const fn register_count(self) -> usize {
        match self {
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StmatrixLayout {
    Normal,
    Transposed,
}

/// Closed semantic identity for the generated `ldmatrix` family.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LdmatrixVariant {
    pub shape: LdmatrixShape,
    pub multiplicity: LdmatrixMultiplicity,
    pub layout: LdmatrixLayout,
    pub element: LdmatrixElement,
    pub state_space: LdmatrixStateSpace,
}

impl LdmatrixVariant {
    pub const fn register_count(&self) -> usize {
        let matrices = self.multiplicity.register_count();
        match self.shape {
            LdmatrixShape::M8n8 | LdmatrixShape::M8n16 => matrices,
            LdmatrixShape::M16n16 => matrices * 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixShape {
    M8n8,
    M8n16,
    M16n16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixMultiplicity {
    X1,
    X2,
    X4,
}

impl LdmatrixMultiplicity {
    pub const fn register_count(self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixLayout {
    Normal,
    Transposed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixElement {
    B16,
    B8,
    B8x16B4x16P64,
    B8x16B6x16P32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixStateSpace {
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LdmatrixSafety {
    pub participation: LdmatrixParticipation,
    pub address_contract: LdmatrixAddressContract,
    pub memory_order: LdmatrixMemoryOrder,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixParticipation {
    AllWarpLanesSameInstruction,
    AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixAddressContract {
    WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadableWithSm75Replication,
    WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadable,
    WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedThirtyTwoBytesReadable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixMemoryOrder {
    Weak,
}

/// Closed contract for the in-register warp matrix transpose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Movmatrix {
    pub participation: MovmatrixParticipation,
    pub adapter: MovmatrixAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovmatrixParticipation {
    AllWarpLanesSameInstructionNoExitedLanes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovmatrixAdapter {
    PackedB16x2U32ToPackedB16x2U32,
}

/// Closed contract for register-only warp-level `mma.sync` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMma {
    pub shape: RegisterMmaShape,
    pub operation: RegisterMmaOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<RegisterMmaKind>,
    pub accumulator: RegisterMmaAccumulator,
    pub a_element: RegisterMmaElement,
    pub b_element: RegisterMmaElement,
    pub a_layout: RegisterMmaLayout,
    pub b_layout: RegisterMmaLayout,
    pub overflow: RegisterMmaOverflow,
    pub participation: RegisterMmaParticipation,
    pub adapter: RegisterMmaAdapter,
    pub compatibility_source: RegisterMmaCompatibilitySource,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaKind {
    Standard,
    F8f6f4,
    Mxf8f6f4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaShape {
    M8n8k4,
    M8n8k16,
    M8n8k32,
    M16n8k4,
    M16n8k8,
    M16n8k16,
    M16n8k32,
    M16n8k64,
    M8n8k128,
    M16n8k128,
    M16n8k256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaOperation {
    Multiply,
    AndPopc,
    XorPopc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaAccumulator {
    F16,
    F32,
    F64,
    S32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaElement {
    B1,
    Bf16,
    E2m1,
    E2m3,
    E3m2,
    E4m3,
    E5m2,
    F16,
    Tf32,
    F64,
    S4,
    U4,
    S8,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaLayout {
    Row,
    Col,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaOverflow {
    NotApplicable,
    Wrapping,
    Satfinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaParticipation {
    AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
}

/// Rust `C, A, B` fragment shape used by the importer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaAdapter {
    C2U32A2U32B1U32ToD2U32,
    C2U32A4U32B2U32ToD2U32,
    C4F32A2U32B1U32ToD4F32,
    C4F32A4U32B2U32ToD4F32,
    C2F64A1F64B1F64ToD2F64,
    C2I32A1U32B1U32ToD2I32,
    C4I32A4U32B2U32ToD4I32,
    C4I32A2U32B1U32ToD4I32,
    C4F32A4U32B2U32Scales2U32Selectors4U16ToD4F32,
}

/// Where the stable `cuda_device::wmma` callable is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaCompatibilitySource {
    ExistingStub,
    GeneratedStub,
}

/// Closed semantic contract for register-only sparse `mma.sp` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMma {
    pub shape: SparseMmaShape,
    pub accumulator: SparseMmaAccumulator,
    pub a_element: SparseMmaElement,
    pub b_element: SparseMmaElement,
    pub a_layout: SparseMmaLayout,
    pub b_layout: SparseMmaLayout,
    pub overflow: SparseMmaOverflow,
    pub metadata: SparseMmaMetadata,
    pub selector: SparseMmaSelector,
    pub participation: SparseMmaParticipation,
    pub adapter: SparseMmaAdapter,
    pub llvm_adapter: SparseMmaLlvmAdapter,
    pub compatibility_source: SparseMmaCompatibilitySource,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaShape {
    M16n8k8,
    M16n8k16,
    M16n8k32,
    M16n8k64,
    M16n8k128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaAccumulator {
    F16,
    F32,
    S32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaElement {
    F16,
    Bf16,
    Tf32,
    E2m1,
    E2m3,
    E3m2,
    E4m3,
    E5m2,
    S4,
    U4,
    S8,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaLayout {
    Row,
    Col,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaOverflow {
    NotApplicable,
    Wrapping,
    Satfinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaMetadata {
    Standard,
    Ordered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
// The shared prefix makes each variant's accepted immediate range explicit in catalog data.
#[allow(clippy::enum_variant_names)]
pub enum SparseMmaSelector {
    ImmediateZeroThroughThree,
    ImmediateZeroOrOne,
    ImmediateZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaParticipation {
    AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
}

/// Rust `C, A, B, metadata, selector` shape used by the importer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaAdapter {
    C2U32A2U32B2U32MetadataU32SelectorU32ToD2U32,
    C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32,
    C4F32A2U32B2U32MetadataU32SelectorU32ToD4F32,
    C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
    C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32,
    C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32,
}

/// LLVM `A, B, C, metadata, selector` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaLlvmAdapter {
    A2V2F16B2V2F16C2V2F16MetadataI32SelectorI32ToD2V2F16,
    A4V2F16B4V2F16C2V2F16MetadataI32SelectorI32ToD2V2F16,
    A2V2F16B2V2F16C4F32MetadataI32SelectorI32ToD4F32,
    A4V2F16B4V2F16C4F32MetadataI32SelectorI32ToD4F32,
    A2I32B2I32C2V2F16MetadataI32SelectorI32ToD2V2F16,
    A4I32B4I32C2V2F16MetadataI32SelectorI32ToD2V2F16,
    A2I32B2I32C4F32MetadataI32SelectorI32ToD4F32,
    A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32,
    A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32,
    A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32,
}

/// Where the stable `cuda_device::wmma` callable is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaCompatibilitySource {
    GeneratedStub,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixAdapter {
    SingleResultDirect,
    MultipleResultsToArray,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldmatrix_register_count_includes_the_shape_width() {
        let variant = |shape, multiplicity| LdmatrixVariant {
            shape,
            multiplicity,
            layout: LdmatrixLayout::Normal,
            element: LdmatrixElement::B8,
            state_space: LdmatrixStateSpace::Shared,
        };

        assert_eq!(
            variant(LdmatrixShape::M8n16, LdmatrixMultiplicity::X4).register_count(),
            4
        );
        assert_eq!(
            variant(LdmatrixShape::M16n16, LdmatrixMultiplicity::X1).register_count(),
            2
        );
        assert_eq!(
            variant(LdmatrixShape::M16n16, LdmatrixMultiplicity::X2).register_count(),
            4
        );
    }

    #[test]
    fn blackwell_ldmatrix_address_contracts_keep_readable_widths_distinct() {
        assert_eq!(
            serde_json::from_str::<LdmatrixAddressContract>(
                r#""warp_lane_addresses_mapped_by_multiplicity_sixteen_byte_aligned_sixteen_bytes_readable""#
            )
            .unwrap(),
            LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadable
        );
        assert_eq!(
            serde_json::from_str::<LdmatrixAddressContract>(
                r#""warp_lane_addresses_mapped_by_multiplicity_sixteen_byte_aligned_thirty_two_bytes_readable""#
            )
            .unwrap(),
            LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedThirtyTwoBytesReadable
        );
    }

    #[test]
    fn movmatrix_contract_rejects_open_ended_policy() {
        let valid = r#"
participation = "all_warp_lanes_same_instruction_no_exited_lanes"
adapter = "packed_b16x2_u32_to_packed_b16x2_u32"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<Movmatrix>(valid).unwrap();
        assert_eq!(
            parsed.participation,
            MovmatrixParticipation::AllWarpLanesSameInstructionNoExitedLanes
        );

        for invalid in [
            valid.replace(
                "all_warp_lanes_same_instruction_no_exited_lanes",
                "participating_lanes",
            ),
            valid.replace("packed_b16x2_u32_to_packed_b16x2_u32", "direct_u32"),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Movmatrix>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn register_mma_contract_parses_unsigned_k16_generated_stub() {
        let valid = r#"
shape = "m16n8k16"
operation = "multiply"
accumulator = "s32"
a_element = "s8"
b_element = "u8"
a_layout = "row"
b_layout = "col"
overflow = "satfinite"
participation = "all_warp_lanes_same_instruction_and_qualifiers_no_exited_lanes"
adapter = "c4_i32_a2_u32_b1_u32_to_d4_i32"
compatibility_source = "generated_stub"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<RegisterMma>(valid).unwrap();
        assert_eq!(parsed.b_element, RegisterMmaElement::U8);
        assert_eq!(parsed.adapter, RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32);
        assert_eq!(
            parsed.compatibility_source,
            RegisterMmaCompatibilitySource::GeneratedStub
        );

        for invalid in [
            valid.replace("b_element = \"u8\"", "b_element = \"i8\""),
            valid.replace(
                "adapter = \"c4_i32_a2_u32_b1_u32_to_d4_i32\"",
                "adapter = \"direct\"",
            ),
            valid.replace(
                "compatibility_source = \"generated_stub\"",
                "compatibility_source = \"automatic\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<RegisterMma>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn sparse_mma_contract_closes_the_selector_and_metadata_modes() {
        let valid = r#"
shape = "m16n8k32"
accumulator = "s32"
a_element = "s8"
b_element = "u8"
a_layout = "row"
b_layout = "col"
overflow = "satfinite"
metadata = "standard"
selector = "immediate_zero_or_one"
participation = "all_warp_lanes_same_instruction_and_qualifiers_no_exited_lanes"
adapter = "c4_i32_a2_u32_b2_u32_metadata_u32_selector_u32_to_d4_i32"
llvm_adapter = "a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32"
compatibility_source = "generated_stub"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<SparseMma>(valid).unwrap();
        assert_eq!(parsed.metadata, SparseMmaMetadata::Standard);
        assert_eq!(parsed.selector, SparseMmaSelector::ImmediateZeroOrOne);

        let ordered = valid.replace("metadata = \"standard\"", "metadata = \"ordered\"");
        assert_eq!(
            toml::from_str::<SparseMma>(&ordered).unwrap().metadata,
            SparseMmaMetadata::Ordered
        );

        let k64 = ordered
            .replace("shape = \"m16n8k32\"", "shape = \"m16n8k64\"")
            .replace(
                "selector = \"immediate_zero_or_one\"",
                "selector = \"immediate_zero\"",
            )
            .replace(
                "adapter = \"c4_i32_a2_u32_b2_u32_metadata_u32_selector_u32_to_d4_i32\"",
                "adapter = \"c4_i32_a4_u32_b4_u32_metadata_u32_selector_u32_to_d4_i32\"",
            )
            .replace(
                "llvm_adapter = \"a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
                "llvm_adapter = \"a4_i32_b4_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
            );
        let parsed_k64 = toml::from_str::<SparseMma>(&k64).unwrap();
        assert_eq!(parsed_k64.shape, SparseMmaShape::M16n8k64);
        assert_eq!(parsed_k64.selector, SparseMmaSelector::ImmediateZero);
        assert_eq!(
            parsed_k64.adapter,
            SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32
        );
        assert_eq!(
            parsed_k64.llvm_adapter,
            SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32
        );

        let int4 = ordered
            .replace("shape = \"m16n8k32\"", "shape = \"m16n8k64\"")
            .replace("a_element = \"s8\"", "a_element = \"s4\"")
            .replace("b_element = \"u8\"", "b_element = \"u4\"");
        let parsed_int4 = toml::from_str::<SparseMma>(&int4).unwrap();
        assert_eq!(parsed_int4.a_element, SparseMmaElement::S4);
        assert_eq!(parsed_int4.b_element, SparseMmaElement::U4);

        let k128_int4 = int4
            .replace("shape = \"m16n8k64\"", "shape = \"m16n8k128\"")
            .replace(
                "selector = \"immediate_zero_or_one\"",
                "selector = \"immediate_zero\"",
            )
            .replace(
                "adapter = \"c4_i32_a2_u32_b2_u32_metadata_u32_selector_u32_to_d4_i32\"",
                "adapter = \"c4_i32_a4_u32_b4_u32_metadata_u32_selector_u32_to_d4_i32\"",
            )
            .replace(
                "llvm_adapter = \"a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
                "llvm_adapter = \"a4_i32_b4_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
            );
        let parsed_k128_int4 = toml::from_str::<SparseMma>(&k128_int4).unwrap();
        assert_eq!(parsed_k128_int4.shape, SparseMmaShape::M16n8k128);
        assert_eq!(parsed_k128_int4.selector, SparseMmaSelector::ImmediateZero);

        for invalid in [
            valid.replace(
                "selector = \"immediate_zero_or_one\"",
                "selector = \"runtime\"",
            ),
            valid.replace("metadata = \"standard\"", "metadata = \"unreviewed\""),
            valid.replace(
                "llvm_adapter = \"a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
                "llvm_adapter = \"c_then_a_then_b\"",
            ),
            valid.replace("shape = \"m16n8k32\"", "shape = \"m16n8k256\""),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<SparseMma>(&invalid).is_err(), "{invalid}");
        }
    }
}
