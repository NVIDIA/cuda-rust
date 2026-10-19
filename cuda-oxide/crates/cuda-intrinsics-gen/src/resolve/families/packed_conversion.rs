/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedIntrinsic, IntrinsicBackend, IntrinsicSource,
    OverlayBackendLowering, OverlayIntrinsic, PackedConversionAdapter,
    PackedConversionDestinationFormat, PackedConversionFp8Admission, PackedConversionFp8Direction,
    PackedConversionFp8F16x2Admission, PackedConversionFp8Format, PackedConversionRounding,
    PackedConversionSaturation, PackedConversionSourceFormat, RuntimeValidation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

use crate::resolve::guards::*;

pub(in crate::resolve) struct PackedConversionRecipe {
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) rust_name: &'static str,
    pub(in crate::resolve) compatibility_path: &'static str,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) llvm_result: &'static str,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn packed_conversion_recipe(
    conversion: &crate::model::PackedConversion,
) -> Option<PackedConversionRecipe> {
    match conversion.source_format {
        PackedConversionSourceFormat::F32x2 => packed_conversion_recipe_f32x2(conversion),
        PackedConversionSourceFormat::E4m3x2
        | PackedConversionSourceFormat::E5m2x2
        | PackedConversionSourceFormat::F16x2 => packed_conversion_recipe_fp8_f16x2(conversion),
    }
}

/// Recipes for the packed FP8 conversions whose other side is `f16x2`.
///
/// Keyed on source as well as destination: unpacking to `f16x2` shares its
/// destination, rounding, and saturation with the scalar-`f32` recipe for
/// `cvt.rn.f16x2.f32`, so the source format is what separates them.
pub(in crate::resolve) fn packed_conversion_recipe_fp8_f16x2(
    conversion: &crate::model::PackedConversion,
) -> Option<PackedConversionRecipe> {
    match (
        conversion.source_format,
        conversion.destination_format,
        conversion.rounding,
        conversion.saturation,
    ) {
        (
            PackedConversionSourceFormat::F16x2,
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_e4m3x2_f16x2",
            abi_id: "i0822",
            operation_key: "packed.convert.f16x2.e4m3x2.nearest_even.satfinite",
            rust_name: "cvt_rn_satfinite_e4m3x2_f16x2",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_e4m3x2_f16x2",
            dialect_op_type: "CvtRnSatfiniteE4m3x2F16x2Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_e4m3x2_f16x2",
            source_record: "int_nvvm_f16x2_to_e4m3x2_rn",
            llvm_symbol: "llvm.nvvm.f16x2.to.e4m3x2.rn",
            llvm_result: "i16",
            summary: "Converts packed f16x2 to packed e4m3x2 with nearest-even finite saturation, preserving half order.",
        }),
        (
            PackedConversionSourceFormat::F16x2,
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_relu_e4m3x2_f16x2",
            abi_id: "i0823",
            operation_key: "packed.convert.f16x2.e4m3x2.nearest_even.satfinite.relu",
            rust_name: "cvt_rn_satfinite_relu_e4m3x2_f16x2",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_relu_e4m3x2_f16x2",
            dialect_op_type: "CvtRnSatfiniteReluE4m3x2F16x2Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_relu_e4m3x2_f16x2",
            source_record: "int_nvvm_f16x2_to_e4m3x2_rn_relu",
            llvm_symbol: "llvm.nvvm.f16x2.to.e4m3x2.rn.relu",
            llvm_result: "i16",
            summary: "Converts packed f16x2 to packed e4m3x2 with nearest-even finite saturation and ReLU, preserving half order.",
        }),
        (
            PackedConversionSourceFormat::F16x2,
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_e5m2x2_f16x2",
            abi_id: "i0824",
            operation_key: "packed.convert.f16x2.e5m2x2.nearest_even.satfinite",
            rust_name: "cvt_rn_satfinite_e5m2x2_f16x2",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_e5m2x2_f16x2",
            dialect_op_type: "CvtRnSatfiniteE5m2x2F16x2Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_e5m2x2_f16x2",
            source_record: "int_nvvm_f16x2_to_e5m2x2_rn",
            llvm_symbol: "llvm.nvvm.f16x2.to.e5m2x2.rn",
            llvm_result: "i16",
            summary: "Converts packed f16x2 to packed e5m2x2 with nearest-even finite saturation, preserving half order.",
        }),
        (
            PackedConversionSourceFormat::F16x2,
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_relu_e5m2x2_f16x2",
            abi_id: "i0825",
            operation_key: "packed.convert.f16x2.e5m2x2.nearest_even.satfinite.relu",
            rust_name: "cvt_rn_satfinite_relu_e5m2x2_f16x2",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_relu_e5m2x2_f16x2",
            dialect_op_type: "CvtRnSatfiniteReluE5m2x2F16x2Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_relu_e5m2x2_f16x2",
            source_record: "int_nvvm_f16x2_to_e5m2x2_rn_relu",
            llvm_symbol: "llvm.nvvm.f16x2.to.e5m2x2.rn.relu",
            llvm_result: "i16",
            summary: "Converts packed f16x2 to packed e5m2x2 with nearest-even finite saturation and ReLU, preserving half order.",
        }),
        (
            PackedConversionSourceFormat::E4m3x2,
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_f16x2_e4m3x2",
            abi_id: "i0826",
            operation_key: "packed.convert.e4m3x2.f16x2.nearest_even",
            rust_name: "cvt_rn_f16x2_e4m3x2",
            compatibility_path: "cuda_device::convert::cvt_rn_f16x2_e4m3x2",
            dialect_op_type: "CvtRnF16x2E4m3x2Op",
            dialect_op_name: "nvvm.cvt_rn_f16x2_e4m3x2",
            source_record: "int_nvvm_e4m3x2_to_f16x2_rn",
            llvm_symbol: "llvm.nvvm.e4m3x2.to.f16x2.rn",
            llvm_result: "v2f16",
            summary: "Converts packed e4m3x2 to packed f16x2, preserving byte order.",
        }),
        (
            PackedConversionSourceFormat::E4m3x2,
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_relu_f16x2_e4m3x2",
            abi_id: "i0827",
            operation_key: "packed.convert.e4m3x2.f16x2.nearest_even.relu",
            rust_name: "cvt_rn_relu_f16x2_e4m3x2",
            compatibility_path: "cuda_device::convert::cvt_rn_relu_f16x2_e4m3x2",
            dialect_op_type: "CvtRnReluF16x2E4m3x2Op",
            dialect_op_name: "nvvm.cvt_rn_relu_f16x2_e4m3x2",
            source_record: "int_nvvm_e4m3x2_to_f16x2_rn_relu",
            llvm_symbol: "llvm.nvvm.e4m3x2.to.f16x2.rn.relu",
            llvm_result: "v2f16",
            summary: "Converts packed e4m3x2 to packed f16x2 with ReLU, preserving byte order.",
        }),
        (
            PackedConversionSourceFormat::E5m2x2,
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_f16x2_e5m2x2",
            abi_id: "i0828",
            operation_key: "packed.convert.e5m2x2.f16x2.nearest_even",
            rust_name: "cvt_rn_f16x2_e5m2x2",
            compatibility_path: "cuda_device::convert::cvt_rn_f16x2_e5m2x2",
            dialect_op_type: "CvtRnF16x2E5m2x2Op",
            dialect_op_name: "nvvm.cvt_rn_f16x2_e5m2x2",
            source_record: "int_nvvm_e5m2x2_to_f16x2_rn",
            llvm_symbol: "llvm.nvvm.e5m2x2.to.f16x2.rn",
            llvm_result: "v2f16",
            summary: "Converts packed e5m2x2 to packed f16x2, preserving byte order.",
        }),
        (
            PackedConversionSourceFormat::E5m2x2,
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_relu_f16x2_e5m2x2",
            abi_id: "i0829",
            operation_key: "packed.convert.e5m2x2.f16x2.nearest_even.relu",
            rust_name: "cvt_rn_relu_f16x2_e5m2x2",
            compatibility_path: "cuda_device::convert::cvt_rn_relu_f16x2_e5m2x2",
            dialect_op_type: "CvtRnReluF16x2E5m2x2Op",
            dialect_op_name: "nvvm.cvt_rn_relu_f16x2_e5m2x2",
            source_record: "int_nvvm_e5m2x2_to_f16x2_rn_relu",
            llvm_symbol: "llvm.nvvm.e5m2x2.to.f16x2.rn.relu",
            llvm_result: "v2f16",
            summary: "Converts packed e5m2x2 to packed f16x2 with ReLU, preserving byte order.",
        }),
        _ => None,
    }
}

pub(in crate::resolve) fn packed_conversion_recipe_f32x2(
    conversion: &crate::model::PackedConversion,
) -> Option<PackedConversionRecipe> {
    match (
        conversion.destination_format,
        conversion.rounding,
        conversion.saturation,
    ) {
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_f32x2_bf16x2",
            abi_id: "i0071",
            operation_key: "packed.convert.f32x2.bf16x2.nearest_even",
            rust_name: "cvt_f32x2_bf16x2",
            compatibility_path: "cuda_device::convert::cvt_bf16x2_f32",
            dialect_op_type: "CvtF32x2Bf16x2Op",
            dialect_op_name: "nvvm.cvt_f32x2_bf16x2",
            source_record: "int_nvvm_ff2bf16x2_rn",
            llvm_symbol: "llvm.nvvm.ff2bf16x2.rn",
            llvm_result: "v2bf16",
            summary: "Converts two f32 values to packed bf16x2 with the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_f16x2_f32",
            abi_id: "i0081",
            operation_key: "packed.convert.f32x2.f16x2.nearest_even",
            rust_name: "cvt_f16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_f16x2_f32",
            dialect_op_type: "CvtF16x2F32Op",
            dialect_op_name: "nvvm.cvt_f16x2_f32",
            source_record: "int_nvvm_ff2f16x2_rn",
            llvm_symbol: "llvm.nvvm.ff2f16x2.rn",
            llvm_result: "v2f16",
            summary: "Converts two f32 values to packed f16x2 with nearest-even rounding and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::TowardZero,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rz_f16x2_f32",
            abi_id: "i0082",
            operation_key: "packed.convert.f32x2.f16x2.toward_zero",
            rust_name: "cvt_rz_f16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rz_f16x2_f32",
            dialect_op_type: "CvtRzF16x2F32Op",
            dialect_op_name: "nvvm.cvt_rz_f16x2_f32",
            source_record: "int_nvvm_ff2f16x2_rz",
            llvm_symbol: "llvm.nvvm.ff2f16x2.rz",
            llvm_result: "v2f16",
            summary: "Converts two f32 values to packed f16x2 with toward-zero rounding and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_relu_f16x2_f32",
            abi_id: "i0083",
            operation_key: "packed.convert.f32x2.f16x2.nearest_even.relu",
            rust_name: "cvt_rn_relu_f16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_relu_f16x2_f32",
            dialect_op_type: "CvtRnReluF16x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_relu_f16x2_f32",
            source_record: "int_nvvm_ff2f16x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff2f16x2.rn.relu",
            llvm_result: "v2f16",
            summary: "Converts two f32 values to packed f16x2 with nearest-even rounding, ReLU, and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_relu_bf16x2_f32",
            abi_id: "i0084",
            operation_key: "packed.convert.f32x2.bf16x2.nearest_even.relu",
            rust_name: "cvt_rn_relu_bf16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_relu_bf16x2_f32",
            dialect_op_type: "CvtRnReluBf16x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_relu_bf16x2_f32",
            source_record: "int_nvvm_ff2bf16x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff2bf16x2.rn.relu",
            llvm_result: "v2bf16",
            summary: "Converts two f32 values to packed bf16x2 with nearest-even rounding, ReLU, and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::TowardZero,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rz_bf16x2_f32",
            abi_id: "i0085",
            operation_key: "packed.convert.f32x2.bf16x2.toward_zero",
            rust_name: "cvt_rz_bf16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rz_bf16x2_f32",
            dialect_op_type: "CvtRzBf16x2F32Op",
            dialect_op_name: "nvvm.cvt_rz_bf16x2_f32",
            source_record: "int_nvvm_ff2bf16x2_rz",
            llvm_symbol: "llvm.nvvm.ff2bf16x2.rz",
            llvm_result: "v2bf16",
            summary: "Converts two f32 values to packed bf16x2 with toward-zero rounding and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_e4m3x2_f32",
            abi_id: "i0259",
            operation_key: "packed.convert.f32x2.e4m3x2.nearest_even.satfinite",
            rust_name: "cvt_rn_satfinite_e4m3x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32",
            dialect_op_type: "CvtRnSatfiniteE4m3x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_e4m3x2_f32",
            source_record: "int_nvvm_ff_to_e4m3x2_rn",
            llvm_symbol: "llvm.nvvm.ff.to.e4m3x2.rn",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e4m3x2 with nearest-even finite saturation and the first argument in the low byte.",
        }),
        (
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_relu_e4m3x2_f32",
            abi_id: "i0260",
            operation_key: "packed.convert.f32x2.e4m3x2.nearest_even.satfinite.relu",
            rust_name: "cvt_rn_satfinite_relu_e4m3x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_relu_e4m3x2_f32",
            dialect_op_type: "CvtRnSatfiniteReluE4m3x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_relu_e4m3x2_f32",
            source_record: "int_nvvm_ff_to_e4m3x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff.to.e4m3x2.rn.relu",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e4m3x2 with nearest-even finite saturation, ReLU, and the first argument in the low byte.",
        }),
        (
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_e5m2x2_f32",
            abi_id: "i0261",
            operation_key: "packed.convert.f32x2.e5m2x2.nearest_even.satfinite",
            rust_name: "cvt_rn_satfinite_e5m2x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_e5m2x2_f32",
            dialect_op_type: "CvtRnSatfiniteE5m2x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_e5m2x2_f32",
            source_record: "int_nvvm_ff_to_e5m2x2_rn",
            llvm_symbol: "llvm.nvvm.ff.to.e5m2x2.rn",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e5m2x2 with nearest-even finite saturation and the first argument in the low byte.",
        }),
        (
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_relu_e5m2x2_f32",
            abi_id: "i0262",
            operation_key: "packed.convert.f32x2.e5m2x2.nearest_even.satfinite.relu",
            rust_name: "cvt_rn_satfinite_relu_e5m2x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_relu_e5m2x2_f32",
            dialect_op_type: "CvtRnSatfiniteReluE5m2x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_relu_e5m2x2_f32",
            source_record: "int_nvvm_ff_to_e5m2x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff.to.e5m2x2.rn.relu",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e5m2x2 with nearest-even finite saturation, ReLU, and the first argument in the low byte.",
        }),
        _ => None,
    }
}

pub(in crate::resolve) fn packed_conversion_ptx_modifiers(
    conversion: &crate::model::PackedConversion,
) -> Vec<&'static str> {
    let rounding = match conversion.rounding {
        PackedConversionRounding::NearestEven => "rn",
        PackedConversionRounding::TowardZero => "rz",
    };
    let format = match conversion.destination_format {
        PackedConversionDestinationFormat::Bf16x2 => "bf16x2",
        PackedConversionDestinationFormat::E4m3x2 => "e4m3x2",
        PackedConversionDestinationFormat::E5m2x2 => "e5m2x2",
        PackedConversionDestinationFormat::F16x2 => "f16x2",
    };
    let mut modifiers = vec![rounding];
    match conversion.saturation {
        PackedConversionSaturation::None => {}
        PackedConversionSaturation::Relu => modifiers.push("relu"),
        PackedConversionSaturation::Satfinite => modifiers.push("satfinite"),
        PackedConversionSaturation::SatfiniteRelu => modifiers.extend(["satfinite", "relu"]),
    }
    modifiers.extend([format, conversion.source_format.ptx_token()]);
    modifiers
}

/// Whether either side of the conversion is a packed FP8 format.
pub(in crate::resolve) fn packed_conversion_uses_fp8(
    conversion: &crate::model::PackedConversion,
) -> bool {
    matches!(
        conversion.source_format,
        PackedConversionSourceFormat::E4m3x2 | PackedConversionSourceFormat::E5m2x2
    ) || matches!(
        conversion.destination_format,
        PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2
    )
}

/// Source operand types, in Rust, dialect, and LLVM spellings.
///
/// A packed source arrives in one register; `f32x2` names two scalar operands.
pub(in crate::resolve) fn packed_conversion_source_types(
    conversion: &crate::model::PackedConversion,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    match conversion.source_format {
        PackedConversionSourceFormat::F32x2 => (
            vec!["f32".into(), "f32".into()],
            vec!["f32".into(), "f32".into()],
            vec!["f32".into(), "f32".into()],
        ),
        PackedConversionSourceFormat::F16x2 => {
            (vec!["u32".into()], vec!["i32".into()], vec!["v2f16".into()])
        }
        PackedConversionSourceFormat::E4m3x2 | PackedConversionSourceFormat::E5m2x2 => {
            (vec!["u16".into()], vec!["i16".into()], vec!["i16".into()])
        }
    }
}

pub(in crate::resolve) fn packed_conversion_result_width(
    conversion: &crate::model::PackedConversion,
) -> u32 {
    match conversion.destination_format {
        PackedConversionDestinationFormat::Bf16x2 | PackedConversionDestinationFormat::F16x2 => 32,
        PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2 => 16,
    }
}

pub(in crate::resolve) fn packed_conversion_floor(
    conversion: &crate::model::PackedConversion,
) -> (&'static str, &'static str) {
    // FP8 on either side carries the Ada floor, including when FP8 is the
    // source and the destination is the older `f16x2`.
    if packed_conversion_uses_fp8(conversion) {
        return ("8.1", "sm_89");
    }
    match conversion.destination_format {
        PackedConversionDestinationFormat::Bf16x2 | PackedConversionDestinationFormat::F16x2 => {
            ("7.0", "sm_80")
        }
        PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2 => {
            ("8.1", "sm_89")
        }
    }
}

pub(in crate::resolve) fn packed_conversion_backend_mechanism(
    conversion: &crate::model::PackedConversion,
    backend: IntrinsicBackend,
) -> BackendLoweringMechanism {
    match (packed_conversion_uses_typed_nvvm(conversion), backend) {
        (true, IntrinsicBackend::LlvmNvptx) => BackendLoweringMechanism::TypedNvvm,
        _ => BackendLoweringMechanism::InlinePtx,
    }
}

/// Whether the conversion lowers through a typed NVVM intrinsic call.
///
/// Only the scalar-`f32` pair does. The `f16x2` sources and destinations carry
/// their halves in one integer register on the MIR side, while the typed
/// intrinsics are declared over `<2 x half>`; routing them through inline PTX
/// avoids a packed-half MIR type for no change in emitted code. The recorded
/// evidence assembles both routes to a byte-identical cubin, so the typed route
/// can be switched on later without changing the PTX contract.
pub(in crate::resolve) fn packed_conversion_uses_typed_nvvm(
    conversion: &crate::model::PackedConversion,
) -> bool {
    conversion.source_format == PackedConversionSourceFormat::F32x2
        && matches!(
            conversion.destination_format,
            PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2
        )
}

pub(in crate::resolve) fn packed_conversion_lowering(
    conversion: &crate::model::PackedConversion,
) -> &'static str {
    if packed_conversion_uses_typed_nvvm(conversion) {
        "generated_packed_conversion_backend"
    } else {
        "generated_packed_conversion_inline_ptx"
    }
}

pub(in crate::resolve) fn expand_packed_conversion_fp8_admission(
    admission: &PackedConversionFp8Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "FP8 conversion runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        admission.destination_formats
            == [
                PackedConversionDestinationFormat::E4m3x2,
                PackedConversionDestinationFormat::E5m2x2,
            ],
        "compact FP8 conversion admission must list the canonical two formats"
    );
    ensure!(
        admission.saturations
            == [
                PackedConversionSaturation::Satfinite,
                PackedConversionSaturation::SatfiniteRelu,
            ],
        "compact FP8 conversion admission must list base and ReLU finite saturation"
    );
    ensure!(
        admission.product_count
            == admission
                .destination_formats
                .len()
                .checked_mul(admission.saturations.len())
                .context("compact FP8 conversion product count overflow")?
            && admission.product_count == 4,
        "compact FP8 conversion product_count must be exactly 4"
    );

    let mut records = Vec::with_capacity(admission.product_count);
    for &destination_format in &admission.destination_formats {
        for &saturation in &admission.saturations {
            let conversion = crate::model::PackedConversion {
                source_format: PackedConversionSourceFormat::F32x2,
                destination_format,
                rounding: PackedConversionRounding::NearestEven,
                saturation,
                adapter: PackedConversionAdapter::ReverseHighLowOperands,
            };
            records.push(packed_conversion_overlay_record(
                conversion,
                &admission.llvm_evidence_profile,
                &admission.libnvvm_evidence_profile,
            )?);
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

pub(in crate::resolve) fn expand_packed_conversion_fp8_f16x2_admission(
    admission: &PackedConversionFp8F16x2Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "FP8 f16x2 conversion runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        admission.fp8_formats
            == [
                PackedConversionFp8Format::E4m3x2,
                PackedConversionFp8Format::E5m2x2,
            ],
        "compact FP8 f16x2 conversion admission must list the canonical two formats"
    );
    ensure!(
        admission.directions
            == [
                PackedConversionFp8Direction::Pack,
                PackedConversionFp8Direction::Unpack,
            ],
        "compact FP8 f16x2 conversion admission must list both conversion directions"
    );
    ensure!(
        admission.relu_variants,
        "compact FP8 f16x2 conversion admission must admit the ReLU variants"
    );
    ensure!(
        admission.product_count
            == admission
                .fp8_formats
                .len()
                .checked_mul(admission.directions.len())
                .and_then(|count| count.checked_mul(2))
                .context("compact FP8 f16x2 conversion product count overflow")?
            && admission.product_count == 8,
        "compact FP8 f16x2 conversion product_count must be exactly 8"
    );

    let mut records = Vec::with_capacity(admission.product_count);
    for &fp8_format in &admission.fp8_formats {
        for &direction in &admission.directions {
            for relu in [false, true] {
                let conversion = match direction {
                    // Narrowing to FP8 always saturates to finite.
                    PackedConversionFp8Direction::Pack => crate::model::PackedConversion {
                        source_format: PackedConversionSourceFormat::F16x2,
                        destination_format: match fp8_format {
                            PackedConversionFp8Format::E4m3x2 => {
                                PackedConversionDestinationFormat::E4m3x2
                            }
                            PackedConversionFp8Format::E5m2x2 => {
                                PackedConversionDestinationFormat::E5m2x2
                            }
                        },
                        rounding: PackedConversionRounding::NearestEven,
                        saturation: if relu {
                            PackedConversionSaturation::SatfiniteRelu
                        } else {
                            PackedConversionSaturation::Satfinite
                        },
                        adapter: PackedConversionAdapter::Identity,
                    },
                    // Widening back to f16x2 is exact, so it carries no
                    // saturation modifier of its own.
                    PackedConversionFp8Direction::Unpack => crate::model::PackedConversion {
                        source_format: match fp8_format {
                            PackedConversionFp8Format::E4m3x2 => {
                                PackedConversionSourceFormat::E4m3x2
                            }
                            PackedConversionFp8Format::E5m2x2 => {
                                PackedConversionSourceFormat::E5m2x2
                            }
                        },
                        destination_format: PackedConversionDestinationFormat::F16x2,
                        rounding: PackedConversionRounding::NearestEven,
                        saturation: if relu {
                            PackedConversionSaturation::Relu
                        } else {
                            PackedConversionSaturation::None
                        },
                        adapter: PackedConversionAdapter::Identity,
                    },
                };
                records.push(packed_conversion_overlay_record(
                    conversion,
                    &admission.llvm_evidence_profile,
                    &admission.libnvvm_evidence_profile,
                )?);
            }
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

pub(in crate::resolve) fn packed_conversion_overlay_record(
    conversion: crate::model::PackedConversion,
    llvm_evidence_profile: &str,
    libnvvm_evidence_profile: &str,
) -> Result<OverlayIntrinsic> {
    let recipe = packed_conversion_recipe(&conversion)
        .context("compact FP8 conversion is outside the closed recipe set")?;
    let result_width = packed_conversion_result_width(&conversion);
    let rust_result = format!("u{result_width}");
    let dialect_result = format!("i{result_width}");
    let (minimum_ptx, minimum_sm) = packed_conversion_floor(&conversion);
    let (rust_arguments, dialect_operands, llvm_arguments) =
        packed_conversion_source_types(&conversion);
    // `cvt` writes one destination and reads every source operand.
    let ptx_operands = conversion
        .source_format
        .operand_count()
        .checked_add(1)
        .context("packed-conversion operand count overflow")?;
    Ok(OverlayIntrinsic {
        id: recipe.id.into(),
        abi_id: String::new(),
        operation_key: recipe.operation_key.into(),
        family: "packed_conversion".into(),
        source: None,
        source_record: Some(recipe.source_record.into()),
        rust_module: "convert".into(),
        rust_name: recipe.rust_name.into(),
        rust_arguments,
        rust_result: rust_result.clone(),
        safe: true,
        must_use: false,
        safe_allowlist_reason: Some("This conversion has no caller obligations.".into()),
        public_rust_path: format!("cuda_intrinsics::convert::{}", recipe.rust_name),
        compatibility_rust_paths: vec![recipe.compatibility_path.into()],
        dialect_op_type: recipe.dialect_op_type.into(),
        dialect_op_name: recipe.dialect_op_name.into(),
        dialect_operands,
        dialect_results: vec![dialect_result],
        llvm_symbol: Some(recipe.llvm_symbol.into()),
        resolved_llvm_symbol: None,
        llvm_arguments,
        llvm_results: vec![recipe.llvm_result.into()],
        pure: true,
        memory: "none".into(),
        convergent: false,
        execution_scope: "thread".into(),
        minimum_ptx: minimum_ptx.into(),
        minimum_sm: Some(minimum_sm.into()),
        ptx_result: rust_result,
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: "9.7.9.22 Data Movement and Conversion Instructions: cvt".into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt".into(),
        lowering: packed_conversion_lowering(&conversion).into(),
        backend_lowerings: [
            (IntrinsicBackend::LlvmNvptx, llvm_evidence_profile),
            (IntrinsicBackend::LibNvvm, libnvvm_evidence_profile),
        ]
        .into_iter()
        .map(|(backend, evidence_profile)| OverlayBackendLowering {
            backend,
            mechanism: packed_conversion_backend_mechanism(&conversion, backend),
            evidence_profile: evidence_profile.into(),
            targets: None,
            minimum_ptx: Some(minimum_ptx.into()),
            minimum_sm: Some(minimum_sm.into()),
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
        packed_conversion: Some(conversion.clone()),
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
        tcgen05: None,
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: InstructionPattern {
            mnemonic: "cvt".into(),
            modifiers: packed_conversion_ptx_modifiers(&conversion)
                .into_iter()
                .map(str::to_owned)
                .collect(),
            operands: vec![OperandPattern::Register; ptx_operands],
        },
        summary: recipe.summary.into(),
    })
}

pub(in crate::resolve) fn validate_packed_conversion_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let conversion = policy
        .packed_conversion
        .as_ref()
        .with_context(|| format!("{} has no closed packed-conversion contract", policy.id))?;
    // The adapter is a function of the source arity: two scalar operands are
    // reversed so the first Rust argument lands in the low half, while a single
    // packed operand is forwarded unchanged.
    let expected_adapter = match conversion.source_format {
        PackedConversionSourceFormat::F32x2 => PackedConversionAdapter::ReverseHighLowOperands,
        PackedConversionSourceFormat::E4m3x2
        | PackedConversionSourceFormat::E5m2x2
        | PackedConversionSourceFormat::F16x2 => PackedConversionAdapter::Identity,
    };
    ensure!(
        conversion.adapter == expected_adapter,
        "{} requests an unsupported packed-conversion source or adapter",
        policy.id
    );
    let recipe = packed_conversion_recipe(conversion).with_context(|| {
        format!(
            "{} requests an unsupported packed-conversion source, destination, rounding, or saturation combination",
            policy.id
        )
    })?;
    let result_width = packed_conversion_result_width(conversion);
    let rust_result = format!("u{result_width}");
    let dialect_result = format!("i{result_width}");
    let (minimum_ptx, minimum_sm) = packed_conversion_floor(conversion);
    let (expected_rust_arguments, expected_dialect_operands, expected_llvm_arguments) =
        packed_conversion_source_types(conversion);
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && source
                == &IntrinsicSource::LlvmImported {
                    source_record: recipe.source_record.into(),
                }
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments == expected_llvm_arguments
            && policy.llvm_results == [recipe.llvm_result],
        "{} packed-conversion identity or LLVM source changed",
        policy.id
    );
    let declaration = declaration.context("packed conversion has no imported declaration")?;
    ensure!(
        declaration.properties == ["IntrNoCreateUndefOrPoison", "IntrNoMem", "IntrSpeculatable"]
            && declaration.selections.is_empty(),
        "{} selectionless packed-conversion declaration changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "convert"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == expected_rust_arguments
            && policy.rust_result == rust_result
            && policy.safe
            && !policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            && policy.public_rust_path == format!("cuda_intrinsics::convert::{}", recipe.rust_name)
            && policy.compatibility_rust_paths == [recipe.compatibility_path],
        "{} must preserve its safe non-must-use conversion API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == expected_dialect_operands
            && policy.dialect_results == [dialect_result.as_str()]
            && policy.lowering == packed_conversion_lowering(conversion),
        "{} is outside the closed packed-conversion dialect and lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == minimum_ptx
            && policy.minimum_sm.as_deref() == Some(minimum_sm)
            && policy.ptx_result == rust_result
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.9.22 Data Movement and Conversion Instructions: cvt"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt",
        "{} packed-conversion effects, carrier, target floor, or PTX provenance disagree",
        policy.id
    );
    let expected_ptx_operands = vec![
        OperandPattern::Register;
        conversion
            .source_format
            .operand_count()
            .checked_add(1)
            .context("packed-conversion operand count overflow")?
    ];
    ensure!(
        policy.expected_ptx.mnemonic == "cvt"
            && policy.expected_ptx.modifiers == packed_conversion_ptx_modifiers(conversion)
            && policy.expected_ptx.operands == expected_ptx_operands,
        "{} expected PTX does not match its closed conversion recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} packed-conversion summary does not match its closed recipe",
        policy.id
    );
    let backend_pairs = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect::<BTreeSet<_>>();
    let expected_pairs = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
        .map(|backend| {
            (
                backend,
                packed_conversion_backend_mechanism(conversion, backend),
            )
        })
        .into_iter()
        .collect::<BTreeSet<_>>();
    ensure!(
        policy.backend_lowerings.len() == 2 && backend_pairs == expected_pairs,
        "{} must define exactly the reviewed packed-conversion backend routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        ensure!(
            lowering.minimum_ptx.as_deref() == Some(minimum_ptx)
                && lowering.minimum_sm.as_deref() == Some(minimum_sm)
                && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its exact packed-conversion floor",
            policy.id,
            lowering.backend
        );
    }
    ensure_no_other_family_contract(policy, "packed conversion")?;
    Ok(())
}
