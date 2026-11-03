/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogIntrinsic, ExtendedMinMaxFormat, ExtendedMinMaxNan,
    ExtendedMinMaxOperation, ExtendedMinMaxSubnormal, IntegerMinMaxFormat, IntegerMinMaxOperation,
    IntrinsicBackend, ScalarArithmeticFormat, ScalarArithmeticOperation, ScalarArithmeticRounding,
    ScalarArithmeticSaturation, ScalarArithmeticSubnormal, ScalarConversionRounding,
    ScalarConversionSaturation, ScalarMathFormat, ScalarMathOperation, ScalarMathPrecision,
    ScalarMathSubnormal,
};

pub(in crate::render) fn scalar_conversion_rounding_attr(
    record: &CatalogIntrinsic,
) -> &'static str {
    match record
        .scalar_conversion
        .as_ref()
        .expect("scalar-conversion contract")
        .rounding
    {
        ScalarConversionRounding::NearestAway => "ScalarConversionRoundingAttr::NearestAway",
        ScalarConversionRounding::NearestEven => "ScalarConversionRoundingAttr::NearestEven",
        ScalarConversionRounding::TowardZero => "ScalarConversionRoundingAttr::TowardZero",
    }
}

pub(in crate::render) fn scalar_conversion_saturation_attr(
    record: &CatalogIntrinsic,
) -> &'static str {
    match record
        .scalar_conversion
        .as_ref()
        .expect("scalar-conversion contract")
        .saturation
    {
        ScalarConversionSaturation::None => "ScalarConversionSaturationAttr::None",
        ScalarConversionSaturation::Relu => "ScalarConversionSaturationAttr::Relu",
        ScalarConversionSaturation::Satfinite => "ScalarConversionSaturationAttr::Satfinite",
        ScalarConversionSaturation::ReluSatfinite => {
            "ScalarConversionSaturationAttr::ReluSatfinite"
        }
    }
}

pub(in crate::render) fn scalar_conversion_ptx_mnemonic(record: &CatalogIntrinsic) -> String {
    format!(
        "{}.{}",
        record.expected_ptx.mnemonic,
        record.expected_ptx.modifiers.join(".")
    )
}

pub(in crate::render) fn extended_minmax_contract(
    record: &CatalogIntrinsic,
) -> &crate::model::ExtendedMinMax {
    record
        .extended_minmax
        .as_ref()
        .expect("extended-minmax contract")
}

pub(in crate::render) fn extended_minmax_format_attr(record: &CatalogIntrinsic) -> &'static str {
    match extended_minmax_contract(record).format {
        ExtendedMinMaxFormat::F32 => "ExtendedMinMaxFormatAttr::F32",
        ExtendedMinMaxFormat::F16 => "ExtendedMinMaxFormatAttr::F16",
        ExtendedMinMaxFormat::Bf16 => "ExtendedMinMaxFormatAttr::Bf16",
        ExtendedMinMaxFormat::F16x2 => "ExtendedMinMaxFormatAttr::F16x2",
        ExtendedMinMaxFormat::Bf16x2 => "ExtendedMinMaxFormatAttr::Bf16x2",
    }
}

pub(in crate::render) fn extended_minmax_operation_attr(record: &CatalogIntrinsic) -> &'static str {
    match extended_minmax_contract(record).operation {
        ExtendedMinMaxOperation::Min => "ExtendedMinMaxOperationAttr::Min",
        ExtendedMinMaxOperation::Max => "ExtendedMinMaxOperationAttr::Max",
    }
}

pub(in crate::render) fn extended_minmax_subnormal_attr(record: &CatalogIntrinsic) -> &'static str {
    match extended_minmax_contract(record).subnormal {
        ExtendedMinMaxSubnormal::Preserve => "ExtendedMinMaxSubnormalAttr::Preserve",
        ExtendedMinMaxSubnormal::Ftz => "ExtendedMinMaxSubnormalAttr::Ftz",
    }
}

pub(in crate::render) fn extended_minmax_nan_attr(record: &CatalogIntrinsic) -> &'static str {
    match extended_minmax_contract(record).nan {
        ExtendedMinMaxNan::Number => "ExtendedMinMaxNanAttr::Number",
        ExtendedMinMaxNan::Nan => "ExtendedMinMaxNanAttr::Nan",
    }
}

pub(in crate::render) fn extended_minmax_xorsign_abs_attr(
    record: &CatalogIntrinsic,
) -> &'static str {
    if extended_minmax_contract(record).xorsign_abs {
        "ExtendedMinMaxXorSignAbsAttr::Enabled"
    } else {
        "ExtendedMinMaxXorSignAbsAttr::Disabled"
    }
}

pub(in crate::render) fn extended_minmax_carrier(record: &CatalogIntrinsic) -> &'static str {
    match extended_minmax_contract(record).format {
        ExtendedMinMaxFormat::F32 => "MinMaxCarrier::F32",
        ExtendedMinMaxFormat::F16 | ExtendedMinMaxFormat::Bf16 => "MinMaxCarrier::Half16",
        ExtendedMinMaxFormat::F16x2 | ExtendedMinMaxFormat::Bf16x2 => "MinMaxCarrier::PackedU32",
    }
}

pub(in crate::render) fn extended_minmax_rust_type(record: &CatalogIntrinsic) -> &'static str {
    match extended_minmax_contract(record).format {
        ExtendedMinMaxFormat::F32 => "f32",
        ExtendedMinMaxFormat::F16 | ExtendedMinMaxFormat::Bf16 => "u16",
        ExtendedMinMaxFormat::F16x2 | ExtendedMinMaxFormat::Bf16x2 => "u32",
    }
}

pub(in crate::render) fn extended_minmax_ptx_mnemonic(record: &CatalogIntrinsic) -> String {
    format!(
        "{}.{}",
        record.expected_ptx.mnemonic,
        record.expected_ptx.modifiers.join(".")
    )
}

pub(in crate::render) fn scalar_arithmetic_contract(
    record: &CatalogIntrinsic,
) -> &crate::model::ScalarArithmetic {
    record
        .scalar_arithmetic
        .as_ref()
        .expect("scalar-arithmetic contract")
}

pub(in crate::render) fn scalar_arithmetic_format_attr(record: &CatalogIntrinsic) -> &'static str {
    match scalar_arithmetic_contract(record).format {
        ScalarArithmeticFormat::F32 => "ScalarArithmeticFormatAttr::F32",
        ScalarArithmeticFormat::F64 => "ScalarArithmeticFormatAttr::F64",
    }
}

pub(in crate::render) fn scalar_arithmetic_operation_attr(
    record: &CatalogIntrinsic,
) -> &'static str {
    match scalar_arithmetic_contract(record).operation {
        ScalarArithmeticOperation::Mul => "ScalarArithmeticOperationAttr::Mul",
        ScalarArithmeticOperation::Div => "ScalarArithmeticOperationAttr::Div",
        ScalarArithmeticOperation::Fma => "ScalarArithmeticOperationAttr::Fma",
        ScalarArithmeticOperation::Add => "ScalarArithmeticOperationAttr::Add",
    }
}

pub(in crate::render) fn scalar_arithmetic_rounding_attr(
    record: &CatalogIntrinsic,
) -> &'static str {
    match scalar_arithmetic_contract(record).rounding {
        ScalarArithmeticRounding::Rn => "ScalarArithmeticRoundingAttr::Rn",
        ScalarArithmeticRounding::Rz => "ScalarArithmeticRoundingAttr::Rz",
        ScalarArithmeticRounding::Rm => "ScalarArithmeticRoundingAttr::Rm",
        ScalarArithmeticRounding::Rp => "ScalarArithmeticRoundingAttr::Rp",
    }
}

pub(in crate::render) fn scalar_arithmetic_subnormal_attr(
    record: &CatalogIntrinsic,
) -> &'static str {
    match scalar_arithmetic_contract(record).subnormal {
        ScalarArithmeticSubnormal::Preserve => "ScalarArithmeticSubnormalAttr::Preserve",
        ScalarArithmeticSubnormal::Ftz => "ScalarArithmeticSubnormalAttr::Ftz",
    }
}

pub(in crate::render) fn scalar_arithmetic_saturation_attr(
    record: &CatalogIntrinsic,
) -> &'static str {
    match scalar_arithmetic_contract(record).saturation {
        ScalarArithmeticSaturation::None => "ScalarArithmeticSaturationAttr::None",
        ScalarArithmeticSaturation::Sat => "ScalarArithmeticSaturationAttr::Sat",
    }
}

pub(in crate::render) fn scalar_arithmetic_arity(record: &CatalogIntrinsic) -> usize {
    match scalar_arithmetic_contract(record).operation {
        ScalarArithmeticOperation::Mul
        | ScalarArithmeticOperation::Div
        | ScalarArithmeticOperation::Add => 2,
        ScalarArithmeticOperation::Fma => 3,
    }
}

pub(in crate::render) fn scalar_arithmetic_rust_type(record: &CatalogIntrinsic) -> &'static str {
    match scalar_arithmetic_contract(record).format {
        ScalarArithmeticFormat::F32 => "f32",
        ScalarArithmeticFormat::F64 => "f64",
    }
}

pub(in crate::render) fn scalar_arithmetic_llvm_type(record: &CatalogIntrinsic) -> &'static str {
    match scalar_arithmetic_contract(record).format {
        ScalarArithmeticFormat::F32 => "float",
        ScalarArithmeticFormat::F64 => "double",
    }
}

pub(in crate::render) fn scalar_arithmetic_ptx_mnemonic(record: &CatalogIntrinsic) -> String {
    format!(
        "{}.{}",
        record.expected_ptx.mnemonic,
        record.expected_ptx.modifiers.join(".")
    )
}

pub(in crate::render) fn scalar_arithmetic_llvm_mechanism(
    record: &CatalogIntrinsic,
) -> BackendLoweringMechanism {
    record
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .expect("scalar arithmetic has an LLVM-NVPTX route")
        .mechanism
}

pub(in crate::render) fn scalar_math_contract(
    record: &CatalogIntrinsic,
) -> &crate::model::ScalarMath {
    record.scalar_math.as_ref().expect("scalar-math contract")
}

pub(in crate::render) fn scalar_math_format_attr(record: &CatalogIntrinsic) -> &'static str {
    match scalar_math_contract(record).format {
        ScalarMathFormat::F16 => "ScalarMathFormatAttr::F16",
        ScalarMathFormat::F32 => "ScalarMathFormatAttr::F32",
        ScalarMathFormat::F64 => "ScalarMathFormatAttr::F64",
    }
}

pub(in crate::render) fn scalar_math_operation_attr(record: &CatalogIntrinsic) -> &'static str {
    match scalar_math_contract(record).operation {
        ScalarMathOperation::Sin => "ScalarMathOperationAttr::Sin",
        ScalarMathOperation::Cos => "ScalarMathOperationAttr::Cos",
        ScalarMathOperation::Ex2 => "ScalarMathOperationAttr::Ex2",
        ScalarMathOperation::Lg2 => "ScalarMathOperationAttr::Lg2",
        ScalarMathOperation::Rcp => "ScalarMathOperationAttr::Rcp",
        ScalarMathOperation::Rsqrt => "ScalarMathOperationAttr::Rsqrt",
        ScalarMathOperation::Sqrt => "ScalarMathOperationAttr::Sqrt",
        ScalarMathOperation::Tanh => "ScalarMathOperationAttr::Tanh",
    }
}

pub(in crate::render) fn scalar_math_precision_attr(record: &CatalogIntrinsic) -> &'static str {
    match scalar_math_contract(record).precision {
        ScalarMathPrecision::Approx => "ScalarMathPrecisionAttr::Approx",
        ScalarMathPrecision::Rn => "ScalarMathPrecisionAttr::Rn",
        ScalarMathPrecision::Rz => "ScalarMathPrecisionAttr::Rz",
        ScalarMathPrecision::Rm => "ScalarMathPrecisionAttr::Rm",
        ScalarMathPrecision::Rp => "ScalarMathPrecisionAttr::Rp",
    }
}

pub(in crate::render) fn scalar_math_subnormal_attr(record: &CatalogIntrinsic) -> &'static str {
    match scalar_math_contract(record).subnormal {
        ScalarMathSubnormal::Preserve => "ScalarMathSubnormalAttr::Preserve",
        ScalarMathSubnormal::Ftz => "ScalarMathSubnormalAttr::Ftz",
    }
}

pub(in crate::render) fn scalar_math_llvm_type(record: &CatalogIntrinsic) -> &'static str {
    match scalar_math_contract(record).format {
        ScalarMathFormat::F16 => "i16",
        ScalarMathFormat::F32 => "float",
        ScalarMathFormat::F64 => "double",
    }
}

pub(in crate::render) fn scalar_math_ptx_mnemonic(record: &CatalogIntrinsic) -> String {
    format!(
        "{}.{}",
        record.expected_ptx.mnemonic,
        record.expected_ptx.modifiers.join(".")
    )
}

pub(in crate::render) fn scalar_math_llvm_mechanism(
    record: &CatalogIntrinsic,
) -> BackendLoweringMechanism {
    record
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .expect("scalar math has an LLVM-NVPTX route")
        .mechanism
}

pub(in crate::render) fn integer_minmax_ptx_mnemonic(record: &CatalogIntrinsic) -> &'static str {
    let minmax = record
        .integer_minmax
        .as_ref()
        .expect("integer-min/max record");
    match (minmax.format, minmax.operation, minmax.relu) {
        (IntegerMinMaxFormat::S32, IntegerMinMaxOperation::Min, true) => "min.relu.s32",
        (IntegerMinMaxFormat::S32, IntegerMinMaxOperation::Max, true) => "max.relu.s32",
        (IntegerMinMaxFormat::S16x2, IntegerMinMaxOperation::Min, false) => "min.s16x2",
        (IntegerMinMaxFormat::S16x2, IntegerMinMaxOperation::Max, false) => "max.s16x2",
        (IntegerMinMaxFormat::U16x2, IntegerMinMaxOperation::Min, false) => "min.u16x2",
        (IntegerMinMaxFormat::U16x2, IntegerMinMaxOperation::Max, false) => "max.u16x2",
        (IntegerMinMaxFormat::S16x2, IntegerMinMaxOperation::Min, true) => "min.relu.s16x2",
        (IntegerMinMaxFormat::S16x2, IntegerMinMaxOperation::Max, true) => "max.relu.s16x2",
        // `integer_minmax_recipe` rejects those combinations before a record
        // can exist.
        (IntegerMinMaxFormat::S32, _, false) | (IntegerMinMaxFormat::U16x2, _, true) => {
            panic!("{} is outside the closed integer-min/max recipe", record.id)
        }
    }
}
