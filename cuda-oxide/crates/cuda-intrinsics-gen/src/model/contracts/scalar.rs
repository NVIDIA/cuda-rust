/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::super::core::RuntimeValidation;
use serde::{Deserialize, Serialize};

/// Closed semantic contract for byte permutation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prmt {
    pub mode: PrmtMode,
    pub adapter: PrmtAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrmtMode {
    Generic,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrmtAdapter {
    DirectThreeOperands,
    InsertZeroSecondSource,
}

/// Closed identity and source adapter for generated packed integer dot products.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DotProduct {
    pub operation: DotProductOperation,
    pub signedness: DotProductSignedness,
    pub adapter: DotProductAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DotProductOperation {
    Dp2a,
    Dp4a,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DotProductSignedness {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DotProductAdapter {
    DirectThreeOperands,
    InsertLowHalfFalse,
}

/// Closed identity and carrier contract for packed floating-point ALU ops.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedAlu {
    pub format: PackedAluFormat,
    /// Hardware floor of the native PTX instruction, independent of the
    /// target floor admitted by cuda-oxide.
    pub native_minimum_sm: u16,
    pub operation: PackedAluOperation,
    pub adapter: PackedAluAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAluFormat {
    Bf16x2,
    F16x2,
    F32x2,
}

/// Closed identity and carrier contract for extended integer min/max ops.
///
/// Covers the PTX ISA 8.0 integer min/max extensions: the `.relu` saturation
/// qualifier on `s32` and `s16x2`, plus the packed `s16x2`/`u16x2` forms.
/// These are the DPX-adjacent shapes ptxas fuses into `VIMNMX`-family SASS.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegerMinMax {
    pub format: IntegerMinMaxFormat,
    /// Hardware floor of the native PTX instruction, independent of the
    /// target floor admitted by cuda-oxide.
    pub native_minimum_sm: u16,
    pub operation: IntegerMinMaxOperation,
    pub relu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegerMinMaxFormat {
    S32,
    S16x2,
    U16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegerMinMaxOperation {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAluOperation {
    Add,
    AddFtz,
    Sub,
    SubFtz,
    Mul,
    MulFtz,
    Fma,
    FmaFtz,
    FmaSat,
    FmaFtzSat,
    FmaRelu,
    FmaFtzRelu,
    Min,
    Max,
    Neg,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAluAdapter {
    DirectPackedU32,
    DirectPackedU64,
}

/// Closed contract for scalar floating-point arithmetic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarArithmetic {
    pub format: ScalarArithmeticFormat,
    pub operation: ScalarArithmeticOperation,
    pub rounding: ScalarArithmeticRounding,
    pub subnormal: ScalarArithmeticSubnormal,
    pub saturation: ScalarArithmeticSaturation,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticFormat {
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticOperation {
    Mul,
    Div,
    Fma,
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticRounding {
    Rn,
    Rz,
    Rm,
    Rp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticSubnormal {
    Preserve,
    Ftz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticSaturation {
    None,
    Sat,
}

/// Closed contract for unary scalar floating-point math operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarMath {
    pub format: ScalarMathFormat,
    pub operation: ScalarMathOperation,
    pub precision: ScalarMathPrecision,
    pub subnormal: ScalarMathSubnormal,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarMathFormat {
    F16,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarMathOperation {
    Sin,
    Cos,
    /// Routes through inline PTX: LLVM 22's tblgen models ex2 as the
    /// overloaded `int_nvvm_ex2_approx{,_ftz}` records (anyfloat, no DAG
    /// selection pattern), so the evidence contract cannot admit a typed
    /// call. The legacy `llvm.nvvm.ex2.approx.f`/`.ftz.f` names still select
    /// directly on both llc 21 and 22, so this is promotable to a typed
    /// route once the import resolves the overloaded family.
    Ex2,
    Lg2,
    Rcp,
    Rsqrt,
    Sqrt,
    /// Routes through inline PTX: LLVM 22.1.2's tblgen export carries no
    /// record for tanh at all (llc 22 selects `llvm.nvvm.tanh.approx.f32`
    /// via NVVMIntrinsic-class matching, and llc 21 miscompiles it into an
    /// extern funcall), so this is the family's only PTX-native source.
    /// Hardware floor is sm_75 (PTX ISA 7.0); the family contract gates it
    /// at the attested sm_80 evidence floor like every other variant.
    Tanh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarMathPrecision {
    Approx,
    Rn,
    Rz,
    Rm,
    Rp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarMathSubnormal {
    Preserve,
    Ftz,
}

/// Closed identity and carrier contract for extended floating-point min/max.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendedMinMax {
    pub format: ExtendedMinMaxFormat,
    pub operation: ExtendedMinMaxOperation,
    pub subnormal: ExtendedMinMaxSubnormal,
    pub nan: ExtendedMinMaxNan,
    pub xorsign_abs: bool,
    pub adapter: ExtendedMinMaxAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxFormat {
    F32,
    F16,
    Bf16,
    F16x2,
    Bf16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxOperation {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxSubnormal {
    Preserve,
    Ftz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxNan {
    Number,
    Nan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxAdapter {
    DirectF32,
    /// A single 16-bit float carried as its `u16` bit pattern, matching how
    /// `packed_conversion` already carries `e4m3x2` and `e5m2x2` operands.
    DirectHalfU16,
    DirectPackedU32,
}

/// Closed contract for converting between scalar and packed values.
///
/// The source may be a scalar pair (`f32x2`, two operands) or an already-packed
/// 16-bit or 32-bit register (`f16x2`, `e4m3x2`, `e5m2x2`, one operand), so the
/// operand arity follows [`PackedConversionSourceFormat`] rather than being
/// fixed at two.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedConversion {
    pub source_format: PackedConversionSourceFormat,
    pub destination_format: PackedConversionDestinationFormat,
    pub rounding: PackedConversionRounding,
    pub saturation: PackedConversionSaturation,
    pub adapter: PackedConversionAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionSourceFormat {
    E4m3x2,
    E5m2x2,
    F16x2,
    F32x2,
}

impl PackedConversionSourceFormat {
    /// Number of source operands the conversion takes.
    ///
    /// `f32x2` names two scalar `f32` operands; every packed source arrives in a
    /// single register.
    pub fn operand_count(self) -> usize {
        match self {
            Self::F32x2 => 2,
            Self::E4m3x2 | Self::E5m2x2 | Self::F16x2 => 1,
        }
    }

    /// PTX source-type token, used as the trailing `cvt` modifier.
    pub fn ptx_token(self) -> &'static str {
        match self {
            Self::E4m3x2 => "e4m3x2",
            Self::E5m2x2 => "e5m2x2",
            Self::F16x2 => "f16x2",
            Self::F32x2 => "f32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionDestinationFormat {
    Bf16x2,
    E4m3x2,
    E5m2x2,
    F16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionRounding {
    NearestEven,
    TowardZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionSaturation {
    None,
    Relu,
    Satfinite,
    SatfiniteRelu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionAdapter {
    /// Forward the single packed source operand unchanged.
    ///
    /// PTX orders `cvt` operands as `d, a`, so a one-operand conversion needs no
    /// reordering to keep the Rust argument order.
    Identity,
    /// Swap the two scalar operands.
    ///
    /// PTX writes the second source operand into the low half, so the operands
    /// are reversed to keep the first Rust argument in the low half.
    ReverseHighLowOperands,
}

/// Closed contract for one scalar floating-point conversion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarConversion {
    pub source_format: ScalarConversionSourceFormat,
    pub destination_format: ScalarConversionDestinationFormat,
    pub rounding: ScalarConversionRounding,
    pub saturation: ScalarConversionSaturation,
    pub result_representation: ScalarConversionResultRepresentation,
    pub adapter: ScalarConversionAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionSourceFormat {
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionDestinationFormat {
    Tf32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionRounding {
    NearestAway,
    NearestEven,
    TowardZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionSaturation {
    None,
    Relu,
    Satfinite,
    ReluSatfinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionResultRepresentation {
    RawU32Bits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionAdapter {
    DirectF32ToRawU32Bits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_alu_contract_rejects_unknown_format_operation_and_adapter() {
        let valid = r#"
format = "bf16x2"
native_minimum_sm = 80
operation = "fma"
adapter = "direct_packed_u32"
"#;
        toml::from_str::<PackedAlu>(valid).unwrap();
        for invalid in [
            valid.replace("format = \"bf16x2\"", "format = \"bf16\""),
            valid.replace("native_minimum_sm = 80\n", ""),
            valid.replace("native_minimum_sm = 80", "native_minimum_sm = \"80\""),
            valid.replace("operation = \"fma\"", "operation = \"mad\""),
            valid.replace(
                "adapter = \"direct_packed_u32\"",
                "adapter = \"bitcast_any\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<PackedAlu>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn packed_conversion_contract_rejects_open_ended_policy() {
        let valid = r#"
source_format = "f32x2"
destination_format = "bf16x2"
rounding = "nearest_even"
saturation = "none"
adapter = "reverse_high_low_operands"
"#;
        toml::from_str::<PackedConversion>(valid).unwrap();
        for invalid in [
            valid.replace("source_format = \"f32x2\"", "source_format = \"f64x2\""),
            valid.replace(
                "destination_format = \"bf16x2\"",
                "destination_format = \"f8x2\"",
            ),
            valid.replace("rounding = \"nearest_even\"", "rounding = \"zero\""),
            valid.replace("saturation = \"none\"", "saturation = \"finite\""),
            valid.replace(
                "adapter = \"reverse_high_low_operands\"",
                "adapter = \"direct\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<PackedConversion>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }
}
