/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogIntrinsic, DotProductAdapter, DotProductOperation,
    DotProductSignedness, IntrinsicBackend, PackedAluAdapter, PackedAluFormat, PackedAluOperation,
    PackedConversionAdapter, PackedConversionDestinationFormat, PackedConversionRounding,
    PackedConversionSaturation, PackedConversionSourceFormat,
};

pub(in crate::render) fn packed_alu_format_shape(
    format: PackedAluFormat,
) -> (
    &'static str,
    bool,
    &'static str,
    &'static str,
    PackedAluAdapter,
) {
    match format {
        PackedAluFormat::Bf16x2 => (
            "bf16x2",
            false,
            "u32",
            "i32",
            PackedAluAdapter::DirectPackedU32,
        ),
        PackedAluFormat::F16x2 => (
            "f16x2",
            true,
            "u32",
            "i32",
            PackedAluAdapter::DirectPackedU32,
        ),
        PackedAluFormat::F32x2 => (
            "f32x2",
            true,
            "u64",
            "i64",
            PackedAluAdapter::DirectPackedU64,
        ),
    }
}

pub(in crate::render) fn packed_alu_width(record: &CatalogIntrinsic) -> u32 {
    match record
        .packed_alu
        .as_ref()
        .expect("packed-ALU record")
        .adapter
    {
        PackedAluAdapter::DirectPackedU32 => 32,
        PackedAluAdapter::DirectPackedU64 => 64,
    }
}

pub(in crate::render) fn packed_alu_register_constraint(record: &CatalogIntrinsic) -> &'static str {
    match packed_alu_width(record) {
        32 => "r",
        64 => "l",
        _ => unreachable!("closed packed-ALU carrier width"),
    }
}

pub(in crate::render) fn dot_product_ptx(record: &CatalogIntrinsic) -> &'static str {
    let dot = record.dot_product.as_ref().expect("dot-product record");
    match (dot.operation, dot.signedness, dot.adapter) {
        (
            DotProductOperation::Dp4a,
            DotProductSignedness::Signed,
            DotProductAdapter::DirectThreeOperands,
        ) => "dp4a.s32.s32 $0, $1, $2, $3;",
        (
            DotProductOperation::Dp4a,
            DotProductSignedness::Unsigned,
            DotProductAdapter::DirectThreeOperands,
        ) => "dp4a.u32.u32 $0, $1, $2, $3;",
        (
            DotProductOperation::Dp2a,
            DotProductSignedness::Signed,
            DotProductAdapter::InsertLowHalfFalse,
        ) => "dp2a.lo.s32.s32 $0, $1, $2, $3;",
        (
            DotProductOperation::Dp2a,
            DotProductSignedness::Unsigned,
            DotProductAdapter::InsertLowHalfFalse,
        ) => "dp2a.lo.u32.u32 $0, $1, $2, $3;",
        combination => panic!("unsupported generated dot-product recipe {combination:?}"),
    }
}

pub(in crate::render) fn packed_alu_ptx_mnemonic(record: &CatalogIntrinsic) -> &'static str {
    let packed = record.packed_alu.as_ref().expect("packed-ALU record");
    match (packed.format, packed.operation, packed.adapter) {
        (PackedAluFormat::Bf16x2, PackedAluOperation::Add, PackedAluAdapter::DirectPackedU32) => {
            "add.rn.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Sub, PackedAluAdapter::DirectPackedU32) => {
            "sub.rn.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Mul, PackedAluAdapter::DirectPackedU32) => {
            "mul.rn.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Fma, PackedAluAdapter::DirectPackedU32) => {
            "fma.rn.bf16x2"
        }
        (
            PackedAluFormat::Bf16x2,
            PackedAluOperation::FmaRelu,
            PackedAluAdapter::DirectPackedU32,
        ) => "fma.rn.relu.bf16x2",
        (PackedAluFormat::Bf16x2, PackedAluOperation::Min, PackedAluAdapter::DirectPackedU32) => {
            "min.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Max, PackedAluAdapter::DirectPackedU32) => {
            "max.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Neg, PackedAluAdapter::DirectPackedU32) => {
            "neg.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Abs, PackedAluAdapter::DirectPackedU32) => {
            "abs.bf16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Add, PackedAluAdapter::DirectPackedU32) => {
            "add.rn.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Sub, PackedAluAdapter::DirectPackedU32) => {
            "sub.rn.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Mul, PackedAluAdapter::DirectPackedU32) => {
            "mul.rn.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Fma, PackedAluAdapter::DirectPackedU32) => {
            "fma.rn.f16x2"
        }
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::FmaRelu,
            PackedAluAdapter::DirectPackedU32,
        ) => "fma.rn.relu.f16x2",
        (PackedAluFormat::F16x2, PackedAluOperation::FmaFtz, PackedAluAdapter::DirectPackedU32) => {
            "fma.rn.ftz.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::FmaSat, PackedAluAdapter::DirectPackedU32) => {
            "fma.rn.sat.f16x2"
        }
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::FmaFtzSat,
            PackedAluAdapter::DirectPackedU32,
        ) => "fma.rn.ftz.sat.f16x2",
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::FmaFtzRelu,
            PackedAluAdapter::DirectPackedU32,
        ) => "fma.rn.ftz.relu.f16x2",
        (PackedAluFormat::F16x2, PackedAluOperation::Min, PackedAluAdapter::DirectPackedU32) => {
            "min.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Max, PackedAluAdapter::DirectPackedU32) => {
            "max.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Neg, PackedAluAdapter::DirectPackedU32) => {
            "neg.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Abs, PackedAluAdapter::DirectPackedU32) => {
            "abs.f16x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::Add, PackedAluAdapter::DirectPackedU64) => {
            "add.rn.f32x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::AddFtz, PackedAluAdapter::DirectPackedU64) => {
            "add.rn.ftz.f32x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::Sub, PackedAluAdapter::DirectPackedU64) => {
            "sub.rn.f32x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::SubFtz, PackedAluAdapter::DirectPackedU64) => {
            "sub.rn.ftz.f32x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::Mul, PackedAluAdapter::DirectPackedU64) => {
            "mul.rn.f32x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::MulFtz, PackedAluAdapter::DirectPackedU64) => {
            "mul.rn.ftz.f32x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::Fma, PackedAluAdapter::DirectPackedU64) => {
            "fma.rn.f32x2"
        }
        (PackedAluFormat::F32x2, PackedAluOperation::FmaFtz, PackedAluAdapter::DirectPackedU64) => {
            "fma.rn.ftz.f32x2"
        }
        // bf16x2 has no NVPTX selection pattern for the ftz and sat fma forms;
        // `packed_alu_recipe` rejects those pairs before a record can exist.
        combination => panic!("unsupported generated packed-ALU recipe {combination:?}"),
    }
}

pub(in crate::render) fn packed_conversion_destination(record: &CatalogIntrinsic) -> &'static str {
    match record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record")
        .destination_format
    {
        PackedConversionDestinationFormat::Bf16x2 => "bf16x2",
        PackedConversionDestinationFormat::E4m3x2 => "e4m3x2",
        PackedConversionDestinationFormat::E5m2x2 => "e5m2x2",
        PackedConversionDestinationFormat::F16x2 => "f16x2",
    }
}

pub(in crate::render) fn packed_conversion_element(record: &CatalogIntrinsic) -> &'static str {
    match record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record")
        .destination_format
    {
        PackedConversionDestinationFormat::Bf16x2 => "bf16",
        PackedConversionDestinationFormat::E4m3x2 => "e4m3",
        PackedConversionDestinationFormat::E5m2x2 => "e5m2",
        PackedConversionDestinationFormat::F16x2 => "f16",
    }
}

/// Whether the conversion is one of the closed, reviewed packed-conversion
/// recipes: scalar `f32` pairs narrowed to a packed format, `f16x2` narrowed to
/// packed FP8, or packed FP8 widened back to `f16x2`.
pub(in crate::render) fn packed_conversion_is_closed_recipe(
    conversion: &crate::model::PackedConversion,
) -> bool {
    use PackedConversionDestinationFormat as Dst;
    use PackedConversionRounding as Round;
    use PackedConversionSaturation as Sat;
    use PackedConversionSourceFormat as Src;

    let adapter_matches = conversion.adapter
        == match conversion.source_format {
            Src::F32x2 => PackedConversionAdapter::ReverseHighLowOperands,
            Src::E4m3x2 | Src::E5m2x2 | Src::F16x2 => PackedConversionAdapter::Identity,
        };

    adapter_matches
        && matches!(
            (
                conversion.source_format,
                conversion.destination_format,
                conversion.rounding,
                conversion.saturation,
            ),
            (Src::F32x2, Dst::Bf16x2, Round::NearestEven, Sat::None)
                | (Src::F32x2, Dst::Bf16x2, Round::NearestEven, Sat::Relu)
                | (Src::F32x2, Dst::Bf16x2, Round::TowardZero, Sat::None)
                | (Src::F32x2, Dst::F16x2, Round::NearestEven, Sat::None)
                | (Src::F32x2, Dst::F16x2, Round::NearestEven, Sat::Relu)
                | (Src::F32x2, Dst::F16x2, Round::TowardZero, Sat::None)
                | (Src::F32x2, Dst::E4m3x2, Round::NearestEven, Sat::Satfinite)
                | (
                    Src::F32x2,
                    Dst::E4m3x2,
                    Round::NearestEven,
                    Sat::SatfiniteRelu
                )
                | (Src::F32x2, Dst::E5m2x2, Round::NearestEven, Sat::Satfinite)
                | (
                    Src::F32x2,
                    Dst::E5m2x2,
                    Round::NearestEven,
                    Sat::SatfiniteRelu
                )
                | (Src::F16x2, Dst::E4m3x2, Round::NearestEven, Sat::Satfinite)
                | (
                    Src::F16x2,
                    Dst::E4m3x2,
                    Round::NearestEven,
                    Sat::SatfiniteRelu
                )
                | (Src::F16x2, Dst::E5m2x2, Round::NearestEven, Sat::Satfinite)
                | (
                    Src::F16x2,
                    Dst::E5m2x2,
                    Round::NearestEven,
                    Sat::SatfiniteRelu
                )
                | (Src::E4m3x2, Dst::F16x2, Round::NearestEven, Sat::None)
                | (Src::E4m3x2, Dst::F16x2, Round::NearestEven, Sat::Relu)
                | (Src::E5m2x2, Dst::F16x2, Round::NearestEven, Sat::None)
                | (Src::E5m2x2, Dst::F16x2, Round::NearestEven, Sat::Relu)
        )
}

pub(in crate::render) fn packed_conversion_source(
    record: &CatalogIntrinsic,
) -> PackedConversionSourceFormat {
    record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record")
        .source_format
}

/// Register width of a single packed source operand.
pub(in crate::render) fn packed_conversion_source_width(record: &CatalogIntrinsic) -> u32 {
    match packed_conversion_source(record) {
        PackedConversionSourceFormat::F16x2 => 32,
        PackedConversionSourceFormat::E4m3x2 | PackedConversionSourceFormat::E5m2x2 => 16,
        PackedConversionSourceFormat::F32x2 => {
            unreachable!("f32x2 conversions do not have a single packed source")
        }
    }
}

/// Rust argument types, one per source operand.
pub(in crate::render) fn packed_conversion_rust_arguments(
    record: &CatalogIntrinsic,
) -> Vec<&'static str> {
    match packed_conversion_source(record) {
        PackedConversionSourceFormat::F32x2 => vec!["f32", "f32"],
        PackedConversionSourceFormat::F16x2 => vec!["u32"],
        PackedConversionSourceFormat::E4m3x2 | PackedConversionSourceFormat::E5m2x2 => vec!["u16"],
    }
}

/// Dialect operand types, one per source operand.
pub(in crate::render) fn packed_conversion_dialect_operands(
    record: &CatalogIntrinsic,
) -> Vec<&'static str> {
    match packed_conversion_source(record) {
        PackedConversionSourceFormat::F32x2 => vec!["f32", "f32"],
        PackedConversionSourceFormat::F16x2 => vec!["i32"],
        PackedConversionSourceFormat::E4m3x2 | PackedConversionSourceFormat::E5m2x2 => vec!["i16"],
    }
}

pub(in crate::render) fn packed_conversion_result_width(record: &CatalogIntrinsic) -> u32 {
    match record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record")
        .destination_format
    {
        PackedConversionDestinationFormat::Bf16x2 | PackedConversionDestinationFormat::F16x2 => 32,
        PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2 => 16,
    }
}

pub(in crate::render) fn packed_conversion_rust_type(record: &CatalogIntrinsic) -> &'static str {
    match packed_conversion_result_width(record) {
        16 => "u16",
        32 => "u32",
        _ => unreachable!("closed packed-conversion result width"),
    }
}

pub(in crate::render) fn packed_conversion_dialect_type(record: &CatalogIntrinsic) -> &'static str {
    match packed_conversion_result_width(record) {
        16 => "i16",
        32 => "i32",
        _ => unreachable!("closed packed-conversion result width"),
    }
}

/// Inline-asm constraint string: one result register, then one per source
/// operand. `h` is a 16-bit register, `r` a 32-bit one, and `f` an f32.
pub(in crate::render) fn packed_conversion_constraint(record: &CatalogIntrinsic) -> &'static str {
    match (
        packed_conversion_result_width(record),
        packed_conversion_source(record),
    ) {
        (16, PackedConversionSourceFormat::F32x2) => "=h,f,f",
        (32, PackedConversionSourceFormat::F32x2) => "=r,f,f",
        (16, PackedConversionSourceFormat::F16x2) => "=h,r",
        (32, PackedConversionSourceFormat::E4m3x2 | PackedConversionSourceFormat::E5m2x2) => "=r,h",
        _ => unreachable!("closed packed-conversion result width and source format"),
    }
}

/// Whether the conversion lowers through a typed NVVM intrinsic call rather
/// than inline PTX. Only the scalar-`f32` pair narrowed to FP8 does; see the
/// matching predicate in `resolve/families/packed_conversion.rs` for why.
fn packed_conversion_uses_typed_nvvm(record: &CatalogIntrinsic) -> bool {
    let conversion = record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record");
    conversion.source_format == PackedConversionSourceFormat::F32x2
        && matches!(
            conversion.destination_format,
            PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2
        )
}

pub(in crate::render) fn packed_conversion_lowering_name(
    record: &CatalogIntrinsic,
) -> &'static str {
    if packed_conversion_uses_typed_nvvm(record) {
        "generated_packed_conversion_backend"
    } else {
        "generated_packed_conversion_inline_ptx"
    }
}

pub(in crate::render) fn packed_conversion_typed_llvm_name(
    record: &CatalogIntrinsic,
) -> Option<String> {
    let route = record
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .expect("packed conversion has an LLVM-NVPTX route");
    match route.mechanism {
        BackendLoweringMechanism::TypedNvvm => Some(record.llvm_identifier()),
        BackendLoweringMechanism::InlinePtx => None,
    }
}

pub(in crate::render) fn packed_conversion_ptx_mnemonic(record: &CatalogIntrinsic) -> String {
    let conversion = record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record");
    debug_assert_eq!(
        conversion.adapter,
        match conversion.source_format {
            PackedConversionSourceFormat::F32x2 => PackedConversionAdapter::ReverseHighLowOperands,
            PackedConversionSourceFormat::E4m3x2
            | PackedConversionSourceFormat::E5m2x2
            | PackedConversionSourceFormat::F16x2 => PackedConversionAdapter::Identity,
        }
    );
    let rounding = match conversion.rounding {
        PackedConversionRounding::NearestEven => "rn",
        PackedConversionRounding::TowardZero => "rz",
    };
    let saturation = match conversion.saturation {
        PackedConversionSaturation::None => "",
        PackedConversionSaturation::Relu => ".relu",
        PackedConversionSaturation::Satfinite => ".satfinite",
        PackedConversionSaturation::SatfiniteRelu => ".satfinite.relu",
    };
    format!(
        "cvt.{rounding}{saturation}.{}.{}",
        packed_conversion_destination(record),
        conversion.source_format.ptx_token()
    )
}
