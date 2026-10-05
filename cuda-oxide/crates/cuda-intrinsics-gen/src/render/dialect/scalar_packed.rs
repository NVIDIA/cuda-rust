/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{CatalogFile, PackedConversionSourceFormat};
use crate::render::common::rust_header;
use crate::render::families::{
    extended_minmax, extended_minmax_format_attr, extended_minmax_nan_attr,
    extended_minmax_operation_attr, extended_minmax_subnormal_attr,
    extended_minmax_xorsign_abs_attr, integer_minmax_ptx_mnemonic, integer_minmaxes,
    packed_alu_ptx_mnemonic, packed_alu_width, packed_alus, packed_conversion_element,
    packed_conversion_result_width, packed_conversion_source, packed_conversion_source_width,
    packed_conversions, scalar_arithmetic_format_attr, scalar_arithmetic_operation_attr,
    scalar_arithmetic_rounding_attr, scalar_arithmetic_saturation_attr,
    scalar_arithmetic_subnormal_attr, scalar_arithmetics, scalar_conversion_rounding_attr,
    scalar_conversion_saturation_attr, scalar_conversions, scalar_math_format_attr,
    scalar_math_operation_attr, scalar_math_precision_attr, scalar_math_subnormal_attr,
    scalar_maths,
};
use std::fmt::Write as _;

pub(in crate::render) fn render_dialect_integer_minmax(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for generated extended integer min/max.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in integer_minmaxes(catalog) {
        let signedness = if record.rust.result == "i32" {
            "Signed"
        } else {
            "Unsigned"
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "///\n/// Lowers to `{}`.",
            integer_minmax_ptx_mnemonic(record)
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, arg0: Value, arg1: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, 32, Signedness::{signedness});\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![arg0, arg1],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !(0..2).all(|index| is_i32(ctx, op.get_operand(index).get_type(ctx)))\n            || !is_i32(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly 2 operands and one result",
                record.dialect.op_name
            ),
            format!(
                "{} operands and result must be 32-bit integers",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in integer_minmaxes(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_packed_atomic(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for the closed generated packed-atomic family.

use dialect_mir::types::{address_space, MirPtrType};
use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.packed_atomic_format", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicFormatAttr { F16x2, Bf16x2 }

#[pliron_attr(name = "nvvm.packed_atomic_state_space", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicStateSpaceAttr { Global }

#[pliron_attr(name = "nvvm.packed_atomic_ordering", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicOrderingAttr { Relaxed }

#[pliron_attr(name = "nvvm.packed_atomic_scope", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicScopeAttr { Gpu }

#[pliron_attr(name = "nvvm.packed_atomic_rounding", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicRoundingAttr { Rn }

#[pliron_attr(name = "nvvm.packed_atomic_subnormal", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicSubnormalAttr { NoFtz }

#[pliron_attr(name = "nvvm.packed_atomic_atomicity", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicAtomicityAttr { PerElement }

/// Packed global atomic add with exact format and semantic attributes.
#[pliron_op(
    name = "nvvm.packed_atomic_add",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
    attributes = (
        nvvm_packed_atomic_format: PackedAtomicFormatAttr,
        nvvm_packed_atomic_state_space: PackedAtomicStateSpaceAttr,
        nvvm_packed_atomic_ordering: PackedAtomicOrderingAttr,
        nvvm_packed_atomic_scope: PackedAtomicScopeAttr,
        nvvm_packed_atomic_rounding: PackedAtomicRoundingAttr,
        nvvm_packed_atomic_subnormal: PackedAtomicSubnormalAttr,
        nvvm_packed_atomic_atomicity: PackedAtomicAtomicityAttr
    )
)]
pub struct PackedAtomicAddOp;

impl PackedAtomicAddOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(
        ctx: &mut Context,
        address: Value,
        addend: Value,
        format: PackedAtomicFormatAttr,
    ) -> Ptr<Operation> {
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![u32_ty.into()],
            vec![address, addend],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_packed_atomic_format(ctx, format);
        this.set_attr_nvvm_packed_atomic_state_space(ctx, PackedAtomicStateSpaceAttr::Global);
        this.set_attr_nvvm_packed_atomic_ordering(ctx, PackedAtomicOrderingAttr::Relaxed);
        this.set_attr_nvvm_packed_atomic_scope(ctx, PackedAtomicScopeAttr::Gpu);
        this.set_attr_nvvm_packed_atomic_rounding(ctx, PackedAtomicRoundingAttr::Rn);
        this.set_attr_nvvm_packed_atomic_subnormal(ctx, PackedAtomicSubnormalAttr::NoFtz);
        this.set_attr_nvvm_packed_atomic_atomicity(ctx, PackedAtomicAtomicityAttr::PerElement);
        this.get_operation()
    }
}

fn is_u32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx).downcast_ref::<IntegerType>().is_some_and(|integer| {
        integer.width() == 32 && integer.signedness() == Signedness::Unsigned
    })
}

fn verify_packed_atomic_signature(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
    op_name: &str,
) -> Result<(), Error> {
    let op = op_ptr.deref(ctx);
    if op.get_num_operands() != 2 || op.get_num_results() != 1 {
        return verify_err!(op.loc(), "{} requires exactly two operands and one result", op_name);
    }
    let pointer_ty = op.get_operand(0).get_type(ctx);
    let pointer_object = pointer_ty.deref(ctx);
    let Some(pointer) = pointer_object.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "{} address must be a MIR pointer", op_name);
    };
    if !pointer.is_mutable()
        || !matches!(pointer.address_space(), address_space::GENERIC | address_space::GLOBAL)
        || !is_u32(ctx, pointer.pointee)
    {
        return verify_err!(op.loc(), "{} address must be a mutable generic/global pointer to u32", op_name);
    }
    if !is_u32(ctx, op.get_operand(1).get_type(ctx)) || !is_u32(ctx, op.get_result(0).get_type(ctx)) {
        return verify_err!(op.loc(), "{} addend and result must be u32", op_name);
    }
    Ok(())
}

impl Verify for PackedAtomicAddOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_packed_atomic_signature(ctx, self.get_operation(), "nvvm.packed_atomic_add")?;
        let op = self.get_operation().deref(ctx);
        if self.get_attr_nvvm_packed_atomic_format(ctx).is_none()
            || self.get_attr_nvvm_packed_atomic_state_space(ctx).as_deref() != Some(&PackedAtomicStateSpaceAttr::Global)
            || self.get_attr_nvvm_packed_atomic_ordering(ctx).as_deref() != Some(&PackedAtomicOrderingAttr::Relaxed)
            || self.get_attr_nvvm_packed_atomic_scope(ctx).as_deref() != Some(&PackedAtomicScopeAttr::Gpu)
            || self.get_attr_nvvm_packed_atomic_rounding(ctx).as_deref() != Some(&PackedAtomicRoundingAttr::Rn)
            || self.get_attr_nvvm_packed_atomic_subnormal(ctx).as_deref() != Some(&PackedAtomicSubnormalAttr::NoFtz)
            || self.get_attr_nvvm_packed_atomic_atomicity(ctx).as_deref() != Some(&PackedAtomicAtomicityAttr::PerElement)
        {
            return verify_err!(op.loc(), "nvvm.packed_atomic_add has a missing or unsupported semantic attribute");
        }
        Ok(())
    }
}

/// Compatibility operation for the existing `nvvm.atom_add_f16x2` carrier.
#[pliron_op(
    name = "nvvm.atom_add_f16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct NvvmAtomAddF16x2Op;

impl NvvmAtomAddF16x2Op {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }
}

impl Verify for NvvmAtomAddF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_packed_atomic_signature(ctx, self.get_operation(), "nvvm.atom_add_f16x2")
    }
}

/// Compatibility operation for the existing `nvvm.atom_add_bf16x2` carrier.
#[pliron_op(
    name = "nvvm.atom_add_bf16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct NvvmAtomAddBf16x2Op;

impl NvvmAtomAddBf16x2Op {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }
}

impl Verify for NvvmAtomAddBf16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_packed_atomic_signature(ctx, self.get_operation(), "nvvm.atom_add_bf16x2")
    }
}

pub(super) fn register(ctx: &mut Context) {
    PackedAtomicFormatAttr::register(ctx);
    PackedAtomicStateSpaceAttr::register(ctx);
    PackedAtomicOrderingAttr::register(ctx);
    PackedAtomicScopeAttr::register(ctx);
    PackedAtomicRoundingAttr::register(ctx);
    PackedAtomicSubnormalAttr::register(ctx);
    PackedAtomicAtomicityAttr::register(ctx);
    PackedAtomicAddOp::register(ctx);
    NvvmAtomAddF16x2Op::register(ctx);
    NvvmAtomAddBf16x2Op::register(ctx);
}
"##,
    );
    output
}

pub(in crate::render) fn render_dialect_packed_alu(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for generated packed floating-point arithmetic.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_integer_width(\n    ctx: &Context,\n    ty: pliron::r#type::TypeHandle,\n    width: u32,\n) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\n",
    );
    for record in packed_alus(catalog) {
        let arity = record.rust.arguments.len();
        let width = packed_alu_width(record);
        let parameters = (0..arity)
            .map(|index| format!("arg{index}: Value"))
            .collect::<Vec<_>>()
            .join(", ");
        let operands = (0..arity)
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "///\n/// Lowers to `{}`.",
            packed_alu_ptx_mnemonic(record)
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{arity}>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, {parameters}) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, {width}, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![{operands}],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != {arity} || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !(0..{arity}).all(|index| is_integer_width(ctx, op.get_operand(index).get_type(ctx), {width}))\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), {width})\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly {arity} operands and one result",
                record.dialect.op_name
            ),
            format!(
                "{} operands and result must be {width}-bit integers",
                record.dialect.op_name,
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in packed_alus(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_packed_conversion(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for generated packed conversion.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{FP32Type, IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_f32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()\n}\n\nfn is_integer_width(\n    ctx: &Context,\n    ty: pliron::r#type::TypeHandle,\n    width: u32,\n) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\n",
    );
    for record in packed_conversions(catalog) {
        let result_width = packed_conversion_result_width(record);
        writeln!(output, "/// {}", record.summary).unwrap();
        match packed_conversion_source(record) {
            PackedConversionSourceFormat::F32x2 => {
                writeln!(
                    output,
                    "///\n/// The first input becomes the low {} lane; the second becomes the high lane.",
                    packed_conversion_element(record)
                )
                .unwrap();
                writeln!(
                    output,
                    "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
                    record.dialect.op_name
                )
                .unwrap();
                writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
                writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
                writeln!(
                    output,
                    "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, low: Value, high: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, {result_width}, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![low, high],\n            vec![],\n            0,\n        )\n    }}\n}}",
                )
                .unwrap();
                writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
                writeln!(
                    output,
                    "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_f32(ctx, op.get_operand(0).get_type(ctx))\n            || !is_f32(ctx, op.get_operand(1).get_type(ctx))\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), {})\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
                    format!("{} requires two operands and one result", record.dialect.op_name),
                    result_width,
                    format!(
                        "{} requires f32 operands and one {result_width}-bit integer result",
                        record.dialect.op_name,
                    ),
                )
                .unwrap();
            }
            PackedConversionSourceFormat::E4m3x2
            | PackedConversionSourceFormat::E5m2x2
            | PackedConversionSourceFormat::F16x2 => {
                let source_width = packed_conversion_source_width(record);
                writeln!(
                    output,
                    "///\n/// The single input carries both packed lanes, and lane order is preserved."
                )
                .unwrap();
                writeln!(
                    output,
                    "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],\n)]",
                    record.dialect.op_name
                )
                .unwrap();
                writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
                writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
                writeln!(
                    output,
                    "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, packed: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, {result_width}, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![packed],\n            vec![],\n            0,\n        )\n    }}\n}}",
                )
                .unwrap();
                writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
                writeln!(
                    output,
                    "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 1 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_integer_width(ctx, op.get_operand(0).get_type(ctx), {source_width})\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), {result_width})\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
                    format!("{} requires one operand and one result", record.dialect.op_name),
                    format!(
                        "{} requires one {source_width}-bit integer operand and one {result_width}-bit integer result",
                        record.dialect.op_name,
                    ),
                )
                .unwrap();
            }
        }
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in packed_conversions(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_scalar_conversion(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for generated `f32` to TF32 conversions.

use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.scalar_conversion_rounding", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarConversionRoundingAttr {
    NearestAway,
    NearestEven,
    TowardZero,
}

#[pliron_attr(name = "nvvm.scalar_conversion_saturation", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarConversionSaturationAttr {
    None,
    Relu,
    Satfinite,
    ReluSatfinite,
}

/// Converts one `f32` value and returns the raw TF32 bits.
#[pliron_op(
    name = "nvvm.scalar_conversion",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
    attributes = (
        nvvm_scalar_conversion_rounding: ScalarConversionRoundingAttr,
        nvvm_scalar_conversion_saturation: ScalarConversionSaturationAttr
    )
)]
pub struct ScalarConversionOp;

impl ScalarConversionOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(
        ctx: &mut Context,
        value: Value,
        rounding: ScalarConversionRoundingAttr,
        saturation: ScalarConversionSaturationAttr,
    ) -> Ptr<Operation> {
        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![value],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_scalar_conversion_rounding(ctx, rounding);
        this.set_attr_nvvm_scalar_conversion_saturation(ctx, saturation);
        this.get_operation()
    }
}

impl Verify for ScalarConversionOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let Some(rounding) = self.get_attr_nvvm_scalar_conversion_rounding(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_conversion requires rounding");
        };
        let Some(saturation) = self.get_attr_nvvm_scalar_conversion_saturation(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_conversion requires saturation");
        };
        let admitted = matches!(
            (&*rounding, &*saturation),
"##,
    );
    for (index, record) in scalar_conversions(catalog).enumerate() {
        if index != 0 {
            output.push_str(" |\n");
        }
        writeln!(
            output,
            "            ({}, {})",
            scalar_conversion_rounding_attr(record),
            scalar_conversion_saturation_attr(record),
        )
        .unwrap();
    }
    output.push_str(
        r##"        );
        if !admitted {
            return verify_err!(op.loc(), "nvvm.scalar_conversion variant is not admitted");
        }
        if op
            .get_operand(0)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<FP32Type>()
            .is_none()
        {
            return verify_err!(op.loc(), "nvvm.scalar_conversion operand must be f32");
        }
        if op
            .get_result(0)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_none_or(|integer| integer.width() != 32)
        {
            return verify_err!(op.loc(), "nvvm.scalar_conversion result must be a 32-bit integer");
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    ScalarConversionRoundingAttr::register(ctx);
    ScalarConversionSaturationAttr::register(ctx);
    ScalarConversionOp::register(ctx);
}
"##,
    );
    output
}

pub(in crate::render) fn render_dialect_scalar_arithmetic(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for generated scalar arithmetic.

use pliron::{
    attribute::Attribute,
    builtin::{op_interfaces::NResultsInterface, types::{FP32Type, FP64Type}},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.scalar_arithmetic_format", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarArithmeticFormatAttr { F32, F64 }

#[pliron_attr(name = "nvvm.scalar_arithmetic_operation", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarArithmeticOperationAttr { Mul, Div, Fma, Add }

#[pliron_attr(name = "nvvm.scalar_arithmetic_rounding", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarArithmeticRoundingAttr { Rn, Rz, Rm, Rp }

#[pliron_attr(name = "nvvm.scalar_arithmetic_subnormal", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarArithmeticSubnormalAttr { Preserve, Ftz }

#[pliron_attr(name = "nvvm.scalar_arithmetic_saturation", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarArithmeticSaturationAttr { None, Sat }

/// Scalar arithmetic with explicit PTX modifiers.
#[pliron_op(
    name = "nvvm.scalar_arithmetic",
    format,
    interfaces = [NResultsInterface<1>],
    attributes = (
        nvvm_scalar_arithmetic_format: ScalarArithmeticFormatAttr,
        nvvm_scalar_arithmetic_operation: ScalarArithmeticOperationAttr,
        nvvm_scalar_arithmetic_rounding: ScalarArithmeticRoundingAttr,
        nvvm_scalar_arithmetic_subnormal: ScalarArithmeticSubnormalAttr,
        nvvm_scalar_arithmetic_saturation: ScalarArithmeticSaturationAttr
    )
)]
pub struct ScalarArithmeticOp;

impl ScalarArithmeticOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(
        ctx: &mut Context,
        operands: Vec<Value>,
        format: ScalarArithmeticFormatAttr,
        operation: ScalarArithmeticOperationAttr,
        rounding: ScalarArithmeticRoundingAttr,
        subnormal: ScalarArithmeticSubnormalAttr,
        saturation: ScalarArithmeticSaturationAttr,
    ) -> Ptr<Operation> {
        let result_ty = match &format {
            ScalarArithmeticFormatAttr::F32 => FP32Type::get(ctx).into(),
            ScalarArithmeticFormatAttr::F64 => FP64Type::get(ctx).into(),
        };
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            operands,
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_scalar_arithmetic_format(ctx, format);
        this.set_attr_nvvm_scalar_arithmetic_operation(ctx, operation);
        this.set_attr_nvvm_scalar_arithmetic_rounding(ctx, rounding);
        this.set_attr_nvvm_scalar_arithmetic_subnormal(ctx, subnormal);
        this.set_attr_nvvm_scalar_arithmetic_saturation(ctx, saturation);
        this.get_operation()
    }
}

impl Verify for ScalarArithmeticOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let Some(format) = self.get_attr_nvvm_scalar_arithmetic_format(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_arithmetic requires a format");
        };
        let Some(operation) = self.get_attr_nvvm_scalar_arithmetic_operation(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_arithmetic requires an operation");
        };
        let Some(rounding) = self.get_attr_nvvm_scalar_arithmetic_rounding(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_arithmetic requires rounding");
        };
        let Some(subnormal) = self.get_attr_nvvm_scalar_arithmetic_subnormal(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_arithmetic requires a subnormal mode");
        };
        let Some(saturation) = self.get_attr_nvvm_scalar_arithmetic_saturation(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_arithmetic requires saturation");
        };
        let admitted = matches!(
            (&*format, &*operation, &*rounding, &*subnormal, &*saturation),
"##,
    );
    for (index, record) in scalar_arithmetics(catalog).enumerate() {
        if index != 0 {
            output.push_str(" |\n");
        }
        writeln!(
            output,
            "            ({}, {}, {}, {}, {})",
            scalar_arithmetic_format_attr(record),
            scalar_arithmetic_operation_attr(record),
            scalar_arithmetic_rounding_attr(record),
            scalar_arithmetic_subnormal_attr(record),
            scalar_arithmetic_saturation_attr(record),
        )
        .unwrap();
    }
    output.push_str(
        r##"        );
        if !admitted {
            return verify_err!(op.loc(), "nvvm.scalar_arithmetic variant is not admitted");
        }
        let expected_operands = match &*operation {
            ScalarArithmeticOperationAttr::Mul
            | ScalarArithmeticOperationAttr::Div
            | ScalarArithmeticOperationAttr::Add => 2,
            ScalarArithmeticOperationAttr::Fma => 3,
        };
        if op.get_num_operands() != expected_operands || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.scalar_arithmetic requires {} operands and one result",
                expected_operands,
            );
        }
        let type_matches = |ty: pliron::r#type::TypeHandle| match &*format {
            ScalarArithmeticFormatAttr::F32 => ty.deref(ctx).downcast_ref::<FP32Type>().is_some(),
            ScalarArithmeticFormatAttr::F64 => ty.deref(ctx).downcast_ref::<FP64Type>().is_some(),
        };
        if !(0..expected_operands).all(|index| type_matches(op.get_operand(index).get_type(ctx)))
            || !type_matches(op.get_result(0).get_type(ctx))
        {
            return verify_err!(op.loc(), "nvvm.scalar_arithmetic types do not match its format");
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    ScalarArithmeticFormatAttr::register(ctx);
    ScalarArithmeticOperationAttr::register(ctx);
    ScalarArithmeticRoundingAttr::register(ctx);
    ScalarArithmeticSubnormalAttr::register(ctx);
    ScalarArithmeticSaturationAttr::register(ctx);
    ScalarArithmeticOp::register(ctx);
}
"##,
    );
    output
}

pub(in crate::render) fn render_dialect_scalar_math(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for generated scalar math.

use pliron::{
    attribute::Attribute,
    builtin::{op_interfaces::{NOpdsInterface, NResultsInterface}, types::{FP32Type, FP64Type, IntegerType, Signedness}},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.scalar_math_format", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarMathFormatAttr { F16, F32, F64 }

#[pliron_attr(name = "nvvm.scalar_math_operation", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarMathOperationAttr { Sin, Cos, Ex2, Lg2, Rcp, Rsqrt, Sqrt, Tanh }

#[pliron_attr(name = "nvvm.scalar_math_precision", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarMathPrecisionAttr { Approx, Rn, Rz, Rm, Rp }

#[pliron_attr(name = "nvvm.scalar_math_subnormal", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ScalarMathSubnormalAttr { Preserve, Ftz }

/// Scalar math with exact PTX modifiers.
#[pliron_op(
    name = "nvvm.scalar_math",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
    attributes = (
        nvvm_scalar_math_format: ScalarMathFormatAttr,
        nvvm_scalar_math_operation: ScalarMathOperationAttr,
        nvvm_scalar_math_precision: ScalarMathPrecisionAttr,
        nvvm_scalar_math_subnormal: ScalarMathSubnormalAttr
    )
)]
pub struct ScalarMathOp;

impl ScalarMathOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(
        ctx: &mut Context,
        operand: Value,
        format: ScalarMathFormatAttr,
        operation: ScalarMathOperationAttr,
        precision: ScalarMathPrecisionAttr,
        subnormal: ScalarMathSubnormalAttr,
    ) -> Ptr<Operation> {
        let result_ty = match &format {
            ScalarMathFormatAttr::F16 => IntegerType::get(ctx, 16, Signedness::Unsigned).into(),
            ScalarMathFormatAttr::F32 => FP32Type::get(ctx).into(),
            ScalarMathFormatAttr::F64 => FP64Type::get(ctx).into(),
        };
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![operand],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_scalar_math_format(ctx, format);
        this.set_attr_nvvm_scalar_math_operation(ctx, operation);
        this.set_attr_nvvm_scalar_math_precision(ctx, precision);
        this.set_attr_nvvm_scalar_math_subnormal(ctx, subnormal);
        this.get_operation()
    }
}

impl Verify for ScalarMathOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let Some(format) = self.get_attr_nvvm_scalar_math_format(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_math requires a format");
        };
        let Some(operation) = self.get_attr_nvvm_scalar_math_operation(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_math requires an operation");
        };
        let Some(precision) = self.get_attr_nvvm_scalar_math_precision(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_math requires a precision");
        };
        let Some(subnormal) = self.get_attr_nvvm_scalar_math_subnormal(ctx) else {
            return verify_err!(op.loc(), "nvvm.scalar_math requires a subnormal mode");
        };
        let admitted = matches!(
            (&*format, &*operation, &*precision, &*subnormal),
"##,
    );
    for (index, record) in scalar_maths(catalog).enumerate() {
        if index != 0 {
            output.push_str(" |\n");
        }
        writeln!(
            output,
            "            ({}, {}, {}, {})",
            scalar_math_format_attr(record),
            scalar_math_operation_attr(record),
            scalar_math_precision_attr(record),
            scalar_math_subnormal_attr(record),
        )
        .unwrap();
    }
    output.push_str(
        r##"        );
        if !admitted {
            return verify_err!(op.loc(), "nvvm.scalar_math variant is not admitted");
        }
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.scalar_math requires one operand and one result",
            );
        }
        let type_matches = |ty: pliron::r#type::TypeHandle| match &*format {
            ScalarMathFormatAttr::F16 => ty.deref(ctx).downcast_ref::<IntegerType>().is_some_and(|integer| integer.width() == 16),
            ScalarMathFormatAttr::F32 => ty.deref(ctx).downcast_ref::<FP32Type>().is_some(),
            ScalarMathFormatAttr::F64 => ty.deref(ctx).downcast_ref::<FP64Type>().is_some(),
        };
        if !type_matches(op.get_operand(0).get_type(ctx))
            || !type_matches(op.get_result(0).get_type(ctx))
        {
            return verify_err!(op.loc(), "nvvm.scalar_math types do not match its format");
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    ScalarMathFormatAttr::register(ctx);
    ScalarMathOperationAttr::register(ctx);
    ScalarMathPrecisionAttr::register(ctx);
    ScalarMathSubnormalAttr::register(ctx);
    ScalarMathOp::register(ctx);
}
"##,
    );
    output
}

pub(in crate::render) fn render_dialect_extended_minmax(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for extended floating-point min/max.

use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.extended_minmax_format", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ExtendedMinMaxFormatAttr { F32, F16, Bf16, F16x2, Bf16x2 }

#[pliron_attr(name = "nvvm.extended_minmax_operation", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ExtendedMinMaxOperationAttr { Min, Max }

#[pliron_attr(name = "nvvm.extended_minmax_subnormal", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ExtendedMinMaxSubnormalAttr { Preserve, Ftz }

#[pliron_attr(name = "nvvm.extended_minmax_nan", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ExtendedMinMaxNanAttr { Number, Nan }

#[pliron_attr(name = "nvvm.extended_minmax_xorsign_abs", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ExtendedMinMaxXorSignAbsAttr { Disabled, Enabled }

/// Extended min/max with exact PTX modifiers.
#[pliron_op(
    name = "nvvm.extended_minmax",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
    attributes = (
        nvvm_extended_minmax_format: ExtendedMinMaxFormatAttr,
        nvvm_extended_minmax_operation: ExtendedMinMaxOperationAttr,
        nvvm_extended_minmax_subnormal: ExtendedMinMaxSubnormalAttr,
        nvvm_extended_minmax_nan: ExtendedMinMaxNanAttr,
        nvvm_extended_minmax_xorsign_abs: ExtendedMinMaxXorSignAbsAttr
    )
)]
pub struct ExtendedMinMaxOp;

impl ExtendedMinMaxOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        a: Value,
        b: Value,
        format: ExtendedMinMaxFormatAttr,
        operation: ExtendedMinMaxOperationAttr,
        subnormal: ExtendedMinMaxSubnormalAttr,
        nan: ExtendedMinMaxNanAttr,
        xorsign_abs: ExtendedMinMaxXorSignAbsAttr,
    ) -> Ptr<Operation> {
        let result_ty = match &format {
            ExtendedMinMaxFormatAttr::F32 => FP32Type::get(ctx).into(),
            ExtendedMinMaxFormatAttr::F16 | ExtendedMinMaxFormatAttr::Bf16 => {
                IntegerType::get(ctx, 16, Signedness::Unsigned).into()
            }
            ExtendedMinMaxFormatAttr::F16x2 | ExtendedMinMaxFormatAttr::Bf16x2 => {
                IntegerType::get(ctx, 32, Signedness::Unsigned).into()
            }
        };
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![a, b],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_extended_minmax_format(ctx, format);
        this.set_attr_nvvm_extended_minmax_operation(ctx, operation);
        this.set_attr_nvvm_extended_minmax_subnormal(ctx, subnormal);
        this.set_attr_nvvm_extended_minmax_nan(ctx, nan);
        this.set_attr_nvvm_extended_minmax_xorsign_abs(ctx, xorsign_abs);
        this.get_operation()
    }
}

impl Verify for ExtendedMinMaxOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let Some(format) = self.get_attr_nvvm_extended_minmax_format(ctx) else {
            return verify_err!(op.loc(), "nvvm.extended_minmax requires a format");
        };
        let Some(operation) = self.get_attr_nvvm_extended_minmax_operation(ctx) else {
            return verify_err!(op.loc(), "nvvm.extended_minmax requires an operation");
        };
        let Some(subnormal) = self.get_attr_nvvm_extended_minmax_subnormal(ctx) else {
            return verify_err!(op.loc(), "nvvm.extended_minmax requires a subnormal mode");
        };
        let Some(nan) = self.get_attr_nvvm_extended_minmax_nan(ctx) else {
            return verify_err!(op.loc(), "nvvm.extended_minmax requires a NaN mode");
        };
        let Some(xorsign_abs) = self.get_attr_nvvm_extended_minmax_xorsign_abs(ctx) else {
            return verify_err!(op.loc(), "nvvm.extended_minmax requires an xorsign.abs mode");
        };
        let admitted = matches!(
            (&*format, &*operation, &*subnormal, &*nan, &*xorsign_abs),
"##,
    );
    for (index, record) in extended_minmax(catalog).enumerate() {
        if index != 0 {
            output.push_str(" |\n");
        }
        writeln!(
            output,
            "            ({}, {}, {}, {}, {})",
            extended_minmax_format_attr(record),
            extended_minmax_operation_attr(record),
            extended_minmax_subnormal_attr(record),
            extended_minmax_nan_attr(record),
            extended_minmax_xorsign_abs_attr(record),
        )
        .unwrap();
    }
    output.push_str(
        r##"        );
        if !admitted {
            return verify_err!(op.loc(), "nvvm.extended_minmax variant is not admitted");
        }
        let type_matches = |ty: pliron::r#type::TypeHandle| match &*format {
            ExtendedMinMaxFormatAttr::F32 => ty.deref(ctx).downcast_ref::<FP32Type>().is_some(),
            ExtendedMinMaxFormatAttr::F16 | ExtendedMinMaxFormatAttr::Bf16 => ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 16),
            ExtendedMinMaxFormatAttr::F16x2 | ExtendedMinMaxFormatAttr::Bf16x2 => ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 32),
        };
        if !type_matches(op.get_operand(0).get_type(ctx))
            || !type_matches(op.get_operand(1).get_type(ctx))
            || !type_matches(op.get_result(0).get_type(ctx))
        {
            return verify_err!(op.loc(), "nvvm.extended_minmax types do not match its format");
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    ExtendedMinMaxFormatAttr::register(ctx);
    ExtendedMinMaxOperationAttr::register(ctx);
    ExtendedMinMaxSubnormalAttr::register(ctx);
    ExtendedMinMaxNanAttr::register(ctx);
    ExtendedMinMaxXorSignAbsAttr::register(ctx);
    ExtendedMinMaxOp::register(ctx);
}
"##,
    );
    output
}

pub(in crate::render) fn render_dialect_prmt(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for the closed generated `prmt` family.

use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::NResultsInterface,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.prmt_mode", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PrmtModeAttr {
    Generic,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
}

/// Byte permutation whose exact mode is carried by an attribute.
#[pliron_op(
    name = "nvvm.prmt",
    format,
    interfaces = [NResultsInterface<1>],
    attributes = (nvvm_prmt_mode: PrmtModeAttr)
)]
pub struct PrmtOp;

impl PrmtOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(ctx: &mut Context, operands: Vec<Value>, mode: PrmtModeAttr) -> Ptr<Operation> {
        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.into()],
            operands,
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_prmt_mode(ctx, mode);
        this.get_operation()
    }
}

impl Verify for PrmtOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let Some(mode) = self.get_attr_nvvm_prmt_mode(ctx) else {
            return verify_err!(op.loc(), "nvvm.prmt requires a mode attribute");
        };
        let expected_operands = match &*mode {
            PrmtModeAttr::Generic | PrmtModeAttr::F4e | PrmtModeAttr::B4e => 3,
            PrmtModeAttr::Rc8 | PrmtModeAttr::Ecl | PrmtModeAttr::Ecr | PrmtModeAttr::Rc16 => 2,
        };
        if op.get_num_operands() != expected_operands || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.prmt {:?} requires {} operands and one result",
                mode,
                expected_operands
            );
        }
        for index in 0..expected_operands {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 32)
            {
                return verify_err!(op.loc(), "nvvm.prmt operand {} must be a 32-bit integer", index);
            }
        }
        let result_ty = op.get_result(0).get_type(ctx);
        if result_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_none_or(|integer| integer.width() != 32)
        {
            return verify_err!(op.loc(), "nvvm.prmt result must be a 32-bit integer");
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    PrmtModeAttr::register(ctx);
    PrmtOp::register(ctx);
}
"##,
    );
    output
}
