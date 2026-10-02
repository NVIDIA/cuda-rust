/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::CatalogFile;
use crate::render::common::rust_header;
use crate::render::families::{
    ldmatrix, ldmatrix_compat_op, movmatrix, register_mma_attr_variants, register_mma_carriers,
    register_mma_compat_op_type, register_mmas, sparse_mma_attr_variants, sparse_mma_carriers,
    sparse_mma_selector_values, sparse_mmas, stmatrices, stmatrix_variant,
};
use std::fmt::Write as _;

pub(in crate::render) fn render_dialect_movmatrix(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    if movmatrix(catalog).next().is_none() {
        output.push_str(
            "//! Structural operation for generated in-register matrix transpose.\n\nuse pliron::context::Context;\n\npub(super) fn register(_ctx: &mut Context) {}\n",
        );
        return output;
    }
    let record = movmatrix(catalog).next().unwrap();
    output.push_str(
        "//! Structural operation for generated in-register matrix transpose.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    writeln!(output, "/// {}", record.summary).unwrap();
    writeln!(
        output,
        "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],\n)]\npub struct {};\n",
        record.dialect.op_name, record.dialect.op_type
    )
    .unwrap();
    writeln!(
        output,
        "impl {} {{\n    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, value: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![value],\n            vec![],\n            0,\n        )\n    }}\n}}\n",
        record.dialect.op_type
    )
    .unwrap();
    writeln!(
        output,
        "impl Verify for {} {{\n    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 1 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_i32(ctx, op.get_operand(0).get_type(ctx))\n            || !is_i32(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n\npub(super) fn register(ctx: &mut Context) {{\n    {}::register(ctx);\n}}\n",
        record.dialect.op_type,
        format!("{} requires one operand and one result", record.dialect.op_name),
        format!(
            "{} operand and result must be 32-bit integers",
            record.dialect.op_name
        ),
        record.dialect.op_type,
    )
    .unwrap();
    output
}

pub(in crate::render) fn render_dialect_ldmatrix(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! Structural operation and compatibility carriers for generated `ldmatrix`.

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

#[pliron_attr(name = "nvvm.ldmatrix_shape", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixShapeAttr {
    M8n8,
    M8n16,
    M16n16,
}

impl LdmatrixShapeAttr {
    pub const fn register_count(&self, multiplicity: &LdmatrixMultiplicityAttr) -> usize {
        let matrices = multiplicity.register_count();
        match self {
            Self::M8n8 | Self::M8n16 => matrices,
            Self::M16n16 => matrices * 2,
        }
    }
}

#[pliron_attr(name = "nvvm.ldmatrix_multiplicity", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixMultiplicityAttr {
    X1,
    X2,
    X4,
}

impl LdmatrixMultiplicityAttr {
    pub const fn register_count(&self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

#[pliron_attr(name = "nvvm.ldmatrix_layout", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixLayoutAttr {
    Normal,
    Transposed,
}

#[pliron_attr(name = "nvvm.ldmatrix_element", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixElementAttr {
    B16,
    B8,
    B8x16B4x16P64,
    B8x16B6x16P32,
}

#[pliron_attr(name = "nvvm.ldmatrix_state_space", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixStateSpaceAttr {
    Shared,
}

/// A warp-cooperative matrix load whose exact variant is carried by attributes.
#[pliron_op(
    name = "nvvm.ldmatrix",
    format,
    interfaces = [NOpdsInterface<1>],
    attributes = (
        nvvm_ldmatrix_shape: LdmatrixShapeAttr,
        nvvm_ldmatrix_multiplicity: LdmatrixMultiplicityAttr,
        nvvm_ldmatrix_layout: LdmatrixLayoutAttr,
        nvvm_ldmatrix_element: LdmatrixElementAttr,
        nvvm_ldmatrix_state_space: LdmatrixStateSpaceAttr
    )
)]
pub struct LdmatrixOp;

impl LdmatrixOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(
        ctx: &mut Context,
        address: Value,
        shape: LdmatrixShapeAttr,
        multiplicity: LdmatrixMultiplicityAttr,
        layout: LdmatrixLayoutAttr,
        element: LdmatrixElementAttr,
        state_space: LdmatrixStateSpaceAttr,
    ) -> Ptr<Operation> {
        let register_count = shape.register_count(&multiplicity);
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![u32_ty.into(); register_count],
            vec![address],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_ldmatrix_shape(ctx, shape);
        this.set_attr_nvvm_ldmatrix_multiplicity(ctx, multiplicity);
        this.set_attr_nvvm_ldmatrix_layout(ctx, layout);
        this.set_attr_nvvm_ldmatrix_element(ctx, element);
        this.set_attr_nvvm_ldmatrix_state_space(ctx, state_space);
        this.get_operation()
    }
}

impl Verify for LdmatrixOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operands: Vec<_> = op.operands().collect();
        if operands.len() != 1 {
            return verify_err!(op.loc(), "nvvm.ldmatrix requires exactly one address operand");
        }

        let Some(shape) = self.get_attr_nvvm_ldmatrix_shape(ctx) else {
            return verify_err!(op.loc(), "nvvm.ldmatrix requires a shape attribute");
        };
        let Some(multiplicity) = self.get_attr_nvvm_ldmatrix_multiplicity(ctx) else {
            return verify_err!(op.loc(), "nvvm.ldmatrix requires a multiplicity attribute");
        };
        let layout = self.get_attr_nvvm_ldmatrix_layout(ctx);
        let element = self.get_attr_nvvm_ldmatrix_element(ctx);
        let expected_pointee_width = if element.as_deref() == Some(&LdmatrixElementAttr::B16) {
            32
        } else {
            8
        };
        let address_type = operands[0].get_type(ctx);
        let address_type = address_type.deref(ctx);
        if let Some(pointer) = address_type.downcast_ref::<MirPtrType>() {
            if !matches!(pointer.address_space, address_space::GENERIC | address_space::SHARED) {
                return verify_err!(
                    op.loc(),
                    "nvvm.ldmatrix pointer must be generic (p0) or shared (p3), not address space {}",
                    pointer.address_space
                );
            }
            let pointee = pointer.pointee.deref(ctx);
            let Some(pointee) = pointee.downcast_ref::<IntegerType>() else {
                return verify_err!(op.loc(), "nvvm.ldmatrix pointer must point to an unsigned integer");
            };
            if pointee.width() != expected_pointee_width
                || pointee.signedness() != Signedness::Unsigned
            {
                return verify_err!(
                    op.loc(),
                    "nvvm.ldmatrix pointer element width does not match its matrix element format"
                );
            }
        } else if !address_type
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| {
                integer.width() == 32 && integer.signedness() == Signedness::Unsigned
            })
        {
            return verify_err!(
                op.loc(),
                "nvvm.ldmatrix address must be a compatible MIR pointer or u32 shared address"
            );
        }
        let supported = matches!(
            (&*shape, layout.as_deref(), element.as_deref()),
            (LdmatrixShapeAttr::M8n8, Some(_), Some(LdmatrixElementAttr::B16))
            |
            (
                LdmatrixShapeAttr::M8n16,
                Some(LdmatrixLayoutAttr::Normal),
                Some(
                    LdmatrixElementAttr::B8x16B4x16P64
                        | LdmatrixElementAttr::B8x16B6x16P32,
                ),
            )
            |
            (
                LdmatrixShapeAttr::M16n16,
                Some(LdmatrixLayoutAttr::Transposed),
                Some(
                    LdmatrixElementAttr::B8
                        | LdmatrixElementAttr::B8x16B4x16P64
                        | LdmatrixElementAttr::B8x16B6x16P32,
                ),
            )
        );
        if !supported
            || (*shape == LdmatrixShapeAttr::M16n16
                && *multiplicity == LdmatrixMultiplicityAttr::X4)
            || self.get_attr_nvvm_ldmatrix_state_space(ctx).as_deref()
                != Some(&LdmatrixStateSpaceAttr::Shared)
        {
            return verify_err!(op.loc(), "nvvm.ldmatrix has a missing or unsupported variant attribute");
        }

        let register_count = shape.register_count(&multiplicity);
        if op.get_num_results() != register_count {
            return verify_err!(
                op.loc(),
                "nvvm.ldmatrix {:?} requires {} u32 results",
                multiplicity,
                register_count
            );
        }
        for index in 0..register_count {
            let ty = op.get_result(index).get_type(ctx);
            let ty_object = ty.deref(ctx);
            let Some(integer) = ty_object.downcast_ref::<IntegerType>() else {
                return verify_err!(op.loc(), "nvvm.ldmatrix result {} must be u32", index);
            };
            if integer.width() != 32 || integer.signedness() != Signedness::Unsigned {
                return verify_err!(op.loc(), "nvvm.ldmatrix result {} must be u32", index);
            }
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    LdmatrixShapeAttr::register(ctx);
    LdmatrixMultiplicityAttr::register(ctx);
    LdmatrixLayoutAttr::register(ctx);
    LdmatrixElementAttr::register(ctx);
    LdmatrixStateSpaceAttr::register(ctx);
    LdmatrixOp::register(ctx);
}
"##,
    );

    let compatibility = ldmatrix(catalog)
        .filter_map(|record| ldmatrix_compat_op(record).map(|compat| (record, compat)))
        .collect::<Vec<_>>();
    assert_eq!(compatibility.len(), 6);

    let mut definitions = String::from(
        r#"
fn verify_compat_ldmatrix(
    ctx: &Context,
    op: Ptr<Operation>,
    op_name: &str,
    result_count: usize,
) -> Result<(), Error> {
    let op = op.deref(ctx);
    let operands: Vec<_> = op.operands().collect();
    if operands.len() != 1 {
        return verify_err!(op.loc(), "{} requires one shared-memory address", op_name);
    }
    let address_type = operands[0].get_type(ctx);
    let address_type = address_type.deref(ctx);
    let is_pointer = address_type.downcast_ref::<MirPtrType>().is_some();
    let is_u32 = address_type
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            integer.width() == 32 && integer.signedness() == Signedness::Unsigned
        });
    if !is_pointer && !is_u32 {
        return verify_err!(op.loc(), "{} operand 0 must be a MIR pointer or u32", op_name);
    }
    if op.get_num_results() != result_count {
        return verify_err!(
            op.loc(),
            "{} requires {} register results",
            op_name,
            result_count
        );
    }
    for index in 0..result_count {
        let ty = op.get_result(index).get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(integer) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(op.loc(), "{} result {} must be an integer", op_name, index);
        };
        if integer.width() != 32 {
            return verify_err!(op.loc(), "{} result {} must be 32 bits", op_name, index);
        }
    }
    Ok(())
}

"#,
    );
    for (record, (op_type, op_name)) in &compatibility {
        let result_count = record
            .ldmatrix
            .as_ref()
            .expect("ldmatrix compatibility record")
            .variant
            .multiplicity
            .register_count();
        writeln!(
            definitions,
            "/// Compatibility carrier for `{op_name}`.\n#[pliron_op(\n    name = {op_name:?},\n    format,\n    interfaces = [NOpdsInterface<1>, NResultsInterface<{result_count}>],\n)]\npub struct {op_type};\n\nimpl {op_type} {{\n    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n}}\n\nimpl Verify for {op_type} {{\n    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_compat_ldmatrix(ctx, self.get_operation(), {op_name:?}, {result_count})\n    }}\n}}\n"
        )
        .unwrap();
    }
    let register_start = output
        .find("pub(super) fn register(ctx: &mut Context) {")
        .expect("ldmatrix register function");
    output.insert_str(register_start, &definitions);
    let register_anchor = "    LdmatrixOp::register(ctx);\n";
    let mut registrations = String::from(register_anchor);
    for (_, (op_type, _)) in compatibility {
        writeln!(registrations, "    {op_type}::register(ctx);").unwrap();
    }
    output = output.replacen(register_anchor, &registrations, 1);
    output
}

pub(in crate::render) fn render_dialect_stmatrix(catalog: &CatalogFile, hash: &str) -> String {
    assert_eq!(stmatrices(catalog).count(), 4);
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! Structural operations for the four generated `stmatrix` stores.

use dialect_mir::types::MirPtrType;
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

fn verify_stmatrix_operands(
    ctx: &Context,
    op: Ptr<Operation>,
    op_name: &str,
    register_count: usize,
) -> Result<(), Error> {
    let op = op.deref(ctx);
    let operands: Vec<_> = op.operands().collect();
    if operands.len() != register_count + 1 {
        return verify_err!(
            op.loc(),
            "{} requires one pointer and {} register operands",
            op_name,
            register_count
        );
    }
    if operands[0]
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_none()
    {
        return verify_err!(op.loc(), "{} operand 0 must be a MIR pointer", op_name);
    }
    for (index, register) in operands.iter().enumerate().skip(1) {
        let ty = register.get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(integer) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(
                op.loc(),
                "{} register operand {} must be an integer",
                op_name,
                index - 1
            );
        };
        if integer.width() != 32 {
            return verify_err!(
                op.loc(),
                "{} register operand {} must be 32 bits",
                op_name,
                index - 1
            );
        }
    }
    Ok(())
}

"#,
    );
    for record in stmatrices(catalog) {
        let (multiplicity, _) = stmatrix_variant(record).expect("stmatrix variant");
        let count = multiplicity.register_count();
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{}>, NResultsInterface<0>],\n)]",
            record.dialect.op_name,
            count + 1
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_stmatrix_operands(ctx, self.get_operation(), {:?}, {count})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in stmatrices(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_register_mma(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! One structural operation for generated register-only `mma.sync` variants.

use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{FP32Type, FP64Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.register_mma_shape", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum RegisterMmaShapeAttr { M8n8k4, M8n8k16, M8n8k32, M8n8k128, M16n8k4, M16n8k8, M16n8k16, M16n8k32, M16n8k64, M16n8k128, M16n8k256 }

#[pliron_attr(name = "nvvm.register_mma_operation", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum RegisterMmaOperationAttr { Multiply, AndPopc, XorPopc }

#[pliron_attr(name = "nvvm.register_mma_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum RegisterMmaKindAttr { Standard, F8f6f4, Mxf8f6f4 }

#[pliron_attr(name = "nvvm.register_mma_accumulator", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum RegisterMmaAccumulatorAttr { F16, F32, F64, S32 }

#[pliron_attr(name = "nvvm.register_mma_element", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum RegisterMmaElementAttr { Bf16, E2m1, E2m3, E3m2, E4m3, E5m2, F16, Tf32, F64, B1, S4, U4, S8, U8 }

#[pliron_attr(name = "nvvm.register_mma_layout", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum RegisterMmaLayoutAttr { Row, Col }

#[pliron_attr(name = "nvvm.register_mma_overflow", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum RegisterMmaOverflowAttr { NotApplicable, Wrapping, Satfinite }

#[pliron_op(
    name = "nvvm.register_mma",
    format,
    attributes = (
        nvvm_register_mma_shape: RegisterMmaShapeAttr,
        nvvm_register_mma_operation: RegisterMmaOperationAttr,
        nvvm_register_mma_kind: RegisterMmaKindAttr,
        nvvm_register_mma_accumulator: RegisterMmaAccumulatorAttr,
        nvvm_register_mma_a_element: RegisterMmaElementAttr,
        nvvm_register_mma_b_element: RegisterMmaElementAttr,
        nvvm_register_mma_a_layout: RegisterMmaLayoutAttr,
        nvvm_register_mma_b_layout: RegisterMmaLayoutAttr,
        nvvm_register_mma_overflow: RegisterMmaOverflowAttr
    )
)]
pub struct RegisterMmaOp;

impl RegisterMmaOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    /// Defaults older multiply-only IR to multiply.
    pub fn operation_or_multiply(&self, ctx: &Context) -> RegisterMmaOperationAttr {
        self.get_attr_nvvm_register_mma_operation(ctx)
            .as_deref()
            .cloned()
            .unwrap_or(RegisterMmaOperationAttr::Multiply)
    }

    /// Infers the kind used by older generated IR.
    pub fn kind_or_inferred(&self, ctx: &Context) -> RegisterMmaKindAttr {
        if let Some(kind) = self.get_attr_nvvm_register_mma_kind(ctx).as_deref() {
            return kind.clone();
        }
        let low_format = |element: Option<&RegisterMmaElementAttr>| {
            matches!(
                element,
                Some(
                    RegisterMmaElementAttr::E2m1
                        | RegisterMmaElementAttr::E2m3
                        | RegisterMmaElementAttr::E3m2
                        | RegisterMmaElementAttr::E4m3
                        | RegisterMmaElementAttr::E5m2
                )
            )
        };
        let old_f8f6f4 = self.get_attr_nvvm_register_mma_shape(ctx).as_deref()
            == Some(&RegisterMmaShapeAttr::M16n8k32)
            && low_format(self.get_attr_nvvm_register_mma_a_element(ctx).as_deref())
            && low_format(self.get_attr_nvvm_register_mma_b_element(ctx).as_deref());
        if old_f8f6f4 {
            RegisterMmaKindAttr::F8f6f4
        } else {
            RegisterMmaKindAttr::Standard
        }
    }
}

#[derive(Clone, Copy)]
enum MmaCarrier { F32, F64, I32, U16, U32 }

fn is_carrier(ctx: &Context, ty: TypeHandle, carrier: MmaCarrier) -> bool {
    match carrier {
        MmaCarrier::F32 => ty.deref(ctx).downcast_ref::<FP32Type>().is_some(),
        MmaCarrier::F64 => ty.deref(ctx).downcast_ref::<FP64Type>().is_some(),
        MmaCarrier::I32 | MmaCarrier::U16 | MmaCarrier::U32 => {
            let expected = if matches!(carrier, MmaCarrier::I32) {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            ty.deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| {
                    integer.width()
                        == if matches!(carrier, MmaCarrier::U16) { 16 } else { 32 }
                        && integer.signedness() == expected
                })
        }
    }
}

impl Verify for RegisterMmaOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operation = self.operation_or_multiply(ctx);
        let kind = self.kind_or_inferred(ctx);
        let recipe: (&[MmaCarrier], &[MmaCarrier]) = match (
            self.get_attr_nvvm_register_mma_shape(ctx).as_deref(),
            operation,
            kind,
            self.get_attr_nvvm_register_mma_accumulator(ctx).as_deref(),
            self.get_attr_nvvm_register_mma_a_element(ctx).as_deref(),
            self.get_attr_nvvm_register_mma_b_element(ctx).as_deref(),
            self.get_attr_nvvm_register_mma_a_layout(ctx).as_deref(),
            self.get_attr_nvvm_register_mma_b_layout(ctx).as_deref(),
            self.get_attr_nvvm_register_mma_overflow(ctx).as_deref(),
        ) {
"#,
    );
    for record in register_mmas(catalog) {
        let (
            shape,
            operation,
            kind,
            accumulator,
            a_element,
            b_element,
            a_layout,
            b_layout,
            overflow,
        ) = register_mma_attr_variants(record);
        let (operands, results) = register_mma_carriers(record);
        writeln!(
            output,
            "            (Some(&{shape}), {operation}, {kind}, Some(&{accumulator}), Some(&{a_element}), Some(&{b_element}), Some(&{a_layout}), Some(&{b_layout}), Some(&{overflow})) => ({operands}, {results}),"
        )
        .unwrap();
    }
    output.push_str(
        r#"            _ => return verify_err!(op.loc(), "nvvm.register_mma has a missing or unsupported variant"),
        };
        let operands: Vec<_> = op.operands().collect();
        if operands.len() != recipe.0.len() || op.get_num_results() != recipe.1.len() {
            return verify_err!(op.loc(), "nvvm.register_mma has the wrong register count");
        }
        for (index, (operand, carrier)) in operands.iter().zip(recipe.0).enumerate() {
            if !is_carrier(ctx, operand.get_type(ctx), *carrier) {
                return verify_err!(op.loc(), "nvvm.register_mma operand {} has the wrong carrier type", index);
            }
        }
        for (index, carrier) in recipe.1.iter().enumerate() {
            if !is_carrier(ctx, op.get_result(index).get_type(ctx), *carrier) {
                return verify_err!(op.loc(), "nvvm.register_mma result {} has the wrong carrier type", index);
            }
        }
        Ok(())
    }
}

"#,
    );
    output.push_str(
        r#"
fn is_compat_carrier(ctx: &Context, ty: TypeHandle, carrier: MmaCarrier) -> bool {
    match carrier {
        MmaCarrier::F32 => ty.deref(ctx).downcast_ref::<FP32Type>().is_some(),
        MmaCarrier::F64 => ty.deref(ctx).downcast_ref::<FP64Type>().is_some(),
        MmaCarrier::I32 | MmaCarrier::U16 | MmaCarrier::U32 => {
            let expected_width = if matches!(carrier, MmaCarrier::U16) {
                16
            } else {
                32
            };
            ty.deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == expected_width)
        }
    }
}

fn verify_compat_register_mma(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
    op_name: &str,
    operand_carriers: &[MmaCarrier],
    result_carriers: &[MmaCarrier],
) -> Result<(), Error> {
    let op = op_ptr.deref(ctx);
    let operands: Vec<_> = op.operands().collect();
    if operands.len() != operand_carriers.len() || op.get_num_results() != result_carriers.len() {
        return verify_err!(op.loc(), "{} has the wrong register count", op_name);
    }
    for (index, (operand, carrier)) in operands.iter().zip(operand_carriers).enumerate() {
        if !is_compat_carrier(ctx, operand.get_type(ctx), *carrier) {
            return verify_err!(op.loc(), "{} operand {} has the wrong carrier type", op_name, index);
        }
    }
    for (index, carrier) in result_carriers.iter().enumerate() {
        if !is_compat_carrier(ctx, op.get_result(index).get_type(ctx), *carrier) {
            return verify_err!(op.loc(), "{} result {} has the wrong carrier type", op_name, index);
        }
    }
    Ok(())
}

"#,
    );
    for record in
        register_mmas(catalog).filter(|record| register_mma_compat_op_type(record).is_some())
    {
        let op_type = register_mma_compat_op_type(record).unwrap();
        let op_name = format!("nvvm.{}", record.id);
        let operand_count = record.dialect.operands.len();
        let result_count = record.dialect.results.len();
        let (operands, results) = register_mma_carriers(record);
        writeln!(output, "/// Compatibility carrier for `{}`.", record.id).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {op_name:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<{result_count}>],\n)]\npub struct {op_type};\n"
        )
        .unwrap();
        writeln!(
            output,
            "impl {op_type} {{\n    pub fn new(op: Ptr<Operation>) -> Self {{ Self {{ op }} }}\n}}\n"
        )
        .unwrap();
        writeln!(
            output,
            "impl Verify for {op_type} {{\n    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_compat_register_mma(ctx, self.get_operation(), {op_name:?}, {operands}, {results})\n    }}\n}}\n"
        )
        .unwrap();
    }
    output.push_str(
        "pub(super) fn register(ctx: &mut Context) {\n    RegisterMmaShapeAttr::register(ctx);\n    RegisterMmaOperationAttr::register(ctx);\n    RegisterMmaKindAttr::register(ctx);\n    RegisterMmaAccumulatorAttr::register(ctx);\n    RegisterMmaElementAttr::register(ctx);\n    RegisterMmaLayoutAttr::register(ctx);\n    RegisterMmaOverflowAttr::register(ctx);\n    RegisterMmaOp::register(ctx);\n",
    );
    for record in
        register_mmas(catalog).filter(|record| register_mma_compat_op_type(record).is_some())
    {
        writeln!(
            output,
            "    {}::register(ctx);",
            register_mma_compat_op_type(record).unwrap()
        )
        .unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_sparse_mma(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! One structural operation for generated sparse MMA variants.

use dialect_mir::ops::MirConstantOp;
use pliron::{
    attribute::Attribute,
    builtin::{attributes::IntegerAttr, ops::ConstantOp, types::{FP32Type, IntegerType, Signedness}},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.sparse_mma_shape", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum SparseMmaShapeAttr { M16n8k8, M16n8k16, M16n8k32, M16n8k64, M16n8k128 }

#[pliron_attr(name = "nvvm.sparse_mma_accumulator", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum SparseMmaAccumulatorAttr { F16, F32, S32 }

#[pliron_attr(name = "nvvm.sparse_mma_element", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum SparseMmaElementAttr { F16, Bf16, Tf32, E2m1, E2m3, E3m2, E4m3, E5m2, S4, U4, S8, U8 }

#[pliron_attr(name = "nvvm.sparse_mma_layout", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum SparseMmaLayoutAttr { Row, Col }

#[pliron_attr(name = "nvvm.sparse_mma_overflow", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum SparseMmaOverflowAttr { NotApplicable, Wrapping, Satfinite }

#[pliron_attr(name = "nvvm.sparse_mma_metadata", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum SparseMmaMetadataAttr { Standard, Ordered }

#[pliron_attr(name = "nvvm.sparse_mma_selector", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
#[allow(clippy::enum_variant_names)]
pub enum SparseMmaSelectorAttr { ImmediateZeroThroughThree, ImmediateZeroOrOne, ImmediateZero }

#[pliron_op(
    name = "nvvm.sparse_mma",
    format,
    attributes = (
        nvvm_sparse_mma_shape: SparseMmaShapeAttr,
        nvvm_sparse_mma_accumulator: SparseMmaAccumulatorAttr,
        nvvm_sparse_mma_a_element: SparseMmaElementAttr,
        nvvm_sparse_mma_b_element: SparseMmaElementAttr,
        nvvm_sparse_mma_a_layout: SparseMmaLayoutAttr,
        nvvm_sparse_mma_b_layout: SparseMmaLayoutAttr,
        nvvm_sparse_mma_overflow: SparseMmaOverflowAttr,
        nvvm_sparse_mma_metadata: SparseMmaMetadataAttr,
        nvvm_sparse_mma_selector: SparseMmaSelectorAttr
    )
)]
pub struct SparseMmaOp;

impl SparseMmaOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }
}

#[derive(Clone, Copy)]
enum MmaCarrier { F32, I32, U32 }

fn is_carrier(ctx: &Context, ty: TypeHandle, carrier: MmaCarrier) -> bool {
    match carrier {
        MmaCarrier::F32 => ty.deref(ctx).downcast_ref::<FP32Type>().is_some(),
        MmaCarrier::I32 | MmaCarrier::U32 => {
            let expected = if matches!(carrier, MmaCarrier::I32) {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            ty.deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 32 && integer.signedness() == expected)
        }
    }
}

fn constant_u32(ctx: &Context, value: Value) -> Option<u64> {
    let defining_op = value.defining_op()?;
    if let Some(constant) = Operation::get_op::<MirConstantOp>(defining_op, ctx) {
        return constant.get_attr_value(ctx).map(|value| value.value().to_u64());
    }
    let constant = Operation::get_op::<ConstantOp>(defining_op, ctx)?;
    let value = constant.get_value(ctx);
    (&*value as &dyn Attribute)
        .downcast_ref::<IntegerAttr>()
        .map(|value| value.value().to_u64())
}

impl Verify for SparseMmaOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let (expected_operands, expected_results, selector_upper_exclusive): (&[MmaCarrier], &[MmaCarrier], usize) = match (
            self.get_attr_nvvm_sparse_mma_shape(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_accumulator(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_a_element(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_b_element(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_a_layout(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_b_layout(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_overflow(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_metadata(ctx).as_deref(),
            self.get_attr_nvvm_sparse_mma_selector(ctx).as_deref(),
        ) {
"#,
    );
    for record in sparse_mmas(catalog) {
        let (
            shape,
            accumulator,
            a_element,
            b_element,
            a_layout,
            b_layout,
            overflow,
            metadata,
            selector,
        ) = sparse_mma_attr_variants(record);
        let (expected_operands, expected_results) = sparse_mma_carriers(record);
        let selector_upper_exclusive = sparse_mma_selector_values(record).len();
        writeln!(
            output,
            "            (Some(&{shape}), Some(&{accumulator}), Some(&{a_element}), Some(&{b_element}), Some(&{a_layout}), Some(&{b_layout}), Some(&{overflow}), Some(&{metadata}), Some(&{selector})) => ({expected_operands}, {expected_results}, {selector_upper_exclusive}),"
        )
        .unwrap();
    }
    output.push_str(
        r#"            _ => return verify_err!(op.loc(), "nvvm.sparse_mma has a missing or unsupported variant"),
        };
        let operands: Vec<_> = op.operands().collect();
        if operands.len() != expected_operands.len() || op.get_num_results() != expected_results.len() {
            return verify_err!(op.loc(), "nvvm.sparse_mma has the wrong register count");
        }
        for (index, (operand, carrier)) in operands.iter().zip(expected_operands).enumerate() {
            if !is_carrier(ctx, operand.get_type(ctx), *carrier) {
                return verify_err!(op.loc(), "nvvm.sparse_mma operand {} has the wrong carrier type", index);
            }
        }
        for (index, carrier) in expected_results.iter().enumerate() {
            if !is_carrier(ctx, op.get_result(index).get_type(ctx), *carrier) {
                return verify_err!(op.loc(), "nvvm.sparse_mma result {} has the wrong carrier type", index);
            }
        }
        if constant_u32(ctx, operands[expected_operands.len() - 1])
            .is_none_or(|selector| selector >= selector_upper_exclusive as u64)
        {
            return verify_err!(op.loc(), "nvvm.sparse_mma selector must be a compile-time constant in 0..{}", selector_upper_exclusive);
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    SparseMmaShapeAttr::register(ctx);
    SparseMmaAccumulatorAttr::register(ctx);
    SparseMmaElementAttr::register(ctx);
    SparseMmaLayoutAttr::register(ctx);
    SparseMmaOverflowAttr::register(ctx);
    SparseMmaMetadataAttr::register(ctx);
    SparseMmaSelectorAttr::register(ctx);
    SparseMmaOp::register(ctx);
}
"#,
    );
    output
}
