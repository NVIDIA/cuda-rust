/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    ActiveMaskAdapter, CatalogFile, CpAsyncControlOperation, CpAsyncSourceSize,
    MbarrierBasicAdapter, MbarrierBasicOperation, MbarrierExtendedOperation, MbarrierStateSpace,
    VoteMode, WarpBarrierAdapter, WarpMatchMode, WarpShuffleAdapter, WarpShuffleValueKind,
};
use crate::render::common::{rust_header, source_label};
use crate::render::families::{
    active_masks, cp_async_controls, cp_async_copies, cp_async_mbarriers, dot_product_ptx,
    dot_products, elect_intrinsics, mbarrier_basics, mbarrier_extended, redux, sregs,
    sync_intrinsics, vote_intrinsics, warp_barriers, warp_matches, warp_shuffles,
};
use std::fmt::Write as _;

pub(in crate::render) fn render_dialect_elect(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated warp leader-election operation.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::IntegerType,\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::{TypeHandle, Typed},\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in elect_intrinsics(catalog) {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<1>, NResultsInterface<2>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 1 || op.get_num_results() != 2 {\n            return verify_err!(op.loc(), \"nvvm.elect_sync requires one i32 operand and results [i32, i1]\");\n        }\n        if !is_integer_width(ctx, op.get_operand(0).get_type(ctx), 32)\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), 32)\n            || !is_integer_width(ctx, op.get_result(1).get_type(ctx), 1)\n        {\n            return verify_err!(op.loc(), \"nvvm.elect_sync requires one i32 operand and results [i32, i1]\");\n        }\n        Ok(())\n    }\n}\n\n",
        );
    }
    output.push_str(
        "fn is_integer_width(ctx: &Context, ty: TypeHandle, width: u32) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\npub(super) fn register(ctx: &mut Context) {\n",
    );
    for record in elect_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_sreg(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural NVVM operations for generated special-register reads.\n\nuse pliron::{\n    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},\n    builtin::types::IntegerType,\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in sregs(catalog) {
        let width = record.scalar_width().unwrap();
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Catalog ID `{}`; {} returns one `i{width}` result.",
            record.id,
            source_label(record)
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self {\n");
        writeln!(output, "        Self {{ op }}").unwrap();
        output.push_str("    }\n}\n");
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        output.push_str("    fn verify(&self, ctx: &Context) -> Result<(), Error> {\n");
        writeln!(
            output,
            "        verify_scalar_result(ctx, self.get_operation(), {:?}, {width})",
            record.dialect.op_name
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    output.push_str(
        "fn verify_scalar_result(\n    ctx: &Context,\n    op: Ptr<Operation>,\n    name: &str,\n    width: u32,\n) -> Result<(), Error> {\n    let op = op.deref(ctx);\n    let ty = op.get_result(0).get_type(ctx);\n    let ty_object = ty.deref(ctx);\n    let Some(integer) = ty_object.downcast_ref::<IntegerType>() else {\n        return verify_err!(op.loc(), \"{} result must be an integer\", name);\n    };\n    if integer.width() != width {\n        return verify_err!(\n            op.loc(),\n            \"{} result must be a {}-bit integer\",\n            name,\n            width\n        );\n    }\n    Ok(())\n}\n\npub(super) fn register(ctx: &mut Context) {\n",
    );
    for record in sregs(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_sync(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for generated CTA synchronization.\n\nuse pliron::{\n    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},\n    context::{Context, Ptr},\n    op::Op,\n    operation::Operation,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in sync_intrinsics(catalog) {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    verifier = \"succ\",\n    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n}\n\n",
        );
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in sync_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_vote(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for the generated `vote.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_integer_width(ctx: &Context, ty: pliron::r#type::TypeHandle, width: u32) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\n",
    );
    for record in vote_intrinsics(catalog) {
        let result_width = match record.vote.as_ref().unwrap().mode {
            VoteMode::All | VoteMode::Any | VoteMode::Uni => 1,
            VoteMode::Ballot => 32,
        };
        let result_signedness = if result_width == 32 {
            "Unsigned"
        } else {
            "Signless"
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// Operands are `[member_mask, predicate]`. The generated verifier keeps\n/// the mask, predicate, and result types exact.\n",
        );
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
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, member_mask: Value, predicate: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, {result_width}, Signedness::{result_signedness});\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![member_mask, predicate],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_integer_width(ctx, op.get_operand(0).get_type(ctx), 32)\n            || !is_integer_width(ctx, op.get_operand(1).get_type(ctx), 1)\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), {result_width})\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly two operands [member_mask, predicate] and one result",
                record.dialect.op_name
            ),
            format!(
                "{} requires i32 member mask, i1 predicate, and i{result_width} result",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in vote_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_active_mask(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for the generated active warp mask.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in active_masks(catalog) {
        debug_assert_eq!(
            record.active_mask.as_ref().unwrap().adapter,
            ActiveMaskAdapter::DirectZeroOperandMask
        );
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n\n    pub fn build(ctx: &mut Context) -> Ptr<Operation> {\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![],\n            vec![],\n            0,\n        )\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        let valid = op.get_num_operands() == 0\n            && op.get_num_results() == 1\n            && op.get_result(0).get_type(ctx).deref(ctx)\n                .downcast_ref::<IntegerType>()\n                .is_some_and(|integer| integer.width() == 32);\n        if !valid {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!("{} requires no operands and one i32 result", record.dialect.op_name)
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in active_masks(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_cp_async_copy(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! Structural operations for classic `cp.async` instructions.

use dialect_mir::{ops::MirConstantOp, types::{MirPtrType, address_space}};
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        ops::ConstantOp,
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
use pliron_derive::pliron_op;

fn verify_pointer(
    ctx: &Context,
    op: &Operation,
    value: Value,
    role: &str,
    allowed_address_spaces: &[u32],
) -> Result<(), Error> {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let Some(pointer) = ty.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "{role} must be a MIR pointer");
    };
    if !allowed_address_spaces.contains(&pointer.address_space) {
        return verify_err!(op.loc(), "{role} has the wrong address space");
    }
    Ok(())
}

fn verify_cp_async_copy(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
    has_source_size: bool,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    let expected_operands = if has_source_size { 3 } else { 2 };
    if op.get_num_operands() != expected_operands || op.get_num_results() != 0 {
        return verify_err!(op.loc(), "{name} has the wrong operand or result count");
    }
    verify_pointer(
        ctx,
        &op,
        op.get_operand(0),
        "shared destination",
        &[address_space::GENERIC, address_space::SHARED],
    )?;
    verify_pointer(
        ctx,
        &op,
        op.get_operand(1),
        "global source",
        &[address_space::GENERIC, address_space::GLOBAL],
    )?;
    if has_source_size {
        let ty = op.get_operand(2).get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(integer) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(op.loc(), "source size must be u32");
        };
        if integer.width() != 32 || integer.signedness() != Signedness::Unsigned {
            return verify_err!(op.loc(), "source size must be u32");
        }
    }
    Ok(())
}

fn verify_cp_async_control(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
    has_immediate: bool,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    let expected_operands = usize::from(has_immediate);
    if op.get_num_operands() != expected_operands || op.get_num_results() != 0 {
        return verify_err!(op.loc(), "{name} has the wrong operand or result count");
    }
    if has_immediate {
        let value = op.get_operand(0);
        let ty = value.get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(integer) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(op.loc(), "maximum pending group count must be u32");
        };
        if integer.width() != 32 || integer.signedness() != Signedness::Unsigned {
            return verify_err!(op.loc(), "maximum pending group count must be u32");
        }
        let Some(defining_op) = value.defining_op() else {
            return verify_err!(op.loc(), "maximum pending group count must be a compile-time constant");
        };
        if Operation::get_op::<MirConstantOp>(defining_op, ctx).is_none()
            && Operation::get_op::<ConstantOp>(defining_op, ctx).is_none()
        {
            return verify_err!(op.loc(), "maximum pending group count must be a compile-time constant");
        }
    }
    Ok(())
}

fn verify_cp_async_mbarrier(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    if op.get_num_operands() != 1 || op.get_num_results() != 0 {
        return verify_err!(op.loc(), "{name} has the wrong operand or result count");
    }
    let ty = op.get_operand(0).get_type(ctx);
    let ty = ty.deref(ctx);
    let Some(pointer) = ty.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "mbarrier address must be a MIR pointer");
    };
    let pointee = pointer.pointee.deref(ctx);
    let valid_pointee = pointee
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            integer.width() == 64 && integer.signedness() == Signedness::Unsigned
        });
    if !pointer.is_mutable
        || !valid_pointee
        || !matches!(
            pointer.address_space,
            address_space::GENERIC | address_space::SHARED
        )
    {
        return verify_err!(
            op.loc(),
            "mbarrier address must be a mutable generic/shared pointer to u64"
        );
    }
    Ok(())
}

"#,
    );
    for record in cp_async_copies(catalog) {
        let copy = record.cp_async_copy.as_ref().unwrap();
        let dynamic = copy.source_size == CpAsyncSourceSize::Runtime;
        let operand_count = if dynamic { 3 } else { 2 };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<0>],\n)]\npub struct {};",
            record.dialect.op_name, record.dialect.op_type
        )
        .unwrap();
        writeln!(output, "impl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n");
        if dynamic {
            output.push_str(
                "    pub fn build(ctx: &mut Context, shared_dst: Value, global_src: Value, source_size: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![shared_dst, global_src, source_size], vec![], 0)\n    }\n",
            );
        } else {
            output.push_str(
                "    pub fn build(ctx: &mut Context, shared_dst: Value, global_src: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![shared_dst, global_src], vec![], 0)\n    }\n",
            );
        }
        output.push_str("}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_cp_async_copy(ctx, self.get_operation(), {:?}, {dynamic})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    for record in cp_async_controls(catalog) {
        let control = record.cp_async_control.as_ref().unwrap();
        let has_immediate = control.operation == CpAsyncControlOperation::WaitGroup;
        let operand_count = usize::from(has_immediate);
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<0>],\n)]\npub struct {};",
            record.dialect.op_name, record.dialect.op_type
        )
        .unwrap();
        writeln!(output, "impl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n");
        if has_immediate {
            output.push_str(
                "    pub fn build(ctx: &mut Context, max_pending: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![max_pending], vec![], 0)\n    }\n",
            );
        } else {
            output.push_str(
                "    pub fn build(ctx: &mut Context) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0)\n    }\n",
            );
        }
        output.push_str("}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_cp_async_control(ctx, self.get_operation(), {:?}, {has_immediate})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    for record in cp_async_mbarriers(catalog) {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],\n)]\npub struct {};",
            record.dialect.op_name, record.dialect.op_type
        )
        .unwrap();
        writeln!(output, "impl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n    pub fn build(ctx: &mut Context, barrier: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![barrier], vec![], 0)\n    }\n}\n\n",
        );
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_cp_async_mbarrier(ctx, self.get_operation(), {:?})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in cp_async_copies(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    for record in cp_async_controls(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    for record in cp_async_mbarriers(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_mbarrier_basic(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! Structural operations for the basic shared-memory mbarrier lifecycle.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
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
use pliron_derive::pliron_op;

#[derive(Clone, Copy)]
enum MbarrierBasicShape {
    Init,
    Arrive,
    ArriveNoComplete,
    TestWait,
    Inval,
}

fn is_integer(
    ctx: &Context,
    value: Value,
    width: u32,
    signedness: Signedness,
) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            integer.width() == width && integer.signedness() == signedness
        })
}

fn verify_barrier_pointer(ctx: &Context, op: &Operation, value: Value) -> Result<(), Error> {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let Some(pointer) = ty.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "mbarrier address must be a MIR pointer");
    };
    if !matches!(pointer.address_space, address_space::GENERIC | address_space::SHARED) {
        return verify_err!(op.loc(), "mbarrier address must be generic or shared");
    }
    Ok(())
}

fn verify_mbarrier_basic(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
    shape: MbarrierBasicShape,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    let (operands, results) = match shape {
        MbarrierBasicShape::Init => (2, 0),
        MbarrierBasicShape::Arrive => (1, 1),
        MbarrierBasicShape::ArriveNoComplete | MbarrierBasicShape::TestWait => (2, 1),
        MbarrierBasicShape::Inval => (1, 0),
    };
    if op.get_num_operands() != operands || op.get_num_results() != results {
        return verify_err!(op.loc(), "{name} has the wrong operand or result count");
    }
    verify_barrier_pointer(ctx, &op, op.get_operand(0))?;
    match shape {
        MbarrierBasicShape::Init => {
            if !is_integer(ctx, op.get_operand(1), 32, Signedness::Unsigned) {
                return verify_err!(op.loc(), "mbarrier expected count must be u32");
            }
        }
        MbarrierBasicShape::Arrive => {
            if !is_integer(ctx, op.get_result(0), 64, Signedness::Unsigned) {
                return verify_err!(op.loc(), "mbarrier arrival token must be u64");
            }
        }
        MbarrierBasicShape::ArriveNoComplete => {
            if !is_integer(ctx, op.get_operand(1), 32, Signedness::Unsigned)
                || !is_integer(ctx, op.get_result(0), 64, Signedness::Unsigned)
            {
                return verify_err!(op.loc(), "mbarrier no-complete arrival requires a u32 count and u64 opaque state");
            }
        }
        MbarrierBasicShape::TestWait => {
            if !is_integer(ctx, op.get_operand(1), 64, Signedness::Unsigned)
                || !is_integer(ctx, op.get_result(0), 1, Signedness::Signless)
            {
                return verify_err!(op.loc(), "mbarrier test-wait requires a u64 token and i1 result");
            }
        }
        MbarrierBasicShape::Inval => {}
    }
    Ok(())
}

"#,
    );
    for record in mbarrier_basics(catalog) {
        let mbarrier = record.mbarrier_basic.as_ref().unwrap();
        debug_assert_eq!(mbarrier.state_space, MbarrierStateSpace::Shared);
        let (operand_count, result_count, shape, build) = match mbarrier.operation {
            MbarrierBasicOperation::Init => {
                debug_assert_eq!(
                    mbarrier.adapter,
                    MbarrierBasicAdapter::InitPointerCountToVoid
                );
                (
                    2,
                    0,
                    "Init",
                    "    pub fn build(ctx: &mut Context, barrier: Value, expected_count: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![barrier, expected_count], vec![], 0)\n    }\n",
                )
            }
            MbarrierBasicOperation::Arrive => {
                debug_assert_eq!(mbarrier.adapter, MbarrierBasicAdapter::ArrivePointerToToken);
                (
                    1,
                    1,
                    "Arrive",
                    "    pub fn build(ctx: &mut Context, barrier: Value) -> Ptr<Operation> {\n        let token_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![token_ty.into()], vec![barrier], vec![], 0)\n    }\n",
                )
            }
            MbarrierBasicOperation::ArriveNoComplete => {
                debug_assert_eq!(
                    mbarrier.adapter,
                    MbarrierBasicAdapter::ArriveNoCompletePointerCountToToken
                );
                (
                    2,
                    1,
                    "ArriveNoComplete",
                    "    pub fn build(ctx: &mut Context, barrier: Value, count: Value) -> Ptr<Operation> {\n        let token_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![token_ty.into()], vec![barrier, count], vec![], 0)\n    }\n",
                )
            }
            MbarrierBasicOperation::TestWait => {
                debug_assert_eq!(
                    mbarrier.adapter,
                    MbarrierBasicAdapter::TestWaitPointerTokenToPredicate
                );
                (
                    2,
                    1,
                    "TestWait",
                    "    pub fn build(ctx: &mut Context, barrier: Value, token: Value) -> Ptr<Operation> {\n        let predicate_ty = IntegerType::get(ctx, 1, Signedness::Signless);\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![predicate_ty.into()], vec![barrier, token], vec![], 0)\n    }\n",
                )
            }
            MbarrierBasicOperation::Inval => {
                debug_assert_eq!(mbarrier.adapter, MbarrierBasicAdapter::InvalPointerToVoid);
                (
                    1,
                    0,
                    "Inval",
                    "    pub fn build(ctx: &mut Context, barrier: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![barrier], vec![], 0)\n    }\n",
                )
            }
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<{result_count}>],\n)]\npub struct {};",
            record.dialect.op_name, record.dialect.op_type
        )
        .unwrap();
        writeln!(output, "impl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n");
        output.push_str(build);
        output.push_str("}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_mbarrier_basic(ctx, self.get_operation(), {:?}, MbarrierBasicShape::{shape})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in mbarrier_basics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_mbarrier_extended(
    catalog: &CatalogFile,
    hash: &str,
) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! Structural operations for extended mbarrier and async-proxy instructions.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
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
use pliron_derive::pliron_op;

#[derive(Clone, Copy)]
enum MbarrierExtendedShape {
    PointerU32ToU64,
    RawU64ToVoid,
    PointerU64ToI1,
    PointerU32ToI1,
    ZeroToVoid,
    U32ToVoid,
}

fn is_integer(ctx: &Context, value: Value, width: u32, signedness: Signedness) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            integer.width() == width && integer.signedness() == signedness
        })
}

fn verify_barrier_pointer(ctx: &Context, op: &Operation, value: Value) -> Result<(), Error> {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let Some(pointer) = ty.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "mbarrier address must be a MIR pointer");
    };
    if !matches!(pointer.address_space, address_space::GENERIC | address_space::SHARED) {
        return verify_err!(op.loc(), "mbarrier address must be generic or shared");
    }
    Ok(())
}

fn verify_mbarrier_extended(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
    shape: MbarrierExtendedShape,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    let (operands, results) = match shape {
        MbarrierExtendedShape::PointerU32ToU64
        | MbarrierExtendedShape::PointerU64ToI1
        | MbarrierExtendedShape::PointerU32ToI1 => (2, 1),
        MbarrierExtendedShape::RawU64ToVoid | MbarrierExtendedShape::U32ToVoid => (1, 0),
        MbarrierExtendedShape::ZeroToVoid => (0, 0),
    };
    if op.get_num_operands() != operands || op.get_num_results() != results {
        return verify_err!(op.loc(), "{name} has the wrong operand or result count");
    }
    match shape {
        MbarrierExtendedShape::PointerU32ToU64 => {
            verify_barrier_pointer(ctx, &op, op.get_operand(0))?;
            if !is_integer(ctx, op.get_operand(1), 32, Signedness::Unsigned)
                || !is_integer(ctx, op.get_result(0), 64, Signedness::Unsigned)
            {
                return verify_err!(op.loc(), "mbarrier arrival requires u32 bytes and a u64 token");
            }
        }
        MbarrierExtendedShape::PointerU64ToI1 => {
            verify_barrier_pointer(ctx, &op, op.get_operand(0))?;
            if !is_integer(ctx, op.get_operand(1), 64, Signedness::Unsigned)
                || !is_integer(ctx, op.get_result(0), 1, Signedness::Signless)
            {
                return verify_err!(op.loc(), "mbarrier wait requires a u64 token and i1 result");
            }
        }
        MbarrierExtendedShape::PointerU32ToI1 => {
            verify_barrier_pointer(ctx, &op, op.get_operand(0))?;
            if !is_integer(ctx, op.get_operand(1), 32, Signedness::Unsigned)
                || !is_integer(ctx, op.get_result(0), 1, Signedness::Signless)
            {
                return verify_err!(op.loc(), "mbarrier wait requires u32 parity and i1 result");
            }
        }
        MbarrierExtendedShape::RawU64ToVoid => {
            if !is_integer(ctx, op.get_operand(0), 64, Signedness::Unsigned) {
                return verify_err!(op.loc(), "remote mbarrier address must be u64");
            }
        }
        MbarrierExtendedShape::U32ToVoid => {
            if !is_integer(ctx, op.get_operand(0), 32, Signedness::Unsigned) {
                return verify_err!(op.loc(), "nanosleep duration must be u32");
            }
        }
        MbarrierExtendedShape::ZeroToVoid => {}
    }
    Ok(())
}

"#,
    );
    for record in mbarrier_extended(catalog) {
        let contract = record.mbarrier_extended.as_ref().unwrap();
        let (operand_count, result_count, shape, build) = match contract.operation {
            MbarrierExtendedOperation::ArriveExpectTxCta
            | MbarrierExtendedOperation::ArriveExpectTxCluster => (
                2,
                1,
                "PointerU32ToU64",
                "    pub fn build(ctx: &mut Context, barrier: Value, bytes: Value) -> Ptr<Operation> {\n        let token_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![token_ty.into()], vec![barrier, bytes], vec![], 0)\n    }\n",
            ),
            MbarrierExtendedOperation::ArriveRemoteCluster => (
                1,
                0,
                "RawU64ToVoid",
                "    pub fn build(ctx: &mut Context, address: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![address], vec![], 0)\n    }\n",
            ),
            MbarrierExtendedOperation::TryWaitTokenCta => (
                2,
                1,
                "PointerU64ToI1",
                "    pub fn build(ctx: &mut Context, barrier: Value, token: Value) -> Ptr<Operation> {\n        let result_ty = IntegerType::get(ctx, 1, Signedness::Signless);\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![result_ty.into()], vec![barrier, token], vec![], 0)\n    }\n",
            ),
            MbarrierExtendedOperation::TryWaitParityCta
            | MbarrierExtendedOperation::TryWaitParityCluster => (
                2,
                1,
                "PointerU32ToI1",
                "    pub fn build(ctx: &mut Context, barrier: Value, parity: Value) -> Ptr<Operation> {\n        let result_ty = IntegerType::get(ctx, 1, Signedness::Signless);\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![result_ty.into()], vec![barrier, parity], vec![], 0)\n    }\n",
            ),
            MbarrierExtendedOperation::FenceProxyAsyncSharedCta
            | MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster
            | MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster
            | MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster => (
                0,
                0,
                "ZeroToVoid",
                "    pub fn build(ctx: &mut Context) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0)\n    }\n",
            ),
            MbarrierExtendedOperation::Nanosleep => (
                1,
                0,
                "U32ToVoid",
                "    pub fn build(ctx: &mut Context, ns: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![ns], vec![], 0)\n    }\n",
            ),
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<{result_count}>],\n)]\npub struct {};",
            record.dialect.op_name, record.dialect.op_type
        )
        .unwrap();
        writeln!(output, "impl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n");
        output.push_str(build);
        output.push_str("}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_mbarrier_extended(ctx, self.get_operation(), {:?}, MbarrierExtendedShape::{shape})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in mbarrier_extended(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_warp_match(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for the generated `match.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_integer_width(ctx: &Context, ty: pliron::r#type::TypeHandle, width: u32) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\n",
    );
    for record in warp_matches(catalog) {
        let warp_match = record.warp_match.as_ref().unwrap();
        let value_width = warp_match.value_width.bits();
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// Operands are `[member_mask, value]`; the result is a 32-bit lane mask.\n",
        );
        if warp_match.mode == WarpMatchMode::All {
            output.push_str(
                "/// LLVM also returns a predicate, which the established API discards.\n",
            );
        }
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n\n    pub fn build(ctx: &mut Context, member_mask: Value, value: Value) -> Ptr<Operation> {\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![member_mask, value],\n            vec![],\n            0,\n        )\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_integer_width(ctx, op.get_operand(0).get_type(ctx), 32)\n            || !is_integer_width(ctx, op.get_operand(1).get_type(ctx), {value_width})\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), 32)\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly [member_mask, value] and one mask result",
                record.dialect.op_name
            ),
            format!(
                "{} requires i32 member mask, i{value_width} value, and i32 result",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in warp_matches(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_warp_barrier(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for generated warp synchronization.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::IntegerType,\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in warp_barriers(catalog) {
        debug_assert_eq!(
            record.warp_barrier.as_ref().unwrap().adapter,
            WarpBarrierAdapter::DirectMemberMask
        );
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// The operand is the 32-bit warp participation mask.\n");
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n\n    pub fn build(ctx: &mut Context, member_mask: Value) -> Ptr<Operation> {\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![],\n            vec![member_mask],\n            vec![],\n            0,\n        )\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 1 || op.get_num_results() != 0 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_i32(ctx, op.get_operand(0).get_type(ctx)) {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly one member-mask operand and no results",
                record.dialect.op_name
            ),
            format!("{} member mask must be i32", record.dialect.op_name),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in warp_barriers(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_warp_shuffle(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for the generated `shfl.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{FP32Type, IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\nfn is_i64(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 64)\n}\n\nfn is_f32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()\n}\n\n",
    );
    for record in warp_shuffles(catalog) {
        let shuffle = record.warp_shuffle.as_ref().unwrap();
        debug_assert!(matches!(
            (shuffle.value_kind, shuffle.adapter),
            (
                WarpShuffleValueKind::I32 | WarpShuffleValueKind::F32,
                WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
            ) | (
                WarpShuffleValueKind::I64,
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
            )
        ));
        let (result_builder, value_check, value_label) = match shuffle.value_kind {
            WarpShuffleValueKind::I32 => (
                "IntegerType::get(ctx, 32, Signedness::Unsigned).into()",
                "is_i32",
                "i32",
            ),
            WarpShuffleValueKind::F32 => ("FP32Type::get(ctx).into()", "is_f32", "f32"),
            WarpShuffleValueKind::I64 => (
                "IntegerType::get(ctx, 64, Signedness::Unsigned).into()",
                "is_i64",
                "i64",
            ),
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// Operands are `[member_mask, value, lane_or_delta]`.\n");
        if shuffle.value_kind == WarpShuffleValueKind::I64 {
            output.push_str(
                "/// Lowering splits the value into two `b32` halves, shuffles both, and reassembles it.\n",
            );
        } else {
            output.push_str(
                "/// Generated lowering inserts the fixed clamp required by the selected shuffle mode.\n",
            );
        }
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(\n        ctx: &mut Context,\n        member_mask: Value,\n        value: Value,\n        lane_or_delta: Value,\n    ) -> Ptr<Operation> {{\n        let result_ty: pliron::r#type::TypeHandle = {result_builder};\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty],\n            vec![member_mask, value, lane_or_delta],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 3 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_i32(ctx, op.get_operand(0).get_type(ctx))\n            || !{value_check}(ctx, op.get_operand(1).get_type(ctx))\n            || !is_i32(ctx, op.get_operand(2).get_type(ctx))\n            || !{value_check}(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly [member_mask, value, lane_or_delta] and one result",
                record.dialect.op_name
            ),
            format!(
                "{} requires i32 mask/lane and {value_label} value/result",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in warp_shuffles(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_dotprod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for generated packed integer dot products.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in dot_products(catalog) {
        let signedness = match record.rust.result.as_str() {
            "i32" => "Signed",
            "u32" => "Unsigned",
            result => panic!("unsupported dot-product result {result}"),
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Lowers to `{}`.",
            dot_product_ptx(record).replace("$0, $1, $2, $3", "%d, %a, %b, %c")
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, a: Value, b: Value, c: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, 32, Signedness::{signedness});\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![a, b, c],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 3 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !(0..3).all(|index| is_i32(ctx, op.get_operand(index).get_type(ctx)))\n            || !is_i32(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly three operands and one result",
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
    for record in dot_products(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

pub(in crate::render) fn render_dialect_redux(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    let has_f32 = redux(catalog).any(|record| record.dialect.results == ["f32"]);
    if has_f32 {
        output.push_str(
            "//! Structural NVVM operations for the closed generated `redux.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{FP32Type, IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\nfn is_f32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()\n}\n\n",
        );
    } else {
        output.push_str(
            "//! Structural NVVM operations for the closed generated `redux.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
        );
    }
    for record in redux(catalog) {
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// Dialect operands are `[member_mask, value]`; generated lowering adapts\n/// them to LLVM's `(value, member_mask)` signature.\n",
        );
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        let (result_type, value_predicate, type_error) = match record.rust.result.as_str() {
            "u32" => (
                "IntegerType::get(ctx, 32, Signedness::Unsigned)",
                "is_i32",
                format!(
                    "{} member mask, value, and result must be 32-bit integers",
                    record.dialect.op_name
                ),
            ),
            "i32" => (
                "IntegerType::get(ctx, 32, Signedness::Signed)",
                "is_i32",
                format!(
                    "{} member mask, value, and result must be 32-bit integers",
                    record.dialect.op_name
                ),
            ),
            "f32" => (
                "FP32Type::get(ctx)",
                "is_f32",
                format!(
                    "{} member mask must be a 32-bit integer and value and result must be f32",
                    record.dialect.op_name
                ),
            ),
            result => panic!("unsupported redux result {result}"),
        };
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, member_mask: Value, value: Value) -> Ptr<Operation> {{\n        let result_ty = {result_type};\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![member_mask, value],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_i32(ctx, op.get_operand(0).get_type(ctx))\n            || !{value_predicate}(ctx, op.get_operand(1).get_type(ctx))\n            || !{value_predicate}(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly two operands [member_mask, value] and one result",
                record.dialect.op_name
            ),
            type_error,
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in redux(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}
