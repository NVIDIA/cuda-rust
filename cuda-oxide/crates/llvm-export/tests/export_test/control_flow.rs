/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{NvvmExportConfig, NvvmIrDialect, export_module_to_string_with_config},
    ops::{BrOp, CondBrOp, ConstantOp, FuncOp, ReturnOp, UndefOp},
    types::{FuncType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    utils::apint::APInt,
};
use std::num::NonZero;

use crate::common::module_top_block;

#[test]
fn exporter_rejects_extra_predecessor_values_before_emitting_phis() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_branch_arity".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let func = FuncOp::new(&mut ctx, "invalid_branch".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let destination = BasicBlock::new(&mut ctx, None, vec![]);
    destination.insert_at_back(region, &ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, Box::new(one_attr));
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);
    BrOp::new(&mut ctx, destination, vec![one_value])
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(destination, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("extra predecessor values must be rejected");
    assert!(
        error.contains("supplies 1 values") && error.contains("expects 0 block arguments"),
        "{error}"
    );
}

#[test]
fn exporter_rejects_distinct_values_on_duplicate_conditional_edges() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "duplicate_conditional_edge".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let func_ty = FuncType::get(
        &ctx,
        i32_ty.into(),
        vec![i1_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "duplicate_edge".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let condition = entry.deref(&ctx).get_argument(0);
    let if_true = entry.deref(&ctx).get_argument(1);
    let if_false = entry.deref(&ctx).get_argument(2);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let destination = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    destination.insert_at_back(region, &ctx);
    CondBrOp::new(
        &mut ctx,
        condition,
        destination,
        vec![if_true],
        destination,
        vec![if_false],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    let result = destination.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(destination, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("pliron permits same-destination conditional edges with distinct values");
    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("LLVM PHIs cannot distinguish duplicate predecessor edges");
    assert!(
        error.contains("both edges with different forwarded values"),
        "{error}"
    );
}

#[test]
fn exporter_deduplicates_identical_values_on_duplicate_conditional_edges() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "identical_conditional_edge".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let func_ty = FuncType::get(
        &ctx,
        i32_ty.into(),
        vec![i1_ty.into(), i32_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "identical_edge".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let condition = entry.deref(&ctx).get_argument(0);
    let value = entry.deref(&ctx).get_argument(1);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let destination = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    destination.insert_at_back(region, &ctx);
    CondBrOp::new(
        &mut ctx,
        condition,
        destination,
        vec![value],
        destination,
        vec![value],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    let result = destination.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(destination, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("identical duplicate-edge values can use one PHI predecessor");
    let phi = ir
        .lines()
        .find(|line| line.contains(" = phi i32 "))
        .expect("destination block must contain a PHI");
    assert_eq!(phi.matches("%entry").count(), 1, "{phi}");
}

#[test]
fn phi_can_reference_undef_from_a_later_block() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "later_undef_phi".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let func_ty = FuncType::get(
        &ctx,
        i32_ty.into(),
        vec![i1_ty.into(), i32_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "choose_undef".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let condition = entry.deref(&ctx).get_argument(0);
    let fallback = entry.deref(&ctx).get_argument(1);
    let region = func.get_operation().deref(&ctx).get_region(0);

    // The join precedes both predecessors in print order, so its PHI depends
    // on the exporter's whole-function value-name pre-pass.
    let join = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    join.insert_at_back(region, &ctx);
    let undef_block = BasicBlock::new(&mut ctx, None, vec![]);
    undef_block.insert_at_back(region, &ctx);
    let value_block = BasicBlock::new(&mut ctx, None, vec![]);
    value_block.insert_at_back(region, &ctx);

    CondBrOp::new(
        &mut ctx,
        condition,
        undef_block,
        vec![],
        value_block,
        vec![],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    let undef = UndefOp::new(&mut ctx, i32_ty.into());
    let undef_value = undef.get_operation().deref(&ctx).get_result(0);
    undef.get_operation().insert_at_back(undef_block, &ctx);
    BrOp::new(&mut ctx, join, vec![undef_value])
        .get_operation()
        .insert_at_back(undef_block, &ctx);
    BrOp::new(&mut ctx, join, vec![fallback])
        .get_operation()
        .insert_at_back(value_block, &ctx);

    let result = join.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(join, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("later-block undef must be available while exporting an earlier PHI");
    assert!(
        ir.lines()
            .any(|line| line.contains(" = phi i32 ") && line.contains("[ undef,")),
        "{ir}"
    );
}
