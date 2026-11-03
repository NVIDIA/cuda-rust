/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared scaffolding for the `convert::ops` lowering tests.
//!
//! The `convert_*` functions take a live `DialectConversionRewriter` owned by
//! pliron's conversion driver and aren't constructible standalone, so every
//! test builds a minimal MIR module, runs the full `lower_mir_to_llvm` pass,
//! and inspects the lowered LLVM dialect ops. These helpers cover the parts
//! that are identical across op categories (module/kernel construction and
//! op lookup); category-specific builders live next to their tests.
//!
//! Not every helper is used by every test module, so dead-code is allowed.
#![allow(dead_code)]

use dialect_mir::ops as mir;
use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::TypeAttr;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::FunctionType;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;

/// Fresh context with every dialect this crate's lowering needs registered.
pub(crate) fn make_ctx() -> Context {
    let mut ctx = Context::new();
    // The LLVM dialect (llvm-export, re-exporting pliron-llvm) auto-registers on
    // Context creation; only the local dialects need an explicit register call.
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    crate::register(&mut ctx);
    ctx
}

/// Build a module with one `MirFuncOp` `kernel_func` taking `arg_tys` and
/// returning `ret_tys`. Returns the module pointer and the function's entry
/// block; the caller appends ops (including the terminator) before lowering.
pub(crate) fn build_kernel(
    ctx: &mut Context,
    arg_tys: Vec<TypeHandle>,
    ret_tys: Vec<TypeHandle>,
) -> (Ptr<Operation>, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_ty = FunctionType::get(ctx, arg_tys.clone(), ret_tys);
    let func_op_ptr = Operation::new(
        ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(ctx, "kernel_func".try_into().unwrap());

    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, arg_tys);
    entry.insert_at_back(region, ctx);

    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, ctx);

    (module_ptr, entry)
}

/// Append a basic block taking `arg_tys` to `kernel_func`'s region.
pub(crate) fn append_block(
    ctx: &mut Context,
    entry: Ptr<BasicBlock>,
    arg_tys: Vec<TypeHandle>,
) -> Ptr<BasicBlock> {
    let region = entry.deref(ctx).get_parent_region().unwrap();
    let block = BasicBlock::new(ctx, None, arg_tys);
    block.insert_at_back(region, ctx);
    block
}

/// Append a `mir.return` carrying `vals` (empty for a void return) to `block`.
pub(crate) fn append_mir_return(ctx: &mut Context, block: Ptr<BasicBlock>, vals: Vec<Value>) {
    let op = Operation::new(
        ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vals,
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
}

/// All blocks of `kernel_func` in the lowered module.
///
/// `convert_func` builds a fresh `llvm.func` with a prologue entry block (which
/// reconstructs aggregate args and branches to the inlined MIR region), so the
/// lowered function generally has more than one block and a converted op can
/// land in any of them. Tests iterate across all of them.
pub(crate) fn kernel_blocks(ctx: &Context, module_ptr: Ptr<Operation>) -> Vec<Ptr<BasicBlock>> {
    let module_op = module_ptr.deref(ctx);
    let region = module_op.get_region(0);
    let module_block = region.deref(ctx).iter(ctx).next().unwrap();
    for op in module_block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        return func_region.deref(ctx).iter(ctx).collect();
    }
    panic!("kernel_func not found in lowered module");
}

/// The module's top-level block (where the lowered `llvm.func`/globals live).
pub(crate) fn module_top_block(ctx: &Context, module_ptr: Ptr<Operation>) -> Ptr<BasicBlock> {
    module_ptr
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .unwrap()
}

/// Number of `T` ops across `blocks`.
pub(crate) fn count_ops<T: Op>(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> usize {
    blocks
        .iter()
        .flat_map(|b| b.deref(ctx).iter(ctx))
        .filter(|op| Operation::get_op::<T>(*op, ctx).is_some())
        .count()
}

/// First `T` op across `blocks`, if any.
pub(crate) fn find_first<T: Op>(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> Option<T> {
    blocks
        .iter()
        .flat_map(|b| b.deref(ctx).iter(ctx))
        .find_map(|op| Operation::get_op::<T>(op, ctx))
}

/// All `T` ops across `blocks`.
pub(crate) fn find_all<T: Op>(ctx: &Context, blocks: &[Ptr<BasicBlock>]) -> Vec<T> {
    blocks
        .iter()
        .flat_map(|b| b.deref(ctx).iter(ctx))
        .filter_map(|op| Operation::get_op::<T>(op, ctx))
        .collect()
}
