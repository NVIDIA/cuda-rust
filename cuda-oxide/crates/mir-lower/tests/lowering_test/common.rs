/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

/// Helper: fresh context with all dialects registered.
pub(super) fn make_test_ctx() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);
    ctx
}

/// Helper: build a module + MirFuncOp("kernel_func") with given arg types,
/// returning the module ptr and entry block.
pub(super) fn build_test_kernel(
    ctx: &mut Context,
    arg_tys: Vec<pliron::r#type::TypeHandle>,
) -> (
    pliron::context::Ptr<Operation>,
    pliron::context::Ptr<pliron::basic_block::BasicBlock>,
) {
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::types::FunctionType;

    let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_ty = FunctionType::get(ctx, arg_tys.clone(), vec![]);
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

/// Helper: append a mir.return (void) to a block.
pub(super) fn append_return(
    ctx: &mut Context,
    block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
) {
    let ret = Operation::new(
        ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(block, ctx);
}

pub(super) fn lowered_kernel_body(
    ctx: &Context,
    module_ptr: pliron::context::Ptr<Operation>,
) -> Vec<pliron::context::Ptr<Operation>> {
    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
    for op in module_block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        return func_region
            .deref(ctx)
            .iter(ctx)
            .flat_map(|block| block.deref(ctx).iter(ctx))
            .collect();
    }
    panic!("lowered kernel function not found")
}
