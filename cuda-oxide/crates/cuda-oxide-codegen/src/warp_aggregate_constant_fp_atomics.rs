/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Opt-in aggregation of discarded device-scope relaxed f16/f32 atomic adds with
//! finite compile-time increments.
//!
//! This deliberately changes both floating-point accumulation and the atomic
//! modification sequence: callers must opt in through
//! `CUDA_OXIDE_MIR_PASSES=warp-aggregate-constant-fp-atomics` and use it only
//! for accumulators whose intermediate values are not observed for
//! synchronization, control flow, or per-update accounting.
//!
//! This is a workload-specific pass: the added warp collectives can regress
//! low-contention updates, so it is deliberately never enabled by default.

use dialect_mir::{
    attributes::MirCastKindAttr,
    ops::{
        arithmetic::{MirBitAndOp, MirMulOp},
        call::MirCallOp,
        cast::MirCastOp,
        comparison::MirEqOp,
        constants::{MirConstantOp, MirFloatConstantOp},
        control_flow::{MirCondBranchOp, MirGotoOp},
        function::MirFuncOp,
    },
    rust_intrinsics::CALLEE_CTPOP,
};
use dialect_nvvm::ops::{
    ActiveMaskOp, MatchAnySyncI64Op, NvvmAtomicOpInterface, ReadPtxSregLanemaskLtOp,
    atomic::{AtomicOrdering, AtomicRmwKind, AtomicScope, NvvmAtomicRmwOp},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr},
        op_interfaces::OperandSegmentInterface,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    pass::{AnalysisManager, Pass, PassResult},
    r#type::{Typed, TypedHandle},
    utils::apint::APInt,
    value::Value,
};
use std::num::NonZero;

/// Largest f16 magnitude whose product with the maximum 32-lane group is
/// still finite.
const MAX_F16_MAGNITUDE_FOR_WARP_SUM: u16 = 0x67ff;

#[derive(Default)]
pub struct WarpAggregateConstantFpAtomicsPass;

pub fn build_pass() -> Box<dyn Pass> {
    Box::new(WarpAggregateConstantFpAtomicsPass)
}

impl Pass for WarpAggregateConstantFpAtomicsPass {
    fn name(&self) -> &str {
        "warp-aggregate-constant-fp-atomics"
    }

    fn run(
        &mut self,
        module: Ptr<Operation>,
        ctx: &mut Context,
        _: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let mut changed = false;
        let region = module.deref(ctx).get_region(0);
        let funcs: Vec<_> = region
            .deref(ctx)
            .iter(ctx)
            .flat_map(|b| b.deref(ctx).iter(ctx).collect::<Vec<_>>())
            .filter(|&o| Operation::get_op::<MirFuncOp>(o, ctx).is_some())
            .collect();
        let mut blocks = funcs
            .into_iter()
            .flat_map(|func| {
                let body = func.deref(ctx).get_region(0);
                body.deref(ctx).iter(ctx).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        while let Some(block) = blocks.pop() {
            let atomic = block.deref(ctx).iter(ctx).find(|&op| eligible(ctx, op));
            if let Some(atomic) = atomic {
                // `rewrite` moves the suffix to a fresh continuation. Requeue
                // only that block, so each original operation is considered at
                // most once instead of rescanning the entire module per match.
                blocks.push(rewrite(ctx, block, atomic)?);
                changed = true;
            }
        }
        let mut result = PassResult::default();
        if changed {
            result.ir_changed = pliron::irbuild::IRStatus::Changed;
        }
        Ok(result)
    }
}

fn eligible(ctx: &Context, op: Ptr<Operation>) -> bool {
    let Some(a) = Operation::get_op::<NvvmAtomicRmwOp>(op, ctx) else {
        return false;
    };
    if a.rmw_kind(ctx) != AtomicRmwKind::FAdd
        || a.ordering(ctx) != AtomicOrdering::Relaxed
        || a.scope(ctx) != AtomicScope::Device
    {
        return false;
    }
    if !op.deref(ctx).get_result(0).uses(ctx).is_empty() {
        return false;
    }
    let Some(c) = a
        .val_opd(ctx)
        .defining_op()
        .and_then(|o| Operation::get_op::<MirFloatConstantOp>(o, ctx))
    else {
        return false;
    };
    c.get_attr_float_value_f16(ctx)
        .is_some_and(|v| f16_constant_is_eligible(v.to_bits()))
        || c.get_attr_float_value(ctx)
            .is_some_and(|v| f32_constant_is_eligible(f32::from(v.clone())))
}

fn f16_constant_is_eligible(bits: u16) -> bool {
    let magnitude = bits & 0x7fff;
    magnitude != 0 && magnitude <= MAX_F16_MAGNITUDE_FOR_WARP_SUM
}

fn f32_constant_is_eligible(value: f32) -> bool {
    value.is_finite() && value != 0.0 && value.abs() <= f32::MAX / 32.0
}

fn constant_before(
    ctx: &mut Context,
    ty: pliron::r#type::TypeHandle,
    n: u64,
    before: Ptr<Operation>,
) -> Value {
    let width = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .expect("integer")
        .width();
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    );
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).expect("integer");
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(
            typed,
            APInt::from_u64(n, NonZero::new(width as usize).unwrap()),
        ),
    );
    op.insert_before(ctx, before);
    op.deref(ctx).get_result(0)
}

fn stamp_generated(ctx: &mut Context, op: Ptr<Operation>, name: &str) {
    let marker = crate::__private::generated_intrinsic_marker_by_op_name(name)
        .expect("generated warp operation has an ABI marker");
    op.deref_mut(ctx).attributes.set(
        Identifier::try_from(crate::__private::GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
        StringAttr::new(marker.to_owned()),
    );
}

fn rewrite(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    atomic: Ptr<Operation>,
) -> pliron::result::Result<Ptr<BasicBlock>> {
    let i1t = IntegerType::get(ctx, 1, Signedness::Unsigned).into();
    let i32t = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let i64t = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let a = NvvmAtomicRmwOp::new(atomic);
    let ptr = a.ptr_opd(ctx);
    let increment = a.val_opd(ctx);
    let float_ty = increment.get_type(ctx);
    let is_one = increment
        .defining_op()
        .and_then(|o| Operation::get_op::<MirFloatConstantOp>(o, ctx))
        .is_some_and(|c| {
            c.get_attr_float_value_f16(ctx)
                .is_some_and(|v| v.to_bits() == 0x3c00)
                || c.get_attr_float_value(ctx)
                    .is_some_and(|v| f32::from(v.clone()).to_bits() == 1.0f32.to_bits())
        });

    let cast = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![i64t],
        vec![ptr],
        vec![],
        0,
    );
    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PointerExposeAddress);
    cast.insert_before(ctx, atomic);
    let active = ActiveMaskOp::build(ctx);
    stamp_generated(ctx, active, "nvvm.activemask");
    active.insert_before(ctx, atomic);
    let active_value = active.deref(ctx).get_result(0);
    let key = cast.deref(ctx).get_result(0);
    let same = MatchAnySyncI64Op::build(ctx, active_value, key);
    stamp_generated(ctx, same, "nvvm.match_any_sync_i64");
    same.insert_before(ctx, atomic);
    let lower = Operation::new(
        ctx,
        ReadPtxSregLanemaskLtOp::get_concrete_op_info(),
        vec![i32t],
        vec![],
        vec![],
        0,
    );
    stamp_generated(ctx, lower, "nvvm.read_ptx_sreg_lanemask_lt");
    lower.insert_before(ctx, atomic);
    let same_value = same.deref(ctx).get_result(0);
    let lower_value = lower.deref(ctx).get_result(0);
    let and = Operation::new(
        ctx,
        MirBitAndOp::get_concrete_op_info(),
        vec![i32t],
        vec![same_value, lower_value],
        vec![],
        0,
    );
    and.insert_before(ctx, atomic);
    let zero = constant_before(ctx, i32t, 0, atomic);
    let and_value = and.deref(ctx).get_result(0);
    let eq = Operation::new(
        ctx,
        MirEqOp::get_concrete_op_info(),
        vec![i1t],
        vec![and_value, zero],
        vec![],
        0,
    );
    eq.insert_before(ctx, atomic);
    let pop = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![i32t],
        vec![same_value],
        vec![],
        0,
    );
    MirCallOp::new(pop).set_attr_callee(ctx, StringAttr::new(CALLEE_CTPOP.into()));
    pop.insert_before(ctx, atomic);
    let pop_value = pop.deref(ctx).get_result(0);
    let as_float = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![float_ty],
        vec![pop_value],
        vec![],
        0,
    );
    MirCastOp::new(as_float).set_attr_cast_kind(ctx, MirCastKindAttr::IntToFloat);
    as_float.insert_before(ctx, atomic);
    let as_float_value = as_float.deref(ctx).get_result(0);
    let aggregated_increment = if is_one {
        as_float_value
    } else {
        let mul = Operation::new(
            ctx,
            MirMulOp::get_concrete_op_info(),
            vec![float_ty],
            vec![increment, as_float_value],
            vec![],
            0,
        );
        mul.insert_before(ctx, atomic);
        mul.deref(ctx).get_result(0)
    };
    Operation::remove_operand(atomic, ctx, 1);
    Operation::push_operand(atomic, ctx, aggregated_increment);

    let leader = BasicBlock::new(ctx, None, vec![]);
    leader.insert_after(ctx, block);
    let cont = BasicBlock::new(ctx, None, vec![]);
    cont.insert_after(ctx, leader);
    let suffix: Vec<_> = block
        .deref(ctx)
        .iter(ctx)
        .skip_while(|&op| op != atomic)
        .collect();
    for op in suffix {
        op.unlink(ctx);
        if op == atomic {
            op.insert_at_back(leader, ctx);
        } else {
            op.insert_at_back(cont, ctx);
        }
    }
    let goto = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![cont],
        0,
    );
    goto.insert_at_back(leader, ctx);
    let (operands, segs) = MirCondBranchOp::compute_segment_sizes(vec![
        vec![eq.deref(ctx).get_result(0)],
        vec![],
        vec![],
    ]);
    let branch = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![leader, cont],
        0,
    );
    MirCondBranchOp::new(branch).set_operand_segment_sizes(ctx, segs);
    branch.insert_at_back(block, ctx);
    Ok(cont)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{ops::control_flow::MirReturnOp, types::MirPtrType};
    use pliron::builtin::{
        attributes::{FPSingleAttr, TypeAttr},
        op_interfaces::SymbolOpInterface,
        ops::ModuleOp,
        types::{FP32Type, FunctionType},
    };

    fn module_block(ctx: &mut Context, module: &ModuleOp) -> Ptr<BasicBlock> {
        let region = module.get_operation().deref(ctx).get_region(0);
        let existing = { region.deref(ctx).iter(ctx).next() };
        existing.unwrap_or_else(|| {
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(region, ctx);
            block
        })
    }

    fn append_f32_atomic_kernel(
        ctx: &mut Context,
        module: &ModuleOp,
        scope: AtomicScope,
        ordering: AtomicOrdering,
        increments: &[f32],
        result_is_used: bool,
    ) {
        let module_block = module_block(ctx, module);
        let f32_ty = FP32Type::get(ctx);
        let ptr_ty = MirPtrType::get_generic(ctx, f32_ty.into(), true);
        let results = result_is_used.then(|| f32_ty.into()).into_iter().collect();
        let function_type = FunctionType::get(ctx, vec![ptr_ty.into()], results);
        let op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, op, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, "atomics".try_into().unwrap());
        let entry = BasicBlock::new(ctx, None, vec![ptr_ty.into()]);
        let function_region = function.get_operation().deref(ctx).get_region(0);
        entry.insert_at_back(function_region, ctx);
        let ptr = entry.deref(ctx).get_argument(0);

        let mut final_result = None;
        for &increment in increments {
            let constant = Operation::new(
                ctx,
                MirFloatConstantOp::get_concrete_op_info(),
                vec![f32_ty.into()],
                vec![],
                vec![],
                0,
            );
            MirFloatConstantOp::new(constant)
                .set_attr_float_value(ctx, FPSingleAttr::from(increment));
            constant.insert_at_back(entry, ctx);
            let value = constant.deref(ctx).get_result(0);
            let atomic = NvvmAtomicRmwOp::build(
                ctx,
                ptr,
                value,
                f32_ty.into(),
                AtomicRmwKind::FAdd,
                ordering.clone(),
                scope.clone(),
            );
            atomic.get_operation().insert_at_back(entry, ctx);
            final_result = Some(atomic.get_operation().deref(ctx).get_result(0));
        }
        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            result_is_used
                .then(|| final_result.unwrap())
                .into_iter()
                .collect(),
            vec![],
            0,
        )
        .insert_at_back(entry, ctx);
        function.get_operation().insert_at_back(module_block, ctx);
    }

    fn atomic_kernel_ops(ctx: &Context, module: &ModuleOp) -> Vec<Ptr<Operation>> {
        let module_region = module.get_operation().deref(ctx).get_region(0);
        module_region
            .deref(ctx)
            .iter(ctx)
            .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
            .filter_map(|op| Operation::get_op::<MirFuncOp>(op, ctx))
            .flat_map(|function| {
                let body = function.get_operation().deref(ctx).get_region(0);
                body.deref(ctx)
                    .iter(ctx)
                    .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn run_pass(module: &ModuleOp, ctx: &mut Context) {
        WarpAggregateConstantFpAtomicsPass
            .run(module.get_operation(), ctx, &mut AnalysisManager::default())
            .unwrap();
    }

    #[test]
    fn f16_constant_increment_requires_a_finite_32_lane_sum() {
        assert!(f16_constant_is_eligible(0x3c00)); // +1
        assert!(f16_constant_is_eligible(0xbc00)); // -1
        assert!(f16_constant_is_eligible(MAX_F16_MAGNITUDE_FOR_WARP_SUM));
        assert!(!f16_constant_is_eligible(0));
        assert!(!f16_constant_is_eligible(0x8000)); // -0
        assert!(!f16_constant_is_eligible(0x6800)); // overflows after x32
        assert!(!f16_constant_is_eligible(0x7c00)); // +infinity
        assert!(!f16_constant_is_eligible(0x7e00)); // NaN
    }

    #[test]
    fn f32_constant_increment_requires_a_finite_32_lane_sum() {
        assert!(f32_constant_is_eligible(1.0));
        assert!(f32_constant_is_eligible(-2.0));
        assert!(f32_constant_is_eligible(f32::MAX / 32.0));
        assert!(!f32_constant_is_eligible(0.0));
        assert!(!f32_constant_is_eligible(-0.0));
        assert!(!f32_constant_is_eligible(f32::MAX));
        assert!(!f32_constant_is_eligible(f32::INFINITY));
        assert!(!f32_constant_is_eligible(f32::NAN));
    }

    #[test]
    fn transforms_each_eligible_atomic_and_scales_non_unit_constants() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        append_f32_atomic_kernel(
            &mut ctx,
            &module,
            AtomicScope::Device,
            AtomicOrdering::Relaxed,
            &[1.0, 2.0],
            false,
        );

        run_pass(&module, &mut ctx);
        let ops = atomic_kernel_ops(&ctx, &module);
        assert_eq!(
            ops.iter()
                .filter(|&&op| Operation::get_op::<MatchAnySyncI64Op>(op, &ctx).is_some())
                .count(),
            2
        );
        assert_eq!(
            ops.iter()
                .filter(|&&op| Operation::get_op::<MirMulOp>(op, &ctx).is_some())
                .count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|&&op| Operation::get_op::<NvvmAtomicRmwOp>(op, &ctx).is_some())
                .count(),
            2
        );
        let requirements = crate::generated::collect_generated_intrinsic_requirements(
            &ctx,
            module.get_operation(),
            crate::generated::GeneratedMarkerPolicy::Required,
        )
        .unwrap();
        let match_any = requirements
            .targets
            .iter()
            .find(|target| target.id == "match_any_i64_sync")
            .unwrap();
        assert_eq!(
            match_any.requirement.minimum_ptx,
            crate::generated_intrinsic_targets::GeneratedPtxVersion::from_encoded(60)
        );
        assert_eq!(
            match_any.requirement.hardware,
            crate::generated_intrinsic_targets::GeneratedHardwareTarget::AnyOf(&[
                crate::generated_intrinsic_targets::GeneratedHardwareAlternative::MinimumSm(70),
            ])
        );
    }

    #[test]
    fn leaves_non_device_or_non_relaxed_atomics_unchanged() {
        for (scope, ordering) in [
            (AtomicScope::System, AtomicOrdering::Relaxed),
            (AtomicScope::Device, AtomicOrdering::Acquire),
        ] {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            dialect_nvvm::register(&mut ctx);
            let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
            append_f32_atomic_kernel(&mut ctx, &module, scope, ordering, &[1.0], false);

            run_pass(&module, &mut ctx);
            let ops = atomic_kernel_ops(&ctx, &module);
            assert!(
                ops.iter()
                    .all(|&op| Operation::get_op::<MatchAnySyncI64Op>(op, &ctx).is_none())
            );
            assert_eq!(
                ops.iter()
                    .filter(|&&op| Operation::get_op::<NvvmAtomicRmwOp>(op, &ctx).is_some())
                    .count(),
                1
            );
        }
    }

    #[test]
    fn leaves_atomic_results_used_by_the_program_unchanged() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        append_f32_atomic_kernel(
            &mut ctx,
            &module,
            AtomicScope::Device,
            AtomicOrdering::Relaxed,
            &[1.0],
            true,
        );

        run_pass(&module, &mut ctx);
        let ops = atomic_kernel_ops(&ctx, &module);
        assert!(
            ops.iter()
                .all(|&op| Operation::get_op::<MatchAnySyncI64Op>(op, &ctx).is_none())
        );
    }
}
