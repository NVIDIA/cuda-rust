/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Terminator translation: MIR terminators → `dialect-mir` control flow.
//!
//! This module translates MIR terminators (return, goto, call, switch, etc.)
//! into `dialect-mir` operations. GPU intrinsics from `cuda_device` are
//! expanded inline to `dialect-nvvm` operations.
//!
//! All non-entry blocks are argument-less: cross-block data flow travels
//! through the per-local alloca slots owned by [`ValueMap`], so every branch
//! terminator emitted here is zero-operand. For example, a MIR `goto` whose
//! successor reads a local set by the predecessor translates as:
//!
//! ```text
//! // Rust MIR
//! bb0: { _1 = 42_i32; goto -> bb1 }
//! bb1: { _0 = _1;     return }
//!
//! // dialect-mir (pre-mem2reg)
//! ^bb0:
//!   %s1 = mir.alloca          : !mir.ptr<i32>
//!   %c  = mir.constant 42_i32 : i32
//!   mir.store %c, %s1
//!   mir.goto ^bb1                    // zero-operand; _1 flows via %s1
//! ^bb1:                              // no block arguments
//!   %r = mir.load %s1 : i32
//!   mir.return %r : i32
//! ```
//!
//! The `mem2reg` pass (run later in [`crate::pipeline`]) folds these slot
//! round-trips into SSA, so the above collapses to a direct `mir.return %c`.
//!
//! # Function Name Resolution
//!
//! `extract_func_info` uses `CrateDef::name()` which returns fully qualified
//! names (FQDNs, e.g. `helper_fn::cuda_oxide_device_<hash>_vecadd`). This FQDN is
//! used as both `pattern_name` (for intrinsic matching against paths like
//! `cuda_device::thread::threadIdx_x`) and `call_name` (for non-generic calls).
//! The collector produces matching FQDNs, and the lowering layer converts
//! `::` to `__` on both sides.
//!
//! # Module Structure
//!
//! - [`helpers`]: Common utilities (`emit_goto`, `emit_function_call`)
//! - [`intrinsics`]: GPU intrinsic handlers organized by category:
//!   - `indexing`: Thread/block IDs, `index_1d`, `index_2d::<S>`, `index_2d_runtime`
//!   - `sync`: Barriers, mbarrier operations
//!   - `warp`: Shuffle, vote primitives
//!   - `wgmma`: Hopper matrix operations
//!   - `tcgen05`: Blackwell tensor core operations
//!   - `tma`: Tensor memory access
//!   - `memory`: SharedArray indexing, stmatrix

pub mod drop_glue;
pub mod helpers;
pub mod intrinsics;

use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts::{
    eval_usize_const, is_cuda_device_fn, is_std_precondition_check, self_ty_is_shared_array,
};
use crate::translator::location::span_to_location;
use crate::translator::rvalue;
use crate::translator::values::{ValueMap, maybe_ptr_coerce};
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{
    MirAssertOp, MirCondBranchOp, MirConstantOp, MirEqOp, MirGotoOp, MirNotOp, MirReturnOp,
    MirUnrollHintOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::{input_err, input_error};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::ConstantKind;
/// Translates a MIR terminator to Pliron IR control flow operation(s).
///
/// Handles all MIR terminator kinds:
/// - `Return`: Function return
/// - `Goto`: Unconditional branch
/// - `SwitchInt`: Multi-way branch (for enums, match)
/// - `Assert`: Runtime assertions with panic on failure
/// - `Call`: Function/intrinsic calls
/// - `Drop`: Destructor calls (no-op for Copy types)
/// - `Unreachable`: Marks unreachable code
///
/// # GPU Intrinsics
///
/// Calls to `cuda_device` functions are expanded inline to `dialect-nvvm` operations.
/// This includes thread indexing, synchronization, warp primitives, and
/// tensor core operations.
#[allow(clippy::too_many_arguments)]
pub fn translate_terminator(
    ctx: &mut Context,
    body: &mir::Body,
    term: &mir::Terminator,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    rustc_mono_successors: &[usize],
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    let loc = span_to_location(ctx, term.source_info.span);

    match &term.kind {
        mir::TerminatorKind::Return => {
            translate_return(ctx, body, value_map, block_ptr, prev_op, loc)
        }

        mir::TerminatorKind::Goto { target } => {
            translate_goto(ctx, *target, block_ptr, prev_op, block_map, loc)
        }

        mir::TerminatorKind::Assert {
            cond,
            expected,
            msg,
            target,
            unwind,
        } => translate_assert(
            ctx, body, cond, *expected, msg, *target, unwind, block_ptr, prev_op, value_map,
            block_map, loc,
        ),

        mir::TerminatorKind::Call {
            func,
            args,
            destination,
            target,
            unwind,
        } => translate_call(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            unwind,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            legaliser,
        ),

        mir::TerminatorKind::SwitchInt { discr, targets } => match rustc_mono_successors {
            // rustc evaluated this switch for the concrete instance. Emit
            // that exact edge instead of evaluating the converted public MIR
            // a second time.
            [target] => translate_goto(ctx, *target, block_ptr, prev_op, block_map, loc),
            [] => input_err!(
                loc,
                TranslationErr::invalid_op(
                    "rustc supplied no successor for a reachable SwitchInt".to_string()
                )
            ),
            _ => translate_switch(
                ctx, body, discr, targets, block_ptr, prev_op, value_map, block_map, loc,
            ),
        },

        mir::TerminatorKind::Drop {
            place,
            target,
            unwind,
        } => translate_drop(
            ctx, body, place, *target, unwind, block_ptr, prev_op, value_map, block_map, legaliser,
            loc,
        ),

        mir::TerminatorKind::Unreachable => {
            // Create an unreachable operation
            let op = Operation::new(
                ctx,
                dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);
            if let Some(prev) = prev_op {
                op.insert_after(ctx, prev);
            } else {
                op.insert_at_front(block_ptr, ctx);
            }
            Ok(op)
        }

        _ => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Terminator kind {:?} not yet implemented",
                term.kind
            ))
        ),
    }
}

// ============================================================================
// Core Terminator Handlers
// ============================================================================

/// Translates a MIR `Return` terminator to a `mir.return` operation.
///
/// Handles the return value (`_0`) from the function:
/// - For non-unit returns: passes the return value as an operand
/// - For unit returns (empty tuple): emits return with no operands
///
/// # MIR Semantics
///
/// In MIR, local 0 (`_0`) holds the return value. The return terminator
/// transfers control back to the caller with this value.
fn translate_return(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // MIR local `_0` holds the return value. In the alloca + load/store model
    // we emit a `mir.load` from its slot to materialise the SSA value, then
    // pass it as the `mir.return` operand. ZSTs (including `()` kernel
    // returns) have no slot, so we simply emit a bare `return`.
    let return_local = mir::Local::from(0usize);
    let return_decl = &body.locals()[return_local];
    let return_type = types::translate_type(ctx, &return_decl.ty)?;
    let is_unit_return = {
        use dialect_mir::types::MirTupleType;
        let return_type_obj = return_type.deref(ctx);
        if let Some(tuple_ty) = return_type_obj.downcast_ref::<MirTupleType>() {
            tuple_ty.get_types().is_empty()
        } else {
            false
        }
    };
    let loaded = value_map.load_local(ctx, return_local, block_ptr, prev_op);

    let (operands, terminator_prev_op) = match loaded {
        Some((load_op, val)) => {
            if is_unit_return {
                // Unit return: the load we just emitted is dead, but harmless;
                // leave it as prev_op so the return chains after it.
                (vec![], Some(load_op))
            } else {
                let (val, prev_op) =
                    maybe_ptr_coerce(ctx, val, return_type, block_ptr, Some(load_op));
                (vec![val], prev_op)
            }
        }
        None => {
            let return_ty = types::translate_type(ctx, &body.locals()[return_local].ty)?;
            let is_unit = return_ty
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirTupleType>()
                .is_some_and(|tuple| tuple.get_types().is_empty());
            if types::is_zst_type(ctx, return_ty) && !is_unit {
                // Non-unit ZST returns still need a MIR-level value so the
                // `mir.return` verifier agrees with the function signature.
                // LLVM lowering erases the value and emits a void return.
                let undef = dialect_mir::ops::MirUndefOp::new(ctx, return_ty).get_operation();
                undef.deref_mut(ctx).set_loc(loc.clone());
                if let Some(prev) = prev_op {
                    undef.insert_after(ctx, prev);
                } else {
                    undef.insert_at_front(block_ptr, ctx);
                }
                (vec![undef.deref(ctx).get_result(0)], Some(undef))
            } else {
                (vec![], prev_op)
            }
        }
    };

    let op = Operation::new(
        ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![], // No results
        operands,
        vec![], // No successors
        0,      // No regions
    );
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = terminator_prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok(op)
}

/// Translates a MIR `Goto` terminator to a zero-operand `mir.goto` operation.
///
/// Non-entry blocks carry no arguments; cross-block data flow travels through
/// per-local alloca slots instead.
#[allow(clippy::too_many_arguments)]
fn translate_goto(
    ctx: &mut Context,
    target: mir::BasicBlockIdx,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let target_idx: usize = target;
    let target_block = block_map[target_idx];

    let op = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target_block],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok(op)
}

/// Translates a MIR `Assert` terminator to `mir.assert` followed by `mir.goto`.
///
/// Asserts that a condition matches the expected value, trapping on failure.
/// On success, branches to the target block.
///
/// # GPU Constraints
///
/// The CUDA toolchain does not support unwinding today; we treat all
/// unwind edges as unreachable.
///
/// # Condition Handling
///
/// If `expected == false`, the condition is negated before the assert:
/// - `assert!(cond, expected=true)` → assert condition is true
/// - `assert!(cond, expected=false)` → assert condition is false (negated)
///
/// # Unchecked indexing
///
/// When the body's unchecked-indexing policy is set (per-kernel
/// `#[kernel(unchecked_indexing)]` marker or the whole-build
/// `CUDA_OXIDE_UNCHECKED_INDEXING=1` switch), asserts whose message is
/// `AssertMessage::BoundsCheck` are elided: control flows unconditionally to
/// the success target and the condition is never materialized. This is legal
/// because the success edge carries no block arguments. An out-of-bounds
/// index then is undefined behavior, exactly like `get_unchecked`. Every
/// other assert kind (overflow, division/remainder by zero, misaligned
/// pointer, ...) keeps its compare-and-trap lowering, and panics that arrive
/// as `core::panicking::*` calls (including range-indexing failures) are
/// unaffected.
#[allow(clippy::too_many_arguments)]
fn translate_assert(
    ctx: &mut Context,
    body: &mir::Body,
    cond: &mir::Operand,
    expected: bool,
    msg: &mir::AssertMessage,
    target: mir::BasicBlockIdx,
    unwind: &mir::UnwindAction,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // The CUDA toolchain doesn't support stack unwinding today (the hardware
    // could, but nvcc/ptxas don't wire it up). We ignore the unwind action
    // and only generate code for the success path. External crates (like core)
    // may carry unwind edges in their MIR; those are dead code on GPU -- if a
    // panic occurs, the GPU thread traps.
    let _ = unwind;

    // Opt-in bounds-check elision: replace the assert with an unconditional
    // goto to the success target without translating the (now dead)
    // condition. Only `BoundsCheck` messages qualify; see the doc comment.
    if value_map.unchecked_indexing() && matches!(msg, mir::AssertMessage::BoundsCheck { .. }) {
        return translate_goto(ctx, target, block_ptr, prev_op, block_map, loc);
    }

    // Translate the condition operand.
    //
    // MIR assert conditions are operands, not necessarily places. In
    // particular, rustc can retain boolean constants for guaranteed-failure
    // blocks and deliberate traps. The shared operand translator preserves the
    // old Copy/Move path by delegating to `translate_place`.
    //
    // Keep RuntimeChecks fail-closed here. `translate_operand` currently lowers
    // those session-dependent flags to `false` without access to rustc's
    // `Session`; accepting them as assert conditions would silently change the
    // requested assertion policy. They need separate session-policy plumbing.
    let (cond_value, mut last_inserted) = match cond {
        mir::Operand::Copy(_) | mir::Operand::Move(_) | mir::Operand::Constant(_) => {
            rvalue::translate_operand(ctx, body, cond, value_map, block_ptr, prev_op, loc.clone())?
        }
        mir::Operand::RuntimeChecks(_) => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "RuntimeChecks conditions in assert require session-policy lowering"
                        .to_string(),
                )
            );
        }
    };

    // Apply negation if expected == false
    let final_cond = if !expected {
        let bool_type = types::get_bool_type(ctx);
        let not_op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![bool_type.to_handle()],
            vec![cond_value],
            vec![],
            0,
        );
        not_op.deref_mut(ctx).set_loc(loc.clone());

        if let Some(prev) = last_inserted {
            not_op.insert_after(ctx, prev);
        } else if let Some(prev) = prev_op {
            not_op.insert_after(ctx, prev);
        } else {
            not_op.insert_at_front(block_ptr, ctx);
        }

        last_inserted = Some(not_op);
        not_op.deref(ctx).get_result(0)
    } else {
        cond_value
    };

    // Keep the potentially trapping check separate from its success branch.
    // Generic CFG merging may erase the goto, but must retain the assertion.
    let op = MirAssertOp::new(ctx, final_cond).get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_inserted {
        op.insert_after(ctx, prev);
    } else if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok(helpers::emit_goto(ctx, target, op, block_map, loc))
}

/// Build the comparison constant for one `SwitchInt` arm, typed as the
/// discriminant it is compared against.
///
/// Arm values arrive as `u128` bit patterns at the discriminant's width, so
/// the constant is built at that width and a scrutinee wider than 64 bits
/// keeps its whole value. Narrower widths truncate, which is what the
/// discriminant's own type already did to the value.
///
/// Enum tags cannot reach here above 64 bits. `translate_adt_type` rejects a
/// discriminant carrier wider than that when it builds the enum type, so the
/// widths above 64 that arrive here belong to integer scrutinees.
fn switch_arm_constant(
    ctx: &mut Context,
    val: u128,
    width: usize,
    signedness: Signedness,
) -> pliron::builtin::attributes::IntegerAttr {
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let width_nz = NonZeroUsize::new(width).expect("integer discriminants have a non-zero width");
    pliron::builtin::attributes::IntegerAttr::new(
        IntegerType::get(ctx, width as u32, signedness),
        APInt::from_u128(val, width_nz),
    )
}

/// Translates a MIR `SwitchInt` terminator to conditional branches.
///
/// Handles multi-way branches used for `match` expressions and enum dispatch:
///
/// # Boolean Switch (1 branch)
///
/// Uses `mir.cond_branch`:
/// - `switchInt(bool) → [0: bb_false, otherwise: bb_true]`
/// - Creates comparison or negation as needed
///
/// # Multi-way Switch (N branches)
///
/// Creates a chain of conditional branches:
/// ```text
/// current:     cmp0 = (discr == v0); cond_br cmp0, t0, intermediate_1
/// intermediate_1: cmp1 = (discr == v1); cond_br cmp1, t1, intermediate_2
/// ...
/// intermediate_N: cmpN = (discr == vN); cond_br cmpN, tN, otherwise
/// ```
#[allow(clippy::too_many_arguments)]
fn translate_switch(
    ctx: &mut Context,
    body: &mir::Body,
    discr: &mir::Operand,
    targets: &mir::SwitchTargets,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Translate discriminant
    let (discr_value, last_op) = match discr {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?
        }
        _ => {
            rvalue::translate_operand(ctx, body, discr, value_map, block_ptr, prev_op, loc.clone())?
        }
    };

    let branches: Vec<_> = targets.branches().collect();
    let otherwise_idx: usize = targets.otherwise();

    // For bool switches (2 branches), use MirCondBranchOp
    if branches.len() == 1 {
        let (val, target_bb) = branches[0];
        let target_idx: usize = target_bb;

        // For MirCondBranchOp, we need an i1 (boolean) condition.
        // If discr is already i1, we can use it directly (with appropriate target ordering).
        // Otherwise, we need to create a comparison: (discr == val).
        use pliron::r#type::Typed;
        let discr_ty = discr_value.get_type(ctx);
        let bool_ty = types::get_bool_type(ctx);

        let (condition, last_inserted_op) = if discr_ty == bool_ty.to_handle() {
            // discr is already i1
            // For boolean switch: val=0 means "if false", val=1 means "if true"
            // switchInt(bool) -> [0: bb_false, otherwise: bb_true]
            // Since val == 0 means "go to target when discr == 0", we need condition = !discr
            if val == 0 {
                // Create NOT operation: condition = !discr
                let not_op = Operation::new(
                    ctx,
                    MirNotOp::get_concrete_op_info(),
                    vec![bool_ty.to_handle()],
                    vec![discr_value],
                    vec![],
                    0,
                );
                not_op.deref_mut(ctx).set_loc(loc.clone());
                if let Some(prev) = last_op {
                    not_op.insert_after(ctx, prev);
                } else {
                    not_op.insert_at_front(block_ptr, ctx);
                }
                let cond = not_op.deref(ctx).get_result(0);
                (cond, Some(not_op))
            } else {
                // val == 1: condition is discr itself
                (discr_value, last_op)
            }
        } else {
            // discr is not i1 (e.g., u32 from lane_id(), or enum discriminant)
            // Create comparison: condition = (discr == val)
            let (width, signedness) =
                if let Some(int_ty) = discr_ty.deref(ctx).downcast_ref::<IntegerType>() {
                    (int_ty.width() as usize, int_ty.signedness())
                } else {
                    (64, Signedness::Unsigned) // Default to 64-bit unsigned if we can't determine
                };

            // Create constant for val with SAME type as discriminant.
            let int_attr = switch_arm_constant(ctx, val, width, signedness);

            let const_op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![discr_ty],
                vec![],
                vec![],
                0,
            );
            const_op.deref_mut(ctx).set_loc(loc.clone());
            let const_op_wrapped = MirConstantOp::new(const_op);
            const_op_wrapped.set_attr_value(ctx, int_attr);

            if let Some(prev) = last_op {
                const_op_wrapped.get_operation().insert_after(ctx, prev);
            } else {
                const_op_wrapped
                    .get_operation()
                    .insert_at_front(block_ptr, ctx);
            }
            let const_val = const_op_wrapped.get_operation().deref(ctx).get_result(0);

            // Create comparison operation: discr == val
            let eq_op = Operation::new(
                ctx,
                MirEqOp::get_concrete_op_info(),
                vec![bool_ty.to_handle()],
                vec![discr_value, const_val],
                vec![],
                0,
            );
            eq_op.deref_mut(ctx).set_loc(loc.clone());
            eq_op.insert_after(ctx, const_op_wrapped.get_operation());

            let cond = eq_op.deref(ctx).get_result(0);
            (cond, Some(eq_op))
        };

        // With condition = (discr == val) [or !discr for boolean val==0 case]:
        // true_target = target (go here when condition is true, i.e., discr == val)
        // false_target = otherwise (go here when condition is false)
        let true_idx = target_idx;
        let false_idx = otherwise_idx;

        let true_block = block_map[true_idx];
        let false_block = block_map[false_idx];

        // Alloca + load/store model: both branch successors are argument-less;
        // the cond_br carries only its boolean condition.
        let (flat_operands, segment_sizes) =
            MirCondBranchOp::compute_segment_sizes(vec![vec![condition], vec![], vec![]]);

        let op = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            flat_operands,
            vec![true_block, false_block],
            0,
        );
        Operation::get_op::<MirCondBranchOp>(op, ctx)
            .expect("MirCondBranchOp")
            .set_operand_segment_sizes(ctx, segment_sizes);
        op.deref_mut(ctx).set_loc(loc);

        // Use last_inserted_op (which accounts for NOT/EQ ops created above)
        if let Some(prev) = last_inserted_op {
            op.insert_after(ctx, prev);
        } else if let Some(prev) = last_op {
            op.insert_after(ctx, prev);
        } else {
            op.insert_at_front(block_ptr, ctx);
        }

        return Ok(op);
    }

    // For multi-way switches, create a chain of conditional branches
    // switchInt(discr) -> [v0: t0, v1: t1, ..., otherwise: default]
    // Becomes:
    //   current_block: cmp0 = discr == v0; cond_br cmp0, t0, intermediate_1
    //   intermediate_1: cmp1 = discr == v1; cond_br cmp1, t1, intermediate_2
    //   ...
    //   intermediate_N-1: cmpN = discr == v(N-1); cond_br cmpN, t(N-1), default
    use pliron::r#type::Typed;

    let n = branches.len();
    let discr_ty = discr_value.get_type(ctx);
    let bool_ty = types::get_bool_type(ctx);
    let (width, signedness) =
        if let Some(int_ty) = discr_ty.deref(ctx).downcast_ref::<IntegerType>() {
            (int_ty.width() as usize, int_ty.signedness())
        } else {
            (64, Signedness::Unsigned) // Default to 64-bit unsigned
        };

    // Create N-1 intermediate blocks for the comparison chain
    let mut intermediate_blocks: Vec<Ptr<BasicBlock>> = Vec::new();
    let mut prev_block = block_ptr;
    for _ in 0..(n - 1) {
        let intermediate = BasicBlock::new(ctx, None, vec![]);
        intermediate.insert_after(ctx, prev_block);
        intermediate_blocks.push(intermediate);
        prev_block = intermediate;
    }

    // Process each branch in the chain
    let mut current_block_ptr = block_ptr;
    let mut current_prev_op = last_op;

    for (i, (val, target_bb)) in branches.iter().enumerate() {
        let target_idx: usize = *target_bb;
        let target_block = block_map[target_idx];

        // Determine the "else" block (next in chain or otherwise).
        let else_block: Ptr<BasicBlock> = if i < n - 1 {
            intermediate_blocks[i]
        } else {
            block_map[otherwise_idx]
        };

        // Create constant for comparison with SAME type as discriminant.
        let int_attr = switch_arm_constant(ctx, *val, width, signedness);

        let const_op = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![discr_ty],
            vec![],
            vec![],
            0,
        );
        const_op.deref_mut(ctx).set_loc(loc.clone());
        let const_op_wrapped = MirConstantOp::new(const_op);
        const_op_wrapped.set_attr_value(ctx, int_attr);

        if let Some(prev) = current_prev_op {
            const_op_wrapped.get_operation().insert_after(ctx, prev);
        } else {
            const_op_wrapped
                .get_operation()
                .insert_at_front(current_block_ptr, ctx);
        }

        let const_val = const_op_wrapped.get_operation().deref(ctx).get_result(0);

        // Create comparison: discr == val
        let cmp_op = Operation::new(
            ctx,
            MirEqOp::get_concrete_op_info(),
            vec![bool_ty.to_handle()],
            vec![discr_value, const_val],
            vec![],
            0,
        );
        cmp_op.deref_mut(ctx).set_loc(loc.clone());
        cmp_op.insert_after(ctx, const_op_wrapped.get_operation());

        let condition = cmp_op.deref(ctx).get_result(0);

        // Alloca + load/store model: argument-less successor blocks; the
        // cond_br carries only its boolean condition.
        let (flat_operands, segment_sizes) =
            MirCondBranchOp::compute_segment_sizes(vec![vec![condition], vec![], vec![]]);

        let branch_op = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            flat_operands,
            vec![target_block, else_block],
            0,
        );
        Operation::get_op::<MirCondBranchOp>(branch_op, ctx)
            .expect("MirCondBranchOp")
            .set_operand_segment_sizes(ctx, segment_sizes);
        branch_op.deref_mut(ctx).set_loc(loc.clone());
        branch_op.insert_after(ctx, cmp_op);

        // Move to next intermediate block for next iteration
        if i < n - 1 {
            current_block_ptr = intermediate_blocks[i];
            current_prev_op = None;
        }
    }

    // Return the terminator from the original block
    let first_branch = block_ptr
        .deref(ctx)
        .iter(ctx)
        .last()
        .expect("Block should have terminator after multi-way switch translation");

    Ok(first_branch)
}

/// Translates a MIR `Drop` terminator.
///
/// rustc emits `TerminatorKind::Drop` only for places whose type has drop
/// glue. Two cases:
///
/// 1. **Provably no-op glue** (fast path): when the monomorphized drop glue
///    does nothing observable (checked by [`drop_glue::drop_glue_is_noop`]),
///    the terminator lowers to a plain branch to its target block.
///    The common source pattern is `for x in arr` over a by-value
///    array: the loop's `core::array::IntoIter<T, N>` has an
///    `impl Drop`, but for element types without drop glue that
///    destructor folds to nothing. This avoids the overhead of emitting
///    a function call for drops that are statically proven to do nothing.
///
/// 2. **Effectful glue** (fallback): when the no-op proof fails, the type
///    has a genuine destructor. We emit a device-side call to the
///    monomorphized `drop_in_place::<T>` function, which the collector has
///    already gathered and translated as a regular device function.
///
/// Suppressing drop glue on a Copy-shaped value (e.g. wrapping in
/// `core::mem::ManuallyDrop`) prevents the Drop terminator from being
/// emitted in the first place and lets the kernel compile without any
/// drop overhead.
///
/// The unwind action is ignored: device code is panic=abort, so there is
/// nothing that could unwind.
#[allow(clippy::too_many_arguments)]
fn translate_drop(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    target: mir::BasicBlockIdx,
    _unwind: &mir::UnwindAction,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    legaliser: &mut Legaliser,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let dropped_ty = place.ty(body.locals()).map_err(|e| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "drop terminator: failed to compute place type: {e:?}"
            ))
        )
    })?;

    // Fast path: if the drop is provably a no-op, emit a plain branch.
    if drop_glue::drop_glue_is_noop(dropped_ty) {
        return translate_goto(ctx, target, block_ptr, prev_op, block_map, loc);
    }

    // Fallback: emit a device-side call to drop_in_place::<T>(ptr).
    drop_glue::emit_drop_glue(
        ctx, body, place, dropped_ty, target, block_ptr, prev_op, value_map, block_map, legaliser,
        loc,
    )
}

// ============================================================================
// Call Translation (includes intrinsic dispatch)
// ============================================================================

/// True when `fn_def` is `core::ptr::drop_in_place` or `core::ptr::drop_glue`,
/// the only functions whose monomorphizations resolve to rustc's drop-glue
/// shims. Same crate + path-segment matching idiom as the callable-trait
/// detection below.
///
/// `drop_glue` (`#[lang = "drop_glue"]`) exists since nightly-2026-08-28:
/// `drop_in_place` is now an ordinary `core` function whose body forwards to
/// the compiler-replaced `drop_glue` shim, so a translated `drop_in_place`
/// body contains a plain Call to `drop_glue::<T>` that needs the same
/// noop-elision as direct `drop_in_place` calls.
fn is_drop_in_place_callee(fn_def: &rustc_public::ty::FnDef) -> bool {
    if fn_def.krate().name.as_str() != "core" {
        return false;
    }
    let method_name = fn_def.def_id().name();
    let method = method_name.as_str().rsplit("::").next().unwrap_or("");
    let Some(parent_def) = fn_def.def_id().parent() else {
        return false;
    };
    let parent_name = parent_def.name();
    let parent = parent_name.as_str().rsplit("::").next().unwrap_or("");
    (method == "drop_in_place" || method == "drop_glue") && parent == "ptr"
}

/// Emit a branch to `target` as the only effect of a call we are eliding:
/// the callee does nothing observable and has no device definition (a UB
/// precondition check, or provably no-op drop glue). `emit_goto` needs a
/// prior op to anchor after, so if the block has none yet we plant a dead
/// `false` constant first.
fn emit_elided_call_goto(
    ctx: &mut Context,
    target_idx: usize,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> Ptr<Operation> {
    let anchor = if let Some(p) = prev_op {
        p
    } else {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::utils::apint::APInt;
        use std::num::NonZeroUsize;

        let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let dummy = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![bool_ty.into()],
            vec![],
            vec![],
            0,
        );
        dummy.deref_mut(ctx).set_loc(loc.clone());
        let const_op = MirConstantOp::new(dummy);
        let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
        const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
        let dummy = const_op.get_operation();
        dummy.insert_at_front(block_ptr, ctx);
        dummy
    };
    helpers::emit_goto(ctx, target_idx, anchor, block_map, loc)
}

/// Translates a MIR `Call` terminator to Pliron IR operations.
///
/// This is the main entry point for function call translation. It handles:
///
/// 1. **Intrinsic dispatch**: Calls to `cuda_device::*` are expanded inline
///    to `dialect-nvvm` operations (thread IDs, barriers, warp ops, etc.)
///
/// 2. **Closure calls**: `FnOnce::call_once`, `FnMut::call_mut`, `Fn::call`
///    require unpacking tuple arguments before calling the closure body
///
/// 3. **Regular calls**: Other functions are emitted as `mir.call` operations
///
/// # GPU Constraints
///
/// Unwind edges are treated as unreachable (CUDA toolchain limitation, not HW).
///
/// # Flow
///
/// ```text
/// Call → extract_func_info → try_dispatch_intrinsic
///                         ↓ (if intrinsic)
///                         intrinsics::* handlers
///                         ↓ (if closure)
///                         translate_closure_call
///                         ↓ (otherwise)
///                         helpers::emit_function_call
/// ```
#[allow(clippy::too_many_arguments)]
fn translate_call(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<mir::BasicBlockIdx>,
    unwind: &mir::UnwindAction,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    // See comment in translate_assert for rationale.
    let _ = unwind;

    // Convert target to Option<usize>
    let target_usize = target.map(|t| t);

    // Extract function info
    let (pattern_name, call_name, substs_args, type_substs) = extract_func_info(func, &loc)?;
    let loc = call_debug_location(pattern_name.as_deref(), loc);

    // Is the trait-method Self type SharedArray? Shared with
    // `values::classify_call` so intrinsic dispatch and destination-slot
    // classification can't drift.
    let on_shared_array = substs_args.as_ref().is_some_and(self_ty_is_shared_array);

    // Skip precondition_check calls - these are UB check assertions that are
    // dead code because we return false for RuntimeChecks(UbChecks).
    // The MIR still contains these calls, but they're in dead branches.
    if let Some(target_idx) = target_usize
        && let mir::Operand::Constant(const_op) = func
        && let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, _)) =
            const_op.const_.ty().kind()
        && is_std_precondition_check(&fn_def)
    {
        // Just emit a goto to the target block, skipping the call entirely
        return Ok(emit_elided_call_goto(
            ctx, target_idx, block_ptr, prev_op, block_map, loc,
        ));
    }

    // Elide a call to provably no-op drop glue. An explicit
    // `ptr::drop_in_place::<T>` call (e.g. reached from inside another type's
    // `Drop::drop`, or from a libcore wrapper) resolves either to rustc's
    // empty shim for a `T` with no drop glue (`InstanceKind::DropGlue(_,
    // None)`) or to a shim body the shared no-op proof can discharge. The
    // collector refuses to collect exactly that set (`process_call_operand`
    // skips DropGlue callees via the same predicate), so emitting the call
    // would dangle as `Symbol ...drop_in_place... not found` at verification
    // time. The glue does nothing, so drop the call and branch straight to
    // the target -- the same elision the `Drop` terminator path performs via
    // `drop_glue_is_noop`, consulting the same shared predicate
    // (`drop_instance_is_noop`, whose fast path covers the empty shim) so
    // collection and emission stay in lockstep.
    if let Some(target_idx) = target_usize
        && let mir::Operand::Constant(const_op) = func
        && let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(
            fn_def,
            ref substs,
        )) = const_op.const_.ty().kind()
        && is_drop_in_place_callee(&fn_def)
        && let Ok(instance) = rustc_public::mir::mono::Instance::resolve(fn_def, substs)
        && drop_glue::drop_instance_is_noop(&instance)
    {
        return Ok(emit_elided_call_goto(
            ctx, target_idx, block_ptr, prev_op, block_map, loc,
        ));
    }

    // Identify the actual core callable-trait methods. Matching text in an
    // arbitrary function name is not sufficient: a user function named, for
    // example, `call_once_helper` must remain an ordinary call.
    if let Some(call_info) = callable_trait_call_info(func)
        && !args.is_empty()
    {
        if receiver_is_closure(&args[0], body) {
            return translate_closure_call(
                ctx,
                body,
                &call_name,
                call_info.resolved_is_shim,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                legaliser,
            );
        }
        if let Some(function_item) = extract_function_item_target(&args[0], body, &loc)? {
            return translate_function_item_call(
                ctx,
                body,
                &function_item,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                legaliser,
            );
        }
    }

    // Per-loop unroll marker from `#[unroll]` / `#[unroll(N)]`. The enclosing
    // `#[kernel]` or `#[device]` macro injects this call at the start of the loop
    // body, so we plant a `mir.unroll_hint` op right here, inside that loop
    // body. The loop-unroll pass later maps the hint back to its enclosing loop
    // (via LoopInfo) and consumes it, so it never reaches lowering. The factor
    // is the call's const generic (`0` = full unroll). Then branch to the
    // target as usual.
    // Match both the full path (`cuda_device::thread::__unroll_config`) and the
    // re-exported short path (`cuda_device::__unroll_config`), mirroring the
    // robust suffix match in `body::detect_unroll_config`.
    if is_cuda_device_const_marker(func, "__unroll_config") {
        let Some(factor) = extract_unroll_factor(func) else {
            return input_err!(
                loc,
                TranslationErr::invalid_op(
                    "could not read the const-generic factor from an unroll marker",
                )
            );
        };
        if factor == 1 || factor > 1024 {
            return input_err!(
                loc,
                TranslationErr::invalid_op(format!(
                    "partial unroll factor must be in 2..=1024, or 0 for full unrolling; got {factor}"
                ))
            );
        }
        let Some(target) = target_usize else {
            return input_err!(
                loc,
                TranslationErr::invalid_op("an unroll marker call has no return target")
            );
        };
        let hint = MirUnrollHintOp::new(ctx, factor).get_operation();
        hint.deref_mut(ctx).set_loc(loc.clone());
        match prev_op {
            Some(prev) => hint.insert_after(ctx, prev),
            None => hint.insert_at_front(block_ptr, ctx),
        }
        return Ok(helpers::emit_goto(ctx, target, hint, block_map, loc));
    }

    // Handle DynamicSharedArray specially to extract the ALIGN const generic.
    // The crate anchor tightens the name-substring gate: a user fn merely
    // named like it falls through to ordinary call handling. Kept in
    // lockstep with the classifier in `values::classify_call`.
    if let Some(ref name) = pattern_name
        && name.contains("DynamicSharedArray")
        && (name.contains("::get") || name.contains("::offset"))
        && is_cuda_device_fn(func)
    {
        // DynamicSharedArray substs arrive in declaration order, with the
        // ALIGN default already materialized by rustc:
        //
        //   DynamicSharedArray<T, ALIGN = 16>
        //     [0] T      type
        //     [1] ALIGN  const usize
        //
        // A read failure here must be an error, never a guessed 16: a
        // silently 16-aligned extern smem symbol corrupts TMA at runtime.
        let Some(substs) = substs_args.as_ref() else {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "DynamicSharedArray call carries no generic substitutions"
                )
            );
        };
        let Some(rustc_public::ty::GenericArgKind::Const(c)) = substs.0.get(1) else {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "DynamicSharedArray substs missing the ALIGN const generic at position 1"
                )
            );
        };
        let alignment: u64 = eval_usize_const(c, "DynamicSharedArray ALIGN", Some(&loc))?;

        if name.contains("::get") {
            // Both get() and get_raw() use the same handler with offset 0
            return intrinsics::memory::emit_dynamic_shared_get(
                ctx,
                body,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                0,         // byte_offset = 0 for get() and get_raw()
                alignment, // User-specified or default alignment
            );
        } else if name.contains("::offset") {
            // DynamicSharedArray::offset(byte_offset) - get pointer at byte offset
            return intrinsics::memory::emit_dynamic_shared_offset(
                ctx,
                body,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                alignment, // User-specified or default alignment
            );
        }
    }

    // Try to dispatch core::sync::atomic intrinsics (std::intrinsics::atomic_*)
    // These use const generics for ordering, so we intercept them here before
    // the regular intrinsic dispatch and extract generics from the func operand.
    if let Some(ref name) = pattern_name
        && intrinsics::atomic::is_core_atomic_intrinsic(name)
    {
        return intrinsics::atomic::dispatch_core_intrinsic(
            ctx,
            body,
            func,
            args,
            destination,
            &target_usize,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            name,
        );
    }

    // `assert_inhabited::<T>()` is a compile-time validity check that rustc
    // plants in `MaybeUninit::assume_init_read`, which the `for x in arr`
    // loop machinery calls for every yielded element (issue #138). The
    // intrinsic panics only when `T` has no possible values at all (an
    // "uninhabited" type such as `core::convert::Infallible`); for any
    // ordinary type it compiles to nothing. We decide which case applies
    // from the monomorphized type's layout: uninhabited types are exactly
    // those whose layout has `VariantsShape::Empty`. Inhabited types lower
    // to a unit no-op; uninhabited ones lower to a trap.
    // If the generic argument or its layout cannot be read, fall through
    // to the loud "not yet supported" rejection below.
    if let Some(ref name) = pattern_name
        && (name == "core::intrinsics::assert_inhabited"
            || name == "std::intrinsics::assert_inhabited")
        && let mir::Operand::Constant(const_op) = func
        && let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, substs)) =
            const_op.const_.ty().kind()
        && let Some(rustc_public::ty::GenericArgKind::Type(checked_ty)) = substs.0.first()
        && let Ok(layout) = checked_ty.layout()
    {
        let uninhabited = matches!(
            layout.shape().variants,
            rustc_public::abi::VariantsShape::Empty
        );
        if uninhabited {
            return Ok(emit_trap_unreachable_after(ctx, block_ptr, prev_op, loc));
        }
        return helpers::emit_unit_noop_intrinsic(
            ctx,
            body,
            destination,
            &target_usize,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            name,
        );
    }

    // Try to dispatch as intrinsic
    if let Some(ref name) = pattern_name
        && let Some(result) = try_dispatch_intrinsic(
            ctx,
            body,
            func,
            name,
            args,
            destination,
            &target_usize,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
            on_shared_array,
            &type_substs,
        )?
    {
        return Ok(result);
    }

    // Handle diverging calls (calls that never return, like unwrap_failed, panic, etc.)
    // These have no target block because the function never returns.
    //
    // The callee is dropped and replaced by an immediate trap. That applies
    // to every `-> !` callee reaching this point, including a user
    // `#[device]` fn that legitimately diverges (e.g. `loop {}`); full Rust
    // fidelity would emit the call for resolvable callees the way
    // `translate_function_item_call` does. Dropping the call is a
    // pre-existing semantic: the trap only makes it safe, where the bare
    // `unreachable` previously emitted here was UB that let `opt` delete
    // the whole panic path.
    //
    // Panic entry points additionally never get here with any statements
    // translated ahead of them: `block::translate_block` recognizes the same
    // shape via [`is_dropped_panic_call`] and emits the trap directly.
    if target_usize.is_none() {
        // This is a diverging call (returns !) - emit trap + unreachable
        // Examples: unwrap_failed(), panic!(), abort()
        return Ok(emit_trap_unreachable_after(ctx, block_ptr, prev_op, loc));
    }

    // A call to a rustc intrinsic that no dispatch arm above recognized can
    // never be emitted as a regular function call: rustc resolves intrinsics
    // to `InstanceKind::Intrinsic`, the collector skips those by design, so
    // no definition for the symbol will ever exist in the module. Emitting
    // the call anyway would only fail much later, as a confusing
    // "Symbol ... not found" verifier error on the LLVM dialect module.
    // Fail here instead, with the intrinsic's name and source location, so
    // each gap surfaces as an actionable per-site diagnostic (issue #137).
    if let Some(ref name) = pattern_name
        && (name.starts_with("core::intrinsics::") || name.starts_with("std::intrinsics::"))
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "rustc intrinsic `{name}` is not yet supported on the device"
            ))
        );
    }

    // The collector skips the entire `libm` crate: every libm call must be
    // intercepted by the float-math dispatch above and rerouted to a
    // libdevice intrinsic, so no definition for a libm symbol ever exists in
    // the module. A libm function the dispatch does not recognize would only
    // fail much later, as a bare "Symbol libm__cbrtf not found" verifier
    // error on the LLVM dialect module. Fail here instead, with the
    // function's name and source location.
    if let Some(ref name) = pattern_name
        && intrinsics::float_math::is_libm_path(name)
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "libm function `{name}` is not yet mapped to a libdevice intrinsic; \
                 add it to `from_libm_path` in the mir-importer float-math dispatch"
            ))
        );
    }

    // Not an intrinsic - emit regular function call
    let raw_name = call_name.unwrap_or_else(|| "unknown_function".to_string());
    let legal_name = legaliser.legalise(&raw_name);

    // Foreign items have no `mir.func` body to provide an independent symbol
    // signature. Preserve the exact resolved rustc declaration on the call so
    // final module verification can still reject pointer-kind laundering at
    // this ABI boundary. This must be derived from `func`, not reconstructed
    // from the already-created call operands or destination.
    let external_callee_type = translate_foreign_callee_type(ctx, func)?;

    // Type the call result from the caller's destination place, not from the
    // callee's declared signature. The declared signature of a trait method
    // is written against the trait, so its return type can be an unresolved
    // associated-type projection such as `<&Foo as Mul>::Output` (issue #133),
    // which the type translator cannot turn into a concrete layout. The
    // destination local in the caller's monomorphized MIR already has that
    // projection resolved (`Foo`), and it is by construction the exact type
    // the call result is stored into, so the `mir.call` result type and the
    // destination slot always agree. The callee `mir.func` return type is
    // independently derived from the callee body's return place, which is
    // normalized the same way, so caller and callee stay consistent.
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;

    helpers::emit_function_call(
        ctx,
        body,
        legal_name.as_ref(),
        args,
        destination,
        return_type,
        external_callee_type,
        &target_usize,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}

/// Translate the exact signature of a resolved Rust foreign item.
///
/// Ordinary Rust callees are independently typed by their `mir.func` op and
/// therefore return `None`. Foreign declarations have no MIR body, so this
/// `FunctionType` is attached to the call with `AbiBoundary` authority and is
/// checked again by the final dialect verifier.
fn translate_foreign_callee_type(
    ctx: &mut Context,
    func: &mir::Operand,
) -> TranslationResult<Option<TypeHandle>> {
    use dialect_mir::types::MirTupleType;
    use rustc_public::mir::mono::Instance;
    use rustc_public::ty::{RigidTy, TyKind};

    let mir::Operand::Constant(constant) = func else {
        return Ok(None);
    };
    if *constant.const_.kind() != ConstantKind::ZeroSized {
        return Ok(None);
    }
    let TyKind::RigidTy(RigidTy::FnDef(fn_def, substs)) = constant.const_.ty().kind() else {
        return Ok(None);
    };
    let Some(instance) = Instance::resolve(fn_def, &substs).ok() else {
        return Ok(None);
    };
    if !instance.is_foreign_item() {
        return Ok(None);
    }

    let fn_ty = instance.ty();
    let Some(signature) = fn_ty.kind().fn_sig() else {
        return input_err!(
            pliron::location::Location::Unknown,
            TranslationErr::unsupported(format!(
                "resolved foreign item `{}` has no function signature",
                instance.name()
            ))
        );
    };
    let signature = signature.skip_binder();
    ensure_foreign_callee_is_not_variadic(&instance.name().to_string(), signature.c_variadic)?;
    let mut arguments = Vec::with_capacity(signature.inputs().len());
    for argument in signature.inputs() {
        arguments.push(types::translate_type(ctx, argument)?);
    }

    let output = signature.output();
    let output = types::translate_type(ctx, &output)?;
    let is_unit = output
        .deref(ctx)
        .downcast_ref::<MirTupleType>()
        .is_some_and(|tuple| tuple.get_types().is_empty());
    let results = if is_unit { vec![] } else { vec![output] };

    Ok(Some(FunctionType::get(ctx, arguments, results).into()))
}

/// The builtin source `FunctionType` and MIR call operation are fixed-arity.
/// Refuse a C-variadic foreign declaration instead of recording only its fixed
/// prefix and incorrectly presenting that incomplete type as the exact ABI.
fn ensure_foreign_callee_is_not_variadic(
    callee_name: &str,
    c_variadic: bool,
) -> TranslationResult<()> {
    if c_variadic {
        return input_err!(
            pliron::location::Location::Unknown,
            TranslationErr::unsupported(format!(
                "resolved foreign item `{callee_name}` is C-variadic; variadic foreign calls are not supported"
            ))
        );
    }
    Ok(())
}

// stable-MIR reports the function's canonical definition path rather than the
// `cuda_device::thread::__internal` re-export spelling emitted by the macro,
// but accept both spellings like the metadata markers below so a toolchain
// bump changing visible-path selection cannot silently drop the match.
const KERNEL_SCOPE_CONSTRUCTORS: &[&str] = &[
    "cuda_device::__internal::make_kernel_scope",
    "cuda_device::thread::__internal::make_kernel_scope",
];

// These start-of-kernel calls are compiler metadata, not source statements.
// They are removed during import, but the replacement control-flow edge must
// also stay off the attribute line which caused the macro to inject them.
const KERNEL_METADATA_MARKERS: &[&str] = &[
    "cuda_device::__launch_bounds_config",
    "cuda_device::thread::__launch_bounds_config",
    "cuda_device::__launch_contract_config",
    "cuda_device::thread::__launch_contract_config",
    "cuda_device::__launch_contract_block_config",
    "cuda_device::thread::__launch_contract_block_config",
    "cuda_device::__grid_constant_config",
    "cuda_device::thread::__grid_constant_config",
    "cuda_device::__unchecked_indexing_config",
    "cuda_device::thread::__unchecked_indexing_config",
    "cuda_device::cluster::__cluster_config",
    "cuda_device::shared::__dynamic_shared_alignment",
];

/// Keep the macro-injected launch-context constructor out of user line tables.
///
/// Its Rust span intentionally stays at the proc-macro invocation so any
/// type-checker or borrow-checker diagnostic points into the user's item. The
/// call itself is compiler-generated setup, though, and must not make a
/// function breakpoint stop on `#[kernel]` (or whichever adjacent attribute
/// happened to invoke the macro). Separating the debug location here lets the
/// diagnostic span and runtime line table serve those different purposes.
fn call_debug_location(pattern_name: Option<&str>, loc: Location) -> Location {
    if pattern_name.is_some_and(|name| {
        KERNEL_SCOPE_CONSTRUCTORS.contains(&name) || KERNEL_METADATA_MARKERS.contains(&name)
    }) {
        llvm_export::artificial_debug_location()
    } else {
        loc
    }
}

/// Handle `FnOnce::call_once`, `FnMut::call_mut`, or `Fn::call` when the
/// receiver is a function item.
///
/// Rust-call ABI passes `(self, tuple_args)` to the trait shim, but the
/// function item's real body expects only the tuple elements as ordinary
/// arguments. Lowering the shim as an ordinary call leaves a dangling
/// `<fn item as FnOnce>::call_once` symbol because no MIR body is collected
/// for that shim.
#[allow(clippy::too_many_arguments)]
fn translate_function_item_call(
    ctx: &mut Context,
    body: &mir::Body,
    function_item: &FunctionItemTarget,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirCallOp, MirExtractFieldOp};
    use pliron::builtin::attributes::StringAttr;
    use pliron::identifier::Identifier;

    if function_item.requires_direct_dispatch {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "calling `{}` through Fn/FnMut/FnOnce is not yet supported because that target requires intrinsic or external dispatch; wrap it in a local `#[device]` function and pass the wrapper instead",
                function_item.name
            ))
        );
    }

    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;
    let callee = legaliser.legalise(&function_item.name).to_string();

    let mut unpacked_args = Vec::new();
    let mut last_op = prev_op;

    if let Some(tuple_arg) = args.get(1) {
        let (tuple_value, tuple_last_op) = rvalue::translate_operand(
            ctx,
            body,
            tuple_arg,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = tuple_last_op;

        let tuple_ty = tuple_value.get_type(ctx);
        let element_types: Option<Vec<_>> = {
            let tuple_ty_obj = tuple_ty.deref(ctx);
            tuple_ty_obj
                .downcast_ref::<dialect_mir::types::MirTupleType>()
                .map(|mir_tuple_ty| mir_tuple_ty.get_types().to_vec())
        };

        if let Some(element_types) = element_types {
            for (i, elem_ty) in element_types.iter().enumerate() {
                let extract_op = Operation::new(
                    ctx,
                    MirExtractFieldOp::get_concrete_op_info(),
                    vec![*elem_ty],
                    vec![tuple_value],
                    vec![],
                    0,
                );
                extract_op.deref_mut(ctx).set_loc(loc.clone());

                let mir_extract = MirExtractFieldOp::new(extract_op);
                mir_extract.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(i as u32));

                if let Some(prev) = last_op {
                    extract_op.insert_after(ctx, prev);
                } else {
                    extract_op.insert_at_front(block_ptr, ctx);
                }
                last_op = Some(extract_op);
                unpacked_args.push(extract_op.deref(ctx).get_result(0));
            }
        } else {
            unpacked_args.push(tuple_value);
        }
    }

    let prepared_destination = if target.is_some() {
        let (prepared, prepared_last_op) = helpers::prepare_destination_write(
            ctx,
            body,
            destination,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = prepared_last_op;
        Some(prepared)
    } else {
        None
    };

    let call_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![return_type],
        unpacked_args,
        vec![],
        0,
    );
    call_op.deref_mut(ctx).set_loc(loc.clone());

    call_op.deref_mut(ctx).attributes.set(
        Identifier::try_from("callee").unwrap(),
        StringAttr::new(callee),
    );

    if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
    } else {
        call_op.insert_at_front(block_ptr, ctx);
    }

    if target.is_none() {
        return Ok(emit_unreachable_after(ctx, block_ptr, Some(call_op), loc));
    }

    let result_value = call_op.deref(ctx).get_result(0);
    let last_inserted = helpers::finish_destination_write(
        ctx,
        prepared_destination.expect("returning call prepared its destination"),
        result_value,
        value_map,
        block_ptr,
        call_op,
        loc.clone(),
    )?;

    if let Some(target_idx) = target {
        Ok(helpers::emit_goto(
            ctx,
            *target_idx,
            last_inserted,
            block_map,
            loc,
        ))
    } else {
        Ok(call_op)
    }
}

/// Handle closure trait method calls (FnOnce::call_once, FnMut::call_mut, Fn::call).
///
/// These calls pass arguments as a tuple, but the closure body expects unpacked args:
/// - MIR: `<{closure} as FnMut<(u32,)>>::call_mut(self_ref, tuple_args)`
/// - Closure body expects: `fn(self_ref, unpacked_arg1, unpacked_arg2, ...)`
///
/// We detect these calls and unpack the tuple argument before calling the closure.
///
/// ## Important: Closure Body Resolution
///
/// When an `Fn` or `FnMut` closure is requested through `FnOnce`,
/// `Instance::resolve` returns a `ClosureOnce` adapter shim rather than the
/// closure body. Other closure calls may resolve directly to the body. In
/// either case, extract the closure DefId from `args[0]` so the emitted device
/// call targets the body; `resolved_is_shim` separately controls receiver
/// adaptation.
///
/// See `device_closures/README.md` for detailed documentation.
#[allow(clippy::too_many_arguments)]
fn translate_closure_call(
    ctx: &mut Context,
    body: &mir::Body,
    call_name: &Option<String>,
    resolved_is_shim: bool,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirCallOp, MirExtractFieldOp};
    use pliron::builtin::attributes::StringAttr;
    use pliron::identifier::Identifier;
    use pliron::r#type::Typed;

    // Same reasoning as the regular-call path: the trait-level signature of
    // `FnOnce::call_once` types its result as the projection
    // `<{closure} as FnOnce<Args>>::Output`. The caller's destination local
    // carries the already-resolved concrete type, so use that.
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;

    // Resolve the body independently from the callable-trait method. Besides
    // selecting the body rather than a ClosureOnce adapter, this gives us the
    // body's actual first parameter. That parameter is authoritative for the
    // receiver kind: `&Closure` is SharedRef, `&mut Closure` is UniqueRef, and
    // an FnOnce body receives the closure by value.
    let Some(closure_body) = extract_closure_body_target(&args[0], body) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "failed to resolve the closure body and receiver ABI for `{}`",
                call_name.as_deref().unwrap_or("unknown closure")
            ))
        );
    };
    let callee = legaliser.legalise(&closure_body.name).to_string();
    let expected_receiver_type = types::translate_type(ctx, &closure_body.receiver_ty)?;

    // Translate self argument (args[0])
    let (self_value, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Translate tuple argument (args[1])
    let (tuple_value, tuple_last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = tuple_last_op;

    // Derive the exact receiver reference from the closure body's first
    // parameter. A ClosureOnce adapter may pass the closure by value and need
    // a synthesized borrow. A direct `FnMut::call_mut` can also resolve to an
    // `Fn` closure body: MIR then supplies `&mut Closure` while that body
    // expects `&Closure`, so preserve the existing address with an explicit
    // reborrow instead of taking the address of the reference value itself.
    let receiver_borrow_mutability = match closure_body.receiver_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, _, mutability))
            if self_value.get_type(ctx) != expected_receiver_type =>
        {
            Some(mutability)
        }
        _ => None,
    };

    let self_arg = if let Some(mutability) = receiver_borrow_mutability {
        let is_mutable = mutability == mir::Mutability::Mut;

        if self_value
            .get_type(ctx)
            .deref(ctx)
            .is::<dialect_mir::types::MirPtrType>()
        {
            // The callable-trait receiver is already an address. Retype the
            // same pointee at the Rust reborrow boundary (`&mut C -> &C` for
            // an Fn closure invoked through FnMut is the common case).
            let cast_op = Operation::new(
                ctx,
                dialect_mir::ops::MirCastOp::get_concrete_op_info(),
                vec![expected_receiver_type],
                vec![self_value],
                vec![],
                0,
            );
            cast_op.deref_mut(ctx).set_loc(loc.clone());
            let cast = dialect_mir::ops::MirCastOp::new(cast_op);
            cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
            cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::Reborrow);
            if let Some(prev) = last_op {
                cast_op.insert_after(ctx, prev);
            } else {
                cast_op.insert_at_front(block_ptr, ctx);
            }
            last_op = Some(cast_op);
            cast_op.deref(ctx).get_result(0)
        } else {
            if !resolved_is_shim {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(
                        "closure body expects a borrowed receiver but the direct callable-trait argument is not a pointer"
                            .to_string()
                    )
                );
            }

            // Reproduce the ClosureOnce adapter's borrow before calling the
            // closure body directly.
            let ref_op = Operation::new(
                ctx,
                dialect_mir::ops::MirRefOp::get_concrete_op_info(),
                vec![expected_receiver_type],
                vec![self_value],
                vec![],
                0,
            );
            ref_op.deref_mut(ctx).set_loc(loc.clone());
            let ref_op_wrapper = dialect_mir::ops::MirRefOp::new(ref_op);
            ref_op_wrapper.set_mutable(ctx, is_mutable);
            ref_op_wrapper.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::Reborrow);

            if let Some(prev) = last_op {
                ref_op.insert_after(ctx, prev);
            } else {
                ref_op.insert_at_front(block_ptr, ctx);
            }
            last_op = Some(ref_op);
            ref_op.deref(ctx).get_result(0)
        }
    } else {
        // For call_mut/call the MIR operand is already the body's exact
        // reference type; for FnOnce the body receives the value unchanged.
        self_value
    };

    if self_arg.get_type(ctx) != expected_receiver_type {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "closure call receiver type does not match the resolved closure body parameter"
                    .to_string()
            )
        );
    }

    // Build unpacked arguments, starting with the original or adapted receiver.
    let mut unpacked_args = vec![self_arg];

    // Unpack the tuple - extract each field
    let tuple_ty = tuple_value.get_type(ctx);
    let element_types: Option<Vec<_>> = {
        let tuple_ty_obj = tuple_ty.deref(ctx);
        tuple_ty_obj
            .downcast_ref::<dialect_mir::types::MirTupleType>()
            .map(|mir_tuple_ty| mir_tuple_ty.get_types().to_vec())
    };

    if let Some(element_types) = element_types {
        for (i, elem_ty) in element_types.iter().enumerate() {
            // Create extract_field operation
            let extract_op = Operation::new(
                ctx,
                MirExtractFieldOp::get_concrete_op_info(),
                vec![*elem_ty],
                vec![tuple_value],
                vec![],
                0,
            );
            extract_op.deref_mut(ctx).set_loc(loc.clone());

            let mir_extract = MirExtractFieldOp::new(extract_op);
            mir_extract.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(i as u32));

            // Insert after previous op
            if let Some(prev) = last_op {
                extract_op.insert_after(ctx, prev);
            } else {
                extract_op.insert_at_front(block_ptr, ctx);
            }
            last_op = Some(extract_op);

            // Get the extracted value
            let elem_value = extract_op.deref(ctx).get_result(0);
            unpacked_args.push(elem_value);
        }
    } else {
        // Not a tuple type - just pass as is (single arg case)
        unpacked_args.push(tuple_value);
    }

    let prepared_destination = if target.is_some() {
        let (prepared, prepared_last_op) = helpers::prepare_destination_write(
            ctx,
            body,
            destination,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = prepared_last_op;
        Some(prepared)
    } else {
        None
    };

    // Now emit the call with unpacked arguments
    let call_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![return_type],
        unpacked_args,
        vec![],
        0,
    );
    call_op.deref_mut(ctx).set_loc(loc.clone());

    // Set callee attribute
    let callee_attr = StringAttr::new(callee);
    call_op
        .deref_mut(ctx)
        .attributes
        .set(Identifier::try_from("callee").unwrap(), callee_attr);

    // Insert the call
    let call_op = if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
        call_op
    } else {
        call_op.insert_at_front(block_ptr, ctx);
        call_op
    };

    if target.is_none() {
        return Ok(emit_unreachable_after(ctx, block_ptr, Some(call_op), loc));
    }

    let result_value = call_op.deref(ctx).get_result(0);
    let last_inserted = helpers::finish_destination_write(
        ctx,
        prepared_destination.expect("returning call prepared its destination"),
        result_value,
        value_map,
        block_ptr,
        call_op,
        loc.clone(),
    )?;

    // Emit goto to target
    if let Some(target_idx) = target {
        Ok(helpers::emit_goto(
            ctx,
            *target_idx,
            last_inserted,
            block_map,
            loc,
        ))
    } else {
        Ok(call_op)
    }
}

/// Terminates the block with a bare `mir.unreachable`.
///
/// Only for paths that can never execute: rustc-proven unreachability
/// (`TerminatorKind::Unreachable`) or code after a call that is emitted into
/// the module and genuinely never returns.
///
/// Runtime-reachable panic paths whose diverging call is dropped must use
/// [`emit_trap_unreachable_after`] instead.
fn emit_unreachable_after(
    ctx: &mut Context,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }
    op
}

/// Returns true for the panic entry points in `core` (and the `std`
/// re-export) that mark a basic block as a panic path.
///
/// This is the single source of truth for that test. The codegen collector
/// (`rustc-codegen-cuda/src/collector.rs`) imports it so the callees it
/// deliberately does not collect are exactly the calls this translator drops
/// in favour of a device trap: every call this predicate accepts has to be
/// dropped here instead of being emitted as a call to a symbol the module
/// never defines.
///
/// The substring match is intentionally broad: a user function whose path
/// contains a `panicking` module segment is also treated as a panic entry
/// and trapped rather than translated.
pub fn is_panic_entry_path(fn_path: &str) -> bool {
    fn_path.contains("::panicking::") || fn_path.contains("::rt::panic")
}

/// True when `term` is a diverging call into a panic entry point, i.e. exactly
/// the shape `translate_call` (private to this module) drops in favour of a
/// device trap.
///
/// Callers use this to recognize a block that lowers to nothing but a trap,
/// before spending any work on its contents.
pub fn is_dropped_panic_call(term: &mir::Terminator) -> bool {
    let mir::TerminatorKind::Call {
        func: mir::Operand::Constant(const_op),
        target: None,
        ..
    } = &term.kind
    else {
        return false;
    };
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, _)) =
        const_op.const_.ty().kind()
    else {
        return false;
    };
    is_panic_entry_path(fn_def.name().as_str())
}

/// Lowers a block whose terminator [`is_dropped_panic_call`] to the device
/// trap alone, at the panic call's source location.
pub fn emit_dropped_panic_trap(
    ctx: &mut Context,
    term: &mir::Terminator,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) -> Ptr<Operation> {
    let loc = span_to_location(ctx, term.source_info.span);
    emit_trap_unreachable_after(ctx, block_ptr, prev_op, loc)
}

/// Terminates the block with `nvvm.trap` followed by `mir.unreachable`.
///
/// For runtime-reachable diverging paths whose call is not emitted (dropped
/// panic calls). A thread reaching it aborts the kernel (`trap;` in PTX).
fn emit_trap_unreachable_after(
    ctx: &mut Context,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    let trap_op = dialect_nvvm::ops::TrapOp::build(ctx);
    trap_op.deref_mut(ctx).set_loc(loc.clone());
    // `nvvm.trap` is a generated intrinsic; the target-requirements
    // verifier rejects it without its exact ABI marker (`i0295` is
    // trap's append-only id in intrinsics/abi-v1.toml, the same marker
    // the generated `debug::trap` dispatch attaches).
    helpers::set_generated_intrinsic_marker(ctx, trap_op, "v1:i0295");
    if let Some(prev) = prev_op {
        trap_op.insert_after(ctx, prev);
    } else {
        trap_op.insert_at_front(block_ptr, ctx);
    }
    emit_unreachable_after(ctx, block_ptr, Some(trap_op), loc)
}

/// True only when the rust-call receiver is itself a closure.
fn receiver_is_closure(receiver: &mir::Operand, body: &mir::Body) -> bool {
    let Some(ty) = operand_type(receiver, body) else {
        return false;
    };

    let inner = match ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, inner, _)) => inner,
        _ => ty,
    };

    matches!(
        inner.kind(),
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(_, _))
    )
}

struct FunctionItemTarget {
    name: String,
    requires_direct_dispatch: bool,
}

fn extract_function_item_target(
    receiver: &mir::Operand,
    body: &mir::Body,
    loc: &Location,
) -> TranslationResult<Option<FunctionItemTarget>> {
    let Some(ty) = operand_type(receiver, body) else {
        return Ok(None);
    };
    let inner = match ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, inner, _)) => inner,
        _ => ty,
    };

    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, substs)) =
        inner.kind()
    else {
        return Ok(None);
    };

    let Some(instance) = rustc_public::mir::mono::Instance::resolve(fn_def, &substs).ok() else {
        return Ok(None);
    };
    let crate_name = fn_def.krate().name;
    let generated_direct_call_only =
        intrinsics::generated::require_supported_raw_intrinsic(fn_def, loc)?.is_some();
    Ok(Some(FunctionItemTarget {
        name: function_item_call_name(instance),
        requires_direct_dispatch: fn_def.is_intrinsic()
            || instance.is_foreign_item()
            || !instance.has_body()
            || matches!(crate_name.as_str(), "cuda_device" | "cuda-device" | "libm")
            || generated_direct_call_only,
    }))
}

fn function_item_call_name(instance: rustc_public::mir::mono::Instance) -> String {
    if instance.is_foreign_item() || !instance.args().0.is_empty() {
        instance.mangled_name()
    } else {
        instance.name().to_string()
    }
}

fn operand_type(receiver: &mir::Operand, body: &mir::Body) -> Option<rustc_public::ty::Ty> {
    receiver.ty(body.locals()).ok()
}

struct CallableTraitCallInfo {
    resolved_is_shim: bool,
}

/// Recognize the three callable traits by compiler identity available through
/// `rustc_public`.
///
/// These methods are defined in `core`, use the rust-call ABI, and have an
/// exact parent/method pair (`Fn::call`, `FnMut::call_mut`, or
/// `FnOnce::call_once`). The combined check cannot be spoofed by a user item
/// whose name merely contains one of those strings. The resolved instance may
/// be either a shim or the closure body itself, so instance kind is returned as
/// lowering information rather than used as the recognition predicate.
fn callable_trait_call_info(func: &mir::Operand) -> Option<CallableTraitCallInfo> {
    use rustc_public::mir::mono::{Instance, InstanceKind};
    use rustc_public::ty::{Abi, RigidTy, TyKind};

    let mir::Operand::Constant(const_op) = func else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(fn_def, substs)) = const_op.const_.ty().kind() else {
        return None;
    };
    if fn_def.fn_sig().skip_binder().abi != Abi::RustCall || fn_def.krate().name.as_str() != "core"
    {
        return None;
    }

    let method_name = fn_def.def_id().name();
    let method = method_name.as_str().rsplit("::").next()?;
    let parent_name = fn_def.def_id().parent()?.name();
    let parent = parent_name.as_str().rsplit("::").next()?;
    let is_callable_method = matches!(
        (parent, method),
        ("Fn", "call") | ("FnMut", "call_mut") | ("FnOnce", "call_once")
    );
    if !is_callable_method {
        return None;
    }

    let instance = Instance::resolve(fn_def, &substs).ok()?;
    Some(CallableTraitCallInfo {
        resolved_is_shim: instance.kind == InstanceKind::Shim,
    })
}

struct ClosureBodyTarget {
    name: String,
    receiver_ty: rustc_public::ty::Ty,
}

/// Resolves the closure body and its actual receiver parameter.
///
/// When instance resolution selects an adapter shim for a call like:
///   `<{closure} as FnOnce<(u32,)>>::call_once(closure_ref, args_tuple)`
///
/// the resolved call name identifies the shim, not the closure body. Extract
/// the closure's DefId from `args[0]` and resolve the body independently.
///
/// The closure argument can be:
/// - A direct closure value (type is `Closure(def, substs)`)
/// - A reference to a closure (type is `Ref(_, Closure(def, substs), _)`)
/// - A mutable reference (same pattern)
fn extract_closure_body_target(
    closure_arg: &mir::Operand,
    body: &mir::Body,
) -> Option<ClosureBodyTarget> {
    let closure_ty = operand_type(closure_arg, body)?;

    // Unwrap references to get the actual closure type
    let inner_ty = match closure_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, inner, _)) => inner,
        _ => closure_ty,
    };

    // Extract the closure DefId and substs
    let (closure_def, substs) = match inner_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(def, substs)) => {
            (def, substs.clone())
        }
        _ => return None,
    };

    // Get the closure body instance directly. Resolving the callable-trait
    // method can legitimately select a ClosureOnce adapter instead.
    //
    // The closure_def.def_id() gives us the DefId of the closure body.
    // We construct the mangled name by creating an FnDef and resolving it.
    use rustc_public::mir::mono::Instance;
    use rustc_public::ty::FnDef;

    // Create an FnDef from the closure's DefId
    let fn_def = FnDef(closure_def.def_id());

    let instance = Instance::resolve(fn_def, &substs).ok().or_else(|| {
        // Fallback: try the old resolve_closure method.
        Instance::resolve_closure(closure_def, &substs, rustc_public::ty::ClosureKind::FnOnce).ok()
    })?;
    let receiver_ty = instance.body()?.arg_locals().first()?.ty;

    Some(ClosureBodyTarget {
        name: instance.mangled_name(),
        receiver_ty,
    })
}

/// Read the const-generic `FACTOR` from a `__unroll_config::<FACTOR>()` callee
/// (`0` = full unroll).
///
/// Returns `None` when the callee is malformed instead of silently turning the
/// request into a full unroll.
fn extract_unroll_factor(func: &mir::Operand) -> Option<u32> {
    use rustc_public::ty::{RigidTy, TyConstKind, TyKind};
    let mir::Operand::Constant(constant) = func else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, args)) = constant.const_.ty().kind() else {
        return None;
    };
    let definition_name = definition.name();
    if definition.krate().name.as_str() != "cuda_device"
        || (definition_name != "__unroll_config" && !definition_name.ends_with("::__unroll_config"))
        || args.0.len() != 1
    {
        return None;
    }
    if let Some(arg) = args.0.first()
        && let rustc_public::ty::GenericArgKind::Const(c) = arg
    {
        return match c.kind() {
            TyConstKind::Value(_, alloc) => {
                alloc.read_uint().ok().and_then(|v| u32::try_from(v).ok())
            }
            _ => c
                .eval_target_usize()
                .ok()
                .and_then(|v| u32::try_from(v).ok()),
        };
    }
    None
}

/// Match a compiler marker by its resolved definition, not by a debug string.
///
/// A user function with the same spelling must remain an ordinary call. The
/// `FnDef` identity also guarantees that generic arguments were type-checked by
/// rustc before the importer reads them.
fn is_cuda_device_const_marker(func: &mir::Operand, expected_name: &str) -> bool {
    use rustc_public::CrateDef;
    use rustc_public::ty::{RigidTy, TyKind};

    let mir::Operand::Constant(constant) = func else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, _)) = constant.const_.ty().kind() else {
        return false;
    };
    if definition.krate().name.as_str() != "cuda_device" {
        return false;
    }
    let definition_name = definition.name();
    definition_name == expected_name || definition_name.ends_with(&format!("::{expected_name}"))
}

/// Extracts function metadata from a MIR function operand.
///
/// Returns a tuple of:
/// - `pattern_name`: The function's simple name (e.g., `"cuda_device::index_1d"`)
/// - `call_name`: The name used for the call target in generated code
/// - `substs`: the callee's generic substitutions, in declaration order
///   (`Self` first for trait methods)
///
/// Deliberately NOT returned: the callee's declared return type. The
/// declared `fn_sig` of a trait method is written against the trait, so its
/// output can be an unresolved associated-type projection such as
/// `<&Foo as Mul>::Output` (issue #133). Call results are instead typed from
/// the caller's destination place, which rustc has already monomorphized and
/// normalized. If a callee-signature type is ever genuinely needed here,
/// resolve the instance first (`Instance::resolve`) and query the signature
/// on the resolved instance so associated types arrive normalized.
///
/// This information is used to:
/// 1. Match intrinsic patterns by `pattern_name` (full FQDN, e.g. `cuda_device::thread::threadIdx_x`)
/// 2. Read the trait-method `Self` type from the substs (Index/IndexMut on
///    `SharedArray`) and const generics (`DynamicSharedArray`'s `ALIGN`)
/// 3. Generate the correct call target name (FQDN for non-generic, mangled for generic)
///
/// # Naming strategy
///
/// `CrateDef::name()` returns the fully qualified name (FQDN) in the
/// `rustc_public` API (e.g. `helper_fn::cuda_oxide_device_<hash>_vecadd_device`).
/// We use the raw `FnDef` name as `pattern_name` for intrinsic matching, then
/// use the resolved `Instance::name()` as `call_name` for monomorphic calls.
/// Raw `FnDef` substitutions are not authoritative for this decision: concrete
/// trait impl calls can carry a trait self type in the operand while resolving
/// to a monomorphic instance, and the resolved impl FQDN can differ from the
/// trait item FQDN.
///
/// For resolved instances that still carry generic args, `Instance::mangled_name`
/// is used instead, which the collector also matches via `compute_export_name`.
/// Non-generic FQDNs with characters such as `<`, `>`, and `::` are passed raw to
/// the same pliron `Legaliser` used for definition symbols.
///
/// Foreign items (`extern "C"` block declarations) are the exception: they
/// have no MIR body, so the collector never exports a definition under the
/// FQDN and the device linker (libdevice, external LTOIR) only knows the
/// link symbol. `call_name` for those is `Instance::mangled_name`, which is
/// the link symbol (it honours `#[link_name]`).
fn extract_func_info(
    func: &mir::Operand,
    loc: &Location,
) -> TranslationResult<(
    Option<String>,
    Option<String>,
    Option<rustc_public::ty::GenericArgs>,
    Vec<rustc_public::ty::Ty>,
)> {
    Ok(match func {
        mir::Operand::Constant(const_op) => match const_op.const_.kind() {
            ConstantKind::ZeroSized => {
                let ty_kind = const_op.const_.ty().kind();
                match &ty_kind {
                    rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(
                        fn_def,
                        substs,
                    )) => {
                        use rustc_public::mir::mono::Instance;

                        let pattern_name =
                            intrinsics::generated::require_supported_raw_intrinsic(*fn_def, loc)?
                                .unwrap_or_else(|| fn_def.name().as_str().to_string());

                        let resolved = Instance::resolve(*fn_def, substs).ok();
                        let call_name = if let Some(instance) = resolved {
                            if instance.is_foreign_item() {
                                // Foreign items (`extern "C"` blocks) have no MIR
                                // body, so no definition is ever exported under
                                // the FQDN. Emit the call under the link symbol
                                // (e.g. `__nv_asinf`), which is what libdevice or
                                // externally linked LTOIR actually provides.
                                instance.mangled_name()
                            } else if !instance.args().0.is_empty() {
                                instance.mangled_name()
                            } else {
                                instance.name().to_string()
                            }
                        } else {
                            pattern_name.clone()
                        };

                        let type_substs = substs
                            .0
                            .iter()
                            .filter_map(|arg| match arg {
                                rustc_public::ty::GenericArgKind::Type(ty) => Some(*ty),
                                _ => None,
                            })
                            .collect();
                        (
                            Some(pattern_name),
                            Some(call_name),
                            Some(substs.clone()),
                            type_substs,
                        )
                    }
                    _ => (None, None, None, vec![]),
                }
            }
            _ => (None, None, None, vec![]),
        },
        _ => (None, None, None, vec![]),
    })
}

/// Emit `core::intrinsics::copy` (`ptr::copy`) as a `mir.memmove`.
///
/// rustc's intrinsic argument order is `(src, dst, count)` with `count` in
/// pointee elements; `mir.memmove` takes `(dst, src, count)`, so the pointers
/// are reordered here. `copy` is the overlap-safe variant, so it lowers to
/// `llvm.memmove` (the non-overlapping `copy_nonoverlapping` instead reaches
/// MIR as a `CopyNonOverlapping` statement and lowers to `mir.memcpy`). The
/// intrinsic returns `()`, so a unit value is produced for the destination
/// before branching to the target, mirroring `emit_typed_swap`.
#[allow(clippy::too_many_arguments)]
fn emit_ptr_memmove(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirConstructTupleOp, MirMemmoveOp};
    use dialect_mir::types::MirTupleType;

    if args.len() != 3 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "ptr::copy intrinsic requires (src, dst, count) operands".to_string()
            )
        );
    }

    // rustc order: (src, dst, count).
    let (src, last) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (dst, last) =
        rvalue::translate_operand(ctx, body, &args[1], value_map, block_ptr, last, loc.clone())?;
    let (count, last) =
        rvalue::translate_operand(ctx, body, &args[2], value_map, block_ptr, last, loc.clone())?;

    let (prepared_destination, prepared_last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last,
        loc.clone(),
    )?;

    // `mir.memmove` operand order is (dst, src, count). The typed builder
    // stamps the elem_type fact from dst for lowering's byte count.
    let xfer = MirMemmoveOp::build(ctx, dst, src, count)?;
    xfer.deref_mut(ctx).set_loc(loc.clone());
    match prepared_last_op {
        Some(p) => xfer.insert_after(ctx, p),
        None => xfer.insert_at_front(block_ptr, ctx),
    }

    // The intrinsic yields `()`; materialize it after the memmove and write it
    // through the destination prepared before the memory operation ran.
    let unit_ty = MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, xfer);
    let unit_val = unit_op.deref(ctx).get_result(0);

    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "ptr::copy intrinsic call without target not supported",
    )
}

/// Lower `core::intrinsics::typed_swap_nonoverlapping::<T>(x, y)`, the
/// primitive behind `core::mem::swap`/`mem::replace`, as load/load/store/store.
/// The two pointers are guaranteed non-overlapping, so the temp-free crossover
/// `t0 = *x; t1 = *y; *x = t1; *y = t0` is valid (the loaded SSA values are
/// captured before either store runs). Returns a unit result + goto target.
#[allow(clippy::too_many_arguments)]
fn emit_typed_swap(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirConstructTupleOp, MirLoadOp, MirStoreOp};
    use dialect_mir::types::{MirPtrType, MirTupleType};

    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "typed_swap_nonoverlapping requires two pointer operands".to_string()
            )
        );
    }

    let (ptr_x, last) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (ptr_y, last) =
        rvalue::translate_operand(ctx, body, &args[1], value_map, block_ptr, last, loc.clone())?;

    let elem_ty = {
        let t = ptr_x.get_type(ctx);
        let r = t.deref(ctx);
        match r.downcast_ref::<MirPtrType>() {
            Some(p) => p.pointee,
            None => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(
                        "typed_swap_nonoverlapping operand is not a pointer".to_string()
                    )
                );
            }
        }
    };

    let (prepared_destination, prepared_last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last,
        loc.clone(),
    )?;

    // t0 = *x
    let load_x = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![elem_ty],
        vec![ptr_x],
        vec![],
        0,
    );
    load_x.deref_mut(ctx).set_loc(loc.clone());
    match prepared_last_op {
        Some(p) => load_x.insert_after(ctx, p),
        None => load_x.insert_at_front(block_ptr, ctx),
    }
    let vx = load_x.deref(ctx).get_result(0);

    // t1 = *y
    let load_y = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![elem_ty],
        vec![ptr_y],
        vec![],
        0,
    );
    load_y.deref_mut(ctx).set_loc(loc.clone());
    load_y.insert_after(ctx, load_x);
    let vy = load_y.deref(ctx).get_result(0);

    // *x = t1
    let store_x = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_x, vy],
        vec![],
        0,
    );
    store_x.deref_mut(ctx).set_loc(loc.clone());
    store_x.insert_after(ctx, load_y);

    // *y = t0
    let store_y = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_y, vx],
        vec![],
        0,
    );
    store_y.deref_mut(ctx).set_loc(loc.clone());
    store_y.insert_after(ctx, store_x);

    // unit result
    let unit_ty = MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, store_y);
    let unit_val = unit_op.deref(ctx).get_result(0);

    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "typed_swap_nonoverlapping call without target not supported",
    )
}

/// Dispatches `cuda_device` intrinsic calls to their respective handlers.
///
/// Returns `Ok(Some(op))` if the call was an intrinsic, `Ok(None)` otherwise.
///
/// # Intrinsic Categories
///
/// | Category          | Examples                                          |
/// |-------------------|---------------------------------------------------|
/// | Thread Position   | `threadIdx_x`, `blockIdx_y`, `blockDim_x`         |
/// | Index Helpers     | `index_1d`, `index_2d::<S>`, `index_2d_runtime`, `index_2d_row`, `index_2d_col` |
/// | Synchronization   | `sync_threads`, `mbarrier_*`, `fence_*`           |
/// | Warp Primitives   | `shuffle_*`, `vote_*`, `lane_id`                  |
/// | WGMMA (Hopper `sm_90a`) | `wgmma_fence`, `wgmma_mma_*`, `make_smem_desc` |
/// | TMA               | `cp_async_bulk_tensor_*_g2s/s2g`, `wait_group`    |
/// | Tcgen05 (Blackwell)| `tcgen05_alloc`, `tcgen05_mma_*`, `tcgen05_ld_*` |
/// | Memory            | `SharedArray::index`, `stmatrix_*`, `cvt_*`       |
/// | Layout            | `size_of_val`, `align_of_val`                     |
/// | DisjointSlice     | `len`                                             |
#[allow(clippy::too_many_arguments)]
fn try_dispatch_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    name: &str,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    on_shared_array: bool,
    type_substs: &[rustc_public::ty::Ty],
) -> TranslationResult<Option<Ptr<Operation>>> {
    intrinsics::wgmma::reject_unsupported(name, loc.clone())?;

    if let Some(operation) = intrinsics::iket::try_dispatch(
        ctx,
        body,
        name,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc.clone(),
        type_substs,
    )? {
        return Ok(Some(operation));
    }

    if let Some(operation) = intrinsics::generated::try_dispatch_generated_intrinsic(
        ctx,
        body,
        name,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc.clone(),
    )? {
        return Ok(Some(operation));
    }

    if let Some(kind) = intrinsics::asm::InlinePtxCallKind::from_path(name) {
        return Ok(Some(intrinsics::asm::emit_inline_ptx(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            kind,
        )?));
    }

    if name == "core::intrinsics::typed_swap_nonoverlapping"
        || name == "std::intrinsics::typed_swap_nonoverlapping"
    {
        return Ok(Some(emit_typed_swap(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(intrinsic) = intrinsics::layout::RustLayoutIntrinsic::from_core_path(name) {
        return Ok(Some(intrinsics::layout::emit_rust_layout_intrinsic(
            ctx,
            body,
            intrinsic,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            type_substs,
        )?));
    }

    if let Some(intrinsic) = intrinsics::bitops::RustBitIntrinsic::from_core_path(name) {
        return Ok(Some(intrinsics::bitops::emit_rust_bit_intrinsic(
            ctx,
            body,
            intrinsic,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(intrinsic) = intrinsics::saturating::RustSaturatingIntrinsic::from_core_path(name) {
        return Ok(Some(
            intrinsics::saturating::emit_rust_saturating_intrinsic(
                ctx,
                body,
                intrinsic,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?,
        ));
    }

    if let Some(intrinsic) = intrinsics::exact_div::RustExactDivIntrinsic::from_core_path(name) {
        return Ok(Some(intrinsics::exact_div::emit_rust_exact_div_intrinsic(
            ctx,
            body,
            intrinsic,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(intrinsic) = intrinsics::bigint::RustBigIntIntrinsic::from_core_path(name) {
        return Ok(Some(intrinsics::bigint::emit_rust_bigint_intrinsic(
            ctx,
            body,
            intrinsic,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(is_f64) = intrinsics::float_math::libm_sincos_is_f64(name) {
        return Ok(Some(intrinsics::float_math::emit_sincos(
            ctx,
            body,
            is_f64,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(intrinsic) = intrinsics::float_math::RustFloatMathIntrinsic::from_core_path(name) {
        return Ok(Some(
            intrinsics::float_math::emit_rust_float_math_intrinsic(
                ctx,
                body,
                intrinsic,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?,
        ));
    }

    match name {
        // =================================================================
        // Compiler Hints
        // These intrinsics only guide optimization and do not affect semantics.
        // =================================================================
        "core::intrinsics::cold_path" | "std::intrinsics::cold_path" => {
            Ok(Some(helpers::emit_unit_noop_intrinsic(
                ctx,
                body,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                name,
            )?))
        }
        "core::intrinsics::copy" | "std::intrinsics::copy" => Ok(Some(emit_ptr_memmove(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "core::intrinsics::select_unpredictable" | "std::intrinsics::select_unpredictable" => {
            // `select_unpredictable(b, true_val, false_val)` is the
            // compiler-hint form of `if b { true_val } else { false_val }`,
            // a branchless ternary libcore emits from sorting/`Ord` helpers.
            // Emit a placeholder call carrying the three operands; mir-lower
            // turns it into an LLVM `select`. `bool` lowers to `i1`, exactly
            // what the select condition needs.
            let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;
            Ok(Some(helpers::emit_function_call(
                ctx,
                body,
                dialect_mir::rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE,
                args,
                destination,
                return_type,
                None,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "core::intrinsics::volatile_load" | "std::intrinsics::volatile_load" => {
            Ok(Some(intrinsics::memory::emit_volatile_load(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "core::intrinsics::volatile_store" | "std::intrinsics::volatile_store" => {
            Ok(Some(intrinsics::memory::emit_volatile_store(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }

        "core::intrinsics::arith_offset" | "std::intrinsics::arith_offset" => {
            Ok(Some(intrinsics::memory::emit_arith_offset(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }

        "core::intrinsics::ptr_offset_from" | "std::intrinsics::ptr_offset_from" => {
            Ok(Some(intrinsics::memory::emit_ptr_offset_from(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }

        "core::intrinsics::ptr_offset_from_unsigned"
        | "std::intrinsics::ptr_offset_from_unsigned" => {
            Ok(Some(intrinsics::memory::emit_ptr_offset_from_unsigned(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }

        // =================================================================
        // Thread Index Helpers (from intrinsics::indexing)
        //
        // The index helpers are normal Rust functions over generated leaf
        // intrinsics. Their bodies use the normal function-call path.
        // =================================================================
        "cuda_device::thread::__internal::index_1d"
        | "cuda_device::index_1d"
        | "cuda_device::thread::index_1d"
        | "cuda_device::index_2d_row"
        | "cuda_device::thread::index_2d_row"
        | "cuda_device::index_2d_col"
        | "cuda_device::thread::index_2d_col"
        | "cuda_device::thread::__internal::index_2d"
        | "cuda_device::thread::__internal::index_2d_runtime"
        | "cuda_device::index_2d"
        | "cuda_device::thread::index_2d"
        | "cuda_device::index_2d_runtime"
        | "cuda_device::thread::index_2d_runtime"
        | "cuda_device::thread::__internal::warp_index"
        | "cuda_device::warp_index"
        | "cuda_device::thread::warp_index" => Ok(None),

        // =================================================================
        // Debug & Profiling (from intrinsics::debug)
        // =================================================================
        "cuda_device::debug::__gpu_assertfail" => Ok(Some(intrinsics::debug::emit_assertfail(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::debug::__gpu_vprintf" => Ok(Some(intrinsics::debug::emit_vprintf(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),

        // =================================================================
        // Compile-time cluster configuration
        // =================================================================
        "cuda_device::cluster::__cluster_config" => {
            // Compile-time cluster configuration marker from #[cluster(x,y,z)] attribute.
            // The cluster dimensions are extracted in body.rs during MIR scanning.
            // This call generates no runtime code - just emit a goto to the target block.
            //
            // We need a prev_op to insert after. If none exists, create a dummy constant.
            let actual_prev_op = match prev_op {
                Some(op) => op,
                None => {
                    // Create a dummy i1 constant to use as insertion point
                    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
                    let dummy = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![bool_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    dummy.deref_mut(ctx).set_loc(loc.clone());
                    let const_op = MirConstantOp::new(dummy);
                    use pliron::builtin::attributes::IntegerAttr;
                    use pliron::utils::apint::APInt;
                    use std::num::NonZeroUsize;
                    let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
                    const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
                    let dummy = const_op.get_operation();
                    dummy.insert_at_front(block_ptr, ctx);
                    dummy
                }
            };
            Ok(Some(helpers::emit_goto(
                ctx,
                target.expect("__cluster_config must have target"),
                actual_prev_op,
                block_map,
                loc,
            )))
        }
        "cuda_device::__launch_bounds_config"
        | "cuda_device::thread::__launch_bounds_config"
        | "cuda_device::__launch_contract_config"
        | "cuda_device::thread::__launch_contract_config"
        | "cuda_device::__launch_contract_block_config"
        | "cuda_device::thread::__launch_contract_block_config"
        | "cuda_device::__grid_constant_config"
        | "cuda_device::thread::__grid_constant_config"
        | "cuda_device::__unchecked_indexing_config"
        | "cuda_device::thread::__unchecked_indexing_config" => {
            let expected_marker = match name {
                "cuda_device::__launch_bounds_config"
                | "cuda_device::thread::__launch_bounds_config" => "__launch_bounds_config",
                "cuda_device::__launch_contract_config"
                | "cuda_device::thread::__launch_contract_config" => "__launch_contract_config",
                "cuda_device::__launch_contract_block_config"
                | "cuda_device::thread::__launch_contract_block_config" => {
                    "__launch_contract_block_config"
                }
                "cuda_device::__grid_constant_config"
                | "cuda_device::thread::__grid_constant_config" => "__grid_constant_config",
                "cuda_device::__unchecked_indexing_config"
                | "cuda_device::thread::__unchecked_indexing_config" => {
                    "__unchecked_indexing_config"
                }
                _ => unreachable!("launch metadata arm matched an unknown marker"),
            };
            if !is_cuda_device_const_marker(func, expected_marker) {
                return Ok(None);
            }
            // Compile-time metadata marker. Launch bounds and the
            // unchecked-indexing flag are extracted in body.rs; the contract
            // marker selects the kernel's typed launch context during macro
            // expansion. No marker emits runtime code. Emit only the
            // control-flow edge to the call's target.
            //
            // We need a prev_op to insert after. If none exists, create a dummy constant.
            let actual_prev_op = match prev_op {
                Some(op) => op,
                None => {
                    // Create a dummy i1 constant to use as insertion point
                    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
                    let dummy = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![bool_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    dummy.deref_mut(ctx).set_loc(loc.clone());
                    let const_op = MirConstantOp::new(dummy);
                    use pliron::builtin::attributes::IntegerAttr;
                    use pliron::utils::apint::APInt;
                    use std::num::NonZeroUsize;
                    let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
                    const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
                    let dummy = const_op.get_operation();
                    dummy.insert_at_front(block_ptr, ctx);
                    dummy
                }
            };
            Ok(Some(helpers::emit_goto(
                ctx,
                target.expect("launch metadata marker must have target"),
                actual_prev_op,
                block_map,
                loc,
            )))
        }
        "cuda_device::shared::__dynamic_shared_alignment" => {
            // Zero-cost marker injected by #[launch_contract]. body.rs records
            // the const alignment on the kernel; no runtime call survives.
            let actual_prev_op = match prev_op {
                Some(op) => op,
                None => {
                    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
                    let dummy = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![bool_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    dummy.deref_mut(ctx).set_loc(loc.clone());
                    let const_op = MirConstantOp::new(dummy);
                    use pliron::builtin::attributes::IntegerAttr;
                    use pliron::utils::apint::APInt;
                    use std::num::NonZeroUsize;
                    let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
                    const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
                    let dummy = const_op.get_operation();
                    dummy.insert_at_front(block_ptr, ctx);
                    dummy
                }
            };
            Ok(Some(helpers::emit_goto(
                ctx,
                target.expect("__dynamic_shared_alignment must have target"),
                actual_prev_op,
                block_map,
                loc,
            )))
        }
        // =================================================================
        // WGMMA (from intrinsics::wgmma)
        // =================================================================
        "cuda_device::wgmma::make_smem_desc" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_make_smem_desc(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_mma_m64n64k16_f32_bf16(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::wgmma::wgmma_mma_m64n128k16_f32_bf16" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_mma_m64n128k16_f32_bf16(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_f16" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_mma_m64n64k16_f32_f16(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::wgmma::wgmma_mma_m64n64k8_f32_tf32" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_mma_m64n64k8_f32_tf32(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }

        // =================================================================
        // DisjointSlice and SharedArray operations
        // =================================================================
        "cuda_device::DisjointSlice::len" => Ok(Some(intrinsics::indexing::emit_len(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),

        // Trait method - check the substs' Self type for SharedArray
        // Note: Index/IndexMut can appear as either std::ops or core::ops
        "std::ops::IndexMut::index_mut" | "core::ops::IndexMut::index_mut" if on_shared_array => {
            Ok(Some(intrinsics::memory::emit_shared_array_index(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                true,
            )?))
        }
        "std::ops::Index::index" | "core::ops::Index::index" if on_shared_array => {
            Ok(Some(intrinsics::memory::emit_shared_array_index(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                false,
            )?))
        }

        // DisjointSlice methods using prefix matching (covers generic
        // instantiations whose full path includes type parameters).
        // Note: `get_mut` and `get_unchecked_mut` are `#[inline]` in
        // cuda-device and are always inlined by rustc before MIR reaches the
        // translator, so element access never surfaces here as a call. Only
        // `len` needs a dispatch arm.
        path if path.starts_with("cuda_device::DisjointSlice::") => {
            if let Some(method) = path.rsplit("::").next() {
                match method {
                    "len" => Ok(Some(intrinsics::indexing::emit_len(
                        ctx,
                        body,
                        args,
                        destination,
                        target,
                        block_ptr,
                        prev_op,
                        value_map,
                        block_map,
                        loc,
                    )?)),
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }

        // Explicit generic-to-shared address conversion for hardware SMEM
        // descriptors (the CUDA C++ `__cvta_generic_to_shared_offset` analog).
        "cuda_device::shared::cvta_generic_to_shared_offset" => Ok(Some(
            intrinsics::memory::emit_cvta_generic_to_shared_offset(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                64,
            )?,
        )),
        "cuda_device::shared::cvta_generic_to_shared_u32" => Ok(Some(
            intrinsics::memory::emit_cvta_generic_to_shared_offset(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                32,
            )?,
        )),

        // Public SharedArray pointer conversions all narrow the shared-memory
        // base to a generic pointer. Recognition is shared with destination
        // address-space classification in translator::values.
        path if super::shared_array_pointer_method(path).is_some() => {
            Ok(Some(intrinsics::memory::emit_shared_array_as_ptr(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }

        // Note: DynamicSharedArray operations are handled specially before this function
        // to extract the ALIGN const generic parameter. See the handling above
        // try_dispatch_intrinsic() call.

        // =================================================================
        // Atomic Operations (all cuda_device::atomic::* types and scopes)
        // =================================================================
        path if intrinsics::atomic::is_atomic_path(path) => intrinsics::atomic::dispatch(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            path,
        ),

        // Not an intrinsic
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_signature_rejects_c_variadic_before_claiming_an_exact_abi() {
        assert!(ensure_foreign_callee_is_not_variadic("fixed", false).is_ok());
        assert!(ensure_foreign_callee_is_not_variadic("variadic", true).is_err());
    }

    #[test]
    fn generated_kernel_setup_calls_have_no_user_debug_location() {
        let user_location = Location::Named {
            name: "kernel attribute".into(),
            child_loc: Box::new(Location::Unknown),
        };

        for constructor in KERNEL_SCOPE_CONSTRUCTORS {
            assert_eq!(
                call_debug_location(Some(constructor), user_location.clone()),
                llvm_export::artificial_debug_location()
            );
        }
        assert_eq!(
            call_debug_location(
                Some("cuda_device::__launch_bounds_config"),
                user_location.clone()
            ),
            llvm_export::artificial_debug_location(),
            "zero-cost launch metadata must not leave an attribute line behind"
        );
        assert_eq!(
            call_debug_location(
                Some("cuda_device::thread::__internal::index_1d"),
                user_location.clone()
            ),
            user_location,
            "user-written indexing calls keep their source locations"
        );
    }

    /// A `SwitchInt` arm keeps its whole value at 128 bits, and narrower
    /// discriminants truncate exactly as the previous `u64` construction did.
    /// Every width at or below 64 is the behaviour this shares with the enum
    /// tags that dominate the switches in the tree, so it is pinned here
    /// alongside the widths that used to be rejected.
    #[test]
    fn switch_arm_constants_are_built_at_the_discriminant_width() {
        use pliron::utils::apint::APInt;
        use std::num::NonZeroUsize;

        let ctx = &mut Context::new();

        for (width, value) in [
            (8usize, 0xffu128),
            (16, 0xbeef),
            (32, 0xdead_beef),
            (64, u128::from(u64::MAX)),
        ] {
            let attr = switch_arm_constant(ctx, value, width, Signedness::Unsigned);
            let expected = APInt::from_u64(value as u64, NonZeroUsize::new(width).unwrap());
            assert_eq!(
                attr.value(),
                expected,
                "width {width} must match the previous u64 construction"
            );
        }

        // Above 64 bits the u64 construction could not represent the value at
        // all, which is what used to be rejected.
        let wide = (1u128 << 100) | 1;
        let attr = switch_arm_constant(ctx, wide, 128, Signedness::Unsigned);
        assert_eq!(attr.value().to_u128(), wide);
        assert_eq!(attr.value().bw(), 128);
    }

    /// The importer traps exactly the calls the codegen collector declines to
    /// collect. If the two predicates drift apart, one side emits a call to a
    /// symbol the other never defines.
    #[test]
    fn panic_entry_paths_agree_with_the_collector_predicate() {
        assert!(is_panic_entry_path("core::panicking::panic"));
        assert!(is_panic_entry_path("core::panicking::panic_fmt"));
        assert!(is_panic_entry_path("core::panicking::panic_bounds_check"));
        assert!(is_panic_entry_path("std::rt::panic_fmt"));

        assert!(!is_panic_entry_path(
            "core::slice::<impl [T]>::split_at_mut"
        ));
        assert!(!is_panic_entry_path("my_crate::panic_helper"));
        assert!(!is_panic_entry_path("my_crate::panicking_but_mine"));
    }

    /// A dropped panic call must leave `nvvm.trap` (carrying its ABI marker,
    /// or the target-requirements verifier rejects it) followed by
    /// `mir.unreachable`, and nothing else: the message-building statements
    /// the call would have consumed are never translated.
    #[test]
    fn dropped_panic_block_lowers_to_marked_trap_then_unreachable() {
        use dialect_mir::ops::MirUnreachableOp;
        use pliron::builtin::attributes::StringAttr;
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let block_ptr = BasicBlock::new(&mut ctx, None, vec![]);
        emit_trap_unreachable_after(&mut ctx, block_ptr, None, Location::Unknown);

        let ops: Vec<Ptr<Operation>> = block_ptr.deref(&ctx).iter(&ctx).collect();
        assert_eq!(
            ops.len(),
            2,
            "a dropped panic block holds trap + unreachable"
        );
        let trap = Operation::get_op::<dialect_nvvm::ops::TrapOp>(ops[0], &ctx)
            .expect("first op is `nvvm.trap`");
        assert!(
            Operation::get_op::<MirUnreachableOp>(ops[1], &ctx).is_some(),
            "second op is `mir.unreachable`"
        );

        let marker_key =
            Identifier::try_from(cuda_oxide_codegen::__private::GENERATED_INTRINSIC_MARKER_ATTR)
                .unwrap();
        let trap_ref = trap.get_operation().deref(&ctx);
        let marker: &StringAttr = trap_ref
            .attributes
            .get(&marker_key)
            .expect("`nvvm.trap` carries its generated-intrinsic ABI marker");
        assert_eq!(String::from(marker.clone()), "v1:i0295");
    }
}
