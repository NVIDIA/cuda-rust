/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Drop glue analysis and emission for device code.
//!
//! This module provides two capabilities:
//!
//! 1. **No-op analysis** ([`drop_glue_is_noop`]): a conservative static proof
//!    that dropping a value does nothing observable, allowing the `Drop`
//!    terminator to be lowered as a plain branch (fast path).
//!
//! 2. **Drop glue emission** (`emit_drop_glue`): when the no-op proof fails,
//!    emits a device-side call to the monomorphized `drop_in_place::<T>`
//!    function, which the collector has already gathered for translation.
//!
//! # No-op analysis
//!
//! rustc keeps a MIR `Drop` terminator for any type whose destructor
//! *might* do something. That includes types whose destructor turns out
//! to do nothing once generics are filled in. The motivating case is
//! `for x in arr` over a by-value array `[u32; 4]`: the loop iterates a
//! `core::array::IntoIter<u32, 4>`, whose destructor drops the
//! not-yet-yielded elements by calling `drop_in_place` on the "alive"
//! sub-slice of its element buffer. For `T = u32` the elements have no
//! destructor, so `drop_in_place::<[u32]>` resolves to rustc's empty
//! drop shim and the whole chain does nothing at runtime: it only
//! shuffles index values between local variables on the way to the
//! empty shim. The same holds for any element type without drop glue,
//! including plain `Copy` structs.
//!
//! What counts as "nothing observable":
//!
//! - Statements that exist only for analysis or storage bookkeeping
//!   (`StorageLive`, `Nop`, coverage markers, ...).
//! - Writes that stay inside the glue's own stack frame, i.e. the
//!   destination is a local variable and not a pointer dereference.
//!   Such writes die when the function returns.
//! - Calls and nested drops whose target passes this same check.
//!
//! Anything else (writes through pointers, asserts, inline asm, calls we
//! cannot resolve or whose body we cannot see) fails the proof.
//!
//! The walk follows constant `switchInt` discriminants, so branches that
//! the compiler has already folded shut (for example checks behind the
//! `UbChecks` flag, which is off for device builds) do not have to be
//! proven; only code that can actually execute does.
//!
//! # Drop glue emission
//!
//! When the no-op proof fails, the type has an effectful destructor that
//! must actually run. `emit_drop_glue` resolves the monomorphized
//! `drop_in_place::<T>` instance, obtains its mangled symbol name (which
//! the collector uses as the export name for the translated device
//! function), and emits a `mir.call` passing the dropped place's address
//! as the sole `*mut T` argument.
//!
//! # Device-side destructor safety notes
//!
//! - Device-side destructors run synchronously; there is no async cleanup.
//! - Panicking in a destructor on device is not recoverable (panic=abort).
//! - Drop order follows Rust's standard LIFO semantics.
//! - Types with `Drop` must ensure their drop implementation is
//!   device-compatible (no host-only syscalls, allocations, etc.).

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::{ValueMap, establish_declared_pointer_type};
use crate::translator::{rvalue, types};
use dialect_mir::attributes::MirPointerKindAuthorityAttr;
use dialect_mir::ops::MirCallOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::input_error;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use rustc_public::mir;
use rustc_public::mir::mono::Instance;
use rustc_public::ty::{RigidTy, Ty, TyKind};

/// Cap on the call-chain depth the proof is willing to follow. Real
/// no-op drop glue is shallow (the `IntoIter` case above is two levels:
/// the `drop_in_place` shim, then `Drop::drop`). The cap only exists so
/// pathological inputs cannot make the compiler crawl a huge call graph.
const MAX_PROOF_DEPTH: usize = 16;

/// Returns true when dropping a value of `dropped_ty` is provably a
/// no-op, meaning the `Drop` terminator can be lowered to a plain
/// branch without skipping any observable destructor work.
///
/// `dropped_ty` must be fully monomorphized (no generic parameters
/// left), which holds for every body the importer translates.
///
/// This is a thin wrapper over [`drop_instance_is_noop`], the single
/// no-op predicate shared by every site that must agree on whether a
/// drop is observable.
pub fn drop_glue_is_noop(dropped_ty: Ty) -> bool {
    drop_instance_is_noop(&Instance::resolve_drop_in_place(dropped_ty))
}

/// Returns true when the given `drop_in_place` instance is provably a
/// no-op, by walking the monomorphized shim MIR and proving every
/// reachable path does nothing observable (body-level proof).
///
/// This is THE no-op predicate for drop glue. Three decisions must stay
/// in exact lockstep, and all three consult this function:
///
/// 1. Whether `translate_drop` emits a `drop_in_place` call or a plain
///    branch (via [`drop_glue_is_noop`]).
/// 2. Whether the collector gathers the shim for translation
///    (`rustc-codegen-cuda`'s `collector::process_drop_place`).
/// 3. Whether device codegen keeps the shim in the translation set
///    (`rustc-codegen-cuda`'s `device_codegen` no-op filter).
///
/// If these drift, emitted calls reference uncollected symbols (or dead
/// shim bodies get translated and fail on unsupported constructs). Any
/// widening of this proof must be sound for ALL callers: a `Drop` impl
/// whose fields are trivially droppable can still be observable (e.g. a
/// raw-pointer RAII guard whose `drop` writes through the pointer), so
/// type-level "all fields drop-free" reasoning is NOT a valid fallback.
///
/// Must be called inside a `rustc_internal::run()` context so that
/// stable MIR queries are available.
pub fn drop_instance_is_noop(instance: &Instance) -> bool {
    instance_is_noop(instance, &mut Vec::new())
}

/// Emits a device-side call to `drop_in_place::<T>` for the given place.
///
/// This is the fallback path when [`drop_glue_is_noop`] returns false,
/// meaning the type has an effectful destructor that must actually execute.
/// The function:
///
/// 1. Resolves the monomorphized `drop_in_place::<T>` instance
/// 2. Obtains its mangled name (matching what the collector exported)
/// 3. Legalises the name through the same `Legaliser` used for all call sites
/// 4. Computes the dropped place's internal in-memory address
/// 5. Establishes the callee's declared pointer type (`*mut T`, or `&mut T`
///    for the nightly-2026-08-28 `core::ptr::drop_glue` shims) at the Rust
///    ABI boundary
/// 6. Emits a `mir.call` to the drop function with that pointer as argument
/// 7. Branches to the successor block
///
/// The caller (`translate_drop`) has already checked that the no-op fast path
/// does not apply.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_drop_glue(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    dropped_ty: Ty,
    target: mir::BasicBlockIdx,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    legaliser: &mut Legaliser,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Resolve the monomorphized drop_in_place::<T> instance.
    let drop_instance = Instance::resolve_drop_in_place(dropped_ty);
    let Some(drop_body) = drop_instance.body() else {
        return Err(input_error!(
            loc,
            TranslationErr::unsupported(
                "drop glue: resolved drop_in_place instance has no MIR body".to_string()
            )
        ));
    };
    let Some(declared_parameter) = drop_body.arg_locals().first() else {
        return Err(input_error!(
            loc,
            TranslationErr::unsupported(
                "drop glue: resolved drop_in_place instance has no pointer parameter".to_string()
            )
        ));
    };
    // Historically the drop shim was `drop_in_place::<T>(*mut T)`. Since
    // nightly-2026-08-28 rustc builds it from `core::ptr::drop_glue`
    // (`#[lang = "drop_glue"]`, rustc_mir_transform::shim::build_drop_shim),
    // whose declared parameter is `&mut T`. Both spellings are the same
    // single-pointer ABI; accept either and let the translated parameter
    // type drive the ABI-boundary pointer establishment below.
    if !matches!(
        declared_parameter.ty.kind(),
        TyKind::RigidTy(
            RigidTy::RawPtr(_, mir::Mutability::Mut) | RigidTy::Ref(_, _, mir::Mutability::Mut)
        )
    ) {
        return Err(input_error!(
            loc,
            TranslationErr::unsupported(
                "drop glue: drop shim's parameter is not the expected *mut/&mut pointer"
                    .to_string()
            )
        ));
    }
    let declared_parameter_type = types::translate_type(ctx, &declared_parameter.ty)?;

    // The mangled name is what the collector uses as the export/symbol name
    // for the translated device function.
    let mangled = drop_instance.mangled_name();
    let callee_name = legaliser.legalise(&mangled);

    // drop_in_place::<T> takes a single argument: *mut T (a mutable pointer
    // to the value being dropped). We obtain this by taking the address of
    // the dropped place.
    //
    // For a simple local `_N`, this is just the alloca slot pointer.
    // For a projected place like `_N.field`, we compute the field address.
    let (place_ptr, last_op) =
        compute_drop_place_address(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;
    let (place_ptr, last_op) = establish_declared_pointer_type(
        ctx,
        place_ptr,
        declared_parameter_type,
        block_ptr,
        last_op,
        MirPointerKindAuthorityAttr::AbiBoundary,
    );
    if place_ptr.get_type(ctx) != declared_parameter_type {
        return Err(input_error!(
            loc,
            TranslationErr::unsupported(
                "drop glue: dropped-place address does not match the drop shim's pointer parameter"
                    .to_string()
            )
        ));
    }

    // The return type of drop_in_place is `()` (unit / void).
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);

    // Emit the call: `drop_in_place::<T>(ptr)`
    let call_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![place_ptr],
        vec![],
        0,
    );
    call_op.deref_mut(ctx).set_loc(loc.clone());

    let callee_attr = pliron::builtin::attributes::StringAttr::new(callee_name.to_string());
    call_op.deref_mut(ctx).attributes.set(
        pliron::identifier::Identifier::try_from("callee").unwrap(),
        callee_attr,
    );

    if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
    } else {
        call_op.insert_at_front(block_ptr, ctx);
    }

    // Branch to the successor block.
    let target_block = block_map[target];
    let goto_op = Operation::new(
        ctx,
        dialect_mir::ops::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target_block],
        0,
    );
    goto_op.deref_mut(ctx).set_loc(loc);
    goto_op.insert_after(ctx, call_op);

    Ok(goto_op)
}

/// Computes the in-memory address of the place being dropped.
///
/// `drop_in_place::<T>` expects a `*mut T` argument. For a bare local `_N`
/// this is just the local's alloca slot (which is already a pointer). For a
/// projected place (`_N.field`, `(*_N)`, etc.) we walk the projection chain
/// via `translate_place_address`.
fn compute_drop_place_address(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(pliron::value::Value, Option<Ptr<Operation>>)> {
    // Try the full projection-aware address computation first.
    if let Some((addr, last_op)) = rvalue::translate_place_address(
        ctx,
        body,
        value_map,
        place,
        /* is_mutable */ true,
        block_ptr,
        prev_op,
        loc.clone(),
    )? {
        return Ok((addr, last_op));
    }

    // Fallback: for a bare local with no projection, the alloca slot IS
    // the pointer we need.
    if place.projection.is_empty()
        && let Some(slot) = value_map.get_slot(place.local)
    {
        return Ok((slot, prev_op));
    }

    Err(input_error!(
        loc,
        TranslationErr::unsupported(format!(
            "drop glue: cannot compute in-memory address for dropped place {:?}",
            place
        ))
    ))
}

// ============================================================================
// No-op analysis (unchanged from original)
// ============================================================================

/// Proves that calling `instance` does nothing observable.
///
/// `in_progress` holds the mangled names of instances currently being
/// proven further up the call stack. If we meet one of them again we
/// treat the cycle as harmless: a cycle cannot *introduce* an
/// observable effect, every effect would already have failed the proof
/// on some statement or terminator along the way.
fn instance_is_noop(instance: &Instance, in_progress: &mut Vec<String>) -> bool {
    // Fast path: a type with no drop glue at all resolves to an "empty"
    // drop shim that exists only to fill vtable slots. rustc_public
    // exposes that directly.
    if instance.is_empty_shim() {
        return true;
    }

    if in_progress.len() >= MAX_PROOF_DEPTH {
        return false;
    }

    let name = instance.mangled_name();
    if in_progress.contains(&name) {
        return true;
    }

    // No body means we cannot see what the call does (an intrinsic, a
    // foreign function, ...). The proof must fail.
    let Some(body) = instance.body() else {
        return false;
    };

    in_progress.push(name);
    let result = body_is_noop(&body, in_progress);
    in_progress.pop();
    result
}

/// Walks every block of `body` reachable from the entry block and
/// checks that nothing observable happens on the way to `return`.
fn body_is_noop(body: &mir::Body, in_progress: &mut Vec<String>) -> bool {
    let mut visited = vec![false; body.blocks.len()];
    let mut worklist: Vec<mir::BasicBlockIdx> = vec![0];

    while let Some(idx) = worklist.pop() {
        if std::mem::replace(&mut visited[idx], true) {
            continue;
        }
        let block = &body.blocks[idx];

        for stmt in &block.statements {
            if !statement_is_noop(&stmt.kind) {
                return false;
            }
        }

        match &block.terminator.kind {
            // Reaching `return` with only no-op work behind us is the
            // success case. `unreachable` cannot execute in a valid
            // program, so it cannot contribute an effect either.
            mir::TerminatorKind::Return | mir::TerminatorKind::Unreachable => {}

            mir::TerminatorKind::Goto { target } => worklist.push(*target),

            mir::TerminatorKind::SwitchInt { discr, targets } => {
                match const_operand_bits(discr) {
                    // Known discriminant: only the matching branch can
                    // run, so only that branch needs to be a no-op.
                    // This is what skips the dead "really drop the
                    // elements" branch in `IntoIter`'s destructor.
                    Some(value) => {
                        let target = targets
                            .branches()
                            .find(|(branch_value, _)| *branch_value == value)
                            .map(|(_, target)| target)
                            .unwrap_or_else(|| targets.otherwise());
                        worklist.push(target);
                    }
                    // Unknown discriminant: every branch must be a
                    // no-op.
                    None => worklist.extend(targets.all_targets()),
                }
            }

            // A nested drop is fine when the dropped value's own glue
            // passes this same proof.
            mir::TerminatorKind::Drop { place, target, .. } => {
                let Ok(place_ty) = place.ty(body.locals()) else {
                    return false;
                };
                let nested = Instance::resolve_drop_in_place(place_ty);
                if !instance_is_noop(&nested, in_progress) {
                    return false;
                }
                worklist.push(*target);
            }

            // A call is fine when we can resolve exactly which function
            // runs and that function passes this same proof. The
            // `drop_in_place` shim for a type with an `impl Drop` is a
            // single such call to `<T as Drop>::drop`.
            mir::TerminatorKind::Call {
                func,
                destination,
                target: Some(target),
                ..
            } => {
                if place_writes_through_pointer(destination) {
                    return false;
                }
                let Ok(func_ty) = func.ty(body.locals()) else {
                    return false;
                };
                let TyKind::RigidTy(RigidTy::FnDef(def, args)) = func_ty.kind() else {
                    // A function pointer or other indirect callee: we
                    // do not know what runs.
                    return false;
                };
                let Ok(callee) = Instance::resolve(def, &args) else {
                    return false;
                };
                if !instance_is_noop(&callee, in_progress) {
                    return false;
                }
                worklist.push(*target);
            }

            // Everything else (asserts, diverging calls, inline asm,
            // resume/abort) either has an effect or might not return.
            // Asserts fail the proof deliberately: following only the
            // success edge would silently delete a would-be device trap.
            // A failed proof now emits the real drop_in_place call, so
            // there is no pressure to widen the proof here.
            _ => return false,
        }
    }

    true
}

/// Returns true when executing this statement at runtime can have no
/// effect observable outside the enclosing function.
fn statement_is_noop(kind: &mir::StatementKind) -> bool {
    use mir::StatementKind;
    match kind {
        // Storage markers, analysis-only annotations, and literal
        // no-ops. None of these produce machine code with effects.
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Nop
        | StatementKind::ConstEvalCounter
        | StatementKind::Coverage(_)
        | StatementKind::PlaceMention(_)
        | StatementKind::FakeRead(..)
        | StatementKind::AscribeUserType { .. } => true,

        // MIR rvalues are pure (they compute a value; only the
        // assignment's destination writes anything), so an assignment
        // is harmless as long as the destination stays inside this
        // function's own stack frame: a local, or a field of a local,
        // but never a pointer dereference.
        StatementKind::Assign(place, _) | StatementKind::SetDiscriminant { place, .. } => {
            !place_writes_through_pointer(place)
        }

        // `assume` is a no-op at runtime (it only carries information
        // for the optimiser). `copy_nonoverlapping` writes through a
        // raw pointer and is never a no-op.
        StatementKind::Intrinsic(mir::NonDivergingIntrinsic::Assume(_)) => true,
        StatementKind::Intrinsic(mir::NonDivergingIntrinsic::CopyNonOverlapping(_)) => false,
    }
}

/// Returns true when writing to `place` writes through a pointer, i.e.
/// the write can land in memory that outlives the function. Writes to
/// plain locals (including fields of locals) vanish when the function
/// returns and are therefore unobservable.
fn place_writes_through_pointer(place: &mir::Place) -> bool {
    place
        .projection
        .iter()
        .any(|elem| matches!(elem, mir::ProjectionElem::Deref))
}

/// Extracts the raw bit value of a constant operand, e.g. the `false`
/// in `switchInt(false)`. Returns `None` for anything that is not a
/// fully evaluated scalar constant.
fn const_operand_bits(operand: &mir::Operand) -> Option<u128> {
    match operand {
        mir::Operand::Constant(const_op) => {
            let rustc_public::ty::ConstantKind::Allocated(alloc) = const_op.const_.kind() else {
                return None;
            };
            alloc.read_uint().ok()
        }
        // Keep proof reachability synchronized with collection and operand
        // translation through the device-wide runtime-check policy.
        mir::Operand::RuntimeChecks(_) => Some(u128::from(crate::DEVICE_RUNTIME_CHECKS_VALUE)),
        _ => None,
    }
}
