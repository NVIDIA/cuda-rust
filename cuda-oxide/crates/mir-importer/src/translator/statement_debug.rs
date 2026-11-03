/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Restores rustc's statement-level debug assignments in full-debug builds.
//!
//! `ReferencePropagation` can leave this shape:
//!
//! ```text
//! ptr = &slice[i]      -> removed from executable MIR
//! DBG(ptr = &slice[i]) -> kept by rustc
//! ```
//!
//! Stable MIR omits the `DBG` record, so the codegen bridge carries it in a
//! sidecar. Supported runtime-index references use the destination local's
//! existing debugger stack home:
//!
//! ```text
//! &slice[i] -> ptr's existing debug stack slot -> CUDA-GDB
//! &array[i] -> ptr's existing debug stack slot -> CUDA-GDB
//! ```
//!
//! A stack slot is used because `ptxas` drops or misrepresents the affected
//! register-only and multi-value pointer locations. This adds code only in
//! full-debug mode. Preflight accepts only non-argument, non-return locals used
//! solely for debug info, with initialized inputs and supported `AssignRef`
//! events. Otherwise it removes that local's debug metadata and emits no address
//! calculation or store.

use super::{rvalue, types, values};
use crate::error::{TranslationErr, TranslationResult};
use crate::pipeline::{StatementDebugInfo, StatementDebugInfoMap};
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirPointerKindAuthorityAttr;
use dialect_mir::ops::suppress_debug_local_from_slot;
use dialect_mir::types::MirPtrType;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err_noloc;
use pliron::location::Location;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use rustc_public::mir::visit::MirVisitor;
use rustc_public::mir::{self, StatementKind};
use rustc_public::ty::{RigidTy, TyKind, UintTy};

#[derive(Clone, Copy)]
struct SpillState {
    seen: bool,
    seen_emitted: bool,
    representable: bool,
}

impl Default for SpillState {
    fn default() -> Self {
        Self {
            seen: false,
            seen_emitted: false,
            representable: true,
        }
    }
}

#[derive(Clone, Copy)]
struct AssignRefSources {
    base: mir::Local,
    index: mir::Local,
}

/// Prove which statement-debug destinations can safely use their existing
/// backing slot as a debugger-visible home.
///
/// This runs after entry allocas have been created but before any reachable
/// block is translated, so a rejected destination can have all debug metadata
/// removed without leaving partial executable operations behind.
pub(crate) fn prepare_statement_debug_spills(
    ctx: &mut Context,
    body: &mir::Body,
    debug_info: &StatementDebugInfoMap,
    reachable: &std::collections::BTreeSet<usize>,
    rustc_mono_successors: &[Vec<usize>],
    value_map: &mut ValueMap,
) {
    let mut states = vec![SpillState::default(); body.locals().len()];
    let block_entry_initialization =
        definitely_initialized_at_block_entries(body, reachable, rustc_mono_successors);

    for (block_index, (block, mir_block)) in debug_info.blocks.iter().zip(&body.blocks).enumerate()
    {
        let is_reachable = reachable.contains(&block_index);
        let is_emitted =
            is_reachable && !super::terminator::is_dropped_panic_call(&mir_block.terminator);
        let mut initialized = block_entry_initialization[block_index].clone();

        let mut record_events = |infos: &[StatementDebugInfo], initialized: &[bool]| {
            for info in infos {
                let destination = match info {
                    StatementDebugInfo::AssignRef { destination, .. }
                    | StatementDebugInfo::InvalidAssign { destination } => *destination,
                };
                let Some(state) = states.get_mut(destination) else {
                    continue;
                };
                state.seen = true;

                // Unreachable blocks have no emitted PC, so their events do not
                // affect a live location history. Tracking `seen` still suppresses
                // a destination whose events are all dead.
                if !is_reachable {
                    continue;
                }

                state.seen_emitted |= is_emitted;
                if !is_emitted {
                    // `translate_block` drops statements in reachable panic-call
                    // blocks. Reject their events rather than promise a store.
                    state.representable = false;
                    continue;
                }

                let representable = match info {
                    StatementDebugInfo::AssignRef { destination, place } => {
                        let sources =
                            assign_ref_is_representable(ctx, body, value_map, *destination, place);
                        let initialized_at_boundary = sources.is_some_and(|sources| {
                            initialized.get(sources.base).copied().unwrap_or(false)
                                && initialized.get(sources.index).copied().unwrap_or(false)
                        });
                        sources.is_some() && initialized_at_boundary
                    }
                    StatementDebugInfo::InvalidAssign { .. } => false,
                };
                state.representable &= representable;
            }
        };

        for (statement_index, statement) in mir_block.statements.iter().enumerate() {
            record_events(&block.before_statements[statement_index], &initialized);
            apply_statement_initialization(
                body,
                block_index,
                statement_index,
                statement,
                &mut initialized,
            );
        }
        record_events(&block.before_terminator, &initialized);
    }

    let executable_uses = executable_local_uses(body, reachable);
    let arg_count = body.arg_locals().len();
    for (destination, state) in states.into_iter().enumerate() {
        if !state.seen {
            continue;
        }
        let local = mir::Local::from(destination);
        let is_debug_only_ghost =
            destination > arg_count && !executable_uses.get(destination).copied().unwrap_or(true);
        let has_slot = value_map.get_slot(local).is_some();
        let enabled = state.seen_emitted && state.representable && is_debug_only_ghost && has_slot;

        value_map.set_statement_debug_spill(local, enabled);
        if !enabled && let Some(slot) = value_map.get_slot(local) {
            suppress_debug_local_from_slot(ctx, slot);
        }
    }
}

/// Emit all eligible debug-home stores attached to one MIR statement boundary.
///
/// Exactly one `mir.store` is emitted for every enabled `AssignRef` event.
/// Rejected destinations were disabled (and their metadata suppressed) during
/// preflight, before this function can emit address-calculation operations.
pub(crate) fn translate_statement_debug_info(
    ctx: &mut Context,
    body: &mir::Body,
    infos: &[StatementDebugInfo],
    value_map: &ValueMap,
    block: Ptr<BasicBlock>,
    mut prev_op: Option<Ptr<Operation>>,
) -> TranslationResult<Option<Ptr<Operation>>> {
    for info in infos {
        let StatementDebugInfo::AssignRef { destination, place } = info else {
            continue;
        };
        if !value_map.statement_debug_spill_enabled(*destination) {
            continue;
        }

        let Some((destination_ty, is_mutable)) = destination_reference_type(body, *destination)
        else {
            return input_err_noloc!(TranslationErr::invalid_op(
                "statement-debug spill destination changed after preflight"
            ));
        };
        let Some((address, address_prev)) = rvalue::translate_place_address(
            ctx,
            body,
            value_map,
            place,
            is_mutable,
            block,
            prev_op,
            Location::Unknown,
        )?
        else {
            return input_err_noloc!(TranslationErr::invalid_op(
                "representable statement-debug address failed after preflight"
            ));
        };

        let expected_ty = types::translate_type(ctx, &destination_ty)?;
        let (address, address_prev) = values::establish_declared_pointer_type(
            ctx,
            address,
            expected_ty,
            block,
            address_prev,
            MirPointerKindAuthorityAttr::Reborrow,
        );
        let Some(store) = value_map.store_local(ctx, *destination, address, block, address_prev)
        else {
            return input_err_noloc!(TranslationErr::invalid_op(
                "statement-debug destination slot disappeared after preflight"
            ));
        };
        prev_op = Some(store);
    }

    Ok(prev_op)
}

/// Accept the two runtime-index reference shapes validated by the full-debug
/// stack-home bridge.
///
/// `&slice[index]` is the original #1259 case. `&fixed_array[index]` extends
/// the same mechanism to one direct runtime index on a fixed-size array. Fields,
/// downcasts, subslices, constant indexes, and longer projection chains are
/// rejected before the general address walker runs.
fn assign_ref_is_representable(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    destination: mir::Local,
    place: &mir::Place,
) -> Option<AssignRefSources> {
    let (destination_ty, destination_is_mutable) = destination_reference_type(body, destination)?;
    let TyKind::RigidTy(RigidTy::Ref(_, destination_pointee, _)) = destination_ty.kind() else {
        return None;
    };
    if place.local == destination {
        return None;
    }

    let index = match place.projection.as_slice() {
        [
            mir::ProjectionElem::Deref,
            mir::ProjectionElem::Index(index),
        ] => {
            let base_ty = body.local_decl(place.local).map(|decl| decl.ty)?;
            let (slice_ty, base_is_mutable) = match base_ty.kind() {
                TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability))
                | TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
                    (pointee, matches!(mutability, mir::Mutability::Mut))
                }
                _ => return None,
            };
            if destination_is_mutable && !base_is_mutable {
                return None;
            }
            let TyKind::RigidTy(RigidTy::Slice(element)) = slice_ty.kind() else {
                return None;
            };
            if element != destination_pointee {
                return None;
            }
            *index
        }
        [mir::ProjectionElem::Index(index)] => {
            // Keep the first fixed-array extension intentionally narrow: one
            // immutable reference and one direct runtime usize index.
            if destination_is_mutable {
                return None;
            }
            let base_ty = body.local_decl(place.local).map(|decl| decl.ty)?;
            let TyKind::RigidTy(RigidTy::Array(element, _)) = base_ty.kind() else {
                return None;
            };
            if element != destination_pointee {
                return None;
            }
            *index
        }
        _ => return None,
    };

    if index == destination
        || place.ty(body.locals()).ok() != Some(destination_pointee)
        || !matches!(
            body.local_decl(index).map(|decl| decl.ty.kind()),
            Some(TyKind::RigidTy(RigidTy::Uint(UintTy::Usize)))
        )
    {
        return None;
    }

    let destination_slot = value_map.get_slot(destination)?;
    let expected_destination_ty = types::translate_type(ctx, &destination_ty).ok()?;
    let base_ty = body.local_decl(place.local).map(|decl| decl.ty)?;
    let expected_base_ty = types::translate_type(ctx, &base_ty).ok()?;
    let index_ty = body.local_decl(index).map(|decl| decl.ty)?;
    let expected_index_ty = types::translate_type(ctx, &index_ty).ok()?;

    if !slot_stores_type(ctx, destination_slot, expected_destination_ty)
        || !value_map
            .get_slot(place.local)
            .is_some_and(|slot| slot_stores_type(ctx, slot, expected_base_ty))
        || !value_map
            .get_slot(index)
            .is_some_and(|slot| slot_stores_type(ctx, slot, expected_index_ty))
    {
        return None;
    }

    Some(AssignRefSources {
        base: place.local,
        index,
    })
}

fn slot_stores_type(
    ctx: &Context,
    slot: pliron::value::Value,
    expected: pliron::r#type::TypeHandle,
) -> bool {
    slot.get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_some_and(|slot| slot.pointee == expected)
}

fn destination_reference_type(
    body: &mir::Body,
    destination: mir::Local,
) -> Option<(rustc_public::ty::Ty, bool)> {
    let destination_ty = body.local_decl(destination)?.ty;
    let TyKind::RigidTy(RigidTy::Ref(_, _, mutability)) = destination_ty.kind() else {
        return None;
    };
    Some((destination_ty, matches!(mutability, mir::Mutability::Mut)))
}

/// Compute the locals that are definitely initialized on entry to each block
/// in rustc's exact monomorphized, non-unwind CFG.
///
/// This is a small must-analysis used only as a safety gate. It begins with
/// arguments initialized, intersects predecessor states at joins, models
/// `StorageLive`/`StorageDead`, whole-local assignments, moves, drops, and
/// normal-return call destinations, and otherwise errs on the side of
/// treating a local as uninitialized. It never changes executable MIR.
fn definitely_initialized_at_block_entries(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
    rustc_mono_successors: &[Vec<usize>],
) -> Vec<Vec<bool>> {
    let local_count = body.locals().len();
    let mut entry_state = vec![false; local_count];
    for argument in entry_state
        .iter_mut()
        .take(body.arg_locals().len() + 1)
        .skip(1)
    {
        *argument = true;
    }

    // Must analyses start non-entry blocks at top. Successive predecessor
    // intersections monotonically remove facts until the fixed point.
    let mut block_entries = vec![vec![true; local_count]; body.blocks.len()];
    if !block_entries.is_empty() {
        block_entries[0] = entry_state;
    }

    loop {
        let mut incoming: Vec<Option<Vec<bool>>> = vec![None; body.blocks.len()];
        for &block_index in reachable {
            let outgoing = initialization_successor_states(
                body,
                block_index,
                &block_entries[block_index],
                &rustc_mono_successors[block_index],
            );
            for (successor, state) in outgoing {
                if !reachable.contains(&successor) || successor == 0 {
                    continue;
                }
                match &mut incoming[successor] {
                    Some(current) => {
                        for (current, new) in current.iter_mut().zip(state) {
                            *current &= new;
                        }
                    }
                    slot @ None => *slot = Some(state),
                }
            }
        }

        let mut changed = false;
        for &block_index in reachable {
            if block_index == 0 {
                continue;
            }
            let next = incoming[block_index]
                .take()
                .unwrap_or_else(|| vec![false; local_count]);
            if block_entries[block_index] != next {
                block_entries[block_index] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Unreachable blocks have no runtime initialization facts.
    for (block_index, state) in block_entries.iter_mut().enumerate() {
        if !reachable.contains(&block_index) {
            state.fill(false);
        }
    }
    block_entries
}

fn initialization_successor_states(
    body: &mir::Body,
    block_index: usize,
    entry: &[bool],
    successors: &[usize],
) -> Vec<(usize, Vec<bool>)> {
    let block = &body.blocks[block_index];
    let mut state = entry.to_vec();
    for (statement_index, statement) in block.statements.iter().enumerate() {
        apply_statement_initialization(body, block_index, statement_index, statement, &mut state);
    }

    kill_moved_terminator_operands(body, block_index, &block.terminator, &mut state);

    match &block.terminator.kind {
        mir::TerminatorKind::Drop { place, .. } => {
            set_initialized(&mut state, place.local, false);
        }
        mir::TerminatorKind::Call { destination, .. } => {
            if destination.projection.is_empty() {
                set_initialized(&mut state, destination.local, false);
            }
        }
        mir::TerminatorKind::InlineAsm { operands, .. } => {
            for output in operands
                .iter()
                .filter_map(|operand| operand.out_place.as_ref())
            {
                if output.projection.is_empty() {
                    set_initialized(&mut state, output.local, false);
                }
            }
        }
        mir::TerminatorKind::Goto { .. }
        | mir::TerminatorKind::SwitchInt { .. }
        | mir::TerminatorKind::Resume
        | mir::TerminatorKind::Abort
        | mir::TerminatorKind::Return
        | mir::TerminatorKind::Unreachable
        | mir::TerminatorKind::Assert { .. } => {}
    }

    successors
        .iter()
        .copied()
        .map(|successor| {
            let mut successor_state = state.clone();
            match &block.terminator.kind {
                mir::TerminatorKind::Call {
                    destination,
                    target: Some(target),
                    ..
                } if *target == successor && destination.projection.is_empty() => {
                    set_initialized(&mut successor_state, destination.local, true);
                }
                mir::TerminatorKind::InlineAsm {
                    operands,
                    destination: Some(target),
                    ..
                } if *target == successor => {
                    for output in operands
                        .iter()
                        .filter_map(|operand| operand.out_place.as_ref())
                    {
                        if output.projection.is_empty() {
                            set_initialized(&mut successor_state, output.local, true);
                        }
                    }
                }
                _ => {}
            }
            (successor, successor_state)
        })
        .collect()
}

fn apply_statement_initialization(
    body: &mir::Body,
    block_index: usize,
    statement_index: usize,
    statement: &mir::Statement,
    initialized: &mut [bool],
) {
    let mut move_killer = MoveKiller { initialized };
    move_killer.visit_statement(
        statement,
        mir::visit::statement_location(body, &block_index, statement_index),
    );

    match &statement.kind {
        StatementKind::Assign(place, _) if place.projection.is_empty() => {
            set_initialized(initialized, place.local, true);
        }
        StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
            set_initialized(initialized, *local, false);
        }
        StatementKind::Assign(..)
        | StatementKind::FakeRead(..)
        | StatementKind::SetDiscriminant { .. }
        | StatementKind::PlaceMention(..)
        | StatementKind::AscribeUserType { .. }
        | StatementKind::Coverage(..)
        | StatementKind::Intrinsic(..)
        | StatementKind::ConstEvalCounter
        | StatementKind::Nop => {}
    }
}

fn kill_moved_terminator_operands(
    body: &mir::Body,
    block_index: usize,
    terminator: &mir::Terminator,
    initialized: &mut [bool],
) {
    let mut move_killer = MoveKiller { initialized };
    move_killer.visit_terminator(
        terminator,
        mir::visit::terminator_location(body, &block_index),
    );
}

struct MoveKiller<'a> {
    initialized: &'a mut [bool],
}

impl MirVisitor for MoveKiller<'_> {
    fn visit_operand(&mut self, operand: &mir::Operand, _location: mir::visit::Location) {
        if let mir::Operand::Move(place) = operand {
            // A projected move may leave other fields initialized, but this
            // bounded analysis tracks only whole-local availability. Killing
            // the base is conservative and cannot enable instrumentation.
            set_initialized(self.initialized, place.local, false);
        }
    }
}

fn set_initialized(initialized: &mut [bool], local: mir::Local, value: bool) {
    if let Some(initialized) = initialized.get_mut(local) {
        *initialized = value;
    }
}

/// Mark locals touched by executable statements or terminators in rustc's
/// reachable subgraph. Storage markers, type ascriptions, coverage, fake
/// reads, and place mentions do not produce executable reads or writes and do
/// not disqualify an otherwise ghost destination.
fn executable_local_uses(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Vec<bool> {
    struct LocalUseVisitor {
        used: Vec<bool>,
    }

    impl MirVisitor for LocalUseVisitor {
        fn visit_local(
            &mut self,
            local: &mir::Local,
            _context: mir::visit::PlaceContext,
            _location: mir::visit::Location,
        ) {
            if let Some(used) = self.used.get_mut(*local) {
                *used = true;
            }
        }
    }

    let mut visitor = LocalUseVisitor {
        used: vec![false; body.locals().len()],
    };
    for &block_index in reachable {
        let block = &body.blocks[block_index];
        for (statement_index, statement) in block.statements.iter().enumerate() {
            if matches!(
                statement.kind,
                StatementKind::Assign(..)
                    | StatementKind::SetDiscriminant { .. }
                    | StatementKind::Intrinsic(..)
            ) {
                visitor.visit_statement(
                    statement,
                    mir::visit::statement_location(body, &block_index, statement_index),
                );
            }
        }
        visitor.visit_terminator(
            &block.terminator,
            mir::visit::terminator_location(body, &block_index),
        );
    }
    visitor.used
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::MirStoreOp;
    use llvm_export::ops::{
        DebugFragment, DebugFragmentVariableInfo, DebugLocalTypeKind, DebugLocalVariableInfo,
        DebugProjectedVariableInfo, DebugWholeVariableInfo,
    };
    use pliron::context::Context;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    use rustc_public::CrateDef;

    fn with_statement_debug_fixture(
        function_suffix: &'static str,
        test: impl FnOnce(&mir::Body) + Send + 'static,
    ) {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_statement_debug_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("statement_debug_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
#[inline(never)]
fn keep_index(index: usize) -> usize { index }

#[inline(never)]
pub fn probe(input: &[i32], index: usize, _output: &mut i32) -> i32 {
    let computed = keep_index(index);
    if index & 1 == 0 {
        let branch_index = keep_index(index);
        core::hint::black_box(branch_index);
    }
    if index == usize::MAX {
        panic!("debug-only panic path: {index}");
    }
    let ptr: *const i32 = &input[computed] as *const i32;
    unsafe { *ptr }
}

#[inline(never)]
pub fn probe_array(values: [i32; 4], index: usize) -> i32 {
    let computed = keep_index(index) & 3;
    let projected_runtime = &values[computed];
    core::hint::black_box(*projected_runtime)
}
"#,
        )
        .unwrap();

        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let sysroot_output = std::process::Command::new(rustc)
            .args(["--print", "sysroot"])
            .output()
            .expect("query rustc sysroot");
        assert!(sysroot_output.status.success(), "rustc --print sysroot");
        let sysroot = String::from_utf8(sysroot_output.stdout)
            .expect("sysroot path is UTF-8")
            .trim()
            .to_string();

        let args = vec![
            "rustc".to_string(),
            "--edition=2024".to_string(),
            "--crate-type=rlib".to_string(),
            "--crate-name=statement_debug_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Cpanic=abort".to_string(),
            "-Cdebuginfo=2".to_string(),
            "-Copt-level=3".to_string(),
            "-Zmir-enable-passes=-ScalarReplacementOfAggregates,-SingleUseConsts".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let body = rustc_public::all_local_items()
                        .into_iter()
                        .find(|item| item.name().ends_with(function_suffix))
                        .and_then(|item| item.body())
                        .expect("statement-debug fixture body");
                    test(&body);
                    std::ops::ControlFlow::<(), _>::Continue(())
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();
    }

    fn with_pointer_fixture(test: impl FnOnce(&mir::Body) + Send + 'static) {
        with_statement_debug_fixture("::probe", test);
    }

    fn with_array_fixture(test: impl FnOnce(&mir::Body) + Send + 'static) {
        with_statement_debug_fixture("::probe_array", test);
    }

    fn debug_local(body: &mir::Body, name: &str) -> mir::Local {
        body.var_debug_info
            .iter()
            .find(|info| info.name == name)
            .and_then(|info| match &info.value {
                mir::VarDebugInfoContents::Place(place) if place.projection.is_empty() => {
                    Some(place.local)
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("whole debug local `{name}`"))
    }

    fn cfg(body: &mir::Body) -> (Vec<Vec<usize>>, std::collections::BTreeSet<usize>) {
        let successors: Vec<_> = body
            .blocks
            .iter()
            .map(|block| block.terminator.successors())
            .collect();
        let mut reachable = std::collections::BTreeSet::from([0]);
        let mut worklist = vec![0];
        while let Some(block) = worklist.pop() {
            for &successor in &successors[block] {
                if reachable.insert(successor) {
                    worklist.push(successor);
                }
            }
        }
        (successors, reachable)
    }

    fn empty_debug_map(body: &mir::Body) -> StatementDebugInfoMap {
        StatementDebugInfoMap {
            blocks: body
                .blocks
                .iter()
                .map(|block| crate::pipeline::StatementDebugInfoBlock {
                    before_statements: vec![Vec::new(); block.statements.len()],
                    before_terminator: Vec::new(),
                })
                .collect(),
        }
    }

    fn event_boundary_mut<'a>(
        debug_info: &'a mut StatementDebugInfoMap,
        body: &mir::Body,
        block: usize,
    ) -> &'a mut Vec<StatementDebugInfo> {
        if body.blocks[block].statements.is_empty() {
            &mut debug_info.blocks[block].before_terminator
        } else {
            &mut debug_info.blocks[block].before_statements[0]
        }
    }

    fn event_boundary<'a>(
        debug_info: &'a StatementDebugInfoMap,
        body: &mir::Body,
        block: usize,
    ) -> &'a [StatementDebugInfo] {
        if body.blocks[block].statements.is_empty() {
            &debug_info.blocks[block].before_terminator
        } else {
            &debug_info.blocks[block].before_statements[0]
        }
    }

    fn make_value_map(
        body: &mir::Body,
        locals: &[mir::Local],
    ) -> (Context, Ptr<BasicBlock>, ValueMap, Option<Ptr<Operation>>) {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let mut value_map = ValueMap::new(body.locals().len());
        let mut previous = None;
        for &local in locals {
            let ty = types::translate_type(
                &mut ctx,
                &body
                    .local_decl(local)
                    .expect("fixture local declaration")
                    .ty,
            )
            .expect("fixture type translates");
            let (alloca, slot) = ValueMap::emit_alloca(&mut ctx, ty, block, previous);
            value_map.set_slot(local, slot);
            previous = Some(alloca);
        }
        (ctx, block, value_map, previous)
    }

    fn attach_primary_debug_identity(ctx: &mut Context, slot: pliron::value::Value, name: &str) {
        let op = slot.defining_op().expect("alloca result");
        llvm_export::ops::set_debug_local_variable(
            ctx,
            op,
            DebugLocalVariableInfo {
                name: name.to_string(),
                argument_index: None,
                ty: DebugLocalTypeKind::Pointer {
                    name: "&i32".to_string(),
                    size_bits: 64,
                },
            },
        );
    }

    fn attach_debug_identities(ctx: &mut Context, slot: pliron::value::Value) {
        attach_primary_debug_identity(ctx, slot, "ptr");
        let op = slot.defining_op().expect("alloca result");
        let ty = DebugLocalTypeKind::Pointer {
            name: "&i32".to_string(),
            size_bits: 64,
        };
        llvm_export::ops::set_debug_whole_variable_aliases(
            ctx,
            op,
            &[DebugWholeVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "ptr_alias".to_string(),
                    argument_index: None,
                    ty: ty.clone(),
                },
                source_scope: Some(1),
                declaration: None,
            }],
        );
        llvm_export::ops::set_debug_projected_variables(
            ctx,
            op,
            &[DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "ptr_projected".to_string(),
                    argument_index: None,
                    ty: ty.clone(),
                },
                dereference_base: false,
                offset_bytes: 0,
                source_scope: Some(1),
                declaration: None,
            }],
        );
        llvm_export::ops::set_debug_fragment_variables(
            ctx,
            op,
            &[DebugFragmentVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: "ptr_fragment".to_string(),
                    argument_index: None,
                    ty,
                },
                fragment: DebugFragment {
                    offset_bits: 0,
                    size_bits: 64,
                },
                source_scope: Some(1),
                declaration: None,
            }],
        );
    }

    fn return_block(body: &mir::Body) -> usize {
        body.blocks
            .iter()
            .position(|block| matches!(block.terminator.kind, mir::TerminatorKind::Return))
            .expect("probe return block")
    }

    fn valid_event(
        destination: mir::Local,
        base: mir::Local,
        index: mir::Local,
    ) -> StatementDebugInfo {
        StatementDebugInfo::AssignRef {
            destination,
            place: mir::Place {
                local: base,
                projection: vec![
                    mir::ProjectionElem::Deref,
                    mir::ProjectionElem::Index(index),
                ],
            },
        }
    }

    #[test]
    fn assign_ref_emits_one_typed_store_after_dominating_sources() {
        with_pointer_fixture(|body| {
            let base = debug_local(body, "input");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "ptr");
            let event_block = return_block(body);
            let (successors, reachable) = cfg(body);

            assert!(
                body.blocks.iter().any(|block| matches!(
                    &block.terminator.kind,
                    mir::TerminatorKind::Call {
                        destination: call_destination,
                        target: Some(_),
                        ..
                    } if call_destination.projection.is_empty()
                        && call_destination.local == index
                )),
                "computed index must be initialized by a predecessor call destination"
            );
            let initialized =
                definitely_initialized_at_block_entries(body, &reachable, &successors);
            assert!(
                initialized[event_block][index],
                "the predecessor call store must dominate the chosen debug boundary"
            );

            let mut debug_info = empty_debug_map(body);
            event_boundary_mut(&mut debug_info, body, event_block).push(valid_event(
                destination,
                base,
                index,
            ));
            let (mut ctx, block, mut value_map, previous) =
                make_value_map(body, &[base, index, destination]);
            let destination_slot = value_map.get_slot(destination).unwrap();
            attach_debug_identities(&mut ctx, destination_slot);

            prepare_statement_debug_spills(
                &mut ctx,
                body,
                &debug_info,
                &reachable,
                &successors,
                &mut value_map,
            );
            assert!(value_map.statement_debug_spill_enabled(destination));
            let last = translate_statement_debug_info(
                &mut ctx,
                body,
                event_boundary(&debug_info, body, event_block),
                &value_map,
                block,
                previous,
            )
            .expect("supported AssignRef translates")
            .expect("AssignRef emits a final store");

            let stores: Vec<_> = block
                .deref(&ctx)
                .iter(&ctx)
                .filter_map(|op| Operation::get_op::<MirStoreOp>(op, &ctx))
                .filter(|store| store.address_opd(&ctx) == destination_slot)
                .collect();
            assert_eq!(
                stores.len(),
                1,
                "one event boundary emits exactly one home store"
            );
            assert_eq!(stores[0].get_operation(), last);
            let destination_ty =
                types::translate_type(&mut ctx, &body.local_decl(destination).unwrap().ty).unwrap();
            assert_eq!(stores[0].value_opd(&ctx).get_type(&ctx), destination_ty);
        });
    }

    #[test]
    fn fixed_array_runtime_index_emits_stack_home_store() {
        with_array_fixture(|body| {
            let base = debug_local(body, "values");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "projected_runtime");
            let place = mir::Place {
                local: base,
                projection: vec![mir::ProjectionElem::Index(index)],
            };

            let (mut ctx, block, mut value_map, previous) =
                make_value_map(body, &[base, index, destination]);
            let destination_slot = value_map.get_slot(destination).unwrap();
            attach_primary_debug_identity(&mut ctx, destination_slot, "projected_runtime");

            let sources =
                assign_ref_is_representable(&mut ctx, body, &value_map, destination, &place)
                    .expect("direct fixed-array runtime index is representable");
            assert_eq!(sources.base, base);
            assert_eq!(sources.index, index);

            value_map.set_statement_debug_spill(destination, true);
            let before = block.deref(&ctx).iter(&ctx).count();
            let event = StatementDebugInfo::AssignRef { destination, place };
            let last = translate_statement_debug_info(
                &mut ctx,
                body,
                &[event],
                &value_map,
                block,
                previous,
            )
            .expect("runtime-index AssignRef translates")
            .expect("runtime-index AssignRef emits a final store");

            assert!(
                block.deref(&ctx).iter(&ctx).count() > before,
                "stack-home reconstruction emits address operations and a final store"
            );
            let stores: Vec<_> = block
                .deref(&ctx)
                .iter(&ctx)
                .filter_map(|op| Operation::get_op::<MirStoreOp>(op, &ctx))
                .filter(|store| store.address_opd(&ctx) == destination_slot)
                .collect();
            assert_eq!(
                stores.len(),
                1,
                "one fixed-array runtime-index event emits exactly one home store"
            );
            assert_eq!(stores[0].get_operation(), last);
            let destination_ty =
                types::translate_type(&mut ctx, &body.local_decl(destination).unwrap().ty).unwrap();
            assert_eq!(stores[0].value_opd(&ctx).get_type(&ctx), destination_ty);
        });
    }

    #[test]
    fn fixed_array_multiple_runtime_indices_fail_closed() {
        with_array_fixture(|body| {
            let base = debug_local(body, "values");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "projected_runtime");
            let place = mir::Place {
                local: base,
                projection: vec![
                    mir::ProjectionElem::Index(index),
                    mir::ProjectionElem::Index(index),
                ],
            };
            let (mut ctx, _block, value_map, _previous) =
                make_value_map(body, &[base, index, destination]);

            assert!(
                assign_ref_is_representable(&mut ctx, body, &value_map, destination, &place)
                    .is_none(),
                "multiple runtime indices must remain outside the bounded stack-home bridge"
            );
        });
    }

    fn assert_destination_suppressed(ctx: &Context, value_map: &ValueMap, destination: mir::Local) {
        assert!(!value_map.statement_debug_spill_enabled(destination));
        let op = value_map
            .get_slot(destination)
            .unwrap()
            .defining_op()
            .unwrap();
        assert!(llvm_export::ops::debug_local_variable(ctx, op).is_none());
        assert!(llvm_export::ops::debug_whole_variable_aliases(ctx, op).is_empty());
        assert!(llvm_export::ops::debug_projected_variables(ctx, op).is_empty());
        assert!(llvm_export::ops::debug_fragment_variables(ctx, op).is_empty());
    }

    #[test]
    fn invalid_assign_fails_closed_without_partial_operations() {
        with_pointer_fixture(|body| {
            let base = debug_local(body, "input");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "ptr");
            let event_block = return_block(body);
            let (successors, reachable) = cfg(body);
            let mut debug_info = empty_debug_map(body);
            let events = event_boundary_mut(&mut debug_info, body, event_block);
            events.push(valid_event(destination, base, index));
            events.push(StatementDebugInfo::InvalidAssign { destination });

            let (mut ctx, block, mut value_map, previous) =
                make_value_map(body, &[base, index, destination]);
            attach_debug_identities(&mut ctx, value_map.get_slot(destination).unwrap());
            prepare_statement_debug_spills(
                &mut ctx,
                body,
                &debug_info,
                &reachable,
                &successors,
                &mut value_map,
            );
            assert_destination_suppressed(&ctx, &value_map, destination);
            let before = block.deref(&ctx).iter(&ctx).count();
            let result = translate_statement_debug_info(
                &mut ctx,
                body,
                event_boundary(&debug_info, body, event_block),
                &value_map,
                block,
                previous,
            )
            .expect("disabled destination is skipped");
            assert_eq!(result, previous);
            assert_eq!(block.deref(&ctx).iter(&ctx).count(), before);
        });
    }

    #[test]
    fn unsupported_assign_ref_fails_closed_without_partial_operations() {
        with_pointer_fixture(|body| {
            let base = debug_local(body, "input");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "ptr");
            let event_block = return_block(body);
            let (successors, reachable) = cfg(body);
            let mut debug_info = empty_debug_map(body);
            event_boundary_mut(&mut debug_info, body, event_block).extend([
                valid_event(destination, base, index),
                StatementDebugInfo::AssignRef {
                    destination,
                    place: mir::Place {
                        local: base,
                        projection: vec![mir::ProjectionElem::Deref],
                    },
                },
            ]);

            let (mut ctx, block, mut value_map, previous) =
                make_value_map(body, &[base, index, destination]);
            attach_debug_identities(&mut ctx, value_map.get_slot(destination).unwrap());
            prepare_statement_debug_spills(
                &mut ctx,
                body,
                &debug_info,
                &reachable,
                &successors,
                &mut value_map,
            );
            assert_destination_suppressed(&ctx, &value_map, destination);
            let before = block.deref(&ctx).iter(&ctx).count();
            let result = translate_statement_debug_info(
                &mut ctx,
                body,
                event_boundary(&debug_info, body, event_block),
                &value_map,
                block,
                previous,
            )
            .expect("unsupported destination is skipped");
            assert_eq!(result, previous);
            assert_eq!(block.deref(&ctx).iter(&ctx).count(), before);
        });
    }

    #[test]
    fn mutable_destination_rejects_immutable_slice_base() {
        with_pointer_fixture(|body| {
            let base = debug_local(body, "input");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "_output");
            let (mut ctx, _block, value_map, _previous) =
                make_value_map(body, &[base, index, destination]);
            let place = mir::Place {
                local: base,
                projection: vec![
                    mir::ProjectionElem::Deref,
                    mir::ProjectionElem::Index(index),
                ],
            };
            assert!(
                assign_ref_is_representable(&mut ctx, body, &value_map, destination, &place,)
                    .is_none(),
                "an immutable slice cannot honestly reconstruct an `&mut T` debug value"
            );
        });
    }

    #[test]
    fn event_in_dropped_panic_block_fails_closed_without_partial_operations() {
        with_pointer_fixture(|body| {
            let base = debug_local(body, "input");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "ptr");
            let panic_block = body
                .blocks
                .iter()
                .position(|block| {
                    super::super::terminator::is_dropped_panic_call(&block.terminator)
                })
                .expect("fixture contains a reachable dropped-panic block");
            let (successors, reachable) = cfg(body);
            assert!(reachable.contains(&panic_block));

            let mut debug_info = empty_debug_map(body);
            event_boundary_mut(&mut debug_info, body, panic_block).push(valid_event(
                destination,
                base,
                index,
            ));
            let (mut ctx, block, mut value_map, previous) =
                make_value_map(body, &[base, index, destination]);
            attach_debug_identities(&mut ctx, value_map.get_slot(destination).unwrap());
            prepare_statement_debug_spills(
                &mut ctx,
                body,
                &debug_info,
                &reachable,
                &successors,
                &mut value_map,
            );
            assert_destination_suppressed(&ctx, &value_map, destination);
            let before = block.deref(&ctx).iter(&ctx).count();
            let result = translate_statement_debug_info(
                &mut ctx,
                body,
                event_boundary(&debug_info, body, panic_block),
                &value_map,
                block,
                previous,
            )
            .expect("dropped-block destination is skipped");
            assert_eq!(result, previous);
            assert_eq!(block.deref(&ctx).iter(&ctx).count(), before);
        });
    }

    #[test]
    fn uninitialized_join_source_fails_closed_without_partial_operations() {
        with_pointer_fixture(|body| {
            let base = debug_local(body, "input");
            let branch_index = debug_local(body, "branch_index");
            let destination = debug_local(body, "ptr");
            let (successors, reachable) = cfg(body);
            let mut predecessor_counts = vec![0usize; body.blocks.len()];
            for &block in &reachable {
                for &successor in &successors[block] {
                    if reachable.contains(&successor) {
                        predecessor_counts[successor] += 1;
                    }
                }
            }
            let initialized =
                definitely_initialized_at_block_entries(body, &reachable, &successors);
            let join = reachable
                .iter()
                .copied()
                .find(|block| {
                    predecessor_counts[*block] >= 2
                        && !initialized[*block][branch_index]
                        && !super::super::terminator::is_dropped_panic_call(
                            &body.blocks[*block].terminator,
                        )
                })
                .expect("fixture has a join where branch_index is not definitely initialized");

            let mut debug_info = empty_debug_map(body);
            event_boundary_mut(&mut debug_info, body, join).push(valid_event(
                destination,
                base,
                branch_index,
            ));
            let (mut ctx, block, mut value_map, previous) =
                make_value_map(body, &[base, branch_index, destination]);
            attach_debug_identities(&mut ctx, value_map.get_slot(destination).unwrap());
            prepare_statement_debug_spills(
                &mut ctx,
                body,
                &debug_info,
                &reachable,
                &successors,
                &mut value_map,
            );
            assert_destination_suppressed(&ctx, &value_map, destination);
            let before = block.deref(&ctx).iter(&ctx).count();
            let result = translate_statement_debug_info(
                &mut ctx,
                body,
                event_boundary(&debug_info, body, join),
                &value_map,
                block,
                previous,
            )
            .expect("uninitialized-source destination is skipped");
            assert_eq!(result, previous);
            assert_eq!(block.deref(&ctx).iter(&ctx).count(), before);
        });
    }

    #[test]
    fn mono_unreachable_events_do_not_erase_reachable_location() {
        with_pointer_fixture(|body| {
            let base = debug_local(body, "input");
            let index = debug_local(body, "computed");
            let destination = debug_local(body, "ptr");
            let emitted_block = return_block(body);
            let (successors, mut reachable) = cfg(body);
            let mut reaches_emitted = std::collections::BTreeSet::from([emitted_block]);
            loop {
                let old_len = reaches_emitted.len();
                for (block, block_successors) in successors.iter().enumerate() {
                    if block_successors
                        .iter()
                        .any(|successor| reaches_emitted.contains(successor))
                    {
                        reaches_emitted.insert(block);
                    }
                }
                if reaches_emitted.len() == old_len {
                    break;
                }
            }
            let dead_block = reachable
                .iter()
                .copied()
                .find(|block| {
                    *block != 0 && *block != emitted_block && !reaches_emitted.contains(block)
                })
                .expect("fixture has a non-entry block to mark mono-unreachable");
            reachable.remove(&dead_block);

            let mut debug_info = empty_debug_map(body);
            event_boundary_mut(&mut debug_info, body, emitted_block).push(valid_event(
                destination,
                base,
                index,
            ));
            event_boundary_mut(&mut debug_info, body, dead_block).extend([
                valid_event(destination, base, index),
                StatementDebugInfo::InvalidAssign { destination },
                StatementDebugInfo::AssignRef {
                    destination,
                    place: mir::Place {
                        local: base,
                        projection: vec![mir::ProjectionElem::Deref],
                    },
                },
            ]);

            let (mut ctx, block, mut value_map, previous) =
                make_value_map(body, &[base, index, destination]);
            attach_debug_identities(&mut ctx, value_map.get_slot(destination).unwrap());
            prepare_statement_debug_spills(
                &mut ctx,
                body,
                &debug_info,
                &reachable,
                &successors,
                &mut value_map,
            );
            assert!(
                value_map.statement_debug_spill_enabled(destination),
                "events with no emitted PC must not erase a reachable identity"
            );

            translate_statement_debug_info(
                &mut ctx,
                body,
                event_boundary(&debug_info, body, emitted_block),
                &value_map,
                block,
                previous,
            )
            .expect("reachable event translates");
            let destination_slot = value_map.get_slot(destination).unwrap();
            let stores = block
                .deref(&ctx)
                .iter(&ctx)
                .filter_map(|op| Operation::get_op::<MirStoreOp>(op, &ctx))
                .filter(|store| store.address_opd(&ctx) == destination_slot)
                .count();
            assert_eq!(stores, 1, "one emitted event produces one store");
        });
    }
}
