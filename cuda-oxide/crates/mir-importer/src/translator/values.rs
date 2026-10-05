/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value mapping: MIR locals → alloca slots and associated helper ops.
//!
//! Every non-ZST MIR local is backed by a single `mir.alloca` emitted at the
//! top of the function's entry block. Defs lower to `mir.store` and uses to
//! `mir.load`. This module owns the per-local slot map and the emitters for
//! the three slot operations.
//!
//! The `mem2reg` pass in `pipeline.rs` promotes the scalar slots back into
//! SSA before LLVM lowering, so at steady state the only `mir.alloca`s that
//! survive are those whose addresses actually escape.
//!
//! # Slot address-space inference
//!
//! Rust's reference / raw-pointer types carry no address-space information
//! (`&mut f32`, `*const u32` are distinct pointer kinds but both default to a
//! generic-address-space `MirPtrType`). On GPU, however, intermediate locals frequently end up holding pointers in
//! a concrete address space — e.g. `let p = &mut TILE_A[i]` on a
//! `SharedArray` produces an `addrspace(3)` pointer, yet the Rust local is
//! typed `&mut f32` (generic).
//!
//! Picking the slot's addrspace from Rust's declared type alone causes every
//! store of such a concrete-addrspace pointer to go through a
//! `mir.cast <PtrToPtr>` → LLVM `addrspacecast` → PTX `cvta.shared.u64`,
//! and subsequent loads of that pointer to hit the generic (runtime-
//! dispatched) store path instead of native `st.shared.*`.
//!
//! [`SlotAddrSpaceMap`] is a pre-scan over the MIR body that, per local,
//! infers the pointee address space from the *writes* into that local. The
//! answer is used by `body::emit_entry_allocas` via
//! [`align_pointer_addr_space`] to pick an alloca pointee that matches what
//! actually gets stored.

use super::facts;
use super::facts::self_ty_is_shared_array;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{MirAllocaOp, MirCastOp, MirLoadOp, MirStoreOp};
use dialect_mir::types::{MirPointerKind, MirPtrType, MirSliceType, address_space};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::mir::alloc::GlobalAlloc;
use rustc_public::ty::{ConstantKind, RigidTy, TyKind};

/// Maps MIR locals to their alloca slots.
///
/// # Invariants
///
/// - `slots.len() == num_locals` after construction.
/// - Each `Some(slot)` entry carries a value whose type is [`MirPtrType`];
///   the pointee is the local's Pliron-IR type.
/// - ZST locals (and the unit return slot) remain `None` in `slots`.
pub struct ValueMap {
    slots: Vec<Option<Value>>,
    /// Full-debug-only destinations whose rustc `StmtDebugInfo::AssignRef`
    /// records may be materialized into their otherwise unused backing slot.
    /// The body preflight enables a bit only after proving the local has no
    /// ordinary reachable MIR use and every debug transition is supported.
    statement_debug_spills: Vec<bool>,
    /// Whether this body is collecting full variable debug metadata.  Keeping
    /// this on the existing per-body translation state lets shared-static
    /// identity resolution stay entirely out of Off/LineTables builds without
    /// threading another flag through every operand helper.
    debug_variables: bool,
    /// Per-body unchecked-indexing policy, resolved once by
    /// [`super::body::translate_body`] from the `__unchecked_indexing_config`
    /// marker and the `CUDA_OXIDE_UNCHECKED_INDEXING` environment switch.
    /// When set, `translate_assert` elides `AssertMessage::BoundsCheck`
    /// terminators (out-of-bounds indexing becomes UB, like `get_unchecked`).
    unchecked_indexing: bool,
}

impl ValueMap {
    /// Creates a new map with capacity for the given number of MIR locals.
    pub fn new(num_locals: usize) -> Self {
        Self {
            slots: vec![None; num_locals],
            statement_debug_spills: vec![false; num_locals],
            debug_variables: false,
            unchecked_indexing: false,
        }
    }

    /// Enable source-identity work used only by full variable debug info.
    pub fn set_debug_variables(&mut self, enabled: bool) {
        self.debug_variables = enabled;
    }

    /// Whether full variable debug metadata is enabled for this body.
    pub fn debug_variables(&self) -> bool {
        self.debug_variables
    }

    /// Record the resolved unchecked-indexing policy for this body.
    pub fn set_unchecked_indexing(&mut self, enabled: bool) {
        self.unchecked_indexing = enabled;
    }

    /// Whether bounds-check asserts in this body are elided.
    pub fn unchecked_indexing(&self) -> bool {
        self.unchecked_indexing
    }

    /// Return the alloca pointer backing `local`, or `None` if the local is
    /// ZST / has not been given a slot.
    pub fn get_slot(&self, local: mir::Local) -> Option<Value> {
        let idx: usize = local;
        self.slots.get(idx).copied().flatten()
    }

    /// Record the alloca pointer for `local`. Expected to be called once per
    /// non-ZST local during body setup in [`super::body::translate_body`].
    pub fn set_slot(&mut self, local: mir::Local, slot: Value) {
        let idx: usize = local;
        if idx < self.slots.len() {
            self.slots[idx] = Some(slot);
        }
    }

    /// Enable or disable the targeted statement-debug spill for `local`.
    pub(crate) fn set_statement_debug_spill(&mut self, local: mir::Local, enabled: bool) {
        let idx: usize = local;
        if let Some(slot) = self.statement_debug_spills.get_mut(idx) {
            *slot = enabled;
        }
    }

    /// Whether `local` passed the statement-debug spill preflight.
    pub(crate) fn statement_debug_spill_enabled(&self, local: mir::Local) -> bool {
        let idx: usize = local;
        self.statement_debug_spills
            .get(idx)
            .copied()
            .unwrap_or(false)
    }

    /// Emit a `mir.alloca` for `elem_ty` and insert it into `block`.
    ///
    /// The result pointer lives in the generic address space and is marked
    /// mutable; the alloca's pointee carries the allocated element type. If
    /// `prev_op` is provided, the op is linked immediately after it; otherwise
    /// it is inserted at the front of `block`.
    ///
    /// Returns the inserted op and its result pointer value.
    pub fn emit_alloca(
        ctx: &mut Context,
        elem_ty: TypeHandle,
        block: Ptr<BasicBlock>,
        prev_op: Option<Ptr<Operation>>,
    ) -> (Ptr<Operation>, Value) {
        let ptr_ty = MirPtrType::get_generic(ctx, elem_ty, /* is_mutable */ true);
        let op = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        insert_at(ctx, op, block, prev_op);
        let result = op.deref(ctx).get_result(0);
        (op, result)
    }

    /// Emit `mir.load` from `local`'s slot. Returns `None` for ZST / unset
    /// locals.
    pub fn load_local(
        &self,
        ctx: &mut Context,
        local: mir::Local,
        block: Ptr<BasicBlock>,
        prev_op: Option<Ptr<Operation>>,
    ) -> Option<(Ptr<Operation>, Value)> {
        let slot = self.get_slot(local)?;
        let elem_ty = slot_pointee(ctx, slot);
        let op = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![elem_ty],
            vec![slot],
            vec![],
            0,
        );
        insert_at(ctx, op, block, prev_op);
        let result = op.deref(ctx).get_result(0);
        Some((op, result))
    }

    /// Emit `mir.store` of `value` into `local`'s slot. Returns `None` for ZST
    /// / unset locals.
    ///
    /// When `value` is pointer-like and its representation differs from the
    /// slot's declared pointee, a `mir.cast <PtrToPtr>` is inserted only for
    /// representation-compatible transitions. Local storage does not establish
    /// Rust pointer/reference semantics: `Erased` cannot regain a concrete kind
    /// here, and distinct concrete kinds cannot be interconverted.
    pub fn store_local(
        &self,
        ctx: &mut Context,
        local: mir::Local,
        value: Value,
        block: Ptr<BasicBlock>,
        prev_op: Option<Ptr<Operation>>,
    ) -> Option<Ptr<Operation>> {
        let slot = self.get_slot(local)?;
        let slot_elem_ty = slot_pointee(ctx, slot);
        let (value, prev_op) = maybe_ptr_coerce(ctx, value, slot_elem_ty, block, prev_op);
        let op = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, value],
            vec![],
            0,
        );
        insert_at(ctx, op, block, prev_op);
        Some(op)
    }
}

/// If `value` and `target_ty` are pointer-like MIR types with the same
/// pointee/element shape, emit a `mir.cast <PtrToPtr>` to the exact declared
/// target type, including its pointer kind.
///
/// Boundary counterpart of [`maybe_ptr_coerce`]: use it only where rustc has
/// declared the result type of a new pointer-producing operation, never for
/// generic storage normalization.
pub(crate) fn establish_declared_pointer_type(
    ctx: &mut Context,
    value: Value,
    target_ty: TypeHandle,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    authority: MirPointerKindAuthorityAttr,
) -> (Value, Option<Ptr<Operation>>) {
    let value_ty = value.get_type(ctx);
    if value_ty == target_ty {
        return (value, prev_op);
    }

    let compatible = {
        let value_ref = value_ty.deref(ctx);
        let target_ref = target_ty.deref(ctx);

        match (
            value_ref.downcast_ref::<MirPtrType>(),
            target_ref.downcast_ref::<MirPtrType>(),
        ) {
            (Some(value_ptr), Some(target_ptr)) => value_ptr.pointee == target_ptr.pointee,
            _ => match (
                value_ref.downcast_ref::<MirSliceType>(),
                target_ref.downcast_ref::<MirSliceType>(),
            ) {
                (Some(value_slice), Some(target_slice)) => {
                    value_slice.element_ty == target_slice.element_ty
                }
                _ => false,
            },
        }
    };

    if !compatible {
        return (value, prev_op);
    }

    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![target_ty],
        vec![value],
        vec![],
        0,
    );
    let cast = MirCastOp::new(cast_op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    cast.set_pointer_kind_authority(ctx, authority);
    insert_at(ctx, cast_op, block, prev_op);

    (cast_op.deref(ctx).get_result(0), Some(cast_op))
}

/// Whether generic representation normalization may change `source` into `target`.
///
/// Generic helpers may preserve a concrete Rust pointer kind or deliberately
/// forget it by converting to [`MirPointerKind::Erased`]. They must never
/// recover a concrete kind from `Erased`, because that would make a sequence
/// such as `SharedRef -> Erased -> UniqueRef` able to manufacture uniqueness.
/// Establishing a new concrete kind belongs to an explicit Rust semantic
/// boundary such as `Rvalue::Ref`, `Rvalue::AddressOf`, or a rustc-declared
/// cast/coercion.
pub(crate) fn generic_pointer_kind_retype_allowed(
    source: MirPointerKind,
    target: MirPointerKind,
) -> bool {
    source == target || target == MirPointerKind::Erased
}

/// If `value` and `target_ty` are compatible pointer-like MIR types, emit a
/// `mir.cast <PtrToPtr>` to the exact target type. For thin pointers this
/// intentionally retains the pre-existing behavior of bridging pointee-type
/// mismatches; the semantic kind must still be preserved (or explicitly
/// erased). Fat slices require the same element type and the same kind policy.
///
/// This helper performs representation normalization only. It must not create
/// a Rust reference/raw-pointer category from an `Erased` value or switch
/// directly between distinct concrete categories.
pub(crate) fn maybe_ptr_coerce(
    ctx: &mut Context,
    value: Value,
    target_ty: TypeHandle,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) -> (Value, Option<Ptr<Operation>>) {
    let value_ty = value.get_type(ctx);
    if value_ty == target_ty {
        return (value, prev_op);
    }

    let compatible = {
        let value_ref = value_ty.deref(ctx);
        let target_ref = target_ty.deref(ctx);

        match (
            value_ref.downcast_ref::<MirPtrType>(),
            target_ref.downcast_ref::<MirPtrType>(),
        ) {
            (Some(value_ptr), Some(target_ptr)) => {
                value_ptr.is_mutable == target_ptr.is_mutable
                    && generic_pointer_kind_retype_allowed(value_ptr.kind, target_ptr.kind)
            }
            _ => match (
                value_ref.downcast_ref::<MirSliceType>(),
                target_ref.downcast_ref::<MirSliceType>(),
            ) {
                (Some(value_slice), Some(target_slice)) => {
                    value_slice.element_ty == target_slice.element_ty
                        && value_slice.is_mutable == target_slice.is_mutable
                        && generic_pointer_kind_retype_allowed(value_slice.kind, target_slice.kind)
                }
                _ => false,
            },
        }
    };

    if !compatible {
        return (value, prev_op);
    }

    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![target_ty],
        vec![value],
        vec![],
        0,
    );
    insert_at(ctx, cast_op, block, prev_op);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let cast_value = cast_op.deref(ctx).get_result(0);
    (cast_value, Some(cast_op))
}

/// Recover the pointee (element) type of a slot value. Panics if the value is
/// not a `MirPtrType`; this invariant is established when a slot is recorded
/// via [`ValueMap::set_slot`] after an [`ValueMap::emit_alloca`] call.
fn slot_pointee(ctx: &Context, slot: Value) -> TypeHandle {
    let ptr_ty = slot.get_type(ctx);
    ptr_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .expect("ValueMap slot must carry a MirPtrType value")
        .pointee
}

/// Insert `op` after `prev_op` if provided, else at the front of `block`.
fn insert_at(
    ctx: &mut Context,
    op: Ptr<Operation>,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) {
    match prev_op {
        Some(prev) => op.insert_after(ctx, prev),
        None => op.insert_at_front(block, ctx),
    }
}

// =============================================================================
// Slot address-space inference
// =============================================================================

/// Per-local inferred address space for the alloca slot's *pointee* type.
///
/// Only meaningful when the local's translated type is itself a pointer
/// (`MirPtrType`). For non-pointer locals this state is computed but never
/// consulted — the slot pointee has no addrspace field to override.
///
/// The lattice is monotone (`Uninit → Known(n) → Generic`, never backwards);
/// this guarantees the fixed-point in [`SlotAddrSpaceMap::analyze`] terminates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotAddrSpace {
    /// No classified writes observed yet. The slot will fall back to the
    /// Rust-declared address space in [`SlotAddrSpaceMap::effective`]. This
    /// is the critical "trust the type" state: if the classifier has
    /// nothing confident to say, we defer to Rust's typed addrspace rather
    /// than demoting to generic.
    Uninit,
    /// Every classified write so far produced a pointer in this address
    /// space; the slot can safely be typed to match.
    Known(u32),
    /// Classified writes from multiple, disagreeing address spaces were
    /// observed. The slot must stay generic (`addrspace(0)`) so
    /// `maybe_ptr_coerce` can cast every store site to match.
    Generic,
}

impl SlotAddrSpace {
    /// Monotone join of two observations on the same slot.
    ///
    /// - `Uninit` is the identity (no observation ≡ no change).
    /// - Two `Known(n)` that agree stay `Known(n)`.
    /// - Two disagreeing `Known(_)` collapse to `Generic`.
    /// - `Generic` is the absorbing element (once demoted, stay demoted).
    fn merge(self, other: SlotAddrSpace) -> SlotAddrSpace {
        use SlotAddrSpace::*;
        match (self, other) {
            (Uninit, x) | (x, Uninit) => x,
            (Generic, _) | (_, Generic) => Generic,
            (Known(a), Known(b)) if a == b => Known(a),
            (Known(_), Known(_)) => Generic,
        }
    }
}

/// Result of classifying a single write's right-hand side.
///
/// Fed into [`SlotAddrSpace::merge`] by the analyzer driver.
#[derive(Debug, Clone, Copy)]
enum WriteClass {
    /// We are confident this write produced a pointer in this address space.
    /// Merging promotes an `Uninit` slot to `Known(n)` and disagreeing
    /// `Known(_)` slots to `Generic`.
    Classified(u32),
    /// The write produced something we deliberately don't reason about
    /// (aggregates, arithmetic, casts, complex projections, `Ref`/`AddressOf`,
    /// arbitrary function returns not in the intrinsic whitelist, …).
    ///
    /// The analyzer resolves this to the destination's declared lowering.
    /// For ordinary references and raw pointers that is generic address space
    /// zero, so a reachable unknown write prevents unsound narrowing to a
    /// concrete space. Special pointer stand-ins such as `&mut SharedArray<_>`
    /// retain their declared shared address space.
    Unclassified,
    /// The write is `_y = _x`-style propagation from a local whose state is
    /// still [`SlotAddrSpace::Uninit`]. That's a timing artefact of the
    /// fixed-point iteration, not a genuine "unknown" — we skip this write
    /// and re-examine it on the next iteration, by which point the source
    /// local will have either classified or stayed `Uninit`.
    Pending,
}

/// Per-local result of the address-space pre-scan.
///
/// Indexed by [`mir::Local`]; `body::emit_entry_allocas` consults this once
/// per non-ZST local to decide the alloca pointee's addrspace.
pub struct SlotAddrSpaceMap {
    classes: Vec<SlotAddrSpace>,
}

impl SlotAddrSpaceMap {
    /// Infer per-local slot pointee address spaces by pre-scanning only the
    /// blocks rustc selected for this concrete monomorphized instance.
    ///
    /// Each iteration walks every statement and every `Call` terminator,
    /// classifies the RHS, and merges observations into the destination
    /// local's state. Unclassified pointer writes contribute the pointer's
    /// declared lowering; only temporarily unresolved copy chains are skipped.
    ///
    /// Convergence: each local can transition at most
    /// `Uninit → Known(n) → Generic` (two steps). Propagation chains
    /// `_a = _b = … = _z` are bounded by `num_locals`, so `num_locals + 2`
    /// iterations are guaranteed sufficient.
    pub fn analyze(
        body: &mir::Body,
        reachable: &std::collections::BTreeSet<usize>,
        num_args: usize,
        declared_addr_spaces: &[Option<u32>],
    ) -> Self {
        let num_locals = body.locals().len();
        let mut classes = vec![SlotAddrSpace::Uninit; num_locals];

        // Function arguments are live writes performed at entry, outside the
        // MIR statement list. Seed their declared address spaces so copy
        // chains originating at a generic pointer argument cannot be mistaken
        // for evidence that a destination is exclusively shared/global/etc.
        for (class, declared) in classes
            .iter_mut()
            .zip(declared_addr_spaces)
            .skip(1)
            .take(num_args)
        {
            if let Some(address_space) = *declared {
                *class = SlotAddrSpace::Known(address_space);
            }
        }

        let cap = num_locals.saturating_add(2).max(2);
        for _ in 0..cap {
            let mut changed = false;

            for &block_idx in reachable {
                let block = &body.blocks[block_idx];
                for stmt in &block.statements {
                    let mir::StatementKind::Assign(place, rvalue) = &stmt.kind else {
                        continue;
                    };
                    if !place.projection.is_empty() {
                        continue;
                    }
                    let class = classify_rvalue(rvalue, &classes);
                    let local_idx: usize = place.local;
                    let declared = declared_addr_spaces.get(local_idx).copied().flatten();
                    if let Some(observation) = resolve(class, declared)
                        && merge_into(&mut classes, place.local, observation)
                    {
                        changed = true;
                    }
                }

                let mir::TerminatorKind::Call {
                    func, destination, ..
                } = &block.terminator.kind
                else {
                    continue;
                };
                if !destination.projection.is_empty() {
                    continue;
                }
                let class = classify_call(func);
                let local_idx: usize = destination.local;
                let declared = declared_addr_spaces.get(local_idx).copied().flatten();
                if let Some(observation) = resolve(class, declared)
                    && merge_into(&mut classes, destination.local, observation)
                {
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        Self { classes }
    }

    /// Effective address space for `local`'s slot pointee.
    ///
    /// - `Known(n)` → `n` (the inferred addrspace).
    /// - `Generic`  → `address_space::GENERIC` (writes disagreed or were
    ///   unclassified).
    /// - `Uninit`   → `rust_declared` (no classified writes seen; keep
    ///   whatever `translate_type` produced).
    pub fn effective(&self, local: mir::Local, rust_declared: u32) -> u32 {
        match self
            .classes
            .get(local)
            .copied()
            .unwrap_or(SlotAddrSpace::Uninit)
        {
            SlotAddrSpace::Uninit => rust_declared,
            SlotAddrSpace::Known(n) => n,
            SlotAddrSpace::Generic => address_space::GENERIC,
        }
    }
}

/// Turn a [`WriteClass`] into a [`SlotAddrSpace`] observation. An unknown
/// pointer-producing write contributes its destination's declared lowering;
/// non-pointer writes have no address-space observation. `Pending` is the
/// only skipped state and is revisited by the bounded fixed-point loop.
fn resolve(class: WriteClass, declared: Option<u32>) -> Option<SlotAddrSpace> {
    match class {
        WriteClass::Classified(n) => Some(SlotAddrSpace::Known(n)),
        WriteClass::Unclassified => declared.map(SlotAddrSpace::Known),
        WriteClass::Pending => None,
    }
}

/// Merge `observation` into `classes[local]`. Returns `true` if the slot's
/// state changed.
fn merge_into(
    classes: &mut [SlotAddrSpace],
    local: mir::Local,
    observation: SlotAddrSpace,
) -> bool {
    let Some(slot) = classes.get_mut(local) else {
        return false;
    };
    let merged = slot.merge(observation);
    if merged != *slot {
        *slot = merged;
        true
    } else {
        false
    }
}

/// Classify the write produced by an `Assign(_, rvalue)` statement.
///
/// The rule set is intentionally narrow: when in doubt, return
/// [`WriteClass::Unclassified`] so the destination's declared lowering is
/// observed. The safety invariant is "an ordinary pointer slot is narrowed
/// to a concrete address space only when every reachable write agrees";
/// incomplete classification therefore leaves ordinary pointers generic.
fn classify_rvalue(rvalue: &mir::Rvalue, classes: &[SlotAddrSpace]) -> WriteClass {
    match rvalue {
        // `_y = _x` — propagate `_x`'s current classification. `Move` and
        // `Copy` are indistinguishable for addrspace purposes. `CopyForDeref`
        // behaves the same at this layer.
        mir::Rvalue::Use(mir::Operand::Copy(place), _)
        | mir::Rvalue::Use(mir::Operand::Move(place), _)
        | mir::Rvalue::CopyForDeref(place)
            if place.projection.is_empty() =>
        {
            propagate_from_local(place.local, classes)
        }
        // `_y = CONSTANT` — a constant-operand pointer to a shared-memory
        // static (e.g. `&mut TILE_A` where `TILE_A: SharedArray<...>`)
        // lowers to a `mir.shared_alloc` in `translate_operand` whose
        // result is `addrspace(3)`. The matching `WriteClass::Classified`
        // here keeps the destination slot typed to match, avoiding the
        // otherwise-inevitable `PtrToPtr` narrow-to-generic cast.
        mir::Rvalue::Use(mir::Operand::Constant(const_op), _) => classify_constant(const_op),
        // Every other rvalue shape (aggregates, arithmetic, casts, complex
        // projections, `Ref`/`AddressOf`, …) we decline to reason about
        // here. The matching `Call`-terminator classifier handles the
        // pointer-producing intrinsics; the remaining cases can be
        // tightened in a follow-up if a benchmark asks for it.
        _ => WriteClass::Unclassified,
    }
}

/// Classify a constant operand's address space.
///
/// Kept in sync with `rvalue::is_shared_array_pointer` /
/// `is_barrier_pointer` / ordinary static handling — those gate the emitter;
/// this gates the slot address-space classifier.
fn classify_constant(const_op: &mir::ConstOperand) -> WriteClass {
    let ty = const_op.const_.ty();
    let TyKind::RigidTy(RigidTy::RawPtr(pointee, _) | RigidTy::Ref(_, pointee, _)) = ty.kind()
    else {
        return WriteClass::Unclassified;
    };

    if let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = pointee.kind()
        && (super::types::is_cuda_device_adt(&adt_def, "SharedArray")
            || super::types::is_cuda_device_adt(&adt_def, "Barrier"))
    {
        return WriteClass::Classified(address_space::SHARED);
    }

    let ConstantKind::Allocated(alloc) = const_op.const_.kind() else {
        return WriteClass::Unclassified;
    };
    if alloc.is_null().unwrap_or(false) {
        return WriteClass::Unclassified;
    }
    let Some((_, prov)) = alloc.provenance.ptrs.first() else {
        return WriteClass::Unclassified;
    };
    match GlobalAlloc::from(prov.0) {
        GlobalAlloc::Static(static_def) => {
            // `#[constant]` statics live in addrspace(4) and are recognised
            // by the `ConstantMemory<T>` wrapper on the static's declared type.
            // Other statics live in addrspace(1).
            let static_ty = static_def.ty();
            if is_constant_wrapper_type(&static_ty) {
                WriteClass::Classified(address_space::CONSTANT)
            } else {
                WriteClass::Classified(address_space::GLOBAL)
            }
        }
        _ => WriteClass::Unclassified,
    }
}

/// `true` if `ty` is `cuda_device::ConstantMemory<_>`. Detection by trimmed ADT
/// name, mirroring the `SharedArray | Barrier` check above in
/// [`classify_constant`].
pub(super) fn is_constant_wrapper_type(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};
    let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = ty.kind() else {
        return false;
    };
    super::types::is_cuda_device_adt(&adt_def, "ConstantMemory")
}

/// Classify the write produced by a `Call` terminator's destination.
///
/// Mirrors the intrinsic dispatch table in
/// `translator/terminator/mod.rs::try_dispatch_intrinsic`. Any intrinsic
/// whose emitter unconditionally produces a pointer in a specific address
/// space is listed here; new intrinsics should add an entry on the same
/// commit that adds their emitter.
fn classify_call(func: &mir::Operand) -> WriteClass {
    let mir::Operand::Constant(const_op) = func else {
        return WriteClass::Unclassified;
    };
    if !matches!(const_op.const_.kind(), ConstantKind::ZeroSized) {
        return WriteClass::Unclassified;
    }
    let TyKind::RigidTy(RigidTy::FnDef(fn_def, substs)) = const_op.const_.ty().kind() else {
        return WriteClass::Unclassified;
    };
    let path = fn_def.name();
    let on_shared_array = self_ty_is_shared_array(&substs);

    // --- addrspace 3 (shared) producers -------------------------------------
    //
    // `SharedArray::index` / `IndexMut::index_mut` on a `SharedArray<T, N>`
    // lower to `emit_shared_array_index`, which offsets the shared-memory
    // base pointer and returns `*mut T addrspace(3)`.
    if on_shared_array
        && matches!(
            path.as_str(),
            "std::ops::Index::index"
                | "core::ops::Index::index"
                | "std::ops::IndexMut::index_mut"
                | "core::ops::IndexMut::index_mut"
        )
    {
        return WriteClass::Classified(address_space::SHARED);
    }

    // `DynamicSharedArray::<T, ALIGN>::{get, get_raw, offset}` all hand back
    // pointers into the extern-shared region (`addrspace(3)`). The crate
    // anchor keeps a user type merely named like it from being classified;
    // the dispatch gate in `translator::terminator` applies the same anchor.
    if fn_def.krate().name.as_str() == "cuda_device"
        && path.contains("DynamicSharedArray")
        && (path.contains("::get") || path.contains("::offset"))
    {
        return WriteClass::Classified(address_space::SHARED);
    }

    // --- addrspace 7 (cluster shared) producers ------------------------------
    //
    // `map_shared_rank` and `map_shared_rank_mut` return mapped DSMEM pointers.
    // Their result slots must retain addrspace(7); otherwise `store_local` would
    // insert a PtrToPtr cast back to generic and a later dereference would lose
    // the `ld.shared::cluster` / `st.shared::cluster` selection contract.
    if matches!(
        path.as_str(),
        "cuda_device::cluster::map_shared_rank" | "cuda_device::cluster::map_shared_rank_mut"
    ) {
        return WriteClass::Classified(address_space::CLUSTER_SHARED);
    }

    // --- explicit narrow to generic -----------------------------------------
    //
    // Public SharedArray pointer conversions deliberately `cvta.shared` the
    // base pointer into the generic address space, so the callee sees
    // `addrspace(0)`. Keep recognition shared with intrinsic dispatch.
    if super::shared_array_pointer_method(&path).is_some() {
        return WriteClass::Classified(address_space::GENERIC);
    }

    WriteClass::Unclassified
}

/// Inherit classification from a source local (for `_y = _x` chains).
fn propagate_from_local(local: mir::Local, classes: &[SlotAddrSpace]) -> WriteClass {
    match classes.get(local).copied().unwrap_or(SlotAddrSpace::Uninit) {
        SlotAddrSpace::Known(n) => WriteClass::Classified(n),
        SlotAddrSpace::Generic => WriteClass::Classified(address_space::GENERIC),
        // Source hasn't been classified yet in this iteration — try again
        // on the next pass rather than prematurely demoting the destination.
        SlotAddrSpace::Uninit => WriteClass::Pending,
    }
}

/// If `elem_ty` is a `MirPtrType`, return it with `target` replacing the
/// current address space; otherwise return `elem_ty` unchanged.
///
/// Used by `body::emit_entry_allocas` to override a Rust-declared pointer
/// addrspace with the one inferred by [`SlotAddrSpaceMap`]. The pointer kind
/// is preserved exactly: address-space inference must never turn `&mut T` into
/// an erased/raw pointer, or vice versa.
pub fn align_pointer_addr_space(ctx: &mut Context, elem_ty: TypeHandle, target: u32) -> TypeHandle {
    let ptr_info = elem_ty.deref(ctx).downcast_ref::<MirPtrType>().map(|pt| {
        (
            pt.pointee,
            pt.address_space,
            facts::pointer_origin_of_ptr_carrier(pt),
        )
    });
    let Some((pointee, current, origin)) = ptr_info else {
        return elem_ty;
    };
    if current == target {
        return elem_ty;
    }
    facts::mint_ptr_type(ctx, pointee, target, origin).into()
}

/// Extract a pointer type's address space, or `None` if `elem_ty` is not a
/// [`MirPtrType`]. Useful as the `rust_declared` fallback for
/// [`SlotAddrSpaceMap::effective`].
pub fn pointer_addr_space(ctx: &Context, elem_ty: TypeHandle) -> Option<u32> {
    elem_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|pt| pt.address_space)
}

#[cfg(test)]
// Tests build kinded fixture types directly; production code mints via facts::PointerOrigin.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn rust_boundary_store_establishes_declared_kind_from_erased() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let pointee: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signed).into();
        let erased: TypeHandle =
            MirPtrType::get_shared_with_kind(&mut ctx, pointee, true, MirPointerKind::Erased)
                .into();
        let declared: TypeHandle =
            MirPtrType::get_shared_with_kind(&mut ctx, pointee, true, MirPointerKind::RawMut)
                .into();

        let block = BasicBlock::new(&mut ctx, None, vec![erased]);
        let value = block.deref(&ctx).get_argument(0);
        let (retyped, cast) = establish_declared_pointer_type(
            &mut ctx,
            value,
            declared,
            block,
            None,
            MirPointerKindAuthorityAttr::AbiBoundary,
        );

        assert_eq!(
            retyped.get_type(&ctx),
            declared,
            "an intrinsic-result store is a Rust-typed boundary: the declared kind wins"
        );
        assert!(
            cast.is_some(),
            "the boundary retype must be an explicit cast"
        );
    }

    #[test]
    fn rust_boundary_store_requires_matching_pointee() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let source_pointee: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signed).into();
        let target_pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let erased: TypeHandle = MirPtrType::get_generic(&mut ctx, source_pointee, true).into();
        let declared: TypeHandle = MirPtrType::get_generic_with_kind(
            &mut ctx,
            target_pointee,
            true,
            MirPointerKind::RawMut,
        )
        .into();

        let block = BasicBlock::new(&mut ctx, None, vec![erased]);
        let value = block.deref(&ctx).get_argument(0);
        let (unchanged, cast) = establish_declared_pointer_type(
            &mut ctx,
            value,
            declared,
            block,
            None,
            MirPointerKindAuthorityAttr::AbiBoundary,
        );

        assert_eq!(unchanged.get_type(&ctx), erased);
        assert!(
            cast.is_none(),
            "a boundary retype must not hide a pointee representation mismatch"
        );
    }

    #[test]
    fn pointer_like_local_coercion_does_not_recover_kind_from_erased() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let element: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let erased: TypeHandle = MirSliceType::get(&mut ctx, element).into();
        let expected: TypeHandle =
            MirSliceType::get_with_kind(&mut ctx, element, MirPointerKind::SharedRef).into();

        let block = BasicBlock::new(&mut ctx, None, vec![erased]);
        let value = block.deref(&ctx).get_argument(0);
        let (value, cast) = maybe_ptr_coerce(&mut ctx, value, expected, block, None);

        assert_eq!(value.get_type(&ctx), erased);
        assert!(
            cast.is_none(),
            "generic normalization must not recover SharedRef from Erased"
        );
    }

    #[test]
    fn pointer_like_local_coercion_keeps_thin_pointee_bridge() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let source_pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let target_pointee: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let source: TypeHandle = MirPtrType::get_generic_with_kind(
            &mut ctx,
            source_pointee,
            true,
            MirPointerKind::RawMut,
        )
        .into();
        let target: TypeHandle = MirPtrType::get_generic_with_kind(
            &mut ctx,
            target_pointee,
            true,
            MirPointerKind::RawMut,
        )
        .into();
        let block = BasicBlock::new(&mut ctx, None, vec![source]);
        let value = block.deref(&ctx).get_argument(0);

        let (coerced, cast) = maybe_ptr_coerce(&mut ctx, value, target, block, None);

        assert_eq!(coerced.get_type(&ctx), target);
        assert!(
            cast.is_some(),
            "thin-pointer representation coercion must retain the historical pointee bridge"
        );
    }

    #[test]
    fn pointer_like_local_coercion_cannot_launder_through_erased() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, false, MirPointerKind::SharedRef)
                .into();
        let erased: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let unique: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::UniqueRef)
                .into();
        let block = BasicBlock::new(&mut ctx, None, vec![shared]);
        let value = block.deref(&ctx).get_argument(0);

        let (erased_value, erase_cast) = maybe_ptr_coerce(&mut ctx, value, erased, block, None);
        assert_eq!(erased_value.get_type(&ctx), erased);
        assert!(
            erase_cast.is_some(),
            "forgetting a concrete kind is allowed"
        );

        let previous_anchor = erase_cast.expect("erasing the concrete kind must emit a cast");
        let (laundered, recover_anchor) =
            maybe_ptr_coerce(&mut ctx, erased_value, unique, block, Some(previous_anchor));
        assert_eq!(laundered.get_type(&ctx), erased);
        assert!(
            recover_anchor.is_some(),
            "rejected recovery must preserve the previous insertion anchor"
        );
    }

    #[test]
    fn pointer_like_local_coercion_rejects_concrete_kind_changes() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        for (source_kind, source_mutable, target_kind, target_mutable) in [
            (
                MirPointerKind::SharedRef,
                false,
                MirPointerKind::UniqueRef,
                true,
            ),
            (
                MirPointerKind::RawConst,
                false,
                MirPointerKind::RawMut,
                true,
            ),
            (
                MirPointerKind::RawConst,
                false,
                MirPointerKind::SharedRef,
                false,
            ),
        ] {
            let source: TypeHandle =
                MirPtrType::get_generic_with_kind(&mut ctx, pointee, source_mutable, source_kind)
                    .into();
            let target: TypeHandle =
                MirPtrType::get_generic_with_kind(&mut ctx, pointee, target_mutable, target_kind)
                    .into();
            let block = BasicBlock::new(&mut ctx, None, vec![source]);
            let value = block.deref(&ctx).get_argument(0);

            let (coerced, cast) = maybe_ptr_coerce(&mut ctx, value, target, block, None);

            assert_eq!(
                coerced.get_type(&ctx),
                source,
                "generic normalization must not change {source_kind:?} into {target_kind:?}"
            );
            assert!(
                cast.is_none(),
                "generic normalization must not synthesize a concrete pointer-kind transition"
            );
        }
    }

    #[test]
    fn pointer_like_local_coercion_rejects_fat_pointer_kind_changes() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let element: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let source: TypeHandle =
            MirSliceType::get_with_kind(&mut ctx, element, MirPointerKind::RawConst).into();
        let target: TypeHandle =
            MirSliceType::get_with_kind(&mut ctx, element, MirPointerKind::SharedRef).into();
        let block = BasicBlock::new(&mut ctx, None, vec![source]);
        let value = block.deref(&ctx).get_argument(0);

        let (coerced, cast) = maybe_ptr_coerce(&mut ctx, value, target, block, None);

        assert_eq!(coerced.get_type(&ctx), source);
        assert!(cast.is_none());
    }

    #[test]
    fn pointer_like_local_coercion_cannot_invent_writable_erased_carriers() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let element: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        let immutable_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, element, false).into();
        let mutable_ptr: TypeHandle = MirPtrType::get_generic(&mut ctx, element, true).into();
        let ptr_block = BasicBlock::new(&mut ctx, None, vec![immutable_ptr]);
        let ptr_value = ptr_block.deref(&ctx).get_argument(0);
        let (ptr_result, ptr_cast) =
            maybe_ptr_coerce(&mut ctx, ptr_value, mutable_ptr, ptr_block, None);
        assert_eq!(ptr_result.get_type(&ctx), immutable_ptr);
        assert!(ptr_cast.is_none());

        let immutable_slice: TypeHandle =
            MirSliceType::get_with_mutability(&mut ctx, element, false).into();
        let mutable_slice: TypeHandle =
            MirSliceType::get_with_mutability(&mut ctx, element, true).into();
        let slice_block = BasicBlock::new(&mut ctx, None, vec![immutable_slice]);
        let slice_value = slice_block.deref(&ctx).get_argument(0);
        let (slice_result, slice_cast) =
            maybe_ptr_coerce(&mut ctx, slice_value, mutable_slice, slice_block, None);
        assert_eq!(slice_result.get_type(&ctx), immutable_slice);
        assert!(slice_cast.is_none());
    }

    #[test]
    fn reachable_unknown_pointer_write_prevents_concrete_narrowing() {
        let shared = resolve(
            WriteClass::Classified(address_space::SHARED),
            Some(address_space::GENERIC),
        )
        .unwrap();
        let unknown = resolve(WriteClass::Unclassified, Some(address_space::GENERIC)).unwrap();

        assert_eq!(shared.merge(unknown), SlotAddrSpace::Generic);
    }

    #[test]
    fn generic_source_propagates_as_a_generic_observation() {
        let source = mir::Local::from(0usize);
        assert!(matches!(
            propagate_from_local(source, &[SlotAddrSpace::Generic]),
            WriteClass::Classified(space) if space == address_space::GENERIC
        ));
    }

    #[test]
    fn address_space_alignment_preserves_pointer_kind() {
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let unique: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::UniqueRef)
                .into();

        let aligned = align_pointer_addr_space(&mut ctx, unique, address_space::SHARED);
        let aligned = aligned.deref(&ctx);
        let aligned = aligned
            .downcast_ref::<MirPtrType>()
            .expect("aligned type must remain a MIR pointer");

        assert_eq!(aligned.address_space, address_space::SHARED);
        assert_eq!(aligned.kind, MirPointerKind::UniqueRef);
        assert!(aligned.is_mutable);
    }
}
