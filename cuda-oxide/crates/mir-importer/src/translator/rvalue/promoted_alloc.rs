/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Anonymous rustc allocations as immutable device globals.
//!
//! A reference constant points into memory rustc evaluated at compile time:
//! a promoted temporary (`&(3, 4)`), the value of `const C: &T`, or whatever a
//! constant's pointer field refers to. That pointer is `&'static`, so safe code
//! can return it from any function. Its storage must therefore outlive every
//! frame, which only module-level storage does:
//!
//! ```text
//!   const PAIR: &(u32, u32) = &(3, 4);
//!
//!   rustc:  PAIR = ptr { AllocId(7) + 0 }      AllocId(7) = 03 00 00 00 04 00 00 00
//!                         │
//!                         ▼
//!   MIR:    %g = mir.global_alloc  { key = sha256(align 4, 0300000004000000),
//!                                    immutable, initializer = 0300000004000000 }
//!           %p = mir.cast %g        (to the declared `&(u32, u32)`, generic)
//!                         │
//!                         ▼
//!   LLVM:   @__device_global_0 = addrspace(1) constant [8 x i8] c"\03\00\00\00\04\00\00\00"
//! ```
//!
//! Each rustc allocation becomes one byte-image global, keyed by its contents
//! (see [`promoted_allocation_global_key`]), so every reference into it that
//! takes this path (`PAIR` and `&PAIR.1` alike) shares the same storage and two
//! pointers into it compare equal. The key, and so the global's identity, does
//! not change when unrelated constants are added elsewhere. A
//! reference whose pointee is a whole array (`&[T; N]`) still takes the
//! promoted-table path in `promoted.rs`, keyed by content, so `&[3, 4]` and a
//! `&u32` into the same allocation can get separate globals. Rust does not
//! promise one address across uses of a constant, so that costs duplicate
//! bytes, not correctness. A pointee whose layout only lowering can check
//! declares the same key typed as that pointee; lowering validates that view
//! against the shared bytes and keeps one global. Pointers stored inside the allocation stay relocations
//! ([`RelocationTarget`]): a pointer to another anonymous allocation points at
//! that allocation's own global, and a pointer to a Rust static points at the
//! static's global.
//!
//! [`RelocationTarget`]: super::statics::RelocationTarget

use super::promoted::{dialect_stored_size, promotable_array_element};
use super::static_global::{
    MaterializedStaticGlobal, StaticGlobalMaterializationState,
    ensure_initializer_relocation_targets, get_static_pointer_info,
    pointer_into_materialized_global, set_global_initializer_hex_attr,
    set_global_initializer_relocations_attr, translate_static_global_pointer,
};
use super::statics::{
    allocation_initializer_data, bytes_to_hex, encode_global_initializer_relocations,
    static_global_key,
};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types;
use dialect_mir::ops::MirGlobalAllocOp;
use dialect_mir::types::{MirArrayType, MirPtrType};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::mir::Mutability;
use rustc_public::mir::alloc::{AllocId, GlobalAlloc};
use rustc_public_bridge::IndexedVal;

/// Lowering key of the global that holds anonymous rustc allocation `alloc_id`.
///
/// The key names the storage, never a view of it: every declaration of the
/// allocation, byte-image or typed, carries it, so lowering emits one global
/// per allocation.
///
/// An immutable allocation is keyed by a SHA-256 digest of its contents:
/// alignment, byte image, and each relocation's offset and target. A target is
/// a static's own key or, for anonymous memory, that allocation's digest, so
/// the key is a function of what the global holds and nothing else. rustc
/// numbers allocations in the order they are interned, which follows the
/// hash-based mono-item order, so keying by `AllocId` renamed a global whenever
/// unrelated constants were added. Two immutable allocations with the same
/// contents get one global; Rust does not guarantee distinct addresses for
/// constants, so sharing their storage is sound. A mutable allocation (the
/// memory behind `&mut [..]` in a `static mut` initializer) keeps its
/// `AllocId`, since merging writable storage would alias it.
pub(super) fn promoted_allocation_global_key(
    alloc_id: AllocId,
    loc: Location,
) -> TranslationResult<String> {
    let mut fingerprints = AllocationFingerprints::default();
    let fingerprint = fingerprints.of(alloc_id, &loc)?;
    Ok(dialect_mir::ops::encode_promoted_global_key(&fingerprint))
}

/// Content fingerprints behind [`promoted_allocation_global_key`], memoized
/// for one key computation.
///
/// A parent hashes each child's fixed-size digest rather than embedding the
/// child's whole description, so a key stays the same size however deep or
/// shared the pointer graph is.
#[derive(Default)]
struct AllocationFingerprints {
    /// Fingerprints already computed in this walk.
    done: std::collections::HashMap<AllocId, String>,
    /// Allocations whose fingerprint is being built, so a relocation cycle
    /// among them fails instead of recursing forever.
    visiting: Vec<AllocId>,
}

impl AllocationFingerprints {
    /// Fingerprint of anonymous allocation `alloc_id`.
    fn of(&mut self, alloc_id: AllocId, loc: &Location) -> TranslationResult<String> {
        use sha2::{Digest, Sha256};

        if let Some(done) = self.done.get(&alloc_id) {
            return Ok(done.clone());
        }

        let GlobalAlloc::Memory(alloc) = GlobalAlloc::from(alloc_id) else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "promoted constant {alloc_id:?} is not a memory allocation"
                ))
            );
        };
        if alloc.mutability == Mutability::Mut {
            return Ok(format!("rustc-alloc:{}", alloc_id.to_index()));
        }
        if self.visiting.contains(&alloc_id) {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "promoted constant {alloc_id:?} reaches itself through its pointer relocations"
                ))
            );
        }

        self.visiting.push(alloc_id);
        // every field is length-prefixed, so distinct contents cannot hash
        // the same byte stream
        let mut hasher = Sha256::new();
        hasher.update(alloc.align.to_le_bytes());
        hasher.update((alloc.bytes.len() as u64).to_le_bytes());
        // uninitialized bytes are emitted as zero, so they hash as zero
        hasher.update(
            alloc
                .bytes
                .iter()
                .map(|byte| byte.unwrap_or(0))
                .collect::<Vec<_>>(),
        );
        let mut entries = alloc.provenance.ptrs.clone();
        entries.sort_by_key(|(offset, _)| *offset);
        hasher.update((entries.len() as u64).to_le_bytes());
        for (offset, provenance) in entries {
            let target = match GlobalAlloc::from(provenance.0) {
                GlobalAlloc::Static(static_def) => static_global_key(&static_def),
                GlobalAlloc::Memory(_) => self.of(provenance.0, loc)?,
                other => {
                    return input_err!(
                        loc.clone(),
                        TranslationErr::unsupported(format!(
                            "promoted constant {alloc_id:?} has a pointer relocation to \
                             unsupported allocation {other:?}; only Rust statics and constant \
                             memory allocations have device storage"
                        ))
                    );
                }
            };
            hasher.update((offset as u64).to_le_bytes());
            hasher.update((target.len() as u64).to_le_bytes());
            hasher.update(target.as_bytes());
        }
        self.visiting.pop();

        let fingerprint = format!(
            "rustc-alloc-content:sha256:{}",
            bytes_to_hex(&hasher.finalize())
        );
        self.done.insert(alloc_id, fingerprint.clone());
        Ok(fingerprint)
    }
}

/// Emit, once per `state`, the device global holding anonymous allocation
/// `alloc_id`, and every global its pointer slots relocate to.
///
/// The global lives in CUDA global memory (`addrspace(1)`) and holds rustc's
/// exact byte image. It is marked immutable, which exports as LLVM `constant`,
/// unless rustc interned the allocation as mutable. `typed_view` selects the
/// declaration's `global_type`; `None` declares the `[N x u8]` byte image.
/// Either way the declaration carries the allocation's one key (see
/// [`promoted_allocation_global_key`]), so a typed view is another view of the
/// same storage, not a copy.
///
/// The global is registered in `state` before its relocations are walked, so a
/// cycle through statics stays finite. Only the first request against a fresh
/// `state` can carry a typed view; relocation targets need just the key, so
/// whichever declaration `state` already holds serves them.
pub(super) fn ensure_promoted_allocation_global(
    ctx: &mut Context,
    alloc_id: AllocId,
    typed_view: Option<TypeHandle>,
    block_ptr: Ptr<BasicBlock>,
    loc: Location,
    state: &mut StaticGlobalMaterializationState,
) -> TranslationResult<MaterializedStaticGlobal> {
    use pliron::builtin::attributes::{StringAttr, TypeAttr};

    let global_key = promoted_allocation_global_key(alloc_id, loc.clone())?;
    if let Some(existing) = state.globals.get(&global_key) {
        return Ok(*existing);
    }

    let GlobalAlloc::Memory(alloc) = GlobalAlloc::from(alloc_id) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "promoted constant {alloc_id:?} is not a memory allocation"
            ))
        );
    };
    let description = format!("promoted constant {alloc_id:?}");
    let initializer = allocation_initializer_data(&alloc, &description, loc.clone())?;
    let allocation_size = initializer.bytes.len() as u64;

    let global_ty = match typed_view {
        Some(ty) => ty,
        None => {
            let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            MirArrayType::get(ctx, byte_ty, allocation_size).into()
        }
    };
    let global_ptr_ty = MirPtrType::get_global(ctx, global_ty, false);

    let global_op = Operation::new(
        ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![global_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    global_op.deref_mut(ctx).set_loc(loc.clone());

    let global_alloc = MirGlobalAllocOp::new(global_op);
    global_alloc.set_attr_global_type(ctx, TypeAttr::new(global_ty));
    global_alloc.set_attr_global_key(ctx, StringAttr::new(global_key.clone()));
    set_global_initializer_hex_attr(ctx, global_op, &bytes_to_hex(&initializer.bytes));
    if !initializer.relocations.is_empty() {
        let encoded = encode_global_initializer_relocations(&initializer.relocations);
        set_global_initializer_relocations_attr(ctx, global_op, &encoded);
    }
    // rustc allocations are at least byte aligned, and lowering requires the
    // alignment of every initialized global
    global_alloc.set_alignment_value(ctx, initializer.alignment.max(1));
    // promoted temporaries and constant values are interned immutable; only
    // mutable memory (a promoted `&mut []`) keeps an ordinary writable global
    if alloc.mutability == Mutability::Not {
        global_alloc.mark_immutable(ctx);
    }

    match state.last_op {
        Some(previous) => global_op.insert_after(ctx, previous),
        None => global_op.insert_at_front(block_ptr, ctx),
    }
    state.last_op = Some(global_op);

    let materialized = MaterializedStaticGlobal::new(
        global_op,
        global_op.deref(ctx).get_result(0),
        allocation_size,
    );
    state.globals.insert(global_key, materialized);

    ensure_initializer_relocation_targets(
        ctx,
        &initializer.relocations,
        &description,
        block_ptr,
        loc,
        state,
    )?;

    Ok(materialized)
}

/// Translate a pointer to `byte_offset` bytes into anonymous allocation
/// `alloc_id`, typed as `result_ptr_ty`.
///
/// `pointee_size` is rustc's size of the pointee when the caller knows the Rust
/// type; otherwise the converted MIR type's stored size is used.
///
/// The pointee's reads go through its converted LLVM type, so that layout must
/// match rustc's byte image. The pointer normally targets the allocation's
/// shared byte-image global, which requires the pointee to pass the importer's
/// promotion gate: the same proof that admits `&[T; N]` tables to immutable
/// globals. A pointee outside that gate (a payload-carrying enum, say) can
/// still be referenced when it covers the whole allocation: it then declares
/// the allocation's global typed as the pointee, and lowering checks rustc's
/// bytes against that type's LLVM layout, exactly as for an initialized device
/// static, while keeping the one shared global. Anything else is refused
/// rather than read through an unproven layout.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_promoted_allocation_pointer(
    ctx: &mut Context,
    alloc_id: AllocId,
    byte_offset: u64,
    pointee_size: Option<u64>,
    result_ptr_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let pointee_ty = {
        let ty_ref = result_ptr_ty.deref(ctx);
        let ptr_ty = ty_ref.downcast_ref::<MirPtrType>().ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(
                "promoted constant pointer must have a thin pointer type"
            ))
        })?;
        ptr_ty.pointee
    };

    let GlobalAlloc::Memory(alloc) = GlobalAlloc::from(alloc_id) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "promoted constant {alloc_id:?} is not a memory allocation"
            ))
        );
    };
    let allocation_size = alloc.bytes.len() as u64;
    let view_size = pointee_size.or_else(|| dialect_stored_size(ctx, pointee_ty));

    // `dialect_stored_size` reports a zero size as unknown, so only a
    // non-empty pointee needs the stored-size agreement
    let stored_size_agrees = view_size == Some(0)
        || (view_size.is_some() && dialect_stored_size(ctx, pointee_ty) == view_size);
    let byte_view_is_faithful = stored_size_agrees && promotable_array_element(ctx, pointee_ty);
    let covers_allocation = byte_offset == 0 && view_size == Some(allocation_size);

    let typed_view = if byte_view_is_faithful {
        None
    } else if covers_allocation {
        Some(pointee_ty)
    } else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "pointer {byte_offset} bytes into a {allocation_size}-byte promoted constant \
                 views it as {}, whose device layout cuda-oxide cannot prove matches rustc's; \
                 only a reference to the whole constant is supported for this type",
                pointee_ty.deref(ctx).disp(ctx)
            ))
        );
    };

    let mut state = StaticGlobalMaterializationState {
        globals: std::collections::HashMap::new(),
        last_op: prev_op,
    };
    let materialized = ensure_promoted_allocation_global(
        ctx,
        alloc_id,
        typed_view,
        block_ptr,
        loc.clone(),
        &mut state,
    )?;

    pointer_into_materialized_global(
        ctx,
        materialized,
        state.last_op.unwrap_or(materialized.global_op),
        &format!("promoted constant {alloc_id:?}"),
        pointee_ty,
        result_ptr_ty,
        byte_offset,
        loc,
    )
}

/// Translate a thin pointer constant that carries provenance.
///
/// `alloc` is the constant's own allocation: one pointer-width slot whose bytes
/// are the addend and whose single provenance entry names the target. A Rust
/// static target reuses the static's global; an anonymous allocation becomes
/// the immutable global from [`translate_promoted_allocation_pointer`].
pub(super) fn translate_reference_constant(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let ptr_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
    let [(0, provenance)] = alloc.provenance.ptrs.as_slice() else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "thin pointer constant of type {rust_ty:?} must carry one relocation at byte 0, \
                 found provenance entries at {:?}",
                alloc
                    .provenance
                    .ptrs
                    .iter()
                    .map(|(offset, _)| *offset)
                    .collect::<Vec<_>>()
            ))
        );
    };
    if alloc.bytes.len() != ptr_width {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "thin pointer constant of type {rust_ty:?} has {} bytes, expected {ptr_width}",
                alloc.bytes.len()
            ))
        );
    }
    let byte_offset = alloc.read_partial_uint(0..ptr_width).map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "failed to read the addend of pointer constant {rust_ty:?}: {e:?}"
        )))
    })? as u64;

    let Some((pointee_rust_ty, origin)) = get_static_pointer_info(rust_ty) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "pointer constant with provenance has unsupported Rust type {rust_ty:?}"
            ))
        );
    };
    let pointee_layout = pointee_rust_ty.layout().map_err(|e| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "pointer constant {rust_ty:?} has a pointee without a layout: {e:?}"
            ))
        )
    })?;
    let pointee_shape = pointee_layout.shape();
    if !pointee_shape.is_sized() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "thin pointer constant {rust_ty:?} has an unsized pointee"
            ))
        );
    }

    match GlobalAlloc::from(provenance.0) {
        GlobalAlloc::Static(static_def) => {
            let pointee_mir_ty = types::translate_type(ctx, &pointee_rust_ty)?;
            translate_static_global_pointer(
                ctx,
                &static_def,
                pointee_mir_ty,
                const_ty_ptr,
                origin.is_mutable(),
                byte_offset,
                block_ptr,
                prev_op,
                loc,
            )
        }
        GlobalAlloc::Memory(_) => translate_promoted_allocation_pointer(
            ctx,
            provenance.0,
            byte_offset,
            Some(pointee_shape.size.bytes() as u64),
            const_ty_ptr,
            block_ptr,
            prev_op,
            loc,
        ),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "pointer constant {rust_ty:?} points at {other:?}, which has no device storage"
            ))
        ),
    }
}
