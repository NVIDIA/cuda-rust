/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The importer's rustc oracle: every question we ask rustc through the
//! `rustc_public` typed APIs is answered here.
//!
//! ```text
//! rustc (typed APIs) ──► facts.rs ──► the rest of the importer
//! ```
//!
//! The contract: a fact is either read precisely from a typed API, or it is
//! a hard error. Never a guess, never a Debug string. Identity checks
//! (is this cuda-device's `SharedArray`?) are the one soft spot: a miss
//! falls through to generic handling, it does not error.
//!
//! New rustc questions go HERE, nowhere else.
//!
//! # PointerOrigin
//!
//! Concrete pointer kinds (`&T`, `&mut T`, `*const T`, `*mut T`) are minted
//! only here. Evidence goes in, a kinded type comes out:
//!
//! ```text
//! rustc evidence                    named ABI rules
//! (Ty, BorrowKind, RawPtrKind,     (DynamicSharedArray result,
//!  Mutability, in-IR carriers)      DisjointSlice data ptr)
//!        │                                │
//!        └────────► PointerOrigin ◄───────┘
//!                        │
//!                     mint_*  ──► kinded MirPtrType / MirSliceType
//! ```
//!
//! The raw `*_with_kind` constructors are clippy-banned everywhere else in
//! the workspace, so a concrete kind can't be conjured without one of the
//! origins above. Erased pointers need no origin; use the plain
//! constructors (`MirPtrType::get*`, `MirSliceType::get*`).

use crate::error::{TranslationErr, TranslationResult};
use pliron::location::Location;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::{AdtKind, ConstantKind};
use rustc_public_bridge::IndexedVal;

// ============================================================================
// Driver-provided facts (resolved with a TyCtxt the translator lacks)
// ============================================================================

/// Lang-item `DefId`s the type translator compares projections against.
///
/// The translator has no `TyCtxt`, so the driver (which does) resolves these
/// once per compilation and hands them in through `run_pipeline`:
///
/// ```text
/// driver (TyCtxt) --resolve lang items--> KnownDefs --> run_pipeline
///                                                           |
///                              translate_type <-- thread_local
/// ```
///
/// `None` fields simply never match, so a missing id degrades to the
/// existing "Alias type not yet supported" hard error instead of a guess.
#[derive(Clone, Copy, Default)]
pub struct KnownDefs {
    /// The `FnOnce::Output` associated type. `Fn` and `FnMut` declare no
    /// `Output` of their own (they inherit `FnOnce`'s), so this one id
    /// covers projections through all three traits.
    pub fn_once_output: Option<rustc_public::DefId>,
    /// The `core::ops::Index` trait.
    pub index_trait: Option<rustc_public::DefId>,
    /// The `core::ops::IndexMut` trait.
    pub index_mut_trait: Option<rustc_public::DefId>,
}

thread_local! {
    // Freshly set at every `run_pipeline` entry. The stable `DefId`s inside
    // are only meaningful within the surrounding `rustc_internal::run`
    // context, so they must never be cached beyond a single pipeline run.
    static KNOWN_DEFS: std::cell::Cell<KnownDefs> = const {
        std::cell::Cell::new(KnownDefs {
            fn_once_output: None,
            index_trait: None,
            index_mut_trait: None,
        })
    };
}

/// Installs the driver-resolved lang-item ids for this pipeline run,
/// replacing whatever a previous run on this thread left behind.
pub(crate) fn set_known_defs(defs: KnownDefs) {
    KNOWN_DEFS.with(|cell| cell.set(defs));
}

/// The lang-item ids for the current pipeline run (all `None` if the driver
/// never provided them).
pub(crate) fn known_defs() -> KnownDefs {
    KNOWN_DEFS.with(|cell| cell.get())
}

/// Validate the source reference and the fully specialized read-only storage
/// contract using rustc's type system. MIR locals have erased regions, so the
/// outer lifetime is read from the declaration before instantiation instead.
pub(crate) fn validate_grid_constant_parameter(
    instance: &mir::mono::Instance,
    parameter: usize,
    pointee: rustc_public::ty::Ty,
) -> Result<(), String> {
    use rustc_middle::ty::{self, TypingEnv};
    use rustc_public::rustc_internal;

    ty::tls::with(|tcx| {
        let instance = rustc_internal::internal(tcx, *instance);
        let signature = tcx.fn_sig(instance.def_id()).skip_binder().skip_binder();
        let parameter_ty = signature.inputs().get(parameter).ok_or_else(|| {
            format!(
                "grid-constant source parameter {parameter} is absent from the kernel declaration"
            )
        })?;
        let ty::Ref(region, _, _) = parameter_ty.kind() else {
            return Err(format!(
                "grid-constant source parameter {parameter} must be declared as an immutable reference"
            ));
        };
        let launch_lifetime = match region.kind() {
            ty::ReBound(
                _,
                ty::BoundRegion {
                    kind: ty::BoundRegionKind::Anon,
                    ..
                },
            ) => true,
            ty::ReBound(
                _,
                ty::BoundRegion {
                    kind: ty::BoundRegionKind::Named(id),
                    ..
                },
            ) => tcx.item_name(id) == rustc_span::symbol::kw::UnderscoreLifetime,
            _ => false,
        };
        if !launch_lifetime {
            return Err(format!(
                "grid-constant source parameter {parameter} must use an elided reference lifetime (&T or &'_ T); its storage lives only for this kernel launch"
            ));
        }

        let pointee = rustc_internal::internal(tcx, pointee);
        if !pointee.is_freeze(tcx, TypingEnv::fully_monomorphized()) {
            return Err(format!(
                "grid-constant source parameter {parameter} contains interior mutability; launch-parameter storage is read-only, including UnsafeCell, Cell, and atomic fields"
            ));
        }
        Ok(())
    })
}

/// A declaration's numeric parameter indices belong only to its own entry.
/// rustc retains the precise inlining owner in source scopes, even after it
/// has substituted caller locals for the original callee's arguments.
pub(crate) fn validate_grid_constant_marker_owner(
    instance: &mir::mono::Instance,
    block_index: usize,
    is_kernel: bool,
) -> Result<(), String> {
    if !is_kernel {
        return Err(
            "grid-constant ABI declaration is only valid on a kernel entry, not a callable device helper"
                .to_string(),
        );
    }
    rustc_middle::ty::tls::with(|tcx| {
        let instance = rustc_public::rustc_internal::internal(tcx, *instance);
        let body = tcx.instance_mir(instance.def);
        let block = body.basic_blocks.iter().nth(block_index).ok_or_else(|| {
            "grid-constant marker's MIR block has no corresponding rustc source scope".to_string()
        })?;
        if let Some(owner) = block
            .terminator()
            .source_info
            .scope
            .inlined_instance(&body.source_scopes)
        {
            return Err(format!(
                "grid-constant ABI declaration was inlined from `{}`; a device helper or another kernel cannot change this kernel's launch ABI",
                tcx.def_path_str(owner.def_id())
            ));
        }
        Ok(())
    })
}

// ============================================================================
// Identity facts (crate-anchored; a miss falls through, never errors)
// ============================================================================

/// `true` when `adt_def` is cuda-device's type named `name`.
///
/// Special CUDA types (`SharedArray`, `Barrier`, ...) are recognised by bare
/// type name, so the defining-crate anchor is what keeps a user type that
/// merely shares the name out of the special-case translation. On a miss,
/// callers fall through to generic ADT handling — never an error.
pub(crate) fn is_cuda_device_adt(adt_def: &rustc_public::ty::AdtDef, name: &str) -> bool {
    adt_def.trimmed_name() == name && adt_def.krate().name.as_str() == "cuda_device"
}

/// `true` if `func` is an fn item defined in the `cuda_device` crate.
///
/// Anchors name-substring dispatch gates (e.g. `DynamicSharedArray::get`)
/// so a user fn merely spelled like a cuda-device one can't hijack an
/// intrinsic lowering; it falls through to ordinary call handling instead.
pub(crate) fn is_cuda_device_fn(func: &mir::Operand) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};

    let mir::Operand::Constant(constant) = func else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, _)) = constant.const_.ty().kind() else {
        return false;
    };
    definition.krate().name.as_str() == "cuda_device"
}

/// True when `fn_def` is a `precondition_check` shim generated by libcore's
/// `assert_unsafe_precondition!` macro: a nested `const fn` literally named
/// `precondition_check` inside the guarded function (e.g.
/// `core::num::<impl usize>::unchecked_sub::precondition_check`). The macro
/// expands in `core` and (for `Vec` internals) `alloc` — the same crates
/// whose shim bodies the collector skips. Crate anchor + exact tail-segment
/// match, so a user fn such as `my_precondition_check` stays a normal call.
pub(crate) fn is_std_precondition_check(fn_def: &rustc_public::ty::FnDef) -> bool {
    if !matches!(fn_def.krate().name.as_str(), "core" | "alloc") {
        return false;
    }
    let name = fn_def.name();
    name.as_str().rsplit("::").next() == Some("precondition_check")
}

/// `true` if a trait-method call's `Self` type is `cuda_device::SharedArray`.
///
/// rustc puts a trait method's substs in declaration order, `Self` first:
///
/// ```text
/// Index::index on SharedArray<f32, 256>
///   substs = [ SharedArray<f32, 256, 0>,  usize ]
///              ^ Self at position 0       ^ Idx
/// ```
///
/// A miss is a legitimate fall-through (indexing some other type), not an
/// error. Used by both `values::classify_call` and the Index/IndexMut
/// dispatch in `translator::terminator` -- one helper, so the two can't
/// drift.
pub(crate) fn self_ty_is_shared_array(substs: &rustc_public::ty::GenericArgs) -> bool {
    use rustc_public::ty::{GenericArgKind, RigidTy, TyKind};
    let Some(GenericArgKind::Type(ty)) = substs.0.first() else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = ty.kind() else {
        return false;
    };
    is_cuda_device_adt(&adt_def, "SharedArray")
}

// ============================================================================
// Pointer-origin facts (the only importer path to kinded pointer types)
// ============================================================================

// The one place in the importer allowed to touch the clippy-banned kinded
// constructors: every mint below is justified by a rustc-witnessed origin.
#[allow(clippy::disallowed_methods)]
mod pointer_origin {
    use dialect_mir::types::{MirPointerKind, MirPtrType, MirSliceType};
    use pliron::context::Context;
    use pliron::r#type::{TypeHandle, TypedHandle};
    use rustc_public::mir;

    /// A rustc-witnessed origin for a Rust pointer category.
    ///
    /// Fields are private on purpose: the only way to obtain one is to show
    /// rustc evidence to a constructor below (or to invoke a named ABI rule),
    /// so a concrete [`MirPointerKind`] can never appear out of thin air.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct PointerOrigin {
        kind: MirPointerKind,
        mutable: bool,
    }

    impl PointerOrigin {
        /// Machine-level mutability implied by the origin (carrier
        /// mutability for propagated `Erased` origins).
        pub(crate) fn is_mutable(self) -> bool {
            self.mutable
        }

        /// Whether rustc witnessed this origin as a Rust reference.
        pub(crate) fn is_reference(self) -> bool {
            self.kind.is_reference()
        }
    }

    // --- rustc-derived origins ---------------------------------------------

    /// `Some((pointee, origin))` iff `ty` is a reference (`RigidTy::Ref`) or
    /// a raw pointer (`RigidTy::RawPtr`).
    ///
    /// This is the signature-level coupler: every rustc-declared
    /// param/return/local/constant pointer type flows through here on its way
    /// into `dialect-mir`.
    pub(crate) fn pointer_origin_of_ty(
        ty: &rustc_public::ty::Ty,
    ) -> Option<(rustc_public::ty::Ty, PointerOrigin)> {
        use rustc_public::ty::{RigidTy, TyKind};
        match ty.kind() {
            TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
                Some((pointee, pointer_origin_of_raw_mutability(mutability)))
            }
            TyKind::RigidTy(RigidTy::Ref(_region, pointee, mutability)) => {
                let mutable = mutability == mir::Mutability::Mut;
                Some((
                    pointee,
                    PointerOrigin {
                        kind: MirPointerKind::from_reference_mutability(mutable),
                        mutable,
                    },
                ))
            }
            _ => None,
        }
    }

    /// From an `Rvalue::Ref`'s borrow kind: `Mut { .. }` is `UniqueRef`,
    /// everything else (shared and fake borrows) is `SharedRef`.
    pub(crate) fn pointer_origin_of_borrow(kind: mir::BorrowKind) -> PointerOrigin {
        let mutable = matches!(kind, mir::BorrowKind::Mut { .. });
        PointerOrigin {
            kind: MirPointerKind::from_reference_mutability(mutable),
            mutable,
        }
    }

    /// From an `Rvalue::AddressOf`'s raw-pointer kind: `Mut` is `RawMut`,
    /// `Const` and `FakeForPtrMetadata` are `RawConst`.
    pub(crate) fn pointer_origin_of_raw_ptr(kind: mir::RawPtrKind) -> PointerOrigin {
        let mutable = matches!(kind, mir::RawPtrKind::Mut);
        PointerOrigin {
            kind: MirPointerKind::from_raw_mutability(mutable),
            mutable,
        }
    }

    /// From a raw-pointer `Mutability` (e.g. an `AggregateKind::RawPtr`).
    pub(crate) fn pointer_origin_of_raw_mutability(m: mir::Mutability) -> PointerOrigin {
        let mutable = m == mir::Mutability::Mut;
        PointerOrigin {
            kind: MirPointerKind::from_raw_mutability(mutable),
            mutable,
        }
    }

    /// Propagation, never strengthening: reuse the kind and mutability an
    /// in-IR fat-pointer carrier already holds (`Erased` stays `Erased`).
    pub(crate) fn pointer_origin_of_slice_carrier(slice: &MirSliceType) -> PointerOrigin {
        PointerOrigin {
            kind: slice.pointer_kind(),
            mutable: slice.is_mutable(),
        }
    }

    /// Thin-pointer analog of [`pointer_origin_of_slice_carrier`]: address
    /// projections and address-space retypes copy the base pointer's kind
    /// verbatim.
    pub(crate) fn pointer_origin_of_ptr_carrier(ptr: &MirPtrType) -> PointerOrigin {
        PointerOrigin {
            kind: ptr.pointer_kind(),
            mutable: ptr.is_mutable(),
        }
    }

    // --- named ABI rules -----------------------------------------------------

    /// ABI RULE: `cuda_device::DynamicSharedArray`'s public API hands out
    /// `*mut T`. Extern-shared storage and its internal arithmetic stay
    /// `Erased`; this is the one boundary where the result becomes `RawMut`.
    pub(crate) fn abi_dynamic_shared_array_result() -> PointerOrigin {
        PointerOrigin {
            kind: MirPointerKind::RawMut,
            mutable: true,
        }
    }

    /// ABI RULE: `cuda_device::DisjointSlice<'_, T>` stores its data pointer
    /// as `*mut T`, so its data/element pointers are always `RawMut`.
    pub(crate) fn abi_disjoint_slice_data_ptr() -> PointerOrigin {
        PointerOrigin {
            kind: MirPointerKind::RawMut,
            mutable: true,
        }
    }

    // --- minting: the only importer path to kinded pointer/slice types ------

    /// Kinded pointer with an explicit address space.
    pub(crate) fn mint_ptr_type(
        ctx: &mut Context,
        pointee: TypeHandle,
        address_space: u32,
        origin: PointerOrigin,
    ) -> TypedHandle<MirPtrType> {
        MirPtrType::get_with_kind(ctx, pointee, origin.mutable, address_space, origin.kind)
    }

    /// Kinded pointer in the generic address space (0).
    pub(crate) fn mint_generic_ptr_type(
        ctx: &mut Context,
        pointee: TypeHandle,
        origin: PointerOrigin,
    ) -> TypedHandle<MirPtrType> {
        MirPtrType::get_generic_with_kind(ctx, pointee, origin.mutable, origin.kind)
    }

    /// Kinded pointer in the shared-memory address space (3).
    pub(crate) fn mint_shared_ptr_type(
        ctx: &mut Context,
        pointee: TypeHandle,
        origin: PointerOrigin,
    ) -> TypedHandle<MirPtrType> {
        MirPtrType::get_shared_with_kind(ctx, pointee, origin.mutable, origin.kind)
    }

    /// Kinded slice/fat-pointer carrier.
    pub(crate) fn mint_slice_type(
        ctx: &mut Context,
        element_ty: TypeHandle,
        origin: PointerOrigin,
    ) -> TypedHandle<MirSliceType> {
        MirSliceType::get_with_mutability_and_kind(ctx, element_ty, origin.mutable, origin.kind)
    }
}

pub(crate) use pointer_origin::*;

// ============================================================================
// Kernel reference validity facts (typed rustc_public reads only)
// ============================================================================

/// Validity facts that are safe to expose on a Rust-reference kernel parameter.
///
/// Presence of this fact proves non-nullness. `pointee_alignment` is rustc's
/// ABI alignment for the pointee represented by the physical data pointer.
/// No aliasing, lifetime, or dereferenceability guarantee is encoded here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReferenceParamValidity {
    pub(crate) pointee_alignment: u64,
}

/// Return validity facts for the currently audited Rust-reference kernel scope.
///
/// The only accepted evidence is the typed `rustc_public` type/layout API:
///
/// * `RigidTy::Ref` proves that the source value is a Rust reference and thus
///   that its data pointer is non-null.
/// * a sized pointee's `Ty::layout()` supplies its ABI alignment;
/// * for `&[T]` / `&mut [T]`, the physical data pointer points at `T`, so the
///   element layout supplies the alignment.
///
/// Other DSTs, raw pointers, ADTs such as `DisjointSlice`, and any type whose
/// required typed layout is unavailable fail closed with no fact.
pub(crate) fn reference_param_validity(
    ty: &rustc_public::ty::Ty,
) -> Option<ReferenceParamValidity> {
    use rustc_public::ty::{RigidTy, TyKind};

    // Reuse #1186's typed pointer-origin oracle rather than independently
    // recognizing reference provenance here. Raw pointers are therefore
    // rejected by the same rustc_public-derived classification that mints
    // concrete MIR pointer kinds.
    let (pointee, origin) = pointer_origin_of_ty(ty)?;
    if !origin.is_reference() {
        return None;
    }

    let alignment = match pointee.kind() {
        TyKind::RigidTy(RigidTy::Slice(element)) => {
            let shape = element.layout().ok()?.shape();
            if !shape.is_sized() {
                return None;
            }
            shape.abi_align
        }
        _ => {
            let shape = pointee.layout().ok()?.shape();
            if !shape.is_sized() {
                return None;
            }
            shape.abi_align
        }
    };

    (alignment != 0 && alignment.is_power_of_two()).then_some(ReferenceParamValidity {
        pointee_alignment: alignment,
    })
}

// ============================================================================
// Constant facts (typed reads; exact or hard error)
// ============================================================================

/// Evaluate a const generic to a target usize. Translation runs on
/// monomorphized bodies, so a const that doesn't evaluate means
/// polymorphic MIR reached codegen: hard error, never a guess.
///
/// `what` names the const generic in the error (e.g. `"SharedArray N"`);
/// `loc` attaches a source location when the caller has one.
pub(crate) fn eval_usize_const(
    c: &rustc_public::ty::TyConst,
    what: &str,
    loc: Option<&Location>,
) -> TranslationResult<u64> {
    c.eval_target_usize().map_err(|e| {
        let err = TranslationErr::unsupported(format!(
            "{what} const generic did not evaluate to a target usize: {e:?}"
        ));
        match loc {
            Some(loc) => input_error!(loc.clone(), err),
            None => input_error_noloc!(err),
        }
    })
}

/// Read an enum constant's tag and map it to `(variant index, variant name)`.
///
/// Only valid for direct-tagged enums (e.g. `#[repr(u8)]` fieldless enums like
/// `cuda_device::atomic::AtomicOrdering`) where the constant's allocation IS
/// the tag:
///
/// ```text
/// alloc bytes [0x02] --read_uint--> tag 2 --discriminant match--> (2, "Release")
/// ```
///
/// Niche-encoded enums store no direct tag, so this mapping would be wrong for
/// them; the discriminant match errors out instead of guessing. Every failure
/// here is a hard error: inventing a variant would silently change semantics
/// (the old Debug-string scrape defaulted to variant 0, turning SeqCst atomics
/// into Relaxed ones).
pub(crate) fn extract_enum_variant(
    mir_const: &rustc_public::ty::MirConst,
    loc: &Location,
) -> TranslationResult<(usize, String)> {
    use rustc_public::ty::{RigidTy, TyConstKind, TyKind, VariantIdx};

    let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = mir_const.ty().kind() else {
        return input_err!(
            loc.clone(),
            TranslationErr::type_error(format!(
                "expected an enum constant, got a constant of type {:?}",
                mir_const.ty()
            ))
        );
    };
    if adt_def.kind() != AdtKind::Enum {
        return input_err!(
            loc.clone(),
            TranslationErr::type_error(format!(
                "expected an enum constant, got a {:?} constant of type {}",
                adt_def.kind(),
                adt_def.trimmed_name()
            ))
        );
    }

    // Pull the raw tag out of the constant. read_uint() refuses uninitialized
    // bytes, so a malformed const errors instead of yielding a made-up tag.
    let (tag, tag_width_bytes) = match mir_const.kind() {
        ConstantKind::Allocated(alloc) => {
            let tag = alloc.read_uint().map_err(|e| {
                input_error!(
                    loc.clone(),
                    TranslationErr::invalid_op(format!(
                        "cannot read the tag of enum {} from its const allocation: {e:?}",
                        adt_def.trimmed_name()
                    ))
                )
            })?;
            (tag, alloc.bytes.len())
        }
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            TyConstKind::Value(_, alloc) => {
                let tag = alloc.read_uint().map_err(|e| {
                    input_error!(
                        loc.clone(),
                        TranslationErr::invalid_op(format!(
                            "cannot read the tag of enum {} from its const allocation: {e:?}",
                            adt_def.trimmed_name()
                        ))
                    )
                })?;
                (tag, alloc.bytes.len())
            }
            other => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "non-value type-level enum constant: {other:?}"
                    ))
                );
            }
        },
        ConstantKind::ZeroSized => {
            // No tag bytes to read; only unambiguous when there is exactly
            // one variant to pick.
            let variants = adt_def.variants();
            if variants.len() == 1 {
                return Ok((0, variants[0].name()));
            }
            return input_err!(
                loc.clone(),
                TranslationErr::invalid_op(format!(
                    "zero-sized constant of enum {} which has {} variants: no tag to read",
                    adt_def.trimmed_name(),
                    variants.len()
                ))
            );
        }
        ConstantKind::Unevaluated(unevaluated) => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "unevaluated enum constant {:?}; use a literal variant",
                    unevaluated.def
                ))
            );
        }
        ConstantKind::Param(param) => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "unmonomorphized const param {} used as an enum value",
                    param.name
                ))
            );
        }
    };

    if tag_width_bytes == 0 {
        return input_err!(
            loc.clone(),
            TranslationErr::invalid_op(format!(
                "empty const allocation for enum {}: no tag to read",
                adt_def.trimmed_name()
            ))
        );
    }

    // The stored tag is truncated to the physical tag width, while
    // discriminant_for_variant reports full-width values; compare masked
    // (same trick as rvalue's discriminant_to_variant_index).
    let mask = if tag_width_bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (tag_width_bytes * 8)) - 1
    };
    for (idx, variant) in adt_def.variants().iter().enumerate() {
        let discr = adt_def.discriminant_for_variant(VariantIdx::to_val(idx));
        if discr.val & mask == tag & mask {
            return Ok((idx, variant.name()));
        }
    }
    input_err!(
        loc.clone(),
        TranslationErr::invalid_op(format!(
            "enum {} has no variant with tag {tag}",
            adt_def.trimmed_name()
        ))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_constant_checks_freeze_lifetimes_and_marker_ownership() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_grid_constant_facts_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("fixture.rs");
        std::fs::write(
            &fixture,
            r#"
#![allow(dead_code)]
use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::AtomicU32;
pub struct Nested<T> { value: T }
pub fn ordinary(_: &[u32; 32]) {}
pub fn anonymous(_: &'_ [u32; 32]) {}
pub fn cell(_: &Cell<u32>) {}
pub fn nested_cell(_: &Nested<[Cell<u32>; 2]>) {}
pub fn unsafe_cell(_: &UnsafeCell<u32>) {}
pub fn atomic(_: &AtomicU32) {}
pub fn pointer_to_cell(_: &*const Cell<u32>) {}
pub fn reference_to_cell(_: &&Cell<u32>) {}
pub fn static_lifetime(_: &'static u32) {}
pub fn named_lifetime<'a>(_: &'a u32) {}
pub fn early_lifetime<'a: 'static>(_: &'a u32) {}
#[inline(never)]
pub fn generic<T>(_: &Nested<T>) {}
pub fn instantiate_cell(value: &Nested<Cell<u32>>) { generic(value); }
pub fn instantiate_plain(value: &Nested<u32>) { generic(value); }
#[inline(never)]
pub unsafe fn marker() {}
#[inline(never)]
pub fn direct(_: &u32) { unsafe { marker(); } }
#[inline(always)]
pub fn helper(_: &u32) { unsafe { marker(); } }
pub fn caller(value: &u32) { helper(value); }
#[inline(always)]
pub fn another_kernel(_: &u32) { unsafe { marker(); } }
pub fn kernel_caller(value: &u32) { another_kernel(value); }
"#,
        )
        .unwrap();
        let output = std::process::Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .unwrap();
        assert!(output.status.success());
        let sysroot = String::from_utf8(output.stdout).unwrap();
        let args = vec![
            "rustc".to_string(),
            "--edition=2024".to_string(),
            "--crate-type=rlib".to_string(),
            "--crate-name=grid_constant_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Zmir-opt-level=4".to_string(),
            "-Copt-level=3".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={}", sysroot.trim()),
            fixture.display().to_string(),
        ];
        let results = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    use rustc_public::ty::{RigidTy, TyKind};
                    let mut results = std::collections::BTreeMap::new();
                    for item in rustc_public::all_local_items() {
                        let name = item.name();
                        let short = name.rsplit("::").next().unwrap();
                        let Ok(instance) = mir::mono::Instance::try_from(item) else {
                            continue;
                        };
                        let Some(body) = instance.body() else {
                            continue;
                        };
                        let Some(parameter) = body.locals().get(1) else {
                            continue;
                        };
                        let TyKind::RigidTy(RigidTy::Ref(_, pointee, _)) = parameter.ty.kind()
                        else {
                            continue;
                        };
                        results.insert(
                            short.to_string(),
                            validate_grid_constant_parameter(&instance, 0, pointee),
                        );
                        if matches!(short, "instantiate_cell" | "instantiate_plain") {
                            let callee = body
                                .blocks
                                .iter()
                                .find_map(|block| {
                                    let mir::TerminatorKind::Call {
                                        func: mir::Operand::Constant(constant),
                                        ..
                                    } = &block.terminator.kind
                                    else {
                                        return None;
                                    };
                                    let TyKind::RigidTy(RigidTy::FnDef(def, args)) =
                                        constant.const_.ty().kind()
                                    else {
                                        return None;
                                    };
                                    mir::mono::Instance::resolve(def, &args).ok()
                                })
                                .expect("generic call survives");
                            let callee_body = callee.body().unwrap();
                            let TyKind::RigidTy(RigidTy::Ref(_, pointee, _)) =
                                callee_body.locals()[1].ty.kind()
                            else {
                                panic!("specialized reference parameter");
                            };
                            results.insert(
                                format!("{short}_generic"),
                                validate_grid_constant_parameter(&callee, 0, pointee),
                            );
                        }
                        if matches!(short, "direct" | "caller" | "kernel_caller") {
                            let (index, _) = body
                                .blocks
                                .iter()
                                .enumerate()
                                .find(|(_, block)| {
                                    matches!(
                                        block.terminator.kind,
                                        mir::TerminatorKind::Call { .. }
                                    )
                                })
                                .expect("marker call survives inlining");
                            results.insert(
                                format!("{short}_owner"),
                                validate_grid_constant_marker_owner(&instance, index, true),
                            );
                            results.insert(
                                format!("{short}_helper"),
                                validate_grid_constant_marker_owner(&instance, index, false),
                            );
                        }
                    }
                    std::ops::ControlFlow::<(), _>::Continue(results)
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .unwrap();
        std::fs::remove_dir_all(&root).ok();
        for name in [
            "ordinary",
            "anonymous",
            "pointer_to_cell",
            "reference_to_cell",
            "direct_owner",
            "instantiate_plain_generic",
        ] {
            assert_eq!(results[name], Ok(()), "{name}");
        }
        for name in [
            "cell",
            "nested_cell",
            "unsafe_cell",
            "atomic",
            "instantiate_cell_generic",
        ] {
            assert!(
                results[name]
                    .as_ref()
                    .unwrap_err()
                    .contains("interior mutability"),
                "{name}"
            );
        }
        for name in ["static_lifetime", "named_lifetime", "early_lifetime"] {
            assert!(
                results[name]
                    .as_ref()
                    .unwrap_err()
                    .contains("elided reference lifetime"),
                "{name}"
            );
        }
        for name in ["caller_owner", "kernel_caller_owner"] {
            assert!(
                results[name].as_ref().unwrap_err().contains("inlined from"),
                "{name}"
            );
        }
        assert!(
            results["direct_helper"]
                .as_ref()
                .unwrap_err()
                .contains("kernel entry")
        );
    }

    #[test]
    fn reference_param_validity_uses_only_typed_rustc_public_facts() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_reference_validity_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("reference_validity_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
#[repr(align(16))]
pub struct AlignedZst;

pub struct ByValue {
    pub pointer: *const f32,
}

pub fn reference_validity(
    _shared: &f32,
    _unique: &mut f32,
    _slice: &[f32],
    _unique_slice: &mut [f32],
    _align_one: &u8,
    _zst: &AlignedZst,
    _raw: *const f32,
    _by_value: ByValue,
) {}
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
            "--crate-name=reference_validity_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Zmir-opt-level=0".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let facts = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    use rustc_public::CrateDef;

                    let body = rustc_public::all_local_items()
                        .into_iter()
                        .find(|item| item.name().ends_with("::reference_validity"))
                        .and_then(|item| item.body())
                        .expect("fixture function body");
                    let facts = body
                        .locals()
                        .iter()
                        .skip(1)
                        .take(8)
                        .map(|decl| reference_param_validity(&decl.ty))
                        .collect::<Vec<_>>();
                    std::ops::ControlFlow::<(), _>::Continue(facts)
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            facts,
            vec![
                Some(ReferenceParamValidity {
                    pointee_alignment: 4,
                }),
                Some(ReferenceParamValidity {
                    pointee_alignment: 4,
                }),
                Some(ReferenceParamValidity {
                    pointee_alignment: 4,
                }),
                Some(ReferenceParamValidity {
                    pointee_alignment: 4,
                }),
                Some(ReferenceParamValidity {
                    pointee_alignment: 1,
                }),
                Some(ReferenceParamValidity {
                    pointee_alignment: 16,
                }),
                None,
                None,
            ]
        );
    }
}
