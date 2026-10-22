/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Closed library roles retained until mandatory MIR lowering.

use pliron::{attribute::Attribute, context::Context};
use pliron_derive::{format, pliron_attr};

/// Fixed-width numeric source type; pointer fields retain their own types.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryScalar {
    Signed(u16),
    Unsigned(u16),
    Float(u16),
}

/// Native specialization facts independent of any provider response.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NativeSpec {
    pub scalar: LibraryScalar,
    pub items: u32,
}

/// Global-memory access selected by a genuine resource marker.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryAccess {
    Read,
    Write,
}

/// Nominal witness classification captured before its type is erased.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryWitness {
    ReadOnlySlice,
    CollectiveCarrier(u64),
}

/// Native contracts supported by the checked-Block prefix.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryNativeMetadata {
    Raw {
        spec: NativeSpec,
        block_threads: u32,
    },
    CheckedBlock {
        spec: NativeSpec,
    },
    CheckedBlockResource {
        spec: NativeSpec,
        access: LibraryAccess,
        witness: LibraryWitness,
    },
}
impl LibraryNativeMetadata {
    pub fn spec(self) -> NativeSpec {
        match self {
            Self::Raw { spec, .. }
            | Self::CheckedBlock { spec }
            | Self::CheckedBlockResource { spec, .. } => spec,
        }
    }
}

/// Genuine source-native semantics; this role does not prove participation.
#[pliron_attr(name = "mir.library_role", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryRoleAttr {
    Native(LibraryNativeMetadata),
}

/// One source terminator in a concrete emitted Rust body.
///
/// `block` is the original MIR block index, not the block's position after
/// transformations. Context-specialized helpers have distinct caller symbols.
#[pliron_attr(name = "mir.library_site", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LibraryCallSite {
    pub caller: String,
    pub block: usize,
}

/// Original source identity used by diagnostics and contextual helper naming.
#[pliron_attr(name = "mir.library_function_source", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LibraryFunctionSourceAttr {
    pub export: String,
    pub mangled: String,
    pub diagnostic: String,
    pub checked_ancestor: bool,
}

/// Authenticated source type class for one original kernel argument.
#[pliron_attr(name = "mir.library_root_argument", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryRootArgumentClass {
    Scalar,
    ResourceCandidate,
    Kernel,
}

/// First-selected source edge, distinct from the complete allowed-target set.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryCallKind {
    Direct,
    Callable,
    Drop,
}

/// Exact source edge selected in the original collector order.
#[pliron_attr(name = "mir.library_call_target", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LibraryCallTargetAttr {
    pub kind: LibraryCallKind,
    pub export: String,
    pub marker: String,
}

/// Source call classifications whose ordinary expansion erases their identity.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibrarySourceEffectKind {
    BlockSync,
    BlockQuery,
    LaneQuery,
    DynamicShared,
    UnitHint,
    ScalarIntrinsic,
    PointerOffset,
    UnsupportedIntrinsic,
    UnresolvedCall,
    NontrivialDrop,
    InlineAssembly,
    Abort,
}

/// Genuine source event retained independently of expanded instruction names.
#[pliron_attr(name = "mir.library_source_effect", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LibrarySourceEffectAttr {
    pub site: LibraryCallSite,
    pub kind: LibrarySourceEffectKind,
}

/// Only the source mutation distinctions consumed by library validation.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryWriteKind {
    Bulk,
    Indirect { nominal_protected: bool },
}

/// Original statement identity and nominal restriction; no alias verdict.
#[pliron_attr(name = "mir.library_write_source", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LibraryWriteSourceAttr {
    pub site: LibraryCallSite,
    pub statement: usize,
    pub kind: LibraryWriteKind,
}

/// Actual successor ordinal for an original exhaustive-enum impossible edge.
#[pliron_attr(name = "mir.library_impossible_successor", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LibraryImpossibleSuccessorAttr(pub u32);

/// Stamp an imported terminator with its original, not physical, source block.
pub fn set_library_source_site(
    ctx: &mut Context,
    op: pliron::context::Ptr<pliron::operation::Operation>,
    site: LibraryCallSite,
) {
    op.deref_mut(ctx).attributes.set(
        pliron::identifier::Identifier::try_new("library_terminator_site".to_string()).unwrap(),
        site,
    );
}

/// Original block identity for an actual imported terminator.
pub fn library_source_site(
    ctx: &Context,
    op: pliron::context::Ptr<pliron::operation::Operation>,
) -> Option<LibraryCallSite> {
    op.deref(ctx)
        .attributes
        .get::<LibraryCallSite>(
            &pliron::identifier::Identifier::try_new("library_terminator_site".to_string())
                .unwrap(),
        )
        .cloned()
}

/// Mark only the actual successor corresponding to a proved source impossible edge.
pub fn set_library_impossible_successor(
    ctx: &mut Context,
    op: pliron::context::Ptr<pliron::operation::Operation>,
    index: u32,
) {
    op.deref_mut(ctx).attributes.set(
        pliron::identifier::Identifier::try_new("library_impossible_successor".to_string())
            .unwrap(),
        LibraryImpossibleSuccessorAttr(index),
    );
}

/// The original impossible edge, if one exists on this actual terminator.
pub fn library_impossible_successor(
    ctx: &Context,
    op: pliron::context::Ptr<pliron::operation::Operation>,
) -> Option<u32> {
    op.deref(ctx)
        .attributes
        .get::<LibraryImpossibleSuccessorAttr>(
            &pliron::identifier::Identifier::try_new("library_impossible_successor".to_string())
                .unwrap(),
        )
        .map(|attr| attr.0)
}

/// One monomorphized call whose Rust tuple and provider ABI already agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryCallPlan {
    /// Exact external symbol emitted by the native provider.
    pub symbol: String,
    /// Expected marker identity at this exact MIR terminator.
    pub marker: String,
    /// Split uniform/payload operands of the checked full-block marker.
    pub checked_block: bool,
    /// An additional typed memory witness, consumed only by compiler checks.
    pub resource_witness: bool,
    /// Size of the assigned shared scratch view in bytes.
    pub scratch_bytes: u32,
    /// Required alignment of the shared scratch base.
    pub scratch_alignment: u32,
}

pub(crate) fn register(ctx: &mut Context) {
    LibraryRoleAttr::register(ctx);
    LibraryCallSite::register(ctx);
    LibraryFunctionSourceAttr::register(ctx);
    LibraryRootArgumentClass::register(ctx);
    LibraryCallTargetAttr::register(ctx);
    LibrarySourceEffectAttr::register(ctx);
    LibraryWriteSourceAttr::register(ctx);
    LibraryImpossibleSuccessorAttr::register(ctx);
}
