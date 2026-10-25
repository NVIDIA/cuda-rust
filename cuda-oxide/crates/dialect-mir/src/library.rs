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

/// Source descriptor policy. Zero capacity/alignment retain inferred meaning.
#[format]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LibraryStoragePolicy {
    pub capacity: u32,
    pub alignment: u32,
    pub auto_sync: bool,
    pub exclusive: bool,
}

/// Explicit descriptor use is distinct from managed implicit scratch.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryStorageArg {
    Implicit,
    Explicit(LibraryStoragePolicy),
}

/// Evaluated source array layout, including requested reservation alignment.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LibraryReserveLayout {
    pub scalar: LibraryScalar,
    pub count: u32,
    pub bytes: u32,
    pub alignment: u32,
}

/// Native contracts retained for collective admission.
#[format]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryNativeMetadata {
    Raw {
        spec: NativeSpec,
        block_threads: u32,
    },
    Callback {
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
    CheckedWarp {
        spec: NativeSpec,
        width: u32,
        access: LibraryAccess,
        witness: LibraryWitness,
    },
    CheckedBlockStorage {
        spec: NativeSpec,
        access: LibraryAccess,
        witness: LibraryWitness,
        storage: LibraryStorageArg,
    },
    CheckedBlockPayload {
        spec: NativeSpec,
        storage: LibraryStorageArg,
    },
    CheckedBlockPayloadCallback {
        spec: NativeSpec,
        storage: LibraryStorageArg,
    },
    CheckedWarpPayload {
        spec: NativeSpec,
        width: u32,
    },
}
impl LibraryNativeMetadata {
    pub fn spec(self) -> NativeSpec {
        match self {
            Self::Raw { spec, .. }
            | Self::Callback { spec, .. }
            | Self::CheckedBlock { spec }
            | Self::CheckedBlockResource { spec, .. }
            | Self::CheckedWarp { spec, .. }
            | Self::CheckedBlockStorage { spec, .. }
            | Self::CheckedBlockPayload { spec, .. }
            | Self::CheckedBlockPayloadCallback { spec, .. }
            | Self::CheckedWarpPayload { spec, .. } => spec,
        }
    }
}

/// Genuine definition class, before source calls are expanded or erased.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryCallbackDefinition {
    Rust,
    Cuda,
    Foreign,
    Intrinsic,
}

/// Concrete dispatch identity; the two named shims have separately rooted receivers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryCallbackDispatch {
    Direct,
    Indirect,
    FunctionItemShim,
    ClosureOnceShim,
}

/// Original monomorphically reachable terminators relevant to callback admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibraryCallbackCall {
    Direct {
        definition: LibraryCallbackDefinition,
        diagnostic: String,
    },
    Indirect,
    InlineAssembly,
}

/// Source distinctions consumed by callback admission, independent of Rust types.
/// Local nominal types include unused declarations; calls include only original
/// monomorphically reachable terminators. This is not a block or value graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryCallbackFunction {
    pub diagnostic: String,
    pub export: Option<String>,
    pub definition: LibraryCallbackDefinition,
    pub dispatch: LibraryCallbackDispatch,
    pub has_body: bool,
    pub cuda_local_types: Vec<String>,
    pub calls: Vec<LibraryCallbackCall>,
    pub callees: Vec<String>,
}

/// Genuine nominal environment shape, tied to the original selected callback site.
/// Capture contents are not values or uniformity claims in this source record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LibraryCallbackEnvironment {
    FunctionItem,
    Closure {
        capture_count: usize,
        /// Actual upvar type when exactly one capture exists, not a value proof.
        single_capture: Option<LibraryScalar>,
    },
}

/// Original selected callback roots and complete collector identities, including
/// callees that have no imported symbol. Missing records cannot imply admission.
#[derive(Clone, Debug, Default)]
pub struct LibraryCallbackSources {
    pub checked_environments:
        std::collections::BTreeMap<LibraryCallSite, LibraryCallbackEnvironment>,
    pub roots: std::collections::BTreeMap<LibraryCallSite, String>,
    pub functions: std::collections::BTreeMap<String, LibraryCallbackFunction>,
}

/// Genuine source-native semantics; this role does not prove participation.
#[pliron_attr(name = "mir.library_role", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryRoleAttr {
    /// Original position before evaluating a native marker's runtime operands.
    ///
    /// This has the same site/callee as its paired native operation. It carries
    /// no participation or completion authority. Lowering consumes both and
    /// emits any planned pre-call barrier here, before the operand producers.
    NativeOperandBoundary,
    /// Original native marker semantics, before tuple flattening or scratch insertion.
    Native(LibraryNativeMetadata),
    /// Genuine pure x-fastest group index, retained until after admission.
    WarpGroupIndex { width: u32 },
    /// Genuine low-lane membership predicate for a physical Warp prefix.
    WarpPrefixMember { width: u32 },
    /// Genuine canonical aligned tile synchronization, independent of native plans.
    SyncWarp { width: u32 },
    /// Create a descriptor with a new source identity.
    StorageNew(LibraryStoragePolicy),
    /// Consume a descriptor and change its policy, preserving its identity.
    StorageReconfigure {
        old: LibraryStoragePolicy,
        new: LibraryStoragePolicy,
    },
    /// Borrow a descriptor to create a typed reservation token.
    StorageReserve {
        policy: LibraryStoragePolicy,
        layout: LibraryReserveLayout,
    },
    /// Consume a reservation token; neither synchronization nor descriptor release.
    StorageRelease,
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
    StorageCandidate,
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

/// Compiler-owned operation at one exact MIR terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibraryOperation {
    Native,
    StorageNew,
    StorageReconfigure,
    StorageReserve,
    StorageRelease,
    Redirect,
}

/// One monomorphized call whose Rust tuple and provider ABI already agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryCallPlan {
    pub operation: LibraryOperation,
    pub storage_witness: bool,
    pub scratch_offset: u32,
    /// None means Block; Some(width) is a validated aligned tile or low prefix.
    pub warp_width: Option<u32>,
    /// Padded bytes per replica; zero for nonreplicated views.
    pub scratch_stride: u32,
    pub planned_root: Option<String>,
    pub barrier_before: bool,
    pub barrier_after: bool,
    /// Exact external symbol emitted by the native provider.
    pub symbol: String,
    /// Expected marker identity at this exact MIR terminator.
    pub marker: String,
    /// Split uniform/payload operands of the checked full-block marker.
    pub checked_block: bool,
    /// An additional typed memory witness, consumed only by compiler checks.
    pub resource_witness: bool,
    /// Scalar-only controls and local input/output pointers, without memory.
    pub payload_only: bool,
    /// Append the borrowed callback environment after injected scratch.
    pub callback: bool,
    /// Size of the assigned shared scratch view in bytes.
    pub scratch_bytes: u32,
    /// Required alignment of the shared scratch base.
    pub scratch_alignment: u32,
}

impl LibraryCallPlan {
    pub fn internal(operation: LibraryOperation, marker: String) -> Self {
        Self {
            operation,
            marker,
            symbol: String::new(),
            storage_witness: false,
            scratch_offset: 0,
            warp_width: None,
            scratch_stride: 0,
            planned_root: None,
            barrier_before: false,
            barrier_after: false,
            checked_block: false,
            resource_witness: false,
            payload_only: false,
            callback: false,
            scratch_bytes: 0,
            scratch_alignment: 1,
        }
    }
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
