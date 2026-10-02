/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared state types for `dialect-mir` → LLVM dialect lowering.
//!
//! The DialectConversion framework handles value mapping and block mapping
//! automatically. This module provides the CUDA-specific state types that
//! certain ops need during conversion.

use rustc_hash::FxHashMap;

use crate::LoweringOptions;
use llvm_export::ops::DebugGlobalVariableInfo;
use pliron::{context::Context, identifier::Identifier, r#type::TypeHandle};

mod options_storage {
    pliron::dict_key!(LOWERING_OPTIONS_KEY, "cuda_oxide_mir_lower_options");
}

/// Store the options for the active lowering pass in pliron's per-compilation
/// context. Conversion interfaces only receive the context, so this keeps
/// policy explicit without consulting process-global environment variables in
/// individual operation converters.
pub(crate) fn set_lowering_options(ctx: &mut Context, options: LoweringOptions) {
    if let Some(index) = ctx
        .aux_data_map
        .get(&options_storage::LOWERING_OPTIONS_KEY)
        .copied()
    {
        ctx.aux_data[index] = Box::new(options);
    } else {
        let index = ctx.aux_data.insert(Box::new(options));
        ctx.aux_data_map
            .insert(options_storage::LOWERING_OPTIONS_KEY.clone(), index);
    }
}

/// Read options for the active lowering pass.
///
/// The default preserves the historical behavior for callers that use the
/// original `lower_mir_to_llvm` entry point.
pub(crate) fn lowering_options(ctx: &Context) -> LoweringOptions {
    ctx.aux_data_map
        .get(&options_storage::LOWERING_OPTIONS_KEY)
        .and_then(|index| ctx.aux_data[*index].downcast_ref::<LoweringOptions>())
        .copied()
        .unwrap_or_default()
}

/// Name for one generated module-scope global of a lowering-owned family
/// (`__shared_mem`, `__device_global`, `__dynamic_smem`).
///
/// `suffix` is what makes the name unique within one module: the per-module
/// counter for the counter-named families, the owning function's symbol for
/// the dynamic shared-memory pool. But every module starts its counters at
/// zero and a helper shared across crates keeps its owner symbol, so two
/// crates' PTX can define the same name — and `load_all_ptx_bundles_merged`
/// textually concatenates the bundles, where the duplicate definition fails
/// driver JIT compilation, or (for the dynamic pool's duplicate extern
/// declarations) silently keeps whichever alignment came first (#1277).
///
/// When the lowering options carry a module disambiguator, it is woven
/// between family and suffix so the name is unique across every compilation
/// whose PTX can end up in one merged module; without one, the historical
/// undecorated name is kept.
pub(crate) fn module_namespaced_symbol(ctx: &Context, family: &str, suffix: &str) -> String {
    match lowering_options(ctx).module_disambiguator {
        Some(disambiguator) => format!("{family}_{disambiguator:016x}_{suffix}"),
        None => format!("{family}_{suffix}"),
    }
}

/// Semantic class of a shared-memory declaration.
///
/// Static shared storage and CUDA's dynamic `extern __shared__` pool can have
/// the same element type, extent, and alignment, but they are not
/// interchangeable declarations.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SharedGlobalKind {
    /// A fixed-size `mir.shared_alloc` declaration.
    Static,
    /// A per-function dynamic `mir.extern_shared` declaration.
    DynamicExtern,
}

/// Storage declaration associated with one shared-memory allocation key.
#[derive(Clone, PartialEq, Eq)]
pub struct SharedGlobalDeclaration {
    /// Whether the declaration is fixed-size storage or dynamic extern storage.
    pub kind: SharedGlobalKind,
    /// MIR element type stored by the shared allocation.
    pub mir_elem_type: TypeHandle,
    /// Number of elements in the shared allocation.
    pub size: u64,
    /// Explicit alignment, or zero when natural alignment is requested.
    pub alignment: u64,
    /// Source-level debug identity carried on the allocation, when present.
    pub debug_info: Option<DebugGlobalVariableInfo>,
    /// Exported symbol of the function owning a function-local static, when
    /// present. The exporter scopes the variable's DIE to that function's
    /// `DISubprogram`.
    pub debug_owner_function: Option<String>,
}

impl SharedGlobalDeclaration {
    /// Equality over the physical storage identity only.
    ///
    /// Divergent debug identity on one key is not a storage conflict: the
    /// metadata is optional and fails open (dropped), while a physical
    /// mismatch fails the lowering.
    pub fn same_storage(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.mir_elem_type == other.mir_elem_type
            && self.size == other.size
            && self.alignment == other.alignment
    }
}

/// Lowered symbol and declaration associated with one shared-memory key.
#[derive(Clone)]
pub struct SharedGlobalRecord {
    /// LLVM symbol created for this allocation.
    pub symbol: Identifier,
    /// Storage declaration that every reuse of the key must match.
    pub declaration: SharedGlobalDeclaration,
}

/// Map from shared memory allocation keys to their checked declarations.
///
/// In CUDA kernels, shared memory is declared as module-level globals with
/// address space 3. When multiple operations reference the same shared allocation
/// (identified by a key string), they refer to the same global only when their
/// complete storage declarations agree.
pub type SharedGlobalsMap = FxHashMap<String, SharedGlobalRecord>;

/// Storage declaration associated with one ordinary device-global key.
#[derive(Clone, PartialEq, Eq)]
pub struct DeviceGlobalDeclaration {
    /// MIR semantic/storage-validation type of the global.
    pub mir_type: TypeHandle,
    /// Explicit allocation alignment, or zero when unspecified.
    pub alignment: u64,
    /// Lowered LLVM address space.
    pub addr_space: u32,
    /// Exact initialized byte image, when present.
    pub initializer_hex: Option<String>,
    /// Encoded symbolic relocations within the initializer, when present.
    pub initializer_relocations: Option<String>,
    /// Source-level debug identity carried on the allocation, when present.
    pub debug_info: Option<DebugGlobalVariableInfo>,
    /// Whether lowering exports the storage as immutable.
    pub immutable: bool,
}

/// Lowered symbol and declaration associated with one device-global key.
#[derive(Clone)]
pub struct DeviceGlobalRecord {
    /// LLVM symbol created for this allocation.
    pub symbol: Identifier,
    /// Storage declaration that every reuse of the key must match.
    pub declaration: DeviceGlobalDeclaration,
}

/// Map from ordinary device static keys to their checked declarations.
///
/// Ordinary Rust `static` / `static mut` values used from device code live in
/// CUDA global memory (address space 1), not shared memory. A key names one
/// allocation; conflicting declarations fail instead of silently reusing a
/// symbol with the wrong type, alignment, address space, initializer, or
/// debug identity.
pub type DeviceGlobalsMap = FxHashMap<String, DeviceGlobalRecord>;

/// Tracking for dynamic shared memory alignment per lowered function.
///
/// Maps function name to `(symbol_name, max_alignment)`.
///
/// Each function that owns a dynamic shared-memory access gets a symbol. Before
/// conversion, the pass combines the alignment requested by the function body
/// with every propagated launch-contract marker that can reach it. This
/// ensures a helper shared by several kernels uses their strongest requirement.
pub type DynamicSmemAlignmentMap = FxHashMap<String, (pliron::identifier::Identifier, u64)>;
