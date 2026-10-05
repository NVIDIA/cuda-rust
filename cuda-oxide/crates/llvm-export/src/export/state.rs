/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exporter state and kernel bookkeeping.

use pliron::{basic_block::BasicBlock, context::Ptr, r#type::TypeHandle, value::Value};
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::ops::{
    DebugGlobalVariableInfo, DebugLocalTypeKind, DebugLocalVariableInfo, DebugSourceScopeMap,
};

use super::{
    config::{DebugKind, FunctionLocalStaticPlacement, NvvmIrDialect},
    externs::DeviceExternDecl,
};

/// Map from block to its predecessors with the values passed to each predecessor.
/// Used for PHI node generation when exporting to LLVM IR.
pub(super) type PredecessorMap = FxHashMap<Ptr<BasicBlock>, Vec<(Ptr<BasicBlock>, Vec<Value>)>>;

/// Cluster dimensions for a kernel (from `#[cluster(x,y,z)]` attribute).
pub(super) struct KernelClusterConfig {
    pub(super) name: String,
    pub(super) dim_x: u32,
    pub(super) dim_y: u32,
    pub(super) dim_z: u32,
}

/// Block geometry declared for a kernel entry.
///
/// ptxas rejects an entry carrying both `.maxntid` and `.reqntid`, so an entry
/// declares one or the other. An exact shape is the stronger statement and
/// displaces a thread maximum, which is why these are alternatives rather than
/// two fields.
#[derive(Clone, Copy)]
pub(super) enum KernelBlockGeometry {
    /// Maximum threads per block from `#[launch_bounds(max, _)]`, emitted as
    /// `maxntid*`. Bounds the product `x * y * z` and says nothing per axis.
    MaxThreads(u32),
    /// Exact block shape from `#[launch_contract(block = (x, y, z))]`, emitted
    /// as `reqntid*`. The CUDA driver enforces it per axis at launch.
    ExactBlock(u32, u32, u32),
}

/// Launch geometry for a kernel, from `#[launch_bounds(max, min)]` and from an
/// exact `#[launch_contract(block = (x, y, z))]`.
///
/// A kernel declaring neither is never recorded.
pub(super) struct KernelLaunchBounds {
    pub(super) name: String,
    pub(super) geometry: KernelBlockGeometry,
    pub(super) min_blocks: Option<u32>, // None if not specified (0 in attribute)
}

/// Basic kernel info (for backends that need annotations for all kernels).
pub(super) struct KernelInfo {
    pub(super) name: String,
}

/// One direct aggregate argument/return whose source ABI alignment exceeds
/// the natural alignment representable by its LLVM type.
///
/// NVVM's `align` function property numbers the return as position 0 and
/// arguments from 1.
pub(super) struct FunctionAbiAlignment {
    pub(super) name: String,
    pub(super) position: u16,
    pub(super) alignment: u16,
}

/// Kernel parameters whose by-value storage remains in kernel parameter space
/// and is shared by the whole grid. NVVM numbers parameters from one.
pub(super) struct KernelGridConstants {
    pub(super) name: String,
    pub(super) positions: Vec<u32>,
}

/// The by-value storage type is part of a kernel's ABI even though the
/// compiler's pointer type deliberately erases pointees. Retain it before
/// exporting any body so forward references see the same signature.
#[derive(Clone, Copy)]
pub(super) struct GridConstantParameter {
    pub(super) index: usize,
    pub(super) pointee: TypeHandle,
    pub(super) alignment: u64,
}

#[derive(Clone, Copy)]
pub(super) struct GlobalSymbolInfo {
    pub(super) value_type: TypeHandle,
    pub(super) address_space: u32,
}

#[derive(Clone)]
pub(super) struct GlobalSourceInfo {
    pub(super) symbol: String,
    pub(super) value_type: TypeHandle,
    pub(super) address_space: u32,
    pub(super) initializer_size: Option<u64>,
}

pub(super) struct ModuleExportState<'a> {
    pub(super) ctx: &'a pliron::context::Context,
    /// Track if any convergent operations were used (for emitting attributes section)
    pub(super) convergent_used: bool,
    /// Track kernels with cluster configurations for nvvm.annotations metadata
    pub(super) cluster_kernels: Vec<KernelClusterConfig>,
    /// Track kernels with launch bounds for nvvm.annotations metadata
    pub(super) launch_bounds_kernels: Vec<KernelLaunchBounds>,
    /// Track ALL kernels (for backends that require annotations for every kernel)
    pub(super) all_kernels: Vec<KernelInfo>,
    /// Direct aggregate argument/return alignments that LLVM structural types
    /// cannot encode and NVVM therefore requires as `"align"` annotations.
    pub(super) function_abi_alignments: Vec<FunctionAbiAlignment>,
    /// Parameters carrying LLVM `byval` plus NVVM `grid_constant` semantics.
    pub(super) grid_constant_kernels: Vec<KernelGridConstants>,
    /// Whether to print `ptx_kernel` on kernel definitions.
    pub(super) emit_ptx_kernel_keyword: bool,
    /// Track device function names for @llvm.used (standalone device fn compilation)
    pub(super) device_functions: Vec<String>,
    /// Defined globals retain external linkage because CUDA host code can
    /// resolve them by name (for example through `cuModuleGetGlobal`).
    pub(super) public_globals: Vec<String>,
    /// Globals explicitly consumed outside device code and therefore rooted in
    /// `@llvm.used` so materialization cannot discard them.
    pub(super) retained_globals: Vec<String>,
    /// Emitted function signatures keyed by their final, prefix-stripped name.
    pub(super) function_types: FxHashMap<String, TypeHandle>,
    pub(super) function_grid_constants: FxHashMap<String, Vec<GridConstantParameter>>,
    /// Original pliron symbol spelling for each final exported function name.
    /// Device-extern declarations can only suppress an exact-name declaration;
    /// a prefixed alias would otherwise emit a second definition/declaration.
    pub(super) function_source_names: FxHashMap<String, String>,
    /// Functions with a region are emitted independently and therefore cannot
    /// also be supplied as an external side-table declaration.
    pub(super) function_definitions: HashSet<String>,
    /// Exact device-extern ABI signatures. Lowered pliron pointers are opaque;
    /// this side table retains pointees for legacy LLVM 7 declarations and
    /// call-boundary adapters.
    pub(super) device_externs: FxHashMap<String, DeviceExternDecl>,
    /// Global value types/address spaces, indexed before any function body is
    /// emitted so `addressof` is independent of top-level textual order.
    pub(super) global_symbols: FxHashMap<String, GlobalSymbolInfo>,
    /// Device globals indexed by their stable rustc source key.
    ///
    /// Relocation metadata refers to this key because ordinary globals receive
    /// generated LLVM symbol names during MIR lowering.
    pub(super) global_sources: FxHashMap<String, GlobalSourceInfo>,
    /// Next `!N` metadata ID in this module.
    ///
    /// LLVM has one flat numbered metadata namespace per module. Today this is
    /// used for NVVM annotations/version nodes; debug-info nodes will use the
    /// same counter so the exporter never has to guess which IDs are free.
    next_metadata_id: usize,
    /// Which debug metadata tier this export should emit.
    pub(super) debug_kind: DebugKind,
    /// Where function-local statics are retained (per the consuming LLVM).
    pub(super) debug_function_local_static_placement: FunctionLocalStaticPlacement,
    /// NVVM textual dialect, or `None` for the ordinary PTX/llc path.
    pub(super) nvvm_ir_dialect: Option<NvvmIrDialect>,
    /// The single compile unit used for Stage 2 line-table debug info.
    pub(super) debug_compile_unit: Option<usize>,
    /// `DIFile` nodes keyed by the source path they describe.
    pub(super) debug_files: FxHashMap<PathBuf, usize>,
    /// Shared empty function type used by all line-table-only subprograms.
    pub(super) debug_subroutine_type: Option<usize>,
    /// `DISubprogram` file paths, used to create file-correct nested scopes.
    pub(super) debug_subprogram_files: FxHashMap<usize, PathBuf>,
    /// The one real `DISubprogram` allocated for each exported function name.
    /// AS3 globals are emitted before functions, so their owners are reserved
    /// in a module prepass and reused when the definition is written.
    pub(super) debug_function_subprograms: FxHashMap<String, usize>,
    /// Source scope for functions that own function-local shared statics.
    pub(super) debug_shared_function_scopes: FxHashMap<String, DebugSharedFunctionScope>,
    /// Fallback line/column for calls that LLVM requires to have a location.
    pub(super) debug_subprogram_fallbacks: FxHashMap<usize, (i32, i32)>,
    /// `DILexicalBlockFile` nodes keyed by `(parent scope, file path)`.
    pub(super) debug_file_scopes: FxHashMap<(usize, PathBuf), usize>,
    /// `DILexicalBlock` nodes keyed by `(parent scope, file, line, column)`.
    pub(super) debug_lexical_blocks: FxHashMap<(usize, PathBuf, i32, i32), usize>,
    /// Inlined callee `DISubprogram` nodes keyed by `(name, file, line)`.
    pub(super) debug_inlined_subprograms: FxHashMap<(String, PathBuf, i32), usize>,
    /// MIR source-scope tables keyed by the owning function `DISubprogram`.
    pub(super) debug_source_scope_maps: FxHashMap<usize, DebugSourceScopeMap>,
    /// Resolved MIR source scopes keyed by `(function DISubprogram, source scope id)`.
    pub(super) debug_resolved_source_scopes: FxHashMap<(usize, u32), ResolvedDebugScope>,
    /// `DILocation` nodes keyed by `(scope, line, column, inlined-at location)`.
    pub(super) debug_locations: FxHashMap<(usize, i32, i32, Option<usize>), usize>,
    /// `DIType` nodes keyed by the simple debug type they describe.
    pub(super) debug_types: FxHashMap<DebugLocalTypeKind, usize>,
    /// Uniqued nested `DINamespace` nodes, keyed by parent scope and segment.
    pub(super) debug_namespaces: FxHashMap<(Option<usize>, String), usize>,
    /// Global expressions already created for a physical linkage name.
    /// A repeated linkage must carry the exact same source identity.
    pub(super) debug_global_variables:
        FxHashMap<String, (DebugGlobalVariableInfo, u32, Option<String>, usize)>,
    /// Module globals retained by the compile unit.
    pub(super) debug_global_expressions: Vec<usize>,
    /// Function-local static expressions retained by their owning
    /// `DISubprogram` (keyed by its metadata id), under the
    /// [`FunctionLocalStaticPlacement::SubprogramRetainedNodes`] placement.
    pub(super) debug_subprogram_retained_globals: FxHashMap<usize, Vec<usize>>,
    /// Whether the compile unit's immutable globals tuple has been finalized.
    pub(super) debug_globals_finalized: bool,
    /// `DILocalVariable` nodes keyed by scope, source line, and local identity.
    pub(super) debug_local_variables:
        FxHashMap<(usize, PathBuf, i32, DebugLocalVariableInfo), usize>,
    /// Numbered debug metadata definitions, in allocation order.
    pub(super) debug_nodes: Vec<(usize, String)>,
    /// Whether any function emitted `llvm.dbg.declare`.
    pub(super) debug_declare_used: bool,
    /// Whether any function emitted `llvm.dbg.value`.
    pub(super) debug_value_used: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ResolvedDebugScope {
    pub(super) scope: usize,
    pub(super) inlined_at: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DebugSharedFunctionScope {
    pub(super) namespace: Vec<String>,
    pub(super) name: String,
}

impl<'a> ModuleExportState<'a> {
    pub(super) fn new(
        ctx: &'a pliron::context::Context,
        emit_ptx_kernel_keyword: bool,
        debug_kind: DebugKind,
        nvvm_ir_dialect: Option<NvvmIrDialect>,
        debug_function_local_static_placement: FunctionLocalStaticPlacement,
    ) -> Self {
        Self {
            ctx,
            convergent_used: false,
            cluster_kernels: Vec::new(),
            launch_bounds_kernels: Vec::new(),
            all_kernels: Vec::new(),
            function_abi_alignments: Vec::new(),
            grid_constant_kernels: Vec::new(),
            emit_ptx_kernel_keyword,
            device_functions: Vec::new(),
            public_globals: Vec::new(),
            retained_globals: Vec::new(),
            function_types: FxHashMap::default(),
            function_grid_constants: FxHashMap::default(),
            function_source_names: FxHashMap::default(),
            function_definitions: HashSet::new(),
            device_externs: FxHashMap::default(),
            global_symbols: FxHashMap::default(),
            global_sources: FxHashMap::default(),
            next_metadata_id: 0,
            debug_kind,
            debug_function_local_static_placement,
            nvvm_ir_dialect,
            debug_compile_unit: None,
            debug_files: FxHashMap::default(),
            debug_subroutine_type: None,
            debug_subprogram_files: FxHashMap::default(),
            debug_function_subprograms: FxHashMap::default(),
            debug_shared_function_scopes: FxHashMap::default(),
            debug_subprogram_fallbacks: FxHashMap::default(),
            debug_file_scopes: FxHashMap::default(),
            debug_lexical_blocks: FxHashMap::default(),
            debug_inlined_subprograms: FxHashMap::default(),
            debug_source_scope_maps: FxHashMap::default(),
            debug_resolved_source_scopes: FxHashMap::default(),
            debug_locations: FxHashMap::default(),
            debug_types: FxHashMap::default(),
            debug_namespaces: FxHashMap::default(),
            debug_global_variables: FxHashMap::default(),
            debug_global_expressions: Vec::new(),
            debug_subprogram_retained_globals: FxHashMap::default(),
            debug_globals_finalized: false,
            debug_local_variables: FxHashMap::default(),
            debug_nodes: Vec::new(),
            debug_declare_used: false,
            debug_value_used: false,
        }
    }

    pub(super) fn legacy_typed_pointers(&self) -> bool {
        self.nvvm_ir_dialect
            .is_some_and(NvvmIrDialect::uses_typed_pointers)
    }

    pub(super) fn function_type(&self, name: &str) -> Result<TypeHandle, String> {
        self.function_types
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing exported function type for `@{name}`"))
    }

    pub(super) fn device_extern(&self, name: &str) -> Option<&DeviceExternDecl> {
        self.device_externs.get(name)
    }

    pub(super) fn alloc_metadata_id(&mut self) -> usize {
        let id = self.next_metadata_id;
        self.next_metadata_id += 1;
        id
    }

    #[cfg(test)]
    pub(super) fn next_metadata_id(&self) -> usize {
        self.next_metadata_id
    }

    /// Check if a function name is a known convergent intrinsic.
    ///
    /// These intrinsics require warp-synchronous execution semantics and must
    /// be marked convergent to prevent LLVM from applying optimizations that
    /// would break GPU synchronization (like duplicating them into divergent branches).
    pub(super) fn is_convergent_intrinsic(name: &str) -> bool {
        // Block-level barriers
        name == "llvm.nvvm.barrier0"
            || name.starts_with("llvm.nvvm.barrier")
            // mbarrier operations
            || name.starts_with("llvm.nvvm.mbarrier")
            // Warp shuffles (though LLVM usually handles these)
            || name.starts_with("llvm.nvvm.shfl")
            // Warp votes
            || name.starts_with("llvm.nvvm.vote")
            // Warp match collectives (match.{any,all}.sync.*)
            || name.starts_with("llvm.nvvm.match")
            // Warp-level barrier (bar.warp.sync). Note the block-level
            // `barrier` prefix above does not match the `bar.` spelling.
            || name == "llvm.nvvm.bar.warp.sync"
            // Active-lane mask query; its result depends on warp convergence.
            || name == "llvm.nvvm.activemask"
            // Warp reductions (redux.sync.*)
            || name.starts_with("llvm.nvvm.redux")
            // Async bulk operations (TMA)
            || name.starts_with("llvm.nvvm.cp.async.bulk")
            // Warpgroup register reconfiguration
            // (setmaxnreg.{inc,dec}.sync.aligned.u32): all warps of the
            // warpgroup must execute it together, so it is convergent and
            // side-effecting and must not be sunk into divergent branches.
            || name.starts_with("llvm.nvvm.setmaxnreg")
    }
}

#[cfg(test)]
mod tests {
    use super::ModuleExportState;

    #[test]
    fn warp_match_collectives_are_convergent() {
        // `match.{any,all}.sync.*` are warp collectives: every participating
        // lane must execute together, so the exported declaration must carry
        // the `convergent` attribute. These are the exact dotted names produced
        // when lowering the `match_*_sync_*` ops (underscores -> dots on export).
        for name in [
            "llvm.nvvm.match.any.sync.i32",
            "llvm.nvvm.match.any.sync.i64",
            "llvm.nvvm.match.all.sync.i32p",
            "llvm.nvvm.match.all.sync.i64p",
        ] {
            assert!(
                ModuleExportState::is_convergent_intrinsic(name),
                "{name} should be flagged convergent"
            );
        }
    }

    #[test]
    fn bar_warp_sync_is_convergent() {
        // `bar.warp.sync` is a warp-level barrier; the `barrier` prefix used
        // for block-level barriers does not match the `bar.` spelling, so it
        // needs its own coverage.
        assert!(ModuleExportState::is_convergent_intrinsic(
            "llvm.nvvm.bar.warp.sync"
        ));
    }

    #[test]
    fn activemask_is_convergent() {
        // `activemask` returns the set of currently converged lanes, so its
        // result depends on the convergence state and must not be moved.
        assert!(ModuleExportState::is_convergent_intrinsic(
            "llvm.nvvm.activemask"
        ));
    }

    #[test]
    fn non_collective_intrinsics_are_not_convergent() {
        // Plain ALU/special-register intrinsics must NOT be flagged convergent.
        // The lane-position masks are read-only sregs (dotted names produced by
        // lowering `lanemask_*`: underscores -> dots on export), so despite
        // being warp-related they carry no convergence constraint.
        for name in [
            "llvm.nvvm.read.ptx.sreg.tid.x",
            "llvm.nvvm.read.ptx.sreg.laneid",
            "llvm.nvvm.read.ptx.sreg.lanemask.lt",
            "llvm.nvvm.read.ptx.sreg.lanemask.le",
            "llvm.nvvm.read.ptx.sreg.lanemask.eq",
            "llvm.nvvm.read.ptx.sreg.lanemask.ge",
            "llvm.nvvm.read.ptx.sreg.lanemask.gt",
        ] {
            assert!(
                !ModuleExportState::is_convergent_intrinsic(name),
                "{name} should not be flagged convergent"
            );
        }
    }

    #[test]
    fn setmaxnreg_is_convergent() {
        // `setmaxnreg.{inc,dec}.sync.aligned.u32` reconfigures the register
        // file for the whole warpgroup; every warp must execute it together,
        // so the exported declaration must carry `convergent`.
        for name in [
            "llvm.nvvm.setmaxnreg.inc.sync.aligned.u32",
            "llvm.nvvm.setmaxnreg.dec.sync.aligned.u32",
        ] {
            assert!(
                ModuleExportState::is_convergent_intrinsic(name),
                "{name} should be flagged convergent"
            );
        }
    }

    #[test]
    fn counted_cta_barriers_are_convergent() {
        // The counted CTA barrier family (barrier.cta.{sync,arrive} with a
        // thread count, aligned or not) is covered by the `llvm.nvvm.barrier`
        // prefix; this checks that coverage against the exact names the
        // generated lowerings produce.
        for name in [
            "llvm.nvvm.barrier.cta.sync.count",
            "llvm.nvvm.barrier.cta.sync.aligned.count",
            "llvm.nvvm.barrier.cta.arrive.count",
            "llvm.nvvm.barrier.cta.arrive.aligned.count",
            "llvm.nvvm.barrier.cta.sync.aligned.all",
        ] {
            assert!(
                ModuleExportState::is_convergent_intrinsic(name),
                "{name} should be flagged convergent"
            );
        }
    }

    #[test]
    fn redux_sync_intrinsics_are_convergent() {
        // The exact name produced when lowering `redux_sync_add`
        // (`llvm_nvvm_redux_sync_add` -> dotted form on export).
        assert!(ModuleExportState::is_convergent_intrinsic(
            "llvm.nvvm.redux.sync.add"
        ));
        // The whole redux.sync integer family is a warp collective and must be
        // flagged convergent (the `llvm.nvvm.redux` prefix covers every name
        // lowered by the redux ops).
        for name in [
            "llvm.nvvm.redux.sync.umin",
            "llvm.nvvm.redux.sync.min",
            "llvm.nvvm.redux.sync.umax",
            "llvm.nvvm.redux.sync.max",
            "llvm.nvvm.redux.sync.and",
            "llvm.nvvm.redux.sync.or",
            "llvm.nvvm.redux.sync.xor",
        ] {
            assert!(
                ModuleExportState::is_convergent_intrinsic(name),
                "{name} should be convergent"
            );
        }
        // A plain ALU/sreg intrinsic must NOT be flagged convergent.
        assert!(!ModuleExportState::is_convergent_intrinsic(
            "llvm.nvvm.read.ptx.sreg.tid.x"
        ));
    }
}
