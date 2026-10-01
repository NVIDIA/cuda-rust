/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Index-based loops are used intentionally for parallel array iteration patterns
#![allow(clippy::needless_range_loop)]

//! # `dialect-mir` → LLVM dialect Lowering
//!
//! This crate implements the lowering pass that converts
//! [`dialect-mir`][dialect_mir] operations into LLVM dialect operations
//! (provided by `pliron-llvm`, re-exported via [`llvm_export`]), with
//! GPU-specific operations lowered to inline PTX assembly or NVVM
//! intrinsic calls.
//!
//! ## Overview
//!
//! `mir-lower` bridges cuda-oxide's Rust-semantic dialect (`dialect-mir`)
//! to the LLVM dialect. After lowering, ordinary PTX builds go directly to
//! `llvm-export`. NVVM builds first pass through `nvvm-transforms`, which
//! adjusts the LLVM module for the selected libNVVM dialect.
//!
//! ## Compilation Pipeline Position
//!
//! ```text
//! Rust Source Code
//!        │
//!        ▼
//! ┌──────────────┐
//! │   rustc      │  (extracts Stable MIR)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ mir-importer │  (Stable MIR → dialect-mir, mem2reg, annotated unroll)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │  mir-lower   │  ◄── THIS CRATE (dialect-mir → LLVM dialect)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │ llvm-export  │  (exports to LLVM IR)
//! └──────┬───────┘
//!        │
//!        ▼
//! ┌──────────────┐
//! │     llc      │  (LLVM IR → PTX)
//! └──────────────┘
//! ```
//!
//! ## Architecture
//!
//! The crate uses pliron's `DialectConversion` framework for the lowering
//! pass. The framework handles IR walking, def-before-use ordering, type
//! conversion, and block argument patching automatically. Each
//! `dialect-mir` / `dialect-nvvm` op declares its own conversion via the
//! `MirToLlvmConversion` op interface.
//!
//! ### Core Modules
//!
//! - **[`conversion_interface`]**: The `MirToLlvmConversion` op interface
//!   trait. Each `dialect-mir` / `dialect-nvvm` op implements this to
//!   declare how it lowers to the LLVM dialect.
//!
//! - **[`context`]**: CUDA-specific state maps (`SharedGlobalsMap`,
//!   `DynamicSmemAlignmentMap`) used during conversion.
//!
//! - **[`helpers`]**: Utility functions for creating LLVM dialect
//!   constants, declaring intrinsics, and navigating the IR hierarchy.
//!
//! ### Conversion Modules ([`convert`])
//!
//! - **[`convert::types`]**: Type conversion from `dialect-mir` types to
//!   LLVM dialect types.
//!
//! - **[`convert::ops`]**: Operation converters organized by semantic category:
//!   - `arithmetic` - Binary/unary math operations
//!   - `memory` - Load, store, alloca, pointer arithmetic
//!   - `control_flow` - Branch, return, assert
//!   - `constants` - Integer and float constants
//!   - `cast` - Type conversions (int↔float, widening, narrowing)
//!   - `aggregate` - Struct/tuple/enum operations
//!   - `call` - Function calls
//!
//! - **[`convert::intrinsics`]**: GPU intrinsic converters:
//!   - `basic` - Thread/block IDs, barrier
//!   - `warp` - Shuffle, vote operations
//!   - `mbarrier` - Asynchronous barriers (Hopper+)
//!   - `tma` - Tensor Memory Accelerator (Hopper+)
//!   - `wgmma` - Warpgroup Matrix Multiply-Accumulate (Hopper)
//!   - `tcgen05` - 5th-gen Tensor Core (Blackwell)
//!   - `stmatrix` - Shared memory matrix store
//!
//! ## Usage
//!
//! ```ignore
//! use mir_lower::lower_mir_to_llvm;
//! use pliron::context::Context;
//!
//! let mut ctx = Context::new();
//! // ... register dialects, translate MIR into dialect-mir ...
//!
//! lower_mir_to_llvm(&mut ctx, module_op)?;
//!
//! // module_op now contains LLVM dialect operations
//! ```
//!
//! ## GPU Intrinsic Lowering Strategy
//!
//! GPU intrinsics are lowered using two strategies:
//!
//! 1. **LLVM Intrinsic Calls**: For operations with direct NVVM intrinsic
//!    equivalents (e.g., `llvm_nvvm_read_ptx_sreg_tid_x` for thread ID).
//!
//! 2. **Inline PTX Assembly**: For complex operations without direct intrinsics,
//!    or where inline PTX provides better control (e.g., tcgen05, wgmma MMA).
//!    Uses `llvm.inlineasm` with the `convergent` attribute for warp-synchronous
//!    semantics.

#![warn(missing_docs)]

pub mod context;
pub mod conversion_interface;
pub mod convert;
pub mod helpers;
pub mod lowering;
mod packed_shared_local_storage;
pub mod scalarize_block_args;
pub mod type_conversion_interface;
mod wgmma_deferred_accumulator;

use rustc_hash::FxHashMap;

use pliron::{
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::{Context, Ptr},
    irbuild::dialect_conversion::{
        DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
    },
    irbuild::{listener::Recorder, rewriter::IRRewriter},
    location::Located,
    op::{Op, op_cast},
    operation::Operation,
    opts::simplify_cfg::remove_blocks_inside_op,
    result::Result,
    r#type::{TypeHandle, Typed, type_impls},
};

use context::{DeviceGlobalsMap, DynamicSmemAlignmentMap, SharedGlobalsMap};
use conversion_interface::MirToLlvmConversion as MirToLlvmConversionInterface;
use convert::types::convert_type;
use type_conversion_interface::MirConvertibleType;

// ============================================================================
// DialectConversion driver
// ============================================================================

/// Backend whose intrinsic ABI the lowering pass must emit.
///
/// LLVM's NVPTX backend and NVIDIA's libNVVM accept overlapping but not
/// identical intrinsic signatures. The pipeline chooses this once, before
/// any typed MIR operation is lowered, so generated intrinsic conversions do
/// not have to guess from environment variables or partially lowered IR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IntrinsicBackend {
    /// Emit intrinsic forms consumed by LLVM's NVPTX backend (`llc`).
    #[default]
    LlvmNvptx,
    /// Emit intrinsic forms consumed by NVIDIA's libNVVM compiler.
    LibNvvm,
}

/// Options controlling the `dialect-mir` to LLVM dialect lowering pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoweringOptions {
    /// Whether ordinary floating-point multiply/add or multiply/subtract
    /// expressions may contract into fused operations.
    ///
    /// This does not affect explicit fused operations such as `f32::mul_add`.
    pub allow_fma_contraction: bool,
    /// Intrinsic ABI expected by the selected LLVM-to-device backend.
    pub intrinsic_backend: IntrinsicBackend,
    /// Stable identity of the compilation this module belongs to, woven into
    /// generated module-scope symbols (`__shared_mem_*`, `__device_global_*`,
    /// `__dynamic_smem_*`).
    ///
    /// The per-module counters below make the counter-named families unique
    /// within one module, but every module starts its counters at zero, so
    /// two modules lowered from different crates both define
    /// `__shared_mem_0`; a dynamic-pool owner shared across crates likewise
    /// repeats its `__dynamic_smem_<owner>` extern in both. Their PTX is
    /// textually concatenated by `load_all_ptx_bundles_merged`, where a
    /// duplicate definition fails driver JIT compilation and duplicate
    /// extern declarations silently keep the first alignment (#1277). A
    /// caller whose output can be merged with other compilations passes a
    /// value that is stable for one compilation and distinct across crates —
    /// `rustc-codegen-cuda` passes the crate's `StableCrateId` hash. `None`
    /// preserves the undecorated historical names.
    pub module_disambiguator: Option<u64>,
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self {
            allow_fma_contraction: true,
            intrinsic_backend: IntrinsicBackend::LlvmNvptx,
            module_disambiguator: None,
        }
    }
}

/// `dialect-mir` → LLVM dialect conversion driver.
///
/// Implements pliron's `DialectConversion` trait. The `rewrite` method uses
/// `op_cast`-based dispatch via the `MirToLlvmConversion` op interface,
/// so each `dialect-mir` / `dialect-nvvm` op declares its own lowering.
///
/// Holds CUDA-specific state that certain ops need during conversion:
/// shared-memory global deduplication and dynamic shared-memory alignment.
pub struct MirToLlvmConversionDriver {
    /// Shared memory global deduplication across all functions.
    pub shared_globals: SharedGlobalsMap,
    /// Device global deduplication across all functions.
    pub device_globals: DeviceGlobalsMap,
    /// Per-owning-function dynamic shared memory alignment tracking.
    pub dynamic_smem_alignments: DynamicSmemAlignmentMap,
    /// Next `__shared_mem_N` index. Scoped to one driver instance (one
    /// `lower_mir_to_llvm` call, i.e. one module), not a process-global
    /// counter, so the assigned index is a function of this module's own
    /// MIR walk order rather than of how many OTHER modules have lowered a
    /// shared allocation earlier in the process. See #706.
    pub next_shared_mem_index: usize,
    /// Next `__device_global_N` index. Scoped to one driver instance for the
    /// same reason as `next_shared_mem_index`: the assigned index is a
    /// function of this module's own MIR walk order rather than of how many
    /// OTHER modules have lowered a device global earlier in the process.
    /// See #706.
    pub next_device_global_index: usize,
}

fn is_mir_or_nvvm_op(ctx: &Context, op: Ptr<Operation>) -> bool {
    let opid = Operation::get_opid(op, ctx);
    let dialect = opid.dialect.to_string();
    dialect == "mir" || dialect == "nvvm"
}

/// True for a `builtin.constant` whose result is a signed/unsigned (non-signless)
/// integer. That is the only `builtin.constant` lowering must touch: `sccp` can
/// materialise such a constant (it carries the MIR integer type, e.g. `ui32`), and
/// lowering must normalise it to a signless LLVM integer like it does for
/// `mir.constant`, else the LLVM module ends up with mismatched operand types
/// (a signless op fed by a signed/unsigned constant). A *signless* builtin.constant
/// is left alone (legal; the textual exporter emits it). Because the conversion
/// emits a signless constant, this predicate is false for the result, so the
/// DialectConversion worklist converges instead of looping.
fn is_signed_builtin_constant(ctx: &Context, op: Ptr<Operation>) -> bool {
    if Operation::get_opid(op, ctx) != pliron::builtin::ops::ConstantOp::get_opid_static() {
        return false;
    }
    let res_ty = op.deref(ctx).get_result(0).get_type(ctx);
    res_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|it| it.signedness() != Signedness::Signless)
}

impl DialectConversion for MirToLlvmConversionDriver {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        // A signless `builtin.constant` is left alone: it is legal and the textual
        // exporter emits it directly. But sccp can materialise a builtin.constant
        // carrying a signed/unsigned MIR integer type; lowering must normalise that
        // to signless (like `mir.constant`), or the LLVM module gets mismatched
        // operand types. So mark ONLY a non-signless builtin.constant convertible.
        // Its conversion emits a signless constant (no longer convertible), so this
        // converges — it does NOT loop the way marking *every* builtin.constant did.
        is_mir_or_nvvm_op(ctx, op) || is_signed_builtin_constant(ctx, op)
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        let ty_ref = ty.deref(ctx);

        // Signed/unsigned integers need signless normalisation (LLVM convention).
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            return int_ty.signedness() != Signedness::Signless;
        }

        type_impls::<dyn MirConvertibleType>(&*ty_ref)
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        convert_type(ctx, ty).map_err(|e| pliron::input_error_noloc!("{e}"))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let opid = Operation::get_opid(op, ctx);
        let loc = op.deref(ctx).loc();

        // Special-case ops that need CUDA pass-level state.
        if opid == dialect_mir::ops::MirFuncOp::get_opid_static() {
            return lowering::convert_func(
                ctx,
                rewriter,
                op,
                operands_info,
                &mut self.shared_globals,
                &mut self.dynamic_smem_alignments,
            );
        }
        if opid == dialect_mir::ops::MirSharedAllocOp::get_opid_static() {
            return convert::ops::memory::convert_shared_alloc_dc(
                ctx,
                rewriter,
                op,
                operands_info,
                &mut self.shared_globals,
                &mut self.next_shared_mem_index,
            );
        }
        if opid == dialect_mir::ops::MirGlobalAllocOp::get_opid_static() {
            return convert::ops::memory::convert_global_alloc_dc(
                ctx,
                rewriter,
                op,
                operands_info,
                &mut self.device_globals,
                &mut self.next_device_global_index,
            );
        }
        if opid == dialect_mir::ops::MirExternSharedOp::get_opid_static() {
            return convert::ops::memory::convert_extern_shared_dc(
                ctx,
                rewriter,
                op,
                operands_info,
                &mut self.shared_globals,
                &mut self.dynamic_smem_alignments,
            );
        }

        // Generic dispatch for all other ops via op_cast.
        let op_obj = Operation::get_op_dyn(op, ctx);
        let Some(converter) = op_cast::<dyn MirToLlvmConversionInterface>(op_obj.as_ref()) else {
            return pliron::input_err!(
                loc,
                "Unsupported MIR/NVVM op for lowering: {}",
                Operation::get_opid(op, ctx)
            );
        };
        converter.convert(ctx, rewriter, operands_info)
    }
}

/// Runs the `dialect-mir` → LLVM dialect lowering pass on the given module.
///
/// This is the main entry point for the lowering pass. It uses pliron's
/// `DialectConversion` framework to walk the IR, convert types, and
/// dispatch per-op conversion logic.
///
/// # Arguments
///
/// * `ctx` - Mutable reference to the pliron context
/// * `module_op` - Pointer to the module operation to transform
///
/// # Returns
///
/// `Ok(())` if all operations were successfully converted.
pub fn lower_mir_to_llvm(ctx: &mut Context, module_op: Ptr<Operation>) -> Result<()> {
    lower_mir_to_llvm_with_options(ctx, module_op, LoweringOptions::default())
}

/// Runs the `dialect-mir` → LLVM dialect lowering pass with explicit options.
///
/// Use this entry point when the caller needs compilation-wide floating-point
/// policy such as disabling implicit FMA contraction.
pub fn lower_mir_to_llvm_with_options(
    ctx: &mut Context,
    module_op: Ptr<Operation>,
    options: LoweringOptions,
) -> Result<()> {
    // Standalone users can enter here without cuda-oxide-codegen's preparation
    // pipeline. Verify the complete typed MIR tree before any transform reads
    // pointer-kind claims.
    dialect_mir::verification::verify_pointer_kind_producers(ctx, module_op)?;
    module_op.deref(ctx).verify(ctx)?;
    context::set_lowering_options(ctx, options);
    // WGMMA pointer-form MMA operations are only sound when their complete
    // asynchronous lifetime can be closed before LLVM sees pending accumulator
    // state. Canonical [[f32; 8]; 4] accumulators use explicit SSA values for
    // linear groups, counted K-loops, and proven static partial-wait pipelines;
    // unsupported pointer shapes retain the deferred pointer-group fallback.
    // Run this while MIR control flow and unsigned constants are still intact.
    wgmma_deferred_accumulator::fuse_deferred_accumulators(ctx, module_op)?;
    // WGMMA fusion is the final MIR-producing transform in this entry point.
    // Reverify immediately before dialect conversion erases pointer kinds so a
    // future transform cannot bypass the provenance invariant.
    dialect_mir::verification::verify_pointer_kind_producers(ctx, module_op)?;
    module_op.deref(ctx).verify(ctx)?;
    // Dynamic shared-memory operations may live in device helpers. Compute
    // every kernel-to-helper requirement while the complete MIR call graph is
    // still available; function conversion removes that graph incrementally.
    lowering::propagate_kernel_dynamic_shared_alignments(ctx, module_op);
    // Prove the complete address-use path for every narrow packed-AS3 carrier
    // local immediately before conversion. The resulting per-op TypeAttrs are
    // lowering capabilities, not inferred provenance: calls, block arguments,
    // casts, nested projections, returns, and unknown uses fail closed here.
    packed_shared_local_storage::prepare_packed_shared_local_storage(ctx, module_op)?;
    let mut conversion = MirToLlvmConversionDriver {
        shared_globals: FxHashMap::default(),
        device_globals: FxHashMap::default(),
        dynamic_smem_alignments: FxHashMap::default(),
        next_shared_mem_index: 0,
        next_device_global_index: 0,
    };
    // pliron's DialectConversion now reports an IRStatus (Changed/Unchanged);
    // lowering only cares about success, so discard it.
    apply_dialect_conversion(ctx, &mut conversion, module_op)?;
    // Conversions of diverging ops (e.g. `nvvm.assertfail`) erase everything
    // after the noreturn call, including the block's terminator. A successor
    // whose only predecessor was that terminator is now unreachable, and if it
    // still carries block arguments it cannot be exported (a PHI needs one
    // incoming value per predecessor). Erase every block the entry can no
    // longer reach.
    let mut rewriter = IRRewriter::<Recorder>::default();
    remove_blocks_inside_op(module_op, ctx, &mut rewriter);
    // mem2reg promotes enum/struct slots into whole-aggregate block arguments,
    // which export as PHIs of first-class aggregates. LLVM's -O2 pipeline
    // cannot split those (SROA only handles allocas), so e.g. an iterator
    // loop merging `Option<(f32, f32)>` keeps a materialized discriminant and
    // an extra branch per iteration. Split such arguments into scalar leaves
    // so the exported IR carries scalar PHIs, as SROA would produce.
    scalarize_block_args::scalarize_aggregate_block_args(ctx, module_op)?;
    Ok(())
}

/// Register the `dialect-mir` → LLVM dialect lowering pass with a pliron context.
///
/// This is a placeholder for future pass manager integration.
/// Currently, the pass is invoked directly via [`lower_mir_to_llvm`].
pub fn register(_ctx: &mut Context) {
    // Placeholder for future pass manager integration
}
