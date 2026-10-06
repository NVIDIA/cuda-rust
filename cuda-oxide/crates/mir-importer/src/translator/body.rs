/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Function body translation: MIR → `mir.func`.
//!
//! Translates complete MIR function bodies into `dialect-mir` `mir.func` operations.
//!
//! # Responsibilities
//!
//! - Extract function signature (arguments, return type)
//! - Create and link pliron IR basic blocks (entry block carries function
//!   parameters; every other block is argument-less)
//! - Emit one `mir.alloca` per non-ZST MIR local at the top of the entry
//!   block and record the slot in [`ValueMap`]
//! - Translate every reachable block in order; unwind-only cleanup blocks
//!   are patched with `mir.unreachable`
//! - Detect compile-time kernel attributes (`#[cluster(...)]`,
//!   `#[launch_bounds(...)]`)

use super::block;
use super::facts;
use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::pipeline::StatementDebugInfoMap;
use crate::translator::location::span_to_location;
use crate::translator::values::{self, SlotAddrSpaceMap, ValueMap};
use dialect_mir::attributes::ReferenceParamValidityAttr;
use dialect_mir::ops::MirFuncOp;
use dialect_mir::types::address_space;
use llvm_export::export::DebugKind;
use llvm_export::ops::{
    DebugEnumDiscriminant, DebugEnumVariant, DebugFragment, DebugFragmentVariableInfo,
    DebugLocalTypeKind, DebugLocalVariableInfo, DebugProjectedVariableInfo, DebugSourcePosition,
    DebugSourceScopeMap, DebugTypeMember, DebugWholeVariableInfo, LocalMemoryProvenanceAttr,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::context::{Context, Ptr};
use pliron::identifier::{Identifier, Legaliser};
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::{input_err_noloc, input_error_noloc};
use reserved_oxide_symbols::{
    MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX, MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX,
};
use rustc_public::CrateDefType;

// Re-export rustc_public types for convenience
use rustc_hash::FxHashMap;
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::mir::mono;
use rustc_public::ty::{ConstantKind, FloatTy, IntTy, RigidTy, Ty, TyKind, UintTy};
use rustc_public_bridge::IndexedVal;

/// Cluster dimensions extracted from `#[cluster(x,y,z)]` attribute.
///
/// These are detected by scanning MIR for `cuda_device::cluster::__cluster_config::<X,Y,Z>()`
/// marker calls injected by the `#[cluster]` macro.
#[derive(Debug, Clone, Copy)]
pub struct ClusterDims {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Launch bounds extracted from `#[launch_bounds(max, min)]` attribute.
///
/// These are detected by scanning MIR for `cuda_device::thread::__launch_bounds_config::<MAX,MIN>()`
/// marker calls injected by the `#[launch_bounds]` macro.
#[derive(Debug, Clone, Copy)]
pub struct LaunchBounds {
    /// Maximum threads per block (.maxntid in PTX)
    pub max_threads: u32,
    /// Minimum blocks per SM (.minnctapersm in PTX), 0 if unspecified
    pub min_blocks: u32,
}

/// Exact block shape declared by `#[launch_contract(block = (x, y, z))]`.
///
/// Detected by scanning MIR for
/// `cuda_device::thread::__launch_contract_block_config::<X, Y, Z>()` marker
/// calls. Emitted as `.reqntid x, y, z`, which the CUDA driver enforces per
/// axis at launch.
#[derive(Debug, Clone, Copy)]
pub struct ContractBlock {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Minimum extern-shared alignment declared by `#[launch_contract]`.
#[derive(Debug, Clone, Copy)]
pub struct DynamicSharedAlignment {
    pub bytes: u64,
}

/// Scans MIR for `__cluster_config::<X, Y, Z>()` marker and extracts cluster dimensions.
///
/// The `#[cluster(x,y,z)]` macro injects this call at the start of the kernel.
/// We scan the MIR to find it and extract the const generic parameters.
///
/// Returns `Some(ClusterDims)` if found, `None` otherwise.
fn detect_cluster_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Option<ClusterDims> {
    use rustc_public::ty::TyConstKind;

    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        // Use let-else for early continue pattern
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let fn_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (fn_name != "__cluster_config" && !fn_name.ends_with("::__cluster_config"))
        {
            continue;
        }

        // Extract const generic args (X, Y, Z)
        let mut dims = [1u32, 1u32, 1u32];
        for (i, arg) in args.0.iter().take(3).enumerate() {
            let rustc_public::ty::GenericArgKind::Const(c) = arg else {
                continue;
            };
            dims[i] = match c.kind() {
                TyConstKind::Value(_, alloc) => alloc.read_uint().ok().map(|v| v as u32),
                _ => c.eval_target_usize().ok().map(|v| v as u32),
            }
            .unwrap_or(dims[i]);
        }

        return Some(ClusterDims {
            x: dims[0],
            y: dims[1],
            z: dims[2],
        });
    }
    None
}

/// Source parameter indices declared with `#[grid_constant]`.
///
/// `#[kernel]` replaces each parameter attribute with one
/// `__grid_constant_config::<INDEX>()` marker. Indices remain source-level
/// here; mir-lower maps them through slice scalarization to LLVM parameter
/// positions exactly once.
fn detect_grid_constant_params(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
    instance: &mono::Instance,
    is_kernel: bool,
) -> Result<Vec<usize>, String> {
    use rustc_public::ty::TyConstKind;

    let mut indices = std::collections::BTreeSet::new();
    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };
        let definition_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (definition_name != "__grid_constant_config"
                && !definition_name.ends_with("::__grid_constant_config"))
        {
            continue;
        }
        facts::validate_grid_constant_marker_owner(instance, block_idx, is_kernel)?;
        if args.0.len() != 1 {
            return Err(format!(
                "cuda_device grid-constant marker has {} generic arguments; expected exactly 1",
                args.0.len()
            ));
        }
        let rustc_public::ty::GenericArgKind::Const(index) = &args.0[0] else {
            return Err("cuda_device grid-constant parameter index is not a constant".to_string());
        };
        let raw = match index.kind() {
            TyConstKind::Value(_, allocation) => allocation.read_uint().map_err(|error| {
                format!("could not read grid-constant parameter index: {error:?}")
            })?,
            _ => u128::from(index.eval_target_usize().map_err(|error| {
                format!("could not evaluate grid-constant parameter index: {error:?}")
            })?),
        };
        let index = usize::try_from(raw)
            .map_err(|_| format!("grid-constant parameter index {raw} does not fit usize"))?;
        if !indices.insert(index) {
            return Err(format!(
                "kernel contains duplicate grid-constant marker for source parameter {index}"
            ));
        }
    }
    Ok(indices.into_iter().collect())
}

/// Scans MIR for `__launch_bounds_config::<MAX, MIN>()` marker and extracts launch bounds.
///
/// The `#[launch_bounds(max, min)]` macro injects this call at the start of the kernel.
/// We scan the MIR to find it and extract the const generic parameters.
///
/// Returns `Some(LaunchBounds)` if found, `None` otherwise.
fn detect_launch_bounds_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Result<Option<LaunchBounds>, String> {
    use rustc_public::ty::TyConstKind;

    let mut detected: Option<LaunchBounds> = None;
    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let definition_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (definition_name != "__launch_bounds_config"
                && !definition_name.ends_with("::__launch_bounds_config"))
        {
            continue;
        }

        if args.0.len() != 2 {
            return Err(format!(
                "cuda_device launch-bounds marker has {} generic arguments; expected exactly 2",
                args.0.len()
            ));
        }
        let mut values = [0u32; 2];
        for (index, (name, arg)) in ["maximum threads", "minimum blocks"]
            .into_iter()
            .zip(args.0.iter())
            .enumerate()
        {
            let rustc_public::ty::GenericArgKind::Const(value) = arg else {
                return Err(format!(
                    "cuda_device launch-bounds {name} argument is not a constant"
                ));
            };
            let raw = match value.kind() {
                TyConstKind::Value(_, allocation) => allocation.read_uint().map_err(|error| {
                    format!("could not read launch-bounds {name} constant: {error:?}")
                })?,
                _ => u128::from(value.eval_target_usize().map_err(|error| {
                    format!("could not evaluate launch-bounds {name} constant: {error:?}")
                })?),
            };
            values[index] = u32::try_from(raw)
                .map_err(|_| format!("launch-bounds {name} value {raw} does not fit in u32"))?;
        }
        if values[0] == 0 {
            return Err("launch-bounds maximum threads must be greater than zero".to_string());
        }
        let bounds = LaunchBounds {
            max_threads: values[0],
            min_blocks: values[1],
        };
        if let Some(existing) = detected {
            if existing.max_threads != bounds.max_threads
                || existing.min_blocks != bounds.min_blocks
            {
                return Err(format!(
                    "a kernel contains conflicting cuda_device launch-bounds markers: ({}, {}) and ({}, {})",
                    existing.max_threads,
                    existing.min_blocks,
                    bounds.max_threads,
                    bounds.min_blocks,
                ));
            }
        } else {
            detected = Some(bounds);
        }
    }
    Ok(detected)
}

/// Scans MIR for `__launch_contract_block_config::<X, Y, Z>()` and extracts the
/// exact block shape declared by `#[launch_contract(block = (x, y, z))]`.
///
/// Returns `Some(ContractBlock)` if found, `None` otherwise.
fn detect_contract_block_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Result<Option<ContractBlock>, String> {
    use rustc_public::ty::TyConstKind;

    let mut detected: Option<ContractBlock> = None;
    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let definition_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (definition_name != "__launch_contract_block_config"
                && !definition_name.ends_with("::__launch_contract_block_config"))
        {
            continue;
        }

        if args.0.len() != 3 {
            return Err(format!(
                "cuda_device launch-contract block marker has {} generic arguments; expected exactly 3",
                args.0.len()
            ));
        }
        let mut values = [0u32; 3];
        for (index, (axis, arg)) in ["x", "y", "z"].into_iter().zip(args.0.iter()).enumerate() {
            let rustc_public::ty::GenericArgKind::Const(value) = arg else {
                return Err(format!(
                    "cuda_device launch-contract block {axis} argument is not a constant"
                ));
            };
            let raw = match value.kind() {
                TyConstKind::Value(_, allocation) => allocation.read_uint().map_err(|error| {
                    format!("could not read launch-contract block {axis} constant: {error:?}")
                })?,
                _ => u128::from(value.eval_target_usize().map_err(|error| {
                    format!("could not evaluate launch-contract block {axis} constant: {error:?}")
                })?),
            };
            values[index] = u32::try_from(raw).map_err(|_| {
                format!("launch-contract block {axis} value {raw} does not fit in u32")
            })?;
            if values[index] == 0 {
                return Err(format!(
                    "launch-contract block {axis} dimension must be greater than zero"
                ));
            }
        }
        let shape = ContractBlock {
            x: values[0],
            y: values[1],
            z: values[2],
        };
        if let Some(existing) = detected {
            if existing.x != shape.x || existing.y != shape.y || existing.z != shape.z {
                return Err(format!(
                    "a kernel contains conflicting cuda_device launch-contract block markers: ({}, {}, {}) and ({}, {}, {})",
                    existing.x, existing.y, existing.z, shape.x, shape.y, shape.z,
                ));
            }
        } else {
            detected = Some(shape);
        }
    }
    Ok(detected)
}

/// Rejects an exact block shape that needs more threads than
/// `#[launch_bounds]` allows.
///
/// An exact block displaces the thread maximum in the emitted PTX, because
/// ptxas rejects an entry carrying both `.maxntid` and `.reqntid`. A maximum
/// below the required thread count would therefore be dropped in silence, and
/// the kernel would launch at a shape its author ruled out. A maximum at or
/// above the required count is redundant rather than contradictory, since
/// `.reqntid` is the stronger statement, so it stays allowed.
fn validate_block_against_bounds(bounds: LaunchBounds, block: ContractBlock) -> Result<(), String> {
    let required = u64::from(block.x) * u64::from(block.y) * u64::from(block.z);
    if required > u64::from(bounds.max_threads) {
        return Err(format!(
            "a kernel declares #[launch_contract(block = ({}, {}, {}))], needing {} threads per block, and #[launch_bounds({})], allowing at most {}",
            block.x, block.y, block.z, required, bounds.max_threads, bounds.max_threads,
        ));
    }
    Ok(())
}

/// Scans MIR for the `__unchecked_indexing_config::<ENABLED>()` marker
/// injected by `#[kernel(unchecked_indexing)]` and extracts its const bool.
///
/// Returns `Ok(true)` when a marker with `ENABLED = true` is reachable in
/// this body. The marker call itself is stripped later during terminator
/// translation; this scan only records the policy.
fn detect_unchecked_indexing_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Result<bool, String> {
    use rustc_public::ty::TyConstKind;

    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let definition_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (definition_name != "__unchecked_indexing_config"
                && !definition_name.ends_with("::__unchecked_indexing_config"))
        {
            continue;
        }

        if args.0.len() != 1 {
            return Err(format!(
                "cuda_device unchecked-indexing marker has {} generic arguments; expected exactly 1",
                args.0.len()
            ));
        }
        let rustc_public::ty::GenericArgKind::Const(value) = &args.0[0] else {
            return Err(
                "cuda_device unchecked-indexing marker argument is not a constant".to_string(),
            );
        };
        let enabled = match value.kind() {
            TyConstKind::Value(_, allocation) => allocation.read_bool().map_err(|error| {
                format!("could not read unchecked-indexing marker constant: {error:?}")
            })?,
            _ => {
                value.eval_target_usize().map_err(|error| {
                    format!("could not evaluate unchecked-indexing marker constant: {error:?}")
                })? != 0
            }
        };
        if enabled {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether the whole-build unchecked-indexing switch is on.
///
/// `CUDA_OXIDE_UNCHECKED_INDEXING=1` (or `true`) elides bounds-check asserts
/// in every translated body, including separately translated `#[device]`
/// functions that the per-kernel marker cannot reach.
fn unchecked_indexing_env_enabled() -> bool {
    std::env::var("CUDA_OXIDE_UNCHECKED_INDEXING")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Scans MIR for the dynamic-shared alignment marker injected by
/// `#[launch_contract]` and extracts its const generic argument. The importer
/// records the value before removing the call from the executable path.
fn detect_dynamic_shared_alignment(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Option<DynamicSharedAlignment> {
    use rustc_public::ty::TyConstKind;

    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };
        let fn_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (fn_name != "__dynamic_shared_alignment"
                && !fn_name.ends_with("::__dynamic_shared_alignment"))
        {
            continue;
        }
        let rustc_public::ty::GenericArgKind::Const(alignment) = args.0.first()? else {
            continue;
        };
        let bytes = match alignment.kind() {
            TyConstKind::Value(_, alloc) => alloc.read_uint().ok().map(|value| value as u64),
            _ => alignment.eval_target_usize().ok(),
        }?;
        return Some(DynamicSharedAlignment { bytes });
    }
    None
}

/// Return the non-unwind successors of a block's terminator.
///
/// [`mir::Terminator::successors`] includes unwind cleanup blocks alongside
/// "normal" control-flow targets. The CUDA toolchain does not support stack
/// unwinding (hardware could, but `nvcc`/`ptxas` never wire it up), so the
/// translator treats unwind cleanups as dead code. This helper strips them
/// out so the worklist only visits blocks that matter on GPU. Monomorphized
/// branch reachability is supplied separately by rustc's collector; the
/// importer must not reconstruct a second constant-evaluation model from the
/// converted public MIR.
fn non_unwind_successors(block: &mir::BasicBlock) -> Vec<usize> {
    use mir::TerminatorKind::*;
    match &block.terminator.kind {
        Goto { target } => vec![*target],
        SwitchInt { targets, .. } => targets.all_targets(),
        Return | Resume | Abort | Unreachable => vec![],
        Drop { target, .. } | Assert { target, .. } => vec![*target],
        Call { target, .. } => target.map(|t| vec![t]).unwrap_or_default(),
        InlineAsm { destination, .. } => destination.map(|t| vec![t]).unwrap_or_default(),
    }
}

fn validate_monomorphized_successor_shape(
    body_block_count: usize,
    rustc_mir_block_count: usize,
    rustc_mono_successors: &[Vec<usize>],
) -> Result<(), String> {
    if body_block_count != rustc_mir_block_count {
        return Err(format!(
            "rustc/public MIR CFG mismatch: collector recorded {rustc_mir_block_count} blocks but importer received {body_block_count}"
        ));
    }
    if rustc_mono_successors.len() != body_block_count {
        return Err(format!(
            "rustc collector supplied successor lists for {} blocks but the public MIR body has {body_block_count}",
            rustc_mono_successors.len()
        ));
    }
    for (source, successors) in rustc_mono_successors.iter().enumerate() {
        if let Some(target) = successors
            .iter()
            .copied()
            .find(|target| *target >= body_block_count)
        {
            return Err(format!(
                "rustc collector edge {source} -> {target} is outside the {body_block_count}-block public MIR body"
            ));
        }
    }
    Ok(())
}

fn validate_monomorphized_successors(
    body: &mir::Body,
    rustc_mir_block_count: usize,
    rustc_mono_successors: &[Vec<usize>],
) -> Result<(), String> {
    validate_monomorphized_successor_shape(
        body.blocks.len(),
        rustc_mir_block_count,
        rustc_mono_successors,
    )?;
    for (source, successors) in rustc_mono_successors.iter().enumerate() {
        let public_successors = body.blocks[source].terminator.successors();
        if let Some(target) = successors
            .iter()
            .copied()
            .find(|target| !public_successors.contains(target))
        {
            return Err(format!(
                "rustc collector edge {source} -> {target} does not exist in the converted public MIR CFG"
            ));
        }
    }
    Ok(())
}

/// BFS from the entry block following rustc's exact per-block monomorphized
/// successors, intersected with the importer's existing non-unwind policy.
///
/// The result is a sorted set of reachable-on-GPU block indices; unwind-only
/// cleanup blocks end up outside this set and are filled in with
/// `mir.unreachable` by [`translate_body`] so pliron verification still
/// passes. Constant switches and device runtime-check switches are never
/// re-evaluated here: the collector's edges are the semantic source of truth.
fn compute_reachable_blocks(
    body: &mir::Body,
    rustc_mono_successors: &[Vec<usize>],
) -> std::collections::BTreeSet<usize> {
    let mut reachable = std::collections::BTreeSet::new();
    let mut frontier: Vec<usize> = vec![0];
    reachable.insert(0);
    while let Some(idx) = frontier.pop() {
        let non_unwind: std::collections::BTreeSet<_> = non_unwind_successors(&body.blocks[idx])
            .into_iter()
            .collect();
        for &succ in &rustc_mono_successors[idx] {
            if non_unwind.contains(&succ) && reachable.insert(succ) {
                frontier.push(succ);
            }
        }
    }
    reachable
}

#[derive(Clone)]
struct LocalDebugInfo {
    variable: DebugLocalVariableInfo,
    loc: pliron::location::Location,
    source_scope: u32,
    declaration: Option<DebugSourcePosition>,
}

#[derive(Default)]
struct CollectedDebugLocals {
    whole: FxHashMap<mir::Local, Vec<LocalDebugInfo>>,
    projected: FxHashMap<mir::Local, Vec<DebugProjectedVariableInfo>>,
    fragments: FxHashMap<mir::Local, Vec<DebugFragmentVariableInfo>>,
}

/// Build full-debug bindings for whole locals, supported place projections,
/// and rustc scalar-replacement fragments.
///
/// A composite record describes one storage piece of a larger source variable.
/// The stable MIR `composite.ty` is the complete source type and
/// `composite.projection` identifies the piece inside it. For now fragments are
/// accepted only when rustc stores the piece in a whole MIR local and the
/// composite projection is a static `Field` chain. This is the scalar-replaced
/// aggregate shape emitted by rustc and keeps the location semantics exact.
///
/// Ordinary projected bindings retain the existing support for static fields,
/// forward constant indices, enum payload fields, and one leading thin-pointer
/// dereference. Dynamic indices, dereference-index chains, repeated dereferences,
/// fat pointers, slices, opaque casts, and non-field composite projections are
/// skipped rather than approximated.
fn collect_debug_locals(ctx: &mut Context, body: &mir::Body) -> CollectedDebugLocals {
    let mut collected = CollectedDebugLocals::default();

    for info in &body.var_debug_info {
        let name = info.name.to_string();
        if name.is_empty() {
            continue;
        }

        let mir::VarDebugInfoContents::Place(place) = &info.value else {
            continue;
        };
        let local = place.local;
        let local_idx: usize = local;
        if local_idx == 0 {
            continue;
        }

        if let Some(composite) = &info.composite {
            // A promoted `dbg.value` for the backing local denotes the fragment
            // value itself. Supporting a projected storage place would require
            // extracting that subvalue after promotion, so fail closed here.
            if !place.projection.is_empty() {
                continue;
            }
            let Some(fragment) = debug_fragment(composite) else {
                continue;
            };
            let Some(ty) = debug_type_for_ty(&composite.ty) else {
                continue;
            };
            if fragment.offset_bits == 0
                && layout_size_bits(&composite.ty) == Some(fragment.size_bits)
            {
                collected
                    .whole
                    .entry(local)
                    .or_default()
                    .push(LocalDebugInfo {
                        variable: DebugLocalVariableInfo {
                            name,
                            argument_index: info.argument_index,
                            ty,
                        },
                        loc: span_to_location(ctx, info.source_info.span),
                        source_scope: info.source_info.scope,
                        declaration: debug_source_position(info.source_info.span),
                    });
                continue;
            }

            collected
                .fragments
                .entry(local)
                .or_default()
                .push(DebugFragmentVariableInfo {
                    variable: DebugLocalVariableInfo {
                        name,
                        argument_index: info.argument_index,
                        ty,
                    },
                    fragment,
                    source_scope: Some(info.source_info.scope),
                    declaration: debug_source_position(info.source_info.span),
                });
            continue;
        }

        if place.projection.is_empty() {
            let Some(decl) = body.local_decl(local) else {
                continue;
            };
            let Some(ty) = debug_type_for_ty(&decl.ty) else {
                continue;
            };

            collected
                .whole
                .entry(local)
                .or_default()
                .push(LocalDebugInfo {
                    variable: DebugLocalVariableInfo {
                        name,
                        argument_index: info.argument_index,
                        ty,
                    },
                    loc: span_to_location(ctx, info.source_info.span),
                    source_scope: info.source_info.scope,
                    declaration: debug_source_position(info.source_info.span),
                });
            continue;
        }

        let Some(projection) = debug_projection(body, place) else {
            continue;
        };
        let Some(ty) = debug_type_for_ty(&projection.ty) else {
            continue;
        };

        // rustc treats projected argument bindings as local variables rather
        // than formal argument variables; only whole-place arguments receive
        // a DWARF argument index.
        collected
            .projected
            .entry(local)
            .or_default()
            .push(DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name,
                    argument_index: None,
                    ty,
                },
                dereference_base: projection.dereference_base,
                offset_bytes: projection.offset_bytes,
                source_scope: Some(info.source_info.scope),
                declaration: debug_source_position(info.source_info.span),
            });
    }

    collected
}

fn debug_fragment(fragment: &mir::VarDebugInfoFragment) -> Option<DebugFragment> {
    let whole_size_bits = layout_size_bits(&fragment.ty)?;
    if whole_size_bits == 0 {
        return None;
    }

    let mut current_ty = fragment.ty;
    let mut offset_bytes = 0u64;
    for elem in &fragment.projection {
        let mir::ProjectionElem::Field(field_idx, field_ty) = elem else {
            return None;
        };
        let layout = current_ty.layout().ok()?;
        let shape = layout.shape();
        let rustc_public::abi::FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return None;
        };
        offset_bytes = offset_bytes.checked_add(offsets.get(*field_idx)?.bytes() as u64)?;
        current_ty = *field_ty;
    }

    let offset_bits = offset_bytes.checked_mul(8)?;
    let size_bits = layout_size_bits(&current_ty)?;
    if size_bits == 0 || offset_bits.checked_add(size_bits)? > whole_size_bits {
        return None;
    }

    Some(DebugFragment {
        offset_bits,
        size_bits,
    })
}

#[derive(Clone, Copy)]
struct ResolvedDebugProjection {
    dereference_base: bool,
    offset_bytes: u64,
    ty: Ty,
}

/// Resolve the location expression and final type of a supported MIR projection.
///
/// Static `Field`/forward-`ConstantIndex` chains retain the #939 behavior. A
/// single leading `Deref` is additionally accepted when the base has one pointer
/// word of storage; after that dereference only `Field` projections are allowed.
/// This deliberately rejects fat references/raw pointers, repeated dereferences,
/// and dereference-plus-index chains instead of emitting an approximate location.
fn debug_projection(body: &mir::Body, place: &mir::Place) -> Option<ResolvedDebugProjection> {
    let mut current_ty = body.local_decl(place.local)?.ty;
    let mut dereference_base = false;
    let mut offset_bytes = 0u64;
    let mut enum_variant = None;

    for (index, elem) in place.projection.iter().enumerate() {
        match elem {
            mir::ProjectionElem::Deref if index == 0 && !dereference_base => {
                // CUDA device pointers are one 64-bit word. Requiring the source
                // pointer/reference layout to match rejects fat pointers such as
                // `&[T]`/`&str` before we model them with the wrong DWARF stack op.
                if current_ty.layout().ok()?.shape().size.bytes() != 8 {
                    return None;
                }
                current_ty = match current_ty.kind() {
                    TyKind::RigidTy(RigidTy::RawPtr(pointee, _)) => pointee,
                    TyKind::RigidTy(RigidTy::Ref(_, pointee, _)) => pointee,
                    _ => return None,
                };
                dereference_base = true;
            }
            mir::ProjectionElem::Downcast(variant) => {
                // Downcasts are supported only inside the base local: after a
                // dereference the tested recipe allows static fields alone.
                if enum_variant.is_some() || dereference_base {
                    return None;
                }
                let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = current_ty.kind() else {
                    return None;
                };
                if !matches!(adt_def.kind(), rustc_public::ty::AdtKind::Enum) {
                    return None;
                }
                enum_variant = Some(variant.to_index());
            }
            mir::ProjectionElem::Field(field_idx, field_ty) => {
                let layout = current_ty.layout().ok()?;
                let shape = layout.shape();
                let field_offset = if let Some(variant) = enum_variant.take() {
                    crate::translator::layout::enum_variant_field_offsets(
                        &shape,
                        variant,
                        pliron::location::Location::Unknown,
                    )
                    .ok()?
                    .get(*field_idx)
                    .copied()? as u64
                } else {
                    let rustc_public::abi::FieldsShape::Arbitrary { offsets } = &shape.fields
                    else {
                        return None;
                    };
                    offsets.get(*field_idx)?.bytes() as u64
                };
                offset_bytes = offset_bytes.checked_add(field_offset)?;
                current_ty = *field_ty;
            }
            mir::ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end: false,
            } if !dereference_base => {
                if enum_variant.is_some() {
                    return None;
                }
                let layout = current_ty.layout().ok()?;
                let shape = layout.shape();
                let rustc_public::abi::FieldsShape::Array { stride, count } = &shape.fields else {
                    return None;
                };
                if *offset >= *count {
                    return None;
                }
                let element_offset = (stride.bytes() as u64).checked_mul(*offset)?;
                offset_bytes = offset_bytes.checked_add(element_offset)?;
                let TyKind::RigidTy(RigidTy::Array(element, _)) = current_ty.kind() else {
                    return None;
                };
                current_ty = element;
            }
            _ => return None,
        }
    }

    if enum_variant.is_some() {
        return None;
    }

    Some(ResolvedDebugProjection {
        dereference_base,
        offset_bytes,
        ty: current_ty,
    })
}

fn debug_source_position(span: rustc_public::ty::Span) -> Option<DebugSourcePosition> {
    let file = span.get_filename();
    let lines = span.get_lines();
    if file.is_empty() || lines.start_line == 0 || lines.start_col == 0 {
        return None;
    }
    Some(DebugSourcePosition {
        file: std::path::PathBuf::from(file),
        line: lines.start_line as i32,
        column: lines.start_col as i32,
    })
}

/// Source-level names for MIR locals, independent of the selected debug tier.
///
/// Full variable debug information is deliberately optional, but the local
/// memory diagnostic still needs a useful source identity in optimized builds.
/// `var_debug_info` is already available in stable MIR and does not force LLVM
/// debug metadata emission, so keep this lightweight map separate from
/// [`collect_debug_locals`].
fn collect_local_source_names(body: &mir::Body) -> FxHashMap<mir::Local, String> {
    let mut names = FxHashMap::default();
    for info in &body.var_debug_info {
        let local = match &info.value {
            mir::VarDebugInfoContents::Place(place) if place.projection.is_empty() => place.local,
            mir::VarDebugInfoContents::Place(_) | mir::VarDebugInfoContents::Const(_) => continue,
        };
        let name = info.name.to_string();
        if !name.is_empty() {
            names.entry(local).or_insert(name);
        }
    }
    names
}

/// Compact source-level type spelling for local-memory diagnostics.
fn local_memory_type_name(ty: &Ty) -> String {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => "bool".to_string(),
        TyKind::RigidTy(RigidTy::Int(int_ty)) => int_name(int_ty).to_string(),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => uint_name(uint_ty).to_string(),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => float_name(float_ty).to_string(),
        TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
            raw_pointer_name(pointee, mutability)
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
            reference_name(pointee, mutability)
        }
        TyKind::RigidTy(RigidTy::Tuple(subtypes)) => format!(
            "({})",
            subtypes
                .iter()
                .map(local_memory_type_name)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TyKind::RigidTy(RigidTy::Array(element, len)) => {
            let count = array_len_const(&len)
                .map(|count| count.to_string())
                .unwrap_or_else(|| "?".to_string());
            format!("[{}; {count}]", local_memory_type_name(&element))
        }
        TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => adt_def.trimmed_name(),
        // Closure and coroutine environments reach here through
        // `var_debug_info` merged from MIR-inlined callees (iterator adapters
        // name their closure parameters, e.g. `f`). Their `{ty:?}` dump spells
        // DefIds and generic args recursively and can run to many kilobytes,
        // which would then be hex-encoded into an SSA value name; LLVM's
        // textual parser mis-lexes identifiers that long. Spell them the way
        // rustc diagnostics do instead.
        TyKind::RigidTy(RigidTy::Closure(..)) => "{closure}".to_string(),
        TyKind::RigidTy(
            RigidTy::Coroutine(..) | RigidTy::CoroutineClosure(..) | RigidTy::CoroutineWitness(..),
        ) => "{coroutine}".to_string(),
        _ => bounded_type_spelling(ty),
    }
}

/// Debug-format spelling for type kinds without a dedicated compact arm,
/// hard-capped in length.
///
/// The spelling exists to be read in a one-line warning and travels inside an
/// SSA value name, so an unbounded `{ty:?}` dump is never acceptable here even
/// for kinds this function does not anticipate.
fn bounded_type_spelling(ty: &Ty) -> String {
    const MAX_TYPE_SPELLING_BYTES: usize = 64;
    let mut spelled = format!("{ty:?}");
    if spelled.len() > MAX_TYPE_SPELLING_BYTES {
        let mut cut = MAX_TYPE_SPELLING_BYTES;
        while !spelled.is_char_boundary(cut) {
            cut -= 1;
        }
        spelled.truncate(cut);
        spelled.push_str("...");
    }
    spelled
}

/// Describe one MIR local as the provenance attribute carried by `mir.alloca`.
///
/// The attribute stays a first-class IR citizen through lowering; only the
/// textual LLVM exporter serializes it (hex-encoded into the alloca's SSA
/// name), so arbitrary Rust identifiers and type spellings cannot make
/// invalid LLVM IR.
fn local_memory_provenance(local_idx: usize, name: &str, ty: &Ty) -> LocalMemoryProvenanceAttr {
    let size_bytes = ty
        .layout()
        .ok()
        .map(|layout| layout.shape().size.bytes() as u64)
        .unwrap_or(0);
    LocalMemoryProvenanceAttr {
        local_index: local_idx as u64,
        size_bytes,
        binding_name: name.into(),
        type_name: local_memory_type_name(ty).into(),
    }
}

/// Maximum nesting depth for composite debug types. Guards against deeply
/// nested or (via generics) pathological value-type trees; beyond this we omit
/// the inner detail rather than recurse without bound.
const MAX_DEBUG_TYPE_DEPTH: usize = 8;

pub(crate) fn debug_type_for_ty(ty: &Ty) -> Option<DebugLocalTypeKind> {
    debug_type_for_ty_at(ty, 0)
}

/// Describe the compiler-materialized backing array for a shared-memory marker.
///
/// `SharedArray<T, N>` is a zero-sized Rust marker, but its device storage is
/// the physical `[T; N]` allocation created by `mir.shared_alloc`. Building the
/// debug type from `T` and `N` keeps that physical/logical object independent
/// of the marker's own zero-sized layout.
pub(crate) fn debug_shared_array_type(element_ty: &Ty, count: u64) -> Option<DebugLocalTypeKind> {
    if count == 0 {
        return None;
    }
    let element = debug_type_for_ty_at(element_ty, 1)?;
    if !debug_type_graph_supported_for_shared(&element) {
        return None;
    }
    let element_size_bits = layout_size_bits(element_ty)?;
    if element.size_bits() != element_size_bits {
        return None;
    }
    let size_bits = element_size_bits.checked_mul(count)?;
    Some(DebugLocalTypeKind::Array {
        name: format!("[{}; {count}]", short_ty_name(element_ty)),
        size_bits,
        element: Box::new(element),
        count,
    })
}

/// Opaque `Pointer` debug types carry no pointee type. Emitting them as an
/// AS3 array element (including through a composite) would advertise an
/// array-of-void-pointer shape, so those globals are omitted. `TypedPointer`
/// carries a complete finite pointee tree and is admitted when that tree is
/// itself supported.
fn debug_type_graph_supported_for_shared(ty: &DebugLocalTypeKind) -> bool {
    match ty {
        DebugLocalTypeKind::Basic { .. } => true,
        DebugLocalTypeKind::Pointer { .. } => false,
        DebugLocalTypeKind::TypedPointer { pointee, .. } => {
            debug_type_graph_supported_for_shared(pointee)
        }
        DebugLocalTypeKind::Array { element, .. } => debug_type_graph_supported_for_shared(element),
        DebugLocalTypeKind::Struct { members, .. } => members
            .iter()
            .all(|member| debug_type_graph_supported_for_shared(&member.ty)),
        DebugLocalTypeKind::Enum {
            discriminant,
            variants,
            ..
        } => {
            discriminant
                .as_ref()
                .is_none_or(|discriminant| debug_type_graph_supported_for_shared(&discriminant.ty))
                && variants.iter().all(|variant| {
                    variant
                        .members
                        .iter()
                        .all(|member| debug_type_graph_supported_for_shared(&member.ty))
                })
        }
    }
}

fn debug_type_for_ty_at(ty: &Ty, depth: usize) -> Option<DebugLocalTypeKind> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => Some(DebugLocalTypeKind::Basic {
            name: "bool".to_string(),
            size_bits: 8,
            encoding: "DW_ATE_boolean",
        }),
        TyKind::RigidTy(RigidTy::Int(int_ty)) => Some(DebugLocalTypeKind::Basic {
            name: int_name(int_ty).to_string(),
            size_bits: (int_ty.num_bytes() * 8) as u64,
            encoding: "DW_ATE_signed",
        }),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => Some(DebugLocalTypeKind::Basic {
            name: uint_name(uint_ty).to_string(),
            size_bits: (uint_ty.num_bytes() * 8) as u64,
            encoding: "DW_ATE_unsigned",
        }),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => Some(DebugLocalTypeKind::Basic {
            name: float_name(float_ty).to_string(),
            size_bits: float_size_bits(float_ty),
            encoding: "DW_ATE_float",
        }),
        TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
            debug_typed_pointer_type(ty, &pointee, DebugPointerKind::Raw(mutability), depth)
                .or_else(|| {
                    Some(DebugLocalTypeKind::Pointer {
                        name: raw_pointer_name(pointee, mutability),
                        size_bits: 64,
                    })
                })
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
            debug_typed_pointer_type(ty, &pointee, DebugPointerKind::Reference(mutability), depth)
                .or_else(|| {
                    Some(DebugLocalTypeKind::Pointer {
                        name: reference_name(pointee, mutability),
                        size_bits: 64,
                    })
                })
        }
        TyKind::RigidTy(RigidTy::Closure(closure_def, substs)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            let upvar_tys = types::closure_upvar_tys(&substs)?;
            let fields = upvar_tys
                .into_iter()
                .enumerate()
                .map(|(idx, upvar_ty)| (format!("capture_{idx}"), upvar_ty));
            debug_struct_type(ty, format!("{:?}", closure_def.def_id()), fields, depth)
        }
        TyKind::RigidTy(RigidTy::Tuple(subtypes)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            let name = format!(
                "({})",
                subtypes
                    .iter()
                    .map(short_ty_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let fields = subtypes
                .iter()
                .enumerate()
                .map(|(idx, sub)| (format!("__{idx}"), *sub));
            debug_struct_type(ty, name, fields, depth)
        }
        TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            match adt_def.kind() {
                rustc_public::ty::AdtKind::Struct => {
                    let variants = adt_def.variants();
                    if variants.len() != 1 {
                        return None;
                    }
                    let name = adt_def.trimmed_name();
                    let fields = variants[0]
                        .fields()
                        .into_iter()
                        .map(|field| (field.name.to_string(), field.ty_with_args(&substs)));
                    debug_struct_type(ty, name, fields, depth)
                }
                rustc_public::ty::AdtKind::Enum => debug_enum_type(ty, depth),
                rustc_public::ty::AdtKind::Union => None,
            }
        }
        TyKind::RigidTy(RigidTy::Array(elem_ty, len_const)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            let count = array_len_const(&len_const)?;
            let element = debug_type_for_ty_at(&elem_ty, depth + 1)?;
            let size_bits = layout_size_bits(ty)?;
            Some(DebugLocalTypeKind::Array {
                name: format!("[{}; {count}]", short_ty_name(&elem_ty)),
                size_bits,
                element: Box::new(element),
                count,
            })
        }
        _ => None,
    }
}

/// Describe a one-word pointer/reference and its safely bounded pointee.
///
/// Rust references and raw pointers to dynamically-sized types are fat
/// pointers. Encoding those as a typed `DW_TAG_pointer_type` would discard
/// their metadata word, so this helper declines them and lets the caller retain
/// the legacy opaque pointer metadata. The same fallback applies when a pointee
/// is unsupported or exceeds the recursion bound. Composite pointees are
/// deliberately excluded here: `DebugLocalTypeKind` is currently a tree, so
/// following an ADT such as `Node { next: *const Node }` would recursively
/// unroll the same type instead of representing the cyclic DWARF graph.
#[derive(Clone, Copy)]
enum DebugPointerKind {
    Raw(mir::Mutability),
    Reference(mir::Mutability),
}

fn debug_typed_pointer_type(
    pointer: &Ty,
    pointee: &Ty,
    kind: DebugPointerKind,
    depth: usize,
) -> Option<DebugLocalTypeKind> {
    if depth >= MAX_DEBUG_TYPE_DEPTH {
        return None;
    }

    let size_bits = layout_size_bits(pointer)?;
    let pointer_word_bits =
        rustc_public::target::MachineInfo::target_pointer_width().bytes() as u64 * 8;
    if size_bits != pointer_word_bits {
        return None;
    }

    // Build and validate the bounded pointee first. The source-facing pointer
    // name is then derived from that accepted tree, so rejected pathological
    // source types cannot recurse or grow a name before the depth gate fires.
    let pointee = debug_pointer_pointee_type(pointee, depth + 1)?;
    let name = debug_pointer_name(kind, &pointee);
    Some(DebugLocalTypeKind::TypedPointer {
        name,
        size_bits,
        pointee: Box::new(pointee),
    })
}

fn debug_pointer_name(kind: DebugPointerKind, pointee: &DebugLocalTypeKind) -> String {
    let pointee = typed_pointer_pointee_display_name(pointee);
    match kind {
        DebugPointerKind::Raw(mir::Mutability::Mut) => format!("*mut {pointee}"),
        DebugPointerKind::Raw(mir::Mutability::Not) => format!("*const {pointee}"),
        DebugPointerKind::Reference(mir::Mutability::Mut) => format!("&mut {pointee}"),
        DebugPointerKind::Reference(mir::Mutability::Not) => format!("&{pointee}"),
    }
}

fn typed_pointer_pointee_display_name(ty: &DebugLocalTypeKind) -> &str {
    match ty {
        DebugLocalTypeKind::Basic { name, .. }
        | DebugLocalTypeKind::TypedPointer { name, .. }
        | DebugLocalTypeKind::Array { name, .. } => name,
        DebugLocalTypeKind::Pointer { .. }
        | DebugLocalTypeKind::Struct { .. }
        | DebugLocalTypeKind::Enum { .. } => {
            unreachable!("typed pointer pointees use only the bounded typed subset")
        }
    }
}

/// Build the finite subset that is safe to embed beneath a pointer node.
///
/// Primitive leaves, fixed arrays whose elements stay inside this subset, and
/// nested thin pointers/references form an acyclic tree. Tuples, ADTs, enums,
/// closures, and every other composite are rejected even when the general
/// local-variable debug path can describe part of them. `char` and unit are
/// also rejected until `DebugLocalTypeKind` has an accurate encoding for them;
/// treating either as an integer or null-base pointer would be misleading.
fn debug_pointer_pointee_type(ty: &Ty, depth: usize) -> Option<DebugLocalTypeKind> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => Some(DebugLocalTypeKind::Basic {
            name: "bool".to_string(),
            size_bits: 8,
            encoding: "DW_ATE_boolean",
        }),
        TyKind::RigidTy(RigidTy::Int(int_ty)) => Some(DebugLocalTypeKind::Basic {
            name: int_name(int_ty).to_string(),
            size_bits: (int_ty.num_bytes() * 8) as u64,
            encoding: "DW_ATE_signed",
        }),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => Some(DebugLocalTypeKind::Basic {
            name: uint_name(uint_ty).to_string(),
            size_bits: (uint_ty.num_bytes() * 8) as u64,
            encoding: "DW_ATE_unsigned",
        }),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => Some(DebugLocalTypeKind::Basic {
            name: float_name(float_ty).to_string(),
            size_bits: float_size_bits(float_ty),
            encoding: "DW_ATE_float",
        }),
        TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
            debug_typed_pointer_type(ty, &pointee, DebugPointerKind::Raw(mutability), depth)
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
            debug_typed_pointer_type(ty, &pointee, DebugPointerKind::Reference(mutability), depth)
        }
        TyKind::RigidTy(RigidTy::Array(element, len)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            let count = array_len_const(&len)?;
            let element_debug = debug_pointer_pointee_type(&element, depth + 1)?;
            let element_name = typed_pointer_pointee_display_name(&element_debug);
            Some(DebugLocalTypeKind::Array {
                name: format!("[{element_name}; {count}]"),
                size_bits: layout_size_bits(ty)?,
                element: Box::new(element_debug),
                count,
            })
        }
        _ => None,
    }
}

/// Build a `DICompositeType`-shaped struct/tuple from rustc's real layout.
///
/// Member offsets come from `ty.layout()` (so `repr(Rust)` field reordering is
/// honored), not declaration order. Fields whose type we cannot yet describe,
/// and zero-sized fields (e.g. `PhantomData`), are omitted; the remaining
/// members keep their correct offsets.
fn debug_struct_type(
    ty: &Ty,
    name: String,
    fields: impl Iterator<Item = (String, Ty)>,
    depth: usize,
) -> Option<DebugLocalTypeKind> {
    let layout = ty.layout().ok()?;
    let shape = layout.shape();
    let offsets: Vec<u64> = match &shape.fields {
        rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
            offsets.iter().map(|off| off.bytes() as u64).collect()
        }
        _ => return None,
    };
    let size_bits = shape.size.bytes() as u64 * 8;

    let mut members = Vec::new();
    for (idx, (field_name, field_ty)) in fields.enumerate() {
        let offset_bytes = *offsets.get(idx)?;
        let Some(member_ty) = debug_type_for_ty_at(&field_ty, depth + 1) else {
            continue;
        };
        if member_ty.size_bits() == 0 {
            continue;
        }
        members.push(DebugTypeMember {
            name: field_name,
            offset_bits: offset_bytes * 8,
            ty: member_ty,
        });
    }

    if members.is_empty() {
        return None;
    }

    Some(DebugLocalTypeKind::Struct {
        name,
        size_bits,
        members,
    })
}

/// Build a Rust enum debug type using rustc's physical layout.
///
/// This mirrors rustc's native DWARF representation rather than guessing from
/// source-level `Option`/`Result` shapes: a top-level structure contains a
/// variant part, whose discriminator is either the direct integer tag or the
/// integer-normalized niche carrier. Variant payload fields use rustc's exact
/// per-variant offsets. For niche layouts the untagged variant deliberately has
/// no discriminant value, so the debugger treats it as the default branch.
fn debug_enum_type(ty: &Ty, depth: usize) -> Option<DebugLocalTypeKind> {
    let TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) = ty.kind() else {
        return None;
    };
    if !matches!(adt_def.kind(), rustc_public::ty::AdtKind::Enum) {
        return None;
    }

    #[derive(Clone, Copy)]
    enum DebugEnumLayout {
        Direct {
            width: u64,
        },
        Niche {
            width: u64,
            niche_variant_start: usize,
            niche_start: u128,
            untagged_variant: usize,
        },
        Single,
        Empty,
    }

    let layout_shape = ty.layout().ok()?.shape();
    let size_bits = layout_shape.size.bytes() as u64 * 8;

    let (discriminant, debug_layout) = match &layout_shape.variants {
        rustc_public::abi::VariantsShape::Multiple {
            tag,
            tag_encoding: rustc_public::abi::TagEncoding::Direct,
            tag_field,
            ..
        } => {
            let primitive = match tag {
                rustc_public::abi::Scalar::Initialized { value, .. }
                | rustc_public::abi::Scalar::Union { value } => *value,
            };
            let rustc_public::abi::Primitive::Int { length, signed } = primitive else {
                return None;
            };
            let width = length.bits() as u64;
            if width == 0 || width > 64 {
                return None;
            }
            let offset_bits = crate::translator::layout::enum_tag_offset(
                &layout_shape.fields,
                *tag_field,
                pliron::location::Location::Unknown,
            )
            .ok()? as u64
                * 8;
            let tag_ty = DebugLocalTypeKind::Basic {
                name: format!("{}{}", if signed { "i" } else { "u" }, width),
                size_bits: width,
                encoding: if signed {
                    "DW_ATE_signed"
                } else {
                    "DW_ATE_unsigned"
                },
            };
            (
                Some(DebugEnumDiscriminant {
                    offset_bits,
                    ty: Box::new(tag_ty),
                }),
                DebugEnumLayout::Direct { width },
            )
        }
        rustc_public::abi::VariantsShape::Multiple {
            tag,
            tag_encoding:
                rustc_public::abi::TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                },
            tag_field,
            ..
        } => {
            let primitive = match tag {
                rustc_public::abi::Scalar::Initialized { value, .. }
                | rustc_public::abi::Scalar::Union { value } => *value,
            };
            let width = primitive
                .size(&rustc_public::target::MachineInfo::target())
                .bits() as u64;
            if width == 0 || width > 64 {
                return None;
            }
            let offset_bits = crate::translator::layout::enum_tag_offset(
                &layout_shape.fields,
                *tag_field,
                pliron::location::Location::Unknown,
            )
            .ok()? as u64
                * 8;

            // rustc normalizes niche carriers, including pointer niches, to an
            // unsigned integer of the same physical width for DWARF.
            let tag_name = match primitive {
                rustc_public::abi::Primitive::Pointer(_) if width == 64 => "usize".to_string(),
                _ => format!("u{width}"),
            };
            let tag_ty = DebugLocalTypeKind::Basic {
                name: tag_name,
                size_bits: width,
                encoding: "DW_ATE_unsigned",
            };
            (
                Some(DebugEnumDiscriminant {
                    offset_bits,
                    ty: Box::new(tag_ty),
                }),
                DebugEnumLayout::Niche {
                    width,
                    niche_variant_start: niche_variants.start().to_index(),
                    niche_start: *niche_start,
                    untagged_variant: untagged_variant.to_index(),
                },
            )
        }
        rustc_public::abi::VariantsShape::Single { .. } => (None, DebugEnumLayout::Single),
        rustc_public::abi::VariantsShape::Empty => (None, DebugEnumLayout::Empty),
    };

    let source_variants = adt_def.variants();
    let mut variants = Vec::with_capacity(source_variants.len());

    for (variant_index, variant) in source_variants.iter().enumerate() {
        let fields = variant.fields();
        let field_offsets: Vec<u64> = match &layout_shape.variants {
            rustc_public::abi::VariantsShape::Single { index }
                if index.to_index() != variant_index =>
            {
                vec![0; fields.len()]
            }
            rustc_public::abi::VariantsShape::Empty => vec![0; fields.len()],
            _ => crate::translator::layout::enum_variant_field_offsets(
                &layout_shape,
                variant_index,
                pliron::location::Location::Unknown,
            )
            .ok()?
            .into_iter()
            .map(|offset| offset as u64)
            .collect(),
        };

        let mut members = Vec::new();
        for (field_index, field) in fields.into_iter().enumerate() {
            let field_ty = field.ty_with_args(&substs);
            let Some(member_ty) = debug_type_for_ty_at(&field_ty, depth + 1) else {
                continue;
            };
            if member_ty.size_bits() == 0 {
                continue;
            }
            let offset_bytes = *field_offsets.get(field_index)?;
            let source_name = field.name.to_string();
            let member_name = if source_name.parse::<usize>().ok() == Some(field_index) {
                format!("__{field_index}")
            } else {
                source_name
            };
            members.push(DebugTypeMember {
                name: member_name,
                offset_bits: offset_bytes * 8,
                ty: member_ty,
            });
        }

        let discriminant_value = match debug_layout {
            DebugEnumLayout::Direct { width } => {
                let variant_idx = rustc_public::ty::VariantIdx::to_val(variant_index);
                let raw = adt_def.discriminant_for_variant(variant_idx).val;
                truncate_debug_discriminant(raw, width)
            }
            DebugEnumLayout::Niche {
                width,
                niche_variant_start,
                niche_start,
                untagged_variant,
            } => {
                if variant_index == untagged_variant {
                    None
                } else {
                    let raw = (variant_index as u128)
                        .wrapping_sub(niche_variant_start as u128)
                        .wrapping_add(niche_start);
                    truncate_debug_discriminant(raw, width)
                }
            }
            DebugEnumLayout::Single | DebugEnumLayout::Empty => None,
        };

        variants.push(DebugEnumVariant {
            name: variant.name().to_string(),
            discriminant: discriminant_value,
            members,
        });
    }

    Some(DebugLocalTypeKind::Enum {
        name: adt_def.trimmed_name(),
        size_bits,
        discriminant,
        variants,
    })
}

/// Truncate a physical discriminant to the width LLVM will attach as
/// `extraData` on the corresponding variant member.
fn truncate_debug_discriminant(value: u128, width: u64) -> Option<u64> {
    if width == 0 || width > 64 {
        return None;
    }
    let mask = if width == 64 {
        u128::from(u64::MAX)
    } else {
        (1u128 << width) - 1
    };
    Some((value & mask) as u64)
}

/// Total size of `ty` in bits from its layout, or `None` if unavailable.
fn layout_size_bits(ty: &Ty) -> Option<u64> {
    Some(ty.layout().ok()?.shape().size.bytes() as u64 * 8)
}

/// Evaluate a fixed array's length constant to a `u64`.
fn array_len_const(len_const: &rustc_public::ty::TyConst) -> Option<u64> {
    match len_const.kind() {
        rustc_public::ty::TyConstKind::Value(_, alloc) => {
            let mut arr = [0u8; 8];
            for (i, byte) in alloc.bytes.iter().take(8).enumerate() {
                arr[i] = (*byte)?;
            }
            Some(u64::from_le_bytes(arr))
        }
        _ => None,
    }
}

/// A short, human-readable name for a type, used only for composite display.
fn short_ty_name(ty: &Ty) -> String {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => "bool".to_string(),
        TyKind::RigidTy(RigidTy::Int(int_ty)) => int_name(int_ty).to_string(),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => uint_name(uint_ty).to_string(),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => float_name(float_ty).to_string(),
        TyKind::RigidTy(RigidTy::RawPtr(..)) | TyKind::RigidTy(RigidTy::Ref(..)) => {
            "ptr".to_string()
        }
        TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => adt_def.trimmed_name(),
        _ => "_".to_string(),
    }
}

fn int_name(ty: IntTy) -> &'static str {
    match ty {
        IntTy::Isize => "isize",
        IntTy::I8 => "i8",
        IntTy::I16 => "i16",
        IntTy::I32 => "i32",
        IntTy::I64 => "i64",
        IntTy::I128 => "i128",
    }
}

fn uint_name(ty: UintTy) -> &'static str {
    match ty {
        UintTy::Usize => "usize",
        UintTy::U8 => "u8",
        UintTy::U16 => "u16",
        UintTy::U32 => "u32",
        UintTy::U64 => "u64",
        UintTy::U128 => "u128",
    }
}

fn float_name(ty: FloatTy) -> &'static str {
    match ty {
        FloatTy::F16 => "f16",
        FloatTy::F32 => "f32",
        FloatTy::F64 => "f64",
        FloatTy::F128 => "f128",
    }
}

fn float_size_bits(ty: FloatTy) -> u64 {
    match ty {
        FloatTy::F16 => 16,
        FloatTy::F32 => 32,
        FloatTy::F64 => 64,
        FloatTy::F128 => 128,
    }
}

fn raw_pointer_name(pointee: Ty, mutability: mir::Mutability) -> String {
    let mutability = match mutability {
        mir::Mutability::Mut => "mut ",
        mir::Mutability::Not => "const ",
    };
    format!("*{mutability}{}", simple_type_name(&pointee))
}

fn reference_name(pointee: Ty, mutability: mir::Mutability) -> String {
    let mutability = match mutability {
        mir::Mutability::Mut => "mut ",
        mir::Mutability::Not => "",
    };
    format!("&{mutability}{}", simple_type_name(&pointee))
}

fn simple_type_name(ty: &Ty) -> &'static str {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => "bool",
        TyKind::RigidTy(RigidTy::Int(int_ty)) => int_name(int_ty),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => uint_name(uint_ty),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => float_name(float_ty),
        _ => "_",
    }
}

/// Emit one `mir.alloca` per non-ZST MIR local at the top of the entry block,
/// then store each function argument into its backing slot.
///
/// This is the foundation of the alloca + load/store translator model: every
/// non-ZST MIR local is backed by a single stack slot recorded in `value_map`
/// via [`ValueMap::set_slot`]. Function arguments (which arrive as entry-block
/// arguments) are immediately stored into their slots so subsequent blocks can
/// load them without needing SSA block arguments.
///
/// `num_args` is the number of function arguments (MIR locals `1..=num_args`).
///
/// Returns the last operation emitted, so the caller can pass it to
/// [`block::translate_block`] as `entry_prev_op` to append block contents
/// **after** this setup (otherwise `insert_at_front` would push the alloca
/// chain past the block terminator).
///
/// # ZST locals
///
/// Locals whose Rust type is zero-sized (unit tuple, empty structs, `!`, …)
/// are skipped entirely: they get no slot in [`ValueMap`] and any attempted
/// load/store short-circuits.
///
/// # Unsupported types
///
/// [`types::translate_type`] can fail for locals whose types aren't supported
/// yet (e.g. ghost locals in kernels targeting unsupported surfaces). Those
/// locals simply get no slot; any later attempt to use them still errors out
/// through the existing unsupported-type code paths.
fn emit_entry_allocas(
    ctx: &mut Context,
    body: &mir::Body,
    entry_block: Ptr<BasicBlock>,
    num_args: usize,
    value_map: &mut ValueMap,
    debug_kind: DebugKind,
    debug_source_scopes: Option<&DebugSourceScopeMap>,
    reachable: &std::collections::BTreeSet<usize>,
) -> Option<Ptr<Operation>> {
    let mut prev_op: Option<Ptr<Operation>> = None;
    let debug_locals = if debug_kind.variables_enabled() {
        collect_debug_locals(ctx, body)
    } else {
        CollectedDebugLocals::default()
    };
    let local_source_names = collect_local_source_names(body);

    // Translate local types once up front. The address-space analyzer uses
    // each pointer local's declared lowering as the conservative fallback for
    // writes it cannot classify, and the allocation loop reuses the same
    // handles below.
    let mut mir_types = Vec::with_capacity(body.locals().len());
    for local_decl in body.locals() {
        let mir_ty = if types::is_rust_type_zst(&local_decl.ty) {
            None
        } else {
            types::translate_type(ctx, &local_decl.ty).ok()
        };
        mir_types.push(mir_ty);
    }
    let declared_addr_spaces: Vec<Option<u32>> = mir_types
        .iter()
        .map(|mir_ty| {
            mir_ty
                .as_ref()
                .and_then(|mir_ty| values::pointer_addr_space(ctx, *mir_ty))
        })
        .collect();

    // Pre-scan only rustc-reachable writes. A slot is narrowed to a concrete
    // address space only when every reachable write agrees; unknown writes
    // retain their declared lowering (normally generic address space zero).
    let slot_addr_spaces =
        SlotAddrSpaceMap::analyze(body, reachable, num_args, &declared_addr_spaces);

    for local_idx in 0..body.locals().len() {
        let local = mir::Local::from(local_idx);
        let Some(mir_ty) = mir_types[local_idx] else {
            continue;
        };

        // Override the Rust-declared addrspace with the inferred one for
        // pointer slots. Non-pointer slots are untouched by
        // `align_pointer_addr_space`.
        let rust_declared = declared_addr_spaces[local_idx].unwrap_or(address_space::GENERIC);
        let target = slot_addr_spaces.effective(local, rust_declared);
        let mir_ty = values::align_pointer_addr_space(ctx, mir_ty, target);

        let (op, slot) = ValueMap::emit_alloca(ctx, mir_ty, entry_block, prev_op);

        // Tag only named Rust source locals. Compiler temporaries and lowering-
        // synthesized LLVM allocas must not turn verbose builds into a stream of
        // warnings that cannot be attributed back to user code.
        if let Some(source_name) = local_source_names.get(&local)
            && let Some(decl) = body.local_decl(local)
        {
            llvm_export::ops::set_local_memory_provenance(
                ctx,
                op,
                local_memory_provenance(local_idx, source_name, &decl.ty),
            );
        }

        if let Some(infos) = debug_locals.whole.get(&local)
            && let Some(info) = infos.first()
        {
            llvm_export::ops::set_debug_local_variable(ctx, op, info.variable.clone());
            if debug_source_scopes
                .is_some_and(|map| map.scopes.iter().any(|scope| scope.id == info.source_scope))
            {
                llvm_export::ops::set_debug_local_source_scope(ctx, op, info.source_scope);
            }
            op.deref_mut(ctx).set_loc(info.loc.clone());
            let aliases: Vec<_> = infos[1..]
                .iter()
                .map(|alias| DebugWholeVariableInfo {
                    variable: alias.variable.clone(),
                    source_scope: Some(alias.source_scope),
                    declaration: alias.declaration.clone(),
                })
                .collect();
            if !aliases.is_empty() {
                llvm_export::ops::set_debug_whole_variable_aliases(ctx, op, &aliases);
            }
        }
        if let Some(projected) = debug_locals.projected.get(&local) {
            llvm_export::ops::set_debug_projected_variables(ctx, op, projected);
        }
        if let Some(fragments) = debug_locals.fragments.get(&local) {
            llvm_export::ops::set_debug_fragment_variables(ctx, op, fragments);
        }
        prev_op = Some(op);
        value_map.set_slot(local, slot);
    }

    for arg_idx in 0..num_args {
        let local = mir::Local::from(arg_idx + 1);
        let block_arg = entry_block.deref(ctx).get_argument(arg_idx);
        if let Some(op) = value_map.store_local(ctx, local, block_arg, entry_block, prev_op) {
            prev_op = Some(op);
        }
    }

    prev_op
}

/// Translates a MIR function body to a pliron IR `mir.func` operation.
///
/// # Process
///
/// 1. Extract signature (arg types from MIR locals 1..N, return from local 0)
/// 2. Create `mir.func` with signature and optional `gpu_kernel` attribute
/// 3. Create one pliron block per MIR block. The entry block carries the
///    function parameters; every other block is argument-less (cross-block
///    data flow travels through per-local alloca slots)
/// 4. Emit one `mir.alloca` per non-ZST local at the top of the entry block
///    and seed the argument slots from the block's parameters
/// 5. Translate every reachable block in index order
///
/// # Arguments
///
/// * `ctx` - Pliron IR context
/// * `body` - MIR function body
/// * `instance` - Monomorphized instance (with concrete generic args)
/// * `rustc_mir_block_count` - Block count recorded from the rustc MIR body
///   before conversion to public MIR
/// * `rustc_mono_successors` - Exact per-block successor edges computed by
///   rustc's monomorphization rules under the device runtime-check policy
/// * `is_kernel` - Add `gpu_kernel` attribute for kernel entry points
/// * `is_inline_always` - Add `alwaysinline` attribute (non-kernel functions
///   marked `#[inline(always)]` in rustc)
/// * `override_name` - Custom export name (defaults to instance name)
pub fn translate_body(
    ctx: &mut Context,
    body: &mir::Body,
    instance: &mono::Instance,
    rustc_mir_block_count: usize,
    rustc_mono_successors: &[Vec<usize>],
    is_kernel: bool,
    is_inline_always: bool,
    override_name: Option<&str>,
    legaliser: &mut Legaliser,
    debug_kind: DebugKind,
    debug_source_scopes: Option<&DebugSourceScopeMap>,
    statement_debug_info: Option<&StatementDebugInfoMap>,
) -> TranslationResult<Ptr<Operation>> {
    if let Some(statement_debug_info) = statement_debug_info {
        if statement_debug_info.blocks.len() != body.blocks.len() {
            return input_err_noloc!(TranslationErr::invalid_op(format!(
                "statement debug sidecar has {} blocks, but stable MIR has {}",
                statement_debug_info.blocks.len(),
                body.blocks.len()
            )));
        }
        for (block_index, (debug_block, mir_block)) in statement_debug_info
            .blocks
            .iter()
            .zip(&body.blocks)
            .enumerate()
        {
            if debug_block.before_statements.len() != mir_block.statements.len() {
                return input_err_noloc!(TranslationErr::invalid_op(format!(
                    "statement debug sidecar block {block_index} has {} statement boundaries, but stable MIR has {} statements",
                    debug_block.before_statements.len(),
                    mir_block.statements.len()
                )));
            }
        }
    }

    // Establish and validate rustc's exact per-instance reachability before
    // any whole-body semantic scan. Dead blocks must not influence function
    // attributes, pointer-slot address spaces, or later code emission.
    if let Err(error) =
        validate_monomorphized_successors(body, rustc_mir_block_count, rustc_mono_successors)
    {
        return input_err_noloc!(TranslationErr::invalid_op(error));
    }
    let reachable = compute_reachable_blocks(body, rustc_mono_successors);

    // Create a value map to track MIR locals -> pliron IR values
    let num_locals = body.locals().len();
    let mut value_map = ValueMap::new(num_locals);
    value_map.set_debug_variables(debug_kind.variables_enabled());

    // Resolve the per-body unchecked-indexing policy. Like the dynamic-shared
    // marker, the `#[kernel(unchecked_indexing)]` marker is scanned on any
    // function: generic kernel expansion forwards it to the generated entry
    // but also keeps the original in the `#[inline(always)]` implementation
    // helper, and either body may be the one translated here. The whole-build
    // environment switch additionally covers separately translated
    // `#[device]` functions that carry no marker.
    let unchecked_indexing = match detect_unchecked_indexing_config(body, &reachable) {
        Ok(marker_enabled) => marker_enabled || unchecked_indexing_env_enabled(),
        Err(error) => {
            return input_err_noloc!(TranslationErr::invalid_op(error));
        }
    };
    value_map.set_unchecked_indexing(unchecked_indexing);
    if unchecked_indexing && std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
        eprintln!("  Unchecked indexing enabled: bounds-check asserts elided");
    }

    // Get function argument types for the first block
    // In MIR, locals[0] is the return value, locals[1..arg_count+1] are function arguments
    let mut arg_types = Vec::new();

    // Determine argument count from the function type in the instance
    // Get the function signature to determine the number of arguments
    let fn_ty = instance.ty();
    let num_args = match fn_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, _)) => {
            // Get the function signature from fn_sig()
            let sig_binder = fn_ty.kind().fn_sig().unwrap();
            // Skip the binder to get the actual signature
            let sig = sig_binder.skip_binder();
            let inputs = sig.inputs();
            inputs.len()
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(_, _)) => {
            // Closures use RustCall ABI where:
            // - MIR local 1 = self (closure environment, even if ZST)
            // - MIR locals 2..N = unpacked arguments from the fn_sig's tuple input
            //
            // fn_sig().inputs() returns just the tuple, NOT including self.
            // We need to count: 1 (self) + unpacked tuple elements
            let sig_binder = fn_ty.kind().fn_sig().unwrap();
            let sig = sig_binder.skip_binder();
            let inputs = sig.inputs();

            // The input should be a single tuple (RustCall convention)
            let tuple_arg_count = if inputs.len() == 1 {
                // Get the tuple type and count its elements
                if let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(
                    tuple_tys,
                )) = inputs[0].kind()
                {
                    tuple_tys.len()
                } else {
                    // Not a tuple - use 1 (single arg)
                    1
                }
            } else {
                // Multiple inputs (shouldn't happen with RustCall, but handle it)
                inputs.len()
            };

            // Total args = 1 (self) + unpacked tuple elements
            1 + tuple_arg_count
        }
        _ => {
            return input_err_noloc!(TranslationErr::unsupported(format!(
                "Expected FnDef or Closure type for function, got {:?}",
                fn_ty.kind()
            )));
        }
    };

    // Validate the source contract before translating argument types, so DSTs
    // and interior-mutability failures report the grid-constant declaration
    // rather than an incidental failure deeper in type lowering.
    let grid_constant_params =
        detect_grid_constant_params(body, &reachable, instance, is_kernel)
            .map_err(|error| input_error_noloc!(TranslationErr::invalid_op(error)))?;
    let mut grid_constant_layouts = Vec::with_capacity(grid_constant_params.len());
    for source_index in grid_constant_params {
        if source_index >= num_args {
            return input_err_noloc!(TranslationErr::invalid_op(format!(
                "grid-constant source parameter index {source_index} is out of range for {num_args} parameters"
            )));
        }
        let local = mir::Local::from(source_index + 1);
        let parameter_ty = body.locals()[local].ty;
        let TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) = parameter_ty.kind() else {
            return input_err_noloc!(TranslationErr::invalid_op(format!(
                "grid-constant source parameter {source_index} is not a reference"
            )));
        };
        if mutability != mir::Mutability::Not {
            return input_err_noloc!(TranslationErr::invalid_op(format!(
                "grid-constant source parameter {source_index} is mutable"
            )));
        }
        facts::validate_grid_constant_parameter(instance, source_index, pointee)
            .map_err(|error| input_error_noloc!(TranslationErr::invalid_op(error)))?;
        let layout = pointee.layout().map_err(|error| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "could not query grid-constant parameter {source_index} pointee layout: {error:?}"
            )))
        })?;
        let shape = layout.shape();
        if !shape.is_sized() {
            return input_err_noloc!(TranslationErr::invalid_op(format!(
                "grid-constant source parameter {source_index} requires a sized pointee"
            )));
        }
        if shape.size.bytes() == 0 {
            return input_err_noloc!(TranslationErr::invalid_op(format!(
                "grid-constant source parameter {source_index} has a zero-sized pointee"
            )));
        }
        grid_constant_layouts.push((source_index, pointee, shape.abi_align));
    }

    for arg_idx in 0..num_args {
        // MIR local index for arguments: local 1, 2, 3, ... (0 is return value)
        let local = mir::Local::from(arg_idx + 1);
        let local_decl = &body.locals()[local];
        let ty = &local_decl.ty;
        let arg_type = types::translate_type(ctx, ty)?;
        arg_types.push(arg_type);
    }

    // Get return type (local 0)
    let return_local = mir::Local::from(0usize);
    let return_decl = &body.locals()[return_local];
    let return_type_ptr = types::translate_type(ctx, &return_decl.ty)?;

    // Unit-tuple returns become a void `mir.func` signature. We skip the
    // result so `MirReturnOp` isn't expected to carry an unused `()` operand
    // (`mir-lower` reconstructs the unit value at LLVM lowering time).
    let is_unit_return = {
        let return_type_obj = return_type_ptr.deref(ctx);
        if let Some(tuple_ty) = return_type_obj.downcast_ref::<dialect_mir::types::MirTupleType>() {
            tuple_ty.get_types().is_empty()
        } else {
            false
        }
    };

    let return_types = if is_unit_return {
        vec![]
    } else {
        vec![return_type_ptr]
    };

    // Create function type for signature
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::types::FunctionType;
    let func_type = FunctionType::get(
        ctx,
        arg_types.clone(), // inputs
        return_types,      // results
    );
    let func_type_attr = TypeAttr::new(func_type.into());

    // Create a mir.func operation with a region for the function body
    let op_ptr = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![], // No result types
        vec![], // No operands
        vec![], // No successors
        1,      // 1 region for function body
    );

    // Set the function location from rustc's body span. This becomes the
    // default scope line for line-table debug info once LLVM export is enabled.
    let loc = span_to_location(ctx, body.span);
    op_ptr.deref_mut(ctx).set_loc(loc);

    // Create MirFuncOp and set the function type attribute and symbol name
    let mir_func_op = MirFuncOp::new(ctx, op_ptr, func_type_attr);

    let name_str = if let Some(name) = override_name {
        name.to_string()
    } else {
        instance.name().to_string()
    };
    mir_func_op.set_symbol_name(ctx, legaliser.legalise(&name_str));

    // Keep the Rust/source-facing name independent of the physical symbol.
    // Kernels deliberately retain their user-visible export name: their MIR
    // instance is the macro-generated device implementation, whose diagnostic
    // name is not the name callers launch or set breakpoints on.  Device
    // helpers use stable MIR's fully specialized diagnostic name, while their
    // physical symbol may be legalized (non-generic) or mangled (generic).
    let debug_name = function_debug_name(instance, is_kernel, &name_str);
    llvm_export::ops::set_debug_function_name(ctx, op_ptr, &debug_name);

    // Stamp only the currently audited Rust-ABI kernel reference facts.
    // `facts.rs` is the semantic oracle: this layer merely associates each
    // typed rustc_public proof with its source argument index. Foreign-ABI
    // kernels intentionally remain bare even when a parameter happens to have
    // a reference-shaped Rust type.
    let has_rust_abi = fn_ty.kind().fn_sig().is_some_and(|signature| {
        matches!(signature.skip_binder().abi, rustc_public::ty::Abi::Rust)
    });
    if is_kernel && has_rust_abi {
        for arg_idx in 0..num_args {
            let local = mir::Local::from(arg_idx + 1);
            let source_ty = &body.locals()[local].ty;
            let Some(validity) = facts::reference_param_validity(source_ty) else {
                continue;
            };
            mir_func_op.set_reference_param_validity(
                ctx,
                arg_idx,
                ReferenceParamValidityAttr(validity.pointee_alignment),
            );
        }
    }

    if is_kernel {
        // Add "gpu_kernel" attribute to the mir.func operation.
        // This will be used by the lowering pass to set the "gpu_kernel" attribute on the llvm.func.
        use pliron::builtin::attributes::StringAttr;
        let kernel_attr = StringAttr::new("true".to_string());
        let key: Identifier = "gpu_kernel".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, kernel_attr);

        for (source_index, pointee, alignment) in grid_constant_layouts {
            let pointee_ty = types::translate_type(ctx, &pointee)?;
            let type_key: Identifier =
                format!("{MIR_GRID_CONSTANT_POINTEE_ATTR_PREFIX}{source_index}")
                    .as_str()
                    .try_into()
                    .expect("grid-constant pointee attribute name is valid");
            let align_key: Identifier =
                format!("{MIR_GRID_CONSTANT_ALIGN_ATTR_PREFIX}{source_index}")
                    .as_str()
                    .try_into()
                    .expect("grid-constant alignment attribute name is valid");
            let align_ty = pliron::builtin::types::IntegerType::get(
                ctx,
                64,
                pliron::builtin::types::Signedness::Unsigned,
            );
            let align = pliron::utils::apint::APInt::from_u64(
                alignment,
                std::num::NonZero::new(64).expect("64 is non-zero"),
            );
            let mut operation = mir_func_op.get_operation().deref_mut(ctx);
            operation
                .attributes
                .set(type_key, TypeAttr::new(pointee_ty));
            operation.attributes.set(
                align_key,
                pliron::builtin::attributes::IntegerAttr::new(align_ty, align),
            );
        }

        // Detect compile-time cluster configuration from #[cluster(x,y,z)] attribute
        if let Some(cluster_dims) = detect_cluster_config(body, &reachable) {
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::Signedness;
            use pliron::utils::apint::APInt;
            use std::num::NonZero;

            // Add cluster_dim_x/y/z attributes
            // These will be used by the LLVM export to emit nvvm.annotations metadata
            let u32_ty = pliron::builtin::types::IntegerType::get(ctx, 32, Signedness::Unsigned);
            let width = NonZero::new(32).unwrap();

            // Create APInt values for each dimension
            let apint_x = APInt::from_u32(cluster_dims.x, width);
            let apint_y = APInt::from_u32(cluster_dims.y, width);
            let apint_z = APInt::from_u32(cluster_dims.z, width);

            let x_attr = IntegerAttr::new(u32_ty, apint_x);
            let y_attr = IntegerAttr::new(u32_ty, apint_y);
            let z_attr = IntegerAttr::new(u32_ty, apint_z);

            let x_key: Identifier = "cluster_dim_x".try_into().unwrap();
            let y_key: Identifier = "cluster_dim_y".try_into().unwrap();
            let z_key: Identifier = "cluster_dim_z".try_into().unwrap();

            let mut op_mut = mir_func_op.get_operation().deref_mut(ctx);
            op_mut.attributes.set(x_key, x_attr);
            op_mut.attributes.set(y_key, y_attr);
            op_mut.attributes.set(z_key, z_attr);

            if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
                eprintln!(
                    "  Cluster config detected: {}x{}x{}",
                    cluster_dims.x, cluster_dims.y, cluster_dims.z
                );
            }
        }

        // Detect compile-time launch bounds from #[launch_bounds(max, min)] attribute
        let launch_bounds = match detect_launch_bounds_config(body, &reachable) {
            Ok(bounds) => bounds,
            Err(error) => {
                return input_err_noloc!(TranslationErr::invalid_op(error));
            }
        };

        // Detect the exact block shape from #[launch_contract(block = (x,y,z))].
        // The exporter emits this as reqntid and suppresses maxntid, which ptxas
        // rejects alongside it.
        let contract_block = match detect_contract_block_config(body, &reachable) {
            Ok(block) => block,
            Err(error) => {
                return input_err_noloc!(TranslationErr::invalid_op(error));
            }
        };

        if let (Some(bounds), Some(block)) = (launch_bounds, contract_block)
            && let Err(error) = validate_block_against_bounds(bounds, block)
        {
            return input_err_noloc!(TranslationErr::invalid_op(error));
        }

        if let Some(launch_bounds) = launch_bounds {
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::Signedness;
            use pliron::utils::apint::APInt;
            use std::num::NonZero;

            // Add maxntid and minctasm attributes
            // These will be used by the LLVM export to emit nvvm.annotations metadata
            let u32_ty = pliron::builtin::types::IntegerType::get(ctx, 32, Signedness::Unsigned);
            let width = NonZero::new(32).unwrap();

            // Create APInt values
            let apint_max = APInt::from_u32(launch_bounds.max_threads, width);
            let max_attr = IntegerAttr::new(u32_ty, apint_max);
            let max_key: Identifier = "maxntid".try_into().unwrap();

            let mut op_mut = mir_func_op.get_operation().deref_mut(ctx);
            op_mut.attributes.set(max_key, max_attr);

            // Only add minctasm if it's non-zero (specified)
            if launch_bounds.min_blocks > 0 {
                let apint_min = APInt::from_u32(launch_bounds.min_blocks, width);
                let min_attr = IntegerAttr::new(u32_ty, apint_min);
                let min_key: Identifier = "minctasm".try_into().unwrap();
                op_mut.attributes.set(min_key, min_attr);
            }

            if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
                if launch_bounds.min_blocks > 0 {
                    eprintln!(
                        "  Launch bounds detected: maxntid={}, minctasm={}",
                        launch_bounds.max_threads, launch_bounds.min_blocks
                    );
                } else {
                    eprintln!(
                        "  Launch bounds detected: maxntid={}",
                        launch_bounds.max_threads
                    );
                }
            }
        }

        if let Some(contract_block) = contract_block {
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::Signedness;
            use pliron::utils::apint::APInt;
            use std::num::NonZero;

            let u32_ty = pliron::builtin::types::IntegerType::get(ctx, 32, Signedness::Unsigned);
            let width = NonZero::new(32).unwrap();

            let mut op_mut = mir_func_op.get_operation().deref_mut(ctx);
            for (key, value) in [
                ("reqntid_x", contract_block.x),
                ("reqntid_y", contract_block.y),
                ("reqntid_z", contract_block.z),
            ] {
                let attr = IntegerAttr::new(u32_ty, APInt::from_u32(value, width));
                let key: Identifier = key.try_into().unwrap();
                op_mut.attributes.set(key, attr);
            }

            if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
                eprintln!(
                    "  Launch contract block detected: reqntid={}x{}x{}",
                    contract_block.x, contract_block.y, contract_block.z
                );
            }
        }
    }

    // Attribute macros may run before `#[kernel]`. Generic expansion forwards
    // that marker to the entry but also keeps the original in its helper, so
    // record markers on any function. mir-lower treats every marked local
    // function as a propagation root and carries the minimum to its callees.
    if let Some(alignment) = detect_dynamic_shared_alignment(body, &reachable) {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::builtin::types::Signedness;
        use pliron::utils::apint::APInt;
        use std::num::NonZero;

        let u64_ty = pliron::builtin::types::IntegerType::get(ctx, 64, Signedness::Unsigned);
        let value = APInt::from_u64(alignment.bytes, NonZero::new(64).unwrap());
        let key: Identifier = "dynamic_shared_alignment".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, IntegerAttr::new(u64_ty, value));

        if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
            eprintln!(
                "  Dynamic shared-memory contract alignment detected: {}",
                alignment.bytes
            );
        }
    }

    if let Some(scope_map) = debug_source_scopes
        && debug_kind.variables_enabled()
    {
        llvm_export::ops::set_debug_source_scope_map(ctx, op_ptr, scope_map);
    }

    set_alwaysinline_attr_from_flag(ctx, &mir_func_op, is_kernel, is_inline_always);

    // Get the function body region (region 0)
    let region_ptr = op_ptr.deref(ctx).get_region(0);

    // -------------------------------------------------------------------------
    // PHASE 1: Create all pliron IR blocks
    // -------------------------------------------------------------------------
    //
    // Only the entry block receives block arguments (the function's formal
    // parameters). Every other block is argument-less: cross-block data flow
    // travels through the per-local alloca slots, not block arguments.
    let mut block_map: Vec<Ptr<BasicBlock>> = Vec::new();

    for (idx, _mir_block) in body.blocks.iter().enumerate() {
        let arg_types_for_block = if idx == 0 { arg_types.clone() } else { vec![] };

        let block_ptr = BasicBlock::new(ctx, None, arg_types_for_block);
        block_map.push(block_ptr);
    }

    // Link all blocks into the function's region.
    for (idx, block_ptr) in block_map.iter().enumerate() {
        if idx == 0 {
            block_ptr.insert_at_front(region_ptr, ctx);
        } else {
            block_ptr.insert_after(ctx, block_map[idx - 1]);
        }
    }

    // -------------------------------------------------------------------------
    // PHASE 1.5: Entry-block allocas + argument stores
    // -------------------------------------------------------------------------
    //
    // Every non-ZST MIR local is backed by a single stack slot emitted at the
    // top of the entry block; its pointer is recorded in `value_map` via
    // `set_slot`. Function arguments are eagerly stored into their slots so
    // later blocks can `load_local` them without needing block arguments.
    //
    // The `mem2reg` pass in `pipeline.rs` promotes the scalar slots back into
    // SSA before LLVM lowering.
    let entry_last_op = emit_entry_allocas(
        ctx,
        body,
        block_map[0],
        num_args,
        &mut value_map,
        debug_kind,
        debug_source_scopes,
        &reachable,
    );

    if debug_kind.variables_enabled()
        && let Some(statement_debug_info) = statement_debug_info
    {
        super::statement_debug::prepare_statement_debug_spills(
            ctx,
            body,
            statement_debug_info,
            &reachable,
            rustc_mono_successors,
            &mut value_map,
        );
    }

    // -------------------------------------------------------------------------
    // PHASE 2: Translate reachable blocks
    // -------------------------------------------------------------------------
    //
    // Every local flows through its stack slot, so blocks have no inter-block
    // ordering dependency and can be translated in a single index-order pass.
    // Unwind-only cleanup blocks are skipped here (see
    // [`non_unwind_successors`]) and patched with `mir.unreachable` below.
    let mut blocks_processed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for idx in reachable.iter().copied() {
        let mir_block = &body.blocks[idx];
        let block_ptr = block_map[idx];
        let entry_prev_op = if idx == 0 { entry_last_op } else { None };
        let block_debug_info = debug_kind
            .variables_enabled()
            .then(|| statement_debug_info.map(|map| &map.blocks[idx]))
            .flatten();
        block::translate_block(
            ctx,
            body,
            mir_block,
            idx,
            block_ptr,
            &mut value_map,
            &block_map,
            &rustc_mono_successors[idx],
            legaliser,
            entry_prev_op,
            block_debug_info,
        )?;
        blocks_processed.insert(idx);
    }

    // Unwind cleanup blocks are unreachable on GPU but pliron still requires
    // every block to have a terminator, so we stitch `mir.unreachable` onto
    // the ones we skipped above. Later passes are free to drop them as dead
    // code.
    for (idx, &block_ptr) in block_map.iter().enumerate().take(body.blocks.len()) {
        if !blocks_processed.contains(&idx) {
            let unreachable_op = Operation::new(
                ctx,
                dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            unreachable_op.insert_at_front(block_ptr, ctx);
        }
    }

    Ok(op_ptr)
}

/// Choose the debugger-visible spelling independently of a function's symbol.
///
/// Device-helper names intentionally keep concrete generic arguments. Erasing
/// them would make distinct monomorphizations indistinguishable. Supporting a
/// shorthand such as `Type::method` for every specialization requires proper
/// namespace/type DIEs (or a separately agreed alias policy), not lossy names.
fn function_debug_name(instance: &mono::Instance, is_kernel: bool, export_name: &str) -> String {
    if is_kernel {
        export_name.to_string()
    } else {
        instance.name().to_string()
    }
}

/// Propagate `#[inline(always)]` as an LLVM `alwaysinline` function
/// attribute. Kernel entry points are excluded because they're `.entry` in PTX
/// and never callees, so marking them `alwaysinline` would be a no-op at best
/// and rejected by LLVM at worst.
fn set_alwaysinline_attr_from_flag(
    ctx: &mut Context,
    mir_func_op: &MirFuncOp,
    is_kernel: bool,
    is_inline_always: bool,
) {
    if is_inline_always && !is_kernel {
        let attr = pliron::builtin::attributes::StringAttr::new("true".to_string());
        let key: Identifier = "alwaysinline".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, attr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::TypeAttr, op_interfaces::SymbolOpInterface, ops::ModuleOp,
            types::FunctionType,
        },
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
    };

    #[test]
    fn collector_reachability_requires_the_same_public_mir_cfg() {
        let valid = [vec![2], vec![], vec![3], vec![]];
        assert!(validate_monomorphized_successor_shape(4, 4, &valid).is_ok());
        assert!(validate_monomorphized_successor_shape(4, 5, &valid).is_err());
        assert!(validate_monomorphized_successor_shape(4, 4, &valid[..3]).is_err());
        assert!(
            validate_monomorphized_successor_shape(4, 4, &[vec![4], vec![], vec![], vec![]])
                .is_err()
        );
    }

    #[test]
    fn an_exact_block_may_not_need_more_threads_than_launch_bounds_allows() {
        let block = |x, y, z| ContractBlock { x, y, z };
        let bounds = |max_threads| LaunchBounds {
            max_threads,
            min_blocks: 0,
        };

        // The shapes the `cuda_module_contract` example declares: the maximum
        // equals the required thread count on every axis product.
        assert!(validate_block_against_bounds(bounds(256), block(256, 1, 1)).is_ok());
        assert!(validate_block_against_bounds(bounds(64), block(8, 8, 1)).is_ok());
        // A maximum above the requirement is redundant, and allowed.
        assert!(validate_block_against_bounds(bounds(1024), block(16, 16, 1)).is_ok());

        // A maximum below the requirement contradicts it. Without this the
        // exporter would drop the maximum and emit the larger shape.
        let error = validate_block_against_bounds(bounds(128), block(16, 16, 1))
            .expect_err("256 threads exceed a 128-thread maximum");
        assert!(error.contains("256 threads per block"), "{error}");
        assert!(error.contains("at most 128"), "{error}");
        assert!(validate_block_against_bounds(bounds(255), block(256, 1, 1)).is_err());

        // The product is computed in u64, so a shape whose axes multiply past
        // u32 is rejected rather than wrapping to a value under the maximum.
        assert!(validate_block_against_bounds(bounds(1024), block(65_536, 65_536, 1)).is_err());
    }

    #[test]
    fn stable_mir_function_names_are_source_facing_and_specialized() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_function_debug_name_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("function_debug_name_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
#[inline(never)]
pub fn plain(value: u32) -> u32 { value + 1 }

#[inline(never)]
pub fn generic<T>(value: T) -> T { value }

pub struct Wrapper<T>(pub T);

impl<T> Wrapper<T> {
    #[inline(never)]
    pub fn get_mut(&mut self) -> &mut T { &mut self.0 }
}

#[inline(never)]
pub fn cuda_oxide_device_generated_kernel(mut wrapped: Wrapper<u16>) -> u32 {
    let _ = generic::<u64>(3);
    let _ = wrapped.get_mut();
    plain(7)
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
            "--crate-name=function_debug_name_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Zmir-opt-level=0".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let names = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let item = rustc_public::all_local_items()
                        .into_iter()
                        .find(|item| {
                            item.name()
                                .ends_with("::cuda_oxide_device_generated_kernel")
                        })
                        .expect("fixture entry item");
                    let instance = mono::Instance::try_from(item).expect("entry instance");
                    let body = instance.body().expect("entry body");
                    let mut callees = Vec::new();

                    for block in &body.blocks {
                        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
                            continue;
                        };
                        let mir::Operand::Constant(constant) = func else {
                            continue;
                        };
                        let ConstantKind::ZeroSized = constant.const_.kind() else {
                            continue;
                        };
                        let TyKind::RigidTy(RigidTy::FnDef(definition, args)) =
                            constant.const_.ty().kind()
                        else {
                            continue;
                        };
                        let Some(callee) = mono::Instance::resolve(definition, &args).ok() else {
                            continue;
                        };
                        callees.push((
                            callee.name().to_string(),
                            callee.def.name().to_string(),
                            callee.mangled_name().to_string(),
                        ));
                    }

                    std::ops::ControlFlow::<(), _>::Continue((
                        instance.name().to_string(),
                        function_debug_name(&instance, true, "visible_kernel"),
                        callees,
                    ))
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            names.0,
            "function_debug_name_fixture::cuda_oxide_device_generated_kernel"
        );
        assert_eq!(names.1, "visible_kernel");
        let source_names: std::collections::BTreeSet<_> = names
            .2
            .iter()
            .map(|(source, _, _)| source.as_str())
            .collect();
        assert_eq!(
            source_names,
            std::collections::BTreeSet::from([
                "function_debug_name_fixture::Wrapper::<u16>::get_mut",
                "function_debug_name_fixture::generic::<u64>",
                "function_debug_name_fixture::plain",
            ])
        );
        let definition_names: std::collections::BTreeSet<_> = names
            .2
            .iter()
            .map(|(_, definition, _)| definition.as_str())
            .collect();
        assert_eq!(
            definition_names,
            std::collections::BTreeSet::from([
                "function_debug_name_fixture::Wrapper::<T>::get_mut",
                "function_debug_name_fixture::generic",
                "function_debug_name_fixture::plain",
            ]),
            "definition names are not specialized enough for debugger overloads"
        );
        for (source_name, _, mangled_name) in names.2 {
            assert_ne!(source_name, mangled_name);
            assert!(
                mangled_name.starts_with("_R"),
                "expected a Rust linkage symbol, got {mangled_name}"
            );
        }
    }

    #[test]
    fn inline_always_flag_reaches_llvm_func_attr_before_export() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
        let module_op = module.get_operation();
        let module_region = module_op.deref(&ctx).get_region(0);
        let module_block = {
            let existing = {
                let region = module_region.deref(&ctx);
                region.iter(&ctx).next()
            };
            if let Some(block) = existing {
                block
            } else {
                let block = BasicBlock::new(&mut ctx, None, vec![]);
                block.insert_at_back(module_region, &ctx);
                block
            }
        };

        let func_type = FunctionType::get(&ctx, vec![], vec![]);
        let func_type_attr = TypeAttr::new(func_type.into());
        let mir_func = {
            let op = Operation::new(
                &mut ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let func = MirFuncOp::new(&mut ctx, op, func_type_attr);
            func.set_symbol_name(&mut ctx, "inline_helper".try_into().unwrap());
            func
        };

        set_alwaysinline_attr_from_flag(&mut ctx, &mir_func, false, true);
        llvm_export::ops::set_debug_function_name(
            &mut ctx,
            mir_func.get_operation(),
            "source_crate::inline_helper",
        );
        mir_func.get_operation().insert_at_back(module_block, &ctx);

        mir_lower::register(&mut ctx);
        mir_lower::lower_mir_to_llvm(&mut ctx, module_op).expect("lowering succeeds");

        let llvm_func = {
            let block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
            block
                .deref(&ctx)
                .iter(&ctx)
                .find_map(|op| Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx))
                .expect("lowered LLVM function")
        };

        let key: Identifier = "alwaysinline".try_into().unwrap();
        assert!(
            llvm_func
                .get_operation()
                .deref(&ctx)
                .attributes
                .0
                .contains_key(&key),
            "`is_inline_always` must become an LLVM dialect alwaysinline attribute before export",
        );
        assert_eq!(
            llvm_export::ops::debug_function_name(&ctx, llvm_func.get_operation()).as_deref(),
            Some("source_crate::inline_helper"),
            "MIR-to-LLVM lowering must preserve the source-facing function name",
        );
    }

    /// Exercise `debug_fragment` against composite `VarDebugInfo` produced by
    /// rustc's scalar-replacement pass instead of constructing synthetic types.
    ///
    /// The fixture deliberately creates an aggregate local and enables MIR
    /// optimization so SROA rewrites its debug binding into field fragments.
    /// The test then checks that every supported fragment matches rustc's real
    /// layout and that a non-field projection fails closed.
    #[test]
    fn scalar_replacement_debug_fragments_follow_rustc_layout_and_fail_closed() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_fragment_debug_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("fragment_debug_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
pub fn scalarized_pair(a: u32, b: u64) -> u64 {
    let pair = (a, b);
    pair.1.wrapping_add(pair.0 as u64)
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
            "--crate-name=fragment_debug_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Cdebuginfo=2".to_string(),
            "-Zmir-opt-level=3".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let result = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let mut checked = 0usize;
                    let mut rejected_non_field = false;

                    for body in rustc_public::all_local_items()
                        .into_iter()
                        .filter_map(|item| item.body())
                    {
                        for info in &body.var_debug_info {
                            if info.name != "pair" {
                                continue;
                            }
                            let Some(composite) = &info.composite else {
                                continue;
                            };
                            let [mir::ProjectionElem::Field(field_idx, field_ty)] =
                                composite.projection.as_slice()
                            else {
                                continue;
                            };

                            let fragment = debug_fragment(composite)
                                .expect("SROA field fragment should be supported");
                            let whole_layout = composite.ty.layout().expect("whole layout").shape();
                            let rustc_public::abi::FieldsShape::Arbitrary { offsets } =
                                &whole_layout.fields
                            else {
                                panic!("tuple fragment must use arbitrary field offsets");
                            };
                            let expected_offset_bits = offsets[*field_idx].bytes() as u64 * 8;
                            let expected_size_bits = field_ty
                                .layout()
                                .expect("field layout")
                                .shape()
                                .size
                                .bytes()
                                as u64
                                * 8;

                            assert_eq!(fragment.offset_bits, expected_offset_bits);
                            assert_eq!(fragment.size_bits, expected_size_bits);
                            checked += 1;

                            let mut invalid = composite.clone();
                            invalid.projection[0] = mir::ProjectionElem::Deref;
                            rejected_non_field |= debug_fragment(&invalid).is_none();
                        }
                    }

                    std::ops::ControlFlow::<(), _>::Continue((checked, rejected_non_field))
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        assert!(
            result.0 >= 2,
            "fixture should produce at least two SROA debug fragments, got {}",
            result.0
        );
        assert!(
            result.1,
            "debug_fragment must reject non-field composite projections"
        );
    }

    /// Reference propagation may map several source bindings to one optimized
    /// MIR local. Keep every whole-variable identity instead of silently
    /// retaining only the first one in the map.
    #[test]
    fn reference_propagation_retains_every_whole_variable_identity() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_whole_debug_alias_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("whole_debug_alias_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
#[inline(never)]
pub fn probe(x: &i32) -> i32 {
    let first: *const i32 = x as *const i32;
    let second: *const i32 = x as *const i32;
    unsafe { *first + *second }
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
            "--crate-name=whole_debug_alias_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Cdebuginfo=2".to_string(),
            "-Copt-level=3".to_string(),
            "-Zmir-enable-passes=-ScalarReplacementOfAggregates,-SingleUseConsts".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let identities = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let body = rustc_public::all_local_items()
                        .into_iter()
                        .find(|item| item.name().ends_with("::probe"))
                        .and_then(|item| item.body())
                        .expect("probe body");
                    let source_locals: Vec<_> = body
                        .var_debug_info
                        .iter()
                        .filter(|info| matches!(info.name.as_str(), "x" | "first" | "second"))
                        .map(|info| match &info.value {
                            mir::VarDebugInfoContents::Place(place)
                                if place.projection.is_empty() =>
                            {
                                place.local
                            }
                            other => panic!("whole binding expected, got {other:?}"),
                        })
                        .collect();
                    assert_eq!(
                        source_locals.len(),
                        3,
                        "fixture has three source identities"
                    );
                    assert!(
                        source_locals.iter().all(|local| *local == source_locals[0]),
                        "ReferencePropagation should coalesce x/first/second onto one MIR local"
                    );

                    let mut ctx = Context::new();
                    crate::translator::register_dialects(&mut ctx);
                    let collected = collect_debug_locals(&mut ctx, &body);
                    let infos = collected
                        .whole
                        .get(&source_locals[0])
                        .expect("coalesced local retains whole debug identities");
                    std::ops::ControlFlow::<(), _>::Continue(
                        infos
                            .iter()
                            .map(|info| {
                                (
                                    info.variable.name.clone(),
                                    info.variable.argument_index,
                                    info.variable.ty.clone(),
                                    info.source_scope,
                                    info.declaration.clone(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            identities
                .iter()
                .map(|identity| identity.0.as_str())
                .collect::<Vec<_>>(),
            ["x", "first", "second"],
            "source order determines the primary identity and two explicit aliases"
        );
        assert_eq!(identities[0].1, Some(1));
        assert_eq!(identities[1].1, None);
        assert_eq!(identities[2].1, None);
        assert!(identities.iter().all(|identity| identity.4.is_some()));
        assert!(
            identities.iter().all(|identity| {
                matches!(
                    &identity.2,
                    DebugLocalTypeKind::TypedPointer { name, .. } if name == "&i32"
                )
            }),
            "optimized debug types must match rustc's surviving `&i32` place"
        );
        assert_ne!(
            identities[0].3, identities[1].3,
            "each identity retains its own lexical scope"
        );
    }

    /// Thin pointers preserve a finite pointee debug tree, while composites,
    /// fat pointers, and unsupported pointees retain the opaque fallback.
    #[test]
    fn pointer_debug_types_type_safe_pointees_and_preserve_opaque_fallbacks() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_pointer_debug_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("pointer_debug_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
pub struct Pair {
    pub first: u32,
    pub second: u64,
}

pub struct Node {
    pub next: *const Node,
    pub value: u32,
}

pub fn pointer_debug_types(
    _thin: *mut u32,
    _reference: &u64,
    _nested: *const *mut i32,
    _array: *const [u16; 4],
    _composite: *const Pair,
    _tuple: *const (u32, u64),
    _fat: *const [u8],
    _fat_reference: &str,
    _unsupported: *const fn(),
    _unit: *const (),
    _char: *const char,
    _self_referential: *const Node,
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
            "--crate-name=pointer_debug_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Zmir-opt-level=0".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let debug_types = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let body = rustc_public::all_local_items()
                        .into_iter()
                        .find_map(|item| item.body())
                        .expect("fixture function has a body");
                    let debug_types = body
                        .locals()
                        .iter()
                        .skip(1) // return place
                        .take(12)
                        .map(|decl| debug_type_for_ty(&decl.ty))
                        .collect::<Vec<_>>();
                    std::ops::ControlFlow::<(), _>::Continue(debug_types)
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();
        assert_eq!(debug_types.len(), 12, "one debug type result per argument");

        let Some(DebugLocalTypeKind::TypedPointer {
            name,
            size_bits,
            pointee,
        }) = &debug_types[0]
        else {
            panic!(
                "thin raw pointer must be described, got {:?}",
                debug_types[0]
            );
        };
        assert_eq!(name, "*mut u32");
        assert_eq!(*size_bits, 64);
        assert!(matches!(
            pointee.as_ref(),
            DebugLocalTypeKind::Basic {
                name,
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            } if name == "u32"
        ));

        let Some(DebugLocalTypeKind::TypedPointer {
            name,
            size_bits,
            pointee,
        }) = &debug_types[1]
        else {
            panic!("thin reference must be described, got {:?}", debug_types[1]);
        };
        assert_eq!(name, "&u64");
        assert_eq!(*size_bits, 64);
        assert!(matches!(
            pointee.as_ref(),
            DebugLocalTypeKind::Basic {
                name,
                size_bits: 64,
                encoding: "DW_ATE_unsigned",
            } if name == "u64"
        ));

        let Some(DebugLocalTypeKind::TypedPointer {
            name,
            size_bits,
            pointee,
        }) = &debug_types[2]
        else {
            panic!(
                "nested thin pointer must be described, got {:?}",
                debug_types[2]
            );
        };
        assert_eq!(name, "*const *mut i32");
        assert_eq!(*size_bits, 64);
        assert!(matches!(
            pointee.as_ref(),
            DebugLocalTypeKind::TypedPointer {
                name,
                size_bits: 64,
                pointee,
            } if name == "*mut i32" && matches!(
                pointee.as_ref(),
                DebugLocalTypeKind::Basic {
                    name,
                    size_bits: 32,
                    encoding: "DW_ATE_signed",
                } if name == "i32"
            )
        ));

        let Some(DebugLocalTypeKind::TypedPointer {
            name,
            size_bits,
            pointee,
        }) = &debug_types[3]
        else {
            panic!(
                "pointer to fixed array must be described, got {:?}",
                debug_types[3]
            );
        };
        assert_eq!(name, "*const [u16; 4]");
        assert_eq!(*size_bits, 64);
        assert!(matches!(
            pointee.as_ref(),
            DebugLocalTypeKind::Array {
                name,
                size_bits: 64,
                count: 4,
                element,
            } if name == "[u16; 4]" && matches!(
                element.as_ref(),
                DebugLocalTypeKind::Basic {
                    name,
                    size_bits: 16,
                    encoding: "DW_ATE_unsigned",
                } if name == "u16"
            )
        ));

        fn assert_opaque_pointer(ty: &Option<DebugLocalTypeKind>, expected_name: &str) {
            let Some(DebugLocalTypeKind::Pointer { name, size_bits }) = ty else {
                panic!("unsupported pointer must retain opaque metadata, got {ty:?}");
            };
            assert_eq!(name, expected_name);
            assert_eq!(*size_bits, 64, "compatibility fallback preserves old width");
        }

        assert_opaque_pointer(&debug_types[4], "*const _");
        assert_opaque_pointer(&debug_types[5], "*const _");
        assert_opaque_pointer(&debug_types[6], "*const _");
        assert_opaque_pointer(&debug_types[7], "&_");
        assert_opaque_pointer(&debug_types[8], "*const _");
        assert_opaque_pointer(&debug_types[9], "*const _");
        assert_opaque_pointer(&debug_types[10], "*const _");
        assert_opaque_pointer(&debug_types[11], "*const _");
    }

    /// Closure environments must be described as composite debug types with
    /// member offsets taken from rustc's real layout, not declaration order.
    ///
    /// `debug_type_for_ty` needs a live compiler session (closure types and
    /// layouts only exist inside one), so this test drives the pinned rustc
    /// in-process on a small fixture via `rustc_public::run!`, extracts the
    /// closure-typed local, and asserts on the returned plain data outside
    /// the session. The fixture is compiled with the same MIR flags
    /// cargo-oxide uses for full device debug (`-Copt-level=3` with
    /// ScalarReplacementOfAggregates and SingleUseConsts disabled), so the
    /// closure local survives to MIR as in a real full-debug build.
    ///
    /// The `u32`-before-`u64` capture order is deliberate: rustc's layout
    /// sorts closure fields by descending alignment, placing the `u64` at
    /// offset 0 and the `u32` at offset 8. Sequential declaration-order
    /// offsets would put `capture_0` at 0, so this fails loudly if the
    /// composite type ever stops using the layout's field offsets.
    #[test]
    fn closure_environment_debug_type_uses_layout_offsets() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_closure_debug_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("closure_debug_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
pub fn closure_host(a: u32, b: u64) -> u32 {
    let add = move |x: u32| x + a + (b as u32);
    add(1)
}
"#,
        )
        .unwrap();

        // The rustup shim resolves the same pinned toolchain this test binary
        // was built with, so the in-process driver and the sysroot agree.
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
            "--crate-name=closure_debug_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Copt-level=3".to_string(),
            "-Zmir-enable-passes=-JumpThreading".to_string(),
            "-Zmir-enable-passes=-ScalarReplacementOfAggregates,-SingleUseConsts".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        // rustc needs more stack than the default test-thread allowance.
        let debug_type = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let closure_ty = rustc_public::all_local_items()
                        .into_iter()
                        .filter_map(|item| item.body())
                        .flat_map(|body| body.locals().to_vec())
                        .map(|decl| decl.ty)
                        .find(|ty| matches!(ty.kind(), TyKind::RigidTy(RigidTy::Closure(..))))
                        .expect("fixture must contain a closure-typed local");
                    std::ops::ControlFlow::<(), _>::Continue(debug_type_for_ty(&closure_ty))
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        let Some(DebugLocalTypeKind::Struct {
            size_bits, members, ..
        }) = debug_type
        else {
            panic!("closure environment must produce a composite debug type, got {debug_type:?}");
        };
        assert_eq!(size_bits, 128, "u64 + u32 environment is 16 bytes");
        assert_eq!(members.len(), 2, "one member per capture");

        assert_eq!(members[0].name, "capture_0");
        assert_eq!(
            members[0].offset_bits, 64,
            "the u32 capture sits after the u64 in rustc's layout"
        );
        match &members[0].ty {
            DebugLocalTypeKind::Basic {
                name, size_bits, ..
            } => {
                assert_eq!(name, "u32");
                assert_eq!(*size_bits, 32);
            }
            other => panic!("capture_0 must be a basic u32, got {other:?}"),
        }

        assert_eq!(members[1].name, "capture_1");
        assert_eq!(
            members[1].offset_bits, 0,
            "the u64 capture is layout-first despite being declared second"
        );
        match &members[1].ty {
            DebugLocalTypeKind::Basic {
                name, size_bits, ..
            } => {
                assert_eq!(name, "u64");
                assert_eq!(*size_bits, 64);
            }
            other => panic!("capture_1 must be a basic u64, got {other:?}"),
        }
    }
}
