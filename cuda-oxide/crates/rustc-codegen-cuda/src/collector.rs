/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # Device Function Collector
//!
//! This module identifies all functions that must be compiled for the GPU, starting
//! from kernel entry points and transitively collecting all reachable callees.
//!
//! ## How It Works
//!
//! The collector performs a breadth-first traversal of the MIR call graph:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                         DEVICE FUNCTION COLLECTION                              │
//! │                                                                                 │
//! │   Input: Codegen Units (CGUs) from rustc                                        │
//! │          Each CGU contains monomorphized function instances                     │
//! │                                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 1: Find Kernel Entry Points                                       │   │
//! │   │                                                                         │   │
//! │   │  Scan all CGUs for functions whose names contain the reserved           │   │
//! │   │  KERNEL_PREFIX from `reserved-oxide-symbols` (the #[kernel] macro       │   │
//! │   │  renames `fn foo` into the hash-suffixed `cuda_oxide_*` namespace).     │   │
//! │   │                                                                         │   │
//! │   │  Example:                                                               │   │
//! │   │    #[kernel]                                                            │   │
//! │   │    fn add_one(data: *mut i32, len: usize) { ... }                       │   │
//! │   │                                                                         │   │
//! │   │    Becomes: cuda_oxide_kernel_<hash>_add_one in MIR                     │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 2: Walk Call Graph (Worklist Algorithm)                           │   │
//! │   │                                                                         │   │
//! │   │  worklist = [kernel1, kernel2, ...]                                     │   │
//! │   │  seen = {}                                                              │   │
//! │   │  result = []                                                            │   │
//! │   │                                                                         │   │
//! │   │  while worklist not empty:                                              │   │
//! │   │      func = worklist.pop()                                              │   │
//! │   │      mir = tcx.instance_mir(func)  ◄─── Gets OPTIMIZED MIR              │   │
//! │   │                                                                         │   │
//! │   │      for terminator in mir.basic_blocks:                                │   │
//! │   │          if terminator is Call:                                         │   │
//! │   │              callee = resolve_callee(terminator)                        │   │
//! │   │              if should_collect(callee) and callee not in seen:          │   │
//! │   │                  worklist.push(callee)                                  │   │
//! │   │                  seen.insert(callee)                                    │   │
//! │   │                                                                         │   │
//! │   │      result.push(func)                                                  │   │
//! │   │                                                                         │   │
//! │   │  return result                                                          │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  Output: Vec<CollectedFunction>                                         │   │
//! │   │                                                                         │   │
//! │   │  Each contains:                                                         │   │
//! │   │    - instance: The monomorphized Instance<'tcx>                         │   │
//! │   │    - is_kernel: true for entry points, false for callees                │   │
//! │   │    - export_name: Name to use in PTX                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Which Functions Are Collected
//!
//! We collect functions from these crates:
//!
//! | Crate                    | What's Collected                               | What's Filtered                          |
//! |--------------------------|------------------------------------------------|------------------------------------------|
//! | Local crate              | Everything reachable from kernels              | —                                        |
//! | External crates          | Kernels (`cuda_oxide_kernel_<hash>_*`)         | —                                        |
//! | `cuda_device`            | Non-intrinsic functions                        | Intrinsic stubs (just `unreachable!()`)  |
//! | `core`                   | Iterators, Option, etc.                        | `fmt::*`, `panicking::*`                 |
//! | `alloc`                  | Vec, Box, String (if GPU allocator configured) | —                                        |
//! | Other `no_std` crates    | All reachable functions                        | —                                        |
//!
//! ## Cross-Crate Kernel Support
//!
//! Library crates can export generic kernels that get monomorphized when used:
//!
//! ```rust,ignore
//! // my_cuda_lib/src/lib.rs
//! #[kernel]
//! pub fn reduce<T: Add>(data: &[T], out: &mut T) { ... }
//!
//! // my_app/src/main.rs
//! use my_cuda_lib::reduce;
//! unsafe { cuda_launch! { kernel: reduce::<f32>, ... } }  // PTX generated here!
//! ```
//!
//! Functions from `std` are FORBIDDEN because they require OS/threads/IO. The
//! one exception is std's inherent float wrappers (`std::f32::<impl f32>::atan`
//! and friends), which are collected; see `is_std_float_inherent_method`.
//!
//! ## MIR Access
//!
//! We access MIR via `tcx.instance_mir(instance.def)`, which returns **optimized MIR**.
//! This is the same MIR that would go to LLVM for native compilation. The optimization
//! level depends on the `-C opt-level` flag passed to rustc.
//!
//! ## Export Names and FQDN Alignment
//!
//! Export names must match what the MIR translator (`extract_func_info` in
//! `terminator/mod.rs`) produces for call targets. Both sides choose the same
//! raw name, then feed it through pliron's shared identifier legaliser.
//!
//! The collector uses [`DeviceCollector::fqdn()`] to produce FQDNs matching
//! `CrateDef::name()`/`Instance::name()` on the `rustc_public` side, including
//! concrete impl arguments from rustc's resolved `Instance`.
//!
//! For resolved instances that still carry generic args, or for raw-name
//! collisions, we use rustc's mangled symbol name (e.g., `_RNvMNtNtCs...`).
//! Invalid FQDN characters such as `<`, `>`, and `::` are not a separate naming
//! policy: they are handled by the same legaliser on the definition and call
//! sides.
//!
//! Kernel export names are separate — they use `compute_kernel_export_name`
//! with human-readable base names derived from the `#[kernel]` macro.

use mir_importer::is_panic_entry_path;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_index::{Idx, bit_set::DenseBitSet};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    BasicBlock, ConstOperand, ConstValue, Location, START_BLOCK, TerminatorKind,
};
use rustc_middle::mono::{CodegenUnit, MonoItem};
use rustc_middle::ty::{
    Instance, InstanceKind, ShimKind, Ty, TyCtxt, TyKind, TypeVisitableExt, TypingEnv,
};
use rustc_span::Span;
use std::collections::{HashMap, HashSet, VecDeque};

/// Blocks reachable under the values CUDA Oxide emits for device-only runtime
/// checks.
///
/// rustc's `mono_reachable_as_bitset` is the right semantic source for
/// monomorphized constants, but it evaluates `Operand::RuntimeChecks` from the
/// host compilation session. The MIR importer deliberately emits every such
/// query as `false` on device. Override only that operand shape here so call
/// collection and MIR import cannot choose different switch arms.
fn device_mono_reachable_as_bitset<'tcx>(
    body: &rustc_middle::mir::Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> DenseBitSet<BasicBlock> {
    let mut reachable = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut worklist = vec![START_BLOCK];

    while let Some(bb) = worklist.pop() {
        if !reachable.insert(bb) {
            continue;
        }

        worklist.extend(device_mono_successors(
            &body.basic_blocks[bb],
            tcx,
            instance,
        ));
    }

    reachable
}

#[derive(Debug, Clone)]
pub(crate) struct DeviceMonoReachability {
    pub(crate) block_count: usize,
    pub(crate) successors: Vec<Vec<usize>>,
}

fn device_mono_successors<'tcx>(
    block: &rustc_middle::mir::BasicBlockData<'tcx>,
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Vec<BasicBlock> {
    if let TerminatorKind::SwitchInt {
        discr: rustc_middle::mir::Operand::RuntimeChecks(_),
        targets,
    } = &block.terminator().kind
    {
        vec![device_runtime_checks_target(targets)]
    } else {
        block.mono_successors(tcx, instance).collect()
    }
}

/// Compute the exact per-block successor edges that collection used, for the
/// MIR importer. `rustc_public_bridge::BodyBuilder` monomorphizes the same body
/// in place, so block indices must remain stable; the importer also receives
/// and verifies `block_count` and every edge before trusting them.
pub(crate) fn device_mono_reachability<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> DeviceMonoReachability {
    let body = tcx.instance_mir(instance.def);
    DeviceMonoReachability {
        block_count: body.basic_blocks.len(),
        successors: body
            .basic_blocks
            .iter()
            .map(|block| {
                device_mono_successors(block, tcx, instance)
                    .into_iter()
                    .map(Idx::index)
                    .collect()
            })
            .collect(),
    }
}

fn device_runtime_checks_target(targets: &rustc_middle::mir::SwitchTargets) -> BasicBlock {
    targets.target_for_value(u128::from(mir_importer::DEVICE_RUNTIME_CHECKS_VALUE))
}

/// Result of checking if a function should be collected for device compilation.
#[derive(Debug)]
enum CollectDecision {
    /// Collect this function for device compilation.
    Collect,
    /// Skip this function intentionally (e.g., core::fmt::*, core::panicking::*).
    /// These are filtered out because they can't compile to PTX, but calling them
    /// is expected (panic paths, debug assertions) and will be handled by panic=abort.
    SkipIntentional,
    /// Error: function is from a forbidden crate (std, alloc, etc.).
    /// Device code cannot call these - this is a user error.
    Forbidden { crate_name: String, fn_path: String },
}

// The prefix constants and substring/extractor helpers used below
// (`KERNEL_PREFIX`, `is_kernel_symbol`, `kernel_base_name`, etc.) live in
// the workspace-internal `reserved-oxide-symbols` crate. That crate is the
// single source of truth for the cuda_oxide_* naming contract; see its
// crate-level docs for the layered API and the hash-suffix rationale.
//
// Each prefix contains the magic component `246e25db_`, which makes a
// substring like "cuda_oxide_kernel_" — without the hash — never falsely
// match. The mutual-exclusion guarantee between `DEVICE_PREFIX` and
// `DEVICE_EXTERN_PREFIX` means we no longer need the historical
// "test extern first" ordering dance that lived here previously.
use reserved_oxide_symbols::{
    device_extern_base_name, is_current_device_symbol, is_current_kernel_symbol,
    is_device_extern_symbol, is_device_symbol, is_kernel_symbol, is_legacy_kernel_symbol,
    is_ptx_merge_required_marker, kernel_base_name,
};

/// Sanitize a symbol name for use as a PTX identifier.
///
/// PTX identifiers must match `[a-zA-Z_][a-zA-Z0-9_]*`. This function:
/// - Replaces `$` with `_` (legacy mangling uses `$LT$`, `$GT$`, `$u20$`, etc.)
/// - Replaces `.` with `_` (legacy mangling uses `..` for `::`)
///
/// This must be kept in sync with `mir-importer/src/translator/terminator/mod.rs`
/// which sanitizes call target names the same way.
pub fn sanitize_ptx_name(name: &str) -> String {
    name.replace(['$', '.'], "_")
}

/// Compute the export name for a kernel.
///
/// Naming scheme:
/// - Non-generic kernel                 -> `base_name`
/// - Type/const-generic specialization  -> `base_name + "_TID_" + hex32`
///
/// where `hex32` is the lowercase hex form of
/// `tcx.type_id_hash(instance_ty).as_u128()` and `instance_ty` is the concrete
/// generated kernel function-item type: `FnDef(def_id, [type and const args])`.
/// Hashing that one type gives every specialization a fixed-length identity,
/// preserves generic argument order, and lets rustc own the canonical encoding
/// of const values instead of duplicating it in cuda-oxide.
///
/// The host computes the same value via
/// `cuda_host::type_id_u128_of_val(&kernel_entry::<T, N>)`. Both sides see the
/// same `FnDef` type and go through rustc's region-erasing stable-hash pipeline.
///
/// The scheme is uniform — closures, named types, integers, references
/// — all funnel through one path. That intentionally collapses the
/// older closure-special-case (`_L<line>C<col>`) and the older named-
/// type case (`_Debug-formatted_name`) into the same shape, so closure-
/// generic kernels (`map<T, F: Fn(T) -> T + Copy>`) can finally be
/// launched through the typed `module.<kernel>(...)` API. The host-side
/// `GenericCudaKernel::ptx_name` impl emitted by `#[kernel]` /
/// `#[cuda_module]` produces the exact same string from the type
/// and const specialization represented by the function item at the call site.
fn compute_kernel_export_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    base_name: &str,
) -> String {
    if !tcx
        .generics_of(instance.def_id())
        .requires_monomorphization(tcx)
    {
        return base_name.to_string();
    }

    let instance_ty = instance.ty(tcx, TypingEnv::fully_monomorphized());
    debug_assert!(!instance_ty.has_non_region_param());
    let hash = tcx.type_id_hash(instance_ty).as_u128();
    format!("{}_TID_{:032x}", base_name, hash)
}

/// A function collected for GPU compilation.
///
/// This struct captures everything needed to compile a function to PTX:
/// - The monomorphized instance (with concrete generic arguments)
/// - Whether it's a kernel entry point or a device helper
/// - The name to export in PTX
#[derive(Debug, Clone)]
pub struct CollectedFunction<'tcx> {
    /// The fully monomorphized function instance.
    ///
    /// For generic functions like `add::<f32>`, this contains the concrete
    /// type substitutions. We use this to get the MIR body with all types resolved.
    pub instance: Instance<'tcx>,

    /// True if this is a GPU kernel entry point.
    ///
    /// Kernels are marked with `.entry` in PTX and can be launched from the host.
    /// Non-kernel functions are marked with `.func` and can only be called from device code.
    pub is_kernel: bool,

    /// The name to export in PTX.
    ///
    /// For kernels: the user-visible name (e.g., `add_one`)
    /// For generics: the mangled symbol name (e.g., `_RNvMNtNtCs...`)
    pub export_name: String,
}

/// An external device function declaration (for linking with external LTOIR).
///
/// Unlike `CollectedFunction`, these have no MIR body - they're just declarations
/// that will be emitted as LLVM `declare` statements for nvJitLink to resolve.
#[derive(Debug, Clone)]
pub struct DeviceExternDecl {
    /// The DefId of the extern function declaration
    pub def_id: DefId,

    /// The export name (the original function name, e.g., "cub_block_reduce_sum")
    pub export_name: String,

    /// NVVM attributes extracted from the declaration
    pub attrs: DeviceExternAttrs,
}

/// NVVM attributes for device extern declarations.
///
/// NOTE: These attributes are currently **not emitted** to the LLVM IR output.
/// When linking LTOIR via nvJitLink, the external library's LTOIR already contains
/// proper attributes (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
/// nvJitLink uses the definition's attributes during LTO, making attributes on our
/// declarations redundant.
///
/// The `#[convergent]`, `#[pure]`, and `#[readonly]` Rust attributes are still parsed
/// but their values are not used in code generation. This struct is retained for
/// potential future use or debugging.
#[derive(Debug, Clone, Default)]
pub struct DeviceExternAttrs {
    /// Function is convergent (all threads must execute together).
    /// NOTE: Not currently emitted - external LTOIR has proper convergent attrs.
    pub is_convergent: bool,

    /// Function is pure (no side effects, result depends only on inputs).
    /// NOTE: Not currently emitted - external LTOIR has proper memory attrs.
    pub is_pure: bool,

    /// Function is read-only (only reads memory, doesn't write).
    /// NOTE: Not currently emitted - external LTOIR has proper memory attrs.
    pub is_readonly: bool,
}

/// Result of the collection process.
///
/// Contains both compiled functions (with MIR bodies) and external device
/// declarations (for FFI with external LTOIR).
#[derive(Debug)]
pub struct CollectionResult<'tcx> {
    /// Functions to compile (kernels and device helpers with MIR bodies).
    pub functions: Vec<CollectedFunction<'tcx>>,

    /// External device function declarations (no MIR, emit as `declare`).
    pub device_externs: Vec<DeviceExternDecl>,

    /// Whether this crate needs the run-time loader to merge PTX bundles.
    ///
    /// This is true for a `#[cuda_module]` containing a generic kernel and
    /// for a concrete specialization of a generic kernel imported from a
    /// dependency. Cubin materialization must reject this case until it can
    /// reproduce the same cross-crate linking semantics.
    pub requires_ptx_bundle_merge: bool,
}

/// Counts kernel functions across all codegen units.
///
/// This is a quick scan to determine if device compilation is needed.
/// Returns 0 if no kernels are found, allowing the backend to skip device codegen entirely.
///
/// Note: Only counts fully monomorphized kernels. Generic kernel definitions
/// (like `scale<T>`) are skipped - only concrete instantiations count.
pub fn count_kernels_in_cgus<'tcx>(tcx: TyCtxt<'tcx>, cgus: &[CodegenUnit<'tcx>]) -> usize {
    let mut count = 0;
    for cgu in cgus {
        for (item, _data) in cgu.items() {
            if let MonoItem::Fn(instance) = item
                && is_kernel_function(tcx, instance.def_id())
                && is_fully_monomorphized(tcx, *instance)
            {
                count += 1;
            }
        }
    }
    count
}

/// Counts standalone device function definitions across all codegen units.
///
/// Returns 0 if no standalone device functions are found.
/// Used alongside `count_kernels_in_cgus` to determine if device compilation is needed.
pub fn count_device_fns_in_cgus<'tcx>(tcx: TyCtxt<'tcx>, cgus: &[CodegenUnit<'tcx>]) -> usize {
    let mut count = 0;
    for cgu in cgus {
        for (item, _data) in cgu.items() {
            if let MonoItem::Fn(instance) = item
                && is_device_function(tcx, instance.def_id())
                && is_fully_monomorphized(tcx, *instance)
            {
                count += 1;
            }
        }
    }
    count
}

/// Find a device-code root emitted before the scoped Cargo cache protocol.
///
/// This deliberately inspects both local and external `DefId`s. A generic
/// kernel defined by an older macro may have no monomorphized root in its
/// defining crate; its first concrete `MonoItem` then appears only in a
/// downstream consumer. Recognizing the legacy name there prevents that
/// consumer from seeding a cache entry that would stay fresh across device
/// output or architecture changes.
pub fn unsupported_codegen_protocol_root_in_cgus<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
) -> Option<String> {
    cgus.iter().find_map(|cgu| {
        cgu.items().iter().find_map(|(item, _data)| {
            let MonoItem::Fn(instance) = item else {
                return None;
            };
            let def_id = instance.def_id();
            let item_name = tcx.opt_item_name(def_id)?;
            unsupported_codegen_protocol_root(item_name.as_str()).then(|| tcx.def_path_str(def_id))
        })
    })
}

fn unsupported_codegen_protocol_root(name: &str) -> bool {
    (is_kernel_symbol(name) && !is_current_kernel_symbol(name))
        || (is_device_symbol(name) && !is_current_device_symbol(name))
}

/// Checks if a function is a kernel entry point.
///
/// Detection is based on the generated namespace in the *final* def-path
/// segment: it must start with `KERNEL_PREFIX` (currently
/// `cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_`) or, for roots emitted
/// before the scoped Cargo cache protocol, `LEGACY_KERNEL_PREFIX`. Legacy
/// roots stay classified so the protocol handshake can reject them with an
/// explicit diagnostic instead of silently skipping device codegen. The
/// final-segment check matters because a named item nested inside a kernel
/// has the kernel symbol as an earlier path segment, but is a device helper
/// rather than another entry point.
///
/// ```text
/// User writes:        Macro expands to:
/// ┌─────────────────┐  ┌────────────────────────────────────────────────┐
/// │ #[kernel]       │  │ #[no_mangle]                                   │
/// │ fn add_one(...) │ ⇒│ pub fn cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_add_one(...) │
/// └─────────────────┘  └────────────────────────────────────────────────┘
/// ```
pub fn is_kernel_function(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    is_kernel_entry_def_path(&tcx.def_path_str(def_id))
}

/// Returns `true` when `def_path` names a kernel entry point *itself*, as
/// opposed to an item nested inside a kernel body.
///
/// A closure or named `fn` defined inside a `#[kernel]` has the kernel's name
/// as an earlier path segment (`...::cuda_oxide_kernel_<hash>_k::helper`).
/// Nested items are plain device functions, exported under their canonical
/// mangled symbol by the call-graph walk. Rooting them as kernels would give
/// generic nested fns a `_TID_<hash>` export name that no call site references,
/// failing module verification with "Symbol ... not found".
pub(crate) fn is_kernel_entry_def_path(def_path: &str) -> bool {
    is_current_kernel_symbol(def_path) || is_legacy_kernel_symbol(def_path)
}

/// Checks if a function is a standalone device function definition.
///
/// Detection is based on the `DEVICE_PREFIX` substring added by the
/// `#[device]` macro on `fn` items. The mutual-exclusion property
/// documented in `reserved-oxide-symbols` means we don't need an explicit
/// exclusion of device-extern symbols here — `is_device_symbol` handles it.
pub fn is_device_function(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    is_device_symbol(&tcx.def_path_str(def_id))
}

/// Checks if an Instance is fully monomorphized (no unresolved type or const parameters).
///
/// For generic kernels like `scale<T>`, the CGU may contain both:
/// - The generic definition (with T as a type parameter)
/// - Concrete instantiations (with T = f32, T = i32, etc.)
///
/// We only want to process concrete instantiations since we can't generate
/// PTX for generic code - the device compiler needs concrete types.
///
/// Returns false if any substitution argument is still a type parameter.
pub fn is_fully_monomorphized<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let generics = tcx.generics_of(instance.def_id());

    // The complete generic-argument list includes types, consts, and regions.
    // Regions are erased for codegen, while any remaining type or const
    // parameter means this is still a template rather than a specialization.
    if instance.args.has_non_region_param() {
        return false;
    }

    // Second check: does the def itself have generics that need substitution?
    // Even if args is empty, the function might be generic but not properly instantiated.
    if generics.requires_monomorphization(tcx) && instance.args.is_empty() {
        return false;
    }

    true
}

/// Fit a function path into the forbidden-crate diagnostic box.
///
/// The box pads this field with `{:<48}`, which counts characters, so the
/// budget is a character budget. Two things follow, and the previous
/// `if fn_path.len() > 48 { &fn_path[..45] }` got both wrong for a path that
/// is not pure ASCII:
///
/// * `len()` is bytes. A 30-character path spelled with multi-byte characters
///   can exceed 48 bytes and be truncated even though it would have fit.
/// * `&fn_path[..45]` is a byte index. When byte 45 lands inside a multi-byte
///   character it panics with "byte index 45 is not a char boundary" -- so
///   emitting the diagnostic for a forbidden call would abort the compiler
///   instead of printing the error the box exists to print. Rust identifiers
///   may be non-ASCII, so a path can reach here in that shape.
///
/// Truncation keeps `width - 3` characters and appends an ellipsis, so the
/// result never exceeds `width` characters.
fn truncate_path_for_box(fn_path: &str, width: usize) -> String {
    debug_assert!(width > 3, "the ellipsis needs room");
    if fn_path.chars().count() <= width {
        return fn_path.to_owned();
    }
    let kept: String = fn_path.chars().take(width - 3).collect();
    format!("{kept}...")
}

/// `std::sys::cmath::*` names we allow in device code and rewrite to GPU math.
///
/// When you call `x.tan()` (also `atan`, `acos`, `cbrt`, the hyperbolics,
/// `exp_m1`, `ln_1p`, `hypot`, ...), the compiler turns it into a call to a
/// tiny `std` wrapper like `std::sys::cmath::tan`, which on a CPU forwards to
/// the system C math library. The GPU has no such library, and device code
/// may not call into `std`, so our "no std on the GPU" guard would normally
/// reject the call.
///
/// We never actually run `std` here: mir-importer rewrites each of these
/// names to the matching NVIDIA libdevice function (`__nv_tan`, `__nv_sinh`,
/// ...). This list just tells the guard "these are fine, we handle them."
/// Keep it in sync with the `std::sys::cmath` matches in float_math.rs.
///
/// A few functions (`sin`, `cos`, `exp`, ...) take a different, allowed route
/// on this toolchain: the compiler lowers them to a builtin in `core`, so
/// they never reach here. The ones below still go through `std` because they
/// are not part of `core_float_math`; `sin`/`cos` are listed defensively in
/// case a build ever takes the `std` route too. Only `std::`-prefixed names
/// belong here; `core`-based shims are already allowed.
fn is_intrinsic_lowered_cmath_shim(fn_path: &str) -> bool {
    matches!(
        fn_path,
        "std::sys::cmath::sinf"
            | "std::sys::cmath::sin"
            | "std::sys::cmath::cosf"
            | "std::sys::cmath::cos"
            | "std::sys::cmath::tanf"
            | "std::sys::cmath::tan"
            | "std::sys::cmath::asinf"
            | "std::sys::cmath::asin"
            | "std::sys::cmath::acosf"
            | "std::sys::cmath::acos"
            | "std::sys::cmath::atan2f"
            | "std::sys::cmath::atan2"
            | "std::sys::cmath::atanf"
            | "std::sys::cmath::atan"
            | "std::sys::cmath::cbrtf"
            | "std::sys::cmath::cbrt"
            | "std::sys::cmath::sinhf"
            | "std::sys::cmath::sinh"
            | "std::sys::cmath::coshf"
            | "std::sys::cmath::cosh"
            | "std::sys::cmath::tanhf"
            | "std::sys::cmath::tanh"
            // Inverse hyperbolics: on this nightly (rustc e457a7b0d) only
            // asinh/acosh are `std::sys::cmath` shims; atanh is still the
            // pure-Rust `ln_1p` formula and cmath declares no atanh, so its
            // entries are defensive, for when std makes the same move.
            | "std::sys::cmath::asinhf"
            | "std::sys::cmath::asinh"
            | "std::sys::cmath::acoshf"
            | "std::sys::cmath::acosh"
            | "std::sys::cmath::atanhf"
            | "std::sys::cmath::atanh"
            | "std::sys::cmath::expm1f"
            | "std::sys::cmath::expm1"
            | "std::sys::cmath::log1pf"
            | "std::sys::cmath::log1p"
            | "std::sys::cmath::hypotf"
            | "std::sys::cmath::hypot"
    )
}

/// `std`'s inherent float methods: `std::f32::<impl f32>::atan` and friends.
///
/// `x.atan()` resolves to a one-line `#[inline]` wrapper in `std` whose body
/// bottoms out in a `core` intrinsic or a `std::sys::cmath` shim. Whether the
/// kernel ever *calls* that wrapper depends on rustc's MIR inliner, which is
/// off under `-C incremental` (every Cargo dev-profile build, `cargo oxide
/// test`, `CARGO_INCREMENTAL=1`):
///
/// ```text
/// build                          kernel MIR calls              std guard
/// release, non-incremental       std::sys::cmath::atanf        whitelisted shim
/// -C incremental / opt-level 0   std::f32::<impl f32>::atan    was: forbidden crate `std`
/// ```
///
/// Nothing in these wrappers touches the OS or the heap, so collect them like
/// any other reachable pure function. Every stable method's shim is in
/// `is_intrinsic_lowered_cmath_shim`; the unstable `gamma`/`erf` family and
/// `f128` still stop one level deeper, at their shim, with the guard naming
/// that shim.
fn is_std_float_inherent_method(fn_path: &str) -> bool {
    ["f16", "f32", "f64", "f128"]
        .iter()
        .any(|ty| fn_path.starts_with(&format!("std::{ty}::<impl {ty}>::")))
}

/// True when `def_path` (rustc's definition path without the crate, as
/// `DefPath::to_string_no_crate_verbose` prints it) names an item under core's
/// `core_arch::<arch>` module for any `<arch>` other than `nvptx`.
///
/// ```text
/// user wrote        core::arch::x86_64::_rdtsc
/// defined at        ::core_arch::x86::rdtsc::_rdtsc
/// inlines into      ::core_arch::x86::rdtsc::rdtsc      (foreign fn)   → true
/// ::core_arch::nvptx::_syncthreads                      GPU's own module → false
/// ::core_arch::simd::i32x4::new                         portable SIMD    → false
/// ::hint::spin_loop                                     not core_arch    → false
/// ```
///
/// - Kernel MIR comes from the host-target session, so the host CPU's
///   `core::arch` module is fully visible to device code with no PTX lowering.
/// - Most of its intrinsics carry `#[target_feature]` and are refused by that
///   attribute first. This covers the ones without one (`_rdtsc`) and the
///   private foreign declarations they inline into.
/// - The definition path is used instead of `def_path_str`, whose re-export
///   spelling varies (`std::arch::x86_64`, `core::arch::x86_64`).
/// - Intrinsics that are pure `asm!` (`__cpuid`) leave no call to see; they
///   still fail in the translator, as an unsupported `InlineAsm` terminator.
fn is_host_cpu_arch_def_path(def_path: &str) -> bool {
    let Some(rest) = def_path.strip_prefix("::core_arch::") else {
        return false;
    };
    let Some((module, _item)) = rest.split_once("::") else {
        return false;
    };
    !matches!(module, "nvptx" | "simd" | "macros")
}

/// True for the x86 intrinsic `core::hint::spin_loop` expands to on the host
/// (`_mm_pause`, and the foreign `pause` it inlines into), so the guard can
/// name what the user actually wrote.
fn is_spin_loop_expansion(def_path: &str) -> bool {
    is_host_cpu_arch_def_path(def_path)
        && matches!(def_path.rsplit("::").next(), Some("pause" | "_mm_pause"))
}

/// Returns true for hidden `cuda_device::ptx_asm!` marker functions.
///
/// These markers have host-side `unreachable!()` bodies, but the MIR importer
/// rewrites their call sites to `nvvm.inline_ptx`. Collecting the marker body
/// itself would incorrectly pull panic-message machinery into device code.
fn is_ptx_asm_marker_path(fn_path: &str) -> bool {
    fn has_arity_suffix(path: &str, prefix: &str) -> bool {
        path.strip_prefix(prefix)
            .is_some_and(|suffix| suffix.parse::<usize>().is_ok())
    }

    has_arity_suffix(fn_path, "cuda_device::ptx::__ptx_asm_out_")
        || has_arity_suffix(fn_path, "cuda_device::ptx::__ptx_asm_void_")
}

/// Returns true for the hidden `__unroll_config::<FACTOR>` marker function.
///
/// `#[unroll]` / `#[unroll(N)]` on a loop makes the `#[kernel]` or `#[device]`
/// macro plant a call to this marker at the top of the loop body. The MIR
/// importer rewrites that call to a `mir.unroll_hint` op (consumed by the
/// loop-unroll pass) and never emits a real call. So the marker's empty body must
/// not be collected, or it would show up as a dead `.func` in the generated PTX.
/// Matches both the re-exported path (`cuda_device::__unroll_config`) and the
/// full path.
fn is_unroll_marker_path(fn_path: &str) -> bool {
    fn_path.contains("::__unroll_config")
}

/// Returns true for zero-cost compile-time configuration markers planted by
/// proc macros (launch metadata and the unchecked-indexing flag).
///
/// The MIR importer consumes these calls while translating their containing
/// function. Their own empty bodies are not device functions and must not be
/// collected into the output module. Match both the defining `thread` path and
/// cuda-device's root re-export spelling used by rustc's path printer.
fn is_launch_metadata_marker_path(fn_path: &str) -> bool {
    matches!(
        fn_path,
        "cuda_device::__launch_bounds_config"
            | "cuda_device::thread::__launch_bounds_config"
            | "cuda_device::__launch_contract_config"
            | "cuda_device::thread::__launch_contract_config"
            | "cuda_device::__launch_contract_block_config"
            | "cuda_device::thread::__launch_contract_block_config"
            | "cuda_device::__grid_constant_config"
            | "cuda_device::thread::__grid_constant_config"
            | "cuda_device::__unchecked_indexing_config"
            | "cuda_device::thread::__unchecked_indexing_config"
    ) || fn_path.strip_prefix("cuda_device::").is_some_and(|path| {
        path.starts_with("__launch_bounds_config::<")
            || path.starts_with("thread::__launch_bounds_config::<")
            || path.starts_with("__launch_contract_config::<")
            || path.starts_with("thread::__launch_contract_config::<")
            || path.starts_with("__launch_contract_block_config::<")
            || path.starts_with("thread::__launch_contract_block_config::<")
            || path.starts_with("__grid_constant_config::<")
            || path.starts_with("thread::__grid_constant_config::<")
            || path.starts_with("__unchecked_indexing_config::<")
            || path.starts_with("thread::__unchecked_indexing_config::<")
    })
}

/// Marker substring of the panic message used by the public
/// `cuda_device::thread::index_*` stubs (see `cuda-device/src/thread.rs`).
///
/// Those public items exist only so imports resolve; real call sites
/// inside `#[kernel]` / `#[device]` bodies are rewritten by the proc
/// macros to `thread::__internal::*`. When this message shows up in
/// device-reachable MIR, a stub was reached through code that the macros
/// never rewrote, which means a helper function is missing `#[device]`.
const MISSING_DEVICE_STUB_MARKER: &str = "called outside #[kernel] / #[device]";

/// If `fn_path` names one of the Rust global-allocator entry points,
/// returns the bare shim name (for use in the diagnostic), else `None`.
///
/// Every heap allocation (`Vec`, `Box`, `String`, ...) eventually funnels
/// into one of these. They have no MIR body (they are resolved by the
/// linker against the program's `#[global_allocator]`), and no device-side
/// allocator exists, so reaching one from a kernel can never work. This
/// list is the single switch point to revisit if a device allocator ever
/// lands.
fn rust_alloc_shim_name(fn_path: &str) -> Option<&str> {
    const SHIMS: [&str; 6] = [
        "__rust_alloc",
        "__rust_alloc_zeroed",
        "__rust_dealloc",
        "__rust_realloc",
        "__rust_no_alloc_shim_is_unstable_v2",
        "handle_alloc_error",
    ];
    let last = fn_path.rsplit("::").next().unwrap_or(fn_path);
    // None of these names is actually reserved: a user may legally define
    // their own `__rust_alloc` or `handle_alloc_error`, and such functions
    // compile for the device like any other. So a bare-name match is not
    // enough; the path must also come from the sysroot allocator machinery
    // (`alloc::alloc::*`, or the `std::alloc::*` re-export spelling). The
    // caller additionally skips local definitions by `DefId`.
    let matched = SHIMS.iter().find(|s| **s == last).copied()?;
    if !(fn_path.starts_with("alloc::") || fn_path.starts_with("std::alloc::")) {
        return None;
    }
    Some(matched)
}

/// If `constant` is a `&str` constant, returns its text.
///
/// Used to inspect panic message strings in device-reachable MIR. Only
/// fat-pointer string constants are inspected; everything else returns
/// `None`. Evaluation failures (e.g. a constant in still-generic MIR)
/// also return `None`, which simply means "no message found".
fn const_str_text<'tcx>(tcx: TyCtxt<'tcx>, constant: &ConstOperand<'tcx>) -> Option<String> {
    let ty = constant.const_.ty();
    let TyKind::Ref(_, pointee, _) = ty.kind() else {
        return None;
    };
    if !matches!(pointee.kind(), TyKind::Str) {
        return None;
    }
    let val = constant
        .const_
        .eval(tcx, TypingEnv::fully_monomorphized(), constant.span)
        .ok()?;
    // `try_get_slice_bytes_for_diagnostics` ICEs on scalar / zero-sized
    // values, so only call it for the representations a `&str` constant
    // can actually have.
    if !matches!(val, ConstValue::Slice { .. } | ConstValue::Indirect { .. }) {
        return None;
    }
    let bytes = val.try_get_slice_bytes_for_diagnostics(tcx)?;
    Some(String::from_utf8_lossy(bytes).into_owned())
}

/// Extracts the stub function name out of a stub panic message.
///
/// The message reads `internal error: entered unreachable code:
/// thread::index_1d called outside #[kernel] / #[device] ...`, so the
/// stub name is the last word before `" called outside"`.
fn stub_name_from_marker_message(text: &str) -> &str {
    text.split(" called outside")
        .next()
        .and_then(|prefix| prefix.rsplit(' ').next())
        .filter(|name| !name.is_empty())
        .unwrap_or("thread::index_*")
}

/// Maps a MIR `SourceInfo` back to the span the user wrote.
///
/// The MIR inliner copies callee statements (with their callee-file
/// spans) into the caller and records the original call site in the
/// source-scope tree instead of in the span itself. Walking the chain of
/// inlined scopes recovers the outermost call site, the one that lives in
/// this body's own source; `source_callsite()` then additionally unwinds
/// any macro expansions (`vec!`, `panic!`, ...) sitting on top of it.
fn outermost_user_span<'tcx>(
    mir: &rustc_middle::mir::Body<'tcx>,
    source_info: rustc_middle::mir::SourceInfo,
) -> Span {
    let mut span = source_info.span;
    let mut scope = Some(source_info.scope);
    while let Some(s) = scope {
        let data = &mir.source_scopes[s];
        if let Some((_, callsite)) = data.inlined {
            span = callsite;
        }
        scope = data.inlined_parent_scope;
    }
    span.source_callsite()
}

/// MIR visitor that records every `&str` constant in the visited range,
/// together with where it appeared (statement vs. terminator) and its
/// source span.
struct StrConstScan<'tcx> {
    tcx: TyCtxt<'tcx>,
    found: Vec<(Location, Span, String)>,
}

impl<'tcx> Visitor<'tcx> for StrConstScan<'tcx> {
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, location: Location) {
        if let Some(text) = const_str_text(self.tcx, constant) {
            self.found.push((location, constant.span, text));
        }
        self.super_const_operand(constant, location);
    }
}

/// Provenance of a function discovered during the call-graph walk:
/// which root (kernel or standalone device fn) the walk started from, and
/// the nearest call site that still lives in user-written code.
///
/// Sysroot-internal call chains (e.g. `Box::new` -> `box_new_uninit` ->
/// `Global::alloc_impl` -> `__rust_alloc`) carry spans that point into
/// the standard library sources, which is exactly the kind of span the
/// inscrutable historic errors exposed. Propagating the last user-code
/// span along each discovery edge lets diagnostics point at the line in
/// the kernel that started the chain instead.
#[derive(Clone)]
struct DiscoveryCtx {
    /// Export name of the root the walk started from.
    root_name: String,
    /// True when the root is a `#[kernel]` entry point (as opposed to a
    /// standalone `#[device]` function).
    root_is_kernel: bool,
    /// Nearest enclosing user-code span on the discovery path.
    user_span: Span,
}

impl DiscoveryCtx {
    /// "kernel" or "device function", for diagnostics that name the root.
    fn root_kind(&self) -> &'static str {
        if self.root_is_kernel {
            "kernel"
        } else {
            "device function"
        }
    }
}

/// Collects all device-reachable functions starting from kernel entry points.
///
/// This is the main entry point for device function collection. It:
///
/// 1. Finds all kernel entry points in the CGUs
/// 2. Walks the call graph from each kernel
/// 3. Returns all functions that need to be compiled to PTX
///
/// ## Parameters
///
/// - `tcx`: The type context containing all MIR bodies
/// - `cgus`: Codegen units from `tcx.collect_and_partition_mono_items()`
/// - `verbose`: If true, prints collection progress to stderr
///
/// ## Returns
///
/// A `CollectionResult` containing:
/// - `functions`: Collected functions with MIR bodies (kernels first, then callees)
/// - `device_externs`: External device function declarations (for FFI with external LTOIR)
pub fn collect_device_functions<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    verbose: bool,
) -> CollectionResult<'tcx> {
    let mut collector = DeviceCollector::new(tcx, verbose);
    let mut requires_ptx_bundle_merge = cgus.iter().any(|cgu| {
        cgu.items().iter().any(|(item, _data)| match item {
            MonoItem::Static(def_id) => is_ptx_merge_required_marker(&tcx.def_path_str(*def_id)),
            _ => false,
        })
    });

    // Find all kernel entry points
    for cgu in cgus {
        for (item, _data) in cgu.items() {
            if let MonoItem::Fn(instance) = item
                && is_kernel_function(tcx, instance.def_id())
            {
                // Skip generic (non-monomorphized) instances.
                // For generic kernels like scale<T>, the CGU contains both:
                // - The generic definition (scale<T>) - skip this
                // - Concrete instantiations (scale::<f32>) - process this
                if !is_fully_monomorphized(tcx, *instance) {
                    if verbose {
                        let name = tcx.def_path_str(instance.def_id());
                        eprintln!(
                            "[collector] Skipping non-monomorphized kernel: {} (needs type/const specialization)",
                            name
                        );
                    }
                    continue;
                }

                // A downstream monomorphization of a generic kernel may be
                // present even when the defining crate's private module
                // marker is not. Such modules use the all-PTX-bundles loader,
                // so strict ahead-of-time cubin materialization cannot safely
                // compile only this crate's artifact.
                requires_ptx_bundle_merge |= tcx
                    .generics_of(instance.def_id())
                    .requires_monomorphization(tcx);

                let name = tcx.def_path_str(instance.def_id());
                // Extract the kernel base name by stripping the reserved
                // `cuda_oxide_kernel_<hash>_` prefix. Cross-crate kernels look
                // like `kernel_lib::cuda_oxide_kernel_<hash>_scale`; the
                // helper handles both bare and FQDN forms uniformly.
                let base_name = kernel_base_name(&name)
                    .map(str::to_string)
                    .unwrap_or_else(|| name.rsplit("::").next().unwrap_or(&name).to_string());

                // Compute a unique export name for this kernel monomorphization.
                // Non-generic kernels keep the base name (e.g. "vecadd").
                // Type/const-generic kernels (including closure-generic) get
                // "<base>_TID_<hex32>", where <hex32> is the hash of the
                // concrete kernel function-item type. The host-side
                // `ptx_name()` emitted by `#[kernel]` / `#[cuda_module]`
                // computes the same string from the same `FnDef` type.
                let export_name = compute_kernel_export_name(tcx, *instance, &base_name);

                if verbose {
                    eprintln!("[collector] Found kernel: {} -> {}", name, export_name);
                }

                collector.add_root(*instance, true, export_name);
            }
        }
    }

    // Find standalone device function roots (Phase 2: device functions without kernels).
    // Only scan when there are NO kernels — when kernels exist, device functions are
    // already collected transitively via the call graph walk.
    if collector.worklist.is_empty() {
        for cgu in cgus {
            for (item, _data) in cgu.items() {
                if let MonoItem::Fn(instance) = item
                    && is_device_function(tcx, instance.def_id())
                    && is_fully_monomorphized(tcx, *instance)
                {
                    let raw_name = tcx.def_path_str(instance.def_id());

                    // Skip closures inside device functions
                    if raw_name.contains("{closure") || raw_name.contains("::closure") {
                        continue;
                    }

                    // Use FQDN so the export name matches what the MIR translator
                    // sees via `CrateDef::name()` on the call side. The lowering
                    // layer converts `::` to `__` on both sides.
                    let name = collector.fqdn(*instance);
                    let export_name = collector.compute_export_name(&name, *instance);

                    if verbose {
                        eprintln!(
                            "[collector] Found standalone device function: {} -> {}",
                            name, export_name
                        );
                    }

                    // Add as a non-kernel root — produces .func (not .entry) in PTX
                    collector.add_root(*instance, false, export_name);
                }
            }
        }
    }

    // Process the worklist to collect all reachable functions
    let mut result = collector.collect();
    result.requires_ptx_bundle_merge = requires_ptx_bundle_merge;
    result
}

/// Worklist-based collector for device-reachable functions.
///
/// Uses breadth-first traversal to discover all functions reachable from kernels.
/// This ensures we don't miss any callees, even through deep call chains.
struct DeviceCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// Mangled names of functions already seen (prevents duplicates and infinite loops).
    /// We use mangled names because they uniquely identify each monomorphization,
    /// unlike DefId which is shared across all instantiations of a generic function.
    seen: HashSet<String>,
    /// Export names already used (prevents name conflicts in PTX).
    used_export_names: HashSet<String>,
    /// Functions awaiting processing.
    worklist: VecDeque<CollectedFunction<'tcx>>,
    /// Discovery provenance per collected function, keyed by mangled
    /// symbol name (same key as `seen`). Used by diagnostics to name the
    /// originating kernel and to point at user code instead of sysroot
    /// internals.
    discovery: HashMap<String, DiscoveryCtx>,
    /// Functions collected so far, in discovery order.
    result: Vec<CollectedFunction<'tcx>>,
    /// External device function declarations collected (for FFI with external LTOIR).
    device_externs: Vec<DeviceExternDecl>,
    /// DefIds of device externs already seen (prevents duplicates).
    seen_device_externs: HashSet<DefId>,
    /// Whether we've already emitted the "`DynamicSharedArray::get`
    /// needs `shared_mem_bytes` set at launch" warning. Fired at most
    /// once per program — the message is procedural advice about the
    /// kernel-launch contract, not anything per-call.
    warned_dynamic_shared_array: bool,
    /// Print progress to stderr.
    verbose: bool,
}

impl<'tcx> DeviceCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, verbose: bool) -> Self {
        Self {
            tcx,
            seen: HashSet::new(),
            used_export_names: HashSet::new(),
            worklist: VecDeque::new(),
            discovery: HashMap::new(),
            result: Vec::new(),
            device_externs: Vec::new(),
            seen_device_externs: HashSet::new(),
            warned_dynamic_shared_array: false,
            verbose,
        }
    }

    /// Returns the fully qualified domain name (FQDN) for an instance.
    ///
    /// `def_path_str()` normally omits the crate name for local items (e.g. returns
    /// `cuda_oxide_device_<hash>_vecadd` instead of
    /// `helper_fn::cuda_oxide_device_<hash>_vecadd`).
    /// This method asks rustc's printer to resolve crate names exactly like
    /// `rustc_public`, ensuring that call sites and definitions use identical
    /// strings before lowering converts `::` to `__`.
    ///
    /// Use `with_no_trimmed_paths!` because rustc's display-oriented path
    /// printer can otherwise shorten concrete impl type arguments, e.g.
    /// `Wrapper::<Vec2>::dot_plus`, while stable MIR call sites use the full
    /// `Wrapper::<crate_name::Vec2>::dot_plus` spelling.
    fn fqdn(&self, instance: Instance<'tcx>) -> String {
        let def_id = instance.def_id();
        rustc_middle::ty::print::with_resolve_crate_name!(
            rustc_middle::ty::print::with_no_trimmed_paths!(
                self.tcx.def_path_str_with_args(def_id, instance.args)
            )
        )
    }

    /// Adds a root function (kernel) to start collection from.
    fn add_root(&mut self, instance: Instance<'tcx>, is_kernel: bool, export_name: String) {
        // Use mangled name as the unique key - this distinguishes different
        // monomorphizations of the same generic function (e.g., map<f32, Closure1>
        // vs map<f32, Closure2>)
        let mangled = self.tcx.symbol_name(instance).name.to_string();
        if self.seen.insert(mangled.clone()) {
            // A root is its own provenance: diagnostics fall back to its
            // definition site until a more precise user-code call site is
            // recorded along a discovery edge.
            self.discovery.insert(
                mangled,
                DiscoveryCtx {
                    root_name: export_name.clone(),
                    root_is_kernel: is_kernel,
                    user_span: self.tcx.def_span(instance.def_id()),
                },
            );
            self.used_export_names.insert(export_name.clone());
            self.worklist.push_back(CollectedFunction {
                instance,
                is_kernel,
                export_name,
            });
        }
    }

    /// Runs collection to completion, returning all discovered functions and extern declarations.
    fn collect(mut self) -> CollectionResult<'tcx> {
        while let Some(func) = self.worklist.pop_front() {
            let def_id = func.instance.def_id();

            // Look up where this function was discovered from. Every
            // enqueued function gets an entry; the fallback only guards
            // against future call paths that forget to record one.
            let mangled = self.tcx.symbol_name(func.instance).name.to_string();
            let ctx = self
                .discovery
                .get(&mangled)
                .cloned()
                .unwrap_or_else(|| DiscoveryCtx {
                    root_name: func.export_name.clone(),
                    root_is_kernel: func.is_kernel,
                    user_span: self.tcx.def_span(def_id),
                });

            // Host-CPU-only functions (`#[target_feature]`, `core::arch::<host>`)
            // can reach device code because kernel MIR comes from the
            // host-target session. Refuse them here, the one point every
            // function with a body passes through (roots, callees, closures,
            // fn items, trait-dispatched impls). Callees without a body are
            // checked at the top of `process_call_operand`. Drop glue and
            // the `FnPtr::addr` shim are the only compiler-built shims that
            // reach this loop, and neither carries attributes; any other shim
            // kind (a reified `#[target_feature]` fn, say) must be checked.
            if !matches!(
                func.instance.def,
                InstanceKind::Shim(ShimKind::DropGlue(..) | ShimKind::FnPtrAsPtr(..))
            ) {
                self.check_host_cpu_only(def_id, ctx.user_span, &ctx, None);
            }

            // Get MIR body if available. For drop glue shims
            // (InstanceKind::DropGlue), `is_mir_available` may return false
            // because the shim is compiler-generated, but `instance_mir`
            // still provides the body.
            let has_mir = self.tcx.is_mir_available(def_id)
                || matches!(
                    func.instance.def,
                    // Compiler-built shims with no HIR body but an
                    // `instance_mir` body: drop glue, and (since
                    // nightly-2026-08-28) the `<fn(..) as FnPtr>::as_ptr`
                    // shim that backs `FnPtr::addr`.
                    InstanceKind::Shim(ShimKind::DropGlue(..) | ShimKind::FnPtrAsPtr(..))
                );
            if has_mir {
                // Use instance_mir for monomorphized MIR.
                // This returns OPTIMIZED MIR (post -C opt-level passes).
                let mir = self.tcx.instance_mir(func.instance.def);

                if self.verbose {
                    eprintln!(
                        "[collector] Processing {} ({} basic blocks)",
                        func.export_name,
                        mir.basic_blocks.len()
                    );
                }

                // Skip blocks that are unreachable after monomorphization —
                // e.g. the const-false arm of `if S::CONST_FLAG` in a generic
                // fn. rustc's own collector walks only mono-reachable blocks
                // (see `Body::mono_successors`), so dead-arm callees are never
                // instantiated anywhere else; collecting them here would
                // demand symbols that can never resolve, and reject panic-only
                // hooks that are never actually called. This exact set is
                // also carried across the rustc_public bridge to the MIR
                // importer, so collection and translation have one semantic
                // owner instead of two constant evaluators.
                let reachable = device_mono_reachable_as_bitset(mir, self.tcx, func.instance);

                // Fail fast with an actionable diagnostic when this body
                // contains panic-formatting machinery the device pipeline
                // cannot compile (issue #76). Skip for drop glue shims:
                // their MIR is compiler-generated and may contain panic
                // paths (e.g. for assertion failures) that are unreachable
                // in practice; the mir-importer handles these via its
                // existing unreachable-block patching.
                if !matches!(
                    func.instance.def,
                    InstanceKind::Shim(ShimKind::DropGlue(..))
                ) {
                    self.check_panic_machinery(mir, &func, &ctx, &reachable);
                }

                // Walk the reachable basic blocks looking for calls.
                // Pass the caller so we can substitute its args into callees
                // and attribute diagnostics to the right discovery path.
                for (bb, bb_data) in mir.basic_blocks.iter_enumerated() {
                    if !reachable.contains(bb) {
                        continue;
                    }
                    if let Some(ref terminator) = bb_data.terminator {
                        self.process_terminator(terminator, mir, &func, &ctx);
                    }
                }
            }

            self.result.push(func);
        }

        CollectionResult {
            functions: self.result,
            device_externs: self.device_externs,
            requires_ptx_bundle_merge: false,
        }
    }

    /// Adds an external device function declaration (for FFI with external LTOIR).
    fn add_device_extern(&mut self, def_id: DefId, full_name: &str) {
        // Skip if already seen
        if !self.seen_device_externs.insert(def_id) {
            return;
        }

        // Extract the original function name (strip the prefix)
        // The #[link_name] attribute on the extern fn has the original name.
        // `device_extern_base_name` returns the part after DEVICE_EXTERN_PREFIX
        // and works for both bare and FQDN forms.
        let export_name = device_extern_base_name(full_name)
            .map(str::to_string)
            .unwrap_or_else(|| full_name.to_string());

        // Extract NVVM attributes from the function's attributes
        let attrs = self.extract_device_extern_attrs(def_id);

        if self.verbose {
            eprintln!(
                "[collector] Found device extern: {} (convergent={}, pure={}, readonly={})",
                export_name, attrs.is_convergent, attrs.is_pure, attrs.is_readonly
            );
        }

        self.device_externs.push(DeviceExternDecl {
            def_id,
            export_name,
            attrs,
        });
    }

    /// Extract NVVM attributes from a device extern function's rustc attributes.
    fn extract_device_extern_attrs(&self, def_id: DefId) -> DeviceExternAttrs {
        use rustc_span::Symbol;
        let mut attrs = DeviceExternAttrs::default();

        let check = |name| {
            self.tcx
                .get_attrs_by_path(def_id, &[Symbol::intern(name)])
                .next()
                .is_some()
        };
        attrs.is_convergent = check("convergent");
        attrs.is_pure = check("pure");
        attrs.is_readonly = check("readonly");

        attrs
    }

    /// Process a terminator to find function calls.
    ///
    /// `caller` is the collected function containing this terminator.
    /// We use its instance args to substitute into callee args when the
    /// caller is generic, and its discovery context (`ctx`) to attribute
    /// diagnostics to the right kernel and user-code span.
    fn process_terminator(
        &mut self,
        terminator: &rustc_middle::mir::Terminator<'tcx>,
        mir: &rustc_middle::mir::Body<'tcx>,
        caller: &CollectedFunction<'tcx>,
        ctx: &DiscoveryCtx,
    ) {
        match &terminator.kind {
            TerminatorKind::Call { func, .. } => {
                // The span of the whole call expression, mapped back through
                // MIR inlining and macro expansion to the line the user wrote.
                // (The function operand's own span is reset to a dummy by MIR
                // inlining, so it is useless for diagnostics.)
                let call_span = outermost_user_span(mir, terminator.source_info);
                self.process_call_operand(func, call_span, caller, ctx);
            }
            TerminatorKind::Drop { place, .. } => {
                // Collect `drop_in_place::<T>` when T has non-trivial drop glue.
                // Without this, the MIR translator would emit a call to a
                // `drop_in_place` symbol that has no definition in the module.
                self.process_drop_place(place, mir, caller, ctx);
            }
            _ => {}
        }
    }

    /// Collect the `drop_in_place::<T>` function for a `Drop` terminator.
    ///
    /// When the MIR contains a `Drop` terminator for a place of type `T`,
    /// rustc generates a `drop_in_place::<T>` shim that calls `<T as Drop>::drop`
    /// (if `T` implements `Drop`) and recursively drops each field. The
    /// mir-importer needs this function to exist as a device function so it
    /// can emit a call to it.
    ///
    /// The shim's MIR body may itself contain `Call` terminators (to
    /// `<T as Drop>::drop` and to nested `drop_in_place::<FieldTy>`) and
    /// `Drop` terminators (for fields). These are discovered transitively
    /// when the shim's body is walked by `collect()` on subsequent iterations.
    ///
    /// Drop glue instances resolve to `InstanceKind::DropGlue`, not
    /// `InstanceKind::Item`, so the regular `process_call_operand` path
    /// would skip them. This dedicated handler resolves the drop instance
    /// directly and enqueues it.
    fn process_drop_place(
        &mut self,
        place: &rustc_middle::mir::Place<'tcx>,
        mir: &rustc_middle::mir::Body<'tcx>,
        caller: &CollectedFunction<'tcx>,
        ctx: &DiscoveryCtx,
    ) {
        use rustc_middle::ty::EarlyBinder;

        // Compute the type of the dropped place, substituting the caller's
        // generic args to get a fully monomorphized type.
        let place_ty = place.ty(mir, self.tcx).ty;
        let place_ty = self.tcx.instantiate_and_normalize_erasing_regions(
            caller.instance.args,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, place_ty),
        );

        // Resolve drop_in_place::<T>. This returns the drop glue shim
        // (InstanceKind::DropGlue) which wraps the actual Drop::drop call.
        let drop_instance = Instance::resolve_drop_glue(self.tcx, place_ty);

        // DropGlue(_, None) is an empty shim for types that need no
        // destructor. The mir-importer's no-op analysis will lower these
        // as plain branches, so there's nothing to collect.
        if let InstanceKind::Shim(ShimKind::DropGlue(_, None)) = drop_instance.def {
            return;
        }

        let mangled = self.tcx.symbol_name(drop_instance).name.to_string();
        if self.seen.contains(&mangled) {
            return;
        }

        // Ensure the type is fully monomorphized
        if !is_fully_monomorphized(self.tcx, drop_instance) {
            if self.verbose {
                eprintln!(
                    "[collector] Skipping non-monomorphized drop_in_place: {:?}",
                    place_ty
                );
            }
            return;
        }

        // Skip drop glue whose shim body is provably a no-op: the
        // mir-importer's translate_drop lowers such drops to plain
        // branches and never references the shim, so collecting it would
        // only translate dead code and transitively pull in stdlib
        // `Drop::drop` bodies (e.g. `core::array::IntoIter`'s
        // `PolymorphicIter`) whose MIR the device pipeline cannot
        // compile. This is the same predicate the importer's emit
        // decision and the device_codegen translation filter consult,
        // so the three decisions cannot drift.
        if self.drop_glue_is_noop(drop_instance) {
            if self.verbose {
                eprintln!("[collector] Skipping no-op drop_in_place::<{:?}>", place_ty);
            }
            return;
        }

        let drop_ctx = DiscoveryCtx {
            root_name: ctx.root_name.clone(),
            root_is_kernel: ctx.root_is_kernel,
            user_span: ctx.user_span,
        };

        // Use the mangled name as the export name for drop glue instances.
        // The mir-importer uses Instance::resolve_drop_in_place followed by
        // mangled_name() to derive the callee symbol, so the export name
        // must match. Drop glue always has generic args (the dropped type),
        // so mangled_name is the correct choice (never the simple FQDN).
        let export_name = sanitize_ptx_name(&mangled);

        if self.verbose {
            eprintln!(
                "[collector] Discovered drop_in_place::<{:?}> -> {}",
                place_ty, export_name
            );
        }

        self.discovery.insert(mangled.clone(), drop_ctx);
        self.seen.insert(mangled);
        self.used_export_names.insert(export_name.clone());
        self.worklist.push_back(CollectedFunction {
            instance: drop_instance,
            is_kernel: false,
            export_name,
        });
    }

    /// Returns true when the drop glue `instance` is provably a no-op.
    ///
    /// Delegates to `mir_importer::drop_instance_is_noop`, the single
    /// shared predicate that also drives the importer's emit decision
    /// (`translate_drop`) and the `device_codegen` translation filter.
    /// Sharing one predicate keeps collection, emission, and translation
    /// in lockstep: everything the importer calls is collected, and
    /// nothing dead is translated.
    ///
    /// The predicate needs stable MIR queries, so a scoped
    /// `rustc_internal::run` context is set up per query. Collection runs
    /// before `device_codegen` enters its own context, so this never
    /// nests; if it ever does, `run` returns an error and we fall back to
    /// collecting the shim, a safe over-approximation that the
    /// `device_codegen` filter prunes with the same predicate.
    fn drop_glue_is_noop(&self, instance: Instance<'tcx>) -> bool {
        use rustc_public::rustc_internal;
        rustc_internal::run(self.tcx, || {
            mir_importer::drop_instance_is_noop(&rustc_internal::stable(instance))
        })
        .unwrap_or(false)
    }

    /// Process a call operand to extract and add the callee.
    ///
    /// This is where we enforce the `no_std` requirement for device code.
    /// If the call target is from a forbidden crate (std, alloc, etc.),
    /// we panic with a clear error message.
    ///
    /// `caller` is the collected function containing this call, used to
    /// substitute its generic args into the callee's args when needed.
    /// `ctx` is the caller's discovery provenance, used for diagnostics.
    fn process_call_operand(
        &mut self,
        func: &rustc_middle::mir::Operand<'tcx>,
        call_span: Span,
        caller: &CollectedFunction<'tcx>,
        ctx: &DiscoveryCtx,
    ) {
        use rustc_middle::mir::Operand;
        use rustc_middle::ty::EarlyBinder;

        let Operand::Constant(const_op) = func else {
            return;
        };

        let ty = const_op.const_.ty();
        let TyKind::FnDef(def_id, args) = ty.kind() else {
            return;
        };
        let fn_path = self.tcx.def_path_str(*def_id);
        if fn_path.contains("DynamicSharedArray")
            && (fn_path.contains("::get")
                || fn_path.contains("::get_raw")
                || fn_path.contains("::offset"))
            && !self.warned_dynamic_shared_array
        {
            self.warned_dynamic_shared_array = true;
            self.tcx.sess.dcx().span_warn(
                const_op.span,
                "`DynamicSharedArray` returns CUDA dynamic shared memory; make sure the kernel launch config provides enough `shared_mem_bytes`",
            );
        }

        // CRITICAL: Substitute the caller's args into the callee's args.
        //
        // When walking the MIR of a generic function like
        // `cuda_oxide_kernel_<hash>_scale<T>`, calls to other functions may have
        // generic args like `[T]`. We substitute the caller's concrete args
        // (e.g., `[f32]`) to get the actual monomorphized callee.
        //
        // Example:
        //   Caller: cuda_oxide_kernel_<hash>_scale::<f32> (args = [f32])
        //   Call in MIR: scale<T>(...)  (args = [T])
        //   After substitution: scale::<f32> (args = [f32])
        //
        // `TyKind::FnDef` args sit behind a `Binder` since rust-lang/rust
        // PR "place FnDef behind a binder" (8fb83aba335): the binder scopes
        // the function's LATE-bound (lifetime) vars. By the time a FnDef
        // type appears in built MIR those binders have been instantiated
        // (with dummy regions), so no bound vars can remain here; the
        // caller's still-generic EARLY-bound params (`T`) are not binder
        // vars and are substituted by `instantiate_and_normalize` below.
        // rustc_monomorphize::collector uses `args.no_bound_vars().unwrap()`
        // at its equivalent Call-terminator sites; we follow it rather than
        // `skip_binder()`, which would silently discard a bound var if one
        // ever appeared.
        let args = args
            .no_bound_vars()
            .expect("FnDef args in built MIR carry no late-bound vars");
        let args = self.tcx.instantiate_and_normalize_erasing_regions(
            caller.instance.args,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, args),
        );

        // The call site is the best span while the caller is user code;
        // afterwards keep the last user-code span recorded on the walk. Used
        // for the host-CPU guard here and for every callee context below.
        let user_span = if caller.instance.def_id().is_local() && !call_span.is_dummy() {
            call_span
        } else {
            ctx.user_span
        };

        // Host-CPU guard, before any skip below can drop the callee silently.
        // A body-less callee (the foreign `rdtsc` that `_rdtsc` inlines into)
        // never reaches the worklist check, and the verbose trace shows it
        // leaves this function through an early return before the "no MIR"
        // skip, so this is the one place that sees it. The check is cheap
        // when it does not fire: one cached attribute query and one crate
        // compare. Name the caller only when it is not the root itself.
        let via = (caller.export_name != ctx.root_name).then_some(caller.instance.def_id());
        self.check_host_cpu_only(*def_id, user_span, ctx, via);

        // Check if function is from a crate we should compile
        match self.should_collect_from_crate(*def_id) {
            CollectDecision::Collect => {
                // Continue processing below
            }
            CollectDecision::SkipIntentional => {
                // Silently skip (fmt, panicking, etc.)
                return;
            }
            CollectDecision::Forbidden {
                crate_name,
                fn_path,
            } => {
                // ERROR: Device code is calling a forbidden crate!
                // Build a formatted error box (68 char inner width)
                let border = "═".repeat(68);
                let empty_line = format!("║{:68}║", "");

                let fn_display = truncate_path_for_box(&fn_path, 48);

                // Build the "From crate" line with proper padding
                let crate_line = format!("║ From crate: '{}'", crate_name);
                let crate_padded = format!("{:<69}║", crate_line);

                // Build the last line with proper padding
                let last_line = format!(
                    "║ The '{}' crate cannot run on GPU (requires OS/heap).",
                    crate_name
                );
                let last_padded = format!("{:<69}║", last_line);

                panic!(
                    "\n\n\
╔{border}╗
║{title:^68}║
╠{border}╣
║ Device code calls: {fn:<48}║
{crate_line}
{empty}
║ Only these crates are allowed in device code:                      ║
║   • Local crate (your kernel code)                                 ║
║   • cuda_device (GPU intrinsics)                                     ║
║   • core (no_std standard library)                                 ║
{empty}
{last_line}
╚{border}╝
\n",
                    border = border,
                    title = "CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE",
                    fn = fn_display,
                    crate_line = crate_padded,
                    empty = empty_line,
                    last_line = last_padded,
                );
            }
        }

        // Derive the discovery provenance for whatever this call edge leads
        // to. While we are still inside user-written code (the caller is in
        // the local crate), the call site itself is the most precise span we
        // will ever have; `source_callsite()` walks macro expansions and
        // MIR-inlined frames back to the line the user actually wrote. Once
        // the walk has left user code (sysroot internals), keep the last
        // user-code span we recorded.
        let callee_ctx = DiscoveryCtx {
            root_name: ctx.root_name.clone(),
            root_is_kernel: ctx.root_is_kernel,
            user_span,
        };

        // Callable-trait shims do not necessarily have a MIR body of their own.
        // Identify the actual `Fn`, `FnMut`, or `FnOnce` trait through rustc's
        // metadata, then collect only the trait's `Self` type (the receiver).
        // Function-name matching is unsafe here: user functions may legally
        // contain strings such as `call_once`.
        let is_callable_trait_method = self
            .tcx
            .trait_of_assoc(*def_id)
            .is_some_and(|trait_id| self.tcx.fn_trait_kind_from_def_id(trait_id).is_some());
        if is_callable_trait_method
            && let Some(receiver_ty) = args.iter().next().and_then(|arg| arg.as_type())
        {
            self.enqueue_callable_trait_receiver_body(receiver_ty, call_span, caller, &callee_ctx);
            // Don't return - continue to try resolving the trait method too
            // (even though it may fail, we still want to try).
        }

        // Try to resolve the instance with substitutions first, so we can
        // check if we've already seen THIS specific monomorphization
        let typing_env = TypingEnv::fully_monomorphized();
        let Some(resolved) = Instance::try_resolve(self.tcx, typing_env, *def_id, args)
            .ok()
            .flatten()
        else {
            return;
        };

        // Skip already-seen monomorphizations (use mangled name as unique key)
        let mangled = self.tcx.symbol_name(resolved).name.to_string();
        if self.seen.contains(&mangled) {
            return;
        }

        // Skip non-monomorphized instances (still have generic type parameters).
        // This happens when walking the generic definition's MIR - the call args
        // are still generic. We only want to collect concrete instantiations.
        if !is_fully_monomorphized(self.tcx, resolved) {
            if self.verbose {
                eprintln!(
                    "[collector] Skipping non-monomorphized callee: {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return;
        }

        // Skip intrinsics and other special functions, but allow DropGlue
        // instances through so that drop_in_place calls inside drop shim
        // bodies (e.g. for array/slice element drops) are collected.
        if !matches!(
            resolved.def,
            InstanceKind::Item(_)
                | InstanceKind::Shim(ShimKind::DropGlue(..) | ShimKind::FnPtrAsPtr(..))
        ) {
            return;
        }

        // For DropGlue instances discovered via Call terminators (rather
        // than Drop terminators), route them through the same collection
        // logic as process_drop_place to avoid duplicating the enqueue path.
        if let InstanceKind::Shim(ShimKind::DropGlue(_, Some(_))) = resolved.def {
            let mangled = self.tcx.symbol_name(resolved).name.to_string();
            if self.seen.contains(&mangled) {
                return;
            }
            if !is_fully_monomorphized(self.tcx, resolved) {
                return;
            }
            // Skip provably no-op drop glue; same shared predicate as
            // process_drop_place, for the same lockstep reason.
            if self.drop_glue_is_noop(resolved) {
                return;
            }
            let callee_ctx = DiscoveryCtx {
                root_name: ctx.root_name.clone(),
                root_is_kernel: ctx.root_is_kernel,
                user_span,
            };
            let export_name = sanitize_ptx_name(&mangled);
            if self.verbose {
                eprintln!(
                    "[collector] Discovered drop_in_place (via call) -> {}",
                    export_name
                );
            }
            self.discovery.insert(mangled.clone(), callee_ctx);
            self.seen.insert(mangled);
            self.used_export_names.insert(export_name.clone());
            self.worklist.push_back(CollectedFunction {
                instance: resolved,
                is_kernel: false,
                export_name,
            });
            return;
        }

        // Empty drop glue (DropGlue with None type) has no body to collect.
        if let InstanceKind::Shim(ShimKind::DropGlue(_, None)) = resolved.def {
            return;
        }

        let raw_name = self.tcx.def_path_str(resolved.def_id());
        let crate_name = self.tcx.crate_name(resolved.def_id().krate);
        if self.is_generated_intrinsic_placeholder_or_report_mismatch(
            crate_name.as_str(),
            &raw_name,
            resolved.def_id(),
            call_span,
        ) {
            if self.verbose {
                eprintln!("[collector] Skipping generated intrinsic placeholder: {raw_name}");
            }
            return;
        }

        // Check if this is a device extern declaration (FFI with external LTOIR).
        // These have no MIR body but should be emitted as LLVM `declare` statements.
        if is_device_extern_symbol(&raw_name) {
            self.add_device_extern(resolved.def_id(), &raw_name);
            return;
        }

        // Heap allocation guard (issue #108): every `Vec` / `Box` / `String`
        // allocation funnels into the `__rust_alloc` shim family, which has
        // no MIR body and no device-side implementation. Without this guard
        // the walk silently continues into `alloc::` internals and the user
        // eventually gets a constant-translation error spanned into
        // `alloc/src/boxed.rs`. Fail here instead, at the first point where
        // the allocator is provably reached from device code.
        //
        // Only sysroot functions can be the real allocator entry points. A
        // user is free to define their own fn named `__rust_alloc` (the
        // name is not reserved), and that compiles for the device like any
        // other function, so local definitions must not trip the guard.
        if !resolved.def_id().is_local()
            && let Some(shim) = rust_alloc_shim_name(&raw_name)
        {
            self.report_heap_allocation(shim, caller, &callee_ctx);
        }

        if is_ptx_asm_marker_path(&raw_name) {
            if self.verbose {
                eprintln!("[collector] Skipping inline PTX marker: {raw_name}");
            }
            return;
        }

        if is_unroll_marker_path(&raw_name) {
            if self.verbose {
                eprintln!("[collector] Skipping unroll marker: {raw_name}");
            }
            return;
        }

        if is_launch_metadata_marker_path(&raw_name) {
            if self.verbose {
                eprintln!("[collector] Skipping launch metadata marker: {raw_name}");
            }
            return;
        }

        // Skip functions without MIR bodies (extern intrinsics like cuda_device::threadIdx_x).
        // These are handled specially by the terminator translator in mir-importer
        // which dispatches them to NVVM intrinsic operations.
        //
        // Compiler-built shims (`FnPtrAsPtr`) have no HIR body either, so
        // `is_mir_available` is false for their def_id, but `instance_mir`
        // synthesises their body; do not skip those.
        if !self.tcx.is_mir_available(resolved.def_id())
            && !matches!(resolved.def, InstanceKind::Shim(ShimKind::FnPtrAsPtr(..)))
        {
            if self.verbose {
                eprintln!(
                    "[collector] Skipping extern/intrinsic (no MIR): {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return;
        }

        // Check if it has an unreachable body (intrinsic placeholder)
        if self.is_unreachable_body(resolved.def_id()) {
            // Genuine intrinsic placeholders live in `cuda_device` and are
            // rewritten to NVVM ops by the translator, so skipping them is
            // correct. But a USER function can also collapse to a panic-only
            // body, most commonly when a helper without `#[device]` calls
            // `thread::index_1d()` and gets the host-only panicking stub
            // inlined (issue #76). Silently skipping such a function leaves
            // a dangling call that later fails module verification with
            // "Symbol ... not found". Diagnose it here instead.
            self.check_unreachable_callee(resolved, call_span, caller, &callee_ctx);
            if self.verbose {
                eprintln!(
                    "[collector] Skipping intrinsic (unreachable body): {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return;
        }

        // Use FQDN so the export name matches what the MIR translator
        // sees via `CrateDef::name()` on the call side.
        let name = self.fqdn(resolved);
        let export_name = self.compute_export_name(&name, resolved);

        if self.verbose {
            eprintln!("[collector] Discovered callee: {} -> {}", name, export_name);
        }

        self.discovery.insert(mangled.clone(), callee_ctx);
        self.seen.insert(mangled);
        self.worklist.push_back(CollectedFunction {
            instance: resolved,
            is_kernel: false,
            export_name,
        });
    }

    fn enqueue_callable_trait_receiver_body(
        &mut self,
        ty: Ty<'tcx>,
        call_span: Span,
        caller: &CollectedFunction<'tcx>,
        ctx: &DiscoveryCtx,
    ) {
        let typing_env = TypingEnv::fully_monomorphized();
        let (kind, instance) = match ty.kind() {
            TyKind::Closure(closure_def_id, closure_substs) => {
                let Some(instance) =
                    Instance::try_resolve(self.tcx, typing_env, *closure_def_id, closure_substs)
                        .ok()
                        .flatten()
                else {
                    return;
                };
                ("closure", instance)
            }
            TyKind::FnDef(fn_def_id, fn_args) => {
                // This ty comes from fully-monomorphized MIR, so the FnDef
                // binder (late-bound lifetimes only) has been instantiated;
                // assert that instead of `skip_binder()`, matching
                // rustc_monomorphize::collector's
                // `args.no_bound_vars().unwrap()` discipline.
                let fn_args = fn_args
                    .no_bound_vars()
                    .expect("FnDef args in monomorphized MIR carry no late-bound vars");
                let Some(instance) =
                    Instance::try_resolve(self.tcx, typing_env, *fn_def_id, fn_args)
                        .ok()
                        .flatten()
                else {
                    return;
                };
                ("function item", instance)
            }
            _ => return,
        };

        let target = self.tcx.def_path_str(instance.def_id());
        let crate_name = self.tcx.crate_name(instance.def_id().krate);
        if self.is_generated_intrinsic_placeholder_or_report_mismatch(
            crate_name.as_str(),
            &target,
            instance.def_id(),
            call_span,
        ) {
            self.tcx
                .dcx()
                .struct_span_fatal(
                    call_span,
                    format!(
                        "`{target}` cannot be used as a function item in device code because generated CUDA intrinsics require direct call-site lowering"
                    ),
                )
                .with_help(
                    "wrap the intrinsic call in a local `#[device]` function and pass that wrapper instead",
                )
                .emit()
        }

        match self.should_collect_from_crate(instance.def_id()) {
            CollectDecision::Collect => {}
            CollectDecision::SkipIntentional => {
                let target = self.tcx.def_path_str(instance.def_id());
                self.tcx
                    .dcx()
                    .struct_span_fatal(
                        call_span,
                        format!(
                            "`{target}` cannot be used as a function item in device code because it requires special call-site lowering"
                        ),
                    )
                    .with_help(
                        "wrap the call in a local `#[device]` function and pass that wrapper instead",
                    )
                    .emit()
            }
            CollectDecision::Forbidden {
                crate_name,
                fn_path,
            } => self
                .tcx
                .dcx()
                .struct_span_fatal(
                    call_span,
                    format!(
                        "device code cannot call function item `{fn_path}` from forbidden crate `{crate_name}`"
                    ),
                )
                .with_note(format!(
                    "the call is reachable from `{}`",
                    ctx.root_name
                ))
                .emit(),
        }

        let mangled = self.tcx.symbol_name(instance).name.to_string();
        if self.seen.contains(&mangled) {
            return;
        }
        if !is_fully_monomorphized(self.tcx, instance) {
            return;
        }
        if !matches!(instance.def, InstanceKind::Item(_)) {
            return;
        }
        if !self.tcx.is_mir_available(instance.def_id()) {
            return;
        }
        if self.is_unreachable_body(instance.def_id()) {
            self.check_unreachable_callee(instance, call_span, caller, ctx);
            return;
        }

        let name = self.fqdn(instance);
        let export_name = self.compute_export_name(&name, instance);

        if self.verbose {
            eprintln!(
                "[collector] Discovered {kind} body (via trait call): {name} -> {export_name}"
            );
        }

        self.discovery.insert(mangled.clone(), ctx.clone());
        self.seen.insert(mangled);
        self.worklist.push_back(CollectedFunction {
            instance,
            is_kernel: false,
            export_name,
        });
    }

    /// Determines if a function from a given crate should be collected.
    ///
    /// Returns a [`CollectDecision`] indicating:
    /// - `Collect`: Function should be collected for device compilation
    /// - `SkipIntentional`: Skip silently (core::fmt, core::panicking)
    /// - `Forbidden`: Error - function is from a forbidden crate (std, etc.)
    ///
    /// ## Kernel Entry Points (Cross-Crate Support)
    ///
    /// Kernels (detected via `is_kernel_symbol` from `reserved-oxide-symbols`)
    /// are allowed from ANY crate. This enables library crates to export
    /// generic kernels that get monomorphized when used in an application.
    ///
    /// ## Allowed Crates (for non-kernel callees)
    ///
    /// - Local crate: The user's kernel code
    /// - `cuda_device`: Our GPU intrinsics library
    /// - `core`: Standard library core (iterators, Option, etc.)
    /// - `alloc`: Heap allocation (if user has configured a GPU allocator)
    /// - Any crate reachable from a kernel (transitive closure)
    ///
    /// ## Intentionally Skipped
    ///
    /// - `core::fmt::*`: Format trait machinery uses function pointers
    /// - `core::panicking::*`: Panic handling (handled by panic=abort)
    ///
    /// ## Forbidden (Error)
    ///
    /// - `std`: OS, I/O, threads - can't run on GPU. Two exceptions: the
    ///   inherent float wrappers (`std::f32::<impl f32>::*`, collected) and
    ///   the `std::sys::cmath` shims (skipped, lowered to libdevice).
    fn should_collect_from_crate(&self, def_id: DefId) -> CollectDecision {
        // Always collect from local crate
        if def_id.krate == LOCAL_CRATE {
            return CollectDecision::Collect;
        }

        let crate_name = self.tcx.crate_name(def_id.krate);
        let name_str = crate_name.as_str();

        // The `libm` crate (glam's `nostd-libm` float backend) is intercepted at
        // every call site by mir-importer's float-math dispatch and lowered to
        // libdevice intrinsics (`__nv_sqrtf`, `__nv_sinf`, ...). Its bodies must
        // therefore NOT be collected: translating libm's generic software-float
        // implementations (e.g. `libm::math::generic::sqrt::sqrt_round`) would
        // both be wasted work and trip importer gaps. Skip the whole crate; any
        // libm function we don't yet intercept will surface as a missing symbol,
        // signalling that `from_libm_path` needs another entry.
        if name_str == "libm" {
            return CollectDecision::SkipIntentional;
        }

        // Check if this is a kernel entry point. Kernels can come from ANY
        // crate — this enables library crates to export generic kernels that
        // get monomorphized when used in an application.
        // Unnamed items (closures) have no item name — `item_name` ICEs on
        // them — and can never be kernel entry points, so fall through to
        // the crate-based decision for them.
        if let Some(fn_name) = self.tcx.opt_item_name(def_id)
            && is_kernel_symbol(fn_name.as_str())
        {
            return CollectDecision::Collect;
        }

        // Forbidden crate: std (OS, I/O, threads) - absolutely can't run on GPU
        if name_str == "std" {
            let fn_path = self.tcx.def_path_str(def_id);
            // The `#[inline]` float wrappers (`f32::atan`, `f64::sinh`, ...)
            // are pure and bottom out in a `core` intrinsic or a cmath shim.
            // They only reach the guard when the MIR inliner is off, so
            // collect them as ordinary device functions.
            if is_std_float_inherent_method(&fn_path) {
                return CollectDecision::Collect;
            }
            // A handful of `std::sys::cmath::*` libm shims are intercepted
            // by mir-importer's float-math intrinsic dispatch and lowered
            // directly to libdevice (`__nv_atan2f` etc.). They never enter
            // device codegen, so silently skip them here instead of tripping
            // the std-crate guard. This is what makes `f32::atan2`,
            // `f32::atan`, and the f64 counterparts usable from device code,
            // whether the kernel calls the shim directly (MIR-opt inlined the
            // wrapper) or through the collected wrapper above.
            if is_intrinsic_lowered_cmath_shim(&fn_path) {
                return CollectDecision::SkipIntentional;
            }
            return CollectDecision::Forbidden {
                crate_name: name_str.to_string(),
                fn_path,
            };
        }

        // Allowed external crates (no_std compatible)
        // - core: iterators, Option, Result, traits, etc.
        // - alloc: Vec, Box, String (if user has GPU allocator)
        // - cuda_device: our GPU intrinsics
        let allowed = matches!(name_str, "core" | "alloc" | "cuda_device" | "cuda-device");

        if !allowed {
            // For other external crates, we allow them if they're reachable from a kernel.
            // This enables cross-crate device functions. The key safety is:
            // 1. std is explicitly forbidden above
            // 2. Any crate that uses std won't compile to PTX anyway (missing symbols)
            // 3. User gets a clear link error if they try to use incompatible code
            //
            // This permissive approach enables:
            // - Library crates with device helper functions
            // - Math libraries (e.g., libm, num-traits)
            // - Custom device abstractions
            if self.verbose {
                let fn_path = self.tcx.def_path_str(def_id);
                eprintln!(
                    "[collector] Allowing function from external crate '{}': {}",
                    name_str, fn_path
                );
            }
            return CollectDecision::Collect;
        }

        // Filter out problematic modules (intentional skip, not an error)
        let path = self.tcx.def_path_str(def_id);

        // Skip formatting and panic machinery (uses FnPtr types we can't translate)
        if path.contains("::fmt::") || path.contains("::panicking::") {
            return CollectDecision::SkipIntentional;
        }

        // Skip precondition_check functions - these are UB check assertions that use
        // string types for panic messages. Since our queries return false for
        // RuntimeChecks(UbChecks), these functions are never actually called at runtime.
        // Example: core::num::<impl usize>::unchecked_sub::precondition_check
        if path.contains("precondition_check") {
            return CollectDecision::SkipIntentional;
        }

        // NOTE: We no longer skip arithmetic trait methods (Mul::mul, Add::add, etc.)
        // These become device functions with call overhead, but that's a separate
        // optimization issue (forced-inline on monomorphic small bodies).
        //
        // Legacy mangled names (from prebuilt sysroot) contain $ characters which are
        // invalid PTX identifiers. We sanitize these in compute_export_name().

        CollectDecision::Collect
    }

    /// Computes the export name for a function.
    ///
    /// `name` must be the FQDN (from [`Self::fqdn`]) so that non-generic export names
    /// match what `CrateDef::name()` returns on the call side. Both sides feed
    /// the FQDN through pliron's `Legaliser`, which replaces every
    /// non-`[A-Za-z0-9]` character with `_`. We return the *raw* FQDN here so
    /// that the call-side legaliser and the export-side legaliser (in
    /// `body.rs::translate_body`) see the same input string and resolve to the
    /// same canonical identifier. Pre-legalising here would alias to a
    /// different input string and trigger spurious dedupe suffixes.
    ///
    /// For resolved instances that still carry generic args, or for raw-name
    /// collisions, we fall back to the mangled symbol name since the MIR
    /// translator uses `Instance::mangled_name()` for the matching call sites.
    /// Concrete impl paths that merely contain characters like `<`, `>`, and
    /// `::` remain raw FQDNs and are legalized later on both sides.
    fn compute_export_name(&mut self, name: &str, instance: Instance<'tcx>) -> String {
        // CRITICAL: If the instance has generic args, we MUST use mangled name.
        // The MIR translator uses mangled names for generic function calls
        // (see terminator/mod.rs::extract_func_info), so we must match that here.
        // Without this, the call site uses "_RINv...mapf..." but we export as "map".
        let has_generic_args = !instance.args.is_empty();

        let simple_name = name.to_string();

        if has_generic_args || self.used_export_names.contains(&simple_name) {
            // Use mangled symbol name to avoid conflicts.
            // This handles generics (e.g., ptr::add::<i32>) and name collisions.
            let mangled = self.tcx.symbol_name(instance).name.to_string();

            // Sanitize for PTX: replace $ with _ (legacy mangling uses $LT$, $GT$, etc.)
            let sanitized = sanitize_ptx_name(&mangled);

            self.used_export_names.insert(sanitized.clone());
            sanitized
        } else {
            self.used_export_names.insert(simple_name.clone());
            simple_name
        }
    }

    /// Checks if a function body is just `unreachable!()` (intrinsic placeholder).
    ///
    /// cuda_device intrinsics have placeholder bodies that panic when called on host:
    ///
    /// ```rust,ignore
    /// pub fn threadIdx_x() -> u32 {
    ///     unreachable!("threadIdx_x called outside CUDA kernel context")
    /// }
    /// ```
    ///
    /// These are translated specially to PTX intrinsics by the MIR translator.
    /// We skip collecting them because their panic bodies would pull in `FnPtr`
    /// types we can't handle.
    fn is_unreachable_body(&self, def_id: DefId) -> bool {
        if !self.tcx.is_mir_available(def_id) {
            return false;
        }

        let mir = self.tcx.optimized_mir(def_id);

        // Quick check: intrinsic bodies are very small (1-2 blocks)
        if mir.basic_blocks.len() > 2 {
            return false;
        }

        // Check for panic calls
        for bb_data in mir.basic_blocks.iter() {
            let Some(ref terminator) = bb_data.terminator else {
                continue;
            };
            match &terminator.kind {
                TerminatorKind::Call { func, .. } => {
                    if let Some(callee_def_id) = self.get_call_def_id(func) {
                        let path = self.tcx.def_path_str(callee_def_id);
                        // Match panic functions from both core (no_std) and std:
                        // - core::panicking::* (no_std mode)
                        // - std::rt::panic_fmt (std mode - unreachable!() expands to this)
                        if is_panic_entry_path(&path) {
                            return true;
                        }
                    }
                }
                TerminatorKind::Unreachable => {}
                _ => return false,
            }
        }

        false
    }

    /// Recognize a generated intrinsic placeholder or reject a mismatched raw
    /// intrinsic crate before its panic-only host body enters collection.
    fn is_generated_intrinsic_placeholder_or_report_mismatch(
        &self,
        crate_name: &str,
        displayed_path: &str,
        def_id: DefId,
        call_span: Span,
    ) -> bool {
        if crate::generated_intrinsics::is_generated_intrinsic_crate(crate_name) {
            // `def_path_str` intentionally prefers a public re-export path.
            // Intrinsic ABI matching instead uses rustc's underlying DefPath,
            // whose parent chain is stable and still contains the hidden ABI
            // namespace and opaque per-intrinsic ID.
            let canonical_path = format!(
                "{}{}",
                crate_name,
                self.tcx.def_path(def_id).to_string_no_crate_verbose()
            );
            if !crate::generated_intrinsics::is_generated_intrinsic_canonical_path(&canonical_path)
            {
                self.report_generated_intrinsic_abi_mismatch(&canonical_path, call_span);
            }
            return true;
        }

        crate::generated_intrinsics::is_generated_intrinsic_placeholder(crate_name, displayed_path)
    }

    /// Emit a focused error when the raw declaration crate and compiler were
    /// generated from different intrinsic ABI versions or identity tables.
    fn report_generated_intrinsic_abi_mismatch(&self, path: &str, call_span: Span) -> ! {
        let canonical_paths =
            crate::generated_intrinsics::GENERATED_INTRINSIC_CANONICAL_PATHS.join(", ");
        let public_paths = crate::generated_intrinsics::GENERATED_INTRINSIC_PUBLIC_PATHS.join(", ");
        self.tcx
            .dcx()
            .struct_span_fatal(
                call_span,
                format!(
                    "cuda-intrinsics ABI mismatch: `{path}` is not recognized by this cuda-oxide compiler"
                ),
            )
            .with_note(format!(
                "this compiler supports generated intrinsic ABI v{} in hidden namespace `{}`",
                crate::generated_intrinsics::GENERATED_INTRINSIC_ABI,
                crate::generated_intrinsics::GENERATED_INTRINSIC_ABI_NAMESPACE,
            ))
            .with_note(format!("recognized canonical path(s): {canonical_paths}"))
            .with_note(format!("supported public source path(s): {public_paths}"))
            .with_help(
                "use the cuda-intrinsics crate generated by this cuda-oxide revision, then rebuild the device crate",
            )
            .emit()
    }

    /// Emits the heap-allocation diagnostic (issue #108) and aborts.
    ///
    /// `shim` is the allocator entry point that was reached, `caller` is
    /// the collected function whose body contains the call, and `ctx`
    /// carries the originating root and the nearest user-code span.
    fn report_heap_allocation(
        &self,
        shim: &str,
        caller: &CollectedFunction<'tcx>,
        ctx: &DiscoveryCtx,
    ) -> ! {
        let caller_path = self.tcx.def_path_str(caller.instance.def_id());
        let root_kind = ctx.root_kind();
        self.tcx
            .dcx()
            .struct_span_fatal(
                ctx.user_span,
                "heap allocation is not supported in kernels (no device allocator); \
                 use fixed-size arrays or SharedArray",
            )
            .with_note(format!(
                "device code starting at {root_kind} `{}` reaches the Rust allocator \
                 entry point `{shim}` through `{caller_path}`",
                ctx.root_name
            ))
            .with_note(
                "`Vec`, `Box`, `String`, and everything else that allocates relies on a \
                 global heap allocator, which does not exist on the GPU",
            )
            .with_help(
                "store the data in a fixed-size array `[T; N]` instead, or in a \
                 `SharedArray` when the scratch space should be shared by the thread block",
            )
            .emit()
    }

    /// Refuses a device-reachable function that only makes sense on the host
    /// CPU, at the user-code site that reached it.
    ///
    /// ```text
    /// #[kernel] fn k(..) ─► helper ─► _mm256_add_ps   #[target_feature(enable = "avx2")]  signal 1
    ///                    ─► _rdtsc ─► rdtsc (foreign) ::core_arch::x86::rdtsc::rdtsc      signal 2
    /// ```
    ///
    /// - Kernel MIR comes from the host-target session, so rustc has already
    ///   accepted these for the host. None has a PTX lowering; without this
    ///   guard they fail deep in the translator, or lower to something the
    ///   author did not mean.
    /// - Signal 1: `#[target_feature]` on the definition. The MIR inliner never
    ///   inlines a callee whose feature set differs from the caller's, so the
    ///   call survives to here as a call.
    /// - Signal 2: a definition under core's `core_arch::<arch>` for a
    ///   non-`nvptx` arch, see [`is_host_cpu_arch_def_path`]. Checked only for
    ///   items in `core`, after the cached attribute query, so the common case
    ///   costs one query and one crate compare.
    /// - Called from the worklist pop (every function with a body, including
    ///   trait-dispatched impls and closures) and at the top of
    ///   `process_call_operand` (every call edge, before any skip), which is
    ///   the only place that sees body-less callees such as inlined foreign
    ///   declarations.
    ///
    /// `via` is the collected function whose body contains the call, when it
    /// is not the root itself.
    fn check_host_cpu_only(
        &self,
        def_id: DefId,
        span: Span,
        ctx: &DiscoveryCtx,
        via: Option<DefId>,
    ) {
        use rustc_middle::middle::codegen_fn_attrs::TargetFeatureKind;

        // Report the features the author wrote (`avx2`), not the ones rustc
        // derives from them (`avx`, `sse4.2`, ...). `Forced` is the unsafe
        // `force_target_feature` spelling and is just as host-only.
        //
        // Closures are exempt: rustc copies the enclosing function's features
        // onto them so their bodies may call its intrinsics, but the closure
        // itself is ordinary code. A portable closure launched from an avx2
        // host function must keep compiling; anything host-only inside its
        // body is caught when that body is walked.
        let features: Vec<String> = if self.tcx.is_closure_like(def_id) {
            Vec::new()
        } else {
            self.tcx
                .codegen_fn_attrs(def_id)
                .target_features
                .iter()
                .filter(|f| {
                    matches!(
                        f.kind,
                        TargetFeatureKind::Enabled | TargetFeatureKind::Forced
                    )
                })
                .map(|f| f.name.to_string())
                .collect()
        };

        let in_core = self.tcx.crate_name(def_id.krate).as_str() == "core";
        let def_path = in_core.then(|| self.tcx.def_path(def_id).to_string_no_crate_verbose());
        let is_arch_intrinsic = def_path.as_deref().is_some_and(is_host_cpu_arch_def_path);

        if features.is_empty() && !is_arch_intrinsic {
            return;
        }

        let fn_path = self.tcx.def_path_str(def_id);
        let message = if !features.is_empty() {
            format!(
                "`{fn_path}` requires host CPU target features (`{}`) and cannot run on the GPU",
                features.join("`, `")
            )
        } else {
            format!(
                "`{fn_path}` is a `core::arch` intrinsic for the host CPU (`{}`) and cannot run on the GPU",
                self.tcx.sess.target.arch
            )
        };

        let root_kind = ctx.root_kind();
        let reach = match via {
            Some(caller) => format!(
                "device code starting at {root_kind} `{}` reaches it through `{}`",
                ctx.root_name,
                self.tcx.def_path_str(caller)
            ),
            None => format!(
                "device code starting at {root_kind} `{}` reaches it",
                ctx.root_name
            ),
        };

        let mut diag = self
            .tcx
            .dcx()
            .struct_span_fatal(span, message)
            .with_note(reach)
            .with_note(
                "cuda-oxide compiles kernels from the host target's MIR, so code selected by \
                 `cfg(target_arch = ...)`, `core::arch` intrinsics, and `#[target_feature]` \
                 functions for the host CPU are visible to device code; none of them has a \
                 PTX lowering",
            );
        // Portable code can land here without the user writing anything
        // host-specific: `core::hint::spin_loop` expands to `_mm_pause` on
        // x86, `<[u8]>::is_ascii` picks an SSE2 path, `half` picks its f16c
        // path under `-C target-cpu=native`. Say so, instead of telling the
        // user to remove host-CPU code they never wrote.
        if is_spin_loop_expansion(def_path.as_deref().unwrap_or("")) {
            diag = diag.with_note(
                "this is what `core::hint::spin_loop` expands to on the host; it has no GPU \
                 lowering yet, so wait with `cuda_device::barrier::nanosleep` or a plain loop \
                 instead",
            );
        } else if !def_id.is_local() && via.is_some_and(|caller| !caller.is_local()) {
            // Reached through a non-local function, so the user did not call
            // it directly: a dependency or `core` picked the host path.
            diag = diag.with_note(format!(
                "the host build selected this code inside crate `{}` through `cfg(target_arch)` \
                 / `cfg(target_feature)` (widened by `-C target-cpu=native` if set); the portable \
                 alternative cannot be re-selected from device code",
                self.tcx.crate_name(def_id.krate)
            ));
        }
        diag.with_help(
            "keep host-CPU code out of `#[kernel]` and `#[device]` functions and the \
             helpers they call; when it came from `core` or a dependency, use a `cuda_device` \
             equivalent or a plain loop instead",
        )
        .emit()
    }

    /// Diagnoses a callee whose entire body is panic machinery (issue #76).
    ///
    /// Reached from the `is_unreachable_body` skip in
    /// [`Self::process_call_operand`]. Genuine intrinsic placeholders (the
    /// `cuda_device` stubs the translator rewrites by name) must keep
    /// being skipped silently, so this only reports two specific cases:
    ///
    /// 1. The body contains the `thread::index_*` stub panic message.
    ///    That message can only end up in device-reachable code when a
    ///    helper without `#[device]` called the public stub, so the fix
    ///    (annotate the helper) is reported with certainty.
    /// 2. The callee is a user-crate function with a normal return type.
    ///    Skipping it would leave a call to a symbol that is never
    ///    defined, which later fails module verification with an opaque
    ///    "Symbol ... not found" error.
    ///
    /// Functions that are declared diverging (return type `!`) are left
    /// alone: their call sites have no target block, so the translator
    /// already lowers them to LLVM `unreachable` (note: NOT a trap;
    /// the optimizer may delete paths that provably reach it, a known
    /// gap tracked separately).
    fn check_unreachable_callee(
        &self,
        resolved: Instance<'tcx>,
        call_span: Span,
        caller: &CollectedFunction<'tcx>,
        ctx: &DiscoveryCtx,
    ) {
        let def_id = resolved.def_id();
        let mir = self.tcx.optimized_mir(def_id);

        // Look for the index-stub panic message in the callee's body.
        let mut scan = StrConstScan {
            tcx: self.tcx,
            found: Vec::new(),
        };
        scan.visit_body(mir);
        let marker = scan
            .found
            .iter()
            .find(|(_, _, text)| text.contains(MISSING_DEVICE_STUB_MARKER));

        let caller_path = self.tcx.def_path_str(caller.instance.def_id());
        let callee_path = self.tcx.def_path_str(def_id);
        let user_call_span = call_span;
        let root_kind = ctx.root_kind();

        if let Some((_, _, text)) = marker {
            let stub = stub_name_from_marker_message(text);
            let crate_name = self.tcx.crate_name(def_id.krate);
            let is_cuda_device = matches!(crate_name.as_str(), "cuda_device" | "cuda-device");
            let message = format!(
                "`{stub}` only works inside `#[kernel]` / `#[device]` functions; \
                 here it resolves to a host-only stub that panics instead of \
                 reading the thread index"
            );
            if is_cuda_device {
                // The collected caller invokes the public stub directly:
                // the macros never rewrote this call site, so the caller
                // itself is the function missing the annotation.
                self.tcx
                    .dcx()
                    .struct_span_fatal(user_call_span, message)
                    .with_note(format!(
                        "`{caller_path}` is not annotated, so the macro rewrite that \
                         turns `{stub}` into the device intrinsic never ran on this call"
                    ))
                    .with_help(format!(
                        "annotate `{caller_path}` with `#[device]`, or compute the index \
                         in the kernel and pass it in as a parameter"
                    ))
                    .emit()
            } else {
                // The stub body was inlined into an un-annotated helper;
                // point at the helper definition and the device call site.
                self.tcx
                    .dcx()
                    .struct_span_fatal(self.tcx.def_span(def_id), message)
                    .with_span_note(
                        user_call_span,
                        format!(
                            "`{callee_path}` is called from device code here \
                             (reached from {root_kind} `{}`)",
                            ctx.root_name
                        ),
                    )
                    .with_note(format!(
                        "`{callee_path}` is not annotated, so the macro rewrite that \
                         turns `{stub}` into the device intrinsic never ran inside it"
                    ))
                    .with_help(format!(
                        "annotate `{callee_path}` with `#[device]`, or compute the index \
                         in the kernel and pass it in as a parameter"
                    ))
                    .emit()
            }
        }

        // No stub marker: leave non-local functions to the existing silent
        // skip (this is what keeps the real `cuda_device` intrinsic
        // placeholders and `core`'s cold panic wrappers working), and leave
        // declared-diverging functions to the translator's `unreachable` lowering.
        if !def_id.is_local() || mir.return_ty().is_never() {
            return;
        }

        self.tcx
            .dcx()
            .struct_span_fatal(
                self.tcx.def_span(def_id),
                format!(
                    "`{callee_path}` is called from device code but its body is \
                     nothing but a panic, which cannot be compiled for the GPU"
                ),
            )
            .with_span_note(
                user_call_span,
                format!(
                    "called from device code here (reached from {root_kind} `{}`)",
                    ctx.root_name
                ),
            )
            .with_note(
                "likely causes: a function used in device code without `#[device]` \
                 (its body then resolves to a host-only stub that panics), or a \
                 function that unconditionally panics",
            )
            .with_help(
                "annotate device helpers with `#[device]`; if the panic is \
                 intentional, declare the function as diverging (`-> !`) so the \
                 call lowers to LLVM `unreachable`",
            )
            .emit()
    }

    /// Diagnoses a missing `#[device]` annotation found through the panic
    /// machinery of a collected body (issue #76).
    ///
    /// Called from [`Self::collect`] for every function that is about to be
    /// translated. A basic block ending in a call into `core::panicking`
    /// is a panic path; the string constants it materializes are the panic
    /// message (or the pieces of a `format_args!` template).
    ///
    /// A *genuine* panic is not this function's business: the mir-importer
    /// drops the diverging call, skips the dead message-building statements
    /// and traps instead (`mir-importer`'s
    /// `translator::block::translate_block`), so `panic!`, `unwrap`,
    /// `expect` and every core panic reached through them compile.
    ///
    /// One message is still a hard error, because trapping it would bury a
    /// real bug: the `thread::index_*` stub marker. It can only appear in
    /// device-reachable MIR when a helper without `#[device]` called the
    /// public stub and got its host-only panicking body inlined. Lowering
    /// that to a trap would turn "you forgot an annotation" into a thread
    /// that silently aborts the kernel, so it is reported here instead.
    fn check_panic_machinery(
        &self,
        mir: &rustc_middle::mir::Body<'tcx>,
        func: &CollectedFunction<'tcx>,
        ctx: &DiscoveryCtx,
        reachable: &DenseBitSet<BasicBlock>,
    ) {
        for (bb, bb_data) in mir.basic_blocks.iter_enumerated() {
            // A panic in a mono-unreachable block never runs on device.
            if !reachable.contains(bb) {
                continue;
            }
            let Some(term) = &bb_data.terminator else {
                continue;
            };
            let TerminatorKind::Call { func: callee, .. } = &term.kind else {
                continue;
            };
            let Some(callee_did) = self.get_call_def_id(callee) else {
                continue;
            };
            if !is_panic_entry_path(&self.tcx.def_path_str(callee_did)) {
                continue;
            }

            // Collect the string constants feeding this panic block.
            let mut scan = StrConstScan {
                tcx: self.tcx,
                found: Vec::new(),
            };
            scan.visit_basic_block_data(bb, bb_data);

            // Stub marker: the `thread::index_*` stub was inlined into this
            // body, so a function on the way here is missing `#[device]`.
            let stub_marker = scan
                .found
                .iter()
                .find(|(_, _, text)| text.contains(MISSING_DEVICE_STUB_MARKER));

            // Any other panic message is fine: the message-building
            // statements are dead once the diverging call is dropped, so the
            // importer skips them and the path lowers to a device trap. Note
            // it under `--verbose` so the dropped message stays traceable.
            let Some((_, _, marker_text)) = stub_marker else {
                if self.verbose
                    && let Some((_, _, text)) = scan.found.first()
                {
                    eprintln!(
                        "[collector] Panic message dropped, path traps on device: {text:?} \
                         (in `{}`)",
                        self.tcx.def_path_str(func.instance.def_id())
                    );
                }
                continue;
            };

            let func_path = self.tcx.def_path_str(func.instance.def_id());
            let root_kind = ctx.root_kind();
            // When the panic sits directly in the root's body, naming the
            // (mangled) containing function adds nothing; name the root once.
            let location_note = if func.export_name == ctx.root_name {
                format!(
                    "the panic is inside the body of {root_kind} `{}`",
                    ctx.root_name
                )
            } else {
                format!(
                    "the panic lives in `{func_path}`, reached from {root_kind} `{}`",
                    ctx.root_name
                )
            };

            // Best user-facing span: the panic call site, mapped back
            // through MIR inlining and macro expansion to user code.
            let user_span = outermost_user_span(mir, term.source_info);

            let stub = stub_name_from_marker_message(marker_text);
            self.tcx
                .dcx()
                .struct_span_fatal(
                    user_span,
                    format!(
                        "`{stub}` only works inside `#[kernel]` / `#[device]` \
                         functions; here it resolves to a host-only stub that \
                         panics instead of reading the thread index"
                    ),
                )
                .with_note(location_note)
                .with_help(format!(
                    "annotate the helper that calls `{stub}` with `#[device]`, or \
                     compute the index in the kernel and pass it in as a parameter"
                ))
                .emit()
        }
    }

    /// Extracts the DefId from a call operand.
    fn get_call_def_id(&self, func: &rustc_middle::mir::Operand<'tcx>) -> Option<DefId> {
        use rustc_middle::mir::Operand;

        let Operand::Constant(const_op) = func else {
            return None;
        };

        let ty = const_op.const_.ty();
        if let TyKind::FnDef(def_id, _) = ty.kind() {
            Some(*def_id)
        } else {
            None
        }
    }
}

/// Dumps MIR info for collected device functions.
///
/// This is useful for debugging to see what was collected and verify
/// the MIR statistics (basic blocks, locals, args) look reasonable.
pub fn dump_device_mir_info<'tcx>(tcx: TyCtxt<'tcx>, functions: &[CollectedFunction<'tcx>]) {
    eprintln!("\n=== Device Functions MIR Info ===");
    for func in functions {
        let def_id = func.instance.def_id();
        eprintln!(
            "\n{} [{}]:",
            func.export_name,
            if func.is_kernel { "kernel" } else { "device" }
        );

        if tcx.is_mir_available(def_id) {
            let mir = tcx.instance_mir(func.instance.def);
            eprintln!("  - {} basic blocks", mir.basic_blocks.len());
            eprintln!("  - {} local variables", mir.local_decls.len());
            eprintln!("  - {} args", mir.arg_count);

            // Show return type
            let ret_ty = mir.local_decls[rustc_middle::mir::RETURN_PLACE].ty;
            eprintln!("  - returns: {:?}", ret_ty);
        } else {
            eprintln!("  - MIR not available");
        }
    }
    eprintln!("=================================\n");
}

#[cfg(test)]
mod tests {
    use super::{
        device_runtime_checks_target, is_host_cpu_arch_def_path, is_kernel_entry_def_path,
        is_spin_loop_expansion, is_std_float_inherent_method, truncate_path_for_box,
        unsupported_codegen_protocol_root,
    };
    use reserved_oxide_symbols::{
        DEVICE_PREFIX, KERNEL_PREFIX, LEGACY_DEVICE_PREFIX, LEGACY_KERNEL_PREFIX,
        PTX_MERGE_REQUIRED_PREFIX, is_ptx_merge_required_marker, ptx_merge_required_marker,
    };
    use rustc_middle::mir::BasicBlock;

    #[test]
    fn std_float_wrappers_are_collected_not_forbidden() {
        for ty in ["f16", "f32", "f64", "f128"] {
            assert!(is_std_float_inherent_method(&format!(
                "std::{ty}::<impl {ty}>::atan"
            )));
        }
        assert!(is_std_float_inherent_method("std::f64::<impl f64>::atan2"));
        assert!(is_std_float_inherent_method(
            "std::f32::<impl f32>::sin_cos"
        ));
        // The shim a wrapper calls is skipped rather than collected, core's
        // float methods never reach the std guard, and the rest of std stays
        // forbidden.
        assert!(!is_std_float_inherent_method("std::sys::cmath::atanf"));
        assert!(!is_std_float_inherent_method("core::f32::<impl f32>::abs"));
        assert!(!is_std_float_inherent_method("std::f32::consts::PI"));
        assert!(!is_std_float_inherent_method("std::io::stdio::_print"));
    }

    #[test]
    fn host_cpu_arch_definition_paths_are_recognised() {
        // Definition paths as `DefPath::to_string_no_crate_verbose` prints
        // them: what `_rdtsc` inlines into, an AVX intrinsic, a NEON one.
        assert!(is_host_cpu_arch_def_path("::core_arch::x86::rdtsc::rdtsc"));
        assert!(is_host_cpu_arch_def_path(
            "::core_arch::x86::avx::_mm256_add_ps"
        ));
        assert!(is_host_cpu_arch_def_path(
            "::core_arch::aarch64::neon::generated::vaddq_f32"
        ));
        assert!(is_host_cpu_arch_def_path(
            "::core_arch::riscv_shared::pause"
        ));
    }

    #[test]
    fn gpu_and_non_arch_definition_paths_are_not_host_cpu_intrinsics() {
        // The GPU's own module, never visible under a host target today but
        // the right answer if it ever is.
        assert!(!is_host_cpu_arch_def_path(
            "::core_arch::nvptx::_syncthreads"
        ));
        // stdarch's private SIMD vector types and macro helpers under core_arch.
        assert!(!is_host_cpu_arch_def_path("::core_arch::simd::i32x4::new"));
        assert!(!is_host_cpu_arch_def_path("::core_arch::macros::foo"));
        // Items outside core_arch, including `core::arch::breakpoint`, which
        // lives in core's `arch` module, not in `core_arch`.
        assert!(!is_host_cpu_arch_def_path("::arch::breakpoint"));
        assert!(!is_host_cpu_arch_def_path("::hint::spin_loop"));
        assert!(!is_host_cpu_arch_def_path("::ptr::read"));
        // Re-export spellings are never passed in; only definition paths are.
        assert!(!is_host_cpu_arch_def_path("core::arch::x86_64::_rdtsc"));
    }

    #[test]
    fn spin_loop_expansions_get_the_extra_note() {
        assert!(is_spin_loop_expansion("::core_arch::x86::sse2::pause"));
        assert!(is_spin_loop_expansion("::core_arch::x86::sse2::_mm_pause"));
        assert!(!is_spin_loop_expansion("::core_arch::x86::rdtsc::rdtsc"));
        assert!(!is_spin_loop_expansion("::hint::spin_loop"));
    }

    #[test]
    fn ptx_merge_marker_matches_only_the_final_path_component() {
        let marker = ptx_merge_required_marker("map");
        assert!(is_ptx_merge_required_marker(&marker));
        assert!(is_ptx_merge_required_marker(&format!(
            "crate_name::kernels::{marker}"
        )));
        assert!(!is_ptx_merge_required_marker(&format!(
            "crate_name::prefix_{marker}"
        )));
        assert!(!is_ptx_merge_required_marker(PTX_MERGE_REQUIRED_PREFIX));
        assert!(!is_ptx_merge_required_marker("crate_name::ordinary_static"));
    }

    #[test]
    fn scoped_cache_protocol_rejects_legacy_local_and_external_roots() {
        assert!(!unsupported_codegen_protocol_root(&format!(
            "local::{KERNEL_PREFIX}map"
        )));
        assert!(!unsupported_codegen_protocol_root(&format!(
            "local::{DEVICE_PREFIX}helper"
        )));
        assert!(unsupported_codegen_protocol_root(&format!(
            "legacy::{LEGACY_KERNEL_PREFIX}map"
        )));
        assert!(unsupported_codegen_protocol_root(&format!(
            "dependency::{LEGACY_KERNEL_PREFIX}generic_kernel"
        )));
        assert!(unsupported_codegen_protocol_root(&format!(
            "legacy::{LEGACY_DEVICE_PREFIX}helper"
        )));
        assert!(unsupported_codegen_protocol_root(&format!(
            "{LEGACY_KERNEL_PREFIX}codegen_v1_map"
        )));
        assert!(unsupported_codegen_protocol_root(&format!(
            "crate::{KERNEL_PREFIX}module::{LEGACY_KERNEL_PREFIX}map"
        )));
        assert!(!unsupported_codegen_protocol_root(
            "ordinary::host_function"
        ));
    }

    #[test]
    fn runtime_checks_select_the_device_false_target() {
        let false_target = BasicBlock::from_usize(1);
        let true_target = BasicBlock::from_usize(2);
        let targets = rustc_middle::mir::SwitchTargets::static_if(0, false_target, true_target);
        assert_eq!(device_runtime_checks_target(&targets), false_target);
    }

    #[test]
    fn kernel_def_paths_are_entry_points() {
        // Bare and fully-qualified kernel names: the marker is the final segment.
        assert!(is_kernel_entry_def_path(&format!("{KERNEL_PREFIX}vecadd")));
        assert!(is_kernel_entry_def_path(&format!(
            "kernels::{KERNEL_PREFIX}vecadd"
        )));
        assert!(is_kernel_entry_def_path(&format!(
            "my_crate::kernels::{KERNEL_PREFIX}vecadd"
        )));
        // Legacy roots still classify as entry points. They must not vanish
        // as ordinary host functions: the scoped-cache handshake rejects
        // them with an explicit diagnostic (see
        // `unsupported_codegen_protocol_root`), and builds outside the
        // protocol continue to compile them.
        assert!(is_kernel_entry_def_path(&format!(
            "legacy::{LEGACY_KERNEL_PREFIX}vecadd"
        )));
    }

    #[test]
    fn items_nested_inside_kernel_bodies_are_not_entry_points() {
        // A named fn defined inside a kernel body: the kernel name is a path
        // prefix, not the final segment. Rooting it would mint a `_TID_`
        // export name (generic case) that no call site references.
        assert!(!is_kernel_entry_def_path(&format!(
            "{KERNEL_PREFIX}softmax::reduce_workspace_max"
        )));
        assert!(!is_kernel_entry_def_path(&format!(
            "my_crate::kernels::{KERNEL_PREFIX}softmax::helper"
        )));
        // Deeper nesting (fn inside a block inside the kernel).
        assert!(!is_kernel_entry_def_path(&format!(
            "{KERNEL_PREFIX}softmax::inner::helper"
        )));
    }

    #[test]
    fn closures_inside_kernel_bodies_are_not_entry_points() {
        assert!(!is_kernel_entry_def_path(&format!(
            "{KERNEL_PREFIX}map::{{closure#0}}"
        )));
        assert!(!is_kernel_entry_def_path(&format!(
            "kernels::{KERNEL_PREFIX}map::{{closure#1}}"
        )));
    }

    #[test]
    fn non_kernel_paths_are_not_entry_points() {
        assert!(!is_kernel_entry_def_path("my_crate::helpers::sum_slice"));
        // The marker must begin the final segment. A substring match here
        // would let ordinary near-miss names masquerade as kernel entries.
        assert!(!is_kernel_entry_def_path(&format!(
            "my_crate::helpers::not_{KERNEL_PREFIX}vecadd"
        )));
        assert!(!is_kernel_entry_def_path(&format!(
            "my_crate::helpers::not_{LEGACY_KERNEL_PREFIX}vecadd"
        )));
        assert!(!is_kernel_entry_def_path(&format!(
            "my_crate::helpers::prefix{KERNEL_PREFIX}vecadd"
        )));
        assert!(!is_kernel_entry_def_path(""));
    }

    /// The box pads this field with `{:<48}`, so the budget is characters.
    ///
    /// The previous code tested `fn_path.len() > 48` (bytes) and then sliced
    /// `&fn_path[..45]` (a byte index). A path whose byte 45 falls inside a
    /// multi-byte character panicked with "byte index 45 is not a char
    /// boundary", which aborted the compiler while it was trying to print the
    /// forbidden-crate error.
    #[test]
    fn truncating_a_path_for_the_box_respects_char_boundaries() {
        // 'é' is two bytes, so every char boundary sits at an even index and
        // byte 45 lands mid-character.
        let path = "é".repeat(60);
        assert!(path.chars().count() > 48, "fixture must exceed the budget");
        assert!(!path.is_char_boundary(45), "fixture must straddle byte 45");

        let shown = truncate_path_for_box(&path, 48);

        assert!(shown.ends_with("..."));
        assert_eq!(shown.chars().count(), 48);
        assert!(path.starts_with(shown.trim_end_matches('.')));
    }

    #[test]
    fn a_path_that_fits_is_left_alone() {
        let path = "core::fmt::Debug::fmt";
        assert_eq!(truncate_path_for_box(path, 48), path);
    }

    /// Bytes are not characters: a path of 30 characters spelled with
    /// multi-byte characters exceeds 48 bytes, and the old byte test truncated
    /// it even though it fits the box.
    #[test]
    fn a_multibyte_path_within_the_char_budget_is_not_truncated() {
        let path = "é".repeat(30);
        assert!(path.len() > 48, "fixture must exceed the byte budget");
        assert_eq!(path.chars().count(), 30);
        assert_eq!(truncate_path_for_box(&path, 48), path);
    }

    #[test]
    fn truncation_never_exceeds_the_width() {
        for count in [49, 60, 200] {
            let ascii = "a".repeat(count);
            assert_eq!(truncate_path_for_box(&ascii, 48).chars().count(), 48);
        }
    }
}
