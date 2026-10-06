/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # Device Code Generation via cuda-oxide Pipeline
//!
//! This module bridges rustc's internal MIR representation to cuda-oxide's
//! existing MIR→PTX pipeline using `rustc_public::rustc_internal` to convert
//! between internal and stable_mir types.
//!
//! ## The Bridge Problem
//!
//! We have two different MIR representations:
//!
//! | API                         | Used By                       | Type                                |
//! |-----------------------------|-------------------------------|-------------------------------------|
//! | `rustc_middle` (internal)   | rustc internals, this backend | `rustc_middle::ty::Instance<'tcx>`  |
//! | `rustc_public` (stable MIR) | mir-importer pipeline         | `rustc_public::mir::mono::Instance` |
//!
//! The cuda-oxide pipeline (mir-importer) was built using `rustc_public` APIs because
//! they're more stable. But as a codegen backend, we receive `rustc_middle` types from
//! rustc. This module bridges between them.
//!
//! ## Bridge Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                         DEVICE CODE GENERATION                                  │
//! │                                                                                 │
//! │   Input: Vec<CollectedFunction<'tcx>>                                           │
//! │          (using rustc_middle::ty::Instance)                                     │
//! │                                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 1: Enter stable_mir Context                                       │   │
//! │   │                                                                         │   │
//! │   │  rustc_internal::run(tcx, || { ... })                                   │   │
//! │   │                                                                         │   │
//! │   │  This sets up the Tables and CompilerCtxt that enable type conversion   │   │
//! │   │  between rustc_middle and rustc_public types.                           │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 2: Convert Instances                                              │   │
//! │   │                                                                         │   │
//! │   │  for each CollectedFunction<'tcx>:                                      │   │
//! │   │      stable_instance = rustc_internal::stable(func.instance)            │   │
//! │   │                                                                         │   │
//! │   │  This converts:                                                         │   │
//! │   │    rustc_middle::ty::Instance<'tcx>                                     │   │
//! │   │         ▼                                                               │   │
//! │   │    rustc_public::mir::mono::Instance                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 3: Run cuda-oxide Pipeline                                        │   │
//! │   │                                                                         │   │
//! │   │  mir_importer::run_pipeline(&stable_functions, &config)                 │   │
//! │   │                                                                         │   │
//! │   │  Pipeline stages:                                                       │   │
//! │   │    1. Rust MIR → `dialect-mir` (alloca form)                            │   │
//! │   │    2. `dialect-mir` → `dialect-mir` (mem2reg → SSA)                     │   │
//! │   │    3. Apply annotated loop unrolling                                    │   │
//! │   │    4. `dialect-mir` → LLVM dialect (via `mir-lower`)                    │   │
//! │   │    5. LLVM dialect → textual LLVM IR (.ll)                              │   │
//! │   │    6. LLVM IR → PTX via `llc` (.ptx)                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  Output: DeviceCodegenResult                                            │   │
//! │   │                                                                         │   │
//! │   │    - ptx_path: Path to generated .ptx file                              │   │
//! │   │    - ll_path: Path to generated .ll file                                │   │
//! │   │    - target: GPU target (e.g., "sm_80", "sm_90a")                       │   │
//! │   │    - ptx_content: PTX as string, when PTX was generated                 │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Why This Design?
//!
//! We chose to bridge to stable_mir rather than rewrite mir-importer because:
//!
//! 1. **Code reuse**: mir-importer already works and is well-tested
//! 2. **Stability**: rustc_public APIs change less than rustc internals
//! 3. **Simplicity**: ~100 lines of bridge code vs rewriting the pipeline
//! 4. **Maintainability**: Changes to mir-importer automatically work here
//!
//! The cost is one extra type conversion step, but this happens once per function
//! and is negligible compared to actual compilation time.

use crate::collector::{CollectedFunction, DeviceExternDecl};
use llvm_export::ops::{
    DebugInlinedScope, DebugSourcePosition, DebugSourceScope, DebugSourceScopeLocation,
    DebugSourceScopeMap,
};
use rustc_hir::def::DefKind;
use rustc_middle::mir::interpret::{AllocId, GlobalAlloc, Scalar};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{ConstOperand, ConstValue, Location};
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::layout::{LayoutCx, LayoutOf};
use rustc_middle::ty::{EarlyBinder, Instance, InstanceKind, ShimKind, TypingEnv};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_session::config::DebugInfo;
use rustc_span::def_id::DefId;
use rustc_span::{DUMMY_SP, Span, hygiene};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeviceExternTypePosition {
    Parameter,
    Result,
    Pointee,
}

/// Convert a Rust device-extern type to the LLVM type supported at the
/// external function boundary.
///
/// Raw-pointer pointees are preserved recursively. Rustc-proven
/// `#[repr(transparent)]` wrappers are recursively peeled to their ABI-relevant
/// field before classification, so scalar and pointer wrappers preserve the
/// same external ABI as their underlying value. Unsupported C ABI types return
/// an error instead of being treated as an arbitrary pointer.
///
/// Integer types smaller than 32 bits keep their NARROW IR type (`i8`,
/// `i16`, `i1` for `bool`) and carry a `signext`/`zeroext` ABI attribute,
/// exactly like clang's NVPTXABIInfo and rustc's nvptx64 callconv; the NVPTX
/// backend performs the `.param.b32` widening. This keeps the emitted
/// `declare` byte-for-byte compatible with clang/nvcc-compiled LTOIR
/// definitions and lets cuda-oxide's own narrow SSA values flow into the
/// call with no inserted conversions. `f16` is passed as `half` directly
/// since NVPTX has native f16 support.
fn rustc_ty_to_device_extern_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    position: DeviceExternTypePosition,
) -> Result<mir_importer::DeviceExternType, String> {
    use mir_importer::DeviceExternType as E;

    if ty.is_c_void(tcx) {
        return if position == DeviceExternTypePosition::Pointee {
            // LLVM spells a C void pointer as `i8*` in typed-pointer IR.
            Ok(E::Integer(8))
        } else {
            Err("`c_void` is only supported behind a pointer".to_string())
        };
    }

    // For pointer pointees, all integer widths are valid without extension.
    // For by-value parameters/returns, sub-32-bit integers keep their narrow
    // type and gain the sign/zero extension ABI attribute at that width.
    let signed_integer = |bits: u32| {
        if position == DeviceExternTypePosition::Pointee || matches!(bits, 32 | 64) {
            Ok(E::Integer(bits))
        } else if matches!(bits, 8 | 16) {
            // NVPTX ABI: narrow type with signext (clang NVPTXABIInfo shape).
            Ok(E::SignExtInteger(bits))
        } else {
            Err(format!(
                "`i{bits}` is not supported by value in a device extern; use i32/i64 or pass a pointer"
            ))
        }
    };

    let unsigned_integer = |bits: u32| {
        if position == DeviceExternTypePosition::Pointee || matches!(bits, 32 | 64) {
            Ok(E::Integer(bits))
        } else if matches!(bits, 8 | 16) {
            // NVPTX ABI: narrow type with zeroext (clang NVPTXABIInfo shape).
            Ok(E::ZeroExtInteger(bits))
        } else {
            Err(format!(
                "`u{bits}` is not supported by value in a device extern; use u32/u64 or pass a pointer"
            ))
        }
    };

    match ty.kind() {
        TyKind::Int(int_ty) => match int_ty {
            rustc_middle::ty::IntTy::I8 => signed_integer(8),
            rustc_middle::ty::IntTy::I16 => signed_integer(16),
            rustc_middle::ty::IntTy::I32 => signed_integer(32),
            rustc_middle::ty::IntTy::I64 => signed_integer(64),
            rustc_middle::ty::IntTy::I128 => signed_integer(128),
            rustc_middle::ty::IntTy::Isize => signed_integer(64), // nvptx64
        },
        TyKind::Uint(uint_ty) => match uint_ty {
            rustc_middle::ty::UintTy::U8 => unsigned_integer(8),
            rustc_middle::ty::UintTy::U16 => unsigned_integer(16),
            rustc_middle::ty::UintTy::U32 => unsigned_integer(32),
            rustc_middle::ty::UintTy::U64 => unsigned_integer(64),
            rustc_middle::ty::UintTy::U128 => unsigned_integer(128),
            rustc_middle::ty::UintTy::Usize => unsigned_integer(64), // nvptx64
        },
        TyKind::Float(float_ty) => match float_ty {
            // NVPTX supports native f16 (LLVM `half`) in all positions.
            rustc_middle::ty::FloatTy::F16 => Ok(E::Float16),
            rustc_middle::ty::FloatTy::F32 => Ok(E::Float32),
            rustc_middle::ty::FloatTy::F64 => Ok(E::Float64),
            rustc_middle::ty::FloatTy::F128 => {
                Err("f128 device externs are not supported".to_string())
            }
        },
        TyKind::Adt(adt_def, _) if adt_def.repr().transparent() => {
            // Do not infer the transparent field from source syntax here.
            // Rustc's layout engine already knows which field is ABI-relevant,
            // including nested wrappers and 1-ZST marker fields.
            let layout_cx = LayoutCx::new(tcx, TypingEnv::fully_monomorphized());
            let layout = layout_cx.layout_of(ty).map_err(|err| {
                format!(
                    "failed to compute layout for repr(transparent) device-extern type `{ty}`: {err:?}"
                )
            })?;
            let peeled = layout.peel_transparent_wrappers(&layout_cx);

            // A transparent type with no peelable non-1ZST field is not a
            // scalar/pointer wrapper that this device-extern ABI can represent.
            if peeled.ty == ty {
                return Err(format!(
                    "`{ty}` is repr(transparent) but has no ABI-relevant field supported by device externs"
                ));
            }

            rustc_ty_to_device_extern_type(tcx, peeled.ty, position)
        }
        TyKind::RawPtr(pointee, _) | TyKind::Ref(_, pointee, _) => {
            let pointee = if matches!(pointee.kind(), TyKind::Tuple(fields) if fields.is_empty()) {
                // Rust's `*mut ()` is its common spelling for a void pointer.
                E::Integer(8)
            } else {
                rustc_ty_to_device_extern_type(tcx, *pointee, DeviceExternTypePosition::Pointee)?
            };
            Ok(E::pointer_to(pointee, 0))
        }
        TyKind::Array(element, len) if position == DeviceExternTypePosition::Pointee => {
            let len = len.try_to_target_usize(tcx).ok_or_else(|| {
                format!("device-extern array length for `{ty}` is not a concrete constant")
            })?;
            let element =
                rustc_ty_to_device_extern_type(tcx, *element, DeviceExternTypePosition::Pointee)?;
            Ok(E::Array {
                element: Box::new(element),
                len,
            })
        }
        TyKind::Tuple(fields)
            if fields.is_empty() && position == DeviceExternTypePosition::Result =>
        {
            Ok(E::Void)
        }
        TyKind::Bool => {
            if position == DeviceExternTypePosition::Pointee {
                // Behind a pointer, bool is just i8 (Rust's bool is 1 byte).
                Ok(E::Integer(8))
            } else {
                // NVPTX ABI: bool stays i1 with zeroext, matching both
                // clang's `zeroext i1` and cuda-oxide's own i1 SSA values.
                Ok(E::ZeroExtInteger(1))
            }
        }
        // `char` is already a 32-bit Unicode scalar, so it uses a plain i32
        // slot without a signext/zeroext attribute in every position.
        TyKind::Char => unsigned_integer(32),
        TyKind::Never => Err("never-returning device externs are not yet supported".to_string()),
        _ => Err(format!(
            "unsupported device-extern ABI type `{ty}`; use scalar C types or raw pointers to supported scalar/array pointees"
        )),
    }
}

/// Result of device code generation.
///
/// Contains paths to generated artifacts and the payload selected for
/// embedding in the host binary.
///
/// `ptx_path`, `ll_path` and `ptx_content` are written by
/// `generate_device_code` and never read back inside this crate: they record
/// what codegen produced, which is what the module diagram above documents.
/// Nothing links this crate as a library (it is a `dylib` rustc loads through
/// `-Zcodegen-backend`), so `pub` does not make them reachable either. Those
/// three fields carry their own suppressions below; the remaining fields are
/// read in `lib.rs` and stay lint-checked.
pub struct DeviceCodegenResult {
    /// Path to generated PTX assembly file.
    ///
    /// In NVVM IR modes this is the would-be PTX path and may not exist.
    #[expect(
        dead_code,
        reason = "recorded codegen output, kept for future diagnostics"
    )]
    pub ptx_path: PathBuf,
    /// Path to generated LLVM IR file.
    #[expect(
        dead_code,
        reason = "recorded codegen output, kept for future diagnostics"
    )]
    pub ll_path: PathBuf,
    /// GPU target architecture used (e.g., "sm_80", "sm_90a", "sm_100a").
    ///
    /// Auto-detected based on GPU features used, or overridden via
    /// `CUDA_OXIDE_TARGET` environment variable.
    pub target: String,
    /// PTX content as a string, ready for embedding in the host binary.
    ///
    /// NVVM IR / LTOIR flows intentionally skip PTX generation.
    #[expect(
        dead_code,
        reason = "recorded codegen output, kept for future diagnostics"
    )]
    pub ptx_content: Option<String>,
    /// Device artifact payload selected for embedding.
    pub artifact: Option<DeviceCodegenArtifact>,
    /// Whether later compilation stages may contract ordinary floating-point
    /// multiply/add expressions.
    pub allow_fma_contraction: bool,
    /// Debug policy used when exporting the NVVM IR. Later materialization
    /// stages must preserve this policy instead of silently compiling with
    /// their own defaults.
    pub debug_kind: llvm_export::export::DebugKind,
    /// Source launch bounds for kernel entries, keyed by exported kernel name.
    pub kernel_launch_bounds: BTreeMap<String, mir_importer::KernelLaunchBounds>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCodegenArtifactKind {
    Ptx,
    NvvmIr,
    Ltoir,
    Cubin,
}

pub struct DeviceCodegenArtifact {
    pub kind: DeviceCodegenArtifactKind,
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Configuration for device codegen.
///
/// Controls output paths and diagnostic output during compilation.
pub struct DeviceCodegenConfig {
    /// Output directory for generated files (.ll, .ptx).
    pub output_dir: PathBuf,
    /// Base name for output files (e.g., "kernel" → kernel.ll, kernel.ptx).
    pub output_name: String,
    /// Print verbose progress to stderr.
    pub verbose: bool,
    /// Dump raw rustc MIR before translation.
    pub dump_rustc_mir: bool,
    /// Dump the `dialect-mir` module during compilation.
    pub dump_mir_dialect: bool,
    /// Dump the LLVM dialect module during compilation.
    pub dump_llvm_dialect: bool,
}

impl Default for DeviceCodegenConfig {
    fn default() -> Self {
        Self {
            output_dir: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            output_name: "kernel".to_string(),
            verbose: false,
            dump_rustc_mir: false,
            dump_mir_dialect: false,
            dump_llvm_dialect: false,
        }
    }
}

fn build_debug_source_scope_map<'tcx>(
    tcx: TyCtxt<'tcx>,
    func: &CollectedFunction<'tcx>,
) -> DebugSourceScopeMap {
    let mir = tcx.instance_mir(func.instance.def);
    let scopes = mir
        .source_scopes
        .iter_enumerated()
        .map(|(scope, data)| {
            let inlined = data.inlined.map(|(callee, callsite)| {
                let callee = tcx.instantiate_and_normalize_erasing_regions(
                    func.instance.args,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, callee),
                );
                let callsite = hygiene::walk_chain_collapsed(callsite, mir.span);
                DebugInlinedScope {
                    callee_name: rustc_middle::ty::print::with_no_trimmed_paths!(
                        callee.to_string()
                    ),
                    callsite: debug_position_from_span(tcx, callsite),
                }
            });

            DebugSourceScope {
                id: scope.as_u32(),
                parent: data.parent_scope.map(|parent| parent.as_u32()),
                span: debug_position_from_span(tcx, data.span.source_callsite()),
                inlined,
            }
        })
        .collect();

    let mut locations = Vec::new();
    for block in mir.basic_blocks.iter() {
        for stmt in &block.statements {
            if let Some(pos) = debug_position_from_span(tcx, stmt.source_info.span) {
                locations.push(DebugSourceScopeLocation {
                    pos,
                    scope: stmt.source_info.scope.as_u32(),
                });
            }
        }

        let terminator = block.terminator();
        if let Some(pos) = debug_position_from_span(tcx, terminator.source_info.span) {
            locations.push(DebugSourceScopeLocation {
                pos,
                scope: terminator.source_info.scope.as_u32(),
            });
        }
    }
    locations.sort_by(|lhs, rhs| {
        (&lhs.pos.file, lhs.pos.line, lhs.pos.column, lhs.scope).cmp(&(
            &rhs.pos.file,
            rhs.pos.line,
            rhs.pos.column,
            rhs.scope,
        ))
    });
    locations.dedup();

    DebugSourceScopeMap { scopes, locations }
}

fn debug_position_from_span(tcx: TyCtxt<'_>, span: Span) -> Option<DebugSourcePosition> {
    let (file, line, column, _, _) = tcx.sess.source_map().span_to_location_info(span);
    let file = file?;
    if line == 0 || column == 0 {
        return None;
    }

    Some(DebugSourcePosition {
        file: file.name.prefer_local_unconditionally().to_string().into(),
        line: line as i32,
        column: column as i32,
    })
}

/// Rebuild the monomorphized internal MIR used as stable MIR's input.
///
/// We only read `StmtDebugInfo`, which contains places. Stable MIR's additional
/// constant-evaluation walk cannot change these records.
fn monomorphized_mir_for_statement_debug_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> rustc_middle::mir::Body<'tcx> {
    let instance = match instance.def {
        InstanceKind::Intrinsic(def_id) => Instance::new_raw(def_id, instance.args),
        _ => instance,
    };
    let body = tcx.instance_mir(instance.def).clone();
    if !instance.args.is_empty() || tcx.def_kind(instance.def_id()) != DefKind::AnonConst {
        instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, body),
        )
    } else {
        body
    }
}

/// Preserve rustc's debug-only statement assignments across the
/// rustc-internal to stable-MIR boundary.
///
/// Capturing `StmtDebugInfo` directly is intentional: its `AssignRef` place may
/// contain a runtime `ProjectionElem::Index`, while ordinary
/// `VarDebugInfoContents::Place` does not permit that projection.
///
/// This function must run inside `rustc_internal::run`, so `stable(place)` can
/// intern every monomorphized projection type in the active bridge tables.
fn collect_statement_debug_info<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> mir_importer::StatementDebugInfoMap {
    use rustc_middle::mir::StmtDebugInfo;
    use rustc_public::rustc_internal;

    fn convert(info: &StmtDebugInfo<'_>) -> mir_importer::StatementDebugInfo {
        match info {
            StmtDebugInfo::AssignRef(destination, place) => {
                mir_importer::StatementDebugInfo::AssignRef {
                    destination: destination.index(),
                    place: rustc_internal::stable(*place),
                }
            }
            StmtDebugInfo::InvalidAssign(destination) => {
                mir_importer::StatementDebugInfo::InvalidAssign {
                    destination: destination.index(),
                }
            }
        }
    }

    let body = monomorphized_mir_for_statement_debug_info(tcx, instance);
    mir_importer::StatementDebugInfoMap {
        blocks: body
            .basic_blocks
            .iter()
            .map(|block| mir_importer::StatementDebugInfoBlock {
                before_statements: block
                    .statements
                    .iter()
                    .map(|statement| statement.debuginfos.iter().map(convert).collect())
                    .collect(),
                before_terminator: block
                    .after_last_stmt_debuginfos
                    .iter()
                    .map(convert)
                    .collect(),
            })
            .collect(),
    }
}

#[derive(Clone, Debug)]
struct OwnedStaticDebugIdentity {
    def_id: DefId,
    identity: mir_importer::DebugGlobalVariableIdentity,
    storage: StaticDebugStorage,
    is_function_local: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StaticDebugStorage {
    OrdinaryAs1,
    SharedArrayAs3,
    BarrierAs3,
}

/// Identify the two cuda-device marker types whose static storage is
/// materialized in NVPTX shared memory rather than at the marker's Rust layout.
fn shared_static_debug_storage(tcx: TyCtxt<'_>, def_id: DefId) -> Option<StaticDebugStorage> {
    // `type_of(..).instantiate_identity()` returns `Unnormalized<Ty>` since
    // the eager-normalization refactor (rust-lang/rust#155345). We are in a
    // fully monomorphized codegen context inspecting a static's declared
    // type, so follow `TyCtxt::static_ptr_ty`'s discipline and normalize the
    // wrapper away (a `static X: SomeAlias = ..` must still be recognized as
    // the underlying cuda-device marker ADT) rather than `skip_normalization`.
    let ty = tcx.normalize_erasing_regions(
        TypingEnv::fully_monomorphized(),
        tcx.type_of(def_id).instantiate_identity(),
    );
    let TyKind::Adt(adt, _) = ty.kind() else {
        return None;
    };
    if tcx.crate_name(adt.did().krate).as_str() != "cuda_device" {
        return None;
    }
    // `def_path_str` follows rustc's visible-parent map and can spell a
    // re-exported type as `cuda_device::SharedArray`. The canonical DefPath is
    // independent of re-export visibility and distinguishes the two marker
    // definitions exactly inside the already-validated crate.
    match tcx
        .def_path(adt.did())
        .to_string_no_crate_verbose()
        .as_str()
    {
        "::shared::SharedArray" => Some(StaticDebugStorage::SharedArrayAs3),
        "::barrier::Barrier" => Some(StaticDebugStorage::BarrierAs3),
        _ => None,
    }
}

/// Whether this static's definition is owned by a function-like item.
///
/// `DefKind::Static::nested` is unrelated: it identifies anonymous allocations
/// synthesized inside another static, while ordinary named block-local statics
/// (including `TILE`) have `nested: false`.
fn static_is_function_local(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let mut current = tcx.parent(def_id);
    loop {
        if matches!(
            tcx.def_kind(current),
            DefKind::Fn | DefKind::AssocFn | DefKind::Closure
        ) {
            return true;
        }
        let Some(parent) = tcx.def_key(current).parent else {
            return false;
        };
        current = DefId {
            krate: current.krate,
            index: parent,
        };
    }
}

/// Full-debug-only provenance walk for statics reachable by device code.
///
/// Every direct static reference begins as a constant in one of the already
/// collected, monomorphized device instances. Following that constant's CTFE
/// allocation graph finds both its static definition and targets reached only
/// through initializer relocations. Unlike seeding from every host CGU static,
/// this does not attempt to build debug types for unrelated host-only statics;
/// it also covers upstream statics, which are intentionally absent from local
/// CGUs when rustc decides they need not be codegenerated locally.
struct StaticDebugProvenance<'tcx> {
    tcx: TyCtxt<'tcx>,
    current_instance: Option<Instance<'tcx>>,
    statics: HashSet<DefId>,
    pending_allocations: Vec<AllocId>,
    seen_allocations: HashSet<AllocId>,
}

impl<'tcx> StaticDebugProvenance<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            current_instance: None,
            statics: HashSet::new(),
            pending_allocations: Vec::new(),
            seen_allocations: HashSet::new(),
        }
    }

    fn record_static(&mut self, def_id: DefId) {
        if !self.statics.insert(def_id) {
            return;
        }
        if let Ok(initializer) = self.tcx.eval_static_initializer(def_id) {
            self.pending_allocations.extend(
                initializer
                    .inner()
                    .provenance()
                    .ptrs()
                    .values()
                    .map(|provenance| provenance.alloc_id()),
            );
        }
    }

    fn record_const_value(&mut self, value: ConstValue) {
        match value {
            ConstValue::Scalar(Scalar::Ptr(pointer, _)) => {
                self.pending_allocations.push(pointer.provenance.alloc_id());
            }
            ConstValue::Indirect { alloc_id, .. } => self.pending_allocations.push(alloc_id),
            ConstValue::Slice { alloc_id, .. } => self.pending_allocations.push(alloc_id),
            ConstValue::Scalar(_) | ConstValue::ZeroSized => {}
        }
    }

    fn drain_allocations(&mut self) {
        while let Some(alloc_id) = self.pending_allocations.pop() {
            if !self.seen_allocations.insert(alloc_id) {
                continue;
            }

            match self.tcx.global_alloc(alloc_id) {
                GlobalAlloc::Static(def_id) => self.record_static(def_id),
                GlobalAlloc::Memory(allocation) => self.pending_allocations.extend(
                    allocation
                        .inner()
                        .provenance()
                        .ptrs()
                        .values()
                        .map(|provenance| provenance.alloc_id()),
                ),
                GlobalAlloc::Function { .. }
                | GlobalAlloc::VTable(..)
                | GlobalAlloc::TypeId { .. } => {}
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for StaticDebugProvenance<'tcx> {
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, location: Location) {
        let instance = self
            .current_instance
            .expect("static provenance visitor must be bound to an instance");
        let constant_value = instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, constant.const_),
        );
        if let Ok(value) =
            constant_value.eval(self.tcx, TypingEnv::fully_monomorphized(), constant.span)
        {
            self.record_const_value(value);
        }
        self.super_const_operand(constant, location);
    }
}

fn static_debug_namespace(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    source_function_names: &HashMap<DefId, String>,
) -> Option<Vec<String>> {
    let mut reversed = Vec::new();
    let mut current = tcx.parent(def_id);

    loop {
        let key = tcx.def_key(current);
        let mut segment = source_function_names
            .get(&current)
            .cloned()
            .unwrap_or_default();
        if segment.is_empty() {
            rustc_codegen_ssa::debuginfo::type_names::push_item_name(
                tcx,
                current,
                false,
                &mut segment,
            );
        }
        if segment.is_empty() {
            return None;
        }
        reversed.push(segment);

        let Some(parent) = key.parent else {
            break;
        };
        current = DefId {
            krate: current.krate,
            index: parent,
        };
    }

    reversed.reverse();
    Some(reversed)
}

fn collect_static_debug_identities<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
) -> Result<Vec<OwnedStaticDebugIdentity>, DeviceCodegenError> {
    let mut provenance = StaticDebugProvenance::new(tcx);
    let source_function_names: HashMap<_, _> = functions
        .iter()
        .filter(|function| function.is_kernel)
        .map(|function| (function.instance.def_id(), function.export_name.clone()))
        .collect();

    for function in functions {
        provenance.current_instance = Some(function.instance);
        provenance.visit_body(tcx.instance_mir(function.instance.def));
    }
    provenance.current_instance = None;
    provenance.drain_allocations();

    let mut statics: Vec<_> = provenance.statics.into_iter().collect();
    statics.sort_by_key(|def_id| tcx.def_path_str(*def_id));
    statics
        .into_iter()
        .filter_map(|def_id| {
            // Anonymous `nested: true` allocations do not necessarily have an
            // item type. Exclude them before `shared_static_debug_storage`
            // queries `type_of`; named function-local statics are `nested:
            // false` and remain eligible.
            if !matches!(tcx.def_kind(def_id), DefKind::Static { nested: false, .. }) {
                return None;
            }
            let storage =
                shared_static_debug_storage(tcx, def_id).unwrap_or(StaticDebugStorage::OrdinaryAs1);
            Some((def_id, storage))
        })
        .map(|(def_id, storage)| {
            let name = tcx.item_name(def_id).to_string();
            let namespace = static_debug_namespace(tcx, def_id, &source_function_names)
                .ok_or_else(|| {
                    DeviceCodegenError::Translation(format!(
                        "cannot represent the source namespace for device static `{}`",
                        tcx.def_path_str(def_id)
                    ))
                })?;
            let declaration_span = hygiene::walk_chain_collapsed(tcx.def_span(def_id), DUMMY_SP);
            let declaration = debug_position_from_span(tcx, declaration_span).ok_or_else(|| {
                DeviceCodegenError::Translation(format!(
                    "cannot represent the definition span for device static `{}`",
                    tcx.def_path_str(def_id)
                ))
            })?;
            Ok(OwnedStaticDebugIdentity {
                def_id,
                storage,
                is_function_local: storage != StaticDebugStorage::OrdinaryAs1
                    && static_is_function_local(tcx, def_id),
                identity: mir_importer::DebugGlobalVariableIdentity {
                    name,
                    namespace,
                    declaration,
                    is_local_to_unit: !tcx.is_reachable_non_generic(def_id),
                },
            })
        })
        .collect()
}

/// Errors that can occur during device code generation.
///
/// Most translation failures arrive as `cuda_oxide_codegen::PipelineError`;
/// `Translation` reports failures while preparing rustc-owned side tables at
/// the frontend boundary.
#[derive(Debug)]
pub enum DeviceCodegenError {
    /// No kernels were found to compile.
    NoKernels,
    /// Failed to enter or exit stable_mir context.
    StableMirError(String),
    /// MIR to Pliron IR translation failed.
    Translation(String),
    /// PTX generation (llc invocation) failed.
    PtxGeneration(String),
    /// A `#[device] extern` signature could not be represented exactly.
    InvalidDeviceExternSignature(String),
    /// IO error (file read/write).
    Io(std::io::Error),
}

impl std::fmt::Display for DeviceCodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoKernels => write!(f, "No kernel functions found"),
            Self::StableMirError(msg) => write!(f, "stable_mir error: {}", msg),
            Self::Translation(msg) => write!(f, "Translation failed: {}", msg),
            Self::PtxGeneration(msg) => write!(f, "PTX generation failed: {}", msg),
            Self::InvalidDeviceExternSignature(msg) => {
                write!(f, "Invalid device-extern signature: {msg}")
            }
            Self::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for DeviceCodegenError {}

impl From<std::io::Error> for DeviceCodegenError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Generates PTX for device functions using the cuda-oxide pipeline.
///
/// This is the main entry point for device codegen. It bridges between
/// rustc's internal types (`rustc_middle`) and mir-importer's stable_mir-based
/// pipeline (`rustc_public`).
///
/// ## Parameters
///
/// - `tcx`: The type context from rustc
/// - `functions`: Collected device functions from the collector module
/// - `config`: Output and diagnostic configuration
///
/// ## Returns
///
/// - `Ok(DeviceCodegenResult)`: Paths to generated .ll and .ptx files
/// - `Err(DeviceCodegenError)`: Description of what went wrong
///
/// ## Pipeline Stages
///
/// ```text
/// CollectedFunction<'tcx>
///         │
///         ├──▶ rustc_internal::stable() ──▶ rustc_public::Instance
///         │
///         └──▶ mir_importer::run_pipeline()
///                     │
///                     ├──▶ `dialect-mir` (alloca form)
///                     │
///                     ├──▶ `dialect-mir` (mem2reg → SSA)
///                     ├──▶ annotated loop unroll
///                     │
///                     ├──▶ LLVM dialect
///                     │
///                     ├──▶ textual LLVM IR (.ll)
///                     │
///                     └──▶ PTX (.ptx) via `llc`
/// ```
pub fn generate_device_code<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
    device_externs: &[DeviceExternDecl],
    config: &DeviceCodegenConfig,
) -> Result<DeviceCodegenResult, DeviceCodegenError> {
    use rustc_public::rustc_internal;

    if functions.is_empty() {
        return Err(DeviceCodegenError::NoKernels);
    }

    if config.verbose {
        eprintln!(
            "[device_codegen] Compiling {} functions, {} device externs to PTX via cuda-oxide pipeline",
            functions.len(),
            device_externs.len()
        );
        for func in functions {
            eprintln!(
                "[device_codegen]   {} {}",
                if func.is_kernel { "kernel" } else { "device" },
                func.export_name
            );
        }
        for decl in device_externs {
            eprintln!(
                "[device_codegen]   extern {} (convergent={}, pure={}, readonly={})",
                decl.export_name,
                decl.attrs.is_convergent,
                decl.attrs.is_pure,
                decl.attrs.is_readonly
            );
        }
    }

    // Prepare data we need to pass into the stable_mir closure
    // (closures can't capture references to local TyCtxt data)
    let export_names: Vec<(String, bool)> = functions
        .iter()
        .map(|f| (f.export_name.clone(), f.is_kernel))
        .collect();
    let debug_scope_maps: Vec<_> = functions
        .iter()
        .map(|f| build_debug_source_scope_map(tcx, f))
        .collect();

    // Convert device externs to mir-importer format
    // We extract signature info from rustc here since we have access to TyCtxt
    let stable_device_externs: Vec<mir_importer::DeviceExternDecl> = device_externs
        .iter()
        .map(|decl| {
            // Get function signature from rustc
            let fn_sig = tcx.fn_sig(decl.def_id).instantiate_identity();
            let fn_sig = fn_sig.skip_binder();

            if !matches!(fn_sig.abi(), rustc_abi::ExternAbi::C { unwind: false }) {
                return Err(DeviceCodegenError::InvalidDeviceExternSignature(format!(
                    "`{}` uses ABI {:?}; device externs must use `extern \"C\"` without unwinding",
                    decl.export_name,
                    fn_sig.abi()
                )));
            }

            if fn_sig.c_variadic() {
                return Err(DeviceCodegenError::InvalidDeviceExternSignature(format!(
                    "`{}` is variadic; variadic device externs are not supported",
                    decl.export_name
                )));
            }

            let param_types = fn_sig
                .inputs()
                .iter()
                .enumerate()
                .map(|(index, ty)| {
                    rustc_ty_to_device_extern_type(tcx, *ty, DeviceExternTypePosition::Parameter)
                        .map_err(|reason| {
                            DeviceCodegenError::InvalidDeviceExternSignature(format!(
                                "`{}` parameter {} (`{ty}`): {reason}",
                                decl.export_name, index
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let result_ty = fn_sig.output();
            let return_type =
                rustc_ty_to_device_extern_type(tcx, result_ty, DeviceExternTypePosition::Result)
                    .map_err(|reason| {
                        DeviceCodegenError::InvalidDeviceExternSignature(format!(
                            "`{}` result (`{result_ty}`): {reason}",
                            decl.export_name
                        ))
                    })?;

            Ok(mir_importer::DeviceExternDecl {
                export_name: decl.export_name.clone(),
                param_types,
                return_type,
                attrs: mir_importer::DeviceExternAttrs {
                    is_convergent: decl.attrs.is_convergent,
                    is_pure: decl.attrs.is_pure,
                    is_readonly: decl.attrs.is_readonly,
                },
            })
        })
        .collect::<Result<_, _>>()?;

    let output_dir = config.output_dir.clone();
    let output_name = config.output_name.clone();
    let verbose = config.verbose;
    let show_rustc_mir = config.dump_rustc_mir;
    let show_mir = config.dump_mir_dialect;
    let show_llvm = config.dump_llvm_dialect;
    let debug_kind = device_debug_kind(tcx.sess.opts.debuginfo);
    let static_debug_identities = if debug_kind.variables_enabled() {
        collect_static_debug_identities(tcx, functions)?
    } else {
        Vec::new()
    };

    // Print raw rustc MIR if requested (before conversion to stable_mir)
    if show_rustc_mir {
        use rustc_middle::ty::print::with_no_trimmed_paths;

        eprintln!();
        eprintln!("=== Rustc MIR (before translation) ===");
        for func in functions {
            let mir = tcx.instance_mir(func.instance.def);
            eprintln!();
            eprintln!("fn {} {{", func.export_name);

            // Print locals
            eprintln!(
                "    let mut _0: {:?};",
                mir.local_decls[rustc_middle::mir::Local::from_u32(0)].ty
            );
            for (local, decl) in mir.local_decls.iter_enumerated().skip(1) {
                let mutability = if decl.mutability == rustc_middle::mir::Mutability::Mut {
                    "mut "
                } else {
                    ""
                };
                eprintln!("    let {}_{}:  {:?};", mutability, local.index(), decl.ty);
            }

            // Print debug info
            for debug_info in &mir.var_debug_info {
                with_no_trimmed_paths!(eprintln!(
                    "    debug {:?} => {:?};",
                    debug_info.name, debug_info.value
                ));
            }

            // Print basic blocks
            for (bb_idx, bb_data) in mir.basic_blocks.iter_enumerated() {
                eprintln!("    bb{}: {{", bb_idx.index());
                for stmt in &bb_data.statements {
                    with_no_trimmed_paths!(eprintln!("        {:?}", stmt));
                }
                with_no_trimmed_paths!(eprintln!("        {:?}", bb_data.terminator().kind));
                eprintln!("    }}");
            }
            eprintln!("}}");
        }
        eprintln!();
    }

    // Enter stable_mir context and run the pipeline.
    //
    // rustc_internal::run() does the following:
    // 1. Creates Tables for type/instance interning
    // 2. Sets up thread-local CompilerCtxt
    // 3. Runs our closure with access to stable() conversion
    // 4. Tears down the context and returns our result
    // Pre-compute `#[inline(always)]` flags before entering the stable_mir
    // context, since the query lives on `rustc_middle::TyCtxt` and is not
    // exposed through stable_mir. Preserving this hint avoids making helper
    // boundaries depend entirely on later optimizer heuristics.
    let inline_always_flags: Vec<bool> = functions
        .iter()
        .map(|func| {
            let def_id = func.instance.def_id();
            matches!(
                tcx.codegen_fn_attrs(def_id).inline,
                rustc_hir::attrs::InlineAttr::Always | rustc_hir::attrs::InlineAttr::Force { .. }
            )
        })
        .collect();
    let device_mono_reachability: Vec<crate::collector::DeviceMonoReachability> = functions
        .iter()
        .map(|func| crate::collector::device_mono_reachability(tcx, func.instance))
        .collect();

    let result = rustc_internal::run(tcx, || {
        let mut debug_global_variables = BTreeMap::new();
        for owned in &static_debug_identities {
            let stable_item = rustc_internal::stable(MonoItem::Static(owned.def_id));
            let rustc_public::mir::mono::MonoItem::Static(static_def) = stable_item else {
                unreachable!("internal static MonoItem must remain a stable static MonoItem")
            };
            let static_ty = static_def.ty();
            let info = match owned.storage {
                StaticDebugStorage::SharedArrayAs3 => {
                    mir_importer::build_debug_shared_array_variable_info(
                        owned.identity.clone(),
                        &static_ty,
                    )
                }
                StaticDebugStorage::OrdinaryAs1 | StaticDebugStorage::BarrierAs3 => {
                    mir_importer::build_debug_global_variable_info(
                        owned.identity.clone(),
                        &static_ty,
                    )
                }
            };
            let Some(mut info) = info else {
                // The local-variable debug path is deliberately best-effort
                // for semantic types it cannot yet describe (notably unions).
                // Globals must fail closed in the same way: omitting this one
                // DIE is preferable to attaching the physical byte-array type,
                // and an unsupported static must not disable correct metadata
                // for every other AS1 global in the module.
                continue;
            };
            info.is_function_local = owned.is_function_local;
            let key = mir_importer::device_static_global_key(&static_def);
            if let Some(previous) = debug_global_variables.insert(key.clone(), info.clone())
                && previous != info
            {
                return Err(mir_importer::PipelineError::Translation(format!(
                    "conflicting debug identities for device static `{key}`"
                )));
            }
        }

        // Convert internal Instance<'tcx> to stable_mir Instance.
        // Drop glue instances whose bodies are provably no-ops are filtered
        // out: the mir-importer's translate_drop fast-path emits a plain
        // branch for them and never references the function, so translating
        // their (potentially complex) shim bodies is both unnecessary and
        // can fail on constructs the device pipeline does not support.
        // The collector already skips these at discovery time with the
        // same shared predicate (collector::process_drop_place), so this
        // filter is a final guard that keeps translation in lockstep with
        // emission if a future collection path forgets the check.
        let stable_functions: Vec<mir_importer::CollectedFunction> = functions
            .iter()
            .zip(export_names.iter())
            .zip(debug_scope_maps.iter())
            .zip(inline_always_flags.iter())
            .zip(device_mono_reachability.iter())
            .filter_map(
                |(
                    (((func, (export_name, is_kernel)), debug_source_scopes), is_inline_always),
                    reachability,
                )| {
                    // Use rustc_internal::stable() to convert the Instance.
                    // This is the key bridge between rustc_middle and rustc_public types.
                    let stable_instance = rustc_internal::stable(func.instance);
                    let statement_debug_info = debug_kind
                        .variables_enabled()
                        .then(|| collect_statement_debug_info(tcx, func.instance));

                    // Skip no-op drop glue: the mir-importer lowers these as
                    // plain branches (via drop_glue_is_noop) and never emits a
                    // call, so the function body is dead. Translating it would
                    // fail on IntoIter and similar stdlib shims whose MIR
                    // contains constructs the device pipeline does not support.
                    if matches!(
                        func.instance.def,
                        InstanceKind::Shim(ShimKind::DropGlue(..))
                    ) && mir_importer::drop_instance_is_noop(&stable_instance)
                    {
                        return None;
                    }

                    Some(mir_importer::CollectedFunction {
                        instance: stable_instance,
                        rustc_mir_block_count: reachability.block_count,
                        rustc_mono_successors: reachability.successors.clone(),
                        is_kernel: *is_kernel,
                        export_name: export_name.clone(),
                        debug_source_scopes: Some(debug_source_scopes.clone()),
                        statement_debug_info,
                        is_inline_always: *is_inline_always,
                    })
                },
            )
            .collect();

        // Check for NVVM IR mode (set by cargo oxide --emit-nvvm-ir)
        let emit_nvvm_ir = std::env::var("CUDA_OXIDE_EMIT_NVVM_IR").is_ok();

        if verbose {
            eprintln!(
                "[device_codegen] Converted {} functions to stable_mir format",
                stable_functions.len()
            );
            if emit_nvvm_ir {
                eprintln!("[device_codegen] NVVM IR mode enabled");
            }
        }

        let target_arch = std::env::var("CUDA_OXIDE_TARGET").ok();
        let device_arch_hint = mir_importer::pipeline::DeviceArchHint::from_env_value(
            std::env::var("CUDA_OXIDE_DEVICE_ARCH"),
        );
        let allow_fma_contraction = std::env::var_os("CUDA_OXIDE_NO_FMA").is_none();

        if verbose && !allow_fma_contraction {
            eprintln!("[device_codegen] FMA contraction disabled");
        }

        // This crate's PTX bundle can be textually merged with other crates'
        // bundles at load time (`load_all_ptx_bundles_merged`), so
        // counter-named module-scope symbols (`__shared_mem_N`,
        // `__device_global_N`) must be unique per compilation, not only per
        // module. The stable crate id hashes the crate name and metadata:
        // deterministic across rebuilds of one crate, distinct across the
        // crates of one binary (#1277).
        let module_disambiguator =
            Some(tcx.stable_crate_id(rustc_hir::def_id::LOCAL_CRATE).as_u64());

        // Create pipeline config
        let pipeline_config = mir_importer::PipelineConfig {
            output_dir: output_dir.clone(),
            output_name: output_name.clone(),
            verbose,
            show_mir_dialect: show_mir,
            show_llvm_dialect: show_llvm,
            emit_nvvm_ir,
            target_arch,
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint,
            debug_kind,
            debug_global_variables,
            allow_fma_contraction,
            module_disambiguator,
        };

        // Resolve the lang-item DefIds the type translator compares
        // projections against (e.g. FnOnce::Output). This must happen here,
        // inside rustc_internal::run, because stable() needs the thread-local
        // conversion tables and the resulting ids are only valid within this
        // context. `None` (no_core builds) just means the comparisons in the
        // importer never match and it errors instead of guessing.
        let lang_items = tcx.lang_items();
        let known_defs = mir_importer::KnownDefs {
            fn_once_output: lang_items.fn_once_output().map(rustc_internal::stable),
            index_trait: lang_items.index_trait().map(rustc_internal::stable),
            index_mut_trait: lang_items.index_mut_trait().map(rustc_internal::stable),
        };

        // Run the cuda-oxide pipeline!
        // Rust MIR → `dialect-mir` → mem2reg → unroll → LLVM dialect → LLVM IR → PTX.
        // Device externs are emitted as `declare` statements in LLVM IR
        mir_importer::run_pipeline(
            &stable_functions,
            &stable_device_externs,
            &pipeline_config,
            known_defs,
        )
    });

    // Handle the result from rustc_internal::run.
    // We have nested Results: outer from run(), inner from run_pipeline().
    match result {
        Ok(pipeline_result) => match pipeline_result {
            Ok(compilation_result) => {
                let artifact = read_compilation_artifact(&compilation_result)?;
                let ptx_content = match artifact.as_ref() {
                    Some(artifact) if artifact.kind == DeviceCodegenArtifactKind::Ptx => {
                        Some(String::from_utf8(artifact.bytes.clone()).map_err(|e| {
                            DeviceCodegenError::PtxGeneration(format!(
                                "generated PTX is not valid UTF-8: {e}"
                            ))
                        })?)
                    }
                    _ => None,
                };

                if config.verbose {
                    if let Some(artifact) = artifact.as_ref() {
                        eprintln!(
                            "[device_codegen] Embeddable artifact generated: {} ({:?}, target: {})",
                            artifact.name, artifact.kind, compilation_result.target
                        );
                    } else {
                        eprintln!(
                            "[device_codegen] No embeddable artifact found for {} (target: {})",
                            compilation_result.ll_path.display(),
                            compilation_result.target
                        );
                    }
                }

                Ok(DeviceCodegenResult {
                    ptx_path: compilation_result.ptx_path,
                    ll_path: compilation_result.ll_path,
                    target: compilation_result.target,
                    ptx_content,
                    artifact,
                    allow_fma_contraction: compilation_result.allow_fma_contraction,
                    debug_kind,
                    kernel_launch_bounds: compilation_result.kernel_launch_bounds,
                })
            }
            Err(pipeline_err) => Err(DeviceCodegenError::PtxGeneration(format!(
                "{}",
                pipeline_err
            ))),
        },
        Err(stable_mir_err) => Err(DeviceCodegenError::StableMirError(format!(
            "{:?}",
            stable_mir_err
        ))),
    }
}

fn device_debug_kind(rustc_debug: DebugInfo) -> llvm_export::export::DebugKind {
    device_debug_kind_with_override(
        rustc_debug,
        std::env::var("CUDA_OXIDE_DEBUG").ok().as_deref(),
    )
}

fn device_debug_kind_with_override(
    rustc_debug: DebugInfo,
    override_value: Option<&str>,
) -> llvm_export::export::DebugKind {
    // The alias table lives in cuda-artifact-finalizer so cargo-oxide's
    // selective full-debug build policy and this DWARF emission level can
    // never disagree about what a value means.
    if let Some(policy) =
        override_value.and_then(cuda_artifact_finalizer::DebugPolicy::parse_env_override)
    {
        return match policy {
            cuda_artifact_finalizer::DebugPolicy::None => llvm_export::export::DebugKind::Off,
            cuda_artifact_finalizer::DebugPolicy::LineTables => {
                llvm_export::export::DebugKind::LineTables
            }
            cuda_artifact_finalizer::DebugPolicy::Full => llvm_export::export::DebugKind::Full,
        };
    }

    match rustc_debug {
        DebugInfo::None => llvm_export::export::DebugKind::Off,
        DebugInfo::LineDirectivesOnly
        | DebugInfo::LineTablesOnly
        | DebugInfo::Limited
        | DebugInfo::Full => llvm_export::export::DebugKind::LineTables,
    }
}

fn read_compilation_artifact(
    result: &mir_importer::CompilationResult,
) -> Result<Option<DeviceCodegenArtifact>, DeviceCodegenError> {
    let kind = match result.artifact_kind {
        mir_importer::CompilationArtifactKind::Ptx => DeviceCodegenArtifactKind::Ptx,
        mir_importer::CompilationArtifactKind::NvvmIr => DeviceCodegenArtifactKind::NvvmIr,
        mir_importer::CompilationArtifactKind::Ltoir => DeviceCodegenArtifactKind::Ltoir,
        mir_importer::CompilationArtifactKind::Cubin => DeviceCodegenArtifactKind::Cubin,
    };

    match std::fs::read(&result.artifact_path) {
        Ok(bytes) => Ok(Some(DeviceCodegenArtifact {
            kind,
            name: result
                .artifact_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("device-artifact")
                .to_string(),
            bytes,
        })),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(DeviceCodegenError::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DeviceCodegenConfig::default();
        assert!(!config.verbose);
        assert_eq!(config.output_name, "kernel");
    }

    #[test]
    fn device_debug_kind_follows_rustc_debuginfo() {
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::None, None),
            llvm_export::export::DebugKind::Off
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::LineTablesOnly, None),
            llvm_export::export::DebugKind::LineTables
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::Full, None),
            llvm_export::export::DebugKind::LineTables
        );
    }

    #[test]
    fn device_debug_kind_env_override_wins() {
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::Full, Some("off")),
            llvm_export::export::DebugKind::Off
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::None, Some(" Line-Tables ")),
            llvm_export::export::DebugKind::LineTables
        );
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::None, Some("full")),
            llvm_export::export::DebugKind::Full
        );
        // The nvcc-style numeric spelling goes through the same shared
        // parser, so it must select full debug here exactly like "full".
        assert_eq!(
            device_debug_kind_with_override(DebugInfo::None, Some("2")),
            llvm_export::export::DebugKind::Full
        );
    }

    #[test]
    fn read_compilation_artifact_uses_declared_nvvm_ir_path() {
        let temp_dir = unique_temp_dir("cuda-codegen-artifact");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let ll_path = temp_dir.join("demo.ll");
        let ptx_path = temp_dir.join("demo.ptx");
        std::fs::write(&ll_path, b"nvvm ir").unwrap();
        std::fs::write(&ptx_path, b"stale ptx").unwrap();

        let result = mir_importer::CompilationResult {
            ll_path: ll_path.clone(),
            ptx_path,
            artifact_path: ll_path,
            artifact_kind: mir_importer::CompilationArtifactKind::NvvmIr,
            target: "sm_90".to_string(),
            allow_fma_contraction: false,
            kernel_launch_bounds: BTreeMap::new(),
        };

        let artifact = read_compilation_artifact(&result).unwrap().unwrap();
        assert_eq!(artifact.kind, DeviceCodegenArtifactKind::NvvmIr);
        assert_eq!(artifact.name, "demo.ll");
        assert_eq!(artifact.bytes, b"nvvm ir");

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn read_compilation_artifact_reads_declared_cubin_path() {
        let temp_dir = unique_temp_dir("cuda-codegen-artifact");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let ll_path = temp_dir.join("demo.ll");
        let cubin_path = temp_dir.join("demo.cubin");
        let ptx_path = temp_dir.join("demo.ptx");
        std::fs::write(&cubin_path, b"cubin").unwrap();

        let result = mir_importer::CompilationResult {
            ll_path,
            ptx_path,
            artifact_path: cubin_path,
            artifact_kind: mir_importer::CompilationArtifactKind::Cubin,
            target: "sm_90".to_string(),
            allow_fma_contraction: true,
            kernel_launch_bounds: BTreeMap::new(),
        };

        let artifact = read_compilation_artifact(&result).unwrap().unwrap();
        assert_eq!(artifact.kind, DeviceCodegenArtifactKind::Cubin);
        assert_eq!(artifact.name, "demo.cubin");
        assert_eq!(artifact.bytes, b"cubin");

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{}-{nanos}", std::process::id()))
    }
}
