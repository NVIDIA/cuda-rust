/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compilation pipeline: MIR → `dialect-mir` → LLVM dialect → LLVM IR → PTX.
//!
//! Orchestrates the full compilation flow from collected MIR functions to
//! executable PTX code.
//!
//! # Pipeline Steps
//!
//! ```text
//! MIR -> dialect-mir -> verify -> prepare -> library proof/provider binding
//!     -> optional transforms -> LLVM dialect -> LLVM IR -> PTX
//! ```
//!
//! Normal preparation promotes locals with `mem2reg`. Full variable debug keeps
//! stack slots and skips optional transforms; the required library proof uses
//! those slots at the same preparation boundary.
//!
//! # GPU Target Selection
//!
//! The pipeline auto-detects GPU features in the generated LLVM IR and selects
//! an appropriate target:
//!
//! | Feature                       | Target  | Architecture         |
//! |-------------------------------|---------|----------------------|
//! | tcgen05/TMEM                  | sm_100a | Blackwell datacenter |
//! | TMA multicast                 | sm_100a | Blackwell datacenter |
//! | WGMMA                         | sm_90a  | Hopper only          |
//! | TMA/mbarrier                  | sm_100  | Hopper+ compatible   |
//! | bf16x2 add/sub/mul            | sm_90   | Hopper+ compatible   |
//! | other bf16x2 ALU              | sm_80   | Ampere+ compatible   |
//! | INT8 `mma.m16n8k32`           | sm_80   | PTX 7.0+             |
//! | `cp.async` (non-bulk)         | sm_80   | Ampere+              |
//! | Basic CUDA                    | sm_80   | Ampere+ (max compat) |
//!
//! Override with `CUDA_OXIDE_TARGET=<target>` environment variable.

use cuda_oxide_codegen::__private::{
    BackendOptions, ModuleArtifactKind, ModulePipelineRequest, OutputFiles, PipelineTrace,
    PreparedModuleStage, append_to_module, prepare_translated_module, verify_operation,
};
pub use cuda_oxide_codegen::__private::{
    DeviceArchHint, DeviceExternAttrs, DeviceExternDecl, PipelineError,
};
use llvm_export::export::DebugKind;
pub use llvm_export::export::DeviceExternType;
use llvm_export::ops::{DebugGlobalVariableInfo, DebugSourcePosition};
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::context::Context;
use pliron::identifier::Legaliser;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::Typed;
use rustc_public::mir::mono::Instance;
use rustc_public::ty::Ty;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn stderr_pipeline_trace(message: &str) {
    eprintln!("{message}");
}

/// A function collected for GPU compilation.
///
/// Represents a monomorphized function instance that will be translated to PTX.
/// For generic functions like `add::<f32>`, the instance contains the concrete
/// type substitutions.
#[derive(Debug, Clone)]
pub struct CollectedFunction {
    /// The monomorphized stable_mir instance (includes concrete generic args).
    pub instance: Instance,
    /// Number of blocks in the rustc MIR body from which `instance.body()` is
    /// converted. The importer verifies that conversion preserved the CFG.
    pub rustc_mir_block_count: usize,
    /// Exact per-block rustc successors for this monomorphized instance under
    /// CUDA Oxide's device runtime-check policy.
    pub rustc_mono_successors: Vec<Vec<usize>>,
    /// True if this is a GPU kernel entry point (has `#[kernel]` attribute).
    pub is_kernel: bool,
    /// The name to export in PTX. For kernels, this is the user-visible name.
    pub export_name: String,
    /// Original source identity used by the retained library analyses.
    pub library_source: Option<dialect_mir::library::LibraryFunctionSourceAttr>,
    /// Authenticated classes in original kernel argument order.
    pub library_root_arguments: Vec<dialect_mir::library::LibraryRootArgumentClass>,
    /// rustc MIR source-scope data used to build inlined debug scopes.
    pub debug_source_scopes: Option<llvm_export::ops::DebugSourceScopeMap>,
    /// rustc's statement-level debug assignments, aligned with stable MIR.
    ///
    /// Stable MIR omits these records, so the codegen bridge carries them in a
    /// sidecar. Full debug uses supported events to keep affected pointer values
    /// in stack slots; other modes emit no extra code.
    pub statement_debug_info: Option<StatementDebugInfoMap>,
    /// The function's `#[inline]` request from rustc's `CodegenFnAttrs`. The
    /// stable_mir API does not expose inline attributes, so this is queried via
    /// `rustc_middle::TyCtxt::codegen_fn_attrs` in `rustc-codegen-cuda` and
    /// threaded through.
    ///
    /// `mir-lower` turns the request into the matching LLVM function attribute
    /// on the definition, so the matched LLVM middle-end (`opt -O2`) honors it
    /// before PTX generation; no separate inliner pass is added. This keeps
    /// helper boundaries, `#[inline(never)]` ones included, from depending
    /// entirely on optimizer heuristics.
    pub inline: Option<dialect_mir::attributes::InlineIntentAttr>,
}

/// Per-function statement debug records aligned with the stable MIR body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatementDebugInfoMap {
    /// One entry per MIR basic block, in block-index order.
    pub blocks: Vec<StatementDebugInfoBlock>,
}

/// Debug records attached to one MIR basic block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatementDebugInfoBlock {
    /// Records emitted immediately before each primary statement. The vector
    /// length must equal the corresponding stable MIR statement count.
    pub before_statements: Vec<Vec<StatementDebugInfo>>,
    /// Records emitted after the last statement and immediately before the
    /// block terminator.
    pub before_terminator: Vec<StatementDebugInfo>,
}

/// The subset of rustc's internal `StmtDebugInfo` needed by code generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementDebugInfo {
    /// At this point, source variables associated with `destination` are
    /// represented by the address of `place`.
    ///
    /// This sidecar preserves rustc's internal statement-debug place directly,
    /// so it may contain `ProjectionElem::Index` even though ordinary
    /// `VarDebugInfoContents::Place` rejects runtime indices.
    AssignRef {
        destination: rustc_public::mir::Local,
        place: rustc_public::mir::Place,
    },
    /// The source variables associated with this local no longer have a valid
    /// value and must be reported as unavailable.
    InvalidAssign {
        destination: rustc_public::mir::Local,
    },
}

/// Device artifact format produced by a successful pipeline run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationArtifactKind {
    /// Textual PTX assembly, loadable by the CUDA driver.
    Ptx,
    /// NVVM-compatible LLVM IR, intended for libNVVM/nvJitLink.
    NvvmIr,
    /// Binary LTOIR, intended for nvJitLink.
    Ltoir,
    /// Final cubin image, loadable by the CUDA driver.
    Cubin,
}

/// Launch bounds attached to one kernel entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelLaunchBounds {
    /// Maximum threads per block declared by `#[launch_bounds]`.
    pub max_threads: u32,
    /// Requested minimum resident blocks per SM, when explicitly provided.
    pub min_blocks: Option<u32>,
}

/// Output paths, target, and artifact format from successful compilation.
pub struct CompilationResult {
    /// Path to generated LLVM IR (`.ll` file).
    pub ll_path: std::path::PathBuf,
    /// Path to generated PTX assembly (`.ptx` file).
    pub ptx_path: std::path::PathBuf,
    /// Path to the artifact that should be embedded or consumed by the caller.
    pub artifact_path: std::path::PathBuf,
    /// Format of `artifact_path`.
    pub artifact_kind: CompilationArtifactKind,
    /// GPU target architecture used (e.g., `sm_90a`, `sm_80`).
    pub target: String,
    /// Floating-point contraction policy that later compilation stages must
    /// preserve.
    pub allow_fma_contraction: bool,
    /// Per-kernel source launch bounds preserved for post-link diagnostics.
    pub kernel_launch_bounds: BTreeMap<String, KernelLaunchBounds>,
    /// Finalized resource requirements, keyed by concrete exported kernel name.
    pub kernel_launch_facts: BTreeMap<String, oxide_artifacts::KernelLaunchFacts>,
}

/// Configuration for the compilation pipeline.
#[derive(Clone)]
pub struct PipelineConfig {
    /// Directory for output files (`.ll`, `.ptx`).
    pub output_dir: std::path::PathBuf,
    /// Base name for output files (e.g., `"kernel"` → `kernel.ll`, `kernel.ptx`).
    pub output_name: String,
    /// Print progress messages to stdout.
    pub verbose: bool,
    /// Dump the `dialect-mir` module after translation (for debugging).
    pub show_mir_dialect: bool,
    /// Dump the LLVM dialect module after lowering (for debugging).
    pub show_llvm_dialect: bool,
    /// Emit NVVM IR suitable for libNVVM or other NVVM-compatible tools.
    ///
    /// When true:
    /// - Uses full NVPTX datalayout
    /// - Adds `@llvm.used` to preserve kernels from optimization
    /// - Adds `!nvvm.annotations` for all kernels
    /// - Adds `!nvvmir.version` metadata
    /// - Outputs `.ll` file in NVVM IR format
    ///
    /// The output can be compiled to LTOIR using `nvvmCompileProgram -gen-lto`.
    ///
    /// Pre-Blackwell targets use the legacy LLVM 7 dialect; Blackwell and
    /// newer targets use the modern opaque-pointer dialect. Architecture is
    /// controlled by `target_arch` or `device_arch_hint` (normally populated
    /// by `cargo oxide`). When an ordinary build switches to NVVM IR after
    /// detecting libdevice, the pipeline may instead select the module's
    /// feature-based target floor.
    pub emit_nvvm_ir: bool,
    /// Explicit CUDA target used to choose NVVM IR syntax.
    ///
    /// Normally set by `cargo oxide --arch` or `CUDA_OXIDE_TARGET`.
    pub target_arch: Option<String>,
    /// Human-readable name for whatever set `target_arch`, used when a target
    /// error has to say where the target came from.
    ///
    /// Defaults to `"PipelineConfig::target_arch"`. A caller that read the
    /// target from somewhere the user would recognise sets its own label:
    /// `rustc-codegen-cuda` reads `CUDA_OXIDE_TARGET` itself and says so.
    /// Only meaningful when `target_arch` is `Some`.
    pub target_arch_source: &'static str,
    /// Detected architecture of the local GPU (`CUDA_OXIDE_DEVICE_ARCH`).
    ///
    /// Valid hints are used only when no explicit target is provided; invalid
    /// hints always produce a target-selection error.
    pub device_arch_hint: Option<DeviceArchHint>,
    /// Device debug metadata tier.
    pub debug_kind: DebugKind,
    /// Source identities and semantic types for device statics,
    /// keyed by [`device_static_global_key`]. The shared carrier reuses the
    /// reviewed tagged identity rather than defining a second key domain;
    /// source paths remain display-only because same-leaf block statics can
    /// have identical paths.
    ///
    /// The rustc frontend populates this only for full debug builds. Keeping
    /// the map module-scoped lets every per-function reference receive the
    /// same identity before MIR lowering deduplicates physical globals.
    pub debug_global_variables: BTreeMap<String, DebugGlobalVariableInfo>,
    /// Whether ordinary floating-point multiply/add or multiply/subtract
    /// expressions may contract into fused operations.
    ///
    /// Explicit fused operations, such as `f32::mul_add`, are unaffected.
    pub allow_fma_contraction: bool,
    /// Stable per-compilation identity woven into counter-named module-scope
    /// symbols (`__shared_mem_*`, `__device_global_*`) and the dynamic
    /// shared-memory pool externs (`__dynamic_smem_*`) during MIR lowering.
    ///
    /// The rustc frontend passes the crate's `StableCrateId` hash so that
    /// bundles from different crates define no common module-scope symbol
    /// when `load_all_ptx_bundles_merged` concatenates their PTX (#1277).
    /// `None` keeps the undecorated historical names.
    pub module_disambiguator: Option<u64>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            output_dir: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            output_name: "kernel".to_string(),
            verbose: true,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: false,
            target_arch: None,
            target_arch_source: "PipelineConfig::target_arch",
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            debug_global_variables: BTreeMap::new(),
            allow_fma_contraction: true,
            module_disambiguator: None,
        }
    }
}

/// Rustc-owned source identity for one static, captured before entering the
/// stable-MIR closure. The semantic type is converted separately inside that
/// closure so it remains the Rust type even when initialized storage later
/// lowers to a physical byte array.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DebugGlobalVariableIdentity {
    pub name: String,
    pub namespace: Vec<String>,
    pub declaration: DebugSourcePosition,
    pub is_local_to_unit: bool,
}

/// Return the opaque, per-compilation identity used for one device static.
///
/// `StaticDef::name()` is a display path, not an identity: two same-named
/// statics in sibling blocks can have the same path. For ordinary Rust
/// symbols, rustc's codegen symbol includes the DefPath disambiguator needed
/// to keep those allocations distinct and retains upstream crate identity. An
/// explicit `#[export_name]` is preserved verbatim. The symbol is wrapped in a
/// domain tag so neither form can alias a compiler-made promoted key, then used
/// only as a join key within one compilation; source-facing debug names and
/// namespaces are carried separately.
pub fn device_static_global_key(static_def: &rustc_public::mir::mono::StaticDef) -> String {
    let symbol = rustc_public::mir::mono::Instance::from(*static_def)
        .mangled_name()
        .to_string();
    dialect_mir::ops::encode_rust_static_global_key(&symbol)
}

/// Combine rustc-owned identity with a stable-MIR semantic type.
pub fn build_debug_global_variable_info(
    identity: DebugGlobalVariableIdentity,
    ty: &Ty,
) -> Option<DebugGlobalVariableInfo> {
    let debug_ty = crate::translator::body::debug_type_for_ty(ty)?;
    let semantic_size_bits = ty.layout().ok()?.shape().size.bytes() as u64 * 8;
    if debug_ty.size_bits() != semantic_size_bits {
        // The shared local-variable type builder currently spells every
        // reference as one machine pointer. That is exact for thin references,
        // but not for slices/trait objects. A global DIE whose type size does
        // not match rustc's semantic layout would be actively misleading, so
        // omit it until the richer fat-pointer representation exists.
        return None;
    }
    Some(DebugGlobalVariableInfo {
        name: identity.name,
        namespace: identity.namespace,
        ty: debug_ty,
        declaration: identity.declaration,
        is_local_to_unit: identity.is_local_to_unit,
        is_function_local: false,
    })
}

/// Combine rustc-owned identity with the physical `[T; N]` backing type of a
/// `SharedArray<T, N>` static.
///
/// The declared Rust marker is a ZST and therefore cannot be used as the
/// variable's type. The importer already materializes one shared allocation
/// containing `N` elements of `T`; this builder describes that same object.
pub fn build_debug_shared_array_variable_info(
    identity: DebugGlobalVariableIdentity,
    ty: &Ty,
) -> Option<DebugGlobalVariableInfo> {
    use rustc_public::ty::{GenericArgKind, RigidTy, TyKind};

    let TyKind::RigidTy(RigidTy::Adt(_, generic_args)) = ty.kind() else {
        return None;
    };
    let element_ty = generic_args.0.iter().find_map(|arg| match arg {
        GenericArgKind::Type(ty) => Some(*ty),
        _ => None,
    })?;
    let count = generic_args.0.iter().find_map(|arg| match arg {
        GenericArgKind::Const(value) => value.eval_target_usize().ok(),
        _ => None,
    })?;
    let ty = crate::translator::body::debug_shared_array_type(&element_ty, count)?;
    Some(DebugGlobalVariableInfo {
        name: identity.name,
        namespace: identity.namespace,
        ty,
        declaration: identity.declaration,
        is_local_to_unit: identity.is_local_to_unit,
        is_function_local: false,
    })
}

/// Attach module-owned static identities to every per-function materialization.
/// MIR lowering subsequently uniques these operations by `global_key`.
fn attach_debug_global_variables(
    ctx: &mut Context,
    func_op: pliron::context::Ptr<Operation>,
    globals: &BTreeMap<String, DebugGlobalVariableInfo>,
) {
    if globals.is_empty() {
        return;
    }

    let owner_function = Operation::get_op::<dialect_mir::ops::MirFuncOp>(func_op, ctx)
        .map(|function| function.get_symbol_name(ctx).to_string());
    let region = func_op.deref(ctx).get_region(0);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let operations: Vec<_> = blocks
        .iter()
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .collect();

    for op in operations {
        if let Some(global) = Operation::get_op::<dialect_mir::ops::MirGlobalAllocOp>(op, ctx) {
            let result_ty = global
                .get_operation()
                .deref(ctx)
                .get_result(0)
                .get_type(ctx);
            let is_debuggable_device_global = result_ty
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirPtrType>()
                .is_some_and(|pointer| {
                    matches!(
                        pointer.address_space,
                        dialect_mir::types::address_space::GLOBAL
                            | dialect_mir::types::address_space::CONSTANT
                    )
                });
            if is_debuggable_device_global
                && let Some(key) = global
                    .get_attr_global_key(ctx)
                    .map(|key| String::from(key.clone()))
                && let Some(info) = globals.get(&key)
            {
                llvm_export::ops::set_debug_global_variable(ctx, op, info);
            }
            continue;
        }

        if let Some(shared) = Operation::get_op::<dialect_mir::ops::MirSharedAllocOp>(op, ctx)
            && let Some(key) = shared
                .get_attr_source_key(ctx)
                .map(|key| String::from(key.clone()))
            && let Some(info) = globals.get(&key)
        {
            llvm_export::ops::set_debug_global_variable(ctx, op, info);
            if info.is_function_local
                && let Some(owner) = &owner_function
            {
                llvm_export::ops::set_debug_global_owner_function(ctx, op, owner);
            }
        }
    }
}

/// Merges `config` over the environment-derived backend defaults.
///
/// Environment-derived compatibility options are read once at the rustc
/// frontend boundary. Explicit pipeline configuration retains precedence.
fn backend_options_for(config: &PipelineConfig) -> BackendOptions {
    let mut backend_options =
        BackendOptions::from_env_with_device_hint(config.device_arch_hint.clone());
    if config.target_arch.is_some() {
        backend_options.target_arch = config.target_arch.clone();
        // The label travels with the value it describes. Overriding the
        // target and leaving `from_env`'s "CUDA_OXIDE_TARGET" in place made
        // every target error blame an env var the caller may never have set.
        backend_options.target_arch_source = config.target_arch_source;
    }
    backend_options.verbose = backend_options.verbose || config.verbose;
    backend_options.no_fma = !config.allow_fma_contraction;
    backend_options.module_disambiguator = config.module_disambiguator;
    backend_options
}

/// Select single-block-cluster semantics only for a known pre-sm90 target.
/// Invalid inputs are still diagnosed by the shared backend validator.
fn cluster_query_fallback_target(options: &BackendOptions) -> Option<String> {
    if let Some(target) = &options.target_arch {
        let target = cuda_oxide_codegen::experimental::Target::parse(target).ok()?;
        return (target.capability() < 90).then(|| target.sm());
    }
    match &options.device_arch_hint {
        Some(DeviceArchHint::Valid(arch)) if arch.as_ref().capability() < 90 => {
            Some(arch.as_ref().sm())
        }
        _ => None,
    }
}

/// Runs the full compilation pipeline on collected functions.
///
/// # Pipeline Steps
///
/// 1. Register the `dialect-mir`, `dialect-nvvm`, and LLVM dialects
/// 2. Translate each function's MIR body into `dialect-mir`
/// 3. Verify the `dialect-mir` module
/// 4. Unless full variable-debug mode is enabled, run `mem2reg` to promote slot
///    allocas back into SSA
/// 5. In the same modes, unroll annotated loops and clean up changed functions
/// 6. Lower `dialect-mir` → LLVM dialect (via `mir-lower`)
/// 7. Verify the LLVM dialect module
/// 8. Export the LLVM dialect to a `.ll` file (including device extern declarations)
/// 9. Invoke `llc` to generate PTX (or emit LTOIR/NVVM IR when requested)
///
/// # Target Selection
///
/// Automatically detects GPU features (WGMMA, TMA, tcgen05) and selects
/// an appropriate SM target. Can be overridden via `CUDA_OXIDE_TARGET`.
///
/// # Device Externs
///
/// External device function declarations (from `#[device] extern "C" { ... }`)
/// are emitted as LLVM `declare` statements. These are resolved at link time
/// by nvJitLink when linking with external LTOIR (e.g., CCCL libraries).
///
/// # Known Defs
///
/// `known_defs` carries lang-item `DefId`s (FnOnce::Output, Index, IndexMut)
/// resolved by the driver, which holds the `TyCtxt` this crate lacks. The
/// ids are only valid inside the caller's `rustc_internal::run` context.
/// Pass `Default::default()` when no functions need type translation.
///
/// # Errors
///
/// Returns [`PipelineError`] with details on which step failed.
pub fn run_pipeline(
    functions: &[CollectedFunction],
    device_externs: &[DeviceExternDecl],
    config: &PipelineConfig,
    known_defs: crate::translator::facts::KnownDefs,
) -> Result<CompilationResult, PipelineError> {
    import_pipeline_with_libraries(
        functions,
        device_externs,
        config,
        known_defs,
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        Default::default(),
    )?
    .finish()
}

/// An imported module and the configuration captured for its compilation.
///
/// The returned state retains no rustc-facing identities. Preparation borrows
/// this owner and takes its completion data once; the paused stage never owns
/// references into itself.
#[doc(hidden)]
#[must_use = "finish the imported module to emit its artifact"]
pub struct ImportedModule {
    ctx: Context,
    module_op_ptr: pliron::context::Ptr<Operation>,
    config: PipelineConfig,
    backend_options: BackendOptions,
    device_externs: Vec<DeviceExternDecl>,
    legaliser: Legaliser,
    symbols: BTreeMap<String, String>,
    marker_exports: BTreeMap<String, String>,
    source_targets: BTreeMap<crate::LibraryCallSite, BTreeSet<String>>,
    callback_sources: crate::LibraryCallbackSources,
    outputs: ImportOutputs,
    completion: Option<ImportCompletion>,
}

struct ImportOutputs {
    llvm_ir: PathBuf,
    ptx: PathBuf,
    stale: Vec<PathBuf>,
}

struct ImportCompletion {
    kernel_names: Vec<String>,
    kernel_launch_bounds: BTreeMap<String, KernelLaunchBounds>,
    library_roots: BTreeMap<String, oxide_artifacts::KernelLaunchFacts>,
}

/// An imported module paused at the mandatory library-analysis boundary.
///
/// Normal builds have promoted locals; Full debug keeps the original slots.
/// The importer owner must outlive this consuming stage. Its saved ordinary
/// resource constraints survive preparation even when operations are removed.
#[doc(hidden)]
#[must_use = "finish the prepared import to emit its artifact"]
pub struct PreparedImport<'a> {
    stage: PreparedModuleStage<'a, 'a>,
    config: &'a PipelineConfig,
    outputs: &'a ImportOutputs,
    completion: ImportCompletion,
    ordinary: crate::kernel_resources::OrdinaryResources,
    legaliser: &'a mut Legaliser,
    symbols: BTreeMap<String, String>,
    marker_exports: BTreeMap<String, String>,
    source_targets: BTreeMap<crate::LibraryCallSite, BTreeSet<String>>,
    callback_sources: crate::LibraryCallbackSources,
    original_roles: BTreeMap<crate::LibraryCallSite, dialect_mir::library::LibraryRoleAttr>,
    library_rooted_functions: BTreeMap<String, BTreeSet<String>>,
    library_validation: Option<mir_transforms::library::Validation>,
    callback_symbols: BTreeMap<String, String>,
    libraries_bound: bool,
    library_calls: BTreeMap<crate::LibraryCallSite, crate::LibraryCallPlan>,
    library_externs: Vec<DeviceExternDecl>,
}

impl ImportedModule {
    /// Prepare this module once, preserving the ordinary resource constraints.
    pub fn prepare(&mut self) -> Result<PreparedImport<'_>, PipelineError> {
        let completion = self.completion.take().ok_or_else(|| {
            PipelineError::Translation("this imported module was already prepared".into())
        })?;
        let ordinary = crate::kernel_resources::capture_ordinary(
            &self.ctx,
            self.module_op_ptr,
            completion.kernel_names.clone(),
        )
        .map_err(PipelineError::Translation)?;
        let original_roles = mir_transforms::library::capture_library_roles(
            self.module_op_ptr,
            &self.ctx,
            &self.symbols,
        )
        .map_err(|error| PipelineError::Translation(error.disp(&self.ctx).to_string()))?;
        let request = ModulePipelineRequest::for_rust_pipeline(
            &self.device_externs,
            self.config.emit_nvvm_ir,
            &self.backend_options,
            self.config.debug_kind,
            OutputFiles {
                llvm_ir: &self.outputs.llvm_ir,
                ptx: &self.outputs.ptx,
                stale_before_export: &self.outputs.stale,
            },
            PipelineTrace {
                verbose: self.config.verbose,
                dump_mir: self.config.show_mir_dialect,
                dump_llvm: self.config.show_llvm_dialect,
                sink: Some(stderr_pipeline_trace),
            },
        );
        let stage = prepare_translated_module(&mut self.ctx, self.module_op_ptr, &request)?;
        Ok(PreparedImport {
            stage,
            config: &self.config,
            outputs: &self.outputs,
            completion,
            ordinary,
            legaliser: &mut self.legaliser,
            symbols: std::mem::take(&mut self.symbols),
            marker_exports: std::mem::take(&mut self.marker_exports),
            source_targets: std::mem::take(&mut self.source_targets),
            callback_sources: std::mem::take(&mut self.callback_sources),
            callback_symbols: BTreeMap::new(),
            original_roles,
            library_rooted_functions: BTreeMap::new(),
            library_validation: None,
            libraries_bound: false,
            library_calls: BTreeMap::new(),
            library_externs: Vec::new(),
        })
    }

    /// Prepare, lower library operations and emit the captured compilation.
    pub fn finish(mut self) -> Result<CompilationResult, PipelineError> {
        self.prepare()?.finish()
    }
}

impl PreparedImport<'_> {
    /// Admit source-root geometry before native provider or layout work.
    pub fn admit_library_geometry(
        &mut self,
        rooted_functions: &BTreeMap<String, BTreeSet<String>>,
    ) -> Result<BTreeMap<String, [u32; 3]>, PipelineError> {
        let (ctx, module, _) = self.stage.ir();
        mir_transforms::library::admit_library_roots(
            module,
            ctx,
            rooted_functions,
            &self.symbols,
            &self.original_roles,
        )
        .map_err(|error| PipelineError::Translation(error.disp(ctx).to_string()))
    }

    /// Prove the prepared library program before executing its native provider.
    pub fn validate_libraries(
        &mut self,
        rooted_functions: &BTreeMap<String, BTreeSet<String>>,
    ) -> Result<BTreeMap<String, [u32; 3]>, PipelineError> {
        if self.library_validation.is_some() || self.libraries_bound {
            return Err(PipelineError::Translation(
                "this prepared import already has its library proof".into(),
            ));
        }
        let shapes = self.admit_library_geometry(rooted_functions)?;
        let (ctx, module, analyses) = self.stage.ir();
        let callback_symbols = mir_transforms::library::admit_library_callbacks(
            module,
            ctx,
            &self.callback_sources,
            &self.symbols,
        )
        .map_err(PipelineError::Translation)?;
        let retained = callback_symbols.values().cloned().collect::<Vec<_>>();
        let module_op = Operation::get_op::<pliron::builtin::ops::ModuleOp>(module, ctx)
            .expect("imported module");
        llvm_export::export::retain_functions(ctx, &module_op, &retained);
        self.callback_symbols = callback_symbols
            .into_iter()
            .map(|(identity, symbol)| {
                (
                    identity,
                    llvm_export::export::exported_function_name(&symbol),
                )
            })
            .collect();
        let validation = mir_transforms::library::analyze_libraries(
            module,
            ctx,
            analyses,
            rooted_functions,
            &self.symbols,
            &self.original_roles,
            &shapes,
            None,
        )
        .map_err(|error| PipelineError::Translation(error.disp(ctx).to_string()))?;
        self.library_rooted_functions = rooted_functions.clone();
        self.library_validation = Some(validation);
        Ok(shapes)
    }

    /// Actual emitted thunk names available only after callback admission.
    pub fn callback_symbols(&self) -> &BTreeMap<String, String> {
        &self.callback_symbols
    }

    /// Join provider layouts with the proof, then bind the final shared plan.
    pub fn finish_library_planning(
        &mut self,
        native_plans: &mir_transforms::library::RootedNativePlans,
        externs: Vec<DeviceExternDecl>,
        roots: BTreeMap<String, oxide_artifacts::KernelLaunchFacts>,
    ) -> Result<(), PipelineError> {
        let validation = self.library_validation.take().ok_or_else(|| {
            PipelineError::Translation("native library planning requires its source proof".into())
        })?;
        let mut plans =
            mir_transforms::library::allocate_library_storage(&validation, native_plans)
                .map_err(PipelineError::Translation)?;
        let (ctx, module, analyses) = self.stage.ir();
        mir_transforms::library::analyze_libraries(
            module,
            ctx,
            analyses,
            &self.library_rooted_functions,
            &self.symbols,
            &self.original_roles,
            &validation.shapes,
            Some(&mut plans),
        )
        .map_err(|error| PipelineError::Translation(error.disp(ctx).to_string()))?;
        let assembly = mir_transforms::library::assemble_library_plans(
            module,
            ctx,
            &self.library_rooted_functions,
            &self.source_targets,
            &validation,
            &plans,
            native_plans,
            roots,
        )
        .map_err(PipelineError::Translation)?;
        self.bind_libraries(
            assembly.contexts,
            assembly.calls,
            assembly.redirects,
            externs,
            assembly.roots,
        )
    }

    /// Bind the actual provider/layout result and materialize its helper contexts.
    /// Source names and physical symbols remain distinct throughout this step.
    fn bind_libraries(
        &mut self,
        contexts: BTreeMap<String, String>,
        mut calls: BTreeMap<crate::LibraryCallSite, crate::LibraryCallPlan>,
        redirects: crate::CallTargetRedirects,
        externs: Vec<DeviceExternDecl>,
        roots: BTreeMap<String, oxide_artifacts::KernelLaunchFacts>,
    ) -> Result<(), PipelineError> {
        if self.libraries_bound {
            return Err(PipelineError::Translation(
                "this prepared import already has its library plan".into(),
            ));
        }
        for name in contexts.keys() {
            self.symbols
                .entry(name.clone())
                .or_insert_with(|| self.legaliser.legalise(name).to_string());
        }
        // Resource tags use the physical module names. Source sites and
        // authenticated redirect identities keep their raw-name conventions.
        for call in calls.values_mut() {
            if let Some(root) = &mut call.planned_root {
                *root = self.symbols.get(root).cloned().ok_or_else(|| {
                    PipelineError::Translation("planned root has no imported symbol".into())
                })?;
            }
        }
        let physical_contexts = contexts
            .iter()
            .map(|(emitted, source)| {
                let emitted = self.symbols.get(emitted).cloned();
                let source = self.symbols.get(source).cloned();
                match (emitted, source) {
                    (Some(emitted), Some(source)) => Ok((emitted, source)),
                    _ => Err(PipelineError::Translation(
                        "library context has no imported source symbol".into(),
                    )),
                }
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let materialize = !contexts.is_empty()
            || !redirects.is_empty()
            || calls
                .values()
                .any(|plan| plan.operation == crate::LibraryOperation::Redirect);
        if materialize {
            let raw_names: BTreeMap<_, _> = self
                .symbols
                .iter()
                .map(|(raw, symbol)| (symbol.clone(), raw.clone()))
                .collect();
            crate::kernel_resources::bind_ordinary_contexts(
                &mut self.ordinary,
                &physical_contexts,
                |site, caller, callee| {
                    let raw_caller = raw_names
                        .get(caller)
                        .ok_or("saved resource caller has no original name")?;
                    mir_transforms::library::resolve_library_call_target(
                        site,
                        raw_caller,
                        callee,
                        &self.symbols,
                        &self.source_targets,
                        &self.marker_exports,
                        &calls,
                        &redirects,
                    )
                },
            )
            .map_err(PipelineError::Translation)?;
        }
        crate::kernel_resources::validate_ordinary_compatibility(&self.ordinary, &calls)
            .map_err(PipelineError::Translation)?;
        let (ctx, module, analyses) = self.stage.ir();
        if materialize {
            mir_transforms::library::materialize_library_contexts(
                module,
                ctx,
                analyses,
                &contexts,
                &self.symbols,
                &self.source_targets,
                &self.marker_exports,
                &calls,
                &redirects,
            )
            .map_err(|error| PipelineError::Translation(error.disp(ctx).to_string()))?;
        }
        let physical_roots = roots
            .into_iter()
            .map(|(root, facts)| {
                self.symbols
                    .get(&root)
                    .cloned()
                    .map(|symbol| (symbol, facts))
                    .ok_or_else(|| {
                        PipelineError::Translation("library root has no imported symbol".into())
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let functions: Vec<_> = module
            .deref(ctx)
            .regions()
            .flat_map(|region| {
                region
                    .deref(ctx)
                    .iter(ctx)
                    .flat_map(|block| block.deref(ctx).iter(ctx))
            })
            .filter_map(|op| dialect_mir::ops::MirFuncOp::wrap(ctx, op))
            .collect();
        for function in functions {
            let op = function.get_operation();
            let symbol = function.get_symbol_name(ctx).to_string();
            if let Some(plan) = physical_roots.get(&symbol) {
                crate::kernel_resources::apply_library_block(ctx, op, plan)
                    .map_err(PipelineError::Translation)?;
            }
            if self.config.debug_kind.variables_enabled() {
                // The clone keeps static identities and source scopes. Only
                // the containing emitted function's linkage identity changes.
                attach_debug_global_variables(ctx, op, &self.config.debug_global_variables);
            }
        }
        self.library_calls = calls;
        self.library_externs = externs;
        self.completion.library_roots = physical_roots;
        self.libraries_bound = true;
        Ok(())
    }

    /// Lower the validated library plan and finalize resources exactly once.
    pub fn finish(self) -> Result<CompilationResult, PipelineError> {
        if !self.libraries_bound
            && self.original_roles.values().any(|role| {
                !matches!(
                    role,
                    dialect_mir::library::LibraryRoleAttr::WarpGroupIndex { .. }
                )
            })
        {
            return Err(PipelineError::Translation(
                "native library operations require a finalized library plan".into(),
            ));
        }
        if !self.libraries_bound {
            // Direct callers have no native plan to bind. Library callers
            // validate the same saved ordinary facts during binding.
            crate::kernel_resources::validate_ordinary_compatibility(
                &self.ordinary,
                &self.library_calls,
            )
            .map_err(PipelineError::Translation)?;
        }
        let Self {
            stage,
            config,
            outputs,
            completion,
            ordinary,
            library_calls,
            library_externs,
            ..
        } = self;
        let mut stage = stage.bind_device_externs(library_externs)?;
        let (ctx, module, analyses) = stage.ir();
        mir_transforms::library::lower_planned_library_operations(
            module,
            ctx,
            analyses,
            &library_calls,
        )
        .map_err(|error| PipelineError::Translation(error.disp(ctx).to_string()))?;
        let kernel_launch_facts = crate::kernel_resources::complete_planned(
            ctx,
            module,
            ordinary,
            &completion.library_roots,
        )
        .map_err(PipelineError::Translation)?;
        // Resource relocation changes MIR addresses and root attributes.
        let mut invalidated = pliron::pass::PassResult::default();
        invalidated.ir_changed = pliron::irbuild::IRStatus::Changed;
        analyses.retain_preserved(&invalidated);
        let generated = stage.finish()?;
        let artifact_kind = match generated.artifact_kind {
            ModuleArtifactKind::NvvmIr => {
                write_nvvm_compile_options_sidecar(
                    &config.output_dir,
                    &config.output_name,
                    config.allow_fma_contraction,
                    config.debug_kind,
                )?;
                // The target is the completion record for the options sidecar.
                write_nvvm_target_sidecar(
                    &config.output_dir,
                    &config.output_name,
                    &generated.target,
                )?;
                CompilationArtifactKind::NvvmIr
            }
            ModuleArtifactKind::Ptx => CompilationArtifactKind::Ptx,
        };
        Ok(CompilationResult {
            artifact_path: match generated.artifact_kind {
                ModuleArtifactKind::NvvmIr => outputs.llvm_ir.clone(),
                ModuleArtifactKind::Ptx => outputs.ptx.clone(),
            },
            artifact_kind,
            ll_path: outputs.llvm_ir.clone(),
            ptx_path: outputs.ptx.clone(),
            target: generated.target,
            allow_fma_contraction: config.allow_fma_contraction,
            kernel_launch_bounds: completion.kernel_launch_bounds,
            kernel_launch_facts,
        })
    }
}

/// Import and verify functions without starting resource finalization or codegen.
///
/// Rustc identities are used only during import. The returned module owns its
/// context and snapshots configuration, authenticated typed operations and
/// original call identities. Provider plans are bound after the prepared-MIR proof.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn import_pipeline_with_libraries(
    functions: &[CollectedFunction],
    device_externs: &[DeviceExternDecl],
    config: &PipelineConfig,
    known_defs: crate::translator::facts::KnownDefs,
    library_semantics: BTreeMap<crate::LibraryCallSite, crate::LibraryCallMetadata>,
    source_targets: BTreeMap<crate::LibraryCallSite, BTreeSet<String>>,
    selected_call_targets: BTreeMap<
        crate::LibraryCallSite,
        dialect_mir::library::LibraryCallTargetAttr,
    >,
    callback_sources: crate::LibraryCallbackSources,
) -> Result<ImportedModule, PipelineError> {
    crate::library::set_semantics(library_semantics);
    crate::library::set_selected_call_targets(selected_call_targets);
    // Install the driver-resolved lang-item ids and emitted-function identities
    // for this run. Replace (rather than merge) both thread-local sets so
    // stale state from a previous run on this thread cannot survive.
    crate::translator::facts::set_known_defs(known_defs);
    crate::translator::facts::set_emitted_functions(
        functions
            .iter()
            .map(|func| func.instance.mangled_name().to_string())
            .collect(),
    );

    prepare_output_dir(&config.output_dir)?;

    let mut backend_options = backend_options_for(config);
    let cluster_query_target = cluster_query_fallback_target(&backend_options);
    let mut used_cluster_query_fallback = false;

    let mut ctx = Context::new();

    // Step 1: Register dialects
    crate::translator::register_dialects(&mut ctx);

    // Step 2: Create module
    let module_name: pliron::identifier::Identifier = config
        .output_name
        .clone()
        .try_into()
        .unwrap_or_else(|_| "kernel".try_into().unwrap());
    let module = pliron::builtin::ops::ModuleOp::new(&mut ctx, module_name);
    let module_op_ptr = module.get_operation();

    let mut legaliser = Legaliser::default();
    let mut symbols = BTreeMap::new();
    let mut marker_exports = BTreeMap::new();
    let mut kernel_launch_bounds = BTreeMap::new();

    // Step 3: Translate all functions
    for func in functions {
        if config.verbose {
            eprintln!(
                "Translating {}: {}",
                if func.is_kernel {
                    "kernel"
                } else {
                    "device fn"
                },
                func.export_name
            );
        }

        let body = func
            .instance
            .body()
            .ok_or_else(|| PipelineError::NoBody(func.export_name.clone()))?;

        if let Some(source) = &func.library_source {
            if source.export != func.export_name || source.mangled != func.instance.mangled_name() {
                return Err(PipelineError::Translation(format!(
                    "{}: library source identity does not match the imported function",
                    func.export_name
                )));
            }
            let expected_arguments = if func.is_kernel {
                body.arg_locals().len()
            } else {
                0
            };
            if func.library_root_arguments.len() != expected_arguments {
                return Err(PipelineError::Translation(format!(
                    "{}: library argument classes do not match the imported root arguments",
                    func.export_name
                )));
            }
        } else if !func.library_root_arguments.is_empty() {
            return Err(PipelineError::Translation(format!(
                "{}: library argument classes require their source identity",
                func.export_name
            )));
        }

        let (func_op_ptr, used_fallback) = crate::translator::body::translate_body(
            &mut ctx,
            &body,
            &func.instance,
            func.rustc_mir_block_count,
            &func.rustc_mono_successors,
            func.is_kernel,
            func.inline,
            Some(&func.export_name),
            &mut legaliser,
            config.debug_kind,
            func.debug_source_scopes.as_ref(),
            func.statement_debug_info.as_ref(),
            cluster_query_target.is_some(),
        )
        .map_err(|e| {
            // Use .disp(&ctx) for rich error formatting with location and backtrace
            PipelineError::Translation(format!("{}: {}", func.export_name, e.disp(&ctx)))
        })?;
        used_cluster_query_fallback |= used_fallback;

        let imported_function = Operation::get_op::<dialect_mir::ops::MirFuncOp>(func_op_ptr, &ctx)
            .expect("body translation produces a MIR function");
        if let Some(source) = &func.library_source {
            imported_function.set_library_function_source(&mut ctx, source.clone());
        }
        for (index, &class) in func.library_root_arguments.iter().enumerate() {
            imported_function.set_library_root_argument(&mut ctx, index, class);
        }

        symbols.insert(
            func.export_name.clone(),
            legaliser.legalise(&func.export_name).to_string(),
        );
        marker_exports.insert(func.instance.mangled_name(), func.export_name.clone());

        if config.debug_kind.variables_enabled() {
            attach_debug_global_variables(&mut ctx, func_op_ptr, &config.debug_global_variables);
        }

        // Dump the per-function IR BEFORE verification so users can see
        // what the translator produced even when verification fails. If we
        // verified first and bailed, `--show-mir-dialect` / `CUDA_OXIDE_DUMP_MIR`
        // would silently print nothing for the offending function.
        if config.show_mir_dialect {
            eprintln!(
                "\n=== dialect-mir func: {} (pre-verify) ===",
                func.export_name
            );
            eprintln!("{}", func_op_ptr.deref(&ctx).disp(&ctx));
        }

        verify_operation(&ctx, func_op_ptr, &func.export_name)?;

        if func.is_kernel
            && let Some(bounds) = launch_bounds_from_mir_func(&ctx, func_op_ptr)
        {
            kernel_launch_bounds.insert(func.export_name.clone(), bounds);
        }

        // Append to module
        append_to_module(&ctx, module_op_ptr, func_op_ptr);
    }

    // Ordinary Rust calls cross from shared-address-space values to generic
    // pointer parameters. Use the imported callee's real signature, retaining
    // all unsupported mismatches for the normal module verifier.
    crate::translator::terminator::helpers::normalize_internal_shared_pointer_arguments(
        &mut ctx,
        module_op_ptr,
    );
    for declaration in device_externs {
        symbols.insert(
            declaration.export_name.clone(),
            legaliser.legalise(&declaration.export_name).to_string(),
        );
    }

    // Cluster fallback has already selected a target-dependent implementation.
    // Pin its advisory target before preparing the module.
    if used_cluster_query_fallback && backend_options.target_arch.is_none() {
        backend_options.target_arch = cluster_query_target;
        backend_options.target_arch_source =
            "the device hint, pinned by pre-sm90 cluster query fallback";
    }
    let outputs = ImportOutputs {
        llvm_ir: config.output_dir.join(format!("{}.ll", config.output_name)),
        ptx: config
            .output_dir
            .join(format!("{}.ptx", config.output_name)),
        stale: stale_compilation_artifact_paths(&config.output_dir, &config.output_name),
    };
    let kernel_names = functions
        .iter()
        .filter(|function| function.is_kernel)
        .map(|function| symbols[&function.export_name].clone())
        .collect();
    Ok(ImportedModule {
        ctx,
        module_op_ptr,
        config: config.clone(),
        backend_options,
        device_externs: device_externs.to_vec(),
        legaliser,
        symbols,
        marker_exports,
        source_targets,
        callback_sources,
        outputs,
        completion: Some(ImportCompletion {
            kernel_names,
            kernel_launch_bounds,
            library_roots: BTreeMap::new(),
        }),
    })
}

fn launch_bounds_from_mir_func(
    ctx: &Context,
    func_op: pliron::context::Ptr<pliron::operation::Operation>,
) -> Option<KernelLaunchBounds> {
    use pliron::builtin::attributes::IntegerAttr;

    let max_key: pliron::identifier::Identifier = "maxntid".try_into().ok()?;
    let min_key: pliron::identifier::Identifier = "minctasm".try_into().ok()?;
    let attributes = &func_op.deref(ctx).attributes;
    let max_threads = attributes
        .get::<IntegerAttr>(&max_key)
        .and_then(|attribute| u32::try_from(attribute.value().to_u64()).ok())?;
    let min_blocks = attributes
        .get::<IntegerAttr>(&min_key)
        .and_then(|attribute| u32::try_from(attribute.value().to_u64()).ok())
        .filter(|value| *value != 0);

    Some(KernelLaunchBounds {
        max_threads,
        min_blocks,
    })
}

/// Ensures the configured output directory exists before any emission step.
///
/// The pipeline writes every generated artifact under `PipelineConfig::output_dir`.
/// Creating the directory at the pipeline boundary lets callers provide fresh
/// sidecar paths without separately seeding them first.
fn prepare_output_dir(output_dir: &Path) -> Result<(), PipelineError> {
    std::fs::create_dir_all(output_dir).map_err(|e| {
        PipelineError::Export(format!(
            "failed to create output directory {}: {}",
            output_dir.display(),
            e
        ))
    })
}

/// Records the resolved NVVM target alongside the emitted `.ll`.
///
/// The `.target` sidecar carries the completion marker that tells the consumer
/// the sibling `.options` file is present and required. These sidecars are a
/// host artifact concern (`oxide-artifacts`), so they stay in `mir-importer`
/// rather than the rustc-free `cuda-oxide-codegen` backend.
fn write_nvvm_target_sidecar(
    output_dir: &Path,
    output_name: &str,
    target: &str,
) -> Result<(), PipelineError> {
    let path = output_dir.join(format!("{output_name}.target"));
    std::fs::write(
        &path,
        format!(
            "{target}\n{}\n",
            oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
        ),
    )
    .map_err(|error| {
        PipelineError::Export(format!(
            "failed to record NVVM target in {}: {error}",
            path.display()
        ))
    })
}

/// Records the compile-wide FMA and debug policies that downstream libNVVM and
/// nvJitLink stages must preserve, next to the emitted `.ll`.
fn write_nvvm_compile_options_sidecar(
    output_dir: &Path,
    output_name: &str,
    allow_fma_contraction: bool,
    debug_kind: DebugKind,
) -> Result<(), PipelineError> {
    let path = output_dir.join(format!("{output_name}.options"));
    let debug_policy = match debug_kind {
        DebugKind::Off => oxide_artifacts::ArtifactDebugPolicy::None,
        DebugKind::LineTables => oxide_artifacts::ArtifactDebugPolicy::LineTables,
        DebugKind::Full => oxide_artifacts::ArtifactDebugPolicy::Full,
    };
    let options = oxide_artifacts::ArtifactCompileOptions::new()
        .with_fma_contraction(allow_fma_contraction)
        .with_debug_policy(debug_policy);
    std::fs::write(&path, options.sidecar_text()).map_err(|error| {
        PipelineError::Export(format!(
            "failed to record NVVM compile options in {}: {error}",
            path.display()
        ))
    })
}

/// Paths the pipeline may remove or replace when compiling one module.
/// Callers can preflight these before compilation to protect external inputs.
pub fn stale_compilation_artifact_paths(
    output_dir: &Path,
    output_name: &str,
) -> Vec<std::path::PathBuf> {
    [
        "ll",
        "linked.ll",
        "linked.opt.ll",
        "ptx",
        "target",
        "options",
        "ltoir",
        "cubin",
        "cubin.target",
    ]
    .into_iter()
    .map(|suffix| output_dir.join(format!("{output_name}.{suffix}")))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm_export::ops::DebugLocalTypeKind;
    use std::fs;

    #[test]
    fn cluster_query_fallback_requires_a_known_pre90_target() {
        for (explicit, hint, expected) in [
            (None, None, None),
            (None, Some("sm_80"), Some("sm_80")),
            (None, Some("sm_120"), None),
            (None, Some("invalid"), None),
            (Some("sm_80"), Some("sm_120"), Some("sm_80")),
            (Some("sm_120"), Some("sm_80"), None),
            (Some("invalid"), Some("sm_80"), None),
        ] {
            let mut options = BackendOptions::default();
            options.target_arch = explicit.map(str::to_owned);
            options.device_arch_hint = hint.map(|hint| DeviceArchHint::parse(hint.to_owned()));
            assert_eq!(
                cluster_query_fallback_target(&options).as_deref(),
                expected,
                "explicit={explicit:?}, hint={hint:?}"
            );
        }
    }

    #[test]
    fn genuine_warp_query_retains_identity_and_finishes_without_native_plans() {
        use dialect_mir::library::LibraryRoleAttr;
        use dialect_mir::ops::{MirCallOp, MirFuncOp, MirLibraryOp};
        use pliron::common_traits::Verify;
        use rustc_public::{CrateDef, mir};
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_warp_query_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        // A real external definition exercises the module/DefId lookup; the
        // same-spelled local function must remain an ordinary Rust call.
        let dependency = root.join("device.rs");
        fs::write(&dependency, r#"
pub mod library {
    #[inline(never)] pub fn cuda_oxide_library_v1_246e25db_warp_group_index<const WIDTH:usize>() -> u32 { WIDTH as u32 }
}
"#).unwrap();
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let output = std::process::Command::new(&rustc)
            .args([
                "--edition=2024",
                "--crate-type=rlib",
                "--crate-name=cuda_device",
                "--emit=metadata",
            ])
            .arg("--out-dir")
            .arg(&root)
            .arg(&dependency)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let fixture = root.join("query.rs");
        fs::write(&fixture, r#"
extern crate cuda_device;
#[inline(never)] pub fn cuda_oxide_library_v1_246e25db_warp_group_index<const WIDTH:usize>() -> u32 { 77 }
pub fn query() -> u32 { cuda_device::library::cuda_oxide_library_v1_246e25db_warp_group_index::<8>() }
pub fn invalid_width() -> u32 { cuda_device::library::cuda_oxide_library_v1_246e25db_warp_group_index::<3>() }
pub fn lookalike() -> u32 { cuda_oxide_library_v1_246e25db_warp_group_index::<8>() }
"#).unwrap();
        let sysroot = std::process::Command::new(&rustc)
            .args(["--print", "sysroot"])
            .output()
            .unwrap();
        assert!(sysroot.status.success());
        let args = vec![
            "rustc".to_string(),
            "--edition=2024".into(),
            "--crate-type=rlib".into(),
            "--crate-name=query_fixture".into(),
            "--emit=metadata".into(),
            "-Zmir-opt-level=0".into(),
            format!(
                "--extern=cuda_device={}",
                root.join("libcuda_device.rmeta").display()
            ),
            format!("--out-dir={}", root.display()),
            format!(
                "--sysroot={}",
                String::from_utf8(sysroot.stdout).unwrap().trim()
            ),
            fixture.display().to_string(),
        ];
        let output_dir = root.join("output");
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let collected = |name: &str| {
                        let item = rustc_public::all_local_items()
                            .into_iter()
                            .find(|i| i.name() == format!("query_fixture::{name}"))
                            .unwrap();
                        let instance = Instance::try_from(item).unwrap();
                        let body = instance.body().unwrap();
                        CollectedFunction {
                            instance,
                            rustc_mir_block_count: body.blocks.len(),
                            rustc_mono_successors: body
                                .blocks
                                .iter()
                                .map(|b| b.terminator.successors())
                                .collect(),
                            is_kernel: false,
                            export_name: name.into(),
                            library_source: None,
                            library_root_arguments: Vec::new(),
                            debug_source_scopes: None,
                            statement_debug_info: None,
                            inline: None,
                        }
                    };
                    let config = PipelineConfig {
                        output_dir,
                        output_name: "query".into(),
                        verbose: false,
                        emit_nvvm_ir: true,
                        target_arch: Some("sm_100".into()),
                        ..PipelineConfig::default()
                    };
                    let import = |function, config: &PipelineConfig| {
                        import_pipeline_with_libraries(
                            &[function],
                            &[],
                            config,
                            Default::default(),
                            Default::default(),
                            Default::default(),
                            Default::default(),
                            Default::default(),
                        )
                    };
                    let invalid = import(collected("invalid_width"), &config)
                        .err()
                        .expect("invalid genuine width rejected");
                    assert!(
                        invalid.to_string().contains("warp group index width"),
                        "{invalid}"
                    );
                    let fake = import(collected("lookalike"), &config).unwrap();
                    let mut fake_calls = 0;
                    for block in fake
                        .module_op_ptr
                        .deref(&fake.ctx)
                        .get_region(0)
                        .deref(&fake.ctx)
                        .iter(&fake.ctx)
                    {
                        for op in block.deref(&fake.ctx).iter(&fake.ctx) {
                            if Operation::get_op::<MirFuncOp>(op, &fake.ctx).is_none() {
                                continue;
                            }
                            for block in op
                                .deref(&fake.ctx)
                                .get_region(0)
                                .deref(&fake.ctx)
                                .iter(&fake.ctx)
                            {
                                for op in block.deref(&fake.ctx).iter(&fake.ctx) {
                                    assert!(
                                        Operation::get_op::<MirLibraryOp>(op, &fake.ctx).is_none()
                                    );
                                    fake_calls += usize::from(
                                        Operation::get_op::<MirCallOp>(op, &fake.ctx).is_some(),
                                    );
                                }
                            }
                        }
                    }
                    assert_eq!(fake_calls, 1, "same spelling gains no query authority");
                    for debug in [DebugKind::Off, DebugKind::Full] {
                        for bind_empty in [false, true] {
                            let function = collected("query");
                            let body = function.instance.body().unwrap();
                            let site = body
                                .blocks
                                .iter()
                                .position(|b| {
                                    matches!(b.terminator.kind, mir::TerminatorKind::Call { .. })
                                })
                                .unwrap();
                            let mut config = config.clone();
                            config.debug_kind = debug;
                            let mut imported = import(function, &config).unwrap();
                            let mut prepared = imported.prepare().unwrap();
                            assert_eq!(prepared.original_roles.len(), 1);
                            assert_eq!(
                                prepared.original_roles.get(&crate::LibraryCallSite {
                                    caller: "query".into(),
                                    block: site
                                }),
                                Some(&LibraryRoleAttr::WarpGroupIndex { width: 8 })
                            );
                            let (ctx, module, _) = prepared.stage.ir();
                            let mut queries = 0;
                            for block in module.deref(ctx).get_region(0).deref(ctx).iter(ctx) {
                                for op in block.deref(ctx).iter(ctx) {
                                    if Operation::get_op::<MirFuncOp>(op, ctx).is_none() {
                                        continue;
                                    }
                                    for block in op.deref(ctx).get_region(0).deref(ctx).iter(ctx) {
                                        for op in block.deref(ctx).iter(ctx) {
                                            if let Some(query) =
                                                Operation::get_op::<MirLibraryOp>(op, ctx)
                                            {
                                                query.verify(ctx).unwrap();
                                                queries += 1;
                                                assert_eq!(op.deref(ctx).get_num_operands(), 0);
                                                assert_eq!(op.deref(ctx).get_num_results(), 1);
                                                assert!(
                                                    query.get_attr_library_signature(ctx).is_none()
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            assert_eq!(queries, 1, "query survives actual Normal/Full preparation");
                            if bind_empty {
                                let roots = BTreeMap::from([(
                                    "query".into(),
                                    BTreeSet::from(["query".into()]),
                                )]);
                                assert!(prepared.validate_libraries(&roots).unwrap().is_empty());
                                prepared
                                    .finish_library_planning(
                                        &BTreeMap::new(),
                                        Vec::new(),
                                        BTreeMap::new(),
                                    )
                                    .unwrap();
                            }
                            let result = prepared.finish().unwrap();
                            let llvm = fs::read_to_string(result.ll_path).unwrap();
                            for register in ["tid.x", "tid.y", "tid.z", "ntid.x", "ntid.y"] {
                                assert!(
                                    llvm.contains(&format!(
                                        "call i32 @llvm.nvvm.read.ptx.sreg.{register}()"
                                    )),
                                    "{llvm}"
                                );
                            }
                            assert!(llvm.contains("udiv i32"), "{llvm}");
                            assert!(!llvm.contains(
                                "call i32 @cuda_oxide_library_v1_246e25db_warp_group_index"
                            ));
                        }
                    }
                    std::ops::ControlFlow::<(), _>::Continue(())
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("query fixture compiles");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cluster_query_lowering_preserves_target_selection_and_raw_floors() {
        const CHILD: &str = "CUDA_OXIDE_TEST_CLUSTER_QUERY_PIPELINE";
        if std::env::var_os(CHILD).is_none() {
            // Hint-only cases must not inherit an explicit target. Keep
            // environment changes out of the parallel test process.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "pipeline::tests::cluster_query_lowering_preserves_target_selection_and_raw_floors",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env_remove("CUDA_OXIDE_TARGET")
                .env_remove("CUDA_OXIDE_DEVICE_ARCH")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_cluster_queries_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).unwrap();
        let fixture = root.join("queries.rs");
        // These non-inlined placeholders reproduce the defining paths at the
        // importer boundary. Their bodies are not collected: actual dispatch
        // must replace each call, never execute the sentinel value.
        fs::write(
            &fixture,
            r#"
#![allow(dead_code, non_snake_case)]
macro_rules! queries {
    ($($name:ident),*) => {$(
        #[inline(never)] pub fn $name() -> u32 { 99 }
    )*};
}
mod cluster {
    queries!(cluster_ctaidX, cluster_ctaidY, cluster_ctaidZ,
             cluster_nctaidX, cluster_nctaidY, cluster_nctaidZ,
             __cluster_idxX, __cluster_idxY, __cluster_idxZ,
             __cluster_grid_dimX, __cluster_grid_dimY, __cluster_grid_dimZ);
}
mod warp {
    #[inline(never)] pub fn elect_sync(_: u32) -> (u32, bool) { (99, false) }
}
mod __cuda_oxide_intrinsic_abi_v1 { queries!(i0266); }
fn compatibility() -> (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    use cluster::*;
    (cluster_ctaidX(), cluster_ctaidY(), cluster_ctaidZ(),
     cluster_nctaidX(), cluster_nctaidY(), cluster_nctaidZ(),
     __cluster_idxX(), __cluster_idxY(), __cluster_idxZ(),
     __cluster_grid_dimX(), __cluster_grid_dimY(), __cluster_grid_dimZ())
}
fn compatibility_and_elect() -> (u32, u32) {
    (cluster::cluster_nctaidX(), warp::elect_sync(u32::MAX).0)
}
fn elect_only() -> u32 { warp::elect_sync(u32::MAX).0 }
fn raw() -> u32 { __cuda_oxide_intrinsic_abi_v1::i0266() }
"#,
        )
        .unwrap();
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let sysroot = std::process::Command::new(rustc)
            .args(["--print", "sysroot"])
            .output()
            .unwrap();
        assert!(sysroot.status.success());
        let sysroot = String::from_utf8(sysroot.stdout).unwrap();

        for crate_name in ["cuda_device", "cuda_intrinsics"] {
            let args = vec![
                "rustc".to_string(),
                "--edition=2024".to_string(),
                "--crate-type=rlib".to_string(),
                format!("--crate-name={crate_name}"),
                "--emit=metadata".to_string(),
                "-Zmir-opt-level=0".to_string(),
                format!("--out-dir={}", root.display()),
                format!("--sysroot={}", sysroot.trim()),
                fixture.display().to_string(),
            ];
            let output_dir = root.join(crate_name);
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    rustc_public::run!(&args, || {
                        use rustc_public::CrateDef as _;

                        let compile = |name: &str, target: Option<&str>, hint: Option<&str>| {
                            let item = rustc_public::all_local_items()
                                .into_iter()
                                .find(|item| item.name() == format!("{crate_name}::{name}"))
                                .expect("fixture entry");
                            let instance = Instance::try_from(item).unwrap();
                            let body = instance.body().unwrap();
                            let function = CollectedFunction {
                                instance,
                                rustc_mir_block_count: body.blocks.len(),
                                // These monomorphic, branch-free fixtures have
                                // no specialization-dependent dead successors.
                                rustc_mono_successors: body
                                    .blocks
                                    .iter()
                                    .map(|block| block.terminator.successors())
                                    .collect(),
                                is_kernel: false,
                                export_name: name.to_string(),
                                library_source: None,
                                library_root_arguments: Vec::new(),
                                debug_source_scopes: None,
                                statement_debug_info: None,
                                inline: None,
                            };
                            let config = PipelineConfig {
                                output_dir: output_dir.clone(),
                                output_name: name.to_string(),
                                verbose: false,
                                emit_nvvm_ir: true,
                                target_arch: target.map(str::to_string),
                                device_arch_hint: hint
                                    .map(|hint| DeviceArchHint::parse(hint.to_string())),
                                ..PipelineConfig::default()
                            };
                            run_pipeline(&[function], &[], &config, Default::default())
                                .map(|result| {
                                    (result.target, fs::read_to_string(result.ll_path).unwrap())
                                })
                                .map_err(|error| error.to_string())
                        };
                        if crate_name == "cuda_device" {
                            for (target, hint) in
                                [(Some("sm_80"), Some("sm_120")), (None, Some("sm_80"))]
                            {
                                let (target, ir) = compile("compatibility", target, hint).unwrap();
                                assert_eq!(target, "sm_80");
                                for (register_index, register) in
                                    ["ctaid", "nctaid"].iter().enumerate()
                                {
                                    for (axis_index, axis) in ["x", "y", "z"].iter().enumerate() {
                                        let instruction = format!(
                                            "call i32 @llvm.nvvm.read.ptx.sreg.{register}.{axis}()"
                                        );
                                        let value = ir
                                            .lines()
                                            .find_map(|line| {
                                                line.contains(&instruction).then(|| {
                                                    line.trim().split_once(" = ").unwrap().0
                                                })
                                            })
                                            .expect("compatibility register read");
                                        let slot = 6 + register_index * 3 + axis_index;
                                        assert!(
                                            ir.lines().any(|line| {
                                                line.contains("insertvalue")
                                                    && line
                                                        .contains(&format!("i32 {value}, {slot}"))
                                            }),
                                            "{ir}"
                                        );
                                    }
                                }
                                assert!(!ir.contains("sreg.cluster"));
                                assert!(!ir.contains("sreg.ncluster"));
                                for index in 0..6 {
                                    let value = u32::from(index >= 3);
                                    assert!(
                                        ir.lines().any(|line| {
                                            line.contains("insertvalue")
                                                && line.contains(&format!("i32 {value}, {index}"))
                                        }),
                                        "{ir}"
                                    );
                                }
                            }
                            let error = compile("compatibility_and_elect", None, Some("sm_80"))
                                .expect_err("used fallback must pin the advisory hint");
                            assert!(error.contains("sm_80") && error.contains("Sm90"), "{error}");
                            let (target, _) = compile("elect_only", None, Some("sm_80")).unwrap();
                            assert_eq!(target, "sm_90", "unused fallback must not pin the hint");
                            let (target, ir) = compile("compatibility", None, None).unwrap();
                            assert_eq!(target, "sm_90");
                            assert!(
                                ir.contains("call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.x()")
                            );
                        } else {
                            let error = compile("raw", Some("sm_80"), None)
                                .expect_err("raw intrinsics retain their native target floor");
                            assert!(
                                error.contains("v1:i0266") && error.contains("sm_90"),
                                "{error}"
                            );
                            let (target, ir) = compile("raw", Some("sm_120"), None).unwrap();
                            assert_eq!(target, "sm_120");
                            assert!(
                                ir.contains("call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.x()")
                            );
                        }
                        std::ops::ControlFlow::<(), ()>::Continue(())
                    })
                })
                .unwrap()
                .join()
                .unwrap()
                .expect("in-process fixture compilation succeeds");
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_pipeline_config_default_values() {
        let config = PipelineConfig::default();

        assert_eq!(config.output_name, "kernel");
        assert!(config.verbose);
        assert!(!config.show_mir_dialect);
        assert!(!config.show_llvm_dialect);
        assert!(!config.emit_nvvm_ir);
        assert_eq!(config.target_arch, None);
        assert_eq!(config.device_arch_hint, None);
        assert_eq!(config.debug_kind, DebugKind::Off);
    }

    #[test]
    fn static_debug_identity_attaches_to_as1_and_as4_only() {
        use dialect_mir::ops::{MirFuncOp, MirGlobalAllocOp};
        use dialect_mir::types::MirPtrType;
        use pliron::builtin::attributes::{StringAttr, TypeAttr};
        use pliron::builtin::ops::ModuleOp;
        use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
        use pliron::r#type::TypeHandle;

        fn append_global(
            ctx: &mut Context,
            block: pliron::context::Ptr<pliron::basic_block::BasicBlock>,
            result_ty: TypeHandle,
            value_ty: TypeHandle,
            key: &str,
        ) -> pliron::context::Ptr<Operation> {
            let op = Operation::new(
                ctx,
                MirGlobalAllocOp::get_concrete_op_info(),
                vec![result_ty],
                vec![],
                vec![],
                0,
            );
            let global = MirGlobalAllocOp::new(op);
            global.set_attr_global_type(ctx, TypeAttr::new(value_ty));
            global.set_attr_global_key(ctx, StringAttr::new(key.to_string()));
            op.insert_at_back(block, ctx);
            op
        }

        fn info(name: &str, line: i32) -> DebugGlobalVariableInfo {
            DebugGlobalVariableInfo {
                name: name.to_string(),
                namespace: vec!["debug_fixture".to_string(), "kernels".to_string()],
                ty: DebugLocalTypeKind::Basic {
                    name: "u32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_unsigned",
                },
                declaration: DebugSourcePosition {
                    file: std::path::PathBuf::from("/tmp/debug_fixture.rs"),
                    line,
                    column: 1,
                },
                is_local_to_unit: true,
                is_function_local: false,
            }
        }

        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "debug_attachment".try_into().unwrap());
        let module_region = module.get_operation().deref(&ctx).get_region(0);
        let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
        let function_ty = FunctionType::get(&ctx, vec![], vec![]);
        let function_op = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(function_ty.into()));
        function.set_symbol_name(&mut ctx, "kernel".try_into().unwrap());
        function_op.insert_at_back(module_block, &ctx);
        let function_region = function_op.deref(&ctx).get_region(0);
        let function_block = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        function_block.insert_at_back(function_region, &ctx);

        let value_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let as1_ty: TypeHandle = MirPtrType::get_global(&mut ctx, value_ty, false).into();
        let as4_ty: TypeHandle = MirPtrType::get_constant(&mut ctx, value_ty, false).into();
        let as3_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, value_ty, false).into();
        let as0_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, value_ty, false).into();
        let ordinary = append_global(&mut ctx, function_block, as1_ty, value_ty, "ordinary");
        let constant = append_global(&mut ctx, function_block, as4_ty, value_ty, "constant");
        let wrong_address_space = append_global(
            &mut ctx,
            function_block,
            as3_ty,
            value_ty,
            "wrong-address-space",
        );
        let generic_address_space = append_global(
            &mut ctx,
            function_block,
            as0_ty,
            value_ty,
            "generic-address-space",
        );
        let missing_identity = append_global(
            &mut ctx,
            function_block,
            as4_ty,
            value_ty,
            "missing-identity",
        );

        let ordinary_info = info("GLOBAL_COUNTER", 7);
        let constant_info = info("SCALE", 11);
        let wrong_address_space_info = info("NOT_A_STATIC_STORAGE_CLASS", 13);
        let globals = BTreeMap::from([
            ("ordinary".to_string(), ordinary_info.clone()),
            ("constant".to_string(), constant_info.clone()),
            ("wrong-address-space".to_string(), wrong_address_space_info),
            (
                "generic-address-space".to_string(),
                info("MUST_NOT_ATTACH_TO_AS0", 15),
            ),
        ]);

        attach_debug_global_variables(&mut ctx, function_op, &globals);

        assert_eq!(
            llvm_export::ops::debug_global_variable(&ctx, ordinary),
            Some(ordinary_info),
            "ordinary AS1 globals must retain their source identity"
        );
        assert_eq!(
            llvm_export::ops::debug_global_variable(&ctx, constant),
            Some(constant_info),
            "constant-memory AS4 globals must retain their source identity"
        );
        assert!(
            llvm_export::ops::debug_global_variable(&ctx, wrong_address_space).is_none(),
            "the importer gate must not attach static metadata to other pointer address spaces"
        );
        assert!(
            llvm_export::ops::debug_global_variable(&ctx, generic_address_space).is_none(),
            "the importer allow-list must fail closed for AS0 rather than accepting every non-shared address space"
        );
        assert!(
            llvm_export::ops::debug_global_variable(&ctx, missing_identity).is_none(),
            "an uncollected global key must continue to fail closed"
        );
    }

    #[test]
    fn global_and_shared_debug_types_fail_closed_for_unsupported_graphs() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_global_debug_types_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).unwrap();
        let fixture = root.join("global_debug_types.rs");
        fs::write(
            &fixture,
            r#"
#![allow(dead_code)]

union Word {
    value: u32,
    bytes: [u8; 4],
}

struct SharedMarker<T, const N: usize>(core::marker::PhantomData<T>);
unsafe impl<T, const N: usize> Sync for SharedMarker<T, N> {}
struct ContainsPointer { pointer: *mut u32 }

static SCALAR: u64 = 7;
static UNION_VALUE: Word = Word { value: 11 };
static SLICE_VIEW: &[u8] = &[1, 2, 3];
static SHARED_I32: SharedMarker<i32, 32> = SharedMarker(core::marker::PhantomData);
static SHARED_UNION: SharedMarker<Word, 4> = SharedMarker(core::marker::PhantomData);
static SHARED_POINTER: SharedMarker<*mut u32, 4> = SharedMarker(core::marker::PhantomData);
static SHARED_POINTER_STRUCT: SharedMarker<ContainsPointer, 2> = SharedMarker(core::marker::PhantomData);
static SHARED_OPAQUE_POINTER: SharedMarker<*mut ContainsPointer, 4> = SharedMarker(core::marker::PhantomData);

fn same_leaf(flag: bool) {
    if flag {
        static SAME: SharedMarker<i16, 8> = SharedMarker(core::marker::PhantomData);
        let _ = &SAME;
    } else {
        static SAME: SharedMarker<i16, 8> = SharedMarker(core::marker::PhantomData);
        let _ = &SAME;
    }
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
            "--crate-name=global_debug_types".to_string(),
            "--emit=metadata".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let supported = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    use rustc_public::CrateDef as _;

                    let results = rustc_public::local_crate()
                        .statics()
                        .into_iter()
                        .map(|static_def| {
                            let qualified_name = static_def.name().to_string();
                            let identity = DebugGlobalVariableIdentity {
                                name: "FIXTURE_STATIC".to_string(),
                                namespace: vec!["global_debug_types".to_string()],
                                declaration: DebugSourcePosition {
                                    file: std::path::PathBuf::from("global_debug_types.rs"),
                                    line: 1,
                                    column: 1,
                                },
                                is_local_to_unit: true,
                            };
                            (
                                qualified_name,
                                rustc_public::mir::mono::Instance::from(static_def)
                                    .mangled_name()
                                    .to_string(),
                                build_debug_global_variable_info(identity, &static_def.ty())
                                    .is_some(),
                                build_debug_shared_array_variable_info(
                                    DebugGlobalVariableIdentity {
                                        name: "SHARED".to_string(),
                                        namespace: vec!["global_debug_types".to_string()],
                                        declaration: DebugSourcePosition {
                                            file: std::path::PathBuf::from("global_debug_types.rs"),
                                            line: 1,
                                            column: 1,
                                        },
                                        is_local_to_unit: true,
                                    },
                                    &static_def.ty(),
                                ),
                            )
                        })
                        .collect::<Vec<_>>();
                    std::ops::ControlFlow::<(), _>::Continue(results)
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        fs::remove_dir_all(&root).ok();
        let status = |leaf: &str| {
            supported
                .iter()
                .find(|(name, _, _, _)| name.ends_with(&format!("::{leaf}")))
                .unwrap_or_else(|| panic!("fixture static `{leaf}` was not found"))
                .2
        };
        assert!(status("SCALAR"), "a scalar semantic type is exact");
        assert!(
            !status("UNION_VALUE"),
            "unsupported unions must not acquire a misleading byte-storage DIE"
        );
        assert!(
            !status("SLICE_VIEW"),
            "a fat reference must be omitted while the shared type builder describes only one pointer word"
        );
        let shared = |leaf: &str| {
            supported
                .iter()
                .find(|(name, _, _, _)| name.ends_with(&format!("::{leaf}")))
                .unwrap_or_else(|| panic!("fixture static `{leaf}` was not found"))
                .3
                .as_ref()
        };
        let info = shared("SHARED_I32").expect("a supported shared array gets metadata");
        assert!(matches!(
            &info.ty,
            DebugLocalTypeKind::Array {
                size_bits: 1024,
                count: 32,
                element,
                ..
            } if matches!(element.as_ref(), DebugLocalTypeKind::Basic { size_bits: 32, .. })
        ));
        assert!(shared("SHARED_UNION").is_none());
        // Thin pointers with a supported pointee carry a complete
        // `TypedPointer` graph (#1126) and are admitted, directly and as a
        // composite member.
        let pointer_info =
            shared("SHARED_POINTER").expect("a typed thin-pointer element gets metadata");
        assert!(matches!(
            &pointer_info.ty,
            DebugLocalTypeKind::Array {
                size_bits: 256,
                count: 4,
                element,
                ..
            } if matches!(
                element.as_ref(),
                DebugLocalTypeKind::TypedPointer { size_bits: 64, pointee, .. }
                    if matches!(pointee.as_ref(), DebugLocalTypeKind::Basic { size_bits: 32, .. })
            )
        ));
        assert!(shared("SHARED_POINTER_STRUCT").is_some());
        // A composite pointee has no bounded tree, so the pointer falls back
        // to the opaque legacy form and the whole graph is rejected.
        assert!(
            shared("SHARED_OPAQUE_POINTER").is_none(),
            "legacy untyped pointers must be rejected"
        );
        let same_leaf: Vec<_> = supported
            .iter()
            .filter(|(name, _, _, _)| name.ends_with("::same_leaf::SAME"))
            .collect();
        assert_eq!(same_leaf.len(), 2, "both block-local statics are collected");
        assert_eq!(
            same_leaf[0].0, same_leaf[1].0,
            "Stable MIR source paths expose the adversarial collision"
        );
        assert_ne!(
            same_leaf[0].1, same_leaf[1].1,
            "mangled static instances are the injective AS3 join keys"
        );
        assert!(same_leaf.iter().all(|entry| entry.3.is_some()));
    }

    #[test]
    fn device_static_global_keys_distinguish_same_path_block_statics() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_static_global_keys_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).unwrap();
        let fixture = root.join("static_global_keys.rs");
        fs::write(
            &fixture,
            r#"
#![allow(dead_code)]

fn opposite_blocks(select_left: bool) -> u64 {
    if select_left {
        static VALUE: u64 = 11;
        VALUE + VALUE
    } else {
        static VALUE: u64 = 29;
        VALUE + VALUE
    }
}

fn adversarial_export_name() -> u64 {
    #[unsafe(export_name = "__cuda_oxide_promoted_type_collision")]
    static EXPORTED: u64 = 41;
    EXPORTED
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
            "--crate-name=static_global_keys".to_string(),
            "--emit=metadata".to_string(),
            "-Zmir-opt-level=0".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let (identities, exported) = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    use rustc_public::CrateDef as _;

                    let statics = rustc_public::local_crate().statics();
                    let results = statics
                        .iter()
                        .copied()
                        .filter(|static_def| {
                            static_def.name().ends_with("::opposite_blocks::VALUE")
                        })
                        .map(|static_def| {
                            let source_name = static_def.name();
                            let mangled = rustc_public::mir::mono::Instance::from(static_def)
                                .mangled_name()
                                .to_string();
                            let key = device_static_global_key(&static_def);
                            let repeated_key = device_static_global_key(&static_def);
                            let initializer = static_def
                                .eval_initializer()
                                .expect("evaluate block-local static")
                                .read_uint()
                                .expect("read u64 block-local static");
                            (source_name, mangled, key, repeated_key, initializer)
                        })
                        .collect::<Vec<_>>();
                    let exported = statics
                        .into_iter()
                        .find(|static_def| static_def.name().ends_with("::EXPORTED"))
                        .map(|static_def| {
                            let symbol = rustc_public::mir::mono::Instance::from(static_def)
                                .mangled_name()
                                .to_string();
                            let static_key = device_static_global_key(&static_def);
                            let promoted_key =
                                dialect_mir::ops::encode_promoted_global_key(&symbol);
                            (symbol, static_key, promoted_key)
                        })
                        .expect("find adversarial exported static");
                    std::ops::ControlFlow::<(), _>::Continue((results, exported))
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        fs::remove_dir_all(&root).ok();
        assert_eq!(identities.len(), 2, "both block statics must be discovered");
        assert_eq!(
            identities[0].0, identities[1].0,
            "the regression requires the two source display paths to collide"
        );
        let mut values = identities
            .iter()
            .map(|identity| identity.4)
            .collect::<Vec<_>>();
        values.sort_unstable();
        assert_eq!(values, [11, 29], "the statics have distinct storage values");
        for (_, mangled, key, repeated_key, _) in &identities {
            assert_eq!(
                dialect_mir::ops::rust_static_symbol_from_global_key(key),
                Some(mangled.as_str()),
                "the tagged carrier must preserve rustc's exact symbol identity"
            );
            assert_eq!(
                repeated_key, key,
                "repeated references to one definition must reuse its key"
            );
        }
        assert_ne!(
            identities[0].2, identities[1].2,
            "DefPath disambiguators must keep same-path statics distinct"
        );
        assert_eq!(
            exported.0, "__cuda_oxide_promoted_type_collision",
            "rustc must expose the user-controlled export name as its symbol"
        );
        assert_eq!(
            dialect_mir::ops::rust_static_symbol_from_global_key(&exported.1),
            Some(exported.0.as_str())
        );
        assert_ne!(
            exported.1, exported.2,
            "static and promoted origins must remain distinct for an identical payload"
        );
    }

    #[test]
    fn stale_artifact_invalidation_removes_every_competing_output() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_stale_artifacts_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).unwrap();
        for suffix in [
            "ll",
            "linked.ll",
            "linked.opt.ll",
            "ptx",
            "target",
            "options",
            "ltoir",
            "cubin",
            "cubin.target",
        ] {
            fs::write(root.join(format!("kernel.{suffix}")), b"stale").unwrap();
        }
        let cached_cubin =
            root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
        fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
        fs::write(&cached_cubin, b"persistent cache entry").unwrap();

        let config = PipelineConfig {
            output_dir: root.clone(),
            output_name: "kernel".to_string(),
            verbose: false,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: true,
            target_arch: Some("sm_86".to_string()),
            target_arch_source: "PipelineConfig::target_arch",
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            debug_global_variables: BTreeMap::new(),
            allow_fma_contraction: true,
            module_disambiguator: None,
        };
        let result = run_pipeline(&[], &[], &config, Default::default()).expect("pipeline run");

        assert_eq!(result.artifact_kind, CompilationArtifactKind::NvvmIr);
        assert_ne!(fs::read(&result.ll_path).unwrap(), b"stale");
        for suffix in [
            "linked.ll",
            "linked.opt.ll",
            "ptx",
            "ltoir",
            "cubin",
            "cubin.target",
        ] {
            assert!(!root.join(format!("kernel.{suffix}")).exists(), "{suffix}");
        }
        assert_ne!(fs::read(root.join("kernel.target")).unwrap(), b"stale");
        assert_ne!(fs::read(root.join("kernel.options")).unwrap(), b"stale");
        assert_eq!(
            fs::read(&cached_cubin).unwrap(),
            b"persistent cache entry",
            "content-addressed cache entries must survive compiler cleanup"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_pipeline_creates_missing_output_dir_before_export() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_mir_importer_output_dir_{}_{}",
            std::process::id(),
            unique
        ));
        let output_dir = root.join("fresh").join("nested");
        fs::remove_dir_all(&root).ok();
        assert!(!output_dir.exists());

        let config = PipelineConfig {
            output_dir: output_dir.clone(),
            output_name: "empty".to_string(),
            verbose: false,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: true,
            target_arch: Some("sm_86".to_string()),
            target_arch_source: "PipelineConfig::target_arch",
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            debug_global_variables: BTreeMap::new(),
            allow_fma_contraction: true,
            module_disambiguator: None,
        };

        let result = run_pipeline(&[], &[], &config, Default::default()).expect("pipeline run");

        assert!(output_dir.is_dir());
        assert!(result.ll_path.is_file());
        assert_eq!(result.artifact_path, result.ll_path);
        assert_eq!(result.artifact_kind, CompilationArtifactKind::NvvmIr);
        assert_eq!(result.target, "sm_86");
        assert_eq!(
            fs::read_to_string(output_dir.join("empty.target")).unwrap(),
            format!(
                "sm_86\n{}\n",
                oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
            )
        );
        assert_eq!(
            fs::read_to_string(output_dir.join("empty.options")).unwrap(),
            oxide_artifacts::ArtifactCompileOptions::new()
                .with_fma_contraction(true)
                .sidecar_text()
        );

        fs::remove_dir_all(&root).expect("clean up temp output dir");
    }

    #[test]
    fn paused_import_owns_emission_settings_and_externs() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_paused_import_{}_{}",
            std::process::id(),
            unique
        ));
        let mut config = PipelineConfig {
            output_dir: root.clone(),
            output_name: "captured".to_string(),
            verbose: false,
            emit_nvvm_ir: true,
            target_arch: Some("sm_86".to_string()),
            allow_fma_contraction: false,
            ..PipelineConfig::default()
        };
        let mut externs = vec![DeviceExternDecl {
            export_name: "captured_extern".to_string(),
            param_types: vec![],
            return_type: DeviceExternType::Void,
            attrs: DeviceExternAttrs::default(),
        }];
        let first = import_pipeline_with_libraries(
            &[],
            &externs,
            &config,
            Default::default(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            Default::default(),
        )
        .unwrap();
        verify_operation(&first.ctx, first.module_op_ptr, "paused import").unwrap();
        assert!(!root.join("captured.ll").exists());

        // A second import replaces the importer thread-local state. Neither it
        // nor later caller changes may alter the first module's emission.
        config.output_name = "later".to_string();
        config.target_arch = Some("sm_80".to_string());
        config.allow_fma_contraction = true;
        externs.clear();
        let later = run_pipeline(&[], &externs, &config, Default::default()).unwrap();
        drop(config);
        drop(externs);
        let captured = first.finish().unwrap();

        assert_eq!(later.target, "sm_80");
        assert_eq!(captured.target, "sm_86");
        assert_eq!(captured.ll_path, root.join("captured.ll"));
        assert_eq!(captured.artifact_kind, CompilationArtifactKind::NvvmIr);
        assert!(!captured.allow_fma_contraction);
        assert!(
            fs::read_to_string(&captured.ll_path)
                .unwrap()
                .contains("declare void @captured_extern()")
        );
        assert!(
            !fs::read_to_string(&later.ll_path)
                .unwrap()
                .contains("captured_extern")
        );
        let options = fs::read_to_string(root.join("captured.options")).unwrap();
        assert!(
            !oxide_artifacts::ArtifactCompileOptions::from_sidecar_text(&options)
                .unwrap()
                .fma_contraction_enabled()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nvvm_sidecar_preserves_deferred_debug_policy() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_nvvm_debug_options_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).unwrap();

        for (name, debug_kind, expected_debug) in [
            (
                "off",
                DebugKind::Off,
                oxide_artifacts::ArtifactDebugPolicy::None,
            ),
            (
                "lines",
                DebugKind::LineTables,
                oxide_artifacts::ArtifactDebugPolicy::LineTables,
            ),
            (
                "full",
                DebugKind::Full,
                oxide_artifacts::ArtifactDebugPolicy::Full,
            ),
        ] {
            write_nvvm_compile_options_sidecar(&root, name, false, debug_kind).unwrap();
            let text = fs::read_to_string(root.join(format!("{name}.options"))).unwrap();
            let options =
                oxide_artifacts::ArtifactCompileOptions::from_sidecar_text(&text).unwrap();
            assert!(!options.fma_contraction_enabled());
            assert_eq!(options.debug_policy(), expected_debug);
        }

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn structured_device_extern_survives_pre_lowering_insertion() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_mir_importer_extern_{}_{}",
            std::process::id(),
            unique
        ));
        let config = PipelineConfig {
            output_dir: root.clone(),
            output_name: "extern_only".to_string(),
            verbose: false,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: true,
            target_arch: Some("sm_86".to_string()),
            target_arch_source: "PipelineConfig::target_arch",
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            debug_global_variables: BTreeMap::new(),
            allow_fma_contraction: true,
            module_disambiguator: None,
        };
        let externs = [DeviceExternDecl {
            export_name: "consume_float".to_string(),
            param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 0)],
            return_type: DeviceExternType::Void,
            attrs: DeviceExternAttrs::default(),
        }];

        let result =
            run_pipeline(&[], &externs, &config, Default::default()).expect("pipeline run");
        let ir = fs::read_to_string(result.ll_path).expect("read exported IR");
        assert!(
            ir.contains("declare void @consume_float(float*)"),
            "structured pointee must survive through export:\n{ir}"
        );
        assert!(
            !ir.split(|c: char| !c.is_ascii_alphanumeric())
                .any(|token| token == "ptr"),
            "legacy device-extern output must not contain opaque pointers:\n{ir}"
        );

        fs::remove_dir_all(&root).expect("clean up temp output dir");
    }

    #[test]
    fn an_explicit_config_target_is_labelled_as_its_own_source() {
        let config = PipelineConfig {
            target_arch: Some("sm_86".to_string()),
            ..PipelineConfig::default()
        };
        let options = backend_options_for(&config);
        assert_eq!(options.target_arch.as_deref(), Some("sm_86"));
        assert_eq!(options.target_arch_source, "PipelineConfig::target_arch");
    }

    #[test]
    fn a_caller_supplied_label_survives_the_override() {
        let config = PipelineConfig {
            target_arch: Some("sm_86".to_string()),
            target_arch_source: "cargo oxide --arch",
            ..PipelineConfig::default()
        };
        assert_eq!(
            backend_options_for(&config).target_arch_source,
            "cargo oxide --arch"
        );
    }

    #[test]
    fn the_env_label_stands_when_the_config_sets_no_target() {
        let config = PipelineConfig::default();
        assert_eq!(config.target_arch, None);
        assert_eq!(
            backend_options_for(&config).target_arch_source,
            "CUDA_OXIDE_TARGET"
        );
    }
}
