/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared post-translation compiler pipeline.
//!
//! Frontends stop after assembling a `dialect-mir` module. This module owns
//! every destructive compiler stage after that boundary so the rustc frontend
//! and the standalone frontend cannot silently diverge.

use crate::error::PipelineError;
use crate::export::{
    DebugExport, DeviceExternDecl, export_llvm_ir, function_local_static_placement_for_llc,
    module_uses_libdevice, render_llvm_ir, resolve_nvvm_target_with_generated,
    unresolved_external_symbols, unresolved_libdevice_ptx_declarations,
    validate_nvvm_debug_support,
};
use crate::generated::{
    GeneratedMarkerPolicy, collect_generated_intrinsic_requirements_for_backend,
};
use crate::generated_intrinsic_targets::GeneratedIntrinsicBackend;
use crate::iket::{has_iket_operations, materialize as materialize_iket, strip as strip_iket};
use crate::llvm_tools::LlvmToolchain;
use crate::lower::{add_device_extern_declarations, lower_to_llvm};
use crate::options::{BackendOptions, IketInstrumentation};
use crate::prep::{MirPreparation, prepare_mir_module};
use crate::ptx::{
    PtxModule, discover_llvm_toolchain, generate_ptx_discovered, generate_ptx_with_toolchain,
};
use crate::target::detect_features_in_llvm_text;
use crate::verify::verify_operation;
use llvm_export::export::{DebugKind, FunctionLocalStaticPlacement, NvvmIrDialect};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::printable::Printable;
use std::path::{Path, PathBuf};

/// Files used to publish one compiler result.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct OutputFiles<'a> {
    /// Textual LLVM IR output.
    pub llvm_ir: &'a Path,
    /// PTX output. This remains absent for an NVVM IR result.
    pub ptx: &'a Path,
    /// Older competing artifacts to remove immediately before export.
    pub stale_before_export: &'a [PathBuf],
}

/// Legacy command-line tracing requested by the rustc frontend.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineTrace {
    /// Print compiler-stage progress.
    pub verbose: bool,
    /// Print the module before and after MIR preparation.
    pub dump_mir: bool,
    /// Print the lowered module before final verification.
    pub dump_llvm: bool,
    /// Receives trace lines and legacy diagnostics. `None` keeps the backend quiet.
    pub sink: Option<fn(&str)>,
}

impl PipelineTrace {
    fn emit(&self, message: impl AsRef<str>) {
        if let Some(sink) = self.sink {
            sink(message.as_ref());
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OutputPolicy {
    SelfContainedPtx {
        /// Whether `__nv_*` symbols may be left for the IR-level libdevice link.
        allow_libdevice: bool,
    },
    ExternalLinkAllowed {
        request_nvvm_ir: bool,
    },
}

#[derive(Clone, Copy, Debug)]
enum ToolchainPolicy<'a> {
    Discover,
    Explicit(&'a LlvmToolchain),
}

/// Backend controls whose device hint has passed the shared input boundary.
/// Fields and construction stay private so controls and proof cannot disagree.
pub(crate) struct ValidatedBackendOptions<'a> {
    options: BackendOptions,
    device_arch: Option<&'a cuda_target_spec::DeviceArch>,
}

impl<'a> ValidatedBackendOptions<'a> {
    fn new(options: &'a BackendOptions) -> Result<Self, PipelineError> {
        let device_arch = options
            .device_arch_hint
            .as_ref()
            .map(crate::options::DeviceArchHint::as_device_arch)
            .transpose()?;
        // Downstream controls contain no raw hint; selection uses only the proof.
        let options = BackendOptions {
            device_arch_hint: None,
            ..options.clone()
        };
        Ok(Self {
            options,
            device_arch,
        })
    }

    pub(crate) fn options(&self) -> &BackendOptions {
        &self.options
    }

    pub(crate) fn device_arch(&self) -> Option<&cuda_target_spec::DeviceArch> {
        self.device_arch
    }

    #[cfg(test)]
    pub(crate) fn for_test(options: &'a BackendOptions) -> Result<Self, PipelineError> {
        Self::new(options)
    }
}

/// Complete request for the shared post-translation pipeline.
#[doc(hidden)]
pub struct ModulePipelineRequest<'a> {
    device_externs: &'a [DeviceExternDecl],
    output_policy: OutputPolicy,
    backend: &'a BackendOptions,
    debug_kind: DebugKind,
    toolchain: ToolchainPolicy<'a>,
    files: OutputFiles<'a>,
    trace: PipelineTrace,
    generated_marker_policy: GeneratedMarkerPolicy,
}

impl<'a> ModulePipelineRequest<'a> {
    /// Build the link-capable request used by the rustc frontend.
    #[doc(hidden)]
    pub fn for_rust_pipeline(
        device_externs: &'a [DeviceExternDecl],
        request_nvvm_ir: bool,
        backend: &'a BackendOptions,
        debug_kind: DebugKind,
        files: OutputFiles<'a>,
        trace: PipelineTrace,
    ) -> Self {
        Self {
            device_externs,
            output_policy: OutputPolicy::ExternalLinkAllowed { request_nvvm_ir },
            backend,
            debug_kind,
            toolchain: ToolchainPolicy::Discover,
            files,
            trace,
            generated_marker_policy: GeneratedMarkerPolicy::Required,
        }
    }

    pub(crate) fn for_standalone_ptx(
        backend: &'a BackendOptions,
        debug_kind: DebugKind,
        toolchain: &'a LlvmToolchain,
        allow_libdevice: bool,
        files: OutputFiles<'a>,
    ) -> Self {
        Self {
            device_externs: &[],
            output_policy: OutputPolicy::SelfContainedPtx { allow_libdevice },
            backend,
            debug_kind,
            toolchain: ToolchainPolicy::Explicit(toolchain),
            files,
            trace: PipelineTrace::default(),
            generated_marker_policy: GeneratedMarkerPolicy::Optional,
        }
    }
}

/// Artifact format emitted by the shared pipeline.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleArtifactKind {
    /// Textual PTX generated by `llc`.
    Ptx,
    /// NVVM-compatible LLVM IR for a later libNVVM/nvJitLink step.
    NvvmIr,
}

/// Successful shared-pipeline result.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModulePipelineOutput {
    /// Format written to the requested artifact files.
    pub artifact_kind: ModuleArtifactKind,
    /// Concrete target selected from explicit options, the device hint, or IR features.
    pub target: String,
    /// Messages retained for the legacy CLI to print.
    pub diagnostics: Vec<String>,
}

/// Compile an already-translated `dialect-mir` module to PTX or NVVM IR.
#[doc(hidden)]
pub fn compile_translated_module(
    ctx: &mut Context,
    module: Ptr<Operation>,
    request: &ModulePipelineRequest<'_>,
) -> Result<ModulePipelineOutput, PipelineError> {
    let mut validated = ValidatedBackendOptions::new(request.backend)?;

    if request.trace.dump_mir {
        request
            .trace
            .emit("\n=== dialect-mir module (pre-verify) ===");
        request
            .trace
            .emit(format!("{}", module.deref(ctx).disp(ctx)));
    }
    if request.trace.verbose {
        request.trace.emit("\n=== Verifying dialect-mir module ===");
    }

    // `CUDA_OXIDE_IKET=off` strips the semantic annotation ops up front,
    // before the full-debug preparation gate below: a disabled build of an
    // annotated kernel must compile in every configuration, including a
    // full-debug build in which the unstripped IKET operations would
    // otherwise be rejected before the policy was ever consulted.
    if request.backend.iket == IketInstrumentation::Disabled {
        strip_iket(ctx, module)?;
    }

    // IKET's placeholder ABI is keyed by the concrete sm_* family, but the
    // definitive target is normally resolved only after LLVM lowering
    // (`generate_ptx_discovered` on the PTX path, `resolve_nvvm_target_with_generated`
    // on the NVVM path), where a device hint that cannot lower a detected
    // feature is silently raised to the feature floor. Materializing from
    // the pre-resolution hint could then bake a placeholder shape for a
    // family the module is never compiled for. So when IKET operations are
    // present, promote the hint to the pipeline's explicit target: both
    // resolvers honor an explicit target exactly (they validate it and fail
    // loudly instead of raising it), so the placeholder shape and the
    // compiled target can no longer diverge.
    if validated.options.target_arch.is_none()
        && validated.device_arch.is_some()
        && has_iket_operations(ctx, module)
    {
        validated.options.target_arch = validated.device_arch.map(|arch| arch.as_ref().sm());
        validated.options.target_arch_source = "the detected GPU, pinned by IKET materialization";
    }
    let backend = validated.options();

    let promote_and_unroll = !request.debug_kind.variables_enabled();
    if !promote_and_unroll && has_iket_operations(ctx, module) {
        return Err(PipelineError::Lowering(
            "IKET requires MIR preparation; full variable debug information is not supported for an instrumented kernel"
                .to_owned(),
        ));
    }
    if request.trace.verbose {
        if promote_and_unroll {
            request
                .trace
                .emit("\n=== Running shared mem2reg + annotated loop-unroll preparation ===");
        } else {
            request
                .trace
                .emit("\n=== Skipping mem2reg (full debug keeps locals in memory) ===");
        }
    }
    prepare_mir_module(
        ctx,
        module,
        MirPreparation {
            promote_and_unroll,
            verbose: request.trace.verbose,
            mir_pass_pipeline: backend.mir_pass_pipeline.as_deref(),
        },
    )?;
    if request.trace.verbose {
        request.trace.emit("dialect-mir preparation successful ✓");
    }
    if request.trace.dump_mir && promote_and_unroll {
        request
            .trace
            .emit("\n=== dialect-mir module (after preparation) ===");
        request
            .trace
            .emit(format!("{}", module.deref(ctx).disp(ctx)));
    }

    // After the pinning above, `backend.target_arch` is exactly the target
    // the rest of the pipeline will compile for (or `None`, in which case
    // materialization fails with its explicit-target requirement).
    materialize_iket(ctx, module, backend.target_arch.as_deref(), &backend.iket)?;

    // Calls need structured extern declarations before lowering so pointer
    // address spaces are preserved by the call converter.
    if !request.device_externs.is_empty() {
        if request.trace.verbose {
            request.trace.emit(format!(
                "\n=== Adding {} device extern declarations ===",
                request.device_externs.len()
            ));
        }
        add_device_extern_declarations(ctx, module, request.device_externs)?;
    }

    // IKET materialization and extern insertion both run after the ordinary
    // MIR preparation gates. Verify the resulting mixed module before backend
    // selection and lowering so no late producer can bypass MIR pointer-kind
    // or call-signature invariants.
    verify_operation(ctx, module, "module immediately before MIR lowering")?;

    // Discover libdevice.10.bc early so the backend decision can account for
    // IR-level linking. When available, `needs_libdevice` no longer forces the
    // NVVM IR path — the PTX path links libdevice at the LLVM IR level instead.
    // "Available" requires BOTH the bitcode file and a same-major `llvm-link`
    // for the `llc` this compilation will use: deciding from file existence
    // alone would commit to the PTX path and then ship PTX with unresolved
    // `.extern .func __nv_*` (a cuModuleLoad-time failure) whenever the
    // linker is missing. Without either piece, libdevice kernels fall back
    // to the NVVM IR path as they did before IR-level linking existed.
    let libdevice_path = libnvvm_sys::find_libdevice().ok();
    let can_ir_link_libdevice = libdevice_path.is_some()
        && match request.toolchain {
            ToolchainPolicy::Discover => crate::llvm_tools::libdevice_ir_linking_available(backend),
            ToolchainPolicy::Explicit(toolchain) => toolchain.llvm_link.is_some(),
        };

    // Choose the intrinsic ABI while calls are still typed MIR. Generated
    // intrinsics may have different LLVM signatures for `llc` and libNVVM, so
    // discovering NVVM mode only after lowering is too late.
    let backend_selection =
        select_pre_lowering_backend(ctx, module, request.output_policy, can_ir_link_libdevice);
    let generated_requirements = collect_generated_intrinsic_requirements_for_backend(
        ctx,
        module,
        request.generated_marker_policy,
        match backend_selection.intrinsic_backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => GeneratedIntrinsicBackend::LlvmNvptx,
            mir_lower::IntrinsicBackend::LibNvvm => GeneratedIntrinsicBackend::LibNvvm,
        },
    )?;

    if request.trace.verbose {
        request
            .trace
            .emit("\n=== Lowering dialect-mir → LLVM dialect ===");
    }
    lower_to_llvm(
        ctx,
        module,
        !backend.no_fma,
        backend_selection.intrinsic_backend,
        backend.module_disambiguator,
    )?;

    let lowered_module_uses_libdevice = module_uses_libdevice(ctx, module);
    if lowered_module_uses_libdevice != backend_selection.needs_libdevice {
        return Err(PipelineError::Lowering(format!(
            "libdevice backend selection changed during MIR lowering: typed MIR detection was {}, but lowered LLVM detection was {}; refusing to continue with a backend chosen from inconsistent requirements",
            backend_selection.needs_libdevice, lowered_module_uses_libdevice
        )));
    }
    let needs_libdevice = backend_selection.needs_libdevice;
    let emit_nvvm_ir = backend_selection.emit_nvvm_ir;
    let requested_nvvm_ir = matches!(
        request.output_policy,
        OutputPolicy::ExternalLinkAllowed {
            request_nvvm_ir: true
        }
    );
    if needs_libdevice && request.trace.verbose {
        if !emit_nvvm_ir && can_ir_link_libdevice {
            request.trace.emit(format!(
                "\n=== Detected CUDA libdevice (`__nv_*`) calls; \
                 linking libdevice.10.bc at IR level ({}) ===",
                libdevice_path.as_ref().unwrap().display()
            ));
        } else if !requested_nvvm_ir && emit_nvvm_ir {
            request.trace.emit(
                "\n=== Detected CUDA libdevice (`__nv_*`) calls; \
                 auto-emitting NVVM IR (skip llc) ===",
            );
        }
    }

    // NVVM pointer syntax depends on the target. Detect the feature floor from
    // the same pre-legalization text the ordinary PTX path would inspect.
    //
    // Two full-text renders happen here (`export_llvm_ir` below does its own
    // render for the real artifact): deliberate, not an oversight. This
    // preview runs pre-legalization; the real export runs post-legalization
    // (which mutates the module for the chosen NVVM dialect) through a
    // different export config. Caching one render for the other would be
    // unsound.
    let automatic_features = if emit_nvvm_ir {
        let preview = render_llvm_ir(
            ctx,
            module,
            request.device_externs,
            false,
            None,
            DebugExport {
                kind: request.debug_kind,
                function_local_static_placement: FunctionLocalStaticPlacement::CompileUnitGlobals,
            },
        )?;
        Some(detect_features_in_llvm_text(&preview))
    } else {
        None
    };

    let (nvvm_target, nvvm_dialect) = if emit_nvvm_ir {
        let target = resolve_nvvm_target_with_generated(
            backend.target_arch.as_deref(),
            validated.device_arch(),
            automatic_features,
            &generated_requirements,
        )?;
        let dialect = if target.uses_legacy_llvm() {
            NvvmIrDialect::LegacyLlvm7
        } else {
            NvvmIrDialect::Modern
        };
        validate_nvvm_debug_support(&target, dialect, request.debug_kind)?;
        (Some(target), Some(dialect))
    } else {
        (None, None)
    };

    if let Some(dialect) = nvvm_dialect {
        if request.trace.verbose {
            if dialect == NvvmIrDialect::LegacyLlvm7 {
                request
                    .trace
                    .emit("\n=== Legalizing LLVM dialect for legacy NVVM ===");
            } else {
                request
                    .trace
                    .emit("\n=== Legalizing NVVM bit-intrinsic widths ===");
            }
        }
        let capability = nvvm_target
            .as_ref()
            .expect("an NVVM dialect is only selected alongside a resolved NVVM target")
            .capability();
        nvvm_transforms::legalize_for_nvvm(ctx, module, dialect, capability)
            .map_err(|error| PipelineError::Lowering(error.disp(ctx).to_string()))?;
    }

    if request.trace.dump_llvm {
        request.trace.emit("\n=== LLVM dialect (pre-verify) ===");
        request
            .trace
            .emit(format!("{}", module.deref(ctx).disp(ctx)));
    }
    if request.trace.verbose {
        request.trace.emit("=== Verifying LLVM dialect module ===");
    }
    verify_operation(ctx, module, "llvm module").map_err(as_lowered_verification)?;
    if request.trace.verbose {
        request.trace.emit("LLVM dialect verification successful ✓");
    }

    if let OutputPolicy::SelfContainedPtx { allow_libdevice } = request.output_policy {
        if let Some(error) = libdevice_availability_error(
            allow_libdevice,
            needs_libdevice,
            can_ir_link_libdevice,
            libdevice_path.is_some(),
        ) {
            return Err(error);
        }
        let symbols = unlinkable_symbols(unresolved_external_symbols(ctx, module), allow_libdevice);
        if !symbols.is_empty() {
            return Err(PipelineError::UnsupportedLinking { symbols });
        }
    }

    if request.trace.verbose {
        let mode = if emit_nvvm_ir { "NVVM IR" } else { "PTX" };
        request
            .trace
            .emit(format!("\n=== Exporting to LLVM IR ({mode} mode) ==="));
    }
    // A function-local static has one verifier-accepted home per LLVM major
    // (see `FunctionLocalStaticPlacement`), so the PTX path resolves the
    // `llc` that will assemble this module before the debug graph is
    // exported. libNVVM's LLVM takes the compile-unit form.
    let mut discovered_toolchain: Option<LlvmToolchain> = None;
    let ptx_toolchain: Option<&LlvmToolchain> = if emit_nvvm_ir {
        None
    } else {
        Some(match request.toolchain {
            ToolchainPolicy::Explicit(toolchain) => toolchain,
            ToolchainPolicy::Discover => {
                &*discovered_toolchain.insert(discover_llvm_toolchain(backend)?)
            }
        })
    };
    let function_local_static_placement = ptx_toolchain.map_or(
        FunctionLocalStaticPlacement::CompileUnitGlobals,
        |toolchain| function_local_static_placement_for_llc(toolchain.llc_major),
    );

    remove_stale_files(request.files.stale_before_export)?;
    let exported = export_llvm_ir(
        ctx,
        module,
        request.device_externs,
        request.files.llvm_ir,
        emit_nvvm_ir,
        nvvm_dialect,
        DebugExport {
            kind: request.debug_kind,
            function_local_static_placement,
        },
    )?;
    if request.trace.verbose {
        request.trace.emit(format!(
            "LLVM IR written to {}",
            request.files.llvm_ir.display()
        ));
    }

    if emit_nvvm_ir {
        if request.trace.verbose {
            let reason = if needs_libdevice {
                "libdevice present"
            } else {
                "NVVM IR requested"
            };
            request.trace.emit(format!(
                "\n=== Skipping llc ({reason}); consumer owns libNVVM/nvJitLink build ==="
            ));
        }
        return Ok(ModulePipelineOutput {
            artifact_kind: ModuleArtifactKind::NvvmIr,
            target: nvvm_target
                .expect("NVVM target was resolved before export")
                .sm(),
            diagnostics: Vec::new(),
        });
    }

    if request.trace.verbose {
        request.trace.emit("\n=== Generating PTX ===");
    }
    let ptx_libdevice = if needs_libdevice && can_ir_link_libdevice {
        libdevice_path.as_deref()
    } else {
        None
    };
    let ptx_module = PtxModule {
        llvm_ir: request.files.llvm_ir,
        output: request.files.ptx,
        public_symbols: &exported.public_symbols,
    };
    let toolchain = ptx_toolchain.expect("the PTX toolchain is resolved before export");
    let generated = match request.toolchain {
        ToolchainPolicy::Discover => generate_ptx_discovered(
            ptx_module,
            request.debug_kind,
            &validated,
            toolchain,
            request.trace.sink,
            &generated_requirements,
            ptx_libdevice,
        )?,
        ToolchainPolicy::Explicit(_) => generate_ptx_with_toolchain(
            ptx_module,
            request.debug_kind,
            &validated,
            toolchain,
            &generated_requirements,
            ptx_libdevice,
        )?,
    };
    // Self-contained PTX promises a single artifact the CUDA driver can load
    // with no further step, so an unresolved `__nv_*` symbol here is a
    // compile-time failure: `llvm-link --only-needed` resolves whatever it
    // finds in libdevice.10.bc and stays silent about the rest, and `opt` and
    // `llc` both exit 0 on what remains, so the alternative is PTX that fails
    // only at `cuModuleLoad` on the device, with no diagnostic. Under
    // `ExternalLinkAllowed` (the rustc frontend) an unresolved `__nv_*` stays
    // legitimate: that consumer owns a later libNVVM/nvJitLink step and
    // resolves it there, so this check must not run for that policy.
    if matches!(request.output_policy, OutputPolicy::SelfContainedPtx { .. }) {
        let ptx_text = std::fs::read_to_string(request.files.ptx).map_err(|error| {
            PipelineError::PtxGeneration(format!(
                "failed to read generated PTX to check for unresolved libdevice symbols ({}): {error}",
                request.files.ptx.display()
            ))
        })?;
        let unresolved = unresolved_libdevice_ptx_declarations(&ptx_text).map_err(|error| {
            PipelineError::PtxGeneration(format!(
                "failed to parse generated PTX to check for unresolved libdevice symbols ({}): {error}",
                request.files.ptx.display()
            ))
        })?;
        if !unresolved.is_empty() {
            return Err(PipelineError::UnsupportedLinking {
                symbols: unresolved,
            });
        }
    }

    if request.trace.verbose {
        request.trace.emit(format!(
            "✓ PTX written to {} (target: {})",
            request.files.ptx.display(),
            generated.target
        ));
    }
    Ok(ModulePipelineOutput {
        artifact_kind: ModuleArtifactKind::Ptx,
        target: generated.target,
        diagnostics: generated.diagnostics,
    })
}

/// Backend decision made from the typed module before MIR lowering starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreLoweringBackendSelection {
    /// Whether the lowered module will contain CUDA libdevice (`__nv_*`)
    /// calls under the selected intrinsic backend. Rounding and sign
    /// (`fabs`/`copysign`) placeholders count only when the LibNvvm backend
    /// is selected; on the NVPTX path they lower to native LLVM intrinsics
    /// instead.
    needs_libdevice: bool,
    /// Whether the pipeline must stop at NVVM IR instead of invoking `llc`.
    emit_nvvm_ir: bool,
    /// Intrinsic ABI to use during MIR-to-LLVM conversion.
    intrinsic_backend: mir_lower::IntrinsicBackend,
}

fn select_pre_lowering_backend(
    ctx: &Context,
    module: Ptr<Operation>,
    output_policy: OutputPolicy,
    can_ir_link_libdevice: bool,
) -> PreLoweringBackendSelection {
    // The rustc frontend supplies typed MIR, while the standalone frontend may
    // also supply an already-lowered LLVM declaration in the same module.
    // Inspect both representations before choosing an intrinsic ABI.
    //
    // Two flavors of libdevice use exist:
    //
    // * strict: calls that lower to `__nv_*` under every intrinsic backend
    //   (direct `__nv_*` externs and the transcendental-family placeholders);
    // * backend-dependent: the rounding and sign (`fabs`/`copysign`)
    //   placeholders, which lower to native LLVM intrinsics on the NVPTX
    //   path and fall back to libdevice only when the module is emitted as
    //   NVVM IR (the legacy LLVM 7 dialect has no `llvm.roundeven.*`).
    //
    // The float `max`/`min` placeholders are in neither flavor: they expand
    // to an ordered compare/select under every backend and never produce a
    // `__nv_*` call.
    //
    // Only strict use participates in the emit-NVVM-IR decision; otherwise a
    // kernel using only rounding or sign ops would drag in the libNVVM
    // dependency it no longer needs.
    let uses_strict_libdevice =
        typed_mir_uses_libdevice(ctx, module) || module_uses_libdevice(ctx, module);
    let uses_backend_dependent_libdevice = typed_mir_calls_match(
        ctx,
        module,
        &dialect_mir::rust_intrinsics::is_backend_dependent_libdevice_placeholder,
    );
    let emit_nvvm_ir =
        should_emit_nvvm_ir(output_policy, uses_strict_libdevice, can_ir_link_libdevice);
    let intrinsic_backend = if emit_nvvm_ir {
        mir_lower::IntrinsicBackend::LibNvvm
    } else {
        mir_lower::IntrinsicBackend::LlvmNvptx
    };
    // Predict the `__nv_*` symbols the lowered module will actually contain,
    // for the post-lowering consistency check: rounding and sign placeholders
    // lower to `__nv_*` precisely when the LibNvvm backend was chosen (and
    // LibNvvm is chosen exactly when NVVM IR is emitted).
    let needs_libdevice =
        uses_strict_libdevice || (emit_nvvm_ir && uses_backend_dependent_libdevice);

    PreLoweringBackendSelection {
        needs_libdevice,
        emit_nvvm_ir,
        intrinsic_backend,
    }
}

/// Whether any typed MIR call in `module` requires libdevice under every
/// intrinsic backend (a direct `__nv_*` extern or a strict placeholder).
fn typed_mir_uses_libdevice(ctx: &Context, module: Ptr<Operation>) -> bool {
    typed_mir_calls_match(ctx, module, &|callee: &str| {
        callee.starts_with("__nv_")
            || dialect_mir::rust_intrinsics::is_libdevice_backed_placeholder(callee)
    })
}

/// Walk `op` and its regions and return whether any `mir.call` callee
/// satisfies `matches`.
fn typed_mir_calls_match(
    ctx: &Context,
    op: Ptr<Operation>,
    matches: &dyn Fn(&str) -> bool,
) -> bool {
    if let Some(call) = Operation::get_op::<dialect_mir::ops::MirCallOp>(op, ctx)
        && let Some(callee_attr) = call.get_attr_callee(ctx)
    {
        let callee: String = (*callee_attr).clone().into();
        if matches(&callee) {
            return true;
        }
    }

    let op_ref = op.deref(ctx);
    for region in op_ref.regions() {
        let region_ref = region.deref(ctx);
        for block in region_ref.iter(ctx) {
            let block_ref = block.deref(ctx);
            for child_op in block_ref.iter(ctx) {
                if typed_mir_calls_match(ctx, child_op, matches) {
                    return true;
                }
            }
        }
    }
    false
}

/// Determines whether the pipeline should emit NVVM IR instead of PTX.
///
/// When `can_ir_link_libdevice` is true, `__nv_*` calls will be resolved by
/// linking `libdevice.10.bc` at the LLVM IR level (via `llvm-link`), so
/// `uses_strict_libdevice` alone no longer forces the NVVM IR path. This
/// avoids the legacy LLVM 7 dialect that cannot represent f16 on
/// pre-Blackwell targets.
///
/// Only strict libdevice use (transcendentals and direct `__nv_*` calls) is
/// consulted. Rounding and sign intrinsics never force the NVVM fallback: on
/// the PTX path they lower to native LLVM intrinsics with no libdevice
/// involvement, and they use libdevice only when NVVM IR is emitted for
/// another reason. Float `max`/`min` never involve libdevice at all.
fn should_emit_nvvm_ir(
    policy: OutputPolicy,
    uses_strict_libdevice: bool,
    can_ir_link_libdevice: bool,
) -> bool {
    match policy {
        OutputPolicy::SelfContainedPtx { .. } => false,
        OutputPolicy::ExternalLinkAllowed { request_nvvm_ir } => {
            if request_nvvm_ir {
                return true;
            }
            // Fall back to NVVM IR only when the kernel needs libdevice AND
            // we cannot resolve the calls via IR-level linking.
            uses_strict_libdevice && !can_ir_link_libdevice
        }
    }
}

/// Narrow collected unresolved symbols to those the artifact cannot resolve.
///
/// Under `allow_libdevice`, `__nv_*` entry points are dropped: the IR-level
/// `llvm-link` step further down resolves them against `libdevice.10.bc`.
/// Every other unresolved symbol still fails the compilation, so
/// `UnsupportedLinking` keeps its meaning for device externs.
fn unlinkable_symbols(mut symbols: Vec<String>, allow_libdevice: bool) -> Vec<String> {
    if allow_libdevice {
        symbols.retain(|symbol| !crate::export::is_libdevice_symbol(symbol));
    }
    symbols
}

/// Whether a compilation that opted into libdevice can be completed.
///
/// Returns the failure to report, or `None` when compilation may continue.
/// Factored out of `compile_translated_module` so the decision is testable
/// with no module, no toolchain, and no CUDA installation.
///
/// `can_ir_link_libdevice` is false when either piece is missing, so
/// `libdevice_found` is what separates a missing CUDA toolkit from an
/// incomplete LLVM toolchain. Neither is a defect in the caller's module,
/// and the two have different fixes, so the message names which one it is.
fn libdevice_availability_error(
    allow_libdevice: bool,
    needs_libdevice: bool,
    can_ir_link_libdevice: bool,
    libdevice_found: bool,
) -> Option<PipelineError> {
    if !allow_libdevice || !needs_libdevice || can_ir_link_libdevice {
        return None;
    }
    let message = if libdevice_found {
        "no `llvm-link` sharing the selected `llc`'s LLVM major was found"
    } else {
        "`libdevice.10.bc` was not found in any CUDA installation"
    };
    Some(PipelineError::LibdeviceUnavailable {
        message: message.to_string(),
    })
}

fn as_lowered_verification(error: PipelineError) -> PipelineError {
    match error {
        PipelineError::Verification {
            message, operation, ..
        } => PipelineError::LoweredVerification { message, operation },
        other => other,
    }
}

fn remove_stale_files(paths: &[PathBuf]) -> Result<(), PipelineError> {
    for path in paths {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(PipelineError::Export(format!(
                    "failed to invalidate stale CUDA artifact {}: {error}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_device_hints_fail_at_the_shared_input_boundary() {
        for raw in [
            "compute_120",
            "compute_80",
            "sm_",
            "foo",
            "sm_05",
            "sm_120f",
            "not-an-arch",
        ] {
            for nvvm in [false, true] {
                for explicit in [None, Some("sm_120"), Some("foo")] {
                    let mut ctx = Context::new();
                    let module = typed_mir_test_module(&mut ctx, &[]);
                    let backend = BackendOptions {
                        device_arch_hint: Some(crate::options::DeviceArchHint::parse(
                            raw.to_string(),
                        )),
                        target_arch: explicit.map(str::to_string),
                        ..BackendOptions::default()
                    };
                    let request = ModulePipelineRequest::for_rust_pipeline(
                        &[],
                        nvvm,
                        &backend,
                        DebugKind::Off,
                        OutputFiles {
                            llvm_ir: Path::new("unused.ll"),
                            ptx: Path::new("unused.ptx"),
                            stale_before_export: &[],
                        },
                        PipelineTrace::default(),
                    );
                    let Err(error) = compile_translated_module(&mut ctx, module, &request) else {
                        panic!("invalid hint must fail at input");
                    };
                    assert!(matches!(error, PipelineError::TargetSelection { .. }));
                    assert_eq!(
                        error.to_string(),
                        format!(
                            "invalid CUDA_OXIDE_DEVICE_ARCH `{raw}`: expected sm_<capability> with an optional `a` suffix"
                        )
                    );
                    assert_eq!(
                        crate::api::CompileError::from(error).stage(),
                        crate::api::CompilationStage::Input
                    );
                }
            }
        }
    }

    fn typed_mir_test_module(ctx: &mut Context, callees: &[&str]) -> Ptr<Operation> {
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::attributes::{StringAttr, TypeAttr};
        use pliron::builtin::op_interfaces::SymbolOpInterface;
        use pliron::builtin::types::FunctionType;
        use pliron::op::Op;

        dialect_mir::register(ctx);
        dialect_nvvm::register(ctx);
        let module = pliron::builtin::ops::ModuleOp::new(ctx, "test".try_into().unwrap());
        let module_op = module.get_operation();

        let function_type = FunctionType::get(ctx, vec![], vec![]);
        let function_op = Operation::new(
            ctx,
            dialect_mir::ops::MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function =
            dialect_mir::ops::MirFuncOp::new(ctx, function_op, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, "kernel".try_into().unwrap());

        let body = function_op.deref(ctx).get_region(0);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(body, ctx);
        for callee in callees {
            let call_op = Operation::new(
                ctx,
                dialect_mir::ops::MirCallOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            let call = dialect_mir::ops::MirCallOp::new(call_op);
            call.set_attr_callee(ctx, StringAttr::new((*callee).to_string()));
            call_op.insert_at_back(block, ctx);
        }

        crate::lower::append_to_module(ctx, module_op, function_op);
        module_op
    }

    #[test]
    fn standalone_never_silently_switches_to_a_linkable_artifact() {
        // Opting into libdevice must not open the NVVM IR route: the standalone
        // surface promises one artifact kind out, whatever the linking policy.
        for allow_libdevice in [false, true] {
            let policy = OutputPolicy::SelfContainedPtx { allow_libdevice };
            for uses_strict_libdevice in [false, true] {
                for can_ir_link_libdevice in [false, true] {
                    assert!(
                        !should_emit_nvvm_ir(policy, uses_strict_libdevice, can_ir_link_libdevice),
                        "self-contained PTX emitted NVVM IR for \
                         allow_libdevice={allow_libdevice}, \
                         uses_strict_libdevice={uses_strict_libdevice}, \
                         can_ir_link_libdevice={can_ir_link_libdevice}"
                    );
                }
            }
        }
    }

    #[test]
    fn link_capable_pipeline_honors_requests_and_libdevice() {
        let automatic = OutputPolicy::ExternalLinkAllowed {
            request_nvvm_ir: false,
        };
        let requested = OutputPolicy::ExternalLinkAllowed {
            request_nvvm_ir: true,
        };
        // No libdevice, no IR linking available → PTX path.
        assert!(!should_emit_nvvm_ir(automatic, false, false));
        // Libdevice needed, no IR linking → falls back to NVVM IR.
        assert!(should_emit_nvvm_ir(automatic, true, false));
        // Explicit NVVM IR request always wins.
        assert!(should_emit_nvvm_ir(requested, false, false));
    }

    #[test]
    fn ir_linking_avoids_nvvm_ir_for_libdevice() {
        let automatic = OutputPolicy::ExternalLinkAllowed {
            request_nvvm_ir: false,
        };
        // Libdevice needed, IR linking available → PTX path (not NVVM IR).
        assert!(!should_emit_nvvm_ir(automatic, true, true));
        // No libdevice needed, IR linking available → PTX path (unchanged).
        assert!(!should_emit_nvvm_ir(automatic, false, true));
        // Explicit NVVM IR request still forces NVVM IR even with IR linking.
        let requested = OutputPolicy::ExternalLinkAllowed {
            request_nvvm_ir: true,
        };
        assert!(should_emit_nvvm_ir(requested, false, true));
    }

    #[test]
    fn typed_libdevice_without_ir_linking_falls_back_to_nvvm() {
        let mut ctx = Context::new();
        let module =
            typed_mir_test_module(&mut ctx, &[dialect_mir::rust_intrinsics::CALLEE_SIN_F32]);
        // No IR-level linking: libdevice forces NVVM IR (legacy behavior).
        let selection = select_pre_lowering_backend(
            &ctx,
            module,
            OutputPolicy::ExternalLinkAllowed {
                request_nvvm_ir: false,
            },
            false,
        );
        assert!(selection.needs_libdevice);
        assert!(selection.emit_nvvm_ir);
        assert_eq!(
            selection.intrinsic_backend,
            mir_lower::IntrinsicBackend::LibNvvm
        );
    }

    #[test]
    fn typed_libdevice_with_ir_linking_stays_on_ptx_path() {
        let mut ctx = Context::new();
        let module =
            typed_mir_test_module(&mut ctx, &[dialect_mir::rust_intrinsics::CALLEE_SIN_F32]);
        // IR-level linking available: libdevice does NOT force NVVM IR.
        let selection = select_pre_lowering_backend(
            &ctx,
            module,
            OutputPolicy::ExternalLinkAllowed {
                request_nvvm_ir: false,
            },
            true,
        );
        assert!(selection.needs_libdevice);
        assert!(!selection.emit_nvvm_ir);
        assert_eq!(
            selection.intrinsic_backend,
            mir_lower::IntrinsicBackend::LlvmNvptx
        );
    }

    /// Rounding placeholders are backend-dependent: with PTX output they
    /// lower to native LLVM intrinsics, so a rounding-only module must not
    /// be flagged as needing libdevice and must not fall back to NVVM IR,
    /// regardless of whether IR-level libdevice linking is available.
    #[test]
    fn rounding_only_module_stays_on_self_sufficient_ptx_path() {
        let mut ctx = Context::new();
        let module = typed_mir_test_module(
            &mut ctx,
            &[
                dialect_mir::rust_intrinsics::CALLEE_FLOOR_F32,
                dialect_mir::rust_intrinsics::CALLEE_ROUNDEVEN_F64,
            ],
        );
        for can_ir_link_libdevice in [false, true] {
            let selection = select_pre_lowering_backend(
                &ctx,
                module,
                OutputPolicy::ExternalLinkAllowed {
                    request_nvvm_ir: false,
                },
                can_ir_link_libdevice,
            );
            assert!(!selection.needs_libdevice);
            assert!(!selection.emit_nvvm_ir);
            assert_eq!(
                selection.intrinsic_backend,
                mir_lower::IntrinsicBackend::LlvmNvptx
            );
        }
    }

    /// With an explicit NVVM IR request, the LibNvvm backend lowers rounding
    /// placeholders to `__nv_*` libdevice calls (the legacy LLVM 7 dialect
    /// has no `llvm.roundeven.*`), so the prediction must flag libdevice or
    /// the post-lowering consistency check would abort the build.
    #[test]
    fn rounding_only_module_uses_libdevice_when_nvvm_ir_is_requested() {
        let mut ctx = Context::new();
        let module =
            typed_mir_test_module(&mut ctx, &[dialect_mir::rust_intrinsics::CALLEE_FLOOR_F32]);
        let selection = select_pre_lowering_backend(
            &ctx,
            module,
            OutputPolicy::ExternalLinkAllowed {
                request_nvvm_ir: true,
            },
            false,
        );
        assert!(selection.needs_libdevice);
        assert!(selection.emit_nvvm_ir);
        assert_eq!(
            selection.intrinsic_backend,
            mir_lower::IntrinsicBackend::LibNvvm
        );
    }

    /// The sign placeholders (`fabs`/`copysign`) are backend-dependent like
    /// rounding, and the float `max`/`min` placeholders never use libdevice
    /// at all: a module using only these ops must stay on the
    /// self-sufficient PTX path with no libdevice prediction.
    #[test]
    fn sign_and_minmax_only_module_stays_on_self_sufficient_ptx_path() {
        let mut ctx = Context::new();
        let module = typed_mir_test_module(
            &mut ctx,
            &[
                dialect_mir::rust_intrinsics::CALLEE_FABS,
                dialect_mir::rust_intrinsics::CALLEE_COPYSIGN_F32,
                dialect_mir::rust_intrinsics::CALLEE_MAXNUM_NSZ_F32,
                dialect_mir::rust_intrinsics::CALLEE_MINNUM_NSZ_F64,
            ],
        );
        for can_ir_link_libdevice in [false, true] {
            let selection = select_pre_lowering_backend(
                &ctx,
                module,
                OutputPolicy::ExternalLinkAllowed {
                    request_nvvm_ir: false,
                },
                can_ir_link_libdevice,
            );
            assert!(!selection.needs_libdevice);
            assert!(!selection.emit_nvvm_ir);
            assert_eq!(
                selection.intrinsic_backend,
                mir_lower::IntrinsicBackend::LlvmNvptx
            );
        }
    }

    /// Under an explicit NVVM IR request the sign placeholders fall back to
    /// `__nv_fabsf`/`__nv_copysignf` (so libdevice must be predicted), while
    /// `max`/`min` expand to compare/select under every backend and
    /// contribute nothing to the prediction.
    #[test]
    fn sign_module_needs_libdevice_under_nvvm_ir_but_minmax_never_does() {
        let mut ctx = Context::new();
        let nvvm_requested = OutputPolicy::ExternalLinkAllowed {
            request_nvvm_ir: true,
        };

        let sign_module =
            typed_mir_test_module(&mut ctx, &[dialect_mir::rust_intrinsics::CALLEE_FABS]);
        let sign = select_pre_lowering_backend(&ctx, sign_module, nvvm_requested, false);
        assert!(sign.needs_libdevice);
        assert!(sign.emit_nvvm_ir);
        assert_eq!(sign.intrinsic_backend, mir_lower::IntrinsicBackend::LibNvvm);

        let minmax_module = typed_mir_test_module(
            &mut ctx,
            &[dialect_mir::rust_intrinsics::CALLEE_MAXNUM_NSZ_F32],
        );
        let minmax = select_pre_lowering_backend(&ctx, minmax_module, nvvm_requested, false);
        assert!(!minmax.needs_libdevice);
        assert!(minmax.emit_nvvm_ir);
        assert_eq!(
            minmax.intrinsic_backend,
            mir_lower::IntrinsicBackend::LibNvvm
        );
    }

    /// Mixing rounding with a strict libdevice op follows the strict op's
    /// backend choice; rounding contributes to the libdevice prediction only
    /// when that choice lands on NVVM IR.
    #[test]
    fn mixed_rounding_and_transcendental_follows_the_strict_op() {
        let mut ctx = Context::new();
        let module = typed_mir_test_module(
            &mut ctx,
            &[
                dialect_mir::rust_intrinsics::CALLEE_FLOOR_F32,
                dialect_mir::rust_intrinsics::CALLEE_SIN_F32,
            ],
        );
        let automatic = OutputPolicy::ExternalLinkAllowed {
            request_nvvm_ir: false,
        };

        // No IR-level linking: sin forces NVVM IR; rounding joins it on
        // libdevice under the LibNvvm backend.
        let no_link = select_pre_lowering_backend(&ctx, module, automatic, false);
        assert!(no_link.needs_libdevice);
        assert!(no_link.emit_nvvm_ir);
        assert_eq!(
            no_link.intrinsic_backend,
            mir_lower::IntrinsicBackend::LibNvvm
        );

        // IR-level linking available: PTX path; sin resolves against
        // libdevice.10.bc while rounding lowers to LLVM intrinsics.
        let ir_link = select_pre_lowering_backend(&ctx, module, automatic, true);
        assert!(ir_link.needs_libdevice);
        assert!(!ir_link.emit_nvvm_ir);
        assert_eq!(
            ir_link.intrinsic_backend,
            mir_lower::IntrinsicBackend::LlvmNvptx
        );
    }

    #[test]
    fn explicit_nvvm_and_standalone_ptx_choose_distinct_intrinsic_abis() {
        let mut ctx = Context::new();
        let ordinary =
            typed_mir_test_module(&mut ctx, &[dialect_mir::rust_intrinsics::CALLEE_FADD_FAST]);
        let nvvm = select_pre_lowering_backend(
            &ctx,
            ordinary,
            OutputPolicy::ExternalLinkAllowed {
                request_nvvm_ir: true,
            },
            false,
        );
        assert!(!nvvm.needs_libdevice);
        assert!(nvvm.emit_nvvm_ir);
        assert_eq!(nvvm.intrinsic_backend, mir_lower::IntrinsicBackend::LibNvvm);

        let standalone = select_pre_lowering_backend(
            &ctx,
            ordinary,
            OutputPolicy::SelfContainedPtx {
                allow_libdevice: false,
            },
            false,
        );
        assert!(!standalone.emit_nvvm_ir);
        assert_eq!(
            standalone.intrinsic_backend,
            mir_lower::IntrinsicBackend::LlvmNvptx
        );
    }

    #[test]
    fn final_verification_is_not_reported_as_mir_preparation() {
        let classified = as_lowered_verification(PipelineError::Verification {
            name: "llvm module".to_string(),
            message: "invalid lowered op".to_string(),
            operation: Some("llvm.bad".to_string()),
        });
        assert!(matches!(
            classified,
            PipelineError::LoweredVerification { message, operation }
                if message == "invalid lowered op" && operation.as_deref() == Some("llvm.bad")
        ));
    }

    #[test]
    fn self_contained_linking_rejects_every_unresolved_symbol() {
        let symbols = vec![
            "__nv_sqrtf".to_string(),
            "vprintf".to_string(),
            "my_device_extern".to_string(),
        ];
        assert_eq!(
            unlinkable_symbols(symbols.clone(), false),
            symbols,
            "without the libdevice opt-in nothing is filtered"
        );
    }

    #[test]
    fn libdevice_linking_drops_only_libdevice_symbols() {
        let symbols = vec![
            "__nv_erff".to_string(),
            "__nv_sqrtf".to_string(),
            "my_device_extern".to_string(),
            "vprintf".to_string(),
        ];
        assert_eq!(
            unlinkable_symbols(symbols, true),
            vec!["my_device_extern".to_string(), "vprintf".to_string()],
            "device externs still fail the compilation"
        );
    }

    #[test]
    fn libdevice_linking_accepts_a_purely_libdevice_module() {
        let symbols = vec!["__nv_sqrtf".to_string(), "__nv_expf".to_string()];
        assert!(unlinkable_symbols(symbols, true).is_empty());
    }

    #[test]
    fn libdevice_symbol_predicate_matches_the_nv_prefix_only() {
        assert!(crate::export::is_libdevice_symbol("__nv_sqrtf"));
        assert!(crate::export::is_libdevice_symbol("__nv_"));
        assert!(!crate::export::is_libdevice_symbol("__nvvm_thing"));
        assert!(!crate::export::is_libdevice_symbol("nv_sqrtf"));
        assert!(!crate::export::is_libdevice_symbol("my___nv_sqrtf"));
    }

    #[test]
    fn libdevice_availability_names_the_missing_piece() {
        // Missing bitcode: the CUDA toolkit was not found.
        let error = libdevice_availability_error(true, true, false, false)
            .expect("an unlinkable libdevice module must fail");
        let message = error.to_string();
        assert!(
            message.contains("libdevice.10.bc"),
            "message names the missing bitcode: {message}"
        );

        // Bitcode present, linker absent: the LLVM toolchain is incomplete.
        let error = libdevice_availability_error(true, true, false, true)
            .expect("an unlinkable libdevice module must fail");
        let message = error.to_string();
        assert!(
            message.contains("llvm-link"),
            "message names the missing linker: {message}"
        );
    }

    #[test]
    fn libdevice_availability_permits_every_compilable_case() {
        // Not opted in: the ordinary unresolved-symbol rejection handles it.
        assert!(libdevice_availability_error(false, true, false, false).is_none());
        // Opted in but the module needs no libdevice: nothing to link.
        assert!(libdevice_availability_error(true, false, false, false).is_none());
        // Opted in, needed, and linkable: the ordinary path proceeds.
        assert!(libdevice_availability_error(true, true, true, true).is_none());
    }
}
