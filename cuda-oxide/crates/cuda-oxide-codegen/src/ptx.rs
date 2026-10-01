/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::generated::GeneratedModuleRequirements;
use crate::llvm_tools::LlvmToolchain;
use crate::options::BackendOptions;
use crate::pipeline::ValidatedBackendOptions;
use crate::target::{
    ModuleRequirements, PtxIsaRequirement, detect_module_requirements_in_llvm_file,
    merge_generated_module_requirements, merge_generated_module_requirements_for_target,
    required_ptx_feature, resolve_ptx_target_with_generated, validate_ptx_isa_for_llvm_major,
    validate_target_features, validate_target_for_llvm_major,
};
use llvm_export::export::DebugKind;
use ptx_parse::{Document, EditScript, split_top_level};
use std::path::{Path, PathBuf};

/// Links `libdevice.10.bc` into the emitted IR using `llvm-link`.
///
/// Resolves `__nv_*` calls (CUDA math library) at the IR level so they are
/// inlined and optimized by `opt -O2` before `llc` lowers to PTX. This
/// avoids the legacy NVVM IR path (which uses the LLVM 7 dialect and cannot
/// represent f16 types on pre-Blackwell targets).
///
/// `--internalize --only-needed` mirrors clang's
/// `LinkOnlyNeeded | InternalizeLinkedSymbols`: libdevice bodies have plain
/// external linkage, so without both flags all ~350 definitions are pulled
/// in, survive GlobalDCE, and llc exports every one as a `.visible .func
/// __nv_*` PTX body (a one-call kernel balloons from ~130 to ~22,000 lines
/// and later cuLink/nvJitLink steps hit duplicate-symbol collisions). With
/// the flags, only the referenced bodies are imported, as `internal`, and
/// `opt -O2` inlines or discards them.
///
/// Failure is a hard error: the pipeline chooses the PTX path for a
/// libdevice kernel only after confirming `llvm-link` is resolvable, so a
/// link failure here must not degrade into PTX with unresolved
/// `.extern .func __nv_*` that only fails later at cuModuleLoad.
fn link_libdevice(
    ll_path: &Path,
    libdevice_path: &Path,
    toolchain: &LlvmToolchain,
    diagnostic_sink: Option<fn(&str)>,
    diagnostics: &mut Vec<String>,
    verbose: bool,
) -> Result<PathBuf, PipelineError> {
    let Some(llvm_link) = toolchain.llvm_link.as_ref() else {
        return Err(PipelineError::PtxGeneration(
            "libdevice linking is required, but no `llvm-link` matching the selected `llc` \
             is available; install the matching LLVM tools or set CUDA_OXIDE_LLVM_LINK"
                .to_string(),
        ));
    };

    let linked_path = ll_path.with_extension("linked.ll");
    match std::process::Command::new(&llvm_link.path)
        .arg("-S")
        .arg("--internalize")
        .arg("--only-needed")
        .arg(ll_path)
        .arg(libdevice_path)
        .arg("-o")
        .arg(&linked_path)
        .output()
    {
        Ok(output) if output.status.success() => {
            if verbose {
                record_diagnostic(
                    diagnostics,
                    diagnostic_sink,
                    format!(
                        "llvm-link: linked libdevice ({}) → {}",
                        libdevice_path.display(),
                        linked_path.display()
                    ),
                );
            }
            Ok(linked_path)
        }
        Ok(output) => Err(PipelineError::PtxGeneration(format!(
            "llvm-link ({}) failed with status {} while linking libdevice ({}):\n{}",
            llvm_link.path,
            output.status,
            libdevice_path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ))),
        Err(error) => Err(PipelineError::PtxGeneration(format!(
            "failed to run llvm-link ({}) while linking libdevice ({}): {error}",
            llvm_link.path,
            libdevice_path.display()
        ))),
    }
}

/// Runs LLVM's middle-end on the emitted IR before `llc`.
///
/// Modules with explicit `@llvm.used` roots internalize every other definition
/// before the default O2 pipeline so fully inlined helpers are eligible for
/// global dead-code elimination. Modules without an explicit root set retain
/// the historical `opt -O2` path.
///
/// This is what consumes the per-op ABI alignment we emit: the
/// LoadStoreVectorizer fuses aligned aggregate/element accesses, SROA
/// scalarizes stack aggregates, and InferAddressSpaces promotes generic
/// pointers to `.global` (LDG/STG). Gated on alignment — fusion only fires
/// when loads/stores carry matching `align N` hints.
///
/// The `opt` binary comes from the resolved [`LlvmToolchain`], which
/// guarantees it shares the LLVM major of the `llc` that will consume its
/// output (issue #150: an LLVM 22 `opt` emits sizeless
/// `llvm.lifetime.start/end` intrinsics that an LLVM 21 `llc` rejects).
///
/// Returns the optimized path plus caller-owned diagnostics. The standalone v1 API
/// is strict; the legacy rustc path retains its warn-and-continue behavior.
/// LLVM's verifier prints this when a module's debug metadata is malformed.
/// It then strips every debug node instead of failing, so `opt` and `llc`
/// exit 0 and emit an artifact with no `.loc`, no `.file`, and no DWARF.
const LLVM_INVALID_DEBUG_INFO_WARNING: &str = "ignoring invalid debug info";

/// Fail when a tool that exited successfully reported that it discarded the
/// module's debug metadata.
///
/// A debug build whose debug graph is silently dropped is worse than a failed
/// build: cuda-gdb then resolves every source breakpoint to its host shadow
/// and never stops in the kernel. Debug-off builds carry no graph to reject,
/// so their stderr is left alone.
fn reject_dropped_debug_info(
    tool: &str,
    stderr: &[u8],
    debug_kind: DebugKind,
) -> Result<(), String> {
    if !debug_kind.line_tables_enabled() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(stderr);
    if !stderr.contains(LLVM_INVALID_DEBUG_INFO_WARNING) {
        return Ok(());
    }
    Err(format!(
        "{tool} rejected the module's debug metadata and would have emitted an artifact \
         without it, but device debug information was requested:\n{}",
        stderr.trim()
    ))
}

fn optimize_ll(
    ll_path: &Path,
    public_symbols: &[String],
    toolchain: &LlvmToolchain,
    opts: &BackendOptions,
    strict: bool,
    debug_kind: DebugKind,
) -> Result<(Option<PathBuf>, Vec<String>), PipelineError> {
    if opts.no_opt {
        return Ok((None, Vec::new()));
    }
    let Some(opt) = toolchain.opt.as_ref() else {
        if strict {
            return Err(PipelineError::Optimization(
            "optimization was requested, but no `opt` matching the selected `llc` is available; \
             install the matching LLVM tools or explicitly disable optimization"
                .to_string(),
            ));
        }
        return Ok((None, Vec::new()));
    };

    let optimization_args = optimization_args(public_symbols)?;

    let opt_ll = ll_path.with_extension("opt.ll");
    match std::process::Command::new(&opt.path)
        .args(&optimization_args)
        .arg(ll_path)
        .arg("-S")
        .arg("-o")
        .arg(&opt_ll)
        .output()
    {
        Ok(output) if output.status.success() => {
            reject_dropped_debug_info(&format!("opt ({})", opt.path), &output.stderr, debug_kind)
                .map_err(PipelineError::Optimization)?;
            let diagnostics = opts
                .verbose
                .then(|| {
                    format!(
                        "opt {} via {}: {}",
                        optimization_args.join(" "),
                        opt.path,
                        opt_ll.display()
                    )
                })
                .into_iter()
                .collect();
            Ok((Some(opt_ll), diagnostics))
        }
        Ok(output) => {
            let message = format!(
                "opt ({}) failed with status {}:\n{}",
                opt.path,
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
            if strict {
                Err(PipelineError::Optimization(message))
            } else {
                Ok((
                    None,
                    vec![format!(
                        "warning: {message}\nwarning: continuing with unoptimized IR"
                    )],
                ))
            }
        }
        Err(error) => {
            let message = format!("failed to run opt ({}): {error}", opt.path);
            if strict {
                Err(PipelineError::Optimization(message))
            } else {
                Ok((
                    None,
                    vec![format!(
                        "warning: {message}; continuing with unoptimized IR"
                    )],
                ))
            }
        }
    }
}

/// Disable SimplifyCFG's switch-to-lookup-table transform in `opt`.
///
/// The transform replaces a dense switch with a constant data table of its
/// results. Fine on CPUs, fatal here when the results are shared-memory
/// pointers (a kernel `match` picking a per-stage buffer): shared addresses
/// are per-block runtime values, so a compile-time table of them is an
/// impossible object, and every tool after `opt` handles it badly:
///
/// ```text
/// match stage { 0 => &SMEM_A0, .. }        the failure chain:
///        │ opt: builds the table            opt   builds it     (trigger)
///        ▼                                  llc   prints it,
/// .global .u64 table[4] =                         exit 0        (silence)
///     { __shared_mem_0, .. }   ← invalid   ptxas / driver JIT
///     { generic(__shared_mem_0), .. } too        reject: CUDA_ERROR_
///                                                INVALID_PTX    (too late)
/// ```
///
/// ptxas rejects BOTH encodings with "Variable used as initial value not
/// in .global or .const state space"; on a real GPU the module dies at JIT
/// load with error 218.
///
/// Why this appeared with LLVM 23, and why the flag is permanent:
///
/// - `opt` decides "is the table trick worth it?" from its target GPU. We
///   pass no `-mcpu`, so the built-in default decides: LLVM 22 assumed
///   sm_30 (trick off); LLVM 23 assumes sm_75 (trick on) after
///   llvm/llvm-project PR #176021 (commit 9fc5fd0ad689).
/// - The bug was always latent, not new: LLVM 22 with an explicit modern
///   `-mcpu` builds the same bad tables, because upstream's
///   `validLookupTableConstant` never checks address spaces. So passing a
///   real `-mcpu` would make things worse, not better.
/// - This cl::opt is the supported control, accepted identically by every
///   `opt` we support (LLVM 21+), and it stays even after upstream learns
///   to reject shared-space table constants: our LLVM floor spans the
///   broken majors for years. Precedent: NVPTX already disables relative
///   lookup tables wholesale (llvm/llvm-project#159748).
///
/// What we get instead is better anyway: switches keep their branch form
/// and llc lowers dense ones to `brx.idx` code-label tables, which PTX
/// encodes fine, and which swap a dependent `.global` data load on the hot
/// path for a few uniform ALU ops.
const DISABLE_SWITCH_LOOKUP_TABLES: &str = "-switch-to-lookup=false";

/// Disable llc's late branch folding so loops keep the two-jump layout
/// ptxas's SASS unroller recognizes.
///
/// LLVM 23's NVPTX backend gained `reverseBranchCondition`
/// (llvm/llvm-project PR #191889, commit d55166c23bf1, follow-up PR
/// #191890, commit 205f4bf6cc03), which lets BranchFolding and
/// MachineBlockPlacement collapse the classic loop branch idiom into a
/// single negated conditional with fallthrough:
///
/// ```text
/// LLVM 22 layout (ptxas unrolls)       LLVM 23 layout (ptxas gives up)
///
/// guard:  @%p bra body;    ─┐          guard:  @!%p bra exit;
///         bra.uni exit;     │ taken            (falls through to body)
/// body:   ...             ◄─┘          body:   ...
/// latch:  @%p bra exit;                latch:  @%p bra body;
///         bra.uni body;   ← continue           (falls through to exit)
/// exit:                                exit:
/// ```
///
/// Why we turn it off:
///
/// - ptxas's SASS loop unroller keys on the taken-target orientation at
///   both ends of the loop: the guard's conditional taken-target must
///   enter the preheader/body, and the latch must be "conditional exit +
///   `bra.uni` continue". The folded negated forms defeat it.
/// - Upstream enabled the folding with no opt-out (PR #191889), so the
///   only supported control is disabling the two late machine passes.
/// - Both flags are generic disable-only cl::opts: they skip layout
///   transforms and change no semantics, so there is no correctness
///   surface. llc 21.1.8, 22.1.2, 22.1.7, and 23.1.0 all accept them.
/// - Measured on gemm_views' sgemm_naive_raw (RTX 5090, live benches,
///   bit-identical numerics): folded layout drops 38 -> 26 registers and
///   FFMA 18 -> 4, costing 26% throughput (7218 -> 5708 GFLOPS). With
///   these flags: 38 registers, 7226 GFLOPS. Full-suite sweep (890
///   kernels, 197 modules): 45 kernels change registers, every one the
///   SASS unroller re-enabling (register counts and SASS bodies grow,
///   e.g. the table-lookup scan loops now fully unroll) or an exact
///   return to the pre-LLVM-23 baseline; zero unexplained movers, all
///   affected examples numerically verified on hardware. PTX grows
///   +2.6% in lines suite-wide, all redundant jumps ptxas discards.
///
/// Permanent until ptxas learns to unroll both layouts or upstream adds
/// an opt-out for the NVPTX folding; an internal NVBug and an LLVM issue
/// are to be filed to track both ends.
const DISABLE_BRANCH_FOLD: &str = "-disable-branch-fold";

/// Companion to [`DISABLE_BRANCH_FOLD`]: MachineBlockPlacement performs
/// the same rotation and tail layout on its own, so both passes must be
/// off or the folded form comes back.
const DISABLE_BLOCK_PLACEMENT: &str = "-disable-block-placement";

/// The unconditional head of every `llc` invocation: target selection plus
/// the branch-layout controls ptxas depends on (see
/// [`DISABLE_BRANCH_FOLD`]). Kept as a function so the tests can assert
/// the argument list verbatim.
fn base_llc_args(target: &str) -> Vec<String> {
    vec![
        "-march=nvptx64".to_string(),
        format!("-mcpu={target}"),
        DISABLE_BRANCH_FOLD.to_string(),
        DISABLE_BLOCK_PLACEMENT.to_string(),
    ]
}

/// Full-debug modules need PTX ISA 7.5 or newer declared, whatever the
/// target's own floor is.
///
/// llc's DWARF emission writes label-difference expressions into debug
/// sections:
///
/// ```text
/// .section .debug_pubnames
/// {
/// .b32 $L__pubNames_end0-$L__pubNames_start0   ← "labels1 - labels2
///                                                 expression in .section"
///                                                 = PTX ISA 7.5 feature
/// ```
///
/// but still declares the target's default `.version` (7.0 at sm_80), so
/// ptxas rejects the module. Observed with llc-22 (CI's floor pin); llc-23
/// emits its debug sections differently and dodges it. There is no 7.5
/// requirement spelling in the supported set, so raise to the nearest one,
/// 7.8; [`required_ptx_feature`] already refuses to downgrade targets whose
/// floor is at or above it. Caught by the all-examples compile-only ptxas
/// gate; line-tables debug emits no such expressions and stays untouched.
fn ptx_isa_with_debug_floor(
    requirement: PtxIsaRequirement,
    debug_kind: DebugKind,
) -> PtxIsaRequirement {
    if debug_kind.variables_enabled() {
        requirement.max(PtxIsaRequirement::new(78))
    } else {
        requirement
    }
}

/// Build the middle-end arguments for a self-contained PTX module.
///
/// The LLVM exporter returns the module's externally consumed definitions as
/// typed export metadata: entry kernels (or standalone device functions) plus
/// host-visible globals. Passing that root set directly avoids inferring
/// visibility from symbol spelling or rendered LLVM text.
/// Once ordinary inlining has copied a non-root helper into every caller,
/// internalization lets GlobalDCE remove it instead of asking `llc` to emit an
/// unreachable `.visible .func` body.
fn optimization_args(public_symbols: &[String]) -> Result<Vec<String>, PipelineError> {
    if public_symbols.is_empty() {
        return Ok(vec![
            "-O2".to_string(),
            DISABLE_SWITCH_LOOKUP_TABLES.to_string(),
        ]);
    }

    if let Some(symbol) = public_symbols.iter().find(|symbol| symbol.contains(',')) {
        return Err(PipelineError::Optimization(format!(
            "external symbol `{symbol}` cannot be represented in LLVM's comma-separated internalization API list"
        )));
    }

    Ok(vec![
        "-passes=internalize,default<O2>".to_string(),
        format!("-internalize-public-api-list={}", public_symbols.join(",")),
        DISABLE_SWITCH_LOOKUP_TABLES.to_string(),
    ])
}

/// Legacy rustc-pipeline result, including messages the CLI should print.
#[doc(hidden)]
#[allow(missing_docs)]
#[derive(Debug)]
pub struct GeneratedPtx {
    pub target: String,
    pub diagnostics: Vec<String>,
}

struct PtxBackend<'a> {
    options: &'a ValidatedBackendOptions<'a>,
    toolchain: &'a LlvmToolchain,
    generated: &'a GeneratedModuleRequirements,
}

/// One module's artifact paths plus its externally consumed symbol roots.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub struct PtxModule<'a> {
    /// Textual LLVM IR input.
    pub llvm_ir: &'a Path,
    /// PTX output path.
    pub output: &'a Path,
    /// Symbols the internalization pass must keep external.
    pub public_symbols: &'a [String],
}

/// Generates PTX from LLVM IR using `llc`.
///
/// LLVM 21+ is the minimum supported version: earlier `llc` releases reject
/// the modern TMA / tcgen05 / WGMMA intrinsic signatures that cuda-oxide emits
/// (e.g. the 10-operand `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d` with
/// `addrspace(7)` + CTA group parameter requires LLVM 21). If
/// `opts.llc_override` (historically `CUDA_OXIDE_LLC`) is set, it is used
/// exclusively; power users can point it at an older `llc` at their own risk.
///
/// `opt` and `llc` are resolved together via [`LlvmToolchain`] so the
/// middle-end never runs under a different LLVM major than the backend
/// (issue #150).
///
/// Target arch resolves (highest priority first) to: `opts.target_arch`
/// (historically `CUDA_OXIDE_TARGET`), else the detected-GPU hint
/// `opts.device_arch_hint` (historically `CUDA_OXIDE_DEVICE_ARCH`) when that
/// GPU can run the kernel, else the minimum arch the IR's features require.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
#[cfg(test)]
pub(crate) fn generate_ptx(
    module: PtxModule<'_>,
    debug_kind: DebugKind,
    validated: &ValidatedBackendOptions<'_>,
    diagnostic_sink: Option<fn(&str)>,
    generated: &GeneratedModuleRequirements,
    libdevice_path: Option<&Path>,
) -> Result<GeneratedPtx, PipelineError> {
    let toolchain = discover_llvm_toolchain(validated.options())?;
    generate_ptx_discovered(
        module,
        debug_kind,
        validated,
        &toolchain,
        diagnostic_sink,
        generated,
        libdevice_path,
    )
}

/// Resolve the `llc` / `opt` / `llvm-link` set the PTX path will use.
///
/// Split from [`generate_ptx`] so the pipeline can learn the `llc` major
/// before it exports the debug graph, whose function-local static placement
/// depends on that major.
pub(crate) fn discover_llvm_toolchain(
    opts: &BackendOptions,
) -> Result<LlvmToolchain, PipelineError> {
    LlvmToolchain::resolve(opts).ok_or_else(|| {
        PipelineError::PtxGeneration(
            "No working llc found.\n\
             cuda-oxide tries (in order): opts.llc_override (CUDA_OXIDE_LLC), the \
             Rust toolchain's llvm-tools llc, then llc-23 / llc-22 / llc-21 on PATH. \
             LLVM 21+ is required (earlier versions reject the TMA / tcgen05 / \
             WGMMA intrinsic signatures we emit).\n\
             Easiest fix: `rustup component add llvm-tools` (auto-picked up).\n\
             Alternative: `sudo apt install llvm-21` (or `llvm-22`).\n\
             Or set opts.llc_override (CUDA_OXIDE_LLC) to a specific binary."
                .to_string(),
        )
    })
}

/// Generate PTX with a toolchain from [`discover_llvm_toolchain`], reporting
/// its selection diagnostics through `diagnostic_sink`.
pub(crate) fn generate_ptx_discovered(
    module: PtxModule<'_>,
    debug_kind: DebugKind,
    validated: &ValidatedBackendOptions<'_>,
    toolchain: &LlvmToolchain,
    diagnostic_sink: Option<fn(&str)>,
    generated: &GeneratedModuleRequirements,
    libdevice_path: Option<&Path>,
) -> Result<GeneratedPtx, PipelineError> {
    let opts = validated.options();
    let mut diagnostics = toolchain.diagnostics.clone();
    if !opts.no_opt && toolchain.opt.is_none() {
        diagnostics.push(
            "warning: continuing with unoptimized IR (as with CUDA_OXIDE_NO_OPT=1)".to_string(),
        );
    }
    if opts.verbose {
        diagnostics.push(format!(
            "LLVM toolchain: llc = {}, opt = {}, llvm-link = {}",
            crate::llvm_tools::describe_tool(&toolchain.llc_path, toolchain.llc_major),
            match &toolchain.opt {
                Some(tool) => crate::llvm_tools::describe_tool(&tool.path, tool.major),
                None => "(skipped)".to_string(),
            },
            match &toolchain.llvm_link {
                Some(tool) => crate::llvm_tools::describe_tool(&tool.path, tool.major),
                None => "(not found)".to_string(),
            }
        ));
    }
    emit_diagnostics(diagnostic_sink, &diagnostics);
    let mut generated = generate_ptx_impl(
        module,
        debug_kind,
        PtxBackend {
            options: validated,
            toolchain,
            generated,
        },
        false,
        diagnostic_sink,
        libdevice_path,
    )?;
    diagnostics.append(&mut generated.diagnostics);
    generated.diagnostics = diagnostics;
    Ok(generated)
}

/// Generate PTX with an already-resolved toolchain.
///
/// The standalone compiler uses this entry point so discovery is explicit
/// and one [`LlvmToolchain`] can be reused across compilations.
pub(crate) fn generate_ptx_with_toolchain(
    module: PtxModule<'_>,
    debug_kind: DebugKind,
    validated: &ValidatedBackendOptions<'_>,
    toolchain: &LlvmToolchain,
    generated: &GeneratedModuleRequirements,
    libdevice_path: Option<&Path>,
) -> Result<GeneratedPtx, PipelineError> {
    generate_ptx_impl(
        module,
        debug_kind,
        PtxBackend {
            options: validated,
            toolchain,
            generated,
        },
        true,
        None,
        libdevice_path,
    )
}

fn generate_ptx_impl(
    module: PtxModule<'_>,
    debug_kind: DebugKind,
    backend: PtxBackend<'_>,
    strict_optimization: bool,
    diagnostic_sink: Option<fn(&str)>,
    libdevice_path: Option<&Path>,
) -> Result<GeneratedPtx, PipelineError> {
    let PtxBackend {
        options: validated,
        toolchain,
        generated,
    } = backend;
    let opts = validated.options();
    // Explicit, hard override: `--arch` or a caller-set `opts.target_arch`.
    let explicit_override = opts.target_arch.clone();
    // Advisory hint: the arch of the GPU in this machine, forwarded by
    // `cargo oxide run`. Used only when that GPU can actually run the kernel.
    let device_hint = validated.device_arch();

    let requirements = merge_generated_module_requirements(
        detect_module_requirements_in_llvm_file(module.llvm_ir)?,
        generated,
    )
    .map_err(PipelineError::PtxGeneration)?;
    let detected = requirements.features;

    // Resolve the final target:
    //   1. explicit override -- accepted only if it can lower the kernel's
    //      features; reject an invalid floor before llc emits unusable PTX.
    //   2. detected-device hint -- used only if that GPU can run the kernel;
    //      otherwise we build for the feature floor. The resulting PTX will not
    //      load on this GPU, but feature-gated examples handle that at load time
    //      (cuModuleLoad reports INVALID_PTX and they skip execution).
    //   3. neither set -- the feature floor.
    let (target, target_source) = resolve_ptx_target_with_generated(
        explicit_override.as_deref(),
        opts.target_arch_source,
        device_hint,
        detected,
        generated,
    )?;
    let requirements =
        merge_generated_module_requirements_for_target(requirements, generated, &target)
            .map_err(PipelineError::PtxGeneration)?;

    let mut diagnostics = Vec::new();
    if opts.verbose {
        record_diagnostic(
            &mut diagnostics,
            diagnostic_sink,
            format!(
                "Target: {} (from {target_source}; detected {detected:?})",
                target.sm()
            ),
        );
    }

    validate_target_for_llvm_major(&target, toolchain.llc_major)
        .map_err(PipelineError::PtxGeneration)?;

    // Link libdevice at the IR level when the kernel uses `__nv_*` calls.
    // This resolves (and later inlines) CUDA math functions without forcing
    // the legacy NVVM IR path, which cannot represent f16 on pre-Blackwell.
    let linked = match libdevice_path {
        Some(lp) => Some(link_libdevice(
            module.llvm_ir,
            lp,
            toolchain,
            diagnostic_sink,
            &mut diagnostics,
            opts.verbose,
        )?),
        None => None,
    };
    let post_link_input: &Path = linked.as_deref().unwrap_or(module.llvm_ir);

    // Run the LLVM middle-end (opt -O2) before llc. Source requirements are
    // detected above so target selection cannot lose a source-level contract
    // merely because optimization elides it. Requirements are detected again
    // from the exact llc input below because linking and optimization can also
    // introduce backend intrinsics (notably llvm.stacksave/stackrestore).
    //
    // Full-debug is a `-G`-style build: it keeps eligible imported locals in
    // memory and describes them with `llvm.dbg.declare`. Running `opt -O2`
    // would promote those slots to registers and collapse their live ranges,
    // turning most in-scope locals into `<optimized out>` under cuda-gdb. So we
    // feed the unoptimized IR straight to llc when variable info is requested,
    // matching nvcc `-G`. (llc itself uses `-O0` for these builds below.)
    let optimized = if debug_kind.variables_enabled() {
        if opts.verbose {
            record_diagnostic(
                &mut diagnostics,
                diagnostic_sink,
                "Skipping opt -O2 (full debug keeps locals inspectable)".to_string(),
            );
        }
        None
    } else {
        let (optimized, mut opt_diagnostics) = optimize_ll(
            post_link_input,
            module.public_symbols,
            toolchain,
            opts,
            strict_optimization,
            debug_kind,
        )?;
        for diagnostic in opt_diagnostics.drain(..) {
            record_diagnostic(&mut diagnostics, diagnostic_sink, diagnostic);
        }
        optimized
    };
    let llc_input: &Path = optimized.as_deref().unwrap_or(post_link_input);

    // Diagnose Rust locals only from the successful post-O2 input that will
    // actually reach llc. Looking earlier would report slots that LLVM SROA
    // still removes, while full-debug and no-opt builds intentionally retain
    // stack storage and therefore are not meaningful promotion diagnostics.
    if opts.verbose && optimized.is_some() && !debug_kind.variables_enabled() {
        match crate::local_memory_diagnostic::diagnose_file(llc_input) {
            Ok(local_memory_diagnostics) => {
                for diagnostic in local_memory_diagnostics {
                    record_diagnostic(&mut diagnostics, diagnostic_sink, diagnostic);
                }
            }
            Err(error) => record_diagnostic(
                &mut diagnostics,
                diagnostic_sink,
                format!(
                    "warning: could not inspect optimized LLVM IR for local-memory promotion diagnostics ({}): {error}",
                    llc_input.display()
                ),
            ),
        }
    }

    let llc_requirements = detect_module_requirements_in_llvm_file(llc_input)?;
    let requirements = merge_module_requirements(requirements, llc_requirements);
    let requirements =
        merge_generated_module_requirements_for_target(requirements, generated, &target)
            .map_err(PipelineError::PtxGeneration)?;
    validate_target_features(&target, requirements.features).map_err(|reason| {
        PipelineError::TargetSelection {
            target: target.sm(),
            reason: format!("{reason} (requirements from the final LLVM input)"),
        }
    })?;
    validate_ptx_isa_for_llvm_major(requirements.ptx_isa, toolchain.llc_major)
        .map_err(PipelineError::PtxGeneration)?;

    let llc_desc = if toolchain.llc_from_env {
        format!("llc_override ({})", toolchain.llc_path)
    } else {
        format!("llc ({})", toolchain.llc_path)
    };
    if opts.verbose {
        let source = if toolchain.llc_from_env {
            "from opts.llc_override"
        } else {
            "auto-detected"
        };
        record_diagnostic(
            &mut diagnostics,
            diagnostic_sink,
            format!(
                "Using llc: {} ({source})",
                crate::llvm_tools::describe_tool(&toolchain.llc_path, toolchain.llc_major)
            ),
        );
    }

    let mut llc_cmd = std::process::Command::new(&toolchain.llc_path);
    llc_cmd.args(base_llc_args(&target.sm()));
    if let Some(feature) = required_ptx_feature(
        &target,
        ptx_isa_with_debug_floor(requirements.ptx_isa, debug_kind),
    )
    .map_err(PipelineError::PtxGeneration)?
    {
        llc_cmd.arg(format!("-mattr={feature}"));
    }
    // Full-debug (`-G`-style): run llc at -O0 so its own mem2reg/SROA does not
    // promote the stack slots we deliberately kept in memory, which would
    // invalidate the `llvm.dbg.declare` locations cuda-gdb reads.
    if debug_kind.variables_enabled() {
        llc_cmd.arg("-O0");
    }
    // Fuse fmul+fadd/fsub into fma.rn.f32, matching nvcc's default --fmad=true.
    // The IR-side `contract` flag (set during lowering when contraction is
    // allowed) grants permission; this llc flag activates the NVPTX backend's
    // contract mode. `opts.no_fma` (allow_fma_contraction = !no_fma) drives both
    // stages, so IR permission and this backend gate cannot disagree.
    if !opts.no_fma {
        llc_cmd.arg("-fp-contract=fast");
    }
    // Match nvcc's precision defaults when libdevice is in the module.
    // libdevice selects between correctly-rounded and approximate
    // implementations by reading these through `__nvvm_reflect`, and LLVM
    // resolves an unset reflect variable to 0, which is the approximate
    // branch. nvcc defaults `-prec-sqrt=true` and `-prec-div=true`, so 0
    // diverges from it silently and in the direction of lower accuracy.
    // `__CUDA_FTZ` is left unset deliberately: nvcc defaults `-ftz=false`,
    // which 0 already matches.
    //
    // Not every libdevice build branches division on `__CUDA_PREC_DIV`;
    // current libdevice builds define no such reflect name, so here the
    // argument matches no `__nvvm_reflect` call and has no effect. It stays
    // because `NVVMReflectPass` silently drops a name absent from the
    // module, emitting neither a warning nor a failure, and any libdevice
    // build that does branch on `__CUDA_PREC_DIV` needs the same
    // nvcc-matching value.
    if linked.is_some() {
        llc_cmd
            .arg("--nvvm-reflect-add=__CUDA_PREC_SQRT=1")
            .arg("--nvvm-reflect-add=__CUDA_PREC_DIV=1");
    }
    let result = llc_cmd.arg(llc_input).arg("-o").arg(module.output).output();

    match result {
        Ok(output) if output.status.success() => {
            reject_dropped_debug_info(&llc_desc, &output.stderr, debug_kind)
                .map_err(PipelineError::PtxGeneration)?;
            verify_no_shared_symbols_in_initializers(module.output)?;
            verify_no_leaked_intrinsic_externs(module.output)?;
            if matches!(debug_kind, DebugKind::LineTables) {
                strip_target_debug_from_ptx(module.output)?;
                if opts.verbose {
                    record_diagnostic(
                        &mut diagnostics,
                        diagnostic_sink,
                        "line-table debug: stripped PTX target debug flag; source line tables remain"
                            .to_string(),
                    );
                }
            }
            Ok(GeneratedPtx {
                target: target.to_string(),
                diagnostics,
            })
        }
        Ok(output) => Err(PipelineError::PtxGeneration(format!(
            "{} failed:\n{}",
            llc_desc,
            String::from_utf8_lossy(&output.stderr).trim()
        ))),
        Err(e) => Err(PipelineError::PtxGeneration(format!("{llc_desc}: {e}"))),
    }
}

fn merge_module_requirements(
    source: ModuleRequirements,
    llc_input: ModuleRequirements,
) -> ModuleRequirements {
    ModuleRequirements {
        features: source.features | llc_input.features,
        ptx_isa: source.ptx_isa.max(llc_input.ptx_isa),
    }
}

fn emit_diagnostics(sink: Option<fn(&str)>, diagnostics: &[String]) {
    if let Some(sink) = sink {
        for diagnostic in diagnostics {
            sink(diagnostic);
        }
    }
}

fn record_diagnostic(diagnostics: &mut Vec<String>, sink: Option<fn(&str)>, diagnostic: String) {
    if let Some(sink) = sink {
        sink(&diagnostic);
    }
    diagnostics.push(diagnostic);
}

/// Build-time diagnostic: reject PTX whose `.global`/`.const` initializers
/// reference `.shared` symbols.
///
/// The safety net behind [`DISABLE_SWITCH_LOOKUP_TABLES`]. That flag turns
/// off the one KNOWN producer of this impossible object; this scan catches
/// any future producer, whichever pass invents it:
///
/// ```text
/// llc: prints the bad initializer, exit 0, says nothing
///        │
///        ▼ this scan (microseconds, right after llc)
/// .global .u64 t[4] = { __shared_mem_0, .. }   → BUILD FAILS, offending
///                     { generic(...), .. }       line printed, instead of
///                                                CUDA_ERROR_INVALID_PTX
///                                                at JIT load on a real GPU
/// ```
///
/// ptxas rejects both encodings ("Variable used as initial value not in
/// .global or .const state space"); a shared address is a per-block runtime
/// value and can never sit in a data-space initializer.
fn verify_no_shared_symbols_in_initializers(ptx_path: &Path) -> Result<(), PipelineError> {
    let ptx = std::fs::read_to_string(ptx_path).map_err(|e| {
        PipelineError::PtxGeneration(format!(
            "failed to read PTX for shared-initializer verification ({}): {e}",
            ptx_path.display()
        ))
    })?;
    if let Err(message) = scan_for_shared_symbols_in_initializers(&ptx) {
        return Err(PipelineError::PtxGeneration(format!(
            "{} contains a .global/.const initializer referencing a .shared symbol; \
             ptxas rejects this (\"Variable used as initial value not in .global or \
             .const state space\") and driver JIT fails with CUDA_ERROR_INVALID_PTX:\n\
             {message}\n\
             This is a compiler bug: an optimization materialized shared-memory \
             addresses into a data-space initializer.",
            ptx_path.display()
        )));
    }
    Ok(())
}

/// Pure scan half of [`verify_no_shared_symbols_in_initializers`]: returns
/// the offending line on failure.
fn scan_for_shared_symbols_in_initializers(ptx: &str) -> Result<(), String> {
    // Pass 1: collect the names of `.shared`-space variables. Declarations
    // look like `.shared .align 16 .b8 __shared_mem_0[16384];`, optionally
    // with a linking directive (`.visible`, `.extern`, `.weak`, ...) first.
    let shared_symbols: Vec<&str> = ptx
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            let decl = trimmed
                .strip_prefix(".visible ")
                .or_else(|| trimmed.strip_prefix(".extern "))
                .or_else(|| trimmed.strip_prefix(".weak "))
                .or_else(|| trimmed.strip_prefix(".common "))
                .unwrap_or(trimmed);
            if !decl.starts_with(".shared") {
                return None;
            }
            // The symbol is the last identifier before `[`, `;`, or `=`.
            let name_part = decl.split(['[', ';', '=']).next()?;
            let name = name_part.split_whitespace().next_back()?;
            (!name.starts_with('.')).then_some(name)
        })
        .collect();
    if shared_symbols.is_empty() {
        return Ok(());
    }

    // Pass 2: any initialized `.global`/`.const` declaration whose
    // initializer mentions a shared symbol (as a whole identifier) is
    // unassemblable, whether bare or wrapped in `generic(...)`.
    for line in ptx.lines() {
        let trimmed = line.trim_start();
        let decl = trimmed
            .strip_prefix(".visible ")
            .or_else(|| trimmed.strip_prefix(".extern "))
            .or_else(|| trimmed.strip_prefix(".weak "))
            .unwrap_or(trimmed);
        if !(decl.starts_with(".global") || decl.starts_with(".const")) {
            continue;
        }
        let Some((_, initializer)) = decl.split_once('=') else {
            continue;
        };
        for symbol in &shared_symbols {
            if contains_ptx_identifier(initializer, symbol) {
                return Err(format!("  {}", line.trim()));
            }
        }
    }
    Ok(())
}

/// Build-time diagnostic: reject PTX where llc leaked an unsupported LLVM
/// intrinsic as an extern function declaration.
///
/// An llc that predates an intrinsic does not error on it. It "lowers" the
/// call by inventing an extern function with the LLVM-internal name:
///
/// ```text
/// our IR:  call @llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3(...)
///            │ llc too old to know stmatrix: exit 0, says nothing
///            ▼
/// PTX:  .extern .func llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3
///                      ▲
///        dots are not legal in PTX identifiers → ptxas: "Parsing error
///        near '.nvvm'" (or driver JIT CUDA_ERROR_INVALID_PTX at runtime)
/// ```
///
/// This scan names the intrinsic and the real remedy (a newer llc /
/// `CUDA_OXIDE_LLC`) instead of letting a cryptic assembler parse error or
/// a runtime JIT failure stand in for "your llc is too old". Found the
/// hard way: CI's llc floor pin predated `llvm.nvvm.stmatrix.*` (needs
/// llc-22+) and shipped unassemblable tcgen05 PTX for as long as nothing
/// assembled it.
fn verify_no_leaked_intrinsic_externs(ptx_path: &Path) -> Result<(), PipelineError> {
    let ptx = std::fs::read_to_string(ptx_path).map_err(|e| {
        PipelineError::PtxGeneration(format!(
            "failed to read PTX for leaked-intrinsic verification ({}): {e}",
            ptx_path.display()
        ))
    })?;
    if let Err(symbol) = scan_for_leaked_intrinsic_externs(&ptx) {
        return Err(PipelineError::PtxGeneration(format!(
            "{} declares `.extern .func {symbol}`: llc did not recognize this \
             LLVM intrinsic and leaked its dotted internal name into the PTX, \
             which ptxas cannot parse (PTX identifiers cannot contain '.'). \
             The llc in use is too old for this intrinsic; use a newer llc \
             (set CUDA_OXIDE_LLC or install a newer LLVM).",
            ptx_path.display()
        )));
    }
    Ok(())
}

/// Pure scan half of [`verify_no_leaked_intrinsic_externs`]: returns the
/// offending symbol on failure.
///
/// A leaked intrinsic shows up as `.extern .func <name>` (optionally with a
/// parenthesized return-param group before the name) where `<name>` contains
/// a `.`. Dots are impossible in identifiers the exporter emits, so any hit
/// is an llc-side leak, not user code.
fn scan_for_leaked_intrinsic_externs(ptx: &str) -> Result<(), String> {
    for line in ptx.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix(".extern") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix(".func") else {
            continue;
        };
        // Skip an optional `(.param ...)` return group before the name.
        let mut rest = rest.trim_start();
        if rest.starts_with('(') {
            let Some(close) = rest.find(')') else {
                continue;
            };
            rest = rest[close + 1..].trim_start();
        }
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.'))
            .collect();
        if name.contains('.') {
            return Err(name);
        }
    }
    Ok(())
}

/// Whole-identifier containment for PTX symbols (`__shared_mem_1` must not
/// match inside `__shared_mem_10`). PTX identifiers use `[A-Za-z0-9_$]`.
fn contains_ptx_identifier(haystack: &str, symbol: &str) -> bool {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '$';
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(symbol) {
        let begin = start + pos;
        let end = begin + symbol.len();
        let before_ok = begin == 0 || !haystack[..begin].chars().next_back().is_some_and(is_ident);
        let after_ok = !haystack[end..].chars().next().is_some_and(is_ident);
        if before_ok && after_ok {
            return true;
        }
        start = begin + 1;
    }
    false
}

fn strip_target_debug_from_ptx(ptx_path: &Path) -> Result<(), PipelineError> {
    let ptx = std::fs::read_to_string(ptx_path).map_err(|e| {
        PipelineError::PtxGeneration(format!(
            "failed to read PTX for line-table debug cleanup ({}): {e}",
            ptx_path.display()
        ))
    })?;
    let stripped = strip_target_debug_from_ptx_text(&ptx).map_err(|error| {
        PipelineError::PtxGeneration(format!(
            "failed to edit PTX for line-table debug cleanup ({}): {error}",
            ptx_path.display()
        ))
    })?;
    if stripped != ptx {
        std::fs::write(ptx_path, stripped).map_err(|e| {
            PipelineError::PtxGeneration(format!(
                "failed to write PTX after line-table debug cleanup ({}): {e}",
                ptx_path.display()
            ))
        })?;
    }
    Ok(())
}

fn strip_target_debug_from_ptx_text(ptx: &str) -> Result<String, String> {
    let document = Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut edits = EditScript::new();
    for directive in document
        .directives()
        .iter()
        .filter(|directive| directive.name() == ".target")
    {
        let Some(arguments) = split_top_level(directive.arguments()) else {
            continue;
        };
        if arguments.first().is_none_or(|arch| *arch == "debug")
            || !arguments[1..].contains(&"debug")
        {
            continue;
        }
        let replacement = arguments
            .into_iter()
            .filter(|argument| *argument != "debug")
            .collect::<Vec<_>>()
            .join(", ");
        edits
            .replace(directive.arguments_span(), replacement)
            .map_err(|error| error.to_string())?;
    }
    edits.apply(ptx).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    static LEGACY_DIAGNOSTICS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    #[cfg(unix)]
    fn collect_legacy_diagnostic(message: &str) {
        LEGACY_DIAGNOSTICS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(message.to_string());
    }

    /// Locates a POSIX utility the tests drive as a stand-in for a real tool.
    ///
    /// The location is not portable: Linux ships `true` and `false` in `/bin`,
    /// while macOS ships them only in `/usr/bin`. Hardcoding either directory
    /// makes a `#[cfg(unix)]` test fail on a platform that predicate covers, so
    /// resolve the path instead of assuming one.
    #[cfg(unix)]
    fn posix_utility(name: &str) -> String {
        ["/bin", "/usr/bin"]
            .iter()
            .map(|directory| format!("{directory}/{name}"))
            .find(|path| Path::new(path).exists())
            .unwrap_or_else(|| panic!("no `{name}` utility in /bin or /usr/bin"))
    }

    #[test]
    #[cfg(unix)]
    fn legacy_opt_failure_warns_but_standalone_mode_fails() {
        let opt_path = posix_utility("false");
        let toolchain = LlvmToolchain {
            llc_path: posix_utility("true"),
            llc_major: Some(21),
            llc_from_env: false,
            opt: Some(crate::llvm_tools::OptTool {
                path: opt_path.clone(),
                major: Some(21),
            }),
            llvm_link: None,
            diagnostics: Vec::new(),
        };
        let opts = BackendOptions::default();
        let input = Path::new("unused.ll");

        let (optimized, diagnostics) =
            optimize_ll(input, &[], &toolchain, &opts, false, DebugKind::Off).unwrap();
        assert!(optimized.is_none());
        assert!(diagnostics[0].contains("continuing with unoptimized IR"));

        let error = optimize_ll(input, &[], &toolchain, &opts, true, DebugKind::Off).unwrap_err();
        assert!(matches!(&error, PipelineError::Optimization(_)));
        assert!(
            error
                .to_string()
                .contains(&format!("opt ({opt_path}) failed"))
        );
    }

    #[test]
    fn ptx_optimization_internalizes_helpers_but_preserves_public_roots() {
        let symbols = vec!["constant_data".into(), "first_kernel".into()];
        assert_eq!(
            optimization_args(&symbols).unwrap(),
            [
                "-passes=internalize,default<O2>",
                "-internalize-public-api-list=constant_data,first_kernel",
                "-switch-to-lookup=false",
            ]
        );
    }

    #[test]
    fn modules_without_public_roots_keep_the_existing_optimization_pipeline() {
        assert_eq!(
            optimization_args(&[]).unwrap(),
            ["-O2", "-switch-to-lookup=false"]
        );
    }

    /// Full-debug DWARF uses label-difference expressions (a PTX ISA 7.5
    /// feature), so the requirement floor rises to the nearest supported
    /// spelling, 7.8; targets already at or above it are untouched, and
    /// line-tables/off never raise (see [`ptx_isa_with_debug_floor`]).
    #[test]
    fn full_debug_raises_the_ptx_isa_floor_only_when_below() {
        assert_eq!(
            ptx_isa_with_debug_floor(PtxIsaRequirement::Default, DebugKind::Full),
            PtxIsaRequirement::new(78)
        );
        assert_eq!(
            ptx_isa_with_debug_floor(PtxIsaRequirement::new(70), DebugKind::Full),
            PtxIsaRequirement::new(78)
        );
        assert_eq!(
            ptx_isa_with_debug_floor(PtxIsaRequirement::new(86), DebugKind::Full),
            PtxIsaRequirement::new(86)
        );
        assert_eq!(
            ptx_isa_with_debug_floor(PtxIsaRequirement::Default, DebugKind::LineTables),
            PtxIsaRequirement::Default
        );
        assert_eq!(
            ptx_isa_with_debug_floor(PtxIsaRequirement::Default, DebugKind::Off),
            PtxIsaRequirement::Default
        );
        // End to end: at sm_80 (floor 7.0) the raised requirement emits the
        // feature; at sm_90a (floor 8.0) it is already satisfied.
        assert_eq!(
            required_ptx_feature(
                &"sm_80".parse().unwrap(),
                ptx_isa_with_debug_floor(PtxIsaRequirement::Default, DebugKind::Full)
            )
            .unwrap(),
            Some("+ptx78")
        );
        assert_eq!(
            required_ptx_feature(
                &"sm_90a".parse().unwrap(),
                ptx_isa_with_debug_floor(PtxIsaRequirement::Default, DebugKind::Full)
            )
            .unwrap(),
            None
        );
    }

    /// Every llc invocation must carry the branch-layout controls: without
    /// them LLVM 23's BranchFolding/MachineBlockPlacement rewrite loop
    /// branches into a form ptxas's SASS unroller does not recognize (see
    /// [`DISABLE_BRANCH_FOLD`]).
    #[test]
    fn llc_base_args_keep_the_ptxas_friendly_branch_layout() {
        assert_eq!(
            base_llc_args("sm_90"),
            [
                "-march=nvptx64",
                "-mcpu=sm_90",
                "-disable-branch-fold",
                "-disable-block-placement",
            ]
        );
    }

    #[test]
    fn final_llc_input_requirements_are_unioned_with_source_requirements() {
        use crate::target::{DetectedFeatures, PtxIsaRequirement};

        let source = ModuleRequirements {
            features: DetectedFeatures::Sm80,
            ptx_isa: PtxIsaRequirement::new(70),
        };
        let llc_input = ModuleRequirements {
            features: DetectedFeatures::DynamicStack,
            ptx_isa: PtxIsaRequirement::new(73),
        };

        let merged = merge_module_requirements(source, llc_input);
        assert_eq!(
            merged.features,
            DetectedFeatures::Sm80 | DetectedFeatures::DynamicStack
        );
        assert_eq!(merged.ptx_isa, PtxIsaRequirement::new(73));
        assert_eq!(
            required_ptx_feature(&"sm_80".parse().unwrap(), merged.ptx_isa).unwrap(),
            Some("+ptx73")
        );
    }

    /// The post-llc scan must catch every encoding of a `.shared` symbol in
    /// a data-space initializer (ptxas rejects bare and `generic()` alike)
    /// while staying quiet on legal PTX, including integer switch tables and
    /// shared symbols whose names prefix one another.
    #[test]
    fn shared_symbols_in_data_initializers_are_detected() {
        // Legal: plain shared declarations, integer lookup tables, and
        // global-to-global initializers.
        let clean = "\
.visible .global .align 8 .u64 table[4] = {1, 2, 3, 4};
.shared .align 16 .b8 __shared_mem_0[16384];
.visible .entry k()
{
\t.shared .align 8 .b8 local_buf[64];
\tst.shared.u32 [%rd1], %r2;
}
.global .align 8 .u64 ptr_table[1] = {generic(some_global)};
.global .align 4 .b8 some_global[4];
";
        assert!(scan_for_shared_symbols_in_initializers(clean).is_ok());

        // Bare reference: the exact shape LLVM 23's switch lookup tables
        // produced (`.global .u64 switch_$_table[4] = {__shared_mem_0, ...}`).
        let bare = "\
.shared .align 16 .b8 __shared_mem_0[16384];
.global .align 8 .u64 switch_$_table_$_kernel[4] = {__shared_mem_0, __shared_mem_0, __shared_mem_0, __shared_mem_0};
";
        let error = scan_for_shared_symbols_in_initializers(bare).unwrap_err();
        assert!(error.contains("switch_$_table_$_kernel"));

        // generic()-wrapped references are equally unassemblable.
        let wrapped = "\
.shared .align 16 .b8 stage_buf[64];
.visible .global .align 8 .u64 t[1] = {generic(stage_buf)};
";
        assert!(scan_for_shared_symbols_in_initializers(wrapped).is_err());

        // .const initializers follow the same ptxas rule as .global.
        let const_space = "\
.shared .align 8 .b8 s[8];
.const .align 8 .u64 c[1] = {s};
";
        assert!(scan_for_shared_symbols_in_initializers(const_space).is_err());

        // Identifier boundaries: shared `__shared_mem_1` must not flag an
        // initializer that references the (global) `__shared_mem_10`.
        let prefixed = "\
.shared .align 8 .b8 __shared_mem_1[8];
.global .align 8 .b8 __shared_mem_10[8];
.global .align 8 .u64 t[1] = {__shared_mem_10};
";
        assert!(scan_for_shared_symbols_in_initializers(prefixed).is_ok());
    }

    #[test]
    fn leaked_intrinsic_extern_is_rejected_with_its_name() {
        // Exactly what llc-21 emits for an intrinsic it predates (observed
        // for llvm.nvvm.stmatrix.* in gemm_sol under the CI floor pin).
        let leaked = ".version 8.6\n.target sm_100a\n\
             .extern .func llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3\n\
             (\n\t.param .b64 p0\n)\n;\n";
        let symbol = scan_for_leaked_intrinsic_externs(leaked).unwrap_err();
        assert_eq!(symbol, "llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3");

        // Return-param group before the name is skipped, name still found.
        let with_ret = ".extern .func (.param .b32 ret) llvm.nvvm.foo.bar (\n";
        assert_eq!(
            scan_for_leaked_intrinsic_externs(with_ret).unwrap_err(),
            "llvm.nvvm.foo.bar"
        );

        // Legitimate dotless externs (vprintf, malloc-style) stay accepted.
        let legit = ".extern .func (.param .b32 ret) vprintf (\n\
             .param .b64 fmt,\n.param .b64 args\n);\n\
             .extern .func my_device_helper();\n";
        assert!(scan_for_leaked_intrinsic_externs(legit).is_ok());
    }

    #[test]
    fn unrepresentable_public_root_is_rejected() {
        let error = optimization_args(&["invalid,root".into()]).unwrap_err();
        assert!(matches!(error, PipelineError::Optimization(_)));
        assert!(error.to_string().contains("invalid,root"));
    }

    #[test]
    #[cfg(unix)]
    fn legacy_tool_warnings_survive_a_later_llc_failure() {
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_legacy_diagnostics_{}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let ll_path = root.join("module.ll");
        let ptx_path = root.join("module.ptx");
        let llc_path = root.join("llc-999");
        std::fs::write(&ll_path, "define void @kernel() { ret void }\n").unwrap();
        std::fs::write(
            &llc_path,
            "#!/bin/sh\nif [ \"${1:-}\" = \"--version\" ]; then echo 'LLVM version 999.0.0'; exit 0; fi\necho 'deliberate llc failure' >&2\nexit 1\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&llc_path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&llc_path, permissions).unwrap();

        LEGACY_DIAGNOSTICS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        let opts = BackendOptions {
            target_arch: Some("sm_80".to_string()),
            no_opt: false,
            llc_override: Some(llc_path),
            ..BackendOptions::default()
        };
        let error = generate_ptx(
            PtxModule {
                llvm_ir: &ll_path,
                output: &ptx_path,
                public_symbols: &[],
            },
            DebugKind::Off,
            &ValidatedBackendOptions::for_test(&opts).unwrap(),
            Some(collect_legacy_diagnostic),
            &GeneratedModuleRequirements::default(),
            None,
        )
        .unwrap_err();
        assert!(matches!(error, PipelineError::PtxGeneration(_)));

        let diagnostics = LEGACY_DIAGNOSTICS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            diagnostics
                .iter()
                .any(|message| message.contains("LLVM optimization is unavailable")),
            "{diagnostics:?}"
        );
        assert!(
            diagnostics
                .iter()
                .any(|message| message.contains("continuing with unoptimized IR")),
            "{diagnostics:?}"
        );
        drop(diagnostics);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Writes an executable stand-in for `llc`/`opt` that answers `--version`
    /// with LLVM 23, writes a minimal artifact to its `-o` path, and reports
    /// on stderr that it discarded the module's debug metadata.
    #[cfg(unix)]
    fn write_debug_dropping_tool(path: &Path, artifact: &str) {
        std::fs::write(
            path,
            format!(
                "#!/bin/sh\n\
                 if [ \"${{1:-}}\" = \"--version\" ]; then echo 'LLVM version 23.1.0'; exit 0; fi\n\
                 out=''\n\
                 while [ $# -gt 0 ]; do if [ \"$1\" = \"-o\" ]; then out=\"$2\"; shift; fi; shift; done\n\
                 printf '%s' '{artifact}' > \"$out\"\n\
                 echo 'warning: ignoring invalid debug info in module.ll' >&2\n\
                 exit 0\n"
            ),
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(path, permissions).unwrap();
    }

    /// llc exits 0 after LLVM's verifier strips a malformed debug graph, so
    /// a debug build must treat that warning as a failure rather than ship
    /// PTX with no line table (the 2026-09 LLVM 23 function-local static
    /// regression, where cuda-gdb could only stop at host shadow lines).
    #[test]
    #[cfg(unix)]
    fn debug_builds_fail_when_llc_drops_the_debug_graph() {
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_llc_dropped_debug_{}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let ll_path = root.join("module.ll");
        let ptx_path = root.join("module.ptx");
        let llc_path = root.join("llc-fake");
        std::fs::write(&ll_path, "define void @kernel() { ret void }\n").unwrap();
        write_debug_dropping_tool(&llc_path, ".version 8.7\n.target sm_80\n.address_size 64\n");
        let opts = BackendOptions {
            target_arch: Some("sm_80".to_string()),
            no_opt: true,
            llc_override: Some(llc_path),
            ..BackendOptions::default()
        };
        let module = || PtxModule {
            llvm_ir: &ll_path,
            output: &ptx_path,
            public_symbols: &[],
        };

        for debug_kind in [DebugKind::Full, DebugKind::LineTables] {
            let error = generate_ptx(
                module(),
                debug_kind,
                &ValidatedBackendOptions::for_test(&opts).unwrap(),
                None,
                &GeneratedModuleRequirements::default(),
                None,
            )
            .unwrap_err();
            assert!(
                matches!(&error, PipelineError::PtxGeneration(message)
                    if message.contains("rejected the module's debug metadata")
                        && message.contains(LLVM_INVALID_DEBUG_INFO_WARNING)),
                "{debug_kind:?}: {error}"
            );
        }

        // Without debug information there is no graph to lose; the same
        // warning is not a failure.
        generate_ptx(
            module(),
            DebugKind::Off,
            &ValidatedBackendOptions::for_test(&opts).unwrap(),
            None,
            &GeneratedModuleRequirements::default(),
            None,
        )
        .expect("debug-off builds ignore the verifier's debug warning");
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Line-table builds run `opt -O2`, which prints the same warning and
    /// exits 0 when it strips the debug graph; that must fail even in the
    /// lenient (non-strict) optimization mode.
    #[test]
    #[cfg(unix)]
    fn debug_builds_fail_when_opt_drops_the_debug_graph() {
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_opt_dropped_debug_{}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let ll_path = root.join("module.ll");
        let opt_path = root.join("opt-fake");
        std::fs::write(&ll_path, "define void @kernel() { ret void }\n").unwrap();
        write_debug_dropping_tool(&opt_path, "define void @kernel() { ret void }\n");
        let toolchain = LlvmToolchain {
            llc_path: posix_utility("true"),
            llc_major: Some(23),
            llc_from_env: false,
            opt: Some(crate::llvm_tools::OptTool {
                path: opt_path.to_string_lossy().into_owned(),
                major: Some(23),
            }),
            llvm_link: None,
            diagnostics: Vec::new(),
        };
        let opts = BackendOptions::default();

        let error = optimize_ll(
            &ll_path,
            &[],
            &toolchain,
            &opts,
            false,
            DebugKind::LineTables,
        )
        .unwrap_err();
        assert!(
            matches!(&error, PipelineError::Optimization(message)
                if message.contains("rejected the module's debug metadata")),
            "{error}"
        );

        let (optimized, _) =
            optimize_ll(&ll_path, &[], &toolchain, &opts, false, DebugKind::Off).unwrap();
        assert!(
            optimized.is_some(),
            "debug-off optimization keeps opt's output"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Whether `c` can appear in a PTX identifier (`followsym`).
    fn is_ptx_identifier_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_' || c == '$'
    }

    /// Whether `line` contains `name` as a complete identifier token.
    ///
    /// A plain substring check treats `__nv_sin` as present in a line that
    /// only mentions `__nv_sinf`, so a call to the latter would wrongly mark
    /// the former as referenced. Require the match to end at a
    /// non-identifier character (and not to be the tail of a longer
    /// identifier either).
    fn contains_identifier_token(line: &str, name: &str) -> bool {
        let mut search_from = 0;
        while let Some(pos) = line[search_from..].find(name) {
            let start = search_from + pos;
            let end = start + name.len();
            let before_ok = line[..start]
                .chars()
                .next_back()
                .is_none_or(|c| !is_ptx_identifier_char(c));
            let after_ok = line[end..]
                .chars()
                .next()
                .is_none_or(|c| !is_ptx_identifier_char(c));
            if before_ok && after_ok {
                return true;
            }
            search_from = start + 1;
        }
        false
    }

    /// Names of `.visible .func` symbols in `ptx` that start with `__nv_`.
    ///
    /// llc prints void-returning definitions as `.visible .func __nv_foo(`
    /// but value-returning ones as
    /// `.visible .func  (.param .b32 func_retval0) __nv_clz(`, so a naive
    /// `.visible .func __nv_` substring check misses everything with a
    /// return value. Skip past the optional return-parameter clause and
    /// inspect the declared symbol name itself.
    fn exported_nv_functions(ptx: &str) -> Vec<String> {
        let mut exported: Vec<String> = Vec::new();
        for line in ptx.lines() {
            let Some(rest) = line.trim_start().strip_prefix(".visible") else {
                continue;
            };
            let Some(idx) = rest.find(".func") else {
                continue;
            };
            let mut rest = rest[idx + ".func".len()..].trim_start();
            // Skip the `(.param .b32 func_retval0)` clause of
            // value-returning functions.
            if let Some(after_open) = rest.strip_prefix('(') {
                let Some(close) = after_open.find(')') else {
                    continue;
                };
                rest = after_open[close + 1..].trim_start();
            }
            let name: String = rest
                .chars()
                .take_while(|c| is_ptx_identifier_char(*c))
                .collect();
            if name.starts_with("__nv_") {
                exported.push(name);
            }
        }
        exported
    }

    /// `__nv_*` function definitions in `ptx` that no PTX `call` references.
    ///
    /// A definition line carries `.func`/`.entry` and opens a body (it does
    /// not end with `;` like the forward declarations llc prints for
    /// callees). Anything imported from libdevice but never called is bloat
    /// that `--internalize --only-needed` + `opt` must have eliminated.
    fn unreferenced_nv_definitions(ptx: &str) -> Vec<String> {
        let mut defined: Vec<String> = Vec::new();
        for line in ptx.lines() {
            let trimmed = line.trim();
            if !trimmed.contains(".func") || trimmed.ends_with(';') {
                continue;
            }
            if let Some(idx) = trimmed.find("__nv_") {
                let name: String = trimmed[idx..]
                    .chars()
                    .take_while(|c| is_ptx_identifier_char(*c))
                    .collect();
                defined.push(name);
            }
        }
        defined.retain(|name| {
            !ptx.lines()
                .any(|line| line.contains("call") && contains_identifier_token(line, name))
        });
        defined
    }

    /// Toolchain-free coverage for the PTX detectors used by the libdevice
    /// link regression test (which skips on machines without llc/opt/
    /// llvm-link/libdevice).
    #[test]
    fn nv_detectors_handle_retval_clauses_and_identifier_boundaries() {
        let ptx = "\
.visible .entry kernel(
.visible .func __nv_void_helper(
.visible .func  (.param .b32 func_retval0) __nv_clz(
.func  (.param .b32 func_retval0) __nv_internal_only(
\tcall.uni (retval0), __nv_sinf, (param0);
";
        // Value-returning exports (retval clause between `.func` and the
        // name) must be caught, internal (non-.visible) ones must not.
        assert_eq!(exported_nv_functions(ptx), ["__nv_void_helper", "__nv_clz"]);

        // `__nv_sin` is not referenced by a call to `__nv_sinf`.
        let call_line = "\tcall.uni (retval0), __nv_sinf, (param0);";
        assert!(contains_identifier_token(call_line, "__nv_sinf"));
        assert!(!contains_identifier_token(call_line, "__nv_sin"));
        assert!(!contains_identifier_token("x__nv_sinf(", "__nv_sinf"));

        let defs_and_call = "\
.visible .func __nv_sin(
.func  (.param .b32 func_retval0) __nv_sinf(
\tcall.uni (retval0), __nv_sinf, (param0);
";
        assert_eq!(unreferenced_nv_definitions(defs_and_call), ["__nv_sin"]);
    }

    /// Regression test for IR-level libdevice linking: without
    /// `--internalize --only-needed` on the `llvm-link` invocation, all ~350
    /// libdevice bodies keep external linkage, survive `opt -O2`, and a
    /// one-call kernel's PTX balloons from ~130 to ~22,000 lines with 349
    /// exported `.visible .func __nv_*` definitions.
    #[test]
    fn linked_libdevice_ptx_has_no_unreferenced_nv_definitions() {
        // Needs a full toolchain (llc + same-major opt and llvm-link) and a
        // discoverable libdevice.10.bc; skip quietly on machines without a
        // CUDA toolkit or LLVM tools.
        let opts = BackendOptions {
            target_arch: Some("sm_80".to_string()),
            ..BackendOptions::default()
        };
        let Some(toolchain) = LlvmToolchain::resolve(&opts) else {
            return;
        };
        if toolchain.opt.is_none() || toolchain.llvm_link.is_none() {
            return;
        }
        let Ok(libdevice) = libnvvm_sys::find_libdevice() else {
            return;
        };

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_libdevice_link_{}_{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let ll_path = root.join("kernel.ll");
        let ptx_path = root.join("kernel.ptx");
        std::fs::write(
            &ll_path,
            "target datalayout = \"e-i64:64-i128:128-v16:16-v32:32-n16:32:64\"\n\
             target triple = \"nvptx64-nvidia-cuda\"\n\
             \n\
             declare float @__nv_sinf(float)\n\
             \n\
             define ptx_kernel void @kernel(ptr %out, float %x) {\n\
               %s = call float @__nv_sinf(float %x)\n\
               store float %s, ptr %out\n\
               ret void\n\
             }\n",
        )
        .unwrap();

        let target = generate_ptx_with_toolchain(
            PtxModule {
                llvm_ir: &ll_path,
                output: &ptx_path,
                public_symbols: &["kernel".to_string()],
            },
            DebugKind::Off,
            &ValidatedBackendOptions::for_test(&opts).unwrap(),
            &toolchain,
            &GeneratedModuleRequirements::default(),
            Some(&libdevice),
        )
        .unwrap();
        assert_eq!(target.target, "sm_80");

        let ptx = std::fs::read_to_string(&ptx_path).unwrap();
        assert!(
            ptx.contains(".visible .entry kernel"),
            "the kernel itself must stay exported:\n{ptx}"
        );
        let exported = exported_nv_functions(&ptx);
        assert!(
            exported.is_empty(),
            "libdevice bodies must be internalized, not exported; found {} `.visible .func` \
             definitions: {exported:?}",
            exported.len()
        );
        let unreferenced = unreferenced_nv_definitions(&ptx);
        assert!(
            unreferenced.is_empty(),
            "linked PTX contains unreferenced __nv_* definitions: {unreferenced:?}"
        );
        assert!(
            ptx.lines().count() < 1_000,
            "linked PTX for a one-call kernel should be O(100) lines, got {}",
            ptx.lines().count()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Regression test for switch-to-lookup-table suppression: a switch whose
    /// phi results are per-stage `.shared` buffer pointers (the multi-stage
    /// pipeline pattern in the tcgen05 GEMM examples) must not become a
    /// `.global` data table. LLVM 23's `opt` builds `[N x ptr addrspace(3)]`
    /// tables for it, and llc prints the entries as bare `.shared` symbols in
    /// a `.global` initializer, which ptxas rejects ("Variable used as
    /// initial value not in .global or .const state space") and driver JIT
    /// fails with CUDA_ERROR_INVALID_PTX.
    #[test]
    fn switch_over_shared_pointers_does_not_become_a_global_lookup_table() {
        let opts = BackendOptions {
            target_arch: Some("sm_80".to_string()),
            ..BackendOptions::default()
        };
        let Some(toolchain) = LlvmToolchain::resolve(&opts) else {
            return;
        };
        if toolchain.opt.is_none() {
            return;
        }

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_shared_switch_{}_{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let ll_path = root.join("kernel.ll");
        let ptx_path = root.join("kernel.ptx");
        // A 4-way stage switch selecting both a `.shared`-space pointer and
        // its generic (addrspacecast) counterpart, mirroring how mbarrier
        // (shared-space) and TMA destination (generic) pointers reach the
        // merge block in the real pipelines.
        std::fs::write(
            &ll_path,
            "target datalayout = \"e-i64:64-i128:128-v16:16-v32:32-n16:32:64\"\n\
             target triple = \"nvptx64-nvidia-cuda\"\n\
             \n\
             @stage0 = addrspace(3) global [64 x i8] zeroinitializer, align 8\n\
             @stage1 = addrspace(3) global [64 x i8] zeroinitializer, align 8\n\
             @stage2 = addrspace(3) global [64 x i8] zeroinitializer, align 8\n\
             @stage3 = addrspace(3) global [64 x i8] zeroinitializer, align 8\n\
             \n\
             define ptx_kernel void @kernel(ptr %out, i32 %x) {\n\
             entry:\n\
               %s = and i32 %x, 3\n\
               switch i32 %s, label %unreach [\n\
                 i32 0, label %c0\n\
                 i32 1, label %c1\n\
                 i32 2, label %c2\n\
                 i32 3, label %merge\n\
               ]\n\
             unreach:\n\
               unreachable\n\
             c0:\n\
               br label %merge\n\
             c1:\n\
               br label %merge\n\
             c2:\n\
               br label %merge\n\
             merge:\n\
               %shared = phi ptr addrspace(3) [ @stage0, %c0 ], [ @stage1, %c1 ], \
                 [ @stage2, %c2 ], [ @stage3, %entry ]\n\
               %generic = phi ptr [ addrspacecast (ptr addrspace(3) @stage0 to ptr), %c0 ], \
                 [ addrspacecast (ptr addrspace(3) @stage1 to ptr), %c1 ], \
                 [ addrspacecast (ptr addrspace(3) @stage2 to ptr), %c2 ], \
                 [ addrspacecast (ptr addrspace(3) @stage3 to ptr), %entry ]\n\
               %v1 = load i32, ptr addrspace(3) %shared, align 4\n\
               %v2 = load i32, ptr %generic, align 4\n\
               %sum = add i32 %v1, %v2\n\
               store i32 %sum, ptr %out, align 4\n\
               ret void\n\
             }\n",
        )
        .unwrap();

        generate_ptx_with_toolchain(
            PtxModule {
                llvm_ir: &ll_path,
                output: &ptx_path,
                public_symbols: &[
                    "kernel".to_string(),
                    "stage0".to_string(),
                    "stage1".to_string(),
                    "stage2".to_string(),
                    "stage3".to_string(),
                ],
            },
            DebugKind::Off,
            &ValidatedBackendOptions::for_test(&opts).unwrap(),
            &toolchain,
            &GeneratedModuleRequirements::default(),
            None,
        )
        .unwrap();

        let ptx = std::fs::read_to_string(&ptx_path).unwrap();
        // Mechanism: with `-switch-to-lookup=false` no data table forms.
        assert!(
            !ptx.contains("switch_$_table"),
            "opt built a switch lookup table despite -switch-to-lookup=false:\n{ptx}"
        );
        // Property: no initializer may reference a `.shared` symbol at all.
        // ptxas rejects both the bare `= {sym}` and the wrapped
        // `= {generic(sym)}` forms ("Variable used as initial value not in
        // .global or .const state space").
        for line in ptx.lines() {
            if let Some((_, init)) = line.split_once("= {") {
                for stage in ["stage0", "stage1", "stage2", "stage3"] {
                    assert!(
                        !init.contains(stage),
                        ".shared symbol in a .global initializer (invalid PTX): {line}"
                    );
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Regression test for libdevice reflect defaults: `libdevice.10.bc`
    /// selects between a correctly-rounded and an approximate implementation
    /// by reading `__CUDA_PREC_SQRT` through `__nvvm_reflect`. LLVM resolves an
    /// unset reflect variable to 0 and so picks the approximate branch, while
    /// nvcc defaults `-prec-sqrt=true`. Without the reflect arguments on `llc`,
    /// a `sqrt` kernel silently compiles to `sqrt.approx.f32`.
    #[test]
    fn linked_libdevice_honors_nvcc_precision_defaults() {
        let opts = BackendOptions {
            target_arch: Some("sm_80".to_string()),
            ..BackendOptions::default()
        };
        let Some(toolchain) = LlvmToolchain::resolve(&opts) else {
            return;
        };
        if toolchain.opt.is_none() || toolchain.llvm_link.is_none() {
            return;
        }
        let Ok(libdevice) = libnvvm_sys::find_libdevice() else {
            return;
        };

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_libdevice_prec_{}_{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let ll_path = root.join("kernel.ll");
        let ptx_path = root.join("kernel.ptx");
        std::fs::write(
            &ll_path,
            "target datalayout = \"e-i64:64-i128:128-v16:16-v32:32-n16:32:64\"\n\
             target triple = \"nvptx64-nvidia-cuda\"\n\
             \n\
             declare float @__nv_sqrtf(float)\n\
             \n\
             define ptx_kernel void @kernel(ptr %out, float %x) {\n\
               %s = call float @__nv_sqrtf(float %x)\n\
               store float %s, ptr %out\n\
               ret void\n\
             }\n",
        )
        .unwrap();

        generate_ptx_with_toolchain(
            PtxModule {
                llvm_ir: &ll_path,
                output: &ptx_path,
                public_symbols: &["kernel".to_string()],
            },
            DebugKind::Off,
            &ValidatedBackendOptions::for_test(&opts).unwrap(),
            &toolchain,
            &GeneratedModuleRequirements::default(),
            Some(&libdevice),
        )
        .unwrap();

        let ptx = std::fs::read_to_string(&ptx_path).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert!(
            ptx.contains("sqrt.rn.f32"),
            "libdevice sqrt must resolve to the correctly-rounded instruction, \
             matching nvcc's -prec-sqrt=true default:\n{ptx}"
        );
        assert!(
            !ptx.contains("sqrt.approx"),
            "no approximate sqrt may survive:\n{ptx}"
        );
    }

    #[test]
    fn line_table_ptx_cleanup_strips_only_target_debug_flag() {
        let ptx = "\
.version 8.9
.target sm_120a, debug
.address_size 64

.section .debug_info
\t.b8 1;
";

        let stripped = strip_target_debug_from_ptx_text(ptx).unwrap();

        assert!(
            stripped.contains(".target sm_120a\n"),
            "line-table mode should not ask the driver for debug compilation:\n{stripped}"
        );
        assert!(
            stripped.contains(".section .debug_info"),
            "line-table mode must keep the emitted DWARF sections:\n{stripped}"
        );
    }

    #[test]
    fn line_table_ptx_cleanup_preserves_other_target_options() {
        let ptx = ".target sm_90a, texmode_independent, debug\n";

        let stripped = strip_target_debug_from_ptx_text(ptx).unwrap();

        assert_eq!(stripped, ".target sm_90a, texmode_independent\n");
    }
}
