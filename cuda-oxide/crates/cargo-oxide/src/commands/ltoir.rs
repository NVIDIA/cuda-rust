/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::Path;

use super::*;

// =============================================================================
// emit-ltoir command
// =============================================================================

/// Compile a crate's device code to a binary LTOIR artifact in one step.
///
/// `cargo oxide build --emit-nvvm-ir` produces NVVM IR, which a consumer then
/// has to run through libNVVM separately to get linkable LTOIR. This folds both
/// halves into one command for the Tile-to-SIMT interop workflow (#96): it
/// builds the crate in NVVM IR mode, then compiles the emitted `<crate>.ll`
/// with libNVVM `-gen-lto` and writes `<crate>.ltoir` (or `output`) plus the
/// matching `.target` and `.options` files used for runtime loading and final
/// nvJitLink policy.
///
/// `arch` is required because LTOIR is architecture-specific. It accepts
/// `sm_XX`, `compute_XX`, or a bare `XX`, all mapped to libNVVM's
/// `-arch=compute_XX`.
#[allow(clippy::too_many_arguments)]
pub fn emit_ltoir(
    ctx: &Context,
    example: &str,
    arch: &str,
    features: Option<&str>,
    output: Option<&Path>,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if load_interop_config(&example_dir).is_some_and(|config| !config.device_crates.is_empty()) {
        eprintln!("Error: emit-ltoir does not support metadata interop examples.");
        eprintln!("Point it at a single SIMT device crate instead.");
        std::process::exit(1);
    }

    // Normalize once: libNVVM consumes compute_XX, while the compiler records
    // and nvJitLink consumes the equivalent sm_XX spelling.
    let parsed_arch = parse_nvvm_arch(arch).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });
    let sm_arch = parsed_arch.sm();

    // Step 1: build in NVVM IR mode so the backend writes `<crate>.ll` as
    // libNVVM-ready NVVM IR. codegen_build exits on build failure. Pass
    // quiet=true so the intermediate "✓ Build succeeded" line is suppressed;
    // emit_ltoir prints its own unified summary at the end.
    codegen_build(
        ctx,
        example,
        verbose,
        true,
        Some(&sm_arch),
        features,
        None,
        no_fmad,
        unchecked_indexing,
        device_debug,
        false,
        false,
    );

    // Step 2: compile that NVVM IR to LTOIR via libNVVM -gen-lto.
    let ll_path = emitted_ll_path(&example_dir, example);
    let ir = std::fs::read(&ll_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read emitted NVVM IR at {}: {e}",
            ll_path.display()
        );
        std::process::exit(1);
    });
    let source_options_path = ll_path.with_extension("options");
    let source_options = std::fs::read_to_string(&source_options_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read emitted compile options at {}: {e}",
            source_options_path.display()
        );
        std::process::exit(1);
    });
    let compile_options = oxide_artifacts::ArtifactCompileOptions::from_sidecar_text(
        &source_options,
    )
    .unwrap_or_else(|e| {
        eprintln!(
            "Error: invalid emitted compile options at {}: {e}",
            source_options_path.display()
        );
        std::process::exit(1);
    });

    let compute_arch = parsed_arch.compute();
    let ltoir = compile_nvvm_to_ltoir(&ir, example, &parsed_arch, compile_options);

    // Step 3: write the artifact.
    let out_path = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_ltoir_path(&example_dir, example));
    for metadata_path in [
        out_path.with_extension("target"),
        out_path.with_extension("options"),
    ] {
        match std::fs::remove_file(&metadata_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                eprintln!(
                    "Error: could not clear stale LTOIR metadata {}: {error}",
                    metadata_path.display()
                );
                std::process::exit(1);
            }
        }
    }
    std::fs::write(&out_path, &ltoir).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR to {}: {e}",
            out_path.display()
        );
        std::process::exit(1);
    });
    let options_path = out_path.with_extension("options");
    std::fs::write(&options_path, compile_options.sidecar_text()).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR compile options to {}: {e}",
            options_path.display()
        );
        std::process::exit(1);
    });
    let target_path = out_path.with_extension("target");
    std::fs::write(
        &target_path,
        format!(
            "{sm_arch}\n{}\n",
            oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
        ),
    )
    .unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR target metadata to {}: {e}",
            target_path.display()
        );
        std::process::exit(1);
    });

    println!();
    println!(
        "✓ LTOIR written to {} ({} bytes, {compute_arch})",
        out_path.display(),
        ltoir.len()
    );
}

/// Normalize a target architecture to libNVVM's `compute_XX` form.
///
/// Accepts `sm_XX` (the form `--arch` and the rest of cargo-oxide use),
/// `compute_XX` (passed through), or a bare `XX`.
pub(super) fn parse_nvvm_arch(
    arch: &str,
) -> Result<cuda_artifact_finalizer::CudaArch, cuda_artifact_finalizer::CudaArchParseError> {
    let normalized = if arch.starts_with("sm_") || arch.starts_with("compute_") {
        arch.to_string()
    } else {
        format!("compute_{arch}")
    };
    normalized.parse()
}

/// Compile NVVM IR text to binary LTOIR with libNVVM `-gen-lto`. Exits with a
/// diagnostic on any libNVVM failure (the program log is attached to the error).
///
fn compile_nvvm_to_ltoir(
    ir: &[u8],
    name: &str,
    arch: &cuda_artifact_finalizer::CudaArch,
    compile_options: oxide_artifacts::ArtifactCompileOptions,
) -> Vec<u8> {
    let compiler = cuda_artifact_finalizer::NvvmCompiler::discover().unwrap_or_else(|e| {
        eprintln!("Error: could not initialize the CUDA artifact compiler: {e}");
        eprintln!("libNVVM ships with the CUDA Toolkit at <CUDA>/nvvm/lib64/libnvvm.so.");
        eprintln!("Run `cargo oxide doctor` to check your toolkit setup.");
        std::process::exit(1);
    });
    let options = finalization_options_from_artifact(arch, compile_options);
    compiler
        .compile_nvvm_ir_to_ltoir(name, ir, &options)
        .unwrap_or_else(|e| {
            eprintln!("Error: libNVVM -gen-lto compilation failed: {e}");
            std::process::exit(1);
        })
}

pub(super) fn finalization_options_from_artifact(
    arch: &cuda_artifact_finalizer::CudaArch,
    compile_options: oxide_artifacts::ArtifactCompileOptions,
) -> cuda_artifact_finalizer::FinalizationOptions {
    let debug = match compile_options.debug_policy() {
        oxide_artifacts::ArtifactDebugPolicy::None => cuda_artifact_finalizer::DebugPolicy::None,
        oxide_artifacts::ArtifactDebugPolicy::LineTables => {
            cuda_artifact_finalizer::DebugPolicy::LineTables
        }
        oxide_artifacts::ArtifactDebugPolicy::Full => cuda_artifact_finalizer::DebugPolicy::Full,
    };
    cuda_artifact_finalizer::FinalizationOptions::new(arch.clone())
        .with_fma_contraction(compile_options.fma_contraction_enabled())
        .with_debug_policy(debug)
}
