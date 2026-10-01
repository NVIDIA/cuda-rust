/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::process::Command;

use super::*;

// =============================================================================
// Run command
// =============================================================================

/// Build and run an example with the custom codegen backend.
///
/// Cleans stale artifacts, sets encoded rustc flags to point at the backend `.so`,
/// and invokes `cargo run --release` from the example directory. Environment
/// variables control output format (PTX / NVVM IR) and verbosity. Trailing
/// `app_args` are forwarded to the example binary after `--`.
#[allow(clippy::too_many_arguments)]
pub fn codegen_run(
    ctx: &Context,
    example: &str,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    device_features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialize_cubin: bool,
    app_args: &[String],
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let interop = load_interop_config(&example_dir);

    let output_format = format_label(emit_nvvm_ir);
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, emit_nvvm_ir);
    // Target precedence for `cargo oxide run` (highest first):
    //   1. --arch <sm_XX>            explicit user override   -> CUDA_OXIDE_TARGET
    //   2. CUDA_OXIDE_TARGET=<sm_XX> explicit env override (from the parent)
    //   3. detected GPU arch (via nvidia-smi) -> CUDA_OXIDE_DEVICE_ARCH (a hint)
    //   4. backend feature-based default (`select_target` in mir-importer)
    //
    // Slot 3 is a HINT, not an override: the backend builds for the detected
    // GPU only when that GPU can run the kernel. If the kernel needs a newer
    // arch (tcgen05 needs sm_100a even on a consumer sm_120 GPU), the backend
    // builds for the required arch and the module simply skips at load time.
    // The detected arch is used as a target hint for `run`. `build` may probe
    // the first GPU reported by `nvidia-smi` for diagnostics, but keeps backend
    // target selection untouched because it may legitimately cross-compile for
    // another machine.
    let detected_device_arch =
        detect_run_target_arch(target_arch, emit_nvvm_ir || materialization.enabled());

    if let Some(interop) = interop.filter(|config| !config.device_crates.is_empty()) {
        codegen_run_interop(
            ctx,
            example,
            &example_dir,
            &interop,
            verbose,
            emit_nvvm_ir,
            target_arch,
            detected_device_arch.as_deref(),
            features,
            device_features,
            bin,
            no_fmad,
            unchecked_indexing,
            device_debug,
            &materialization,
            app_args,
        );
        return;
    }
    if device_features.is_some() {
        eprintln!("Error: --device-features requires metadata-declared interop device crates.");
        std::process::exit(2);
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA: {}", example);
    println!("=========================================");
    println!();
    if materialization.enabled() {
        println!("Output format: materialized cubin");
        println!(
            "Target arch: {}",
            configured_arch_label(ctx, arch)
                .expect("materialization requires a configured architecture")
        );
        println!();
    } else if emit_nvvm_ir {
        println!("Output format: {}", output_format);
        println!(
            "Target arch: {}",
            configured_arch_label(ctx, arch)
                .expect("--emit-nvvm-ir requires a configured architecture")
        );
        println!();
    } else if let Some(dev) = detected_device_arch.as_deref() {
        // Surface the detected GPU so it isn't silent magic. It is a hint, not
        // a hard target: the backend builds for it unless a kernel needs a
        // newer arch (e.g. tcgen05 forces sm_100a even on a consumer sm_120
        // GPU), so the final PTX target may differ.
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
        println!();
    }
    println!("This is the proper cargo workflow:");
    println!("  CARGO_ENCODED_RUSTFLAGS=<cuda-oxide flags> cargo run");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--release"]).current_dir(&example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }
    if !app_args.is_empty() {
        cmd.arg("--").args(app_args);
    }

    apply_common_codegen_env(
        &mut cmd,
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );
    let fingerprint = standard_codegen_fingerprint(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        detected_device_arch.as_deref(),
        &materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    );
    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch, &materialization);
    apply_device_arch_hint(&mut cmd, target_arch, detected_device_arch.as_deref());

    if let Some(bin) = bin {
        println!("Building and running {} (bin: {})...", example, bin);
    } else {
        println!("Building and running {}...", example);
    }
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!("\nFailed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// Sanitize command
// =============================================================================

/// Build an example and run the produced host binary under NVIDIA Compute
/// Sanitizer.
#[allow(clippy::too_many_arguments)]
pub fn codegen_sanitize(
    ctx: &Context,
    example: &str,
    tool: &str,
    sanitizer_args: &[String],
    application_args: &[String],
    verbose: bool,
    arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialize_cubin: bool,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let interop = load_interop_config(&example_dir);
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, false);
    let detected_device_arch = detect_run_target_arch(target_arch, materialization.enabled());

    if let Some(interop) = interop.filter(|config| !config.device_crates.is_empty()) {
        reject_interop_output_mode(false, &materialization);
        println!("=========================================");
        println!("RUSTC-CODEGEN-CUDA SANITIZE INTEROP: {}", example);
        println!("=========================================");
        if let Some(kind) = &interop.kind {
            println!("Interop kind: {}", kind);
        }
        if let Some(dev) = detected_device_arch.as_deref() {
            println!("Detected GPU arch: {dev} (via nvidia-smi)");
        }
        println!("Compute Sanitizer tool: {tool}");
        println!();

        build_interop_device_crates(
            ctx,
            &example_dir,
            &interop,
            verbose,
            target_arch,
            detected_device_arch.as_deref(),
            None,
            InteropDeviceBuildOptions::for_route(
                InteropDeviceBuildRoute::Sanitize,
                no_fmad,
                unchecked_indexing,
                device_debug,
                false,
            ),
            &materialization,
        );
        let binary = build_host_cargo(ctx, example, &example_dir, features, bin, verbose);
        run_compute_sanitizer(
            ctx,
            &example_dir,
            tool,
            sanitizer_args,
            application_args,
            &binary,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA SANITIZE: {}", example);
    println!("=========================================");
    if let Some(dev) = detected_device_arch.as_deref() {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }
    println!("Compute Sanitizer tool: {tool}");
    println!();

    touch_main_rs(&example_dir);
    let binary = codegen_build_host_binary(
        ctx,
        example,
        &example_dir,
        verbose,
        target_arch,
        detected_device_arch.as_deref(),
        features,
        bin,
        no_fmad,
        unchecked_indexing,
        device_debug,
        &materialization,
    );
    run_compute_sanitizer(
        ctx,
        &example_dir,
        tool,
        sanitizer_args,
        application_args,
        &binary,
    );
}

pub(super) fn build_arch_warning(
    arch_configured: bool,
    local_arch: Option<&str>,
) -> Option<String> {
    if arch_configured {
        return None;
    }

    let local_arch = local_arch?;

    Some(format!(
        "Warning: no target architecture is configured; \
     `cargo oxide build` will use the backend default. \
     The first GPU reported by `nvidia-smi` has architecture {local_arch}. \
     Pass `--arch <sm_XX>` to target a specific architecture explicitly."
    ))
}

pub fn warn_for_default_build_arch(ctx: &Context, arch: Option<&str>) {
    let arch_configured = has_configured_arch(ctx, arch);
    let local_arch = if arch_configured {
        None
    } else {
        detect_local_device_arch()
    };

    if let Some(warning) = build_arch_warning(arch_configured, local_arch.as_deref()) {
        eprintln!("{warning}");
    }
}

// =============================================================================
// Build command (compile only, don't run)
// =============================================================================

/// Compile an example without running it.
///
/// Same as [`codegen_run`] but uses `cargo build --release` instead of
/// `cargo run`. Useful for cross-compilation or when the target hardware
/// (e.g., Blackwell tensor cores) isn't available on the build machine.
#[allow(clippy::too_many_arguments)]
pub fn codegen_build(
    ctx: &Context,
    example: &str,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    device_features: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    debug_assertions: bool,
    materialize_cubin: bool,
) {
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, emit_nvvm_ir);
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if let Some(interop) =
        load_interop_config(&example_dir).filter(|config| !config.device_crates.is_empty())
    {
        codegen_build_interop(
            ctx,
            example,
            &example_dir,
            &interop,
            verbose,
            emit_nvvm_ir,
            target_arch,
            features,
            device_features,
            no_fmad,
            unchecked_indexing,
            device_debug,
            debug_assertions,
            &materialization,
        );
        return;
    }
    if device_features.is_some() {
        eprintln!("Error: --device-features requires metadata-declared interop device crates.");
        std::process::exit(2);
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA BUILD: {}", example);
    println!("=========================================");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_common_codegen_env(
        &mut cmd,
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );
    let fingerprint = standard_codegen_fingerprint(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        None,
        &materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::release_like(debug_assertions),
        &[],
        &fingerprint,
    );
    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch, &materialization);

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// Inspect command
// =============================================================================

/// Build an example as PTX and print the generated artifact.
#[allow(clippy::too_many_arguments)]
pub fn codegen_inspect_ptx(
    ctx: &Context,
    example: &str,
    arch: Option<&str>,
    features: Option<&str>,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
) {
    let materialization_enabled = materialization_requested(ctx, false).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(2);
    });

    if materialization_enabled {
        eprintln!("Error: inspect requires PTX output, but {MATERIALIZE_ENV} is enabled");
        std::process::exit(2);
    }

    let nvvm_ir_enabled = nvvm_ir_requested(ctx).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(2);
    });

    if nvvm_ir_enabled {
        eprintln!("Error: inspect requires PTX output, but CUDA_OXIDE_EMIT_NVVM_IR is enabled");
        std::process::exit(2);
    }

    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };
    if load_interop_config(&example_dir).is_some_and(|interop| {
        interop
            .device_crates
            .iter()
            .any(|device_crate| device_crate.artifact_kind != InteropArtifactKind::Ptx)
    }) {
        eprintln!(
            "Error: inspect requires PTX output, but metadata declares a non-PTX device artifact."
        );
        std::process::exit(2);
    }

    codegen_build(
        ctx,
        example,
        verbose,
        false,
        arch,
        features,
        None,
        no_fmad,
        unchecked_indexing,
        device_debug,
        false,
        false,
    );

    for path in ptx_artifact_paths(&example_dir, example) {
        print_ptx_artifact(&path).unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });
    }
}
