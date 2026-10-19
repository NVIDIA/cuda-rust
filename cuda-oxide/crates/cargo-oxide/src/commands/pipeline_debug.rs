/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::process::Command;

use super::*;

// =============================================================================
// Pipeline command
// =============================================================================

/// Show verbose pipeline progress and the available intermediate artifacts.
///
/// Enables all diagnostic env vars (`CUDA_OXIDE_VERBOSE`, `SHOW_RUSTC_MIR`,
/// `DUMP_MIR`, `DUMP_LLVM`) so the user can see MIR collection, the
/// `dialect-mir` module (pre- and post-`mem2reg`), the LLVM dialect
/// module, textual LLVM IR, and the final PTX or NVVM IR. After the build,
/// generated artifacts are printed to stdout.
#[allow(clippy::too_many_arguments)]
pub fn codegen_show_pipeline(
    ctx: &Context,
    example: &str,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialize_cubin: bool,
) {
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, emit_nvvm_ir);
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if load_interop_config(&example_dir).is_some_and(|config| !config.device_crates.is_empty()) {
        reject_interop_output_mode(emit_nvvm_ir, &materialization);
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA PIPELINE: {}", example);
    println!("=========================================");
    println!();
    let target_arch_label = configured_arch_label(ctx, arch);
    match (
        materialization.enabled(),
        emit_nvvm_ir,
        target_arch_label.as_deref(),
    ) {
        (true, _, Some(target_arch)) => {
            println!("Output format: materialized cubin (arch: {target_arch})")
        }
        (false, true, Some(target_arch)) => {
            println!("Output format: NVVM IR (arch: {})", target_arch)
        }
        (false, false, Some(target_arch)) => {
            println!("Output format: PTX (arch override: {})", target_arch)
        }
        (false, false, None) => println!("Output format: PTX (auto-detected arch)"),
        (true, _, None) | (false, true, None) => {
            unreachable!("IR/final materialization requires a configured architecture")
        }
    }
    println!();
    println!("Required flags (applied via CARGO_ENCODED_RUSTFLAGS):");
    println!("  -C opt-level=3              MIR optimization");
    println!("  -C debug-assertions=off     Remove debug checks");
    println!("  -Z mir-enable-passes=-JumpThreading");
    println!("                              Prevent barrier duplication");
    println!("  -Z always-encode-mir        Emit MIR for all reachable device deps");
    println!();
    println!("Note: panic=abort is NOT required - the codegen backend treats");
    println!("      unwind paths as unreachable (CUDA toolchain limitation, not HW).");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    // Apply the common codegen environment first. `apply_codegen_rustflags` then
    // reads the resolved `CUDA_OXIDE_DEBUG` value and adds the full-debug rustc
    // controls, matching the build/run order.
    apply_common_codegen_env(
        &mut cmd,
        ctx,
        true,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );
    cmd.env("CUDA_OXIDE_SHOW_RUSTC_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_LLVM", "1");
    let fingerprint = pipeline_codegen_fingerprint(
        ctx,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
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

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");

    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }

    show_generated_artifacts(&example_dir, example);
}

// =============================================================================
// Debug command
// =============================================================================

/// Build with debug info and launch cuda-gdb (or cgdb).
///
/// Compiles the example with `-C debuginfo=2` on top of the normal release
/// flags, then launches the debugger on the resulting binary. Prints a
/// quick-reference cheat sheet for common cuda-gdb commands before handing
/// control to the debugger.
#[allow(clippy::too_many_arguments)]
pub fn codegen_debug(
    ctx: &Context,
    example: &str,
    arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    use_cgdb: bool,
    use_tui: bool,
    materialize_cubin: bool,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, false);
    if load_interop_config(&example_dir).is_some_and(|config| !config.device_crates.is_empty()) {
        reject_interop_output_mode(false, &materialization);
    }

    let cuda_gdb = find_cuda_toolkit_executable(ctx, "cuda-gdb", CUDA_GDB_FALLBACK_PATHS)
        .unwrap_or_else(|| {
            eprintln!("Error: cuda-gdb not found!");
            eprintln!();
            eprintln!("Make sure CUDA toolkit is installed and cuda-gdb is in your PATH");
            eprintln!("or configured CUDA toolkit root:");
            eprintln!("  export PATH=\"/usr/local/cuda/bin:$PATH\"");
            eprintln!("  export CUDA_TOOLKIT_PATH=/usr/local/cuda");
            std::process::exit(1);
        });

    let cgdb_path = if use_cgdb {
        Some(find_executable("cgdb", &[]).unwrap_or_else(|| {
            eprintln!("Error: cgdb not found!");
            eprintln!("Install with: sudo apt install cgdb");
            std::process::exit(1);
        }))
    } else {
        None
    };

    let detected_device_arch = detect_run_target_arch(target_arch, materialization.enabled());

    if let Some(bin) = bin {
        println!("Building {} (bin: {}) with debug info...", example, bin);
    } else {
        println!("Building {} with debug info...", example);
    }
    if let Some(dev) = detected_device_arch.as_deref() {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }

    clean_generated_files(&example_dir, example);

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_config_env(&mut cmd, ctx);
    let fingerprint = standard_codegen_fingerprint(
        ctx,
        false,
        false,
        false,
        DeviceDebug::Off,
        false,
        target_arch,
        detected_device_arch.as_deref(),
        &materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLikeWithDebugInfo,
        &[],
        &fingerprint,
    );
    cmd.env("CARGO_PROFILE_RELEASE_DEBUG", "2");
    apply_output_mode(&mut cmd, false, target_arch, &materialization);
    apply_device_arch_hint(&mut cmd, target_arch, detected_device_arch.as_deref());
    apply_ld_library_path(&mut cmd, ctx);

    let binary =
        run_cargo_build_for_executable(&mut cmd, &example_dir, bin).unwrap_or_else(|message| {
            eprintln!("Failed to build {example}: {message}");
            std::process::exit(1);
        });
    if !binary.exists() {
        eprintln!(
            "Error: Cargo reported executable artifact {}, but it does not exist",
            binary.display()
        );
        std::process::exit(1);
    }

    if cgdb_path.is_some() {
        println!("Launching cgdb (cuda-gdb frontend)...");
    } else {
        println!(
            "Launching cuda-gdb{}...",
            if use_tui { " (TUI mode)" } else { "" }
        );
    }
    println!();
    println!("Quick reference:");
    println!("  set cuda break_on_launch application");
    println!("                           - Break at start of any kernel");
    println!("  run                      - Start the program");
    println!("  info cuda kernels        - List active kernels");
    println!("  info cuda threads        - List GPU threads");
    println!("  cuda thread (0,0,0)      - Switch to thread");
    println!("  cuda block (0,0,0)       - Switch to block");
    println!("  print <var>              - Print variable");
    println!("  next / step / continue   - Execution control");
    println!("  quit                     - Exit debugger");
    if cgdb_path.is_some() {
        println!();
        println!("cgdb shortcuts:");
        println!("  Esc                      - Focus source window (vim keys work)");
        println!("  i                        - Focus command window");
        println!("  space                    - Set breakpoint on current line");
        println!("  o                        - Open file dialog");
    } else if use_tui {
        println!();
        println!("TUI shortcuts:");
        println!("  Ctrl+x a                 - Toggle TUI mode");
        println!("  Ctrl+x 2                 - Split view (source + asm)");
        println!("  Ctrl+l                   - Refresh screen");
    }
    println!();

    let status = if let Some(cgdb) = cgdb_path {
        Command::new(cgdb)
            .arg("-d")
            .arg(&cuda_gdb)
            .arg(&binary)
            .current_dir(&example_dir)
            .status()
            .expect("Failed to launch cgdb")
    } else {
        let mut gdb_cmd = Command::new(&cuda_gdb);
        if use_tui {
            gdb_cmd.arg("--tui");
        }
        gdb_cmd.arg(&binary);
        gdb_cmd.current_dir(&example_dir);
        gdb_cmd.status().expect("Failed to launch cuda-gdb")
    };

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}
