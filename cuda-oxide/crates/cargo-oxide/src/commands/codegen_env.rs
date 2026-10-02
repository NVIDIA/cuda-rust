/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::backend;
use crate::cutlass_toolchain;
use std::path::Path;
use std::process::Command;

use super::*;

pub(super) const ENCODED_RUSTFLAGS_SEPARATOR: char = '\u{1f}';

/// Internal cfg used only by full-debug builds to outline
/// `DisjointSlice::get_mut`.
///
/// This fixes false CUDA-GDB helper frames without disabling MIR inlining
/// globally. User-provided copies are stripped so other modes keep their usual
/// code shape.
pub(super) const FULL_DEBUG_GET_MUT_OUTLINE_CFG: &str =
    "cuda_oxide_internal_outline_disjoint_get_mut_v1";

/// Profile-related rustc flags owned by cuda-oxide.
///
/// Backend selection and MIR/symbol invariants are always applied separately.
/// `CargoSelected` deliberately adds no optimization, assertion, or debug-info
/// flags so Cargo's chosen profile remains authoritative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CodegenProfilePolicy {
    CargoSelected,
    ReleaseLike,
    ReleaseLikeWithDebugAssertions,
    ReleaseLikeWithDebugInfo,
}

impl CodegenProfilePolicy {
    pub(super) fn release_like(debug_assertions: bool) -> Self {
        if debug_assertions {
            Self::ReleaseLikeWithDebugAssertions
        } else {
            Self::ReleaseLike
        }
    }
}

/// Construct boundary-preserving rustc flags for Cargo.
///
/// `RUSTFLAGS` is whitespace-split by Cargo, which corrupts a single flag
/// containing spaces. `CARGO_ENCODED_RUSTFLAGS` uses unit separators and keeps
/// every configured array element and `--device-cfg` value intact.
fn build_encoded_rustflags(
    ctx: &Context,
    profile: CodegenProfilePolicy,
    device_cfgs: &[String],
) -> String {
    let existing_encoded = std::env::var("CARGO_ENCODED_RUSTFLAGS").ok();
    let existing = std::env::var("RUSTFLAGS").ok();
    let mut explicit_rustflags = Vec::new();
    for cfg in device_cfgs {
        explicit_rustflags.push("--cfg".to_string());
        explicit_rustflags.push(cfg.clone());
    }
    build_encoded_rustflags_with_existing(
        &ctx.backend_so,
        profile,
        &ctx.config.extra_rustflags,
        &explicit_rustflags,
        existing_encoded.as_deref(),
        existing.as_deref(),
    )
}

pub(super) fn build_encoded_rustflags_with_existing(
    backend_so: &Path,
    profile: CodegenProfilePolicy,
    configured_rustflags: &[String],
    explicit_rustflags: &[String],
    existing_encoded_rustflags: Option<&str>,
    existing_rustflags: Option<&str>,
) -> String {
    // Project flags are defaults, inherited flags are user overrides, and
    // explicit wrapper flags are stronger. cuda-oxide's compiler invariants
    // come last because rustc resolves repeated -C/-Z options last-one-wins.
    let mut flags = configured_rustflags.to_vec();

    if let Some(existing) = existing_encoded_rustflags {
        flags.extend(
            existing
                .split(ENCODED_RUSTFLAGS_SEPARATOR)
                .filter(|flag| !flag.is_empty())
                .map(str::to_string),
        );
    } else if let Some(existing) = existing_rustflags {
        // Match Cargo's legacy RUSTFLAGS behavior when converting it to the
        // encoded representation.
        flags.extend(existing.split_whitespace().map(str::to_string));
    }
    flags.extend(explicit_rustflags.iter().cloned());
    strip_wrapper_owned_codegen_cfgs(&mut flags);
    flags.push(format!("-Zcodegen-backend={}", backend_so.display()));
    if matches!(
        profile,
        CodegenProfilePolicy::ReleaseLike
            | CodegenProfilePolicy::ReleaseLikeWithDebugAssertions
            | CodegenProfilePolicy::ReleaseLikeWithDebugInfo
    ) {
        flags.push("-Copt-level=3".to_string());
        if profile == CodegenProfilePolicy::ReleaseLikeWithDebugAssertions {
            // rustc normally enables overflow checks together with debug
            // assertions. Keep them independent: overflow-check MIR changes
            // code shape enough to break pattern-sensitive device lowerings.
            flags.extend([
                "-Cdebug-assertions=on".to_string(),
                "-Coverflow-checks=off".to_string(),
            ]);
        } else {
            flags.push("-Cdebug-assertions=off".to_string());
        }
    }
    flags.extend([
        "-Zmir-enable-passes=-JumpThreading".to_string(),
        // Device codegen is whole-program: `collector` walks the call graph from
        // each `#[kernel]` and must emit every reachable dependency function into
        // one module. rustc encodes cross-crate MIR only for `#[inline]`/generic
        // items, so a non-`#[inline]`, non-generic dependency function that cannot
        // be inlined away (canonically: a recursive one) would be *called* but
        // never *defined* -> LLVM verification fails with "Symbol <crate>__<fn>
        // not found". Encode all MIR so any reachable dependency function is
        // device-compilable. This applies build-wide (like the other required
        // flags), so it also encodes MIR for host-only deps — an intentional,
        // interim trade (rmeta size) until a surgical device-dep-scoped or
        // per-crate device-link path lands. It matches the established approach
        // for whole-program-MIR tools (e.g. Miri).
        "-Zalways-encode-mir".to_string(),
        "-Csymbol-mangling-version=v0".to_string(),
    ]);
    if profile == CodegenProfilePolicy::ReleaseLikeWithDebugInfo {
        flags.push("-Cdebuginfo=2".to_string());
    }
    flags.join(&ENCODED_RUSTFLAGS_SEPARATOR.to_string())
}

fn strip_wrapper_owned_codegen_cfgs(flags: &mut Vec<String>) {
    fn is_wrapper_owned_cfg(value: &str) -> bool {
        [
            LEGACY_CODEGEN_FINGERPRINT_CFG,
            LEGACY_MATERIALIZER_PROVENANCE_CFG,
            FULL_DEBUG_GET_MUT_OUTLINE_CFG,
        ]
        .iter()
        .any(|name| {
            value
                .strip_prefix(name)
                .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with('='))
        })
    }

    let mut retained = Vec::with_capacity(flags.len());
    let mut index = 0;
    while index < flags.len() {
        let flag = &flags[index];
        if flag == "--cfg"
            && flags
                .get(index + 1)
                .is_some_and(|value| is_wrapper_owned_cfg(value))
        {
            index += 2;
            continue;
        }
        if flag
            .strip_prefix("--cfg=")
            .is_some_and(is_wrapper_owned_cfg)
        {
            index += 1;
            continue;
        }
        retained.push(flag.clone());
        index += 1;
    }
    *flags = retained;
}

/// Report the debug policy `CUDA_OXIDE_DEBUG` selects in this environment,
/// as one lowercase token on stdout.
///
/// Tooling that has to know whether a build will be a full-debug build --
/// `scripts/smoketest.sh` decides that way whether its optimized code-shape
/// gates apply at all -- would otherwise restate `parse_env_override`'s
/// alias, case and whitespace rules in another language. A second
/// implementation of a policy is a second answer waiting to disagree with the
/// first, and this one is easy to get wrong: `2` is full debug, so is `FULL`,
/// and so is a value padded with a non-breaking space.
pub fn print_debug_policy() {
    println!(
        "{}",
        debug_policy_token(std::env::var("CUDA_OXIDE_DEBUG").ok().as_deref())
    );
}

/// `None` means the variable is unset, which is not the same as a value the
/// parser does not recognize: the first leaves the caller's own default in
/// place, the second is a value someone wrote expecting it to mean something.
pub(super) fn debug_policy_token(value: Option<&str>) -> &'static str {
    let Some(value) = value else {
        return "unset";
    };
    match cuda_artifact_finalizer::DebugPolicy::parse_env_override(value) {
        Some(cuda_artifact_finalizer::DebugPolicy::None) => "none",
        Some(cuda_artifact_finalizer::DebugPolicy::LineTables) => "line-tables",
        Some(cuda_artifact_finalizer::DebugPolicy::Full) => "full",
        None => "unrecognized",
    }
}

fn command_requests_full_device_debug_with_env(
    cmd: &Command,
    inherited_debug: Option<&str>,
) -> bool {
    let effective_debug = match cmd
        .get_envs()
        .find(|(name, _)| *name == std::ffi::OsStr::new("CUDA_OXIDE_DEBUG"))
    {
        Some((_, Some(value))) => Some(value.to_string_lossy().into_owned()),
        Some((_, None)) => None,
        None => inherited_debug.map(str::to_owned),
    };

    // The backend and wrapper share this parser, so every full-debug alias
    // (including `2`) selects the same rustc controls.
    effective_debug.is_some_and(|value| {
        cuda_artifact_finalizer::DebugPolicy::parse_env_override(&value)
            == Some(cuda_artifact_finalizer::DebugPolicy::Full)
    })
}

/// Rustc controls used only by full device debug.
///
/// Full debug keeps imported, non-ZST MIR locals in stack slots by skipping
/// `mem2reg` and LLVM optimization. It cannot restore locals rustc has already
/// removed, so two MIR passes are excluded:
///
/// ```text
/// ScalarReplacementOfAggregates -> closure split -> captures disappear
/// SingleUseConsts                -> local folded  -> local disappears
/// ```
///
/// Other MIR passes stay enabled:
///
/// ```text
/// ReferencePropagation -> keep debug event -> write a debug-only stack home
/// MIR inlining         -> enabled, except `DisjointSlice::get_mut`
/// ```
///
/// We avoid `-Zmir-opt-level=0` because its extra MIR shapes do not all import.
/// These controls affect full-debug builds only.
pub(super) const FULL_DEBUG_MIR_RUSTFLAGS: &[&str] =
    &["-Zmir-enable-passes=-ScalarReplacementOfAggregates,-SingleUseConsts"];

pub(super) fn append_full_debug_rustflags(
    encoded: &mut String,
    cmd: &Command,
    inherited_debug: Option<&str>,
) {
    if !command_requests_full_device_debug_with_env(cmd, inherited_debug) {
        return;
    }
    for flag in ["--cfg", FULL_DEBUG_GET_MUT_OUTLINE_CFG]
        .into_iter()
        .chain(FULL_DEBUG_MIR_RUSTFLAGS.iter().copied())
    {
        if !encoded.is_empty() {
            encoded.push(ENCODED_RUSTFLAGS_SEPARATOR);
        }
        encoded.push_str(flag);
    }
}

fn apply_codegen_rustflags(
    cmd: &mut Command,
    ctx: &Context,
    profile: CodegenProfilePolicy,
    device_cfgs: &[String],
) {
    let mut encoded = build_encoded_rustflags(ctx, profile, device_cfgs);
    let inherited_debug = std::env::var("CUDA_OXIDE_DEBUG").ok();
    append_full_debug_rustflags(&mut encoded, cmd, inherited_debug.as_deref());

    cmd.env("CARGO_ENCODED_RUSTFLAGS", encoded)
        .env_remove("RUSTFLAGS");
}

/// Apply the two deliberately different Cargo cache boundaries:
///
/// - the exact backend binary is global because it compiles every crate;
/// - mode/architecture/tool settings are an env dependency recorded only by
///   CUDA macros in crates that can own or instantiate device code.
pub(super) fn apply_codegen_configuration(
    cmd: &mut Command,
    ctx: &Context,
    profile: CodegenProfilePolicy,
    user_device_cfgs: &[String],
    codegen_fingerprint: &str,
) -> Result<(), String> {
    let backend_digest = backend_artifact_digest(&ctx.backend_so)?;
    let mut global_cfgs = Vec::with_capacity(user_device_cfgs.len() + 1);
    global_cfgs.push(format!("{BACKEND_IDENTITY_CFG}=\"{backend_digest}\""));
    global_cfgs.extend(user_device_cfgs.iter().cloned());

    apply_codegen_rustflags(cmd, ctx, profile, &global_cfgs);
    cmd.env(CODEGEN_FINGERPRINT_ENV, codegen_fingerprint);
    Ok(())
}

pub(super) fn apply_codegen_configuration_or_exit(
    cmd: &mut Command,
    ctx: &Context,
    profile: CodegenProfilePolicy,
    user_device_cfgs: &[String],
    codegen_fingerprint: &str,
) {
    apply_codegen_configuration(cmd, ctx, profile, user_device_cfgs, codegen_fingerprint)
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });
}

/// Set environment variables for the codegen backend.
///
/// `arch` is an explicit pin (`--arch`); it becomes `CUDA_OXIDE_TARGET`, the
/// hard override the backend honors as-is. The auto-detected GPU arch is *not*
/// routed here -- see [`apply_device_arch_hint`].
pub(super) fn apply_output_mode(
    cmd: &mut Command,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    materialization: &MaterializationMode,
) {
    if let Some(target_arch) = arch {
        cmd.env("CUDA_OXIDE_TARGET", target_arch);
    }
    if emit_nvvm_ir || materialization.enabled() {
        cmd.env("CUDA_OXIDE_EMIT_NVVM_IR", "1");
    }
    materialization.apply(cmd);
}

pub(super) fn configured_arch<'a>(ctx: &'a Context, cli_arch: Option<&'a str>) -> Option<&'a str> {
    if cli_arch.is_some() || std::env::var("CUDA_OXIDE_TARGET").is_ok() {
        cli_arch
    } else {
        ctx.config
            .default_arch
            .as_deref()
            .or_else(|| project_config_env(ctx, "CUDA_OXIDE_TARGET"))
    }
}

pub(super) fn configured_arch_label(ctx: &Context, cli_arch: Option<&str>) -> Option<String> {
    cli_arch
        .map(str::to_string)
        .or_else(|| std::env::var("CUDA_OXIDE_TARGET").ok())
        .or_else(|| ctx.config.default_arch.clone())
        .or_else(|| project_config_env(ctx, "CUDA_OXIDE_TARGET").map(str::to_string))
}

pub fn has_configured_arch(ctx: &Context, cli_arch: Option<&str>) -> bool {
    cli_arch.is_some()
        || std::env::var("CUDA_OXIDE_TARGET").is_ok()
        || ctx.config.default_arch.is_some()
        || project_config_env(ctx, "CUDA_OXIDE_TARGET").is_some()
}

pub(super) fn apply_config_env(cmd: &mut Command, ctx: &Context) {
    for (key, value) in &ctx.config.env {
        if matches!(key.as_str(), "RUSTFLAGS" | "CARGO_ENCODED_RUSTFLAGS") {
            continue;
        }
        // Project values are defaults. An explicitly inherited environment is
        // stronger, and command-specific CLI/internal settings are applied
        // after this helper and are stronger still.
        if std::env::var_os(key).is_none() {
            cmd.env(key, value);
        }
    }
    apply_managed_cutlass_compiler(cmd, ctx);
}

fn apply_managed_cutlass_compiler(cmd: &mut Command, ctx: &Context) {
    let backend = std::env::var_os(DEVICE_BACKEND_ENV)
        .map(|value| value.as_encoded_bytes().to_vec())
        .or_else(|| {
            project_config_env(ctx, DEVICE_BACKEND_ENV).map(|value| value.as_bytes().to_vec())
        });
    if !backend.as_deref().is_some_and(selector_is_cutlass_mlir) {
        return;
    }

    let explicitly_configured = std::env::var_os(cutlass_toolchain::CUTLASS_COMPILER_ENV).is_some()
        || project_config_env(ctx, cutlass_toolchain::CUTLASS_COMPILER_ENV).is_some();
    if explicitly_configured {
        return;
    }

    match cutlass_toolchain::resolve_official() {
        Ok(Some(paths)) => apply_resolved_cutlass_compiler(cmd, Some(&paths), false),
        Ok(None) => {
            eprintln!(
                "Error: CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir needs the managed CUTLASS compiler"
            );
            eprintln!("Install the pinned official release with:");
            eprintln!("  cargo oxide toolchain install cutlass");
            eprintln!(
                "Or explicitly set {}.",
                cutlass_toolchain::CUTLASS_COMPILER_ENV
            );
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("Error: managed CUTLASS compiler installation is invalid: {error}");
            eprintln!("Repair it with `cargo oxide toolchain install cutlass`.");
            std::process::exit(2);
        }
    }
}

pub(super) fn apply_resolved_cutlass_compiler(
    cmd: &mut Command,
    resolved: Option<&cutlass_toolchain::ResolvedCutlass>,
    explicitly_configured: bool,
) {
    if explicitly_configured {
        return;
    }
    if let Some(paths) = resolved {
        cmd.env(
            cutlass_toolchain::CUTLASS_COMPILER_ENV,
            &paths.compiler_library,
        );
    }
}

pub(super) fn apply_common_codegen_env(
    cmd: &mut Command,
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
) {
    apply_config_env(cmd, ctx);
    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }
    if no_fmad {
        cmd.env("CUDA_OXIDE_NO_FMA", "1");
    }
    if unchecked_indexing {
        cmd.env("CUDA_OXIDE_UNCHECKED_INDEXING", "1");
    }
    // An explicit flag outranks an ambient `CUDA_OXIDE_DEBUG`, matching how
    // `--no-fmad` outranks `CUDA_OXIDE_NO_FMA`. `DeviceDebug::Off` exports
    // nothing rather than `off`, so omitting the flag cannot silently cancel a
    // debug level the environment or project config already asked for.
    if let Some(level) = device_debug.env_value() {
        cmd.env("CUDA_OXIDE_DEBUG", level);
    }
    apply_ld_library_path(cmd, ctx);
}

/// Give Compute Sanitizer source line attribution without disabling normal
/// device optimization. An explicit process or project setting remains
/// authoritative, including an intentional `CUDA_OXIDE_DEBUG=off`. So does an
/// explicit `--lineinfo` / `--device-debug` flag: `apply_common_codegen_env`
/// has already exported its level onto `cmd`, and the default must not
/// overwrite it.
pub(super) fn apply_default_sanitizer_line_tables(
    cmd: &mut Command,
    ctx: &Context,
    device_debug: DeviceDebug,
) {
    apply_default_sanitizer_line_tables_with_env(
        cmd,
        ctx,
        std::env::var_os("CUDA_OXIDE_DEBUG").is_some(),
        device_debug,
    );
}

/// `apply_default_sanitizer_line_tables` with the `CUDA_OXIDE_DEBUG` probe
/// injected.
///
/// `env_debug_set` is presence-only, matching the `var_os` check it replaces.
/// Injected so a unit test can assert the defaulting without an exported
/// `CUDA_OXIDE_DEBUG` suppressing it. `device_debug` carries the CLI flag:
/// any level other than [`DeviceDebug::Off`] is an explicit request that
/// outranks the line-tables default.
pub(super) fn apply_default_sanitizer_line_tables_with_env(
    cmd: &mut Command,
    ctx: &Context,
    env_debug_set: bool,
    device_debug: DeviceDebug,
) {
    if device_debug == DeviceDebug::Off
        && !env_debug_set
        && project_config_env(ctx, "CUDA_OXIDE_DEBUG").is_none()
    {
        cmd.env("CUDA_OXIDE_DEBUG", "line-tables");
    }
}

pub(super) fn apply_interop_device_codegen_options(
    cmd: &mut Command,
    ctx: &Context,
    verbose: bool,
    options: InteropDeviceBuildOptions,
) {
    let inherited_debug = std::env::var_os("CUDA_OXIDE_DEBUG");
    apply_interop_device_codegen_options_with_env(
        cmd,
        ctx,
        verbose,
        options,
        inherited_debug.as_deref(),
    );
}

/// `apply_interop_device_codegen_options` with the inherited
/// `CUDA_OXIDE_DEBUG` value injected.
///
/// Interop device crates are separate Cargo builds, so resolve their effective
/// debug policy here in the same order as the regular path: an explicit CLI
/// level, then the inherited environment, then project configuration. Compute
/// Sanitizer's line-table policy is only a default after all three.
pub(super) fn apply_interop_device_codegen_options_with_env(
    cmd: &mut Command,
    ctx: &Context,
    verbose: bool,
    options: InteropDeviceBuildOptions,
    inherited_debug: Option<&std::ffi::OsStr>,
) {
    apply_common_codegen_env(
        cmd,
        ctx,
        verbose,
        options.no_fmad,
        options.unchecked_indexing,
        DeviceDebug::Off,
    );

    let effective_debug = options
        .device_debug
        .env_value()
        .map(std::ffi::OsString::from)
        .or_else(|| inherited_debug.map(std::ffi::OsStr::to_os_string))
        .or_else(|| project_config_env(ctx, "CUDA_OXIDE_DEBUG").map(std::ffi::OsString::from))
        .or_else(|| {
            options
                .sanitizer_line_tables
                .then(|| std::ffi::OsString::from("line-tables"))
        });
    if let Some(debug) = effective_debug {
        cmd.env("CUDA_OXIDE_DEBUG", debug);
    }
}

/// Forward the auto-detected GPU arch as a *hint* via `CUDA_OXIDE_DEVICE_ARCH`.
///
/// Unlike `CUDA_OXIDE_TARGET` (a hard override), this is advisory: the backend
/// builds for the detected GPU only when that GPU can actually run the kernel.
/// If the kernel needs a newer arch (e.g. tcgen05 / cta_group TMA multicast
/// need sm_100a, which a consumer sm_120 GPU lacks), the backend builds for the
/// required arch instead. Skipped when the user pinned `--arch` (that explicit
/// choice already went to `CUDA_OXIDE_TARGET`).
pub(super) fn apply_device_arch_hint(
    cmd: &mut Command,
    explicit_arch: Option<&str>,
    detected_device_arch: Option<&str>,
) {
    if let (None, Some(dev)) = (explicit_arch, detected_device_arch) {
        cmd.env("CUDA_OXIDE_DEVICE_ARCH", dev);
    }
}

/// Pick a runnable target for `cargo oxide run` when the user has not pinned
/// one explicitly.
///
/// # Precedence
///
/// `cargo oxide run` resolves the target architecture in this order, highest
/// priority first:
///
/// 1. `--arch <sm_XX>`            (explicit user override)
/// 2. `CUDA_OXIDE_TARGET=<sm_XX>` (explicit env override, set in the parent
///    process before invoking `cargo oxide run`)
/// 3. **This function**: the compute capability of the first GPU reported by
///    `nvidia-smi`, forwarded as the `CUDA_OXIDE_DEVICE_ARCH` *hint*. Emits
///    the arch-specific `sm_XYa` form for cc >= 9.0 (so the backend can lower
///    WGMMA / tcgen05 / TMA-multicast when the GPU supports them) and the
///    plain `sm_XY` form for cc < 9.0.
/// 4. Backend feature-based default (`select_target` in
///    `mir-importer::pipeline`), which picks the minimum `sm_XX` required by
///    the IR shape (e.g. `Basic -> sm_80`, `Cluster -> sm_90`, `Tma -> sm_100`).
///
/// Slot 3 is advisory: the backend builds for the detected GPU only when that
/// GPU can run the kernel, otherwise it falls back to slot 4 (the arch the
/// kernel requires). This function returns `Some(sm_XY[a])` to fill slot 3, or
/// `None` (falling through to slot 4) when the machine has no usable GPU.
///
/// # Why only `run`
///
/// `run` immediately loads the generated module on the local GPU and launches
/// the kernel, so a target older than the local GPU's compute capability is
/// the only safe default. `build` and `pipeline` may legitimately
/// cross-compile to a different machine, so they keep the backend's
/// feature-based default untouched.
///
/// # Why this is needed even with the backend default
///
/// The backend's `select_target` picks the minimum `sm_XX` the IR requires.
/// `Basic → sm_80` is a fine *compilation* baseline, but PTX for `sm_80` will
/// not load on a Turing (`sm_75`) GPU because the JIT refuses
/// forward-incompatible PTX. Detecting the device CC in `run` keeps the
/// generated module loadable on the actual hardware that will execute it.
///
/// # When this returns `None`
///
/// - The user passed `--arch` (slot 1 wins).
/// - `CUDA_OXIDE_TARGET` is set in the environment (slot 2 wins).
/// - `--emit-nvvm-ir` is in effect (NVVM IR mode requires explicit `--arch`,
///   enforced by the CLI parser).
/// - No CUDA driver / GPU is available on the machine (CI runners without
///   GPUs, headless build boxes), or `nvidia-smi` is missing or broken. The
///   caller falls through to slot 4 and the backend's feature-based default
///   applies.
pub(super) fn detect_run_target_arch(arch: Option<&str>, emit_nvvm_ir: bool) -> Option<String> {
    detect_run_target_arch_with_env(
        arch,
        emit_nvvm_ir,
        std::env::var_os("CUDA_OXIDE_TARGET").is_some(),
    )
}

/// `detect_run_target_arch` with the `CUDA_OXIDE_TARGET` probe injected.
///
/// `env_target_set` is presence-only, matching the `var_os` check it replaces.
/// Injected so a unit test can exercise the slot-2 skip without exporting the
/// variable: `set_var` would be a data race against the `vars_os` reads the
/// fingerprint helpers perform on other test threads.
pub(super) fn detect_run_target_arch_with_env(
    arch: Option<&str>,
    emit_nvvm_ir: bool,
    env_target_set: bool,
) -> Option<String> {
    if arch.is_some() || emit_nvvm_ir || env_target_set {
        return None;
    }

    detect_local_device_arch()
}

pub(super) fn detect_local_device_arch() -> Option<String> {
    query_device_compute_cap().map(format_sm_arch)
}

/// Query the compute capability of the first GPU via `nvidia-smi`.
///
/// Runs `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and parses
/// the first output line. A subprocess probe (rather than the CUDA driver
/// API) keeps cargo-oxide free of any link-time or dlopen dependency on
/// `libcuda`, so the subcommand builds and runs on machines with no CUDA
/// toolkit and no driver; `scripts/smoketest.sh` derives `sm_XX` from
/// `nvidia-smi` the same way.
///
/// Caveat: `nvidia-smi` enumerates GPUs in PCI bus order, while CUDA's
/// default device order is fastest-first, so on heterogeneous multi-GPU
/// machines this may describe a different GPU than CUDA device 0. That is
/// safe because `CUDA_OXIDE_DEVICE_ARCH` is advisory (the backend only
/// honors a compatible hint) and `--arch` / `CUDA_OXIDE_TARGET` remain hard
/// overrides.
fn query_device_compute_cap() -> Option<(u32, u32)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    parse_compute_cap(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the first line of `nvidia-smi --query-gpu=compute_cap` output as a
/// `(major, minor)` compute-capability pair. Returns `None` for anything
/// that is not shaped `<digits>.<digits>`.
pub(super) fn parse_compute_cap(stdout: &str) -> Option<(u32, u32)> {
    parse_compute_cap_field(stdout.lines().next()?)
}

/// Parse a single `compute_cap` CSV field (e.g. `"12.0"`).
///
/// Only the `<digits>.<digits>` shape is accepted: `nvidia-smi` prints its
/// failure banners ("NVIDIA-SMI has failed ...") to *stdout*, sometimes with
/// exit status 0, so this shape check is the real gate, not the exit status.
fn parse_compute_cap_field(field: &str) -> Option<(u32, u32)> {
    let (major, minor) = field.trim().split_once('.')?;
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(major) || !all_digits(minor) {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Query the name, compute capability, and driver version of the first GPU
/// via `nvidia-smi`, for doctor's driver / GPU report. Same trust rules as
/// [`query_device_compute_cap`]. The driver version matters for triage:
/// PTX-JIT and driver-API compatibility bugs are driver-version-specific,
/// and the bug-report template points reporters at this line.
pub(super) fn query_gpu_name_cap_and_driver() -> Option<(String, (u32, u32), String)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    parse_gpu_name_cap_and_driver(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the first line of `nvidia-smi
/// --query-gpu=name,compute_cap,driver_version` output into the GPU name,
/// `(major, minor)` pair, and driver version. Splits on the LAST two commas:
/// GPU names may contain commas in principle, `compute_cap` and
/// `driver_version` never do.
pub(super) fn parse_gpu_name_cap_and_driver(stdout: &str) -> Option<(String, (u32, u32), String)> {
    let line = stdout.lines().next()?;
    let (rest, driver) = line.rsplit_once(',')?;
    let (name, cap) = rest.rsplit_once(',')?;
    Some((
        name.trim().to_string(),
        parse_compute_cap_field(cap)?,
        driver.trim().to_string(),
    ))
}

/// Format a `(major, minor)` compute-capability tuple as the `sm_XX` /
/// `sm_XXX[a]` string the codegen backend expects on `CUDA_OXIDE_TARGET`.
///
/// Concatenates without a separator, matching CUDA conventions:
/// `(7, 5)` → `"sm_75"`, `(12, 0)` → `"sm_120a"`.
///
/// # Arch-specific (`a`) suffix
///
/// Compute capability ≥ 9.0 always has an arch-specific PTX target (`sm_90a`,
/// `sm_100a`, `sm_103a`, `sm_120a`, …) that is a strict superset of the plain
/// target on that chip. The `a` form is what unlocks WGMMA on Hopper and
/// `tcgen05` / TMA multicast / `cta_group::*` on Blackwell datacenter — and
/// every chip that reports cc ≥ 9.0 *is* the `a`-variant chip in NVIDIA's
/// lineup (there is no consumer Hopper, no non-`a` sm_100, and so on).
///
/// For GPU auto-detection, this formats the compute capability of the first GPU
/// reported by `nvidia-smi`. `cargo oxide run` uses the result as an advisory
/// target hint, while `cargo oxide build` may display it for diagnostics without
/// changing the build target. Emitting the `a` form for a detected GPU:
///
/// - **No false negatives:** kernels that need `tcgen05` / WGMMA compile and
///   load on that GPU (was: silent fallback to `sm_100` / `sm_90` and a
///   `ptxas: 'tcgen05.alloc' not supported on .target 'sm_100'` failure).
/// - **No false positives:** cc < 9.0 keeps the plain `sm_XY` form, since
///   there is no `sm_80a` / `sm_86a` / `sm_89a` target in the PTX ISA.
/// - **Strict superset:** PTX targeting `sm_XYa` accepts every kernel that
///   would have compiled for plain `sm_XY`; the `a` form only permits
///   *additional* arch-specific intrinsics.
pub(super) fn format_sm_arch((major, minor): (u32, u32)) -> String {
    if major >= 9 {
        format!("sm_{}{}a", major, minor)
    } else {
        format!("sm_{}{}", major, minor)
    }
}

fn inherited_or_configured_env(ctx: &Context, key: &str) -> Option<String> {
    std::env::var(key).ok().or_else(|| {
        ctx.config
            .env
            .iter()
            .find(|(configured_key, _)| configured_key == key)
            .map(|(_, value)| value.clone())
    })
}

/// Build `LD_LIBRARY_PATH` for the child cargo process.
///
/// Includes the rustc sysroot lib (for `librustc_driver.so` etc.), the
/// libmathdx lib (when `LIBMATHDX_PATH` is set), and any existing
/// `LD_LIBRARY_PATH` from the parent environment.
pub(super) fn apply_ld_library_path(cmd: &mut Command, ctx: &Context) {
    let mut ld_paths: Vec<String> = Vec::new();
    if let Some(sysroot) = backend::get_rustc_sysroot() {
        ld_paths.push(format!("{}/lib", sysroot));
    }
    if let Some(libmathdx_path) = inherited_or_configured_env(ctx, "LIBMATHDX_PATH") {
        ld_paths.push(format!("{}/lib", libmathdx_path));
    }
    if let Some(existing) = inherited_or_configured_env(ctx, "LD_LIBRARY_PATH") {
        ld_paths.push(existing);
    }
    if !ld_paths.is_empty() {
        cmd.env("LD_LIBRARY_PATH", ld_paths.join(":"));
    }
}
