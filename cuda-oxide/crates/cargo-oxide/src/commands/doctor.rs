/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::backend;
use crate::backend_source::{self, DependencySource, short_rev};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

// =============================================================================
// Doctor command
// =============================================================================

/// Find the first compiler that can report a non-empty resource directory.
/// Probe the required operation directly: `--version` alone does not establish
/// that a wrapper or incomplete installation can answer this query.
pub(super) fn clang_resource_dir<'a>(candidates: &[&'a str]) -> Option<(&'a str, String)> {
    candidates.iter().find_map(|&name| {
        let output = Command::new(name)
            .arg("-print-resource-dir")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (!dir.is_empty()).then_some((name, dir))
    })
}

/// Parsed contents of a `rust-toolchain.toml` pin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RustToolchainPin {
    pub(crate) channel: String,
    pub(crate) components: Vec<String>,
}

/// Components that doctor treats as hard requirements for the cuda-oxide
/// pipeline even if `rust-toolchain.toml` stops listing them: `rust-src`
/// (device-side core sources), `rustc-dev` (rustc_private, required to build
/// the codegen backend), and `llvm-tools`.
const DOCTOR_REQUIRED_COMPONENTS: &[&str] = &["rust-src", "rustc-dev", "llvm-tools"];

/// The components doctor verifies for a pin: everything the pin itself lists,
/// plus the [`DOCTOR_REQUIRED_COMPONENTS`] floor.
///
/// rustup auto-installs every component named in `rust-toolchain.toml` when it
/// installs the pinned toolchain, so a pinned component that is absent from
/// `rustup component list --installed` means a broken or manually trimmed
/// install and is worth failing doctor over. The floor guards against a future
/// edit of the pin file dropping a component the pipeline genuinely needs.
pub(super) fn doctor_verified_components(pin: &RustToolchainPin) -> Vec<String> {
    let mut required: Vec<String> = pin.components.clone();
    for component in DOCTOR_REQUIRED_COMPONENTS {
        if !required.iter().any(|existing| existing == component) {
            required.push((*component).to_string());
        }
    }
    required
}

/// Parse a `rust-toolchain.toml` document for channel and components.
pub(crate) fn parse_rust_toolchain_toml(contents: &str) -> Result<RustToolchainPin, String> {
    let value: toml::Value =
        toml::from_str(contents).map_err(|error| format!("invalid TOML: {error}"))?;
    let toolchain = value
        .get("toolchain")
        .ok_or_else(|| "missing [toolchain] table".to_string())?;
    let channel = toolchain
        .get("channel")
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|channel| !channel.is_empty())
        .ok_or_else(|| "missing toolchain.channel".to_string())?
        .to_string();
    let components = match toolchain.get("components") {
        None => Vec::new(),
        Some(toml::Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(|name| name.trim().to_string())
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        "toolchain.components entries must be non-empty strings".to_string()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err("toolchain.components must be an array of strings".to_string());
        }
    };
    Ok(RustToolchainPin {
        channel,
        components,
    })
}

/// True when `rustup show active-toolchain` output matches the pinned channel.
///
/// The toolchain name is the first whitespace-delimited token of the first
/// line in every rustup output format seen so far:
///
/// - pre-1.28 and 1.29+: `nightly-2026-08-28-<triple> (default)` or
///   `nightly-2026-08-28-<triple> (overridden by '<path>')` on one line
///   (verified against rustup 1.29.0);
/// - 1.28.x: the bare name on the first line with the reason on a second
///   `active because: ...` line.
pub(crate) fn active_toolchain_matches_channel(active_toolchain: &str, channel: &str) -> bool {
    let active = active_toolchain
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .next()
        .unwrap_or("");
    if active.is_empty() || channel.is_empty() {
        return false;
    }
    active == channel || active.starts_with(&format!("{channel}-"))
}

/// Return required components that are absent from `rustup component list --installed`.
pub(super) fn missing_rustup_components<S: AsRef<str>>(
    installed_list: &str,
    required: &[S],
) -> Vec<String> {
    required
        .iter()
        .map(AsRef::as_ref)
        .filter(|component| !rustup_component_installed(installed_list, component))
        .map(str::to_string)
        .collect()
}

fn rustup_component_installed(installed_list: &str, component: &str) -> bool {
    installed_list.lines().any(|line| {
        let name = line.split_whitespace().next().unwrap_or("");
        name == component || name.starts_with(&format!("{component}-"))
    })
}

fn doctor_report_toolchain_pin(ctx: &Context, ok: &mut bool) {
    let toolchain_file = ctx.workspace_root.join("rust-toolchain.toml");
    print!("rust-toolchain.toml... ");
    if !toolchain_file.exists() {
        println!("✗ not found at {}", toolchain_file.display());
        *ok = false;
        return;
    }

    let contents = match std::fs::read_to_string(&toolchain_file) {
        Ok(contents) => contents,
        Err(error) => {
            println!("✗ present but unreadable ({error})");
            *ok = false;
            return;
        }
    };

    let pin = match parse_rust_toolchain_toml(&contents) {
        Ok(pin) => pin,
        Err(error) => {
            println!("✗ present but invalid ({error})");
            *ok = false;
            return;
        }
    };
    println!("✓ channel {}", pin.channel);

    print!("Pinned toolchain active... ");
    match Command::new("rustup")
        .args(["show", "active-toolchain"])
        .output()
    {
        Ok(output) if output.status.success() => {
            let active = String::from_utf8_lossy(&output.stdout);
            let active = active.trim();
            if active_toolchain_matches_channel(active, &pin.channel) {
                println!("✓ {active}");
            } else {
                println!(
                    "✗ active `{active}`, expected `{pin_channel}`",
                    pin_channel = pin.channel
                );
                eprintln!(
                    "  Install/select the pin with `rustup toolchain install {}` and reopen the shell",
                    pin.channel
                );
                eprintln!("  in this workspace so rust-toolchain.toml can select it.");
                *ok = false;
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("✗ rustup show active-toolchain failed");
            if !stderr.trim().is_empty() {
                eprintln!("  {}", stderr.trim());
            }
            *ok = false;
        }
        Err(_) => {
            println!("✗ rustup not found");
            eprintln!("  Install rustup from https://rustup.rs/ so doctor can verify the pin.");
            *ok = false;
        }
    }

    let required = doctor_verified_components(&pin);

    print!("Required rustup components... ");
    match Command::new("rustup")
        .args([
            "component",
            "list",
            "--installed",
            "--toolchain",
            &pin.channel,
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            let installed = String::from_utf8_lossy(&output.stdout);
            let missing = missing_rustup_components(&installed, &required);
            if missing.is_empty() {
                println!("✓ {}", required.join(", "));
            } else {
                println!("✗ missing {}", missing.join(", "));
                eprintln!(
                    "  Install with `rustup component add --toolchain {} {}`",
                    pin.channel,
                    missing.join(" ")
                );
                *ok = false;
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("✗ could not list components for {}", pin.channel);
            if !stderr.trim().is_empty() {
                eprintln!("  {}", stderr.trim());
            }
            eprintln!(
                "  Try `rustup toolchain install {channel} -c {components}`",
                channel = pin.channel,
                components = required.join(" -c ")
            );
            *ok = false;
        }
        Err(_) => {
            println!("✗ rustup not found");
            *ok = false;
        }
    }
}

/// Reports whether the cached backend came from the commit this project's
/// cuda-oxide dependency resolves to.
///
/// Resolves offline: doctor must not fetch, and the dependency is already
/// checked out once the project has resolved its lockfile. Informational,
/// never fatal: a mismatch heals itself on the next build.
fn doctor_report_backend_source(ctx: &Context) {
    print!("Backend source (cuda-oxide commit)... ");
    // Both pins sit above the dependency in backend discovery, so the
    // dependency's commit plays no part while either is set.
    if std::env::var_os("CUDA_OXIDE_BACKEND").is_some() || ctx.config.backend.is_some() {
        println!(
            "- backend pinned by CUDA_OXIDE_BACKEND or `.cargo/cuda-oxide.toml`; the \
             dependency's commit is not consulted"
        );
        return;
    }
    // Read-only resolution needs a lockfile to read; a fresh project has none
    // until its first build, which is the normal state, not a fault.
    if !ctx.workspace_root.join("Cargo.lock").is_file() {
        println!(
            "- no Cargo.lock yet; run `cargo oxide build` once, then doctor can compare the \
             cache with the dependency"
        );
        return;
    }
    let check = backend_source_check(
        backend_source::resolve_dependency_source(&ctx.workspace_root, true),
        backend::cached_backend_source_rev(),
    );
    println!("{}", check.headline);
    for line in check.details {
        println!("  {line}");
    }
}

/// The backend-source verdict, from the resolved dependency and the commit
/// recorded in the shared cache. Reuses the config check's struct so doctor
/// prints it the same way; pure, so every branch is testable.
pub(super) fn backend_source_check(
    source: Result<Option<DependencySource>, String>,
    recorded: Option<String>,
) -> OxideConfigCheck {
    let check = |headline: String, details: Vec<String>| OxideConfigCheck {
        headline,
        details,
        failed: false,
    };
    let source = match source {
        Ok(source) => source,
        Err(error) => {
            return check(
                format!("- could not resolve the cuda-oxide dependency offline ({error})"),
                Vec::new(),
            );
        }
    };
    let Some(source) = source else {
        return check(
            "- no cuda-device/cuda-host dependency; the cache follows cuda-oxide main".to_string(),
            Vec::new(),
        );
    };
    let Some(expected) = source.rev() else {
        return check(
            format!("✓ {} builds in place", source.describe()),
            Vec::new(),
        );
    };
    match recorded {
        Some(recorded) if recorded == expected => check(
            format!(
                "✓ cache built from {}, matching this project's dependency",
                short_rev(expected)
            ),
            Vec::new(),
        ),
        Some(recorded) => check(
            format!(
                "⚠ cache built from {} but this project depends on {}",
                short_rev(&recorded),
                short_rev(expected)
            ),
            vec![
                "The next `cargo oxide build` or `run` rebuilds the cache from the dependency."
                    .to_string(),
            ],
        ),
        None => check(
            format!(
                "⚠ cache records no commit (built by an older cargo-oxide); this project depends on {}",
                short_rev(expected)
            ),
            vec![
                "The next `cargo oxide build` or `run` rebuilds the cache and records it."
                    .to_string(),
            ],
        ),
    }
}

/// Validate the development environment.
///
/// Checks for: Rust nightly toolchain, `rust-toolchain.toml`, the codegen
/// backend `.so` (informational), CUDA headers (`cuda.h`), CUDA toolkit
/// (`nvcc`, libNVVM, nvJitLink, libdevice), LLVM (`llc`), clang/libclang,
/// the NVIDIA driver / GPU (informational), and optionally `cuda-gdb` /
/// `compute-sanitizer`.
/// Exits non-zero if any required check fails.
///
/// Doctor itself needs neither the CUDA toolkit nor a driver: every check
/// is a subprocess, a filesystem probe, or a runtime `dlopen`, and the
/// caller resolves the context via [`resolve_passive_context`] so nothing is
/// built first. This is what lets it diagnose a bare machine (issue #87).
pub fn doctor(ctx: &Context) {
    println!("cargo-oxide environment check");
    println!("==============================");
    println!();

    let mut ok = true;

    // 1. Rust toolchain
    print!("Rust nightly toolchain... ");
    match Command::new("rustc").args(["--version"]).output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            let version = version.trim();
            if version.contains("nightly") {
                println!("✓ {}", version);
            } else {
                println!("✗ expected nightly, got: {}", version);
                ok = false;
            }
        }
        _ => {
            println!("✗ rustc not found");
            ok = false;
        }
    }

    // 2. rust-toolchain.toml pin + active channel + required components
    doctor_report_toolchain_pin(ctx, &mut ok);

    // 3. Backend .so. Informational, not fatal: `run`/`build`/`pipeline`
    // build the backend on demand, so "not built yet" is a healthy state
    // for a fresh clone.
    print!("Codegen backend... ");
    if ctx.backend_so.exists() {
        println!("✓ {}", ctx.backend_so.display());
    } else {
        println!("- not built yet (run `cargo oxide setup`)");
    }

    // 3a. Project config (`.cargo/cuda-oxide.toml`)
    doctor_report_oxide_config(ctx, &mut ok);

    // 3b. Shared cache. The check above reports the backend this context
    // resolves to, which inside the repository is the local build. A project
    // outside the repository resolves to the cache instead, so the two can
    // disagree while every other check passes.
    print!("Shared cache (external projects)... ");
    match backend::cached_backend_path() {
        Some(cached) => match backend::compare_cache_to_local(&cached, &ctx.backend_so) {
            backend::CacheReport::Absent => {
                println!("- empty; external projects build on first use");
            }
            backend::CacheReport::UpToDate => {
                println!("✓ {}", cached.display());
            }
            backend::CacheReport::OlderThanLocal => {
                println!("⚠ {}", cached.display());
                println!("  Older than the backend built here, so projects outside this");
                println!("  repository would load a different one. Run `cargo oxide setup`");
                println!("  to publish, or set CUDA_OXIDE_BACKEND to pin an explicit path.");
            }
        },
        None => println!("- cache directory unknown (set CARGO_HOME or HOME)"),
    }

    // 3c. Backend source. Outside the repository the backend is built from
    // the commit Cargo resolved for the project's cuda-oxide dependency, and
    // the cache records which commit it came from. Informational: a mismatch
    // heals itself on the next build.
    if !ctx.is_workspace {
        doctor_report_backend_source(ctx);
    }

    // 4. CUDA headers (cuda.h). The host `cuda-bindings` crate cannot build
    // without them; cargo-oxide itself deliberately can, which is what makes
    // this check reachable on a toolkit-less machine instead of dying inside
    // cuda-bindings' build script (issue #87).
    print!("CUDA headers (cuda.h)... ");
    let target_dir_override = std::env::var("CUDA_TOOLKIT_TARGET_DIR").ok();
    let toolkit_choice = resolve_cuda_toolkit(
        |key| std::env::var_os(key),
        |root| validate_cuda_toolkit(root, target_dir_override.as_deref()),
    );
    let header_candidates = cuda_header_candidates(
        &toolkit_choice.root,
        target_dir_override.as_deref(),
        std::env::consts::ARCH,
        std::env::consts::OS,
    );
    match (
        &toolkit_choice.source,
        header_candidates.iter().find(|path| path.is_file()),
    ) {
        (ToolkitSource::Explicit(var), Some(found)) => {
            println!("✓ {} (via {var})", found.display());
        }
        (ToolkitSource::Discovered, Some(found)) => {
            println!(
                "✓ {} (discovered; neither CUDA_TOOLKIT_PATH nor CUDA_HOME is set)",
                found.display()
            );
        }
        _ => {
            println!(
                "✗ no usable CUDA toolkit; nearest root probed was `{}`",
                toolkit_choice.root.display()
            );
            eprintln!("  Probed for cuda.h:");
            for candidate in &header_candidates {
                eprintln!("    {}", candidate.display());
            }
            // The rejections are the point when more than one toolkit is
            // installed: a root can be turned down for a missing header or
            // for its version, and without saying which, the later failure
            // inside cuda-bindings names a header and not the choice.
            if !toolkit_choice.rejected.is_empty() {
                eprintln!("  Turned down, in order:");
                for reason in &toolkit_choice.rejected {
                    eprintln!("    {reason}");
                }
            }
            eprintln!("  Host crates (cuda-bindings) cannot build without cuda.h. Set");
            eprintln!("  CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA 13.0+ toolkit install");
            eprintln!("  root; when neither is set, these roots are tried in order:");
            for candidate in DEFAULT_TOOLKIT_CANDIDATES {
                eprintln!("    {candidate}");
            }
            ok = false;
        }
    }

    // 5. CUDA toolkit -- same discovery order as `cuda.h`, libNVVM, libdevice
    // and `compute-sanitizer` below. `nvcc` lives in the toolkit's `bin/`, so
    // probing PATH alone reported the toolkit missing on exactly the install
    // this project documents: CUDA_HOME set, toolkit not on PATH.
    print!("CUDA toolkit (nvcc)... ");
    match doctor_toolkit_tool(ctx, "nvcc") {
        Some(path) => match Command::new(&path).arg("--version").output() {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = version.lines().find(|l| l.contains("release")) {
                    println!("✓ {} ({})", line.trim(), path.display());
                } else {
                    println!("✓ {}", path.display());
                }
            }
            _ => println!("✓ {}", path.display()),
        },
        None => {
            let toolkit = cuda_toolkit_root(|key| {
                std::env::var(key)
                    .ok()
                    .filter(|value| !value.trim().is_empty())
                    .or_else(|| project_config_env(ctx, key).map(str::to_owned))
            });
            println!("✗ nvcc not found");
            eprintln!(
                "  Probed PATH, {}/bin/nvcc, and the standard install roots.",
                toolkit.display()
            );
            eprintln!("  Set CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA 13.0+ toolkit install");
            eprintln!("  root; when neither is set, these roots are tried in order:");
            for candidate in DEFAULT_TOOLKIT_CANDIDATES {
                eprintln!("    {candidate}");
            }
            ok = false;
        }
    }

    // 5b. libNVVM + nvJitLink + libdevice (only required when a kernel uses
    // CUDA libdevice math, e.g. sin/cos/exp/pow). All three ship with the
    // CUDA Toolkit; checking them here surfaces missing or split packagings
    // before a runtime failure inside `cuda_host::ltoir::load_kernel_module`.
    print!("libNVVM (libnvvm.so)... ");
    match libnvvm_sys::LibNvvm::load() {
        Ok(nvvm) => match nvvm.version() {
            Ok((major, minor)) => println!("✓ libNVVM {}.{}", major, minor),
            Err(_) => println!("✓ (version query failed but library loaded)"),
        },
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math");
            eprintln!("  (sin/cos/exp/pow/...). Ships with the CUDA Toolkit at");
            eprintln!("  <CUDA>/nvvm/lib64/libnvvm.so. No separate download.");
            ok = false;
        }
    }

    print!("nvJitLink (libnvJitLink.so)... ");
    match nvjitlink_sys::LibNvJitLink::load() {
        Ok(nvj) => match nvj.version() {
            Some((major, minor)) => println!("✓ nvJitLink {}.{}", major, minor),
            None => println!("✓ (version symbol not exported on this CTK)"),
        },
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math.");
            eprintln!("  Ships with the CUDA Toolkit at <CUDA>/lib64/libnvJitLink.so.");
            ok = false;
        }
    }

    print!("libdevice (libdevice.10.bc)... ");
    match libnvvm_sys::find_libdevice() {
        Ok(path) => println!("✓ {}", path.display()),
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math.");
            eprintln!("  Ships with the CUDA Toolkit at");
            eprintln!("  <CUDA>/nvvm/libdevice/libdevice.10.bc. Override the search");
            eprintln!("  with `CUDA_OXIDE_LIBDEVICE=<path>` if you have it elsewhere.");
            ok = false;
        }
    }

    // 6. llc (LLVM static compiler for PTX)
    //
    // cuda-oxide requires LLVM 21+: earlier releases reject modern TMA /
    // tcgen05 / WGMMA intrinsic signatures. Probe in the same order as the
    // pipeline:
    //   1. `CUDA_OXIDE_LLC` (caller-supplied override)
    //   2. Rust toolchain's `llvm-tools` component (auto-installed via rustup)
    //   3. `llc-23`, `llc-22`, `llc-21`, `llc` on `PATH`
    // Whatever we pick, reject if the major version is < 21.
    print!("llc (LLVM)... ");

    // The pipeline's primary entry: the `llc` bundled with the pinned Rust
    // toolchain's `llvm-tools` component. Built with the NVPTX backend
    // enabled, so the typical novice path is `rustup component add llvm-tools`
    // and that's it. Surface the absolute path so doctor's output matches
    // what the pipeline actually invokes.
    let rustup_llc_path: Option<String> = Command::new("rustc")
        .args(["--print", "sysroot", "--print", "host-tuple"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|output| {
            let stdout = String::from_utf8(output.stdout).ok()?;
            let mut lines = stdout.lines();
            let sysroot = lines.next()?;
            let host = lines.next()?;
            let path: std::path::PathBuf = [sysroot, "lib", "rustlib", host, "bin", "llc"]
                .iter()
                .collect();
            path.is_file()
                .then(|| path.to_str().map(str::to_string))
                .flatten()
        });

    let mut candidates: Vec<String> = Vec::new();
    if let Ok(env_llc) = std::env::var("CUDA_OXIDE_LLC") {
        candidates.push(env_llc);
    }
    if let Some(rustup) = rustup_llc_path.clone() {
        candidates.push(rustup);
    }
    for name in ["llc-23", "llc-22", "llc-21", "llc"] {
        candidates.push(name.to_string());
    }

    let llc_pick = candidates.iter().find_map(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                (
                    candidate.clone(),
                    String::from_utf8_lossy(&o.stdout).into_owned(),
                )
            })
    });
    match llc_pick {
        Some((binary, stdout)) => {
            let banner = stdout
                .lines()
                .find(|l| l.contains("LLVM version"))
                .unwrap_or("(version unknown)")
                .trim()
                .to_string();
            let major = banner
                .split("LLVM version")
                .nth(1)
                .and_then(|rest| rest.trim().split('.').next())
                .and_then(|s| s.parse::<u32>().ok());
            match major {
                Some(v) if v >= 21 => println!("✓ {} ({})", banner, binary),
                Some(v) => {
                    println!("✗ {} ({}) — need LLVM 21+", banner, binary);
                    eprintln!(
                        "  Your `{}` reports LLVM {}, which rejects the TMA / tcgen05 /",
                        binary, v
                    );
                    eprintln!("  WGMMA intrinsic signatures cuda-oxide emits. Install a newer");
                    eprintln!("  toolchain (`rustup component add llvm-tools` is usually enough,");
                    eprintln!("  or `sudo apt install llvm-21`) and either add it to PATH or set");
                    eprintln!("  `CUDA_OXIDE_LLC=/path/to/llc`.");
                    ok = false;
                }
                None => println!("✓ {} ({}, version could not be parsed)", banner, binary),
            }
        }
        None => {
            println!("✗ llc not found");
            eprintln!("  cuda-oxide probes (in order): $CUDA_OXIDE_LLC, the Rust toolchain's");
            eprintln!("  llvm-tools llc, then llc-23/llc-22/llc-21/llc on PATH. Easiest fix:");
            eprintln!("    rustup component add llvm-tools");
            eprintln!("  Alternative: `sudo apt install llvm-21` (older versions reject");
            eprintln!("  modern TMA / tcgen05 / WGMMA intrinsics).");
            ok = false;
        }
    }

    // 7. clang / libclang resource dir (host `cuda-bindings` / bindgen)
    //
    // The host `cuda-bindings` crate's build.rs runs bindgen, which loads
    // libclang at runtime to parse `wrapper.h`. That parse pulls in
    // `<stddef.h>`, which must be served from clang's own resource
    // directory — the system/GCC copy is not compatible. Fresh installs of
    // bare `libclang1-*` (without the matching `libclang-common-*-dev`)
    // leave `/usr/lib/clang/*/include` empty and bindgen explodes with a
    // mysterious "'stddef.h' file not found". Catch that up front.
    print!("clang / libclang resource dir... ");
    let clangs = [
        "clang", "clang-22", "clang-21", "clang-20", "clang-19", "clang-18", "clang-17",
        "clang-16", "clang-15",
    ];

    match clang_resource_dir(&clangs) {
        Some((name, ref dir))
            if std::path::Path::new(&format!("{}/include/stddef.h", dir)).exists() =>
        {
            println!("✓ {dir} (via {name})");
        }
        Some((name, ref dir)) => {
            println!("✗ resource dir present but `include/stddef.h` missing: {dir} (via {name})");
            eprintln!("  Host `cuda-bindings` uses bindgen, which needs clang's own stddef.h.");
            eprintln!("  Install the matching dev headers: sudo apt install clang-21");
            eprintln!("  (or libclang-common-21-dev)");
            ok = false;
        }
        None => {
            println!("✗ no clang could report its resource directory");
            eprintln!(
                "  Host `cuda-bindings` uses bindgen, which needs clang + its resource headers."
            );
            eprintln!("  Install with: sudo apt install clang-21");
            eprintln!("  (or at minimum `libclang-common-21-dev` alongside your libclang)");
            ok = false;
        }
    }

    // 8. NVIDIA driver / GPU. Informational, not fatal: only `cargo oxide
    // run` (kernel execution) needs a driver. Cross-compiling and GPU-less
    // CI boxes are supported workflows (`build`/`pipeline` work fine), and
    // the examples-compile CI job is exactly that.
    print!("NVIDIA driver / GPU... ");
    match query_gpu_name_cap_and_driver() {
        Some((name, (major, minor), driver)) => {
            println!(
                "✓ {} (compute capability {}.{}, driver {})",
                name, major, minor, driver
            );
        }
        None => {
            // Some containers mount the kernel driver without shipping
            // nvidia-smi; /proc distinguishes "driver loaded, tool broken"
            // from "no driver at all".
            if Path::new("/proc/driver/nvidia/version").exists() {
                println!("- driver loaded, but nvidia-smi is missing or not reporting a GPU");
                eprintln!("  A kernel-mode NVIDIA driver is present (/proc/driver/nvidia/");
                eprintln!("  version), but `nvidia-smi` did not report a usable GPU.");
                eprintln!("  `cargo oxide run` may still work; arch auto-detection will fall");
                eprintln!("  back to the backend default (override with --arch=<sm_XX>).");
            } else {
                println!("- no NVIDIA driver detected");
                eprintln!("  Only `cargo oxide run` (kernel execution) needs the driver;");
                eprintln!("  `cargo oxide build` and `pipeline` work without one.");
            }
        }
    }

    // 9. cuda-gdb (optional) -- same discovery order as `debug`, which is the
    // command this line is predicting. Probing PATH alone made doctor report a
    // cuda-gdb missing that `cargo oxide debug` would have found.
    print!("cuda-gdb (optional)... ");
    match doctor_toolkit_tool(ctx, "cuda-gdb") {
        Some(path) => match Command::new(&path).arg("--version").output() {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = version.lines().next() {
                    println!("✓ {} ({})", line.trim(), path.display());
                } else {
                    println!("✓ {}", path.display());
                }
            }
            _ => println!("✓ {}", path.display()),
        },
        None => println!("- not found (only needed for `cargo oxide debug`)"),
    }

    // 10. compute-sanitizer (optional) — same discovery order as `sanitize`
    print!("compute-sanitizer (optional)... ");
    match doctor_toolkit_tool(ctx, "compute-sanitizer") {
        Some(path) => match Command::new(&path).arg("--version").output() {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                // `compute-sanitizer --version` prints a banner and a
                // copyright line before the actual "Version ..." line.
                let line = version
                    .lines()
                    .map(str::trim)
                    .find(|line| line.starts_with("Version"))
                    .or_else(|| version.lines().next().map(str::trim));
                if let Some(line) = line {
                    println!("✓ {} ({})", line, path.display());
                } else {
                    println!("✓ {}", path.display());
                }
            }
            _ => println!("✓ {}", path.display()),
        },
        None => {
            println!("- not found (only needed for `cargo oxide sanitize`)");
        }
    }

    println!();
    if ok {
        println!("✅ Environment looks good!");
    } else {
        println!("❌ Some checks failed. Fix the issues above and re-run `cargo oxide doctor`.");
        std::process::exit(1);
    }
}

/// The environment variables `cuda-bindings` consults, in its order
/// (`build.rs`, `TOOLKIT_ENV_VARS`).
const TOOLKIT_ENV_VARS: [&str; 2] = ["CUDA_TOOLKIT_PATH", "CUDA_HOME"];

/// The roots `cuda-bindings` probes when neither variable is set, in its
/// order (`build.rs`, `default_cuda_toolkit_candidates`). This is a fixed
/// preference list, not a version sort, and `/usr/local/cuda` is its *last*
/// entry rather than the default.
#[cfg(not(windows))]
pub(super) const DEFAULT_TOOLKIT_CANDIDATES: [&str; 4] = [
    "/usr/local/cuda-13.3",
    "/usr/local/cuda-13.2",
    "/usr/local/cuda-13",
    "/usr/local/cuda",
];
#[cfg(windows)]
pub(super) const DEFAULT_TOOLKIT_CANDIDATES: [&str; 2] = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
];

/// The floor `cuda-bindings` enforces (`build.rs`, `MIN_CUDA_VERSION`):
/// `CUDA_VERSION` as `cuda.h` spells it, so 13.0 is 13000.
const MIN_CUDA_VERSION: u32 = 13_000;

/// How a toolkit root was arrived at, so doctor can report the same thing the
/// build script would have decided rather than a plausible-looking guess.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ToolkitSource {
    /// A variable was set and its toolkit validated.
    Explicit(&'static str),
    /// No variable was set; this candidate was the first to validate.
    Discovered,
    /// Nothing validated. `root` is only somewhere to point the message.
    Unresolved,
}

/// A toolkit root together with how it was chosen and what was turned down on
/// the way, which is the part a user needs when two toolkits are installed.
#[derive(Debug)]
pub(super) struct ToolkitChoice {
    pub(super) root: PathBuf,
    pub(super) source: ToolkitSource,
    /// `<root or variable>: <reason>`, in probe order.
    pub(super) rejected: Vec<String>,
}

/// Resolve the CUDA toolkit the way the host `cuda-bindings` build script
/// does: each variable in turn, then the fixed candidate list, accepting the
/// first root that carries a `cuda.h` at or above the version floor.
///
/// Mirrors BY HAND `resolve_cuda_toolkit`, `find_default_cuda_toolkit` and
/// `validate_cuda_toolkit` in the shared `cuda-bindings` build script, which
/// lives in NVlabs/cutile-rs (`cuda-bindings/build.rs`) and is pinned at
/// `cuda-bindings = "0.3.1"` in the root manifest: doctor cannot import it
/// because build-script logic is not a library. If that discovery changes,
/// mirror it here.
///
/// The previous mirror had drifted to "first set variable, else
/// `/usr/local/cuda`", which is wrong in three ways that matter on a machine
/// with more than one toolkit installed (#1265): there is a candidate list
/// before that last entry, each candidate must actually contain a `cuda.h`
/// new enough to clear the floor, and a variable set to an empty string is an
/// error rather than a reason to look elsewhere. Doctor reported a healthy
/// toolkit while the build resolved a different one and then failed on a
/// header, which is precisely the case doctor exists to explain.
///
/// `validate` answers "is this root a usable toolkit", returning its
/// `CUDA_VERSION` or why not; it is injected so the ordering can be tested
/// without a filesystem.
pub(super) fn resolve_cuda_toolkit(
    mut get_env: impl FnMut(&str) -> Option<std::ffi::OsString>,
    mut validate: impl FnMut(&Path) -> Result<u32, String>,
) -> ToolkitChoice {
    for var in TOOLKIT_ENV_VARS {
        let Some(value) = get_env(var) else {
            continue;
        };
        // An explicitly set variable is answered, not stepped over: the build
        // script fails on it rather than falling through to discovery, so a
        // typo in CUDA_HOME must not read here as a healthy default install.
        if value.is_empty() {
            return ToolkitChoice {
                root: last_resort_toolkit_root(),
                source: ToolkitSource::Unresolved,
                rejected: vec![format!("{var} is set to an empty string")],
            };
        }
        let value = PathBuf::from(value);
        return match validate(&value) {
            Ok(_) => ToolkitChoice {
                root: value,
                source: ToolkitSource::Explicit(var),
                rejected: Vec::new(),
            },
            Err(reason) => ToolkitChoice {
                rejected: vec![format!("{var}={} is invalid: {reason}", value.display())],
                root: value,
                source: ToolkitSource::Unresolved,
            },
        };
    }

    let mut rejected = Vec::new();
    for candidate in DEFAULT_TOOLKIT_CANDIDATES {
        match validate(Path::new(candidate)) {
            Ok(_) => {
                return ToolkitChoice {
                    root: PathBuf::from(candidate),
                    source: ToolkitSource::Discovered,
                    rejected,
                };
            }
            Err(reason) => rejected.push(format!("{candidate}: {reason}")),
        }
    }

    ToolkitChoice {
        root: last_resort_toolkit_root(),
        source: ToolkitSource::Unresolved,
        rejected,
    }
}

/// Where to point a message when nothing resolved. Not a discovery result:
/// the build script errors in this situation, and the callers that only want
/// a path to probe still need one.
fn last_resort_toolkit_root() -> PathBuf {
    DEFAULT_TOOLKIT_CANDIDATES
        .last()
        .expect("candidate list is never empty")
        .into()
}

/// `resolve_cuda_toolkit` against the real filesystem, for callers that want
/// only the root.
pub(super) fn cuda_toolkit_root(mut get_env: impl FnMut(&str) -> Option<String>) -> PathBuf {
    resolve_cuda_toolkit(
        |key| get_env(key).map(Into::into),
        |root| {
            validate_cuda_toolkit(
                root,
                std::env::var("CUDA_TOOLKIT_TARGET_DIR").ok().as_deref(),
            )
        },
    )
    .root
}

/// Accept `root` as a toolkit exactly as `validate_cuda_toolkit` does: it
/// must be a directory holding a `cuda.h` in one of the layouts
/// [`cuda_header_candidates`] lists, and that header's `CUDA_VERSION` must
/// clear [`MIN_CUDA_VERSION`].
pub(super) fn validate_cuda_toolkit(
    root: impl AsRef<Path>,
    target_dir_override: Option<&str>,
) -> Result<u32, String> {
    let root = root.as_ref();
    if !root.is_dir() {
        return Err(format!("{} is not a directory", root.display()));
    }
    let candidates = cuda_header_candidates(
        root,
        target_dir_override,
        std::env::consts::ARCH,
        std::env::consts::OS,
    );
    let Some(header) = candidates.iter().find(|path| path.is_file()) else {
        return Err(format!("{} does not contain cuda.h", root.display()));
    };
    let contents = std::fs::read_to_string(header)
        .map_err(|error| format!("{} could not be read: {error}", header.display()))?;
    let version = cuda_version_from_header(&contents)
        .ok_or_else(|| format!("could not find CUDA_VERSION in {}", header.display()))?;
    if version < MIN_CUDA_VERSION {
        return Err(format!(
            "CUDA toolkit {} is too old; the host crates require CUDA 13.0+",
            format_cuda_version(version)
        ));
    }
    Ok(version)
}

/// `#define CUDA_VERSION <number>` out of a `cuda.h`, as the build script
/// reads it: the first such line, by whitespace-separated tokens.
pub(super) fn cuda_version_from_header(contents: &str) -> Option<u32> {
    contents.lines().find_map(|line| {
        let mut tokens = line.split_whitespace();
        match (tokens.next(), tokens.next(), tokens.next()) {
            (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
            _ => None,
        }
    })
}

/// `13000` as `13.0`, matching the build script's `format_cuda_version`.
pub(super) fn format_cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

/// Candidate `cuda.h` paths under `toolkit`, in probe order: the standard
/// `include/` layout first, then the redistributable `targets/<dir>/include`
/// layouts. CUDA names the target dirs after the GPU platform, not the Rust
/// triple: x86_64 Linux hosts use `x86_64-linux`; aarch64 Linux is ambiguous
/// between servers (`sbsa-linux`) and Tegra (`aarch64-linux`), so both are
/// probed in that order. A non-blank `target_dir_override` (the
/// `CUDA_TOOLKIT_TARGET_DIR` variable, like nvcc's `-target-dir`) replaces
/// all candidates with that single directory, including the standard layout.
///
/// Mirrors BY HAND the selection table in the shared `cuda-bindings` build
/// sources in NVlabs/cutile-rs (`cuda-bindings/toolkit_target.rs`,
/// `resolve_toolkit_include_candidates`): doctor cannot import it because
/// build-script sources are not a library. If the selection there changes,
/// mirror it here.
///
/// `arch` and `os` are the host CPU architecture and OS; the caller passes
/// `std::env::consts::ARCH` / `std::env::consts::OS` (doctor runs at
/// runtime, so there are no cargo target cfgs to consult). Injected as
/// parameters for unit tests.
pub(super) fn cuda_header_candidates(
    toolkit: impl AsRef<Path>,
    target_dir_override: Option<&str>,
    arch: &str,
    os: &str,
) -> Vec<PathBuf> {
    let base = toolkit.as_ref();
    if let Some(dir) = target_dir_override.filter(|dir| !dir.trim().is_empty()) {
        // Match cuda-bindings: a named target tree is exclusive. The top-level
        // include directory usually aliases the host tree and must not hide a
        // missing or incompatible explicitly selected tree.
        return vec![base.join("targets").join(dir).join("include/cuda.h")];
    }
    let mut candidates = vec![base.join("include/cuda.h")];
    let target_dirs: &[&str] = match (arch, os) {
        ("x86_64", "linux") => &["x86_64-linux"],
        ("aarch64", "linux") => &["sbsa-linux", "aarch64-linux"],
        _ => &[],
    };
    for dir in target_dirs {
        candidates.push(base.join("targets").join(dir).join("include/cuda.h"));
    }
    candidates
}
