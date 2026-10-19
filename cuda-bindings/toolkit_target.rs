// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Selection of the CUDA toolkit `targets/<dir>` tree for a build target.
//
// Regular `//` comments, not `//!`: this file is pulled in with `include!`
// from both `build.rs` and `tests/toolkit_target.rs`, and an inner doc
// comment is only legal at the top of a real module. Cargo never builds a
// build script as a test target, so sharing the source this way is what lets
// the table below be covered by `cargo test` rather than only by whichever
// machine happens to run the build.

/// CUDA toolkit `targets/` directory names to probe for a build target, most
/// specific first. Empty when CUDA ships no `targets/` tree for that platform.
///
/// `target_arch` and `target_os` are cargo's `CARGO_CFG_TARGET_ARCH` and
/// `CARGO_CFG_TARGET_OS` for the *build target*, not the host. Those two are
/// used rather than splitting the `TARGET` triple because the triple's field
/// count varies (`arch-vendor-os-env` vs `arch-os-env`) and because
/// `aarch64-linux-android` would otherwise look like a Linux target.
///
/// CUDA names these trees after the GPU platform, not the Rust triple:
///
/// | platform | directory |
/// |---|---|
/// | x86_64 Linux | `x86_64-linux` |
/// | aarch64 Linux, server (Grace, Ampere Altra) | `sbsa-linux` |
/// | aarch64 Linux, Tegra (Jetson, Drive) | `aarch64-linux` |
///
/// aarch64 returns both server and Tegra names because a Rust aarch64 Linux
/// triple does not distinguish them -- Jetson builds natively as
/// `aarch64-unknown-linux-gnu`, exactly like a Grace server. The callers probe
/// these in order and take the first that actually contains `cuda.h`, so a
/// single-target install resolves unambiguously either way. `sbsa-linux` is
/// listed first so that an install carrying both trees resolves to the server
/// directory.
///
/// This is deliberately *not* a glob over `targets/*`: on a multi-target
/// install that would let another architecture's headers and stubs shadow the
/// right ones, because `sbsa-linux` sorts before `x86_64-linux`. Every
/// candidate returned here is for the requested architecture only.
fn toolkit_target_dirs(target_arch: &str, target_os: &str) -> &'static [&'static str] {
    if target_os != "linux" {
        return &[];
    }
    match target_arch {
        "x86_64" => &["x86_64-linux"],
        "aarch64" => &["sbsa-linux", "aarch64-linux"],
        _ => &[],
    }
}

/// Include directories to probe for `cuda.h`, in priority order.
///
/// When `override_dir` carries a non-blank `CUDA_TOOLKIT_TARGET_DIR` value,
/// that names one `targets/` tree by hand, like nvcc's `-target-dir` flag,
/// and `{toolkit}/targets/<override>/include` is the *only* candidate: the
/// standard top-level `include/` is deliberately not consulted, because on
/// standard installs it exists (usually as a symlink to the host
/// architecture's `targets/` tree) and would silently shadow the requested
/// tree on every cross-build. The override is not existence-checked either;
/// a wrong value fails with the clear discovery error listing what was
/// probed, never a silent fallback.
///
/// Without an override, the standard top-level `{toolkit}/include` comes
/// first, then `{toolkit}/targets/<dir>/include` for each table candidate
/// from [`toolkit_target_dirs`].
///
/// Fully-qualified `std::path` types, because `build.rs` already imports
/// `Path` and `PathBuf` and this file is `include!`d into it.
fn resolve_toolkit_include_candidates(
    toolkit: &std::path::Path,
    override_dir: Option<&str>,
    target_arch: &str,
    target_os: &str,
) -> Vec<std::path::PathBuf> {
    match override_dir.filter(|dir| !dir.trim().is_empty()) {
        Some(dir) => vec![toolkit.join("targets").join(dir).join("include")],
        None => {
            let mut candidates = vec![toolkit.join("include")];
            for target_dir in toolkit_target_dirs(target_arch, target_os) {
                candidates.push(toolkit.join("targets").join(target_dir).join("include"));
            }
            candidates
        }
    }
}

/// The first candidate that actually contains `cuda.h`.
fn select_include_dir(candidates: &[std::path::PathBuf]) -> Option<&std::path::PathBuf> {
    candidates.iter().find(|dir| dir.join("cuda.h").is_file())
}
