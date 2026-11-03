// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage for the build script's CUDA `targets/<dir>` selection table.
//!
//! `build.rs` pulls in the same file with `include!`, because cargo never
//! builds a build script as a test target.

include!("../toolkit_target.rs");

/// `(target_arch, target_os, expected candidates)`.
const CASES: &[(&str, &str, &[&str])] = &[
    // x86_64 Linux: the one server layout.
    ("x86_64", "linux", &["x86_64-linux"]),
    // aarch64 Linux is ambiguous between server and Tegra, so both are
    // offered. `sbsa-linux` stays first so an install that carries both
    // trees resolves to the server directory.
    ("aarch64", "linux", &["sbsa-linux", "aarch64-linux"]),
    // Non-Linux hosts have no `targets/` tree at all. An arch-only check
    // would fall through and claim a Linux directory: `aarch64-apple-darwin`
    // would resolve to `sbsa-linux` and `x86_64-pc-windows-msvc` to
    // `x86_64-linux`.
    ("aarch64", "macos", &[]),
    ("x86_64", "macos", &[]),
    ("x86_64", "windows", &[]),
    ("aarch64", "windows", &[]),
    // `aarch64-linux-android` splits into components containing "linux" but
    // is not a CUDA platform; keying off `CARGO_CFG_TARGET_OS` rejects it.
    ("aarch64", "android", &[]),
    // Architectures CUDA does not ship.
    ("riscv64", "linux", &[]),
    ("powerpc64le", "linux", &[]),
    ("", "", &[]),
];

#[test]
fn toolkit_target_dirs_matches_table() {
    for &(arch, os, expected) in CASES {
        assert_eq!(
            toolkit_target_dirs(arch, os),
            expected,
            "unexpected targets/ candidates for arch={arch:?} os={os:?}"
        );
    }
}

#[test]
fn every_candidate_is_a_linux_directory_for_its_own_arch() {
    // The property a naive `targets/*` glob breaks: a candidate must never
    // belong to another architecture, or a multi-target install hands the
    // build the wrong headers.
    for &(arch, os, expected) in CASES {
        for dir in expected {
            assert!(
                dir.ends_with("-linux"),
                "candidate {dir:?} for arch={arch:?} os={os:?} is not a Linux tree"
            );
            let owner_arch = dir.trim_end_matches("-linux");
            assert!(
                owner_arch == arch || (arch == "aarch64" && owner_arch == "sbsa"),
                "candidate {dir:?} belongs to {owner_arch:?}, not arch={arch:?}"
            );
        }
    }
}

/// Builds a throwaway toolkit tree containing a `cuda.h` at each given
/// relative directory, and returns its root. No temp-dir crate: this package
/// takes no new dependencies.
fn fake_toolkit(tag: &str, include_dirs: &[&str]) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "cuda-bindings-toolkit-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    for dir in include_dirs {
        let full = root.join(dir);
        std::fs::create_dir_all(&full).expect("create fake toolkit dir");
        std::fs::write(full.join("cuda.h"), b"/* fake */\n").expect("write fake cuda.h");
    }
    root
}

#[test]
fn tegra_layout_resolves_to_aarch64_linux() {
    // The layout this table exists for: CUDA for Tegra ships only
    // `targets/aarch64-linux/`, and no top-level `include/`.
    let root = fake_toolkit("tegra", &["targets/aarch64-linux/include"]);
    let candidates = resolve_toolkit_include_candidates(&root, None, "aarch64", "linux");
    let selected = select_include_dir(&candidates).expect("Tegra layout must resolve");
    assert_eq!(selected, &root.join("targets/aarch64-linux/include"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn sbsa_layout_still_resolves_to_sbsa_linux() {
    let root = fake_toolkit("sbsa", &["targets/sbsa-linux/include"]);
    let candidates = resolve_toolkit_include_candidates(&root, None, "aarch64", "linux");
    let selected = select_include_dir(&candidates).expect("sbsa layout must resolve");
    assert_eq!(selected, &root.join("targets/sbsa-linux/include"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn multi_target_install_never_hands_x86_64_the_sbsa_headers() {
    // The failure mode that rules out a `targets/*` glob: `sbsa-linux` sorts
    // before `x86_64-linux`.
    let root = fake_toolkit(
        "multi",
        &[
            "targets/sbsa-linux/include",
            "targets/x86_64-linux/include",
            "targets/aarch64-linux/include",
        ],
    );
    let candidates = resolve_toolkit_include_candidates(&root, None, "x86_64", "linux");
    let selected = select_include_dir(&candidates).expect("x86_64 layout must resolve");
    assert_eq!(selected, &root.join("targets/x86_64-linux/include"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn top_level_include_wins_over_any_targets_tree() {
    let root = fake_toolkit("standard", &["include", "targets/x86_64-linux/include"]);
    let candidates = resolve_toolkit_include_candidates(&root, None, "x86_64", "linux");
    let selected = select_include_dir(&candidates).expect("standard layout must resolve");
    assert_eq!(selected, &root.join("include"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn env_override_is_the_sole_include_candidate() {
    // CUDA_TOOLKIT_TARGET_DIR names one tree by hand, like nvcc's
    // `-target-dir`: neither the table's candidates nor the top-level
    // `include/` are consulted at all.
    let root = fake_toolkit(
        "override",
        &[
            "targets/sbsa-linux/include",
            "targets/aarch64-linux/include",
        ],
    );
    let candidates =
        resolve_toolkit_include_candidates(&root, Some("aarch64-linux"), "aarch64", "linux");
    assert_eq!(candidates, vec![root.join("targets/aarch64-linux/include")]);
    let selected = select_include_dir(&candidates).expect("override layout must resolve");
    assert_eq!(selected, &root.join("targets/aarch64-linux/include"));
    let _ = std::fs::remove_dir_all(&root);

    // A blank override means "unset": the table decides, as before.
    for blank in [None, Some(""), Some("  ")] {
        assert_eq!(
            resolve_toolkit_include_candidates(&root, blank, "aarch64", "linux"),
            vec![
                root.join("include"),
                root.join("targets/sbsa-linux/include"),
                root.join("targets/aarch64-linux/include"),
            ],
            "blank override {blank:?} must fall back to the table"
        );
    }
}

#[test]
fn env_override_beats_the_top_level_include() {
    // On a standard install, `include/` is (usually a symlink to) the host
    // architecture's `targets/` tree. A cross-build override must not let it
    // shadow the requested tree.
    let root = fake_toolkit("override-cross", &["include", "targets/sbsa-linux/include"]);
    let candidates =
        resolve_toolkit_include_candidates(&root, Some("sbsa-linux"), "x86_64", "linux");
    assert_eq!(candidates, vec![root.join("targets/sbsa-linux/include")]);
    let selected = select_include_dir(&candidates).expect("override must resolve");
    assert_eq!(selected, &root.join("targets/sbsa-linux/include"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn wrong_env_override_fails_instead_of_falling_back() {
    // The override is existence-probed, not trusted: a typo must surface as
    // the "does not contain cuda.h" error listing what was probed, never as
    // a silent fallback to the top-level include or another architecture's
    // tree, even on a standard install where `include/` exists.
    let root = fake_toolkit("override-bad", &["include", "targets/sbsa-linux/include"]);
    let candidates =
        resolve_toolkit_include_candidates(&root, Some("aarch64-qnx"), "aarch64", "linux");
    assert_eq!(candidates, vec![root.join("targets/aarch64-qnx/include")]);
    assert!(
        select_include_dir(&candidates).is_none(),
        "a wrong override must not resolve the top-level or sbsa-linux headers"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn non_linux_probes_only_the_top_level_include() {
    let root = fake_toolkit("darwin", &["targets/sbsa-linux/include"]);
    let candidates = resolve_toolkit_include_candidates(&root, None, "aarch64", "macos");
    assert_eq!(candidates, vec![root.join("include")]);
    assert!(
        select_include_dir(&candidates).is_none(),
        "a macOS build must not resolve the sbsa-linux headers"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn non_linux_never_resolves_a_target_dir() {
    for os in ["macos", "windows", "android", "ios", "freebsd", "none"] {
        for arch in ["x86_64", "aarch64", "riscv64"] {
            assert!(
                toolkit_target_dirs(arch, os).is_empty(),
                "arch={arch:?} os={os:?} must not resolve a targets/ directory"
            );
        }
    }
}
