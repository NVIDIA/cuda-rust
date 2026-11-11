/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared CUDA Toolkit discovery for compiler libraries and libdevice.
//!
//! These helpers return candidate paths in discovery order. They do not require
//! the CUDA Toolkit, a driver, or a GPU to be present.

use std::{
    cmp::Ordering,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

const ROOT_ENV_VARS: &[&str] = &["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"];
const WINDOWS_CUDA_DEFAULT_ROOT: &str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
const LINUX_CUDA_DEFAULT_ROOTS: &[&str] = &["/usr/local/cuda", "/opt/cuda"];
const LIBNVVM_WINDOWS_PREFIX: &str = "nvvm64_";
const NVJITLINK_WINDOWS_PREFIX: &str = "nvJitLink_";

/// Candidate paths to the libNVVM dynamic library.
pub fn libnvvm_dll_candidates(target: &str) -> Vec<PathBuf> {
    libnvvm_dll_candidates_from_roots(root_candidates(DefaultRoots::for_target(target)), target)
}

/// Candidate paths to the nvJitLink dynamic library.
pub fn nvjitlink_dll_candidates(target: &str) -> Vec<PathBuf> {
    nvjitlink_dll_candidates_from_roots(root_candidates(DefaultRoots::for_target(target)), target)
}

/// Candidate paths to CUDA libdevice bitcode.
pub fn libdevice_candidates(target: &str) -> Vec<PathBuf> {
    libdevice_candidates_from_roots(root_candidates(DefaultRoots::for_target(target)))
}

#[derive(Clone, Copy)]
enum DefaultRoots {
    Linux,
    Windows,
}

impl DefaultRoots {
    fn for_target(target: &str) -> Self {
        if is_windows_target(target) {
            Self::Windows
        } else {
            Self::Linux
        }
    }
}

fn root_candidates(defaults: DefaultRoots) -> Vec<PathBuf> {
    root_candidates_from_env(std::env::vars_os(), defaults)
}

fn root_candidates_from_env<I>(env: I, defaults: DefaultRoots) -> Vec<PathBuf>
where
    I: IntoIterator<Item = (OsString, OsString)>,
{
    let entries = env
        .into_iter()
        .filter_map(|(key, value)| key.into_string().ok().map(|key| (key, value)))
        .collect::<Vec<_>>();
    if matches!(defaults, DefaultRoots::Linux) {
        // Match upstream std::env::var semantics, including empty and repeated roots.
        return ROOT_ENV_VARS
            .iter()
            .filter_map(|key| env_value(&entries, key))
            .filter_map(|value| value.to_str())
            .map(PathBuf::from)
            .chain(LINUX_CUDA_DEFAULT_ROOTS.iter().map(PathBuf::from))
            .collect();
    }
    let mut roots = Vec::new();

    for key in ROOT_ENV_VARS {
        if let Some(value) = env_value(&entries, key) {
            push_if_not_empty(&mut roots, value);
        }
    }

    let mut versioned = entries
        .iter()
        .filter(|(key, value)| key.starts_with("CUDA_PATH_V") && !value.is_empty())
        .collect::<Vec<_>>();
    versioned.sort_by(|(left_key, _), (right_key, _)| {
        compare_cuda_path_version_vars(right_key, left_key).then_with(|| left_key.cmp(right_key))
    });
    for (_, value) in versioned {
        push_if_not_empty(&mut roots, value);
    }

    roots.extend(windows_default_roots());

    dedup(roots)
}

fn env_value<'a>(entries: &'a [(String, OsString)], key: &str) -> Option<&'a OsStr> {
    entries
        .iter()
        .find_map(|(entry_key, value)| (entry_key == key).then_some(value.as_os_str()))
}

fn push_if_not_empty(roots: &mut Vec<PathBuf>, value: &OsStr) {
    if !value.is_empty() {
        roots.push(PathBuf::from(value));
    }
}

fn windows_default_roots() -> Vec<PathBuf> {
    let base = PathBuf::from(WINDOWS_CUDA_DEFAULT_ROOT);
    versioned_cuda_roots(&base)
}

fn versioned_cuda_roots(base: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(base) else {
        return Vec::new();
    };
    let mut roots = entries
        .flatten()
        .filter_map(|entry| {
            if !entry.file_type().ok()?.is_dir() {
                return None;
            }
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let version = cuda_version_parts(name)?;
            Some((version, name.to_ascii_lowercase(), path))
        })
        .collect::<Vec<_>>();
    roots.sort_by(
        |(left_version, left_name, left_path), (right_version, right_name, right_path)| {
            right_version
                .cmp(left_version)
                .then_with(|| left_name.cmp(right_name))
                .then_with(|| left_path.cmp(right_path))
        },
    );
    roots.into_iter().map(|(_, _, path)| path).collect()
}

fn windows_runtime_library_candidates<F, D>(
    roots: &[PathBuf],
    search_dirs: &[PathBuf],
    file_prefix: &str,
    dir_for_root: F,
) -> Vec<PathBuf>
where
    F: Fn(&Path) -> D,
    D: IntoIterator<Item = PathBuf>,
{
    let mut candidates = Vec::new();
    for root in roots {
        for dir in dir_for_root(root) {
            candidates.extend(versioned_windows_dlls(&dir, file_prefix));
        }
    }
    for dir in search_dirs {
        candidates.extend(versioned_windows_dlls(dir, file_prefix));
    }
    dedup(candidates)
}

fn windows_runtime_search_dirs() -> Vec<PathBuf> {
    windows_path_dirs(std::env::var_os("PATH").as_deref())
}

fn windows_path_dirs(path: Option<&OsStr>) -> Vec<PathBuf> {
    dedup(
        path.into_iter()
            .flat_map(std::env::split_paths)
            .filter(|dir| dir.is_absolute())
            .collect(),
    )
}

fn versioned_windows_dlls(directory: &Path, file_prefix: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut files = entries
        .flatten()
        .filter_map(|entry| {
            if !entry.file_type().ok()?.is_file() {
                return None;
            }
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let version = windows_dll_version(name, file_prefix)?;
            Some((version, name.to_ascii_lowercase(), path))
        })
        .collect::<Vec<_>>();
    files.sort_by(
        |(left_version, left_name, left_path), (right_version, right_name, right_path)| {
            right_version
                .cmp(left_version)
                .then_with(|| left_name.cmp(right_name))
                .then_with(|| left_path.cmp(right_path))
        },
    );
    files.into_iter().map(|(_, _, path)| path).collect()
}

fn compare_cuda_path_version_vars(left: &str, right: &str) -> Ordering {
    numeric_version_parts(left).cmp(&numeric_version_parts(right))
}

fn cuda_version_parts(value: &str) -> Option<Vec<u32>> {
    let version = value
        .strip_prefix('v')
        .or_else(|| value.strip_prefix('V'))?;
    let parts = version
        .split('.')
        .map(str::parse)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!parts.is_empty()).then_some(parts)
}

fn windows_dll_version(value: &str, file_prefix: &str) -> Option<Vec<u32>> {
    let prefix = value.get(..file_prefix.len())?;
    if !prefix.eq_ignore_ascii_case(file_prefix) {
        return None;
    }
    let version_with_suffix = value.get(file_prefix.len()..)?;
    let suffix_start = version_with_suffix.len().checked_sub(".dll".len())?;
    let version = version_with_suffix.get(..suffix_start)?;
    let suffix = version_with_suffix.get(suffix_start..)?;
    if !suffix.eq_ignore_ascii_case(".dll") {
        return None;
    }
    numeric_version_parts(version)
}

fn numeric_version_parts(value: &str) -> Option<Vec<u32>> {
    let parts = value
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    (!parts.is_empty()).then_some(parts)
}

fn is_windows_target(target: &str) -> bool {
    target.contains("windows")
}

fn dedup(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.contains(&path) {
            deduped.push(path);
        }
    }
    deduped
}

fn libnvvm_dll_candidates_from_roots(roots: Vec<PathBuf>, target: &str) -> Vec<PathBuf> {
    if is_windows_target(target) {
        windows_runtime_library_candidates(
            &roots,
            &windows_runtime_search_dirs(),
            LIBNVVM_WINDOWS_PREFIX,
            |root| {
                [
                    root.join("nvvm").join("bin").join("x64"),
                    root.join("nvvm").join("bin"),
                ]
            },
        )
    } else {
        roots
            .into_iter()
            .map(|root| root.join("nvvm").join("lib64").join("libnvvm.so"))
            .collect()
    }
}

fn nvjitlink_dll_candidates_from_roots(roots: Vec<PathBuf>, target: &str) -> Vec<PathBuf> {
    if is_windows_target(target) {
        windows_runtime_library_candidates(
            &roots,
            &windows_runtime_search_dirs(),
            NVJITLINK_WINDOWS_PREFIX,
            |root| [root.join("bin").join("x64"), root.join("bin")],
        )
    } else {
        roots
            .into_iter()
            .flat_map(|root| ["lib64", "lib"].map(|dir| root.join(dir).join("libnvJitLink.so")))
            .collect()
    }
}

fn libdevice_candidates_from_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    roots
        .into_iter()
        .map(|root| root.join("nvvm").join("libdevice").join("libdevice.10.bc"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    const WINDOWS_TARGET: &str = "x86_64-pc-windows-msvc";
    const LINUX_TARGET: &str = "x86_64-unknown-linux-gnu";

    #[test]
    fn linux_candidates_match_upstream_in_order() {
        for values in [
            [None, None, None],
            [
                Some("/cuda/toolkit"),
                Some("/cuda/home"),
                Some("/cuda/path"),
            ],
            [Some(""), Some("/opt/cuda"), Some("/opt/cuda")],
        ] {
            let env = ROOT_ENV_VARS
                .iter()
                .zip(values)
                .filter_map(|(key, value)| {
                    value.map(|value| (OsString::from(key), OsString::from(value)))
                })
                .chain([(
                    OsString::from("CUDA_PATH_V99_0"),
                    OsString::from("/windows-only"),
                )]);
            let roots = root_candidates_from_env(env, DefaultRoots::Linux);
            let expected: Vec<_> = values
                .into_iter()
                .flatten()
                .chain(["/usr/local/cuda", "/opt/cuda"])
                .map(PathBuf::from)
                .collect();
            assert_eq!(roots, expected);
            assert_eq!(
                libnvvm_dll_candidates_from_roots(roots.clone(), LINUX_TARGET),
                expected
                    .iter()
                    .map(|root| root.join("nvvm/lib64/libnvvm.so"))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                nvjitlink_dll_candidates_from_roots(roots.clone(), LINUX_TARGET),
                expected
                    .iter()
                    .flat_map(|root| {
                        ["lib64", "lib"].map(|dir| root.join(dir).join("libnvJitLink.so"))
                    })
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                libdevice_candidates_from_roots(roots),
                expected
                    .iter()
                    .map(|root| root.join("nvvm/libdevice/libdevice.10.bc"))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn windows_path_does_not_add_the_current_directory() {
        let absolute = std::env::temp_dir().join("cuda-toolkit-path");
        let path = std::env::join_paths([
            Path::new(""),
            Path::new("."),
            Path::new("relative"),
            absolute.as_path(),
            absolute.as_path(),
        ])
        .unwrap();
        assert_eq!(windows_path_dirs(None), Vec::<PathBuf>::new());
        assert_eq!(windows_path_dirs(Some(&path)), [absolute]);
    }

    #[test]
    fn windows_toolkit_candidates_precede_path_candidates() {
        let base = unique_temp_dir("cuda-toolkit-order");
        let toolkit = base.join("toolkit");
        let path_dir = base.join("path");
        std::fs::create_dir_all(toolkit.join("bin")).unwrap();
        std::fs::create_dir_all(&path_dir).unwrap();
        let first = toolkit.join("bin/nvJitLink_130_0.dll");
        let last = path_dir.join("nvJitLink_999_0.dll");
        std::fs::write(&first, []).unwrap();
        std::fs::write(&last, []).unwrap();
        assert_eq!(
            windows_runtime_library_candidates(
                &[toolkit],
                &[path_dir],
                NVJITLINK_WINDOWS_PREFIX,
                |root| [root.join("bin")],
            ),
            [first, last]
        );
        std::fs::remove_dir_all(base).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_runtime_scan_excludes_cwd_dlls() {
        const CHILD: &str = "CUDA_TOOLKIT_DISCOVERY_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let cwd = std::env::current_dir().unwrap();
            let base = cwd.parent().unwrap();
            for (actual, relative, filename) in [
                (
                    libnvvm_dll_candidates(WINDOWS_TARGET),
                    "nvvm/bin/x64",
                    "nvvm64_1_0.dll",
                ),
                (
                    nvjitlink_dll_candidates(WINDOWS_TARGET),
                    "bin",
                    "nvJitLink_1_0.dll",
                ),
            ] {
                assert_eq!(
                    actual.first(),
                    Some(&base.join("toolkit").join(relative).join(filename))
                );
                assert_eq!(actual.last(), Some(&base.join("path").join(filename)));
                assert!(actual.iter().all(|path| !path.starts_with(&cwd)));
            }
            return;
        }
        let base = unique_temp_dir("cuda-toolkit-cwd");
        let cwd = base.join("cwd");
        let toolkit = base.join("toolkit");
        let path = base.join("path");
        for directory in [
            &cwd,
            &cwd.join("relative"),
            &path,
            &toolkit.join("bin"),
            &toolkit.join("nvvm/bin/x64"),
        ] {
            std::fs::create_dir_all(directory).unwrap();
            for filename in ["nvvm64_1_0.dll", "nvJitLink_1_0.dll"] {
                std::fs::write(directory.join(filename), []).unwrap();
            }
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "tests::windows_runtime_scan_excludes_cwd_dlls",
                "--nocapture",
            ])
            .current_dir(&cwd)
            .env(CHILD, "1");
        for (key, _) in std::env::vars_os() {
            if key
                .to_string_lossy()
                .to_ascii_uppercase()
                .starts_with("CUDA_")
            {
                command.env_remove(key);
            }
        }
        command
            .env(CHILD, "1")
            .env("CUDA_TOOLKIT_PATH", &toolkit)
            .env(
                "PATH",
                std::env::join_paths([Path::new(""), Path::new("."), Path::new("relative"), &path])
                    .unwrap(),
            );
        let output = command
            .output()
            .expect("run discovery in an isolated process");
        std::fs::remove_dir_all(base).unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{now}", std::process::id()))
    }

    #[test]
    fn windows_default_roots_are_discovered_in_numeric_order() {
        let base = unique_temp_dir("cuda-toolkit-roots");
        for version in ["v13.9", "v13.10", "v12.99", "not-a-version"] {
            std::fs::create_dir_all(base.join(version)).expect("create Toolkit root fixture");
        }
        std::fs::write(base.join("v99.0"), []).expect("create non-directory fixture");

        let names = versioned_cuda_roots(&base)
            .into_iter()
            .map(|path| {
                path.file_name()
                    .expect("Toolkit root has a file name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();

        assert_eq!(names, ["v13.10", "v13.9", "v12.99"]);
        std::fs::remove_dir_all(base).expect("remove Toolkit root fixture");
    }

    #[test]
    fn windows_targets_do_not_inherit_linux_default_roots() {
        let roots =
            root_candidates_from_env(Vec::<(OsString, OsString)>::new(), DefaultRoots::Windows);

        for root in LINUX_CUDA_DEFAULT_ROOTS {
            assert!(!roots.contains(&PathBuf::from(root)));
        }
    }

    #[test]
    fn versioned_cuda_path_variables_are_sorted_numerically() {
        let roots = root_candidates_from_env(
            vec![
                (
                    OsString::from("CUDA_PATH_V13_9"),
                    OsString::from(r"D:\CUDA\v13.9"),
                ),
                (
                    OsString::from("CUDA_PATH_V13_10"),
                    OsString::from(r"D:\CUDA\v13.10"),
                ),
            ],
            DefaultRoots::Windows,
        );

        assert_eq!(roots.first(), Some(&PathBuf::from(r"D:\CUDA\v13.10")));
        assert_eq!(roots.get(1), Some(&PathBuf::from(r"D:\CUDA\v13.9")));
    }

    #[test]
    fn windows_dll_scan_rejects_malformed_names_without_panicking() {
        for name in [
            "nvJitLink_\u{e9}abc",
            "nvJitLink_\u{1f4a0}ab",
            "nvJitLink_130_0.d\u{e9}",
            "nvJitLink_130_0",
            "nvJitLink_.dll",
            "nvJitLink_4294967296.dll",
        ] {
            assert_eq!(windows_dll_version(name, NVJITLINK_WINDOWS_PREFIX), None);
        }
    }

    #[test]
    fn windows_dll_scan_is_case_insensitive_and_numeric() {
        let root = unique_temp_dir("cuda-toolkit-dlls");
        let directory = root.join("bin").join("x64");
        std::fs::create_dir_all(&directory).expect("create DLL fixture directory");
        for name in [
            "nvJitLink_99_0.dll",
            "nvJitLink_130_9.dll",
            "NVJITLINK_130_10.DLL",
            "not-nvJitLink_999_0.dll",
        ] {
            std::fs::write(directory.join(name), []).expect("create DLL fixture");
        }
        std::fs::create_dir(directory.join("nvJitLink_999_0.dll"))
            .expect("create non-file DLL fixture");

        let candidates = windows_runtime_library_candidates(
            std::slice::from_ref(&root),
            &[],
            NVJITLINK_WINDOWS_PREFIX,
            |root| [root.join("bin").join("x64")],
        );
        let names = candidates
            .into_iter()
            .map(|path| {
                path.file_name()
                    .expect("DLL candidate has a file name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            names,
            [
                "NVJITLINK_130_10.DLL",
                "nvJitLink_130_9.dll",
                "nvJitLink_99_0.dll",
            ]
        );
        std::fs::remove_dir_all(root).expect("remove DLL fixture");
    }

    #[test]
    fn cuda_path_only_is_first_root() {
        let cuda_path = OsString::from(r"D:\NVIDIA\CUDA\current");
        let roots = root_candidates_from_env(
            vec![(OsString::from("CUDA_PATH"), cuda_path.clone())],
            DefaultRoots::Windows,
        );

        assert_eq!(roots.first(), Some(&PathBuf::from(cuda_path)));
    }

    #[test]
    fn cuda_toolkit_path_precedes_cuda_path() {
        let roots = root_candidates_from_env(
            vec![
                (
                    OsString::from("CUDA_PATH"),
                    OsString::from(r"D:\CUDA\from-cuda-path"),
                ),
                (
                    OsString::from("CUDA_TOOLKIT_PATH"),
                    OsString::from(r"D:\CUDA\from-toolkit-path"),
                ),
            ],
            DefaultRoots::Windows,
        );

        assert_eq!(
            roots.first(),
            Some(&PathBuf::from(r"D:\CUDA\from-toolkit-path"))
        );
        assert_eq!(
            roots.get(1),
            Some(&PathBuf::from(r"D:\CUDA\from-cuda-path"))
        );
    }

    #[test]
    fn spaces_in_cuda_root_are_preserved() {
        let root = PathBuf::from(r"D:\CUDA Toolkit Installs\current");
        let roots = root_candidates_from_env(
            vec![(OsString::from("CUDA_PATH"), root.clone().into_os_string())],
            DefaultRoots::Windows,
        );

        assert!(
            libdevice_candidates_from_roots(roots)
                .contains(&root.join("nvvm").join("libdevice").join("libdevice.10.bc"))
        );
    }
}
