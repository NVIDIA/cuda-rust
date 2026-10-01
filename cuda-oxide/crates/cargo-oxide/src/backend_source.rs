/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Where a project outside the repository gets its backend source.
//!
//! Kernels compile against the `cuda-device` / `cuda-host` crates at the commit
//! Cargo resolved for them. The backend that lowers those kernels,
//! `librustc_codegen_cuda`, lives in the same repository and has to come from
//! the same commit: `cuda-device` only declares stubs (every method body is
//! `unreachable!()`), and the backend recognises each stub by its path and
//! supplies the whole lowering. A backend from another commit compiles the
//! kernels differently, or not at all, with no error pointing at the cause.
//!
//! So the backend is built from the checkout Cargo already made for the
//! dependency instead of from a separately cloned `main`:
//!
//! ```text
//! Cargo.lock   cuda-device = git+https://github.com/NVlabs/cuda-oxide.git#<sha>
//!                  │  cargo metadata: package.manifest_path
//!                  ▼
//! ~/.cargo/git/checkouts/cuda-oxide-<hash>/<sha>/crates/cuda-device/Cargo.toml
//!                  │  walk up to the repository root
//!                  ▼
//! ~/.cargo/git/checkouts/cuda-oxide-<hash>/<sha>/crates/rustc-codegen-cuda   built
//! ~/.cargo/git/checkouts/cuda-oxide-<hash>/<sha>/rust-toolchain.toml         nightly it needs
//! ```
//!
//! The checkout's `rust-toolchain.toml` matters as much as the code: the
//! backend is a rustc plugin and only loads into the toolchain that built it,
//! so that file names the nightly the project itself has to use.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The cuda-oxide crates a project depends on directly. Any one of them
/// identifies the checkout: they ship in a single repository.
const CUDA_OXIDE_CRATES: &[&str] = &["cuda-device", "cuda-host", "cuda-macros"];

/// The backend crate, relative to a historical (flat) cuda-oxide checkout root.
pub const CODEGEN_CRATE_SUBDIR: &str = "crates/rustc-codegen-cuda";

/// The backend crate in the combined repository, relative to the git root.
pub const NESTED_CODEGEN_CRATE_SUBDIR: &str = "cuda-oxide/crates/rustc-codegen-cuda";

/// Returns the codegen crate in either supported repository layout.
pub fn codegen_crate_in_checkout(root: &Path) -> PathBuf {
    let nested = root.join(NESTED_CODEGEN_CRATE_SUBDIR);
    if nested.join("Cargo.toml").is_file() {
        nested
    } else {
        root.join(CODEGEN_CRATE_SUBDIR)
    }
}

/// The cuda-oxide checkout a project's dependency resolves to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencySource {
    /// A git dependency: Cargo's checkout of the repository at `rev`.
    Git {
        /// Repository root of the checkout.
        checkout: PathBuf,
        /// Full commit hash Cargo resolved (the `#<sha>` of the package source).
        rev: String,
    },
    /// A path dependency: a local checkout of the repository.
    Path {
        /// Repository root of the checkout.
        checkout: PathBuf,
    },
}

impl DependencySource {
    /// Repository root of the checkout.
    pub fn checkout(&self) -> &Path {
        match self {
            Self::Git { checkout, .. } | Self::Path { checkout } => checkout,
        }
    }

    /// The backend crate inside the checkout.
    pub fn codegen_crate(&self) -> PathBuf {
        codegen_crate_in_checkout(self.checkout())
    }

    /// The commit the checkout is at, for git dependencies.
    pub fn rev(&self) -> Option<&str> {
        match self {
            Self::Git { rev, .. } => Some(rev),
            Self::Path { .. } => None,
        }
    }

    /// One-line identity for messages.
    pub fn describe(&self) -> String {
        match self {
            Self::Git { rev, .. } => format!("cuda-oxide {} (git dependency)", short_rev(rev)),
            Self::Path { checkout } => format!(
                "cuda-oxide checkout {} (path dependency)",
                checkout.display()
            ),
        }
    }
}

/// Abbreviated commit hash for messages.
pub fn short_rev(rev: &str) -> &str {
    rev.get(..10).unwrap_or(rev)
}

/// The nightly a checkout pins in its `rust-toolchain.toml`, when readable.
pub fn pinned_channel(checkout: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(checkout.join("rust-toolchain.toml")).ok()?;
    crate::commands::parse_rust_toolchain_toml(&contents)
        .ok()
        .map(|pin| pin.channel)
}

/// Resolves the cuda-oxide checkout the project in `project_dir` depends on.
///
/// Returns `Ok(None)` when no cuda-oxide crate is in the dependency graph.
/// With `read_only`, Cargo may neither fetch nor write `Cargo.lock`
/// (`--offline --locked`); passive commands such as `doctor` use this so a
/// diagnostic never touches the network or the project.
pub fn resolve_dependency_source(
    project_dir: &Path,
    read_only: bool,
) -> Result<Option<DependencySource>, String> {
    let metadata = cargo_metadata(project_dir, read_only)?;
    dependency_source_from_metadata(&metadata)
}

/// Reads the dependency checkout out of `cargo metadata` output.
///
/// Every cuda-oxide crate in the graph must resolve to one checkout; two
/// checkouts would mean two commits and no single backend to build for them.
pub fn dependency_source_from_metadata(
    metadata: &serde_json::Value,
) -> Result<Option<DependencySource>, String> {
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "`cargo metadata` output has no packages".to_string())?;

    let mut sources: Vec<DependencySource> = Vec::new();
    for package in packages {
        let Some(name) = package.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if !CUDA_OXIDE_CRATES.contains(&name) {
            continue;
        }
        let manifest = package
            .get("manifest_path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("`cargo metadata` lists `{name}` without a manifest_path"))?;
        let rev = match package.get("source").and_then(serde_json::Value::as_str) {
            None => None,
            Some(source) => Some(
                git_source_rev(source)
                    .ok_or_else(|| {
                        format!(
                            "`{name}` comes from `{source}`; only git and path dependencies \
                             carry the cuda-oxide checkout the backend is built from"
                        )
                    })?
                    .to_string(),
            ),
        };
        let checkout = checkout_root(Path::new(manifest)).ok_or_else(|| {
            format!(
                "`{name}` at {manifest} is not inside a cuda-oxide checkout (no \
                 `{CODEGEN_CRATE_SUBDIR}` above it), so there is no backend to build"
            )
        })?;
        let source = match rev {
            Some(rev) => DependencySource::Git { checkout, rev },
            None => DependencySource::Path { checkout },
        };
        if !sources.contains(&source) {
            sources.push(source);
        }
    }

    match sources.as_slice() {
        [] => Ok(None),
        [source] => Ok(Some(source.clone())),
        several => Err(format!(
            "cuda-oxide crates resolve from more than one checkout, so there is no single \
             backend to build:\n{}",
            several
                .iter()
                .map(|source| format!("  {}", source.describe()))
                .collect::<Vec<_>>()
                .join("\n")
        )),
    }
}

/// The resolved commit of a git package source.
///
/// Cargo writes git sources as `git+<url>?<rev|branch|tag>=<spec>#<sha>`; the
/// fragment is always the full commit hash, whatever the spec was.
fn git_source_rev(source: &str) -> Option<&str> {
    let rest = source.strip_prefix("git+")?;
    let (_, rev) = rest.rsplit_once('#')?;
    (!rev.is_empty()).then_some(rev)
}

/// Walks up from a crate manifest to the repository root, recognised by the
/// backend crate living under it.
///
/// The walk never leaves the repository the manifest belongs to: it stops at
/// the first directory that is itself a repository root (`.git`, or the
/// `.cargo-ok` marker Cargo writes into every git checkout). Without that
/// boundary a trimmed checkout would keep climbing and could latch onto an
/// unrelated cuda-oxide tree higher up, such as a developer's own clone.
fn checkout_root(manifest: &Path) -> Option<PathBuf> {
    let mut dir = manifest.parent()?;
    loop {
        // cuda-oxide/ has the crates; its parent owns the workspace and pin.
        if dir.join("Cargo.toml").is_file()
            && codegen_crate_in_checkout(dir).join("Cargo.toml").is_file()
        {
            return Some(dir.to_path_buf());
        }
        if dir.join(".git").exists() || dir.join(".cargo-ok").exists() {
            return None;
        }
        dir = dir.parent()?;
    }
}

/// `cargo metadata` for the project, with dependencies.
///
/// `--all-features`, so a cuda-oxide crate declared `optional = true` behind
/// a feature is still found: the backend must follow that commit whichever
/// features the build enables. Read-only (`doctor`): `--offline --locked`, so
/// a missing or stale `Cargo.lock` is reported instead of being resolved
/// against whatever the local git database holds and written to disk, and
/// stderr is captured so the caller can fold Cargo's reason into its report.
/// Otherwise the user's own lock discipline (`--locked` / `--frozen` /
/// `--offline`, which cargo-oxide passes through to cargo) is honoured, so
/// resolving here never writes a lockfile the build was told to refuse to
/// touch, and Cargo's progress ("Updating git repository ...") streams to the
/// terminal since a first fetch can take a while.
fn cargo_metadata(project_dir: &Path, read_only: bool) -> Result<serde_json::Value, String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["metadata", "--format-version=1", "--all-features"])
        .current_dir(project_dir);
    if read_only {
        cmd.args(["--offline", "--locked"]);
    } else {
        cmd.args(lock_discipline_flags(std::env::args()));
        cmd.stderr(Stdio::inherit());
    }
    let output = cmd
        .output()
        .map_err(|error| format!("could not start `cargo metadata`: {error}"))?;
    if !output.status.success() {
        // Cargo's diagnostic, not its first chatter: a concurrent cargo can
        // put "Blocking waiting for file lock on package cache" ahead of the
        // `error:` line, and `doctor` prints this reason verbatim.
        let stderr = String::from_utf8_lossy(&output.stderr);
        let lines = || {
            stderr
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
        };
        let reason = lines()
            .find(|line| line.starts_with("error"))
            .or_else(|| lines().next())
            .map(|line| format!(": {line}"))
            .unwrap_or_default();
        return Err(format!(
            "`cargo metadata` failed ({}){reason}",
            output.status
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("could not parse `cargo metadata` output: {error}"))
}

/// The lock-discipline flags in a `cargo oxide` command line, to repeat on the
/// `cargo metadata` call. cargo-oxide defines none of these itself, so any
/// occurrence before a bare `--` is a passthrough argument for cargo; after
/// `--` they belong to the program being run and are ignored.
fn lock_discipline_flags(args: impl IntoIterator<Item = String>) -> Vec<String> {
    args.into_iter()
        .take_while(|arg| arg != "--")
        .filter(|arg| matches!(arg.as_str(), "--locked" | "--frozen" | "--offline"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::SystemTime;

    const REPO: &str = "https://github.com/NVlabs/cuda-oxide.git";
    const SHA: &str = "a1b4f11882592fae9d022c86a9d8d1a4c9426980";

    /// Lays out what Cargo's checkout of the repository looks like: the crate
    /// the project depends on plus the backend crate beside it.
    fn checkout_with_backend(root: &Path, crate_name: &str) -> PathBuf {
        for sub in [CODEGEN_CRATE_SUBDIR, &format!("crates/{crate_name}")] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("Cargo.toml"), "[package]\n").unwrap();
        }
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\"]\nexclude = [\"crates/rustc-codegen-cuda\"]\n",
        )
        .unwrap();
        root.join("crates").join(crate_name).join("Cargo.toml")
    }

    fn package(name: &str, source: Option<&str>, manifest: &Path) -> serde_json::Value {
        json!({
            "name": name,
            "source": source,
            "manifest_path": manifest,
        })
    }

    fn metadata(packages: Vec<serde_json::Value>) -> serde_json::Value {
        json!({ "packages": packages })
    }

    /// The common case: `cuda-device = { git = ..., rev = ... }`. The backend
    /// must come from Cargo's checkout of that same commit, located by walking
    /// up from the crate's manifest.
    #[test]
    fn git_dependency_resolves_to_cargos_checkout_and_commit() {
        let root = tempdir();
        let checkout = root.join("checkouts/cuda-oxide-6d394bb0/a1b4f11");
        let manifest = checkout_with_backend(&checkout, "cuda-device");
        let source = format!("git+{REPO}?rev=a1b4f118#{SHA}");
        let md = metadata(vec![
            package("maxproj", None, &root.join("Cargo.toml")),
            package("cuda-device", Some(&source), &manifest),
        ]);

        let resolved = dependency_source_from_metadata(&md).unwrap().unwrap();
        assert_eq!(
            resolved,
            DependencySource::Git {
                checkout: checkout.clone(),
                rev: SHA.to_string(),
            }
        );
        assert_eq!(
            resolved.codegen_crate(),
            checkout.join(CODEGEN_CRATE_SUBDIR)
        );
        assert_eq!(resolved.rev(), Some(SHA));
    }

    /// A project pulls in several cuda-oxide crates (`cuda-device` brings
    /// `cuda-macros`). They share one checkout, so they resolve to one source
    /// rather than tripping the "more than one checkout" error.
    #[test]
    fn several_crates_from_one_checkout_resolve_once() {
        let root = tempdir();
        let checkout = root.join("a1b4f11");
        let device = checkout_with_backend(&checkout, "cuda-device");
        let macros = checkout_with_backend(&checkout, "cuda-macros");
        let source = format!("git+{REPO}#{SHA}");
        let md = metadata(vec![
            package("cuda-device", Some(&source), &device),
            package("cuda-macros", Some(&source), &macros),
        ]);

        let resolved = dependency_source_from_metadata(&md).unwrap().unwrap();
        assert_eq!(resolved.rev(), Some(SHA));
        assert_eq!(resolved.checkout(), checkout);
    }

    /// `cuda-device = { path = "../cuda-oxide/crates/cuda-device" }`: a local
    /// checkout with no commit to record; the backend builds in place there.
    #[test]
    fn path_dependency_resolves_to_the_local_checkout() {
        let root = tempdir();
        let checkout = root.join("dev/cuda-oxide");
        let manifest = checkout_with_backend(&checkout, "cuda-host");
        let md = metadata(vec![package("cuda-host", None, &manifest)]);

        let resolved = dependency_source_from_metadata(&md).unwrap().unwrap();
        assert_eq!(resolved, DependencySource::Path { checkout });
        assert_eq!(resolved.rev(), None);
    }

    /// No cuda-oxide crate in the graph means nothing pins a commit; the
    /// caller falls back to the shared cache and the `main` clone.
    #[test]
    fn no_cuda_oxide_crates_means_no_source() {
        let root = tempdir();
        let md = metadata(vec![
            package("maxproj", None, &root.join("Cargo.toml")),
            package(
                "serde",
                Some("registry+https://github.com/rust-lang/crates.io-index"),
                &root.join("registry/serde/Cargo.toml"),
            ),
        ]);
        assert_eq!(dependency_source_from_metadata(&md).unwrap(), None);
    }

    /// Two different commits in one graph cannot share a backend. Refusing
    /// beats silently picking one: kernels from the other crate would be
    /// lowered by a backend that never saw them.
    #[test]
    fn crates_from_two_commits_are_an_error() {
        let root = tempdir();
        let old = checkout_with_backend(&root.join("b22efa9"), "cuda-device");
        let new = checkout_with_backend(&root.join("a1b4f11"), "cuda-host");
        let md = metadata(vec![
            package(
                "cuda-device",
                Some(&format!(
                    "git+{REPO}#b22efa99e000000000000000000000000000000000"
                )),
                &old,
            ),
            package("cuda-host", Some(&format!("git+{REPO}#{SHA}")), &new),
        ]);

        let error = dependency_source_from_metadata(&md).unwrap_err();
        assert!(error.contains("more than one checkout"), "{error}");
        assert!(error.contains("b22efa99e0"), "{error}");
        assert!(error.contains("a1b4f11882"), "{error}");
    }

    /// A registry tarball ships one crate, not the repository, so no backend
    /// source exists for it. Say so instead of walking up into `~/.cargo`.
    #[test]
    fn registry_dependency_is_an_error() {
        let root = tempdir();
        let manifest = root.join("registry/src/cuda-device-0.2.1/Cargo.toml");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(&manifest, "[package]\n").unwrap();
        let md = metadata(vec![package(
            "cuda-device",
            Some("registry+https://github.com/rust-lang/crates.io-index"),
            &manifest,
        )]);

        let error = dependency_source_from_metadata(&md).unwrap_err();
        assert!(error.contains("only git and path dependencies"), "{error}");
    }

    /// A checkout without the backend crate (a trimmed fork, a partial copy)
    /// has nothing to build; the error names what was expected where.
    #[test]
    fn checkout_without_the_backend_crate_is_an_error() {
        let root = tempdir();
        if checkout_root(&root.join("Cargo.toml")).is_some() {
            return; // the temp dir itself sits inside a cuda-oxide checkout
        }
        let manifest = root.join("somewhere/crates/cuda-device/Cargo.toml");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(&manifest, "[package]\n").unwrap();
        let md = metadata(vec![package(
            "cuda-device",
            Some(&format!("git+{REPO}#{SHA}")),
            &manifest,
        )]);

        let error = dependency_source_from_metadata(&md).unwrap_err();
        assert!(error.contains(CODEGEN_CRATE_SUBDIR), "{error}");
    }

    /// The walk stops at the repository the manifest belongs to. A trimmed
    /// checkout (marked `.cargo-ok` like every Cargo git checkout) nested under
    /// a directory that does hold a full cuda-oxide tree must NOT resolve to
    /// that outer tree: it is somebody else's checkout at some other commit.
    #[test]
    fn checkout_root_never_climbs_out_of_the_repository() {
        let outer = tempdir();
        checkout_with_backend(&outer, "cuda-device");
        let trimmed = outer.join("nested/cuda-oxide-abcd/0123456");
        let manifest = trimmed.join("crates/cuda-device/Cargo.toml");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(&manifest, "[package]\n").unwrap();

        // Without a repository marker the walk climbs to the outer tree.
        assert_eq!(checkout_root(&manifest), Some(outer.clone()));

        // With one it stops there and reports nothing.
        std::fs::write(trimmed.join(".cargo-ok"), b"").unwrap();
        assert_eq!(checkout_root(&manifest), None);

        // A plain git clone is bounded the same way.
        std::fs::remove_file(trimmed.join(".cargo-ok")).unwrap();
        std::fs::create_dir_all(trimmed.join(".git")).unwrap();
        assert_eq!(checkout_root(&manifest), None);

        // Worktrees use a .git file instead of a directory.
        std::fs::remove_dir(trimmed.join(".git")).unwrap();
        std::fs::write(trimmed.join(".git"), "gitdir: /unused\n").unwrap();
        assert_eq!(checkout_root(&manifest), None);

        // The marker directory itself is still eligible when it has the crate.
        checkout_with_backend(&trimmed, "cuda-device");
        assert_eq!(checkout_root(&manifest), Some(trimmed));
    }

    /// The parser is fed real `cargo metadata` output, not only hand-built
    /// JSON: a package with no cuda-oxide dependency resolves to nothing, and
    /// a path dependency on a checkout resolves to it with `source: null` the
    /// way Cargo emits it. Skipped when no `cargo` is on PATH.
    #[test]
    fn real_cargo_metadata_resolves_a_path_dependency() {
        if Command::new("cargo").arg("--version").output().is_err() {
            return; // no cargo here; nothing to observe
        }
        let root = tempdir();
        let checkout = root.join("cuda-oxide");
        checkout_with_backend(&checkout, "cuda-device");
        let device = checkout.join("crates/cuda-device");
        std::fs::create_dir_all(device.join("src")).unwrap();
        std::fs::write(
            device.join("Cargo.toml"),
            "[package]\nname = \"cuda-device\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(device.join("src/lib.rs"), "").unwrap();

        let plain = root.join("plain");
        std::fs::create_dir_all(plain.join("src")).unwrap();
        std::fs::write(
            plain.join("Cargo.toml"),
            "[package]\nname = \"plain\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(plain.join("src/lib.rs"), "").unwrap();
        assert_eq!(
            resolve_dependency_source(&plain, false).unwrap(),
            None,
            "a project without cuda-oxide crates pins no commit"
        );

        let user = root.join("user");
        std::fs::create_dir_all(user.join("src")).unwrap();
        std::fs::write(
            user.join("Cargo.toml"),
            format!(
                "[package]\nname = \"user\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
                 [dependencies]\ncuda-device = {{ path = {:?} }}\n",
                device.display().to_string()
            ),
        )
        .unwrap();
        std::fs::write(user.join("src/lib.rs"), "").unwrap();
        // Read-only first: with no Cargo.lock yet, `--locked` must refuse
        // rather than write one (this is what keeps `doctor` from silently
        // resolving a fresh scaffold against a stale local git database).
        let error = resolve_dependency_source(&user, true).unwrap_err();
        assert!(error.contains("--locked"), "{error}");
        assert!(
            !user.join("Cargo.lock").exists(),
            "read-only must not write"
        );

        let resolved = resolve_dependency_source(&user, false)
            .unwrap()
            .expect("the path dependency must resolve to its checkout");
        assert_eq!(resolved.rev(), None, "a path dependency has no commit");
        assert_eq!(
            resolved.checkout().canonicalize().unwrap(),
            checkout.canonicalize().unwrap()
        );

        // Once the lockfile exists, read-only resolution agrees with it.
        assert!(user.join("Cargo.lock").exists());
        assert_eq!(
            resolve_dependency_source(&user, true).unwrap(),
            Some(resolved)
        );

        // `--all-features`: a cuda-oxide crate behind an optional feature is
        // still the commit the backend must follow.
        let gated = root.join("gated");
        std::fs::create_dir_all(gated.join("src")).unwrap();
        std::fs::write(
            gated.join("Cargo.toml"),
            format!(
                "[package]\nname = \"gated\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
                 [features]\ngpu = [\"dep:cuda-device\"]\n\
                 [dependencies]\ncuda-device = {{ path = {:?}, optional = true }}\n",
                device.display().to_string()
            ),
        )
        .unwrap();
        std::fs::write(gated.join("src/lib.rs"), "").unwrap();
        let resolved = resolve_dependency_source(&gated, false)
            .unwrap()
            .expect("an optional cuda-oxide dependency must still be found");
        assert_eq!(
            resolved.checkout().canonicalize().unwrap(),
            checkout.canonicalize().unwrap()
        );
    }

    /// Only the flags meant for cargo count: anything after a bare `--` is
    /// for the program `cargo oxide run` launches.
    #[test]
    fn lock_discipline_flags_are_taken_from_the_cargo_side_only() {
        let args = |list: &[&str]| list.iter().map(|arg| arg.to_string()).collect::<Vec<_>>();
        assert_eq!(
            lock_discipline_flags(args(&["cargo-oxide", "build", "--locked", "--release"])),
            ["--locked"]
        );
        assert_eq!(
            lock_discipline_flags(args(&[
                "cargo-oxide",
                "run",
                "x",
                "--frozen",
                "--offline",
                "--",
                "--offline"
            ])),
            ["--frozen", "--offline"]
        );
        assert!(lock_discipline_flags(args(&["cargo-oxide", "run", "--", "--locked"])).is_empty());
    }

    /// `doctor` prints this error verbatim, so a broken manifest must yield
    /// the exit status plus Cargo's own first line, not an empty reason.
    #[test]
    fn real_cargo_metadata_failure_carries_cargos_reason() {
        if Command::new("cargo").arg("--version").output().is_err() {
            return; // no cargo here; nothing to observe
        }
        let broken = tempdir();
        std::fs::write(broken.join("Cargo.toml"), "[package]\n").unwrap();

        let error = resolve_dependency_source(&broken, true).unwrap_err();
        assert!(error.starts_with("`cargo metadata` failed ("), "{error}");
        assert!(
            error.contains("): error"),
            "the reason must carry Cargo's first stderr line: {error}"
        );
    }

    /// The fragment is the resolved commit no matter how the dependency was
    /// spelled (`rev`, `branch`, `tag`, or nothing).
    #[test]
    fn git_source_rev_reads_the_fragment() {
        assert_eq!(
            git_source_rev(&format!("git+{REPO}?rev=a1b4f118#{SHA}")),
            Some(SHA)
        );
        assert_eq!(
            git_source_rev(&format!("git+{REPO}?branch=main#{SHA}")),
            Some(SHA)
        );
        assert_eq!(git_source_rev(&format!("git+{REPO}#{SHA}")), Some(SHA));
        assert_eq!(git_source_rev(&format!("git+{REPO}")), None);
        assert_eq!(
            git_source_rev("registry+https://github.com/rust-lang/crates.io-index"),
            None
        );
    }

    /// The nightly a commit needs is whatever its `rust-toolchain.toml` says;
    /// that is what the mismatch message tells the user to set.
    #[test]
    fn pinned_channel_reads_the_checkout_toolchain_file() {
        let checkout = tempdir();
        std::fs::write(
            checkout.join("rust-toolchain.toml"),
            "[toolchain]\nchannel = \"nightly-2026-08-28\"\ncomponents = [\"rustc-dev\"]\n",
        )
        .unwrap();
        assert_eq!(
            pinned_channel(&checkout),
            Some("nightly-2026-08-28".to_string())
        );
        assert_eq!(pinned_channel(&checkout.join("absent")), None);
    }

    #[test]
    fn describe_abbreviates_the_commit() {
        let source = DependencySource::Git {
            checkout: PathBuf::from("/x"),
            rev: SHA.to_string(),
        };
        assert_eq!(source.describe(), "cuda-oxide a1b4f11882 (git dependency)");
        assert_eq!(short_rev("abc"), "abc");
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "cargo-oxide-backend-source-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
