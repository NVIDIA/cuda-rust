/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::backend;
use std::path::{Path, PathBuf};

use super::*;

/// Project-local cuda-oxide defaults loaded from `.cargo/cuda-oxide.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OxideConfig {
    /// Explicit backend shared object path.
    pub backend: Option<PathBuf>,
    /// Default CUDA architecture for codegen commands.
    pub default_arch: Option<String>,
    /// Additional rustflags appended after cuda-oxide's required flags.
    pub extra_rustflags: Vec<String>,
    /// Environment variables applied to child Cargo invocations.
    pub env: Vec<(String, String)>,
}

/// Pre-resolved context shared across all commands.
///
/// Built once at startup by [`resolve_context`] and passed by reference to
/// every command handler. Avoids repeated filesystem walks and backend builds.
pub struct Context {
    /// Absolute path to the workspace root (contains top-level `Cargo.toml`).
    pub workspace_root: PathBuf,
    /// Path to `crates/rustc-codegen-cuda` (backend source tree).
    pub codegen_crate: PathBuf,
    /// Path to `crates/rustc-codegen-cuda/examples/`.
    pub examples_dir: PathBuf,
    /// Path to the built `librustc_codegen_cuda.so` shared object.
    pub backend_so: PathBuf,
    /// True when running from inside the cuda-oxide workspace; false for
    /// standalone projects scaffolded by `cargo oxide new`.
    pub is_workspace: bool,
    /// Project-local cuda-oxide defaults.
    pub config: OxideConfig,
}

impl Context {
    /// The CUDA Oxide component containing the selected backend. The checkout
    /// root may instead be the parent stable workspace in the monorepo.
    pub(super) fn oxide_workspace_root(&self) -> &Path {
        if !self.is_workspace {
            return &self.workspace_root;
        }
        self.codegen_crate
            .parent()
            .and_then(Path::parent)
            .expect("workspace backend must live under crates/rustc-codegen-cuda")
    }
}

/// Resolve the workspace root and backend, or exit with a helpful error.
///
/// Supports two modes:
/// - **Workspace mode**: CWD is inside the cuda-oxide repo (detected by the
///   flat `crates/rustc-codegen-cuda` path or the combined repository's
///   `cuda-oxide/crates/rustc-codegen-cuda` path). Examples are resolved from
///   the workspace examples directory.
/// - **Standalone mode**: CWD has a `Cargo.toml` but is not inside the
///   workspace. The backend is built from the commit the project's cuda-oxide
///   dependency resolves to, or taken from the shared cache when that already
///   holds it (see `backend::standalone_backend`). Commands like `run`
///   operate on the current directory directly.
pub fn resolve_context() -> Context {
    if let Some(workspace_root) = backend::find_workspace_root() {
        let codegen_crate = backend::codegen_crate_path(&workspace_root);
        let examples_dir = codegen_crate.join("examples");
        let config = load_oxide_config(&workspace_root);
        let backend_so = backend::find_or_build_backend(&workspace_root, config.backend.as_deref());
        return Context {
            workspace_root,
            codegen_crate,
            examples_dir,
            backend_so,
            is_workspace: true,
            config,
        };
    }

    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error: cannot determine current directory: {}", e);
        std::process::exit(1);
    });

    if cwd.join("Cargo.toml").is_file() {
        let config = load_oxide_config(&cwd);
        let backend_so = backend::find_or_build_backend(&cwd, config.backend.as_deref());
        return Context {
            workspace_root: cwd.clone(),
            codegen_crate: cwd.clone(),
            examples_dir: cwd.clone(),
            backend_so,
            is_workspace: false,
            config,
        };
    }

    eprintln!("Error: Could not find cuda-oxide workspace or a standalone Cargo.toml.");
    eprintln!();
    eprintln!("Run from inside the cuda-oxide repository, or from a project created");
    eprintln!("with `cargo oxide new <name>`.");
    std::process::exit(1);
}

/// Resolve a context for commands that must not build or fetch the backend.
///
/// Identical discovery to [`resolve_context`], except the backend `.so` is
/// only located via [`backend::backend_so_candidate`], never built and never
/// cloned, and an invalid `.cargo/cuda-oxide.toml` degrades to defaults with
/// a warning instead of exiting (so `doctor` can report it as a failed
/// check). Passive commands such as `doctor`, `clean`, `list` and `fmt` must
/// remain usable without triggering backend setup or network access.
/// `run`/`build`/`pipeline`/`setup` still build the backend on demand.
pub fn resolve_passive_context() -> Context {
    if let Some(workspace_root) = backend::find_workspace_root() {
        let codegen_crate = backend::codegen_crate_path(&workspace_root);
        let examples_dir = codegen_crate.join("examples");
        let config = load_oxide_config_lenient(&workspace_root);
        let backend_so = backend::backend_so_candidate(&workspace_root, config.backend.as_deref());
        return Context {
            workspace_root,
            codegen_crate,
            examples_dir,
            backend_so,
            is_workspace: true,
            config,
        };
    }

    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error: cannot determine current directory: {}", e);
        std::process::exit(1);
    });

    if cwd.join("Cargo.toml").is_file() {
        let config = load_oxide_config_lenient(&cwd);
        let backend_so = backend::backend_so_candidate(&cwd, config.backend.as_deref());
        return Context {
            workspace_root: cwd.clone(),
            codegen_crate: cwd.clone(),
            examples_dir: cwd.clone(),
            backend_so,
            is_workspace: false,
            config,
        };
    }

    eprintln!("Error: Could not find cuda-oxide workspace or a standalone Cargo.toml.");
    eprintln!();
    eprintln!("Run from inside the cuda-oxide repository, or from a project created");
    eprintln!("with `cargo oxide new <name>`.");
    std::process::exit(1);
}

// =============================================================================
// Helpers
// =============================================================================

/// Load `.cargo/cuda-oxide.toml`, exiting on an invalid config.
///
/// Build commands ([`resolve_context`]) stay strict: they must not run with
/// a config the user wrote but cargo-oxide cannot honor.
pub(super) fn load_oxide_config(workspace_root: &Path) -> OxideConfig {
    match inspect_oxide_config(workspace_root) {
        OxideConfigInspection::Missing => OxideConfig::default(),
        OxideConfigInspection::Valid { config, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            config
        }
        OxideConfigInspection::Invalid { errors, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            for error in errors {
                eprintln!("Error: {error}");
            }
            std::process::exit(1);
        }
    }
}

/// Load `.cargo/cuda-oxide.toml`, falling back to defaults on an invalid
/// config instead of exiting.
///
/// Passive commands ([`resolve_passive_context`]: `doctor`, `clean`, ...)
/// must stay usable with a broken config. `doctor` in particular re-inspects
/// the file and reports the failure as a regular failed check, which it can
/// only do if context resolution survives long enough for the scan to start.
pub(super) fn load_oxide_config_lenient(workspace_root: &Path) -> OxideConfig {
    match inspect_oxide_config(workspace_root) {
        OxideConfigInspection::Missing => OxideConfig::default(),
        OxideConfigInspection::Valid { config, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            config
        }
        OxideConfigInspection::Invalid { errors, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            for error in errors {
                eprintln!("Warning: {error}");
            }
            eprintln!("Warning: ignoring invalid cuda-oxide config and continuing with defaults");
            OxideConfig::default()
        }
    }
}

/// Result of reading `.cargo/cuda-oxide.toml` without exiting the process.
///
/// `doctor` uses this so a bad config is reported alongside other checks
/// instead of aborting before the rest of the environment scan runs.
#[derive(Debug)]
pub(super) enum OxideConfigInspection {
    Missing,
    Valid {
        config: OxideConfig,
        warnings: Vec<String>,
    },
    Invalid {
        errors: Vec<String>,
        warnings: Vec<String>,
    },
}

pub(super) fn inspect_oxide_config(workspace_root: &Path) -> OxideConfigInspection {
    let config_path = workspace_root.join(".cargo/cuda-oxide.toml");
    if !config_path.exists() {
        return OxideConfigInspection::Missing;
    }

    let source = match std::fs::read_to_string(&config_path) {
        Ok(source) => source,
        Err(error) => {
            return OxideConfigInspection::Invalid {
                errors: vec![format!(
                    "could not read cuda-oxide config {}: {error}",
                    config_path.display()
                )],
                warnings: Vec::new(),
            };
        }
    };

    let document: toml::Value = match toml::from_str(&source) {
        Ok(document) => document,
        Err(error) => {
            return OxideConfigInspection::Invalid {
                errors: vec![format!(
                    "could not parse cuda-oxide config {}: {error}",
                    config_path.display()
                )],
                warnings: Vec::new(),
            };
        }
    };

    let Some(table) = document.as_table() else {
        return OxideConfigInspection::Invalid {
            errors: vec![format!(
                "cuda-oxide config {} must be a TOML table",
                config_path.display()
            )],
            warnings: Vec::new(),
        };
    };

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let backend = match optional_config_string(table, "backend", &config_path) {
        Ok(value) => value
            .map(PathBuf::from)
            .map(|path| absolutize_config_path(path, &config_path)),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let default_arch = match optional_config_string(table, "default-arch", &config_path) {
        Ok(value) => {
            if let Some(ref arch) = value {
                // Validate with the same parser the consumers use
                // (`parse_nvvm_arch` normalizes `sm_XX` / `compute_XX` / bare
                // `XX` into a `CudaArch`), so load-time validation is exactly
                // as permissive as what a build would accept. Non-`sm_XX`
                // spellings work but are advisory-warned: `sm_XX` is the form
                // `--arch` and the rest of cargo-oxide document.
                match parse_nvvm_arch(arch) {
                    Ok(parsed) => {
                        if !arch.starts_with("sm_") {
                            warnings.push(format!(
                                "cuda-oxide config {} spells `default-arch` as `{arch}`; \
                                 prefer the `{}` form used by `--arch`",
                                config_path.display(),
                                parsed.sm()
                            ));
                        }
                    }
                    Err(error) => {
                        errors.push(format!(
                            "cuda-oxide config {} field `default-arch`: {error}",
                            config_path.display()
                        ));
                    }
                }
            }
            value
        }
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let extra_rustflags = match optional_config_string_array(table, "extra-rustflags", &config_path)
    {
        Ok(value) => value,
        Err(error) => {
            errors.push(error);
            Vec::new()
        }
    };
    let env = match table.get("env") {
        None => Vec::new(),
        Some(value) => match parse_config_env(value, &config_path) {
            Ok(env) => {
                for (key, _) in &env {
                    if matches!(key.as_str(), "RUSTFLAGS" | "CARGO_ENCODED_RUSTFLAGS") {
                        warnings.push(format!(
                            "cuda-oxide config {} `[env]` key `{key}` is ignored; \
                             use `extra-rustflags` for project rustc defaults",
                            config_path.display()
                        ));
                    }
                }
                env
            }
            Err(error) => {
                errors.push(error);
                Vec::new()
            }
        },
    };

    if !errors.is_empty() {
        return OxideConfigInspection::Invalid { errors, warnings };
    }

    OxideConfigInspection::Valid {
        config: OxideConfig {
            backend,
            default_arch,
            extra_rustflags,
            env,
        },
        warnings,
    }
}

/// Outcome of doctor's project-config check, separated from printing so
/// tests can assert the doctor-level behavior (headline, detail lines,
/// pass/fail) directly.
pub(super) struct OxideConfigCheck {
    /// Line printed after the check label.
    pub(super) headline: String,
    /// Indented detail lines (warnings, then errors).
    pub(super) details: Vec<String>,
    /// Whether the check failed (flips doctor to a nonzero exit).
    pub(super) failed: bool,
}

pub(super) fn check_oxide_config(workspace_root: &Path) -> OxideConfigCheck {
    let config_path = workspace_root.join(".cargo/cuda-oxide.toml");
    match inspect_oxide_config(workspace_root) {
        OxideConfigInspection::Missing => OxideConfigCheck {
            headline: "- not present (using defaults)".to_string(),
            details: Vec::new(),
            failed: false,
        },
        OxideConfigInspection::Valid { config, warnings } => OxideConfigCheck {
            headline: match &config.default_arch {
                Some(arch) => format!("✓ {} (default-arch = {arch})", config_path.display()),
                None => format!("✓ {}", config_path.display()),
            },
            details: warnings
                .into_iter()
                .map(|warning| format!("⚠ {warning}"))
                .collect(),
            failed: false,
        },
        OxideConfigInspection::Invalid { errors, warnings } => OxideConfigCheck {
            headline: format!("✗ {}", config_path.display()),
            details: warnings
                .into_iter()
                .map(|warning| format!("⚠ {warning}"))
                .chain(errors.into_iter().map(|error| format!("✗ {error}")))
                .collect(),
            failed: true,
        },
    }
}

pub(super) fn doctor_report_oxide_config(ctx: &Context, ok: &mut bool) {
    print!("Project config (.cargo/cuda-oxide.toml)... ");
    let check = check_oxide_config(&ctx.workspace_root);
    println!("{}", check.headline);
    for line in check.details {
        println!("  {line}");
    }
    if check.failed {
        *ok = false;
    }
}

fn absolutize_config_path(path: PathBuf, config_path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

fn optional_config_string(
    table: &toml::Table,
    key: &str,
    config_path: &Path,
) -> Result<Option<String>, String> {
    match table.get(key) {
        None => Ok(None),
        Some(value) => value.as_str().map(|s| Some(s.to_string())).ok_or_else(|| {
            format!(
                "cuda-oxide config {} field `{key}` must be a string",
                config_path.display()
            )
        }),
    }
}

fn optional_config_string_array(
    table: &toml::Table,
    key: &str,
    config_path: &Path,
) -> Result<Vec<String>, String> {
    match table.get(key) {
        None => Ok(Vec::new()),
        Some(value) => {
            let array = value.as_array().ok_or_else(|| {
                format!(
                    "cuda-oxide config {} field `{key}` must be an array of strings",
                    config_path.display()
                )
            })?;
            array
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_string).ok_or_else(|| {
                        format!(
                            "cuda-oxide config {} field `{key}` must be an array of strings",
                            config_path.display()
                        )
                    })
                })
                .collect()
        }
    }
}

fn parse_config_env(
    value: &toml::Value,
    config_path: &Path,
) -> Result<Vec<(String, String)>, String> {
    let table = value.as_table().ok_or_else(|| {
        format!(
            "cuda-oxide config {} field `env` must be a table of strings",
            config_path.display()
        )
    })?;
    let mut env: Vec<_> = table
        .iter()
        .map(|(key, value)| {
            let value = value.as_str().ok_or_else(|| {
                format!(
                    "cuda-oxide config {} env value `{key}` must be a string",
                    config_path.display()
                )
            })?;
            Ok((key.clone(), value.to_string()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    env.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(env)
}

/// Example roots belonging to the selected CUDA Oxide component.
pub(super) fn example_roots(ctx: &Context) -> Vec<PathBuf> {
    let mut roots = vec![ctx.examples_dir.clone()];
    if ctx.is_workspace {
        let cute_examples = ctx.oxide_workspace_root().join("cuteir/examples");
        if cute_examples.is_dir() {
            roots.push(cute_examples);
        }
    }
    roots
}

pub(super) fn resolve_example_dir(ctx: &Context, example: &str) -> PathBuf {
    let roots = example_roots(ctx);
    let matches: Vec<_> = roots
        .iter()
        .map(|root| root.join(example))
        .filter(|path| path.is_dir())
        .collect();
    if let [example_dir] = matches.as_slice() {
        return example_dir.clone();
    }
    if matches.len() > 1 {
        eprintln!("Error: Example name is ambiguous: {example}");
        for path in matches {
            eprintln!("  - {}", path.display());
        }
        eprintln!("Example names must be unique across all configured roots.");
        std::process::exit(1);
    }

    eprintln!("Error: Example not found: {example}");
    eprintln!();
    eprintln!("Available examples:");
    let mut names: Vec<_> = roots
        .iter()
        .filter_map(|root| std::fs::read_dir(root).ok())
        .flat_map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().is_dir())
                .map(|entry| entry.file_name().to_string_lossy().to_string())
        })
        .collect();
    names.sort();
    names.dedup();
    for name in names {
        eprintln!("  - {name}");
    }
    std::process::exit(1);
}
