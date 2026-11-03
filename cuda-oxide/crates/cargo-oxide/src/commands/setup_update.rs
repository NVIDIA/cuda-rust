/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::backend;
use crate::backend_source;
use std::path::{Path, PathBuf};

use super::*;

// =============================================================================
// Setup command
// =============================================================================

/// What `cargo oxide setup` does for the current project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupPlan {
    /// Inside the monorepo: build the local backend source and publish it to
    /// the shared cache.
    BuildAndPublish { codegen_crate: PathBuf },
    /// Outside the monorepo: rebuild the backend from the project's
    /// cuda-oxide dependency, the same refresh `update` performs.
    RefreshFromDependency { project_dir: PathBuf },
    /// Outside the monorepo with a backend pin set: refuse, as `update` does.
    RefusePinned(String),
}

/// Chooses what `setup` does, with the ambient `CUDA_OXIDE_BACKEND` injected
/// for the same reason as [`pin_refusal_with_env`].
pub(super) fn plan_setup_with_env(
    ctx: &Context,
    backend_env: Option<std::ffi::OsString>,
) -> SetupPlan {
    // inside the workspace the local source is the backend, and publishing it
    // is the point of setup there, so a pin changes nothing
    if ctx.is_workspace {
        return SetupPlan::BuildAndPublish {
            codegen_crate: ctx.codegen_crate.clone(),
        };
    }

    // outside the workspace `codegen_crate` is the user's project, not a
    // backend source tree, so building it compiles the user's crate and finds
    // no backend artifact (#1688). The backend source is the project's
    // cuda-oxide dependency, which `update` already knows how to rebuild from
    if let Some(refusal) = pin_refusal_with_env("setup", ctx, backend_env) {
        return SetupPlan::RefusePinned(refusal);
    }
    SetupPlan::RefreshFromDependency {
        project_dir: ctx.workspace_root.clone(),
    }
}

/// Explicitly build (or rebuild) the codegen backend.
///
/// Normally the backend is built automatically on every `run`/`build`/`pipeline`
/// invocation. `setup` exists for first-time setup, CI, or after pulling new
/// changes when you want to rebuild without running an example.
///
/// Inside the workspace this builds the local source and publishes it to the
/// shared cache. Outside it, this rebuilds the backend from the project's
/// cuda-oxide dependency exactly like `cargo oxide update`. The caller
/// resolves the context with [`resolve_setup_context`], which never builds
/// the backend, so each mode builds it once.
pub fn setup(ctx: &Context) {
    match plan_setup_with_env(ctx, std::env::var_os("CUDA_OXIDE_BACKEND")) {
        SetupPlan::BuildAndPublish { codegen_crate } => build_and_publish(&codegen_crate),
        SetupPlan::RefreshFromDependency { project_dir } => {
            refresh_from_dependency(&project_dir);
            println!();
            println!("You can now use:");
            println!("  cargo oxide run");
            println!("  cargo oxide build");
        }
        SetupPlan::RefusePinned(refusal) => {
            eprintln!("Error: {refusal}");
            std::process::exit(1);
        }
    }
}

fn build_and_publish(codegen_crate: &Path) {
    println!("Building cuda-oxide codegen backend...");
    println!();

    let built_so = backend::build_backend_from_source(codegen_crate);

    println!();
    println!("✓ Backend is ready. You can now use:");
    println!("  cargo oxide run <example>");
    println!("  cargo oxide build <example>");

    // A project outside this repository resolves the backend through the
    // shared cache, since `find_workspace_root` finds no
    // `crates/rustc-codegen-cuda` above it. Publishing the build there keeps
    // those projects on the backend that was just built instead of on whatever
    // the cache last held. The cache records the commit it came from, so only
    // projects whose cuda-oxide dependency resolves to this checkout's HEAD
    // pick it up; any other project rebuilds from its own dependency.
    match backend::publish_to_cache(&built_so, codegen_crate) {
        Some(published) => {
            println!();
            println!("✓ Published to {}", published.path.display());
            match published.source_rev {
                Some(rev) => println!(
                    "  Projects outside this repo whose cuda-oxide dependency resolves to {} will use this build.",
                    backend_source::short_rev(&rev)
                ),
                None => println!(
                    "  Projects outside this repo without a cuda-oxide dependency will use this build."
                ),
            }
        }
        None => {
            eprintln!();
            eprintln!("Warning: could not publish the backend to the shared cache.");
            eprintln!("Projects outside this repo may keep using an older build.");
            eprintln!("Set CUDA_OXIDE_BACKEND to this build to override.");
        }
    }
}

/// How `cargo oxide update` should behave for the current project mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdatePlan {
    /// Inside the monorepo: tell the user to run `setup` (non-destructive).
    AdviseSetup,
    /// Inside the monorepo with `--force`: rebuild via `setup`.
    RunSetup,
    /// Outside the monorepo: clear and rebuild the shared cache.
    RefreshCache,
}

pub fn plan_update(is_workspace: bool, force: bool) -> UpdatePlan {
    match (is_workspace, force) {
        (true, false) => UpdatePlan::AdviseSetup,
        (true, true) => UpdatePlan::RunSetup,
        (false, _) => UpdatePlan::RefreshCache,
    }
}

/// The backend pin that outranks the shared cache `command` would refresh,
/// if any.
///
/// Both the `CUDA_OXIDE_BACKEND` env var and a `.cargo/cuda-oxide.toml`
/// `backend` entry sit above the cache in backend discovery, so a refreshed
/// cache would never be consulted while either is set. `update`, and `setup`
/// outside the workspace, refuse rather than mislead.
fn pin_refusal(command: &str, ctx: &Context) -> Option<String> {
    pin_refusal_with_env(command, ctx, std::env::var_os("CUDA_OXIDE_BACKEND"))
}

/// `pin_refusal` with the ambient `CUDA_OXIDE_BACKEND` injected.
///
/// The env var is checked before the project pin, so resolution has to be
/// injectable for unit tests: a developer with `CUDA_OXIDE_BACKEND` exported
/// would otherwise get the env refusal for every input, including the
/// unpinned case that must return `None`. Same rationale as
/// `nvvm_ir_requested_with_env`.
pub(super) fn pin_refusal_with_env(
    command: &str,
    ctx: &Context,
    backend_env: Option<std::ffi::OsString>,
) -> Option<String> {
    if backend_env.is_some() {
        return Some(format!(
            "CUDA_OXIDE_BACKEND is set, so `cargo oxide {command}` will not\n\
             modify the shared cache. Unset CUDA_OXIDE_BACKEND and re-run, or\n\
             rebuild the pinned backend path yourself."
        ));
    }
    ctx.config.backend.as_deref().map(|pinned| {
        format!(
            "`.cargo/cuda-oxide.toml` pins the backend to {}, so\n\
             `cargo oxide {command}` will not modify the shared cache. Remove the\n\
             `backend` entry and re-run, or rebuild the pinned path yourself.",
            pinned.display()
        )
    })
}

/// Rebuilds the backend of the project in `project_dir`, outside the
/// workspace, from the commit its cuda-oxide dependency resolves to (see
/// [`backend::refresh_cached_backend`]). Shared by `update` and `setup`, so
/// both commands refresh a standalone project the same way.
fn refresh_from_dependency(project_dir: &Path) {
    // each route prints its own specific line (cached from the dependency,
    // built in place for a path dependency, or fetched from main), so the
    // framing here stays route-neutral
    println!("Building the codegen backend for this project...");
    println!();
    let so = backend::refresh_cached_backend(project_dir);
    println!();
    println!("✓ Backend ready at {}", so.display());
}

/// Refresh the codegen backend used by this project.
///
/// Inside the cuda-oxide workspace the authoritative backend is the local
/// source tree, so the default path points at `cargo oxide setup`. Outside
/// the workspace, the shared `~/.cargo/cuda-oxide/` cache is cleared and
/// rebuilt from the commit the project's cuda-oxide dependency resolves to
/// (or from a fresh `main` clone when the project has no such dependency).
pub fn update(ctx: &Context, force: bool) {
    if let Some(refusal) = pin_refusal("update", ctx) {
        eprintln!("Error: {refusal}");
        std::process::exit(1);
    }

    match plan_update(ctx.is_workspace, force) {
        UpdatePlan::AdviseSetup => {
            println!("Inside the cuda-oxide workspace the codegen backend is built from");
            println!("local source (`crates/rustc-codegen-cuda`).");
            println!();
            println!("Run `cargo oxide setup` to rebuild and publish to the shared cache,");
            println!("or pass `--force` to run setup from this command.");
        }
        UpdatePlan::RunSetup => {
            println!("`--force` requested inside the workspace; running setup...");
            println!();
            setup(ctx);
        }
        UpdatePlan::RefreshCache => refresh_from_dependency(&ctx.workspace_root),
    }
}
