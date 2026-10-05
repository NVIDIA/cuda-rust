/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Linker-based registry for cuTile modules.
//!
//! Each `#[cutile::module]` registers a [`CutileModuleEntry`] in the
//! [`CUTILE_MODULES`] distributed slice via [`linkme`]. The JIT discovers
//! modules at JIT time by walking the kernel's `use` graph and looking up
//! each registered absolute path in this registry.
//!
//! ## How cross-crate registration works
//!
//! `linkme`'s [`distributed_slice`] exploits linker-section concatenation:
//! each registration emits a `static` value tagged with a platform-specific
//! section attribute. At link time the linker concatenates every object
//! file's same-named sections into one contiguous region. At runtime
//! [`CUTILE_MODULES`] iterates that region as a slice of entries. This
//! contract is between the macro and the linker, not Rust's module system —
//! every crate that emits cuTile module entries with the agreed slice name
//! participates in the same registry.

use crate::ast::Module;
use linkme::distributed_slice;

/// Re-export so macro-emitted code at user crates can name `linkme`
/// through `cutile::cutile_compiler::registry::linkme` without requiring
/// `linkme` to be a direct dependency of every crate that uses
/// `#[cutile::module]`.
pub use linkme;

/// Registry entry for a single cuTile module.
///
/// Every `#[cutile::module]` emits a `static` of this type into
/// [`CUTILE_MODULES`]. The JIT iterates the slice at startup and indexes
/// entries by [`absolute_path`](Self::absolute_path).
pub struct CutileModuleEntry {
    /// Absolute Rust path, captured via `module_path!()` at the definition
    /// site (e.g. `"cutile::core"`, `"my_crate::kernels"`).
    pub absolute_path: &'static str,

    /// Per-module AST builder. Returns just *this* module's [`Module`] —
    /// no recursive dep aggregation. Dep traversal happens at JIT time by
    /// walking `use` statements against this registry.
    pub build: fn() -> Module,
}

/// Distributed slice of all `#[cutile::module]`-annotated modules linked
/// into the binary.
///
/// Read at JIT time to look up modules by absolute path.
#[distributed_slice]
pub static CUTILE_MODULES: [CutileModuleEntry] = [..];
