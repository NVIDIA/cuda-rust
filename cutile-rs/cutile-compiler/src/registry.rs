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
use crate::jit_cache::{hex, put_field};
use linkme::distributed_slice;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Re-export so macro-emitted code at user crates can name `linkme`
/// through `cutile::cutile_compiler::registry::linkme` without requiring
/// `linkme` to be a direct dependency of every crate that uses
/// `#[cutile::module]`.
pub use linkme;

/// How a single cuTile module's [`Module`] is built, and what it is built
/// from. `#[cutile::module]` emits one per module as `__MODULE_SOURCE`;
/// every entry registering that module, aliases included, points at it, so
/// a builder cannot be paired with another module's identity.
pub struct ModuleSource {
    /// Per-module AST builder. Returns just *this* module's [`Module`] —
    /// no recursive dep aggregation. Dep traversal happens at JIT time by
    /// walking `use` statements against this registry.
    pub build: fn() -> Module,

    /// Hex SHA-256 over the macro's own source and the inputs `build` takes
    /// from it: module name, captured source text, source file, and the line
    /// and column the spans are anchored at.
    pub identity: &'static str,
}

/// Registry entry for a single cuTile module.
///
/// Every `#[cutile::module]` emits a `static` of this type into
/// [`CUTILE_MODULES`]. The JIT iterates the slice at startup and indexes
/// entries by [`absolute_path`](Self::absolute_path).
pub struct CutileModuleEntry {
    /// Absolute Rust path, captured via `module_path!()` at the definition
    /// site (e.g. `"cutile::core"`, `"my_crate::kernels"`).
    pub absolute_path: &'static str,

    /// The module registered at this path.
    pub source: &'static ModuleSource,
}

/// Distributed slice of all `#[cutile::module]`-annotated modules linked
/// into the binary.
///
/// Read at JIT time to look up modules by absolute path.
#[distributed_slice]
pub static CUTILE_MODULES: [CutileModuleEntry] = [..];

/// Hex SHA-256 over every registered `(absolute_path, identity)` pair: it
/// changes whenever a module the JIT can resolve a `use` to changes, or a
/// path starts or stops resolving.
pub fn registry_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        let mut entries: Vec<(&str, &str)> = CUTILE_MODULES
            .iter()
            .map(|e| (e.absolute_path, e.source.identity))
            .collect();
        // Slice order follows link order, which the digest must not.
        entries.sort_unstable();
        let mut h = Sha256::new();
        h.update(b"cutile-module-registry-v1\0");
        for (path, identity) in entries {
            put_field(&mut h, path.as_bytes());
            put_field(&mut h, identity.as_bytes());
        }
        hex(&h.finalize())
    })
}
