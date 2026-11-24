/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Classification of `use` statements walked by `CUDATileModules::from_kernel`.
//!
//! When the JIT walks a kernel module's `use` graph against the linker
//! registry, each use path falls into one of three buckets:
//!
//! - [`UseClassification::Registered`] — path matches an entry in
//!   [`crate::registry::CUTILE_MODULES`]; load it and recurse on its own
//!   `use` statements.
//! - [`UseClassification::AllowedExternal`] — path's crate root is on the
//!   static allowlist of supported external sources (`ALLOWED_EXTERNAL_PREFIXES`).
//!   Pass through silently; the JIT trusts these names exist.
//! - [`UseClassification::Other`] — neither. Recorded in the import catalog
//!   so that, if the kernel later references an unresolved name, the JIT
//!   can point back to the use statement and explain that the source is
//!   not supported in cuTile kernels.
//!
//! The static lists are intentionally small. cuTile kernels live in a
//! restricted environment; widening these lists is a deliberate decision,
//! not a default.

use crate::ast::Module;
use std::collections::HashMap;
use syn::{ItemMod, UseTree};

/// Crate-roots whose paths are passed through silently when they appear in
/// kernel `use` statements. The kernel writer is trusted to import only
/// supported items from these (e.g. `half::f16`, `half::bf16`).
const ALLOWED_EXTERNAL_PREFIXES: &[&str] = &["half"];

/// Crate-roots that mark a use path as Rust standard-library. Used only to
/// add a one-line hint to the unresolved-name error — kernels can't run
/// stdlib code, but the rule is the same as for any other unsupported
/// source.
const STDLIB_PREFIXES: &[&str] = &["std", "core", "alloc", "proc_macro", "test"];

/// One name introduced into kernel scope by a `use` statement, paired with
/// the source path it came from.
///
/// Glob imports (`use foo::bar::*;`) carry no specific imported name; they
/// are recorded with `name = None` and may still inform error messages
/// (the JIT can warn that *some* unresolved names might originate there).
#[derive(Debug, Clone)]
pub struct UseImport {
    /// The local name introduced into scope (or `None` for glob imports).
    /// For renames (`use foo::Bar as Baz;`), this is the alias (`"Baz"`).
    pub name: Option<String>,
    /// The full path being imported (e.g., `"std::collections::HashMap"`).
    pub path: String,
}

/// Outcome of classifying a single `use` path against the linker registry
/// and the static allowlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UseClassification {
    /// Use path resolves into the linker registry. The associated value
    /// is the registered absolute path (the longest matching prefix).
    Registered { absolute_path: String },

    /// Use path's crate root is on the static allowlist. No registry
    /// lookup needed; pass through silently.
    AllowedExternal,

    /// Use path is neither registered nor on the allowlist. Recorded in
    /// the import catalog for later error enrichment.
    Other,
}

/// Catalog mapping a name introduced into kernel scope to the source path
/// it came from, for use paths classified as [`UseClassification::Other`].
///
/// Built by `CUDATileModules::from_kernel` while walking the kernel's
/// use graph; consulted on name-resolution failures to produce a tailored
/// error.
pub type UseCatalog = HashMap<String, String>;

/// Classify a `use` path against the linker registry and the static
/// allowlist. Pure function; the registry is passed in.
pub fn classify_use(use_path: &str, registry: &HashMap<&str, fn() -> Module>) -> UseClassification {
    let head = use_path.split("::").next().unwrap_or("");
    if ALLOWED_EXTERNAL_PREFIXES.contains(&head) {
        return UseClassification::AllowedExternal;
    }
    if let Some(registered) = find_longest_registered_prefix(use_path, registry) {
        return UseClassification::Registered {
            absolute_path: registered.to_string(),
        };
    }
    UseClassification::Other
}

fn find_longest_registered_prefix<'a>(
    use_path: &str,
    registry: &HashMap<&'a str, fn() -> Module>,
) -> Option<&'a str> {
    let segments: Vec<&str> = use_path.split("::").collect();
    for end in (1..=segments.len()).rev() {
        let candidate = segments[..end].join("::");
        if let Some(key) = registry.keys().find(|k| **k == candidate) {
            return Some(*key);
        }
    }
    None
}

/// True if the path's crate root is one of the Rust stdlib crates.
/// Used only to soften the unresolved-name error with an extra hint.
pub fn is_stdlib_path(path: &str) -> bool {
    let head = path.split("::").next().unwrap_or("");
    STDLIB_PREFIXES.contains(&head)
}

/// Walk an `ItemMod`'s top-level `use` statements and collect (name, path)
/// pairs for each item brought into scope.
///
/// - `use foo::bar::Baz;` → one entry: `name = Some("Baz"), path = "foo::bar::Baz"`
/// - `use foo::bar::Baz as Quux;` → `name = Some("Quux"), path = "foo::bar::Baz"`
/// - `use foo::bar::*;` → `name = None, path = "foo::bar"`
/// - `use foo::bar::{Baz, Qux};` → two entries
pub fn collect_use_imports(item_mod: &ItemMod) -> Vec<UseImport> {
    let mut out = Vec::new();
    if let Some((_, items)) = &item_mod.content {
        for item in items {
            if let syn::Item::Use(use_item) = item {
                walk_use_tree(&use_item.tree, &[], &mut out);
            }
        }
    }
    out
}

fn walk_use_tree(tree: &UseTree, prefix: &[String], out: &mut Vec<UseImport>) {
    match tree {
        UseTree::Path(p) => {
            let mut new_prefix = prefix.to_vec();
            new_prefix.push(p.ident.to_string());
            walk_use_tree(&p.tree, &new_prefix, out);
        }
        UseTree::Name(n) => {
            let mut path = prefix.to_vec();
            let name = n.ident.to_string();
            path.push(name.clone());
            out.push(UseImport {
                name: Some(name),
                path: path.join("::"),
            });
        }
        UseTree::Rename(r) => {
            let mut path = prefix.to_vec();
            path.push(r.ident.to_string());
            out.push(UseImport {
                name: Some(r.rename.to_string()),
                path: path.join("::"),
            });
        }
        UseTree::Glob(_) => {
            out.push(UseImport {
                name: None,
                path: prefix.join("::"),
            });
        }
        UseTree::Group(g) => {
            for sub in &g.items {
                walk_use_tree(sub, prefix, out);
            }
        }
    }
}

/// Compose the unresolved-name error context message when the failing
/// name is in the catalog.
///
/// Returns `None` if the name is not catalogued, signalling the resolver
/// to fall through to its plain "name not found" path.
pub fn unresolved_name_hint(name: &str, catalog: &UseCatalog) -> Option<String> {
    let import_path = catalog.get(name)?;
    let stdlib_hint = if is_stdlib_path(import_path) {
        " Rust standard-library types have no GPU representation in cuTile kernels."
    } else {
        ""
    };
    Some(format!(
        "`{name}` was imported from `{import_path}`, which is not supported in \
         cuTile kernels.{stdlib_hint} Only types from `#[cutile::module]`-annotated \
         modules and the cuTile external allowlist (`half::*`) are available in \
         kernel bodies.",
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_uses_allowlist() {
        let registry: HashMap<&str, fn() -> Module> = HashMap::new();
        assert_eq!(
            classify_use("half::f16", &registry),
            UseClassification::AllowedExternal,
        );
        assert_eq!(
            classify_use("half::bf16", &registry),
            UseClassification::AllowedExternal,
        );
    }

    #[test]
    fn classify_other_for_unknown() {
        let registry: HashMap<&str, fn() -> Module> = HashMap::new();
        assert_eq!(
            classify_use("std::collections::HashMap", &registry),
            UseClassification::Other,
        );
        assert_eq!(
            classify_use("rayon::slice::ParallelSliceMut", &registry),
            UseClassification::Other,
        );
        assert_eq!(
            classify_use("crate::utils::compute", &registry),
            UseClassification::Other,
        );
    }

    #[test]
    fn classify_registered_finds_longest_prefix() {
        // Stub builder; we only care about path lookup, not the Module shape.
        fn dummy_build() -> Module {
            Module::new("dummy", syn::parse_quote! { pub mod dummy {} })
        }
        let mut registry: HashMap<&str, fn() -> Module> = HashMap::new();
        registry.insert("cutile::core", dummy_build);
        registry.insert("cutile::core::sub", dummy_build);

        assert_eq!(
            classify_use("cutile::core::Tile", &registry),
            UseClassification::Registered {
                absolute_path: "cutile::core".into()
            },
        );
        // Longest-prefix wins.
        assert_eq!(
            classify_use("cutile::core::sub::nested", &registry),
            UseClassification::Registered {
                absolute_path: "cutile::core::sub".into()
            },
        );
    }

    #[test]
    fn is_stdlib_path_recognises_stdlib() {
        assert!(is_stdlib_path("std::collections::HashMap"));
        assert!(is_stdlib_path("core::mem::size_of"));
        assert!(is_stdlib_path("alloc::vec::Vec"));
        assert!(!is_stdlib_path("half::f16"));
        assert!(!is_stdlib_path("crate::foo"));
        assert!(!is_stdlib_path("rayon::iter"));
    }

    #[test]
    fn collect_use_imports_handles_all_tree_shapes() {
        let m: ItemMod = syn::parse_quote! {
            mod m {
                use foo::Bar;
                use foo::Baz as Quux;
                use foo::nested::*;
                use foo::group::{One, Two as RenamedTwo};
            }
        };
        let imports = collect_use_imports(&m);
        let by_path: HashMap<String, Option<String>> =
            imports.into_iter().map(|i| (i.path, i.name)).collect();

        assert_eq!(by_path.get("foo::Bar"), Some(&Some("Bar".into())));
        assert_eq!(by_path.get("foo::Baz"), Some(&Some("Quux".into())));
        assert_eq!(by_path.get("foo::nested"), Some(&None));
        assert_eq!(by_path.get("foo::group::One"), Some(&Some("One".into())));
        assert_eq!(
            by_path.get("foo::group::Two"),
            Some(&Some("RenamedTwo".into()))
        );
    }

    #[test]
    fn unresolved_hint_appends_stdlib_note_for_std_paths() {
        let mut catalog: UseCatalog = HashMap::new();
        catalog.insert("HashMap".into(), "std::collections::HashMap".into());
        catalog.insert("compute".into(), "rayon::slice::compute".into());

        let std_hint = unresolved_name_hint("HashMap", &catalog).unwrap();
        assert!(std_hint.contains("std::collections::HashMap"));
        assert!(std_hint.contains("standard-library"));

        let other_hint = unresolved_name_hint("compute", &catalog).unwrap();
        assert!(other_hint.contains("rayon::slice::compute"));
        assert!(!other_hint.contains("standard-library"));

        assert!(unresolved_name_hint("not_in_catalog", &catalog).is_none());
    }
}
