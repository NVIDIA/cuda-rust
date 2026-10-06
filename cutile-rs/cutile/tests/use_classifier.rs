/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end tests for the use-classifier import catalog.
//!
//! When `CUDATileModules::from_kernel` walks a kernel's `use` graph, names
//! brought into scope by use statements that point to unsupported sources
//! (stdlib, third-party crates, unannotated user modules) get recorded in
//! the import catalog. The JIT consults the catalog on name-resolution
//! failures so the user sees `X was imported from path::to::X, which is
//! not supported in cuTile kernels` instead of a bare `undefined function`
//! error.

use cutile_compiler::ast::{Module, SpanBase};
use cutile_compiler::compiler::CUDATileModules;
use syn::parse_quote;

#[cutile::module]
mod use_classifier_test_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn noop_kernel<const S: [i32; 1]>(out: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(0.0f32, out.shape());
        out.store(tile);
    }
}

use use_classifier_test_module::__module_ast_self;

/// Build a synthetic kernel module whose `use` statements come from various
/// unsupported sources, run it through `from_kernel`, and check the
/// resulting catalog.
fn make_synthetic_kernel(uses: syn::ItemMod) -> Module {
    Module::with_span_base("synthetic_kernel", uses, SpanBase::unknown())
}

#[test]
fn from_kernel_catalogs_stdlib_imports() {
    let kernel_ast: syn::ItemMod = parse_quote! {
        pub mod synthetic_kernel {
            use std::collections::HashMap;
            use cutile::core::*;
        }
    };
    let modules = CUDATileModules::from_kernel(make_synthetic_kernel(kernel_ast))
        .expect("from_kernel should construct CUDATileModules");

    let hint = modules
        .unresolved_name_hint("HashMap")
        .expect("HashMap should be in the catalog");
    assert!(
        hint.contains("std::collections::HashMap"),
        "hint should reference the import path; got: {hint}"
    );
    assert!(
        hint.contains("standard-library"),
        "stdlib imports should get the stdlib hint; got: {hint}"
    );
}

#[test]
fn from_kernel_catalogs_third_party_imports() {
    let kernel_ast: syn::ItemMod = parse_quote! {
        pub mod synthetic_kernel {
            use rayon::slice::ParallelSliceMut;
            use cutile::core::*;
        }
    };
    let modules = CUDATileModules::from_kernel(make_synthetic_kernel(kernel_ast))
        .expect("from_kernel should construct CUDATileModules");

    let hint = modules
        .unresolved_name_hint("ParallelSliceMut")
        .expect("ParallelSliceMut should be in the catalog");
    assert!(hint.contains("rayon::slice::ParallelSliceMut"));
    assert!(
        !hint.contains("standard-library"),
        "non-stdlib imports should NOT get the stdlib hint; got: {hint}"
    );
}

#[test]
fn from_kernel_catalogs_unannotated_user_module() {
    let kernel_ast: syn::ItemMod = parse_quote! {
        pub mod synthetic_kernel {
            use my_crate::utils::compute;
            use cutile::core::*;
        }
    };
    let modules = CUDATileModules::from_kernel(make_synthetic_kernel(kernel_ast))
        .expect("from_kernel should construct CUDATileModules");

    let hint = modules
        .unresolved_name_hint("compute")
        .expect("compute should be in the catalog");
    assert!(hint.contains("my_crate::utils::compute"));
}

#[test]
fn from_kernel_does_not_catalog_registered_imports() {
    // `cutile::core::*` resolves through the registry alias — the names
    // it brings in are JIT-resolvable, so the catalog stays empty for
    // them.
    let kernel = __module_ast_self();
    let modules = CUDATileModules::from_kernel(kernel).expect("from_kernel should succeed");
    assert!(
        modules.unresolved_name_hint("Tile").is_none(),
        "names from a registered cuTile module should not be catalogued"
    );
    assert!(
        modules.unresolved_name_hint("Tensor").is_none(),
        "names from a registered cuTile module should not be catalogued"
    );
}

#[test]
fn from_kernel_does_not_catalog_allowed_external_imports() {
    let kernel_ast: syn::ItemMod = parse_quote! {
        pub mod synthetic_kernel {
            use half::f16;
            use half::bf16;
            use cutile::core::*;
        }
    };
    let modules = CUDATileModules::from_kernel(make_synthetic_kernel(kernel_ast))
        .expect("from_kernel should construct CUDATileModules");
    assert!(
        modules.unresolved_name_hint("f16").is_none(),
        "names on the external allowlist should not be catalogued"
    );
    assert!(
        modules.unresolved_name_hint("bf16").is_none(),
        "names on the external allowlist should not be catalogued"
    );
}

#[test]
fn from_kernel_catalogs_renamed_imports_under_alias() {
    // `use foo::Bar as Baz;` — the catalog should be keyed on the alias
    // (the name actually visible in scope), not the original.
    let kernel_ast: syn::ItemMod = parse_quote! {
        pub mod synthetic_kernel {
            use std::collections::HashMap as Map;
            use cutile::core::*;
        }
    };
    let modules = CUDATileModules::from_kernel(make_synthetic_kernel(kernel_ast))
        .expect("from_kernel should construct CUDATileModules");

    assert!(
        modules.unresolved_name_hint("HashMap").is_none(),
        "the original name should NOT be in the catalog when renamed"
    );
    let hint = modules
        .unresolved_name_hint("Map")
        .expect("the alias should be catalogued");
    assert!(hint.contains("std::collections::HashMap"));
}

#[test]
fn from_kernel_handles_grouped_imports() {
    let kernel_ast: syn::ItemMod = parse_quote! {
        pub mod synthetic_kernel {
            use std::collections::{HashMap, BTreeMap};
            use cutile::core::*;
        }
    };
    let modules = CUDATileModules::from_kernel(make_synthetic_kernel(kernel_ast))
        .expect("from_kernel should construct CUDATileModules");

    assert!(modules.unresolved_name_hint("HashMap").is_some());
    assert!(modules.unresolved_name_hint("BTreeMap").is_some());
}

#[test]
fn from_kernel_does_not_catalog_dep_modules_imports() {
    // Transitive dep modules' use statements (e.g. `cutile::core`'s
    // `pub use super::atomic;` and similar internal re-exports) should NOT
    // populate the catalog — only the kernel's own use statements do.
    // Otherwise the catalog gets polluted with names like `atomic`,
    // `cmp_ordering`, etc. that legitimately resolve through other paths.
    let kernel = __module_ast_self();
    let modules =
        CUDATileModules::from_kernel(kernel).expect("from_kernel should construct CUDATileModules");

    // These names ARE imported (via `pub use super::*`) inside cutile's
    // _core.rs, but only as transitive-dep imports. They must not appear
    // in the catalog.
    for name in &[
        "atomic",
        "cmp_ordering",
        "ftz",
        "rounding",
        "scope",
        "ordering",
        "tma",
        "padding",
        "predicate",
        "overflow",
    ] {
        assert!(
            modules.unresolved_name_hint(name).is_none(),
            "transitive-dep import `{name}` should NOT be in the catalog",
        );
    }
}

#[test]
fn legacy_new_constructor_yields_empty_catalog() {
    // `CUDATileModules::new(Vec<Module>)` doesn't walk a kernel's use
    // graph, so the catalog is empty; the resolver falls through to the
    // bare error path. (Tests calling `new` predate the catalog and
    // shouldn't get hint enrichment.)
    let kernel = __module_ast_self();
    let modules = CUDATileModules::new(vec![kernel]).expect("new should succeed");
    assert!(modules.unresolved_name_hint("HashMap").is_none());
    assert!(modules.unresolved_name_hint("anything").is_none());
}
