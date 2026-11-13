/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for the LINKING-Phase-A/B linker registry. Verifies that
//! `#[cutile::module]` self-registers each module into the
//! `CUTILE_MODULES` distributed slice and that `CUDATileModules::from_kernel`
//! walks the kernel's `use` graph against the registry to discover deps.

use cutile_compiler::compiler::CUDATileModules;
use cutile_compiler::registry::{registry_digest, CutileModuleEntry, CUTILE_MODULES};

#[cutile::module]
mod registry_phase_a_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn noop_kernel<const S: [i32; 1]>(out: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(0.0f32, out.shape());
        out.store(tile);
    }
}

use registry_phase_a_module::__module_ast_self;

#[test]
fn registry_contains_phase_a_module() {
    let entries: Vec<&CutileModuleEntry> = CUTILE_MODULES.iter().collect();
    assert!(
        !entries.is_empty(),
        "CUTILE_MODULES should not be empty — at least the cutile core module \
         and the test's own module should self-register"
    );

    let test_path_suffix = "registry_phase_a_module";
    let test_entry = entries
        .iter()
        .find(|e| e.absolute_path.ends_with(test_path_suffix))
        .unwrap_or_else(|| {
            let paths: Vec<&str> = entries.iter().map(|e| e.absolute_path).collect();
            panic!(
                "CUTILE_MODULES missing this test's module (`{}`). Registered: {:?}",
                test_path_suffix, paths
            )
        });

    let module = (test_entry.source.build)();
    assert!(
        module.absolute_path().ends_with(test_path_suffix),
        "build closure should yield Module with absolute_path matching the entry; \
         entry={}, built={}",
        test_entry.absolute_path,
        module.absolute_path()
    );
}

#[test]
fn aliases_share_their_module_source_and_digest_covers_registry() {
    let entry = |path: &str| {
        CUTILE_MODULES
            .iter()
            .find(|e| e.absolute_path == path)
            .unwrap_or_else(|| panic!("`{path}` not registered"))
    };
    for (alias, canonical) in [
        ("cutile::core", "cutile::_core::core"),
        ("cutile::tileir", "cutile::_tileir::tileir"),
    ] {
        assert!(std::ptr::eq(entry(alias).source, entry(canonical).source));
    }
    let is_digest = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
    assert!(CUTILE_MODULES.iter().all(|e| is_digest(e.source.identity)));
    assert_ne!(
        entry("cutile::_core::core").source.identity,
        entry("cutile::_tileir::tileir").source.identity
    );
    let digest = registry_digest();
    assert!(is_digest(digest));
    assert_eq!(digest, registry_digest());
}

#[test]
fn modules_from_registry_loads_named_paths() {
    let our_path: &str = CUTILE_MODULES
        .iter()
        .find(|e| e.absolute_path.ends_with("registry_phase_a_module"))
        .expect("test's own module should be registered")
        .absolute_path;

    let modules = CUDATileModules::modules_from_registry(&[our_path])
        .expect("registry lookup should succeed for a registered path");

    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0].absolute_path(), our_path);
}

#[test]
fn from_registry_errors_on_missing_path() {
    let result = CUDATileModules::modules_from_registry(&["definitely_not_registered_module_xyz"]);
    assert!(
        result.is_err(),
        "lookup of unregistered path should fail; got Ok"
    );
}

#[test]
fn from_kernel_resolves_use_cutile_core_via_alias() {
    // `from_kernel` walks `use cutile::core::*` against the registry. The
    // public `cutile::core` path is a re-export of `cutile::_core::core` —
    // resolved via the manual alias entry registered in `cutile/src/lib.rs`
    // next to `pub use _core::core;`.
    let kernel = __module_ast_self();
    let modules = CUDATileModules::from_kernel(kernel)
        .expect("from_kernel should construct a CUDATileModules");

    let resolver_paths: std::collections::HashSet<&str> =
        modules.modules().keys().map(|s| s.as_str()).collect();
    assert!(
        resolver_paths.contains("registry_phase_a_module"),
        "kernel module should be present in the resolver; resolver has: {resolver_paths:?}",
    );
    assert!(
        resolver_paths.contains("core"),
        "cutile core should be resolved via the cutile::core alias entry; \
         resolver has: {resolver_paths:?}",
    );
}

#[test]
fn from_kernel_picks_up_canonical_path_use() {
    // Verify the use-walk works when the kernel imports a registered
    // module by its canonical `module_path!()` path. We construct a
    // synthetic kernel inline whose `use` references this test module's
    // own absolute_path.
    use syn::parse_quote;

    let our_path: &str = CUTILE_MODULES
        .iter()
        .find(|e| e.absolute_path.ends_with("registry_phase_a_module"))
        .expect("test's own module should be registered")
        .absolute_path;

    let our_path_segments: Vec<&str> = our_path.split("::").collect();
    let path_tokens: proc_macro2::TokenStream = our_path_segments
        .iter()
        .map(|s| syn::Ident::new(s, proc_macro2::Span::call_site()))
        .map(|id| quote::quote! { #id })
        .reduce(|a, b| quote::quote! { #a :: #b })
        .unwrap();

    let synthetic_kernel: syn::ItemMod = parse_quote! {
        pub mod synthetic_kernel {
            use #path_tokens ::*;
        }
    };

    let kernel = cutile_compiler::ast::Module::with_span_base(
        "synthetic_kernel",
        synthetic_kernel,
        cutile_compiler::ast::SpanBase::unknown(),
    );

    let modules = CUDATileModules::from_kernel(kernel)
        .expect("from_kernel should walk the synthetic kernel's `use` graph");

    let resolver_paths: std::collections::HashSet<&str> =
        modules.modules().keys().map(|s| s.as_str()).collect();
    assert!(
        resolver_paths.contains("registry_phase_a_module"),
        "from_kernel should pick up the canonical-path import; \
         resolver has: {resolver_paths:?}",
    );
}
