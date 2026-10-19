/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke tests for `#[cutile::module]` submodule support.
//!
//! Verifies that a `mod inner { ... }` nested inside a `#[cutile::module]`
//! body is processed recursively by the macro: items inside the submodule
//! go through the same item walker, and the submodule namespace is
//! preserved in the rustc-emitted output.
//!
//! The JIT side picks up submodule bodies for free because
//! `__module_ast_self` captures the entire pre-expansion source text via
//! `Span::source_text()` — the submodule body is a substring of that text
//! and so is part of the parsed `syn::ItemMod` the JIT receives.

#[cutile::module]
mod submodule_smoke_module {
    use cutile::core::*;

    pub mod inner {
        pub trait Sealed {}

        pub struct Marker;
        impl Sealed for Marker {}

        pub const INNER_CONST: i32 = 7;

        pub fn helper(x: i32) -> i32 {
            x + 1
        }
    }

    #[cutile::entry()]
    fn noop_kernel<const S: [i32; 1]>(out: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(0.0f32, out.shape());
        out.store(tile);
    }
}

use submodule_smoke_module::__module_ast_self;

#[test]
fn submodule_items_are_visible_to_rustc() {
    // Items declared inside the submodule are reachable from outside.
    use submodule_smoke_module::inner;
    assert_eq!(inner::INNER_CONST, 7);
    assert_eq!(inner::helper(41), 42);
    fn assert_sealed<T: inner::Sealed>() {}
    assert_sealed::<inner::Marker>();
}

#[test]
fn submodule_body_appears_in_jit_ast() {
    // The JIT captures the entire pre-expansion source text. Confirm the
    // submodule's body is present in the parsed `syn::ItemMod` that
    // `__module_ast_self` builds.
    let module = __module_ast_self();
    let item: &syn::ItemMod = module.ast();
    let content = item
        .content
        .as_ref()
        .expect("module body must be inline at this point");

    let inner_mod = content
        .1
        .iter()
        .find_map(|item| match item {
            syn::Item::Mod(m) if m.ident == "inner" => Some(m),
            _ => None,
        })
        .expect("submodule `inner` should be present in the JIT-side AST");

    let inner_body = inner_mod
        .content
        .as_ref()
        .expect("submodule `inner` should have an inline body in captured source");

    let kinds: Vec<&'static str> = inner_body
        .1
        .iter()
        .map(|item| match item {
            syn::Item::Trait(_) => "trait",
            syn::Item::Struct(_) => "struct",
            syn::Item::Impl(_) => "impl",
            syn::Item::Const(_) => "const",
            syn::Item::Fn(_) => "fn",
            _ => "other",
        })
        .collect();

    assert!(
        kinds.contains(&"trait") && kinds.contains(&"const") && kinds.contains(&"fn"),
        "expected trait / const / fn in inner submodule, got {kinds:?}"
    );
}

#[test]
fn submodule_items_flatten_for_jit_name_resolver() {
    // The JIT's name resolver flattens nested submodules into the parent
    // module's namespace, so a fn declared in `mod inner { … }` resolves
    // as if it were at the outer module's top level.
    use cutile_compiler::compiler::CUDATileModules;
    use cutile_compiler::passes::name_resolution::{NameResolver, Res};

    let modules = CUDATileModules::from_kernel(__module_ast_self())
        .expect("from_kernel should construct CUDATileModules");

    let asts: Vec<(String, syn::ItemMod)> = modules
        .modules()
        .iter()
        .map(|(name, m)| (name.clone(), m.clone()))
        .collect();

    let resolver = NameResolver::build(&asts).expect("name resolver build");

    let helper_path: syn::Path = syn::parse_quote!(helper);
    let res = resolver.resolve_path(&helper_path, "submodule_smoke_module");
    assert!(
        matches!(res, Res::Def(_, _)),
        "fn `helper` declared in `mod inner` should be resolvable as if it \
         were at the parent's top level (flat namespace); got {res:?}"
    );

    let marker_path: syn::Path = syn::parse_quote!(Marker);
    let res = resolver.resolve_path(&marker_path, "submodule_smoke_module");
    assert!(
        matches!(res, Res::Def(_, _)),
        "struct `Marker` from `mod inner` should also be flattened; \
         got {res:?}"
    );
}
