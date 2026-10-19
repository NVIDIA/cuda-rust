/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module registry: thin wrapper around [`NameResolver`] that preserves
//! the compiler's existing lookup API.
//!
//! All item storage and name resolution lives in the [`NameResolver`]
//! (Pass 1 output). This struct provides backward-compatible accessors
//! so the rest of the compiler doesn't need to change yet.

use crate::ast::{Module, SourceLocation, SpanBase};
use crate::error::{JITError, SpannedJITError};
use crate::generics::{GenericVars, TypeInstance};
use crate::kernel_naming::KernelNaming;
use crate::passes::name_resolution::NameResolver;
use crate::syn_utils::*;
use quote::ToTokens;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use syn::spanned::Spanned;
use syn::{
    Expr, ExprMethodCall, GenericArgument, GenericParam, ImplItem, ImplItemFn, ItemConst, ItemFn,
    ItemImpl, ItemMod, ItemStatic, ItemStruct, ItemTrait, ItemType, PathArguments, Type,
};

/// Aggregated index of all DSL modules, types, impls, and functions.
///
/// Internally delegates to [`NameResolver`] for all item storage and
/// name resolution. The flat accessors (`primitives`, `structs`, etc.)
/// are backward-compatible shims that will be removed as the compiler
/// migrates to path-based resolution.
pub struct CUDATileModules {
    /// The name resolver (Pass 1 output). Owns all items and resolves paths.
    pub(crate) name_resolver: NameResolver,

    /// Span bases for source location mapping.
    pub(crate) span_bases: HashMap<String, SpanBase>,

    /// Catalog of names brought into kernel scope by `use` statements
    /// written in the kernel module itself — i.e. the module passed to
    /// [`Self::from_kernel`] — and classified as
    /// [`crate::use_classifier::UseClassification::Other`] (not registered,
    /// not on the static external allowlist). Maps
    /// `imported_name → import_path`. Empty when the legacy `new`
    /// constructor is used. Consulted by name-resolution failure paths to
    /// produce a tailored error pointing back to the use statement.
    ///
    /// Use statements in transitively-loaded dep modules (registered
    /// cuTile modules walked through the use graph) are *not* catalogued.
    /// Deps are trusted infrastructure; their internal imports are
    /// implementation details, not user-facing.
    pub(crate) use_catalog: crate::use_classifier::UseCatalog,

    /// The module passed to [`Self::from_kernel`], whose scope resolves the
    /// consts named in kernel-level types (parameter shapes, `static`
    /// globals). `None` for the legacy `new` constructor.
    pub(crate) kernel_module: Option<String>,
}

// ---------------------------------------------------------------------------
// Backward-compatible field-like accessors
// ---------------------------------------------------------------------------

impl CUDATileModules {
    /// Flat map of all modules: name → ItemMod.
    /// Compatibility shim — prefer `name_resolver.module(name)`.
    pub fn modules(&self) -> &HashMap<String, ItemMod> {
        self.name_resolver.all_modules()
    }

    /// Flat map of all primitives: (trait, type) → ItemImpl.
    /// Compatibility shim — passed to generics/types functions.
    pub fn primitives(&self) -> &HashMap<(String, String), ItemImpl> {
        self.name_resolver.primitives()
    }

    /// Flat map of all structs: name → ItemStruct.
    /// Compatibility shim.
    pub fn structs(&self) -> &HashMap<String, ItemStruct> {
        self.name_resolver.structs()
    }

    /// Flat map of all traits: name → ItemTrait.
    /// Compatibility shim.
    pub fn traits(&self) -> &HashMap<String, ItemTrait> {
        self.name_resolver.traits()
    }

    /// Flat map of all type aliases: name → ItemType.
    /// Compatibility shim.
    pub fn type_aliases(&self) -> &HashMap<String, ItemType> {
        self.name_resolver.type_aliases()
    }

    /// Flat map of all constants: name → ItemConst. Last definition wins for
    /// a name several modules define; anything that resolves a const for a
    /// particular module must use [`Self::consts_visible_from`]. Kept for the
    /// legacy `new` constructor, which has no kernel module to scope by.
    pub fn consts(&self) -> &HashMap<String, ItemConst> {
        self.name_resolver.consts()
    }

    /// The constants visible from `module`, by name: its own definitions,
    /// then its imports, then core, then a unique (or unanimous) global
    /// definition. See [`NameResolver::resolve_const`].
    pub fn consts_visible_from(&self, module: &str) -> Rc<HashMap<String, ItemConst>> {
        self.name_resolver.consts_visible_from(module)
    }

    /// Flat map of all static items: name → ItemStatic.
    /// Compatibility shim.
    pub fn statics(&self) -> &HashMap<String, ItemStatic> {
        self.name_resolver.statics()
    }

    /// Flat map of all struct impls: struct_name → [(module, ItemImpl)].
    /// Compatibility shim.
    pub fn struct_impls(&self) -> &HashMap<String, Vec<(String, ItemImpl)>> {
        self.name_resolver.struct_impls()
    }

    /// Flat map of all trait impls: (trait, self_ty) → [(module, ItemImpl)].
    /// Compatibility shim.
    pub fn trait_impls(&self) -> &HashMap<(String, String), Vec<(String, ItemImpl)>> {
        self.name_resolver.trait_impls()
    }

    /// Flat map of all functions: name → (module, ItemFn).
    /// Compatibility shim.
    pub fn functions(&self) -> &HashMap<String, (String, ItemFn)> {
        self.name_resolver.functions()
    }

    /// Expand DSL-visible type aliases and const paths in a Rust type, with
    /// consts resolved from the kernel module's scope (see
    /// [`Self::normalize_type_aliases_in`]). Modules built with the legacy
    /// `new` constructor have no kernel module and fall back to the flat map.
    pub fn normalize_type_aliases(&self, ty: &Type) -> Result<Type, JITError> {
        self.normalize_type_aliases_in(ty, self.kernel_module.as_deref())
    }

    /// Expand DSL-visible type aliases in a Rust type, then replace paths to
    /// module-level consts (`Tensor<f32, {[N]}>`) with the definition visible
    /// from `module` — the module whose source spelled the type.
    pub fn normalize_type_aliases_in(
        &self,
        ty: &Type,
        module: Option<&str>,
    ) -> Result<Type, JITError> {
        let normalize_error = |msg: String| {
            SourceLocation::unknown().jit_error(&format!(
                "failed to normalize type `{}`: {msg}",
                ty.to_token_stream()
            ))
        };
        let normalized = crate::type_aliases::normalize_type_aliases(ty, self.type_aliases())
            .map_err(normalize_error)?;
        let scoped;
        let consts: &HashMap<String, ItemConst> = match module {
            Some(module) => {
                scoped = self.consts_visible_from(module);
                &scoped
            }
            None => self.consts(),
        };
        crate::type_aliases::normalize_const_paths_in_type(&normalized, consts)
            .map_err(normalize_error)
    }

    /// Return the import-catalog hint message for `name` if the name was
    /// brought into kernel scope by a `use` statement that classified as
    /// [`crate::use_classifier::UseClassification::Other`]. Returns
    /// `None` if the name is not catalogued (e.g. when the kernel was
    /// constructed via the legacy `new` constructor that doesn't walk
    /// the use graph, or when the name was supposed to come from the
    /// kernel's own items / a registered module / a stdlib alias).
    ///
    /// Caller should append the returned string to its primary error
    /// message — the hint is suffix-style ("X was imported from ...").
    pub fn unresolved_name_hint(&self, name: &str) -> Option<String> {
        crate::use_classifier::unresolved_name_hint(name, &self.use_catalog)
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl CUDATileModules {
    pub fn new(modules_vec: Vec<Module>) -> Result<Self, JITError> {
        // Collect module ASTs and span bases.
        let mut module_asts: Vec<(String, ItemMod)> = Vec::new();
        let mut span_bases: HashMap<String, SpanBase> = HashMap::new();

        for module in &modules_vec {
            let module_name = module.name().to_string();
            let module_ast = module.ast();

            if module_ast.content.is_none() {
                return module
                    .resolve_span(&module_ast.span())
                    .jit_error_result(&format!(
                        "module `{module_name}` must have a body (non-empty content)"
                    ));
            }

            module_asts.push((module_name.clone(), module_ast.clone()));
            let span_base = module.span_base().clone();
            span_bases.insert(module_name.clone(), span_base.clone());
            let absolute_path = module.absolute_path();
            if absolute_path != module_name {
                span_bases.insert(absolute_path.to_string(), span_base);
            }
        }

        // Pass 1: Build the name resolver (indexes all items, processes imports).
        let name_resolver = NameResolver::build(&module_asts)?;

        Ok(CUDATileModules {
            name_resolver,
            span_bases,
            // Legacy `new` doesn't have a single kernel module to walk;
            // import-catalog enrichment and const scoping are unavailable
            // for callers on this path. New code should use `from_kernel`.
            use_catalog: HashMap::new(),
            kernel_module: None,
        })
    }

    /// Build a `Vec<Module>` by looking up each absolute path in the
    /// linker-registry [`crate::registry::CUTILE_MODULES`] and invoking
    /// its `build` closure.
    ///
    /// LINKING Phase A entry point: lets callers exercise the registry
    /// path while [`CUDATileModules::new`] still consumes the legacy
    /// chained `_module_asts()` output. Phase B uses [`Self::from_kernel`]
    /// for use-graph-driven dep discovery; this fn is retained for
    /// path-list-driven test scenarios.
    ///
    /// Returns an error listing all registered paths if any requested
    /// path is missing.
    pub fn modules_from_registry(paths: &[&str]) -> Result<Vec<Module>, JITError> {
        use crate::registry::CUTILE_MODULES;
        let registry: HashMap<&str, fn() -> Module> = CUTILE_MODULES
            .iter()
            .map(|e| (e.absolute_path, e.build))
            .collect();
        paths
            .iter()
            .map(|path| {
                registry
                    .get(*path)
                    .copied()
                    .map(|build| build())
                    .ok_or_else(|| {
                        let mut registered: Vec<&&str> = registry.keys().collect();
                        registered.sort();
                        SourceLocation::unknown().jit_error(&format!(
                            "module `{path}` not found in linker registry; \
                         registered modules: {registered:?}",
                        ))
                    })
            })
            .collect()
    }

    /// Build a [`CUDATileModules`] from a kernel [`Module`] by walking its
    /// `use` statements iteratively against the linker-registry.
    ///
    /// LINKING Phase B entry point. Replaces the old chained
    /// `_module_asts()` traversal: instead of the macro emitting calls to
    /// every `_module_asts()` of every declared dep, the JIT walks the
    /// kernel's `use` graph at JIT time and consults the global
    /// [`crate::registry::CUTILE_MODULES`] registry for each registered
    /// path. Unregistered paths (`std::`, `half::*`, plain Rust submodules,
    /// etc.) are skipped silently.
    ///
    /// Algorithm:
    /// 1. Seed the working set with `kernel`.
    /// 2. Collect all `use` paths from `kernel`'s top-level items.
    /// 3. For each path, find the longest prefix registered in the slice;
    ///    if not yet visited, build that module's AST and recurse on its
    ///    own `use` paths.
    /// 4. Iterate until the working set stabilizes; build a [`NameResolver`]
    ///    over `kernel + all collected deps`.
    pub fn from_kernel(kernel: Module) -> Result<Self, JITError> {
        use crate::registry::CUTILE_MODULES;
        use crate::use_classifier::{
            classify_use, collect_use_imports, UseCatalog, UseClassification,
        };

        let registry: HashMap<&str, fn() -> Module> = CUTILE_MODULES
            .iter()
            .map(|e| (e.absolute_path, e.build))
            .collect();

        let mut working_set: Vec<Module> = vec![kernel];
        let kernel_path = working_set[0].absolute_path().to_string();
        let kernel_module = working_set[0].name().to_string();
        // `registry_visited` skips redundant work when the same registry
        // key is pulled in from multiple use statements. `module_visited`
        // dedupes by the built module's actual absolute_path so two
        // registry aliases pointing at the same builder (e.g.
        // `cutile::core` and `cutile::_core::core`) only add one Module.
        let mut registry_visited: HashSet<String> = HashSet::new();
        let mut module_visited: HashSet<String> = working_set
            .iter()
            .map(|m| m.absolute_path().to_string())
            .collect();
        let mut use_catalog: UseCatalog = HashMap::new();

        // Each queue entry pairs a (name, use-path) import with the
        // absolute path of the module it appeared in, so relative paths can
        // be resolved against the right module before registry lookup.
        struct Pending {
            name: Option<String>,
            path: String,
            owning_module: String,
        }
        let mut queue: Vec<Pending> = collect_use_imports(working_set[0].ast())
            .into_iter()
            .map(|imp| Pending {
                name: imp.name,
                path: imp.path,
                owning_module: kernel_path.clone(),
            })
            .collect();

        while let Some(pending) = queue.pop() {
            let resolved = resolve_relative_use_path(&pending.path, &pending.owning_module);
            // `appears_in_kernel_module` is true when the use statement
            // we're processing was written in the kernel module itself —
            // i.e. `pending.owning_module` is the same module that was
            // passed to `from_kernel`. False for uses pulled in from
            // transitive deps (registered cuTile modules walked recursively).
            //
            // We catalog only kernel-module uses. Deps are trusted
            // infrastructure; their internal `pub use super::atomic;`-style
            // imports are implementation details that would otherwise
            // pollute the catalog with false-positive entries that never
            // get queried by name resolution.
            let appears_in_kernel_module = pending.owning_module == kernel_path;
            match classify_use(&resolved, &registry) {
                UseClassification::Registered { absolute_path } => {
                    if !registry_visited.insert(absolute_path.clone()) {
                        continue;
                    }
                    let build_fn = registry[absolute_path.as_str()];
                    let module = build_fn();
                    let owning = module.absolute_path().to_string();
                    if !module_visited.insert(owning.clone()) {
                        continue;
                    }
                    queue.extend(collect_use_imports(module.ast()).into_iter().map(|imp| {
                        Pending {
                            name: imp.name,
                            path: imp.path,
                            owning_module: owning.clone(),
                        }
                    }));
                    working_set.push(module);
                }
                UseClassification::AllowedExternal => {
                    // Pass through; trusted external (e.g. `half::f16`).
                }
                UseClassification::Other => {
                    // Glob imports (`name = None`) are skipped — we wouldn't
                    // know which name to key on.
                    if appears_in_kernel_module {
                        if let Some(name) = pending.name {
                            // Record the FIRST seen path for a name; later
                            // duplicates don't overwrite the more useful entry.
                            use_catalog.entry(name).or_insert(pending.path);
                        }
                    }
                }
            }
        }

        let mut modules = Self::new(working_set)?;
        modules.use_catalog = use_catalog;
        modules.kernel_module = Some(kernel_module);
        Ok(modules)
    }
}

/// Resolve leading `crate`, `self`, and `super` segments in a `use` path
/// against the absolute path of the module containing that use.
fn resolve_relative_use_path(use_path: &str, owning_module: &str) -> String {
    let mut path_segments = use_path.split("::");
    let Some(first) = path_segments.next() else {
        return use_path.to_string();
    };

    match first {
        "crate" => {
            let Some(crate_root) = owning_module.split("::").next() else {
                return use_path.to_string();
            };
            join_segments([crate_root].into_iter().chain(path_segments))
        }
        "self" => join_segments(owning_module.split("::").chain(path_segments)),
        "super" => {
            let mut base = owning_module.split("::").collect::<Vec<_>>();
            base.pop();
            while let Some("super") = path_segments.clone().next() {
                path_segments.next();
                if base.pop().is_none() {
                    return use_path.to_string();
                }
            }
            join_segments(base.into_iter().chain(path_segments))
        }
        _ => use_path.to_string(),
    }
}

fn join_segments<'a>(segments: impl Iterator<Item = &'a str>) -> String {
    segments
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("::")
}

// ---------------------------------------------------------------------------
// Span / location utilities
// ---------------------------------------------------------------------------

impl CUDATileModules {
    pub fn get_span_base(&self, module_name: &str) -> Option<&SpanBase> {
        self.span_bases.get(module_name)
    }

    pub fn resolve_span(&self, module_name: &str, span: &proc_macro2::Span) -> SourceLocation {
        match self.span_bases.get(module_name) {
            Some(base) => base.resolve_span(span),
            None => SourceLocation::unknown(),
        }
    }

    pub fn get_source_file(&self, module_name: &str) -> Option<&str> {
        self.span_bases.get(module_name).and_then(|sb| {
            if sb.file.is_empty() {
                None
            } else {
                Some(sb.file.as_str())
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Item lookup (backward-compatible methods)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TraitMatchCtx {
    type_params: HashSet<String>,
    const_array_params: HashMap<String, Option<usize>>,
    const_scalar_params: HashSet<String>,
    caller_array_params: HashMap<String, Vec<i32>>,
    type_bindings: HashMap<String, String>,
    array_bindings: HashMap<String, Vec<i32>>,
    const_bindings: HashMap<String, i32>,
}

fn find_impl_method<'a>(item_impl: &'a ItemImpl, method_name: &str) -> Option<&'a ImplItemFn> {
    item_impl.items.iter().find_map(|item| match item {
        ImplItem::Fn(method) if method.sig.ident == method_name => Some(method),
        _ => None,
    })
}

/// Returns `false` only when an impl provably cannot apply to the receiver:
/// an impl const generic declared with a literal array rank (`const D: [i32; 2]`)
/// appears in the impl self type at a position where the receiver's resolved
/// const-generic array has a different rank. Variadic impls (`[i32; N]`) and
/// anything unresolved stay compatible, preserving the permissive name-based
/// lookup. This disambiguates same-named methods defined per rank, like
/// `PartitionIndex::components` for rank 2 and rank 3.
pub(crate) fn impl_self_type_rank_matches(
    item_impl: &ItemImpl,
    receiver_ty: &Type,
    generic_vars: &GenericVars,
) -> bool {
    let mut const_ranks: HashMap<String, usize> = HashMap::new();
    for param in &item_impl.generics.params {
        if let syn::GenericParam::Const(const_param) = param {
            if let Type::Array(type_array) = &const_param.ty {
                if let syn::Expr::Lit(len_lit) = &type_array.len {
                    if let syn::Lit::Int(len_int) = &len_lit.lit {
                        if let Ok(rank) = len_int.base10_parse::<usize>() {
                            const_ranks.insert(const_param.ident.to_string(), rank);
                        }
                    }
                }
            }
        }
    }
    if const_ranks.is_empty() {
        return true;
    }
    let Some(impl_args) = maybe_generic_args(&item_impl.self_ty) else {
        return true;
    };
    let Some(receiver_args) = maybe_generic_args(receiver_ty) else {
        return true;
    };
    for (impl_arg, receiver_arg) in impl_args.args.iter().zip(receiver_args.args.iter()) {
        let param_ident = match impl_arg {
            syn::GenericArgument::Const(syn::Expr::Path(path)) => path.path.get_ident(),
            syn::GenericArgument::Type(Type::Path(type_path)) => type_path.path.get_ident(),
            _ => None,
        };
        let Some(param_ident) = param_ident else {
            continue;
        };
        let Some(&expected_rank) = const_ranks.get(&param_ident.to_string()) else {
            continue;
        };
        let Some(receiver_cga) =
            crate::generics::get_cga_from_generic_argument(receiver_arg, generic_vars)
        else {
            continue;
        };
        if receiver_cga.len() != expected_rank {
            return false;
        }
    }
    true
}

pub(crate) fn instantiate_type_for_lookup(
    ty: &Type,
    generic_vars: &GenericVars,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Type {
    generic_vars
        .instantiate_type(ty, primitives)
        .map(|inst| inst.get_instantiated_type().clone())
        .unwrap_or_else(|_| ty.clone())
}

fn trait_impl_matches_call(
    item_impl: &ItemImpl,
    impl_method: &ImplItemFn,
    call_arg_rust_tys: &[Type],
    generic_vars: &GenericVars,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> bool {
    let self_ty = &*item_impl.self_ty;
    let (param_types, _return_type) = get_sig_types(&impl_method.sig, Some(self_ty));
    if param_types.len() != call_arg_rust_tys.len() {
        return false;
    }

    let mut ctx = TraitMatchCtx {
        caller_array_params: generic_vars.inst_array.clone(),
        ..Default::default()
    };
    collect_generics_for_trait_match(&item_impl.generics, &mut ctx);
    collect_generics_for_trait_match(&impl_method.sig.generics, &mut ctx);

    for (param_ty, call_arg_ty) in param_types.iter().zip(call_arg_rust_tys) {
        let actual_ty = instantiate_type_for_lookup(call_arg_ty, generic_vars, primitives);
        if !unify_trait_type(param_ty, &actual_ty, &mut ctx) {
            return false;
        }
    }

    true
}

fn collect_generics_for_trait_match(generics: &syn::Generics, ctx: &mut TraitMatchCtx) {
    for param in &generics.params {
        match param {
            GenericParam::Type(type_param) => {
                ctx.type_params.insert(type_param.ident.to_string());
            }
            GenericParam::Const(const_param) => match &const_param.ty {
                Type::Array(array_ty) => {
                    ctx.const_array_params
                        .insert(const_param.ident.to_string(), const_array_len(array_ty));
                }
                _ => {
                    ctx.const_scalar_params
                        .insert(const_param.ident.to_string());
                }
            },
            GenericParam::Lifetime(_) => {}
        }
    }
}

fn const_array_len(array_ty: &syn::TypeArray) -> Option<usize> {
    expr_i32(&array_ty.len).map(|len| len as usize)
}

fn strip_reference_ty(ty: &Type) -> &Type {
    match ty {
        Type::Reference(type_ref) => strip_reference_ty(&type_ref.elem),
        _ => ty,
    }
}

fn unify_trait_type(pattern: &Type, actual: &Type, ctx: &mut TraitMatchCtx) -> bool {
    let pattern = strip_reference_ty(pattern);
    let actual = strip_reference_ty(actual);

    if let Type::Path(pattern_path) = pattern {
        let pattern_segment = pattern_path.path.segments.last().unwrap();
        let pattern_ident = pattern_segment.ident.to_string();
        if ctx.type_params.contains(&pattern_ident)
            && matches!(pattern_segment.arguments, PathArguments::None)
        {
            return bind_type_param(ctx, &pattern_ident, actual);
        }
    }

    let (Type::Path(pattern_path), Type::Path(actual_path)) = (pattern, actual) else {
        return pattern.to_token_stream().to_string() == actual.to_token_stream().to_string();
    };

    let pattern_segment = pattern_path.path.segments.last().unwrap();
    let actual_segment = actual_path.path.segments.last().unwrap();
    if pattern_segment.ident != actual_segment.ident {
        return false;
    }

    match (&pattern_segment.arguments, &actual_segment.arguments) {
        (PathArguments::None, PathArguments::None) => true,
        (
            PathArguments::AngleBracketed(pattern_args),
            PathArguments::AngleBracketed(actual_args),
        ) => {
            let mut pattern_args = pattern_args.clone();
            let mut actual_args = actual_args.clone();
            strip_generic_args_lifetimes(&mut pattern_args);
            strip_generic_args_lifetimes(&mut actual_args);
            if pattern_args.args.len() != actual_args.args.len() {
                return false;
            }
            pattern_args.args.iter().zip(actual_args.args.iter()).all(
                |(pattern_arg, actual_arg)| unify_trait_generic_arg(pattern_arg, actual_arg, ctx),
            )
        }
        _ => false,
    }
}

fn unify_trait_generic_arg(
    pattern_arg: &GenericArgument,
    actual_arg: &GenericArgument,
    ctx: &mut TraitMatchCtx,
) -> bool {
    match (pattern_arg, actual_arg) {
        (GenericArgument::Type(pattern_ty), GenericArgument::Type(actual_ty)) => {
            if let Some(param_name) = const_array_param_type_name(pattern_ty, ctx) {
                let Some(actual_shape) = const_array_shape_from_type(actual_ty, ctx) else {
                    return false;
                };
                return bind_array_param_checked(ctx, &param_name, actual_shape);
            }
            unify_trait_type(pattern_ty, actual_ty, ctx)
        }
        (GenericArgument::Type(pattern_ty), GenericArgument::Const(actual_expr)) => {
            let Some(param_name) = const_array_param_type_name(pattern_ty, ctx) else {
                return false;
            };
            let Some(actual_shape) = const_shape(actual_expr) else {
                return false;
            };
            bind_array_param_checked(ctx, &param_name, actual_shape)
        }
        (GenericArgument::Const(pattern_expr), GenericArgument::Type(actual_ty)) => {
            let Some(actual_shape) = const_array_shape_from_type(actual_ty, ctx) else {
                return false;
            };
            unify_shape_pattern(pattern_expr, &actual_shape, ctx)
        }
        (GenericArgument::Const(pattern_expr), GenericArgument::Const(actual_expr)) => {
            unify_trait_const(pattern_expr, actual_expr, ctx)
        }
        (GenericArgument::Lifetime(_), GenericArgument::Lifetime(_)) => true,
        _ => false,
    }
}

fn const_array_param_type_name(ty: &Type, ctx: &TraitMatchCtx) -> Option<String> {
    let Type::Path(type_path) = strip_reference_ty(ty) else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    let ident = segment.ident.to_string();
    ctx.const_array_params.contains_key(&ident).then_some(ident)
}

fn const_array_shape_from_type(ty: &Type, ctx: &TraitMatchCtx) -> Option<Vec<i32>> {
    let Type::Path(type_path) = strip_reference_ty(ty) else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    let ident = segment.ident.to_string();
    ctx.array_bindings
        .get(&ident)
        .cloned()
        .or_else(|| ctx.caller_array_params.get(&ident).cloned())
}

fn bind_type_param(ctx: &mut TraitMatchCtx, name: &str, actual: &Type) -> bool {
    let actual = actual.to_token_stream().to_string();
    match ctx.type_bindings.get(name) {
        Some(bound) => bound == &actual,
        None => {
            ctx.type_bindings.insert(name.to_string(), actual);
            true
        }
    }
}

fn unify_trait_const(pattern: &Expr, actual: &Expr, ctx: &mut TraitMatchCtx) -> bool {
    if let Expr::Path(pattern_path) = pattern {
        let ident = get_ident_from_path_expr(pattern_path).to_string();
        if let Some(expected_len) = ctx.const_array_params.get(&ident).copied().flatten() {
            let Some(actual_shape) = const_shape(actual) else {
                return false;
            };
            if actual_shape.len() != expected_len {
                return false;
            }
            return bind_array_param_checked(ctx, &ident, actual_shape);
        }
        if ctx.const_array_params.contains_key(&ident) {
            let Some(actual_shape) = const_shape(actual) else {
                return false;
            };
            return bind_array_param_checked(ctx, &ident, actual_shape);
        }
    }

    let Some(actual_shape) = const_shape(actual) else {
        return pattern.to_token_stream().to_string() == actual.to_token_stream().to_string();
    };
    unify_shape_pattern(pattern, &actual_shape, ctx)
}

fn bind_array_param(ctx: &mut TraitMatchCtx, name: &str, actual: Vec<i32>) -> bool {
    match ctx.array_bindings.get(name) {
        Some(bound) => bound == &actual,
        None => {
            ctx.array_bindings.insert(name.to_string(), actual);
            true
        }
    }
}

fn bind_array_param_checked(ctx: &mut TraitMatchCtx, name: &str, actual: Vec<i32>) -> bool {
    if let Some(expected_len) = ctx.const_array_params.get(name).copied().flatten() {
        if actual.len() != expected_len {
            return false;
        }
    }
    bind_array_param(ctx, name, actual)
}

fn unify_shape_pattern(pattern: &Expr, actual_shape: &[i32], ctx: &mut TraitMatchCtx) -> bool {
    let Some(pattern_elems) = const_shape_elements(pattern) else {
        return false;
    };
    if pattern_elems.len() != actual_shape.len() {
        return false;
    }

    pattern_elems
        .iter()
        .zip(actual_shape.iter())
        .all(|(pattern_elem, actual_dim)| match expr_i32(pattern_elem) {
            Some(pattern_dim) => pattern_dim == *actual_dim,
            None => {
                let Expr::Path(path) = pattern_elem else {
                    return false;
                };
                let ident = get_ident_from_path_expr(path).to_string();
                if !ctx.const_scalar_params.contains(&ident) {
                    return false;
                }
                match ctx.const_bindings.get(&ident) {
                    Some(bound) => *bound == *actual_dim,
                    None => {
                        ctx.const_bindings.insert(ident, *actual_dim);
                        true
                    }
                }
            }
        })
}

fn const_shape(expr: &Expr) -> Option<Vec<i32>> {
    const_shape_elements(expr).and_then(|elems| elems.iter().map(expr_i32).collect())
}

fn const_shape_elements(expr: &Expr) -> Option<Vec<Expr>> {
    match expr {
        Expr::Block(block_expr) => match block_expr.block.stmts.as_slice() {
            [syn::Stmt::Expr(inner, _)] => const_shape_elements(inner),
            _ => None,
        },
        Expr::Array(array_expr) => Some(array_expr.elems.iter().cloned().collect()),
        Expr::Repeat(repeat_expr) => {
            let elem = *repeat_expr.expr.clone();
            let len = expr_i32(&repeat_expr.len)? as usize;
            Some(vec![elem; len])
        }
        _ => None,
    }
}

fn expr_i32(expr: &Expr) -> Option<i32> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int_lit) => int_lit.base10_parse::<i32>().ok(),
            _ => None,
        },
        Expr::Unary(unary) => {
            if !matches!(unary.op, syn::UnOp::Neg(_)) {
                return None;
            }
            expr_i32(&unary.expr).map(|value| -value)
        }
        Expr::Paren(paren) => expr_i32(&paren.expr),
        _ => None,
    }
}

impl CUDATileModules {
    pub fn get_primitives_attrs(
        &self,
        trait_name: &str,
        rust_type_name: &str,
    ) -> Option<SingleMetaList> {
        self.name_resolver
            .get_primitive_attrs(trait_name, rust_type_name)
    }

    pub fn get_cuda_tile_type_attrs(&self, ident: &str) -> Option<SingleMetaList> {
        self.name_resolver.get_type_attrs(ident)
    }

    pub fn get_function_by_name(&self, function_name: &str) -> Option<&(String, ItemFn)> {
        let canonical_name = KernelNaming::canonical_public_name(function_name);
        self.name_resolver.functions().get(canonical_name.as_str())
    }

    pub fn get_cuda_tile_op_attrs(&self, ident: &str) -> Option<SingleMetaList> {
        self.name_resolver.get_op_attrs(ident)
    }

    pub fn get_fn_item(
        &self,
        module_name: &str,
        function_name: &str,
    ) -> Result<&(String, ItemFn), JITError> {
        if !self.name_resolver.has_module(module_name) {
            return JITError::generic(&format!("undefined module: `{module_name}`"));
        }
        match self.get_function_by_name(function_name) {
            Some(function) => Ok(function),
            None => JITError::generic(&format!("undefined function: `{function_name}`")),
        }
    }

    pub fn get_fn_entry_attrs(&self, fn_item: &ItemFn) -> Result<SingleMetaList, JITError> {
        let entry_attrs = get_meta_list_by_last_segment("entry", &fn_item.attrs);
        let Some(entry_attrs) = entry_attrs else {
            return JITError::generic("function is missing a required `#[entry(...)]` attribute");
        };
        Ok(entry_attrs)
    }

    pub fn get_entry_arg_bool_by_function_name(
        &self,
        module_name: &str,
        function_name: &str,
        name: &str,
    ) -> Result<bool, JITError> {
        let (_, fn_item) = self.get_fn_item(module_name, function_name)?;
        let entry_attrs = self.get_fn_entry_attrs(fn_item)?;
        Ok(entry_attrs.parse_bool(name).unwrap_or(false))
    }

    pub fn get_entry_arg_string_by_function_name(
        &self,
        module_name: &str,
        function_name: &str,
        name: &str,
    ) -> Result<Option<String>, JITError> {
        let (_, fn_item) = self.get_fn_item(module_name, function_name)?;
        let entry_attrs = self.get_fn_entry_attrs(fn_item)?;
        Ok(entry_attrs.parse_string(name))
    }

    pub fn get_impl_item_fn(
        &self,
        receiver_rust_ty: &syn::Type,
        method_call_expr: &ExprMethodCall,
        generic_vars: &GenericVars,
        call_arg_rust_tys: &[syn::Type],
    ) -> Result<Option<(String, ItemImpl, ImplItemFn)>, JITError> {
        let method_name = method_call_expr.method.to_string();

        let receiver_lookup_ty =
            instantiate_type_for_lookup(receiver_rust_ty, generic_vars, self.primitives());
        let receiver_type_str = get_type_ident(&receiver_lookup_ty)
            .or_else(|| get_type_ident(receiver_rust_ty))
            .map(|ident| ident.to_string());

        // Inherent methods win before trait methods, matching Rust's lookup shape.
        if let Some(receiver_type_str) = receiver_type_str.as_deref() {
            if let Some(impls_vec) = self.name_resolver.struct_impls().get(receiver_type_str) {
                for (module_name, item_impl) in impls_vec {
                    if !impl_self_type_rank_matches(item_impl, &receiver_lookup_ty, generic_vars) {
                        continue;
                    }
                    if let Some(impl_method) = find_impl_method(item_impl, &method_name) {
                        return Ok(Some((
                            module_name.clone(),
                            item_impl.clone(),
                            impl_method.clone(),
                        )));
                    }
                }
            }
        }

        let mut trait_candidates = Vec::new();
        if matches!(
            generic_vars.instantiate_type(receiver_rust_ty, self.primitives())?,
            TypeInstance::ElementType(_)
        ) {
            if let Some(impls) = self
                .name_resolver
                .trait_impls()
                .get(&("BroadcastScalar".to_string(), "E".to_string()))
            {
                trait_candidates.extend(impls.iter().cloned());
            }
        }
        if let Some(receiver_type_str) = receiver_type_str.as_deref() {
            for ((_trait_name, self_name), impls) in self.name_resolver.trait_impls() {
                if self_name == receiver_type_str {
                    trait_candidates.extend(impls.iter().cloned());
                }
            }
        }

        let mut selected: Option<(String, ItemImpl, ImplItemFn)> = None;
        for (module_name, item_impl) in trait_candidates {
            for item in &item_impl.items {
                match item {
                    ImplItem::Fn(impl_item_fn) => {
                        let impl_item_fn_name = impl_item_fn.sig.ident.to_string();
                        if method_name != impl_item_fn_name {
                            continue;
                        }
                        let dispatch_matches = trait_impl_matches_call(
                            &item_impl,
                            impl_item_fn,
                            call_arg_rust_tys,
                            generic_vars,
                            self.primitives(),
                        );
                        if dispatch_matches {
                            if selected.is_some() {
                                return Err(JITError::Generic(format!(
                                    "ambiguous trait method dispatch for `{method_name}`"
                                )));
                            }
                            selected = Some((
                                module_name.clone(),
                                item_impl.clone(),
                                impl_item_fn.clone(),
                            ));
                        }
                    }
                    _ => continue,
                }
            }
        }
        Ok(selected)
    }

    pub fn get_struct_field_type(&self, struct_name: &str, field_name: &str) -> Option<Type> {
        self.name_resolver
            .get_struct_field_type(struct_name, field_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_records_span_base_by_short_name_and_absolute_path() {
        let ast: ItemMod = syn::parse_quote! {
            mod my_module {}
        };
        let span_base = SpanBase::new("source.rs".to_string(), 12, 4);
        let mut module = Module::with_span_base("my_module", ast, span_base.clone());
        module.set_absolute_path("my_crate::my_module".to_string());

        let modules = CUDATileModules::new(vec![module]).expect("module construction");

        assert_eq!(
            modules
                .get_span_base("my_module")
                .map(|base| base.file.as_str()),
            Some("source.rs")
        );
        assert_eq!(
            modules
                .get_span_base("my_crate::my_module")
                .map(|base| base.file.as_str()),
            Some("source.rs")
        );
    }

    #[test]
    fn relative_use_paths_resolve_against_owning_module() {
        assert_eq!(
            resolve_relative_use_path("super::activations::relu", "gpu::inter_module::my_kernels"),
            "gpu::inter_module::activations::relu"
        );
        assert_eq!(
            resolve_relative_use_path("self::helpers::relu", "gpu::inter_module::my_kernels"),
            "gpu::inter_module::my_kernels::helpers::relu"
        );
        assert_eq!(
            resolve_relative_use_path("crate::core::Tile", "cutile::core"),
            "cutile::core::Tile"
        );
        assert_eq!(
            resolve_relative_use_path("cutile::core::Tile", "gpu::inter_module::my_kernels"),
            "cutile::core::Tile"
        );
    }

    #[test]
    fn method_lookup_prefers_inherent_impl_before_trait_impl() {
        let ast: ItemMod = syn::parse_quote! {
            mod my_module {
                struct Foo;

                trait Pick {
                    fn pick(&self) -> i32;
                }

                impl Foo {
                    fn pick(&self) -> i64 {
                        0i64
                    }
                }

                impl Pick for Foo {
                    fn pick(&self) -> i32 {
                        0i32
                    }
                }
            }
        };
        let modules =
            CUDATileModules::new(vec![Module::new("my_module", ast)]).expect("module construction");
        let method_call: ExprMethodCall = syn::parse_quote!(foo.pick());
        let receiver_ty: Type = syn::parse_quote!(Foo);
        let call_arg_rust_tys = vec![receiver_ty.clone()];

        let Some((_module_name, _impl_item, impl_method)) = modules
            .get_impl_item_fn(
                &receiver_ty,
                &method_call,
                &GenericVars::empty_unchecked(),
                &call_arg_rust_tys,
            )
            .expect("method lookup")
        else {
            panic!("expected method lookup to select a method");
        };
        let (_inputs, return_ty) = get_sig_types(&impl_method.sig, Some(&receiver_ty));

        assert_eq!(return_ty.to_token_stream().to_string(), "i64");
    }
}
