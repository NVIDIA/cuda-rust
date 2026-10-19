/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pass 1: Name Resolution
//!
//! Mirrors rustc's name resolution architecture (simplified for the DSL):
//!
//! - [`DefId`] identifies any top-level definition (module + name)
//! - [`Res`] is the result of resolving a `syn::Path`
//! - [`Namespace`] separates types from values
//! - [`NameResolver`] takes `syn::Path` + calling module → `Res`
//! - Items are stored per-module in [`ModuleItems`] (out-of-band, like HIR)
//!
//! Reference: <https://rustc-dev-guide.rust-lang.org/name-resolution.html>

use crate::error::JITError;
use crate::syn_utils::*;
use quote::ToTokens;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use syn::{
    ImplItem, ImplItemFn, Item, ItemConst, ItemFn, ItemImpl, ItemMod, ItemStatic, ItemStruct,
    ItemTrait, ItemType, UseTree,
};

// ---------------------------------------------------------------------------
// Core types (rustc equivalents)
// ---------------------------------------------------------------------------

/// Identifies a top-level definition. Equivalent to rustc's `DefId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DefId {
    /// The module that defines this item.
    pub module: String,
    /// The item's name within that module.
    pub name: String,
}

/// What kind of definition a [`DefId`] refers to. Equivalent to rustc's `DefKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefKind {
    /// A function (including `#[cuda_tile::op]` functions).
    Fn,
    /// A struct (including `#[cuda_tile::ty]` structs).
    Struct,
    /// A trait definition.
    Trait,
    /// A type alias definition.
    TypeAlias,
    /// A module-level constant.
    Const,
    /// A module-level static item.
    Static,
    /// An associated function (method on a struct).
    AssocFn,
}

/// The result of resolving a name or path. Equivalent to rustc's `Res`.
#[derive(Debug, Clone)]
pub enum Res {
    /// A top-level definition.
    Def(DefKind, DefId),
    /// A local variable or function parameter (resolved during compilation,
    /// not during Pass 1 — included here for completeness).
    Local(String),
    /// A primitive type (`f32`, `i32`, etc.).
    PrimTy(String),
    /// Resolution failed.
    Err,
}

impl Res {
    /// Get the DefId if this is a Def resolution.
    pub fn def_id(&self) -> Option<&DefId> {
        match self {
            Res::Def(_, id) => Some(id),
            _ => None,
        }
    }

    /// Get the DefKind if this is a Def resolution.
    pub fn def_kind(&self) -> Option<DefKind> {
        match self {
            Res::Def(kind, _) => Some(*kind),
            _ => None,
        }
    }
}

/// Namespace for name resolution. Types and values don't collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Namespace {
    /// Structs, type aliases, traits.
    Type,
    /// Functions, variables, constants.
    Value,
}

// ---------------------------------------------------------------------------
// Per-module item storage (out-of-band, like HIR)
// ---------------------------------------------------------------------------

/// All items defined in a single module, indexed for fast lookup.
pub struct ModuleItems {
    pub functions: HashMap<String, ItemFn>,
    pub structs: HashMap<String, ItemStruct>,
    pub traits: HashMap<String, ItemTrait>,
    pub type_aliases: HashMap<String, ItemType>,
    pub consts: HashMap<String, ItemConst>,
    pub statics: HashMap<String, ItemStatic>,
    pub struct_impls: HashMap<String, Vec<ItemImpl>>,
    pub trait_impls: HashMap<(String, String), Vec<ItemImpl>>,
    pub primitives: HashMap<(String, String), ItemImpl>,
}

impl ModuleItems {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
            structs: HashMap::new(),
            traits: HashMap::new(),
            type_aliases: HashMap::new(),
            consts: HashMap::new(),
            statics: HashMap::new(),
            struct_impls: HashMap::new(),
            trait_impls: HashMap::new(),
            primitives: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Name resolver
// ---------------------------------------------------------------------------

/// Resolves `syn::Path` values to [`Res`] results using per-module indexes
/// and import maps. This is Pass 1 of the compilation pipeline.
///
/// Mirrors rustc's `Resolver` — paths stay intact in the AST, resolution
/// is a side-table lookup, items are in flat maps keyed by [`DefId`].
pub struct NameResolver {
    /// Per-module item indexes (our HIR items maps).
    items: HashMap<String, ModuleItems>,
    /// Module ASTs.
    modules: HashMap<String, ItemMod>,
    /// Per-module import maps: module_name → { local_name → source_module }.
    imports: HashMap<String, HashMap<String, String>>,
    /// The core module name (has `#[cuda_tile::ty]` annotations).
    core_module: Option<String>,

    // -- Cached flat maps for backward compatibility --
    /// All primitives across all modules, flattened.
    cached_primitives: HashMap<(String, String), ItemImpl>,
    /// All functions across all modules: name → (module_name, ItemFn).
    cached_functions: HashMap<String, (String, ItemFn)>,
    /// All structs across all modules: name → ItemStruct.
    cached_structs: HashMap<String, ItemStruct>,
    /// All traits across all modules: name → ItemTrait.
    cached_traits: HashMap<String, ItemTrait>,
    /// All type aliases across all modules: name → ItemType.
    cached_type_aliases: HashMap<String, ItemType>,
    /// All constants across all modules: name → ItemConst. Last definition
    /// wins for a name several modules define, so this map is only for the
    /// legacy callers that have no module to scope by; see
    /// [`Self::consts_visible_from`].
    cached_consts: HashMap<String, ItemConst>,
    /// Per-module memo of [`Self::consts_visible_from`]: type normalization
    /// asks for it on every compiled type.
    visible_consts: RefCell<HashMap<String, Rc<HashMap<String, ItemConst>>>>,
    /// All static items across all modules: name → ItemStatic.
    cached_statics: HashMap<String, ItemStatic>,
    /// All struct impls across all modules: struct_name → [(module_name, ItemImpl)].
    cached_struct_impls: HashMap<String, Vec<(String, ItemImpl)>>,
    /// All trait impls across all modules: (trait, self_ty) → [(module_name, ItemImpl)].
    cached_trait_impls: HashMap<(String, String), Vec<(String, ItemImpl)>>,
}

impl NameResolver {
    fn collect_use_imports(
        items_block: &[Item],
        items: &HashMap<String, ModuleItems>,
        module_imports: &mut HashMap<String, String>,
    ) {
        for item in items_block {
            match item {
                Item::Use(use_item) => {
                    Self::process_use_tree(&use_item.tree, &[], items, module_imports);
                }
                Item::Mod(submod) => {
                    if let Some((_, sub_items)) = &submod.content {
                        Self::collect_use_imports(sub_items, items, module_imports);
                    }
                }
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn index_items(
        items_block: &[Item],
        module_name: &str,
        mi: &mut ModuleItems,
        has_cuda_tile_ty: &mut bool,
        cached_functions: &mut HashMap<String, (String, ItemFn)>,
        cached_structs: &mut HashMap<String, ItemStruct>,
        cached_traits: &mut HashMap<String, ItemTrait>,
        cached_type_aliases: &mut HashMap<String, ItemType>,
        cached_consts: &mut HashMap<String, ItemConst>,
        cached_statics: &mut HashMap<String, ItemStatic>,
        cached_struct_impls: &mut HashMap<String, Vec<(String, ItemImpl)>>,
        cached_trait_impls: &mut HashMap<(String, String), Vec<(String, ItemImpl)>>,
        cached_primitives: &mut HashMap<(String, String), ItemImpl>,
    ) -> Result<(), JITError> {
        for item in items_block {
            match item {
                Item::Fn(f) => {
                    let name = f.sig.ident.to_string();
                    mi.functions.insert(name.clone(), f.clone());
                    // Flat cache: duplicate check matches old behavior.
                    if cached_functions
                        .insert(name.clone(), (module_name.to_string(), f.clone()))
                        .is_some()
                    {
                        return Err(JITError::generic_err(
                            &format!("duplicate functions are not supported; try renaming your function: {name}"),
                        ));
                    }
                }
                Item::Struct(s) => {
                    let name = s.ident.to_string();
                    mi.structs.insert(name.clone(), s.clone());
                    cached_structs.insert(name, s.clone());
                }
                Item::Trait(t) => {
                    let name = t.ident.to_string();
                    mi.traits.insert(name.clone(), t.clone());
                    cached_traits.insert(name, t.clone());
                }
                Item::Type(t) => {
                    let name = t.ident.to_string();
                    mi.type_aliases.insert(name.clone(), t.clone());
                    cached_type_aliases.insert(name, t.clone());
                }
                Item::Const(c) => {
                    let name = c.ident.to_string();
                    // A module's items — nested submodules included — form
                    // one flat scope here; a second definition of the name
                    // would silently replace the first.
                    if mi.consts.insert(name.clone(), c.clone()).is_some() {
                        return Err(JITError::generic_err(&format!(
                            "duplicate const `{name}` in module `{module_name}`: nested submodules \
                             share the module's flat scope; rename one of them"
                        )));
                    }
                    cached_consts.insert(name, c.clone());
                }
                Item::Static(s) => {
                    let name = s.ident.to_string();
                    mi.statics.insert(name.clone(), s.clone());
                    cached_statics.insert(name, s.clone());
                }
                Item::Impl(impl_item) => {
                    let self_ident = get_type_str(&impl_item.self_ty);
                    let trait_ident = impl_item
                        .trait_
                        .as_ref()
                        .map(|(_, path, _)| path.segments.last().unwrap().ident.to_string());

                    match (&self_ident, &trait_ident) {
                        (Some(self_name), Some(trait_name)) => {
                            if get_meta_list("cuda_tile :: ty", &impl_item.attrs).is_some() {
                                *has_cuda_tile_ty = true;
                                let key = (trait_name.clone(), self_name.clone());
                                mi.primitives.insert(key.clone(), impl_item.clone());
                                cached_primitives.insert(key, impl_item.clone());
                            } else {
                                let key = (trait_name.clone(), self_name.clone());
                                mi.trait_impls
                                    .entry(key.clone())
                                    .or_default()
                                    .push(impl_item.clone());
                                cached_trait_impls
                                    .entry(key)
                                    .or_default()
                                    .push((module_name.to_string(), impl_item.clone()));
                            }
                        }
                        (Some(self_name), None) => {
                            mi.struct_impls
                                .entry(self_name.clone())
                                .or_default()
                                .push(impl_item.clone());
                            cached_struct_impls
                                .entry(self_name.clone())
                                .or_default()
                                .push((module_name.to_string(), impl_item.clone()));
                        }
                        _ => {}
                    }
                }
                Item::Mod(submod) => {
                    if let Some((_, sub_items)) = &submod.content {
                        Self::index_items(
                            sub_items,
                            module_name,
                            mi,
                            has_cuda_tile_ty,
                            cached_functions,
                            cached_structs,
                            cached_traits,
                            cached_type_aliases,
                            cached_consts,
                            cached_statics,
                            cached_struct_impls,
                            cached_trait_impls,
                            cached_primitives,
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Build the resolver from parsed module ASTs. This is Pass 1.
    pub fn build(module_asts: &[(String, ItemMod)]) -> Result<Self, JITError> {
        let mut items: HashMap<String, ModuleItems> = HashMap::new();
        let mut modules: HashMap<String, ItemMod> = HashMap::new();
        let mut core_module: Option<String> = None;

        // Also build the cached flat maps during indexing.
        let mut cached_primitives: HashMap<(String, String), ItemImpl> = HashMap::new();
        let mut cached_functions: HashMap<String, (String, ItemFn)> = HashMap::new();
        let mut cached_structs: HashMap<String, ItemStruct> = HashMap::new();
        let mut cached_traits: HashMap<String, ItemTrait> = HashMap::new();
        let mut cached_type_aliases: HashMap<String, ItemType> = HashMap::new();
        let mut cached_consts: HashMap<String, ItemConst> = HashMap::new();
        let mut cached_statics: HashMap<String, ItemStatic> = HashMap::new();
        let mut cached_struct_impls: HashMap<String, Vec<(String, ItemImpl)>> = HashMap::new();
        let mut cached_trait_impls: HashMap<(String, String), Vec<(String, ItemImpl)>> =
            HashMap::new();

        // Phase 1: Index all items per module. Submodules nested inside a
        // `#[cutile::module]` (added once `cutile-macro` learned to recurse
        // into `Item::Mod`) are flattened into the parent module's namespace
        // for resolution purposes — they exist for human organization
        // (and rustc namespacing), but the JIT treats the whole module as
        // one flat scope.
        for (module_name, module_ast) in module_asts {
            modules.insert(module_name.clone(), module_ast.clone());
            let mut mi = ModuleItems::new();
            let Some(content) = &module_ast.content else {
                items.insert(module_name.clone(), mi);
                continue;
            };

            let mut has_cuda_tile_ty = false;
            Self::index_items(
                &content.1,
                module_name,
                &mut mi,
                &mut has_cuda_tile_ty,
                &mut cached_functions,
                &mut cached_structs,
                &mut cached_traits,
                &mut cached_type_aliases,
                &mut cached_consts,
                &mut cached_statics,
                &mut cached_struct_impls,
                &mut cached_trait_impls,
                &mut cached_primitives,
            )?;
            if has_cuda_tile_ty {
                core_module = Some(module_name.clone());
            }
            items.insert(module_name.clone(), mi);
        }

        // Phase 2: Process `use` statements to build import maps. Walks
        // nested submodules too so their `use` statements feed the same
        // flat import map as the parent (consistent with Phase 1).
        let mut imports: HashMap<String, HashMap<String, String>> = HashMap::new();
        for (module_name, module_ast) in module_asts {
            let mut module_imports: HashMap<String, String> = HashMap::new();
            if let Some(content) = &module_ast.content {
                Self::collect_use_imports(&content.1, &items, &mut module_imports);
            }
            imports.insert(module_name.clone(), module_imports);
        }

        Ok(NameResolver {
            items,
            modules,
            imports,
            core_module,
            cached_primitives,
            cached_functions,
            cached_structs,
            cached_traits,
            cached_type_aliases,
            cached_consts,
            cached_statics,
            cached_struct_impls,
            cached_trait_impls,
            visible_consts: RefCell::new(HashMap::new()),
        })
    }

    // -----------------------------------------------------------------------
    // Path resolution (the new API)
    // -----------------------------------------------------------------------

    /// Resolve a `syn::Path` to a [`Res`] in the context of `calling_module`.
    ///
    /// Handles:
    /// - Unqualified: `reshape` → local → imports → core → global
    /// - Qualified: `core::reshape` → look in module `core`
    /// - Fully qualified: `cutile::core::reshape` → strip crate prefix
    pub fn resolve_path(&self, path: &syn::Path, calling_module: &str) -> Res {
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();

        match segments.len() {
            0 => Res::Err,
            1 => {
                // Unqualified name: resolve through scope chain.
                let name = &segments[0];
                self.resolve_unqualified(name, calling_module)
            }
            2 => {
                // module::item or Type::method
                let (qualifier, name) = (&segments[0], &segments[1]);
                // Try as module::item first.
                if let Some(res) = self.resolve_in_module(name, qualifier) {
                    return res;
                }
                // Try as Type::method (associated function).
                if let Some((module, _, _method)) = self.find_method(qualifier, name) {
                    return Res::Def(
                        DefKind::AssocFn,
                        DefId {
                            module: module.to_string(),
                            name: name.clone(),
                        },
                    );
                }
                // Might be a qualified marker path like `ftz::Enabled` — not a
                // function/struct, so return Err for now (the compiler handles
                // these specially via UserType).
                Res::Err
            }
            _ => {
                // 3+ segments: strip crate-level prefixes and retry.
                // `cutile::core::reshape` → try `core::reshape` → try `reshape` in core.
                // Walk from the end to find a module match.
                for i in 0..segments.len() - 1 {
                    let candidate_module = &segments[i];
                    let item_name = &segments[segments.len() - 1];
                    if let Some(res) = self.resolve_in_module(item_name, candidate_module) {
                        return res;
                    }
                }
                // Fallback: try the last segment as unqualified.
                let name = &segments[segments.len() - 1];
                self.resolve_unqualified(name, calling_module)
            }
        }
    }

    /// Resolve an unqualified name through the scope chain:
    /// local module → imports → core → global fallback.
    fn resolve_unqualified(&self, name: &str, calling_module: &str) -> Res {
        // 1. Local definition.
        if let Some(res) = self.resolve_in_module(name, calling_module) {
            return res;
        }

        // 2. Explicit import.
        if let Some(module_imports) = self.imports.get(calling_module) {
            if let Some(source_module) = module_imports.get(name) {
                if let Some(res) = self.resolve_in_module(name, source_module) {
                    return res;
                }
            }
        }

        // 3. Implicit core import.
        if let Some(core) = &self.core_module {
            if calling_module != core {
                if let Some(res) = self.resolve_in_module(name, core) {
                    return res;
                }
            }
        }

        // 4. Global fallback (backward compatibility).
        for (module_name, mi) in &self.items {
            if let Some(res) = Self::lookup_in_items(name, module_name, mi) {
                return res;
            }
        }

        Res::Err
    }

    /// Try to resolve `name` in a specific module.
    fn resolve_in_module(&self, name: &str, module: &str) -> Option<Res> {
        let mi = self.items.get(module)?;
        Self::lookup_in_items(name, module, mi)
    }

    /// Look up a name in a module's items, returning a Res.
    fn lookup_in_items(name: &str, module: &str, mi: &ModuleItems) -> Option<Res> {
        if mi.functions.contains_key(name) {
            return Some(Res::Def(
                DefKind::Fn,
                DefId {
                    module: module.to_string(),
                    name: name.to_string(),
                },
            ));
        }
        if mi.structs.contains_key(name) {
            return Some(Res::Def(
                DefKind::Struct,
                DefId {
                    module: module.to_string(),
                    name: name.to_string(),
                },
            ));
        }
        if mi.traits.contains_key(name) {
            return Some(Res::Def(
                DefKind::Trait,
                DefId {
                    module: module.to_string(),
                    name: name.to_string(),
                },
            ));
        }
        if mi.type_aliases.contains_key(name) {
            return Some(Res::Def(
                DefKind::TypeAlias,
                DefId {
                    module: module.to_string(),
                    name: name.to_string(),
                },
            ));
        }
        if mi.consts.contains_key(name) {
            return Some(Res::Def(
                DefKind::Const,
                DefId {
                    module: module.to_string(),
                    name: name.to_string(),
                },
            ));
        }
        if mi.statics.contains_key(name) {
            return Some(Res::Def(
                DefKind::Static,
                DefId {
                    module: module.to_string(),
                    name: name.to_string(),
                },
            ));
        }
        None
    }

    // -----------------------------------------------------------------------
    // Item accessors (given a DefId, return the item)
    // -----------------------------------------------------------------------

    /// Get a function by its DefId.
    pub fn get_fn(&self, def_id: &DefId) -> Option<&ItemFn> {
        self.items.get(&def_id.module)?.functions.get(&def_id.name)
    }

    /// Get a struct by its DefId.
    pub fn get_struct(&self, def_id: &DefId) -> Option<&ItemStruct> {
        self.items.get(&def_id.module)?.structs.get(&def_id.name)
    }

    /// Get a trait by its DefId.
    pub fn get_trait(&self, def_id: &DefId) -> Option<&ItemTrait> {
        self.items.get(&def_id.module)?.traits.get(&def_id.name)
    }

    /// Get a type alias by its DefId.
    pub fn get_type_alias(&self, def_id: &DefId) -> Option<&ItemType> {
        self.items
            .get(&def_id.module)?
            .type_aliases
            .get(&def_id.name)
    }

    /// Get a module-level constant by its DefId.
    pub fn get_const(&self, def_id: &DefId) -> Option<&ItemConst> {
        self.items.get(&def_id.module)?.consts.get(&def_id.name)
    }

    /// Get a module-level static by its DefId.
    pub fn get_static(&self, def_id: &DefId) -> Option<&ItemStatic> {
        self.items.get(&def_id.module)?.statics.get(&def_id.name)
    }

    /// Resolve a module-level constant by name from `calling_module`, along
    /// the scope chain of [`Self::resolve_path`]: the module's own item, an
    /// explicit import, the implicit core import, and finally any module —
    /// but only when exactly one module defines the name, or every
    /// definition is token-identical. A name that several modules define
    /// differently, and that the calling module neither defines nor imports,
    /// is ambiguous and resolves to nothing rather than to whichever module
    /// happened to be indexed last (which is what the flat map did: a helper
    /// module's `N` replaced the kernel's own `N` in its tile shapes).
    pub fn resolve_const(&self, name: &str, calling_module: &str) -> Option<&ItemConst> {
        let in_module = |module: &str| self.items.get(module).and_then(|mi| mi.consts.get(name));
        if let Some(item) = in_module(calling_module) {
            return Some(item);
        }
        if let Some(source) = self
            .imports
            .get(calling_module)
            .and_then(|imports| imports.get(name))
        {
            if let Some(item) = in_module(source) {
                return Some(item);
            }
        }
        if let Some(item) = self.core_module.as_deref().and_then(in_module) {
            return Some(item);
        }
        let mut definitions = self
            .items
            .values()
            .filter_map(|mi| mi.consts.get(name))
            .peekable();
        let first = definitions.next()?;
        let same_text = |item: &ItemConst| {
            item.to_token_stream().to_string() == first.to_token_stream().to_string()
        };
        definitions.all(same_text).then_some(first)
    }

    /// Every constant visible from `module`, by name (see
    /// [`Self::resolve_const`]). Memoized per module.
    pub fn consts_visible_from(&self, module: &str) -> Rc<HashMap<String, ItemConst>> {
        if let Some(cached) = self.visible_consts.borrow().get(module) {
            return cached.clone();
        }
        let names: BTreeSet<&str> = self
            .items
            .values()
            .flat_map(|mi| mi.consts.keys().map(String::as_str))
            .collect();
        let visible: HashMap<String, ItemConst> = names
            .into_iter()
            .filter_map(|name| {
                self.resolve_const(name, module)
                    .map(|item| (name.to_string(), item.clone()))
            })
            .collect();
        let visible = Rc::new(visible);
        self.visible_consts
            .borrow_mut()
            .insert(module.to_string(), visible.clone());
        visible
    }

    /// Iterate all indexed static items with their defining module.
    pub fn all_statics(&self) -> impl Iterator<Item = (&str, &ItemStatic)> {
        self.items.iter().flat_map(|(module_name, items)| {
            items
                .statics
                .values()
                .map(move |item| (module_name.as_str(), item))
        })
    }

    /// Find a method on a struct. Searches all modules' impls.
    pub fn find_method(
        &self,
        struct_name: &str,
        method_name: &str,
    ) -> Option<(&str, &ItemImpl, ImplItemFn)> {
        for (module_name, mi) in &self.items {
            if let Some(impls) = mi.struct_impls.get(struct_name) {
                for impl_item in impls {
                    for item in &impl_item.items {
                        if let ImplItem::Fn(f) = item {
                            if f.sig.ident == method_name {
                                return Some((module_name.as_str(), impl_item, f.clone()));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Get a primitive type impl by (trait_name, rust_type_name).
    pub fn get_primitive(&self, trait_name: &str, rust_type: &str) -> Option<&ItemImpl> {
        let key = (trait_name.to_string(), rust_type.to_string());
        for mi in self.items.values() {
            if let Some(impl_item) = mi.primitives.get(&key) {
                return Some(impl_item);
            }
        }
        None
    }

    /// Get `#[cuda_tile::ty]` attrs on a primitive type.
    pub fn get_primitive_attrs(&self, trait_name: &str, rust_type: &str) -> Option<SingleMetaList> {
        let impl_item = self.get_primitive(trait_name, rust_type)?;
        get_meta_list("cuda_tile :: ty", &impl_item.attrs)
    }

    /// Get a trait impl by (trait_name, self_type).
    pub fn get_trait_impl(&self, trait_name: &str, self_type: &str) -> Option<(&str, &ItemImpl)> {
        let key = (trait_name.to_string(), self_type.to_string());
        for (module_name, mi) in &self.items {
            if let Some(impls) = mi.trait_impls.get(&key) {
                let Some(impl_item) = impls.first() else {
                    continue;
                };
                return Some((module_name.as_str(), impl_item));
            }
        }
        None
    }

    /// Get `#[cuda_tile::ty]` attrs on a struct.
    pub fn get_type_attrs(&self, struct_name: &str) -> Option<SingleMetaList> {
        for mi in self.items.values() {
            if let Some(s) = mi.structs.get(struct_name) {
                return get_meta_list("cuda_tile :: ty", &s.attrs);
            }
        }
        None
    }

    /// Get `#[cuda_tile::op]` attrs on a function.
    pub fn get_op_attrs(&self, fn_name: &str) -> Option<SingleMetaList> {
        for mi in self.items.values() {
            if let Some(f) = mi.functions.get(fn_name) {
                return get_meta_list("cuda_tile :: op", &f.attrs);
            }
        }
        None
    }

    /// Get struct field type by struct and field name.
    pub fn get_struct_field_type(&self, struct_name: &str, field_name: &str) -> Option<syn::Type> {
        for mi in self.items.values() {
            if let Some(s) = mi.structs.get(struct_name) {
                for field in &s.fields {
                    if let Some(ident) = &field.ident {
                        if ident == field_name {
                            return Some(field.ty.clone());
                        }
                    }
                }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Module access
    // -----------------------------------------------------------------------

    pub fn module(&self, name: &str) -> Option<&ItemMod> {
        self.modules.get(name)
    }

    pub fn has_module(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    pub fn core_module(&self) -> Option<&str> {
        self.core_module.as_deref()
    }

    // -----------------------------------------------------------------------
    // Backward-compatible flat map accessors
    // -----------------------------------------------------------------------

    /// Flat map of all primitives. Compatibility shim for code that passes
    /// `&self.modules.primitives` to generics/types functions.
    pub fn primitives(&self) -> &HashMap<(String, String), ItemImpl> {
        &self.cached_primitives
    }

    /// Flat map of all functions: name → (module_name, ItemFn).
    pub fn functions(&self) -> &HashMap<String, (String, ItemFn)> {
        &self.cached_functions
    }

    /// Flat map of all structs: name → ItemStruct.
    pub fn structs(&self) -> &HashMap<String, ItemStruct> {
        &self.cached_structs
    }

    /// Flat map of all traits: name → ItemTrait.
    pub fn traits(&self) -> &HashMap<String, ItemTrait> {
        &self.cached_traits
    }

    /// Flat map of all type aliases: name → ItemType.
    pub fn type_aliases(&self) -> &HashMap<String, ItemType> {
        &self.cached_type_aliases
    }

    /// Flat map of all constants: name → ItemConst.
    pub fn consts(&self) -> &HashMap<String, ItemConst> {
        &self.cached_consts
    }

    /// Flat map of all static items: name → ItemStatic.
    pub fn statics(&self) -> &HashMap<String, ItemStatic> {
        &self.cached_statics
    }

    /// Flat map of all struct impls: struct_name → [(module_name, ItemImpl)].
    pub fn struct_impls(&self) -> &HashMap<String, Vec<(String, ItemImpl)>> {
        &self.cached_struct_impls
    }

    /// Flat map of all trait impls: (trait, self_ty) → [(module, ItemImpl)].
    pub fn trait_impls(&self) -> &HashMap<(String, String), Vec<(String, ItemImpl)>> {
        &self.cached_trait_impls
    }

    /// All module ASTs.
    pub fn all_modules(&self) -> &HashMap<String, ItemMod> {
        &self.modules
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// List all modules that define a given name.
    pub fn find_all_definitions(&self, name: &str) -> Vec<&str> {
        self.items
            .iter()
            .filter(|(_, mi)| {
                mi.functions.contains_key(name)
                    || mi.structs.contains_key(name)
                    || mi.traits.contains_key(name)
                    || mi.type_aliases.contains_key(name)
                    || mi.consts.contains_key(name)
            })
            .map(|(module_name, _)| module_name.as_str())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Use-tree processing
    // -----------------------------------------------------------------------

    fn process_use_tree(
        tree: &UseTree,
        path_prefix: &[String],
        items: &HashMap<String, ModuleItems>,
        imports: &mut HashMap<String, String>,
    ) {
        match tree {
            UseTree::Path(path) => {
                let mut prefix = path_prefix.to_vec();
                prefix.push(path.ident.to_string());
                Self::process_use_tree(&path.tree, &prefix, items, imports);
            }
            UseTree::Name(name) => {
                if let Some(source) = path_prefix.last() {
                    imports.insert(name.ident.to_string(), source.clone());
                }
            }
            UseTree::Glob(_) => {
                if let Some(source) = path_prefix.last() {
                    if let Some(mi) = items.get(source) {
                        for name in mi.functions.keys() {
                            imports.insert(name.clone(), source.clone());
                        }
                        for name in mi.structs.keys() {
                            imports.insert(name.clone(), source.clone());
                        }
                        for name in mi.traits.keys() {
                            imports.insert(name.clone(), source.clone());
                        }
                        for name in mi.type_aliases.keys() {
                            imports.insert(name.clone(), source.clone());
                        }
                        for name in mi.consts.keys() {
                            imports.insert(name.clone(), source.clone());
                        }
                    }
                }
            }
            UseTree::Group(group) => {
                for tree in &group.items {
                    Self::process_use_tree(tree, path_prefix, items, imports);
                }
            }
            UseTree::Rename(rename) => {
                if let Some(source) = path_prefix.last() {
                    imports.insert(rename.rename.to_string(), source.clone());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn make_module(name: &str, items_vec: Vec<Item>) -> (String, ItemMod) {
        let ident = syn::Ident::new(name, proc_macro2::Span::call_site());
        let module: ItemMod = parse_quote! {
            mod #ident {
                #(#items_vec)*
            }
        };
        (name.to_string(), module)
    }

    fn parse_path(s: &str) -> syn::Path {
        syn::parse_str(s).unwrap()
    }

    #[test]
    fn resolve_unqualified_local() {
        let (name, module) = make_module("test_mod", vec![parse_quote! { fn my_func() {} }]);
        let resolver = NameResolver::build(&[(name, module)]).unwrap();
        let res = resolver.resolve_path(&parse_path("my_func"), "test_mod");
        match res {
            Res::Def(DefKind::Fn, def_id) => {
                assert_eq!(def_id.module, "test_mod");
                assert_eq!(def_id.name, "my_func");
            }
            _ => panic!("expected Def(Fn, ...), got {:?}", res),
        }
    }

    #[test]
    fn resolve_qualified_module_item() {
        let (a, a_mod) = make_module("mod_a", vec![parse_quote! { fn helper() {} }]);
        let (b, b_mod) = make_module("mod_b", vec![parse_quote! { fn other() {} }]);
        let resolver = NameResolver::build(&[(a, a_mod), (b, b_mod)]).unwrap();

        // Qualified: mod_a::helper from mod_b.
        let res = resolver.resolve_path(&parse_path("mod_a::helper"), "mod_b");
        match res {
            Res::Def(DefKind::Fn, def_id) => assert_eq!(def_id.module, "mod_a"),
            _ => panic!("expected Def, got {:?}", res),
        }
    }

    #[test]
    fn resolve_unknown_returns_err() {
        let (name, module) = make_module("test_mod", vec![parse_quote! { fn my_func() {} }]);
        let resolver = NameResolver::build(&[(name, module)]).unwrap();
        assert!(matches!(
            resolver.resolve_path(&parse_path("nonexistent"), "test_mod"),
            Res::Err
        ));
    }

    #[test]
    fn duplicate_function_names_rejected() {
        // The flat cache (backward compat) rejects duplicate function names.
        // Once the compiler fully migrates to module-scoped lookup, this
        // restriction can be relaxed.
        let (a, a_mod) = make_module("mod_a", vec![parse_quote! { fn dup() -> i32 { 1 } }]);
        let (b, b_mod) = make_module("mod_b", vec![parse_quote! { fn dup() -> i32 { 2 } }]);
        assert!(NameResolver::build(&[(a, a_mod), (b, b_mod)]).is_err());
    }

    #[test]
    fn cross_module_resolution() {
        // mod_a defines helper, mod_b defines other. mod_b can find helper via fallback.
        let (a, a_mod) = make_module("mod_a", vec![parse_quote! { fn helper() {} }]);
        let (b, b_mod) = make_module("mod_b", vec![parse_quote! { fn other() {} }]);
        let resolver = NameResolver::build(&[(a, a_mod), (b, b_mod)]).unwrap();

        match resolver.resolve_path(&parse_path("helper"), "mod_b") {
            Res::Def(_, def_id) => assert_eq!(def_id.module, "mod_a"),
            _ => panic!("expected Def"),
        }
        // mod_b resolves its own function locally.
        match resolver.resolve_path(&parse_path("other"), "mod_b") {
            Res::Def(_, def_id) => assert_eq!(def_id.module, "mod_b"),
            _ => panic!("expected Def"),
        }
    }

    #[test]
    fn resolve_struct() {
        let (name, module) = make_module("test_mod", vec![parse_quote! { struct Foo {} }]);
        let resolver = NameResolver::build(&[(name, module)]).unwrap();
        match resolver.resolve_path(&parse_path("Foo"), "test_mod") {
            Res::Def(DefKind::Struct, def_id) => {
                assert_eq!(def_id.name, "Foo");
                assert!(resolver.get_struct(&def_id).is_some());
            }
            _ => panic!("expected Def(Struct, ...)"),
        }
    }

    #[test]
    fn cached_flat_maps_populated() {
        let (a, a_mod) = make_module(
            "mod_a",
            vec![
                parse_quote! { fn alpha() {} },
                parse_quote! { struct Beta {} },
            ],
        );
        let (b, b_mod) = make_module("mod_b", vec![parse_quote! { fn gamma() {} }]);
        let resolver = NameResolver::build(&[(a, a_mod), (b, b_mod)]).unwrap();

        assert!(resolver.functions().contains_key("alpha"));
        assert!(resolver.functions().contains_key("gamma"));
        assert!(resolver.structs().contains_key("Beta"));
    }

    /// Same-named consts in two modules resolve per module: the kernel's own
    /// definition wins for the kernel, the helper's for the helper, and a
    /// third module that imports one of them gets that one. Only the flat
    /// map is last-wins.
    #[test]
    fn same_named_consts_resolve_per_module() {
        let (a, a_mod) = make_module(
            "kernel_mod",
            vec![
                parse_quote! { use helper_mod::helper; },
                parse_quote! { const N: i32 = 128; },
            ],
        );
        let (b, b_mod) = make_module(
            "helper_mod",
            vec![
                parse_quote! { const N: i32 = 64; },
                parse_quote! { const ONLY_HERE: i32 = 7; },
                parse_quote! { pub fn helper() {} },
            ],
        );
        let (c, c_mod) = make_module("importer_mod", vec![parse_quote! { use helper_mod::N; }]);
        let (d, d_mod) = make_module("bystander_mod", vec![parse_quote! { fn other() {} }]);
        let resolver =
            NameResolver::build(&[(a, a_mod), (b, b_mod), (c, c_mod), (d, d_mod)]).unwrap();
        let value = |item: &ItemConst| crate::type_aliases::const_item_i32_value(item).unwrap();

        assert_eq!(
            value(resolver.resolve_const("N", "kernel_mod").unwrap()),
            128
        );
        assert_eq!(
            value(resolver.resolve_const("N", "helper_mod").unwrap()),
            64
        );
        assert_eq!(
            value(resolver.resolve_const("N", "importer_mod").unwrap()),
            64
        );
        // Neither defined nor imported, and defined differently elsewhere:
        // ambiguous, so unresolved — never "whichever was indexed last".
        assert!(resolver.resolve_const("N", "bystander_mod").is_none());
        // A unique global definition still resolves from anywhere.
        assert_eq!(
            value(
                resolver
                    .resolve_const("ONLY_HERE", "bystander_mod")
                    .unwrap()
            ),
            7
        );

        let kernel_view = resolver.consts_visible_from("kernel_mod");
        assert_eq!(value(&kernel_view["N"]), 128);
        assert_eq!(value(&kernel_view["ONLY_HERE"]), 7);
        let bystander_view = resolver.consts_visible_from("bystander_mod");
        assert!(!bystander_view.contains_key("N"));
        assert!(bystander_view.contains_key("ONLY_HERE"));
    }

    /// A const defined twice inside one module's flat scope (a nested
    /// submodule reusing the name) is rejected instead of silently replaced.
    #[test]
    fn duplicate_const_within_a_module_is_rejected() {
        let (name, module) = make_module(
            "dup_mod",
            vec![
                parse_quote! { const N: i32 = 1; },
                parse_quote! { mod inner { const N: i32 = 2; } },
            ],
        );
        let err = NameResolver::build(&[(name, module)])
            .err()
            .expect("a duplicate const must be rejected");
        assert!(
            err.to_string().contains("duplicate const `N`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn get_fn_via_def_id() {
        let (name, module) = make_module("test_mod", vec![parse_quote! { fn my_func() {} }]);
        let resolver = NameResolver::build(&[(name, module)]).unwrap();
        let def_id = DefId {
            module: "test_mod".into(),
            name: "my_func".into(),
        };
        let f = resolver.get_fn(&def_id).unwrap();
        assert_eq!(f.sig.ident, "my_func");
    }
}
