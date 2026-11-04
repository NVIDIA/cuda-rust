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
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use syn::visit::Visit;
use syn::{
    ImplItem, ImplItemFn, Item, ItemConst, ItemFn, ItemImpl, ItemMod, ItemStatic, ItemStruct,
    ItemTrait, ItemType, UseTree, Visibility,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// The paths of the `mod`s nested in the module, outermost first.
    pub nested_mods: HashSet<Vec<String>>,
    /// Whether code in a nested `mod` names a `super::` path. The JIT does
    /// not track which `mod` holds a call, so such a module's `super::` call
    /// paths are refused rather than resolved from the wrong `mod`.
    pub super_in_nested_mod: bool,
    /// Structs, traits and type aliases that the module defines twice under
    /// one name, in two nested `mod`s or in one and at its top level. Nested
    /// `mod`s share the module's flat scope, so such a name stands for
    /// neither item: resolving it is an error.
    pub duplicates: HashSet<(DefKind, String)>,
    /// The path of the nested `mod` that declares each item not at the
    /// module's top level, by kind and name.
    pub nested_items: HashMap<(DefKind, String), Vec<String>>,
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
            nested_mods: HashSet::new(),
            super_in_nested_mod: false,
            duplicates: HashSet::new(),
            nested_items: HashMap::new(),
        }
    }

    /// The error for resolving `name`, found as a `kind` in `module`, when
    /// the module defines it more than once (see [`Self::duplicates`]).
    fn duplicate_error(&self, kind: DefKind, name: &str, module: &str) -> Option<String> {
        if !self.duplicates.contains(&(kind, name.to_string())) {
            return None;
        }
        let noun = match kind {
            DefKind::Struct => "struct",
            DefKind::Trait => "trait",
            DefKind::TypeAlias => "type alias",
            DefKind::Fn | DefKind::Const | DefKind::Static | DefKind::AssocFn => "item",
        };
        Some(duplicate_message(noun, name, module))
    }
}

/// What a single-name `use` brings into scope.
#[derive(Clone, PartialEq)]
struct Import {
    /// The loaded module that the `use` prefix names (see
    /// [`NameResolver::module_named_in_use`]), or the prefix's last segment
    /// (`crate` for an empty prefix) when it names none.
    source: String,
    /// The name in `source`; it differs from the local name for
    /// `use a::f as g`.
    original: String,
    /// The loaded module, and the `mod` path in it, that the whole path
    /// names, when it names one: `use a::m;` imports module `m`.
    module: Option<(String, Vec<String>)>,
}

/// A module's `use` statements, nested submodules included.
#[derive(Default)]
struct ModuleImports {
    /// Single-name imports, by local name.
    names: HashMap<String, Import>,
    /// Named imports exported by `pub use` (including restricted visibility).
    public_names: HashSet<String>,
    /// Names that a `use` at the module's top level imports from the
    /// module's own scope (`use self::inner::f;`): the module's top-level
    /// code means that item by the plain name.
    top_level_own: HashSet<String>,
    /// Glob imports (`use a::m::*`): the source module, as
    /// [`Import::source`], the path of the `mod` nested in it that the glob
    /// names (empty for the module itself), and whether the glob is `pub`.
    globs: Vec<(String, Vec<String>, bool)>,
    /// Local names that two `use`s, in different nested `mod`s, import from
    /// different places. Nested `mod`s share one scope, so they are errors.
    conflicts: HashSet<String>,
}

impl ModuleImports {
    /// Records `local` as imported into `module`'s scope.
    fn insert(
        &mut self,
        module: &str,
        nested: &[String],
        local: String,
        target: Import,
        public: bool,
    ) {
        if public {
            self.public_names.insert(local.clone());
        }
        if nested.is_empty() && target.source == module && target.original == local {
            self.top_level_own.insert(local.clone());
        }
        // `use super::x` in a nested `mod` names the shared scope's `x`:
        // the other import of `x`, if any, is the one to follow.
        let relative = |import: &Import| {
            import.source == "super" || (import.source == module && import.original == local)
        };
        match self.names.get(&local) {
            Some(known) if *known == target || relative(&target) => return,
            Some(known) if !relative(known) => {
                self.conflicts.insert(local.clone());
            }
            _ => {}
        }
        self.names.insert(local, target);
    }
}

/// Indexes `item` as `name` in one of a module's flat maps. Nested `mod`s
/// share the module's flat scope, so a second definition of the name in the
/// module would silently replace the first: that is an error.
fn insert_unique<T: Clone>(
    map: &mut HashMap<String, T>,
    name: &str,
    item: &T,
    noun: &str,
    module_name: &str,
) -> Result<(), JITError> {
    if map.insert(name.to_string(), item.clone()).is_some() {
        return Err(JITError::generic_err(&duplicate_message(
            noun,
            name,
            module_name,
        )));
    }
    Ok(())
}

/// Indexes `item` as `name` in one of a module's flat maps, and records a
/// second definition of the name as a duplicate of `kind`: only resolving
/// the name is an error (see [`ModuleItems::duplicates`]).
fn insert_or_record<T: Clone>(
    map: &mut HashMap<String, T>,
    duplicates: &mut HashSet<(DefKind, String)>,
    kind: DefKind,
    name: &str,
    item: &T,
) {
    if map.insert(name.to_string(), item.clone()).is_some() {
        duplicates.insert((kind, name.to_string()));
    }
}

/// Sets `cache[name]` to the last definition, in `order`, of a module that
/// defines `name` as a `kind` once, or removes it when every module that
/// defines it defines it twice (see [`ModuleItems::duplicates`]). Another
/// module's own definition is not lost to one module's duplicate.
fn keep_single_definition<T: Clone>(
    cache: &mut HashMap<String, T>,
    items: &HashMap<String, ModuleItems>,
    order: &[&str],
    kind: &DefKind,
    name: &str,
    map: impl Fn(&ModuleItems) -> &HashMap<String, T>,
) {
    let single = order.iter().rev().find_map(|module| {
        let mi = items.get(*module)?;
        let item = map(mi).get(name)?;
        (!mi.duplicates.contains(&(*kind, name.to_string()))).then(|| item.clone())
    });
    match single {
        Some(item) => {
            cache.insert(name.to_string(), item);
        }
        None => {
            cache.remove(name);
        }
    }
}

fn duplicate_message(noun: &str, name: &str, module_name: &str) -> String {
    format!(
        "duplicate {noun} `{name}` in module `{module_name}`: nested submodules share \
         the module's flat scope; rename one of them"
    )
}

fn is_pub(vis: &Visibility) -> bool {
    !matches!(vis, Visibility::Inherited)
}

/// The kind of the item `mi` defines as `name`, and whether it is `pub`.
/// `kind` restricts the lookup to one kind; `None` tries every kind in the
/// order of [`NameResolver::resolve_path`].
fn find(mi: &ModuleItems, kind: Option<DefKind>, name: &str) -> Option<(DefKind, bool)> {
    [
        (DefKind::Fn, mi.functions.get(name).map(|i| &i.vis)),
        (DefKind::Struct, mi.structs.get(name).map(|i| &i.vis)),
        (DefKind::Trait, mi.traits.get(name).map(|i| &i.vis)),
        (
            DefKind::TypeAlias,
            mi.type_aliases.get(name).map(|i| &i.vis),
        ),
        (DefKind::Const, mi.consts.get(name).map(|i| &i.vis)),
        (DefKind::Static, mi.statics.get(name).map(|i| &i.vis)),
    ]
    .into_iter()
    // A struct lookup also accepts aliases in the type namespace. The
    // caller expands them in their defining module before reading a marker.
    .filter(|(found, _)| {
        kind.is_none_or(|kind| {
            kind == *found || (kind == DefKind::Struct && *found == DefKind::TypeAlias)
        })
    })
    .find_map(|(found, vis)| Some((found, is_pub(vis?))))
}

/// Finds a path that starts with `super`, except in `use` items and
/// `pub(super)` visibilities.
struct SuperPathFinder(bool);

impl<'ast> Visit<'ast> for SuperPathFinder {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.0 |= path.segments.first().is_some_and(|s| s.ident == "super");
        syn::visit::visit_path(self, path);
    }
    fn visit_item_use(&mut self, _: &'ast syn::ItemUse) {}
    fn visit_visibility(&mut self, _: &'ast Visibility) {}
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
    /// Each module's absolute Rust path (`module_path!()`), by module name.
    /// A module missing here is at the path its name spells.
    module_paths: HashMap<String, String>,
    /// Per-module `use` statements.
    imports: HashMap<String, ModuleImports>,
    /// The core module name (has `#[cuda_tile::ty]` annotations).
    core_module: Option<String>,

    // -- Cached flat maps for backward compatibility --
    /// All primitives across all modules, flattened.
    cached_primitives: HashMap<(String, String), ItemImpl>,
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
    /// Records the `use`s of the `mod` at `nested` in `module`, and of the
    /// `mod`s nested in it.
    fn collect_use_imports(
        &self,
        module: &str,
        nested: &[String],
        items_block: &[Item],
        module_imports: &mut ModuleImports,
    ) {
        for item in items_block {
            match item {
                Item::Use(use_item) => {
                    self.process_use_tree(
                        module,
                        nested,
                        &use_item.tree,
                        &[],
                        is_pub(&use_item.vis),
                        module_imports,
                    );
                }
                Item::Mod(submod) => {
                    if let Some((_, sub_items)) = &submod.content {
                        let mut sub_path = nested.to_vec();
                        sub_path.push(submod.ident.to_string());
                        self.collect_use_imports(module, &sub_path, sub_items, module_imports);
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
        nested_path: &[String],
        mi: &mut ModuleItems,
        has_cuda_tile_ty: &mut bool,
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
            let declared = match item {
                Item::Fn(f) => Some((DefKind::Fn, &f.sig.ident)),
                Item::Struct(s) => Some((DefKind::Struct, &s.ident)),
                Item::Trait(t) => Some((DefKind::Trait, &t.ident)),
                Item::Type(t) => Some((DefKind::TypeAlias, &t.ident)),
                Item::Const(c) => Some((DefKind::Const, &c.ident)),
                Item::Static(s) => Some((DefKind::Static, &s.ident)),
                _ => None,
            };
            if let Some((kind, ident)) = declared.filter(|_| !nested_path.is_empty()) {
                mi.nested_items
                    .insert((kind, ident.to_string()), nested_path.to_vec());
            }
            match item {
                Item::Fn(f) => {
                    // Functions are per module: two modules may define the
                    // same name (see `Self::resolve_in_scope`).
                    let name = f.sig.ident.to_string();
                    insert_unique(&mut mi.functions, &name, f, "function", module_name)?;
                }
                // Nested `mod`s may each define a type of one name, as
                // marker namespaces (`ftz::Enabled`, `nan::Enabled`) do.
                Item::Struct(s) => {
                    let name = s.ident.to_string();
                    let kind = DefKind::Struct;
                    insert_or_record(&mut mi.structs, &mut mi.duplicates, kind, &name, s);
                    cached_structs.insert(name, s.clone());
                }
                Item::Trait(t) => {
                    let name = t.ident.to_string();
                    let kind = DefKind::Trait;
                    insert_or_record(&mut mi.traits, &mut mi.duplicates, kind, &name, t);
                    cached_traits.insert(name, t.clone());
                }
                Item::Type(t) => {
                    let name = t.ident.to_string();
                    let kind = DefKind::TypeAlias;
                    insert_or_record(&mut mi.type_aliases, &mut mi.duplicates, kind, &name, t);
                    cached_type_aliases.insert(name, t.clone());
                }
                Item::Const(c) => {
                    let name = c.ident.to_string();
                    insert_unique(&mut mi.consts, &name, c, "const", module_name)?;
                    cached_consts.insert(name, c.clone());
                }
                Item::Static(s) => {
                    let name = s.ident.to_string();
                    insert_unique(&mut mi.statics, &name, s, "static", module_name)?;
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
                    if nested_path.is_empty() {
                        let mut finder = SuperPathFinder(false);
                        finder.visit_item_mod(submod);
                        mi.super_in_nested_mod |= finder.0;
                    }
                    if let Some((_, sub_items)) = &submod.content {
                        let mut sub_path = nested_path.to_vec();
                        sub_path.push(submod.ident.to_string());
                        mi.nested_mods.insert(sub_path.clone());
                        Self::index_items(
                            sub_items,
                            module_name,
                            &sub_path,
                            mi,
                            has_cuda_tile_ty,
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

    /// Build the resolver from parsed module ASTs, by module name, each
    /// module at the absolute path its name spells. This is Pass 1.
    pub fn build(module_asts: &[(String, ItemMod)]) -> Result<Self, JITError> {
        Self::build_with_paths(module_asts, HashMap::new())
    }

    /// [`Self::build`], with modules at the absolute Rust paths
    /// `module_paths` gives by module name. `use` paths resolve against
    /// them.
    pub fn build_with_paths(
        module_asts: &[(String, ItemMod)],
        module_paths: HashMap<String, String>,
    ) -> Result<Self, JITError> {
        let mut items: HashMap<String, ModuleItems> = HashMap::new();
        let mut modules: HashMap<String, ItemMod> = HashMap::new();
        let mut core_module: Option<String> = None;

        // Also build the cached flat maps during indexing.
        let mut cached_primitives: HashMap<(String, String), ItemImpl> = HashMap::new();
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
                &[],
                &mut mi,
                &mut has_cuda_tile_ty,
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
        // A name a module defines twice stands for neither definition, so
        // the name-only maps (type normalization, struct layouts) hold
        // another module's single definition of it, or nothing.
        let order: Vec<&str> = module_asts.iter().map(|(name, _)| name.as_str()).collect();
        for (kind, name) in items.values().flat_map(|mi| &mi.duplicates) {
            match kind {
                DefKind::Struct => {
                    keep_single_definition(&mut cached_structs, &items, &order, kind, name, |mi| {
                        &mi.structs
                    })
                }
                DefKind::Trait => {
                    keep_single_definition(&mut cached_traits, &items, &order, kind, name, |mi| {
                        &mi.traits
                    })
                }
                DefKind::TypeAlias => keep_single_definition(
                    &mut cached_type_aliases,
                    &items,
                    &order,
                    kind,
                    name,
                    |mi| &mi.type_aliases,
                ),
                DefKind::Fn | DefKind::Const | DefKind::Static | DefKind::AssocFn => {}
            }
        }

        let mut resolver = NameResolver {
            items,
            modules,
            module_paths,
            imports: HashMap::new(),
            core_module,
            cached_primitives,
            cached_structs,
            cached_traits,
            cached_type_aliases,
            cached_consts,
            cached_statics,
            cached_struct_impls,
            cached_trait_impls,
            visible_consts: RefCell::new(HashMap::new()),
        };

        // Phase 2: Process `use` statements to build import maps. Walks
        // nested submodules too so their `use` statements feed the same
        // flat import map as the parent (consistent with Phase 1).
        for (module_name, module_ast) in module_asts {
            let mut module_imports = ModuleImports::default();
            if let Some(content) = &module_ast.content {
                resolver.collect_use_imports(module_name, &[], &content.1, &mut module_imports);
            }
            resolver.imports.insert(module_name.clone(), module_imports);
        }
        Ok(resolver)
    }

    // -----------------------------------------------------------------------
    // Path resolution (the new API)
    // -----------------------------------------------------------------------

    /// Resolve a `syn::Path` to a [`Res`] in the context of `calling_module`.
    ///
    /// Handles:
    /// - Unqualified: `reshape` → local → imports → core → global
    /// - Qualified: `core::reshape`, `crate::a::tools::SCALE` → look in the
    ///   module the qualifier names (see [`Self::module_for_path`]); a
    ///   qualifier that names no loaded module resolves only `Type::method`
    pub fn resolve_path(&self, path: &syn::Path, calling_module: &str) -> Res {
        let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();

        match segments.len() {
            0 => Res::Err,
            1 => {
                // Unqualified name: resolve through scope chain.
                let name = &segments[0];
                self.resolve_unqualified(name, calling_module)
            }
            _ => {
                // `module::item` (the module named as `resolve_scoped` in
                // `CUDATileModules` names it) or `Type::method`.
                let (name, qualifier) = segments.split_last().expect("two or more segments");
                if qualifier[0] == "super" && self.super_in_nested_mod(calling_module) {
                    // The JIT does not know which `mod` a `super::` path is
                    // written in: only a name one loaded module defines.
                    return match self.modules_defining(name, None).as_slice() {
                        [module] => self.resolve_in_module(name, module).unwrap_or(Res::Err),
                        _ => Res::Err,
                    };
                }
                if let Some(module) = self.module_for_path(qualifier, calling_module) {
                    // A qualifier that names a module never falls back to
                    // another module's item of the same name.
                    return match self.resolve_in_scope(name, &module, None) {
                        Ok(Some(def)) => self
                            .resolve_in_module(&def.name, &def.module)
                            .unwrap_or(Res::Err),
                        Ok(None) | Err(_) => Res::Err,
                    };
                }
                // Try as Type::method (associated function).
                let ty = &qualifier[qualifier.len() - 1];
                if let Some((module, _, _method)) = self.find_method(ty, name) {
                    return Res::Def(
                        DefKind::AssocFn,
                        DefId {
                            module: module.to_string(),
                            name: name.clone(),
                        },
                    );
                }
                // A qualified marker path like `ftz::Enabled`, or a module
                // the JIT has no source for: the compiler handles markers
                // specially via UserType.
                Res::Err
            }
        }
    }

    /// Resolve an unqualified name through the scope chain:
    /// local module → imports → core → global fallback. The name is
    /// `Res::Err` when a nested `mod`'s item makes it ambiguous (see
    /// [`Self::nested_item_conflict`]).
    fn resolve_unqualified(&self, name: &str, calling_module: &str) -> Res {
        // 1-2. Local definition, then an import (renames and re-exports
        // followed).
        if let Ok(Some(def)) = self.resolve_in_scope(name, calling_module, None) {
            if self
                .nested_item_conflict(name, calling_module, None, &def)
                .is_some()
            {
                return Res::Err;
            }
            if let Some(res) = self.resolve_in_module(&def.name, &def.module) {
                return res;
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

    /// Look up a name in a module's items, returning a Res: `Res::Err` for a
    /// name the module defines twice.
    fn lookup_in_items(name: &str, module: &str, mi: &ModuleItems) -> Option<Res> {
        find(mi, None, name).map(|(kind, _)| {
            if mi.duplicate_error(kind, name, module).is_some() {
                return Res::Err;
            }
            Res::Def(
                kind,
                DefId {
                    module: module.to_string(),
                    name: name.to_string(),
                },
            )
        })
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
    /// module's `N` replaced the kernel's own `N` in its tile shapes). So is
    /// a name that a nested `mod`'s const and an import or core share (see
    /// `nested_item_conflict`).
    pub fn resolve_const(&self, name: &str, calling_module: &str) -> Option<&ItemConst> {
        let in_module = |module: &str| self.items.get(module).and_then(|mi| mi.consts.get(name));
        let kind = Some(DefKind::Const);
        if let Ok(Some(def)) = self.resolve_in_scope(name, calling_module, kind) {
            if self
                .nested_item_conflict(name, calling_module, kind, &def)
                .is_some()
            {
                return None;
            }
            return self.get_const(&def);
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

    // -----------------------------------------------------------------------
    // Function resolution
    // -----------------------------------------------------------------------

    /// Look `name` up in `module`'s scope: its own items (nested submodules
    /// included; only `pub` ones when reached `via_glob`), then a name it
    /// imports, followed to the module it names under its original name
    /// (`use a::f as g`), then its glob imports. Only exported imports are
    /// followed when reached through another module's glob. Two globs that bring
    /// in different items make the name ambiguous, an error.
    fn lookup_scoped(
        &self,
        name: &str,
        module: &str,
        kind: Option<DefKind>,
        via_glob: bool,
        visited: &mut HashSet<(String, String, bool)>,
    ) -> Result<Option<DefId>, String> {
        // Cycle guard: imports may loop (`use super::*` both ways).
        if !visited.insert((module.to_string(), name.to_string(), via_glob)) {
            return Ok(None);
        }
        let Some(mi) = self.items.get(module) else {
            return Ok(None);
        };
        if let Some((found, public)) = find(mi, kind, name) {
            if !public && via_glob {
                return Ok(None);
            }
            if let Some(error) = mi.duplicate_error(found, name, module) {
                return Err(error);
            }
            return Ok(Some(DefId {
                module: module.to_string(),
                name: name.to_string(),
            }));
        }
        self.lookup_imported(name, module, kind, via_glob, visited)
    }

    /// [`Self::lookup_scoped`] past `module`'s own items: the names it
    /// imports, then its glob imports.
    fn lookup_imported(
        &self,
        name: &str,
        module: &str,
        kind: Option<DefKind>,
        via_glob: bool,
        visited: &mut HashSet<(String, String, bool)>,
    ) -> Result<Option<DefId>, String> {
        let Some(module_imports) = self.imports.get(module) else {
            return Ok(None);
        };
        if module_imports.conflicts.contains(name) {
            return Err(format!(
                "`{name}` is imported from different places in nested `mod`s of module \
                 `{module}`, which share one scope in the cuTile JIT; rename one import"
            ));
        }
        if let Some(import) = module_imports.names.get(name) {
            if via_glob && !module_imports.public_names.contains(name) {
                return Ok(None);
            }
            let source = import.source.as_str();
            if let Some(def) = self.lookup_scoped(&import.original, source, kind, false, visited)? {
                return Ok(Some(def));
            }
            // A function imported from a module the JIT has no source for
            // is not another module's function of the same name.
            if kind == Some(DefKind::Fn)
                && !matches!(source, "super" | "crate")
                && !self.items.contains_key(source)
            {
                return Err(format!(
                    "`{name}` is imported from `{source}`, which is not a `#[cutile::module]` \
                     the cuTile JIT loaded; import it from its cuTile module"
                ));
            }
        }
        let mut found: Vec<DefId> = Vec::new();
        for (source, nested, public) in &module_imports.globs {
            if via_glob && !public {
                continue;
            }
            let def = if nested.is_empty() {
                self.lookup_scoped(name, source, kind, true, visited)?
            } else {
                self.lookup_in_nested_mod(name, source, nested, kind)?
            };
            if let Some(def) = def {
                if !found.contains(&def) {
                    found.push(def);
                }
            }
        }
        if found.len() > 1 {
            let mut modules: Vec<String> =
                found.iter().map(|d| format!("`{}`", d.module)).collect();
            modules.sort();
            return Err(format!(
                "`{name}` is ambiguous in module `{module}`: glob imports bring it in from \
                 modules {}; import it by name or name it through its module",
                modules.join(", ")
            ));
        }
        Ok(found.pop())
    }

    /// The `pub` item `name` that the `mod` at `nested` in `module` declares
    /// itself, for a glob of that `mod` (`use a::m::inner::*`). Nested
    /// `mod`s share the module's flat scope, so the module's other items,
    /// and the `use`s of any of its `mod`s, are not the `mod`'s.
    fn lookup_in_nested_mod(
        &self,
        name: &str,
        module: &str,
        nested: &[String],
        kind: Option<DefKind>,
    ) -> Result<Option<DefId>, String> {
        let Some(mi) = self.items.get(module) else {
            return Ok(None);
        };
        let Some((found, true)) = find(mi, kind, name) else {
            return Ok(None);
        };
        if mi
            .nested_items
            .get(&(found, name.to_string()))
            .map(Vec::as_slice)
            != Some(nested)
        {
            return Ok(None);
        }
        if let Some(error) = mi.duplicate_error(found, name, module) {
            return Err(error);
        }
        Ok(Some(DefId {
            module: module.to_string(),
            name: name.to_string(),
        }))
    }

    /// The item `name` in `module`'s scope (see [`Self::lookup_scoped`]),
    /// of `kind`, or of any kind for `None`.
    pub fn resolve_in_scope(
        &self,
        name: &str,
        module: &str,
        kind: Option<DefKind>,
    ) -> Result<Option<DefId>, String> {
        self.lookup_scoped(name, module, kind, false, &mut HashSet::new())
    }

    /// The error for `name`, written unqualified in `module` and found
    /// there as `def`, when `def` is an item only a `mod` nested in the
    /// module declares, and that the module does not import by name, while
    /// an import, a glob import or core also has the name. Rust takes the
    /// nested item inside its `mod` and the other one outside it, and the
    /// JIT does not track which `mod` holds the code.
    fn nested_item_conflict(
        &self,
        name: &str,
        module: &str,
        kind: Option<DefKind>,
        def: &DefId,
    ) -> Option<String> {
        if def.module != module || def.name != name {
            return None;
        }
        let mi = self.items.get(module)?;
        let (found, _) = find(mi, kind, name)?;
        let nested = mi.nested_items.get(&(found, name.to_string()))?;
        // `use self::inner::f;` at the module's top level brings the nested
        // item in by name there. The same `use` inside another nested `mod`
        // does not: top-level code still means the import or core's item.
        let imports = self.imports.get(module);
        if imports.is_some_and(|i| i.top_level_own.contains(name)) {
            return None;
        }
        let mut visited = HashSet::from([(module.to_string(), name.to_string(), false)]);
        let core = self.core_module.as_deref().filter(|core| *core != module);
        let other = match self.lookup_imported(name, module, kind, false, &mut visited) {
            Ok(None) => match core.map(|core| self.resolve_in_scope(name, core, kind)) {
                Some(Ok(Some(other))) => other,
                Some(Err(error)) => return Some(error),
                Some(Ok(None)) | None => return None,
            },
            Ok(Some(other)) => other,
            Err(error) => return Some(error),
        };
        if other == *def {
            return None;
        }
        Some(format!(
            "`{name}` in module `{module}` names both `{}::{name}`, which a nested `mod` \
             declares, and `{}::{}`, which an import or core provides; the cuTile JIT does \
             not track which `mod` holds the code, so name one by its path or rename it",
            nested.join("::"),
            other.module,
            other.name
        ))
    }

    /// Resolve an unqualified function or struct name from `calling_module`:
    /// its own items and imports, then the implicit core module, then the
    /// only loaded module that defines it. Several such modules are an
    /// error that names them, never an arbitrary pick, and so is a name
    /// that both a nested `mod`'s item and an import or core have (see
    /// `nested_item_conflict`).
    pub fn resolve_unqualified_in_scope(
        &self,
        name: &str,
        calling_module: &str,
        kind: DefKind,
    ) -> Result<Option<DefId>, String> {
        if let Some(def) = self.resolve_in_scope(name, calling_module, Some(kind))? {
            if let Some(error) = self.nested_item_conflict(name, calling_module, Some(kind), &def) {
                return Err(error);
            }
            return Ok(Some(def));
        }
        if let Some(core) = &self.core_module {
            if core != calling_module {
                if let Some(def) = self.resolve_in_scope(name, core, Some(kind))? {
                    return Ok(Some(def));
                }
            }
        }
        match self.modules_defining(name, Some(kind)).as_slice() {
            [] => Ok(None),
            [module] => self.defined_item(module, name, kind).map(Some),
            modules => Err(format!(
                "`{name}` is defined in several modules (`{}`); import it with `use` or name \
                 it through its module",
                modules.join("`, `")
            )),
        }
    }

    /// `name` as the item of `kind` that `module`, one of
    /// [`Self::modules_defining`], defines: an error when the module defines
    /// it twice (see [`ModuleItems::duplicates`]).
    pub fn defined_item(&self, module: &str, name: &str, kind: DefKind) -> Result<DefId, String> {
        let duplicate = self.items.get(module).and_then(|mi| {
            let (found, _) = find(mi, Some(kind), name)?;
            mi.duplicate_error(found, name, module)
        });
        match duplicate {
            Some(error) => Err(error),
            None => Ok(DefId {
                module: module.to_string(),
                name: name.to_string(),
            }),
        }
    }

    /// The loaded modules that define `name` as an item of `kind`, or of
    /// any kind for `None`, sorted.
    pub fn modules_defining(&self, name: &str, kind: Option<DefKind>) -> Vec<&str> {
        let mut modules: Vec<&str> = self
            .items
            .iter()
            .filter(|(_, mi)| find(mi, kind, name).is_some())
            .map(|(module, _)| module.as_str())
            .collect();
        modules.sort();
        modules
    }

    /// The loaded module that the path qualifier `qualifier` names from
    /// `calling_module`, in order:
    /// 1. the calling module itself when the qualifier (after a leading
    ///    `self`) is the path of a `mod` nested in it: nested `mod`s share its
    ///    scope (see [`Self::build`]);
    /// 2. a module its first segment names, a module the caller imports
    ///    (`use a::m;`, `use a::n as m;`) or a loaded module, with the rest
    ///    of the qualifier a `mod` path nested in that module;
    /// 3. the calling module when the qualifier ends a nested `mod` path, as
    ///    `self::sub` does from inside `mod inner` for `inner::sub`;
    /// 4. with `by_last_segment`, the loaded module the last segment names.
    pub fn module_for_qualifier(
        &self,
        qualifier: &[String],
        calling_module: &str,
        by_last_segment: bool,
    ) -> Option<String> {
        let path = match qualifier {
            [first, rest @ ..] if first == "self" && !rest.is_empty() => rest,
            path => path,
        };
        if self.has_nested_mod_path(calling_module, path) {
            return Some(calling_module.to_string());
        }
        let (first, rest) = path.split_first()?;
        let imported = self
            .imports
            .get(calling_module)
            .and_then(|module_imports| module_imports.names.get(first));
        let head = match imported.and_then(|import| import.module.clone()) {
            Some(module) => Some(module),
            None if self.items.contains_key(first) => Some((first.clone(), Vec::new())),
            None => None,
        };
        if let Some((module, mut nested)) = head {
            nested.extend_from_slice(rest);
            return (nested.is_empty() || self.has_nested_mod_path(&module, &nested))
                .then_some(module);
        }
        if self
            .items
            .get(calling_module)
            .is_some_and(|mi| mi.nested_mods.iter().any(|nested| nested.ends_with(path)))
        {
            return Some(calling_module.to_string());
        }
        let last = path.last().filter(|_| by_last_segment)?;
        let keyword = matches!(last.as_str(), "crate" | "self" | "super");
        (!keyword && self.items.contains_key(last)).then(|| last.clone())
    }

    /// The loaded module that `qualifier`, the path before an item's name
    /// written in `calling_module`, names: for a `crate::`, `self::` or
    /// `super::` path, or a path of several segments, the loaded module at
    /// that absolute path, or the one holding a `mod` there; otherwise (or
    /// failing that) [`Self::module_for_qualifier`]. A relative path's last
    /// segment alone would name an unrelated module, so it is read only for
    /// a caller without an absolute path.
    pub fn module_for_path(&self, qualifier: &[String], calling_module: &str) -> Option<String> {
        let first = qualifier.first()?;
        let relative = matches!(first.as_str(), "crate" | "self" | "super");
        let caller_path = self.module_path(calling_module);
        if relative || qualifier.len() > 1 {
            let path = qualifier.join("::");
            let absolute = if relative {
                crate::compiler::_module::resolve_relative_use_path(&path, caller_path)
            } else {
                path
            };
            if let Some((module, _)) = self.module_at(&absolute) {
                return Some(module);
            }
        }
        let by_last_segment = !relative || !caller_path.contains("::");
        self.module_for_qualifier(qualifier, calling_module, by_last_segment)
    }

    /// Whether `path` names `mod`s nested in `module`, outermost first.
    pub fn has_nested_mod_path(&self, module: &str, path: &[String]) -> bool {
        self.items
            .get(module)
            .is_some_and(|mi| mi.nested_mods.contains(path))
    }

    /// `module`'s absolute Rust path (`module_path!()`).
    pub fn module_path<'a>(&'a self, module: &'a str) -> &'a str {
        self.module_paths.get(module).map_or(module, String::as_str)
    }

    /// The loaded module at the absolute module path `absolute`, or the
    /// innermost one holding a `mod` there, with that `mod`'s path in it
    /// (nested `mod`s share the module's scope, see [`Self::build`]).
    pub fn module_at(&self, absolute: &str) -> Option<(String, Vec<String>)> {
        let mut found: Option<(String, Vec<String>)> = None;
        for module in self.items.keys() {
            let path = self.module_path(module);
            let nested: Vec<String> = match absolute.strip_prefix(path) {
                Some("") => Vec::new(),
                Some(rest) => match rest.strip_prefix("::") {
                    Some(rest) => rest.split("::").map(str::to_string).collect(),
                    None => continue,
                },
                None => continue,
            };
            if !nested.is_empty() && !self.has_nested_mod_path(module, &nested) {
                continue;
            }
            if found
                .as_ref()
                .is_none_or(|(_, known)| nested.len() < known.len())
            {
                found = Some((module.clone(), nested));
            }
        }
        found
    }

    /// The loaded module, and the `mod` path in it, that `path` names when a
    /// `use` in the `mod` at `nested` in `module` spells it: a `crate::`,
    /// `self::` or `super::` path from that `mod`, a `mod` nested in it, or
    /// an absolute path. Failing those, a module named by the last segment
    /// that is a module name, with the rest a `mod` path in it: only a
    /// module whose short name no other loaded module has is known by it
    /// (see `CUDATileModules::new`), so this never picks one of two modules
    /// that share a name.
    fn module_named_in_use(
        &self,
        module: &str,
        nested: &[String],
        path: &[String],
    ) -> Option<(String, Vec<String>)> {
        let first = path.first()?;
        let found = if matches!(first.as_str(), "crate" | "self" | "super") {
            let scope: Vec<&str> = std::iter::once(self.module_path(module))
                .chain(nested.iter().map(String::as_str))
                .collect();
            self.module_at(&crate::compiler::_module::resolve_relative_use_path(
                &path.join("::"),
                &scope.join("::"),
            ))
        } else {
            let in_scope: Vec<String> = nested.iter().chain(path).cloned().collect();
            if self.has_nested_mod_path(module, &in_scope) {
                Some((module.to_string(), in_scope))
            } else {
                self.module_at(&path.join("::"))
            }
        };
        found.or_else(|| {
            path.iter().enumerate().rev().find_map(|(at, segment)| {
                let nested = &path[at + 1..];
                (self.items.contains_key(segment)
                    && (nested.is_empty() || self.has_nested_mod_path(segment, nested)))
                .then(|| (segment.clone(), nested.to_vec()))
            })
        })
    }

    /// Whether code in a `mod` nested in `module` names a `super::` path.
    pub fn super_in_nested_mod(&self, module: &str) -> bool {
        self.items
            .get(module)
            .is_some_and(|mi| mi.super_in_nested_mod)
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
    // Use-tree processing
    // -----------------------------------------------------------------------

    /// Record the names a `use` tree in the `mod` at `nested` in `module`
    /// imports. `use a::m::f;` maps `f` to `f` in module `a::m`;
    /// `use a::m::f as g;` maps `g` to it; `use a::m;` maps `m` to `m` in
    /// `a`, which [`Self::module_for_qualifier`] reads as the module `a::m`;
    /// `use a::m::*;` records a glob of `a::m`.
    fn process_use_tree(
        &self,
        module: &str,
        nested: &[String],
        tree: &UseTree,
        path_prefix: &[String],
        public: bool,
        imports: &mut ModuleImports,
    ) {
        match tree {
            UseTree::Path(path) => {
                let mut prefix = path_prefix.to_vec();
                prefix.push(path.ident.to_string());
                self.process_use_tree(module, nested, &path.tree, &prefix, public, imports);
            }
            UseTree::Name(name) => {
                let target = self.use_target(module, nested, path_prefix, &name.ident.to_string());
                let local = if name.ident == "self" {
                    target.original.clone()
                } else {
                    name.ident.to_string()
                };
                imports.insert(module, nested, local, target, public);
            }
            UseTree::Rename(rename) => {
                let target =
                    self.use_target(module, nested, path_prefix, &rename.ident.to_string());
                imports.insert(module, nested, rename.rename.to_string(), target, public);
            }
            UseTree::Glob(_) => {
                // A glob of the module's own scope (`use super::*` in a
                // nested `mod`) brings in nothing new.
                let (source, in_source) = self
                    .module_named_in_use(module, nested, path_prefix)
                    .unwrap_or_else(|| {
                        let last = path_prefix.last().cloned();
                        (last.unwrap_or_else(|| "crate".into()), Vec::new())
                    });
                if !path_prefix.is_empty() && source != module {
                    imports.globs.push((source, in_source, public));
                }
            }
            UseTree::Group(group) => {
                for tree in &group.items {
                    self.process_use_tree(module, nested, tree, path_prefix, public, imports);
                }
            }
        }
    }

    /// What `use prefix::name` in the `mod` at `nested` in `module`
    /// imports. `self` in `use a::m::{self as alias}` names `m` in `a`, not
    /// an item called `self` in `m`.
    fn use_target(&self, module: &str, nested: &[String], prefix: &[String], name: &str) -> Import {
        let (prefix, name) = if name == "self" {
            let (imported, parents) = prefix
                .split_last()
                .expect("a grouped self import has a prefix");
            (parents, imported.as_str())
        } else {
            (prefix, name)
        };
        let mut path = prefix.to_vec();
        path.push(name.to_string());
        Import {
            source: self.use_source(module, nested, prefix),
            original: name.to_string(),
            module: self.module_named_in_use(module, nested, &path),
        }
    }

    /// The loaded module a `use` prefix names (see
    /// [`Self::module_named_in_use`]), or else its last segment (`crate`
    /// for an empty prefix).
    fn use_source(&self, module: &str, nested: &[String], prefix: &[String]) -> String {
        match self.module_named_in_use(module, nested, prefix) {
            Some((source, _)) => source,
            None => prefix.last().cloned().unwrap_or_else(|| "crate".into()),
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

    fn fn_module(resolved: Result<Option<DefId>, String>) -> String {
        resolved
            .expect("resolution error")
            .expect("function not found")
            .module
    }

    /// Two modules may define the same function name: each resolves in its
    /// own scope, a module that imports one gets that one, and a module that
    /// does neither gets an error naming both.
    #[test]
    fn duplicate_function_names_resolve_per_module() {
        let (a, a_mod) = make_module("mod_a", vec![parse_quote! { pub fn dup() -> i32 { 1 } }]);
        let (b, b_mod) = make_module("mod_b", vec![parse_quote! { pub fn dup() -> i32 { 2 } }]);
        let (c, c_mod) = make_module("mod_c", vec![parse_quote! { use super::mod_b::dup; }]);
        let (d, d_mod) = make_module("mod_d", vec![parse_quote! { fn other() {} }]);
        let resolver =
            NameResolver::build(&[(a, a_mod), (b, b_mod), (c, c_mod), (d, d_mod)]).unwrap();
        let resolve = |module| resolver.resolve_unqualified_in_scope("dup", module, DefKind::Fn);

        assert_eq!(fn_module(resolve("mod_a")), "mod_a");
        assert_eq!(fn_module(resolve("mod_b")), "mod_b");
        assert_eq!(fn_module(resolve("mod_c")), "mod_b");
        let err = resolve("mod_d").unwrap_err();
        assert!(
            err.contains("several modules") && err.contains("`mod_a`, `mod_b`"),
            "the error must name both modules: {err}"
        );
    }

    /// `use m::f as g` resolves `g` to `f` in `m`, and `use a::m as x`
    /// makes `x::f` name `m`'s `f`.
    #[test]
    fn renamed_function_and_module_imports() {
        let (m, m_mod) = make_module("lib_m", vec![parse_quote! { pub fn f() -> i32 { 1 } }]);
        let (k, k_mod) = make_module(
            "kernel_mod",
            vec![
                parse_quote! { use super::lib_m::f as g; },
                parse_quote! { use super::lib_m as x; },
                parse_quote! { fn f() -> i32 { 2 } },
            ],
        );
        let resolver = NameResolver::build(&[(m, m_mod), (k, k_mod)]).unwrap();

        let g = resolver
            .resolve_unqualified_in_scope("g", "kernel_mod", DefKind::Fn)
            .unwrap()
            .unwrap();
        assert_eq!((g.module.as_str(), g.name.as_str()), ("lib_m", "f"));
        assert_eq!(
            fn_module(resolver.resolve_unqualified_in_scope("f", "kernel_mod", DefKind::Fn)),
            "kernel_mod"
        );
        assert_eq!(
            resolver
                .module_for_qualifier(&["x".into()], "kernel_mod", true)
                .as_deref(),
            Some("lib_m")
        );
    }

    /// A qualifier names the caller's nested `mod` only by its full path
    /// (or a path's tail, after the imports); an imported module wins over
    /// a deeper `mod` of that name, and an unknown head is never skipped.
    #[test]
    fn qualifiers_name_nested_mods_by_their_path() {
        let (lib, lib_mod) = make_module(
            "lib",
            vec![
                parse_quote! { pub fn f() -> i32 { 1 } },
                parse_quote! { pub mod inner { pub fn g() -> i32 { 3 } } },
            ],
        );
        let (k, k_mod) = make_module(
            "kernel_mod",
            vec![
                parse_quote! { use super::lib; },
                parse_quote! { use super::lib::inner; },
                parse_quote! { mod a { pub mod lib { pub fn h() {} } } },
                parse_quote! { mod helpers { pub fn f() -> i32 { 2 } } },
            ],
        );
        let (inner, inner_mod) =
            make_module("inner", vec![parse_quote! { pub fn g() -> i32 { 4 } }]);
        let resolver =
            NameResolver::build(&[(lib, lib_mod), (k, k_mod), (inner, inner_mod)]).unwrap();
        let module = |qualifier: &[&str]| {
            let qualifier: Vec<String> = qualifier.iter().map(|s| s.to_string()).collect();
            resolver.module_for_qualifier(&qualifier, "kernel_mod", true)
        };
        assert_eq!(module(&["lib"]).as_deref(), Some("lib"));
        assert_eq!(module(&["helpers"]).as_deref(), Some("kernel_mod"));
        assert_eq!(module(&["self", "helpers"]).as_deref(), Some("kernel_mod"));
        assert_eq!(module(&["a", "lib"]).as_deref(), Some("kernel_mod"));
        // `inner` is imported as `lib`'s nested mod, not the loaded `inner`.
        assert_eq!(module(&["inner"]).as_deref(), Some("lib"));
        assert_eq!(module(&["lib", "inner"]).as_deref(), Some("lib"));
        assert_eq!(module(&["other", "helpers"]), None);
        assert_eq!(module(&["lib", "missing"]), None);
    }

    /// A function imported from a module the JIT has no source for is an
    /// error, not another loaded module's function of that name. Imports
    /// from `self` or a nested `mod` stay in the module's own scope.
    #[test]
    fn imports_from_unloaded_modules_are_errors() {
        let (lib, lib_mod) = make_module("lib", vec![parse_quote! { pub fn scale() {} }]);
        let (k, k_mod) = make_module(
            "kernel_mod",
            vec![
                parse_quote! { use super::plain::scale; },
                parse_quote! { use inner::f as g; },
                parse_quote! { use self::h as h2; },
                parse_quote! { mod inner { pub fn f() {} } },
                parse_quote! { fn h() {} },
            ],
        );
        let resolver = NameResolver::build(&[(lib, lib_mod), (k, k_mod)]).unwrap();
        let err = resolver
            .resolve_unqualified_in_scope("scale", "kernel_mod", DefKind::Fn)
            .unwrap_err();
        assert!(err.contains("`plain`"), "{err}");
        for (local, original) in [("g", "f"), ("h2", "h")] {
            let def = resolver
                .resolve_unqualified_in_scope(local, "kernel_mod", DefKind::Fn)
                .unwrap()
                .unwrap();
            assert_eq!(
                (def.module.as_str(), def.name.as_str()),
                ("kernel_mod", original)
            );
        }
    }

    /// Nested `mod`s share one scope: a name two of them import from
    /// different places is an error, not the last `use`. A nested `mod`'s
    /// `use super::f` names the shared scope's `f`, so it is not one.
    #[test]
    fn conflicting_imports_in_nested_mods_are_errors() {
        let (a, a_mod) = make_module("lib_a", vec![parse_quote! { pub fn f() {} }]);
        let (b, b_mod) = make_module("lib_b", vec![parse_quote! { pub fn f() {} }]);
        let (k, k_mod) = make_module(
            "kernel_mod",
            vec![
                parse_quote! { use super::lib_a::f; },
                parse_quote! { mod m { use super::super::lib_b::f; } },
            ],
        );
        let (c, c_mod) = make_module(
            "kernel_c",
            vec![
                parse_quote! { use super::lib_a::f; },
                parse_quote! { mod m { use super::f; } },
            ],
        );
        let resolver =
            NameResolver::build(&[(a, a_mod), (b, b_mod), (k, k_mod), (c, c_mod)]).unwrap();
        let err = resolver
            .resolve_unqualified_in_scope("f", "kernel_mod", DefKind::Fn)
            .unwrap_err();
        assert!(err.contains("nested `mod`s"), "{err}");
        assert_eq!(
            fn_module(resolver.resolve_unqualified_in_scope("f", "kernel_c", DefKind::Fn)),
            "lib_a"
        );
    }

    /// A `use` is followed through every module that re-exports the name,
    /// and a same-named struct elsewhere does not interfere.
    #[test]
    fn reexport_chains_are_followed() {
        let (a, a_mod) = make_module("st_lib", vec![parse_quote! { pub struct Lower; }]);
        let (o, o_mod) = make_module("st_other", vec![parse_quote! { pub struct Lower; }]);
        let (w, w_mod) = make_module(
            "st_wrap",
            vec![parse_quote! { pub use super::st_lib::Lower; }],
        );
        let (h, h_mod) = make_module("st_hub", vec![parse_quote! { pub use super::st_wrap::*; }]);
        let (k, k_mod) = make_module(
            "st_kernel",
            vec![parse_quote! { use super::st_hub::Lower; }],
        );
        let resolver =
            NameResolver::build(&[(a, a_mod), (o, o_mod), (w, w_mod), (h, h_mod), (k, k_mod)])
                .unwrap();

        let def = resolver
            .resolve_unqualified_in_scope("Lower", "st_kernel", DefKind::Struct)
            .unwrap()
            .unwrap();
        assert_eq!(
            (def.module.as_str(), def.name.as_str()),
            ("st_lib", "Lower")
        );
        assert_eq!(
            resolver.resolve_in_scope("Lower", "st_kernel", Some(DefKind::Fn)),
            Ok(None)
        );
    }

    /// Two glob imports that bring in different functions of one name make
    /// an unqualified call ambiguous; a named import wins, and a private
    /// item is not glob-imported.
    #[test]
    fn glob_imports_of_one_name_are_ambiguous() {
        let (a, a_mod) = make_module("glob_a", vec![parse_quote! { pub fn f() {} }]);
        let (b, b_mod) = make_module("glob_b", vec![parse_quote! { pub fn f() {} }]);
        let (p, p_mod) = make_module("glob_private", vec![parse_quote! { fn f() {} }]);
        let (k, k_mod) = make_module(
            "glob_k",
            vec![
                parse_quote! { use super::glob_a::*; },
                parse_quote! { use super::glob_b::*; },
            ],
        );
        let (n, n_mod) = make_module(
            "glob_named",
            vec![
                parse_quote! { use super::glob_a::*; },
                parse_quote! { use super::glob_b::*; },
                parse_quote! { use super::glob_b::f; },
            ],
        );
        let (q, q_mod) = make_module(
            "glob_pub",
            vec![
                parse_quote! { use super::glob_a::*; },
                parse_quote! { use super::glob_private::*; },
            ],
        );
        let resolver = NameResolver::build(&[
            (a, a_mod),
            (b, b_mod),
            (p, p_mod),
            (k, k_mod),
            (n, n_mod),
            (q, q_mod),
        ])
        .unwrap();
        let resolve = |module| resolver.resolve_unqualified_in_scope("f", module, DefKind::Fn);

        let err = resolve("glob_k").unwrap_err();
        assert!(err.contains("ambiguous"), "unexpected error: {err}");
        assert_eq!(fn_module(resolve("glob_named")), "glob_b");
        assert_eq!(fn_module(resolve("glob_pub")), "glob_a");
    }

    #[test]
    fn globs_follow_only_exported_imports() {
        for (import, exported) in [
            (parse_quote! { use super::source::f; }, false),
            (parse_quote! { use super::source::*; }, false),
            (parse_quote! { pub use super::source::f; }, true),
            (parse_quote! { pub use super::source::*; }, true),
        ] {
            let resolver = NameResolver::build(&[
                make_module("source", vec![parse_quote! { pub fn f() {} }]),
                make_module("wrapper", vec![import]),
                make_module("caller", vec![parse_quote! { use super::wrapper::*; }]),
            ])
            .unwrap();
            let resolve = |module| resolver.resolve_in_scope("f", module, Some(DefKind::Fn));
            assert_eq!(fn_module(resolve("wrapper")), "source");
            let from_caller = resolve("caller").unwrap();
            assert_eq!(from_caller.is_some(), exported);
            if let Some(def) = from_caller {
                assert_eq!(def.module, "source");
            }
        }
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

        assert_eq!(
            resolver.modules_defining("alpha", Some(DefKind::Fn)),
            ["mod_a"]
        );
        assert_eq!(
            resolver.modules_defining("gamma", Some(DefKind::Fn)),
            ["mod_b"]
        );
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

    /// Functions share the module's flat scope like consts: two nested
    /// `mod`s defining one name are rejected, not silently replaced.
    #[test]
    fn duplicate_functions_within_a_module_are_rejected() {
        let (name, module) = make_module(
            "dup_mod",
            vec![
                parse_quote! { pub mod a { pub fn scale() -> i32 { 5 } } },
                parse_quote! { pub mod b { pub fn scale() -> i32 { 7 } } },
            ],
        );
        let err = NameResolver::build(&[(name, module)])
            .err()
            .expect("a duplicate function must be rejected");
        assert!(
            err.to_string().contains("duplicate function `scale`"),
            "unexpected error: {err}"
        );
    }

    /// Types that a parent and a nested `mod`, or two nested `mod`s, define
    /// under one name (marker namespaces such as `ftz::Enabled` and
    /// `nan::Enabled`) do not stop the module from loading. Resolving the
    /// name is an error, and the name-only maps hold neither definition.
    #[test]
    fn duplicate_types_within_a_module_fail_only_when_resolved() {
        let (name, module) = make_module(
            "dup_mod",
            vec![
                parse_quote! { pub struct Fill; },
                parse_quote! { pub mod inner { pub struct Fill; } },
                parse_quote! { pub mod a { pub trait Mode {} pub type Elem = f32; } },
                parse_quote! { pub mod b { pub trait Mode {} pub type Elem = i32; } },
                parse_quote! { pub fn other() {} },
            ],
        );
        let (user, user_mod) = make_module("user_mod", vec![parse_quote! { fn f() {} }]);
        let resolver =
            NameResolver::build(&[(name, module), (user, user_mod)]).expect("the module loads");
        assert_eq!(
            fn_module(resolver.resolve_unqualified_in_scope("other", "dup_mod", DefKind::Fn)),
            "dup_mod"
        );
        for (name, kind, noun) in [
            ("Fill", DefKind::Struct, "struct"),
            ("Mode", DefKind::Trait, "trait"),
            ("Elem", DefKind::TypeAlias, "type alias"),
        ] {
            let expected = format!("duplicate {noun} `{name}` in module `dup_mod`");
            let err = resolver
                .resolve_in_scope(name, "dup_mod", Some(kind))
                .unwrap_err();
            assert!(err.contains(&expected), "{err}");
            let err = resolver
                .resolve_unqualified_in_scope(name, "user_mod", kind)
                .unwrap_err();
            assert!(err.contains(&expected), "{err}");
            assert!(
                matches!(
                    resolver.resolve_path(&parse_path(name), "dup_mod"),
                    Res::Err
                ),
                "{name}"
            );
        }
        assert!(!resolver.structs().contains_key("Fill"));
        assert!(!resolver.traits().contains_key("Mode"));
        assert!(!resolver.type_aliases().contains_key("Elem"));

        // Another module's single definition of the name stays, before or
        // after the module that defines it twice.
        for own_first in [true, false] {
            let (dup, dup_mod) = make_module(
                "dup_mod",
                vec![
                    parse_quote! { pub mod wide { pub struct Fill; pub trait Mode {} pub type Elem = f64; } },
                    parse_quote! { pub mod narrow { pub struct Fill; pub trait Mode {} pub type Elem = i32; } },
                ],
            );
            let (own, own_mod) = make_module(
                "own_mod",
                vec![
                    parse_quote! { struct Fill { x: f32 } },
                    parse_quote! { trait Mode { fn own(); } },
                    parse_quote! { type Elem = f32; },
                ],
            );
            let asts = if own_first {
                [(own, own_mod), (dup, dup_mod)]
            } else {
                [(dup, dup_mod), (own, own_mod)]
            };
            let resolver = NameResolver::build(&asts).unwrap();
            let elem = &resolver.type_aliases()["Elem"].ty;
            assert_eq!(elem.to_token_stream().to_string(), "f32");
            assert_eq!(resolver.structs()["Fill"].fields.len(), 1);
            assert_eq!(resolver.traits()["Mode"].items.len(), 1);
        }
    }

    /// Nested `mod` paths are recorded, and a `super::` path in a nested
    /// `mod` is flagged, but not `use super::*` or `pub(super)`.
    #[test]
    fn nested_mods_and_their_super_paths() {
        let (name, module) = make_module(
            "nest_mod",
            vec![parse_quote! {
                pub mod a {
                    use super::*;
                    pub(super) mod b { pub(super) fn deep() {} }
                }
            }],
        );
        let (flagged, flagged_mod) = make_module(
            "super_mod",
            vec![parse_quote! { mod a { fn g() { super::f() } } }],
        );
        let resolver = NameResolver::build(&[(name, module), (flagged, flagged_mod)]).unwrap();
        assert!(resolver.has_nested_mod_path("nest_mod", &["a".into(), "b".into()]));
        assert!(!resolver.has_nested_mod_path("nest_mod", &["b".into()]));
        assert!(!resolver.super_in_nested_mod("nest_mod"));
        assert!(resolver.super_in_nested_mod("super_mod"));
    }

    /// A qualified const path names the module its qualifier names, by
    /// absolute path when two loaded modules share the short name `tools`:
    /// never the calling module's own same-named const.
    #[test]
    fn qualified_const_paths_name_modules_by_path() {
        let module = |path: &str, items: Vec<Item>| {
            let (_, ast) = make_module(path.rsplit("::").next().unwrap(), items);
            (path.to_string(), ast)
        };
        let asts = [
            module(
                "app::side_c::tools",
                vec![parse_quote! { pub const SCALE: i32 = 43; }],
            ),
            module(
                "app::side_d::tools",
                vec![parse_quote! { pub const SCALE: i32 = 47; }],
            ),
            module(
                "app::kernel",
                vec![
                    parse_quote! { const SCALE: i32 = 53; },
                    parse_quote! { mod local_inner { pub const INNER: i32 = 59; } },
                ],
            ),
        ];
        let paths = asts.iter().map(|(n, _)| (n.clone(), n.clone())).collect();
        let resolver = NameResolver::build_with_paths(&asts, paths).unwrap();
        let resolve = |path: &str| match resolver.resolve_path(&parse_path(path), "app::kernel") {
            Res::Def(DefKind::Const, def) => {
                let item = resolver.get_const(&def).unwrap();
                Some(crate::type_aliases::const_item_i32_value(item).unwrap())
            }
            _ => None,
        };
        assert_eq!(resolve("crate::side_c::tools::SCALE"), Some(43));
        assert_eq!(resolve("crate::side_d::tools::SCALE"), Some(47));
        assert_eq!(resolve("app::side_d::tools::SCALE"), Some(47));
        assert_eq!(resolve("SCALE"), Some(53));
        assert_eq!(resolve("self::SCALE"), Some(53));
        assert_eq!(resolve("self::local_inner::INNER"), Some(59));
        assert_eq!(resolve("crate::side_e::tools::SCALE"), None);
        assert_eq!(resolve("crate::side_c::tools::MISSING"), None);
    }

    /// A glob of a `mod` nested in a loaded module brings in that `mod`'s
    /// items only, not the whole module's flat scope: `exp`, which the
    /// module declares outside the `mod`, stays core's.
    #[test]
    fn globs_of_nested_mods_import_only_their_items() {
        let resolver = NameResolver::build(&[
            make_module(
                "fastmath",
                vec![
                    parse_quote! { pub fn exp() {} },
                    parse_quote! { pub mod activations { pub fn relu() {} fn hidden() {} } },
                ],
            ),
            make_module("core", vec![parse_quote! { pub fn exp() {} }]),
            make_module(
                "kernel_mod",
                vec![
                    parse_quote! { use super::core::*; },
                    parse_quote! { use super::fastmath::activations::*; },
                ],
            ),
        ])
        .unwrap();
        let resolve = |name| resolver.resolve_unqualified_in_scope(name, "kernel_mod", DefKind::Fn);
        assert_eq!(fn_module(resolve("exp")), "core");
        assert_eq!(fn_module(resolve("relu")), "fastmath");
        assert_eq!(
            resolver.resolve_in_scope("hidden", "kernel_mod", Some(DefKind::Fn)),
            Ok(None)
        );
    }

    /// `use super::approx::exp;` inside nested `mod fast` names the nested
    /// `exp` there only: a plain `exp` stays ambiguous with core's. The
    /// same `use` at the module's top level picks the nested `exp`.
    #[test]
    fn only_top_level_uses_pick_a_nested_item() {
        let kernel = |import: Item| {
            make_module(
                "kernel_mod",
                vec![
                    parse_quote! { use super::core::*; },
                    parse_quote! { mod approx { pub fn exp() {} } },
                    import,
                ],
            )
        };
        let core = || make_module("core", vec![parse_quote! { pub fn exp() {} }]);
        let in_fast = NameResolver::build(&[
            core(),
            kernel(parse_quote! { mod fast { use super::approx::exp; } }),
        ])
        .unwrap();
        let err = in_fast
            .resolve_unqualified_in_scope("exp", "kernel_mod", DefKind::Fn)
            .unwrap_err();
        assert!(
            err.contains("`approx::exp`") && err.contains("`core::exp`"),
            "{err}"
        );
        let top_level =
            NameResolver::build(&[core(), kernel(parse_quote! { use self::approx::exp; })])
                .unwrap();
        assert_eq!(
            fn_module(top_level.resolve_unqualified_in_scope("exp", "kernel_mod", DefKind::Fn)),
            "kernel_mod"
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
