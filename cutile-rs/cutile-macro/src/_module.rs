/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module-macro orchestration.
//!
//! Walks the items inside `#[cutile::module]`, dispatches each to the right
//! transformer, and stitches the result back into a `pub mod` body.
//!
//! ## Two-track design: macro output is for rustc; the JIT re-parses original source
//!
//! The macro emits Rust code (rank-instance struct/impl specializations,
//! shadow traits + rank-instance impls + free-fn wrappers, kernel
//! launchers) that **only rustc consumes**. None of this expanded code is
//! visible to the JIT compiler at kernel-compilation time.
//!
//! Instead, [`module_asts`] (below) captures the **original pre-expansion
//! source text** via `Span::source_text()` and stores it as a string literal.
//! At runtime, `_module_asts()` re-parses that string with `syn::parse_str`
//! and hands the resulting AST to the JIT. The JIT works from the user's
//! original generic functions (e.g. `pub fn store_tile<E, const S: [i32; N]>(...)`),
//! not from the macro-emitted rank-instance specializations.
//!
//! Implications:
//!
//! - Anything emitted by the macro affects only rustc's type-checking and
//!   call resolution. It does not change what the JIT sees or how it
//!   instantiates.
//! - The shadow-trait synthesizer ([`crate::shadow_dispatch`]) and the rank-instance
//!   expander ([`crate::rank_instantiation`]) both serve rustc's needs (correct
//!   types and ergonomic call resolution); the JIT does its own rank-instance
//!   instantiation from the original generic source.
//! - When reasoning about JIT behavior, look at the user-source AST as
//!   captured — not at `cargo expand` output.
//!
//! ## Per-item routing
//!
//! - **`fn`** — entry kernels (`#[entry]`) get a launcher generated; ops
//!   tagged `#[cuda_tile::variadic_op]` go through shadow-dispatch synthesis;
//!   everything else is rank-instantiated for any literal CGAs in its
//!   signature.
//! - **`struct`** — `#[cuda_tile::variadic_struct]` produces the rank-instance
//!   variants (`Tile_0..Tile_6`); plain structs pass through.
//! - **`trait`** — `#[cuda_tile::variadic_trait]` desugars to a single
//!   CGA-erased shadow trait.
//! - **`impl`** — `#[cuda_tile::variadic_impl]` (with or without
//!   `#[variadic_trait_impl]`) emits rank-instance impl variants of the shadow trait
//!   trait or of the inherent type.
//! - **`use`** statements feed the cross-module AST aggregation.
//!
//! [`module_asts`] generates the runtime hook that returns the captured
//! source AST for the JIT.

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use proc_macro2::{LineColumn, Span, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::{env, fs};

use syn::{
    parse_file, parse_macro_input, parse_quote, AngleBracketedGenericArguments, GenericArgument,
    GenericParam, ItemFn, ItemImpl, ItemMod, ItemStruct, ItemTrait, ItemType, Path,
};

use crate::error::{Error, SpannedError};
use crate::kernel_launcher_generator::generate_kernel_launcher;
use crate::rank_instantiation::*;
use crate::shadow_dispatch::{
    desugar_variadic_trait_decl, desugar_variadic_trait_impl, emit_shadow_dispatch,
};
use crate::validate_dsl_syntax::validate_entry_point_parameters;
use cutile_compiler::kernel_naming::KernelNaming;
use cutile_compiler::syn_utils::*;

fn line_column_to_offset(source: &str, loc: LineColumn) -> Option<usize> {
    let mut line_start = 0usize;
    let mut current_line = 1usize;

    for line in source.split_inclusive('\n') {
        if current_line == loc.line {
            let column_offset = byte_offset_for_char_column(line, loc.column)?;
            return Some(line_start + column_offset);
        }
        line_start += line.len();
        current_line += 1;
    }

    if current_line == loc.line {
        let tail = &source[line_start..];
        let column_offset = byte_offset_for_char_column(tail, loc.column)?;
        return Some(line_start + column_offset);
    }

    None
}

fn byte_offset_for_char_column(line: &str, column: usize) -> Option<usize> {
    if column == 0 {
        return Some(0);
    }

    if column == line.chars().count() {
        return Some(line.len());
    }

    line.char_indices().nth(column).map(|(idx, _)| idx)
}

fn source_slice_from_file(path: &str, start: LineColumn, end: LineColumn) -> Option<String> {
    let source = fs::read_to_string(path).ok()?;
    let start_offset = line_column_to_offset(&source, start)?;
    let end_offset = line_column_to_offset(&source, end)?;
    source.get(start_offset..end_offset).map(str::to_string)
}

/// Returns the path to the CUDA tile AST module.
///
/// This is used throughout the code generation to reference AST types.
pub fn get_ast_path(tile_rust_crate_root: &Ident) -> Path {
    let s = format!("{tile_rust_crate_root}::cutile_compiler::ast");
    syn::parse::<Path>(s.parse().unwrap()).unwrap()
}

/// Returns the path to the linker-registry module on `cutile-compiler`.
///
/// LINKING Phase A: each module registers itself in `CUTILE_MODULES` as well
/// as emitting the legacy `_module_asts()`. This path lets macro-emitted code
/// name the registry without each downstream crate depending on `linkme`
/// directly.
pub fn get_registry_path(tile_rust_crate_root: &Ident) -> Path {
    let s = format!("{tile_rust_crate_root}::cutile_compiler::registry");
    syn::parse::<Path>(s.parse().unwrap()).unwrap()
}

/// Returns the identifier for the per-module self-only AST builder.
///
/// Per-module entry that returns *just this module's* [`Module`]. Called
/// by the linker-registry entry's `build` closure.
pub fn get_self_ast_ident() -> Ident {
    Ident::new("__module_ast_self", Span::call_site())
}

/// Main entry point for the module macro.
///
/// Transforms a Rust module containing GPU kernel code into:
/// - Concrete Rust functions (possibly expanded from variadics)
/// - MLIR AST builder functions
/// - Kernel launcher functions
///
/// ## Processing Pipeline
///
/// 1. Parse module attributes (`tile_rust_crate`)
/// 2. Iterate through module items
/// 3. For each item:
///    - Validate syntax (for functions)
///    - Generate AST representation
///    - Handle variadic expansion if needed
///    - Generate kernel launchers if `#[entry]` function
/// 4. Emit per-module AST builder + linker-registry entry
///
/// ## Attributes
///
/// - `tile_rust_crate=true` - This module is within the cutile crate
///
/// ## Generated Structure
///
/// ```rust,ignore
/// pub mod my_module {
///     // Imports and dependencies
///     use cutile_compiler::ast;
///     use cutile::tile_kernel::*;
///
///     // Per-module AST builder used by the linker-registry entry below.
///     pub fn __module_ast_self() -> Module { ... }
///
///     // Self-registers this module into the global linker-registry slice.
///     #[linkme::distributed_slice(CUTILE_MODULES)]
///     static __CUTILE_MODULE_ENTRY_FOO: CutileModuleEntry = ...;
///
///     // Concrete items (functions, structs, etc.)
///     fn __cutile_user_impl_my_kernel(...) { ... }
///
///     // Kernel launchers (for #[entry] functions)
///     pub fn my_kernel<_K0: KernelInput<T>, ...>(...) -> MyKernel<T, _K0, DI>
///         // unified launcher — accepts Tensor<T>, Arc<Tensor<T>>, &Tensor<T>
///     pub fn my_kernel_apply(...) -> MyKernel<T, Arc<Tensor<T>>, DI>
///         // internal tuple-input variant (used by api.rs)
/// }
/// ```
// TODO (hme): Prevent reserved names from being used.
// TODO (hme): Validate supported modules.
pub fn module(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attributes as SingleMetaList);
    let is_tile_rust_crate = attrs.parse_bool("tile_rust_crate").unwrap_or(false);
    let tile_rust_crate_root = Ident::new(
        if is_tile_rust_crate {
            "crate"
        } else {
            "cutile"
        },
        Span::call_site(),
    );

    // Get raw item source as a fallback for source text
    // capture.  The primary path uses `Span::source_text()` (which preserves
    // comments), but if that is unavailable we fall back to
    // `TokenStream::to_string()` (which strips comments).
    let raw_item_source = item.to_string();

    let mut module_item = parse_macro_input!(item as ItemMod);
    module_item.attrs = attrs.into();

    match module_inner(&module_item, &tile_rust_crate_root, raw_item_source) {
        Ok(ts) => ts.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Process the items inside a `#[cutile::module]` (or a submodule of one)
/// and return the macro-emitted code for rustc.
///
/// Returns `(concrete_items, entry_functions)`:
/// - `concrete_items` are the rustc-emitted item tokens (functions,
///   structs, impls, etc.) in their declaration order.
/// - `entry_functions` are kernel launcher tokens generated for each
///   `#[cutile::entry]` fn found at this nesting level.
///
/// Submodules (`mod inner { ... }`) are processed recursively: the
/// submodule's items go through the same item walker, then are wrapped
/// back into the original `mod inner { ... }` shell so the namespace is
/// preserved in the rustc-emitted output. The JIT side captures the
/// entire pre-expansion source text via `__module_ast_self`, so the
/// submodule body is automatically available to the JIT's name resolver
/// as part of the captured AST — no separate registry registration per
/// submodule.
fn process_items(
    items: &[syn::Item],
    parent_name: &Ident,
    tile_rust_crate_root: &Ident,
) -> Result<(Vec<TokenStream2>, Vec<TokenStream2>), Error> {
    let mut concrete_items: Vec<TokenStream2> = vec![];
    let mut entry_functions: Vec<TokenStream2> = vec![];
    let type_aliases = cutile_compiler::type_aliases::collect_type_aliases(items);

    for item in items {
        match item {
            syn::Item::Use(use_item) => {
                concrete_items.push(use_item.to_token_stream());
            }
            syn::Item::Fn(function_item) => {
                let entry_attrs = get_meta_list(
                    format!("{} :: entry", tile_rust_crate_root).as_str(),
                    &function_item.attrs,
                );
                if entry_attrs.is_some() {
                    entry_functions.push(kernel_launcher(
                        parent_name,
                        function_item,
                        &type_aliases,
                        tile_rust_crate_root,
                    )?);
                };
                concrete_items.push(function(
                    function_item.clone(),
                    tile_rust_crate_root,
                    &type_aliases,
                )?);
            }
            syn::Item::Struct(struct_item) => {
                let item_clone = struct_item.clone();
                concrete_items.push(structure(item_clone)?.into());
            }
            syn::Item::Trait(trait_item) => {
                let item_clone = trait_item.clone();
                concrete_items.push(trait_(item_clone)?.into());
            }
            syn::Item::Type(type_item) => {
                concrete_items.push(instantiate_type_alias_for_rank(type_item)?.to_token_stream());
            }
            syn::Item::Impl(impl_item) => {
                let item_clone = impl_item.clone();
                concrete_items.push(implementation(item_clone)?.into());
            }
            syn::Item::Macro(macro_item) => {
                let item_clone = macro_item.clone();
                concrete_items.push(item_clone.to_token_stream());
            }
            syn::Item::Const(const_item) => {
                concrete_items.push(const_item.to_token_stream());
            }
            syn::Item::Static(static_item) => {
                let mut concrete = instantiate_static_for_rank(static_item)?;
                clear_attributes(HashSet::from(["cuda_tile :: global"]), &mut concrete.attrs);
                concrete_items.push(concrete.to_token_stream());
            }
            syn::Item::Mod(submod) => {
                let Some(sub_content) = &submod.content else {
                    return submod.err(
                        "Submodule inside `#[cutile::module]` must have an inline body \
                         (`mod foo { ... }`); file-loaded submodules (`mod foo;`) are \
                         not supported because the macro needs the body at expansion time.",
                    );
                };
                let (sub_concrete, sub_entries) =
                    process_items(&sub_content.1, &submod.ident, tile_rust_crate_root)?;
                let sub_name = &submod.ident;
                let sub_attrs = &submod.attrs;
                let sub_vis = &submod.vis;
                let sub_module = quote! {
                    #(#sub_attrs)*
                    #sub_vis mod #sub_name {
                        #(#sub_concrete)*
                        #(#sub_entries)*
                    }
                };
                concrete_items.push(sub_module);
            }
            other => {
                return other.err("Unsupported item type in module.");
            }
        }
    }
    Ok((concrete_items, entry_functions))
}

/// Fallible inner implementation of the `module` macro.
fn module_inner(
    module_item: &ItemMod,
    tile_rust_crate_root: &Ident,
    raw_item_source: String,
) -> Result<TokenStream2, Error> {
    let Some(content) = &module_item.content else {
        return module_item.err("Non-empty module expected.");
    };
    let name = &module_item.ident;
    let (concrete_items, entry_functions) = process_items(&content.1, name, tile_rust_crate_root)?;
    let ast_path = get_ast_path(tile_rust_crate_root);
    let ast_module_item: ItemMod = module_item.clone();
    let ast_module_tokens = emit_module_ast_self_and_registry_entry(
        ast_module_item,
        tile_rust_crate_root,
        raw_item_source.clone(),
    );

    // Compute SHA-256 source hash at macro expansion time.
    let source_hash = format!("{:x}", Sha256::digest(raw_item_source.as_bytes()));

    // Per-module source hash, used by the launch/compile path for cache keying.
    // Warmup no longer needs generated entry metadata: pre-compilation is done
    // per kernel via the `.compile()` terminal (see `TileKernel`).
    let source_hash_const = quote! {
        /// SHA-256 hash of the module source, computed at compile time.
        /// Changes whenever any kernel source in this module changes.
        ///
        /// **Covers this module only.** The JIT walks the use-graph and links in
        /// the dependency modules a kernel calls, so editing one of those changes
        /// the generated cubin while this constant stays put.
        ///
        /// That makes it unsound as a persistent cache key on its own. Within a
        /// process it is fine: reaching an edited helper requires a rebuild, which
        /// restarts the process. The on-disk cache instead keys on the serialized
        /// Tile IR bytecode, which already has every dependency inlined.
        pub const _SOURCE_HASH: &str = #source_hash;
    };

    let res = if entry_functions.is_empty() {
        quote! {
            pub mod #name {
                #![allow(nonstandard_style)]
                #![allow(dead_code)]
                #![allow(unused_variables)]
                // Kernel source is a Rust subset with its own idioms (`a = a + b`,
                // explicit closures for reduce regions, trailing `return`); clippy's
                // host-Rust style lints do not apply to it.
                #![allow(clippy::all)]
                // Module asts and generated type data.
                use #ast_path;
                #ast_module_tokens
                #(#concrete_items)*
                // Warmup metadata.
                #source_hash_const
            }
        }
    } else {
        quote! {
            pub mod #name {
                #![allow(dead_code)]
                // Kernel source is a Rust subset with its own idioms; see above.
                #![allow(clippy::all)]
                // Entry point dependencies.
                // Use of this macro requires cutile,
                // so all dependencies should be imported relative to cutile.
                use std::{iter::zip, future::{Future, IntoFuture}, collections::HashMap, sync::Arc};
                use #tile_rust_crate_root::error::{*};
                use #tile_rust_crate_root::DType;
                use #tile_rust_crate_root::{tensor};
                use #tile_rust_crate_root::tensor::{KernelInput, KernelInputStored, KernelOutput, KernelOutputStored, SpecializationBits};
                use #tile_rust_crate_root::tile_kernel::{*};
                use #tile_rust_crate_root::cuda_async::error::{*};
                use #tile_rust_crate_root::cuda_async::scheduling_policies::SchedulingPolicy;
                use #tile_rust_crate_root::cuda_core::{Device, Function, Module, Stream, DriverError, LaunchConfig};
                // use #tile_rust_crate_root::cutile_compiler::cuda_tile::ModuleOperation;
                // use #tile_rust_crate_root::cutile_compiler::compiler::{CUDATileModules, CUDATileFunctionCompiler};
                // Module asts and generated type data.
                use #ast_path;
                #ast_module_tokens
                #(#concrete_items)*
                // Entry point code.
                #(#entry_functions)*
                // Warmup metadata.
                #source_hash_const
            }
        }
    };
    Ok(res)
}

/// Processes trait definitions.
///
/// Handles trait definitions that may be variadic (rank-polymorphic). Traits marked
/// with `#[cuda_tile::variadic_trait]` are instantiated into one rank-instance per rank.
///
/// ## Variadic Traits
///
/// Variadic traits are instantiated into one trait per rank (1D through 4D):
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_trait(N=4)]
/// pub trait MyTrait<const D: [i32; N]> {
///     fn method(&self) -> Tile<f32, D>;
/// }
///
/// // Output: MyTrait_1, MyTrait_2, MyTrait_3, MyTrait_4
/// ```
///
/// ## Const Generic Desugaring
///
/// For non-variadic traits, const generic array parameters are desugared to make
/// them compatible with the MLIR type system.
pub fn trait_(mut item: ItemTrait) -> Result<TokenStream, Error> {
    // println!("implementation {ident}: {attributes:#?}");
    let attributes = get_meta_list("cuda_tile :: variadic_trait", &item.attrs);
    let is_unchecked = get_meta_list("cuda_tile :: unchecked", &item.attrs);
    if is_unchecked.is_some() {
        return Ok(quote! {}.into());
    }
    clear_attributes(
        HashSet::from(["cuda_tile :: variadic_trait", "cuda_tile :: ty"]),
        &mut item.attrs,
    );
    let res = match attributes {
        Some(attributes)
            if attributes.name_as_str().as_deref() == Some("cuda_tile :: variadic_trait") =>
        {
            desugar_variadic_trait_decl(&item)?
        }
        // Non-variadic traits pass through untouched — there are no rank-instance
        // expansion to do.
        _ => quote! { #item },
    };
    Ok(res.into())
}

/// Processes trait and inherent implementations.
///
/// Handles `impl` blocks that may be variadic. Implementations marked with
/// `#[cuda_tile::variadic_impl]` are instantiated into one rank-instance per rank.
///
/// ## Variadic Implementations
///
/// Variadic implementations are instantiated alongside their corresponding variadic types:
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_impl(N=4)]
/// impl<E: ElementType, const D: [i32; N]> Tile<E, D> {
///     pub fn shape(&self) -> Shape<D> { ... }
/// }
///
/// // Output: Impl blocks for Tile_1, Tile_2, Tile_3, Tile_4
/// ```
///
/// ## Operator Implementations
///
/// Operator traits (`Add`, `Sub`, etc.) are also expanded variadically:
///
/// ```rust,ignore
/// #[cuda_tile::variadic_impl(N=4)]
/// impl<E, const D: [i32; N]> ops::Add<Tile<E, D>> for Tile<E, D> {
///     type Output = Tile<E, D>;
///     fn add(self, rhs: Tile<E, D>) -> Tile<E, D> { ... }
/// }
/// ```
pub fn implementation(mut item: ItemImpl) -> Result<TokenStream, Error> {
    let attributes = get_meta_list("cuda_tile :: variadic_impl", &item.attrs);
    let is_unchecked = get_meta_list("cuda_tile :: unchecked", &item.attrs);
    if is_unchecked.is_some() {
        return Ok(quote! {}.into());
    }
    // Capture trait-impl marker before clear_attributes wipes it. When this
    // marker is present we route to the shadow-trait desugaring (which
    // handles its own rank-instance enumeration via the trait's CGAs).
    // `#[variadic_impl]` is optional in that case.
    let is_variadic_trait_impl =
        get_meta_list("cuda_tile :: variadic_trait_impl", &item.attrs).is_some();
    clear_attributes(
        HashSet::from([
            "cuda_tile :: variadic_trait_impl",
            "cuda_tile :: variadic_impl",
            "cuda_tile :: ty",
        ]),
        &mut item.attrs,
    );
    let res = if is_variadic_trait_impl {
        // `variadic_trait_impl` owns the dispatch — `variadic_impl(N=…)` is
        // accepted alongside it (and ignored) for ergonomics.
        desugar_variadic_trait_impl(&item)?
    } else {
        match attributes {
            Some(attributes) => {
                let items = variadic_impl(&attributes, item)?;
                quote! {
                    #(#items)*
                }
            }
            None => {
                let item = instantiate_impl_for_rank(&item)?;
                quote! { #item }
            }
        }
    };
    Ok(res.into())
}

/// Processes struct definitions.
///
/// Handles struct definitions that may be variadic. Structs marked with
/// `#[cuda_tile::variadic_struct]` are instantiated into one rank-instance per rank.
///
/// ## Variadic Structs
///
/// The most common variadic structs in cuTile Rust are the core types:
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_struct(N=4, constructor="new")]
/// pub struct Tile<E: ElementType, const D: [i32; N]> {
///     _type: PhantomData<E>
/// }
///
/// // Output: Tile_1, Tile_2, Tile_3, Tile_4
/// // Plus optional constructor implementations if specified
/// ```
///
/// ## Constructor Generation
///
/// If `constructor="name"` is specified in the attribute, a constructor method
/// is automatically generated for each expanded struct.
///
/// ## Const Generic Desugaring
///
/// For non-variadic structs, const generic array parameters are desugared to
/// be compatible with the type system.
pub fn structure(mut item: ItemStruct) -> Result<TokenStream, Error> {
    let attributes = get_meta_list("cuda_tile :: variadic_struct", &item.attrs);
    clear_attributes(
        HashSet::from(["cuda_tile :: variadic_struct", "cuda_tile :: ty"]),
        &mut item.attrs,
    );
    // println!("structure {ident}: {attributes:#?}");
    let res = match attributes {
        Some(attributes) => match attributes.name_as_str().unwrap().as_str() {
            "cuda_tile :: variadic_struct" => {
                let items = variadic_struct(&attributes, item)?;
                let structs = items.iter().map(|item| item.0.clone()).collect::<Vec<_>>();
                let maybe_impls = items
                    .iter()
                    .filter(|item| item.1.is_some())
                    .collect::<Vec<_>>();
                let impls = maybe_impls
                    .iter()
                    .map(|item| item.1.clone().unwrap())
                    .collect::<Vec<_>>();
                quote! {
                    #(#structs)*
                    #(#impls)*
                }
            }
            _ => {
                let item = instantiate_struct_for_rank(&item)?;
                quote! { #item }
            }
        },
        None => {
            let item = instantiate_struct_for_rank(&item)?;
            quote! { #item }
        }
    };
    Ok(res.into())
}

/// Processes function definitions.
///
/// Transforms GPU kernel functions and DSL helper functions. Handles:
/// - Variadic operations (rank-polymorphic functions)
/// - Compiler operations (builtin operations like `cast`, `convert`)
/// - Entry point validation
/// - AST building code generation
///
/// ## Variadic Operations
///
/// Functions marked with `#[cuda_tile::variadic_op]` are expanded:
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_op(N=4)]
/// pub fn load_tile<E: ElementType, const S: [i32; N]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
///     // ...
/// }
///
/// // Output: load_tile_1, load_tile_2, load_tile_3, load_tile_4
/// ```
///
/// ## Compiler Operations
///
/// Functions marked with `#[cuda_tile::compiler_op]` are treated as compiler built-ins
/// and generate appropriate MLIR operations.
///
/// ## Entry Points
///
/// Functions marked with `#[entry]` are validated and have launchers generated.
pub fn function(
    mut item: ItemFn,
    tile_rust_crate_root: &Ident,
    type_aliases: &HashMap<String, ItemType>,
) -> Result<TokenStream2, Error> {
    let is_entry = get_meta_list_by_last_segment("entry", &item.attrs).is_some();
    if is_entry {
        let validation_item = cutile_compiler::type_aliases::normalize_item_fn_param_type_aliases(
            &item,
            type_aliases,
        )
        .map_err(|msg| crate::error::syn_err(item.sig.ident.span(), &msg))?;
        validate_entry_point_parameters(&validation_item)?
    }
    let attributes = get_meta_list("cuda_tile :: variadic_op", &item.attrs);
    // Any function annotated with `#[cuda_tile::variadic_op(...)]` is
    // rank-polymorphic and gets shadow-dispatch synthesis (trait + rank-instance
    // impls + free-fn wrapper). The rank arguments (`N = 6`, `M = 6`) are
    // retained for documentation but the emitter uses `MAX_RANK` internally.
    let emit_trait = attributes.is_some();
    // Optional `method = "..."` attribute lets the trait method name differ
    // from the fn name (e.g. `reshape_ptr` → method `reshape` so callers
    // write `ptr_tile.reshape(shape)`). Default: use the fn's own ident.
    let method_override: Option<Ident> = attributes
        .as_ref()
        .and_then(|a| a.parse_string("method"))
        .map(|s| Ident::new(&s, Span::call_site()));
    // Optional `trait_name = "..."` lets the synthesized shadow-trait name
    // differ from PascalCase(fn_ident). Used to dodge collisions with user-
    // defined traits of the same name (e.g. `broadcast_scalar` whose synth
    // `BroadcastScalar` collides with the user's `BroadcastScalar` trait).
    let trait_name_override: Option<Ident> = attributes
        .as_ref()
        .and_then(|a| a.parse_string("trait_name"))
        .map(|s| Ident::new(&s, Span::call_site()));
    // Snapshot the original ItemFn before any mutations, for the shadow-dispatch
    // emitter which needs the CGA-form signature.
    let original_item_for_shadow_dispatch = if emit_trait { Some(item.clone()) } else { None };
    clear_attributes(
        HashSet::from([
            "cuda_tile :: variadic_op",
            "cuda_tile :: op",
            "cuda_tile :: compiler_op",
        ]),
        &mut item.attrs,
    );
    clear_attributes(
        HashSet::from([format!("{} :: entry", tile_rust_crate_root).as_str()]),
        &mut item.attrs,
    );
    if is_entry {
        let kernel_naming = KernelNaming::new(item.sig.ident.to_string().as_str());
        let internal_name = kernel_naming.user_impl_name();
        item.sig.ident = Ident::new(internal_name.as_str(), item.sig.ident.span());
    }
    // Rank-polymorphic fns (`#[cuda_tile::variadic_op(...)]`) are handled
    // entirely by shadow dispatch — no rank-instance free-fn specializations are
    // emitted. Non-variadic fns fall through to CGA desugaring for any stray
    // const-array generics.
    let concrete_items = if emit_trait {
        vec![]
    } else {
        vec![instantiate_function_for_rank(&item)?]
    };
    let shadow_dispatch_tokens = match original_item_for_shadow_dispatch {
        Some(orig) => emit_shadow_dispatch(&orig, method_override, trait_name_override)?,
        None => TokenStream2::new(),
    };
    let result = quote! {
        #(#concrete_items)*
        #shadow_dispatch_tokens
    };
    Ok(result)
}

/// Generates the complete kernel launcher struct and implementations.
///
/// Creates a launcher struct that implements `TileKernel`, `DeviceOp`, and
/// `IntoFuture` for a kernel entry point. This enables the ergonomic launcher API
/// for launching GPU kernels.
///
/// ## Parameters
///
/// - `module_ident`: The module identifier containing the kernel
/// - `item`: The kernel function AST (marked with `#[entry]`)
///
/// ## Returns
///
/// Token stream containing:
/// 1. The launcher struct definition (e.g., `pub struct MyKernel<T, DI> { ... }`)
/// 2. `TileKernel` impl (provides `.grid()`, `.const_grid()`, `.generics()` methods)
/// 3. `DeviceOp` impl (provides `.execute()` for actual kernel launch)
/// 4. `IntoFuture` impl (enables `.await` syntax)
///
/// ## Generated API
///
/// For a kernel `fn my_kernel<T>(x: &mut Tensor<T, [128]>)`, generates:
/// ```rust,ignore
/// let result = MyKernel::launch(input_data)
///     .grid((num_blocks, 1, 1))
///     .await;
/// ```
///
/// ## Entry Attributes
///
/// Respects `#[entry(print_ir = true)]` to print the generated launcher code.
/// Value kinds `#[cutile::entry(..)]` keys accept.
#[derive(Clone, Copy)]
enum EntryValueKind {
    /// A `true` / `false` literal.
    Bool,
    /// A string literal.
    Str,
    /// An expression its consumer parses itself (`preconditions`,
    /// `optimization_hints`), validated there.
    Expr,
}

/// Every key `#[cutile::entry(..)]` honors, with the value kind it expects.
///
/// The JIT looks keys up by name and reads them as literals at first launch
/// (`SingleMetaList::parse_bool` / `parse_string`), so a misspelled key was
/// silently ignored and a non-literal value panicked at runtime. Both are
/// checked here, at expansion, with a span on the offending token.
const ENTRY_KEYS: &[(&str, EntryValueKind)] = &[
    ("print_ir", EntryValueKind::Bool),
    ("dump_mlir_dir", EntryValueKind::Str),
    ("unchecked_accesses", EntryValueKind::Bool),
    ("deny_in_kernel_checks", EntryValueKind::Bool),
    ("preconditions", EntryValueKind::Expr),
    ("optimization_hints", EntryValueKind::Expr),
];

fn validate_entry_attribute(item: &ItemFn) -> Result<(), Error> {
    let Some(attr) = get_attribute("entry", &item.attrs, true) else {
        return Ok(());
    };
    // The bare form `#[cutile::entry]` has nothing to check.
    let syn::Meta::List(list) = &attr.meta else {
        return Ok(());
    };
    let entries = list
        .parse_args_with(syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
        .map_err(|e| {
            crate::error::syn_err(
                e.span(),
                &format!("malformed `#[cutile::entry(..)]` arguments: {e}"),
            )
        })?;
    for meta in &entries {
        // `key = value`, or a bare boolean key (`unchecked_accesses`), which
        // the JIT reads as `true`.
        let (path, value) = match meta {
            syn::Meta::NameValue(name_value) => (&name_value.path, Some(&name_value.value)),
            syn::Meta::Path(path) => (path, None),
            syn::Meta::List(list) => {
                return list.err(
                    "`#[cutile::entry(..)]` arguments must be `key = value` pairs or bare \
                     boolean keys",
                );
            }
        };
        let Some(key) = path.get_ident().map(ToString::to_string) else {
            return path.err("`#[cutile::entry(..)]` keys must be plain identifiers");
        };
        let Some((_, kind)) = ENTRY_KEYS.iter().find(|(known, _)| *known == key) else {
            let known = ENTRY_KEYS
                .iter()
                .map(|(k, _)| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ");
            return path.err(&format!(
                "unknown `#[cutile::entry]` key `{key}`; expected one of {known}"
            ));
        };
        let Some(value) = value else {
            if matches!(kind, EntryValueKind::Bool) {
                continue;
            }
            return path.err(&format!("`{key}` requires a value (`{key} = ...`)"));
        };
        let literal = match value {
            syn::Expr::Lit(lit) => Some(&lit.lit),
            _ => None,
        };
        match (kind, literal) {
            (EntryValueKind::Bool, Some(syn::Lit::Bool(_))) => {}
            (EntryValueKind::Bool, _) => {
                return value.err(&format!(
                    "`{key}` expects a boolean literal (`true` or `false`)"
                ));
            }
            (EntryValueKind::Str, Some(syn::Lit::Str(_))) => {}
            (EntryValueKind::Str, _) => {
                return value.err(&format!("`{key}` expects a string literal"));
            }
            (EntryValueKind::Expr, _) => {}
        }
    }
    Ok(())
}

pub fn kernel_launcher(
    module_ident: &Ident,
    item: &ItemFn,
    type_aliases: &HashMap<String, ItemType>,
    tile_rust_crate_root: &Ident,
) -> Result<TokenStream2, Error> {
    validate_entry_attribute(item)?;
    let module_name = module_ident.to_string();
    let function_name = item.sig.ident.to_string();
    let kernel_naming = KernelNaming::new(function_name.as_str());
    let function_entry_name = kernel_naming.entry_name();
    let launcher_name = function_name.to_case(Case::UpperCamel).to_string();
    let launcher_args_name = format!("{}Args", launcher_name);
    let unsafety = item.sig.unsafety;
    let launcher_item =
        cutile_compiler::type_aliases::normalize_item_fn_param_type_aliases(item, type_aliases)
            .map_err(|msg| crate::error::syn_err(item.sig.ident.span(), &msg))?;

    let (
        required_generics,
        (stored_args_type, returned_args_type),
        device_op_impl,
        kernel_input_info,
    ) = generate_kernel_launcher(
        &launcher_item,
        &module_name,
        &function_name,
        function_entry_name.as_str(),
        &launcher_name,
        &launcher_args_name,
        tile_rust_crate_root,
    )?;

    let launcher_ident = Ident::new(launcher_name.as_str(), Span::call_site());

    let generic_params = required_generics.get_required_generics();
    let generic_args = required_generics.get_generic_args();

    // Build struct generics: kernel params + _K: KernelInput + _P: KernelOutput + DI
    let mut struct_generics = generic_params.clone();
    for (ki_idx, ki_name) in kernel_input_info.type_param_names.iter().enumerate() {
        let elem = &kernel_input_info.element_type_names[ki_idx];
        struct_generics.params.push(
            syn::parse_str::<GenericParam>(&format!("{ki_name}: KernelInput<{elem}>")).unwrap(),
        );
    }
    for (ko_idx, ko_name) in kernel_input_info.ko_type_param_names.iter().enumerate() {
        let elem = &kernel_input_info.ko_element_type_names[ko_idx];
        struct_generics.params.push(
            syn::parse_str::<GenericParam>(&format!("{ko_name}: KernelOutput<{elem}>")).unwrap(),
        );
    }
    let device_op_param: GenericParam = parse_quote! { DI: DeviceOp<Output=#stored_args_type> };
    struct_generics.params.push(device_op_param.clone());

    let mut struct_args = generic_args.clone();
    for ki_name in &kernel_input_info.type_param_names {
        struct_args
            .args
            .push(syn::parse_str::<GenericArgument>(ki_name).unwrap());
    }
    for ko_name in &kernel_input_info.ko_type_param_names {
        struct_args
            .args
            .push(syn::parse_str::<GenericArgument>(ko_name).unwrap());
    }
    let device_op_arg: GenericArgument = parse_quote! { DI };
    struct_args.args.push(device_op_arg.clone());

    // impl TileKernel
    let tile_kernel_impl_type_params = struct_generics.clone();
    let tile_kernel_type_args: AngleBracketedGenericArguments =
        parse_quote! { <#returned_args_type, #device_op_arg, #stored_args_type> };

    // impl IntoFuture
    let into_future_impl_type_params = struct_generics.clone();

    // Build PhantomData to consume KernelInput type params and kernel type params.
    let mut phantom_types: Vec<syn::Type> = vec![];
    for ki_name in &kernel_input_info.type_param_names {
        phantom_types.push(syn::parse_str::<syn::Type>(ki_name.as_str()).unwrap());
    }
    for ko_name in &kernel_input_info.ko_type_param_names {
        phantom_types.push(syn::parse_str::<syn::Type>(ko_name.as_str()).unwrap());
    }
    // Also include kernel type params (T, SrcType, etc.) that may not appear
    // directly in the struct fields now that arg types use KernelInput associated types.
    for param in &generic_params.params {
        if let syn::GenericParam::Type(tp) = param {
            phantom_types.push(syn::parse_str::<syn::Type>(&tp.ident.to_string()).unwrap());
        }
    }
    let ki_phantom_types = phantom_types;

    let result = quote! {

        pub struct #launcher_ident #struct_generics {
            _const_grid: bool,
            _grid: (u32, u32, u32),
            input: Option<DI>,
            function_generics: Option<Vec<String>>,
            _phantom: std::marker::PhantomData<( #(#ki_phantom_types,)* )>,
            _compile_options: CompileOptions,
            // When true, `execute` skips its launch block (set by `.compile()`).
            _compile_only: bool,
            _programmatic_dependent_launch: bool,
        }

        impl #tile_kernel_impl_type_params #launcher_ident #struct_args {
            pub #unsafety fn launch(input: DI) -> Self {
                Self {
                    _const_grid: false,
                    _grid: (0, 0, 0),
                    input: Some(input),
                    function_generics: None,
                    _phantom: std::marker::PhantomData,
                    _compile_options: CompileOptions::default(),
                    _compile_only: false,
                    _programmatic_dependent_launch: false,
                }
            }

            // JIT-compiles and caches this specialization without launching.
            //
            // Runs through `sync` / `sync_on` like every other terminal: the
            // input ops execute under the execution lock, and the stream is
            // synchronized before the materialized inputs are dropped — they
            // may be real tensors that an input op allocated and filled on it.
            pub fn compile(mut self) -> Result<(), DeviceError> {
                self._compile_only = true;
                self.sync()?;
                Ok(())
            }

            pub fn compile_on(mut self, stream: &Arc<Stream>) -> Result<(), DeviceError> {
                self._compile_only = true;
                self.sync_on(stream)?;
                Ok(())
            }

            /// Permit this kernel to overlap its predecessor on the same stream.
            /// Disabled by default; requires Tile IR 13.4 and sm_90 or newer.
            /// This is a launch setting, independent of compile options.
            ///
            /// # Safety
            /// Every predecessor-dependent access must be token-ordered after
            /// `gdc_wait_tko`. Source order alone does not establish that chain.
            /// Work before the wait must not race unfinished predecessor work.
            /// Neither kernel may depend on overlap for progress, and resources
            /// must remain alive through their last use by either kernel.
            pub unsafe fn programmatic_dependent_launch(mut self) -> Self {
                self._programmatic_dependent_launch = true;
                self
            }

            /// Resolves the specialization identity for this launch without compiling or launching the kernel.
            pub fn specialize(self) -> Result<Specialization<ModuleAstFn>, DeviceError> {
                let stream = with_default_device_policy(|policy| policy.next_stream())??;
                self.specialize_on(&stream)
            }

            /// Resolves the specialization identity for this launch on `stream` without compiling or launching the kernel.
            pub fn specialize_on(self, stream: &Arc<Stream>) -> Result<Specialization<ModuleAstFn>, DeviceError> {
                // Same discipline as `compile_on`: resolving executes the input
                // ops, so it takes the execution lock and drains the stream
                // before their outputs are dropped.
                with_context(move |ctx| value(unsafe { self._resolve_specialization(ctx) }))
                    .sync_on(stream)?
            }

            pub fn l1_cache_key(self) -> Result<TileFunctionKey, DeviceError> {
                Ok(self.specialize()?.into_l1_cache_key())
            }

            pub fn l1_cache_key_on(self, stream: &Arc<Stream>) -> Result<TileFunctionKey, DeviceError> {
                Ok(self.specialize_on(stream)?.into_l1_cache_key())
            }

            pub fn l2_cache_key(self) -> Result<String, DeviceError> {
                self.specialize()?
                    .l2_cache_key()
                    .map_err(|error| DeviceError::from(Error::from(error)))
            }

            pub fn l2_cache_key_on(self, stream: &Arc<Stream>) -> Result<String, DeviceError> {
                self.specialize_on(stream)?
                    .l2_cache_key()
                    .map_err(|error| DeviceError::from(Error::from(error)))
            }
        }

        impl #tile_kernel_impl_type_params TileKernel #tile_kernel_type_args for #launcher_ident #struct_args {
            fn grid(mut self, grid: (u32, u32, u32)) -> Self {
                self._grid = grid;
                self._const_grid = false;
                self
            }
            fn const_grid(mut self, grid: (u32, u32, u32)) -> Self {
                self._grid = grid;
                self._const_grid = true;
                self
            }
            fn get_launch_grid(&self) -> (u32, u32, u32) {
                self._grid
            }
            fn generics(mut self, generics: Vec<String>) -> Self {
                self.function_generics = Some(generics);
                self
            }
            fn compile_options(mut self, options: CompileOptions) -> Self {
                self._compile_options = options;
                self
            }
        }
        impl #into_future_impl_type_params IntoFuture for #launcher_ident #struct_args {
            type Output = Result<#returned_args_type, DeviceError>;
            type IntoFuture = DeviceFuture<#returned_args_type, #launcher_ident #struct_args>;
            fn into_future(self) -> Self::IntoFuture {
                match with_default_device_policy(|policy| { let stream = policy.next_stream()?; Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream))) }) {
                    Ok(Ok(future)) => future,
                    Ok(Err(e)) => DeviceFuture::failed(e),
                    Err(e) => DeviceFuture::failed(e),
                }
            }
        }

        // Implements DeviceOp, along with the generated launcher functions.
        #device_op_impl
    };

    let Some(_entry_attrs) = get_meta_list_by_last_segment("entry", &item.attrs) else {
        return item.sig.ident.err(&format!(
            "Unexpected entry point {function_name}: Missing entry annotation."
        ));
    };

    if let Ok(dir) = env::var("DUMP_KERNEL_LAUNCHER_DIR") {
        let file = parse_file(&result.to_string()).expect("Failed to parse file.");
        let filename = format!("{module_name}_{function_name}_launcher.rs");
        let path = PathBuf::from(dir).join(filename);
        let contents = file_item_string_pretty(&file);
        fs::write(path.clone(), contents).unwrap_or_else(|_| panic!("Failed to write {path:?}"));
        // Writes the string as bytes
    }
    Ok(result)
}

/// Generates the module AST builder function.
/// Creates a function that returns the AST representations of all kernels and
/// functions defined in the module. This AST is used by the MLIR compiler to
/// generate GPU code.
///
/// ## Parameters
///
/// - `item`: The module AST
/// - `module_ast_calls`: Vector of strings containing calls to AST builder functions
///   for each kernel/function in the module (e.g., `["my_kernel_asts()", "helper_asts()"]`)
///
/// ## Returns
///
/// Token stream containing a function like:
/// ```rust,ignore
/// pub fn module_name_asts() -> Vec<cuda_ast::Ast> {
///     vec![my_kernel_asts(), helper_asts()]
/// }
/// ```
///
/// ## Purpose
///
/// This function aggregates all ASTs from a module, enabling the module to be
/// compiled as a unit. The ASTs are later passed to the MLIR compiler for code
/// generation.
///
/// ## Source Location Tracking
///
/// At proc macro expansion time, `proc_macro2` spans carry real file / line /
/// column information (on nightly with the `span-locations` feature).  We
/// exploit this to recover **exact** source locations for *every* node in the
/// syn AST at JIT compile time, using the following scheme:
///
/// 1. Construct a `Span` covering the entire `ItemMod` (from the `mod`
///    keyword to the closing `}`) and call `Span::source_text()` to obtain
///    the **verbatim** original source text — whitespace, newlines, comments,
///    and all.  (`TokenStream::to_string()` strips comments, so we must use
///    `source_text()` instead.)
/// 2. Record the **span base** – `(file, base_line, base_col)` – from the
///    module's opening token via `Span::file()` and `Span::start()`.
/// 3. At runtime, feed the source text to `syn::parse_str` instead of
///    re-quoting.  Because the string is character-for-character identical to
///    the original source, the resulting spans have line/column numbers that
///    map 1-to-1 with the original file layout.
/// 4. Any runtime span can then be resolved to an absolute position via:
///    ```text
///    abs_line = base_line + (span_line − 1)
///    abs_col  = if span_line == 1 { base_col + span_col } else { span_col }
///    ```
///
/// This gives exact file / line / column for every node – statements,
/// expressions, sub-expressions, individual tokens – without requiring any
/// up-front walk or key-based lookup table.
///
/// ### Fallback
///
/// If `Span::source_text()` is unavailable (e.g. on a stable compiler, or
/// when the span doesn't map to real source), we fall back to
/// `TokenStream::to_string()`.  This produces comment-free text, so line
/// numbers may be shifted earlier by the number of stripped comment lines.
pub fn emit_module_ast_self_and_registry_entry(
    item: ItemMod,
    tile_rust_crate_root: &Ident,
    raw_item_source: String,
) -> TokenStream2 {
    // Use the visibility's span when present (so `pub mod foo { ... }`
    // covers the full text), otherwise fall back to `mod`.
    let item_start_span = match &item.vis {
        syn::Visibility::Public(vis_pub) => vis_pub.span,
        syn::Visibility::Restricted(vis_r) => vis_r.pub_token.span,
        syn::Visibility::Inherited => item.mod_token.span,
    };
    let source_file = item_start_span.file().to_string();
    let base_line = item_start_span.start().line;
    let base_col = item_start_span.start().column;

    // Verbatim source text of the parent module.
    let source_text = {
        let full_span = item
            .content
            .as_ref()
            .and_then(|(brace, _)| item_start_span.join(brace.span.close()));
        let file_slice = item.content.as_ref().and_then(|(brace, _)| {
            source_slice_from_file(
                &source_file,
                item_start_span.start(),
                brace.span.close().end(),
            )
        });

        full_span
            .and_then(|sp| sp.source_text())
            .or(file_slice)
            .unwrap_or(raw_item_source)
    };

    emit_ast_self_and_registry(
        &item.ident,
        &source_text,
        &source_file,
        base_line,
        base_col,
        tile_rust_crate_root,
    )
}

/// Emit `__module_ast_self()` + linker-registry entry given pre-computed
/// source text and span anchor.
///
/// The parent-module path computes these from the macro invocation; the
/// file-loaded-submodule path computes them from the file on disk
/// (`source_text` = file contents, `source_file` = file path,
/// `base_line` = 1, `base_col` = 0). In both cases, spans on the parsed
/// AST map back to real `(file, line, col)` triples.
fn emit_ast_self_and_registry(
    name: &Ident,
    source_text: &str,
    source_file: &str,
    base_line: usize,
    base_col: usize,
    tile_rust_crate_root: &Ident,
) -> TokenStream2 {
    let ast_path = get_ast_path(tile_rust_crate_root);
    let registry_path = get_registry_path(tile_rust_crate_root);
    let self_ast_ident = get_self_ast_ident();
    let name_string = name.to_string();
    let registry_static_ident =
        format_ident!("__CUTILE_MODULE_ENTRY_{}", name_string.to_uppercase());

    quote! {
        // Per-module self-only AST builder. Builds *just* this module's
        // `Module` value. Called by the linker-registry entry's `build`
        // closure at JIT time.
        pub fn #self_ast_ident() -> #ast_path::Module {
            use #ast_path::syn;

            let source_text: &str = #source_text;
            let parsed_mod: syn::ItemMod = syn::parse_str(source_text)
                .expect("__module_ast_self: failed to re-parse captured source text");

            let span_base = #ast_path::SpanBase::new(
                #source_file.to_string(),
                #base_line,
                #base_col,
            );

            let mut this_ast = #ast_path::Module::with_span_base(
                #name_string,
                parsed_mod,
                span_base,
            );
            this_ast.set_absolute_path(module_path!().to_string());
            this_ast
        }

        // Linker-registry entry. Self-registers this module into the
        // global `CUTILE_MODULES` distributed slice at link time so the
        // JIT can discover it via `CUDATileModules::from_kernel`.
        #[#registry_path::linkme::distributed_slice(#registry_path::CUTILE_MODULES)]
        #[linkme(crate = #registry_path::linkme)]
        static #registry_static_ident: #registry_path::CutileModuleEntry =
            #registry_path::CutileModuleEntry {
                absolute_path: ::std::module_path!(),
                build: #self_ast_ident,
            };
    }
}
