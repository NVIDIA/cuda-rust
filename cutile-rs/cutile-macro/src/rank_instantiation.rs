/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Rank-instance instantiation of CGA-bearing items.
//!
//! Items that use a const-generic-array generic — `const X: [i32; N]` — are
//! rank-polymorphic at the source level but Rust can't compile them as such
//! (the array length `N` is variable across instantiations). This module
//! turns each such item into one concrete copy per rank instance, with `X` replaced
//! by individual scalar consts `X_0, X_1, …, X_{R-1}` and every type whose
//! generic args reference the CGA suffixed accordingly.
//!
//! ## What gets emitted
//!
//! For a struct annotated `#[cuda_tile::variadic_struct(N = 4)]`:
//!
//! ```text
//! pub struct Tile<E, const D: [i32; N]> { /* ... */ }
//!
//! // rank-instance specializations:
//! pub struct Tile_1<E, const D_0: i32> { /* ... */ }
//! pub struct Tile_2<E, const D_0: i32, const D_1: i32> { /* ... */ }
//! pub struct Tile_3<E, const D_0: i32, const D_1: i32, const D_2: i32> { /* ... */ }
//! pub struct Tile_4<E, ...> { /* ... */ }
//! ```
//!
//! Inherent impls on a CGA-bearing struct expand the same way, with their
//! method bodies rewritten so any reference to `D`, `Shape<D>`, `Tile<E, D>`,
//! etc. picks up the rank-instance form. Rust then sees concrete `Tile_R` types
//! and resolves everything normally; the macro doesn't do any type inference
//! or call-site rewriting beyond that — see [`crate::shadow_dispatch`] for
//! how rank-polymorphic *operations* (`#[variadic_op]` fns and
//! `#[variadic_trait]` declarations) are emitted as a single rank-polymorphic trait
//! plus rank-instance impls so user calls resolve via rustc's normal trait
//! lookup.
//!
//! ## How it's wired
//!
//! Public entry points:
//!
//! - [`variadic_struct`] / [`variadic_impl`] — produce the multi-rank
//!   specializations from a single source item.
//! - [`instantiate_struct_for_rank`] / [`instantiate_function_for_rank`] /
//!   [`instantiate_impl_for_rank`] — apply the rank-instance substitution to a single
//!   item already constrained to one rank (e.g. an `#[entry]` kernel with
//!   `const S: [i32; 1]`).
//!
//! Each path builds a [`RankBindings`] (the active CGA → length mapping
//! for the rank being emitted) and walks the item with a [`RankInstantiator`]
//! that implements [`syn::visit_mut::VisitMut`]. The visitor handles:
//!
//! - **Type paths** — `Tile<E, S>` → `Tile_R<E, S_0, …>`.
//! - **Expression paths** — `iota::<i32, S>` → `iota::<i32, S_0, …>` (the fn
//!   ident is preserved; only the CGA args expand).
//! - **Struct literals** — `Shape::<{[1, 2]}> { … }` → `Shape_2 { … }`.
//! - **Bare CGA refs in expressions** — `S` → its bound `u32`, `S[i]` → `S_i`.
//! - **`const_shape!` / `const_array!` macros** — pre-expanded to typed
//!   `Shape_R::<…>::const_new()` / `Array_R::<…>::const_new()` calls.
//!
//! All of this is purely structural: there's no name-keyed registry of
//! variadic types and no return-type inference — a path is "rank-polymorphic"
//! because it has CGA args, full stop.

use crate::error::{syn_err, Error};
use cutile_compiler::syn_utils::*;
use cutile_compiler::types::parse_signed_literal_as_i32;
use proc_macro2::{Ident, Span};
use quote::{format_ident, ToTokens};
use std::collections::BTreeMap;
use std::collections::HashMap;
use syn::{
    parse::Parser,
    parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
    AngleBracketedGenericArguments, Expr, ExprPath, ExprStruct, FnArg, GenericArgument,
    GenericParam, Generics, ImplItem, ImplItemFn, ItemFn, ItemImpl, ItemStatic, ItemStruct,
    ItemType, Macro, Path, PathArguments, PathSegment, ReturnType, Signature, Stmt, Token, Type,
};

/// Rank suffix string from a list of dimension counts:
/// `[2]` → `"2"`, `[2, 3]` → `"2_3"`. Used by [`concrete_name`].
pub fn rank_suffix(n: &[u32]) -> String {
    n.iter()
        .map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join("_")
}

/// Build a concrete rank-instance type name by appending the rank suffix:
/// `concrete_name("Tile", &[2])` → `"Tile_2"`.
pub fn concrete_name(name: &str, n: &[u32]) -> String {
    format!("{}_{}", name, rank_suffix(n))
}

/// Tracks const generic array instantiations during macro expansion.
///
/// This struct maintains mappings between const generic array parameters and their
/// concrete instantiations. It's used during variadic type and operation rewriting
/// to track which arrays have been instantiated with which dimensions.
///
/// ## Fields
///
/// - `inst_u32`: Maps length variable names (e.g., `"N"`) to their concrete values (e.g., `2`)
/// - `var_arrays`: Maps array names to their variable CGA parameter definitions
/// - `inst_array`: Maps array names to their concrete instantiated parameters
///
/// ## Example
///
/// For a function `fn foo<const N: usize, const S: [i32; N]>(...)`, this tracks:
/// - `inst_u32`: `{"N" => 2}`
/// - `var_arrays`: `{"S" => VarCGAParameter { name: "S", length_var: "N" }}`
/// - `inst_array`: `{"S" => CGAParameter { name: "S", length: 2 }}`
///
/// Tracks const generic array instantiations during variadic macro expansion.
///
/// Maps length variable names to concrete values and array names to their
/// instantiated parameters.
#[derive(Debug, Clone)]
pub struct RankBindings {
    inst_u32: HashMap<String, u32>,
    var_arrays: HashMap<String, VarCGAParameter>,
    inst_array: HashMap<String, CGAParameter>,
}

impl RankBindings {
    fn new() -> Self {
        let inst_u32: HashMap<String, u32> = HashMap::new();
        let inst_array: HashMap<String, CGAParameter> = HashMap::new();
        let var_arrays: HashMap<String, VarCGAParameter> = HashMap::new();
        RankBindings {
            inst_u32,
            inst_array,
            var_arrays,
        }
    }
    fn from_variadic(
        cga_lengths: &VariadicLengthItem,
        var_cgas: &[VarCGAParameter],
    ) -> Result<Self, Error> {
        let mut inst_u32: HashMap<String, u32> = HashMap::new();
        let mut inst_array: HashMap<String, CGAParameter> = HashMap::new();
        let mut var_arrays: HashMap<String, VarCGAParameter> = HashMap::new();
        for (length_var_name, length_instance) in &cga_lengths.variadic_length_instance {
            inst_u32.insert(length_var_name.clone(), *length_instance as u32);
        }
        for (cga, (length_var_name, length_instance)) in
            var_cgas.iter().zip(cga_lengths.cga_length_instance.iter())
        {
            let length_instance = *length_instance as u32;
            if length_var_name != &cga.length_var {
                return Err(syn_err(
                    Span::call_site(),
                    &format!(
                        "CGA length var name mismatch: expected '{}', got '{}'",
                        cga.length_var, length_var_name
                    ),
                ));
            }
            if let Some(existing_length) = inst_u32.insert(length_var_name.clone(), length_instance)
            {
                if existing_length != length_instance {
                    return Err(syn_err(
                        Span::call_site(),
                        &format!(
                            "CGA length instance mismatch for '{}': expected {}, got {}",
                            length_var_name, existing_length, length_instance
                        ),
                    ));
                }
            }
            inst_array.insert(cga.name.clone(), cga.instance(length_instance));
            var_arrays.insert(cga.name.clone(), cga.clone());
        }
        Ok(RankBindings {
            inst_u32,
            inst_array,
            var_arrays,
        })
    }
    fn from_generics(generics: &Generics) -> Result<Self, Error> {
        let (cga_param, _u32_param) = parse_cgas(generics);
        let inst_u32: HashMap<String, u32> = HashMap::new();
        let mut inst_array: HashMap<String, CGAParameter> = HashMap::new();
        let var_arrays: HashMap<String, VarCGAParameter> = HashMap::new();
        for cga in cga_param {
            inst_array.insert(cga.name.clone(), cga.clone());
        }
        Ok(RankBindings {
            inst_u32,
            inst_array,
            var_arrays,
        })
    }
    fn instantiate_var_cgas(&self, var_cgas: &[VarCGAParameter]) -> Result<Self, Error> {
        let mut result = self.clone();
        for cga in var_cgas {
            if !result.inst_u32.contains_key(&cga.length_var) {
                return Err(syn_err(
                    Span::call_site(),
                    &format!(
                        "instantiate_var_cgas: Missing inst_u32 entry for '{}'",
                        cga.length_var
                    ),
                ));
            }
            let n = result.inst_u32.get(&cga.length_var).unwrap();
            result.inst_array.insert(cga.name.clone(), cga.instance(*n));
            result.var_arrays.insert(cga.name.clone(), cga.clone());
        }
        Ok(result)
    }
    fn instantiate_new_var_cgas(
        &self,
        n_list: &[u32],
        var_cgas: &[VarCGAParameter],
    ) -> Result<Self, Error> {
        let mut result = self.clone();
        for i in 0..n_list.len() {
            let n: u32 = n_list[i];
            let cga = &var_cgas[i];
            if result.inst_u32.contains_key(&cga.length_var) {
                return Err(syn_err(
                    Span::call_site(),
                    &format!(
                        "instantiate_new_var_cgas: inst_u32 already contains entry for '{}'",
                        cga.length_var
                    ),
                ));
            }
            result.inst_u32.insert(cga.length_var.clone(), n);
            result.inst_array.insert(cga.name.clone(), cga.instance(n));
            result.var_arrays.insert(cga.name.clone(), cga.clone());
        }
        Ok(result)
    }
}

/// Iterator for generating all combinations of const generic array lengths.
///
/// This iterator generates const instances for variadic types and operations by
/// iterating through all valid combinations of array lengths. For example, if
/// a variadic struct has `N=4`, this generates instances for N=0, 1, 2, 3, 4.
///
/// ## Fields
///
/// - `i`: Current iteration index
/// - `i_max`: Maximum number of iterations (product of all unique lengths)
/// - `cga_lengths`: List of (length_var_name, max_length) tuples
/// - `arrays`: Reference to the const generic array parameters being instantiated
///
/// ## Example
/// For `#[variadic_struct(N=4)]` with two arrays depending on N, this generates:
/// - (0, 0), (1, 1), ..., (4, 4) - 5 total combinations
///
/// Iterates over all combinations of const generic array lengths for variadic expansion.
#[derive(Debug)]
struct VariadicLengthIterator {
    i: usize,
    i_max: usize,
    variadic_lengths: BTreeMap<String, usize>, // Deterministic order is required for correctness.
    cga_length_vars: Vec<String>,
}

impl VariadicLengthIterator {
    fn new(attribute_list: &SingleMetaList, arrays: &[VarCGAParameter]) -> Result<Self, Error> {
        let mut i_max: usize = 1;
        let mut variadic_lengths: BTreeMap<String, usize> = BTreeMap::new();
        if let Some(variadic_length_vars) = attribute_list.parse_string_arr("variadic_length_vars")
        {
            for var in variadic_length_vars {
                let len = (attribute_list.parse_int(var.as_str()).ok_or_else(|| {
                    syn_err(
                        Span::call_site(),
                        &format!("Missing attribute value for '{var}'"),
                    )
                })? + 1) as usize;
                i_max *= len;
                if variadic_lengths.insert(var.clone(), len).is_some() {
                    return Err(syn_err(
                        Span::call_site(),
                        &format!("Duplicate variadic_length_var '{var}'"),
                    ));
                }
            }
        }
        let mut cga_length_vars = vec![];
        for cga in arrays {
            let var = cga.length_var.clone();
            cga_length_vars.push(var.clone());
            // This is so we don't need to explicitly specify the variadic_length_vars attribute.
            let len = (attribute_list.parse_int(var.as_str()).ok_or_else(|| {
                syn_err(
                    Span::call_site(),
                    &format!("Missing attribute value for '{var}'"),
                )
            })? + 1) as usize;
            if variadic_lengths.contains_key(&var) {
                if *variadic_lengths.get(&var).unwrap() != len {
                    return Err(syn_err(
                        Span::call_site(),
                        &format!("Variadic length mismatch for '{var}'"),
                    ));
                }
            } else {
                i_max *= len;
                variadic_lengths.insert(var.clone(), len);
            }
        }
        Ok(VariadicLengthIterator {
            i: 0,
            i_max,
            variadic_lengths,
            cga_length_vars,
        })
    }
}

/// A single combination of length variable values produced by `VariadicLengthIterator`.
pub struct VariadicLengthItem {
    variadic_length_instance: BTreeMap<String, usize>,
    cga_length_instance: Vec<(String, usize)>,
}

impl VariadicLengthItem {
    /// Returns CGA lengths as a vector, ordered by CGA declaration order.
    pub fn vec_of_cga_lengths(&self) -> Vec<u32> {
        // Ordered by key.
        self.cga_length_instance
            .iter()
            .map(|x| x.1 as u32)
            .collect::<Vec<_>>()
    }
    /// Returns unique length variable values, ordered by variable name.
    pub fn vec_of_unique_lengths(&self) -> Vec<u32> {
        // Ordered by key.
        self.variadic_length_instance
            .values()
            .map(|x| *x as u32)
            .collect::<Vec<_>>()
    }
}

impl Iterator for VariadicLengthIterator {
    type Item = VariadicLengthItem;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.i_max {
            let mut variadic_length_instance: BTreeMap<String, usize> = BTreeMap::new();
            let mut i = self.i;
            for (len_var, len) in self.variadic_lengths.iter() {
                // BTree iter sorts by key.
                let pos = i % len;
                i /= len;
                variadic_length_instance.insert(len_var.clone(), pos);
            }
            self.i += 1;
            let mut cga_length_instance: Vec<(String, usize)> = vec![];
            for len_var in &self.cga_length_vars {
                let len = *variadic_length_instance
                    .get(len_var)
                    .unwrap_or_else(|| panic!("Unexpected length var {len_var}"));
                cga_length_instance.push((len_var.clone(), len));
            }
            Some(VariadicLengthItem {
                variadic_length_instance,
                cga_length_instance,
            })
        } else {
            None
        }
    }
}

/// Extracts variable const generic array parameters from a `Generics` clause.
pub fn cgas_from_generics(generics: &Generics) -> Vec<VarCGAParameter> {
    let mut result: Vec<VarCGAParameter> = vec![];
    for param in &generics.params {
        match param {
            GenericParam::Type(_type_param) => continue,
            GenericParam::Const(const_param) => match &const_param.ty {
                Type::Array(_ty_arr) => {
                    let arr_type_param = VarCGAParameter::from_const_param(const_param);
                    result.push(arr_type_param);
                }
                _ => continue,
            },
            _ => continue,
        }
    }
    result
}

/// Expands a variadic struct into multiple rank-specific versions.
///
/// Takes a struct with const generic array parameters and generates concrete
/// versions for each rank (typically 1-4).
///
/// ## Parameters
///
/// - `attributes`: Macro attributes (e.g., `N=4`, `constructor="new"`)
/// - `item`: The struct definition to expand
///
/// ## Returns
///
/// A vector of (struct, optional impl) pairs, one for each rank. The impl is
/// generated if a constructor name is specified in the attributes.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_struct(N=4, constructor="new")]
/// pub struct Tile<E, const D: [i32; N]> { _type: PhantomData<E> }
///
/// // Generates:
/// // - Tile_1, Tile_2, Tile_3, Tile_4 structs
/// // - Optional impl blocks with constructors if constructor="new" is specified
/// ```
pub fn variadic_struct(
    attributes: &SingleMetaList,
    item: ItemStruct,
) -> Result<Vec<(ItemStruct, Option<ItemImpl>)>, Error> {
    let cgas = cgas_from_generics(&item.generics);
    let cga_iter = VariadicLengthIterator::new(attributes, &cgas)?;

    // The struct's generics define the CGAs; their names + index types come
    // from the AST directly (no static registry lookup). The constructor
    // attribute decides whether to emit an inherent impl with constructors.
    let maybe_constructor_name = attributes.parse_string("constructor");
    let base_name = item.ident.to_string();
    // Mixed static/dynamic constructors are emitted only for structs that
    // actually have a slice-typed `dims` field to write to. Detected
    // syntactically rather than via a registry flag.
    let has_dims_field = struct_has_dims_field(&item);
    let mut result: Vec<(ItemStruct, Option<ItemImpl>)> = vec![];
    for var_cga_iter_item in cga_iter {
        let mut concrete = item.clone();
        let const_instances = RankBindings::from_variadic(&var_cga_iter_item, &cgas)?;
        let concrete_ident = Ident::new(
            &concrete_name(&base_name, &var_cga_iter_item.vec_of_cga_lengths()),
            concrete.ident.span(),
        );
        concrete.ident = concrete_ident;
        rewrite_generics_for_rank(&mut concrete.generics, &const_instances)?;
        let concrete_impl = if maybe_constructor_name.is_some() {
            let mut type_params: Vec<String> = vec![];
            let mut type_args: Vec<String> = vec![];
            let mut constructors: Vec<String> = vec![];
            for (cga_idx, cga) in cgas.iter().enumerate() {
                let n = var_cga_iter_item.vec_of_cga_lengths()[cga_idx];
                let cga_name = &cga.name;
                let cga_index_type = &cga.element_type;
                for dim_idx in 0..n {
                    type_params.push(format!("const {cga_name}{dim_idx}: {cga_index_type}"));
                    type_args.push(format!("{cga_name}{dim_idx}"));
                }
                if has_dims_field {
                    // Emit one constructor per supported number of dynamic
                    // dims (0..=n). The const variant uses an empty slice.
                    for num_dynamic in 0..(n + 1) {
                        let struct_name = concrete.ident.to_string();
                        let constructor_name = format!(
                            "{}_{}",
                            maybe_constructor_name.clone().unwrap(),
                            num_dynamic
                        );
                        let dim_type_str = "i32";
                        let dyn_constructor = format!(
                            r#"
                             pub fn {constructor_name}(dims: &'a [{dim_type_str}; {num_dynamic}]) -> Self {{
                                 {struct_name} {{ dims: dims }}
                             }}
                         "#
                        );
                        constructors.push(dyn_constructor);
                        if num_dynamic == 0 {
                            let constructor_name =
                                maybe_constructor_name.clone().unwrap().to_string();
                            let const_constructor = format!(
                                r#"
                             pub fn const_{constructor_name}() -> Self {{
                                 {struct_name} {{ dims: &[] }}
                             }}
                         "#
                            );
                            constructors.push(const_constructor);
                        }
                    }
                }
            }
            if constructors.is_empty() {
                None
            } else {
                let name = concrete.ident.to_string();
                let impl_generics = type_params.join(",");
                let impl_constructors = constructors.join("\n");
                let impl_type_args = type_args.join(",");
                let constructor_impl = format!(
                    r#"
                impl<'a, {impl_generics}> {name}<'a, {impl_type_args}> {{
                        {impl_constructors}
                    }}
                "#
                );
                let parsed_impl =
                    syn::parse::<ItemImpl>(constructor_impl.parse().map_err(|_| {
                        syn_err(item.ident.span(), "Failed to parse constructor impl")
                    })?)
                    .map_err(|e| {
                        syn_err(
                            item.ident.span(),
                            &format!("Failed to parse constructor impl: {e}"),
                        )
                    })?;
                Some(parsed_impl)
            }
        } else {
            None
        };
        result.push((concrete, concrete_impl));
    }
    Ok(result)
}

/// True if the struct has a field named `dims` whose type is a reference to a
/// slice (e.g. `&'a [i32]`). Used by `variadic_struct` to decide whether to
/// emit dynamic-dim constructors — purely syntactic, no registry lookup.
fn struct_has_dims_field(item: &ItemStruct) -> bool {
    let syn::Fields::Named(fields) = &item.fields else {
        return false;
    };
    fields.named.iter().any(|f| {
        let is_dims = f.ident.as_ref().is_some_and(|i| i == "dims");
        if !is_dims {
            return false;
        }
        matches!(&f.ty, Type::Reference(r) if matches!(&*r.elem, Type::Slice(_)))
    })
}

/// Expands a variadic implementation into multiple rank-specific versions.
///
/// Generates an impl block for each rank, with all types and method signatures
/// updated to use the rank-specific type names.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_impl(N=4)]
/// impl<E, const D: [i32; N]> Tile<E, D> {
///     pub fn shape(&self) -> Shape<D> { ... }
/// }
///
/// // Output:
/// impl<E, const D: [i32; 1]> Tile_1<E, D> { ... }
/// impl<E, const D: [i32; 2]> Tile_2<E, D> { ... }
/// impl<E, const D: [i32; 3]> Tile_3<E, D> { ... }
/// impl<E, const D: [i32; 4]> Tile_4<E, D> { ... }
/// ```
pub fn variadic_impl(attributes: &SingleMetaList, item: ItemImpl) -> Result<Vec<ItemImpl>, Error> {
    let cgas = cgas_from_generics(&item.generics);
    let cga_iter = VariadicLengthIterator::new(attributes, &cgas)?;

    let mut result: Vec<ItemImpl> = vec![];
    for n_list in cga_iter {
        let bindings = RankBindings::from_variadic(&n_list, &cgas)?;
        result.push(RankInstantiator::new(bindings).rewrite_impl(&item)?);
    }
    Ok(result)
}

/// Desugars const generic arrays in a function signature's generics, inputs, and output.
fn rewrite_fn_sig(sig: &mut Signature, const_instances: &RankBindings) -> Result<(), Error> {
    rewrite_generics_for_rank(&mut sig.generics, const_instances)?;
    for input in sig.inputs.iter_mut() {
        match input {
            FnArg::Receiver(_receiver) => {
                // Leave this.
            }
            FnArg::Typed(fn_param) => {
                let fn_param_type = rewrite_type_for_rank(&fn_param.ty, const_instances)?;
                *fn_param.ty = fn_param_type;
            }
        }
    }
    if let ReturnType::Type(_, return_type) = &mut sig.output {
        **return_type = rewrite_type_for_rank(&return_type.clone(), const_instances)?;
    }
    Ok(())
}

/// Expands const generic array params in a `Generics` clause into individual const params.
fn rewrite_generics_for_rank(
    generics: &mut Generics,
    const_instances: &RankBindings,
) -> Result<(), Error> {
    let mut concrete_type_params = generics.params.clone();
    concrete_type_params.clear();
    for param in generics.params.iter() {
        match param {
            GenericParam::Const(const_param) => match &const_param.ty {
                Type::Array(_ty_arr) => {
                    let const_param_name = const_param.ident.to_string();
                    let cga = const_instances
                        .inst_array
                        .get(const_param_name.as_str())
                        .ok_or_else(|| {
                            syn_err(
                                const_param.ident.span(),
                                &format!("Missing inst_array entry for '{const_param_name}'"),
                            )
                        })?;
                    if cga.element_type != "i32" {
                        return Err(syn_err(
                            const_param.ident.span(),
                            &format!("Expected element_type 'i32', got '{}'", cga.element_type),
                        ));
                    }
                    for i in 0..cga.length {
                        let const_str = format!("const {}{}: {}", cga.name, i, cga.element_type);
                        let generic_param =
                            syn::parse::<GenericParam>(const_str.parse().map_err(|_| {
                                syn_err(
                                    const_param.ident.span(),
                                    &format!("Failed to parse generic param '{const_str}'"),
                                )
                            })?)
                            .map_err(|e| {
                                syn_err(
                                    const_param.ident.span(),
                                    &format!("Failed to parse generic param '{const_str}': {e}"),
                                )
                            })?;
                        concrete_type_params.push(generic_param);
                    }
                }
                _ => concrete_type_params.push(param.clone()),
            },
            _ => concrete_type_params.push(param.clone()),
        }
    }
    generics.params = concrete_type_params;
    Ok(())
}

/// Expands a CGA path into angle-bracketed individual const generic arguments.
fn instantiate_cga(
    path: &Path,
    instances: &RankBindings,
) -> Result<AngleBracketedGenericArguments, Error> {
    let _result_path = path.clone();
    let last_seg = path.segments.last().ok_or_else(|| {
        syn_err(
            path.span(),
            "Expected at least one path segment in instantiate_cga",
        )
    })?;
    let param_name = last_seg.ident.to_string();
    // Is it a variadic type or is it expecting a variadic type parameter?
    if instances.inst_array.contains_key(&param_name) {
        // The type is a const generic array, e.g. the D in f(..., shape: D) -> ()
        let cga = instances.inst_array.get(&param_name).unwrap();
        let mut generic_args_result: Vec<String> = vec![];
        for j in 0..cga.length {
            generic_args_result.push(format!("{}{}", cga.name, j));
        }
        let formatted = format!("<{}>", generic_args_result.join(","));
        Ok(
            syn::parse::<AngleBracketedGenericArguments>(formatted.parse().map_err(|_| {
                syn_err(
                    path.span(),
                    &format!("Failed to parse angle bracketed args '{formatted}'"),
                )
            })?)
            .map_err(|e| {
                syn_err(
                    path.span(),
                    &format!("Failed to parse angle bracketed args '{formatted}': {e}"),
                )
            })?,
        )
    } else {
        Err(syn_err(
            path.span(),
            &format!("{} is not a const generic array.", path.to_token_stream()),
        ))
    }
}

/// Where in the AST a path is being desugared. Drives whether the *last*
/// segment's ident gets rank-instance-suffixed when it carries a CGA arg.
///
/// In a **type** position every segment whose generic args reference a CGA
/// is a variadic type by construction — `Tile<E, S>` is `Tile_R<E, S_0, …>`,
/// and the same goes for non-last segments of a multi-segment type path.
///
/// In an **expression-path** position, the last segment names a function or
/// associated item (e.g. `iota::<i32, S>`, `Tile::<E, S>::method`); rustc
/// resolves it through the shadow-trait wrapper or inherent impl, which
/// keeps the unsuffixed name. Earlier segments in the same path are still
/// type-like (UFCS uses them that way) and still get suffixed.
#[derive(Copy, Clone, Eq, PartialEq)]
enum PathContext {
    Type,
    ExprPath,
}

/// Desugars variadic types in a path, replacing CGA syntax with concrete type names and args.
fn rewrite_path_for_rank(
    path: &Path,
    instances: &RankBindings,
    context: PathContext,
) -> Result<Path, Error> {
    let mut result_path = path.clone();
    let last_idx = path.segments.len().saturating_sub(1);
    for (i, seg) in path.segments.iter().enumerate() {
        let param_name = seg.ident.to_string();
        // Is it a variadic type or is it expecting a variadic type parameter?
        if instances.inst_array.contains_key(&param_name) {
            // The type is a const generic array: f(..., shape: D) -> ()
            // The result produced by this case is not supported syntax.
            return Err(syn_err(
                seg.ident.span(),
                &format!(
                    "Unexpected use of rewrite_path_for_rank for {}",
                    path.to_token_stream()
                ),
            ));
        } else {
            // The last segment of an expression path names a fn/associated
            // item — preserve its ident even when its generic args reference
            // a CGA. All other segments behave like type positions.
            let skip_suffix = context == PathContext::ExprPath && i == last_idx;
            let (last_type_ident, last_seg_args) = match &seg.arguments {
                PathArguments::AngleBracketed(type_params) => {
                    let (type_ident, last_seg_args) =
                        instantiate_cga_args(instances, &seg.ident, type_params, skip_suffix)?;
                    (
                        type_ident.clone(),
                        PathArguments::AngleBracketed(last_seg_args),
                    )
                }
                PathArguments::None => (seg.ident.clone(), PathArguments::None),
                _ => return Err(syn_err(seg.ident.span(), "Unexpected Path arguments.")),
            };
            let result_seg = PathSegment {
                ident: last_type_ident,
                arguments: last_seg_args,
            };
            result_path.segments[i] = result_seg.clone();
        }
    }
    Ok(result_path)
}

/// Desugars variadic types within angle-bracketed generic arguments.
///
/// Handles two CGA-arg shapes:
/// - **Bare CGA name** (e.g. `<E, D>` where `D: [i32; N]`): expand to the
///   per-rank scalar dim refs `D_0, D_1, …, D_{K-1}` so the outer generic
///   accepts them as const args.
/// - **Type referencing a CGA** (e.g. `<E, Tile<E, D>>`): recurse into
///   `rewrite_type_for_rank` which suffixes / expands as appropriate.
fn rewrite_generic_args_for_rank(
    generic_args: &mut AngleBracketedGenericArguments,
    const_instances: &RankBindings,
) -> Result<(), Error> {
    let mut new_args: syn::punctuated::Punctuated<GenericArgument, syn::Token![,]> =
        syn::punctuated::Punctuated::new();
    for arg in &generic_args.args {
        match arg {
            GenericArgument::Type(ty) => {
                if let Type::Path(type_path) = ty {
                    if let Some(ident) = type_path.path.get_ident() {
                        let ident_str = ident.to_string();
                        if let Some(cga) = const_instances.inst_array.get(&ident_str) {
                            for j in 0..cga.length {
                                let dim = format_ident!("{}{}", cga.name, j);
                                new_args.push(parse_quote! { #dim });
                            }
                            continue;
                        }
                    }
                }
                new_args.push(GenericArgument::Type(rewrite_type_for_rank(
                    ty,
                    const_instances,
                )?));
            }
            other => new_args.push(other.clone()),
        }
    }
    generic_args.args = new_args;
    Ok(())
}

/// Recursively desugars const generic array syntax within a type.
fn rewrite_type_for_rank(ty: &Type, instances: &RankBindings) -> Result<Type, Error> {
    // Desugar const generic arrays as they appear as const generic arguments.
    Ok(match ty {
        Type::Path(type_path) => {
            // Special case: For Option<T>, recursively desugar T but don't try to
            // expand Option itself as a variadic type
            let last_segment = type_path.path.segments.last().ok_or_else(|| {
                syn_err(
                    type_path.span(),
                    "Expected at least one path segment in rewrite_type_for_rank",
                )
            })?;
            if last_segment.ident == "Option" {
                let mut result_type = type_path.clone();
                if let PathArguments::AngleBracketed(args) = &last_segment.arguments {
                    let mut new_args = args.clone();
                    // Recursively desugar the type inside Option
                    for arg in &mut new_args.args {
                        if let GenericArgument::Type(inner_ty) = arg {
                            *inner_ty = rewrite_type_for_rank(inner_ty, instances)?;
                        }
                    }
                    let last_idx = result_type.path.segments.len() - 1;
                    result_type.path.segments[last_idx].arguments =
                        PathArguments::AngleBracketed(new_args);
                }
                return Ok(result_type.into());
            }

            let mut result_type = type_path.clone();
            let path = rewrite_path_for_rank(&result_type.path, instances, PathContext::Type)?;
            result_type.path = path;
            // println!("rewrite_type_for_rank: ")
            result_type.into()
        }
        Type::Array(type_array) => {
            let mut result = type_array.clone();
            *result.elem = rewrite_type_for_rank(&type_array.elem, instances)?;
            let arr_len = result.len.to_token_stream().to_string();
            if instances.inst_u32.contains_key(&arr_len) {
                let n = instances.inst_u32.get(&arr_len).unwrap();
                result.len = syn::parse::<Expr>(format!("{}", n).parse().map_err(|_| {
                    syn_err(
                        type_array.span(),
                        &format!("Failed to parse array length '{n}'"),
                    )
                })?)
                .map_err(|e| {
                    syn_err(
                        type_array.span(),
                        &format!("Failed to parse array length '{n}': {e}"),
                    )
                })?;
            }
            result.into()
        }
        Type::Reference(ref_type) => {
            let mut result = ref_type.clone();
            *result.elem = rewrite_type_for_rank(&ref_type.elem, instances)?;
            result.into()
            // unimplemented!("Type::Reference not implemented: {:#?}", ref_type)
        }
        Type::Tuple(tuple_type) => {
            let mut result = tuple_type.clone();
            for elem in &mut result.elems {
                *elem = rewrite_type_for_rank(elem, instances)?;
            }
            Type::Tuple(result)
        }
        _ => ty.clone(),
    })
}

/// Expand a path segment's generic args per rank instance: substitute each CGA arg
/// for its scalar dim refs (e.g. `S` → `S_0, S_1, …`) and (unless
/// `skip_suffix`) suffix the path's ident with the rank.
fn instantiate_cga_args(
    instances: &RankBindings,
    type_ident: &Ident,
    generic_args: &AngleBracketedGenericArguments,
    skip_suffix: bool,
) -> Result<(Ident, AngleBracketedGenericArguments), Error> {
    let mut instantiated_param_name = type_ident.to_string();
    let mut generic_args_result: Vec<String> = vec![];
    let mut rank_suffixes: Vec<u32> = vec![];

    for generic_arg in &generic_args.args {
        match generic_arg {
            GenericArgument::Type(type_param) => {
                match type_param {
                    Type::Path(type_path) => {
                        let last_ident = type_path
                            .path
                            .segments
                            .last()
                            .ok_or_else(|| {
                                syn_err(type_path.span(), "Expected at least one path segment")
                            })?
                            .ident
                            .to_string();
                        if instances.inst_array.contains_key(&last_ident) {
                            // Shape<D> / Tile<E, S> / etc. — substitute the
                            // CGA into rank-instance scalar dims, and (for type
                            // segments) suffix the path ident with the rank.
                            let cga = instances.inst_array.get(&last_ident).unwrap();
                            for j in 0..cga.length {
                                generic_args_result.push(format!("{}{}", cga.name, j));
                            }
                            rank_suffixes.push(cga.length);
                        } else {
                            // Not a const generic array instance, just convert to string
                            // This handles regular types like Tile<i1, S>, Token, etc.
                            generic_args_result.push(generic_arg.to_token_stream().to_string());
                        }
                        // println!("{n_list:?}, expand Type::Path {:?}: {:?}", generic_arg.to_token_stream().to_string(), generic_args_result);
                    }
                    Type::Reference(type_ref) => {
                        // References in generic arguments (e.g., Option<&str>) can be kept as-is
                        generic_args_result.push(type_ref.to_token_stream().to_string());
                    }
                    _ => {
                        generic_args_result.push(generic_arg.to_token_stream().to_string());
                    }
                }
            }
            GenericArgument::Const(const_param) => {
                // println!("expand GenericArgument::Const? {const_param:#?}");
                match const_param {
                    Expr::Block(block_expr) => {
                        // TODO (hme): Would be great to get rid of this syntax.
                        // This is something like Tensor<E, {[...]}>
                        if block_expr.block.stmts.len() != 1 {
                            return Err(syn_err(
                                block_expr.span(),
                                &format!(
                                    "Expected exactly 1 statement in block expression, got {}",
                                    block_expr.block.stmts.len()
                                ),
                            ));
                        }
                        let statement = &block_expr.block.stmts[0];
                        let Stmt::Expr(statement_expr, _) = statement else {
                            return Err(syn_err(block_expr.span(), "Unexpected block expression."));
                        };
                        match statement_expr {
                            Expr::Array(array_expr) => {
                                // This is something like Tensor<E, {[1, 2, -1]}>
                                let rank = array_expr.elems.len();
                                for elem in &array_expr.elems {
                                    generic_args_result
                                        .push(format_rank_const_expr(elem, instances)?);
                                }
                                rank_suffixes.push(rank as u32);
                            }
                            Expr::Repeat(repeat_expr) => {
                                // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                                let thing_to_repeat =
                                    format_rank_const_expr(&repeat_expr.expr, instances)?;
                                let num_repetitions = match &*repeat_expr.len {
                                    Expr::Path(len_path) => {
                                        // This is something like Tensor<E, {[-1; N]}>
                                        let num_rep_var = len_path.to_token_stream().to_string();
                                        if !instances.inst_u32.contains_key(&num_rep_var) {
                                            return Err(syn_err(
                                                len_path.span(),
                                                &format!(
                                                    "Expected instance for generic argument {}",
                                                    num_rep_var
                                                ),
                                            ));
                                        }
                                        let num_repetitions =
                                            *instances.inst_u32.get(&num_rep_var).unwrap();
                                        for _ in 0..num_repetitions {
                                            generic_args_result.push(thing_to_repeat.clone());
                                        }
                                        num_repetitions
                                    }
                                    Expr::Lit(len_lit) => {
                                        // This is something like Tensor<E, {[-1; 3]}>
                                        let num_repetitions: u32 = len_lit
                                            .to_token_stream()
                                            .to_string()
                                            .parse::<u32>()
                                            .map_err(|e| {
                                                syn_err(
                                                    len_lit.span(),
                                                    &format!(
                                                        "Failed to parse repeat length as u32: {e}"
                                                    ),
                                                )
                                            })?;
                                        for _ in 0..num_repetitions {
                                            generic_args_result.push(thing_to_repeat.clone());
                                        }
                                        num_repetitions
                                    }
                                    _ => {
                                        return Err(syn_err(
                                            generic_args.span(),
                                            "Unexpected repeat expression.",
                                        ))
                                    }
                                };
                                rank_suffixes.push(num_repetitions);
                            }
                            _ => {
                                return Err(syn_err(
                                    block_expr.span(),
                                    "Unexpected block expression.",
                                ))
                            }
                        }
                    }
                    Expr::Lit(lit_expr) => {
                        generic_args_result.push(lit_expr.lit.to_token_stream().to_string());
                    }
                    _ => {
                        generic_args_result.push(generic_arg.to_token_stream().to_string());
                    }
                }
            }
            _ => {
                generic_args_result.push(generic_arg.to_token_stream().to_string());
            }
        }
    }
    if !skip_suffix && !rank_suffixes.is_empty() {
        instantiated_param_name = concrete_name(&type_ident.to_string(), &rank_suffixes);
    }
    let instantiated_param_ident = Ident::new(instantiated_param_name.as_str(), type_ident.span());
    let formatted = format!("<{}>", generic_args_result.join(","));
    Ok((
        instantiated_param_ident,
        syn::parse::<AngleBracketedGenericArguments>(formatted.parse().map_err(|_| {
            syn_err(
                type_ident.span(),
                &format!("Failed to parse angle bracketed args '{formatted}'"),
            )
        })?)
        .map_err(|e| {
            syn_err(
                type_ident.span(),
                &format!("Failed to parse angle bracketed args '{formatted}': {e}"),
            )
        })?,
    ))
}

fn format_rank_const_expr(expr: &Expr, instances: &RankBindings) -> Result<String, Error> {
    if let Expr::Index(index) = expr {
        if let Expr::Path(path) = index.expr.as_ref() {
            let name = path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_default();
            if let Some(cga) = instances.inst_array.get(&name) {
                let i = parse_signed_literal_as_i32(&index.index);
                if !(0 <= i && (i as u32) < cga.length) {
                    return Err(syn_err(
                        index.index.span(),
                        &format!(
                            "Index {i} out of bounds for CGA `{}` of length {}",
                            cga.name, cga.length
                        ),
                    ));
                }
                return Ok(format!("{}{}", cga.name, i as u32));
            }
        }
    }
    Ok(expr.to_token_stream().to_string())
}

// ---------------------------------------------------------------------------
// Per-rank AST rewriter
// ---------------------------------------------------------------------------
//
// `RankInstantiator` carries a [`RankBindings`] (the active CGA → length
// mapping for the rank we're specializing to) and walks an item via
// [`syn::visit_mut::VisitMut`], rewriting:
//
//   * **Type paths** (`Tile<E, S>` → `Tile_R<E, S_0, …>`) via
//     [`rewrite_type_for_rank`] in `visit_type_mut`.
//   * **Expression paths** (`iota::<i32, S>` → `iota::<i32, S_0, …>`) — the
//     final segment's ident is preserved (it names a fn / associated item)
//     while CGA args get expanded.
//   * **Struct-literal paths** (`Shape::<{[1,2]}> { … }`) — type-context
//     suffixing.
//   * **CGA references in expressions**: bare `S` → its bound `u32`,
//     `S[i]` → `S_i`.
//   * **`const_shape!` / `const_array!` macro calls** → pre-expanded to
//     `Shape_R::<…>::const_new()` / `Array_R::<…>::const_new()`.
//
// Errors during traversal are accumulated in `self.error` (since `VisitMut`
// methods can't return `Result`) and surfaced by `into_result` at the end.

pub struct RankInstantiator {
    bindings: RankBindings,
    error: Option<Error>,
}

impl RankInstantiator {
    pub fn new(bindings: RankBindings) -> Self {
        Self {
            bindings,
            error: None,
        }
    }

    fn into_result<T>(self, value: T) -> Result<T, Error> {
        match self.error {
            Some(err) => Err(err),
            None => Ok(value),
        }
    }

    /// Rewrite a struct's field types per rank instance. The struct's ident and
    /// generics are the caller's responsibility.
    pub fn rewrite_struct(mut self, item: &ItemStruct) -> Result<ItemStruct, Error> {
        let mut item = item.clone();
        for field in &mut item.fields {
            self.visit_type_mut(&mut field.ty);
        }
        self.into_result(item)
    }

    /// Rewrite a free-fn signature (generics, args, return) and its body.
    pub fn rewrite_function(mut self, item: &ItemFn) -> Result<ItemFn, Error> {
        let mut item = item.clone();
        rewrite_fn_sig(&mut item.sig, &self.bindings)?;
        self.visit_block_mut(&mut item.block);
        self.into_result(item)
    }

    /// Rewrite an impl block: Self type, generics, trait-path args, and
    /// every non-`#[variadic_impl_fn]` method (sig + body).
    pub fn rewrite_impl(mut self, item: &ItemImpl) -> Result<ItemImpl, Error> {
        let mut item = item.clone();
        let original_self_ty = (*item.self_ty).clone();
        *item.self_ty = rewrite_type_for_rank(&item.self_ty, &self.bindings)?;
        rewrite_generics_for_rank(&mut item.generics, &self.bindings)?;
        if let Some(trait_) = &mut item.trait_ {
            let path = &mut trait_.1;
            if path.segments.is_empty() {
                return Err(syn_err(
                    path.span(),
                    "Expected at least one path segment in trait path",
                ));
            }
            let last_seg = path.segments.last_mut().unwrap();
            if let PathArguments::AngleBracketed(path_args) = &mut last_seg.arguments {
                rewrite_generic_args_for_rank(path_args, &self.bindings)?
            }
        }

        let mut impl_items: Vec<ImplItem> = Vec::new();
        for item_in_impl in &mut item.items {
            match item_in_impl {
                ImplItem::Type(type_impl) => {
                    let mut result = type_impl.clone();
                    result.ty = rewrite_type_for_rank(&type_impl.ty, &self.bindings)?;
                    impl_items.push(ImplItem::Type(result));
                }
                ImplItem::Const(c) => impl_items.push(ImplItem::Const(c.clone())),
                ImplItem::Fn(fn_impl) => {
                    if get_meta_list("cuda_tile :: variadic_impl_fn", &fn_impl.attrs).is_some() {
                        // Multi-rank methods now go through trait dispatch;
                        // the inherent variants would be redundant.
                        continue;
                    }
                    let mut result = fn_impl.clone();
                    self.rewrite_impl_method(&original_self_ty, &mut result);
                    if let Some(error) = self.error.take() {
                        return Err(error);
                    }
                    impl_items.push(ImplItem::Fn(result));
                }
                _ => return Err(syn_err(item_in_impl.span(), "Unsupported impl item.")),
            }
        }
        item.items = impl_items;
        self.into_result(item)
    }

    /// Rewrite a single method inside an impl block. Method-level CGAs
    /// extend the rank context for the duration of this method.
    fn rewrite_impl_method(&mut self, _self_ty: &Type, item: &mut ImplItemFn) {
        let cgas = cgas_from_generics(&item.sig.generics);
        let inner_bindings = match self.bindings.instantiate_var_cgas(&cgas) {
            Ok(b) => b,
            Err(e) => {
                self.error = Some(e);
                return;
            }
        };
        let prev = std::mem::replace(&mut self.bindings, inner_bindings);
        if let Err(e) = rewrite_fn_sig(&mut item.sig, &self.bindings) {
            self.error = Some(e);
        } else {
            self.visit_block_mut(&mut item.block);
        }
        self.bindings = prev;
    }

    /// Build `Shape_R::<{[…]}>::const_new()` (or the `Array` equivalent)
    /// from a `const_shape!` / `const_array!` token list. Idents inside the
    /// macro that name an in-scope CGA expand to rank-instance dim refs.
    fn expand_shape_macro(&self, mac: &Macro, kind: &str) -> Result<Expr, Error> {
        let mut args: Vec<String> = Vec::new();
        let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
        let exprs = parser
            .parse2(mac.tokens.clone())
            .map_err(|e| syn_err(mac.span(), &format!("Failed to parse {kind}! args: {e}")))?;
        let expr_count = exprs.len();
        for expr in exprs {
            match &expr {
                Expr::Path(path) if path.path.segments.len() == 1 => {
                    let ident = &path.path.segments[0].ident;
                    let name = ident.to_string();
                    if let Some(cga) = self.bindings.inst_array.get(&name) {
                        if expr_count != 1 {
                            return Err(syn_err(
                                expr.span(),
                                &format!(
                                    "`{name}` names a const generic array; use it alone or index it as `{name}[i]`"
                                ),
                            ));
                        }
                        for i in 0..cga.length {
                            args.push(format!("{}{}", cga.name, i));
                        }
                        continue;
                    }
                }
                Expr::Index(index) => {
                    if let Expr::Path(path) = index.expr.as_ref() {
                        let name = path
                            .path
                            .segments
                            .last()
                            .map(|segment| segment.ident.to_string())
                            .unwrap_or_default();
                        if let Some(cga) = self.bindings.inst_array.get(&name) {
                            let i = parse_signed_literal_as_i32(&index.index);
                            if !(0 <= i && (i as u32) < cga.length) {
                                return Err(syn_err(
                                    index.index.span(),
                                    &format!(
                                        "Index {i} out of bounds for CGA `{}` of length {}",
                                        cga.name, cga.length
                                    ),
                                ));
                            }
                            args.push(format!("{}{}", cga.name, i as u32));
                            continue;
                        }
                    }
                }
                _ => {}
            }
            args.push(expr.to_token_stream().to_string());
        }
        let cga_str = format!("{{[{}]}}", args.join(", "));
        let ty_str = if kind == "const_shape" || kind == "shape" {
            "Shape"
        } else {
            "Array"
        };
        let expr_str = format!("{ty_str}::<{cga_str}>::const_new()");
        syn::parse_str::<Expr>(&expr_str)
            .map_err(|e| syn_err(mac.span(), &format!("Failed to parse '{expr_str}': {e}")))
    }
}

impl VisitMut for RankInstantiator {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if self.error.is_some() {
            return;
        }
        // (1) Pre-expand `const_shape!` / `const_array!` into a typed
        // `const_new()` call so subsequent path/type rewrites can specialize it.
        if let Expr::Macro(em) = expr {
            let name = em
                .mac
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if name == "const_shape" || name == "shape" || name == "const_array" {
                match self.expand_shape_macro(&em.mac, &name) {
                    Ok(new_expr) => {
                        *expr = new_expr;
                        // Continue visiting so the new `Shape::<…>::const_new()`
                        // gets path-rewritten (`Shape` → `Shape_R`).
                        visit_mut::visit_expr_mut(self, expr);
                        return;
                    }
                    Err(e) => {
                        self.error = Some(e);
                        return;
                    }
                }
            }
        }
        // (2) Bare CGA-length name as expression value (e.g. `S` of
        // `const S: [i32; N]`, with `N` bound): substitute its `u32` value.
        if let Expr::Path(ep) = expr {
            if ep.path.segments.len() == 1 {
                let name = ep.path.segments[0].ident.to_string();
                if let Some(n) = self.bindings.inst_u32.get(&name).copied() {
                    *expr = parse_quote! { #n };
                    return;
                }
            }
        }
        // (3) `S[i]` where `S` is an in-scope CGA: replace with `S_i`.
        if let Expr::Index(ei) = expr {
            if let Expr::Path(p) = &*ei.expr {
                let name = p
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if let Some(cga) = self.bindings.inst_array.get(&name).cloned() {
                    let i = parse_signed_literal_as_i32(&ei.index);
                    if !(0 <= i && (i as u32) < cga.length) {
                        self.error = Some(syn_err(
                            ei.index.span(),
                            &format!(
                                "Index {i} out of bounds for CGA `{}` of length {}",
                                cga.name, cga.length
                            ),
                        ));
                        return;
                    }
                    let dim_ident = Ident::new(&format!("{}{}", cga.name, i as u32), p.span());
                    *expr = parse_quote! { #dim_ident };
                    return;
                }
            }
        }
        // Default: recurse into sub-expressions.
        visit_mut::visit_expr_mut(self, expr);
    }

    fn visit_expr_path_mut(&mut self, e: &mut ExprPath) {
        if self.error.is_some() {
            return;
        }
        // Don't fall through to the default visitor — its recursion would
        // visit each path arg as a `Type` (and our `visit_type_mut` would
        // then route a bare-CGA arg like `S` through `rewrite_path_for_rank`,
        // which rejects it). Instead, let `rewrite_path_for_rank` handle the
        // whole path including its args via `instantiate_cga_args`.
        match rewrite_path_for_rank(&e.path, &self.bindings, PathContext::ExprPath) {
            Ok(p) => e.path = p,
            Err(err) => self.error = Some(err),
        }
    }

    fn visit_expr_struct_mut(&mut self, s: &mut ExprStruct) {
        if self.error.is_some() {
            return;
        }
        // Walk field expressions manually — we want to recurse into them,
        // but NOT into the path (whose args may be bare CGAs we shouldn't
        // route through `visit_type_mut`).
        for field in &mut s.fields {
            self.visit_expr_mut(&mut field.expr);
        }
        if let Some(rest) = &mut s.rest {
            self.visit_expr_mut(rest);
        }
        match rewrite_path_for_rank(&s.path, &self.bindings, PathContext::Type) {
            Ok(p) => s.path = p,
            Err(err) => self.error = Some(err),
        }
    }

    fn visit_type_mut(&mut self, ty: &mut Type) {
        if self.error.is_some() {
            return;
        }
        // `rewrite_type_for_rank` handles its own recursion (Array length
        // substitution, Reference/Tuple traversal, Option<T> pass-through,
        // and Type::Path delegation to `rewrite_path_for_rank`).
        match rewrite_type_for_rank(ty, &self.bindings) {
            Ok(t) => *ty = t,
            Err(e) => self.error = Some(e),
        }
    }
}

/// Desugars const generic array syntax in a struct definition.
///
/// Transforms const generic array parameters (e.g., `const D: [i32; N]`) into
/// individual const generic parameters (e.g., `const D_0: i32, const D_1: i32, ...`).
pub fn instantiate_struct_for_rank(item: &ItemStruct) -> Result<ItemStruct, Error> {
    let bindings = RankBindings::from_generics(&item.generics)?;
    RankInstantiator::new(bindings).rewrite_struct(item)
}

/// Desugars const generic array syntax in a function definition.
///
/// Rewrites function signatures and bodies to replace const generic arrays
/// with expanded const generic parameters.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// fn my_fn<const S: [i32; 2]>(x: Tile<f32, S>) -> Shape<S> { }
///
/// // Output:
/// fn my_fn<const S_0: i32, const S_1: i32>(x: Tile_2<f32, S_0, S_1>) -> Shape_2<S_0, S_1> { }
/// ```
pub fn instantiate_function_for_rank(item: &ItemFn) -> Result<ItemFn, Error> {
    let bindings = RankBindings::from_generics(&item.sig.generics)?;
    RankInstantiator::new(bindings).rewrite_function(item)
}

/// Desugars const generic array syntax in a type alias.
///
/// This keeps same-module concrete aliases usable in rustc-visible expanded
/// code, e.g. `type Block = Tensor<f32, {[4]}>`.
pub fn instantiate_type_alias_for_rank(item: &ItemType) -> Result<ItemType, Error> {
    let bindings = RankBindings::from_generics(&item.generics)?;
    let mut item = item.clone();
    rewrite_generics_for_rank(&mut item.generics, &bindings)?;
    *item.ty = rewrite_type_for_rank(&item.ty, &bindings)?;
    Ok(item)
}

/// Desugars const generic array syntax in a static item.
///
/// This keeps `static COUNTER: Global<AtomicI32, { [] }> = Global::new(0i32);`
/// valid in rustc-visible expanded code while preserving the original source
/// text for the JIT.
pub fn instantiate_static_for_rank(item: &ItemStatic) -> Result<ItemStatic, Error> {
    let bindings = RankBindings::new();
    let mut item = item.clone();
    *item.ty = rewrite_type_for_rank(&item.ty, &bindings)?;
    let concrete_type_ident = concrete_type_ident(&item.ty);
    let mut instantiator = RankInstantiator::new(bindings);
    instantiator.visit_expr_mut(&mut item.expr);
    let mut item = instantiator.into_result(item)?;
    if let Some(concrete_type_ident) = concrete_type_ident {
        rewrite_static_constructor_path(&mut item.expr, &concrete_type_ident);
    }
    Ok(item)
}

fn concrete_type_ident(ty: &Type) -> Option<Ident> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    type_path
        .path
        .segments
        .last()
        .map(|segment| segment.ident.clone())
}

fn rewrite_static_constructor_path(expr: &mut Expr, concrete_type_ident: &Ident) {
    let Expr::Call(call) = expr else {
        return;
    };
    let Expr::Path(path) = &mut *call.func else {
        return;
    };
    if path.path.segments.len() < 2 {
        return;
    }
    let Some(last) = path.path.segments.last() else {
        return;
    };
    if last.ident != "new" {
        return;
    }
    if let Some(first) = path.path.segments.first_mut() {
        first.ident = concrete_type_ident.clone();
    }
}

/// Desugars const generic array syntax in an impl block.
///
/// Transforms impl blocks to use desugared const generic parameters.
pub fn instantiate_impl_for_rank(item: &ItemImpl) -> Result<ItemImpl, Error> {
    let bindings = RankBindings::from_generics(&item.generics)?;
    RankInstantiator::new(bindings).rewrite_impl(item)
}
