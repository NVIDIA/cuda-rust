/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[cuda_module]` expansion driver, including nested-module support.

pub(crate) mod constants;
pub(crate) mod contract;
pub(crate) mod launchers;
pub(crate) mod model;

use crate::common::{
    attr_path_ends_with, has_attr_named, impl_trait_parameter_error, internal_ident,
    track_codegen_environment,
};
use crate::cuda_module::constants::{
    collect_cuda_module_constants, cuda_module_items_with_constant_symbols,
    generate_cuda_module_constant_field, generate_cuda_module_constant_initializer,
    generate_cuda_module_constant_resolver_method, generate_cuda_module_set_constant_method,
};
use crate::cuda_module::contract::{
    cuda_module_cluster_dim, cuda_module_cooperative, cuda_module_launch_contract,
};
use crate::cuda_module::launchers::{
    cuda_kernel_marker_name, cuda_module_function_field, generate_cuda_module_async_launch_method,
    generate_cuda_module_kernel_signature, generate_cuda_module_launch_contract_impl,
    generate_cuda_module_launch_method, generate_cuda_module_owned_async_launch_method,
    generate_cuda_module_prepare_launch_methods, has_codegen_generics,
};
use crate::cuda_module::model::{
    CudaModuleKernel, add_cuda_module_disjoint_abi_bounds,
    add_cuda_module_disjoint_contract_bounds, add_cuda_module_uniform_bounds, cuda_module_params,
};
use crate::launch_attrs::{add_launch_bounds_evaluatability_from_attrs, rewrite_loop_unroll_attrs};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use reserved_oxide_symbols::{
    DEVICE_CODEGEN_CRATE_ENV, artifact_anchor_symbol, artifact_anchor_symbol_v2,
    ptx_merge_required_marker,
};
use syn::{
    Ident, Item, ItemFn, ItemMod, LitStr, Token, parse_macro_input, parse_quote,
    punctuated::Punctuated,
};

pub(crate) fn cuda_module_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    track_codegen_environment();
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "cuda_module does not take arguments yet",
        )
        .to_compile_error()
        .into();
    }

    let input = parse_macro_input!(item as ItemMod);
    match expand_cuda_module(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Expands `#[cuda_module]`, emitting the host surface when the `host`
/// feature is on.
pub(crate) fn expand_cuda_module(module: ItemMod) -> syn::Result<TokenStream2> {
    expand_cuda_module_inner(module, cfg!(feature = "host"))
}

/// `expand_cuda_module` with the host-surface decision passed in.
///
/// The decision is a parameter rather than a `cfg!` read inside the body so
/// that tests can exercise both settings from one build. They otherwise
/// could not: this crate dev-depends on `cuda-host`, which turns the `host`
/// feature back on under feature unification even for
/// `cargo test --no-default-features`.
pub(crate) fn expand_cuda_module_inner(
    module: ItemMod,
    emit_host: bool,
) -> syn::Result<TokenStream2> {
    let module_attrs = &module.attrs;
    let vis = &module.vis;
    let ident = &module.ident;
    let Some((_brace, items)) = &module.content else {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "cuda_module requires an inline module so kernel signatures are visible",
        ));
    };

    let constants = collect_cuda_module_constants(items, ident)?;
    let transformed = transform_cuda_module_items(items, &mut Vec::new(), &[], false, emit_host)?;
    if transformed.kernels.is_empty() {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "cuda_module found no #[kernel] functions in this module",
        ));
    }
    reject_conflicting_kernel_names(&transformed.kernels)?;
    reject_reserved_loaded_module(items)?;

    let direct_kernels = &transformed.kernels[..transformed.direct_kernel_count];
    reject_reserved_loaded_module_methods(direct_kernels, false)?;
    let module_items = cuda_module_items_with_constant_symbols(&transformed.items, &constants);

    let non_generic_kernels = direct_kernels.iter().filter(|kernel| !kernel.is_generic);
    let function_fields = non_generic_kernels.clone().map(|kernel| {
        let cfg_attrs = &kernel.cfg_attrs;
        let field = cuda_module_function_field(&kernel.fn_name);
        quote! {
            #(#cfg_attrs)*
            #field: ::cuda_core::CudaFunction,
        }
    });

    let function_initializers = non_generic_kernels.map(|kernel| {
        let cfg_attrs = &kernel.cfg_attrs;
        let field = cuda_module_function_field(&kernel.fn_name);
        let marker = cuda_kernel_marker_name(&kernel.fn_name);
        quote! {
            #(#cfg_attrs)*
            #field: module.load_function(<#marker as ::cuda_host::CudaKernel>::PTX_NAME)?,
        }
    });

    let artifact_anchor_statements = cuda_module_artifact_anchor_statements(&transformed.kernels)?;
    let has_generic = transformed.kernels.iter().any(|k| k.is_generic);
    let ptx_merge_required_markers = transformed.kernels.iter().filter_map(|kernel| {
        if !kernel.is_generic {
            return None;
        }
        let marker = internal_ident(&ptx_merge_required_marker(&kernel.fn_name.to_string()));
        let cfg_attrs = &kernel.effective_cfg_attrs;
        Some(quote! {
            #(#cfg_attrs)*
            // Consumed by the codegen collector. This enabled generic kernel
            // requires run-time PTX bundle merging, which ahead-of-time cubin
            // materialization cannot represent yet.
            #[doc(hidden)]
            #[used]
            #[allow(dead_code, non_upper_case_globals)]
            static #marker: u8 = 0;
        })
    });
    let enable_generic_loader_statements = transformed.kernels.iter().filter_map(|kernel| {
        if !kernel.is_generic {
            return None;
        }
        let cfg_attrs = &kernel.effective_cfg_attrs;
        Some(quote! {
            #(#cfg_attrs)*
            let _ = {
                __cuda_oxide_has_enabled_generic_kernel = true;
            };
        })
    });
    let has_launch_contract = transformed
        .kernels
        .iter()
        .any(|kernel| kernel.launch_contract.is_some());
    // Read the bundle from the binary the linker placed the artifact anchor
    // in, so a module compiled into a shared object (a plugin, a Python
    // extension, an evcxr cell) loads its own artifact instead of searching
    // the process executable. Without an anchor there is nothing to locate
    // and the executable stays the only candidate.
    let embedded_module_loader = quote! {
        match __cuda_oxide_artifact_anchor {
            ::core::option::Option::Some(anchor) => {
                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    ::cuda_host::load_embedded_module_from_anchor(ctx, name, anchor)?
                }
                #[cfg(not(any(target_os = "linux", target_os = "android")))]
                {
                    let _ = anchor;
                    ::cuda_host::load_embedded_module(ctx, name)?
                }
            }
            ::core::option::Option::None => ::cuda_host::load_embedded_module(ctx, name)?,
        }
    };
    let module_loader = if has_generic {
        // A syntactically present generic kernel may be removed by cfg. Make
        // the loader decision under the exact same effective cfg chain as its
        // marker, so eligibility and run-time behavior cannot disagree.
        quote! {
            #[allow(unused_mut)]
            let mut __cuda_oxide_has_enabled_generic_kernel = false;
            #(#enable_generic_loader_statements)*
            let module = if __cuda_oxide_has_enabled_generic_kernel {
                let _ = name; // merged load ignores the crate-name hint
                ::cuda_host::load_all_ptx_bundles_merged(ctx)?
            } else {
                #embedded_module_loader
            };
        }
    } else {
        quote! {
            let module = #embedded_module_loader;
        }
    };
    let constant_fields = constants.iter().map(generate_cuda_module_constant_field);
    let constant_initializers = constants
        .iter()
        .map(generate_cuda_module_constant_initializer);
    let launch_contract_impls = direct_kernels
        .iter()
        .filter_map(generate_cuda_module_launch_contract_impl);
    let kernel_signatures = direct_kernels
        .iter()
        .filter_map(generate_cuda_module_kernel_signature);
    let prepare_launch_methods = direct_kernels
        .iter()
        .filter_map(generate_cuda_module_prepare_launch_methods);
    let launch_methods = direct_kernels
        .iter()
        .map(generate_cuda_module_launch_method);
    let constant_resolver_methods = constants
        .iter()
        .map(generate_cuda_module_constant_resolver_method);
    let set_constant_methods = constants
        .iter()
        .map(generate_cuda_module_set_constant_method);
    let async_module_items = if cfg!(feature = "async") && has_launch_contract {
        quote! {
            /// Loads this package's embedded artifact for a contracted module.
            ///
            /// # Safety
            ///
            /// For a non-generic module, the selected package bundle must be
            /// the artifact compiled from this `cuda_module`; package names are
            /// not yet unique across all library and binary targets. For a
            /// generic module, the merged PTX set must contain each matching
            /// specialization and no conflicting entry definition.
            pub unsafe fn load_async(
                device_id: usize,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::cuda_async::simt::error::DeviceError> {
                // SAFETY: upheld by this function's caller.
                unsafe { load_async_named(device_id, env!("CARGO_PKG_NAME")) }
            }

            /// Loads a caller-selected artifact for this contracted module.
            ///
            /// # Safety
            ///
            /// Every selected kernel must have the exact ABI and resource
            /// semantics declared by this `cuda_module`. A matching symbol
            /// name alone is not sufficient.
            pub unsafe fn load_async_named(
                device_id: usize,
                name: &str,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::cuda_async::simt::error::DeviceError> {
                ::cuda_host::load_cuda_module_from_async_context(device_id, |ctx| {
                    // SAFETY: upheld by this function's caller.
                    unsafe { load_named(ctx, name) }
                })
            }
        }
    } else if cfg!(feature = "async") {
        quote! {
            pub fn load_async(
                device_id: usize,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::cuda_async::simt::error::DeviceError> {
                load_async_named(device_id, env!("CARGO_PKG_NAME"))
            }

            pub fn load_async_named(
                device_id: usize,
                name: &str,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::cuda_async::simt::error::DeviceError> {
                ::cuda_host::load_cuda_module_from_async_context(device_id, |ctx| load_named(ctx, name))
            }
        }
    } else {
        TokenStream2::new()
    };
    let load_definition = if has_launch_contract {
        quote! {
            /// Loads this package's embedded artifact for a contracted module.
            ///
            /// # Safety
            ///
            /// For a non-generic module, the selected package bundle must be
            /// the artifact compiled from this `cuda_module`; package names are
            /// not yet unique across all library and binary targets. For a
            /// generic module, the merged PTX set must contain each matching
            /// specialization and no conflicting entry definition.
            pub unsafe fn load(
                ctx: &::std::sync::Arc<::cuda_core::CudaContext>,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::EmbeddedModuleError> {
                // SAFETY: upheld by this function's caller.
                unsafe { load_named(ctx, env!("CARGO_PKG_NAME")) }
            }
        }
    } else {
        quote! {
            pub fn load(
                ctx: &::std::sync::Arc<::cuda_core::CudaContext>,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::EmbeddedModuleError> {
                load_named(ctx, env!("CARGO_PKG_NAME"))
            }
        }
    };
    let load_named_definition = if has_launch_contract {
        quote! {
            /// Loads a caller-selected artifact for this contracted module.
            ///
            /// # Safety
            ///
            /// Every selected kernel must have the exact ABI and resource
            /// semantics declared by this `cuda_module`. A matching symbol
            /// name alone is not sufficient.
            pub unsafe fn load_named(
                ctx: &::std::sync::Arc<::cuda_core::CudaContext>,
                name: &str,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::EmbeddedModuleError> {
                #artifact_anchor_statements
                #module_loader
                // SAFETY: upheld by this function's caller.
                unsafe { from_module(module) }.map_err(::cuda_host::EmbeddedModuleError::Driver)
            }
        }
    } else {
        quote! {
            pub fn load_named(
                ctx: &::std::sync::Arc<::cuda_core::CudaContext>,
                name: &str,
            ) -> ::core::result::Result<LoadedModule, ::cuda_host::EmbeddedModuleError> {
                #artifact_anchor_statements
                #module_loader
                from_module(module).map_err(::cuda_host::EmbeddedModuleError::Driver)
            }
        }
    };
    let from_module_definition = if has_launch_contract {
        quote! {
            /// Binds caller-provided CUDA code to this module's launch API.
            ///
            /// # Safety
            ///
            /// Every loaded kernel must have the exact ABI and resource
            /// semantics declared by this `cuda_module`. A matching symbol
            /// name alone is not sufficient.
            pub unsafe fn from_module(
                module: ::std::sync::Arc<::cuda_core::CudaModule>,
            ) -> ::core::result::Result<LoadedModule, ::cuda_core::DriverError> {
                Ok(LoadedModule {
                    __module: module.clone(),
                    __generic_functions: ::std::sync::Arc::new(
                        ::std::sync::Mutex::new(::std::collections::HashMap::new())
                    ),
                    #(#function_initializers)*
                    #(#constant_initializers)*
                })
            }
        }
    } else {
        quote! {
            pub fn from_module(
                module: ::std::sync::Arc<::cuda_core::CudaModule>,
            ) -> ::core::result::Result<LoadedModule, ::cuda_core::DriverError> {
                Ok(LoadedModule {
                    __module: module.clone(),
                    __generic_functions: ::std::sync::Arc::new(
                        ::std::sync::Mutex::new(::std::collections::HashMap::new())
                    ),
                    #(#function_initializers)*
                    #(#constant_initializers)*
                })
            }
        }
    };
    let async_launch_methods = if cfg!(feature = "async") {
        let async_launch_methods = direct_kernels
            .iter()
            .map(generate_cuda_module_async_launch_method);
        let owned_async_launch_methods = direct_kernels
            .iter()
            .map(generate_cuda_module_owned_async_launch_method);
        quote! {
            #(#async_launch_methods)*
            #(#owned_async_launch_methods)*
        }
    } else {
        TokenStream2::new()
    };

    // Everything below names `::cuda_host` or `::cuda_core`. The kernels
    // themselves, and the PTX-merge markers the codegen collector consumes, do
    // not -- so a crate that only compiles kernels can take cuda-macros with
    // `default-features = false` and stop depending on the host stack.
    let host_items = if emit_host {
        quote! {
            #(#kernel_signatures)*
            #(#launch_contract_impls)*

            #[derive(Clone, Debug)]
            #[allow(non_snake_case)]
            pub struct LoadedModule {
                __module: ::std::sync::Arc<::cuda_core::CudaModule>,
                __generic_functions: ::std::sync::Arc<
                    ::std::sync::Mutex<
                        ::std::collections::HashMap<&'static str, ::cuda_core::CudaFunction>
                    >
                >,
                #(#function_fields)*
                #(#constant_fields)*
            }

            #load_definition

            #load_named_definition

            #from_module_definition

            #async_module_items

            impl LoadedModule {
                pub fn as_cuda_module(&self) -> &::std::sync::Arc<::cuda_core::CudaModule> {
                    &self.__module
                }

                #(#launch_methods)*
                #(#prepare_launch_methods)*
                #(#constant_resolver_methods)*
                #(#set_constant_methods)*
                #async_launch_methods
            }
        }
    } else {
        TokenStream2::new()
    };

    Ok(quote! {
        #(#module_attrs)*
        #vis mod #ident {
            #(#module_items)*
            #(#ptx_merge_required_markers)*
            #host_items
        }
    })
}

pub(crate) struct CudaModuleLevel {
    pub(crate) items: Vec<Item>,
    pub(crate) kernels: Vec<CudaModuleKernel>,
    pub(crate) direct_kernel_count: usize,
}

/// Recursively rewrite only inline child modules.
///
/// Every module that owns a kernel (or contains a deeper module that does)
/// receives its own `LoadedModule`. That keeps generated method signatures in
/// the same Rust scope as the source kernel. File-backed modules and
/// `include!` invocations are preserved but not traversed: their contents are
/// not present in an attribute macro's input token stream, and reproducing
/// rustc's module loader in a proc macro is neither complete nor hygienic.
pub(crate) fn transform_cuda_module_items(
    items: &[Item],
    module_path: &mut Vec<Ident>,
    ancestor_cfg_attrs: &[syn::Attribute],
    generate_nested_support: bool,
    emit_host: bool,
) -> syn::Result<CudaModuleLevel> {
    let mut transformed_items = Vec::with_capacity(items.len());
    let mut direct_kernels = Vec::new();
    let mut descendant_kernels = Vec::new();

    for item in items {
        match item {
            Item::Fn(item_fn) => {
                if let Some(kernel) = cuda_module_kernel(item_fn, module_path, ancestor_cfg_attrs)?
                {
                    direct_kernels.push(kernel);
                }
                transformed_items.push(item.clone());
            }
            Item::Mod(item_mod) => {
                let Some((_brace, nested_items)) = &item_mod.content else {
                    // Attribute macros receive the declaration, not the file's
                    // contents. Preserve it exactly, but do not pretend that
                    // kernels behind this boundary were discovered.
                    transformed_items.push(item.clone());
                    continue;
                };

                module_path.push(item_mod.ident.clone());
                let mut nested_cfg_attrs = ancestor_cfg_attrs.to_vec();
                nested_cfg_attrs.extend(cuda_module_cfg_attrs(&item_mod.attrs)?);
                let nested = transform_cuda_module_items(
                    nested_items,
                    module_path,
                    &nested_cfg_attrs,
                    true,
                    emit_host,
                )?;
                module_path.pop();

                let mut transformed_mod = item_mod.clone();
                transformed_mod
                    .content
                    .as_mut()
                    .expect("inline module content disappeared")
                    .1 = nested.items;
                descendant_kernels.extend(nested.kernels);
                transformed_items.push(Item::Mod(transformed_mod));
            }
            _ => transformed_items.push(item.clone()),
        }
    }

    let direct_kernel_count = direct_kernels.len();
    let mut kernels = direct_kernels;
    kernels.extend(descendant_kernels);

    if generate_nested_support && !kernels.is_empty() {
        reject_reserved_loaded_module(items)?;
        reject_reserved_loaded_module_methods(&kernels[..direct_kernel_count], true)?;
        let support =
            generate_nested_cuda_module_support(&kernels[..direct_kernel_count], emit_host);
        let mut support_items = syn::parse2::<syn::File>(support)?.items;
        transformed_items.append(&mut support_items);
    }

    Ok(CudaModuleLevel {
        items: transformed_items,
        kernels,
        direct_kernel_count,
    })
}

fn reject_reserved_loaded_module(items: &[Item]) -> syn::Result<()> {
    for item in items {
        let ident = match item {
            Item::Const(item) => Some(&item.ident),
            Item::Enum(item) => Some(&item.ident),
            Item::ExternCrate(item) => item
                .rename
                .as_ref()
                .map(|(_as_token, rename)| rename)
                .or(Some(&item.ident)),
            Item::Fn(item) => Some(&item.sig.ident),
            Item::Mod(item) => Some(&item.ident),
            Item::Static(item) => Some(&item.ident),
            Item::Struct(item) => Some(&item.ident),
            Item::Trait(item) => Some(&item.ident),
            Item::TraitAlias(item) => Some(&item.ident),
            Item::Type(item) => Some(&item.ident),
            Item::Union(item) => Some(&item.ident),
            Item::Use(item) => cuda_module_loaded_module_use_binding(&item.tree),
            _ => None,
        };
        if ident.is_some_and(|ident| cuda_module_ident_key(ident) == "LoadedModule") {
            return Err(syn::Error::new_spanned(
                ident.expect("checked above"),
                "#[cuda_module] reserves the name `LoadedModule` in every inline namespace that contains kernels",
            ));
        }
    }
    Ok(())
}

fn reject_reserved_loaded_module_methods(
    kernels: &[CudaModuleKernel],
    nested: bool,
) -> syn::Result<()> {
    for kernel in kernels {
        let name = cuda_module_ident_key(&kernel.fn_name);
        let reserved = name == "as_cuda_module" || (nested && name == "from_parent");
        if reserved {
            let scope = if name == "from_parent" {
                "nested kernel namespaces"
            } else {
                "every kernel namespace"
            };
            return Err(syn::Error::new_spanned(
                &kernel.fn_name,
                format!(
                    "#[cuda_module] reserves launcher method name `{name}` in {scope}; rename the kernel"
                ),
            ));
        }
    }
    Ok(())
}

/// Return the collision key for generated Rust and PTX namespaces.
///
/// `syn::Ident::to_string()` preserves a raw prefix, but `step` and `r#step`
/// name the same Rust identifier and must collide in cuda-oxide's bare PTX
/// namespace. Guards compare this normalized key before generating anything.
fn cuda_module_ident_key(ident: &Ident) -> String {
    let spelling = ident.to_string();
    spelling.strip_prefix("r#").unwrap_or(&spelling).to_string()
}

fn cuda_module_loaded_module_use_binding(tree: &syn::UseTree) -> Option<&Ident> {
    match tree {
        syn::UseTree::Path(path) => cuda_module_loaded_module_use_binding(&path.tree),
        syn::UseTree::Name(name) => {
            (cuda_module_ident_key(&name.ident) == "LoadedModule").then_some(&name.ident)
        }
        syn::UseTree::Rename(rename) => {
            (cuda_module_ident_key(&rename.rename) == "LoadedModule").then_some(&rename.rename)
        }
        syn::UseTree::Group(group) => group
            .items
            .iter()
            .find_map(cuda_module_loaded_module_use_binding),
        syn::UseTree::Glob(_) => None,
    }
}

fn cuda_module_kernel(
    item_fn: &ItemFn,
    module_path: &[Ident],
    ancestor_cfg_attrs: &[syn::Attribute],
) -> syn::Result<Option<CudaModuleKernel>> {
    if !has_attr_named(&item_fn.attrs, "kernel") {
        return Ok(None);
    }
    if let Some(err) = impl_trait_parameter_error(item_fn, "kernel") {
        return Err(err);
    }
    let cluster_dim = cuda_module_cluster_dim(&item_fn.attrs)?;
    let cooperative = cuda_module_cooperative(&item_fn.attrs)?;
    let params = cuda_module_params(item_fn)?;
    let launch_contract =
        cuda_module_launch_contract(&item_fn.attrs, &item_fn.sig.ident, &params, cluster_dim)?;
    // `#[cuda_module]` expands before both the nested function attributes and
    // the recursive module rewrite. Mirror the evaluatability predicates that
    // `#[launch_bounds]` and `#[kernel]` add to each concrete entry while
    // retaining the nested module's path and inherited cfg attributes.
    let mut configured_item = item_fn.clone();
    rewrite_loop_unroll_attrs(&mut configured_item)?;
    add_launch_bounds_evaluatability_from_attrs(&mut configured_item)?;
    let mut generics = configured_item.sig.generics;
    if let Some(contract) = launch_contract.as_ref() {
        add_cuda_module_disjoint_contract_bounds(&mut generics, &params, contract.domain);
    }
    // A `Uniform` parameter carries its proof with or without a launch
    // contract, so this bound is not conditional on one.
    add_cuda_module_uniform_bounds(&mut generics, &params);
    // The launch packet's shape (two or three words per slice) must match the
    // resolved device type with or without a launch contract, so this bound
    // is unconditional too.
    add_cuda_module_disjoint_abi_bounds(&mut generics, &params);
    let is_generic = has_codegen_generics(&item_fn.sig.generics);
    let cfg_attrs = cuda_module_cfg_attrs(&item_fn.attrs)?;
    let mut effective_cfg_attrs = ancestor_cfg_attrs.to_vec();
    effective_cfg_attrs.extend(cfg_attrs.clone());
    // PreparedLaunch proves launch geometry and declared resources, while
    // Copy/Freeze only describe the copied parameter storage. None proves
    // device accessibility or synchronization of values reached through its
    // fields. Keep that proof at the host launch boundary for every grid
    // parameter, including generic and otherwise safe source kernels.
    let launch_unsafety = item_fn.sig.unsafety.or_else(|| {
        params
            .iter()
            .any(|parameter| parameter.grid_constant)
            .then(|| parse_quote!(unsafe))
    });
    Ok(Some(CudaModuleKernel {
        module_path: module_path.to_vec(),
        vis: item_fn.vis.clone(),
        cfg_attrs,
        effective_cfg_attrs,
        method_attrs: cuda_module_method_attrs(&item_fn.attrs),
        launch_unsafety,
        fn_name: item_fn.sig.ident.clone(),
        generics,
        params,
        cluster_dim,
        cooperative,
        launch_contract,
        is_generic,
    }))
}

fn generate_nested_cuda_module_support(
    kernels: &[CudaModuleKernel],
    emit_host: bool,
) -> TokenStream2 {
    let launch_contract_impls = kernels
        .iter()
        .filter_map(generate_cuda_module_launch_contract_impl);
    let kernel_signatures = kernels
        .iter()
        .filter_map(generate_cuda_module_kernel_signature);
    let prepare_launch_methods = kernels
        .iter()
        .filter_map(generate_cuda_module_prepare_launch_methods);
    let non_generic_kernels = kernels.iter().filter(|kernel| !kernel.is_generic);
    let function_fields = non_generic_kernels.clone().map(|kernel| {
        let cfg_attrs = &kernel.cfg_attrs;
        let field = cuda_module_function_field(&kernel.fn_name);
        quote! {
            #(#cfg_attrs)*
            #field: ::cuda_core::CudaFunction,
        }
    });
    let function_initializers = non_generic_kernels.map(|kernel| {
        let cfg_attrs = &kernel.cfg_attrs;
        let field = cuda_module_function_field(&kernel.fn_name);
        let marker = cuda_kernel_marker_name(&kernel.fn_name);
        quote! {
            #(#cfg_attrs)*
            #field: module.load_function(<#marker as ::cuda_host::CudaKernel>::PTX_NAME)?,
        }
    });
    let launch_methods = kernels.iter().map(generate_cuda_module_launch_method);
    let async_launch_methods = if cfg!(feature = "async") {
        let borrowed = kernels.iter().map(generate_cuda_module_async_launch_method);
        let owned = kernels
            .iter()
            .map(generate_cuda_module_owned_async_launch_method);
        quote! {
            #(#borrowed)*
            #(#owned)*
        }
    } else {
        TokenStream2::new()
    };

    // Host-only, exactly as in `expand_cuda_module_inner`; see the note there.
    if !emit_host {
        return TokenStream2::new();
    }

    quote! {
        #(#kernel_signatures)*
        #(#launch_contract_impls)*

        #[derive(Clone, Debug)]
        #[allow(non_snake_case)]
        pub struct LoadedModule {
            __module: ::std::sync::Arc<::cuda_core::CudaModule>,
            __generic_functions: ::std::sync::Arc<
                ::std::sync::Mutex<
                    ::std::collections::HashMap<&'static str, ::cuda_core::CudaFunction>
                >
            >,
            #(#function_fields)*
        }

        impl LoadedModule {
            /// Bind this namespace's launchers to a module loaded by its
            /// immediate parent namespace.
            pub fn from_parent(
                parent: &super::LoadedModule,
            ) -> ::core::result::Result<Self, ::cuda_core::DriverError> {
                let module = parent.as_cuda_module().clone();
                Ok(Self {
                    __module: module.clone(),
                    __generic_functions: parent.__generic_functions.clone(),
                    #(#function_initializers)*
                })
            }

            pub fn as_cuda_module(&self) -> &::std::sync::Arc<::cuda_core::CudaModule> {
                &self.__module
            }

            #(#launch_methods)*
            #(#prepare_launch_methods)*
            #async_launch_methods
        }
    }
}

/// Reject duplicate bare kernel names anywhere in one `#[cuda_module]` tree.
///
/// Launcher methods are namespace-qualified, but cuda-oxide's current PTX
/// entry naming is not: `stage1::step` and `stage2::step` both export `step`.
/// Until the backend and `#[kernel]` share a qualified entry-name contract,
/// accepting the pair would risk resolving a launcher to the wrong entry.
/// The restriction is therefore syntactic and also applies to cfg-gated
/// alternatives; proc macros cannot prove that arbitrary cfg predicates are
/// mutually exclusive.
fn reject_conflicting_kernel_names(kernels: &[CudaModuleKernel]) -> syn::Result<()> {
    let mut names: std::collections::HashMap<String, &CudaModuleKernel> =
        std::collections::HashMap::new();
    for kernel in kernels {
        let name = cuda_module_ident_key(&kernel.fn_name);
        if let Some(previous) = names.insert(name.clone(), kernel) {
            return Err(syn::Error::new(
                kernel.fn_name.span(),
                format!(
                    "cuda-oxide PTX entry names are currently bare function names, so \
                     #[cuda_module] requires kernel names to be unique across its inline \
                     module tree: `{name}` in {second} conflicts with `{name}` in {first}; \
                     rename one of the kernels",
                    name = name,
                    first = cuda_module_path_description(&previous.module_path),
                    second = cuda_module_path_description(&kernel.module_path),
                ),
            ));
        }
    }
    Ok(())
}

fn cuda_module_path_description(module_path: &[Ident]) -> String {
    if module_path.is_empty() {
        "the module root".to_string()
    } else {
        format!(
            "`{}`",
            module_path
                .iter()
                .map(Ident::to_string)
                .collect::<Vec<_>>()
                .join("::")
        )
    }
}

/// Generate the statements that pin this crate's embedded device artifact
/// into the final binary.
///
/// The codegen backend stores each crate's compiled device code (PTX,
/// cubin, NVVM IR, or LTOIR) in a `.oxart` data section of a small extra
/// object file. When the crate that holds the `#[cuda_module]` is a
/// *library*, that object becomes one member of the crate's `.rlib`
/// archive, and linkers only extract an archive member when it defines a
/// symbol that some already-linked object references. The backend defines
/// a global anchor symbol inside the artifact object for exactly this
/// purpose; here we emit the matching reference. Reading the anchor's
/// address through `black_box` inside `load_named()` means that any
/// program calling `load()` carries an undefined reference to the anchor,
/// which forces the linker to pull the artifact member out of the rlib.
/// Without this handshake the bundle was silently dropped and `load()`
/// failed at runtime with `ModuleNotFound` (issue #72).
///
/// Without an owner filter, both sides keep using the legacy package+version
/// anchor for compatibility with older wrappers and backends. A non-empty
/// owner filter activates the v2 package+version+crate+binary identity. That
/// target-specific identity prevents an unselected binary from satisfying a
/// selected library's reference (or vice versa); an unselected new macro emits
/// no reference at all. The backend also keeps a weak legacy alias for older
/// macro expansions in mixed-version builds.
///
/// The reference is only emitted when the module is guaranteed to produce
/// an artifact for this crate. Generic kernels are monomorphized (and
/// their PTX embedded) in the *consuming* crate, so a module with only
/// generic kernels yields no artifact here, and an anchor reference would
/// be an undefined-symbol link error. The same reasoning extends to
/// cfg-gated kernels: root `load()` emits one equivalent guarded reference per
/// concrete kernel in the complete inline tree. Each reference carries the
/// kernel's effective ancestor-plus-local availability attributes, so a module
/// containing only nested kernels is still independently loadable while no
/// anchor is referenced when every concrete kernel is absent.
///
/// The anchor's address also tells `load_named()` where the bundle is: the
/// linker keeps the anchor and the `.oxart` section in the same binary, so
/// the loader reads that binary rather than the process executable, which
/// differs whenever the module was compiled into a shared object. Each
/// reference records the address in `__cuda_oxide_artifact_anchor`; it stays
/// `None` when no reference is emitted or every concrete kernel is cfg'd out.
fn cuda_module_artifact_anchor_statements(
    kernels: &[CudaModuleKernel],
) -> syn::Result<TokenStream2> {
    let references = cuda_module_artifact_anchor_references(kernels)?;
    Ok(quote! {
        // Keep-alive handshake with the codegen backend: see the macro
        // crate's `cuda_module_artifact_anchor_statements` for details.
        #[allow(unused_mut)]
        let mut __cuda_oxide_artifact_anchor: ::core::option::Option<&::core::primitive::u8> =
            ::core::option::Option::None;
        #(#references)*
    })
}

/// One guarded anchor reference per concrete kernel, or none when this crate
/// produces no artifact; see [`cuda_module_artifact_anchor_statements`].
fn cuda_module_artifact_anchor_references(
    kernels: &[CudaModuleKernel],
) -> syn::Result<Vec<TokenStream2>> {
    let (Ok(package_name), Ok(package_version), Ok(crate_name)) = (
        std::env::var("CARGO_PKG_NAME"),
        std::env::var("CARGO_PKG_VERSION"),
        std::env::var("CARGO_CRATE_NAME"),
    ) else {
        // Not built by cargo (e.g. a raw rustc invocation): the backend
        // falls back to crate-name-based bundle naming and we cannot
        // reproduce it exactly, so skip the anchor rather than risk an
        // undefined symbol.
        return Ok(Vec::new());
    };

    let owner_filter = proc_macro::tracked::env_var(DEVICE_CODEGEN_CRATE_ENV).ok();
    let owner_selection = device_codegen_owner_selection(owner_filter.as_deref(), &crate_name);
    if owner_selection == Some(false) {
        // The backend deliberately omits this crate's artifact. Omitting the
        // reference as well keeps the host link valid without pretending that
        // a loadable bundle exists.
        return Ok(Vec::new());
    }

    if !kernels.iter().any(|kernel| !kernel.is_generic) {
        return Ok(Vec::new());
    }

    let binary_name = std::env::var("CARGO_BIN_NAME").ok();
    let anchor = if owner_selection.is_some() {
        artifact_anchor_symbol_v2(
            &package_name,
            &package_version,
            &crate_name,
            binary_name.as_deref(),
        )
    } else {
        artifact_anchor_symbol(&package_name, &package_version)
    };
    let anchor_name = LitStr::new(&anchor, proc_macro2::Span::call_site());
    let references = kernels
        .iter()
        .filter(|kernel| !kernel.is_generic)
        .map(|kernel| {
            let cfg_attrs = &kernel.effective_cfg_attrs;
            quote! {
                #(#cfg_attrs)*
                let _ = {
                    unsafe extern "C" {
                        #[link_name = #anchor_name]
                        static CUDA_OXIDE_BUNDLE_ANCHOR: ::core::primitive::u8;
                    }
                    __cuda_oxide_artifact_anchor = ::core::option::Option::Some(
                        ::std::hint::black_box(unsafe {
                            // The backend defines this initialized byte in the
                            // artifact object retained by this image. Borrow it
                            // only for this loader call; no reference escapes.
                            &CUDA_OXIDE_BUNDLE_ANCHOR
                        }),
                    );
                };
            }
        })
        .collect();
    Ok(references)
}

pub(crate) fn device_codegen_owner_selection(raw: Option<&str>, crate_name: &str) -> Option<bool> {
    let crate_name = crate_name.trim().replace('-', "_");
    raw.and_then(|raw| {
        let owners: Vec<_> = raw
            .split(',')
            .map(|name| name.trim().replace('-', "_"))
            .filter(|name| !name.is_empty())
            .collect();
        (!owners.is_empty()).then(|| owners.iter().any(|owner| owner == &crate_name))
    })
}

pub(super) fn cuda_module_method_attrs(attrs: &[syn::Attribute]) -> Vec<syn::Attribute> {
    attrs
        .iter()
        .filter(|attr| attr_path_ends_with(attr, "doc"))
        .cloned()
        .collect()
}

/// Copy only the part of `cfg` / `cfg_attr` that controls whether an item
/// exists. A kernel may use `cfg_attr` for function-only attributes such as
/// `inline`; copying those onto generated fields or statements would be
/// invalid. The nested `cfg` / `cfg_attr` availability semantics are retained;
/// unrelated conditional attributes are omitted from generated items.
pub(super) fn cuda_module_cfg_attrs(attrs: &[syn::Attribute]) -> syn::Result<Vec<syn::Attribute>> {
    let mut filtered = Vec::new();
    for attr in attrs {
        if attr_path_ends_with(attr, "cfg") {
            filtered.push(attr.clone());
        } else if attr_path_ends_with(attr, "cfg_attr")
            && let Some(attr) = filter_cuda_module_cfg_attr(attr)?
        {
            filtered.push(attr);
        }
    }
    Ok(filtered)
}

fn filter_cuda_module_cfg_attr(attr: &syn::Attribute) -> syn::Result<Option<syn::Attribute>> {
    let syn::Meta::List(list) = &attr.meta else {
        return Err(syn::Error::new_spanned(attr, "cfg_attr requires arguments"));
    };
    let parser = Punctuated::<syn::Meta, Token![,]>::parse_terminated;
    let mut args = syn::parse::Parser::parse2(parser, list.tokens.clone())?.into_iter();
    let predicate = args
        .next()
        .ok_or_else(|| syn::Error::new_spanned(attr, "cfg_attr requires a predicate"))?;
    let mut nested = Vec::new();
    for meta in args {
        if let Some(meta) = filter_cuda_module_cfg_meta(meta)? {
            nested.push(meta);
        }
    }
    if nested.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parse_quote!(#[cfg_attr(#predicate, #(#nested),*)])))
    }
}

fn filter_cuda_module_cfg_meta(meta: syn::Meta) -> syn::Result<Option<syn::Meta>> {
    if meta
        .path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "cfg")
    {
        return Ok(Some(meta));
    }
    if meta
        .path()
        .segments
        .last()
        .is_none_or(|segment| segment.ident != "cfg_attr")
    {
        return Ok(None);
    }
    filter_cuda_module_nested_cfg_attr(meta)
}

fn filter_cuda_module_nested_cfg_attr(meta: syn::Meta) -> syn::Result<Option<syn::Meta>> {
    let syn::Meta::List(list) = &meta else {
        return Err(syn::Error::new_spanned(meta, "cfg_attr requires arguments"));
    };
    let parser = Punctuated::<syn::Meta, Token![,]>::parse_terminated;
    let mut args = syn::parse::Parser::parse2(parser, list.tokens.clone())?.into_iter();
    let predicate = args
        .next()
        .ok_or_else(|| syn::Error::new_spanned(&meta, "cfg_attr requires a predicate"))?;
    let mut nested = Vec::new();
    for meta in args {
        if let Some(meta) = filter_cuda_module_cfg_meta(meta)? {
            nested.push(meta);
        }
    }
    if nested.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parse_quote!(cfg_attr(#predicate, #(#nested),*))))
    }
}
