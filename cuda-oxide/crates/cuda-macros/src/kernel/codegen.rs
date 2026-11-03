/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel body generation: simple kernels, generic instantiations, and
//! the hidden device twins.

use crate::common::{attr_path_ends_with, internal_ident};
use crate::cuda_module::launchers::{
    codegen_generic_arguments, generic_arguments, generic_phantom_type,
};
use crate::kernel::scope::{
    append_kernel_scope_parameter, explicit_kernel_scope, explicit_kernel_scope_bindings,
    find_closure_generic, find_closure_param, forwarding_inputs, inject_thread_index_scope,
    is_unchecked_indexing_config_marker, kernel_scope_binding, rewrite_thread_index_calls,
    strip_grid_constant_config_markers, strip_unchecked_indexing_config_marker,
    top_level_kernel_configuration_markers, unchecked_indexing_impl_clone,
};
use crate::launch_attrs::validate_routed_launch_contract_requires;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use reserved_oxide_symbols::{INSTANTIATE_PREFIX, KERNEL_PREFIX};
use syn::{FnArg, GenericParam, Ident, ItemFn, Pat, Type};

/// Generate a generic kernel that will be instantiated from call sites (nvcc-style)
pub(super) fn generate_generic_kernel_no_instantiation(
    input: ItemFn,
    explicit_scope: Option<Ident>,
) -> TokenStream {
    generic_kernel_no_instantiation_tokens(input, explicit_scope).into()
}

/// Expansion body of [`generate_generic_kernel_no_instantiation`], split out
/// so unit tests can inspect the generated items without a live proc-macro
/// bridge.
pub(crate) fn generic_kernel_no_instantiation_tokens(
    mut input: ItemFn,
    explicit_scope: Option<Ident>,
) -> TokenStream2 {
    let entry_inputs = input.sig.inputs.clone();
    // A routed `#[launch_contract]` expands later on the generated entry
    // wrapper, whose synthetic parameter names cannot resolve source-level
    // `requires` identifiers, so those relations are validated here while the
    // original signature is still in scope.
    if let Err(error) = validate_routed_launch_contract_requires(&input.attrs, &entry_inputs) {
        return error.to_compile_error();
    }
    let has_explicit_scope = explicit_scope.is_some();
    let rewritten_scope = if let Some(ident) = explicit_scope {
        Some(explicit_kernel_scope(&mut input, ident))
    } else {
        rewrite_thread_index_calls(&mut input, false)
    };
    if let Some(scope) = &rewritten_scope {
        append_kernel_scope_parameter(&mut input, scope);
    }

    // Attributes written below `#[kernel]` still belong to the source item
    // when this macro runs. Route CUDA entry directives to the generated entry
    // function, keep ordinary Rust attributes on the user-facing
    // implementation, and copy cfg gates to every generated item.
    let (implementation_attrs, entry_attrs, cfg_attrs) = route_generic_kernel_attrs(&input.attrs);
    let entry_config_markers = top_level_kernel_configuration_markers(&input);
    strip_grid_constant_config_markers(&mut input);
    // The unchecked-indexing marker may only live in generated kernel entry
    // functions and their hidden unchecked twin, never in the user-named
    // implementation helper: a marker left in the helper would extend
    // bounds-check elision to every other kernel that calls (and inlines)
    // the helper, without those kernels opting in.
    let unchecked_indexing = input
        .block
        .stmts
        .iter()
        .any(is_unchecked_indexing_config_marker);
    if unchecked_indexing {
        strip_unchecked_indexing_config_marker(&mut input);
    }
    let unchecked_impl =
        unchecked_indexing.then(|| unchecked_indexing_impl_clone(&input, &implementation_attrs));
    // The entry calls the hidden twin, so a private opted kernel's user-named
    // helper may otherwise be dead code; it stays emitted (bounds-checked)
    // for other device code to call.
    let helper_dead_code = if unchecked_indexing {
        quote! { #[allow(dead_code)] }
    } else {
        quote! {}
    };
    let fn_name = &input.sig.ident;
    let vis = &input.vis;
    let generics = &input.sig.generics;
    let where_clause = &input.sig.generics.where_clause;
    let inputs = &input.sig.inputs;
    let output = &input.sig.output;
    let constness = &input.sig.constness;
    let unsafety = &input.sig.unsafety;
    let abi = &input.sig.abi;
    let block = &input.block;

    let kernel_name = format_ident!("{}{}", KERNEL_PREFIX, fn_name);
    let instantiate_name = format_ident!("{}{}", INSTANTIATE_PREFIX, fn_name);

    let (wrapper_inputs, arg_names) = match forwarding_inputs(&entry_inputs) {
        Ok(forwarding) => forwarding,
        Err(err) => return err.to_compile_error(),
    };
    let args_info: Vec<_> = arg_names
        .iter()
        .zip(entry_inputs.iter())
        .filter_map(|(name, arg)| {
            let FnArg::Typed(pat_type) = arg else {
                return None;
            };
            Some((name, &*pat_type.ty))
        })
        .collect();

    // Find the closure generic type (looks for Fn/FnMut/FnOnce bounds)
    let closure_generic = find_closure_generic(generics);

    // Function turbofish arguments contain both types and consts in source
    // order. Lifetimes are inferred and therefore do not appear here.
    let codegen_args = codegen_generic_arguments(generics);
    let kernel_turbofish = if codegen_args.is_empty() {
        quote! {}
    } else {
        quote! { ::<#(#codegen_args),*> }
    };
    let ptx_name_fn = format_ident!("{}_ptx_name", fn_name);
    let instantiate_helper = if let Some(closure_type_name) = closure_generic {
        if let Some((_closure_idx, (_closure_name, closure_type))) =
            find_closure_param(&args_info, &closure_type_name)
        {
            quote! {
                /// Auto-generated helper to force kernel monomorphization.
                ///
                /// Takes the closure by *reference* so its anonymous type
                /// is bound to the generic parameter `F` at the call site
                /// without moving the closure — the caller still needs the
                /// closure value to push as the kernel argument right
                /// after. Then forces rustc to emit a CGU entry for the
                /// concrete `#kernel_name::<...>` instantiation. Returns
                /// the PTX export name produced by the kernel's
                /// `GenericCudaKernel::ptx_name()` impl, which is the
                /// single source of truth for the on-wire name on the
                /// host side.
                ///
                /// Bound is intentionally not `'static`: closures that
                /// borrow non-`'static` data (e.g. capture `&[T]`) still
                /// monomorphize cleanly. The caller is responsible for
                /// keeping that borrow alive across the asynchronous
                /// launch — `cuda_host::type_id_u128` does not enforce
                /// this.
                #[doc(hidden)]
                #(#cfg_attrs)*
                #[inline(never)]
                #vis fn #instantiate_name #generics (_: &#closure_type) -> &'static str #where_clause {
                    #ptx_name_fn #kernel_turbofish ()
                }
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    // Generate the GenericCudaKernel trait implementation for unified compilation
    let generic_cuda_kernel_impl =
        generate_generic_cuda_kernel_impl(fn_name, vis, generics, where_clause, &cfg_attrs);
    let mut implementation_args: Vec<TokenStream2> =
        arg_names.iter().map(|name| quote! { #name }).collect();
    if let Some(scope) = &rewritten_scope {
        let scope_ident = &scope.ident;
        if has_explicit_scope {
            implementation_args.push(quote! { #scope_ident });
        } else {
            implementation_args.push(quote! { &#scope_ident });
        }
    }
    let scope_bindings = rewritten_scope
        .as_ref()
        .map(|scope| {
            if has_explicit_scope {
                explicit_kernel_scope_bindings(scope)
            } else {
                vec![kernel_scope_binding(scope)]
            }
        })
        .unwrap_or_default();
    // An opted-in kernel's entry calls the hidden unchecked twin; everyone
    // else (user code included) calls the user-named, bounds-checked helper.
    let (implementation_target, unchecked_impl_item) = match &unchecked_impl {
        Some((clone_name, tokens)) => (quote! { #clone_name }, tokens.clone()),
        None => (quote! { #fn_name }, quote! {}),
    };
    let implementation_call =
        quote! { #implementation_target #kernel_turbofish (#(#implementation_args),*) };
    let implementation_call = if unsafety.is_some() {
        quote! { unsafe { #implementation_call } }
    } else {
        implementation_call
    };

    quote! {
        // Original generic kernel implementation
        #(#implementation_attrs)*
        #helper_dead_code
        #[inline(always)]
        #vis #constness #unsafety #abi fn #fn_name #generics (#inputs) #output #where_clause
        #block

        #unchecked_impl_item

        // Entry point for collector - NOT inlined so we can detect it
        // When called with concrete types, this instantiates the kernel
        // Synthetic wrapper parameter names make every irrefutable source
        // pattern forwardable without carrying local binding `mut`.
        #(#entry_attrs)*
        #[inline(never)]
        #vis #constness #unsafety #abi fn #kernel_name #generics (#(#wrapper_inputs),*) #output #where_clause {
            #(#entry_config_markers)*
            #(#scope_bindings)*
            #implementation_call
        }

        #instantiate_helper

        #generic_cuda_kernel_impl
    }
}

/// Route attributes when one generic source function becomes several items.
///
/// CUDA entry directives must decorate the generated collector entry, not the
/// inline implementation helper. Configuration gates must decorate every item
/// to avoid leaving a marker or helper behind for a disabled kernel. Other
/// user-facing attributes (documentation, lints, deprecation, etc.) stay on
/// the implementation with the original Rust name.
pub(crate) fn route_generic_kernel_attrs(
    attrs: &[syn::Attribute],
) -> (
    Vec<syn::Attribute>,
    Vec<syn::Attribute>,
    Vec<syn::Attribute>,
) {
    let is_cfg = |attr: &syn::Attribute| {
        attr_path_ends_with(attr, "cfg") || attr_path_ends_with(attr, "cfg_attr")
    };
    let is_entry_directive = |attr: &syn::Attribute| {
        attr_path_ends_with(attr, "launch_bounds")
            || attr_path_ends_with(attr, "launch_contract")
            || attr_path_ends_with(attr, "cluster_launch")
            || attr_path_ends_with(attr, "cooperative_launch")
    };

    let cfg_attrs = attrs.iter().filter(|attr| is_cfg(attr)).cloned().collect();
    let implementation_attrs = attrs
        .iter()
        .filter(|attr| !is_entry_directive(attr) && !attr_path_ends_with(attr, "inline"))
        .cloned()
        .collect();
    let entry_attrs = attrs
        .iter()
        .filter(|attr| is_cfg(attr) || is_entry_directive(attr))
        .cloned()
        .collect();

    (implementation_attrs, entry_attrs, cfg_attrs)
}

/// Generate a dummy binding for a given type.
/// Used by instantiate! helper to create zero-valued arguments.
///
/// The generated values are never actually executed - they exist only to force
/// rustc to monomorphize the kernel with the correct types.
fn _generate_dummy_binding(name: &Ident, ty: &Type) -> TokenStream2 {
    match ty {
        // Special case: &[T] or &mut [T] → empty slice literal
        // (slices don't implement Default and can't be safely zeroed)
        Type::Reference(type_ref) if matches!(&*type_ref.elem, Type::Slice(_)) => {
            if let Type::Slice(slice) = &*type_ref.elem {
                let elem_ty = &slice.elem;
                if type_ref.mutability.is_some() {
                    quote! { let #name: &mut [#elem_ty] = &mut []; }
                } else {
                    quote! { let #name: &[#elem_ty] = &[]; }
                }
            } else {
                unreachable!()
            }
        }

        // Everything else: use mem::zeroed()
        // Safe because this code never actually runs - it only exists to
        // force monomorphization of the kernel with the correct types.
        _ => {
            quote! { let #name: #ty = unsafe { core::mem::zeroed() }; }
        }
    }
}

/// Generate a simple non-generic kernel
pub(super) fn generate_simple_kernel(
    mut input: ItemFn,
    explicit_scope: Option<Ident>,
) -> TokenStream {
    if let Some(ident) = explicit_scope {
        let scope = explicit_kernel_scope(&mut input, ident);
        let bindings = explicit_kernel_scope_bindings(&scope);
        input.block.stmts.splice(0..0, bindings);
    } else {
        inject_thread_index_scope(&mut input);
    }

    let fn_name = input.sig.ident.clone();
    let new_name = format_ident!("{}{}", KERNEL_PREFIX, fn_name);

    // Clone the original function for the CudaKernel impl
    let original_fn = input.clone();
    input.sig.ident = new_name;

    // PTX entry name is the unprefixed user name; the collector strips
    // KERNEL_PREFIX when generating PTX.
    let ptx_entry_name = fn_name.to_string();

    // Generate the CudaKernel trait implementation (host-side only)
    // This provides the PTX name for cuda_launch! to look up
    let cuda_kernel_impl = generate_cuda_kernel_impl(
        &fn_name,
        &ptx_entry_name,
        &original_fn,
        cfg!(feature = "host"),
    );

    let expanded = quote! {
        #[unsafe(no_mangle)]
        #input

        #cuda_kernel_impl
    };

    TokenStream::from(expanded)
}

/// Generate the GenericCudaKernel trait implementation for a generic kernel.
///
/// For generic kernels like `fn tile<T, const N: usize>()`, emits:
///
/// ```ignore
/// pub struct __tile_CudaKernel<T, const N: usize>(PhantomData<*const T>);
/// impl<T, const N: usize> GenericCudaKernel for __tile_CudaKernel<T, N> {
///     fn ptx_name() -> &'static str {
///         // "tile_TID_<hex32>" — one 32-char hash of the concrete
///         // generated kernel function-item type.
///     }
/// }
/// pub fn tile_ptx_name<T, const N: usize>() -> &'static str {
///     // Retains `kernel_entry::<T, N>` and delegates to the marker above.
/// }
/// ```
///
/// The body computes the same string the backend writes into the PTX:
/// `<base>_TID_<hex32>`, where `<hex32>` is
/// `cuda_host::type_id_u128_of_val(&kernel_entry::<T, N>)` rendered as 32
/// lowercase hex chars. The backend hashes the same concrete `FnDef` type,
/// whose ordered generic arguments include both types and const values.
///
/// Bound on the impl is `where_clause` verbatim — typically `Copy` on
/// each value-passed generic. We deliberately do not add `'static`:
/// `type_id_u128` has bound `T: ?Sized`, so closure types that borrow
/// non-`'static` data still satisfy the marker's bounds and can be
/// launched through the typed `module.<kernel>(...)` path. Keeping the
/// borrow alive across `stream.synchronize()` remains the caller's
/// responsibility, exactly as it was under the previous `type_name`
/// scheme.
///
/// Deliberately NOT gated by the `host` feature: generic kernels remain
/// host-coupled by design for now, because the TypeId naming machinery
/// lives in `cuda_host`.
fn generate_generic_cuda_kernel_impl(
    fn_name: &Ident,
    vis: &syn::Visibility,
    generics: &syn::Generics,
    where_clause: &Option<syn::WhereClause>,
    cfg_attrs: &[syn::Attribute],
) -> TokenStream2 {
    let marker_name = format_ident!("__{}_CudaKernel", fn_name);
    let ptx_name_fn = format_ident!("{}_ptx_name", fn_name);
    let kernel_name = format_ident!("{}{}", KERNEL_PREFIX, fn_name);
    let base_name = fn_name.to_string();
    let generic_params: Vec<_> = generics.params.iter().collect();
    let marker_args = generic_arguments(generics);
    let codegen_args = codegen_generic_arguments(generics);
    let phantom_type = generic_phantom_type(generics);
    let marker_type = if marker_args.is_empty() {
        quote! { #marker_name }
    } else {
        quote! { #marker_name <#(#marker_args),*> }
    };
    let kernel_turbofish = if codegen_args.is_empty() {
        quote! {}
    } else {
        quote! { ::<#(#codegen_args),*> }
    };
    let (impl_generics, _, _) = generics.split_for_impl();
    let hash = internal_ident("__cuda_oxide_kernel_hash");

    quote! {
        /// Marker type for a generic kernel; implements `GenericCudaKernel`.
        /// Its generic parameters mirror the kernel's generic parameters.
        #(#cfg_attrs)*
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        pub struct #marker_name<#(#generic_params),*>(
            core::marker::PhantomData<#phantom_type>
        ) #where_clause;

        #(#cfg_attrs)*
        impl #impl_generics ::cuda_host::GenericCudaKernel for #marker_type
        #where_clause
        {
            fn ptx_name() -> &'static str {
                let #hash = ::cuda_host::type_id_u128_of_val(
                    &#kernel_name #kernel_turbofish
                );
                ::cuda_host::__intern_generic_kernel_name(#base_name, #hash)
            }
        }

        /// Retains this concrete kernel specialization and returns its PTX entry name.
        ///
        /// The reify cast is what makes rustc collect the monomorphized kernel
        /// item for the device backend; `black_box` keeps the otherwise unused
        /// cast from being optimized away without touching any pointer.
        #(#cfg_attrs)*
        #[inline(never)]
        #vis fn #ptx_name_fn #generics () -> &'static str #where_clause {
            ::core::hint::black_box(#kernel_name #kernel_turbofish as *const ());
            <#marker_type as ::cuda_host::GenericCudaKernel>::ptx_name()
        }
    }
}

/// Generate the CudaKernel trait implementation for a kernel function.
///
/// This generates a marker struct that implements `CudaKernel`, allowing
/// `cuda_launch!` to look up the PTX entry point name at compile time.
///
/// Emitted only under the `host` feature: the impl names `cuda_host`, and a
/// crate that only compiles kernels never looks a PTX entry name up.
pub(crate) fn generate_cuda_kernel_impl(
    fn_name: &Ident,
    ptx_name: &str,
    _func: &ItemFn,
    emit_host: bool,
) -> TokenStream2 {
    if !emit_host {
        return TokenStream2::new();
    }

    // Create a marker struct for this kernel
    // We use a struct because Rust doesn't allow trait impls on function pointers easily
    let marker_name = format_ident!("__{}_CudaKernel", fn_name);

    quote! {
        /// Marker type for the kernel, implements CudaKernel trait.
        /// This enables cuda_launch! to look up the PTX entry point name.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        pub struct #marker_name;

        impl cuda_host::CudaKernel for #marker_name {
            const PTX_NAME: &'static str = #ptx_name;
        }
    }
}

/// Generate wrapper kernels for a generic kernel
pub(super) fn generate_generic_kernel(
    input: ItemFn,
    instantiate_types: Vec<Type>,
    explicit_scope: Option<Ident>,
) -> TokenStream {
    generic_kernel_instantiation_tokens(input, instantiate_types, explicit_scope).into()
}

/// Expansion body of [`generate_generic_kernel`] (legacy `#[kernel(Type, ...)]`
/// instantiation), split out so unit tests can inspect the generated items
/// without a live proc-macro bridge.
pub(crate) fn generic_kernel_instantiation_tokens(
    mut input: ItemFn,
    instantiate_types: Vec<Type>,
    explicit_scope: Option<Ident>,
) -> TokenStream2 {
    let entry_inputs = input.sig.inputs.clone();
    // Same as the no-instantiation path: `requires` relations of a routed
    // `#[launch_contract]` must be validated against the source parameter
    // names before they are lost to the generated wrappers.
    if let Err(error) = validate_routed_launch_contract_requires(&input.attrs, &entry_inputs) {
        return error.to_compile_error();
    }
    let has_explicit_scope = explicit_scope.is_some();
    let rewritten_scope = if let Some(ident) = explicit_scope {
        Some(explicit_kernel_scope(&mut input, ident))
    } else {
        rewrite_thread_index_calls(&mut input, false)
    };
    if let Some(scope) = &rewritten_scope {
        append_kernel_scope_parameter(&mut input, scope);
    }

    let (implementation_attrs, entry_attrs, _cfg_attrs) = route_generic_kernel_attrs(&input.attrs);
    let entry_config_markers = top_level_kernel_configuration_markers(&input);
    strip_grid_constant_config_markers(&mut input);
    input.attrs.clear();
    // Same containment rule as the no-instantiation path: the marker never
    // stays in the re-emitted user-named helper; opted entries call a hidden
    // unchecked twin instead.
    let unchecked_indexing = input
        .block
        .stmts
        .iter()
        .any(is_unchecked_indexing_config_marker);
    if unchecked_indexing {
        strip_unchecked_indexing_config_marker(&mut input);
    }
    let unchecked_impl =
        unchecked_indexing.then(|| unchecked_indexing_impl_clone(&input, &implementation_attrs));
    // The wrappers call the hidden twin, so a private opted kernel's
    // user-named helper may otherwise be dead code; it stays emitted
    // (bounds-checked) for other device code to call.
    let helper_dead_code = if unchecked_indexing {
        quote! { #[allow(dead_code)] }
    } else {
        quote! {}
    };
    let fn_name = &input.sig.ident;
    let vis = &input.vis;
    let generics = &input.sig.generics;
    let (implementation_target, unchecked_impl_item) = match &unchecked_impl {
        Some((clone_name, tokens)) => (quote! { #clone_name }, tokens.clone()),
        None => (quote! { #fn_name }, quote! {}),
    };

    // Extract the type parameter name (assume single type param for now)
    let type_param = generics
        .params
        .iter()
        .find_map(|p| {
            if let GenericParam::Type(tp) = p {
                Some(&tp.ident)
            } else {
                None
            }
        })
        .expect("Expected type parameter");

    // Extract function arguments (excluding self)
    let args: Vec<_> = entry_inputs.iter().collect();

    // Build the argument pattern and types for wrappers
    let arg_names: Vec<TokenStream2> = args
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg
                && let Pat::Ident(pat_ident) = &*pat_type.pat
            {
                return Some(quote! { #pat_ident });
            }
            None
        })
        .collect();

    // For each instantiation type, generate a wrapper that substitutes the type
    let wrappers: Vec<TokenStream2> = instantiate_types
        .iter()
        .map(|inst_type| {
            let entry_attrs = &entry_attrs;
            let entry_config_markers = &entry_config_markers;
            let implementation_target = &implementation_target;
            // Get a clean name for the type (for the kernel name suffix)
            let type_name = get_type_name(inst_type);
            let wrapper_name = format_ident!("{}{}_{}", KERNEL_PREFIX, fn_name, type_name);

            // Export name (what appears in PTX)
            let export_name_str = format!("{}_{}", fn_name, type_name);
            let scope_bindings = rewritten_scope
                .as_ref()
                .map(|scope| {
                    if has_explicit_scope {
                        explicit_kernel_scope_bindings(scope)
                    } else {
                        vec![kernel_scope_binding(scope)]
                    }
                })
                .unwrap_or_default();
            let mut implementation_args = arg_names.clone();
            if let Some(scope) = &rewritten_scope {
                let scope_ident = &scope.ident;
                if has_explicit_scope {
                    implementation_args.push(quote! { #scope_ident });
                } else {
                    implementation_args.push(quote! { &#scope_ident });
                }
            }

            // Generate wrapper function args with substituted types
            let wrapper_args: Vec<TokenStream2> = args
                .iter()
                .map(|arg| {
                    if let FnArg::Typed(pat_type) = arg {
                        let pat = &pat_type.pat;
                        let ty = &pat_type.ty;
                        // Substitute type parameter with concrete type
                        let subst_ty = substitute_type(ty, type_param, inst_type);
                        quote! { #pat: #subst_ty }
                    } else {
                        quote! { #arg }
                    }
                })
                .collect();

            quote! {
                #(#entry_attrs)*
                #[unsafe(no_mangle)]
                #[unsafe(export_name = #export_name_str)]
                #vis fn #wrapper_name(#(#wrapper_args),*) {
                    #(#entry_config_markers)*
                    #(#scope_bindings)*
                    #implementation_target::<#inst_type>(#(#implementation_args),*);
                }
            }
        })
        .collect();

    // Keep the original generic function (without #[no_mangle] - it's not an entry point)
    // and add all the wrapper kernels
    quote! {
        #(#implementation_attrs)*
        #helper_dead_code
        #[inline(always)]
        #input

        #unchecked_impl_item

        #(#wrappers)*
    }
}

/// Get a clean name from a type for use in function names
fn get_type_name(ty: &Type) -> String {
    match ty {
        Type::Path(type_path) => {
            // Get the last segment (e.g., "Scale" from "crate::Scale")
            type_path
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_else(|| "Unknown".to_string())
        }
        _ => "Unknown".to_string(),
    }
}

/// Substitute a type parameter with a concrete type in a type expression
fn substitute_type(ty: &Type, param: &syn::Ident, replacement: &Type) -> TokenStream2 {
    match ty {
        Type::Path(type_path) => {
            // Check if this is just the type parameter
            if type_path.path.is_ident(param) {
                return quote! { #replacement };
            }
            quote! { #ty }
        }
        Type::Reference(type_ref) => {
            let elem = substitute_type(&type_ref.elem, param, replacement);
            let lifetime = &type_ref.lifetime;
            let mutability = &type_ref.mutability;
            quote! { &#lifetime #mutability #elem }
        }
        _ => quote! { #ty },
    }
}
