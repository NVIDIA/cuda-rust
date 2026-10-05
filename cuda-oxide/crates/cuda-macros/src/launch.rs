/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `cuda_launch!` and `cuda_launch_async!` input parsing and expansion.

use crate::common::internal_ident;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use reserved_oxide_symbols::INSTANTIATE_PREFIX;
use syn::{
    Ident, Token, bracketed, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

// ============================================================================
// cuda_launch! Macro (unified compilation)
// ============================================================================

/// Try to extract closure from an expression.
///
/// Closure marshalling no longer needs per-capture extraction: the
/// backend emits a single byval `.param` for the whole closure struct,
/// and the host pushes one scalar. The closure literal is still parsed
/// out of the launch args so the `instantiate_name` helper has a
/// concrete `&F` to bind the kernel's generic closure type to.
fn as_closure_expr(expr: &syn::Expr) -> Option<&syn::ExprClosure> {
    match expr {
        syn::Expr::Closure(closure) => Some(closure),
        syn::Expr::Group(group) => as_closure_expr(&group.expr),
        syn::Expr::Paren(paren) => as_closure_expr(&paren.expr),
        _ => None,
    }
}

/// Argument type for cuda_launch! - same as LaunchArg but renamed for clarity
pub(crate) enum CudaLaunchArg {
    /// Direct expression - passed via .arg()
    Direct(syn::Expr),
    /// Slice with explicit length - passed as ptr + len
    SliceWithLen(syn::Expr),
    /// Mutable slice with explicit length - passed as ptr + len
    SliceMutWithLen(syn::Expr),
    /// Closure expression. The closure value is pushed as a single byval
    /// scalar argument; the backend emits a matching single .param entry
    /// for aggregate kernel parameters. No per-capture decomposition.
    Closure { closure_expr: syn::ExprClosure },
}

impl Parse for CudaLaunchArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Check for tagged arguments
        if input.peek(Ident) {
            let ident: Ident = input.fork().parse()?;
            match ident.to_string().as_str() {
                "slice" => {
                    input.parse::<Ident>()?;
                    let content;
                    parenthesized!(content in input);
                    let expr: syn::Expr = content.parse()?;
                    return Ok(CudaLaunchArg::SliceWithLen(expr));
                }
                "slice_mut" => {
                    input.parse::<Ident>()?;
                    let content;
                    parenthesized!(content in input);
                    let expr: syn::Expr = content.parse()?;
                    return Ok(CudaLaunchArg::SliceMutWithLen(expr));
                }
                // "move" keyword starts a move closure
                "move" => {
                    // Parse the full closure expression (move |args| body)
                    let expr: syn::Expr = input.parse()?;
                    if let Some(closure) = as_closure_expr(&expr) {
                        return Ok(CudaLaunchArg::Closure {
                            closure_expr: closure.clone(),
                        });
                    }
                    // Not a closure, treat as direct expression
                    return Ok(CudaLaunchArg::Direct(expr));
                }
                _ => {}
            }
        }

        // Check for closure starting with `|` (non-move closure)
        if input.peek(Token![|]) {
            let expr: syn::Expr = input.parse()?;
            if let Some(closure) = as_closure_expr(&expr) {
                return Ok(CudaLaunchArg::Closure {
                    closure_expr: closure.clone(),
                });
            }
            // Shouldn't happen, but fallback to direct
            return Ok(CudaLaunchArg::Direct(expr));
        }

        // Default: direct expression
        let expr: syn::Expr = input.parse()?;

        // Check if the parsed expression happens to be a closure
        if let Some(closure) = as_closure_expr(&expr) {
            return Ok(CudaLaunchArg::Closure {
                closure_expr: closure.clone(),
            });
        }

        Ok(CudaLaunchArg::Direct(expr))
    }
}

/// Input for cuda_launch! macro
pub(crate) struct CudaLaunchInput {
    /// Kernel path - can be simple name or path with generics: `scale` or `scale::<f32>`
    pub(crate) kernel: syn::Path,
    pub(crate) stream: syn::Expr,
    pub(crate) module: syn::Expr,
    pub(crate) config: syn::Expr,
    pub(crate) args: Vec<CudaLaunchArg>,
    /// Optional cluster dimensions (x, y, z) for thread block cluster launches.
    /// When present, uses `cuLaunchKernelEx` via `launch_cluster()` instead of `cuLaunchKernel`.
    pub(crate) cluster_dim: Option<syn::Expr>,
    /// Optional cooperative-launch flag. When `true`, the kernel is launched
    /// via `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 1`,
    /// which is required for `cuda_device::grid::sync()` to work.
    pub(crate) cooperative: Option<syn::Expr>,
}

/// Return the generated sibling of a kernel while preserving its module path
/// and explicit generic arguments.
///
/// For example, `kernels::map::<F, 4>` becomes
/// `kernels::__map_CudaKernel::<F, 4>` when `sibling_name` is the marker name.
pub(crate) fn kernel_sibling_path(kernel: &syn::Path, sibling_name: Ident) -> syn::Path {
    let mut sibling = kernel.clone();
    sibling
        .segments
        .last_mut()
        .expect("kernel path must have segments")
        .ident = sibling_name;
    sibling
}

impl CudaLaunchInput {
    /// Extract the base kernel name (without generics) and generic arguments
    fn kernel_parts(&self) -> (Ident, Option<&syn::PathArguments>) {
        let last_segment = self
            .kernel
            .segments
            .last()
            .expect("kernel path must have segments");
        let base_name = last_segment.ident.clone();
        let generics = match &last_segment.arguments {
            syn::PathArguments::None => None,
            args => Some(args),
        };
        (base_name, generics)
    }

    /// Check if this is a generic kernel (has type parameters)
    fn is_generic(&self) -> bool {
        self.kernel
            .segments
            .last()
            .map(|seg| !matches!(seg.arguments, syn::PathArguments::None))
            .unwrap_or(false)
    }
}

impl Parse for CudaLaunchInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut kernel = None;
        let mut stream = None;
        let mut module = None;
        let mut config = None;
        let mut args = Vec::new();
        let mut cluster_dim = None;
        let mut cooperative = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "kernel" => kernel = Some(input.parse()?),
                "stream" => stream = Some(input.parse()?),
                "module" => module = Some(input.parse()?),
                "config" => config = Some(input.parse()?),
                "cluster_dim" => cluster_dim = Some(input.parse()?),
                "cooperative" => cooperative = Some(input.parse()?),
                "args" => {
                    let content;
                    bracketed!(content in input);
                    if !content.is_empty() {
                        let parsed: Punctuated<CudaLaunchArg, Token![,]> =
                            Punctuated::parse_terminated(&content)?;
                        args = parsed.into_iter().collect();
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown field: {}. Expected: kernel, stream, module, config, cluster_dim, cooperative, args",
                            key
                        ),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        Ok(CudaLaunchInput {
            kernel: kernel.ok_or_else(|| syn::Error::new(input.span(), "missing 'kernel'"))?,
            stream: stream.ok_or_else(|| syn::Error::new(input.span(), "missing 'stream'"))?,
            module: module.ok_or_else(|| syn::Error::new(input.span(), "missing 'module'"))?,
            config: config.ok_or_else(|| syn::Error::new(input.span(), "missing 'config'"))?,
            args,
            cluster_dim,
            cooperative,
        })
    }
}

pub(crate) fn expand_cuda_launch(input: CudaLaunchInput) -> TokenStream2 {
    let stream = &input.stream;
    let module = &input.module;
    let config = &input.config;
    let cluster_dim = &input.cluster_dim;
    let cooperative = &input.cooperative;

    // Get base kernel name and generic arguments
    let (kernel_base, _generics) = input.kernel_parts();

    // Build the marker type name for CudaKernel lookup
    let marker = kernel_sibling_path(&input.kernel, format_ident!("__{}_CudaKernel", kernel_base));
    let ptx_name_helper =
        kernel_sibling_path(&input.kernel, format_ident!("{}_ptx_name", kernel_base));
    let args_ident = internal_ident("__cuda_oxide_args");
    let closure_ident = internal_ident("__cuda_oxide_closure");
    let ptx_name_ident = internal_ident("__cuda_oxide_ptx_name");
    let module_ident = internal_ident("__cuda_oxide_module");
    let function_ident = internal_ident("__cuda_oxide_function");
    let config_ident = internal_ident("__cuda_oxide_config");
    let cooperative_ident = internal_ident("__cuda_oxide_cooperative");
    let error_ident = internal_ident("__cuda_oxide_error");
    let static_ptx_name_ident = internal_ident("__CUDA_OXIDE_PTX_NAME");

    // Check if any argument is a closure (for special handling)
    let has_closure = input
        .args
        .iter()
        .any(|arg| matches!(arg, CudaLaunchArg::Closure { .. }));

    // Extract closure info if present (for monomorphization). Only the
    // first closure is treated as the type-inference anchor; the macro
    // currently supports at most one closure parameter per kernel.
    let closure_info: Option<&syn::ExprClosure> = input.args.iter().find_map(|arg| {
        if let CudaLaunchArg::Closure { closure_expr } = arg {
            Some(closure_expr)
        } else {
            None
        }
    });

    // Generate argument marshaling code.
    //
    // Each argument becomes a stack-local variable whose address is pushed
    // into a `Vec<*mut c_void>`. This directly matches what cuLaunchKernel
    // expects: an array of pointers-to-argument-values. No trait dispatch
    // (PushKernelArg) or heap allocation per arg.
    let arg_code: Vec<TokenStream2> = input
        .args
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let val_name = internal_ident(&format!("__cuda_oxide_arg_{i}"));
            match arg {
                CudaLaunchArg::Direct(expr) => {
                    quote! {
                        let mut #val_name = #expr;
                        #args_ident.push(&mut #val_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::SliceWithLen(expr) => {
                    let ptr_name = internal_ident(&format!("__cuda_oxide_arg_{i}_ptr"));
                    let len_name = internal_ident(&format!("__cuda_oxide_arg_{i}_len"));
                    quote! {
                        let #val_name = &#expr;
                        let mut #ptr_name = #val_name.cu_deviceptr();
                        let mut #len_name = #val_name.len() as u64;
                        #args_ident.push(&mut #ptr_name as *mut _ as *mut std::ffi::c_void);
                        #args_ident.push(&mut #len_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::SliceMutWithLen(expr) => {
                    let ptr_name = internal_ident(&format!("__cuda_oxide_arg_{i}_ptr"));
                    let len_name = internal_ident(&format!("__cuda_oxide_arg_{i}_len"));
                    quote! {
                        let #val_name = &mut #expr;
                        let mut #ptr_name = #val_name.cu_deviceptr();
                        let mut #len_name = #val_name.len() as u64;
                        #args_ident.push(&mut #ptr_name as *mut _ as *mut std::ffi::c_void);
                        #args_ident.push(&mut #len_name as *mut _ as *mut std::ffi::c_void);
                    }
                }
                CudaLaunchArg::Closure { .. } => {
                    // Push the whole closure as a single byval scalar. The
                    // backend emits a single byval kernel parameter for
                    // aggregate (struct / closure) entry-point args, so
                    // pushing `__closure` once matches what the device-side
                    // `.param` declaration expects.
                    //
                    // Routed through `push_kernel_scalar` so ZST closures
                    // (zero captures) are dropped from the host packet —
                    // matching the backend, which drops their `.param`
                    // declaration too. Move closures push by value;
                    // non-move closures push the closure struct (which
                    // contains host references the GPU dereferences via
                    // HMM).
                    let _ = i;
                    quote! {
                        ::cuda_host::push_kernel_scalar(&mut #args_ident, &mut #closure_ident);
                    }
                }
            }
        })
        .collect();

    // Build the instantiate helper name (for closures)
    let instantiate = kernel_sibling_path(
        &input.kernel,
        format_ident!("{}{}", INSTANTIATE_PREFIX, kernel_base),
    );

    // Generate the launch call — regular, cluster, cooperative, or both.
    //
    // All paths use the stream-aware cuda_core helpers. Those helpers bind the
    // stream's owning CUDA context to the calling thread and then delegate to
    // the raw cuLaunchKernel/cuLaunchKernelEx wrappers.
    let launch_call = match (&cluster_dim, &cooperative) {
        (Some(cdim), Some(coop)) => quote! {
            {
                let #config_ident = #config;
                let #cooperative_ident: bool = #coop;
                if #cooperative_ident {
                    cuda_core::launch_kernel_ex_cooperative_on_stream(
                        &#function_ident,
                        #config_ident.grid_dim,
                        #config_ident.block_dim,
                        #config_ident.shared_mem_bytes,
                        #cdim,
                        (#stream).as_ref(),
                        &mut #args_ident,
                    )
                } else {
                    cuda_core::launch_kernel_ex_on_stream(
                        &#function_ident,
                        #config_ident.grid_dim,
                        #config_ident.block_dim,
                        #config_ident.shared_mem_bytes,
                        #cdim,
                        (#stream).as_ref(),
                        &mut #args_ident,
                    )
                }
            }
        },
        (Some(cdim), None) => quote! {
            {
                let #config_ident = #config;
                cuda_core::launch_kernel_ex_on_stream(
                    &#function_ident,
                    #config_ident.grid_dim,
                    #config_ident.block_dim,
                    #config_ident.shared_mem_bytes,
                    #cdim,
                    (#stream).as_ref(),
                    &mut #args_ident,
                )
            }
        },
        (None, Some(coop)) => quote! {
            {
                let #config_ident = #config;
                let #cooperative_ident: bool = #coop;
                if #cooperative_ident {
                    cuda_core::launch_kernel_cooperative_on_stream(
                        &#function_ident,
                        #config_ident.grid_dim,
                        #config_ident.block_dim,
                        #config_ident.shared_mem_bytes,
                        (#stream).as_ref(),
                        &mut #args_ident,
                    )
                } else {
                    cuda_core::launch_kernel_on_stream(
                        &#function_ident,
                        #config_ident.grid_dim,
                        #config_ident.block_dim,
                        #config_ident.shared_mem_bytes,
                        (#stream).as_ref(),
                        &mut #args_ident,
                    )
                }
            }
        },
        (None, None) => quote! {
            {
                let #config_ident = #config;
                cuda_core::launch_kernel_on_stream(
                    &#function_ident,
                    #config_ident.grid_dim,
                    #config_ident.block_dim,
                    #config_ident.shared_mem_bytes,
                    (#stream).as_ref(),
                    &mut #args_ident,
                )
            }
        },
    };

    if has_closure {
        let closure_expr = closure_info.expect("has_closure but no closure_info");

        // The on-wire PTX name comes from the kernel's
        // GenericCudaKernel::ptx_name() impl (via the instantiate helper).
        // The helper takes `&F` so we can keep ownership of `__closure`
        // and push it as the byval kernel argument right after — the
        // backend's kernel-boundary ABI emits a single .param for the
        // whole closure struct, matching this single push.
        let _ = closure_expr.span();

        quote! {
            {
                let mut #closure_ident = #closure_expr;
                let #ptx_name_ident: &'static str = #instantiate(&#closure_ident);
                let #module_ident = &#module;
                // On a miss, the helper panics with the host/device
                // type-identity divergence diagnosis when the module holds
                // this kernel under a different `_TID_` hash; an ordinary
                // miss keeps this macro's long-standing panic message.
                let #function_ident = #module_ident.load_function(#ptx_name_ident).unwrap_or_else(|#error_ident| {
                    ::cuda_host::panic_generic_kernel_load_failed(
                        #module_ident,
                        stringify!(#kernel_base),
                        #ptx_name_ident,
                        #error_ident,
                    )
                });

                let mut #args_ident: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                #launch_call
            }
        }
    } else if input.is_generic() {
        quote! {
            {
                let #ptx_name_ident = #ptx_name_helper();
                let #module_ident = &#module;
                // Same divergence-aware failure path as the closure branch.
                let #function_ident = #module_ident.load_function(#ptx_name_ident).unwrap_or_else(|#error_ident| {
                    ::cuda_host::panic_generic_kernel_load_failed(
                        #module_ident,
                        stringify!(#kernel_base),
                        #ptx_name_ident,
                        #error_ident,
                    )
                });

                let mut #args_ident: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                #launch_call
            }
        }
    } else {
        quote! {
            {
                const #static_ptx_name_ident: &str = <#marker as cuda_host::CudaKernel>::PTX_NAME;
                let #function_ident = #module.load_function(#static_ptx_name_ident).unwrap_or_else(|#error_ident| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        #static_ptx_name_ident,
                        #error_ident,
                    )
                });

                let mut #args_ident: Vec<*mut std::ffi::c_void> = Vec::new();
                #(#arg_code)*

                #launch_call
            }
        }
    }
}

// ============================================================================
// cuda_launch_async! Macro (async path via cuda-async)
// ============================================================================

/// Parsed input for the [`cuda_launch_async!`](crate::cuda_launch_async) macro.
///
/// Unlike [`CudaLaunchInput`], this struct has no `stream` field. The stream
/// is assigned later by the `SchedulingPolicy` when the returned
/// `AsyncKernelLaunch` is `.sync()`'d or `.await`'d.
pub(crate) struct CudaLaunchAsyncInput {
    /// Path to the `#[kernel]` function, possibly with generic arguments.
    kernel: syn::Path,
    /// Expression resolving to an `Arc<CudaModule>` that contains the compiled PTX.
    module: syn::Expr,
    /// Expression resolving to a [`LaunchConfig`](https://docs.rs/cuda-core/latest/cuda_core/simt/launch/struct.LaunchConfig.html) (grid/block dims, shared mem).
    config: syn::Expr,
    /// Kernel arguments: `slice(x)`, `slice_mut(x)`, direct values, or closures.
    args: Vec<CudaLaunchArg>,
}

impl CudaLaunchAsyncInput {
    /// Splits the kernel path into its base identifier and optional generic arguments.
    /// For `vecadd::<f32>` returns `("vecadd", Some(<f32>))`.
    fn kernel_parts(&self) -> (Ident, Option<&syn::PathArguments>) {
        let last_segment = self
            .kernel
            .segments
            .last()
            .expect("kernel path must have segments");
        let base_name = last_segment.ident.clone();
        let generics = match &last_segment.arguments {
            syn::PathArguments::None => None,
            args => Some(args),
        };
        (base_name, generics)
    }

    /// Returns `true` if the kernel path has explicit generic type arguments.
    fn is_generic(&self) -> bool {
        self.kernel
            .segments
            .last()
            .map(|seg| !matches!(seg.arguments, syn::PathArguments::None))
            .unwrap_or(false)
    }
}

/// Parses the `cuda_launch_async! { kernel: ..., module: ..., config: ..., args: [...] }` syntax.
///
/// Fields can appear in any order. The `args` field uses bracket syntax with the same
/// argument forms as `cuda_launch!`: `slice(x)`, `slice_mut(x)`, direct values, or closures.
impl Parse for CudaLaunchAsyncInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut kernel = None;
        let mut module = None;
        let mut config = None;
        let mut args = Vec::new();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match key.to_string().as_str() {
                "kernel" => kernel = Some(input.parse()?),
                "module" => module = Some(input.parse()?),
                "config" => config = Some(input.parse()?),
                "args" => {
                    let content;
                    bracketed!(content in input);
                    if !content.is_empty() {
                        let parsed: Punctuated<CudaLaunchArg, Token![,]> =
                            Punctuated::parse_terminated(&content)?;
                        args = parsed.into_iter().collect();
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown field: {}. Expected: kernel, module, config, args",
                            key
                        ),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        Ok(CudaLaunchAsyncInput {
            kernel: kernel.ok_or_else(|| syn::Error::new(input.span(), "missing 'kernel'"))?,
            module: module.ok_or_else(|| syn::Error::new(input.span(), "missing 'module'"))?,
            config: config.ok_or_else(|| syn::Error::new(input.span(), "missing 'config'"))?,
            args,
        })
    }
}

pub(crate) fn expand_cuda_launch_async(input: CudaLaunchAsyncInput) -> TokenStream2 {
    let module = &input.module;
    let config = &input.config;
    let (kernel_base, _generics) = input.kernel_parts();
    let marker = kernel_sibling_path(&input.kernel, format_ident!("__{}_CudaKernel", kernel_base));
    let ptx_name_helper =
        kernel_sibling_path(&input.kernel, format_ident!("{}_ptx_name", kernel_base));
    let instantiate = kernel_sibling_path(
        &input.kernel,
        format_ident!("{}{}", INSTANTIATE_PREFIX, kernel_base),
    );
    let has_closure = input
        .args
        .iter()
        .any(|arg| matches!(arg, CudaLaunchArg::Closure { .. }));
    let closure_expr = input.args.iter().find_map(|arg| {
        if let CudaLaunchArg::Closure { closure_expr } = arg {
            Some(closure_expr)
        } else {
            None
        }
    });
    let closure_ident = internal_ident("__cuda_oxide_closure");
    let ptx_name_ident = internal_ident("__cuda_oxide_ptx_name");
    let module_ident = internal_ident("__cuda_oxide_module");
    let function_ident = internal_ident("__cuda_oxide_function");
    let launch_ident = internal_ident("__cuda_oxide_launch");
    let error_ident = internal_ident("__cuda_oxide_error");
    let static_ptx_name_ident = internal_ident("__CUDA_OXIDE_PTX_NAME");

    let arg_code: Vec<TokenStream2> = input
        .args
        .iter()
        .enumerate()
        .map(|(i, arg)| {
            let tmp_name = internal_ident(&format!("__cuda_oxide_arg_{i}"));
            match arg {
                CudaLaunchArg::Direct(expr) => {
                    quote! {
                        #launch_ident.push_scalar_arg(#expr);
                    }
                }
                CudaLaunchArg::SliceWithLen(expr) => {
                    let len_name = internal_ident(&format!("__cuda_oxide_arg_{i}_len"));
                    quote! {
                        let #tmp_name = &#expr;
                        #launch_ident.push_arg(Box::new(#tmp_name.cu_deviceptr()));
                        let #len_name = #tmp_name.len() as u64;
                        #launch_ident.push_arg(Box::new(#len_name));
                    }
                }
                CudaLaunchArg::SliceMutWithLen(expr) => {
                    let len_name = internal_ident(&format!("__cuda_oxide_arg_{i}_len"));
                    quote! {
                        let #tmp_name = &mut #expr;
                        #launch_ident.push_arg(Box::new(#tmp_name.cu_deviceptr()));
                        let #len_name = #tmp_name.len() as u64;
                        #launch_ident.push_arg(Box::new(#len_name));
                    }
                }
                CudaLaunchArg::Closure { .. } => {
                    // Push the whole closure as one byval scalar so the
                    // host packet matches the single aggregate `.param`
                    // at the kernel boundary. ZST closures are omitted
                    // to keep later packet slots aligned.
                    quote! {
                        if ::core::mem::size_of_val(&#closure_ident) != 0 {
                            #launch_ident.push_scalar_arg(#closure_ident);
                        }
                    }
                }
            }
        })
        .collect();

    if has_closure {
        let closure_expr = closure_expr.expect("has_closure but no closure expression");
        quote! {
            {
                let #closure_ident = #closure_expr;
                let #ptx_name_ident: &'static str = #instantiate(&#closure_ident);
                let #module_ident = &#module;
                // On a miss, the helper panics with the host/device
                // type-identity divergence diagnosis when the module holds
                // this kernel under a different `_TID_` hash; an ordinary
                // miss keeps this macro's long-standing panic message.
                let #function_ident = #module_ident.load_function(#ptx_name_ident).unwrap_or_else(|#error_ident| {
                    ::cuda_host::panic_generic_kernel_load_failed(
                        #module_ident,
                        stringify!(#kernel_base),
                        #ptx_name_ident,
                        #error_ident,
                    )
                });
                let mut #launch_ident = cuda_async::simt::launch::AsyncKernelLaunchBuilder::new(
                    std::sync::Arc::new(#function_ident),
                );
                #(#arg_code)*
                #launch_ident.finalize_unchecked(#config)
            }
        }
    } else if input.is_generic() {
        quote! {
            {
                let #ptx_name_ident = #ptx_name_helper();
                let #module_ident = &#module;
                // Same divergence-aware failure path as the closure branch.
                let #function_ident = #module_ident.load_function(#ptx_name_ident).unwrap_or_else(|#error_ident| {
                    ::cuda_host::panic_generic_kernel_load_failed(
                        #module_ident,
                        stringify!(#kernel_base),
                        #ptx_name_ident,
                        #error_ident,
                    )
                });
                let mut #launch_ident = cuda_async::simt::launch::AsyncKernelLaunchBuilder::new(
                    std::sync::Arc::new(#function_ident),
                );
                #(#arg_code)*
                #launch_ident.finalize_unchecked(#config)
            }
        }
    } else {
        quote! {
            {
                const #static_ptx_name_ident: &str =
                    <#marker as cuda_host::CudaKernel>::PTX_NAME;
                let #function_ident = #module.load_function(#static_ptx_name_ident).unwrap_or_else(|#error_ident| {
                    panic!(
                        "Failed to load kernel `{}` (expected PTX entry `{}`): {:?}",
                        stringify!(#kernel_base),
                        #static_ptx_name_ident,
                        #error_ident,
                    )
                });
                let mut #launch_ident = cuda_async::simt::launch::AsyncKernelLaunchBuilder::new(
                    std::sync::Arc::new(#function_ident),
                );
                #(#arg_code)*
                #launch_ident.finalize_unchecked(#config)
            }
        }
    }
}
