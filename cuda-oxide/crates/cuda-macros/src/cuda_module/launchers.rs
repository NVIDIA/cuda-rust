/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generated host-side launch method families for `#[cuda_module]`
//! kernels: sync, prepared, async, and owned-async.

use crate::common::{cuda_module_async_lifetime, internal_ident};
use crate::cuda_module::contract::{
    DynamicSharedContract, RequiresLenAccess, generate_requires_checks,
};
use crate::cuda_module::model::{
    CudaModuleKernel, CudaModuleParam, CudaModuleParamMarshal, cuda_module_kernel_marker_type,
    cuda_module_uniform_scalar,
};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{GenericParam, Ident};

pub(super) fn generate_cuda_module_kernel_signature(
    kernel: &CudaModuleKernel,
) -> Option<TokenStream2> {
    if kernel.is_generic {
        return None;
    }

    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let signature_name = cuda_kernel_signature_name(&kernel.fn_name);
    let marker = cuda_module_kernel_marker_type(kernel);
    let arguments = kernel.params.iter().map(|param| {
        let name = param.name.to_string();
        let kind = cuda_module_argument_kind(param, &kernel.generics);
        quote! {
            ::cuda_host::CudaKernelArgument {
                name: #name,
                kind: #kind,
            }
        }
    });

    Some(quote! {
        #(#cfg_attrs)*
        #[allow(
            dead_code,
            non_upper_case_globals,
            reason = "generated signatures preserve kernel names and may be consumed externally"
        )]
        #vis const #signature_name: ::cuda_host::CudaKernelSignature =
            ::cuda_host::CudaKernelSignature {
                entry: <#marker as ::cuda_host::CudaKernel>::PTX_NAME,
                arguments: &[#(#arguments),*],
            };
    })
}

fn cuda_module_argument_kind(param: &CudaModuleParam, generics: &syn::Generics) -> TokenStream2 {
    if let Some(ty) = &param.grid_constant_ty {
        let ty = signature_layout_type(ty, generics);
        return quote! {
            ::cuda_host::CudaKernelArgumentKind::GridConstant {
                size: ::core::mem::size_of::<#ty>(),
                alignment: ::core::mem::align_of::<#ty>(),
            }
        };
    }

    match &param.marshal {
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => quote! {
            ::cuda_host::CudaKernelArgumentKind::DeviceSlice {
                writable: false,
                row_width: false,
            }
        },
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => quote! {
            ::cuda_host::CudaKernelArgumentKind::DeviceSlice {
                writable: true,
                row_width: false,
            }
        },
        CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => quote! {
            ::cuda_host::CudaKernelArgumentKind::DeviceSlice {
                writable: true,
                row_width: true,
            }
        },
        CudaModuleParamMarshal::Scalar => {
            if matches!(param.device_ty, syn::Type::Ptr(_)) {
                quote! { ::cuda_host::CudaKernelArgumentKind::DevicePointer }
            } else {
                let scalar =
                    cuda_module_scalar_kind(&signature_layout_type(&param.device_ty, generics));
                quote! { ::cuda_host::CudaKernelArgumentKind::Scalar(#scalar) }
            }
        }
    }
}

// Function lifetimes do not change layout and are out of scope in module constants.
fn signature_layout_type(ty: &syn::Type, generics: &syn::Generics) -> syn::Type {
    struct EraseLifetimes<'a>(&'a syn::Generics);
    impl syn::visit_mut::VisitMut for EraseLifetimes<'_> {
        fn visit_lifetime_mut(&mut self, lifetime: &mut syn::Lifetime) {
            if self
                .0
                .lifetimes()
                .any(|param| param.lifetime.ident == lifetime.ident)
            {
                *lifetime = syn::Lifetime::new("'static", lifetime.apostrophe);
            }
        }
    }
    let mut ty = ty.clone();
    syn::visit_mut::VisitMut::visit_type_mut(&mut EraseLifetimes(generics), &mut ty);
    ty
}

fn cuda_module_scalar_kind(ty: &syn::Type) -> TokenStream2 {
    let scalar_ty = cuda_module_uniform_scalar(ty).unwrap_or(ty);
    if let syn::Type::Path(type_path) = scalar_ty
        && type_path.qself.is_none()
        && type_path.path.leading_colon.is_some()
        && type_path.path.segments.len() == 3
        && matches!(
            type_path.path.segments[0].ident.to_string().as_str(),
            "core" | "std"
        )
        && type_path.path.segments[1].ident == "primitive"
        && type_path
            .path
            .segments
            .iter()
            .all(|segment| segment.arguments.is_empty())
    {
        let kind = match type_path.path.segments[2].ident.to_string().as_str() {
            "bool" => Some(quote! { ::cuda_host::CudaKernelScalarKind::Bool }),
            "u8" => Some(quote! { ::cuda_host::CudaKernelScalarKind::U8 }),
            "u16" => Some(quote! { ::cuda_host::CudaKernelScalarKind::U16 }),
            "u32" => Some(quote! { ::cuda_host::CudaKernelScalarKind::U32 }),
            "u64" => Some(quote! { ::cuda_host::CudaKernelScalarKind::U64 }),
            "u128" => Some(quote! { ::cuda_host::CudaKernelScalarKind::U128 }),
            "usize" => Some(quote! { ::cuda_host::CudaKernelScalarKind::Usize }),
            "i8" => Some(quote! { ::cuda_host::CudaKernelScalarKind::I8 }),
            "i16" => Some(quote! { ::cuda_host::CudaKernelScalarKind::I16 }),
            "i32" => Some(quote! { ::cuda_host::CudaKernelScalarKind::I32 }),
            "i64" => Some(quote! { ::cuda_host::CudaKernelScalarKind::I64 }),
            "i128" => Some(quote! { ::cuda_host::CudaKernelScalarKind::I128 }),
            "isize" => Some(quote! { ::cuda_host::CudaKernelScalarKind::Isize }),
            "f32" => Some(quote! { ::cuda_host::CudaKernelScalarKind::F32 }),
            "f64" => Some(quote! { ::cuda_host::CudaKernelScalarKind::F64 }),
            _ => None,
        };
        if let Some(kind) = kind {
            return kind;
        }
    }

    quote! {
        ::cuda_host::CudaKernelScalarKind::Opaque {
            size: ::core::mem::size_of::<#scalar_ty>(),
            alignment: ::core::mem::align_of::<#scalar_ty>(),
        }
    }
}

/// Attach the same nested-allocation obligations to every host launch family.
/// Raw launchers already have a Safety section; prepared grid launchers gain
/// one because their additional unsafe contract is independent of geometry.
fn grid_constant_launch_safety_docs(
    kernel: &CudaModuleKernel,
    include_heading: bool,
) -> Option<TokenStream2> {
    if !kernel
        .params
        .iter()
        .any(|parameter| parameter.grid_constant)
    {
        return None;
    }
    let heading = include_heading.then(|| quote! { #[doc = "# Safety"] });
    Some(quote! {
        #[doc = ""]
        #heading
        #[doc = "Grid-constant values are copied into launch storage, but allocations referenced by their fields are neither copied nor kept alive by that copy. Any references carried in the value must be valid on the device. All memory reached through the value must remain accessible and satisfy Rust aliasing and cross-thread synchronization requirements throughout device execution, including any deferred execution of an async operation. The caller must prevent incompatible host/device accesses and keep referenced allocations alive until the device work completes. `Copy`, storage immutability, and prepared launch checks do not prove these obligations. The caller must also uphold any safety requirements documented on the source kernel."]
    })
}

pub(super) fn generate_cuda_module_launch_contract_impl(
    kernel: &CudaModuleKernel,
) -> Option<TokenStream2> {
    let contract = kernel.launch_contract.as_ref()?;
    let cfg_attrs = &kernel.cfg_attrs;
    let marker_ty = cuda_module_kernel_marker_type(kernel);
    // `#[kernel]` erases lifetime-only generic lists because lifetimes do not
    // create codegen instances. Mirror the marker that `#[kernel]` actually
    // emits instead of applying erased lifetimes to a non-generic marker.
    let generics = if kernel.is_generic {
        kernel.generics.clone()
    } else {
        syn::Generics::default()
    };
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let config_ty = match contract.domain {
        1 => quote! { ::cuda_core::LaunchConfig1D },
        2 => quote! { ::cuda_core::LaunchConfig2D },
        3 => quote! { ::cuda_core::LaunchConfig3D },
        _ => unreachable!(),
    };
    let max_threads_binding = internal_ident("__cuda_oxide_max_threads");
    let block = if let Some((x, y, z)) = contract.exact_block {
        quote! { ::cuda_core::BlockRequirement::Exact((#x, #y, #z)) }
    } else {
        contract
            .max_block_threads
            .as_ref()
            .expect("validated contract without exact block has launch bounds");
        quote! { ::cuda_core::BlockRequirement::MaxThreads(#max_threads_binding) }
    };
    let launch_bounds_assertions = contract.max_block_threads.as_ref().map(|maximum| {
        let maximum = &maximum.expr;
        let exact_assertion = contract.exact_block.map(|(x, y, z)| {
            let exact_threads = u128::from(x) * u128::from(y) * u128::from(z);
            quote! {
                assert!(
                    #exact_threads <= (#max_threads_binding) as u128,
                    "launch_contract exact block exceeds launch_bounds maximum threads",
                );
            }
        });
        quote! {
            let #max_threads_binding: u32 = #maximum;
            assert!(
                #max_threads_binding > 0,
                "launch_bounds maximum threads must be greater than zero",
            );
            #exact_assertion
        }
    });
    let alignment = contract.dynamic_shared_alignment;
    let dynamic_shared = match contract.dynamic_shared {
        DynamicSharedContract::Exact(bytes) => quote! {
            ::cuda_core::DynamicSharedMemoryRequirement::Exact {
                bytes: #bytes,
                min_alignment: #alignment,
            }
        },
        DynamicSharedContract::Range {
            min_bytes,
            max_bytes,
        } => quote! {
            ::cuda_core::DynamicSharedMemoryRequirement::Range {
                min_bytes: #min_bytes,
                max_bytes: #max_bytes,
                min_alignment: #alignment,
            }
        },
    };
    let kernel_name = kernel.fn_name.to_string();
    let cluster = contract.cluster_tokens(kernel.cluster_dim);
    let cooperative = kernel.cooperative.then(|| quote! { .with_cooperative() });
    let (major, minor) = contract.min_compute_capability;
    let compute_capability = ((major, minor) != (0, 0)).then(|| {
        quote! { .with_min_compute_capability(#major, #minor) }
    });
    let coordinates = contract
        .u32_coordinates
        .then(|| quote! { .with_u32_coordinates() });

    Some(quote! {
        #(#cfg_attrs)*
        impl #impl_generics ::cuda_core::KernelLaunchContract for #marker_ty
        #where_clause
        {
            type Config = #config_ty;
            const SPEC: ::cuda_core::LaunchContractSpec = {
                #launch_bounds_assertions
                ::cuda_core::LaunchContractSpec::new(#kernel_name, #block, #dynamic_shared)
                    #cluster
                    #cooperative
                    #compute_capability
                    #coordinates
            };
        }
    })
}

pub(super) fn generate_cuda_module_prepare_launch_methods(
    kernel: &CudaModuleKernel,
) -> Option<TokenStream2> {
    kernel.launch_contract.as_ref()?;
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let fn_name = &kernel.fn_name;
    let prepare_name = format_ident!("prepare_{}", fn_name);
    let prepare_for_name = format_ident!("prepare_{}_for", fn_name);
    let marker_ty = cuda_module_kernel_marker_type(kernel);
    let generics = kernel.generics.clone();
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let function_binding = cuda_module_function_binding(kernel);
    let function = internal_ident("__cuda_oxide_function");
    let config = internal_ident("__cuda_oxide_config");
    let codegen_args = codegen_generic_arguments(&kernel.generics);
    let turbofish = if codegen_args.is_empty() {
        quote! {}
    } else {
        quote! { ::<#(#codegen_args),*> }
    };
    let type_params: Vec<_> = kernel
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(type_param) => Some(&type_param.ident),
            GenericParam::Lifetime(_) | GenericParam::Const(_) => None,
        })
        .collect();
    let witness_params = type_params.iter().enumerate().map(|(index, ty)| {
        let name = internal_ident(&format!("__cuda_oxide_type_witness_{index}"));
        quote! { #name: &#ty }
    });
    let prepare_for = (!type_params.is_empty()).then(|| {
        quote! {
            #(#cfg_attrs)*
            #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
            #vis fn #prepare_for_name #impl_generics (
                &self,
                #(#witness_params,)*
                #config: <#marker_ty as ::cuda_core::KernelLaunchContract>::Config,
            ) -> ::core::result::Result<
                ::cuda_core::PreparedLaunch<#marker_ty>,
                ::cuda_core::LaunchContractError,
            >
            #where_clause
            {
                self.#prepare_name #turbofish (#config)
            }
        }
    });

    Some(quote! {
        #(#cfg_attrs)*
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis fn #prepare_name #impl_generics (
            &self,
            #config: <#marker_ty as ::cuda_core::KernelLaunchContract>::Config,
        ) -> ::core::result::Result<
            ::cuda_core::PreparedLaunch<#marker_ty>,
            ::cuda_core::LaunchContractError,
        >
        #where_clause
        {
            #function_binding
            unsafe {
                ::cuda_core::PreparedLaunch::<#marker_ty>::__prepare(
                    #function.clone(),
                    #config,
                )
            }
        }

        #prepare_for
    })
}

pub(super) fn generate_cuda_module_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    if kernel.launch_contract.is_some() {
        generate_cuda_module_prepared_launch_method(kernel)
    } else {
        generate_cuda_module_legacy_launch_method(kernel)
    }
}

fn generate_cuda_module_legacy_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let grid_safety_docs = grid_constant_launch_safety_docs(kernel, false);
    let fn_name = &kernel.fn_name;
    let generics = cuda_module_launch_generics(kernel);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params = kernel.params.iter().map(|param| {
        let name = &param.name;
        let host_ty = &param.sync_host_ty;
        quote! { #name: #host_ty }
    });
    let arg_marshalling = kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| cuda_module_arg_marshalling(index, param));
    let function_binding = cuda_module_function_binding(kernel);
    let launch_call = cuda_module_launch_call(kernel);
    let stream = internal_ident("__cuda_oxide_stream");
    let config = internal_ident("__cuda_oxide_config");
    let args = internal_ident("__cuda_oxide_args");

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[doc = "Launches this kernel with an unverified raw launch configuration."]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = "The launch dimensions and resources must satisfy every indexing, memory-access, launch-bounds, and dynamic-shared-memory assumption made by the kernel. Dimensions not represented by the kernel's index model must not introduce overlapping or out-of-bounds accesses. The caller must also uphold any safety requirements documented on the kernel itself."]
        #grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis unsafe fn #fn_name #impl_generics (
            &self,
            #stream: &::cuda_core::CudaStream,
            #config: ::cuda_core::simt::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<(), ::cuda_core::DriverError>
        #where_clause
        {
            #function_binding
            let mut #args: ::std::vec::Vec<*mut ::std::ffi::c_void> = ::std::vec::Vec::new();
            #(#arg_marshalling)*
            #launch_call
        }
    }
}

fn generate_cuda_module_prepared_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let unsafety = &kernel.launch_unsafety;
    let grid_safety_docs = grid_constant_launch_safety_docs(kernel, true);
    let unchecked_grid_safety_docs = grid_constant_launch_safety_docs(kernel, false);
    let fn_name = &kernel.fn_name;
    let unchecked_name = format_ident!("{}_unchecked", fn_name);
    let marker_ty = cuda_module_kernel_marker_type(kernel);
    let generics = cuda_module_launch_generics(kernel);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params: Vec<_> = kernel
        .params
        .iter()
        .map(|param| {
            let name = &param.name;
            let host_ty = &param.sync_host_ty;
            quote! { #name: #host_ty }
        })
        .collect();
    let prepared_arg_marshalling = kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| cuda_module_arg_marshalling(index, param));
    let unchecked_arg_marshalling = kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| cuda_module_arg_marshalling(index, param));
    let unchecked_function_binding = cuda_module_function_binding(kernel);
    let launch_call = cuda_module_launch_call(kernel);
    let unchecked_launch_call = cuda_module_launch_call(kernel);
    let requires_checks = generate_requires_checks(kernel, RequiresLenAccess::SyncBuffer);
    let stream = internal_ident("__cuda_oxide_stream");
    let prepared = internal_ident("__cuda_oxide_prepared");
    let function = internal_ident("__cuda_oxide_function");
    let config = internal_ident("__cuda_oxide_config");
    let args = internal_ident("__cuda_oxide_args");

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis #unsafety fn #fn_name #impl_generics (
            &self,
            #stream: &::cuda_core::CudaStream,
            #prepared: &::cuda_core::PreparedLaunch<#marker_ty>,
            #(#params),*
        ) -> ::core::result::Result<(), ::cuda_core::LaunchContractError>
        #where_clause
        {
            #prepared.validate_stream(#stream)?;
            #requires_checks
            let #function = #prepared.function();
            let #config = #prepared.__raw_config();
            let mut #args: ::std::vec::Vec<*mut ::std::ffi::c_void> = ::std::vec::Vec::new();
            #(#prepared_arg_marshalling)*
            (#launch_call).map_err(::cuda_core::LaunchContractError::from)
        }

        #(#cfg_attrs)*
        #[doc = "Unchecked launch escape hatch for this contracted kernel."]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = "The caller must uphold the kernel's declared geometry, resource, capability, and context contract, including any `requires` size requirements. This escape hatch intentionally skips the contract's checks, so an undersized buffer is not caught before the kernel runs."]
        #unchecked_grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis unsafe fn #unchecked_name #impl_generics (
            &self,
            #stream: &::cuda_core::CudaStream,
            #config: ::cuda_core::simt::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<(), ::cuda_core::DriverError>
        #where_clause
        {
            #unchecked_function_binding
            let mut #args: ::std::vec::Vec<*mut ::std::ffi::c_void> = ::std::vec::Vec::new();
            #(#unchecked_arg_marshalling)*
            #unchecked_launch_call
        }
    }
}

pub(super) fn generate_cuda_module_async_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    if kernel.launch_contract.is_some() {
        generate_cuda_module_prepared_async_launch_method(kernel)
    } else {
        generate_cuda_module_legacy_async_launch_method(kernel)
    }
}

fn generate_cuda_module_legacy_async_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let grid_safety_docs = grid_constant_launch_safety_docs(kernel, false);
    let fn_name = format_ident!("{}_async", kernel.fn_name);
    let generics = cuda_module_async_launch_generics(kernel);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params = kernel.params.iter().map(|param| {
        let name = &param.name;
        let host_ty = &param.async_host_ty;
        quote! { #name: #host_ty }
    });
    let arg_marshalling = kernel.params.iter().map(cuda_module_async_arg_marshalling);
    let function_binding = cuda_module_function_binding(kernel);
    let config = internal_ident("__cuda_oxide_config");
    let function = internal_ident("__cuda_oxide_function");
    let launch = internal_ident("__cuda_oxide_launch");
    let async_lifetime = cuda_module_async_lifetime();
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    let set_cluster_dim = cluster_dim.map(|cluster_dim| {
        quote! {
            ::cuda_host::set_async_kernel_cluster_dim(&mut #launch, #cluster_dim);
        }
    });
    let set_cooperative = kernel.cooperative.then(|| {
        quote! {
            ::cuda_host::set_async_kernel_cooperative(&mut #launch, true);
        }
    });

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[doc = "Builds a lazy async launch from an unverified raw launch configuration."]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = "Before scheduling the returned operation, the launch dimensions and resources must satisfy every indexing, memory-access, launch-bounds, and dynamic-shared-memory assumption made by the kernel. Dimensions not represented by the kernel's index model must not introduce overlapping or out-of-bounds accesses. The caller must also uphold any safety requirements documented on the kernel itself."]
        #grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis unsafe fn #fn_name #impl_generics (
            &self,
            #config: ::cuda_core::simt::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<::cuda_host::AsyncKernelLaunch<#async_lifetime>, ::cuda_core::DriverError>
        #where_clause
        {
            #function_binding
            let mut #launch = ::cuda_host::new_async_kernel_launch_builder(#function.clone());
            #set_cluster_dim
            #set_cooperative
            #(#arg_marshalling)*
            // SAFETY: this method is unsafe and its caller must uphold the raw
            // launch configuration contract documented above.
            Ok(unsafe { #launch.finalize_unchecked(#config) })
        }
    }
}

fn generate_cuda_module_prepared_async_launch_method(kernel: &CudaModuleKernel) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let unsafety = &kernel.launch_unsafety;
    let grid_safety_docs = grid_constant_launch_safety_docs(kernel, true);
    let unchecked_grid_safety_docs = grid_constant_launch_safety_docs(kernel, false);
    let fn_name = format_ident!("{}_async", kernel.fn_name);
    let unchecked_name = format_ident!("{}_async_unchecked", kernel.fn_name);
    let marker_ty = cuda_module_kernel_marker_type(kernel);
    let generics = cuda_module_async_launch_generics(kernel);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params: Vec<_> = kernel
        .params
        .iter()
        .map(|param| {
            let name = &param.name;
            let host_ty = &param.async_host_ty;
            quote! { #name: #host_ty }
        })
        .collect();
    let prepared_marshalling = kernel.params.iter().map(cuda_module_async_arg_marshalling);
    let unchecked_marshalling = kernel.params.iter().map(cuda_module_async_arg_marshalling);
    let unchecked_function_binding = cuda_module_function_binding(kernel);
    let prepared = internal_ident("__cuda_oxide_prepared");
    let config = internal_ident("__cuda_oxide_config");
    let function = internal_ident("__cuda_oxide_function");
    let launch = internal_ident("__cuda_oxide_launch");
    let async_lifetime = cuda_module_async_lifetime();
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    let prepared_cluster = cluster_dim.as_ref().map(|cluster_dim| {
        quote! { ::cuda_host::set_async_kernel_cluster_dim(&mut #launch, #cluster_dim); }
    });
    let unchecked_cluster = cluster_dim.as_ref().map(|cluster_dim| {
        quote! { ::cuda_host::set_async_kernel_cluster_dim(&mut #launch, #cluster_dim); }
    });
    let prepared_cooperative = kernel.cooperative.then(|| {
        quote! { ::cuda_host::set_async_kernel_cooperative(&mut #launch, true); }
    });
    let unchecked_cooperative = kernel.cooperative.then(|| {
        quote! { ::cuda_host::set_async_kernel_cooperative(&mut #launch, true); }
    });
    let requires_checks = generate_requires_checks(kernel, RequiresLenAccess::AsyncRef);
    let launch_ty = quote! {
        ::cuda_host::PreparedAsyncKernelLaunch<#async_lifetime, #marker_ty>
    };
    let final_value = quote! {
        unsafe {
            ::cuda_host::new_prepared_async_kernel_launch(
                #launch,
                (*#prepared).clone(),
            )
        }
    };
    // Size requirements are evaluated eagerly, before the launch is
    // enqueued, so a violated relation must surface as a typed error. Only
    // contracts that declare `requires` pay for the `Result` wrapper.
    let (return_ty, final_expr, requires_doc) = if requires_checks.is_some() {
        (
            quote! {
                ::core::result::Result<#launch_ty, ::cuda_core::LaunchContractError>
            },
            quote! { ::core::result::Result::Ok(#final_value) },
            Some(quote! {
                #[doc = ""]
                #[doc = "Returns a `LaunchContractError` without enqueueing anything if a declared `requires` size requirement does not hold for the supplied arguments."]
            }),
        )
    } else {
        (launch_ty, final_value, None)
    };

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #requires_doc
        #grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis #unsafety fn #fn_name #impl_generics (
            &self,
            #prepared: &::cuda_core::PreparedLaunch<#marker_ty>,
            #(#params),*
        ) -> #return_ty
        #where_clause
        {
            #requires_checks
            let mut #launch =
                ::cuda_host::new_async_kernel_launch_builder(#prepared.function().clone());
            #prepared_cluster
            #prepared_cooperative
            #(#prepared_marshalling)*
            // SAFETY: PreparedLaunch validated this kernel-branded config and
            // the builder has not exposed a way to mutate it after finalization.
            let #launch = unsafe {
                #launch.finalize_unchecked(#prepared.__raw_config())
            };
            #final_expr
        }

        #(#cfg_attrs)*
        #[doc = "Unchecked async launch escape hatch for this contracted kernel."]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = "The caller must uphold the kernel's declared geometry, resource, capability, and context contract when the returned operation is scheduled, including any `requires` size requirements. This escape hatch intentionally skips the contract's checks, so an undersized buffer is not caught before the kernel runs."]
        #unchecked_grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis unsafe fn #unchecked_name #impl_generics (
            &self,
            #config: ::cuda_core::simt::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<
            ::cuda_host::AsyncKernelLaunch<#async_lifetime>,
            ::cuda_core::DriverError,
        >
        #where_clause
        {
            #unchecked_function_binding
            let mut #launch =
                ::cuda_host::new_async_kernel_launch_builder(#function.clone());
            #unchecked_cluster
            #unchecked_cooperative
            #(#unchecked_marshalling)*
            // SAFETY: this method is unsafe and its caller must uphold the
            // contracted kernel's raw launch requirements documented above.
            Ok(unsafe { #launch.finalize_unchecked(#config) })
        }
    }
}

pub(super) fn generate_cuda_module_owned_async_launch_method(
    kernel: &CudaModuleKernel,
) -> TokenStream2 {
    if kernel.launch_contract.is_some() {
        generate_cuda_module_prepared_owned_async_launch_method(kernel)
    } else {
        generate_cuda_module_legacy_owned_async_launch_method(kernel)
    }
}

fn generate_cuda_module_legacy_owned_async_launch_method(
    kernel: &CudaModuleKernel,
) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let grid_safety_docs = grid_constant_launch_safety_docs(kernel, false);
    let fn_name = format_ident!("{}_async_owned", kernel.fn_name);
    let resources = cuda_module_owned_resource_params(kernel);
    let generics = cuda_module_owned_async_launch_generics(kernel, &resources);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params = kernel.params.iter().enumerate().map(|(index, param)| {
        let name = &param.name;
        match &param.marshal {
            CudaModuleParamMarshal::Scalar => {
                let host_ty = &param.async_host_ty;
                quote! { #name: #host_ty }
            }
            CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
                let resource_ty = cuda_module_owned_resource_type(index);
                quote! { #name: #resource_ty }
            }
            CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
                let resource_ty = cuda_module_owned_resource_type(index);
                quote! { mut #name: #resource_ty }
            }
            // The owned resource carries the buffer; `RowWidthOwned` adds the
            // row width the kernel will index it by.
            CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => {
                let resource_ty = cuda_module_owned_resource_type(index);
                quote! { mut #name: ::cuda_host::RowWidthOwned<#resource_ty> }
            }
        }
    });
    let arg_marshalling = kernel
        .params
        .iter()
        .map(cuda_module_owned_async_arg_marshalling);
    let function_binding = cuda_module_function_binding(kernel);
    let config = internal_ident("__cuda_oxide_config");
    let function = internal_ident("__cuda_oxide_function");
    let launch = internal_ident("__cuda_oxide_launch");
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    let set_cluster_dim = cluster_dim.map(|cluster_dim| {
        quote! {
            ::cuda_host::set_async_kernel_cluster_dim(&mut #launch, #cluster_dim);
        }
    });
    let set_cooperative = kernel.cooperative.then(|| {
        quote! {
            ::cuda_host::set_async_kernel_cooperative(&mut #launch, true);
        }
    });
    let resources_ty = cuda_module_owned_resources_ty(&resources);
    let resource_names = resources.iter().map(|(_, name, _, _, _)| name);
    let resources_expr = if resources.is_empty() {
        quote! { () }
    } else {
        quote! { (#(#resource_names),*) }
    };

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[doc = "Builds a lazy owned async launch from an unverified raw launch configuration."]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = "Before scheduling the returned operation, the launch dimensions and resources must satisfy every indexing, memory-access, launch-bounds, and dynamic-shared-memory assumption made by the kernel. Dimensions not represented by the kernel's index model must not introduce overlapping or out-of-bounds accesses. The caller must also uphold any safety requirements documented on the kernel itself."]
        #grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis unsafe fn #fn_name #impl_generics (
            &self,
            #config: ::cuda_core::simt::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<::cuda_host::OwnedAsyncKernelLaunch<#resources_ty>, ::cuda_core::DriverError>
        #where_clause
        {
            #function_binding
            let mut #launch =
                ::cuda_host::new_async_kernel_launch_builder(#function.clone());
            #set_cluster_dim
            #set_cooperative
            #(#arg_marshalling)*
            // SAFETY: this method is unsafe and its caller must uphold the raw
            // launch configuration contract documented above.
            let #launch: ::cuda_host::AsyncKernelLaunch<'static> =
                unsafe { #launch.finalize_unchecked(#config) };
            Ok(::cuda_host::new_owned_async_kernel_launch(#launch, #resources_expr))
        }
    }
}

fn generate_cuda_module_prepared_owned_async_launch_method(
    kernel: &CudaModuleKernel,
) -> TokenStream2 {
    let vis = &kernel.vis;
    let cfg_attrs = &kernel.cfg_attrs;
    let method_attrs = &kernel.method_attrs;
    let unsafety = &kernel.launch_unsafety;
    let grid_safety_docs = grid_constant_launch_safety_docs(kernel, true);
    let unchecked_grid_safety_docs = grid_constant_launch_safety_docs(kernel, false);
    let fn_name = format_ident!("{}_async_owned", kernel.fn_name);
    let unchecked_name = format_ident!("{}_async_owned_unchecked", kernel.fn_name);
    let marker_ty = cuda_module_kernel_marker_type(kernel);
    let resources = cuda_module_owned_resource_params(kernel);
    let generics = cuda_module_owned_async_launch_generics(kernel, &resources);
    let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();
    let params: Vec<_> = kernel
        .params
        .iter()
        .enumerate()
        .map(|(index, param)| {
            let name = &param.name;
            match &param.marshal {
                CudaModuleParamMarshal::Scalar => {
                    let host_ty = &param.async_host_ty;
                    quote! { #name: #host_ty }
                }
                CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
                    let resource_ty = cuda_module_owned_resource_type(index);
                    quote! { #name: #resource_ty }
                }
                CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
                    let resource_ty = cuda_module_owned_resource_type(index);
                    quote! { mut #name: #resource_ty }
                }
                // The owned resource carries the buffer; `RowWidthOwned` adds
                // the row width the kernel will index it by.
                CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => {
                    let resource_ty = cuda_module_owned_resource_type(index);
                    quote! { mut #name: ::cuda_host::RowWidthOwned<#resource_ty> }
                }
            }
        })
        .collect();
    let prepared_marshalling = kernel
        .params
        .iter()
        .map(cuda_module_owned_async_arg_marshalling);
    let unchecked_marshalling = kernel
        .params
        .iter()
        .map(cuda_module_owned_async_arg_marshalling);
    let unchecked_function_binding = cuda_module_function_binding(kernel);
    let prepared = internal_ident("__cuda_oxide_prepared");
    let config = internal_ident("__cuda_oxide_config");
    let function = internal_ident("__cuda_oxide_function");
    let launch = internal_ident("__cuda_oxide_launch");
    let owned = internal_ident("__cuda_oxide_owned");
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    let prepared_cluster = cluster_dim.as_ref().map(|cluster_dim| {
        quote! { ::cuda_host::set_async_kernel_cluster_dim(&mut #launch, #cluster_dim); }
    });
    let unchecked_cluster = cluster_dim.as_ref().map(|cluster_dim| {
        quote! { ::cuda_host::set_async_kernel_cluster_dim(&mut #launch, #cluster_dim); }
    });
    let prepared_cooperative = kernel.cooperative.then(|| {
        quote! { ::cuda_host::set_async_kernel_cooperative(&mut #launch, true); }
    });
    let unchecked_cooperative = kernel.cooperative.then(|| {
        quote! { ::cuda_host::set_async_kernel_cooperative(&mut #launch, true); }
    });
    let resources_ty = cuda_module_owned_resources_ty(&resources);
    let prepared_resource_names = resources.iter().map(|(_, name, _, _, _)| name);
    let unchecked_resource_names = resources.iter().map(|(_, name, _, _, _)| name);
    let prepared_resources_expr = if resources.is_empty() {
        quote! { () }
    } else {
        quote! { (#(#prepared_resource_names),*) }
    };
    let unchecked_resources_expr = if resources.is_empty() {
        quote! { () }
    } else {
        quote! { (#(#unchecked_resource_names),*) }
    };
    let requires_checks = generate_requires_checks(kernel, RequiresLenAccess::OwnedValue);
    let launch_ty = quote! {
        ::cuda_host::PreparedOwnedAsyncKernelLaunch<#resources_ty, #marker_ty>
    };
    let final_value = quote! {
        unsafe {
            ::cuda_host::new_prepared_owned_async_kernel_launch(
                #owned,
                (*#prepared).clone(),
            )
        }
    };
    // Same policy as the borrowed async launcher: `requires` relations are
    // checked eagerly, before any resource is moved into the launch, and a
    // violation surfaces as a typed error instead of an enqueued fault.
    let (return_ty, final_expr, requires_doc) = if requires_checks.is_some() {
        (
            quote! {
                ::core::result::Result<#launch_ty, ::cuda_core::LaunchContractError>
            },
            quote! { ::core::result::Result::Ok(#final_value) },
            Some(quote! {
                #[doc = ""]
                #[doc = "Returns a `LaunchContractError` without enqueueing anything if a declared `requires` size requirement does not hold for the supplied arguments."]
            }),
        )
    } else {
        (launch_ty, final_value, None)
    };

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #requires_doc
        #grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis #unsafety fn #fn_name #impl_generics (
            &self,
            #prepared: &::cuda_core::PreparedLaunch<#marker_ty>,
            #(#params),*
        ) -> #return_ty
        #where_clause
        {
            #requires_checks
            let mut #launch =
                ::cuda_host::new_async_kernel_launch_builder(#prepared.function().clone());
            #prepared_cluster
            #prepared_cooperative
            #(#prepared_marshalling)*
            // SAFETY: PreparedLaunch validated this kernel-branded config and
            // the builder has not exposed a way to mutate it after finalization.
            let #launch: ::cuda_host::AsyncKernelLaunch<'static> = unsafe {
                #launch.finalize_unchecked(#prepared.__raw_config())
            };
            let #owned =
                ::cuda_host::new_owned_async_kernel_launch(#launch, #prepared_resources_expr);
            #final_expr
        }

        #(#cfg_attrs)*
        #[doc = "Unchecked owned-async launch escape hatch for this contracted kernel."]
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = "The caller must uphold the kernel's declared geometry, resource, capability, and context contract when the returned operation is scheduled, including any `requires` size requirements. This escape hatch intentionally skips the contract's checks, so an undersized buffer is not caught before the kernel runs."]
        #unchecked_grid_safety_docs
        #[allow(clippy::multiple_bound_locations, clippy::too_many_arguments)]
        #vis unsafe fn #unchecked_name #impl_generics (
            &self,
            #config: ::cuda_core::simt::LaunchConfig,
            #(#params),*
        ) -> ::core::result::Result<
            ::cuda_host::OwnedAsyncKernelLaunch<#resources_ty>,
            ::cuda_core::DriverError,
        >
        #where_clause
        {
            #unchecked_function_binding
            let mut #launch =
                ::cuda_host::new_async_kernel_launch_builder(#function.clone());
            #unchecked_cluster
            #unchecked_cooperative
            #(#unchecked_marshalling)*
            // SAFETY: this method is unsafe and its caller must uphold the
            // contracted kernel's raw launch requirements documented above.
            let #launch: ::cuda_host::AsyncKernelLaunch<'static> =
                unsafe { #launch.finalize_unchecked(#config) };
            Ok(::cuda_host::new_owned_async_kernel_launch(
                #launch,
                #unchecked_resources_expr,
            ))
        }
    }
}

fn cuda_module_launch_generics(kernel: &CudaModuleKernel) -> syn::Generics {
    let mut generics = kernel.generics.clone();
    for param in &kernel.params {
        if matches!(param.marshal, CudaModuleParamMarshal::Scalar) {
            let host_ty = &param.sync_host_ty;
            generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote! { #host_ty: ::cuda_host::KernelScalar });
        }
    }
    generics
}

fn cuda_module_owned_async_launch_generics(
    kernel: &CudaModuleKernel,
    resources: &[(usize, Ident, TokenStream2, bool, bool)],
) -> syn::Generics {
    let mut generics = kernel.generics.clone();
    for (index, _, elem_ty, writable, _) in resources {
        let resource_ty = cuda_module_owned_resource_type(*index);
        generics.params.push(syn::parse_quote! { #resource_ty });
        let predicate: syn::WherePredicate = if *writable {
            syn::parse_quote! {
                #resource_ty: ::cuda_host::KernelSliceArgMut<Elem = #elem_ty> + Send + 'static
            }
        } else {
            syn::parse_quote! {
                #resource_ty: ::cuda_host::KernelSliceArg<Elem = #elem_ty> + Send + 'static
            }
        };
        generics.make_where_clause().predicates.push(predicate);
    }
    for param in &kernel.params {
        if matches!(param.marshal, CudaModuleParamMarshal::Scalar) {
            let host_ty = &param.async_host_ty;
            generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote! { #host_ty: ::cuda_host::KernelScalar + 'static });
        }
    }
    generics
}

fn cuda_module_async_launch_generics(kernel: &CudaModuleKernel) -> syn::Generics {
    let mut generics = kernel.generics.clone();
    let async_lifetime = cuda_module_async_lifetime();
    let lifetime_param = syn::LifetimeParam::new(async_lifetime.clone());
    generics
        .params
        .insert(0, syn::GenericParam::Lifetime(lifetime_param));
    for param in &kernel.params {
        if matches!(param.marshal, CudaModuleParamMarshal::Scalar) {
            let host_ty = &param.async_host_ty;
            generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote! { #host_ty: ::cuda_host::KernelScalar + #async_lifetime });
        }
    }
    generics
}

fn cuda_module_owned_resource_params(
    kernel: &CudaModuleKernel,
) -> Vec<(usize, Ident, TokenStream2, bool, bool)> {
    kernel
        .params
        .iter()
        .enumerate()
        .filter_map(|(index, param)| match &param.marshal {
            CudaModuleParamMarshal::Scalar => None,
            CudaModuleParamMarshal::ReadOnlyDeviceBuffer { elem_ty } => {
                Some((index, param.name.clone(), elem_ty.clone(), false, false))
            }
            CudaModuleParamMarshal::WritableDeviceBuffer { elem_ty } => {
                Some((index, param.name.clone(), elem_ty.clone(), true, false))
            }
            // A row-width slice still owns a buffer resource; the width
            // rides beside it in `RowWidthOwned`.
            CudaModuleParamMarshal::RowWidthDeviceBuffer { elem_ty } => {
                Some((index, param.name.clone(), elem_ty.clone(), true, true))
            }
        })
        .collect()
}

fn cuda_module_owned_resource_type(index: usize) -> Ident {
    internal_ident(&format!("__CudaModuleArg{index}"))
}

fn cuda_module_owned_resources_ty(
    resources: &[(usize, Ident, TokenStream2, bool, bool)],
) -> TokenStream2 {
    if resources.is_empty() {
        quote! { () }
    } else {
        let resource_tys = resources.iter().map(|(index, _, _, _, has_row_width)| {
            let resource_ty = cuda_module_owned_resource_type(*index);
            if *has_row_width {
                quote! { ::cuda_host::RowWidthOwned<#resource_ty> }
            } else {
                quote! { #resource_ty }
            }
        });
        quote! { (#(#resource_tys),*) }
    }
}

fn cuda_module_arg_marshalling(index: usize, param: &CudaModuleParam) -> TokenStream2 {
    let name = &param.name;
    let args = internal_ident("__cuda_oxide_args");
    let value_name = internal_ident(&format!("__cuda_oxide_arg_{index}"));
    match param.marshal {
        CudaModuleParamMarshal::Scalar => {
            quote! {
                let mut #value_name = #name;
                ::cuda_host::push_kernel_scalar(&mut #args, &mut #value_name);
            }
        }
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
            let ptr_name = internal_ident(&format!("__cuda_oxide_arg_{index}_ptr"));
            let len_name = internal_ident(&format!("__cuda_oxide_arg_{index}_len"));
            quote! {
                let (mut #ptr_name, mut #len_name) =
                    ::cuda_host::read_only_device_buffer_arg(#name);
                ::cuda_host::push_kernel_device_slice(
                    &mut #args,
                    &mut #ptr_name,
                    &mut #len_name,
                );
            }
        }
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
            let ptr_name = internal_ident(&format!("__cuda_oxide_arg_{index}_ptr"));
            let len_name = internal_ident(&format!("__cuda_oxide_arg_{index}_len"));
            quote! {
                let (mut #ptr_name, mut #len_name) =
                    ::cuda_host::writable_device_buffer_arg(#name);
                ::cuda_host::push_kernel_device_slice(
                    &mut #args,
                    &mut #ptr_name,
                    &mut #len_name,
                );
            }
        }
        CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => {
            let ptr_name = internal_ident(&format!("__cuda_oxide_arg_{index}_ptr"));
            let len_name = internal_ident(&format!("__cuda_oxide_arg_{index}_len"));
            let width_name = internal_ident(&format!("__cuda_oxide_arg_{index}_row_width"));
            quote! {
                let (mut #ptr_name, mut #len_name, mut #width_name) =
                    ::cuda_host::row_width_device_buffer_arg(#name);
                ::cuda_host::push_kernel_row_width_device_slice(
                    &mut #args,
                    &mut #ptr_name,
                    &mut #len_name,
                    &mut #width_name,
                );
            }
        }
    }
}

fn cuda_module_owned_async_arg_marshalling(param: &CudaModuleParam) -> TokenStream2 {
    let name = &param.name;
    let launch = internal_ident("__cuda_oxide_launch");
    match param.marshal {
        CudaModuleParamMarshal::Scalar => {
            quote! {
                ::cuda_host::push_async_kernel_scalar(&mut #launch, #name);
            }
        }
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_read_only_device_slice(&mut #launch, &#name);
            }
        }
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_writable_device_slice(&mut #launch, &mut #name);
            }
        }
        CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_owned_row_width_device_slice(&mut #launch, &mut #name);
            }
        }
    }
}

fn cuda_module_async_arg_marshalling(param: &CudaModuleParam) -> TokenStream2 {
    let name = &param.name;
    let launch = internal_ident("__cuda_oxide_launch");
    match param.marshal {
        CudaModuleParamMarshal::Scalar => {
            quote! {
                ::cuda_host::push_async_kernel_scalar(&mut #launch, #name);
            }
        }
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_read_only_device_slice(&mut #launch, #name);
            }
        }
        CudaModuleParamMarshal::WritableDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_writable_device_slice(&mut #launch, #name);
            }
        }
        CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => {
            quote! {
                ::cuda_host::push_async_row_width_device_slice(&mut #launch, #name);
            }
        }
    }
}

fn cuda_module_function_binding(kernel: &CudaModuleKernel) -> TokenStream2 {
    let function = internal_ident("__cuda_oxide_function");
    if kernel.is_generic {
        let ptx_name_fn = format_ident!("{}_ptx_name", kernel.fn_name);
        let codegen_args = codegen_generic_arguments(&kernel.generics);
        let turbofish = if codegen_args.is_empty() {
            quote! {}
        } else {
            quote! { ::<#(#codegen_args),*> }
        };
        let ptx_name = internal_ident("__cuda_oxide_ptx_name");
        let function_storage = internal_ident("__cuda_oxide_function_storage");
        let cache = internal_ident("__cuda_oxide_function_cache");
        let error = internal_ident("__cuda_oxide_error");
        quote! {
            let #ptx_name = #ptx_name_fn #turbofish ();
            let #function_storage = {
                let mut #cache = self
                    .__generic_functions
                    .lock()
                    .expect("cuda_module generic function cache poisoned");
                if let Some(#function) = #cache.get(#ptx_name) {
                    #function.clone()
                } else {
                    // A `_TID_` lookup miss can mean the host and the device
                    // backend disagreed on the kernel's type-identity hash;
                    // the diagnosis panics with a self-explaining message in
                    // that case and passes ordinary misses through unchanged.
                    let #function = self
                        .__module
                        .load_function(#ptx_name)
                        .map_err(|#error| {
                            ::cuda_host::diagnose_generic_kernel_load_error(
                                &self.__module,
                                #ptx_name,
                                #error,
                            )
                        })?;
                    #cache.insert(#ptx_name, #function.clone());
                    #function
                }
            };
            let #function = &#function_storage;
        }
    } else {
        let field = cuda_module_function_field(&kernel.fn_name);
        quote! {
            let #function = &self.#field;
        }
    }
}

fn cuda_module_launch_call(kernel: &CudaModuleKernel) -> TokenStream2 {
    let function = internal_ident("__cuda_oxide_function");
    let stream = internal_ident("__cuda_oxide_stream");
    let config = internal_ident("__cuda_oxide_config");
    let args = internal_ident("__cuda_oxide_args");
    let cluster_dim = kernel.cluster_dim.map(|(x, y, z)| quote! { (#x, #y, #z) });
    match (cluster_dim, kernel.cooperative) {
        (Some(cluster_dim), true) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_ex_cooperative_on_stream(
                    #function,
                    #config.grid_dim,
                    #config.block_dim,
                    #config.shared_mem_bytes,
                    #cluster_dim,
                    #stream,
                    &mut #args,
                )
            }
        },
        (Some(cluster_dim), false) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_ex_on_stream(
                    #function,
                    #config.grid_dim,
                    #config.block_dim,
                    #config.shared_mem_bytes,
                    #cluster_dim,
                    #stream,
                    &mut #args,
                )
            }
        },
        (None, true) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_cooperative_on_stream(
                    #function,
                    #config.grid_dim,
                    #config.block_dim,
                    #config.shared_mem_bytes,
                    #stream,
                    &mut #args,
                )
            }
        },
        (None, false) => quote! {
            unsafe {
                ::cuda_core::launch_kernel_on_stream(
                    #function,
                    #config.grid_dim,
                    #config.block_dim,
                    #config.shared_mem_bytes,
                    #stream,
                    &mut #args,
                )
            }
        },
    }
}

/// Whether rustc must create distinct code for this generic parameter list.
///
/// Lifetimes are deliberately excluded: rustc erases them before codegen, so
/// they cannot distinguish PTX entry points. Type and const parameters both
/// participate in monomorphization and therefore in kernel identity.
pub(crate) fn has_codegen_generics(generics: &syn::Generics) -> bool {
    generics
        .params
        .iter()
        .any(|param| matches!(param, GenericParam::Type(_) | GenericParam::Const(_)))
}

/// Ordered generic arguments accepted by a function turbofish.
///
/// Rust function turbofish syntax omits lifetime arguments. Type and const
/// identifiers must remain in declaration order so mixed `<T, const N>`
/// kernels instantiate the exact function item requested by the caller.
pub(crate) fn codegen_generic_arguments(generics: &syn::Generics) -> Vec<TokenStream2> {
    generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(type_param) => {
                let ident = &type_param.ident;
                Some(quote! { #ident })
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                Some(quote! { #ident })
            }
            GenericParam::Lifetime(_) => None,
        })
        .collect()
}

/// Ordered arguments for applying a generated marker type.
///
/// Unlike function turbofish syntax, a type application retains lifetime
/// parameters. The specialization hash still ignores their identity because
/// rustc's `TypeId` pipeline erases regions.
pub(crate) fn generic_arguments(generics: &syn::Generics) -> Vec<TokenStream2> {
    generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(lifetime_param) => {
                let lifetime = &lifetime_param.lifetime;
                quote! { #lifetime }
            }
            GenericParam::Type(type_param) => {
                let ident = &type_param.ident;
                quote! { #ident }
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                quote! { #ident }
            }
        })
        .collect()
}

/// Types used by `PhantomData` so generated marker types correctly witness
/// every lifetime and type parameter. Const parameters are part of the marker
/// type by construction and need no field-level witness.
pub(crate) fn generic_phantom_type(generics: &syn::Generics) -> TokenStream2 {
    let witnesses: Vec<TokenStream2> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Lifetime(lifetime_param) => {
                let lifetime = &lifetime_param.lifetime;
                Some(quote! { &#lifetime () })
            }
            GenericParam::Type(type_param) => {
                let ident = &type_param.ident;
                Some(quote! { *const #ident })
            }
            GenericParam::Const(_) => None,
        })
        .collect();

    if witnesses.is_empty() {
        quote! { () }
    } else {
        quote! { (#(#witnesses,)*) }
    }
}

pub(super) fn cuda_module_function_field(fn_name: &Ident) -> Ident {
    format_ident!("__{}_function", fn_name)
}

pub(super) fn cuda_kernel_marker_name(fn_name: &Ident) -> Ident {
    format_ident!("__{}_CudaKernel", fn_name)
}

pub(super) fn cuda_kernel_signature_name(fn_name: &Ident) -> Ident {
    let name = fn_name.to_string();
    let name = name.strip_prefix("r#").unwrap_or(&name);
    format_ident!("{}_CUDA_SIGNATURE", name)
}
