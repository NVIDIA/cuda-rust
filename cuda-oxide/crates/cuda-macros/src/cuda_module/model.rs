/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel and parameter models for `#[cuda_module]`: marshalling
//! classification and host-side type mapping.

use crate::common::{cuda_module_async_lifetime, grid_constant_pointee, internal_ident};
use crate::cuda_module::contract::CudaModuleLaunchContract;
use crate::cuda_module::launchers::{cuda_kernel_marker_name, generic_arguments};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{FnArg, GenericArgument, Ident, ItemFn, Pat, PathArguments, Token, Type, parse_quote};

pub(crate) struct CudaModuleKernel {
    pub(super) module_path: Vec<Ident>,
    pub(super) vis: syn::Visibility,
    /// Availability attributes written directly on the kernel. Generated
    /// fields and methods live in the same module, so ancestor attributes are
    /// already inherited there.
    pub(crate) cfg_attrs: Vec<syn::Attribute>,
    /// Availability attributes from every enclosing inline module followed by
    /// the kernel's own attributes. Root-level artifact references need the
    /// complete chain because they live outside those child modules.
    pub(crate) effective_cfg_attrs: Vec<syn::Attribute>,
    pub(super) method_attrs: Vec<syn::Attribute>,
    /// Host launch safety is independent of the source function's Rust safety:
    /// copying a grid parameter does not prove its nested values device-valid.
    pub(super) launch_unsafety: Option<Token![unsafe]>,
    pub(crate) fn_name: Ident,
    pub(super) generics: syn::Generics,
    pub(super) params: Vec<CudaModuleParam>,
    pub(super) cluster_dim: Option<(u32, u32, u32)>,
    pub(super) cooperative: bool,
    pub(super) launch_contract: Option<CudaModuleLaunchContract>,
    pub(super) is_generic: bool,
}

pub(crate) struct CudaModuleParam {
    pub(crate) name: Ident,
    /// Type as written in the device entry declaration.
    pub(crate) device_ty: Type,
    /// Pointee carried by `#[grid_constant]`, when present.
    pub(crate) grid_constant_ty: Option<Type>,
    pub(crate) sync_host_ty: TokenStream2,
    pub(crate) async_host_ty: TokenStream2,
    pub(crate) marshal: CudaModuleParamMarshal,
    pub(crate) grid_constant: bool,
    pub(crate) mutable_slice: bool,
    pub(crate) disjoint_slice_ty: Option<Type>,
    pub(crate) disjoint_slice_elem: Option<TokenStream2>,
    /// Declared type and carried scalar of a `Uniform<T>` parameter, used to
    /// bound the generated impl by the sealed proof trait so a local type also
    /// named `Uniform` cannot borrow the scalar host ABI.
    pub(crate) uniform_ty: Option<Type>,
    pub(crate) uniform_scalar: Option<Type>,
    /// Integer classification of a scalar parameter's declared type, used to
    /// decide which scalars may appear in `requires` relations. Always
    /// `Other` for non-scalar parameters.
    pub(crate) scalar_int: ScalarIntClass,
}

pub(crate) enum CudaModuleParamMarshal {
    Scalar,
    ReadOnlyDeviceBuffer {
        elem_ty: TokenStream2,
    },
    WritableDeviceBuffer {
        elem_ty: TokenStream2,
    },
    /// A writable buffer whose index space carries a runtime row width, so
    /// the host supplies the width and the packet gains a third slot.
    RowWidthDeviceBuffer {
        elem_ty: TokenStream2,
    },
}

/// Integer classification of a kernel scalar parameter as declared in source.
///
/// `requires` relations widen every operand to `u64`, which is lossless for
/// unsigned scalars but raises sign-extension questions for signed ones, so
/// v1 accepts only `Unsigned` scalars and rejects the rest at expansion time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarIntClass {
    /// `u8`, `u16`, `u32`, `u64`, or `usize`.
    Unsigned,
    /// `i8`, `i16`, `i32`, `i64`, or `isize`.
    Signed,
    /// Anything else (floats, pointers, generics, non-scalar parameters).
    Other,
}

pub(crate) fn scalar_int_class(ty: &Type) -> ScalarIntClass {
    // A `Uniform<T>` parameter is marshalled as `T` and is evaluated as `T` on
    // the host, so a relation over it has the same widening behaviour as one
    // over the bare scalar.
    if let Some(scalar) = cuda_module_uniform_scalar(ty) {
        return scalar_int_class(scalar);
    }

    let Type::Path(type_path) = ty else {
        return ScalarIntClass::Other;
    };
    if type_path.qself.is_some() {
        return ScalarIntClass::Other;
    }
    let Some(ident) = type_path.path.get_ident() else {
        return ScalarIntClass::Other;
    };
    match ident.to_string().as_str() {
        "u8" | "u16" | "u32" | "u64" | "usize" => ScalarIntClass::Unsigned,
        "i8" | "i16" | "i32" | "i64" | "isize" => ScalarIntClass::Signed,
        _ => ScalarIntClass::Other,
    }
}

pub(super) fn cuda_module_params(item_fn: &ItemFn) -> syn::Result<Vec<CudaModuleParam>> {
    item_fn
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(receiver) => Err(syn::Error::new_spanned(
                receiver,
                "cuda_module kernels cannot take self parameters",
            )),
            FnArg::Typed(pat_type) => cuda_module_param_from_typed(pat_type),
        })
        .collect()
}

pub(crate) fn cuda_module_param_from_typed(
    pat_type: &syn::PatType,
) -> syn::Result<CudaModuleParam> {
    let Pat::Ident(pat_ident) = &*pat_type.pat else {
        return Err(syn::Error::new_spanned(
            &pat_type.pat,
            "cuda_module only supports simple identifier kernel parameters",
        ));
    };
    let name = pat_ident.ident.clone();
    let grid_constant_pointee = grid_constant_pointee(pat_type)?;
    let (sync_host_ty, async_host_ty, marshal) =
        cuda_module_host_type(&pat_type.ty, grid_constant_pointee.as_ref())?;
    let mutable_slice = cuda_module_slice_elem(&pat_type.ty).is_some_and(|(_, mutable)| mutable);
    let disjoint_slice_elem = cuda_module_disjoint_slice_elem(&pat_type.ty);
    let disjoint_slice_ty = disjoint_slice_elem
        .as_ref()
        .map(|_| pat_type.ty.as_ref().clone());
    let uniform_scalar = cuda_module_uniform_scalar(&pat_type.ty).cloned();
    let uniform_ty = uniform_scalar
        .as_ref()
        .map(|_| pat_type.ty.as_ref().clone());
    Ok(CudaModuleParam {
        name,
        device_ty: pat_type.ty.as_ref().clone(),
        sync_host_ty,
        async_host_ty,
        marshal,
        grid_constant: grid_constant_pointee.is_some(),
        grid_constant_ty: grid_constant_pointee,
        mutable_slice,
        disjoint_slice_ty,
        disjoint_slice_elem,
        uniform_ty,
        uniform_scalar,
        scalar_int: scalar_int_class(&pat_type.ty),
    })
}

fn cuda_module_host_type(
    ty: &Type,
    grid_constant_pointee: Option<&Type>,
) -> syn::Result<(TokenStream2, TokenStream2, CudaModuleParamMarshal)> {
    let async_lifetime = cuda_module_async_lifetime();
    if let Some(pointee) = grid_constant_pointee {
        return Ok((
            quote! { #pointee },
            quote! { #pointee },
            CudaModuleParamMarshal::Scalar,
        ));
    }
    if let Some((elem_ty, mutable)) = cuda_module_slice_elem(ty) {
        let sync_host_ty = if mutable {
            quote! { &mut ::cuda_core::DeviceBuffer<#elem_ty> }
        } else {
            quote! { &::cuda_core::DeviceBuffer<#elem_ty> }
        };
        let (async_host_ty, marshal) = if mutable {
            (
                quote! { &#async_lifetime mut impl ::cuda_host::KernelSliceArgMut<Elem = #elem_ty> },
                CudaModuleParamMarshal::WritableDeviceBuffer {
                    elem_ty: quote! { #elem_ty },
                },
            )
        } else {
            (
                quote! { &#async_lifetime impl ::cuda_host::KernelSliceArg<Elem = #elem_ty> },
                CudaModuleParamMarshal::ReadOnlyDeviceBuffer {
                    elem_ty: quote! { #elem_ty },
                },
            )
        };
        return Ok((sync_host_ty, async_host_ty, marshal));
    }

    if let Some(elem_ty) = cuda_module_disjoint_slice_elem(ty) {
        if cuda_module_disjoint_slice_has_row_width(ty) {
            return Ok((
                quote! { ::cuda_host::RowWidth<'_, #elem_ty> },
                quote! { ::cuda_host::RowWidth<#async_lifetime, #elem_ty> },
                CudaModuleParamMarshal::RowWidthDeviceBuffer {
                    elem_ty: quote! { #elem_ty },
                },
            ));
        }
        return Ok((
            quote! { &mut ::cuda_core::DeviceBuffer<#elem_ty> },
            quote! { &#async_lifetime mut impl ::cuda_host::KernelSliceArgMut<Elem = #elem_ty> },
            CudaModuleParamMarshal::WritableDeviceBuffer {
                elem_ty: quote! { #elem_ty },
            },
        ));
    }

    // A `Uniform<T>` parameter is marshalled exactly like `T`. The host takes
    // the bare scalar because the host is what makes the value uniform: one
    // marshalled value reaches every thread of the launch.
    if let Some(scalar) = cuda_module_uniform_scalar(ty) {
        return Ok((
            quote! { #scalar },
            quote! { #scalar },
            CudaModuleParamMarshal::Scalar,
        ));
    }

    if matches!(ty, Type::Reference(_)) {
        return Err(syn::Error::new_spanned(
            ty,
            "cuda_module only supports slice references; use &[T], &mut [T], DisjointSlice<T>, a raw pointer, or a by-value KernelScalar",
        ));
    }

    Ok((
        quote! { #ty },
        quote! { #ty },
        CudaModuleParamMarshal::Scalar,
    ))
}

fn cuda_module_slice_elem(ty: &Type) -> Option<(TokenStream2, bool)> {
    let Type::Reference(type_ref) = ty else {
        return None;
    };
    let Type::Slice(slice) = &*type_ref.elem else {
        return None;
    };
    let elem = &slice.elem;
    Some((quote! { #elem }, type_ref.mutability.is_some()))
}

/// Scalar carried by a `Uniform<T>` kernel parameter, if the type is spelled
/// that way.
///
/// The host method takes the bare `T`: the host is the source of the
/// uniformity proof, since it marshals one value into the launch packet for
/// the whole grid. `Uniform<T>` is `#[repr(transparent)]`, so the launch packet
/// is byte-identical either way.
pub(crate) fn cuda_module_uniform_scalar(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Uniform" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let scalar = args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })?;
    Some(scalar)
}

/// True when a `DisjointSlice`'s index space carries a runtime row width.
///
/// Matched on the spelling of the index-space argument, exactly as the element
/// type is. The spelling only selects the host ABI; `SpaceLayout` is what
/// decides whether the device type really carries the width, and a mismatch
/// between the two is a type error at the generated call rather than a silent
/// packet-shape difference.
fn cuda_module_disjoint_slice_has_row_width(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(segment) = type_path.path.segments.last() else {
        return false;
    };
    if segment.ident != "DisjointSlice" {
        return false;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return false;
    };
    let mut space = args.args.iter().filter_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    space.nth(1).is_some_and(|space| {
        let Type::Path(space_path) = space else {
            return false;
        };
        space_path.path.segments.last().is_some_and(|segment| {
            segment.ident == "RuntimeRowMajorTiles" || segment.ident == "Runtime2DIndex"
        })
    })
}

fn cuda_module_disjoint_slice_elem(ty: &Type) -> Option<TokenStream2> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "DisjointSlice" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let type_args: Vec<_> = args
        .args
        .iter()
        .filter_map(|arg| match arg {
            GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect();
    let elem = *type_args.first()?;
    Some(quote! { #elem })
}

/// Adds a semantic launch-domain proof for every writable device slice.
///
/// The macro only uses the outer `DisjointSlice` spelling to select the host
/// ABI. The Rust compiler decides whether the *resolved, complete type* is a
/// genuine cuda-device slice with a compatible index space. This makes type
/// aliases work without letting a local look-alike bypass the contract.
pub(super) fn add_cuda_module_disjoint_contract_bounds(
    generics: &mut syn::Generics,
    params: &[CudaModuleParam],
    domain: u8,
) {
    for param in params {
        let (Some(device_ty), Some(element_ty)) =
            (&param.disjoint_slice_ty, &param.disjoint_slice_elem)
        else {
            continue;
        };
        let (device_ty, bound_lifetime) = cuda_module_disjoint_bound_type(device_ty);
        generics.make_where_clause().predicates.push(parse_quote! {
            for<#bound_lifetime> #device_ty:
                ::cuda_device::__LaunchContractDisjointSlice<#element_ty, #domain>
        });
    }
}

/// Requires every `DisjointSlice` parameter's resolved type to carry exactly
/// the launch-packet shape the macro chose for it.
///
/// The macro picks the two-word `(ptr, len)` or three-word `(ptr, len, width)`
/// host marshalling from the index space's spelling, which type aliases can
/// defeat: `type Rt = RuntimeRowMajorTiles<1, 1>;` spells a flat slice over a
/// runtime-width space, and the launch would then push two kernel parameters
/// for a three-parameter kernel, making the driver read past the argument
/// array. The sealed `__LaunchContractDisjointSliceAbi` trait is the semantic
/// authority: only the genuine `DisjointSlice` whose index space really has
/// (`true`) or really lacks (`false`) a runtime row width satisfies the bound, so
/// a spelling/semantics mismatch is a compile error instead of a malformed
/// launch packet.
pub(super) fn add_cuda_module_disjoint_abi_bounds(
    generics: &mut syn::Generics,
    params: &[CudaModuleParam],
) {
    for param in params {
        let (Some(device_ty), Some(element_ty)) =
            (&param.disjoint_slice_ty, &param.disjoint_slice_elem)
        else {
            continue;
        };
        let has_row_width = matches!(
            param.marshal,
            CudaModuleParamMarshal::RowWidthDeviceBuffer { .. }
        );
        let (device_ty, bound_lifetime) = cuda_module_disjoint_bound_type(device_ty);
        generics.make_where_clause().predicates.push(parse_quote! {
            for<#bound_lifetime> #device_ty:
                ::cuda_device::__LaunchContractDisjointSliceAbi<#element_ty, #has_row_width>
        });
    }
}

/// Requires every `Uniform<T>` parameter to be cuda-device's own type.
///
/// The host ABI for these parameters is chosen from the spelling `Uniform<T>`,
/// so without this bound a local type of the same name would be marshalled as
/// a bare `T` while presenting whatever layout it liked.
pub(super) fn add_cuda_module_uniform_bounds(
    generics: &mut syn::Generics,
    params: &[CudaModuleParam],
) {
    for param in params {
        let (Some(device_ty), Some(scalar_ty)) = (&param.uniform_ty, &param.uniform_scalar) else {
            continue;
        };
        generics.make_where_clause().predicates.push(parse_quote! {
            #device_ty: ::cuda_device::__LaunchContractUniform<#scalar_ty>
        });
    }
}

/// Makes the elided `DisjointSlice` lifetime explicit so the complete device
/// type can appear in an impl-level where-clause.
fn cuda_module_disjoint_bound_type(ty: &Type) -> (Type, syn::Lifetime) {
    let mut ty = ty.clone();
    let Type::Path(type_path) = &mut ty else {
        unreachable!("only recognized DisjointSlice paths reach this helper");
    };
    let segment = type_path
        .path
        .segments
        .last_mut()
        .expect("recognized DisjointSlice path has a final segment");
    let PathArguments::AngleBracketed(args) = &mut segment.arguments else {
        unreachable!("recognized DisjointSlice path has generic arguments");
    };
    let bound_ident = internal_ident("__cuda_oxide_disjoint");
    let bound_lifetime = syn::Lifetime::new(&format!("'{}", bound_ident), bound_ident.span());
    if let Some(GenericArgument::Lifetime(lifetime)) = args.args.first_mut() {
        *lifetime = bound_lifetime.clone();
    } else {
        let previous = core::mem::take(&mut args.args);
        args.args
            .push(GenericArgument::Lifetime(bound_lifetime.clone()));
        args.args.extend(previous);
    }
    (ty, bound_lifetime)
}

pub(super) fn cuda_module_kernel_marker_type(kernel: &CudaModuleKernel) -> TokenStream2 {
    let marker = cuda_kernel_marker_name(&kernel.fn_name);
    if !kernel.is_generic {
        return quote! { #marker };
    }
    let marker_args = generic_arguments(&kernel.generics);
    if marker_args.is_empty() {
        quote! { #marker }
    } else {
        quote! { #marker<#(#marker_args),*> }
    }
}
