/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::cuda_module::contract::LaunchContractArgs;
use crate::cuda_module::{
    device_codegen_owner_selection, expand_cuda_module, expand_cuda_module_inner,
    transform_cuda_module_items,
};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use reserved_oxide_symbols::PTX_MERGE_REQUIRED_PREFIX;
use reserved_oxide_symbols::ptx_merge_required_marker;
use syn::{ItemMod, parse_quote};

/// Expands a `#[cuda_module]` body and returns the generated tokens as a
/// whitespace-free string, so tests can assert on call paths without
/// caring how `quote!` spaces out `::` separators.
fn expand_to_compact_string(module: ItemMod) -> String {
    expand_cuda_module(module)
        .expect("cuda_module expansion failed")
        .to_string()
        .replace(' ', "")
}

/// Expands a `#[cuda_module]` body with the host surface suppressed.
fn expand_device_only_to_compact_string(module: ItemMod) -> String {
    expand_cuda_module_inner(module, false)
        .expect("cuda_module expansion failed")
        .to_string()
        .replace(' ', "")
}

fn one_kernel_module() -> ItemMod {
    parse_quote! {
        mod kernels {
            #[kernel]
            fn scale(out: *mut f32) {}
        }
    }
}

#[test]
fn grid_constant_uses_one_by_value_host_argument() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            struct TensorMap([u8; 128]);

            #[kernel]
            fn copy(#[grid_constant] descriptor: &TensorMap, out: *mut u32) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("descriptor:TensorMap,out:*mutu32"),
        "{expanded}"
    );
    assert!(
        expanded.contains("letmut__cuda_oxide_arg_0=descriptor;::cuda_host::push_kernel_scalar"),
        "{expanded}"
    );
    assert!(
        expanded.contains(
            "constcopy_CUDA_SIGNATURE:::cuda_host::CudaKernelSignature="
        ) && expanded.contains(
            "kind:::cuda_host::CudaKernelArgumentKind::GridConstant{size:::core::mem::size_of::<TensorMap>(),alignment:::core::mem::align_of::<TensorMap>()"
        ) && expanded.contains(
            "name:\"out\",kind:::cuda_host::CudaKernelArgumentKind::DevicePointer"
        ),
        "{expanded}"
    );
}

#[test]
fn concrete_kernel_signature_is_generated_from_launcher_arguments() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn classify(flag: ::core::primitive::bool, count: ::core::primitive::u32, data: &[f32], out: *mut f32) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("pubconstclassify_CUDA_SIGNATURE:::cuda_host::CudaKernelSignature")
            && expanded.contains(
                "name:\"flag\",kind:::cuda_host::CudaKernelArgumentKind::Scalar(::cuda_host::CudaKernelScalarKind::Bool)"
            )
            && expanded.contains(
                "name:\"count\",kind:::cuda_host::CudaKernelArgumentKind::Scalar(::cuda_host::CudaKernelScalarKind::U32)"
            )
            && expanded.contains(
                "name:\"data\",kind:::cuda_host::CudaKernelArgumentKind::DeviceSlice{writable:false,row_width:false"
            )
            && expanded.contains(
                "name:\"out\",kind:::cuda_host::CudaKernelArgumentKind::DevicePointer"
            ),
        "{expanded}"
    );
}

#[test]
fn grid_constant_launch_safety_is_independent_of_source_safety_and_geometry() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(domain = 1, block = (1, 1, 1))]
            pub fn grid(#[grid_constant] value: &Payload) {}

            #[kernel]
            #[launch_contract(domain = 1, block = (1, 1, 1))]
            pub fn generic<T: Copy>(#[grid_constant] value: &T) {}

            #[kernel]
            #[launch_contract(domain = 1, block = (1, 1, 1))]
            pub fn ordinary(value: Payload) {}
        }
    };
    let expanded: ItemMod = syn::parse2(expand_cuda_module(module).unwrap()).unwrap();
    let items = &expanded.content.as_ref().unwrap().1;
    // The device Rust signatures and their fingerprint inputs remain safe.
    for name in ["grid", "generic", "ordinary"] {
        let source = items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(function) if function.sig.ident == name => Some(function),
                _ => None,
            })
            .expect("source kernel is preserved");
        assert!(source.sig.unsafety.is_none(), "{name}");
    }
    let mut checked = 0;
    for item in items {
        let syn::Item::Impl(implementation) = item else {
            continue;
        };
        for item in &implementation.items {
            let syn::ImplItem::Fn(method) = item else {
                continue;
            };
            let name = method.sig.ident.to_string();
            if ["grid", "generic"]
                .iter()
                .any(|prefix| name == *prefix || name.starts_with(&format!("{prefix}_")))
            {
                assert!(
                    method.sig.unsafety.is_some(),
                    "{name} must require a host proof"
                );
                let docs = method
                    .attrs
                    .iter()
                    .filter_map(|attribute| {
                        let syn::Meta::NameValue(value) = &attribute.meta else {
                            return None;
                        };
                        if !value.path.is_ident("doc") {
                            return None;
                        }
                        let syn::Expr::Lit(value) = &value.value else {
                            return None;
                        };
                        let syn::Lit::Str(value) = &value.lit else {
                            return None;
                        };
                        Some(value.value())
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                assert!(docs.contains("# Safety"), "{name}: {docs}");
                assert!(
                    docs.contains("until the device work completes"),
                    "{name}: {docs}"
                );
                assert!(
                    docs.contains("allocations referenced by their fields"),
                    "{name}: {docs}"
                );
                checked += 1;
            } else if matches!(
                name.as_str(),
                "ordinary" | "ordinary_async" | "ordinary_async_owned"
            ) {
                assert!(
                    method.sig.unsafety.is_none(),
                    "ordinary scalar policy is unchanged"
                );
            }
        }
    }
    assert_eq!(checked, if cfg!(feature = "async") { 12 } else { 4 });
}

/// With the host surface off, nothing in the expansion may name the
/// `cuda-host` -> `cuda-core` -> `cuda-bindings` -> `cuda.h` stack. That is
/// the whole point of the feature: a crate that only compiles kernels
/// should not have to build it.
#[test]
fn device_only_cuda_module_names_no_host_crate() {
    let expanded = expand_device_only_to_compact_string(one_kernel_module());
    assert!(
        !expanded.contains("cuda_host"),
        "device-only expansion must not name cuda_host: {expanded}"
    );
    assert!(
        !expanded.contains("cuda_core"),
        "device-only expansion must not name cuda_core: {expanded}"
    );
    assert!(
        !expanded.contains("LoadedModule"),
        "device-only expansion must not emit the loader type: {expanded}"
    );
}

/// Gating must remove only the host surface. The kernel itself still has
/// to reach the codegen collector.
#[test]
fn device_only_cuda_module_still_emits_the_kernel() {
    let expanded = expand_device_only_to_compact_string(one_kernel_module());
    assert!(
        expanded.contains("#[kernel]fnscale"),
        "the kernel must survive gating: {expanded}"
    );
}

/// The default build is unchanged: this is the additive half of the
/// contract, and existing consumers depend on it.
#[test]
fn host_cuda_module_still_emits_the_loader() {
    let expanded = expand_to_compact_string(one_kernel_module());
    assert!(
        expanded.contains("::cuda_host::") && expanded.contains("LoadedModule"),
        "host expansion must keep the loader: {expanded}"
    );
}

/// A nested inline module gets its own `LoadedModule`, so it needs the
/// same gate as the outer one.
#[test]
fn device_only_nested_module_emits_no_loader() {
    let module: ItemMod = parse_quote! {
        mod outer {
            mod inner {
                #[kernel]
                fn scale(out: *mut f32) {}
            }
        }
    };
    let expanded = expand_device_only_to_compact_string(module);
    assert!(
        !expanded.contains("LoadedModule") && !expanded.contains("cuda_host"),
        "nested device-only expansion must emit no loader: {expanded}"
    );
}

#[test]
fn device_codegen_owner_selection_matches_backend_name_rules() {
    assert_eq!(device_codegen_owner_selection(None, "gpu_lib"), None);
    assert_eq!(device_codegen_owner_selection(Some(" , "), "gpu_lib"), None);
    assert_eq!(
        device_codegen_owner_selection(Some("gpu-lib, math_gpu"), "gpu_lib"),
        Some(true)
    );
    assert_eq!(
        device_codegen_owner_selection(Some("gpu-lib, math_gpu"), "host_app"),
        Some(false)
    );
}

#[test]
fn cooperative_kernel_launches_through_cooperative_driver_call() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[cooperative_launch]
            pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    // The sync launch method must route through the cooperative driver
    // entry point (cuLaunchKernelEx + CU_LAUNCH_ATTRIBUTE_COOPERATIVE)
    // instead of plain cuLaunchKernel.
    assert!(
        expanded.contains("launch_kernel_cooperative_on_stream"),
        "expected cooperative launch call in generated tokens:\n{expanded}"
    );
    assert!(
        !expanded.contains("launch_kernel_on_stream"),
        "plain launch call should be replaced by the cooperative one:\n{expanded}"
    );
}

#[test]
fn plain_kernel_keeps_plain_driver_call() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn plain_kernel(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("launch_kernel_on_stream"),
        "expected plain launch call in generated tokens:\n{expanded}"
    );
    assert!(
        !expanded.contains("launch_kernel_cooperative_on_stream"),
        "cooperative call must not appear without #[cooperative_launch]:\n{expanded}"
    );
}

#[test]
fn cuda_module_routes_type_and_const_generics_through_name_helper() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn mixed<'a, T: Copy + 'a, const N: usize>(
                input: &'a [T],
                output: &mut [T],
            ) {
            }
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("mixed_ptx_name::<T,N>()"),
        "typed launch must forward type and const arguments to the name helper:\n{expanded}"
    );
    assert!(
        expanded.contains(PTX_MERGE_REQUIRED_PREFIX),
        "generic cuda_module must emit the compiler-visible PTX-merge marker:\n{expanded}"
    );
}

#[test]
fn non_generic_cuda_module_does_not_emit_ptx_merge_marker() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn plain(output: &mut [u32]) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        !expanded.contains(PTX_MERGE_REQUIRED_PREFIX),
        "non-generic cuda_module must remain eligible for cubin materialization:\n{expanded}"
    );
}

#[test]
fn cfg_gated_generic_uses_the_same_gate_for_marker_and_loader() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn plain(output: &mut [u32]) {}

            #[cfg(feature = "generic")]
            #[kernel]
            pub fn optional<T: Copy>(value: T) {}
        }
    };
    let expanded = expand_to_compact_string(module);
    let marker = ptx_merge_required_marker("optional");

    assert!(
            expanded.contains(&format!(
                "#[cfg(feature=\"generic\")]#[doc(hidden)]#[used]#[allow(dead_code,non_upper_case_globals)]static{marker}:u8=0"
            )),
            "marker must inherit the generic kernel's cfg:\n{expanded}"
        );
    assert!(
        expanded.contains(
            "#[cfg(feature=\"generic\")]let_={__cuda_oxide_has_enabled_generic_kernel=true;};"
        ),
        "loader selection must inherit the generic kernel's cfg:\n{expanded}"
    );
    assert!(
            expanded.contains(&format!("if__cuda_oxide_has_enabled_generic_kernel{{let_=name;::cuda_host::load_all_ptx_bundles_merged(ctx)?}}else{{{ANCHORED_EMBEDDED_LOADER}}}")),
            "loader must fall back to the embedded artifact when no generic kernel is enabled:\n{expanded}"
        );
}

/// The embedded-artifact loader: read the bundle from the binary that maps
/// the artifact anchor, or from the executable when no anchor was referenced.
const ANCHORED_EMBEDDED_LOADER: &str = concat!(
    "match__cuda_oxide_artifact_anchor{::core::option::Option::Some(anchor)=>{",
    "#[cfg(any(target_os=\"linux\",target_os=\"android\"))]",
    "{::cuda_host::load_embedded_module_from_anchor(ctx,name,anchor)?}",
    "#[cfg(not(any(target_os=\"linux\",target_os=\"android\")))]",
    "{let_=anchor;::cuda_host::load_embedded_module(ctx,name)?}",
    "}::core::option::Option::None=>::cuda_host::load_embedded_module(ctx,name)?,}",
);

#[test]
fn generated_loader_preserves_other_platform_executable_lookup() {
    let module: ItemMod = parse_quote! {
        mod kernels { #[kernel] pub fn plain() {} }
    };
    let expanded = expand_to_compact_string(module);
    assert!(expanded.contains(
        "#[cfg(not(any(target_os=\"linux\",target_os=\"android\")))]{let_=anchor;::cuda_host::load_embedded_module(ctx,name)?}"
    ), "{expanded}");
    assert!(expanded.contains(
        "#[cfg(any(target_os=\"linux\",target_os=\"android\"))]{::cuda_host::load_embedded_module_from_anchor(ctx,name,anchor)?}"
    ), "{expanded}");
}

#[test]
fn non_generic_cuda_module_loads_from_the_binary_that_maps_its_anchor() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn plain(output: &mut [u32]) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("#[allow(unused_mut)]letmut__cuda_oxide_artifact_anchor:::core::option::Option<&::core::primitive::u8>=::core::option::Option::None;"),
        "load_named must start without an anchor address:\n{expanded}"
    );
    assert!(
        expanded.contains(&format!("letmodule={ANCHORED_EMBEDDED_LOADER};")),
        "loader must prefer the binary that maps the anchor:\n{expanded}"
    );
}

#[test]
fn nested_generic_marker_inherits_every_ancestor_cfg() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[cfg(feature = "outer")]
            mod nested {
                #[cfg(target_os = "linux")]
                #[kernel]
                pub fn map<T: Copy>(value: T) {}
            }
        }
    };
    let expanded = expand_to_compact_string(module);
    let marker = ptx_merge_required_marker("map");
    assert!(
            expanded.contains(&format!(
                "#[cfg(feature=\"outer\")]#[cfg(target_os=\"linux\")]#[doc(hidden)]#[used]#[allow(dead_code,non_upper_case_globals)]static{marker}:u8=0"
            )),
            "root marker must use the nested kernel's effective cfg chain:\n{expanded}"
        );
}

#[test]
fn cooperative_plus_cluster_kernel_uses_combined_driver_call() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[cluster_launch(2, 1, 1)]
            #[cooperative_launch]
            pub fn clustered_grid_sync_kernel(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    // cuLaunchKernelEx accepts both attributes in one attrs array, so the
    // combination is allowed and routes through the combined helper.
    assert!(
        expanded.contains("launch_kernel_ex_cooperative_on_stream"),
        "expected combined cluster+cooperative launch call:\n{expanded}"
    );
    assert!(
        !expanded.contains("launch_kernel_ex_on_stream"),
        "cluster-only call should be replaced by the combined one:\n{expanded}"
    );
}

#[cfg(feature = "async")]
#[test]
fn cooperative_kernel_sets_async_builder_knob() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[cooperative_launch]
            pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    // Both the borrowed-async and owned-async builder methods set the
    // cooperative knob, exactly like set_async_kernel_cluster_dim is set
    // for #[cluster_launch].
    assert_eq!(
        expanded.matches("set_async_kernel_cooperative").count(),
        2,
        "expected the cooperative knob in both async builder methods:\n{expanded}"
    );
}

#[test]
fn cooperative_launch_with_arguments_is_rejected() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[cooperative_launch(4)]
            pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {}
        }
    };
    let error = expand_cuda_module(module).expect_err("expected expansion to fail");
    assert!(
        error
            .to_string()
            .contains("cooperative_launch takes no arguments"),
        "unexpected error message: {error}"
    );
}

#[test]
fn nested_inline_module_kernels_generate_namespace_local_launchers() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn top(mut out: DisjointSlice<u32>) {}

            pub mod stage1 {
                #[kernel]
                pub fn scale(mut out: DisjointSlice<f32>) {}
            }

            pub mod stage2 {
                pub mod inner {
                    #[kernel]
                    pub fn shift(mut out: DisjointSlice<f32>) {}
                }
            }
        }
    };
    let expanded = expand_to_compact_string(module);
    assert!(
        expanded.contains("pubmodstage1{#[kernel]pubfnscale")
            && expanded.contains("<__scale_CudaKernelas::cuda_host::CudaKernel>"),
        "expected the stage1 launcher to resolve its marker locally:\n{expanded}"
    );
    assert!(
        expanded.contains("pubmodinner{#[kernel]pubfnshift")
            && expanded.contains("<__shift_CudaKernelas::cuda_host::CudaKernel>"),
        "expected the doubly nested launcher to resolve its marker locally:\n{expanded}"
    );
    assert!(
        expanded.matches("pubstructLoadedModule").count() == 4,
        "expected root, stage1, stage2, and inner module views:\n{expanded}"
    );
    assert!(
        expanded.contains("from_parent(parent:&super::LoadedModule")
            && expanded.contains("parent.__generic_functions.clone()"),
        "expected nested views to share the parent module and cache:\n{expanded}"
    );
}

#[test]
fn nested_launch_contract_stays_in_its_namespace_and_taints_root_loading() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            pub mod stage {
                #[kernel]
                #[launch_contract(domain = 1, block = (64, 1, 1))]
                pub fn map(mut out: DisjointSlice<u32>) {}
            }
        }
    };
    let expanded = expand_to_compact_string(module);

    assert_eq!(
        expanded
            .matches("impl::cuda_core::KernelLaunchContract")
            .count(),
        1,
        "the contract impl must be emitted once in the child namespace:\n{expanded}"
    );
    assert_eq!(
        expanded.matches("fnprepare_map(").count(),
        1,
        "the prepared launcher must be emitted once in the child namespace:\n{expanded}"
    );
    assert!(
        expanded.contains("pubunsafefnload("),
        "a descendant contract must make root artifact binding unsafe:\n{expanded}"
    );
    assert!(
        expanded.contains("pubmodstage{")
            && expanded.contains("from_parent(parent:&super::LoadedModule"),
        "main's nested module view must remain intact:\n{expanded}"
    );
}

#[test]
fn nested_generic_kernel_uses_local_ptx_name_helper() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn top(mut out: DisjointSlice<u32>) {}

            pub mod stage1 {
                #[kernel]
                pub fn map<F: Fn(f32) -> f32 + Copy>(f: F, mut out: DisjointSlice<f32>) {}
            }
        }
    };
    let expanded = expand_to_compact_string(module);
    assert!(
        expanded.contains("let__cuda_oxide_ptx_name=map_ptx_name::<F>()"),
        "expected the generic binding to call the namespace-local ptx-name helper:\n{expanded}"
    );
    assert!(
        !expanded.contains("stage1::map_ptx_name"),
        "the nested launcher must not resolve its private helper from the parent:\n{expanded}"
    );
}

#[test]
fn nested_kernel_tracks_local_and_effective_cfg_availability() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[cfg(feature = "outer")]
            mod outer {
                #[cfg_attr(feature = "inner", cfg(target_os = "linux"), allow(dead_code))]
                mod inner {
                    #[cfg(target_arch = "x86_64")]
                    #[kernel]
                    fn nested() {}
                }
            }
        }
    };
    let items = &module.content.expect("inline module").1;
    let transformed =
        transform_cuda_module_items(items, &mut Vec::new(), &[], false, true).unwrap();
    let kernel = transformed
        .kernels
        .iter()
        .find(|kernel| kernel.fn_name == "nested")
        .expect("nested kernel was not collected");
    let local_attrs = &kernel.cfg_attrs;
    let effective_attrs = &kernel.effective_cfg_attrs;
    let local = quote!(#(#local_attrs)*).to_string().replace(' ', "");
    let effective = quote!(#(#effective_attrs)*).to_string().replace(' ', "");

    assert_eq!(local, "#[cfg(target_arch=\"x86_64\")]");
    assert!(effective.contains("#[cfg(feature=\"outer\")]"));
    assert!(effective.contains("#[cfg_attr(feature=\"inner\",cfg(target_os=\"linux\"))]"));
    assert!(!effective.contains("allow(dead_code)"));
    assert!(effective.ends_with("#[cfg(target_arch=\"x86_64\")]"));
}

#[test]
fn conflicting_kernel_names_across_modules_are_rejected() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            pub mod stage1 {
                #[kernel]
                pub fn step(mut out: DisjointSlice<f32>) {}
            }

            pub mod stage2 {
                #[kernel]
                pub fn step(mut out: DisjointSlice<f32>) {}
            }
        }
    };
    let error = expand_cuda_module(module).expect_err("expected expansion to fail");
    let message = error.to_string();
    assert!(
        message.contains("requires kernel names to be unique")
            && message.contains("stage1")
            && message.contains("stage2"),
        "unexpected error message: {message}"
    );
}

#[test]
fn launch_contract_generates_prepared_and_unsafe_raw_paths() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared = 1024,
                dynamic_shared_alignment = 128,
                min_compute_capability = (8, 0),
            )]
            pub fn map(input: &[u32], mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("impl::cuda_core::KernelLaunchContractfor__map_CudaKernel"));
    assert!(expanded.contains("typeConfig=::cuda_core::LaunchConfig1D"));
    assert!(expanded.contains("fnprepare_map("));
    assert!(expanded.contains("PreparedLaunch<__map_CudaKernel>"));
    assert!(expanded.contains("fnmap("));
    assert!(
        !expanded.contains("unsafefnmap("),
        "a safe source kernel must keep its prepared launch method safe: {expanded}"
    );
    assert!(
        expanded.contains("__cuda_oxide_prepared:&::cuda_core::PreparedLaunch<__map_CudaKernel>")
    );
    assert!(expanded.contains("unsafefnmap_unchecked("));
    assert!(expanded.contains("__cuda_oxide_config:::cuda_core::simt::LaunchConfig"));
    assert!(expanded.contains("min_alignment:128u32"));
    assert!(expanded.contains("with_min_compute_capability(8u32,0u32)"));
    assert!(expanded.contains("with_u32_coordinates()"));
}

#[test]
fn uncontracted_sync_launch_is_always_unsafe() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn map(input: &[u32], mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("pubunsafefnmap("));
    assert!(expanded.contains("#Safety"));
    assert!(expanded.contains("unverifiedrawlaunchconfiguration"));
}

#[test]
fn unsafe_source_kernel_keeps_prepared_launch_unsafe() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(domain = 1, block = (64, 1, 1))]
            pub unsafe fn map(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("pubunsafefnmap("));
    assert!(expanded.contains("pubunsafefnmap_unchecked("));
    #[cfg(feature = "async")]
    {
        assert!(expanded.contains("pubunsafefnmap_async"));
        assert!(expanded.contains("pubunsafefnmap_async_owned"));
    }
}

#[test]
fn contracted_module_requires_provenance_for_every_loader() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(domain = 1, block = (64, 1, 1))]
            pub fn map(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("pubunsafefnload("));
    assert!(expanded.contains("pubunsafefnload_named("));
    assert!(expanded.contains("pubunsafefnfrom_module("));
    assert!(expanded.contains("#Safety"));
    #[cfg(feature = "async")]
    {
        assert!(expanded.contains("pubunsafefnload_async("));
        assert!(expanded.contains("pubunsafefnload_async_named("));
    }
}

#[test]
fn uncontracted_module_preserves_safe_custom_loaders() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn map(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("pubfnload_named("));
    assert!(expanded.contains("pubfnfrom_module("));
    assert!(!expanded.contains("pubunsafefnload_named("));
    assert!(!expanded.contains("pubunsafefnfrom_module("));
    #[cfg(feature = "async")]
    assert!(expanded.contains("pubfnload_async_named("));
}

#[test]
fn generic_contract_brand_and_prepare_witness_keep_specialization_type() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, dynamic_shared = 0)]
            pub fn apply<F: Fn(u32) -> u32 + Copy>(op: F, out: *mut u32) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("KernelLaunchContractfor__apply_CudaKernel<F>"));
    assert!(expanded.contains("PreparedLaunch<__apply_CudaKernel<F>>"));
    assert!(expanded.contains("fnprepare_apply_for<F"));
    assert!(expanded.contains("__cuda_oxide_type_witness_0:&F"));
    assert!(
        expanded.contains("let__cuda_oxide_max_threads:u32=128"),
        "{expanded}"
    );
    assert!(expanded.contains("BlockRequirement::MaxThreads(__cuda_oxide_max_threads)"));
}

/// The `requires` demo module used by the size-requirement tests: two
/// relations, one with arithmetic, over a scalar pair and two buffers.
fn requires_demo_module() -> ItemMod {
    parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(
                domain = 1,
                block = (128, 1, 1),
                requires = (input.len() >= n * stride, output.len() >= n),
            )]
            pub fn scaled_copy(
                n: u32,
                stride: u32,
                input: &[f32],
                mut output: DisjointSlice<f32>,
            ) {
            }
        }
    }
}

#[test]
fn requires_relations_generate_overflow_safe_checks_in_checked_launchers_only() {
    let expanded = expand_to_compact_string(requires_demo_module());

    // Every operand is widened to u64 and arithmetic goes through
    // checked ops with a typed overflow error.
    assert!(expanded.contains("SizeRequirementViolated"), "{expanded}");
    assert!(expanded.contains("SizeRequirementOverflow"), "{expanded}");
    assert!(expanded.contains("checked_mul"), "{expanded}");
    assert!(expanded.contains("(nasu64)"), "{expanded}");
    // The relation's source text rides along for the error message.
    // (`expand_to_compact_string` strips spaces inside string literals
    // too, so the expected text is compacted.)
    assert!(expanded.contains("\"input.len()>=n*stride\""), "{expanded}");
    assert!(expanded.contains("\"output.len()>=n\""), "{expanded}");

    // Each checked launcher evaluates both relations; the `_unchecked`
    // escape hatches never do. Without the async feature only the sync
    // prepared launcher exists (2 relations); with it, the `_async` and
    // `_async_owned` twins check too (2 relations each).
    #[cfg(not(feature = "async"))]
    assert_eq!(
        expanded.matches("SizeRequirementViolated").count(),
        2,
        "{expanded}"
    );
    #[cfg(feature = "async")]
    assert_eq!(
        expanded.matches("SizeRequirementViolated").count(),
        6,
        "{expanded}"
    );
}

#[cfg(feature = "async")]
#[test]
fn requires_relations_wrap_async_launchers_in_result() {
    let expanded = expand_to_compact_string(requires_demo_module());

    // The async launchers report a violated relation as a typed error
    // instead of enqueueing, so their return types gain a Result.
    assert!(
        expanded.contains("::core::result::Result<::cuda_host::PreparedAsyncKernelLaunch<"),
        "{expanded}"
    );
    assert!(
        expanded.contains("::core::result::Result<::cuda_host::PreparedOwnedAsyncKernelLaunch<"),
        "{expanded}"
    );
    // Slice lengths come from the KernelSliceArg trait: by reference for
    // the borrowed async launcher, by value for the owned one.
    assert!(
        expanded.contains("KernelSliceArg::len(input)"),
        "{expanded}"
    );
    assert!(
        expanded.contains("KernelSliceArg::len(&input)"),
        "{expanded}"
    );

    // A contract without `requires` keeps the plain infallible async
    // signatures.
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(domain = 1, block = (128, 1, 1))]
            pub fn plain(input: &[f32], mut output: DisjointSlice<f32>) {}
        }
    };
    let plain = expand_to_compact_string(module);
    assert!(
        !plain.contains("Result<::cuda_host::PreparedAsyncKernelLaunch<"),
        "{plain}"
    );
    assert!(
        !plain.contains("Result<::cuda_host::PreparedOwnedAsyncKernelLaunch<"),
        "{plain}"
    );
}

#[test]
fn requires_rejects_relations_outside_the_v1_grammar() {
    let expand_with_requires = |requires: &str| {
        let attr = format!("domain = 1, block = (64, 1, 1), requires = ({requires})");
        let attr: TokenStream2 = attr.parse().expect("attr tokens");
        let module: ItemMod = parse_quote! {
            mod kernels {
                #[kernel]
                #[launch_contract(#attr)]
                pub fn bad(n: u32, input: &[f32], mut output: DisjointSlice<f32>) {}
            }
        };
        expand_cuda_module(module)
    };
    let reject = |requires: &str, expected: &str| {
        let error = expand_with_requires(requires)
            .expect_err(&format!("`{requires}` should be rejected"))
            .to_string();
        assert!(
            error.contains(expected),
            "`{requires}` produced unexpected error: {error}"
        );
    };

    reject("input.len() >= 1.5", "only integer literals");
    reject("input.len() >= 1 && output.len() >= 1", "must compare with");
    reject("input.len() + n", "must compare with");
    reject("input", "must be a comparison");
    reject("input.len() as u64 >= 1", "unsupported expression");
    reject("input.capacity() >= n", "only the `.len()` method");
    reject(
        "input.len() >= (n >= 1)",
        "nested comparisons are not supported",
    );
    reject("input.len() >= self::n", "bare kernel parameter names");

    // The full accepted grammar expands cleanly.
    expand_with_requires("(input.len() - 1) * 2 + 0 >= n * 3")
        .expect("grammar-conformant relation must expand");
}

#[test]
fn requires_parser_accepts_relation_lists_and_rejects_degenerate_forms() {
    let args: LaunchContractArgs =
        syn::parse_str("domain = 1, requires = (a.len() >= n, b.len() >= n * 2)").unwrap();
    assert_eq!(args.requires.len(), 2);

    // `.err()` instead of `.unwrap_err()`: LaunchContractArgs holds
    // syn::Expr, which has no Debug without syn's extra-traits feature.
    let empty = syn::parse_str::<LaunchContractArgs>("domain = 1, requires = ()")
        .err()
        .expect("empty requires list must be rejected");
    assert!(empty.to_string().contains("at least one relation"));

    let duplicate = syn::parse_str::<LaunchContractArgs>(
        "domain = 1, requires = (a.len() >= 1), requires = (b.len() >= 1)",
    )
    .err()
    .expect("duplicate requires key must be rejected");
    assert!(
        duplicate
            .to_string()
            .contains("duplicate launch_contract field")
    );
}

#[test]
fn cfg_gated_duplicate_kernel_names_are_rejected_until_ptx_names_are_qualified() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[cfg(feature = "fast")]
            pub fn step(mut out: DisjointSlice<f32>) {}

            #[kernel]
            #[cfg(not(feature = "fast"))]
            pub fn step(mut out: DisjointSlice<f32>) {}
        }
    };
    let error = expand_cuda_module(module).expect_err("bare PTX names still collide");
    assert!(
        error
            .to_string()
            .contains("PTX entry names are currently bare"),
        "unexpected error message: {error}"
    );
}

#[test]
fn contract_without_block_or_launch_bounds_is_rejected() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(domain = 1, dynamic_shared = 0)]
            pub fn missing_block(input: &[u32]) {}
        }
    };
    let error = expand_cuda_module(module).expect_err("contract should fail closed");
    assert!(
        error
            .to_string()
            .contains("requires either an exact `block = (x, y, z)` or #[launch_bounds")
    );
}

#[test]
fn file_backed_modules_and_include_macros_are_preserved_without_being_walked() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn top() {}

            mod helper;
            include!("helper_items.rs");
        }
    };
    let expanded = expand_to_compact_string(module);
    assert!(
        expanded.contains("modhelper;"),
        "file module was changed: {expanded}"
    );
    assert!(
        expanded.contains("include!(\"helper_items.rs\");"),
        "include invocation was changed: {expanded}"
    );
}

#[test]
fn loaded_module_is_reserved_in_nested_kernel_namespaces() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn top() {}

            mod child {
                struct LoadedModule;

                #[kernel]
                pub fn nested() {}
            }
        }
    };
    let error = expand_cuda_module(module).expect_err("reserved type name must fail clearly");
    assert!(
        error
            .to_string()
            .contains("reserves the name `LoadedModule`"),
        "unexpected error message: {error}"
    );
}

#[test]
fn raw_and_plain_kernel_spellings_share_one_conflict_key() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            mod plain {
                #[kernel]
                fn step() {}
            }

            mod raw {
                #[kernel]
                fn r#step() {}
            }
        }
    };
    let error = expand_cuda_module(module).expect_err("raw prefix must not evade PTX guard");
    let message = error.to_string();
    assert!(
        message.contains("`step`") && message.contains("plain") && message.contains("raw"),
        "unexpected error message: {message}"
    );
}

#[test]
fn raw_loaded_module_spelling_is_reserved() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            mod child {
                struct r#LoadedModule;

                #[kernel]
                fn nested() {}
            }
        }
    };
    let error = expand_cuda_module(module).expect_err("raw prefix must not evade reservation");
    assert!(
        error
            .to_string()
            .contains("reserves the name `LoadedModule`"),
        "unexpected error message: {error}"
    );
}

#[test]
fn renamed_extern_crate_cannot_claim_loaded_module() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            extern crate core as r#LoadedModule;

            #[kernel]
            fn root() {}
        }
    };
    let error = expand_cuda_module(module).expect_err("extern-crate rename must be checked");
    assert!(
        error
            .to_string()
            .contains("reserves the name `LoadedModule`"),
        "unexpected error message: {error}"
    );
}

#[test]
fn generated_loaded_module_method_names_are_reserved() {
    let root: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            fn as_cuda_module() {}
        }
    };
    let error = expand_cuda_module(root).expect_err("root accessor name must be reserved");
    assert!(
        error.to_string().contains("method name `as_cuda_module`"),
        "unexpected error message: {error}"
    );

    let nested: ItemMod = parse_quote! {
        mod kernels {
            mod child {
                #[kernel]
                fn from_parent() {}
            }
        }
    };
    let error = expand_cuda_module(nested).expect_err("nested constructor name must be reserved");
    assert!(
        error.to_string().contains("method name `from_parent`"),
        "unexpected error message: {error}"
    );
}

#[test]
fn contracted_mut_slice_is_rejected_in_favor_of_disjoint_slice() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(domain = 1, block = (64, 1, 1))]
            pub fn aliased(out: &mut [u32]) {}
        }
    };
    let error = expand_cuda_module(module).expect_err("mutable slice must fail closed");
    assert!(
        error
            .to_string()
            .contains("contracted kernels cannot take `&mut [T]`")
    );
}

#[test]
fn disjoint_slice_packet_shape_is_bound_to_the_resolved_type() {
    // The spelling `Rt` hides a runtime row width, so the macro selects
    // the two-word host ABI. The generated launch methods must carry the
    // semantic `HAS_ROW_WIDTH = false` bound for Rust to reject at the call
    // site once `Rt` resolves to a runtime-width index space.
    let module: ItemMod = parse_quote! {
        mod kernels {
            type Rt = RuntimeRowMajorTiles<1, 1>;

            #[kernel]
            pub fn alias_hides_row_width(mut out: DisjointSlice<f32, Rt>) {}
        }
    };
    let expanded = expand_to_compact_string(module);
    assert!(
        expanded.contains(
            "for<'__cuda_oxide_disjoint>DisjointSlice<'__cuda_oxide_disjoint,f32,Rt>:\
                 ::cuda_device::__LaunchContractDisjointSliceAbi<f32,false>"
        ),
        "flat spelling must bind HAS_ROW_WIDTH = false: {expanded}"
    );

    // The direct spelling selects the three-word ABI and must bind
    // `HAS_ROW_WIDTH = true`.
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn really_has_row_width(mut out: DisjointSlice<f32, Runtime2DIndex>) {}
        }
    };
    let expanded = expand_to_compact_string(module);
    assert!(
        expanded.contains(
            "for<'__cuda_oxide_disjoint>DisjointSlice<'__cuda_oxide_disjoint,f32,\
                 Runtime2DIndex>:::cuda_device::__LaunchContractDisjointSliceAbi<f32,true>"
        ),
        "runtime-width spelling must bind HAS_ROW_WIDTH = true: {expanded}"
    );
}

#[test]
fn aliased_disjoint_index_space_is_checked_by_rust_type_resolution() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            type Alias = Index2D<128>;

            #[kernel]
            #[launch_contract(domain = 2, block = (16, 16, 1))]
            pub fn aliased(mut out: DisjointSlice<u32, Alias>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains(
            "for<'__cuda_oxide_disjoint>DisjointSlice<'__cuda_oxide_disjoint,u32,Alias>:"
        ),
        "the bound must preserve the original index-space alias: {expanded}"
    );
    assert!(expanded.contains("__LaunchContractDisjointSlice<u32,2u8>"));
}

#[test]
fn one_dimensional_index_space_is_not_accepted_by_identifier_spelling() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            type Index2D = Index1D;

            #[kernel]
            #[launch_contract(domain = 2, block = (16, 16, 1))]
            pub fn wrong_rank(mut out: DisjointSlice<u32, Index2D>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains(
            "for<'__cuda_oxide_disjoint>DisjointSlice<'__cuda_oxide_disjoint,u32,Index2D>:"
        ),
        "the bound must preserve the misleading alias for Rust to resolve: {expanded}"
    );
    assert!(expanded.contains("__LaunchContractDisjointSlice<u32,2u8>"));
}

#[test]
fn local_disjoint_slice_lookalike_gets_the_genuine_type_bound() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            struct DisjointSlice<'a, T, IndexSpace = Index1D> {
                value: &'a mut T,
                index: PhantomData<IndexSpace>,
            }

            #[kernel]
            #[launch_contract(domain = 1, block = (64, 1, 1))]
            pub fn fake(mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("for<'__cuda_oxide_disjoint>DisjointSlice<'__cuda_oxide_disjoint,u32>:"),
        "the look-alike must receive the genuine cuda-device trait bound: {expanded}"
    );
    assert!(expanded.contains("__LaunchContractDisjointSlice<u32,1u8>"));
}

#[test]
fn launch_bounds_accepts_a_two_dimensional_block_with_the_same_product() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(256)]
            #[launch_contract(domain = 2, block = (16, 16, 1))]
            pub fn tiled(input: &[u32]) {}
        }
    };
    let expanded = expand_to_compact_string(module);
    assert!(expanded.contains("BlockRequirement::Exact((16u32,16u32,1u32))"));
}

#[test]
fn exact_block_cannot_exceed_compiled_launch_bounds_thread_count() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(256)]
            #[launch_contract(domain = 2, block = (17, 16, 1))]
            pub fn impossible(input: &[u32]) {}
        }
    };
    let error = expand_cuda_module(module).expect_err("bounds mismatch must fail closed");
    assert!(error.to_string().contains("272 threads"));
    assert!(error.to_string().contains("launch_bounds(256)"));
}

#[cfg(feature = "async")]
#[test]
fn contracted_async_methods_return_immutable_wrappers() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_contract(domain = 1, block = (64, 1, 1))]
            pub fn map(input: &[u32], mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("PreparedAsyncKernelLaunch<'__cuda_oxide_async,__map_CudaKernel>"));
    assert!(expanded.contains("PreparedOwnedAsyncKernelLaunch<"));
    assert!(expanded.contains("fnmap_async_unchecked"));
    assert!(expanded.contains("fnmap_async_owned_unchecked"));
    assert!(expanded.matches("unsafe").count() >= 4);
}

#[cfg(feature = "async")]
#[test]
fn uncontracted_async_launch_methods_are_always_unsafe() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn map(input: &[u32], mut out: DisjointSlice<u32>) {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("pubunsafefnmap_async"));
    assert!(expanded.contains("pubunsafefnmap_async_owned"));
    assert_eq!(
        expanded.matches("new_async_kernel_launch_builder").count(),
        2
    );
    assert_eq!(expanded.matches("finalize_unchecked").count(), 2);
}

#[test]
fn cuda_module_host_methods_keep_policy_expression_bounds() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
            pub fn configured<P: Policy>() {
                let mut i = 0u32;
                #[unroll(P::UNROLL)]
                while i < 8 { i += 1; }
            }
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("[();(P::MAX_THREADS)asusize]:"));
    assert!(expanded.contains("[();(P::MIN_BLOCKS)asusize]:"));
    assert!(expanded.contains("[();(P::UNROLL)asusize]:"));
    assert!(expanded.contains("pubfnconfigured<P:Policy>"));
}

#[test]
fn nested_cuda_module_host_methods_keep_policy_expression_bounds() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            pub mod stage {
                use super::*;

                #[kernel]
                #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
                pub fn configured<P: Policy>() {
                    let mut i = 0u32;
                    #[unroll(P::UNROLL)]
                    while i < 8 { i += 1; }
                }
            }
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(expanded.contains("pubmodstage"));
    assert!(expanded.contains("from_parent(parent:&super::LoadedModule"));
    assert!(expanded.contains("[();(P::MAX_THREADS)asusize]:"));
    assert!(expanded.contains("[();(P::MIN_BLOCKS)asusize]:"));
    assert!(expanded.contains("[();(P::UNROLL)asusize]:"));
    assert!(expanded.contains("pubfnconfigured<P:Policy>"));
}

#[test]
fn launch_contract_carries_deferred_launch_bounds_into_host_spec() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(P::MAX_THREADS)]
            #[launch_contract(domain = 1)]
            pub fn configured<P: Policy>() {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("let__cuda_oxide_max_threads:u32=P::MAX_THREADS"),
        "policy maximum must be part of the host contract: {expanded}"
    );
    assert!(
        expanded.contains("__cuda_oxide_max_threads>0"),
        "{expanded}"
    );
    assert!(expanded.contains("BlockRequirement::MaxThreads(__cuda_oxide_max_threads)"));
    assert!(expanded.contains("[();(P::MAX_THREADS)asusize]:"));
}

#[test]
fn exact_block_checks_a_deferred_launch_bound_per_specialization() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(P::MAX_THREADS)]
            #[launch_contract(domain = 1, block = (64, 1, 1))]
            pub fn configured<P: Policy>() {}
        }
    };
    let expanded = expand_to_compact_string(module);

    assert!(
        expanded.contains("BlockRequirement::Exact((64u32,1u32,1u32))"),
        "an explicit block must remain exact: {expanded}"
    );
    assert!(
        expanded.contains("64u128<=(__cuda_oxide_max_threads)asu128"),
        "the policy bound must be checked for each specialization: {expanded}"
    );
}

#[test]
fn launch_contract_allows_deferred_minimum_blocks() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            #[launch_bounds(256, P::MIN_BLOCKS)]
            #[launch_contract(domain = 1)]
            pub fn configured<P: Policy>() {}
        }
    };

    expand_cuda_module(module)
        .expect("minimum blocks affects device occupancy metadata, not the host launch contract");
}

#[test]
fn generic_kernel_lookup_misses_route_through_the_divergence_diagnosis() {
    let module: ItemMod = parse_quote! {
        mod kernels {
            #[kernel]
            pub fn scale<T: Copy>(factor: T, mut out: DisjointSlice<T>) {}
        }
    };
    let expanded = expand_to_compact_string(module);
    assert!(
        expanded.contains("::cuda_host::diagnose_generic_kernel_load_error"),
        "a generic `_TID_` lookup miss must be routed through the host/device \
         type-identity divergence diagnosis: {expanded}"
    );
    assert!(
        expanded.contains("&self.__module,"),
        "the diagnosis needs the loaded module to consult its retained \
         `.entry` names: {expanded}"
    );
}

#[test]
fn non_generic_kernel_lookups_never_consult_the_divergence_diagnosis() {
    // Non-generic kernels have fixed PTX names; a miss cannot be a
    // type-identity divergence, so the plain `?` propagation stays.
    let expanded = expand_to_compact_string(one_kernel_module());
    assert!(
        !expanded.contains("diagnose_generic_kernel_load_error"),
        "non-generic lookups must keep the ordinary error path: {expanded}"
    );
}
