/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::common::attr_path_ends_with;
use crate::kernel::codegen::{
    generate_cuda_kernel_impl, generic_kernel_instantiation_tokens,
    generic_kernel_no_instantiation_tokens, route_generic_kernel_attrs,
};
use crate::kernel::scope::{
    explicit_kernel_scope, explicit_kernel_scope_bindings, forwarding_inputs,
    inject_thread_index_scope, is_grid_constant_config_marker, is_kernel_configuration_marker,
    is_unchecked_indexing_config_marker, top_level_kernel_configuration_markers,
};
use crate::kernel::{KernelArgs, inject_grid_constant_markers};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use reserved_oxide_symbols::{KERNEL_PREFIX, KERNEL_SCOPE_LOCAL};
use syn::{Expr, ItemFn, Stmt, parse_quote, spanned::Spanned};

/// A bare `#[kernel]` outside any `#[cuda_module]` carries its own
/// `CudaKernel` impl, the other reference a kernel-only crate cannot
/// resolve.
#[test]
fn bare_kernel_marker_impl_is_host_only() {
    let func: ItemFn = parse_quote! {
        fn scale(out: *mut f32) {}
    };
    let name = format_ident!("scale");

    let device_only = generate_cuda_kernel_impl(&name, "scale", &func, false).to_string();
    assert!(
        device_only.is_empty(),
        "device-only build must emit no marker impl: {device_only}"
    );

    let host = generate_cuda_kernel_impl(&name, "scale", &func, true).to_string();
    assert!(
        host.contains("CudaKernel"),
        "host build must keep the marker impl: {host}"
    );
}

#[test]
fn forwarding_inputs_name_every_irrefutable_parameter_pattern() {
    let function: ItemFn = parse_quote! {
        fn patterns(_: u32, (left, right): (u16, u16), mut value: u8) {}
    };
    let (inputs, names) = forwarding_inputs(&function.sig.inputs).unwrap();
    let forwarded = quote! {
        fn wrapper(#(#inputs),*) {
            target(#(#names),*)
        }
    }
    .to_string()
    .replace(' ', "");

    assert!(forwarded.contains("__cuda_oxide_arg_0:u32"));
    assert!(forwarded.contains("__cuda_oxide_arg_1:(u16,u16)"));
    assert!(forwarded.contains("__cuda_oxide_arg_2:u8"));
    assert!(forwarded.contains("target(__cuda_oxide_arg_0,__cuda_oxide_arg_1,__cuda_oxide_arg_2)"));
}

#[test]
fn grid_constant_parameter_becomes_one_source_index_marker() {
    let mut function: ItemFn = parse_quote! {
        fn copy(prefix: u32, #[grid_constant] descriptor: &TensorMap, out: *mut u32) {
            use_descriptor(descriptor, out);
        }
    };

    inject_grid_constant_markers(&mut function).unwrap();
    let expanded = quote!(#function).to_string().replace(' ', "");

    assert!(!expanded.contains("#[grid_constant]"));
    assert!(expanded.contains(
        "unsafe{::cuda_device::thread::__grid_constant_config::<1usize>();}use_descriptor"
    ));
}

#[test]
fn grid_constant_marker_is_a_forwardable_configuration_marker() {
    let marker: Stmt = parse_quote! {
        unsafe { ::cuda_device::thread::__grid_constant_config::<0usize>(); }
    };
    assert!(is_kernel_configuration_marker(&marker));
    assert!(is_grid_constant_config_marker(&marker));
}

#[test]
fn grid_constant_rejects_mutable_reference() {
    let mut function: ItemFn = parse_quote! {
        fn copy(#[grid_constant] descriptor: &mut TensorMap) {}
    };

    let error = inject_grid_constant_markers(&mut function).unwrap_err();
    assert!(error.to_string().contains("read-only"));
}

#[test]
fn grid_constant_rejects_lifetimes_that_can_outlive_the_launch() {
    for source in [
        "fn copy(#[grid_constant] descriptor: &'static TensorMap) {}",
        "fn copy<'a>(#[grid_constant] descriptor: &'a TensorMap) {}",
        "fn copy<'a: 'static>(#[grid_constant] descriptor: &'a TensorMap) {}",
    ] {
        let mut function: ItemFn = syn::parse_str(source).unwrap();
        let error = inject_grid_constant_markers(&mut function).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("lives only for this kernel launch")
        );
    }
}

#[test]
fn grid_constant_accepts_an_anonymous_lifetime() {
    let mut function: ItemFn = parse_quote! {
        fn copy(#[grid_constant] descriptor: &'_ TensorMap) {}
    };
    inject_grid_constant_markers(&mut function).unwrap();
}

#[test]
fn generic_kernel_routes_entry_directives_without_losing_cfg() {
    let function: ItemFn = parse_quote! {
        #[doc = "configured kernel"]
        #[cfg(feature = "configured")]
        #[launch_bounds(256, 2)]
        #[cluster_launch(2, 1, 1)]
        fn configured<const N: usize>() {}
    };

    let (implementation, entry, cfg) = route_generic_kernel_attrs(&function.attrs);

    assert!(
        implementation
            .iter()
            .any(|attr| attr_path_ends_with(attr, "doc"))
    );
    assert!(
        implementation
            .iter()
            .any(|attr| attr_path_ends_with(attr, "cfg"))
    );
    assert!(
        !implementation
            .iter()
            .any(|attr| attr_path_ends_with(attr, "launch_bounds"))
    );
    assert!(
        !implementation
            .iter()
            .any(|attr| attr_path_ends_with(attr, "cluster_launch"))
    );

    assert!(entry.iter().any(|attr| attr_path_ends_with(attr, "cfg")));
    assert!(
        entry
            .iter()
            .any(|attr| attr_path_ends_with(attr, "launch_bounds"))
    );
    assert!(
        entry
            .iter()
            .any(|attr| attr_path_ends_with(attr, "cluster_launch"))
    );
    assert!(!entry.iter().any(|attr| attr_path_ends_with(attr, "doc")));

    assert_eq!(cfg.len(), 1);
    assert!(attr_path_ends_with(&cfg[0], "cfg"));
}

#[test]
fn kernel_launch_context_uses_the_contract_domain_and_coordinate_width_in_either_order() {
    let mut attributed: ItemFn = parse_quote! {
        #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
        fn attributed() {
            let _ = thread::index_1d_u32(launch_context);
        }
    };
    let scope = explicit_kernel_scope(&mut attributed, format_ident!("launch_context"));
    attributed
        .block
        .stmts
        .splice(0..0, explicit_kernel_scope_bindings(&scope));
    let attributed = quote!(#attributed).to_string().replace(' ', "");
    assert!(attributed.contains("make_kernel_scope::<::cuda_device::thread::__internal::Domain1,::cuda_device::thread::__internal::U32Coordinates>"));
    assert!(attributed.contains("letlaunch_context:::cuda_device::thread::LaunchContextRef"));

    let mut expanded_first: ItemFn = parse_quote! {
        fn expanded_first() {
            unsafe {
                ::cuda_device::thread::__launch_contract_config::<1, true>();
            }
            let _ = thread::index_1d_u32(launch_context);
        }
    };
    let scope = explicit_kernel_scope(&mut expanded_first, format_ident!("launch_context"));
    expanded_first
        .block
        .stmts
        .splice(0..0, explicit_kernel_scope_bindings(&scope));
    let expanded_first = quote!(#expanded_first).to_string().replace(' ', "");
    assert!(expanded_first.contains("make_kernel_scope::<::cuda_device::thread::__internal::Domain1,::cuda_device::thread::__internal::U32Coordinates>"));

    let mut two_dimensional: ItemFn = parse_quote! {
        #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
        fn two_dimensional() {
            let _ = thread::coord_2d_u32(launch_context);
        }
    };
    let scope = explicit_kernel_scope(&mut two_dimensional, format_ident!("launch_context"));
    two_dimensional
        .block
        .stmts
        .splice(0..0, explicit_kernel_scope_bindings(&scope));
    let two_dimensional = quote!(#two_dimensional).to_string().replace(' ', "");
    assert!(two_dimensional.contains("make_kernel_scope::<::cuda_device::thread::__internal::Domain2,::cuda_device::thread::__internal::U32Coordinates>"));
}

#[test]
fn thread_index_rewrite_preserves_the_user_call_span_with_multiple_attributes() {
    let mut input: ItemFn = syn::parse_str(
        r#"
/// The documentation and neighboring attributes must not become line-table
/// locations for the first executable indexing expression.
#[allow(dead_code)]
#[launch_bounds(256, 2)]
fn documented_kernel() {
    let idx = thread::index_1d();
    consume(idx);
}
"#,
    )
    .unwrap();
    let source_attr_count = input.attrs.len();

    let Stmt::Local(original_local) = &input.block.stmts[0] else {
        panic!("fixture starts with a local")
    };
    let original_call = original_local
        .init
        .as_ref()
        .map(|init| &*init.expr)
        .and_then(|expr| match expr {
            Expr::Call(call) => Some(call),
            _ => None,
        })
        .expect("fixture local is initialized by a call");
    let user_call_start = original_call.span().start();
    assert!(
        user_call_start.line > 5,
        "fixture must distinguish attributes"
    );

    inject_thread_index_scope(&mut input);

    assert_eq!(
        input.attrs.len(),
        source_attr_count,
        "all source attributes stay on the item"
    );
    let Stmt::Local(rewritten_local) = &input.block.stmts[1] else {
        panic!("the user local follows the generated scope binding")
    };
    let rewritten_call = rewritten_local
        .init
        .as_ref()
        .map(|init| &*init.expr)
        .and_then(|expr| match expr {
            Expr::Call(call) => Some(call),
            _ => None,
        })
        .expect("rewritten local remains a call expression");

    assert_eq!(
        rewritten_call.span().start(),
        user_call_start,
        "rewriting must not relocate the user expression onto an attribute"
    );
    let Expr::Path(path) = &*rewritten_call.func else {
        panic!("rewritten callee remains a path")
    };
    assert!(
        path.path
            .segments
            .iter()
            .any(|segment| segment.ident == "__internal")
    );
    assert_eq!(
        rewritten_call.args[0].span().start(),
        user_call_start,
        "the injected scope reference belongs to the user's call site"
    );
}

#[test]
fn kernel_launch_context_argument_composes_with_legacy_instantiations() {
    let args: KernelArgs = syn::parse_str("f32, launch_context = launch_context, f64").unwrap();
    assert_eq!(args.instantiate_types.len(), 2);
    assert_eq!(args.launch_context.unwrap(), "launch_context");

    let duplicate = syn::parse_str::<KernelArgs>("launch_context = first, launch_context = second")
        .err()
        .unwrap();
    assert!(duplicate.to_string().contains("duplicate `launch_context`"));

    let unknown = syn::parse_str::<KernelArgs>("context = launch_context")
        .err()
        .unwrap();
    assert!(
        unknown
            .to_string()
            .contains("unknown #[kernel] named argument")
    );
}

#[test]
fn kernel_unchecked_indexing_flag_parses_and_composes() {
    let bare: KernelArgs = syn::parse_str("unchecked_indexing").unwrap();
    assert!(bare.unchecked_indexing);
    assert!(bare.instantiate_types.is_empty());
    assert!(bare.launch_context.is_none());

    let composed: KernelArgs =
        syn::parse_str("f32, launch_context = launch_context, unchecked_indexing").unwrap();
    assert!(composed.unchecked_indexing);
    assert_eq!(composed.instantiate_types.len(), 1);
    assert_eq!(composed.launch_context.unwrap(), "launch_context");

    let leading: KernelArgs = syn::parse_str("unchecked_indexing, f32").unwrap();
    assert!(leading.unchecked_indexing);
    assert_eq!(leading.instantiate_types.len(), 1);

    let default: KernelArgs = syn::parse_str("f32, launch_context = launch_context").unwrap();
    assert!(!default.unchecked_indexing);

    let duplicate = syn::parse_str::<KernelArgs>("unchecked_indexing, unchecked_indexing")
        .err()
        .unwrap();
    assert!(
        duplicate
            .to_string()
            .contains("duplicate `unchecked_indexing`")
    );

    // The flag is a bare word, never a named argument.
    let named = syn::parse_str::<KernelArgs>("unchecked_indexing = true")
        .err()
        .unwrap();
    assert!(
        named
            .to_string()
            .contains("unknown #[kernel] named argument")
    );
}

#[test]
fn unchecked_indexing_marker_is_a_forwardable_configuration_marker() {
    // The exact statement the `#[kernel]` macro injects must be recognized
    // by the generic-entry marker forwarding, so the flag reaches the
    // generated entry wrapper even when the implementation helper is
    // translated separately.
    let marker: Stmt = parse_quote! {
        ::cuda_device::thread::__unchecked_indexing_config::<true>();
    };
    assert!(is_kernel_configuration_marker(&marker));
    assert!(is_unchecked_indexing_config_marker(&marker));

    let kernel: ItemFn = parse_quote! {
        fn body() {
            ::cuda_device::thread::__unchecked_indexing_config::<true>();
            work();
        }
    };
    let markers = top_level_kernel_configuration_markers(&kernel);
    assert_eq!(markers.len(), 1);
}

/// Renders the function item named `name` from an expansion, panicking
/// with the full expansion when it is missing.
fn expansion_fn_source(expanded: &TokenStream2, name: &str) -> String {
    let file: syn::File =
        syn::parse2(expanded.clone()).expect("generated tokens must parse as items");
    let function = file
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(item_fn) if item_fn.sig.ident == name => Some(item_fn),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expansion has no fn `{name}`:\n{expanded}"));
    quote!(#function).to_string().replace(' ', "")
}

/// The kernel body exactly as `kernel()` hands it to the generic
/// expansion paths: the unchecked-indexing marker is already statement 0.
fn opted_generic_kernel() -> ItemFn {
    parse_quote! {
        pub fn scaled_gather<T: Copy>(a: &[T], mut c: DisjointSlice<T>) {
            ::cuda_device::thread::__unchecked_indexing_config::<true>();
            work(a, &mut c);
        }
    }
}

#[test]
fn generic_expansion_confines_unchecked_marker_to_entry_and_hidden_twin() {
    let expanded = generic_kernel_no_instantiation_tokens(opted_generic_kernel(), None);

    // The user-named implementation helper is ordinary callable Rust and
    // must NOT carry the marker: rustc may inline it into other kernels
    // that never opted in.
    let helper = expansion_fn_source(&expanded, "scaled_gather");
    assert!(
        !helper.contains("__unchecked_indexing_config"),
        "helper leaked the marker:\n{helper}"
    );

    // The generated entry wrapper keeps the forwarded marker and calls
    // the hidden unchecked twin instead of the user-named helper.
    let entry = expansion_fn_source(&expanded, &format!("{KERNEL_PREFIX}scaled_gather"));
    assert!(
        entry.contains("__unchecked_indexing_config"),
        "entry wrapper lost the marker:\n{entry}"
    );
    assert!(
        entry.contains("__cuda_oxide_unchecked_impl_scaled_gather::<T>"),
        "entry wrapper does not call the unchecked twin:\n{entry}"
    );

    let twin = expansion_fn_source(&expanded, "__cuda_oxide_unchecked_impl_scaled_gather");
    assert!(twin.contains("__unchecked_indexing_config"));
}

#[test]
fn generic_expansion_confines_grid_constant_marker_to_entry() {
    let kernel: ItemFn = parse_quote! {
        pub fn read<T: Copy>(descriptor: &Descriptor, out: *mut u32, tag: T) {
            unsafe { ::cuda_device::thread::__grid_constant_config::<0usize>(); }
            work(descriptor, out, tag);
        }
    };
    let expanded = generic_kernel_no_instantiation_tokens(kernel, None);

    let helper = expansion_fn_source(&expanded, "read");
    assert!(!helper.contains("__grid_constant_config"));
    let entry = expansion_fn_source(&expanded, &format!("{KERNEL_PREFIX}read"));
    assert!(entry.contains("__grid_constant_config"));
}

#[test]
fn legacy_instantiation_confines_unchecked_marker_to_entry_and_hidden_twin() {
    // Legacy `#[kernel(Type, ...)]` instantiation supports by-value
    // parameters of the single type parameter (see
    // `kernel_launch_context_api.rs`'s `explicit` kernel).
    let kernel: ItemFn = parse_quote! {
        pub fn scaled_gather<T: Copy>(value: T) {
            ::cuda_device::thread::__unchecked_indexing_config::<true>();
            work(value);
        }
    };
    let expanded = generic_kernel_instantiation_tokens(kernel, vec![parse_quote! { f32 }], None);

    let helper = expansion_fn_source(&expanded, "scaled_gather");
    assert!(
        !helper.contains("__unchecked_indexing_config"),
        "helper leaked the marker:\n{helper}"
    );

    let entry = expansion_fn_source(&expanded, &format!("{KERNEL_PREFIX}scaled_gather_f32"));
    assert!(
        entry.contains("__unchecked_indexing_config"),
        "entry wrapper lost the marker:\n{entry}"
    );
    assert!(
        entry.contains("__cuda_oxide_unchecked_impl_scaled_gather::<f32>"),
        "entry wrapper does not call the unchecked twin:\n{entry}"
    );

    let twin = expansion_fn_source(&expanded, "__cuda_oxide_unchecked_impl_scaled_gather");
    assert!(twin.contains("__unchecked_indexing_config"));
}

#[test]
fn legacy_instantiation_confines_grid_constant_marker_to_entry() {
    let kernel: ItemFn = parse_quote! {
        pub fn read<T: Copy>(descriptor: &Descriptor, tag: T) {
            unsafe { ::cuda_device::thread::__grid_constant_config::<0usize>(); }
            work(descriptor, tag);
        }
    };
    let expanded = generic_kernel_instantiation_tokens(kernel, vec![parse_quote! { u32 }], None);

    let helper = expansion_fn_source(&expanded, "read");
    assert!(!helper.contains("__grid_constant_config"));
    let entry = expansion_fn_source(&expanded, &format!("{KERNEL_PREFIX}read_u32"));
    assert!(entry.contains("__grid_constant_config"));
}

#[test]
fn non_opted_generic_expansion_has_no_unchecked_twin_or_marker() {
    let kernel: ItemFn = parse_quote! {
        pub fn plain<T: Copy>(a: &[T], mut c: DisjointSlice<T>) {
            work(a, &mut c);
        }
    };
    let expanded = generic_kernel_no_instantiation_tokens(kernel, None).to_string();
    assert!(!expanded.contains("__unchecked_indexing_config"));
    assert!(!expanded.contains("__cuda_oxide_unchecked_impl"));
    assert!(!expanded.contains("dead_code"));
}

#[test]
fn unchecked_twin_carries_helper_lint_attributes() {
    // A lint the author suppressed on the kernel must not re-fire on the
    // byte-identical twin body (that would break -Dwarnings builds), and
    // the user-named helper must be allowed to go dead: the generated
    // entry calls the twin instead.
    let kernel: ItemFn = parse_quote! {
        #[allow(unused_variables)]
        pub fn scaled_gather<T: Copy>(a: &[T], mut c: DisjointSlice<T>) {
            ::cuda_device::thread::__unchecked_indexing_config::<true>();
            work(a, &mut c);
        }
    };
    let expanded = generic_kernel_no_instantiation_tokens(kernel, None);

    let twin = expansion_fn_source(&expanded, "__cuda_oxide_unchecked_impl_scaled_gather");
    assert!(
        twin.contains("allow(unused_variables)"),
        "twin dropped the helper's lint attributes:\n{twin}"
    );

    let helper = expansion_fn_source(&expanded, "scaled_gather");
    assert!(helper.contains("allow(unused_variables)"));
    assert!(
        helper.contains("allow(dead_code)"),
        "opted helper is not allowed to go dead:\n{helper}"
    );

    // The entry wrapper is the live path; it must not be dead-code
    // suppressed.
    let entry = expansion_fn_source(&expanded, &format!("{KERNEL_PREFIX}scaled_gather"));
    assert!(!entry.contains("dead_code"));
}

#[test]
fn raw_identifier_kernel_names_build_a_valid_unchecked_twin() {
    // `Display` of `r#gen` keeps the `r#` prefix, which `Ident::new`
    // rejects; twin naming must strip it like `format_ident!` does.
    let kernel: ItemFn = parse_quote! {
        pub fn r#gen<T: Copy>(value: T) {
            ::cuda_device::thread::__unchecked_indexing_config::<true>();
            work(value);
        }
    };

    let expanded = generic_kernel_no_instantiation_tokens(kernel.clone(), None);
    let twin = expansion_fn_source(&expanded, "__cuda_oxide_unchecked_impl_gen");
    assert!(twin.contains("__unchecked_indexing_config"));

    let expanded = generic_kernel_instantiation_tokens(kernel, vec![parse_quote! { f32 }], None);
    let twin = expansion_fn_source(&expanded, "__cuda_oxide_unchecked_impl_gen");
    assert!(twin.contains("__unchecked_indexing_config"));
}

#[test]
fn generic_kernels_validate_routed_launch_contract_requires_at_source_names() {
    // A #[launch_contract] written below #[kernel] is routed onto the
    // generated entry wrapper, whose synthetic parameter names defeat
    // attribute-site validation; #[kernel] must validate the relations
    // against the original signature instead.
    let typo: ItemFn = parse_quote! {
        #[cuda_device::launch_contract(
            domain = 1,
            block = (64, 1, 1),
            requires = (input.len() >= n),
        )]
        pub fn scaled<T: Copy>(input: &[T]) {}
    };
    let expanded = generic_kernel_no_instantiation_tokens(typo.clone(), None).to_string();
    assert!(expanded.contains("compile_error"), "{expanded}");
    assert!(expanded.contains("unknown identifier `n`"), "{expanded}");

    let expanded =
        generic_kernel_instantiation_tokens(typo, vec![parse_quote! { f32 }], None).to_string();
    assert!(expanded.contains("compile_error"), "{expanded}");
    assert!(expanded.contains("unknown identifier `n`"), "{expanded}");

    let well_formed: ItemFn = parse_quote! {
        #[cuda_device::launch_contract(
            domain = 1,
            block = (64, 1, 1),
            requires = (input.len() >= n),
        )]
        pub fn scaled<T: Copy>(n: u32, input: &[T]) {}
    };
    let expanded = generic_kernel_no_instantiation_tokens(well_formed, None).to_string();
    assert!(!expanded.contains("compile_error"), "{expanded}");
}

#[test]
fn path_spelled_unchecked_indexing_still_parses_as_instantiation_type() {
    // The bare word is reserved for the flag, but a type literally named
    // `unchecked_indexing` remains reachable through any path spelling.
    for spelling in [
        "self::unchecked_indexing",
        "crate::unchecked_indexing",
        "types::unchecked_indexing",
    ] {
        let args: KernelArgs = syn::parse_str(spelling).unwrap();
        assert!(!args.unchecked_indexing, "`{spelling}` parsed as the flag");
        assert_eq!(
            args.instantiate_types.len(),
            1,
            "`{spelling}` did not parse as an instantiation type"
        );
    }
}

#[test]
fn explicit_fast_functions_are_not_syntax_rewritten() {
    let mut function: ItemFn = parse_quote! {
        #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
        fn ordinary_names() {
            let local = index_1d_u32();
            let proof = alias(launch_context);
            consume(local, proof);
        }
    };
    let scope = explicit_kernel_scope(&mut function, format_ident!("launch_context"));
    let expanded = quote!(#function).to_string().replace(' ', "");

    assert!(expanded.contains("index_1d_u32()"));
    assert!(expanded.contains("alias(launch_context)"));
    assert!(!expanded.contains("__internal::index_1d_u32"));
    assert_eq!(scope.ident, "launch_context");
}

#[test]
fn generic_kernel_routes_launch_contract_to_entry_only() {
    let kernel: ItemFn = parse_quote! {
        #[doc = "kept on the helper"]
        #[cuda_device::launch_bounds(128)]
        #[cuda_device::launch_contract(
            domain = 1,
            dynamic_shared = 256,
            dynamic_shared_alignment = 64,
        )]
        #[cuda_device::cluster_launch(2, 1, 1)]
        #[cuda_device::cooperative_launch]
        pub fn map<T: Copy>(value: T) {}
    };

    let (implementation, entry, _cfg) = route_generic_kernel_attrs(&kernel.attrs);
    let entry_names: Vec<_> = entry
        .iter()
        .map(|attr| attr.path().segments.last().unwrap().ident.to_string())
        .collect();
    let implementation_names: Vec<_> = implementation
        .iter()
        .map(|attr| attr.path().segments.last().unwrap().ident.to_string())
        .collect();

    assert_eq!(
        entry_names,
        [
            "launch_bounds",
            "launch_contract",
            "cluster_launch",
            "cooperative_launch",
        ]
    );
    assert_eq!(implementation_names, ["doc"]);
}

#[test]
fn generic_kernel_forwards_only_exact_top_level_configuration_markers() {
    let kernel: ItemFn = parse_quote! {
        fn map<T>() {
            ::cuda_device::thread::__launch_bounds_config::<64, 2>();
            unsafe {
                ::cuda_device::thread::__launch_contract_config::<1, true>();
            }
            ::cuda_device::thread::__launch_contract_block_config::<64, 1, 1>();
            ::cuda_device::cluster::__cluster_config::<2, 1, 1>();
            ::cuda_device::shared::__dynamic_shared_alignment::<128>();
            cuda_device::thread::__launch_bounds_config::<4, 1>();
            unrelated();
            other::cuda_device::thread::__launch_bounds_config::<32, 1>();
            cuda_device::thread::__launch_bounds_config::<16, 1>(7);
            {
                cuda_device::thread::__launch_bounds_config::<8, 1>();
            }
        }
    };

    let markers = top_level_kernel_configuration_markers(&kernel);
    let forwarded = quote!(#(#markers)*).to_string().replace(' ', "");

    assert_eq!(markers.len(), 5);
    assert!(forwarded.contains("__launch_bounds_config::<64,2>()"));
    assert!(forwarded.contains("__launch_contract_config::<1,true>()"));
    assert!(forwarded.contains("__launch_contract_block_config::<64,1,1>()"));
    assert!(forwarded.contains("__cluster_config::<2,1,1>()"));
    assert!(forwarded.contains("__dynamic_shared_alignment::<128>()"));
    assert!(!forwarded.contains("unrelated"));
    assert!(!forwarded.contains("::<4,1>()"));
    assert!(!forwarded.contains("other::cuda_device"));
    assert!(!forwarded.contains("::<16,1>(7)"));
    assert!(!forwarded.contains("::<8,1>()"));
}

/// The two hygiene regressions for the injected launch-context scope spell
/// `KERNEL_SCOPE_LOCAL`-derived names as *identifiers*, so no compiler
/// check ties them back to the constant. Rename the constant and both
/// fixtures quietly stop colliding with the generated bindings: they keep
/// passing while testing nothing, because there is no longer anything to
/// collide with. Pin the coupling so a rename fails here and names the
/// fixtures that have to move with it.
#[test]
fn hygiene_fixtures_still_collide_with_the_generated_scope_names() {
    let storage = format!("{KERNEL_SCOPE_LOCAL}_storage");

    let launch_context = include_str!("../../tests/pass/kernel_launch_context_api.rs");
    assert!(
        launch_context.contains(&storage),
        "tests/pass/kernel_launch_context_api.rs must bind `{storage}` so \
             `generated_storage_name_is_hygienic` exercises the mixed-site \
             hygiene of the generated storage binding"
    );

    let const_generic =
        include_str!("../../../rustc-codegen-cuda/examples/const_generic/src/main.rs");
    assert!(
        const_generic.contains(KERNEL_SCOPE_LOCAL),
        "examples/const_generic must declare a const generic named \
             `{KERNEL_SCOPE_LOCAL}` so `call_site_ident_avoiding_item` is \
             forced down its rename path"
    );
}
