/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::cuda_module::contract::{LaunchContractArgs, validate_requires_relations};
use crate::launch::{
    CudaLaunchAsyncInput, CudaLaunchInput, expand_cuda_launch, expand_cuda_launch_async,
    kernel_sibling_path,
};
use crate::launch_attrs::{
    LaunchBoundsArgs, LoopUnrollAttrVisitor, UnrollArgs, add_const_evaluatable_bound,
    inject_launch_contract_markers, rewrite_loop_unroll_attrs, standalone_requires_params,
};
use quote::{format_ident, quote};
use reserved_oxide_symbols::INSTANTIATE_PREFIX;
use syn::{ItemFn, Stmt, parse_quote, visit_mut::VisitMut};

#[test]
fn generated_kernel_siblings_preserve_qualified_paths_and_generics() {
    let kernel: syn::Path = parse_quote! { kernels::map::<_, 4> };
    let instantiate = kernel_sibling_path(&kernel, format_ident!("{}map", INSTANTIATE_PREFIX));
    let marker = kernel_sibling_path(&kernel, format_ident!("__map_CudaKernel"));
    let ptx_name = kernel_sibling_path(&kernel, format_ident!("map_ptx_name"));

    let instantiate = quote! { #instantiate }.to_string().replace(' ', "");
    let marker = quote! { #marker }.to_string().replace(' ', "");
    let ptx_name = quote! { #ptx_name }.to_string().replace(' ', "");
    assert_eq!(
        instantiate,
        format!("kernels::{}map::<_,4>", INSTANTIATE_PREFIX)
    );
    assert_eq!(marker, "kernels::__map_CudaKernel::<_,4>");
    assert_eq!(ptx_name, "kernels::map_ptx_name::<_,4>");
}

#[test]
fn cuda_launch_accepts_combined_cluster_and_cooperative() {
    let input: CudaLaunchInput = parse_quote! {
        kernel: clustered_grid_sync_kernel,
        stream: stream,
        module: module,
        config: config,
        cluster_dim: (2, 1, 1),
        cooperative: true,
        args: []
    };
    assert!(input.cluster_dim.is_some());
    assert!(input.cooperative.is_some());

    let expanded = expand_cuda_launch(input).to_string().replace(' ', "");
    assert!(
        expanded.contains("launch_kernel_ex_cooperative_on_stream"),
        "expected combined cluster+cooperative cuda_launch expansion:\n{expanded}"
    );
    assert!(
        !expanded.contains("launch_kernel_cooperative_on_stream"),
        "non-cluster cooperative helper must not be used when cluster_dim is set:\n{expanded}"
    );
}

#[test]
fn standalone_launch_contract_validates_requires_against_fn_signature() {
    let kernel: ItemFn = parse_quote! {
        pub fn scaled(n: u32, input: &[f32], mut output: DisjointSlice<f32>) {}
    };
    let params = standalone_requires_params(&kernel).expect("source-named params must model");

    let good: LaunchContractArgs = syn::parse_str(
        "domain = 1, block = (64, 1, 1), requires = (input.len() >= n, output.len() >= n)",
    )
    .unwrap();
    assert!(validate_requires_relations(&good.requires, &params).is_ok());

    let typo: LaunchContractArgs =
        syn::parse_str("domain = 1, block = (64, 1, 1), requires = (input.len() >= m)").unwrap();
    let error = validate_requires_relations(&typo.requires, &params).unwrap_err();
    assert!(
        error.to_string().contains("unknown identifier `m`"),
        "{error}"
    );
}

#[test]
fn standalone_requires_validation_skips_generated_generic_entry_wrappers() {
    // Generic entry wrappers carry synthetic parameter names; the source
    // relations were already validated by #[cuda_module] (or are written
    // against names this wrapper no longer has), so the attribute-site
    // validation must stand down rather than reject valid contracts.
    let wrapper: ItemFn = parse_quote! {
        fn entry<T>(__cuda_oxide_arg_0: &[T], __cuda_oxide_arg_1: u32) {}
    };
    assert!(standalone_requires_params(&wrapper).is_none());
}

#[test]
fn standalone_requires_models_unmarshallable_params_as_opaque_scalars() {
    // `&bool` is rejected by the cuda_module marshaller; standalone
    // validation still names the parameter precisely instead of calling
    // it an unknown identifier.
    let kernel: ItemFn = parse_quote! {
        pub fn odd(flag: &bool, n: u32) {}
    };
    let params = standalone_requires_params(&kernel).unwrap();
    let args: LaunchContractArgs =
        syn::parse_str("domain = 1, block = (64, 1, 1), requires = (n >= flag)").unwrap();
    let error = validate_requires_relations(&args.requires, &params).unwrap_err();
    assert!(
        error.to_string().contains("not an unsigned integer scalar"),
        "{error}"
    );
}

#[test]
fn launch_contract_injects_one_alignment_marker_only_when_shared_is_used() {
    let args: LaunchContractArgs =
        syn::parse_str("domain = 1, dynamic_shared = 256, dynamic_shared_alignment = 64").unwrap();
    let mut helper: ItemFn = parse_quote! { fn helper<T>() {} };
    inject_launch_contract_markers(&args, &mut helper);
    let expanded = quote!(#helper).to_string();
    assert_eq!(expanded.matches("__launch_contract_config").count(), 1);
    assert_eq!(expanded.matches("__dynamic_shared_alignment").count(), 1);
    assert!(expanded.contains("64usize"));

    let zero_args: LaunchContractArgs = syn::parse_str("domain = 1, dynamic_shared = 0").unwrap();
    let mut zero_helper: ItemFn = parse_quote! { fn zero_helper<T>() {} };
    inject_launch_contract_markers(&zero_args, &mut zero_helper);
    assert!(
        !quote!(#zero_helper)
            .to_string()
            .contains("__dynamic_shared_alignment")
    );
    assert_eq!(
        quote!(#zero_helper)
            .to_string()
            .matches("__launch_contract_config")
            .count(),
        1
    );
}

#[test]
fn launch_contract_injects_the_block_marker_only_for_an_exact_block() {
    let exact: LaunchContractArgs =
        syn::parse_str("domain = 2, block = (8, 8, 1), dynamic_shared = 0").unwrap();
    let mut kernel: ItemFn = parse_quote! { fn kernel() {} };
    inject_launch_contract_markers(&exact, &mut kernel);
    let expanded = quote!(#kernel).to_string().replace(' ', "");
    assert_eq!(
        expanded.matches("__launch_contract_block_config").count(),
        1
    );
    assert!(expanded.contains("__launch_contract_block_config::<8u32,8u32,1u32>"));

    // Without an exact block the kernel keeps whatever `#[launch_bounds]`
    // declares, so no exact shape reaches the device compiler.
    let bounded: LaunchContractArgs = syn::parse_str("domain = 1, dynamic_shared = 0").unwrap();
    let mut unbounded_kernel: ItemFn = parse_quote! { fn unbounded_kernel() {} };
    inject_launch_contract_markers(&bounded, &mut unbounded_kernel);
    assert!(
        !quote!(#unbounded_kernel)
            .to_string()
            .contains("__launch_contract_block_config")
    );
}

#[test]
fn launch_contract_keeps_every_marker_when_block_and_shared_are_both_declared() {
    let args: LaunchContractArgs = syn::parse_str(
        "domain = 1, block = (256, 1, 1), dynamic_shared = 128, dynamic_shared_alignment = 32",
    )
    .unwrap();
    let mut kernel: ItemFn = parse_quote! { fn kernel() {} };
    inject_launch_contract_markers(&args, &mut kernel);
    let expanded = quote!(#kernel).to_string();
    assert_eq!(expanded.matches("__launch_contract_config").count(), 1);
    assert_eq!(
        expanded.matches("__launch_contract_block_config").count(),
        1
    );
    assert_eq!(expanded.matches("__dynamic_shared_alignment").count(), 1);
}

#[test]
fn raw_async_launch_macro_leaves_finalization_caller_unsafe() {
    let inputs: [CudaLaunchAsyncInput; 3] = [
        parse_quote! {
            kernel: kernels::map,
            module: module,
            config: config,
            args: []
        },
        parse_quote! {
            kernel: kernels::map::<u32>,
            module: module,
            config: config,
            args: []
        },
        parse_quote! {
            kernel: kernels::map,
            module: module,
            config: config,
            args: [|value: u32| value]
        },
    ];

    for input in inputs {
        let expanded = expand_cuda_launch_async(input).to_string().replace(' ', "");
        assert!(expanded.contains("AsyncKernelLaunchBuilder::new"));
        assert!(expanded.contains(".finalize_unchecked(config)"));
        assert!(!expanded.contains("set_launch_config"));
        assert!(
            !expanded.contains("unsafe{"),
            "the macro must not hide its raw-launch safety obligation: {expanded}"
        );
    }
}

/// Runs the per-loop `#[unroll]` visitor over `func`'s body and returns the
/// rewritten function as a whitespace-free string, panicking on any
/// recorded parse error.
fn run_loop_unroll_visitor(mut func: ItemFn) -> String {
    rewrite_loop_unroll_attrs(&mut func)
        .unwrap_or_else(|err| panic!("unexpected unroll-attr error: {err}"));
    quote!(#func).to_string().replace(' ', "")
}

#[test]
fn bare_unroll_on_while_injects_factor_zero_marker() {
    let func: ItemFn = parse_quote! {
        fn k() {
            let mut i = 0u32;
            #[unroll]
            while i < 4 { i += 1; }
        }
    };
    let out = run_loop_unroll_visitor(func);
    // Bare #[unroll] => full unroll (factor 0).
    assert!(
        out.contains("cuda_device::thread::__unroll_config::<{0u32}>()"),
        "expected factor-0 marker:\n{out}"
    );
    // The expression attribute must be stripped so rustc never sees it.
    assert!(
        !out.contains("#[unroll]"),
        "the #[unroll] attribute should be removed:\n{out}"
    );
}

#[test]
fn unroll_n_on_for_loop_injects_factor_n_marker() {
    let func: ItemFn = parse_quote! {
        fn k() {
            #[unroll(4)]
            for i in 0..8 { let _ = i; }
        }
    };
    let out = run_loop_unroll_visitor(func);
    assert!(
        out.contains("cuda_device::thread::__unroll_config::<{4}>()"),
        "expected factor-4 marker:\n{out}"
    );
    assert!(
        !out.contains("#[unroll"),
        "attribute should be removed:\n{out}"
    );
}

#[test]
fn unroll_on_bare_loop_injects_marker() {
    let func: ItemFn = parse_quote! {
        fn k() {
            #[unroll(2)]
            loop { break; }
        }
    };
    let out = run_loop_unroll_visitor(func);
    assert!(
        out.contains("cuda_device::thread::__unroll_config::<{2}>()"),
        "expected factor-2 marker on `loop`:\n{out}"
    );
}

#[test]
fn nested_annotated_loop_is_handled() {
    let func: ItemFn = parse_quote! {
        fn k() {
            for outer in 0..2 {
                if outer == 0 {
                    #[unroll(3)]
                    for inner in 0..4 { let _ = inner; }
                }
            }
        }
    };
    let out = run_loop_unroll_visitor(func);
    assert!(
        out.contains("cuda_device::thread::__unroll_config::<{3}>()"),
        "expected marker injected into nested loop:\n{out}"
    );
}

#[test]
fn loop_without_unroll_attr_is_unchanged() {
    // A function whose loops carry no #[unroll] must be byte-identical
    // before and after the visitor runs (no marker, no edits).
    let src: ItemFn = parse_quote! {
        fn k() {
            let mut i = 0u32;
            while i < 4 { i += 1; }
            for j in 0..2 { let _ = j; }
        }
    };
    let before = quote!(#src).to_string();
    let after = run_loop_unroll_visitor(src);
    assert_eq!(
        before.replace(' ', ""),
        after,
        "loops without #[unroll] must not be modified"
    );
    assert!(
        !after.contains("__unroll_config"),
        "no marker should be injected without #[unroll]:\n{after}"
    );
}

#[test]
fn malformed_unroll_attr_records_error() {
    // `#[unroll(1, 2)]` has too many args; the visitor must record an error.
    let mut func: ItemFn = parse_quote! {
        fn k() {
            #[unroll(1, 2)]
            for i in 0..4 { let _ = i; }
        }
    };
    let mut visitor = LoopUnrollAttrVisitor::default();
    visitor.visit_block_mut(&mut func.block);
    assert!(
        visitor.error.is_some(),
        "malformed #[unroll(1, 2)] should record a parse error"
    );
}

#[test]
fn partial_unroll_factor_must_be_at_least_two() {
    assert!(syn::parse_str::<UnrollArgs>("0").is_err());
    assert!(syn::parse_str::<UnrollArgs>("1").is_err());
    assert_eq!(
        syn::parse_str::<UnrollArgs>("2")
            .unwrap()
            .factor
            .literal_value,
        Some(2)
    );
    assert!(syn::parse_str::<UnrollArgs>("1025").is_err());
}

#[test]
fn launch_bounds_keeps_policy_expressions_typed() {
    let mut function: ItemFn = parse_quote! {
        fn configured<P: Policy>() {}
    };
    let args: LaunchBoundsArgs = syn::parse_str("P::MAX_THREADS * 2, P::MIN_BLOCKS").unwrap();
    add_const_evaluatable_bound(&mut function.sig.generics, &args.max_threads);
    add_const_evaluatable_bound(&mut function.sig.generics, &args.min_blocks);

    let max_threads = &args.max_threads.expr;
    let min_blocks = &args.min_blocks.expr;
    let marker: Stmt = parse_quote! {
        ::cuda_device::thread::__launch_bounds_config::<
            { #max_threads },
            { #min_blocks },
        >();
    };
    function.block.stmts.insert(0, marker);
    let output = quote!(#function).to_string().replace(' ', "");

    assert!(
        output.contains("__launch_bounds_config"),
        "missing marker: {output}"
    );
    assert!(output.contains("{P::MAX_THREADS*2}"), "{output}");
    assert!(output.contains("{P::MIN_BLOCKS}"), "{output}");
    assert!(output.contains("[();(P::MAX_THREADS*2)asusize]:"));
    assert!(output.contains("[();(P::MIN_BLOCKS)asusize]:"));
}

#[test]
fn policy_unroll_expression_gets_marker_and_evaluatability_bound() {
    let func: ItemFn = parse_quote! {
        fn k<P: Policy>() {
            let mut i = 0u32;
            #[unroll(P::UNROLL)]
            while i < 8 { i += 1; }
        }
    };
    let output = run_loop_unroll_visitor(func);

    assert!(output.contains("__unroll_config::<{P::UNROLL}>()"));
    assert!(output.contains("[();(P::UNROLL)asusize]:"));
}

#[test]
fn function_local_unroll_const_stays_in_block_scope() {
    let func: ItemFn = parse_quote! {
        fn k() {
            const FACTOR: u32 = 4;
            let mut i = 0u32;
            #[unroll(FACTOR)]
            while i < 8 { i += 1; }
        }
    };
    let output = run_loop_unroll_visitor(func);

    assert!(output.contains("__unroll_config::<{FACTOR}>()"));
    assert!(
        !output.contains("[();(FACTOR)asusize]:"),
        "a block-local const must not be copied into the function signature: {output}"
    );
}

#[test]
fn generic_and_closure_launches_route_misses_through_the_divergence_panic() {
    let generic: CudaLaunchInput = parse_quote! {
        kernel: kernels::map::<u32>,
        stream: stream,
        module: module,
        config: config,
        args: []
    };
    let closure: CudaLaunchInput = parse_quote! {
        kernel: kernels::map,
        stream: stream,
        module: module,
        config: config,
        args: [|value: u32| value]
    };
    for input in [generic, closure] {
        let expanded = expand_cuda_launch(input).to_string().replace(' ', "");
        assert!(
            expanded.contains("::cuda_host::panic_generic_kernel_load_failed"),
            "generic `_TID_` misses must go through the type-identity \
             divergence diagnosis: {expanded}"
        );
        assert!(
            !expanded.contains("Failedtoloadkernel"),
            "the plain-miss message now lives in the helper, not the \
             expansion: {expanded}"
        );
    }

    let async_generic: CudaLaunchAsyncInput = parse_quote! {
        kernel: kernels::map::<u32>,
        module: module,
        config: config,
        args: []
    };
    let async_closure: CudaLaunchAsyncInput = parse_quote! {
        kernel: kernels::map,
        module: module,
        config: config,
        args: [|value: u32| value]
    };
    for input in [async_generic, async_closure] {
        let expanded = expand_cuda_launch_async(input).to_string().replace(' ', "");
        assert!(
            expanded.contains("::cuda_host::panic_generic_kernel_load_failed"),
            "async generic `_TID_` misses must go through the type-identity \
             divergence diagnosis: {expanded}"
        );
    }
}

#[test]
fn non_generic_launches_keep_the_plain_lookup_panic() {
    // A fixed-name kernel cannot diverge on a `_TID_` hash, so its miss
    // keeps the macro-local panic (and stays duck-typed over the module
    // expression, which trybuild fixtures rely on).
    let plain: CudaLaunchInput = parse_quote! {
        kernel: kernels::map,
        stream: stream,
        module: module,
        config: config,
        args: []
    };
    let expanded = expand_cuda_launch(plain).to_string().replace(' ', "");
    assert!(
        !expanded.contains("panic_generic_kernel_load_failed"),
        "{expanded}"
    );
    assert!(expanded.contains("Failedtoloadkernel"), "{expanded}");

    let async_plain: CudaLaunchAsyncInput = parse_quote! {
        kernel: kernels::map,
        module: module,
        config: config,
        args: []
    };
    let expanded = expand_cuda_launch_async(async_plain)
        .to_string()
        .replace(' ', "");
    assert!(
        !expanded.contains("panic_generic_kernel_load_failed"),
        "{expanded}"
    );
    assert!(expanded.contains("Failedtoloadkernel"), "{expanded}");
}
