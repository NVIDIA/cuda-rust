/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(dead_code, non_upper_case_globals, clippy::needless_lifetimes)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
#[cfg(feature = "async")]
use cuda_host::cuda_async::simt::device_box::DeviceBox;
#[cfg(feature = "async")]
use cuda_host::cuda_async::simt::device_operation::DeviceOperation;
use cuda_host::cuda_module;
use cuda_macros::{cooperative_launch, kernel, launch_contract};

#[cfg(feature = "async")]
type TwoF32Buffers = (DeviceBox<[f32]>, DeviceBox<[f32]>);

#[repr(C)]
#[derive(Clone, Copy)]
struct AffineParams {
    scale: f32,
    bias: f32,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn scalar_args(
        scale: f32,
        params: AffineParams,
        raw: *const f32,
        input: &[f32],
        output: &mut [f32],
    ) {
        let _ = (scale, params, raw, input, output);
    }

    #[kernel]
    pub fn copy_closure<F: Fn(u32) -> u32 + Copy>(op: F, output: &mut [u32]) {
        let _ = (op, output);
    }

    #[kernel]
    pub fn const_only<const N: usize>(output: &mut [u32]) {
        let _ = (N, output);
    }

    #[kernel]
    pub fn mixed<T: Copy, const N: usize>(value: T, output: &mut [T]) {
        let _ = (value, N, output);
    }

    #[kernel]
    pub fn lifetime_mixed<'a, T: Copy + 'a, const N: usize>(input: &'a [T], output: &mut [T]) {
        let _ = (input, output, N);
    }

    #[kernel]
    pub fn hygiene_probe<
        '__cuda_oxide_async,
        const stream: usize,
        const config: usize,
        const __cuda_oxide_arg_0: usize,
        const __cuda_oxide_arg_1: usize,
        const __cuda_oxide_kernel_hash: usize,
        const __cuda_oxide_kernel_ptr: usize,
        const __cuda_oxide_force_mono: usize,
        const __cuda_oxide_ptx_name: usize,
        const __cuda_oxide_function: usize,
        const __cuda_oxide_function_storage: usize,
        const __cuda_oxide_function_cache: usize,
        const __cuda_oxide_launch: usize,
        const __CudaModuleArg0: usize,
        const __CudaModuleArg1: usize,
    >(
        __cuda_oxide_args: &'__cuda_oxide_async [u32],
        output: &mut [u32],
    ) {
        let _ = (__cuda_oxide_args, output);
    }

    // This kernel keeps the prepared-launch generators honest after the
    // const-generic hygiene work. Its user names intentionally match every
    // private name used by the sync, borrowed-async, and owned-async paths.
    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn contracted_hygiene_probe<
        '__cuda_oxide_async,
        F: Copy + '__cuda_oxide_async,
        const __cuda_oxide_stream: usize,
        const __cuda_oxide_config: usize,
        const __cuda_oxide_prepared: usize,
        const __cuda_oxide_function: usize,
        const __cuda_oxide_args: usize,
        const __cuda_oxide_launch: usize,
        const __cuda_oxide_owned: usize,
        const __cuda_oxide_type_witness_0: usize,
    >(
        stream: &'__cuda_oxide_async [u32],
        config: &[u32],
        prepared: F,
    ) {
        let _ = (
            stream,
            config,
            prepared,
            __cuda_oxide_stream,
            __cuda_oxide_config,
            __cuda_oxide_prepared,
            __cuda_oxide_function,
            __cuda_oxide_args,
            __cuda_oxide_launch,
            __cuda_oxide_owned,
            __cuda_oxide_type_witness_0,
        );
    }

    #[kernel]
    pub unsafe fn unsafe_raw_pointer(raw: *mut f32) {
        let _ = raw;
    }

    /// `#[cooperative_launch]` routes every generated launch method through
    /// the cooperative driver entry points; this kernel pins that the
    /// generated sync, async, and owned-async methods still typecheck.
    #[kernel]
    #[cooperative_launch]
    pub fn cooperative_grid_sync(output: &mut [u32]) {
        let _ = output;
    }

    #[kernel]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        dynamic_shared = 0,
        min_compute_capability = (8, 0),
    )]
    pub fn contracted_read(input: &[u32]) {
        let _ = input;
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn contracted_closure<F: Fn(u32) -> u32 + Copy>(op: F, input: &[u32]) {
        let _ = (op, input);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = 0)]
    pub fn contracted_const<const N: usize>(input: &[u32]) {
        let _ = (N, input);
    }

    /// `requires` size requirements generate host-side checks in every
    /// checked launcher. This kernel pins that the generated sync launcher
    /// keeps its signature, the async launchers become fallible (relations
    /// are proven before enqueue), and the unchecked escape hatches still
    /// typecheck without the checks.
    #[kernel]
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        dynamic_shared = 0,
        requires = (input.len() >= n * 2, n >= 1),
    )]
    pub fn contracted_sizes(n: u32, input: &[u32]) {
        let _ = (n, input);
    }
}

#[cfg(feature = "async")]
fn assert_unit_operation<O: DeviceOperation<Output = ()>>(op: O) {
    let _ = op;
}

#[cfg(feature = "async")]
fn assert_owned_two_f32_buffers<O: DeviceOperation<Output = TwoF32Buffers>>(op: O) {
    let _ = op;
}

#[cfg(feature = "async")]
fn assert_owned_u32_buffer<O: DeviceOperation<Output = DeviceBox<[u32]>>>(op: O) {
    let _ = op;
}

#[cfg(feature = "async")]
fn assert_owned_unit<O: DeviceOperation<Output = ()>>(op: O) {
    let _ = op;
}

unsafe fn generated_methods_accept_kernel_scalar_types(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
    input: &DeviceBuffer<f32>,
    input_u32: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<f32>,
    output_u32: &mut DeviceBuffer<u32>,
) -> Result<(), cuda_core::DriverError> {
    let params = AffineParams {
        scale: 2.0,
        bias: 1.0,
    };
    let raw = core::ptr::null::<f32>();

    let raw_mut = core::ptr::null_mut::<f32>();
    let offset = 5u32;
    let op = move |x: u32| x + offset;

    // Raw configurations are deliberately an unsafe launch path: this
    // typechecking probe asserts that each configuration matches the kernel's
    // indexing and resource assumptions.
    unsafe {
        module.scalar_args(stream, config, 2.0, params, raw, input, output)?;
        module.copy_closure(stream, config, op, output_u32)?;
        module.const_only::<4>(stream, config, output_u32)?;
        module.mixed::<u32, 8>(stream, config, 7, output_u32)?;
        module.lifetime_mixed::<u32, 4>(stream, config, input_u32, output_u32)?;
        module.unsafe_raw_pointer(stream, config, raw_mut)?;
        module.cooperative_grid_sync(stream, config, output_u32)?;
    }

    Ok(())
}

fn generated_prepared_methods_bind_the_exact_kernel_and_specialization(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<u32>,
) -> Result<(), cuda_core::LaunchContractError> {
    let prepared = module.prepare_contracted_read(LaunchConfig1D::new(1, 64, 0))?;
    module.contracted_read(stream, &prepared, input)?;

    let offset = 7u32;
    let op = move |value: u32| value + offset;
    let generic = module.prepare_contracted_closure_for(&op, LaunchConfig1D::new(1, 64, 0))?;
    module.contracted_closure(stream, &generic, op, input)?;

    let const_generic = module.prepare_contracted_const::<4>(LaunchConfig1D::new(1, 64, 0))?;
    module.contracted_const::<4>(stream, &const_generic, input)?;

    // `requires` relations keep the sync launcher signature; violations
    // surface through the existing LaunchContractError return.
    let sizes = module.prepare_contracted_sizes(LaunchConfig1D::new(1, 64, 0))?;
    module.contracted_sizes(stream, &sizes, 4, input)?;

    unsafe {
        module.contracted_read_unchecked(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            input,
        )?;
        // The unchecked escape hatch skips the size checks and stays
        // on the plain DriverError path.
        module.contracted_sizes_unchecked(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            },
            4,
            input,
        )?;
    }
    Ok(())
}

#[cfg(feature = "async")]
unsafe fn generated_async_methods_accept_borrowed_buffers(
    module: &kernels::LoadedModule,
    config: LaunchConfig,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    async_input: &DeviceBox<[f32]>,
    async_output: &mut DeviceBox<[f32]>,
    async_output_u32: &mut DeviceBox<[u32]>,
) -> Result<(), cuda_core::DriverError> {
    let params = AffineParams {
        scale: 2.0,
        bias: 1.0,
    };
    let raw = core::ptr::null::<f32>();
    let raw_mut = core::ptr::null_mut::<f32>();

    let offset = 5u32;
    let offset_ref = &offset;
    let op = |x: u32| x + *offset_ref;

    // Finalizing a lazy operation with an unproved raw configuration is the
    // async unsafe boundary, just as submitting it is for synchronous launch.
    unsafe {
        let launch = module.scalar_args_async(config, 2.0, params, raw, input, output)?;
        assert_unit_operation(launch);

        let launch =
            module.scalar_args_async(config, 2.0, params, raw, async_input, async_output)?;
        assert_unit_operation(launch);

        let launch = module.copy_closure_async(config, op, async_output_u32)?;
        assert_unit_operation(launch);

        let launch = module.const_only_async::<4>(config, async_output_u32)?;
        assert_unit_operation(launch);

        let launch = module.mixed_async::<u32, 8>(config, 7, async_output_u32)?;
        assert_unit_operation(launch);

        let launch = module.unsafe_raw_pointer_async(config, raw_mut)?;
        assert_unit_operation(launch);

        let launch = module.cooperative_grid_sync_async(config, async_output_u32)?;
        assert_unit_operation(launch);
    }

    Ok(())
}

#[cfg(feature = "async")]
fn generated_prepared_async_methods_are_immutable_operations(
    module: &kernels::LoadedModule,
    input: &DeviceBuffer<u32>,
    async_input: &DeviceBox<[u32]>,
) -> Result<(), cuda_core::LaunchContractError> {
    let prepared = module.prepare_contracted_read(LaunchConfig1D::new(1, 64, 0))?;
    assert_unit_operation(module.contracted_read_async(&prepared, input));
    assert_unit_operation(module.contracted_read_async(&prepared, async_input));

    let offset = 7u32;
    let op = move |value: u32| value + offset;
    let generic = module.prepare_contracted_closure_for(&op, LaunchConfig1D::new(1, 64, 0))?;
    assert_unit_operation(module.contracted_closure_async(&generic, op, async_input));

    let const_generic = module.prepare_contracted_const::<4>(LaunchConfig1D::new(1, 64, 0))?;
    assert_unit_operation(module.contracted_const_async::<4>(&const_generic, async_input));

    // With `requires` relations the borrowed-async launcher is fallible:
    // every relation is proven before anything is enqueued.
    let sizes = module.prepare_contracted_sizes(LaunchConfig1D::new(1, 64, 0))?;
    assert_unit_operation(module.contracted_sizes_async(&sizes, 4, input)?);
    assert_unit_operation(module.contracted_sizes_async(&sizes, 4, async_input)?);
    Ok(())
}

#[cfg(feature = "async")]
unsafe fn generated_owned_async_methods_accept_owned_buffers(
    module: &kernels::LoadedModule,
    config: LaunchConfig,
    async_input: DeviceBox<[f32]>,
    async_output: DeviceBox<[f32]>,
    async_output_u32: DeviceBox<[u32]>,
    async_coop_output_u32: DeviceBox<[u32]>,
) -> Result<(), cuda_core::DriverError> {
    let params = AffineParams {
        scale: 2.0,
        bias: 1.0,
    };
    let raw = core::ptr::null::<f32>();
    let raw_mut = core::ptr::null_mut::<f32>();

    let offset = 5u32;
    let op = move |x: u32| x + offset;

    unsafe {
        let launch: cuda_host::OwnedAsyncKernelLaunch<TwoF32Buffers> =
            module.scalar_args_async_owned(config, 2.0, params, raw, async_input, async_output)?;
        assert_owned_two_f32_buffers(launch);

        let launch: cuda_host::OwnedAsyncKernelLaunch<DeviceBox<[u32]>> =
            module.copy_closure_async_owned(config, op, async_output_u32)?;
        assert_owned_u32_buffer(launch);

        let launch: cuda_host::OwnedAsyncKernelLaunch<()> =
            module.unsafe_raw_pointer_async_owned(config, raw_mut)?;
        assert_owned_unit(launch);

        let launch: cuda_host::OwnedAsyncKernelLaunch<DeviceBox<[u32]>> =
            module.cooperative_grid_sync_async_owned(config, async_coop_output_u32)?;
        assert_owned_u32_buffer(launch);
    }

    Ok(())
}

#[cfg(feature = "async")]
unsafe fn generated_owned_async_methods_forward_const_generics(
    module: &kernels::LoadedModule,
    config: LaunchConfig,
    const_output: DeviceBox<[u32]>,
    mixed_output: DeviceBox<[u32]>,
) -> Result<(), cuda_core::DriverError> {
    unsafe {
        let launch: cuda_host::OwnedAsyncKernelLaunch<DeviceBox<[u32]>> =
            module.const_only_async_owned::<4, _>(config, const_output)?;
        assert_owned_u32_buffer(launch);

        let launch: cuda_host::OwnedAsyncKernelLaunch<DeviceBox<[u32]>> =
            module.mixed_async_owned::<u32, 8, _>(config, 7, mixed_output)?;
        assert_owned_u32_buffer(launch);
    }

    Ok(())
}

#[cfg(feature = "async")]
fn generated_prepared_owned_async_methods_are_immutable_operations(
    module: &kernels::LoadedModule,
    async_input: DeviceBox<[u32]>,
    const_input: DeviceBox<[u32]>,
    sizes_input: DeviceBox<[u32]>,
) -> Result<(), cuda_core::LaunchContractError> {
    let prepared = module.prepare_contracted_read(LaunchConfig1D::new(1, 64, 0))?;
    let launch = module.contracted_read_async_owned(&prepared, async_input);
    assert_owned_u32_buffer(launch);

    let const_generic = module.prepare_contracted_const::<4>(LaunchConfig1D::new(1, 64, 0))?;
    let launch = module.contracted_const_async_owned::<4, _>(&const_generic, const_input);
    assert_owned_u32_buffer(launch);

    // With `requires` relations the owned-async launcher is fallible too;
    // relations are proven before any resource is moved into the launch.
    let sizes = module.prepare_contracted_sizes(LaunchConfig1D::new(1, 64, 0))?;
    let launch = module.contracted_sizes_async_owned(&sizes, 4, sizes_input)?;
    assert_owned_u32_buffer(launch);
    Ok(())
}

#[test]
fn generated_cuda_module_api_typechecks() {
    let _ = generated_methods_accept_kernel_scalar_types;
    let _ = generated_prepared_methods_bind_the_exact_kernel_and_specialization;
    #[cfg(feature = "async")]
    let _ = generated_async_methods_accept_borrowed_buffers;
    #[cfg(feature = "async")]
    let _ = generated_prepared_async_methods_are_immutable_operations;
    #[cfg(feature = "async")]
    let _ = generated_owned_async_methods_accept_owned_buffers;
    #[cfg(feature = "async")]
    let _ = generated_owned_async_methods_forward_const_generics;
    #[cfg(feature = "async")]
    let _ = generated_prepared_owned_async_methods_are_immutable_operations;
}

#[test]
fn concrete_kernel_signature_matches_typed_launcher_arguments() {
    use cuda_host::{CudaKernelArgumentKind as Argument, CudaKernelScalarKind as Scalar};

    let signature = kernels::scalar_args_CUDA_SIGNATURE;
    assert_eq!(signature.entry, "scalar_args");
    assert_eq!(signature.arguments.len(), 5);
    assert_eq!(signature.arguments[0].name, "scale");
    assert_eq!(
        signature.arguments[0].kind,
        Argument::Scalar(Scalar::Opaque {
            size: 4,
            alignment: 4
        })
    );
    assert_eq!(signature.arguments[1].name, "params");
    assert_eq!(
        signature.arguments[1].kind,
        Argument::Scalar(Scalar::Opaque {
            size: core::mem::size_of::<AffineParams>(),
            alignment: core::mem::align_of::<AffineParams>(),
        })
    );
    assert_eq!(signature.arguments[2].kind, Argument::DevicePointer);
    assert_eq!(
        signature.arguments[3].kind,
        Argument::DeviceSlice {
            writable: false,
            row_width: false,
        }
    );
    assert_eq!(
        signature.arguments[4].kind,
        Argument::DeviceSlice {
            writable: true,
            row_width: false,
        }
    );
}

// =============================================================================
// PTX naming contract
//
// These tests pin down the shape of the host-side `GenericCudaKernel::ptx_name`
// output. The const_generic and cross_crate_kernel executable verifiers compare
// these host names with actual backend-generated PTX entries.
//
// On-wire shape: `<base>_TID_<hex32>`. The single `<hex32>` hashes the concrete
// kernel function-item type, so it includes ordered type and const arguments
// while staying fixed-length regardless of generic arity.
// =============================================================================

fn is_lowercase_hex_32(s: &str) -> bool {
    s.len() == 32
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn split_tid_name<'a>(name: &'a str, base: &str) -> &'a str {
    let expected_prefix = format!("{base}_TID_");
    name.strip_prefix(&expected_prefix)
        .unwrap_or_else(|| panic!("expected `{name}` to start with `{expected_prefix}`"))
}

#[test]
fn ptx_name_for_closure_generic_matches_tid_scheme() {
    let offset = 5u32;
    let op = move |x: u32| x + offset;
    fn name_for<F: Fn(u32) -> u32 + Copy>(_f: F) -> &'static str {
        kernels::copy_closure_ptx_name::<F>()
    }

    let name = name_for(op);
    let hex = split_tid_name(name, "copy_closure");
    assert!(
        is_lowercase_hex_32(hex),
        "expected `<base>_TID_<32hex>`; got `{name}` (suffix `{hex}`)"
    );
}

#[test]
fn ptx_name_is_stable_per_closure_type() {
    let offset = 7u32;
    let op = move |x: u32| x + offset;
    fn name_for<F: Fn(u32) -> u32 + Copy>(_f: F) -> &'static str {
        kernels::copy_closure_ptx_name::<F>()
    }
    let a = name_for(op);
    let b = name_for(op);
    assert_eq!(a, b, "same closure type must produce the same PTX name");
}

#[test]
fn ptx_name_separates_distinct_closure_types() {
    let factor = 2u32;
    let op1 = move |x: u32| x + factor;
    let op2 = move |x: u32| x * factor;
    fn name_for<F: Fn(u32) -> u32 + Copy>(_f: F) -> &'static str {
        kernels::copy_closure_ptx_name::<F>()
    }
    let n1 = name_for(op1);
    let n2 = name_for(op2);
    assert_ne!(
        n1, n2,
        "two distinct closure literals must produce different PTX names ({n1} vs {n2})"
    );
}

#[test]
fn ptx_name_separates_const_specializations() {
    let n0 = kernels::const_only_ptx_name::<0>();
    let n4 = kernels::const_only_ptx_name::<4>();
    let n8 = kernels::const_only_ptx_name::<8>();

    assert_ne!(n0, n4, "zero must remain a distinct const value");
    assert_ne!(n4, n8, "const values must participate in kernel identity");
    assert!(is_lowercase_hex_32(split_tid_name(n0, "const_only")));
    assert!(is_lowercase_hex_32(split_tid_name(n4, "const_only")));
    assert!(is_lowercase_hex_32(split_tid_name(n8, "const_only")));
}

#[test]
fn generic_kernel_marker_remains_compatible() {
    use cuda_host::GenericCudaKernel;

    let direct = <kernels::__const_only_CudaKernel<4> as GenericCudaKernel>::ptx_name();
    assert_eq!(direct, kernels::const_only_ptx_name::<4>());
}

#[test]
fn ptx_name_separates_mixed_specializations() {
    let u32_n4 = kernels::mixed_ptx_name::<u32, 4>();
    let u32_n8 = kernels::mixed_ptx_name::<u32, 8>();
    let i32_n4 = kernels::mixed_ptx_name::<i32, 4>();

    assert_ne!(u32_n4, u32_n8);
    assert_ne!(u32_n4, i32_n4);
}

#[test]
fn ptx_name_helper_preserves_lifetime_bounds() {
    fn name_for<'a, T: Copy + 'a, const N: usize>(_input: &'a [T]) -> &'static str {
        kernels::lifetime_mixed_ptx_name::<T, N>()
    }

    let input = [1u32, 2, 3];
    let name = name_for::<u32, 4>(&input);
    assert!(is_lowercase_hex_32(split_tid_name(name, "lifetime_mixed")));
}

#[test]
fn generated_names_do_not_capture_user_const_generics() {
    let name = kernels::hygiene_probe_ptx_name::<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>();
    assert!(is_lowercase_hex_32(split_tid_name(name, "hygiene_probe")));
}

#[allow(non_camel_case_types, non_snake_case)]
#[cuda_module]
mod signature_regressions {
    use super::*;
    type f32 = ::core::primitive::u32;
    type Count = ::core::primitive::u64;

    #[kernel]
    pub fn step<'a>(shadow: f32, alias: Count, tag: core::marker::PhantomData<&'a ()>) {
        let _ = (shadow, alias, tag);
    }

    #[kernel]
    pub fn STEP(value: ::core::primitive::f32, count: ::std::primitive::u32) {
        let _ = (value, count);
    }

    #[kernel]
    pub fn r#type(value: ::core::primitive::bool) {
        let _ = value;
    }

    #[cfg(any())]
    #[kernel]
    pub fn disabled(value: MissingType) {}

    pub mod nested {
        use super::*;
        #[kernel]
        pub fn nested_step(value: ::core::primitive::i32) {
            let _ = value;
        }
    }
}

#[test]
fn signature_metadata_preserves_aliases_lifetimes_case_and_scope() {
    use cuda_host::{CudaKernelArgumentKind as Argument, CudaKernelScalarKind as Scalar};
    let lower = signature_regressions::step_CUDA_SIGNATURE;
    let upper = signature_regressions::STEP_CUDA_SIGNATURE;
    assert_ne!(lower.entry, upper.entry);
    assert_eq!(
        lower.arguments[0].kind,
        Argument::Scalar(Scalar::Opaque {
            size: 4,
            alignment: 4
        })
    );
    assert_eq!(
        lower.arguments[1].kind,
        Argument::Scalar(Scalar::Opaque {
            size: 8,
            alignment: 8
        })
    );
    assert_eq!(
        lower.arguments[2].kind,
        Argument::Scalar(Scalar::Opaque {
            size: 0,
            alignment: 1
        })
    );
    assert_eq!(upper.arguments[0].kind, Argument::Scalar(Scalar::F32));
    assert_eq!(upper.arguments[1].kind, Argument::Scalar(Scalar::U32));
    assert_eq!(
        signature_regressions::type_CUDA_SIGNATURE.arguments[0].kind,
        Argument::Scalar(Scalar::Bool)
    );
    let nested = signature_regressions::nested::nested_step_CUDA_SIGNATURE;
    assert_ne!(nested.entry, lower.entry);
    assert_eq!(nested.arguments[0].kind, Argument::Scalar(Scalar::I32));
}
