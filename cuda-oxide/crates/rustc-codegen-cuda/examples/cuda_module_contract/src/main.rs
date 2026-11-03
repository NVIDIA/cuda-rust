/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused `#[cuda_module]` host ABI contract test.
//!
//! The kernel intentionally mixes common host-side argument shapes the typed
//! module macro must lower correctly: scalars, slice, raw device pointer, and
//! `DisjointSlice` output.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct GridConstants {
    values: [u32; 32],
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn ordinary_shared_owner(value: u32) {
        let shared = DynamicSharedArray::<u32, 16>::get();
        unsafe {
            core::ptr::write_volatile(shared, value);
        }
    }

    #[inline(never)]
    fn ordinary_shared_forward(value: u32) {
        ordinary_shared_owner(value);
    }

    /// Two entries share the same transitive helper. The helper's single PTX
    /// declaration must use the stronger contract from either caller.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        dynamic_shared = 128,
        dynamic_shared_alignment = 32,
    )]
    pub fn helper_contract_32(value: u32) {
        ordinary_shared_forward(value);
    }

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        dynamic_shared = 128,
        dynamic_shared_alignment = 256,
    )]
    pub fn helper_contract_256(value: u32) {
        ordinary_shared_forward(value);
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mixed_abi(
        scale: f32,
        bias: f32,
        extra: f32,
        input: &[f32],
        raw_offsets: *const f32,
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = output.get_mut(idx) {
            let offset = unsafe { *raw_offsets.add(idx_raw) };
            *out_elem = input[idx_raw] * scale + bias + extra + offset;
        }
    }

    /// One source declaration drives both sides of the launch ABI: device code
    /// receives a read-only parameter-space reference while the generated host
    /// method accepts and marshals the 128-byte value directly.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn grid_constant_read(
        mut output: DisjointSlice<u32>,
        #[grid_constant] constants: &GridConstants,
    ) {
        let index = thread::index_1d();
        let linear = index.get();
        if let Some(output) = output.get_mut(index) {
            *output = constants.values[linear];
        }
    }

    /// Generic grid-constant parameters keep their by-value entry ABI without
    /// leaking that ABI into the callable generic helper.
    ///
    /// # Safety
    /// `output` must contain 32 writable elements, owned by this block.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub unsafe fn generic_grid_constant<T: Copy>(
        #[grid_constant] constants: &GridConstants,
        output: *mut u32,
        _tag: T,
    ) {
        let index = thread::threadIdx_x() as usize;
        unsafe {
            *output.add(index) = constants.values[index];
        }
    }

    /// Calls the generic helper without opting this entry into grid-constant
    /// ABI. Its first parameter must remain one ordinary device pointer.
    ///
    /// # Safety
    /// `constants` must point to a valid, immutable `GridConstants` value in
    /// device memory for the duration of the launch. `output` must contain
    /// 32 writable elements, owned by this block.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub unsafe fn ordinary_calls_generic(constants: *const GridConstants, output: *mut u32) {
        unsafe { generic_grid_constant::<u8>(&*constants, output, 0) };
    }

    #[inline(never)]
    fn read_grid_constant(constants: &GridConstants, index: usize) -> u32 {
        constants.values[index]
    }

    /// Two descriptors after a dropped ZST and a flattened slice. Recording
    /// both addresses across two blocks also detects per-thread local copies.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn mixed_grid_constants(
        _empty: (),
        input: &[u32],
        #[grid_constant] first: &GridConstants,
        _other_empty: (),
        #[grid_constant] second: &GridConstants,
        mut output: DisjointSlice<u32>,
        mut first_addresses: DisjointSlice<u64>,
        mut second_addresses: DisjointSlice<u64>,
    ) {
        let index = thread::index_1d();
        let linear = index.get();
        if let Some(output) = output.get_mut(index) {
            *output = input[linear]
                .wrapping_add(read_grid_constant(first, linear % 32))
                .wrapping_add(read_grid_constant(second, 31 - linear % 32));
        }
        if let Some(address) = first_addresses.get_mut(thread::index_1d()) {
            *address = first as *const GridConstants as u64;
        }
        if let Some(address) = second_addresses.get_mut(thread::index_1d()) {
            *address = second as *const GridConstants as u64;
        }
    }

    /// Size requirements: the generated checked launchers prove every
    /// `requires` relation on the CPU before marshalling, so an undersized
    /// buffer becomes a typed `LaunchContractError` instead of a device
    /// fault. Evaluation is overflow-safe: operands widen to u64 and the
    /// arithmetic uses checked ops. The `_unchecked` escape hatch skips
    /// these checks just as it skips the geometry checks.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= n * stride, output.len() >= n),
    )]
    pub fn strided_scale(n: usize, stride: usize, input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i < n
            && let Some(out) = output.get_mut(index)
        {
            *out = input[i * stride] * 2.0;
        }
    }

    /// Compile-time proof that a contract alignment is merged with alignment
    /// requested by the body. The body asks for 16 bytes; the contract raises
    /// the emitted extern-shared declaration to 128 bytes.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 1024,
        dynamic_shared_alignment = 128,
    )]
    pub fn aligned_dynamic_shared(mut output: DisjointSlice<u8>) {
        let index = thread::index_1d();
        let linear = index.get();
        let shared = DynamicSharedArray::<u8, 16>::get_raw();
        unsafe {
            *shared.add(thread::threadIdx_x() as usize) = linear as u8;
        }
        if let Some(output) = output.get_mut(index) {
            *output = unsafe { *shared.add(thread::threadIdx_x() as usize) };
        }
    }

    /// Generic/closure pin: the prepared brand and compiler-side alignment
    /// marker must both survive monomorphization onto the exported wrapper.
    // Deliberately put both configuration attributes above #[kernel]. They
    // expand into body markers before the generic entry wrapper is generated.
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        dynamic_shared = 256,
        dynamic_shared_alignment = 64,
    )]
    #[launch_bounds(64)]
    #[kernel]
    pub fn generic_aligned<F: Fn(u32) -> u32 + Copy>(op: F, mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let linear = index.get();
        let shared = DynamicSharedArray::<u32, 16>::get();
        unsafe {
            *shared.add(thread::threadIdx_x() as usize) = op(linear as u32);
        }
        if let Some(output) = output.get_mut(index) {
            *output = unsafe { *shared.add(thread::threadIdx_x() as usize) };
        }
    }
}

/// Compile-only coverage for `#[kernel(Type)]`, the explicit-instantiation
/// form. Its concrete entry still calls a generic helper, so the entry's
/// alignment contract must propagate to the helper that owns shared memory.
mod explicit_instantiation {
    use super::*;

    // The explicit-instantiation expansion must forward pre-expanded markers
    // just like the call-site monomorphization path above.
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 128,
        dynamic_shared_alignment = 32,
    )]
    #[kernel(u32, launch_context = launch_context)]
    pub fn explicit_aligned<T: Copy>(value: T) {
        let _index = thread::index_1d_u32(launch_context);
        let shared = DynamicSharedArray::<T, 8>::get();
        unsafe {
            core::ptr::write_volatile(shared, value);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        return verify_launch_contract_ptx();
    }

    println!("=== cuda_module ABI Contract Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this example has one device-code owner and all specializations
    // are instantiated in this same crate. Either loading route therefore
    // produces exactly the entries described by the generated host API.
    let module = unsafe {
        if std::env::args().any(|argument| argument == "--load-nvvm") {
            // The generic module's default loader merges PTX bundles. Select
            // this crate's single bundle explicitly to exercise libNVVM IR.
            kernels::from_module(cuda_host::load_embedded_module(
                &ctx,
                "cuda_module_contract",
            )?)?
        } else {
            kernels::load(&ctx)?
        }
    };

    const N: usize = 1024;
    let scale = 1.5f32;
    let bias = 2.0f32;
    let extra = 7.0f32;
    let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let offset_host: Vec<f32> = (0..N).map(|i| (i % 5) as f32).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
    let offset_dev = DeviceBuffer::from_host(&stream, &offset_host)?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let launch = module.prepare_mixed_abi(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;

    module.mixed_abi(
        &stream,
        &launch,
        scale,
        bias,
        extra,
        &input_dev,
        offset_dev.cu_deviceptr() as *const f32,
        &mut output_dev,
    )?;

    let output = output_dev.to_host_vec(&stream)?;
    let errors = (0..N)
        .filter(|&i| {
            let expected = input_host[i] * scale + bias + extra + offset_host[i];
            (output[i] - expected).abs() > 1e-5
        })
        .count();

    assert_eq!(errors, 0, "mixed ABI kernel produced {errors} errors");

    let constants = GridConstants {
        values: core::array::from_fn(|index| 0x600d_0000 | index as u32),
    };
    let mut constants_output = DeviceBuffer::<u32>::zeroed(&stream, constants.values.len())?;
    let constants_launch = module.prepare_grid_constant_read(LaunchConfig1D::new(1, 32, 0))?;
    // SAFETY: GridConstants contains only initialized u32 values, with no
    // nested pointers or references. The output covers all 32 threads and
    // stays alive until this synchronous launcher completes.
    unsafe {
        module.grid_constant_read(&stream, &constants_launch, &mut constants_output, constants)?;
    }
    assert_eq!(constants_output.to_host_vec(&stream)?, constants.values);

    let generic_constant_output = DeviceBuffer::<u32>::zeroed(&stream, 32)?;
    let tag = 0_u8;
    let generic_constant_launch =
        module.prepare_generic_grid_constant_for(&tag, LaunchConfig1D::new(1, 32, 0))?;
    // SAFETY: the copied constants and tag contain only initialized integers.
    // Exactly one block writes its 32 distinct output elements; the device
    // allocation remains alive until the synchronous launcher completes.
    unsafe {
        module.generic_grid_constant(
            &stream,
            &generic_constant_launch,
            constants,
            generic_constant_output.cu_deviceptr() as *mut u32,
            tag,
        )?;
    }
    assert_eq!(
        generic_constant_output.to_host_vec(&stream)?,
        constants.values
    );

    // Calling the generic helper from an ordinary pointer entry must preserve
    // the pointer ABI. The descriptor remains a separate device allocation.
    let device_constants = DeviceBuffer::from_host(&stream, &constants.values)?;
    assert_eq!(
        device_constants.cu_deviceptr() % align_of::<GridConstants>() as u64,
        0
    );
    let ordinary_launch = module.prepare_ordinary_calls_generic(LaunchConfig1D::new(1, 32, 0))?;
    // SAFETY: device_constants contains one initialized descriptor and remains
    // alive until the synchronous generated launcher completes.
    unsafe {
        module.ordinary_calls_generic(
            &stream,
            &ordinary_launch,
            device_constants.cu_deviceptr() as *const GridConstants,
            generic_constant_output.cu_deviceptr() as *mut u32,
        )?;
    }
    assert_eq!(
        generic_constant_output.to_host_vec(&stream)?,
        constants.values
    );

    let second = GridConstants {
        values: core::array::from_fn(|index| 0x1200_0000 | (index as u32 * 7)),
    };
    let mixed_input: Vec<u32> = (0..64).map(|index| index * 11).collect();
    let mixed_device_input = DeviceBuffer::from_host(&stream, &mixed_input)?;
    let mut mixed_output = DeviceBuffer::<u32>::zeroed(&stream, 64)?;
    let mut first_addresses = DeviceBuffer::<u64>::zeroed(&stream, 64)?;
    let mut second_addresses = DeviceBuffer::<u64>::zeroed(&stream, 64)?;
    let mixed_launch = module.prepare_mixed_grid_constants(LaunchConfig1D::new(2, 32, 0))?;
    // SAFETY: both copied descriptors contain only initialized integers.
    // The input and disjoint output allocations each cover all 64 threads
    // and remain alive until this synchronous launcher completes.
    unsafe {
        module.mixed_grid_constants(
            &stream,
            &mixed_launch,
            (),
            &mixed_device_input,
            constants,
            (),
            second,
            &mut mixed_output,
            &mut first_addresses,
            &mut second_addresses,
        )?;
    }
    let expected: Vec<u32> = mixed_input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .wrapping_add(constants.values[index % 32])
                .wrapping_add(second.values[31 - index % 32])
        })
        .collect();
    assert_eq!(mixed_output.to_host_vec(&stream)?, expected);
    let first_addresses = first_addresses.to_host_vec(&stream)?;
    let second_addresses = second_addresses.to_host_vec(&stream)?;
    assert_ne!(first_addresses[0], second_addresses[0]);
    for addresses in [first_addresses, second_addresses] {
        assert_ne!(addresses[0], 0);
        assert_eq!(addresses[0] % 64, 0);
        assert!(addresses.iter().all(|address| *address == addresses[0]));
    }

    let mut generic_output = DeviceBuffer::<u32>::zeroed(&stream, N)?;
    let add_three = |value: u32| value + 3;
    let generic_launch = module.prepare_generic_aligned_for(
        &add_three,
        LaunchConfig1D::new((N as u32).div_ceil(64), 64, 256),
    )?;
    module.generic_aligned(&stream, &generic_launch, add_three, &mut generic_output)?;
    let generic_output = generic_output.to_host_vec(&stream)?;
    assert!(
        generic_output
            .iter()
            .enumerate()
            .all(|(index, &value)| value == index as u32 + 3),
        "generic prepared launch produced an unexpected value",
    );

    // --- Size requirements (`requires`) ---
    let n: usize = 256;
    let stride: usize = 2;
    let strided_input: Vec<f32> = (0..n * stride).map(|i| i as f32).collect();
    let strided_input_dev = DeviceBuffer::from_host(&stream, &strided_input)?;
    let mut strided_output_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let strided_launch =
        module.prepare_strided_scale(LaunchConfig1D::new((n as u32).div_ceil(128), 128, 0))?;

    // (a) Buffers satisfying every relation launch normally.
    module.strided_scale(
        &stream,
        &strided_launch,
        n,
        stride,
        &strided_input_dev,
        &mut strided_output_dev,
    )?;
    let strided_output = strided_output_dev.to_host_vec(&stream)?;
    assert!(
        strided_output
            .iter()
            .enumerate()
            .all(|(i, &value)| value == (i * stride) as f32 * 2.0),
        "strided scale produced an unexpected value",
    );

    // (b) An undersized buffer fails fast on the CPU: the launcher returns a
    // typed error carrying the violated relation's source text and both
    // evaluated sides, and nothing reaches the GPU.
    let undersized_dev = DeviceBuffer::from_host(&stream, &strided_input[..64])?;
    let violation = module.strided_scale(
        &stream,
        &strided_launch,
        n,
        stride,
        &undersized_dev,
        &mut strided_output_dev,
    );
    match violation {
        Err(
            error @ cuda_core::LaunchContractError::SizeRequirementViolated {
                relation,
                lhs,
                rhs,
                ..
            },
        ) => {
            println!("rejected undersized launch on the CPU: {error}");
            assert_eq!(relation, "input.len() >= n * stride");
            assert_eq!(lhs, 64);
            assert_eq!(rhs, 512);
        }
        other => panic!("expected SizeRequirementViolated, got {other:?}"),
    }

    // (c) Relation arithmetic is overflow-safe: an operand product leaving
    // the u64 range is its own typed error, not a wrapped comparison.
    let overflow = module.strided_scale(
        &stream,
        &strided_launch,
        usize::MAX,
        stride,
        &strided_input_dev,
        &mut strided_output_dev,
    );
    match overflow {
        Err(error @ cuda_core::LaunchContractError::SizeRequirementOverflow { relation, .. }) => {
            println!("rejected overflowing relation on the CPU: {error}");
            assert_eq!(relation, "input.len() >= n * stride");
        }
        other => panic!("expected SizeRequirementOverflow, got {other:?}"),
    }

    // The two rejected launches left the stream healthy; a valid launch
    // still succeeds afterwards.
    module.strided_scale(
        &stream,
        &strided_launch,
        n,
        stride,
        &strided_input_dev,
        &mut strided_output_dev,
    )?;
    stream.synchronize()?;

    println!("SUCCESS: mixed ABI typed launch passed");
    Ok(())
}

fn verify_launch_contract_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda_module_contract.ptx");
    let ptx = std::fs::read_to_string(&ptx_path)?;
    let document = ptx_parse::Document::parse(&ptx)?;
    let aligned_symbol = ".extern .shared .align 128 .b8 __dynamic_smem_aligned_dynamic_shared[];";
    if !ptx.contains(aligned_symbol) {
        return Err(format!(
            "{} does not contain the contract-enforced dynamic shared-memory alignment",
            ptx_path.display()
        )
        .into());
    }
    if !ptx.lines().any(|line| {
        line.contains(".extern .shared .align 64 .b8 __dynamic_smem_")
            && line.contains("generic_aligned")
    }) {
        return Err(
            "generic launch contract alignment did not reach its PTX specialization".into(),
        );
    }
    if !ptx.lines().any(|line| {
        line.contains(".extern .shared .align 32 .b8 __dynamic_smem_")
            && line.contains("explicit_aligned")
    }) {
        return Err("explicit generic instantiation alignment did not reach its PTX helper".into());
    }
    if !ptx.lines().any(|line| {
        line.contains(".extern .shared .align 256 .b8 __dynamic_smem_")
            && line.contains("ordinary_shared_owner")
    }) {
        return Err(
            "shared ordinary helper did not receive the strongest calling-kernel alignment".into(),
        );
    }

    // A kernel declaring an exact `block` carries that shape into PTX as
    // `.reqntid`, which the driver enforces on every axis. Every kernel in
    // this example declares one, including the generic and the explicitly
    // instantiated kernels whose contracts expand before `#[kernel]` and
    // reach the entry wrapper through marker forwarding.
    for (entry, geometry) in [
        ("aligned_dynamic_shared", ".reqntid 256, 1, 1"),
        ("mixed_abi", ".reqntid 256, 1, 1"),
        ("grid_constant_read", ".reqntid 32, 1, 1"),
        ("mixed_grid_constants", ".reqntid 32, 1, 1"),
        ("ordinary_calls_generic", ".reqntid 32, 1, 1"),
        ("strided_scale", ".reqntid 128, 1, 1"),
        ("helper_contract_32", ".reqntid 32, 1, 1"),
        ("helper_contract_256", ".reqntid 32, 1, 1"),
        ("explicit_aligned_u32", ".reqntid 32, 1, 1"),
    ] {
        verify_entry_geometry(&document, entry, false, entry, geometry)?;
    }

    verify_entry_geometry(
        &document,
        "generic_aligned_TID_",
        true,
        "generic_aligned specialization",
        ".reqntid 64, 1, 1",
    )?;
    verify_entry_geometry(
        &document,
        "generic_grid_constant_TID_",
        true,
        "generic grid-constant specialization",
        ".reqntid 32, 1, 1",
    )?;

    let generic = document
        .callables()
        .iter()
        .find(|callable| {
            callable.body_text().is_some()
                && callable.kind() == ptx_parse::CallableKind::Entry
                && callable.name().starts_with("generic_grid_constant_TID_")
        })
        .ok_or("missing generic grid-constant specialization")?
        .definition_header_text()
        .ok_or("generic grid-constant entry has no complete header")?;
    let generic_params = ptx_parameters(generic);
    if generic_params.len() != 3 || !is_descriptor_parameter(&generic_params[0]) {
        return Err("generic grid-constant specialization lost its by-value parameter ABI".into());
    }

    let ordinary = document
        .callables()
        .iter()
        .find(|callable| {
            callable.body_text().is_some()
                && callable.kind() == ptx_parse::CallableKind::Entry
                && callable.name() == "ordinary_calls_generic"
        })
        .ok_or("missing ordinary generic-helper caller")?
        .definition_header_text()
        .ok_or("ordinary helper caller has no complete header")?;
    let ordinary_params = ptx_parameters(ordinary);
    if ordinary_params.len() != 2
        || !ordinary_params
            .iter()
            .all(|parameter| parameter.starts_with(".param .u64 "))
    {
        return Err("grid-constant ABI leaked into an ordinary helper caller".into());
    }

    let mixed = document
        .callables()
        .iter()
        .find(|callable| {
            callable.kind() == ptx_parse::CallableKind::Entry
                && callable.name() == "mixed_grid_constants"
                && callable.body_text().is_some()
        })
        .ok_or("missing mixed grid-constant entry")?;
    let mixed_params = ptx_parameters(
        mixed
            .definition_header_text()
            .ok_or("missing mixed entry header")?,
    );
    if mixed_params.len() != 10
        || !mixed_params.iter().enumerate().all(|(index, parameter)| {
            if index == 2 || index == 3 {
                is_descriptor_parameter(parameter)
            } else {
                parameter.starts_with(".param .u64 ")
            }
        })
    {
        return Err(format!(
            "mixed grid-constant parameter positions/layout changed: {mixed_params:?}"
        )
        .into());
    }

    println!("SUCCESS: prepared-launch PTX contract verified");
    Ok(())
}

fn ptx_parameters(header: &str) -> Vec<String> {
    header
        .lines()
        .filter_map(|line| {
            let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
            line.starts_with(".param ")
                .then(|| line.trim_end_matches(',').to_owned())
        })
        .collect()
}

fn is_descriptor_parameter(parameter: &str) -> bool {
    parameter.starts_with(".param .align 64 .b8 ") && parameter.ends_with("[128]")
}

/// Assert one entry's launch geometry, and that it declares exactly one of the
/// two mutually exclusive directives.
///
/// ptxas rejects an entry carrying both `.maxntid` and `.reqntid`
/// ("Conflicting directives: .maxntid and .reqntid cannot both be specified"),
/// so a kernel with an exact block emits `.reqntid` in place of the thread
/// maximum. Every kernel here declares `#[launch_bounds]`, so this is the case
/// that would regress if the exporter stopped suppressing one of them.
fn verify_entry_geometry(
    document: &ptx_parse::Document<'_>,
    symbol: &str,
    match_prefix: bool,
    entry: &str,
    expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let definition = document
        .callables()
        .iter()
        .find(|callable| {
            callable.body_text().is_some()
                && callable.kind() == ptx_parse::CallableKind::Entry
                && if match_prefix {
                    callable.name().starts_with(symbol)
                } else {
                    callable.name() == symbol
                }
        })
        .ok_or_else(|| format!("missing or incomplete PTX entry {entry}"))?;
    let body = definition.text();

    if !body.contains(expected) {
        return Err(format!("PTX entry {entry} lost its launch geometry `{expected}`").into());
    }
    if body.contains(".maxntid") && body.contains(".reqntid") {
        return Err(format!(
            "PTX entry {entry} declares both .maxntid and .reqntid, which ptxas rejects"
        )
        .into());
    }

    Ok(())
}
