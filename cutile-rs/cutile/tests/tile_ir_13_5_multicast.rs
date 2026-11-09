/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Two-GPU NVLink SHARP execution coverage. Run explicitly with --ignored.
//! The compile test is driver-free; the execution test fails if prerequisites
//! are absent, so a machine without NVLS cannot report a false pass.
use cuda_core::{vmm, Context, IntoResult};
use cutile::compile_api::KernelCompiler;
use cutile::prelude::*;
use cutile_ir::bytecode::BytecodeVersion;
mod common;
#[path = "support/tile_ir_reference.rs"]
mod reference;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    use cutile::tileir::*;
    #[cutile::entry()]
    unsafe fn nvls_load_reduce(input: *mut i32, output: *mut i32) {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut i32, { [] }> = unsafe { ptr_with_attr_multicast(base) };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[1] };
        let tensor: Tensor<i32, { [4] }> =
            unsafe { make_tensor_view(base, shape![4], strides, new_token_unordered()) };
        let view: Partition<i32, { [4] }> = unsafe {
            make_partition_view(
                &tensor,
                shape![4],
                padding::None,
                dim_map::Identity,
                new_token_unordered(),
            )
        };
        let (value, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
            multimem_load_reduce_view_tko(
                &view,
                [0i32],
                multimem::Add,
                ordering::Acquire,
                scope::System,
                None,
                [true],
                shape![4],
            )
        };
        let output_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(output);
        let output_tensor: Tensor<i32, { [4] }> =
            unsafe { make_tensor_view(output_base, shape![4], strides, loaded) };
        let output_view: Partition<i32, { [4] }> = unsafe {
            make_partition_view(
                &output_tensor,
                shape![4],
                padding::None,
                dim_map::Identity,
                loaded,
            )
        };
        let _stored = unsafe {
            store_view_raw(
                &output_view,
                value,
                [0i32],
                Some(loaded),
                ordering::Release,
                scope::System,
                tma::Disabled,
                None,
                [true],
            )
        };
    }
    #[cutile::entry()]
    unsafe fn nvls_reduce(input: *mut i32) {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut i32, { [] }> = unsafe { ptr_with_attr_multicast(base) };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[1] };
        let tensor: Tensor<i32, { [4] }> =
            unsafe { make_tensor_view(base, shape![4], strides, new_token_unordered()) };
        let view: Partition<i32, { [4] }> = unsafe {
            make_partition_view(
                &tensor,
                shape![4],
                padding::None,
                dim_map::Identity,
                new_token_unordered(),
            )
        };
        let value: Tile<i32, { [4] }> = constant(5, shape![4]);
        let _reduced = unsafe {
            multimem_reduce_view_tko(
                &view,
                value,
                [0i32],
                multimem::Add,
                ordering::Release,
                scope::System,
                None,
                [true],
            )
        };
    }
    #[cutile::entry()]
    unsafe fn nvls_store(input: *mut i32) {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut i32, { [] }> = unsafe { ptr_with_attr_multicast(base) };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[1] };
        let tensor: Tensor<i32, { [4] }> =
            unsafe { make_tensor_view(base, shape![4], strides, new_token_unordered()) };
        let view: Partition<i32, { [4] }> = unsafe {
            make_partition_view(
                &tensor,
                shape![4],
                padding::None,
                dim_map::Identity,
                new_token_unordered(),
            )
        };
        let value: Tile<i32, { [4] }> = constant(37, shape![4]);
        let _stored = unsafe {
            multimem_store_view_tko(
                &view,
                value,
                [0i32],
                ordering::Release,
                scope::System,
                None,
                [true],
            )
        };
    }
}

#[test]
fn compile_multicast_execution_kernels() {
    common::with_test_stack(|| {
        for kernel in ["nvls_load_reduce", "nvls_reduce", "nvls_store"] {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            reference::verify_if_configured(&artifacts.bytecode().unwrap());
        }
    });
}

fn copy_to_device(address: u64, data: &[i32; 4]) {
    unsafe {
        cuda_core::sys::cuMemcpyHtoD_v2(address, data.as_ptr().cast(), std::mem::size_of_val(data))
            .result()
            .unwrap();
    }
}
fn copy_from_device(address: u64) -> [i32; 4] {
    let mut data = [0i32; 4];
    unsafe {
        cuda_core::sys::cuMemcpyDtoH_v2(
            data.as_mut_ptr().cast(),
            address,
            std::mem::size_of_val(&data),
        )
        .result()
        .unwrap();
    }
    data
}

#[test]
#[ignore = "requires CUDA 13.5 and two GPUs on an NVLink SHARP-capable fabric"]
fn multicast_load_reduce_and_store_execute_from_each_gpu() {
    common::with_test_stack(|| {
        let contexts = [
            Context::new(0).expect("GPU 0 required"),
            Context::new(1).expect("GPU 1 required"),
        ];
        let devices = contexts.each_ref().map(|ctx| ctx.cu_device());
        for &dev in &devices {
            assert!(
                vmm::multicast_supported(dev).unwrap(),
                "NVLink SHARP multicast required"
            );
        }
        let granularity =
            vmm::multicast_granularity(2, 1 << 20, vmm::MulticastGranularity::Recommended).unwrap();
        let size = vmm::align_size(1 << 20, granularity);
        let team = vmm::MulticastObject::new(2, size).unwrap();
        for &dev in &devices {
            team.add_device(dev).unwrap();
        }
        let mut physical = Vec::new();
        let mut bindings = Vec::new();
        for (ctx, &dev) in contexts.iter().zip(&devices) {
            ctx.bind_to_thread().unwrap();
            let bytes = vmm::align_size(size, vmm::allocation_granularity(dev).unwrap());
            let allocation = vmm::PhysicalAllocation::new(dev, bytes).unwrap();
            bindings.push(team.bind_mem(0, &allocation, 0, size).unwrap());
            physical.push(allocation);
        }
        // Each mapping must be dropped before its virtual address reservation.
        let mut mappings = Vec::new();
        for (i, ctx) in contexts.iter().enumerate() {
            ctx.bind_to_thread().unwrap();
            let uc = vmm::VirtualReservation::new(size, 0).unwrap();
            let uc_map = vmm::Mapping::new(uc.base(), size, &physical[i], 0).unwrap();
            vmm::set_access(uc.base(), size, &[devices[i]]).unwrap();
            let mc = vmm::VirtualReservation::new(size, granularity).unwrap();
            let mc_map = vmm::Mapping::new_multicast(mc.base(), size, &team, 0).unwrap();
            vmm::set_access(mc.base(), size, &[devices[i]]).unwrap();
            mappings.push((uc_map, uc, mc_map, mc));
        }
        for initiator in 0..2 {
            // Host copies and completed launches provide the inter-device
            // synchronization. Tile IR tokens order only one kernel's accesses.
            for i in 0..2 {
                contexts[i].bind_to_thread().unwrap();
                let base = 10 * i as i32;
                copy_to_device(
                    mappings[i].1.base(),
                    &[base + 1, base + 2, base + 3, base + 4],
                );
            }
            let ctx = &contexts[initiator];
            ctx.bind_to_thread().unwrap();
            let stream = ctx.new_stream().unwrap();
            let output = api::zeros::<i32>(&[4]).sync_on(&stream).unwrap();
            let multicast = mappings[initiator].3.base();
            unsafe {
                kernels::nvls_load_reduce(
                    cutile::tile_kernel::DevicePointer::from_cu_deviceptr(multicast),
                    output.device_pointer(),
                )
            }
            .grid((1, 1, 1))
            .sync_on(&stream)
            .unwrap();
            assert_eq!(
                output.to_host_vec().sync_on(&stream).unwrap(),
                [12, 14, 16, 18]
            );
            unsafe {
                kernels::nvls_reduce(cutile::tile_kernel::DevicePointer::from_cu_deviceptr(
                    multicast,
                ))
            }
            .grid((1, 1, 1))
            .sync_on(&stream)
            .unwrap();
            for i in 0..2 {
                contexts[i].bind_to_thread().unwrap();
                let base = 10 * i as i32;
                assert_eq!(
                    copy_from_device(mappings[i].1.base()),
                    [base + 6, base + 7, base + 8, base + 9]
                );
            }
            ctx.bind_to_thread().unwrap();
            unsafe {
                kernels::nvls_store(cutile::tile_kernel::DevicePointer::from_cu_deviceptr(
                    multicast,
                ))
            }
            .grid((1, 1, 1))
            .sync_on(&stream)
            .unwrap();
            for i in 0..2 {
                contexts[i].bind_to_thread().unwrap();
                assert_eq!(copy_from_device(mappings[i].1.base()), [37; 4]);
            }
        }
        for (ctx, mapping) in contexts.iter().zip(mappings) {
            ctx.bind_to_thread().unwrap();
            drop(mapping);
        }
        drop(bindings);
        drop(team);
        drop(physical);
    });
}

#[test]
#[ignore = "requires a CUDA 13.5 assembler/backend; no GPU required"]
fn assemble_multicast_execution_kernels() {
    common::with_test_stack(|| {
        assert_eq!(
            std::env::var("CUTILE_BYTECODE_VERSION").as_deref(),
            Ok("13.5"),
            "set CUTILE_BYTECODE_VERSION=13.5 to prevent assembler fallback"
        );
        for kernel in ["nvls_load_reduce", "nvls_reduce", "nvls_store"] {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .unwrap();
            let cubin = cutile_compiler::cuda_tile_runtime_utils::compile_tile_ir_module(
                artifacts.module(),
                "sm_100",
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            assert!(cubin.starts_with(b"\x7fELF"), "expected cubin ELF");
        }
    });
}
