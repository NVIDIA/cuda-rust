// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::thread::index_1d_u32 as fast_index;
use cuda_device::{kernel, launch_contract};

#[kernel(launch_context = launch_context)]
#[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
pub fn simple(value: u32) {
    let _ = (fast_index(launch_context), value);
}

#[kernel(launch_context = launch_context)]
#[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
pub fn generic<const N: u32>(value: u32) {
    let _ = (fast_index(launch_context), value, N);
}

#[kernel(u32, launch_context = launch_context)]
#[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
pub fn explicit<T: Copy>(value: T) {
    let _ = (fast_index(launch_context), value);
}

#[kernel(launch_context = launch_context)]
#[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
pub fn generated_storage_name_is_hygienic(cuda_oxide_kernel_scope_246e25db_storage: u32) {
    let cuda_oxide_kernel_scope_246e25db_storage =
        cuda_oxide_kernel_scope_246e25db_storage.wrapping_add(1);
    let _ = (
        fast_index(launch_context),
        cuda_oxide_kernel_scope_246e25db_storage,
    );
}

fn entry_abis_do_not_contain_the_launch_context() {
    let _: fn(u32) = cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_simple;
    let _: fn(u32) = cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_generic::<4>;
    let _: fn(u32) = cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_explicit_u32;
    let _: fn(u32) =
        cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_generated_storage_name_is_hygienic;
}

fn main() {}
