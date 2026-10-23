/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-pass probe for SM100 raw instruction operands and canonical carriers.
#![allow(dead_code)]
use ctm_rs::{async_copy, f32x2, fence, float, mbarrier, memory, tcgen05, thread, warp};

/// Canonical carrier identity is retained across both public crates.
fn canonical_simd<const N: usize>(value: memory::CuSimd<u32, N>) -> cuda_device::CuSimd<u32, N> {
    value
}

/// Canonical packed global-memory element identity is retained.
fn canonical_u32x4(value: memory::U32x4) -> cuda_device::vector::U32x4 {
    value
}

/// Type-check explicit copy, barrier, group-1 commit and TMEM instructions; never run on host.
unsafe fn sm100_step(bar: *mut mbarrier::Barrier, shared: *mut u32, global: *const u32, addr: u32) {
    unsafe {
        mbarrier::mbarrier_init(bar, 1);
        let _ = mbarrier::mbarrier_arrive(bar);
        let _ = mbarrier::mbarrier_arrive_expect_tx(bar, 1, 4);
        let _ = mbarrier::mbarrier_arrive_expect_tx_cluster(bar, 1, 4);
        let _ = mbarrier::mbarrier_try_wait_parity(bar, 0);
        let _ = mbarrier::mbarrier_try_wait_parity_cluster_relaxed(bar, 1);
        fence::fence_proxy_async_acquire_sync_restrict();
        fence::fence_proxy_async_release_sync_restrict();
        fence::fence_sync_restrict_acquire();
        async_copy::cp_async_ca_4_raw(shared, global);
        async_copy::cp_async_mbarrier_arrive_noinc_raw(bar);
        tcgen05::commit_address(memory::SharedAddress::from_raw(bar));
        thread::setmaxnreg_inc::<256>();
        thread::setmaxnreg_dec::<24>();
        let data: cuda_device::CuSimd<u32, 2> = tcgen05::tmem::load_32x32b_x2(addr);
        tcgen05::tmem::load_wait();
        tcgen05::tmem::store_32x32b_x2(addr, data);
        let data: cuda_device::CuSimd<u32, 4> = tcgen05::tmem::load_32x32b_x4(addr);
        tcgen05::tmem::load_wait();
        tcgen05::tmem::store_32x32b_x4(addr, data);
        let data: cuda_device::CuSimd<u32, 8> = tcgen05::tmem::load_32x32b_x8(addr);
        tcgen05::tmem::load_wait();
        tcgen05::tmem::store_32x32b_x8(addr, data);
        let data: cuda_device::CuSimd<u32, 16> = tcgen05::tmem::load_32x32b_x16(addr);
        tcgen05::tmem::load_wait();
        tcgen05::tmem::store_32x32b_x16(addr, data);
        let data: cuda_device::CuSimd<u32, 32> = tcgen05::tmem::load_32x32b_x32(addr);
        tcgen05::tmem::load_wait();
        tcgen05::tmem::store_32x32b_x32(addr, data);
        tcgen05::tmem::store_wait();
    }
    unsafe { tcgen05::relinquish_alloc_permit() };
    let x = f32x2::pack_f32x2(1.0, 2.0);
    let y = f32x2::add_f32x2(x, x);
    let z = f32x2::mul_f32x2(x, y);
    let _ = f32x2::fma_f32x2(x, y, z);
    let _ = float::lg2_approx_f32(f32x2::unpack_f32x2_hi(z));
    let _ = unsafe { warp::all(true) };
    let _ = unsafe { warp::shuffle_xor(0, 1) };
    let _ = unsafe { warp::shuffle_sync(u32::MAX, 0, 0) };
    unsafe { warp::bar_warp_sync(u32::MAX) };
    let mut i = 0;
    while i < 2 {
        thread::__unroll_config::<0>();
        i += 1;
    }
    let mut j = 0;
    while j < 2048 {
        thread::__unroll_config::<1024>();
        j += 1;
    }
}

/// Host pack/unpack operations preserve both lanes' exact bits.
fn main() {
    let packed = f32x2::pack_f32x2(f32::from_bits(0x8000_0000), f32::from_bits(0x7fc1_2345));
    assert_eq!(f32x2::unpack_f32x2_lo(packed).to_bits(), 0x8000_0000);
    assert_eq!(f32x2::unpack_f32x2_hi(packed).to_bits(), 0x7fc1_2345);
}
