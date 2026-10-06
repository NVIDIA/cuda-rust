/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![no_std]

/// The shared body keeps the direct, local-helper, and dependency-helper probes
/// identical. Only placement across a function boundary changes.
#[macro_export]
macro_rules! shared_probe_body {
    ($smem:expr, $offset:expr) => {{
        let smem: *mut u8 = $smem;
        let offset: usize = $offset;
        // The caller provides 2048 live shared bytes, aligned to 1024, and one
        // participating thread per CTA. The data slot and barrier do not overlap.
        unsafe {
            let full = smem.add(1024).cast::<cuda_device::barrier::Barrier>();
            let data = smem.add(offset).cast::<u64>();
            cuda_device::barrier::mbarrier_init(full, 1);
            let pending = !cuda_device::barrier::mbarrier_try_wait_parity(full, 0);
            let _token = cuda_device::barrier::mbarrier_arrive(full);
            // A failed barrier must report failure rather than hang the GPU.
            let mut complete = 0u64;
            let mut attempts = 0u32;
            while attempts < 1024 {
                if cuda_device::barrier::mbarrier_try_wait_parity(full, 0) {
                    complete = if pending { 1 } else { 2 };
                    break;
                }
                attempts += 1;
            }
            let value = core::ptr::read_volatile(data);
            core::ptr::write_volatile(data, value.wrapping_mul(3) ^ 0x1234_5678);
            complete
        }
    }};
}

/// Exercise a real dependency-crate body with a raw shared pointer argument.
///
/// # Safety
/// `smem` must address 2048 shared bytes aligned to 1024. One thread per CTA
/// calls this function. `offset` is a multiple of eight below 1024; its u64
/// slot is initialized. The barrier at byte 1024 is unused before this call.
#[cfg_attr(feature = "force-inline", inline(always))]
#[cfg_attr(
    not(any(feature = "force-inline", feature = "heuristic-inline")),
    inline(never)
)]
pub unsafe fn cross_crate_probe(smem: *mut u8, offset: usize) -> u64 {
    shared_probe_body!(smem, offset)
}
