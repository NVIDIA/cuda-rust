/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration test: VMM physical allocation, VA reservation, and mapping.
//!
//! Exercises the full single-GPU VMM lifecycle: query the allocation
//! granularity, create physical memory, reserve a virtual range, map the
//! memory, grant access, roundtrip data through the mapping, and tear
//! down in a leak-free order (mapping before reservation and physical
//! allocation handle).
//!
//! Skips when no GPU is present.

use cuda_core::{vmm, Device, IntoResult};
use std::mem::MaybeUninit;

#[test]
fn align_size_handles_general_granularities() {
    assert_eq!(vmm::align_size(0, 6), 0);
    assert_eq!(vmm::align_size(12, 6), 12);
    assert_eq!(vmm::align_size(13, 6), 18);
}

#[test]
#[should_panic(expected = "granularity must be nonzero")]
fn align_size_rejects_zero_granularity() {
    vmm::align_size(1, 0);
}

#[test]
#[should_panic(expected = "aligned size overflows usize")]
fn align_size_rejects_overflow() {
    vmm::align_size(usize::MAX, 2);
}

fn gpu_count() -> Result<usize, cuda_core::DriverError> {
    unsafe { cuda_core::init(0)? };
    let mut count = MaybeUninit::uninit();
    unsafe {
        cuda_core::sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

#[test]
fn vmm_single_gpu_roundtrip() {
    let count = match gpu_count() {
        Ok(count) => count,
        Err(error) => {
            eprintln!("SKIPPED: CUDA unavailable ({error:?})");
            return;
        }
    };
    if count == 0 {
        eprintln!("SKIPPED: vmm_single_gpu_roundtrip requires a GPU");
        return;
    }

    let device = Device::new(0).expect("GPU 0 device");
    let cu_device = device.cu_device();

    let granularity = vmm::allocation_granularity(cu_device).expect("allocation granularity query");
    assert!(granularity > 0, "granularity must be positive");
    let size = vmm::align_size(1 << 20, granularity);

    let phys = vmm::PhysicalAllocation::new(cu_device, size).expect("cuMemCreate");
    assert_eq!(phys.size(), size);

    let va = vmm::VirtualReservation::new(size, 0).expect("cuMemAddressReserve");
    assert!(va.base() != 0, "reserved VA must be nonzero");
    assert_eq!(va.size(), size);

    {
        let _map = vmm::Mapping::new(va.base(), size, &phys, 0).expect("cuMemMap");
        vmm::set_access(va.base(), size, &[cu_device]).expect("cuMemSetAccess");

        // Roundtrip a pattern through the mapped range.
        let pattern: Vec<u32> = (0..1024_u32)
            .map(|value| value.wrapping_mul(2654435761))
            .collect();
        let bytes = std::mem::size_of_val(pattern.as_slice());
        let mut readback = vec![0_u32; pattern.len()];
        unsafe {
            cuda_core::sys::cuMemcpyHtoD_v2(
                va.base(),
                pattern.as_ptr() as *const std::ffi::c_void,
                bytes,
            )
            .result()
            .expect("HtoD through the mapping");
            cuda_core::sys::cuMemcpyDtoH_v2(
                readback.as_mut_ptr() as *mut std::ffi::c_void,
                va.base(),
                bytes,
            )
            .result()
            .expect("DtoH through the mapping");
        }
        assert_eq!(readback, pattern, "data must roundtrip through the mapping");
        // The mapping drops here, before the reservation and the physical
        // allocation handle, matching a leak-free teardown order.
    }
}
