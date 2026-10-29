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
//! allocation handle, which the mapping's borrows enforce).
//!
//! Skips when no GPU is present.

use cuda_core::{vmm, Device, DriverError, IntoResult};
use std::mem::MaybeUninit;

const INVALID_VALUE: DriverError =
    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);

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
        let map = vmm::Mapping::new(&va, 0, &phys, 0, size).expect("cuMemMap");
        assert_eq!(map.va(), va.base());
        assert_eq!(map.size(), size);
        assert!(std::ptr::eq(map.reservation(), &va));
        assert!(map.physical().is_some_and(|p| std::ptr::eq(p, &phys)));
        assert!(map.multicast().is_none());
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
        // allocation handle it borrows; the borrows make any other order a
        // compile error.
    }
}

/// The driver cannot tell where our reservation ends (VA ranges are
/// process-wide), so `Mapping::new` bounds the range itself before calling
/// `cuMemMap`, and reports a wrap or an overrun as `CUDA_ERROR_INVALID_VALUE`.
#[test]
fn mapping_rejects_ranges_outside_the_reservation_or_allocation() {
    match gpu_count() {
        Ok(count) if count > 0 => {}
        Ok(_) => {
            eprintln!("SKIPPED: mapping bounds test requires a GPU");
            return;
        }
        Err(error) => {
            eprintln!("SKIPPED: CUDA unavailable ({error:?})");
            return;
        }
    }

    let device = Device::new(0).expect("GPU 0 device");
    let cu_device = device.cu_device();
    let granularity = vmm::allocation_granularity(cu_device).expect("allocation granularity query");
    let size = vmm::align_size(1 << 20, granularity);

    let phys = vmm::PhysicalAllocation::new(cu_device, size).expect("cuMemCreate");
    // Twice the allocation, so there is a valid second half to map into.
    let va = vmm::VirtualReservation::new(2 * size, 0).expect("cuMemAddressReserve");

    assert_eq!(
        vmm::Mapping::new(&va, 2 * size, &phys, 0, size).err(),
        Some(INVALID_VALUE),
        "a range past the end of the reservation is refused"
    );
    assert_eq!(
        vmm::Mapping::new(&va, 0, &phys, granularity, size).err(),
        Some(INVALID_VALUE),
        "a range past the end of the allocation is refused"
    );
    assert_eq!(
        vmm::Mapping::new(&va, usize::MAX, &phys, 0, size).err(),
        Some(INVALID_VALUE),
        "offset arithmetic that would wrap is refused"
    );

    let map = vmm::Mapping::new(&va, size, &phys, 0, size).expect("the second half is in range");
    assert_eq!(map.va(), va.base() + size as u64);
    assert!(std::ptr::eq(map.reservation(), &va));
}
