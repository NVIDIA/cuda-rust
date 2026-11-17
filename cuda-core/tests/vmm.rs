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

use cuda_core::{vmm, Context, IntoResult};
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

    let device = Context::new(0).expect("GPU 0 device");
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

#[test]
fn vmm_locality_domain_roundtrip() {
    match gpu_count() {
        Ok(count) if count > 0 => {}
        Ok(_) => {
            eprintln!("SKIPPED: vmm_locality_domain_roundtrip requires a GPU");
            return;
        }
        Err(error) => {
            eprintln!("SKIPPED: CUDA unavailable ({error:?})");
            return;
        }
    }
    let device = Context::new(0).expect("GPU 0 device");
    let cu_device = device.cu_device();
    // Every GPU has locality domain 0 (the whole memory of a single-die GPU).
    let location = vmm::MemLocation::LocalityDomain {
        device: cu_device,
        domain: 0,
    };
    let granularity =
        match vmm::allocation_granularity_at(location, vmm::AllocationGranularity::Recommended) {
            Ok(granularity) => granularity,
            Err(error)
                if error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
                    || error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE =>
            {
                eprintln!("SKIPPED: no locality-domain allocations here ({error:?})");
                return;
            }
            Err(error) => panic!("locality-domain granularity query: {error:?}"),
        };
    assert!(granularity > 0, "granularity must be positive");
    let minimum = vmm::allocation_granularity_at(location, vmm::AllocationGranularity::Minimum)
        .expect("minimum granularity");
    assert_eq!(
        granularity % minimum,
        0,
        "recommended is a multiple of minimum"
    );

    let phys = vmm::PhysicalAllocation::new_at(location, granularity).expect("cuMemCreate");
    assert_eq!(phys.location(), location);
    assert_eq!(phys.device(), cu_device);
    let va = vmm::VirtualReservation::new(granularity, 0).expect("cuMemAddressReserve");
    {
        let _map = vmm::Mapping::new(va.base(), granularity, &phys, 0).expect("cuMemMap");
        vmm::set_access(va.base(), granularity, &[cu_device]).expect("cuMemSetAccess");
        let pattern: Vec<u32> = (0..1024_u32).map(|v| v ^ 0x5a5a_5a5a).collect();
        let bytes = std::mem::size_of_val(pattern.as_slice());
        let mut readback = vec![0_u32; pattern.len()];
        unsafe {
            cuda_core::sys::cuMemcpyHtoD_v2(va.base(), pattern.as_ptr().cast(), bytes)
                .result()
                .expect("HtoD through the mapping");
            cuda_core::sys::cuMemcpyDtoH_v2(readback.as_mut_ptr().cast(), va.base(), bytes)
                .result()
                .expect("DtoH through the mapping");
        }
        assert_eq!(readback, pattern, "data must roundtrip through the mapping");
    }
}

#[test]
fn locality_domain_rejects_unencodable_device() {
    let location = vmm::MemLocation::LocalityDomain {
        device: 256,
        domain: 0,
    };
    let error = vmm::PhysicalAllocation::new_at(location, 1 << 21)
        .err()
        .expect("device 256 cannot be encoded");
    assert_eq!(
        error.0,
        cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_DEVICE
    );
}

#[test]
fn vmm_locality_domains_place_each_piece() {
    if !matches!(gpu_count(), Ok(count) if count > 0) {
        eprintln!("SKIPPED: vmm_locality_domains_place_each_piece requires a GPU");
        return;
    }
    let device = Context::new(0).expect("GPU 0 device");
    let cu_device = device.cu_device();
    let domains = device
        .locality_domain_count()
        .expect("locality domain count");
    if domains < 2 {
        eprintln!("SKIPPED: GPU 0 has {domains} locality domain(s)");
        return;
    }
    let location = |domain| vmm::MemLocation::LocalityDomain {
        device: cu_device,
        domain,
    };
    let g = vmm::allocation_granularity_at(location(0), vmm::AllocationGranularity::Recommended)
        .expect("granularity");
    let n = domains as usize;
    let va = vmm::VirtualReservation::new(n * g, g).expect("cuMemAddressReserve");
    let maps: Vec<_> = (0..n)
        .map(|d| {
            let phys = vmm::PhysicalAllocation::new_at(location(d as u8), g).expect("cuMemCreate");
            vmm::Mapping::new(va.base() + (d * g) as u64, g, &phys, 0).expect("cuMemMap")
        })
        .collect();
    vmm::set_access(va.base(), n * g, &[cu_device]).expect("cuMemSetAccess");
    for d in 0..n {
        let domain =
            vmm::locality_domain_of(va.base() + (d * g) as u64).expect("locality_domain_of");
        assert_eq!(
            domain,
            Some(d as u8),
            "piece {d} lives in locality domain {d}"
        );
    }
    drop(maps);
}

#[test]
fn interleaved_memory_has_no_locality_domain() {
    if !matches!(gpu_count(), Ok(count) if count > 0) {
        eprintln!("SKIPPED: interleaved_memory_has_no_locality_domain requires a GPU");
        return;
    }
    let device = Context::new(0).expect("GPU 0 device");
    if cuda_core::sys::POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL.is_none() {
        eprintln!("SKIPPED: bindings predate locality domains");
        return;
    }
    device.bind_to_thread().expect("bind");
    let ptr = unsafe { cuda_core::memory::malloc_sync(1 << 20) }.expect("cuMemAlloc");
    let domain = vmm::locality_domain_of(ptr);
    unsafe { cuda_core::memory::free_sync(ptr) }.expect("cuMemFree");
    assert_eq!(domain.expect("locality_domain_of"), None);
}
