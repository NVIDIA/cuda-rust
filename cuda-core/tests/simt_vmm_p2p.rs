/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration test: VMM + P2P cross-GPU memory access.
//!
//! This test does the following:
//! 1. Checks that the machine has at least 2 GPUs.
//! 2. Creates one CUDA context for GPU 0 and one for GPU 1.
//! 3. Asks CUDA whether those two GPUs support direct peer access.
//! 4. Enables peer access in both directions.
//! 5. Allocates physical memory only on GPU 0.
//! 6. Reserves one VA range on GPU 0 and maps that GPU 0 physical memory there.
//! 7. Reserves a separate VA range on GPU 1 and maps the same GPU 0 physical
//!    memory there too.
//! 8. Grants both GPUs access to both mappings with `cuMemSetAccess`.
//! 9. Writes a known `u32` pattern through GPU 0's mapping.
//! 10. Reads back through GPU 1's mapping.
//! 11. Asserts the bytes match exactly.
//! 12. Tears everything down in the required order: mappings first, then VA
//!     reservations, then physical allocation, then peer access.
//!
//! Requires a system with at least 2 CUDA-capable GPUs with P2P support.

use cuda_core::simt::context::CudaContext;
use cuda_core::simt::peer;
use cuda_core::simt::vmm;
use cuda_core::{DriverError, IntoResult};
use std::mem::MaybeUninit;

const INVALID_VALUE: DriverError =
    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);

fn gpu_count() -> Result<usize, cuda_core::DriverError> {
    unsafe { cuda_core::init(0)? };
    let mut count = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

#[test]
fn vmm_alloc_map_set_access_roundtrip() {
    let ctx = CudaContext::new(0).expect("failed to create context for GPU 0");
    let granularity =
        vmm::allocation_granularity(ctx.cu_device()).expect("failed to query granularity");
    assert!(granularity > 0, "granularity must be positive");

    let alloc_size = vmm::align_size(4096, granularity);

    let phys =
        vmm::PhysicalAllocation::new(ctx.cu_device(), alloc_size).expect("cuMemCreate failed");
    assert_eq!(phys.size(), alloc_size);

    let va = vmm::VirtualReservation::new(alloc_size, 0).expect("cuMemAddressReserve failed");
    assert_ne!(va.base(), 0);

    let mapping = vmm::Mapping::new(&va, 0, &phys, 0, alloc_size).expect("cuMemMap failed");
    assert_eq!(mapping.va(), va.base());
    assert_eq!(mapping.size(), alloc_size);
    assert!(std::ptr::eq(mapping.reservation(), &va));
    assert!(mapping.physical().is_some_and(|p| std::ptr::eq(p, &phys)));

    vmm::set_access(va.base(), alloc_size, &[ctx.cu_device()]).expect("cuMemSetAccess failed");

    let stream = ctx.default_stream();
    let pattern: Vec<u8> = (0..64).map(|i| (i * 3 + 7) as u8).collect();
    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            va.base(),
            pattern.as_ptr(),
            pattern.len(),
            stream.cu_stream(),
        )
        .expect("HtoD memcpy failed");
    }
    ctx.synchronize().expect("sync failed");

    let mut readback = vec![0u8; 64];
    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            readback.as_mut_ptr(),
            va.base(),
            readback.len(),
            stream.cu_stream(),
        )
        .expect("DtoH memcpy failed");
    }
    ctx.synchronize().expect("sync failed");

    assert_eq!(readback, pattern, "single-GPU VMM roundtrip failed");

    // The mapping borrows both, so this is the only order that compiles.
    drop(mapping);
    drop(va);
    drop(phys);
}

/// The driver cannot tell where our reservation ends (VA ranges are
/// process-wide), so `Mapping::new` bounds the range itself before calling
/// `cuMemMap`, and reports a wrap or an overrun as `CUDA_ERROR_INVALID_VALUE`.
#[test]
fn mapping_rejects_ranges_outside_the_reservation_or_allocation() {
    let ctx = CudaContext::new(0).expect("failed to create context for GPU 0");
    let granularity =
        vmm::allocation_granularity(ctx.cu_device()).expect("failed to query granularity");
    let size = vmm::align_size(4096, granularity);

    let phys = vmm::PhysicalAllocation::new(ctx.cu_device(), size).expect("cuMemCreate failed");
    // Twice the allocation, so there is a valid second half to map into.
    let va = vmm::VirtualReservation::new(2 * size, 0).expect("cuMemAddressReserve failed");

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

#[test]
fn p2p_vmm_cross_gpu_access() {
    let count = gpu_count().expect("failed to get device count");
    if count < 2 {
        eprintln!("SKIPPED: p2p_vmm_cross_gpu_access requires 2+ GPUs (found {count})");
        return;
    }

    let ctx0 = CudaContext::new(0).expect("GPU 0 context");
    let ctx1 = CudaContext::new(1).expect("GPU 1 context");

    let can_p2p = peer::can_access_peer(&ctx0, &ctx1).expect("can_access_peer query failed");
    if !can_p2p {
        eprintln!("SKIPPED: GPU 0 cannot access GPU 1 via P2P");
        return;
    }

    peer::enable_peer_access(&ctx0, &ctx1).expect("enable P2P 0→1 failed");
    peer::enable_peer_access(&ctx1, &ctx0).expect("enable P2P 1→0 failed");

    // --- Allocate physical memory on GPU 0 ---
    let granularity = vmm::allocation_granularity(ctx0.cu_device()).expect("granularity query");
    let alloc_size = vmm::align_size(4096, granularity);
    let phys = vmm::PhysicalAllocation::new(ctx0.cu_device(), alloc_size).expect("cuMemCreate");

    // --- Reserve VA and map on GPU 0 ---
    let va0 = vmm::VirtualReservation::new(alloc_size, 0).expect("VA reserve for GPU 0");
    let map0 = vmm::Mapping::new(&va0, 0, &phys, 0, alloc_size).expect("map on GPU 0");
    vmm::set_access(
        va0.base(),
        alloc_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )
    .expect("set_access for GPU 0 mapping");

    // --- Reserve VA and map the SAME physical memory on GPU 1 ---
    let va1 = vmm::VirtualReservation::new(alloc_size, 0).expect("VA reserve for GPU 1");
    let map1 = vmm::Mapping::new(&va1, 0, &phys, 0, alloc_size).expect("map on GPU 1");
    vmm::set_access(
        va1.base(),
        alloc_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )
    .expect("set_access for GPU 1 mapping");

    // --- GPU 0 writes data through its VA ---
    let pattern: Vec<u32> = (0..256).collect();
    let byte_len = pattern.len() * std::mem::size_of::<u32>();
    let stream0 = ctx0.default_stream();
    ctx0.bind_to_thread().expect("bind ctx0");
    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            va0.base(),
            pattern.as_ptr(),
            byte_len,
            stream0.cu_stream(),
        )
        .expect("HtoD via GPU 0 VA");
    }
    ctx0.synchronize().expect("sync GPU 0");

    // --- GPU 1 reads the data through its VA (P2P read of GPU 0's physical memory) ---
    let mut readback = vec![0u32; 256];
    let stream1 = ctx1.default_stream();
    ctx1.bind_to_thread().expect("bind ctx1");
    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            readback.as_mut_ptr(),
            va1.base(),
            byte_len,
            stream1.cu_stream(),
        )
        .expect("DtoH via GPU 1 VA");
    }
    ctx1.synchronize().expect("sync GPU 1");

    assert_eq!(
        readback, pattern,
        "P2P cross-GPU VMM read failed: GPU 1 did not see GPU 0's data"
    );

    // --- Cleanup in correct order: mappings before VA/phys ---
    drop(map1);
    drop(map0);
    drop(va1);
    drop(va0);
    drop(phys);

    peer::disable_peer_access(&ctx0, &ctx1).expect("disable P2P 0→1");
    peer::disable_peer_access(&ctx1, &ctx0).expect("disable P2P 1→0");
}
