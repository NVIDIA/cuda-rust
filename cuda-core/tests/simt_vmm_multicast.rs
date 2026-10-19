/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration test: VMM multicast objects (NVLink SHARP / NVLS).
//!
//! Builds a two-GPU multicast team and verifies the host-side plumbing:
//! create the object, add both devices, bind per-GPU physical memory, map
//! unicast and multicast views, roundtrip data through the bound memory,
//! and tear down in the required order.
//!
//! The switch-side reduction semantics can only be exercised from device
//! code (`multimem.ld_reduce` / `multimem.st`); that lives in the
//! `nvls_all_reduce` example. Host code must not dereference the multicast
//! mapping, so this test only establishes and tears it down.
//!
//! Requires an NVLink-switch system (e.g. HGX H100/B200); skips elsewhere.
//! Compiled out entirely when the CUDA toolkit predates the multicast API
//! (CUDA 12.1); see cuda-core's build script probe.
#![cfg(cuda_has_multicast)]

use cuda_core::simt::context::CudaContext;
use cuda_core::simt::vmm;
use cuda_core::IntoResult;
use std::mem::MaybeUninit;

fn gpu_count() -> Result<usize, cuda_core::DriverError> {
    unsafe { cuda_core::init(0)? };
    let mut count = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

#[test]
fn multicast_two_gpu_team_roundtrip() {
    let count = gpu_count().expect("failed to get device count");
    if count < 2 {
        eprintln!("SKIPPED: multicast_two_gpu_team_roundtrip requires 2+ GPUs (found {count})");
        return;
    }

    let ctx0 = CudaContext::new(0).expect("GPU 0 context");
    let ctx1 = CudaContext::new(1).expect("GPU 1 context");
    let devices = [ctx0.cu_device(), ctx1.cu_device()];

    for (i, &dev) in devices.iter().enumerate() {
        let supported = vmm::multicast_supported(dev).expect("multicast_supported query failed");
        if !supported {
            eprintln!("SKIPPED: GPU {i} does not support switch multicast (NVLS)");
            return;
        }
    }

    let min_bytes = 1 << 20;
    let granularity = vmm::multicast_granularity(
        devices.len() as u32,
        min_bytes,
        vmm::MulticastGranularity::Recommended,
    )
    .expect("cuMulticastGetGranularity failed");
    assert!(granularity > 0, "granularity must be positive");
    let size = vmm::align_size(min_bytes, granularity);

    let team =
        vmm::MulticastObject::new(devices.len() as u32, size).expect("cuMulticastCreate failed");
    assert_eq!(team.size(), size);
    assert_eq!(team.num_devices(), 2);
    for &dev in &devices {
        team.add_device(dev).expect("cuMulticastAddDevice failed");
    }

    // The physical size must satisfy the allocation granularity as well as
    // the multicast granularity, so align to both.
    let contexts = [&ctx0, &ctx1];
    let mut physes = Vec::new();
    let mut bindings = Vec::new();
    for (i, ctx) in contexts.iter().enumerate() {
        ctx.bind_to_thread().expect("bind context");
        let alloc_gran =
            vmm::allocation_granularity(devices[i]).expect("allocation granularity query");
        let phys_size = vmm::align_size(size, alloc_gran);
        let phys = vmm::PhysicalAllocation::new(devices[i], phys_size).expect("cuMemCreate failed");
        assert_eq!(phys.device(), devices[i]);
        let binding = team
            .bind_mem(0, &phys, 0, size)
            .expect("cuMulticastBindMem failed");
        assert_eq!(binding.device(), devices[i]);
        physes.push(phys);
        bindings.push(binding);
    }

    // Per GPU: (uc_va, uc_map, mc_va, mc_map)
    let mut mappings = Vec::new();
    for (i, ctx) in contexts.iter().enumerate() {
        ctx.bind_to_thread().expect("bind context");

        let uc_va = vmm::VirtualReservation::new(size, 0).expect("unicast VA reserve");
        let uc_map =
            vmm::Mapping::new(uc_va.base(), size, &physes[i], 0).expect("unicast cuMemMap");
        vmm::set_access(uc_va.base(), size, &[devices[i]]).expect("unicast set_access");

        let mc_va = vmm::VirtualReservation::new(size, granularity).expect("multicast VA reserve");
        let mc_map =
            vmm::Mapping::new_multicast(mc_va.base(), size, &team, 0).expect("multicast cuMemMap");
        vmm::set_access(mc_va.base(), size, &[devices[i]]).expect("multicast set_access");

        mappings.push((uc_va, uc_map, mc_va, mc_map));
    }

    ctx0.bind_to_thread().expect("bind ctx0");
    let pattern: Vec<u32> = (0..256).map(|i| i * 3 + 7).collect();
    let byte_len = pattern.len() * std::mem::size_of::<u32>();
    let stream0 = ctx0.default_stream();
    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            mappings[0].0.base(),
            pattern.as_ptr(),
            byte_len,
            stream0.cu_stream(),
        )
        .expect("HtoD via unicast view");
    }
    ctx0.synchronize().expect("sync GPU 0");

    let mut readback = vec![0u32; 256];
    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            readback.as_mut_ptr(),
            mappings[0].0.base(),
            byte_len,
            stream0.cu_stream(),
        )
        .expect("DtoH via unicast view");
    }
    ctx0.synchronize().expect("sync GPU 0");
    assert_eq!(
        readback, pattern,
        "unicast roundtrip through bound memory failed"
    );

    // Teardown order: mappings, then bindings, then team, then physical
    // allocations.
    drop(mappings);
    drop(bindings);
    drop(team);
    drop(physes);
}
