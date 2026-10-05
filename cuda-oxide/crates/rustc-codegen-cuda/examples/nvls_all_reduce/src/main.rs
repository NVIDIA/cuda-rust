/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-process NVLS (NVLink SHARP) all-reduce across every
//! multicast-capable GPU in the machine.
//!
//! The host builds a [`vmm::MulticastObject`] team: each GPU binds its own
//! physical copy of a staging region and maps two views of it -- a unicast
//! VA (that GPU's own copy, used to stage inputs and read results) and a
//! multicast VA, where device-side `multimem.*` instructions operate on all
//! bound copies at once: `multimem.ld_reduce` makes the NVSwitch sum the
//! value across every GPU, and `multimem.st` broadcasts the result back.
//!
//! Each GPU launches one kernel over its 1/N shard, doing a fused
//! `ld_reduce` + `st` per 16-byte float4 group -- one switch round-trip per
//! element, no cross-GPU barriers, no P2P copies. The `multimem`
//! instructions are emitted with `ptx_asm!`; codegen detects them and
//! raises the module target to sm_90/PTX 8.6 automatically.
//!
//! Requires an NVLink-switch system (HGX/DGX H100 or B200, CUDA 12.1+).
//! Prints `skipping:` and exits cleanly anywhere else. Building this
//! example needs a CUDA 12.1+ toolkit: cuda-core compiles out the
//! multicast wrappers on older toolkits (see its build script probe).

use cuda_core::CudaContext;
use cuda_core::IntoResult;
use cuda_core::simt::LaunchConfig;
use cuda_core::simt::vmm;
use cuda_device::{cuda_module, kernel, ptx_asm};
use std::mem::MaybeUninit;
use std::sync::Arc;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::thread;

    /// One thread = one 16-byte float4 group of this GPU's shard.
    ///
    /// `mc` is the multicast VA; `base_group`/`num_groups` select the shard
    /// in float4 groups. The reduced vector stays in PTX-local registers:
    /// the switch sums the group across all bound GPUs, then the store
    /// broadcasts the sum back into every copy.
    #[kernel]
    pub fn nvls_all_reduce_f32(mc: u64, base_group: u32, num_groups: u32) {
        let i = thread::index_1d().get();
        if i < num_groups as usize {
            let addr = mc + (base_group as u64 + i as u64) * 16;
            unsafe {
                ptx_asm!(
                    "{ .reg .f32 t0, t1, t2, t3;\n\t\
                       multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {t0, t1, t2, t3}, [%0];\n\t\
                       multimem.st.relaxed.sys.global.v4.f32 [%0], {t0, t1, t2, t3}; }",
                    in("l") addr,
                    clobber("memory")
                );
            }
        }
    }
}

/// Number of f32 elements to all-reduce (must be a multiple of 4).
const ELEMS: usize = 1 << 20;

/// The value GPU `g` contributes at element `i`. Small integers, so the f32
/// sums are exact and order-independent.
fn input_value(gpu: usize, i: usize) -> f32 {
    ((gpu + 1) + (i % 1024)) as f32
}

fn gpu_count() -> Result<usize, cuda_core::DriverError> {
    unsafe { cuda_core::init(0)? };
    let mut count = MaybeUninit::uninit();
    unsafe {
        cuda_core::sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

fn main() {
    println!("nvls_all_reduce example");

    let ngpu = gpu_count().expect("failed to get device count");
    if ngpu < 2 {
        println!("skipping: NVLS all-reduce requires 2+ GPUs (found {ngpu})");
        return;
    }

    let contexts: Vec<Arc<CudaContext>> = (0..ngpu)
        .map(|i| CudaContext::new(i).unwrap_or_else(|e| panic!("GPU {i} context: {e:?}")))
        .collect();
    let devices: Vec<_> = contexts.iter().map(|c| c.cu_device()).collect();

    for (i, &dev) in devices.iter().enumerate() {
        if !vmm::multicast_supported(dev).expect("multicast_supported query") {
            println!(
                "skipping: GPU {i} does not support switch multicast (NVLS needs an NVLink-switch system)"
            );
            return;
        }
    }
    println!("Team: {ngpu} multicast-capable GPUs, {ELEMS} f32 elements");

    let bytes = ELEMS * std::mem::size_of::<f32>();
    let granularity =
        vmm::multicast_granularity(ngpu as u32, bytes, vmm::MulticastGranularity::Recommended)
            .expect("multicast granularity query");
    let size = vmm::align_size(bytes, granularity);

    let team = vmm::MulticastObject::new(ngpu as u32, size).expect("cuMulticastCreate");
    for &dev in &devices {
        team.add_device(dev).expect("cuMulticastAddDevice");
    }

    // Per GPU: (uc_va, uc_map, mc_va, mc_map). Teardown order at the end of
    // main: views, then bindings, then team, then physes.
    let mut physes = Vec::new();
    let mut bindings = Vec::new();
    let mut views = Vec::new();
    for (i, ctx) in contexts.iter().enumerate() {
        ctx.bind_to_thread().expect("bind context");

        let alloc_gran = vmm::allocation_granularity(devices[i]).expect("allocation granularity");
        let phys = vmm::PhysicalAllocation::new(devices[i], vmm::align_size(size, alloc_gran))
            .expect("cuMemCreate");
        let binding = team
            .bind_mem(0, &phys, 0, size)
            .expect("cuMulticastBindMem");

        let uc_va = vmm::VirtualReservation::new(size, 0).expect("unicast VA reserve");
        let uc_map = vmm::Mapping::new(uc_va.base(), size, &phys, 0).expect("unicast map");
        vmm::set_access(uc_va.base(), size, &[devices[i]]).expect("unicast set_access");

        let mc_va = vmm::VirtualReservation::new(size, granularity).expect("multicast VA reserve");
        let mc_map =
            vmm::Mapping::new_multicast(mc_va.base(), size, &team, 0).expect("multicast map");
        vmm::set_access(mc_va.base(), size, &[devices[i]]).expect("multicast set_access");

        physes.push(phys);
        bindings.push(binding);
        views.push((uc_va, uc_map, mc_va, mc_map));
    }

    // Stage each GPU's input into its own (unicast) copy.
    for (g, ctx) in contexts.iter().enumerate() {
        ctx.bind_to_thread().expect("bind context");
        let input: Vec<f32> = (0..ELEMS).map(|i| input_value(g, i)).collect();
        let stream = ctx.default_stream();
        unsafe {
            cuda_core::simt::memory::memcpy_htod_async(
                views[g].0.base(),
                input.as_ptr(),
                bytes,
                stream.cu_stream(),
            )
            .expect("stage input");
        }
        ctx.synchronize().expect("sync after staging");
    }

    // Each GPU reduces its 1/N shard of float4 groups through the multicast
    // view.
    let groups = ELEMS / 4;
    let chunk = groups.div_ceil(ngpu);
    for (g, ctx) in contexts.iter().enumerate() {
        ctx.bind_to_thread().expect("bind context");
        let base = (g * chunk).min(groups);
        let len = chunk.min(groups - base);
        if len == 0 {
            continue;
        }
        let module = kernels::load(ctx).expect("load embedded CUDA module");
        let stream = ctx.default_stream();
        // SAFETY: every thread touches one float4 group inside the mapped,
        // access-granted multicast region; shards are disjoint across GPUs.
        unsafe {
            module.nvls_all_reduce_f32(
                &stream,
                LaunchConfig::for_num_elems(len as u32),
                views[g].2.base(),
                base as u32,
                len as u32,
            )
        }
        .expect("kernel launch");
    }
    for ctx in &contexts {
        ctx.bind_to_thread().expect("bind context");
        ctx.synchronize().expect("sync after reduce");
    }

    // Every GPU's copy must now hold the sum of all inputs.
    for (g, ctx) in contexts.iter().enumerate() {
        ctx.bind_to_thread().expect("bind context");
        let mut result = vec![0f32; ELEMS];
        let stream = ctx.default_stream();
        unsafe {
            cuda_core::simt::memory::memcpy_dtoh_async(
                result.as_mut_ptr(),
                views[g].0.base(),
                bytes,
                stream.cu_stream(),
            )
            .expect("read back result");
        }
        ctx.synchronize().expect("sync after readback");

        for (i, &got) in result.iter().enumerate() {
            let expected: f32 = (0..ngpu).map(|src| input_value(src, i)).sum();
            if got != expected {
                eprintln!("GPU {g} mismatch at element {i}: expected {expected}, got {got}");
                std::process::exit(1);
            }
        }
        println!("GPU {g}: all {ELEMS} elements correct");
    }

    drop(views);
    drop(bindings);
    drop(team);
    drop(physes);

    println!("SUCCESS: NVLS all-reduce across {ngpu} GPUs verified");
}
