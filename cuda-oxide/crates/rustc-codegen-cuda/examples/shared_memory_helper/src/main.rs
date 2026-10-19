/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compare a direct shared-memory operation with local and dependency helpers.
//! `inline(never)` requests a MIR call boundary; inspect generated code to
//! determine whether the backend also keeps the calls.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DynamicSharedArray, cuda_module, kernel, launch_bounds, thread};

#[cfg(all(feature = "local-helper", feature = "cross-crate-helper"))]
compile_error!("Select at most one helper feature; no features selects the direct control");

#[cfg(all(feature = "force-inline", feature = "heuristic-inline"))]
compile_error!("Select at most one inlining policy");

#[cfg(feature = "local-helper")]
#[cfg_attr(feature = "force-inline", inline(always))]
#[cfg_attr(
    not(any(feature = "force-inline", feature = "heuristic-inline")),
    inline(never)
)]
unsafe fn local_probe(smem: *mut u8, offset: usize) -> u64 {
    shared_memory_helper_lib::shared_probe_body!(smem, offset)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(1)]
    pub fn shared_memory_probe(output: *mut u64, seed: u64, byte_offset: u64) {
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        let block = thread::blockIdx_x() as usize;
        let offset = byte_offset as usize;
        unsafe {
            let data = smem.add(offset).cast::<u64>();
            core::ptr::write_volatile(data, seed.wrapping_add(block as u64));

            #[cfg(feature = "local-helper")]
            let completed = local_probe(smem, offset);
            #[cfg(feature = "cross-crate-helper")]
            let completed = shared_memory_helper_lib::cross_crate_probe(smem, offset);
            #[cfg(not(any(feature = "local-helper", feature = "cross-crate-helper")))]
            let completed = shared_memory_helper_lib::shared_probe_body!(smem, offset);

            *output.add(block * 2) = core::ptr::read_volatile(data);
            *output.add(block * 2 + 1) = completed;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const BLOCKS: usize = 4;
    let variant = if cfg!(feature = "local-helper") {
        "local-helper"
    } else if cfg!(feature = "cross-crate-helper") {
        "cross-crate-helper"
    } else {
        "direct"
    };
    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    if major < 9 {
        println!(
            "skipping: shared-memory barrier probe requires sm_90+ (device is sm_{major}{minor})"
        );
        return Ok(());
    }
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let output = DeviceBuffer::<u64>::zeroed(&stream, BLOCKS * 2)?;
    let config = LaunchConfig {
        grid_dim: (BLOCKS as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 2048,
    };
    let mut cases = 0;
    for seed in [0u64, 17, u64::MAX] {
        for offset in [0u64, 64, 248] {
            // SAFETY: four independent CTAs, one thread each; output holds two
            // u64s per CTA. Data offsets are aligned and disjoint from the
            // barrier. Every launch receives a fresh shared-memory allocation.
            unsafe {
                module.shared_memory_probe(
                    &stream,
                    config,
                    output.cu_deviceptr() as *mut u64,
                    seed,
                    offset,
                )?;
            }
            let got = output.to_host_vec(&stream)?;
            for block in 0..BLOCKS {
                let expected = seed.wrapping_add(block as u64).wrapping_mul(3) ^ 0x1234_5678;
                assert_eq!(
                    got[block * 2],
                    expected,
                    "{variant}: data, seed={seed}, offset={offset}, block={block}"
                );
                assert_eq!(
                    got[block * 2 + 1],
                    1,
                    "{variant}: barrier, seed={seed}, offset={offset}, block={block}"
                );
            }
            cases += 1;
        }
    }
    println!(
        "PASS {variant}: {cases} launches, {} CTA data/barrier checks",
        cases * BLOCKS
    );
    Ok(())
}
