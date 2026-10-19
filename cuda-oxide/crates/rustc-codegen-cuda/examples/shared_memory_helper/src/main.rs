/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compare a direct shared-memory operation with local and dependency helpers.
//! `inline(never)` requests a MIR call boundary; inspect generated code to
//! determine whether the backend also keeps the calls.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DynamicSharedArray, SharedPtr, barrier::Barrier};
use cuda_device::{cuda_module, kernel, launch_bounds, thread};

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

/// The shared memory a typed probe works on, as a struct field.
#[derive(Clone, Copy)]
struct Region<T> {
    base: SharedPtr<T>,
    offset: usize,
}

/// The probe body over a typed shared pointer, in a generic helper that is not
/// inlined. Unlike the raw `*mut u8` helpers above, it compiles: `SharedPtr`
/// is a CTA-shared pointer by its type, so the helper's parameter is one too,
/// and the barrier is addressed without converting a generic pointer.
///
/// # Safety
/// As for `shared_probe_body!`: 2048 live shared bytes aligned to 1024, one
/// thread per CTA, and an initialized u64 at `offset`, below byte 1024.
#[inline(never)]
unsafe fn shared_ptr_probe<T>(region: Region<T>) -> u64 {
    unsafe {
        let bytes = region.base.cast::<u8>();
        let full: *mut Barrier = bytes.byte_add(1024).cast::<Barrier>().as_ptr();
        let data = bytes.byte_add(region.offset).cast::<u64>().as_ptr();
        cuda_device::barrier::mbarrier_init(full, 1);
        let pending = !cuda_device::barrier::mbarrier_try_wait_parity(full, 0);
        let _token = cuda_device::barrier::mbarrier_arrive(full);
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
}

/// Two typed shared pointers in one struct, passed by value.
#[derive(Clone, Copy)]
struct Pair {
    a: SharedPtr<f32>,
    b: SharedPtr<f32>,
}

/// Writes through both fields of a by-value `Pair`. The call boundary splits
/// the struct into two parameters; both must stay CTA-shared pointers.
///
/// # Safety
/// Both fields must point at live shared memory with at least `index + 1`
/// `f32` slots.
#[inline(never)]
unsafe fn write_pair(pair: Pair, index: usize, value: f32) {
    unsafe {
        pair.a.add(index).as_ptr().write(value);
        pair.b.add(index).as_ptr().write(value + 1000.0);
    }
}

/// Returns one slot of a `Pair` as a typed shared pointer. The return type
/// must stay a CTA-shared pointer so the caller's store needs no conversion.
///
/// # Safety
/// As for `write_pair`.
#[inline(never)]
unsafe fn select_slot(pair: Pair, index: usize, second: bool) -> SharedPtr<f32> {
    unsafe {
        if second {
            pair.b.add(index)
        } else {
            pair.a.add(index)
        }
    }
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

    /// The same probe through `SharedPtr`, a typed CTA-shared pointer, passed
    /// to a non-inlined generic helper inside a struct.
    #[kernel]
    #[launch_bounds(1)]
    pub fn shared_ptr_memory_probe(output: *mut u64, seed: u64, byte_offset: u64) {
        let smem = DynamicSharedArray::<u32, 1024>::shared_ptr();
        let block = thread::blockIdx_x() as usize;
        let offset = byte_offset as usize;
        unsafe {
            let data = smem.byte_add(offset).cast::<u64>().as_ptr();
            core::ptr::write_volatile(data, seed.wrapping_add(block as u64));
            let completed = shared_ptr_probe(Region { base: smem, offset });
            *output.add(block * 2) = core::ptr::read_volatile(data);
            *output.add(block * 2 + 1) = completed;
        }
    }

    /// `SharedPtr` shapes a helper boundary must keep: a two-field struct of
    /// typed shared pointers passed by value, and a typed shared pointer
    /// returned from a helper and used as a store target.
    #[kernel]
    #[launch_bounds(64)]
    pub fn shared_ptr_shapes_probe(output: *mut f32, seed: f32) {
        const LANES: usize = 64;
        let lane = thread::threadIdx_x() as usize;
        let smem = DynamicSharedArray::<f32, 16>::shared_ptr();
        // SAFETY: the launch supplies 2048 shared bytes, enough for the two
        // 64-slot regions, and every lane writes only its own index and one
        // slot of its partner lane, after a barrier.
        unsafe {
            let pair = Pair {
                a: smem,
                b: smem.add(LANES),
            };
            let value = seed + lane as f32;
            write_pair(pair, lane, value);
            thread::sync_threads();
            // Each lane overwrites one slot of its partner lane through the
            // pointer a helper returns; odd lanes pick the second field.
            let partner = LANES - 1 - lane;
            let slot = select_slot(pair, partner, lane % 2 == 1);
            slot.as_ptr().write(-value);
            thread::sync_threads();
            *output.add(lane) = *pair.a.add(lane).as_ptr();
            *output.add(LANES + lane) = *pair.b.add(lane).as_ptr();
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
    for typed in [false, true] {
        let variant = if typed { "shared-ptr" } else { variant };
        let mut cases = 0;
        for seed in [0u64, 17, u64::MAX] {
            for offset in [0u64, 64, 248] {
                let out = output.cu_deviceptr() as *mut u64;
                // SAFETY: four independent CTAs, one thread each; output holds
                // two u64s per CTA. Data offsets are aligned and disjoint from
                // the barrier. Every launch receives a fresh shared-memory
                // allocation.
                unsafe {
                    if typed {
                        module.shared_ptr_memory_probe(&stream, config, out, seed, offset)?;
                    } else {
                        module.shared_memory_probe(&stream, config, out, seed, offset)?;
                    }
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
    }

    const LANES: usize = 64;
    let shapes_output = DeviceBuffer::<f32>::zeroed(&stream, LANES * 2)?;
    let shapes_config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (LANES as u32, 1, 1),
        shared_mem_bytes: 2048,
    };
    let seeds = [0.0f32, 17.0, 1024.0];
    for seed in seeds {
        let out = shapes_output.cu_deviceptr() as *mut f32;
        // SAFETY: one CTA of 64 lanes; output holds two f32 per lane, and the
        // kernel uses 512 of the 2048 shared bytes supplied.
        unsafe { module.shared_ptr_shapes_probe(&stream, shapes_config, out, seed)? };
        let got = shapes_output.to_host_vec(&stream)?;
        for lane in 0..LANES {
            let partner = LANES - 1 - lane;
            // Lane `partner` overwrote one of this lane's slots: the second
            // field when `partner` is odd, the first field otherwise.
            let overwritten = -(seed + partner as f32);
            let (want_a, want_b) = if partner % 2 == 1 {
                (seed + lane as f32, overwritten)
            } else {
                (overwritten, seed + lane as f32 + 1000.0)
            };
            assert_eq!(
                got[lane], want_a,
                "shared-ptr-shapes: field a, seed={seed}, lane={lane}"
            );
            assert_eq!(
                got[LANES + lane],
                want_b,
                "shared-ptr-shapes: field b, seed={seed}, lane={lane}"
            );
        }
    }
    println!(
        "PASS shared-ptr-shapes: {} launches, {} slot checks",
        seeds.len(),
        seeds.len() * LANES * 2
    );
    Ok(())
}
