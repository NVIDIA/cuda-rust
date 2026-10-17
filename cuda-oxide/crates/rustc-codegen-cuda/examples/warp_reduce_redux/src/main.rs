/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `warp_reduce` on the Ampere single-instruction form, and its controls.
//!
//! `warp_reduce` reduces a full warp with a `shfl`-based butterfly: five
//! rounds of shuffle-and-combine, ten instructions. Ampere does the integer
//! cases in one (`redux.sync`), which #811 measured at 4.74x for the
//! primitive. The form is not chosen at run time -- there is no branch in the
//! emitted code -- it is chosen by the `cuda_oxide_sm_at_least` cfg that
//! `cuda-device`'s build script derives from the target, so the same source
//! builds both ways:
//!
//! ```text
//!   cargo oxide build warp_reduce_redux --arch sm_80    redux.sync, no shfl
//!   cargo oxide build warp_reduce_redux --arch sm_75    shfl butterfly
//! ```
//!
//! Three things have to stay true, and each has a kernel below:
//!
//!   * the nine integer pairs must take the fast form and still be correct;
//!   * `f32` must not -- there is no `redux` for floats before sm_100, so a
//!     hook that claimed one would name an instruction that does not exist;
//!   * a sub-warp tile must not -- `redux.sync` reduces over the lanes in
//!     `membermask`, and a `WarpTile<16>`'s contract is not that shape yet.
//!
//! Run with:
//!   cargo oxide run warp_reduce_redux

use cuda_device::cooperative_groups::{
    ThreadGroup,
    ops::{BitAnd, BitOr, BitXor, Max, Min, Sum},
    this_thread_block, warp_reduce,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Every `u32` op `warp_reduce` supports, over one full warp.
    ///
    /// Lane `l` contributes `l` for sum/min/max, and a single distinct bit
    /// `1 << l` for the bitwise ops, so each answer pins the whole warp:
    /// an `or` or `xor` of all 32 single-bit values is `0xFFFFFFFF`, and an
    /// `and` of their complements is `0` -- every bit is cleared by exactly
    /// one lane. A reduction that silently dropped lanes could not produce
    /// those.
    ///
    /// Lane 0 writes `[sum, min, max, and, or, xor]`.
    #[kernel]
    pub fn reduce_u32_all_ops(mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let lane = warp.thread_rank();

        let v_one = 1u32 << lane;
        let v_inv = !v_one;

        let r_sum = warp_reduce::<u32, Sum, _>(&warp, lane);
        let r_min = warp_reduce::<u32, Min, _>(&warp, lane);
        let r_max = warp_reduce::<u32, Max, _>(&warp, lane);
        let r_and = warp_reduce::<u32, BitAnd, _>(&warp, v_inv);
        let r_or = warp_reduce::<u32, BitOr, _>(&warp, v_one);
        let r_xor = warp_reduce::<u32, BitXor, _>(&warp, v_one);

        if lane == 0 {
            // SAFETY: the host allocates six elements for this kernel.
            unsafe {
                *out.get_unchecked_mut(0) = r_sum;
                *out.get_unchecked_mut(1) = r_min;
                *out.get_unchecked_mut(2) = r_max;
                *out.get_unchecked_mut(3) = r_and;
                *out.get_unchecked_mut(4) = r_or;
                *out.get_unchecked_mut(5) = r_xor;
            }
        }
    }

    /// The signed ops, with values that straddle zero (`-16..=15`).
    ///
    /// Signedness is the part a single-instruction form can get wrong
    /// silently: `redux.sync.min.s32` and `.u32` disagree on exactly these
    /// inputs, and only the signed answer is `-16`.
    ///
    /// Lane 0 writes `[sum, min, max]`.
    #[kernel]
    pub fn reduce_i32_signed_ops(mut out: DisjointSlice<i32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let lane = warp.thread_rank();
        let v = lane as i32 - 16;

        let r_sum = warp_reduce::<i32, Sum, _>(&warp, v);
        let r_min = warp_reduce::<i32, Min, _>(&warp, v);
        let r_max = warp_reduce::<i32, Max, _>(&warp, v);

        if lane == 0 {
            // SAFETY: the host allocates three elements for this kernel.
            unsafe {
                *out.get_unchecked_mut(0) = r_sum;
                *out.get_unchecked_mut(1) = r_min;
                *out.get_unchecked_mut(2) = r_max;
            }
        }
    }

    /// Control: `f32` keeps the butterfly at every floor.
    ///
    /// There is no `redux` for floats before sm_100, so this must reduce the
    /// same way at sm_75 and sm_80 and stay correct.
    ///
    /// Lane 0 writes `[sum, min, max]`.
    #[kernel]
    pub fn reduce_f32_control(mut out: DisjointSlice<f32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let lane = warp.thread_rank();
        let v = lane as f32;

        let r_sum = warp_reduce::<f32, Sum, _>(&warp, v);
        let r_min = warp_reduce::<f32, Min, _>(&warp, v);
        let r_max = warp_reduce::<f32, Max, _>(&warp, v);

        if lane == 0 {
            // SAFETY: the host allocates three elements for this kernel.
            unsafe {
                *out.get_unchecked_mut(0) = r_sum;
                *out.get_unchecked_mut(1) = r_min;
                *out.get_unchecked_mut(2) = r_max;
            }
        }
    }

    /// Control: a `WarpTile<16>` keeps the butterfly, and each half reduces
    /// only its own lanes.
    ///
    /// Lane `l` contributes its rank within the tile, so both tiles answer
    /// `0 + 1 + ... + 15 = 120`. A fast path that reduced the whole warp
    /// instead would answer `496` and be caught here.
    ///
    /// Tile leader `t` writes `out[t]`.
    #[kernel]
    pub fn reduce_subwarp_control(mut out: DisjointSlice<u32>) {
        let tile = this_thread_block().tiled_partition::<16>();
        let rank = tile.thread_rank();
        let tile_index = thread::index_1d().get() / 16;

        let r_sum = warp_reduce::<u32, Sum, _>(&tile, rank);

        if rank == 0 {
            // SAFETY: the host allocates one element per tile (two).
            unsafe {
                *out.get_unchecked_mut(tile_index as usize) = r_sum;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== warp_reduce: single-instruction form and its controls ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // One warp is the whole contract; a second would only repeat it.
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failed = false;

    // ===== Test 1: every u32 op =====
    println!("\n--- Test 1: warp_reduce over u32, all six ops ---");
    let mut u32_dev = DeviceBuffer::<u32>::zeroed(&stream, 6).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its writes.
    unsafe { module.reduce_u32_all_ops((stream).as_ref(), cfg, &mut u32_dev) }
        .expect("Kernel launch failed");
    let got = u32_dev.to_host_vec(&stream).unwrap();
    let want: [u32; 6] = [496, 0, 31, 0, u32::MAX, u32::MAX];
    println!("[sum, min, max, and, or, xor] = {:?}", got);
    println!("expected                      = {:?}", want);
    if got == want {
        println!("✓ every u32 reduction correct");
    } else {
        println!("✗ u32 reduction mismatch!");
        failed = true;
    }

    // ===== Test 2: signed ops across zero =====
    println!("\n--- Test 2: warp_reduce over i32, values -16..=15 ---");
    let mut i32_dev = DeviceBuffer::<i32>::zeroed(&stream, 3).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its writes.
    unsafe { module.reduce_i32_signed_ops((stream).as_ref(), cfg, &mut i32_dev) }
        .expect("Kernel launch failed");
    let got = i32_dev.to_host_vec(&stream).unwrap();
    let want: [i32; 3] = [-16, -16, 15];
    println!("[sum, min, max] = {:?}", got);
    println!(
        "expected        = {:?}   (unsigned min/max would read 0 / 4294967295)",
        want
    );
    if got == want {
        println!("✓ signed reductions correct");
    } else {
        println!("✗ signed reduction mismatch!");
        failed = true;
    }

    // ===== Test 3: f32 control =====
    println!("\n--- Test 3: control, f32 keeps the butterfly ---");
    let mut f32_dev = DeviceBuffer::<f32>::zeroed(&stream, 3).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its writes.
    unsafe { module.reduce_f32_control((stream).as_ref(), cfg, &mut f32_dev) }
        .expect("Kernel launch failed");
    let got = f32_dev.to_host_vec(&stream).unwrap();
    let want: [f32; 3] = [496.0, 0.0, 31.0];
    println!("[sum, min, max] = {:?}", got);
    println!("expected        = {:?}", want);
    if got == want {
        println!("✓ f32 reductions correct");
    } else {
        println!("✗ f32 reduction mismatch!");
        failed = true;
    }

    // ===== Test 4: sub-warp control =====
    println!("\n--- Test 4: control, WarpTile<16> reduces only its own lanes ---");
    let mut tile_dev = DeviceBuffer::<u32>::zeroed(&stream, 2).unwrap();
    // SAFETY: launch shape/resources match the kernel; the buffer covers its writes.
    unsafe { module.reduce_subwarp_control((stream).as_ref(), cfg, &mut tile_dev) }
        .expect("Kernel launch failed");
    let got = tile_dev.to_host_vec(&stream).unwrap();
    let want: [u32; 2] = [120, 120];
    println!("per-tile sums = {:?}", got);
    println!(
        "expected      = {:?}   (a whole-warp reduction would read 496)",
        want
    );
    if got == want {
        println!("✓ sub-warp tiles reduce independently");
    } else {
        println!("✗ sub-warp reduction mismatch!");
        failed = true;
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: warp_reduce is correct in both forms");
}
