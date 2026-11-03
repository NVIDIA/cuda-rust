/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Grid-scoped intrinsics: barriers and queries that span the entire kernel launch.
//!
//! The grid is the top-level CUDA execution scope: every block in every cluster
//! belongs to it. Most kernels never need a grid handle — block-level work is
//! independent. The two cases where this module matters are:
//!
//! - **Iterative algorithms** (rehash, multi-pass solvers, BFS frontiers) that
//!   need every block to finish phase `k` before any block starts phase `k+1`.
//! - **Cooperative reductions** that produce one final value visible to every
//!   thread in the launch.
//!
//! # The cooperative launch contract
//!
//! [`sync`] is **only valid in cooperative kernel launches**. The host must
//! use `#[cooperative_launch]` on a `#[cuda_module]` kernel, call
//! `cuda_core::launch_kernel_cooperative`, or use
//! `unsafe { cuda_launch! { cooperative: true, ... } }`; a normal launch deadlocks
//! because not every block is guaranteed to be co-resident on the GPU and
//! the driver does not populate the per-launch grid workspace pointer.
//!
//! # Implementation
//!
//! [`sync`] is the byte-for-byte equivalent of NVCC's
//! `cooperative_groups::grid_group::sync()` lowering — there is no external
//! dependency on `libcudadevrt.a`:
//!
//! 1. Block-wide `bar.sync 0` so every thread in the block is at the barrier.
//! 2. Block leader (`tid == 0`) reads the per-launch grid workspace pointer
//!    out of PTX environment registers `%envreg1` (high 32 bits) and
//!    `%envreg2` (low 32 bits). The CUDA driver writes the pointer there
//!    automatically for cooperative launches. (Convention matches the
//!    public CUDA driver-ABI header.)
//! 3. Block leader does an atomic add (release semantics, gpu scope) on the
//!    workspace's barrier counter. The "GPU master" block (`blockIdx == 0`)
//!    contributes a value crafted so that the counter's high bit flips
//!    exactly when every block has arrived; every other block contributes 1.
//! 4. Block leader spin-loads the counter with acquire semantics until the
//!    counter's high bit has flipped relative to its arrival value.
//! 5. Final block-wide `bar.sync 0` releases every thread.
//!
//! The phase-flip approach is self-resetting: the counter's high bit toggles
//! on every successful sync, so consecutive `grid::sync()` calls within a
//! single launch (or back-to-back launches that share the workspace) work
//! without any host intervention.

use crate::atomic::{AtomicOrdering, DeviceAtomicU32};
use crate::thread;

include!("generated/grid_sreg.rs");
include!("generated/grid_dependency.rs");

/// Layout of the per-launch grid workspace populated by the CUDA driver.
///
/// Matches `cooperative_groups::details::grid_workspace` in the CUDA toolkit:
/// the first u32 is the workspace size, the second u32 is the barrier counter.
#[repr(C)]
struct GridWorkspace {
    _ws_size: u32,
    barrier: u32,
}

/// Test whether the high bit of `current` differs from `prev` — i.e., whether
/// the grid barrier has flipped phase since `prev` was sampled.
#[inline(always)]
fn bar_has_flipped(prev: u32, current: u32) -> bool {
    ((prev ^ current) & 0x8000_0000) != 0
}

/// Synchronise every thread in every block of the cooperative grid.
///
/// All `gridDim.x * gridDim.y * gridDim.z` blocks must reach this call;
/// any block that doesn't deadlocks the rest. Memory writes done by any
/// thread before the barrier are visible to every thread after.
///
/// # Cooperative launch required
///
/// The kernel must be launched with cooperative semantics. From the host,
/// either mark the kernel `#[cooperative_launch]` inside a `#[cuda_module]`
/// (preferred), or use the unsafe lower-level paths:
///
/// ```ignore
/// // SAFETY: args match my_kernel's signature.
/// unsafe {
///     cuda_launch! {
///         kernel: my_kernel,
///         stream: stream,
///         module: module,
///         config: cfg,
///         cooperative: true,
///         args: [/* ... */]
///     }
/// }?;
/// // or
/// unsafe { cuda_core::launch_kernel_cooperative(&func, grid, block, 0, &stream, &mut params) }?;
/// ```
///
/// A non-cooperative launch will deadlock at the first `grid::sync()` call.
///
/// # Example
///
/// ```rust,ignore
/// #[kernel]
/// pub fn rehash(buckets: &mut [Bucket]) {
///     let gid = thread::index_1d();
///     // Phase 1: every thread reads its old slot.
///     let snapshot = read_bucket(buckets, gid);
///
///     grid::sync();
///
///     // Phase 2: every thread writes to its NEW slot. Safe because
///     // phase 1 reads completed across the entire grid before any
///     // phase 2 write starts.
///     write_to_new_slot(buckets, gid, snapshot);
/// }
/// ```
#[inline(always)]
pub fn sync() {
    let hi = envreg1() as u64;
    let lo = envreg2() as u64;
    let workspace_addr = (hi << 32) | lo;
    let workspace = workspace_addr as *mut GridWorkspace;
    let barrier_ptr = unsafe { &raw mut (*workspace).barrier };

    thread::sync_threads();

    let is_cta_master = thread::threadIdx_x() | thread::threadIdx_y() | thread::threadIdx_z() == 0;

    if is_cta_master {
        let expected = thread::gridDim_x()
            .wrapping_mul(thread::gridDim_y())
            .wrapping_mul(thread::gridDim_z());
        let is_gpu_master = thread::blockIdx_x() | thread::blockIdx_y() | thread::blockIdx_z() == 0;

        let nb = if is_gpu_master {
            0x8000_0000u32.wrapping_sub(expected.wrapping_sub(1))
        } else {
            1u32
        };

        let barrier = unsafe { DeviceAtomicU32::from_ptr(barrier_ptr) };
        let old_arrive = barrier.fetch_add(nb, AtomicOrdering::Release);

        loop {
            let current = barrier.load(AtomicOrdering::Acquire);
            if bar_has_flipped(old_arrive, current) {
                break;
            }
        }
    }

    thread::sync_threads();
}
