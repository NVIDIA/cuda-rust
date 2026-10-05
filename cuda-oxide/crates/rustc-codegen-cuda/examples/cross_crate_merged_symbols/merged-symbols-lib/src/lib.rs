/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Library half of the `cross_crate_merged_symbols` regression test (#1277).
//!
//! Everything here has a same-shaped twin in the binary crate, on purpose:
//! each pair used to collide when `load_all_ptx_bundles_merged` concatenated
//! the two crates' PTX bundles.
//!
//! - `lib_shared_roundtrip` owns a static `SharedArray`: both crates' first
//!   shared allocation used to be named `__shared_mem_0`.
//! - `lib_global_read` reads this crate's `static mut`: both crates' first
//!   ordinary device global used to be named `__device_global_0`.
//! - `lib_pool_addr` reaches the dynamic pool through [`pool_base_addr`],
//!   which the binary crate's kernel calls too — so both bundles carry a
//!   copy of the helper, each with the pool alignment of the launch
//!   contracts that reach it in that crate (128 here, 16 in the binary).
//!   Before crate-namespaced symbols, the merged module kept whichever
//!   `.extern .shared .align` declaration came first.

use cuda_device::{
    DisjointSlice, DynamicSharedArray, SharedArray, cuda_module, kernel, launch_bounds,
    launch_contract, thread,
};

/// Value published through this crate's device global, so the host can tell
/// whose global a merged-module read came from.
pub const LIB_TAG: u32 = 0x11AA_22BB;

static mut LIB_SLOT: u32 = LIB_TAG;

/// Dynamic shared-memory helper shared by kernels in BOTH crates.
///
/// The pool symbol is owned by this function, and its alignment is the
/// strongest launch-contract requirement that reaches it in the crate being
/// compiled — which differs between the two crates of this example.
pub fn pool_base_addr() -> u64 {
    let smem: *mut u64 = DynamicSharedArray::<u64>::get();
    smem as u64
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn lib_shared_roundtrip(mut out: DisjointSlice<f64>) {
        static mut S: SharedArray<f64, 1> = SharedArray::UNINIT;
        if thread::threadIdx_x() == 0 {
            // SAFETY: only thread 0 accesses slot 0, before the block barrier.
            unsafe { S[0] = 1.5 };
        }
        thread::sync_threads();
        if thread::threadIdx_x() == 0 && !out.is_empty() {
            // SAFETY: thread 0 only, length checked.
            unsafe { *out.as_mut_ptr() = S[0] };
        }
    }

    #[kernel]
    pub fn lib_global_read(mut out: DisjointSlice<u32>) {
        if thread::threadIdx_x() == 0 && !out.is_empty() {
            // SAFETY: thread 0 only, length checked; the static is only read.
            unsafe { *out.as_mut_ptr() = LIB_SLOT };
        }
    }

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        dynamic_shared = 256,
        dynamic_shared_alignment = 128,
    )]
    pub fn lib_pool_addr(mut out: DisjointSlice<u64>) {
        if thread::threadIdx_x() == 0 && !out.is_empty() {
            // SAFETY: thread 0 only, length checked.
            unsafe { *out.as_mut_ptr() = pool_base_addr() };
        }
    }
}
