/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for matched `opt`/`llc` toolchain resolution.
//!
//! The pipeline resolves `opt` and `llc` as a same-LLVM-major pair, so a
//! pinned `CUDA_OXIDE_LLC` always gets compatible middle-end IR:
//!
//!   CUDA_OXIDE_LLC=/usr/bin/llc-21 cargo oxide build toolchain_pairing_smoke --arch sm_90
//!
//! picks an LLVM 21 `opt` (or skips the middle-end with a warning when no
//! same-major `opt` exists). The kernel is shaped so `opt -O2`'s inliner
//! inserts llvm.lifetime.start/end markers, whose signature changed in
//! LLVM 22 (the `i64` size parameter was dropped): exactly the IR that a
//! mismatched opt/llc pair fails the verifier on. Guards the fix for
//! issue #150.

use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, gpu_printf, kernel, thread};

    // Kernel shape kept verbatim from issue #150: this exact pattern is
    // what makes opt's inliner introduce llvm.lifetime.start/end markers.
    #[allow(clippy::redundant_pattern_matching)]
    #[kernel]
    pub fn misbehave(mut dst: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        if let Some(_) = dst.get_mut(idx) {
            check3();
        }
    }

    fn check3() {
        {
            let (r, borrow) = 0x4bf9_0000u32.overflowing_sub(0xf329_0000);
            gpu_printf!("{:x}, {}\n", r, borrow);
        }
    }
}

fn main() {
    // No GPU needed: this repro only exercises device compilation. The
    // SUCCESS marker satisfies the smoketest standard-category pass rule.
    println!("toolchain_pairing_smoke: device compile-only repro - SUCCESS");
}
