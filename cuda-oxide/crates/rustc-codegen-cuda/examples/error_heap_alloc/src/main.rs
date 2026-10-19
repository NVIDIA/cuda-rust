/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: heap allocation (`Vec`, `Box`) inside a kernel
//! (issue #108).
//!
//! There is no device-side `#[global_allocator]`, so the Rust allocator
//! shims (`__rust_alloc` and friends) have no implementation on the GPU.
//! Without a dedicated diagnostic the collector silently walks into
//! `alloc::` internals and the user gets an inscrutable const-translation
//! error spanned into `alloc/src/boxed.rs`.
//!
//! Usage:
//!   cargo oxide run error_heap_alloc
//!
//! Expected: the build FAILS with this exact diagnostic (pinned):
//!
//! ```text
//! error: heap allocation is not supported in kernels (no device
//!        allocator); use fixed-size arrays or SharedArray
//! ```
//!
//! pointing at the `vec![...]` line below, with notes naming the kernel
//! and the allocator entry point that was reached.

use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn heap_alloc_kernel(out: DisjointSlice<u32>) {
    let mut out = out;
    let idx = thread::index_1d();
    // BUG UNDER TEST: `vec!` heap-allocates; there is no device allocator.
    // The vec! IS the shape under test, so the "use an array" lint is
    // exactly the rewrite this fixture must not perform.
    #[allow(clippy::useless_vec)]
    let v = vec![1u32, 2, 3, 4];
    if let Some(slot) = out.get_mut(idx) {
        *slot = v[0];
    }
}

fn main() {
    println!("=== error_heap_alloc ===");
    println!("This example is intentionally broken: the kernel heap-allocates");
    println!("with vec!. The build must FAIL at codegen time with a diagnostic");
    println!("naming the kernel and explaining that no device allocator exists.");
    println!();
    println!("If you see this message, the build did NOT fail and the");
    println!("diagnostic regression has returned.");
}
