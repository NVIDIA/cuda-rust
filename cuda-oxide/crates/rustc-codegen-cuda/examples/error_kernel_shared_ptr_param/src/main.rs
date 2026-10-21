/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a `#[kernel]` parameter struct that holds a `SharedPtr`.
//!
//! A kernel receives its parameters in `.param` space, filled by the host at
//! launch. Shared memory is allocated per block by the device, so the host
//! holds no shared-memory address it could pass. A bare `SharedPtr<T>`
//! parameter is refused for that reason (`error_kernel_shared_param` covers
//! the bare case with `*mut Barrier`), and a struct or array that carries one
//! must be refused the same way. Before the exporter looked inside
//! aggregates, `Args { p: SharedPtr<f32>, n: u32 }` compiled and the kernel
//! converted the host-supplied bytes with `cvta.shared.u64` as if they were a
//! shared address.
//!
//! Usage:
//!   cargo oxide run error_kernel_shared_ptr_param
//!
//! Expected: the build FAILS with a diagnostic containing (pinned):
//!
//! ```text
//! holds a pointer into shared memory
//! ```

use cuda_device::{SharedPtr, kernel};

/// Kernel arguments that carry a typed shared pointer next to a plain field.
#[derive(Clone, Copy)]
pub struct Args {
    pub p: SharedPtr<f32>,
    pub n: u32,
}

/// # Safety
///
/// Never launched: the device compilation is expected to fail before a host
/// binary that could launch it exists.
#[kernel]
pub unsafe fn shared_ptr_param_kernel(args: Args) {
    // BUG UNDER TEST: `args.p` points into shared memory, which the host
    // cannot supply at launch. The refusal fires on the entry signature
    // alone; the body deliberately does nothing else, so the diagnostic
    // under test is the only error this fixture can produce.
    let _ = args;
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
