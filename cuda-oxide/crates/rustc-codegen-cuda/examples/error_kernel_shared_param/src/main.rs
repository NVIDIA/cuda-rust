/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a `#[kernel]` parameter that points into shared memory.
//!
//! A kernel receives its parameters in `.param` space, filled by the host at
//! launch. Shared memory is allocated per block by the device, so the host
//! holds no shared-memory address it could pass. `*mut Barrier` lowers to an
//! `addrspace(3)` pointer, and letting it through to the entry signature
//! produces `.ptr .shared` PTX that ptxas assembles but the driver refuses at
//! module load, taking every other kernel in the module down with it.
//!
//! The exporter must reject the signature at compile time instead. A barrier
//! belongs in a device-side `static mut BAR: Barrier`, as in
//! `examples/barrier`.
//!
//! Usage:
//!   cargo oxide run error_kernel_shared_param
//!
//! Expected: the build FAILS with a diagnostic containing (pinned):
//!
//! ```text
//! is a pointer into shared memory
//! ```

use cuda_device::{barrier::Barrier, kernel};

/// # Safety
///
/// Never launched: the device compilation is expected to fail before a host
/// binary that could launch it exists.
#[kernel]
pub unsafe fn shared_param_kernel(barrier: *mut Barrier) {
    // BUG UNDER TEST: `barrier` points into shared memory, which the host
    // cannot supply at launch. The refusal fires on the entry signature
    // alone; the body deliberately does nothing else, so the diagnostic
    // under test is the only error this fixture can produce.
    let _ = barrier;
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
