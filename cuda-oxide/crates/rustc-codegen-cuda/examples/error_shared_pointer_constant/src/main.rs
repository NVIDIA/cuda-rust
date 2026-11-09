/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a constant whose type stores a pointer to a `Barrier`.
//!
//! A `&Barrier` or `*const SharedArray` translates to a shared-memory
//! (`addrspace(3)`) pointer. A pointer stored inside a constant is a
//! relocation to an initialized device global, which always lives in the
//! global or constant address space. Here the inner `&Barrier` points at
//! anonymous constant memory, so reading `*INDIRECT` would reinterpret a
//! global address as a shared one.
//!
//! The importer must refuse the constant at compile time instead.
//!
//! Usage:
//!   cargo oxide build error_shared_pointer_constant
//!
//! Expected: the build FAILS with a diagnostic containing (pinned):
//!
//! ```text
//! stores a pointer to shared memory
//! ```

use cuda_device::{Barrier, device, kernel};

/// A constant whose inner pointer has a shared-memory type but points at
/// anonymous constant memory.
const INDIRECT: &&Barrier = &&Barrier::UNINIT;

/// Reads the pointer stored in `INDIRECT`.
#[device]
#[inline(never)]
pub fn barrier_address() -> *const Barrier {
    // the stored shared-memory pointer would be a relocation to an ordinary
    // read-only global, so the importer must refuse it
    *INDIRECT
}

/// # Safety
///
/// Never launched: the device compilation is expected to fail before a host
/// binary that could launch it exists.
#[kernel]
pub unsafe fn shared_pointer_kernel(output: *mut usize) {
    unsafe { output.write(barrier_address() as usize) };
}

/// Host entry point; never reached because device compilation fails first.
fn main() {
    println!("This negative example should fail during device compilation.");
}
