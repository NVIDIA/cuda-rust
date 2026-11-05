/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a device static that holds a `Barrier` inside another type
//! (issue #1727).
//!
//! Only a static whose whole type is `SharedArray` or `Barrier` gets
//! per-block shared storage. `(Barrier, u32)` is placed in global memory, but
//! `&raw mut WRAPPED.0` is still typed as a shared-memory pointer. The build
//! used to succeed with an `addrspace(1)` to `addrspace(3)` cast, which LLVM
//! treats as undefined, and PTX that read an unassigned register.
//!
//! The compiler must reject the static at compile time instead. Declare the
//! barrier as its own `static mut BAR: Barrier`, as in `examples/barrier`.
//!
//! Usage:
//!   cargo oxide run error_static_nested_shared
//!
//! Expected: the build FAILS with a diagnostic containing (pinned):
//!
//! ```text
//! shared-memory types are only supported as the whole type of a static
//! ```

use cuda_device::barrier::mbarrier_init;
use cuda_device::{Barrier, kernel};

/// BUG UNDER TEST: a `Barrier` stored as a tuple field rather than as the
/// whole static. Pointers to the field are shared-memory pointers, but the
/// static itself would live in global memory.
static mut WRAPPED: (Barrier, u32) = (Barrier::UNINIT, 0);

/// # Safety
///
/// Never launched: the device compilation is expected to fail before a host
/// binary that could launch it exists.
#[kernel]
pub unsafe fn k(out: *mut u64) {
    let p = unsafe { &raw mut WRAPPED.0 };
    unsafe { mbarrier_init(p, 1) };
    unsafe { out.write(p as u64) };
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
