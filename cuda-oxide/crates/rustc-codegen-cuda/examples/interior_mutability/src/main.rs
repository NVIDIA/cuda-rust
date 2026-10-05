/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for `core::cell::Cell` and `core::cell::UnsafeCell`.
//!
//! The kernels use shared references to guarded local structs so the device path
//! exercises interior mutability through field projections and raw pointers
//! without requiring allocator support.
//!
//! Usage:
//!   cargo oxide run interior_mutability
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run interior_mutability

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;
    use core::cell::{Cell, UnsafeCell};

    const LEFT_GUARD: u32 = 0xA5A5_5A5A;
    const RIGHT_GUARD: u32 = 0x5A5A_A5A5;

    #[repr(C)]
    struct CellState {
        left_guard: u32,
        value: Cell<u32>,
        right_guard: u32,
    }

    #[repr(C)]
    struct UnsafeCellState {
        left_guard: u32,
        value: UnsafeCell<u32>,
        right_guard: u32,
    }

    /// Exercise `Cell::get`, `Cell::set`, and `Cell::replace` through `&CellState`.
    #[kernel]
    pub fn cell_operations(seed: u32, mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let state = CellState {
            left_guard: LEFT_GUARD,
            value: Cell::new(seed),
            right_guard: RIGHT_GUARD,
        };
        let shared = &state;

        let before = shared.value.get();
        shared.value.set(seed + 7);
        let after_set = shared.value.get();
        let replaced = shared.value.replace(seed + 19);
        let after_replace = shared.value.get();

        unsafe {
            // One thread owns this output region for the duration of the kernel.
            let ptr = out.as_mut_ptr();
            ptr.write(before);
            ptr.add(1).write(after_set);
            ptr.add(2).write(replaced);
            ptr.add(3).write(after_replace);
            ptr.add(4).write(shared.left_guard);
            ptr.add(5).write(shared.right_guard);
        }
    }

    /// Exercise `UnsafeCell::get` followed by raw-pointer load and store.
    #[kernel]
    pub fn unsafe_cell_operations(seed: u32, mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let state = UnsafeCellState {
            left_guard: LEFT_GUARD,
            value: UnsafeCell::new(seed),
            right_guard: RIGHT_GUARD,
        };
        let shared = &state;
        let value_ptr = shared.value.get();

        unsafe {
            // SAFETY: this kernel has one active thread and no other access to
            // the contents of `state.value` while this raw pointer is used.
            let before = value_ptr.read();
            value_ptr.write(seed + 31);
            let after = value_ptr.read();

            // One thread owns this output region for the duration of the kernel.
            let ptr = out.as_mut_ptr();
            ptr.write(before);
            ptr.add(1).write(after);
            ptr.add(2).write(shared.left_guard);
            ptr.add(3).write(shared.right_guard);
        }
    }
}

fn main() {
    println!("=== interior_mutability ===");

    const SEED: u32 = 10;
    const LEFT_GUARD: u32 = 0xA5A5_5A5A;
    const RIGHT_GUARD: u32 = 0x5A5A_A5A5;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(1);

    let mut cell_out = DeviceBuffer::<u32>::zeroed(&stream, 6).expect("cell output allocation");
    unsafe { module.cell_operations(&stream, cfg, SEED, &mut cell_out) }
        .expect("cell_operations launch");

    let cell_result = cell_out.to_host_vec(&stream).expect("copy Cell results");
    assert_eq!(cell_result[0], SEED, "Cell::get initial value");
    println!("PASS: Cell::get");

    assert_eq!(cell_result[1], SEED + 7, "Cell::set result");
    println!("PASS: Cell::set");

    assert_eq!(cell_result[2], SEED + 7, "Cell::replace returned value");
    assert_eq!(cell_result[3], SEED + 19, "Cell::replace stored value");
    println!("PASS: Cell::replace");

    assert_eq!(cell_result[4], LEFT_GUARD, "Cell left guard");
    assert_eq!(cell_result[5], RIGHT_GUARD, "Cell right guard");
    println!("PASS: Cell guard fields preserved");

    let mut unsafe_out =
        DeviceBuffer::<u32>::zeroed(&stream, 4).expect("UnsafeCell output allocation");
    unsafe { module.unsafe_cell_operations(&stream, cfg, SEED, &mut unsafe_out) }
        .expect("unsafe_cell_operations launch");

    let unsafe_result = unsafe_out
        .to_host_vec(&stream)
        .expect("copy UnsafeCell results");
    assert_eq!(unsafe_result[0], SEED, "UnsafeCell::get initial read");
    assert_eq!(unsafe_result[1], SEED + 31, "UnsafeCell::get write/read");
    println!("PASS: UnsafeCell::get raw-pointer read/write");

    assert_eq!(unsafe_result[2], LEFT_GUARD, "UnsafeCell left guard");
    assert_eq!(unsafe_result[3], RIGHT_GUARD, "UnsafeCell right guard");
    println!("PASS: UnsafeCell guard fields preserved");

    println!("PASS: interior_mutability");
}
