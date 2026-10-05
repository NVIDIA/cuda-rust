/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel Library - exports generic kernels for use in other crates
//!
//! This crate demonstrates cross-crate kernel support:
//! - Kernels are defined here with #[kernel] attribute
//! - They can be generic over types
//! - The application crate imports and uses them
//! - PTX is generated when the application is compiled (monomorphization happens there)

use core::ops::{Add, Mul};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Generic scale kernel - multiplies each element by a factor.
    ///
    /// This kernel is exported from the library and can be instantiated
    /// with different types in the consuming application.
    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[idx_raw] * factor;
        }
    }

    /// Generic vector addition kernel.
    #[kernel]
    pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }

    /// Const-generic kernel instantiated by the consuming binary.
    #[kernel]
    pub fn add_const<const VALUE: u32>(input: &[u32], mut output: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = output.get_mut(idx) {
            *out_elem = input[idx_raw] + VALUE;
        }
    }

    /// A device-side helper function (not a kernel).
    /// This tests that non-kernel functions from library crates also work.
    #[inline]
    pub fn device_helper<T: Copy + Mul<Output = T>>(x: T, y: T) -> T {
        x * y
    }

    /// Kernel that uses the device helper function.
    #[kernel]
    pub fn scale_with_helper<T: Copy + Mul<Output = T>>(
        factor: T,
        input: &[T],
        mut out: DisjointSlice<T>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = device_helper(input[idx_raw], factor);
        }
    }
}
