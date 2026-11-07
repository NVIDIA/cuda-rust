/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a reference to a constant that holds a `DisjointSlice`.
//!
//! `DisjointSlice::empty()` folds to a constant, and import rebuilds it as a
//! slice value when it is passed by value. A *reference* to such a constant,
//! such as the promoted `&(Slice::empty(), 1)` below, would be materialized
//! as a value in the current function and its address taken, so a helper
//! that returns the reference as `&'static` would hand its caller a dead
//! local. Import refuses the reference instead, for a bare slice and for a
//! slice nested in an aggregate alike.
//!
//! Usage:
//!   cargo oxide run error_disjoint_slice_const_ref
//!
//! Expected: the build FAILS with this exact diagnostic (pinned):
//!
//! ```text
//! a reference to a constant holding a DisjointSlice is not supported
//! ```
//!
//! pointing at the `TAGGED` read in `tagged_ref`.

use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// A runtime-width slice, whose constant image ends in padding
type Slice = DisjointSlice<'static, u32, Runtime2DIndex>;

/// A promoted reference to an aggregate holding an empty slice
const TAGGED: &(Slice, u32) = &(Slice::empty(), 1);

#[cuda_module]
mod kernels {
    use super::*;

    /// Return the promoted reference from a call that is not inlined, so the
    /// reference outlives the helper's frame.
    #[device]
    #[inline(never)]
    fn tagged_ref() -> &'static (Slice, u32) {
        TAGGED
    }

    /// Read the slice's length through the returned reference.
    #[kernel]
    pub fn read_tagged_len(mut out: DisjointSlice<u32>) {
        let tagged = tagged_ref();
        let n = tagged.0.len() as u32 + tagged.1;
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = n;
        }
    }
}

fn main() {
    println!("=== error_disjoint_slice_const_ref ===");
    println!("This example is intentionally broken: a device helper returns a");
    println!("reference to a constant holding a DisjointSlice. The build must");
    println!("FAIL at codegen time with a diagnostic refusing the reference.");
    println!();
    println!("If you see this message, the build did NOT fail and the");
    println!("diagnostic regression has returned.");
}
