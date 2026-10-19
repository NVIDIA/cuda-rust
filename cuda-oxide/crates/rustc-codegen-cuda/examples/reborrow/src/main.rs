/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `Rvalue::Reborrow` translation in device code.
//!
//! `feature(reborrow)` (rust-lang/rust#145612) lets a user ADT that
//! implements the `core::marker::Reborrow` trait be passed by value
//! repeatedly: each use site gets an implicit reborrow instead of a move.
//! Since nightly-2026-08-28 those implicit reborrows appear in MIR as a
//! dedicated `Rvalue::Reborrow(Ty, Mutability, Place)`. The importer arm
//! has two halves, and this example exercises both:
//!
//! - `Mutability::Mut` (the `Reborrow` trait): the target type equals the
//!   source type, a plain place read. GVN folds this variant into plain
//!   copies at mir-opt-level>0, so this crate's Cargo.toml scopes
//!   `-Zmir-opt-level=0` to itself (profile-rustflags); the rvalue then
//!   reaches the importer intact in every build of this example.
//! - `Mutability::Not` (the `CoerceShared` trait): a same-layout coercion
//!   into a distinct shared-view ADT, taking the importer's transmute-cast
//!   path. GVN leaves this variant unoptimised, so it reaches the importer
//!   in release builds too.
//!
//! Both halves are verified by stubbing the arm: the stub fails this
//! example's device build. The `Rvalue::Reborrow` the importer receives is
//! the same at either MIR level; only the neighbourhood differs (helpers
//! stay un-inlined at level 0), and the inlined neighbourhood is exercised
//! by every other example.
//!
//! The kernel below passes a `Reborrow` wrapper around `&mut f32` to a
//! helper twice, then passes it twice where its `CoerceShared` view type
//! is expected; without the importer's `Rvalue::Reborrow` arm this fails
//! device codegen with an "Unsupported construct" error.

#![feature(reborrow)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Reborrowable exclusive view of one element. Exactly one lifetime
    /// parameter, as the `Reborrow` trait requires.
    pub struct MutView<'a> {
        data: &'a mut f32,
    }

    impl<'a> core::marker::Reborrow for MutView<'a> {}

    /// Shared view of one element: the `CoerceShared` target of `MutView`.
    /// The trait's coherence rules force it to have the identical memory
    /// layout to `MutView`.
    #[derive(Clone, Copy)]
    pub struct View<'a> {
        data: &'a f32,
    }

    impl<'a> core::marker::CoerceShared<View<'a>> for MutView<'a> {}

    fn add_one(v: MutView<'_>) {
        *v.data += 1.0;
    }

    fn read(v: View<'_>) -> f32 {
        *v.data
    }

    #[kernel]
    pub fn reborrow(mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let mut sum = 0.0;
        if let Some(elem) = c.get_mut(idx) {
            let v = MutView { data: elem };
            // Each call takes `v` by value; the compiler inserts an implicit
            // reborrow (`Rvalue::Reborrow(MutView, Mut, v)`) so `v` stays
            // live for the second call. That second call is the proof the
            // reborrow happened.
            add_one(v);
            add_one(v);
            // Each `read(v)` passes the exclusive view where the shared
            // `View` is expected; the compiler inserts an implicit
            // `Rvalue::Reborrow(View, Not, v)` (CoerceShared) at each call
            // site. Reading twice proves both coercions happened and `v`
            // stayed live throughout.
            let a = read(v);
            let b = read(v);
            sum = a + b;
        }
        // Written outside `v`'s scope: `CoerceShared` pins the shared view
        // to the source's reborrow lifetime (its coherence rules demand the
        // same lifetime argument on both), so a write through `v` after a
        // `read(v)` is rejected by borrowck (E0506). `ThreadIndex` is not
        // `Copy`, so re-derive it for the second lookup.
        if let Some(elem) = c.get_mut(thread::index_1d()) {
            *elem = sum;
        }
    }
}

fn main() {
    println!("=== Rvalue::Reborrow device regression ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; the buffer covers its
    // accesses.
    unsafe { module.reborrow(&stream, LaunchConfig::for_num_elems(N as u32), &mut c_dev) }
        .expect("Kernel launch failed");

    let c_host = c_dev.to_host_vec(&stream).unwrap();

    // 0.0 -> 2.0 (two Mut-reborrow increments) -> 4.0 (sum of two
    // CoerceShared reads written back).
    let mut errors = 0;
    for (i, &value) in c_host.iter().enumerate() {
        if (value - 4.0).abs() > 1e-6 {
            if errors < 5 {
                eprintln!("  Error at [{i}]: expected 4.0, got {value}");
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!(
            "✓ SUCCESS: all {N} elements saw two Mut reborrows and two CoerceShared reborrows"
        );
    } else {
        println!("✗ FAILED: {errors} errors");
        std::process::exit(1);
    }
}
