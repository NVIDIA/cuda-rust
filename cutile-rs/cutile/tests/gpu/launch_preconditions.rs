/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The launcher enforces declared preconditions end to end: a launch whose
//! real shapes violate a declared fact is rejected before any GPU work,
//! naming the fact. The compile-time half (declared facts entail the
//! binding obligations, nothing emitted anywhere) lives in
//! `optimization_hints.rs`; this file holds the launch half, which needs a
//! real device. Moved here from `optimization_hints.rs`, which the CI CPU
//! job runs on driverless runners.

#![allow(deprecated)] // the kernel deliberately exercises the with_bounds path

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod launch_preconditions_module {
    use cutile::core::*;

    /// Declared divisibility: with `dim(x, k) % 64 == 0` declared (and so
    /// verified by the launcher before the kernel runs), both binding
    /// obligations are entailed at JIT.
    #[cutile::entry(
        preconditions = (
            dim(x, 0) % 64 == 0,
            dim(x, 1) % 64 == 0,
        )
    )]
    fn declared_divisibility_binding<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = Dim::new(x.shape()[0] / BM);
        let n = Dim::new(x.shape()[1] / BN);
        let part = x.partition(shape![BM, BN]).with_bounds((m, n));
        for i in m {
            for j in n {
                let tile = part.load(coord((i, j)));
                z.store(tile);
            }
        }
    }
}

#[test]
fn the_launcher_enforces_a_declared_divisibility() {
    common::with_test_stack(|| {
        use cutile::tensor::PartitionMut;
        let generics = || vec!["64".to_string(), "64".to_string()];
        let x = cutile::api::zeros::<f32>(&[128, 128])
            .sync()
            .expect("alloc x");
        let mut z = cutile::api::zeros::<f32>(&[64, 64])
            .sync()
            .expect("alloc z");
        launch_preconditions_module::declared_divisibility_binding(
            (&mut z).partition([64, 64]),
            &x,
        )
        .generics(generics())
        .sync()
        .expect("a divisible shape must launch");

        let bad = cutile::api::zeros::<f32>(&[100, 128])
            .sync()
            .expect("alloc bad");
        let result = launch_preconditions_module::declared_divisibility_binding(
            (&mut z).partition([64, 64]),
            &bad,
        )
        .generics(generics())
        .sync();
        let Err(err) = result else {
            panic!("a non-divisible shape must be rejected at launch");
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("% 64 == 0 failed"),
            "expected the precondition failure to name the declared fact, got: {msg}"
        );
    });
}
