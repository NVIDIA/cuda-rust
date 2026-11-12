/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU integration tests for `cutile::bench` (device-event timing).

use cuda_core::Context;
use cutile::bench::{do_bench, do_bench_controlled, do_bench_paired, BenchOptions};
use cutile::prelude::*;
use cutile_compiler::compile_control::{CompileControl, CompileInterrupted};
use std::time::Duration;

fn quick_opts() -> BenchOptions {
    BenchOptions {
        warmup: Duration::from_millis(5),
        rep: Duration::from_millis(20),
        min_reps: 3,
        max_reps: 50,
        clear_l2: true,
    }
}

#[test]
fn do_bench_times_a_kernel() {
    let device = Context::new(0).expect("device 0");
    let stream = device.new_stream().expect("stream");

    let result = do_bench(&stream, &quick_opts(), |s| {
        api::full(1.0f32, &[1 << 20])
            .sync_on(s)
            .map(|_| ())
            .map_err(Into::into)
    })
    .expect("do_bench");

    assert!(result.reps() >= 3, "at least min_reps timed reps");
    assert!(
        result.times_ms().iter().all(|t| t.is_finite() && *t >= 0.0),
        "all rep times finite and non-negative: {:?}",
        result.times_ms()
    );
    assert!(result.min_ms() <= result.median_ms());
    assert!(result.median_ms() > 0.0, "a 1M-element fill takes time");
}

#[test]
fn do_bench_paired_runs_equal_reps() {
    let device = Context::new(0).expect("device 0");
    let stream = device.new_stream().expect("stream");

    let (a, b) = do_bench_paired(
        &stream,
        &quick_opts(),
        |s| {
            api::full(1.0f32, &[1 << 20])
                .sync_on(s)
                .map(|_| ())
                .map_err(Into::into)
        },
        |s| {
            api::full(2.0f32, &[1 << 18])
                .sync_on(s)
                .map(|_| ())
                .map_err(Into::into)
        },
    )
    .expect("do_bench_paired");

    assert_eq!(a.reps(), b.reps(), "arms must be measured in pairs");
    assert!(a.reps() >= 3);
    assert!(a.median_ms() > 0.0 && b.median_ms() > 0.0);
}

#[test]
fn closure_errors_propagate() {
    let device = Context::new(0).expect("device 0");
    let stream = device.new_stream().expect("stream");

    let err = do_bench(&stream, &quick_opts(), |_| {
        Err(cutile::error::Error::Tensor(cutile::error::TensorError(
            "intentional".into(),
        )))
    });
    assert!(err.is_err(), "closure errors must not be swallowed");
}

#[test]
fn cancellation_never_returns_a_partial_measurement() {
    let device = Context::new(0).expect("device 0");
    let stream = device.new_stream().expect("stream");
    for cancel_before_start in [true, false] {
        let control = CompileControl::default();
        if cancel_before_start {
            control.cancel();
        }
        let mut calls = 0;
        let error = do_bench_controlled(&stream, &quick_opts(), &control, |_| {
            calls += 1;
            control.cancel();
            Ok(())
        })
        .unwrap_err();
        let cutile::error::Error::Anyhow(error) = error else {
            panic!("expected typed cancellation");
        };
        assert_eq!(error.downcast_ref(), Some(&CompileInterrupted::Cancelled));
        assert_eq!(calls, usize::from(!cancel_before_start));
    }
}
