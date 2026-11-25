/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU integration tests for `cutile::bench` (device-event timing).

use cuda_core::Context;
use cutile::bench::{
    do_bench, do_bench_controlled, do_bench_paired, do_bench_paired_controlled, BenchOptions,
};
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

#[test]
fn controlled_pairs_reverse_order_and_preserve_sample_alignment() {
    let stream = Context::new(0).unwrap().new_stream().unwrap();
    let order = std::cell::RefCell::new(Vec::new());
    let opts = BenchOptions {
        warmup: Duration::ZERO,
        min_reps: 4,
        max_reps: 4,
        ..quick_opts()
    };
    let (a, b) = do_bench_paired_controlled(
        &stream,
        &opts,
        &CompileControl::default(),
        |s| {
            order.borrow_mut().push('a');
            api::full(1.0f32, &[1 << 20])
                .sync_on(s)
                .map(|_| ())
                .map_err(Into::into)
        },
        |s| {
            order.borrow_mut().push('b');
            api::full(2.0f32, &[1 << 18])
                .sync_on(s)
                .map(|_| ())
                .map_err(Into::into)
        },
    )
    .unwrap();
    // One warmup pair and one estimate pair precede the four measured pairs.
    assert_eq!(
        &*order.borrow(),
        &['a', 'b', 'a', 'b', 'a', 'b', 'b', 'a', 'a', 'b', 'b', 'a']
    );
    assert_eq!((a.reps(), b.reps()), (4, 4));
    assert!(a
        .times_ms()
        .iter()
        .chain(b.times_ms())
        .all(|&v| v.is_finite() && v > 0.0));
}

#[test]
fn cancelling_between_arms_never_returns_an_unmatched_pair() {
    let stream = Context::new(0).unwrap().new_stream().unwrap();
    let control = CompileControl::default();
    let (mut a_calls, mut b_calls) = (0, 0);
    let opts = BenchOptions {
        warmup: Duration::ZERO,
        ..quick_opts()
    };
    let error = do_bench_paired_controlled(
        &stream,
        &opts,
        &control,
        |_| {
            a_calls += 1;
            if a_calls == 3 {
                control.cancel();
            } // First measured arm.
            Ok(())
        },
        |_| {
            b_calls += 1;
            Ok(())
        },
    )
    .unwrap_err();
    let cutile::error::Error::Anyhow(error) = error else {
        panic!("expected typed cancellation")
    };
    assert_eq!(error.downcast_ref(), Some(&CompileInterrupted::Cancelled));
    assert_eq!((a_calls, b_calls), (3, 2));
}
