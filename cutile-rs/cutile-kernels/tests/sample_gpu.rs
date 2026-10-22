/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU execution coverage for `sample`.

use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;
use cutile_kernels::sampling::sample;

const STACK: usize = 64_000_000;
const BLOCK: usize = 8;

/// Launches `sample` and returns one token per row.
fn run(
    batch: usize,
    logits: Vec<f32>,
    u: Vec<f32>,
    len: i32,
    temperature: f32,
    k: i32,
    p: f32,
) -> Vec<u32> {
    let logits_t = api::copy_host_vec_to_device(&Arc::new(logits))
        .sync()
        .expect("upload logits")
        .reshape(&[batch, BLOCK])
        .expect("reshape logits");
    let u_t = api::copy_host_vec_to_device(&Arc::new(u))
        .sync()
        .expect("upload u");
    let mut out = api::zeros::<u32>(&[batch]).sync().expect("alloc out");
    let _ = unsafe {
        sample(
            &logits_t,
            &u_t,
            (&mut out).partition([1]),
            len,
            temperature,
            k,
            p,
        )
    }
    .sync()
    .expect("launch sample");
    out.to_host_vec().sync().expect("read out")
}

#[test]
fn sample_semantics() {
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(|| {
            let row_a = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let a = run(2, [row_a, row_a].concat(), vec![0.0, 0.999], 4, 1.0, 1, 1.0);
            assert_eq!(a, vec![2, 2], "top-k=1 should always pick the argmax");

            let row_b = [0.0f32; 8];
            let b = run(
                4,
                [row_b, row_b, row_b, row_b].concat(),
                vec![0.1, 0.3, 0.6, 0.9],
                4,
                1.0,
                8,
                1.0,
            );
            assert_eq!(b, vec![0, 1, 2, 3], "inverse-CDF bin selection");

            let c = run(
                1,
                vec![0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.99],
                4,
                1.0,
                8,
                0.5,
            );
            assert_eq!(c, vec![2], "top-p should keep only the dominant token");

            let d = run(
                1,
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.999],
                3,
                1.0,
                8,
                1.0,
            );
            assert_eq!(d, vec![2], "masked vocab tail must not be sampled");
        })
        .expect("spawn")
        .join()
        .expect("sample_semantics thread panicked");
}
