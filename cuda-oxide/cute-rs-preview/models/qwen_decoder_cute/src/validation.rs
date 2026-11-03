/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent synthetic checks, including observable BF16 rounding boundaries.
use crate::{
    model::{Batch, Runner},
    operator::Result,
    reference,
};
use cuda_core::{CudaContext, DeviceBuffer};
use cute_rs::{
    BFloat16,
    decoder::{BiasWeights, EmbeddingWeights, TokenIds},
    ffn::{InputProjection, ProjectionWeights},
};

/// Extract BF16 storage bits for exact comparisons, including signed zero.
fn bits(values: &[BFloat16]) -> Vec<u16> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Run GPU primitive, decoder-layer and generation fixtures on device zero.
///
/// Checks exact BF16 results for small primitive cases and verifies safe batch
/// construction from a host thread without a current context. CUDA/setup
/// errors are returned; primitive mismatches and worker failures panic.
pub fn run() -> Result<()> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let runner = Runner::new(stream.clone())?;
    crate::operator::validate_prepared_tables()?;
    // A fresh host thread has no current CUDA context. Safe construction must
    // bind the supplied stream's context before allocating its input storage.
    let thread_stream = stream.clone();
    let from_worker = std::thread::spawn(move || -> std::result::Result<Vec<BFloat16>, String> {
        let batch = Batch::<3>::from_host(&thread_stream, 1, &[1., -2., 3.].map(reference::round))
            .map_err(|e| e.to_string())?;
        batch.read(&thread_stream).map_err(|e| e.to_string())
    })
    .join()
    .expect("input worker panicked")
    .expect("safe Batch::from_host must bind its context");
    assert_eq!(
        bits(&from_worker),
        bits(&[1., -2., 3.].map(reference::round))
    );
    let mut failures = Vec::new();
    let mut check = |label, actual: Vec<BFloat16>, expected: Vec<BFloat16>| {
        if bits(&actual) != bits(&expected) {
            failures.push(label);
        }
    };
    let table: Vec<_> = (0..21).map(|i| reference::round(i as f64)).collect();
    let table_dev = DeviceBuffer::from_host(&stream, &table)?;
    let weights = EmbeddingWeights::<3, 7>::new(&table_dev, 7)?;
    let ids = TokenIds::new(&[2, 0, 2, 1])?;
    check(
        "embedding",
        runner.embed(&weights, ids)?.read(&stream)?,
        reference::embedding(&table, ids.as_slice(), 7),
    );
    let left = Batch::<3>::from_host(&stream, 1, &[1., 2., -3.].map(reference::round))?;
    let right = Batch::<3>::from_host(&stream, 1, &[4., -5., 6.].map(reference::round))?;
    check(
        "add",
        runner.add(&left, &right)?.read(&stream)?,
        [5., -3., 3.].map(reference::round).to_vec(),
    );
    check(
        "silu",
        runner.silu(&left)?.read(&stream)?,
        reference::silu(&[1., 2., -3.].map(reference::round)),
    );
    let joined = runner.concat(&left, &right)?;
    check(
        "concat",
        joined.read(&stream)?,
        [1., 2., -3., 4., -5., 6.].map(reference::round).to_vec(),
    );
    let rows = Batch::<1>::from_host(
        &stream,
        3,
        &[1., 1. / 256., 1. / 256.].map(reference::round),
    )?;
    check(
        "sum_rows",
        runner.sum_rows(&rows)?.read(&stream)?,
        vec![BFloat16::from_bits(0x3f81)],
    );
    let x = Batch::<1>::from_host(&stream, 1, &[BFloat16::from_bits(0x3f81)])?;
    let w = DeviceBuffer::from_host(&stream, &[BFloat16::from_bits(0x3f81)])?;
    let b = DeviceBuffer::from_host(&stream, &[BFloat16::from_bits(0x3b80)])?;
    let weight = ProjectionWeights::<InputProjection, 1, 1>::new(&w, 1)?;
    let bias = BiasWeights::<InputProjection, 1>::new(&b)?;
    check(
        "biased_linear",
        runner.project(&x, &weight, Some(&bias))?.read(&stream)?,
        vec![BFloat16::from_bits(0x3f83)],
    );
    assert!(
        failures.is_empty(),
        "primitive numerical failures: {failures:?}"
    );
    println!("Qwen input/adaptor primitive fixtures passed");
    crate::layer_validation::run(&runner)?;
    crate::generation_validation::run(&runner)?;
    Ok(())
}
