/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Synthetic decoder checks with independent stage and complete-layer oracles.
use crate::{
    attention_reference,
    model::{Batch, DecoderCache, Runner, Trace},
    operator::{self, Result},
    reference::{self, LayerWeights, round, widen},
    rotary::{HostTables, RotarySpec},
    rotary_reference,
};
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    BFloat16,
    decoder::{AttentionWeights, DecoderWeights, FinalNorm, NormWeights},
    ffn::{FfnWeights, ProjectionWeights},
};

/// GPU allocations live outside their immutable, role-checked views.
pub(crate) struct DeviceLayer {
    /// Owned hidden-width input RMSNorm vector.
    input_norm: DeviceBuffer<BFloat16>,
    /// Owned dense `[query_width, hidden]` projection.
    query: DeviceBuffer<BFloat16>,
    /// Owned dense `[kv_width, hidden]` projection.
    key: DeviceBuffer<BFloat16>,
    /// Owned dense `[kv_width, hidden]` value projection.
    value: DeviceBuffer<BFloat16>,
    /// Owned dense `[hidden, query_width]` attention output projection.
    output: DeviceBuffer<BFloat16>,
    /// Owned 128-value per-head query RMSNorm vector.
    q_norm: DeviceBuffer<BFloat16>,
    /// Owned 128-value per-head key RMSNorm vector.
    k_norm: DeviceBuffer<BFloat16>,
    /// Owned hidden-width post-attention RMSNorm vector.
    post_norm: DeviceBuffer<BFloat16>,
    /// Owned dense `[intermediate, hidden]` FFN gate projection.
    gate: DeviceBuffer<BFloat16>,
    /// Owned dense `[intermediate, hidden]` FFN expansion projection.
    up: DeviceBuffer<BFloat16>,
    /// Owned dense `[hidden, intermediate]` FFN contraction projection.
    down: DeviceBuffer<BFloat16>,
}

impl DeviceLayer {
    /// Upload every host fixture weight, propagating allocation and transfer errors.
    pub(crate) fn new(stream: &CudaStream, host: &LayerWeights) -> Result<Self> {
        Ok(Self {
            input_norm: DeviceBuffer::from_host(stream, &host.input_norm)?,
            query: DeviceBuffer::from_host(stream, &host.query)?,
            key: DeviceBuffer::from_host(stream, &host.key)?,
            value: DeviceBuffer::from_host(stream, &host.value)?,
            output: DeviceBuffer::from_host(stream, &host.output)?,
            q_norm: DeviceBuffer::from_host(stream, &host.q_norm)?,
            k_norm: DeviceBuffer::from_host(stream, &host.k_norm)?,
            post_norm: DeviceBuffer::from_host(stream, &host.post_norm)?,
            gate: DeviceBuffer::from_host(stream, &host.gate)?,
            up: DeviceBuffer::from_host(stream, &host.up)?,
            down: DeviceBuffer::from_host(stream, &host.down)?,
        })
    }

    /// Borrow role-checked D128 decoder views for the requested H/I/QW/KW widths.
    ///
    /// Returns an error if an allocation does not satisfy its typed extent.
    pub(crate) fn weights<const H: usize, const I: usize, const QW: usize, const KW: usize>(
        &self,
    ) -> Result<DecoderWeights<'_, H, I, QW, KW, 128>> {
        Ok(DecoderWeights::new(
            AttentionWeights::new(
                ProjectionWeights::new(&self.query, H)?,
                ProjectionWeights::new(&self.key, H)?,
                ProjectionWeights::new(&self.value, H)?,
                ProjectionWeights::new(&self.output, QW)?,
                NormWeights::new(&self.q_norm)?,
                NormWeights::new(&self.k_norm)?,
            ),
            NormWeights::new(&self.input_norm)?,
            NormWeights::new(&self.post_norm)?,
            FfnWeights::new(
                ProjectionWeights::new(&self.gate, H)?,
                ProjectionWeights::new(&self.up, H)?,
                ProjectionWeights::new(&self.down, I)?,
            ),
        ))
    }
}

/// Generate a dense `[output, input]` BF16 matrix sensitive to both coordinates and seed.
fn matrix(input: usize, output: usize, seed: u64) -> Vec<BFloat16> {
    (0..output)
        .flat_map(|row| {
            (0..input).map(move |column| {
                // Mix both coordinates so swapped orientation and roles change the result.
                let mut x = (row as u64).wrapping_mul(0x9e37_79b9)
                    ^ (column as u64).wrapping_mul(0x85eb_ca6b)
                    ^ seed.wrapping_mul(0xc2b2_ae35);
                x ^= x >> 17;
                x = x.wrapping_mul(0x27d4_eb2d);
                x ^= x >> 15;
                let signed = (x % 17) as i32 - 8;
                round(f64::from(if signed == 0 { 1 } else { signed }) / 1024.0)
            })
        })
        .collect()
}

/// Generate nonunit per-coordinate BF16 normalization weights to expose rounding changes.
fn norm_fixture(width: usize, seed: usize) -> Vec<BFloat16> {
    (0..width)
        .map(|i| round(0.875 + ((i * 5 + seed) % 17) as f64 / 64.0))
        .collect()
}

/// Build distinct D128 layer weights for hidden/intermediate widths and query/KV head counts.
fn fixture_with_seed(h: usize, i: usize, q: usize, k: usize, seed: u64) -> LayerWeights {
    LayerWeights {
        input_norm: norm_fixture(h, seed as usize),
        query: matrix(h, q * 128, seed + 1),
        key: matrix(h, k * 128, seed + 2),
        value: matrix(h, k * 128, seed + 3),
        output: matrix(q * 128, h, seed + 4),
        q_norm: norm_fixture(128, seed as usize + 5),
        k_norm: norm_fixture(128, seed as usize + 6),
        post_norm: norm_fixture(h, seed as usize + 7),
        gate: matrix(h, i, seed + 8),
        up: matrix(h, i, seed + 9),
        down: matrix(i, h, seed + 10),
    }
}

/// Build the standard deterministic nonzero layer fixture using seed three.
pub(crate) fn fixture(h: usize, i: usize, q: usize, k: usize) -> LayerWeights {
    fixture_with_seed(h, i, q, k, 3)
}

/// Generate dense `[rows, width]` BF16 inputs varying with row and channel.
fn input_fixture(rows: usize, width: usize) -> Vec<BFloat16> {
    (0..rows)
        .flat_map(|row| {
            (0..width).map(move |column| {
                let signed = ((row * 131 + column * 17 + column / 13 * 11 + 7) % 97) as i32 - 48;
                round(f64::from(signed) / 64.0)
            })
        })
        .collect()
}

/// Copy `count` rotary rows starting at `start` into checked D128 host tables.
///
/// Each coefficient row has 64 values; an out-of-range slice panics.
fn table_slice(tables: &HostTables<128>, start: usize, count: usize) -> Result<HostTables<128>> {
    Ok(HostTables::from_coefficients(
        count,
        tables.cos()[start * 64..(start + count) * 64].to_vec(),
        tables.sin()[start * 64..(start + count) * 64].to_vec(),
    )?)
}

/// Counters and worst absolute error shared across synthetic layer assertions.
#[derive(Default)]
struct Checks {
    /// Number of values accepted by completed numerical, exact and guard comparisons.
    values: usize,
    /// Largest absolute difference observed by tolerant comparisons.
    maximum: f64,
}

impl Checks {
    /// Check finite equal-length values against `atol + rtol * abs(expected)`.
    ///
    /// Returns the first labeled mismatch and accumulates accepted comparison statistics.
    fn close(
        &mut self,
        label: &str,
        actual: &[BFloat16],
        expected: &[BFloat16],
        atol: f64,
        rtol: f64,
    ) -> Result<()> {
        if actual.len() != expected.len() {
            return Err(format!("{label}: output length mismatch").into());
        }
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let a = widen(actual);
            let e = widen(expected);
            let error = (a - e).abs();
            if !a.is_finite() || !e.is_finite() || error > atol + rtol * e.abs() {
                return Err(format!(
                    "{label}[{index}]: actual={a:.9e}, expected={e:.9e}, tolerance={:.9e}",
                    atol + rtol * e.abs()
                )
                .into());
            }
            self.maximum = self.maximum.max(error);
        }
        self.values += actual.len();
        Ok(())
    }

    /// Require matching lengths and BF16 bits, then count the compared values.
    fn exact(&mut self, label: &str, actual: &[BFloat16], expected: &[BFloat16]) -> Result<()> {
        if actual.len() != expected.len()
            || actual
                .iter()
                .zip(expected)
                .any(|(a, e)| a.to_bits() != e.to_bits())
        {
            return Err(format!("{label}: storage bits differ").into());
        }
        self.values += actual.len();
        Ok(())
    }

    /// Require the suffix starting at `start` to contain only the allocation guard bits.
    ///
    /// The caller supplies an in-bounds suffix; mismatches return a labeled error.
    fn guards(&mut self, label: &str, values: &[BFloat16], start: usize) -> Result<()> {
        if values[start..]
            .iter()
            .any(|x| x.to_bits() != operator::GUARD)
        {
            return Err(format!("{label}: unwritten cache or allocation guard changed").into());
        }
        self.values += values.len() - start;
        Ok(())
    }
}

/// Borrow a named recorded stage or return a missing-stage error.
fn stage<'a>(trace: &'a Trace, name: &str) -> Result<&'a [BFloat16]> {
    trace
        .stages
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, values)| values.as_slice())
        .ok_or_else(|| format!("missing stage {name}").into())
}

/// Validate one D128 layer against stage-local and complete FP64 graph oracles.
///
/// Uses actual stored inputs for stage tolerances and original inputs for the
/// full graph; checks exact rotated-K/unrotated-V cache writes and guards.
/// Returns the number of accepted values or the first mismatch/CUDA error.
#[allow(clippy::too_many_arguments)]
fn layer_case<
    const H: usize,
    const I: usize,
    const Q: usize,
    const K: usize,
    const QW: usize,
    const KW: usize,
>(
    runner: &Runner,
    rows: usize,
    spec: RotarySpec<128>,
    positions: &[[u32; 3]],
    label: &str,
) -> Result<usize> {
    println!("Checking {label}: H={H} I={I} Q={Q} KV={K} rows={rows}");
    let stream = runner.stream();
    let input = input_fixture(rows, H);
    let host = fixture(H, I, Q, K);
    let device = DeviceLayer::new(stream, &host)?;
    let weights = device.weights::<H, I, QW, KW>()?;
    let batch = Batch::<H>::from_host(stream, rows, &input)?;
    let tables = spec.tables(positions)?;
    let mut cache = operator::KvCache::<K, 128>::new(stream, rows + 2)?;
    let mut trace = Trace {
        enabled: true,
        stages: Vec::new(),
    };
    let output =
        runner.layer::<H, I, Q, K, QW, KW>(&batch, &weights, &mut cache, &tables, &mut trace)?;
    let mut checks = Checks::default();
    let mut compare = |name: &str, expected: Vec<BFloat16>| -> Result<()> {
        checks.close(name, stage(&trace, name)?, &expected, 2e-5, 0.008)
    };
    compare(
        "normalized",
        reference::norm(&input, &host.input_norm, rows, H, 1e-6),
    )?;
    compare(
        "q",
        reference::linear(stage(&trace, "normalized")?, &host.query, rows, H, QW, None),
    )?;
    compare(
        "k",
        reference::linear(stage(&trace, "normalized")?, &host.key, rows, H, KW, None),
    )?;
    compare(
        "v",
        reference::linear(stage(&trace, "normalized")?, &host.value, rows, H, KW, None),
    )?;
    compare(
        "q_normed",
        reference::norm(stage(&trace, "q")?, &host.q_norm, rows * Q, 128, 1e-6),
    )?;
    compare(
        "k_normed",
        reference::norm(stage(&trace, "k")?, &host.k_norm, rows * K, 128, 1e-6),
    )?;
    compare(
        "rotated_q",
        rotary_reference::rotate(
            stage(&trace, "q_normed")?,
            tables.cos(),
            tables.sin(),
            Q,
            128,
        ),
    )?;
    let (cache_k, cache_v) = cache.snapshot(stream)?;
    let rotated_k = rotary_reference::rotate(
        stage(&trace, "k_normed")?,
        tables.cos(),
        tables.sin(),
        K,
        128,
    );
    compare(
        "attention",
        attention_reference::attention(
            stage(&trace, "rotated_q")?,
            &cache_k[..rows * KW],
            &cache_v[..rows * KW],
            Q,
            K,
            128,
            rows,
            rows,
            Some(0),
        ),
    )?;
    compare(
        "projected",
        reference::linear(stage(&trace, "attention")?, &host.output, rows, QW, H, None),
    )?;
    compare(
        "residual",
        reference::add(&input, stage(&trace, "projected")?),
    )?;
    compare(
        "post_normed",
        reference::norm(stage(&trace, "residual")?, &host.post_norm, rows, H, 1e-6),
    )?;
    compare(
        "gate",
        reference::linear(stage(&trace, "post_normed")?, &host.gate, rows, H, I, None),
    )?;
    compare(
        "up",
        reference::linear(stage(&trace, "post_normed")?, &host.up, rows, H, I, None),
    )?;
    compare("silu", reference::silu(stage(&trace, "gate")?))?;
    let product = stage(&trace, "silu")?
        .iter()
        .zip(stage(&trace, "up")?)
        .map(|(&s, &u)| round(widen(s) * widen(u)))
        .collect();
    compare("product", product)?;
    compare(
        "down",
        reference::linear(stage(&trace, "product")?, &host.down, rows, I, H, None),
    )?;
    compare(
        "output",
        reference::add(stage(&trace, "residual")?, stage(&trace, "down")?),
    )?;
    checks.exact("rotated K cache", &cache_k[..rows * KW], &rotated_k)?;
    checks.exact(
        "unrotated V cache",
        &cache_v[..rows * KW],
        stage(&trace, "v")?,
    )?;
    checks.guards("K", &cache_k, rows * KW)?;
    checks.guards("V", &cache_v, rows * KW)?;
    let expected = reference::layer(&input, &host, H, I, Q, K, tables.cos(), tables.sin());
    // The earlier comparisons isolate each kernel using actual stored GPU
    // inputs. These follow the independent graph from its original input.
    for (name, values) in [
        ("normalized", &expected.normalized),
        ("q", &expected.q),
        ("k", &expected.k),
        ("v", &expected.v),
        ("q_normed", &expected.q_normed),
        ("k_normed", &expected.k_normed),
        ("rotated_q", &expected.rotated_q),
        ("attention", &expected.attention),
        ("projected", &expected.projected),
        ("residual", &expected.residual),
        ("post_normed", &expected.post_normed),
        ("gate", &expected.gate),
        ("up", &expected.up),
        ("silu", &expected.silu),
        ("product", &expected.product),
        ("down", &expected.down),
    ] {
        checks.close(
            &format!("complete graph {name}"),
            stage(&trace, name)?,
            values,
            0.002,
            0.025,
        )?;
    }
    checks.close(
        "complete graph rotated K",
        &cache_k[..rows * KW],
        &expected.rotated_k,
        0.002,
        0.025,
    )?;
    checks.close(
        "complete layer",
        &output.read(stream)?,
        &expected.output,
        0.002,
        0.025,
    )?;
    assert_eq!(cache.len(), rows);
    println!(
        "PASS {label} H={H} I={I} Q={Q} KV={K} rows={rows}: {} stage/reference values, max_abs={:.8e} (stage atol=2e-5 rtol=.008; complete atol=.002 rtol=.025)",
        checks.values, checks.maximum
    );
    Ok(checks.values)
}

/// Validate a two-layer tail-width stack over 65 tokens with distinct layer weights.
///
/// Compares whole/chunked/single-token execution, cache prefix preservation,
/// empty inputs, capacity rejection, reset and poisoning on foreign contexts.
fn stack_case(runner: &Runner) -> Result<usize> {
    /// Ragged hidden width for scalar/vector tail coverage.
    const H: usize = 17;
    /// Ragged FFN width crossing a 32-value boundary.
    const I: usize = 33;
    /// Query heads sharing one key/value head.
    const Q: usize = 2;
    /// Key/value heads in the small grouped-query fixture.
    const K: usize = 1;
    /// Dense query width: two D128 heads.
    const QW: usize = 256;
    /// Dense key/value width: one D128 head.
    const KW: usize = 128;
    /// Token count crossing the 64-key attention split boundary.
    const ROWS: usize = 65;
    let stream = runner.stream();
    let input = input_fixture(ROWS, H);
    let host = [fixture(H, I, Q, K), fixture_with_seed(H, I, Q, K, 17)];
    let device = [
        DeviceLayer::new(stream, &host[0])?,
        DeviceLayer::new(stream, &host[1])?,
    ];
    let weights = [
        device[0].weights::<H, I, QW, KW>()?,
        device[1].weights::<H, I, QW, KW>()?,
    ];
    let final_host = norm_fixture(H, 31);
    let final_device = DeviceBuffer::from_host(stream, &final_host)?;
    let final_norm = NormWeights::<FinalNorm, H>::new(&final_device)?;
    let positions: Vec<_> = (0..ROWS).map(|i| [(i * 7) as u32; 3]).collect();
    let tables = RotarySpec::<128>::standard(1e6, 65536)?.tables(&positions)?;
    let batch = Batch::<H>::from_host(stream, ROWS, &input)?;
    let mut whole_cache = DecoderCache::<H, I, K>::new(stream, 2, ROWS)?;
    let whole = runner
        .forward::<H, I, Q, K, QW, KW>(&batch, &weights, &final_norm, &mut whole_cache, &tables)?
        .read(stream)?;
    let whole_snapshots = whole_cache.snapshots(stream)?;
    let mut checks = Checks::default();
    let first = reference::layer(&input, &host[0], H, I, Q, K, tables.cos(), tables.sin());
    let second = reference::layer(
        &first.output,
        &host[1],
        H,
        I,
        Q,
        K,
        tables.cos(),
        tables.sin(),
    );
    let expected = reference::norm(&second.output, &final_host, ROWS, H, 1e-6);
    checks.close(
        "two-layer stack plus final norm",
        &whole,
        &expected,
        0.002,
        0.025,
    )?;
    let mut chunk_cache = DecoderCache::<H, I, K>::new(stream, 2, ROWS)?;
    let mut start = 0;
    let mut combined = Vec::new();
    for count in [0, 1, 3, 61] {
        let before = chunk_cache.snapshots(stream)?;
        let chunk = Batch::<H>::from_host(stream, count, &input[start * H..(start + count) * H])?;
        let coefficients = table_slice(&tables, start, count)?;
        combined.extend(
            runner
                .forward::<H, I, Q, K, QW, KW>(
                    &chunk,
                    &weights,
                    &final_norm,
                    &mut chunk_cache,
                    &coefficients,
                )?
                .read(stream)?,
        );
        let after = chunk_cache.snapshots(stream)?;
        for (index, ((before_k, before_v), (after_k, after_v))) in
            before.iter().zip(&after).enumerate()
        {
            checks.exact(
                &format!("layer {index} cached K prefix"),
                &after_k[..start * KW],
                &before_k[..start * KW],
            )?;
            checks.exact(
                &format!("layer {index} cached V prefix"),
                &after_v[..start * KW],
                &before_v[..start * KW],
            )?;
            checks.guards("chunk K tail", after_k, (start + count) * KW)?;
            checks.guards("chunk V tail", after_v, (start + count) * KW)?;
        }
        start += count;
        assert_eq!(chunk_cache.len(), start);
        assert!(!chunk_cache.is_poisoned());
    }
    checks.close("whole versus chunked", &combined, &whole, 2e-5, 0.008)?;
    for ((whole_k, whole_v), (chunk_k, chunk_v)) in
        whole_snapshots.iter().zip(chunk_cache.snapshots(stream)?)
    {
        checks.close(
            "whole/chunk K",
            &chunk_k[..ROWS * KW],
            &whole_k[..ROWS * KW],
            2e-5,
            0.008,
        )?;
        checks.close(
            "whole/chunk V",
            &chunk_v[..ROWS * KW],
            &whole_v[..ROWS * KW],
            2e-5,
            0.008,
        )?;
        checks.guards("whole K allocation", whole_k, ROWS * KW)?;
        checks.guards("whole V allocation", whole_v, ROWS * KW)?;
    }
    let before = chunk_cache.snapshots(stream)?;
    let one = Batch::<H>::from_host(stream, 1, &input[..H])?;
    let one_table = table_slice(&tables, 0, 1)?;
    assert!(
        runner
            .forward::<H, I, Q, K, QW, KW>(
                &one,
                &weights,
                &final_norm,
                &mut chunk_cache,
                &one_table
            )
            .is_err()
    );
    assert_eq!(chunk_cache.len(), ROWS);
    assert!(!chunk_cache.is_poisoned());
    for ((old_k, old_v), (new_k, new_v)) in before.iter().zip(chunk_cache.snapshots(stream)?) {
        checks.exact("capacity rejection K", &new_k, old_k)?;
        checks.exact("capacity rejection V", &new_v, old_v)?;
    }
    chunk_cache.reset();
    assert_eq!(chunk_cache.len(), 0);
    assert!(!chunk_cache.is_poisoned());
    let empty = Batch::<H>::from_host(stream, 0, &[])?;
    let empty_table = table_slice(&tables, 0, 0)?;
    // Exercise the real forward error path using a cache from another owning
    // context. No fault injection, malformed buffers or forced device failure.
    let other_context = cuda_core::CudaContext::new(0)?;
    let other_stream = other_context.default_stream();
    let mut foreign_cache = DecoderCache::<H, I, K>::new(&other_stream, 2, 1)?;
    assert!(
        runner
            .forward::<H, I, Q, K, QW, KW>(
                &one,
                &weights,
                &final_norm,
                &mut foreign_cache,
                &one_table
            )
            .is_err()
    );
    assert!(foreign_cache.is_poisoned());
    assert_eq!(foreign_cache.len(), 0);
    assert!(
        runner
            .forward::<H, I, Q, K, QW, KW>(
                &empty,
                &weights,
                &final_norm,
                &mut foreign_cache,
                &empty_table
            )
            .is_err()
    );
    foreign_cache.reset();
    assert!(!foreign_cache.is_poisoned());
    assert_eq!(foreign_cache.len(), 0);
    assert!(
        runner
            .forward::<H, I, Q, K, QW, KW>(
                &empty,
                &weights[..1],
                &final_norm,
                &mut chunk_cache,
                &empty_table
            )
            .is_err()
    );
    let empty_result = runner.forward::<H, I, Q, K, QW, KW>(
        &empty,
        &weights,
        &final_norm,
        &mut chunk_cache,
        &empty_table,
    )?;
    assert!(empty_result.read(stream)?.is_empty());
    assert_eq!(chunk_cache.len(), 0);
    for ((old_k, old_v), (new_k, new_v)) in before.iter().zip(chunk_cache.snapshots(stream)?) {
        checks.exact("reset and empty K bytes", &new_k, old_k)?;
        checks.exact("reset and empty V bytes", &new_v, old_v)?;
    }
    // Replay every row separately to exercise decode across the 64/65-key boundary.
    let mut decoded = Vec::new();
    for row in 0..ROWS {
        let one = Batch::<H>::from_host(stream, 1, &input[row * H..(row + 1) * H])?;
        let coefficients = table_slice(&tables, row, 1)?;
        decoded.extend(
            runner
                .forward::<H, I, Q, K, QW, KW>(
                    &one,
                    &weights,
                    &final_norm,
                    &mut chunk_cache,
                    &coefficients,
                )?
                .read(stream)?,
        );
    }
    checks.close(
        "whole versus single-token decode after reset",
        &decoded,
        &whole,
        2e-5,
        0.008,
    )?;
    assert_eq!(chunk_cache.len(), ROWS);
    assert!(!chunk_cache.is_poisoned());
    println!(
        "PASS two-layer H17/I33/Q2/K1 stack: 65 rows, whole/chunked/single-token, reset/empty/capacity, {} reference/cache values, max_abs={:.8e}",
        checks.values, checks.maximum
    );
    Ok(checks.values)
}

/// Run full-width Talker/predictor layer oracles and ragged two-layer cache fixtures.
///
/// Returns the total accepted comparison count; CUDA and comparison errors
/// propagate, while cache state invariant failures panic.
pub fn run(runner: &Runner) -> Result<usize> {
    let talker_positions = [[0, 0, 0], [7, 13, 19], [31, 37, 43]];
    let mut count = layer_case::<2048, 6144, 16, 8, 2048, 1024>(
        runner,
        3,
        RotarySpec::interleaved(1e6, 32768, [24, 20, 20])?,
        &talker_positions,
        "Talker layer",
    )?;
    let predictor_positions = [[0; 3], [11; 3], [65535; 3]];
    count += layer_case::<1024, 3072, 16, 8, 2048, 1024>(
        runner,
        3,
        RotarySpec::standard(1e6, 65536)?,
        &predictor_positions,
        "predictor layer",
    )?;
    let small_positions: Vec<_> = (0..65).map(|i| [i as u32; 3]).collect();
    count += layer_case::<17, 33, 2, 1, 256, 128>(
        runner,
        65,
        RotarySpec::standard(1e6, 65536)?,
        &small_positions,
        "tail layer",
    )?;
    count += stack_case(runner)?;
    Ok(count)
}
