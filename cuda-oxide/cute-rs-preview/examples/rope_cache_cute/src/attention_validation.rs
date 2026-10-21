/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! GPU acceptance for causal/noncausal GQA over an owned, incrementally filled cache.
use crate::operator::{AttentionInputs, KvCache, Result, Runner};
use crate::rotary::RotarySpec;
use crate::{attention_reference, reference};
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::Bf16;

/// Check exact constant-valued attention with zero queries across the 64-key split boundary.
fn uniform(runner: &Runner, stream: &CudaStream) -> Result<usize> {
    let tokens = 65;
    let q = DeviceBuffer::from_host(stream, &vec![Bf16::from_bits(0); tokens * 2 * 128])?;
    let k = DeviceBuffer::from_host(stream, &vec![Bf16::from_bits(0); tokens * 128])?;
    let v = DeviceBuffer::from_host(stream, &vec![Bf16::from_f32(3.0); tokens * 128])?;
    let mut cache = KvCache::<1, 128>::new(stream, tokens + 3)?;
    let tables = RotarySpec::<128>::standard(1e6, 128)?.tables(&vec![[0; 3]; tokens])?;
    let query = runner.append(
        &mut cache,
        AttentionInputs::<2, 1, 128>::new(&q, &k, &v, tokens)?,
        &tables,
    )?;
    let output = runner.attend(&query, &cache, Some(0))?;
    assert_eq!(output.tokens(), tokens);
    let actual = crate::query_output(stream, output.buffer(), tokens * 2 * 128)?;
    crate::assert_bits(
        "uniform GQA",
        &actual,
        &vec![Bf16::from_f32(3.0); actual.len()],
    );
    println!("PASS uniform BF16 GQA across a 64-key split boundary");
    Ok(actual.len())
}

/// BF16 output comparison with one-ULP-scale relative tolerance near normal values.
fn compare(label: &str, actual: &[Bf16], expected: &[Bf16]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut maximum = 0.0_f64;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let a = reference::widen(a);
        let b = reference::widen(b);
        let error = (a - b).abs();
        assert!(
            a.is_finite() && b.is_finite() && error <= 2e-5 + 0.008 * b.abs(),
            "{label} [{i}]: actual={a}, expected={b}, error={error}"
        );
        maximum = maximum.max(error);
    }
    maximum
}

/// Compare full attention and incremental append/attention over identical positions.
fn sequence<const Q: usize, const K: usize>(
    runner: &Runner,
    stream: &CudaStream,
    spec: &RotarySpec<128>,
    positions: &[[u32; 3]],
    chunks: &[usize],
    scale: f64,
) -> Result<usize> {
    let tokens = positions.len();
    assert_eq!(chunks.iter().sum::<usize>(), tokens);
    let scaled = |seed, length| {
        crate::fixture(length, seed)
            .into_iter()
            .map(|x| reference::round(reference::widen(x) * scale))
            .collect::<Vec<_>>()
    };
    let q = scaled(11, tokens * Q * 128);
    let k = scaled(23, tokens * K * 128);
    let v = crate::fixture(tokens * K * 128, 37);
    let tables = spec.tables(positions)?;
    let expected_q = reference::rotate(&q, tables.cos(), tables.sin(), Q, 128);
    let expected_k = reference::rotate(&k, tables.cos(), tables.sin(), K, 128);
    let q_dev = DeviceBuffer::from_host(stream, &q)?;
    let k_dev = DeviceBuffer::from_host(stream, &k)?;
    let v_dev = DeviceBuffer::from_host(stream, &v)?;
    let mut cache = KvCache::<K, 128>::new(stream, tokens + 17)?;
    let query = runner.append(
        &mut cache,
        AttentionInputs::<Q, K, 128>::new(&q_dev, &k_dev, &v_dev, tokens)?,
        &tables,
    )?;
    let before = cache.snapshot(stream)?;
    let output = runner.attend(&query, &cache, Some(0))?;
    assert_eq!(output.tokens(), tokens);
    let actual = crate::query_output(stream, output.buffer(), q.len())?;
    let expected = attention_reference::attention(
        &expected_q,
        &expected_k,
        &v,
        Q,
        K,
        128,
        tokens,
        tokens,
        Some(0),
    );
    let mut maximum = compare("causal prefill", &actual, &expected);
    let output = runner.attend(&query, &cache, None)?;
    let noncausal = crate::query_output(stream, output.buffer(), q.len())?;
    let expected_noncausal = attention_reference::attention(
        &expected_q,
        &expected_k,
        &v,
        Q,
        K,
        128,
        tokens,
        tokens,
        None,
    );
    maximum = maximum.max(compare("noncausal", &noncausal, &expected_noncausal));

    let mut incremental = KvCache::<K, 128>::new(stream, tokens + 17)?;
    let mut start = 0;
    for &count in chunks {
        let query_range = start * Q * 128..(start + count) * Q * 128;
        let key_range = start * K * 128..(start + count) * K * 128;
        let q_chunk = DeviceBuffer::from_host(stream, &q[query_range.clone()])?;
        let k_chunk = DeviceBuffer::from_host(stream, &k[key_range.clone()])?;
        let v_chunk = DeviceBuffer::from_host(stream, &v[key_range])?;
        let tables = spec.tables(&positions[start..start + count])?;
        let query = runner.append(
            &mut incremental,
            AttentionInputs::<Q, K, 128>::new(&q_chunk, &k_chunk, &v_chunk, count)?,
            &tables,
        )?;
        let output = runner.attend(&query, &incremental, Some(start))?;
        let result = crate::query_output(stream, output.buffer(), count * Q * 128)?;
        maximum = maximum.max(compare(
            "incremental oracle",
            &result,
            &expected[query_range.clone()],
        ));
        compare(
            "prefill versus incremental",
            &result,
            &actual[query_range.clone()],
        );
        // The same short query against the complete cache tests future masking
        // and query/key length differences independently of the incremental cache.
        let output = runner.attend(&query, &cache, Some(start))?;
        let result = crate::query_output(stream, output.buffer(), count * Q * 128)?;
        maximum = maximum.max(compare(
            "cached offset",
            &result,
            &expected[query_range.clone()],
        ));
        let output = runner.attend(&query, &cache, None)?;
        let result = crate::query_output(stream, output.buffer(), count * Q * 128)?;
        maximum = maximum.max(compare(
            "short noncausal query",
            &result,
            &expected_noncausal[query_range],
        ));
        start += count;
    }
    let after = cache.snapshot(stream)?;
    crate::assert_bits("attention preserves keys", &after.0, &before.0);
    crate::assert_bits("attention preserves values", &after.1, &before.1);
    assert_eq!(cache.len(), tokens);
    assert!(runner.attend(&query, &cache, Some(tokens + 1)).is_err());
    if tokens > 0 {
        cache.reset();
        assert!(runner.attend(&query, &cache, None).is_err());
        assert_eq!(cache.len(), 0);
    }
    println!(
        "PASS BF16 attention Q={Q} KV={K} D=128 tokens={tokens} scale={scale}: max_abs_error={maximum:.8e}"
    );
    Ok(q.len() * 5)
}

/// Entire future splits and a partial future tile contain poisoned K/V values.
fn poisoned_future(runner: &Runner, stream: &CudaStream) -> Result<usize> {
    let zero = vec![Bf16::from_bits(0); 128];
    let key = DeviceBuffer::from_host(stream, &zero)?;
    let value = DeviceBuffer::from_host(stream, &vec![Bf16::from_f32(2.0); 128])?;
    let query_buffer = DeviceBuffer::from_host(stream, &vec![Bf16::from_bits(0); 256])?;
    let spec = RotarySpec::<128>::standard(1e6, 256)?;
    let mut cache = KvCache::<1, 128>::new(stream, 145)?;
    let query = runner.append(
        &mut cache,
        AttentionInputs::<2, 1, 128>::new(&query_buffer, &key, &value, 1)?,
        &spec.tables(&[[0; 3]])?,
    )?;
    let future = 128;
    let poison = DeviceBuffer::from_host(stream, &vec![Bf16::from_bits(0x7fc1); future * 128])?;
    let queries = DeviceBuffer::from_host(stream, &vec![Bf16::from_bits(0); future * 256])?;
    runner.append(
        &mut cache,
        AttentionInputs::<2, 1, 128>::new(&queries, &poison, &poison, future)?,
        &spec.tables(&vec![[0; 3]; future])?,
    )?;
    let output = runner.attend(&query, &cache, Some(0))?;
    let actual = crate::query_output(stream, output.buffer(), 256)?;
    crate::assert_bits(
        "masked future poison",
        &actual,
        &vec![Bf16::from_f32(2.0); 256],
    );
    println!("PASS masked future K/V poison and neutral split states");
    Ok(actual.len())
}

/// Run GPU attention fixtures for MHA/GQA/MQA, causal masking, tails, and incremental caches.
///
/// Prints comparison counts after the independent FP64 oracle, guard checks, and
/// full-prefill/incremental comparisons succeed; this is synthetic model-component evidence.
pub fn run(runner: &Runner, stream: &CudaStream) -> Result<()> {
    let mut checked = uniform(runner, stream)?;
    checked += poisoned_future(runner, stream)?;
    let standard = RotarySpec::<128>::standard(1e6, 65536)?;
    let multimodal = RotarySpec::<128>::interleaved(1e6, 32768, [24, 20, 20])?;
    let positions = |count| (0..count).map(|i| [i * 7; 3]).collect::<Vec<_>>();
    checked += sequence::<16, 8>(runner, stream, &standard, &positions(1), &[1], 1.0)?;
    checked += sequence::<4, 4>(runner, stream, &standard, &positions(63), &[1, 61, 1], 1.0)?;
    checked += sequence::<6, 2>(runner, stream, &standard, &positions(64), &[0, 63, 1], 1.0)?;
    let axes: Vec<_> = (0..65).map(|i| [i * 3, i * 5 + 1, i * 11 + 7]).collect();
    checked += sequence::<16, 8>(runner, stream, &multimodal, &axes, &[0, 63, 1, 1], 1.0)?;
    checked += sequence::<16, 8>(
        runner,
        stream,
        &standard,
        &positions(129),
        &[1, 62, 1, 1, 64],
        4.0,
    )?;
    checked += sequence::<4, 1>(runner, stream, &standard, &positions(17), &[1; 17], 1.0)?;
    checked += sequence::<16, 8>(runner, stream, &standard, &[], &[0], 1.0)?;
    println!(
        "PASS: {checked} BF16 attention output comparisons; prefill, chunks and single-token decode"
    );
    Ok(())
}
