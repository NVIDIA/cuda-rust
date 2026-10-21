/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 rotary positions, owned KV cache and grouped-query attention validation.
/// Checked split-attention launches over a completed cache prefix.
mod attention;
/// CuTe shared-memory staging, 64-key partial attention, and split merging.
mod attention_kernel;
/// Independent FP64 grouped-query attention oracle.
mod attention_reference;
/// GPU fixtures for causal windows, split tails, and incremental attention.
mod attention_validation;
/// Register-tile rotary pairs and bit-preserving cache append kernels.
mod kernel;
/// Owned cache, reusable rotary tables, and synchronous execution boundary.
mod operator;
/// Independent integer BF16 rounding and split-half rotation oracle.
mod reference;
/// Checked host generation and storage of standard or interleaved rotary tables.
mod rotary;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cute_rs::Bf16;
use operator::{AttentionInputs, GUARD, KvCache, Result, Runner};
use rotary::{HostTables, RotarySpec};

/// Deliberately asymmetric finite inputs distinguish heads, coordinates and roles.
fn fixture(len: usize, seed: usize) -> Vec<Bf16> {
    (0..len)
        .map(|i| reference::round(((i * 17 + (i / 7) * 3 + seed * 11) % 67) as f64 / 16.0 - 2.0))
        .collect()
}

/// Compare complete BF16 encodings, reporting the first unequal element and label.
fn assert_bits(label: &str, actual: &[Bf16], expected: &[Bf16]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{label} at {i}");
    }
}

/// Require every unwritten element to retain the BF16 sentinel bit pattern.
fn assert_guard(label: &str, values: &[Bf16]) {
    assert!(
        values.iter().all(|v| v.to_bits() == GUARD),
        "{label} overwritten"
    );
}

/// Copy a result to the host, validate trailing guards, and return logical elements.
fn query_output(
    stream: &CudaStream,
    output: &DeviceBuffer<Bf16>,
    logical: usize,
) -> Result<Vec<Bf16>> {
    let mut values = vec![Bf16::from_bits(0); output.len()];
    output.copy_to_host(stream, &mut values)?;
    assert_guard("query guard", &values[logical..]);
    values.truncate(logical);
    Ok(values)
}

/// Compare whole and chunked rotary/cache appends against exact BF16 reference bits.
///
/// `positions` supplies one axis triple per token and `chunks` must sum to that
/// count. Checks completed prefixes, unwritten slots, capacity rejection, and reset;
/// returns the number of logical Q/K/V reference values in the full sequence.
fn validate<const Q: usize, const K: usize, const D: usize>(
    runner: &Runner,
    stream: &CudaStream,
    spec: &RotarySpec<D>,
    positions: &[[u32; 3]],
    chunks: &[usize],
) -> Result<usize> {
    let tokens = positions.len();
    assert_eq!(chunks.iter().sum::<usize>(), tokens);
    let q = fixture(tokens * Q * D, 1);
    let k = fixture(tokens * K * D, 2);
    let v = fixture(tokens * K * D, 3);
    let tables = spec.tables(positions)?;
    let expected_q = reference::rotate(&q, tables.cos(), tables.sin(), Q, D);
    let expected_k = reference::rotate(&k, tables.cos(), tables.sin(), K, D);
    let q_dev = DeviceBuffer::from_host(stream, &q)?;
    let k_dev = DeviceBuffer::from_host(stream, &k)?;
    let v_dev = DeviceBuffer::from_host(stream, &v)?;
    let mut whole = KvCache::<K, D>::new(stream, tokens)?;
    assert_eq!(whole.capacity(), tokens);
    let output = runner.append(
        &mut whole,
        AttentionInputs::<Q, K, D>::new(&q_dev, &k_dev, &v_dev, tokens)?,
        &tables,
    )?;
    assert_eq!(output.tokens(), tokens);
    assert_bits(
        "one-shot query",
        &query_output(stream, output.buffer(), q.len())?,
        &expected_q,
    );
    let whole_snapshot = whole.snapshot(stream)?;
    assert_bits("one-shot key", &whole_snapshot.0[..k.len()], &expected_k);
    assert_bits("unrotated value", &whole_snapshot.1[..v.len()], &v);
    assert_guard("key allocation guard", &whole_snapshot.0[k.len()..]);
    assert_guard("value allocation guard", &whole_snapshot.1[v.len()..]);

    let mut cache = KvCache::<K, D>::new(stream, tokens)?;
    let mut start = 0;
    for &chunk in chunks {
        let end = start + chunk;
        let q_dev = DeviceBuffer::from_host(stream, &q[start * Q * D..end * Q * D])?;
        let k_dev = DeviceBuffer::from_host(stream, &k[start * K * D..end * K * D])?;
        let v_dev = DeviceBuffer::from_host(stream, &v[start * K * D..end * K * D])?;
        let tables = spec.tables(&positions[start..end])?;
        let output = runner.append(
            &mut cache,
            AttentionInputs::<Q, K, D>::new(&q_dev, &k_dev, &v_dev, chunk)?,
            &tables,
        )?;
        assert_eq!(cache.len(), end);
        assert_bits(
            "chunk query",
            &query_output(stream, output.buffer(), chunk * Q * D)?,
            &expected_q[start * Q * D..end * Q * D],
        );
        let (key, value) = cache.snapshot(stream)?;
        assert_bits(
            "visible keys",
            &key[..end * K * D],
            &expected_k[..end * K * D],
        );
        assert_bits("visible values", &value[..end * K * D], &v[..end * K * D]);
        assert_guard("unwritten key slots", &key[end * K * D..]);
        assert_guard("unwritten value slots", &value[end * K * D..]);
        start = end;
    }
    let snapshot = cache.snapshot(stream)?;
    assert_bits("chunked/whole keys", &snapshot.0, &whole_snapshot.0);
    assert_bits("chunked/whole values", &snapshot.1, &whole_snapshot.1);

    // A full cache must reject another token without changing its cursor/data.
    let q1 = DeviceBuffer::from_host(stream, &fixture(Q * D, 4))?;
    let k1 = DeviceBuffer::from_host(stream, &fixture(K * D, 5))?;
    let v1 = DeviceBuffer::from_host(stream, &fixture(K * D, 6))?;
    let table1 = spec.tables(&[[0; 3]])?;
    assert!(
        runner
            .append(
                &mut cache,
                AttentionInputs::<Q, K, D>::new(&q1, &k1, &v1, 1)?,
                &table1
            )
            .is_err()
    );
    assert_eq!(cache.len(), tokens);
    let after_error = cache.snapshot(stream)?;
    assert_bits("rejected append key", &after_error.0, &snapshot.0);
    assert_bits("rejected append value", &after_error.1, &snapshot.1);
    let empty = DeviceBuffer::from_host(stream, &[])?;
    assert!(AttentionInputs::<Q, K, D>::new(&q1, &k1, &empty, 1).is_err());
    assert!(
        runner
            .append(
                &mut cache,
                AttentionInputs::<Q, K, D>::new(&empty, &empty, &empty, 0)?,
                &table1
            )
            .is_err()
    );
    assert_eq!(cache.len(), tokens);

    cache.reset();
    assert_eq!(cache.len(), 0);
    if tokens > 0 {
        let output = runner.append(
            &mut cache,
            AttentionInputs::<Q, K, D>::new(&q1, &k1, &v1, 1)?,
            &table1,
        )?;
        assert_bits(
            "position-zero identity",
            &query_output(stream, output.buffer(), Q * D)?,
            &fixture(Q * D, 4),
        );
        assert_eq!(cache.len(), 1);
        let reset = cache.snapshot(stream)?;
        assert_bits("reset key", &reset.0[..K * D], &fixture(K * D, 5));
        assert_bits("reset value", &reset.1[..K * D], &fixture(K * D, 6));
    }
    println!(
        "PASS RoPE/cache Q={Q} KV={K} D={D} tokens={tokens}: one-shot, chunks, guards and rejection"
    );
    Ok(tokens * (Q + 2 * K) * D)
}

/// Require the exact rotary result that distinguishes product rounding from one final cast.
fn check_product_rounding(runner: &Runner, stream: &CudaStream) -> Result<()> {
    let q = DeviceBuffer::from_host(stream, &[0x3f81, 0x3f02].map(Bf16::from_bits))?;
    let table = HostTables::<2>::from_coefficients(
        1,
        vec![Bf16::from_bits(0x3f35)],
        vec![Bf16::from_bits(0x3f35)],
    )?;
    let mut cache = KvCache::<1, 2>::new(stream, 1)?;
    let result = runner.append(
        &mut cache,
        AttentionInputs::<1, 1, 2>::new(&q, &q, &q, 1)?,
        &table,
    )?;
    assert_bits(
        "separate product rounding",
        &query_output(stream, result.buffer(), 2)?,
        &[0x3eb4, 0x3f89].map(Bf16::from_bits),
    );
    println!("PASS exact product-rounding discriminator (cast-once would produce 0x3eb5)");
    Ok(())
}

/// Prior safe asynchronous writes must finish before the safe runner reads input.
fn check_pending_input(runner: &Runner, stream: &CudaStream) -> Result<()> {
    use std::sync::mpsc;
    use std::time::Duration;
    let producer = stream.context().new_stream()?;
    let mut input = DeviceBuffer::from_host(stream, &[Bf16::from_bits(0x3f80); 2])?;
    let mut cache = KvCache::<1, 2>::new(stream, 1)?;
    let table = RotarySpec::<2>::standard(1e6, 8)?.tables(&[[0; 3]])?;
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    producer.launch_host_function(move || {
        let _ = entered_tx.send(());
        // Driver callbacks must not call CUDA APIs. Bound failure-path waits.
        let _ = release_rx.recv_timeout(Duration::from_secs(10));
    })?;
    entered_rx.recv_timeout(Duration::from_secs(10))?;
    input.zero_async(&producer)?;
    assert!(!producer.query()?, "producer must still be pending");
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(250));
        let _ = release_tx.send(());
    });
    let result = runner.append(
        &mut cache,
        AttentionInputs::<1, 1, 2>::new(&input, &input, &input, 1)?,
        &table,
    );
    let completion = producer.synchronize();
    release.join().expect("producer release thread panicked");
    let output = result?;
    completion?;
    let zero = [Bf16::from_bits(0); 2];
    assert_bits(
        "pending query",
        &query_output(stream, output.buffer(), 2)?,
        &zero,
    );
    let snapshot = cache.snapshot(stream)?;
    assert_bits("pending key", &snapshot.0[..2], &zero);
    assert_bits("pending value", &snapshot.1[..2], &zero);
    assert_eq!(cache.len(), 1);
    println!("PASS pending input writes on a separate stream");
    Ok(())
}

/// Validate rotary modes, prepared tables, cache lifecycle, and attention on CUDA device 0.
fn main() -> Result<()> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let runner = Runner::new(stream.clone())?;
    check_product_rounding(&runner, &stream)?;
    check_pending_input(&runner, &stream)?;
    operator::validate_prepared_tables()?;
    let standard = RotarySpec::<128>::standard(1e6, 65536)?;
    let mut positions: Vec<_> = (0..71).map(|i| [i * 13; 3]).collect();
    positions[70] = [65535; 3];
    let mut checked = validate::<16, 8, 128>(
        &runner,
        &stream,
        &standard,
        &positions,
        &[0, 1, 3, 0, 61, 6],
    )?;
    let multimodal = RotarySpec::<128>::interleaved(1e6, 32768, [24, 20, 20])?;
    let mut positions: Vec<_> = (0..71).map(|i| [i * 11, i * 17 + 3, i * 19 + 7]).collect();
    positions[70] = [32767, 32766, 32765];
    checked +=
        validate::<16, 8, 128>(&runner, &stream, &multimodal, &positions, &[3, 0, 1, 61, 6])?;
    let small = RotarySpec::<6>::standard(1e6, 32)?;
    let positions: Vec<_> = (0..17).map(|i| [i; 3]).collect();
    checked += validate::<6, 2, 6>(&runner, &stream, &small, &positions, &[0, 1, 3, 13])?;
    checked += validate::<6, 2, 6>(&runner, &stream, &small, &[], &[0])?;
    println!(
        "PASS: {checked} BF16 query/key/value reference values; stage 4a synthetic validation"
    );
    attention_validation::run(&runner, &stream)?;
    Ok(())
}
