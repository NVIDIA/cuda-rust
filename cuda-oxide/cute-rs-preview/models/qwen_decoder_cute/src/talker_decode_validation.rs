/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! GPU checks of persistent Talker arithmetic, cache publication and ownership.

use super::TalkerDecodePlan;
use crate::checkpoint::QwenWeights;
use crate::decoder_execution::Completion;
use crate::generation::{self, TALKER_LAYERS, TalkerStep};
use crate::model::{Batch, DecoderCache, Runner};
use crate::operator::Result;
use cuda_core::{CudaContext, CudaStream};
use cute_rs::BFloat16;
use std::sync::Arc;

/// Complete K/V backing arrays per layer, including unused capacity and guards.
type Snapshots = Vec<(Vec<BFloat16>, Vec<BFloat16>)>;

/// Reject different extents or BF16 payloads, including differing NaN guard bits.
fn same_bits(actual: &[BFloat16], expected: &[BFloat16], label: &str) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: extent {} != {}", actual.len(), expected.len()).into());
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(format!(
                "{label}[{index}]: BF16 bits {:04x} != {:04x}",
                actual.to_bits(),
                expected.to_bits()
            )
            .into());
        }
    }
    Ok(())
}

/// Compare completed hidden/logit values instead of relying on selected tokens.
fn same_step(
    stream: &CudaStream,
    actual: &TalkerStep,
    expected: &TalkerStep,
    label: &str,
) -> Result<()> {
    same_bits(
        &actual.last_hidden.read(stream)?,
        &expected.last_hidden.read(stream)?,
        &format!("{label} hidden"),
    )?;
    same_bits(
        &actual.logits.read(stream)?,
        &expected.logits.read(stream)?,
        &format!("{label} logits"),
    )
}

/// Read completed private cache allocations without changing their logical prefix.
fn snapshots(plan: &mut TalkerDecodePlan) -> Result<Snapshots> {
    let stream = plan.stream.clone();
    plan.cache
        .layers_mut()
        .iter()
        .map(|layer| {
            Ok((
                layer.key.to_host_vec(&stream)?,
                layer.value.to_host_vec(&stream)?,
            ))
        })
        .collect()
}

/// Check every cache value, unused-capacity sentinel and allocation guard bit.
fn same_snapshots(actual: &Snapshots, expected: &Snapshots, label: &str) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: cache layer count differs").into());
    }
    for (layer, ((actual_k, actual_v), (expected_k, expected_v))) in
        actual.iter().zip(expected).enumerate()
    {
        same_bits(actual_k, expected_k, &format!("{label} layer {layer} K"))?;
        same_bits(actual_v, expected_v, &format!("{label} layer {layer} V"))?;
    }
    Ok(())
}

/// Compare persistent state against independently executed synchronous appends.
fn same_cache(
    plan: &mut TalkerDecodePlan,
    original: &DecoderCache<2048, 6144, 8>,
    label: &str,
) -> Result<()> {
    if plan.len() != original.len() || plan.cache.is_poisoned() || original.is_poisoned() {
        return Err(format!("{label}: cache prefix or poison state differs").into());
    }
    let actual = snapshots(plan)?;
    let expected = original.snapshots(&plan.stream)?;
    same_snapshots(&actual, &expected, label)
}

/// Upload finite, signed, nonconstant rows; neighboring rows use different phases.
fn input(stream: &CudaStream, rows: usize, seed: usize) -> Result<Batch<2048>> {
    let values = (0..rows * 2048)
        .map(|index| {
            let row = index / 2048;
            let column = index % 2048;
            BFloat16::from_f32(
                ((column * (17 + seed) + row * 29 + seed * 13) % 113) as f32 / 64.0 - 0.875,
            )
        })
        .collect::<Vec<_>>();
    Batch::from_host(stream, rows, &values)
}

/// Unequal MRoPE axes deliberately differ from the physical cache cursor.
///
/// Fixtures use fewer than 132 tokens, so all three axes remain below 32768.
fn positions(start: usize, rows: usize) -> Vec<[u32; 3]> {
    (start..start + rows)
        .map(|token| {
            let token = token as u32;
            [token * 3 + 1, token * 5 + 2, token * 7 + 3]
        })
        .collect()
}

/// Independently prefill two caches, then consume only one into the new plan.
///
/// Full backing-array equality verifies transfer preserves the completed prefix,
/// unused capacity and sentinel guards. Returns an independent synchronous oracle.
fn paired_prefix(
    stream: &Arc<CudaStream>,
    weights: &Arc<QwenWeights>,
    prefix: usize,
    capacity: usize,
) -> Result<(Runner, DecoderCache<2048, 6144, 8>, TalkerDecodePlan)> {
    let runner = Runner::new(stream.clone())?;
    let mut original = DecoderCache::new(stream, TALKER_LAYERS, capacity)?;
    let mut transferred = DecoderCache::new(stream, TALKER_LAYERS, capacity)?;
    let input = input(stream, prefix, 7)?;
    let positions = positions(0, prefix);
    let talker = weights.talker()?;
    let expected = generation::talker_step(&runner, &talker, &mut original, &input, &positions)?;
    let actual = generation::talker_step(&runner, &talker, &mut transferred, &input, &positions)?;
    same_step(stream, &actual, &expected, "independent prefill")?;
    let mut plan = TalkerDecodePlan::new(stream.clone(), weights.clone(), transferred)?;
    same_cache(&mut plan, &original, "consuming prefill transfer")?;
    Ok((runner, original, plan))
}

/// Invalid input must fail before poisoning or changing the published cursor.
fn rejects_without_poison(
    plan: &mut TalkerDecodePlan,
    input: &Batch<2048>,
    positions: &[[u32; 3]],
    label: &str,
) -> Result<()> {
    let prefix = plan.len();
    if plan.run(input, positions).is_ok() || plan.len() != prefix || plan.cache.is_poisoned() {
        return Err(
            format!("{label}: invalid call succeeded, advanced or poisoned the plan").into(),
        );
    }
    Ok(())
}

/// Reject malformed rows, positions and contexts without touching any cache byte.
fn input_rejections(plan: &mut TalkerDecodePlan, stream: &Arc<CudaStream>) -> Result<()> {
    let before = snapshots(plan)?;
    let empty = Batch::<2048>::from_host(stream, 0, &[])?;
    let two = input(stream, 2, 3)?;
    let one = input(stream, 1, 5)?;
    let next = positions(plan.len(), 1);
    let extra = positions(plan.len(), 2);
    rejects_without_poison(plan, &empty, &[], "empty rows")?;
    rejects_without_poison(plan, &two, &extra, "two rows")?;
    rejects_without_poison(plan, &one, &[], "missing position")?;
    rejects_without_poison(plan, &one, &extra, "extra position")?;
    rejects_without_poison(plan, &one, &[[32768; 3]], "out-of-range MRoPE position")?;
    let foreign_context = CudaContext::new(0)?;
    let foreign = input(&foreign_context.default_stream(), 1, 5)?;
    rejects_without_poison(plan, &foreign, &next, "foreign context")?;
    stream.context().bind_to_thread()?;
    same_snapshots(&snapshots(plan)?, &before, "rejected inputs")
}

/// A blocked nondefault producer must complete before persistent kernels read input.
///
/// Outputs and every device allocation are prepared before the producer is held;
/// this call stays within one attention split count. An incidental allocation
/// fence therefore cannot hide a missing readiness fence in `run_into`.
fn foreign_producer(
    stream: &Arc<CudaStream>,
    weights: &Arc<QwenWeights>,
    runner: &Runner,
    original: &mut DecoderCache<2048, 6144, 8>,
    plan: &mut TalkerDecodePlan,
) -> Result<()> {
    let source = input(stream, 1, 31)?;
    let positions = positions(plan.len(), 1);
    let expected =
        generation::talker_step(runner, &weights.talker()?, original, &source, &positions)?;
    let mut delayed = Batch::from_host(stream, 1, &[BFloat16::from_bits(0x7fc1); 2048])?;
    let output = TalkerStep {
        last_hidden: Batch::allocate(stream, 1)?,
        logits: Batch::allocate(stream, 1)?,
    };
    let foreign = stream.context().new_stream()?;
    let (release, wait_release) = std::sync::mpsc::channel();
    let (started, wait_started) = std::sync::mpsc::channel();
    let foreign_completion = Completion::new(&foreign);
    foreign.launch_host_function(move || {
        let _ = started.send(());
        let _ = wait_release.recv_timeout(std::time::Duration::from_secs(5));
    })?;
    delayed
        .buffer_mut()
        .copy_from_device_async(source.buffer(), &foreign)?;
    wait_started.recv_timeout(std::time::Duration::from_secs(5))?;
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = release.send(());
    });
    let actual = plan.run_into(&delayed, &positions, output);
    let completed = foreign_completion.finish();
    releaser
        .join()
        .map_err(|_| "Talker foreign producer release failed")?;
    completed?;
    let actual = actual?;
    drop(source);
    drop(delayed);
    same_step(
        stream,
        &actual,
        &expected,
        "delayed foreign-stream producer",
    )?;
    same_cache(plan, original, "delayed foreign-stream cache")
}

/// Cross one split boundary, retain returned owners, and reject a full cache.
///
/// Prefixes 62 and 126 exercise visible lengths 63/64/65 and 127/128/129;
/// the fourth append tests readiness without resizing attention scratch.
fn boundary(stream: &Arc<CudaStream>, weights: &Arc<QwenWeights>, prefix: usize) -> Result<()> {
    let (runner, mut original, mut plan) = paired_prefix(stream, weights, prefix, prefix + 4)?;
    let mut retained = None;
    for step in 0..3 {
        let input = input(stream, 1, 11 + step)?;
        let positions = positions(prefix + step, 1);
        let expected = generation::talker_step(
            &runner,
            &weights.talker()?,
            &mut original,
            &input,
            &positions,
        )?;
        let actual = plan.run(&input, &positions)?;
        drop(input);
        same_step(stream, &actual, &expected, "split-boundary output")?;
        same_cache(&mut plan, &original, "split-boundary cache")?;
        if let Some((actual, expected)) = &retained {
            same_step(stream, actual, expected, "output retained across append")?;
        } else {
            retained = Some((actual, expected));
        }
    }
    input_rejections(&mut plan, stream)?;
    foreign_producer(stream, weights, &runner, &mut original, &mut plan)?;
    let before = snapshots(&mut plan)?;
    let input = input(stream, 1, 19)?;
    let positions = positions(plan.len(), 1);
    rejects_without_poison(&mut plan, &input, &positions, "full cache")?;
    same_snapshots(&snapshots(&mut plan)?, &before, "capacity rejection")?;
    let (actual, expected) = retained.ok_or("no retained Talker output")?;
    drop(plan);
    drop(runner);
    same_step(
        stream,
        &actual,
        &expected,
        "output retained after plan and runner drop",
    )?;
    println!(
        "PASS Talker persistent prefix {prefix}: three boundary appends, exact hidden/logits/full K/V/guards, retained outputs, readiness, input/capacity rejection"
    );
    Ok(())
}

/// An active partial guard failure must drain work and prevent prefix publication.
///
/// The guard is checked after each layer, before scratch is reused. The failed
/// append may write unpublished rows, but old prefix bits must remain unchanged;
/// retry must reject without executing another append or clearing sticky poison.
fn guard_failure(stream: &Arc<CudaStream>, weights: &Arc<QwenWeights>) -> Result<()> {
    let (_runner, _original, mut plan) = paired_prefix(stream, weights, 1, 3)?;
    let before = snapshots(&mut plan)?;
    let mut partials = plan.scratch.partials.to_host_vec(&plan.stream)?;
    partials[16 * 130] = 0.0;
    plan.scratch
        .partials
        .copy_from_host(&plan.stream, &partials)?;
    let input = input(stream, 1, 23)?;
    let positions = positions(1, 1);
    if plan.run(&input, &positions).is_ok()
        || plan.len() != 1
        || !plan.cache.is_poisoned()
        || plan.status.to_host_vec(&plan.stream)?[0] & 2 == 0
        || !plan.stream.query()?
    {
        return Err("Talker guard failure was not completed, rejected and left unpublished".into());
    }
    let failed = snapshots(&mut plan)?;
    for (layer, ((before_k, before_v), (after_k, after_v))) in
        before.iter().zip(&failed).enumerate()
    {
        same_bits(
            &after_k[..1024],
            &before_k[..1024],
            &format!("failed layer {layer} K prefix"),
        )?;
        same_bits(
            &after_v[..1024],
            &before_v[..1024],
            &format!("failed layer {layer} V prefix"),
        )?;
        same_bits(
            &after_k[2 * 1024..],
            &before_k[2 * 1024..],
            &format!("failed layer {layer} K suffix"),
        )?;
        same_bits(
            &after_v[2 * 1024..],
            &before_v[2 * 1024..],
            &format!("failed layer {layer} V suffix"),
        )?;
    }
    if plan.run(&input, &positions).is_ok() || plan.len() != 1 || !plan.cache.is_poisoned() {
        return Err("Talker guard failure did not reject retry with sticky poison".into());
    }
    same_snapshots(&snapshots(&mut plan)?, &failed, "poisoned retry")?;
    println!(
        "PASS Talker active guard: completion, unchanged published prefix, sticky poison and retry rejection"
    );
    Ok(())
}

/// The shared completion owner must drain submitted work on errors and unwinding.
///
/// A delayed host callback provides observable pending work without injecting
/// production-only hooks or relying on an invalid GPU memory access.
fn completion_cleanup(stream: &Arc<CudaStream>) -> Result<()> {
    for unwind in [false, true] {
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_observed = observed.clone();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let _completion = Completion::new(stream);
            stream.launch_host_function(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                callback_observed.store(true, std::sync::atomic::Ordering::Release);
            })?;
            if unwind {
                panic!("intentional Talker completion cleanup probe");
            }
            Err("intentional Talker submission error".into())
        }));
        if unwind != outcome.is_err()
            || !observed.load(std::sync::atomic::Ordering::Acquire)
            || !stream.query()?
        {
            return Err("Talker completion did not drain work on error or unwinding".into());
        }
    }
    Ok(())
}

/// Compare persistent one-row Talker decode with independently prefilled synchronous caches.
///
/// Uses real pinned weights, unequal MRoPE axes, both split transitions, complete
/// cache snapshots, caller-owned outputs and delayed foreign-stream producers.
/// Expected host panics from the completion probe are caught. CUDA, arithmetic,
/// guard or ownership failures propagate; this is not a timing benchmark.
pub(crate) fn validate(stream: Arc<CudaStream>, weights: Arc<QwenWeights>) -> Result<()> {
    stream.context().bind_to_thread()?;
    completion_cleanup(&stream.context().new_stream()?)?;
    boundary(&stream, &weights, 62)?;
    boundary(&stream, &weights, 126)?;
    guard_failure(&stream, &weights)?;
    println!(
        "PASS Talker persistent eager exact hidden/logits/cache parity, split boundaries, MRoPE axes, retained owners, foreign-stream readiness and failure completion"
    );
    Ok(())
}
