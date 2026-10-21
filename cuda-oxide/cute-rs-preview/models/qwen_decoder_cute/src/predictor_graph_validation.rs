/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! GPU contract checks for the exclusive predictor execution region.
use super::*;
use crate::{generation, model::Runner};

/// Require exact predicted codes and ordered BF16 embedding-sum bits.
///
/// Downloads both results and returns a labeled mismatch or readback error.
fn same_frame(
    stream: &CudaStream,
    actual: &CodecFrame,
    expected: &CodecFrame,
    label: &str,
) -> Result<()> {
    if actual.remaining_codes != expected.remaining_codes {
        return Err(format!(
            "{label}: predictor codes differ: {:?} != {:?}",
            actual.remaining_codes, expected.remaining_codes
        )
        .into());
    }
    let actual = actual.embedding_sum.read(stream)?;
    let expected = expected.embedding_sum.read(stream)?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(&expected)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(format!("{label}: ordered BF16 embedding sum differs").into());
    }
    Ok(())
}

/// Upload two deterministic, distinct `[1, 2048]` predictor inputs from a seed.
fn inputs(stream: &CudaStream, seed: usize) -> Result<(Batch<2048>, Batch<2048>)> {
    let values = |offset: usize| {
        (0..2048)
            .map(|index| {
                Bf16::from_f32(((index * (17 + seed) + offset) % 113) as f32 / 64.0 - 0.875)
            })
            .collect::<Vec<_>>()
    };
    Ok((
        Batch::from_host(stream, 1, &values(13 + seed))?,
        Batch::from_host(stream, 1, &values(71 + seed))?,
    ))
}

/// Compare device greedy selection with the host policy for ties, zeros and extremes.
///
/// Also rejects NaN/infinity with bounded fallback ID zero and verifies that
/// the error status remains set after a later finite selection.
fn selector(stream: &Arc<CudaStream>) -> Result<()> {
    let module = predictor_graph_kernel::kernels::load(stream.context())?;
    let mut codes = DeviceBuffer::from_host(stream, &[u32::MAX; REMAINING_CODES])?;
    let mut status = DeviceBuffer::from_host(stream, &[0u32])?;
    let mut fixtures = Vec::new();
    let mut tied = vec![Bf16::from_f32(-65536.0); 2048];
    tied[33] = Bf16::from_f32(65536.0);
    tied[1] = tied[33];
    fixtures.push(tied);
    fixtures.push(vec![Bf16::from_bits(0x8000); 2048]);
    let mut extreme = vec![Bf16::from_bits(0xff7f); 2048];
    extreme[2047] = Bf16::from_bits(0x7f7f);
    fixtures.push(extreme);
    for bits in [0x7fc1, 0x7f80, 0xff80] {
        let mut invalid = vec![Bf16::from_f32(1.0); 2048];
        invalid[2047] = Bf16::from_bits(bits);
        fixtures.push(invalid);
    }
    for (index, fixture) in fixtures.iter().enumerate() {
        let logits = DeviceBuffer::from_host(stream, fixture)?;
        let expected = generation::greedy_token::<2048>(fixture);
        let completion = Completion::new(stream);
        let launched = (|| -> Result<()> {
            // SAFETY: exact vocabulary, fifteen codes, one status slot, one full
            // warp; all owners retained by the unconditional completion scope.
            unsafe {
                module.reset_status(stream, one_warp(), &mut status)?;
                module.greedy(stream, one_warp(), &logits, &mut codes, &mut status, index)?;
            }
            Ok(())
        })();
        let completed = completion.finish();
        launched?;
        completed?;
        let actual = codes.to_host_vec(stream)?[index];
        let flag = status.to_host_vec(stream)?[0];
        match expected {
            Ok(code) if actual == code && flag == 0 => {}
            Err(_) if actual == 0 && flag == 1 => {}
            _ => {
                return Err(format!(
                    "predictor device greedy fixture {index} differs from host policy"
                )
                .into());
            }
        }
    }
    // An invalid earlier step remains an error after a later finite selection.
    let logits = DeviceBuffer::from_host(stream, &fixtures[0])?;
    unsafe {
        module.greedy(stream, one_warp(), &logits, &mut codes, &mut status, 14)?;
    }
    stream.synchronize()?;
    if status.to_host_vec(stream)? != [1] {
        return Err("predictor nonfinite status was not sticky".into());
    }
    Ok(())
}

/// A pitched table may end immediately after its final logical row, without
/// storage for a trailing pitch. Exercise that row and both output guards.
fn pitched_embedding(stream: &Arc<CudaStream>) -> Result<()> {
    let module = predictor_graph_kernel::kernels::load(stream.context())?;
    let stride = 2050;
    let mut table = vec![Bf16::from_bits(0); 2047 * stride + 2048];
    for column in 0..2048 {
        table[2047 * stride + column] = Bf16::from_bits(0x3f00 + (column % 127) as u16);
    }
    let expected = &table[2047 * stride..];
    let table = DeviceBuffer::from_host(stream, &table)?;
    let codes = DeviceBuffer::from_host(stream, &[2047u32; REMAINING_CODES])?;
    let guard = Bf16::from_bits(0x7fc1);
    let mut feedback = DeviceBuffer::from_host(stream, &vec![guard; 2048 + 8])?;
    let mut rows = DeviceBuffer::from_host(stream, &vec![guard; 4096 + 8])?;
    let completion = Completion::new(stream);
    // SAFETY: the selected final row has 2048 initialized elements despite
    // lacking trailing padding; output ranges are separate and fully allocated.
    let launched = unsafe {
        module.embedding(
            stream,
            grid(2048),
            &table,
            &codes,
            &mut feedback,
            &mut rows,
            0,
            stride,
        )
    };
    let completed = completion.finish();
    launched?;
    completed?;
    let feedback = feedback.to_host_vec(stream)?;
    let rows = rows.to_host_vec(stream)?;
    let exact = |actual: &[Bf16], expected: &[Bf16]| {
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    };
    if !exact(&feedback[..2048], expected)
        || !exact(&rows[2048..4096], expected)
        || feedback[2048..]
            .iter()
            .chain(&rows[..2048])
            .chain(&rows[4096..])
            .any(|value| value.to_bits() != guard.to_bits())
    {
        return Err("predictor pitched final embedding row or guards differ".into());
    }
    Ok(())
}

/// Probe capture cleanup on ordinary errors, host unwinding and invalidated capture.
///
/// Temporarily clears the current context to require cleanup to rebind its
/// owner, then verifies that the stream is usable outside capture.
fn capture_cleanup(stream: &Arc<CudaStream>) -> Result<()> {
    for unwind in [false, true] {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let _capture = Capture::begin(stream)?;
            // Simulate unrelated host work (or a panic hook) changing current
            // context. Cleanup must rebind its retained owner before EndCapture.
            unsafe {
                driver(sys::cuCtxSetCurrent(std::ptr::null_mut()))?;
            }
            if unwind {
                panic!("intentional predictor capture cleanup probe");
            }
            Err("intentional predictor capture error".into())
        }));
        if unwind != outcome.is_err()
            || (!unwind
                && !matches!(&outcome, Ok(Err(error)) if error.to_string() == "intentional predictor capture error"))
        {
            return Err("capture cleanup probe did not reach expected outcome".into());
        }
        let mut state = 0;
        unsafe {
            driver(sys::cuStreamIsCapturing(stream.cu_stream(), &mut state))?;
        }
        if state != sys::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_NONE {
            return Err("capture guard left stream capturing".into());
        }
        stream.synchronize()?;
    }
    // Invalidated captures also require EndCapture before ordinary stream use.
    let capture = Capture::begin(stream)?;
    if stream.synchronize().is_ok() {
        return Err("capture invalidation probe unexpectedly synchronized".into());
    }
    drop(capture);
    stream.synchronize()?;
    Ok(())
}

/// Require completion guards to drain delayed stream work on return and unwinding.
///
/// A host callback marks completion; its flag and stream readiness must both
/// be visible after the guard is dropped.
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
                panic!("intentional predictor completion cleanup probe");
            }
            Err("intentional predictor enqueue error".into())
        }));
        if unwind != outcome.is_err()
            || !observed.load(std::sync::atomic::Ordering::Acquire)
            || !stream.query()?
        {
            return Err("predictor completion failed on return or unwinding".into());
        }
    }
    Ok(())
}

/// Validate captured/eager predictor plans against the original synchronous path.
///
/// Checks exact codes/sums, replay ownership, rejected inputs, foreign-stream
/// readiness and sticky guard poisoning. Intentional host panics are caught
/// by cleanup probes. Returns validation or CUDA errors; uses loaded weights.
pub(crate) fn validate(stream: Arc<CudaStream>, weights: Arc<QwenWeights>) -> Result<()> {
    stream.context().bind_to_thread()?;
    let test_stream = stream.context().new_stream()?;
    selector(&test_stream)?;
    pitched_embedding(&test_stream)?;
    capture_cleanup(&test_stream)?;
    completion_cleanup(&test_stream)?;
    let runner = Runner::new(stream.clone())?;
    let mut graph = PredictorPlan::new(stream.clone(), weights.clone(), true)?;
    let mut eager = PredictorPlan::new(stream.clone(), weights.clone(), false)?;
    let mut retained = None;
    for seed in [0, 7, 0] {
        let (last, zero) = inputs(&stream, seed)?;
        let original =
            generation::predictor_greedy_frame(&runner, &weights.predictor()?, &last, &zero)?;
        let captured = graph.run(&last, &zero)?;
        let uncaptured = eager.run(&last, &zero)?;
        same_frame(&stream, &captured, &original, "graph versus original")?;
        same_frame(
            &stream,
            &uncaptured,
            &original,
            "eager plan versus original",
        )?;
        if let Some((prior, expected)) = &retained {
            same_frame(&stream, prior, expected, "output retained across replay")?;
        }
        // Caller inputs may be destroyed immediately after synchronous return.
        drop(last);
        drop(zero);
        retained = Some((captured, original));
    }

    // Invalid row/context checks precede enqueue and do not poison a valid plan.
    let (last, zero) = inputs(&stream, 3)?;
    let empty = Batch::<2048>::from_host(&stream, 0, &[])?;
    if graph.run(&empty, &zero).is_ok() || graph.poisoned {
        return Err("predictor accepted invalid input rows or poisoned on rejection".into());
    }
    let other_context = cuda_core::CudaContext::new(0)?;
    let wrong = Batch::from_host(
        &other_context.default_stream(),
        1,
        &[Bf16::from_bits(0); 2048],
    )?;
    if graph.run(&wrong, &zero).is_ok() || graph.poisoned {
        return Err("predictor accepted a foreign context or poisoned on rejection".into());
    }
    if validate_weights(&other_context.default_stream(), &weights.predictor()?).is_ok() {
        return Err("predictor accepted foreign-context weights".into());
    }
    // Release every caller-owned checkpoint reference and the eager plan. The
    // following successful replays depend on the graph's own retained Arc.
    drop(eager);
    drop(weights);
    stream.context().bind_to_thread()?;

    // A pending foreign producer replaces NaNs. Without context readiness this
    // graph consumes invalid logits and fails; implicit stream ordering cannot
    // rescue it because both streams are nonblocking.
    let foreign = stream.context().new_stream()?;
    let mut delayed = Batch::from_host(&stream, 1, &[Bf16::from_bits(0x7fc1); 2048])?;
    let mut delayed_zero = Batch::from_host(&stream, 1, &[Bf16::from_bits(0x7fc1); 2048])?;
    let expected = graph.run(&last, &zero)?;
    let output = Batch::allocate(&stream, 1)?;
    let (release, wait_release) = std::sync::mpsc::channel();
    let (started, wait_started) = std::sync::mpsc::channel();
    foreign.launch_host_function(move || {
        let _ = started.send(());
        let _ = wait_release.recv_timeout(std::time::Duration::from_secs(5));
    })?;
    delayed
        .buffer_mut()
        .copy_from_device_async(last.buffer(), &foreign)?;
    delayed_zero
        .buffer_mut()
        .copy_from_device_async(zero.buffer(), &foreign)?;
    wait_started.recv_timeout(std::time::Duration::from_secs(5))?;
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = release.send(());
    });
    let actual = graph.run_into(&delayed, &delayed_zero, output);
    releaser
        .join()
        .map_err(|_| "foreign producer release failed")?;
    same_frame(
        &stream,
        &actual?,
        &expected,
        "delayed foreign-stream producer",
    )?;

    // A corrupted active guard is checked at each layer before its scratch can
    // be reused, propagated to the frame boundary, and poisons further replay.
    let mut scratch = graph.decode.layer.partials.to_host_vec(&graph.stream)?;
    scratch[16 * attention_kernel::STATE_SIZE] = 0.0;
    graph
        .decode
        .layer
        .partials
        .copy_from_host(&graph.stream, &scratch)?;
    let error = graph
        .run(&last, &zero)
        .err()
        .ok_or("predictor failed to reject corrupted scratch guard")?;
    if !error.to_string().contains("status 2") || !graph.poisoned || graph.run(&last, &zero).is_ok()
    {
        return Err("predictor guard failure did not poison replay".into());
    }
    drop(graph);
    if let Some((prior, expected)) = retained {
        same_frame(
            &stream,
            &prior,
            &expected,
            "output survives plan destruction",
        )?;
    }
    println!(
        "PASS predictor graph/eager/original exact codes and embedding sums, replay ownership, greedy policy, guard rejection, foreign-stream readiness, capture/error/unwind completion"
    );
    Ok(())
}
