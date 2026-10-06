/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU integration test: a real (tiny) tuning run end to end.

#![cfg(feature = "experimental-tune")]

use cuda_core::Device;
use cutile::bench::BenchOptions;
use cutile::prelude::*;
use cutile::tune::{Autotuner, Config, ParamValue, TrialState};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn quick_bench() -> BenchOptions {
    BenchOptions {
        warmup: Duration::from_millis(5),
        rep: Duration::from_millis(15),
        min_reps: 3,
        max_reps: 20,
        clear_l2: true,
    }
}

/// Tunes a fill "kernel" over candidate sizes; one candidate is invalid by
/// construction. Exercises: gate rejection recorded, all candidates visited,
/// a winner produced, JSONL log written and resumable.
#[test]
fn end_to_end_grid_tune_on_gpu() {
    let device = Device::new(0).expect("device 0");
    let stream = device.new_stream().expect("stream");

    let configs: Vec<Config> = [1usize << 16, 1 << 20, 0]
        .into_iter()
        .map(|n| Config::new([("N", ParamValue::Int(n as i64))]))
        .collect();

    let log = std::env::temp_dir().join(format!("cutile_autotune_test_{}", std::process::id()));
    let _ = std::fs::remove_file(&log);

    let setup_calls = AtomicUsize::new(0);
    let run = |log_path: &std::path::Path| {
        Autotuner::new("fill_tune")
            .configs(configs.clone())
            .bench(quick_bench())
            .log(log_path)
            .run(&stream, |_, config| {
                setup_calls.fetch_add(1, Ordering::Relaxed);
                let n = config.int("N").unwrap() as usize;
                if n == 0 {
                    return Err(cutile::error::Error::Tensor(cutile::error::TensorError(
                        "empty shape is not tunable".into(),
                    )));
                }
                // Correctness gate: run once, verify on host.
                let t = api::full(3.0f32, &[n])
                    .sync_on(&stream)
                    .map_err(cutile::error::Error::from)?;
                let host = t
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(cutile::error::Error::from)?;
                if host[0] != 3.0 {
                    return Err(cutile::error::Error::Tensor(cutile::error::TensorError(
                        "gate: wrong fill value".into(),
                    )));
                }
                Ok(move |s: &std::sync::Arc<cuda_core::Stream>| {
                    api::full(3.0f32, &[n])
                        .sync_on(s)
                        .map(|_| ())
                        .map_err(Into::into)
                })
            })
            .expect("tuning run")
    };

    let state = run(&log);
    // 3 search trials + 2 paired-runoff trials for the two finalists.
    assert_eq!(state.trials.len(), 5, "search trials plus runoff pair");
    let invalid: Vec<_> = state
        .trials
        .iter()
        .filter(|t| matches!(t.state, TrialState::Invalid { .. }))
        .collect();
    assert_eq!(invalid.len(), 1, "the N=0 candidate is invalid");
    assert!(invalid[0].config_id.contains("N=0"));
    let best = state.best.expect("a winner");
    assert_eq!(
        best.int("N"),
        Some(1 << 16),
        "smaller fill should be faster"
    );

    // Resume: measured candidates are not re-searched; the previously
    // Invalid one is retried (its invalidity may have been transient) and
    // the two finalists are re-measured in the paired runoff — by design,
    // the winner decision is always contemporaneous.
    let calls_before = setup_calls.load(Ordering::Relaxed);
    let resumed = run(&log);
    let resumed_calls = setup_calls.load(Ordering::Relaxed) - calls_before;
    assert_eq!(
        resumed_calls, 3,
        "resume setup calls = 1 invalid retry + 2 runoff finalists"
    );
    assert!(resumed.best.is_some());
    let _ = std::fs::remove_file(&log);
}

// ── Record end-to-end with a real kernel and real L2 verification ─────────

#[cutile::module]
mod tune_test_module {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn scale<const N: i32>(z: &mut Tensor<f32, { [N] }>, x: &Tensor<f32, { [-1] }>) {
        let tx = load_tile_like(x, z);
        z.store(tx + tx);
    }
}

/// Tunes over tile sizes of a real kernel, records the winner with its real
/// L2 key, and proves load_verified accepts the honest workspace and refuses
/// a tampered key.
#[test]
fn record_end_to_end_with_real_l2_verification() {
    use cutile::tune::{Record, RecordEntry, Workspace};

    let device = Device::new(0).expect("device 0");
    let stream = device.new_stream().expect("stream");
    let n: usize = 1 << 18;

    let configs: Vec<Config> = [128i64, 256]
        .into_iter()
        .map(|t| Config::new([("TILE", ParamValue::Int(t))]))
        .collect();

    let outcome = Autotuner::new("scale_tune")
        .configs(configs.clone())
        .bench(quick_bench())
        .run(&stream, |_, config| {
            let tile = config.int("TILE").unwrap() as usize;
            let launch = move |s: &std::sync::Arc<cuda_core::Stream>| {
                let x = api::ones::<f32>(&[n]).sync_on(s)?;
                let z = api::zeros::<f32>(&[n]).sync_on(s)?;
                let (z, _x) = tune_test_module::scale(z.partition([tile]), x).sync_on(s)?;
                let _ = z.unpartition();
                Ok::<(), cutile::error::Error>(())
            };
            // Correctness gate: one verified run.
            let x = api::ones::<f32>(&[n]).sync_on(&stream)?;
            let z = api::zeros::<f32>(&[n]).sync_on(&stream)?;
            let (z, _x) = tune_test_module::scale(z.partition([tile]), x).sync_on(&stream)?;
            let host = z.unpartition().to_host_vec().sync_on(&stream)?;
            if host[0] != 2.0 {
                return Err(cutile::error::Error::Tensor(cutile::error::TensorError(
                    "gate: wrong scale result".into(),
                )));
            }
            Ok(move |s: &std::sync::Arc<cuda_core::Stream>| launch(s))
        })
        .expect("tuning run");

    let best = outcome.best.clone().expect("winner");
    let best_tile = best.int("TILE").unwrap() as usize;

    // Record the winner with its REAL persistent-cache key.
    let l2_key_for = |tile: usize| {
        let x = api::ones::<f32>(&[n]);
        let z = api::zeros::<f32>(&[n]);
        tune_test_module::scale(
            z.sync_on(&stream).unwrap().partition([tile]),
            x.sync_on(&stream).unwrap(),
        )
        .l2_cache_key()
        .expect("l2 key")
    };
    let winner_key = l2_key_for(best_tile);

    let ws = Workspace {
        kernel: "scale_tune".into(),
        source_hash: tune_test_module::_SOURCE_HASH.to_string(),
        arch: cutile::cutile_compiler::cuda_tile_runtime_utils::get_gpu_name(0),
        tileiras_fingerprint:
            cutile::cutile_compiler::cuda_tile_runtime_utils::tileiras_fingerprint().to_string(),
        space_hash: Some(cutile::tune::space_hash(&configs)),
    };
    let mut record = Record::new(&ws);
    record.insert(RecordEntry {
        bucket: format!("n={n}"),
        config: best.clone(),
        median_ms: outcome
            .trials
            .iter()
            .find(|t| t.config_id == best.id)
            .and_then(|t| t.median_ms())
            .unwrap(),
        samples: 3,
        l2_key: Some(cutile::tune::L2Key::current(winner_key.clone())),
    });
    let path = std::env::temp_dir().join(format!("cutile_record_gpu_{}.json", std::process::id()));
    record.save(&path).expect("save");

    // Honest workspace + real recomputation: accepted, no warnings.
    let (loaded, warnings) = Record::load_verified(&path, &ws, |entry| {
        let tile = entry.config.int("TILE").unwrap() as usize;
        Ok(Some(l2_key_for(tile)))
    })
    .expect("verified load");
    assert!(warnings.is_empty(), "no drift expected: {warnings:?}");
    assert_eq!(loaded.get(&format!("n={n}")).unwrap().config, best);

    // Tampered stored key: refused.
    let mut tampered = loaded.clone();
    tampered.entries[0].l2_key = Some(cutile::tune::L2Key::current("0".repeat(64)));
    tampered.save(&path).expect("save tampered");
    let err = Record::load_verified(&path, &ws, |entry| {
        let tile = entry.config.int("TILE").unwrap() as usize;
        Ok(Some(l2_key_for(tile)))
    })
    .unwrap_err();
    assert!(err.to_string().contains("l2 key for bucket"));
    assert!(err.to_string().contains("re-tune"));

    let _ = std::fs::remove_file(&path);
}
