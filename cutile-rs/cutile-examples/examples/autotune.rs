/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tune, persist, verify and dispatch RMSNorm configurations (experimental).
//!
//! Run with:
//! ```sh
//! cargo run -p cutile-examples --example autotune --features experimental-tune
//! ```
//! With no arguments, runs a self-contained tune/save/load/dispatch demo.
//! Use `-- tune /path/to/rms_norm.json` for offline tuning, then
//! `-- run /path/to/rms_norm.json` in a separate process to reuse the winners.
//! `run` never tunes: missing or rejected records/buckets use a fixed fallback.

use cuda_async::device_operation::DeviceOp;
use cuda_core::{Context, Stream};
use cutile::api::{randn, zeros};
use cutile::error::{tensor_error, Error};
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cutile::tune::{
    Autotuner, Config, L2Key, ParamValue, Record, RecordEntry, TrialState, Workspace,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    /// Row-wise RMS normalization; BLOCK_SIZE is the tuned axis.
    #[cutile::entry()]
    fn rms_norm<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        out: &mut Tensor<f32, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f32, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let rms: f32 = 1.0f32 / (rms / n + eps);
        let rms: Tile<f32, { [] }> =
            sqrt(scalar_to_tile(rms), rounding::NegativeInf, ftz::Disabled);
        let rms: f32 = tile_to_scalar(rms);
        let rms: Tile<f32, { [1, BLOCK_SIZE] }> = rms.broadcast(tile_shape);

        let w_part: Partition<f32, { [BLOCK_SIZE] }> = w.partition(shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * rms * tw;
            unsafe { out_part.store(tout, [0i32, j]) };
        }
    }
}

use my_module::rms_norm;

// Exact shape buckets avoid claiming a winner measured at one size is optimal
// at another. A real application may choose coarser buckets in its dispatcher.
const N: usize = 1024;
const TUNED_ROWS: [usize; 2] = [64, 512];
const EPS: f32 = 1e-8;

fn bucket(rows: usize) -> String {
    format!("m={rows},n={N}")
}

fn config(block_size: i64) -> Config {
    Config::new([("BLOCK_SIZE", ParamValue::Int(block_size))])
}

fn block_size(config: &Config) -> Option<usize> {
    let size = usize::try_from(config.int("BLOCK_SIZE")?).ok()?;
    (size > 0 && size <= N && N.is_multiple_of(size)).then_some(size)
}

struct Problem {
    rows: usize,
    x: Arc<Tensor<f32>>,
    w: Arc<Tensor<f32>>,
    reference: Vec<f32>,
}

impl Problem {
    fn new(rows: usize, stream: &Arc<Stream>) -> Result<Self, Error> {
        let x: Arc<Tensor<f32>> = randn(0.0, 1.0, [rows, N], None).sync_on(stream)?.into();
        let w: Arc<Tensor<f32>> = randn(0.0, 1.0, [N], None).sync_on(stream)?.into();
        let host_x = (&x).to_host_vec().sync_on(stream)?;
        let host_w = (&w).to_host_vec().sync_on(stream)?;
        let reference = host_x
            .chunks_exact(N)
            .flat_map(|row| {
                let mean_square = row.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / N as f64;
                let scale = (mean_square + f64::from(EPS)).sqrt().recip();
                row.iter()
                    .zip(&host_w)
                    .map(move |(&x, &w)| (f64::from(x) * scale * f64::from(w)) as f32)
            })
            .collect();
        Ok(Self {
            rows,
            x,
            w,
            reference,
        })
    }

    fn generics(&self, config: &Config) -> Result<Vec<String>, Error> {
        let size = block_size(config).ok_or_else(|| tensor_error("invalid RMSNorm block size"))?;
        Ok(vec![N.to_string(), size.to_string()])
    }

    // The same kernel, arguments and generics are used for key verification,
    // tuning and dispatch. Computing the key does not compile/load a cubin.
    fn l2_key(&self, config: &Config, stream: &Arc<Stream>) -> Result<String, Error> {
        let out = zeros::<f32>(&[self.rows, N])
            .sync_on(stream)?
            .partition([1, N]);
        Ok(rms_norm(self.x.clone(), self.w.clone(), out, EPS)
            .generics(self.generics(config)?)
            .l2_cache_key()?)
    }

    fn execute_and_check(&self, config: &Config, stream: &Arc<Stream>) -> Result<(), Error> {
        let out = zeros::<f32>(&[self.rows, N])
            .sync_on(stream)?
            .partition([1, N]);
        let (_, _, out, _) = rms_norm(self.x.clone(), self.w.clone(), out, EPS)
            .generics(self.generics(config)?)
            .sync_on(stream)?;
        let actual = out.unpartition().to_host_vec().sync_on(stream)?;
        if actual.len() != self.reference.len()
            || actual
                .iter()
                .zip(&self.reference)
                .any(|(&a, &b)| !a.is_finite() || (a - b).abs() > 1e-4 + 1e-4 * b.abs())
        {
            return Err(tensor_error("RMSNorm disagrees with the CPU reference"));
        }
        Ok(())
    }
}

fn tune(
    problems: &[Problem],
    configs: &[Config],
    workspace: &Workspace,
    stream: &Arc<Stream>,
    path: &Path,
) -> Result<(), Error> {
    let mut record = Record::new(workspace);
    for problem in problems {
        println!("Tuning {}", bucket(problem.rows));
        let output = Autotuner::new(&format!("rms_norm/{}", bucket(problem.rows)))
            .configs(configs.to_vec())
            .prune(|c| block_size(c).is_some())
            .run(stream, |stream, config| {
                // Numerical correctness is checked before timing; each candidate
                // owns an output buffer, reused throughout its measurements.
                problem.execute_and_check(config, stream)?;
                let generics = problem.generics(config)?;
                let out = zeros::<f32>(&[problem.rows, N])
                    .sync_on(stream)?
                    .partition([1, N]);
                let (x, w) = (problem.x.clone(), problem.w.clone());
                let mut out_slot = Some(out);
                Ok(move |s: &Arc<Stream>| {
                    let out = out_slot.take().expect("output buffer");
                    let (_, _, out, _) = rms_norm(x.clone(), w.clone(), out, EPS)
                        .generics(generics.clone())
                        .sync_on(s)?;
                    out_slot = Some(out);
                    Ok(())
                })
            })?;
        let best = output
            .best
            .ok_or_else(|| tensor_error("no valid tuning candidate"))?;
        // The paired runoff appends new trials: use the LAST measured trial of
        // the chosen winner, not its earlier sequential-search measurement.
        let (median_ms, samples) = output
            .trials
            .iter()
            .rev()
            .find_map(|trial| {
                if trial.config_id != best.id {
                    return None;
                }
                match &trial.state {
                    TrialState::Measured {
                        median_ms, reps, ..
                    } => Some((*median_ms, *reps)),
                    _ => None,
                }
            })
            .ok_or_else(|| tensor_error("winner has no measurement"))?;
        println!("{}: {} ({median_ms:.4} ms)", bucket(problem.rows), best.id);
        let l2_key = Some(L2Key::current(problem.l2_key(&best, stream)?));
        record.insert(RecordEntry {
            bucket: bucket(problem.rows),
            config: best,
            median_ms,
            samples,
            l2_key,
        });
    }
    record.save(path)?;
    println!("Saved {}", path.display());
    Ok(())
}

fn load_record(
    path: &Path,
    workspace: &Workspace,
    problems: &[Problem],
    configs: &[Config],
    stream: &Arc<Stream>,
) -> Option<Record> {
    match Record::load_verified(path, workspace, |entry| {
        let problem = problems
            .iter()
            .find(|p| bucket(p.rows) == entry.bucket)
            .ok_or_else(|| tensor_error("unknown shape bucket"))?;
        if !configs.contains(&entry.config) || block_size(&entry.config).is_none() {
            return Err(tensor_error("recorded config is not an eligible candidate"));
        }
        Ok(Some(problem.l2_key(&entry.config, stream)?))
    }) {
        Ok((record, warnings)) => {
            for warning in &warnings {
                eprintln!("Tuning record: {warning}");
            }
            // Application policy, not a library requirement: even timing-only
            // drift uses the fallback here. Never silently bless a stale record
            // by copying its source/toolchain identity into the live workspace.
            if warnings.is_empty() {
                Some(record)
            } else {
                eprintln!("Using fallback; re-tune offline to refresh this record.");
                None
            }
        }
        Err(error) => {
            eprintln!("Tuning record unavailable or rejected: {error}; using fallback.");
            None
        }
    }
}

fn select_config(
    record: Option<&Record>,
    rows: usize,
    configs: &[Config],
) -> (Config, &'static str) {
    if let Some(entry) = record.and_then(|r| r.get(&bucket(rows))) {
        if configs.contains(&entry.config) && block_size(&entry.config).is_some() {
            return (entry.config.clone(), "record");
        }
    }
    (config(128), "fallback")
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, path) = match args.as_slice() {
        [] => (
            "demo",
            std::env::temp_dir().join(format!("cutile-rms-norm-{}.json", std::process::id())),
        ),
        [mode, path] if mode == "tune" || mode == "run" => (mode.as_str(), PathBuf::from(path)),
        _ => return Err(tensor_error("usage: autotune [tune|run RECORD_PATH]")),
    };
    if !cutile_examples::requirements::BASELINE.check("autotune", 0)? {
        return Ok(());
    }
    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let problems: Vec<Problem> = TUNED_ROWS
        .into_iter()
        .map(|m| Problem::new(m, &stream))
        .collect::<Result<_, _>>()?;
    // 4096 is deliberately ineligible and pruned before timing.
    let configs: Vec<Config> = [32, 64, 128, 256, 4096].into_iter().map(config).collect();
    let workspace = Workspace {
        kernel: "rms_norm".into(),
        source_hash: my_module::_SOURCE_HASH.to_string(),
        arch: cutile_compiler::cuda_tile_runtime_utils::get_gpu_name(0),
        tileiras_fingerprint: cutile_compiler::cuda_tile_runtime_utils::tileiras_fingerprint()
            .to_string(),
        space_hash: Some(cutile::tune::space_hash(&configs)),
    };
    if mode != "run" {
        tune(&problems, &configs, &workspace, &stream, &path)?;
    }
    if mode == "tune" {
        return Ok(());
    }

    // Deliberately reload from disk instead of dispatching with output.best.
    // In production, perform this once at startup and retain the verified record.
    let record = load_record(&path, &workspace, &problems, &configs, &stream);
    if mode == "demo" && record.is_none() {
        return Err(tensor_error("fresh record failed verification"));
    }
    for problem in problems
        .iter()
        .chain(std::iter::once(&Problem::new(256, &stream)?))
    {
        // 256 rows was not tuned: demonstrates a missing-bucket fallback even
        // when the record itself is valid. Different buckets may share a winner.
        let (selected, source) = select_config(record.as_ref(), problem.rows, &configs);
        problem.execute_and_check(&selected, &stream)?;
        println!(
            "PASS {}: {} from {source}",
            bucket(problem.rows),
            selected.id
        );
    }
    if mode == "demo" {
        std::fs::remove_file(&path)
            .map_err(|e| tensor_error(&format!("remove demo record: {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_selects_each_bucket_and_falls_back_on_miss() {
        let configs = vec![config(32), config(128), config(256)];
        let ws = Workspace {
            kernel: "rms_norm".into(),
            source_hash: "source".into(),
            arch: "sm_89".into(),
            tileiras_fingerprint: "toolchain".into(),
            space_hash: None,
        };
        let mut record = Record::new(&ws);
        for (rows, size) in [(64, 32), (512, 256)] {
            record.insert(RecordEntry {
                bucket: bucket(rows),
                config: config(size),
                median_ms: 1.0,
                samples: 5,
                l2_key: None,
            });
        }
        assert_eq!(
            select_config(Some(&record), 64, &configs),
            (config(32), "record")
        );
        assert_eq!(
            select_config(Some(&record), 512, &configs),
            (config(256), "record")
        );
        assert_eq!(
            select_config(Some(&record), 256, &configs),
            (config(128), "fallback")
        );
        assert_eq!(select_config(None, 64, &configs), (config(128), "fallback"));
        record.entries[0].config = config(4096);
        assert_eq!(
            select_config(Some(&record), 64, &configs),
            (config(128), "fallback")
        );
    }

    #[test]
    fn kernel_requires_a_positive_divisor_of_n() {
        for size in [-1, 0, 3, 4096] {
            assert!(block_size(&config(size)).is_none());
        }
        assert_eq!(block_size(&config(128)), Some(128));
    }
}
