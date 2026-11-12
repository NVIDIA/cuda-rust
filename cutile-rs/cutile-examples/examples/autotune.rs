/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pipeline CPU preparation and GPU evaluation with `cutile::tune` (experimental).
//!
//! Run with:
//! ```sh
//! cargo run -p cutile-examples --example autotune --features experimental-tune
//! ```

use cuda_async::device_operation::DeviceOp;
use cuda_core::Context;
use cutile::api::{randn, zeros};
use cutile::bench::{do_bench_controlled, BenchOptions};
use cutile::error::Error;
use cutile::tensor::{PartitionMut, Tensor, ToHostVec};
use cutile::tile_kernel::jit_compile_count;
use cutile::tile_kernel::TileKernel;
use cutile::tune::pipeline::{Failure, Pipeline, RandomOrder, RunStatus};
use cutile::tune::{Config, ParamValue, Trial};
use std::sync::Arc;
use std::time::Duration;

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
        let mut out_part: PartitionMut<f32, { [1, BLOCK_SIZE] }> = out.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * rms * tw;
            out_part.store(tout, [0i32, j]);
        }
    }
}

use my_module::rms_norm;

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("autotune", 0)? {
        return Ok(());
    }

    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let (m, n) = (512usize, 1024usize);
    let eps: f32 = 1e-8;

    let mut workers = 1;
    let mut overlap = true;
    let mut budget_ms = 30_000;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workers" => {
                workers = args
                    .next()
                    .ok_or_else(|| example_error("missing worker count"))?
                    .parse()
                    .map_err(|_| example_error("invalid worker count"))?
            }
            "--budget-ms" => {
                budget_ms = args
                    .next()
                    .ok_or_else(|| example_error("missing budget"))?
                    .parse()
                    .map_err(|_| example_error("invalid budget"))?
            }
            "--serial" => overlap = false,
            _ => {
                return Err(example_error(
                    "usage: autotune [--workers N] [--budget-ms N] [--serial]",
                ))
            }
        }
    }

    // Input creation and the CPU reference are fixture setup, outside the search
    // budget. Resolve, compile, load, check and benchmark costs are inside it.
    let x: Arc<Tensor<f32>> = randn(0.0, 1.0, [m, n], Some(0)).sync_on(&stream)?.into();
    let w: Arc<Tensor<f32>> = randn(0.0, 1.0, [n], Some(1)).sync_on(&stream)?.into();
    let hx = (&x).to_host_vec().sync_on(&stream)?;
    let hw = (&w).to_host_vec().sync_on(&stream)?;
    let mut reference = Vec::with_capacity(m * n);
    for row in hx.chunks_exact(n) {
        let rms = (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n as f64
            + f64::from(eps))
        .sqrt();
        reference.extend(
            row.iter()
                .zip(&hw)
                .map(|(&v, &weight)| f64::from(v) * f64::from(weight) / rms),
        );
    }
    let configs: Vec<Config> = [128i64, 16, 32, 64, 256, 512, 1024]
        .into_iter()
        .map(|bs| Config::new([("BLOCK_SIZE", ParamValue::Int(bs))]))
        .collect();
    let mut strategy = RandomOrder::new(42).with_initial([0]);
    let output = Pipeline::default()
        .workers(workers)
        .overlap(overlap)
        .budget(Duration::from_millis(budget_ms))
        .trial_limit(Duration::from_secs(10))
        .run(
            &configs,
            &mut strategy,
            |config| {
                // Resolve real argument metadata on the GPU lane. The job owns
                // no tensors or CUDA stream and builds its AST on a CPU worker.
                let mut out = zeros::<f32>(&[m, n])
                    .sync_on(&stream)
                    .map_err(Error::from)
                    .map_err(Failure::from)?;
                rms_norm(x.clone(), w.clone(), (&mut out).partition([1, n]), eps)
                    .generics(vec![
                        n.to_string(),
                        config.int("BLOCK_SIZE").unwrap().to_string(),
                    ])
                    .specialize_on(&stream)
                    .map_err(Error::from)
                    .map_err(Failure::from)?
                    .into_compile_job()
                    .map_err(Failure::from)
            },
            |job, control| job.compile(control).map_err(Failure::from),
            |config, prepared, control| {
                let mut out = zeros::<f32>(&[m, n])
                    .sync_on(&stream)
                    .map_err(Error::from)
                    .map_err(Failure::from)?;
                let generics = vec![n.to_string(), config.int("BLOCK_SIZE").unwrap().to_string()];
                let actual = rms_norm(x.clone(), w.clone(), (&mut out).partition([1, n]), eps)
                    .generics(generics.clone())
                    .specialize_on(&stream)
                    .map_err(Error::from)
                    .map_err(Failure::from)?;
                if actual.l1_cache_key() != prepared.key() {
                    return Err(Failure::infrastructure(
                        "prepared specialization differs from actual launch",
                    ));
                }
                let disk_hit = prepared.disk_cache_hit();
                prepared.install(control).map_err(Failure::from)?;
                let before = jit_compile_count();
                rms_norm(x.clone(), w.clone(), (&mut out).partition([1, n]), eps)
                    .generics(generics.clone())
                    .sync_on(&stream)
                    .map_err(Error::from)
                    .map_err(Failure::from)?;
                if jit_compile_count() != before {
                    return Err(Failure::infrastructure(
                        "foreground launch unexpectedly compiled again",
                    ));
                }
                let out = Arc::new(out);
                let values = (&out)
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(Error::from)
                    .map_err(Failure::from)?;
                for (i, (&actual, &expected)) in values.iter().zip(&reference).enumerate() {
                    if !actual.is_finite()
                        || (f64::from(actual) - expected).abs() > 2e-4 * (1.0 + expected.abs())
                    {
                        return Err(Failure::invalid(format!(
                            "numerical mismatch at {i}: {actual} vs {expected}"
                        )));
                    }
                }
                let mut out = Arc::try_unwrap(out).map_err(|_| {
                    Failure::infrastructure("output still retained after synchronized copy")
                })?;
                let measurement =
                    do_bench_controlled(&stream, &BenchOptions::default(), control, |s| {
                        rms_norm(x.clone(), w.clone(), (&mut out).partition([1, n]), eps)
                            .generics(generics.clone())
                            .sync_on(s)?;
                        Ok(())
                    })
                    .map_err(Failure::from)?;
                eprintln!(
                    "{}: correctness passed, prepared binary reused, disk_hit={disk_hit}",
                    config.id
                );
                Ok(Trial::measured(
                    &config.id,
                    measurement.median_ms(),
                    measurement.min_ms(),
                    measurement.reps(),
                ))
            },
        )
        .map_err(|error| example_error(&error.reason))?;
    println!(
        "workers={workers} overlap={overlap} status={:?} elapsed_ms={:.3}",
        output.status,
        output.elapsed.as_secs_f64() * 1e3
    );
    for observation in &output.observations {
        println!(
            "{}: prepare_ms={:?} evaluate_ms={:?} score_ms={:?} failure={:?}",
            observation.config_id,
            observation.stages.prepare.map(|d| d.as_secs_f64() * 1e3),
            observation.stages.evaluate.map(|d| d.as_secs_f64() * 1e3),
            observation.score_ms(),
            observation.failure
        );
    }
    println!(
        "best observed (requires fresh incumbent comparison): {:?}",
        output.best_observed.map(|c| c.id)
    );
    if output.status == RunStatus::Incomplete {
        return Err(example_error(
            output.error.as_deref().unwrap_or("incomplete tuning run"),
        ));
    }
    Ok(())
}

fn example_error(message: &str) -> Error {
    cutile::error::kernel_launch_error(message)
}
