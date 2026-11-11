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
use cuda_core::{Context, Stream};
use cutile::api::{randn, zeros};
use cutile::bench::{do_bench_controlled, do_bench_paired_controlled};
use cutile::error::Error;
use cutile::tensor::{PartitionMut, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cutile::tile_kernel::{contains_cuda_function, jit_compile_count};
use cutile::tune::pipeline::{
    BayesianOptions, BayesianSearch, ConfirmationOptions, Failure, GridOrder, ParameterScale,
    Pipeline, RandomOrder, Strategy, WorkControl,
};
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

struct Problem {
    x: Arc<Tensor<f32>>,
    w: Arc<Tensor<f32>>,
    reference: Vec<f64>,
    m: usize,
    n: usize,
    eps: f32,
    stream: Arc<Stream>,
}

impl Problem {
    fn generics(&self, config: &Config) -> Vec<String> {
        vec![
            self.n.to_string(),
            config.int("BLOCK_SIZE").unwrap().to_string(),
        ]
    }

    fn launch(&self, config: &Config, out: &mut Tensor<f32>) -> Result<(), Error> {
        rms_norm(
            self.x.clone(),
            self.w.clone(),
            out.partition([1, self.n]),
            self.eps,
        )
        .generics(self.generics(config))
        .sync_on(&self.stream)?;
        Ok(())
    }

    // Fresh validation also runs during confirmation; a screening pass is not
    // reused as evidence that either paired arm is still correct.
    fn check(
        &self,
        config: &Config,
        mut out: Tensor<f32>,
        control: &WorkControl,
    ) -> Result<Tensor<f32>, Failure> {
        control.check().map_err(Failure::from)?;
        // An incumbent need not have been screened. Keep a possible first
        // compilation under the confirmation deadline too.
        let specialization = rms_norm(
            self.x.clone(),
            self.w.clone(),
            (&mut out).partition([1, self.n]),
            self.eps,
        )
        .generics(self.generics(config))
        .specialize_on(&self.stream)
        .map_err(Error::from)
        .map_err(Failure::from)?;
        if !contains_cuda_function(specialization.l1_cache_key()) {
            specialization
                .into_compile_job()
                .map_err(Failure::from)?
                .compile(control)
                .map_err(Failure::from)?
                .install(control)
                .map_err(Failure::from)?;
        }
        self.launch(config, &mut out).map_err(Failure::from)?;
        let out = Arc::new(out);
        let values = (&out)
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(Error::from)
            .map_err(Failure::from)?;
        for (i, (&actual, &expected)) in values.iter().zip(&self.reference).enumerate() {
            if !actual.is_finite()
                || (f64::from(actual) - expected).abs() > 2e-4 * (1.0 + expected.abs())
            {
                return Err(Failure::invalid(format!(
                    "numerical mismatch at {i}: {actual} vs {expected}"
                )));
            }
        }
        control.check().map_err(Failure::from)?;
        Arc::try_unwrap(out)
            .map_err(|_| Failure::infrastructure("output still retained after synchronized copy"))
    }

    fn output(&self) -> Result<Tensor<f32>, Failure> {
        zeros::<f32>(&[self.m, self.n])
            .sync_on(&self.stream)
            .map_err(Error::from)
            .map_err(Failure::from)
    }
}

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
    let mut evidence_path = None;
    let mut search = String::from("bayes");
    let mut seed = 42;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--search" => {
                search = args.next().ok_or_else(|| example_error("missing search method"))?;
            }
            "--seed" => {
                seed = args.next().ok_or_else(|| example_error("missing seed"))?.parse()
                    .map_err(|_| example_error("invalid seed"))?;
            }
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
            "--evidence" => {
                evidence_path = Some(
                    args.next()
                        .ok_or_else(|| example_error("missing evidence path"))?,
                )
            }
            _ => return Err(example_error(
                "usage: autotune [--search bayes|random|grid] [--seed N] [--workers N] [--budget-ms N] [--serial] [--evidence NEW_FILE]",
            )),
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
    let problem = Problem {
        x,
        w,
        reference,
        m,
        n,
        eps,
        stream,
    };
    let bayesian_options = BayesianOptions {
        seed,
        scales: [("BLOCK_SIZE".into(), ParameterScale::Log2)].into(),
        ..Default::default()
    };
    // All methods start at the shipping incumbent for a fair comparison.
    // BayesianSearch itself has no preferred initial candidate.
    let mut strategy: Box<dyn Strategy> = match search.as_str() {
        "bayes" => Box::new(
            BayesianSearch::new(&configs, bayesian_options.clone())
                .and_then(|s| s.with_initial([0]))
                .map_err(|e| example_error(&e.reason))?,
        ),
        "random" => Box::new(RandomOrder::new(seed).with_initial([0])),
        "grid" => Box::new(GridOrder),
        _ => return Err(example_error("search must be bayes, random or grid")),
    };
    let confirmation = ConfirmationOptions::new(0);
    let output = Pipeline::default()
        .workers(workers)
        .overlap(overlap)
        .budget(Duration::from_millis(budget_ms))
        .trial_limit(Duration::from_secs(10))
        .run_confirmed(
            &configs,
            &mut strategy,
            &confirmation,
            |config| {
                // Resolve real argument metadata on the GPU lane. The job owns
                // no tensors or CUDA stream and builds its AST on a CPU worker.
                let mut out = problem.output()?;
                rms_norm(
                    problem.x.clone(),
                    problem.w.clone(),
                    (&mut out).partition([1, n]),
                    eps,
                )
                .generics(problem.generics(config))
                .specialize_on(&problem.stream)
                .map_err(Error::from)
                .map_err(Failure::from)?
                .into_compile_job()
                .map_err(Failure::from)
            },
            |job, control| job.compile(control).map_err(Failure::from),
            |config, prepared, opts, control| {
                let mut out = problem.output()?;
                let actual = rms_norm(
                    problem.x.clone(),
                    problem.w.clone(),
                    (&mut out).partition([1, n]),
                    eps,
                )
                .generics(problem.generics(config))
                .specialize_on(&problem.stream)
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
                let mut out = problem.check(config, out, control)?;
                if jit_compile_count() != before {
                    return Err(Failure::infrastructure(
                        "foreground launch unexpectedly compiled again",
                    ));
                }
                let measurement = do_bench_controlled(&problem.stream, opts, control, |_| {
                    problem.launch(config, &mut out)
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
            |incumbent, candidate, opts, control| {
                let mut a = problem.check(incumbent, problem.output()?, control)?;
                let mut b = problem.check(candidate, problem.output()?, control)?;
                do_bench_paired_controlled(
                    &problem.stream,
                    opts,
                    control,
                    |_| problem.launch(incumbent, &mut a),
                    |_| problem.launch(candidate, &mut b),
                )
                .map_err(Failure::from)
            },
        )
        .map_err(|error| example_error(&error.reason))?;
    println!(
        "search={search} seed={seed} workers={workers} overlap={overlap} status={:?} elapsed_ms={:.3}",
        output.status,
        output.elapsed.as_secs_f64() * 1e3
    );
    for observation in &output.search.observations {
        println!(
            "{}: prepare_ms={:?} evaluate_ms={:?} score_ms={:?} failure={:?}",
            observation.config_id,
            observation.stages.prepare.map(|d| d.as_secs_f64() * 1e3),
            observation.stages.evaluate.map(|d| d.as_secs_f64() * 1e3),
            observation.score_ms(),
            observation.failure
        );
    }
    for comparison in &output.confirmations {
        println!(
            "{} round={}: median_improvement={:?} passes={:?} failure={:?}",
            comparison.candidate_id,
            comparison.round,
            comparison.evidence.as_ref().map(|e| e.median_improvement),
            comparison.evidence.as_ref().map(|e| e.passes),
            comparison.failure
        );
    }
    println!(
        "selected={} status={:?} (incumbent={})",
        output.selected.id, output.status, output.incumbent.id
    );
    if let Some(path) = evidence_path {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| example_error(&e.to_string()))?;
        let mut evidence =
            serde_json::to_value(&output).map_err(|e| example_error(&e.to_string()))?;
        evidence["strategy"] = serde_json::json!({
            "name": search, "seed": seed, "bayesian_options": bayesian_options,
            "workers": workers, "overlap": overlap, "budget_ms": budget_ms,
        });
        evidence["configs"] = serde_json::json!(configs);
        evidence["workload"] = serde_json::json!({"kernel": "rms_norm", "rows": m, "columns": n});
        serde_json::to_writer_pretty(file, &evidence).map_err(|e| example_error(&e.to_string()))?;
    }
    if let Some(failure) = output.failure {
        eprintln!("no replacement: {failure}");
        if matches!(
            failure.kind,
            cutile::tune::pipeline::FailureKind::Infrastructure
                | cutile::tune::pipeline::FailureKind::Invalid
        ) {
            return Err(example_error(&failure.reason));
        }
    }
    Ok(())
}

fn example_error(message: &str) -> Error {
    cutile::error::kernel_launch_error(message)
}
