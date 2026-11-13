/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! FP16 QKV projection -> FP32 Q/K RMSNorm -> FP8 Q and paged KV stores.
//!
//! No arguments: full-output correctness checks on a small tail/scatter fixture.
//! `--bench`: opt-in CUDA graph timing; `--help` lists shapes and tuning options.
//! Use `--profile b200` to evaluate SM100 schedules with CUDA 13.5 tooling.
//! Compiler support for individual schedules can vary. Defaults remain unchanged;
//! timings do not imply GEMM/epilogue overlap.

use cuda_async::cuda_graph::{CudaGraph, Scope};
use cuda_core::{f8e4m3fn, Context, Stream};
use cutile::half::f16;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use cutile_kernels::experimental::fused_qkv;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const D: usize = 128;
const EPS: f32 = 1e-6;
const SENTINEL: u8 = 0x55;

#[derive(Clone, Debug)]
struct Config {
    bench: bool,
    automatic_schedule: bool,
    cga: usize,
    m_list: Vec<usize>,
    k: usize,
    q_heads: usize,
    kv_heads: usize,
    bm: usize,
    bk: usize,
    group_m: usize,
    programs: usize,
    samples: usize,
    replays: usize,
    warmups: usize,
    nontrivial: bool,
}

impl Config {
    fn for_problem(&self, problem: Problem, sm_count: usize, sm100: bool) -> Self {
        let mut config = self.clone();
        if self.automatic_schedule
            && sm100
            && sm_count > 0
            && problem.k == 4096
            && problem.hq == 32
            && problem.hkv == 8
            && [128, 256, 512, 1024, 2048, 4096, 8192].contains(&problem.m)
        {
            config.bm = 128;
            config.bk = if problem.m >= 4096 { 64 } else { 128 };
            config.group_m = if problem.m == 2048 { 1 } else { 8 };
            config.programs = if problem.m == 256 { 0 } else { sm_count };
            config.cga = if problem.m == 128 { 2 } else { 1 };
            if config.programs > 0 {
                config.programs = config.programs.div_ceil(config.cga) * config.cga;
            }
        }
        config
    }

    fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.iter().any(|arg| arg == "--help") {
            println!(
                "fused_qkv [--bench] [--m-list 128,256,512,1024,2048,4096,8192]\n\
                [--k 4096] [--q-heads 32] [--kv-heads 8] [--bm 64] [--bk 64]\n\
                [--group-m 8] [--programs 0] [--samples 20] [--replays 1000]\n\
                [--warmups 50] [--nontrivial] [--profile default|b200]\n\
                No arguments runs the small correctness fixture on ordinary/persistent grids.\n\
                --bench enables timing; --profile b200 selects evaluated SM100 schedules.\n\
                b200 requires standard benchmark dimensions and cannot mix with manual schedules.\n\
                programs=0 launches one CTA per output tile.\n\
                --nontrivial uses non-unit gamma/scales and permuted cache slots.\n\
                Other options require --bench. Head dimension is fixed at 128."
            );
            return Ok(None);
        }
        let bench = args.iter().any(|arg| arg == "--bench");
        let manual_schedule = args
            .iter()
            .any(|arg| matches!(arg.as_str(), "--bm" | "--bk" | "--group-m" | "--programs"));
        let mut config = Self {
            bench,
            automatic_schedule: false,
            cga: 1,
            m_list: if bench {
                vec![128, 256, 512, 1024, 2048, 4096, 8192]
            } else {
                vec![67]
            },
            k: if bench { 4096 } else { 96 },
            q_heads: if bench { 32 } else { 4 },
            kv_heads: if bench { 8 } else { 2 },
            bm: 64,
            bk: 64,
            group_m: 8,
            programs: 0,
            samples: 20,
            replays: 1000,
            warmups: 50,
            nontrivial: !bench,
        };
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            if arg == "--bench" {
                continue;
            }
            if !bench {
                return Err("options require --bench (or use --help)".into());
            }
            if arg == "--nontrivial" {
                config.nontrivial = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?;
            match arg.as_str() {
                "--m-list" => {
                    config.m_list = value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--k" => config.k = value.parse()?,
                "--q-heads" => config.q_heads = value.parse()?,
                "--kv-heads" => config.kv_heads = value.parse()?,
                "--bm" => config.bm = value.parse()?,
                "--bk" => config.bk = value.parse()?,
                "--group-m" => config.group_m = value.parse()?,
                "--programs" => config.programs = value.parse()?,
                "--profile" => match value.as_str() {
                    "default" => config.automatic_schedule = false,
                    "b200" => config.automatic_schedule = true,
                    _ => return Err("profile must be default or b200".into()),
                },
                "--samples" => config.samples = value.parse()?,
                "--replays" => config.replays = value.parse()?,
                "--warmups" => config.warmups = value.parse()?,
                _ => return Err(format!("unknown option {arg}").into()),
            }
        }
        if config.automatic_schedule
            && (manual_schedule
                || config.k != 4096
                || config.q_heads != 32
                || config.kv_heads != 8
                || config
                    .m_list
                    .iter()
                    .any(|m| ![128, 256, 512, 1024, 2048, 4096, 8192].contains(m)))
        {
            return Err(
                "b200 requires standard benchmark shapes and no manual schedule options".into(),
            );
        }
        if config.samples == 0
            || config.replays == 0
            || config.group_m == 0
            || config.m_list.is_empty()
            || config.programs > i32::MAX as usize
        {
            return Err(
                "samples, replays and group-m must be positive; programs must fit i32".into(),
            );
        }
        for tile in [config.bm, config.bk] {
            if ![16, 32, 64, 128].contains(&tile) {
                return Err("bm and bk must be 16, 32, 64 or 128".into());
            }
        }
        for &m in &config.m_list {
            Problem::new(m, &config)?;
        }
        Ok(Some(config))
    }
}

#[derive(Clone, Copy)]
struct Problem {
    m: usize,
    k: usize,
    hq: usize,
    hkv: usize,
    capacity: usize,
    scales: [f32; 3],
}

fn checked_extent(factors: &[usize]) -> Result<usize> {
    let count = factors.iter().try_fold(1usize, |a, b| a.checked_mul(*b));
    match count {
        Some(count) if count > 0 && count <= i32::MAX as usize => Ok(count),
        _ => {
            Err("dimensions, allocation lengths and schedule products must fit positive i32".into())
        }
    }
}

impl Problem {
    fn new(m: usize, config: &Config) -> Result<Self> {
        for dim in [m, config.k, config.q_heads, config.kv_heads] {
            checked_extent(&[dim])?;
        }
        if !config.k.is_multiple_of(8) {
            return Err("k must be a multiple of 8 for 16-byte TMA row alignment".into());
        }
        let capacity = if config.bench {
            (m + 16).div_ceil(16) * 16
        } else {
            96
        };
        let problem = Self {
            m,
            k: config.k,
            hq: config.q_heads,
            hkv: config.kv_heads,
            capacity,
            scales: if config.nontrivial {
                [0.0625, 0.125, 0.25]
            } else {
                [0.0625; 3]
            },
        };
        checked_extent(&[problem.n(), problem.k])?;
        checked_extent(&[m, problem.k])?;
        checked_extent(&[m, problem.hq, D])?;
        checked_extent(&[capacity, problem.hkv, D])?;
        checked_extent(&[config.group_m, problem.heads()])?;
        checked_extent(&[m.div_ceil(config.bm), problem.heads()])?;
        checked_extent(&[m + config.bm - 1])?;
        checked_extent(&[problem.k + config.bk - 1])?;
        Ok(problem)
    }
    fn heads(self) -> usize {
        self.hq + 2 * self.hkv
    }
    fn n(self) -> usize {
        self.heads() * D
    }
    fn lengths(self) -> [usize; 3] {
        [
            self.m * self.hq * D,
            self.capacity * self.hkv * D,
            self.capacity * self.hkv * D,
        ]
    }
}

struct Fixture {
    x: Vec<f16>,
    w: Vec<f16>,
    gq: Vec<f32>,
    gk: Vec<f32>,
    slots: Vec<i64>,
}

impl Fixture {
    fn new(p: Problem, config: &Config) -> Self {
        let x = (0..p.m * p.k)
            .map(|i| {
                f16::from_f32(if i < p.k {
                    0.0
                } else {
                    (((i * 13 + i / 7) % 17) as f32 - 8.0) * 0.125
                })
            })
            .collect();
        let w = (0..p.n() * p.k)
            .map(|i| f16::from_f32((((i * 7 + i / 13) % 13) as f32 - 6.0) * 0.0625))
            .collect();
        let gq = (0..D)
            .map(|i| {
                if config.nontrivial {
                    0.5 + (i % 7) as f32 * 0.125
                } else {
                    1.0
                }
            })
            .collect();
        let gk = (0..D)
            .map(|i| {
                if config.nontrivial {
                    -0.75 + (i % 11) as f32 * 0.25
                } else {
                    1.0
                }
            })
            .collect();
        let slots = (0..p.m)
            .map(|i| {
                if !config.bench {
                    if i % 11 == 0 {
                        -1
                    } else {
                        ((i * 5) % p.capacity) as i64
                    }
                } else if config.nontrivial {
                    // Reversal with an offset is a bijection for every capacity.
                    ((p.capacity - 1 - i + 7) % p.capacity) as i64
                } else {
                    i as i64
                }
            })
            .collect();
        Self {
            x,
            w,
            gq,
            gk,
            slots,
        }
    }

    fn validate_slots(&self, p: Problem) -> Result<()> {
        let mut seen = vec![false; p.capacity];
        for &slot in &self.slots {
            if slot < 0 {
                continue;
            }
            let slot = usize::try_from(slot)?;
            if slot >= p.capacity || std::mem::replace(&mut seen[slot], true) {
                return Err("nonnegative cache slots must be unique and in bounds".into());
            }
        }
        Ok(())
    }
}

fn decode_fp8(bits: u8) -> f64 {
    let sign = if bits & 128 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 15;
    let frac = bits & 7;
    let magnitude = if exp == 0 {
        f64::from(frac) / 512.0
    } else if exp == 15 && frac == 7 {
        f64::NAN
    } else {
        (1.0 + f64::from(frac) / 8.0) * 2.0f64.powi(i32::from(exp) - 7)
    };
    sign * magnitude
}

// Independent, intentionally simple host round-to-nearest-even converter.
fn encode_fp8(value: f64) -> u8 {
    assert!(value.is_finite());
    let magnitude = value.abs().min(448.0);
    let mut best = 0u8;
    let mut error = f64::INFINITY;
    for bits in 0u8..=126 {
        let candidate = (decode_fp8(bits) - magnitude).abs();
        if candidate < error || (candidate == error && bits & 1 == 0) {
            best = bits;
            error = candidate;
        }
    }
    best | if value.is_sign_negative() { 128 } else { 0 }
}

/// Full FP64 dot products for selected rows, with independent FP8 quantization.
/// None means a written element whose numerical value is not checked in this run.
fn reference(p: Problem, f: &Fixture, rows: &[usize]) -> [Vec<Option<u8>>; 3] {
    let lengths = p.lengths();
    let mut output = [
        vec![None; lengths[0]],
        vec![Some(SENTINEL); lengths[1]],
        vec![Some(SENTINEL); lengths[2]],
    ];
    for &slot in &f.slots {
        if slot >= 0 {
            let range = slot as usize * p.hkv * D..(slot as usize + 1) * p.hkv * D;
            output[1][range.clone()].fill(None);
            output[2][range].fill(None);
        }
    }
    for &row in rows {
        for head in 0..p.heads() {
            let mut acc = [0.0f64; D];
            for (col, value) in acc.iter_mut().enumerate() {
                for inner in 0..p.k {
                    *value += f64::from(f.x[row * p.k + inner].to_f32())
                        * f64::from(f.w[(head * D + col) * p.k + inner].to_f32());
                }
            }
            let kind = if head < p.hq {
                0
            } else if head < p.hq + p.hkv {
                1
            } else {
                2
            };
            if kind < 2 {
                let inv_rms = (acc.iter().map(|v| v * v).sum::<f64>() / D as f64 + f64::from(EPS))
                    .sqrt()
                    .recip();
                let gamma = if kind == 0 { &f.gq } else { &f.gk };
                for col in 0..D {
                    acc[col] *= inv_rms * f64::from(gamma[col]);
                }
            }
            let offset = if kind == 0 {
                row * p.hq * D + head * D
            } else {
                if f.slots[row] < 0 {
                    continue;
                }
                let cache_head = head - p.hq - if kind == 2 { p.hkv } else { 0 };
                f.slots[row] as usize * p.hkv * D + cache_head * D
            };
            for col in 0..D {
                output[kind][offset + col] = Some(encode_fp8(acc[col] * f64::from(p.scales[kind])));
            }
        }
    }
    output
}

fn check(name: &str, got: &[f8e4m3fn], expected: &[Option<u8>], written: &[bool]) {
    assert_eq!(got.len(), expected.len());
    assert_eq!(written.len(), expected.len());
    let (mut checked, mut changed_codes, mut max_abs) = (0, 0, 0.0f64);
    for (index, (actual, want)) in got.iter().zip(expected).enumerate() {
        // Every output must be finite, even when only some rows have an oracle.
        assert!(
            decode_fp8(actual.0).is_finite(),
            "{name}[{index}]: nonfinite output"
        );
        let Some(want) = *want else {
            continue;
        };
        checked += 1;
        if !written[index] {
            assert_eq!(
                actual.0, SENTINEL,
                "{name}[{index}]: unwritten cache slot changed"
            );
            continue;
        }
        let value = decode_fp8(actual.0);
        let reference = decode_fp8(want);
        // Allow one FP8 spacing for FP32 reductions/rsqrt versus the FP64 oracle.
        let magnitude_bits = want & 127;
        let upper = decode_fp8((magnitude_bits + 1).min(126));
        let lower = decode_fp8(magnitude_bits.saturating_sub(1));
        let spacing = (upper - reference.abs())
            .max(reference.abs() - lower)
            .max(1.0 / 512.0);
        let err = (value - reference).abs();
        assert!(
            err <= spacing * 1.001,
            "{name}[{index}]: got {value} (0x{:02x}), expected {reference} (0x{want:02x})",
            actual.0
        );
        max_abs = max_abs.max(err);
        changed_codes += usize::from(actual.0 != want);
    }
    println!("{name}: {checked}/{} elements checked against oracle/sentinels; differing FP8 codes={changed_codes}, max_abs={max_abs}", got.len());
}

struct Tensors {
    x: Arc<Tensor<f16>>,
    w: Arc<Tensor<f16>>,
    gq: Arc<Tensor<f32>>,
    gk: Arc<Tensor<f32>>,
    slots: Arc<Tensor<i64>>,
    q: Arc<Tensor<f8e4m3fn>>,
    kc: Arc<Tensor<f8e4m3fn>>,
    vc: Arc<Tensor<f8e4m3fn>>,
}

impl Tensors {
    fn reset_outputs(&self, p: Problem, stream: &Arc<Stream>) -> Result<()> {
        // SAFETY: these distinct FP8 allocations are retained by self and idle
        // on entry. No aliases access them during this same-stream byte fill.
        let result = (|| -> Result<()> {
            stream.context().bind_to_thread()?;
            for (tensor, len) in [&self.q, &self.kc, &self.vc].into_iter().zip(p.lengths()) {
                unsafe {
                    cuda_core::memory::memset_d8_async(
                        tensor.device_pointer().cu_deviceptr(),
                        SENTINEL,
                        len,
                        stream.cu_stream(),
                    )?;
                }
            }
            Ok(())
        })();
        stream.synchronize()?;
        result
    }

    fn new(p: Problem, f: Fixture, stream: &Arc<Stream>) -> Result<Self> {
        f.validate_slots(p)?;
        Ok(Self {
            x: api::copy_host_vec_to_device(&Arc::new(f.x))
                .reshape(&[p.m, p.k])
                .sync_on(stream)?
                .into(),
            w: api::copy_host_vec_to_device(&Arc::new(f.w))
                .reshape(&[p.n(), p.k])
                .sync_on(stream)?
                .into(),
            gq: api::copy_host_vec_to_device(&Arc::new(f.gq))
                .sync_on(stream)?
                .into(),
            gk: api::copy_host_vec_to_device(&Arc::new(f.gk))
                .sync_on(stream)?
                .into(),
            slots: api::copy_host_vec_to_device(&Arc::new(f.slots))
                .sync_on(stream)?
                .into(),
            q: api::full(f8e4m3fn(SENTINEL), &[p.m, p.hq * D])
                .sync_on(stream)?
                .into(),
            kc: api::full(f8e4m3fn(SENTINEL), &[p.capacity, p.hkv * D])
                .sync_on(stream)?
                .into(),
            vc: api::full(f8e4m3fn(SENTINEL), &[p.capacity, p.hkv * D])
                .sync_on(stream)?
                .into(),
        })
    }

    fn check(
        &self,
        expected: &[Vec<Option<u8>>; 3],
        written: &[Vec<bool>; 3],
        stream: &Arc<Stream>,
    ) -> Result<()> {
        for (i, (name, tensor)) in [("Q", &self.q), ("K cache", &self.kc), ("V cache", &self.vc)]
            .into_iter()
            .enumerate()
        {
            check(
                name,
                &tensor.to_host_vec().sync_on(stream)?,
                &expected[i],
                &written[i],
            );
        }
        Ok(())
    }
}

fn launch(
    p: Problem,
    config: &Config,
    programs: usize,
    t: &Tensors,
    stream: &Arc<Stream>,
    scope: Option<&Scope>,
) -> std::result::Result<(), cuda_async::error::DeviceError> {
    // SAFETY: validated shapes and specialization satisfy the entry contract.
    // Allocations are distinct, contiguous and retained by the operation/graph.
    // Slots are unique/in bounds or negative. All launches and reads are ordered
    // on one stream; asynchronous replays are drained before any owner is dropped.
    let op = unsafe {
        fused_qkv(
            t.x.clone(),
            t.w.clone(),
            t.gq.clone(),
            t.gk.clone(),
            t.slots.clone(),
            t.q.clone(),
            t.kc.clone(),
            t.vc.clone(),
            p.m as i32,
            p.k as i32,
            p.hq as i32,
            p.hkv as i32,
            p.capacity as i64,
            EPS,
            p.scales[0],
            p.scales[1],
            p.scales[2],
        )
    }
    .generics(
        [config.bm, config.bk, config.group_m]
            .map(|n| n.to_string())
            .to_vec(),
    )
    .grid((programs as u32, 1, 1))
    .compile_options(
        CompileOptions::default()
            .num_cta_in_cga(config.cga as i32)
            .occupancy(1),
    );
    if let Some(scope) = scope {
        let _ = scope.record(op)?;
    } else {
        let _ = op.sync_on(stream)?;
    }
    Ok(())
}

fn time_graph(graph: &CudaGraph<()>, stream: &Arc<Stream>, config: &Config) -> Result<f64> {
    let start = stream.context().new_event()?;
    let stop = stream.context().new_event()?;
    let timing = (|| -> Result<f64> {
        // SAFETY: graph retains its allocations, uses this stream exclusively,
        // and all queued launches are drained below on success and error.
        for _ in 0..config.warmups {
            unsafe { graph.launch().async_on(stream)? };
        }
        start.record(stream)?;
        for _ in 0..config.replays {
            unsafe { graph.launch().async_on(stream)? };
        }
        stop.record(stream)?;
        stop.synchronize()?;
        let us = f64::from(start.elapsed_time(&stop)?) * 1000.0 / config.replays as f64;
        if !us.is_finite() || us <= 0.0 {
            return Err("invalid CUDA event timing".into());
        }
        Ok(us)
    })();
    stream.synchronize()?;
    timing
}

fn run(config: Config) -> Result<()> {
    if !cutile_examples::requirements::FP8.check("fused_qkv", 0)? {
        return Ok(());
    }
    let context = Context::new(0)?;
    if config.automatic_schedule && context.compute_capability()? != (10, 0) {
        return Err("the b200 profile requires an SM100 device".into());
    }
    let stream = context.new_stream()?;
    println!("fused_qkv {config:?}; FP16 X/W, FP32 accumulation/norm, E4M3FN outputs, head_dim={D}, eps={EPS}, occupancy=1");
    for &m in &config.m_list {
        let p = Problem::new(m, &config)?;
        let config = config.for_problem(
            p,
            context.multiprocessor_count()? as usize,
            context.compute_capability()? == (10, 0),
        );
        let fixture = Fixture::new(p, &config);
        fixture.validate_slots(p)?;
        let rows: Vec<_> = if config.bench && m > 8 {
            (0..8).map(|i| i * (m - 1) / 7).collect()
        } else {
            (0..m).collect()
        };
        println!(
            "M={m}, N={}, capacity={}, inverse_scales={:?}, oracle_rows={rows:?}",
            p.n(),
            p.capacity,
            p.scales
        );
        let expected = reference(p, &fixture, &rows);
        let mut cache_written = vec![false; p.capacity * p.hkv * D];
        for &slot in &fixture.slots {
            if slot >= 0 {
                cache_written[slot as usize * p.hkv * D..(slot as usize + 1) * p.hkv * D]
                    .fill(true);
            }
        }
        let written = [
            vec![true; p.m * p.hq * D],
            cache_written.clone(),
            cache_written,
        ];
        let t = Tensors::new(p, fixture, &stream)?;
        let tiles = p.m.div_ceil(config.bm) * p.heads();
        let grids = if config.bench {
            vec![if config.programs == 0 {
                tiles
            } else {
                config.programs
            }]
        } else {
            vec![tiles, 2]
        };
        for programs in grids {
            println!(
                "grid.x={programs}, bm={}, bk={}, group_m={}, cga={}",
                config.bm, config.bk, config.group_m, config.cga
            );
            t.reset_outputs(p, &stream)?;
            launch(p, &config, programs, &t, &stream, None)?;
            t.check(&expected, &written, &stream)?;
            let graph = CudaGraph::scope(&stream, |scope| {
                launch(p, &config, programs, &t, &stream, Some(scope))
            })?;
            t.reset_outputs(p, &stream)?;
            graph.launch().sync_on(&stream)?;
            t.check(&expected, &written, &stream)?;
            if config.bench {
                let mut samples = Vec::with_capacity(config.samples);
                println!(
                    "sample_csv,m,k,q_heads,kv_heads,bm,bk,group_m,programs,sample,latency_us"
                );
                for sample in 0..config.samples {
                    let us = time_graph(&graph, &stream, &config)?;
                    println!(
                        "sample_csv,{m},{},{},{},{},{},{},{programs},{sample},{us:.6}",
                        p.k, p.hq, p.hkv, config.bm, config.bk, config.group_m
                    );
                    samples.push(us);
                }
                t.check(&expected, &written, &stream)?;
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                samples.sort_by(f64::total_cmp);
                let median = (samples[(samples.len() - 1) / 2] + samples[samples.len() / 2]) / 2.0;
                println!(
                    "M={m}: mean={mean:.3} us, median={median:.3} us, min={:.3} us, max={:.3} us",
                    samples[0],
                    samples[samples.len() - 1]
                );
            }
        }
    }
    println!(
        "fused_qkv passed. B200 performance and epilogue overlap require separate evaluation."
    );
    Ok(())
}

fn main() -> Result<()> {
    if let Some(config) = Config::parse(&std::env::args().skip(1).collect::<Vec<_>>())? {
        run(config)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(args: &str) -> Result<Option<Config>> {
        Config::parse(
            &args
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn rejects_unsafe_or_invalid_launch_configs() {
        for args in [
            "--bench --m-list 0",
            "--bench --k 97",
            "--bench --q-heads 0",
            "--bench --kv-heads 0",
            "--bench --bm 3",
            "--bench --bk 256",
            "--bench --group-m 0",
            "--bench --programs 2147483648",
            "--bench --k 2147483640",
            "--bench --q-heads 2147483647",
            "--bench --samples 0",
            "--bench --replays 0",
            "--bench --k",
            "--bench --unknown 1",
            "--k 128",
        ] {
            assert!(config(args).is_err(), "accepted {args}");
        }
        assert!(config("--help").unwrap().is_none());
    }

    #[test]
    fn cache_slots_are_unique_and_checked_before_launch() {
        // Capacity 1040 catches permutations that incorrectly assume gcd(5, capacity)=1.
        let c = config("--bench --m-list 1024 --k 8 --q-heads 1 --kv-heads 1 --nontrivial")
            .unwrap()
            .unwrap();
        let p = Problem::new(1024, &c).unwrap();
        let mut f = Fixture::new(p, &c);
        f.validate_slots(p).unwrap();
        f.slots[1] = f.slots[0];
        assert!(f.validate_slots(p).is_err());
        f.slots[1] = p.capacity as i64;
        assert!(f.validate_slots(p).is_err());
        f.slots[1] = -1;
        f.validate_slots(p).unwrap();
    }

    #[test]
    fn fp8_reference_roundtrips_and_saturates() {
        for code in 0u8..=254 {
            if code & 127 != 127 {
                assert_eq!(encode_fp8(decode_fp8(code)), code);
            }
        }
        assert_eq!(encode_fp8(1000.0), 0x7e);
        assert_eq!(encode_fp8(-1000.0), 0xfe);
        assert_eq!(encode_fp8(1.0625), 0x38); // even low neighbor
        assert_eq!(encode_fp8(1.1875), 0x3a); // even high neighbor
    }

    #[test]
    fn grouped_scheduler_visits_every_tile_once() {
        for nm in 1usize..=20 {
            for nh in [1, 4, 48] {
                for programs in [1, 2, 7, nm * nh, nm * nh + 1] {
                    let mut seen = vec![false; nm * nh];
                    for pid in 0..programs {
                        for tile in (pid..nm * nh).step_by(programs) {
                            let first = tile / (8 * nh) * 8;
                            let gm = (nm - first).min(8);
                            let tm = first + tile % gm;
                            let head = tile % (8 * nh) / gm;
                            assert!(tm < nm && head < nh);
                            assert!(!seen[tm * nh + head]);
                            seen[tm * nh + head] = true;
                        }
                    }
                    assert!(seen.iter().all(|v| *v));
                }
            }
        }
    }
}

#[cfg(test)]
mod schedule_tests {
    use super::*;

    #[test]
    fn automatic_sm100_schedule_selects_persistent_tiles() {
        let args = vec![
            "--bench".to_string(),
            "--profile".to_string(),
            "b200".to_string(),
        ];
        let config = Config::parse(&args).unwrap().unwrap();
        let problem = Problem::new(4096, &config).unwrap();
        let schedule = config.for_problem(problem, 148, true);
        assert_eq!(
            (schedule.bm, schedule.bk, schedule.programs, schedule.cga),
            (128, 64, 148, 1)
        );
        assert_eq!(config.for_problem(problem, 148, false).bm, 64);
    }

    #[test]
    fn manual_schedule_is_not_overridden() {
        let args = vec!["--bench".to_string(), "--bm".to_string(), "32".to_string()];
        let config = Config::parse(&args).unwrap().unwrap();
        let problem = Problem::new(4096, &config).unwrap();
        let schedule = config.for_problem(problem, 148, true);
        assert_eq!((schedule.bm, schedule.bk, schedule.programs), (32, 64, 0));
    }

    #[test]
    fn default_schedule_is_not_overridden() {
        let config = Config::parse(&["--bench".to_string()]).unwrap().unwrap();
        for &m in &config.m_list {
            let schedule = config.for_problem(Problem::new(m, &config).unwrap(), 148, true);
            assert_eq!(
                (schedule.bm, schedule.bk, schedule.programs, schedule.cga),
                (64, 64, 0, 1)
            );
        }
    }

    #[test]
    fn b200_profile_rejects_manual_schedules_and_other_shapes() {
        for option in [
            "--bm",
            "--bk",
            "--group-m",
            "--programs",
            "--m-list",
            "--k",
            "--q-heads",
            "--kv-heads",
        ] {
            let args = ["--bench", "--profile", "b200", option, "16"].map(str::to_string);
            assert!(Config::parse(&args).is_err(), "{option}");
        }
    }
}
