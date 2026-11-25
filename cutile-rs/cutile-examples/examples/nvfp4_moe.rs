/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Uses cutile-kernels' NVFP4 mixture of experts kernels by Emmet Zhang et al.
 * The opt-in B200 profile provides SM100 schedules for CUDA 13.5 evaluation.
 * Compiler support for individual schedules can vary.
 *
 * Run with: cargo run --release -p cutile-examples --example nvfp4_moe
 *
 * Pipeline: device routing -> input quantization -> FC1 + ReLU-squared +
 * requantization -> FC2 -> route-weighted BF16 output. Both routing kernels
 * are exercised across power-of-two batch sizes from 1 through 256. CUDA
 * events measure graph replay latency after warmup, with output validation
 * outside the timed region.
 *
 * The small synthetic fixture uses H=I=1024, eight allocated experts, top-k=4,
 * and four active experts. Constant packed weights and deterministic routes
 * admit an analytical reference. --profile b200 opts into bounded persistent
 * grids and per-batch schedules. --shape large selects H=6144, I=1536, E=192.
 */

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cuda_core::{Context, Stream};
use cutile::api;
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use std::sync::Arc;

use cutile_kernels::experimental::nvfp4_moe as kernels;

use kernels::{
    finalize_kernel, grouped_fc1_kernel, grouped_fc2_kernel, nvfp4_quantize_kernel,
    routing_group_histogram_kernel, routing_group_pairwise_kernel,
};

// Kernel config
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    hidden: usize,
    intermediate: usize,
    experts: usize,
}

const SMALL_SHAPE: Shape = Shape {
    hidden: 1024,
    intermediate: 1024,
    experts: 8,
};
const LARGE_SHAPE: Shape = Shape {
    hidden: 6144,
    intermediate: 1536,
    experts: 192,
};
const MAX_NUM_TOKENS: usize = 256;
const BATCH_SIZES: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, MAX_NUM_TOKENS];
const TOP_K: usize = 4;
const TILE_M: usize = 128;
const TILE_K: usize = 512;
const QUANTIZE_TILE: usize = 1024;
const TILE_N: usize = 16;
const FINALIZE_TILE: usize = 256;
const ROUTING_PAIRWISE_MAX_ROUTED: usize = kernels::ROUTING_PAIRWISE_CAPACITY as usize;
const TIMING_WARMUPS: usize = 50;
const TIMING_REPLAYS: usize = 1000;
const TIMING_SAMPLES: usize = 20;

// For simple precision test
const ROUTING_PATTERN: [f32; TOP_K] = [0.125, 0.25, 0.25, 0.375];
const UP_FACTORS: [f32; TOP_K] = [0.5, 1.0, 2.0, 1.0];
const DOWN_FACTORS: [f32; TOP_K] = [1.0, 2.0, 0.5, 1.0];

const INPUT_SCALE: f32 = 1.0;

impl Shape {
    fn max_num_groups(self, routed_size: usize, tile_n: usize) -> usize {
        let filled_experts = self.experts.min(routed_size);
        filled_experts + (routed_size - filled_experts) / tile_n
    }
}

#[derive(Debug)]
struct Config {
    shape: Shape,
    tuned: bool,
    check_only: bool,
    batches: Vec<usize>,
}

impl Config {
    fn parse(args: &[String]) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if args.iter().any(|arg| arg == "--help") {
            println!(
                "nvfp4_moe [--shape small|large] [--profile default|b200] \\
                 [--batch 1|2|4|8|16|32|64|128|256] [--check-only]\\n\\
                 small: H=I=1024, experts=8; large: H=6144, I=1536, experts=192.\\n\\
                 Both use the same synthetic fixture, four active experts and top-k=4.\\n\\
                 b200 selects SM100 profiles; support depends on the selected assembler.\\n\\
                 --check-only validates direct launches and graph replay without timing.\\n\\
                 No arguments preserves the original small fixture, profiles and full batch sweep."
            );
            return Ok(None);
        }
        let mut config = Self {
            shape: SMALL_SHAPE,
            tuned: false,
            check_only: false,
            batches: BATCH_SIZES.to_vec(),
        };
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            if arg == "--check-only" {
                config.check_only = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?;
            match (arg.as_str(), value.as_str()) {
                ("--shape", "small") => config.shape = SMALL_SHAPE,
                ("--shape", "large") => config.shape = LARGE_SHAPE,
                ("--profile", "default") => config.tuned = false,
                ("--profile", "b200") => config.tuned = true,
                ("--batch", _) => {
                    let batch = value.parse()?;
                    if !BATCH_SIZES.contains(&batch) {
                        return Err("batch must be a power of two from 1 through 256".into());
                    }
                    config.batches = vec![batch];
                }
                _ => return Err(format!("unknown option or value: {arg} {value}").into()),
            }
        }
        Ok(Some(config))
    }
}

fn hidden_value(token: usize) -> f32 {
    let magnitude = 0.375 * [1.0, 2.0, 4.0][token % 3];
    if token.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

fn expert_id(token: usize, slot: usize) -> usize {
    (token + slot) % TOP_K
}

fn routing_weight(token: usize, slot: usize) -> f32 {
    ROUTING_PATTERN[(token + slot) % TOP_K]
}

fn expected_output(token: usize) -> bf16 {
    let hidden = hidden_value(token);
    let mut total = 0.0f32;
    for slot in 0..TOP_K {
        let expert = expert_id(token, slot);
        let pre_activation = hidden * UP_FACTORS[expert];
        let relu = pre_activation.max(0.0);
        let activated = relu * relu;
        let fc2 = activated * DOWN_FACTORS[expert];
        total += routing_weight(token, slot) * fc2;
    }
    bf16::from_f32(total)
}

#[derive(Clone, Copy, Debug)]
struct Schedule {
    tile_n: usize,
    fc1_workers: usize,
    fc2_workers: usize,
    fc1_occupancy: i32,
    fc2_occupancy: i32,
    fc1_latencies: [usize; 2],
    fc2_latencies: [usize; 3],
    finalize_tile: usize,
    finalize_occupancy: i32,
    tuned: bool,
}

impl Schedule {
    fn new(batch: usize, sm_count: usize, tuned: bool, shape: Shape) -> Self {
        let routed = batch * TOP_K;
        let baseline = Self {
            tile_n: TILE_N,
            fc1_workers: shape.max_num_groups(routed, TILE_N),
            fc2_workers: shape.max_num_groups(routed, TILE_N),
            fc1_occupancy: 1,
            fc2_occupancy: 1,
            fc1_latencies: [1, 1],
            fc2_latencies: [1, 1, 1],
            finalize_tile: FINALIZE_TILE,
            finalize_occupancy: 1,
            tuned: false,
        };
        if !tuned || sm_count == 0 {
            return baseline;
        }
        let (
            tile_n,
            fc1_occupancy,
            fc1_ctas,
            fc1_latencies,
            fc2_occupancy,
            fc2_ctas,
            fc2_latencies,
            finalize_tile,
            finalize_occupancy,
        ) = match batch {
            1 => (8, 2, 8, [2, 2], 2, 4, [1, 6, 4], 256, 16),
            2 => (8, 1, 3, [0, 1], 2, 4, [0, 8, 7], 512, 6),
            4 => (8, 8, 1, [0, 1], 3, 4, [4, 2, 3], 512, 6),
            8 => (8, 6, 4, [2, 8], 3, 3, [8, 2, 7], 256, 16),
            16 => (16, 6, 7, [1, 2], 4, 2, [7, 3, 7], 256, 2),
            32 => (32, 6, 2, [1, 6], 4, 3, [2, 7, 4], 512, 4),
            64 => (64, 4, 2, [0, 0], 7, 1, [3, 3, 2], 512, 6),
            128 => (64, 8, 7, [4, 0], 6, 1, [4, 2, 4], 1024, 6),
            256 => (64, 4, 1, [0, 0], 4, 1, [3, 2, 3], 2048, 6),
            _ => return baseline,
        };
        let output_tiles = shape.intermediate / TILE_M;
        let required_waves = (routed.div_ceil(tile_n) * output_tiles).div_ceil(sm_count);
        let fc1_workers = shape
            .max_num_groups(routed, tile_n)
            .min((sm_count * fc1_ctas.min(required_waves) / output_tiles).max(1));
        let fc2_workers = shape
            .max_num_groups(routed, tile_n)
            .min((sm_count * fc2_ctas / (shape.hidden / TILE_M)).max(1));
        Self {
            tile_n,
            fc1_workers,
            fc2_workers,
            fc1_occupancy,
            fc2_occupancy,
            fc1_latencies,
            fc2_latencies,
            finalize_tile: finalize_tile.min(shape.hidden),
            finalize_occupancy,
            tuned: true,
        }
    }

    fn fc1_options(self) -> CompileOptions {
        if !self.tuned {
            return CompileOptions::default();
        }
        CompileOptions::default()
            .occupancy(self.fc1_occupancy)
            .num_worker_warps_per_cta(8)
            .num_cta_in_cga(if self.tile_n == 32 { 2 } else { 1 })
    }

    fn fc2_options(self) -> CompileOptions {
        if !self.tuned {
            return CompileOptions::default();
        }
        CompileOptions::default()
            .occupancy(self.fc2_occupancy)
            .num_cta_in_cga(if self.tile_n == 32 { 2 } else { 1 })
    }

    fn finalize_options(self) -> CompileOptions {
        if self.tuned {
            CompileOptions::default().occupancy(self.finalize_occupancy)
        } else {
            CompileOptions::default()
        }
    }
}

struct MoeTensors {
    shape: Shape,
    schedule: Schedule,
    hidden: Arc<Tensor<f16>>,
    expert_ids: Arc<Tensor<i32>>,
    routing_weights: Arc<Tensor<bf16>>,
    group_tokens: Arc<Tensor<i32>>,
    group_experts: Arc<Tensor<i32>>,
    group_limits: Arc<Tensor<i32>>,
    expanded_to_permuted: Arc<Tensor<i32>>,
    expert_counts: Arc<Tensor<i32>>,
    expert_offsets: Arc<Tensor<i32>>,
    num_groups: Arc<Tensor<i32>>,
    quantized_hidden: Arc<Tensor<f4e2m1fnx2>>,
    quantized_hidden_scales: Arc<Tensor<f8e4m3fn>>,
    grouped_inter: Arc<Tensor<f4e2m1fnx2>>,
    grouped_inter_scales: Arc<Tensor<f8e4m3fn>>,
    grouped_out: Arc<Tensor<bf16>>,
    w_up: Arc<Tensor<f4e2m1fnx2>>,
    w_up_scales: Arc<Tensor<f8e4m3fn>>,
    w_up_global: Arc<Tensor<f32>>,
    w_down: Arc<Tensor<f4e2m1fnx2>>,
    w_down_scales: Arc<Tensor<f8e4m3fn>>,
    w_down_global: Arc<Tensor<f32>>,
    output: Arc<Tensor<bf16>>,
    num_tokens: usize,
}

fn allocate_tensors(
    num_tokens: usize,
    config: &Config,
    stream: &Arc<Stream>,
) -> Result<MoeTensors, Box<dyn std::error::Error>> {
    let shape = config.shape;
    let schedule = Schedule::new(
        num_tokens,
        stream.context().multiprocessor_count()? as usize,
        config.tuned,
        shape,
    );
    let tile_n = schedule.tile_n;
    let routed_size = num_tokens * TOP_K;
    let max_groups = shape.max_num_groups(routed_size, tile_n);
    let hidden_host = (0..num_tokens)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(hidden_value(token)), shape.hidden))
        .collect::<Vec<_>>();
    let hidden: Arc<Tensor<f16>> = api::copy_host_vec_to_device(&Arc::new(hidden_host))
        .reshape(&[num_tokens, shape.hidden])
        .sync_on(stream)?
        .into();
    let expert_ids_host = (0..num_tokens)
        .flat_map(|token| (0..TOP_K).map(move |slot| expert_id(token, slot) as i32))
        .collect::<Vec<_>>();
    let expert_ids: Arc<Tensor<i32>> = api::copy_host_vec_to_device(&Arc::new(expert_ids_host))
        .reshape(&[num_tokens, TOP_K])
        .sync_on(stream)?
        .into();
    let routing_host = (0..num_tokens)
        .flat_map(|token| (0..TOP_K).map(move |slot| bf16::from_f32(routing_weight(token, slot))))
        .collect::<Vec<_>>();
    let routing_weights: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(routing_host))
        .reshape(&[num_tokens, TOP_K])
        .sync_on(stream)?
        .into();

    let fp4_one = f4e2m1fnx2::from_nibbles(0x2, 0x2);
    let scale_one = f8e4m3fn(0x38);
    let w_up: Arc<Tensor<f4e2m1fnx2>> = api::full(
        fp4_one,
        &[shape.experts * shape.intermediate, shape.hidden / 2],
    )
    .sync_on(stream)?
    .into();
    let w_up_scales: Arc<Tensor<f8e4m3fn>> = api::full(
        scale_one,
        &[
            shape.experts * shape.intermediate / 128,
            shape.hidden / 64,
            2,
            256,
        ],
    )
    .sync_on(stream)?
    .into();
    let mut w_up_global_host = vec![1.0f32 / shape.hidden as f32; shape.experts];
    for expert in 0..TOP_K {
        w_up_global_host[expert] = UP_FACTORS[expert] / shape.hidden as f32;
    }
    let w_up_global: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(w_up_global_host))
        .reshape(&[shape.experts])
        .sync_on(stream)?
        .into();

    let w_down: Arc<Tensor<f4e2m1fnx2>> = api::full(
        fp4_one,
        &[shape.experts * shape.hidden, shape.intermediate / 2],
    )
    .sync_on(stream)?
    .into();
    let w_down_scales: Arc<Tensor<f8e4m3fn>> = api::full(
        scale_one,
        &[
            shape.experts * shape.hidden / 128,
            shape.intermediate / 64 * 512,
        ],
    )
    .sync_on(stream)?
    .into();
    let mut w_down_global_host = vec![1.0f32 / shape.intermediate as f32; shape.experts];
    for expert in 0..TOP_K {
        w_down_global_host[expert] = DOWN_FACTORS[expert] / shape.intermediate as f32;
    }
    let w_down_global: Arc<Tensor<f32>> =
        api::copy_host_vec_to_device(&Arc::new(w_down_global_host))
            .reshape(&[shape.experts])
            .sync_on(stream)?
            .into();

    let group_tokens = api::zeros::<i32>(&[max_groups, tile_n])
        .sync_on(stream)?
        .into();
    let group_experts = api::zeros::<i32>(&[max_groups]).sync_on(stream)?.into();
    let group_limits = api::zeros::<i32>(&[max_groups]).sync_on(stream)?.into();
    let expanded_to_permuted = api::zeros::<i32>(&[num_tokens, TOP_K])
        .sync_on(stream)?
        .into();
    let expert_counts = api::zeros::<i32>(&[shape.experts]).sync_on(stream)?.into();
    let expert_offsets = api::zeros::<i32>(&[shape.experts + 1])
        .sync_on(stream)?
        .into();
    let num_groups = api::zeros::<i32>(&[1]).sync_on(stream)?.into();
    let quantized_hidden = api::zeros::<f4e2m1fnx2>(&[num_tokens, shape.hidden / 2])
        .sync_on(stream)?
        .into();
    let quantized_hidden_scales = api::zeros::<f8e4m3fn>(&[num_tokens, shape.hidden / 16])
        .sync_on(stream)?
        .into();
    let grouped_inter = api::zeros::<f4e2m1fnx2>(&[max_groups, tile_n, shape.intermediate / 2])
        .sync_on(stream)?
        .into();
    let grouped_inter_scales =
        api::zeros::<f8e4m3fn>(&[max_groups, shape.intermediate / 16, tile_n])
            .sync_on(stream)?
            .into();
    let grouped_out = api::zeros::<bf16>(&[max_groups, tile_n, shape.hidden])
        .sync_on(stream)?
        .into();
    let output = api::zeros::<bf16>(&[num_tokens, shape.hidden])
        .sync_on(stream)?
        .into();

    Ok(MoeTensors {
        shape,
        schedule,
        hidden,
        expert_ids,
        routing_weights,
        group_tokens,
        group_experts,
        group_limits,
        expanded_to_permuted,
        expert_counts,
        expert_offsets,
        num_groups,
        quantized_hidden,
        quantized_hidden_scales,
        grouped_inter,
        grouped_inter_scales,
        grouped_out,
        w_up,
        w_up_scales,
        w_up_global,
        w_down,
        w_down_scales,
        w_down_global,
        output,
        num_tokens,
    })
}

// Launch once on a stream to prepare the kernels, then capture the complete CUDA graph.
#[derive(Clone, Copy)]
enum LaunchTarget<'a> {
    Stream(&'a Arc<Stream>),
    Graph(&'a cuda_async::cuda_graph::Scope),
}

fn launch(
    target: LaunchTarget<'_>,
    op: impl GraphNode,
) -> Result<(), cuda_async::error::DeviceError> {
    match target {
        LaunchTarget::Stream(stream) => {
            let _ = op.sync_on(stream)?;
        }
        LaunchTarget::Graph(scope) => {
            let _ = scope.record(op)?;
        }
    }
    Ok(())
}

fn run_routing(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let shape = tensors.shape;
    let tile_n = tensors.schedule.tile_n;
    let routed_size = tensors.num_tokens * TOP_K;
    if routed_size <= ROUTING_PAIRWISE_MAX_ROUTED {
        let op = unsafe {
            routing_group_pairwise_kernel(
                tensors.group_tokens.clone(),
                tensors.group_experts.clone(),
                tensors.group_limits.clone(),
                tensors.expanded_to_permuted.clone(),
                tensors.num_groups.clone(),
                tensors.expert_ids.clone(),
                TOP_K as i32,
                routed_size as i32,
                tile_n as i32,
                tile_n.trailing_zeros() as i32,
            )
        }
        .grid((1, 1, 1));
        launch(target, op)?;
    } else {
        let experts_pow2 = shape.experts.next_power_of_two();
        let op = unsafe {
            routing_group_histogram_kernel(
                tensors.group_tokens.clone(),
                tensors.group_experts.clone(),
                tensors.group_limits.clone(),
                tensors.expanded_to_permuted.clone(),
                tensors.expert_counts.clone(),
                tensors.expert_offsets.clone(),
                tensors.num_groups.clone(),
                tensors.expert_ids.clone(),
                TOP_K as i32,
                shape.experts as i32,
                routed_size as i32,
                tile_n as i32,
                tile_n.trailing_zeros() as i32,
            )
        }
        .generics(vec![
            (MAX_NUM_TOKENS * TOP_K).to_string(),
            experts_pow2.to_string(),
        ])
        .grid((1, 1, 1));
        launch(target, op)?;
    }
    Ok(())
}

fn run_quantize(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let shape = tensors.shape;
    let op = unsafe {
        nvfp4_quantize_kernel(
            tensors.quantized_hidden.clone(),
            tensors.quantized_hidden_scales.clone(),
            tensors.hidden.clone(),
            INPUT_SCALE,
        )
    }
    .generics(vec![
        QUANTIZE_TILE.to_string(),
        (QUANTIZE_TILE / 2).to_string(),
        (QUANTIZE_TILE / 16).to_string(),
        (QUANTIZE_TILE / 16).to_string(),
        "16".to_string(),
        (QUANTIZE_TILE / 16).to_string(),
    ])
    .grid((
        tensors.num_tokens as u32,
        (shape.hidden / QUANTIZE_TILE) as u32,
        1,
    ));
    launch(target, op)
}

fn run_fc1(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let shape = tensors.shape;
    let tile_n = tensors.schedule.tile_n;
    let num_output_tiles = shape.intermediate / TILE_M;
    let worker_groups = tensors.schedule.fc1_workers;
    let op = unsafe {
        grouped_fc1_kernel(
            tensors.grouped_inter.clone(),
            tensors.grouped_inter_scales.clone(),
            tensors.quantized_hidden.clone(),
            tensors.quantized_hidden_scales.clone(),
            (shape.hidden / 2) as i32,
            (shape.hidden / 16) as i32,
            tensors.group_tokens.clone(),
            tensors.group_experts.clone(),
            tensors.group_limits.clone(),
            tensors.num_groups.clone(),
            tensors.w_up.clone(),
            tensors.w_up_scales.clone(),
            tensors.w_up_global.clone(),
            INPUT_SCALE,
            shape.intermediate as i32,
            (shape.hidden / TILE_K) as i32,
        )
    }
    .generics(vec![
        TILE_M.to_string(),
        (TILE_M / 32).to_string(),
        tile_n.to_string(),
        TILE_K.to_string(),
        (TILE_K / 64).to_string(),
        (TILE_K / 2).to_string(),
        (TILE_K / 16).to_string(),
        (TILE_M / 2).to_string(),
        (TILE_M / 16).to_string(),
        tensors.schedule.fc1_latencies[0].to_string(),
        tensors.schedule.fc1_latencies[1].to_string(),
        tile_n.to_string(),
        (TILE_M / 16).to_string(),
        tile_n.to_string(),
        (TILE_M / 16).to_string(),
        "16".to_string(),
    ])
    .grid((num_output_tiles as u32, worker_groups as u32, 1))
    .compile_options(tensors.schedule.fc1_options());
    launch(target, op)
}

fn run_fc2(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let shape = tensors.shape;
    let tile_n = tensors.schedule.tile_n;
    let num_output_tiles = shape.hidden / TILE_M;
    let group_workers = tensors.schedule.fc2_workers;
    let op = unsafe {
        grouped_fc2_kernel(
            tensors.grouped_out.clone(),
            tensors.grouped_inter.clone(),
            tensors.grouped_inter_scales.clone(),
            tensors.group_experts.clone(),
            tensors.num_groups.clone(),
            tensors.w_down.clone(),
            tensors.w_down_scales.clone(),
            tensors.w_down_global.clone(),
            shape.hidden as i32,
            (shape.intermediate / TILE_K) as i32,
        )
    }
    .generics(vec![
        TILE_M.to_string(),
        (TILE_M / 32).to_string(),
        tile_n.to_string(),
        TILE_K.to_string(),
        (TILE_K / 64).to_string(),
        (TILE_K / 2).to_string(),
        (TILE_K / 16).to_string(),
        (TILE_K / 64 * 512).to_string(),
        tensors.schedule.fc2_latencies[0].to_string(),
        tensors.schedule.fc2_latencies[1].to_string(),
        tensors.schedule.fc2_latencies[2].to_string(),
    ])
    .grid((num_output_tiles as u32, group_workers as u32, 1))
    .compile_options(tensors.schedule.fc2_options());
    launch(target, op)
}

fn run_finalize(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let shape = tensors.shape;
    let tile_n = tensors.schedule.tile_n;
    let op = unsafe {
        finalize_kernel(
            tensors.output.clone(),
            tensors.grouped_out.clone(),
            tensors.expanded_to_permuted.clone(),
            tensors.routing_weights.clone(),
            TOP_K as i32,
            tile_n as i32,
            tile_n.trailing_zeros() as i32,
        )
    }
    .generics(vec![tensors.schedule.finalize_tile.to_string()])
    .grid((
        tensors.num_tokens as u32,
        (shape.hidden / tensors.schedule.finalize_tile) as u32,
        1,
    ))
    .compile_options(tensors.schedule.finalize_options());
    launch(target, op)
}

fn run_pipeline(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    run_routing(target, tensors)?;
    run_quantize(target, tensors)?;
    run_fc1(target, tensors)?;
    run_fc2(target, tensors)?;
    run_finalize(target, tensors)
}

fn check_output(
    tensors: &MoeTensors,
    stream: &Arc<Stream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let shape = tensors.shape;
    let output = tensors.output.clone().to_host_vec().sync_on(stream)?;
    let mut max_abs_error = 0.0f32;
    let mut max_reference = 1.0f32;
    for (index, actual) in output.iter().enumerate() {
        let token = index / shape.hidden;
        let actual = actual.to_f32();
        if !actual.is_finite() {
            return Err(format!(
                "output check failed: non-finite output at index {index}: {actual}"
            )
            .into());
        }
        let reference = expected_output(token).to_f32();
        max_abs_error = max_abs_error.max((actual - reference).abs());
        max_reference = max_reference.max(reference.abs());
    }
    let normalized_error = max_abs_error / max_reference;
    if normalized_error > 0.01 {
        return Err(format!(
            "output check failed: normalized max error {normalized_error:.6}, max abs error {max_abs_error:.6}, max |reference| {max_reference:.6}"
        )
        .into());
    }
    println!(
        "Output check passed: normalized max error={normalized_error:.6}, max_abs_error={max_abs_error:.6}"
    );
    Ok(())
}

/// Mean time per graph replay in microseconds, excluding warmup, compilation,
/// allocation, and output checking. The events use the graph's launch stream.
fn time_graph(
    graph: &CudaGraph<()>,
    stream: &Arc<Stream>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start = stream.context().new_event()?;
    let stop = stream.context().new_event()?;
    let timing = (|| -> Result<f64, Box<dyn std::error::Error>> {
        // SAFETY: the graph retains its buffers and remains alive throughout
        // these same-stream launches. The stream is drained below on success
        // and error before the caller can release the graph or its tensors.
        for _ in 0..TIMING_WARMUPS {
            unsafe { graph.launch().async_on(stream)? };
        }
        start.record(stream)?;
        for _ in 0..TIMING_REPLAYS {
            unsafe { graph.launch().async_on(stream)? };
        }
        stop.record(stream)?;
        stop.synchronize()?;
        let latency_us = f64::from(start.elapsed_time(&stop)?) * 1000.0 / TIMING_REPLAYS as f64;
        if !latency_us.is_finite() || latency_us <= 0.0 {
            return Err("invalid CUDA event timing".into());
        }
        Ok(latency_us)
    })();
    // SAFETY: all work submitted here uses this live stream. Drain it even
    // when an enqueue or event operation fails after earlier submissions.
    stream.synchronize()?;
    timing
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(config) = Config::parse(&args)? else {
        return Ok(());
    };
    if !cutile_examples::requirements::BLOCK_SCALED_MMA.check("nvfp4_moe", 0)? {
        return Ok(());
    }

    let device = Context::new(0)?;
    if config.tuned && device.compute_capability()? != (10, 0) {
        return Err("the b200 profile requires an SM100 device".into());
    }
    let stream = device.new_stream()?;
    println!("NVFP4 MoE batch-size benchmark");
    println!(
        "H={}, I={}, experts={}, top-k={TOP_K}, profile={}",
        config.shape.hidden,
        config.shape.intermediate,
        config.shape.experts,
        if config.tuned { "b200" } else { "default" }
    );
    if config.check_only {
        println!("Correctness only: direct launches and graph replay; timing disabled");
    } else {
        println!(
            "CUDA graph timing: {TIMING_SAMPLES} samples, {TIMING_REPLAYS} replays/sample, \
             {TIMING_WARMUPS} warmups/sample"
        );
    }

    // Batches through four use pairwise routing; larger batches use histogram
    // routing. Sweep in order so a failure identifies the first failing size.
    let mut latencies = Vec::with_capacity(BATCH_SIZES.len());
    for &num_tokens in &config.batches {
        println!("Batch {num_tokens}:");
        let tensors = allocate_tensors(num_tokens, &config, &stream)?;
        println!("Schedule: {:?}", tensors.schedule);
        // SAFETY: the fixture and launch geometry keep every unchecked access
        // within its allocation. Each stage owns its writes, shares read-only
        // inputs, and completes before the next stage consumes its outputs.
        // Tensors stay alive through graph replay and the final synchronization.
        run_pipeline(LaunchTarget::Stream(&stream), &tensors)?;
        println!("Batch {num_tokens}, direct launches:");
        check_output(&tensors, &stream)?;

        let graph = CudaGraph::scope(&stream, |scope| {
            run_pipeline(LaunchTarget::Graph(scope), &tensors)
        })?;
        graph.launch().sync_on(&stream)?;
        println!("Batch {num_tokens}, graph replay:");
        check_output(&tensors, &stream)?;

        if config.check_only {
            continue;
        }
        let mut samples_us = Vec::with_capacity(TIMING_SAMPLES);
        for sample in 0..TIMING_SAMPLES {
            let latency_us = time_graph(&graph, &stream)?;
            println!("sample_csv,{num_tokens},{sample},{latency_us:.6}");
            samples_us.push(latency_us);
        }
        check_output(&tensors, &stream)?;
        let mean_us = samples_us.iter().sum::<f64>() / TIMING_SAMPLES as f64;
        println!("Batch {num_tokens}: mean graph latency = {mean_us:.3} us");
        latencies.push((num_tokens, mean_us));
    }
    if !latencies.is_empty() {
        println!("\nBatch size | Mean graph latency (us)");
    }
    for (batch, mean_us) in latencies {
        println!("{batch:>10} | {mean_us:>23.3}");
    }
    println!("NVFP4 MoE passed.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Config>, Box<dyn std::error::Error>> {
        Config::parse(&args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn default_fixture_and_profile_are_unchanged() {
        let config = parse(&[]).unwrap().unwrap();
        assert_eq!(config.shape, SMALL_SHAPE);
        assert!(!config.tuned);
        assert!(!config.check_only);
        assert_eq!(config.batches, BATCH_SIZES);
        let schedule = Schedule::new(256, 148, config.tuned, config.shape);
        assert!(!schedule.tuned);
        assert_eq!(schedule.tile_n, TILE_N);
        assert_eq!(
            schedule.fc1_workers,
            SMALL_SHAPE.max_num_groups(256 * TOP_K, TILE_N)
        );
        assert_eq!(schedule.finalize_tile, FINALIZE_TILE);
    }

    #[test]
    fn explicit_large_fixture_and_focused_check() {
        let config = parse(&[
            "--shape",
            "large",
            "--profile",
            "b200",
            "--batch",
            "16",
            "--check-only",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(config.shape, LARGE_SHAPE);
        assert!(config.tuned);
        assert!(config.check_only);
        assert_eq!(config.batches, [16]);
    }

    #[test]
    fn invalid_options_are_rejected() {
        for args in [
            vec!["--shape", "unknown"],
            vec!["--profile", "unknown"],
            vec!["--batch", "0"],
            vec!["--batch", "3"],
            vec!["--batch", "512"],
            vec!["--batch"],
            vec!["--unknown", "1"],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn b200_schedules_cover_every_batch_with_bounded_workers() {
        for shape in [SMALL_SHAPE, LARGE_SHAPE] {
            for batch in BATCH_SIZES {
                let schedule = Schedule::new(batch, 148, true, shape);
                let capacity = shape.max_num_groups(batch * TOP_K, schedule.tile_n);
                assert!(schedule.tuned);
                assert!((1..=capacity).contains(&schedule.fc1_workers));
                assert!((1..=capacity).contains(&schedule.fc2_workers));
                assert!(shape.hidden.is_multiple_of(schedule.finalize_tile));
                let cluster = if schedule.tile_n == 32 { 2 } else { 1 };
                assert!((shape.intermediate / TILE_M).is_multiple_of(cluster));
                assert!((shape.hidden / TILE_M).is_multiple_of(cluster));
            }
        }
    }

    #[test]
    fn large_shape_keeps_the_evaluated_schedule() {
        let small = Schedule::new(256, 148, true, SMALL_SHAPE);
        let large = Schedule::new(256, 148, true, LARGE_SHAPE);
        assert_eq!(
            (
                large.tile_n,
                large.fc1_workers,
                large.fc2_workers,
                large.finalize_tile
            ),
            (64, 12, 3, 2048)
        );
        assert_eq!(small.finalize_tile, 1024);
    }
}
