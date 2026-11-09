/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Uses cutile-kernels' NVFP4 mixture of experts kernels by Emmet Zhang et al.
 * Peak kernel performance requires the CUDA Toolkit 13.5 Tile IR assembler.
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
 * admit an analytical reference. Performance will be optimized in a future
 * version.
 */

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cuda_core::{Context, Stream};
use cutile::api;
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::{bf16, f16};
use cutile::prelude::*;
use std::sync::Arc;

use cutile_kernels::experimental::nvfp4_moe as kernels;

use kernels::{
    finalize_kernel, grouped_fc1_kernel, grouped_fc2_kernel, nvfp4_quantize_kernel,
    routing_group_histogram_kernel, routing_group_pairwise_kernel,
};

// Kernel config
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 1024;
const NUM_EXPERTS: usize = 8;
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

fn max_num_groups(routed_size: usize, tile_n: usize) -> usize {
    let filled_experts = NUM_EXPERTS.min(routed_size);
    filled_experts + (routed_size - filled_experts) / tile_n
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

struct MoeTensors {
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
    stream: &Arc<Stream>,
) -> Result<MoeTensors, Box<dyn std::error::Error>> {
    let routed_size = num_tokens * TOP_K;
    let max_groups = max_num_groups(routed_size, TILE_N);
    let hidden_host = (0..num_tokens)
        .flat_map(|token| std::iter::repeat_n(f16::from_f32(hidden_value(token)), HIDDEN))
        .collect::<Vec<_>>();
    let hidden: Arc<Tensor<f16>> = api::copy_host_vec_to_device(&Arc::new(hidden_host))
        .reshape(&[num_tokens, HIDDEN])
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
    let w_up: Arc<Tensor<f4e2m1fnx2>> =
        api::full(fp4_one, &[NUM_EXPERTS * INTERMEDIATE, HIDDEN / 2])
            .sync_on(stream)?
            .into();
    let w_up_scales: Arc<Tensor<f8e4m3fn>> = api::full(
        scale_one,
        &[NUM_EXPERTS * INTERMEDIATE / 128, HIDDEN / 64, 2, 256],
    )
    .sync_on(stream)?
    .into();
    let mut w_up_global_host = vec![1.0f32 / HIDDEN as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_up_global_host[expert] = UP_FACTORS[expert] / HIDDEN as f32;
    }
    let w_up_global: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(w_up_global_host))
        .reshape(&[NUM_EXPERTS])
        .sync_on(stream)?
        .into();

    let w_down: Arc<Tensor<f4e2m1fnx2>> =
        api::full(fp4_one, &[NUM_EXPERTS * HIDDEN, INTERMEDIATE / 2])
            .sync_on(stream)?
            .into();
    let w_down_scales: Arc<Tensor<f8e4m3fn>> = api::full(
        scale_one,
        &[NUM_EXPERTS * HIDDEN / 128, INTERMEDIATE / 64 * 512],
    )
    .sync_on(stream)?
    .into();
    let mut w_down_global_host = vec![1.0f32 / INTERMEDIATE as f32; NUM_EXPERTS];
    for expert in 0..TOP_K {
        w_down_global_host[expert] = DOWN_FACTORS[expert] / INTERMEDIATE as f32;
    }
    let w_down_global: Arc<Tensor<f32>> =
        api::copy_host_vec_to_device(&Arc::new(w_down_global_host))
            .reshape(&[NUM_EXPERTS])
            .sync_on(stream)?
            .into();

    let group_tokens = api::zeros::<i32>(&[max_groups, TILE_N])
        .sync_on(stream)?
        .into();
    let group_experts = api::zeros::<i32>(&[max_groups]).sync_on(stream)?.into();
    let group_limits = api::zeros::<i32>(&[max_groups]).sync_on(stream)?.into();
    let expanded_to_permuted = api::zeros::<i32>(&[num_tokens, TOP_K])
        .sync_on(stream)?
        .into();
    let expert_counts = api::zeros::<i32>(&[NUM_EXPERTS]).sync_on(stream)?.into();
    let expert_offsets = api::zeros::<i32>(&[NUM_EXPERTS + 1])
        .sync_on(stream)?
        .into();
    let num_groups = api::zeros::<i32>(&[1]).sync_on(stream)?.into();
    let quantized_hidden = api::zeros::<f4e2m1fnx2>(&[num_tokens, HIDDEN / 2])
        .sync_on(stream)?
        .into();
    let quantized_hidden_scales = api::zeros::<f8e4m3fn>(&[num_tokens, HIDDEN / 16])
        .sync_on(stream)?
        .into();
    let grouped_inter = api::zeros::<f4e2m1fnx2>(&[max_groups, TILE_N, INTERMEDIATE / 2])
        .sync_on(stream)?
        .into();
    let grouped_inter_scales = api::zeros::<f8e4m3fn>(&[max_groups, INTERMEDIATE / 16, TILE_N])
        .sync_on(stream)?
        .into();
    let grouped_out = api::zeros::<bf16>(&[max_groups, TILE_N, HIDDEN])
        .sync_on(stream)?
        .into();
    let output = api::zeros::<bf16>(&[num_tokens, HIDDEN])
        .sync_on(stream)?
        .into();

    Ok(MoeTensors {
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
                TILE_N as i32,
                TILE_N.trailing_zeros() as i32,
            )
        }
        .grid((1, 1, 1));
        launch(target, op)?;
    } else {
        let experts_pow2 = NUM_EXPERTS.next_power_of_two();
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
                NUM_EXPERTS as i32,
                routed_size as i32,
                TILE_N as i32,
                TILE_N.trailing_zeros() as i32,
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
        (HIDDEN / QUANTIZE_TILE) as u32,
        1,
    ));
    launch(target, op)
}

fn run_fc1(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let num_output_tiles = INTERMEDIATE / TILE_M;
    let worker_groups = max_num_groups(tensors.num_tokens * TOP_K, TILE_N);
    let op = unsafe {
        grouped_fc1_kernel(
            tensors.grouped_inter.clone(),
            tensors.grouped_inter_scales.clone(),
            tensors.quantized_hidden.clone(),
            tensors.quantized_hidden_scales.clone(),
            (HIDDEN / 2) as i32,
            (HIDDEN / 16) as i32,
            tensors.group_tokens.clone(),
            tensors.group_experts.clone(),
            tensors.group_limits.clone(),
            tensors.num_groups.clone(),
            tensors.w_up.clone(),
            tensors.w_up_scales.clone(),
            tensors.w_up_global.clone(),
            INPUT_SCALE,
            INTERMEDIATE as i32,
            (HIDDEN / TILE_K) as i32,
        )
    }
    .generics(vec![
        TILE_M.to_string(),
        (TILE_M / 32).to_string(),
        TILE_N.to_string(),
        TILE_K.to_string(),
        (TILE_K / 64).to_string(),
        (TILE_K / 2).to_string(),
        (TILE_K / 16).to_string(),
        (TILE_M / 2).to_string(),
        (TILE_M / 16).to_string(),
        "1".to_string(),
        "1".to_string(),
        TILE_N.to_string(),
        (TILE_M / 16).to_string(),
        TILE_N.to_string(),
        (TILE_M / 16).to_string(),
        "16".to_string(),
    ])
    .grid((num_output_tiles as u32, worker_groups as u32, 1));
    launch(target, op)
}

fn run_fc2(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let num_output_tiles = HIDDEN / TILE_M;
    let group_workers = max_num_groups(tensors.num_tokens * TOP_K, TILE_N);
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
            HIDDEN as i32,
            (INTERMEDIATE / TILE_K) as i32,
        )
    }
    .generics(vec![
        TILE_M.to_string(),
        (TILE_M / 32).to_string(),
        TILE_N.to_string(),
        TILE_K.to_string(),
        (TILE_K / 64).to_string(),
        (TILE_K / 2).to_string(),
        (TILE_K / 16).to_string(),
        (TILE_K / 64 * 512).to_string(),
        "1".to_string(),
        "1".to_string(),
        "1".to_string(),
    ])
    .grid((num_output_tiles as u32, group_workers as u32, 1));
    launch(target, op)
}

fn run_finalize(
    target: LaunchTarget<'_>,
    tensors: &MoeTensors,
) -> Result<(), cuda_async::error::DeviceError> {
    let op = unsafe {
        finalize_kernel(
            tensors.output.clone(),
            tensors.grouped_out.clone(),
            tensors.expanded_to_permuted.clone(),
            tensors.routing_weights.clone(),
            TOP_K as i32,
            TILE_N as i32,
            TILE_N.trailing_zeros() as i32,
        )
    }
    .generics(vec![FINALIZE_TILE.to_string()])
    .grid((
        tensors.num_tokens as u32,
        (HIDDEN / FINALIZE_TILE) as u32,
        1,
    ));
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
    let output = tensors.output.clone().to_host_vec().sync_on(stream)?;
    let mut max_abs_error = 0.0f32;
    let mut max_reference = 1.0f32;
    for (index, actual) in output.iter().enumerate() {
        let token = index / HIDDEN;
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
    if !cutile_examples::requirements::BLOCK_SCALED_MMA.check("nvfp4_moe", 0)? {
        return Ok(());
    }

    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    println!("NVFP4 MoE batch-size benchmark");
    println!("H={HIDDEN}, I={INTERMEDIATE}, experts={NUM_EXPERTS}, top-k={TOP_K}, tile_n={TILE_N}");
    println!(
        "CUDA graph timing: {TIMING_SAMPLES} samples, {TIMING_REPLAYS} replays/sample, \
         {TIMING_WARMUPS} warmups/sample"
    );

    // Batches through four use pairwise routing; larger batches use histogram
    // routing. Sweep in order so a failure identifies the first failing size.
    let mut latencies = Vec::with_capacity(BATCH_SIZES.len());
    for num_tokens in BATCH_SIZES {
        println!("Batch {num_tokens}:");
        let tensors = allocate_tensors(num_tokens, &stream)?;
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

        let mut samples_us = Vec::with_capacity(TIMING_SAMPLES);
        for _ in 0..TIMING_SAMPLES {
            samples_us.push(time_graph(&graph, &stream)?);
        }
        check_output(&tensors, &stream)?;
        let mean_us = samples_us.iter().sum::<f64>() / TIMING_SAMPLES as f64;
        println!("Batch {num_tokens}: mean graph latency = {mean_us:.3} us");
        latencies.push((num_tokens, mean_us));
    }
    println!("\nBatch size | Mean graph latency (us)");
    for (batch, mean_us) in latencies {
        println!("{batch:>10} | {mean_us:>23.3}");
    }
    println!("NVFP4 MoE passed.");
    Ok(())
}
