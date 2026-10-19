/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * CUDA graphs example — scoped capture with borrowed buffers.
 *
 * Demonstrates scope(...).graph(stream), which captures a CUDA graph
 * using an imperative closure. Each s.record(op) records immediately,
 * releasing borrows — so the same buffer can be written by one kernel
 * and read by the next.
 *
 * No Arc, no try_partition, no take_output, no SharedDeviceOp.
 *
 * Usage:
 *   cargo run -p cutile-examples --example cuda_graphs
 */

use cuda_core::{Device, Stream};
use cutile::error::Error;
use cutile::prelude::*;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// Kernels
// ═══════════════════════════════════════════════════════════════════════════════

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn rms_norm<const D: i32, const BS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        x: &Tensor<f32, { [-1, D] }>,
        w: &Tensor<f32, { [D] }>,
        eps: f32,
    ) {
        let shape: Shape<{ [1, BS] }> = shape![1, BS];
        let tiles: i32 = D / BS;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let x_part: Partition<f32, { [1, BS] }> = x.partition(shape);
        let mut rms: Tile<f32, { [1, BS] }> = constant(0.0, shape);
        for j in 0i32..tiles {
            let t: Tile<f32, { [1, BS] }> = x_part.load([row, j]);
            rms = rms + t * t;
        }
        let s: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let s: f32 = tile_to_scalar(s.reshape(shape![]));
        let n: f32 = convert_scalar(D);
        let inv: f32 = 1.0f32 / (s / n + eps);
        let inv_tile: Tile<f32, { [] }> =
            sqrt(scalar_to_tile(inv), rounding::NegativeInf, ftz::Disabled);
        let inv: f32 = tile_to_scalar(inv_tile);
        let scale: Tile<f32, { [1, BS] }> = inv.broadcast(shape);
        let w_part: Partition<f32, { [BS] }> = w.partition(shape![BS]);
        let mut out_part: PartitionMut<f32, { [1, BS] }> = out.partition_mut(shape);
        for j in 0i32..tiles {
            let t: Tile<f32, { [1, BS] }> = x_part.load([row, j]);
            let tw: Tile<f32, { [1, BS] }> = w_part.load([j]).reshape(shape);
            out_part.store(t * scale * tw, [0i32, j]);
        }
    }

    #[cutile::entry()]
    pub fn matvec<const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let x_part = x.partition(shape![1, BK]);
        let w_part = w.partition(shape![BN, BK]);
        let mut acc = out.load().reshape(shape![BN, 1]);
        for k in 0i32..(K / BK) {
            let tx = x_part.load([0i32, k]).reshape(shape![1, BK]);
            let tw = w_part.load([pid.0, k]);
            let txt: Tile<f32, { [BK, 1] }> = tx.transpose();
            acc = mma(tw, txt, acc);
        }
        out.store(acc.reshape(shape![BN]));
    }

    #[cutile::entry()]
    pub fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let ta: Tile<f32, { [B] }> = a.load_like(out);
        let tb: Tile<f32, { [B] }> = b.load_like(out);
        out.store(ta + tb);
    }
}

use crate::kernels::{add, matvec, rms_norm};

// ═══════════════════════════════════════════════════════════════════════════════
// Model
// ═══════════════════════════════════════════════════════════════════════════════

struct Config {
    d: usize,
    n_layers: usize,
    block: usize,
    bn: usize,
    bk: usize,
    eps: f32,
}

impl Config {
    fn rms_generics(&self) -> Vec<String> {
        vec![self.d.to_string(), self.block.to_string()]
    }
    fn mv_generics(&self) -> Vec<String> {
        vec![self.bn.to_string(), self.bk.to_string(), self.d.to_string()]
    }
}

struct LayerWeights {
    norm_w: Tensor<f32>,
    wq: Tensor<f32>,
    wo: Tensor<f32>,
}

/// Pre-allocated per-layer buffers. Plain Tensors — no Arc needed.
struct LayerBuffers {
    norm: Tensor<f32>,     // (1, d)
    q: Tensor<f32>,        // (d,)
    o: Tensor<f32>,        // (d,)
    residual: Tensor<f32>, // (d,)
}

impl LayerBuffers {
    fn allocate(d: usize, stream: &Arc<Stream>) -> Result<Self, Error> {
        Ok(Self {
            norm: api::zeros(&[1, d]).sync_on(stream)?,
            q: api::zeros(&[d]).sync_on(stream)?,
            o: api::zeros(&[d]).sync_on(stream)?,
            residual: api::zeros(&[d]).sync_on(stream)?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graph model — scoped capture
// ═══════════════════════════════════════════════════════════════════════════════

struct GraphModel {
    graph: CudaGraph<()>,
    input: Tensor<f32>, // (d,) — copy new embedding here before launch
    buffers: Vec<LayerBuffers>,
}

impl GraphModel {
    fn new(cfg: &Config, weights: &[LayerWeights], stream: &Arc<Stream>) -> Result<Self, Error> {
        let mut input: Tensor<f32> = api::ones::<f32>(&[cfg.d]).sync_on(stream)?;
        let mut buffers: Vec<_> = (0..cfg.n_layers)
            .map(|_| LayerBuffers::allocate(cfg.d, stream))
            .collect::<Result<_, _>>()?;
        unsafe { stream.synchronize() }?;

        // Capture the forward pass as a CUDA graph. Each s.record()
        // records a graph node, releasing borrows between kernels.
        let graph = CudaGraph::scope(stream, |s| {
            for (w, bufs) in weights.iter().zip(buffers.iter_mut()) {
                // Create a (1,d) view of input for rms_norm. The view borrows
                // input — after record consumes the op, the borrow is released.
                let hidden_2d = input.view(&[1, cfg.d])?;

                // RMSNorm: hidden(1,d) → norm(1,d)
                s.record(
                    rms_norm(
                        (&mut bufs.norm).partition([1, cfg.d]),
                        &hidden_2d,
                        &w.norm_w,
                        cfg.eps,
                    )
                    .generics(cfg.rms_generics()),
                )?;
                // hidden_2d dropped — input no longer borrowed.

                // Q projection: norm(1,d) @ wq^T → q(d,)
                s.record(
                    matvec((&mut bufs.q).partition([cfg.bn]), &bufs.norm, &w.wq)
                        .generics(cfg.mv_generics()),
                )?;

                // O projection: q(1,d) @ wo^T → o(d,)
                let q_2d = bufs.q.view(&[1, cfg.d])?;
                s.record(
                    matvec((&mut bufs.o).partition([cfg.bn]), &q_2d, &w.wo)
                        .generics(cfg.mv_generics()),
                )?;

                // Residual: hidden(d,) + o(d,) → residual(d,)
                s.record(add(
                    (&mut bufs.residual).partition([cfg.block]),
                    &input,
                    &bufs.o,
                ))?;

                // Copy residual into input for the next layer.
                s.record(api::memcpy(&mut input, &bufs.residual))?;
            }
            Ok(())
        })?;

        Ok(Self {
            graph,
            input,
            buffers,
        })
    }

    fn output(&self) -> &Tensor<f32> {
        &self.buffers.last().unwrap().residual
    }

    fn forward(&mut self, embedding: &Tensor<f32>) -> Result<(), DeviceError> {
        self.graph.update(api::memcpy(&mut self.input, embedding))?;
        self.graph.launch().sync_on(self.graph.stream())?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Eager forward — same kernels, no graph, for validation
// ═══════════════════════════════════════════════════════════════════════════════

fn eager_forward(
    cfg: &Config,
    weights: &[LayerWeights],
    input: &Tensor<f32>,
    stream: &Arc<Stream>,
) -> Result<Vec<f32>, Error> {
    let mut buffers: Vec<_> = (0..cfg.n_layers)
        .map(|_| LayerBuffers::allocate(cfg.d, stream))
        .collect::<Result<_, _>>()?;

    let mut input_buf = input.dup().sync_on(stream)?;

    for (w, bufs) in weights.iter().zip(buffers.iter_mut()) {
        let hidden_2d = input_buf.view(&[1, cfg.d])?;
        rms_norm(
            (&mut bufs.norm).partition([1, cfg.d]),
            &hidden_2d,
            &w.norm_w,
            cfg.eps,
        )
        .generics(cfg.rms_generics())
        .sync_on(stream)?;

        matvec((&mut bufs.q).partition([cfg.bn]), &bufs.norm, &w.wq)
            .generics(cfg.mv_generics())
            .sync_on(stream)?;

        let q_2d = bufs.q.view(&[1, cfg.d])?;
        matvec((&mut bufs.o).partition([cfg.bn]), &q_2d, &w.wo)
            .generics(cfg.mv_generics())
            .sync_on(stream)?;

        add(
            (&mut bufs.residual).partition([cfg.block]),
            &input_buf,
            &bufs.o,
        )
        .sync_on(stream)?;

        api::memcpy(&mut input_buf, &bufs.residual).sync_on(stream)?;
    }

    let host: Vec<f32> = buffers
        .last()
        .unwrap()
        .residual
        .dup()
        .to_host_vec()
        .sync_on(stream)?;
    Ok(host)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("cuda_graphs", 0)? {
        return Ok(());
    }

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let cfg = Config {
        d: 2048,
        n_layers: 22,
        block: 128,
        bn: 16,
        bk: 16,
        eps: 1e-5,
    };

    println!("Allocating model: d={}, layers={}", cfg.d, cfg.n_layers);

    // Deterministic weights for validation.
    let weights: Vec<LayerWeights> = (0..cfg.n_layers)
        .map(|_| {
            Ok(LayerWeights {
                norm_w: api::ones::<f32>(&[cfg.d]).sync_on(&stream)?,
                wq: api::ones::<f32>(&[cfg.d, cfg.d]).sync_on(&stream)?,
                wo: api::ones::<f32>(&[cfg.d, cfg.d]).sync_on(&stream)?,
            })
        })
        .collect::<Result<_, Error>>()?;

    let mut model = GraphModel::new(&cfg, &weights, &stream)?;
    println!("Graph captured.");

    // ── Validation ──────────────────────────────────────────────────────────

    let test_input = api::ones::<f32>(&[cfg.d]).sync_on(&stream)?;

    // Graph path.
    model.forward(&test_input)?;
    let graph_output: Vec<f32> = model
        .output()
        .dup()
        .to_host_vec()
        .sync_on(model.graph.stream())?;

    // Eager path.
    let eager_output = eager_forward(&cfg, &weights, &test_input, &stream)?;

    // With deterministic weights (ones), results should be identical.
    let max_diff = graph_output
        .iter()
        .zip(eager_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Validation: max abs diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-2,
        "Graph output diverged from eager: max_diff={max_diff}"
    );
    println!("Validation passed.");

    // ── Timing ──────────────────────────────────────────────────────────────

    let n_tokens = 32;
    let start = Instant::now();
    for _ in 0..n_tokens {
        model.forward(&test_input)?;
    }
    let elapsed = start.elapsed();
    let tps = n_tokens as f64 / elapsed.as_secs_f64();
    println!(
        "{n_tokens} tokens in {:.2}s = {tps:.1} tok/s",
        elapsed.as_secs_f64()
    );

    Ok(())
}
