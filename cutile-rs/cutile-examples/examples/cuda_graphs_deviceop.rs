/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * CUDA graphs example — safe device-operation combinators.
 *
 * Demonstrates how to use the DeviceOp combinator API (zip!, then,
 * unzip, shared, etc.) to build a multi-kernel forward pass and capture it
 * into a CUDA graph for efficient replay.
 *
 * Usage:
 *   cargo run -p cutile-examples --example cuda_graphs_deviceop
 */

use cuda_core::{Device, Stream};
use cutile::error::Error;
use cutile::prelude::*;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// Kernels — a minimal neural network layer: RMSNorm → MatVec → Add
// ═══════════════════════════════════════════════════════════════════════════════

#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// RMS normalization: out = rms_norm(x) * w
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

    /// Matrix-vector multiply: out = x @ w^T
    /// x: (1, K), w: (N, K), out: (N,)
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

    /// Element-wise add: out = a + b
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
// Model definition
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
    norm_w: Arc<Tensor<f32>>, // (D,)
    wq: Arc<Tensor<f32>>,     // (D, D)
    wo: Arc<Tensor<f32>>,     // (D, D)
}

/// Pre-allocated per-layer intermediate buffers (reused every token).
struct LayerBuffers {
    norm: Arc<Tensor<f32>>,     // (1, d) — output of rms_norm
    q: Arc<Tensor<f32>>,        // (d,)   — output of Q matvec
    o: Arc<Tensor<f32>>,        // (d,)   — output of O matvec
    residual: Arc<Tensor<f32>>, // (d,)   — output of residual add
}

impl LayerBuffers {
    fn allocate(d: usize, stream: &Arc<Stream>) -> Result<Self, Error> {
        Ok(Self {
            norm: api::zeros(&[1, d]).sync_on(stream)?.into(),
            q: api::zeros(&[d]).sync_on(stream)?.into(),
            o: api::zeros(&[d]).sync_on(stream)?.into(),
            residual: api::zeros(&[d]).sync_on(stream)?.into(),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graph-backed model
// ═══════════════════════════════════════════════════════════════════════════════

/// A model whose forward pass is captured as a CUDA graph.
///
/// Construction builds the lazy operation graph and captures it. After that,
/// [`forward`] copies a new embedding into the baked-in input buffer and
/// replays the graph. The output is read directly from the last layer's
/// residual buffer.
struct GraphModel {
    graph: CudaGraph<()>,
    input: Tensor<f32>,
    output: Arc<Tensor<f32>>,
}

impl GraphModel {
    fn new(cfg: &Config, weights: &[LayerWeights], stream: &Arc<Stream>) -> Result<Self, Error> {
        let input: Tensor<f32> = api::rand([cfg.d], None).sync_on(stream)?;
        let buffers: Vec<_> = (0..cfg.n_layers)
            .map(|_| LayerBuffers::allocate(cfg.d, stream))
            .collect::<Result<_, _>>()?;
        unsafe { stream.synchronize() }?;

        // Create an Arc that shares input's device memory for build_forward.
        // This requires aliased storage (input + input_arc point to same
        // memory). The scope-based cuda_graphs.rs example avoids this
        // entirely with TensorView + safe memcpy.
        let input_arc: Arc<Tensor<f32>> = unsafe { input.into_shared_alias() };

        // build_forward takes buffers by value — Arc refcounts drop to 1
        // so try_partition succeeds. Returns a SharedDeviceOp pointing to
        // the last residual buffer.
        let (forward_op, output_shared) = Self::build_forward(cfg, weights, &input_arc, buffers);
        let graph = forward_op.graph_on(stream.clone())?;

        // The graph executed the op once during capture. The SharedDeviceOp
        // cached the result — sync it to get the output Arc.
        let output: Arc<Tensor<f32>> = output_shared.sync_on(stream)?;

        Ok(Self {
            graph,
            input,
            output,
        })
    }

    /// Returns the output tensor. The graph writes into this device memory
    /// on each `forward()` call.
    fn output(&self) -> &Arc<Tensor<f32>> {
        &self.output
    }

    /// Returns the stream the graph was captured on.
    fn stream(&self) -> &Arc<Stream> {
        self.graph.stream()
    }

    /// Copy a new embedding into the input buffer and replay the graph.
    fn forward(&mut self, embedding: &Tensor<f32>) -> Result<(), DeviceError> {
        self.graph.update(api::memcpy(&mut self.input, embedding))?;
        self.graph.launch().sync_on(self.graph.stream())?;
        Ok(())
    }

    /// Build the lazy forward pass.
    ///
    /// Takes `buffers` by value so each `Arc<Tensor>` has refcount 1,
    /// allowing `try_partition` to succeed. The graph captures the Arcs
    /// via `SharedDeviceOp`, keeping the device memory alive for replay.
    fn build_forward(
        cfg: &Config,
        weights: &[LayerWeights],
        input: &Arc<Tensor<f32>>,
        buffers: Vec<LayerBuffers>,
    ) -> (impl DeviceOp<Output = ()>, SharedDeviceOp<Tensor<f32>>) {
        let mut hidden: SharedDeviceOp<Tensor<f32>> =
            cuda_async::device_operation::shared(input.clone());

        let mut ops: Vec<BoxedDeviceOp<()>> = Vec::with_capacity(buffers.len());

        for (w, bufs) in weights.iter().zip(buffers) {
            // RMSNorm: hidden(1,d) × norm_w → norm(1,d)
            let norm = rms_norm(
                bufs.norm
                    .try_partition([1, cfg.d])
                    .expect("sole buffer owner"),
                hidden.clone().reshape(&[1, cfg.d]),
                w.norm_w.clone(),
                cfg.eps,
            )
            .generics(cfg.rms_generics())
            .first()
            .unpartition()
            .shared();

            // Q projection: norm @ wq^T → q
            let q = matvec(
                bufs.q.try_partition([cfg.bn]).expect("sole buffer owner"),
                norm.clone(),
                w.wq.clone(),
            )
            .generics(cfg.mv_generics())
            .first()
            .unpartition()
            .shared();

            // O projection: q @ wo^T → o
            let o = matvec(
                bufs.o.try_partition([cfg.bn]).expect("sole buffer owner"),
                q.clone().reshape(&[1, cfg.d]),
                w.wo.clone(),
            )
            .generics(cfg.mv_generics())
            .first()
            .unpartition()
            .shared();

            // Residual add: hidden + o → residual
            let residual = add(
                bufs.residual
                    .try_partition([cfg.block])
                    .expect("sole buffer owner"),
                hidden.clone().reshape(&[cfg.d]),
                o.clone(),
            )
            .first()
            .unpartition()
            .shared();

            hidden = residual.clone();

            // Discard the output — buffers are written in place.
            ops.push(zip!(norm, q, o, residual).map(|_| ()).boxed());
        }

        (DeviceOpVec::new(ops).map(|_| ()), hidden)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Warmup — compile all kernels once so JIT cost is excluded from benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

fn warmup(cfg: &Config, weights: &[LayerWeights], stream: &Arc<Stream>) -> Result<(), Error> {
    println!("Warming up (compiling kernels)...");
    let h: Arc<Tensor<f32>> = api::rand([cfg.d], None).sync_on(stream)?.into();
    let h_2d: Arc<Tensor<f32>> = h
        .dup()
        .sync_on(stream)?
        .reshape(&[1, cfg.d])
        .unwrap()
        .into();
    let w = &weights[0];

    let norm_out: Partition<Tensor<f32>> = api::zeros(&[1, cfg.d])
        .sync_on(stream)?
        .partition([1, cfg.d]);
    let _ = rms_norm(norm_out, h_2d.clone(), w.norm_w.clone(), cfg.eps)
        .generics(cfg.rms_generics())
        .sync_on(stream)?;

    let q_out: Partition<Tensor<f32>> = api::zeros(&[cfg.d]).sync_on(stream)?.partition([cfg.bn]);
    let _ = matvec(q_out, h_2d.clone(), w.wq.clone())
        .generics(cfg.mv_generics())
        .sync_on(stream)?;

    let add_out: Partition<Tensor<f32>> =
        api::zeros(&[cfg.d]).sync_on(stream)?.partition([cfg.block]);
    let _ = add(add_out, h.clone(), h.clone()).sync_on(stream)?;

    println!("Warmup done.");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Eager forward pass — run the same kernels without a graph, for validation
// ═══════════════════════════════════════════════════════════════════════════════

fn eager_forward(
    cfg: &Config,
    weights: &[LayerWeights],
    input: &Arc<Tensor<f32>>,
    stream: &Arc<Stream>,
) -> Result<Vec<f32>, Error> {
    let buffers: Vec<_> = (0..cfg.n_layers)
        .map(|_| LayerBuffers::allocate(cfg.d, stream))
        .collect::<Result<_, _>>()?;

    let (forward_op, output_shared) = GraphModel::build_forward(cfg, weights, input, buffers);
    forward_op.sync_on(stream)?;

    let output: Arc<Tensor<f32>> = output_shared.sync_on(stream)?;
    let host: Vec<f32> = output.dup().to_host_vec().sync_on(stream)?;
    Ok(host)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<(), Error> {
    if !cutile_examples::requirements::BASELINE.check("cuda_graphs_deviceop", 0)? {
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
    let n_tokens = 32;

    println!(
        "Allocating fake model: d={}, layers={}",
        cfg.d, cfg.n_layers
    );

    // Deterministic weights for validation — ones produce exact results
    // so the graph vs eager comparison catches pointer/memory bugs, not
    // floating-point non-determinism.
    let weights: Vec<LayerWeights> = (0..cfg.n_layers)
        .map(|_| {
            Ok(LayerWeights {
                norm_w: api::ones::<f32>(&[cfg.d]).sync_on(&stream)?.into(),
                wq: api::ones::<f32>(&[cfg.d, cfg.d]).sync_on(&stream)?.into(),
                wo: api::ones::<f32>(&[cfg.d, cfg.d]).sync_on(&stream)?.into(),
            })
        })
        .collect::<Result<_, Error>>()?;

    // Warmup: compile all kernels so JIT cost is excluded.
    warmup(&cfg, &weights, &stream)?;

    // Build and capture the forward pass as a CUDA graph.
    let mut model = GraphModel::new(&cfg, &weights, &stream)?;
    println!("Graph captured.");

    // ── Validation ──────────────────────────────────────────────────────────
    // Run with a known input, compare graph output against eager execution.

    let test_input: Tensor<f32> = api::ones::<f32>(&[cfg.d]).sync_on(&stream)?;

    // Graph path.
    model.forward(&test_input)?;
    let graph_output: Vec<f32> = model.output().dup().to_host_vec().sync_on(model.stream())?;

    // Eager path (same kernels, no graph).
    let test_input_arc: Arc<Tensor<f32>> = test_input.into();
    let eager_output = eager_forward(&cfg, &weights, &test_input_arc, &stream)?;

    // With deterministic weights (ones), graph and eager should produce
    // identical results. Any diff indicates a pointer/memory bug.
    let max_diff = graph_output
        .iter()
        .zip(eager_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Validation: max abs diff between graph and eager = {max_diff:.2e}");
    assert!(
        max_diff < 1e-2,
        "Graph output diverged from eager: max_diff={max_diff}"
    );
    println!("Validation passed.");

    // ── Timing ──────────────────────────────────────────────────────────────

    let embeddings: Vec<Tensor<f32>> = (0..n_tokens)
        .map(|i| -> Result<_, Error> {
            Ok(api::rand([cfg.d], Some(1000 + i as u64)).sync_on(&stream)?)
        })
        .collect::<Result<_, _>>()?;

    let start = Instant::now();
    for embedding in &embeddings {
        model.forward(embedding)?;
    }
    let elapsed = start.elapsed();

    let tps = n_tokens as f64 / elapsed.as_secs_f64();
    println!(
        "{n_tokens} tokens in {:.2}s = {tps:.1} tok/s",
        elapsed.as_secs_f64()
    );

    Ok(())
}
