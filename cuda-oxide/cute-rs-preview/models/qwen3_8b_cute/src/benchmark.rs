/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end `Qwen3-8B` BF16 and NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`) engine and benchmark suite.
//!
//! Measures prefill (`tokens/sec`), autoregressive decode (`tokens/sec`), weight memory
//! savings, and numerical cosine similarity between BF16 and NVFP4 on RTX 5090 (`sm_120a`).

use crate::checkpoint::{
    OwnedNvfp4Layer, Qwen3Bf16Stack, Qwen3Nvfp4Stack, load_qwen3_bf16_checkpoint,
    synthesize_qwen3_bf16,
};
use crate::config::{
    EPSILON, HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, KV_WIDTH, MAX_POSITION, Q_HEADS, Q_WIDTH,
    ROPE_THETA,
};
use crate::model::Batch;
use crate::norm_kernel;
use crate::nvfp4::Nvfp4Runner;
use crate::operator::Result;
use crate::rotary::RotarySpec;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cute_rs::Bf16;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Summary metrics for one precision mode (`BF16` or `NVFP4`) on `Qwen3-8B`.
#[derive(Clone, Debug)]
pub struct ModeMetrics {
    /// Precision label (`"BF16"` or `"NVFP4 (E2M1 + E4M3, SF=16)"`).
    pub label: &'static str,
    /// Number of executed decoder layers.
    pub layers: usize,
    /// Total device weight footprint in bytes.
    pub weight_bytes: usize,
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Prefill latency in milliseconds.
    pub prefill_ms: f64,
    /// Prefill throughput in tokens per second.
    pub prefill_tok_per_sec: f64,
    /// Number of autoregressive decode steps measured.
    pub decode_steps: usize,
    /// Mean per-token decode latency in milliseconds.
    pub decode_ms_per_tok: f64,
    /// Autoregressive decode throughput in tokens per second.
    pub decode_tok_per_sec: f64,
}

/// Complete comparison report between `Qwen3-8B` BF16 and NVFP4 execution.
#[derive(Clone, Debug)]
pub struct BenchmarkReport {
    /// Metrics for the BF16 Tensor-Core baseline.
    pub bf16: ModeMetrics,
    /// Metrics for the NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`) block-scaled path.
    pub nvfp4: ModeMetrics,
    /// Cosine similarity between the final normalized hidden states of BF16 and NVFP4.
    pub cosine_similarity: f64,
    /// Compression ratio of total layer weights (`bf16.weight_bytes / nvfp4.weight_bytes`).
    pub compression_ratio: f64,
}

/// Pre-allocated device KV-cache for one NVFP4 `Qwen3-8B` layer (`[capacity, KV_HEADS, HEAD_DIM]`).
pub struct Nvfp4KvCache {
    /// Rotated key cache `[capacity, KV_HEADS, HEAD_DIM]`.
    pub key: DeviceBuffer<Bf16>,
    /// Unrotated value cache `[capacity, KV_HEADS, HEAD_DIM]`.
    pub value: DeviceBuffer<Bf16>,
    /// Current valid token prefix length.
    pub len: usize,
    /// Maximum token capacity.
    pub capacity: usize,
}

impl Nvfp4KvCache {
    /// Allocate a zero-initialized KV cache for `capacity` tokens on `stream`.
    pub fn new(stream: &CudaStream, capacity: usize) -> Result<Self> {
        let elems = capacity * KV_WIDTH + 8;
        let zero = vec![Bf16::from_bits(0); elems];
        Ok(Self {
            key: DeviceBuffer::from_host(stream, &zero)?,
            value: DeviceBuffer::from_host(stream, &zero)?,
            len: 0,
            capacity,
        })
    }
}

/// Pre-allocated device scratch buffers for zero-allocation, stream-ordered 36-layer NVFP4 execution.
struct Nvfp4Scratch {
    hidden_a: Option<DeviceBuffer<Bf16>>,
    hidden_b: Option<DeviceBuffer<Bf16>>,
    normalized: DeviceBuffer<Bf16>,
    q: DeviceBuffer<Bf16>,
    k: DeviceBuffer<Bf16>,
    v: DeviceBuffer<Bf16>,
    qn: DeviceBuffer<Bf16>,
    kn: DeviceBuffer<Bf16>,
    rot_q: DeviceBuffer<Bf16>,
    rot_k: DeviceBuffer<Bf16>,
    partials: DeviceBuffer<f32>,
    attended: DeviceBuffer<Bf16>,
    projected: DeviceBuffer<Bf16>,
    residual: DeviceBuffer<Bf16>,
    post: DeviceBuffer<Bf16>,
    gate: DeviceBuffer<Bf16>,
    up: DeviceBuffer<Bf16>,
    silu: DeviceBuffer<Bf16>,
    product: DeviceBuffer<Bf16>,
    down: DeviceBuffer<Bf16>,
}

impl Nvfp4Scratch {
    fn new(stream: &CudaStream, max_rows: usize, max_seq: usize) -> Result<Self> {
        let splits = max_seq.div_ceil(64);
        let partial_elems = max_rows * Q_HEADS * splits * crate::attention_kernel::STATE_SIZE + 8;
        let z_bf16 = |n: usize| DeviceBuffer::from_host(stream, &vec![Bf16::from_bits(0); n + 8]);
        Ok(Self {
            hidden_a: Some(z_bf16(max_rows * HIDDEN)?),
            hidden_b: Some(z_bf16(max_rows * HIDDEN)?),
            normalized: z_bf16(max_rows * HIDDEN)?,
            q: z_bf16(max_rows * Q_WIDTH)?,
            k: z_bf16(max_rows * KV_WIDTH)?,
            v: z_bf16(max_rows * KV_WIDTH)?,
            qn: z_bf16(max_rows * Q_WIDTH)?,
            kn: z_bf16(max_rows * KV_WIDTH)?,
            rot_q: z_bf16(max_rows * Q_WIDTH)?,
            rot_k: z_bf16(max_rows * KV_WIDTH)?,
            partials: DeviceBuffer::from_host(stream, &vec![0.0f32; partial_elems])?,
            attended: z_bf16(max_rows * Q_WIDTH)?,
            projected: z_bf16(max_rows * HIDDEN)?,
            residual: z_bf16(max_rows * HIDDEN)?,
            post: z_bf16(max_rows * HIDDEN)?,
            gate: z_bf16(max_rows * INTERMEDIATE)?,
            up: z_bf16(max_rows * INTERMEDIATE)?,
            silu: z_bf16(max_rows * INTERMEDIATE)?,
            product: z_bf16(max_rows * INTERMEDIATE)?,
            down: z_bf16(max_rows * HIDDEN)?,
        })
    }
}

/// Combined `Qwen3-8B` execution engine supporting both BF16 and NVFP4 (`E2M1` + `E4M3`) stacks.
pub struct Qwen3Engine {
    /// Shared execution stream.
    stream: Arc<CudaStream>,
    /// NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`) linear projection runner.
    nvfp4_runner: Nvfp4Runner,
    /// Loaded RMSNorm module for stream-ordered execution.
    norms: norm_kernel::kernels::LoadedModule,
    /// Loaded rotary/append module for stream-ordered execution.
    rotary_mod: crate::kernel::kernels::LoadedModule,
    /// Loaded split-KV attention module for stream-ordered execution.
    attn_mod: crate::attention_kernel::kernels::LoadedModule,
    /// Loaded elementwise add module for stream-ordered execution.
    elem_mod: crate::element_kernel::kernels::LoadedModule,
    /// Loaded BF16 MMA & SwiGLU activation module for stream-ordered execution.
    ffn_mod: crate::ffn_kernel::kernels::LoadedModule,
    /// Pre-allocated scratch buffers for zero-sync 36-layer execution.
    scratch: std::sync::Mutex<Nvfp4Scratch>,
    /// Standard 1D RoPE specification (`theta = 1_000_000.0`, `D = 128`).
    rotary: RotarySpec<HEAD_DIM>,
}

/// Make a 128-thread 1D launch configuration for `work` items.
fn grid1d(work: usize) -> cuda_core::simt::LaunchConfig {
    cuda_core::simt::LaunchConfig {
        grid_dim: (work.div_ceil(128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Qwen3Engine {
    /// Load all BF16, NVFP4, RMSNorm, RoPE, and GQA modules onto `stream`.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let ctx = stream.context();
        Ok(Self {
            nvfp4_runner: Nvfp4Runner::new(stream.clone())?,
            norms: norm_kernel::kernels::load(ctx)?,
            rotary_mod: crate::kernel::kernels::load(ctx)?,
            attn_mod: unsafe { crate::attention_kernel::kernels::load(ctx)? },
            elem_mod: crate::element_kernel::kernels::load(ctx)?,
            ffn_mod: unsafe { crate::ffn_kernel::kernels::load(ctx)? },
            scratch: std::sync::Mutex::new(Nvfp4Scratch::new(&stream, 256, 512)?),
            rotary: RotarySpec::standard(ROPE_THETA, MAX_POSITION)?,
            stream,
        })
    }

    /// Borrow the execution stream.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Enqueue one BF16 projection asynchronously on `self.stream`.
    fn enqueue_project_bf16(
        &self,
        input: &DeviceBuffer<Bf16>,
        weight: &DeviceBuffer<Bf16>,
        output: &mut DeviceBuffer<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
    ) -> Result<()> {
        unsafe {
            if rows <= 2 {
                let launch = self.ffn_mod.prepare_decode_linear_bf16(LaunchConfig1D::new(
                    outputs.div_ceil(8) as u32,
                    32,
                    0,
                ))?;
                self.ffn_mod.decode_linear_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight,
                    output,
                    rows,
                    width,
                    outputs,
                    width,
                    width,
                )?;
            } else {
                let tiles = rows.div_ceil(16) * outputs.div_ceil(8);
                let cfg = cuda_core::simt::LaunchConfig {
                    grid_dim: (tiles.div_ceil(4) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                self.ffn_mod.linear_bf16(
                    &self.stream,
                    cfg,
                    input,
                    weight,
                    output,
                    rows,
                    width,
                    outputs,
                    width,
                    width,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue one `Qwen3-8B` NVFP4 decoder layer asynchronously on `self.stream`.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_layer_nvfp4(
        &self,
        hidden_in: &DeviceBuffer<Bf16>,
        hidden_out: &mut DeviceBuffer<Bf16>,
        rows: usize,
        layer: &OwnedNvfp4Layer,
        cache: &mut Nvfp4KvCache,
        cos: &DeviceBuffer<Bf16>,
        sin: &DeviceBuffer<Bf16>,
        s: &mut Nvfp4Scratch,
    ) -> Result<()> {
        let start = cache.len;
        let key_tokens = start + rows;
        let splits = key_tokens.div_ceil(64);
        let heads = rows * Q_HEADS;
        let stream = &self.stream;

        unsafe {
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                hidden_in,
                &layer.input_norm,
                &mut s.normalized,
                rows,
                HIDDEN,
                EPSILON,
            )?;
            self.nvfp4_runner.enqueue_qkv(
                &s.normalized,
                rows,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &mut s.q,
                &mut s.k,
                &mut s.v,
            )?;
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                &s.q,
                &layer.q_norm,
                &mut s.qn,
                rows * Q_HEADS,
                HEAD_DIM,
                EPSILON,
            )?;
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                &s.k,
                &layer.k_norm,
                &mut s.kn,
                rows * KV_HEADS,
                HEAD_DIM,
                EPSILON,
            )?;
            self.rotary_mod.rotate_bf16(
                stream,
                grid1d(rows * (Q_WIDTH / 2)),
                &s.qn,
                cos,
                sin,
                &mut s.rot_q,
                rows,
                Q_HEADS,
                HEAD_DIM,
            )?;
            self.rotary_mod.rotate_bf16(
                stream,
                grid1d(rows * (KV_WIDTH / 2)),
                &s.kn,
                cos,
                sin,
                &mut s.rot_k,
                rows,
                KV_HEADS,
                HEAD_DIM,
            )?;
            self.rotary_mod.append_bf16(
                stream,
                grid1d(rows * KV_WIDTH),
                &s.rot_k,
                &s.v,
                &mut cache.key,
                &mut cache.value,
                start * KV_WIDTH,
                rows * KV_WIDTH,
            )?;
            let partial_launch = self.attn_mod.prepare_partial(LaunchConfig1D::new(
                (heads * splits) as u32,
                128,
                crate::attention_kernel::PARTIAL_SHARED_BYTES,
            ))?;
            self.attn_mod.partial(
                stream,
                &partial_launch,
                &s.rot_q,
                &cache.key,
                &cache.value,
                &mut s.partials,
                rows as u32,
                key_tokens as u32,
                Q_HEADS as u32,
                KV_HEADS as u32,
                start as u32,
                true,
            )?;
            let merge_launch =
                self.attn_mod
                    .prepare_merge(LaunchConfig1D::new(heads as u32, 32, 0))?;
            self.attn_mod.merge(
                stream,
                &merge_launch,
                &s.partials,
                &mut s.attended,
                rows as u32,
                Q_HEADS as u32,
                splits as u32,
            )?;
            self.nvfp4_runner.enqueue_project(
                &s.attended,
                rows,
                &layer.o_proj,
                &mut s.projected,
            )?;
            self.elem_mod.add_bf16(
                stream,
                grid1d(rows * HIDDEN),
                hidden_in,
                &s.projected,
                &mut s.residual,
                rows * HIDDEN,
            )?;
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                &s.residual,
                &layer.post_attention_norm,
                &mut s.post,
                rows,
                HIDDEN,
                EPSILON,
            )?;
            self.nvfp4_runner.enqueue_gate_up(
                &s.post,
                rows,
                &layer.gate_proj,
                &layer.up_proj,
                &mut s.gate,
                &mut s.up,
            )?;
            self.ffn_mod.activate_bf16(
                stream,
                grid1d((rows * INTERMEDIATE).div_ceil(8)),
                &s.gate,
                &s.up,
                &mut s.silu,
                &mut s.product,
                rows * INTERMEDIATE,
            )?;
            self.nvfp4_runner
                .enqueue_project(&s.product, rows, &layer.down_proj, &mut s.down)?;
            self.elem_mod.add_bf16(
                stream,
                grid1d(rows * HIDDEN),
                &s.residual,
                &s.down,
                hidden_out,
                rows * HIDDEN,
            )?;
        }
        cache.len = key_tokens;
        Ok(())
    }

    /// Execute the `Qwen3-8B` BF16 stack over `input` at 1D token `positions`.
    /// Enqueue one `Qwen3-8B` BF16 decoder layer asynchronously on `self.stream`.
    #[allow(clippy::too_many_arguments)]
    fn enqueue_layer_bf16(
        &self,
        hidden_in: &DeviceBuffer<Bf16>,
        hidden_out: &mut DeviceBuffer<Bf16>,
        rows: usize,
        layer: &crate::checkpoint::OwnedBf16Layer,
        cache: &mut Nvfp4KvCache,
        cos: &DeviceBuffer<Bf16>,
        sin: &DeviceBuffer<Bf16>,
        s: &mut Nvfp4Scratch,
    ) -> Result<()> {
        let start = cache.len;
        let key_tokens = start + rows;
        let splits = key_tokens.div_ceil(64);
        let heads = rows * Q_HEADS;
        let stream = &self.stream;

        unsafe {
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                hidden_in,
                &layer.input_norm,
                &mut s.normalized,
                rows,
                HIDDEN,
                EPSILON,
            )?;
            self.enqueue_project_bf16(&s.normalized, &layer.q_proj, &mut s.q, rows, HIDDEN, Q_WIDTH)?;
            self.enqueue_project_bf16(&s.normalized, &layer.k_proj, &mut s.k, rows, HIDDEN, KV_WIDTH)?;
            self.enqueue_project_bf16(&s.normalized, &layer.v_proj, &mut s.v, rows, HIDDEN, KV_WIDTH)?;
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                &s.q,
                &layer.q_norm,
                &mut s.qn,
                rows * Q_HEADS,
                HEAD_DIM,
                EPSILON,
            )?;
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                &s.k,
                &layer.k_norm,
                &mut s.kn,
                rows * KV_HEADS,
                HEAD_DIM,
                EPSILON,
            )?;
            self.rotary_mod.rotate_bf16(
                stream,
                grid1d(rows * (Q_WIDTH / 2)),
                &s.qn,
                cos,
                sin,
                &mut s.rot_q,
                rows,
                Q_HEADS,
                HEAD_DIM,
            )?;
            self.rotary_mod.rotate_bf16(
                stream,
                grid1d(rows * (KV_WIDTH / 2)),
                &s.kn,
                cos,
                sin,
                &mut s.rot_k,
                rows,
                KV_HEADS,
                HEAD_DIM,
            )?;
            self.rotary_mod.append_bf16(
                stream,
                grid1d(rows * KV_WIDTH),
                &s.rot_k,
                &s.v,
                &mut cache.key,
                &mut cache.value,
                start * KV_WIDTH,
                rows * KV_WIDTH,
            )?;
            let partial_launch = self.attn_mod.prepare_partial(LaunchConfig1D::new(
                (heads * splits) as u32,
                128,
                crate::attention_kernel::PARTIAL_SHARED_BYTES,
            ))?;
            self.attn_mod.partial(
                stream,
                &partial_launch,
                &s.rot_q,
                &cache.key,
                &cache.value,
                &mut s.partials,
                rows as u32,
                key_tokens as u32,
                Q_HEADS as u32,
                KV_HEADS as u32,
                start as u32,
                true,
            )?;
            let merge_launch =
                self.attn_mod
                    .prepare_merge(LaunchConfig1D::new(heads as u32, 32, 0))?;
            self.attn_mod.merge(
                stream,
                &merge_launch,
                &s.partials,
                &mut s.attended,
                rows as u32,
                Q_HEADS as u32,
                splits as u32,
            )?;
            self.enqueue_project_bf16(
                &s.attended,
                &layer.o_proj,
                &mut s.projected,
                rows,
                Q_WIDTH,
                HIDDEN,
            )?;
            self.elem_mod.add_bf16(
                stream,
                grid1d(rows * HIDDEN),
                hidden_in,
                &s.projected,
                &mut s.residual,
                rows * HIDDEN,
            )?;
            crate::norm_specialized::enqueue(
                &self.norms,
                stream,
                &s.residual,
                &layer.post_attention_norm,
                &mut s.post,
                rows,
                HIDDEN,
                EPSILON,
            )?;
            self.enqueue_project_bf16(
                &s.post,
                &layer.gate_proj,
                &mut s.gate,
                rows,
                HIDDEN,
                INTERMEDIATE,
            )?;
            self.enqueue_project_bf16(
                &s.post,
                &layer.up_proj,
                &mut s.up,
                rows,
                HIDDEN,
                INTERMEDIATE,
            )?;
            self.ffn_mod.activate_bf16(
                stream,
                grid1d((rows * INTERMEDIATE).div_ceil(8)),
                &s.gate,
                &s.up,
                &mut s.silu,
                &mut s.product,
                rows * INTERMEDIATE,
            )?;
            self.enqueue_project_bf16(
                &s.product,
                &layer.down_proj,
                &mut s.down,
                rows,
                INTERMEDIATE,
                HIDDEN,
            )?;
            self.elem_mod.add_bf16(
                stream,
                grid1d(rows * HIDDEN),
                &s.residual,
                &s.down,
                hidden_out,
                rows * HIDDEN,
            )?;
        }
        cache.len = key_tokens;
        Ok(())
    }

    /// Execute the `Qwen3-8B` BF16 stack over `input` asynchronously on `self.stream`.
    pub fn forward_bf16(
        &self,
        input: &Batch<HIDDEN>,
        stack: &Qwen3Bf16Stack,
        caches: &mut [Nvfp4KvCache],
        cos: &DeviceBuffer<Bf16>,
        sin: &DeviceBuffer<Bf16>,
    ) -> Result<Batch<HIDDEN>> {
        if caches.len() != stack.layers.len() {
            return Err("BF16 layer count does not match KV cache count".into());
        }
        let rows = input.rows();
        let mut s = self.scratch.lock().map_err(|_| "poisoned scratch")?;
        let mut hidden_a = s.hidden_a.take().ok_or("missing hidden_a scratch")?;
        let mut hidden_b = s.hidden_b.take().ok_or("missing hidden_b scratch")?;

        self.enqueue_layer_bf16(
            input.buffer(),
            &mut hidden_a,
            rows,
            &stack.layers[0],
            &mut caches[0],
            cos,
            sin,
            &mut s,
        )?;
        for (idx, (layer, cache)) in stack.layers[1..].iter().zip(&mut caches[1..]).enumerate() {
            if idx % 2 == 0 {
                self.enqueue_layer_bf16(
                    &hidden_a,
                    &mut hidden_b,
                    rows,
                    layer,
                    cache,
                    cos,
                    sin,
                    &mut s,
                )?;
            } else {
                self.enqueue_layer_bf16(
                    &hidden_b,
                    &mut hidden_a,
                    rows,
                    layer,
                    cache,
                    cos,
                    sin,
                    &mut s,
                )?;
            }
        }
        let final_in = if stack.layers.len() % 2 == 1 {
            &hidden_a
        } else {
            &hidden_b
        };
        let mut out = self.nvfp4_runner.allocate_batch::<HIDDEN>(rows)?;
        unsafe {
            crate::norm_specialized::enqueue(
                &self.norms,
                &self.stream,
                final_in,
                &stack.final_norm,
                out.buffer_mut(),
                rows,
                HIDDEN,
                EPSILON,
            )?;
        }
        s.hidden_a = Some(hidden_a);
        s.hidden_b = Some(hidden_b);
        Ok(out)
    }

    /// Execute the `Qwen3-8B` NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`) stack over `input`.
    ///
    /// Enqueues all 36 layers asynchronously onto `self.stream` over pre-allocated
    /// device scratch buffers without per-layer allocation or host synchronization.
    pub fn forward_nvfp4(
        &self,
        input: &Batch<HIDDEN>,
        stack: &Qwen3Nvfp4Stack,
        caches: &mut [Nvfp4KvCache],
        cos: &DeviceBuffer<Bf16>,
        sin: &DeviceBuffer<Bf16>,
    ) -> Result<Batch<HIDDEN>> {
        if caches.len() != stack.layers.len() {
            return Err("NVFP4 layer count does not match KV cache count".into());
        }
        let rows = input.rows();
        let mut s = self.scratch.lock().map_err(|_| "poisoned scratch")?;
        let mut hidden_a = s.hidden_a.take().ok_or("missing hidden_a scratch")?;
        let mut hidden_b = s.hidden_b.take().ok_or("missing hidden_b scratch")?;

        self.enqueue_layer_nvfp4(
            input.buffer(),
            &mut hidden_a,
            rows,
            &stack.layers[0],
            &mut caches[0],
            cos,
            sin,
            &mut s,
        )?;
        for (idx, (layer, cache)) in stack.layers[1..].iter().zip(&mut caches[1..]).enumerate() {
            if idx % 2 == 0 {
                self.enqueue_layer_nvfp4(
                    &hidden_a,
                    &mut hidden_b,
                    rows,
                    layer,
                    cache,
                    cos,
                    sin,
                    &mut s,
                )?;
            } else {
                self.enqueue_layer_nvfp4(
                    &hidden_b,
                    &mut hidden_a,
                    rows,
                    layer,
                    cache,
                    cos,
                    sin,
                    &mut s,
                )?;
            }
        }
        let final_in = if stack.layers.len() % 2 == 1 {
            &hidden_a
        } else {
            &hidden_b
        };
        let mut out = self.nvfp4_runner.allocate_batch::<HIDDEN>(rows)?;
        unsafe {
            crate::norm_specialized::enqueue(
                &self.norms,
                &self.stream,
                final_in,
                &stack.final_norm,
                out.buffer_mut(),
                rows,
                HIDDEN,
                EPSILON,
            )?;
        }
        s.hidden_a = Some(hidden_a);
        s.hidden_b = Some(hidden_b);
        Ok(out)
    }
}

/// Compute cosine similarity between two equal-length BF16 slices in `f64`.
#[must_use]
pub fn cosine_similarity_bf16(a: &[Bf16], b: &[Bf16]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x.to_f32() as f64;
        let yf = y.to_f32() as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Run the end-to-end `Qwen3-8B` BF16 vs. NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`) benchmark on GPU 0.
pub fn run_benchmark(
    num_layers: usize,
    prompt_len: usize,
    decode_steps: usize,
    checkpoint_path: Option<&Path>,
) -> Result<BenchmarkReport> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let engine = Qwen3Engine::new(stream.clone())?;

    let bf16_stack = if let Some(path) = checkpoint_path {
        load_qwen3_bf16_checkpoint(&stream, path, num_layers)?
    } else {
        synthesize_qwen3_bf16(&stream, num_layers)?
    };
    let nvfp4_stack = Qwen3Nvfp4Stack::from_bf16_gpu(&stream, &engine.nvfp4_runner, &bf16_stack)?;

    let capacity = prompt_len + decode_steps + 16;
    let mut bf16_caches: Vec<Nvfp4KvCache> = (0..num_layers)
        .map(|_| Nvfp4KvCache::new(&stream, capacity))
        .collect::<Result<_>>()?;
    let mut nvfp4_caches: Vec<Nvfp4KvCache> = (0..num_layers)
        .map(|_| Nvfp4KvCache::new(&stream, capacity))
        .collect::<Result<_>>()?;

    let prompt_host: Vec<Bf16> = (0..prompt_len * HIDDEN)
        .map(|i| Bf16::from_f32(((i % 97) as f32 - 48.0) * 0.03125))
        .collect();
    let prompt_batch = Batch::<HIDDEN>::from_host(&stream, prompt_len, &prompt_host)?;
    let prompt_positions: Vec<u32> = (0..prompt_len as u32).collect();

    let prompt_pos3: Vec<[u32; 3]> = prompt_positions.iter().map(|&p| [p, p, p]).collect();
    let prompt_tables = engine.rotary.tables(&prompt_pos3)?;
    let prompt_cos = DeviceBuffer::from_host(&stream, prompt_tables.cos())?;
    let prompt_sin = DeviceBuffer::from_host(&stream, prompt_tables.sin())?;

    let mut decode_cos_sin = Vec::with_capacity(decode_steps);
    for step in 0..decode_steps {
        let p = (prompt_len + step) as u32;
        let t = engine.rotary.tables(&[[p, p, p]])?;
        decode_cos_sin.push((
            DeviceBuffer::from_host(&stream, t.cos())?,
            DeviceBuffer::from_host(&stream, t.sin())?,
        ));
    }

    // Warm up both paths and reset caches before timed runs.
    let _ = engine.forward_bf16(
        &prompt_batch,
        &bf16_stack,
        &mut bf16_caches,
        &prompt_cos,
        &prompt_sin,
    )?;
    for c in &mut bf16_caches {
        c.len = 0;
    }
    let _ = engine.forward_nvfp4(
        &prompt_batch,
        &nvfp4_stack,
        &mut nvfp4_caches,
        &prompt_cos,
        &prompt_sin,
    )?;
    for c in &mut nvfp4_caches {
        c.len = 0;
    }
    stream.synchronize()?;

    // Timed BF16 prefill + decode.
    let t0 = Instant::now();
    let bf16_prefill_out = engine.forward_bf16(
        &prompt_batch,
        &bf16_stack,
        &mut bf16_caches,
        &prompt_cos,
        &prompt_sin,
    )?;
    stream.synchronize()?;
    let bf16_prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let step_host: Vec<Bf16> = prompt_host[..HIDDEN].to_vec();
    let mut step_in = Batch::<HIDDEN>::from_host(&stream, 1, &step_host)?;
    let t_dec0 = Instant::now();
    for (cos, sin) in &decode_cos_sin {
        step_in = engine.forward_bf16(&step_in, &bf16_stack, &mut bf16_caches, cos, sin)?;
    }
    stream.synchronize()?;
    let bf16_decode_total_ms = t_dec0.elapsed().as_secs_f64() * 1000.0;

    // Timed NVFP4 (E2M1 + E4M3, SF=16) prefill + decode.
    let t1 = Instant::now();
    let nvfp4_prefill_out = engine.forward_nvfp4(
        &prompt_batch,
        &nvfp4_stack,
        &mut nvfp4_caches,
        &prompt_cos,
        &prompt_sin,
    )?;
    stream.synchronize()?;
    let nvfp4_prefill_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let mut nv_step_in = Batch::<HIDDEN>::from_host(&stream, 1, &step_host)?;
    let t_dec1 = Instant::now();
    for (cos, sin) in &decode_cos_sin {
        nv_step_in = engine.forward_nvfp4(&nv_step_in, &nvfp4_stack, &mut nvfp4_caches, cos, sin)?;
    }
    stream.synchronize()?;
    let nvfp4_decode_total_ms = t_dec1.elapsed().as_secs_f64() * 1000.0;

    let bf16_vec = bf16_prefill_out.read(&stream)?;
    let nvfp4_vec = nvfp4_prefill_out.read(&stream)?;
    let cosine_similarity = cosine_similarity_bf16(&bf16_vec, &nvfp4_vec);

    let bf16_bytes = bf16_stack.byte_len();
    let nvfp4_bytes = nvfp4_stack.byte_len();

    Ok(BenchmarkReport {
        bf16: ModeMetrics {
            label: "BF16 (Tensor Core m16n8k16)",
            layers: num_layers,
            weight_bytes: bf16_bytes,
            prompt_tokens: prompt_len,
            prefill_ms: bf16_prefill_ms,
            prefill_tok_per_sec: (prompt_len as f64) / (bf16_prefill_ms / 1000.0),
            decode_steps,
            decode_ms_per_tok: bf16_decode_total_ms / (decode_steps as f64),
            decode_tok_per_sec: (decode_steps as f64) / (bf16_decode_total_ms / 1000.0),
        },
        nvfp4: ModeMetrics {
            label: "NVFP4 (E2M1 + E4M3, SF_VEC = 16)",
            layers: num_layers,
            weight_bytes: nvfp4_bytes,
            prompt_tokens: prompt_len,
            prefill_ms: nvfp4_prefill_ms,
            prefill_tok_per_sec: (prompt_len as f64) / (nvfp4_prefill_ms / 1000.0),
            decode_steps,
            decode_ms_per_tok: nvfp4_decode_total_ms / (decode_steps as f64),
            decode_tok_per_sec: (decode_steps as f64) / (nvfp4_decode_total_ms / 1000.0),
        },
        cosine_similarity,
        compression_ratio: (bf16_bytes as f64) / (nvfp4_bytes as f64),
    })
}
