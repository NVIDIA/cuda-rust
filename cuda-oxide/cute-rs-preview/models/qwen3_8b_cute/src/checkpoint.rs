/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Multi-shard safetensors loader and fast GPU weight initializer for `Qwen/Qwen3-8B`.
//!
//! Supports both:
//! - Loading a HuggingFace `Qwen/Qwen3-8B` directory (`model.safetensors.index.json` or
//!   single `model.safetensors` file), and
//! - Fast deterministic GPU initialization (`init_synthetic_bf16_layer` / `init_synthetic_nvfp4_layer`)
//!   so all 36 layers (`H = 4096, I = 12288, Q = 32, KV = 8, D = 128`) can be benchmarked on RTX 5090
//!   in milliseconds without requiring an external 16 GB download.

use crate::config::{HEAD_DIM, HIDDEN, INTERMEDIATE, KV_WIDTH, Q_WIDTH};
use crate::nvfp4::Nvfp4Projection;
use crate::operator::Result;
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::Bf16;
use cute_rs::decoder::{
    AttentionWeights, DecoderWeights, FinalNorm, InputNorm, KeyNorm, NormWeights,
    PostAttentionNorm, QueryNorm,
};
use cute_rs::ffn::{
    AttentionOutput, Down, FfnWeights, Gate, Key, ProjectionWeights, Query, Up, Value,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Device kernels for fast deterministic initialization of `Qwen3-8B` BF16 and NVFP4 weights.
#[allow(missing_docs, clippy::missing_docs_in_private_items)]
#[cuda_module]
pub mod init_kernels {
    use super::*;

    /// Populate a BF16 weight buffer with bounded, deterministic pseudo-orthogonal values.
    #[kernel]
    pub fn fill_bf16(mut out: DisjointSlice<Bf16>, length: usize, seed: u32, scale: f32) {
        let tid = thread::index_1d().get();
        if tid >= length {
            return;
        }
        let mut x = (tid as u32).wrapping_mul(0x9e37_79b9) ^ seed.wrapping_mul(0x85eb_ca6b);
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        let signed = ((x & 0xffff) as i32 - 32768) as f32 * (1.0 / 32768.0);
        let mut tensor =
            cute_rs::tensor::partition::StridedTensorMut::from_disjoint_slice(&mut out);
        unsafe {
            tensor.store_at(tid, Bf16::from_f32(signed * scale));
        }
    }

    /// Populate an RMSNorm weight buffer with values near `1.0`.
    #[kernel]
    pub fn fill_norm_bf16(mut out: DisjointSlice<Bf16>, length: usize) {
        let tid = thread::index_1d().get();
        if tid >= length {
            return;
        }
        let mut tensor =
            cute_rs::tensor::partition::StridedTensorMut::from_disjoint_slice(&mut out);
        unsafe {
            tensor.store_at(tid, Bf16::from_f32(1.0));
        }
    }
}

/// Owned device buffers for one `Qwen3-8B` decoder layer in BF16.
pub struct OwnedBf16Layer {
    /// Pre-attention RMSNorm scale (`[4096]`).
    pub input_norm: DeviceBuffer<Bf16>,
    /// Post-attention RMSNorm scale (`[4096]`).
    pub post_attention_norm: DeviceBuffer<Bf16>,
    /// Query projection (`[4096, 4096]`).
    pub q_proj: DeviceBuffer<Bf16>,
    /// Key projection (`[1024, 4096]`).
    pub k_proj: DeviceBuffer<Bf16>,
    /// Value projection (`[1024, 4096]`).
    pub v_proj: DeviceBuffer<Bf16>,
    /// Output projection (`[4096, 4096]`).
    pub o_proj: DeviceBuffer<Bf16>,
    /// Per-head query RMSNorm scale (`[128]`).
    pub q_norm: DeviceBuffer<Bf16>,
    /// Per-head key RMSNorm scale (`[128]`).
    pub k_norm: DeviceBuffer<Bf16>,
    /// SwiGLU gate projection (`[12288, 4096]`).
    pub gate_proj: DeviceBuffer<Bf16>,
    /// SwiGLU up projection (`[12288, 4096]`).
    pub up_proj: DeviceBuffer<Bf16>,
    /// SwiGLU down projection (`[4096, 12288]`).
    pub down_proj: DeviceBuffer<Bf16>,
}

impl OwnedBf16Layer {
    /// Borrow this layer's buffers as a role-checked `cute_rs::decoder::DecoderWeights` view.
    pub fn view(
        &self,
    ) -> Result<DecoderWeights<'_, HIDDEN, INTERMEDIATE, Q_WIDTH, KV_WIDTH, HEAD_DIM>> {
        let attention = AttentionWeights::new(
            ProjectionWeights::<Query, HIDDEN, Q_WIDTH>::new(&self.q_proj, HIDDEN)?,
            ProjectionWeights::<Key, HIDDEN, KV_WIDTH>::new(&self.k_proj, HIDDEN)?,
            ProjectionWeights::<Value, HIDDEN, KV_WIDTH>::new(&self.v_proj, HIDDEN)?,
            ProjectionWeights::<AttentionOutput, Q_WIDTH, HIDDEN>::new(&self.o_proj, Q_WIDTH)?,
            NormWeights::<QueryNorm, HEAD_DIM>::new(&self.q_norm)?,
            NormWeights::<KeyNorm, HEAD_DIM>::new(&self.k_norm)?,
        );
        let ffn = FfnWeights::new(
            ProjectionWeights::<Gate, HIDDEN, INTERMEDIATE>::new(&self.gate_proj, HIDDEN)?,
            ProjectionWeights::<Up, HIDDEN, INTERMEDIATE>::new(&self.up_proj, HIDDEN)?,
            ProjectionWeights::<Down, INTERMEDIATE, HIDDEN>::new(&self.down_proj, INTERMEDIATE)?,
        );
        Ok(DecoderWeights::new(
            attention,
            NormWeights::<InputNorm, HIDDEN>::new(&self.input_norm)?,
            NormWeights::<PostAttentionNorm, HIDDEN>::new(&self.post_attention_norm)?,
            ffn,
        ))
    }
}

/// Owned device buffers for one `Qwen3-8B` decoder layer with NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`) linear projections.
pub struct OwnedNvfp4Layer {
    /// Pre-attention RMSNorm scale (`[4096]`, kept in BF16).
    pub input_norm: DeviceBuffer<Bf16>,
    /// Post-attention RMSNorm scale (`[4096]`, kept in BF16).
    pub post_attention_norm: DeviceBuffer<Bf16>,
    /// Per-head query RMSNorm scale (`[128]`, kept in BF16).
    pub q_norm: DeviceBuffer<Bf16>,
    /// Per-head key RMSNorm scale (`[128]`, kept in BF16).
    pub k_norm: DeviceBuffer<Bf16>,
    /// NVFP4 query projection (`[4096, 4096]`).
    pub q_proj: Nvfp4Projection<Query, HIDDEN, Q_WIDTH>,
    /// NVFP4 key projection (`[1024, 4096]`).
    pub k_proj: Nvfp4Projection<Key, HIDDEN, KV_WIDTH>,
    /// NVFP4 value projection (`[1024, 4096]`).
    pub v_proj: Nvfp4Projection<Value, HIDDEN, KV_WIDTH>,
    /// NVFP4 attention output projection (`[4096, 4096]`).
    pub o_proj: Nvfp4Projection<AttentionOutput, Q_WIDTH, HIDDEN>,
    /// NVFP4 SwiGLU gate projection (`[12288, 4096]`).
    pub gate_proj: Nvfp4Projection<Gate, HIDDEN, INTERMEDIATE>,
    /// NVFP4 SwiGLU up projection (`[12288, 4096]`).
    pub up_proj: Nvfp4Projection<Up, HIDDEN, INTERMEDIATE>,
    /// NVFP4 SwiGLU down projection (`[4096, 12288]`).
    pub down_proj: Nvfp4Projection<Down, INTERMEDIATE, HIDDEN>,
}

impl OwnedNvfp4Layer {
    /// Quantize an [`OwnedBf16Layer`] into an [`OwnedNvfp4Layer`] on the GPU via `runner`.
    pub fn from_bf16_gpu(
        stream: &CudaStream,
        runner: &crate::nvfp4::Nvfp4Runner,
        bf16: &OwnedBf16Layer,
    ) -> Result<Self> {
        let view = bf16.view()?;
        let a = view.attention();
        let f = view.ffn();
        Ok(Self {
            input_norm: DeviceBuffer::from_host(stream, &bf16.input_norm.to_host_vec(stream)?)?,
            post_attention_norm: DeviceBuffer::from_host(
                stream,
                &bf16.post_attention_norm.to_host_vec(stream)?,
            )?,
            q_norm: DeviceBuffer::from_host(stream, &bf16.q_norm.to_host_vec(stream)?)?,
            k_norm: DeviceBuffer::from_host(stream, &bf16.k_norm.to_host_vec(stream)?)?,
            q_proj: runner.quantize_weights(a.query())?,
            k_proj: runner.quantize_weights(a.key())?,
            v_proj: runner.quantize_weights(a.value())?,
            o_proj: runner.quantize_weights(a.output())?,
            gate_proj: runner.quantize_weights(f.gate())?,
            up_proj: runner.quantize_weights(f.up())?,
            down_proj: runner.quantize_weights(f.down())?,
        })
    }

    /// Quantize an [`OwnedBf16Layer`] into an [`OwnedNvfp4Layer`] on `stream`.
    pub fn from_bf16(stream: &CudaStream, bf16: &OwnedBf16Layer) -> Result<Self> {
        let view = bf16.view()?;
        let a = view.attention();
        let f = view.ffn();
        Ok(Self {
            input_norm: DeviceBuffer::from_host(stream, &bf16.input_norm.to_host_vec(stream)?)?,
            post_attention_norm: DeviceBuffer::from_host(
                stream,
                &bf16.post_attention_norm.to_host_vec(stream)?,
            )?,
            q_norm: DeviceBuffer::from_host(stream, &bf16.q_norm.to_host_vec(stream)?)?,
            k_norm: DeviceBuffer::from_host(stream, &bf16.k_norm.to_host_vec(stream)?)?,
            q_proj: Nvfp4Projection::from_bf16(stream, a.query())?,
            k_proj: Nvfp4Projection::from_bf16(stream, a.key())?,
            v_proj: Nvfp4Projection::from_bf16(stream, a.value())?,
            o_proj: Nvfp4Projection::from_bf16(stream, a.output())?,
            gate_proj: Nvfp4Projection::from_bf16(stream, f.gate())?,
            up_proj: Nvfp4Projection::from_bf16(stream, f.up())?,
            down_proj: Nvfp4Projection::from_bf16(stream, f.down())?,
        })
    }

    /// Total GPU memory footprint of this layer's NVFP4 linear projections and BF16 norms.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.q_proj.byte_len()
            + self.k_proj.byte_len()
            + self.v_proj.byte_len()
            + self.o_proj.byte_len()
            + self.gate_proj.byte_len()
            + self.up_proj.byte_len()
            + self.down_proj.byte_len()
            + (self.input_norm.len()
                + self.post_attention_norm.len()
                + self.q_norm.len()
                + self.k_norm.len())
                * 2
    }
}

/// Complete `Qwen3-8B` BF16 weight stack (`layers` decoder layers plus final RMSNorm).
pub struct Qwen3Bf16Stack {
    /// Ordered decoder layers (`36` in the full `Qwen/Qwen3-8B` checkpoint).
    pub layers: Vec<OwnedBf16Layer>,
    /// Final RMSNorm scale (`[4096]`).
    pub final_norm: DeviceBuffer<Bf16>,
}

impl Qwen3Bf16Stack {
    /// Borrow the final RMSNorm buffer as a role-checked [`NormWeights`].
    pub fn final_norm_view(&self) -> Result<NormWeights<'_, FinalNorm, HIDDEN>> {
        Ok(NormWeights::new(&self.final_norm)?)
    }

    /// Total device bytes held by all BF16 layers and the final norm.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        let per_layer = (Q_WIDTH * HIDDEN
            + 2 * KV_WIDTH * HIDDEN
            + HIDDEN * Q_WIDTH
            + 3 * HIDDEN * INTERMEDIATE
            + 2 * HIDDEN
            + 2 * HEAD_DIM)
            * 2;
        self.layers.len() * per_layer + self.final_norm.len() * 2
    }
}

/// Complete `Qwen3-8B` NVFP4 weight stack (`layers` decoder layers plus final BF16 RMSNorm).
pub struct Qwen3Nvfp4Stack {
    /// Ordered NVFP4 decoder layers.
    pub layers: Vec<OwnedNvfp4Layer>,
    /// Final RMSNorm scale (`[4096]`, kept in BF16).
    pub final_norm: DeviceBuffer<Bf16>,
}

impl Qwen3Nvfp4Stack {
    /// Quantize a complete [`Qwen3Bf16Stack`] on the GPU into NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`).
    pub fn from_bf16_gpu(
        stream: &CudaStream,
        runner: &crate::nvfp4::Nvfp4Runner,
        bf16: &Qwen3Bf16Stack,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(bf16.layers.len());
        for layer in &bf16.layers {
            layers.push(OwnedNvfp4Layer::from_bf16_gpu(stream, runner, layer)?);
        }
        stream.synchronize()?;
        Ok(Self {
            layers,
            final_norm: DeviceBuffer::from_host(stream, &bf16.final_norm.to_host_vec(stream)?)?,
        })
    }

    /// Quantize a complete [`Qwen3Bf16Stack`] into NVFP4 (`E2M1` + `E4M3`, `SF_VEC = 16`).
    pub fn from_bf16(stream: &CudaStream, bf16: &Qwen3Bf16Stack) -> Result<Self> {
        let mut layers = Vec::with_capacity(bf16.layers.len());
        for layer in &bf16.layers {
            layers.push(OwnedNvfp4Layer::from_bf16(stream, layer)?);
        }
        Ok(Self {
            layers,
            final_norm: DeviceBuffer::from_host(stream, &bf16.final_norm.to_host_vec(stream)?)?,
        })
    }

    /// Borrow the final RMSNorm buffer as a role-checked [`NormWeights`].
    pub fn final_norm_view(&self) -> Result<NormWeights<'_, FinalNorm, HIDDEN>> {
        Ok(NormWeights::new(&self.final_norm)?)
    }

    /// Total device bytes held by all NVFP4 layers and the final norm.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.layers
            .iter()
            .map(OwnedNvfp4Layer::byte_len)
            .sum::<usize>()
            + self.final_norm.len() * 2
    }
}

/// Allocate and initialize `num_layers` synthetic `Qwen3-8B` BF16 layers directly on the GPU.
pub fn synthesize_qwen3_bf16(stream: &CudaStream, num_layers: usize) -> Result<Qwen3Bf16Stack> {
    if num_layers == 0 {
        return Err("synthesize_qwen3_bf16 requires at least one layer".into());
    }
    let init = init_kernels::load(stream.context())?;
    let alloc_filled = |len: usize, seed: u32, scale: f32| -> Result<DeviceBuffer<Bf16>> {
        let mut buf = DeviceBuffer::<Bf16>::zeroed(stream, len)?;
        let cfg = LaunchConfig {
            grid_dim: (len.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            init.fill_bf16(stream, cfg, &mut buf, len, seed, scale)?;
        }
        Ok(buf)
    };
    let alloc_norm = |len: usize| -> Result<DeviceBuffer<Bf16>> {
        let mut buf = DeviceBuffer::<Bf16>::zeroed(stream, len)?;
        let cfg = LaunchConfig {
            grid_dim: (len.div_ceil(128) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            init.fill_norm_bf16(stream, cfg, &mut buf, len)?;
        }
        Ok(buf)
    };

    let mut layers = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let base_seed = (l as u32 + 1) * 100;
        layers.push(OwnedBf16Layer {
            input_norm: alloc_norm(HIDDEN)?,
            post_attention_norm: alloc_norm(HIDDEN)?,
            q_proj: alloc_filled(Q_WIDTH * HIDDEN, base_seed + 1, 0.02)?,
            k_proj: alloc_filled(KV_WIDTH * HIDDEN, base_seed + 2, 0.02)?,
            v_proj: alloc_filled(KV_WIDTH * HIDDEN, base_seed + 3, 0.02)?,
            o_proj: alloc_filled(HIDDEN * Q_WIDTH, base_seed + 4, 0.02)?,
            q_norm: alloc_norm(HEAD_DIM)?,
            k_norm: alloc_norm(HEAD_DIM)?,
            gate_proj: alloc_filled(INTERMEDIATE * HIDDEN, base_seed + 5, 0.02)?,
            up_proj: alloc_filled(INTERMEDIATE * HIDDEN, base_seed + 6, 0.02)?,
            down_proj: alloc_filled(HIDDEN * INTERMEDIATE, base_seed + 7, 0.02)?,
        });
    }
    let final_norm = alloc_norm(HIDDEN)?;
    stream.synchronize()?;
    Ok(Qwen3Bf16Stack { layers, final_norm })
}

/// HuggingFace `model.safetensors.index.json` weight-map schema.
#[derive(Deserialize)]
struct SafetensorsIndex {
    /// Maps tensor names (`model.layers.0.self_attn.q_proj.weight`) to shard filenames.
    weight_map: BTreeMap<String, String>,
}

/// Read one named BF16 tensor of `shape` from an opened safetensors file.
fn read_bf16_from_safetensors(
    file: &mut std::fs::File,
    metadata: &safetensors::tensor::Metadata,
    data_start: u64,
    name: &str,
    shape: &[usize],
) -> Result<Vec<Bf16>> {
    use safetensors::tensor::Dtype;
    use std::io::{Read, Seek, SeekFrom};

    let info = metadata
        .info(name)
        .ok_or_else(|| format!("missing safetensors tensor {name}"))?;
    if info.dtype != Dtype::BF16 || info.shape != shape {
        return Err(format!(
            "tensor {name} expected BF16 {shape:?}, got {:?} {:?}",
            info.dtype, info.shape
        )
        .into());
    }
    let (start, end) = info.data_offsets;
    let count = (end - start) / 2;
    let mut values = Vec::with_capacity(count);
    file.seek(SeekFrom::Start(data_start + start as u64))?;
    let mut chunk = [0u8; 64 * 1024];
    while values.len() < count {
        let n = (count - values.len()).min(chunk.len() / 2);
        file.read_exact(&mut chunk[..n * 2])?;
        for pair in chunk[..n * 2].as_chunks::<2>().0 {
            values.push(Bf16::from_bits(u16::from_le_bytes(*pair)));
        }
    }
    Ok(values)
}

/// Open a safetensors file and parse its header metadata and payload offset.
fn open_safetensors(path: &Path) -> Result<(std::fs::File, safetensors::tensor::Metadata, u64)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len == 0 || header_len > 32 * 1024 * 1024 {
        return Err("safetensors header length is invalid".into());
    }
    let mut header = vec![0u8; usize::try_from(header_len)?];
    file.read_exact(&mut header)?;
    let metadata: safetensors::tensor::Metadata = serde_json::from_slice(&header)?;
    Ok((file, metadata, 8 + header_len))
}

/// Load `num_layers` of `Qwen/Qwen3-8B` from a safetensors file or sharded directory.
pub fn load_qwen3_bf16_checkpoint(
    stream: &CudaStream,
    path: impl AsRef<Path>,
    num_layers: usize,
) -> Result<Qwen3Bf16Stack> {
    let path = path.as_ref();
    let index_path = path.join("model.safetensors.index.json");
    let mut shards = BTreeMap::new();
    let weight_map: BTreeMap<String, String> = if path.is_dir() && index_path.exists() {
        let index: SafetensorsIndex = serde_json::from_slice(&std::fs::read(&index_path)?)?;
        index.weight_map
    } else {
        shards.insert(String::new(), open_safetensors(path)?);
        BTreeMap::new()
    };

    let mut read_tensor = |name: &str, shape: &[usize]| -> Result<DeviceBuffer<Bf16>> {
        let (file, meta, data_start) = if let Some(shard) = weight_map.get(name) {
            if !shards.contains_key(shard) {
                shards.insert(shard.clone(), open_safetensors(&path.join(shard))?);
            }
            shards.get_mut(shard).expect("inserted shard")
        } else {
            shards
                .values_mut()
                .next()
                .ok_or("no safetensors shard opened")?
        };
        let host = read_bf16_from_safetensors(file, meta, *data_start, name, shape)?;
        Ok(DeviceBuffer::from_host(stream, &host)?)
    };

    let mut layers = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let p = format!("model.layers.{l}");
        layers.push(OwnedBf16Layer {
            input_norm: read_tensor(&format!("{p}.input_layernorm.weight"), &[HIDDEN])?,
            post_attention_norm: read_tensor(
                &format!("{p}.post_attention_layernorm.weight"),
                &[HIDDEN],
            )?,
            q_proj: read_tensor(&format!("{p}.self_attn.q_proj.weight"), &[Q_WIDTH, HIDDEN])?,
            k_proj: read_tensor(&format!("{p}.self_attn.k_proj.weight"), &[KV_WIDTH, HIDDEN])?,
            v_proj: read_tensor(&format!("{p}.self_attn.v_proj.weight"), &[KV_WIDTH, HIDDEN])?,
            o_proj: read_tensor(&format!("{p}.self_attn.o_proj.weight"), &[HIDDEN, Q_WIDTH])?,
            q_norm: read_tensor(&format!("{p}.self_attn.q_norm.weight"), &[HEAD_DIM])?,
            k_norm: read_tensor(&format!("{p}.self_attn.k_norm.weight"), &[HEAD_DIM])?,
            gate_proj: read_tensor(
                &format!("{p}.mlp.gate_proj.weight"),
                &[INTERMEDIATE, HIDDEN],
            )?,
            up_proj: read_tensor(&format!("{p}.mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN])?,
            down_proj: read_tensor(
                &format!("{p}.mlp.down_proj.weight"),
                &[HIDDEN, INTERMEDIATE],
            )?,
        });
    }
    let final_norm = read_tensor("model.norm.weight", &[HIDDEN])?;
    Ok(Qwen3Bf16Stack { layers, final_norm })
}
