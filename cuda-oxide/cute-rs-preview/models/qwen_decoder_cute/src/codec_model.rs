/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Owned BF16 12 Hz codec decoder: checked codebooks, transformer and waveform.

use crate::{
    checkpoint::SafeTensorFile,
    codec_attention,
    codec_ops::{self, CheckpointLayout, ConvWeights},
    model::{self, Batch},
    operator::Result,
    rotary::RotarySpec,
};
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    BFloat16,
    decoder::{
        BiasWeights, EmbeddingWeights, FinalNorm, InputNorm, NormRole, NormWeights,
        PostAttentionNorm, TokenIds,
    },
    ffn::{
        AttentionOutput, ChannelContract, ChannelExpand, Down, Gate, InputProjection, Key,
        OutputProjection, ProjectionRole, ProjectionWeights, Query, Up, Value,
    },
};
use safetensors::tensor::Dtype;
use std::{collections::BTreeMap, path::Path, sync::Arc};

/// Number of 24 kHz mono samples produced by each codec frame.
pub const SAMPLES_PER_FRAME: usize = 1920;
/// Output waveform sample rate in samples per second.
pub const SAMPLE_RATE: usize = 24_000;
/// Maximum newly decoded frames per official streaming chunk.
const CHUNK_FRAMES: usize = 300;
/// Previously decoded frames recomputed at the start of later chunks.
const LEFT_CONTEXT: usize = 25;
/// Ordered residual-quantizer groups in each audio frame.
const CODEBOOKS: usize = 16;
/// Number of valid IDs in every codec codebook.
const VOCAB: usize = 2048;
/// Expected number of decoder-only checkpoint tensors.
const CODEC_WEIGHTS: usize = 269;

/// Convert frames to samples, rejecting element-count or FP32 allocation-byte overflow.
fn sample_count(frames: usize) -> Result<usize> {
    frames
        .checked_mul(SAMPLES_PER_FRAME)
        .and_then(|samples| {
            samples
                .checked_mul(size_of::<f32>())
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .map(|_| samples)
        })
        .ok_or_else(|| "codec waveform extent overflow".into())
}

/// Reject sample extent overflow and any frame/group ID outside the 2048-entry vocabulary.
fn validate_codes(codes: &[[u32; CODEBOOKS]]) -> Result<()> {
    sample_count(codes.len())?;
    for (frame, values) in codes.iter().enumerate() {
        for (group, &code) in values.iter().enumerate() {
            if code as usize >= VOCAB {
                return Err(format!(
                    "codec ID {code} at frame {frame} group {group} exceeds vocabulary {VOCAB}"
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Frame interval and waveform crop used for one overlapped decoding pass.
#[derive(Debug, PartialEq, Eq)]
struct Chunk {
    /// Inclusive input frame offset, including recomputed left context.
    start: usize,
    /// Exclusive input frame offset after up to 300 new frames.
    end: usize,
    /// Leading waveform samples removed to discard the recomputed context.
    crop: usize,
}
/// Split frames into 300-frame chunks with up to 25 context frames and sample crops.
///
/// Returns an empty schedule for zero frames and rejects output extent overflow.
fn chunks(frames: usize) -> Result<Vec<Chunk>> {
    sample_count(frames)?;
    Ok((0..frames)
        .step_by(CHUNK_FRAMES)
        .map(|start| {
            let context = start.min(LEFT_CONTEXT);
            Chunk {
                start: start - context,
                end: start.saturating_add(CHUNK_FRAMES).min(frames),
                crop: context * SAMPLES_PER_FRAME,
            }
        })
        .collect())
}

/// Required decoder checkpoint name and exact physical dimensions.
struct Spec {
    /// Fully qualified safetensors key, including the `decoder.` prefix.
    name: String,
    /// Physical checkpoint dimensions in their stored axis order.
    shape: Vec<usize>,
}
/// Build the 269-tensor inference schema, excluding encoder-only projections.
fn specs() -> Vec<Spec> {
    let mut specs = Vec::new();
    let mut add = |name: String, shape: &[usize]| {
        specs.push(Spec {
            name: format!("decoder.{name}"),
            shape: shape.to_vec(),
        })
    };
    add("pre_conv.conv.weight".into(), &[1024, 512, 3]);
    add("pre_conv.conv.bias".into(), &[1024]);
    for (name, output, input) in [("input_proj", 512, 1024), ("output_proj", 1024, 512)] {
        add(format!("pre_transformer.{name}.weight"), &[output, input]);
        add(format!("pre_transformer.{name}.bias"), &[output]);
    }
    add("pre_transformer.norm.weight".into(), &[512]);
    for layer in 0..8 {
        let prefix = format!("pre_transformer.layers.{layer}");
        for name in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn_layer_scale.scale",
            "mlp_layer_scale.scale",
        ] {
            add(format!("{prefix}.{name}"), &[512]);
        }
        for (name, output, input) in [
            ("self_attn.q_proj", 1024, 512),
            ("self_attn.k_proj", 1024, 512),
            ("self_attn.v_proj", 1024, 512),
            ("self_attn.o_proj", 512, 1024),
            ("mlp.gate_proj", 1024, 512),
            ("mlp.up_proj", 1024, 512),
            ("mlp.down_proj", 512, 1024),
        ] {
            add(format!("{prefix}.{name}.weight"), &[output, input]);
        }
    }
    for (name, count) in [("rvq_first", 1), ("rvq_rest", 15)] {
        add(
            format!("quantizer.{name}.output_proj.weight"),
            &[512, 256, 1],
        );
        for group in 0..count {
            add(
                format!("quantizer.{name}.vq.layers.{group}._codebook.embedding_sum"),
                &[VOCAB, 256],
            );
            add(
                format!("quantizer.{name}.vq.layers.{group}._codebook.cluster_usage"),
                &[VOCAB],
            );
        }
    }
    for stage in 0..2 {
        let p = format!("upsample.{stage}");
        add(format!("{p}.0.conv.weight"), &[1024, 1024, 2]);
        add(format!("{p}.0.conv.bias"), &[1024]);
        add(format!("{p}.1.dwconv.conv.weight"), &[1024, 1, 7]);
        add(format!("{p}.1.dwconv.conv.bias"), &[1024]);
        for name in ["norm.weight", "norm.bias", "gamma"] {
            add(format!("{p}.1.{name}"), &[1024]);
        }
        for (name, output, input) in [("pwconv1", 4096, 1024), ("pwconv2", 1024, 4096)] {
            add(format!("{p}.1.{name}.weight"), &[output, input]);
            add(format!("{p}.1.{name}.bias"), &[output]);
        }
    }
    add("decoder.0.conv.weight".into(), &[1536, 1024, 7]);
    add("decoder.0.conv.bias".into(), &[1536]);
    for (stage, input, output, kernel) in [
        (1, 1536, 768, 16),
        (2, 768, 384, 10),
        (3, 384, 192, 8),
        (4, 192, 96, 6),
    ] {
        let p = format!("decoder.{stage}.block");
        for name in ["alpha", "beta"] {
            add(format!("{p}.0.{name}"), &[input]);
        }
        add(format!("{p}.1.conv.weight"), &[input, output, kernel]);
        add(format!("{p}.1.conv.bias"), &[output]);
        for residual in 2..5 {
            for act in ["act1", "act2"] {
                for name in ["alpha", "beta"] {
                    add(format!("{p}.{residual}.{act}.{name}"), &[output]);
                }
            }
            for (conv, kernel) in [("conv1", 7), ("conv2", 1)] {
                add(
                    format!("{p}.{residual}.{conv}.conv.weight"),
                    &[output, output, kernel],
                );
                add(format!("{p}.{residual}.{conv}.conv.bias"), &[output]);
            }
        }
    }
    for name in ["alpha", "beta"] {
        add(format!("decoder.5.{name}"), &[96]);
    }
    add("decoder.6.conv.weight".into(), &[1, 96, 7]);
    add("decoder.6.conv.bias".into(), &[1]);
    specs
}

/// Checkpoint buffers own their allocations; typed operation views borrow them.
/// The two encoder-only quantizer input projections are intentionally not loaded.
struct Weights {
    /// Owned checkpoint tensors, narrowed from finite FP32 to finite BF16.
    tensors: BTreeMap<String, DeviceBuffer<BFloat16>>,
    /// Sixteen normalized `[2048, 256]` embedding tables in codec group order.
    codebooks: Vec<Batch<256>>,
}
impl Weights {
    /// Validate the complete FP32 schema, upload BF16 weights and normalize codebooks.
    ///
    /// Rejects missing tensors, shape/dtype mismatches, nonfinite values and BF16
    /// overflow; checkpoint I/O and CUDA failures are propagated.
    fn load(stream: &CudaStream, ops: &codec_ops::Runner, path: &Path) -> Result<Self> {
        let mut file = SafeTensorFile::open(path)?;
        let specs = specs();
        if specs.len() != CODEC_WEIGHTS {
            return Err("codec checkpoint contract count mismatch".into());
        }
        // Reject every layout/dtype mismatch before allocating any device tensor.
        for spec in &specs {
            if file.shape(&spec.name)? != spec.shape || file.dtype(&spec.name)? != Dtype::F32 {
                return Err(
                    format!("codec tensor {} must be FP32 {:?}", spec.name, spec.shape).into(),
                );
            }
        }
        let mut tensors = BTreeMap::new();
        for spec in specs {
            let host = file.f32(&spec.name, &spec.shape)?;
            if host.iter().any(|value| !value.is_finite()) {
                return Err(format!("codec tensor {} contains nonfinite data", spec.name).into());
            }
            let narrow: Vec<_> = host.into_iter().map(BFloat16::from_f32).collect();
            if narrow.iter().any(|value| !value.to_f32().is_finite()) {
                return Err(format!("codec tensor {} overflows BF16 storage", spec.name).into());
            }
            tensors.insert(spec.name, DeviceBuffer::from_host(stream, &narrow)?);
        }
        let mut result = Self {
            tensors,
            codebooks: Vec::with_capacity(CODEBOOKS),
        };
        // Normalize in CUDA after checkpoint narrowing, matching official BF16
        // loading. These immutable tables are then reused across every chunk.
        for index in 0..CODEBOOKS {
            let (group, layer) = if index == 0 {
                ("rvq_first", 0)
            } else {
                ("rvq_rest", index - 1)
            };
            let prefix = format!("quantizer.{group}.vq.layers.{layer}._codebook");
            let table = result.tensor(&format!("{prefix}.embedding_sum"))?;
            let usage = result.tensor(&format!("{prefix}.cluster_usage"))?;
            let normalized = ops.normalize_rows_buffer::<256>(table, VOCAB, usage, 1e-5)?;
            result.codebooks.push(normalized);
        }
        Ok(result)
    }
    /// Borrow a decoder tensor by its suffix, returning an error if it was not loaded.
    fn tensor(&self, name: &str) -> Result<&DeviceBuffer<BFloat16>> {
        self.tensors
            .get(&format!("decoder.{name}"))
            .ok_or_else(|| format!("missing codec weight {name}").into())
    }
    /// Borrow dense `[OUT, IN]` projection weights with their semantic role.
    ///
    /// Missing weights and invalid typed extents return errors.
    fn projection<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        name: &str,
    ) -> Result<ProjectionWeights<'_, Role, IN, OUT>> {
        Ok(ProjectionWeights::new(
            self.tensor(&format!("{name}.weight"))?,
            IN,
        )?)
    }
    /// Borrow an OUT-element bias vector, checking its extent and assigning `Role`.
    /// The caller chooses the role; the type cannot identify mislabeled weights.
    fn bias<Role: ProjectionRole, const OUT: usize>(
        &self,
        name: &str,
    ) -> Result<BiasWeights<'_, Role, OUT>> {
        Ok(BiasWeights::new(self.tensor(&format!("{name}.bias"))?)?)
    }
    /// Borrow a C-element normalization weight vector with its typed role.
    fn norm<Role: NormRole, const C: usize>(&self, name: &str) -> Result<NormWeights<'_, Role, C>> {
        Ok(NormWeights::new(self.tensor(&format!("{name}.weight"))?)?)
    }
    /// Borrow grouped convolution weights in the requested physical orientation.
    ///
    /// Optionally resolves bias and propagates missing-tensor or geometry errors.
    fn convolution<const IN: usize, const OUT: usize, const K: usize>(
        &self,
        name: &str,
        layout: CheckpointLayout,
        groups: usize,
        bias: bool,
    ) -> Result<ConvWeights<'_, IN, OUT, K>> {
        ConvWeights::new(
            self.tensor(&format!("{name}.weight"))?,
            if bias {
                Some(self.tensor(&format!("{name}.bias"))?)
            } else {
                None
            },
            layout,
            groups,
        )
    }
}

/// A named host snapshot of one decoder intermediate for fixture comparison.
pub struct Stage {
    /// Stable stage path matching reference fixture tensor names.
    pub name: String,
    /// Time steps in the captured intermediate.
    pub rows: usize,
    /// Channels per time step.
    pub width: usize,
    /// Dense time-major `[rows, width]` BF16 storage.
    pub values: Vec<BFloat16>,
}
/// Optional ordered host snapshots; disabled traces perform no readback.
#[derive(Default)]
pub struct Trace {
    /// Whether record calls download and retain intermediates.
    pub enabled: bool,
    /// Snapshots appended in decoder execution order.
    pub stages: Vec<Stage>,
}
impl Trace {
    /// Append a dense host snapshot when enabled, propagating readback errors.
    fn record<const C: usize>(
        &mut self,
        stream: &CudaStream,
        name: &str,
        value: &Batch<C>,
    ) -> Result<()> {
        if self.enabled {
            self.stages.push(Stage {
                name: name.into(),
                rows: value.rows(),
                width: C,
                values: value.read(stream)?,
            });
        }
        Ok(())
    }
}

/// Safe B=1 waveform decoder. Every intermediate is an owned typed GPU batch.
pub struct Runner {
    /// Stream shared by all decoder components and waveform readback.
    stream: Arc<CudaStream>,
    /// Language-decoder primitives reused for projections, RMSNorm and residuals.
    math: model::Runner,
    /// Checked convolution, LayerNorm and waveform activation executor.
    ops: codec_ops::Runner,
    /// Rotary D64 sliding-attention executor for the eight transformer layers.
    attention: codec_attention::Runner,
    /// Owned decoder tensors and pre-normalized codebooks.
    weights: Weights,
}
impl Runner {
    /// Load embedded kernels and the checked decoder-only FP32 checkpoint.
    ///
    /// Weights are narrowed to BF16 once. Returns checkpoint schema, I/O,
    /// nonfinite-data, allocation or CUDA module errors.
    pub fn load(stream: Arc<CudaStream>, checkpoint: &Path) -> Result<Self> {
        let math = model::Runner::new(stream.clone())?;
        let ops = codec_ops::Runner::new(stream.clone())?;
        let attention = codec_attention::Runner::new(stream.clone())?;
        let weights = Weights::load(&stream, &ops, checkpoint)?;
        Ok(Self {
            stream,
            math,
            ops,
            attention,
            weights,
        })
    }
    /// Project every input row with a named dense weight and optional fused bias.
    ///
    /// The projection path rounds the completed FP32 accumulation to BF16.
    fn linear<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &Batch<IN>,
        name: &str,
        bias: bool,
    ) -> Result<Batch<OUT>> {
        let weights = self.weights.projection::<Role, IN, OUT>(name)?;
        let bias = if bias {
            Some(self.weights.bias::<Role, OUT>(name)?)
        } else {
            None
        };
        self.math.project(input, &weights, bias.as_ref())
    }
    /// Project to BF16, broadcast the bias to each row, then add in BF16 storage.
    ///
    /// The intermediate rounding matches the reference strided input adapter.
    fn project_then_bias<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &Batch<IN>,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: &BiasWeights<'_, Role, OUT>,
    ) -> Result<Batch<OUT>> {
        let projected = self.math.project(input, weight, None)?;
        let bias_rows = self.math.embed(
            &EmbeddingWeights::<1, OUT>::new(bias.buffer(), OUT)?,
            TokenIds::new(&vec![0; input.rows()])?,
        )?;
        self.math.add(&projected, &bias_rows)
    }
    /// Discriminator for the strided input adapter's observable BF16 rounding.
    /// A future fused implementation must retain the intermediate conversion.
    pub fn validate_input_bias_rounding(&self) -> Result<()> {
        let input = Batch::<2>::from_host(
            &self.stream,
            2,
            &[1.0, 0.00390625, 1.0, 0.00390625].map(BFloat16::from_f32),
        )?;
        let weight = DeviceBuffer::from_host(&self.stream, &[BFloat16::from_f32(1.0); 2])?;
        let bias = DeviceBuffer::from_host(&self.stream, &[BFloat16::from_f32(0.0009765625)])?;
        let weight = ProjectionWeights::<InputProjection, 2, 1>::new(&weight, 2)?;
        let bias = BiasWeights::<InputProjection, 1>::new(&bias)?;
        let separate = self
            .project_then_bias(&input, &weight, &bias)?
            .read(&self.stream)?;
        let fused = self
            .math
            .project(&input, &weight, Some(&bias))?
            .read(&self.stream)?;
        if separate.iter().any(|value| value.to_f32() != 1.0)
            || fused.iter().any(|value| value.to_f32() != 1.0078125)
        {
            return Err(
                "codec strided input adapter lost its BF16 dot-before-bias rounding".into(),
            );
        }
        Ok(())
    }
    /// Apply a named causal grouped convolution, preserving the number of time steps.
    fn conv<const IN: usize, const OUT: usize, const K: usize>(
        &self,
        input: &Batch<IN>,
        name: &str,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Batch<OUT>> {
        self.ops.conv(
            input,
            &self
                .weights
                .convolution::<IN, OUT, K>(name, CheckpointLayout::Conv, groups, bias)?,
            dilation,
        )
    }
    /// Apply a named biased transpose convolution, producing `input.rows() * stride` rows.
    fn transpose<const IN: usize, const OUT: usize, const K: usize>(
        &self,
        input: &Batch<IN>,
        name: &str,
        stride: usize,
    ) -> Result<Batch<OUT>> {
        self.ops.transpose(
            input,
            &self
                .weights
                .convolution::<IN, OUT, K>(name, CheckpointLayout::Transpose, 1, true)?,
            stride,
        )
    }
    /// Apply SnakeBeta with the named per-channel logarithmic alpha/beta parameters.
    fn snake<const C: usize>(&self, input: &Batch<C>, name: &str) -> Result<Batch<C>> {
        self.ops.snake(
            input,
            self.weights.tensor(&format!("{name}.alpha"))?,
            self.weights.tensor(&format!("{name}.beta"))?,
        )
    }
    /// Decode sixteen codebook IDs per frame into a `[frames, 512]` activation.
    ///
    /// The first group is projected separately; the remaining fifteen embeddings
    /// are added in group order with BF16 rounding before their shared projection.
    fn quantize(&self, codes: &[[u32; CODEBOOKS]]) -> Result<Batch<512>> {
        let embed = |index: usize| -> Result<Batch<256>> {
            let ids: Vec<_> = codes.iter().map(|frame| frame[index]).collect();
            self.math.embed(
                &EmbeddingWeights::<VOCAB, 256>::new(self.weights.codebooks[index].buffer(), 256)?,
                TokenIds::new(&ids)?,
            )
        };
        let first = embed(0)?;
        let first =
            self.conv::<256, 512, 1>(&first, "quantizer.rvq_first.output_proj", 1, 1, false)?;
        let mut rest = embed(1)?;
        for group in 2..CODEBOOKS {
            rest = self.math.add(&rest, &embed(group)?)?;
        }
        let rest =
            self.conv::<256, 512, 1>(&rest, "quantizer.rvq_rest.output_proj", 1, 1, false)?;
        self.math.add(&first, &rest)
    }
    /// Run the input adapter, eight D64 sliding-attention blocks and output adapter.
    ///
    /// Each block uses 16 heads, a 72-token causal window and learned residual
    /// scales. RMSNorm uses epsilon 1e-5; optional trace entries capture the first
    /// layer projections and each layer output. CUDA/weight/shape errors propagate.
    fn transformer(&self, input: &Batch<1024>, trace: &mut Trace) -> Result<Batch<1024>> {
        let positions: Vec<_> = (0..input.rows())
            .map(|row| Ok([u32::try_from(row)?; 3]))
            .collect::<Result<_>>()?;
        let tables = RotarySpec::<64>::standard(10_000.0, 8000)?.tables(&positions)?;
        // The official NCT convolution output is transposed into a strided NTC
        // view. PyTorch's strided Linear path rounds the dot product to BF16
        // before the bias addition. Preserve that observable rounding boundary
        // even though our Batch is already contiguous and time-major.
        let weight = self
            .weights
            .projection::<InputProjection, 1024, 512>("pre_transformer.input_proj")?;
        let bias = self
            .weights
            .bias::<InputProjection, 512>("pre_transformer.input_proj")?;
        let mut hidden = self.project_then_bias(input, &weight, &bias)?;
        trace.record(&self.stream, "pre_transformer/input_proj", &hidden)?;
        for layer in 0..8 {
            let prefix = format!("pre_transformer.layers.{layer}");
            let name = format!("pre_transformer/layers/{layer}");
            let normalized = self.math.norm_with_epsilon(
                &hidden,
                &self
                    .weights
                    .norm::<InputNorm, 512>(&format!("{prefix}.input_layernorm"))?,
                1e-5,
            )?;
            if layer == 0 {
                trace.record(
                    &self.stream,
                    &format!("{name}/input_layernorm"),
                    &normalized,
                )?;
            }
            let q = self.linear::<Query, 512, 1024>(
                &normalized,
                &format!("{prefix}.self_attn.q_proj"),
                false,
            )?;
            let k = self.linear::<Key, 512, 1024>(
                &normalized,
                &format!("{prefix}.self_attn.k_proj"),
                false,
            )?;
            let v = self.linear::<Value, 512, 1024>(
                &normalized,
                &format!("{prefix}.self_attn.v_proj"),
                false,
            )?;
            if layer == 0 {
                trace.record(&self.stream, &format!("{name}/self_attn/q_proj"), &q)?;
                trace.record(&self.stream, &format!("{name}/self_attn/k_proj"), &k)?;
                trace.record(&self.stream, &format!("{name}/self_attn/v_proj"), &v)?;
            }
            let attended = self.attention.run::<16, 1024>(&q, &k, &v, &tables, 72)?;
            let projected = self.linear::<AttentionOutput, 1024, 512>(
                &attended,
                &format!("{prefix}.self_attn.o_proj"),
                false,
            )?;
            let residual = self.ops.scaled_add(
                &projected,
                &hidden,
                self.weights
                    .tensor(&format!("{prefix}.self_attn_layer_scale.scale"))?,
            )?;
            let normalized = self.math.norm_with_epsilon(
                &residual,
                &self.weights.norm::<PostAttentionNorm, 512>(&format!(
                    "{prefix}.post_attention_layernorm"
                ))?,
                1e-5,
            )?;
            let gate = self.linear::<Gate, 512, 1024>(
                &normalized,
                &format!("{prefix}.mlp.gate_proj"),
                false,
            )?;
            let up =
                self.linear::<Up, 512, 1024>(&normalized, &format!("{prefix}.mlp.up_proj"), false)?;
            let product = self.math.swiglu(&gate, &up)?;
            let down = self.linear::<Down, 1024, 512>(
                &product,
                &format!("{prefix}.mlp.down_proj"),
                false,
            )?;
            hidden = self.ops.scaled_add(
                &down,
                &residual,
                self.weights
                    .tensor(&format!("{prefix}.mlp_layer_scale.scale"))?,
            )?;
            trace.record(&self.stream, &name, &hidden)?;
        }
        let hidden = self.math.norm_with_epsilon(
            &hidden,
            &self
                .weights
                .norm::<FinalNorm, 512>("pre_transformer.norm")?,
            1e-5,
        )?;
        self.linear::<OutputProjection, 512, 1024>(&hidden, "pre_transformer.output_proj", true)
    }
    /// Double time resolution and apply the selected ConvNeXt-style residual stage.
    ///
    /// Depthwise convolution and affine LayerNorm precede 1024→4096→1024 channel
    /// projections, erf GELU and a rounded learned-scale residual addition.
    fn upsample(&self, input: &Batch<1024>, stage: usize) -> Result<Batch<1024>> {
        let prefix = format!("upsample.{stage}");
        let input = self.transpose::<1024, 1024, 2>(input, &format!("{prefix}.0.conv"), 2)?;
        let dw =
            self.conv::<1024, 1024, 7>(&input, &format!("{prefix}.1.dwconv.conv"), 1, 1024, true)?;
        let norm = self.ops.layer_norm(
            &dw,
            self.weights.tensor(&format!("{prefix}.1.norm.weight"))?,
            self.weights.tensor(&format!("{prefix}.1.norm.bias"))?,
            1e-6,
        )?;
        let expanded =
            self.linear::<ChannelExpand, 1024, 4096>(&norm, &format!("{prefix}.1.pwconv1"), true)?;
        let activated = self.ops.gelu(&expanded)?;
        let output = self.linear::<ChannelContract, 4096, 1024>(
            &activated,
            &format!("{prefix}.1.pwconv2"),
            true,
        )?;
        self.ops.scaled_add(
            &output,
            &input,
            self.weights.tensor(&format!("{prefix}.1.gamma"))?,
        )
    }
    /// Apply two SnakeBeta/convolution pairs and add the original dense activation.
    ///
    /// The seven-tap first convolution uses the supplied dilation; the second is
    /// pointwise. Each operation retains its own BF16 output rounding.
    fn residual<const C: usize>(
        &self,
        input: &Batch<C>,
        prefix: &str,
        dilation: usize,
    ) -> Result<Batch<C>> {
        let value = self.snake(input, &format!("{prefix}.act1"))?;
        let value =
            self.conv::<C, C, 7>(&value, &format!("{prefix}.conv1.conv"), dilation, 1, true)?;
        let value = self.snake(&value, &format!("{prefix}.act2"))?;
        let value = self.conv::<C, C, 1>(&value, &format!("{prefix}.conv2.conv"), 1, 1, true)?;
        self.math.add(input, &value)
    }
    /// Apply SnakeBeta, transpose upsampling and residual units with dilations 1, 3 and 9.
    fn block<const IN: usize, const OUT: usize, const K: usize>(
        &self,
        input: &Batch<IN>,
        stage: usize,
        stride: usize,
    ) -> Result<Batch<OUT>> {
        let prefix = format!("decoder.{stage}.block");
        let value = self.snake(input, &format!("{prefix}.0"))?;
        let mut value =
            self.transpose::<IN, OUT, K>(&value, &format!("{prefix}.1.conv"), stride)?;
        for (index, dilation) in [(2, 1), (3, 3), (4, 9)] {
            value = self.residual(&value, &format!("{prefix}.{index}"), dilation)?;
        }
        Ok(value)
    }
    /// Decode 1..=325 frames, including any left context, into BF16 mono samples.
    ///
    /// The output has `codes.len() * 1920` rows and is clamped to `[-1, 1]`.
    /// Enabled traces retain named intermediate readbacks. Invalid IDs, chunk
    /// lengths, output length mismatches and CUDA/weight failures return errors.
    pub fn decode_chunk(&self, codes: &[[u32; CODEBOOKS]], trace: &mut Trace) -> Result<Batch<1>> {
        validate_codes(codes)?;
        if codes.is_empty() || codes.len() > CHUNK_FRAMES + LEFT_CONTEXT {
            return Err("codec chunk must contain 1..=325 frames including context".into());
        }
        let quantized = self.quantize(codes)?;
        trace.record(&self.stream, "quantizer", &quantized)?;
        let input = self.conv::<512, 1024, 3>(&quantized, "pre_conv.conv", 1, 1, true)?;
        trace.record(&self.stream, "pre_conv", &input)?;
        let hidden = self.transformer(&input, trace)?;
        trace.record(&self.stream, "pre_transformer", &hidden)?;
        let hidden = self.upsample(&hidden, 0)?;
        trace.record(&self.stream, "upsample/0/1", &hidden)?;
        let hidden = self.upsample(&hidden, 1)?;
        trace.record(&self.stream, "upsample/1/1", &hidden)?;
        let hidden = self.conv::<1024, 1536, 7>(&hidden, "decoder.0.conv", 1, 1, true)?;
        trace.record(&self.stream, "decoder/0", &hidden)?;
        let hidden = self.block::<1536, 768, 16>(&hidden, 1, 8)?;
        trace.record(&self.stream, "decoder/1", &hidden)?;
        let hidden = self.block::<768, 384, 10>(&hidden, 2, 5)?;
        trace.record(&self.stream, "decoder/2", &hidden)?;
        let hidden = self.block::<384, 192, 8>(&hidden, 3, 4)?;
        trace.record(&self.stream, "decoder/3", &hidden)?;
        let hidden = self.block::<192, 96, 6>(&hidden, 4, 3)?;
        trace.record(&self.stream, "decoder/4", &hidden)?;
        let hidden = self.snake(&hidden, "decoder.5")?;
        let hidden = self.conv::<96, 1, 7>(&hidden, "decoder.6.conv", 1, 1, true)?;
        trace.record(&self.stream, "decoder/6", &hidden)?;
        let waveform = self.ops.clamp(&hidden, -1.0, 1.0)?;
        if waveform.rows() != sample_count(codes.len())? {
            return Err("codec graph produced the wrong sample length".into());
        }
        trace.record(&self.stream, "waveform", &waveform)?;
        Ok(waveform)
    }
    /// Match official 300-frame chunks with 25 frames of recomputed left context.
    /// Waveform tensor arithmetic runs on the GPU. Host code prepares positions
    /// and rotary coefficients, reads the final waveform, crops overlap and
    /// widens BF16 samples exactly.
    /// Return `codes.len() * 1920` FP32 samples by exact widening of BF16 output;
    /// empty input returns an empty vector. Rejects invalid IDs or extent overflow
    /// and propagates reservation, checkpoint-view and CUDA failures.
    pub fn decode(&self, codes: &[[u32; CODEBOOKS]]) -> Result<Vec<f32>> {
        validate_codes(codes)?;
        let mut output = Vec::new();
        output.try_reserve_exact(sample_count(codes.len())?)?;
        for chunk in chunks(codes.len())? {
            let waveform =
                self.decode_chunk(&codes[chunk.start..chunk.end], &mut Trace::default())?;
            output.extend(
                waveform
                    .read(&self.stream)?
                    .into_iter()
                    .skip(chunk.crop)
                    .map(BFloat16::to_f32),
            );
        }
        Ok(output)
    }
}

/// Host-only schema, code-ID and chunk/crop contract checks.
#[cfg(test)]
mod tests {
    use super::*;
    /// Check vocabulary limits, empty input and sample-allocation overflow.
    #[test]
    fn frame_ids_and_audio_extents_are_checked_before_gpu_work() {
        assert!(validate_codes(&[[2047; 16]]).is_ok());
        assert!(validate_codes(&[[2048; 16]]).is_err());
        assert!(validate_codes(&[]).is_ok());
        assert_eq!(sample_count(4).unwrap(), 7680);
        assert!(sample_count(usize::MAX).is_err());
    }
    /// Verify chunk overlap is recomputed and cropped exactly once from output.
    #[test]
    fn chunk_overlap_is_recomputed_and_cropped_once() {
        assert_eq!(
            chunks(601).unwrap(),
            [
                Chunk {
                    start: 0,
                    end: 300,
                    crop: 0
                },
                Chunk {
                    start: 275,
                    end: 600,
                    crop: 48000
                },
                Chunk {
                    start: 575,
                    end: 601,
                    crop: 48000
                },
            ]
        );
        for frames in [0, 1, 299, 300, 301, 600, 601] {
            let samples: usize = chunks(frames)
                .unwrap()
                .iter()
                .map(|c| (c.end - c.start) * 1920 - c.crop)
                .sum();
            assert_eq!(samples, frames * 1920);
        }
    }
    /// Require unique decoder-only keys and exclude encoder input projections.
    #[test]
    fn inference_schema_excludes_encoder_and_quantizer_input_projections() {
        let values = specs();
        assert_eq!(values.len(), 269);
        assert_eq!(
            values
                .iter()
                .map(|s| &s.name)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            269
        );
        assert!(values.iter().all(|s| s.name.starts_with("decoder.")));
        assert!(
            !values
                .iter()
                .any(|s| s.name.contains("quantizer.") && s.name.contains("input_proj"))
        );
    }
}
