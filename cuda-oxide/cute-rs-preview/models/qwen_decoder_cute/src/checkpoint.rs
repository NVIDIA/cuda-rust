/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Bounded safetensors reads and owned, role-checked Qwen checkpoint weights.

use crate::{
    generation::{
        PREDICTOR_LAYERS, PredictorWeights, REMAINING_CODES, TALKER_LAYERS, TalkerWeights,
        TextWeights,
    },
    operator::Result,
};
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    BFloat16,
    decoder::{AttentionWeights, BiasWeights, DecoderWeights, EmbeddingWeights, NormWeights},
    ffn::{FfnWeights, ProjectionWeights},
};
use safetensors::tensor::{Dtype, Metadata, TensorInfo};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

/// Maximum JSON metadata length in bytes, checked before allocating its buffer.
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// Seekable safetensors reader with validated metadata and on-demand tensor reads.
///
/// Construction checks the complete file extent; tensor methods allocate only the
/// requested typed values plus a bounded conversion buffer. Payload values are
/// loaded exactly and are not checked for finiteness.
pub struct SafeTensorFile<R = std::fs::File> {
    /// Owned seekable source; each tensor read seeks to its declared payload offset.
    reader: R,
    /// Validated tensor shapes, dtypes, and contiguous non-overlapping byte extents.
    metadata: Metadata,
    /// Absolute byte offset immediately after the length prefix and JSON metadata.
    data_start: u64,
}

impl SafeTensorFile<File> {
    /// Open a checkpoint and validate its bounded header and exact payload length.
    ///
    /// Returns filesystem, metadata, shape/offset, or truncation errors before any
    /// tensor payload is materialized.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_reader(File::open(path)?)
    }
}

impl<R: Read + Seek> SafeTensorFile<R> {
    /// Parse a seekable source after measuring its length, rejecting malformed metadata
    /// and any missing or unclaimed payload bytes. The reader must support seeking
    /// from the beginning and end; no tensor payload is read here.
    fn from_reader(mut reader: R) -> Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let mut length = [0; 8];
        reader.read_exact(&mut length)?;
        let header_len = u64::from_le_bytes(length);
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err("safetensors header exceeds the 16 MiB bound or is empty".into());
        }
        let data_start = header_len.checked_add(8).ok_or("header extent overflow")?;
        if data_start > file_len {
            return Err("safetensors header is truncated".into());
        }
        let mut header = vec![0; usize::try_from(header_len)?];
        reader.read_exact(&mut header)?;
        // Metadata's Deserialize validates every tensor's dtype, shape product,
        // element byte count and contiguous, non-overlapping data offsets.
        let metadata: Metadata = serde_json::from_slice(&header)?;
        let expected_len = data_start
            .checked_add(u64::try_from(metadata.data_len())?)
            .ok_or("file extent overflow")?;
        if expected_len != file_len {
            return Err(
                "safetensors payload is truncated or contains unclaimed trailing bytes".into(),
            );
        }
        Ok(Self {
            reader,
            metadata,
            data_start,
        })
    }

    /// Borrow a named tensor's logical dimensions, or fail if its name is absent.
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.info(name)?.shape)
    }

    /// Return a named tensor's declared storage dtype without converting its payload.
    pub fn dtype(&self, name: &str) -> Result<Dtype> {
        Ok(self.info(name)?.dtype)
    }

    /// Return owned tensor names in metadata offset order.
    pub fn names(&self) -> Vec<String> {
        self.metadata.offset_keys()
    }

    /// Resolve one tensor name with a checkpoint-specific missing-name error.
    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.metadata
            .info(name)
            .ok_or_else(|| format!("missing checkpoint tensor {name}").into())
    }

    /// Require an exact name, dimension sequence, and dtype match before reading.
    fn check(&self, name: &str, shape: &[usize], dtype: Dtype) -> Result<&TensorInfo> {
        let info = self.info(name)?;
        if info.dtype != dtype || info.shape != shape {
            return Err(format!(
                "checkpoint tensor {name}: expected {dtype:?} {shape:?}, received {:?} {:?}",
                info.dtype, info.shape
            )
            .into());
        }
        Ok(info)
    }

    /// Decode a validated tensor in at most 64 KiB byte chunks.
    ///
    /// Internal callers supply a nonzero element width `N` matching `dtype` and no
    /// larger than the conversion buffer. Returns shape/dtype, allocation, seek,
    /// and short-read errors; preserves decoder-produced values without rounding.
    fn read_values<T, const N: usize>(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: Dtype,
        decode: impl Fn([u8; N]) -> T,
    ) -> Result<Vec<T>> {
        let info = self.check(name, shape, dtype)?;
        let (start, end) = info.data_offsets;
        let count = (end - start) / N;
        let offset = self
            .data_start
            .checked_add(u64::try_from(start)?)
            .ok_or("tensor offset overflow")?;
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        self.reader.seek(SeekFrom::Start(offset))?;
        // Keep only this tensor's typed host values and a small conversion
        // buffer. No duplicate full checkpoint or full tensor byte allocation.
        let mut chunk = [0u8; 64 * 1024];
        while values.len() < count {
            let elements = (count - values.len()).min(chunk.len() / N);
            self.reader.read_exact(&mut chunk[..elements * N])?;
            for bytes in chunk[..elements * N].as_chunks::<N>().0 {
                values.push(decode(*bytes));
            }
        }
        Ok(values)
    }

    /// Exact little-endian BF16 storage load; never converts another dtype.
    pub fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<Vec<BFloat16>> {
        self.read_values(name, shape, Dtype::BF16, |bytes| {
            BFloat16::from_bits(u16::from_le_bytes(bytes))
        })
    }

    /// Exact FP32 load. Any later narrowing is an explicit caller decision.
    pub fn f32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.read_values(name, shape, Dtype::F32, f32::from_le_bytes)
    }
}

/// One required checkpoint tensor in the pinned CustomVoice shape contract.
struct TensorSpec {
    /// Full safetensors key, including model, layer, and projection role.
    name: String,
    /// Exact dimensions in elements; projection matrices use `[output, input]` order.
    shape: Vec<usize>,
}

/// Enumerate all required BF16 tensors for the pinned Talker, predictor, and text adapter.
///
/// The generated list includes per-layer norms and projections plus every
/// codebook embedding/head; it is not inferred from arbitrary checkpoint metadata.
fn qwen_specs() -> Vec<TensorSpec> {
    let mut specs = Vec::new();
    let mut add = |name: String, shape: &[usize]| {
        specs.push(TensorSpec {
            name,
            shape: shape.to_vec(),
        })
    };
    for (prefix, hidden, intermediate, layers) in [
        ("talker.model", 2048, 6144, TALKER_LAYERS),
        ("talker.code_predictor.model", 1024, 3072, PREDICTOR_LAYERS),
    ] {
        add(format!("{prefix}.norm.weight"), &[hidden]);
        for layer in 0..layers {
            let layer = format!("{prefix}.layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{layer}.{name}.weight"), &[hidden]);
            }
            for name in ["q_norm", "k_norm"] {
                add(format!("{layer}.self_attn.{name}.weight"), &[128]);
            }
            for (name, output, input) in [
                ("q_proj", 2048, hidden),
                ("k_proj", 1024, hidden),
                ("v_proj", 1024, hidden),
                ("o_proj", hidden, 2048),
            ] {
                add(format!("{layer}.self_attn.{name}.weight"), &[output, input]);
            }
            for (name, output, input) in [
                ("gate_proj", intermediate, hidden),
                ("up_proj", intermediate, hidden),
                ("down_proj", hidden, intermediate),
            ] {
                add(format!("{layer}.mlp.{name}.weight"), &[output, input]);
            }
        }
    }
    add("talker.codec_head.weight".into(), &[3072, 2048]);
    add("talker.model.codec_embedding.weight".into(), &[3072, 2048]);
    add("talker.model.text_embedding.weight".into(), &[151936, 2048]);
    for name in ["linear_fc1", "linear_fc2"] {
        add(
            format!("talker.text_projection.{name}.weight"),
            &[2048, 2048],
        );
        add(format!("talker.text_projection.{name}.bias"), &[2048]);
    }
    add(
        "talker.code_predictor.small_to_mtp_projection.weight".into(),
        &[1024, 2048],
    );
    add(
        "talker.code_predictor.small_to_mtp_projection.bias".into(),
        &[1024],
    );
    for group in 0..REMAINING_CODES {
        add(
            format!("talker.code_predictor.model.codec_embedding.{group}.weight"),
            &[2048, 2048],
        );
        add(
            format!("talker.code_predictor.lm_head.{group}.weight"),
            &[2048, 1024],
        );
    }
    specs
}

/// Validate every pinned tensor name/shape/dtype and reject additional tensors.
///
/// Returns the validated loading order without allocating device memory.
fn validate_qwen<R: Read + Seek>(file: &SafeTensorFile<R>) -> Result<Vec<TensorSpec>> {
    let specs = qwen_specs();
    for spec in &specs {
        file.check(&spec.name, &spec.shape, Dtype::BF16)?;
    }
    if file.names().len() != specs.len() {
        return Err("checkpoint tensor count differs from the pinned CustomVoice contract".into());
    }
    Ok(specs)
}

/// Each checkpoint allocation is owned exactly once. Typed decoder views borrow
/// these buffers; they cannot outlive the weights or exchange projection roles.
pub struct QwenWeights {
    /// Sole owners of uploaded BF16 buffers, indexed by their checkpoint tensor names.
    tensors: BTreeMap<String, DeviceBuffer<BFloat16>>,
}

impl QwenWeights {
    /// Validate the full pinned checkpoint contract, then upload one tensor at a time.
    ///
    /// Each upload completes before its host values are released. The returned owner
    /// keeps all device allocations alive for borrowed typed views. Metadata, I/O,
    /// allocation, and CUDA errors propagate; partial owners are dropped on failure.
    pub fn load(stream: &CudaStream, path: impl AsRef<Path>) -> Result<Self> {
        let mut file = SafeTensorFile::open(path)?;
        // Validate the complete model contract before allocating any GPU memory.
        let specs = validate_qwen(&file)?;
        stream.context().bind_to_thread()?;
        let mut tensors = BTreeMap::new();
        for spec in specs {
            let values = file.bf16(&spec.name, &spec.shape)?;
            // Safe from_host waits for upload completion before host values drop.
            let buffer = DeviceBuffer::from_host(stream, &values)?;
            tensors.insert(spec.name, buffer);
        }
        Ok(Self { tensors })
    }

    /// Borrow a loaded allocation by its exact key; missing keys are errors.
    fn tensor(&self, name: &str) -> Result<&DeviceBuffer<BFloat16>> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing loaded tensor {name}").into())
    }

    /// Borrow one decoder layer with hidden width `H`, FFN width `I`, and 128-value heads.
    ///
    /// Projection rows are output channels and their dense strides are input widths.
    /// Constructors check buffer extents while assigning distinct projection/norm roles.
    fn layer<const H: usize, const I: usize>(
        &self,
        prefix: &str,
        index: usize,
    ) -> Result<DecoderWeights<'_, H, I, 2048, 1024, 128>> {
        let prefix = format!("{prefix}.layers.{index}");
        let get = |suffix: &str| self.tensor(&format!("{prefix}.{suffix}.weight"));
        Ok(DecoderWeights::new(
            AttentionWeights::new(
                ProjectionWeights::new(get("self_attn.q_proj")?, H)?,
                ProjectionWeights::new(get("self_attn.k_proj")?, H)?,
                ProjectionWeights::new(get("self_attn.v_proj")?, H)?,
                ProjectionWeights::new(get("self_attn.o_proj")?, 2048)?,
                NormWeights::new(get("self_attn.q_norm")?)?,
                NormWeights::new(get("self_attn.k_norm")?)?,
            ),
            NormWeights::new(get("input_layernorm")?)?,
            NormWeights::new(get("post_attention_layernorm")?)?,
            FfnWeights::new(
                ProjectionWeights::new(get("mlp.gate_proj")?, H)?,
                ProjectionWeights::new(get("mlp.up_proj")?, H)?,
                ProjectionWeights::new(get("mlp.down_proj")?, I)?,
            ),
        ))
    }

    /// Borrow all 28 Talker layers, final norm, 3072-token head, and codec embedding.
    ///
    /// Views remain tied to this weight owner; missing tensors or invalid extents fail.
    pub fn talker(&self) -> Result<TalkerWeights<'_>> {
        Ok(TalkerWeights {
            layers: (0..TALKER_LAYERS)
                .map(|index| self.layer("talker.model", index))
                .collect::<Result<_>>()?,
            norm: NormWeights::new(self.tensor("talker.model.norm.weight")?)?,
            codec_head: ProjectionWeights::new(self.tensor("talker.codec_head.weight")?, 2048)?,
            codec_embedding: EmbeddingWeights::new(
                self.tensor("talker.model.codec_embedding.weight")?,
                2048,
            )?,
        })
    }

    /// Borrow five predictor layers and fifteen 2048-token embedding/head pairs.
    ///
    /// The biased input adapter maps Talker width 2048 to predictor width 1024.
    /// Views cannot outlive the loaded buffers; invalid names or shapes fail.
    pub fn predictor(&self) -> Result<PredictorWeights<'_>> {
        let embeddings = (0..REMAINING_CODES)
            .map(|group| {
                Ok(EmbeddingWeights::new(
                    self.tensor(&format!(
                        "talker.code_predictor.model.codec_embedding.{group}.weight"
                    ))?,
                    2048,
                )?)
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| "predictor embedding count mismatch")?;
        let heads = (0..REMAINING_CODES)
            .map(|group| {
                Ok(ProjectionWeights::new(
                    self.tensor(&format!("talker.code_predictor.lm_head.{group}.weight"))?,
                    1024,
                )?)
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| "predictor head count mismatch")?;
        Ok(PredictorWeights {
            layers: (0..PREDICTOR_LAYERS)
                .map(|index| self.layer("talker.code_predictor.model", index))
                .collect::<Result<_>>()?,
            norm: NormWeights::new(self.tensor("talker.code_predictor.model.norm.weight")?)?,
            projection: ProjectionWeights::new(
                self.tensor("talker.code_predictor.small_to_mtp_projection.weight")?,
                2048,
            )?,
            bias: BiasWeights::new(
                self.tensor("talker.code_predictor.small_to_mtp_projection.bias")?,
            )?,
            embeddings,
            heads,
        })
    }

    /// Borrow the 151936-token text table and two biased 2048-wide projection layers.
    ///
    /// Returns role-checked views tied to this owner, rejecting missing or invalid buffers.
    pub fn text(&self) -> Result<TextWeights<'_>> {
        Ok(TextWeights {
            embedding: EmbeddingWeights::new(
                self.tensor("talker.model.text_embedding.weight")?,
                2048,
            )?,
            first: ProjectionWeights::new(
                self.tensor("talker.text_projection.linear_fc1.weight")?,
                2048,
            )?,
            first_bias: BiasWeights::new(self.tensor("talker.text_projection.linear_fc1.bias")?)?,
            second: ProjectionWeights::new(
                self.tensor("talker.text_projection.linear_fc2.weight")?,
                2048,
            )?,
            second_bias: BiasWeights::new(self.tensor("talker.text_projection.linear_fc2.bias")?)?,
        })
    }
}

/// CPU-only safetensors corruption checks and an opt-in pinned-checkpoint metadata check.
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build an in-memory safetensors file from raw JSON and payload bytes, including malformed cases.
    fn fixture(header: &str, bytes: &[u8]) -> Vec<u8> {
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(bytes);
        file
    }

    /// Check the external pinned checkpoint has exactly the expected tensor set and payload size.
    #[test]
    #[ignore = "requires the pinned model.safetensors path in QWEN_CHECKPOINT"]
    fn pinned_checkpoint_header_matches_typed_weights() {
        let path = std::env::var("QWEN_CHECKPOINT").expect("set QWEN_CHECKPOINT");
        let file = SafeTensorFile::open(path).unwrap();
        assert_eq!(validate_qwen(&file).unwrap().len(), 404);
        assert_eq!(file.metadata.data_len(), 3_833_352_704);
    }

    /// Verify exact BF16 bit preservation and errors for wrong names, shapes, and dtypes.
    #[test]
    fn reads_named_bf16_without_converting_payload_bits() {
        let header = r#"{"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut reader = SafeTensorFile::from_reader(Cursor::new(fixture(
            header,
            &[0, 0, 128, 63, 0xc1, 0x7f, 0, 0x80],
        )))
        .unwrap();
        assert_eq!(
            reader
                .bf16("b", &[2])
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0x7fc1, 0x8000]
        );
        assert_eq!(reader.f32("a", &[1]).unwrap(), [1.0]);
        assert!(reader.bf16("missing", &[2]).is_err());
        assert!(reader.bf16("b", &[1, 2]).is_err());
        assert!(reader.f32("b", &[2]).is_err());
        assert!(reader.bf16("a", &[1]).is_err());
    }

    /// Reject short, oversized, and malformed headers before payload allocation.
    #[test]
    fn rejects_bad_header_lengths_before_reading_or_allocating_payload() {
        for bytes in [
            vec![],
            vec![0; 7],
            (MAX_HEADER_BYTES + 1).to_le_bytes().to_vec(),
            u64::MAX.to_le_bytes().to_vec(),
            fixture("{", &[]),
        ] {
            assert!(SafeTensorFile::from_reader(Cursor::new(bytes)).is_err());
        }
        let mut truncated = 64u64.to_le_bytes().to_vec();
        truncated.extend_from_slice(b"{}");
        assert!(SafeTensorFile::from_reader(Cursor::new(truncated)).is_err());
    }

    /// Require the file length to match metadata exactly, with no trailing bytes.
    #[test]
    fn rejects_truncated_payload_and_unclaimed_trailing_bytes() {
        let header = r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        for bytes in [&[0u8; 3][..], &[0u8; 5][..]] {
            assert!(SafeTensorFile::from_reader(Cursor::new(fixture(header, bytes))).is_err());
        }
    }

    /// Exercise invalid offsets, dtype declarations, shape products, and element byte counts.
    #[test]
    fn rejects_overlap_holes_reversed_offsets_and_wrong_element_count() {
        for header in [
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[4,2]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}"#,
            r#"{"a":{"dtype":"unknown","shape":[1],"data_offsets":[0,2]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[18446744073709551615,2],"data_offsets":[0,2]}}"#,
        ] {
            assert!(
                SafeTensorFile::from_reader(Cursor::new(fixture(header, &[0; 6]))).is_err(),
                "accepted {header}"
            );
        }
    }
}
