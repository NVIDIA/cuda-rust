/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Teacher-forced numerical acceptance against pinned official BF16 fixtures.

use crate::{
    checkpoint::QwenWeights,
    model::{Batch, Runner, Trace},
    operator::{GUARD, KvCache, Result},
    rotary::{HostTables, RotarySpec},
};
use cuda_core::CudaContext;
use cute_rs::Bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    time::Instant,
};

// Frozen before the held-out prompt. See reference/README.md for calibration:
// isolated layers remove cross-layer accumulation; full graphs retain it.
/// Return the frozen relative-L2 gate: 0.01 for isolated layers, otherwise 0.05.
///
/// `QWEN_ISOLATE_LAYERS` selects diagnostic layer isolation by its presence.
fn relative_l2_limit() -> f64 {
    if std::env::var_os("QWEN_ISOLATE_LAYERS").is_some() {
        0.01
    } else {
        0.05
    }
}

/// Checked raw fixture location, byte extent and normalized content identity.
struct Descriptor {
    /// Relative path consisting only of normal components under the fixture directory.
    file: String,
    /// Exact file extent in bytes implied by the shape and element size.
    bytes: usize,
    /// Expected lowercase 64-digit SHA-256 digest.
    sha256: String,
}

/// Validate fixture shape, dtype, byte count, relative path and SHA-256 syntax.
///
/// Rejects dimension/byte overflow and any mismatch with the requested dense
/// shape. File contents and resolved-path confinement are checked when read.
fn descriptor(
    entry: &Value,
    shape: &[usize],
    dtype: &str,
    element_bytes: usize,
) -> Result<Descriptor> {
    let stored_shape = entry["shape"]
        .as_array()
        .ok_or("fixture shape must be an array")?
        .iter()
        .map(|d| {
            d.as_u64()
                .ok_or("invalid fixture dimension")
                .and_then(|d| usize::try_from(d).map_err(|_| "fixture dimension overflow"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if stored_shape != shape || entry["dtype"].as_str() != Some(dtype) {
        return Err(format!(
            "fixture expects {dtype} {shape:?}, received {} {stored_shape:?}",
            entry["dtype"]
        )
        .into());
    }
    let bytes = shape
        .iter()
        .try_fold(element_bytes, |count, &dimension| {
            count.checked_mul(dimension)
        })
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or("fixture byte extent overflow")?;
    if entry["bytes"].as_u64() != Some(u64::try_from(bytes)?) {
        return Err("fixture byte count does not match shape and dtype".into());
    }
    let file = entry["file"].as_str().ok_or("fixture filename missing")?;
    if file.is_empty()
        || Path::new(file)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("fixture path must remain under its reference directory".into());
    }
    let sha256 = entry["sha256"].as_str().ok_or("fixture checksum missing")?;
    if sha256.len() != 64 || !sha256.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("fixture SHA256 must contain 64 hexadecimal characters".into());
    }
    Ok(Descriptor {
        file: file.into(),
        bytes,
        sha256: sha256.to_ascii_lowercase(),
    })
}

/// Check actual local checkpoint bytes against the revision shipped with this example.
/// Call once before model loading and exclude this disk work from warm timings.
pub fn verify_pinned_artifacts(model_directory: &Path) -> Result<Value> {
    let manifest: Value = serde_json::from_str(include_str!("../reference/artifacts.json"))?;
    verify_artifact_manifest(model_directory, &manifest)
}

/// Require every compiled checkpoint/config/tokenizer identity in the fixture manifest.
///
/// Byte lengths and case-insensitive SHA-256 values must match; revision
/// labels alone do not establish artifact identity. Additional entries remain allowed.
fn validate_reference_artifact_pins(manifest: &Value) -> Result<()> {
    let pinned: Value = serde_json::from_str(include_str!("../reference/artifacts.json"))?;
    let required = pinned["artifacts"]
        .as_object()
        .ok_or("compiled artifact pins are missing")?;
    let declared = manifest["artifacts"]
        .as_object()
        .ok_or("reference checkpoint artifact identities are missing")?;
    // Revision strings are labels, not proof of identity. Every pinned weight,
    // config and tokenizer artifact must agree before inspecting local files.
    // Additional declared artifacts still receive the path/byte/hash checks below.
    for (name, expected) in required {
        let received = declared
            .get(name)
            .ok_or_else(|| format!("reference is missing pinned artifact {name}"))?;
        let hash_matches = received["sha256"]
            .as_str()
            .zip(expected["sha256"].as_str())
            .is_some_and(|(received, expected)| received.eq_ignore_ascii_case(expected));
        if received["bytes"].as_u64() != expected["bytes"].as_u64() || !hash_matches {
            return Err(format!("reference artifact differs from compiled pin: {name}").into());
        }
    }
    Ok(())
}

/// Stream-hash every declared artifact and return its observed length and SHA-256.
///
/// Requires the core model/codec files, rejects path or symlink escapes and
/// length/hash mismatches, and propagates filesystem/manifest errors. Verification
/// checks the bytes read; it does not lock the files against later changes.
fn verify_artifact_manifest(model_directory: &Path, manifest: &Value) -> Result<Value> {
    use std::io::Read;
    let directory = model_directory.canonicalize()?;
    let artifacts = manifest["artifacts"]
        .as_object()
        .ok_or("reference checkpoint artifact identities are missing")?;
    for required in [
        "config.json",
        "model.safetensors",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
    ] {
        if !artifacts.contains_key(required) {
            return Err(format!("reference identity missing for {required}").into());
        }
    }
    let mut actual = serde_json::Map::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for (name, expected) in artifacts {
        if name.is_empty()
            || Path::new(name)
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(format!("artifact path escapes model directory: {name}").into());
        }
        let path = directory.join(name).canonicalize()?;
        if !path.starts_with(&directory) {
            return Err(format!("artifact resolves outside model directory: {name}").into());
        }
        let mut file = fs::File::open(path)?;
        let metadata = file.metadata()?;
        let bytes = expected["bytes"]
            .as_u64()
            .ok_or("artifact byte count missing")?;
        let expected_hash = expected["sha256"]
            .as_str()
            .ok_or("artifact SHA256 missing")?;
        if !metadata.is_file() || metadata.len() != bytes {
            return Err(format!("checkpoint artifact extent mismatch: {name}").into());
        }
        if expected_hash.len() != 64 || !expected_hash.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("invalid checkpoint artifact SHA256: {name}").into());
        }
        let mut digest = Sha256::new();
        let mut read_bytes = 0u64;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            read_bytes = read_bytes
                .checked_add(count as u64)
                .ok_or("artifact length overflow")?;
            if read_bytes > bytes {
                return Err(
                    format!("checkpoint artifact changed during verification: {name}").into(),
                );
            }
            digest.update(&buffer[..count]);
        }
        let hash = format!("{:x}", digest.finalize());
        if read_bytes != bytes || !hash.eq_ignore_ascii_case(expected_hash) {
            return Err(format!("checkpoint artifact checksum mismatch: {name}").into());
        }
        actual.insert(name.clone(), json!({"bytes":read_bytes,"sha256":hash}));
    }
    Ok(Value::Object(actual))
}

/// Checked raw reference fixtures, shared by decoder and waveform acceptance.
pub struct Reference {
    /// Canonical root used to confine fixture reads, including symlink resolution.
    directory: PathBuf,
    /// Parsed fixture metadata and declared model/source provenance.
    manifest: Value,
}

impl Reference {
    /// Read a manifest of at most 16 MiB and require schema 1 and the pinned BF16 revisions.
    ///
    /// Canonicalization, file I/O, JSON and provenance failures return errors.
    /// Artifact contents are verified separately by `verify_artifacts`.
    pub fn open(directory: &Path) -> Result<Self> {
        let directory = directory.canonicalize()?;
        let path = directory.join("manifest.json");
        if fs::metadata(&path)?.len() > 16 * 1024 * 1024 {
            return Err("reference manifest exceeds 16 MiB".into());
        }
        let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
        if manifest["schema_version"] != 1
            || manifest["model_revision"] != "0c0e3051f131929182e2c023b9537f8b1c68adfe"
            || manifest["source_revision"] != "022e286b98fbec7e1e916cb940cdf532cd9f488e"
            || manifest["dtype"] != "BF16"
        {
            return Err("reference must use the pinned Qwen model/source and BF16 schema".into());
        }
        Ok(Self {
            directory,
            manifest,
        })
    }

    /// Borrow the parsed manifest, including step metadata and tensor descriptors.
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }

    /// Verify the checkpoint bytes used by this process against fixture provenance.
    /// Run this once before timed execution, rather than trusting declared hashes.
    pub fn verify_artifacts(&self, model_directory: &Path) -> Result<Value> {
        validate_reference_artifact_pins(&self.manifest)?;
        verify_artifact_manifest(model_directory, &self.manifest)
    }

    /// Return whether the manifest declares a tensor with the given fixture name.
    pub fn contains(&self, name: &str) -> bool {
        self.manifest["tensors"].get(name).is_some()
    }

    /// Read a tensor only after checking its descriptor and canonical path confinement.
    ///
    /// Requires exact file size and SHA-256 identity; malformed metadata, path
    /// escape, changed bytes and filesystem failures return errors.
    fn bytes(
        &self,
        name: &str,
        shape: &[usize],
        dtype: &str,
        element_bytes: usize,
    ) -> Result<Vec<u8>> {
        let entry = &self.manifest["tensors"][name];
        let spec =
            descriptor(entry, shape, dtype, element_bytes).map_err(|e| format!("{name}: {e}"))?;
        let path = self.directory.join(&spec.file).canonicalize()?;
        if !path.starts_with(&self.directory) {
            return Err(format!("fixture {name} resolves outside reference directory").into());
        }
        if fs::metadata(&path)?.len() != u64::try_from(spec.bytes)? {
            return Err(format!("fixture {name} file length differs from manifest").into());
        }
        let bytes = fs::read(path)?;
        if bytes.len() != spec.bytes || format!("{:x}", Sha256::digest(&bytes)) != spec.sha256 {
            return Err(format!("fixture {name} checksum or byte count mismatch").into());
        }
        Ok(bytes)
    }

    /// Read an exact-shape BF16 fixture from little-endian 16-bit storage.
    ///
    /// Preserves every storage bit; descriptor, path, I/O and checksum failures propagate.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<Bf16>> {
        Ok(self
            .bytes(name, shape, "BF16", 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| Bf16::from_bits(u16::from_le_bytes(*b)))
            .collect())
    }

    /// Read an exact-shape F32 fixture from little-endian 32-bit storage.
    ///
    /// Descriptor, path, I/O and checksum failures propagate; finiteness is checked by consumers.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self
            .bytes(name, shape, "F32", 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    /// Read an exact-shape I64 fixture from little-endian 64-bit storage.
    ///
    /// Descriptor, path, I/O and checksum failures propagate.
    pub fn i64(&self, name: &str, shape: &[usize]) -> Result<Vec<i64>> {
        Ok(self
            .bytes(name, shape, "I64", 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect())
    }
}

/// Decode nonempty reference positions into one three-axis u32 tuple per row.
///
/// Talker preserves three MRoPE axes; predictor duplicates its single axis.
/// Rejects missing/short arrays and nonfinite, fractional or out-of-range IDs.
fn positions(step: &Value, talker: bool) -> Result<Vec<[u32; 3]>> {
    let rows = usize::try_from(step["rows"].as_u64().ok_or("step row count missing")?)?;
    if rows == 0 {
        return Err("acceptance steps require nonempty input".into());
    }
    let mut result = vec![[0; 3]; rows];
    for axis in 0..if talker { 3 } else { 1 } {
        let values = if talker {
            &step["position_ids"][axis][0]
        } else {
            &step["position_ids"][0]
        };
        let values = values.as_array().ok_or("position IDs are missing")?;
        if values.len() != rows {
            return Err("position rows differ from input rows".into());
        }
        for (position, value) in result.iter_mut().zip(values) {
            let value = value.as_f64().ok_or("position ID is not numeric")?;
            if !value.is_finite()
                || value < 0.0
                || value > f64::from(u32::MAX)
                || value.fract() != 0.0
            {
                return Err("position IDs must be exact nonnegative u32 integers".into());
            }
            if talker {
                position[axis] = value as u32;
            } else {
                *position = [value as u32; 3];
            }
        }
    }
    Ok(result)
}

/// Select the first-index maximum after suppression, minimum length and repetition penalty.
///
/// The sign-sensitive penalty is applied in FP32 to previously emitted IDs.
/// Rejects invalid policy fields, empty/oversized vocabularies, nonfinite logits
/// or a policy that suppresses every candidate.
fn policy_token(logits: &[Bf16], previous: &[u32], generation: &Value) -> Result<u32> {
    if logits.is_empty() || logits.len() > u32::MAX as usize {
        return Err("invalid logits vocabulary".into());
    }
    let penalty = generation["repetition_penalty"]
        .as_f64()
        .ok_or("repetition penalty missing")? as f32;
    if !penalty.is_finite() || penalty <= 0.0 {
        return Err("repetition penalty must be finite and positive".into());
    }
    let minimum = usize::try_from(
        generation["min_new_tokens"]
            .as_u64()
            .ok_or("minimum generation length missing")?,
    )?;
    let eos = generation["eos_token_id"]
        .as_u64()
        .ok_or("EOS token missing")?;
    let suppressed = generation["suppress_tokens"]
        .as_array()
        .ok_or("suppression list missing")?;
    let mut best = None;
    for (index, value) in logits.iter().enumerate() {
        let mut value = value.to_f32();
        if !value.is_finite() {
            return Err(format!("nonfinite logit at {index}").into());
        }
        if suppressed.iter().any(|v| v.as_u64() == Some(index as u64))
            || (previous.len() < minimum && index as u64 == eos)
        {
            continue;
        }
        if previous.contains(&(index as u32)) {
            value = if value < 0.0 {
                value * penalty
            } else {
                value / penalty
            };
        }
        if best.is_none_or(|(_, highest)| value > highest) {
            best = Some((index as u32, value));
        }
    }
    best.map(|(index, _)| index)
        .ok_or_else(|| "generation policy suppressed every token".into())
}

/// Accumulated numerical measurements, token decisions and acceptance failures.
#[derive(Default)]
struct Report {
    /// Per-tensor error metrics and pass/fail decisions for JSON output.
    comparisons: Vec<Value>,
    /// Recorded raw or policy-processed token decisions and diagnostic margins.
    predictions: Vec<Value>,
    /// Numerical gates, required prediction disagreements and execution failures.
    failures: Vec<String>,
}

impl Report {
    /// Record maximum absolute error, relative L2 and cosine similarity for two tensors.
    ///
    /// Returns errors for unequal/empty lengths or nonfinite values. Exceeding
    /// the configured relative-L2 limit records a failure while allowing later
    /// comparisons to run. Identical zero tensors have relative L2 zero and cosine one.
    fn compare(&mut self, name: &str, actual: &[Bf16], expected: &[Bf16]) -> Result<()> {
        if actual.len() != expected.len() || actual.is_empty() {
            return Err(format!("{name}: comparison lengths differ or are empty").into());
        }
        let (mut maximum, mut square_error, mut reference_norm, mut actual_norm, mut dot) =
            (0.0f64, 0.0, 0.0, 0.0, 0.0);
        let mut maximum_index = 0;
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            let (a, b) = (f64::from(a.to_f32()), f64::from(b.to_f32()));
            if !a.is_finite() || !b.is_finite() {
                return Err(format!("{name}[{index}]: nonfinite values {a}/{b}").into());
            }
            let error = (a - b).abs();
            if error > maximum {
                maximum = error;
                maximum_index = index;
            }
            square_error += error * error;
            reference_norm += b * b;
            actual_norm += a * a;
            dot += a * b;
        }
        let relative_l2 = if reference_norm == 0.0 {
            if square_error == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (square_error / reference_norm).sqrt()
        };
        let cosine = if actual_norm == 0.0 || reference_norm == 0.0 {
            if actual_norm == reference_norm {
                1.0
            } else {
                0.0
            }
        } else {
            dot / (actual_norm * reference_norm).sqrt()
        };
        let limit = relative_l2_limit();
        let passed = relative_l2 <= limit;
        self.comparisons.push(json!({"name":name,"values":actual.len(),"max_abs":maximum,"max_abs_index":maximum_index,"actual_at_max":actual[maximum_index].to_f32(),"reference_at_max":expected[maximum_index].to_f32(),"relative_l2":relative_l2,"cosine":cosine,"passed":passed}));
        if !passed {
            let message = format!(
                "{name}: relative L2 {relative_l2:.6} > {limit}, max abs {maximum:.6} at {maximum_index} ({:.6}/{:.6})",
                actual[maximum_index].to_f32(),
                expected[maximum_index].to_f32()
            );
            eprintln!("{message}");
            self.failures.push(message);
        }
        Ok(())
    }

    /// Load the checked BF16 fixture and append its comparison with the actual tensor.
    fn fixture(
        &mut self,
        reference: &Reference,
        name: &str,
        actual: &[Bf16],
        shape: &[usize],
    ) -> Result<()> {
        self.compare(name, actual, &reference.bf16(name, shape)?)
    }

    /// Record token agreement and mark a mismatch as failure only when required.
    fn prediction(&mut self, name: &str, actual: u32, expected: u32, required: bool) {
        self.predictions.push(json!({"name":name,"actual":actual,"reference":expected,"agreement":actual==expected,"required":required}));
        if required && actual != expected {
            self.failures.push(format!(
                "{name}: predicted code {actual}, expected {expected}"
            ));
        }
    }
}

/// Return the first-index maximum, rejecting empty/nonfinite logits or index overflow.
fn raw_argmax(values: &[Bf16]) -> Result<u32> {
    if values.is_empty() {
        return Err("empty logits".into());
    }
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if !value.to_f32().is_finite() {
            return Err("nonfinite logits".into());
        }
        if value.to_f32() > values[best].to_f32() {
            best = index;
        }
    }
    Ok(u32::try_from(best)?)
}

/// Describe the five highest finite logits, index-stable ties and the top-two margin.
///
/// Returns errors for empty/nonfinite input; a single token has no margin.
fn top_logits(values: &[Bf16]) -> Result<Value> {
    raw_argmax(values)?;
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        values[b]
            .to_f32()
            .total_cmp(&values[a].to_f32())
            .then(a.cmp(&b))
    });
    let margin = if order.len() > 1 {
        Some(values[order[0]].to_f32() - values[order[1]].to_f32())
    } else {
        None
    };
    Ok(
        json!({"top":order.iter().take(5).map(|&index|json!({"token":index,"logit":values[index].to_f32()})).collect::<Vec<_>>(),"top_two_margin":margin}),
    )
}

/// Map traced layer stages to official dense fixture shapes and compare each.
///
/// H/I select hidden/intermediate widths; query/KV geometry is fixed at
/// 16/8 D128 heads. Rotated Q and output are checked outside this trace loop.
fn compare_trace<const H: usize, const I: usize>(
    report: &mut Report,
    reference: &Reference,
    prefix: &str,
    rows: usize,
    trace: &Trace,
) -> Result<()> {
    for (name, values) in &trace.stages {
        let shape = match *name {
            "q_normed" => vec![1, rows, 16, 128],
            "k_normed" => vec![1, rows, 8, 128],
            "q" | "attention" => vec![1, rows, 2048],
            "k" | "v" => vec![1, rows, 1024],
            "gate" | "up" | "silu" | "product" => vec![1, rows, I],
            "rotated_q" | "output" => continue,
            _ => vec![1, rows, H],
        };
        report.fixture(reference, &format!("{prefix}/{name}"), values, &shape)?;
    }
    Ok(())
}

/// Compare written D128 KV-cache prefixes and require untouched unwritten tails/guards.
fn compare_cache(
    report: &mut Report,
    reference: &Reference,
    runner: &Runner,
    prefix: &str,
    cache: &KvCache<8, 128>,
) -> Result<()> {
    let (key, value) = cache.snapshot(runner.stream())?;
    let count = cache.len() * 8 * 128;
    for (name, values) in [("cache_key", key), ("cache_value", value)] {
        report.fixture(
            reference,
            &format!("{prefix}/{name}"),
            &values[..count],
            &[1, cache.len(), 8, 128],
        )?;
        if values[count..].iter().any(|v| v.to_bits() != GUARD) {
            return Err(format!("{prefix}/{name}: unwritten cache or guard changed").into());
        }
    }
    Ok(())
}

/// Build rotary tables from recorded positions and compare selected cosine/sine values.
///
/// Talker uses three-axis interleaving with `[24, 20, 20]` sections; predictor uses
/// standard rotary positions. Returns checked `[rows, 64]` coefficient tables.
fn tables(
    reference: &Reference,
    report: &mut Report,
    prefix: &str,
    step: &Value,
    talker: bool,
) -> Result<HostTables<128>> {
    let positions = positions(step, talker)?;
    let rows = positions.len();
    let tables = if talker {
        RotarySpec::interleaved(1_000_000.0, 32768, [24, 20, 20])?
    } else {
        RotarySpec::standard(1_000_000.0, 65536)?
    }
    .tables(&positions)?;
    for (name, actual) in [("cos", tables.cos()), ("sin", tables.sin())] {
        let shape = if talker {
            vec![3, 1, rows, 128]
        } else {
            vec![1, rows, 128]
        };
        let source = reference.bf16(&format!("{prefix}/{name}"), &shape)?;
        let mut expected = Vec::with_capacity(rows * 64);
        for row in 0..rows {
            for frequency in 0..64 {
                let candidate = frequency % 3;
                let axis = if talker && candidate != 0 && frequency / 3 < 20 {
                    candidate
                } else {
                    0
                };
                expected.push(source[axis * rows * 128 + row * 128 + frequency]);
            }
        }
        report.compare(&format!("{prefix}/{name}_selected"), actual, &expected)?;
    }
    Ok(tables)
}

/// Run teacher-forced Talker steps and one complete 15-step predictor frame.
///
/// Loads checkpoint weights on device zero, compares stages/caches/logits and
/// requires recorded policy decisions. Layer-isolation mode feeds reference
/// previous-layer outputs to limit accumulated error. Setup/reference/CUDA
/// failures return immediately; numerical mismatches accumulate in the report.
fn execute(model_dir: &Path, reference: &Reference, report: &mut Report) -> Result<()> {
    let isolate = std::env::var_os("QWEN_ISOLATE_LAYERS").is_some();
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let runner = Runner::new(stream.clone())?;
    println!("Loading checked Qwen weights...");
    let owned = QwenWeights::load(&stream, model_dir.join("model.safetensors"))?;
    let talker = owned.talker()?;
    let predictor = owned.predictor()?;
    let talker_steps = reference.manifest()["talker_steps"]
        .as_array()
        .ok_or("Talker steps missing")?;
    let predictor_steps = reference.manifest()["predictor_steps"]
        .as_array()
        .ok_or("predictor steps missing")?;
    if talker_steps.is_empty() || predictor_steps.len() != 15 {
        return Err(
            "expected nonempty Talker sequence and one complete 15-step predictor frame".into(),
        );
    }
    let sequences = reference
        .i64("generation/sequences", &[1, talker_steps.len()])?
        .into_iter()
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let capacity = talker_steps
        .iter()
        .try_fold(0usize, |sum, step| {
            sum.checked_add(step["rows"].as_u64()? as usize)
        })
        .ok_or("Talker row count overflow")?;
    let mut caches = (0..28)
        .map(|_| KvCache::<8, 128>::new(&stream, capacity))
        .collect::<Result<Vec<_>>>()?;
    for (step_index, step) in talker_steps.iter().enumerate() {
        let prefix = format!("talker/step_{step_index:03}");
        let tables = tables(reference, report, &prefix, step, true)?;
        let rows = tables.tokens();
        let mut hidden = Batch::<2048>::from_host(
            &stream,
            rows,
            &reference.bf16(&format!("{prefix}/input"), &[1, rows, 2048])?,
        )?;
        for (layer, (weights, cache)) in talker.layers.iter().zip(&mut caches).enumerate() {
            // Diagnostic only: remove cross-layer error accumulation by feeding
            // each layer the official previous-layer output at every time step.
            if isolate && layer > 0 {
                hidden = Batch::<2048>::from_host(
                    &stream,
                    rows,
                    &reference.bf16(
                        &format!("{prefix}/layer_{:02}/output", layer - 1),
                        &[1, rows, 2048],
                    )?,
                )?;
            }
            let label = format!("{prefix}/layer_{layer:02}");
            let mut trace = Trace {
                enabled: layer == 0,
                ..Trace::default()
            };
            hidden = runner.layer::<2048, 6144, 16, 8, 2048, 1024>(
                &hidden, weights, cache, &tables, &mut trace,
            )?;
            if layer == 0 {
                compare_trace::<2048, 6144>(report, reference, &label, rows, &trace)?;
            }
            report.fixture(
                reference,
                &format!("{label}/output"),
                &hidden.read(&stream)?,
                &[1, rows, 2048],
            )?;
            compare_cache(report, reference, &runner, &label, cache)?;
        }
        let hidden = runner.norm(&hidden, &talker.norm)?;
        report.fixture(
            reference,
            &format!("{prefix}/hidden"),
            &hidden.read(&stream)?,
            &[1, rows, 2048],
        )?;
        let logits = runner
            .project(&hidden, &talker.codec_head, None)?
            .read(&stream)?;
        let expected = reference.bf16(&format!("{prefix}/logits"), &[1, rows, 3072])?;
        report.compare(&format!("{prefix}/logits"), &logits, &expected)?;
        let last = &logits[(rows - 1) * 3072..];
        report.prediction(
            &format!("{prefix}/raw_argmax"),
            raw_argmax(last)?,
            raw_argmax(&expected[(rows - 1) * 3072..])?,
            false,
        );
        let token = policy_token(
            last,
            &sequences[..step_index],
            &reference.manifest()["generation"],
        )?;
        report.prediction(
            &format!("{prefix}/processed_code"),
            token,
            sequences[step_index],
            true,
        );
        println!(
            "{prefix}: predicted {token}, reference {}",
            sequences[step_index]
        );
    }
    drop(caches);
    let mut caches = (0..5)
        .map(|_| KvCache::<8, 128>::new(&stream, 16))
        .collect::<Result<Vec<_>>>()?;
    for (step_index, step) in predictor_steps.iter().enumerate() {
        let prefix = format!("predictor/step_{step_index:03}");
        let tables = tables(reference, report, &prefix, step, false)?;
        let rows = tables.tokens();
        let adapter = Batch::<2048>::from_host(
            &stream,
            rows,
            &reference.bf16(&format!("{prefix}/adapter_input"), &[1, rows, 2048])?,
        )?;
        let mut hidden = runner.project(&adapter, &predictor.projection, Some(&predictor.bias))?;
        report.fixture(
            reference,
            &format!("{prefix}/input"),
            &hidden.read(&stream)?,
            &[1, rows, 1024],
        )?;
        for (layer, (weights, cache)) in predictor.layers.iter().zip(&mut caches).enumerate() {
            // Diagnostic only: remove cross-layer error accumulation by feeding
            // each layer the official previous-layer output at every time step.
            if isolate && layer > 0 {
                hidden = Batch::<1024>::from_host(
                    &stream,
                    rows,
                    &reference.bf16(
                        &format!("{prefix}/layer_{:02}/output", layer - 1),
                        &[1, rows, 1024],
                    )?,
                )?;
            }
            let label = format!("{prefix}/layer_{layer:02}");
            let mut trace = Trace {
                enabled: layer == 0,
                ..Trace::default()
            };
            hidden = runner.layer::<1024, 3072, 16, 8, 2048, 1024>(
                &hidden, weights, cache, &tables, &mut trace,
            )?;
            if layer == 0 {
                compare_trace::<1024, 3072>(report, reference, &label, rows, &trace)?;
            }
            report.fixture(
                reference,
                &format!("{label}/output"),
                &hidden.read(&stream)?,
                &[1, rows, 1024],
            )?;
            compare_cache(report, reference, &runner, &label, cache)?;
        }
        let hidden = runner.norm(&hidden, &predictor.norm)?;
        report.fixture(
            reference,
            &format!("{prefix}/hidden"),
            &hidden.read(&stream)?,
            &[1, rows, 1024],
        )?;
        let logits = runner
            .project(&hidden, &predictor.heads[step_index], None)?
            .read(&stream)?;
        let expected = reference.bf16(&format!("{prefix}/logits"), &[1, rows, 2048])?;
        report.compare(&format!("{prefix}/logits"), &logits, &expected)?;
        let token = raw_argmax(&logits[(rows - 1) * 2048..])?;
        let official = raw_argmax(&expected[(rows - 1) * 2048..])?;
        report.prediction(&format!("{prefix}/code"), token, official, true);
        let frame = reference
            .manifest()
            .get("predictor_frame")
            .map(|value| value.as_u64().ok_or("invalid predictor frame index"))
            .transpose()?
            .unwrap_or(0);
        let frame = usize::try_from(frame)?;
        let recorded = reference.manifest()["codes"][frame][step_index + 1]
            .as_u64()
            .ok_or("selected-frame codec token missing")?;
        let prediction = report
            .predictions
            .last_mut()
            .ok_or("missing prediction record")?;
        prediction["frame"] = json!(frame);
        prediction["actual_logits"] = top_logits(&logits[(rows - 1) * 2048..])?;
        prediction["reference_logits"] = top_logits(&expected[(rows - 1) * 2048..])?;
        if u64::from(official) != recorded {
            return Err(format!(
                "{prefix}: exported logits do not reproduce recorded reference token"
            )
            .into());
        }
        println!("{prefix}: predicted {token}, reference {official}");
    }
    Ok(())
}

/// Verify provenance, execute checkpoint acceptance and write a JSON report.
///
/// `QWEN_REPORT` overrides `/tmp/qwen-stage4d-report.json`. Execution errors
/// are recorded before report writing; artifact/open/write failures return
/// directly. Any recorded acceptance failure produces an error after writing.
pub fn run(model_dir: &Path, reference_dir: &Path) -> Result<()> {
    let reference = Reference::open(reference_dir)?;
    let actual_artifacts = reference.verify_artifacts(model_dir)?;
    let mut report = Report::default();
    let started = Instant::now();
    if let Err(error) = execute(model_dir, &reference, &mut report) {
        report.failures.push(error.to_string());
    }
    let path = std::env::var_os("QWEN_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/qwen-stage4d-report.json"));
    let summary = json!({"schema_version":1,"mode":"teacher-forced checkpoint correctness; not a performance benchmark","isolated_layers":std::env::var_os("QWEN_ISOLATE_LAYERS").is_some(),"model_directory":model_dir,"reference_directory":reference_dir,"reference_provenance":reference.manifest(),"actual_artifacts":actual_artifacts,"relative_l2_limit":relative_l2_limit(),"elapsed_seconds":started.elapsed().as_secs_f64(),"passed":report.failures.is_empty(),"comparisons":report.comparisons,"predictions":report.predictions,"failures":report.failures});
    fs::write(&path, serde_json::to_vec_pretty(&summary)?)?;
    println!(
        "Checkpoint comparison report: {} ({} comparisons, {} failures)",
        path.display(),
        report.comparisons.len(),
        report.failures.len()
    );
    if !report.failures.is_empty() {
        return Err(format!(
            "checkpoint acceptance failed: {}; see {}",
            report.failures[0],
            path.display()
        )
        .into());
    }
    Ok(())
}

/// Host-only fixture provenance, path confinement, position and token-policy checks.
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Check first-index ordering for tied logits and reject empty/nonfinite diagnostics.
    #[test]
    fn logits_diagnostic_retains_ties_and_rejects_nonfinite() {
        let result = top_logits(&[1., 3., 3., 2.].map(Bf16::from_f32)).unwrap();
        assert_eq!(result["top"][0]["token"], 1);
        assert_eq!(result["top"][1]["token"], 2);
        assert_eq!(result["top_two_margin"], 0.);
        assert!(top_logits(&[]).is_err());
        assert!(top_logits(&[Bf16::from_f32(f32::NAN)]).is_err());
    }

    /// Create four tiny temporary artifact files and a matching self-declared manifest.
    ///
    /// These deliberately do not match compiled checkpoint pins. Tests remove the
    /// returned directory; filesystem setup failures panic.
    fn artifact_fixture() -> (PathBuf, Reference) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        /// Per-process counter preventing temporary directory collisions between tests.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "qwen-artifact-check-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(directory.join("speech_tokenizer")).unwrap();
        let mut artifacts = serde_json::Map::new();
        for name in [
            "config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
        ] {
            fs::write(directory.join(name), b"weights").unwrap();
            artifacts.insert(
                name.into(),
                json!({"bytes":7,"sha256":format!("{:x}", Sha256::digest(b"weights"))}),
            );
        }
        let reference = Reference {
            directory: directory.clone(),
            manifest: json!({"artifacts":artifacts}),
        };
        (directory, reference)
    }

    /// Reject a self-consistent replacement checkpoint that differs from compiled pins.
    #[test]
    fn reference_cannot_repin_a_replaced_checkpoint_by_updating_its_hash() {
        let (directory, reference) = artifact_fixture();
        // These replacement bytes match the fixture's self-declared hashes.
        // They must still fail the immutable checkpoint identity contract.
        assert!(reference.verify_artifacts(&directory).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    /// Reject every missing pin and mutations to its byte count or SHA-256.
    #[test]
    fn reference_requires_all_checkpoint_config_and_tokenizer_pins() {
        let pinned: Value =
            serde_json::from_str(include_str!("../reference/artifacts.json")).unwrap();
        assert!(validate_reference_artifact_pins(&pinned).is_ok());
        for name in pinned["artifacts"].as_object().unwrap().keys() {
            let mut missing = pinned.clone();
            missing["artifacts"].as_object_mut().unwrap().remove(name);
            assert!(
                validate_reference_artifact_pins(&missing).is_err(),
                "{name}"
            );
            let mut modified = pinned.clone();
            modified["artifacts"][name]["sha256"] = json!("0".repeat(64));
            assert!(
                validate_reference_artifact_pins(&modified).is_err(),
                "{name}"
            );
            modified = pinned.clone();
            modified["artifacts"][name]["bytes"] = json!(0);
            assert!(
                validate_reference_artifact_pins(&modified).is_err(),
                "{name}"
            );
        }
    }

    /// Detect changed artifact contents even when the replacement has identical length.
    #[test]
    fn artifact_verification_detects_same_length_weight_changes() {
        let (directory, reference) = artifact_fixture();
        let actual = verify_artifact_manifest(&directory, &reference.manifest).unwrap();
        assert_eq!(actual.as_object().unwrap().len(), 4);
        assert_eq!(actual["model.safetensors"]["bytes"], 7);
        fs::write(directory.join("model.safetensors"), b"WEIGHTS").unwrap();
        assert!(verify_artifact_manifest(&directory, &reference.manifest).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    /// Reject missing core identities and absolute or parent-traversing artifact paths.
    #[test]
    fn artifact_verification_rejects_missing_identity_and_path_escape() {
        let (directory, mut reference) = artifact_fixture();
        let original = reference.manifest.clone();
        for path in [
            "../outside",
            "/tmp/outside",
            "speech_tokenizer/../../outside",
        ] {
            reference.manifest = original.clone();
            reference.manifest["artifacts"][path] = original["artifacts"]["config.json"].clone();
            assert!(verify_artifact_manifest(&directory, &reference.manifest).is_err());
        }
        reference.manifest = original;
        reference.manifest["artifacts"]
            .as_object_mut()
            .unwrap()
            .remove("model.safetensors");
        assert!(verify_artifact_manifest(&directory, &reference.manifest).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    /// Reject a declared artifact whose symlink resolves outside the model root.
    #[cfg(unix)]
    #[test]
    fn artifact_verification_rejects_a_symlink_outside_model_directory() {
        let (directory, reference) = artifact_fixture();
        let outside = directory.with_extension("outside");
        fs::write(&outside, b"weights").unwrap();
        fs::remove_file(directory.join("config.json")).unwrap();
        std::os::unix::fs::symlink(&outside, directory.join("config.json")).unwrap();
        assert!(verify_artifact_manifest(&directory, &reference.manifest).is_err());
        fs::remove_dir_all(directory).unwrap();
        fs::remove_file(outside).unwrap();
    }

    /// Reject incompatible shapes/dtypes/byte lengths and nonconfined fixture paths.
    #[test]
    fn descriptor_requires_exact_shape_dtype_extent_and_confined_path() {
        let good = json!({"file":"talker/step_000/input.bin", "dtype":"BF16", "shape":[1,2,3], "bytes":12, "sha256":"a".repeat(64)});
        let spec = descriptor(&good, &[1, 2, 3], "BF16", 2).unwrap();
        assert_eq!(spec.bytes, 12);
        assert!(descriptor(&good, &[6], "BF16", 2).is_err());
        assert!(descriptor(&good, &[1, 2, 3], "F32", 4).is_err());
        for path in ["../input.bin", "/tmp/input.bin", "talker/../input.bin", ""] {
            let mut bad = good.clone();
            bad["file"] = json!(path);
            assert!(descriptor(&bad, &[1, 2, 3], "BF16", 2).is_err());
        }
        for bytes in [11, 13] {
            let mut bad = good.clone();
            bad["bytes"] = json!(bytes);
            assert!(descriptor(&bad, &[1, 2, 3], "BF16", 2).is_err());
        }
    }

    /// Preserve Talker axis order and reject fractional or incomplete position arrays.
    #[test]
    fn positions_preserve_mrope_axes_and_reject_fractional_or_short_rows() {
        let mut talker = json!({"rows":2,"position_ids":[[[1.0,2.0]],[[3.0,4.0]],[[5.0,6.0]]]});
        assert_eq!(positions(&talker, true).unwrap(), [[1, 3, 5], [2, 4, 6]]);
        talker["position_ids"][0][0][1] = json!(2.5);
        assert!(positions(&talker, true).is_err());
        let predictor = json!({"rows":2,"position_ids":[[0,1]]});
        assert_eq!(positions(&predictor, false).unwrap(), [[0; 3], [1; 3]]);
        assert!(positions(&json!({"rows":2,"position_ids":[[0]]}), false).is_err());
    }

    /// Verify suppression, minimum output length and sign-sensitive repetition penalties.
    #[test]
    fn greedy_policy_enforces_suppression_min_length_and_sign_sensitive_penalty() {
        let config = json!({"suppress_tokens":[3],"eos_token_id":2,"min_new_tokens":2,"repetition_penalty":2.0});
        let logits = [5.0, 4.0, 8.0, 100.0].map(Bf16::from_f32);
        assert_eq!(policy_token(&logits, &[], &config).unwrap(), 0);
        assert_eq!(policy_token(&logits, &[0], &config).unwrap(), 1);
        assert_eq!(policy_token(&logits, &[0, 1], &config).unwrap(), 2);
        let negative = [-1.0, -1.5, -9.0, -100.0].map(Bf16::from_f32);
        assert_eq!(policy_token(&negative, &[0], &config).unwrap(), 1);
        assert!(policy_token(&[Bf16::from_f32(f32::NAN)], &[], &config).is_err());
    }
}
