/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Pinned official codec fixtures, checked layout conversion and waveform metrics.
use crate::{
    checkpoint_validation::Reference,
    codec_model::{Runner, SAMPLES_PER_FRAME, Stage, Trace},
    operator::Result,
};
use cute_rs::BFloat16;
use std::path::Path;

/// Error and signal metrics computed over finite aligned waveform/stage values.
#[derive(Debug)]
struct Metrics {
    /// Number of scalar values compared.
    count: usize,
    /// Largest absolute difference in the stage values' native units.
    max_abs: f64,
    /// Root squared-error/squared-reference ratio; infinity for nonzero error on zero signal.
    relative_rms: f64,
    /// 10 log10(reference energy/error energy), in decibels; infinity for exact output.
    snr_db: f64,
}
/// Compute FP64 error metrics for equal nonempty finite FP32 arrays.
///
/// Rejects empty/mismatched extents or nonfinite data; exact results have
/// relative RMS zero and infinite SNR, including an all-zero reference.
fn metrics(actual: &[f32], expected: &[f32]) -> Result<Metrics> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("codec comparison requires equal nonempty lengths".into());
    }
    let mut error = 0.0f64;
    let mut signal = 0.0f64;
    let mut maximum = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            return Err("codec output/reference contains nonfinite values".into());
        }
        let difference = a as f64 - b as f64;
        maximum = maximum.max(difference.abs());
        error += difference * difference;
        signal += (b as f64) * (b as f64);
    }
    let relative_rms = if error == 0.0 {
        0.0
    } else if signal == 0.0 {
        f64::INFINITY
    } else {
        (error / signal).sqrt()
    };
    let snr_db = if error == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (signal / error).log10()
    };
    Ok(Metrics {
        count: actual.len(),
        max_abs: maximum,
        relative_rms,
        snr_db,
    })
}

/// Transpose a batch-one `[channel, time]` BF16 fixture to `[time, channel]`.
///
/// Preserves storage bits and rejects incompatible or overflowing extents.
fn time_major(values: &[BFloat16], rows: usize, channels: usize) -> Result<Vec<BFloat16>> {
    if rows.checked_mul(channels) != Some(values.len()) {
        return Err("codec NCT tensor extent mismatch".into());
    }
    let mut output = vec![BFloat16::from_bits(0); values.len()];
    for t in 0..rows {
        for c in 0..channels {
            output[t * channels + c] = values[c * rows + t];
        }
    }
    Ok(output)
}

/// Read a checked `[1, frames, 16]` I64 fixture into ordered u32 codec frames.
///
/// Rejects malformed shape/counts, invalid integer conversions and fixture errors;
/// vocabulary bounds are checked by the decoder.
fn codes(reference: &Reference, label: &str) -> Result<Vec<[u32; 16]>> {
    let name = format!("codec/{label}/codes");
    let shape = reference.manifest()["tensors"][&name]["shape"]
        .as_array()
        .ok_or("codec input shape missing")?;
    if shape.len() != 3 || shape[0].as_u64() != Some(1) || shape[2].as_u64() != Some(16) {
        return Err("codec input must have shape [1,frames,16]".into());
    }
    let frames = usize::try_from(shape[1].as_u64().ok_or("invalid codec frame count")?)?;
    let values = reference.i64(&name, &[1, frames, 16])?;
    values
        .as_chunks::<16>()
        .0
        .iter()
        .map(|frame| {
            let mut result = [0; 16];
            for (slot, &value) in result.iter_mut().zip(frame) {
                *slot = u32::try_from(value)?;
            }
            Ok(result)
        })
        .collect()
}

/// Validate codec stages and waveforms against pinned official fixture artifacts.
///
/// Runs required short/generated cases and available long-window/chunk-boundary
/// cases on device zero. Every compared stage must have relative RMS <= 0.05.
/// Writes measurements to `CUTE_CODEC_REPORT` or `/tmp/qwen-codec-metrics.json`;
/// returns provenance, shape, CUDA, report I/O or numerical acceptance errors.
pub fn run(model_dir: &Path, reference_dir: &Path) -> Result<()> {
    let reference = Reference::open(reference_dir)?;
    let artifacts = reference.verify_artifacts(model_dir)?;
    let context = cuda_core::CudaContext::new(0)?;
    let stream = context.default_stream();
    let runner = Runner::load(
        stream.clone(),
        &model_dir.join("speech_tokenizer/model.safetensors"),
    )?;
    runner.validate_input_bias_rounding()?;
    let mut comparisons = 0;
    let mut worst = 0.0f64;
    let mut failures = Vec::new();
    let mut measurements = Vec::new();
    let mut labels = vec!["two_frames", "four_frames", "generated"];
    for label in ["long_window", "chunk_boundary"] {
        if reference.contains(&format!("codec/{label}/codes")) {
            labels.push(label);
        }
    }
    let long_window_covered = labels.contains(&"long_window");
    let chunk_boundary_covered = labels.contains(&"chunk_boundary");
    for label in labels {
        let mut required = vec!["waveform"];
        if !matches!(label, "long_window" | "chunk_boundary") {
            required.extend([
                "pre_conv",
                "pre_transformer/input_proj",
                "pre_transformer/layers/0/input_layernorm",
                "pre_transformer/layers/0/self_attn/q_proj",
                "pre_transformer/layers/0/self_attn/k_proj",
                "pre_transformer/layers/0/self_attn/v_proj",
                "pre_transformer/layers/0",
                "pre_transformer",
                "decoder/0",
                "decoder/1",
                "decoder/2",
                "decoder/3",
                "decoder/4",
                "decoder/6",
            ]);
        }
        for stage in required {
            if !reference.contains(&format!("codec/{label}/{stage}")) {
                return Err(
                    format!("required codec reference stage missing: {label}/{stage}").into(),
                );
            }
        }
        let input = codes(&reference, label)?;
        let mut trace = Trace {
            enabled: true,
            ..Trace::default()
        };
        if matches!(label, "long_window" | "chunk_boundary") {
            let waveform = runner.decode(&input)?;
            if waveform.len() != input.len() * SAMPLES_PER_FRAME {
                return Err("chunked codec waveform length mismatch".into());
            }
            trace.stages.push(Stage {
                name: "waveform".into(),
                rows: waveform.len(),
                width: 1,
                // decode widens completed BF16 samples exactly; this conversion
                // restores their storage representation for the shared reader.
                values: waveform.into_iter().map(BFloat16::from_f32).collect(),
            });
        } else {
            let output = runner.decode_chunk(&input, &mut trace)?;
            if output.rows() != input.len() * SAMPLES_PER_FRAME {
                return Err("codec waveform length mismatch".into());
            }
        }
        for stage in trace.stages {
            let name = format!("codec/{label}/{}", stage.name);
            if !reference.contains(&name) {
                continue;
            }
            let actual: Vec<_> = stage.values.into_iter().map(BFloat16::to_f32).collect();
            let expected = if stage.name == "waveform" {
                reference.f32(&name, &[stage.rows])?
            } else {
                let token_major = stage.name.starts_with("pre_transformer");
                let shape = if token_major {
                    [1, stage.rows, stage.width]
                } else {
                    [1, stage.width, stage.rows]
                };
                let values = reference.bf16(&name, &shape)?;
                let values = if token_major {
                    values
                } else {
                    time_major(&values, stage.rows, stage.width)?
                };
                values.into_iter().map(BFloat16::to_f32).collect()
            };
            let stats = metrics(&actual, &expected)?;
            println!(
                "codec {label}/{}: n={} max_abs={:.7e} relative_rms={:.7e} snr_db={:.3}",
                stage.name, stats.count, stats.max_abs, stats.relative_rms, stats.snr_db
            );
            comparisons += stats.count;
            worst = worst.max(stats.relative_rms);
            measurements.push(serde_json::json!({
                "fixture": label,
                "stage": stage.name,
                "values": stats.count,
                "max_abs": stats.max_abs,
                "relative_rms": stats.relative_rms,
                "snr_db": stats.snr_db.is_finite().then_some(stats.snr_db),
                "exact": stats.relative_rms == 0.0,
                "passed": stats.relative_rms <= 0.05,
            }));
            // A propagated BF16 graph bound, not a bitwise equality claim.
            // Print every stage so a failed gate identifies the earliest drift.
            if stats.relative_rms > 0.05 {
                failures.push(format!(
                    "{label}/{} relative RMS {:.6}",
                    stage.name, stats.relative_rms
                ));
            }
        }
    }
    let report = serde_json::json!({
        "schema_version": 1,
        "model_revision": reference.manifest()["model_revision"],
        "source_revision": reference.manifest()["source_revision"],
        "codec_checkpoint": reference.manifest()["artifacts"]["speech_tokenizer/model.safetensors"],
        "verified_artifacts": artifacts,
        "reference_attention": reference.manifest()["attention"],
        "reference_sdpa_backend": reference.manifest()["sdpa_backend"],
        "implementation": "cuda-oxide CuTe/CTM BF16 waveform decoder",
        "gate": {"metric": "relative_rms", "maximum": 0.05},
        "values_checked": comparisons,
        "long_window_covered": long_window_covered,
        "chunk_boundary_covered": chunk_boundary_covered,
        "worst_relative_rms": worst,
        "passed": failures.is_empty(),
        "measurements": measurements,
        "failures": failures,
    });
    let report_path = std::env::var_os("CUTE_CODEC_REPORT")
        .unwrap_or_else(|| "/tmp/qwen-codec-metrics.json".into());
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    println!("codec metrics: {}", Path::new(&report_path).display());
    if !failures.is_empty() {
        return Err(format!(
            "codec reference relative RMS exceeds 0.05: {}",
            failures.join(", ")
        )
        .into());
    }
    println!("codec checkpoint: {comparisons} values checked, worst relative RMS {worst:.7e}");
    Ok(())
}

/// Host checks for fixture transposition and waveform metric edge cases.
#[cfg(test)]
mod tests {
    use super::*;
    /// Verify physical channel-first transposition and reject mismatched/overflowing extents.
    #[test]
    fn reference_channel_first_layout_is_transposed_not_reinterpreted() {
        let source = [1., 2., 3., 10., 20., 30.].map(BFloat16::from_f32);
        let got: Vec<_> = time_major(&source, 3, 2)
            .unwrap()
            .into_iter()
            .map(BFloat16::to_f32)
            .collect();
        assert_eq!(got, [1., 10., 2., 20., 3., 30.]);
        assert!(time_major(&source, 4, 2).is_err());
        assert!(time_major(&source, usize::MAX, 2).is_err());
    }
    /// Check exact/missing-signal metrics and reject nonfinite or empty comparisons.
    #[test]
    fn waveform_metrics_reject_nonfinite_and_measure_missing_signal() {
        let exact = metrics(&[1., -1.], &[1., -1.]).unwrap();
        assert_eq!(exact.relative_rms, 0.);
        assert_eq!(exact.snr_db, f64::INFINITY);
        let missing = metrics(&[0., 0.], &[1., -1.]).unwrap();
        assert_eq!(missing.relative_rms, 1.);
        assert_eq!(missing.snr_db, 0.);
        assert!(metrics(&[f32::NAN], &[0.]).is_err());
        assert!(metrics(&[0.], &[f32::INFINITY]).is_err());
        assert!(metrics(&[0.], &[]).is_err());
        assert!(metrics(&[], &[]).is_err());
    }
}
