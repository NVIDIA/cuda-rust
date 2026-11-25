/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Decode launch geometry, CLI options, and static-shape selection.

use crate::fixture::{HEAD_DIM, Shape};
use std::path::PathBuf;

/// Number of key/value rows handled by one partial CTA.
pub const KV_CHUNK: usize = 64;
/// FP32 elements per split: a maximum, a sum, and 64 numerator values.
pub const SCRATCH_STRIDE: usize = 66;
/// Threads in each partial-attention CTA.
pub const PARTIAL_THREADS: u32 = 128;
/// Shared bytes reserved for staged K/V, scores, and reduction state.
pub const PARTIAL_SHARED_BYTES: u32 = 17_696;
/// One warp combines a head's partial results.
pub const MERGE_THREADS: u32 = 32;

/// Policy for choosing a runtime or precompiled shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Specialization {
    /// Use a precompiled shape when available, otherwise runtime dimensions.
    Auto,
    /// Read the KV length and head count from kernel arguments.
    Runtime,
    /// Require a precompiled kernel for the exact requested shape.
    Static,
}

impl Specialization {
    /// Return the CLI spelling used by the comparison driver.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Runtime => "runtime",
            Self::Static => "static",
        }
    }
}

/// Validated options for a decode run or CuTeDSL comparison.
#[derive(Debug)]
pub struct Args {
    /// Run the Rust kernel alone instead of launching the CuTeDSL comparison.
    pub rust_only: bool,
    /// Optional Python interpreter for the comparison process.
    pub python: Option<PathBuf>,
    /// Compare every supported mode combination for the selected workload.
    pub all_cases: bool,
    /// Attention dimensions and mask after CLI validation.
    pub shape: Shape,
    /// Requested runtime/static kernel selection policy.
    pub specialization: Specialization,
    /// Power-of-two multiplier applied to Q and K, leaving V unchanged.
    pub input_scale: u32,
    /// Use an all-zero Q tensor to exercise uniform softmax.
    pub zero_query: bool,
    /// CUDA device ordinal after visible-device remapping.
    pub device: usize,
    /// Untimed invocations before collecting samples.
    pub warmup: usize,
    /// Number of timed samples; zero requests correctness checks only.
    pub iters: usize,
    /// Optional path for raw little-endian FP16 output in standalone mode.
    pub output: Option<PathBuf>,
    /// Optional path for the standalone or comparison JSON report.
    pub json: Option<PathBuf>,
}

/// Checked grid and scratch sizes for the two decode launches.
#[derive(Debug)]
pub struct LaunchPlan {
    /// Number of 64-key splits per head, including a short final split.
    pub splits: u32,
    /// Total partial CTAs across all heads and splits.
    pub partial_blocks: u32,
    /// FP32 elements needed for all partial states.
    pub scratch_elements: usize,
}

impl LaunchPlan {
    /// Validate the decode shape and compute sizes without overflowing kernel indices.
    pub fn new(shape: Shape) -> Result<Self, String> {
        if shape.seq_q != 1 || shape.causal {
            return Err("the decode example requires --seq-q 1 and --noncausal".into());
        }
        if shape.seq_k == 0 || shape.heads == 0 {
            return Err("KV length and head count must be positive".into());
        }
        // The prefill fixture's validator has tile restrictions that do not
        // apply to decode. Keep the common allocation/index bounds here.
        for length in [shape.seq_q, shape.seq_k] {
            let elements = length
                .checked_mul(shape.heads)
                .and_then(|n| n.checked_mul(HEAD_DIM));
            if elements.is_none_or(|n| n > u32::MAX as usize) {
                return Err("tensor element count must fit u32 indexing".into());
            }
        }
        let splits = shape.seq_k.div_ceil(KV_CHUNK);
        let partial_blocks = shape
            .heads
            .checked_mul(splits)
            .ok_or("partial grid size overflow")?;
        let scratch_elements = partial_blocks
            .checked_mul(SCRATCH_STRIDE)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or("scratch element count must fit u32 indexing")?;
        scratch_elements
            .checked_mul(size_of::<f32>())
            .ok_or("scratch byte count overflow")?;
        Ok(Self {
            splits: u32::try_from(splits).map_err(|_| "split count must fit u32")?,
            partial_blocks: u32::try_from(partial_blocks)
                .map_err(|_| "partial grid must fit u32")?,
            scratch_elements,
        })
    }
}

impl Args {
    /// Return the exact precompiled shape or explain why it is unavailable.
    fn static_kernel_shape(&self) -> Result<(u32, u32), String> {
        let dimensions = (self.shape.seq_k, self.shape.heads);
        if matches!(dimensions, (64, 8) | (65, 3) | (1024, 8) | (4097, 8)) {
            Ok((dimensions.0 as u32, dimensions.1 as u32))
        } else {
            Err(format!(
                "no static specialization for Sk={}, H={}; available (Sk,H): (64,8), (65,3), (1024,8), (4097,8)",
                dimensions.0, dimensions.1,
            ))
        }
    }

    /// Select static dimensions, or `(0, 0)` for the runtime kernel.
    pub fn kernel_shape(&self) -> Result<(u32, u32), String> {
        match self.specialization {
            Specialization::Runtime => Ok((0, 0)),
            Specialization::Auto => Ok(self.static_kernel_shape().unwrap_or((0, 0))),
            Specialization::Static => self.static_kernel_shape(),
        }
    }

    /// Parse and validate CLI options; return `None` when help is requested.
    pub fn parse_from(cli: impl IntoIterator<Item = String>) -> Result<Option<Self>, String> {
        let mut args = Self {
            rust_only: false,
            python: None,
            all_cases: false,
            shape: Shape {
                seq_q: 1,
                seq_k: 1024,
                heads: 8,
                causal: false,
            },
            specialization: Specialization::Auto,
            input_scale: 1,
            zero_query: false,
            device: 0,
            warmup: 100,
            iters: 31,
            output: None,
            json: None,
        };
        let mut cli = cli.into_iter();
        let mut mask = None;
        while let Some(key) = cli.next() {
            match key.as_str() {
                "--help" | "-h" => return Ok(None),
                "--causal" | "--noncausal" => {
                    let causal = key == "--causal";
                    if mask.is_some_and(|previous| previous != causal) {
                        return Err("--causal and --noncausal are mutually exclusive".into());
                    }
                    mask = Some(causal);
                    args.shape.causal = causal;
                }
                "--rust-only" => args.rust_only = true,
                "--all-cases" => args.all_cases = true,
                "--python" => {
                    args.python = Some(cli.next().ok_or("missing value for --python")?.into());
                }
                "--zero-query" => args.zero_query = true,
                "--specialization" => {
                    let value = cli.next().ok_or("missing value for --specialization")?;
                    args.specialization = match value.as_str() {
                        "auto" => Specialization::Auto,
                        "runtime" => Specialization::Runtime,
                        "static" => Specialization::Static,
                        _ => return Err("--specialization must be auto, runtime, or static".into()),
                    };
                }
                "--seq-q" | "--seq-k" | "--heads" | "--input-scale" | "--device" | "--warmup"
                | "--iters" | "--output" | "--json" => {
                    let value = cli
                        .next()
                        .ok_or_else(|| format!("missing value for {key}"))?;
                    if key == "--output" {
                        args.output = Some(value.into());
                        continue;
                    }
                    if key == "--json" {
                        args.json = Some(value.into());
                        continue;
                    }
                    let number = value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid nonnegative integer for {key}: {value}"))?;
                    match key.as_str() {
                        "--seq-q" => args.shape.seq_q = number,
                        "--seq-k" => args.shape.seq_k = number,
                        "--heads" => args.shape.heads = number,
                        "--input-scale" => {
                            args.input_scale = u32::try_from(number)
                                .map_err(|_| "--input-scale must be 1, 4, or 16")?
                        }
                        "--device" => args.device = number,
                        "--warmup" => args.warmup = number,
                        "--iters" => args.iters = number,
                        _ => unreachable!(),
                    }
                }
                _ => return Err(format!("unknown argument {key}")),
            }
        }
        crate::comparison::validate_mode(
            args.rust_only,
            args.python.as_ref(),
            args.output.as_ref(),
        )?;
        if args.all_cases && args.rust_only {
            return Err("--all-cases cannot be combined with --rust-only".into());
        }
        LaunchPlan::new(args.shape)?;
        if ![1, 4, 16].contains(&args.input_scale) {
            return Err("--input-scale must be 1, 4, or 16".into());
        }
        if args.all_cases {
            args.static_kernel_shape()?;
        } else {
            args.kernel_shape()?;
        }
        Ok(Some(args))
    }
}

/// Reject a static kernel whose dimensions differ from the workload.
pub fn validate_kernel_shape<const SEQ_K: u32, const HEADS: u32>(
    shape: Shape,
) -> Result<(), String> {
    if (SEQ_K, HEADS) == (0, 0) {
        return Ok(());
    }
    if SEQ_K == 0 || HEADS == 0 || shape.seq_k != SEQ_K as usize || shape.heads != HEADS as usize {
        return Err(format!(
            "kernel specialization Sk={SEQ_K}, H={HEADS} does not match workload Sk={}, H={}",
            shape.seq_k, shape.heads,
        ));
    }
    Ok(())
}

/// Tests for decode CLI validation, specialization selection, and launch bounds.
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a noncausal single-query shape for launch-plan tests.
    fn shape(seq_k: usize, heads: usize) -> Shape {
        Shape {
            seq_q: 1,
            seq_k,
            heads,
            causal: false,
        }
    }

    /// Parse borrowed CLI words without reading process arguments.
    fn parse(cli: &[&str]) -> Result<Option<Args>, String> {
        Args::parse_from(cli.iter().map(|value| (*value).into()))
    }

    /// Check that raw output is standalone-only and Python selection is comparison-only.
    #[test]
    fn comparison_cli_requires_standalone_mode_for_output() {
        assert!(parse(&["--python", "/tmp/python"]).is_ok());
        assert!(parse(&["--output", "o.f16"]).is_err());
        assert!(parse(&["--rust-only", "--output", "o.f16"]).is_ok());
        assert!(parse(&["--rust-only", "--python", "/tmp/python"]).is_err());
    }

    /// Check that the comparison matrix accepts every supported static shape in each mode.
    #[test]
    fn all_cases_accepts_every_static_shape_and_selected_specialization() {
        for (seq_k, heads) in [(64, 8), (65, 3), (1024, 8), (4097, 8)] {
            for mode in ["auto", "runtime", "static"] {
                let args = parse(&[
                    "--all-cases",
                    "--seq-k",
                    &seq_k.to_string(),
                    "--heads",
                    &heads.to_string(),
                    "--specialization",
                    mode,
                ])
                .unwrap()
                .unwrap();
                assert!(args.all_cases);
                assert!(!args.shape.causal);
            }
        }
    }

    /// Ensure all-cases comparisons require a matching static kernel in every mode.
    #[test]
    fn all_cases_rejects_shapes_without_static_kernels_even_in_runtime_mode() {
        for (seq_k, heads) in [(1, 8), (64, 3), (65, 8), (1024, 3), (4097, 3)] {
            for mode in ["auto", "runtime", "static"] {
                let error = parse(&[
                    "--all-cases",
                    "--seq-k",
                    &seq_k.to_string(),
                    "--heads",
                    &heads.to_string(),
                    "--specialization",
                    mode,
                ])
                .unwrap_err();
                assert!(error.contains("static specialization"), "{error}");
            }
        }
    }

    /// Reject options that cannot participate in the decode comparison matrix.
    #[test]
    fn all_cases_rejects_standalone_execution_output_and_causal_attention() {
        for cli in [
            &["--all-cases", "--rust-only"][..],
            &["--rust-only", "--all-cases"],
            &["--all-cases", "--output", "o.f16"],
            &["--all-cases", "--causal"],
        ] {
            assert!(parse(cli).is_err(), "{cli:?}");
        }
    }

    /// Check CTA and scratch counts at cache lengths around split boundaries.
    #[test]
    fn launch_plan_covers_short_caches_and_split_tails() {
        for (seq_k, splits, blocks, scratch_elements) in [
            (1, 1, 3, 198),
            (63, 1, 3, 198),
            (64, 1, 3, 198),
            (65, 2, 6, 396),
            (127, 2, 6, 396),
            (128, 2, 6, 396),
            (129, 3, 9, 594),
            (1024, 16, 48, 3168),
            (4097, 65, 195, 12870),
        ] {
            let plan = LaunchPlan::new(shape(seq_k, 3)).unwrap();
            assert_eq!(
                (plan.splits, plan.partial_blocks, plan.scratch_elements),
                (splits, blocks, scratch_elements),
                "seq_k={seq_k}"
            );
        }
    }

    /// Reject empty dimensions, multiple queries, and causal decode.
    #[test]
    fn launch_plan_rejects_invalid_decode_workloads() {
        for invalid in [
            shape(0, 1),
            shape(1, 0),
            Shape {
                seq_q: 0,
                ..shape(64, 1)
            },
            Shape {
                seq_q: 64,
                ..shape(64, 1)
            },
            Shape {
                causal: true,
                ..shape(1, 1)
            },
        ] {
            assert!(LaunchPlan::new(invalid).is_err(), "{invalid:?}");
        }
    }

    /// Check both tensor and scratch indexing limits before allocation.
    #[test]
    fn launch_plan_rejects_input_and_scratch_index_overflow() {
        for invalid in [
            shape(usize::MAX, 1),
            shape(1, usize::MAX),
            shape(1, 67_108_864), // Q/K/V exceed u32 elements.
            shape(1, 65_075_263), // Tensors fit; the 66-element scratch slots do not.
        ] {
            assert!(LaunchPlan::new(invalid).is_err(), "{invalid:?}");
        }
        let plan = LaunchPlan::new(shape(1, 65_075_262)).unwrap();
        assert_eq!(plan.partial_blocks, 65_075_262);
        assert_eq!(plan.scratch_elements, 4_294_967_292);
    }

    /// Check defaults and the CLI path for verifying a short final split.
    #[test]
    fn cli_selects_decode_defaults_and_accepts_tail_verification() {
        let defaults = parse(&[]).unwrap().unwrap();
        assert!(!defaults.all_cases);
        assert_eq!(
            (
                defaults.shape.seq_q,
                defaults.shape.seq_k,
                defaults.shape.heads,
                defaults.shape.causal
            ),
            (1, 1024, 8, false)
        );
        let args = parse(&[
            "--seq-q",
            "1",
            "--seq-k",
            "65",
            "--heads",
            "3",
            "--noncausal",
            "--input-scale",
            "16",
            "--zero-query",
            "--device",
            "2",
            "--warmup",
            "7",
            "--iters",
            "0",
            "--rust-only",
            "--output",
            "o.f16",
            "--json",
            "r.json",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            (args.shape.seq_k, args.shape.heads, args.input_scale),
            (65, 3, 16)
        );
        assert!(args.zero_query);
        assert_eq!((args.device, args.warmup, args.iters), (2, 7, 0));
        assert_eq!(args.output, Some(PathBuf::from("o.f16")));
        assert_eq!(args.json, Some(PathBuf::from("r.json")));
        assert!(parse(&["--help"]).unwrap().is_none());
    }

    /// Reject invalid dimensions, values, masks, and unknown options.
    #[test]
    fn cli_rejects_unsupported_shapes_and_malformed_arguments() {
        for cli in [
            &["--causal"][..],
            &["--seq-q", "64"],
            &["--seq-k", "0"],
            &["--heads", "0"],
            &["--seq-k", "-1"],
            &["--iters", "-1"],
            &["--input-scale", "2"],
            &["--seq-k"],
            &["--unknown"],
            &["--splits", "2"],
            &["--device", "abc"],
            &["--causal", "--noncausal"],
            &["--noncausal", "--causal"],
        ] {
            assert!(parse(cli).is_err(), "{cli:?}");
        }
    }

    /// Check that every supported shape selects the requested static or runtime kernel.
    #[test]
    fn specialization_selects_only_exact_aot_shapes() {
        assert_eq!(
            parse(&[]).unwrap().unwrap().kernel_shape().unwrap(),
            (1024, 8)
        );
        for (seq_k, heads) in [(64, 8), (65, 3), (1024, 8), (4097, 8)] {
            for mode in ["auto", "static", "runtime"] {
                let args = parse(&[
                    "--seq-k",
                    &seq_k.to_string(),
                    "--heads",
                    &heads.to_string(),
                    "--specialization",
                    mode,
                ])
                .unwrap()
                .unwrap();
                let expected = if mode == "runtime" {
                    (0, 0)
                } else {
                    (seq_k, heads)
                };
                assert_eq!(args.kernel_shape().unwrap(), expected);
            }
        }
    }

    /// Allow runtime fallback in auto mode while rejecting unavailable static shapes.
    #[test]
    fn specialization_falls_back_only_when_permitted() {
        for (seq_k, heads) in [(1, 8), (64, 3), (65, 8), (1024, 3), (4097, 3)] {
            for mode in ["auto", "runtime"] {
                let args = parse(&[
                    "--seq-k",
                    &seq_k.to_string(),
                    "--heads",
                    &heads.to_string(),
                    "--specialization",
                    mode,
                ])
                .unwrap()
                .unwrap();
                assert_eq!(args.kernel_shape().unwrap(), (0, 0));
            }
            assert!(
                parse(&[
                    "--seq-k",
                    &seq_k.to_string(),
                    "--heads",
                    &heads.to_string(),
                    "--specialization",
                    "static"
                ])
                .is_err()
            );
        }
        for cli in [&["--specialization"][..], &["--specialization", "invalid"]] {
            assert!(parse(cli).is_err());
        }
    }

    /// Check that static dimensions are validated before a kernel launch.
    #[test]
    fn specialization_rejects_kernel_shape_mismatches_before_launch() {
        assert!(validate_kernel_shape::<0, 0>(shape(65, 3)).is_ok());
        assert!(validate_kernel_shape::<65, 3>(shape(65, 3)).is_ok());
        assert!(validate_kernel_shape::<65, 3>(shape(64, 3)).is_err());
        assert!(validate_kernel_shape::<65, 3>(shape(65, 8)).is_err());
        assert!(validate_kernel_shape::<0, 3>(shape(65, 3)).is_err());
        assert!(validate_kernel_shape::<65, 0>(shape(65, 3)).is_err());
    }
}
