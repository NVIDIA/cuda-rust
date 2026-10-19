/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pins the public shape of target-selection failures.
//!
//! Target and feature-floor rejections are decided before `llc` is reached, so
//! they belong to [`CompilationStage::Input`]. They used to arrive as
//! `PipelineError::PtxGeneration`, which the standalone API surfaced as a
//! codegen failure and phrased in terms of the `CUDA_OXIDE_TARGET` environment
//! variable that a standalone caller never sets.

use cuda_oxide_codegen::__private::PipelineError;
use cuda_oxide_codegen::experimental::{CompilationStage, CompileError};

#[test]
fn target_selection_maps_to_an_input_stage_compile_error() {
    let error = CompileError::from(PipelineError::TargetSelection {
        target: "sm_75".to_string(),
        reason: "CUDA target sm_75 cannot lower detected feature Sm80".to_string(),
    });

    let CompileError::TargetSelection { target, reason } = &error else {
        panic!("expected CompileError::TargetSelection, got {error:?}");
    };
    assert_eq!(target, "sm_75");
    assert_eq!(
        reason,
        "CUDA target sm_75 cannot lower detected feature Sm80"
    );
    assert_eq!(error.stage(), CompilationStage::Input);
}

#[test]
fn a_rejected_target_is_not_reported_as_an_unparsable_one() {
    let error = CompileError::from(PipelineError::TargetSelection {
        target: "sm_75".to_string(),
        reason: "CUDA target sm_75 cannot lower detected feature Sm80".to_string(),
    });

    // sm_75 parses. Rendering the reason under an "invalid CUDA target
    // `sm_75`" prefix would call a valid architecture invalid and name it
    // twice.
    let text = error.to_string();
    assert_eq!(text, "CUDA target sm_75 cannot lower detected feature Sm80");
    assert!(!text.contains("invalid CUDA target"), "{text}");
    assert_eq!(text.matches("sm_75").count(), 1, "{text}");
}

#[test]
fn an_unparsable_target_still_maps_to_the_input_stage() {
    let error = CompileError::from(PipelineError::TargetSelection {
        target: "not-a-target".to_string(),
        reason: "invalid CUDA target `not-a-target`: unrecognised architecture".to_string(),
    });
    assert_eq!(error.stage(), CompilationStage::Input);
    assert!(error.to_string().starts_with("invalid CUDA target"));
}

#[test]
fn target_parse_failures_from_the_typed_api_keep_their_own_variant() {
    let error = cuda_oxide_codegen::experimental::Target::parse("sm_bogus").unwrap_err();
    assert!(
        matches!(error, CompileError::InvalidTarget { .. }),
        "Target::parse still reports a parse failure, got {error:?}"
    );
    assert_eq!(error.stage(), CompilationStage::Input);
}
