/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::mir_pass_registry::{MirPassStage, SelectedMirPasses};
use crate::verify::verify_operation;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::printable::Printable;

/// Controls the reusable dialect-mir preparation stage.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MirPreparation<'a> {
    /// Promote stack slots to SSA and run annotation-driven loop unrolling.
    pub promote_and_unroll: bool,
    /// Print preparation-pass progress notes to stderr. Threaded from the
    /// pipeline's `BackendOptions`; the scalarization passes read this flag
    /// instead of the environment (loop unrolling still checks
    /// `CUDA_OXIDE_VERBOSE` on its own).
    pub verbose: bool,
    /// Optional pass pipeline; `None` or empty preserves the defaults.
    pub mir_pass_pipeline: Option<&'a str>,
}

/// Verify and prepare a dialect-mir module before LLVM lowering.
///
/// The one shared post-translation orchestrator calls this helper for both the
/// rustc and standalone frontends.
#[doc(hidden)]
pub fn prepare_mir_module(
    ctx: &mut Context,
    module: Ptr<Operation>,
    preparation: MirPreparation<'_>,
) -> Result<(), PipelineError> {
    verify_operation(ctx, module, "module")?;
    let has_pass_pipeline = preparation
        .mir_pass_pipeline
        .is_some_and(|pipeline| !pipeline.trim().is_empty());
    if !preparation.promote_and_unroll {
        if has_pass_pipeline {
            return Err(PipelineError::InvalidMirPassPipeline(
                "optional MIR passes are unavailable with full variable debug info".to_string(),
            ));
        }
        return Ok(());
    }

    // Validate every requested pass before any transformation runs. This keeps
    // an invalid later-stage name from leaving a module partially transformed.
    let selected_passes = select_optional_mir_passes(preparation.mir_pass_pipeline)?;

    let mut analyses = pliron::pass::AnalysisManager::default();
    run_optional_mir_passes(
        ctx,
        module,
        &selected_passes,
        MirPassStage::PrePreparation,
        &mut analyses,
    )?;

    // Compiler-owned multi-result device operations are temporarily adapted
    // to Rust aggregate return values by the MIR importer. Prove and remove
    // that ABI-only boundary before mem2reg so the independent register values
    // remain SSA all the way into LLVM/PTX lowering.
    mir_transforms::forward_compiler_result_bundles::forward_compiler_result_bundles(
        module,
        ctx,
        &mut analyses,
        preparation.verbose,
    )
    .map_err(|error| PipelineError::Verification {
        name: "compiler-result forwarding".to_string(),
        message: error.disp(ctx).to_string(),
        operation: None,
    })?;
    verify_operation(ctx, module, "module post-compiler-result-forwarding")?;

    // A by-value aggregate argument initially lives in a MIR alloca. Read-only
    // field/index projections make that alloca non-promotable even though the
    // original entry-block argument is already an SSA value. Canonicalize the
    // validated pointer chains back to value extraction before mem2reg.
    mir_transforms::scalarize_borrowed_aggregate_reads::canonicalize_read_only_aggregate_arguments(
        module,
        ctx,
        preparation.verbose,
    );
    verify_operation(
        ctx,
        module,
        "module post-borrowed-aggregate-read-canonicalization",
    )?;
    pliron::opts::mem2reg::mem2reg(module, ctx, &mut analyses).map_err(|error| {
        PipelineError::Verification {
            name: "mem2reg".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        }
    })?;
    verify_operation(ctx, module, "module post-mem2reg")?;

    // Source-level dynamic indices commonly pass through temporary MIR slots.
    // After mem2reg, bounded forms such as `index % C` are direct SSA values,
    // while result-bundle allocas blocked by dynamic array projections remain.
    // Re-run the same fail-closed forwarding proof here so those bundles can
    // be recovered without teaching the pass to trace arbitrary stack slots.
    mir_transforms::forward_compiler_result_bundles::forward_compiler_result_bundles(
        module,
        ctx,
        &mut analyses,
        preparation.verbose,
    )
    .map_err(|error| PipelineError::Verification {
        name: "post-mem2reg compiler-result forwarding".to_string(),
        message: error.disp(ctx).to_string(),
        operation: None,
    })?;
    verify_operation(
        ctx,
        module,
        "module post-mem2reg-compiler-result-forwarding",
    )?;

    // Formation passes that need promoted SSA values but must still see the
    // original loop CFG run here. In particular, a reduction formation pass
    // cannot safely infer a source loop once generic unrolling has cloned it.
    run_optional_mir_passes(
        ctx,
        module,
        &selected_passes,
        MirPassStage::PostMem2Reg,
        &mut analyses,
    )?;

    // An immutable aggregate pointer argument in an always-inline helper can
    // still retain dynamic field/array pointer chains after mem2reg. Recover
    // bounded read-only accesses in typed MIR before LLVM lowering.
    mir_transforms::scalarize_borrowed_aggregate_reads::
        canonicalize_bounded_borrowed_pointer_arguments(module, ctx, preparation.verbose);
    verify_operation(
        ctx,
        module,
        "module post-borrowed-pointer-read-canonicalization",
    )?;

    mir_transforms::unroll::unroll_annotated_loops(module, ctx, &mut analyses).map_err(
        |error| PipelineError::Verification {
            name: "loop-unroll".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        },
    )?;
    verify_operation(ctx, module, "module post-unroll")?;

    run_optional_mir_passes(
        ctx,
        module,
        &selected_passes,
        MirPassStage::PostPreparation,
        &mut analyses,
    )
}

fn select_optional_mir_passes(spec: Option<&str>) -> Result<SelectedMirPasses, PipelineError> {
    crate::mir_pass_registry::registry()
        .select(spec.unwrap_or_default())
        .map_err(|error| PipelineError::InvalidMirPassPipeline(error.to_string()))
}

fn run_optional_mir_passes(
    ctx: &mut Context,
    module: Ptr<Operation>,
    selected: &SelectedMirPasses,
    stage: MirPassStage,
    analyses: &mut pliron::pass::AnalysisManager,
) -> Result<(), PipelineError> {
    // Nothing selected for this stage: skip the pass-manager run and the extra
    // module verification so a default build pays nothing for the hooks.
    if !selected.has_stage(stage) {
        return Ok(());
    }

    let mut passes = crate::mir_pass_registry::registry().build_stage_pipeline(selected, stage);

    <pliron::pass::Passes as pliron::pass::PassManager>::run_pass(
        &mut passes,
        module,
        ctx,
        analyses,
    )
    .map_err(|error| PipelineError::Verification {
        name: format!("optional MIR passes ({stage:?})"),
        message: error.disp(ctx).to_string(),
        operation: None,
    })?;

    verify_operation(
        ctx,
        module,
        &format!("module post-optional-mir-passes ({stage:?})"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::ops::ModuleOp;
    use pliron::op::Op;

    #[test]
    fn debug_mode_rejects_requested_mir_passes() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let error = prepare_mir_module(
            &mut ctx,
            module.get_operation(),
            MirPreparation {
                promote_and_unroll: false,
                verbose: false,
                mir_pass_pipeline: Some("future-pass"),
            },
        )
        .unwrap_err();
        assert!(matches!(error, PipelineError::InvalidMirPassPipeline(_)));
    }

    #[test]
    fn invalid_staged_pipeline_is_rejected_before_preparation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let error = prepare_mir_module(
            &mut ctx,
            module.get_operation(),
            MirPreparation {
                promote_and_unroll: true,
                verbose: false,
                mir_pass_pipeline: Some("missing-pass"),
            },
        )
        .unwrap_err();
        assert!(matches!(error, PipelineError::InvalidMirPassPipeline(_)));
    }
}
