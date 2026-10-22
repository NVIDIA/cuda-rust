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

/// Preparation state at the boundary before optional post-mem2reg passes.
/// The same analysis manager continues through the remaining preparation.
/// Library hooks retain their typed operations here, including in Full debug;
/// finishing lowers them before optional passes and unrolling.
pub(crate) struct MirPreparationStage {
    selected_passes: Option<SelectedMirPasses>,
    pub(crate) analyses: pliron::pass::AnalysisManager,
    verbose: bool,
}

/// Verify and prepare dialect MIR through post-mem2reg result forwarding.
/// Full variable debug retains unpromoted locals at this same pause.
///
/// The one shared post-translation orchestrator calls this helper for both the
/// rustc and standalone frontends.
#[doc(hidden)]
pub(crate) fn begin_mir_preparation(
    ctx: &mut Context,
    module: Ptr<Operation>,
    preparation: MirPreparation<'_>,
) -> Result<MirPreparationStage, PipelineError> {
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
        return Ok(MirPreparationStage {
            selected_passes: None,
            analyses: pliron::pass::AnalysisManager::default(),
            verbose: preparation.verbose,
        });
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

    Ok(MirPreparationStage {
        selected_passes: Some(selected_passes),
        analyses,
        verbose: preparation.verbose,
    })
}

impl MirPreparationStage {
    /// Resume the remaining preparation on the module owned by the pipeline stage.
    pub(crate) fn finish(
        self,
        ctx: &mut Context,
        module: Ptr<Operation>,
    ) -> Result<(), PipelineError> {
        let mut analyses = self.analyses;
        // Native calls retain their source identity through the proof boundary.
        // Binding precedes optional post-mem2reg transforms and unrolling;
        // Full variable debug uses the same lowering over unpromoted locals.
        mir_transforms::library::lower_library_operations(module, ctx, &mut analyses).map_err(
            |error| PipelineError::Verification {
                name: "library operation lowering".to_string(),
                message: error.disp(ctx).to_string(),
                operation: None,
            },
        )?;
        verify_operation(ctx, module, "module post-library-operation-lowering")?;
        let Some(selected_passes) = self.selected_passes else {
            return Ok(());
        };
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
            canonicalize_bounded_borrowed_pointer_arguments(module, ctx, self.verbose);
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

    /// A source local feeding an otherwise eligible optional atomic transform.
    fn local_atomic_module(ctx: &mut Context) -> (Ptr<Operation>, Ptr<Operation>, Ptr<Operation>) {
        use dialect_mir::ops::{
            MirAllocaOp, MirFloatConstantOp, MirFuncOp, MirLoadOp, MirReturnOp, MirStoreOp,
        };
        use dialect_mir::types::MirPtrType;
        use dialect_nvvm::ops::atomic::{
            AtomicOrdering, AtomicRmwKind, AtomicScope, NvvmAtomicRmwOp,
        };
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::{
            attributes::{FPSingleAttr, StringAttr, TypeAttr},
            op_interfaces::SymbolOpInterface,
            types::{FP32Type, FunctionType},
        };
        dialect_mir::register(ctx);
        dialect_nvvm::register(ctx);
        let module = ModuleOp::new(ctx, "staged".try_into().unwrap()).get_operation();
        let float = FP32Type::get(ctx);
        let ptr = MirPtrType::get_generic(ctx, float.into(), true);
        let function_ty = FunctionType::get(ctx, vec![ptr.into()], vec![float.into()]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(ctx, function, TypeAttr::new(function_ty.into()))
            .set_symbol_name(ctx, "staged".try_into().unwrap());
        let entry = BasicBlock::new(ctx, None, vec![ptr.into()]);
        entry.insert_at_back(function.deref(ctx).get_region(0), ctx);
        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![ptr.into()],
            vec![],
            vec![],
            0,
        );
        alloca.deref_mut(ctx).attributes.set(
            "cuda_oxide_debug_local_name".try_into().unwrap(),
            StringAttr::new("value".into()),
        );
        alloca.insert_at_back(entry, ctx);
        let constant = Operation::new(
            ctx,
            MirFloatConstantOp::get_concrete_op_info(),
            vec![float.into()],
            vec![],
            vec![],
            0,
        );
        MirFloatConstantOp::new(constant).set_attr_float_value(ctx, FPSingleAttr::from(1.0));
        constant.insert_at_back(entry, ctx);
        let slot = alloca.deref(ctx).get_result(0);
        let value = constant.deref(ctx).get_result(0);
        Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, value],
            vec![],
            0,
        )
        .insert_at_back(entry, ctx);
        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![float.into()],
            vec![slot],
            vec![],
            0,
        );
        load.insert_at_back(entry, ctx);
        let loaded = load.deref(ctx).get_result(0);
        let output = entry.deref(ctx).get_argument(0);
        NvvmAtomicRmwOp::build(
            ctx,
            output,
            loaded,
            float.into(),
            AtomicRmwKind::FAdd,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, ctx);
        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![loaded],
            vec![],
            0,
        )
        .insert_at_back(entry, ctx);
        crate::lower::append_to_module(ctx, module, function);
        (module, function, alloca)
    }

    #[test]
    fn pause_promotes_locals_before_running_optional_passes() {
        use pliron::linked_list::ContainsLinkedList;
        let mut ctx = Context::new();
        let (module, function, alloca) = local_atomic_module(&mut ctx);
        let stage = begin_mir_preparation(
            &mut ctx,
            module,
            MirPreparation {
                promote_and_unroll: true,
                verbose: false,
                mir_pass_pipeline: Some("warp-aggregate-constant-fp-atomics"),
            },
        )
        .unwrap();
        assert!(
            alloca.try_deref(&ctx).is_err(),
            "local must already be promoted at the pause"
        );
        let count_matches = |ctx: &Context| {
            function
                .deref(ctx)
                .get_region(0)
                .deref(ctx)
                .iter(ctx)
                .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
                .filter(|&op| {
                    Operation::get_op::<dialect_nvvm::ops::MatchAnySyncI64Op>(op, ctx).is_some()
                })
                .count()
        };
        assert_eq!(
            count_matches(&ctx),
            0,
            "optional pass must not run before the pause"
        );
        stage.finish(&mut ctx, module).unwrap();
        assert_eq!(
            count_matches(&ctx),
            1,
            "consuming finish must run the selected optional pass"
        );
    }

    #[test]
    fn full_debug_stage_preserves_local_slots_through_finish() {
        let mut ctx = Context::new();
        let (module, _, alloca) = local_atomic_module(&mut ctx);
        let original = module.deref(&ctx).disp(&ctx).to_string();
        let stage = begin_mir_preparation(
            &mut ctx,
            module,
            MirPreparation {
                promote_and_unroll: false,
                verbose: false,
                mir_pass_pipeline: None,
            },
        )
        .unwrap();
        assert!(alloca.try_deref(&ctx).is_ok());
        assert_eq!(module.deref(&ctx).disp(&ctx).to_string(), original);
        stage.finish(&mut ctx, module).unwrap();
        assert!(alloca.try_deref(&ctx).is_ok());
        assert_eq!(module.deref(&ctx).disp(&ctx).to_string(), original);
    }

    #[test]
    fn debug_mode_rejects_requested_mir_passes() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let error = begin_mir_preparation(
            &mut ctx,
            module.get_operation(),
            MirPreparation {
                promote_and_unroll: false,
                verbose: false,
                mir_pass_pipeline: Some("future-pass"),
            },
        )
        .err()
        .expect("invalid pipeline must fail before a stage is returned");
        assert!(matches!(error, PipelineError::InvalidMirPassPipeline(_)));
    }

    #[test]
    fn invalid_staged_pipeline_is_rejected_before_preparation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let error = begin_mir_preparation(
            &mut ctx,
            module.get_operation(),
            MirPreparation {
                promote_and_unroll: true,
                verbose: false,
                mir_pass_pipeline: Some("missing-pass"),
            },
        )
        .err()
        .expect("invalid pipeline must fail before a stage is returned");
        assert!(matches!(error, PipelineError::InvalidMirPassPipeline(_)));
    }
}
