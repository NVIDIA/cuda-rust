/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optional MIR passes selected by `CUDA_OXIDE_MIR_PASSES`.
//!
//! Entries declare the earliest compiler stage at which their IR contract is
//! valid. The driver validates the selected names once, then invokes the
//! selected entries at each declared stage. This deliberately keeps an early
//! formation pass separate from late MIR peepholes.

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    pass::{AnalysisManager, Pass, PassResult, Passes},
    result::Result,
};
use thiserror::Error;

type OptCtor = fn() -> Box<dyn Pass>;

/// Extension points in the dialect-MIR preparation pipeline.
///
/// `PrePreparation` sees imported MIR before mem2reg and annotated-loop
/// unrolling. It is the earliest existing hook for passes that require source
/// loop and local-allocation structure. `PostPreparation` is for ordinary SSA
/// cleanups after those standard transformations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirPassStage {
    PrePreparation,
    /// Runs after scalar promotion, while loop structure is still intact.
    PostMem2Reg,
    PostPreparation,
}

#[derive(Clone, Copy)]
struct OptEntry {
    name: &'static str,
    stage: MirPassStage,
    build: OptCtor,
}

/// A fully validated optional pass selection.
///
/// Keep this opaque: callers must select once before any transformation runs,
/// then request the pipeline for each stage in compiler order.
pub struct SelectedMirPasses(Vec<OptEntry>);

impl SelectedMirPasses {
    /// Whether any selected pass is declared for `stage`. Lets the driver skip
    /// a stage entirely (no pass-manager run, no extra module verification)
    /// when nothing was selected for it.
    pub fn has_stage(&self, stage: MirPassStage) -> bool {
        self.0.iter().any(|entry| entry.stage == stage)
    }
}

/// Errors from selecting a MIR pass pipeline.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MirPassPipelineError {
    #[error("empty opt name in pipeline")]
    EmptyName,
    #[error("unknown MIR pass \"{name}\"; available passes: {available}")]
    UnknownName { name: String, available: String },
    #[error("MIR pass \"{name}\" selected more than once in pipeline")]
    Duplicate { name: String },
}

/// The cuda-oxide-owned registry of staged optional MIR passes.
#[derive(Default)]
pub struct MirPassRegistry {
    entries: Vec<OptEntry>,
}

impl MirPassRegistry {
    /// Validate a comma-separated pipeline. Empty specs select no passes.
    ///
    /// The textual order is preserved among entries at the same stage. Stages
    /// themselves always execute in compiler order, regardless of where names
    /// appear in `spec`.
    pub fn select(
        &self,
        spec: &str,
    ) -> std::result::Result<SelectedMirPasses, MirPassPipelineError> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Ok(SelectedMirPasses(Vec::new()));
        }
        let entries = spec
            .split(',')
            .map(str::trim)
            .map(|name| self.lookup(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // A pass name may appear at most once. Running the same pass twice is
        // never what the user meant; reject the spec instead of guessing.
        for (position, entry) in entries.iter().enumerate() {
            if entries[..position]
                .iter()
                .any(|seen| seen.name == entry.name)
            {
                return Err(MirPassPipelineError::Duplicate {
                    name: entry.name.to_owned(),
                });
            }
        }

        Ok(SelectedMirPasses(entries))
    }

    /// Build the selected pipeline for one compiler stage.
    pub fn build_stage_pipeline(
        &self,
        selected: &SelectedMirPasses,
        stage: MirPassStage,
    ) -> Passes {
        let mut passes = Passes::default();
        for entry in selected.0.iter().filter(|entry| entry.stage == stage) {
            passes.add_pass(BoxedPass((entry.build)()));
        }
        passes
    }

    fn lookup(&self, name: &str) -> std::result::Result<OptEntry, MirPassPipelineError> {
        if name.is_empty() {
            return Err(MirPassPipelineError::EmptyName);
        }
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .copied()
            .ok_or_else(|| MirPassPipelineError::UnknownName {
                name: name.to_owned(),
                available: self
                    .entries
                    .iter()
                    .map(|entry| entry.name)
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }
}

/// Build the registry of supported optional CUDA Oxide MIR passes.
pub fn registry() -> MirPassRegistry {
    MirPassRegistry {
        entries: vec![OptEntry {
            name: "warp-aggregate-constant-fp-atomics",
            stage: MirPassStage::PostMem2Reg,
            build: crate::warp_aggregate_constant_fp_atomics::build_pass,
        }],
    }
}

struct BoxedPass(Box<dyn Pass>);

impl Pass for BoxedPass {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn run(
        &mut self,
        operation: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        self.0.run(operation, ctx, analyses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::ops::ModuleOp;
    use pliron::op::Op;
    use std::sync::Mutex;

    static RUNS: Mutex<Vec<&str>> = Mutex::new(Vec::new());
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    struct TestPass(&'static str);

    impl Pass for TestPass {
        fn name(&self) -> &str {
            self.0
        }

        fn run(
            &mut self,
            _operation: Ptr<Operation>,
            _ctx: &mut Context,
            _analyses: &mut AnalysisManager,
        ) -> Result<PassResult> {
            RUNS.lock().unwrap().push(self.0);
            Ok(PassResult::default())
        }
    }

    fn first() -> Box<dyn Pass> {
        Box::new(TestPass("first"))
    }

    fn second() -> Box<dyn Pass> {
        Box::new(TestPass("second"))
    }

    fn early() -> Box<dyn Pass> {
        Box::new(TestPass("early"))
    }

    fn middle() -> Box<dyn Pass> {
        Box::new(TestPass("middle"))
    }

    fn run(passes: &mut Passes) {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        passes
            .run(
                module.get_operation(),
                &mut ctx,
                &mut AnalysisManager::default(),
            )
            .unwrap();
    }

    fn registry_with_test_passes() -> MirPassRegistry {
        MirPassRegistry {
            entries: vec![
                OptEntry {
                    name: "first",
                    stage: MirPassStage::PostPreparation,
                    build: first,
                },
                OptEntry {
                    name: "second",
                    stage: MirPassStage::PostPreparation,
                    build: second,
                },
                OptEntry {
                    name: "early",
                    stage: MirPassStage::PrePreparation,
                    build: early,
                },
                OptEntry {
                    name: "middle",
                    stage: MirPassStage::PostMem2Reg,
                    build: middle,
                },
            ],
        }
    }

    #[test]
    fn production_registry_selects_the_warp_aggregated_fp_atomic_pass() {
        assert!(registry().select("").is_ok());
        assert!(
            registry()
                .select("warp-aggregate-constant-fp-atomics")
                .is_ok()
        );
        assert!(matches!(
            registry().select("missing"),
            Err(MirPassPipelineError::UnknownName { .. })
        ));
    }

    #[test]
    fn selected_passes_run_in_order() {
        let _serial = TEST_LOCK.lock().unwrap();
        RUNS.lock().unwrap().clear();
        let registry = registry_with_test_passes();
        let selected = registry.select("first,second").unwrap();
        let mut passes = registry.build_stage_pipeline(&selected, MirPassStage::PostPreparation);
        run(&mut passes);
        assert_eq!(*RUNS.lock().unwrap(), ["first", "second"]);
    }

    #[test]
    fn invalid_pipeline_does_not_run_a_prefix() {
        let _serial = TEST_LOCK.lock().unwrap();
        RUNS.lock().unwrap().clear();
        assert!(registry_with_test_passes().select("first,missing").is_err());
        assert!(matches!(
            registry_with_test_passes().select("first,"),
            Err(MirPassPipelineError::EmptyName)
        ));
        assert!(RUNS.lock().unwrap().is_empty());
    }

    #[test]
    fn duplicate_pass_name_is_rejected() {
        assert!(matches!(
            registry_with_test_passes().select("first,second,first"),
            Err(MirPassPipelineError::Duplicate { name }) if name == "first"
        ));
    }

    #[test]
    fn has_stage_reports_only_selected_stages() {
        let registry = registry_with_test_passes();
        let selected = registry.select("first,middle").unwrap();
        assert!(selected.has_stage(MirPassStage::PostPreparation));
        assert!(selected.has_stage(MirPassStage::PostMem2Reg));
        assert!(!selected.has_stage(MirPassStage::PrePreparation));
    }

    #[test]
    fn empty_pipeline_runs_nothing() {
        let _serial = TEST_LOCK.lock().unwrap();
        RUNS.lock().unwrap().clear();
        let registry = registry_with_test_passes();
        let selected = registry.select("").unwrap();
        let mut passes = registry.build_stage_pipeline(&selected, MirPassStage::PostPreparation);
        run(&mut passes);
        assert!(RUNS.lock().unwrap().is_empty());
    }

    #[test]
    fn selection_runs_only_entries_declared_for_each_stage() {
        let _serial = TEST_LOCK.lock().unwrap();
        RUNS.lock().unwrap().clear();
        let registry = registry_with_test_passes();
        let selected = registry.select("first,early,middle,second").unwrap();

        let mut pre = registry.build_stage_pipeline(&selected, MirPassStage::PrePreparation);
        run(&mut pre);
        let mut middle = registry.build_stage_pipeline(&selected, MirPassStage::PostMem2Reg);
        run(&mut middle);
        let mut post = registry.build_stage_pipeline(&selected, MirPassStage::PostPreparation);
        run(&mut post);

        assert_eq!(
            *RUNS.lock().unwrap(),
            ["early", "middle", "first", "second"]
        );
    }
}
