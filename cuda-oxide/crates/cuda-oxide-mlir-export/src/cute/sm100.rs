/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Map SM100 CuTe plans to operations accepted by the pinned CUTLASS 4.7 compiler.
//!
//! Allocation, tensor-memory copies, cluster TMA, and shared-memory descriptors
//! use native CuTe operations. CUTLASS 4.7's `sm100.mma` atom cannot select when
//! to fill or release the A collector, so MMA uses an NVVM TCGen05 instruction.
//! CuTe layouts supply its descriptors; the verified tiled-MMA plan supplies
//! its shape and collector sequence:
//!
//! ```text
//! each K16 step: first N partition -> fill A collector
//!               second partition -> last use of A collector
//! ```

#[cfg(test)]
use crate::cute::mma::sm100::instruction_descriptor;
use crate::cute::{
    emit::{Emit, dialect, integer_property, llvm_ptr_type, property, segments},
    epilogue::sm100::emit_epilogue,
    mma::sm100::emit_mma,
    pipeline::sm100::emit_pipeline,
};
use crate::{
    DropAttribute, MlirAttribute, MlirOperation, MlirType, OperationInput, OperationTranslation,
    TranslationError, TranslationRegistry, TranslationSession,
};
#[cfg(test)]
use dialect_cute::attributes::CuteComposedLayoutAttr;
use dialect_cute::{attributes::CuteTmaStorePipelineAttr, sm100_ops::*};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};

/// Register paired SM100 GEMM plans and their tensor-memory, copy, and pipeline recipes.
pub(crate) fn register_cute_sm100_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<CuteSm100TmemPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100MmaPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100EpiloguePlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100ClusterCopyPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100PipelinePlanAttr>(DropAttribute)?;
    /// Bind a typed SM100 source operation to its verified plan reader and runtime action.
    macro_rules! register {
        ($op:ty, $kind:ident, $plan:ident) => {
            registry.register_operation::<$op>(Recipe {
                kind: Kind::$kind,
                read: |ctx, source| {
                    let op = <$op>::wrap(source);
                    op.verify(ctx).map_err(|e| e.to_string())?;
                    op.plan(ctx)
                        .map(Plan::$plan)
                        .ok_or_else(|| "SM100 operation lost its verified plan".into())
                },
            })?;
        };
    }
    register!(CuteSm100TmemAllocOp, Alloc, Tmem);
    register!(CuteSm100TmemDeallocOp, Dealloc, Tmem);
    register!(CuteSm100TiledMmaOp, Mma, Mma);
    register!(CuteSm100TmemEpilogueOp, Epilogue, Epilogue);
    register!(CuteSm100ClusterTmaLoadOp, TmaLoad, Copy);
    register!(CuteSm100PipelineInitOp, PipelineInit, Pipeline);
    register!(CuteSm100PipelineAcquireOp, PipelineAcquire, Pipeline);
    register!(CuteSm100PipelineExpectOp, PipelineExpect, Pipeline);
    register!(CuteSm100PipelineWaitOp, PipelineWait, Pipeline);
    register!(CuteSm100PipelineReleaseOp, PipelineRelease, Pipeline);
    register!(CuteSm100AccumulatorInitOp, AccInit, Pipeline);
    register!(CuteSm100AccumulatorAcquireOp, AccAcquire, Pipeline);
    register!(CuteSm100AccumulatorCommitOp, AccCommit, Pipeline);
    register!(CuteSm100AccumulatorWaitOp, AccWait, Pipeline);
    register!(CuteSm100AccumulatorReleaseOp, AccRelease, Pipeline);
    register!(CuteSm100TmaStoreOp, TmaStore, Epilogue);
    register!(CuteSm100StoreCommitOp, StoreCommit, Store);
    register!(CuteSm100StoreAcquireOp, StoreAcquire, Store);
    register!(CuteSm100StoreTailOp, StoreTail, Store);
    Ok(())
}

/// Select the runtime action after the source plan has been verified.
#[derive(Clone, Copy)]
pub(in crate::cute) enum Kind {
    /// Allocate the plan's tensor-memory columns.
    Alloc,
    /// Release the previously allocated tensor-memory columns.
    Dealloc,
    /// Issue paired FP16 MMA with the plan's A-collector sequence.
    Mma,
    /// Move the accumulator through bias/conversion into shared output storage.
    Epilogue,
    /// Issue the plan's cluster TMA load and completion routing.
    TmaLoad,
    /// Initialize the staged load pipeline's full and empty barriers.
    PipelineInit,
    /// Wait for an empty load stage before issuing its copy.
    PipelineAcquire,
    /// Announce transaction bytes on the load stage's full barrier.
    PipelineExpect,
    /// Wait until the load stage's data is available to its consumers.
    PipelineWait,
    /// Commit consumption so the load stage can be reused.
    PipelineRelease,
    /// Initialize the accumulator pipeline's readiness and reuse barriers.
    AccInit,
    /// Wait until the accumulator storage is available to the MMA producer.
    AccAcquire,
    /// Commit MMA completion to the accumulator's ready barrier.
    AccCommit,
    /// Wait for the accumulator before reading tensor memory.
    AccWait,
    /// Order every lane's reads before one elected arrival permits accumulator reuse.
    AccRelease,
    /// Issue the plan's shared-to-global TMA output copy.
    TmaStore,
    /// Commit issued TMA stores as one bulk group.
    StoreCommit,
    /// Wait until the store queue has capacity for the next group.
    StoreAcquire,
    /// Drain all outstanding TMA store groups.
    StoreTail,
}
/// Verified source plan retained until its runtime action has been emitted.
enum Plan {
    /// Tensor-memory allocation extent and ownership contract.
    Tmem(#[doc = "Allocation extent and tensor-memory ownership plan."] CuteSm100TmemPlanAttr),
    /// Verified paired-MMA shape, shared layouts, and collector-lifetime plan.
    Mma(#[doc = "Paired-MMA shape, layouts, and collector-lifetime plan."] CuteSm100MmaPlanAttr),
    /// Verified accumulator shape, conversion, and shared-output placement plan.
    Epilogue(
        #[doc = "Accumulator conversion and shared-output placement plan."]
        CuteSm100EpiloguePlanAttr,
    ),
    /// Cluster-copy layout, destination, and completion-routing contract.
    Copy(
        #[doc = "Cluster-copy shape, layout, and completion-routing plan."]
        CuteSm100ClusterCopyPlanAttr,
    ),
    /// Load or accumulator barrier counts, stages, and completion routing.
    Pipeline(
        #[doc = "Barrier stages, arrivals, byte counts, and cluster routing."]
        CuteSm100PipelinePlanAttr,
    ),
    /// Maximum permitted number of pending TMA store groups.
    Store(#[doc = "Bound on outstanding hardware TMA store groups."] CuteTmaStorePipelineAttr),
}
/// Pair a runtime action with the source-plan reader that validates its operands.
struct Recipe {
    /// Runtime action implemented by this recipe.
    kind: Kind,
    /// Read and verify the operation's typed plan before target emission.
    read: fn(&Context, Ptr<Operation>) -> Result<Plan, String>,
}

impl OperationTranslation for Recipe {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let plan = (self.read)(ctx, source)?;
        if !input.results.is_empty()
            || !input.attributes.is_empty()
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err(
                "SM100 semantic operation retained unexpected results, attributes or regions"
                    .into(),
            );
        }
        let mut e = Emit {
            session,
            location: input.location,
            ops: vec![],
        };
        let a = input.operands;
        use Kind::*;
        match (self.kind, plan) {
            (Alloc, Plan::Tmem(p)) => {
                let columns = e.constant(32, p.columns as i128)?;
                let token = e.cute_ptr(a[0].clone(), "i32", "smem", 4, None)?;
                let mut alloc =
                    e.operation("cute_nvgpu.arch.sm100.alloc_tmem", vec![columns, token])?;
                alloc
                    .properties
                    .insert("is_two_cta".into(), MlirAttribute::Unit);
                e.ops.push(alloc);
                let mut relinquish =
                    e.operation("cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit", vec![])?;
                relinquish
                    .properties
                    .insert("is_two_cta".into(), MlirAttribute::Unit);
                e.ops.push(relinquish);
            }
            (Dealloc, Plan::Tmem(p)) => {
                let ptr = e.cast(
                    "builtin.unrealized_conversion_cast",
                    a[0].clone(),
                    dialect("!cute.ptr<f32,tmem,align<16>>")?,
                )?;
                let columns = e.constant(32, p.columns as i128)?;
                let mut op =
                    e.operation("cute_nvgpu.arch.sm100.dealloc_tmem", vec![ptr, columns])?;
                op.properties
                    .insert("is_two_cta".into(), MlirAttribute::Unit);
                e.ops.push(op);
            }
            (Mma, Plan::Mma(p)) => emit_mma(&mut e, &a, &p)?,
            (Epilogue, Plan::Epilogue(p)) => emit_epilogue(&mut e, &a, &p)?,
            (TmaLoad, Plan::Copy(p)) => {
                let desc = e.cute_ptr(
                    a[0].clone(),
                    "!cute_nvgpu.tma_descriptor_tiled",
                    "generic",
                    64,
                    None,
                )?;
                let dst = e.cute_ptr(a[1].clone(), "f16", "smem", 128, None)?;
                let rank = e.constant(32, p.leader_rank as i128)?;
                let local_bar = e.llvm_ptr(a[4].clone(), 3)?;
                let remote = e.value("nvvm.mapa", vec![local_bar, rank], llvm_ptr_type(7)?)?;
                // The two-CTA TMA intrinsic encodes the leader's barrier as a
                // shared address. Retype its bits, retaining the mapa rank.
                let bits = e.cast("llvm.ptrtoint", remote, MlirType::Integer(32))?;
                let bar_ptr = e.cast("llvm.inttoptr", bits, llvm_ptr_type(3)?)?;
                let barrier = e.cute_ptr(bar_ptr, "i64", "smem", 8, None)?;
                let one = e.constant(32, 1)?;
                let mask32 = e.binary("arith.shli", one, a[5].clone())?;
                let mask = e.cast("arith.trunci", mask32, MlirType::Integer(16))?;
                let mut op = e.operation(
                    "cute_nvgpu.arch.copy.SM100.tma_load",
                    vec![desc, dst, barrier, a[3].clone(), a[2].clone(), mask],
                )?;
                property(&mut op, "mode", "#cute_nvgpu.tma_load_mode<tiled>")?;
                integer_property(&mut op, "num_cta", 32, p.cta_group as i128);
                segments(&mut op, &[1, 1, 1, 2, 1, 0, 0]);
                e.ops.push(op);
            }
            (TmaStore, Plan::Epilogue(_)) => {
                let desc = e.cute_ptr(
                    a[0].clone(),
                    "!cute_nvgpu.tma_descriptor_tiled",
                    "generic",
                    64,
                    None,
                )?;
                let src = e.cute_ptr(a[1].clone(), "f16", "smem", 128, None)?;
                let mut op = e.operation(
                    "cute_nvgpu.arch.copy.SM100.tma_store",
                    vec![desc, src, a[3].clone(), a[2].clone()],
                )?;
                property(&mut op, "mode", "#cute_nvgpu.tma_store_mode<tiled>")?;
                segments(&mut op, &[1, 1, 2, 0]);
                e.ops.push(op);
            }
            (StoreCommit, Plan::Store(_)) => e.emit("nvvm.cp.async.bulk.commit.group", vec![])?,
            (StoreAcquire | StoreTail, Plan::Store(p)) => {
                let groups = if matches!(self.kind, StoreTail) {
                    0
                } else {
                    p.stages - 1
                };
                let mut op = e.operation("nvvm.cp.async.bulk.wait_group", vec![])?;
                integer_property(&mut op, "group", 32, groups as i128);
                op.properties.insert("read".into(), MlirAttribute::Unit);
                e.ops.push(op);
            }
            (kind, Plan::Pipeline(p)) => emit_pipeline(&mut e, kind, &a, &p)?,
            _ => return Err("SM100 operation/plan mismatch".into()),
        }
        Ok(e.ops)
    }
}

#[cfg(test)]
mod tests {
    //! Paired-GEMM mapping tests for collector lifetime, vector epilogues, and accumulator release.

    use super::*;
    use crate::{CutlassFullCuteMlir22, MlirConsumerProfile};
    use dialect_cute::layout::{ComposedLayout, IntTuple, OffsetUnit, Swizzle};
    use dialect_mir::{
        ops::{MirFuncOp, MirReturnOp},
        types::{MirFP16Type, MirPtrType, address_space},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::TypeAttr,
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FP32Type, FunctionType, IntegerType, Signedness},
        },
        op::Op,
        r#type::TypeHandle,
        value::Value,
    };

    /// Build the canonical row-major FP16 composed layout with a selected swizzle width.
    fn layout(rows: u32, cols: u32, bits: u32) -> CuteComposedLayoutAttr {
        CuteComposedLayoutAttr(
            ComposedLayout::new(
                Swizzle::new(bits, 3, 3),
                0,
                format!("({rows},{cols}):({cols},1)").parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
        )
    }
    /// Build the paired-MMA fixture plan for two independently selected N partitions.
    fn mma_plan(n0: u32, n1: u32) -> CuteSm100MmaPlanAttr {
        CuteSm100MmaPlanAttr {
            m: 256,
            n0,
            n1,
            k: 64,
            tmem_columns: 512,
            cta_group: 2,
            collector: CuteSm100CollectorAttr::AFillLastUse,
            a_layout: layout(128, 64, 3),
            b0_layout: layout(n0 / 2, 64, 3),
            b1_layout: layout(n1 / 2, 64, 3),
        }
    }
    /// Create a typed SM100 function with the runtime operands used by individual recipe tests.
    fn fixture() -> (Context, ModuleOp, Ptr<BasicBlock>, Vec<Value>) {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let u32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let i1: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
        let f32: TypeHandle = FP32Type::get(&ctx).into();
        let f16: TypeHandle = MirFP16Type::get(&ctx).into();
        let halfptr: TypeHandle =
            MirPtrType::get(&mut ctx, f16, true, address_space::SHARED).into();
        let barptr: TypeHandle = MirPtrType::get(&mut ctx, u64, true, address_space::SHARED).into();
        let tokenptr: TypeHandle =
            MirPtrType::get(&mut ctx, u32, true, address_space::SHARED).into();
        let args = vec![
            u32, halfptr, halfptr, halfptr, i1, barptr, u32, u32, f32, tokenptr,
        ];
        let module = ModuleOp::new(&mut ctx, "sm100".try_into().unwrap());
        let ft = FunctionType::get(&ctx, args.clone(), vec![]);
        let func = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let f = MirFuncOp::new(&mut ctx, func, TypeAttr::new(ft.into()));
        f.set_symbol_name(&mut ctx, "sm100_probe".try_into().unwrap());
        module.append_operation(&mut ctx, func, 0);
        let block = BasicBlock::new(&mut ctx, None, args);
        block.insert_at_back(func.deref(&ctx).get_region(0), &ctx);
        let values = block.deref(&ctx).arguments().collect();
        (ctx, module, block, values)
    }
    /// Terminate and render the fixture while intentionally bypassing whole-kernel lifecycle checks.
    fn render(ctx: &mut Context, module: &ModuleOp, block: Ptr<BasicBlock>) -> String {
        let ret = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret.insert_at_back(block, ctx);
        let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
        let registry = profile.build_registry().unwrap();
        // Mapping-unit tests deliberately bypass whole-kernel lifecycle checks;
        // each source operation is still verified by its own translation.
        let target =
            crate::translate_module(ctx, module, &registry, &profile.translation_config()).unwrap();
        crate::render_module(&target).unwrap()
    }
    /// Check each paired-MMA partition uses plan-derived descriptors and the intended collector transition.
    #[test]
    fn paired_mma_derives_native_layout_descriptors_and_collector_lifetimes() {
        for (n0, n1) in [(192, 160), (128, 224)] {
            let (mut ctx, module, block, a) = fixture();
            let op = CuteSm100TiledMmaOp::new(&mut ctx, a[..5].to_vec(), mma_plan(n0, n1));
            op.get_operation().insert_at_back(block, &ctx);
            let text = render(&mut ctx, &module, block);
            assert_eq!(
                text.matches("\"cute_nvgpu.make_umma_smem_desc\"").count(),
                3,
                "{text}"
            );
            assert!(text.contains("S<3,4,3>"), "{text}");
            assert!(
                text.contains(&format!("(({},16),1):((64,1),0)", n0 / 2)),
                "{text}"
            );
            assert!(
                text.contains(&format!("(({},16),1):((64,1),0)", n1 / 2)),
                "{text}"
            );
            assert_eq!(text.matches("\"nvvm.tcgen05.mma\"").count(), 8, "{text}");
            assert_eq!(
                text.matches("#nvvm.tcgen05_mma_collectorop<fill>").count(),
                4,
                "{text}"
            );
            assert_eq!(
                text.matches("#nvvm.tcgen05_mma_collectorop<lastuse>")
                    .count(),
                4,
                "{text}"
            );
            assert!(
                text.contains(&format!("{} : i32", instruction_descriptor(256, n0))),
                "{text}"
            );
            assert!(!text.contains("llvm.inline_asm"), "{text}");
        }
    }
    /// Check TMEM results remain vector-valued through bias, FP16 conversion, and shared chunk stores.
    #[test]
    fn tensor_memory_epilogue_preserves_vector_bias_conversion_and_chunk_stores() {
        let (mut ctx, module, block, a) = fixture();
        let plan = CuteSm100EpiloguePlanAttr {
            layout: layout(128, 32, 2),
            rows: 128,
            columns: 32,
        };
        let op = CuteSm100TmemEpilogueOp::new(&mut ctx, vec![a[0], a[1], a[6], a[8]], plan);
        op.get_operation().insert_at_back(block, &ctx);
        let text = render(&mut ctx, &module, block);
        assert_eq!(
            text.matches("\"cute_nvgpu.arch.copy.SM100.tmem_load\"")
                .count(),
            1,
            "{text}"
        );
        assert_eq!(text.matches("\"vector.broadcast\"").count(), 1, "{text}");
        assert_eq!(text.matches("\"arith.addf\"").count(), 1, "{text}");
        assert_eq!(text.matches("\"arith.truncf\"").count(), 1, "{text}");
        assert_eq!(text.matches("\"llvm.store\"").count(), 4, "{text}");
        assert_eq!(text.matches("alignment = 2 : i64").count(), 4, "{text}");
        assert_eq!(text.matches("\"vector.shuffle\"").count(), 4, "{text}");
        assert!(!text.contains("\"vector.extract\""), "{text}");
        assert!(text.contains("vector<32xf32>"), "{text}");
        assert!(text.contains("vector<32xf16>"), "{text}");
        assert!(text.contains("vector<8xf16>"), "{text}");
        assert!(text.contains("vector<32xi32>"), "{text}");
        assert!(!text.contains("\"nvvm.tcgen05.ld\""), "{text}");
    }
    /// Enumerate every chunk to check element coverage, swizzle mapping, and relative store alignment.
    #[test]
    fn epilogue_vector_chunks_preserve_every_swizzled_element_and_relative_alignment() {
        let ctx = Context::new();
        let plan = CuteSm100EpiloguePlanAttr {
            layout: layout(128, 32, 2),
            rows: 128,
            columns: 32,
        };
        plan.verify(&ctx).unwrap();
        let coordinate = |row, col| IntTuple::Tuple(vec![IntTuple::Leaf(row), IntTuple::Leaf(col)]);
        let mut covered = std::collections::BTreeSet::new();
        for row in 0..i64::from(plan.rows) {
            for col in (0..i64::from(plan.columns)).step_by(8) {
                let base = plan.layout.0.checked_call(&coordinate(row, col)).unwrap();
                // A 16-byte-aligned destination retains that alignment for
                // every chunk. This proves relative alignment only: the raw
                // pointer API still permits merely FP16-aligned destinations.
                assert_eq!((base * 2) % 16, 0, "row {row}, column {col}");
                for lane in 0..8 {
                    let physical = plan
                        .layout
                        .0
                        .checked_call(&coordinate(row, col + lane))
                        .unwrap();
                    assert_eq!(physical, base + lane, "row {row}, column {}", col + lane);
                    assert!(covered.insert(physical), "duplicate offset {physical}");
                }
            }
        }
        assert!(
            covered
                .into_iter()
                .eq(0..i64::from(plan.rows * plan.columns))
        );
    }

    /// Check all lanes fence TMEM reads before a single elected arrival releases the accumulator.
    #[test]
    fn accumulator_release_fences_every_lane_before_electing_one_arrival() {
        let (mut ctx, module, block, a) = fixture();
        let plan = CuteSm100PipelinePlanAttr {
            stages: 1,
            transaction_bytes: 0,
            consumer_arrivals: 8,
            cta_group: 2,
            leader_rank: 0,
            multicast_mask: 3,
        };
        let wait = CuteSm100AccumulatorWaitOp::new(&mut ctx, vec![a[5], a[7]], plan);
        wait.get_operation().insert_at_back(block, &ctx);
        let release = CuteSm100AccumulatorReleaseOp::new(&mut ctx, vec![a[5]], plan);
        release.get_operation().insert_at_back(block, &ctx);
        let text = render(&mut ctx, &module, block);
        assert!(text.contains("\"scf.while\""), "{text}");
        assert!(text.contains("#nvvm.mbar_scope<cluster>"), "{text}");
        let fence = text.find("#nvvm.tcgen05_fence<before>").unwrap();
        let elected = text.find("\"nvvm.elect.sync\"").unwrap();
        let conditional = text.find("\"scf.if\"").unwrap();
        let arrive = text.find("\"nvvm.mbarrier.arrive\"").unwrap();
        assert!(
            fence < elected && elected < conditional && conditional < arrive,
            "{text}"
        );
    }
}
