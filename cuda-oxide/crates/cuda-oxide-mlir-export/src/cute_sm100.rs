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

use dialect_cute::{
    attributes::{CuteComposedLayoutAttr, CuteTmaStorePipelineAttr},
    sm100_ops::*,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};
use pliron_mlir_export::{
    DropAttribute, MlirAttribute, MlirBlock, MlirFloatType, MlirLocation, MlirOperation,
    MlirRegion, MlirResult, MlirType, MlirValueUse, OperationInput, OperationTranslation,
    TranslationError, TranslationRegistry, TranslationSession,
};

pub(crate) fn register_cute_sm100_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<CuteSm100TmemPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100MmaPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100EpiloguePlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100ClusterCopyPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteSm100PipelinePlanAttr>(DropAttribute)?;
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

#[derive(Clone, Copy)]
enum Kind {
    Alloc,
    Dealloc,
    Mma,
    Epilogue,
    TmaLoad,
    PipelineInit,
    PipelineAcquire,
    PipelineExpect,
    PipelineWait,
    PipelineRelease,
    AccInit,
    AccAcquire,
    AccCommit,
    AccWait,
    AccRelease,
    TmaStore,
    StoreCommit,
    StoreAcquire,
    StoreTail,
}
enum Plan {
    Tmem(CuteSm100TmemPlanAttr),
    Mma(CuteSm100MmaPlanAttr),
    Epilogue(CuteSm100EpiloguePlanAttr),
    Copy(CuteSm100ClusterCopyPlanAttr),
    Pipeline(CuteSm100PipelinePlanAttr),
    Store(CuteTmaStorePipelineAttr),
}
struct Recipe {
    kind: Kind,
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

fn emit_mma(
    e: &mut Emit<'_, '_>,
    a: &[MlirValueUse],
    p: &CuteSm100MmaPlanAttr,
) -> Result<(), String> {
    let ad = e.descriptor(a[1].clone(), &p.a_layout, p.m / 2)?;
    let b0d = e.descriptor(a[2].clone(), &p.b0_layout, p.n0 / 2)?;
    let b1d = e.descriptor(a[3].clone(), &p.b1_layout, p.n1 / 2)?;
    let n0 = e.constant(32, p.n0 as i128)?;
    let second = e.binary("arith.addi", a[0].clone(), n0)?;
    let d0 = e.cast("llvm.inttoptr", a[0].clone(), llvm_ptr_type(6)?)?;
    let d1 = e.cast("llvm.inttoptr", second, llvm_ptr_type(6)?)?;
    let id0 = e.constant(32, instruction_descriptor(p.m, p.n0) as i128)?;
    let id1 = e.constant(32, instruction_descriptor(p.m, p.n1) as i128)?;
    let yes = e.constant(1, 1)?;
    // Each pair owns the A collector's complete lifetime. The descriptor
    // advance is K16 FP16 elements expressed in 16-byte descriptor units.
    for kstep in 0..p.k / 16 {
        let offset = e.constant(64, (kstep * 2) as i128)?;
        let av = e.binary("arith.addi", ad.clone(), offset.clone())?;
        let bv0 = e.binary("arith.addi", b0d.clone(), offset.clone())?;
        let bv1 = e.binary("arith.addi", b1d.clone(), offset)?;
        let accumulate = if kstep == 0 {
            a[4].clone()
        } else {
            yes.clone()
        };
        for (dest, b, desc, collector) in [
            (d0.clone(), bv0, id0.clone(), "fill"),
            (d1.clone(), bv1, id1.clone(), "lastuse"),
        ] {
            let mut op = e.operation(
                "nvvm.tcgen05.mma",
                vec![dest, av.clone(), b, desc, accumulate.clone()],
            )?;
            property(&mut op, "mmaKind", "#nvvm.tcgen05_mma_kind<f16>")?;
            property(&mut op, "ctaGroup", "#nvvm.cta_group<cta_2>")?;
            property(
                &mut op,
                "collectorOp",
                &format!("#nvvm.tcgen05_mma_collectorop<{collector}>"),
            )?;
            segments(&mut op, &[1, 1, 1, 1, 1, 0, 0]);
            e.ops.push(op);
        }
    }
    Ok(())
}

fn instruction_descriptor(m: u32, n: u32) -> u32 {
    (1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)
}

fn emit_epilogue(
    e: &mut Emit<'_, '_>,
    a: &[MlirValueUse],
    p: &CuteSm100EpiloguePlanAttr,
) -> Result<(), String> {
    let ptr = e.cast(
        "builtin.unrealized_conversion_cast",
        a[0].clone(),
        dialect("!cute.ptr<f32,tmem,align<16>>")?,
    )?;
    let (result, regs) = e.fresh(MlirType::Vector {
        shape: vec![32],
        element: Box::new(MlirType::Integer(32)),
    });
    let mut load = e.operation("cute_nvgpu.arch.copy.SM100.tmem_load", vec![ptr])?;
    load.results.push(result);
    for name in ["num_dp", "num_b", "num_rep"] {
        integer_property(&mut load, name, 32, 32);
    }
    e.ops.push(load);
    // The native CuTe TMEM load is a synchronous copy operation: CUTLASS
    // inserts its tcgen05.wait::ld before making the register vector available.
    // Keep the register tile vector-valued through the bias and conversion.
    // Scalar extraction here prevents packed FP16 conversion and shared stores.
    let floats = MlirType::Vector {
        shape: vec![32],
        element: Box::new(MlirType::Float(MlirFloatType::F32)),
    };
    let values = e.cast("arith.bitcast", regs, floats.clone())?;
    let bias = e.cast("vector.broadcast", a[3].clone(), floats)?;
    let biased = e.binary("arith.addf", values, bias)?;
    let halves = e.cast(
        "arith.truncf",
        biased,
        MlirType::Vector {
            shape: vec![32],
            element: Box::new(MlirType::Float(MlirFloatType::F16)),
        },
    )?;
    let dst = e.llvm_ptr(a[1].clone(), 3)?;
    let stride = e.constant(32, p.columns as i128)?;
    let row = e.binary("arith.muli", a[2].clone(), stride)?;
    let swizzle = p.layout.0.outer();
    let mask = e.constant(32, (((1u32 << swizzle.bits) - 1) << swizzle.base) as i128)?;
    let shift = e.constant(32, swizzle.shift as i128)?;
    // The verified B64 layout has element swizzle S<2,3,3>: it only
    // changes bits 3 and 4 using bits 6 and 7. Within each eight-element
    // chunk the low three bits therefore remain ordered and contiguous.
    // Swizzle each chunk's base once, then store its eight FP16 values.
    for col in (0..p.columns).step_by(8) {
        let (result, chunk) = e.fresh(MlirType::Vector {
            shape: vec![8],
            element: Box::new(MlirType::Float(MlirFloatType::F16)),
        });
        let mut extract = e.operation("vector.shuffle", vec![halves.clone(), halves.clone()])?;
        extract.results.push(result);
        extract.properties.insert(
            "mask".into(),
            MlirAttribute::DenseI64Array((col..col + 8).map(i64::from).collect()),
        );
        e.ops.push(extract);
        let c = e.constant(32, col as i128)?;
        let plain = e.binary("arith.addi", row.clone(), c)?;
        let shifted = e.binary("arith.shrui", plain.clone(), shift.clone())?;
        let swizzled = e.binary("arith.andi", shifted, mask.clone())?;
        let offset = e.binary("arith.xori", plain, swizzled)?;
        let addr = e.gep(dst.clone(), offset, MlirType::Float(MlirFloatType::F16))?;
        let mut store = e.operation("llvm.store", vec![chunk, addr])?;
        // The raw public pointer only promises FP16 alignment. Preserve that
        // contract: stronger alignment can be inferred from an aligned shared
        // allocation and the chunk's multiple-of-eight element offset.
        integer_property(&mut store, "alignment", 64, 2);
        e.ops.push(store);
    }
    Ok(())
}

fn emit_pipeline(
    e: &mut Emit<'_, '_>,
    kind: Kind,
    a: &[MlirValueUse],
    p: &CuteSm100PipelinePlanAttr,
) -> Result<(), String> {
    use Kind::*;
    match kind {
        PipelineInit | AccInit => {
            for stage in 0..p.stages {
                let slot = e.constant(32, stage as i128)?;
                for (index, count) in if matches!(kind, PipelineInit) {
                    [(0, 1), (1, p.consumer_arrivals)]
                } else {
                    [(0, p.consumer_arrivals), (1, 1)]
                } {
                    let bar = e.barrier(a[index].clone(), Some(slot.clone()))?;
                    let count = e.constant(32, count as i128)?;
                    e.emit("nvvm.mbarrier.init", vec![bar, count])?;
                }
            }
        }
        PipelineAcquire | PipelineWait | AccAcquire | AccWait => {
            let staged = matches!(kind, PipelineAcquire | PipelineWait);
            let bar = e.barrier(a[0].clone(), staged.then(|| a[1].clone()))?;
            let phase = a[if staged { 2 } else { 1 }].clone();
            e.wait_cluster(bar, phase)?;
            if !matches!(kind, PipelineAcquire) {
                e.fence("after")?;
            }
        }
        PipelineExpect => {
            let bar = e.barrier(a[0].clone(), Some(a[1].clone()))?;
            let bytes = e.constant(32, p.transaction_bytes as i128)?;
            let mut op = e.operation("nvvm.mbarrier.arrive.expect_tx", vec![bar, bytes])?;
            property(&mut op, "scope", "#nvvm.mem_scope<cluster>")?;
            op.properties
                .insert("relaxed".into(), MlirAttribute::Bool(true));
            e.ops.push(op);
        }
        PipelineRelease | AccCommit => {
            let bar = e.barrier(
                a[0].clone(),
                matches!(kind, PipelineRelease).then(|| a[1].clone()),
            )?;
            let mask = e.constant(16, p.multicast_mask as i128)?;
            let mut op = e.operation("nvvm.tcgen05.commit", vec![bar, mask])?;
            property(&mut op, "group", "#nvvm.cta_group<cta_2>")?;
            e.ops.push(op);
        }
        AccRelease => {
            // Every lane orders its preceding TMEM reads before one lane of
            // the warp releases this accumulator to the MMA producer.
            e.fence("before")?;
            let elected = e.value("nvvm.elect.sync", vec![], MlirType::Integer(1))?;
            let start = e.ops.len();
            let bar = e.llvm_ptr(a[0].clone(), 3)?;
            let leader = e.constant(32, p.leader_rank as i128)?;
            let remote = e.value("nvvm.mapa", vec![bar, leader], llvm_ptr_type(7)?)?;
            let mut arrive = e.operation("nvvm.mbarrier.arrive", vec![remote])?;
            property(&mut arrive, "scope", "#nvvm.mem_scope<cluster>")?;
            e.ops.push(arrive);
            e.emit("scf.yield", vec![])?;
            let body = e.ops.split_off(start);
            let mut conditional = e.operation("scf.if", vec![elected])?;
            conditional.regions = vec![e.region(body), MlirRegion { blocks: vec![] }];
            e.ops.push(conditional);
        }
        _ => return Err("invalid SM100 pipeline operation".into()),
    }
    Ok(())
}

struct Emit<'a, 'b> {
    session: &'a mut TranslationSession<'b>,
    location: MlirLocation,
    ops: Vec<MlirOperation>,
}
impl Emit<'_, '_> {
    fn operation(&self, name: &str, operands: Vec<MlirValueUse>) -> Result<MlirOperation, String> {
        let mut op = MlirOperation::new(name)?;
        op.operands = operands;
        op.location = self.location.clone();
        Ok(op)
    }
    fn fresh(&mut self, ty: MlirType) -> (MlirResult, MlirValueUse) {
        let id = self.session.fresh_value();
        (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
    }
    fn emit(&mut self, name: &str, operands: Vec<MlirValueUse>) -> Result<(), String> {
        let op = self.operation(name, operands)?;
        self.ops.push(op);
        Ok(())
    }
    fn value(
        &mut self,
        name: &str,
        operands: Vec<MlirValueUse>,
        ty: MlirType,
    ) -> Result<MlirValueUse, String> {
        let (r, v) = self.fresh(ty);
        let mut op = self.operation(name, operands)?;
        op.results.push(r);
        self.ops.push(op);
        Ok(v)
    }
    fn cast(
        &mut self,
        name: &str,
        operand: MlirValueUse,
        ty: MlirType,
    ) -> Result<MlirValueUse, String> {
        self.value(name, vec![operand], ty)
    }
    fn binary(
        &mut self,
        name: &str,
        a: MlirValueUse,
        b: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let ty = a.ty.clone();
        self.value(name, vec![a, b], ty)
    }
    fn constant(&mut self, bits: u32, value: i128) -> Result<MlirValueUse, String> {
        let ty = MlirType::Integer(bits);
        let (r, v) = self.fresh(ty.clone());
        let mut op = self.operation("arith.constant", vec![])?;
        op.results.push(r);
        op.properties
            .insert("value".into(), MlirAttribute::Integer { value, ty });
        self.ops.push(op);
        Ok(v)
    }
    fn llvm_ptr(&mut self, ptr: MlirValueUse, space: u32) -> Result<MlirValueUse, String> {
        let ty = llvm_ptr_type(space)?;
        if ptr.ty == ty {
            Ok(ptr)
        } else {
            self.cast("llvm.addrspacecast", ptr, ty)
        }
    }
    fn cute_ptr(
        &mut self,
        ptr: MlirValueUse,
        element: &str,
        space: &str,
        align: u32,
        swizzle: Option<String>,
    ) -> Result<MlirValueUse, String> {
        let ptr = self.llvm_ptr(ptr, if space == "smem" { 3 } else { 0 })?;
        let suffix = swizzle.map(|s| format!(",{s}")).unwrap_or_default();
        self.cast(
            "builtin.unrealized_conversion_cast",
            ptr,
            dialect(&format!(
                "!cute.ptr<{element},{space},align<{align}>{suffix}>"
            ))?,
        )
    }
    fn gep(
        &mut self,
        ptr: MlirValueUse,
        index: MlirValueUse,
        elem: MlirType,
    ) -> Result<MlirValueUse, String> {
        let (r, v) = self.fresh(ptr.ty.clone());
        let mut op = self.operation("llvm.getelementptr", vec![ptr, index])?;
        op.results.push(r);
        op.properties
            .insert("elem_type".into(), MlirAttribute::Type(elem));
        op.properties.insert(
            "rawConstantIndices".into(),
            MlirAttribute::DenseI32Array(vec![i32::MIN]),
        );
        self.ops.push(op);
        Ok(v)
    }
    fn barrier(
        &mut self,
        ptr: MlirValueUse,
        stage: Option<MlirValueUse>,
    ) -> Result<MlirValueUse, String> {
        let ptr = self.llvm_ptr(ptr, 3)?;
        match stage {
            Some(stage) => self.gep(ptr, stage, MlirType::Integer(64)),
            None => Ok(ptr),
        }
    }
    fn descriptor(
        &mut self,
        ptr: MlirValueUse,
        layout: &CuteComposedLayoutAttr,
        rows: u32,
    ) -> Result<MlirValueUse, String> {
        // Source composed layouts count FP16 elements. CuTe pointers carry
        // BYTE-address swizzles, so the base advances by log2(sizeof(f16)).
        let swizzle = layout.0.outer();
        let ptr = self.cute_ptr(
            ptr,
            "f16",
            "smem",
            128,
            Some(format!(
                "S<{},{},{}>",
                swizzle.bits,
                swizzle.base + 1,
                swizzle.shift
            )),
        )?;
        let (result, desc) = self.fresh(dialect("!cute_nvgpu.smem_desc")?);
        let mut op = self.operation("cute_nvgpu.make_umma_smem_desc", vec![ptr])?;
        op.results.push(result);
        property(
            &mut op,
            "layout",
            &format!("#cute.layout<\"(({rows},16),1):((64,1),0)\">"),
        )?;
        property(&mut op, "major", "#cute_nvgpu.major<k>")?;
        self.ops.push(op);
        self.cast(
            "builtin.unrealized_conversion_cast",
            desc,
            MlirType::Integer(64),
        )
    }
    fn fence(&mut self, kind: &str) -> Result<(), String> {
        let mut op = self.operation("nvvm.tcgen05.fence", vec![])?;
        property(&mut op, "kind", &format!("#nvvm.tcgen05_fence<{kind}>"))?;
        self.ops.push(op);
        Ok(())
    }
    fn region(&mut self, operations: Vec<MlirOperation>) -> MlirRegion {
        MlirRegion {
            blocks: vec![MlirBlock {
                id: self.session.fresh_block(),
                arguments: vec![],
                operations,
            }],
        }
    }
    fn wait_cluster(&mut self, barrier: MlirValueUse, phase: MlirValueUse) -> Result<(), String> {
        // Preserve cluster acquire semantics and retry on a bounded try-wait;
        // the older high-level NVVM blocking operation only has CTA scope.
        let start = self.ops.len();
        let (r, done) = self.fresh(MlirType::Integer(1));
        let mut wait = self.operation("nvvm.mbarrier.wait.parity", vec![barrier, phase])?;
        wait.results.push(r);
        property(&mut wait, "kind", "#nvvm.mbar_wait<try>")?;
        property(&mut wait, "scope", "#nvvm.mbar_scope<cluster>")?;
        property(&mut wait, "order", "#nvvm.mem_order<acquire>")?;
        self.ops.push(wait);
        let yes = self.constant(1, 1)?;
        let again = self.binary("arith.xori", done, yes)?;
        self.emit("scf.condition", vec![again])?;
        let before = self.ops.split_off(start);
        let yield_op = self.operation("scf.yield", vec![])?;
        let mut loop_op = self.operation("scf.while", vec![])?;
        loop_op.regions = vec![self.region(before), self.region(vec![yield_op])];
        self.ops.push(loop_op);
        Ok(())
    }
}
fn dialect(s: &str) -> Result<MlirType, String> {
    MlirType::dialect(s)
}
fn llvm_ptr_type(space: u32) -> Result<MlirType, String> {
    dialect(&if space == 0 {
        "!llvm.ptr".into()
    } else {
        format!("!llvm.ptr<{space}>")
    })
}
fn property(op: &mut MlirOperation, key: &str, value: &str) -> Result<(), String> {
    op.properties
        .insert(key.into(), MlirAttribute::dialect(value)?);
    Ok(())
}
fn integer_property(op: &mut MlirOperation, key: &str, bits: u32, value: i128) {
    op.properties.insert(
        key.into(),
        MlirAttribute::Integer {
            value,
            ty: MlirType::Integer(bits),
        },
    );
}
fn segments(op: &mut MlirOperation, values: &[i32]) {
    op.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(values.to_vec()),
    );
}

#[cfg(test)]
mod tests {
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
        let target = pliron_mlir_export::translate_module(
            ctx,
            module,
            &registry,
            &profile.translation_config(),
        )
        .unwrap();
        pliron_mlir_export::render_module(&target)
    }
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
