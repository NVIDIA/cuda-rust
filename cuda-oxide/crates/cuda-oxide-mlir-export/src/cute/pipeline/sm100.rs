/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 cluster load and accumulator pipeline synchronization.

use crate::cute::{
    emit::{Emit, llvm_ptr_type, property},
    sm100::Kind,
};
use crate::{MlirAttribute, MlirRegion, MlirType, MlirValueUse};
use dialect_cute::sm100_ops::*;

/// Emit cluster load/accumulator barrier actions and their required TCGen05 fences.
pub(in crate::cute) fn emit_pipeline(
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
