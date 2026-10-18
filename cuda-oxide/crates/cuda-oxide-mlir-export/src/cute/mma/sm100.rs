/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paired SM100 GEMM MMA with explicit A-collector lifetime sequencing.

use crate::MlirValueUse;
use crate::cute::emit::{Emit, llvm_ptr_type, property, segments};
use dialect_cute::sm100_ops::*;

/// Emit each K16 MMA step with layout-derived descriptors and explicit A-collector reuse.
pub(in crate::cute) fn emit_mma(
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

/// Encode FP16 MMA shape fields for the verified M and N extents.
pub(in crate::cute) fn instruction_descriptor(m: u32, n: u32) -> u32 {
    (1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)
}
