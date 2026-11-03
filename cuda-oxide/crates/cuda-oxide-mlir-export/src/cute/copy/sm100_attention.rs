/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Independent SM100 attention TMA operations and their descriptor contracts.

use crate::cute::{
    common_sm100::contract,
    emit::{Emit, integer_property, llvm_ptr_type, property, segments},
};
use crate::{MlirOperation, MlirType, OperationInput, OperationTranslation, TranslationSession};
use dialect_cute::sm100_attention_ops::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};

/// Emit independent SM100 attention loads or stores using the verified descriptor plan.
pub(in crate::cute) struct Tma(
    #[doc = "True selects shared-to-global stores; false selects global-to-shared loads."]
    pub(in crate::cute) bool,
);
impl OperationTranslation for Tma {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let p = if self.0 {
            let o = CuteSm100TmaStoreOp::wrap(source);
            o.verify(ctx).map_err(|e| e.to_string())?;
            o.plan(ctx)
        } else {
            let o = CuteSm100TmaLoadOp::wrap(source);
            o.verify(ctx).map_err(|e| e.to_string())?;
            o.plan(ctx)
        }
        .ok_or("missing SM100 TMA plan")?;
        contract(&input)?;
        let mut e = Emit {
            session,
            location: input.location,
            ops: vec![],
        };
        let a = input.operands;
        let desc = e.cute_ptr(
            a[0].clone(),
            "!cute_nvgpu.tma_descriptor_tiled",
            "generic",
            64,
            None,
        )?;
        let data = e.cute_ptr(a[1].clone(), "f16", "smem", 128, None)?;
        if self.0 {
            let mut op = e.operation(
                "cute_nvgpu.arch.copy.SM100.tma_store",
                vec![desc, data, a[2].clone(), a[3].clone(), a[4].clone()],
            )?;
            property(&mut op, "mode", "#cute_nvgpu.tma_store_mode<tiled>")?;
            segments(&mut op, &[1, 1, 3, 0]);
            e.ops.push(op);
        } else {
            let local = e.llvm_ptr(a[5].clone(), 3)?;
            let bar = if p.cta_group == 2 {
                let zero = e.constant(32, 0)?;
                let remote = e.value("nvvm.mapa", vec![local, zero], llvm_ptr_type(7)?)?;
                let bits = e.cast("llvm.ptrtoint", remote, MlirType::Integer(32))?;
                e.cast("llvm.inttoptr", bits, llvm_ptr_type(3)?)?
            } else {
                local
            };
            let bar = e.cute_ptr(bar, "i64", "smem", 8, None)?;
            // Each issuing CTA copies to its own shared tile. Group2 changes
            // completion routing, not the number of destination CTAs; the
            // ordinary two-CTA TMA operation therefore has no multicast mask.
            let args = vec![desc, data, bar, a[2].clone(), a[3].clone(), a[4].clone()];
            let mut op = e.operation("cute_nvgpu.arch.copy.SM100.tma_load", args)?;
            property(&mut op, "mode", "#cute_nvgpu.tma_load_mode<tiled>")?;
            integer_property(&mut op, "num_cta", 32, p.cta_group as i128);
            segments(&mut op, &[1, 1, 1, 3, 0, 0, 0]);
            e.ops.push(op);
        }
        Ok(e.ops)
    }
}
