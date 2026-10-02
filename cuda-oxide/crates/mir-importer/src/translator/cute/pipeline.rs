/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural Rust types for the TMA load-pipeline boundary.
//!
//! The pipeline handle is compiler-only. Its cursor remains an ordinary Rust
//! pair, so loop state still travels as two `u32` values:
//!
//! ```text
//! TmaLoadPipeline<S, W, B>  -> !cute.tma_load_pipeline<S,W,B>
//! PipelineState<Role, S>    -> { slot: u32, phase: u32 }
//! ```

use crate::error::{TranslationErr, TranslationResult};
use dialect_cute::attributes::CutePipelineStateAttr;
use dialect_cute::types::CuteTmaLoadPipelineType;
use pliron::context::Context;
use pliron::input_error_noloc;
use pliron::r#type::TypeHandle;
use rustc_public::ty::{GenericArgs, RigidTy, Ty, TyKind};

use super::tensor::{canonical_adt_path, const_arg, exact_schema, type_arg, unsigned_const};

const TMA_LOAD_PIPELINE: &str = "cute_rs::pipeline::TmaLoadPipeline";
const PIPELINE_STATE: &str = "cute_rs::pipeline::PipelineState";
const PRODUCER: &str = "cute_rs::pipeline::Producer";
const CONSUMER: &str = "cute_rs::pipeline::Consumer";

/// Static facts carried by `TmaLoadPipeline<S, W, B>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TmaLoadPipelineRust {
    pub stages: u64,
    pub consumer_warps: u32,
    pub transaction_bytes: u32,
}

fn exact_adt(ty: &Ty, expected: &str) -> Result<Option<GenericArgs>, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Ok(None);
    };
    if canonical_adt_path(&def) != expected {
        return Ok(None);
    }
    Ok(Some(args))
}

/// Decode the compiler-only load-pipeline handle.
pub(crate) fn decode_tma_load_pipeline(ty: &Ty) -> Result<Option<TmaLoadPipelineRust>, String> {
    let Some(args) = exact_adt(ty, TMA_LOAD_PIPELINE)? else {
        return Ok(None);
    };
    exact_schema(&args, &["Const", "Const", "Const"], TMA_LOAD_PIPELINE)?;
    let stages = unsigned_const(
        &const_arg(&args, 0, TMA_LOAD_PIPELINE)?,
        "pipeline stage count",
    )?;
    let consumer_warps = unsigned_const(
        &const_arg(&args, 1, TMA_LOAD_PIPELINE)?,
        "pipeline consumer-warp count",
    )?;
    let transaction_bytes = unsigned_const(
        &const_arg(&args, 2, TMA_LOAD_PIPELINE)?,
        "pipeline transaction byte count",
    )?;
    let consumer_warps = u32::try_from(consumer_warps)
        .map_err(|_| "pipeline consumer-warp count does not fit in u32".to_owned())?;
    let transaction_bytes = u32::try_from(transaction_bytes)
        .map_err(|_| "pipeline transaction byte count does not fit in u32".to_owned())?;
    validate_pipeline_facts(stages, consumer_warps, transaction_bytes)?;
    Ok(Some(TmaLoadPipelineRust {
        stages,
        consumer_warps,
        transaction_bytes,
    }))
}

fn validate_pipeline_facts(
    stages: u64,
    consumer_warps: u32,
    transaction_bytes: u32,
) -> Result<(), String> {
    if stages == 0 || stages > u64::from(u32::MAX) {
        return Err(format!(
            "pipeline stage count must be between 1 and {}, got {stages}",
            u32::MAX
        ));
    }
    if !(1..=32).contains(&consumer_warps) {
        return Err(format!(
            "pipeline consumer-warp count must be between 1 and 32, got {consumer_warps}"
        ));
    }
    if transaction_bytes == 0 {
        return Err("pipeline transaction byte count must be greater than zero".to_owned());
    }
    stages
        .checked_mul(16)
        .ok_or_else(|| "pipeline barrier storage size overflowed u64".to_owned())?;
    Ok(())
}

fn decode_role(ty: &Ty) -> Result<Option<bool>, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Ok(None);
    };
    let path = canonical_adt_path(&def);
    if !matches!(path.as_str(), PRODUCER | CONSUMER) {
        return Ok(None);
    }
    exact_schema(&args, &[], &path)?;
    Ok(Some(path == PRODUCER))
}

/// Decode the ordinary two-scalar pipeline cursor.
pub(crate) fn decode_pipeline_state(ty: &Ty) -> Result<Option<CutePipelineStateAttr>, String> {
    let Some(args) = exact_adt(ty, PIPELINE_STATE)? else {
        return Ok(None);
    };
    exact_schema(&args, &["Type", "Const"], PIPELINE_STATE)?;
    let role = type_arg(&args, 0, PIPELINE_STATE)?;
    let producer = decode_role(&role)?
        .ok_or_else(|| "PipelineState role must be cute-rs Producer or Consumer".to_owned())?;
    let stages = unsigned_const(
        &const_arg(&args, 1, PIPELINE_STATE)?,
        "pipeline-state stage count",
    )?;
    if stages == 0 || stages > u64::from(u32::MAX) {
        return Err(format!(
            "pipeline-state stage count must be between 1 and {}, got {stages}",
            u32::MAX
        ));
    }
    Ok(Some(if producer {
        CutePipelineStateAttr::producer(stages)
    } else {
        CutePipelineStateAttr::consumer(stages)
    }))
}

/// Decode state behind exactly one reference or raw pointer.
pub(crate) fn decode_pipeline_state_receiver(
    ty: &Ty,
) -> Result<Option<CutePipelineStateAttr>, String> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _)) => {
            decode_pipeline_state(&pointee)
        }
        _ => decode_pipeline_state(ty),
    }
}

fn pipeline_type(ctx: &mut Context, facts: TmaLoadPipelineRust) -> TypeHandle {
    CuteTmaLoadPipelineType::get(
        ctx,
        facts.stages,
        facts.consumer_warps,
        facts.transaction_bytes,
    )
    .into()
}

/// Type-translation hook called before generic ADT lowering.
pub(crate) fn try_translate_tma_load_pipeline_type(
    ctx: &mut Context,
    rust_ty: &Ty,
) -> Option<TranslationResult<TypeHandle>> {
    match decode_tma_load_pipeline(rust_ty) {
        Ok(Some(facts)) => Some(Ok(pipeline_type(ctx, facts))),
        Ok(None) => None,
        Err(error) => Some(Err(input_error_noloc!(TranslationErr::unsupported(
            format!("invalid cute-rs TMA load-pipeline type: {error}")
        )))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_pipeline_type_keeps_all_static_facts() {
        let mut ctx = Context::new();
        dialect_cute::register(&mut ctx);
        let facts = TmaLoadPipelineRust {
            stages: 3,
            consumer_warps: 8,
            transaction_bytes: 17_408,
        };
        let ty = pipeline_type(&mut ctx, facts);
        let ty = ty.deref(&ctx);
        let pipeline = ty.downcast_ref::<CuteTmaLoadPipelineType>().unwrap();
        assert_eq!(pipeline.stages, 3);
        assert_eq!(pipeline.consumer_warps, 8);
        assert_eq!(pipeline.transaction_bytes, 17_408);
        assert_eq!(pipeline.storage_bytes(), Some(48));
    }

    #[test]
    fn load_pipeline_facts_fail_closed() {
        assert!(validate_pipeline_facts(0, 8, 1).is_err());
        assert!(validate_pipeline_facts(3, 33, 1).is_err());
        assert!(validate_pipeline_facts(3, 8, 0).is_err());
    }
}
