/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct TCGen05 primitive mappings for the pinned CUTLASS MLIR consumer.
//!
//! CUDA Oxide represents a TMEM address as a packed `u32` token and a load as
//! scalar register results. MLIR uses an address-space-6 pointer and a vector.
//! Those carrier conversions must not rescale the packed TMEM row/column bits.
//! This pack does not lower through PTX or expand high-level CuTe operations.

use dialect_nvvm::ops::{
    CvtaGenericToSharedOffsetOp, Tcgen05AllocCg2Op, Tcgen05CommitMulticastCg2Op,
    Tcgen05DeallocCg2Op, Tcgen05FenceAfterThreadSyncOp, Tcgen05FenceBeforeThreadSyncOp,
    Tcgen05Ld32x32bX32RawOp, Tcgen05LoadWaitOp, Tcgen05MmaBBufferAttr, Tcgen05MmaBUsageAttr,
    Tcgen05MmaCollectorAAttr, Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr, Tcgen05MmaKindAttr,
    Tcgen05MmaOp, Tcgen05RelinquishAllocPermitCg2Op,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};
use pliron_mlir_export::{
    DropAttribute, MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};

/// Register the two-CTA FP16/TMEM primitives used by the split-N GEMM.
///
/// Other MMA forms deliberately remain unsupported until their operand and
/// selector contracts have a reviewed mapping. Target capability validation
/// remains the caller's responsibility, as for the other intrinsic packs.
pub fn register_nvvm_tcgen05_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    // Read the verified source selectors directly rather than converting
    // source enum discriminants into unrelated target enum discriminants.
    registry.register_attribute::<Tcgen05MmaFormAttr>(DropAttribute)?;
    registry.register_attribute::<Tcgen05MmaKindAttr>(DropAttribute)?;
    registry.register_attribute::<Tcgen05MmaCtaGroupAttr>(DropAttribute)?;
    registry.register_attribute::<Tcgen05MmaCollectorAAttr>(DropAttribute)?;
    registry.register_attribute::<Tcgen05MmaBBufferAttr>(DropAttribute)?;
    registry.register_attribute::<Tcgen05MmaBUsageAttr>(DropAttribute)?;
    registry.register_operation::<Tcgen05MmaOp>(MmaTranslation)?;
    registry.register_operation::<CvtaGenericToSharedOffsetOp>(SharedOffsetTranslation)?;
    registry.register_operation::<Tcgen05Ld32x32bX32RawOp>(LoadTranslation)?;
    registry.register_operation::<Tcgen05AllocCg2Op>(PrimitiveTranslation(Primitive::Alloc))?;
    registry.register_operation::<Tcgen05DeallocCg2Op>(PrimitiveTranslation(Primitive::Dealloc))?;
    registry.register_operation::<Tcgen05CommitMulticastCg2Op>(PrimitiveTranslation(
        Primitive::Commit,
    ))?;
    registry.register_operation::<Tcgen05RelinquishAllocPermitCg2Op>(PrimitiveTranslation(
        Primitive::Relinquish,
    ))?;
    registry.register_operation::<Tcgen05LoadWaitOp>(PrimitiveTranslation(Primitive::LoadWait))?;
    registry.register_operation::<Tcgen05FenceBeforeThreadSyncOp>(PrimitiveTranslation(
        Primitive::FenceBefore,
    ))?;
    registry.register_operation::<Tcgen05FenceAfterThreadSyncOp>(PrimitiveTranslation(
        Primitive::FenceAfter,
    ))?;
    Ok(())
}

struct MmaTranslation;

struct SharedOffsetTranslation;

impl OperationTranslation for SharedOffsetTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CvtaGenericToSharedOffsetOp::new(source)
            .verify(ctx)
            .map_err(|error| error.to_string())?;
        if input.operands.len() != 1
            || input.results.len() != 1
            || !input.attributes.is_empty()
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err("shared offset conversion expects one pointer, one result, and no attributes or control flow".into());
        }
        let mut operations = vec![];
        let pointer = shared_pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            &input.location,
        )?;
        operations.push(operation(
            "llvm.ptrtoint",
            vec![pointer],
            input.results,
            &input.location,
        )?);
        Ok(operations)
    }
}

impl OperationTranslation for MmaTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let mma = Tcgen05MmaOp::new(source);
        mma.verify(ctx).map_err(|error| error.to_string())?;
        if mma.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() != Some(&Tcgen05MmaFormAttr::Shared)
            || mma.get_attr_nvvm_tcgen05_mma_kind(ctx).as_deref() != Some(&Tcgen05MmaKindAttr::F16)
            || mma.get_attr_nvvm_tcgen05_mma_cta_group(ctx).as_deref()
                != Some(&Tcgen05MmaCtaGroupAttr::Cg2)
        {
            return Err(
                "CUTLASS TCGen05 mapping supports shared-A FP16 CTA-group-2 MMA only".into(),
            );
        }
        let collector = match mma.get_attr_nvvm_tcgen05_mma_collector_a(ctx).as_deref() {
            Some(Tcgen05MmaCollectorAAttr::Discard) => "discard",
            Some(Tcgen05MmaCollectorAAttr::LastUse) => "lastuse",
            Some(Tcgen05MmaCollectorAAttr::Fill) => "fill",
            Some(Tcgen05MmaCollectorAAttr::Use) => "use",
            None => return Err("TCGen05 MMA requires a collector-A selector".into()),
        };
        require_input(&mut input, "v1:i0763", 5, 0)?;
        let (cast, tmem) = tmem_pointer(session, input.operands[0].clone(), &input.location)?;
        input.operands[0] = tmem;
        let mut target = operation("nvvm.tcgen05.mma", input.operands, vec![], &input.location)?;
        target.properties.insert(
            "mmaKind".into(),
            MlirAttribute::dialect("#nvvm.tcgen05_mma_kind<f16>")?,
        );
        target.properties.insert("ctaGroup".into(), cta_group()?);
        target.properties.insert(
            "collectorOp".into(),
            MlirAttribute::dialect(format!("#nvvm.tcgen05_mma_collectorop<{collector}>"))?,
        );
        target.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 1, 1, 1, 0, 0]),
        );
        Ok(vec![cast, target])
    }
}

#[derive(Clone, Copy)]
enum Primitive {
    Alloc,
    Dealloc,
    Commit,
    Relinquish,
    LoadWait,
    FenceBefore,
    FenceAfter,
}

struct PrimitiveTranslation(Primitive);

impl OperationTranslation for PrimitiveTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let (name, marker, operands) = match self.0 {
            Primitive::Alloc => {
                Tcgen05AllocCg2Op::new(source)
                    .verify(ctx)
                    .map_err(|error| error.to_string())?;
                ("nvvm.tcgen05.alloc", "v1:i0359", 2)
            }
            Primitive::Dealloc => {
                Tcgen05DeallocCg2Op::new(source)
                    .verify(ctx)
                    .map_err(|error| error.to_string())?;
                ("nvvm.tcgen05.dealloc", "v1:i0360", 2)
            }
            Primitive::Commit => {
                Tcgen05CommitMulticastCg2Op::new(source)
                    .verify(ctx)
                    .map_err(|error| error.to_string())?;
                ("nvvm.tcgen05.commit", "v1:i0365", 2)
            }
            Primitive::Relinquish => {
                Tcgen05RelinquishAllocPermitCg2Op::new(source)
                    .verify(ctx)
                    .map_err(|error| error.to_string())?;
                ("nvvm.tcgen05.relinquish_alloc_permit", "v1:i0361", 0)
            }
            Primitive::LoadWait => {
                Tcgen05LoadWaitOp::new(source)
                    .verify(ctx)
                    .map_err(|error| error.to_string())?;
                ("nvvm.tcgen05.wait", "v1:i0357", 0)
            }
            Primitive::FenceBefore => {
                Tcgen05FenceBeforeThreadSyncOp::new(source)
                    .verify(ctx)
                    .map_err(|error| error.to_string())?;
                ("nvvm.tcgen05.fence", "v1:i0346", 0)
            }
            Primitive::FenceAfter => {
                Tcgen05FenceAfterThreadSyncOp::new(source)
                    .verify(ctx)
                    .map_err(|error| error.to_string())?;
                ("nvvm.tcgen05.fence", "v1:i0347", 0)
            }
        };
        require_input(&mut input, marker, operands, 0)?;
        let mut operations = vec![];
        if matches!(self.0, Primitive::Dealloc) {
            let (cast, pointer) =
                tmem_pointer(session, input.operands[0].clone(), &input.location)?;
            operations.push(cast);
            input.operands[0] = pointer;
        }
        // Native SMEM pointer arguments can be generic after MIR casts.
        // Preserve the address while making the required shared space explicit.
        if matches!(self.0, Primitive::Alloc | Primitive::Commit) {
            input.operands[0] = shared_pointer(
                session,
                &mut operations,
                input.operands[0].clone(),
                &input.location,
            )?;
        }
        let mut target = operation(name, input.operands, vec![], &input.location)?;
        let kind = match self.0 {
            Primitive::LoadWait => Some("#nvvm.tcgen05_wait<load>"),
            Primitive::FenceBefore => Some("#nvvm.tcgen05_fence<before>"),
            Primitive::FenceAfter => Some("#nvvm.tcgen05_fence<after>"),
            _ => None,
        };
        if let Some(kind) = kind {
            target
                .properties
                .insert("kind".into(), MlirAttribute::dialect(kind)?);
        } else {
            target.properties.insert("group".into(), cta_group()?);
        }
        operations.push(target);
        Ok(operations)
    }
}

struct LoadTranslation;

impl OperationTranslation for LoadTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        Tcgen05Ld32x32bX32RawOp::new(source)
            .verify(ctx)
            .map_err(|error| error.to_string())?;
        require_input(&mut input, "v1:i0664", 1, 32)?;
        let (cast, tmem) = tmem_pointer(session, input.operands[0].clone(), &input.location)?;
        let mut operations = vec![cast];
        let (vector_result, vector) = fresh(
            session,
            MlirType::Vector {
                shape: vec![32],
                element: Box::new(MlirType::Integer(32)),
            },
        );
        let mut load = operation(
            "nvvm.tcgen05.ld",
            vec![tmem],
            vec![vector_result],
            &input.location,
        )?;
        load.properties.insert(
            "shape".into(),
            MlirAttribute::dialect("#nvvm.tcgen05_ldst_shape<shape_32x32b>")?,
        );
        operations.push(load);
        // `raw` is asynchronous: retain the caller's explicit tcgen05.wait.
        // Do not insert a wait here and change the source scheduling policy.
        for (index, result) in input.results.into_iter().enumerate() {
            let mut extract = operation(
                "vector.extract",
                vec![vector.clone()],
                vec![result],
                &input.location,
            )?;
            extract.properties.insert(
                "static_position".into(),
                MlirAttribute::DenseI64Array(vec![index as i64]),
            );
            operations.push(extract);
        }
        Ok(operations)
    }
}

fn require_input(
    input: &mut OperationInput,
    marker: &str,
    operands: usize,
    results: usize,
) -> Result<(), String> {
    if input.attributes.remove("cuda_oxide_intrinsic_marker")
        != Some(MlirAttribute::String(marker.into()))
    {
        return Err(format!(
            "TCGen05 operation expected intrinsic marker {marker}"
        ));
    }
    if input.operands.len() != operands
        || input.results.len() != results
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err(format!(
            "TCGen05 operation expected {operands} operands, {results} results, and no residual attributes or control flow"
        ));
    }
    Ok(())
}

fn cta_group() -> Result<MlirAttribute, String> {
    MlirAttribute::dialect("#nvvm.cta_group<cta_2>")
}

fn tmem_pointer(
    session: &mut TranslationSession<'_>,
    token: MlirValueUse,
    location: &MlirLocation,
) -> Result<(MlirOperation, MlirValueUse), String> {
    if token.ty != MlirType::Integer(32) {
        return Err("TMEM address must be a packed 32-bit token".into());
    }
    let (result, pointer) = fresh(session, MlirType::dialect("!llvm.ptr<6>")?);
    Ok((
        operation("llvm.inttoptr", vec![token], vec![result], location)?,
        pointer,
    ))
}

fn shared_pointer(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    pointer: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let shared = MlirType::dialect("!llvm.ptr<3>")?;
    if pointer.ty == shared {
        return Ok(pointer);
    }
    if pointer.ty != MlirType::dialect("!llvm.ptr")? {
        return Err("TCGen05 allocation/commit needs a shared or generic pointer".into());
    }
    let (result, converted) = fresh(session, shared);
    operations.push(operation(
        "llvm.addrspacecast",
        vec![pointer],
        vec![result],
        location,
    )?);
    Ok(converted)
}

fn fresh(session: &mut TranslationSession<'_>, ty: MlirType) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}

fn operation(
    name: &str,
    operands: Vec<MlirValueUse>,
    results: Vec<MlirResult>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = MlirOperation::new(name)?;
    operation.operands = operands;
    operation.results = results;
    operation.location = location.clone();
    Ok(operation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CutlassFullCuteMlir22, MlirConsumerProfile};
    use dialect_mir::{
        ops::{MirFuncOp, MirReturnOp},
        types::{MirPtrType, address_space},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        identifier::Identifier,
        op::Op,
        r#type::TypeHandle,
        value::Value,
    };
    use pliron_mlir_export::render_module;

    fn append<O: Op>(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        operands: Vec<Value>,
        results: Vec<TypeHandle>,
        marker: Option<&str>,
    ) -> Ptr<Operation> {
        let op = Operation::new(ctx, O::get_concrete_op_info(), results, operands, vec![], 0);
        if let Some(marker) = marker {
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from("cuda_oxide_intrinsic_marker").unwrap(),
                StringAttr::new(marker.into()),
            );
        }
        op.insert_at_back(block, ctx);
        op
    }

    fn fixture() -> (Context, ModuleOp, Ptr<Operation>) {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "tcgen05".try_into().unwrap());
        let i1: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
        let i16: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
        let i32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let i64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get(&mut ctx, i64, true, address_space::SHARED).into();
        let generic: TypeHandle =
            MirPtrType::get(&mut ctx, i64, true, address_space::GENERIC).into();
        // Arguments are deliberately distinct: do not alias D, descriptor A,
        // descriptor B, instruction descriptor, columns, and enable-D.
        let arguments = vec![shared, generic, i32, i64, i64, i32, i32, i1, i16];
        let fty = FunctionType::get(&ctx, arguments.clone(), vec![]);
        let f = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(&mut ctx, f, TypeAttr::new(fty.into()));
        function.set_symbol_name(&mut ctx, "tcgen05_probe".try_into().unwrap());
        module.append_operation(&mut ctx, f, 0);
        let block = BasicBlock::new(&mut ctx, None, arguments);
        block.insert_at_back(f.deref(&ctx).get_region(0), &ctx);
        let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
        append::<CvtaGenericToSharedOffsetOp>(&mut ctx, block, vec![a[1]], vec![i64], None);
        append::<Tcgen05AllocCg2Op>(&mut ctx, block, vec![a[1], a[6]], vec![], Some("v1:i0359"));
        append::<Tcgen05RelinquishAllocPermitCg2Op>(
            &mut ctx,
            block,
            vec![],
            vec![],
            Some("v1:i0361"),
        );
        let mut first = None;
        for collector in [
            Tcgen05MmaCollectorAAttr::Fill,
            Tcgen05MmaCollectorAAttr::LastUse,
        ] {
            let op = append::<Tcgen05MmaOp>(
                &mut ctx,
                block,
                vec![a[2], a[3], a[4], a[5], a[7]],
                vec![],
                Some("v1:i0763"),
            );
            let mma = Tcgen05MmaOp::new(op);
            mma.set_attr_nvvm_tcgen05_mma_form(&ctx, Tcgen05MmaFormAttr::Shared);
            mma.set_attr_nvvm_tcgen05_mma_kind(&ctx, Tcgen05MmaKindAttr::F16);
            mma.set_attr_nvvm_tcgen05_mma_cta_group(&ctx, Tcgen05MmaCtaGroupAttr::Cg2);
            mma.set_attr_nvvm_tcgen05_mma_collector_a(&ctx, collector);
            first.get_or_insert(op);
        }
        append::<Tcgen05CommitMulticastCg2Op>(
            &mut ctx,
            block,
            vec![a[0], a[8]],
            vec![],
            Some("v1:i0365"),
        );
        append::<Tcgen05Ld32x32bX32RawOp>(
            &mut ctx,
            block,
            vec![a[2]],
            vec![i32; 32],
            Some("v1:i0664"),
        );
        append::<Tcgen05LoadWaitOp>(&mut ctx, block, vec![], vec![], Some("v1:i0357"));
        append::<Tcgen05FenceBeforeThreadSyncOp>(&mut ctx, block, vec![], vec![], Some("v1:i0346"));
        append::<Tcgen05FenceAfterThreadSyncOp>(&mut ctx, block, vec![], vec![], Some("v1:i0347"));
        append::<Tcgen05DeallocCg2Op>(&mut ctx, block, vec![a[2], a[6]], vec![], Some("v1:i0360"));
        append::<MirReturnOp>(&mut ctx, block, vec![], vec![], None);
        (ctx, module, first.unwrap())
    }

    fn translate(ctx: &Context, module: &ModuleOp) -> Result<String, String> {
        let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
        let target = profile
            .translate_module(ctx, module)
            .map_err(|error| error.to_string())?;
        Ok(render_module(&target))
    }

    #[test]
    fn split_n_collector_chain_and_tmem_register_order_survive_export() {
        let (ctx, module, _) = fixture();
        let text = translate(&ctx, &module).unwrap();
        assert_eq!(text.matches("\"nvvm.tcgen05.mma\"").count(), 2, "{text}");
        assert!(
            text.contains("#nvvm.tcgen05_mma_collectorop<fill>"),
            "{text}"
        );
        assert!(
            text.contains("#nvvm.tcgen05_mma_collectorop<lastuse>"),
            "{text}"
        );
        assert_eq!(text.matches("\"llvm.inttoptr\"").count(), 4, "{text}");
        assert!(
            text.contains("(!llvm.ptr<6>, i64, i64, i32, i1) -> ()"),
            "{text}"
        );
        assert!(text.contains("-> vector<32xi32>"), "{text}");
        assert_eq!(text.matches("\"vector.extract\"").count(), 32, "{text}");
        for index in 0..32 {
            assert!(
                text.contains(&format!("static_position = array<i64: {index}>")),
                "{text}"
            );
        }
        assert_eq!(text.matches("\"nvvm.tcgen05.wait\"").count(), 1);
        assert!(
            text.find("\"nvvm.tcgen05.ld\"").unwrap() < text.find("\"nvvm.tcgen05.wait\"").unwrap()
        );
        assert!(text.contains("#nvvm.tcgen05_fence<before>"));
        assert!(text.contains("#nvvm.tcgen05_fence<after>"));
        // A generic SMEM pointer must be converted before its integer offset
        // is formed, otherwise descriptors encode a generic virtual address.
        assert!(
            text.find("\"llvm.addrspacecast\"").unwrap() < text.find("\"llvm.ptrtoint\"").unwrap()
        );
        assert!(!text.contains("cuda_oxide_intrinsic_marker"));
    }

    #[test]
    fn unsupported_mma_forms_fail_instead_of_losing_their_selectors() {
        let (ctx, module, first) = fixture();
        let mma = Tcgen05MmaOp::new(first);
        mma.set_attr_nvvm_tcgen05_mma_kind(&ctx, Tcgen05MmaKindAttr::Tf32);
        let error = translate(&ctx, &module).unwrap_err();
        assert!(
            error.contains("shared-A FP16 CTA-group-2 MMA only"),
            "{error}"
        );
        mma.set_attr_nvvm_tcgen05_mma_kind(&ctx, Tcgen05MmaKindAttr::F16);
        mma.set_attr_nvvm_tcgen05_mma_cta_group(&ctx, Tcgen05MmaCtaGroupAttr::Cg1);
        let error = translate(&ctx, &module).unwrap_err();
        assert!(
            error.contains("shared-A FP16 CTA-group-2 MMA only"),
            "{error}"
        );
    }

    #[test]
    fn intrinsic_identity_is_checked_before_export() {
        let (ctx, module, first) = fixture();
        first.deref_mut(&ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new("v1:i0001".into()),
        );
        let error = translate(&ctx, &module).unwrap_err();
        assert!(
            error.contains("expected intrinsic marker v1:i0763"),
            "{error}"
        );
    }
}
