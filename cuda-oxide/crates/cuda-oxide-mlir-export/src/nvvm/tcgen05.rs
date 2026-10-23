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

use dialect_nvvm::ops::*;

use crate::{
    DropAttribute, MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
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
    registry.register_operation::<CvtaGenericToSharedOffsetOp>(SharedOffsetTranslation(false))?;
    registry
        .register_operation::<CvtaGenericToClusterSharedOffsetOp>(SharedOffsetTranslation(true))?;
    /// Register a tensor-memory lifecycle instruction with group, arity, identity, and verifier.
    macro_rules! primitive {
        ($op:ty, $marker:literal, $kind:expr, $group:literal, $operands:literal) => {
            registry.register_operation::<$op>(PrimitiveTranslation {
                marker: $marker,
                kind: $kind,
                group: $group,
                operands: $operands,
                verify: |ctx, source| {
                    <$op>::new(source)
                        .verify(ctx)
                        .map_err(|error| error.to_string())
                },
            })?;
        };
    }
    primitive!(Tcgen05AllocOp, "v1:i0343", Primitive::Alloc, 1, 2);
    primitive!(Tcgen05AllocCg2Op, "v1:i0359", Primitive::Alloc, 2, 2);
    primitive!(Tcgen05DeallocOp, "v1:i0344", Primitive::Dealloc, 1, 2);
    primitive!(Tcgen05DeallocCg2Op, "v1:i0360", Primitive::Dealloc, 2, 2);
    primitive!(Tcgen05CommitOp, "v1:i0348", Primitive::Commit, 1, 1);
    primitive!(Tcgen05CommitCg2Op, "v1:i0363", Primitive::Commit, 2, 1);
    primitive!(
        Tcgen05CommitMulticastOp,
        "v1:i0760",
        Primitive::Commit,
        1,
        2
    );
    primitive!(
        Tcgen05CommitMulticastCg2Op,
        "v1:i0365",
        Primitive::Commit,
        2,
        2
    );
    primitive!(
        Tcgen05RelinquishAllocPermitOp,
        "v1:i0345",
        Primitive::Relinquish,
        1,
        0
    );
    primitive!(
        Tcgen05RelinquishAllocPermitCg2Op,
        "v1:i0361",
        Primitive::Relinquish,
        2,
        0
    );
    primitive!(Tcgen05LoadWaitOp, "v1:i0357", Primitive::LoadWait, 1, 0);
    primitive!(Tcgen05StoreWaitOp, "v1:i0358", Primitive::StoreWait, 1, 0);
    primitive!(
        Tcgen05FenceBeforeThreadSyncOp,
        "v1:i0346",
        Primitive::FenceBefore,
        1,
        0
    );
    primitive!(
        Tcgen05FenceAfterThreadSyncOp,
        "v1:i0347",
        Primitive::FenceAfter,
        1,
        0
    );
    /// Register a tensor-memory transfer with exact scalar register count and packing mode.
    macro_rules! memory {
        ($op:ty, $marker:literal, $width:literal, $store:literal, $pack:literal) => {
            registry.register_operation::<$op>(MemoryTranslation {
                marker: $marker,
                width: $width,
                store: $store,
                pack: $pack,
                verify: |ctx, source| {
                    <$op>::new(source)
                        .verify(ctx)
                        .map_err(|error| error.to_string())
                },
            })?;
        };
    }
    memory!(Tcgen05Ld32x32bX1RawOp, "v1:i0654", 1, false, false);
    memory!(Tcgen05Ld32x32bX1Pack16Op, "v1:i0655", 1, false, true);
    memory!(Tcgen05St32x32bX1RawOp, "v1:i0712", 1, true, false);
    memory!(Tcgen05St32x32bX1Unpack16Op, "v1:i0713", 1, true, true);
    memory!(Tcgen05Ld32x32bX2RawOp, "v1:i0656", 2, false, false);
    memory!(Tcgen05Ld32x32bX2Pack16Op, "v1:i0657", 2, false, true);
    memory!(Tcgen05St32x32bX2RawOp, "v1:i0714", 2, true, false);
    memory!(Tcgen05St32x32bX2Unpack16Op, "v1:i0715", 2, true, true);
    memory!(Tcgen05Ld32x32bX4RawOp, "v1:i0658", 4, false, false);
    memory!(Tcgen05Ld32x32bX4Pack16Op, "v1:i0659", 4, false, true);
    memory!(Tcgen05St32x32bX4RawOp, "v1:i0716", 4, true, false);
    memory!(Tcgen05St32x32bX4Unpack16Op, "v1:i0717", 4, true, true);
    memory!(Tcgen05Ld32x32bX8RawOp, "v1:i0660", 8, false, false);
    memory!(Tcgen05Ld32x32bX8Pack16Op, "v1:i0661", 8, false, true);
    memory!(Tcgen05St32x32bX8RawOp, "v1:i0718", 8, true, false);
    memory!(Tcgen05St32x32bX8Unpack16Op, "v1:i0719", 8, true, true);
    memory!(Tcgen05Ld32x32bX16RawOp, "v1:i0662", 16, false, false);
    memory!(Tcgen05Ld32x32bX16Pack16Op, "v1:i0663", 16, false, true);
    memory!(Tcgen05St32x32bX16RawOp, "v1:i0720", 16, true, false);
    memory!(Tcgen05St32x32bX16Unpack16Op, "v1:i0721", 16, true, true);
    memory!(Tcgen05Ld32x32bX32RawOp, "v1:i0664", 32, false, false);
    memory!(Tcgen05Ld32x32bX32Pack16Op, "v1:i0665", 32, false, true);
    memory!(Tcgen05St32x32bX32RawOp, "v1:i0722", 32, true, false);
    memory!(Tcgen05St32x32bX32Unpack16Op, "v1:i0723", 32, true, true);
    memory!(Tcgen05Ld32x32bX64RawOp, "v1:i0666", 64, false, false);
    memory!(Tcgen05Ld32x32bX64Pack16Op, "v1:i0667", 64, false, true);
    memory!(Tcgen05St32x32bX64RawOp, "v1:i0724", 64, true, false);
    memory!(Tcgen05St32x32bX64Unpack16Op, "v1:i0725", 64, true, true);
    memory!(Tcgen05Ld32x32bX128RawOp, "v1:i0668", 128, false, false);
    memory!(Tcgen05Ld32x32bX128Pack16Op, "v1:i0669", 128, false, true);
    memory!(Tcgen05St32x32bX128RawOp, "v1:i0726", 128, true, false);
    memory!(Tcgen05St32x32bX128Unpack16Op, "v1:i0727", 128, true, true);
    Ok(())
}

/// Preserve reviewed TCGen05 MMA operands, CTA group, shape, and A-collector selectors.
struct MmaTranslation;

/// Convert a pointer to an explicit CTA- or cluster-shared integer offset.
struct SharedOffsetTranslation(
    #[doc = "True selects cluster-shared AS7 offsets; false selects CTA-shared AS3 offsets."] bool,
);

impl OperationTranslation for SharedOffsetTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        (if self.0 {
            CvtaGenericToClusterSharedOffsetOp::new(source).verify(ctx)
        } else {
            CvtaGenericToSharedOffsetOp::new(source).verify(ctx)
        })
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
        let pointer = if self.0 && input.operands[0].ty != MlirType::dialect("!llvm.ptr<7>")? {
            let (result, pointer) = fresh(session, MlirType::dialect("!llvm.ptr<7>")?);
            operations.push(operation(
                "llvm.addrspacecast",
                input.operands.clone(),
                vec![result.clone()],
                &input.location,
            )?);
            pointer
        } else if input.operands[0].ty == MlirType::dialect("!llvm.ptr<7>")? {
            input.operands[0].clone()
        } else {
            shared_pointer(
                session,
                &mut operations,
                input.operands[0].clone(),
                &input.location,
            )?
        };
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
        let tensor_a = match mma.get_attr_nvvm_tcgen05_mma_form(ctx).as_deref() {
            Some(Tcgen05MmaFormAttr::Shared) => false,
            Some(Tcgen05MmaFormAttr::Tensor) => true,
            _ => {
                return Err(
                    "CUTLASS TCGen05 mapping supports dense shared-A or tensor-A FP16 MMA only"
                        .into(),
                );
            }
        };
        if mma.get_attr_nvvm_tcgen05_mma_kind(ctx).as_deref() != Some(&Tcgen05MmaKindAttr::F16) {
            return Err(
                "CUTLASS TCGen05 mapping supports dense shared-A or tensor-A FP16 MMA only".into(),
            );
        }
        let group = match mma.get_attr_nvvm_tcgen05_mma_cta_group(ctx).as_deref() {
            Some(Tcgen05MmaCtaGroupAttr::Cg1) => 1,
            Some(Tcgen05MmaCtaGroupAttr::Cg2) => 2,
            None => return Err("TCGen05 MMA requires a CTA group".into()),
        };
        let collector = match mma.get_attr_nvvm_tcgen05_mma_collector_a(ctx).as_deref() {
            Some(Tcgen05MmaCollectorAAttr::Discard) => "discard",
            Some(Tcgen05MmaCollectorAAttr::LastUse) => "lastuse",
            Some(Tcgen05MmaCollectorAAttr::Fill) => "fill",
            Some(Tcgen05MmaCollectorAAttr::Use) => "use",
            None => return Err("TCGen05 MMA requires a collector-A selector".into()),
        };
        require_input(
            &mut input,
            if tensor_a { "v1:i0764" } else { "v1:i0763" },
            5,
            0,
        )?;
        let (cast, tmem) = tmem_pointer(session, input.operands[0].clone(), &input.location)?;
        input.operands[0] = tmem;
        let mut operations = vec![cast];
        if tensor_a {
            let (cast_a, pointer_a) =
                tmem_pointer(session, input.operands[1].clone(), &input.location)?;
            operations.push(cast_a);
            input.operands[1] = pointer_a;
        }
        let mut target = operation("nvvm.tcgen05.mma", input.operands, vec![], &input.location)?;
        target.properties.insert(
            "mmaKind".into(),
            MlirAttribute::dialect("#nvvm.tcgen05_mma_kind<f16>")?,
        );
        target
            .properties
            .insert("ctaGroup".into(), cta_group_value(group)?);
        target.properties.insert(
            "collectorOp".into(),
            MlirAttribute::dialect(format!("#nvvm.tcgen05_mma_collectorop<{collector}>"))?,
        );
        target.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 1, 1, 1, 0, 0]),
        );
        operations.push(target);
        Ok(operations)
    }
}

/// Select one tensor-memory lifecycle or ordering instruction.
#[derive(Clone, Copy)]
enum Primitive {
    /// Allocate the requested tensor-memory columns with the specified CTA group.
    Alloc,
    /// Release the requested tensor-memory column allocation.
    Dealloc,
    /// Signal completion of preceding tensor-core operations to a barrier.
    Commit,
    /// Relinquish the tensor-memory allocation permit.
    Relinquish,
    /// Wait for previously issued tensor-memory loads.
    LoadWait,
    /// Wait for previously issued tensor-memory stores.
    StoreWait,
    /// Order preceding generic accesses before tensor-core accesses.
    FenceBefore,
    /// Order following generic accesses after tensor-core accesses.
    FenceAfter,
}

/// Source-operation verifier run before discarding its intrinsic metadata.
type VerifyOperation = fn(&Context, Ptr<Operation>) -> Result<(), String>;

/// Validate a reviewed lifecycle instruction and emit its explicit CTA-group contract.
struct PrimitiveTranslation {
    /// Canonical catalog identity required before emitting this intrinsic.
    marker: &'static str,
    /// Tensor-memory lifecycle or ordering action to emit.
    kind: Primitive,
    /// Number of CTAs participating in the native instruction.
    group: u32,
    /// Expected number of source runtime operands.
    operands: usize,
    /// Check the source operation's selectors and operand contract.
    verify: VerifyOperation,
}

impl OperationTranslation for PrimitiveTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        (self.verify)(ctx, source)?;
        require_input(&mut input, self.marker, self.operands, 0)?;
        let mut operations = vec![];
        if matches!(self.kind, Primitive::Dealloc) {
            let (cast, pointer) =
                tmem_pointer(session, input.operands[0].clone(), &input.location)?;
            operations.push(cast);
            input.operands[0] = pointer;
        }
        if matches!(self.kind, Primitive::Alloc | Primitive::Commit) {
            input.operands[0] = shared_pointer(
                session,
                &mut operations,
                input.operands[0].clone(),
                &input.location,
            )?;
        }
        let (name, kind) = match self.kind {
            Primitive::Alloc => ("nvvm.tcgen05.alloc", None),
            Primitive::Dealloc => ("nvvm.tcgen05.dealloc", None),
            Primitive::Commit => ("nvvm.tcgen05.commit", None),
            Primitive::Relinquish => ("nvvm.tcgen05.relinquish_alloc_permit", None),
            Primitive::LoadWait => ("nvvm.tcgen05.wait", Some("#nvvm.tcgen05_wait<load>")),
            Primitive::StoreWait => ("nvvm.tcgen05.wait", Some("#nvvm.tcgen05_wait<store>")),
            Primitive::FenceBefore => ("nvvm.tcgen05.fence", Some("#nvvm.tcgen05_fence<before>")),
            Primitive::FenceAfter => ("nvvm.tcgen05.fence", Some("#nvvm.tcgen05_fence<after>")),
        };
        let mut target = operation(name, input.operands, vec![], &input.location)?;
        if let Some(kind) = kind {
            target
                .properties
                .insert("kind".into(), MlirAttribute::dialect(kind)?);
        } else {
            target
                .properties
                .insert("group".into(), cta_group_value(self.group)?);
        }
        operations.push(target);
        Ok(operations)
    }
}

/// Scalar carriers retain the native register order. The native instructions
/// are asynchronous; only explicit wait operations introduce completion waits.
struct MemoryTranslation {
    /// Canonical catalog identity required before emitting this intrinsic.
    marker: &'static str,
    /// Number of scalar register results or inputs in the memory instruction.
    width: usize,
    /// Select a tensor-memory store rather than a load.
    store: bool,
    /// Select the native packed memory-instruction variant.
    pack: bool,
    /// Check the source operation's selectors and operand contract.
    verify: VerifyOperation,
}

impl OperationTranslation for MemoryTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        (self.verify)(ctx, source)?;
        require_input(
            &mut input,
            self.marker,
            if self.store { 1 + self.width } else { 1 },
            if self.store { 0 } else { self.width },
        )?;
        let (cast, tmem) = tmem_pointer(session, input.operands[0].clone(), &input.location)?;
        let mut operations = vec![cast];
        // The pinned NVVM lowering uses scalar i32 for x1; vector<1xi32>
        // would create invalid extractelement/insertelement LLVM IR.
        if self.width == 1 {
            let mut target = if self.store {
                operation(
                    "nvvm.tcgen05.st",
                    vec![tmem, input.operands[1].clone()],
                    vec![],
                    &input.location,
                )?
            } else {
                operation(
                    "nvvm.tcgen05.ld",
                    vec![tmem],
                    input.results,
                    &input.location,
                )?
            };
            target.properties.insert(
                "shape".into(),
                MlirAttribute::dialect("#nvvm.tcgen05_ldst_shape<shape_32x32b>")?,
            );
            if self.pack {
                target.properties.insert(
                    if self.store { "unpack" } else { "pack" }.into(),
                    MlirAttribute::Unit,
                );
            }
            operations.push(target);
            return Ok(operations);
        }
        let vector_type = MlirType::Vector {
            shape: vec![self.width as u64],
            element: Box::new(MlirType::Integer(32)),
        };
        let mut target;
        if self.store {
            let (undef, mut vector) = fresh(session, vector_type.clone());
            operations.push(operation(
                "llvm.mlir.undef",
                vec![],
                vec![undef],
                &input.location,
            )?);
            for (index, scalar) in input.operands.into_iter().skip(1).enumerate() {
                let (result, next) = fresh(session, vector_type.clone());
                let mut insert = operation(
                    "vector.insert",
                    vec![scalar, vector],
                    vec![result],
                    &input.location,
                )?;
                insert.properties.insert(
                    "static_position".into(),
                    MlirAttribute::DenseI64Array(vec![index as i64]),
                );
                operations.push(insert);
                vector = next;
            }
            target = operation(
                "nvvm.tcgen05.st",
                vec![tmem, vector],
                vec![],
                &input.location,
            )?;
        } else {
            let (result, vector) = fresh(session, vector_type);
            target = operation("nvvm.tcgen05.ld", vec![tmem], vec![result], &input.location)?;
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
        }
        target.properties.insert(
            "shape".into(),
            MlirAttribute::dialect("#nvvm.tcgen05_ldst_shape<shape_32x32b>")?,
        );
        if self.pack {
            target.properties.insert(
                if self.store { "unpack" } else { "pack" }.into(),
                MlirAttribute::Unit,
            );
        }
        if self.store {
            operations.push(target);
        } else {
            operations.insert(1, target);
        }
        Ok(operations)
    }
}

/// Check canonical intrinsic identity, operand/result counts, and absence of leftover source structure.
pub(crate) fn require_input(
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

/// Build the reviewed NVVM CTA-group attribute from the source group count.
fn cta_group_value(group: u32) -> Result<MlirAttribute, String> {
    MlirAttribute::dialect(format!("#nvvm.cta_group<cta_{group}>"))
}

/// Retype the packed u32 TMEM token as an AS6 pointer without scaling its row/column bits.
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

/// Convert the source address to CTA shared space before forming its native offset.
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

/// Allocate one SSA identifier and return matching result and operand handles.
pub(crate) fn fresh(
    session: &mut TranslationSession<'_>,
    ty: MlirType,
) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}

/// Build a target operation with the supplied operands, results, and source location.
pub(crate) fn operation(
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
    //! Selector, identity, and register-order tests for tensor-memory instructions.

    use super::*;
    use crate::render_module;
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

    /// Append a typed source operation to the fixture block and return its operation handle.
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

    /// Build a split-N MMA source function with distinct descriptor, allocation, and register operands.
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

    /// Export the source fixture and render mapping MLIR without its executable envelope.
    fn translate(ctx: &Context, module: &ModuleOp) -> Result<String, String> {
        let profile = CutlassFullCuteMlir22::new("sm_100a").unwrap();
        let target = profile
            .translate_module(ctx, module)
            .map_err(|error| error.to_string())?;
        render_module(&target).map_err(|error| error.to_string())
    }

    /// Check paired MMA retains collector fill/reuse/last-use order and TMEM scalar register order.
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
        assert!(text.contains("-> (vector<32xi32>)"), "{text}");
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

    /// Reject MMA variants whose source selector contracts have no reviewed mapping.
    #[test]
    fn unsupported_mma_forms_fail_instead_of_losing_their_selectors() {
        let (ctx, module, first) = fixture();
        let mma = Tcgen05MmaOp::new(first);
        mma.set_attr_nvvm_tcgen05_mma_kind(&ctx, Tcgen05MmaKindAttr::Tf32);
        let error = translate(&ctx, &module).unwrap_err();
        assert!(
            error.contains("dense shared-A or tensor-A FP16 MMA only"),
            "{error}"
        );
    }

    /// Check single-CTA attention MMA retains its tensor-memory A source.
    #[test]
    fn attention_mma_preserves_group_one_and_tensor_source_a() {
        for tensor in [false, true] {
            let (ctx, module, first) = fixture();
            let mma = Tcgen05MmaOp::new(first);
            mma.set_attr_nvvm_tcgen05_mma_cta_group(&ctx, Tcgen05MmaCtaGroupAttr::Cg1);
            if tensor {
                mma.set_attr_nvvm_tcgen05_mma_form(&ctx, Tcgen05MmaFormAttr::Tensor);
                let address = first.deref(&ctx).get_operand(0);
                Operation::replace_operand(first, &ctx, 1, address);
                first.deref_mut(&ctx).attributes.set(
                    "cuda_oxide_intrinsic_marker".try_into().unwrap(),
                    StringAttr::new("v1:i0764".into()),
                );
            }
            let text = translate(&ctx, &module).unwrap();
            assert!(text.contains("#nvvm.cta_group<cta_1>"), "{text}");
            if tensor {
                assert!(
                    text.contains("(!llvm.ptr<6>, !llvm.ptr<6>, i64, i32, i1) -> ()"),
                    "{text}"
                );
            }
        }
    }

    /// Check attention loads/stores retain register counts, packing modes, and caller-specified waits.
    #[test]
    fn attention_tmem_memory_preserves_width_pack_and_explicit_waits() {
        let (mut ctx, module, first) = fixture();
        let block = first.deref(&ctx).get_parent_block().unwrap();
        let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
        let i32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        /// Append a marked tensor-memory operation to the attention memory test fixture.
        macro_rules! insert {
            ($ty:ty, $args:expr, $results:expr, $marker:literal) => {{
                let op = Operation::new(
                    &mut ctx,
                    <$ty>::get_concrete_op_info(),
                    $results,
                    $args,
                    vec![],
                    0,
                );
                op.deref_mut(&ctx).attributes.set(
                    "cuda_oxide_intrinsic_marker".try_into().unwrap(),
                    StringAttr::new($marker.into()),
                );
                op.insert_before(&ctx, first);
                op
            }};
        }
        insert!(
            dialect_nvvm::ops::Tcgen05AllocOp,
            vec![a[0], a[2]],
            vec![],
            "v1:i0343"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05RelinquishAllocPermitOp,
            vec![],
            vec![],
            "v1:i0345"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05CommitOp,
            vec![a[0]],
            vec![],
            "v1:i0348"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05CommitCg2Op,
            vec![a[0]],
            vec![],
            "v1:i0363"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05CommitMulticastOp,
            vec![a[0], a[8]],
            vec![],
            "v1:i0760"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05DeallocOp,
            vec![a[2], a[2]],
            vec![],
            "v1:i0344"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX1RawOp,
            vec![a[2]],
            vec![i32; 1],
            "v1:i0654"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX1Pack16Op,
            vec![a[2]],
            vec![i32; 1],
            "v1:i0655"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX1RawOp,
            vec![a[2]; 2],
            vec![],
            "v1:i0712"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX1Unpack16Op,
            vec![a[2]; 2],
            vec![],
            "v1:i0713"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX2RawOp,
            vec![a[2]],
            vec![i32; 2],
            "v1:i0656"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX2Pack16Op,
            vec![a[2]],
            vec![i32; 2],
            "v1:i0657"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX2RawOp,
            vec![a[2]; 3],
            vec![],
            "v1:i0714"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX2Unpack16Op,
            vec![a[2]; 3],
            vec![],
            "v1:i0715"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX4RawOp,
            vec![a[2]],
            vec![i32; 4],
            "v1:i0658"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX4Pack16Op,
            vec![a[2]],
            vec![i32; 4],
            "v1:i0659"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX4RawOp,
            vec![a[2]; 5],
            vec![],
            "v1:i0716"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX4Unpack16Op,
            vec![a[2]; 5],
            vec![],
            "v1:i0717"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX8RawOp,
            vec![a[2]],
            vec![i32; 8],
            "v1:i0660"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX8Pack16Op,
            vec![a[2]],
            vec![i32; 8],
            "v1:i0661"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX8RawOp,
            vec![a[2]; 9],
            vec![],
            "v1:i0718"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX8Unpack16Op,
            vec![a[2]; 9],
            vec![],
            "v1:i0719"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX16RawOp,
            vec![a[2]],
            vec![i32; 16],
            "v1:i0662"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX16Pack16Op,
            vec![a[2]],
            vec![i32; 16],
            "v1:i0663"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX16RawOp,
            vec![a[2]; 17],
            vec![],
            "v1:i0720"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX16Unpack16Op,
            vec![a[2]; 17],
            vec![],
            "v1:i0721"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX32RawOp,
            vec![a[2]],
            vec![i32; 32],
            "v1:i0664"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX32Pack16Op,
            vec![a[2]],
            vec![i32; 32],
            "v1:i0665"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX32RawOp,
            vec![a[2]; 33],
            vec![],
            "v1:i0722"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX32Unpack16Op,
            vec![a[2]; 33],
            vec![],
            "v1:i0723"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX64RawOp,
            vec![a[2]],
            vec![i32; 64],
            "v1:i0666"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX64Pack16Op,
            vec![a[2]],
            vec![i32; 64],
            "v1:i0667"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX64RawOp,
            vec![a[2]; 65],
            vec![],
            "v1:i0724"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX64Unpack16Op,
            vec![a[2]; 65],
            vec![],
            "v1:i0725"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX128RawOp,
            vec![a[2]],
            vec![i32; 128],
            "v1:i0668"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05Ld32x32bX128Pack16Op,
            vec![a[2]],
            vec![i32; 128],
            "v1:i0669"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX128RawOp,
            vec![a[2]; 129],
            vec![],
            "v1:i0726"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05St32x32bX128Unpack16Op,
            vec![a[2]; 129],
            vec![],
            "v1:i0727"
        );
        insert!(
            dialect_nvvm::ops::Tcgen05StoreWaitOp,
            vec![],
            vec![],
            "v1:i0358"
        );
        let text = translate(&ctx, &module).unwrap();
        assert_eq!(text.matches("\"nvvm.tcgen05.ld\"").count(), 17);
        assert_eq!(text.matches("\"nvvm.tcgen05.st\"").count(), 16);
        assert_eq!(text.matches("pack = unit").count(), 16);
        assert_eq!(text.matches("\"nvvm.tcgen05.wait\"").count(), 2);
        assert!(text.contains("#nvvm.tcgen05_wait<store>"));
        assert!(text.contains("#nvvm.cta_group<cta_1>"));
        for width in [2, 4, 8, 16, 32, 64, 128] {
            assert!(text.contains(&format!("vector<{width}xi32>")), "{text}");
        }
    }

    /// Reject a source instruction carrying a forged native catalog marker.
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
