/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Map tensor views and copies to the official CUTLASS `cute` dialect.
//!
//! Full tiles use CuTe operations; edge tiles use checked scalar accesses:
//!
//! ```text
//! tensor_make -> zipped_divide -> slice -> full-tile copy
//!      |               CUTLASS CuTe views and algorithms
//!      |
//!      +-> is_full / base / scalar tail store
//!              ordinary arith + LLVM, using verified producer provenance
//! ```
//!
//! CUTLASS handles view transformations and bulk copies. Edge stores still
//! need the original logical length and absolute index, so they map to scalar
//! arithmetic and LLVM memory operations.

use dialect_cute::{
    attributes::{
        CuteAlignmentAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr,
        CuteTensorRoleAttr,
    },
    tensor_ops::{
        CuteTensorBaseOp, CuteTensorIsFullOp, CuteTensorLoadIntoOp, CuteTensorMakeOp,
        CuteTensorSliceOp, CuteTensorStoreElementAbsOp, CuteTensorStoreFromOp,
        CuteTensorZippedDivideOp,
    },
    types::CuteTensorViewType,
};
use pliron::{
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_mlir_export::{
    DropAttribute, MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession, TypeTranslation,
};

/// Register the elementwise tensor-view mapping for the pinned CUTLASS profile.
///
/// The caller must register builtin and MIR-core packs as well: this pack
/// intentionally reuses their scalar, pointer, function, and CFG mappings.
pub fn register_cute_elementwise_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_type::<CuteTensorViewType>(TensorViewTypeTranslation)?;
    register_cute_elementwise_mappings(registry)
}

/// Register the elementwise operations and attributes while leaving the
/// shared `cute.tensor_view` type slot to a composed profile.
pub(crate) fn register_cute_elementwise_mappings(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    // These source attributes have already been consumed by the recipes or
    // are encoded in the translated CUTLASS type. They must still be registered
    // so the exporter's missing-mapping preflight can distinguish an explicit
    // drop from an accidental omission.
    registry.register_attribute::<CuteTensorLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<CuteAlignmentAttr>(DropAttribute)?;

    registry.register_operation::<CuteTensorMakeOp>(TensorMakeTranslation)?;
    registry.register_operation::<CuteTensorZippedDivideOp>(TensorZippedDivideTranslation)?;
    registry.register_operation::<CuteTensorSliceOp>(TensorSliceTranslation)?;
    registry.register_operation::<CuteTensorIsFullOp>(TensorIsFullTranslation)?;
    registry.register_operation::<CuteTensorBaseOp>(TensorBaseTranslation)?;
    registry.register_operation::<CuteTensorLoadIntoOp>(TensorLoadIntoTranslation)?;
    registry.register_operation::<CuteTensorStoreFromOp>(TensorStoreFromTranslation)?;
    registry.register_operation::<CuteTensorStoreElementAbsOp>(TensorStoreElementAbsTranslation)?;
    Ok(())
}

pub(crate) struct TensorViewTypeTranslation;

impl TypeTranslation for TensorViewTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let view = source_ref
            .downcast_ref::<CuteTensorViewType>()
            .ok_or_else(|| "expected cute.tensor_view".to_owned())?;
        require_elementwise_view(view)?;

        let element = scalar_spelling(&registry.translate_type(ctx, view.storage)?)?;
        let space = address_space_spelling(view.space)?;
        let layout = layout_spelling(view.layout)?;
        MlirType::dialect(format!(
            "!cute.memref<{element}, {space}, align<{}>, \"{layout}\">",
            view.alignment.0
        ))
    }
}

struct TensorMakeTranslation;

impl OperationTranslation for TensorMakeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_make", &input, 1, 2)?;
        let source_result = source.deref(ctx).get_result(0);
        let view = source_view(ctx, source_result)?;
        require_elementwise_view(&view)?;
        if view.layout != CuteTensorLayoutAttr::Contiguous1D {
            return Err("cute.tensor_make requires a contiguous 1D result".into());
        }

        let element = scalar_spelling(&session.translate_type(view.storage).map_err(stringify)?)?;
        let space = address_space_spelling(view.space)?;
        let pointer_type = cute_pointer_type(&element, space, view.alignment.0)?;
        let shape_type = MlirType::dialect("!cute.shape<\"?{i64}\">")?;
        let layout_type = MlirType::dialect("!cute.layout<\"?{i64}:1\">")?;
        let location = input.location.clone();

        // Rust slice pointers use the generic LLVM address space in their
        // stable two-word ABI. CUTLASS's gmem pointer lowers to NVVM address space
        // 1, so make that transition explicit before crossing the temporary
        // LLVM/CuTe type bridge. CUTLASS can then reconcile the two opposite
        // unrealized casts around the CuTe pointer without trying to use one
        // as an address-space cast.
        let (mut operations, pointer_use) = bridge_llvm_pointer_to_cute(
            session,
            input.operands[0].clone(),
            pointer_type,
            1,
            &location,
        )?;

        let (shape_result, shape_use) = fresh(session, shape_type);
        let shape = operation(
            "cute.make_shape",
            vec![shape_result],
            vec![input.operands[1].clone()],
            &location,
        )?;

        let (layout_result, layout_use) = fresh(session, layout_type);
        let mut layout = operation(
            "cute.make_layout",
            vec![layout_result],
            vec![shape_use],
            &location,
        )?;
        layout.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 0]),
        );

        let view = operation(
            "cute.make_view",
            input.results,
            vec![pointer_use, layout_use],
            &location,
        )?;
        operations.extend([shape, layout, view]);
        Ok(operations)
    }
}

struct TensorZippedDivideTranslation;

impl OperationTranslation for TensorZippedDivideTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_zipped_divide", &input, 1, 1)?;
        let result = source.deref(ctx).get_result(0);
        let view = source_view(ctx, result)?;
        require_elementwise_view(&view)?;
        let CuteTensorLayoutAttr::Zipped1D(tile_size) = view.layout else {
            return Err("cute.tensor_zipped_divide requires a Zipped1D result".into());
        };

        let tiler_type = MlirType::dialect(format!("!cute.shape<\"{tile_size}\">"))?;
        let (tiler_result, tiler_use) = fresh(session, tiler_type);
        let tiler = operation("cute.static", vec![tiler_result], vec![], &input.location)?;
        let divide = operation(
            "cute.zipped_divide",
            input.results,
            vec![input.operands[0].clone(), tiler_use],
            &input.location,
        )?;
        Ok(vec![tiler, divide])
    }
}

struct TensorSliceTranslation;

impl OperationTranslation for TensorSliceTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_slice", &input, 1, 2)?;
        let result = source.deref(ctx).get_result(0);
        let view = source_view(ctx, result)?;
        require_elementwise_view(&view)?;
        if !matches!(view.layout, CuteTensorLayoutAttr::Tile1D(_)) {
            return Err("cute.tensor_slice requires a Tile1D result".into());
        }

        let coord_type = MlirType::dialect("!cute.coord<\"(_,?{i64})\">")?;
        let (coord_result, coord_use) = fresh(session, coord_type);
        let coord = operation(
            "cute.make_coord",
            vec![coord_result],
            vec![input.operands[1].clone()],
            &input.location,
        )?;
        let slice = operation(
            "cute.slice",
            input.results,
            vec![input.operands[0].clone(), coord_use],
            &input.location,
        )?;
        Ok(vec![coord, slice])
    }
}

struct TensorBaseTranslation;

impl OperationTranslation for TensorBaseTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_base", &input, 1, 1)?;
        let tensor = source.deref(ctx).get_operand(0);
        let state = tensor_tile_state(ctx, tensor)?;
        let final_result = input.results[0].clone();
        let (operations, _) =
            emit_saturating_tile_base(session, &state, final_result, &input.location)?;
        Ok(operations)
    }
}

struct TensorIsFullTranslation;

impl OperationTranslation for TensorIsFullTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_is_full", &input, 1, 1)?;
        let tensor = source.deref(ctx).get_operand(0);
        let state = tensor_tile_state(ctx, tensor)?;
        let location = input.location.clone();
        let i64_type = MlirType::Integer(64);
        let i1_type = MlirType::Integer(1);

        let (base_result, base_use) = fresh(session, i64_type.clone());
        let (mut operations, _) =
            emit_saturating_tile_base(session, &state, base_result, &location)?;

        let (width_result, width_use) = fresh(session, i64_type.clone());
        operations.push(integer_constant(
            width_result,
            u64_bits(state.tile_size),
            &location,
        )?);

        let len = session.target_value_use(state.len).map_err(stringify)?;
        let (in_range_result, in_range_use) = fresh(session, i1_type.clone());
        operations.push(integer_compare(
            in_range_result,
            base_use.clone(),
            len.clone(),
            6,
            &location,
        )?);

        let (remaining_result, remaining_use) = fresh(session, i64_type);
        operations.push(binary(
            "arith.subi",
            remaining_result,
            len,
            base_use,
            &location,
        )?);

        let (enough_result, enough_use) = fresh(session, i1_type);
        operations.push(integer_compare(
            enough_result,
            remaining_use,
            width_use,
            9,
            &location,
        )?);

        operations.push(binary(
            "arith.andi",
            input.results[0].clone(),
            in_range_use,
            enough_use,
            &location,
        )?);
        Ok(operations)
    }
}

struct TensorLoadIntoTranslation;

impl OperationTranslation for TensorLoadIntoTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_load_into", &input, 0, 2)?;
        let source_op = CuteTensorLoadIntoOp::wrap(source);
        let alignment = source_op
            .get_attr_load_alignment_bytes(ctx)
            .map(|attribute| attribute.0)
            .ok_or_else(|| "cute.tensor_load_into has no alignment promise".to_owned())?;
        let tensor_value = source.deref(ctx).get_operand(0);
        emit_full_tile_copy(
            ctx,
            session,
            input.operands[0].clone(),
            input.operands[1].clone(),
            tensor_value,
            alignment,
            CopyDirection::Load,
            &input.location,
        )
    }
}

struct TensorStoreFromTranslation;

impl OperationTranslation for TensorStoreFromTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_store_from", &input, 0, 2)?;
        let source_op = CuteTensorStoreFromOp::wrap(source);
        let alignment = source_op
            .get_attr_store_alignment_bytes(ctx)
            .map(|attribute| attribute.0)
            .ok_or_else(|| "cute.tensor_store_from has no alignment promise".to_owned())?;
        let tensor_value = source.deref(ctx).get_operand(1);
        emit_full_tile_copy(
            ctx,
            session,
            input.operands[1].clone(),
            input.operands[0].clone(),
            tensor_value,
            alignment,
            CopyDirection::Store,
            &input.location,
        )
    }
}

struct TensorStoreElementAbsTranslation;

impl OperationTranslation for TensorStoreElementAbsTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape("cute.tensor_store_element_abs", &input, 0, 3)?;
        let source_ref = source.deref(ctx);
        let tensor = source_ref.get_operand(0);
        let absolute_index = source_ref.get_operand(1);
        let value = source_ref.get_operand(2);
        drop(source_ref);

        let state = tensor_tile_state(ctx, tensor)?;
        let data = session.target_value_use(state.data).map_err(stringify)?;
        let index = session
            .target_value_use(absolute_index)
            .map_err(stringify)?;
        let scalar = session.target_value_use(value).map_err(stringify)?;
        let element_type = scalar.ty.clone();
        let location = input.location.clone();

        let (pointer_result, pointer_use) = fresh(session, data.ty.clone());
        let mut gep = operation(
            "llvm.getelementptr",
            vec![pointer_result],
            vec![data, index],
            &location,
        )?;
        gep.properties
            .insert("elem_type".into(), MlirAttribute::Type(element_type));
        gep.properties.insert(
            "noWrapFlags".into(),
            MlirAttribute::Integer {
                value: 0,
                ty: MlirType::Integer(32),
            },
        );
        gep.properties.insert(
            "rawConstantIndices".into(),
            MlirAttribute::DenseI32Array(vec![i32::MIN]),
        );

        let mut store = operation("llvm.store", vec![], vec![scalar, pointer_use], &location)?;
        store.properties.insert(
            "ordering".into(),
            MlirAttribute::Integer {
                value: 0,
                ty: MlirType::Integer(64),
            },
        );
        Ok(vec![gep, store])
    }
}

#[derive(Clone, Copy)]
struct TensorViewState {
    data: Value,
    len: Value,
    tile_size: u64,
    tile_index: Value,
}

fn resolve_tensor_view(
    ctx: &Context,
    value: Value,
    depth: usize,
) -> Result<(Value, Value, Option<u64>, Option<Value>), String> {
    if depth > 16 {
        return Err("cute tensor-view producer chain is unexpectedly deep".into());
    }
    let defining_op = value.defining_op().ok_or_else(|| {
        "cute tensor view reached a block argument; the elementwise mapping needs a direct make/zipped/slice chain".to_owned()
    })?;
    let opid = Operation::get_opid(defining_op, ctx);
    if opid == CuteTensorMakeOp::get_opid_static() {
        let op = defining_op.deref(ctx);
        return Ok((op.get_operand(0), op.get_operand(1), None, None));
    }
    if opid == CuteTensorZippedDivideOp::get_opid_static() {
        let operand = defining_op.deref(ctx).get_operand(0);
        let (data, len, _, tile_index) = resolve_tensor_view(ctx, operand, depth + 1)?;
        let view = source_view(ctx, value)?;
        let tile_size = view.layout.tile_size().ok_or_else(|| {
            "cute.tensor_zipped_divide result has no static tile width".to_owned()
        })?;
        return Ok((data, len, Some(tile_size), tile_index));
    }
    if opid == CuteTensorSliceOp::get_opid_static() {
        let op = defining_op.deref(ctx);
        let operand = op.get_operand(0);
        let selected_index = op.get_operand(1);
        drop(op);
        let (data, len, _, _) = resolve_tensor_view(ctx, operand, depth + 1)?;
        let view = source_view(ctx, value)?;
        let tile_size = view
            .selected_tile_size()
            .ok_or_else(|| "cute.tensor_slice result has no selected tile width".to_owned())?;
        return Ok((data, len, Some(tile_size), Some(selected_index)));
    }
    Err(format!(
        "tensor view is produced by unsupported operation `{opid}`"
    ))
}

fn tensor_tile_state(ctx: &Context, value: Value) -> Result<TensorViewState, String> {
    let (data, len, tile_size, tile_index) = resolve_tensor_view(ctx, value, 0)?;
    Ok(TensorViewState {
        data,
        len,
        tile_size: tile_size
            .ok_or_else(|| "tensor operation needs a selected tile width".to_owned())?,
        tile_index: tile_index
            .ok_or_else(|| "tensor operation needs a selected tile index".to_owned())?,
    })
}

fn emit_saturating_tile_base(
    session: &mut TranslationSession<'_>,
    state: &TensorViewState,
    final_result: MlirResult,
    location: &MlirLocation,
) -> Result<(Vec<MlirOperation>, MlirValueUse), String> {
    let i64_type = MlirType::Integer(64);
    let i1_type = MlirType::Integer(1);
    let tile_index = session
        .target_value_use(state.tile_index)
        .map_err(stringify)?;
    let mut operations = vec![];

    let (width_result, width_use) = fresh(session, i64_type.clone());
    operations.push(integer_constant(
        width_result,
        u64_bits(state.tile_size),
        location,
    )?);

    let (product_result, product_use) = fresh(session, i64_type.clone());
    operations.push(binary(
        "arith.muli",
        product_result,
        tile_index.clone(),
        width_use,
        location,
    )?);

    let (limit_result, limit_use) = fresh(session, i64_type.clone());
    operations.push(integer_constant(
        limit_result,
        u64_bits(u64::MAX / state.tile_size),
        location,
    )?);

    let (overflow_result, overflow_use) = fresh(session, i1_type);
    operations.push(integer_compare(
        overflow_result,
        limit_use,
        tile_index,
        6,
        location,
    )?);

    let (extended_result, extended_use) = fresh(session, i64_type.clone());
    operations.push(unary(
        "arith.extui",
        extended_result,
        overflow_use,
        location,
    )?);

    let (zero_result, zero_use) = fresh(session, i64_type);
    operations.push(integer_constant(zero_result, 0, location)?);

    let (mask_result, mask_use) = fresh(session, MlirType::Integer(64));
    operations.push(binary(
        "arith.subi",
        mask_result,
        zero_use,
        extended_use,
        location,
    )?);

    let final_use = MlirValueUse {
        id: final_result.id,
        ty: final_result.ty.clone(),
    };
    operations.push(binary(
        "arith.ori",
        final_result,
        product_use,
        mask_use,
        location,
    )?);
    Ok((operations, final_use))
}

#[derive(Clone, Copy)]
enum CopyDirection {
    Load,
    Store,
}

#[allow(clippy::too_many_arguments)]
fn emit_full_tile_copy(
    ctx: &Context,
    session: &mut TranslationSession<'_>,
    tensor: MlirValueUse,
    carrier: MlirValueUse,
    source_tensor: Value,
    promised_alignment: u64,
    direction: CopyDirection,
    location: &MlirLocation,
) -> Result<Vec<MlirOperation>, String> {
    let view = source_view(ctx, source_tensor)?;
    require_elementwise_view(&view)?;
    let tile_size = view
        .selected_tile_size()
        .ok_or_else(|| "full-tile copy requires a selected Tile1D view".to_owned())?;
    let storage_bytes = view
        .storage_bytes(ctx)
        .ok_or_else(|| "full-tile copy storage has no whole-byte width".to_owned())?;
    let transfer_bytes = tile_size
        .checked_mul(storage_bytes)
        .ok_or_else(|| "full-tile copy byte width overflows u64".to_owned())?;
    if !matches!(transfer_bytes, 4 | 8 | 16) {
        return Err(format!(
            "CUTLASS universal_copy mapping supports 4, 8, or 16 bytes, got {transfer_bytes}"
        ));
    }
    if promised_alignment < transfer_bytes || !promised_alignment.is_power_of_two() {
        return Err(format!(
            "full-tile copy alignment {promised_alignment} does not cover {transfer_bytes} bytes"
        ));
    }

    let element = scalar_spelling(&session.translate_type(view.storage).map_err(stringify)?)?;
    let space = address_space_spelling(view.space)?;
    let natural_pointer = cute_pointer_type(&element, space, view.alignment.0)?;
    let aligned_pointer = cute_pointer_type(&element, space, promised_alignment)?;
    let aligned_tensor = cute_memref_type(
        &element,
        space,
        promised_alignment,
        &format!("({tile_size}):(1)"),
    )?;
    let carrier_pointer = cute_pointer_type(&element, "rmem", transfer_bytes)?;
    let carrier_view = cute_memref_type(
        &element,
        "rmem",
        transfer_bytes,
        &format!("({tile_size}):(1)"),
    )?;
    let constrained = MlirType::dialect(format!("!cute.i64<divby {promised_alignment}>"))?;
    let atom_type = MlirType::dialect(format!(
        "!cute_nvgpu.atom.universal_copy<{element}, {}b>",
        transfer_bytes * 8
    ))?;

    let mut operations = vec![];
    let (iter_result, iter_use) = fresh(session, natural_pointer);
    operations.push(operation(
        "cute.get_iter",
        vec![iter_result],
        vec![tensor],
        location,
    )?);

    let (address_result, address_use) = fresh(session, MlirType::Integer(64));
    operations.push(operation(
        "cute.ptrtoint",
        vec![address_result],
        vec![iter_use],
        location,
    )?);

    let (assumed_result, assumed_use) = fresh(session, constrained);
    operations.push(operation(
        "cute.assume",
        vec![assumed_result],
        vec![address_use],
        location,
    )?);

    let (aligned_pointer_result, aligned_pointer_use) = fresh(session, aligned_pointer);
    operations.push(operation(
        "cute.inttoptr",
        vec![aligned_pointer_result],
        vec![assumed_use],
        location,
    )?);

    let (aligned_view_result, aligned_view_use) = fresh(session, aligned_tensor);
    operations.push(operation(
        "cute.make_view",
        vec![aligned_view_result],
        vec![aligned_pointer_use],
        location,
    )?);

    // CuTe rmem pointers deliberately lower to generic LLVM pointers because
    // NVVM allocas cannot use the local-memory address space. Synthetic and
    // hand-authored MIR may still present an explicit local pointer, so
    // normalize it to AS0 before the temporary CuTe bridge as well.
    let (carrier_bridge, carrier_pointer_use) =
        bridge_llvm_pointer_to_cute(session, carrier, carrier_pointer, 0, location)?;
    operations.extend(carrier_bridge);

    let (carrier_view_result, carrier_view_use) = fresh(session, carrier_view);
    operations.push(operation(
        "cute.make_view",
        vec![carrier_view_result],
        vec![carrier_pointer_use],
        location,
    )?);

    let (atom_result, atom_use) = fresh(session, atom_type);
    operations.push(operation(
        "cute.make_atom",
        vec![atom_result],
        vec![],
        location,
    )?);

    let (source, destination) = match direction {
        CopyDirection::Load => (aligned_view_use, carrier_view_use),
        CopyDirection::Store => (carrier_view_use, aligned_view_use),
    };
    let mut copy = operation(
        "cute.copy",
        vec![],
        vec![atom_use, source, destination],
        location,
    )?;
    // CUTLASS 4.7's AlgorithmCopyOp uses AttrSizedOperandSegments.  Generic
    // syntax does not infer these four groups, and a missing property reparses
    // as all-zero segments.  Canonicalization may then erase the copy as an
    // effect-free operation, so preserve the exact atom/src/dst/predicate
    // segmentation explicitly.
    copy.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![1, 1, 1, 0]),
    );
    operations.push(copy);
    Ok(operations)
}

fn require_elementwise_view(view: &CuteTensorViewType) -> Result<(), String> {
    if view.format != CuteTensorFormatAttr::Plain {
        return Err(format!(
            "the elementwise CUTLASS pack supports Plain tensor views, got {:?}",
            view.format
        ));
    }
    if view.role != CuteTensorRoleAttr::Generic {
        return Err(format!(
            "the elementwise CUTLASS pack supports Generic tensor roles, got {:?}",
            view.role
        ));
    }
    if view.space != CuteTensorAddressSpaceAttr::Gmem {
        return Err(format!(
            "the elementwise CUTLASS pack supports global-memory tensor views, got {:?}",
            view.space
        ));
    }
    if !matches!(
        view.layout,
        CuteTensorLayoutAttr::Contiguous1D
            | CuteTensorLayoutAttr::Zipped1D(_)
            | CuteTensorLayoutAttr::Tile1D(_)
    ) {
        return Err(format!(
            "the elementwise CUTLASS pack does not map layout {:?}",
            view.layout
        ));
    }
    Ok(())
}

fn source_view(ctx: &Context, value: Value) -> Result<CuteTensorViewType, String> {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<CuteTensorViewType>()
        .cloned()
        .ok_or_else(|| "expected a cute.tensor_view value".to_owned())
}

fn address_space_spelling(space: CuteTensorAddressSpaceAttr) -> Result<&'static str, String> {
    match space {
        CuteTensorAddressSpaceAttr::Gmem => Ok("gmem"),
        other => Err(format!(
            "the elementwise CUTLASS pack does not map address space {other:?}"
        )),
    }
}

fn layout_spelling(layout: CuteTensorLayoutAttr) -> Result<String, String> {
    match layout {
        CuteTensorLayoutAttr::Contiguous1D => Ok("?{i64}:1".into()),
        CuteTensorLayoutAttr::Zipped1D(tile) if tile > 0 => {
            Ok(format!("({tile},?{{i64}}):(1,{tile})"))
        }
        CuteTensorLayoutAttr::Tile1D(tile) if tile > 0 => Ok(format!("({tile}):(1)")),
        other => Err(format!(
            "the elementwise CUTLASS pack does not map layout {other:?}"
        )),
    }
}

fn scalar_spelling(ty: &MlirType) -> Result<String, String> {
    match ty {
        MlirType::Integer(width) => Ok(format!("i{width}")),
        MlirType::Float(pliron_mlir_export::MlirFloatType::F16) => Ok("f16".into()),
        MlirType::Float(pliron_mlir_export::MlirFloatType::F32) => Ok("f32".into()),
        other => Err(format!(
            "the elementwise CUTLASS pack needs an i8, i16, f16, or f32 scalar, got {other:?}"
        )),
    }
}

fn cute_pointer_type(element: &str, space: &str, alignment: u64) -> Result<MlirType, String> {
    MlirType::dialect(format!("!cute.ptr<{element}, {space}, align<{alignment}>>"))
}

/// Cross from an LLVM pointer to a CuTe pointer without asking an unrealized
/// cast to change address spaces.
///
/// CUTLASS lowers the CuTe pointer back to its canonical LLVM address space. When
/// the incoming pointer already has that address space, the two temporary
/// unrealized casts reconcile away. When it does not, the explicit
/// `llvm.addrspacecast` remains and the temporary casts still reconcile away.
fn bridge_llvm_pointer_to_cute(
    session: &mut TranslationSession<'_>,
    pointer: MlirValueUse,
    cute_pointer: MlirType,
    target_address_space: u32,
    location: &MlirLocation,
) -> Result<(Vec<MlirOperation>, MlirValueUse), String> {
    let source_address_space = llvm_pointer_address_space(&pointer.ty)?;
    let mut operations = vec![];
    let pointer = if source_address_space == target_address_space {
        pointer
    } else {
        let target_type = llvm_pointer_type(target_address_space)?;
        let (result, use_) = fresh(session, target_type);
        operations.push(operation(
            "llvm.addrspacecast",
            vec![result],
            vec![pointer],
            location,
        )?);
        use_
    };

    let (result, use_) = fresh(session, cute_pointer);
    operations.push(operation(
        "builtin.unrealized_conversion_cast",
        vec![result],
        vec![pointer],
        location,
    )?);
    Ok((operations, use_))
}

fn llvm_pointer_address_space(ty: &MlirType) -> Result<u32, String> {
    let MlirType::Dialect(spelling) = ty else {
        return Err(format!(
            "CuTe pointer bridge expected an LLVM pointer, got {ty:?}"
        ));
    };
    if spelling == "!llvm.ptr" {
        return Ok(0);
    }
    let Some(address_space) = spelling
        .strip_prefix("!llvm.ptr<")
        .and_then(|suffix| suffix.strip_suffix('>'))
    else {
        return Err(format!(
            "CuTe pointer bridge expected an LLVM pointer, got {spelling}"
        ));
    };
    address_space.parse::<u32>().map_err(|_| {
        format!("CuTe pointer bridge could not parse LLVM address space in {spelling}")
    })
}

fn llvm_pointer_type(address_space: u32) -> Result<MlirType, String> {
    if address_space == 0 {
        MlirType::dialect("!llvm.ptr")
    } else {
        MlirType::dialect(format!("!llvm.ptr<{address_space}>"))
    }
}

fn cute_memref_type(
    element: &str,
    space: &str,
    alignment: u64,
    layout: &str,
) -> Result<MlirType, String> {
    MlirType::dialect(format!(
        "!cute.memref<{element}, {space}, align<{alignment}>, \"{layout}\">"
    ))
}

fn expect_shape(
    name: &str,
    input: &OperationInput,
    results: usize,
    operands: usize,
) -> Result<(), String> {
    if input.results.len() != results || input.operands.len() != operands {
        return Err(format!(
            "{name} expected {results} results and {operands} operands, got {} and {}",
            input.results.len(),
            input.operands.len()
        ));
    }
    if !input.successors.is_empty() || !input.regions.is_empty() || !input.attributes.is_empty() {
        return Err(format!(
            "{name} unexpectedly retained successors, regions, or source-only attributes"
        ));
    }
    Ok(())
}

fn operation(
    name: &str,
    results: Vec<MlirResult>,
    operands: Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = MlirOperation::new(name)?;
    operation.results = results;
    operation.operands = operands;
    operation.location = location.clone();
    Ok(operation)
}

fn fresh(session: &mut TranslationSession<'_>, ty: MlirType) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}

fn integer_constant(
    result: MlirResult,
    value: i128,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = operation("arith.constant", vec![result.clone()], vec![], location)?;
    operation.properties.insert(
        "value".into(),
        MlirAttribute::Integer {
            value,
            ty: result.ty,
        },
    );
    Ok(operation)
}

fn integer_compare(
    result: MlirResult,
    left: MlirValueUse,
    right: MlirValueUse,
    predicate: i128,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = operation("arith.cmpi", vec![result], vec![left, right], location)?;
    operation.properties.insert(
        "predicate".into(),
        MlirAttribute::Integer {
            value: predicate,
            ty: MlirType::Integer(64),
        },
    );
    Ok(operation)
}

fn binary(
    name: &str,
    result: MlirResult,
    left: MlirValueUse,
    right: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(name, vec![result], vec![left, right], location)
}

fn unary(
    name: &str,
    result: MlirResult,
    operand: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(name, vec![result], vec![operand], location)
}

fn u64_bits(value: u64) -> i128 {
    i64::from_ne_bytes(value.to_ne_bytes()) as i128
}

fn stringify(error: impl std::fmt::Display) -> String {
    error.to_string()
}
