/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM120 fragment stores, TMA-store completion, and epilogue synchronization.
//!
//! Each thread's 64 accumulator values are ordered as two M bands, eight N slots,
//! and four registers per slot. The N offsets select two 128x64 shared halves;
//! their bases are 8192 f16 elements apart. This illustrates the register and
//! half selection, before the warp/lane offsets and shared-memory swizzle:
//!
//! ```
//! let n_offsets = [0, 8, 32, 40, 64, 72, 96, 104];
//! let (m_band, n_slot) = (0, 4);
//! let first_register = (m_band * 8 + n_slot) * 4;
//! let half_base = (n_offsets[n_slot] / 64) * 8192;
//! assert_eq!((first_register, half_base), (16, 8192));
//! ```

use crate::cute::{
    common::{
        cast_int, emit_binary, emit_i32_constant, emit_i64_constant, expect_shape, fresh,
        gep_properties, gep_value, integer_constant, operation, stringify,
    },
    mma::sm120::{CarrierLayout, extract_carrier},
};
use crate::{
    MlirAttribute, MlirLocation, MlirOperation, MlirType, MlirValueUse, OperationInput,
    OperationTranslation, TranslationSession,
};
use dialect_cute::{
    attributes::{CuteEpilogueSyncPhaseAttr, CuteTmaStorePipelineAttr},
    epilogue_ops::{
        CuteEpilogueHalfOp, CuteEpilogueStoreFragmentOp, CuteEpilogueSyncOp, CuteTmaStoreAcquireOp,
        CuteTmaStoreCommitOp, CuteTmaStoreTailOp,
    },
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::Typed,
};

/// NVVM operation spelling for a barrier with an explicit participant count.
const COUNTED_CTA_SYNC_OP: &str = "nvvm.barrier.cta.sync";
/// Lower store acquire/tail to a bulk-group wait with the verified pending-group limit.
pub(in crate::cute) struct TmaStoreWaitTranslation {
    /// Source operation selecting queue acquisition or complete queue draining.
    pub(in crate::cute) name: &'static str,
}

impl OperationTranslation for TmaStoreWaitTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let group = match self.name {
            "cute.tma_store_acquire" => {
                let semantic = CuteTmaStoreAcquireOp::wrap(source);
                semantic.verify(ctx).map_err(stringify)?;
                semantic
                    .pipeline(ctx)
                    .and_then(CuteTmaStorePipelineAttr::max_pending)
                    .ok_or_else(|| "tma_store_acquire has no valid pipeline".to_owned())?
            }
            "cute.tma_store_tail" => {
                CuteTmaStoreTailOp::wrap(source)
                    .verify(ctx)
                    .map_err(stringify)?;
                0
            }
            _ => return Err(format!("unknown TMA store wait mapping {}", self.name)),
        };
        expect_shape(self.name, &input, 0, 0)?;
        let mut wait = operation(
            "nvvm.cp.async.bulk.wait_group",
            vec![],
            vec![],
            &input.location,
        )?;
        wait.properties.insert(
            "group".into(),
            MlirAttribute::Integer {
                value: i128::from(group),
                ty: MlirType::Integer(32),
            },
        );
        wait.attributes.insert("read".into(), MlirAttribute::Unit);
        Ok(vec![wait])
    }
}

/// Map a verified result-free, operand-free source operation to its target spelling.
pub(in crate::cute) struct NoOperandRenameTranslation(
    #[doc = "Target operation spelling for the verified zero-operand source operation."]
    pub(in crate::cute) &'static str,
);

impl OperationTranslation for NoOperandRenameTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if self.0 == "nvvm.cp.async.bulk.commit.group" {
            CuteTmaStoreCommitOp::wrap(source)
                .verify(ctx)
                .map_err(stringify)?;
        }
        expect_shape("no-operand rename", &input, 0, 0)?;
        Ok(vec![operation(self.0, vec![], vec![], &input.location)?])
    }
}

/// Select a shared FP16 half and return its pointer plus element count.
pub(in crate::cute) struct EpilogueHalfTranslation;

impl OperationTranslation for EpilogueHalfTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CuteEpilogueHalfOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        input.attributes.remove("epilogue_half_tile");
        expect_shape("cute.epilogue_half", &input, 2, 1)?;
        let tile = semantic
            .tile(ctx)
            .ok_or_else(|| "epilogue_half has no tile".to_owned())?;
        let half = semantic
            .half(ctx)
            .ok_or_else(|| "epilogue_half has no index".to_owned())?;
        let half_elements = tile
            .half_elements()
            .ok_or_else(|| "epilogue half size overflowed".to_owned())?;
        let mut operations = vec![];
        if half.0 == 0 {
            operations.push(operation(
                "builtin.unrealized_conversion_cast",
                vec![input.results[0].clone()],
                vec![input.operands[0].clone()],
                &input.location,
            )?);
        } else {
            let index =
                emit_i64_constant(session, &mut operations, half_elements, &input.location)?;
            let mut gep = operation(
                "llvm.getelementptr",
                vec![input.results[0].clone()],
                vec![input.operands[0].clone(), index],
                &input.location,
            )?;
            gep_properties(&mut gep, epilogue_gep_element_type(), vec![i32::MIN]);
            operations.push(gep);
        }
        operations.push(integer_constant(
            input.results[1].clone(),
            i128::from(half_elements),
            &input.location,
        )?);
        Ok(operations)
    }
}

/// Publish shared stores when required, then synchronize the counted epilogue participants.
pub(in crate::cute) struct EpilogueSyncTranslation;

impl OperationTranslation for EpilogueSyncTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CuteEpilogueSyncOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        input.attributes.remove("epilogue_sync_tile");
        expect_shape("cute.epilogue_sync", &input, 0, 1)?;
        let phase = semantic
            .phase(ctx)
            .ok_or_else(|| "epilogue_sync has no phase".to_owned())?;
        let barrier = semantic
            .barrier(ctx)
            .ok_or_else(|| "epilogue_sync has no barrier".to_owned())?;
        let threads = barrier
            .participant_threads()
            .ok_or_else(|| "epilogue barrier thread count overflowed".to_owned())?;
        let mut operations = vec![];
        let id = emit_i32_constant(
            session,
            &mut operations,
            i128::from(barrier.barrier_id),
            &input.location,
        )?;
        let threads = emit_i32_constant(
            session,
            &mut operations,
            i128::from(threads),
            &input.location,
        )?;
        operations.extend(epilogue_sync_operations(
            phase,
            id,
            threads,
            &input.location,
        )?);
        Ok(operations)
    }
}

/// Pack the verified 64-register accumulator and store its two swizzled shared halves.
pub(in crate::cute) struct EpilogueStoreTranslation;

impl OperationTranslation for EpilogueStoreTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteEpilogueStoreFragmentOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        input.attributes.remove("epilogue_store_fragment_tile");
        expect_shape("cute.epilogue_store_fragment", &input, 0, 4)?;
        let accumulator_layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_operand(3).get_type(ctx),
            session,
            0,
        )?;
        let mut operations = vec![];
        let mut accumulator = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[3].clone(),
            &accumulator_layout,
            &mut accumulator,
            &input.location,
        )?;
        if accumulator.len() != 64 {
            return Err("cute.epilogue_store_fragment needs 64 f32 leaves".into());
        }

        let three = emit_i64_constant(session, &mut operations, 3, &input.location)?;
        let warp_m = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            input.operands[1].clone(),
            three,
            &input.location,
        )?;
        let two = emit_i64_constant(session, &mut operations, 2, &input.location)?;
        let warp_n = emit_binary(
            session,
            &mut operations,
            "arith.shrui",
            input.operands[1].clone(),
            two,
            &input.location,
        )?;
        let lane = cast_int(
            session,
            &mut operations,
            input.operands[2].clone(),
            MlirType::Integer(64),
            "arith.extui",
            &input.location,
        )?;
        let fifteen = emit_i64_constant(session, &mut operations, 15, &input.location)?;
        let lane15 = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            lane.clone(),
            fifteen,
            &input.location,
        )?;
        let warp_row_stride = emit_i64_constant(session, &mut operations, 1024, &input.location)?;
        let warp_rows = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            warp_m,
            warp_row_stride,
            &input.location,
        )?;
        let lane_row_stride = emit_i64_constant(session, &mut operations, 64, &input.location)?;
        let lane_rows = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            lane15,
            lane_row_stride,
            &input.location,
        )?;
        let row_zero = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            warp_rows,
            lane_rows,
            &input.location,
        )?;
        let band_stride = emit_i64_constant(session, &mut operations, 4096, &input.location)?;
        let row_one = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            row_zero.clone(),
            band_stride,
            &input.location,
        )?;
        let columns_per_warp = emit_i64_constant(session, &mut operations, 16, &input.location)?;
        let column_base = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            warp_n,
            columns_per_warp,
            &input.location,
        )?;
        let seven = emit_i64_constant(session, &mut operations, 7, &input.location)?;
        let lane7 = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            lane,
            seven,
            &input.location,
        )?;
        let eight = emit_i64_constant(session, &mut operations, 8, &input.location)?;
        let swizzle = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            lane7,
            eight,
            &input.location,
        )?;
        let half_stride = emit_i64_constant(session, &mut operations, 8192, &input.location)?;

        /// Logical column offsets for the eight accumulator slots within each M band.
        const N_OFFSETS: [u64; 8] = [0, 8, 32, 40, 64, 72, 96, 104];
        for m_band in 0..2_usize {
            for (n_slot, n_offset) in N_OFFSETS.into_iter().enumerate() {
                let local =
                    emit_i64_constant(session, &mut operations, n_offset % 64, &input.location)?;
                let column = emit_binary(
                    session,
                    &mut operations,
                    "arith.addi",
                    column_base.clone(),
                    local,
                    &input.location,
                )?;
                let logical = emit_binary(
                    session,
                    &mut operations,
                    "arith.addi",
                    if m_band == 0 {
                        row_zero.clone()
                    } else {
                        row_one.clone()
                    },
                    column,
                    &input.location,
                )?;
                let mut physical = emit_binary(
                    session,
                    &mut operations,
                    "arith.xori",
                    logical,
                    swizzle.clone(),
                    &input.location,
                )?;
                if n_offset >= 64 {
                    physical = emit_binary(
                        session,
                        &mut operations,
                        "arith.addi",
                        physical,
                        half_stride.clone(),
                        &input.location,
                    )?;
                }
                let pointer = gep_value(
                    session,
                    &mut operations,
                    input.operands[0].clone(),
                    physical,
                    // `physical` is an f16 element offset.  The source path
                    // performs `base.add(physical)` before casting to u8 for
                    // stmatrix, preserving each 16-byte row address.
                    epilogue_gep_element_type(),
                    &input.location,
                )?;
                let cell = (m_band * 8 + n_slot) * 4;
                let top = emit_f16x2_pack(
                    session,
                    &mut operations,
                    accumulator[cell].clone(),
                    accumulator[cell + 1].clone(),
                    &input.location,
                )?;
                let bottom = emit_f16x2_pack(
                    session,
                    &mut operations,
                    accumulator[cell + 2].clone(),
                    accumulator[cell + 3].clone(),
                    &input.location,
                )?;
                let mut store = operation(
                    "nvvm.stmatrix",
                    vec![],
                    vec![pointer, top, bottom],
                    &input.location,
                )?;
                store.properties.insert(
                    "layout".into(),
                    MlirAttribute::dialect("#nvvm.mma_layout<row>")?,
                );
                operations.push(store);
            }
        }
        Ok(operations)
    }
}

/// Convert low/high FP32 lanes to one packed FP16 word using reversed intrinsic operands.
fn emit_f16x2_pack(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    low: MlirValueUse,
    high: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let vector_type = MlirType::Vector {
        shape: vec![2],
        element: Box::new(MlirType::Float(crate::MlirFloatType::F16)),
    };
    let (call_result, converted) = fresh(session, vector_type);
    let mut call = operation(
        "llvm.call_intrinsic",
        vec![call_result],
        // `ff2f16x2` follows PTX source order: its first f32 becomes the
        // high half.  CuTe's `cvt_f16x2_f32(lo, hi)` contract is low-first,
        // so preserve the shared low-first semantic contract by reversing the
        // intrinsic operands.
        f16x2_intrinsic_operands(low, high).into(),
        location,
    )?;
    call.properties.insert(
        "intrin".into(),
        MlirAttribute::String("llvm.nvvm.ff2f16x2.rn".into()),
    );
    call.properties.insert(
        "op_bundle_sizes".into(),
        MlirAttribute::DenseI32Array(vec![]),
    );
    call.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![2, 0]),
    );
    operations.push(call);
    let (bits_result, bits) = fresh(session, MlirType::Integer(32));
    operations.push(operation(
        "llvm.bitcast",
        vec![bits_result],
        vec![converted],
        location,
    )?);
    Ok(bits)
}

/// Put the high lane first to preserve the source API's low-first packed result.
pub(in crate::cute) fn f16x2_intrinsic_operands(
    low: MlirValueUse,
    high: MlirValueUse,
) -> [MlirValueUse; 2] {
    [high, low]
}

/// Build a CTA synchronization using the explicit barrier identifier and thread count.
fn counted_cta_sync(
    barrier_id: MlirValueUse,
    participant_threads: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(
        COUNTED_CTA_SYNC_OP,
        vec![],
        vec![barrier_id, participant_threads],
        location,
    )
}

/// Publish generic CTA-shared stores to the asynchronous proxy used by TMA.
fn async_shared_proxy_fence(location: &MlirLocation) -> Result<MlirOperation, String> {
    let mut fence = operation("nvvm.fence.proxy", vec![], vec![], location)?;
    fence.attributes.insert(
        "kind".into(),
        MlirAttribute::dialect("#nvvm.proxy_kind<async.shared>")?,
    );
    fence.attributes.insert(
        "space".into(),
        MlirAttribute::dialect("#nvvm.shared_space<cta>")?,
    );
    Ok(fence)
}

/// For ReadyForTma emit a proxy fence before counted sync; Reusable needs only sync.
pub(in crate::cute) fn epilogue_sync_operations(
    phase: CuteEpilogueSyncPhaseAttr,
    barrier_id: MlirValueUse,
    participant_threads: MlirValueUse,
    location: &MlirLocation,
) -> Result<Vec<MlirOperation>, String> {
    let mut operations = Vec::with_capacity(match phase {
        CuteEpilogueSyncPhaseAttr::Reusable => 1,
        CuteEpilogueSyncPhaseAttr::ReadyForTma => 2,
    });
    if phase == CuteEpilogueSyncPhaseAttr::ReadyForTma {
        operations.push(async_shared_proxy_fence(location)?);
    }
    operations.push(counted_cta_sync(barrier_id, participant_threads, location)?);
    Ok(operations)
}

/// Return f16 because epilogue pointer offsets count elements rather than bytes.
pub(in crate::cute) fn epilogue_gep_element_type() -> MlirType {
    MlirType::Float(crate::MlirFloatType::F16)
}
