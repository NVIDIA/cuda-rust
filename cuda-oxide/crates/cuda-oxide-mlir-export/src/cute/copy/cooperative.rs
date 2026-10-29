/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Realize verified CuTe cooperative-copy and matrix-load plans for attention.
//!
//! The source copy ABI explicitly requires partial-atom source-size zero filling;
//! it is not an ordinary predicated CuTe copy (which can leave shared bytes
//! untouched). NVVM expresses that contract directly. Both operations retain the
//! verified CuTe thread/value assignment and composed shared layout, including
//! element-to-byte swizzle conversion. Commit, wait, and synchronization stay
//! caller-owned, as do the unsafe pointer, capacity, and participation promises.

use crate::cute::copy::emit::{Emitter, integer, require_input};
use crate::{
    MlirAttribute, MlirOperation, MlirType, OperationInput, OperationTranslation,
    TranslationSession,
};
use cute_layout::{IntTuple, validate_cooperative_copy_plan};
use dialect_cute::ops::CuteCopyG2SOp;
use dialect_mir::types::MirFP16Type;
use pliron::{
    builtin::types::IntegerType,
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::TypeHandle,
};

/// Emit cooperative two-byte storage copies with partial-atom zero filling.
pub(in crate::cute) struct CopyTranslation;
impl OperationTranslation for CopyTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let copy = CuteCopyG2SOp::wrap(source);
        copy.verify(ctx).map_err(|e| e.to_string())?;
        let elem: TypeHandle = copy.get_attr_copy_g2s_elem(ctx).unwrap().clone().into();
        let element = if elem.deref(ctx).is::<MirFP16Type>() {
            MlirType::Float(crate::MlirFloatType::F16)
        } else if elem
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|ty| ty.width() == 16)
        {
            // cp.async transports unchanged bytes. Integer carriers preserve
            // BF16 storage without introducing an FP16 numerical conversion.
            MlirType::Integer(16)
        } else {
            return Err("CuTe cooperative copy supports f16 or 16-bit integer storage only".into());
        };
        input.attributes.remove("copy_g2s_elem");
        require_input(&input, 9)?;
        let atom = copy.get_attr_atom_bytes(ctx).unwrap().0;
        let plan = validate_cooperative_copy_plan(
            atom,
            &copy.get_attr_thread_layout(ctx).unwrap().0,
            &copy.get_attr_value_layout(ctx).unwrap().0,
            &copy.get_attr_tile_layout(ctx).unwrap().0,
            &copy.get_attr_smem_layout(ctx).unwrap().0,
            2,
        )
        .map_err(|e| e.to_string())?;
        let mut e = Emitter::new(session, input.location);
        let global = e.pointer(input.operands[0].clone(), 1)?;
        let shared = e.pointer(input.operands[6].clone(), 3)?;
        // The cooperative-copy contract supplies exactly the verified thread
        // IDs. Preserve that range in the scalar address calculation too.
        let thread_count = e.constant(plan.thread_count, 32)?;
        let thread = e.binary("arith.remui", input.operands[8].clone(), thread_count)?;
        let thread = e.value("arith.extui", MlirType::Integer(64), vec![thread])?;
        let tv_modes = plan.tv_layout.modes();
        let thread_cell = e.index(&tv_modes[0], thread)?;
        let tile_rows = e.c64(plan.tiler_shape[0])?;
        let tile_columns = e.c64(plan.tiler_shape[1])?;
        let row_base = e.binary("arith.muli", input.operands[4].clone(), tile_rows.clone())?;
        let column_base = e.binary(
            "arith.muli",
            input.operands[5].clone(),
            tile_columns.clone(),
        )?;
        // Reject overflowing tile origins through the bounds predicate before
        // choosing a source address. Out-of-range coordinates still zero-fill.
        let last_row_tile =
            e.binary("arith.divui", input.operands[1].clone(), tile_rows.clone())?;
        let last_column_tile = e.binary("arith.divui", input.operands[2].clone(), tile_columns)?;
        let row_tile_valid = e.compare(7, input.operands[4].clone(), last_row_tile)?; // ule
        let column_tile_valid = e.compare(7, input.operands[5].clone(), last_column_tile)?;
        let tile_valid = e.binary("arith.andi", row_tile_valid, column_tile_valid)?;
        let zero = e.c64(0)?;
        let two = e.c64(2)?;
        let values_per_atom = i64::from(atom) / 2;
        let atom_values = e.c64(values_per_atom)?;
        for value_index in (0..plan.values_per_thread).step_by(values_per_atom as usize) {
            let value_cell = tv_modes[1]
                .checked_call(&IntTuple::Leaf(value_index))
                .ok_or_else(|| "verified CuTe value assignment is out of range".to_owned())?;
            let value_cell = e.c64(value_cell)?;
            let cell = e.binary("arith.addi", thread_cell.clone(), value_cell)?;
            let row = e.binary("arith.remui", cell.clone(), tile_rows.clone())?;
            let column = e.binary("arith.divui", cell, tile_rows.clone())?;
            let shared_offset = e.composed(&plan.smem_byte_layout, row.clone(), column.clone())?;
            let destination = e.gep(shared.clone(), shared_offset, MlirType::Integer(8))?;
            let row = e.binary("arith.addi", row_base.clone(), row)?;
            let column = e.binary("arith.addi", column_base.clone(), column)?;
            let row_valid = e.compare(6, row.clone(), input.operands[1].clone())?; // ult
            let column_valid = e.compare(6, column.clone(), input.operands[2].clone())?;
            let valid = e.binary("arith.andi", row_valid, column_valid)?;
            let valid = e.binary("arith.andi", valid, tile_valid.clone())?;
            let remaining = e.binary("arith.subi", input.operands[2].clone(), column.clone())?;
            let remaining = e.binary("arith.minui", remaining, atom_values.clone())?;
            let valid_elements = e.select(valid.clone(), remaining, zero.clone())?;
            let valid_bytes = e.binary("arith.muli", valid_elements, two.clone())?;
            let valid_bytes = e.value("arith.trunci", MlirType::Integer(32), vec![valid_bytes])?;
            let row_offset = e.binary("arith.muli", row, input.operands[3].clone())?;
            let source_offset = e.binary("arith.addi", row_offset, column)?;
            let source_offset = e.select(valid, source_offset, zero.clone())?;
            let source = e.gep(global.clone(), source_offset, element.clone())?;
            let mut op = e.operation(
                "nvvm.cp.async.shared.global",
                vec![destination, source, valid_bytes],
            )?;
            op.properties
                .insert("size".into(), integer(i128::from(atom), 32));
            op.properties.insert(
                "modifier".into(),
                MlirAttribute::dialect(if atom == 16 {
                    "#nvvm<load_cache_modifier cg>"
                } else {
                    "#nvvm<load_cache_modifier ca>"
                })?,
            );
            e.operations.push(op);
        }
        Ok(e.operations)
    }
}
