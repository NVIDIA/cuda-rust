/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Layout-aware operation builder shared by cooperative copy and matrix loads.

use crate::{
    MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse, OperationInput,
    TranslationSession,
};
use cute_layout::{ComposedLayout, Layout, Swizzle};

/// Require the expected operands and no results or untranslated source structure.
pub(in crate::cute) fn require_input(
    input: &OperationInput,
    operands: usize,
) -> Result<(), String> {
    if input.operands.len() != operands
        || !input.results.is_empty()
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err("CuTe attention operation retained unexpected structure".into());
    }
    Ok(())
}

/// Build a signless integer attribute with an explicit bit width.
pub(in crate::cute) fn integer(value: i128, width: u32) -> MlirAttribute {
    MlirAttribute::Integer {
        value,
        ty: MlirType::Integer(width),
    }
}

/// Accumulate scalar MLIR for verified layout indexing and memory accesses.
pub(in crate::cute) struct Emitter<'a, 'ctx> {
    /// Translation session providing unique SSA and block identifiers.
    pub(in crate::cute) session: &'a mut TranslationSession<'ctx>,
    /// Source location attached to every operation emitted by this builder.
    pub(in crate::cute) location: MlirLocation,
    /// Target operations accumulated in execution order.
    pub(in crate::cute) operations: Vec<MlirOperation>,
}
impl<'a, 'ctx> Emitter<'a, 'ctx> {
    /// Start an empty operation sequence using the source location and shared SSA allocator.
    pub(in crate::cute) fn new(
        session: &'a mut TranslationSession<'ctx>,
        location: MlirLocation,
    ) -> Self {
        Self {
            session,
            location,
            operations: vec![],
        }
    }
    /// Allocate one SSA identifier and return matching result and operand handles.
    pub(in crate::cute) fn fresh(&mut self, ty: MlirType) -> (MlirResult, MlirValueUse) {
        let id = self.session.fresh_value();
        (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
    }
    /// Build a result-free operation at this builder's source location without appending it.
    pub(in crate::cute) fn operation(
        &self,
        name: &str,
        operands: Vec<MlirValueUse>,
    ) -> Result<MlirOperation, String> {
        let mut op = MlirOperation::new(name)?;
        op.location = self.location.clone();
        op.operands = operands;
        Ok(op)
    }
    /// Append an operation with one newly allocated result of the requested type.
    pub(in crate::cute) fn value(
        &mut self,
        name: &str,
        ty: MlirType,
        operands: Vec<MlirValueUse>,
    ) -> Result<MlirValueUse, String> {
        let (result, use_) = self.fresh(ty);
        let mut op = self.operation(name, operands)?;
        op.results.push(result);
        self.operations.push(op);
        Ok(use_)
    }
    /// Append a signed 64-bit integer constant.
    pub(in crate::cute) fn c64(&mut self, value: i64) -> Result<MlirValueUse, String> {
        self.constant(value, 64)
    }
    /// Append an integer constant with the requested width in bits.
    pub(in crate::cute) fn constant(
        &mut self,
        value: i64,
        width: u32,
    ) -> Result<MlirValueUse, String> {
        let (result, use_) = self.fresh(MlirType::Integer(width));
        let mut op = self.operation("arith.constant", vec![])?;
        op.results.push(result);
        op.properties
            .insert("value".into(), integer(i128::from(value), width));
        self.operations.push(op);
        Ok(use_)
    }
    /// Append a binary operation preserving the left operand's type.
    pub(in crate::cute) fn binary(
        &mut self,
        name: &str,
        a: MlirValueUse,
        b: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        self.value(name, a.ty.clone(), vec![a, b])
    }
    /// Append an integer comparison using the MLIR predicate number, returning i1.
    pub(in crate::cute) fn compare(
        &mut self,
        predicate: i128,
        a: MlirValueUse,
        b: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let value = self.value("arith.cmpi", MlirType::Integer(1), vec![a, b])?;
        self.operations
            .last_mut()
            .unwrap()
            .properties
            .insert("predicate".into(), integer(predicate, 64));
        Ok(value)
    }
    /// Append an arith.select with the chosen operands' common type.
    pub(in crate::cute) fn select(
        &mut self,
        predicate: MlirValueUse,
        yes: MlirValueUse,
        no: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        self.value("arith.select", yes.ty.clone(), vec![predicate, yes, no])
    }
    /// Convert an LLVM pointer to the requested address space when necessary.
    pub(in crate::cute) fn pointer(
        &mut self,
        pointer: MlirValueUse,
        space: u32,
    ) -> Result<MlirValueUse, String> {
        let ty = MlirType::dialect(format!("!llvm.ptr<{space}>"))?;
        if pointer.ty == ty {
            Ok(pointer)
        } else {
            self.value("llvm.addrspacecast", ty, vec![pointer])
        }
    }
    /// Append a dynamic element-indexed GEP without no-wrap assumptions.
    pub(in crate::cute) fn gep(
        &mut self,
        base: MlirValueUse,
        index: MlirValueUse,
        element: MlirType,
    ) -> Result<MlirValueUse, String> {
        let value = self.value("llvm.getelementptr", base.ty.clone(), vec![base, index])?;
        let op = self.operations.last_mut().unwrap();
        op.properties
            .insert("elem_type".into(), MlirAttribute::Type(element));
        op.properties.insert(
            "rawConstantIndices".into(),
            MlirAttribute::DenseI32Array(vec![i32::MIN]),
        );
        op.properties.insert("noWrapFlags".into(), integer(0, 32));
        Ok(value)
    }
    /// A flat CuTe coordinate expands in column-major order through its modes.
    pub(in crate::cute) fn index(
        &mut self,
        layout: &Layout,
        mut coordinate: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let sizes = layout.shape.flatten();
        let strides = layout.stride.flatten();
        let MlirType::Integer(width) = coordinate.ty else {
            return Err("CuTe layout coordinate must be an integer".into());
        };
        let mut offset = self.constant(0, width)?;
        for (i, (&size, &stride)) in sizes.iter().zip(&strides).enumerate() {
            let size = self.constant(size, width)?;
            let digit = if i + 1 == sizes.len() {
                coordinate.clone()
            } else {
                self.binary("arith.remui", coordinate.clone(), size.clone())?
            };
            let stride = self.constant(stride, width)?;
            let term = self.binary("arith.muli", digit, stride)?;
            offset = self.binary("arith.addi", offset, term)?;
            if i + 1 != sizes.len() {
                coordinate = self.binary("arith.divui", coordinate, size)?;
            }
        }
        Ok(offset)
    }
    /// Sum row and column offsets from a verified two-mode layout.
    pub(in crate::cute) fn linear_pair(
        &mut self,
        layout: &Layout,
        row: MlirValueUse,
        column: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let modes = layout.modes();
        if modes.len() != 2 {
            return Err("CuTe attention shared layout requires two modes".into());
        }
        let row = self.index(&modes[0], row)?;
        let column = self.index(&modes[1], column)?;
        self.binary("arith.addi", row, column)
    }
    /// Apply the outer row/column layout, the fixed offset, and the inner swizzle in order.
    pub(in crate::cute) fn composed(
        &mut self,
        layout: &ComposedLayout,
        row: MlirValueUse,
        column: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let MlirType::Integer(width) = row.ty else {
            return Err("CuTe layout coordinate must be an integer".into());
        };
        let plain = self.linear_pair(layout.outer(), row, column)?;
        let offset = self.constant(layout.offset(), width)?;
        let plain = self.binary("arith.addi", plain, offset)?;
        self.swizzle(layout.inner(), plain)
    }
    /// Apply the layout's XOR bit permutation without changing the offset's integer width.
    pub(in crate::cute) fn swizzle(
        &mut self,
        swizzle: Swizzle,
        plain: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let MlirType::Integer(width) = plain.ty else {
            return Err("CuTe swizzle offset must be an integer".into());
        };
        if swizzle.num_bits == 0 {
            return Ok(plain);
        }
        let mask = self.constant(swizzle.y_mask(), width)?;
        let bits = self.binary("arith.andi", plain.clone(), mask)?;
        let shift = self.constant(i64::from(swizzle.num_shift.unsigned_abs()), width)?;
        let moved = self.binary(
            if swizzle.num_shift >= 0 {
                "arith.shrui"
            } else {
                "arith.shli"
            },
            bits,
            shift,
        )?;
        self.binary("arith.xori", plain, moved)
    }
}
