/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Operation builder shared by SM100 recipes and NVVM numeric mappings.

use crate::{
    MlirAttribute, MlirBlock, MlirLocation, MlirOperation, MlirRegion, MlirResult, MlirType,
    MlirValueUse, TranslationSession,
};
use dialect_cute::attributes::CuteComposedLayoutAttr;

/// Build SM100 and numeric MLIR sequences while sharing SSA allocation and source location.
pub(crate) struct Emit<'a, 'b> {
    /// Translation session providing unique SSA and block identifiers.
    pub(crate) session: &'a mut TranslationSession<'b>,
    /// Source location attached to every operation emitted by this builder.
    pub(crate) location: MlirLocation,
    /// Target operations accumulated in execution order.
    pub(crate) ops: Vec<MlirOperation>,
}
impl Emit<'_, '_> {
    /// Build a result-free operation at this builder's location without appending it.
    pub(crate) fn operation(
        &self,
        name: &str,
        operands: Vec<MlirValueUse>,
    ) -> Result<MlirOperation, String> {
        let mut op = MlirOperation::new(name)?;
        op.operands = operands;
        op.location = self.location.clone();
        Ok(op)
    }
    /// Allocate one SSA identifier and return matching result and operand handles.
    pub(crate) fn fresh(&mut self, ty: MlirType) -> (MlirResult, MlirValueUse) {
        let id = self.session.fresh_value();
        (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
    }
    /// Append a target operation with no results.
    pub(in crate::cute) fn emit(
        &mut self,
        name: &str,
        operands: Vec<MlirValueUse>,
    ) -> Result<(), String> {
        let op = self.operation(name, operands)?;
        self.ops.push(op);
        Ok(())
    }
    /// Append an operation with one fresh result of the supplied type.
    pub(crate) fn value(
        &mut self,
        name: &str,
        operands: Vec<MlirValueUse>,
        ty: MlirType,
    ) -> Result<MlirValueUse, String> {
        let (r, v) = self.fresh(ty);
        let mut op = self.operation(name, operands)?;
        op.results.push(r);
        self.ops.push(op);
        Ok(v)
    }
    /// Append a named one-operand conversion to the requested type.
    pub(crate) fn cast(
        &mut self,
        name: &str,
        operand: MlirValueUse,
        ty: MlirType,
    ) -> Result<MlirValueUse, String> {
        self.value(name, vec![operand], ty)
    }
    /// Append a binary operation with the left operand's result type.
    pub(crate) fn binary(
        &mut self,
        name: &str,
        a: MlirValueUse,
        b: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let ty = a.ty.clone();
        self.value(name, vec![a, b], ty)
    }
    /// Append a signless integer constant with the requested bit width.
    pub(crate) fn constant(&mut self, bits: u32, value: i128) -> Result<MlirValueUse, String> {
        let ty = MlirType::Integer(bits);
        let (r, v) = self.fresh(ty.clone());
        let mut op = self.operation("arith.constant", vec![])?;
        op.results.push(r);
        op.properties
            .insert("value".into(), MlirAttribute::Integer { value, ty });
        self.ops.push(op);
        Ok(v)
    }
    /// Convert a pointer to the requested LLVM address space only when needed.
    pub(crate) fn llvm_ptr(
        &mut self,
        ptr: MlirValueUse,
        space: u32,
    ) -> Result<MlirValueUse, String> {
        let ty = llvm_ptr_type(space)?;
        if ptr.ty == ty {
            Ok(ptr)
        } else {
            self.cast("llvm.addrspacecast", ptr, ty)
        }
    }
    /// Bridge an LLVM pointer to a CuTe pointer with byte alignment and optional byte swizzle.
    pub(crate) fn cute_ptr(
        &mut self,
        ptr: MlirValueUse,
        element: &str,
        space: &str,
        align: u32,
        swizzle: Option<String>,
    ) -> Result<MlirValueUse, String> {
        let ptr = self.llvm_ptr(ptr, if space == "smem" { 3 } else { 0 })?;
        let suffix = swizzle.map(|s| format!(",{s}")).unwrap_or_default();
        self.cast(
            "builtin.unrealized_conversion_cast",
            ptr,
            dialect(&format!(
                "!cute.ptr<{element},{space},align<{align}>{suffix}>"
            ))?,
        )
    }
    /// Append a pointer offset measured in elements of the supplied LLVM type.
    pub(in crate::cute) fn gep(
        &mut self,
        ptr: MlirValueUse,
        index: MlirValueUse,
        elem: MlirType,
    ) -> Result<MlirValueUse, String> {
        let (r, v) = self.fresh(ptr.ty.clone());
        let mut op = self.operation("llvm.getelementptr", vec![ptr, index])?;
        op.results.push(r);
        op.properties
            .insert("elem_type".into(), MlirAttribute::Type(elem));
        op.properties.insert(
            "rawConstantIndices".into(),
            MlirAttribute::DenseI32Array(vec![i32::MIN]),
        );
        self.ops.push(op);
        Ok(v)
    }
    /// Return a CTA-shared barrier pointer, optionally offset by eight-byte barrier slots.
    pub(in crate::cute) fn barrier(
        &mut self,
        ptr: MlirValueUse,
        stage: Option<MlirValueUse>,
    ) -> Result<MlirValueUse, String> {
        let ptr = self.llvm_ptr(ptr, 3)?;
        match stage {
            Some(stage) => self.gep(ptr, stage, MlirType::Integer(64)),
            None => Ok(ptr),
        }
    }
    /// Build a K-major FP16 shared descriptor, converting element swizzles to byte swizzles.
    pub(in crate::cute) fn descriptor(
        &mut self,
        ptr: MlirValueUse,
        layout: &CuteComposedLayoutAttr,
        rows: u32,
    ) -> Result<MlirValueUse, String> {
        // Source composed layouts count FP16 elements. CuTe pointers carry
        // BYTE-address swizzles, so the base advances by log2(sizeof(f16)).
        let swizzle = layout.0.outer();
        let ptr = self.cute_ptr(
            ptr,
            "f16",
            "smem",
            128,
            Some(format!(
                "S<{},{},{}>",
                swizzle.bits,
                swizzle.base + 1,
                swizzle.shift
            )),
        )?;
        let (result, desc) = self.fresh(dialect("!cute_nvgpu.smem_desc")?);
        let mut op = self.operation("cute_nvgpu.make_umma_smem_desc", vec![ptr])?;
        op.results.push(result);
        property(
            &mut op,
            "layout",
            &format!("#cute.layout<\"(({rows},16),1):((64,1),0)\">"),
        )?;
        property(&mut op, "major", "#cute_nvgpu.major<k>")?;
        self.ops.push(op);
        self.cast(
            "builtin.unrealized_conversion_cast",
            desc,
            MlirType::Integer(64),
        )
    }
    /// Append the selected before/after TCGen05 ordering fence.
    pub(in crate::cute) fn fence(&mut self, kind: &str) -> Result<(), String> {
        let mut op = self.operation("nvvm.tcgen05.fence", vec![])?;
        property(&mut op, "kind", &format!("#nvvm.tcgen05_fence<{kind}>"))?;
        self.ops.push(op);
        Ok(())
    }
    /// Wrap operations in one fresh argument-free block and region.
    pub(in crate::cute) fn region(&mut self, operations: Vec<MlirOperation>) -> MlirRegion {
        MlirRegion {
            blocks: vec![MlirBlock {
                id: self.session.fresh_block(),
                arguments: vec![],
                operations,
            }],
        }
    }
    /// Retry bounded parity waits until completion while preserving cluster acquire semantics.
    pub(in crate::cute) fn wait_cluster(
        &mut self,
        barrier: MlirValueUse,
        phase: MlirValueUse,
    ) -> Result<(), String> {
        // Preserve cluster acquire semantics and retry on a bounded try-wait;
        // the older high-level NVVM blocking operation only has CTA scope.
        let start = self.ops.len();
        let (r, done) = self.fresh(MlirType::Integer(1));
        let mut wait = self.operation("nvvm.mbarrier.wait.parity", vec![barrier, phase])?;
        wait.results.push(r);
        property(&mut wait, "kind", "#nvvm.mbar_wait<try>")?;
        property(&mut wait, "scope", "#nvvm.mbar_scope<cluster>")?;
        property(&mut wait, "order", "#nvvm.mem_order<acquire>")?;
        self.ops.push(wait);
        let yes = self.constant(1, 1)?;
        let again = self.binary("arith.xori", done, yes)?;
        self.emit("scf.condition", vec![again])?;
        let before = self.ops.split_off(start);
        let yield_op = self.operation("scf.yield", vec![])?;
        let mut loop_op = self.operation("scf.while", vec![])?;
        loop_op.regions = vec![self.region(before), self.region(vec![yield_op])];
        self.ops.push(loop_op);
        Ok(())
    }
}
/// Wrap a checked type token; the target backend parses its dialect syntax and meaning.
pub(in crate::cute) fn dialect(s: &str) -> Result<MlirType, String> {
    MlirType::dialect(s)
}
/// Spell an opaque LLVM pointer in the requested numeric address space.
pub(crate) fn llvm_ptr_type(space: u32) -> Result<MlirType, String> {
    dialect(&if space == 0 {
        "!llvm.ptr".into()
    } else {
        format!("!llvm.ptr<{space}>")
    })
}
/// Attach a checked textual dialect attribute as a target property.
pub(crate) fn property(op: &mut MlirOperation, key: &str, value: &str) -> Result<(), String> {
    op.properties
        .insert(key.into(), MlirAttribute::dialect(value)?);
    Ok(())
}
/// Attach an integer property with the requested bit width.
pub(crate) fn integer_property(op: &mut MlirOperation, key: &str, bits: u32, value: i128) {
    op.properties.insert(
        key.into(),
        MlirAttribute::Integer {
            value,
            ty: MlirType::Integer(bits),
        },
    );
}
/// Record flattened operand segment lengths for the target operation.
pub(crate) fn segments(op: &mut MlirOperation, values: &[i32]) {
    op.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(values.to_vec()),
    );
}
