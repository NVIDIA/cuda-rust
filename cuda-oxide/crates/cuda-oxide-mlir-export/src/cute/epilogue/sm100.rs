/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 tensor-memory accumulator load, bias, conversion, and shared stores.

use crate::cute::emit::{Emit, dialect, integer_property};
use crate::{MlirAttribute, MlirFloatType, MlirType, MlirValueUse};
use dialect_cute::sm100_ops::*;

/// Load a TMEM vector, add scalar bias, convert to FP16, and store swizzled shared chunks.
pub(in crate::cute) fn emit_epilogue(
    e: &mut Emit<'_, '_>,
    a: &[MlirValueUse],
    p: &CuteSm100EpiloguePlanAttr,
) -> Result<(), String> {
    let ptr = e.cast(
        "builtin.unrealized_conversion_cast",
        a[0].clone(),
        dialect("!cute.ptr<f32,tmem,align<16>>")?,
    )?;
    let (result, regs) = e.fresh(MlirType::Vector {
        shape: vec![32],
        element: Box::new(MlirType::Integer(32)),
    });
    let mut load = e.operation("cute_nvgpu.arch.copy.SM100.tmem_load", vec![ptr])?;
    load.results.push(result);
    for name in ["num_dp", "num_b", "num_rep"] {
        integer_property(&mut load, name, 32, 32);
    }
    e.ops.push(load);
    // The native CuTe TMEM load is a synchronous copy operation: CUTLASS
    // inserts its tcgen05.wait::ld before making the register vector available.
    // Keep the register tile vector-valued through the bias and conversion.
    // Scalar extraction here prevents packed FP16 conversion and shared stores.
    let floats = MlirType::Vector {
        shape: vec![32],
        element: Box::new(MlirType::Float(MlirFloatType::F32)),
    };
    let values = e.cast("arith.bitcast", regs, floats.clone())?;
    let bias = e.cast("vector.broadcast", a[3].clone(), floats)?;
    let biased = e.binary("arith.addf", values, bias)?;
    let halves = e.cast(
        "arith.truncf",
        biased,
        MlirType::Vector {
            shape: vec![32],
            element: Box::new(MlirType::Float(MlirFloatType::F16)),
        },
    )?;
    let dst = e.llvm_ptr(a[1].clone(), 3)?;
    let stride = e.constant(32, p.columns as i128)?;
    let row = e.binary("arith.muli", a[2].clone(), stride)?;
    let swizzle = p.layout.0.outer();
    let mask = e.constant(32, (((1u32 << swizzle.bits) - 1) << swizzle.base) as i128)?;
    let shift = e.constant(32, swizzle.shift as i128)?;
    // The verified B64 layout has element swizzle S<2,3,3>: it only
    // changes bits 3 and 4 using bits 6 and 7. Within each eight-element
    // chunk the low three bits therefore remain ordered and contiguous.
    // Swizzle each chunk's base once, then store its eight FP16 values.
    for col in (0..p.columns).step_by(8) {
        let (result, chunk) = e.fresh(MlirType::Vector {
            shape: vec![8],
            element: Box::new(MlirType::Float(MlirFloatType::F16)),
        });
        let mut extract = e.operation("vector.shuffle", vec![halves.clone(), halves.clone()])?;
        extract.results.push(result);
        extract.properties.insert(
            "mask".into(),
            MlirAttribute::DenseI64Array((col..col + 8).map(i64::from).collect()),
        );
        e.ops.push(extract);
        let c = e.constant(32, col as i128)?;
        let plain = e.binary("arith.addi", row.clone(), c)?;
        let shifted = e.binary("arith.shrui", plain.clone(), shift.clone())?;
        let swizzled = e.binary("arith.andi", shifted, mask.clone())?;
        let offset = e.binary("arith.xori", plain, swizzled)?;
        let addr = e.gep(dst.clone(), offset, MlirType::Float(MlirFloatType::F16))?;
        let mut store = e.operation("llvm.store", vec![chunk, addr])?;
        // The raw public pointer only promises FP16 alignment. Preserve that
        // contract: stronger alignment can be inferred from an aligned shared
        // allocation and the chunk's multiple-of-eight element offset.
        integer_property(&mut store, "alignment", 64, 2);
        e.ops.push(store);
    }
    Ok(())
}
